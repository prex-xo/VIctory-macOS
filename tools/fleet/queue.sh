#!/bin/bash
# The GitHub Actions side of the Mac test fleet (see docs/fleet.md).
# Leases tasks and findings through issue comments, checks results, files
# failure signatures and slowdowns as deduplicated findings, opens automatic
# test rounds, screens free text in fleet issues and expires stale leases.
# Run by the fleet-* workflows with GH_TOKEN set. Never evaluates comment
# text: comments are only matched against fixed patterns and the schema.
#
# Usage: queue.sh event    (reads GITHUB_EVENT_NAME and GITHUB_EVENT_PATH)
#        queue.sh expire
#        queue.sh autotask
# shellcheck disable=SC2016,SC2001 # literal backticks, jq and GraphQL in single quotes
set -euo pipefail
# A step that fails under `set -e` says where, instead of a bare exit 1
# (25 November: claims were leased but never answered, with no output).
set -E
trap 'echo "queue.sh: line $LINENO failed (status $?): $BASH_COMMAND" >&2' ERR
# shellcheck source=tools/fleet/lib.sh
. "$(dirname "$0")/lib.sh"

REPO="${GITHUB_REPOSITORY:?}"
LEGAL_URL="https://github.com/$REPO/blob/main/docs/fleet-legal.md"
FLEET_URL="https://github.com/$REPO/blob/main/docs/fleet.md"

api() { gh api -H "Accept: application/vnd.github+json" "$@"; }
reply() { api "repos/$REPO/issues/$1/comments" -f body="$2" > /dev/null; }
react() { api "repos/$REPO/issues/comments/$1/reactions" -f content="$2" > /dev/null; }
add_label() { api "repos/$REPO/issues/$1/labels" -f "labels[]=$2" > /dev/null; }
remove_label() { api -X DELETE "repos/$REPO/issues/$1/labels/$2" > /dev/null 2>&1 || true; }
unassign() { api -X DELETE "repos/$REPO/issues/$1/assignees" -f "assignees[]=$2" > /dev/null; }
delete_comment() { api -X DELETE "repos/$REPO/issues/comments/$1" > /dev/null; }
close_issue() { api -X PATCH "repos/$REPO/issues/$1" -f state=closed -f state_reason="$2" > /dev/null; }
hide_comment() {
    api graphql -f id="$1" -f query='
        mutation($id: ID!) { minimizeComment(input: {subjectId: $id, classifier: OFF_TOPIC}) { clientMutationId } }' > /dev/null
}
issue_comments() { api --paginate "repos/$REPO/issues/$1/comments?per_page=100" | jq -s 'add // []'; }

reopen_lease() { # ISSUE LOGIN
    [ -n "$2" ] && unassign "$1" "$2"
    remove_label "$1" "$(label leased)"
    add_label "$1" "$(label open)"
}

# ISSUE_JSON -> "task COMMIT CHIP", "finding SIG TEST TOPIC" or nothing.
issue_kind() {
    local author body parsed
    author=$(jq -r .user.login <<< "$1")
    body=$(jq -r '.body // ""' <<< "$1")
    if is_trusted_author "$author" && parsed=$(parse_task "$body"); then
        echo "task $parsed"
    elif is_bot_login "$author" && parsed=$(parse_finding "$body"); then
        echo "finding $parsed"
    fi
    return 0
}

# Like Slurm, a lease has two limits: a heartbeat timeout (no /claim or /beat
# for this long means the worker is gone) and, for tasks, a time limit from
# the claim, so that a worker that keeps beating but never finishes is cut off.
heartbeat_hours_for() { # KIND
    if [ "$1" = finding ]; then conf .heartbeat_hours_finding; else conf .heartbeat_hours; fi
}

# Closed, accepted tasks for a chip family, newest first, as JSON lines.
done_tasks() { # CHIP [ASSIGNEE]
    local who=""
    [ -n "${2:-}" ] && who="&assignee=$2"
    api "repos/$REPO/issues?state=closed&labels=$(label "done"),$(chip_label "$1")$who&sort=updated&direction=desc&per_page=30" |
        jq -c 'map(select(.pull_request == null)) | sort_by(.closed_at) | reverse | .[]'
}

# A task's time limit in hours, from recent results on the same chip family.
task_time_limit() { # CHIP
    local it json
    while read -r it; do
        [ -n "$it" ] || continue
        # A result an older schema wrote no longer passes: skip it. Not
        # `a && b` as the loop's last command, whose status then fails the
        # whole pipeline under pipefail (25 November: every claim on a chip
        # whose last result was an old one went unanswered).
        if json=$(accepted_result "$it"); then
            result_seconds "$json"
        fi
    done < <(done_tasks "$1" | head -n 10) | time_limit_hours "$(conf .time_limit_default_hours)"
}

# Why a worker should be paused, or nothing. A Mac whose last three results
# all failed setup, or whose leases expired three times in a week, is paused
# for 24 hours after the latest one, like a drained Slurm node.
worker_pause_reason() { # LOGIN CHIP
    local who=$1 now recent last json fails=0 n=0 since expired latest
    now=$(date -u +%s)
    while read -r recent; do
        [ -n "$recent" ] || continue
        n=$((n + 1))
        [ "$n" = 1 ] && last=$(jq -r .closed_at <<< "$recent")
        json=$(accepted_result "$recent") || continue
        [ "$(jq -r .setup <<< "$json")" = failed ] && fails=$((fails + 1))
    done < <(done_tasks "$2" "$who" | head -n 3)
    if [ "$n" = 3 ] && [ "$fails" = 3 ] && [ $((now - $(date -u -d "$last" +%s))) -lt 86400 ]; then
        echo "its last three results all failed setup"
        return 0
    fi
    since=$(date -u -d "@$((now - 7 * 86400))" +%Y-%m-%d)
    expired=$(api -X GET search/issues -f q="repo:$REPO in:comments \"lease held by @$who expired\" updated:>$since" \
        --jq '[.total_count, (.items | map(.updated_at) | max // "")] | @tsv' 2> /dev/null || true)
    latest=${expired#*$'\t'}
    if [ "${expired%%$'\t'*}" -ge 3 ] 2> /dev/null && [ -n "$latest" ] &&
        [ $((now - $(date -u -d "$latest" +%s))) -lt 86400 ]; then
        echo "three of its leases expired in the last week"
    fi
    return 0
}

# The accepted /result JSON of a closed task, or nothing.
accepted_result() { # ISSUE_JSON
    local n who kind body json
    n=$(jq -r .number <<< "$1")
    who=$(jq -r '.assignees[0].login // ""' <<< "$1")
    kind=$(issue_kind "$1")
    [ "${kind%% *}" = task ] && [ -n "$who" ] || return 1
    # shellcheck disable=SC2086 # fixed-format fields
    set -- $kind
    body=$(issue_comments "$n" | jq -r --arg w "$who" \
        '[.[] | select(.user.login == $w and (.body | startswith("/result")))] | last | .body // empty')
    json=$(result_json "$body") || return 1
    validate_result "$json" "$2" "$3" > /dev/null || return 1
    printf '%s\n' "$json"
}

# --- findings -------------------------------------------------------------

FINDINGS='[]'
load_findings() {
    FINDINGS=$(api --paginate "repos/$REPO/issues?state=all&labels=$(label finding)&per_page=100" |
        jq -sc 'add // [] | map(select(.pull_request == null and (.user.login | startswith("github-actions"))))')
}
findings_with() { # SIG -> issue numbers, oldest first
    jq -r --arg s "$1" 'map(select((.body // "") | contains("\nsig: \($s)\n"))) | map(.number) | sort | .[]' <<< "$FINDINGS"
}

# Files one sighting of a signature. Opens the finding the first time, and
# reopens it as regressed if it was closed. Duplicates from two results
# landing at once collapse onto the oldest issue.
upsert_finding() { # SIG TEST TOPIC SIGHTING
    local sig=$1 test=$2 topic=$3 sighting=$4 n created state assignee
    n=$(findings_with "$sig" | head -n 1)
    if [ -z "$n" ]; then
        created=$(api "repos/$REPO/issues" -f title="Finding ($topic): $test" \
            -f body="$(finding_body "$sig" "$test" "$topic" "$REPO")" \
            -f "labels[]=$(label finding)" -f "labels[]=$(label open)" -f "labels[]=fleet:topic:$topic" --jq .number)
        load_findings
        n=$(findings_with "$sig" | head -n 1)
        if [ -n "$n" ] && [ "$n" != "$created" ]; then
            reply "$created" "Duplicate of #$n."
            close_issue "$created" not_planned
        fi
        n=${n:-$created}
    else
        state=$(jq -r --argjson n "$n" '.[] | select(.number == $n) | .state' <<< "$FINDINGS")
        if [ "$state" = closed ]; then
            assignee=$(jq -r --argjson n "$n" '.[] | select(.number == $n) | .assignees[0].login // ""' <<< "$FINDINGS")
            api -X PATCH "repos/$REPO/issues/$n" -f state=open -f state_reason=reopened > /dev/null
            reopen_lease "$n" "$assignee"
            add_label "$n" "$(label regressed)"
            sighting="Seen again after this was closed, so it has been reopened as a regression. $sighting"
            load_findings
        fi
    fi
    reply "$n" "$sighting"
}

record_findings() { # RESULT TASK_ISSUE
    local json=$1 task=$2 where sig test topic
    where=$(jq -r '"`\(.upstream_commit[0:12])` · \(.chip) · macOS \(.macos)"' <<< "$json")
    load_findings
    while read -r sig test topic; do
        [ -n "$sig" ] && upsert_finding "$sig" "$test" "$topic" "Seen in #$task: $where."
    done < <(jq -r '.failures[] | "\(.sig) \(.test) \(.topic)"' <<< "$json")
}

# Compares a result with the latest accepted result for an older commit on
# the same chip family, and files test binaries that got much slower.
record_slowdowns() { # RESULT TASK_ISSUE CHIP COMMIT
    local json=$1 task=$2 chip=$3 commit=$4 prev="" candidate key old new
    while read -r candidate; do
        [ -n "$candidate" ] || continue
        [ "$(jq -r .number <<< "$candidate")" = "$task" ] && continue
        [[ $(issue_kind "$candidate") == "task $commit "* ]] && continue
        if prev=$(accepted_result "$candidate"); then break; fi
        prev=""
    done < <(api "repos/$REPO/issues?state=closed&labels=$(label "done"),$(chip_label "$chip")&sort=updated&direction=desc&per_page=20" |
        jq -c 'sort_by(.closed_at) | reverse | .[]')
    [ -n "$prev" ] || return 0
    load_findings
    while IFS="$(printf '\t')" read -r key old new; do
        [ -n "$key" ] || continue
        upsert_finding "$(signature "perf"$'\n'"$key")" "$key" performance \
            "Slower in #$task: $key took $new s at \`${commit:0:12}\` on $chip, against $old s at \`$(jq -r '.upstream_commit[0:12]' <<< "$prev")\`."
    done < <(slowdowns "$prev" "$json")
}

# Compares a result with other runs of the same commit on the same chip
# family. A failure signature in one run but not another is labelled flaky.
record_flaky() { # RESULT TASK_ISSUE CHIP COMMIT
    local json=$1 task=$2 chip=$3 commit=$4 it other sig f
    while read -r it; do
        [ -n "$it" ] || continue
        [ "$(jq -r .number <<< "$it")" = "$task" ] && continue
        [[ $(issue_kind "$it") == "task $commit "* ]] || continue
        other=$(accepted_result "$it") || continue
        load_findings
        while read -r sig; do
            [ -n "$sig" ] || continue
            f=$(findings_with "$sig" | head -n 1)
            [ -n "$f" ] || continue
            jq -e --argjson f "$f" --arg l "$(label flaky)" '.[] | select(.number == $f) | [.labels[].name] | index($l)' \
                > /dev/null <<< "$FINDINGS" && continue
            add_label "$f" "$(label flaky)"
            reply "$f" "Looks flaky: two runs of \`${commit:0:12}\` on $chip disagree (#$task and #$(jq -r .number <<< "$it")). One saw this failure and the other did not."
        done < <(flaky_sigs "$json" "$other")
    done < <(done_tasks "$chip")
}

# For a task that checks a fix, reports on the finding whether the failure
# came back. A failure that came back is already reopened by record_findings.
record_verification() { # RESULT TASK_ISSUE FINDING
    local json=$1 task=$2 f=${3:-} finding sig
    [ -n "$f" ] || return 0
    finding=$(api "repos/$REPO/issues/$f")
    [[ $(issue_kind "$finding") == finding\ * ]] || return 0
    sig=$(parse_finding "$(jq -r '.body // ""' <<< "$finding")" | cut -d ' ' -f 1)
    if jq -e --arg s "$sig" 'any(.failures[]; .sig == $s)' > /dev/null <<< "$json"; then
        return 0
    elif failures_complete "$json"; then
        # Only a complete failure list proves the signature absent: a trimmed
        # list may simply have dropped it.
        reply "$f" "Verified in #$task: not seen at $(jq -r '"`\(.upstream_commit[0:12])` on \(.chip), macOS \(.macos)"' <<< "$json")."
    fi
}

# When a finding is closed as completed (usually by its fix's pull request
# merging), opens high-priority tasks at the latest main for the chips that
# saw it: Slurm's afterok dependency.
verify_round() { # FINDING
    local f=$1 head chips chip tasks opened=""
    head=$(api "repos/$REPO/commits/main" --jq .sha)
    chips=$(issue_comments "$f" | jq -r '.[] | select(.user.login | startswith("github-actions")) | .body' |
        sed -nE 's/.*Apple (M[0-9]{1,2}).*/\1/p' | tr 'M' 'm' | sort -u)
    [ -n "$chips" ] || chips=$(conf '.chips[]')
    tasks=$(api --paginate "repos/$REPO/issues?state=all&labels=$(label task)&per_page=100" | jq -sc 'add // []')
    for chip in $chips; do
        is_chip "$chip" || continue
        jq -e --arg h "$head" --arg c "$chip" --arg f "$f" 'any(.[]; (.body // "") as $b
            | ($b | contains("\ncommit: \($h)\n")) and ($b | contains("\nchip: \($c)\n")) and ($b | contains("\nverifies: \($f)\n")))' \
            > /dev/null <<< "$tasks" && continue
        opened="$opened #$(new_task "$head" "$chip" "" "$(conf .priorities.verify)" "" "$f")"
    done
    [ -n "$opened" ] && reply "$f" "Closed as fixed, so the fleet will check it: verification tasks$opened at \`${head:0:12}\`. If the failure shows up again, this finding reopens."
    return 0
}

new_task() { # COMMIT CHIP MACOS PRIORITY REPLICA VERIFIES -> issue number
    local title
    title="Test matrix: ${1:0:12} on Apple $(tr 'm' 'M' <<< "$2")"
    [ -n "$3" ] && title="$title, macOS $3"
    [ -n "$5" ] && title="$title (run $5)"
    [ -n "$6" ] && title="$title, verifying #$6"
    api "repos/$REPO/issues" -f title="$title" -f body="$(task_body "$1" "$2" "$REPO" "$3" "$4" "$5" "$6")" \
        -f "labels[]=$(label task)" -f "labels[]=$(label open)" -f "labels[]=$(chip_label "$2")" --jq .number
}

# --- commands -------------------------------------------------------------

handle_command() { # ISSUE COMMENT_ID LOGIN COMMAND BODY
    local n=$1 cid=$2 who=$3 cmd=$4 body=$5
    local issue state labels assignee kind kind_label
    issue=$(api "repos/$REPO/issues/$n")
    state=$(jq -r .state <<< "$issue")
    labels=$(jq -c '[.labels[].name]' <<< "$issue")
    assignee=$(jq -r '.assignees[0].login // ""' <<< "$issue")
    has() { jq -e --arg l "$1" 'index($l) != null' > /dev/null <<< "$labels"; }

    kind=$(issue_kind "$issue")
    if [ -z "$kind" ]; then
        react "$cid" -1
        reply "$n" "This issue is not a valid fleet task or finding, so the command was ignored."
        return
    fi
    # shellcheck disable=SC2086 # fixed-format fields
    set -- $kind
    kind_label=$(label "$1")

    case "$cmd" in
        claim)
            if [ "$state" != open ] || ! has "$(label open)" || [ -n "$assignee" ]; then
                react "$cid" -1
                reply "$n" "@$who this is not open for leasing. Run \`tools/fleet/fleet.sh next\` or \`fleet.sh findings\` for another one."
                return
            fi
            local other
            other=$(api "repos/$REPO/issues?state=open&assignee=$who&labels=$(label leased),$kind_label" --jq '.[0].number // empty')
            if [ -n "$other" ]; then
                react "$cid" -1
                reply "$n" "@$who you already hold a lease on #$other. Finish it or \`/release\` it first."
                return
            fi
            if [ "$1" = task ]; then
                local pause
                if grep -qx "commit: $2" <<< "$(done_tasks "$3" "$who" | jq -r '.body // ""')"; then
                    react "$cid" -1
                    reply "$n" "@$who you have already run commit \`${2:0:12}\` on this chip. Another Mac's run is what makes the comparison useful, so please take a different task."
                    return
                fi
                pause=$(worker_pause_reason "$who" "$3")
                if [ -n "$pause" ]; then
                    react "$cid" -1
                    reply "$n" "@$who this worker is paused for 24 hours because $pause. Check \`tools/fleet/fleet.sh doctor\` and the local logs before trying again."
                    return
                fi
            fi
            api "repos/$REPO/issues/$n/assignees" -f "assignees[]=$who" > /dev/null
            local assignees winner
            assignees=$(api "repos/$REPO/issues/$n" --jq '[.assignees[].login]')
            if ! jq -e --arg w "$who" 'index($w) != null' > /dev/null <<< "$assignees"; then
                react "$cid" -1
                reply "$n" "@$who GitHub would not assign this issue to you, so the lease was not granted."
                return
            fi
            # Two claims at once: the earliest /claim comment wins, in every run.
            if [ "$(jq length <<< "$assignees")" -gt 1 ]; then
                winner=$(issue_comments "$n" | jq -r --argjson a "$assignees" \
                    '[.[] | select((.body | test("^\\s*/claim\\s*$")) and (.user.login as $u | $a | index($u) != null))] | first | .user.login')
                while read -r a; do
                    [ "$a" = "$winner" ] || unassign "$n" "$a"
                done < <(jq -r '.[]' <<< "$assignees")
                if [ "$winner" != "$who" ]; then
                    react "$cid" -1
                    reply "$n" "@$who @$winner claimed this first."
                    return
                fi
            fi
            remove_label "$n" "$(label open)"
            add_label "$n" "$(label leased)"
            local beat limit
            beat=$(heartbeat_hours_for "$1")
            react "$cid" +1
            if [ "$1" = task ]; then
                limit=$(task_time_limit "$3")
                reply "$n" "Leased to @$who. Post \`/beat\` at least every $beat hours (\`tools/fleet/fleet.sh run $n\` does it every 30 minutes) and submit within $limit hours, or the task goes back to the queue. Submit with \`fleet.sh submit $n\`, or hand it back with \`fleet.sh release $n <reason>\`.
<!-- fleet-lease: limit=$limit -->"
            else
                reply "$n" "@$who is working on this. Comment \`/beat\` at least every $beat hours to keep the lease, reference this issue from the fix's pull request, or \`/release other\` to hand it back."
            fi
            ;;
        beat)
            if [ "$assignee" = "$who" ] && has "$(label leased)"; then react "$cid" +1; else react "$cid" -1; fi
            ;;
        release\ *)
            if { [ "$assignee" = "$who" ] || is_maintainer "$who"; } && has "$(label leased)"; then
                reopen_lease "$n" "$assignee"
                react "$cid" +1
                reply "$n" "Released (${cmd#release }). This is open again."
            else
                react "$cid" -1
            fi
            ;;
        result)
            local json errors="" commit=${2:-} chip=${3:-} meta
            meta=$(task_meta "$(jq -r '.body // ""' <<< "$issue")")
            if [ "$1" != task ] || [ "$assignee" != "$who" ] || ! has "$(label leased)"; then
                delete_comment "$cid"
                reply "$n" "@$who a result was removed because you do not hold the lease on this task."
                return
            fi
            if ! json=$(result_json "$body") || ! errors=$(validate_result "$json" "$commit" "$chip" "$(jq -r .macos <<< "$meta")"); then
                [ -n "$errors" ] || errors="the comment must be /result followed by one json block and nothing else"
                delete_comment "$cid"
                reply "$n" "@$who your result was removed because it does not match the schema:
$(sed 's/^/- /' <<< "$errors")

Post results with \`tools/fleet/fleet.sh submit $n\` only. See [docs/fleet.md]($FLEET_URL)."
                return
            fi
            local summary
            summary=$(jq -r '
                def row($n; $c; $t): if $c == null then "| \($n) | - | - | - | - |"
                    else "| \($n) | \($c.passed) | \($c.failed) | \($c.ignored) | \($t // "-") s |" end;
                "Result accepted.\n\n**\(.chip)** · `\(.hw_model)` · macOS \(.macos) · Game Porting Toolkit \(.gptk) · setup \(.setup) (\(.timings.setup // "-") s)\n\n"
                + "| Suite | Passed | Failed | Ignored | Wall time |\n|---|---|---|---|---|\n"
                + row("Native"; .native; .timings.native) + "\n" + row("Wine + D3DMetal"; .wine; .timings.wine)
                + (if (.failures | length) > 0
                   then "\n\nFailures (each is filed as a finding):\n\n| Test | Topic | Signature |\n|---|---|---|\n"
                       + (.failures[:30] | map("| `\(.test)` | \(.topic) | `\(.sig)` |") | join("\n")) else "" end)' <<< "$json")
            remove_label "$n" "$(label leased)"
            add_label "$n" "$(label "done")"
            if result_failing "$json"; then add_label "$n" "$(label failing)"; else add_label "$n" "$(label passing)"; fi
            react "$cid" +1
            reply "$n" "$summary"
            close_issue "$n" completed
            record_findings "$json" "$n"
            record_slowdowns "$json" "$n" "$chip" "$commit"
            record_flaky "$json" "$n" "$chip" "$commit"
            record_verification "$json" "$n" "$(jq -r '.verifies // empty' <<< "$meta")"
            ;;
    esac
}

# --- moderation -----------------------------------------------------------

moderate_comment() { # ISSUE COMMENT_ID NODE_ID LOGIN BODY
    local verdict
    verdict=$(moderate "$5")
    case "$verdict" in
        delete\ *)
            delete_comment "$2"
            add_label "$1" "$(label review)"
            reply "$1" "@$4 a comment was removed (${verdict#delete }). Fleet issues never carry game material or links to it. See [docs/fleet-legal.md]($LEGAL_URL)."
            ;;
        hide\ *)
            hide_comment "$3"
            add_label "$1" "$(label review)"
            reply "$1" "A comment was hidden for maintainer review (${verdict#hide }). See [docs/fleet-legal.md]($LEGAL_URL)."
            ;;
    esac
}

moderate_issue() { # ISSUE TEXT
    local verdict
    verdict=$(moderate "$2")
    case "$verdict" in
        delete\ *)
            api -X PATCH "repos/$REPO/issues/$1" -f title="Removed by the fleet moderation bot" \
                -f body="The text of this issue was removed (${verdict#delete }). See [docs/fleet-legal.md]($LEGAL_URL). Maintainers: delete the earlier revision from the edit history." \
                -f state=closed -f state_reason=not_planned > /dev/null
            add_label "$1" "$(label review)"
            api -X PUT "repos/$REPO/issues/$1/lock" -f lock_reason=off-topic > /dev/null
            ;;
        hide\ *)
            add_label "$1" "$(label review)"
            api -X PUT "repos/$REPO/issues/$1/lock" -f lock_reason=off-topic > /dev/null
            reply "$1" "This issue was locked for maintainer review (${verdict#hide }). See [docs/fleet-legal.md]($LEGAL_URL)."
            ;;
    esac
}

# --- entry points ---------------------------------------------------------

handle_event() {
    local ev=${GITHUB_EVENT_PATH:?} action n
    action=$(jq -r .action "$ev")
    n=$(jq -r .issue.number "$ev")
    [ "$(jq -r .sender.type "$ev")" = Bot ] && return 0
    [ "$(jq -r '.issue.pull_request != null' "$ev")" = true ] && return 0

    if [ "${GITHUB_EVENT_NAME:?}" = issues ]; then
        if [ "$action" = closed ]; then
            if [ "$(jq -r .issue.state_reason "$ev")" = completed ] &&
                [[ $(issue_kind "$(jq -c .issue "$ev")") == finding\ * ]]; then
                verify_round "$n"
            fi
        else
            moderate_issue "$n" "$(jq -r '.issue.title + "\n" + (.issue.body // "")' "$ev")"
        fi
        return 0
    fi

    local cid node who body is_fleet cmd
    cid=$(jq -r .comment.id "$ev")
    node=$(jq -r .comment.node_id "$ev")
    who=$(jq -r .comment.user.login "$ev")
    body=$(jq -r '.comment.body // ""' "$ev")
    valid_login "$who" || return 0
    is_fleet=$(jq -r --arg t "$(label task)" --arg f "$(label finding)" \
        '[.issue.labels[].name] | (index($t) != null) or (index($f) != null)' "$ev")

    if [ "$action" = created ] && [ "$is_fleet" = true ]; then
        cmd=$(parse_command "$body")
        if [ -n "$cmd" ]; then
            handle_command "$n" "$cid" "$who" "$cmd" "$body"
            return 0
        fi
    fi
    moderate_comment "$n" "$cid" "$node" "$who" "$body"
}

expire_leases() {
    local now n who kind beat comments last claimed limit why
    now=$(date -u +%s)
    while read -r n who kind; do
        [ -n "$n" ] || continue
        beat=$(heartbeat_hours_for "$kind")
        comments=$(issue_comments "$n")
        last=$(jq -r --arg w "$who" '[.[] | select(.user.login == $w)
            | select(.body | test("^\\s*/(claim|beat)\\s*$")) | .created_at] | max // empty' <<< "$comments")
        claimed=$(jq -r --arg w "$who" '[.[] | select(.user.login == $w)
            | select(.body | test("^\\s*/claim\\s*$")) | .created_at] | max // empty' <<< "$comments")
        limit=$(jq -r '[.[] | .body | capture("fleet-lease: limit=(?<h>[0-9]+)").h] | last // empty' <<< "$comments")
        why=""
        if [ -z "$last" ] || [ $((now - $(date -u -d "$last" +%s))) -gt $((beat * 3600)) ]; then
            why="went $beat hours without a \`/beat\`"
        elif [ "$kind" = task ] && [ -n "$limit" ] && [ -n "$claimed" ] &&
            [ $((now - $(date -u -d "$claimed" +%s))) -gt $((limit * 3600)) ]; then
            why="passed its $limit-hour time limit"
        fi
        if [ -n "$why" ]; then
            [ "$who" = - ] && who=""
            reopen_lease "$n" "$who"
            reply "$n" "The lease held by @${who:-nobody} expired: it $why. This is open again."
        fi
    done < <(api --paginate "repos/$REPO/issues?state=open&labels=$(label leased)&per_page=100" |
        jq -rs --arg f "$(label finding)" 'add // [] | .[]
            | "\(.number) \(.assignees[0].login // "-") \(if ([.labels[].name] | index($f)) then "finding" else "task" end)"')
}

# Opens a round of test-matrix tasks for the commit main points to, unless
# that commit already has tasks or the last round is too recent. Open,
# unleased tasks for older commits are closed as superseded.
# Opens a round of test-matrix tasks for the commit main points to, unless
# that commit already has tasks or the last round is too recent: one task per
# chip family and replica. Open, unclaimed tasks for older commits are closed
# as superseded. Then fills coverage gaps.
auto_round() {
    local head tasks newest age hours chip n r
    head=$(api "repos/$REPO/commits/main" --jq .sha)
    hours=$(conf .auto_round_hours)
    tasks=$(api --paginate "repos/$REPO/issues?state=all&labels=$(label task)&sort=created&direction=desc&per_page=100" |
        jq -sc 'add // [] | map(select(.pull_request == null))')
    if jq -e --arg h "$head" 'any(.[]; (.body // "") | contains("\ncommit: \($h)\n"))' > /dev/null <<< "$tasks"; then
        echo "main ($head) already has tasks"
    else
        newest=$(jq -r 'map(.created_at) | max // empty' <<< "$tasks")
        age=$((hours * 3600))
        [ -n "$newest" ] && age=$(($(date -u +%s) - $(date -u -d "$newest" +%s)))
        if [ "$age" -lt $((hours * 3600)) ]; then
            echo "last round is less than $hours hours old"
        else
            while read -r n; do
                [ -n "$n" ] || continue
                reply "$n" "Superseded by a new round for \`${head:0:12}\`."
                remove_label "$n" "$(label open)"
                close_issue "$n" not_planned
            done < <(jq -r --arg o "$(label open)" '.[] | select(.state == "open" and ([.labels[].name] | index($o))
                and ((.body // "") | contains("\nverifies: ") | not)) | .number' <<< "$tasks")
            for chip in $(conf '.chips[]'); do
                for r in $(seq 1 "$(conf .replicas)"); do
                    [ "$(conf .replicas)" = 1 ] && r=""
                    new_task "$head" "$chip" "" "$(conf .priorities.round)" "$r" "" > /dev/null
                done
            done
            echo "opened a round for $head"
            return 0
        fi
    fi
    fill_gaps "$tasks" "$head"
}

# Once the newest round is gap_delay_hours old, opens one task for each chip
# family and target macOS version that has no accepted result and no task
# yet, so that coverage spreads across macOS versions. Only for chips whose
# round tasks have all been taken.
fill_gaps() { # TASKS_JSON HEAD
    local tasks=$1 head=${2:-} commit created have chip m it json
    commit=$(jq -r 'map(select((.body // "") | contains("\nverifies: ") | not)) | max_by(.created_at) | .body // ""' <<< "$tasks" |
        sed -nE 's/^commit: *([0-9a-f]{40}) *$/\1/p')
    [ -n "$commit" ] || return 0
    # Only for the commit main points to: gaps of an older round would test
    # code that main has moved past (25 November: twelve gap tasks for a
    # round 50 merges behind main).
    [ -z "$head" ] || [ "$commit" = "$head" ] || return 0
    created=$(jq -r --arg c "$commit" 'map(select((.body // "") | contains("\ncommit: \($c)\n"))) | map(.created_at) | min' <<< "$tasks")
    [ $(($(date -u +%s) - $(date -u -d "$created" +%s))) -ge $(($(conf .gap_delay_hours) * 3600)) ] || return 0
    have=""
    while read -r it; do
        [ -n "$it" ] || continue
        json=$(accepted_result "$it") || continue
        have="$have $(jq -r '"\(.chip | ascii_downcase | capture("(?<f>m[0-9]+)").f)/\(.macos | split(".")[0])"' <<< "$json")"
    done < <(jq -c --arg c "$commit" --arg d "$(label "done")" '.[] | select(((.body // "") | contains("\ncommit: \($c)\n"))
        and ([.labels[].name] | index($d)))' <<< "$tasks")
    for chip in $(conf '.chips[]'); do
        # A chip whose ordinary round tasks are still open has nobody on it
        # yet: more specific tasks would only add noise.
        jq -e --arg c "$commit" --arg h "$chip" 'any(.[]; .state == "open" and ((.body // "") as $b
            | ($b | contains("\ncommit: \($c)\n")) and ($b | contains("\nchip: \($h)\n"))
            and ($b | contains("\nmacos: ") | not) and ($b | contains("\nverifies: ") | not)))' \
            > /dev/null <<< "$tasks" && continue
        for m in $(conf '.macos_targets[]'); do
            [[ " $have " == *" $chip/$m "* ]] && continue
            jq -e --arg c "$commit" --arg h "$chip" --arg m "$m" 'any(.[]; (.body // "") as $b
                | ($b | contains("\ncommit: \($c)\n")) and ($b | contains("\nchip: \($h)\n")) and ($b | contains("\nmacos: \($m)\n")))' \
                > /dev/null <<< "$tasks" && continue
            echo "gap: $chip on macOS $m -> #$(new_task "$commit" "$chip" "$m" "$(conf .priorities.gap)" "" "")"
        done
    done
}

case "${1:-}" in
    event) handle_event ;;
    expire) expire_leases ;;
    autotask) auto_round ;;
    *)
        echo "usage: queue.sh event|expire|autotask" >&2
        exit 2
        ;;
esac
