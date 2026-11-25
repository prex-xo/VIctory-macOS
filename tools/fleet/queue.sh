#!/bin/bash
# The GitHub Actions side of the Mac test fleet (see docs/fleet.md).
# Leases fleet tasks through issue comments, checks results, screens free
# text in fleet issues and expires stale leases. Run by the fleet-queue and
# fleet-leases workflows with GH_TOKEN set. Never evaluates comment text:
# comments are only matched against fixed patterns and the result schema.
#
# Usage: queue.sh event    (reads GITHUB_EVENT_NAME and GITHUB_EVENT_PATH)
#        queue.sh expire
# shellcheck disable=SC2016,SC2001 # literal backticks, jq and GraphQL in single quotes
set -euo pipefail
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
hide_comment() {
    api graphql -f id="$1" -f query='
        mutation($id: ID!) { minimizeComment(input: {subjectId: $id, classifier: OFF_TOPIC}) { clientMutationId } }' > /dev/null
}

reopen_task() { # ISSUE LOGIN
    [ -n "$2" ] && unassign "$1" "$2"
    remove_label "$1" "$(label leased)"
    add_label "$1" "$(label open)"
}

# --- commands -------------------------------------------------------------

handle_command() { # ISSUE COMMENT_ID LOGIN COMMAND BODY
    local n=$1 cid=$2 who=$3 cmd=$4 body=$5
    local issue state author labels assignee task commit chip
    issue=$(api "repos/$REPO/issues/$n")
    state=$(jq -r .state <<< "$issue")
    author=$(jq -r .user.login <<< "$issue")
    labels=$(jq -c '[.labels[].name]' <<< "$issue")
    assignee=$(jq -r '.assignees[0].login // ""' <<< "$issue")
    has() { jq -e --arg l "$1" 'index($l) != null' > /dev/null <<< "$labels"; }

    if ! is_maintainer "$author" || ! task=$(parse_task "$(jq -r '.body // ""' <<< "$issue")"); then
        react "$cid" -1
        reply "$n" "This issue is not a valid fleet task, so the command was ignored."
        return
    fi
    commit=${task% *}
    chip=${task#* }

    case "$cmd" in
        claim)
            if [ "$state" != open ] || ! has "$(label open)" || [ -n "$assignee" ]; then
                react "$cid" -1
                reply "$n" "@$who this task is not open for leasing. Run \`tools/fleet/fleet.sh next\` for another one."
                return
            fi
            local other
            other=$(api "repos/$REPO/issues?state=open&assignee=$who&labels=$(label leased)" --jq '.[0].number // empty')
            if [ -n "$other" ]; then
                react "$cid" -1
                reply "$n" "@$who you already hold a lease on #$other. Finish it or \`/release\` it first."
                return
            fi
            api "repos/$REPO/issues/$n/assignees" -f "assignees[]=$who" > /dev/null
            if ! api "repos/$REPO/issues/$n" --jq '.assignees[].login' | grep -qx "$who"; then
                react "$cid" -1
                reply "$n" "@$who GitHub would not assign this issue to you, so the lease was not granted."
                return
            fi
            remove_label "$n" "$(label open)"
            add_label "$n" "$(label leased)"
            local hours until
            hours=$(conf .lease_hours)
            until=$(date -u -d "+$hours hours" '+%Y-%m-%d %H:%M UTC')
            react "$cid" +1
            reply "$n" "Leased to @$who until $until. \`tools/fleet/fleet.sh run $n\` keeps it alive with \`/beat\`. Submit with \`fleet.sh submit $n\`, or hand it back with \`fleet.sh release $n <reason>\`."
            ;;
        beat)
            if [ "$assignee" = "$who" ] && has "$(label leased)"; then
                react "$cid" +1
            else
                react "$cid" -1
            fi
            ;;
        release\ *)
            local reason=${cmd#release }
            if { [ "$assignee" = "$who" ] || is_maintainer "$who"; } && has "$(label leased)"; then
                reopen_task "$n" "$assignee"
                react "$cid" +1
                reply "$n" "Released ($reason). The task is open again."
            else
                react "$cid" -1
            fi
            ;;
        result)
            local json errors
            if [ "$assignee" != "$who" ] || ! has "$(label leased)"; then
                delete_comment "$cid"
                reply "$n" "@$who a result was removed because you do not hold the lease on this task."
                return
            fi
            if ! json=$(result_json "$body") || ! errors=$(validate_result "$json" "$commit" "$chip"); then
                [ -n "${errors:-}" ] || errors="the comment must be /result followed by one json block and nothing else"
                delete_comment "$cid"
                reply "$n" "@$who your result was removed because it does not match the schema:
$(sed 's/^/- /' <<< "$errors")

Post results with \`tools/fleet/fleet.sh submit $n\` only. See [docs/fleet.md]($FLEET_URL)."
                return
            fi
            local summary
            summary=$(jq -r '
                def row($n; $c): if $c == null then "| \($n) | - | - | - |"
                    else "| \($n) | \($c.passed) | \($c.failed) | \($c.ignored) |" end;
                "Result accepted.\n\n**\(.chip)** · `\(.hw_model)` · macOS \(.macos) · Game Porting Toolkit \(.gptk) · setup \(.setup)\n\n"
                + "| Suite | Passed | Failed | Ignored |\n|---|---|---|---|\n"
                + row("Native"; .native) + "\n" + row("Wine + D3DMetal"; .wine)
                + (if (.failed_tests | length) > 0
                   then "\n\nFailing tests:\n```\n" + (.failed_tests[:30] | join("\n")) + "\n```" else "" end)' <<< "$json")
            remove_label "$n" "$(label leased)"
            add_label "$n" "$(label "done")"
            if result_failing "$json"; then add_label "$n" "$(label failing)"; else add_label "$n" "$(label passing)"; fi
            react "$cid" +1
            reply "$n" "$summary"
            api -X PATCH "repos/$REPO/issues/$n" -f state=closed -f state_reason=completed > /dev/null
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
        moderate_issue "$n" "$(jq -r '.issue.title + "\n" + (.issue.body // "")' "$ev")"
        return 0
    fi

    local cid node who body is_task cmd
    cid=$(jq -r .comment.id "$ev")
    node=$(jq -r .comment.node_id "$ev")
    who=$(jq -r .comment.user.login "$ev")
    body=$(jq -r '.comment.body // ""' "$ev")
    valid_login "$who" || return 0
    is_task=$(jq -r --arg l "$(label task)" '[.issue.labels[].name] | index($l) != null' "$ev")

    if [ "$action" = created ] && [ "$is_task" = true ]; then
        cmd=$(parse_command "$body")
        if [ -n "$cmd" ]; then
            handle_command "$n" "$cid" "$who" "$cmd" "$body"
            return 0
        fi
    fi
    moderate_comment "$n" "$cid" "$node" "$who" "$body"
}

expire_leases() {
    local hours now
    hours=$(conf .lease_hours)
    now=$(date -u +%s)
    api --paginate "repos/$REPO/issues?state=open&labels=$(label leased)&per_page=100" |
        jq -rs 'add // [] | .[] | "\(.number) \(.assignees[0].login // "-")"' |
        while read -r n who; do
            local last age
            last=$(api --paginate "repos/$REPO/issues/$n/comments?per_page=100" |
                jq -rs --arg w "$who" 'add // [] | [.[] | select(.user.login == $w)
                    | select(.body | test("^\\s*/(claim|beat)\\s*$")) | .created_at] | max // empty')
            if [ -z "$last" ]; then
                age=$((hours * 3600 + 1))
            else
                age=$((now - $(date -u -d "$last" +%s)))
            fi
            if [ "$age" -gt $((hours * 3600)) ]; then
                [ "$who" = - ] && who=""
                reopen_task "$n" "$who"
                reply "$n" "The lease expired after $hours hours without a \`/beat\`. The task is open again."
            fi
        done
}

case "${1:-}" in
    event) handle_event ;;
    expire) expire_leases ;;
    *)
        echo "usage: queue.sh event|expire" >&2
        exit 2
        ;;
esac
