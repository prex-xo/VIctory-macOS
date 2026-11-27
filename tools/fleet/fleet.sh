#!/bin/bash
# Mac test fleet worker (see docs/fleet.md; agents read docs/fleet-join.md).
# Takes a test-matrix task from this repository's issues, runs the native
# and Wine test suites at the task's commit, and posts the counts. Only the
# checked result JSON is posted. Logs stay in ~/Library/Caches/victory-fleet.
# shellcheck disable=SC2016 # literal backticks, jq and GraphQL in single quotes
set -euo pipefail
# shellcheck source=tools/fleet/lib.sh
. "$(dirname "$0")/lib.sh"

ROOT="$(cd "$FLEET_DIR/../.." && pwd)"
REPO="${FLEET_REPO:-$(conf .repo)}"
CACHE="$HOME/Library/Caches/victory-fleet"
WORK="$CACHE/work"
LEGAL="$ROOT/docs/fleet-legal.md"
# One persistent build folder for every task: setup.sh builds Wine and
# VIctory inside the checkout's target/, so each worktree's target/ is a
# symlink to this, and nothing is rebuilt from scratch between tasks.
TARGET="$CACHE/target"
WINE_BIN="$TARGET/wine-cx/bin/wine"
MIN_FREE_GB=25

say() { printf 'fleet: %s\n' "$*"; }
die() {
    printf 'fleet: %s\n' "$*" >&2
    exit 1
}

usage() {
    cat << 'EOF'
Usage: tools/fleet/fleet.sh <command>

  doctor            check this Mac is ready (run first)
  rules             print the fleet rules (an agent shows them to the person)
  accept --confirmed  record that the person said yes to the rules (or run
                    plain "accept" in Terminal to read and type "I agree")
  gptk              open Apple's download page and wait for the Game Porting Toolkit
  next              list open tasks for this Mac's chip
  claim N           take task or finding N (posts /claim)
  run N             run both test suites for task N (long; posts /beat while running)
  submit N          post the result of task N (posts /result)
  release N REASON  hand task or finding N back; REASON is setup_failed, wrong_hardware, no_time or other
  status            list the tasks and findings you hold
  work              keep going: claim, run and submit tasks one after another until stopped
  stop              make a running "work" loop finish after its current task
  findings [TOPIC]  list unclaimed findings (failure signatures to investigate)
  repro N           rerun finding N's test at the latest main and show the local failure text
EOF
}

me() { gh api user --jq .login; }
macos_major() { sw_vers -productVersion | cut -d . -f 1; }

# Checks out COMMIT in the fleet worktree, with target/ linked to the shared
# build folder.
prepare_worktree() { # COMMIT
    # cmd_work runs cmd_run inside an if, where set -e is suspended for this
    # whole call stack, so every required step checks its own exit status.
    mkdir -p "$TARGET" || return 1
    git -C "$ROOT" fetch --quiet "https://github.com/$REPO.git" "$1" || return 1
    git -C "$ROOT" worktree remove --force "$WORK" > /dev/null 2>&1 || true
    git -C "$ROOT" worktree prune || return 1
    git -C "$ROOT" worktree add --quiet --detach "$WORK" "$1" || return 1
    [ "$(git -C "$WORK" rev-parse HEAD)" = "$1" ] || { say "checkout did not land on $1"; return 1; }
    rm -rf "$WORK/target" || return 1
    ln -s "$TARGET" "$WORK/target" || return 1
}

# Open tasks this Mac can take, best first: highest priority, then oldest.
# Skips tasks that need another macOS version, and commits this person has
# already run on this chip (the queue refuses both anyway). Prints
# "NUMBER<TAB>PRIORITY<TAB>TITLE" lines.
pick_tasks() {
    local chip major done_commits it author body task meta m
    chip=$(my_chip)
    major=$(macos_major)
    done_commits=$(gh issue list -R "$REPO" --state closed --assignee @me --label "$(label "done")" \
        --label "$(chip_label "$chip")" --limit 100 --json body --jq '.[].body' |
        sed -nE 's/^commit: *([0-9a-f]{40}) *$/\1/p' | sort -u) || return 1
    gh issue list -R "$REPO" --state open --label "$(label task)" --label "$(label open)" \
        --label "$(chip_label "$chip")" --limit 100 --json number,title,body,author --jq '.[] | @json' |
        while read -r it; do
            author=$(jq -r '.author.login' <<< "$it")
            is_trusted_author "$author" || continue
            body=$(jq -r '.body' <<< "$it")
            task=$(parse_task "$body") || continue
            grep -qx "${task% *}" <<< "$done_commits" && continue
            meta=$(task_meta "$body")
            m=$(jq -r .macos <<< "$meta")
            [ -z "$m" ] || [ "$m" = "$major" ] || continue
            printf '%s\t%s\t%s\n' "$(jq -r '.number' <<< "$it")" "$(jq -r .priority <<< "$meta")" \
                "$(jq -r '.title' <<< "$it")"
        done | sort -t "$(printf '\t')" -k2,2nr -k1,1n
}
brand() { sysctl -n machdep.cpu.brand_string 2> /dev/null || true; }
my_chip() { chip_family_of "$(brand)"; }
legal_hash() { shasum -a 256 "$LEGAL" | cut -d ' ' -f 1; }
accepted() { [ -f "$CACHE/accepted-legal" ] && [ "$(cut -d ' ' -f 1 "$CACHE/accepted-legal")" = "$(legal_hash)" ]; }
GPTK_URL="https://developer.apple.com/download/all/?q=game%20porting%20toolkit"
gptk_dmg() {
    local candidate newest=""
    # GPTK_DMG names the download, as it does for setup.sh. A worker that
    # launchd starts may not be allowed to read Downloads.
    if [ -z "${1:-}" ] && [ -f "${GPTK_DMG:-}" ]; then
        printf '%s\n' "$GPTK_DMG"
        return
    fi
    # setup.sh chooses the newest download by modification time.
    for candidate in "${1:-$HOME/Downloads}"/Game_Porting_Toolkit*.dmg; do
        [ -f "$candidate" ] || continue
        if [[ -z "$newest" || "$candidate" -nt "$newest" ]]; then
            newest=$candidate
        fi
    done
    if [ -n "$newest" ]; then printf '%s\n' "$newest"; fi
}

gptk_version() {
    gptk_version_of "$(gptk_dmg)"
}

need_number() { [[ ${1:-} =~ ^[0-9]{1,7}$ ]] || die "give a task number"; }

# Loads task N into T_COMMIT, T_CHIP, T_STATE, T_ASSIGNEE and T_LABELS.
# Loads issue N: T_KIND (task or finding), T_STATE, T_ASSIGNEE, T_LABELS,
# plus T_COMMIT and T_CHIP for tasks, or F_SIG, F_TEST and F_TOPIC for findings.
load_issue() {
    local j author body task finding
    j=$(gh issue view "$1" -R "$REPO" --json body,author,state,assignees,labels) || die "cannot read issue #$1"
    author=$(jq -r .author.login <<< "$j")
    body=$(jq -r .body <<< "$j")
    T_STATE=$(jq -r .state <<< "$j")
    T_ASSIGNEE=$(jq -r '.assignees[0].login // ""' <<< "$j")
    T_LABELS=$(jq -r '[.labels[].name] | join(" ")' <<< "$j")
    if is_trusted_author "$author" && task=$(parse_task "$body"); then
        T_KIND=task
        T_COMMIT=${task% *}
        T_CHIP=${task#* }
        T_MACOS=$(task_meta "$body" | jq -r .macos)
    elif is_bot_login "$author" && finding=$(parse_finding "$body"); then
        T_KIND=finding
        read -r F_SIG F_TEST F_TOPIC <<< "$finding"
    else
        die "issue #$1 is not a fleet task or finding"
    fi
}

load_task() {
    load_issue "$1"
    [ "$T_KIND" = task ] || die "issue #$1 is a finding, not a test task"
    [ "$T_CHIP" = "$(my_chip)" ] || die "task #$1 is for $T_CHIP Macs and this is $(brand)"
    [ -z "$T_MACOS" ] || [ "$T_MACOS" = "$(macos_major)" ] || die "task #$1 needs macOS $T_MACOS and this Mac runs $(sw_vers -productVersion)"
}

require_lease() {
    [ "$T_ASSIGNEE" = "$(me)" ] && [[ " $T_LABELS " == *" $(label leased) "* ]] ||
        die "you do not hold the lease on #$1 (it may have expired); run: tools/fleet/fleet.sh claim $1"
}

cmd_doctor() {
    local ok=1 free
    check() {
        if [ "$2" = 1 ]; then say "ok    $1"; else
            say "FAIL  $1: $3"
            ok=0
        fi
    }
    t() { if "$@" > /dev/null 2>&1; then echo 1; else echo 0; fi; }
    check "Apple Silicon Mac" "$(t test "$(sysctl -n hw.optional.arm64 2> /dev/null)" = 1)" "the fleet needs an M-series Mac"
    check "macOS 15 or newer" "$(t test "$(sw_vers -productVersion 2> /dev/null | cut -d . -f 1)" -ge 15)" "update macOS"
    check "known chip ($(brand))" "$(t is_chip "$(my_chip)")" "this chip is not in tools/fleet/fleet.json yet"
    check "Homebrew" "$(t command -v brew)" "the person installs it from https://brew.sh"
    check "git and jq" "$(t sh -c 'command -v git && command -v jq')" "install with: brew install git jq"
    check "GitHub CLI signed in" "$(t gh auth status)" "the person runs: gh auth login"
    if [ -x "$WINE_BIN" ]; then
        check "setup already done once" 1 ""
    else
        free=$(df -g "$HOME" | awk 'NR == 2 { print $4 }')
        check "$MIN_FREE_GB GB free for the first setup" "$(t test "${free:-0}" -ge $MIN_FREE_GB)" "only ${free:-?} GB free"
        check "Game Porting Toolkit in Downloads" "$(t test -n "$(gptk_dmg)")" \
            "run tools/fleet/fleet.sh gptk; the person signs in to Apple, accepts Apple's terms and clicks Download"
    fi
    check "fleet terms accepted" "$(t accepted)" "show the person tools/fleet/fleet.sh rules; after their yes, run: tools/fleet/fleet.sh accept --confirmed"
    if [ "$ok" = 1 ]; then say "this Mac is ready"; else die "fix the FAIL lines above first"; fi
}

cmd_rules() {
    sed -n '/^## The rules/,/^## /p' "$LEGAL" | sed '$d'
    printf 'Full text: %s\n' "https://github.com/$REPO/blob/main/docs/fleet-legal.md"
}

# The person accepts once. Either they type "I agree" in Terminal, or their
# agent shows them the rules (cmd_rules), gets an explicit yes in the chat and
# runs "accept --confirmed". The record notes which route was used.
cmd_accept() {
    local how
    if [ "${1:-}" = --confirmed ]; then
        how=chat
    elif [ -t 0 ] && [ -t 1 ]; then
        cmd_rules
        printf '\nType "I agree" to accept the fleet rules: '
        local answer
        read -r answer
        [ "$answer" = "I agree" ] || die "not accepted"
        how=terminal
    else
        die "show the person the rules (tools/fleet/fleet.sh rules); only after their explicit yes, run: tools/fleet/fleet.sh accept --confirmed"
    fi
    mkdir -p "$CACHE"
    printf '%s %s %s\n' "$(legal_hash)" "$how" "$(date -u +%Y-%m-%dT%H:%M:%SZ)" > "$CACHE/accepted-legal"
    say "fleet rules accepted ($how)"
}

# Opens Apple's download page and waits for the .dmg to finish downloading.
# Apple requires the person to sign in and accept its licence themselves.
cmd_gptk() {
    local dmg size last="" _
    if [ -x "$WINE_BIN" ] || [ -n "$(gptk_dmg)" ]; then
        say "the Game Porting Toolkit is already here"
        return
    fi
    open "$GPTK_URL"
    say "opened Apple's download page. Sign in with any Apple account, accept Apple's terms,"
    say "and click Download on the newest Game Porting Toolkit. Leave the .dmg in Downloads; don't open it."
    say "waiting up to 30 minutes for the download to finish..."
    for _ in $(seq 1 360); do
        sleep 5
        dmg=$(gptk_dmg)
        [ -n "$dmg" ] || continue
        size=$(stat -f %z "$dmg" 2> /dev/null || echo 0)
        if [ "$size" -gt 0 ] && [ "$size" = "$last" ]; then
            say "found $(basename "$dmg") ($((size / 1048576)) MB)"
            return
        fi
        last=$size
    done
    die "no Game Porting Toolkit download after 30 minutes; run tools/fleet/fleet.sh gptk again when ready"
}

cmd_next() {
    local list
    is_chip "$(my_chip)" || die "unknown chip: $(brand)"
    list=$(pick_tasks | head -n 10 | awk -F '\t' '{ printf "#%s\tpriority %s\t%s\n", $1, $2, $3 }')
    if [ -z "$list" ]; then say "no open tasks for this Mac right now"; else printf '%s\n' "$list"; fi
}

cmd_claim() {
    need_number "${1:-}"
    accepted || die "the fleet terms are not accepted yet; run tools/fleet/fleet.sh doctor"
    load_issue "$1"
    [ "$T_KIND" = finding ] || load_task "$1"
    [ "$T_STATE" = OPEN ] && [[ " $T_LABELS " == *" $(label open) "* ]] || die "#$1 is not open"
    local who next_step="tools/fleet/fleet.sh run $1"
    [ "$T_KIND" = finding ] && next_step="tools/fleet/fleet.sh repro $1"
    who=$(me)
    gh issue comment "$1" -R "$REPO" --body "/claim" > /dev/null
    say "claim posted; waiting for the queue to answer"
    for _ in $(seq 1 36); do
        sleep 5
        load_issue "$1"
        if [ "$T_ASSIGNEE" = "$who" ] && [[ " $T_LABELS " == *" $(label leased) "* ]]; then
            say "#$1 is yours. Next: $next_step"
            return
        fi
        # Assignment can appear before the queue finishes updating the labels.
        [ -n "$T_ASSIGNEE" ] && [ "$T_ASSIGNEE" != "$who" ] &&
            die "task #$1 went to someone else; run tools/fleet/fleet.sh next"
    done
    die "no lease after 3 minutes; read the replies on https://github.com/$REPO/issues/$1"
}

cmd_run() {
    need_number "${1:-}"
    local n=$1 logs setup native wine failed binaries result native_status=0 wine_status=0
    accepted || die "the fleet terms are not accepted yet; run tools/fleet/fleet.sh doctor"
    load_task "$n"
    require_lease "$n"
    logs="$CACHE/logs/$n"
    mkdir -p "$logs" || return 1

    say "checking out ${T_COMMIT:0:12} in $WORK (builds are kept in $TARGET)"
    # cmd_work calls this function inside an if, where set -e does not reach
    # it: check the checkout explicitly, before the heartbeat or setup start.
    if ! prepare_worktree "$T_COMMIT"; then
        say "could not prepare the checkout for ${T_COMMIT:0:12} in $WORK; not running setup"
        return 1
    fi

    # Keep the lease alive while the suites run.
    (while sleep 1800; do gh issue comment "$n" -R "$REPO" --body "/beat" > /dev/null 2>&1 || true; done) &
    local beat_pid=$!
    # Capture the PID now; cmd_run's locals are gone when EXIT runs.
    # shellcheck disable=SC2064
    trap "kill $beat_pid 2> /dev/null || true" EXIT

    local t0 t_setup=null t_native=null t_wine=null pn pw
    say "running setup.sh (the first run builds Wine and takes about an hour); log: $logs/setup.log"
    t0=$(date +%s)
    (cd "$WORK" && ./setup.sh) > "$logs/setup.log" 2>&1 && setup=ok || setup=failed
    t_setup=$(($(date +%s) - t0))
    : > "$logs/native.log"
    : > "$logs/wine.log"
    if [ "$setup" = ok ]; then
        say "running the native tests; log: $logs/native.log"
        t0=$(date +%s)
        (cd "$WORK" && PATH="/opt/homebrew/opt/rustup/bin:$PATH" cargo test --workspace --no-fail-fast) \
            > "$logs/native.log" 2>&1 || native_status=$?
        t_native=$(($(date +%s) - t0))
        say "running the Wine + D3DMetal tests; log: $logs/wine.log"
        t0=$(date +%s)
        (cd "$WORK" && WINE="$WINE_BIN" WINEDLLOVERRIDES="d3d12,d3d12core,dxgi=b" \
            tools/macos/test-wine.sh --workspace --no-fail-fast) > "$logs/wine.log" 2>&1 || wine_status=$?
        t_wine=$(($(date +%s) - t0))
    else
        say "setup failed; the result will say so. The log stays on this Mac: $logs/setup.log"
    fi
    native=$(counts_json "$logs/native.log" "$native_status")
    wine=$(counts_json "$logs/wine.log" "$wine_status")
    pn=$(parse_cargo "$logs/native.log")
    pw=$(parse_cargo "$logs/wine.log")
    failed=$(jq -sc 'add | .[0:100]' <<< "$(failures_json "$pn" native) $(failures_json "$pw" wine)")
    binaries=$(jq -sc 'add' <<< "$(binaries_json "$pn" native) $(binaries_json "$pw" wine)")
    # The text behind each signature stays here, for repro and for whoever fixes it.
    { signature_lines "$pn" native; signature_lines "$pw" wine; } > "$logs/signatures.tsv"

    result=$(jq -n --arg commit "$T_COMMIT" --arg model "$(sysctl -n hw.model)" --arg chip "$(brand)" \
        --arg macos "$(sw_vers -productVersion)" --arg gptk "$(gptk_version)" --arg setup "$setup" \
        --argjson native "$native" --argjson wine "$wine" --argjson failed "$failed" --argjson binaries "$binaries" \
        --argjson ts "$t_setup" --argjson tn "$t_native" --argjson tw "$t_wine" \
        '{schema: 1, kind: "test-matrix", upstream_commit: $commit, hw_model: $model, chip: $chip,
          macos: $macos, gptk: $gptk, setup: $setup, native: $native, wine: $wine, failures: $failed,
          timings: {setup: $ts, native: $tn, wine: $tw}, binaries: $binaries}')
    # Store and validate the compact form; a pretty-printed result can exceed
    # the size limit while its compact form does not.
    result=$(shrink_result "$result") ||
        die "the result still exceeds $RESULT_MAX_BYTES bytes after trimming; please report this"
    validate_result "$result" "$T_COMMIT" "$T_CHIP" "$T_MACOS" > "$logs/result-errors.txt" ||
        die "the result does not pass the schema (see $logs/result-errors.txt); please report this"
    printf '%s\n' "$result" > "$logs/result.json" || return 1
    jq . <<< "$result"
    say "done. Show this to the person, then: tools/fleet/fleet.sh submit $n"
}

cmd_submit() {
    need_number "${1:-}"
    local n=$1 file="$CACHE/logs/$1/result.json" result
    load_task "$n"
    require_lease "$n"
    [ -f "$file" ] || die "no result for task #$n yet; run tools/fleet/fleet.sh run $n"
    result=$(jq -c . "$file")
    validate_result "$result" "$T_COMMIT" "$T_CHIP" "$T_MACOS" > /dev/null || die "$file does not pass the schema"
    gh issue comment "$n" -R "$REPO" --body "$(printf '/result\n```json\n%s\n```\n' "$result")" > /dev/null
    say "result posted; waiting for the queue to check it"
    for _ in $(seq 1 24); do
        sleep 5
        if [ "$(gh issue view "$n" -R "$REPO" --json state --jq .state)" = CLOSED ]; then
            say "accepted: https://github.com/$REPO/issues/$n"
            return
        fi
    done
    die "not accepted yet; read the replies on https://github.com/$REPO/issues/$n"
}

cmd_release() {
    need_number "${1:-}"
    is_release_reason "${2:-}" || die "give a reason: $(conf '.release_reasons | join(", ")')"
    load_issue "$1"
    require_lease "$1"
    gh issue comment "$1" -R "$REPO" --body "/release $2" > /dev/null
    say "release posted for #$1"
}

cmd_status() {
    gh issue list -R "$REPO" --state open --assignee @me --label "$(label leased)" \
        --json number,title --jq '.[] | "#\(.number)\t\(.title)"'
}

STOP_FILE="$CACHE/stop"

# Waits up to SECONDS, waking early if someone runs "fleet.sh stop".
nap() {
    local left=$1
    while [ "$left" -gt 0 ] && [ ! -f "$STOP_FILE" ]; do
        sleep 10
        left=$((left - 10))
    done
}

# The continuous worker. Takes one task at a time, for as long as it runs:
# claim, run, submit, repeat. When there is nothing to do it checks again
# every 30 minutes. It stops on "fleet.sh stop", when the rules change, when
# setup fails, or after three tasks in a row go wrong, so that a person can
# look before it uses the Mac any further.
cmd_work() {
    accepted || die "the fleet terms are not accepted yet; run tools/fleet/fleet.sh doctor"
    local chip n held tasks fails=0 done_count=0 last=""
    chip=$(my_chip)
    is_chip "$chip" || die "unknown chip: $(brand)"
    mkdir -p "$CACHE"
    rm -f "$STOP_FILE"
    say "working until stopped. To stop after the current task: tools/fleet/fleet.sh stop"
    while [ ! -f "$STOP_FILE" ]; do
        accepted || die "the fleet rules changed; the person must accept them again (fleet.sh rules, then accept)"
        # The list can still show, for a few seconds, the task whose result
        # the queue just accepted and closed: skip the one done last.
        held=$(gh issue list -R "$REPO" --state open --assignee @me --label "$(label task)" --label "$(label leased)" \
            --json number --jq '.[].number' 2> /dev/null | grep -vx "${last:-0}" | head -n 1 || true)
        if [ -n "$held" ]; then
            n=$held
            say "resuming task #$n"
        else
            # A failed GitHub call is not an empty queue: say so, or a
            # worker that cannot sign in looks idle for ever.
            if ! tasks=$(pick_tasks 2> /dev/null); then
                say "could not read the tasks from GitHub (is gh signed in and online?); trying again in 5 minutes"
                nap 300
                continue
            fi
            n=$(head -n 1 <<< "$tasks" | cut -f 1)
            if [ -z "$n" ]; then
                say "no tasks for this Mac right now ($done_count done so far)"
                prewarm
                say "checking again in 30 minutes"
                nap 1800
                continue
            fi
            if ! (cmd_claim "$n"); then
                say "could not claim #$n; trying the next task"
                nap 60
                continue
            fi
        fi
        if (cmd_run "$n") && (cmd_submit "$n"); then
            fails=0
            last=$n
            done_count=$((done_count + 1))
            say "task #$n done ($done_count so far)"
            if [ "$(jq -r .setup "$CACHE/logs/$n/result.json" 2> /dev/null)" = failed ]; then
                die "setup.sh failed on this Mac (log: $CACHE/logs/$n/setup.log); stopping so that a person can look"
            fi
        else
            fails=$((fails + 1))
            say "task #$n went wrong ($fails in a row)"
            [ "$fails" -lt 3 ] || die "three tasks in a row went wrong; stopping so that a person can look"
            nap 300
        fi
    done
    rm -f "$STOP_FILE"
    say "stopped after $done_count tasks"
}

# Backfill: while idle, pre-build the latest main at the lowest CPU priority,
# so the next task starts with a warm build. Posts nothing.
prewarm() {
    local head
    [ -x "$WINE_BIN" ] || return 0
    head=$(git ls-remote "https://github.com/$REPO.git" refs/heads/main 2> /dev/null | cut -f 1)
    [ -n "$head" ] && [ "$head" != "$(cat "$CACHE/prewarmed" 2> /dev/null)" ] || return 0
    say "idle: pre-building main (${head:0:12}) at low priority so the next task starts warm"
    (prepare_worktree "$head") || return 0
    mkdir -p "$CACHE/logs"
    (cd "$WORK" && PATH="/opt/homebrew/opt/rustup/bin:$PATH" nice -n 19 cargo build --quiet --workspace --tests &&
        PATH="/opt/homebrew/opt/rustup/bin:$PATH" nice -n 19 cargo build --quiet --workspace --tests \
            --target x86_64-pc-windows-gnullvm) \
        > "$CACHE/logs/prewarm.log" 2>&1 || true
    echo "$head" > "$CACHE/prewarmed"
}

cmd_stop() {
    mkdir -p "$CACHE"
    touch "$STOP_FILE"
    say "the work loop will stop after its current task"
}

cmd_findings() {
    local args=(--label "$(label finding)" --label "$(label open)")
    if [ -n "${1:-}" ]; then
        [[ " $FLEET_TOPICS " == *" $1 "* ]] || die "unknown topic; topics: $FLEET_TOPICS"
        args+=(--label "fleet:topic:$1")
    fi
    gh issue list -R "$REPO" --state open "${args[@]}" --limit 30 \
        --json number,title --jq '.[] | "#\(.number)\t\(.title)"'
}

# Reruns a finding's test locally at the latest main. Nothing is posted.
cmd_repro() {
    need_number "${1:-}"
    load_issue "$1"
    [ "$T_KIND" = finding ] || die "#$1 is not a finding"
    [ -x "$WINE_BIN" ] || die "run a test task first so that setup.sh has built everything"
    local suite=${F_TEST%%/*} rest=${F_TEST#*/} filter log
    filter=${rest#*::}
    [ "$filter" = "$rest" ] && filter=""
    log="$CACHE/logs/repro-$1.log"
    mkdir -p "$CACHE/logs"
    prepare_worktree "$(git ls-remote "https://github.com/$REPO.git" refs/heads/main | cut -f 1)"
    say "rerunning $F_TEST ($F_TOPIC) at $(git -C "$WORK" rev-parse --short=12 HEAD); log: $log"
    if [ "$suite" = native ]; then
        (cd "$WORK" && PATH="/opt/homebrew/opt/rustup/bin:$PATH" cargo test --workspace --no-fail-fast -- ${filter:+"$filter"}) > "$log" 2>&1 || true
    else
        (cd "$WORK" && WINE="$WINE_BIN" WINEDLLOVERRIDES="d3d12,d3d12core,dxgi=b" \
            tools/macos/test-wine.sh --workspace --no-fail-fast -- ${filter:+"$filter"}) > "$log" 2>&1 || true
    fi
    local parsed
    parsed=$(parse_cargo "$log")
    if grep -q "^$F_SIG" <<< "$(signature_lines "$parsed" "$suite")"; then
        say "reproduced: same signature $F_SIG. Local failure text (do not post it):"
        signature_lines "$parsed" "$suite" | grep "^$F_SIG" | cut -f 3
    elif grep -q "^F" <<< "$parsed"; then
        say "the test still fails, but with a different signature. Local failure text (do not post it):"
        signature_lines "$parsed" "$suite" | cut -f 2,3
    else
        say "not reproduced at the latest main: $F_TEST passes or did not run (see $log)"
    fi
}

case "${1:-help}" in
    doctor) cmd_doctor ;;
    rules) cmd_rules ;;
    accept) cmd_accept "${2:-}" ;;
    gptk) cmd_gptk ;;
    next) cmd_next ;;
    claim) cmd_claim "${2:-}" ;;
    run) cmd_run "${2:-}" ;;
    submit) cmd_submit "${2:-}" ;;
    release) cmd_release "${2:-}" "${3:-}" ;;
    status) cmd_status ;;
    findings) cmd_findings "${2:-}" ;;
    work) cmd_work ;;
    stop) cmd_stop ;;
    repro) cmd_repro "${2:-}" ;;
    help | -h | --help) usage ;;
    *)
        usage >&2
        exit 2
        ;;
esac
