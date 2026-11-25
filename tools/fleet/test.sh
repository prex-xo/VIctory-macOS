#!/bin/bash
# Tests for the fleet's parsing, validation and moderation (no network).
# Run: tools/fleet/test.sh
# shellcheck disable=SC2016,SC2001 # literal backticks, jq and GraphQL in single quotes
set -euo pipefail
# shellcheck source=tools/fleet/lib.sh
. "$(dirname "$0")/lib.sh"

pass=0
fail=0
ok() { # NAME COMMAND... : the command must succeed
    local name=$1
    shift
    if "$@" > /dev/null 2>&1; then pass=$((pass + 1)); else
        fail=$((fail + 1))
        echo "FAIL: $name"
    fi
}
no() { # NAME COMMAND... : the command must fail
    local name=$1
    shift
    if "$@" > /dev/null 2>&1; then
        fail=$((fail + 1))
        echo "FAIL: $name"
    else pass=$((pass + 1)); fi
}
eq() { # NAME EXPECTED ACTUAL
    if [ "$2" = "$3" ]; then pass=$((pass + 1)); else
        fail=$((fail + 1))
        printf 'FAIL: %s\n  expected: %s\n  actual:   %s\n' "$1" "$2" "$3"
    fi
}

SHA=0123456789abcdef0123456789abcdef01234567
REPO=example/VIctory-macOS
tmp=$(mktemp -d)
trap 'rm -rf "$tmp"' EXIT

# --- tasks
body=$(task_body "$SHA" m3 "$REPO")
eq "task body round-trips" "$SHA m3" "$(parse_task "$body")"
eq "task body passes moderation" "" "$(moderate "$body")"
eq "task body with CRLF" "$SHA m3" "$(parse_task "$(sed 's/$/\r/' <<< "$body")")"
no "task with unknown chip" parse_task "$(task_body "$SHA" m9 "$REPO")"
no "task with short commit" parse_task "$(task_body 0123abc m3 "$REPO")"
no "task without a block" parse_task "commit: $SHA"
no "task with another kind" parse_task "$(sed 's/^kind: test-matrix/kind: shader/' <<< "$body")"

# --- toolkit filenames
eq "toolkit 4 beta 3 filename" "4.0-beta3" "$(gptk_version_of 'Game_Porting_Toolkit_4.0_beta_3.dmg')"
eq "toolkit 2 beta 3 filename" "2.0-beta3" "$(gptk_version_of 'Game_Porting_Toolkit_2.0_beta_3.dmg')"
eq "toolkit stable filename" "3.0" "$(gptk_version_of 'Game_Porting_Toolkit_3.0.dmg')"
eq "toolkit legacy beta filename" "3.0-beta4" "$(gptk_version_of 'Game_Porting_Toolkit_3.0-beta4.dmg')"
eq "toolkit beta without number" "4.0-beta" "$(gptk_version_of 'Game_Porting_Toolkit_4.0_beta.dmg')"
eq "toolkit version prefix" "3.0" "$(gptk_version_of 'Game_Porting_Toolkit_v3.0.dmg')"
eq "toolkit path with spaces" "4.0-beta3" "$(gptk_version_of '/tmp/path with spaces/Game_Porting_Toolkit_4.0 beta 3.dmg')"
eq "missing toolkit filename" "unknown" "$(gptk_version_of '')"
eq "unrecognized toolkit filename" "unknown" "$(gptk_version_of 'Game_Porting_Toolkit_latest.dmg')"
eq "unrelated disk image" "unknown" "$(gptk_version_of '4.0.dmg')"
eq "unfinished toolkit download" "unknown" "$(gptk_version_of 'Game_Porting_Toolkit_4.0_beta_3.dmg.part')"
eq "unsupported toolkit beta number" "unknown" "$(gptk_version_of 'Game_Porting_Toolkit_4.0_beta_10.dmg')"

# --- commands
eq "claim" claim "$(parse_command "/claim")"
eq "claim with spaces" claim "$(parse_command "  /claim  ")"
eq "beat" beat "$(parse_command "/beat")"
eq "release" "release no_time" "$(parse_command "/release no_time")"
eq "release with a bad reason" "" "$(parse_command "/release because")"
eq "claim with extra text" "" "$(parse_command "/claim
please")"
eq "command not on the first line" "" "$(parse_command "hi
/claim")"
eq "plain comment" "" "$(parse_command "Works on my M2 Air")"
eq "result" result "$(parse_command "/result")"

# --- results
good=$(jq -cn --arg c "$SHA" '{schema: 1, kind: "test-matrix", upstream_commit: $c, hw_model: "Mac15,12",
    chip: "Apple M3 Pro", macos: "15.4.1", gptk: "3.0", setup: "ok",
    native: {passed: 300, failed: 0, ignored: 2}, wine: {passed: 340, failed: 3, ignored: 0},
    failed_tests: ["wine/victory_gpu::tests::depth16", "wine/victory_shader::nan_sin"]}')
comment=$(printf '/result\n```json\n%s\n```\n' "$good")
eq "result block extracts" "$good" "$(result_json "$comment")"
no "result with text before the block" result_json "$(printf '/result\nhello\n```json\n{}\n```')"
no "result with text after the block" result_json "$(printf '/result\n```json\n{}\n```\nmore')"
no "result without a block" result_json "/result"
ok "valid result" validate_result "$good" "$SHA" m3
no "wrong commit" validate_result "$good" "${SHA%7}8" m3
no "wrong chip family" validate_result "$good" "$SHA" m2
bad() { validate_result "$(jq -c "$1" <<< "$good")" "$SHA" m3; }
no "extra field" bad '.notes = "see https://example.com"'
no "missing field" bad 'del(.gptk)'
no "free text in hw_model" bad '.hw_model = "Mac15,12 see my link"'
no "free text in chip" bad '.chip = "Apple M3 Pro https://x"'
no "free text in macos" bad '.macos = "15.4 beta (lol)"'
no "free text in a test name" bad '.failed_tests = ["wine/a b"]'
no "test name without a suite" bad '.failed_tests = ["victory_gpu::x"]'
no "too many test names" bad '.failed_tests = [range(101) | "wine/t\(.)"]'
no "negative count" bad '.native.failed = -1'
no "fractional count" bad '.native.failed = 1.5'
no "extra count field" bad '.native.note = 1'
no "bad setup value" bad '.setup = "maybe"'
no "array instead of object" validate_result '[1]' "$SHA" m3
no "not JSON" validate_result 'nope' "$SHA" m3
no "too large" validate_result "$(printf '%9000s' x)" "$SHA" m3
ok "null suites when setup failed" bad '.setup = "failed" | .native = null | .wine = null | .failed_tests = []'
ok "unknown toolkit version" bad '.gptk = "unknown"'
ok "normalized toolkit beta version passes schema" bad '.gptk = "4.0-beta3"'
ok "MacBook model id" bad '.hw_model = "MacBookAir10,1"'
ok "failing result" result_failing "$good"
no "passing result" result_failing "$(jq -c '.wine.failed = 0' <<< "$good")"
ok "setup failure is failing" result_failing "$(jq -c '.setup = "failed" | .native = null | .wine = null' <<< "$good")"
eq "validation error text" "upstream_commit is not the task commit" "$(validate_result "$good" "${SHA%7}8" m3 || true)"

# --- moderation
verdict() { moderate "$1" | cut -d ' ' -f 1; }
eq "clean comment" "" "$(verdict "Setup failed at the Wine build on my M1, log attached locally")"
eq "torrent link" delete "$(verdict "grab it here magnet:?xt=urn:btih:abc")"
eq "file host link" delete "$(verdict "https://MEGA.nz/file/abc")"
eq "game dump talk" hide "$(verdict "where do I get a game dump")"
eq "dumped my game" hide "$(verdict "I dumped my game yesterday")"
eq "decryption" hide "$(verdict "how do I decrypt the package")"
eq "container names" hide "$(verdict "it is an XVD file")"
eq "image attachment" hide "$(verdict "![x](https://github.com/user-attachments/assets/1)")"
eq "long encoded run" hide "$(verdict "$(printf 'A%.0s' $(seq 1 220))")"
eq "words containing short tokens" "" "$(verdict "the dumpster fire of the rpfoo tickets")"

# --- cargo output
cat > "$tmp/native.log" << 'EOF'
running 3 tests
test a::one ... ok
test a::two ... FAILED
test a::three ... ignored
test result: FAILED. 1 passed; 1 failed; 1 ignored; 0 measured; 0 filtered out; finished in 0.01s

running 2 tests
test b::four ... ok
test b::five - should panic ... FAILED
test result: FAILED. 1 passed; 1 failed; 0 ignored; 0 measured; 0 filtered out; finished in 0.00s
EOF
eq "counts are summed" "2 2 1" "$(parse_counts "$tmp/native.log")"
eq "counts json" '{"passed":2,"failed":2,"ignored":1}' "$(counts_json "$tmp/native.log")"
eq "failed names keep only plain names" "native/a::two" "$(failed_names "$tmp/native.log" native)"
printf 'error: could not compile\n' > "$tmp/broken.log"
eq "no test run gives null" null "$(counts_json "$tmp/broken.log")"
printf 'test result: ok. 3 passed; 0 failed; 1 ignored;\nerror: a later test binary failed to build\n' > "$tmp/partial.log"
eq "successful command keeps its counts" '{"passed":3,"failed":0,"ignored":1}' "$(counts_json "$tmp/partial.log" 0)"
eq "failed command cannot pass on partial counts" null "$(counts_json "$tmp/partial.log" 101)"
eq "aborted command cannot pass on partial counts" null "$(counts_json "$tmp/partial.log" 137)"
eq "counted failures retain their counts" '{"passed":2,"failed":2,"ignored":1}' "$(counts_json "$tmp/native.log" 101)"
eq "failed command with no tests gives null" null "$(counts_json "$tmp/broken.log" 101)"
eq "chip family" m3 "$(chip_family_of "Apple M3 Pro")"
eq "chip family of an Intel Mac" "" "$(chip_family_of "Intel(R) Core(TM) i7")"

# --- report
r1=$(jq -c '. + {n: 1, closed: "2026-11-20T10:00:00Z"}' <<< "$good")
r2=$(jq -c '. + {n: 2, closed: "2026-11-21T10:00:00Z", chip: "Apple M1", hw_model: "MacBookAir10,1", failed_tests: [], wine: {passed: 343, failed: 0, ignored: 0}}' <<< "$good")
r3=$(jq -c '. + {n: 3, closed: "2026-11-22T10:00:00Z", chip: "Apple M2", setup: "failed", native: null, wine: null, failed_tests: []}' <<< "$good")
report=$(printf '%s\n%s\n%s\n' "$r1" "$r2" "$r3" | report_markdown)
ok "report counts results" grep -q '^3 accepted results across 1 commits' <<< "$report"
ok "report coverage" grep -q '3 of 5: m1, m2, m3 | 2 of 3' <<< "$report"
ok "report chip-specific failure" grep -q 'victory_gpu::tests::depth16`: fails on m3; passes on m1' <<< "$report"
ok "report lists every result" grep -q '^| #3 |' <<< "$report"
eq "empty report" "None found." "$(: | report_markdown | sed -n '/^## Tests/{n;n;p;}')"

# --- worker EXIT cleanup (stub all network, checkout and build operations)
# A separate shell makes EXIT run after cmd_run's locals have gone out of scope.
cat > "$tmp/worker-cleanup.sh" <<'EOF'
case_dir=$1
setup_status=$2
validation_status=$3
stub_native_status=$4
stub_wine_status=$5
set -- help
. "$0" > /dev/null
CACHE="$case_dir/cache"
WORK="$case_dir/work"
mkdir -p "$WORK/tools/macos"
printf '#!/bin/sh\nexit %s\n' "$setup_status" > "$WORK/setup.sh"
chmod +x "$WORK/setup.sh"
printf '#!/bin/sh\necho "test result: ok. 1 passed; 0 failed; 0 ignored;"\nexit %s\n' "$stub_wine_status" > "$WORK/tools/macos/test-wine.sh"
chmod +x "$WORK/tools/macos/test-wine.sh"
accepted() { return 0; }
load_task() { T_COMMIT=0123456789abcdef0123456789abcdef01234567; T_CHIP=m3; }
require_lease() { return 0; }
git() { if [ "${3:-}" = rev-parse ]; then echo "$T_COMMIT"; fi; return 0; }
gh() { echo 'unexpected GitHub call' >&2; exit 99; }
# Replace the heartbeat subshell itself with sleep, so tests leave no sleep child.
sleep() { exec /bin/sleep 30; }
kill() { printf '%s\n' "$1" > "$case_dir/killed-pid"; builtin kill "$@"; }
cargo() { echo 'test result: ok. 1 passed; 0 failed; 0 ignored;'; return "$stub_native_status"; }
sysctl() { echo Mac15,11; }
sw_vers() { echo 15.4; }
brand() { echo 'Apple M3 Max'; }
gptk_version() { echo 4.0-beta3; }
say() {
    case "$*" in "running setup.sh"*) jobs -p > "$case_dir/heartbeat-pid" ;; esac
}
if [ "$validation_status" != 0 ]; then validate_result() { return 1; }; fi
cmd_run 1
EOF
worker_case() {
    local dir="$tmp/worker-$1" status=0
    mkdir -p "$dir"
    bash -c '. "$1" "$2" "$3" "$4" "$5" "$6"' "$FLEET_DIR/fleet.sh" "$tmp/worker-cleanup.sh" "$dir" "$2" "$3" "${6:-0}" "${7:-0}" > "$dir/output" 2>&1 || status=$?
    eq "worker $1 exit status" "$4" "$status"
    no "worker $1 has no unbound variable" grep -q 'unbound variable' "$dir/output"
    ok "worker $1 ran heartbeat cleanup" test -s "$dir/killed-pid"
    if [ -s "$dir/heartbeat-pid" ]; then
        local pid
        pid=$(cat "$dir/heartbeat-pid")
        no "worker $1 heartbeat stopped" kill -0 "$pid"
        # Always stop this test's job, including when testing the original bug.
        kill "$pid" 2> /dev/null || true
        if [ -s "$dir/killed-pid" ]; then
            eq "worker $1 cleaned up its own heartbeat" "$pid" "$(cat "$dir/killed-pid")"
        fi
    else
        ok "worker $1 started heartbeat" test -s "$dir/heartbeat-pid"
    fi
    if [ -f "$dir/cache/logs/1/result.json" ]; then
        eq "worker $1 setup result" "$5" "$(jq -r .setup "$dir/cache/logs/1/result.json")"
    fi
}
worker_case success 0 0 0 ok
worker_case setup-failure 1 0 0 failed
worker_case invalid-result 0 1 1 unused

# A test command can print passing binaries and still fail or crash later.
suite_exit_case() {
    worker_case "$1" 0 0 0 ok "$2" "$3"
    local result
    result=$(cat "$tmp/worker-$1/cache/logs/1/result.json")
    eq "$1 native counts" "$4" "$(jq -c .native <<< "$result")"
    eq "$1 Wine counts" "$5" "$(jq -c .wine <<< "$result")"
    ok "$1 is a failing result" result_failing "$result"
    ok "$1 still passes schema" validate_result "$result" "$SHA" m3
}
passed_counts='{"passed":1,"failed":0,"ignored":0}'
suite_exit_case native-error 101 0 null "$passed_counts"
suite_exit_case wine-error 0 101 "$passed_counts" null
suite_exit_case both-error 101 101 null null
suite_exit_case wine-aborted 0 137 "$passed_counts" null

# --- claim polling (GitHub updates the assignee and labels separately)
cat > "$tmp/worker-claim.sh" <<'EOF'
case_dir=$1
scenario=$2
set -- help
. "$0" > /dev/null
REPO=example/VIctory-macOS
accepted() { return 0; }
me() { echo worker; }
sleep() { :; }
gh() { printf '%s\n' "$*" >> "$case_dir/posts"; }
load_count=0
load_task() {
    load_count=$((load_count + 1))
    printf '%s\n' "$load_count" > "$case_dir/loads"
    T_STATE=OPEN
    T_ASSIGNEE=""
    T_LABELS="fleet:open"
    [ "$load_count" -gt 1 ] || return 0
    case "$scenario" in
        delayed-label)
            T_ASSIGNEE=worker
            if [ "$load_count" = 3 ]; then T_LABELS=""; fi
            if [ "$load_count" -ge 4 ]; then T_LABELS="fleet:leased"; fi
            ;;
        delayed-assignee)
            if [ "$load_count" -ge 3 ]; then T_ASSIGNEE=worker; T_LABELS="fleet:leased"; fi
            ;;
        immediate) T_ASSIGNEE=worker; T_LABELS="fleet:leased" ;;
        other-worker) T_ASSIGNEE=another-worker; T_LABELS="fleet:leased" ;;
        missing-label) T_ASSIGNEE=worker ;;
    esac
}
cmd_claim 1
EOF
claim_case() {
    local dir="$tmp/claim-$1" status=0
    mkdir -p "$dir"
    bash -c '. "$1" "$2" "$3"' "$FLEET_DIR/fleet.sh" "$tmp/worker-claim.sh" "$dir" "$1" > "$dir/output" 2>&1 || status=$?
    eq "claim $1 exit status" "$2" "$status"
    ok "claim $1 outcome" grep -q "$3" "$dir/output"
    eq "claim $1 polling count" "$4" "$(cat "$dir/loads")"
    eq "claim $1 posts once" 'issue comment 1 -R example/VIctory-macOS --body /claim' "$(cat "$dir/posts")"
}
claim_case delayed-label 0 'task #1 is yours' 4
claim_case delayed-assignee 0 'task #1 is yours' 3
claim_case immediate 0 'task #1 is yours' 2
claim_case other-worker 1 'went to someone else' 2
claim_case missing-label 1 'no lease after 3 minutes' 37

echo "fleet tests: $pass passed, $fail failed"
[ "$fail" = 0 ]
