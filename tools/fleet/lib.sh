#!/bin/bash
# Shared, side-effect-free functions for the Mac test fleet (see docs/fleet.md).
# Sourced by fleet.sh (the worker), queue.sh (the GitHub Actions side),
# maintainer.sh and test.sh. Works with macOS's bash 3.2 and GNU bash.

# shellcheck disable=SC2016 # jq programs in single quotes
FLEET_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
FLEET_CONF="$FLEET_DIR/fleet.json"

conf() { jq -r "$1" "$FLEET_CONF"; }
label() { jq -r --arg k "$1" '.labels[$k]' "$FLEET_CONF"; }
chip_label() { printf 'fleet:%s\n' "$1"; }

is_maintainer() {
    jq -e --arg a "$1" '.maintainers | index($a) != null' "$FLEET_CONF" > /dev/null
}

is_chip() {
    [ -n "$1" ] && jq -e --arg c "$1" '.chips | index($c) != null' "$FLEET_CONF" > /dev/null
}

is_release_reason() {
    [ -n "$1" ] && jq -e --arg r "$1" '.release_reasons | index($r) != null' "$FLEET_CONF" > /dev/null
}

valid_login() { [[ $1 =~ ^[A-Za-z0-9]([A-Za-z0-9-]{0,38})$ ]]; }

# "Apple M3 Pro" -> m3. Prints nothing for anything else.
chip_family_of() {
    sed -nE 's/^Apple (M[0-9]{1,2})( Pro| Max| Ultra)?$/\1/p' <<< "$1" | tr 'M' 'm'
}

# Normalize Apple's DMG filename to the version format accepted by results.
# For example, Game_Porting_Toolkit_4.0_beta_3.dmg becomes 4.0-beta3.
gptk_version_of() { # DMG PATH -> VERSION or unknown
    local v
    v=$(sed -nE 's/^Game_Porting_Toolkit[_ -]?v?(.+)\.dmg$/\1/p' <<< "${1##*/}" |
        sed -E 's/[_ -]beta[_ -]?/-beta/' |
        sed -nE '/^[0-9]{1,2}(\.[0-9]{1,2}){0,2}(-beta[0-9]?)?$/p')
    printf '%s\n' "${v:-unknown}"
}

# The body of a task issue. The HTML comment is what machines read.
task_body() { # COMMIT CHIP REPO
    local chip_upper
    chip_upper=$(tr 'm' 'M' <<< "$2")
    cat << EOF
<!-- fleet-task
kind: test-matrix
commit: $1
chip: $2
-->
**Test matrix: commit \`${1:0:12}\` on an Apple $chip_upper Mac** (any $chip_upper, $chip_upper Pro, $chip_upper Max or $chip_upper Ultra).

Run the native and the Wine test suites at this commit and report the counts. Only a short, strictly checked result is posted. Logs stay on your Mac.

To take this task, point your AI coding agent at [docs/fleet-join.md](https://github.com/$3/blob/main/docs/fleet-join.md), or read [docs/fleet.md](https://github.com/$3/blob/main/docs/fleet.md). Follow [docs/fleet-legal.md](https://github.com/$3/blob/main/docs/fleet-legal.md): nothing from any game is ever posted here.
EOF
}

# Reads a task issue body. Prints "COMMIT CHIP" or fails.
parse_task() {
    local block kind commit chip
    block=$(tr -d '\r' <<< "$1" | awk '/^<!-- fleet-task[ \t]*$/ { f = 1; next } f && /^-->/ { exit } f')
    kind=$(sed -nE 's/^kind: *([a-z-]+) *$/\1/p' <<< "$block" | head -n 1)
    commit=$(sed -nE 's/^commit: *([0-9a-f]{40}) *$/\1/p' <<< "$block" | head -n 1)
    chip=$(sed -nE 's/^chip: *(m[0-9]{1,2}) *$/\1/p' <<< "$block" | head -n 1)
    [ "$kind" = test-matrix ] && [ -n "$commit" ] && is_chip "$chip" || return 1
    printf '%s %s\n' "$commit" "$chip"
}

# Reads a comment. Prints "claim", "beat", "release REASON" or "result",
# or nothing when the comment is not a fleet command. Commands other than
# /result must be the whole comment.
parse_command() {
    local text first rest
    text=$(tr -d '\r' <<< "$1")
    first=$(sed -n '1p' <<< "$text" | sed -E 's/^[[:space:]]+//; s/[[:space:]]+$//')
    rest=$(sed '1d' <<< "$text" | tr -d '[:space:]')
    case "$first" in
        /claim | /beat)
            [ -z "$rest" ] && echo "${first#/}"
            ;;
        "/release "*)
            local reason=${first#/release }
            [ -z "$rest" ] && is_release_reason "$reason" && echo "release $reason"
            ;;
        /result)
            echo result
            ;;
    esac
    return 0
}

# Prints the JSON of a /result comment. Fails unless the comment is exactly
# "/result" followed by one ```json block and nothing else.
result_json() {
    tr -d '\r' <<< "$1" | awk '
        NR == 1 { if ($0 !~ /^\/result[ \t]*$/) bad = 1; next }
        state == 0 && /^```json[ \t]*$/ { state = 1; next }
        state == 1 && /^```[ \t]*$/ { state = 2; next }
        state == 1 { print; next }
        /^[ \t]*$/ { next }
        { bad = 1 }
        END { if (bad || state != 2) exit 1 }'
}

# The result schema. Every field is checked against a narrow pattern, so a
# valid result cannot carry free text, links or file contents.
FLEET_VALIDATE_JQ='
def fields: ["chip","failed_tests","gptk","hw_model","kind","macos","native","schema","setup","upstream_commit","wine"];
def str($re): type == "string" and test($re);
def count: type == "number" and . == floor and . >= 0 and . <= 100000;
def counts: . == null or (type == "object" and (keys == ["failed","ignored","passed"])
    and (.passed | count) and (.failed | count) and (.ignored | count));
if type != "object" then "result must be a JSON object"
else
  (if (keys - fields) != [] then "result has fields that are not in the schema" else empty end),
  (if (fields - keys) != [] then "result is missing fields: " + ((fields - keys) | join(", ")) else empty end),
  (if .schema != 1 then "schema must be 1" else empty end),
  (if .kind != "test-matrix" then "kind must be test-matrix" else empty end),
  (if .upstream_commit != $commit then "upstream_commit is not the task commit" else empty end),
  (if (.hw_model | str("^[A-Za-z]{2,20}[0-9]{1,3},[0-9]{1,3}$")) then empty
   else "hw_model is not a Mac model identifier" end),
  (if (.chip | str("^Apple M[0-9]{1,2}( Pro| Max| Ultra)?$")) | not then "chip is not an Apple M-series name"
   elif (.chip | ascii_downcase | capture("(?<f>m[0-9]+)").f) != $chip then "chip is not the task chip family"
   else empty end),
  (if (.macos | str("^[0-9]{2}(\\.[0-9]{1,2}){0,2}$")) then empty else "macos is not a version number" end),
  (if (.gptk | str("^([0-9]{1,2}(\\.[0-9]{1,2}){0,2}(-beta[0-9]?)?|unknown)$")) then empty
   else "gptk is not a version number or unknown" end),
  (if .setup == "ok" or .setup == "failed" then empty else "setup must be ok or failed" end),
  (if (.native | counts) then empty else "native must be null or {passed, failed, ignored}" end),
  (if (.wine | counts) then empty else "wine must be null or {passed, failed, ignored}" end),
  (if (.failed_tests | type) == "array" and (.failed_tests | length) <= 100
      and all(.failed_tests[]; str("^(native|wine)/[A-Za-z0-9_:]{1,200}$")) then empty
   else "failed_tests must be a list of up to 100 test names" end)
end'

# Prints one error per line and fails if the result is not valid for the task.
validate_result() { # JSON COMMIT CHIP
    local errors
    if [ "${#1}" -gt 8000 ]; then
        echo "result is larger than 8000 bytes"
        return 1
    fi
    if ! jq -e . > /dev/null 2>&1 <<< "$1"; then
        echo "result is not valid JSON"
        return 1
    fi
    errors=$(jq -r --arg commit "$2" --arg chip "$3" "$FLEET_VALIDATE_JQ" <<< "$1")
    if [ -n "$errors" ]; then
        printf '%s\n' "$errors"
        return 1
    fi
}

# A result counts as failing unless setup worked and both suites ran clean.
result_failing() {
    jq -e '.setup != "ok" or .native == null or .wine == null
        or .native.failed > 0 or .wine.failed > 0' > /dev/null <<< "$1"
}

# Screens free text in fleet issues. Prints "delete REASON", "hide REASON"
# or nothing. Deleting is for links that distribute files; hiding sends
# anything that might be game material to a maintainer for review.
moderate() {
    local lower
    lower=$(tr '[:upper:]' '[:lower:]' <<< "$1")
    if grep -Eq 'magnet:\?|\.torrent([^a-z]|$)|(mega\.(nz|io)|mediafire\.com|gofile\.io|pixeldrain\.com|1fichier\.com|rapidgator\.net|anonfiles|krakenfiles\.com|workupload\.com|uploadhaven\.com|send\.cm|terabox\.com|qiwi\.gg|buzzheavier\.com)' <<< "$lower"; then
        echo "delete file-sharing or torrent link"
    elif grep -Eq '[A-Za-z0-9+/=_-]{200,}' <<< "$1"; then
        echo "hide long encoded data"
    elif grep -Eq 'decrypt|(^|[^a-z])(dump(ed|ing|s)?|rip(ped)?) (of )?(the |my |a )?(game|disc|console|title)|(game|disc|title) (dump|rip)|(^|[^a-z])(xvd|xvc|msixvc|cik|odk|rpf)([^a-z]|$)|title ?key|(^|[^a-z])keys? (file|dump)|pirat|warez|cracked|user-attachments|user-images\.githubusercontent|private-user-images|screenshot' <<< "$lower"; then
        echo "hide possible game material or circumvention"
    fi
    return 0
}

# Sums the "test result:" lines of cargo test output. Prints
# "PASSED FAILED IGNORED", or nothing if no test binary ran.
parse_counts() { # LOG
    awk '/^test result: / {
            for (i = 2; i <= NF; i++) {
                if ($i ~ /^passed;?$/) p += $(i - 1)
                if ($i ~ /^failed;?$/) f += $(i - 1)
                if ($i ~ /^ignored;?$/) g += $(i - 1)
            }
            n++
        }
        END { if (n) print p + 0, f + 0, g + 0 }' "$1"
}

counts_json() { # LOG [EXIT STATUS] -> {passed, failed, ignored} or null
    local c status=${2:-0}
    c=$(parse_counts "$1")
    if [ -z "$c" ]; then
        echo null
        return
    fi
    # shellcheck disable=SC2086 # three numbers
    set -- $c
    # Passing binaries do not prove the suite finished: a later build or runner
    # can fail. Keep counted test failures, but mark other unsuccessful runs null.
    if [ "$status" -ne 0 ] && [ "$2" -eq 0 ]; then
        echo null
        return
    fi
    jq -cn --argjson p "$1" --argjson f "$2" --argjson i "$3" '{passed: $p, failed: $f, ignored: $i}'
}

failed_names() { # LOG PREFIX
    sed -nE 's/^test (.+) \.\.\. FAILED$/\1/p' "$1" | grep -E '^[A-Za-z0-9_:]{1,200}$' | sed "s|^|$2/|" || true
}

# Turns accepted results (one JSON object per line, each with "n" for the
# issue number and "closed" for when it closed) into a Markdown report: every
# result, coverage per commit, and tests that fail only on some chips.
report_markdown() {
    jq -rs --argjson chips "$(jq -c .chips "$FLEET_CONF")" '
        def fam: .chip | ascii_downcase | capture("(?<f>m[0-9]+)").f;
        def pf($c): if $c == null then "-" else "\($c.passed) / \($c.failed)" end;
        def failing: .setup != "ok" or .native == null or .wine == null or .native.failed > 0 or .wine.failed > 0;
        sort_by(.closed) as $rs
        | ($rs | group_by(.upstream_commit) | sort_by(map(.closed) | min)) as $commits
        | "# Mac test fleet report\n\n\($rs | length) accepted results across \($commits | length) commits.\n\n"
        + "## Coverage by commit\n\n| Commit | First result | Chip families | Results failing |\n|---|---|---|---|\n"
        + ($commits | map(
            (map(fam) | unique) as $f
            | "| `\(.[0].upstream_commit[0:12])` | \(map(.closed) | min | .[0:10]) | \($f | length) of \($chips | length): \($f | join(", ")) | \(map(select(failing)) | length) of \(length) |")
          | join("\n"))
        + "\n\n## Tests that fail on some chips but not others\n\n"
        + ([$commits[] | . as $g
            | ($g | map(select(.native != null and .wine != null)) | map(fam) | unique) as $ran
            | ($g | map(fam as $c | .failed_tests[] | {t: ., c: $c}) | group_by(.t)[]
               | {t: .[0].t, bad: (map(.c) | unique)})
            | select((.bad | length) < ($ran | length))
            | "- `\($g[0].upstream_commit[0:12])` `\(.t)`: fails on \(.bad | join(", ")); passes on \($ran - .bad | join(", "))"]
           | if length == 0 then "None found." else join("\n") end)
        + "\n\n## All results\n\n| Task | Closed | Commit | Chip | Model | macOS | Toolkit | Native passed / failed | Wine passed / failed |\n|---|---|---|---|---|---|---|---|---|\n"
        + ($rs | map("| #\(.n) | \(.closed[0:10]) | `\(.upstream_commit[0:12])` | \(.chip) | \(.hw_model) | \(.macos) | \(.gptk) | \(pf(.native)) | \(pf(.wine)) |") | join("\n"))'
}
