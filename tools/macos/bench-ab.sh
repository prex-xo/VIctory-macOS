#!/bin/sh
# A/B benchmark of two switch sets at one spot of the world, interleaved
# (A B A B ...) so drift (heat, caches, background work) hits both alike.
# Each run is a full game start through tools/macos/victory.sh in benchmark
# mode: VICTORY_BENCH_WARMUP world frames are skipped, VICTORY_BENCH_EXIT_FRAMES
# frames are measured, then the game exits and writes a JSON report. Input
# comes from VICTORY_INPUT_REPLAY (by world frame), so every run does the
# same; VICTORY_KEYS still steers the menus into the world. One unmeasured
# run first warms the shader caches (with a VICTORY_PREWARM set, prewarmed).
#
#   tools/macos/bench-ab.sh --exe <game executable> \
#       --a "VICTORY_TWIN_UAV_READS=0" --b "VICTORY_TWIN_UAV_READS=1" \
#       [--runs 4] [--frames 300] [--warmup 120] [--replay <jsonl>] [--out <dir>]
#
# Then: cargo xtask bench-compare <out dir>
set -eu
cd "$(dirname "$0")/../.."
root="$PWD"
exe="" a="" b="" runs=4 frames=300 warmup=120
replay="$root/tools/macos/bench/stand-still.json"
out="$root/target/bench/$(date +%Y%m%d-%H%M%S)"
while [ $# -gt 0 ]; do
    case "$1" in
        --exe) exe="$2"; shift 2 ;;
        --a) a="$2"; shift 2 ;;
        --b) b="$2"; shift 2 ;;
        --runs) runs="$2"; shift 2 ;;
        --frames) frames="$2"; shift 2 ;;
        --warmup) warmup="$2"; shift 2 ;;
        --replay) replay="$2"; shift 2 ;;
        --out) out="$2"; shift 2 ;;
        *) echo "unknown argument $1" >&2; exit 2 ;;
    esac
done
[ -n "$exe" ] || { echo "--exe <game executable> is needed" >&2; exit 2; }
[ -f "$replay" ] || { echo "no replay script $replay" >&2; exit 2; }
mkdir -p "$out"
echo "A: $a" > "$out/sets.txt"
echo "B: $b" >> "$out/sets.txt"

# One run: $1 label, $2 switch set (space-separated NAME=value words).
run() {
    report="$out/$1.json"
    echo "== run $1 ($2)"
    # shellcheck disable=SC2086 # the set is words on purpose
    env $2 \
        VICTORY_BENCH_OUT="$report" \
        VICTORY_BENCH_EXIT_FRAMES="$frames" \
        VICTORY_BENCH_WARMUP="$warmup" \
        VICTORY_INPUT_REPLAY="$replay" \
        tools/macos/victory.sh run --execute "$exe" > "$out/$1.log" 2>&1 || true
    [ -f "$report" ] || echo "   no report: see $out/$1.log" >&2
}

run warm "$a"
i=1
while [ "$i" -le "$runs" ]; do
    run "A$i" "$a"
    run "B$i" "$b"
    i=$((i + 1))
done
echo "reports in $out"
cargo xtask bench-compare "$out"
