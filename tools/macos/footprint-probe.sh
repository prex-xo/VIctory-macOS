#!/bin/sh
# Runs victory-gpu's footprint probe under Wine and D3DMetal and reads
# `footprint -p` of its Wine process at each step: the rows of graphics
# memory and the whole footprint, in MB.
# Usage: tools/macos/footprint-probe.sh [test name] [out.tsv]
# The test is footprint_probe (the default) or footprint_of_sparse_heaps.
# WINE and WINEPREFIX are passed to tools/macos/test-wine.sh.
set -e
cd "$(dirname "$0")/../.."
TEST="${1:-footprint_probe}"
OUT="${2:-target/$TEST.tsv}"
DIR="$(mktemp -d /tmp/victory-footprint.XXXXXX)"
trap 'rm -rf "$DIR"' EXIT
VICTORY_FOOTPRINT_DIR="Z:$(printf '%s' "$DIR" | tr / '\\')"
export VICTORY_FOOTPRINT_DIR
tools/macos/test-wine.sh -p victory-gpu --lib "footprint_probe::$TEST" -- --exact --ignored --nocapture --test-threads=1 &
RUN=$!

# A footprint size ("7177 MB", "2160 KB", "0 B") in MB.
mb() { awk '{ v = $1; u = $2; if (u == "GB") v *= 1024; else if (u == "KB") v /= 1024; else if (u == "B") v /= 1048576; printf "%.0f", v }'; }
row() { printf '%s\n' "$FP" | grep -E "  $1\$" | head -1 | awk '{ print $1, $2 }' | mb; }

printf 'step\tgraphics_mapped_dirty\tgraphics_unmapped\tfootprint\twhat\n' > "$OUT"
n=1
while kill -0 "$RUN" 2>/dev/null; do
    if [ -f "$DIR/phase-$n" ]; then
        pid="$(pgrep -f 'victory_gpu-[0-9a-f]+\.exe' | tail -1)"
        FP="$(footprint -p "$pid" 2>/dev/null)"
        total="$(printf '%s\n' "$FP" | awk '/phys_footprint:/ { print $2, $3 }' | mb)"
        printf '%s\t%s\t%s\t%s\t%s\n' "$n" "$(row 'IOAccelerator \(graphics\)')" \
            "$(row 'Owned physical footprint \(unmapped\) \(graphics\)')" "$total" \
            "$(cat "$DIR/phase-$n")" | tee -a "$OUT"
        touch "$DIR/ack-$n"
        n=$((n + 1))
    fi
    sleep 0.2
done
wait "$RUN"
