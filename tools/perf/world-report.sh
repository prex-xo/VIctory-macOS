#!/bin/sh
# Medians of VIctory logs' world frame-rate lines: tools/perf/world-report.sh <log>...
exec python3 - "$@" <<'VICTORY_PERF_PY'
"""For each log: the frame, preparing, GPU and copy-back milliseconds and the
draws of the frame-rate lines that count as the world (at least MIN draws a
frame, default 1000; set MIN=300 for runs that leave work out), from 60 s
after the first line with 30% of MIN, and the median GPU time per draw.
Single runs vary with the scene, so compare draws as well as frame times
(docs/m1-max.md). Reads only the logs it is given.
"""
import os
import re
import statistics
import sys
from datetime import datetime

MIN = float(os.environ.get("MIN", "1000"))
FRAME = re.compile(
    r"([0-9.]+) ms/frame \(([0-9.]+) preparing.*?; ([0-9.]+) on the GPU; "
    r"([0-9.]+) copying back.*?\), ([0-9]+) draws/frame"
)
ANSI = re.compile(r"\x1b\[[0-9;]*m")


def rows_of(path):
    rows, world = [], None
    with open(path, errors="replace") as f:
        for line in f:
            line = ANSI.sub("", line)
            if " fps, " not in line or "ms/frame" not in line:
                continue
            m = FRAME.search(line)
            if not m:
                continue
            try:
                at = datetime.strptime(line[11:19], "%H:%M:%S")
            except ValueError:
                continue
            v = [float(x) for x in m.groups()]
            if world is None and v[4] >= MIN * 0.3:
                world = at
            if world and (at - world).total_seconds() >= 60 and v[4] >= MIN:
                rows.append(v)
    return rows


paths = sys.argv[1:]
if not paths:
    sys.exit("usage: tools/perf/world-report.sh <log>...")
for path in paths:
    name = os.path.basename(path).rsplit(".", 1)[0]
    rows = rows_of(path)
    if not rows:
        print(f"{name}: no world frame-rate lines")
        continue
    med = [statistics.median(c) for c in zip(*rows)]
    per_draw = statistics.median(r[2] / r[4] for r in rows) * 1000
    print(
        f"{name:14s} n={len(rows):3d} frame {med[0]:5.0f} prep {med[1]:4.0f} "
        f"gpu {med[2]:4.0f} back {med[3]:3.0f} draws {med[4]:5.0f}  gpu/draw {per_draw:4.0f} us"
    )
VICTORY_PERF_PY
