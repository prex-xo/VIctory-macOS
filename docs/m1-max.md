# Grand Theft Auto VI on an M1 Max, 27 November 2026

What one tester measured in a day of runs on an M1 Max: what kept the game from starting, what the Wine setup cost, and where a world frame's time goes. The numbers come from 60 unattended benchmark runs and about a dozen hands-on runs. Every benchmark run is in [`tools/perf/results/m1max-world-20261127.json`](../tools/perf/results/m1max-world-20261127.json), with its build, its switches and its report.

The Mac: a MacBook Pro (MacBookPro18,2) with an M1 Max and 64 GB, on macOS 26.6.2, with High Power Mode on. The game ran in a CrossOver 26.3 bottle (Series X profile unless noted), with `VICTORY_FAULT_TRACKING=1` and `VICTORY_PATCH_MASKED=all`.

Where it ended up: standing still in the world with ray tracing off, the frame takes a median of 670 ms (about 1.5 fps), 95th percentile 697 ms. About 490 ms of that is the GPU, 150 ms is preparing host work on the CPU and 25 ms is copying back. In the morning it had been 1.5 to 1.7 s a frame. The switches in main that this setup uses are collected in `tools/macos/sets/m1max.txt` (`VICTORY_SET=m1max`).

## How the runs were measured

A benchmark run needs no one at the Mac:

- `VICTORY_KEYS='2-150/1:Enter'` presses Enter once a second for the first 150 s, which passes the title's "press a button" screen and the menus. Without it a run waits at the title.
- `VICTORY_INPUT_REPLAY=<file>` with one range line and every stick at 0 (`{"from":0,"to":1000000,"lx":0.0,"ly":0.0,"rx":0.0,"ry":0.0,"lt":0.0,"rt":0.0}`) keeps the character standing still, so each run sees the same scene.
- `VICTORY_BENCH_WARMUP=60 VICTORY_BENCH_EXIT_FRAMES=240 VICTORY_BENCH_OUT=<file>.json` measures 240 world frames after 60 warm-up frames, writes the median, p95, p99, 1% low, draws and black frames, and exits cleanly.

A run reaches the world in about 50 s and takes about 4.5 minutes in all. Scene content still differs from run to run, so single runs vary 15% or more: compare draws as well as frame times. Over the morning's runs, frame time fitted `ms = 390 + 0.075 * draws` with a residual standard deviation of 52 ms (D3DMetal 3.0, ray tracing off). Comparing a run's residual against that line separates a switch's effect from a busier scene. `tools/perf/world-report.sh <log>...` prints the medians of a log's world frame-rate lines (frame, preparing, GPU, copy-back, draws).

A build whose shader translator changed translates every shader again on its first run. On this Mac that is about ten minutes of cold frames (11 s each for the first two world minutes), so measure on the second run. Runs labelled `-cold` in the results file are such first runs.

## Starting the game

### Builds and iCloud Drive alongside the game stall its start-up

From 04:18 most launches stalled, with any build, including one that had reached the world four times out of four. They stopped before sign-in (no `XUserAddAsync` after `input device callback kind=0x10`), or later on the loading screen with the streaming counts frozen and DirectStorage idle. A frozen start-up never recovered.

Each stretch of stalls matched a `cargo build --release` running on the same Mac. A launch with the build paused (`kill -STOP` on cargo and its rustc children), otherwise the same, signed in 0.8 s after the input step and reached the world in 4.5 minutes. Later iCloud Drive did the same: `~/Documents` was an iCloud Drive folder, and `bird` and `fileproviderd` spent about two cores syncing 11 GB of build output written there. Three launches with them busy never signed in or took four minutes. The next launch with both paused signed in one second after the input step.

So contention on the host exposes a lost wakeup or race somewhere in VIctory or Wine, which is not found yet. The stalled title thread waits in `+0x1e10090` until a streaming module's pending counters reach 0 (module `0xFFFF` means all of them), called from `+0x1c82c0f`, the engine's loop that loads every requested object by polling with `Sleep` and `SetEvent`. The counters are in the engine object that the getter at `+0x4d9eda0` returns, from offset `0x13a80`: 49 modules of `0xC0` bytes, each with five 64-bit counters `0x18` apart. `VICTORY_STREAM_WATCH=<seconds>` logs the modules with requests left. The fifth counter does not follow the pipeline backlog.

Before a playtest, check that no build runs (`pgrep -f 'cargo build'`) and that iCloud Drive is quiet (`ps -o pcpu= -p $(pgrep -x bird)`). Keep build output out of iCloud Drive folders (iCloud skips a folder whose name ends in `.nosync`).

Ruled out along the way: save data and the telemetry cache, leftover Wine processes, power and battery, screen lock, window focus, `coreaudiod`, MTLCompilerService crashes, iCloud-evicted game files, host pipeline creation, the DirectStorage queues (no stuck warnings), Wine's audio path (`VICTORY_SILENT_AUDIO=1`), Wine's thread-stack scan (`ntdll-skip-host-mappings.patch`), the saved pipeline library, fault tracking (`VICTORY_FAULT_TRACKING=0` stalled too) and failing `ReadFile` retries.

### DirectStorage reads held for a Submit

DirectStorage reads that the title held for a Submit, released later by the watch thread, never woke the workers, so loading stopped. With that fixed (in main as #614), keeping `early-pipelines.txt` no longer froze start-up (it had frozen 11 times out of 11), and pipelines prebuild while the game loads.

### 256-bit stores in the title's DLLs

With fault tracking on, the game froze in the world on Rosetta's `EmulateForward` at a 256-bit store in `xboxbvhbuilder.dll` (`+0x14b2f`) when only the executable's wide-store list was given. Series 1.0.2.1's ray-tracing structure builder DLL has 122 such stores. With a list for every mapped image (`cargo xtask wide-stores` over the cache's images, read through `VICTORY_PATCH_WIDE_DIR`), it no longer froze there.

## The Wine setup

### CrossOver 26.3 lacks the view-tree wineserver

The title maps and unmaps 64 KiB views thousands of times a frame. CrossOver 26.3 as shipped does not have `tools/macos/patches/wineserver-view-tree.patch` (#155, #188), so each such request costs about 1 ms. In the world `wineserver` sat at 54 to 97% CPU, 69% of it in one request handler's list walk.

A copy of the app with only the patched `wineserver` swapped in reached the input step about 11 s after launch instead of about 37 s, and signed in 0.7 s later. `wineserver` used 1 to 10% CPU while loading instead of 75 to 90%. The copy was an APFS clone of the app, with only the outer bundle signed again (`codesign --force --preserve-metadata=entitlements,flags,runtime -s -`, without `--deep`).

Keep CodeWeavers' own `ntdll.so` in such a copy. With an `ntdll.so` built from the CrossOver 26.3 sources in its place, D3DMetal logged "Cannot load DXIL conversion library" and refused every host pipeline: 1,517 refused, 0 draws and a black screen.

Replacing files inside the signed `CrossOver.app` itself breaks its signature, and macOS then offers to move it to the Bin. A copy made with `ditto --noextattr --noqtn` and signed again launches without a prompt.

### Turn on msync

The bottle did not set `WINEMSYNC=1` (CrossOver's MSync toggle), and a launcher that starts Wine with `env -i` has to pass it itself. With it, Wine's sync objects are Mach semaphores instead of `wineserver` requests: `wineserver` dropped to 7 to 17% CPU in the world, and CPU preparation fell from 510 to 323 ms a frame. The frame stayed at about 1.6 s at first, because the world then streamed in more and the GPU had more to draw.

With both the view-tree `wineserver` and msync, CPU preparation fell to about 110 ms and the world streamed in fully, at 4,000 to 4,700 draws a frame. Frames then took about 1.4 s, 1.15 to 1.25 s of it on the GPU (91% busy). Later builds took that down to the figures below.

### D3DMetal's shader cache is per executable name

D3DMetal keys its compiled-shader cache by the executable's name (`$(getconf DARWIN_USER_CACHE_DIR)d3dm/<exe name>/shaders.cache`, about 500 MB here, kept across reboots). A test harness that renamed the executable for every run compiled about 1,600 pipelines cold, about 3 s each. Run every build under one fixed name.

### D3DMetal 4.0 beta 3

On one build, D3DMetal 4.0 beta 3 drew more for less GPU time than CrossOver 26.3's D3DMetal 3.0: 4,300 to 5,000 draws a frame against 3,800 to 3,900, in 518 to 526 ms of GPU against 546 to 565 ms, at about the same frame median (698 ms against 686 to 725 ms; runs `b-d3dm4-1`, `b-d3dm4-2` and `b-base1` to `b-base3`). A later build on 4.0 beta 3 ran at a 694 ms median and 822 ms 95th percentile, where the build before it on 3.0 had 712 and 1,004 ms (`d4-base`, `x-base`), with no black frames. D3DMetal 3.0 reports `enhanced_barriers=false`, so `VICTORY_CLAMP_SYNC` (#588) does nothing there. This agrees with "Use D3DMetal 4.0 beta 3" in `macos.md`.

### Other things seen

- Each launch leaks a Wine audio client every time the title creates its spatial audio again.
- `ReportCrash` uses a whole core while MTLCompilerService crashes during pipeline builds.
- Two games at once froze the whole Mac (GPU memory), as #554 found on a 32 GB Mac.
- `sample <pid> 4` works on the game and on `wineserver`, but cannot unwind through Wine's boundary between PE and Unix code, so VIctory's and the title's frames do not show. Count threads by their last named frame instead. For how busy the GPU is: `ioreg -r -d 1 -c IOAccelerator | grep 'Device Utilization'` (61 to 92% in the world here).

## Where a world frame goes

The best build at 14:45 was main at d1609d9 with #584, #587 and #588, the M1 Max branch's own commits, D3DMetal 4.0 beta 3 and racy 64-bit min/max in compute (below). Medians of its 5-second reports: frame 670 ms, preparing 150 ms, GPU 490 ms, copy-back 25 ms, about 4,970 draws (runs `p-sync` and `p-sync2`).

### The title's commands

Each world frame the title issues about 4,180 mesh draws (3,776 of them indirect), 770 vertex draws, 5,100 compute dispatches (4,135 direct) and 740 enhanced `Barrier` calls. The barriers are mostly `SYNC_ALL_SHADING` on both sides.

Leaving out a whole kind of command (to measure only: the picture breaks) gave this GPU time per frame:

| Left out | GPU ms a frame | Run |
| --- | ---: | --- |
| nothing | 562 | `x-base` |
| compute dispatches | 197 | `k-disp` |
| mesh draws | 264 | `k-mesh` |
| vertex draws | 464 (the scene changed) | `k-vert` |

Compute is the largest share. The M1 has no dynamic caching, no hardware mesh shading and no 64-bit atomics. On this content it is about nine times slower on the GPU than the M5 Pro's 61 ms in #577.

### The costly compute programs

Timing each command in a host list of its own (`VICTORY_GPU_TIMING` in a diagnostics build) named three programs. Leaving each one out, by the program name in its shader container, gave what it costs a normal frame. Isolated timing overstates large passes, because each isolated list also carries the copies that its first use triggers, so trust the leave-out figures:

| Program | What it is | Leaving it out saved | Notes |
| --- | --- | ---: | --- |
| `CS_VisHair` (`b6df66af105d8b2d`) | hair visibility, rasterized in compute | ~158 ms GPU (517 to 359, `x-hair`) | wave64; 13 to 92 small dispatches a frame (1 to 12 groups of 64); a uniform per-pixel loop with one 64-bit `image_atomic_umax` per iteration, so lanes on one strand meet on the emulated lock |
| `CS_SpatialFilter_Std` (`63381c3250b287fa`) | spatial filter | ~50 ms (`x-filt`) | wave64, loops; about one dispatch a frame |
| `CS_DepthProcessingHBO` (`2f03a2df10503cef`) | ambient occlusion depth processing | 15 to 20 ms (`x-hbo`) | wave64, 8x8 groups, 1,255 instructions with no loops and 15 `ds_swizzle`; 115 to 170 ms per dispatch when timed alone, far more than its size suggests |

Leaving all three out ruined the picture (smeared and blown out), so it is no fast mode. The shadow cascades (`MS_VS_CascadeShadows_*`, about 630 depth-only mesh draws a frame) and pedestrian skinning (`CS_PedStandardSkinning`) come next in the timing.

`VICTORY_ATOMIC_LOCKS=0` looked like it saved about 160 ms, but that came from programs not running. Without the locks D3DMetal reports no 64-bit typed atomics, so every pipeline that uses them is refused ("no host pipeline: the host GPU has no 64-bit atomics on typed heap resources"): about 65 alpha pre-pass draws a frame (characters, hair, fences, decals; up to 89 instances a draw) and `CS_VisHair`. Chain-link fences vanished and hair lost strands.

So the lock mechanics themselves are not what costs. A lock early-out for a min/max the texel already covers gained about 1%. 2^20 lock words, `VICTORY_ATOMIC_UMAX=1`, 16 lock rounds instead of 4, and pixel programs' 64-bit atomics through rasterizer-ordered views gained nothing. Dropping the lock and its fences for compute programs' 64-bit min and max saved about 29 ms of GPU a frame (`y-racy`, `p-sync`), with no difference visible in side-by-side captures. Hair and fences were the things to watch.

### Preparing host work

Per frame, from a diagnostics build's copy statistics: about 300 ms of real list work on the GPU, 52 ms setting watches (31.8 ms of it in about 3,700 `VirtualProtect` calls, one per 64 KiB page, which are aliased and cannot be coalesced), 53 ms in the guard fault handler (764 faults, about 70 µs each under Rosetta and Wine), 26 ms of moves and 25 ms of copy-back. 951 enhanced barrier calls became 951 global UAV barriers. D3DMetal's CPU encoding is not a bottleneck (sampled).

Of the CPU preparation's wall time, syncing the title's descriptor heaps ("heap slots the title copied") took 71 to 85 ms for about 260,000 to 280,000 slots, about 60% of it. Refresh batches took 23 to 28 ms, heap texture uploads about 13.5 ms and preludes about 3 ms. The cost is the views made again in D3DMetal, which are serialized on the heap's lock, not the scan: syncing pages on four workers saved only about 11 ms. `VICTORY_STALE_LEFT=1` gained nothing.

Each frame is about ten host submissions, each fully serialized: wait for the one before, prepare, launch. So preparation and copy-back (about 150 ms) never overlap the GPU. `VICTORY_PREP_AHEAD=1` (#584) gained nothing here, because about 5.4 of the 5.6 flights a frame write something back and are not quiet at landing. Overlapping them needs a lazy copy-back.

Leaving pages that the title writes every frame unguarded ("hot pages") was worse in the world: copies grew from about 250 MB to 1 GB a frame, and frames from 0.45 s to between 2.6 and 5.3 s, because every view's refresh copied the unguarded page again. It would need per-page copy deduplication first.

### Switches that did not help here

Measured against the frame-time line or a matched baseline, on D3DMetal 3.0 unless noted. A positive figure means slower.

| Switch | Result |
| --- | --- |
| barrier calls with no title work since the list's last one left out | +3.7% |
| `VICTORY_RENDER_SCALE=0.5` | -3.3% |
| `VICTORY_ATOMIC_UMAX=1` | +1% |
| `VICTORY_JOIN_LISTS=256` | +1.2% (still about 10 host submissions a frame) |
| `VICTORY_SHADER_LOOP_LIMIT=4096` | no gain |
| `ROSETTA_ADVERTISE_AVX=1` | +1.4%, against +2.9% for a baseline on battery |
| `VICTORY_CLAMP_SYNC=1` with title barriers kept to their stages (D3DMetal 4.0 beta 3) | GPU 528 ms against 512 ms |
| `VICTORY_PREP_AHEAD=1` (#584) | no gain (flights not quiet) |
| `VICTORY_RAY_SKIP=1` | no gain |
| `VICTORY_LINK_MESH_OUTPUTS=all` | no gain (cold run) |
| the title's `settings.xml` graphics options | ignored by the console build: the same draws |
| Series S profile | about 3% less CPU only; before #615 it needed `VICTORY_UPSCALER=6` or it divided by zero at `+0x28f03e8` |

The one large gain was holding the ray-tracing structure builder back with the ambient light in its place (`VICTORY_AMBIENT=1`): 716 ms against 820 ms a frame on the same build. Main now holds the builder back unless `VICTORY_BVH_BUILDER=1`.

About a fifth of the bench reports' world frames are counted as having no draws. They are not empty: under asynchronous submission each frame submits the same 17 submissions (about 250 segments), and the next frame then counts up to twice the draws.

## What is left

The two remaining large levers are structural. The title's asynchronous compute could run on a second host queue, where VIctory now serializes every title queue into one stream of flights. CPU preparation could overlap the GPU, which needs copy-back to be lazy. On the GPU, `CS_VisHair`'s loop of 64-bit atomics is the single most expensive thing in the frame on an M1, and a faster exact 64-bit max is tracked in #240.
