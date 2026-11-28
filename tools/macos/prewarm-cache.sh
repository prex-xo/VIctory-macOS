#!/bin/sh
# Synthetic selective-prewarm and restart experiment. Uses an isolated Wine
# prefix and a unique D3DMetal per-executable cache name for every experiment.
# Run from a setup checkout with WINE set to its CrossOver Wine, or use the
# checkout's default. Run output and backups stay in the reported temp dir.
set -eu
cd "$(dirname "$0")/../.."
export PATH="/opt/homebrew/opt/rustup/bin:$PATH"
repo="$PWD"
target=x86_64-pc-windows-gnullvm
count="${PREWARM_SHADERS:-512}"
rounds="${PREWARM_ROUNDS:-24}"
repeats="${PREWARM_REPEATS:-3}"

wine="${WINE:-$repo/target/wine-cx/bin/wine}"
case "$wine" in
    /*) ;;
    *) wine="$PWD/$wine" ;;
esac
winebin="$(dirname "$wine")"
framework="$winebin/../lib/external/D3DMetal.framework"
[ -x "$wine" ] && [ -d "$framework" ] || {
    echo "Set WINE to the CrossOver Wine binary with D3DMetal installed." >&2
    exit 1
}

# Avoid both Wine and executable identity aliasing the user's title.
root="$(mktemp -d /tmp/victory-prewarm.XXXXXXXX)"
root="$(cd "$root" && pwd -P)"
base="victory-prewarm-${root##*.}"
name="$base.exe"
driver="$(getconf DARWIN_USER_CACHE_DIR)d3dm/$name"
[ ! -e "$driver" ] && [ ! -L "$driver" ] || exit 1
echo "Synthetic run output: $root"
echo "Synthetic driver cache: $driver"

# Select the same shader helpers for the application and test build. Run
# from the temp folder and clear custom Victory variables in the Wine child.
cli="${PREWARM_CLI:-}"
fixture="${PREWARM_FIXTURE:-}"
if [ -z "$cli" ] || [ -z "$fixture" ]; then
    DXC_LIB="$framework/Versions/A/Resources/libdxcompiler.dylib"
    if [ -z "${VICTORY_HELPER_DXC:-}" ] && [ -f "$DXC_LIB" ]; then
        sh tools/macos/dxc-d3dmetal.sh "$root/dxc-d3dmetal"
        export DXC_LIB VICTORY_HELPER_DXC="$root/dxc-d3dmetal"
    fi
    cargo build --locked --release -p victory --target "$target"
    cli="$(cargo metadata --format-version 1 --no-deps | plutil -extract target_directory raw -o - -)/$target/release/victory.exe"
    cargo test --locked --release -p victory-gpu --target "$target" --no-run \
        --message-format=json > "$root/build.jsonl"
    fixture="$(sed -nE 's/.*"executable":"([^"]*\/victory_gpu-[^"]*\.exe)".*/\1/p' "$root/build.jsonl")"
fi
[ -f "$cli" ] && [ -f "$fixture" ]
mkdir "$root/cli" "$root/fixture"
cp "$cli" "$root/cli/$name"
cp "$fixture" "$root/fixture/$name"
cli="$root/cli/$name"
fixture="$root/fixture/$name"
{
    git rev-parse HEAD
    git status --short
    sysctl hw.model hw.memsize machdep.cpu.brand_string
    sw_vers
    "$wine" --version
    plutil -extract CFBundleShortVersionString raw -o - "$framework/Resources/Info.plist"
    shasum -a 256 "$cli" "$fixture"
} > "$root/environment.txt"
cd "$root"

child_env() {
    # Keep the Mac user identity so this run uses the documented cache root.
    env -i HOME="$HOME" USER="$USER" PATH="$PATH" \
        WINEPREFIX="$root/prefix" WINEDEBUG=-all \
        'WINEDLLOVERRIDES=mscoree,mshtml=;winedbg.exe=d;d3d12,d3d12core,dxgi=b' \
        "$@"
}

windows_path() {
    printf 'Z:%s' "$(printf '%s' "$1" | tr '/' '\\')"
}

choose_cache() {
    VICTORY_CACHE_DIR="$(windows_path "$1")"
    export VICTORY_CACHE_DIR
}

cli_run() {
    child_env VICTORY_CACHE_DIR="$VICTORY_CACHE_DIR" "$wine" "$cli" "$@"
}

fixture_run() {
    child_env VICTORY_CACHE_DIR="$VICTORY_CACHE_DIR" \
        VICTORY_PREWARM_TEST_PACK="$(windows_path "$root/pack")" \
        VICTORY_PREWARM_TEST_COUNT="$count" VICTORY_PREWARM_TEST_ROUNDS="$rounds" \
        VICTORY_PREWARM_TEST_PHASE="$1" \
        "$wine" "$fixture" --ignored --exact --nocapture \
        pipeline::prewarm_cache_test::synthetic_prewarm_cache
}

size_bytes() {
    if [ -d "$1" ]; then
        find "$1" -type f -exec stat -f %z {} + | awk '{ n += $1 } END { print n+0 }'
    else
        echo 0
    fi
}

pressure() {
    vm_stat
    sysctl vm.swapusage
    memory_pressure -Q
}

child_env "$wine" wineboot -u > "$root/wineboot.txt" 2>&1
child_env "$winebin/wineserver" -w
choose_cache "$root/history"
fixture_run generate > "$root/generate.txt" 2>&1
cat "$root/generate.txt"
fixture_run run > "$root/history.txt" 2>&1
history="$root/history/pipelines/early-pipelines.txt"
selection="$root/selection.txt"
cp "$history" "$selection"
[ "$(awk 'END { print NR }' "$selection")" -eq 4 ]
printf '\n%032x\n' 1 >> "$selection" # deliberately unmatched
[ ! -d "$driver" ] || mv "$driver" "$root/history-driver"

# Empty, unreadable, and malformed selections never create a device/cache.
choose_cache "$root/validation"
: > "$root/empty.txt"
echo 'not a key' > "$root/malformed.txt"
printf '%032x\n' 1 > "$root/stale.txt"
for case_name in empty stale; do
    cli_run translate --cache --pipeline-selection "$(windows_path "$root/$case_name.txt")" \
        "$(windows_path "$root/pack")" > "$root/$case_name.txt.log" 2>&1
    grep -Eq ', 0 matched keys, ' "$root/$case_name.txt.log"
    grep -Eq '0 newly built, 0 refused' "$root/$case_name.txt.log"
    [ ! -e "$driver" ]
done
for case_name in malformed missing; do
    if cli_run translate --cache --pipeline-selection "$(windows_path "$root/$case_name.txt")" \
        "$(windows_path "$root/pack")" > "$root/$case_name.txt.log" 2>&1; then
        echo "Unexpected successful $case_name selection" >&2
        exit 1
    fi
    [ ! -e "$driver" ]
done

job=""
trap '[ -z "$job" ] || kill "$job" 2>/dev/null || true' EXIT
measure_phase() {
    phase="$1"
    i=0
    until grep -Eq "sampling $phase" "$arm/runtime.log"; do
        kill -0 "$job" 2>/dev/null || { cat "$arm/runtime.log"; return 1; }
        [ "$i" -lt 600 ] || { echo "Timed out waiting for $phase"; return 1; }
        i=$((i + 1))
        sleep 0.1
    done
    # The launcher execs Wine in place. The test prints both checkpoints
    # before waiting for the respective resume files.
    lsof -p "$job" > "$arm/$phase.files"
    grep -Eq 'D3DMetal.framework' "$arm/$phase.files"
    footprint -f bytes "$job" > "$arm/$phase.footprint"
    vmmap -summary "$job" > "$arm/$phase.vmmap"
    heap -s "$job" > "$arm/$phase.heap"
    pressure > "$arm/$phase.pressure"
    touch "$arm/signals/$phase"
}

restart_sample() {
    mkdir "$arm/signals"
    # Backgrounding a shell function inserts one wrapper PID. Exec the
    # child here so lsof and the memory tools sample the Wine process.
    (
        exec env -i HOME="$HOME" USER="$USER" PATH="$PATH" \
            WINEPREFIX="$root/prefix" WINEDEBUG=-all \
            'WINEDLLOVERRIDES=mscoree,mshtml=;winedbg.exe=d;d3d12,d3d12core,dxgi=b' \
            VICTORY_CACHE_DIR="$VICTORY_CACHE_DIR" \
            VICTORY_PREWARM_TEST_PACK="$(windows_path "$root/pack")" \
            VICTORY_PREWARM_TEST_ROUNDS="$rounds" VICTORY_PREWARM_TEST_PHASE=run \
            VICTORY_PREWARM_TEST_SIGNALS="$(windows_path "$arm/signals")" \
            "$wine" "$fixture" --ignored --exact --nocapture \
            pipeline::prewarm_cache_test::synthetic_prewarm_cache
    ) > "$arm/runtime.log" 2>&1 &
    job=$!
    measure_phase device
    measure_phase workload
    wait "$job"
    job=""
    grep -E 'device ready in|^workload:' "$arm/runtime.log"
}

prewarm() {
    cli_run translate --cache --jobs 4 "$@" "$(windows_path "$root/pack")"
}

selective() {
    prewarm --pipeline-selection "$(windows_path "$selection")"
}

shader_checksums() (
    cd "$arm/cache/shaders"
    find . -type f -exec shasum -a 256 {} + | LC_ALL=C sort
)

recover() {
    # Retire the driver's own directory, first deliberately leaving the
    # matching host metadata stale; then repeat with both layers backed up.
    cp "$arm/cache/pipelines/early-pipelines.txt" "$arm/early-before.txt"
    shader_checksums > "$arm/shaders-before.txt"
    mv "$driver" "$arm/oversized-driver"
    selective > "$arm/driver-only.log" 2>&1
    [ ! -d "$driver" ] || mv "$driver" "$arm/driver-only-new"
    mkdir "$arm/old-host"
    for file in host-pipelines.bin host-pipelines.next prebuilt-pipelines.txt; do
        source="$arm/cache/pipelines/$file"
        [ ! -f "$source" ] || mv "$source" "$arm/old-host/$file"
    done
    selective > "$arm/recovered.log" 2>&1
    grep -Eq '4 newly built, 0 refused' "$arm/recovered.log"
    cmp "$arm/early-before.txt" "$arm/cache/pipelines/early-pipelines.txt"
    shader_checksums > "$arm/shaders-after.txt"
    cmp "$arm/shaders-before.txt" "$arm/shaders-after.txt"
    fixture_run run > "$arm/recovered-restart.log" 2>&1
    selective > "$arm/recovered-prewarm-again.log" 2>&1
    grep -Eq '4 already prebuilt, 0 loaded, 0 newly built, 0 refused' "$arm/recovered-prewarm-again.log"
    grep -E 'already prebuilt|newly built' "$arm/driver-only.log" "$arm/recovered.log"
    grep -E '^workload:' "$arm/recovered-restart.log"
}

run_arm() {
    arm="$root/$1-$rep"
    mkdir "$arm"
    choose_cache "$arm/cache"
    [ ! -e "$driver" ]
    echo "$1 repetition $rep"
    pressure > "$arm/start.pressure"
    if [ "$1" = full ]; then
        prewarm --pipelines > "$arm/prewarm.log" 2>&1
        grep -Eq "$((count + 2)) newly built, 0 refused" "$arm/prewarm.log"
    else
        selective > "$arm/prewarm.log" 2>&1
        grep -Eq '5 requested keys, 4 matched keys, 1 unmatched keys' "$arm/prewarm.log"
        grep -Eq "$((count - 2)) excluded, 2 duplicates, 1 without a key; 0 already prebuilt, 0 loaded, 4 newly built, 0 refused" "$arm/prewarm.log"
    fi
    grep -E '^selection |^pipelines ' "$arm/prewarm.log"
    size_bytes "$driver" > "$arm/driver-prewarm.bytes"
    restart_sample
    size_bytes "$driver" > "$arm/driver-workload.bytes"
    [ ! -d "$driver" ] || grep -Eq "$name/shaders.cache" "$arm/device.files"
    if [ "$rep" -eq "$repeats" ]; then
        if [ "$1" = full ]; then
            recover
        else
            fixture_run fallback > "$arm/fallback.log" 2>&1
            grep -E '^compute 2 |^workload:' "$arm/fallback.log"
            grep -Eq '^compute 2 .*: 1 made, 0 loaded' "$arm/fallback.log"
        fi
    fi
    [ ! -d "$driver" ] || mv "$driver" "$arm/driver"
}

rep=1
while [ "$rep" -le "$repeats" ]; do
    if [ $((rep % 2)) -eq 1 ]; then
        run_arm full
        run_arm selective
    else
        run_arm selective
        run_arm full
    fi
    rep=$((rep + 1))
done
echo "Completed synthetic measurements in $root"
