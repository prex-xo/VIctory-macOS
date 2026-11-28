#!/bin/sh
exec python3 - "$@" <<'VICTORY_CACHE_PERF_PY'
"""Compare pure cache-path construction on macOS (200,000 paths per batch).
Requires git, Cargo, rustc and Python 3. No cache entries or game data are read.
Timing uses the ordinary allocator; allocation counts use a separate executable.
"""
import argparse
import json
import platform
import statistics
import subprocess
import sys
import tempfile
from pathlib import Path

HELPER = 'pub fn probe()->u64 {let c=Cache{root:PathBuf::from("/tmp/victory-cache/shaders/a-long-cache-component")};let mut sum=0u64;for i in 0..200_000 {let key=std::hint::black_box(CacheKey(blake3::Hash::from_bytes([i as u8;32])));let path=c.path_for(std::hint::black_box(&key));let bytes=path.as_os_str().as_encoded_bytes();sum+=bytes.len() as u64+u64::from(bytes[bytes.len()-1]);std::hint::black_box(path);}sum}'
COUNTER = '\n#[cfg(count_allocs)] static ACTIVE:std::sync::atomic::AtomicBool=std::sync::atomic::AtomicBool::new(false);\n#[cfg(count_allocs)] static CALLS:std::sync::atomic::AtomicUsize=std::sync::atomic::AtomicUsize::new(0);\n#[cfg(count_allocs)] struct Counting;\n#[cfg(count_allocs)] fn record(){if ACTIVE.load(std::sync::atomic::Ordering::Relaxed){CALLS.fetch_add(1,std::sync::atomic::Ordering::Relaxed);}}\n#[cfg(count_allocs)] unsafe impl std::alloc::GlobalAlloc for Counting {\n unsafe fn alloc(&self,l:std::alloc::Layout)->*mut u8 {record();unsafe{std::alloc::System.alloc(l)}}\n unsafe fn alloc_zeroed(&self,l:std::alloc::Layout)->*mut u8 {record();unsafe{std::alloc::System.alloc_zeroed(l)}}\n unsafe fn realloc(&self,p:*mut u8,l:std::alloc::Layout,n:usize)->*mut u8 {record();unsafe{std::alloc::System.realloc(p,l,n)}}\n unsafe fn dealloc(&self,p:*mut u8,l:std::alloc::Layout){unsafe{std::alloc::System.dealloc(p,l)}}\n}\n#[cfg(count_allocs)] #[global_allocator]static ALLOC:Counting=Counting;\n#[cfg(count_allocs)] fn main(){for(name,f)in[("base",base::probe as fn()->u64),("patch",patch::probe as fn()->u64)]{CALLS.store(0,std::sync::atomic::Ordering::Relaxed);ACTIVE.store(true,std::sync::atomic::Ordering::Relaxed);let sum=f();ACTIVE.store(false,std::sync::atomic::Ordering::Relaxed);println!("{} {} {}",name,CALLS.load(std::sync::atomic::Ordering::Relaxed),sum);}}\n'
CLOCK = '#[cfg(target_os = "macos")]\nfn cpu_ns() -> u64 {\n    #[repr(C)]\n    struct Timespec { sec: std::ffi::c_long, ns: std::ffi::c_long }\n    unsafe extern "C" { fn clock_gettime(clock: u32, out: *mut Timespec) -> i32; }\n    let mut t = Timespec { sec: 0, ns: 0 };\n    // Darwin CLOCK_THREAD_CPUTIME_ID, from the SDK\'s _time.h.\n    assert_eq!(unsafe { clock_gettime(16, &mut t) }, 0);\n    t.sec as u64 * 1_000_000_000 + t.ns as u64\n}\n#[cfg(not(target_os = "macos"))]\nfn cpu_ns() -> u64 { 0 }\n'
SOURCE = "crates/victory-pipeline/src/cache.rs"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline", required=True)
    parser.add_argument("--patch", default="HEAD", help="commit or working")
    args = parser.parse_args()
    if sys.platform != "darwin":
        parser.error("The clock uses Darwin CLOCK_THREAD_CPUTIME_ID")
    root = Path(subprocess.check_output(["git", "rev-parse", "--show-toplevel"], text=True).strip())
    def read(ref):
        if ref == "working":
            return (root / SOURCE).read_text()
        return subprocess.check_output(["git", "show", f"{ref}:{SOURCE}"], cwd=root, text=True)
    artifacts = subprocess.check_output(["cargo", "build", "--locked", "--release", "-p", "victory-pipeline", "--jobs", "2", "--target-dir", str(root / "target/perf-cache-path"), "--message-format=json"], cwd=root, text=True)
    extern = {}
    links = set()
    for line in artifacts.splitlines():
        try:
            item = json.loads(line)
        except ValueError:
            continue
        if item.get("reason") == "compiler-artifact" and item["target"]["name"] in ["victory_pipeline", "victory_core", "anyhow", "tracing", "blake3"]:
            files = [f for f in item["filenames"] if f.endswith(".rlib")]
            file = next((f for f in files if "/deps/" in f), files[0])
            extern[item["target"]["name"]] = file
            links.add("dependency=" + str(Path(file).parent))
        if item.get("reason") == "build-script-executed":
            links.update(item.get("linked_paths", []))
    assert len(extern) == 5, extern
    link_args = []
    for name, file in extern.items():
        link_args += ["--extern", name + "=" + file]
    for link in sorted(links):
        link_args += ["-L", link]
    with tempfile.TemporaryDirectory(prefix="victory-cache-path-") as directory:
        temp = Path(directory)
        for kind, ref in [("base", args.baseline), ("patch", args.patch)]:
            (temp / f"{kind}.rs").write_text(read(ref) + "\n" + HELPER)
        source = 'pub use victory_pipeline::Converter;' + f'#[path="{temp}/base.rs"]mod base;#[path="{temp}/patch.rs"]mod patch;' + CLOCK + COUNTER
        source += r"""#[cfg(not(count_allocs))] fn main(){for i in 0..31 {
            let(a,b)=if i%2==0 {let t=cpu_ns();let a=base::probe();let at=cpu_ns()-t;let t=cpu_ns();let b=patch::probe();((at,a),(cpu_ns()-t,b))}
            else {let t=cpu_ns();let b=patch::probe();let bt=cpu_ns()-t;let t=cpu_ns();let a=base::probe();((cpu_ns()-t,a),(bt,b))};
            assert_eq!(a.1,b.1);println!("{} {} {}",a.0,b.0,a.1);
        }}"""
        (temp / "main.rs").write_text(source)
        outputs = {}
        for kind, flags in [("time", []), ("alloc", ["--cfg", "count_allocs"])]:
            binary = temp / kind
            subprocess.run(["rustc", "--edition=2024", "-O", "-A", "unused", str(temp / "main.rs"), "-o", str(binary), *link_args, *flags], cwd=root, check=True)
            outputs[kind] = subprocess.check_output([str(binary)], text=True)
    pairs = [list(map(int, line.split())) for line in outputs["time"].splitlines()][3:]
    allocation_rows = [line.split() for line in outputs["alloc"].splitlines()]
    assert allocation_rows[0][2] == allocation_rows[1][2]
    result = {"baseline": args.baseline, "patch": args.patch, "platform": platform.platform(), "rustc": subprocess.check_output(["rustc", "--version"], cwd=root, text=True).strip(), "units": "thread CPU nanoseconds per 200000 path constructions; includes fixture setup and teardown", "warmup_pairs": 3, "samples": {"probe": {"pairs": pairs, "baseline_median_ns": statistics.median(r[0] for r in pairs), "patch_median_ns": statistics.median(r[1] for r in pairs), "median_paired_ratio": statistics.median(r[1]/r[0] for r in pairs)}}, "allocation_calls": {r[0]: int(r[1]) for r in allocation_rows}, "allocation_note": "Separate instrumented executable counts alloc, alloc_zeroed and realloc calls; timing executable has no counting hooks. Each batch includes one root-path allocation."}
    print(json.dumps(result, indent=2))

if __name__ == "__main__":
    main()

VICTORY_CACHE_PERF_PY
