#!/bin/sh
# Run the standard-library Python probe below; all fixtures are synthetic.
exec python3 - "$@" <<'VICTORY_PERF_PY'
"""Compare XMA packet-field reads (200,000 15-bit fields per batch) on macOS.
Prints raw thread-CPU samples.
No game assets or GPU runtime are used. Requires git, Python 3 and rustc.
"""
import argparse
import json
import platform
import statistics
import subprocess
import sys
import tempfile
from pathlib import Path

SOURCE = 'crates/victory-formats/src/xma.rs'
HELPER = 'pub fn probe() -> u64 { let data=vec![0x5a;8192];let mut out=0u64;for i in 0..200_000 {let mut c=Cursor{data:&data,pos:HEADER_BITS+((i*17)%10000)};out+=u64::from(std::hint::black_box(c.bits(std::hint::black_box(15))).unwrap_or(0));}out}'
CONTROL = False
CLOCK = r"""#[repr(C)] struct Ts { sec: std::ffi::c_long, ns: std::ffi::c_long }
unsafe extern "C" { fn clock_gettime(c: u32, t: *mut Ts) -> i32; }
fn cpu() -> u64 { let mut t=Ts{sec:0,ns:0}; assert_eq!(unsafe{clock_gettime(16,&mut t)},0); t.sec as u64*1_000_000_000+t.ns as u64 }
"""

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline", required=True)
    parser.add_argument("--patch", default="HEAD", help="commit or 'working'")
    args = parser.parse_args()
    if sys.platform != "darwin":
        parser.error("This probe uses Darwin CLOCK_THREAD_CPUTIME_ID")
    root = Path(subprocess.check_output(["git", "rev-parse", "--show-toplevel"], text=True).strip())
    def read(ref, path):
        if ref == "working":
            return (root / path).read_bytes()
        return subprocess.check_output(["git", "show", f"{ref}:{path}"], cwd=root)
    with tempfile.TemporaryDirectory(prefix="victory-perf-") as directory:
        temp = Path(directory)
        for kind, ref in [("base", args.baseline), ("patch", args.patch)]:
            source = read(ref, SOURCE).decode()
            if SOURCE.endswith("xma.rs"):
                table = temp / f"{kind}-tables.rs"
                table.write_bytes(read(ref, "crates/victory-formats/src/xma/tables.rs"))
                source = source.replace("mod tables;", f'#[path="{table}"] mod tables;')
            if SOURCE.endswith("bcpack.rs"):
                table = temp / f"{kind}-tables.bin"
                table.write_bytes(read(ref, "crates/victory-formats/src/bcpack-tables.bin"))
                source = source.replace('include_bytes!("bcpack-tables.bin")', f'include_bytes!("{table}")')
            (temp / f"{kind}.rs").write_text(source + "\n" + HELPER)
        cases = '[ ("probe",base::probe as fn()->u64,patch::probe as fn()->u64) ]'
        if CONTROL:
            cases = '[ ("probe",base::probe as fn()->u64,patch::probe as fn()->u64),("control",base::control as fn()->u64,patch::control as fn()->u64) ]'
        main_source = f'#[path="{temp}/base.rs"] mod base; #[path="{temp}/patch.rs"] mod patch;' + CLOCK
        main_source += 'fn main(){for (name,a,b) in ' + cases + r""" {for i in 0..31 {
            let (x,y)=if i%2==0 { let t=cpu();let x=a();let xt=cpu()-t;let t=cpu();let y=b();((xt,x),(cpu()-t,y)) }
            else {let t=cpu();let y=b();let yt=cpu()-t;let t=cpu();let x=a();((cpu()-t,x),(yt,y))};
            assert_eq!(x.1,y.1);println!("{} {} {} {}",name,x.0,y.0,x.1);
        }}}"""
        (temp / "main.rs").write_text(main_source)
        subprocess.run(["rustc", "--edition=2024", "-O", "-A", "unused", str(temp / "main.rs"), "-o", str(temp / "bench")], check=True)
        raw = subprocess.check_output([str(temp / "bench")], text=True)
    result = {"baseline": args.baseline, "patch": args.patch, "platform": platform.platform(), "rustc": subprocess.check_output(["rustc", "--version"], text=True).strip(), "units": "thread CPU nanoseconds per probe batch", "warmup_pairs": 3, "samples": {}}
    for case in ["probe", "control"] if CONTROL else ["probe"]:
        rows = [list(map(int, line.split()[1:])) for line in raw.splitlines() if line.startswith(case + " ")][3:]
        result["samples"][case] = {"pairs": rows, "baseline_median_ns": statistics.median(r[0] for r in rows), "patch_median_ns": statistics.median(r[1] for r in rows), "median_paired_ratio": statistics.median(r[1]/r[0] for r in rows)}
    print(json.dumps(result, indent=2))

if __name__ == "__main__":
    main()

VICTORY_PERF_PY
