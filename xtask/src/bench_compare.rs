//! `cargo xtask bench-compare <dir>`: compares the A and B runs of
//! `tools/macos/bench-ab.sh`.
//!
//! Reads `<dir>/A<n>.json` and `<dir>/B<n>.json` (`VICTORY_BENCH_OUT`
//! reports), prints each run's median frame time, each side's median, the
//! difference B - A and a bootstrap 95 % interval of it. An interval that
//! holds 0 means the runs cannot tell A from B.

use std::path::Path;

use anyhow::{Context, Result};

/// What a report gives the comparison.
#[derive(Debug, Clone, PartialEq)]
struct Run {
    name: String,
    median: f64,
    p95: f64,
    frames: u64,
}

/// The number after `"key":` at or after `from` in `text`.
fn number_after(text: &str, from: usize, key: &str) -> Option<(f64, usize)> {
    let needle = format!("\"{key}\":");
    let at = from + text[from..].find(&needle)? + needle.len();
    let rest = text[at..].trim_start();
    let end = rest
        .find(|c: char| !(c.is_ascii_digit() || matches!(c, '.' | '-' | '+' | 'e' | 'E')))
        .unwrap_or(rest.len());
    Some((rest[..end].parse().ok()?, at))
}

/// A report's run, `None` if the device was removed (left out).
fn parse(name: &str, text: &str) -> Option<Run> {
    if text.contains("\"device_removed\": true") || text.contains("\"device_removed\":true") {
        return None;
    }
    let frames = number_after(text, 0, "world_frames")?.0 as u64;
    let ms = text.find("\"frame_ms\"")?;
    let median = number_after(text, ms, "median")?.0;
    let p95 = number_after(text, ms, "p95")?.0;
    Some(Run {
        name: name.to_owned(),
        median,
        p95,
        frames,
    })
}

fn median(v: &[f64]) -> f64 {
    let mut s = v.to_vec();
    s.sort_by(f64::total_cmp);
    let n = s.len();
    if n % 2 == 1 {
        s[n / 2]
    } else {
        (s[n / 2 - 1] + s[n / 2]) / 2.0
    }
}

/// A small fixed-seed generator (xorshift64*): the same interval each run.
struct Rng(u64);

impl Rng {
    fn below(&mut self, n: usize) -> usize {
        self.0 ^= self.0 >> 12;
        self.0 ^= self.0 << 25;
        self.0 ^= self.0 >> 27;
        (self.0.wrapping_mul(0x2545_F491_4F6C_DD1D) >> 33) as usize % n
    }
}

/// The bootstrap 95 % interval of median(B) - median(A).
fn interval(a: &[f64], b: &[f64], rounds: usize) -> (f64, f64) {
    let mut rng = Rng(0x9E37_79B9_7F4A_7C15);
    let mut pick =
        |v: &[f64]| -> Vec<f64> { (0..v.len()).map(|_| v[rng.below(v.len())]).collect() };
    let mut diffs: Vec<f64> = (0..rounds)
        .map(|_| median(&pick(b)) - median(&pick(a)))
        .collect();
    diffs.sort_by(f64::total_cmp);
    (diffs[rounds * 25 / 1000], diffs[rounds * 975 / 1000])
}

/// The runs of side `side` (`A` or `B`) in `dir`, in name order.
fn side(dir: &Path, side: char) -> Result<Vec<Run>> {
    let mut paths: Vec<_> = std::fs::read_dir(dir)
        .with_context(|| format!("reading {}", dir.display()))?
        .flatten()
        .map(|e| e.path())
        .filter(|p| {
            let name = p.file_name().and_then(|n| n.to_str()).unwrap_or("");
            name.strip_prefix(side)
                .and_then(|r| r.strip_suffix(".json"))
                .is_some_and(|n| !n.is_empty() && n.bytes().all(|b| b.is_ascii_digit()))
        })
        .collect();
    paths.sort();
    let mut runs = Vec::new();
    for p in paths {
        let name = p
            .file_name()
            .and_then(|n| n.to_str())
            .unwrap_or("?")
            .to_owned();
        let text =
            std::fs::read_to_string(&p).with_context(|| format!("reading {}", p.display()))?;
        match parse(&name, &text) {
            Some(r) => runs.push(r),
            None => println!("  {name}: device removed or no frame times, left out"),
        }
    }
    Ok(runs)
}

pub fn run(dir: &Path) -> Result<()> {
    if let Ok(sets) = std::fs::read_to_string(dir.join("sets.txt")) {
        println!("{}", sets.trim_end());
    }
    let (a, b) = (side(dir, 'A')?, side(dir, 'B')?);
    for r in a.iter().chain(&b) {
        println!(
            "  {}: median {:7.1} ms, p95 {:7.1} ms, {} frames",
            r.name, r.median, r.p95, r.frames
        );
    }
    anyhow::ensure!(
        a.len() >= 2 && b.len() >= 2,
        "need at least two runs a side"
    );
    let (ma, mb): (Vec<f64>, Vec<f64>) = (
        a.iter().map(|r| r.median).collect(),
        b.iter().map(|r| r.median).collect(),
    );
    let (xa, xb) = (median(&ma), median(&mb));
    let (lo, hi) = interval(&ma, &mb, 10_000);
    println!(
        "A median {xa:.1} ms ({:.2} fps), B median {xb:.1} ms ({:.2} fps)",
        1000.0 / xa,
        1000.0 / xb
    );
    println!(
        "B - A {:+.1} ms ({:+.1} %), 95 % interval {lo:+.1} to {hi:+.1} ms",
        xb - xa,
        100.0 * (xb - xa) / xa
    );
    if lo <= 0.0 && 0.0 <= hi {
        println!("the interval holds 0: no difference these runs can show");
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    const REPORT: &str = "{\n  \"world_frames\": 300,\n  \"warmup_frames\": 120,\n  \"frame_ms\": {\n    \"avg\": 341.000,\n    \"median\": 338.500,\n    \"p95\": 390.250,\n    \"p99\": 410.000\n  },\n  \"draws\": {\n    \"avg\": 6800.000\n  },\n  \"device_removed\": false\n}\n";

    #[test]
    fn reads_the_benchmark_report() {
        let r = parse("A1.json", REPORT).unwrap();
        assert_eq!((r.median, r.p95, r.frames), (338.5, 390.25, 300));
        let gone = REPORT.replace("\"device_removed\": false", "\"device_removed\": true");
        assert_eq!(parse("A2.json", &gone), None);
    }

    #[test]
    fn the_interval_tells_a_clear_change_from_noise() {
        let (a, b) = ([343.0, 346.0, 349.0], [304.0, 308.0, 312.0]);
        let (lo, hi) = interval(&a, &b, 10_000);
        assert!(hi < 0.0, "{lo} {hi}");
        let (lo, hi) = interval(&[340.0, 300.0, 380.0], &[330.0, 370.0, 310.0], 10_000);
        assert!(lo <= 0.0 && 0.0 <= hi, "{lo} {hi}");
        assert_eq!(median(&[3.0, 1.0, 2.0, 4.0]), 2.5);
    }

    #[test]
    fn compares_a_folder_of_reports() {
        let dir = std::env::temp_dir().join(format!("xtask-bench-{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        for (name, m) in [
            ("A1", 343.0),
            ("A2", 346.0),
            ("B1", 304.0),
            ("B2", 308.0),
            ("warm", 500.0),
        ] {
            std::fs::write(
                dir.join(format!("{name}.json")),
                REPORT.replace("338.500", &m.to_string()),
            )
            .unwrap();
        }
        let a = side(&dir, 'A').unwrap();
        assert_eq!(
            a.iter().map(|r| r.median).collect::<Vec<_>>(),
            [343.0, 346.0]
        );
        assert!(run(&dir).is_ok());
        std::fs::remove_dir_all(&dir).unwrap();
    }
}
