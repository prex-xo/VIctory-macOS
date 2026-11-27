//! Opt-in deterministic world-frame benchmarking and JSON report output.
//!
//! All reporting and exit behavior are off unless `VICTORY_BENCH_OUT` or
//! `VICTORY_BENCH_EXIT_FRAMES` is set. `world_frame()` advances monotonically
//! once the title presents its first frame with at least `WORLD_DRAW_THRESHOLD`
//! draws so input replay (`victory_hle`) and benchmark capture share one
//! loading-screen-immune timeline.
//!
//! The warm-up (`VICTORY_BENCH_WARMUP`) counts frames of the world only (at
//! least `WORLD_DRAW_THRESHOLD` draws), and starts again after a loading
//! phase (`LOADING_RUN` frames in a row below it): Grand Theft Auto VI shows
//! one brief frame of the world, then loads for another 10-15 s, which used
//! to spend the whole warm-up so the report measured the loading screen.
//! Once measuring has begun, every presented frame is measured.

use std::path::{Component, Path, PathBuf};
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::{Mutex, OnceLock};
use std::time::Instant;

/// Minimum draws in a single presented frame to latch into the 3D world.
pub const WORLD_DRAW_THRESHOLD: u64 = 500;
const DEFAULT_WARMUP_FRAMES: u64 = 30;
/// Frames below `WORLD_DRAW_THRESHOLD` in a row that make a loading phase
/// during the warm-up: about a second of a loading screen at 30 fps. Frames
/// of the world (2-30 fps) do not stay under the threshold that long.
const LOADING_RUN: u64 = 30;
const ROOT_MARKER: &str = ".victory-root";
const MIB: f64 = 1024.0 * 1024.0;

static WORLD_FRAME: AtomicU64 = AtomicU64::new(0);
static LAST_DRAWN: AtomicU64 = AtomicU64::new(0);
static DEVICE_GONE: AtomicBool = AtomicBool::new(false);
static INPUT_SCRIPT_HASH: OnceLock<String> = OnceLock::new();
static TMP_SEQ: AtomicU64 = AtomicU64::new(0);

/// Returns the current world-frame index (`0` while on menus/loading screens,
/// `1` on the first presented frame with >= 500 draws, then incrementing by 1
/// on every subsequent presented frame).
pub fn world_frame() -> u64 {
    WORLD_FRAME.load(Ordering::Relaxed)
}

/// Computes and records the SHA-256 hex digest of the active replay script so
/// benchmark reports bind timings to exact input bytes.
pub fn register_input_script(bytes: &[u8]) -> String {
    let hex = input_script_digest(bytes);
    let _ = INPUT_SCRIPT_HASH.set(hex.clone());
    hex
}

fn input_script_digest(bytes: &[u8]) -> String {
    use sha2::{Digest, Sha256};
    use std::fmt::Write;

    let mut hex = String::with_capacity(64);
    for byte in Sha256::digest(bytes) {
        let _ = write!(hex, "{byte:02x}");
    }
    hex
}

/// Sets the input script digest directly if computed by the caller.
pub fn set_input_script_sha256(hash: String) {
    let _ = INPUT_SCRIPT_HASH.set(hash);
}

/// Records that the host D3D12 device has been removed.
#[cfg(windows)]
pub(crate) fn mark_device_removed() {
    DEVICE_GONE.store(true, Ordering::Relaxed);
}

fn is_device_removed() -> bool {
    if DEVICE_GONE.load(Ordering::Relaxed) {
        return true;
    }
    #[cfg(windows)]
    if crate::host::device_removed() {
        DEVICE_GONE.store(true, Ordering::Relaxed);
        return true;
    }
    false
}

#[derive(Clone, Debug)]
struct BenchConfig {
    enabled: bool,
    out_path: Option<PathBuf>,
    warmup_frames: u64,
    exit_frames: Option<u64>,
}

impl BenchConfig {
    fn from_env() -> Self {
        let out_raw = std::env::var_os("VICTORY_BENCH_OUT").and_then(|v| {
            let s = v.to_string_lossy().trim().to_owned();
            (!s.is_empty()).then(|| PathBuf::from(s))
        });
        let out_path = match out_raw {
            Some(path) => match ensure_outside_repo(&path) {
                Ok(()) => Some(path),
                Err(err) => {
                    tracing::error!(
                        path = %path.display(),
                        error = %err,
                        "refusing VICTORY_BENCH_OUT inside repository"
                    );
                    None
                }
            },
            None => None,
        };
        let warmup_frames = std::env::var("VICTORY_BENCH_WARMUP")
            .ok()
            .and_then(|s| s.trim().parse::<u64>().ok())
            .unwrap_or(DEFAULT_WARMUP_FRAMES);
        let exit_frames = std::env::var("VICTORY_BENCH_EXIT_FRAMES")
            .ok()
            .and_then(|s| s.trim().parse::<u64>().ok())
            .filter(|&n| n > 0);
        let enabled = out_path.is_some() || exit_frames.is_some();
        Self {
            enabled,
            out_path,
            warmup_frames,
            exit_frames,
        }
    }
}

fn config() -> &'static BenchConfig {
    static CONFIG: OnceLock<BenchConfig> = OnceLock::new();
    CONFIG.get_or_init(BenchConfig::from_env)
}

/// Ensures `path` is outside any source tree marked by `.victory-root`.
pub fn ensure_outside_repo(path: &Path) -> Result<(), String> {
    let resolved = resolve_path(path)?;
    let compiled_root = Path::new(env!("CARGO_MANIFEST_DIR")).join("..").join("..");
    if let Ok(root) = resolve_path(&compiled_root)
        && root.join(ROOT_MARKER).is_file()
        && resolved.starts_with(&root)
    {
        return Err(format!(
            "refusing to write benchmark output {} inside repository {}",
            path.display(),
            root.display()
        ));
    }
    for ancestor in resolved.ancestors() {
        if ancestor.join(ROOT_MARKER).is_file() {
            return Err(format!(
                "refusing to write benchmark output {} inside repository {}",
                path.display(),
                ancestor.display()
            ));
        }
    }
    Ok(())
}

fn resolve_path(path: &Path) -> Result<PathBuf, String> {
    let abs = if path.is_absolute() {
        path.to_path_buf()
    } else {
        let cwd = std::env::current_dir()
            .map_err(|e| format!("cannot get current dir: {e}"))?;
        // On Windows/Wine, a root-relative path like `/Users/...` (such as a
        // cross-compiled `CARGO_MANIFEST_DIR` or a POSIX path passed on macOS)
        // has `RootDir` without `Prefix`. `PathBuf::join` would drop `cwd`'s
        // drive prefix; prepend `cwd`'s prefix, falling back to `Z:` under Wine
        // when the POSIX path lives on the host root mapped at `Z:\`.
        if matches!(path.components().next(), Some(Component::RootDir)) {
            let with_prefix = |prefix: &std::ffi::OsStr| {
                let mut os = prefix.to_os_string();
                os.push(path.as_os_str());
                PathBuf::from(os)
            };
            let candidate = match cwd.components().next() {
                Some(Component::Prefix(prefix)) => with_prefix(prefix.as_os_str()),
                _ => cwd.join(path),
            };
            let deepest_existing = |p: &Path| {
                p.ancestors()
                    .find(|a| a.exists())
                    .map_or(0, |a| a.components().count())
            };
            if cfg!(windows) {
                let z = with_prefix(std::ffi::OsStr::new("Z:"));
                if deepest_existing(&z) > deepest_existing(&candidate) {
                    z
                } else {
                    candidate
                }
            } else {
                candidate
            }
        } else {
            cwd.join(path)
        }
    };
    let mut normalized = PathBuf::new();
    for comp in abs.components() {
        match comp {
            Component::CurDir => {}
            Component::ParentDir => {
                normalized.pop();
            }
            Component::RootDir if !normalized.as_os_str().is_empty() => {
                // On Windows, `PathBuf::push(r"\")` drops a preceding drive
                // prefix (`Z:` -> `\`); preserve the prefix by appending raw.
                let mut os = normalized.into_os_string();
                os.push(comp.as_os_str());
                normalized = PathBuf::from(os);
            }
            other => normalized.push(other.as_os_str()),
        }
    }
    let mut tail = Vec::new();
    let mut probe = normalized.as_path();
    loop {
        if probe.exists() {
            let mut base = std::fs::canonicalize(probe)
                .map_err(|e| format!("cannot resolve {}: {e}", probe.display()))?;
            for part in tail.into_iter().rev() {
                base.push(part);
            }
            return Ok(base);
        }
        let Some(name) = probe.file_name() else {
            return Ok(normalized);
        };
        tail.push(name.to_os_string());
        let Some(parent) = probe.parent() else {
            return Ok(normalized);
        };
        probe = parent;
    }
}

#[derive(Clone, Copy, Debug, Default)]
struct MemorySample {
    vram_used_bytes: u64,
    vram_budget_bytes: u64,
    ram_private_bytes: u64,
    ram_working_set_bytes: u64,
}

fn sample_host_memory() -> MemorySample {
    #[cfg(windows)]
    {
        let (vram_used_bytes, vram_budget_bytes) =
            crate::host::bench_video_memory().unwrap_or((0, 0));
        let (ram_private_bytes, ram_working_set_bytes) = crate::host::process_memory();
        MemorySample {
            vram_used_bytes,
            vram_budget_bytes,
            ram_private_bytes,
            ram_working_set_bytes,
        }
    }
    #[cfg(not(windows))]
    {
        MemorySample::default()
    }
}

#[derive(Clone, Debug, PartialEq)]
struct FrameMsStats {
    avg: f64,
    median: f64,
    p95: f64,
    p99: f64,
    low_5pct_fps: f64,
    low_1pct_fps: f64,
}

#[derive(Clone, Debug, PartialEq)]
struct DrawsStats {
    avg: f64,
    min: u64,
    max: u64,
}

#[derive(Clone, Debug, PartialEq)]
struct BenchReport {
    world_frames: u64,
    warmup_frames: u64,
    frame_ms: FrameMsStats,
    draws: DrawsStats,
    zero_draw_world_frames: u64,
    black_frames: u64,
    device_removed: bool,
    vram_used_peak_mb: f64,
    vram_budget_mb: f64,
    ram_private_commit_peak_mb: f64,
    ram_working_set_peak_mb: f64,
    input_script_sha256: Option<String>,
}

impl BenchReport {
    fn to_json(&self) -> String {
        let sha = match &self.input_script_sha256 {
            Some(s) => format!("\"{}\"", escape_json(s)),
            None => "null".to_owned(),
        };
        format!(
            concat!(
                "{{\n",
                "  \"world_frames\": {},\n",
                "  \"warmup_frames\": {},\n",
                "  \"frame_ms\": {{\n",
                "    \"avg\": {:.3},\n",
                "    \"median\": {:.3},\n",
                "    \"p95\": {:.3},\n",
                "    \"p99\": {:.3},\n",
                "    \"low_5pct_fps\": {:.3},\n",
                "    \"low_1pct_fps\": {:.3}\n",
                "  }},\n",
                "  \"draws\": {{\n",
                "    \"avg\": {:.3},\n",
                "    \"min\": {},\n",
                "    \"max\": {}\n",
                "  }},\n",
                "  \"zero_draw_world_frames\": {},\n",
                "  \"black_frames\": {},\n",
                "  \"device_removed\": {},\n",
                "  \"vram_used_peak_mb\": {:.3},\n",
                "  \"vram_budget_mb\": {:.3},\n",
                "  \"ram_private_commit_peak_mb\": {:.3},\n",
                "  \"ram_working_set_peak_mb\": {:.3},\n",
                "  \"input_script_sha256\": {}\n",
                "}}\n"
            ),
            self.world_frames,
            self.warmup_frames,
            self.frame_ms.avg,
            self.frame_ms.median,
            self.frame_ms.p95,
            self.frame_ms.p99,
            self.frame_ms.low_5pct_fps,
            self.frame_ms.low_1pct_fps,
            self.draws.avg,
            self.draws.min,
            self.draws.max,
            self.zero_draw_world_frames,
            self.black_frames,
            self.device_removed,
            self.vram_used_peak_mb,
            self.vram_budget_mb,
            self.ram_private_commit_peak_mb,
            self.ram_working_set_peak_mb,
            sha,
        )
    }
}

fn escape_json(s: &str) -> String {
    let mut out = String::with_capacity(s.len());
    for c in s.chars() {
        match c {
            '"' => out.push_str("\\\""),
            '\\' => out.push_str("\\\\"),
            '\n' => out.push_str("\\n"),
            '\r' => out.push_str("\\r"),
            '\t' => out.push_str("\\t"),
            c if (c as u32) < 0x20 => {
                use std::fmt::Write;
                let _ = write!(out, "\\u{:04x}", c as u32);
            }
            c => out.push(c),
        }
    }
    out
}

struct BenchRecorder {
    warmup_target: u64,
    world_frame: u64,
    prev_drawn: u64,
    warmup_frames: u64,
    /// Frames of the world the warm-up has counted since it last started.
    warmup_world: u64,
    /// Frames below `WORLD_DRAW_THRESHOLD` in a row during the warm-up.
    low_run: u64,
    /// The warm-up is over: every frame is measured from now on.
    measuring: bool,
    frame_ms: Vec<f64>,
    draws: Vec<u64>,
    zero_draw_world_frames: u64,
    black_frames: u64,
    device_removed: bool,
    vram_used_peak_bytes: u64,
    vram_budget_bytes: u64,
    ram_private_peak_bytes: u64,
    ram_working_set_peak_bytes: u64,
}

impl BenchRecorder {
    fn new(warmup_target: u64) -> Self {
        Self {
            warmup_target,
            world_frame: 0,
            prev_drawn: 0,
            warmup_frames: 0,
            warmup_world: 0,
            low_run: 0,
            measuring: false,
            frame_ms: Vec::new(),
            draws: Vec::new(),
            zero_draw_world_frames: 0,
            black_frames: 0,
            device_removed: false,
            vram_used_peak_bytes: 0,
            vram_budget_bytes: 0,
            ram_private_peak_bytes: 0,
            ram_working_set_peak_bytes: 0,
        }
    }

    /// Advances world-frame tracking from cumulative `drawn` count and records
    /// post-warmup metrics. `frame_ms` is `None` on the first ever present
    /// (where no prior present timestamp exists) so a bogus `0.0 ms` sample is
    /// never recorded when `VICTORY_BENCH_WARMUP=0`. Returns
    /// `(world_frame, measured_count)`.
    fn observe(
        &mut self,
        drawn: u64,
        frame_ms: Option<f64>,
        black_frame: bool,
        mem: MemorySample,
        device_removed: bool,
    ) -> (u64, u64) {
        let frame_draws = drawn.saturating_sub(self.prev_drawn);
        self.prev_drawn = drawn;

        if self.world_frame == 0 {
            if frame_draws >= WORLD_DRAW_THRESHOLD {
                self.world_frame = 1;
            } else {
                return (0, self.draws.len() as u64);
            }
        } else {
            self.world_frame += 1;
        }

        // The warm-up ends once it has counted `warmup_target` frames of the
        // world; frames below the threshold do not count, and a loading
        // phase starts it again.
        self.measuring |= self.warmup_world >= self.warmup_target;
        if !self.measuring {
            if frame_draws >= WORLD_DRAW_THRESHOLD {
                self.low_run = 0;
                self.warmup_world += 1;
            } else {
                self.low_run += 1;
                if self.low_run >= LOADING_RUN {
                    self.warmup_world = 0;
                }
            }
            self.warmup_frames += 1;
            return (self.world_frame, self.draws.len() as u64);
        }

        if let Some(dt) = frame_ms {
            self.frame_ms.push(dt.max(0.0));
        }
        self.draws.push(frame_draws);
        if frame_draws == 0 {
            self.zero_draw_world_frames += 1;
        }
        if black_frame {
            self.black_frames += 1;
        }
        self.device_removed |= device_removed;
        self.vram_used_peak_bytes = self.vram_used_peak_bytes.max(mem.vram_used_bytes);
        if mem.vram_budget_bytes > 0 {
            self.vram_budget_bytes = mem.vram_budget_bytes;
        }
        self.ram_private_peak_bytes = self.ram_private_peak_bytes.max(mem.ram_private_bytes);
        self.ram_working_set_peak_bytes = self
            .ram_working_set_peak_bytes
            .max(mem.ram_working_set_bytes);

        (self.world_frame, self.draws.len() as u64)
    }

    fn report(&self, input_script_sha256: Option<String>) -> BenchReport {
        let frame_ms = summarize_frame_ms(&self.frame_ms);
        let draws = summarize_draws(&self.draws);
        BenchReport {
            world_frames: self.draws.len() as u64,
            warmup_frames: self.warmup_frames,
            frame_ms,
            draws,
            zero_draw_world_frames: self.zero_draw_world_frames,
            black_frames: self.black_frames,
            device_removed: self.device_removed,
            vram_used_peak_mb: self.vram_used_peak_bytes as f64 / MIB,
            vram_budget_mb: self.vram_budget_bytes as f64 / MIB,
            ram_private_commit_peak_mb: self.ram_private_peak_bytes as f64 / MIB,
            ram_working_set_peak_mb: self.ram_working_set_peak_bytes as f64 / MIB,
            input_script_sha256,
        }
    }
}

fn percentile_sorted(sorted: &[f64], p: f64) -> f64 {
    if sorted.is_empty() {
        return 0.0;
    }
    if sorted.len() == 1 {
        return sorted[0];
    }
    let idx = (p.clamp(0.0, 1.0) * (sorted.len() - 1) as f64).round() as usize;
    sorted[idx.min(sorted.len() - 1)]
}

fn low_pct_fps(sorted_asc: &[f64], pct: f64) -> f64 {
    if sorted_asc.is_empty() {
        return 0.0;
    }
    let count = ((sorted_asc.len() as f64 * pct).ceil() as usize).clamp(1, sorted_asc.len());
    let tail = &sorted_asc[sorted_asc.len() - count..];
    let avg_ms = tail.iter().sum::<f64>() / count as f64;
    if avg_ms > 0.0 { 1000.0 / avg_ms } else { 0.0 }
}

fn summarize_frame_ms(samples: &[f64]) -> FrameMsStats {
    if samples.is_empty() {
        return FrameMsStats {
            avg: 0.0,
            median: 0.0,
            p95: 0.0,
            p99: 0.0,
            low_5pct_fps: 0.0,
            low_1pct_fps: 0.0,
        };
    }
    let mut sorted = samples.to_vec();
    sorted.sort_by(f64::total_cmp);
    let avg = sorted.iter().sum::<f64>() / sorted.len() as f64;
    let median = if sorted.len().is_multiple_of(2) {
        let mid = sorted.len() / 2;
        0.5 * (sorted[mid - 1] + sorted[mid])
    } else {
        sorted[sorted.len() / 2]
    };
    FrameMsStats {
        avg,
        median,
        p95: percentile_sorted(&sorted, 0.95),
        p99: percentile_sorted(&sorted, 0.99),
        low_5pct_fps: low_pct_fps(&sorted, 0.05),
        low_1pct_fps: low_pct_fps(&sorted, 0.01),
    }
}

fn summarize_draws(samples: &[u64]) -> DrawsStats {
    if samples.is_empty() {
        return DrawsStats {
            avg: 0.0,
            min: 0,
            max: 0,
        };
    }
    let sum: u128 = samples.iter().map(|&d| u128::from(d)).sum();
    DrawsStats {
        avg: sum as f64 / samples.len() as f64,
        min: samples.iter().copied().min().unwrap_or(0),
        max: samples.iter().copied().max().unwrap_or(0),
    }
}

fn write_report_atomically(path: &Path, report: &BenchReport) -> std::io::Result<()> {
    if let Some(parent) = path.parent()
        && !parent.as_os_str().is_empty()
    {
        std::fs::create_dir_all(parent)?;
    }
    let mut tmp_name = path
        .file_name()
        .map(std::ffi::OsStr::to_os_string)
        .unwrap_or_default();
    let pid = std::process::id();
    let seq = TMP_SEQ.fetch_add(1, Ordering::Relaxed);
    tmp_name.push(format!(".tmp.{pid}.{seq}"));
    let tmp = path.with_file_name(tmp_name);
    if let Err(e) = std::fs::write(&tmp, report.to_json().as_bytes())
        .and_then(|()| std::fs::rename(&tmp, path))
    {
        let _ = std::fs::remove_file(&tmp);
        return Err(e);
    }
    Ok(())
}

/// Called once per presented frame from `queue::present`.
pub(crate) fn on_present(drawn: u64, black_frame: bool) {
    let prev = LAST_DRAWN.swap(drawn, Ordering::Relaxed);
    let frame_draws = drawn.saturating_sub(prev);

    let wf = WORLD_FRAME.load(Ordering::Relaxed);
    let next_wf = if wf == 0 {
        if frame_draws >= WORLD_DRAW_THRESHOLD {
            let _ = WORLD_FRAME.compare_exchange(0, 1, Ordering::Relaxed, Ordering::Relaxed);
            1
        } else {
            0
        }
    } else {
        WORLD_FRAME.fetch_add(1, Ordering::Relaxed) + 1
    };

    let cfg = config();
    if !cfg.enabled {
        return;
    }

    struct State {
        recorder: BenchRecorder,
        last_present: Option<Instant>,
    }
    static STATE: Mutex<Option<State>> = Mutex::new(None);
    let now = Instant::now();
    let (should_flush, should_exit, report) = {
        let mut guard = STATE.lock().unwrap_or_else(|e| e.into_inner());
        let state = guard.get_or_insert_with(|| State {
            recorder: BenchRecorder::new(cfg.warmup_frames),
            last_present: None,
        });
        let dt_ms = state
            .last_present
            .map(|prev_at| now.saturating_duration_since(prev_at).as_secs_f64() * 1000.0);
        state.last_present = Some(now);

        if next_wf == 0 {
            return;
        }
        // Keep `recorder.world_frame` synchronized with the atomic `WORLD_FRAME`.
        state.recorder.prev_drawn = prev;
        state.recorder.world_frame = next_wf - 1;
        let mem = if next_wf > cfg.warmup_frames {
            sample_host_memory()
        } else {
            MemorySample::default()
        };
        let removed = if next_wf > cfg.warmup_frames {
            is_device_removed()
        } else {
            false
        };
        let (_, measured) = state
            .recorder
            .observe(drawn, dt_ms, black_frame, mem, removed);
        let done = cfg.exit_frames.is_some_and(|n| measured >= n);
        let periodic = measured == 1 || (measured > 0 && measured.is_multiple_of(30));
        if done || periodic || removed {
            (
                cfg.out_path.is_some(),
                done,
                Some(state.recorder.report(INPUT_SCRIPT_HASH.get().cloned())),
            )
        } else {
            (false, false, None)
        }
    };

    if should_flush
        && let (Some(path), Some(report)) = (cfg.out_path.as_deref(), report.as_ref())
        && let Err(e) = write_report_atomically(path, report)
    {
        tracing::warn!(
            path = %path.display(),
            error = %e,
            "benchmark report write failed"
        );
    }
    if should_exit {
        tracing::info!("VICTORY_BENCH_EXIT_FRAMES reached; exiting cleanly");
        std::process::exit(0);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn input_script_digest_matches_sha256() {
        assert_eq!(
            input_script_digest(b"abc"),
            "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad"
        );
    }

    #[test]
    fn world_frame_latches_at_500_draws_and_excludes_warmup() {
        let mut rec = BenchRecorder::new(3);
        let mem = MemorySample::default();

        // Loading screen frames below 500 draws remain at world_frame == 0.
        assert_eq!(rec.observe(100, Some(16.0), false, mem, false), (0, 0));
        assert_eq!(rec.observe(599, Some(16.0), false, mem, false), (0, 0));

        // First frame with >= 500 draws latches to world_frame == 1 (warmup 1/3).
        assert_eq!(rec.observe(1100, Some(50.0), false, mem, false), (1, 0));
        assert_eq!(rec.observe(1700, Some(40.0), false, mem, false), (2, 0));
        assert_eq!(rec.observe(2300, Some(30.0), false, mem, false), (3, 0));

        // Post-warmup world frames are measured even if draws drop on a subsequent frame.
        assert_eq!(rec.observe(3300, Some(10.0), false, mem, false), (4, 1));
        assert_eq!(rec.observe(3300, Some(20.0), true, mem, false), (5, 2));

        let rep = rec.report(None);
        assert_eq!(rep.warmup_frames, 3);
        assert_eq!(rep.world_frames, 2);
        assert!((rep.frame_ms.avg - 15.0).abs() < 1e-6);
        assert!((rep.frame_ms.median - 15.0).abs() < 1e-6);
        assert_eq!(rep.draws.min, 0);
        assert_eq!(rep.draws.max, 1000);
        assert!((rep.draws.avg - 500.0).abs() < 1e-6);
        assert_eq!(rep.zero_draw_world_frames, 1);
        assert_eq!(rep.black_frames, 1);

        // Warmup=0 with `frame_ms: None` on the first present must not record a bogus 0.0 ms sample.
        let mut zero_warmup = BenchRecorder::new(0);
        assert_eq!(zero_warmup.observe(600, None, false, mem, false), (1, 1));
        assert_eq!(
            zero_warmup.observe(1200, Some(12.0), false, mem, false),
            (2, 2)
        );
        let rep0 = zero_warmup.report(None);
        assert_eq!(rep0.world_frames, 2);
        assert!((rep0.frame_ms.avg - 12.0).abs() < 1e-6);
        assert!((rep0.frame_ms.median - 12.0).abs() < 1e-6);
    }

    /// Feeds `frames` presents of `draws` draws each and returns the last
    /// `(world_frame, measured)`.
    fn feed(rec: &mut BenchRecorder, drawn: &mut u64, draws: u64, frames: u64) -> (u64, u64) {
        let mem = MemorySample::default();
        let mut last = (0, 0);
        for _ in 0..frames {
            *drawn += draws;
            last = rec.observe(*drawn, Some(33.0), false, mem, false);
        }
        last
    }

    /// As seen in the game: one world frame of ~2,000 draws, then 10-15 s
    /// more of a loading screen (2-11 draws at ~25 fps), then the world. The
    /// warm-up must not be spent on the loading screen.
    #[test]
    fn warmup_starts_again_after_a_loading_phase_behind_the_first_world_frame() {
        let mut rec = BenchRecorder::new(150);
        let mut drawn = 0;
        assert_eq!(feed(&mut rec, &mut drawn, 6, 900), (0, 0)); // menus
        assert_eq!(feed(&mut rec, &mut drawn, 2016, 1), (1, 0));
        assert_eq!(feed(&mut rec, &mut drawn, 11, 340), (341, 0));
        // The world: 150 frames of warm-up, then measured.
        assert_eq!(feed(&mut rec, &mut drawn, 4800, 150), (491, 0));
        assert_eq!(feed(&mut rec, &mut drawn, 4800, 3), (494, 3));
        let rep = rec.report(None);
        assert_eq!(rep.warmup_frames, 491);
        assert_eq!(rep.world_frames, 3);
        assert_eq!((rep.draws.min, rep.draws.max), (4800, 4800));
        assert_eq!(rep.zero_draw_world_frames, 0);
    }

    /// Shorter dips below the threshold (a cut, a black frame) neither count
    /// toward the warm-up nor start it again.
    #[test]
    fn brief_dips_neither_count_toward_warmup_nor_restart_it() {
        let mut rec = BenchRecorder::new(4);
        let mut drawn = 0;
        assert_eq!(feed(&mut rec, &mut drawn, 3000, 2), (2, 0));
        assert_eq!(
            feed(&mut rec, &mut drawn, 0, LOADING_RUN - 1),
            (2 + LOADING_RUN - 1, 0)
        );
        assert_eq!(feed(&mut rec, &mut drawn, 3000, 2), (LOADING_RUN + 3, 0));
        // Four frames of the world counted: the next is measured, and once
        // measuring, frames with no draws are measured and counted too.
        assert_eq!(feed(&mut rec, &mut drawn, 3000, 1), (LOADING_RUN + 4, 1));
        assert_eq!(
            feed(&mut rec, &mut drawn, 0, LOADING_RUN),
            (2 * LOADING_RUN + 4, LOADING_RUN + 1)
        );
        let rep = rec.report(None);
        assert_eq!(rep.warmup_frames, LOADING_RUN + 3);
        assert_eq!(rep.zero_draw_world_frames, LOADING_RUN);
    }

    #[test]
    fn percentile_low_fps_and_memory_peaks_compute_accurately() {
        let mut rec = BenchRecorder::new(0);
        let mut drawn = 0u64;
        // 100 samples: 1..=100 ms (99 frames + 1 initial >= 500 draws).
        for i in 1..=100u64 {
            drawn += 500 + i;
            let mem = MemorySample {
                vram_used_bytes: i * (1 << 20),
                vram_budget_bytes: 12288 * (1 << 20),
                ram_private_bytes: (1000 + i) * (1 << 20),
                ram_working_set_bytes: (800 + i) * (1 << 20),
            };
            rec.observe(drawn, Some(i as f64), i == 42, mem, i == 99);
        }
        let rep = rec.report(Some("abc123".to_owned()));
        assert_eq!(rep.world_frames, 100);
        assert_eq!(rep.warmup_frames, 0);
        assert!((rep.frame_ms.avg - 50.5).abs() < 1e-6);
        assert!((rep.frame_ms.median - 50.5).abs() < 1e-6);
        assert!((rep.frame_ms.p95 - 95.0).abs() < 1.0);
        assert!((rep.frame_ms.p99 - 99.0).abs() < 1.0);
        // Slowest 1% (1 frame = 100 ms) -> 10.0 FPS; slowest 5% (96..=100, avg 98 ms) -> 1000/98 FPS.
        assert!((rep.frame_ms.low_1pct_fps - 10.0).abs() < 1e-6);
        assert!((rep.frame_ms.low_5pct_fps - (1000.0 / 98.0)).abs() < 1e-6);
        assert!((rep.vram_used_peak_mb - 100.0).abs() < 1e-6);
        assert!((rep.vram_budget_mb - 12288.0).abs() < 1e-6);
        assert!((rep.ram_private_commit_peak_mb - 1100.0).abs() < 1e-6);
        assert!((rep.ram_working_set_peak_mb - 900.0).abs() < 1e-6);
        assert_eq!(rep.black_frames, 1);
        assert!(rep.device_removed);

        let json = rep.to_json();
        assert!(json.contains("\"world_frames\": 100"));
        assert!(json.contains("\"low_1pct_fps\": 10.000"));
        assert!(json.contains("\"device_removed\": true"));
        assert!(json.contains("\"input_script_sha256\": \"abc123\""));
    }

    #[test]
    fn outside_repo_guard_and_atomic_json_write_work() {
        let inside = Path::new(env!("CARGO_MANIFEST_DIR")).join("bench-out.json");
        assert!(ensure_outside_repo(&inside).is_err());

        let dir = tempfile::tempdir().unwrap();
        let outside = dir.path().join("nested").join("bench.json");
        assert!(ensure_outside_repo(&outside).is_ok());

        let rec = BenchRecorder::new(0);
        let rep = rec.report(Some(blake3::hash(b"script").to_hex().to_string()));
        write_report_atomically(&outside, &rep).unwrap();
        let written = std::fs::read_to_string(&outside).unwrap();
        assert!(written.contains("\"world_frames\": 0"));
        let leftover_tmps: Vec<_> = std::fs::read_dir(outside.parent().unwrap())
            .unwrap()
            .filter_map(Result::ok)
            .filter(|e| e.file_name().to_string_lossy().contains(".tmp."))
            .collect();
        assert!(leftover_tmps.is_empty());
    }
}
