//! Host pipelines are made by workers, not by the title thread that first
//! records a draw with them. Translating a shader and the host driver's
//! compile take from milliseconds to seconds each (structurisation of the
//! biggest compute programs ~0.4 s, some driver compiles 3.4 s), and Grand
//! Theft Auto VI's first world frame wants over 1,200 new pipelines: made
//! on its recording threads, that frame took more than 30 s and the
//! title's GPU watchdog ended the run (decision 0007, "Pipelines made off
//! the title's threads").
//!
//! A first use queues the pipeline and waits for it within a budget per
//! frame, so the cheap ones (menus, one-off passes such as a colour-grading
//! bake) are there at once as before; past the budget a draw or dispatch
//! whose pipeline is not ready is dropped, as an untranslated one is.
//! Pipelines a draw waits for go ahead of those started early (at the
//! title's creation, because an earlier run drew them).
//!
//! shortcut: the budget (half a second of waiting per title frame) is a
//! guess; a pass the title runs only once and whose pipeline takes longer
//! is lost.

use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::{Arc, Condvar, Mutex, OnceLock};
use std::time::{Duration, Instant};

use super::Shader;
use super::host::{GraphicsState, HostPipeline, TargetFormats};
use super::schedule::{
    COMPUTE_BUDGET, Lane, PriorityQueue, WaitBudgets, lane_for, sync_compute_enabled,
};

/// What a host pipeline is made from, and the host pipeline once made.
pub(crate) struct Pending {
    shaders: Vec<Arc<Shader>>,
    pub(crate) graphics: Option<Box<GraphicsState>>,
    /// Compute pipeline eligible for priority queueing and bounded cold wait.
    compute: bool,
    /// When a worker began compiling this pipeline.
    making_since: OnceLock<Instant>,
    /// One-shot diagnostic latches for compute wait timeouts.
    compute_dropped: AtomicBool,
    compute_given_up: AtomicBool,
    queued: AtomicBool,
    /// Whether it went into a requested-work queue ahead of early pre-warms,
    /// and whether a worker took it (it may be in both queues).
    now: AtomicBool,
    taken: AtomicBool,
    made: OnceLock<Option<HostPipeline>>,
    #[cfg(feature = "diagnostics")]
    refusal: OnceLock<(&'static str, Option<i32>)>,
    #[cfg(feature = "diagnostics")]
    pub(super) skipped_shader: Option<String>,
    /// What identifies it across runs (shader containers and state), and
    /// whether a draw has used it yet.
    pub(crate) key: String,
    used: AtomicBool,
    /// The formats its state declares for what a draw draws into (zeros
    /// without a graphics state).
    pub(crate) formats: TargetFormats,
    /// The same shaders and state for other formats, bound at a draw
    /// ([`for_formats`]): a chain from the title's pipeline, one link per
    /// set of formats met, never shortened.
    variant: OnceLock<Arc<Pending>>,
}

// SAFETY: D3D12 objects are free-threaded; the graphics state (whose input
// elements point into its own names) is only read after creation.
unsafe impl Send for Pending {}
// SAFETY: as above.
unsafe impl Sync for Pending {}

impl Pending {
    pub(crate) fn new(
        shaders: Vec<Arc<Shader>>,
        graphics: Option<Box<GraphicsState>>,
        #[cfg(feature = "diagnostics")] skip_hashes: &[String],
    ) -> Pending {
        let key = key(&shaders, graphics.as_deref());
        #[cfg(feature = "diagnostics")]
        log("created", &key, || {
            let state = graphics.as_ref().map(|g| g.describe()).unwrap_or_default();
            format!("{shaders:?}\t{state}")
        });
        #[cfg(feature = "diagnostics")]
        let skipped_shader = super::skipped_shader(&shaders, skip_hashes).map(str::to_owned);
        let made = OnceLock::new();
        // Publish the completed result before the pipeline can be queued.
        #[cfg(feature = "diagnostics")]
        if skipped_shader.is_some() {
            let _ = made.set(None);
        }
        let compute = graphics.is_none()
            && shaders.first().is_some_and(|s| s.stage == "compute")
            && sync_compute_enabled();
        Pending {
            formats: graphics.as_ref().map_or([0; 9], |g| g.formats()),
            variant: OnceLock::new(),
            key,
            shaders,
            graphics,
            compute,
            making_since: OnceLock::new(),
            compute_dropped: AtomicBool::new(false),
            compute_given_up: AtomicBool::new(false),
            queued: AtomicBool::new(false),
            now: AtomicBool::new(false),
            taken: AtomicBool::new(false),
            made,
            #[cfg(feature = "diagnostics")]
            refusal: OnceLock::new(),
            #[cfg(feature = "diagnostics")]
            skipped_shader,
            used: AtomicBool::new(false),
        }
    }

    fn make(&self) {
        self.making_since.get_or_init(Instant::now);
        self.made.get_or_init(|| {
            let (started, cpu) = (Instant::now(), super::thread_cpu());
            let made = HostPipeline::new(&self.shaders, self.graphics.as_deref(), |_reason, _code| {
                #[cfg(feature = "diagnostics")]
                let _ = self.refusal.set((_reason, _code));
            });
            #[cfg(feature = "diagnostics")]
            log_built(&self.key);
            let (user, kernel) = super::thread_cpu_since(cpu);
            tracing::debug!(
                shaders = ?self.shaders,
                ms = started.elapsed().as_millis() as u64,
                user_ms = user,
                kernel_ms = kernel,
                "pipeline made"
            );
            made
        });
        let (lock, done) = &DONE;
        let _held = lock.lock().unwrap_or_else(|e| e.into_inner());
        done.notify_all();
    }
}

/// What identifies a title pipeline across runs: its shader containers
/// and graphics state.
pub(crate) fn key(shaders: &[Arc<Shader>], graphics: Option<&GraphicsState>) -> String {
    let mut key = blake3::Hasher::new();
    for s in shaders {
        key.update(s.stage.as_bytes());
        key.update(s.hash.as_bytes());
    }
    key.update(graphics.map(|g| g.describe()).unwrap_or_default().as_bytes());
    key.finalize().to_hex()[..32].to_owned()
}

/// Why a draw lacks a host pipeline (diagnostics): skipped shader, still
/// building, a recorded refusal (with optional HRESULT and shader list), or
/// the shaders that failed to translate. `None` if it has one.
#[cfg(feature = "diagnostics")]
pub(crate) fn missing(p: &Pending) -> Option<String> {
    if let Some(hash) = p.skipped_shader.as_deref() {
        return Some(format!("skipped shader {hash}"));
    }
    match p.made.get() {
        Some(Some(_)) => None,
        None => Some("being made".into()),
        Some(None) => {
            if let Some(&(reason, code)) = p.refusal.get() {
                return Some(match code {
                    Some(code) => format!("{reason} ({code:#010x}) {:?}", p.shaders),
                    None if p.shaders.is_empty() => reason.to_owned(),
                    None => format!("{reason} {:?}", p.shaders),
                });
            }
            let failed: Vec<String> = p
                .shaders
                .iter()
                .filter(|s| matches!(s.translated.get(), Some(None)))
                .map(|s| format!("{} {}", s.stage, s.hash))
                .collect();
            Some(if failed.is_empty() {
                format!("refused {:?}", p.shaders)
            } else {
                failed.join(" + ")
            })
        }
    }
}

/// Logs the library name of the host pipeline this thread just made or
/// loaded for `key` (diagnostics).
#[cfg(feature = "diagnostics")]
pub(crate) fn log_built(key: &str) {
    if let Some((name, loaded)) = super::library::LAST.take() {
        log("built", key, || format!("{name}\t{}", if loaded { "loaded" } else { "made" }));
    }
}

/// `VICTORY_PIPELINE_LOG=<file>` (diagnostics): a line per title pipeline
/// key created (`created`, its shaders, its state as described) and first
/// drawn or dispatched with (`used`), and its host pipeline's library name
/// once made or loaded (`built`, also from the pre-build), to compare what
/// a run creates with what the pre-build makes.
#[cfg(feature = "diagnostics")]
fn log(what: &str, key: &str, rest: impl FnOnce() -> String) {
    use std::collections::HashSet;
    use std::io::Write;
    static LOG: Mutex<Option<(std::fs::File, HashSet<String>)>> = Mutex::new(None);
    static PATH: OnceLock<Option<std::ffi::OsString>> = OnceLock::new();
    let Some(path) = PATH.get_or_init(|| std::env::var_os("VICTORY_PIPELINE_LOG")) else {
        return;
    };
    let mut log = LOG.lock().unwrap_or_else(|e| e.into_inner());
    if log.is_none() {
        *log = std::fs::File::create(path).ok().map(|f| (f, HashSet::new()));
    }
    let Some((file, seen)) = log.as_mut() else { return };
    if seen.insert(format!("{what} {key}")) {
        let _ = writeln!(file, "{what}\t{key}\t{}", rest());
    }
}

/// Signalled whenever a worker has made a pipeline.
static DONE: (Mutex<()>, Condvar) = (Mutex::new(()), Condvar::new());

/// Per-frame graphics and compute wait budgets.
static BUDGETS: WaitBudgets = WaitBudgets::new();

/// Pipelines queued and not made yet.
static WAITING: AtomicU64 = AtomicU64::new(0);

/// Pipelines queued and not made yet (for the frame line).
pub(crate) fn waiting() -> u64 {
    WAITING.load(Ordering::Relaxed)
}

/// The host pipeline if it is made (or is made within the frame's budget);
/// queues it on first use.
pub(crate) fn ready(p: &Arc<Pending>) -> Option<&HostPipeline> {
    // Listed for an early start in later runs from its first use, made or
    // not: one a run never got made is the one it should start early.
    if !p.used.swap(true, Ordering::Relaxed) {
        super::library::note_used(&p.key);
        #[cfg(feature = "diagnostics")]
        log("used", &p.key, String::new);
    }
    if p.made.get().is_none() {
        start(p, true);
        wait(p, crate::frame::current());
    }
    p.made.get()?.as_ref()
}

/// Pipelines made again for the formats bound at a draw, since the start.
static VARIANTS: AtomicU64 = AtomicU64::new(0);

/// Whether a draw into views of other formats than its pipeline declares
/// gets a host pipeline for those formats ([`for_formats`]). On by
/// default; `VICTORY_TARGET_VARIANTS=0` sends such a draw to the host with
/// the title's pipeline, as before.
fn variants_on() -> bool {
    static ON: OnceLock<bool> = OnceLock::new();
    *ON.get_or_init(|| std::env::var_os("VICTORY_TARGET_VARIANTS").is_none_or(|v| v != "0"))
}

/// The host pipeline for a draw with `p` into views of the formats
/// `bound`, which `p` does not declare (`host::formats_differ`).
#[cold]
pub(crate) fn for_formats<'a>(
    p: &'a Arc<Pending>,
    bound: &TargetFormats,
) -> Option<&'a HostPipeline> {
    match p.graphics.as_deref() {
        Some(g) if variants_on() => {
            ready(variant(p, g, &super::host::formats_for(&p.formats, bound)))
        }
        _ => ready(p),
    }
}

fn variant<'a>(p: &'a Arc<Pending>, g: &GraphicsState, bound: &TargetFormats) -> &'a Arc<Pending> {
    let mut at = p;
    for _ in 0..64 {
        let next = at.variant.get_or_init(|| {
            let so_far = VARIANTS.fetch_add(1, Ordering::Relaxed) + 1;
            tracing::info!(
                pipeline = ?p.shaders,
                declared = ?p.formats,
                bound = ?bound,
                so_far,
                "a pipeline is made again for the render target formats bound at its draw (VICTORY_TARGET_VARIANTS=0 disables)"
            );
            Arc::new(Pending::new(
                p.shaders.clone(),
                Some(Box::new(g.with_formats(bound))),
                #[cfg(feature = "diagnostics")]
                &[],
            ))
        });
        if next.formats == *bound {
            return next;
        }
        at = next;
    }
    p
}

/// Starts making `p` now, without waiting for it.
pub(crate) fn begin(p: &Arc<Pending>) {
    if !p.shaders.is_empty() {
        start(p, false);
    }
}

/// Whether `p`'s host pipeline is made (`Some(true)`), refused or not
/// translated (`Some(false)`), or not yet (`None`); never waits.
pub(crate) fn state(p: &Pending) -> Option<bool> {
    p.made.get().map(Option::is_some)
}

/// Queues `p` for the workers: once, and once more ahead of the early
/// starts if a draw or dispatch waits for it (`now`) while it is still queued.
fn start(p: &Arc<Pending>, now: bool) {
    if let Some(lane) = enqueue(p, now) {
        queue(p.clone(), lane);
    }
}

/// Which lane `p` goes into now, if any.
fn enqueue(p: &Pending, now: bool) -> Option<Lane> {
    let first = !p.queued.swap(true, Ordering::Relaxed);
    if first {
        PEAK.fetch_max(WAITING.fetch_add(1, Ordering::Relaxed) + 1, Ordering::Relaxed);
    }
    let ahead = now && !p.now.swap(true, Ordering::Relaxed);
    (first || ahead && !p.taken.load(Ordering::Relaxed)).then(|| lane_for(ahead, p.compute))
}

/// At the title's creation of a pipeline: starts it at once if an earlier
/// run drew with it (during loading, so the world's first frames find it).
pub(crate) fn early(p: &Arc<Pending>) {
    super::library::note_created(&p.key);
    if p.made.get().is_none() && !p.shaders.is_empty() && super::library::wanted_early(&p.key) {
        super::library::EARLY.fetch_add(1, Ordering::Relaxed);
        start(p, false);
    }
}

/// Waits for `p` while frame `frame` has budget left. Compute pipelines with
/// `VICTORY_SYNC_COMPUTE_PSO` enabled wait against a dedicated bounded
/// wall-clock window so cold one-time dispatches are not dropped behind
/// graphics compilation.
fn wait(p: &Pending, frame: u64) {
    let outcome = BUDGETS.wait_for(
        frame,
        p.compute,
        || p.making_since.get().map_or(Duration::ZERO, Instant::elapsed),
        || p.made.get().is_some(),
        &DONE,
    );
    if outcome.compute_timed_out && !p.compute_dropped.swap(true, Ordering::Relaxed) {
        tracing::info!(
            shaders = ?p.shaders,
            frame,
            waited_ms = outcome.waited.as_millis() as u64,
            budget_secs = COMPUTE_BUDGET.as_secs(),
            "compute dispatch dropped: pipeline was not ready within the frame compute wait budget"
        );
    }
    if outcome.pipeline_timed_out && !p.compute_given_up.swap(true, Ordering::Relaxed) {
        tracing::warn!(
            shaders = ?p.shaders,
            budget_secs = COMPUTE_BUDGET.as_secs(),
            "compute pipeline exceeded per-pipeline wait budget; falling back to the normal frame budget"
        );
    }
}

/// Pipelines for the workers: requested compute first, then requested
/// graphics, then early pre-warms.
static QUEUES: (Mutex<PriorityQueue<Arc<Pending>>>, Condvar) =
    (Mutex::new(PriorityQueue::new()), Condvar::new());

/// Hands `p` to the workers (half the logical processors, 2 to 8), started
/// on first use.
fn queue(p: Arc<Pending>, lane: Lane) {
    static WORKERS: OnceLock<usize> = OnceLock::new();
    let workers = *WORKERS.get_or_init(|| {
        let n = std::thread::available_parallelism().map_or(2, |n| (n.get() / 2).clamp(2, 8));
        (0..n)
            .filter(|i| {
                let spawned = std::thread::Builder::new()
                    .name(format!("victory-pipelines-{i}"))
                    .spawn(work);
                if let Err(e) = &spawned {
                    tracing::warn!(error = %e, "no pipeline worker");
                }
                spawned.is_ok()
            })
            .count()
    });
    if workers == 0 {
        // No worker: made here instead.
        if !p.taken.swap(true, Ordering::Relaxed) {
            p.make();
            WAITING.fetch_sub(1, Ordering::Relaxed);
        }
        return;
    }
    let (lock, work) = &QUEUES;
    lock.lock().unwrap_or_else(|e| e.into_inner()).push(lane, p);
    work.notify_one();
}

/// A worker: makes queued pipelines, those draws wait for first, and saves
/// the cache when idle (first uses of pipelines made before).
fn work() {
    // Below the title's own threads (loading, recording).
    // SAFETY: a plain call on our own thread.
    unsafe {
        use windows::Win32::System::Threading::*;
        let _ = SetThreadPriority(GetCurrentThread(), THREAD_PRIORITY_BELOW_NORMAL);
    }
    let (lock, work) = &QUEUES;
    loop {
        let p = {
            let mut queues = lock.lock().unwrap_or_else(|e| e.into_inner());
            loop {
                if let Some(p) = queues.pop() {
                    break p;
                }
                let (held, waited) = work
                    .wait_timeout(queues, Duration::from_secs(20))
                    .unwrap_or_else(|e| e.into_inner());
                queues = held;
                if waited.timed_out() {
                    drop(queues);
                    super::library::save(false);
                    queues = lock.lock().unwrap_or_else(|e| e.into_inner());
                }
            }
        };
        if p.taken.swap(true, Ordering::Relaxed) {
            continue; // its other queue entry
        }
        p.make();
        if WAITING.fetch_sub(1, Ordering::Relaxed) == 1 {
            drained();
        } else {
            super::library::save(false);
        }
    }
}

/// The most pipelines waiting at once since the workers last ran out.
static PEAK: AtomicU64 = AtomicU64::new(0);

/// The workers ran out of work: logged after a batch of 10 or more, and
/// the cache saved (at most every 20 s).
fn drained() {
    use super::library::{EARLY, LOADED, MADE};
    if PEAK.swap(0, Ordering::Relaxed) >= 10 {
        tracing::info!(
            made = MADE.load(Ordering::Relaxed),
            loaded = LOADED.load(Ordering::Relaxed),
            started_early = EARLY.load(Ordering::Relaxed),
            "host pipelines all made"
        );
    }
    super::library::save(false);
}

#[cfg(test)]
mod tests {
    use super::*;

    /// An early start is queued once; a draw or compute dispatch waiting for
    /// it puts it ahead in its lane, once, unless a worker has taken it.
    #[test]
    fn draws_go_ahead_of_early_starts() {
        let p = Pending::new(
            Vec::new(),
            None,
            #[cfg(feature = "diagnostics")]
            &[],
        );
        assert_eq!(enqueue(&p, false), Some(Lane::Early));
        assert_eq!(enqueue(&p, false), None);
        assert_eq!(enqueue(&p, true), Some(Lane::Graphics));
        assert_eq!(enqueue(&p, true), None);
        let q = Pending::new(
            Vec::new(),
            None,
            #[cfg(feature = "diagnostics")]
            &[],
        );
        assert_eq!(enqueue(&q, false), Some(Lane::Early));
        q.taken.store(true, Ordering::Relaxed);
        assert_eq!(enqueue(&q, true), None);
        let r = Pending::new(
            Vec::new(),
            None,
            #[cfg(feature = "diagnostics")]
            &[],
        );
        assert_eq!(enqueue(&r, true), Some(Lane::Graphics));
        assert_eq!(enqueue(&r, false), None);
        let mut cs = Pending::new(
            Vec::new(),
            None,
            #[cfg(feature = "diagnostics")]
            &[],
        );
        cs.compute = true;
        assert_eq!(enqueue(&cs, false), Some(Lane::Early));
        assert_eq!(enqueue(&cs, true), Some(Lane::Compute));
        assert_eq!(enqueue(&cs, true), None);
        // Counted as waiting by `enqueue`, never handed to a worker.
        WAITING.fetch_sub(4, Ordering::Relaxed);
    }

    /// A first use queues the pipeline and, within the budget, gets the
    /// worker's answer (here: no host pipeline for no shaders).
    #[test]
    fn first_use_is_made_by_a_worker() {
        let p = Arc::new(Pending::new(
            Vec::new(),
            None,
            #[cfg(feature = "diagnostics")]
            &[],
        ));
        assert!(ready(&p).is_none());
        assert!(p.queued.load(Ordering::Relaxed));
        let started = Instant::now();
        while p.made.get().is_none() && started.elapsed() < Duration::from_secs(10) {
            std::thread::yield_now();
        }
        assert!(matches!(p.made.get(), Some(None)));
        #[cfg(feature = "diagnostics")]
        assert_eq!(missing(&p).as_deref(), Some("no vertex or mesh shader"));
    }

    #[cfg(feature = "diagnostics")]
    #[test]
    fn missing_formats_static_refusal_driver_hresult_and_skipped_shader() {
        let shader = Arc::new(Shader {
            stage: "compute",
            hash: "abcdef0123456789".into(),
            full_hash: [0; 32],
            size: 4,
            container: std::sync::Mutex::new(vec![1, 2, 3, 4]),
            translated: OnceLock::new(),
            spilled: OnceLock::new(),
        });

        // 1. Driver refusal keeps HRESULT (hex) and shader identity.
        let driver = Pending::new(vec![shader.clone()], None, &[]);
        driver
            .refusal
            .set(("compute pipeline refused", Some(0x80070057u32 as i32)))
            .unwrap();
        assert_eq!(missing(&driver).as_deref(), Some("being made"));
        assert!(driver.made.set(None).is_ok());
        assert_eq!(
            missing(&driver).as_deref(),
            Some("compute pipeline refused (0x80070057) [compute abcdef0123456789]")
        );

        // 2. Static refusal with shaders includes reason and shader list; without shaders keeps bare reason.
        let static_with = Pending::new(vec![shader.clone()], None, &[]);
        static_with
            .refusal
            .set(("the host GPU has no 64-bit atomics on typed heap resources", None))
            .unwrap();
        assert!(static_with.made.set(None).is_ok());
        assert_eq!(
            missing(&static_with).as_deref(),
            Some("the host GPU has no 64-bit atomics on typed heap resources [compute abcdef0123456789]")
        );

        let static_empty = Pending::new(Vec::new(), None, &[]);
        static_empty
            .refusal
            .set(("no vertex or mesh shader", None))
            .unwrap();
        assert!(static_empty.made.set(None).is_ok());
        assert_eq!(
            missing(&static_empty).as_deref(),
            Some("no vertex or mesh shader")
        );

        // 3. Skipped shader keeps its dedicated label.
        let hashes = crate::pipeline::parse_skip_hashes("ABCDEF0123456789");
        let skipped = Pending::new(vec![shader], None, &hashes);
        assert_eq!(
            missing(&skipped).as_deref(),
            Some("skipped shader abcdef0123456789")
        );
    }

    #[test]
    fn a_pipeline_for_bound_formats_is_made_once() {
        let p = Arc::new(Pending::new(
            Vec::new(),
            Some(Box::new(GraphicsState::declaring(24))),
            #[cfg(feature = "diagnostics")]
            &[],
        ));
        assert_eq!(p.formats, [24, 0, 0, 0, 0, 0, 0, 0, 0]);
        let g = p.graphics.as_deref().unwrap();
        let (a, b) = ([26, 0, 0, 0, 0, 0, 0, 0, 0], [28, 0, 0, 0, 0, 0, 0, 0, 0]);
        let first = variant(&p, g, &a);
        assert_eq!(first.formats, a);
        assert_ne!(first.key, p.key);
        assert!(Arc::ptr_eq(first, variant(&p, g, &a)));
        let second = variant(&p, g, &b);
        assert_eq!(second.formats, b);
        assert!(!Arc::ptr_eq(first, second));
        assert_ne!(second.key, first.key);
        assert!(Arc::ptr_eq(first, variant(&p, g, &a)));
        assert!(Arc::ptr_eq(second, variant(&p, g, &b)));
        assert!(!first.queued.load(Ordering::Relaxed));
        assert!(!p.queued.load(Ordering::Relaxed));
    }

    #[cfg(feature = "diagnostics")]
    #[test]
    fn skipped_pipeline_stays_off_the_queue() {
        let hashes = crate::pipeline::parse_skip_hashes(" 0123456789ABCDEF ");
        let shader = Arc::new(Shader {
            stage: "compute",
            hash: "0123456789abcdef".into(),
            full_hash: [0; 32],
            size: 0,
            container: std::sync::Mutex::new(Vec::new()),
            translated: OnceLock::new(),
            spilled: OnceLock::new(),
        });
        let cold = Arc::new(Pending::new(vec![shader.clone()], None, &hashes));
        let warm = Arc::new(Pending::new(vec![shader.clone()], None, &hashes));
        assert!(ready(&cold).is_none());
        crate::pipeline::library::note_used(&warm.key);
        early(&warm);
        assert_eq!(
            (
                cold.queued.load(Ordering::Relaxed),
                warm.queued.load(Ordering::Relaxed)
            ),
            (false, false)
        );
        assert!(matches!(cold.made.get(), Some(None)));
        assert!(matches!(warm.made.get(), Some(None)));
        assert!(shader.translated.get().is_none());
    }
}
