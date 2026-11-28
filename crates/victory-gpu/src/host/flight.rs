//! Wait-Before-Prep Flight-1 GPU overlap (`VICTORY_FLIGHT1=1`) and opt-in
//! async submission mode (`VICTORY_ASYNC_SUBMIT=1`).
//!
//! At most one GPU submission (`InFlight`) is in flight at a time. While the
//! host GPU executes Flight N, the title's CPU threads immediately return from
//! submission and record Flight N+1 into fresh `HostList`s (`list::restart`
//! sees `Arc::strong_count > 1` and splits off spare lists). Before Flight N+1
//! touches any shared CPU/GPU state (`evict_unused`, `sync_heaps`,
//! `bindless::uploads` / `texture::recycle`, `Shadow::update`, `Mark::arm`),
//! `finish_flight_locked` waits for Flight N's fence, checks its `Mark`,
//! releases worker staging (`Uploads::release_worker`), copies back GPU writes,
//! lands texture readbacks, runs `after` closures and bottom-of-pipe writes,
//! returns the upload list, recycles its `HostList`s, and delivers any attached
//! title fence signals.
//!
//! A background `victory-flight` thread waits on `flight_event` without
//! holding `HELD` or `Host::submitted`, then acquires `Host::submitted` to
//! land the completed flight so fence waiters and BoP pollers wake promptly
//! even when the title submits no further work.

use std::sync::Arc;
use std::sync::atomic::{AtomicU64, Ordering};

use super::{Batch, Holding, Host, HostList, Mark, Step as ListStep};
use crate::object::This;

/// Whether Wait-Before-Prep Flight-1 overlap is enabled (`VICTORY_FLIGHT1=1`,
/// or implicitly enabled when `VICTORY_ASYNC_SUBMIT=1` unless
/// `VICTORY_FLIGHT1=0` is explicitly set; disabled when `VICTORY_GPU_TIMING`
/// is on).
pub(crate) fn flight1_enabled() -> bool {
    #[cfg(test)]
    match TEST_FLIGHT1.load(Ordering::Acquire) {
        1 => return false,
        2 => return true,
        _ => {}
    }
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| {
        #[cfg(feature = "diagnostics")]
        if super::timing() {
            return false;
        }
        let on = match std::env::var("VICTORY_FLIGHT1").as_deref() {
            Ok("1") => true,
            Ok("0") => false,
            _ => async_submit_env(),
        };
        if on {
            tracing::info!(
                "Flight-1 GPU overlap enabled (VICTORY_FLIGHT1=1): wait-before-prep with background flight finisher"
            );
        }
        on
    })
}

/// Whether submissions are queued to the dedicated `victory-submit` thread
/// (`VICTORY_ASYNC_SUBMIT=1`, off by default).
pub(crate) fn async_submit_enabled() -> bool {
    #[cfg(test)]
    match TEST_ASYNC_SUBMIT.load(Ordering::Acquire) {
        1 => return false,
        2 => return true,
        _ => {}
    }
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| {
        let on = async_submit_env();
        if on {
            tracing::info!(
                "Async submission worker enabled (VICTORY_ASYNC_SUBMIT=1): submissions prepared and launched on victory-submit"
            );
        }
        on
    })
}

fn async_submit_env() -> bool {
    std::env::var_os("VICTORY_ASYNC_SUBMIT").is_some_and(|v| v == "1")
}

#[cfg(test)]
static TEST_FLIGHT1: std::sync::atomic::AtomicU8 = std::sync::atomic::AtomicU8::new(0);
#[cfg(test)]
static TEST_ASYNC_SUBMIT: std::sync::atomic::AtomicU8 = std::sync::atomic::AtomicU8::new(0);
#[cfg(test)]
static TEST_PREP_AHEAD: std::sync::atomic::AtomicU8 = std::sync::atomic::AtomicU8::new(0);
#[cfg(test)]
static TEST_MODE_LOCK: std::sync::Mutex<()> = std::sync::Mutex::new(());

#[cfg(test)]
pub(crate) struct ModeGuard {
    _lock: std::sync::MutexGuard<'static, ()>,
    prev_flight1: u8,
    prev_async: u8,
    prev_ahead: u8,
}

#[cfg(test)]
impl ModeGuard {
    pub(crate) fn new(flight1: bool, async_submit: bool) -> Self {
        Self::with_prep_ahead(flight1, async_submit, false)
    }

    pub(crate) fn with_prep_ahead(flight1: bool, async_submit: bool, ahead: bool) -> Self {
        Self::with_mode(flight1, async_submit, if ahead { 2 } else { 1 })
    }

    /// Flight-1 with `VICTORY_PREP_AHEAD=ranges`.
    pub(crate) fn with_prep_ranges() -> Self {
        Self::with_mode(true, false, 3)
    }

    fn with_mode(flight1: bool, async_submit: bool, ahead: u8) -> Self {
        let lock = TEST_MODE_LOCK.lock().unwrap_or_else(|e| e.into_inner());
        crate::queue::drain_submit(crate::queue::Why::Present);
        if let Some(h) = super::host() {
            h.finish_flight_now();
        }
        let prev_flight1 = TEST_FLIGHT1.swap(if flight1 { 2 } else { 1 }, Ordering::AcqRel);
        let prev_async = TEST_ASYNC_SUBMIT.swap(if async_submit { 2 } else { 1 }, Ordering::AcqRel);
        let prev_ahead = TEST_PREP_AHEAD.swap(ahead, Ordering::AcqRel);
        Self {
            _lock: lock,
            prev_flight1,
            prev_async,
            prev_ahead,
        }
    }
}

#[cfg(test)]
impl Drop for ModeGuard {
    fn drop(&mut self) {
        crate::queue::drain_submit(crate::queue::Why::Present);
        if let Some(h) = super::host() {
            h.finish_flight_now();
        }
        TEST_PAUSE_LANDING.store(false, Ordering::Release);
        crate::queue::drain_submit(crate::queue::Why::Present);
        if let Some(h) = super::host() {
            h.finish_flight_now();
        }
        TEST_FLIGHT1.store(self.prev_flight1, Ordering::Release);
        TEST_ASYNC_SUBMIT.store(self.prev_async, Ordering::Release);
        TEST_PREP_AHEAD.store(self.prev_ahead, Ordering::Release);
    }
}

/// Before a flight is given to the host GPU: the pages its shaders may
/// write with no list declaring them are armed (`Shadow::arm_undeclared`),
/// the unordered views of its heaps (the bindless path) and every copy the
/// page tables of its title-memory programs name. Their writes are marked
/// and go back by the pages marked (`Shadow::copy_back_marked`), and a page
/// the title wrote since the copy took it, and that nothing saved, keeps
/// the title's bytes whole, the shaders' there dropped. Waited for, the
/// title's thread that submits cannot write meanwhile; a flight lets it go
/// on, and it writes a frame's data into pages the GPU writes too (on the
/// console both land, byte by byte). Armed, the first title write to such
/// a page saves its bytes from before, and the copy-back keeps both. The
/// copies armed, to disarm when it lands.
fn arm_undeclared(
    segments: &[Arc<HostList>],
    heaps: &[Arc<crate::bindless::Viewed>],
) -> Vec<Arc<crate::shadow::Shadow>> {
    let mut armed: Vec<Arc<crate::shadow::Shadow>> = Vec::new();
    let mut seen = std::collections::HashSet::new();
    let mut arm = |copy: &Arc<crate::shadow::Shadow>, address: usize, len: usize| {
        copy.arm_undeclared(address, len);
        if seen.insert(Arc::as_ptr(copy) as usize) {
            armed.push(copy.clone());
        }
    };
    for heap in heaps {
        crate::bindless::written_ranges(heap, &mut arm);
    }
    for list in segments {
        for step in &list.deferred().copies {
            if let &ListStep::Memory(key) = step {
                crate::memory::copies(key, |copy| {
                    let (start, len) = copy.range();
                    arm(copy, start as usize, len as usize);
                });
            }
        }
    }
    armed
}

/// Tests: the host queue waits on a fence of ours from [`Gate::hold`] to
/// [`Gate::open`] (or the drop): a flight given meanwhile stays in the air.
/// A watchdog opens it after 2 s: another test's submission behind the gate
/// holds the submission lock while it waits, and the thread that would open
/// it may be waiting for that lock.
#[cfg(test)]
pub(crate) struct Gate {
    fence: GateFence,
    opened: Arc<std::sync::atomic::AtomicBool>,
}

#[cfg(test)]
#[derive(Clone)]
struct GateFence(windows::Win32::Graphics::Direct3D12::ID3D12Fence);

// SAFETY: D3D12 fences are free-threaded.
#[cfg(test)]
unsafe impl Send for GateFence {}

#[cfg(test)]
impl GateFence {
    fn open(&self, opened: &std::sync::atomic::AtomicBool) {
        if !opened.swap(true, Ordering::AcqRel) {
            // SAFETY: our fence.
            unsafe { self.0.Signal(1) }.expect("the gate opens");
        }
    }
}

#[cfg(test)]
impl Gate {
    pub(crate) fn hold(host: &Host) -> Gate {
        // SAFETY: plain creation; our queue waits on our fence.
        let fence = unsafe {
            let fence: windows::Win32::Graphics::Direct3D12::ID3D12Fence = host
                .device
                .CreateFence(
                    0,
                    windows::Win32::Graphics::Direct3D12::D3D12_FENCE_FLAG_NONE,
                )
                .expect("a gate fence");
            host.queue()
                .Wait(&fence, 1)
                .expect("the queue waits on the gate");
            GateFence(fence)
        };
        let opened = Arc::new(std::sync::atomic::AtomicBool::new(false));
        let (f, o) = (fence.clone(), opened.clone());
        std::thread::spawn(move || {
            std::thread::sleep(std::time::Duration::from_secs(2));
            f.open(&o);
        });
        Gate { fence, opened }
    }

    pub(crate) fn open(&mut self) {
        self.fence.open(&self.opened);
    }
}

#[cfg(test)]
impl Drop for Gate {
    fn drop(&mut self) {
        self.fence.open(&self.opened);
    }
}

/// Protected by `Host::flight` (leaf lock strictly below `HELD` and
/// `Host::submitted`, above only `Fence::waiters`).
#[derive(Default)]
pub(super) struct FlightState {
    /// The launched flights awaiting wait + landing, oldest first: one, or
    /// with `VICTORY_PREP_AHEAD=1` two. Each is taken once, the oldest
    /// first, by whichever thread lands it under `Host::submitted`.
    flights: std::collections::VecDeque<InFlight>,
    /// True while a flight taken from `flights` lands (its copy-back, BoP
    /// writes and signals): `attach_signal` then queues behind it.
    landing: bool,
    /// Title fences queued behind the landing flight (each kept alive via
    /// `fence::keep`), delivered when it has landed.
    landing_signals: Vec<(usize, u64)>,
    /// Whether the background `victory-flight` finisher thread has been spawned.
    worker_started: bool,
    /// When the host GPU last finished our work (a landed flight's fence, or
    /// a synchronous submission's end) and nothing has been launched since:
    /// the start of the GPU's idle time before the next flight.
    free_since: Option<std::time::Instant>,
}

/// Microseconds the flights launched since the last frame report spent in
/// each step ([`take_flight_times`], in [`Step`]'s order).
static FLIGHT_US: [AtomicU64; 5] = [const { AtomicU64::new(0) }; 5];

/// Flights launched since the last frame report.
static FLIGHTS: AtomicU64 = AtomicU64::new(0);

/// What [`FLIGHT_US`] counts, per flight.
#[derive(Clone, Copy)]
enum Step {
    /// From asking to launch to the flight before it having landed (its GPU
    /// time left and its copy-back), when it had to.
    Landing,
    /// Preparing: refreshing host copies and textures, heaps, preludes (and,
    /// prepared ahead, landing the flight before it where something needed it).
    Preparing,
    /// The host `ExecuteCommandLists` and `Signal` calls.
    Launching,
    /// The GPU's idle time before the flight while no work was there to run.
    IdleUnasked,
    /// The GPU's idle time before the flight while it was being prepared.
    IdlePreparing,
}

fn add_step(step: Step, d: std::time::Duration) {
    FLIGHT_US[step as usize].fetch_add(d.as_micros() as u64, Ordering::Relaxed);
}

/// The GPU's idle time before a flight launched at `launched`, split at
/// when its work was there: from `free` (the GPU finished our last work)
/// to `asked` (the submit worker began this flight) nothing waited to run;
/// from then (or from `free`, if the worker was already waiting for the
/// last flight to land) to `launched` the flight was being prepared.
fn idle_split(
    free: std::time::Instant,
    asked: std::time::Instant,
    launched: std::time::Instant,
) -> (std::time::Duration, std::time::Duration) {
    let ready = asked.max(free);
    (
        ready.saturating_duration_since(free),
        launched.saturating_duration_since(ready),
    )
}

/// The flights launched since the last call, and per [`Step`] their
/// microseconds, cleared (the frame report).
pub(crate) fn take_flight_times() -> (u64, [u64; 5]) {
    (
        FLIGHTS.swap(0, Ordering::Relaxed),
        FLIGHT_US.each_ref().map(|a| a.swap(0, Ordering::Relaxed)),
    )
}

/// Why a flight prepared ahead (`VICTORY_PREP_AHEAD=1`) landed the one in
/// the air before it launched ([`land_ahead`]); [`AHEAD`] counts them.
#[derive(Clone, Copy, Debug)]
pub(crate) enum Early {
    /// The flight in the air was not quiet at landing: its copy-back, marks,
    /// work after it or uploads change what a preparation reads. Landed
    /// before preparing (as without the switch).
    NotQuiet,
    /// The frame's eviction pass gives up host objects.
    Eviction,
    /// Unordered views are made again.
    Remake,
    /// The shared UAV barrier lists are needed.
    Barriers,
    /// A shader-visible heap is made and filled (`Host::quiet`).
    Quiet,
    /// The new flight's title copies, done on the CPU while preparing.
    TitleCopy,
    /// The new flight's shaders mark what they write: the landing of the one
    /// before takes the marks, so it lands before the new one runs.
    Marks,
    /// No second mark could be made.
    NoMark,
    /// `VICTORY_PREP_AHEAD=ranges`: the new flight refreshes or writes host
    /// copy bytes the one in the air writes (or any of a copy its shaders
    /// may mark).
    Overlap,
    /// `VICTORY_PREP_AHEAD=ranges`: the new flight reads title memory on
    /// the CPU (a fill from what the one in the air writes, or host work).
    CpuRead,
}

/// [`AHEAD`]'s slots: flights prepared while one was in the air, of them
/// launched while it still was, then one per [`Early`].
const AHEAD_SLOTS: usize = 2 + Early::CpuRead as usize + 1;

/// Fence-wait microseconds incurred by preparation, indexed by the early
/// landing reason. Copy-back and callbacks after the fence are excluded.
#[cfg(feature = "diagnostics")]
static AHEAD_WAIT_US: [AtomicU64; AHEAD_SLOTS - 2] = [const { AtomicU64::new(0) }; AHEAD_SLOTS - 2];

/// Preparations behind a flight that was not quiet but whose writes were
/// known (`VICTORY_PREP_AHEAD=ranges`), since the last frame report.
static BEHIND_BY_RANGES: AtomicU64 = AtomicU64::new(0);

/// Counts since the last frame report ([`AHEAD_SLOTS`]).
static AHEAD: [AtomicU64; AHEAD_SLOTS] = [const { AtomicU64::new(0) }; AHEAD_SLOTS];

thread_local! {
    /// While this thread prepares a flight behind one in the air: the host
    /// and its submission counter (behind the `Host::submitted` lock this
    /// thread holds), for [`land_ahead`].
    static PREPARING_AHEAD: std::cell::Cell<Option<(*const Host, *mut u64)>> =
        const { std::cell::Cell::new(None) };
    /// While this thread prepares a flight behind one that is not quiet
    /// (`VICTORY_PREP_AHEAD=ranges`): what that one writes, which the
    /// preparation must not read or rewrite before it lands.
    static AHEAD_OF: std::cell::RefCell<Option<Writes>> = const { std::cell::RefCell::new(None) };
}

/// What a flight writes that a preparation behind it must not read early
/// (`VICTORY_PREP_AHEAD=ranges`): known before it runs, unlike the pages
/// its shaders mark, which only their bound (the copies owning mark bits)
/// is.
#[derive(Clone, Default)]
pub(crate) struct Writes {
    /// Title memory, as sorted, merged `[from, to)`: the ranges its shaders
    /// declared (whole 64 KiB pages: the trackers take pages), its
    /// bottom-of-pipe writes, and every copy its shaders may mark.
    spans: Vec<(usize, usize)>,
    /// Host copies (by address) none of whose bytes a preparation may
    /// refresh: those its shaders may mark.
    whole: std::collections::HashSet<usize>,
    /// Host copies its shaders write in declared ranges: a preparation may
    /// refresh other ranges of them, but not declare writes of its own
    /// there (one snapshot, one pending set a copy).
    written: std::collections::HashSet<usize>,
    /// Whether its shaders mark what they write.
    marks: bool,
}

impl Writes {
    /// Adds title range `[from, to)` (unsorted until [`Self::seal`]).
    pub(crate) fn span(&mut self, from: usize, to: usize) {
        if from < to {
            self.spans.push((from, to));
        }
    }

    /// Adds a declared write of copy `copy` at `[from, to)`.
    pub(crate) fn declared(&mut self, copy: usize, from: usize, to: usize) {
        let page = crate::watch::PAGE;
        self.written.insert(copy);
        self.span(from / page * page, to.div_ceil(page).saturating_mul(page));
    }

    /// Adds every byte of copy `copy` at `[from, to)`.
    pub(crate) fn whole(&mut self, copy: usize, from: usize, to: usize) {
        self.whole.insert(copy);
        self.span(from, to);
    }

    /// The flight's shaders mark what they write.
    pub(crate) fn marks(&mut self) {
        self.marks = true;
    }

    /// Sorts and merges the spans.
    fn seal(&mut self) {
        self.spans.sort_unstable();
        let mut merged: Vec<(usize, usize)> = Vec::with_capacity(self.spans.len());
        for &(from, to) in &self.spans {
            match merged.last_mut() {
                Some(last) if from <= last.1 => last.1 = last.1.max(to),
                _ => merged.push((from, to)),
            }
        }
        self.spans = merged;
    }

    /// Whether `[from, to)` shares a byte with the spans (sealed).
    fn overlaps(&self, from: usize, to: usize) -> bool {
        let i = self.spans.partition_point(|&(_, end)| end <= from);
        self.spans.get(i).is_some_and(|&(start, _)| start < to && from < to)
    }

    /// For the preparation behind the flight: sealed, with every copy its
    /// shaders may mark (as of now: a copy made later has no bits the
    /// flight's shaders could mark).
    fn behind(&self) -> Self {
        let mut w = self.clone();
        if w.marks {
            for (copy, from, len) in crate::shadow::mark_owner_extents() {
                w.whole(copy, from, from + len);
            }
        }
        w.seal();
        w
    }
}

/// Before a preparation refreshes `[address, address + len)` of copy
/// `copy`: lands the flight in the air first if it writes there.
pub(crate) fn before_refresh(copy: usize, address: usize, len: usize) {
    let hit = AHEAD_OF.with(|w| {
        w.borrow().as_ref().is_some_and(|w| {
            w.whole.contains(&copy) || w.overlaps(address, address.saturating_add(len))
        })
    });
    if hit {
        land_ahead(Early::Overlap);
    }
}

/// Before a preparation declares that its shaders write
/// `[address, address + len)` of copy `copy` (`Shadow::writing`).
pub(crate) fn before_gpu_write(copy: usize, address: usize, len: usize) {
    let hit = AHEAD_OF.with(|w| {
        w.borrow().as_ref().is_some_and(|w| {
            w.whole.contains(&copy)
                || w.written.contains(&copy)
                || w.overlaps(address, address.saturating_add(len))
        })
    });
    if hit {
        land_ahead(Early::Overlap);
    }
}

/// Before a preparation reads title memory on the CPU: `[address, address
/// + len)`, or anything (`None`, host work).
pub(crate) fn before_cpu_read(range: Option<(usize, usize)>) {
    let hit = AHEAD_OF.with(|w| {
        w.borrow().as_ref().is_some_and(|w| {
            range.is_none_or(|(address, len)| w.overlaps(address, address.saturating_add(len)))
        })
    });
    if hit {
        land_ahead(Early::CpuRead);
    }
}

/// Lands the flights in the air first if this thread prepares one behind
/// them and `why` needs the GPU done with them; nothing otherwise.
pub(crate) fn land_ahead(why: Early) {
    let Some((host, submitted)) = PREPARING_AHEAD.with(std::cell::Cell::get) else {
        return;
    };
    // SAFETY: set by `execute_flight1` around its preparation, on this
    // thread, which holds the lock `submitted` is behind and does not
    // otherwise touch it until it clears the cell.
    let (host, submitted) = unsafe { (&*host, &mut *submitted) };
    if host.has_in_flight() {
        AHEAD[2 + why as usize].fetch_add(1, Ordering::Relaxed);
        PREPARING_AHEAD.with(|c| c.set(None));
        AHEAD_OF.with(|w| w.borrow_mut().take());
        host.finish_flight_locked_for(submitted, None, None, Some(why));
    }
}

/// Whether a submission may be prepared while the one before it is in
/// flight (`VICTORY_PREP_AHEAD=1`, with Flight-1; off by default).
///
/// It is when the flight in the air is quiet at landing
/// ([`HostList::quiet_at_landing`]): its landing then changes nothing the
/// preparation reads. The preparation refreshes host copies in place while
/// that flight's shaders read them: as on the console, the title does not
/// change memory submitted work reads until that work's fence moves, which
/// is at its landing, so a refresh rewrites what the flight reads with the
/// same bytes. Whatever else needs the GPU done with it lands it first
/// ([`land_ahead`], [`Early`]): the frame's eviction, a view remake, the
/// shared barrier lists, a heap fill, title copies (done on the CPU here,
/// on the console by the GPU after the flight), and, before launching, the
/// new flight's write marks (the landing takes the marks bitmaps whole).
pub(crate) fn prep_ahead_enabled() -> bool {
    #[cfg(test)]
    match TEST_PREP_AHEAD.load(Ordering::Acquire) {
        1 => return false,
        2 | 3 => return flight1_enabled(),
        _ => {}
    }
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| {
        let on = flight1_enabled()
            && std::env::var_os("VICTORY_PREP_AHEAD").is_some_and(|v| v == "1" || v == "ranges");
        if on {
            tracing::info!(
                ranges = prep_ranges_enabled(),
                "host submissions prepared while the one before is in flight (VICTORY_PREP_AHEAD)"
            );
        }
        on
    })
}

/// `VICTORY_PREP_AHEAD=ranges` (off by default): a submission is also
/// prepared behind a flight that is not quiet at landing, when what makes
/// it so is known before it runs ([`HostList::ahead_writes`]): ranges its
/// shaders declare, write marks (bound by the copies owning mark bits) and
/// bottom-of-pipe writes outside host copies. The preparation then lands
/// it first only where it would read or rewrite those ([`Writes`],
/// [`Early::Overlap`], [`Early::CpuRead`]).
///
/// Its landing writes title memory the preparation did not read, and puts
/// nothing into host copies: what its shaders wrote is in them already,
/// and a refresh of other bytes of a copy touches neither its snapshot nor
/// its pending ranges (`Shadow::update_ranges` syncs only bytes in pending
/// ranges into the snapshot).
///
/// shortcut: a texture over memory the flight's shaders write is brought up
/// from title memory, which its landing writes: a texture the preparation
/// brings up there is the bytes from before the flight, one submission late,
/// as one written in the same submission already is (`bindless::uploads`).
/// `VICTORY_SYNC_EARLY=1` (off by default): before a submission waits for
/// the one in flight to land (Wait-Before-Prep), the descriptors the title
/// wrote into its heaps are mirrored (`descriptor::sync_heaps`, ~24 ms a
/// frame in the world, run I) while that one still runs on the GPU, so the
/// submit worker waits that much less after it. The preparation syncs again
/// after the landing, for what the title wrote meanwhile and the views the
/// landing made (`descriptor::make_read`), which then costs little.
///
/// As with `VICTORY_PREP_AHEAD=1`, which syncs heaps behind a quiet flight:
/// on the console the title does not change descriptors that work it
/// submitted reads until that work's fence moves, which is at its landing,
/// so the slots a sync rewrites are none the flight in the air reads.
pub(crate) fn sync_early_enabled() -> bool {
    #[cfg(test)]
    if let Some(on) = TEST_SYNC_EARLY.with(std::cell::Cell::get) {
        return on;
    }
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| {
        let on = crate::switch::on("VICTORY_SYNC_EARLY");
        if on {
            tracing::info!("heap views synced while the last host submission runs (VICTORY_SYNC_EARLY=1)");
        }
        on
    })
}

#[cfg(test)]
thread_local! {
    /// [`sync_early_enabled`] on this test's thread.
    static TEST_SYNC_EARLY: std::cell::Cell<Option<bool>> = const { std::cell::Cell::new(None) };
}

/// Heap syncs done while the last flight ran ([`sync_early_enabled`]) and
/// their microseconds, since the last frame report.
static SYNCED_EARLY: [AtomicU64; 2] = [const { AtomicU64::new(0) }; 2];

pub(crate) fn prep_ranges_enabled() -> bool {
    #[cfg(test)]
    match TEST_PREP_AHEAD.load(Ordering::Acquire) {
        3 => return flight1_enabled(),
        1 | 2 => return false,
        _ => {}
    }
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| {
        flight1_enabled() && std::env::var_os("VICTORY_PREP_AHEAD").is_some_and(|v| v == "ranges")
    })
}

/// The frame report's line for asynchronous submission, per frame over
/// `frames` frames: nothing unless a flight was launched.
pub(crate) fn log_frame(frames: f64) {
    let (flights, us) = take_flight_times();
    let worker = crate::queue::take_worker_us();
    let ahead = AHEAD.each_ref().map(|a| a.swap(0, Ordering::Relaxed));
    #[cfg(feature = "diagnostics")]
    let ahead_wait = AHEAD_WAIT_US
        .each_ref()
        .map(|a| a.swap(0, Ordering::Relaxed));
    if flights == 0 {
        return;
    }
    let ms = |us: u64| us as f64 / 1e3 / frames;
    let [landing, preparing, launching, unasked, idle_preparing] = us.map(ms);
    tracing::info!(
        "host submissions in flight per frame: {:.1}; ms per frame: submit worker busy {:.1} (waiting for the last \
         to land {landing:.1}, preparing {preparing:.1}, launching {launching:.1}); host GPU idle before them \
         {:.1} (nothing to run {unasked:.1}, preparing {idle_preparing:.1})",
        flights as f64 / frames,
        ms(worker),
        unasked + idle_preparing,
    );
    let [early, early_us] = SYNCED_EARLY.each_ref().map(|a| a.swap(0, Ordering::Relaxed));
    if sync_early_enabled() {
        tracing::info!(
            "heap views synced while the last host submission ran, per frame: {:.1} syncs, {:.1} ms \
             (VICTORY_SYNC_EARLY=1)",
            early as f64 / frames,
            early_us as f64 / 1e3 / frames,
        );
    }
    if prep_ahead_enabled() {
        let per = |n: u64| n as f64 / frames;
        let p = PROBE.each_ref().map(|a| a.swap(0, Ordering::Relaxed));
        let r = &p[Probe::Optimistic as usize + 1..];
        tracing::info!(
            "prep-ahead probe per frame: flights not quiet at landing, for declared ranges {:.1}, write marks {:.1}, \
             work after {:.1}, bottom-of-pipe writes {:.1}, read-backs {:.1}, table textures {:.1}, uploads {:.1}; \
             preparations right behind one {:.1}: behind one not quiet only for ranges/marks/BoP {:.1}; refreshing \
             nothing it declared {:.1}, nothing it wrote {:.1}, nothing a view write reaches {:.1}; could go ahead \
             by ranges {:.1} (if marks were known ahead {:.1})",
            per(r[0]), per(r[1]), per(r[2]), per(r[3]), per(r[4]), per(r[5]), per(r[6]),
            per(p[Probe::Behind as usize]),
            per(p[Probe::RangesOnly as usize]),
            per(p[Probe::CleanKnown as usize]),
            per(p[Probe::CleanActual as usize]),
            per(p[Probe::CleanBound as usize]),
            per(p[Probe::Ceiling as usize]),
            per(p[Probe::Optimistic as usize]),
        );
        tracing::info!(
            "host submissions prepared while the last was in flight per frame: {:.1}, launched while it still \
             was {:.1}; landed it first for: not quiet at landing {:.1}, eviction {:.1}, view remake {:.1}, \
             barrier lists {:.1}, heap fill {:.1}, title copies {:.1}, marks of the new one {:.1}, no second \
             mark {:.1}; behind one not quiet by ranges {:.1} (VICTORY_PREP_AHEAD=ranges), landed it first for \
             what it writes {:.1}, CPU reads {:.1}",
            per(ahead[0]),
            per(ahead[1]),
            per(ahead[2]),
            per(ahead[3]),
            per(ahead[4]),
            per(ahead[5]),
            per(ahead[6]),
            per(ahead[7]),
            per(ahead[8]),
            per(ahead[9]),
            per(BEHIND_BY_RANGES.swap(0, Ordering::Relaxed)),
            per(ahead[10]),
            per(ahead[11]),
        );
        #[cfg(feature = "diagnostics")]
        {
            let title_copy = ahead_wait[Early::TitleCopy as usize];
            let buffer_overlap = ahead_wait[Early::Overlap as usize];
            let cpu_read = ahead_wait[Early::CpuRead as usize];
            let other = ahead_wait.iter().sum::<u64>() - title_copy - buffer_overlap - cpu_read;
            tracing::info!(
                "prep-ahead GPU fence wait ms/frame: title copies {:.2}, overlapping buffer work {:.2}, \
                 CPU reads {:.2}, other {:.2}",
                ms(title_copy),
                ms(buffer_overlap),
                ms(cpu_read),
                ms(other),
            );
        }
    }
}


/// Why a landed flight was not quiet at landing, as bits of the probe below.
#[derive(Clone, Copy)]
pub(crate) enum NotQuiet {
    /// Ranges its shaders write, declared before it ran.
    Declared,
    /// Its shaders mark what they write through views (known only after).
    Marks,
    /// Work after it (title memory written at computed addresses).
    After,
    /// Bottom-of-pipe writes.
    BottomOfPipe,
    /// Texture read-backs.
    ReadBack,
    /// Table textures it brought up to date.
    Textures,
    /// Heap uploads of its own (their staging is rewound by the next).
    Uploads,
}

const NOT_QUIET: usize = NotQuiet::Uploads as usize + 1;

/// The `VICTORY_PREP_AHEAD` probe's counts since the last frame report:
/// [`Probe`]'s slots, then per [`NotQuiet`] reason the flights that had it.
static PROBE: [AtomicU64; Probe::Optimistic as usize + 1 + NOT_QUIET] =
    [const { AtomicU64::new(0) }; Probe::Optimistic as usize + 1 + NOT_QUIET];

/// What the probe counts of preparations right behind a flight that was not
/// quiet at landing (prepared after it landed, as Wait-Before-Prep does).
#[derive(Clone, Copy)]
enum Probe {
    /// Such preparations.
    Behind,
    /// Of them, behind a flight not quiet only for what a range rule can
    /// check: declared ranges, write marks, bottom-of-pipe writes.
    RangesOnly,
    /// Of them, refreshing nothing in the ranges the flight declared or
    /// wrote at its bottom of pipe (known before it lands).
    CleanKnown,
    /// Of them, refreshing nothing in what the flight actually wrote,
    /// marked pages included (known only once it landed).
    CleanActual,
    /// [`Self::CleanKnown`], and, behind a flight marking its writes,
    /// refreshing no host copy any view writes can reach.
    CleanBound,
    /// [`Self::RangesOnly`] and [`Self::CleanBound`]: what a range rule
    /// could prepare behind its flight.
    Ceiling,
    /// [`Self::RangesOnly`] and [`Self::CleanActual`]: the same, if what a
    /// flight marks were known before it ran.
    Optimistic,
}

/// A flight as it landed, for the probe: its fence, why it was not quiet,
/// the ranges it wrote (declared and bottom-of-pipe), and the pages its
/// shaders marked.
struct Landed {
    fence: u64,
    reasons: u8,
    wrote: Vec<(usize, usize)>,
    marked: Vec<(usize, usize)>,
}

/// What the landing in progress wrote, filled by [`probe_wrote`] and
/// [`probe_marked`] (one landing at a time, under `Host::submitted`).
/// Title ranges as start and length.
type Spans = Vec<(usize, usize)>;
static PROBE_SINK: std::sync::Mutex<(Spans, Spans)> = std::sync::Mutex::new((Vec::new(), Vec::new()));
static LAST_LANDED: std::sync::Mutex<Option<Landed>> = std::sync::Mutex::new(None);

thread_local! {
    /// While a flight is prepared with the probe on: the host-copy ranges
    /// its refreshes rewrite (copy, address, bytes).
    static PROBE_ASKED: std::cell::RefCell<Option<Vec<(usize, usize, usize)>>> =
        const { std::cell::RefCell::new(None) };
}

/// A refresh of a written copy's range (`crate::host::Batch::add`).
pub(crate) fn probe_refresh(copy: usize, address: usize, len: usize) {
    PROBE_ASKED.with(|p| {
        if let Some(asked) = p.borrow_mut().as_mut() {
            asked.push((copy, address, len));
        }
    });
}

/// A landing begins writing back.
pub(crate) fn probe_landing_starts() {
    if prep_ahead_enabled() {
        let mut sink = PROBE_SINK.lock().unwrap_or_else(|e| e.into_inner());
        sink.0.clear();
        sink.1.clear();
    }
}

/// The landing writes `[address, address + len)` of title memory: a range
/// shaders declared, or a bottom-of-pipe write.
pub(crate) fn probe_wrote(address: usize, len: usize) {
    if prep_ahead_enabled() {
        PROBE_SINK.lock().unwrap_or_else(|e| e.into_inner()).0.push((address, len));
    }
}

/// The landing writes back a page shaders marked.
pub(crate) fn probe_marked(address: usize, len: usize) {
    if prep_ahead_enabled() {
        PROBE_SINK.lock().unwrap_or_else(|e| e.into_inner()).1.push((address, len));
    }
}

/// Whether any of the half-open `a` ranges overlaps any of `b` (each as
/// start and length), sorting both.
fn overlaps(a: &mut [(usize, usize)], b: &mut [(usize, usize)]) -> bool {
    a.sort_unstable();
    b.sort_unstable();
    let (mut i, mut j) = (0, 0);
    while i < a.len() && j < b.len() {
        let (a0, a1) = (a[i].0, a[i].0.saturating_add(a[i].1));
        let (b0, b1) = (b[j].0, b[j].0.saturating_add(b[j].1));
        if a0 < b1 && b0 < a1 && a1 > a0 && b1 > b0 {
            return true;
        }
        if a1 <= b1 { i += 1 } else { j += 1 }
    }
    false
}

/// Counts the preparation that just refreshed `asked` against `landed`, the
/// flight that landed right before it.
fn probe_compare(landed: &Landed, asked: &[(usize, usize, usize)]) {
    use NotQuiet as N;
    let bit = |r: N| landed.reasons & (1 << r as u8) != 0;
    let ranges_only = landed.reasons & !((1 << N::Declared as u8) | (1 << N::Marks as u8) | (1 << N::BottomOfPipe as u8)) == 0;
    let mut refreshed: Vec<(usize, usize)> = asked.iter().map(|&(_, a, n)| (a, n)).collect();
    let clean_known = !overlaps(&mut refreshed, &mut landed.wrote.clone());
    let mut actual: Vec<(usize, usize)> = landed.wrote.iter().chain(&landed.marked).copied().collect();
    let clean_actual = !overlaps(&mut refreshed, &mut actual);
    let clean_bound = clean_known && (!bit(N::Marks) || {
        let owners = crate::shadow::mark_owners();
        !asked.iter().any(|(copy, ..)| owners.contains(copy))
    });
    let count = |p: Probe, on: bool| {
        if on {
            PROBE[p as usize].fetch_add(1, Ordering::Relaxed);
        }
    };
    count(Probe::Behind, true);
    count(Probe::RangesOnly, ranges_only);
    count(Probe::CleanKnown, clean_known);
    count(Probe::CleanActual, clean_actual);
    count(Probe::CleanBound, clean_bound);
    count(Probe::Ceiling, ranges_only && clean_bound);
    count(Probe::Optimistic, ranges_only && clean_actual);
}

#[cfg(test)]
pub(super) fn probe_counts() -> Vec<u64> {
    PROBE.iter().map(|a| a.load(Ordering::Relaxed)).collect()
}

struct InFlight {
    fence_value: u64,
    segments: Vec<Arc<HostList>>,
    /// Copies whose pages it armed for its shaders' undeclared writes
    /// ([`arm_undeclared`]), disarmed once it has landed.
    armed: Vec<Arc<crate::shadow::Shadow>>,
    uploads: Option<crate::bindless::Uploads>,
    batch: Batch,
    heaps: Vec<Arc<crate::bindless::Viewed>>,
    split_count: usize,
    started: std::time::Instant,
    prepared: std::time::Instant,
    launched: std::time::Instant,
    waited: bool,
    /// Whether its landing changes nothing a later preparation reads: no
    /// uploads of its own, and every list quiet at landing.
    quiet: bool,
    /// Title fences attached behind it.
    signals: Vec<(usize, u64)>,
    /// Not quiet, but what its landing writes is known
    /// (`VICTORY_PREP_AHEAD=ranges`).
    writes: Option<Writes>,
}

// SAFETY: Every field is `Send` (D3D12 objects are free-threaded; `HostList`
// and `Uploads` — including its owned upload list and `ReadyDrain` — are
// `Send`; `Batch` holds `Arc<Shadow>` and ranges).
unsafe impl Send for InFlight {}

impl Host {
    /// Runs `segments` with Flight-1 overlap when enabled: prepares and
    /// launches them on the host queue and returns while the GPU executes.
    /// Wait-Before-Prep: the flight in the air lands first, unless
    /// `VICTORY_PREP_AHEAD=1` lets this one be prepared behind it
    /// ([`prep_ahead_enabled`]), at most two in flight.
    /// Falls back to synchronous execution when Flight-1 is disabled or when
    /// per-submission diagnostics tracing is active on this frame.
    pub(crate) fn execute_flight1(&self, segments: Vec<Arc<HostList>>, barrier_before: Vec<bool>) {
        if segments.is_empty() {
            return;
        }
        #[cfg(feature = "diagnostics")]
        let traced = crate::capture::listed(crate::queue::presented() + 1);
        #[cfg(not(feature = "diagnostics"))]
        let traced = false;
        if !flight1_enabled() || traced {
            self.execute_sync_segments(segments, barrier_before);
            return;
        }

        let asked = std::time::Instant::now();
        let mut submitted = self.submitted.lock().unwrap_or_else(|e| e.into_inner());
        let _holding = Holding::enter();
        let ahead = prep_ahead_enabled();
        if ahead {
            // At most two in flight: the older of two lands first.
            let front = {
                let state = self.flight.lock().unwrap_or_else(|e| e.into_inner());
                (state.flights.len() >= 2).then(|| state.flights[0].fence_value)
            };
            if let Some(v) = front {
                self.finish_flight_locked(&mut submitted, Some(v), None);
            }
        }
        // Behind one quiet at landing, or (`VICTORY_PREP_AHEAD=ranges`) one
        // whose writes are known: what the preparation must keep off.
        let (behind, ahead_of) = if ahead {
            let state = self.flight.lock().unwrap_or_else(|e| e.into_inner());
            match state.flights.front() {
                Some(f) if state.flights.len() == 1 && f.quiet => (true, None),
                Some(f) if state.flights.len() == 1 => match &f.writes {
                    Some(w) => (true, Some(w.behind())),
                    None => (false, None),
                },
                _ => (false, None),
            }
        } else {
            (false, None)
        };
        if !behind {
            if ahead && self.has_in_flight() {
                AHEAD[2 + Early::NotQuiet as usize].fetch_add(1, Ordering::Relaxed);
            }
            // `VICTORY_SYNC_EARLY=1`: what the title wrote into its heaps is
            // mirrored while Flight N-1 still runs, not after it landed.
            if sync_early_enabled() && self.has_in_flight() {
                let started = std::time::Instant::now();
                crate::descriptor::sync_heaps();
                SYNCED_EARLY[0].fetch_add(1, Ordering::Relaxed);
                SYNCED_EARLY[1].fetch_add(started.elapsed().as_micros() as u64, Ordering::Relaxed);
            }
            // Wait-Before-Prep: drain Flight N-1 completely before Flight N
            // touches any shared Shadow, upload staging pool, or Mark
            // buffer (and, without `VICTORY_SYNC_EARLY`, descriptor heap).
            self.finish_flight_locked(&mut submitted, None, None);
        }
        let landed = std::time::Instant::now();

        let next = *submitted;
        if behind {
            AHEAD[0].fetch_add(1, Ordering::Relaxed);
            let counter: *mut u64 = &mut *submitted;
            PREPARING_AHEAD.with(|c| c.set(Some((self as *const Host, counter))));
            if ahead_of.is_some() {
                BEHIND_BY_RANGES.fetch_add(1, Ordering::Relaxed);
            }
            AHEAD_OF.with(|w| *w.borrow_mut() = ahead_of);
        }
        let probing = ahead && !behind;
        if probing {
            PROBE_ASKED.with(|p| *p.borrow_mut() = Some(Vec::new()));
        }
        let prep = {
            let run_refs: Vec<&HostList> = segments.iter().map(|s| &**s).collect();
            self.prepare_submission(next, &run_refs, &barrier_before, false)
        };
        if probing {
            let asked = PROBE_ASKED.with(|p| p.borrow_mut().take()).unwrap_or_default();
            let landed = LAST_LANDED.lock().unwrap_or_else(|e| e.into_inner()).take();
            // Only right behind the flight that landed last, with nothing
            // submitted between.
            if let Some(landed) = landed.filter(|l| l.reasons != 0 && l.fence == next) {
                probe_compare(&landed, &asked);
            }
        }
        // Still behind a flight in the air: it lands before this one runs if
        // this one marks its writes, or if there is no second mark.
        if behind && segments.iter().any(|s| s.marks_writes()) {
            land_ahead(Early::Marks);
        }
        if behind && self.has_in_flight() && self.mark_for(next + 1).is_none() {
            land_ahead(Early::NoMark);
        }
        PREPARING_AHEAD.with(|c| c.set(None));
        AHEAD_OF.with(|w| w.borrow_mut().take());
        let still_behind = behind && self.has_in_flight();
        if still_behind {
            AHEAD[1].fetch_add(1, Ordering::Relaxed);
        }

        let mut native_boundaries = prep.native_boundaries;
        let fence_value = *submitted + 1;
        let mark = self.mark_for(fence_value);
        // Split marks are shared: kept together behind a flight in the air.
        if mark.is_none() || still_behind {
            native_boundaries.clear();
        }
        let mut split_pool = (!native_boundaries.is_empty())
            .then(|| self.split_marks.lock().unwrap_or_else(|e| e.into_inner()));
        if let Some(pool) = split_pool.as_deref_mut() {
            if let Some(marks) = pool.as_mut()
                && let Err(error) = (marks.len()..native_boundaries.len()).try_for_each(|_| {
                    marks.push(Mark::new(self)?);
                    Ok::<_, windows::core::Error>(())
                })
            {
                tracing::warn!(
                    %error,
                    "could not make split submission marks, keeping the native batch together"
                );
                *pool = None;
            }
            if pool.is_none() {
                native_boundaries.clear();
            }
        }
        let split_marks = split_pool
            .as_deref()
            .and_then(Option::as_deref)
            .unwrap_or(&[]);

        *submitted += 1;
        debug_assert_eq!(*submitted, fence_value);
        super::SUBMISSIONS_BEGUN.store(*submitted, Ordering::Release);
        #[cfg(feature = "diagnostics")]
        crate::resource::life::note(0, crate::resource::life::SUBMITTED, fence_value, 0);

        let mut group = prep.lists;
        if let Some(mark) = mark {
            mark.arm(fence_value);
            group.push(mark.list());
        }
        // Armed before the GPU may write: a title write from here on saves
        // its page.
        let armed = arm_undeclared(&segments, &prep.heaps);
        let queue = self.queue();
        let launched = std::time::Instant::now();
        // SAFETY: closed lists; fence and events are ours.
        let waited = unsafe {
            let mut first = 0;
            for (&end, m) in native_boundaries.iter().zip(split_marks) {
                m.arm(fence_value);
                let mut marked = group[first..end].to_vec();
                marked.push(m.list());
                #[cfg(test)]
                super::tests::native_call(marked.len());
                queue.ExecuteCommandLists(&marked);
                first = end;
            }
            #[cfg(test)]
            super::tests::native_call(group.len() - first);
            queue.ExecuteCommandLists(&group[first..]);
            match queue.Signal(&self.fence, fence_value).and_then(|()| {
                self.fence.SetEventOnCompletion(fence_value, self.event)?;
                self.fence
                    .SetEventOnCompletion(fence_value, self.flight_event)
            }) {
                Ok(()) => true,
                Err(e) => {
                    tracing::warn!(error = %e, "host flight submission not signalled");
                    false
                }
            }
        };
        drop(split_pool);
        let sent = std::time::Instant::now();
        add_step(Step::Landing, landed - asked);
        add_step(Step::Preparing, launched - landed);
        add_step(Step::Launching, sent - launched);
        FLIGHTS.fetch_add(1, Ordering::Relaxed);

        let quiet = prep.uploads.is_none() && segments.iter().all(|s| s.quiet_at_landing());
        let writes = (!quiet && prep.uploads.is_none() && prep_ranges_enabled())
            .then(|| {
                let mut w = Writes::default();
                segments.iter().all(|s| s.ahead_writes(&mut w)).then_some(w)
            })
            .flatten();
        let split_count = native_boundaries.len();
        let flight = InFlight {
            fence_value,
            segments,
            armed,
            uploads: prep.uploads,
            batch: prep.batch,
            heaps: prep.heaps,
            split_count,
            started: prep.started,
            prepared: prep.prepared,
            launched,
            waited,
            quiet,
            signals: Vec::new(),
            writes,
        };
        {
            let mut state = self.flight.lock().unwrap_or_else(|e| e.into_inner());
            debug_assert!(state.flights.len() < 2 && (state.flights.is_empty() || still_behind));
            if let Some(free) = state.free_since.take() {
                let (unasked, preparing) = idle_split(free, asked, launched);
                add_step(Step::IdleUnasked, unasked);
                add_step(Step::IdlePreparing, preparing);
            }
            state.flights.push_back(flight);
            if !state.worker_started {
                state.worker_started = true;
                spawn_flight_worker();
            }
        }
        self.flight_wake.notify_one();
    }

    /// Runs `segments` synchronously and recycles them afterward. Used by the
    /// submit worker when `VICTORY_ASYNC_SUBMIT=1` with `VICTORY_FLIGHT1=0`
    /// and as the fallback for traced frames.
    pub(crate) fn execute_sync_segments(
        &self,
        segments: Vec<Arc<HostList>>,
        barrier_before: Vec<bool>,
    ) {
        let lists: Vec<&HostList> = segments.iter().map(|s| &**s).collect();
        self.execute_sync(&lists, &barrier_before);
        drop(lists);
        segments.into_iter().for_each(crate::list::recycle);
    }

    /// Attaches a title fence signal behind the newest flight in the air (or
    /// the one landing). Returns `true` if attached (the caller must NOT call
    /// `fence::set` immediately), or `false` if no flight is active.
    ///
    /// If `already_kept` is `false`, this calls `fence::keep(fence)` before
    /// storing it.
    pub(crate) fn attach_signal(&self, fence: This, value: u64, already_kept: bool) -> bool {
        let mut state = self.flight.lock().unwrap_or_else(|e| e.into_inner());
        if state.flights.is_empty() && !state.landing {
            return false;
        }
        if !already_kept {
            crate::fence::keep(fence);
        }
        match state.flights.back_mut() {
            Some(newest) => newest.signals.push((fence as usize, value)),
            None => state.landing_signals.push((fence as usize, value)),
        }
        true
    }

    /// Notes that a synchronous submission left the host GPU free at `at`,
    /// for the next flight's idle time.
    pub(super) fn note_gpu_free(&self, at: std::time::Instant) {
        self.flight.lock().unwrap_or_else(|e| e.into_inner()).free_since = Some(at);
    }

    /// Whether a flight is currently in the air or landing.
    pub(crate) fn has_in_flight(&self) -> bool {
        let state = self.flight.lock().unwrap_or_else(|e| e.into_inner());
        !state.flights.is_empty() || state.landing
    }

    /// Waits for and lands every flight in the air now. Returns `true` if a
    /// flight was landed by this call.
    pub(crate) fn finish_flight_now(&self) -> bool {
        if !self.has_in_flight() {
            return false;
        }
        let mut submitted = self.submitted.lock().unwrap_or_else(|e| e.into_inner());
        let _holding = Holding::enter();
        self.finish_flight_locked(&mut submitted, None, None)
    }

    /// Lands flights under `Host::submitted`, oldest first. When
    /// `only_if_fence` is `Some(v)`, lands only the oldest, and only if its
    /// fence value is `v` (used by `victory-flight` so it never blocks on a
    /// newer flight while holding `Host::submitted`); else every one.
    /// `waited_at`, when provided by `victory-flight`, records the exact
    /// instant the GPU fence completed so lock-wait latency is never billed
    /// as GPU execution time.
    pub(super) fn finish_flight_locked(
        &self,
        submitted: &mut u64,
        only_if_fence: Option<u64>,
        waited_at: Option<std::time::Instant>,
    ) -> bool {
        self.finish_flight_locked_for(submitted, only_if_fence, waited_at, None)
    }

    fn finish_flight_locked_for(
        &self,
        submitted: &mut u64,
        only_if_fence: Option<u64>,
        waited_at: Option<std::time::Instant>,
        reason: Option<Early>,
    ) -> bool {
        let mut landed_any = false;
        loop {
            let flight = {
                let mut state = self.flight.lock().unwrap_or_else(|e| e.into_inner());
                match state.flights.front() {
                    Some(f) if only_if_fence.is_none_or(|v| f.fence_value == v) => {
                        let flight = state.flights.pop_front();
                        if let Some(f) = &flight {
                            state.landing = true;
                            state.landing_signals = f.signals.clone();
                        }
                        flight
                    }
                    _ => None,
                }
            };
            let Some(flight) = flight else {
                return landed_any;
            };
            self.land(submitted, flight, waited_at.filter(|_| !landed_any), reason);
            landed_any = true;
            if only_if_fence.is_some() {
                return true;
            }
        }
    }

    /// Lands `flight`, taken from the front of `Host::flight`'s flights.
    fn land(
        &self,
        submitted: &mut u64,
        flight: InFlight,
        waited_at: Option<std::time::Instant>,
        _reason: Option<Early>,
    ) {
        let gpu_done = if flight.waited {
            // SAFETY: plain fence completion query.
            if unsafe { self.fence.GetCompletedValue() } < flight.fence_value {
                #[cfg(feature = "diagnostics")]
                let started = _reason.map(|_| std::time::Instant::now());
                self.wait_for(flight.fence_value);
                let done = std::time::Instant::now();
                #[cfg(feature = "diagnostics")]
                if let Some((reason, started)) = _reason.zip(started) {
                    AHEAD_WAIT_US[reason as usize]
                        .fetch_add(done.duration_since(started).as_micros() as u64, Ordering::Relaxed);
                }
                done
            } else {
                waited_at.unwrap_or_else(std::time::Instant::now)
            }
        } else {
            std::time::Instant::now()
        };
        let gpu_duration = gpu_done.saturating_duration_since(flight.launched);
        #[cfg(test)]
        super::tests::native_call_group(flight.waited);
        {
            let mut state = self.flight.lock().unwrap_or_else(|e| e.into_inner());
            // With a newer flight in the air the GPU is not free.
            if state.flights.is_empty() {
                state.free_since = Some(gpu_done);
            }
        }

        let mark = self.mark_for(flight.fence_value);
        let split_failed = if flight.split_count > 0 {
            let pool = self.split_marks.lock().unwrap_or_else(|e| e.into_inner());
            pool.as_deref().is_some_and(|marks| {
                marks
                    .iter()
                    .take(flight.split_count)
                    .any(|m| !m.saw(flight.fence_value))
            })
        } else {
            false
        };
        if flight.waited
            && let Some(mark) = mark
            && (!mark.saw(flight.fence_value) || split_failed)
            // SAFETY: a plain query.
            && unsafe { self.device.GetDeviceRemovedReason() }.is_ok()
        {
            // A newer flight in the air ran on the same queue: done with it
            // first, as the probe that follows submits behind it.
            let newest = {
                let state = self.flight.lock().unwrap_or_else(|e| e.into_inner());
                state.flights.back().map(|f| f.fence_value)
            };
            if let Some(v) = newest {
                self.wait_for(v);
            }
            self.lost(submitted, mark);
        }
        #[cfg(feature = "diagnostics")]
        crate::resource::life::note(0, crate::resource::life::RAN, flight.fence_value, 0);
        // This flight's fence, not the newest submission's: another may still
        // be in the air (`VICTORY_PREP_AHEAD=1`).
        let done = flight.fence_value;
        super::SUBMISSIONS_DONE.store(done, Ordering::Release);
        // The GPU fence has completed: return worker-owned staging chunks now
        // at the same point as the synchronous path.
        if let Some(u) = &flight.uploads {
            u.release_worker();
        }

        {
            let run_refs: Vec<&HostList> = flight
                .uploads
                .as_ref()
                .map(|u| u.list())
                .into_iter()
                .chain(flight.segments.iter().map(|s| &**s))
                .collect();
            self.finish_post_gpu(
                &run_refs,
                flight.batch,
                flight.heaps,
                flight.started,
                flight.prepared,
                gpu_duration,
                None,
                done,
            );
        }

        // Its copy-back is done: what it armed is as anything else again.
        for copy in &flight.armed {
            copy.disarm_undeclared();
        }

        if prep_ahead_enabled() {
            let mut reasons = flight.segments.iter().fold(0, |b, s| b | s.not_quiet_reasons());
            if flight.uploads.is_some() {
                reasons |= 1 << NotQuiet::Uploads as u8;
            }
            for r in 0..NOT_QUIET {
                if reasons & (1 << r) != 0 {
                    PROBE[Probe::Optimistic as usize + 1 + r].fetch_add(1, Ordering::Relaxed);
                }
            }
            let (wrote, marked) = std::mem::take(&mut *PROBE_SINK.lock().unwrap_or_else(|e| e.into_inner()));
            *LAST_LANDED.lock().unwrap_or_else(|e| e.into_inner()) =
                Some(Landed { fence: done, reasons, wrote, marked });
        }

        // Return the upload list to `bindless::LIST` and recycle host lists
        // only after the GPU fence and post-GPU work have finished.
        drop(flight.uploads);
        flight.segments.into_iter().for_each(crate::list::recycle);

        // Deliver every title fence signal queued behind this flight before
        // it stops landing, so later `queue::signal` calls can never overtake
        // an attached signal.
        let mut state = self.flight.lock().unwrap_or_else(|e| e.into_inner());
        for (fence, value) in state.landing_signals.drain(..) {
            crate::fence::set(fence as This, value);
            crate::fence::let_go(fence as This);
        }
        state.landing = false;
    }
}

#[cfg(test)]
pub(super) static TEST_PAUSE_LANDING: std::sync::atomic::AtomicBool =
    std::sync::atomic::AtomicBool::new(false);

fn spawn_flight_worker() {
    let _ = std::thread::Builder::new()
        .name("victory-flight".into())
        .spawn(|| {
            let Some(host) = super::host() else {
                return;
            };
            loop {
                let (fence_value, waited) = {
                    let mut state = host.flight.lock().unwrap_or_else(|e| e.into_inner());
                    while state.flights.is_empty() {
                        state = host
                            .flight_wake
                            .wait(state)
                            .unwrap_or_else(|e| e.into_inner());
                    }
                    let f = state.flights.front().expect("a flight in the air");
                    (f.fence_value, f.waited)
                };
                // Wait on the GPU fence WITHOUT holding `HELD` or
                // `Host::submitted`.
                if waited {
                    host.wait_for_on_event(host.flight_event, fence_value, 250);
                }
                let gpu_done = std::time::Instant::now();
                #[cfg(test)]
                while TEST_PAUSE_LANDING.load(Ordering::Acquire) {
                    std::thread::sleep(std::time::Duration::from_millis(1));
                }
                // Skip acquiring `Host::submitted` if another thread already
                // claimed and began landing this flight while we waited.
                let still_mine = host
                    .flight
                    .lock()
                    .unwrap_or_else(|e| e.into_inner())
                    .flights
                    .front()
                    .is_some_and(|f| f.fence_value == fence_value);
                if !still_mine {
                    continue;
                }
                let landed = {
                    let mut submitted = host.submitted.lock().unwrap_or_else(|e| e.into_inner());
                    let _holding = Holding::enter();
                    host.finish_flight_locked(&mut submitted, Some(fence_value), Some(gpu_done))
                };
                // Strictly AFTER dropping `Host::submitted`: wake held work or
                // the submit worker in case this flight's BoP writes or fence
                // signals unblocked another queue.
                if landed {
                    crate::queue::on_flight_landed();
                }
            }
        });
}

#[cfg(test)]
mod tests {
    use std::sync::Arc;
    use std::sync::atomic::{AtomicBool, Ordering};
    use std::time::{Duration, Instant};

    use super::*;
    use crate::backend::{HostGpu, HostMemory};
    use crate::host::host;
    use crate::object::{S_OK, This, slot};
    use crate::queue::Written;

    fn make_queue_and_fence() -> (This, This) {
        let q = crate::queue::create_direct_for_test();
        let mut f = std::ptr::null();
        assert_eq!(
            crate::fence::create(
                std::ptr::null(),
                0,
                0,
                &crate::fence::FENCE_IIDS[0],
                &mut f
            ),
            S_OK
        );
        (q, f)
    }

    fn queue_signal(q: This, f: This, value: u64) {
        let hr = crate::queue::signal_for_test(q, f, value);
        assert_eq!(hr, S_OK);
    }

    fn queue_execute(q: This, lists: &[This]) {
        crate::queue::execute_for_test(q, lists);
    }

    fn make_title_list() -> (This, This) {
        let null = std::ptr::null();
        let alloc_iid = crate::object::guid(
            0x6102dee4,
            0xaf59,
            0x4b09,
            [0xb9, 0x99, 0xb4, 0x4d, 0x73, 0xf0, 0x9b, 0x24],
        );
        let list_iid = crate::object::guid(
            0x5b160d0f,
            0xac1b,
            0x4185,
            [0x8b, 0xa8, 0xb3, 0xae, 0x42, 0xa5, 0xa4, 0x55],
        );
        let mut alloc = null;
        assert_eq!(
            crate::list::create_allocator(null, 0, &alloc_iid, &mut alloc),
            S_OK
        );
        let mut list = null;
        assert_eq!(
            crate::list::create_list(null, 0, 0, alloc, null, &list_iid, &mut list),
            S_OK
        );
        (alloc, list)
    }

    fn close_title_list(list: This) {
        let close = slot::<extern "system" fn(This) -> i32>(list, 0x48);
        assert_eq!(close(list), S_OK);
    }

    fn reset_title_list(list: This, alloc: This) {
        let reset = slot::<extern "system" fn(This, This, This) -> i32>(list, 0x50);
        assert_eq!(reset(list, alloc, std::ptr::null()), S_OK);
    }

    fn wait_fence(f: This, target: u64, timeout: Duration) -> bool {
        let start = Instant::now();
        while crate::fence::value(f) < target {
            if start.elapsed() >= timeout {
                return false;
            }
            std::thread::sleep(Duration::from_millis(1));
        }
        true
    }

    /// Under `VICTORY_FLIGHT1=1`, a GPU copy + Shadow copy-back + `after`
    /// callback + bottom-of-pipe write + attached `queue::signal` all land in
    /// order via the background `victory-flight` finisher without requiring a
    /// second submission.
    #[test]
    fn flight1_delivers_signals_bop_writes_and_copy_back_in_order() {
        let Some(host) = host() else {
            return;
        };
        let _mode = ModeGuard::new(true, false);

        let (q, f) = make_queue_and_fence();
        let mut title_buf = vec![0u32; 16];
        let title_addr = title_buf.as_mut_ptr() as u64;
        let shadow = crate::shadow::find(title_addr, 64).expect("shadow");

        // Prepare source upload buffer containing 0xCAFE_BABE.
        let (src_buf, src_ptr) = host.upload_buffer(64).expect("upload buffer");
        // SAFETY: mapped 64-byte upload buffer.
        unsafe {
            src_ptr.cast::<u32>().write_volatile(0xCAFE_BABE);
        }

        let seg = Arc::new(host.new_list().expect("host list"));
        // SAFETY: open host list, valid D3D12 buffers.
        unsafe {
            seg.list
                .CopyBufferRegion(&shadow.buffer, shadow.offset(title_addr), &src_buf, 0, 4);
        }
        seg.keep(&src_buf);
        shadow.gpu_writes(&seg, title_addr, 4, false);

        let after_ran = Arc::new(AtomicBool::new(false));
        let after_flag = after_ran.clone();
        let title_ptr_num = title_addr as usize;
        let saw_copy_back_in_after = Arc::new(AtomicBool::new(false));
        let saw_cb = saw_copy_back_in_after.clone();
        let mut bop_word: u64 = 0;
        let bop_ptr_num = (&raw mut bop_word) as usize;
        let saw_bop_not_yet_written = Arc::new(AtomicBool::new(false));
        let saw_bop = saw_bop_not_yet_written.clone();

        seg.defer_after(move || {
            after_flag.store(true, Ordering::Release);
            // SAFETY: title_buf and bop_word outlive the flight.
            unsafe {
                let copied = (title_ptr_num as *const u32).read_volatile();
                if copied == 0xCAFE_BABE {
                    saw_cb.store(true, Ordering::Release);
                }
                if (bop_ptr_num as *const u64).read_volatile() == 0 {
                    saw_bop.store(true, Ordering::Release);
                }
            }
        });
        seg.defer_write(bop_ptr_num as *mut u8, Written::Value(0x1234_5678), 8);
        seg.close();

        host.execute_flight1(vec![seg], vec![]);
        queue_signal(q, f, 7);

        // Background `victory-flight` must finish the flight and deliver the
        // attached fence signal even with no further submissions.
        assert!(
            wait_fence(f, 7, Duration::from_secs(5)),
            "victory-flight did not deliver attached fence signal"
        );
        assert!(after_ran.load(Ordering::Acquire), "defer_after did not run");
        assert!(
            saw_copy_back_in_after.load(Ordering::Acquire),
            "copy_back must complete before defer_after"
        );
        assert!(
            saw_bop_not_yet_written.load(Ordering::Acquire),
            "BoP writes must run after defer_after"
        );
        // SAFETY: flight has landed and fence reached 7.
        assert_eq!(unsafe { (&raw const bop_word).read_volatile() }, 0x1234_5678);
        assert_eq!(title_buf[0], 0xCAFE_BABE);
        assert!(!host.has_in_flight());
    }

    /// `Host::quiet()` drains any active Flight-1 work cleanly before
    /// returning `Quiet` (preserving D3DMetal Issue #31's GPU-idle guarantee
    /// around shader-visible heap null-fill) with zero deadlock.
    #[test]
    fn flight1_quiet_waits_for_in_flight_before_heap_creation() {
        let Some(host) = host() else {
            return;
        };
        let _mode = ModeGuard::new(true, false);

        let mut bop_word: u32 = 0;
        let seg = Arc::new(host.new_list().expect("host list"));
        seg.defer_write((&raw mut bop_word).cast(), Written::Value(99), 4);
        seg.close();

        host.execute_flight1(vec![seg], vec![]);
        // `quiet()` locks `submitted` and finishes any in-flight work.
        // Force a direct `finish_flight_now` check via `quiet()` (note: off
        // Wine `quiet()` returns None, so also test `finish_flight_now`
        // directly when off Wine).
        let q = host.quiet();
        if q.is_some() {
            assert!(
                !host.has_in_flight(),
                "Host::quiet() returned while a GPU flight was still active"
            );
            // SAFETY: flight has landed before `quiet()` returned.
            assert_eq!(unsafe { (&raw const bop_word).read_volatile() }, 99);
            drop(q);
        } else {
            host.finish_flight_now();
            assert!(!host.has_in_flight());
            // SAFETY: flight has landed after `finish_flight_now()`.
            assert_eq!(unsafe { (&raw const bop_word).read_volatile() }, 99);
        }
    }

    /// Reproduces the 2-submission shadow overlap:
    /// Flight 1 reads `OLD` (`0x11223300`) from a `Shadow` into `out1`, then
    /// the CPU immediately mutates title memory to `NEW` (`0xAABBCC00`) and
    /// submits Flight 2 reading the same `Shadow` into `out2`.
    /// Because Flight-1 enforces Wait-Before-Prep, Flight 1 MUST observe
    /// `OLD` and Flight 2 MUST observe `NEW` (never `NEW, NEW`) across 10
    /// consecutive iterations.
    #[test]
    fn flight1_never_clobbers_in_flight_shadow_data() {
        let Some(host) = host() else {
            return;
        };
        let _mode = ModeGuard::new(true, false);

        // Allocate a page-aligned title buffer so Shadow tracks it cleanly.
        let layout = std::alloc::Layout::from_size_align(4096, 4096).unwrap();
        // SAFETY: valid layout; freed after all flights finish.
        let title_ptr = unsafe { std::alloc::alloc_zeroed(layout) };
        let title_addr = title_ptr as u64;
        let shadow = crate::shadow::find(title_addr, 4096).expect("shadow");

        let (out1_buf, out1_ptr) = host.mapped_buffer(64, true).expect("out1");
        let (out2_buf, out2_ptr) = host.mapped_buffer(64, true).expect("out2");

        for iter in 0..10u32 {
            let old_val = 0x1122_3300u32 | iter;
            let new_val = 0xAABB_CC00u32 | iter;
            // SAFETY: valid 4096-byte title buffer and 64-byte readback buffers.
            unsafe {
                title_ptr.cast::<u32>().write_volatile(old_val);
                out1_ptr.cast::<u32>().write_volatile(0);
                out2_ptr.cast::<u32>().write_volatile(0);
            }

            let seg1 = Arc::new(host.new_list().expect("seg1"));
            shadow.refresh(&seg1, title_addr, 64);
            // SAFETY: open list, valid buffers.
            unsafe {
                seg1.list.CopyBufferRegion(
                    &out1_buf,
                    0,
                    &shadow.buffer,
                    shadow.offset(title_addr),
                    4,
                );
            }
            seg1.keep(&out1_buf);
            seg1.close();

            // Launch Flight 1 asynchronously!
            host.execute_flight1(vec![seg1], vec![]);

            // Immediately overwrite title memory with NEW while Flight 1 is
            // still in flight (exactly the #273 overlap window).
            // SAFETY: valid title buffer.
            unsafe {
                title_ptr.cast::<u32>().write_volatile(new_val);
            }

            let seg2 = Arc::new(host.new_list().expect("seg2"));
            shadow.refresh(&seg2, title_addr, 64);
            // SAFETY: open list, valid buffers.
            unsafe {
                seg2.list.CopyBufferRegion(
                    &out2_buf,
                    0,
                    &shadow.buffer,
                    shadow.offset(title_addr),
                    4,
                );
            }
            seg2.keep(&out2_buf);
            seg2.close();

            // Flight 2 MUST wait for Flight 1 before refreshing `shadow`.
            host.execute_flight1(vec![seg2], vec![]);
            host.finish_flight_now();

            // SAFETY: both flights have landed.
            let got1 = unsafe { out1_ptr.cast::<u32>().read_volatile() };
            // SAFETY: both flights have landed.
            let got2 = unsafe { out2_ptr.cast::<u32>().read_volatile() };
            assert_eq!(
                (got1, got2),
                (old_val, new_val),
                "iter {iter}: Shadow.data clobbered in flight! got ({got1:#x}, {got2:#x}), expected ({old_val:#x}, {new_val:#x})"
            );
        }

        // SAFETY: all flights landed; free title buffer.
        unsafe { std::alloc::dealloc(title_ptr, layout) };
    }

    /// Under `VICTORY_FLIGHT1=1` + `VICTORY_HEAP_WORKER=1`, `bindless::Uploads`
    /// keeps both the upload list and the worker `ReadyDrain` alive across the
    /// flight until the GPU fence completes, then calls `release_worker()` and
    /// returns the upload list to the reuse pool so subsequent submissions see
    /// the uploaded texture and cleanly reset the dirty upload list.
    #[test]
    fn flight1_uploads_releases_heap_worker_drain_after_fence() {
        use crate::compute::{Param, srv, uav};
        use windows::core::s;

        let Some(host) = host() else {
            return;
        };
        let _mode = ModeGuard::new(true, false);
        let _worker = crate::heap_worker::EnableForTest::new();

        const PAGE_BYTES: usize = 64 * 1024;
        const LINEAR: u32 = 1;
        const SHADER: &str = r#"
RWByteAddressBuffer dst : register(u0);
Texture2D<float4> tex : register(t0);
[numthreads(4, 1, 1)]
void main(uint3 id : SV_DispatchThreadID) {
    uint texel = (uint)round(tex.Load(int3(id.x, 0, 0)).x * 255.0);
    dst.Store(id.x * 4, texel);
}
"#;

        let null = std::ptr::null();
        let heap_desc: [u32; 8] = [0, 16, 1, 0, 0, 0, 0, 0];
        let mut heap = null;
        let heap_iid = crate::object::guid(
            0x8efb471d,
            0x616c,
            0x4f49,
            [0x90, 0xf7, 0x12, 0x7b, 0xb7, 0x63, 0xfa, 0x51],
        );
        assert_eq!(
            crate::descriptor::create_heap(null, heap_desc.as_ptr().cast(), &heap_iid, &mut heap),
            S_OK
        );
        let title_heap = crate::descriptor::heap(heap);
        let slot_at = |i: u64| title_heap.start + 32 * i;
        let heap_host = title_heap.host().expect("host view heap");

        let pool = crate::watch::tests::pool(1);
        let addr_a = pool as u64;
        // SAFETY: dedicated pool page.
        unsafe {
            std::ptr::write_bytes(addr_a as *mut u8, 0x44, PAGE_BYTES);
        }
        let placed_res = [3u32, 0, 0, 0, 4, 0, 1, 1 | 1 << 16, 28, 1, 0, LINEAR, 0, 0];
        let mut desc_a = [0u32; 16];
        (desc_a[0], desc_a[1], desc_a[2], desc_a[5]) = (28, 4, 0x1688, 1);
        (desc_a[10], desc_a[11]) = (addr_a as u32, (addr_a >> 32) as u32);

        let dst_desc: [u32; 14] = [1, 0, 0, 0, 256, 0, 1, 1 | 1 << 16, 0, 1, 0, 1, 0, 0];
        let props = [1u32, 0, 0, 0, 0];
        let res_iid = crate::object::guid(
            0x696442be,
            0xa72e,
            0x4059,
            [0xbc, 0x79, 0x5b, 0x5c, 0x98, 0x04, 0x0f, 0xad],
        );
        let mut dst = null;
        assert_eq!(
            crate::resource::create_committed(
                null,
                props.as_ptr().cast(),
                0,
                dst_desc.as_ptr().cast(),
                0,
                std::ptr::null(),
                &res_iid,
                &mut dst,
            ),
            S_OK
        );
        let (dst_bytes, _) = crate::resource::buffer_bytes(dst).unwrap();
        let uav_desc: [u32; 10] = [39, 1, 0, 0, 64, 0, 0, 0, 1, 0];
        crate::descriptor::create_uav(null, dst, null, &uav_desc, slot_at(0));
        crate::descriptor::create_placed_srv(null, &placed_res, &desc_a, slot_at(1));

        _worker.step_heap(title_heap);
        assert!(crate::heap_worker::has_ready());

        let (root, pso) =
            crate::compute::pipeline(SHADER, s!("flight_worker"), 0, &[Param::Table(&[uav(0, 1), srv(0, 1)])])
                .unwrap();
        let seg = Arc::new(host.new_list().unwrap());
        // SAFETY: open list, live objects.
        unsafe {
            seg.list.SetDescriptorHeaps(&[Some(heap_host.heap.clone())]);
            seg.list.SetComputeRootSignature(&root);
            seg.list.SetComputeRootDescriptorTable(0, heap_host.gpu(0));
            seg.list.SetPipelineState(&pso);
            seg.list.Dispatch(1, 1, 1);
        }
        seg.defer_heap(&heap_host.viewed);
        seg.close();

        let (q, f) = make_queue_and_fence();
        host.execute_flight1(vec![seg], vec![]);
        queue_signal(q, f, 1);

        assert!(
            wait_fence(f, 1, Duration::from_secs(5)),
            "victory-flight did not land worker upload flight"
        );
        assert!(!host.has_in_flight());
        assert!(heap_host.kept(1), "slot 1 published at flight submission");
        // Not asserted: `has_ready` is false only until the flight's kick of the
        // heap worker finds more to stage, which depends on timing. The
        // read-back below shows the staged upload ran.
        let out = crate::shadow::find(dst_bytes as u64, 256).unwrap();
        assert_eq!(out.read_host(dst_bytes as u64, 4), [0x44, 0x44, 0x44, 0x44]);

        // A follow-up empty flight exercises resetting the returned dirty upload list.
        let seg2 = Arc::new(host.new_list().unwrap());
        seg2.close();
        host.execute_flight1(vec![seg2], vec![]);
        host.finish_flight_now();
        assert!(!host.has_in_flight());

        crate::object::release_ref(heap);
        crate::object::release_ref(dst);
        crate::watch::tests::free(pool);
    }

    /// The GPU's idle time before a flight is split at when its work was
    /// there: a worker that came to a free GPU had nothing to run until it
    /// came; one already waiting for the last flight to land prepared all
    /// the idle time.
    #[test]
    fn idle_time_is_split_at_when_the_work_was_there() {
        let free = Instant::now();
        let ms = Duration::from_millis;
        assert_eq!(idle_split(free, free + ms(5), free + ms(8)), (ms(5), ms(3)));
        assert_eq!(idle_split(free + ms(4), free, free + ms(10)), (ms(0), ms(6)));
        assert_eq!(idle_split(free, free, free), (ms(0), ms(0)));
    }

    /// Under `VICTORY_ASYNC_SUBMIT=1` the title's `ExecuteCommandLists` and
    /// `Signal` only queue: the title's thread is back while the work waits
    /// for a memory value, may reset the list at once, and never runs host
    /// work itself. The fence moves only once the work ran, after its
    /// bottom-of-pipe write.
    #[test]
    fn async_submit_runs_host_work_off_the_title_thread_and_signals_after_it() {
        let Some(_) = host() else {
            return;
        };
        let _mode = ModeGuard::new(true, true);
        const GREATER_EQUAL: u32 = 7;

        let (q, f) = make_queue_and_fence();
        let (alloc, list) = make_title_list();
        let flag = Box::new(std::sync::atomic::AtomicU64::new(0));
        let flag_addr = std::ptr::from_ref(&*flag) as u64;
        let ran_on = Arc::new(std::sync::Mutex::new(None));
        let mut word: u64 = 0;

        crate::list::wait64(list, flag_addr, GREATER_EQUAL, 1, 0);
        let seg = crate::list::host_list(list).expect("host list");
        let r = ran_on.clone();
        seg.defer_after(move || {
            *r.lock().unwrap() = Some(std::thread::current().id());
        });
        seg.defer_write((&raw mut word).cast(), Written::Value(0x5eed), 8);
        close_title_list(list);
        queue_execute(q, &[list]);
        queue_signal(q, f, 1);

        assert_eq!(crate::fence::value(f), 0, "the fence moved before the work ran");
        assert!(ran_on.lock().unwrap().is_none(), "the held work ran");
        // D3D12 lets a submitted list be reset at once.
        reset_title_list(list, alloc);
        close_title_list(list);

        flag.store(1, Ordering::Release);
        assert!(
            wait_fence(f, 1, Duration::from_secs(5)),
            "the submit worker did not run the work once its wait held"
        );
        // SAFETY: the fence reached 1, so the write landed before it.
        assert_eq!(unsafe { (&raw const word).read_volatile() }, 0x5eed);
        let thread = ran_on.lock().unwrap().expect("the work ran");
        assert_ne!(thread, std::thread::current().id(), "host work ran on the title's thread");
        let (flights, _) = take_flight_times();
        assert!(flights >= 1, "the flight was not counted");
    }

    /// Under `VICTORY_ASYNC_SUBMIT=1` a queue that waits for another
    /// queue's fence completes only after that queue's work: whenever the
    /// title reads the waiting queue's fence as reached, the other fence and
    /// the other queue's bottom-of-pipe write show too, and neither fence
    /// moves before the first queue's work could run. (The worker may join
    /// both queues' work into one host submission; the title sees its
    /// effects only when it has landed.)
    #[test]
    fn async_submit_queue_wait_completes_after_the_signalling_queue_work() {
        let Some(_) = host() else {
            return;
        };
        let _mode = ModeGuard::new(true, true);
        const GREATER_EQUAL: u32 = 7;

        let (qa, fa) = make_queue_and_fence();
        let (qb, fb) = make_queue_and_fence();
        let (_alloc_a, la) = make_title_list();
        let (_alloc_b, lb) = make_title_list();
        let flag = Box::new(std::sync::atomic::AtomicU64::new(0));
        let flag_addr = std::ptr::from_ref(&*flag) as u64;
        let mut a_word: u64 = 0;
        let mut b_word: u64 = 0;

        crate::list::wait64(la, flag_addr, GREATER_EQUAL, 1, 0);
        crate::list::host_list(la)
            .expect("host list a")
            .defer_write((&raw mut a_word).cast(), Written::Value(1), 8);
        close_title_list(la);
        queue_execute(qa, &[la]);
        queue_signal(qa, fa, 1);

        assert_eq!(crate::queue::wait_for_test(qb, fa, 1), S_OK);
        crate::list::host_list(lb)
            .expect("host list b")
            .defer_write((&raw mut b_word).cast(), Written::Value(2), 8);
        close_title_list(lb);
        queue_execute(qb, &[lb]);
        queue_signal(qb, fb, 1);

        assert_eq!((crate::fence::value(fa), crate::fence::value(fb)), (0, 0));
        // SAFETY: plain reads of words the landings write.
        let words = || unsafe { ((&raw const a_word).read_volatile(), (&raw const b_word).read_volatile()) };
        assert_eq!(words(), (0, 0), "held work wrote before its wait held");
        flag.store(1, Ordering::Release);
        let start = Instant::now();
        loop {
            // B's fence first: anything it shows must already show A's.
            let b = crate::fence::value(fb);
            let a = crate::fence::value(fa);
            if b >= 1 {
                assert_eq!(a, 1, "queue b completed before the fence it waited for");
                assert_eq!(words(), (1, 2), "a fence moved before its queue's writes");
                break;
            }
            assert!(start.elapsed() < Duration::from_secs(5), "the waiting queue's work did not run");
            std::thread::yield_now();
        }
    }

    fn flights_in_air(host: &Host) -> usize {
        host.flight.lock().unwrap().flights.len()
    }

    fn ahead_counts() -> [u64; AHEAD_SLOTS] {
        AHEAD.each_ref().map(|a| a.load(Ordering::Relaxed))
    }

    /// A closed host list that copies `value` into the first word of `out`
    /// (a mapped buffer): it writes nothing that goes back to title memory.
    fn quiet_copy(
        host: &Host,
        value: u32,
        out: &windows::Win32::Graphics::Direct3D12::ID3D12Resource,
    ) -> Arc<HostList> {
        let (src, src_ptr) = host.upload_buffer(64).expect("upload buffer");
        // SAFETY: mapped 64-byte upload buffer.
        unsafe { src_ptr.cast::<u32>().write_volatile(value) };
        let seg = Arc::new(host.new_list().expect("host list"));
        // SAFETY: open host list, valid buffers.
        unsafe { seg.list.CopyBufferRegion(out, 0, &src, 0, 4) };
        seg.keep(&src);
        seg.keep(out);
        seg.close();
        seg
    }

    /// Runs `attempt` until it took the path it tests (up to 20 times):
    /// another test's `Host::quiet` (a heap made on its thread) lands every
    /// flight in the air, so the flight a test holds may land before the
    /// decision it tests. What each attempt checks of the results holds
    /// either way.
    fn until_taken(mut attempt: impl FnMut() -> bool) {
        assert!(
            (0..20).any(|_| attempt()),
            "the flight held in the air was always landed by another thread first"
        );
    }

    /// With `VICTORY_PREP_AHEAD=1` a submission behind a quiet flight (one
    /// whose landing writes nothing back) is prepared while it is in the air
    /// and launched behind it: two in flight. Both run, the newer one's
    /// refresh reads what the title wrote, and their fences move in order.
    #[test]
    fn prep_ahead_launches_behind_a_quiet_flight_and_lands_in_order() {
        let Some(host) = host() else {
            return;
        };
        let _mode = ModeGuard::with_prep_ahead(true, false, true);
        until_taken(|| launches_behind_once(host));
    }

    fn launches_behind_once(host: &Host) -> bool {
        // The frame's eviction pass is done: it would land the flight first.
        crate::resource::evict_unused();
        let (_q, f1) = make_queue_and_fence();
        let (_q2, f2) = make_queue_and_fence();
        let layout = std::alloc::Layout::from_size_align(4096, 4096).unwrap();
        // SAFETY: valid layout; freed at the end.
        let title_ptr = unsafe { std::alloc::alloc_zeroed(layout) };
        let title_addr = title_ptr as u64;
        let shadow = crate::shadow::find(title_addr, 4096).expect("shadow");
        let (out1, out1_ptr) = host.mapped_buffer(64, true).expect("out1");
        let (out2, out2_ptr) = host.mapped_buffer(64, true).expect("out2");
        let mut bop: u64 = 0;
        let before = ahead_counts();

        TEST_PAUSE_LANDING.store(true, Ordering::Release);
        host.execute_flight1(vec![quiet_copy(host, 0x1111_0001, &out1)], vec![]);
        assert!(host.attach_signal(f1, 1, false));
        // Another test's synchronous submission may land it at once: then
        // this attempt does not take the path (`until_taken`).
        let first_in_air = flights_in_air(host) == 1;

        // SAFETY: valid title buffer.
        unsafe { title_ptr.cast::<u32>().write_volatile(0x2222_0002) };
        let seg2 = Arc::new(host.new_list().expect("seg2"));
        shadow.refresh(&seg2, title_addr, 64);
        // SAFETY: open list, valid buffers.
        unsafe { seg2.list.CopyBufferRegion(&out2, 0, &shadow.buffer, shadow.offset(title_addr), 4) };
        seg2.keep(&out2);
        seg2.defer_write((&raw mut bop).cast(), Written::Value(7), 8);
        seg2.close();
        host.execute_flight1(vec![seg2], vec![]);
        assert!(host.attach_signal(f2, 1, false));

        let after = ahead_counts();
        let taken = first_in_air
            && after[0] - before[0] == 1
            && after[1] - before[1] == 1
            && flights_in_air(host) == 2;

        TEST_PAUSE_LANDING.store(false, Ordering::Release);
        let start = Instant::now();
        loop {
            let (b, a) = (crate::fence::value(f2), crate::fence::value(f1));
            if b >= 1 {
                assert_eq!(a, 1, "the newer flight's fence moved before the older one's");
                break;
            }
            assert!(start.elapsed() < Duration::from_secs(5), "the flights did not land");
            std::thread::yield_now();
        }
        // SAFETY: both flights have landed.
        unsafe {
            assert_eq!(out1_ptr.cast::<u32>().read_volatile(), 0x1111_0001);
            assert_eq!(out2_ptr.cast::<u32>().read_volatile(), 0x2222_0002);
            assert_eq!((&raw const bop).read_volatile(), 7);
        }
        assert!(!host.has_in_flight());
        drop(shadow);
        // Not freed: the next attempt gets other memory. Freed here, its host
        // copy (still registered, as the title never released it) would be
        // found again at the same address with this attempt's bytes.
        let _ = layout;
        taken
    }

    /// The hazard prep-ahead must not cross: a flight in the air whose
    /// shaders write a host copy, which goes back to title memory when it
    /// lands. A submission refreshing that copy is prepared only after the
    /// flight landed, so it reads the bytes the GPU wrote (now in title
    /// memory), not stale ones over them.
    #[test]
    fn prep_ahead_lands_a_flight_with_copy_back_before_preparing() {
        let Some(host) = host() else {
            return;
        };
        let _mode = ModeGuard::with_prep_ahead(true, false, true);
        until_taken(|| copy_back_once(host));
    }

    fn copy_back_once(host: &Host) -> bool {
        let layout = std::alloc::Layout::from_size_align(4096, 4096).unwrap();
        // SAFETY: valid layout; freed at the end.
        let title_ptr = unsafe { std::alloc::alloc_zeroed(layout) };
        let title_addr = title_ptr as u64;
        let shadow = crate::shadow::find(title_addr, 4096).expect("shadow");
        let (src, src_ptr) = host.upload_buffer(64).expect("upload buffer");
        // SAFETY: mapped 64-byte upload buffer.
        unsafe { src_ptr.cast::<u32>().write_volatile(0xCAFE_BABE) };
        let (out2, out2_ptr) = host.mapped_buffer(64, true).expect("out2");
        let before = ahead_counts();

        TEST_PAUSE_LANDING.store(true, Ordering::Release);
        let seg1 = Arc::new(host.new_list().expect("seg1"));
        // SAFETY: open host list, valid D3D12 buffers.
        unsafe { seg1.list.CopyBufferRegion(&shadow.buffer, shadow.offset(title_addr), &src, 0, 4) };
        seg1.keep(&src);
        shadow.gpu_writes(&seg1, title_addr, 4, false);
        seg1.close();
        assert!(!seg1.quiet_at_landing());
        host.execute_flight1(vec![seg1], vec![]);

        let seg2 = Arc::new(host.new_list().expect("seg2"));
        shadow.refresh(&seg2, title_addr, 64);
        // SAFETY: open list, valid buffers.
        unsafe { seg2.list.CopyBufferRegion(&out2, 0, &shadow.buffer, shadow.offset(title_addr), 4) };
        seg2.keep(&out2);
        seg2.close();
        host.execute_flight1(vec![seg2], vec![]);
        let after = ahead_counts();
        assert_eq!(after[0], before[0], "prepared behind a flight that writes back");
        let taken = after[2 + Early::NotQuiet as usize] - before[2 + Early::NotQuiet as usize] == 1;

        TEST_PAUSE_LANDING.store(false, Ordering::Release);
        host.finish_flight_now();
        // SAFETY: both flights have landed.
        unsafe {
            assert_eq!(title_ptr.cast::<u32>().read_volatile(), 0xCAFE_BABE, "the GPU's write went back");
            assert_eq!(out2_ptr.cast::<u32>().read_volatile(), 0xCAFE_BABE, "the refresh overwrote it");
        }
        drop(shadow);
        // Not freed: the next attempt gets other memory. Freed here, its host
        // copy (still registered, as the title never released it) would be
        // found again at the same address with this attempt's bytes.
        let _ = layout;
        taken
    }

    /// With `VICTORY_SYNC_EARLY=1` the heaps are synced while the flight
    /// before is still in the air, then it lands and the next one runs, in
    /// order.
    #[test]
    fn sync_early_syncs_heaps_while_the_last_flight_runs() {
        let Some(host) = host() else {
            return;
        };
        let _mode = ModeGuard::new(true, false);
        TEST_SYNC_EARLY.with(|c| c.set(Some(true)));
        until_taken(|| {
            let (_q, f1) = make_queue_and_fence();
            let (_q2, f2) = make_queue_and_fence();
            let (out1, out1_ptr) = host.mapped_buffer(64, true).expect("out1");
            let (out2, out2_ptr) = host.mapped_buffer(64, true).expect("out2");
            TEST_PAUSE_LANDING.store(true, Ordering::Release);
            host.execute_flight1(vec![quiet_copy(host, 0x5151_0001, &out1)], vec![]);
            assert!(host.attach_signal(f1, 1, false));
            let in_air = flights_in_air(host) == 1;
            let before = SYNCED_EARLY[0].load(Ordering::Relaxed);
            // The next one lands it itself (the pause holds only the
            // landing thread), after syncing the heaps.
            host.execute_flight1(vec![quiet_copy(host, 0x5151_0002, &out2)], vec![]);
            assert!(host.attach_signal(f2, 1, false));
            let synced = SYNCED_EARLY[0].load(Ordering::Relaxed) > before;
            TEST_PAUSE_LANDING.store(false, Ordering::Release);
            host.finish_flight_now();
            assert_eq!((crate::fence::value(f1), crate::fence::value(f2)), (1, 1));
            // SAFETY: both flights have landed.
            unsafe {
                assert_eq!(out1_ptr.cast::<u32>().read_volatile(), 0x5151_0001);
                assert_eq!(out2_ptr.cast::<u32>().read_volatile(), 0x5151_0002);
            }
            in_air && synced
        });
        TEST_SYNC_EARLY.with(|c| c.set(None));
    }

    #[test]
    fn writes_keep_declared_pages_bottom_of_pipe_bytes_and_whole_copies() {
        let page = crate::watch::PAGE;
        let mut w = Writes::default();
        w.declared(1, page + 8, page + 12);
        w.span(10 * page + 3, 10 * page + 5);
        w.whole(2, 20 * page, 21 * page);
        w.declared(1, 2 * page, 2 * page + 1);
        let w = w.behind();
        // Declared bytes count as their whole pages, joined where they meet.
        assert_eq!(w.spans[0], (page, 3 * page));
        assert!(w.overlaps(page, page + 1) && w.overlaps(3 * page - 1, 3 * page));
        assert!(!w.overlaps(0, page) && !w.overlaps(3 * page, 4 * page));
        // Bottom-of-pipe writes are their bytes only.
        assert!(w.overlaps(10 * page + 4, 10 * page + 100));
        assert!(!w.overlaps(10 * page, 10 * page + 3) && !w.overlaps(10 * page + 5, 11 * page));
        assert!(w.overlaps(20 * page + 77, 20 * page + 78));
        assert!(!w.overlaps(5, 5), "an empty range overlaps nothing");
        assert!(w.whole.contains(&2) && w.written.contains(&1) && !w.whole.contains(&1));
    }

    /// 64 KiB of zeroed title memory on its own pages (never freed: see
    /// [`launches_behind_once`]), and its host copy.
    fn title_page() -> (*mut u8, u64, Arc<crate::shadow::Shadow>) {
        let page = crate::watch::PAGE;
        let layout = std::alloc::Layout::from_size_align(page, page).unwrap();
        // SAFETY: valid layout.
        let ptr = unsafe { std::alloc::alloc_zeroed(layout) };
        let shadow = crate::shadow::find(ptr as u64, page as u64).expect("shadow");
        (ptr, ptr as u64, shadow)
    }

    /// A list whose shader-like copy writes `value` into the first word of
    /// `shadow`'s copy of `at`, declared as a GPU write (it goes back).
    fn writes_back(host: &Host, shadow: &Arc<crate::shadow::Shadow>, at: u64, value: u32) -> Arc<HostList> {
        let (src, src_ptr) = host.upload_buffer(64).expect("upload buffer");
        // SAFETY: mapped 64-byte upload buffer.
        unsafe { src_ptr.cast::<u32>().write_volatile(value) };
        let seg = Arc::new(host.new_list().expect("host list"));
        // SAFETY: open host list, valid D3D12 buffers.
        unsafe { seg.list.CopyBufferRegion(&shadow.buffer, shadow.offset(at), &src, 0, 4) };
        seg.keep(&src);
        shadow.gpu_writes(&seg, at, 4, false);
        seg.close();
        seg
    }

    /// A list refreshing the first 64 bytes of `shadow`'s copy of `at` and
    /// copying its first word into `out`.
    fn reads(
        host: &Host,
        shadow: &Arc<crate::shadow::Shadow>,
        at: u64,
        out: &windows::Win32::Graphics::Direct3D12::ID3D12Resource,
    ) -> Arc<HostList> {
        let seg = Arc::new(host.new_list().expect("host list"));
        shadow.refresh(&seg, at, 64);
        // SAFETY: open list, valid buffers.
        unsafe { seg.list.CopyBufferRegion(out, 0, &shadow.buffer, shadow.offset(at), 4) };
        seg.keep(out);
        seg.close();
        seg
    }

    /// With `VICTORY_PREP_AHEAD=ranges`, a submission behind a flight whose
    /// shaders write other memory (which goes back when it lands) is
    /// prepared and launched while that one is in the air: two in flight.
    /// Its refresh reads what the title wrote, and the first one's write
    /// goes back.
    #[test]
    fn prep_ranges_launches_behind_a_flight_writing_other_memory() {
        let Some(host) = host() else {
            return;
        };
        let _mode = ModeGuard::with_prep_ranges();
        until_taken(|| ranges_behind_once(host));
    }

    fn ranges_behind_once(host: &Host) -> bool {
        crate::resource::evict_unused();
        let (a_ptr, a, a_copy) = title_page();
        let (b_ptr, b, b_copy) = title_page();
        let (out, out_ptr) = host.mapped_buffer(64, true).expect("out");
        let before = ahead_counts();
        let by_ranges = BEHIND_BY_RANGES.load(Ordering::Relaxed);

        TEST_PAUSE_LANDING.store(true, Ordering::Release);
        let seg1 = writes_back(host, &a_copy, a, 0xCAFE_BABE);
        assert!(!seg1.quiet_at_landing());
        host.execute_flight1(vec![seg1], vec![]);
        // The title writes the other memory while the flight is in the air.
        // SAFETY: valid title memory.
        unsafe { b_ptr.cast::<u32>().write_volatile(0x2222_0002) };
        host.execute_flight1(vec![reads(host, &b_copy, b, &out)], vec![]);
        let after = ahead_counts();
        let taken = after[0] - before[0] == 1
            && after[1] - before[1] == 1
            && BEHIND_BY_RANGES.load(Ordering::Relaxed) > by_ranges
            && flights_in_air(host) == 2;

        TEST_PAUSE_LANDING.store(false, Ordering::Release);
        host.finish_flight_now();
        // SAFETY: both flights have landed.
        unsafe {
            assert_eq!(a_ptr.cast::<u32>().read_volatile(), 0xCAFE_BABE, "the GPU's write went back");
            assert_eq!(out_ptr.cast::<u32>().read_volatile(), 0x2222_0002, "the refresh read the title's write");
        }
        taken
    }

    /// The hazard the range rule must not cross: a submission refreshing
    /// bytes the flight in the air writes. That flight lands first (in the
    /// middle of the preparation), so the refresh does not put the title's
    /// older bytes over the GPU's write: both the title and the newer
    /// flight see it.
    #[test]
    fn prep_ranges_lands_before_refreshing_what_the_flight_writes() {
        let Some(host) = host() else {
            return;
        };
        let _mode = ModeGuard::with_prep_ranges();
        until_taken(|| ranges_overlap_once(host));
    }

    fn ranges_overlap_once(host: &Host) -> bool {
        crate::resource::evict_unused();
        let (a_ptr, a, a_copy) = title_page();
        let (out, out_ptr) = host.mapped_buffer(64, true).expect("out");
        let before = ahead_counts();
        let overlap = 2 + Early::Overlap as usize;

        TEST_PAUSE_LANDING.store(true, Ordering::Release);
        host.execute_flight1(vec![writes_back(host, &a_copy, a, 0xCAFE_BABE)], vec![]);
        // The title writes beside what the GPU writes: the copy needs a
        // refresh, which covers the GPU's word too.
        // SAFETY: valid title memory.
        unsafe { a_ptr.add(32).cast::<u32>().write_volatile(0x3333_0003) };
        host.execute_flight1(vec![reads(host, &a_copy, a, &out)], vec![]);
        let after = ahead_counts();
        let taken = after[0] - before[0] == 1 && after[overlap] - before[overlap] == 1;
        assert_eq!(after[1], before[1], "launched behind a flight writing what it refreshed");

        TEST_PAUSE_LANDING.store(false, Ordering::Release);
        host.finish_flight_now();
        // SAFETY: both flights have landed.
        unsafe {
            assert_eq!(a_ptr.cast::<u32>().read_volatile(), 0xCAFE_BABE, "the GPU's write went back");
            assert_eq!(out_ptr.cast::<u32>().read_volatile(), 0xCAFE_BABE, "the refresh overwrote it");
            assert_eq!(a_ptr.add(32).cast::<u32>().read_volatile(), 0x3333_0003);
        }
        taken
    }

    /// A submission whose shaders mark their writes is prepared behind a
    /// quiet flight, but that flight lands before it launches: its landing
    /// takes the write marks, which would be the new one's.
    #[test]
    fn prep_ahead_lands_before_launching_a_flight_that_marks_writes() {
        let Some(host) = host() else {
            return;
        };
        let _mode = ModeGuard::with_prep_ahead(true, false, true);
        until_taken(|| marks_once(host));
    }

    fn marks_once(host: &Host) -> bool {
        crate::resource::evict_unused();
        let (out1, _) = host.mapped_buffer(64, true).expect("out1");
        let (out2, out2_ptr) = host.mapped_buffer(64, true).expect("out2");
        let before = ahead_counts();

        TEST_PAUSE_LANDING.store(true, Ordering::Release);
        host.execute_flight1(vec![quiet_copy(host, 1, &out1)], vec![]);
        let seg2 = quiet_copy(host, 0x3333_0003, &out2);
        seg2.note_unknown_gpu_write();
        assert!(seg2.marks_writes() && !seg2.quiet_at_landing());
        host.execute_flight1(vec![seg2], vec![]);
        let after = ahead_counts();
        let early = |e: Early| after[2 + e as usize] - before[2 + e as usize];
        assert_eq!(after[1], before[1], "launched while the first was in the air");
        let taken = after[0] - before[0] == 1 && early(Early::Marks) == 1;

        TEST_PAUSE_LANDING.store(false, Ordering::Release);
        host.finish_flight_now();
        // SAFETY: the flights have landed.
        assert_eq!(unsafe { out2_ptr.cast::<u32>().read_volatile() }, 0x3333_0003);
        taken
    }

    /// A title copy in a submission is done on the CPU while it is
    /// prepared, which on the console runs on the GPU after the work before
    /// it: prepared behind a flight, the submission lands it first.
    #[test]
    fn prep_ahead_lands_before_a_title_copy() {
        let Some(host) = host() else {
            return;
        };
        let _mode = ModeGuard::with_prep_ahead(true, false, true);
        until_taken(|| title_copy_once(host));
    }

    fn title_copy_once(host: &Host) -> bool {
        crate::resource::evict_unused();
        let (out1, _) = host.mapped_buffer(64, true).expect("out1");
        let from = [0x5eed_u64; 4];
        let mut to = [0u64; 4];
        let before = ahead_counts();

        TEST_PAUSE_LANDING.store(true, Ordering::Release);
        host.execute_flight1(vec![quiet_copy(host, 1, &out1)], vec![]);
        let seg2 = Arc::new(host.new_list().expect("seg2"));
        seg2.defer_copy(to.as_mut_ptr().cast(), from.as_ptr().cast(), 32);
        seg2.close();
        host.execute_flight1(vec![seg2], vec![]);
        let after = ahead_counts();
        let early = |e: Early| after[2 + e as usize] - before[2 + e as usize];
        assert_eq!(after[1], before[1], "launched behind the flight with a title copy");
        let taken = after[0] - before[0] == 1 && early(Early::TitleCopy) == 1;

        TEST_PAUSE_LANDING.store(false, Ordering::Release);
        host.finish_flight_now();
        assert_eq!(to, from);
        taken
    }

    #[test]
    fn overlaps_finds_shared_bytes_only() {
        let a = [(100, 10), (0, 10)];
        assert!(!overlaps(&mut a.clone(), &mut [(10, 90), (110, 5)]));
        assert!(overlaps(&mut a.clone(), &mut [(109, 1)]));
        assert!(!overlaps(&mut [(5, 0)], &mut [(0, 10)]), "an empty range");
    }

    /// The probe, behind a flight whose shaders write a declared range of
    /// one host copy: a preparation refreshing another copy could have gone
    /// ahead by ranges; one refreshing that range could not.
    #[test]
    fn prep_ahead_probe_counts_overlapping_and_disjoint_refreshes() {
        let Some(host) = host() else {
            return;
        };
        let _mode = ModeGuard::with_prep_ahead(true, false, true);
        let layout = std::alloc::Layout::from_size_align(1 << 16, 1 << 16).unwrap();
        // SAFETY: valid layouts; freed at the end.
        let (a_ptr, b_ptr) = unsafe { (std::alloc::alloc_zeroed(layout), std::alloc::alloc_zeroed(layout)) };
        let (a, b) = (a_ptr as u64, b_ptr as u64);
        let (sa, sb) = (crate::shadow::find(a, 4096).expect("a"), crate::shadow::find(b, 4096).expect("b"));
        let (src, _) = host.upload_buffer(64).expect("upload buffer");
        let slot = |s: Probe| s as usize;
        // One round: a flight writing `a`, then one refreshing `refreshed`
        // after the title wrote it.
        let round = |refreshed: u64, shadow: &Arc<crate::shadow::Shadow>| {
            let seg1 = Arc::new(host.new_list().expect("seg1"));
            // SAFETY: open host list, valid buffers.
            unsafe { seg1.list.CopyBufferRegion(&sa.buffer, sa.offset(a), &src, 0, 4) };
            seg1.keep(&src);
            sa.gpu_writes(&seg1, a, 4, false);
            seg1.close();
            host.execute_flight1(vec![seg1], vec![]);
            // SAFETY: the title's buffer.
            unsafe { (refreshed as *mut u32).add(16).write_volatile(0x77) };
            let seg2 = Arc::new(host.new_list().expect("seg2"));
            shadow.refresh(&seg2, refreshed, 256);
            seg2.close();
            let before = probe_counts();
            host.execute_flight1(vec![seg2], vec![]);
            host.finish_flight_now();
            let after = probe_counts();
            [Probe::Behind, Probe::RangesOnly, Probe::CleanKnown, Probe::Ceiling]
                .map(|p| after[slot(p)] - before[slot(p)])
        };
        // Another test's flight landing between the two (the last landed is
        // the process's) leaves a round uncompared: it runs again.
        let compared = |refreshed: u64, shadow: &Arc<crate::shadow::Shadow>| {
            (0..20)
                .map(|_| round(refreshed, shadow))
                .find(|counts| counts[0] == 1)
                .expect("another test's flight landed between the two every time")
        };
        let [_, ranges, clean, ceiling] = compared(b, &sb);
        assert_eq!(ranges, 1, "compared behind a flight with declared writes only");
        assert_eq!((clean, ceiling), (1, 1), "another copy: could go ahead");
        let [_, _, clean, ceiling] = compared(a, &sa);
        assert_eq!((clean, ceiling), (0, 0), "the range it wrote: could not");
        drop((sa, sb));
        // SAFETY: the flights have landed.
        unsafe {
            std::alloc::dealloc(a_ptr, layout);
            std::alloc::dealloc(b_ptr, layout);
        }
    }

    /// The fence [`recording_mapper`] reads, and what it saw: each mapping's
    /// destination and the fence's value then.
    static MAPPING_FENCE: std::sync::atomic::AtomicUsize = std::sync::atomic::AtomicUsize::new(0);
    static MAPPED: std::sync::Mutex<Vec<(u64, u64)>> = std::sync::Mutex::new(Vec::new());

    fn recording_mapper(to: u64, _from: u64, _len: u64) -> bool {
        let fence = MAPPING_FENCE.load(Ordering::Acquire) as This;
        MAPPED.lock().unwrap().push((to, crate::fence::value(fence)));
        true
    }

    /// Under `VICTORY_ASYNC_SUBMIT=1` a page mapping batch behind work its
    /// queue holds does not wait for it: the title's call returns at once,
    /// and the batch is mapped once the work before it ran (its fence, 1,
    /// moved) and before the work after it (fence 2) did.
    #[test]
    fn async_submit_queues_page_mappings_in_their_queues_order() {
        let Some(_) = host() else {
            return;
        };
        let _mode = ModeGuard::new(true, true);
        const GREATER_EQUAL: u32 = 7;
        const UNMAPPED: u32 = 2;
        // No other test of this crate sets a mapper.
        crate::resource::use_page_mapper(recording_mapper);

        let (q, f) = make_queue_and_fence();
        MAPPING_FENCE.store(f as usize, Ordering::Release);
        MAPPED.lock().unwrap().clear();
        let (_alloc_a, la) = make_title_list();
        let (_alloc_b, lb) = make_title_list();
        let flag = Box::new(std::sync::atomic::AtomicU64::new(0));
        let flag_addr = std::ptr::from_ref(&*flag) as u64;

        crate::list::wait64(la, flag_addr, GREATER_EQUAL, 1, 0);
        close_title_list(la);
        queue_execute(q, &[la]);
        queue_signal(q, f, 1);
        let destination = 0x7_1234_0000;
        assert_eq!(crate::queue::map_pages_for_test(q, destination, UNMAPPED, 0, 1), S_OK);
        assert!(MAPPED.lock().unwrap().is_empty(), "mapped ahead of the work its queue holds");
        close_title_list(lb);
        queue_execute(q, &[lb]);
        queue_signal(q, f, 2);

        flag.store(1, Ordering::Release);
        assert!(wait_fence(f, 2, Duration::from_secs(5)), "the queue's work did not run");
        assert_eq!(*MAPPED.lock().unwrap(), [(destination, 1)]);
    }

    /// Under `VICTORY_ASYNC_SUBMIT=1` + `VICTORY_FLIGHT1=1`, submits
    /// multiple lists with immediate title `Reset` (exercising `list::restart`
    /// early-reset splitting), interleaved `Signal`s, and `drain_submit`
    /// (as called by `PresentX`), verifying strict FIFO completion order.
    #[test]
    fn async_submit_drains_cleanly_at_present_and_preserves_order() {
        let Some(host) = host() else {
            return;
        };
        let _mode = ModeGuard::new(true, true);

        let (q, f) = make_queue_and_fence();
        let (alloc, list) = make_title_list();

        let mut words = [0u64; 6];
        for (i, slot_word) in words.iter_mut().enumerate() {
            if i > 0 {
                reset_title_list(list, alloc);
            }
            let seg = crate::list::host_list(list).expect("host list");
            let val = (i as u64 + 1) * 100;
            seg.defer_write((slot_word as *mut u64).cast(), Written::Value(val), 8);
            close_title_list(list);
            queue_execute(q, &[list]);
            queue_signal(q, f, i as u64 + 1);
        }

        // Simulates PresentX draining `victory-submit` + any active flight.
        crate::queue::drain_submit(crate::queue::Why::Present);

        assert_eq!(crate::fence::value(f), 6);
        assert!(!host.has_in_flight());
        for (i, &w) in words.iter().enumerate() {
            assert_eq!(w, (i as u64 + 1) * 100, "submission {i} BoP mismatch");
        }
    }
}
