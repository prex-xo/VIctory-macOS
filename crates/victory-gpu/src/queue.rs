//! Command queues. Submitted lists run on the host GPU and are waited for,
//! so fence signals complete at once.
#![allow(clippy::not_unsafe_ptr_arg_deref)]

use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};

use crate::object::{self, E_INVALIDARG, HRESULT, Iid, Object, QUEUE, S_OK, Table, This, guid};
use crate::resource::PAGEABLE_IIDS;

const QUEUE_IIDS: [Iid; 4] = [
    guid(
        0x0ec870a6,
        0x5d7e,
        0x4c22,
        [0x8c, 0xfc, 0x5b, 0xaa, 0xe0, 0x76, 0x16, 0xed],
    ),
    PAGEABLE_IIDS[0],
    PAGEABLE_IIDS[1],
    PAGEABLE_IIDS[2],
];

#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub(crate) struct QueueDesc {
    kind: u32,
    priority: i32,
    flags: u32,
    nodes: u32,
}

#[cfg(all(test, windows))]
pub(crate) const QUEUE_IID: Iid = QUEUE_IIDS[0];

#[cfg(all(test, windows))]
impl QueueDesc {
    pub(crate) fn compute() -> Self {
        Self {
            kind: 2,
            priority: 0,
            flags: 0,
            nodes: 0,
        }
    }
}

#[cfg(all(test, windows))]
pub(crate) fn submit_one(queue: This, list: This) {
    execute(queue, 1, &list);
}

/// The title reads the list type right after the header.
#[repr(C)]
struct Queue {
    object: Object,
    kind: u32,
    desc: QueueDesc,
}
const _: () = assert!(std::mem::offset_of!(Queue, kind) == 0x10);

fn queue<'a>(this: This) -> &'a Queue {
    // SAFETY: called from queue methods only.
    unsafe { &*this.cast::<Queue>() }
}

pub(crate) extern "system" fn create(
    _: This,
    desc: *const QueueDesc,
    iid: *const Iid,
    out: *mut This,
) -> HRESULT {
    // SAFETY: the title passes a description (null is tolerated).
    let Some(&desc) = (unsafe { desc.as_ref() }) else {
        return E_INVALIDARG;
    };
    tracing::debug!(?desc, "command queue");
    let queue = Box::new(Queue {
        object: if crate::device::newer() {
            Object::new(&NEWER_FUNCTIONS)
        } else {
            Object::new(&FUNCTIONS)
        },
        kind: desc.kind,
        desc,
    });
    object::hand_out(QUEUE, queue, iid, &QUEUE_IIDS, out)
}

/// The console's queue description: an engine or pipe and a queue on it
/// in place of PC's priority and node mask.
#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub(crate) struct QueueDescX {
    kind: u32,
    flags: u32,
    engine_or_pipe: u32,
    queue: u32,
}

/// Device method `CreateCommandQueueX`. The host picks engines itself, so
/// the queue is created as on PC.
pub(crate) extern "system" fn create_x(
    this: This,
    desc: *const QueueDescX,
    iid: *const Iid,
    out: *mut This,
) -> HRESULT {
    // SAFETY: the title passes a description (null is tolerated).
    let Some(&x) = (unsafe { desc.as_ref() }) else {
        return E_INVALIDARG;
    };
    tracing::debug!(desc = ?x, "console command queue");
    let desc = QueueDesc {
        kind: x.kind,
        priority: 0,
        flags: x.flags,
        nodes: 1,
    };
    create(this, &desc, iid, out)
}

extern "system" fn query(this: This, iid: *const Iid, out: *mut This) -> HRESULT {
    object::answer(
        QUEUE,
        iid,
        object::is(iid, &QUEUE_IIDS).then_some(this),
        out,
    )
}
extern "system" fn release(this: This) -> u32 {
    object::release_boxed::<Queue>(this)
}

/// Queues the lists' host lists behind what the queue holds, and runs what
/// may run (`crate::hold`; decision 0007).
extern "system" fn execute(this: This, count: u32, lists: *const This) {
    if lists.is_null() {
        return;
    }
    #[cfg(windows)]
    if crate::host::host().is_some() {
        // SAFETY: the title passes `count` lists.
        let lists = unsafe { std::slice::from_raw_parts(lists, count as usize) };
        title_submits();
        let mut held = held_for(Why::Submission);
        let entered = std::time::Instant::now();
        for &l in lists {
            for segment in crate::list::segments(l) {
                let waits = segment.waits();
                held.push(this as usize, Item::Segment(segment, waits));
            }
        }
        account().submissions += 1;
        if crate::host::async_submit_enabled() {
            drop(held);
            ensure_submit_worker();
            wake_submit();
            add_run(Why::Submission, entered);
        } else if crate::host::flight1_enabled() {
            drop(held);
            add_run(Why::Submission, entered);
            pump_unlocked(Why::Submission);
        } else {
            pump(&mut held, Why::Submission);
        }
        return;
    }
    static ONCE: AtomicBool = AtomicBool::new(false);
    if !ONCE.swap(true, Ordering::Relaxed) {
        tracing::warn!(count, "command lists submitted; no host GPU runs them");
    }
}

#[cfg(all(windows, test))]
pub(crate) fn create_direct_for_test() -> This {
    let null = std::ptr::null();
    let desc = QueueDesc {
        kind: 0,
        priority: 0,
        flags: 0,
        nodes: 0,
    };
    let mut q = null;
    assert_eq!(create(null, &desc, &QUEUE_IIDS[0], &mut q), S_OK);
    q
}

#[cfg(all(windows, test))]
pub(crate) fn execute_for_test(this: This, lists: &[This]) {
    execute(this, lists.len() as u32, lists.as_ptr());
}

#[cfg(all(windows, test))]
pub(crate) fn signal_for_test(this: This, fence: This, value: u64) -> HRESULT {
    signal(this, fence, value)
}

#[cfg(all(windows, test))]
pub(crate) fn wait_for_test(this: This, fence: This, value: u64) -> HRESULT {
    wait(this, fence, value)
}

/// `CopyPageMappingsBatchX` with one batch of one range.
#[cfg(all(windows, test))]
pub(crate) fn map_pages_for_test(this: This, destination: u64, kind: u32, start: u32, count: u32) -> HRESULT {
    let range = PageRange { kind, start, count };
    let batch = PageBatch { destination, count: 1, ranges: &raw const range };
    copy_page_mappings(this, 1, &batch, 0, 0, 0)
}

/// What ran held work: the title's submission, a queue's Signal or Wait, a
/// CPU fence call, PresentX, host work of ours that runs synchronously
/// behind the queued work (`VICTORY_ASYNC_SUBMIT=1` or `VICTORY_FLIGHT1=1`:
/// read-backs, captures, page mappings), our timer, or a completed Flight-1
/// landing.
#[derive(Clone, Copy, Debug)]
pub(crate) enum Why {
    Submission,
    Signal,
    Wait,
    Fence,
    Present,
    Sync,
    Timer,
    #[cfg(windows)]
    Flight,
}

#[cfg(windows)]
use crate::hold::{Item, Ready};

/// Every queue's held work.
#[cfg(windows)]
type Holding = crate::hold::Held<std::sync::Arc<crate::host::HostList>>;

#[cfg(windows)]
static HELD: std::sync::Mutex<Holding> = std::sync::Mutex::new(crate::hold::Held::new());

/// Serializes unlocked pumping (`victory-submit`, `drain_submit`, and
/// `VICTORY_FLIGHT1=1` direct calls). Strictly above `HELD` in the lock DAG.
#[cfg(windows)]
static SUBMIT_DRAIN: std::sync::Mutex<()> = std::sync::Mutex::new(());

#[cfg(windows)]
static SUBMIT_WORK: (std::sync::Mutex<bool>, std::sync::Condvar) =
    (std::sync::Mutex::new(false), std::sync::Condvar::new());

#[cfg(windows)]
thread_local! {
    /// True while this thread holds `SUBMIT_DRAIN` (so nested `Host::execute`
    /// calls on the submit worker do not re-enter `drain_submit`).
    static ON_SUBMIT_THREAD: std::cell::Cell<bool> = const { std::cell::Cell::new(false) };
}

#[cfg(windows)]
struct OnSubmit(bool);

#[cfg(windows)]
impl OnSubmit {
    fn enter() -> Self {
        Self(ON_SUBMIT_THREAD.with(|c| c.replace(true)))
    }
}

#[cfg(windows)]
impl Drop for OnSubmit {
    fn drop(&mut self) {
        ON_SUBMIT_THREAD.with(|c| c.set(self.0));
    }
}

#[cfg(windows)]
fn ensure_submit_worker() {
    static STARTED: std::sync::Once = std::sync::Once::new();
    STARTED.call_once(|| {
        let _ = std::thread::Builder::new()
            .name("victory-submit".into())
            .spawn(submit_worker);
    });
}

#[cfg(windows)]
fn wake_submit() {
    *SUBMIT_WORK.0.lock().unwrap_or_else(|e| e.into_inner()) = true;
    SUBMIT_WORK.1.notify_one();
}

#[cfg(windows)]
fn submit_worker() {
    loop {
        {
            let mut work = SUBMIT_WORK.0.lock().unwrap_or_else(|e| e.into_inner());
            while !*work {
                work = SUBMIT_WORK.1.wait(work).unwrap_or_else(|e| e.into_inner());
            }
            *work = false;
        }
        let started = std::time::Instant::now();
        {
            let _drain = SUBMIT_DRAIN.lock().unwrap_or_else(|e| e.into_inner());
            let _on = OnSubmit::enter();
            pump_ready_under_drain(Why::Submission);
        }
        WORKER_US.fetch_add(started.elapsed().as_micros() as u64, Ordering::Relaxed);
    }
}

/// Microseconds `victory-submit` spent running queued work since the last
/// [`take_worker_us`]: host work that, without `VICTORY_ASYNC_SUBMIT=1`,
/// holds the title's threads.
#[cfg(windows)]
static WORKER_US: AtomicU64 = AtomicU64::new(0);

/// [`WORKER_US`] since the last call, cleared (the frame report).
#[cfg(windows)]
pub(crate) fn take_worker_us() -> u64 {
    WORKER_US.swap(0, Ordering::Relaxed)
}

#[cfg(windows)]
fn start_held_timer() {
    static TIMER: std::sync::Once = std::sync::Once::new();
    TIMER.call_once(|| {
        let _ = std::thread::Builder::new()
            .name("victory held work".into())
            .spawn(|| {
                let tick = HeldTick::new();
                loop {
                    tick.wait();
                    pump_held_if_free(Why::Timer);
                }
            });
    });
}

/// The held-work timer's 2 ms tick. `std::thread::sleep` makes a waitable
/// timer for every call and closes it after (on Windows, a high-resolution
/// one): under Wine each of those is a wineserver round trip, about 500 a
/// second, beside the title's own server calls (a profile of the prison
/// cell on a 24 GB M5 Pro saw `NtCreateTimer` and `NtClose` from this loop
/// at ~11 ms a frame). One periodic timer made once and waited on does the
/// same with one wait a tick; `sleep` stays where it cannot be made.
#[cfg(windows)]
struct HeldTick(Option<windows_sys::Win32::Foundation::HANDLE>);

#[cfg(windows)]
impl HeldTick {
    const MS: u32 = 2;

    fn new() -> Self {
        use windows_sys::Win32::System::Threading::{
            CREATE_WAITABLE_TIMER_HIGH_RESOLUTION, CreateWaitableTimerExW, SetWaitableTimer, TIMER_ALL_ACCESS,
        };
        let make = |flags| {
            // SAFETY: no name, default security; the handle is checked.
            let t = unsafe { CreateWaitableTimerExW(std::ptr::null(), std::ptr::null(), flags, TIMER_ALL_ACCESS) };
            (!t.is_null()).then_some(t)
        };
        let Some(timer) = make(CREATE_WAITABLE_TIMER_HIGH_RESOLUTION).or_else(|| make(0)) else {
            return Self(None);
        };
        // First due in one tick (relative, in 100 ns units), then every tick.
        let due: i64 = -(i64::from(Self::MS) * 10_000);
        // SAFETY: a timer handle of ours; no completion routine.
        let set = unsafe { SetWaitableTimer(timer, &due, Self::MS as i32, None, std::ptr::null(), 0) };
        if set == 0 {
            // SAFETY: the handle made above, closed once.
            unsafe { windows_sys::Win32::Foundation::CloseHandle(timer) };
            return Self(None);
        }
        Self(Some(timer))
    }

    fn wait(&self) {
        use windows_sys::Win32::System::Threading::{INFINITE, WaitForSingleObject};
        match self.0 {
            // SAFETY: the periodic timer made in `new`, never closed (the
            // thread runs for the process's life).
            Some(timer) => unsafe {
                WaitForSingleObject(timer, INFINITE);
            },
            None => std::thread::sleep(std::time::Duration::from_millis(u64::from(Self::MS))),
        }
    }
}

/// Pumps all currently ready segments/signals from `HELD` while holding
/// `SUBMIT_DRAIN` and releasing `HELD` around every host GPU call.
#[cfg(windows)]
fn pump_ready_under_drain(why: Why) -> bool {
    let Some(host) = crate::host::host() else {
        return false;
    };
    let fence = |f: usize| crate::fence::value(f as This);
    let max_lists = join_lists();
    let mut moved_any = false;
    let mut pieces = 0u32;
    let mut segments_count = 0u32;
    let mut unknown_count = 0u32;
    let mut coalesced_bop_total = 0u32;
    let mut hazard_splits_total = 0u32;
    let mut waited_all = Vec::new();
    let started = std::time::Instant::now();
    let mut ran = std::time::Duration::ZERO;
    loop {
        let next = {
            let mut held = held();
            if held.is_empty() {
                None
            } else {
                for (queue, waits) in held.force_older_than(VALVE) {
                    account().forced += 1;
                    tracing::warn!(
                        queue = kind(queue),
                        ?waits,
                        "held work runs without its waits after {VALVE:?} (safety valve)"
                    );
                }
                let will_write = |seg: &std::sync::Arc<crate::host::HostList>,
                                  wait: &crate::hold::ValueWait| {
                    seg.will_write(wait)
                };
                let can_join = |piece: &[(std::sync::Arc<crate::host::HostList>, bool)],
                                next: &std::sync::Arc<crate::host::HostList>| {
                    if piece.len() >= max_lists {
                        return false;
                    }
                    let existing = piece
                        .iter()
                        .fold(crate::hold::BatchFlags::default(), |acc, (s, _)| {
                            acc.merge(s.batch_flags())
                        });
                    crate::hold::can_join_batch(existing, next.batch_flags(), piece.len(), max_lists)
                };
                let item = held.take_ready_with(
                    &read_value,
                    &fence,
                    &will_write,
                    &can_join,
                    coalesce_pieces(),
                );
                if item.is_none() && !held.is_empty() {
                    start_held_timer();
                }
                item
            }
        };
        let Some(ready) = next else {
            break;
        };
        moved_any = true;
        match ready {
            Ready::Action(act) => {
                // The queue's work before it is on the host GPU: land it
                // (as the next preparation would), so its copy-back reaches
                // memory as it was mapped, then act.
                host.finish_flight_now();
                act();
            }
            Ready::Signal { fence, value } => {
                if !crate::host::flight1_enabled()
                    || !host.attach_signal(fence as This, value, true)
                {
                    crate::fence::set(fence as This, value);
                    crate::fence::let_go(fence as This);
                }
            }
            Ready::Segments {
                queues,
                segments,
                signals,
                waited,
                unknown,
                coalesced_bop_waits,
                hazard_splits,
            } => {
                pieces += 1;
                segments_count += segments.len() as u32;
                unknown_count += unknown;
                coalesced_bop_total += coalesced_bop_waits;
                hazard_splits_total += hazard_splits;
                waited_all.extend(waited);
                let barriers: Vec<bool> = segments.iter().map(|&(_, b)| b).collect();
                let run_segments: Vec<std::sync::Arc<crate::host::HostList>> =
                    segments.into_iter().map(|(s, _)| s).collect();
                let at = std::time::Instant::now();
                #[cfg(feature = "diagnostics")]
                crate::copystat::add(crate::copystat::C::PumpExecutes, 1);
                if crate::host::flight1_enabled() {
                    host.execute_flight1(run_segments, barriers);
                } else {
                    host.execute_sync_segments(run_segments, barriers);
                }
                for (fence, value) in signals {
                    if !crate::host::flight1_enabled()
                        || !host.attach_signal(fence as This, value, true)
                    {
                        crate::fence::set(fence as This, value);
                        crate::fence::let_go(fence as This);
                    }
                }
                ran += at.elapsed();
                held().finish_running(&queues);
            }
        }
    }
    if segments_count > 0 || unknown_count > 0 {
        let mut a = account();
        a.pump += started.elapsed().saturating_sub(ran);
        a.pieces += pieces;
        a.segments += segments_count;
        a.unknown += unknown_count;
        a.coalesced_bop_waits += coalesced_bop_total;
        a.hazard_splits += hazard_splits_total;
        for (queue, waited) in waited_all {
            a.released.push((kind(queue), waited, why));
        }
    }
    moved_any
}

/// Runs what may run now on the calling thread (`VICTORY_FLIGHT1=1` without
/// the submit worker), its wait for [`SUBMIT_DRAIN`] and its run counted in
/// [`BLOCKED_US`] as `why`'s.
#[cfg(windows)]
fn pump_unlocked(why: Why) {
    let _drain = lock_timed(&SUBMIT_DRAIN, &BLOCKED_US[why as usize][0]);
    let _on = OnSubmit::enter();
    let entered = std::time::Instant::now();
    pump_ready_under_drain(why);
    add_run(why, entered);
}

/// Drains all runnable queued work on `victory-submit` and waits for any
/// active Flight-1 submission to finish completely (repeating if landing a
/// flight unblocks additional held work).
#[cfg(windows)]
pub(crate) fn drain_submit(why: Why) {
    let Some(host) = crate::host::host() else {
        return;
    };
    if ON_SUBMIT_THREAD.with(std::cell::Cell::get) {
        host.finish_flight_now();
        return;
    }
    // The calling thread waits here for the submit worker's turn and the
    // flight in the air, then runs what is left: counted as `why`'s.
    let _drain = lock_timed(&SUBMIT_DRAIN, &BLOCKED_US[why as usize][0]);
    let _on = OnSubmit::enter();
    let entered = std::time::Instant::now();
    loop {
        let moved = pump_ready_under_drain(why);
        let landed = host.finish_flight_now();
        if !moved && !landed && !host.has_in_flight() {
            break;
        }
    }
    add_run(why, entered);
}

/// Called by synchronous `Host::execute` callers (`present`, `resource.rs`,
/// `composite`, `capture`) before touching the GPU so they execute strictly
/// after any queued async-submit work or in-flight Flight-1 submission.
#[cfg(windows)]
#[track_caller]
pub(crate) fn before_sync() {
    if (crate::host::async_submit_enabled() || crate::host::flight1_enabled())
        && !ON_SUBMIT_THREAD.with(std::cell::Cell::get)
    {
        let caller = std::panic::Location::caller();
        let started = std::time::Instant::now();
        drain_submit(Why::Sync);
        note_sync(caller, started.elapsed());
    }
}

/// What drained the queued work (`VICTORY_ASYNC_SUBMIT=1` or
/// `VICTORY_FLIGHT1=1`), for the frame report ([`log_drains`]).
#[cfg(windows)]
#[derive(Clone, Copy)]
enum Drain {
    /// PresentX waited for all queued work.
    Present,
    /// A page mapping batch was done at once: its queue held nothing and no
    /// host submission was in flight.
    MappingAtOnce,
    /// A page mapping batch was queued behind its queue's work.
    MappingQueued,
}

#[cfg(windows)]
static DRAINS: [AtomicU64; 3] = [const { AtomicU64::new(0) }; 3];

#[cfg(windows)]
fn note_drain(d: Drain) {
    DRAINS[d as usize].fetch_add(1, Ordering::Relaxed);
}

/// Our synchronous host work that drained queued work since the last
/// report: where it was called from, how often, and microseconds waited.
#[cfg(windows)]
static SYNCS: std::sync::Mutex<Vec<(&'static std::panic::Location<'static>, u64, u64)>> =
    std::sync::Mutex::new(Vec::new());

#[cfg(windows)]
fn note_sync(at: &'static std::panic::Location<'static>, waited: std::time::Duration) {
    let mut syncs = SYNCS.lock().unwrap_or_else(|e| e.into_inner());
    let us = waited.as_micros() as u64;
    match syncs.iter_mut().find(|(l, ..)| *l == at) {
        Some((_, n, t)) => (*n, *t) = (*n + 1, *t + us),
        None => syncs.push((at, 1, us)),
    }
}

/// The frame report's line of what drained the queued work, per frame over
/// `frames`: nothing unless something did.
#[cfg(windows)]
pub(crate) fn log_drains(frames: f64) {
    let [present, at_once, queued] = DRAINS.each_ref().map(|a| a.swap(0, Ordering::Relaxed));
    let mut syncs = std::mem::take(&mut *SYNCS.lock().unwrap_or_else(|e| e.into_inner()));
    if present + at_once + queued == 0 && syncs.is_empty() {
        return;
    }
    syncs.sort_by_key(|&(_, _, us)| std::cmp::Reverse(us));
    let per = |n: u64| n as f64 / frames;
    let sites: Vec<String> = syncs
        .iter()
        .take(6)
        .map(|(l, n, us)| format!("{}:{} {:.1}x {:.1} ms", l.file(), l.line(), per(*n), *us as f64 / 1e3 / frames))
        .collect();
    tracing::info!(
        "queued host work drained per frame: by presents {:.1}; page mappings done at once {:.1}, queued \
         behind their queue {:.1}; by our own host work {:.1} ({})",
        per(present),
        per(at_once),
        per(queued),
        per(syncs.iter().map(|s| s.1).sum()),
        if sites.is_empty() { "none".into() } else { sites.join(", ") }
    );
}

/// Called by `victory-flight` (and `Quiet::drop`) strictly AFTER releasing
/// `Host::submitted`, in case a landed flight's BoP writes or fence signals
/// unblocked work held in `HELD`.
#[cfg(windows)]
pub(crate) fn on_flight_landed() {
    if crate::host::async_submit_enabled() {
        ensure_submit_worker();
        wake_submit();
    } else if crate::host::flight1_enabled()
        && let Ok(_drain) = SUBMIT_DRAIN.try_lock()
    {
        let _on = OnSubmit::enter();
        pump_ready_under_drain(Why::Flight);
    }
}

#[cfg(windows)]
fn held() -> std::sync::MutexGuard<'static, Holding> {
    HELD.lock().unwrap_or_else(|e| e.into_inner())
}

/// Microseconds the threads that ran held work spent on it, by what ran it
/// ([`Why`], in its order): `[0]` waiting for the queues' lock, `[1]` in
/// the pump, the host submissions included. The thread that called is held
/// for both: a title thread for all but [`Why::Timer`], which is ours. The
/// frame report divides them by its frames ([`take_blocked`]).
///
/// With `VICTORY_ASYNC_SUBMIT=1` a title's call only queues its work, and
/// `[1]` is that; the threads that wait for the submit worker to drain
/// (PresentX, our synchronous host work) count the lock they wait for in
/// `[0]` and what they then wait for and run in `[1]`. The worker's own
/// time is [`WORKER_US`].
#[cfg(windows)]
static BLOCKED_US: [[AtomicU64; 2]; WHY_NAMES.len()] =
    [const { [const { AtomicU64::new(0) }; 2] }; WHY_NAMES.len()];

/// Adds the microseconds since `entered` to `why`'s time in the pump
/// ([`BLOCKED_US`]`[why][1]`).
#[cfg(windows)]
fn add_run(why: Why, entered: std::time::Instant) {
    BLOCKED_US[why as usize][1].fetch_add(entered.elapsed().as_micros() as u64, Ordering::Relaxed);
}

/// [`held`] for a caller that runs held work for `why`, its wait for the
/// lock counted in [`BLOCKED_US`].
#[cfg(windows)]
fn held_for(why: Why) -> std::sync::MutexGuard<'static, Holding> {
    lock_timed(&HELD, &BLOCKED_US[why as usize][0])
}

/// Locks `lock`, adding the microseconds it waited to `waited`.
#[cfg(any(windows, test))]
fn lock_timed<'a, T>(
    lock: &'a std::sync::Mutex<T>,
    waited: &AtomicU64,
) -> std::sync::MutexGuard<'a, T> {
    let asked = std::time::Instant::now();
    let guard = lock.lock().unwrap_or_else(|e| e.into_inner());
    waited.fetch_add(asked.elapsed().as_micros() as u64, Ordering::Relaxed);
    guard
}

/// [`Why`]'s variants in their order, as the frame report names them: the
/// title's submissions, its queues' Signals and Waits, its CPU fence calls,
/// its presents, our synchronous host work behind queued work, and our
/// timer thread.
#[cfg(windows)]
pub(crate) const WHY_NAMES: [&str; 7] = [
    "title submissions",
    "queue signals",
    "queue waits",
    "fence calls",
    "presents",
    "our syncs",
    "our timer",
];
#[cfg(windows)]
const _: () = assert!(Why::Timer as usize + 1 == WHY_NAMES.len());

/// The title's time from a present returning to its next submission
/// (`ExecuteCommandLists`), the floor of what its CPU work costs a frame:
/// microseconds, and of them the time its threads were held in our queue
/// work meanwhile ([`BLOCKED_US`], all title threads), since the last frame
/// report; then frames measured. The rest is the title's own work, its
/// calls into our command lists and other functions included.
#[cfg(windows)]
static TITLE_GAP: [AtomicU64; 3] = [const { AtomicU64::new(0) }; 3];

/// When the last present returned (microseconds since [`epoch_us`]; 0 once
/// a submission closed the gap), and the queue work held then.
#[cfg(windows)]
static PRESENT_RETURNED: [AtomicU64; 2] = [const { AtomicU64::new(0) }; 2];

#[cfg(windows)]
fn epoch_us() -> u64 {
    static EPOCH: std::sync::OnceLock<std::time::Instant> = std::sync::OnceLock::new();
    EPOCH.get_or_init(std::time::Instant::now).elapsed().as_micros() as u64 + 1
}

/// The title threads' time held in our queue work, all of it ([`BLOCKED_US`]
/// but our timer's).
#[cfg(windows)]
fn held_total() -> u64 {
    BLOCKED_US[..Why::Timer as usize]
        .iter()
        .flat_map(|w| w.iter())
        .map(|a| a.load(Ordering::Relaxed))
        .sum()
}

/// Notes a present returning to the title when dropped.
#[cfg(windows)]
struct PresentReturning;

#[cfg(windows)]
impl Drop for PresentReturning {
    fn drop(&mut self) {
        PRESENT_RETURNED[1].store(held_total(), Ordering::Relaxed);
        PRESENT_RETURNED[0].store(epoch_us(), Ordering::Release);
    }
}

/// The title submits: the first submission since a present closes its gap.
#[cfg(windows)]
fn title_submits() {
    let returned = PRESENT_RETURNED[0].swap(0, Ordering::AcqRel);
    if returned == 0 {
        return;
    }
    let gap = epoch_us().saturating_sub(returned);
    // `take_blocked` may have cleared the counters meanwhile: then nothing
    // is subtracted.
    let held = held_total().saturating_sub(PRESENT_RETURNED[1].load(Ordering::Relaxed)).min(gap);
    TITLE_GAP[0].fetch_add(gap, Ordering::Relaxed);
    TITLE_GAP[1].fetch_add(held, Ordering::Relaxed);
    TITLE_GAP[2].fetch_add(1, Ordering::Relaxed);
}

/// The frame report's line for [`TITLE_GAP`], per frame measured.
#[cfg(windows)]
pub(crate) fn log_title_gap() {
    let [us, held, frames] = TITLE_GAP.each_ref().map(|a| a.swap(0, Ordering::Relaxed));
    if frames == 0 {
        return;
    }
    let ms = |us: u64| us as f64 / 1e3 / frames as f64;
    tracing::info!(
        "title CPU from a present returning to its next submission, ms per frame: {:.1}, of it held in our \
         queue work {:.1}, so the title's own (its calls into our command lists included) {:.1}",
        ms(us),
        ms(held),
        ms(us - held),
    );
}

/// [`BLOCKED_US`] since the last call, cleared: for each [`Why`], the
/// microseconds waiting for the queues' lock and in the pump.
#[cfg(windows)]
pub(crate) fn take_blocked() -> [[u64; 2]; WHY_NAMES.len()] {
    BLOCKED_US
        .each_ref()
        .map(|w| w.each_ref().map(|a| a.swap(0, Ordering::Relaxed)))
}

/// Runs held work that may run now (`why` for the frame line).
pub(crate) fn pump_held(why: Why) {
    #[cfg(windows)]
    {
        if crate::host::async_submit_enabled() {
            ensure_submit_worker();
            wake_submit();
        } else if crate::host::flight1_enabled() {
            pump_unlocked(why);
        } else {
            pump(&mut held_for(why), why);
        }
    }
    #[cfg(not(windows))]
    let _ = why;
}

/// The same unless another thread is at it (calls that poll: a fence's
/// value, our timer).
pub(crate) fn pump_held_if_free(why: Why) {
    #[cfg(windows)]
    {
        if crate::host::async_submit_enabled() {
            if !held().is_empty() {
                ensure_submit_worker();
                wake_submit();
            }
        } else if crate::host::flight1_enabled() {
            if let Ok(_drain) = SUBMIT_DRAIN.try_lock() {
                let _on = OnSubmit::enter();
                let entered = std::time::Instant::now();
                pump_ready_under_drain(why);
                add_run(why, entered);
            }
        } else if let Ok(mut held) = HELD.try_lock() {
            pump(&mut held, why);
        }
    }
    #[cfg(not(windows))]
    let _ = why;
}

/// How long a queue's head may wait before it runs anyway.
// shortcut: a safety valve against a wait nothing satisfies (a write we do
// not make), which would hold the title's fences until its 30 s watchdog
// ends the run; counted and logged, remove once runs show it never fires.
#[cfg(all(windows, not(test)))]
const VALVE: std::time::Duration = std::time::Duration::from_secs(10);
/// Tests share one process and its held work, and a suite under load stalls
/// a test's thread (or the held-work timer, on a lock a slow submission of
/// another test holds) past ten seconds: a held segment then ran without the
/// wait that orders it after another queue's write. No test depends on the
/// valve (the `hold` tests give `force_older_than` a limit of their own).
#[cfg(all(windows, test))]
const VALVE: std::time::Duration = std::time::Duration::from_secs(3600);

/// Whether ready segments across queues and chained bottom-of-pipe waits are
/// coalesced into one host submission (`VICTORY_COALESCE_PIECES`, on by default
/// unless `"0"`).
#[cfg(windows)]
fn coalesce_pieces() -> bool {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| {
        let on = std::env::var_os("VICTORY_COALESCE_PIECES").is_none_or(|v| v != "0");
        if !on {
            tracing::info!("queue piece coalescing disabled (VICTORY_COALESCE_PIECES=0)");
        }
        on
    })
}

/// Maximum number of host command lists joined into one coalesced submission
/// (`VICTORY_JOIN_LISTS`, default 64).
#[cfg(windows)]
fn join_lists() -> usize {
    static N: std::sync::OnceLock<usize> = std::sync::OnceLock::new();
    *N.get_or_init(|| {
        std::env::var("VICTORY_JOIN_LISTS")
            .ok()
            .and_then(|v| v.parse::<usize>().ok())
            .filter(|&n| n >= 1)
            .unwrap_or(64)
    })
}

/// Runs what held work may run now (`why`), its time counted in
/// [`BLOCKED_US`].
#[cfg(windows)]
fn pump(held: &mut Holding, why: Why) {
    let entered = std::time::Instant::now();
    pump_work(held, why);
    add_run(why, entered);
}

#[cfg(windows)]
fn pump_work(held: &mut Holding, why: Why) {
    if held.is_empty() {
        return;
    }
    let Some(host) = crate::host::host() else { return };
    let read = read_value;
    for (queue, waits) in held.force_older_than(VALVE) {
        account().forced += 1;
        tracing::warn!(queue = kind(queue), ?waits, "held work runs without its waits after {VALVE:?} (safety valve)");
    }
    let fence = |f: usize| crate::fence::value(f as This);
    let max_lists = join_lists();
    let will_write = |seg: &std::sync::Arc<crate::host::HostList>, wait: &crate::hold::ValueWait| {
        seg.will_write(wait)
    };
    let can_join = |piece: &[(std::sync::Arc<crate::host::HostList>, bool)],
                    next: &std::sync::Arc<crate::host::HostList>| {
        if piece.len() >= max_lists {
            return false;
        }
        let existing = piece
            .iter()
            .fold(crate::hold::BatchFlags::default(), |acc, (s, _)| {
                acc.merge(s.batch_flags())
            });
        crate::hold::can_join_batch(existing, next.batch_flags(), piece.len(), max_lists)
    };
    let started = std::time::Instant::now();
    let mut ran = std::time::Duration::ZERO;
    let mut run = |segments: Vec<(std::sync::Arc<crate::host::HostList>, bool)>| {
        let lists: Vec<&crate::host::HostList> = segments.iter().map(|(s, _)| &**s).collect();
        let barriers: Vec<bool> = segments.iter().map(|&(_, b)| b).collect();
        let at = std::time::Instant::now();
        #[cfg(feature = "diagnostics")]
        crate::copystat::add(crate::copystat::C::PumpExecutes, 1);
        host.execute_with_barriers(&lists, &barriers);
        ran += at.elapsed();
        drop(lists);
        segments
            .into_iter()
            .for_each(|(s, _)| crate::list::recycle(s));
    };
    let mut signal = |f: usize, value: u64| {
        crate::fence::set(f as This, value);
        crate::fence::let_go(f as This);
    };
    let pumped = held.pump(&mut crate::hold::World {
        read: &read,
        fence: &fence,
        will_write: &will_write,
        can_join: &can_join,
        coalesce: coalesce_pieces(),
        run: &mut run,
        signal: &mut signal,
    });
    if pumped.segments > 0 || pumped.unknown > 0 {
        let mut a = account();
        a.pump += started.elapsed().saturating_sub(ran);
        a.pieces += pumped.pieces;
        a.segments += pumped.segments;
        a.unknown += pumped.unknown;
        a.coalesced_bop_waits += pumped.coalesced_bop_waits;
        a.hazard_splits += pumped.hazard_splits;
        for (queue, waited) in pumped.waited {
            a.released.push((kind(queue), waited, why));
        }
    }
    // Once work stays held, a timer tests it every 2 ms: memory the title
    // writes from the CPU, or the valve, may let it go with no call of ours.
    if held.is_empty() {
        return;
    }
    start_held_timer();
}

/// The value a wait tests (4 or 8 bytes), if the memory can be read.
#[cfg(windows)]
fn read_value(address: u64, bytes: u8) -> Option<u64> {
    let mut value = [0u8; 8];
    crate::host::read_into(address as *const u8, &mut value[..usize::from(bytes.min(8))]).then(|| u64::from_le_bytes(value))
}

/// A queue's type, for logs (0 graphics, 2 compute, 3 copy). Titles do not
/// release queues that hold work.
#[cfg(windows)]
fn kind(queue: usize) -> u32 {
    self::queue(queue as This).kind
}

/// What queues did with held work since the last presented frame.
#[cfg(windows)]
#[derive(Default)]
struct Account {
    submissions: u32,
    pieces: u32,
    segments: u32,
    /// Segments that waited: queue type, how long, what ran them.
    released: Vec<(u32, std::time::Duration, Why)>,
    unknown: u32,
    forced: u32,
    mappings: u32,
    coalesced_bop_waits: u32,
    hazard_splits: u32,
    /// Time in the pump outside host submissions.
    pump: std::time::Duration,
}

#[cfg(windows)]
fn account() -> std::sync::MutexGuard<'static, Account> {
    static ACCOUNT: std::sync::OnceLock<std::sync::Mutex<Account>> = std::sync::OnceLock::new();
    ACCOUNT.get_or_init(Default::default).lock().unwrap_or_else(|e| e.into_inner())
}

/// The frame line of held work: what waited, how long and what released it,
/// and what is still held at the present, with the waits that do not hold.
#[cfg(windows)]
fn account_frame(frame: u64) {
    let (cuts, early) = crate::list::take_cuts();
    let (fence_waits, fence_waited) = crate::fence::take_waited();
    let a = std::mem::take(&mut *account());
    let blocked = held().blocked(&read_value);
    if a.released.is_empty() && blocked.is_empty() && a.forced == 0 && a.unknown == 0 && a.mappings == 0 {
        return;
    }
    let longest = a.released.iter().map(|r| r.1).max().unwrap_or_default();
    let mut by = [0u32; 8];
    for r in &a.released {
        by[r.2 as usize] += 1;
    }
    tracing::info!(
        frame,
        submissions = a.submissions,
        pieces = a.pieces,
        segments = a.segments,
        cuts,
        early_resets = early,
        waited = a.released.len(),
        coalesced_bop_waits = a.coalesced_bop_waits,
        hazard_splits = a.hazard_splits,
        longest_ms = longest.as_secs_f64() * 1e3,
        pump_ms = a.pump.as_secs_f64() * 1e3,
        fence_waits,
        fence_wait_ms = fence_waited.as_secs_f64() * 1e3,
        "held work this frame (submissions; host submissions, segments; lists cut at waits; lists reset or released \
         while held, each keeping its own host list; segments that waited, released by \
         submission/signal/wait/fence/present/sync/timer/flight {by:?}; coalesced BoP waits and hazard splits; time in the pump \
         outside host submissions; events the title waited on for fence values and their summed waits); held at the \
         present: {}",
        blocked.len()
    );
    tracing::debug!(
        frame,
        "held work released this frame (queue type, ms, by): {:?}",
        a.released.iter().map(|(k, d, w)| (k, d.as_secs_f64() * 1e3, w)).collect::<Vec<_>>()
    );
    for (queue, items, waited, unmet) in &blocked {
        let level_warn = blocked.len() > 1;
        let unmet: Vec<String> = unmet
            .iter()
            .map(|w| format!("{:#x} cmp {} {:#x} (is {:?})", w.address, w.comparison, w.reference, read_value(w.address, w.bytes)))
            .collect();
        if level_warn {
            tracing::warn!(frame, queue = kind(*queue), items, waited_ms = waited.as_millis(), ?unmet, "two or more queues hold work at the present");
        } else {
            tracing::info!(frame, queue = kind(*queue), items, waited_ms = waited.as_millis(), ?unmet, "a queue holds work at the present");
        }
    }
    if a.forced > 0 || a.unknown > 0 || a.mappings > 0 {
        tracing::warn!(
            frame,
            forced = a.forced,
            unknown = a.unknown,
            mappings = a.mappings,
            "held work: run without its waits (valve), waits with an unknown comparison (taken as holding), \
             page mappings done ahead of held work"
        );
    }
}

/// One plane of a `PresentX` call (0x48-byte stride): frame token,
/// resource count, and pointer to the plane's resources.
#[repr(C)]
pub(crate) struct Plane {
    token: u64,
    count: u32,
    resources: *const This,
    _rest: [u64; 6],
}
const _: () = assert!(std::mem::size_of::<Plane>() == 0x48);
/// Submission tagged with the frame it belongs to `(count, lists, frame
/// token, flags)`; frames are paced elsewhere, so it runs as a plain one.
extern "system" fn execute_x(this: This, count: u32, lists: *const This, _token: u64, _flags: u32) {
    execute(this, count, lists);
}
/// Signal with sync settings for the top and bottom of the pipe; our
/// submissions are complete when they run, so it signals as Signal does.
extern "system" fn signal2_x(
    this: This,
    fence: This,
    value: u64,
    _bottom: *const u8,
    _top: *const u8,
) -> HRESULT {
    signal(this, fence, value)
}
/// Markers and events only matter to profilers.
extern "system" fn marker(_: This) {}
/// At once, or behind the work the queue holds (its fence, and the memory
/// the title reads it at, move only once that work has run).
extern "system" fn signal(this: This, fence: This, value: u64) -> HRESULT {
    if fence.is_null() {
        return E_INVALIDARG;
    }
    #[cfg(windows)]
    {
        let mut held = held_for(Why::Signal);
        if held.holds_work(this as usize) {
            crate::fence::keep(fence);
            held.push(this as usize, Item::Signal(fence as usize, value));
            if crate::host::async_submit_enabled() {
                drop(held);
                ensure_submit_worker();
                wake_submit();
            } else if crate::host::flight1_enabled() {
                drop(held);
                pump_unlocked(Why::Signal);
            } else {
                pump(&mut held, Why::Signal);
            }
        } else if crate::host::flight1_enabled()
            && crate::host::host().is_some_and(|h| h.attach_signal(fence, value, false))
        {
            drop(held);
        } else {
            crate::fence::set(fence, value);
            // Another queue may wait for this fence.
            if crate::host::async_submit_enabled() {
                drop(held);
                ensure_submit_worker();
                wake_submit();
            } else if crate::host::flight1_enabled() {
                drop(held);
                pump_unlocked(Why::Signal);
            } else {
                pump(&mut held, Why::Signal);
            }
        }
    }
    #[cfg(not(windows))]
    {
        let _ = this;
        crate::fence::set(fence, value);
    }
    S_OK
}
/// The queue's later work waits until the fence reaches `value`.
///
/// shortcut: the fence is not kept alive while the queue waits for it;
/// Grand Theft Auto VI makes no queue wait for a fence.
extern "system" fn wait(this: This, fence: This, value: u64) -> HRESULT {
    if fence.is_null() {
        return E_INVALIDARG;
    }
    #[cfg(windows)]
    if crate::host::host().is_some() && crate::fence::value(fence) < value {
        let mut held = held_for(Why::Wait);
        held.push(this as usize, Item::FenceWait(fence as usize, value));
        if crate::host::async_submit_enabled() {
            drop(held);
            ensure_submit_worker();
            wake_submit();
        } else if crate::host::flight1_enabled() {
            drop(held);
            pump_unlocked(Why::Wait);
        } else {
            pump(&mut held, Why::Wait);
        }
    }
    #[cfg(not(windows))]
    let _ = (this, value);
    S_OK
}
/// Ticks per second of GPU timestamps (see [`Written::GpuTimestamp`]).
extern "system" fn timestamp_frequency(_: This, out: *mut u64) -> HRESULT {
    if out.is_null() {
        return E_INVALIDARG;
    }
    // SAFETY: checked non-null.
    unsafe { out.write(TIMESTAMP_FREQUENCY) };
    S_OK
}
const TIMESTAMP_FREQUENCY: u64 = 100_000_000;

/// What a list's bottom-of-pipe write stores when the list has run.
#[derive(Debug, Clone, Copy)]
pub(crate) enum Written {
    Value(u64),
    /// The GPU clock: [`TIMESTAMP_FREQUENCY`] ticks per second since the
    /// process started (our GPU work is done on the CPU's timeline).
    GpuTimestamp,
    /// The CPU's performance counter.
    SystemTimestamp,
}

impl Written {
    /// Stores the low `bytes` (4 or 8) of the value at `at`.
    pub(crate) fn write(self, at: *mut u8, bytes: usize) {
        let value = match self {
            Written::Value(v) => v,
            Written::GpuTimestamp => {
                static START: std::sync::OnceLock<std::time::Instant> = std::sync::OnceLock::new();
                let elapsed = START.get_or_init(std::time::Instant::now).elapsed();
                (elapsed.as_nanos() / (1_000_000_000 / u128::from(TIMESTAMP_FREQUENCY))) as u64
            }
            Written::SystemTimestamp => system_timestamp(),
        };
        // SAFETY: the title names writable GPU memory (unified with ours).
        unsafe {
            if bytes == 4 {
                at.cast::<u32>().write_unaligned(value as u32);
            } else {
                at.cast::<u64>().write_unaligned(value);
            }
        }
    }
}

fn system_timestamp() -> u64 {
    #[cfg(windows)]
    {
        let mut count = 0;
        // SAFETY: a valid out pointer.
        unsafe { windows_sys::Win32::System::Performance::QueryPerformanceCounter(&mut count) };
        count as u64
    }
    #[cfg(not(windows))]
    0
}

/// Frames presented so far.
static FRAMES: AtomicU64 = AtomicU64::new(0);

pub(crate) fn presented() -> u64 {
    FRAMES.load(Ordering::Relaxed)
}

/// Every plane's resources. Only the start of a plane description is
/// known (token, count, resources), so the planes after the first are
/// found by their stride: the first offset past the first plane where the
/// same frame token is followed by a count of 1 to 16 and a list whose
/// first entry is one of our resources. All reads there are checked.
#[cfg(windows)]
fn plane_resources(planes: u32, plane: *const Plane) -> Vec<Vec<This>> {
    let read_u64 = |at: usize| {
        let mut b = [0u8; 8];
        crate::host::read_into(at as *const u8, &mut b).then(|| u64::from_le_bytes(b))
    };
    let list = |at: usize| -> Option<Vec<This>> {
        let count = read_u64(at + 8)? as u32;
        let resources = read_u64(at + 16)? as usize;
        if !(1..=16).contains(&count) || resources == 0 {
            return None;
        }
        let all: Option<Vec<This>> = (0..count as usize)
            .map(|i| read_u64(resources + 8 * i).map(|r| r as This))
            .collect();
        all.filter(|r| r.iter().all(|&r| crate::resource::is_resource(r)))
    };
    let base = plane as usize;
    if planes == 0 || base == 0 {
        return Vec::new();
    }
    let Some(first) = list(base) else {
        return Vec::new();
    };
    let mut all = vec![first];
    if planes > 1
        && let Some(token) = read_u64(base)
        && let Some(stride) = (24..=256)
            .step_by(8)
            .find(|&o| read_u64(base + o) == Some(token) && list(base + o).is_some())
    {
        for k in 1..planes.min(4) as usize {
            match list(base + k * stride) {
                Some(r) => all.push(r),
                None => break,
            }
        }
    }
    all
}

/// Logs what the title presents whenever it changes: each plane's
/// resources as (width, height, host format) (#15: the first plane held a
/// black 1920x1080 texture while a larger render target held the picture).
#[cfg(windows)]
fn log_planes(frame: u64, planes: u32, found: &[Vec<This>], shown: usize) {
    use std::sync::Mutex;
    /// A resource's width, height and host format.
    type Shown = (u64, u32, i32);
    static LAST: Mutex<Option<(u32, Vec<Vec<Shown>>)>> = Mutex::new(None);
    let sizes: Vec<Vec<Shown>> = found
        .iter()
        .map(|plane| {
            plane
                .iter()
                .map(|&r| {
                    crate::resource::host_resource(r).map_or((0, 0, -1), |t| {
                        // SAFETY: a live resource.
                        let d = unsafe { t.GetDesc() };
                        (d.Width, d.Height, d.Format.0)
                    })
                })
                .collect()
        })
        .collect();
    let now = Some((planes, sizes));
    let mut last = LAST.lock().unwrap_or_else(|e| e.into_inner());
    if *last != now {
        let (planes, sizes) = now.as_ref().expect("just set");
        tracing::info!(
            frame,
            planes,
            found = sizes.len(),
            shown,
            resources = ?sizes,
            "presented planes: each plane's resources as (width, height, format); the first resource of plane `shown` is shown"
        );
        *last = now;
    }
}

/// The plane whose first resource is shown: `VICTORY_PRESENT_PLANE=<n>`
/// (counted from 0) when that plane was found, else the first plane with a
/// resource (#15: on Series S plane 0 is the scene, at the title's output
/// size, and plane 1 the 3840x2160 UI drawn over it).
#[cfg(windows)]
fn shown_plane(found: &[Vec<This>]) -> usize {
    static PLANE: std::sync::OnceLock<Option<usize>> = std::sync::OnceLock::new();
    let want = *PLANE.get_or_init(|| {
        std::env::var("VICTORY_PRESENT_PLANE").ok().and_then(|v| v.parse().ok())
    });
    let areas: Vec<u64> = found
        .iter()
        .map(|plane| {
            plane.first().and_then(|&r| crate::resource::host_resource(r)).map_or(0, |t| {
                // SAFETY: a live resource.
                let d = unsafe { t.GetDesc() };
                d.Width * u64::from(d.Height)
            })
        })
        .collect();
    pick_plane(want, &areas)
}

/// The plane to show given an override and each plane's first resource
/// area (0 when unknown): the override when in range, else the first plane
/// with a known resource, else plane 0.
#[cfg(any(windows, test))]
fn pick_plane(want: Option<usize>, areas: &[u64]) -> usize {
    if let Some(w) = want
        && w < areas.len()
    {
        return w;
    }
    areas.iter().position(|&a| a > 0).unwrap_or(0)
}

/// Shows the first resource of one plane in our window (the first plane
/// unless `VICTORY_PRESENT_PLANE` picks another).
///
/// shortcut: one plane, no scaling or placement, the source expected in
/// the common state.
extern "system" fn present(
    _: This,
    planes: u32,
    plane_parameters: *const Plane,
    _parameters: *const u8,
) -> HRESULT {
    // The title's own time starts when this returns (`TitleGap`).
    #[cfg(windows)]
    let _returning = PresentReturning;
    // What may run before the frame is shown does.
    // shortcut: PresentX does not wait behind work the queue still holds (it
    // shows the frame as far as it ran); the frame line names such work.
    #[cfg(windows)]
    if crate::host::async_submit_enabled() || crate::host::flight1_enabled() {
        note_drain(Drain::Present);
        drain_submit(Why::Present);
    } else {
        pump_held(Why::Present);
    }
    #[cfg(not(windows))]
    pump_held(Why::Present);
    let frame = FRAMES.fetch_add(1, Ordering::Relaxed) + 1;
    #[cfg(windows)]
    account_frame(frame);
    #[cfg(windows)]
    crate::shadow::trim_pool();
    #[cfg(feature = "diagnostics")]
    crate::frame::presented();
    if frame.is_power_of_two() {
        let (draws, dispatches) = crate::list::take_counts();
        #[cfg(windows)]
        let (drawn, dispatched) = {
            static DRAWN: AtomicU64 = AtomicU64::new(0);
            static DISPATCHED: AtomicU64 = AtomicU64::new(0);
            let since = |last: &AtomicU64, now: u64| now - last.swap(now, Ordering::Relaxed);
            (
                since(&DRAWN, crate::draw::drawn()),
                since(&DISPATCHED, crate::draw::dispatched()),
            )
        };
        #[cfg(not(windows))]
        let (drawn, dispatched) = (0, 0);
        tracing::info!(
            frame,
            draws,
            drawn,
            dispatches,
            dispatched,
            "frame presented (since the last line: draws and dispatches, and those run on the host)"
        );
        #[cfg(windows)]
        if frame >= 256 {
            let (hits, misses) = crate::shader_cache::counts();
            tracing::info!(hits, misses, "shaders from the cache and translated so far");
        }
    }
    #[cfg(windows)]
    crate::hdr_probe::on_present(frame);
    // Draws on the host so far, for the frame recorder's world frames
    // (`VICTORY_RECORD_AT=w<n>`) and the benchmark: read where
    // `capture::presented` reads them, so that both tell the same frame as
    // the world's first.
    #[cfg(windows)]
    let drawn = crate::draw::drawn();
    #[cfg(not(windows))]
    let drawn = 0;
    #[cfg(all(windows, feature = "diagnostics"))]
    {
        crate::capture::presented(frame);
        if crate::capture::listed(frame) || crate::capture::has_targets() {
            crate::capture::dump_targets(frame);
        }
    }
    #[cfg(windows)]
    let mut black_frame = false;
    #[cfg(not(windows))]
    let black_frame = false;
    #[cfg(windows)]
    let found = plane_resources(planes, plane_parameters);
    #[cfg(windows)]
    let shown = shown_plane(&found);
    #[cfg(windows)]
    log_planes(frame, planes, &found, shown);
    #[cfg(windows)]
    if let Some(&resource) = found.get(shown).and_then(|p| p.first())
        && let Some(texture) = crate::resource::host_resource(resource)
        && let Some(host) = crate::host::host()
    {

        const COPY_SOURCE: u32 = 0x800;
        let state = crate::resource::host_transition(resource, COPY_SOURCE)
            .map_or(COPY_SOURCE, |(_, before)| before);
        // The other plane's first resource, blended over the picture unless
        // `VICTORY_PRESENT_COMPOSITE=0` (#15: the 4K UI on Series S).
        let other = (found.len() == 2)
            .then(|| found[1 - shown].first().copied())
            .flatten()
            .filter(|&o| o != resource && crate::composite::wanted())
            .and_then(|o| Some((o, crate::resource::host_resource(o)?)));
        let overlay_state = other.as_ref().map(|(o, _)| {
            crate::resource::host_transition(*o, COPY_SOURCE).map_or(COPY_SOURCE, |(_, before)| before)
        });
        let overlay = other.as_ref().zip(overlay_state).map(|((_, t), s)| (t, s));
        let shown = host.present(&texture, state, overlay, frame);
        black_frame = shown.unwrap_or(false);
        if let Some(((o, _), s)) = other.as_ref().zip(overlay_state) {
            crate::resource::host_transition(*o, s);
        }
        crate::resource::host_transition(resource, state);
        // The frame recorder's end of a frame (`VICTORY_RECORD`): the picture
        // is in `state` again.
        crate::replay::record::present(frame, Some((&texture, state)), drawn);
        #[cfg(feature = "diagnostics")]
        blinks(frame, resource, state, shown);
        #[cfg(not(feature = "diagnostics"))]
        let _ = shown;
    } else {
        crate::replay::record::present(frame, None, drawn);
    }
    #[cfg(not(windows))]
    let _ = (planes, plane_parameters);
    crate::bench::on_present(drawn, black_frame);
    S_OK
}
/// Diagnostics: logs a presented frame that is black after one that was
/// not, with what was presented and how many draws ran since.
#[cfg(all(windows, feature = "diagnostics"))]
fn blinks(frame: u64, resource: This, state: u32, shown: Option<bool>) {
    static LAST: std::sync::Mutex<(bool, u64, u64)> = std::sync::Mutex::new((false, 0, 0));
    let mut last = LAST.lock().unwrap_or_else(|e| e.into_inner());
    let drawn = crate::draw::drawn();
    let black = shown.unwrap_or(false);
    let draws = crate::draw::take_frame();
    if black && !last.0 {
        last.2 += 1;
        tracing::warn!(
            frame,
            resource = ?resource,
            state = format_args!("{state:#x}"),
            draws = drawn - last.1,
            copied = shown.is_some(),
            black_frames = last.2,
            levels = ?crate::capture::recent_levels(),
            "a black frame after a frame that was not"
        );
        for d in draws {
            tracing::warn!("black frame {frame}: {d}");
        }
        AFTER.store(frame + 6, Ordering::Relaxed);
    }
    // Six frames on: the levels around it (a fade or a single drop).
    static AFTER: AtomicU64 = AtomicU64::new(0);
    if AFTER.load(Ordering::Relaxed) == frame {
        tracing::warn!(frame, levels = ?crate::capture::recent_levels(), "levels after the black frame");
    }
    *last = (black, drawn, last.2);
}

/// A run of destination pages: how they are mapped (0 pool pages from
/// `start` on, 1 pool page `start` each, 2 unmapped, 3 left as they are)
/// and how many.
#[repr(C)]
#[derive(Clone, Copy)]
pub(crate) struct PageRange {
    kind: u32,
    start: u32,
    count: u32,
}

/// The ranges mapped from a destination address on.
#[repr(C)]
pub(crate) struct PageBatch {
    destination: u64,
    count: u32,
    ranges: *const PageRange,
}

/// Bytes per mapped page.
const PAGE: u64 = 64 << 10;

/// `CopyPageMappingsBatchX(count, batches, pool, pool pages, flags)`: maps
/// pages of a pool (memory the title mapped physical pages into) at the
/// batches' addresses. Grand Theft Auto VI streams textures into pool pages
/// and maps them into its tiled heap this way (title+0xf2c470, ranges of
/// one constant page or of unmapped pages).
///
/// On the console the batch is a queue operation, in the queue's order.
/// With `VICTORY_ASYNC_SUBMIT=1` or `VICTORY_FLIGHT1=1` it is: while the
/// queue holds work or a host submission is in flight, the batch is queued
/// as an [`Item::Action`] (the submit worker lands the flight, maps, then
/// prepares the queue's later work) and the title's thread goes on; else it
/// is done at once. (It drained all queued work before: 124-182 ms of a
/// 428 ms frame on the title's thread.)
///
/// shortcut: otherwise done at once, ahead of work the queue still holds
/// (counted in the frame line).
extern "system" fn copy_page_mappings(
    this: This,
    count: u32,
    batches: *const PageBatch,
    pool: u64,
    pool_pages: u32,
    _flags: u32,
) -> HRESULT {
    let Some(map) = crate::resource::page_mapper() else {
        return E_INVALIDARG;
    };
    if batches.is_null() && count > 0 {
        return E_INVALIDARG;
    }
    // The batches as the title passed them: it may reuse them once we return.
    let batches: Vec<(u64, Vec<PageRange>)> = if count == 0 {
        Vec::new()
    } else {
        // SAFETY: the title passes `count` batches.
        unsafe { std::slice::from_raw_parts(batches, count as usize) }
            .iter()
            .filter(|b| !b.ranges.is_null())
            .map(|b| {
                // SAFETY: each batch holds `count` ranges.
                let ranges = unsafe { std::slice::from_raw_parts(b.ranges, b.count as usize) };
                (b.destination, ranges.to_vec())
            })
            .collect()
    };
    #[cfg(windows)]
    {
        let queued = crate::host::async_submit_enabled() || crate::host::flight1_enabled();
        let mut held = if queued { held_for(Why::Sync) } else { held() };
        if queued
            && (held.holds_work(this as usize)
                || crate::host::host().is_some_and(crate::host::Host::has_in_flight))
        {
            held.push(
                this as usize,
                Item::Action(Box::new(move || map_pages(map, &batches, pool, pool_pages))),
            );
            drop(held);
            note_drain(Drain::MappingQueued);
            if crate::host::async_submit_enabled() {
                ensure_submit_worker();
                wake_submit();
            } else {
                pump_unlocked(Why::Sync);
            }
            return S_OK;
        }
        if held.holds_work(this as usize) {
            account().mappings += 1;
        }
        drop(held);
        if queued {
            note_drain(Drain::MappingAtOnce);
        }
    }
    #[cfg(not(windows))]
    let _ = this;
    map_pages(map, &batches, pool, pool_pages);
    S_OK
}

/// Maps `batches` of `pool` (`pool_pages` pages) with `map`.
fn map_pages(
    map: crate::resource::PageMapper,
    batches: &[(u64, Vec<PageRange>)],
    pool: u64,
    pool_pages: u32,
) {
    let mut ok = true;
    for (destination, ranges) in batches {
        let mut at = *destination;
        for r in ranges {
            let len = u64::from(r.count) * PAGE;
            let inside = |pages: u32| r.start.checked_add(pages).is_some_and(|e| e <= pool_pages);
            let source = pool + u64::from(r.start) * PAGE;
            ok &= match r.kind {
                0 if inside(r.count) => map(at, source, len),
                1 if inside(1) => (0..u64::from(r.count)).all(|k| map(at + k * PAGE, source, PAGE)),
                2 => map(at, 0, len),
                3 => true,
                _ => false,
            };
            at += len;
        }
    }
    if !ok {
        static ONCE: std::sync::Once = std::sync::Once::new();
        ONCE.call_once(|| {
            tracing::warn!(pool = format_args!("{pool:#x}"), "a page mapping failed")
        });
    }
}

extern "system" fn desc(this: This, out: *mut QueueDesc) {
    if !out.is_null() {
        // SAFETY: checked non-null; the title passes room for a description.
        unsafe { out.write(queue(this).desc) };
    }
}

/// The queue methods by their slot in older titles' queues (the 2020-era
/// layout): IUnknown 3, object 4, device child 1, queue 26 (12 as on PC,
/// then console ones).
const METHODS: [(usize, *const ()); 15] = [
    (0, query as *const ()),
    (2, release as *const ()),
    (7, object::parent_device as *const ()), // 0x38
    (10, execute as *const ()),              // 0x50
    (11, marker as *const ()),               // 0x58 set marker
    (12, marker as *const ()),               // 0x60 begin event
    (13, marker as *const ()),               // 0x68 end event
    (14, signal as *const ()),               // 0x70
    (15, wait as *const ()),                 // 0x78
    (16, timestamp_frequency as *const ()),  // 0x80
    (26, desc as *const ()),                 // 0xd0
    (29, copy_page_mappings as *const ()),   // 0xe8 page mapping batch
    (30, present as *const ()),              // 0xf0
    (33, execute_x as *const ()),            // 0x108
    (32, signal2_x as *const ()),            // 0x100
];

/// Newer titles' queues have two more methods before Signal, which moved
/// from 0x70 to 0x80: every slot from 0x70 on is two later (taken here as
/// starting exactly at 0x70, the methods before it kept in place).
/// Inferred from Grand Theft Auto VI's calls: Signal 0x80 `(fence,
/// value)`, timestamp frequency 0x90 (title+0x24841a6), suspend and resume
/// 0xb8/0xc0, PresentX 0x100, the frame-token submission 0x118 `(count,
/// lists, token, flags)` (title+0x24821dd).
const fn newer_slot(slot: usize) -> usize {
    if slot >= 14 { slot + 2 } else { slot }
}

const fn queue_table(newer: bool) -> [*const (); 36] {
    let mut f = object::d3d12_object(object::missing_table::<QUEUE, 36>());
    let mut i = 0;
    while i < METHODS.len() {
        let (slot, method) = METHODS[i];
        f[if newer { newer_slot(slot) } else { slot }] = method;
        i += 1;
    }
    f
}

static FUNCTIONS: Table<36> = Table(queue_table(false));
static NEWER_FUNCTIONS: Table<36> = Table(queue_table(true));

#[cfg(test)]
mod tests {

    /// The title's gap runs from a present returning to its next submission,
    /// once a present: later submissions add nothing.
    #[cfg(windows)]
    #[test]
    fn the_title_gap_runs_from_a_present_to_the_next_submission() {
        let before = super::TITLE_GAP.each_ref().map(|a| a.load(std::sync::atomic::Ordering::Relaxed));
        drop(super::PresentReturning);
        std::thread::sleep(std::time::Duration::from_millis(5));
        super::title_submits();
        super::title_submits();
        let after = super::TITLE_GAP.each_ref().map(|a| a.load(std::sync::atomic::Ordering::Relaxed));
        // Other tests present and submit too, and the frame report takes the
        // counts: only what this one adds at least is checked.
        assert!(after[2] > before[2] || after[2] >= 1, "a frame measured");
        assert!(after[0] >= before[0] + 5_000 || after[0] >= 5_000, "{} then {} us", before[0], after[0]);
    }
    use super::*;
    use crate::object::slot;

    /// A lock another thread holds is waited for, and the wait is counted
    /// in microseconds; a free lock adds next to nothing.
    #[test]
    fn a_timed_lock_counts_its_wait() {
        let lock = std::sync::Mutex::new(());
        let waited = AtomicU64::new(0);
        drop(lock_timed(&lock, &waited));
        assert!(waited.load(Ordering::Relaxed) < 5_000, "a free lock");
        let (taken, take) = std::sync::mpsc::channel();
        std::thread::scope(|s| {
            s.spawn(|| {
                let _held = lock.lock().unwrap();
                taken.send(()).unwrap();
                std::thread::sleep(std::time::Duration::from_millis(30));
            });
            take.recv().unwrap();
            let before = waited.load(Ordering::Relaxed);
            drop(lock_timed(&lock, &waited));
            let counted = waited.load(Ordering::Relaxed) - before;
            assert!(
                counted >= 20_000,
                "waited {counted} us for a lock held 30 ms"
            );
        });
    }

    /// Which plane gets shown: the one `VICTORY_PRESENT_PLANE` names when
    /// it names one, else the first plane with a known resource. It only
    /// weighs the areas it is handed, so it runs on every host.
    #[test]
    fn the_first_plane_is_shown_unless_one_is_named() {
        // Series S: the 1080p scene, then the 4K UI that goes over it.
        let series_s = [1920 * 1080, 3840 * 2160];
        assert_eq!(pick_plane(None, &series_s), 0);
        assert_eq!(pick_plane(Some(1), &series_s), 1);
        // An override past the planes found falls back to the first.
        assert_eq!(pick_plane(Some(5), &series_s), 0);
        // A first plane whose resource is not known yet is passed over.
        assert_eq!(pick_plane(None, &[0, 3840 * 2160]), 1);
        // Series X: one plane. Unknown sizes keep the first.
        assert_eq!(pick_plane(None, &[3840 * 2160]), 0);
        assert_eq!(pick_plane(None, &[0, 0]), 0);
        assert_eq!(pick_plane(None, &[]), 0);
    }

    /// Two planes 64 bytes apart (as a longer plane description would
    /// place them), each listing one of our textures, are both found; a
    /// list naming something else is not taken for a plane. This reads
    /// title memory through the host, so it runs where a host exists.
    #[cfg(windows)]
    #[test]
    fn planes_after_the_first_are_found_by_their_stride() {
        let texture = || {
            let layout = std::alloc::Layout::from_size_align(1 << 16, 1 << 16).unwrap();
            // SAFETY: a non-zero size; leaked, as title memory outlives a test.
            let address = unsafe { std::alloc::alloc_zeroed(layout) } as u64;
            let words: [u32; 14] = [3, 0, 0, 0, 64, 0, 64, 1 | 1 << 16, 28, 1, 0, 0x105, 0, 0];
            let iid = guid(0x696442be, 0xa72e, 0x4059, [0xbc, 0x79, 0x5b, 0x5c, 0x98, 0x04, 0x0f, 0xad]);
            let mut out = std::ptr::null();
            let made = crate::resource::create_at_address(std::ptr::null(), address, words.as_ptr().cast(), 0, std::ptr::null(), &iid, &mut out);
            assert_eq!(made, S_OK);
            out
        };
        let (a, b) = (texture(), texture());
        let (list_a, list_b) = (Box::new([a]), Box::new([b]));
        let mut planes = [0u64; 16];
        planes[0] = 7;
        planes[1] = 1;
        planes[2] = list_a.as_ptr() as u64;
        planes[8] = 7;
        planes[9] = 1;
        planes[10] = list_b.as_ptr() as u64;
        let found = plane_resources(2, planes.as_ptr().cast());
        assert_eq!(found, vec![vec![a], vec![b]]);
        // The second plane names a list of something that is not ours.
        let bogus = Box::new([planes.as_ptr() as This]);
        planes[10] = bogus.as_ptr() as u64;
        assert_eq!(plane_resources(2, planes.as_ptr().cast()), vec![vec![a]]);
    }

    /// Where a queue's signal is: newer layouts have it two slots later.
    /// Another test may switch layouts while this one runs, so the queue's
    /// own table decides.
    fn signal_slot(q: This) -> usize {
        // SAFETY: one of our queues; its first field is the table.
        let table = unsafe { *q.cast::<*const *const ()>() };
        if table == NEWER_FUNCTIONS.0.as_ptr() {
            0x80
        } else {
            0x70
        }
    }

    #[test]
    fn queue_signals_complete_at_once() {
        let null = std::ptr::null();
        let desc = QueueDesc {
            kind: 0,
            priority: 0,
            flags: 0,
            nodes: 0,
        };
        let mut q = null;
        assert_eq!(create(null, &desc, &QUEUE_IIDS[0], &mut q), S_OK);
        // SAFETY: the title reads the type at +0x10.
        assert_eq!(unsafe { q.cast::<u8>().add(0x10).cast::<u32>().read() }, 0);
        let mut fence = null;
        let fence_iid = &crate::fence::FENCE_IIDS[0];
        assert_eq!(
            crate::fence::create(null, 0, 0, fence_iid, &mut fence),
            S_OK
        );
        let signal = slot::<extern "system" fn(This, This, u64) -> HRESULT>(q, signal_slot(q));
        assert_eq!(signal(q, fence, 7), S_OK);
        assert_eq!(
            slot::<extern "system" fn(This) -> u64>(fence, 0x40)(fence),
            7
        );
        assert_eq!(slot::<extern "system" fn(This) -> u32>(q, 0x10)(q), 0);
    }

    /// Grand Theft Auto VI's compute queue: `{COMPUTE, 0, pipe 1, queue 0}`.
    #[test]
    fn console_queues_are_created_as_on_pc() {
        let null = std::ptr::null();
        let desc = QueueDescX {
            kind: 2,
            flags: 0,
            engine_or_pipe: 1,
            queue: 0,
        };
        let mut q = null;
        assert_eq!(create_x(null, &desc, &QUEUE_IIDS[0], &mut q), S_OK);
        assert_eq!(queue(q).desc.kind, 2);
        let none = std::ptr::null();
        assert_eq!(create_x(null, none, &QUEUE_IIDS[0], &mut q), E_INVALIDARG);
    }

    #[test]
    fn newer_queues_have_their_methods_two_slots_later() {
        let (old, new) = (queue_table(false), queue_table(true));
        assert_eq!(old[10], new[10]); // submission stays at 0x50
        assert_eq!(old[14], new[16]); // signal 0x70 → 0x80
        assert_eq!(old[30], new[32]); // present 0xf0 → 0x100
        assert_eq!(new[35], execute_x as *const ()); // 0x118
    }

    #[test]
    fn timestamps_and_values_are_written() {
        let mut at = [0u64; 2];
        Written::Value(0x1_0000_0002).write(at.as_mut_ptr().cast(), 4);
        assert_eq!(at[0], 2);
        Written::GpuTimestamp.write(at[1..].as_mut_ptr().cast(), 8);
        let first = at[1];
        std::thread::sleep(std::time::Duration::from_millis(2));
        Written::GpuTimestamp.write(at[1..].as_mut_ptr().cast(), 8);
        assert!(at[1] > first);
    }
}
