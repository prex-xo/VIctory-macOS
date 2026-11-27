//! Priority-lane scheduling and bounded wait budgets for asynchronous host
//! pipeline compilation.
//!
//! Requested compute pipelines jump ahead of requested graphics pipelines and
//! early pre-warms, and get their own bounded wall-clock wait window
//! (`VICTORY_SYNC_COMPUTE_PSO`, on by default) so one-time compute dispatches
//! (such as initialization or geometry/skinning bakes) are not dropped when a
//! cold frame exhausts the shorter graphics compilation budget.

use std::collections::VecDeque;
use std::ffi::OsStr;
#[cfg(windows)]
use std::sync::OnceLock;
use std::sync::{Condvar, Mutex};
use std::time::{Duration, Instant};

/// Default waiting budget per title frame for graphics pipelines (and for
/// compute pipelines when `VICTORY_SYNC_COMPUTE_PSO=0` or after a pipeline
/// exceeds [`COMPUTE_BUDGET`]).
pub(crate) const GRAPHICS_BUDGET: Duration = Duration::from_millis(500);

/// Wall-clock cap for cold compute pipeline waits, both per title frame (from
/// the frame's first compute wait across all recording threads) and per
/// pipeline (from when a worker starts compiling it). Bounding both prevents
/// a cold frame with many compute shaders or a stalled compiler from tripping
/// a title's frame watchdog.
pub(crate) const COMPUTE_BUDGET: Duration = Duration::from_secs(10);

/// Whether a default-on switch is enabled: unset or any value other than `"0"`.
pub(crate) fn value_is_enabled(value: Option<&OsStr>) -> bool {
    value.is_none_or(|v| v != "0")
}

/// Whether cold compute dispatches wait within [`COMPUTE_BUDGET`] and get
/// priority over queued graphics pipelines (`VICTORY_SYNC_COMPUTE_PSO`:
/// default on; `"0"` disables).
#[cfg(windows)]
pub(crate) fn sync_compute_enabled() -> bool {
    static ON: OnceLock<bool> = OnceLock::new();
    *ON.get_or_init(|| {
        let on = value_is_enabled(std::env::var_os("VICTORY_SYNC_COMPUTE_PSO").as_deref());
        if on {
            tracing::info!(
                secs = COMPUTE_BUDGET.as_secs(),
                "compute dispatches wait for cold pipelines within a bounded budget"
            );
        }
        on
    })
}

/// Worker queue lanes, drained in ascending priority order.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum Lane {
    /// Compute pipelines a live dispatch is waiting for.
    Compute = 0,
    /// Graphics pipelines (or non-prioritized compute) a live draw waits for.
    Graphics = 1,
    /// Early pre-warm pipelines started at creation from a prior run's list.
    Early = 2,
}

/// Selects the worker lane for a pipeline: early pre-warms go to
/// [`Lane::Early`]; requested pipelines go to [`Lane::Compute`] when compute
/// synchronization is enabled for them, else [`Lane::Graphics`].
pub(crate) fn lane_for(now: bool, sync_compute: bool) -> Lane {
    if !now {
        Lane::Early
    } else if sync_compute {
        Lane::Compute
    } else {
        Lane::Graphics
    }
}

/// Three-lane FIFO queue draining [`Lane::Compute`], then [`Lane::Graphics`],
/// then [`Lane::Early`].
pub(crate) struct PriorityQueue<T> {
    lanes: [VecDeque<T>; 3],
}

impl<T> PriorityQueue<T> {
    pub(crate) const fn new() -> Self {
        Self {
            lanes: [VecDeque::new(), VecDeque::new(), VecDeque::new()],
        }
    }

    pub(crate) fn push(&mut self, lane: Lane, item: T) {
        self.lanes[lane as usize].push_back(item);
    }

    pub(crate) fn pop(&mut self) -> Option<T> {
        self.lanes.iter_mut().find_map(VecDeque::pop_front)
    }
}

/// Remaining compute wait allowance: the smaller of what remains in the
/// frame's wall-clock window (`since_frame_first`) and what remains in this
/// pipeline's own compilation window (`making_for`).
pub(crate) fn compute_left(
    since_frame_first: Duration,
    making_for: Duration,
    budget: Duration,
) -> Duration {
    budget
        .saturating_sub(since_frame_first)
        .min(budget.saturating_sub(making_for))
}

/// Result of waiting for a pending pipeline on a recording thread.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) struct WaitOutcome {
    pub(crate) waited: Duration,
    pub(crate) ready: bool,
    pub(crate) used_compute_budget: bool,
    pub(crate) compute_timed_out: bool,
    pub(crate) pipeline_timed_out: bool,
}

/// Tracks per-frame graphics and compute wait budgets independently.
pub(crate) struct WaitBudgets {
    graphics_budget: Duration,
    compute_budget: Duration,
    /// `(frame, cumulative_spent)` for the shared graphics wait budget.
    graphics: Mutex<(u64, Duration)>,
    /// `(frame, first_wait_instant)` for the wall-clock compute wait window.
    compute_first: Mutex<(u64, Option<Instant>)>,
}

impl WaitBudgets {
    #[cfg(windows)]
    pub(crate) const fn new() -> Self {
        Self::with_budgets(GRAPHICS_BUDGET, COMPUTE_BUDGET)
    }

    pub(crate) const fn with_budgets(graphics_budget: Duration, compute_budget: Duration) -> Self {
        Self {
            graphics_budget,
            compute_budget,
            graphics: Mutex::new((u64::MAX, Duration::ZERO)),
            compute_first: Mutex::new((u64::MAX, None)),
        }
    }

    /// Waits on `done` until `is_ready()` becomes true or the applicable
    /// budget expires.
    ///
    /// When `sync_compute` is true and `making_elapsed()` is below
    /// `compute_budget`, waits against the dedicated per-frame wall-clock
    /// compute window without touching the graphics budget. Otherwise waits
    /// against the per-frame graphics budget.
    pub(crate) fn wait_for(
        &self,
        frame: u64,
        sync_compute: bool,
        making_elapsed: impl Fn() -> Duration,
        is_ready: impl Fn() -> bool,
        done: &(Mutex<()>, Condvar),
    ) -> WaitOutcome {
        let making_before = making_elapsed();
        let use_compute = sync_compute && making_before < self.compute_budget;
        let left = if use_compute {
            let mut first = self.compute_first.lock().unwrap_or_else(|e| e.into_inner());
            if first.0 == u64::MAX || frame > first.0 {
                *first = (frame, Some(Instant::now()));
            }
            let since = first.1.map_or(Duration::ZERO, |at| at.elapsed());
            compute_left(since, making_before, self.compute_budget)
        } else {
            let mut spent = self.graphics.lock().unwrap_or_else(|e| e.into_inner());
            if spent.0 != frame {
                *spent = (frame, Duration::ZERO);
            }
            self.graphics_budget.saturating_sub(spent.1)
        };

        if left.is_zero() {
            let ready = is_ready();
            return WaitOutcome {
                waited: Duration::ZERO,
                ready,
                used_compute_budget: use_compute,
                compute_timed_out: use_compute && !ready,
                pipeline_timed_out: use_compute && !ready && making_elapsed() >= self.compute_budget,
            };
        }

        let started = Instant::now();
        let (lock, cond) = done;
        let mut held = lock.lock().unwrap_or_else(|e| e.into_inner());
        while !is_ready() {
            let rest = left.saturating_sub(started.elapsed());
            if rest.is_zero() {
                break;
            }
            held = cond
                .wait_timeout(held, rest)
                .unwrap_or_else(|e| e.into_inner())
                .0;
        }
        drop(held);

        let waited = started.elapsed();
        let ready = is_ready();
        if !use_compute {
            let mut spent = self.graphics.lock().unwrap_or_else(|e| e.into_inner());
            if spent.0 == frame {
                spent.1 = spent.1.saturating_add(waited);
            }
        }
        WaitOutcome {
            waited,
            ready,
            used_compute_budget: use_compute,
            compute_timed_out: use_compute && !ready,
            pipeline_timed_out: use_compute && !ready && making_elapsed() >= self.compute_budget,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::Arc;
    use std::sync::atomic::{AtomicBool, Ordering};

    #[test]
    fn switch_defaults_on_and_disables_on_zero() {
        assert!(value_is_enabled(None));
        assert!(value_is_enabled(Some(OsStr::new("1"))));
        assert!(value_is_enabled(Some(OsStr::new("true"))));
        assert!(!value_is_enabled(Some(OsStr::new("0"))));
    }

    #[test]
    fn requested_compute_jumps_ahead_of_graphics_and_early() {
        let mut q = PriorityQueue::new();
        q.push(lane_for(false, false), "early-gfx");
        q.push(lane_for(false, true), "early-compute");
        q.push(lane_for(true, false), "req-gfx");
        q.push(lane_for(true, true), "req-compute");

        assert_eq!(q.pop(), Some("req-compute"));
        assert_eq!(q.pop(), Some("req-gfx"));
        assert_eq!(q.pop(), Some("early-gfx"));
        assert_eq!(q.pop(), Some("early-compute"));
        assert_eq!(q.pop(), None);
    }

    #[test]
    fn compute_left_respects_frame_and_pipeline_bounds() {
        let s = Duration::from_secs;
        assert_eq!(compute_left(s(0), s(0), COMPUTE_BUDGET), COMPUTE_BUDGET);
        assert_eq!(
            compute_left(COMPUTE_BUDGET, s(0), COMPUTE_BUDGET),
            Duration::ZERO
        );
        assert_eq!(
            compute_left(s(0), COMPUTE_BUDGET, COMPUTE_BUDGET),
            Duration::ZERO
        );
        assert_eq!(
            compute_left(s(3), s(8), COMPUTE_BUDGET),
            COMPUTE_BUDGET - s(8)
        );
        assert_eq!(
            compute_left(s(8), s(3), COMPUTE_BUDGET),
            COMPUTE_BUDGET - s(8)
        );
    }

    /// A cold compute dispatch waits for its pipeline even after the frame's
    /// shorter graphics budget is exhausted, while graphics pipelines remain
    /// asynchronous and drop immediately.
    #[test]
    fn cold_compute_waits_after_graphics_budget_is_exhausted() {
        let budgets = WaitBudgets::with_budgets(Duration::from_millis(30), Duration::from_millis(600));
        let done = Arc::new((Mutex::new(()), Condvar::new()));

        // Exhaust frame 1's graphics budget with a slow graphics pipeline.
        let gfx1 = budgets.wait_for(1, false, || Duration::ZERO, || false, &done);
        assert!(!gfx1.ready);
        assert!(!gfx1.used_compute_budget);

        // A second cold graphics pipeline on the same frame drops immediately.
        let gfx2 = budgets.wait_for(1, false, || Duration::ZERO, || false, &done);
        assert!(!gfx2.ready);
        assert_eq!(gfx2.waited, Duration::ZERO);

        // A cold compute pipeline whose worker finishes in 70 ms (> 30 ms
        // graphics budget) completes instead of dropping.
        let ready = Arc::new(AtomicBool::new(false));
        let ready_worker = ready.clone();
        let done_worker = done.clone();
        let worker = std::thread::spawn(move || {
            std::thread::sleep(Duration::from_millis(70));
            ready_worker.store(true, Ordering::Release);
            let _held = done_worker.0.lock().unwrap();
            done_worker.1.notify_all();
        });

        let cs = budgets.wait_for(
            1,
            true,
            || Duration::ZERO,
            || ready.load(Ordering::Acquire),
            &done,
        );
        worker.join().unwrap();
        assert!(cs.ready, "cold compute dispatch must wait for its pipeline");
        assert!(cs.used_compute_budget);
        assert!(!cs.compute_timed_out);
        assert!(!cs.pipeline_timed_out);
    }

    /// With compute wait disabled (`sync_compute = false`), a compute dispatch
    /// shares the graphics budget and drops once it is spent.
    #[test]
    fn disabled_compute_wait_uses_graphics_budget() {
        let budgets = WaitBudgets::with_budgets(Duration::from_millis(20), Duration::from_millis(500));
        let done = (Mutex::new(()), Condvar::new());

        let first = budgets.wait_for(1, false, || Duration::ZERO, || false, &done);
        assert!(!first.ready);
        let second = budgets.wait_for(1, false, || Duration::ZERO, || false, &done);
        assert!(!second.ready);
        assert!(!second.used_compute_budget);
        assert_eq!(second.waited, Duration::ZERO);
    }

    /// A compute pipeline that exceeds its per-pipeline compilation cap stops
    /// getting the extended compute wait on subsequent frames, and concurrent
    /// compute waits share one wall-clock frame window.
    #[test]
    fn compute_waits_stop_after_pipeline_and_frame_caps() {
        let budgets = WaitBudgets::with_budgets(Duration::from_millis(20), Duration::from_millis(120));
        let done = (Mutex::new(()), Condvar::new());

        // Pipeline has already been compiling for 80 ms of a 120 ms cap.
        let started_making = Instant::now() - Duration::from_millis(80);
        let first = budgets.wait_for(1, true, || started_making.elapsed(), || false, &done);
        assert!(!first.ready);
        assert!(first.used_compute_budget);
        assert!(first.compute_timed_out);
        assert!(first.pipeline_timed_out);
        assert!(first.waited < Duration::from_millis(100));

        // On the next frame, the same stalled pipeline falls back to the short
        // graphics budget rather than burning another compute window.
        let next_frame = budgets.wait_for(2, true, || started_making.elapsed(), || false, &done);
        assert!(!next_frame.ready);
        assert!(!next_frame.used_compute_budget);
        assert!(next_frame.waited < Duration::from_millis(80));

        // Frame 3: once the wall-clock compute window expires, another compute
        // dispatch on the same frame drops immediately without waiting again.
        let frame3_a = budgets.wait_for(3, true, || Duration::ZERO, || false, &done);
        assert!(frame3_a.compute_timed_out);
        let frame3_b = budgets.wait_for(3, true, || Duration::ZERO, || false, &done);
        assert!(frame3_b.compute_timed_out);
        assert_eq!(frame3_b.waited, Duration::ZERO);
    }
}
