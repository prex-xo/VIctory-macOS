//! Frame pacing: the title sets a frame interval and waits for each frame
//! to begin. Ours is a clock started by the first interval set, ticking
//! once per interval; the token is the frame number. Each event kind
//! (origin, vblank, flip) happens once per tick, so two waits for the same
//! kind never end on the same tick: Grand Theft Auto VI polls the origin
//! with "token on time-out" (starting its next frame at once) and is paced
//! by a blocking wait for the flip before each present.
//!
//! shortcut: event offsets are ignored (every event is at the tick); tie
//! the clock to presentation if a title paces by them.
#![allow(clippy::not_unsafe_ptr_arg_deref)]

use std::sync::Mutex;
use std::sync::atomic::{AtomicU64, Ordering};
use std::time::{Duration, Instant};

use crate::object::{HRESULT, S_FALSE, S_OK, This};

/// What a wait returns when its time is up first.
const TIMEOUT: HRESULT = S_FALSE;
/// The frame-origin event: the one that starts frames and makes tokens.
const ORIGIN: u32 = 0;
/// Make the frame's token even if the wait times out (the title starts the
/// frame early).
const TOKEN_ON_TIMEOUT: u32 = 2;
const SIXTIETH: Duration = Duration::from_micros(16_667);

struct Clock {
    start: Instant,
    interval: Duration,
    /// The tick number at `start` (ticks go on across interval changes).
    first: u64,
}

static CLOCK: Mutex<Option<Clock>> = Mutex::new(None);
static FRAME: AtomicU64 = AtomicU64::new(0);

fn clock() -> std::sync::MutexGuard<'static, Option<Clock>> {
    CLOCK.lock().unwrap_or_else(|e| e.into_inner())
}

#[cfg(any(feature = "diagnostics", test))]
fn test_interval(requested: Duration, hz: Option<&str>) -> Duration {
    hz.and_then(|hz| hz.parse::<u32>().ok())
        .and_then(|hz| Duration::from_secs(1).checked_div(hz))
        .filter(|interval| !interval.is_zero())
        .unwrap_or(requested)
}

/// Device method: a frame every `length` microseconds. The period is how
/// many intervals a frame may take to reach the display (pipeline depth),
/// not part of the rate.
pub(crate) extern "system" fn set_interval(
    _: This,
    _output: This,
    length: u32,
    _period: u32,
    _flags: u32,
) -> HRESULT {
    let requested = match length {
        0 => SIXTIETH,
        us => Duration::from_micros(u64::from(us)),
    };
    #[cfg(feature = "diagnostics")]
    let interval = test_interval(requested, std::env::var("VICTORY_FRAME_HZ").ok().as_deref());
    #[cfg(not(feature = "diagnostics"))]
    let interval = requested;
    if interval != requested {
        tracing::info!(?interval, ?requested, "frame interval override");
    } else {
        tracing::info!(?interval, "frame interval");
    }
    install_clock(interval, Instant::now());
    S_OK
}

/// Start a clock ticking once per `interval` from `start`, its first tick
/// after the last one handed out (ticks go on across interval changes).
/// Returns the clock's start and first tick; the tests pace by them.
fn install_clock(interval: Duration, start: Instant) -> (Instant, u64) {
    let first = LAST
        .iter()
        .map(|t| t.load(Ordering::Relaxed))
        .max()
        .unwrap_or(0);
    *clock() = Some(Clock {
        start,
        interval,
        first,
    });
    (start, first)
}

/// Device method: events are scheduled at every tick.
pub(crate) extern "system" fn schedule(
    _: This,
    _kind: u32,
    _offset: u32,
    _list: *const u8,
    _flags: u32,
) -> HRESULT {
    S_OK
}

/// Frames begun so far (frame waits that ended).
pub(crate) fn current() -> u64 {
    FRAME.load(Ordering::Relaxed)
}

/// Per event kind (origin, vblank, flip and later kinds): the last tick a
/// wait for it ended on.
static LAST: [AtomicU64; 3] = [const { AtomicU64::new(0) }; 3];

fn last(kind: u32) -> &'static AtomicU64 {
    &LAST[(kind as usize).min(LAST.len() - 1)]
}

/// The tick a wait for `kind` ends on, and how long until it: the first
/// one after the last a wait for `kind` took (an event happens once per
/// tick: a wait woken a little early must not let the next one end on the
/// same tick), the latest one passed if that is later. A wait begun after
/// its tick passed (a frame slower than the interval) so ends at once, as
/// an event signalled at the tick and left signalled until a wait takes it
/// would (assumed: nothing we have says what a late wait does); waiting for
/// the next tick instead cost the cutscene's frames 0-33 ms each (4 % of
/// the frame thread, run perf3). A wait that
/// comes within `limit` takes the tick here, under the clock's lock, so two
/// threads waiting for one kind never take the same tick. `now` is sampled
/// after taking the lock, so a paused caller does not act on a stale clock.
fn next_tick(kind: u32, limit: Duration, now: impl FnOnce() -> Instant) -> (u64, Duration) {
    let mut clock = clock();
    let now = now();
    let c = clock.get_or_insert_with(|| Clock {
        start: now,
        interval: SIXTIETH,
        first: 0,
    });
    let elapsed = now.saturating_duration_since(c.start).as_nanos();
    let interval = c.interval.as_nanos().max(1);
    let tick = (c.first + (elapsed / interval) as u64).max(last(kind).load(Ordering::Relaxed) + 1);
    let at = u128::from(tick - c.first) * interval;
    let wait = Duration::from_nanos(at.saturating_sub(elapsed) as u64);
    if wait <= limit {
        last(kind).fetch_max(tick, Ordering::Relaxed);
    }
    (tick, wait)
}

/// Device method: blocks until the next tick (or the time-out); an origin
/// wait then hands out the new frame's token. Other events keep the token
/// the title passes. Ticks are absolute (start plus whole intervals), so
/// lateness never accumulates.
pub(crate) extern "system" fn wait(
    _: This,
    kind: u32,
    timeout_ms: u32,
    _list: *const u8,
    flags: u32,
    token: *mut u64,
) -> HRESULT {
    let origin = kind == ORIGIN;
    let limit = Duration::from_millis(u64::from(timeout_ms));
    // The tick is only logged (diagnostics); taking it is what counts.
    #[cfg_attr(not(feature = "diagnostics"), allow(unused_variables))]
    let (tick, wait) = next_tick(kind, limit, Instant::now);
    let hr = if wait > limit {
        std::thread::sleep(limit);
        TIMEOUT
    } else {
        std::thread::sleep(wait);
        S_OK
    };
    #[cfg(feature = "diagnostics")]
    {
        WAITS[0].fetch_add(1, Ordering::Relaxed);
        if hr == S_OK && wait.is_zero() {
            WAITS[1].fetch_add(1, Ordering::Relaxed);
        }
    }
    // A frame begun on a time-out takes no tick: its flip still does.
    if origin && (hr == S_OK || flags & TOKEN_ON_TIMEOUT != 0) {
        // Tokens count frames (consecutive), not ticks.
        let frame = FRAME.fetch_add(1, Ordering::Relaxed) + 1;
        if !token.is_null() {
            // SAFETY: checked non-null; the title passes its token.
            unsafe { token.write(frame) };
        }
    }
    #[cfg(feature = "diagnostics")]
    tracing::debug!(
        kind,
        timeout_ms,
        flags,
        hr,
        tick,
        wait_us = wait.as_micros() as u64,
        frame = current(),
        "frame wait"
    );
    hr
}

/// Diagnostics: frame waits, and those that ended at once (their tick had
/// passed); presents whose flip tick was the previous present's.
#[cfg(feature = "diagnostics")]
static WAITS: [AtomicU64; 3] = [const { AtomicU64::new(0) }; 3];
#[cfg(feature = "diagnostics")]
static PRESENT_TICK: AtomicU64 = AtomicU64::new(0);

/// Diagnostics: a frame was presented (counts presents sharing a flip tick).
#[cfg(feature = "diagnostics")]
pub(crate) fn presented() {
    let flip = LAST[2].load(Ordering::Relaxed);
    if flip != 0 && PRESENT_TICK.swap(flip, Ordering::Relaxed) == flip {
        WAITS[2].fetch_add(1, Ordering::Relaxed);
    }
}

/// Diagnostics: logs [`WAITS`] since the last call.
#[cfg(feature = "diagnostics")]
pub(crate) fn log() {
    let [waits, at_once, shared] = [0, 1, 2].map(|i| WAITS[i].swap(0, Ordering::Relaxed));
    tracing::info!("frame waits: {waits}, {at_once} of them ended at once; presents sharing a flip tick: {shared}");
}

/// Device method: no statistics are kept, so none are returned.
pub(crate) extern "system" fn statistics(
    _: This,
    _token: u64,
    _kinds: u32,
    count: *mut u32,
    _out: *mut u8,
) -> HRESULT {
    if !count.is_null() {
        // SAFETY: checked non-null.
        unsafe { count.write(0) };
    }
    S_OK
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The tests share the global clock, so they must not run at once.
    static SERIAL: Mutex<()> = Mutex::new(());

    /// A null pointer for any of the device methods' pointer parameters.
    fn null<T>() -> *const T {
        std::ptr::null()
    }

    /// Test fixture for the shared clock: holds the `SERIAL` guard
    /// (recovering a poisoned lock) and saves the clock state, restoring it
    /// on drop — even when the test panics — so one failing test cannot
    /// break the next.
    struct Fixture {
        _serial: std::sync::MutexGuard<'static, ()>,
        clock: Option<Clock>,
        last: [u64; 3],
        frame: u64,
    }

    impl Fixture {
        fn new() -> Self {
            let serial = SERIAL.lock().unwrap_or_else(|e| e.into_inner());
            let clock = clock().take();
            let last = LAST.each_ref().map(|t| t.load(Ordering::Relaxed));
            let frame = FRAME.load(Ordering::Relaxed);
            Self {
                _serial: serial,
                clock,
                last,
                frame,
            }
        }

        /// A clock with exactly `interval`, installed straight into the
        /// clock state — the `VICTORY_FRAME_HZ` diagnostics override cannot
        /// change what the test paces by. Returns the clock's start and
        /// first tick.
        fn set_interval(&self, interval: Duration) -> (Instant, u64) {
            install_clock(interval, Instant::now())
        }
    }

    impl Drop for Fixture {
        fn drop(&mut self) {
            *clock() = self.clock.take();
            for (slot, saved) in LAST.iter().zip(self.last.iter()) {
                slot.store(*saved, Ordering::Relaxed);
            }
            FRAME.store(self.frame, Ordering::Relaxed);
        }
    }

    #[test]
    fn a_test_frame_rate_changes_the_interval_only_when_valid() {
        let requested = Duration::from_micros(33_333);
        assert_eq!(
            test_interval(requested, Some("60")),
            Duration::from_nanos(16_666_666)
        );
        for value in [None, Some("0"), Some("invalid"), Some("1000000001")] {
            assert_eq!(test_interval(requested, value), requested);
        }
    }

    #[test]
    fn waits_end_on_ticks_and_count_frames() {
        let fixture = Fixture::new();
        // The device method answers whatever the environment (with
        // `--features diagnostics`, VICTORY_FRAME_HZ may override its
        // interval); the fixture paces the test with the exact interval.
        assert_eq!(set_interval(null(), null(), 50_000, 1, 0), S_OK);
        // A wait's tick is never more than one interval away, so with 50 ms
        // ticks a 1 s time-out always ends on the tick, not on the time-out
        // (that would come back TIMEOUT), however late the thread is woken.
        let _ = fixture.set_interval(Duration::from_millis(50));
        let mut first = 0;
        assert_eq!(wait(null(), 0, 1_000, null(), 0, &mut first), S_OK);
        let mut second = 0;
        assert_eq!(wait(null(), 0, 1_000, null(), 0, &mut second), S_OK);
        assert!(second > first);

        // A zero-time-out wait times out at once while its next tick is still
        // away, and takes no tick. "Still away" must hold however late this
        // thread is woken (at background QoS a wake-up can land whole ticks
        // late), so use a tick of u32::MAX µs (~71 min): the current tick
        // then stands still for the whole test.
        const STILL: Duration = Duration::from_micros(u32::MAX as u64);
        let _ = fixture.set_interval(STILL);
        // Take the origin tick at the clock's first tick, so the next one is
        // a whole interval away.
        let _ = next_tick(0, Duration::MAX, Instant::now);
        let taken = last(0).load(Ordering::Relaxed);
        assert_eq!(wait(null(), 0, 0, null(), 0, &mut second), TIMEOUT);
        // A time-out takes no tick.
        assert_eq!(last(0).load(Ordering::Relaxed), taken);
        // Tokens count frames, a frame started early on a time-out too.
        let mut early = 0;
        assert_eq!(
            wait(null(), 0, 0, null(), TOKEN_ON_TIMEOUT, &mut early),
            TIMEOUT
        );
        assert_eq!(early, second + 1);
        let _ = fixture.set_interval(Duration::from_millis(50));
        let mut next = 0;
        assert_eq!(wait(null(), 0, 1_000, null(), 0, &mut next), S_OK);
        assert_eq!(next, early + 1);
        assert!(LAST[0].load(Ordering::Relaxed) >= 2);

        // The title's loop (the clock is global, so in the same test): an
        // origin poll that times out (and starts the frame), then a blocking
        // flip wait. Each flip wait ends on its own tick — never the one the
        // wait before it ended on — however many ticks a late wake-up skips.
        const FLIP: u32 = 2;
        let _ = fixture.set_interval(Duration::from_millis(50));
        let mut token = 0;
        let flips: Vec<u64> = (0..4)
            .map(|_| {
                let _ = wait(null(), 0, 0, null(), TOKEN_ON_TIMEOUT, &mut token);
                assert_eq!(wait(null(), FLIP, u32::MAX, null(), 0, &mut token), S_OK);
                LAST[2].load(Ordering::Relaxed)
            })
            .collect();
        assert!(flips.windows(2).all(|w| w[1] > w[0]), "{flips:?}");
        // Even when a flip wait begins on the tick another just ended on.
        let (tick, _) = next_tick(FLIP, Duration::ZERO, Instant::now);
        assert!(tick > flips[3]);
        // Exactly consecutive ticks, where no tick can pass between waits:
        // one fixed clock sample at the ~71-minute clock's start keeps the
        // current tick still, so each wait for one kind takes exactly the
        // tick after the last one it took.
        let (start, first_tick) = fixture.set_interval(STILL);
        let ticks: Vec<u64> = (0..4)
            .map(|_| next_tick(FLIP, Duration::MAX, || start).0)
            .collect();
        assert!(ticks.windows(2).all(|w| w[1] == w[0] + 1), "{ticks:?}");
        // A flip begun after its ticks passed (two missed) ends at once, on
        // the latest tick passed; the next one takes its successor. The
        // clock sample is fixed past ticks[3] + 2 intervals, so a scheduler
        // pause cannot move which ticks have passed — a regression waiting
        // for the next future tick instead fails the zero wait here.
        let late_now = start
            .checked_add(STILL * (ticks[3] - first_tick + 3) as u32 + STILL / 2)
            .expect("a few ~71 min intervals fit in an Instant");
        let (late, late_wait) = next_tick(FLIP, Duration::MAX, || late_now);
        assert_eq!(late_wait, Duration::ZERO);
        assert!(late >= ticks[3] + 2, "{late} after {}", ticks[3]);
        let (succ, succ_wait) = next_tick(FLIP, Duration::MAX, || late_now);
        assert_eq!(succ, late + 1);
        assert!(succ_wait > Duration::ZERO);
    }

    /// Two waits for one kind that begin together (two threads) take two
    /// ticks: the first takes its tick before it sleeps. A time-out takes
    /// none. The old code marked the tick after the sleep, so both got it.
    #[test]
    fn waits_begun_together_take_different_ticks() {
        let _fixture = Fixture::new();
        const VBLANK: u32 = 1;
        let long = Duration::from_secs(1);
        // One fixed clock sample for all three polls: the current tick
        // stands still however long the thread is paused, so the
        // zero-time-out poll cannot steal a tick.
        let now = Instant::now();
        let (first, _) = next_tick(VBLANK, long, || now);
        let (second, _) = next_tick(VBLANK, long, || now);
        assert!(second > first, "{second} after {first}");
        next_tick(VBLANK, Duration::ZERO, || now);
        assert_eq!(last(VBLANK).load(Ordering::Relaxed), second);
    }
}
