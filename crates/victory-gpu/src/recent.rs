//! Scratch memory kept by what recent work needed: the texture staging
//! chunks and big staging buffers ([`Peak`]), the snapshot buffers and the
//! big textures' read buffers ([`Pool`]) give back what the last [`WINDOW`]
//! did not use, and keep all it did (a pool kept at its peak for good held
//! 512 MB after one heavy submission).

use std::collections::VecDeque;
use std::time::{Duration, Instant};

/// How long what was used is kept after its last use: ~300 frames at 30
/// fps (a few frames of today's 2-3 s world frames). Steady play, where
/// each frame needs what the last did, never sees a release, nor does a
/// load that recurs within it, and a menu or a loading screen gives its
/// memory back that long after the load. Remaking is the price of a load that comes back
/// after longer (upload buffers cost ~5 ms per 12.7 MB, fresh snapshots a
/// page fault per 4 KiB).
pub(crate) const WINDOW: Duration = Duration::from_secs(10);

/// The most any submission needed of a pool in the last [`WINDOW`].
pub(crate) struct Peak {
    /// (when a submission began, what it needed), oldest first, the needs
    /// falling: one no larger than a later need is dropped, for the later
    /// one outlasts it in the window.
    seen: VecDeque<(Instant, usize)>,
    /// When the submission being prepared began.
    began: Option<Instant>,
}

impl Peak {
    pub(crate) const fn new() -> Self {
        Peak {
            seen: VecDeque::new(),
            began: None,
        }
    }

    /// A submission begins at `now`; the last one, which began when this
    /// was last called, needed `needed`. Returns the most any submission
    /// needed in the last [`WINDOW`], for the pool to keep (the last
    /// submission's need counts from when it began, not from now: a pool
    /// that sat idle must not take a load long gone for a recent one).
    pub(crate) fn begin(&mut self, now: Instant, needed: usize) -> usize {
        if let Some(began) = self.began.replace(now) {
            while self.seen.back().is_some_and(|&(_, n)| n <= needed) {
                self.seen.pop_back();
            }
            self.seen.push_back((began, needed));
        }
        while self
            .seen
            .front()
            .is_some_and(|&(at, _)| now.saturating_duration_since(at) > WINDOW)
        {
            self.seen.pop_front();
        }
        self.seen.front().map_or(0, |&(_, n)| n)
    }
}

/// Snapshot buffers kept for reuse (already touched, so no page faults),
/// each with when it was returned.
pub(crate) struct Pool {
    /// Smallest first, and of equal sizes the last returned first: a buffer
    /// that serves one submission serves the next, and the others of its
    /// size, which would rotate through and never age, go unused.
    free: Vec<(Vec<u8>, Instant)>,
}

impl Pool {
    pub(crate) const fn new() -> Self {
        Pool { free: Vec::new() }
    }

    /// The smallest free buffer with room for `n` bytes, if any.
    pub(crate) fn take(&mut self, n: usize) -> Option<Vec<u8>> {
        let i = self.free.partition_point(|(v, _)| v.capacity() < n);
        (i < self.free.len()).then(|| self.free.remove(i).0)
    }

    /// Takes `v` back, emptied, at `now`. The pool keeps at most `cap`
    /// bytes, the largest buffers.
    pub(crate) fn give(&mut self, mut v: Vec<u8>, now: Instant, cap: usize) {
        v.clear();
        self.keep(v, now, cap);
    }

    /// As [`Pool::give`], but `v` keeps its length: its bytes stay
    /// initialised, which [`Pool::filled`] counts on.
    pub(crate) fn keep(&mut self, v: Vec<u8>, now: Instant, cap: usize) {
        if v.capacity() <= cap {
            let i = self
                .free
                .partition_point(|(f, _)| f.capacity() < v.capacity());
            self.free.insert(i, (v, now));
        }
        let mut total = 0;
        if let Some(i) = self.free.iter().rposition(|(f, _)| {
            total += f.capacity();
            total > cap
        }) {
            self.free.drain(..=i);
        }
    }

    /// `taken` (from [`Pool::take`]) or a new buffer, with `n` bytes or more,
    /// all initialised: a new one is zeroed (exactly `n`, no doubling), a
    /// [`Pool::keep`]t one gives back the bytes it held and only what it
    /// lacks of `n` is written, so reuse costs no fill (a fresh 128 MiB is a
    /// page fault per 4 KiB and the fill). Called outside the pool's lock.
    pub(crate) fn filled(taken: Option<Vec<u8>>, n: usize) -> Vec<u8> {
        let mut v = taken.unwrap_or_else(|| Vec::with_capacity(n));
        if v.len() < n {
            v.resize(n, 0);
        }
        v
    }

    /// Releases the largest buffer that was not returned in the last
    /// [`WINDOW`], if any, to be dropped by the caller outside the pool's
    /// lock: one a call (once a frame), so that freeing a peak's buffers
    /// does not fall on one frame.
    pub(crate) fn trim(&mut self, now: Instant) -> Option<Vec<u8>> {
        let i = self
            .free
            .iter()
            .rposition(|&(_, at)| now.saturating_duration_since(at) > WINDOW)?;
        Some(self.free.remove(i).0)
    }

    /// Bytes held (tests).
    #[cfg(test)]
    pub(crate) fn bytes(&self) -> usize {
        self.free.iter().map(|(v, _)| v.capacity()).sum()
    }
}

/// How many of the oldest idle entries give up what they hold at `now`: the
/// leading ones idle longer than [`WINDOW`], then, while the entries left
/// hold more than `cap` bytes, the oldest of those. `idle` is (when an entry
/// went idle, the bytes it holds), oldest first, so the ones past the window
/// come first.
pub(crate) fn idle_shed(
    idle: impl Iterator<Item = (Instant, u64)> + Clone,
    now: Instant,
    cap: u64,
) -> usize {
    let mut kept: u64 = idle.clone().map(|(_, bytes)| bytes).sum();
    let mut shed = 0;
    for (at, bytes) in idle {
        if kept <= cap && now.saturating_duration_since(at) <= WINDOW {
            break;
        }
        kept -= bytes;
        shed += 1;
    }
    shed
}

#[cfg(test)]
mod tests {
    use super::*;

    const FRAME: Duration = Duration::from_millis(33);
    const MIB: usize = 1 << 20;

    /// Chunks as `crate::texture::recycle` keeps them, counted: what each
    /// submission needs is made if the pool does not hold it (a model of
    /// that function's arithmetic; the policy is `Peak`'s).
    struct Chunks {
        held: usize,
        last: usize,
        peak: Peak,
        made: usize,
    }

    impl Chunks {
        fn submit(&mut self, now: Instant, need: usize) {
            const KEPT: usize = 16;
            self.held = self.held.min(self.peak.begin(now, self.last).min(KEPT));
            self.made += need.saturating_sub(self.held);
            (self.held, self.last) = (self.held.max(need), need);
        }
    }

    #[test]
    fn chunks_follow_the_load_and_steady_state_makes_none() {
        let mut c = Chunks {
            held: 0,
            last: 0,
            peak: Peak::new(),
            made: 0,
        };
        let mut now = Instant::now();
        let mut run = |c: &mut Chunks, n, need| {
            for _ in 0..n {
                now += FRAME;
                c.submit(now, need);
            }
        };
        run(&mut c, 300, 3);
        assert_eq!((c.held, c.made), (3, 3));
        run(&mut c, 30, 16);
        assert_eq!((c.held, c.made), (16, 16));
        // The spike's chunks stay while it is in the window.
        let window = (WINDOW.as_millis() / FRAME.as_millis()) as usize;
        run(&mut c, window - 5, 3);
        assert_eq!(c.held, 16);
        run(&mut c, 10, 3);
        assert_eq!(c.held, 3);
        let made = c.made;
        run(&mut c, 300, 3);
        assert_eq!((c.held, c.made), (3, made), "steady state makes no chunk");
    }

    #[test]
    fn a_need_counts_from_when_its_submission_began() {
        let mut p = Peak::new();
        let t = Instant::now();
        assert_eq!(p.begin(t, 0), 0);
        assert_eq!(p.begin(t + FRAME, 16), 16);
        // A pool idle for longer than the window does not take the load
        // before it for a recent one.
        assert_eq!(p.begin(t + 2 * WINDOW, 3), 0);
        assert_eq!(p.begin(t + 2 * WINDOW + FRAME, 3), 3);
    }

    #[test]
    fn the_peak_falls_in_steps_as_larger_needs_leave_the_window() {
        let mut p = Peak::new();
        let t = Instant::now();
        let at = |s: u64| t + Duration::from_secs(s);
        p.begin(at(0), 0);
        for (s, need) in [(3, 8), (6, 5), (9, 2)] {
            p.begin(at(s), need);
        }
        // The needs began at 0 s (8), 3 s (5) and 6 s (2).
        assert_eq!(p.begin(at(10), 0), 8);
        assert_eq!(p.begin(at(11), 0), 5);
        assert_eq!(p.begin(at(14), 0), 2);
        assert_eq!(p.begin(at(17), 0), 0);
    }

    #[test]
    fn the_pool_takes_the_smallest_that_fits_and_the_last_returned_of_a_size() {
        let mut pool = Pool::new();
        let t = Instant::now();
        let buffer = |n: usize| {
            let mut v = Vec::with_capacity(n);
            v.push(1);
            v
        };
        let (a, b) = (buffer(100), buffer(100));
        let (pa, pb) = (a.as_ptr(), b.as_ptr());
        pool.give(buffer(1000), t, usize::MAX);
        pool.give(a, t, usize::MAX);
        pool.give(b, t, usize::MAX);
        assert!(pool.take(2000).is_none());
        let v = pool.take(50).unwrap();
        assert!(v.is_empty() && v.capacity() >= 100);
        assert_eq!(v.as_ptr(), pb, "the last returned");
        pool.give(v, t, usize::MAX);
        let (x, y) = (pool.take(50).unwrap(), pool.take(50).unwrap());
        assert_eq!((x.as_ptr(), y.as_ptr()), (pb, pa));
        assert!(pool.take(500).unwrap().capacity() >= 1000);
    }

    #[test]
    fn oversized_return_preserves_buffers_that_fit_the_current_cap() {
        let mut pool = Pool::new();
        let now = Instant::now();
        pool.give(Vec::with_capacity(32), now, 128);
        pool.give(Vec::with_capacity(64), now, 128);
        pool.give(Vec::with_capacity(256), now, 64);
        assert_eq!(pool.bytes(), 64);
        assert!(pool.take(64).is_some());
    }

    #[test]
    fn the_pool_keeps_the_largest_within_its_cap() {
        let mut pool = Pool::new();
        let t = Instant::now();
        for n in [10, 40, 20, 30] {
            pool.give(Vec::with_capacity(n), t, 70);
        }
        assert!(pool.bytes() <= 70);
        assert!(pool.take(40).is_some() && pool.take(30).is_some() && pool.take(10).is_none());
    }

    #[test]
    fn the_pool_releases_the_largest_unused_first_and_one_at_a_time() {
        let mut pool = Pool::new();
        let t = Instant::now();
        for n in [4, 64, 16, 32] {
            pool.give(Vec::with_capacity(n * MIB), t, usize::MAX);
        }
        // The 4 and 16 are used each frame; the 32 and 64 are not.
        let mut now = t;
        let mut freed = Vec::new();
        for _ in 0..400 {
            now += FRAME;
            let (a, b) = (pool.take(4 * MIB).unwrap(), pool.take(16 * MIB).unwrap());
            pool.give(a, now, usize::MAX);
            pool.give(b, now, usize::MAX);
            freed.extend(pool.trim(now).map(|v| v.capacity() / MIB));
        }
        assert_eq!(freed, [64, 32]);
        assert_eq!(pool.bytes(), 20 * MIB);
    }

    /// Snapshot buffers as `crate::shadow` takes and returns them: each
    /// submission takes `sizes` (made if the pool has none), then returns
    /// them all, and the frame trims.
    struct Snapshots {
        pool: Pool,
        made: usize,
    }

    impl Snapshots {
        fn frame(&mut self, now: Instant, sizes: &[usize]) {
            let taken: Vec<Vec<u8>> = sizes
                .iter()
                .map(|&n| {
                    self.pool.take(n).unwrap_or_else(|| {
                        self.made += 1;
                        Vec::with_capacity(n)
                    })
                })
                .collect();
            for v in taken {
                self.pool.give(v, now, 512 * MIB);
            }
            drop(self.pool.trim(now));
        }
    }

    #[test]
    fn snapshots_follow_the_load_and_steady_state_makes_none() {
        let steady: Vec<usize> = [64, 48, 32, 16, 8, 4, 2, 2]
            .iter()
            .map(|n| n * MIB)
            .collect();
        let spike: Vec<usize> = steady
            .iter()
            .copied()
            .chain([128 * MIB, 96 * MIB, 64 * MIB])
            .collect();
        let mut s = Snapshots {
            pool: Pool::new(),
            made: 0,
        };
        let mut now = Instant::now();
        let mut run = |s: &mut Snapshots, n, sizes: &[usize]| {
            for _ in 0..n {
                now += FRAME;
                s.frame(now, sizes);
            }
        };
        run(&mut s, 300, &steady);
        let (steady_bytes, made) = (s.pool.bytes(), s.made);
        assert_eq!(made, steady.len());
        run(&mut s, 30, &spike);
        assert_eq!(
            (s.pool.bytes(), s.made),
            (steady_bytes + 288 * MIB, made + 3)
        );
        run(&mut s, 400, &steady);
        assert_eq!(
            s.pool.bytes(),
            steady_bytes,
            "the spike's buffers went back"
        );
        assert_eq!(s.made, made + 3, "and steady state made none");
    }

    #[test]
    fn a_kept_buffer_is_filled_again_only_where_it_grows() {
        let mut pool = Pool::new();
        let t = Instant::now();
        let mut v = Pool::filled(pool.take(1000), 1000);
        assert!(v.len() == 1000 && v.iter().all(|&b| b == 0));
        v[999] = 7;
        let p = v.as_ptr();
        pool.keep(v, t, usize::MAX);
        let v = Pool::filled(pool.take(500), 500);
        assert_eq!(
            (v.as_ptr(), v.len(), v[999]),
            (p, 1000, 7),
            "not written again"
        );
        pool.keep(v, t, usize::MAX);
        let v = Pool::filled(pool.take(1000), 1000);
        assert_eq!((v.as_ptr(), v[999]), (p, 7));
        // Past its length it is zeroed, within its capacity or not.
        pool.keep(v, t, usize::MAX);
        let big = Pool::filled(pool.take(1 << 20), 1 << 20);
        assert!(big.len() == 1 << 20 && big.iter().filter(|&&b| b != 0).count() <= 1);
        // `give` empties, as snapshots need.
        pool.give(big, t, usize::MAX);
        assert!(pool.take(1 << 20).unwrap().is_empty());
    }

    /// Read buffers of big textures as `crate::texture::Scratch` takes and
    /// returns them: each submission refreshes textures of `sizes` (a buffer
    /// made if the pool has none that fits), then the next trims.
    struct Scratches {
        pool: Pool,
        made: usize,
    }

    impl Scratches {
        fn submit(&mut self, now: Instant, sizes: &[usize]) {
            drop(self.pool.trim(now));
            let taken: Vec<Vec<u8>> = sizes
                .iter()
                .map(|&n| {
                    let fits = self.pool.free.iter().any(|(v, _)| v.capacity() >= n);
                    self.made += usize::from(!fits);
                    Pool::filled(self.pool.take(n), n)
                })
                .collect();
            for v in taken {
                self.pool.keep(v, now, 64 * MIB);
            }
        }
    }

    #[test]
    fn big_textures_keep_their_buffers_while_they_change_and_give_them_back() {
        let mut s = Scratches {
            pool: Pool::new(),
            made: 0,
        };
        let mut now = Instant::now();
        let mut run = |s: &mut Scratches, n, sizes: &[usize]| {
            for _ in 0..n {
                now += FRAME;
                s.submit(now, sizes);
            }
        };
        // A big texture changes every submission: one buffer, made once
        // (one a submission before).
        run(&mut s, 300, &[20 * MIB]);
        assert_eq!((s.pool.bytes(), s.made), (20 * MIB, 1));
        // A spike: a second one at once.
        run(&mut s, 30, &[20 * MIB, 24 * MIB]);
        assert_eq!((s.pool.bytes(), s.made), (44 * MIB, 2));
        // The spike's buffer goes when no refresh has used it for a window,
        // and steady state makes none.
        run(&mut s, 400, &[20 * MIB]);
        assert_eq!((s.pool.bytes(), s.made), (20 * MIB, 2));
        // One over the cap is made each time and not kept.
        run(&mut s, 3, &[65 * MIB]);
        assert_eq!((s.pool.bytes(), s.made), (20 * MIB, 5));
    }

    /// Free staging buffers as `crate::texture::recycle` keeps them, counted:
    /// each submission needs `need` bytes in buffers, made if the free ones
    /// do not cover them (a model of that function's arithmetic, buffers
    /// taken by size and the oldest dropped; the policy is `Peak`'s).
    struct Uploads {
        /// Free buffers' sizes, oldest first.
        free: VecDeque<usize>,
        peak: Peak,
        last: Vec<usize>,
        made: usize,
    }

    impl Uploads {
        const KEPT: usize = 64 * MIB;
        const KEPT_MAX: usize = 256 * MIB;

        fn submit(&mut self, now: Instant, need: &[usize]) {
            // `recycle`: the last submission's buffers are free again.
            let needed = self.last.iter().sum();
            let peak = self.peak.begin(now, needed);
            let kept = peak.clamp(Self::KEPT, Self::KEPT_MAX);
            self.free.extend(self.last.drain(..));
            let mut bytes: usize = self.free.iter().sum();
            while bytes > kept {
                bytes -= self.free.pop_front().unwrap();
            }
            // `staging_buffer` for each.
            for &n in need {
                match self.free.iter().position(|&f| f == n) {
                    Some(i) => drop(self.free.remove(i)),
                    None => self.made += 1,
                }
                self.last.push(n);
            }
        }
    }

    #[test]
    fn staging_buffers_follow_the_load_and_steady_state_makes_none() {
        let mut u = Uploads {
            free: VecDeque::new(),
            peak: Peak::new(),
            last: Vec::new(),
            made: 0,
        };
        let mut now = Instant::now();
        let mut run = |u: &mut Uploads, n, need: &[usize]| {
            for _ in 0..n {
                now += FRAME;
                u.submit(now, need);
            }
        };
        // A 128 MiB texture changing every submission: with the 64 MiB cap
        // it was made again each one (its class is over the cap).
        run(&mut u, 300, &[128 * MIB]);
        assert_eq!(u.made, 1);
        assert_eq!(u.free.iter().sum::<usize>(), 0, "in use, none free");
        // A spike of two more: kept while it is in the window, and then not.
        run(&mut u, 30, &[128 * MIB, 64 * MIB, 32 * MIB]);
        assert_eq!(u.made, 3);
        let window = (WINDOW.as_millis() / FRAME.as_millis()) as usize;
        run(&mut u, window - 5, &[128 * MIB]);
        assert!(u.free.iter().sum::<usize>() >= 96 * MIB);
        run(&mut u, 10, &[128 * MIB]);
        assert!(u.free.is_empty(), "the spike's buffers went");
        let made = u.made;
        run(&mut u, 300, &[128 * MIB]);
        assert_eq!(u.made, made, "steady state makes no buffer");
        // A texture over the upper bound is made each time.
        run(&mut u, 3, &[512 * MIB]);
        assert_eq!(u.made, made + 3);
    }

    #[test]
    fn idle_shed_gives_up_what_is_past_the_window_then_the_oldest_over_the_cap() {
        let t = Instant::now();
        let at = |s: u64| t + Duration::from_secs(s);
        let now = at(20);
        let shed = |idle: &[(u64, u64)], cap: u64| {
            idle_shed(idle.iter().map(|&(s, b)| (at(s), b)), now, cap)
        };
        assert_eq!(shed(&[], 0), 0);
        // Exactly one window old is still kept.
        assert_eq!(shed(&[(10, 5), (15, 5)], 10), 0);
        // Past the window (11 s old) goes; the rest fits the cap.
        assert_eq!(shed(&[(9, 5), (15, 5)], 10), 1);
        assert_eq!(shed(&[(0, 1), (1, 2)], u64::MAX), 2);
        // Over the cap, the oldest go until what is left fits.
        assert_eq!(shed(&[(12, 4), (14, 4), (16, 4)], 8), 1);
        assert_eq!(shed(&[(12, 4), (14, 4), (16, 4)], 3), 3);
    }

    /// `idle_shed` against its definition: the least number of oldest entries
    /// after which no entry is past the window and the bytes left fit the cap.
    #[test]
    fn idle_shed_is_the_least_prefix_that_leaves_no_old_entry_and_fits_the_cap() {
        let t = Instant::now();
        let mut seed = 0x9e37_79b9_7f4a_7c15u64;
        let mut next = move |n: u64| {
            seed = seed
                .wrapping_mul(6_364_136_223_846_793_005)
                .wrapping_add(1_442_695_040_888_963_407);
            (seed >> 33) % n
        };
        for _ in 0..200_000 {
            let len = next(12) as usize;
            let mut when = 0;
            let idle: Vec<(Instant, u64)> = (0..len)
                .map(|_| {
                    when += next(6);
                    (t + Duration::from_secs(when), next(8))
                })
                .collect();
            let now = t + Duration::from_secs(next(40));
            let cap = next(30);
            let aged = |at: Instant| now.saturating_duration_since(at) > WINDOW;
            let fits = |k: usize| {
                idle[k..].iter().all(|&(at, _)| !aged(at))
                    && idle[k..].iter().map(|&(_, b)| b).sum::<u64>() <= cap
            };
            let least = (0..=len).find(|&k| fits(k)).unwrap();
            assert_eq!(idle_shed(idle.iter().copied(), now, cap), least);
        }
    }
}
