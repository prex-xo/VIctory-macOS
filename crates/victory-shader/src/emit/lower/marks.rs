//! Wave-coalesced GPU memory marks for view writes, view reads, and
//! title-memory dirty bits.
//!
//! Most lanes in a wave touch the same 64 KiB page or heap slot; checking the
//! wave's leader first collapses a uniform wave's bitmap traffic to one
//! representative lane while preserving every distinct bit and page-straddling
//! write.

use super::*;
use crate::emit::{VIEW_MARK_NONE, VIEW_READS, VIEW_WRITE_BITS};

#[cfg(test)]
thread_local! {
    /// Set by tests that compare wave-coalesced marks against per-lane marks on
    /// the GPU.
    static MARK_LEAD_OFF: std::cell::Cell<bool> = const { std::cell::Cell::new(false) };
}

/// Runs `f` with wave coalescing disabled in [`Lower::mark_lead`], so tests can
/// compare wave-coalesced and per-lane mark bitmaps bit for bit on the GPU.
#[cfg(test)]
pub(crate) fn per_lane_marks<R>(f: impl FnOnce() -> R) -> R {
    struct Reset(bool);
    impl Drop for Reset {
        fn drop(&mut self) {
            MARK_LEAD_OFF.with(|c| c.set(self.0));
        }
    }
    let prev = MARK_LEAD_OFF.with(|c| c.replace(true));
    let _reset = Reset(prev);
    f()
}

fn mark_lead_off() -> bool {
    #[cfg(test)]
    return MARK_LEAD_OFF.with(|c| c.get());
    #[cfg(not(test))]
    false
}

impl<'a> Lower<'a> {
    /// Returns a per-lane predicate that is `true` for the wave's chosen leader
    /// and any active lane whose mark key `n` differs from the leader's, so
    /// lanes targeting the same key coalesce to a single bitmap check/atomic.
    /// Returns `None` when coalescing is disabled for parity testing.
    ///
    /// In pixel shaders, avoids any divergent `!helper` branch before wave
    /// operations by ranking helper lanes above live lanes (`lane | (1 << 30)`)
    /// in an unsigned wave minimum, masking the read index with `(1 << 30) - 1`
    /// so all-helper waves never index out of bounds, and gating the result by
    /// `!helper`.
    fn mark_lead(&mut self, n: V) -> Option<V> {
        if mark_lead_off() {
            return None;
        }
        self.s.flags |= flags::WAVE_OPS;
        let c = self.cur;
        let i32t = self.s.t.i32;
        let i1 = self.s.t.i1;
        if self.s.pixel {
            let helper = self
                .s
                .op(c, "isHelperLane.i1", Attr::ReadOnly, i1, 221, &[]);
            let lane = self
                .s
                .op(c, "waveGetLaneIndex", Attr::ReadOnly, i32t, 111, &[]);
            let hi = self.s.m.i32(1 << 30);
            let helper_key = self.s.m.bin(c, Bin::Or, lane, hi);
            let key = self.s.m.select(c, helper, helper_key, lane);
            let (min_op, unsigned) = (self.s.m.i8(2), self.s.m.i8(1));
            let lead = self.s.op(
                c,
                "waveActiveOp.i32",
                Attr::NoUnwind,
                i32t,
                119,
                &[key, min_op, unsigned],
            );
            let mask = self.s.m.i32((1 << 30) - 1);
            let at = self.s.m.bin(c, Bin::And, lead, mask);
            let f = self
                .s
                .op(c, "waveReadLaneAt.i32", Attr::NoUnwind, i32t, 117, &[n, at]);
            let diff = self.s.m.cmp(c, pred::NE, n, f);
            let is_lead = self.s.m.cmp(c, pred::EQ, lane, lead);
            let keep = self.s.m.bin(c, Bin::Or, diff, is_lead);
            let yes = self.s.m.bool(true);
            let live = self.s.m.bin(c, Bin::Xor, helper, yes);
            return Some(self.s.m.bin(c, Bin::And, live, keep));
        }
        let f = self
            .s
            .op(c, "waveReadLaneFirst.i32", Attr::NoUnwind, i32t, 118, &[n]);
        let first = self
            .s
            .op(c, "waveIsFirstLane", Attr::NoUnwind, i1, 110, &[]);
        let diff = self.s.m.cmp(c, pred::NE, n, f);
        Some(self.s.m.bin(c, Bin::Or, diff, first))
    }

    /// Marks the pages a write of `bytes` bytes at byte `x` of the unordered
    /// buffer view `h` touches, in the lanes `guard` lets through (module
    /// notes, "Write marks"): the bits of its first and last byte, set only
    /// where still clear (both read in one word when they share it, as they
    /// nearly always do), coalesced across the wave by [`Self::mark_lead`].
    /// Nothing for handles that are no heap view, or without
    /// [`Parts::view_marks`].
    pub(super) fn mark_view_write(&mut self, h: V, x: V, bytes: V, guard: Option<V>) {
        if !self.x.view_marks {
            return;
        }
        let Some(&index) = self.s.view_index.get(&h) else {
            return;
        };
        let k = |l: &mut Self, v: u32| l.s.m.i32(v);
        let table = self.s.view_marks_handle();
        let c = self.cur;
        let eight = k(self, 8);
        let at = self.s.m.bin(c, Bin::Mul, index, eight);
        let (base, phase) = self.raw_pair(table, at);
        let none = k(self, VIEW_MARK_NONE);
        let valid = self.s.m.cmp(c, pred::NE, base, none);
        let active = match guard {
            Some(g) => self.s.m.bin(c, Bin::And, g, valid),
            None => valid,
        };
        // Guard active lanes before wave coalescing so an inactive lane
        // (`!valid` or `!guard`) can never claim leadership.
        let act = self.begin_guarded(Some(active));
        let c = self.cur;
        let first = self.s.m.bin(c, Bin::Add, phase, x);
        let (zero, one) = (k(self, 0), k(self, 1));
        let nonempty = self.s.m.cmp(c, pred::NE, bytes, zero);
        let span = self.s.m.bin(c, Bin::Sub, bytes, one);
        let span = self.s.m.select(c, nonempty, span, zero);
        let last = self.s.m.bin(c, Bin::Add, first, span);
        let (k16, cap) = (k(self, 16), k(self, VIEW_WRITE_BITS - 1));
        let (k5, k2, k31) = (k(self, 5), k(self, 2), k(self, 31));
        let mut pages = [zero; 2];
        for (slot, byte) in pages.iter_mut().zip([first, last]) {
            let page = self.s.m.bin(c, Bin::LShr, byte, k16);
            let n = self.s.m.bin(c, Bin::Add, base, page);
            let over = self.s.m.cmp(c, pred::IUGT, n, cap);
            *slot = self.s.m.select(c, over, cap, n);
        }
        let [n_first, n_last] = pages;
        let lead = self.mark_lead(n_first);
        let c = self.cur;
        let enter = match lead {
            Some(lead) => {
                // A lane whose write straddles a 64 KiB boundary must still
                // run even when its first page matches the wave leader's.
                let mut straddle = self.s.m.cmp(c, pred::NE, n_first, n_last);
                if self.s.pixel {
                    let helper =
                        self.s
                            .op(c, "isHelperLane.i1", Attr::ReadOnly, self.s.t.i1, 221, &[]);
                    let yes = self.s.m.bool(true);
                    let live = self.s.m.bin(c, Bin::Xor, helper, yes);
                    straddle = self.s.m.bin(c, Bin::And, straddle, live);
                }
                Some(self.s.m.bin(c, Bin::Or, lead, straddle))
            }
            None => None,
        };
        let led = self.begin_guarded(enter);
        let c = self.cur;
        let word = self.s.m.bin(c, Bin::LShr, n_first, k5);
        let word = self.s.m.bin(c, Bin::Shl, word, k2);
        let bit = self.s.m.bin(c, Bin::And, n_first, k31);
        let first_bit = self.s.m.bin(c, Bin::Shl, one, bit);
        let last_word = self.s.m.bin(c, Bin::LShr, n_last, k5);
        let last_word = self.s.m.bin(c, Bin::Shl, last_word, k2);
        let bit = self.s.m.bin(c, Bin::And, n_last, k31);
        let last_bit = self.s.m.bin(c, Bin::Shl, one, bit);
        // Most writes find their pages marked: a load, not an atomic.
        let bitmap = self.s.view_writes_handle();
        let old = self.raw_dword(bitmap, word);
        let together = self.s.m.cmp(c, pred::EQ, word, last_word);
        let also = self.s.m.select(c, together, last_bit, zero);
        let want = self.s.m.bin(c, Bin::Or, first_bit, also);
        let set = self.s.m.bin(c, Bin::And, old, want);
        let clear = self.s.m.cmp(c, pred::NE, set, want);
        let apart = self.s.m.cmp(c, pred::NE, word, last_word);
        let more = self.s.m.bin(c, Bin::Or, clear, apart);
        let outer = self.begin_guarded(Some(more));
        let iu = self.s.m.undef(self.s.t.i32);
        // Operation 2 is OR; setting a set bit again changes nothing.
        self.atomic_bin_op(self.cur, &[bitmap, k2, word, iu, iu, want]);
        // A write across two bitmap dwords: the last byte's is its own.
        let inner = self.begin_guarded(Some(apart));
        self.atomic_bin_op(self.cur, &[bitmap, k2, last_word, iu, iu, last_bit]);
        self.end_guarded(inner, &[]);
        self.end_guarded(outer, &[]);
        self.end_guarded(led, &[]);
        self.end_guarded(act, &[]);
    }

    /// Sets bit `n` in `bitmap` in the lanes `guard` lets through (`None`:
    /// all), coalesced across the wave by [`Self::mark_lead`] and issuing an
    /// atomic only where a load finds the bit clear. A lane that reads a stale
    /// 0 only ORs again.
    pub(super) fn mark_memory_bit_if_clear(&mut self, bitmap: V, n: V, guard: Option<V>) {
        let act = self.begin_guarded(guard);
        let lead = self.mark_lead(n);
        let led = self.begin_guarded(lead);
        let c = self.cur;
        let (k2, zero) = (self.s.m.i32(2), self.s.m.i32(0));
        let (word, bit) = self.memory_bit(n);
        // Read in every coalesced lane: out of the bitmap reads 0, an index the
        // atomic then drops.
        let old = self.raw_dword(bitmap, word);
        let set = self.s.m.bin(c, Bin::And, old, bit);
        let clear = self.s.m.cmp(c, pred::EQ, set, zero);
        let at = self.begin_guarded(Some(clear));
        let iu = self.s.m.undef(self.s.t.i32);
        self.atomic_bin_op(self.cur, &[bitmap, k2, word, iu, iu, bit]);
        self.end_guarded(at, &[]);
        self.end_guarded(led, &[]);
        self.end_guarded(act, &[]);
    }

    /// Sets the read mark of resource-heap index `index` (module notes,
    /// "Read marks") where it is still clear, coalesced across the wave by
    /// [`Self::mark_lead`]. Nothing without [`Parts::view_marks`].
    pub(super) fn mark_view_read(&mut self, index: V) {
        if !self.x.view_marks {
            return;
        }
        let k = |l: &mut Self, v: u32| l.s.m.i32(v);
        let bitmap = self.s.view_writes_handle();
        let c = self.cur;
        let (base, k5, k2, k31, one, zero) = (
            k(self, VIEW_READS),
            k(self, 5),
            k(self, 2),
            k(self, 31),
            k(self, 1),
            k(self, 0),
        );
        let n = self.s.m.bin(c, Bin::Add, index, base);
        let lead = self.mark_lead(n);
        let led = self.begin_guarded(lead);
        let c = self.cur;
        let word = self.s.m.bin(c, Bin::LShr, n, k5);
        let word = self.s.m.bin(c, Bin::Shl, word, k2);
        let bit = self.s.m.bin(c, Bin::And, n, k31);
        let bit = self.s.m.bin(c, Bin::Shl, one, bit);
        // Most reads find their slot marked: a load, not an atomic.
        let old = self.raw_dword(bitmap, word);
        let set = self.s.m.bin(c, Bin::And, old, bit);
        let clear = self.s.m.cmp(c, pred::EQ, set, zero);
        let at = self.begin_guarded(Some(clear));
        let iu = self.s.m.undef(self.s.t.i32);
        // Operation 2 is OR.
        self.atomic_bin_op(self.cur, &[bitmap, k2, word, iu, iu, bit]);
        self.end_guarded(at, &[]);
        self.end_guarded(led, &[]);
    }
}
