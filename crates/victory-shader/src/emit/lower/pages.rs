//! Reads of title memory that cross a 64 KiB page (`emit`, "Title memory").
//! A lookup in the page table finds the host copy that holds the page of an
//! access's first byte; the access's dwords are then read through that
//! copy's view at offsets from there. The page after it may be another
//! copy's (two host copies next to each other, or none there): through the
//! first copy's view those dwords are past its end and read 0, where the
//! console reads the memory that is there. So a read whose bytes reach the
//! next page looks that page up as well, only where they do, and each of
//! its dwords that lies there is read through that page's copy.

use super::*;

/// A read of title memory whose bytes may reach the next page, kept by the
/// handle [`Lower::memory`] gave for it.
#[derive(Clone, Copy)]
pub(super) struct Tail {
    /// The offset [`Lower::memory`] gave with the handle.
    given: V,
    /// The view of the copy holding the access's last byte where that lies
    /// on the next page; the null view's elsewhere.
    handle: V,
    /// The access's offset in that copy (below 0 by what lies on the first
    /// page).
    at: V,
    /// Whether the access crosses, and its bytes before the page's end.
    crosses: V,
    before: V,
}

impl Lower<'_> {
    /// After [`Lower::memory`]'s lookup for a read of `bytes` bytes at
    /// `(hi:lo)`, which gave `handle` and `offset`: where the read reaches
    /// the next page (in the `active` lanes), that page's lookup, a miss
    /// marked as the first page's is. Loads through `handle` then take the
    /// dwords on that page from its copy ([`Lower::tail_load`]). May end
    /// in a new block.
    pub(super) fn page_tail(&mut self, lo: V, hi: V, bytes: u32, handle: V, offset: V, active: Option<V>) {
        let c = self.cur;
        let k = |l: &mut Self, x: u32| l.s.m.i32(x);
        let (page, low, size) = (k(self, 1 << 16), k(self, 0xFFFF), k(self, bytes));
        let in_page = self.s.m.bin(c, Bin::And, lo, low);
        let before = self.s.m.bin(c, Bin::Sub, page, in_page);
        let mut crosses = self.s.m.cmp(c, pred::IUGT, size, before);
        if let Some(active) = active {
            crosses = self.s.m.bin(c, Bin::And, crosses, active);
        }
        let null = self.memory_null();
        let guard = self.begin_guarded(Some(crosses));
        let (last, mask) = (k(self, bytes - 1), k(self, 0xFFFF));
        let (end_lo, end_hi) = self.add64(lo, hi, last, false);
        let end_hi = self.s.m.bin(self.cur, Bin::And, end_hi, mask);
        let (entry, block, l1) = self.memory_page(end_lo, end_hi);
        let c = self.cur;
        let map = self.s.memory_handle();
        let (k3, first) = (k(self, 3), k(self, 4096 * 4));
        let at = self.s.m.bin(c, Bin::Shl, entry, k3);
        let at = self.s.m.bin(c, Bin::Add, at, first);
        let (u, two, al) = (self.s.m.undef(self.s.t.i32), self.s.m.i8(3), k(self, 4));
        let r = self.raw_op(c, &[map, at, u, two, al]);
        let [view, start] = [0, 1].map(|j| {
            let x = self.s.m.extract(c, r, j);
            self.as_int(x)
        });
        // A page no copy holds reads 0 and is marked missed.
        let miss = self.s.m.cmp(c, pred::EQ, view, null);
        let missed = self.memory_mark_index(entry, block, l1);
        let bitmap = self.s.memory_dirty_handle();
        let marking = self.begin_guarded(Some(miss));
        self.mark_memory_bit(bitmap, missed);
        self.end_guarded(marking, &[]);
        let zero = k(self, 0);
        let [view, start] = self.end_guarded(guard, &[(view, null), (start, zero)])[..] else {
            unreachable!("two values in, two out")
        };
        let c = self.cur;
        let tail = self.s.heap_handle(c, view, false, true, (11, 0));
        self.s.view_index.remove(&tail);
        let at = self.s.m.bin(c, Bin::Sub, lo, start);
        self.tails.insert(
            handle,
            Tail {
                given: offset,
                handle: tail,
                at,
                crosses,
                before,
            },
        );
    }

    /// A raw load through the handle of a read of title memory that may
    /// reach the next page ([`Lower::page_tail`]): each of its dwords is
    /// also read through that page's copy, a load each (the access's
    /// offset there is below 0, and an offset wraps from one load to the
    /// next, not within a load), and taken from there where it lies on
    /// that page ([`Module::tails`]).
    pub(super) fn tail_load(&mut self, block: usize, args: &[V], tail: Tail) -> V {
        let r = self.raw_op(block, args);
        // How far this load is from the access's start.
        let moved = if args[1] == tail.given {
            self.s.m.i32(0)
        } else {
            self.s.m.bin(block, Bin::Sub, args[1], tail.given)
        };
        let fields = self.s.m.int_value(args[3]).map_or(0xF, |mask| mask as u8);
        let one = self.s.m.i8(1);
        let mut tails = [None; 4];
        for (k, field) in tails.iter_mut().enumerate() {
            if fields >> k & 1 == 0 {
                continue;
            }
            let step = self.s.m.i32(4 * k as u32);
            let from = self.s.m.bin(block, Bin::Add, moved, step);
            let past = self.s.m.cmp(block, pred::SGE, from, tail.before);
            let past = self.s.m.bin(block, Bin::And, tail.crosses, past);
            let at = self.s.m.bin(block, Bin::Add, tail.at, from);
            let there = self.raw_op(block, &[tail.handle, at, args[2], one, args[4]]);
            *field = Some((past, there));
        }
        self.s.m.tails.insert(r, tails);
        r
    }
}
