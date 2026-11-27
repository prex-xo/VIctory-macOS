//! Where the copies a submission flushed sit in memory: a title copy
//! that takes bytes out of what each copy holds (victory-gpu's `host`)
//! reaches by its bytes only the copies that can hold them, not every
//! copy flushed.

/// `(start, end, key)` of a copy's memory.
type Entry = (usize, usize, usize);

/// The memory extents of copies, each named by the key its holder is
/// filed under: sorted by start, with the largest end up to each entry
/// beside it, a window `[from, to)` walks only the entries that can
/// overlap it and skips the rest without a look.
///
/// `host` flushes between title copies, so a submission files its ~9,000
/// copies in chunks and looks up after each. The entries are kept in two
/// levels: a large one and a small one that the newest chunks are merged
/// into, and that goes into the large one only once it has grown to a
/// share of it. A lookup walks both. Sorting everything again after each
/// chunk cost a full sort per title copy (5 ms a world submission in
/// `benches/title_copies.rs`), more than walking every copy did.
#[derive(Default)]
pub struct Extents {
    /// Most entries, in start order with their `ends`.
    main: Level,
    /// The newer entries, in start order with their `ends`.
    tail: Level,
    /// Pushed since the last lookup, in no order.
    fresh: Vec<Entry>,
    /// Where a merge builds the new order, kept so that a merge allocates
    /// nothing once it has grown.
    scratch: Vec<Entry>,
}

/// Entries in start order, and beside each the largest end up to it.
#[derive(Default)]
struct Level {
    entries: Vec<Entry>,
    /// The largest end in `entries[..=i]`: past the first entry that
    /// reaches a window's `from`, nothing before it reaches either.
    ends: Vec<usize>,
}

impl Level {
    fn clear(&mut self) {
        self.entries.clear();
        self.ends.clear();
    }

    /// Merges `new`, in start order, into the entries. The entries before
    /// the first new one keep their place and their `ends`; only the rest
    /// is merged and has its `ends` redone.
    fn merge(&mut self, new: &[Entry], scratch: &mut Vec<Entry>) {
        let Some(&(first_new, ..)) = new.first() else {
            return;
        };
        let keep = self
            .entries
            .partition_point(|&(start, ..)| start <= first_new);
        if keep == self.entries.len() {
            self.entries.extend_from_slice(new);
        } else {
            scratch.clear();
            scratch.reserve(self.entries.len() - keep + new.len());
            let (mut i, mut j, old) = (keep, 0, &self.entries);
            while i < old.len() && j < new.len() {
                if old[i].0 <= new[j].0 {
                    scratch.push(old[i]);
                    i += 1;
                } else {
                    scratch.push(new[j]);
                    j += 1;
                }
            }
            scratch.extend_from_slice(&old[i..]);
            scratch.extend_from_slice(&new[j..]);
            self.entries.truncate(keep);
            self.entries.extend_from_slice(scratch);
        }
        self.ends.truncate(keep);
        let mut max = self.ends.last().copied().unwrap_or(0);
        for &(_, end, _) in &self.entries[keep..] {
            max = max.max(end);
            self.ends.push(max);
        }
    }

    /// The keys of the entries that overlap `[from, to)`: those that start
    /// before `to`, walked back while any of them can reach `from` (the
    /// largest end so far says whether), kept where the entry reaches it.
    fn walk(&self, from: usize, to: usize) -> impl Iterator<Item = usize> + '_ {
        let last = self.entries.partition_point(|&(start, ..)| start < to);
        let empty = from >= to;
        (0..last)
            .rev()
            .take_while(move |&i| !empty && self.ends[i] > from)
            .filter(move |&i| self.entries[i].1 > from)
            .map(move |i| self.entries[i].2)
    }
}

/// The small level goes into the large one once it holds more than this
/// share of it (1/`SHARE`): each entry is then moved into the large level
/// a few times per submission, and each lookup merges into a level of at
/// most this share.
const SHARE: usize = 8;

impl Extents {
    /// A copy that flushed, its memory `[start, end)` filed under `key`.
    pub fn flushed(&mut self, key: usize, start: usize, end: usize) {
        self.fresh.push((start, end, key));
    }

    /// Nothing is held any more: as the holders are cleared.
    pub fn clear(&mut self) {
        self.main.clear();
        self.tail.clear();
        self.fresh.clear();
    }

    /// Files the entries pushed since the last lookup: sorted alone, then
    /// merged into the small level, which goes into the large one once
    /// it has grown past its share.
    fn build(&mut self) {
        if self.fresh.is_empty() {
            return;
        }
        // Only the start orders entries: the walk needs no more, and the
        // keys of equal starts may come out in any order.
        self.fresh.sort_unstable_by_key(|&(start, ..)| start);
        // A chunk past the small level's share (all of a submission's
        // copies filed before its first lookup) goes straight in.
        if self.tail.entries.is_empty() && self.fresh.len() * SHARE > self.main.entries.len() {
            self.main.merge(&self.fresh, &mut self.scratch);
            self.fresh.clear();
            return;
        }
        self.tail.merge(&self.fresh, &mut self.scratch);
        self.fresh.clear();
        if self.tail.entries.len() * SHARE > self.main.entries.len() {
            let tail = std::mem::take(&mut self.tail);
            self.main.merge(&tail.entries, &mut self.scratch);
            // Keep the small level's buffers for the next chunks.
            self.tail = tail;
            self.tail.clear();
        }
    }

    /// The keys whose memory can overlap `[from, to)`, in no particular
    /// order. The rest hold none of the window's bytes, whatever is taken
    /// out of the ones named.
    pub fn overlapping(&mut self, from: usize, to: usize) -> impl Iterator<Item = usize> + '_ {
        self.build();
        self.main.walk(from, to).chain(self.tail.walk(from, to))
    }
}

#[cfg(test)]
mod tests {
    use super::Extents;

    /// The keys a window reaches, in pushed order.
    fn keys(extents: &mut Extents, from: usize, to: usize) -> Vec<usize> {
        let mut out: Vec<_> = extents.overlapping(from, to).collect();
        out.sort_unstable();
        out
    }

    /// Only the copies whose memory a window's bytes can be in are
    /// reached: two copies sharing memory are both named for a title
    /// copy into it, the copies before and after are left alone.
    #[test]
    fn a_window_reaches_only_the_copies_its_bytes_can_be_in() {
        let mut extents = Extents::default();
        extents.flushed(1, 0x1_0000, 0x2_0000);
        // Shares 0x1_8000..0x2_0000 with 1.
        extents.flushed(2, 0x1_8000, 0x3_0000);
        extents.flushed(3, 0x5_0000, 0x6_0000);
        extents.flushed(4, 0, 0x100);
        assert_eq!(keys(&mut extents, 0x1_9000, 0x1_a000), vec![1, 2]);
        assert_eq!(keys(&mut extents, 0x2_8000, 0x2_9000), vec![2]);
        assert_eq!(keys(&mut extents, 0x4_0000, 0x4_1000), Vec::<usize>::new());
        assert_eq!(keys(&mut extents, 0x5_8000, 0x6_0000), vec![3]);
        assert_eq!(keys(&mut extents, 0x6_0000, 0x7_0000), Vec::<usize>::new());
        assert_eq!(keys(&mut extents, 0, 0x100), vec![4]);
        assert_eq!(keys(&mut extents, 0x40, 0x40), Vec::<usize>::new());
        assert_eq!(keys(&mut extents, 0x2_0000, 0), Vec::<usize>::new());
    }

    /// Pushed after a lookup and cleared, the extents answer from what
    /// they hold then (`host` flushes between title copies and clears
    /// with its holders).
    #[test]
    fn pushed_after_a_lookup_and_cleared() {
        let mut extents = Extents::default();
        extents.flushed(7, 0x100, 0x200);
        assert_eq!(keys(&mut extents, 0x180, 0x1c0), vec![7]);
        extents.clear();
        assert_eq!(keys(&mut extents, 0x180, 0x1c0), Vec::<usize>::new());
        extents.flushed(8, 0x300, 0x400);
        extents.flushed(9, 0x180, 0x280);
        assert_eq!(keys(&mut extents, 0x190, 0x1a0), vec![9]);
        assert_eq!(keys(&mut extents, 0x380, 0x3c0), vec![8]);
    }

    /// Pushed in chunks between lookups, as `host` flushes between title
    /// copies: each lookup reaches what a walk of everything pushed so
    /// far would, whether the chunk lands before, among or after the
    /// entries already in order.
    #[test]
    fn chunks_pushed_between_lookups_merge_in_order() {
        let mut state = 0x2545_F491_4F6C_DD1Du64;
        let mut next = move || {
            state = state
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            state >> 33
        };
        let mut extents = Extents::default();
        let mut all = Vec::new();
        let mut key = 0usize;
        for round in 0..60 {
            // Some chunks wholly after, some wholly before, most mixed.
            let base = match round % 3 {
                0 => 0x80_0000 + round * 0x1_0000,
                1 => 0x10_0000usize.saturating_sub(round * 0x1_0000),
                _ => 0,
            };
            for _ in 0..(next() % 40) as usize {
                let start = base + (next() % 0x100_0000) as usize;
                let end = start + (next() % 0x4_0000) as usize + 1;
                all.push((start, end, key));
                extents.flushed(key, start, end);
                key += 1;
            }
            for _ in 0..8 {
                let from = (next() % 0x180_0000) as usize;
                let to = from + (next() % 0x2_0000) as usize + 1;
                let mut reached: Vec<_> = extents.overlapping(from, to).collect();
                reached.sort_unstable();
                let mut walked: Vec<_> = all
                    .iter()
                    .filter(|&&(start, end, _)| start < to && end > from)
                    .map(|&(_, _, key)| key)
                    .collect();
                walked.sort_unstable();
                assert_eq!(reached, walked, "round {round}, window {from:#x}..{to:#x}");
            }
            if round == 30 {
                extents.clear();
                all.clear();
            }
        }
    }

    /// Every window of a pile of copies reaches the same keys a walk of
    /// all of them would (what `host`'s title copies keep is decided
    /// entry by entry, so this is the fix's equivalence, in the small).
    #[test]
    fn every_window_reaches_what_a_walk_of_all_would() {
        let mut state = 0x9E37_79B9_7F4A_7C15u64;
        let mut next = move || {
            state = state
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            state >> 33
        };
        let mut extents = Extents::default();
        let mut all = Vec::new();
        for key in 0..500usize {
            // Small extents in a small span, so they crowd and overlap.
            let start = (next() % 0x100_0000) as usize;
            let end = start + (next() % 0x10_0000) as usize + 1;
            all.push((start, end, key));
            extents.flushed(key, start, end);
        }
        for _ in 0..500 {
            let from = (next() % 0x110_0000) as usize;
            let to = from + (next() % 0x8_0000) as usize + 1;
            let mut reached: Vec<_> = extents.overlapping(from, to).collect();
            reached.sort_unstable();
            let mut walked: Vec<_> = all
                .iter()
                .filter(|&&(start, end, _)| start < to && end > from)
                .map(|&(_, _, key)| key)
                .collect();
            walked.sort_unstable();
            assert_eq!(reached, walked, "the window {from:#x}..{to:#x}");
        }
    }
}
