//! Free ranges of the heaps of textures (`crate::placement`): taking a
//! range, giving it back, which heap a new range goes into, and which heap
//! to empty.
//!
//! Under D3DMetal a heap is memory of the process whole, used or not (the
//! "owned physical footprint (unmapped) (graphics)" row of `footprint`),
//! until it is released. Textures given up leave holes, and first fit
//! refills holes in every heap with new textures, so few heaps empty: in
//! world runs on a 32 GB Mac 29 to 47 heaps of 64 MiB held 0.8 to 1.9 GB of
//! free space. [`to_empty`] picks a sparse heap whose textures may all go;
//! giving them up and placing nothing new there lets it empty, and then it
//! is released. Taking the fullest heap with room, or the newest, instead
//! of the first, held more heaps in a model of a title's textures, not
//! fewer.

use std::collections::BTreeMap;

/// The first range of `free` that fits `size` bytes at `align`: its
/// offset, taken out of `free`.
pub(crate) fn take(free: &mut BTreeMap<u64, u64>, size: u64, align: u64) -> Option<u64> {
    let (&start, &len, at) = free.iter().find_map(|(s, l)| {
        let at = s.next_multiple_of(align);
        (at + size <= s + l).then_some((s, l, at))
    })?;
    free.remove(&start);
    if at > start {
        free.insert(start, at - start);
    }
    if at + size < start + len {
        free.insert(at + size, start + len - at - size);
    }
    Some(at)
}

/// Whether [`take`] finds room for `size` bytes at `align` in `free`.
pub(crate) fn has_room(free: &BTreeMap<u64, u64>, size: u64, align: u64) -> bool {
    free.iter()
        .any(|(s, l)| s.next_multiple_of(align) + size <= s + l)
}

/// Gives `[offset, offset + size)` back to `free`, joined with its
/// neighbours.
pub(crate) fn give(free: &mut BTreeMap<u64, u64>, offset: u64, size: u64) {
    let (mut start, mut end) = (offset, offset + size);
    if let Some((&s, &l)) = free.range(..start).next_back()
        && s + l == start
    {
        free.remove(&s);
        start = s;
    }
    if let Some(&l) = free.get(&end) {
        free.remove(&end);
        end += l;
    }
    free.insert(start, end - start);
}

/// The heap a range of `size` bytes at `align` goes into, of heaps with
/// the free ranges `heaps`: the first with room other than `emptying`, or
/// `emptying` when it is the only one with room (better than a new heap).
pub(crate) fn pick<'a>(
    heaps: impl IntoIterator<Item = &'a BTreeMap<u64, u64>>,
    size: u64,
    align: u64,
    emptying: Option<usize>,
) -> Option<usize> {
    let mut last_resort = None;
    for (i, free) in heaps.into_iter().enumerate() {
        if !has_room(free, size, align) {
            continue;
        }
        if Some(i) != emptying {
            return Some(i);
        }
        last_resort = Some(i);
    }
    last_resort
}

/// One heap of textures as [`to_empty`] sees it.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) struct HeapUse {
    pub(crate) id: u64,
    /// Its bytes not given to a texture.
    pub(crate) free: u64,
    /// The bytes of its textures that may be given up now.
    pub(crate) may_go: u64,
}

/// Of `heaps` of `heap` bytes each, the one to empty: none while their free
/// space is at most [`EMPTY_SHARE`] of them, or less than two heaps;
/// else the one with the fewest bytes in use among those whose textures in
/// use may all go.
pub(crate) fn to_empty(heaps: &[HeapUse], heap: u64) -> Option<u64> {
    let total = heaps.len() as u64 * heap;
    let free: u64 = heaps.iter().map(|h| h.free).sum();
    if free < 2 * heap || (free as f64) <= EMPTY_SHARE * total as f64 {
        return None;
    }
    heaps
        .iter()
        .filter(|h| h.free < heap && h.may_go >= heap - h.free)
        .min_by_key(|h| (heap - h.free, h.id))
        .map(|h| h.id)
}

/// The share of the heaps' bytes free above which [`to_empty`] picks a
/// heap. In a model of a title's textures (a world's worth, then idle ones
/// given up), a third brought the heaps' free space from about 1.4 GB to
/// 0.7 GB for about 300 MB of textures made again over the run; a quarter,
/// two heaps a pass, halved it again for twice the textures made again.
pub(crate) const EMPTY_SHARE: f64 = 1.0 / 3.0;

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn ranges_are_taken_first_fit_and_joined_when_given_back() {
        let k = 64 << 10;
        let mut free = BTreeMap::from([(0, 10 * k)]);
        let a = take(&mut free, 3 * k, k).unwrap();
        let b = take(&mut free, 2 * k, k).unwrap();
        let c = take(&mut free, 4 * k, k).unwrap();
        assert_eq!((a, b, c), (0, 3 * k, 5 * k));
        assert_eq!(take(&mut free, 2 * k, k), None, "one 64 KiB range left");
        give(&mut free, b, 2 * k);
        assert_eq!(
            take(&mut free, 3 * k, k),
            None,
            "two ranges of 2 and 1, not joined"
        );
        give(&mut free, c, 4 * k);
        assert_eq!(
            free,
            BTreeMap::from([(3 * k, 7 * k)]),
            "joined with both neighbours"
        );
        give(&mut free, a, 3 * k);
        assert_eq!(free, BTreeMap::from([(0, 10 * k)]));
        // An aligned range inside a free one keeps the padding before it.
        let mut free = BTreeMap::from([(k, 10 * k)]);
        assert_eq!(take(&mut free, k, 4 * k), Some(4 * k));
        assert_eq!(free, BTreeMap::from([(k, 3 * k), (5 * k, 6 * k)]));
    }

    #[test]
    fn a_range_goes_into_the_first_heap_with_room_but_the_one_being_emptied() {
        let k = 64 << 10;
        let full = BTreeMap::new();
        let small = BTreeMap::from([(0, k)]);
        let roomy = BTreeMap::from([(0, 8 * k)]);
        assert_eq!(pick([&full, &small, &roomy], k, k, None), Some(1));
        assert_eq!(
            pick([&full, &small, &roomy], k, k, Some(1)),
            Some(2),
            "skips the one emptied"
        );
        assert_eq!(pick([&full, &small, &roomy], 2 * k, k, None), Some(2));
        assert_eq!(
            pick([&full, &small, &roomy], 2 * k, k, Some(2)),
            Some(2),
            "else it, not a new heap"
        );
        assert_eq!(pick([&full, &small, &roomy], 9 * k, k, None), None);
        // Room counts after alignment: 2 x 64 KiB at k is not 128 KiB aligned.
        let unaligned = BTreeMap::from([(k, 2 * k)]);
        assert!(!has_room(&unaligned, 2 * k, 2 * k));
        assert_eq!(pick([&unaligned, &roomy], 2 * k, 2 * k, None), Some(1));
    }

    #[test]
    fn the_heap_emptied_is_the_sparsest_whose_textures_may_all_go() {
        let heap = 64;
        let h = |id, free, may_go| HeapUse { id, free, may_go };
        // 4 heaps, 144 of 256 free: over a third and two heaps.
        let heaps = [h(0, 40, 24), h(1, 54, 10), h(2, 10, 54), h(3, 40, 24)];
        assert_eq!(to_empty(&heaps, heap), Some(1), "10 in use, all may go");
        // The sparsest holds a texture in use: the next one.
        let heaps = [h(0, 40, 24), h(1, 54, 3), h(2, 10, 54), h(3, 40, 24)];
        assert_eq!(to_empty(&heaps, heap), Some(0), "the first of equals");
        // Empty heaps are released already, not emptied.
        let heaps = [h(0, 64, 0), h(1, 64, 0), h(2, 10, 54)];
        assert_eq!(to_empty(&heaps, heap), Some(2));
        // No heap whose textures may all go.
        let heaps = [h(0, 40, 0), h(1, 54, 0), h(2, 10, 0), h(3, 40, 0)];
        assert_eq!(to_empty(&heaps, heap), None);
        // At most a third free: nothing to gain.
        let heaps = [
            h(0, 30, 34),
            h(1, 10, 54),
            h(2, 10, 54),
            h(3, 30, 34),
            h(4, 5, 59),
            h(5, 0, 64),
        ];
        assert_eq!(to_empty(&heaps, heap), None);
        // Over a third but less than two heaps free.
        let heaps = [h(0, 60, 4), h(1, 20, 44)];
        assert_eq!(to_empty(&heaps, heap), None);
        assert_eq!(to_empty(&[], heap), None);
    }

    /// The model `to_empty` was chosen with: heaps of 64 MiB under a
    /// title's textures, a world's worth and then fewer, each used in half
    /// the passes and given up after three idle ones; one heap emptied a
    /// pass, of textures not used in the pass, and 85 % of what it gives up
    /// made again at once. Returns the heaps held and their free bytes.
    fn model(emptying: bool, seed: u64) -> (usize, u64) {
        const HEAP: u64 = 64 << 20;
        const K: u64 = 64 << 10;
        let mut rng = seed;
        let mut next = move || {
            rng ^= rng << 13;
            rng ^= rng >> 7;
            rng ^= rng << 17;
            rng
        };
        // (id, free ranges).
        let mut heaps: Vec<(u64, BTreeMap<u64, u64>)> = Vec::new();
        let mut ids = 0;
        // (heap id, offset, size, last pass used).
        let mut held: Vec<(u64, u64, u64, u64)> = Vec::new();
        let mut live = 0;
        let place = |heaps: &mut Vec<(u64, BTreeMap<u64, u64>)>,
                     ids: &mut u64,
                     size,
                     avoid: Option<u64>| {
            let skip = avoid.and_then(|a| heaps.iter().position(|h| h.0 == a));
            let i = pick(heaps.iter().map(|h| &h.1), size, K, skip).unwrap_or_else(|| {
                heaps.push((*ids, BTreeMap::from([(0, HEAP)])));
                *ids += 1;
                heaps.len() - 1
            });
            (heaps[i].0, take(&mut heaps[i].1, size, K).unwrap())
        };
        let at =
            |heaps: &[(u64, BTreeMap<u64, u64>)], id| heaps.iter().position(|h| h.0 == id).unwrap();
        let rounds = 60;
        let mut emptied = None;
        for round in 0..=rounds {
            let target = if round < rounds / 2 {
                2048
            } else {
                2048 - 1300 * (round - rounds / 2) / (rounds / 2)
            } << 20;
            let mut i = 0;
            while i < held.len() {
                if next() % 2 == 0 {
                    held[i].3 = round;
                }
                if round > 0 && round - held[i].3 >= 3 {
                    let (h, offset, size, _) = held.swap_remove(i);
                    let h = at(&heaps, h);
                    give(&mut heaps[h].1, offset, size);
                    live -= size;
                } else {
                    i += 1;
                }
            }
            if emptying {
                let uses: Vec<HeapUse> = heaps
                    .iter()
                    .map(|(id, free)| HeapUse {
                        id: *id,
                        free: free.values().sum(),
                        may_go: held
                            .iter()
                            .filter(|t| t.0 == *id && t.3 < round)
                            .map(|t| t.2)
                            .sum(),
                    })
                    .collect();
                emptied = to_empty(&uses, HEAP);
                if let Some(e) = emptied {
                    let (gone, kept): (Vec<_>, Vec<_>) = held.iter().partition(|t| t.0 == e);
                    held = kept;
                    for (h, offset, size, last) in gone {
                        let h = at(&heaps, h);
                        give(&mut heaps[h].1, offset, size);
                        if next() % 100 < 85 {
                            let (h, offset) = place(&mut heaps, &mut ids, size, emptied);
                            held.push((h, offset, size, last));
                        } else {
                            live -= size;
                        }
                    }
                }
            }
            while live < target {
                // 64 KiB to 16 MiB, small ones most often, as textures are.
                let size = K << (next() % 9).min(next() % 9);
                let (h, offset) = place(&mut heaps, &mut ids, size, emptied);
                held.push((h, offset, size, round));
                live += size;
            }
            // Empty heaps past two spares are released, as `Heaps::reap` does.
            let mut spare = 0;
            heaps.retain(|(_, free)| {
                if free.get(&0) == Some(&HEAP) {
                    spare += 1;
                    spare <= 2
                } else {
                    true
                }
            });
        }
        (heaps.len(), heaps.iter().flat_map(|h| h.1.values()).sum())
    }

    #[test]
    fn emptying_sparse_heaps_holds_fewer_heaps_in_the_model() {
        for seed in [1, 7, 42, 1234, 99991] {
            let (before, before_free) = model(false, seed);
            let (after, after_free) = model(true, seed);
            eprintln!(
                "seed {seed}: first fit {before} heaps ({} MB free), emptying {after} heaps ({} MB free)",
                before_free >> 20,
                after_free >> 20
            );
            assert!(
                after < before,
                "seed {seed}: {after} heaps emptying, {before} without"
            );
            assert!(
                after_free < before_free * 3 / 4,
                "seed {seed}: {after_free} free emptying, {before_free} without"
            );
        }
    }
}
