//! Opt-in fast hashing and hash-backed range table for the bindless view
//! registry (`VICTORY_QUICK_SYNC=1`, off by default).
//!
//! During descriptor-heap syncs, the bindless registry (`crate::bindless`)
//! updates slot-to-view and active buffer-range reference counts tens of
//! thousands of times per frame across hundreds of thousands of live entries.
//! Replacing the default `BTreeMap` range table with a hash map and using a
//! lightweight multiply-and-fold hasher for internal integer/tuple keys cuts
//! pointer chasing and per-slot hashing overhead without changing observable
//! registry semantics.
//!
//! Pure `std` so unit tests run on every host as well as under Windows/Wine.

#![cfg_attr(not(windows), allow(dead_code))]

use std::collections::{BTreeMap, HashMap, btree_map, hash_map};
use std::ffi::OsStr;
use std::hash::{BuildHasher, DefaultHasher, Hash, Hasher, RandomState};
use std::sync::OnceLock;

/// Parsed `VICTORY_QUICK_SYNC` feature selection.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) struct QuickSync {
    /// Use multiply-and-fold hashing and hash-backed range counts in `bindless`.
    pub(crate) maps: bool,
    /// Deduplicate and pre-resolve buffer SRV/CBV records per synced heap page.
    pub(crate) views: bool,
}

/// Parses `VICTORY_QUICK_SYNC` (off by default).
///
/// Recognized values (case-insensitive, comma-separated tokens allowed):
/// - unset, `""`, `"0"`, `"false"`, `"off"` => both off
/// - `"1"`, `"true"`, `"on"`, `"all"` => both on
/// - `"maps"` => fast bindless maps only
/// - `"views"` => per-page quick buffer-SRV/CBV resolution only
/// - `"maps,views"` => both on
pub(crate) fn parse_quick_sync(value: Option<&OsStr>) -> QuickSync {
    let Some(raw) = value.and_then(OsStr::to_str) else {
        return QuickSync {
            maps: false,
            views: false,
        };
    };
    let trimmed = raw.trim();
    if trimmed.is_empty() {
        return QuickSync {
            maps: false,
            views: false,
        };
    }
    let mut cfg = QuickSync {
        maps: false,
        views: false,
    };
    for token in trimmed.split(',') {
        match token.trim().to_ascii_lowercase().as_str() {
            "0" | "false" | "off" | "" => {}
            "1" | "true" | "on" | "all" => {
                cfg.maps = true;
                cfg.views = true;
            }
            "maps" => cfg.maps = true,
            "views" => cfg.views = true,
            _ => {}
        }
    }
    cfg
}

/// Cached `VICTORY_QUICK_SYNC` configuration for this process.
pub(crate) fn quick_sync() -> QuickSync {
    static CFG: OnceLock<QuickSync> = OnceLock::new();
    *CFG.get_or_init(|| {
        let cfg = parse_quick_sync(std::env::var_os("VICTORY_QUICK_SYNC").as_deref());
        if cfg.maps || cfg.views {
            tracing::info!(
                maps = cfg.maps,
                views = cfg.views,
                "quick heap sync enabled (VICTORY_QUICK_SYNC)"
            );
        }
        cfg
    })
}

/// Whether fast bindless registry maps (`VICTORY_QUICK_SYNC=1` or `=maps`) are enabled.
pub(crate) fn quick_maps() -> bool {
    quick_sync().maps
}

/// Whether per-page quick buffer-SRV/CBV resolution (`VICTORY_QUICK_SYNC=1` or `=views`) is enabled.
pub(crate) fn quick_views() -> bool {
    quick_sync().views
}

/// Hash builder selecting either `std`'s randomized SipHash (`Standard`) or
/// fast 128-bit multiply-and-fold hashing (`Fold`) for trusted internal keys.
#[derive(Clone)]
pub(crate) enum Hashing {
    Standard(RandomState),
    Fold,
}

impl Hashing {
    pub(crate) fn new(quick: bool) -> Self {
        if quick {
            Self::Fold
        } else {
            Self::Standard(RandomState::new())
        }
    }
}

impl BuildHasher for Hashing {
    type Hasher = FoldHasher;

    fn build_hasher(&self) -> Self::Hasher {
        match self {
            Self::Standard(state) => FoldHasher::Standard(state.build_hasher()),
            Self::Fold => FoldHasher::Fold(0),
        }
    }
}

/// 128-bit odd golden-ratio multiplier for multiply-and-fold mixing.
const FOLD_MULTIPLIER: u128 = 0x9E37_79B9_7F4A_7C15_F39C_C060_5CED_C835;

/// Hasher produced by [`Hashing`].
pub(crate) enum FoldHasher {
    Standard(DefaultHasher),
    Fold(u64),
}

impl FoldHasher {
    #[inline]
    fn mix(state: &mut u64, word: u64) {
        let product = u128::from(*state ^ word).wrapping_mul(FOLD_MULTIPLIER);
        *state = (product as u64) ^ ((product >> 64) as u64);
    }
}

impl Hasher for FoldHasher {
    #[inline]
    fn finish(&self) -> u64 {
        match self {
            Self::Standard(h) => h.finish(),
            Self::Fold(state) => *state,
        }
    }

    #[inline]
    fn write(&mut self, bytes: &[u8]) {
        match self {
            Self::Standard(h) => h.write(bytes),
            Self::Fold(state) => {
                for chunk in bytes.chunks(8) {
                    let mut buf = [0u8; 8];
                    buf[..chunk.len()].copy_from_slice(chunk);
                    Self::mix(state, u64::from_le_bytes(buf));
                }
                // Mix length so byte slices differing only by trailing zero bytes hash distinctly.
                Self::mix(state, bytes.len() as u64);
            }
        }
    }

    #[inline]
    fn write_u8(&mut self, i: u8) {
        match self {
            Self::Standard(h) => h.write_u8(i),
            Self::Fold(state) => Self::mix(state, u64::from(i)),
        }
    }

    #[inline]
    fn write_u16(&mut self, i: u16) {
        match self {
            Self::Standard(h) => h.write_u16(i),
            Self::Fold(state) => Self::mix(state, u64::from(i)),
        }
    }

    #[inline]
    fn write_u32(&mut self, i: u32) {
        match self {
            Self::Standard(h) => h.write_u32(i),
            Self::Fold(state) => Self::mix(state, u64::from(i)),
        }
    }

    #[inline]
    fn write_u64(&mut self, i: u64) {
        match self {
            Self::Standard(h) => h.write_u64(i),
            Self::Fold(state) => Self::mix(state, i),
        }
    }

    #[inline]
    fn write_usize(&mut self, i: usize) {
        match self {
            Self::Standard(h) => h.write_usize(i),
            Self::Fold(state) => Self::mix(state, i as u64),
        }
    }
}

/// Active buffer-range table inside `bindless::Viewed`: either the default
/// `BTreeMap` or a fold-hashed `HashMap` when `quick_maps` is enabled.
pub(crate) enum RangesMap<K, V> {
    Tree(BTreeMap<K, V>),
    Hash(HashMap<K, V, Hashing>),
}

impl<K: Ord + Hash, V> RangesMap<K, V> {
    pub(crate) fn new(quick: bool) -> Self {
        if quick {
            Self::Hash(HashMap::with_hasher(Hashing::Fold))
        } else {
            Self::Tree(BTreeMap::new())
        }
    }

    /// Whether iteration order is unordered (hash-backed) rather than sorted by key.
    #[inline]
    pub(crate) fn is_hashed(&self) -> bool {
        matches!(self, Self::Hash(_))
    }

    #[inline]
    pub(crate) fn entry_or(&mut self, key: K, make: impl FnOnce() -> V) -> &mut V {
        match self {
            Self::Tree(map) => map.entry(key).or_insert_with(make),
            Self::Hash(map) => map.entry(key).or_insert_with(make),
        }
    }

    #[inline]
    pub(crate) fn remove(&mut self, key: &K) -> Option<V> {
        match self {
            Self::Tree(map) => map.remove(key),
            Self::Hash(map) => map.remove(key),
        }
    }

    #[cfg(any(test, feature = "diagnostics"))]
    pub(crate) fn len(&self) -> usize {
        match self {
            Self::Tree(map) => map.len(),
            Self::Hash(map) => map.len(),
        }
    }

    pub(crate) fn iter(&self) -> RangesIter<'_, K, V> {
        match self {
            Self::Tree(map) => RangesIter::Tree(map.iter()),
            Self::Hash(map) => RangesIter::Hash(map.iter()),
        }
    }

    #[cfg(any(test, feature = "diagnostics"))]
    pub(crate) fn keys(&self) -> RangesKeys<'_, K, V> {
        match self {
            Self::Tree(map) => RangesKeys::Tree(map.keys()),
            Self::Hash(map) => RangesKeys::Hash(map.keys()),
        }
    }
}

/// Borrowing iterator over entries of [`RangesMap`] without heap allocation.
pub(crate) enum RangesIter<'a, K, V> {
    Tree(btree_map::Iter<'a, K, V>),
    Hash(hash_map::Iter<'a, K, V>),
}

impl<'a, K, V> Iterator for RangesIter<'a, K, V> {
    type Item = (&'a K, &'a V);

    #[inline]
    fn next(&mut self) -> Option<Self::Item> {
        match self {
            Self::Tree(it) => it.next(),
            Self::Hash(it) => it.next(),
        }
    }
}

/// Borrowing iterator over keys of [`RangesMap`] without heap allocation.
#[cfg(any(test, feature = "diagnostics"))]
pub(crate) enum RangesKeys<'a, K, V> {
    Tree(btree_map::Keys<'a, K, V>),
    Hash(hash_map::Keys<'a, K, V>),
}

#[cfg(any(test, feature = "diagnostics"))]
impl<'a, K, V> Iterator for RangesKeys<'a, K, V> {
    type Item = &'a K;

    #[inline]
    fn next(&mut self) -> Option<Self::Item> {
        match self {
            Self::Tree(it) => it.next(),
            Self::Hash(it) => it.next(),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::HashSet;

    type RangeKey = (usize, u64, u64, bool);

    struct XorShift64(u64);

    impl XorShift64 {
        fn next(&mut self) -> u64 {
            self.0 ^= self.0 << 13;
            self.0 ^= self.0 >> 7;
            self.0 ^= self.0 << 17;
            self.0
        }

        fn range(&mut self, copies: u64, starts: u64) -> RangeKey {
            let copy = 0x7000_0000_0000 + (self.next() % copies) as usize * 0x140;
            let start = 0x1_0000_0000 + (self.next() % starts) * 16;
            let len = 16 + (self.next() % 4096);
            (copy, start, start + len, self.next().is_multiple_of(8))
        }
    }

    #[test]
    fn quick_sync_env_parsing() {
        let parse = |s: &str| parse_quick_sync(Some(OsStr::new(s)));
        assert_eq!(
            parse_quick_sync(None),
            QuickSync {
                maps: false,
                views: false
            }
        );
        for off in ["", "   ", "0", "false", "OFF", "False", "unknown"] {
            assert_eq!(
                parse(off),
                QuickSync {
                    maps: false,
                    views: false
                },
                "value {off:?}"
            );
        }
        for both in ["1", "true", "ON", "all", "maps,views", " views , MAPS "] {
            assert_eq!(
                parse(both),
                QuickSync {
                    maps: true,
                    views: true
                },
                "value {both:?}"
            );
        }
        assert_eq!(
            parse("maps"),
            QuickSync {
                maps: true,
                views: false
            }
        );
        assert_eq!(
            parse("views"),
            QuickSync {
                maps: false,
                views: true
            }
        );
    }

    #[test]
    fn standard_hashing_matches_random_state() {
        let state = RandomState::new();
        let hashing = Hashing::Standard(state.clone());
        for slot in [0u32, 1, 63, 64, 65_535, 639_999, u32::MAX] {
            assert_eq!(hashing.hash_one(slot), state.hash_one(slot));
        }
        let key: RangeKey = (0x7000_1234_5678, 0x1_0000_0040, 0x1_0000_0140, true);
        assert_eq!(hashing.hash_one(key), state.hash_one(key));
        assert_eq!(
            hashing.hash_one(b"sample-key"),
            state.hash_one(b"sample-key")
        );
    }

    #[test]
    fn fold_hasher_avalanche_and_uniqueness() {
        let fold = Hashing::Fold;
        let distribution = |hashes: &[u64]| {
            let low: HashSet<u64> = hashes.iter().map(|h| h & 0xFFFF).collect();
            let high: HashSet<u64> = hashes.iter().map(|h| h >> 57).collect();
            (low.len(), high.len())
        };

        let slot_hashes: Vec<u64> = (0..1u32 << 16).map(|s| fold.hash_one(s)).collect();
        let (low, high) = distribution(&slot_hashes);
        assert!(
            low > 38_000 && high == 128,
            "sequential slot indices: {low} buckets, {high} tags"
        );

        let base_copy = 0x7000_1234_5000usize;
        let addr_hashes: Vec<u64> = (0..1u64 << 16)
            .map(|i| {
                fold.hash_one((
                    base_copy,
                    0x1_0000_0000 + 16 * i,
                    0x1_0000_0100 + 16 * i,
                    false,
                ))
            })
            .collect();
        let (low, high) = distribution(&addr_hashes);
        assert!(
            low > 38_000 && high == 128,
            "sequential buffer addresses: {low} buckets, {high} tags"
        );

        let copy_hashes: Vec<u64> = (0..1usize << 16)
            .map(|i| {
                fold.hash_one((
                    base_copy + 0x140 * i,
                    0x1_0000_0000u64,
                    0x1_0000_0100u64,
                    false,
                ))
            })
            .collect();
        let (low, high) = distribution(&copy_hashes);
        assert!(
            low > 38_000 && high == 128,
            "sequential copy pointers: {low} buckets, {high} tags"
        );

        // Permuting fields, flipping flags, or appending zero bytes must change the hash.
        assert_ne!(
            fold.hash_one((1usize, 2u64, 3u64, false)),
            fold.hash_one((2usize, 1u64, 3u64, false))
        );
        assert_ne!(
            fold.hash_one((1usize, 2u64, 3u64, false)),
            fold.hash_one((1usize, 2u64, 3u64, true))
        );
        assert_ne!(
            fold.hash_one([1u8, 2].as_slice()),
            fold.hash_one([1u8, 2, 0].as_slice())
        );
    }

    #[test]
    fn ranges_map_hash_matches_tree_under_randomized_workload() {
        let mut rng = XorShift64(0x9E37_79B9_7F4A_7C15);
        let mut tree_slots: HashMap<u32, RangeKey, Hashing> =
            HashMap::with_hasher(Hashing::new(false));
        let mut hash_slots: HashMap<u32, RangeKey, Hashing> =
            HashMap::with_hasher(Hashing::new(true));
        let mut tree_ranges: RangesMap<RangeKey, u32> = RangesMap::new(false);
        let mut hash_ranges: RangesMap<RangeKey, u32> = RangesMap::new(true);

        assert!(!tree_ranges.is_hashed());
        assert!(hash_ranges.is_hashed());
        assert!(matches!(tree_ranges, RangesMap::Tree(_)));
        assert!(matches!(hash_ranges, RangesMap::Hash(_)));

        let update = |ranges: &mut RangesMap<RangeKey, u32>, key: RangeKey, add: bool| {
            let count = ranges.entry_or(key, || 0);
            *count = if add {
                *count + 1
            } else {
                count.saturating_sub(1)
            };
            if *count == 0 {
                ranges.remove(&key);
            }
        };

        for step in 0..150_000u64 {
            let slot = (rng.next() % 4_096) as u32;
            let next_range = (!step.is_multiple_of(7)).then(|| rng.range(32, 64));
            for (slots, ranges) in [
                (&mut tree_slots, &mut tree_ranges),
                (&mut hash_slots, &mut hash_ranges),
            ] {
                if let Some(old) = slots.remove(&slot) {
                    update(ranges, old, false);
                }
                if let Some(range) = next_range {
                    update(ranges, range, true);
                    slots.insert(slot, range);
                }
            }
        }

        assert_eq!(tree_ranges.len(), hash_ranges.len());
        assert!(tree_ranges.len() > 500);

        let collect_entries = |ranges: &RangesMap<RangeKey, u32>| {
            let mut out: Vec<(RangeKey, u32)> = ranges.iter().map(|(&k, &v)| (k, v)).collect();
            out.sort_unstable();
            out
        };
        let tree_entries = collect_entries(&tree_ranges);
        let hash_entries = collect_entries(&hash_ranges);
        assert_eq!(tree_entries, hash_entries);

        // Verify counts match exact slot occupancy.
        let mut expected: BTreeMap<RangeKey, u32> = BTreeMap::new();
        for &range in hash_slots.values() {
            *expected.entry(range).or_insert(0) += 1;
        }
        let expected_entries: Vec<(RangeKey, u32)> = expected.into_iter().collect();
        assert_eq!(hash_entries, expected_entries);

        let mut tree_keys: Vec<RangeKey> = tree_ranges.keys().copied().collect();
        let mut hash_keys: Vec<RangeKey> = hash_ranges.keys().copied().collect();
        tree_keys.sort_unstable();
        hash_keys.sort_unstable();
        assert_eq!(tree_keys, hash_keys);
    }
}
