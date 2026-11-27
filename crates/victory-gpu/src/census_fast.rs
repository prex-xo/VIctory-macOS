//! Opt-in fast bindless UAV range index and snapshot cache
//! (`VICTORY_CENSUS_FAST=1`, off by default).
//!
//! Under Wine/D3DMetal, buffer/UAV barriers after heap-writing draws query all
//! active unordered-access buffer ranges in the bindless view registry so
//! overlapping widened three-channel SRVs can be refreshed. Scanning the full
//! range table under the bindless mutex on every barrier contends with title
//! threads mirroring descriptors and walks non-UAV shader-resource ranges that
//! cannot participate in widening.
//!
//! When enabled, each view heap maintains:
//! - [`UnorderedIndex`]: a secondary map holding only active UAV buffer ranges,
//!   updated strictly when a UAV range's slot count transitions between 0 and 1.
//! - [`SnapshotCache`]: a generation-stamped `Arc` slice of the last widening
//!   snapshot. While no UAV range has been added or removed, barrier queries
//!   reuse the cached snapshot without acquiring the bindless mutex at all.
//!
//! Pure `std` so unit tests run on every host as well as under Windows/Wine.

#![cfg_attr(not(windows), allow(dead_code))]

use std::collections::BTreeMap;
use std::sync::atomic::{
    AtomicU64,
    Ordering::{Acquire, Relaxed, Release},
};
use std::sync::{Arc, Mutex, OnceLock};

/// A buffer range key as tracked by the bindless view registry:
/// `(copy_ptr, start_address, end_address, is_uav)`.
pub(crate) type RangeKey = (usize, u64, u64, bool);

/// A widened-write candidate range: `(copy, start_address, byte_length)`.
pub(crate) type WidenedWrite<S> = (Arc<S>, usize, usize);

/// Shared immutable snapshot of active widened-write candidate ranges.
pub(crate) type Snapshot<S> = Arc<[WidenedWrite<S>]>;

/// Whether the fast bindless UAV range cache is enabled
/// (`VICTORY_CENSUS_FAST=1`, off by default).
pub(crate) fn on() -> bool {
    static ON: OnceLock<bool> = OnceLock::new();
    *ON.get_or_init(|| {
        let enabled = crate::switch::on("VICTORY_CENSUS_FAST");
        if enabled {
            tracing::info!(
                "fast bindless UAV range cache enabled (VICTORY_CENSUS_FAST=1)"
            );
        }
        enabled
    })
}

/// Secondary index holding only active unordered-access buffer ranges,
/// maintained under the bindless registry mutex alongside the full range map.
pub(crate) struct UnorderedIndex<S> {
    ranges: BTreeMap<RangeKey, Arc<S>>,
}

impl<S> UnorderedIndex<S> {
    pub(crate) fn new() -> Self {
        Self {
            ranges: BTreeMap::new(),
        }
    }

    /// Records a UAV range whose reference count transitioned from 0 to 1.
    pub(crate) fn insert(&mut self, key: RangeKey, copy: &Arc<S>) {
        self.ranges.insert(key, Arc::clone(copy));
    }

    /// Removes a UAV range whose reference count transitioned from 1 to 0.
    pub(crate) fn remove(&mut self, key: &RangeKey) {
        self.ranges.remove(key);
    }

    /// Materializes non-empty UAV ranges in deterministic key order, matching
    /// the full-map filter in `bindless::widened_writes`.
    pub(crate) fn snapshot(&self) -> Snapshot<S> {
        self.ranges
            .iter()
            .filter(|((_, address, end, _), _)| address < end)
            .map(|(&(_, address, end, _), copy)| {
                (Arc::clone(copy), address as usize, (end - address) as usize)
            })
            .collect()
    }

    #[cfg(test)]
    fn len(&self) -> usize {
        self.ranges.len()
    }
}

/// Lock-free-checked snapshot cache for a heap's active UAV ranges.
///
/// Lock ordering: whenever both locks are held, the bindless registry mutex is
/// always acquired before `cached`. [`Self::get_cached`] acquires only
/// `cached` and never touches the bindless registry mutex.
pub(crate) struct SnapshotCache<S> {
    /// Incremented under the bindless registry mutex whenever [`UnorderedIndex`]
    /// gains or loses an entry.
    generation: AtomicU64,
    /// Last materialized snapshot and the generation it corresponds to.
    cached: Mutex<Option<(u64, Snapshot<S>)>>,
}

impl<S> Default for SnapshotCache<S> {
    fn default() -> Self {
        Self {
            generation: AtomicU64::new(0),
            cached: Mutex::new(None),
        }
    }
}

impl<S> SnapshotCache<S> {
    /// Invalidates the cached snapshot after [`UnorderedIndex`] mutates.
    ///
    /// Must be called while holding the bindless registry mutex so readers
    /// observing the new generation wait for the mutation to complete before
    /// rebuilding. Clears the cached `Arc` slice immediately so released copies
    /// are not retained past their last slot.
    pub(crate) fn invalidate(&self) {
        self.generation.fetch_add(1, Release);
        let dropped = self
            .cached
            .lock()
            .unwrap_or_else(|e| e.into_inner())
            .take();
        drop(dropped);
    }

    /// Returns the cached snapshot if [`UnorderedIndex`] has not changed since
    /// it was built, without acquiring the bindless registry mutex.
    pub(crate) fn get_cached(&self) -> Option<Snapshot<S>> {
        let generation = self.generation.load(Acquire);
        let guard = self.cached.lock().unwrap_or_else(|e| e.into_inner());
        match &*guard {
            Some((cached_gen, snap)) if *cached_gen == generation => Some(Arc::clone(snap)),
            _ => None,
        }
    }

    /// Returns or builds the snapshot for the current generation while the
    /// caller holds the bindless registry mutex.
    pub(crate) fn get_or_build(&self, index: &UnorderedIndex<S>) -> Snapshot<S> {
        let generation = self.generation.load(Relaxed);
        {
            let guard = self.cached.lock().unwrap_or_else(|e| e.into_inner());
            if let Some((cached_gen, snap)) = &*guard
                && *cached_gen == generation
            {
                return Arc::clone(snap);
            }
        }
        let snap = index.snapshot();
        let mut guard = self.cached.lock().unwrap_or_else(|e| e.into_inner());
        *guard = Some((generation, Arc::clone(&snap)));
        snap
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::HashMap;

    #[derive(Debug)]
    struct CopyBuf(#[allow(dead_code)] u32);

    struct Rng(u64);

    impl Rng {
        fn next(&mut self) -> u64 {
            self.0 ^= self.0 << 13;
            self.0 ^= self.0 >> 7;
            self.0 ^= self.0 << 17;
            self.0
        }
    }

    /// Minimal model of `bindless::Inner` exercising `UnorderedIndex` and
    /// `SnapshotCache` on every host platform.
    struct ModelRegistry {
        slots: HashMap<u32, (RangeKey, Arc<CopyBuf>)>,
        ranges: BTreeMap<RangeKey, (u32, Arc<CopyBuf>)>,
        unordered: UnorderedIndex<CopyBuf>,
        notify: bool,
    }

    impl ModelRegistry {
        fn new(notify: bool) -> Self {
            Self {
                slots: HashMap::new(),
                ranges: BTreeMap::new(),
                unordered: UnorderedIndex::new(),
                notify,
            }
        }

        fn count(&mut self, key: RangeKey, copy: &Arc<CopyBuf>, add: bool) -> bool {
            let entry = self
                .ranges
                .entry(key)
                .or_insert_with(|| (0, Arc::clone(copy)));
            let was = entry.0;
            entry.0 = if add {
                entry.0 + 1
            } else {
                entry.0.saturating_sub(1)
            };
            let now = entry.0;
            if now == 0 {
                self.ranges.remove(&key);
            }
            if (was == 0) != (now == 0) && key.3 {
                if now > 0 {
                    self.unordered.insert(key, copy);
                } else {
                    self.unordered.remove(&key);
                }
                return true;
            }
            false
        }

        fn set(
            &mut self,
            cache: &SnapshotCache<CopyBuf>,
            slot: u32,
            seen: Option<(RangeKey, Arc<CopyBuf>)>,
        ) {
            let mut changed = false;
            if let Some((old_key, old_copy)) = self.slots.remove(&slot) {
                changed |= self.count(old_key, &old_copy, false);
            }
            if let Some((key, copy)) = seen {
                changed |= self.count(key, &copy, true);
                self.slots.insert(slot, (key, copy));
            }
            if changed && self.notify {
                cache.invalidate();
            }
        }

        fn full_scan(&self) -> Vec<(usize, usize, usize)> {
            let mut out: Vec<_> = self
                .ranges
                .iter()
                .filter(|((_, addr, end, uav), _)| *uav && addr < end)
                .map(|((_, addr, end, _), (_, copy))| {
                    (Arc::as_ptr(copy) as usize, *addr as usize, (end - addr) as usize)
                })
                .collect();
            out.sort_unstable();
            out
        }
    }

    fn normalize(writes: &[WidenedWrite<CopyBuf>]) -> Vec<(usize, usize, usize)> {
        let mut out: Vec<_> = writes
            .iter()
            .map(|(copy, addr, len)| (Arc::as_ptr(copy) as usize, *addr, *len))
            .collect();
        out.sort_unstable();
        out
    }

    fn query(
        reg: &Mutex<ModelRegistry>,
        cache: &SnapshotCache<CopyBuf>,
    ) -> (Arc<[WidenedWrite<CopyBuf>]>, bool) {
        if let Some(hit) = cache.get_cached() {
            return (hit, true);
        }
        let guard = reg.lock().unwrap();
        (cache.get_or_build(&guard.unordered), false)
    }

    #[test]
    fn tracks_added_replaced_cleared_empty_and_high_slots() {
        let a = Arc::new(CopyBuf(0));
        let b = Arc::new(CopyBuf(1));
        let pa = Arc::as_ptr(&a) as usize;
        let pb = Arc::as_ptr(&b) as usize;
        let reg = Mutex::new(ModelRegistry::new(true));
        let cache = SnapshotCache::default();

        let set = |slot: u32, spec: Option<(&Arc<CopyBuf>, u64, u64, bool)>| {
            let mut guard = reg.lock().unwrap();
            let entry = spec.map(|(copy, start, end, uav)| {
                ((Arc::as_ptr(copy) as usize, start, end, uav), Arc::clone(copy))
            });
            guard.set(&cache, slot, entry);
        };

        assert_eq!(normalize(&query(&reg, &cache).0), []);

        // Added UAV slot.
        set(0, Some((&a, 0x100, 0x200, true)));
        let (first, hit) = query(&reg, &cache);
        assert!(!hit);
        assert_eq!(normalize(&first), [(pa, 0x100, 0x100)]);

        // Repeated query hits cache without rebuilding.
        let (second, hit) = query(&reg, &cache);
        assert!(hit && Arc::ptr_eq(&first, &second));

        // Adding a non-UAV SRV preserves the cached snapshot.
        set(1, Some((&a, 0x300, 0x400, false)));
        let (third, hit) = query(&reg, &cache);
        assert!(hit && Arc::ptr_eq(&first, &third));

        // Duplicate UAV range on a high/out-of-bounds-like slot index.
        set(u32::MAX, Some((&a, 0x100, 0x200, true)));
        assert_eq!(normalize(&query(&reg, &cache).0), [(pa, 0x100, 0x100)]);

        // Clearing slot 0 keeps the range alive because slot u32::MAX still holds it.
        set(0, None);
        assert_eq!(normalize(&query(&reg, &cache).0), [(pa, 0x100, 0x100)]);

        // Replacing slot u32::MAX with a different copy's UAV swaps the active set.
        set(u32::MAX, Some((&b, 0x500, 0x580, true)));
        assert_eq!(normalize(&query(&reg, &cache).0), [(pb, 0x500, 0x80)]);

        // Empty UAV range is tracked in the index but omitted from widening snapshots.
        set(1_000_000, Some((&a, 0x200, 0x200, true)));
        assert_eq!(normalize(&query(&reg, &cache).0), [(pb, 0x500, 0x80)]);
        assert_eq!(reg.lock().unwrap().unordered.len(), 2);

        // Replacing UAV with SRV removes it from widening snapshots.
        set(u32::MAX, Some((&b, 0x500, 0x580, false)));
        assert_eq!(normalize(&query(&reg, &cache).0), []);

        // Clearing an already-empty slot is a no-op.
        set(999_999, None);
        assert_eq!(normalize(&query(&reg, &cache).0), []);
    }

    #[test]
    fn invalidation_releases_removed_copies_immediately() {
        let a = Arc::new(CopyBuf(0));
        let b = Arc::new(CopyBuf(1));
        let pa = Arc::as_ptr(&a) as usize;
        let pb = Arc::as_ptr(&b) as usize;
        let reg = Mutex::new(ModelRegistry::new(true));
        let cache = SnapshotCache::default();

        {
            let mut g = reg.lock().unwrap();
            g.set(&cache, 0, Some(((pa, 0x100, 0x200, true), Arc::clone(&a))));
            g.set(&cache, 1, Some(((pb, 0x100, 0x200, true), Arc::clone(&b))));
        }
        let snap = query(&reg, &cache).0;
        assert_eq!(snap.len(), 2);
        drop(snap);
        assert_eq!(Arc::strong_count(&b), 5);

        reg.lock().unwrap().set(&cache, 1, None);
        assert_eq!(
            Arc::strong_count(&b),
            1,
            "clearing the last slot must drop cached Arc references immediately"
        );

        reg.lock()
            .unwrap()
            .set(&cache, 0, Some(((pa, 0x100, 0x200, false), Arc::clone(&a))));
        assert_eq!(Arc::strong_count(&a), 3);
    }

    fn run_randomized_equivalence(notify: bool) -> Result<(usize, usize), String> {
        let copies: Vec<Arc<CopyBuf>> = (0..5).map(|i| Arc::new(CopyBuf(i))).collect();
        let mut rng = Rng(0x9E37_79B9_7F4A_7C15);
        let reg = Mutex::new(ModelRegistry::new(notify));
        let cache = SnapshotCache::default();
        let (mut hits, mut misses) = (0usize, 0usize);

        for step in 0..25_000u64 {
            let slot = if step.is_multiple_of(31) {
                // Sparse/high indices beyond typical heap sizes.
                1_000_000 + (rng.next() % 16) as u32
            } else {
                (rng.next() % 256) as u32
            };
            let seen = (!step.is_multiple_of(8)).then(|| {
                let c = (rng.next() % 5) as usize;
                let start = 0x1000 + (rng.next() % 32) * 16;
                let len = if rng.next().is_multiple_of(40) {
                    0
                } else {
                    16 + rng.next() % 64
                };
                let key = (
                    Arc::as_ptr(&copies[c]) as usize,
                    start,
                    start + len,
                    rng.next().is_multiple_of(3),
                );
                (key, Arc::clone(&copies[c]))
            });
            reg.lock().unwrap().set(&cache, slot, seen);
            for _ in 0..(step % 3) {
                let (snap, was_hit) = query(&reg, &cache);
                let expected = reg.lock().unwrap().full_scan();
                if normalize(&snap) != expected {
                    return Err(format!(
                        "step {step}: cached {} ranges, full scan found {}",
                        snap.len(),
                        expected.len()
                    ));
                }
                if was_hit {
                    hits += 1;
                } else {
                    misses += 1;
                }
            }
        }
        Ok((hits, misses))
    }

    #[test]
    fn random_mutations_match_full_scan_and_hit_cache() {
        let (hits, misses) = run_randomized_equivalence(true).unwrap();
        assert!(
            hits > 1_000 && misses > 1_000,
            "expected both cache hits ({hits}) and rebuilds ({misses})"
        );
    }

    #[test]
    fn missing_invalidation_is_detected_by_equivalence_check() {
        assert!(run_randomized_equivalence(false).is_err());
    }

    #[test]
    fn concurrent_readers_observe_linearizable_snapshots() {
        let copy = Arc::new(CopyBuf(0));
        let pc = Arc::as_ptr(&copy) as usize;
        let reg = Mutex::new(ModelRegistry::new(true));
        let cache = SnapshotCache::default();
        const TOTAL: u64 = 2_000;

        std::thread::scope(|scope| {
            scope.spawn(|| {
                for k in 0..TOTAL {
                    let mut g = reg.lock().unwrap();
                    g.set(
                        &cache,
                        k as u32,
                        Some(((pc, 16 * k, 16 * k + 8, true), Arc::clone(&copy))),
                    );
                }
            });
            for _ in 0..3 {
                scope.spawn(|| {
                    let mut last_len = 0usize;
                    for _ in 0..10_000 {
                        let snap = normalize(&query(&reg, &cache).0);
                        let n = snap.len();
                        let expected: Vec<_> =
                            (0..n as u64).map(|k| (pc, (16 * k) as usize, 8)).collect();
                        assert_eq!(snap, expected, "snapshot must be a consistent prefix of {n}");
                        assert!(n >= last_len, "observed older snapshot {n} after {last_len}");
                        last_len = n;
                    }
                });
            }
        });
        assert_eq!(query(&reg, &cache).0.len(), TOTAL as usize);
    }
}
