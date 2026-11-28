//! Sets of byte ranges, for the memory a submission has brought up to date
//! and what it still has to (`crate::host`'s batch of refreshes).

#[cfg(any(feature = "diagnostics", test))]
use std::cmp::Reverse;
#[cfg(any(feature = "diagnostics", test))]
use std::collections::BinaryHeap;
#[cfg(any(feature = "diagnostics", test))]
use std::fmt;

use smallvec::SmallVec;

/// Maximum gap between adjacent heap buffer views of the same host copy that
/// are coalesced into one refresh sub-interval (256 KiB, 4 × 64 KiB pages).
/// Larger gaps are kept disjoint so sparse endpoint views on large buffers
/// never flatten into one envelope across untouched or unmapped holes.
#[cfg_attr(not(any(windows, test)), allow(dead_code))]
pub(crate) const MERGE_GAP_BYTES: u64 = 256 * 1024;

/// Maximum number of disjoint sub-intervals kept per `(host copy, uav)` pair.
/// When exceeded, the two adjacent intervals separated by the smallest gap are
/// merged first so the widest holes always stay excluded.
#[cfg_attr(not(any(windows, test)), allow(dead_code))]
pub(crate) const MAX_INTERVALS_PER_COPY: usize = 16;

/// Parses `VICTORY_DISJOINT_SPANS` (opt-in, off by default; `"1"` keeps >256 KiB
/// gaps disjoint instead of collapsing to a single `(min, max)` envelope).
#[cfg_attr(not(any(windows, test)), allow(dead_code))]
pub(crate) fn disjoint_spans_value(value: Option<&std::ffi::OsStr>) -> bool {
    value == Some(std::ffi::OsStr::new("1"))
}

/// Whether disjoint heap buffer refresh sub-intervals are enabled
/// (`VICTORY_DISJOINT_SPANS=1`, off by default).
#[cfg(windows)]
pub(crate) fn disjoint_spans_enabled() -> bool {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| {
        disjoint_spans_value(std::env::var_os("VICTORY_DISJOINT_SPANS").as_deref())
    })
}

/// Coalesces an ascending sequence of half-open `[from, to)` intervals whose
/// gaps are within `gap_threshold`, capping the output to `max_intervals` by
/// merging the smallest adjacent gaps first.
#[cfg_attr(not(any(windows, test)), allow(dead_code))]
pub(crate) fn coalesce_intervals(
    sorted: impl Iterator<Item = (u64, u64)>,
    gap_threshold: u64,
    max_intervals: usize,
) -> Vec<(u64, u64)> {
    let mut out: Vec<(u64, u64)> = Vec::new();
    for (from, to) in sorted {
        if from >= to {
            continue;
        }
        if let Some(last) = out.last_mut()
            && from <= last.1.saturating_add(gap_threshold)
        {
            last.1 = last.1.max(to);
            continue;
        }
        out.push((from, to));
    }
    let cap = max_intervals.max(1);
    while out.len() > cap {
        let mut best_i = 0usize;
        let mut best_gap = u64::MAX;
        for i in 0..out.len() - 1 {
            let gap = out[i + 1].0.saturating_sub(out[i].1);
            if gap < best_gap {
                best_gap = gap;
                best_i = i;
            }
        }
        let next_to = out[best_i + 1].1;
        out[best_i].1 = out[best_i].1.max(next_to);
        out.remove(best_i + 1);
    }
    out
}

/// Byte ranges `[from, to)`, in order and apart: ranges that touch or
/// overlap are one. Most host copies need one range, stored inline.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub(crate) struct Ranges(SmallVec<[(usize, usize); 1]>);

impl Ranges {
    /// The ranges, in order.
    pub(crate) fn iter(&self) -> impl Iterator<Item = (usize, usize)> + '_ {
        self.0.iter().copied()
    }

    /// The ranges as a slice, for a caller that hands them to one call.
    pub(crate) fn as_slice(&self) -> &[(usize, usize)] {
        &self.0
    }

    /// Adds `[from, to)`, joined with the ranges that overlap it or lie
    /// within `gap` bytes of it (the bytes between are then in the set).
    pub(crate) fn insert(&mut self, from: usize, to: usize, gap: usize) {
        if from >= to {
            return;
        }
        // The ranges to join: those ending within `gap` of `from` on, and
        // starting within `gap` of `to` before.
        let first = self
            .0
            .partition_point(|&(_, e)| e.saturating_add(gap) < from);
        let last = first + self.0[first..].partition_point(|&(s, _)| s <= to.saturating_add(gap));
        if first == last {
            self.0.insert(first, (from, to));
            return;
        }
        self.0[first] = (self.0[first].0.min(from), self.0[last - 1].1.max(to));
        self.0.drain(first + 1..last);
    }

    /// Takes `[from, to)` out (a range around it falls in two).
    pub(crate) fn remove(&mut self, from: usize, to: usize) {
        if from >= to {
            return;
        }
        let first = self.0.partition_point(|&(_, e)| e <= from);
        let last = first + self.0[first..].partition_point(|&(s, _)| s < to);
        if first == last {
            return;
        }
        let (s, e) = (self.0[first].0, self.0[last - 1].1);
        let mut kept = first;
        if s < from {
            self.0[kept] = (s, from);
            kept += 1;
        }
        if to < e {
            if kept == last {
                self.0.insert(kept, (to, e));
                return;
            }
            self.0[kept] = (to, e);
            kept += 1;
        }
        self.0.drain(kept..last);
    }

    /// The parts of `[from, to)` not in the set, in order.
    pub(crate) fn missing(
        &self,
        from: usize,
        to: usize,
    ) -> impl Iterator<Item = (usize, usize)> + '_ {
        let first = self.0.partition_point(|&(_, end)| end <= from);
        let mut at = from;
        self.0[first..]
            .iter()
            .copied()
            .take_while(move |&(start, _)| start < to)
            // The requested end closes the last uncovered interval.
            .chain(std::iter::once((to, to)))
            .filter_map(move |(start, end)| {
                let gap = (at, start);
                at = at.max(end);
                (gap.0 < gap.1).then_some(gap)
            })
    }
}

/// How many of a submission's largest refresh ranges are logged.
#[cfg(any(feature = "diagnostics", test))]
const REFRESH_TOP: usize = 8;

/// A bounded summary of the ranges a submission refreshed into host shadows.
/// The ranges are attributed only to their host copy: a batch combines
/// requests from lists, view heaps and page tables, and does not retain a
/// reliable descriptor or view owner for each range.
#[cfg(any(feature = "diagnostics", test))]
#[derive(Default)]
pub(crate) struct RefreshSummary {
    bytes: u64,
    ranges: u64,
    largest: BinaryHeap<Reverse<RefreshedRange>>,
}

/// One refreshed range, ordered largest first and then by title address.
#[cfg(any(feature = "diagnostics", test))]
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
struct RefreshedRange {
    len: u64,
    title: u64,
    host_gpu: u64,
}

#[cfg(any(feature = "diagnostics", test))]
impl Ord for RefreshedRange {
    fn cmp(&self, other: &Self) -> std::cmp::Ordering {
        self.len
            .cmp(&other.len)
            .then_with(|| other.title.cmp(&self.title))
            .then_with(|| other.host_gpu.cmp(&self.host_gpu))
    }
}

#[cfg(any(feature = "diagnostics", test))]
impl PartialOrd for RefreshedRange {
    fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
        Some(self.cmp(other))
    }
}

#[cfg(any(feature = "diagnostics", test))]
impl RefreshSummary {
    /// Counts a range of `len` title bytes and where it lands in host memory.
    pub(crate) fn record(&mut self, title: u64, len: u64, host_gpu: u64) {
        self.bytes = self.bytes.saturating_add(len);
        self.ranges = self.ranges.saturating_add(1);
        if len == 0 {
            return;
        }
        let range = RefreshedRange {
            len,
            title,
            host_gpu,
        };
        if self.largest.len() < REFRESH_TOP {
            self.largest.push(Reverse(range));
        } else if self.largest.peek().is_some_and(|worst| range > worst.0) {
            self.largest.pop();
            self.largest.push(Reverse(range));
        }
    }

    #[cfg(feature = "diagnostics")]
    pub(crate) fn is_empty(&self) -> bool {
        self.ranges == 0
    }
}

#[cfg(any(feature = "diagnostics", test))]
impl fmt::Display for RefreshSummary {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{} ranges, {} bytes", self.ranges, self.bytes)?;
        let mut largest: Vec<_> = self.largest.iter().copied().collect();
        largest.sort_unstable();
        for Reverse(range) in largest {
            write!(
                f,
                "; title {:#x}+{:#x} host_gpu {:#x}",
                range.title, range.len, range.host_gpu
            )?;
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::{REFRESH_TOP, Ranges, RefreshSummary};

    /// Synthetic ranges arrive out of order: totals cover all of them, while
    /// the formatted largest are bounded and ties have a stable order.
    #[test]
    fn refresh_summary_orders_caps_and_formats_ranges() {
        let ranges = [
            (0x1000, 0x20, 0x2000),
            (0x5000, 0x100, 0x9000),
            (0x3000, 0x40, 0x5000),
            (0x8000, 0xc0, 0xd000),
            (0x2000, 0x100, 0x4000),
            (0x6000, 0x10, 0xa000),
            (0x7000, 0x200, 0xb000),
            (0x9000, 0x60, 0xe000),
            (0x4000, 0x80, 0x6000),
            (0xa000, 0x8, 0xf000),
        ];
        let mut summary = RefreshSummary::default();
        for &(title, len, host_gpu) in &ranges {
            summary.record(title, len, host_gpu);
        }
        assert_eq!(summary.ranges, ranges.len() as u64);
        assert_eq!(summary.bytes, ranges.iter().map(|r| r.1).sum());
        assert_eq!(
            summary.to_string(),
            concat!(
                "10 ranges, 1560 bytes",
                "; title 0x7000+0x200 host_gpu 0xb000",
                "; title 0x2000+0x100 host_gpu 0x4000",
                "; title 0x5000+0x100 host_gpu 0x9000",
                "; title 0x8000+0xc0 host_gpu 0xd000",
                "; title 0x4000+0x80 host_gpu 0x6000",
                "; title 0x9000+0x60 host_gpu 0xe000",
                "; title 0x3000+0x40 host_gpu 0x5000",
                "; title 0x1000+0x20 host_gpu 0x2000",
            ),
            "only the top {REFRESH_TOP} are logged, largest first"
        );
        assert!(!summary.to_string().contains("0x6000+0x10"));
        assert!(!summary.to_string().contains("0xa000+0x8"));
    }

    fn set(ranges: &[(usize, usize)]) -> Ranges {
        let mut r = Ranges::default();
        for &(a, b) in ranges {
            r.insert(a, b, 0);
        }
        r
    }

    fn all(r: &Ranges) -> Vec<(usize, usize)> {
        r.iter().collect()
    }

    #[test]
    fn ranges_that_touch_or_overlap_are_one() {
        let mut r = set(&[(10, 20), (30, 40)]);
        assert_eq!(all(&r), [(10, 20), (30, 40)]);
        r.insert(20, 30, 0);
        assert_eq!(all(&r), [(10, 40)], "touching both");
        r.insert(5, 12, 0);
        r.insert(38, 50, 0);
        assert_eq!(all(&r), [(5, 50)], "overlapping");
        r.insert(60, 60, 0);
        assert_eq!(all(&r), [(5, 50)], "nothing added");
    }

    #[test]
    fn a_gap_joins_ranges_close_together() {
        let mut r = Ranges::default();
        r.insert(0, 10, 5);
        r.insert(15, 20, 5);
        assert_eq!(
            all(&r),
            [(0, 20)],
            "5 bytes apart, joined with the bytes between"
        );
        r.insert(26, 30, 5);
        assert_eq!(all(&r), [(0, 20), (26, 30)], "6 bytes apart");
        r.insert(22, 24, 5);
        assert_eq!(all(&r), [(0, 30)], "within 5 of both");
        r.insert(usize::MAX - 10, usize::MAX, usize::MAX);
        assert_eq!(all(&r), [(0, usize::MAX)], "no overflow");
    }

    #[test]
    fn removing_cuts_ranges_and_splits_one_it_is_inside() {
        let mut r = set(&[(0, 100)]);
        r.remove(40, 60);
        assert_eq!(all(&r), [(0, 40), (60, 100)], "split");
        r.remove(0, 10);
        r.remove(90, 200);
        assert_eq!(all(&r), [(10, 40), (60, 90)], "ends cut");
        r.remove(30, 70);
        assert_eq!(
            all(&r),
            [(10, 30), (70, 90)],
            "a range across two, and what is between"
        );
        r.remove(0, 1000);
        assert_eq!(all(&r), [], "all of it");
        let mut r = set(&[(0, 10)]);
        r.remove(5, 5);
        r.remove(7, 3);
        assert_eq!(all(&r), [(0, 10)], "nothing taken out");
    }

    #[test]
    fn missing_lists_the_parts_not_in_the_set() {
        let r = set(&[(10, 20), (30, 40)]);
        let missing = |from, to| r.missing(from, to).collect::<Vec<_>>();
        assert_eq!(missing(0, 50), [(0, 10), (20, 30), (40, 50)]);
        assert_eq!(missing(12, 18), []);
        assert_eq!(missing(15, 35), [(20, 30)]);
        assert_eq!(missing(5, 12), [(5, 10)]);
        assert_eq!(missing(38, 45), [(40, 45)]);
        assert_eq!(missing(50, 60), [(50, 60)]);
        assert_eq!(missing(25, 25), []);
        assert_eq!(
            Ranges::default().missing(3, 9).collect::<Vec<_>>(),
            [(3, 9)]
        );
    }

    /// Every operation against a bitmap, on a long run of random ones.
    #[test]
    fn ranges_match_a_bitmap() {
        const N: usize = 200;
        let mut seed = 0x2545_f491_4f6c_dd1du64;
        let mut next = |n: usize| {
            seed ^= seed << 13;
            seed ^= seed >> 7;
            seed ^= seed << 17;
            (seed % n as u64) as usize
        };
        let mut r = Ranges::default();
        let mut bits = [false; N];
        for step in 0..20_000 {
            let (a, b) = (next(N + 1), next(N + 1));
            let (from, to) = (a.min(b), a.max(b));
            match next(4) {
                0 | 1 => {
                    r.insert(from, to, 0);
                    bits[from..to].fill(true);
                }
                2 => {
                    r.remove(from, to);
                    bits[from..to].fill(false);
                }
                _ => {
                    let mut gaps = [false; N];
                    for (s, e) in r.missing(from, to) {
                        assert!(
                            from <= s && s < e && e <= to,
                            "step {step}: a gap inside the question"
                        );
                        gaps[s..e].fill(true);
                    }
                    for i in 0..N {
                        assert_eq!(
                            gaps[i],
                            (from..to).contains(&i) && !bits[i],
                            "step {step}: byte {i}"
                        );
                    }
                }
            }
            let mut rebuilt = [false; N];
            let mut last_end = None;
            for (s, e) in r.iter() {
                assert!(
                    s < e && last_end.is_none_or(|l| l < s),
                    "step {step}: in order, apart, not empty"
                );
                rebuilt[s..e].fill(true);
                last_end = Some(e);
            }
            assert_eq!(rebuilt, bits, "step {step}");
        }
    }

    #[test]
    fn coalesce_intervals_skips_large_hole_and_respects_gap_and_cap() {
        use super::{
            MAX_INTERVALS_PER_COPY, MERGE_GAP_BYTES, coalesce_intervals, disjoint_spans_value,
        };
        use std::ffi::OsStr;

        assert!(!disjoint_spans_value(None));
        assert!(disjoint_spans_value(Some(OsStr::new("1"))));
        assert!(!disjoint_spans_value(Some(OsStr::new("0"))));

        const BASE: u64 = 0x1000_0000;
        const SIZE: u64 = 160 << 20; // 160 MiB
        let spans = coalesce_intervals(
            [
                (BASE, BASE + 4096),
                (BASE + 32 * 1024, BASE + 36 * 1024),
                (BASE + SIZE - 4096, BASE + SIZE),
            ]
            .into_iter(),
            MERGE_GAP_BYTES,
            MAX_INTERVALS_PER_COPY,
        );
        assert_eq!(
            spans,
            vec![(BASE, BASE + 36 * 1024), (BASE + SIZE - 4096, BASE + SIZE)]
        );

        // Single-envelope fallback (`VICTORY_DISJOINT_SPANS=0` mode).
        let envelope = coalesce_intervals(
            [
                (BASE, BASE + 4096),
                (BASE + 32 * 1024, BASE + 36 * 1024),
                (BASE + SIZE - 4096, BASE + SIZE),
            ]
            .into_iter(),
            u64::MAX,
            1,
        );
        assert_eq!(envelope, vec![(BASE, BASE + SIZE)]);

        // Exact threshold boundary: `MERGE_GAP_BYTES` merges, `MERGE_GAP_BYTES + 1` splits.
        let at_limit = coalesce_intervals(
            [(0, 4096), (4096 + MERGE_GAP_BYTES, 8192 + MERGE_GAP_BYTES)].into_iter(),
            MERGE_GAP_BYTES,
            MAX_INTERVALS_PER_COPY,
        );
        assert_eq!(at_limit, vec![(0, 8192 + MERGE_GAP_BYTES)]);

        let over_limit = coalesce_intervals(
            [
                (0, 4096),
                (4096 + MERGE_GAP_BYTES + 1, 8192 + MERGE_GAP_BYTES + 1),
            ]
            .into_iter(),
            MERGE_GAP_BYTES,
            MAX_INTERVALS_PER_COPY,
        );
        assert_eq!(
            over_limit,
            vec![
                (0, 4096),
                (4096 + MERGE_GAP_BYTES + 1, 8192 + MERGE_GAP_BYTES + 1),
            ]
        );

        // When capped to 2 intervals, the smallest adjacent gap is merged while
        // the giant 100 MiB hole stays excluded.
        let capped = coalesce_intervals(
            [
                (0, 4096),
                (1 << 20, (1 << 20) + 4096),
                (101 << 20, (101 << 20) + 4096),
            ]
            .into_iter(),
            MERGE_GAP_BYTES,
            2,
        );
        assert_eq!(
            capped,
            vec![(0, (1 << 20) + 4096), (101 << 20, (101 << 20) + 4096)]
        );
    }
}
