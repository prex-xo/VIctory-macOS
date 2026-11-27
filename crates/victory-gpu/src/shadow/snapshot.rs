//! Save only the bytes newly covered by each declared GPU write.
//! Spans keep their title-address order; the backing bytes stay in capture order.

use super::{give_back, pooled};

#[derive(Default)]
pub(super) struct Snapshot {
    bytes: Vec<u8>,
    /// The last captured size guides the first pool checkout next submission.
    last_size: usize,
    /// Disjoint spans in address order, pointing into the capture buffer.
    spans: Vec<SnapshotSpan>,
    /// The 64 KiB pieces already compared, by title address.
    pub(super) compared: std::collections::HashSet<usize>,
}

struct SnapshotSpan {
    start: usize,
    end: usize,
    offset: usize,
}

impl Snapshot {
    #[cfg(test)]
    pub(super) fn len(&self) -> usize {
        self.bytes.len()
    }

    /// Keep newly covered bytes, without rebuilding the previously captured ranges.
    /// Return the bytes copied, including moving saved bytes when the backing buffer
    /// grows: the Mac snapshot diagnostics measure that traffic too.
    pub(super) fn capture(&mut self, start: usize, host: &[u8]) -> usize {
        let end = start + host.len();
        let before = self.bytes.len();
        let mut at = start;
        let mut i = self.spans.partition_point(|span| span.end <= at);
        let mut moved = 0;
        if host.len() > self.bytes.capacity() - before {
            let covered: usize = self.spans[i..]
                .iter()
                .take_while(|span| span.start < end)
                .map(|span| span.end.min(end) - span.start.max(start))
                .sum();
            moved = self.reserve(before + (host.len() - covered));
        }
        while at < end {
            if let Some(span) = self.spans.get(i)
                && span.start <= at
            {
                at = span.end;
                i += 1;
                continue;
            }
            let next = self.spans.get(i).map_or(end, |span| end.min(span.start));
            let offset = self.bytes.len();
            self.bytes
                .extend_from_slice(&host[at - start..next - start]);
            // Consecutive captures in address order can still use one span.
            if let Some(previous) = i.checked_sub(1).and_then(|j| self.spans.get_mut(j))
                && previous.end == at
                && previous.offset + (previous.end - previous.start) == offset
            {
                previous.end = next;
            } else {
                self.spans.insert(
                    i,
                    SnapshotSpan {
                        start: at,
                        end: next,
                        offset,
                    },
                );
                i += 1;
            }
            at = next;
        }
        self.bytes.len() - before + moved
    }

    fn reserve(&mut self, required: usize) -> usize {
        let capacity = self.bytes.capacity();
        if required <= capacity {
            return 0;
        }
        // Grow by enough to amortize copying, but use a smaller warm
        // buffer if it can hold the new snapshot.
        let growth = required.max(capacity.saturating_mul(2));
        let preferred = if capacity == 0 {
            growth.max(self.last_size)
        } else {
            growth
        };
        let mut grown = pooled(required, preferred).unwrap_or_else(|| Vec::with_capacity(growth));
        grown.extend_from_slice(&self.bytes);
        let old = std::mem::replace(&mut self.bytes, grown);
        let moved = old.len();
        if capacity != 0 {
            give_back(old);
        }
        moved
    }

    pub(super) fn covers(&self, start: usize, end: usize) -> bool {
        let mut at = start;
        for span in &self.spans[self.spans.partition_point(|span| span.end <= start)..] {
            if span.start > at {
                break;
            }
            at = span.end;
            if at >= end {
                return true;
            }
        }
        false
    }

    /// The original bytes of a changed piece inside a range checked by
    /// `covers`. Assemble fragmented bytes only if they differ from the host.
    pub(super) fn changed_piece<'a>(
        &'a self,
        start: usize,
        host: &[u8],
        scratch: &'a mut Vec<u8>,
    ) -> Option<&'a [u8]> {
        let end = start + host.len();
        debug_assert!(self.covers(start, end));
        let first = self.spans.partition_point(|span| span.end <= start);
        let span = &self.spans[first];
        if end <= span.end {
            let offset = span.offset + (start - span.start);
            let old = &self.bytes[offset..offset + host.len()];
            return (host != old).then_some(old);
        }
        let fragments = || {
            self.spans[first..]
                .iter()
                .take_while(|span| span.start < end)
                .map(|span| {
                    let from = span.start.max(start);
                    let to = span.end.min(end);
                    let offset = span.offset + (from - span.start);
                    (
                        &self.bytes[offset..offset + to - from],
                        from - start..to - start,
                    )
                })
        };
        if fragments().all(|(old, range)| old == &host[range]) {
            return None;
        }
        scratch.clear();
        for (old, _) in fragments() {
            scratch.extend_from_slice(old);
        }
        Some(scratch)
    }

    /// A refresh before the GPU runs changes the baseline for the captured bytes.
    pub(super) fn refresh(&mut self, start: usize, host: &[u8]) {
        let end = start + host.len();
        let first = self.spans.partition_point(|span| span.end <= start);
        for span in self.spans[first..]
            .iter()
            .take_while(|span| span.start < end)
        {
            let from = span.start.max(start);
            let to = span.end.min(end);
            let offset = span.offset + (from - span.start);
            self.bytes[offset..offset + to - from].copy_from_slice(&host[from - start..to - start]);
        }
    }

    pub(super) fn clear(&mut self) {
        self.compared.clear();
        self.spans.clear();
        if !self.bytes.is_empty() {
            self.last_size = self.bytes.len();
            give_back(std::mem::take(&mut self.bytes));
        }
    }
}

#[cfg(test)]
mod tests {
    use super::Snapshot;

    #[test]
    fn overlapping_snapshots_keep_old_bytes_until_a_refresh() {
        let mut snapshot = Snapshot {
            bytes: Vec::with_capacity(7),
            ..Snapshot::default()
        };
        let mut scratch = Vec::new();
        assert_eq!(snapshot.capture(4, &[14, 15]), 2);
        let data = snapshot.bytes.as_ptr();
        assert_eq!(snapshot.capture(4, &[0, 0]), 0);
        assert_eq!(
            snapshot.bytes.as_ptr(),
            data,
            "covered bytes need no growth"
        );
        assert_eq!(snapshot.capture(0, &[10, 11]), 2);
        assert!(!snapshot.covers(0, 6), "there is a gap");
        assert_eq!(snapshot.capture(1, &[21, 12, 13, 24, 25, 16]), 3);
        assert_eq!(snapshot.capture(0, &[0; 7]), 0);
        assert!(snapshot.covers(0, 7));
        assert!(!snapshot.covers(0, 8));
        let original = [10, 11, 12, 13, 14, 15, 16];
        assert!(snapshot.changed_piece(0, &original, &mut scratch).is_none());
        assert!(scratch.is_empty());
        assert_eq!(
            snapshot.changed_piece(0, &[0; 7], &mut scratch).unwrap(),
            original
        );
        snapshot.refresh(1, &[21, 22, 23, 24, 25]);
        assert_eq!(
            snapshot.changed_piece(0, &[0; 7], &mut scratch).unwrap(),
            [10, 21, 22, 23, 24, 25, 16]
        );
        assert_eq!(snapshot.changed_piece(5, &[0], &mut scratch).unwrap(), [25]);
        snapshot.clear();
        assert!(!snapshot.covers(0, 1));
    }

    #[test]
    fn snapshot_traffic_counts_moving_the_saved_bytes_only_on_growth() {
        let mut snapshot = Snapshot {
            bytes: Vec::with_capacity(2),
            ..Snapshot::default()
        };
        assert_eq!(snapshot.capture(0, &[1, 2]), 2);
        assert_eq!(snapshot.capture(1, &[9, 3, 4]), 4, "two new plus two moved");
        assert_eq!(snapshot.capture(0, &[9; 4]), 0);
        let mut scratch = Vec::new();
        assert!(
            snapshot
                .changed_piece(0, &[1, 2, 3, 4], &mut scratch)
                .is_none()
        );
    }
}
