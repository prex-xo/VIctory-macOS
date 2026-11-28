//! Title copies whose bytes are in place already (`VICTORY_SAME_COPIES`,
//! off by default).
//!
//! A title copy between buffers (`CopyBufferRegion` on title memory) runs on
//! the CPU at submission ([`super::copy_mapped`]). Its write marks the
//! destination changed in every tracker holding it, so the destination's
//! host copies are refreshed again afterwards, whole 64 KiB pages. A title
//! that copies the same bytes each frame (a constant block into a buffer
//! that already holds it) pays the copy and those refreshes every time.
//!
//! With `probe`, each title copy is compared first: the source's bytes
//! against the destination's, and against every host copy of the
//! destination. All the same means the copy would change nothing anywhere:
//! the title memory it writes holds those bytes, and the refresh it causes
//! would put the same bytes into the host copies. `probe` only counts them
//! (the frame report's preparing line); `skip` leaves them out: no write, no
//! trackers marked, no refreshes due.
//!
//! The comparison reads through the kernel ([`super::read_into`]), as the
//! copy does, so an unmapped page fails it (and the copy runs as before)
//! instead of faulting. Copies of memory shown at several addresses, copies
//! whose ranges overlap, and copies larger than [`MAX`] are not compared.

use std::sync::atomic::{AtomicU64, Ordering};

/// What `VICTORY_SAME_COPIES` asks for.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub(crate) enum Mode {
    /// Not compared (the default).
    Off,
    /// Compared and counted, still copied.
    Probe,
    /// Compared, and left out when the bytes are in place.
    Skip,
}

/// The largest copy compared: two buffers of it are kept per thread.
pub(crate) const MAX: usize = 16 << 20;

#[cfg(test)]
thread_local! {
    /// The mode on this test's thread: 1 probe, 2 skip, else off (other
    /// tests' copies, on their threads, are not compared).
    pub(crate) static TEST_MODE: std::cell::Cell<u8> = const { std::cell::Cell::new(0) };
}

/// `VICTORY_SAME_COPIES`: `probe` or `skip`; anything else is [`Mode::Off`].
pub(crate) fn mode() -> Mode {
    #[cfg(test)]
    match TEST_MODE.with(std::cell::Cell::get) {
        1 => return Mode::Probe,
        2 => return Mode::Skip,
        _ => return Mode::Off,
    }
    #[allow(unreachable_code)]
    {
        static MODE: std::sync::OnceLock<Mode> = std::sync::OnceLock::new();
        *MODE.get_or_init(|| {
            let mode = parse(std::env::var("VICTORY_SAME_COPIES").ok().as_deref());
            if mode != Mode::Off {
                tracing::info!(?mode, "title copies compared before they run (VICTORY_SAME_COPIES)");
            }
            mode
        })
    }
}

fn parse(value: Option<&str>) -> Mode {
    match value {
        Some("probe") => Mode::Probe,
        Some("skip") => Mode::Skip,
        _ => Mode::Off,
    }
}

/// Since the last frame report: copies compared and their bytes, then those
/// already in place and their bytes, then the microseconds comparing took.
static COUNTS: [AtomicU64; 5] = [const { AtomicU64::new(0) }; 5];

/// [`COUNTS`] since the last call.
pub(crate) fn take() -> [u64; 5] {
    COUNTS.each_ref().map(|c| c.swap(0, Ordering::Relaxed))
}

thread_local! {
    /// The source's and the destination's bytes, reused.
    static SCRATCH: std::cell::RefCell<(Vec<u8>, Vec<u8>)> =
        const { std::cell::RefCell::new((Vec::new(), Vec::new())) };
}

/// Whether copying `len` bytes of title memory from `src` to `dst` would
/// change nothing: the destination and all its host copies hold the
/// source's bytes. Counted; `false` when not compared.
pub(super) fn in_place(dst: usize, src: usize, len: usize) -> bool {
    if len == 0
        || len > MAX
        || (src < dst.saturating_add(len) && dst < src.saturating_add(len))
        || crate::watch::aliased(dst, len)
        || crate::watch::aliased(src, len)
    {
        return false;
    }
    let started = std::time::Instant::now();
    let same = SCRATCH.with(|scratch| {
        let (from, to) = &mut *scratch.borrow_mut();
        from.resize(len, 0);
        // The host copies first: compared in memory, they tell most copies
        // that change something without a second read through the kernel.
        if !super::read_into(src as *const u8, from)
            || !crate::shadow::over(dst, dst + len).iter().all(|copy| copy.holds(dst, from))
        {
            return false;
        }
        to.resize(len, 0);
        super::read_into(dst as *const u8, to) && from == to
    });
    COUNTS[0].fetch_add(1, Ordering::Relaxed);
    COUNTS[1].fetch_add(len as u64, Ordering::Relaxed);
    if same {
        COUNTS[2].fetch_add(1, Ordering::Relaxed);
        COUNTS[3].fetch_add(len as u64, Ordering::Relaxed);
    }
    COUNTS[4].fetch_add(started.elapsed().as_micros() as u64, Ordering::Relaxed);
    same
}

#[cfg(test)]
mod tests {
    use super::{Mode, parse};

    #[test]
    fn the_switch_takes_probe_and_skip_only() {
        assert_eq!(parse(None), Mode::Off);
        assert_eq!(parse(Some("1")), Mode::Off);
        assert_eq!(parse(Some("probe")), Mode::Probe);
        assert_eq!(parse(Some("skip")), Mode::Skip);
    }
}
