//! Diagnostics: what the title's streaming engine still waits for.
//!
//! Grand Theft Auto VI loads by asking its streaming engine for objects and
//! then waiting, in a `Sleep` loop, until every module (49 of them: models,
//! textures, scripts and so on) has no request left pending (title+0x1e10090,
//! module 0xffff from title+0x1c82c0f). When one request never finishes, the
//! title stops there for good, with nothing else to show for it in the log:
//! no host call is pending and DirectStorage is idle. On an M1 Max that
//! happened before sign-in and on the loading screen in most launches of
//! 27 November 2026, old builds as well as new.
//!
//! With `VICTORY_STREAM_WATCH=<seconds>`, a thread reads the engine's
//! per-module counters every that many seconds and logs the modules with
//! requests left, so a stall names what it waits for. It reads only (through
//! `ReadProcessMemory`, so a pointer that is not mapped yet reads as nothing)
//! and only in Series 1.0.2.1, whose getter it checks byte for byte first.
#![cfg_attr(not(windows), allow(dead_code))]

/// The engine's getter, `mov rax, [rip + disp]; ret`, in Series 1.0.2.1.
const GETTER_RVA: usize = 0x4d9_eda0;
const GETTER: [u8; 8] = [0x48, 0x8b, 0x05, 0x99, 0xc4, 0x8a, 0x02, 0xc3];
/// The per-module counters in the engine: 49 modules of 0xc0 bytes from
/// +0x48, each with five 64-bit counts 0x18 bytes apart; their totals
/// follow. The wait loop copies these 0x2c48 bytes and waits for the five
/// counts of one module (or of the totals) to be 0.
const STATS: usize = 0x13a80;
const STATS_LEN: usize = 0x2c48;
const MODULES: usize = 49;
const MODULE_AT: usize = 0x48;
const MODULE_LEN: usize = 0xc0;
const TOTALS_AT: usize = 0x24f0;
const COUNT_STRIDE: usize = 0x18;

/// The five counts at `at` of `stats`.
fn counts(stats: &[u8], at: usize) -> [u64; 5] {
    std::array::from_fn(|i| {
        let o = at + i * COUNT_STRIDE;
        u64::from_le_bytes(stats[o..o + 8].try_into().unwrap_or_default())
    })
}

/// The modules with something left in `stats`, as (module, counts), and the
/// totals.
fn pending(stats: &[u8]) -> (Vec<(usize, [u64; 5])>, [u64; 5]) {
    let modules = (0..MODULES)
        .map(|m| (m, counts(stats, MODULE_AT + m * MODULE_LEN)))
        .filter(|(_, c)| c.iter().any(|&n| n != 0))
        .collect();
    (modules, counts(stats, TOTALS_AT))
}

/// Starts the watch for the title mapped at `base`, if `VICTORY_STREAM_WATCH`
/// asks for it.
#[cfg(windows)]
pub(crate) fn start(base: usize) {
    let Some(every) = std::env::var("VICTORY_STREAM_WATCH")
        .ok()
        .and_then(|v| v.trim().parse::<u64>().ok())
        .filter(|&s| s > 0)
    else {
        return;
    };
    let every = std::time::Duration::from_secs(every);
    let spawned = std::thread::Builder::new()
        .name("victory-stream-watch".into())
        .spawn(move || watch(base, every));
    if let Err(e) = spawned {
        tracing::warn!(%e, "VICTORY_STREAM_WATCH: no watch thread");
    }
}

#[cfg(windows)]
fn read(at: usize, out: &mut [u8]) -> bool {
    use windows_sys::Win32::System::Diagnostics::Debug::ReadProcessMemory;
    use windows_sys::Win32::System::Threading::GetCurrentProcess;
    let mut done = 0usize;
    // SAFETY: our own process; the call checks that the range is readable
    // and copies into `out`, which holds `out.len()` bytes.
    let ok = unsafe {
        ReadProcessMemory(
            GetCurrentProcess(),
            at as *const _,
            out.as_mut_ptr().cast(),
            out.len(),
            &mut done,
        )
    };
    ok != 0 && done == out.len()
}

#[cfg(windows)]
fn watch(base: usize, every: std::time::Duration) {
    let mut getter = [0u8; 8];
    if !read(base + GETTER_RVA, &mut getter) || getter != GETTER {
        tracing::info!(
            "VICTORY_STREAM_WATCH: not Series 1.0.2.1's streaming engine getter at +{GETTER_RVA:#x}; no watch"
        );
        return;
    }
    let disp = i32::from_le_bytes([getter[3], getter[4], getter[5], getter[6]]);
    let global = (base + GETTER_RVA + 7).wrapping_add_signed(disp as isize);
    tracing::info!(every_s = every.as_secs(), "VICTORY_STREAM_WATCH: watching the title's streaming requests");
    let mut stats = vec![0u8; STATS_LEN];
    let mut last = None;
    loop {
        std::thread::sleep(every);
        let mut engine = [0u8; 8];
        if !read(global, &mut engine) {
            continue;
        }
        let engine = usize::from_le_bytes(engine);
        if engine == 0 || !read(engine + STATS, &mut stats) {
            continue;
        }
        let (modules, totals) = pending(&stats);
        // Once per change, and again while something stays pending.
        let now = (modules.clone(), totals);
        if last.as_ref() == Some(&now) && modules.is_empty() {
            continue;
        }
        tracing::info!(
            ?totals,
            modules = ?modules,
            same_as_before = last.as_ref() == Some(&now),
            "streaming requests left (module, five counts)"
        );
        last = Some(now);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Counts are read where the title's wait loop reads them, and a module
    /// with all five at 0 is not listed.
    #[test]
    fn modules_with_requests_left_are_listed_with_the_totals() {
        let mut stats = vec![0u8; STATS_LEN];
        let mut put = |at: usize, v: u64| stats[at..at + 8].copy_from_slice(&v.to_le_bytes());
        put(MODULE_AT + 3 * MODULE_LEN + 2 * COUNT_STRIDE, 7);
        put(MODULE_AT + 48 * MODULE_LEN + 4 * COUNT_STRIDE, 1);
        put(TOTALS_AT, 8);
        let (modules, totals) = pending(&stats);
        assert_eq!(modules, vec![(3, [0, 0, 7, 0, 0]), (48, [0, 0, 0, 0, 1])]);
        assert_eq!(totals, [8, 0, 0, 0, 0]);
    }

    // The totals' five counts fit in the bytes read.
    const _: () = assert!(TOTALS_AT + 5 * COUNT_STRIDE <= STATS_LEN);
}
