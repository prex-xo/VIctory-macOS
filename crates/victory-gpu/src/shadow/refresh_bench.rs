//! What a refresh costs per 64 KiB page the title wrote, with tracking on:
//! the write that faults (title side), the refresh (`Shadow::update`), and
//! a plain copy of the same bytes. Run alone, in release:
//! `cargo test --release -p victory-gpu --lib refresh_cost -- --ignored --nocapture`.

use super::{find, register, unregister};
use crate::watch::{PAGE, tests};
use std::time::Instant;

#[test]
#[ignore = "times refreshes of written pages on a tracked buffer"]
fn refresh_cost_per_written_page() {
    const PAGES: usize = 1024;
    let size = PAGES * PAGE;
    let p = tests::pool(PAGES);
    register(p, p as u64, size as u64);
    let copy = find(p as u64, size as u64).expect("host buffer");
    assert!(!copy.aliased, "this measures copies, not zero-copy");
    copy.update(p, size);
    let mut scratch = vec![0u8; PAGE];
    for (name, stride, same) in [
        ("every page, new bytes", 1, false),
        ("every page, same bytes", 1, true),
        ("every 8th page, new bytes", 8, false),
    ] {
        let mut write_us = Vec::new();
        let mut refresh_us = Vec::new();
        let mut copied = 0;
        for round in 0..7u8 {
            let started = Instant::now();
            for page in (0..PAGES).step_by(stride) {
                let at = (p + page * PAGE + 64) as *mut u8;
                // SAFETY: inside the test's pool.
                unsafe { at.write_volatile(if same { 0 } else { round + 1 }) };
            }
            write_us.push(started.elapsed().as_micros());
            let started = Instant::now();
            copied = copy.update(p, size);
            refresh_us.push(started.elapsed().as_micros());
        }
        let started = Instant::now();
        for page in (0..PAGES).step_by(stride) {
            // SAFETY: inside the test's pool.
            let from = unsafe { std::slice::from_raw_parts((p + page * PAGE) as *const u8, PAGE) };
            scratch.copy_from_slice(std::hint::black_box(from));
        }
        let memcpy = started.elapsed().as_secs_f64() * 1e6;
        write_us.sort_unstable();
        refresh_us.sort_unstable();
        let pages = PAGES.div_ceil(stride) as f64;
        eprintln!(
            "{name}: {pages} pages, writes {:.1} us/page (faults), refresh {:.1} us/page ({} MB copied), memcpy {:.1} us/page",
            write_us[3] as f64 / pages,
            refresh_us[3] as f64 / pages,
            copied >> 20,
            memcpy / pages,
        );
    }
    unregister(p, p as u64);
    drop(copy);
    tests::free(p);
}
