//! Time of the widened-write bookkeeping at barriers, in the world's shape:
//! lists whose heap-writing draws name a heap of many UAV ranges, so each
//! list's history grows to ~90,000 writes over its barriers. Run this test
//! alone, in release, with `--ignored --nocapture --test-threads=1`.

use super::*;
use crate::backend::HostGpu;

#[test]
#[ignore = "times widened bookkeeping at barriers on host lists"]
fn barriers_over_a_long_widened_history() {
    let Some(host) = host() else {
        return; // no host GPU here
    };
    let new = |start: u64, len: u64| {
        let (buffer, data) = host.shared_buffer(len).unwrap();
        // SAFETY: a live buffer.
        let gpu = unsafe { buffer.GetGPUVirtualAddress() };
        Shadow::new(buffer, data, gpu, start, len)
    };
    // A widened view, so barriers widen; the heap's views and the UAVs
    // write a copy it does not overlap (no widening work, only the
    // bookkeeping).
    let parent = new(0x1000, 48);
    let _widened = parent.widened(0x1000, 48, 1).expect("a widened view");
    let written = new(0x100_0000, 1 << 20);
    const RANGES: usize = 15_000;
    let viewed = Arc::new(crate::bindless::Viewed::default());
    for i in 0..RANGES {
        viewed.add_unordered(&written, 0x100_0000 + 64 * i as u64, 64);
    }
    // Ten lists of six barriers each ("a frame"), each barrier after a
    // heap-writing draw and a hundred UAV writes.
    let lists: Vec<HostList> = (0..10).map(|_| host.new_list().unwrap()).collect();
    let mut times = Vec::new();
    for round in 0..7 {
        let started = std::time::Instant::now();
        for list in &lists {
            for _ in 0..6 {
                list.mark_widened_heap(&viewed);
                for i in 0..100 {
                    list.mark_widened_write(&written, 0x100_0000 + 16 * i, 16);
                }
                list.sync_widened();
            }
        }
        let elapsed = started.elapsed();
        for list in &lists {
            list.close();
            list.reset();
        }
        if round >= 2 {
            times.push(elapsed.as_micros());
        }
    }
    times.sort_unstable();
    eprintln!(
        "60 barriers over histories up to {} writes: median {} us (5 warm rounds, {:?})",
        6 * (RANGES + 100),
        times[2],
        times
    );
}
