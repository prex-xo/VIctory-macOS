//! Capture time for repeated declarations on one 64 MiB shadow. Run this
//! test alone, in release, with `--ignored --nocapture --test-threads=1`.

use super::{find, register, unregister};
use crate::watch::{PAGE, tests};
use std::time::Instant;

#[test]
#[ignore = "times snapshot preparation on a mapped D3D12 buffer"]
fn growing_write_ranges() {
    const MIB: usize = 1 << 20;
    const SIZE: usize = 64 * MIB;
    let p = tests::pool(SIZE / PAGE);
    register(p, p as u64, SIZE as u64);
    let copy = find(p as u64, SIZE as u64).expect("host buffer");
    assert!(!copy.aliased, "this measures snapshots, not zero-copy");
    // Gives the host mapping known bytes and pays first-touch costs before timing.
    copy.update(p, SIZE);

    let growing: Vec<_> = (0..=16)
        .map(|i| ((16 - i) * MIB, (32 + 2 * i) * MIB))
        .collect();
    let disjoint = [(0, 16 * MIB), (32 * MIB, 16 * MIB)];
    for (name, ranges) in [
        ("one", &[(0, SIZE)][..]),
        ("growing", growing.as_slice()),
        ("disjoint", &disjoint[..]),
    ] {
        let mut times = Vec::new();
        for round in 0..11 {
            let start = Instant::now();
            for &(offset, bytes) in ranges {
                copy.writing(p + offset, bytes, false);
            }
            let elapsed = start.elapsed();
            // Copy-back and returning buffers to the pool are outside the timer.
            // A changed byte must reach title memory on each iteration.
            let at = p + ranges[0].0 + 5;
            let value = round + 1;
            copy.write_host(at as u64, &[value]);
            for &(offset, bytes) in ranges {
                copy.copy_back(p + offset, bytes);
            }
            // SAFETY: the test owns this pool; copy-back has returned.
            assert_eq!(unsafe { (at as *const u8).read_volatile() }, value);
            if round >= 2 {
                times.push(elapsed.as_micros());
            }
        }
        times.sort_unstable();
        eprintln!("{name}: median {} us (9 warm captures)", times[4]);
    }
    unregister(p, p as u64);
    drop(copy);
    tests::free(p);
}
