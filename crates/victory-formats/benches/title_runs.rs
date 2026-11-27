//! What a step that may write anywhere costs a submission's copies:
//! every copy due again (`host.rs`'s `Batch::title_written`, the
//! `Step::Run` that #259 removes), or only what the step's bytes can
//! reach (`Batch::title_copied`'s extents). A world of ~9,000 copies,
//! every list asking every copy whole between the title's ~80 copies
//! per submission, as the runtime's own comments size it. The bytes are
//! counted, not copied: at the testers' 6-7 GB/s they are the copy time.

use std::collections::HashMap;
use std::hint::black_box;
use std::time::Instant;

use victory_formats::extents::Extents;

const COPIES: usize = 9_000;
const REGIONS: u64 = 40;
const WINDOWS: usize = 80;

struct Lcg(u64);

impl Lcg {
    fn next(&mut self) -> u64 {
        self.0 = self
            .0
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        self.0 >> 33
    }
}

/// The world's copies, as `title_copies` builds them.
fn copies(count: usize, regions: u64, lcg: &mut Lcg) -> Vec<(u64, u64)> {
    let mut copies = Vec::with_capacity(count);
    let place = |lcg: &mut Lcg, len: u64| {
        let region = lcg.next() % regions;
        let pages = len.div_ceil(1 << 16);
        let start = (region << 28) + ((lcg.next() % (4096 - pages.min(4096))) << 16);
        (start, len)
    };
    for _ in 0..8 {
        let len = ((lcg.next() % 192) + 64) << 20;
        copies.push(place(lcg, len));
    }
    for _ in 0..200 {
        let len = ((lcg.next() % 15) + 1) << 20;
        copies.push(place(lcg, len));
    }
    for _ in 0..1_000 {
        let len = ((lcg.next() % 768) + 256) << 10;
        copies.push(place(lcg, len));
    }
    while copies.len() < count {
        let len = ((lcg.next() % 64) + 64) << 10;
        copies.push(place(lcg, len));
    }
    copies
}

/// The title's copies into its buffers per submission, 4-64 KiB each.
fn windows(lcg: &mut Lcg) -> Vec<(usize, usize)> {
    (0..WINDOWS)
        .map(|_| {
            let at = (lcg.next() % (REGIONS << 28)) as usize;
            let len = (((lcg.next() % 61) + 4) << 10) as usize;
            (at, at + len)
        })
        .collect()
}

/// victory-gpu's `Ranges`, the three ops the batch uses.
#[derive(Default)]
struct Ranges(Vec<(usize, usize)>);

impl Ranges {
    fn insert(&mut self, from: usize, to: usize, gap: usize) {
        if from >= to {
            return;
        }
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

    fn remove(&mut self, from: usize, to: usize) {
        if from >= to {
            return;
        }
        let first = self.0.partition_point(|&(_, e)| e <= from);
        let last = first + self.0[first..].partition_point(|&(s, _)| s < to);
        if first == last {
            return;
        }
        let (s, e) = (self.0[first].0, self.0[last - 1].1);
        let left = (s < from).then_some((s, from));
        let right = (to < e).then_some((to, e));
        self.0.splice(first..last, left.into_iter().chain(right));
    }

    fn missing(&self, from: usize, to: usize) -> Vec<(usize, usize)> {
        let first = self.0.partition_point(|&(_, end)| end <= from);
        let mut at = from;
        let mut out = Vec::new();
        for &(start, end) in self.0[first..].iter().take_while(|&&(s, _)| s < to) {
            if at < start {
                out.push((at, start));
            }
            at = at.max(end);
        }
        if at < to {
            out.push((at, to));
        }
        out
    }
}

/// One submission. The first window is the overlapping copy: a
/// `Step::Run` today (`clear`), a `Step::Move` as #259 lands it. Every
/// list asks every copy whole after each window. Returns the first
/// window's due bytes, the submission's total, and the bookkeeping.
fn submission(
    clear: bool,
    world: &[(u64, u64)],
    current: &mut HashMap<usize, Ranges>,
    extents: &mut Extents,
    lcg: &mut Lcg,
) -> (u64, u64, u64) {
    let mut first = 0;
    let mut total = 0;
    let started = Instant::now();
    for (i, &(at, end)) in windows(lcg).iter().enumerate() {
        for key in extents.overlapping(at, end) {
            if let Some(have) = current.get_mut(&key) {
                have.remove(at, end);
            }
        }
        if i == 0 && clear {
            current.clear();
            extents.clear();
        }
        // Every list asks every copy whole.
        let mut pending = Vec::new();
        for (key, &(start, len)) in world.iter().enumerate() {
            let empty = Ranges::default();
            let have = current.get(&key).unwrap_or(&empty);
            for (from, to) in have.missing(start as usize, (start + len) as usize) {
                pending.push((key, from, to));
            }
        }
        for (key, from, to) in pending {
            if current.get(&key).is_none() {
                extents.flushed(
                    key,
                    world[key].0 as usize,
                    (world[key].0 + world[key].1) as usize,
                );
            }
            total += (to - from) as u64;
            if i == 0 {
                first += (to - from) as u64;
            }
            current.entry(key).or_default().insert(from, to, 0);
        }
    }
    (first, total, started.elapsed().as_micros() as u64)
}

fn main() {
    let mut lcg = Lcg(0x9E37_79B9_7F4A_7C15);
    let world = copies(COPIES, REGIONS, &mut lcg);
    println!(
        "bench_title_runs_shape {} copies {} regions",
        world.len(),
        REGIONS
    );

    // The first window both ways, untimed: what is due again says which
    // copies stayed current.
    let mut lcg = Lcg(0x9E37_79B9_7F4A_7C15);
    let windows = windows(&mut lcg);
    let (at, end) = windows[0];
    let mut extents = Extents::default();
    for (key, &(start, len)) in world.iter().enumerate() {
        extents.flushed(key, start as usize, (start + len) as usize);
    }
    let current_of = |world: &[(u64, u64)]| {
        world
            .iter()
            .enumerate()
            .map(|(key, &(start, len))| {
                (key, Ranges(vec![(start as usize, (start + len) as usize)]))
            })
            .collect::<HashMap<usize, Ranges>>()
    };
    let mut clear_current = current_of(&world);
    let mut move_current = current_of(&world);
    let reach: Vec<usize> = extents.overlapping(at, end).collect();
    let held = |map: &HashMap<usize, Ranges>| {
        map.values()
            .map(|r| r.0.iter().map(|&(f, t)| t - f).sum::<usize>())
            .sum::<usize>() as u64
    };
    let whole: u64 = world.iter().map(|&(_, len)| len).sum();
    assert_eq!(held(&move_current), whole, "the world starts current");
    for key in extents.overlapping(at, end) {
        move_current.get_mut(&key).unwrap().remove(at, end);
    }
    let moved = whole - held(&move_current);
    let mut lcg = Lcg(0x9E37_79B9_7F4A_7C15);
    let mut clear_extents = Extents::default();
    for (key, &(start, len)) in world.iter().enumerate() {
        clear_extents.flushed(key, start as usize, (start + len) as usize);
    }
    let (first, _, _) = submission(
        true,
        &world,
        &mut clear_current,
        &mut clear_extents,
        &mut lcg,
    );
    assert_eq!(first, whole, "the clear makes every copy due again");
    let mut move_extents = Extents::default();
    for (key, &(start, len)) in world.iter().enumerate() {
        move_extents.flushed(key, start as usize, (start + len) as usize);
    }
    let (move_first, _, _) = submission(
        false,
        &world,
        &mut current_of(&world),
        &mut move_extents,
        &mut Lcg(0x9E37_79B9_7F4A_7C15),
    );
    assert_eq!(move_first, moved, "the move makes due only what it wrote");
    println!(
        "bench_title_runs_reach {} copies {} bytes moved",
        reach.len(),
        moved
    );

    for (name, clear) in [("clear", true), ("move", false)] {
        let mut lcg = Lcg(0x9E37_79B9_7F4A_7C15);
        let mut samples = Vec::new();
        for _ in 0..9 {
            let mut current = current_of(&world);
            let mut extents = Extents::default();
            for (key, &(start, len)) in world.iter().enumerate() {
                extents.flushed(key, start as usize, (start + len) as usize);
            }
            samples.push(submission(
                clear,
                &world,
                &mut current,
                &mut extents,
                &mut lcg,
            ));
            black_box(&current);
        }
        let pick = |i: usize| {
            let mut all: Vec<_> = samples
                .iter()
                .map(|s| match i {
                    0 => s.0,
                    1 => s.1,
                    _ => s.2,
                })
                .collect();
            all.sort_unstable();
            all[all.len() / 2]
        };
        println!(
            "bench_title_runs_first_mb_{name} {:.1}",
            pick(0) as f64 / (1 << 20) as f64
        );
        println!(
            "bench_title_runs_total_mb_{name} {:.1}",
            pick(1) as f64 / (1 << 20) as f64
        );
        println!("bench_title_runs_walk_us_{name} {}", pick(2));
    }
}
