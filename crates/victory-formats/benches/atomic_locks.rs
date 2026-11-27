//! Time the emulated 64-bit image atomic's forms as the wave's own
//! instruction stream, on CPU: the turn loop `locked_atomic64` emits
//! today; the rounds-then-turns form PR #257 lands (four capped rounds
//! of one attempt per lane not done, then today's turns for whoever is
//! left); and the umax form, which a glc-clear `image_atomic_umax`
//! enables — the wave first reduces each texel's lanes to their
//! maximum key, in registers, one representative a group, and only the
//! representatives lock. The lock words (`ATOMIC_LOCKS` of them,
//! hashed from the coordinates), the spin's compare-exchange and the
//! fenced read-modify-write are the emitted ones (victory-shader
//! `lower.rs`, "Atomic locks"); what the title uses them for is one
//! operation, per #240's disassembly of its shaders: an unsigned
//! maximum of 64-bit keys into R32G32_UINT texels, all in alpha
//! pre-pass pixel shaders, all lanes of a wave per instruction.
//!
//! Four shapes: the three the runtime's own test writes
//! (`image_atomics_of_64_bits`, "as the alpha prepasses use them") —
//! every lane into one texel (one lock word, a turn each); each lane
//! its own texel (a wave of distinct locks); and the test's program,
//! one colliding and one own umax per lane — and between them a
//! quarter-res prepass, four lanes per texel.
//!
//! What the model can and cannot say: the barrier each read-modify-
//! write runs between is a sequentially-consistent `fence` here — a
//! few cycles on a CPU, where the GPU's threadgroup barrier serializes
//! memory traffic — so each form's barrier count is printed beside its
//! ns, and the ns time the op streams themselves (the lock attempts,
//! the fenced loads and stores, the releases, the loop) with no
//! padding: per-value register shuffling is the same in both forms and
//! left out, and the lane masks are index lists the way the wave keeps
//! them in registers. The turn lane's spin under another wave's lock
//! is not modeled (no other wave runs here, and it is the same in both
//! forms); a round's winners are the lowest unfinished lanes per lock
//! word, which is what the announced turn priority gives when the
//! wave's own lanes never hold each other's locks.
//!
//! Every form is asserted to leave the same texels, the turn and round
//! forms to return the same old values lane by lane, and every lock
//! word is freed, against a plain sequential umax — the exactness bar
//! issue #240 set for any faster form.

use std::hint::black_box;
use std::sync::atomic::{fence, AtomicU32, AtomicU64, Ordering};
use std::time::Instant;

/// A wave, as the title's pixel shaders run it.
const LANES: usize = 32;
/// Lock words, as `emit` sizes them: `ATOMIC_LOCKS` dwords.
const LOCKS: usize = 1 << 16;
/// The rounds PR #257 tries before the turns: `ATOMIC_LOCK_ROUNDS`.
const ROUNDS: usize = 4;
/// The texture, 64 by 4 R32G32_UINT texels.
const TEXELS: usize = 64 * 4;
/// Wave visits of one program a sample times.
const VISITS: u32 = 4_096;

/// A deterministic generator, so every run sees the same keys.
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

/// The lock word of texel `(x, y)`: the hash `locked_atomic64` makes
/// of the coordinates, `x` first, masked to the lock words.
fn lock_of(x: u32, y: u32) -> usize {
    let mut hash = 0;
    for (coord, mix) in [(x, 0x9E37_79B1), (y, 0x85EB_CA77)] {
        hash ^= coord.wrapping_mul(mix);
        hash ^= hash >> 15;
    }
    (hash & (LOCKS as u32 - 1)) as usize
}

/// Where texel `(x, y)` sits in the texture.
fn texel_of(x: u32, y: u32) -> usize {
    y as usize * 64 + x as usize
}

/// One lane's umax: the texel it aims at, the key it offers, and the
/// old value the instruction returns.
#[derive(Clone)]
struct Lane {
    x: u32,
    y: u32,
    value: u64,
    old: u64,
}

/// The state one shader dispatch works on: the lock words (the tail of
/// the root UAV `emit` binds for them) and the texture's texels.
struct Memory {
    locks: Vec<AtomicU32>,
    texels: Vec<AtomicU64>,
}

impl Memory {
    fn zeroed() -> Self {
        Self {
            locks: (0..LOCKS).map(|_| AtomicU32::new(0)).collect(),
            texels: (0..TEXELS).map(|_| AtomicU64::new(0)).collect(),
        }
    }

    /// Every lock word free again, as the runtime's test asserts of a
    /// shader that ends.
    fn freed(&self) -> bool {
        self.locks.iter().all(|w| w.load(Ordering::Relaxed) == 0)
    }

    fn texels_eq(&self, other: &Memory) -> bool {
        self.texels
            .iter()
            .zip(&other.texels)
            .all(|(a, b)| a.load(Ordering::Relaxed) == b.load(Ordering::Relaxed))
    }
}

/// What one form's op stream costs in counted terms.
#[derive(Default)]
struct Count {
    /// Barrier calls, two per fenced read-modify-write: the GPU's
    /// serializing term, near-free on CPU.
    barriers: usize,
    /// Lock attempts: compare-exchanges on the lock words.
    attempts: usize,
    /// The loop's iterations: a turn each, or a round.
    rounds: usize,
    /// Wave operations the umax form's reduction adds: two lane reads,
    /// one ballot and two active maxima a group, plus the first ballot.
    wave_ops: usize,
}

/// The form `locked_atomic64` emits: the wave runs a loop over lane
/// indices, and in turn `i` lane `i` alone takes its texel's lock
/// word, reads, writes the maximum and frees it — two barriers each —
/// whatever the other lanes' locks.
fn turn(m: &Memory, lanes: &mut [Lane], count: &mut Count) {
    let mut words = [0usize; LANES];
    for (i, lane) in lanes.iter().enumerate() {
        words[i] = lock_of(lane.x, lane.y);
    }
    for i in 0..lanes.len() {
        count.rounds += 1;
        // Spin: one compare-exchange, which an uncontended word takes.
        let word = &m.locks[words[i]];
        let won = word
            .compare_exchange(0, 1, Ordering::Relaxed, Ordering::Relaxed)
            .is_ok();
        count.attempts += 1;
        // Held: fence, read, write, fence.
        fence(Ordering::SeqCst);
        count.barriers += 1;
        let at = texel_of(lanes[i].x, lanes[i].y);
        let old = m.texels[at].load(Ordering::Relaxed);
        m.texels[at].store(old.max(lanes[i].value), Ordering::Relaxed);
        lanes[i].old = old;
        fence(Ordering::SeqCst);
        count.barriers += 1;
        // Frees it only if taken: an `and` with all ones changes
        // nothing.
        let keep = if won { 0 } else { u32::MAX };
        word.fetch_and(keep, Ordering::Relaxed);
    }
}

/// The rounds PR #257 lands before the turns: per round, every lane
/// that has not finished makes one attempt at its lock word and the
/// winners read-modify-write in that same round — one fenced region,
/// two barriers, not two per lane. Nobody spins in a round, and the
/// four rounds run even when every lane is done (no all-done exit
/// without a wave operation). A lane the rounds leave takes today's
/// turn, one uncontended attempt; a wave whose lanes have distinct
/// locks finishes in the first round and skips the turns.
fn round(m: &Memory, lanes: &mut [Lane], count: &mut Count, start_done: &[bool; LANES]) {
    let mut words = [0usize; LANES];
    for (i, lane) in lanes.iter().enumerate() {
        words[i] = lock_of(lane.x, lane.y);
    }
    // The lanes not done, and this round's winners, as the masks the
    // wave keeps in registers. `start_done` marks the lanes done from
    // the start, as the rounds' helper lanes are.
    let mut done = *start_done;
    let mut rest: [usize; LANES] = core::array::from_fn(|i| i);
    let mut rest_n = 0;
    for (i, d) in done.iter().enumerate() {
        if !d {
            rest[rest_n] = i;
            rest_n += 1;
        }
    }
    let mut won: [usize; LANES] = [0; LANES];
    for _ in 0..ROUNDS {
        count.rounds += 1;
        // Every lane not done attempts once, in lane order.
        let mut won_n = 0;
        let mut still: [usize; LANES] = [0; LANES];
        let mut still_n = 0;
        for &i in &rest[..rest_n] {
            let got = m.locks[words[i]]
                .compare_exchange(0, 1, Ordering::Relaxed, Ordering::Relaxed)
                .is_ok();
            count.attempts += 1;
            if got {
                won[won_n] = i;
                won_n += 1;
            } else {
                still[still_n] = i;
                still_n += 1;
            }
        }
        rest = still;
        rest_n = still_n;
        if won_n == 0 {
            continue;
        }
        // The winners' read-modify-write, one region.
        fence(Ordering::SeqCst);
        count.barriers += 1;
        for &i in &won[..won_n] {
            let at = texel_of(lanes[i].x, lanes[i].y);
            let old = m.texels[at].load(Ordering::Relaxed);
            m.texels[at].store(old.max(lanes[i].value), Ordering::Relaxed);
            lanes[i].old = old;
        }
        fence(Ordering::SeqCst);
        count.barriers += 1;
        // The winners free their words and leave.
        for &i in &won[..won_n] {
            m.locks[words[i]].fetch_and(0, Ordering::Relaxed);
            done[i] = true;
        }
    }
    // Today's turns for whoever the rounds left; the pass runs whole,
    // its finished lanes passing through the head and the latch.
    if rest_n > 0 {
        for i in 0..lanes.len() {
            count.rounds += 1;
            if done[i] {
                continue;
            }
            // Spin: one compare-exchange, which an uncontended word
            // takes.
            let word = &m.locks[words[i]];
            let won = word
                .compare_exchange(0, 1, Ordering::Relaxed, Ordering::Relaxed)
                .is_ok();
            count.attempts += 1;
            // Held: fence, read, write, fence.
            fence(Ordering::SeqCst);
            count.barriers += 1;
            let at = texel_of(lanes[i].x, lanes[i].y);
            let old = m.texels[at].load(Ordering::Relaxed);
            m.texels[at].store(old.max(lanes[i].value), Ordering::Relaxed);
            lanes[i].old = old;
            fence(Ordering::SeqCst);
            count.barriers += 1;
            // Frees it only if taken: an `and` with all ones changes
            // nothing.
            let keep = if won { 0 } else { u32::MAX };
            word.fetch_and(keep, Ordering::Relaxed);
        }
    }
}

/// The umax form: glc clear, so nothing reads the old value and max is
/// associative — the wave first reduces each texel's lanes to their
/// maximum key, in registers, one representative a group, and only the
/// representatives run the rounds and the turns. The reduction is one
/// pass per group: the lowest lane leads, the group's maximum key
/// becomes the lead's, the members are done from the start.
fn umax(m: &Memory, lanes: &mut [Lane], count: &mut Count) {
    let mut start = [false; LANES];
    count.wave_ops += 1;
    for lead in 0..lanes.len() {
        if start[lead] {
            continue;
        }
        count.wave_ops += 5;
        let mut hi = lanes[lead].value >> 32;
        let mut lo = lanes[lead].value & u32::MAX as u64;
        for member in lead + 1..lanes.len() {
            if start[member] || (lanes[member].x, lanes[member].y) != (lanes[lead].x, lanes[lead].y) {
                continue;
            }
            let v = lanes[member].value;
            if v >> 32 > hi {
                (hi, lo) = (v >> 32, v & u32::MAX as u64);
            } else if v >> 32 == hi && v & u32::MAX as u64 > lo {
                lo = v & u32::MAX as u64;
            }
            start[member] = true;
        }
        lanes[lead].value = lo | hi << 32;
    }
    round(m, lanes, count, &start);
}

/// A plain umax in lane order: the old values and texels both forms
/// must leave, exactly.
fn reference(m: &Memory, lanes: &mut [Lane]) {
    for lane in lanes.iter_mut() {
        let at = texel_of(lane.x, lane.y);
        let old = m.texels[at].load(Ordering::Relaxed);
        m.texels[at].store(old.max(lane.value), Ordering::Relaxed);
        lane.old = old;
    }
}

/// One wave's shapes.
#[derive(Clone, Copy)]
enum Shape {
    /// Every lane into one texel: one lock word, a turn each.
    One,
    /// Four lanes per texel, as a quarter-res prepass makes.
    Pool,
    /// Every lane its own texel: distinct lock words.
    Own,
    /// The runtime's test's program: one colliding and one own umax.
    Mixed,
}

/// The shapes' programs: for each instruction of the wave's program,
/// the texel every lane umaxes into.
fn programs(shape: Shape) -> Vec<Vec<(u32, u32)>> {
    let by_lane = |pick: fn(usize) -> (u32, u32)| -> Vec<(u32, u32)> {
        (0..LANES).map(pick).collect()
    };
    match shape {
        Shape::One => vec![by_lane(|_| (0, 0))],
        Shape::Pool => vec![by_lane(|l| (l as u32 / 4, 0))],
        Shape::Own => vec![by_lane(|l| (l as u32, 1))],
        Shape::Mixed => vec![by_lane(|_| (0, 0)), by_lane(|l| (l as u32, 1))],
    }
}

/// One instruction's lanes: the coordinates the shape picks and the
/// keys the LCG offers.
fn lanes(coords: &[(u32, u32)], lcg: &mut Lcg) -> Vec<Lane> {
    coords
        .iter()
        .map(|&(x, y)| Lane {
            x,
            y,
            value: lcg.next(),
            old: 0,
        })
        .collect()
}

/// The old values both forms return, one instruction's lanes.
fn olds(lanes: &[Lane]) -> Vec<u64> {
    lanes.iter().map(|l| l.old).collect()
}

/// The rounds as PR #257 lands them, every lane its own group.
fn round_form(m: &Memory, lanes: &mut [Lane], count: &mut Count) {
    round(m, lanes, count, &[false; LANES]);
}

/// Every form against a plain umax, on fresh state: the same texels,
/// the turn and round forms the same old values lane by lane, the umax
/// form's members never reading one, every lock freed.
fn checked(shape: Shape) {
    let mut lcg = Lcg(0x9E37_79B9_7F4A_7C15);
    for coords in programs(shape) {
        let given = lanes(&coords, &mut lcg);
        let turn_m = Memory::zeroed();
        let mut turn_lanes = given.clone();
        turn(&turn_m, &mut turn_lanes, &mut Count::default());
        let round_m = Memory::zeroed();
        let mut round_lanes = given.clone();
        round_form(&round_m, &mut round_lanes, &mut Count::default());
        let umax_m = Memory::zeroed();
        let mut umax_lanes = given.clone();
        umax(&umax_m, &mut umax_lanes, &mut Count::default());
        let plain = Memory::zeroed();
        let mut plain_lanes = given;
        reference(&plain, &mut plain_lanes);
        assert!(turn_m.freed(), "a lock left taken");
        assert!(round_m.freed(), "a lock left taken");
        assert!(umax_m.freed(), "a lock left taken");
        assert!(turn_m.texels_eq(&round_m), "the texels differ");
        assert!(turn_m.texels_eq(&umax_m), "the texels differ");
        assert!(turn_m.texels_eq(&plain), "the texels differ from a plain umax");
        assert_eq!(olds(&turn_lanes), olds(&round_lanes), "an old value differs");
        assert_eq!(
            olds(&turn_lanes),
            olds(&plain_lanes),
            "an old value differs from a plain umax"
        );
        let members = |lanes: &[Lane]| {
            (0..lanes.len())
                .filter(|&i| (0..i).any(|j| (lanes[j].x, lanes[j].y) == (lanes[i].x, lanes[i].y)))
                .map(|i| lanes[i].old)
                .collect::<Vec<_>>()
        };
        assert!(
            members(&umax_lanes).iter().all(|&old| old == 0),
            "a member read an old value"
        );
    }
}

fn main() {
    println!("bench_atomic_locks_shape {LANES} lanes {LOCKS} lock words");
    let shapes = [
        ("one", Shape::One),
        ("pool", Shape::Pool),
        ("own", Shape::Own),
        ("mixed", Shape::Mixed),
    ];
    let forms = [
        ("turn", turn as fn(&Memory, &mut [Lane], &mut Count)),
        ("round", round_form),
        ("umax", umax),
    ];

    // Untimed: every form exact, every shape.
    for (_, shape) in shapes {
        checked(shape);
    }
    println!("bench_atomic_locks_exact ok");

    // The counted op stream, once per form and shape.
    for (name, form) in forms {
        for (shape_name, shape) in shapes {
            let mut lcg = Lcg(0x9E37_79B9_7F4A_7C15);
            let mut count = Count::default();
            let m = Memory::zeroed();
            for coords in programs(shape) {
                let mut lanes = lanes(&coords, &mut lcg);
                form(&m, &mut lanes, &mut count);
            }
            println!("bench_atomic_locks_{name}_{shape_name}_barriers {}", count.barriers);
            println!("bench_atomic_locks_{name}_{shape_name}_attempts {}", count.attempts);
            println!("bench_atomic_locks_{name}_{shape_name}_rounds {}", count.rounds);
            println!("bench_atomic_locks_{name}_{shape_name}_wave_ops {}", count.wave_ops);
        }
    }

    // The timed op stream: 9 samples of VISITS programs, ns a visit.
    for (name, form) in forms {
        for (shape_name, shape) in shapes {
            let mut lcg = Lcg(0x9E37_79B9_7F4A_7C15);
            let mut samples = Vec::with_capacity(9);
            for _ in 0..9 {
                // Fresh keys per sample; the state the visits work on
                // carries over, as a shader's memory would.
                let mut given: Vec<Vec<Lane>> =
                    programs(shape).iter().map(|c| lanes(c, &mut lcg)).collect();
                let m = Memory::zeroed();
                let mut scratch = Count::default();
                let started = Instant::now();
                for _ in 0..VISITS {
                    for lanes in &mut given {
                        form(&m, lanes, &mut scratch);
                    }
                }
                samples.push(started.elapsed().as_nanos() as f64 / f64::from(VISITS));
                assert!(m.freed(), "a lock left taken");
                black_box(&given);
            }
            samples.sort_by(|a, b| a.total_cmp(b));
            let median = samples[samples.len() / 2];
            let all = samples
                .iter()
                .map(|s| format!("{s:.1}"))
                .collect::<Vec<_>>()
                .join(" ");
            println!("bench_atomic_locks_{name}_{shape_name}_ns {median:.1} ns (all: {all})");
        }
    }
}
