//! Console wave64 compute programs on a host whose waves have at most 32
//! lanes (NVIDIA, Intel: `WaveLaneCountMax` 32). Such a host refuses
//! `WaveSize(64)`, which a wave64 program that needs the console's lanes
//! ([`Lower::console_lanes`]) declares, so its pipeline was never made.
//!
//! Each console wave runs as two host waves of 32 lanes ("halves"): the
//! thread relabelling ([`Lower::thread_id`]) gives each host wave a 32-aligned
//! run of flat thread ids, so host wave `2w + h` holds console wave `w`'s
//! lanes `32h..32h + 31`, host lane `n` being console lane `32h + n`. What
//! a lane reads of its own half is exact. Reads into the other half pass
//! between the two host waves through group-shared memory, between group
//! barriers ([`Lower::exchange_halves`]); barriers need the whole group, so
//! only in a block every lane of the group runs together, the same number
//! of times ([`in_step`]: once, or in a loop on a count every host wave
//! computes alike). Elsewhere they are refused where they are made:
//!
//! * within a half, exact: DPP (rows of 16, DPP8 groups of 8), `ds_swizzle`
//!   (groups of 32 or quads), and `ds_bpermute_b32`, which in wave64 permutes
//!   each half on its own (RDNA2 ISA, "Data Share Operations"; measured on
//!   RDNA2 in upstream PR #573, banbanzhige's). A lane's own bit of a mask
//!   held as a number (a constant, data loaded or computed by scalar code),
//!   `v_writelane_b32` (the lane named takes the scalar) and the row a DPP
//!   row mask names (rows 2 and 3 are the upper half's) are decided by the
//!   console lane ([`Lower::console_lane`]);
//! * lane masks read as data (ballots) hold the other half's lanes: unless
//!   the whole group fits one host wave, the halves pass each other their
//!   ballots ([`Lower::ballot_across_halves`]), for a mask written in such a
//!   block; elsewhere refused ([`MASK_ACROSS`]);
//! * `v_readlane_b32` of a lane that may be in the other half (SGPR spills
//!   read back are not lane reads): each half passes the value of the lane
//!   with that index in it, and the console lane's half picks
//!   ([`Lower::read_lane_across`]);
//! * whole-wave reductions and prefix sums: each half reduces its own lanes,
//!   and they pass each other the result; a prefix in the upper half adds
//!   the lower half's total ([`Lower::reduce_across`]);
//! * `v_readfirstlane_b32` reads the host wave's first lane, which in the
//!   upper half is console lane 32 or later: kept as it is for a value the
//!   same in every lane of the console wave ([`lane_uniform`], what the
//!   compiler reads first lanes for) and in a group of 32 threads or fewer
//!   (one host wave holds the whole console wave); otherwise the lower
//!   half's first lane ([`Lower::first_lane_across`]); refused where no
//!   exchange can go ([`FIRST_ACROSS`]).
//!
//! A refusal in a value no instruction ends up needing costs nothing (the
//! lowering retries with that value missing), so a program is refused only
//! when it really reads across halves.
//!
//! LDS: the console compiler leaves out barriers between the lanes of one
//! wave, and so between all of a group's threads only when the group is one
//! wave. A group of 33 to 64 threads with LDS (two host waves sharing what
//! the console's one wave ordered by itself) is refused; groups of several
//! console waves order their LDS with barriers, and those of 32 threads or
//! fewer are one host wave.
//!
//! Approach after DexNom's fork (DexNom/Victory 5f4c3df, `set_host_max_wave`
//! and the thread relabelling into halves) and upstream PR #2176
//! (n-romanovich: the setting keyed in the shader cache); here the setting
//! is a translation option ([`super::super::Options::host_wave32`]), and
//! cross-half reads are refused where they are lowered instead of by
//! instruction name. Which values are the same in every lane of the console
//! wave ([`lane_uniform`]): justrealyme's analysis (chippy01/VIctory#1),
//! ported from fxceless/victory-fork (#497 there).

use super::*;

/// Console lanes per host wave in halves.
const HALF: u32 = 32;

impl Lower<'_> {
    /// Whether a wave64 program whose lanes follow the console's runs in
    /// halves: the host's waves are 32 lanes at most.
    pub(super) fn halves_wanted(x: &Parts, needs_wave: bool) -> bool {
        needs_wave && x.native.wave64 && x.host_wave32
    }

    /// Why a program in halves cannot run: LDS a group of one console wave
    /// shares without barriers, split across two host waves.
    pub(super) fn halves_lds(group: u32, lds_dwords: u32) -> Option<Why> {
        (lds_dwords > 0 && group > HALF && group <= 64).then_some(LDS_ACROSS)
    }

    /// The flat thread id, relabelled so that each host wave holds a run of
    /// [`Shader::wave_size`]-aligned ids (see [`Lower::thread_id`]); made once,
    /// in the entry block.
    ///
    /// When the group is larger than one host wave and `VICTORY_THREAD_TICKET`
    /// is enabled, waves claim aligned runs from a 1-dword group-shared counter
    /// rather than rounding `WaveReadLaneFirst(flattenedThreadIdInGroup)` down:
    /// 32-lane hosts can tile 2-D groups (e.g. 16x2 tiles in a 32x2 group),
    /// where two waves would otherwise round down to the same base and collide.
    /// Full waves take `w` from the low 16 bits; any partial tail wave takes
    /// its lane count from the high 16 bits past the full blocks, so it cannot
    /// misalign full waves regardless of arrival order.
    pub(super) fn relabelled_flat(&mut self) -> V {
        if let Some(f) = self.relabelled {
            return f;
        }
        let e = self.s.entry;
        let i32t = self.s.t.i32;
        let wave = self.s.wave_size.expect("set with console lanes");
        let group: u32 = self.s.threads.map_or(wave, |t| t.iter().product());
        let half_dwords = if self.halves {
            group.div_ceil(2 * HALF) * 2
        } else {
            0
        };
        let use_ticket = group > wave
            && super::super::thread_ticket()
            && self.s.lds_dwords + half_dwords < GROUP_SHARED_DWORDS;
        let f = if use_ticket {
            let arr = self.s.m.array(1, i32t);
            let counter = self.s.m.global("wave_ticket", arr, 3);
            let zero = self.s.m.i32(0);
            let cell = self.s.m.element(e, counter, zero);
            self.s.m.store(e, cell, zero);
            let sync = self.s.m.i32(1 | 8);
            self.s
                .op(e, "barrier", Attr::NoDuplicate, self.s.t.void, 80, &[sync]);
            let one = self.s.m.i32(1);
            let (sum, unsigned) = (self.s.m.i8(0), self.s.m.i8(1));
            let count = self.s.op(
                e,
                "waveActiveOp.i32",
                Attr::NoUnwind,
                i32t,
                119,
                &[one, sum, unsigned],
            );
            let rank = self.s.op(
                e,
                "wavePrefixOp.i32",
                Attr::NoUnwind,
                i32t,
                121,
                &[one, sum, unsigned],
            );
            let kw = self.s.m.i32(wave);
            let full = self.s.m.cmp(e, pred::EQ, count, kw);
            let first = self.s.m.cmp(e, pred::EQ, rank, zero);
            let sixteen = self.s.m.i32(16);
            let high = self.s.m.bin(e, Bin::Shl, count, sixteen);
            let take = self.s.m.select(e, full, kw, high);
            let add = self.s.m.select(e, first, take, zero);
            let old = self
                .s
                .m
                .atomic(e, super::super::bitcode::Rmw::Add, cell, add);
            let found = self.s.op(
                e,
                "waveReadLaneFirst.i32",
                Attr::NoUnwind,
                i32t,
                118,
                &[old],
            );
            let low_mask = self.s.m.i32(0xffff);
            let low = self.s.m.bin(e, Bin::And, found, low_mask);
            let past = self.s.m.bin(e, Bin::LShr, found, sixteen);
            let blocks = self.s.m.i32(group / wave * wave);
            let tail = self.s.m.bin(e, Bin::Add, blocks, past);
            let base = self.s.m.select(e, full, low, tail);
            self.s.m.bin(e, Bin::Add, base, rank)
        } else {
            let flat = self.s.op(
                e,
                "flattenedThreadIdInGroup.i32",
                Attr::ReadNone,
                i32t,
                96,
                &[],
            );
            let first = self.s.op(
                e,
                "waveReadLaneFirst.i32",
                Attr::NoUnwind,
                i32t,
                118,
                &[flat],
            );
            let mask = self.s.m.i32(!(wave - 1));
            let base = self.s.m.bin(e, Bin::And, first, mask);
            let lane = self
                .s
                .op(e, "waveGetLaneIndex", Attr::ReadOnly, i32t, 111, &[]);
            self.s.m.bin(e, Bin::Add, base, lane)
        };
        self.s.flags |= flags::WAVE_OPS;
        self.relabelled = Some(f);
        f
    }

    /// This lane's index in its console wave: the host lane, or in halves
    /// the host lane in the half the host wave holds.
    pub(super) fn console_lane(&mut self) -> V {
        if !self.halves {
            return self.lane_index();
        }
        let f = self.relabelled_flat();
        let m = self.s.m.i32(self.wave() - 1);
        self.s.m.bin(self.s.entry, Bin::And, f, m)
    }

    /// Whether the group fits one host wave: a console wave64 of 32 threads
    /// or fewer has no lane in the upper half.
    pub(super) fn one_host_wave(&self) -> bool {
        self.s
            .threads
            .is_some_and(|t| t.iter().product::<u32>() <= HALF)
    }

    /// Where the halves pass each other a lane mask written in IR block
    /// `def` (none: a program input, in the entry): there, if every lane of
    /// the group runs it together ([`in_step`]); in the block being
    /// lowered, where the reading instruction starts ([`Lower::level`],
    /// before any guard of its own).
    pub(super) fn across_at(&self, def: Option<usize>) -> Option<usize> {
        if !self.step.get(def.unwrap_or(0)).copied().unwrap_or(false) {
            return None;
        }
        match def {
            None => Some(self.s.entry),
            Some(b) if b == self.cur_ir => Some(self.level),
            Some(b) => self.tail[b],
        }
    }

    /// Whether the halves can pass each other values where the instruction
    /// being lowered is: in halves, in a group of more than one host wave, in
    /// a block every lane of the group runs together ([`in_step`]), before
    /// anything of the instruction's own split it ([`Lower::level`]).
    pub(super) fn across_here(&self) -> bool {
        self.halves
            && !self.one_host_wave()
            && self.cur == self.level
            && self.step.get(self.cur_ir).copied().unwrap_or(false)
    }

    /// The half of the console wave this lane is in: 0 or 1.
    fn half(&mut self, at: usize) -> V {
        let f = self.relabelled_flat();
        let (five, one) = (self.s.m.i32(5), self.s.m.i32(1));
        let slot = self.s.m.bin(at, Bin::LShr, f, five);
        self.s.m.bin(at, Bin::And, slot, one)
    }

    /// Both dwords of the console wave's ballot of `bit`, in halves: each
    /// host wave's own (its half's lanes), passed between them
    /// ([`Lower::exchange_halves`]). A console wave with no upper half has a
    /// high dword of 0. Every lane of the group must run `at` together
    /// ([`Lower::across_at`]).
    pub(super) fn ballot_across_halves(&mut self, bit: V, at: usize) -> Result<(V, V), Why> {
        let i32t = self.s.t.i32;
        let four = self.s.m.named_struct("dx.types.fouri32", vec![i32t; 4]);
        let r = self
            .s
            .op(at, "waveActiveBallot", Attr::NoUnwind, four, 116, &[bit]);
        self.s.flags |= flags::WAVE_OPS;
        let own = self.s.m.extract(at, r, 0);
        let zero = self.s.m.i32(0);
        self.exchange_halves(own, zero, at)
    }

    /// Group-wide logical OR of `bit` across every lane of every host wave in
    /// the threadgroup, so every host wave exits a lockstep predicated loop on
    /// the exact same iteration (never deadlocking at group barriers inside
    /// `exchange_halves`). Reuses `self.half_words` (zero extra LDS bytes).
    pub(super) fn group_any_true(&mut self, bit: V, at: usize) -> Result<V, Why> {
        let group: u32 = self.s.threads.map_or(0, |t| t.iter().product());
        let i32t = self.s.t.i32;
        let four = self.s.m.named_struct("dx.types.fouri32", vec![i32t; 4]);
        let r = self
            .s
            .op(at, "waveActiveBallot", Attr::NoUnwind, four, 116, &[bit]);
        self.s.flags |= flags::WAVE_OPS;
        let own = self.s.m.extract(at, r, 0);
        let zero = self.s.m.i32(0);
        if group <= HALF {
            return Ok(self.s.m.cmp(at, pred::NE, own, zero));
        }
        if group <= 2 * HALF {
            let (lo, hi) = self.exchange_halves(own, zero, at)?;
            let any = self.s.m.bin(at, Bin::Or, lo, hi);
            return Ok(self.s.m.cmp(at, pred::NE, any, zero));
        }
        let dwords = group.div_ceil(2 * HALF) * 2;
        if self.s.lds_dwords + dwords > GROUP_SHARED_DWORDS {
            return Err("no group-shared memory left to pass values between 32-lane halves");
        }
        let f = self.relabelled_flat();
        let words = match self.half_words {
            Some(g) => g,
            None => {
                let arr = self.s.m.array(dwords, i32t);
                let g = self.s.m.global("half_words", arr, 3);
                self.half_words = Some(g);
                g
            }
        };
        let five = self.s.m.i32(5);
        let slot = self.s.m.bin(at, Bin::LShr, f, five);
        let sync = |l: &mut Self| {
            let mode = l.s.m.i32(1 | 8);
            let void = l.s.t.void;
            l.s.op(at, "barrier", Attr::NoDuplicate, void, 80, &[mode]);
        };
        sync(self);
        let mine = self.s.m.element(at, words, slot);
        self.s.m.store(at, mine, own);
        sync(self);
        let host_waves = group.div_ceil(HALF);
        let mut acc = zero;
        for w in 0..host_waves {
            let idx = self.s.m.i32(w);
            let ptr = self.s.m.element(at, words, idx);
            let val = self.s.m.load(at, ptr);
            acc = self.s.m.bin(at, Bin::Or, acc, val);
        }
        Ok(self.s.m.cmp(at, pred::NE, acc, zero))
    }

    /// What each half of this lane's console wave gave as `own` (a dword
    /// the same in every lane of a host wave): (lower half's, upper half's).
    /// Each host wave stores its own in a group-shared dword of its own, and
    /// reads its console wave's two after a group barrier; a barrier before
    /// the store keeps it from overwriting what the other half has not read
    /// of an earlier exchange. A console wave with no upper half (a group not
    /// a multiple of 64) reads `missing` for it. Every lane of the group must
    /// run `at` together ([`in_step`]), in halves.
    pub(super) fn exchange_halves(&mut self, own: V, missing: V, at: usize) -> Result<(V, V), Why> {
        let group: u32 = self.s.threads.map_or(0, |t| t.iter().product());
        // Two dwords per console wave, so the high dword of the last one is
        // inside even where the group has no upper half.
        let dwords = group.div_ceil(2 * HALF) * 2;
        if self.s.lds_dwords + dwords > GROUP_SHARED_DWORDS {
            return Err("no group-shared memory left to pass values between 32-lane halves");
        }
        let i32t = self.s.t.i32;
        let f = self.relabelled_flat();
        let words = match self.half_words {
            Some(g) => g,
            None => {
                let arr = self.s.m.array(dwords, i32t);
                let g = self.s.m.global("half_words", arr, 3);
                self.half_words = Some(g);
                g
            }
        };
        // Host wave `2w + h` (its flat ids `32 (2w + h)` on) holds console
        // wave `w`'s half `h`.
        let five = self.s.m.i32(5);
        let slot = self.s.m.bin(at, Bin::LShr, f, five);
        let not_one = self.s.m.i32(!1);
        let low = self.s.m.bin(at, Bin::And, slot, not_one);
        let one = self.s.m.i32(1);
        let high = self.s.m.bin(at, Bin::Or, slot, one);
        let sync = |l: &mut Self| {
            // Group sync with a group-shared memory fence.
            let mode = l.s.m.i32(1 | 8);
            let void = l.s.t.void;
            l.s.op(at, "barrier", Attr::NoDuplicate, void, 80, &[mode]);
        };
        sync(self);
        let mine = self.s.m.element(at, words, slot);
        self.s.m.store(at, mine, own);
        sync(self);
        let lo = self.s.m.element(at, words, low);
        let lo = self.s.m.load(at, lo);
        let hi = self.s.m.element(at, words, high);
        let mut hi = self.s.m.load(at, hi);
        if !group.is_multiple_of(2 * HALF) {
            let first = self.s.m.bin(at, Bin::Shl, high, five);
            let n = self.s.m.i32(group);
            let there = self.s.m.cmp(at, pred::IULT, first, n);
            hi = self.s.m.select(at, there, hi, missing);
        }
        Ok((lo, hi))
    }

    /// `v_readlane_b32` of `x` at console lane `lane` (the low 6 bits), in
    /// halves ([`Lower::across_here`]): each half reads the lane with that
    /// index in it, and the lane's own half gives the value.
    pub(super) fn read_lane_across(&mut self, x: V, lane: V) -> Result<V, Why> {
        let c = self.cur;
        let (k31, k32, zero) = (self.s.m.i32(31), self.s.m.i32(32), self.s.m.i32(0));
        let index = self.s.m.bin(c, Bin::And, lane, k31);
        self.s.flags |= flags::WAVE_OPS;
        let own = self.read_lane(x, index);
        let (lo, hi) = self.exchange_halves(own, zero, c)?;
        let upper = self.s.m.bin(c, Bin::And, lane, k32);
        let upper = self.s.m.cmp(c, pred::NE, upper, zero);
        Ok(self.s.m.select(c, upper, hi, lo))
    }

    /// `v_readfirstlane_b32` of `x`, in halves ([`Lower::across_here`]): the
    /// first active lane across both 32-lane halves. When `active` is `None`,
    /// every lane is active and the lower half's lane 0 wins; when `active` is
    /// `Some(act)`, each half finds its own first active lane (defaulting to
    /// its lane 0 when inactive) and exchanges `(value, ballot)` so the lower
    /// half wins if any lower lane is active, the upper half wins if only upper
    /// lanes are active, and empty EXEC (neither half active) returns lane 0's
    /// value, matching RDNA2 hardware.
    pub(super) fn first_lane_across(&mut self, x: V, active: Option<V>) -> Result<V, Why> {
        let c = self.cur;
        let (i32t, zero) = (self.s.t.i32, self.s.m.i32(0));
        self.s.flags |= flags::WAVE_OPS;
        let Some(act) = active else {
            let own = self
                .s
                .op(c, "waveReadLaneFirst.i32", Attr::NoUnwind, i32t, 118, &[x]);
            return Ok(self.exchange_halves(own, zero, c)?.0);
        };
        let four = self.s.m.named_struct("dx.types.fouri32", vec![i32t; 4]);
        let b = self
            .s
            .op(c, "waveActiveBallot", Attr::NoUnwind, four, 116, &[act]);
        let ballot = self.s.m.extract(c, b, 0);
        // DXIL opcode 32: FirstbitLo (returns 0..31 for non-zero ballot,
        // 0xffffffff for 0). Default to lane 0 when this half has no active
        // lanes so empty EXEC across both halves returns console lane 0.
        let own_any = self.s.m.cmp(c, pred::NE, ballot, zero);
        let bit = self.unary_bits(c, 32, ballot);
        let lane = self.s.m.select(c, own_any, bit, zero);
        let own = self.read_lane(x, lane);
        let (lo_val, hi_val) = self.exchange_halves(own, zero, c)?;
        let (lo_ballot, hi_ballot) = self.exchange_halves(ballot, zero, c)?;
        let lo_empty = self.s.m.cmp(c, pred::EQ, lo_ballot, zero);
        let hi_any = self.s.m.cmp(c, pred::NE, hi_ballot, zero);
        let pick_hi = self.s.m.bin(c, Bin::And, lo_empty, hi_any);
        Ok(self.s.m.select(c, pick_hi, hi_val, lo_val))
    }

    /// `v_readfirstlane_b32` of `x` within a single host wave: when `active` is
    /// `Some(act)`, finds the first lane where `act` holds (falling back to
    /// lane 0 on empty EXEC, matching RDNA2 hardware); otherwise reads the host
    /// wave's first active lane.
    pub(super) fn first_lane_in_wave(&mut self, x: V, active: Option<V>) -> V {
        let c = self.cur;
        let i32t = self.s.t.i32;
        self.s.flags |= flags::WAVE_OPS;
        let Some(act) = active else {
            return self
                .s
                .op(c, "waveReadLaneFirst.i32", Attr::NoUnwind, i32t, 118, &[x]);
        };
        let zero = self.s.m.i32(0);
        let four = self.s.m.named_struct("dx.types.fouri32", vec![i32t; 4]);
        let b = self
            .s
            .op(c, "waveActiveBallot", Attr::NoUnwind, four, 116, &[act]);
        let lo = self.s.m.extract(c, b, 0);
        let lo_any = self.s.m.cmp(c, pred::NE, lo, zero);
        let lo_bit = self.unary_bits(c, 32, lo);
        let lane = if !self.halves {
            let hi = self.s.m.extract(c, b, 1);
            let hi_any = self.s.m.cmp(c, pred::NE, hi, zero);
            let hi_bit = self.unary_bits(c, 32, hi);
            let k32 = self.s.m.i32(32);
            let hi_lane = self.s.m.bin(c, Bin::Add, hi_bit, k32);
            let fallback = self.s.m.select(c, hi_any, hi_lane, zero);
            self.s.m.select(c, lo_any, lo_bit, fallback)
        } else {
            self.s.m.select(c, lo_any, lo_bit, zero)
        };
        self.read_lane(x, lane)
    }

    /// `Shape::Half(Op::AddH)` 16-bit float wave reduction (low 16 bits = sum
    /// of active lanes' low halves, high 16 bits = lane `hi`'s high half when
    /// active else `ident`'s high half), supporting both wave64 and 32-lane
    /// `halves` mode.
    pub(super) fn reduce_half(
        &mut self,
        src: Value,
        mask: Value,
        ident: u32,
        hi: u32,
    ) -> Result<V, Why> {
        const ACROSS: Why = "a wave reduction over both 32-lane halves (wave64 on a 32-lane host)";
        if self.mask_data != MaskData::Console {
            return Err("a 16-bit wave reduction outside compute");
        }
        let across = self.halves && !self.one_host_wave();
        if across && !self.across_here() {
            return Err(ACROSS);
        }
        let c = self.cur;
        let &m = self
            .x
            .st
            .masks
            .get(&mask)
            .ok_or("a wave reduction's untracked mask")?;
        let mut on = self.boolean(m, c)?;
        if let Some(active) = self.pred_active(self.cur_ir) {
            on = self.s.m.bin(c, Bin::And, on, active);
        }
        let x = self.i32_of(src)?;
        let id = self.s.m.i32(ident);
        let x = self.s.m.select(c, on, x, id);
        self.s.flags |= flags::WAVE_OPS;
        let low = self.s.m.i32(0xFFFF);
        let h = self.s.m.bin(c, Bin::And, x, low);
        let f = self.f16_to_f32(c, h);
        let (sum, signed) = (self.s.m.i8(0), self.s.m.i8(0));
        let r = self.s.op(
            c,
            "waveActiveOp.f32",
            Attr::NoUnwind,
            self.s.t.f32,
            119,
            &[f, sum, signed],
        );
        // In halves, each host wave reduced its own 32 lanes. Pass those f32
        // totals between the halves before rounding once.
        let r = if across {
            self.reduce_across(super::super::wave::Op::AddH, r, r, false, ident)?
        } else {
            r
        };
        let r = self.f32_to_f16(c, r);
        let r = self.s.m.bin(c, Bin::And, r, low);
        // Lane `hi` runs the section on the console even where it is
        // elsewhere on the host; there it holds the identity.
        let i32t = self.s.t.i32;
        let word = if across {
            // Lane `hi` may be in the other host wave: both halves' ballots,
            // as its value is read across them below.
            let (lo, up) = self.ballot_across_halves(on, c)?;
            if hi < 32 { lo } else { up }
        } else {
            let four = self.s.m.named_struct("dx.types.fouri32", vec![i32t; 4]);
            let b = self
                .s
                .op(c, "waveActiveBallot", Attr::NoUnwind, four, 116, &[on]);
            self.s.m.extract(c, b, hi / 32)
        };
        let (sh, one, z) = (self.s.m.i32(hi % 32), self.s.m.i32(1), self.s.m.i32(0));
        let bit = self.s.m.bin(c, Bin::LShr, word, sh);
        let bit = self.s.m.bin(c, Bin::And, bit, one);
        let there = self.s.m.cmp(c, pred::NE, bit, z);
        let lane = self.s.m.i32(hi);
        let at = if across {
            self.read_lane_across(x, lane)?
        } else {
            self.read_lane(x, lane)
        };
        let top = self.s.m.select(c, there, at, id);
        let high = self.s.m.i32(0xFFFF_0000);
        let top = self.s.m.bin(c, Bin::And, top, high);
        Ok(self.s.m.bin(c, Bin::Or, top, r))
    }

    /// A whole-wave reduction or prefix of `op`, in halves
    /// ([`Lower::across_here`]): `own` is this host wave's (its half's
    /// lanes), `total` its half's total (a prefix's; the reduction itself
    /// otherwise), both as the operation's type. A reduction takes `op` of
    /// both halves' totals; a prefix in the upper half adds the lower half's
    /// total. A console wave with no upper half reads `ident` for it.
    pub(super) fn reduce_across(
        &mut self,
        op: super::super::wave::Op,
        own: V,
        total: V,
        prefix: bool,
        ident: u32,
    ) -> Result<V, Why> {
        use super::super::wave::Op;
        let c = self.cur;
        // AddH's half results are accumulated in f32, then rounded to f16 by
        // the caller, just like the single-wave lowering.
        let float = matches!(op, Op::AddF | Op::AddH | Op::MinF | Op::MaxF);
        let bits = if float { self.as_int(total) } else { total };
        let missing = self.s.m.i32(if op == Op::AddH { 0 } else { ident });
        let (lo, hi) = self.exchange_halves(bits, missing, c)?;
        let (lo, hi) = if float {
            (self.as_float(lo), self.as_float(hi))
        } else {
            (lo, hi)
        };
        if prefix {
            let h = self.half(c);
            let zero = self.s.m.i32(0);
            let upper = self.s.m.cmp(c, pred::NE, h, zero);
            let added = self.s.m.bin(c, Bin::Add, lo, own);
            return Ok(self.s.m.select(c, upper, added, own));
        }
        let unsigned_or_signed = |l: &mut Self, p: u8| {
            let less = l.s.m.cmp(c, p, lo, hi);
            l.s.m.select(c, less, lo, hi)
        };
        Ok(match op {
            Op::AddU | Op::AddF | Op::AddH => self.s.m.bin(c, Bin::Add, lo, hi),
            Op::Or => self.s.m.bin(c, Bin::Or, lo, hi),
            Op::And => self.s.m.bin(c, Bin::And, lo, hi),
            Op::Xor => self.s.m.bin(c, Bin::Xor, lo, hi),
            Op::MinU => unsigned_or_signed(self, pred::IULT),
            Op::MaxU => unsigned_or_signed(self, pred::IUGT),
            Op::MinI => unsigned_or_signed(self, pred::SLT),
            Op::MaxI => unsigned_or_signed(self, pred::SGT),
            // FMin, FMax: as the host's own wave minimum and maximum.
            Op::MinF => self.binary(36, lo, hi),
            Op::MaxF => self.binary(35, lo, hi),
        })
    }
}

pub(super) const LDS_ACROSS: Why =
    "LDS one console wave64 shares without barriers, on a host of 32-lane waves";
pub(super) const REDUCTION_ACROSS: Why =
    "a wave reduction over both 32-lane halves (wave64 on a 32-lane host)";
/// A `v_writelane_b32` in a block the predication runs in every lane: the
/// lane it names may be inactive there, and the merge after the block keeps
/// that lane's old value (the console writes it whatever EXEC holds).
pub(super) const WRITE_ACROSS: Why =
    "a lane write in a predicated block (wave64 on a 32-lane host)";
pub(super) const READ_ACROSS: Why =
    "a lane read that may cross 32-lane halves (wave64 on a 32-lane host)";

/// Whether both halves calculate the same selector for a lane read or write.
pub(super) fn uniform_selector(inst: &Inst, alike: &Set<Value>) -> bool {
    let Fields::Valu(v) = &inst.fields else {
        return false;
    };
    match v.src.get(1) {
        Some(Operand::Const(k)) => crate::resource::inline(*k).is_some(),
        Some(Operand::Literal(_) | Operand::Scalar { code: NULL, .. }) => true,
        Some(Operand::Scalar { code, .. }) => {
            inst.read(Reg::S(*code)).is_some_and(|v| alike.contains(&v))
        }
        _ => false,
    }
}

/// Why a lane mask read as data stops in halves: it is written where the
/// lanes of the group may not all be, as often, so no group barrier can pass
/// it between the halves.
pub(super) const MASK_ACROSS: Why = "a lane mask read as data (wave64 on a 32-lane host)";

/// The group-shared dwords a D3D12 compute program may declare, at most
/// (32 KiB).
const GROUP_SHARED_DWORDS: u32 = 8192;

/// Why a first-lane read stops in halves: the console's first lane may be
/// in the other half, and the value is not known to be the same there.
pub(super) const FIRST_ACROSS: Why = "a first-lane read of a value that may differ between 32-lane halves (wave64 on a 32-lane host)";

/// Whether `x` (a wave64 compute program) has a non-waterfall
/// `v_readfirstlane_b32` whose vector source may differ between lanes of the
/// console wave: such a read needs console-wave lane order (and halves on a
/// 32-lane host) so both halves see the same first active lane. Not one each
/// host half can make by itself (`alone`, [`super::first_lane::read_alone`]).
pub(super) fn needs_first_lane(
    x: &Parts,
    alike: &mut Option<HashSet<Value>>,
    alone: &HashSet<Value>,
) -> bool {
    if !x.native.wave64 {
        return false;
    }
    let wf = &x.res.waterfall.reads;
    let is_vec_first = |i: &Inst| -> Option<Value> {
        if i.name() != "v_readfirstlane_b32"
            || i.writes
                .first()
                .is_some_and(|w| wf.contains(&w.1) || alone.contains(&w.1))
        {
            return None;
        }
        let Fields::Valu(v) = &i.fields else {
            return None;
        };
        match v.src.first() {
            Some(&Operand::Vector { reg, .. }) => i.read(Reg::V(reg)),
            _ => None,
        }
    };
    let mut srcs = x
        .p
        .blocks
        .iter()
        .flat_map(|b| &b.insts)
        .filter_map(is_vec_first)
        .peekable();
    if srcs.peek().is_none() {
        return false;
    }
    let uniform = alike.get_or_insert_with(|| lane_uniform(x));
    srcs.any(|val| !uniform.contains(&val))
}

/// Every branch (by its block) with its region: the blocks reachable from
/// it that do not post-dominate it, which only the lanes taking one side may
/// run (the branch's own block among them where a cycle leads back to it).
/// None with an indirect exit.
///
/// Lowering caps an unconditional back edge — a `Jump` to a block at or
/// before the jump, by reverse postorder — with an early return at the loop
/// iteration limit: the latch leaves the program there too. That generated
/// exit is block `n` here, a terminal exit just like a `Ret`, so a closed
/// loop with a capped back edge does not look like it has no way out (which
/// would make everything post-dominate its branches).
pub(super) fn branch_regions(p: &Program, st: &Structured) -> Option<Vec<(usize, Vec<bool>)>> {
    branch_regions_with(p, &st.exits)
}

/// [`branch_regions`] over an explicit `exits` slice (used by `step_and_cap_with`).
pub(super) fn branch_regions_with(
    p: &Program,
    exits: &[LaneExit],
) -> Option<Vec<(usize, Vec<bool>)>> {
    let n = exits.len();
    // Reverse postorder ranks, as lowering numbers them.
    let order = crate::ir::reverse_postorder(&p.blocks);
    let mut rpo = vec![usize::MAX; n];
    for (r, &b) in order.iter().enumerate() {
        rpo[b] = r;
    }
    let capped = |b: usize| {
        matches!(exits[b], LaneExit::Jump(t) if rpo[b] != usize::MAX && rpo[t] <= rpo[b])
    };
    // Block `n` is the virtual exit of every capped latch.
    let m = n + 1;
    let mut succ: Vec<Vec<usize>> = vec![Vec::new(); m];
    for (b, e) in exits.iter().enumerate() {
        match *e {
            LaneExit::Jump(t) => {
                succ[b].push(t);
                if capped(b) {
                    succ[b].push(n);
                }
            }
            LaneExit::Branch {
                taken, not_taken, ..
            } => {
                succ[b].push(taken);
                succ[b].push(not_taken);
            }
            LaneExit::End => {}
            LaneExit::Indirect => return None,
        }
    }
    // Post-dominators, as bitsets (a virtual exit after every end).
    let words = m.div_ceil(64);
    let full = vec![u64::MAX; words];
    let mut pdom: Vec<Vec<u64>> = (0..m)
        .map(|b| {
            if succ[b].is_empty() {
                let mut v = vec![0; words];
                v[b / 64] |= 1 << (b % 64);
                v
            } else {
                full.clone()
            }
        })
        .collect();
    let mut changed = true;
    while changed {
        changed = false;
        for b in (0..m).rev() {
            if succ[b].is_empty() {
                continue;
            }
            let mut v = full.clone();
            for &t in &succ[b] {
                for (w, p) in v.iter_mut().zip(&pdom[t]) {
                    *w &= p;
                }
            }
            v[b / 64] |= 1 << (b % 64);
            if v != pdom[b] {
                pdom[b] = v;
                changed = true;
            }
        }
    }
    let mut regions = Vec::new();
    for (d, e) in exits.iter().enumerate() {
        if !matches!(e, LaneExit::Branch { .. }) {
            continue;
        }
        let mut seen = vec![false; m];
        let mut stack: Vec<usize> = succ[d].clone();
        while let Some(b) = stack.pop() {
            // The branch's own block is in its region if a cycle leads back
            // to it (a loop's latch): only the lanes going round run it.
            if seen[b] || b != d && pdom[d][b / 64] >> (b % 64) & 1 == 1 {
                continue;
            }
            seen[b] = true;
            stack.extend(succ[b].iter().copied());
        }
        regions.push((d, seen[..n].to_vec()));
    }
    Some(regions)
}

/// The thread id's x (`v0` on entry) of a wave64 compute program whose
/// console waves are runs of 64 ids along x (a group 64-aligned along x,
/// or one row): `id >> 6` and up is one value per console wave.
fn wave_tid(x: &Parts) -> Option<Value> {
    let StageIo::Cs(cs) = &x.native.io else {
        return None;
    };
    let [tx, ty, tz] = cs.threads;
    if !x.native.wave64 || !(tx % 64 == 0 || ty * tz == 1) {
        return None;
    }
    x.p.inputs
        .iter()
        .find(|(r, _)| *r == Reg::V(0))
        .map(|(_, v)| *v)
}

/// Entry thread-id VGPRs (`v1`/`v2`) that are wave-uniform in a wave64 compute
/// group: `v1` (`tid.y`) when `ty == 1` or `tx` is a multiple of 64, and `v2`
/// (`tid.z`) when `tz == 1` or `tx * ty` is a multiple of 64.
fn uniform_entry_tids(x: &Parts) -> Vec<Value> {
    let StageIo::Cs(cs) = &x.native.io else {
        return Vec::new();
    };
    if !x.native.wave64 {
        return Vec::new();
    }
    let [tx, ty, tz] = cs.threads;
    let mut out = Vec::new();
    let v1_uniform = ty <= 1 || tx.is_multiple_of(64);
    let v2_uniform = tz <= 1 || (tx * ty).is_multiple_of(64);
    for &(reg, v) in &x.p.inputs {
        if (reg == Reg::V(1) && v1_uniform) || (reg == Reg::V(2) && v2_uniform) {
            out.push(v);
        }
    }
    out
}

/// VGPR values that are the same in every lane of a console wave holding
/// them, so that a first-lane read in either half gives the console's
/// value: written lane by lane from scalars, constants and such values, by
/// an instruction every arriving lane runs or that runs under a condition
/// itself the same in every lane (then only over an old value that is such
/// a value too); loaded from LDS, a buffer or global memory, at such an
/// address (see below); or the thread
/// id's x shifted right past the console wave ([`wave_tid`]). The console
/// compiler keeps wave-uniform data in VGPRs and fetches it with
/// `v_readfirstlane` (a typed buffer's element, say, which only a vector
/// load reads). A scalar is one value, except the dwords a waterfall loop
/// reads, which are each lane's own.
///
/// A global load of dwords counts only where every lane of the console wave
/// runs it together (a lane-dependent guard leaves the other lanes' copy of
/// its result stale), and where no lane of the shader can write what it
/// reads: equal addresses do not prove equal load results where memory can
/// be written, as the console wave's two host waves can reach the load on
/// opposite sides of another lane's store. So the shader must perform no
/// store or atomic that could alias it, or the load's address must resolve
/// to the code's own immutable table (a root SRV and a root UAV can bind the
/// same title-memory address, so a root descriptor is not read-only there).
/// Its scalar base words need positive proof of uniformity too: a root
/// parameter's dword, or a whitelisted scalar op over such values
/// ([`SCALAR_ALOP`]); a missing read or an unwhitelisted def is unknown, so
/// an `s_mov_b32` copy of a per-lane read is not uniform.
///
/// A `buffer_load_*`/`tbuffer_load*` or `ds_read*` counts only where no
/// write the shader performs can alias what it reads (`!writes_global` for
/// a buffer, `!writes_lds` for LDS): where one can, the console wave's two
/// host waves can reach the load on opposite sides of another lane's store
/// or atomic and read different values. Every register it reads must also
/// be positively proven uniform: the VGPR address words in `same` (never a
/// waterfall read), and the SGPR descriptor (`srsrc`, all four dwords) and
/// scalar offset (`soffset`) words in [`same_s`] — an unproven read (an
/// `s_load_dword`'s result, a waterfall dword) is unknown, so a buffer
/// offset derived from an `s_load_dword` is not uniform.
///
/// Where the shader performs a store or atomic (`writes_global`), the VGPR
/// offset and address words of a code-table `global_load_dword*` must come
/// from `pure` instead, the VGPR values only immutable sources write:
/// constants, [`same_s`] scalars, pure VALU ops over those, or a code-table
/// load itself. (`buffer_load_*` and `ds_read*` results enter `same` only
/// where the shader performs no store or LDS write at all, so they cannot
/// differ between the halves and be the offset of a code-table entry read
/// in each half while a first-lane read of the load's result skips the
/// cross-half exchange.) So a value is `pure` only
/// where the control flow around it is pure too: a pure phi needs every
/// predecessor reached through branches whose conditions are proven from
/// `pure`/`same_s`, and a pure guarded write needs a guard proven the same
/// way — a branch or guard on a merely `same` (load-derived) condition can
/// have split the halves, and the phi would pick a different constant in
/// each of them. The same holds for `same` itself where the shader writes
/// memory: a guarded write or a phi then counts only where its guard (or
/// every predecessor branch) is proven from pure values too, since a
/// load-derived guard or branch condition can differ between the halves,
/// and the write would happen in one half and not the other.
///
/// A pure VALU op reads only pure values: every SSA read of it is checked —
/// each VGPR word in `pure`, each SGPR word in [`same_s`] (a 64-bit
/// operand's high word too), never the carry (`src_scc`: lowering reads the
/// live SCC) nor a lane mask as a source; inline constants and literals are
/// not reads.
///
/// A VALU op is alike only where every register dword it reads is alike: its
/// operands, and the old destination IR records in `inst.reads` (inactive
/// lanes keep it, accumulators like `v_dot4c_i32_i8` consume it) except where
/// a full-dword overwrite replaces it and no guard can leave lanes inactive.
/// Each SGPR word it reads needs positive proof of uniformity in [`same_s`]
/// (a lane mask only where a compare of alike values made it); the carry
/// (`src_scc`) is one value per host wave — the two halves can hold
/// different SCCs — so it proves nothing. An op that reads a 64-bit lane
/// mask as data — `v_cndmask_b32`'s select mask, a `*_co_ci_u32` carry-in —
/// is alike only where the mask's per-lane bit is the same in every lane
/// ([`mask_bit_alike`]): a literal all-zero or all-ones mask, or a mask a
/// compare of alike values made.
///
/// A branch is one every lane of the console wave takes together only where
/// every [`Atom::Uniform`] it tests is a scalar compare whose operands are
/// proven the same in every lane of the console wave: an SCC is one value
/// per host wave, but an `s_load_dword` can see different stages of a UAV
/// update in the two halves, and a branch on its result is not wave-all.
///
/// A phi counts where every predecessor is a block every lane of the wave
/// reaches together: no branch on a lane-dependent condition can have split
/// them (a branch on alike values does not). The analysis runs to a fixed
/// point from nothing, so a loop-carried value that needs itself to be
/// alike never is.
pub(super) fn lane_uniform(x: &Parts) -> HashSet<Value> {
    let (p, st) = (x.p, x.st);
    let wave_tid = wave_tid(x);
    let entry_tids = uniform_entry_tids(x);
    let regions = branch_regions(p, st);
    let is_mask = |v: Value| st.masks.contains_key(&v) || st.highs.contains_key(&v);
    let per_lane = |v: Value| x.res.waterfall.reads.contains(&v);
    // Scalar SSA values the same in every lane of a console wave, proven the
    // same way: a root parameter's dword on entry (a root constant, a user
    // SGPR), grown by [`SCALAR_ALOP`] over such values in the fixed point.
    let mut same_s: HashSet<Value> = x.res.params.iter().map(|&(v, _, _)| v).collect();
    // Whether any instruction writes memory a `global_load_dword*` can read
    // (a global, flat, buffer or image store or atomic, or a scalar store,
    // scalar atomic or scalar buffer store or atomic): then a global load
    // can observe another lane's write.
    let writes_global = p.blocks.iter().flat_map(|b| &b.insts).any(|inst| {
        let n = inst.name();
        n.starts_with("global_store_")
            || n.starts_with("global_atomic_")
            || n.starts_with("flat_store_")
            || n.starts_with("flat_atomic_")
            || n.starts_with("buffer_store_")
            || n.starts_with("buffer_atomic_")
            || n.starts_with("tbuffer_store_")
            || n.starts_with("image_store")
            || n.starts_with("image_atomic_")
            || n.starts_with("s_store_")
            || n.starts_with("s_atomic_")
            || n.starts_with("s_buffer_store_")
            || n.starts_with("s_buffer_atomic_")
    });
    // Whether the instruction writes data share (LDS, or GDS with the `gds`
    // flag): a `ds_write*`, `ds_*` atomic, append/consume/gws, ... — any DS
    // op but a read, swizzle or permute (as in `Lower::inst`'s `effect`).
    let ds_writes = |inst: &Inst| {
        matches!(&inst.fields, Fields::Ds { .. })
            && !(inst.name().starts_with("ds_read")
                || inst.name().starts_with("ds_swizzle")
                || inst.name().contains("permute"))
    };
    // Whether any instruction writes LDS (GDS, with the `gds` flag): then a
    // `ds_read*` can observe another lane's write at a different stage in
    // the console wave's two host halves.
    let writes_lds = p
        .blocks
        .iter()
        .flat_map(|b| &b.insts)
        .any(|inst| matches!(&inst.fields, Fields::Ds { gds: false, .. }) && ds_writes(inst));
    let lds_hazard = writes_lds.then(|| lds_write_hazards(p, &ds_writes));
    let writes_gds = p
        .blocks
        .iter()
        .flat_map(|b| &b.insts)
        .any(|inst| matches!(&inst.fields, Fields::Ds { gds: true, .. }) && ds_writes(inst));
    // The traced address of every scalar and `global_*` memory access.
    let addr_of: HashMap<Site, Address> = x.res.scalar.iter().map(|&(site, a)| (site, a)).collect();
    // Whether the access at `site` reads the code's own table: the only
    // binding that stays read-only while the shader performs stores
    // (`writes_global`) — a root SRV and a root UAV can bind the same
    // title-memory address, so a root descriptor is not one. Every dword a
    // `global_load_dword*` at that address loads is covered by the table.
    let is_code = |site: Site| matches!(addr_of.get(&site), Some(Address { param: CODE, .. }));
    let mut tid_aliases: HashSet<Value> = wave_tid.into_iter().collect();
    for block in &p.blocks {
        for inst in &block.insts {
            if inst.name() == "v_mov_b32"
                && let Fields::Valu(v) = &inst.fields
                && v.dpp.is_none()
                && v.sdwa.is_none()
                && let Some(&Operand::Vector { reg, .. }) = v.src.first()
                && inst.read(Reg::V(reg)).is_some_and(|x| tid_aliases.contains(&x))
            {
                tid_aliases.extend(
                    inst.writes
                        .iter()
                        .filter(|w| matches!(w.0, Reg::V(_)))
                        .map(|w| w.1),
                );
            }
        }
    }
    let mut same: HashSet<Value> = entry_tids.iter().copied().collect();
    // VGPR values no store the shader performs can have touched, grown only
    // by immutable sources: constants, [`same_s`] scalars, pure VALU ops over
    // those, and code-table (`is_code`) global loads — never by a
    // `buffer_load_*`, `ds_read*` or other global load, whose result can
    // differ between the console wave's two host halves where one of them
    // observes a store the other has not. A code-table load's VGPR address
    // words must be in here when the shader writes memory (`writes_global`).
    let mut pure: HashSet<Value> = entry_tids.iter().copied().collect();
    // The instruction that writes each SCC value. An [`Atom::Uniform`] is a
    // scalar compare's result: one value per host wave, but the console
    // wave's two halves can compute different scalars (an `s_load_dword`
    // can see different stages of a UAV update in them), so the atom holds
    // in every lane of the console wave only where its instruction reads
    // values proven the same in every lane of it ([`same_s`], or `same` for
    // a VGPR); an unproven read, or the carry itself, is not.
    let scc_def: HashMap<Value, (usize, usize)> = p
        .blocks
        .iter()
        .enumerate()
        .flat_map(|(b, block)| {
            block.insts.iter().enumerate().flat_map(move |(i, inst)| {
                inst.writes
                    .iter()
                    .filter(|(r, _)| *r == Reg::Scc)
                    .map(move |(_, v)| (*v, (b, i)))
            })
        })
        .collect();
    loop {
        let alike_atom = |same: &HashSet<Value>, same_s: &HashSet<Value>, a: &Atom| match *a {
            Atom::Uniform(v) => scc_def.get(&v).is_some_and(|&(b, i)| {
                p.blocks[b].insts[i].reads.iter().all(|(r, rv)| match *r {
                    Reg::S(_) => !per_lane(*rv) && same_s.contains(rv),
                    Reg::V(_) => !per_lane(*rv) && same.contains(rv),
                    Reg::Scc => false,
                })
            }),
            Atom::Lane(v) => same.contains(&v),
            _ => false,
        };
        let alike_bool = |same: &HashSet<Value>, same_s: &HashSet<Value>, b: Bool| {
            st.bools.support(b).iter().all(|a| alike_atom(same, same_s, a))
        };
        // The same, but proven from pure values only: a `buffer_load_*` or
        // `ds_read*` result in `same` can still differ between the console
        // wave's two host halves, so it proves nothing about what both
        // halves compute alike.
        let pure_atom = |pure: &HashSet<Value>, same_s: &HashSet<Value>, a: &Atom| match *a {
            Atom::Uniform(v) => scc_def.get(&v).is_some_and(|&(b, i)| {
                p.blocks[b].insts[i].reads.iter().all(|(r, rv)| match *r {
                    Reg::S(_) => !per_lane(*rv) && same_s.contains(rv),
                    Reg::V(_) => !per_lane(*rv) && pure.contains(rv),
                    Reg::Scc => false,
                })
            }),
            Atom::Lane(v) => pure.contains(&v),
            _ => false,
        };
        let pure_bool = |pure: &HashSet<Value>, same_s: &HashSet<Value>, b: Bool| {
            st.bools.support(b).iter().all(|a| pure_atom(pure, same_s, a))
        };
        // Blocks every lane of a console wave reaches together.
        let wave_all: Vec<bool> = match &regions {
            Some(regions) => {
                let mut w = vec![true; st.exits.len()];
                for (d, region) in regions {
                    let LaneExit::Branch { when, .. } = st.exits[*d] else {
                        continue;
                    };
                    if alike_bool(&same, &same_s, when) {
                        continue;
                    }
                    for (b, r) in region.iter().enumerate() {
                        if *r {
                            w[b] = false;
                        }
                    }
                }
                w
            }
            None => vec![false; st.exits.len()],
        };
        // Blocks every lane of both halves of a console wave reaches
        // together: only a branch whose condition is proven from pure
        // values cannot have split them between the halves.
        let pure_wave_all: Vec<bool> = match &regions {
            Some(regions) => {
                let mut w = vec![true; st.exits.len()];
                for (d, region) in regions {
                    let LaneExit::Branch { when, .. } = st.exits[*d] else {
                        continue;
                    };
                    if pure_bool(&pure, &same_s, when) {
                        continue;
                    }
                    for (b, r) in region.iter().enumerate() {
                        if *r {
                            w[b] = false;
                        }
                    }
                }
                w
            }
            None => vec![false; st.exits.len()],
        };
        let mut next: HashSet<Value> = entry_tids.iter().copied().collect();
        let mut next_s: HashSet<Value> = HashSet::new();
        let mut next_pure: HashSet<Value> = entry_tids.iter().copied().collect();
        for b in crate::ir::reverse_postorder(&p.blocks) {
            let block = &p.blocks[b];
            let whole = block
                .preds
                .iter()
                .all(|&q| wave_all.get(q).copied().unwrap_or(false));
            let whole_pure = block
                .preds
                .iter()
                .all(|&q| pure_wave_all.get(q).copied().unwrap_or(false));
            for phi in &block.phis {
                // Where the shader writes memory, a phi also needs every
                // predecessor reached through branches proven from pure
                // values: a branch on a merely `same` (load-derived)
                // condition may have sent the console wave's two halves
                // down different arms, and the phi would then pick a
                // different value in each of them.
                if matches!(phi.reg, Reg::V(_))
                    && whole
                    && (!writes_global || whole_pure)
                    && phi
                        .args
                        .iter()
                        .all(|a| next.contains(a) || same.contains(a))
                {
                    next.insert(phi.value);
                }
                // A phi of pure values is pure too, where every lane of
                // both halves reaches the block together: a branch on a
                // merely `same` (load-derived) condition may have split the
                // halves, and the phi would then pick a different constant
                // in each of them.
                if matches!(phi.reg, Reg::V(_))
                    && whole_pure
                    && phi
                        .args
                        .iter()
                        .all(|a| next_pure.contains(a) || pure.contains(a))
                {
                    next_pure.insert(phi.value);
                }
            }
            let known = |next: &HashSet<Value>, v: &Value| next.contains(v) || same.contains(v);
            let known_s =
                |next_s: &HashSet<Value>, v: &Value| next_s.contains(v) || same_s.contains(v);
            let pure_known = |next_pure: &HashSet<Value>, v: &Value| {
                next_pure.contains(v) || pure.contains(v)
            };
            // Whether every register `inst` reads is positively proven
            // uniform — a VGPR word in `same` (`next`), an SGPR word in
            // [`same_s`] (`next_s`), never a waterfall read or the carry —
            // except its own old destination, and only where that old
            // destination is read as nothing else: a full-dword overwrite
            // (not a `_d16` load, which keeps half of the old destination
            // as live data) that no guard can leave half-preserved (the
            // guard above already requires the old value alike where
            // lanes can be inactive) and that is not one of the
            // instruction's explicit source operands — an overlapping
            // `buffer_load_dword v1, v1, ...` reads its destination as
            // the address. Also except EXEC, the mask the guard tests.
            // A memory instruction's address, descriptor and offset operands
            // are reads like any other; an unproven one (an `s_load_dword`'s
            // result, say) is unknown.
            let reads_alike =
                |inst: &Inst, next: &HashSet<Value>, next_s: &HashSet<Value>, guarded: bool| {
                    let dst_regs: Vec<Reg> = inst.writes.iter().map(|w| w.0).collect();
                    let src_regs = src_vgprs(inst);
                    // A `_d16` load writes only half of each destination dword
                    // and keeps the old other half as data (see `Lower::inst`).
                    let full_dword = !inst.name().contains("_d16");
                    inst.reads.iter().all(|(r, v)| match *r {
                        Reg::V(_)
                            if !guarded
                                && full_dword
                                && dst_regs.contains(r)
                                && src_regs.as_ref().is_some_and(|s| !s.contains(r)) =>
                        {
                            true
                        }
                        Reg::S(code) if code == EXEC || code == EXEC + 1 => true,
                        Reg::V(_) => !per_lane(*v) && known(next, v),
                        Reg::S(_) => !per_lane(*v) && known_s(next_s, v),
                        Reg::Scc => false,
                    })
                };
            for (i, inst) in block.insts.iter().enumerate() {
                let name = inst.name();
                // Under a condition: only one every lane holds alike, and
                // over an old value that is alike too. A write under a
                // condition proven from pure values over an old pure value
                // is pure as well: a merely `same` (load-derived) guard
                // can differ between the console wave's two halves, and
                // the write would then happen in one half and not the
                // other.
                let mut pure_ok = true;
                if let Some(g) = st.guards[b][i] {
                    if !alike_bool(&same, &same_s, g) {
                        continue;
                    }
                    let old_alike = inst
                        .writes
                        .iter()
                        .filter(|w| matches!(w.0, Reg::V(_)))
                        .all(|w| inst.read(w.0).is_none_or(|old| known(&next, &old)));
                    if !old_alike {
                        continue;
                    }
                    pure_ok = pure_bool(&pure, &same_s, g)
                        && inst
                            .writes
                            .iter()
                            .filter(|w| matches!(w.0, Reg::V(_)))
                            .all(|w| {
                                inst.read(w.0).is_none_or(|old| pure_known(&next_pure, &old))
                            });
                }
                // Where the shader writes memory, a guarded write also needs
                // a pure guard over a pure old value to count as `same`: a
                // merely `same` (load-derived) guard or old value can
                // differ between the console wave's two host halves (one
                // can observe a store the other has not), and the write
                // would then happen in one half and not the other.
                if writes_global && !pure_ok {
                    continue;
                }
                // A whitelisted scalar op over values every lane holds alike
                // (and constants) writes a scalar every lane holds alike. A
                // missing read or an unwhitelisted def is unknown, not alike:
                // an `s_mov_b32` copy of a per-lane read is a new value that
                // carries its per-lane data.
                if SCALAR_ALOP.contains(&name) {
                    let reads_alike = inst.reads.iter().all(|(r, v)| match *r {
                        Reg::S(_) => !per_lane(*v) && known_s(&next_s, v),
                        Reg::V(_) => !per_lane(*v) && known(&next, v),
                        Reg::Scc => false,
                    });
                    if reads_alike {
                        next_s.extend(
                            inst.writes
                                .iter()
                                .filter(|w| matches!(w.0, Reg::S(_)))
                                .map(|w| w.1),
                        );
                    }
                }
                // A buffer load reads one value in every lane only where no
                // store or atomic the shader performs can alias the loaded
                // buffer (`!writes_global`, or `!writes_lds` for an LDS load
                // with the `lds` flag): where one can, the console wave's
                // two host waves can reach the load on opposite sides of
                // another lane's store and read different values. Every
                // register it reads must also be positively proven uniform:
                // the VGPR index/offset words in `same`, the SGPR descriptor
                // (`srsrc`, all four dwords) and scalar offset (`soffset`)
                // words in [`same_s`] — an unproven read (an `s_load_dword`'s
                // result, a waterfall dword) is unknown, so a buffer offset
                // derived from an `s_load_dword` is not uniform.
                if let Fields::Buffer { lds, .. } = &inst.fields
                    && (name.starts_with("buffer_load") || name.starts_with("tbuffer_load"))
                {
                    let mem_written = if *lds { writes_lds } else { writes_global };
                    let operands_alike =
                        reads_alike(inst, &next, &next_s, st.guards[b][i].is_some());
                    if !mem_written && operands_alike {
                        next.extend(
                            inst.writes
                                .iter()
                                .filter(|w| matches!(w.0, Reg::V(_)))
                                .map(|w| w.1),
                        );
                    }
                    continue;
                }
                // An image load or sample reads one value in every lane only
                // where no store can alias it and its operands are alike.
                if let Fields::Image { .. } = &inst.fields
                    && (name.starts_with("image_load")
                        || name.starts_with("image_sample")
                        || name.starts_with("image_gather"))
                {
                    let operands_alike =
                        reads_alike(inst, &next, &next_s, st.guards[b][i].is_some());
                    if !writes_global && operands_alike {
                        next.extend(
                            inst.writes
                                .iter()
                                .filter(|w| matches!(w.0, Reg::V(_)))
                                .map(|w| w.1),
                        );
                    }
                    continue;
                }
                // An LDS read reads one value in every lane where no
                // data-share write can race with it without an intervening
                // `s_barrier` (either before or after the read, or `!writes_gds`
                // for a GDS read), and where its address (and any other
                // register it reads) is proven uniform.
                if let Fields::Ds {
                    dst: Some(_),
                    data,
                    gds,
                    ..
                } = &inst.fields
                    && name.starts_with("ds_read")
                    && data.is_empty()
                {
                    let mem_written = if *gds {
                        writes_gds
                    } else {
                        lds_hazard.as_ref().is_some_and(|h| h[b][i])
                    };
                    let operands_alike =
                        reads_alike(inst, &next, &next_s, st.guards[b][i].is_some());
                    if !mem_written && operands_alike {
                        next.extend(
                            inst.writes
                                .iter()
                                .filter(|w| matches!(w.0, Reg::V(_)))
                                .map(|w| w.1),
                        );
                    }
                    continue;
                }
                // A read-only global load of dwords (not an atomic, and no
                // glc/slc: one value in every lane only where no per-lane
                // write can be observed) at an alike address reads one value
                // in every lane: an SGPR base pair plus an alike VGPR offset,
                // or an alike VGPR pair. It must also run where every lane of
                // the console wave runs it together (a lane-dependent guard
                // leaves the other lanes' copy of its result stale), equal
                // addresses alone do not prove equal results where memory can
                // be written, so the load must also be one no lane of this
                // shader can write to: the shader performs no store or atomic
                // that could alias it (`writes_global`), or its address
                // resolves to the code's own immutable table (`is_code`),
                // which covers every dword it loads. Both scalar base words
                // need positive proof of uniformity: a missing read or an
                // unwhitelisted def is unknown.
                if let Fields::Flat {
                    addr: Some(Operand::Vector { reg, .. }),
                    saddr,
                    lds,
                    cache,
                    ..
                } = &inst.fields
                    && matches!(
                        name,
                        "global_load_dword"
                            | "global_load_dwordx2"
                            | "global_load_dwordx3"
                            | "global_load_dwordx4"
                    )
                    && !lds
                    && !cache.glc
                    && !cache.slc
                {
                    let scalar = |code: u8| {
                        inst.read(Reg::S(code))
                            .is_some_and(|v| known_s(&next_s, &v))
                    };
                    // Where the shader writes memory, the VGPR address words
                    // must be pure: a `buffer_load_*` or `ds_read*`
                    // result in `same` can differ between the two host halves
                    // where one of them observes a store the other has not,
                    // and each half would then read a different code-table
                    // entry.
                    let vector = |reg: u8| {
                        inst.read(Reg::V(reg)).is_some_and(|v| {
                            known(&next, &v)
                                && !per_lane(v)
                                && (!writes_global || pure_known(&next_pure, &v))
                        })
                    };
                    let vector_pure = |reg: u8| {
                        inst.read(Reg::V(reg))
                            .is_some_and(|v| pure_known(&next_pure, &v) && !per_lane(v))
                    };
                    let alike_addr = match saddr {
                        Some(Operand::Scalar { code, .. }) if *code != NULL => {
                            scalar(*code) && scalar(*code + 1) && vector(*reg)
                        }
                        _ => vector(*reg) && vector(*reg + 1),
                    };
                    let code = is_code((b, i));
                    if alike_addr && wave_all[b] && (!writes_global || (code && pure_ok)) {
                        next.extend(
                            inst.writes
                                .iter()
                                .filter(|w| matches!(w.0, Reg::V(_)))
                                .map(|w| w.1),
                        );
                        // The code's own table is immutable, so the load's
                        // result is pure where its address words are pure.
                        let pure_addr = match saddr {
                            Some(Operand::Scalar { code, .. }) if *code != NULL => {
                                scalar(*code) && scalar(*code + 1) && vector_pure(*reg)
                            }
                            _ => vector_pure(*reg) && vector_pure(*reg + 1),
                        };
                        if code && pure_addr && wave_all[b] && pure_ok {
                            next_pure.extend(
                                inst.writes
                                    .iter()
                                    .filter(|w| matches!(w.0, Reg::V(_)))
                                    .map(|w| w.1),
                            );
                        }
                    }
                    continue;
                }
                let ops: Vec<Operand> = match &inst.fields {
                    Fields::Valu(v)
                        if v.dpp.is_none()
                            && v.sdwa.is_none()
                            && !name.starts_with("v_mbcnt")
                            && !name.starts_with("v_cmpx")
                            && !name.contains("lane") =>
                    {
                        v.src.clone()
                    }
                    _ => continue,
                };
                // The thread id's x shifted/masked past the console wave: one
                // value per console wave, in both halves: pure too.
                let inline_u32 = |op: &Operand| match *op {
                    Operand::Const(k @ 128..=192) => Some(u32::from(k - 128)),
                    Operand::Const(k @ 193..=208) => Some((192i32 - i32::from(k)) as u32),
                    Operand::Literal(l) => Some(l),
                    _ => None,
                };
                if !tid_aliases.is_empty() {
                    let is_tid = |op: &Operand| {
                        matches!(*op, Operand::Vector { reg, .. } if inst.read(Reg::V(reg)).is_some_and(|x| tid_aliases.contains(&x)))
                    };
                    let wave_derived = match (name, ops.as_slice()) {
                        ("v_lshrrev_b32", [shift, src, ..])
                            if is_tid(src) && inline_u32(shift).is_some_and(|s| (s & 31) >= 6) =>
                        {
                            true
                        }
                        ("v_and_b32", [a, b, ..])
                            if (is_tid(a) && inline_u32(b).is_some_and(|m| m & 63 == 0))
                                || (is_tid(b) && inline_u32(a).is_some_and(|m| m & 63 == 0)) =>
                        {
                            true
                        }
                        ("v_bfe_u32", [src, start, ..])
                            if is_tid(src) && inline_u32(start).is_some_and(|s| (s & 31) >= 6) =>
                        {
                            true
                        }
                        _ => false,
                    };
                    if wave_derived {
                        next.extend(inst.writes.iter().filter(|w| w.0 != Reg::Scc).map(|w| w.1));
                        if pure_ok {
                            next_pure.extend(
                                inst.writes.iter().filter(|w| w.0 != Reg::Scc).map(|w| w.1),
                            );
                        }
                        continue;
                    }
                }
                // A full-dword overwrite that never consumes the old
                // destination as data or preserves sub-dword lanes (excluding
                // e.g. `v_swap_b32`, `v_dot4c_*`, `v_fmac_*`, and partial
                // 16-bit writes like `v_cvt_f16_f32`). IR records the old
                // destination in `inst.reads` for inactive-lane preservation,
                // so only such an overwrite may skip requiring it.
                let full_overwrite = matches!(
                    name,
                    "v_mov_b32"
                        | "v_add_nc_u32"
                        | "v_sub_nc_u32"
                        | "v_subrev_nc_u32"
                        | "v_and_b32"
                        | "v_or_b32"
                        | "v_xor_b32"
                        | "v_lshlrev_b32"
                        | "v_lshrrev_b32"
                        | "v_ashrrev_i32"
                        | "v_lshlrev_b64"
                        | "v_lshrrev_b64"
                        | "v_ashrrev_i64"
                        | "v_mul_lo_u32"
                        | "v_mul_hi_u32"
                        | "v_mul_u32_u24"
                        | "v_mad_u32_u24"
                        | "v_mad_u64_u32"
                        | "v_lshl_add_u32"
                        | "v_add3_u32"
                        | "v_lshl_or_b32"
                        | "v_and_or_b32"
                        | "v_bfe_u32"
                        | "v_bfe_i32"
                        | "v_bfm_b32"
                        | "v_bfi_b32"
                        | "v_cndmask_b32"
                );
                let src_regs: Vec<Reg> = ops
                    .iter()
                    .flat_map(|op| match *op {
                        Operand::Vector { reg, dwords } => {
                            (0..dwords).map(move |k| Reg::V(reg + k)).collect::<Vec<_>>()
                        }
                        _ => Vec::new(),
                    })
                    .collect();
                // The select mask of a `v_cndmask_b32` or the carry-in of a
                // `*_co_ci_u32` (the third source) is a 64-bit lane mask read
                // per lane: bit `lane_id` of it, not the mask as one value.
                let mask_op = if name == "v_cndmask_b32" || name.contains("_co_ci_") {
                    ops.get(2)
                } else {
                    None
                };
                let guarded = st.guards[b][i].is_some();
                let alike = ops.iter().all(|op| match *op {
                    Operand::Vector { reg, .. } => {
                        // every dword the instruction reads from that register on
                        let read: Vec<Value> = inst
                            .reads
                            .iter()
                            .filter(|r| matches!(r.0, Reg::V(x) if x >= reg))
                            .map(|r| r.1)
                            .collect();
                        !read.is_empty() && read.iter().all(|x| known(&next, x))
                    }
                    // A scalar operand needs positive proof of uniformity
                    // ([`same_s`]): an unproven read (an `s_load_dword`'s
                    // result, a waterfall dword) is unknown. A lane mask is
                    // not one value per lane, unless a compare of alike
                    // values made it. EXEC is the mask the guard tests.
                    Operand::Scalar { code, .. } if code != NULL => {
                        if code == EXEC || code == EXEC + 1 {
                            true
                        } else {
                            inst.read(Reg::S(code)).is_none_or(|x| {
                                if is_mask(x) {
                                    known(&next, &x)
                                } else {
                                    !per_lane(x) && known_s(&next_s, &x)
                                }
                            })
                        }
                    }
                    _ => true,
                }) && mask_op.is_none_or(|op| {
                    // The mask's per-lane bit must be the same in every lane,
                    // which its scalar value being one value does not prove.
                    mask_bit_alike(st, inst, op, |v| known(&next, v), |b| {
                        alike_bool(&same, &same_s, b)
                    })
                }) && inst.reads.iter().all(|(r, v)| match *r {
                    // The old destination, read for inactive-lane preservation
                    // or consumed as an accumulator, must be alike too — except
                    // where a full overwrite replaces it and no guard can
                    // leave lanes of the block inactive.
                    Reg::V(_)
                        if full_overwrite
                            && !guarded
                            && inst.write(*r).is_some()
                            && !src_regs.contains(r) =>
                    {
                        true
                    }
                    Reg::V(_) => known(&next, v),
                    // EXEC is the mask the guard tests.
                    Reg::S(code) if code == EXEC || code == EXEC + 1 => true,
                    // A scalar read needs positive proof of uniformity
                    // ([`same_s`]): an unproven read (an `s_load_dword`'s
                    // result, a waterfall dword) is unknown. A lane mask
                    // (a 64-bit operand's high dword is a read of its own)
                    // is not one value unless a compare of alike values
                    // made it.
                    Reg::S(_) => {
                        if is_mask(*v) {
                            known(&next, v)
                        } else {
                            !per_lane(*v) && known_s(&next_s, v)
                        }
                    }
                    // The carry is one value per host wave, not per console
                    // wave: the halves can hold different SCCs (an
                    // `s_load_dword` that aliases a store reads different
                    // stages of it in them), so `src_scc` proves nothing.
                    Reg::Scc => false,
                });
                if alike {
                    next.extend(inst.writes.iter().filter(|w| w.0 != Reg::Scc).map(|w| w.1));
                }
                // A pure VALU op — every SSA read of it is pure: each VGPR
                // word in `pure`, each SGPR word in [`same_s`] (a 64-bit
                // operand's high word too), never the carry (lowering reads
                // the live SCC for `src_scc`) nor a lane mask as a source.
                // Inline constants and literals are not reads; EXEC is the
                // mask the guard tests. IR records the old destination in
                // `inst.reads` for inactive-lane preservation, so exempt it
                // ONLY for the full-dword overwrites whitelisted above.
                let pure_alike = mask_op.is_none_or(|op| {
                    // The lane mask's per-lane bit must be proven from pure
                    // values too, not merely from `same` ones.
                    mask_bit_alike(st, inst, op, |v| pure_known(&next_pure, v), |b| {
                        pure_bool(&pure, &same_s, b)
                    })
                }) && !ops
                    .iter()
                    .any(|op| matches!(op, Operand::Const(251..=254)))
                    && inst.reads.iter().all(|(r, v)| match *r {
                        Reg::V(_)
                            if full_overwrite
                                && inst.write(*r).is_some()
                                && !src_regs.contains(r) =>
                        {
                            true
                        }
                        Reg::S(code) if code == EXEC || code == EXEC + 1 => true,
                        Reg::V(_) => !per_lane(*v) && pure_known(&next_pure, v),
                        Reg::S(_) => !per_lane(*v) && known_s(&next_s, v),
                        Reg::Scc => false,
                    });
                if alike && pure_ok && pure_alike {
                    next_pure.extend(inst.writes.iter().filter(|w| w.0 != Reg::Scc).map(|w| w.1));
                }
            }
        }
        if next.is_subset(&same) && next_s.is_subset(&same_s) && next_pure.is_subset(&pure) {
            return same;
        }
        same.extend(next);
        same_s.extend(next_s);
        pure.extend(next_pure);
    }
}

/// The VGPR dwords `inst` names as explicit source operands — its address
/// words and, where its data operand is a source (a store or an atomic),
/// those too — or `None` where its fields do not say. A written VGPR among
/// them, as in an overlapping `buffer_load_dword v1, v1, ...`, is read as
/// an operand (the address), not kept for inactive lanes alone.
fn src_vgprs(inst: &Inst) -> Option<Vec<Reg>> {
    let mut out = Vec::new();
    let mut push = |o: &Operand| {
        if let Operand::Vector { reg, dwords } = *o {
            out.extend((0..dwords).map(|k| Reg::V(reg + k)));
        }
    };
    match &inst.fields {
        Fields::Buffer {
            data,
            addr,
            offen,
            idxen,
            ..
        } => {
            // IR reads the address register only with `offen` or `idxen`.
            if *offen || *idxen {
                push(addr);
            }
            if !inst.name().contains("_load_")
                && let Some(d) = data
            {
                push(d);
            }
        }
        Fields::Ds { addr, data, .. } => {
            if let Some(a) = addr {
                push(a);
            }
            for d in data {
                push(d);
            }
        }
        Fields::Flat { addr, data, .. } => {
            if let Some(a) = addr {
                push(a);
            }
            if let Some(d) = data {
                push(d);
            }
        }
        _ => return None,
    }
    Some(out)
}

/// Whether the lane-mask operand `op` of `inst` — the select mask of a
/// `v_cndmask_b32` or the carry-in of a `*_co_ci_u32` — has the same bit in
/// every lane of the console wave. The mask is read per lane (bit `lane_id`
/// of the 64-bit value), so its scalar value being one value in every lane
/// does not make its per-lane bit so: an inline `1`, or a uniform SGPR mask,
/// selects different sources for lane 0 than for lanes 1..63. Proven by a
/// literal all-zero or all-ones mask, by the mask's own per-lane Bool
/// (`bool_ok` — a compare or carry of alike values made it), or by every
/// dword of the mask being alike (`known`) itself.
fn mask_bit_alike(
    st: &Structured,
    inst: &Inst,
    op: &Operand,
    known: impl Fn(&Value) -> bool,
    bool_ok: impl Fn(Bool) -> bool,
) -> bool {
    match *op {
        Operand::Const(k) => match crate::resource::inline(k) {
            Some(x) => {
                let hi = if (x as i32) < 0 { u32::MAX } else { 0 };
                (x == 0 && hi == 0) || (x == u32::MAX && hi == u32::MAX)
            }
            // The special inline constants (`vccz`, `execz`, `scc`, ...) are
            // one-bit flags, not all-zero or all-ones 64-bit lane masks:
            // `vccz` is `0` or `1`, and `1` selects only lane 0.
            None => false,
        },
        Operand::Literal(x) => x == 0 || x == u32::MAX,
        Operand::Scalar { code, dwords } if code != NULL => {
            let Some(lo) = inst.read(Reg::S(code)) else {
                return false;
            };
            let hi = if dwords > 1 {
                inst.read(Reg::S(code + 1))
            } else {
                None
            };
            st.mask_bits(lo, hi).is_some_and(bool_ok)
                || (known(&lo) && hi.is_none_or(|h| known(&h)))
        }
        _ => false,
    }
}

/// For each instruction `(b, i)`, whether an LDS write can reach `(b, i)`
/// without an intervening `s_barrier` (RAW hazard) or be reached from `(b, i)`
/// without an intervening `s_barrier` (WAR hazard).
///
/// When both are false for a `ds_read*` at a uniform address, all waves in the
/// workgroup observe the same LDS contents even when the console wave runs as
/// two host wave32 halves.
fn lds_write_hazards(p: &Program, ds_writes: &impl Fn(&Inst) -> bool) -> Vec<Vec<bool>> {
    let n = p.blocks.len();
    let is_lds_write = |inst: &Inst| {
        matches!(&inst.fields, Fields::Ds { gds: false, .. }) && ds_writes(inst)
    };

    let mut before_in = vec![false; n];
    let mut before_out = vec![false; n];
    loop {
        let mut changed = false;
        for b in 0..n {
            let incoming = p.blocks[b].preds.iter().any(|&pred| before_out[pred]);
            before_in[b] = incoming;
            let mut dirty = incoming;
            for inst in &p.blocks[b].insts {
                if inst.name() == "s_barrier" {
                    dirty = false;
                } else if is_lds_write(inst) {
                    dirty = true;
                }
            }
            if dirty != before_out[b] {
                before_out[b] = dirty;
                changed = true;
            }
        }
        if !changed {
            break;
        }
    }

    let mut after_in = vec![false; n];
    let mut after_out = vec![false; n];
    loop {
        let mut changed = false;
        for b in (0..n).rev() {
            let outgoing = p.blocks[b]
                .exit
                .successors()
                .iter()
                .any(|&succ| after_in[succ]);
            after_out[b] = outgoing;
            let mut dirty = outgoing;
            for inst in p.blocks[b].insts.iter().rev() {
                if inst.name() == "s_barrier" {
                    dirty = false;
                } else if is_lds_write(inst) {
                    dirty = true;
                }
            }
            if dirty != after_in[b] {
                after_in[b] = dirty;
                changed = true;
            }
        }
        if !changed {
            break;
        }
    }

    let mut hazard: Vec<Vec<bool>> = p
        .blocks
        .iter()
        .map(|blk| vec![false; blk.insts.len()])
        .collect();
    for b in 0..n {
        let mut dirty = before_in[b];
        for (i, inst) in p.blocks[b].insts.iter().enumerate() {
            hazard[b][i] = dirty;
            if inst.name() == "s_barrier" {
                dirty = false;
            } else if is_lds_write(inst) {
                dirty = true;
            }
        }
        let mut dirty = after_out[b];
        for (i, inst) in p.blocks[b].insts.iter().enumerate().rev() {
            hazard[b][i] |= dirty;
            if inst.name() == "s_barrier" {
                dirty = false;
            } else if is_lds_write(inst) {
                dirty = true;
            }
        }
    }
    hazard
}

/// Scalar ops that write what they read, one value in every lane of a console
/// wave: copies and arithmetic over scalars and constants. A memory read
/// (`s_load_*`, `s_buffer_load*`) is not one: the console wave's two host
/// waves can observe another lane's store between them. A lane mask read is
/// not one either: only a compare of alike values makes a mask alike, and
/// compares write masks, not SGPRs.
const SCALAR_ALOP: &[&str] = &[
    "s_mov_b32",
    "s_movk_i32",
    "s_add_u32",
    "s_add_i32",
    "s_sub_u32",
    "s_mul_i32",
    "s_mul_hi_i32",
    "s_mul_hi_u32",
    "s_and_b32",
    "s_or_b32",
    "s_xor_b32",
    "s_not_b32",
    "s_lshl_b32",
    "s_lshr_b32",
    "s_ashr_i32",
    "s_min_i32",
    "s_min_u32",
    "s_max_i32",
    "s_max_u32",
    "s_abs_i32",
    "s_getpc_b64",
];
