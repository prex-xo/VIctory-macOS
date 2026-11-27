//! The 64-bit image atomics emulated under a lock word (`emit`, "Atomic locks").

use super::*;

impl<'a> Lower<'a> {
    /// A 64-bit image atomic `op` of `value` (cmpswap: replacing
    /// `compare`) on the texel at `cs` of the globally coherent R32G32_UINT
    /// view `h`, under a lock word (`emit`, "Atomic locks"). The wave's
    /// lanes take turns in a loop over lane indices that every lane runs
    /// alike: in its turn a lane takes its texel's lock, reads, writes and
    /// frees it. Lanes of one wave never wait for each other's locks (that
    /// never finishes on Apple GPUs), and the loop needs no wave operation
    /// in divergent code (D3DMetal's return other lanes' values there).
    /// With [`Parts::atomic_lock_rounds`] the lanes first try their locks
    /// in rounds, all at once, and only the lanes still not done take
    /// turns. `not_rep` marks the lanes a reduction before this made
    /// members of a representative's group, done from the start.
    /// Returns the old value's low and high dwords.
    #[allow(clippy::too_many_arguments)]
    pub(super) fn locked_atomic64(
        &mut self,
        h: V,
        cs: &[V],
        coords: usize,
        op: &str,
        value: V,
        compare: Option<V>,
        not_rep: V,
    ) -> (V, V) {
        let i32t = self.s.t.i32;
        let i64t = self.s.m.int(64);
        let void = self.s.t.void;
        let k = |l: &mut Self, v: u32| l.s.m.i32(v);
        self.s.flags |= flags::WAVE_OPS | flags::TYPED_UAV_LOADS | flags::INT64;
        let c = self.cur;
        // The lock: a hash of the coordinates.
        let mut hash = k(self, 0);
        for (&x, m) in cs
            .iter()
            .take(coords)
            .zip([0x9E37_79B1u32, 0x85EB_CA77, 0xC2B2_AE3D])
        {
            let m = k(self, m);
            let x = self.s.m.bin(c, Bin::Mul, x, m);
            hash = self.s.m.bin(c, Bin::Xor, hash, x);
            let k15 = k(self, 15);
            let r = self.s.m.bin(c, Bin::LShr, hash, k15);
            hash = self.s.m.bin(c, Bin::Xor, hash, r);
        }
        let mask = k(self, crate::emit::ATOMIC_LOCKS - 1);
        let slot = self.s.m.bin(c, Bin::And, hash, mask);
        let k2 = k(self, 2);
        let slot = self.s.m.bin(c, Bin::Shl, slot, k2);
        let base = k(self, crate::emit::ATOMIC_LOCKS_AT);
        let lock = self.s.m.bin(c, Bin::Add, slot, base);
        let locks = self.s.view_writes_handle();
        let me = self
            .s
            .op(c, "waveGetLaneIndex", Attr::ReadNone, i32t, 111, &[]);
        let lanes = self
            .s
            .op(c, "waveGetLaneCount", Attr::ReadNone, i32t, 112, &[]);
        let iu = self.s.m.undef(i32t);
        let (zero, one) = (k(self, 0), k(self, 1));
        // The global unordered-access fence, with the group's where the
        // stage has groups: DXIL validation refuses mode 10 in a pixel
        // shader (found by evan-kolberg, #240).
        let fence_mode = if matches!(self.x.native.stage, HwStage::Cs | HwStage::Ms) {
            2 | 8
        } else {
            2
        };
        let fence = k(self, fence_mode);
        // Under the lock, in block `b`: fence, read, write, fence, free (only
        // if `won`: past the bounded spin a lane goes on without it). The
        // old value's low and high dwords.
        let step = |l: &mut Self, b: usize, won: V| -> (V, V) {
            l.s.op(b, "barrier", Attr::NoDuplicate, void, 80, &[fence]);
            let mut args = vec![h, iu];
            args.extend(cs);
            args.extend([iu, iu, iu]);
            let r = l.s.op(
                b,
                "textureLoad.i32",
                Attr::ReadOnly,
                l.s.t.res_ret_i32,
                66,
                &args,
            );
            let lo = l.s.m.extract(b, r, 0);
            let hi = l.s.m.extract(b, r, 1);
            let l64 = l.s.m.cast(b, Cast::ZExt, lo, i64t);
            let h64 = l.s.m.cast(b, Cast::ZExt, hi, i64t);
            let k32 = l.s.m.int_const(i64t, 32);
            let h64 = l.s.m.bin(b, Bin::Shl, h64, k32);
            let old = l.s.m.bin(b, Bin::Or, l64, h64);
            let m = &mut l.s.m;
            let pick = |m: &mut Module, p: u8| {
                let keep = m.cmp(b, p, old, value);
                m.select(b, keep, old, value)
            };
            let new = match op {
                "add" | "sub" => m.bin(b, Bin::Add, old, value),
                "and" => m.bin(b, Bin::And, old, value),
                "or" => m.bin(b, Bin::Or, old, value),
                "xor" => m.bin(b, Bin::Xor, old, value),
                "smin" => pick(m, pred::SLT),
                "smax" => pick(m, pred::SGT),
                "umin" => pick(m, pred::IULT),
                "umax" => pick(m, pred::IUGT),
                "swap" => value,
                _ => {
                    let same = m.cmp(b, pred::EQ, old, compare.expect("a compare value"));
                    m.select(b, same, value, old)
                }
            };
            let nlo = l.s.m.cast(b, Cast::Trunc, new, i32t);
            let nhi = l.s.m.bin(b, Bin::LShr, new, k32);
            let nhi = l.s.m.cast(b, Cast::Trunc, nhi, i32t);
            let mut args = vec![h];
            args.extend(cs);
            // DXIL stores all four channels; an R32G32 view keeps two.
            args.extend([nlo, nhi, nlo, nhi, l.s.m.i8(0xF)]);
            l.s.op(b, "textureStore.i32", Attr::NoUnwind, void, 67, &args);
            l.s.op(b, "barrier", Attr::NoDuplicate, void, 80, &[fence]);
            // One add either way: a lane that took the lock frees it (it
            // holds 1, so all ones brings it to 0), and a lane that gave up
            // counts itself at `ATOMIC_LOCKS_GAVE_UP_AT` instead.
            let add = l.s.m.i32(0);
            let count = l.s.m.i32(crate::emit::ATOMIC_LOCKS_GAVE_UP_AT);
            let at = l.s.m.select(b, won, lock, count);
            let all = l.s.m.i32(u32::MAX);
            let by = l.s.m.select(b, won, all, one);
            l.atomic_bin_op(b, &[locks, add, at, iu, iu, by]);
            (lo, hi)
        };

        // Rounds ([`Parts::atomic_lock_rounds`]), before the turns: in each
        // of a few rounds that every lane runs alike, every lane not yet done
        // tries its lock once, and the lanes that take it read, write and
        // free it in that round. No lane waits on a lock in a round, so lanes
        // of one wave cannot wait for each other's, and their locks are all
        // free again before the turns. Lanes whose locks differ finish in the
        // first round together, where the turns run them one after another.
        // Only lanes still not done take turns (a lane that met another of
        // its wave on its lock, or one held from another wave in every
        // round). In a pixel program helper lanes are done from the start:
        // their writes and atomics are discarded, so a helper could never
        // take a lock, and in its turn it would spin to the bound.
        let (c, finished, olo0, ohi0) = if self.x.atomic_lock_rounds {
            let i1 = self.s.t.i1;
            // Done from the start: a helper lane, and a lane the umax
            // reduction reduced to a representative's member.
            let start = if self.s.pixel {
                let helper = self.s.op(c, "isHelperLane.i1", Attr::ReadOnly, i1, 221, &[]);
                self.s.m.bin(c, Bin::Or, helper, not_rep)
            } else {
                not_rep
            };
            let yes = self.s.m.bool(true);
            let mut blocks = [0; 5];
            for b in &mut blocks {
                *b = self.s.m.new_block();
            }
            let [head, attempt, held, latch, done] = blocks;
            self.s.m.terminate(c, Term::Br(head));
            let round = self.s.m.phi(head, i32t);
            let was = self.s.m.phi(head, i1);
            let plo = self.s.m.phi(head, i32t);
            let phi_ = self.s.m.phi(head, i32t);
            let open = self.s.m.bin(head, Bin::Xor, was, yes);
            self.s.m.terminate(head, Term::CondBr(open, attempt, latch));
            let got = self.s.op(
                attempt,
                "atomicCompareExchange.i32",
                Attr::NoUnwind,
                i32t,
                79,
                &[locks, lock, iu, iu, zero, one],
            );
            let won = self.s.m.cmp(attempt, pred::EQ, got, zero);
            self.s.m.terminate(attempt, Term::CondBr(won, held, latch));
            let (lo, hi) = step(self, held, won);
            self.s.m.terminate(held, Term::Br(latch));
            let is = self.s.m.phi(latch, i1);
            self.s
                .m
                .set_phi(is, vec![(yes, held), (was, head), (was, attempt)]);
            let olo = self.s.m.phi(latch, i32t);
            self.s
                .m
                .set_phi(olo, vec![(lo, held), (plo, head), (plo, attempt)]);
            let ohi = self.s.m.phi(latch, i32t);
            self.s
                .m
                .set_phi(ohi, vec![(hi, held), (phi_, head), (phi_, attempt)]);
            let next = self.s.m.bin(latch, Bin::Add, round, one);
            let rounds = k(self, crate::emit::ATOMIC_LOCK_ROUNDS);
            let more = self.s.m.cmp(latch, pred::IULT, next, rounds);
            self.s.m.set_phi(round, vec![(zero, c), (next, latch)]);
            self.s.m.set_phi(was, vec![(start, c), (is, latch)]);
            self.s.m.set_phi(plo, vec![(iu, c), (olo, latch)]);
            self.s.m.set_phi(phi_, vec![(iu, c), (ohi, latch)]);
            self.s.m.terminate(latch, Term::CondBr(more, head, done));
            (done, Some((is, yes)), olo, ohi)
        } else {
            (c, None, iu, iu)
        };

        let mut blocks = [0; 5];
        for b in &mut blocks {
            *b = self.s.m.new_block();
        }
        let [head, spin, held, latch, done] = blocks;
        // After the rounds only the lanes still not done take turns, in a
        // branch the others skip: a wave whose lanes all finished in the
        // rounds runs no turn at all (on D3DMetal the turns cost a wave
        // about as much again as the rounds even when every turn is empty).
        match finished {
            Some((is, yes)) => {
                let open = self.s.m.bin(c, Bin::Xor, is, yes);
                // A ballot where every lane has left the rounds, so a lane
                // still in a round's held block (its lock not yet freed)
                // reconverges before its wave-mates' turns: without it, on
                // M1 and M2 GPUs the turns spun to the bound on a lock their
                // own wave held (cocobailey, #240). `open & any` is `open`.
                // All four words, so lanes past 31 count on wider waves;
                // `waveAnyTrue` in its place hung the rounds on an M1 Max.
                let four = self.s.m.named_struct("dx.types.fouri32", vec![i32t; 4]);
                let ballot = self
                    .s
                    .op(c, "waveActiveBallot", Attr::NoUnwind, four, 116, &[open]);
                let mut w = self.s.m.extract(c, ballot, 0);
                for i in 1..4 {
                    let wi = self.s.m.extract(c, ballot, i);
                    w = self.s.m.bin(c, Bin::Or, w, wi);
                }
                let any = self.s.m.cmp(c, pred::NE, w, zero);
                let open = self.s.m.bin(c, Bin::And, open, any);
                self.s.m.terminate(c, Term::CondBr(open, head, done));
            }
            None => self.s.m.terminate(c, Term::Br(head)),
        }

        // Head: lane `i`'s turn.
        let i = self.s.m.phi(head, i32t);
        let plo = self.s.m.phi(head, i32t);
        let phi_ = self.s.m.phi(head, i32t);
        let turn = self.s.m.cmp(head, pred::EQ, me, i);
        self.s.m.terminate(head, Term::CondBr(turn, spin, latch));

        // Spin, bounded by the stage (`atomic_lock_spins`): past the bound a
        // lane goes on without the lock, and its step counts it.
        let got = self.s.op(
            spin,
            "atomicCompareExchange.i32",
            Attr::NoUnwind,
            i32t,
            79,
            &[locks, lock, iu, iu, zero, one],
        );
        let won = self.s.m.cmp(spin, pred::EQ, got, zero);
        let tries = self.s.m.phi(spin, i32t);
        let more = self.s.m.bin(spin, Bin::Add, tries, one);
        self.s.m.set_phi(tries, vec![(zero, head), (more, spin)]);
        let limit = k(self, crate::emit::shader::atomic_lock_spins(self.x.native.stage));
        let out = self.s.m.cmp(spin, pred::IUGE, more, limit);
        let leave = self.s.m.bin(spin, Bin::Or, won, out);
        self.s.m.terminate(spin, Term::CondBr(leave, held, spin));

        // Held: fence, read, write, fence, free.
        let (lo, hi) = step(self, held, won);
        self.s.m.terminate(held, Term::Br(latch));

        // Latch: the next lane.
        let olo = self.s.m.phi(latch, i32t);
        self.s.m.set_phi(olo, vec![(lo, held), (plo, head)]);
        let ohi = self.s.m.phi(latch, i32t);
        self.s.m.set_phi(ohi, vec![(hi, held), (phi_, head)]);
        let next = self.s.m.bin(latch, Bin::Add, i, one);
        let more = self.s.m.cmp(latch, pred::IULT, next, lanes);
        self.s.m.set_phi(i, vec![(zero, c), (next, latch)]);
        self.s.m.set_phi(plo, vec![(olo0, c), (olo, latch)]);
        self.s.m.set_phi(phi_, vec![(ohi0, c), (ohi, latch)]);
        self.s.m.terminate(latch, Term::CondBr(more, head, done));
        self.cur = done;
        if finished.is_none() {
            return (olo, ohi);
        }
        // From the turns, or from the rounds for lanes that skipped them.
        let rlo = self.s.m.phi(done, i32t);
        self.s.m.set_phi(rlo, vec![(olo, latch), (olo0, c)]);
        let rhi = self.s.m.phi(done, i32t);
        self.s.m.set_phi(rhi, vec![(ohi, latch), (ohi0, c)]);
        (rlo, rhi)
    }
}
