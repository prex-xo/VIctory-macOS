//! Where group barriers can go when the host splits a compute group into
//! waves the console's lockstep did not have: the blocks every lane of the
//! group runs together ([`in_step`]), and the barriers a group that is one
//! console wave needs between its lanes' LDS accesses
//! ([`split_wave_barriers`]).

use super::*;

/// Where a compute group that is one console wave needs group barriers
/// when the host splits it into narrower waves: (block, instruction) to
/// sync before, instruction index = the block's length for its end.
///
/// The console runs such a group in lockstep, so its compiler leaves out
/// the barriers between one lane's LDS access and another's. Here a read
/// (or atomic) of LDS that a write or atomic may have changed since the last
/// barrier gets one before it, and so does a write (or atomic) after a read
/// or atomic since then, so that it cannot overwrite what another lane has
/// yet to read. (A write after a write, which the console orders too, gets
/// none.) Barriers need every lane of the group, so they go only in blocks
/// every lane runs together, the same number of times ([`in_step`]): once,
/// or in a loop on a count every host wave computes alike. Such a block that
/// leads into code only some lanes run syncs at its end where that code
/// could otherwise read a write, or overwrite a read, made before it.
///
/// `None`: a read in divergent code may see a write made in divergent code
/// since the last barrier. A write in divergent code after a read there gets
/// no barrier, as before.
#[cfg(test)]
pub(crate) fn split_wave_barriers(p: &Program, st: &Structured) -> Option<Set<(usize, usize)>> {
    let step = step_and_cap(p, st, true).0;
    split_wave_barriers_with_step(p, &step)
}

/// [`split_wave_barriers`] using a precomputed `step` vector (which may
/// include blocks legalized by acyclic predication in `halves`).
pub(crate) fn split_wave_barriers_with_step(
    p: &Program,
    step: &[bool],
) -> Option<Set<(usize, usize)>> {
    /// LDS accesses since the last barrier.
    #[derive(Clone, Copy, Default, PartialEq)]
    struct Since {
        read: bool,
        written: bool,
    }
    impl Since {
        fn or(self, o: Since) -> Since {
            Since {
                read: self.read || o.read,
                written: self.written || o.written,
            }
        }
    }
    // (reads, writes) of an LDS access; an atomic does both.
    let access = |inst: &Inst| {
        let name = inst.name();
        if !matches!(inst.fields, Fields::Ds { .. })
            || name.starts_with("ds_swizzle")
            || name.contains("permute")
        {
            return None;
        }
        Some((
            !name.starts_with("ds_write"),
            !name.starts_with("ds_read"),
        ))
    };
    let n = p.blocks.len();
    let succ = |b: usize| p.blocks[b].exit.successors();
    let divergent_writes = (0..n).any(|b| {
        !step[b]
            && p.blocks[b]
                .insts
                .iter()
                .any(|i| access(i).is_some_and(|(_, w)| w))
    });
    let order = crate::ir::reverse_postorder(&p.blocks);
    let mut out = vec![Since::default(); n];
    loop {
        let mut at = Set::default();
        let mut changed = false;
        for &b in &order {
            let mut since = p.blocks[b]
                .preds
                .iter()
                .fold(Since::default(), |a, &q| a.or(out[q]));
            for (i, inst) in p.blocks[b].insts.iter().enumerate() {
                if inst.name() == "s_barrier" {
                    since = Since::default();
                }
                let Some((reads, writes)) = access(inst) else {
                    continue;
                };
                // A read of what another lane may have written; a write over
                // what another lane may have yet to read.
                let stale = reads && since.written;
                let early = writes && since.read;
                if stale || early {
                    if step[b] {
                        at.insert((b, i));
                        since = Since::default();
                    } else if stale {
                        return None;
                    }
                }
                since = since.or(Since {
                    read: reads,
                    written: writes,
                });
            }
            if step[b]
                && succ(b).iter().any(|&s| !step[s])
                && (since.written || since.read && divergent_writes)
            {
                at.insert((b, p.blocks[b].insts.len()));
                since = Since::default();
            }
            // What reaches a block only grows, so this ends; barriers placed
            // for more than reaches it are still right.
            let since = out[b].or(since);
            changed |= std::mem::replace(&mut out[b], since) != since;
        }
        if !changed {
            return Some(at);
        }
    }
}

/// Per block, whether every lane of the group runs it together, the same
/// number of times, however the host splits the group into waves, so that a
/// group barrier can go there: outside the region of every branch whose
/// condition may differ between waves ([`halves::branch_regions`]). That is
/// a per-lane branch, or one on a scalar not computed alike in every wave.
/// Every block each lane runs once ([`once_per_lane`]) is one; so is a loop
/// whose count every wave computes alike, from the group's inputs,
/// constants and scalar loads through scalar code (not from lane masks,
/// other lanes' values or hardware state). Such a scalar is the same in
/// every console wave of the group too.
///
/// Scalars are assumed alike until shown otherwise (a loop counter needs
/// itself), and the regions and the scalars are narrowed together to a
/// fixed point. With an indirect branch, the blocks each lane runs once.
///
/// The other half is per block whether a lane may have left the program
/// early at a capped back edge's iteration limit before reaching it. Lowering
/// caps an unconditional back edge with an early return at the loop
/// iteration limit (`LOOP_LIMIT`): a lane whose count reaches it leaves the
/// program from the latch. The header's counter is shared by *every* back
/// edge to it, so the cap check holds for every lane only when the latch
/// and the header run together and so does each back edge to the header — a
/// conditional one with a uniform condition. Otherwise one wave's count can
/// reach the limit at the latch while another goes on, and that wave leaves
/// there: no block reachable from that loop is run by every lane of the
/// group, the same number of times, so those blocks are not in step either.
/// The escape reaches past the loop too, along the sibling paths of every
/// divergent branch the latch is reachable from: a lane that took the other
/// arm never passes the latch's cap, so a barrier after that branch can
/// still be reached while a lane in the loop arm has left at the cap. The
/// walk back from the latch stops at a block every lane runs together,
/// though: every lane reconverges there before the cap exit, so a divergent
/// branch earlier than that, whose arms already reconverged there, is no
/// escape.
///
/// With `need_step` false the step half is left out unless the program has
/// an `s_barrier` after a capped back edge, the only reader of the escape
/// half.
pub(crate) fn step_and_cap(
    p: &Program,
    st: &Structured,
    need_step: bool,
) -> (Vec<bool>, Vec<bool>, Set<Value>) {
    step_and_cap_with(p, st, &st.exits, &Set::default(), need_step)
}

/// [`step_and_cap`] evaluated over `exits` (either `st.exits` or exits after
/// selective predication), with `uniform_branches` naming latch blocks whose
/// branch condition has been promoted to a group-wide predicate. It tracks
/// both group-uniform SGPRs and group-uniform VGPRs, so a uniform loop bound
/// loaded via `buffer_load` / `global_load` / `image_load` / `ds_read` and
/// moved through `v_readfirstlane_b32` stays group-uniform.
pub(crate) fn step_and_cap_with(
    p: &Program,
    st: &Structured,
    exits: &[LaneExit],
    uniform_branches: &Set<usize>,
    need_step: bool,
) -> (Vec<bool>, Vec<bool>, Set<Value>) {
    let n = p.blocks.len();
    // Reverse postorder ranks, as lowering numbers them: a Jump to a block at
    // or before the jump is a back edge.
    let order = crate::ir::reverse_postorder(&p.blocks);
    let mut rpo = vec![usize::MAX; n];
    for (r, &b) in order.iter().enumerate() {
        rpo[b] = r;
    }
    let back = |b: usize, t: usize| rpo[b] != usize::MAX && rpo[t] <= rpo[b];
    // The unconditional back edges, (latch, header): lowering counts the
    // loop's iterations and at the limit leaves the program from the latch
    // with an early return.
    let capped: Vec<(usize, usize)> = (0..n)
        .filter_map(|b| match exits[b] {
            LaneExit::Jump(t) if back(b, t) => Some((b, t)),
            _ => None,
        })
        .collect();
    if !need_step
        && (capped.is_empty()
            || !p
                .blocks
                .iter()
                .any(|b| b.insts.iter().any(|i| i.name() == "s_barrier")))
    {
        return (Vec::new(), vec![false; n], Set::default());
    }
    let Some(regions) = halves::branch_regions_with(p, exits) else {
        return (
            if need_step {
                once_per_lane(p)
            } else {
                Vec::new()
            },
            vec![false; n],
            Set::default(),
        );
    };
    let mut succ: Vec<Vec<usize>> = vec![Vec::new(); n];
    for (b, e) in exits.iter().enumerate() {
        match *e {
            LaneExit::Jump(t) => succ[b].push(t),
            LaneExit::Branch {
                taken, not_taken, ..
            } => {
                succ[b].push(taken);
                succ[b].push(not_taken);
            }
            LaneExit::End | LaneExit::Indirect => {}
        }
    }
    // Every back edge to each header: the header's iteration counter is
    // shared by all of them.
    let mut back_to: Vec<Vec<usize>> = vec![Vec::new(); n];
    for (b, e) in exits.iter().enumerate() {
        match *e {
            LaneExit::Jump(t) if back(b, t) => back_to[t].push(b),
            LaneExit::Branch {
                taken, not_taken, ..
            } => {
                if back(b, taken) {
                    back_to[taken].push(b);
                }
                if back(b, not_taken) {
                    back_to[not_taken].push(b);
                }
            }
            _ => {}
        }
    }
    let scalar = |r: Reg| matches!(r, Reg::S(_) | Reg::Scc);
    let vector = |r: Reg| matches!(r, Reg::V(_));
    // Not EXEC or VCC: lane masks.
    let mask = |r: Reg| matches!(r, Reg::S(k) if [EXEC, EXEC + 1, VCC, VCC + 1].contains(&k));
    // Scalar writes no host wave can compute alike: other lanes' values,
    // hardware state, atomics, coherent loads (the group may write them).
    let own = |inst: &Inst| {
        let name = inst.name();
        match &inst.fields {
            Fields::Sop { .. } => {
                name.starts_with("s_getreg")
                    || name.starts_with("s_sendmsg")
                    || name.contains("waveid")
            }
            Fields::Smem { cache, .. } => {
                !(name.starts_with("s_load") || name.starts_with("s_buffer_load")) || cache.glc
            }
            _ => true,
        }
    };
    let together = |d: usize, when: Bool, alike: &Set<Value>| {
        uniform_branches.contains(&d)
            || st
                .bools
                .support(when)
                .iter()
                .all(|a| match *a {
                    Atom::Uniform(v) => alike.contains(&v),
                    Atom::Any(_) => st.literal,
                    _ => false,
                })
    };
    let mut alike: Set<Value> = p
        .inputs
        .iter()
        .filter(|(r, _)| scalar(*r))
        .map(|(_, v)| *v)
        .chain(p.blocks.iter().flat_map(|b| {
            b.phis
                .iter()
                .filter(|f| scalar(f.reg))
                .map(|f| f.value)
                .chain(
                    b.insts
                        .iter()
                        .flat_map(|i| i.writes.iter().filter(|w| scalar(w.0)).map(|w| w.1)),
                )
        }))
        .collect();
    let tracked = |v| {
        st.literal
            && (st.masks.contains_key(&v)
                || st.highs.contains_key(&v)
                || st.scc_votes.contains_key(&v))
    };
    // Group-uniform VGPRs: optimistically all non-entry VGPRs, narrowed with
    // `alike` in lockstep. Entry VGPRs (`v0`/`v1`/`v2` thread ids) are never
    // group-uniform.
    let mut alike_v: Set<Value> = p
        .blocks
        .iter()
        .flat_map(|b| {
            b.phis
                .iter()
                .filter(|f| vector(f.reg))
                .map(|f| f.value)
                .chain(
                    b.insts
                        .iter()
                        .flat_map(|i| i.writes.iter().filter(|w| vector(w.0)).map(|w| w.1)),
                )
        })
        .collect();
    loop {
        let mut step = vec![true; n];
        for (d, region) in &regions {
            let LaneExit::Branch { when, .. } = exits[*d] else {
                continue;
            };
            if !together(*d, when, &alike) {
                for (b, &r) in region.iter().enumerate() {
                    step[b] &= !r;
                }
            }
        }
        // A capped back edge's cap check holds for every lane only when the
        // latch and the header run together and so does every back edge to
        // the header (a conditional one with a uniform condition): one the
        // lanes do not run together can put one wave's count over the limit
        // at the latch while another goes on, and that wave leaves the
        // program there — so no block reachable from the loop is run by
        // every lane of the group, the same number of times.
        let cap_uniform = |latch: usize, header: usize, step: &[bool], alike: &Set<Value>| {
            step[latch]
                && step[header]
                && back_to[header].iter().all(|&b| {
                    if !step[b] {
                        return false;
                    }
                    match exits[b] {
                        LaneExit::Jump(t) => t == header,
                        LaneExit::Branch {
                            when,
                            taken,
                            not_taken,
                        } => {
                            (taken == header || not_taken == header)
                                && together(b, when, alike)
                        }
                        _ => false,
                    }
                })
        };
        let mut escaped = vec![false; n];
        // Reverse adjacency, to walk back from each latch to the branches
        // that can reach it.
        let mut preds: Vec<Vec<usize>> = vec![Vec::new(); n];
        for (b, ss) in succ.iter().enumerate() {
            for &s in ss {
                preds[s].push(b);
            }
        }
        // Mark every block forward-reachable from `from`.
        let mark = |from: usize, escaped: &mut Vec<bool>| {
            let mut seen = vec![false; n];
            let mut stack = vec![from];
            seen[from] = true;
            while let Some(b) = stack.pop() {
                escaped[b] = true;
                for &s in &succ[b] {
                    if !seen[s] {
                        seen[s] = true;
                        stack.push(s);
                    }
                }
            }
        };
        loop {
            let mut changed = false;
            for &(latch, header) in &capped {
                if cap_uniform(latch, header, &step, &alike) {
                    continue;
                }
                // Whether the latch is in step, before the walk from the
                // header below clears it.
                let latch_in_step = step[latch];
                let mut seen = vec![false; n];
                let mut stack = vec![header];
                seen[header] = true;
                while let Some(b) = stack.pop() {
                    escaped[b] = true;
                    if std::mem::replace(&mut step[b], false) {
                        changed = true;
                    }
                    for &s in &succ[b] {
                        if !seen[s] {
                            seen[s] = true;
                            stack.push(s);
                        }
                    }
                }
                // The sibling paths of every divergent branch the latch is
                // reachable from: a lane that took another arm of such a
                // branch never passes the latch's cap, so it can still reach
                // a barrier after the branch while a lane in the loop arm
                // has left the program at the cap. (The branch's own block
                // every lane runs, and a uniform branch sends every lane the
                // same way, so neither is marked here.)
                let arms = |d: usize, escaped: &mut Vec<bool>| {
                    if let LaneExit::Branch { when, .. } = exits[d]
                        && !together(d, when, &alike)
                    {
                        for &s in &succ[d] {
                            mark(s, escaped);
                        }
                    }
                };
                let mut seen = vec![false; n];
                let mut stack = vec![latch];
                if latch_in_step {
                    // The latch is in step, but a back edge to the header
                    // the lanes do not run together shares its counter all
                    // the same: walk back from those sources too.
                    stack.extend(back_to[header].iter().copied().filter(|&b| !step[b]));
                }
                while let Some(b) = stack.pop() {
                    if std::mem::replace(&mut seen[b], true) {
                        continue;
                    }
                    arms(b, &mut escaped);
                    for &p in &preds[b] {
                        arms(p, &mut escaped);
                        // Walk back only through blocks the lanes do not run
                        // together: at a block every lane runs together they
                        // all reconverge before the cap exit, so an earlier
                        // divergent branch whose arms already reconverged
                        // there cannot strand a lane past the cap.
                        if !step[p] {
                            stack.push(p);
                        }
                    }
                }
            }
            if !changed {
                break;
            }
        }
        let mut next: Set<Value> = p
            .inputs
            .iter()
            .filter(|(r, _)| scalar(*r))
            .map(|(_, v)| *v)
            .collect();
        let mut next_v: Set<Value> = Set::default();
        for (b, block) in p.blocks.iter().enumerate() {
            // Every host wave enters by the same edge.
            let entered = step[b] && block.preds.iter().all(|&q| step[q]);
            for phi in &block.phis {
                if scalar(phi.reg)
                    && (tracked(phi.value) || entered && phi.args.iter().all(|a| alike.contains(a)))
                {
                    next.insert(phi.value);
                } else if vector(phi.reg) && entered && phi.args.iter().all(|a| alike_v.contains(a))
                {
                    next_v.insert(phi.value);
                }
            }
            for (i, inst) in block.insts.iter().enumerate() {
                let name = inst.name();
                // Group-uniform scalar ops.
                let from_scalar = inst
                    .reads
                    .iter()
                    .all(|&(r, v)| scalar(r) && (!mask(r) || tracked(v)) && alike.contains(&v));
                let lane_read = st.literal
                    && step[b]
                    && match name {
                        "v_readfirstlane_b32" => true,
                        "v_readlane_b32" => {
                            halves::uniform_selector(inst, &alike)
                                && inst
                                    .writes
                                    .first()
                                    .and_then(|w| st.restored.get(&w.1))
                                    .is_none_or(|source| alike.contains(source))
                        }
                        _ => false,
                    };
                if from_scalar && !own(inst) || lane_read || st.literal {
                    next.extend(inst.writes.iter().filter_map(|&(r, v)| {
                        (scalar(r) && (tracked(v) || from_scalar && !own(inst) || lane_read))
                            .then_some(v)
                    }));
                    if from_scalar && !own(inst) || lane_read {
                        continue;
                    }
                }
                // `v_readfirstlane_b32` of a group-uniform VGPR yields a
                // group-uniform SGPR.
                if name == "v_readfirstlane_b32"
                    && let Fields::Valu(v) = &inst.fields
                    && let Some(&Operand::Vector { reg, .. }) = v.src.first()
                    && inst.read(Reg::V(reg)).is_some_and(|x| alike_v.contains(&x))
                {
                    next.extend(inst.writes.iter().filter(|w| scalar(w.0)).map(|w| w.1));
                    continue;
                }
                // Guarded VGPR writes only stay group-uniform when the guard
                // itself is group-uniform (for simplicity, require no guard).
                if st.guards[b][i].is_some() {
                    continue;
                }
                // Buffer / global / LDS loads at group-uniform addresses.
                match &inst.fields {
                    Fields::Buffer {
                        addr,
                        rsrc,
                        soffset,
                        offen,
                        idxen,
                        cache,
                        ..
                    } if (name.starts_with("buffer_load") || name.starts_with("tbuffer_load"))
                        && !cache.glc =>
                    {
                        let addr_ok = !(*offen || *idxen)
                            || match *addr {
                                Operand::Vector { reg, dwords } => (0..dwords).all(|k| {
                                    inst.read(Reg::V(reg + k))
                                        .is_some_and(|v| alike_v.contains(&v))
                                }),
                                _ => true,
                            };
                        let sc_ok = |op: &Operand| match *op {
                            Operand::Scalar { code, dwords } if code != NULL => (0..dwords)
                                .all(|k| {
                                    let r = Reg::S(code + k);
                                    !mask(r) && inst.read(r).is_none_or(|v| alike.contains(&v))
                                }),
                            _ => true,
                        };
                        if addr_ok && sc_ok(rsrc) && sc_ok(soffset) {
                            next_v.extend(inst.writes.iter().filter(|w| vector(w.0)).map(|w| w.1));
                        }
                    }
                    Fields::Flat {
                        addr: Some(Operand::Vector { reg, dwords }),
                        saddr,
                        cache,
                        ..
                    } if name.starts_with("global_load") && !cache.glc => {
                        let addr_ok = (0..*dwords).all(|k| {
                            inst.read(Reg::V(reg + k))
                                .is_some_and(|v| alike_v.contains(&v))
                        });
                        let saddr_ok = match saddr {
                            Some(Operand::Scalar { code, dwords }) if *code != NULL => (0..*dwords)
                                .all(|k| {
                                    let r = Reg::S(code + k);
                                    !mask(r) && inst.read(r).is_none_or(|v| alike.contains(&v))
                                }),
                            _ => true,
                        };
                        if addr_ok && saddr_ok {
                            next_v.extend(inst.writes.iter().filter(|w| vector(w.0)).map(|w| w.1));
                        }
                    }
                    Fields::Image {
                        addr,
                        rsrc,
                        sampler,
                        cache,
                        ..
                    } if (name.starts_with("image_load")
                        || name.starts_with("image_sample")
                        || name.starts_with("image_gather"))
                        && !cache.glc =>
                    {
                        let addr_ok = addr.iter().all(|op| match *op {
                            Operand::Vector { reg, dwords } => (0..dwords).all(|k| {
                                inst.read(Reg::V(reg + k))
                                    .is_some_and(|v| alike_v.contains(&v))
                            }),
                            _ => true,
                        });
                        let sc_ok = |op: &Operand| match *op {
                            Operand::Scalar { code, dwords } if code != NULL => (0..dwords)
                                .all(|k| {
                                    let r = Reg::S(code + k);
                                    !mask(r) && inst.read(r).is_none_or(|v| alike.contains(&v))
                                }),
                            _ => true,
                        };
                        if addr_ok && sc_ok(rsrc) && sampler.as_ref().is_none_or(sc_ok) {
                            next_v.extend(inst.writes.iter().filter(|w| vector(w.0)).map(|w| w.1));
                        }
                    }
                    Fields::Ds {
                        addr: Some(Operand::Vector { reg, .. }),
                        dst: Some(_),
                        data,
                        ..
                    } if step[b] && name.starts_with("ds_read") && data.is_empty() => {
                        if inst.read(Reg::V(*reg)).is_some_and(|v| alike_v.contains(&v)) {
                            next_v.extend(inst.writes.iter().filter(|w| vector(w.0)).map(|w| w.1));
                        }
                    }
                    Fields::Valu(v)
                        if v.dpp.is_none()
                            && v.sdwa.is_none()
                            && !name.starts_with("v_mbcnt")
                            && !name.starts_with("v_cmpx")
                            && !name.contains("lane")
                            && !name.contains("cndmask")
                            && !name.contains("_co_ci_")
                            && !name.contains("mac")
                            && !name.contains("dot") =>
                    {
                        let ops_ok = v.src.iter().all(|op| match *op {
                            Operand::Vector { reg, dwords } => (0..dwords).all(|k| {
                                inst.read(Reg::V(reg + k))
                                    .is_some_and(|val| alike_v.contains(&val))
                            }),
                            Operand::Scalar { code, dwords } if code != NULL => {
                                (0..dwords).all(|k| {
                                    let r = Reg::S(code + k);
                                    !mask(r) && inst.read(r).is_none_or(|val| alike.contains(&val))
                                })
                            }
                            Operand::Const(_) | Operand::Literal(_) | Operand::Scalar { .. } => {
                                true
                            }
                            Operand::Attr { .. } => false,
                        });
                        if ops_ok {
                            next_v.extend(inst.writes.iter().filter(|w| vector(w.0)).map(|w| w.1));
                        }
                    }
                    _ => {}
                }
            }
        }
        if next.len() == alike.len() && next_v.len() == alike_v.len() {
            return (step, escaped, next);
        }
        alike = next;
        alike_v = next_v;
    }
}

/// The step half of [`step_and_cap`].
pub(crate) fn in_step(p: &Program, st: &Structured) -> Vec<bool> {
    step_and_cap(p, st, true).0
}

/// The step half of [`step_and_cap_with`].
pub(crate) fn in_step_with(
    p: &Program,
    st: &Structured,
    exits: &[LaneExit],
    uniform_branches: &Set<usize>,
) -> Vec<bool> {
    step_and_cap_with(p, st, exits, uniform_branches, true).0
}

/// Per block, whether each lane of the group runs it exactly once: no cycle
/// passes through it, and every path from the entry to an end does. A group
/// barrier needs every lane of the group, so it can go only in these.
pub(crate) fn once_per_lane(p: &Program) -> Vec<bool> {
    let n = p.blocks.len();
    let succ = |b: usize| p.blocks[b].exit.successors();
    // Blocks reachable from `from`'s successors (or from `from` itself when
    // `start`), never entering `avoid`.
    let reach = |from: usize, start: bool, avoid: Option<usize>| {
        let mut seen = vec![false; n];
        let mut stack = if start { vec![from] } else { succ(from) };
        while let Some(b) = stack.pop() {
            if Some(b) == avoid || std::mem::replace(&mut seen[b], true) {
                continue;
            }
            stack.extend(succ(b));
        }
        seen
    };
    (0..n)
        .map(|b| {
            let ends_around = b != 0
                && reach(0, true, Some(b))
                    .iter()
                    .enumerate()
                    .any(|(q, &r)| r && succ(q).is_empty());
            !reach(b, false, None)[b] && !ends_around
        })
        .collect()
}
