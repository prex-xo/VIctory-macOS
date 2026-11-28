//! Rendezvous of the two 32-lane halves of a console wave64 in divergent
//! code (`lower::halves`), where no group barrier can go.
//!
//! In halves a console wave runs as host waves `2w` and `2w + 1`. Values
//! cross between them through group-shared memory; between group barriers
//! ([`Lower::exchange_halves`]) only in a block every lane of the group runs
//! together ([`in_step`]). Here the two host waves of one console wave wait
//! for each other by spinning on group-shared memory instead, which needs
//! no other wave of the group. Measured on an M5 Pro (D3DMetal 4.0b3): two
//! host waves of one group spinning on each other's group-shared words
//! (exchange to publish, a group-shared fence, `or` 0 to poll) never timed
//! out, from 64 to 65,536 groups, with the other half busy for ~100k spins.
//! A lane spinning on another lane of its own host wave never finishes on
//! Apple GPUs; nothing here does that: a host wave only ever waits for the
//! other one.
//!
//! # Where
//!
//! A site is an exchange (a lane mask's ballot, [`MeetSite::Mask`]) in a
//! block that is not [`in_step`]. Its region `R` is that of the outermost
//! divergent branch around it whose branch block `B` is in step
//! ([`meet_regions`]): the blocks only the lanes taking one side of `B`'s
//! branch may run. `R` must be acyclic, entered only from `B`, and left
//! only to one block `J` (`B`'s immediate post-dominator) or the program's
//! end. Elsewhere the exchange stays refused.
//!
//! # Epochs
//!
//! Both halves run `B` equally often (it is in step), so each counts its
//! runs of `B` alike: the epoch `e` (1 when `B` is in no cycle). Every word a
//! half publishes is tagged with it, so a word from an earlier run of `R`
//! never passes for this one. Where `B` is in a cycle, `J` must be in step
//! too and gets a group barrier: no half starts its next run of `R` (and
//! overwrites its words) before the other is done reading this one's. The
//! words are zeroed in the entry block, before one group barrier: group-
//! shared memory starts undefined.
//!
//! # Words (per host wave, per region)
//!
//! * `epoch`: the half's count of runs of `B` (only where `B` is in a cycle);
//! * `count`: `e << 8 | n`, `n` the half's lanes at `B`'s end, published
//!   there (after the `resolved` words are reset to `e << 8`);
//! * per site: `arrived` (`e` once the half ran the site, after `value`),
//!   `value` (the half's own: its ballot), and `resolved` (`e << 8` plus
//!   one per lane of the half that is done with the site this epoch).
//!
//! A lane is done with a site once it ran it (counted after its own wait),
//! or once it took an edge after which the site's block cannot be reached
//! (counted on that edge, at the end of the block it leaves: [`Lower::
//! meet_leave`]), or when it ends. Along a lane's path through the acyclic
//! `R` what it can still reach only shrinks, so it counts each site once.
//!
//! # A site
//!
//! The half stores its value, fences, publishes `arrived = e`, then spins
//! (at most [`MEET_SPINS`] times) until the other half's `arrived` is `e`
//! (its value is read) or its `resolved` equals its `count` with the tag
//! `e` (all its lanes are done with the site without running it: ballot 0).
//! `resolved` is read before `arrived`, with a fence between: a half
//! publishes `arrived` before its lanes count the site resolved, so a full
//! count read first means `arrived` is seen if it was set. A console wave
//! with no upper half (a group not a multiple of 64) waits for nothing.
//!
//! # Why it cannot deadlock
//!
//! Assume, as the lowering of a ballot in divergent code does anyway, that a
//! host wave runs each block of `R` at most once per epoch with all its lanes
//! that reach it, and that it runs the blocks of `R` in one topological order
//! of `R`, the same for both halves (the order of the compiled code, which
//! both run). Let half `A` wait at site `k` for half `H`. `H` is either
//! (a) before `B` of this epoch: it is not blocked by `A` there (a group
//! barrier `A` passed has seen `H` too, and an earlier epoch's sites `A` has
//! all resolved or run), so it comes; (b) in `R`, waiting at a site `q`: if
//! `q` is `k`, both have arrived. If `q` comes before `k` in the common
//! order, `A` has run past `q`: every lane of `A` either ran `q` or is at a
//! block after `q` in topological order, from which `q` cannot be reached,
//! so it counted `q` on the way; so `A`'s `q` is arrived or fully resolved,
//! and `H` goes on. If `q` comes after `k`, the same holds with the roles
//! swapped: every lane of `H` has run `k` or counted it, so `A` goes on;
//! (c) past `R`: all of its lanes have resolved or run every site. A
//! console wave's two halves thus never wait for each other in a cycle.
//! The spin bound is a fuse, not part of the argument: a half that runs
//! out reads the other's lanes as absent.
//!
//! What is assumed is what the single-wave lowering of a ballot in
//! divergent code already relies on (the host runs a block once with the
//! lanes reaching it); a host that ran the halves' divergent paths in
//! different orders could make both wait out [`MEET_SPINS`].

use super::*;

/// How many times a half polls the other's words at a site before it
/// reads the other's lanes as absent.
pub(super) const MEET_SPINS: u32 = 1 << 20;

/// An exchange between the halves of a console wave at a divergent site.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub(crate) enum MeetSite {
    /// The ballot of the lane mask with this low dword, taken right after
    /// the instruction writing it.
    Mask(Value),
}

/// A divergent region whose branch block is in step ([`meet_regions`]).
pub(super) struct MeetRegion {
    /// `B`: the branch block, in step.
    pub branch: usize,
    /// `J`: where every lane leaving the region goes (none: all end).
    pub join: Option<usize>,
    /// The region's blocks.
    pub blocks: Vec<bool>,
    /// `B` is in a cycle: it runs more than once.
    pub looped: bool,
    /// Per block of the region and `B`: the region's blocks reachable from
    /// it by one edge or more.
    pub reach: Map<usize, Vec<bool>>,
}

/// The regions where the halves can meet in a program, and which region
/// each block is in.
#[derive(Default)]
pub(super) struct Meet {
    pub regions: Vec<MeetRegion>,
    pub region_of: Vec<Option<usize>>,
}

/// Each divergent branch whose block is in step (`step`, [`in_step`]) and
/// whose region (`halves::branch_regions`) the halves can meet in: acyclic,
/// entered only from the branch, left only to one block or the end, and,
/// where the branch block is in a cycle, left to a block in step (where a
/// group barrier goes). A block in two such regions is in none.
pub(super) fn meet_regions(p: &Program, st: &Structured, step: &[bool]) -> Meet {
    let n = st.exits.len();
    let mut meet = Meet {
        regions: Vec::new(),
        region_of: vec![None; n],
    };
    let Some(regions) = halves::branch_regions(p, st) else {
        return meet;
    };
    let succ = |b: usize| -> Vec<usize> {
        match st.exits[b] {
            LaneExit::Jump(t) => vec![t],
            LaneExit::Branch {
                taken, not_taken, ..
            } => vec![taken, not_taken],
            _ => Vec::new(),
        }
    };
    let mut preds: Vec<Vec<usize>> = vec![Vec::new(); n];
    for b in 0..n {
        for t in succ(b) {
            preds[t].push(b);
        }
    }
    let mut conflict = vec![false; n];
    for (d, region) in regions {
        if !step.get(d).copied().unwrap_or(false) || region[d] {
            continue;
        }
        let blocks: Vec<usize> = (0..n).filter(|&b| region[b]).collect();
        // A uniform branch: its sides are in step.
        if blocks.is_empty() || blocks.iter().all(|&b| step[b]) {
            continue;
        }
        let entered_from_branch = blocks
            .iter()
            .all(|&b| preds[b].iter().all(|&q| q == d || region[q]));
        let mut outside: Vec<usize> = blocks
            .iter()
            .chain([&d])
            .flat_map(|&b| succ(b))
            .filter(|&t| !region[t])
            .collect();
        outside.sort_unstable();
        outside.dedup();
        let ends = blocks.iter().any(|&b| st.exits[b] == LaneExit::End);
        if !entered_from_branch
            || outside.len() > 1
            || blocks
                .iter()
                .any(|&b| matches!(st.exits[b], LaneExit::Indirect))
        {
            continue;
        }
        let join = outside.first().copied();
        // Acyclic: every block of the region in a topological order.
        let mut ins: Vec<usize> = (0..n)
            .map(|b| preds[b].iter().filter(|&&q| region[q]).count())
            .collect();
        let mut ready: Vec<usize> = blocks.iter().copied().filter(|&b| ins[b] == 0).collect();
        let mut sorted = 0;
        while let Some(b) = ready.pop() {
            sorted += 1;
            for t in succ(b) {
                if region[t] {
                    ins[t] -= 1;
                    if ins[t] == 0 {
                        ready.push(t);
                    }
                }
            }
        }
        if sorted != blocks.len() {
            continue;
        }
        // The branch block in a cycle: reached again from its successors.
        let mut seen = vec![false; n];
        let mut stack = succ(d);
        while let Some(b) = stack.pop() {
            if !std::mem::replace(&mut seen[b], true) {
                stack.extend(succ(b));
            }
        }
        let looped = seen[d];
        if looped && (ends || !join.is_some_and(|j| step[j])) {
            continue;
        }
        let mut reach = Map::default();
        for &u in blocks.iter().chain([&d]) {
            let mut r = vec![false; n];
            let mut stack: Vec<usize> = succ(u).into_iter().filter(|&t| region[t]).collect();
            while let Some(b) = stack.pop() {
                if !std::mem::replace(&mut r[b], true) {
                    stack.extend(succ(b).into_iter().filter(|&t| region[t]));
                }
            }
            reach.insert(u, r);
        }
        let k = meet.regions.len();
        for &b in &blocks {
            if meet.region_of[b].is_some() {
                conflict[b] = true;
            }
            meet.region_of[b] = Some(k);
        }
        meet.regions.push(MeetRegion {
            branch: d,
            join,
            blocks: region,
            looped,
            reach,
        });
    }
    // A block in two regions: neither region's sites there can meet.
    for (b, c) in conflict.iter().enumerate() {
        if *c {
            meet.region_of[b] = None;
        }
    }
    meet
}

/// Where a site's exchange sits in the group-shared words.
#[derive(Clone, Copy)]
pub(super) struct MeetSlot {
    pub region: usize,
    /// The site's three words: `arrived`, `value`, `resolved`.
    pub word: u32,
    pub block: usize,
}

/// The layout of the rendezvous words: per region, its first word (the
/// epoch, then the count) and its sites; the words per host wave.
#[derive(Default)]
pub(super) struct MeetLayout {
    pub slots: Map<MeetSite, MeetSlot>,
    pub region_base: Vec<u32>,
    pub region_sites: Vec<Vec<MeetSite>>,
    /// Mask sites per IR block: (instruction before which, site), in order.
    pub masks: Map<usize, Vec<(usize, MeetSite)>>,
    /// Words per host wave, a multiple of 32.
    pub stride: u32,
}

impl Lower<'_> {
    /// Whether an exchange at `site`, in IR block `block`, can wait for the
    /// other half there; if so (and it is not laid out yet) it is noted for
    /// the next pass (`lower_pruned`), and this pass goes on with a stand-in.
    pub(super) fn meet_later(&mut self, site: MeetSite, block: usize) -> bool {
        let Some(meet) = &self.meet else {
            return false;
        };
        if meet.region_of.get(block).copied().flatten().is_none() || self.meet_plan.contains(&site)
        {
            return false;
        }
        self.meet_found.borrow_mut().insert(site);
        true
    }

    /// Lays out the words of the sites `plan` names (found by an earlier
    /// pass), if group-shared memory has room for them.
    pub(super) fn lay_out_meet(&mut self) {
        let Some(meet) = &self.meet else {
            return;
        };
        let mut sites: Vec<(usize, usize, MeetSite)> = Vec::new();
        for &site in self.meet_plan {
            let MeetSite::Mask(lo) = site;
            let Some(&b) = self.def_block.get(&lo) else {
                continue;
            };
            let Some(_) = meet.region_of[b] else {
                continue;
            };
            // Right after the instruction writing it (a phi: at the start).
            let at = self.def_index.get(&lo).map_or(0, |&i| i + 1);
            sites.push((b, at, site));
        }
        sites.sort_unstable();
        let mut layout = MeetLayout {
            region_base: vec![0; meet.regions.len()],
            region_sites: vec![Vec::new(); meet.regions.len()],
            ..MeetLayout::default()
        };
        for &(b, _, site) in &sites {
            let r = meet.region_of[b].expect("in a region");
            layout.region_sites[r].push(site);
        }
        let mut word = 0;
        for r in 0..meet.regions.len() {
            if layout.region_sites[r].is_empty() {
                continue;
            }
            layout.region_base[r] = word;
            word += 2;
            for &site in &layout.region_sites[r] {
                let &(b, _, _) = sites.iter().find(|s| s.2 == site).expect("listed");
                layout.slots.insert(
                    site,
                    MeetSlot {
                        region: r,
                        word,
                        block: b,
                    },
                );
                word += 3;
            }
        }
        if word == 0 {
            return;
        }
        for &(b, at, site) in &sites {
            layout.masks.entry(b).or_default().push((at, site));
        }
        layout.stride = word.next_multiple_of(32);
        let group: u32 = self.s.threads.map_or(0, |t| t.iter().product());
        let waves = group.div_ceil(64) * 2;
        // Room for these and for the barrier exchanges' words too.
        if self.s.lds_dwords + waves * layout.stride + waves > GROUP_SHARED_DWORDS {
            return;
        }
        self.meet_layout = layout;
    }

    /// The rendezvous words, zeroed by every lane in the entry block before
    /// a group barrier; none without sites.
    pub(super) fn meet_prologue(&mut self) {
        if self.meet_layout.stride == 0 {
            return;
        }
        let group: u32 = self.s.threads.map_or(0, |t| t.iter().product());
        let waves = group.div_ceil(64) * 2;
        let stride = self.meet_layout.stride;
        let i32t = self.s.t.i32;
        let arr = self.s.m.array(waves * stride, i32t);
        let g = self.s.m.global("half_meet", arr, 3);
        self.meet_words = Some(g);
        let c = self.cur;
        let f = self.relabelled_flat();
        let (k5, k31) = (self.s.m.i32(5), self.s.m.i32(31));
        let slot = self.s.m.bin(c, Bin::LShr, f, k5);
        let k = self.s.m.i32(stride);
        let base = self.s.m.bin(c, Bin::Mul, slot, k);
        let lane = self.s.m.bin(c, Bin::And, f, k31);
        let first = self.s.m.bin(c, Bin::Add, base, lane);
        let zero = self.s.m.i32(0);
        for j in (0..stride).step_by(32) {
            let off = self.s.m.i32(j);
            let at = self.s.m.bin(c, Bin::Add, first, off);
            let ptr = self.s.m.element(c, g, at);
            self.s.m.store(c, ptr, zero);
        }
        self.group_barrier();
    }

    /// Element `word` of this host wave's (`other`: the other half's)
    /// rendezvous words, in block `at`.
    fn meet_word(&mut self, at: usize, word: u32, other: bool) -> V {
        let g = self.meet_words.expect("laid out");
        let f = self.relabelled_flat();
        let k5 = self.s.m.i32(5);
        let mut slot = self.s.m.bin(at, Bin::LShr, f, k5);
        if other {
            let one = self.s.m.i32(1);
            slot = self.s.m.bin(at, Bin::Xor, slot, one);
        }
        let k = self.s.m.i32(self.meet_layout.stride);
        let base = self.s.m.bin(at, Bin::Mul, slot, k);
        let w = self.s.m.i32(word);
        let i = self.s.m.bin(at, Bin::Add, base, w);
        self.s.m.element(at, g, i)
    }

    /// A group-shared memory fence, with no sync: allowed where the lanes
    /// of the group are not all together.
    fn meet_fence(&mut self, at: usize) {
        let mode = self.s.m.i32(8);
        let void = self.s.t.void;
        self.s
            .op(at, "barrier", Attr::NoDuplicate, void, 80, &[mode]);
    }

    /// The region IR block `b` is the branch block of, with sites.
    fn meet_branch_of(&self, b: usize) -> Option<usize> {
        let meet = self.meet.as_ref()?;
        (0..meet.regions.len())
            .find(|&r| meet.regions[r].branch == b && !self.meet_layout.region_sites[r].is_empty())
    }

    /// At the start of IR block `b`: a group barrier if it is the join of a
    /// region with sites whose branch block runs more than once (no half
    /// starts the region again before the other is done with it).
    pub(super) fn meet_enter(&mut self, b: usize) {
        if self.meet_layout.stride == 0 {
            return;
        }
        let meet = self.meet.as_ref().expect("laid out");
        let barrier = (0..meet.regions.len()).any(|r| {
            let g = &meet.regions[r];
            g.looped && g.join == Some(b) && !self.meet_layout.region_sites[r].is_empty()
        });
        if barrier {
            self.group_barrier();
        }
    }

    /// The mask sites of IR block `b` before instruction `i` (`None`: all
    /// left, at its end): each mask's ballot, met between the halves.
    pub(super) fn meet_masks(&mut self, b: usize, i: Option<usize>) {
        let Some(list) = self.meet_layout.masks.get_mut(&b) else {
            return;
        };
        let n = list
            .iter()
            .take_while(|(at, _)| i.is_none_or(|i| *at <= i))
            .count();
        let due: Vec<MeetSite> = list.drain(..n).map(|(_, s)| s).collect();
        for site in due {
            let MeetSite::Mask(lo) = site;
            if self.ballots.contains_key(&lo) {
                continue;
            }
            let Some(&m) = self.x.st.masks.get(&lo) else {
                continue;
            };
            let Ok(bit) = self.boolean(m, self.cur) else {
                continue;
            };
            let c = self.cur;
            let i32t = self.s.t.i32;
            let four = self.s.m.named_struct("dx.types.fouri32", vec![i32t; 4]);
            let r = self
                .s
                .op(c, "waveActiveBallot", Attr::NoUnwind, four, 116, &[bit]);
            self.s.flags |= flags::WAVE_OPS;
            let own = self.s.m.extract(c, r, 0);
            let zero = self.s.m.i32(0);
            if let Ok(xy) = self.meet_halves(site, own, zero) {
                self.ballots.insert(lo, xy);
            }
        }
    }

    /// At the end of IR block `b`, before its exit: as a region's branch
    /// block, the half's epoch, its `resolved` words reset and its lane
    /// count published; as the branch block or a block of a region with
    /// sites, the sites its lanes are done with on leaving
    /// ([`Lower::meet_leave`]).
    pub(super) fn meet_exit(&mut self, b: usize) -> Result<(), Why> {
        if self.meet_layout.stride == 0 {
            return Ok(());
        }
        if let Some(r) = self.meet_branch_of(b) {
            self.meet_start(r);
            self.meet_leave(r, b)?;
        }
        let r = self.meet.as_ref().expect("laid out").region_of[b];
        if let Some(r) = r
            && !self.meet_layout.region_sites[r].is_empty()
        {
            self.meet_leave(r, b)?;
        }
        Ok(())
    }

    /// The epoch of region `r` at its branch block's end, the half's
    /// `resolved` words reset to it, then its count of lanes published.
    fn meet_start(&mut self, r: usize) {
        let c = self.cur;
        let base = self.meet_layout.region_base[r];
        let looped = self.meet.as_ref().expect("laid out").regions[r].looped;
        let e = if looped {
            let p = self.meet_word(c, base, false);
            let old = self.s.m.load(c, p);
            let one = self.s.m.i32(1);
            let e = self.s.m.bin(c, Bin::Add, old, one);
            self.s.m.store(c, p, e);
            e
        } else {
            self.s.m.i32(1)
        };
        let k8 = self.s.m.i32(8);
        let tag = self.s.m.bin(c, Bin::Shl, e, k8);
        for site in self.meet_layout.region_sites[r].clone() {
            let w = self.meet_layout.slots[&site].word;
            let p = self.meet_word(c, w + 2, false);
            self.s.m.store(c, p, tag);
        }
        self.meet_fence(c);
        let i32t = self.s.t.i32;
        let four = self.s.m.named_struct("dx.types.fouri32", vec![i32t; 4]);
        let yes = self.s.m.bool(true);
        let ballot = self
            .s
            .op(c, "waveActiveBallot", Attr::NoUnwind, four, 116, &[yes]);
        self.s.flags |= flags::WAVE_OPS;
        let lanes = self.s.m.extract(c, ballot, 0);
        let n = self.unary_bits(c, 31, lanes);
        let count = self.s.m.bin(c, Bin::Or, tag, n);
        let p = self.meet_word(c, base + 1, false);
        self.s.m.atomic(c, Rmw::Xchg, p, count);
        self.meet_epoch.insert(r, e);
    }

    /// The sites of region `r` the lanes leaving IR block `b` (in the
    /// region, or its branch block) are done with: those still ahead of
    /// `b` that the block a lane goes to cannot reach. Each such lane adds
    /// one to the half's `resolved` word of each.
    fn meet_leave(&mut self, r: usize, b: usize) -> Result<(), Why> {
        let meet = self.meet.as_ref().expect("laid out");
        let g = &meet.regions[r];
        let ahead = |u: usize, s: usize| g.reach.get(&u).is_some_and(|x| x[s]);
        // Sites still ahead after going to `v`: in `v` or reachable from it.
        let kept = |v: usize, s: usize| g.blocks[v] && (s == v || ahead(v, s));
        let exit = self.x.st.exits[b];
        let (to, other) = match exit {
            LaneExit::Jump(v) => (Some(v), None),
            LaneExit::Branch {
                taken, not_taken, ..
            } => (Some(taken), Some(not_taken)),
            _ => (None, None),
        };
        // Per site ahead of `b`: its word, and whether it stays ahead on
        // the (taken) and the not-taken way.
        let sites: Vec<(u32, bool, bool)> = self.meet_layout.region_sites[r]
            .iter()
            .map(|s| self.meet_layout.slots[s])
            .filter(|slot| ahead(b, slot.block))
            .map(|slot| {
                let on = |v: Option<usize>| v.is_some_and(|v| kept(v, slot.block));
                (slot.word, on(to), on(other))
            })
            .collect();
        if sites.is_empty() {
            return Ok(());
        }
        let c = self.cur;
        let (zero, one) = (self.s.m.i32(0), self.s.m.i32(1));
        let adds: Vec<(u32, V)> = match exit {
            LaneExit::Branch { when, .. } => {
                let cond = self.boolean(when, c)?;
                let on_taken = self.s.m.select(c, cond, one, zero);
                let on_not = self.s.m.select(c, cond, zero, one);
                sites
                    .iter()
                    .filter_map(|&(w, t, f)| match (t, f) {
                        (false, false) => Some((w, one)),
                        (false, true) => Some((w, on_taken)),
                        (true, false) => Some((w, on_not)),
                        (true, true) => None,
                    })
                    .collect()
            }
            _ => sites
                .iter()
                .filter(|&&(_, t, _)| !t)
                .map(|&(w, _, _)| (w, one))
                .collect(),
        };
        for (w, by) in adds {
            let p = self.meet_word(c, w + 2, false);
            self.s.m.atomic(c, Rmw::Add, p, by);
        }
        Ok(())
    }

    /// What each half of this lane's console wave gives as `own` at the
    /// divergent site `site` (a dword the same in every lane of the host
    /// wave): (lower half's, upper half's), the other's `missing` where it
    /// has no lane there (see the module notes). Lowered at [`Lower::cur`],
    /// which moves past the wait.
    pub(super) fn meet_halves(
        &mut self,
        site: MeetSite,
        own: V,
        missing: V,
    ) -> Result<(V, V), Why> {
        const UNLAID: Why =
            "a rendezvous between 32-lane halves with no words (wave64 on a 32-lane host)";
        let slot = *self.meet_layout.slots.get(&site).ok_or(UNLAID)?;
        let e = *self.meet_epoch.get(&slot.region).ok_or(UNLAID)?;
        let count = self.meet_layout.region_base[slot.region] + 1;
        let w = slot.word;
        let i32t = self.s.t.i32;
        let c = self.cur;
        let value = self.meet_word(c, w + 1, false);
        self.s.m.store(c, value, own);
        self.meet_fence(c);
        let arrived = self.meet_word(c, w, false);
        self.s.m.atomic(c, Rmw::Xchg, arrived, e);
        // A console wave with no upper half waits for nothing.
        let group: u32 = self.s.threads.map_or(0, |t| t.iter().product());
        let gone = if group.is_multiple_of(2 * HALF_LANES) {
            None
        } else {
            let f = self.relabelled_flat();
            let (k5, one) = (self.s.m.i32(5), self.s.m.i32(1));
            let slot = self.s.m.bin(c, Bin::LShr, f, k5);
            let other = self.s.m.bin(c, Bin::Xor, slot, one);
            let first = self.s.m.bin(c, Bin::Shl, other, k5);
            let n = self.s.m.i32(group);
            Some(self.s.m.cmp(c, pred::IUGE, first, n))
        };
        let head = self.s.m.new_block();
        let after = self.s.m.new_block();
        self.s.m.terminate(c, Term::Br(head));
        let spins = self.s.m.phi(head, i32t);
        let zero = self.s.m.i32(0);
        let p = self.meet_word(head, count, true);
        let counted = self.s.m.atomic(head, Rmw::Or, p, zero);
        let p = self.meet_word(head, w + 2, true);
        let resolved = self.s.m.atomic(head, Rmw::Or, p, zero);
        self.meet_fence(head);
        let p = self.meet_word(head, w, true);
        let there = self.s.m.atomic(head, Rmw::Or, p, zero);
        let got = self.s.m.cmp(head, pred::EQ, there, e);
        let k8 = self.s.m.i32(8);
        let tag = self.s.m.bin(head, Bin::LShr, counted, k8);
        let tagged = self.s.m.cmp(head, pred::EQ, tag, e);
        let full = self.s.m.cmp(head, pred::EQ, resolved, counted);
        let none = self.s.m.bin(head, Bin::And, tagged, full);
        let one = self.s.m.i32(1);
        let next = self.s.m.bin(head, Bin::Add, spins, one);
        let bound = self.s.m.i32(MEET_SPINS);
        let spent = self.s.m.cmp(head, pred::IUGE, next, bound);
        let mut stop = self.s.m.bin(head, Bin::Or, got, none);
        stop = self.s.m.bin(head, Bin::Or, stop, spent);
        if let Some(gone) = gone {
            stop = self.s.m.bin(head, Bin::Or, stop, gone);
        }
        self.s.m.terminate(head, Term::CondBr(stop, after, head));
        self.s.m.set_phi(spins, vec![(zero, c), (next, head)]);
        self.meet_fence(after);
        let p = self.meet_word(after, w + 1, true);
        let theirs = self.s.m.load(after, p);
        let theirs = self.s.m.select(after, got, theirs, missing);
        // This half's lanes here are done with the site.
        let p = self.meet_word(after, w + 2, false);
        self.s.m.atomic(after, Rmw::Add, p, one);
        self.cur = after;
        let f = self.relabelled_flat();
        let k5 = self.s.m.i32(5);
        let slot = self.s.m.bin(after, Bin::LShr, f, k5);
        let h = self.s.m.bin(after, Bin::And, slot, one);
        let upper = self.s.m.cmp(after, pred::NE, h, zero);
        let lo = self.s.m.select(after, upper, theirs, own);
        let hi = self.s.m.select(after, upper, own, theirs);
        Ok((lo, hi))
    }
}

/// Console lanes per host wave in halves.
const HALF_LANES: u32 = 32;

/// The group-shared dwords a D3D12 compute program may declare, at most
/// (32 KiB).
const GROUP_SHARED_DWORDS: u32 = 8192;
