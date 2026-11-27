//! `v_readfirstlane_b32` reads a console wave64's 32-lane host halves can
//! make each by itself, with no value passed between them.
//!
//! The console reads the first lane EXEC has across the wave's 64 lanes; a
//! host half, the first it has of its own 32. The two agree, and nothing
//! needs to cross the halves, where:
//!
//! 1. **The value is the same in every lane EXEC has at the read.** Every
//!    instruction that made it ran under the read's own EXEC value (the same
//!    SSA value: no EXEC write between them) and computed it from constants
//!    and scalar registers alone, by plain VALU operations, so every active
//!    lane holds the same function of the same scalars. The scalars are
//!    root parameters and what scalar loads and scalar ALU operations make
//!    of them: what each host half computes for itself in every halves
//!    translation, none of them made from a lane mask or a lane's value.
//! 2. **Only lanes EXEC has at the read see the result.** A half with no
//!    active lane reads a lane of its own that does not hold the value; that
//!    is harmless only if no lane outside the read's EXEC ever sees what it
//!    read. So every instruction that reads the result, or a scalar made from
//!    it, is a scalar ALU operation (whose results are held to the same) or
//!    a vector instruction that writes no scalar register and runs under an
//!    EXEC no wider than the read's: one made from it by compares into EXEC,
//!    ANDs and copies of such masks (`s_and_saveexec`, `s_and_b64`,
//!    `s_andn2_b64` of one, `s_mov_b64` of one, phis of them). No branch,
//!    scalar memory access, EXEC, VCC or M0 write takes it.
//!
//! With both, a half with an active lane reads the console's value and a
//! half without one reads something no lane of the result's readers can
//! see. Where EXEC is empty in the whole console wave, no reader runs at
//! all. The read must be in no cycle, so its EXEC value is one mask.
//!
//! Before #522 such reads were made each by its own host half without these
//! checks; #522 refused every read of a value that may differ between the
//! lanes of the console wave in code the halves cannot meet in, which left
//! out GTA VI's `8bb272e763f76148` (a division by a constant-buffer value
//! whose reciprocal the console compiler reads from the first lane).

use std::collections::{HashMap, HashSet};

use super::Parts;
use crate::gfx10::{Fields, Operand};
use crate::ir::{EXEC, Inst, Reg, VCC, Value};

/// The results (the SGPR value written) of the `v_readfirstlane_b32` reads
/// of a wave64 program each host half can make by itself (module notes).
pub(super) fn read_alone(x: &Parts) -> HashSet<Value> {
    let p = x.p;
    if !x.native.wave64 {
        return HashSet::new();
    }
    let mut def: HashMap<Value, (usize, usize)> = HashMap::new();
    let mut phi_def: HashMap<Value, (usize, usize)> = HashMap::new();
    let mut uses: HashMap<Value, Vec<(usize, usize)>> = HashMap::new();
    let mut phi_uses: HashMap<Value, Vec<(usize, usize)>> = HashMap::new();
    for (b, block) in p.blocks.iter().enumerate() {
        for (k, phi) in block.phis.iter().enumerate() {
            phi_def.insert(phi.value, (b, k));
            for &a in &phi.args {
                phi_uses.entry(a).or_default().push((b, k));
            }
        }
        for (i, inst) in block.insts.iter().enumerate() {
            for &(_, v) in &inst.writes {
                def.insert(v, (b, i));
            }
            for &(_, v) in &inst.reads {
                uses.entry(v).or_default().push((b, i));
            }
        }
    }
    let a = Analysis {
        x,
        def,
        phi_def,
        uses,
        phi_uses,
        params: x.res.params.iter().map(|&(v, _, _)| v).collect(),
    };
    let mut out = HashSet::new();
    for (b, block) in p.blocks.iter().enumerate() {
        for inst in &block.insts {
            if inst.name() != "v_readfirstlane_b32" {
                continue;
            }
            let Some(&(_, result)) = inst.writes.first() else {
                continue;
            };
            if a.alone(b, inst) {
                out.insert(result);
            }
        }
    }
    out
}

/// Values decided (`Some`), and being decided (`None`).
type Memo = HashMap<Value, Option<bool>>;

struct Analysis<'a, 'p> {
    x: &'a Parts<'p>,
    /// Where each value is written: (block, instruction).
    def: HashMap<Value, (usize, usize)>,
    /// Where each phi value is: (block, phi).
    phi_def: HashMap<Value, (usize, usize)>,
    /// The instructions reading each value.
    uses: HashMap<Value, Vec<(usize, usize)>>,
    /// The phis taking each value.
    phi_uses: HashMap<Value, Vec<(usize, usize)>>,
    params: HashSet<Value>,
}

/// Lane masks and other registers a value of the first condition may not be
/// made from or go to: VCC, M0, EXEC (and what lies past them).
fn special(code: u8) -> bool {
    code >= VCC
}

impl Analysis<'_, '_> {
    fn inst(&self, at: (usize, usize)) -> &Inst {
        &self.x.p.blocks[at.0].insts[at.1]
    }

    fn alone(&self, b: usize, read: &Inst) -> bool {
        let Fields::Valu(v) = &read.fields else {
            return false;
        };
        let Some(&Operand::Vector { reg, .. }) = v.src.first() else {
            return false;
        };
        let (Some(src), Some(exec)) = (read.read(Reg::V(reg)), self.exec(read)) else {
            return false;
        };
        let Some(&(_, result)) = read.writes.first() else {
            return false;
        };
        !self.on_cycle(b)
            && self.active_alike(src, exec, &mut HashMap::new())
            && self.seen_only_inside(result, exec)
    }

    /// The EXEC value (both dwords) a vector instruction runs under.
    fn exec(&self, inst: &Inst) -> Option<(Value, Value)> {
        Some((inst.read(Reg::S(EXEC))?, inst.read(Reg::S(EXEC + 1))?))
    }

    fn on_cycle(&self, b: usize) -> bool {
        let blocks = &self.x.p.blocks;
        let mut seen = vec![false; blocks.len()];
        let mut stack = blocks[b].exit.successors();
        while let Some(q) = stack.pop() {
            if q == b {
                return true;
            }
            if !std::mem::replace(&mut seen[q], true) {
                stack.extend(blocks[q].exit.successors());
            }
        }
        false
    }

    /// Condition 1: `v` is the same in every lane active under `exec`.
    /// `memo`: values decided, and those being decided (`None`: a value
    /// that needs itself is not proven).
    fn active_alike(&self, v: Value, exec: (Value, Value), memo: &mut Memo) -> bool {
        match memo.get(&v) {
            Some(&r) => return r.unwrap_or(false),
            None => memo.insert(v, None),
        };
        let r = self.active_alike_of(v, exec, memo);
        memo.insert(v, Some(r));
        r
    }

    fn active_alike_of(&self, v: Value, exec: (Value, Value), memo: &mut Memo) -> bool {
        let Some(&at) = self.def.get(&v) else {
            return false;
        };
        let inst = self.inst(at);
        let Fields::Valu(f) = &inst.fields else {
            return false;
        };
        let name = inst.name();
        let plain = f.dpp.is_none()
            && f.sdwa.is_none()
            && f.carry.is_none()
            && f.opsel == 0
            && ![
                "lane", "cndmask", "mbcnt", "_co_", "interp", "mac", "dot", "cmp", "16", "mix",
                "perm", "sad", "msad", "qsad",
            ]
            .iter()
            .any(|n| name.contains(n));
        if !plain || self.exec(inst) != Some(exec) || !matches!(f.dst, Some(Operand::Vector { .. }))
        {
            return false;
        }
        f.src.iter().all(|op| match *op {
            Operand::Const(c) => c < 251,
            Operand::Literal(_) => true,
            Operand::Scalar { code, dwords } => {
                !special(code)
                    && (0..dwords).all(|k| {
                        inst.read(Reg::S(code + k))
                            .is_some_and(|s| self.scalar_ok(s, &mut HashMap::new()))
                    })
            }
            Operand::Vector { reg, dwords } => (0..dwords).all(|k| {
                inst.read(Reg::V(reg + k))
                    .is_some_and(|s| self.active_alike(s, exec, memo))
            }),
            Operand::Attr { .. } => false,
        })
    }

    /// A scalar value each host half computes alike: a root parameter, or
    /// what scalar loads and scalar ALU operations make of them.
    fn scalar_ok(&self, v: Value, memo: &mut Memo) -> bool {
        if self.params.contains(&v) {
            return true;
        }
        match memo.get(&v) {
            Some(&r) => return r.unwrap_or(false),
            None => memo.insert(v, None),
        };
        let r = self.scalar_ok_of(v, memo);
        memo.insert(v, Some(r));
        r
    }

    fn scalar_ok_of(&self, v: Value, memo: &mut Memo) -> bool {
        if let Some(&(b, k)) = self.phi_def.get(&v) {
            // Only a phi of one value (and itself, around a loop).
            let phi = &self.x.p.blocks[b].phis[k];
            let mut others = phi.args.iter().filter(|&&a| a != v);
            let Some(&first) = others.next() else {
                return false;
            };
            return others.all(|&a| a == first) && self.scalar_ok(first, memo);
        }
        let Some(&at) = self.def.get(&v) else {
            return false;
        };
        let inst = self.inst(at);
        let name = inst.name();
        let reads_ok = |memo: &mut Memo| {
            inst.reads.iter().all(|&(r, rv)| match r {
                Reg::S(code) => !special(code) && self.scalar_ok(rv, memo),
                Reg::Scc | Reg::V(_) => false,
            })
        };
        match &inst.fields {
            Fields::Sop { .. } => {
                let banned = [
                    "exec",
                    "pc",
                    "getreg",
                    "setreg",
                    "bcnt",
                    "ff0",
                    "ff1",
                    "flbit",
                    "wqm",
                    "quadmask",
                    "movrel",
                    "cbranch",
                    "sendmsg",
                    "memtime",
                    "memrealtime",
                    "ttrace",
                ];
                !banned.iter().any(|n| name.contains(n)) && reads_ok(memo)
            }
            Fields::Smem { .. } => {
                (name.starts_with("s_load_") || name.starts_with("s_buffer_load_"))
                    && reads_ok(memo)
            }
            _ => false,
        }
    }

    /// Condition 2: only instructions running under an EXEC no wider than
    /// `exec` see `result`, or scalars made from it.
    fn seen_only_inside(&self, result: Value, exec: (Value, Value)) -> bool {
        let inside = self.masks_inside(exec);
        let mut tainted = vec![result];
        let mut done: HashSet<Value> = HashSet::new();
        while let Some(t) = tainted.pop() {
            if !done.insert(t) {
                continue;
            }
            for &(b, k) in self.phi_uses.get(&t).into_iter().flatten() {
                let phi = &self.x.p.blocks[b].phis[k];
                if !matches!(phi.reg, Reg::S(c) if !special(c)) {
                    return false;
                }
                tainted.push(phi.value);
            }
            for &at in self.uses.get(&t).into_iter().flatten() {
                let inst = self.inst(at);
                if let Some((lo, hi)) = self.exec(inst) {
                    // A vector instruction: only under a mask inside the
                    // read's, and writing no scalar (nor a lane by index).
                    let writes_scalar = inst.writes.iter().any(|w| !matches!(w.0, Reg::V(_)));
                    if !inside.contains(&lo)
                        || !inside.contains(&hi)
                        || writes_scalar
                        || inst.name().contains("lane")
                    {
                        return false;
                    }
                    continue;
                }
                let Fields::Sop { .. } = &inst.fields else {
                    return false;
                };
                let name = inst.name();
                let to_special = inst
                    .writes
                    .iter()
                    .any(|w| matches!(w.0, Reg::S(c) if special(c)));
                if to_special
                    || ["exec", "pc", "cbranch", "setreg", "sendmsg", "movrel"]
                        .iter()
                        .any(|n| name.contains(n))
                {
                    return false;
                }
                tainted.extend(inst.writes.iter().map(|w| w.1));
            }
        }
        true
    }

    /// The EXEC-like mask dwords no wider than `exec`'s: `exec` itself, and
    /// what compares into EXEC, `s_and_saveexec`, ANDs, `s_andn2` and copies
    /// make of them, and phis of them (the greatest such set: around a loop a
    /// phi of masks no wider stays so).
    fn masks_inside(&self, exec: (Value, Value)) -> HashSet<Value> {
        let p = self.x.p;
        // Every SGPR dword value an operation of those kinds writes, by the
        // rule that keeps it inside.
        let mut rules: HashMap<Value, Vec<Value>> = HashMap::new();
        let mut any: HashMap<Value, Vec<Value>> = HashMap::new();
        for block in &p.blocks {
            for phi in &block.phis {
                if matches!(phi.reg, Reg::S(_)) {
                    rules.insert(phi.value, phi.args.clone());
                }
            }
            for inst in &block.insts {
                let name = inst.name();
                let read = |code: u8| inst.read(Reg::S(code));
                let src = |k: usize| match &inst.fields {
                    Fields::Sop { src, .. } => src.get(k).copied(),
                    _ => None,
                };
                let dword = |op: Option<Operand>, d: u8| match op {
                    Some(Operand::Scalar { code, .. }) => read(code + d),
                    _ => None,
                };
                for &(r, w) in &inst.writes {
                    let Reg::S(code) = r else { continue };
                    let d = code & 1;
                    let exec_in = read(EXEC + d);
                    if (code == EXEC || code == EXEC + 1)
                        && (name.starts_with("v_cmpx_") || name == "s_and_saveexec_b64")
                    {
                        if let Some(e) = exec_in {
                            rules.insert(w, vec![e]);
                        }
                        continue;
                    }
                    match name {
                        "s_and_saveexec_b64" => {
                            // The saved copy of the old EXEC.
                            if let Some(e) = exec_in {
                                rules.insert(w, vec![e]);
                            }
                        }
                        "s_mov_b64" => {
                            if let Some(s) = dword(src(0), d) {
                                rules.insert(w, vec![s]);
                            }
                        }
                        "s_andn2_b64" => {
                            if let Some(s) = dword(src(0), d) {
                                rules.insert(w, vec![s]);
                            }
                        }
                        "s_and_b64" => {
                            let both: Vec<Value> = [dword(src(0), d), dword(src(1), d)]
                                .into_iter()
                                .flatten()
                                .collect();
                            any.insert(w, both);
                        }
                        _ => {}
                    }
                }
            }
        }
        let mut inside: HashSet<Value> = rules.keys().chain(any.keys()).copied().collect();
        inside.insert(exec.0);
        inside.insert(exec.1);
        loop {
            let before = inside.len();
            let keep: Vec<Value> = inside
                .iter()
                .copied()
                .filter(|v| {
                    *v == exec.0
                        || *v == exec.1
                        || rules.get(v).is_some_and(|args| {
                            !args.is_empty() && args.iter().all(|a| inside.contains(a))
                        })
                        || any
                            .get(v)
                            .is_some_and(|args| args.iter().any(|a| inside.contains(a)))
                })
                .collect();
            inside = keep.into_iter().collect();
            if inside.len() == before {
                return inside;
            }
        }
    }
}
