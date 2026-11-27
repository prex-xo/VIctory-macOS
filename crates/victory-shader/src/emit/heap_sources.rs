//! Generic extraction of `ResourceDescriptorHeap` texture index sources from
//! pixel-shader IR.
//!
//! One-time composite/bake pixel passes often bind no texture tables at all and
//! instead index textures through `ResourceDescriptorHeap[idx]` where `idx`
//! comes from:
//! - a uint32 word inside a bound buffer (`HeapIndexSource::CbvWord`), and/or
//! - a uint32 field inside a structured buffer iterated by a loop whose bounds
//!   come from bound buffer words or constants
//!   (`HeapIndexSource::StructuredLoopField`).
//!
//! Walking backward from each heap-direct texture descriptor load discovers
//! these bindings generically without any shader hashes, struct sizes, or
//! title-specific offsets.

use std::collections::{BTreeSet, HashMap, HashSet};

use crate::gfx10::{Fields, NULL, Operand};
use crate::ir::{Block, Exit, Inst, Phi, Program, Reg, Value};
use crate::native::{NativeProgram, RootType};
use crate::resource::{Address, Resources, Role, Site, Source};

/// Maximum backward data-flow walk depth from a heap index expression.
const MAX_DEPTH: usize = 10;

/// Identifies a buffer binding read by a shader (either a descriptor table
/// slot or a root descriptor).
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum BufferBinding {
    /// Descriptor at byte offset `table_offset_dwords * 4` in root table `slot`.
    Table {
        slot: u8,
        table_offset_dwords: u32,
    },
    /// Root descriptor (root CBV or root SRV) at root parameter `slot`.
    RootDescriptor { slot: u8 },
}

/// A uint32 scalar known at draw time: a compile-time constant, a word read
/// from a bound buffer, or a root constant.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum ScalarSource {
    Const(u32),
    BufferWord {
        binding: BufferBinding,
        byte_offset: u32,
    },
    RootConstant {
        slot: u8,
        dword: u32,
    },
}

/// Describes where a pixel shader obtains a `ResourceDescriptorHeap` texture
/// index at draw time, extracted generically from its IR.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum HeapIndexSource {
    /// Direct uint32 word at `byte_offset` in `binding`, plus optional constant
    /// `index_offset` (for e.g. `ResourceDescriptorHeap[cb.base + 1]`).
    CbvWord {
        binding: BufferBinding,
        byte_offset: u32,
        index_offset: u32,
    },
    /// Uint32 field at `field_byte_offset` within each record of a structured
    /// buffer `buffer`, iterated across `[start, end)` (or `[start, start + end)`
    /// when `end_is_count` is true), plus `inclusive_extra` (e.g. `1` for `<=`
    /// bounds), plus optional constant `index_offset`.
    /// `stride` is `Some(bytes)` when the shader multiplies or steps by a byte
    /// stride explicitly, or `None` when indexed via hardware `idxen` (in which
    /// case the bound SRV descriptor's `StructureByteStride` supplies it).
    StructuredLoopField {
        buffer: BufferBinding,
        stride: Option<u32>,
        field_byte_offset: u32,
        start: ScalarSource,
        end: ScalarSource,
        end_is_count: bool,
        inclusive_extra: u32,
        index_offset: u32,
    },
}

/// Extracts all statically discoverable `ResourceDescriptorHeap` texture index
/// sources from `p`, sorted and deduplicated deterministically.
pub fn extract(native: &NativeProgram, p: &Program, res: &Resources) -> Vec<HeapIndexSource> {
    let cx = Analyzer::new(native, p, res);
    cx.run()
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum OperandVal {
    Const(u32),
    Val(Value),
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct LinearExpr {
    var: Value,
    scale: u32,
    add: u32,
}

#[derive(Debug, Clone)]
struct NaturalLoop {
    header: usize,
    body: HashSet<usize>,
    entry_preds: Vec<usize>,
    back_preds: Vec<usize>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum CmpOp {
    Lt,
    Le,
    Gt,
    Ge,
    Eq,
    Ne,
}

impl CmpOp {
    fn swap(self) -> Self {
        match self {
            CmpOp::Lt => CmpOp::Gt,
            CmpOp::Le => CmpOp::Ge,
            CmpOp::Gt => CmpOp::Lt,
            CmpOp::Ge => CmpOp::Le,
            CmpOp::Eq => CmpOp::Eq,
            CmpOp::Ne => CmpOp::Ne,
        }
    }

    fn negate(self) -> Self {
        match self {
            CmpOp::Lt => CmpOp::Ge,
            CmpOp::Le => CmpOp::Gt,
            CmpOp::Gt => CmpOp::Le,
            CmpOp::Ge => CmpOp::Lt,
            CmpOp::Eq => CmpOp::Ne,
            CmpOp::Ne => CmpOp::Eq,
        }
    }
}

#[derive(Debug, Clone, Copy)]
struct LoopBound {
    end: ScalarSource,
    end_is_count: bool,
    inclusive_extra: u32,
}

#[derive(Debug, Clone, Copy)]
struct BranchCmp {
    op: CmpOp,
    lhs: OperandVal,
    rhs: OperandVal,
    continue_when_true: bool,
}

#[derive(Debug, Clone, Copy)]
struct InductionVar {
    phi_value: Value,
    init: OperandVal,
    step: i32,
    next: Value,
}

struct Analyzer<'a> {
    native: &'a NativeProgram,
    p: &'a Program,
    res: &'a Resources,
    defs: Vec<Option<Site>>,
    phis: Vec<Option<(usize, usize)>>,
    scalar_map: HashMap<Site, Address>,
    desc_map: HashMap<Site, &'a Source>,
    root_consts: HashMap<Value, (u8, u32)>,
    loops_by_header: HashMap<usize, NaturalLoop>,
}

impl<'a> Analyzer<'a> {
    fn new(native: &'a NativeProgram, p: &'a Program, res: &'a Resources) -> Self {
        let max_val = p
            .inputs
            .iter()
            .map(|i| i.1.0 as usize)
            .chain(p.blocks.iter().flat_map(|b| {
                b.phis
                    .iter()
                    .map(|phi| phi.value.0 as usize)
                    .chain(b.insts.iter().flat_map(|i| i.writes.iter().map(|w| w.1.0 as usize)))
            }))
            .max()
            .map_or(0, |m| m + 1);

        let mut defs = vec![None; max_val];
        let mut phis = vec![None; max_val];
        for (b, block) in p.blocks.iter().enumerate() {
            for (i, phi) in block.phis.iter().enumerate() {
                phis[phi.value.0 as usize] = Some((b, i));
            }
            for (i, inst) in block.insts.iter().enumerate() {
                for &(_, v) in &inst.writes {
                    defs[v.0 as usize] = Some((b, i));
                }
            }
        }

        let scalar_map: HashMap<Site, Address> = res.scalar.iter().copied().collect();
        let desc_map: HashMap<Site, &'a Source> = res
            .descriptors
            .iter()
            .filter(|d| d.role == Role::Resource)
            .map(|d| (d.site, &d.source))
            .collect();
        let root_consts: HashMap<Value, (u8, u32)> = res
            .params
            .iter()
            .filter_map(|&(v, param, dword)| {
                let rp = native.layout.get(param)?;
                (rp.root == RootType::Constants).then_some((v, (rp.slot, dword)))
            })
            .collect();
        let loops_by_header = find_natural_loops(&p.blocks);

        Self {
            native,
            p,
            res,
            defs,
            phis,
            scalar_map,
            desc_map,
            root_consts,
            loops_by_header,
        }
    }

    fn run(&self) -> Vec<HeapIndexSource> {
        let mut out = BTreeSet::new();
        for d in &self.res.descriptors {
            if d.role != Role::Resource {
                continue;
            }
            let inst = self.inst(d.site);
            if !is_texture_read_op(inst) {
                continue;
            }
            for leaf in d.source.leaves() {
                let (Source::Loaded { load, first } | Source::Fetched { load, first }) = leaf
                else {
                    continue;
                };
                let Some(addr) = self.scalar_map.get(load) else {
                    continue;
                };
                if !self
                    .native
                    .layout
                    .get(addr.param)
                    .is_some_and(|rp| rp.root == RootType::ResourceHeap)
                {
                    continue;
                }
                let Some(dyn_val) = addr.dynamic else {
                    continue;
                };
                let base_byte_off = addr.offset.wrapping_add(4 * u32::from(*first));
                let mut unscaled = Vec::new();
                let mut seen_scale = HashSet::new();
                self.unwrap_heap_scale(dyn_val, base_byte_off, 0, &mut seen_scale, &mut unscaled);
                for (idx_val, index_offset) in unscaled {
                    let mut seen_idx = HashSet::new();
                    self.trace_index(idx_val, index_offset, 0, &mut seen_idx, &mut out);
                }
            }
        }
        out.into_iter().collect()
    }

    fn inst(&self, site: Site) -> &'a Inst {
        &self.p.blocks[site.0].insts[site.1]
    }

    fn def(&self, v: Value) -> Option<Site> {
        self.defs.get(v.0 as usize).copied().flatten()
    }

    fn phi(&self, v: Value) -> Option<(usize, &'a Phi)> {
        let (b, i) = self.phis.get(v.0 as usize).copied().flatten()?;
        Some((b, &self.p.blocks[b].phis[i]))
    }

    /// Walks backward from the byte offset of a `ResourceDescriptorHeap` load
    /// to unwrap the `idx << 5` / `idx * 32` descriptor-size scaling.
    fn unwrap_heap_scale(
        &self,
        v: Value,
        extra_bytes: u32,
        depth: usize,
        visited: &mut HashSet<Value>,
        out: &mut Vec<(Value, u32)>,
    ) {
        if depth > MAX_DEPTH || !visited.insert(v) {
            return;
        }
        if let Some((_, phi)) = self.phi(v) {
            for &arg in &phi.args {
                self.unwrap_heap_scale(arg, extra_bytes, depth + 1, visited, out);
            }
            return;
        }
        let Some(site) = self.def(v) else {
            return;
        };
        let inst = self.inst(site);
        let name = inst.name();

        // Direct scaling by 32 (<< 5 or * 32).
        if let Some((idx, add_bytes)) = self.match_scale32(inst) {
            let total_bytes = extra_bytes.wrapping_add(add_bytes);
            out.push((idx, total_bytes >> 5));
            return;
        }

        // Transparent copies and lane broadcasts.
        if let Some(src) = self.copy_source(inst, v) {
            if let OperandVal::Val(sv) = src {
                self.unwrap_heap_scale(sv, extra_bytes, depth + 1, visited, out);
            }
            return;
        }

        // Constant additions/subtractions/low-bit ORs on the byte offset.
        if let Some((base, delta)) = self.const_add_sub(inst, v) {
            self.unwrap_heap_scale(
                base,
                extra_bytes.wrapping_add(delta),
                depth + 1,
                visited,
                out,
            );
            return;
        }
        if matches!(name, "s_or_b32" | "v_or_b32")
            && let Some((a, b)) = self.binary_operands(inst)
        {
            match (self.eval_operand(a), self.eval_operand(b)) {
                (OperandVal::Val(x), OperandVal::Const(c))
                | (OperandVal::Const(c), OperandVal::Val(x))
                    if c < 32 =>
                {
                    self.unwrap_heap_scale(x, extra_bytes, depth + 1, visited, out);
                }
                _ => {}
            }
            return;
        }

        // Selects: walk both arms.
        if let Some((a, b)) = self.select_operands(inst, v) {
            for arm in [a, b] {
                if let OperandVal::Val(av) = arm {
                    self.unwrap_heap_scale(av, extra_bytes, depth + 1, visited, out);
                }
            }
        }
    }

    /// Matches instructions computing `idx * 32 + add_bytes` (`<< 5` or `* 32`).
    fn match_scale32(&self, inst: &Inst) -> Option<(Value, u32)> {
        let name = inst.name();
        match &inst.fields {
            Fields::Sop { src, imm, .. } => match name {
                "s_lshl_b32" | "s_lshl_b64" => {
                    let a = self.eval_operand(self.operand_val(inst, src.first()?)?);
                    let b = self.eval_operand(self.operand_val(inst, src.get(1)?)?);
                    match (a, b) {
                        (OperandVal::Val(x), OperandVal::Const(5)) => Some((x, 0)),
                        _ => None,
                    }
                }
                "s_mul_i32" => {
                    let a = self.eval_operand(self.operand_val(inst, src.first()?)?);
                    let b = self.eval_operand(self.operand_val(inst, src.get(1)?)?);
                    match (a, b) {
                        (OperandVal::Val(x), OperandVal::Const(32))
                        | (OperandVal::Const(32), OperandVal::Val(x)) => Some((x, 0)),
                        _ => None,
                    }
                }
                "s_mulk_i32" if *imm == Some(32) => {
                    let dst = inst.fields_sop_dst_read(inst)?;
                    Some((dst, 0))
                }
                _ => None,
            },
            Fields::Valu(f) if f.abs == 0 && f.neg == 0 && f.sdwa.is_none() && f.dpp.is_none() => {
                match name {
                    "v_lshlrev_b32" | "v_lshlrev_b64" => {
                        let shift = self.eval_operand(self.operand_val(inst, f.src.first()?)?);
                        let val = self.eval_operand(self.operand_val(inst, f.src.get(1)?)?);
                        match (shift, val) {
                            (OperandVal::Const(5), OperandVal::Val(x)) => Some((x, 0)),
                            _ => None,
                        }
                    }
                    "v_lshl_b32" => {
                        let val = self.eval_operand(self.operand_val(inst, f.src.first()?)?);
                        let shift = self.eval_operand(self.operand_val(inst, f.src.get(1)?)?);
                        match (val, shift) {
                            (OperandVal::Val(x), OperandVal::Const(5)) => Some((x, 0)),
                            _ => None,
                        }
                    }
                    "v_lshl_add_u32" => {
                        let val = self.eval_operand(self.operand_val(inst, f.src.first()?)?);
                        let shift = self.eval_operand(self.operand_val(inst, f.src.get(1)?)?);
                        let add = self.eval_operand(self.operand_val(inst, f.src.get(2)?)?);
                        match (val, shift, add) {
                            (OperandVal::Val(x), OperandVal::Const(5), OperandVal::Const(c)) => {
                                Some((x, c))
                            }
                            _ => None,
                        }
                    }
                    "v_mul_lo_u32" | "v_mul_lo_i32" | "v_mul_u32_u24" | "v_mul_i32_i24" => {
                        let a = self.eval_operand(self.operand_val(inst, f.src.first()?)?);
                        let b = self.eval_operand(self.operand_val(inst, f.src.get(1)?)?);
                        match (a, b) {
                            (OperandVal::Val(x), OperandVal::Const(32))
                            | (OperandVal::Const(32), OperandVal::Val(x)) => Some((x, 0)),
                            _ => None,
                        }
                    }
                    "v_mad_u32_u24" | "v_mad_i32_i24" | "v_mad_u64_u32" | "v_mad_i64_i32" => {
                        let a = self.eval_operand(self.operand_val(inst, f.src.first()?)?);
                        let b = self.eval_operand(self.operand_val(inst, f.src.get(1)?)?);
                        let c = self.eval_operand(self.operand_val(inst, f.src.get(2)?)?);
                        match (a, b, c) {
                            (OperandVal::Val(x), OperandVal::Const(32), OperandVal::Const(add))
                            | (OperandVal::Const(32), OperandVal::Val(x), OperandVal::Const(add)) => {
                                Some((x, add))
                            }
                            _ => None,
                        }
                    }
                    _ => None,
                }
            }
            _ => None,
        }
    }

    /// Walks backward from an unscaled uint32 heap index expression to find
    /// `CbvWord` and `StructuredLoopField` sources.
    fn trace_index(
        &self,
        v: Value,
        index_offset: u32,
        depth: usize,
        visited: &mut HashSet<Value>,
        out: &mut BTreeSet<HeapIndexSource>,
    ) {
        if depth > MAX_DEPTH || !visited.insert(v) {
            return;
        }
        if let Some((_, phi)) = self.phi(v) {
            for &arg in &phi.args {
                self.trace_index(arg, index_offset, depth + 1, visited, out);
            }
            return;
        }
        let Some(site) = self.def(v) else {
            return;
        };
        let inst = self.inst(site);

        // 1. Buffer load (scalar CBV/SRV load or vector structured buffer load).
        if self.try_extract_buffer_load(site, inst, v, index_offset, out) {
            return;
        }

        // 2. Transparent copies / lane broadcasts.
        if let Some(src) = self.copy_source(inst, v) {
            if let OperandVal::Val(sv) = src {
                self.trace_index(sv, index_offset, depth + 1, visited, out);
            }
            return;
        }

        // 3. Constant offset on the heap index itself (e.g. `cb.base + 1`).
        if let Some((base, delta)) = self.const_add_sub(inst, v) {
            self.trace_index(
                base,
                index_offset.wrapping_add(delta),
                depth + 1,
                visited,
                out,
            );
            return;
        }

        // 4. Clamping with a constant (`min(idx, max_slot)` / `max(idx, 0)`).
        if matches!(
            inst.name(),
            "s_min_u32" | "s_max_u32" | "v_min_u32" | "v_max_u32"
        ) && let Some((a, b)) = self.binary_operands(inst)
        {
            match (self.eval_operand(a), self.eval_operand(b)) {
                (OperandVal::Val(x), OperandVal::Const(_))
                | (OperandVal::Const(_), OperandVal::Val(x)) => {
                    self.trace_index(x, index_offset, depth + 1, visited, out);
                }
                _ => {}
            }
            return;
        }

        // 5. Conditional selects: walk both arms.
        if let Some((a, b)) = self.select_operands(inst, v) {
            for arm in [a, b] {
                if let OperandVal::Val(av) = arm {
                    self.trace_index(av, index_offset, depth + 1, visited, out);
                }
            }
        }
    }

    /// Inspects a scalar or vector buffer load defining `v`. Returns `true` if
    /// `inst` is a buffer load instruction (whether or not a source was emitted).
    fn try_extract_buffer_load(
        &self,
        site: Site,
        inst: &Inst,
        v: Value,
        index_offset: u32,
        out: &mut BTreeSet<HeapIndexSource>,
    ) -> bool {
        let name = inst.name();
        match &inst.fields {
            Fields::Smem {
                data: Some(Operand::Scalar { code, dwords }),
                offset,
                soffset,
                ..
            } if name.starts_with("s_buffer_load_dword") => {
                let Some(k) = (0..*dwords).find(|&k| inst.write(Reg::S(code + k)) == Some(v))
                else {
                    return true;
                };
                let Some(src) = self.desc_map.get(&site).copied() else {
                    return true;
                };
                let bindings = self.resolve_buffer_bindings(src);
                if bindings.is_empty() {
                    return true;
                }
                let Some(so) = self.operand_val(inst, soffset) else {
                    return true;
                };
                let so = self.eval_operand(so);
                for (binding, built_off) in bindings {
                    let base_off = (*offset as u32)
                        .wrapping_add(built_off)
                        .wrapping_add(4 * u32::from(k));
                    match so {
                        OperandVal::Const(c) => {
                            out.insert(HeapIndexSource::CbvWord {
                                binding,
                                byte_offset: base_off.wrapping_add(c),
                                index_offset,
                            });
                        }
                        OperandVal::Val(dyn_off) => {
                            self.extract_structured_from_offset(
                                binding,
                                dyn_off,
                                false,
                                base_off,
                                index_offset,
                                out,
                            );
                        }
                    }
                }
                true
            }
            Fields::Buffer {
                data: Some(Operand::Vector { reg, dwords }),
                addr: Operand::Vector { reg: areg, .. },
                soffset,
                offset,
                offen,
                idxen,
                ..
            } if name.starts_with("buffer_load_dword") => {
                let Some(k) = (0..*dwords).find(|&k| inst.write(Reg::V(reg + k)) == Some(v))
                else {
                    return true;
                };
                let Some(src) = self.desc_map.get(&site).copied() else {
                    return true;
                };
                let bindings = self.resolve_buffer_bindings(src);
                if bindings.is_empty() {
                    return true;
                }
                let idx_val = if *idxen {
                    let Some(iv) = inst.read(Reg::V(*areg)) else {
                        return true;
                    };
                    Some(self.eval_operand(OperandVal::Val(iv)))
                } else {
                    None
                };
                let off_val = if *offen {
                    let Some(ov) = inst.read(Reg::V(*areg + u8::from(*idxen))) else {
                        return true;
                    };
                    Some(self.eval_operand(OperandVal::Val(ov)))
                } else {
                    None
                };
                let Some(soff) = self.operand_val(inst, soffset) else {
                    return true;
                };
                let soff = self.eval_operand(soff);

                for (binding, built_off) in bindings {
                    let mut const_off = u32::from(*offset)
                        .wrapping_add(built_off)
                        .wrapping_add(4 * u32::from(k));
                    let mut dyn_off: Option<Value> = None;
                    let mut too_dynamic = false;
                    for maybe in [off_val, Some(soff)].into_iter().flatten() {
                        match maybe {
                            OperandVal::Const(c) => const_off = const_off.wrapping_add(c),
                            OperandVal::Val(dv) => {
                                if dyn_off.is_some() {
                                    too_dynamic = true;
                                } else {
                                    dyn_off = Some(dv);
                                }
                            }
                        }
                    }
                    if too_dynamic {
                        continue;
                    }
                    match (idx_val, dyn_off) {
                        // Hardware-indexed structured buffer: `vindex` selects record,
                        // `const_off` is the field byte offset.
                        (Some(OperandVal::Val(iv)), None) => {
                            self.extract_structured_from_offset(
                                binding,
                                iv,
                                true,
                                const_off,
                                index_offset,
                                out,
                            );
                        }
                        // Constant record index 0 with no dynamic offset.
                        (Some(OperandVal::Const(0)), None) | (None, None) => {
                            out.insert(HeapIndexSource::CbvWord {
                                binding,
                                byte_offset: const_off,
                                index_offset,
                            });
                        }
                        // Constant non-zero hardware index: 1-element structured access.
                        (Some(OperandVal::Const(rec_idx)), None) => {
                            out.insert(HeapIndexSource::StructuredLoopField {
                                buffer: binding,
                                stride: None,
                                field_byte_offset: const_off,
                                start: ScalarSource::Const(rec_idx),
                                end: ScalarSource::Const(1),
                                end_is_count: true,
                                inclusive_extra: 0,
                                index_offset,
                            });
                        }
                        // Byte-offset indexed structured buffer (`offen` or `soffset`).
                        (None, Some(dv)) => {
                            self.extract_structured_from_offset(
                                binding,
                                dv,
                                false,
                                const_off,
                                index_offset,
                                out,
                            );
                        }
                        (Some(_), Some(_)) => {}
                    }
                }
                true
            }
            _ => false,
        }
    }

    /// Resolves a dynamic index/byte-offset into a `StructuredLoopField`.
    fn extract_structured_from_offset(
        &self,
        buffer: BufferBinding,
        dyn_val: Value,
        idxen: bool,
        base_field_offset: u32,
        index_offset: u32,
        out: &mut BTreeSet<HeapIndexSource>,
    ) {
        let linear = self.decompose_linear(dyn_val, 0);
        let Some((header, _)) = self.phi(linear.var) else {
            // Not a loop phi: check if it's a single record indexed by a scalar
            // CBV/root-constant value (e.g. `records[cb.end].texIdx`).
            if let Some(start @ (ScalarSource::BufferWord { .. } | ScalarSource::RootConstant { .. })) =
                self.trace_scalar(OperandVal::Val(linear.var), 0)
            {
                let (stride, field_byte_offset) = if idxen {
                    if linear.scale != 1 || linear.add != 0 {
                        return;
                    }
                    (None, base_field_offset)
                } else {
                    if linear.scale == 0 {
                        return;
                    }
                    (Some(linear.scale), base_field_offset.wrapping_add(linear.add))
                };
                out.insert(HeapIndexSource::StructuredLoopField {
                    buffer,
                    stride,
                    field_byte_offset,
                    start,
                    end: ScalarSource::Const(1),
                    end_is_count: true,
                    inclusive_extra: 0,
                    index_offset,
                });
            }
            return;
        };

        let Some(lp) = self.loops_by_header.get(&header) else {
            return;
        };
        let Some(iv) = self.recognize_iv(lp, linear.var) else {
            return;
        };
        if iv.step <= 0 {
            return;
        }
        let step = iv.step as u32;

        if step == 1 {
            // Record-index IV: `i = start, start + 1, ...`
            let Some(start) = self.trace_scalar(iv.init, 0) else {
                return;
            };
            let Some(bound) = self.find_loop_bound(lp, Some(iv), None, Some(iv.init)) else {
                return;
            };
            let (stride, field_byte_offset) = if idxen {
                if linear.scale != 1 || linear.add != 0 {
                    return;
                }
                (None, base_field_offset)
            } else {
                if linear.scale == 0 {
                    return;
                }
                (Some(linear.scale), base_field_offset.wrapping_add(linear.add))
            };
            out.insert(HeapIndexSource::StructuredLoopField {
                buffer,
                stride,
                field_byte_offset,
                start,
                end: bound.end,
                end_is_count: bound.end_is_count,
                inclusive_extra: bound.inclusive_extra,
                index_offset,
            });
        } else if !idxen && linear.scale == 1 {
            // Strength-reduced byte-offset IV: `off = start * stride + init_off, off + stride, ...`
            let stride = step;
            let (start, start_op, init_off) = match iv.init {
                OperandVal::Const(c) => (ScalarSource::Const(0), Some(OperandVal::Const(0)), c),
                OperandVal::Val(init_v) => {
                    let init_lin = self.decompose_linear(init_v, 0);
                    if init_lin.scale == stride {
                        let Some(s) = self.trace_scalar(OperandVal::Val(init_lin.var), 0) else {
                            return;
                        };
                        (s, Some(OperandVal::Val(init_lin.var)), init_lin.add)
                    } else {
                        return;
                    }
                }
            };
            let Some(bound) = self.find_loop_bound(lp, None, Some((iv, stride, init_off)), start_op)
            else {
                return;
            };
            out.insert(HeapIndexSource::StructuredLoopField {
                buffer,
                stride: Some(stride),
                field_byte_offset: base_field_offset.wrapping_add(linear.add).wrapping_add(init_off),
                start,
                end: bound.end,
                end_is_count: bound.end_is_count,
                inclusive_extra: bound.inclusive_extra,
                index_offset,
            });
        }
    }

    /// Recognizes a basic induction variable `iv = phi(init, iv + step)` in
    /// loop `lp`.
    fn recognize_iv(&self, lp: &NaturalLoop, v: Value) -> Option<InductionVar> {
        let (b, phi) = self.phi(v)?;
        if b != lp.header || lp.entry_preds.is_empty() || lp.back_preds.is_empty() {
            return None;
        }
        let block = &self.p.blocks[b];
        let mut init: Option<OperandVal> = None;
        let mut step: Option<(i32, Value)> = None;

        for (&pred, &arg) in block.preds.iter().zip(&phi.args) {
            if lp.entry_preds.contains(&pred) {
                let val = self.eval_operand(OperandVal::Val(arg));
                match init {
                    None => init = Some(val),
                    Some(prev) if prev == val => {}
                    _ => return None,
                }
            } else if lp.back_preds.contains(&pred) {
                let delta = self.step_from(arg, v)?;
                match step {
                    None => step = Some((delta, arg)),
                    Some((prev_d, _)) if prev_d == delta => {}
                    _ => return None,
                }
            }
        }
        let (init, (step, next)) = (init?, step?);
        (step != 0).then_some(InductionVar {
            phi_value: v,
            init,
            step,
            next,
        })
    }

    /// If `next` computes `base + delta` (with signed constant `delta`), returns
    /// `Some(delta)`.
    fn step_from(&self, next: Value, base: Value) -> Option<i32> {
        let mut cur = next;
        let mut acc: i32 = 0;
        for _ in 0..4 {
            if cur == base {
                return (acc != 0).then_some(acc);
            }
            let site = self.def(cur)?;
            let inst = self.inst(site);
            if let Some(OperandVal::Val(src)) = self.copy_source(inst, cur) {
                cur = src;
                continue;
            }
            if let Some((x, delta)) = self.const_add_sub(inst, cur) {
                acc = acc.wrapping_add(delta as i32);
                cur = x;
                continue;
            }
            return None;
        }
        (cur == base && acc != 0).then_some(acc)
    }

    /// Finds the upper bound of loop `lp` by inspecting its exit branches
    /// against either a step-1 index IV, a strength-reduced byte-offset IV, or
    /// any sibling index/countdown IV in the same loop header.
    fn find_loop_bound(
        &self,
        lp: &NaturalLoop,
        index_iv: Option<InductionVar>,
        byte_iv: Option<(InductionVar, u32, u32)>,
        start_init: Option<OperandVal>,
    ) -> Option<LoopBound> {
        // Collect all basic IVs in the loop header so a strength-reduced byte IV
        // can also inherit its bound from a sibling counter or countdown IV.
        let header_ivs: Vec<InductionVar> = self.p.blocks[lp.header]
            .phis
            .iter()
            .filter_map(|phi| self.recognize_iv(lp, phi.value))
            .collect();

        let mut exit_branches: Vec<(usize, usize, usize)> = Vec::new();
        for &b in &lp.body {
            if let Exit::Branch { taken, not_taken } = self.p.blocks[b].exit {
                let t_in = lp.body.contains(&taken);
                let nt_in = lp.body.contains(&not_taken);
                if t_in != nt_in {
                    exit_branches.push((b, taken, not_taken));
                }
            }
        }

        // First try matching directly against the primary IV (index or byte).
        for &(b, taken, _not_taken) in &exit_branches {
            let continue_when_taken = lp.body.contains(&taken);
            let Some(cmp) = self.extract_branch_compare(b, continue_when_taken) else {
                continue;
            };

            if let Some(iv) = index_iv
                && let Some(bound) = self.match_index_bound(iv, cmp)
            {
                return Some(bound);
            }
            if let Some((biv, stride, init_off)) = byte_iv
                && let Some(bound) = self.match_byte_bound(biv, stride, init_off, cmp)
            {
                return Some(bound);
            }
        }

        // Second: try matching against any sibling step-1 count-up IV or
        // step-(-1) countdown IV in the same header.
        for &(b, taken, not_taken) in &exit_branches {
            let continue_when_taken = lp.body.contains(&taken);
            let Some(cmp) = self.extract_branch_compare(b, continue_when_taken) else {
                continue;
            };
            let _ = not_taken;

            for &sib in &header_ivs {
                if sib.step == 1 {
                    if let Some(bound) = self.match_index_bound(sib, cmp) {
                        return Some(bound);
                    }
                } else if sib.step == -1
                    && let Some(bound) = self.match_countdown_bound(sib, start_init, cmp)
                {
                    return Some(bound);
                }
            }
        }

        None
    }

    fn match_index_bound(&self, iv: InductionVar, cmp: BranchCmp) -> Option<LoopBound> {
        let is_iv = |op: OperandVal| match op {
            OperandVal::Val(v) => {
                self.is_same_or_copy(v, iv.phi_value) || self.is_same_or_copy(v, iv.next)
            }
            OperandVal::Const(_) => false,
        };
        let (norm_op, bound_op) = if is_iv(cmp.lhs) {
            (cmp.op, cmp.rhs)
        } else if is_iv(cmp.rhs) {
            (cmp.op.swap(), cmp.lhs)
        } else {
            return None;
        };
        let cont_op = if cmp.continue_when_true {
            norm_op
        } else {
            norm_op.negate()
        };
        let base_extra = match cont_op {
            CmpOp::Lt | CmpOp::Ne => 0u32,
            CmpOp::Le => 1u32,
            _ => return None,
        };
        self.resolve_bound_expr(bound_op, Some(iv.init), base_extra)
    }

    fn match_byte_bound(
        &self,
        biv: InductionVar,
        stride: u32,
        init_off: u32,
        cmp: BranchCmp,
    ) -> Option<LoopBound> {
        let is_biv = |op: OperandVal| match op {
            OperandVal::Val(v) => {
                self.is_same_or_copy(v, biv.phi_value) || self.is_same_or_copy(v, biv.next)
            }
            OperandVal::Const(_) => false,
        };
        let (norm_op, bound_op) = if is_biv(cmp.lhs) {
            (cmp.op, cmp.rhs)
        } else if is_biv(cmp.rhs) {
            (cmp.op.swap(), cmp.lhs)
        } else {
            return None;
        };
        let cont_op = if cmp.continue_when_true {
            norm_op
        } else {
            norm_op.negate()
        };
        let base_extra = match cont_op {
            CmpOp::Lt | CmpOp::Ne => 0u32,
            CmpOp::Le => 1u32,
            _ => return None,
        };
        match bound_op {
            OperandVal::Const(c) => {
                let diff = c.checked_sub(init_off)?;
                diff.is_multiple_of(stride).then_some(LoopBound {
                    end: ScalarSource::Const(diff / stride),
                    end_is_count: false,
                    inclusive_extra: base_extra,
                })
            }
            OperandVal::Val(bv) => {
                let lin = self.decompose_linear(bv, 0);
                if lin.scale != stride {
                    return None;
                }
                let extra_bytes = lin.add.wrapping_sub(init_off);
                if !extra_bytes.is_multiple_of(stride) {
                    return None;
                }
                let extra_recs = extra_bytes / stride;
                self.resolve_bound_expr(
                    OperandVal::Val(lin.var),
                    None,
                    base_extra.saturating_add(extra_recs),
                )
            }
        }
    }

    fn match_countdown_bound(
        &self,
        rem_iv: InductionVar,
        start_init: Option<OperandVal>,
        cmp: BranchCmp,
    ) -> Option<LoopBound> {
        let is_rem = |op: OperandVal| match op {
            OperandVal::Val(v) => {
                self.is_same_or_copy(v, rem_iv.phi_value) || self.is_same_or_copy(v, rem_iv.next)
            }
            OperandVal::Const(_) => false,
        };
        let (norm_op, bound_op) = if is_rem(cmp.lhs) {
            (cmp.op, cmp.rhs)
        } else if is_rem(cmp.rhs) {
            (cmp.op.swap(), cmp.lhs)
        } else {
            return None;
        };
        let cont_op = if cmp.continue_when_true {
            norm_op
        } else {
            norm_op.negate()
        };
        if !matches!((cont_op, bound_op), (CmpOp::Ne | CmpOp::Gt, OperandVal::Const(0))) {
            return None;
        }
        // If `rem_iv.init` is `end - start`, recover `[start, end)` directly.
        if let OperandVal::Val(rv) = rem_iv.init
            && let Some((a, b)) = self.sub_operands(rv)
        {
            let a = self.eval_operand(a);
            let b = self.eval_operand(b);
            if start_init.is_some_and(|s| s == b)
                && let Some(end) = self.trace_scalar(a, 0)
            {
                return Some(LoopBound {
                    end,
                    end_is_count: false,
                    inclusive_extra: 0,
                });
            }
        }
        let count = self.trace_scalar(rem_iv.init, 0)?;
        Some(LoopBound {
            end: count,
            end_is_count: true,
            inclusive_extra: 0,
        })
    }

    /// Resolves a loop bound operand into a `LoopBound`, recognizing `end`,
    /// `end + const`, and `start + count`.
    fn resolve_bound_expr(
        &self,
        bound: OperandVal,
        start_init: Option<OperandVal>,
        base_extra: u32,
    ) -> Option<LoopBound> {
        let bound = self.eval_operand(bound);
        if let Some(end) = self.trace_scalar(bound, 0) {
            return Some(LoopBound {
                end,
                end_is_count: false,
                inclusive_extra: base_extra,
            });
        }
        let OperandVal::Val(bv) = bound else {
            return None;
        };
        // Check `x + c` (constant offset).
        if let Some((x, delta)) = self.const_add_sub_val(bv)
            && (delta as i32) >= 0
            && let Some(end) = self.trace_scalar(OperandVal::Val(x), 0)
        {
            return Some(LoopBound {
                end,
                end_is_count: false,
                inclusive_extra: base_extra.saturating_add(delta),
            });
        }
        // Check `start + count`.
        if let Some((a, b)) = self.add_operands(bv) {
            let a = self.eval_operand(a);
            let b = self.eval_operand(b);
            if let Some(start) = start_init {
                if a == start
                    && let Some(count) = self.trace_scalar(b, 0)
                {
                    return Some(LoopBound {
                        end: count,
                        end_is_count: true,
                        inclusive_extra: base_extra,
                    });
                }
                if b == start
                    && let Some(count) = self.trace_scalar(a, 0)
                {
                    return Some(LoopBound {
                        end: count,
                        end_is_count: true,
                        inclusive_extra: base_extra,
                    });
                }
            }
        }
        None
    }

    /// Extracts the comparison driving the conditional branch at the end of
    /// block `b`.
    fn extract_branch_compare(&self, b: usize, continue_when_taken: bool) -> Option<BranchCmp> {
        let branch = self.p.blocks[b].insts.last()?;
        let (op, lhs, rhs, taken_when_cmp_true) = match branch.name() {
            "s_cbranch_scc0" | "s_cbranch_scc1" => {
                let scc = branch.read(Reg::Scc)?;
                let taken_when_one = branch.name() == "s_cbranch_scc1";
                let (op, lhs, rhs, invert) = self.extract_scc_compare(scc, 0)?;
                (op, lhs, rhs, taken_when_one ^ invert)
            }
            "s_cbranch_vccz" | "s_cbranch_vccnz" => {
                let vcc = branch.read(Reg::S(crate::ir::VCC))?;
                let taken_when_nz = branch.name() == "s_cbranch_vccnz";
                let (op, lhs, rhs, invert) = self.extract_mask_compare(vcc, 0)?;
                (op, lhs, rhs, taken_when_nz ^ invert)
            }
            "s_cbranch_execz" | "s_cbranch_execnz" => {
                let exec = branch.read(Reg::S(crate::ir::EXEC))?;
                let taken_when_nz = branch.name() == "s_cbranch_execnz";
                let (op, lhs, rhs, invert) = self.extract_mask_compare(exec, 0)?;
                (op, lhs, rhs, taken_when_nz ^ invert)
            }
            _ => return None,
        };
        Some(BranchCmp {
            op,
            lhs,
            rhs,
            continue_when_true: continue_when_taken == taken_when_cmp_true,
        })
    }

    fn extract_scc_compare(
        &self,
        scc: Value,
        depth: usize,
    ) -> Option<(CmpOp, OperandVal, OperandVal, bool)> {
        if depth > 6 {
            return None;
        }
        let site = self.def(scc)?;
        let inst = self.inst(site);
        let name = inst.name();
        let Fields::Sop { src, imm, .. } = &inst.fields else {
            return None;
        };
        if let Some(op) = parse_scalar_cmp(name) {
            if let Some(k) = imm {
                // `s_cmpk_* sdst, simm16`: compare register against immediate.
                let lhs = self.eval_operand(self.operand_val(inst, src.first()?)?);
                let rhs = OperandVal::Const(if name.ends_with("_i32") {
                    i32::from(*k as i16) as u32
                } else {
                    u32::from(*k)
                });
                return Some((op, lhs, rhs, false));
            }
            let lhs = self.eval_operand(self.operand_val(inst, src.first()?)?);
            let rhs = self.eval_operand(self.operand_val(inst, src.get(1)?)?);
            return Some((op, lhs, rhs, false));
        }
        // `s_sub_u32` / `s_add_u32` / `s_and_b32` / `s_or_b32` / `s_sub_i32`
        // can set SCC directly (e.g. `s_sub_u32 rem, rem, 1` or `s_and_b64`
        // setting SCC = (result != 0)).
        if matches!(
            name,
            "s_and_b32" | "s_and_b64" | "s_andn2_b32" | "s_andn2_b64" | "s_or_b32" | "s_or_b64"
        ) && let Some(&(_, dst_v)) = inst.writes.first()
        {
            return self.extract_mask_compare(dst_v, depth + 1);
        }
        if matches!(name, "s_add_i32" | "s_sub_i32" | "s_addk_i32")
            && let Some(&(_, dst_v)) = inst.writes.first()
        {
            // For signed add/sub, SCC is overflow (not zero); skip.
            let _ = dst_v;
            return None;
        }
        if matches!(name, "s_and_saveexec_b32" | "s_and_saveexec_b64")
            && let Some(src0) = src.first()
            && let Some(OperandVal::Val(v)) = self.operand_val(inst, src0)
        {
            return self.extract_mask_compare(v, depth + 1);
        }
        None
    }

    fn extract_mask_compare(
        &self,
        mask: Value,
        depth: usize,
    ) -> Option<(CmpOp, OperandVal, OperandVal, bool)> {
        if depth > 6 {
            return None;
        }
        let site = self.def(mask)?;
        let inst = self.inst(site);
        let name = inst.name();
        if let Some(op) = parse_vector_cmp(name)
            && let Fields::Valu(f) = &inst.fields
            && f.abs == 0
            && f.neg == 0
            && f.sdwa.is_none()
            && f.dpp.is_none()
        {
            let lhs = self.eval_operand(self.operand_val(inst, f.src.first()?)?);
            let rhs = self.eval_operand(self.operand_val(inst, f.src.get(1)?)?);
            return Some((op, lhs, rhs, false));
        }
        if let Fields::Sop { src, .. } = &inst.fields {
            match name {
                "s_mov_b32" | "s_mov_b64" | "s_and_saveexec_b32" | "s_and_saveexec_b64" => {
                    if let Some(OperandVal::Val(v)) = self.operand_val(inst, src.first()?) {
                        return self.extract_mask_compare(v, depth + 1);
                    }
                }
                "s_not_b32" | "s_not_b64" => {
                    if let Some(OperandVal::Val(v)) = self.operand_val(inst, src.first()?) {
                        let (op, l, r, inv) = self.extract_mask_compare(v, depth + 1)?;
                        return Some((op, l, r, !inv));
                    }
                }
                "s_and_b32" | "s_and_b64" | "s_or_b32" | "s_or_b64" => {
                    // Typically `exec & cmp_mask`: try both operands.
                    for s in src {
                        if let Some(OperandVal::Val(v)) = self.operand_val(inst, s)
                            && let Some(res) = self.extract_mask_compare(v, depth + 1)
                        {
                            return Some(res);
                        }
                    }
                }
                "s_andn2_b32" | "s_andn2_b64" => {
                    // `dst = src0 & ~src1`: try `src1` (negated) first, then `src0`.
                    if let Some(OperandVal::Val(v)) = src.get(1).and_then(|s| self.operand_val(inst, s))
                        && let Some((op, l, r, inv)) = self.extract_mask_compare(v, depth + 1)
                    {
                        return Some((op, l, r, !inv));
                    }
                    if let Some(OperandVal::Val(v)) = src.first().and_then(|s| self.operand_val(inst, s)) {
                        return self.extract_mask_compare(v, depth + 1);
                    }
                }
                _ => {}
            }
        }
        None
    }

    /// Decomposes an SSA value into `var * scale + add` (bounded depth <= 8),
    /// handling multiply-by-constant AND shift-add chains emitted by strength
    /// reduction.
    fn decompose_linear(&self, v: Value, depth: usize) -> LinearExpr {
        let identity = LinearExpr {
            var: v,
            scale: 1,
            add: 0,
        };
        if depth >= 8 {
            return identity;
        }
        let Some(site) = self.def(v) else {
            return identity;
        };
        let inst = self.inst(site);
        let name = inst.name();

        if let Some(OperandVal::Val(src)) = self.copy_source(inst, v) {
            return self.decompose_linear(src, depth + 1);
        }
        if let Some((base, delta)) = self.const_add_sub(inst, v) {
            let mut inner = self.decompose_linear(base, depth + 1);
            inner.add = inner.add.wrapping_add(delta);
            return inner;
        }

        // Shift-left by constant.
        if let Some((base, shift)) = self.const_shl(inst)
            && shift < 31
        {
            let mut inner = self.decompose_linear(base, depth + 1);
            let factor = 1u32 << shift;
            inner.scale = inner.scale.wrapping_mul(factor);
            inner.add = inner.add.wrapping_mul(factor);
            return inner;
        }

        // Multiply by constant.
        if let Some((base, factor)) = self.const_mul(inst) {
            let mut inner = self.decompose_linear(base, depth + 1);
            inner.scale = inner.scale.wrapping_mul(factor);
            inner.add = inner.add.wrapping_mul(factor);
            return inner;
        }

        // Fused multiply-add or shift-left-add.
        if let Some((base, factor, add_c)) = self.const_mad_or_lshl_add(inst) {
            let mut inner = self.decompose_linear(base, depth + 1);
            inner.scale = inner.scale.wrapping_mul(factor);
            inner.add = inner.add.wrapping_mul(factor).wrapping_add(add_c);
            return inner;
        }

        // `s_lshl{1..4}_add_u32 dst, a, b` -> `(a << n) + b`.
        if let Some(shift) = match name {
            "s_lshl1_add_u32" => Some(1u32),
            "s_lshl2_add_u32" => Some(2u32),
            "s_lshl3_add_u32" => Some(3u32),
            "s_lshl4_add_u32" => Some(4u32),
            _ => None,
        } && let Some((a, b)) = self.binary_operands(inst)
        {
            let a = self.eval_operand(a);
            let b = self.eval_operand(b);
            let factor = 1u32 << shift;
            match (a, b) {
                (OperandVal::Val(av), OperandVal::Const(c)) => {
                    let mut inner = self.decompose_linear(av, depth + 1);
                    inner.scale = inner.scale.wrapping_mul(factor);
                    inner.add = inner.add.wrapping_mul(factor).wrapping_add(c);
                    return inner;
                }
                (OperandVal::Val(av), OperandVal::Val(bv)) => {
                    let la = self.decompose_linear(av, depth + 1);
                    let lb = self.decompose_linear(bv, depth + 1);
                    if la.var == lb.var {
                        return LinearExpr {
                            var: la.var,
                            scale: la.scale.wrapping_mul(factor).wrapping_add(lb.scale),
                            add: la.add.wrapping_mul(factor).wrapping_add(lb.add),
                        };
                    }
                }
                _ => {}
            }
        }

        // `v_lshl_add_u32 dst, a, shift, b` when `a` and `b` share a base var.
        if name == "v_lshl_add_u32"
            && let Fields::Valu(f) = &inst.fields
            && f.abs == 0
            && f.neg == 0
            && f.sdwa.is_none()
            && f.dpp.is_none()
            && let (Some(a), Some(s), Some(b)) = (
                f.src.first().and_then(|o| self.operand_val(inst, o)),
                f.src.get(1).and_then(|o| self.operand_val(inst, o)),
                f.src.get(2).and_then(|o| self.operand_val(inst, o)),
            )
            && let (OperandVal::Val(av), OperandVal::Const(sh), OperandVal::Val(bv)) =
                (self.eval_operand(a), self.eval_operand(s), self.eval_operand(b))
            && sh < 31
        {
            let factor = 1u32 << sh;
            let la = self.decompose_linear(av, depth + 1);
            let lb = self.decompose_linear(bv, depth + 1);
            if la.var == lb.var {
                return LinearExpr {
                    var: la.var,
                    scale: la.scale.wrapping_mul(factor).wrapping_add(lb.scale),
                    add: la.add.wrapping_mul(factor).wrapping_add(lb.add),
                };
            }
        }

        // Combine `a + b` or `a - b` when both decompose to the same `var`
        // (LLVM shift-add multiply-by-constant chains, e.g. `i * 19 = (i << 4) + (i << 1) + i`).
        if let Some((a, b)) = self.add_operands_inst(inst)
            && let (OperandVal::Val(av), OperandVal::Val(bv)) =
                (self.eval_operand(a), self.eval_operand(b))
        {
            let la = self.decompose_linear(av, depth + 1);
            let lb = self.decompose_linear(bv, depth + 1);
            if la.var == lb.var {
                return LinearExpr {
                    var: la.var,
                    scale: la.scale.wrapping_add(lb.scale),
                    add: la.add.wrapping_add(lb.add),
                };
            }
        }
        if let Some((a, b)) = self.sub_operands_inst(inst)
            && let (OperandVal::Val(av), OperandVal::Val(bv)) =
                (self.eval_operand(a), self.eval_operand(b))
        {
            let la = self.decompose_linear(av, depth + 1);
            let lb = self.decompose_linear(bv, depth + 1);
            if la.var == lb.var {
                return LinearExpr {
                    var: la.var,
                    scale: la.scale.wrapping_sub(lb.scale),
                    add: la.add.wrapping_sub(lb.add),
                };
            }
        }

        identity
    }

    /// Traces a scalar expression back to a compile-time constant, a constant-
    /// offset buffer word, or a root constant.
    fn trace_scalar(&self, op: OperandVal, depth: usize) -> Option<ScalarSource> {
        if depth > MAX_DEPTH {
            return None;
        }
        let op = self.eval_operand(op);
        let v = match op {
            OperandVal::Const(c) => return Some(ScalarSource::Const(c)),
            OperandVal::Val(v) => v,
        };
        if let Some(&(slot, dword)) = self.root_consts.get(&v) {
            return Some(ScalarSource::RootConstant { slot, dword });
        }
        let site = self.def(v)?;
        let inst = self.inst(site);
        let name = inst.name();

        if let Some(src) = self.copy_source(inst, v) {
            return self.trace_scalar(src, depth + 1);
        }
        if matches!(
            name,
            "s_min_u32" | "v_min_u32" | "s_min_i32" | "v_min_i32"
        ) && let Some((a, b)) = self.binary_operands(inst)
        {
            match (self.eval_operand(a), self.eval_operand(b)) {
                (OperandVal::Val(x), OperandVal::Const(_))
                | (OperandVal::Const(_), OperandVal::Val(x)) => {
                    return self.trace_scalar(OperandVal::Val(x), depth + 1);
                }
                _ => {}
            }
        }

        match &inst.fields {
            Fields::Smem {
                data: Some(Operand::Scalar { code, dwords }),
                offset,
                soffset,
                ..
            } if name.starts_with("s_buffer_load_dword") => {
                let k = (0..*dwords).find(|&k| inst.write(Reg::S(code + k)) == Some(v))?;
                let src = self.desc_map.get(&site).copied()?;
                let bindings = self.resolve_buffer_bindings(src);
                let &(binding, built_off) = bindings.first()?;
                let OperandVal::Const(so_c) = self.eval_operand(self.operand_val(inst, soffset)?)
                else {
                    return None;
                };
                let byte_offset = (*offset as u32)
                    .wrapping_add(built_off)
                    .wrapping_add(so_c)
                    .wrapping_add(4 * u32::from(k));
                Some(ScalarSource::BufferWord {
                    binding,
                    byte_offset,
                })
            }
            Fields::Buffer {
                data: Some(Operand::Vector { reg, dwords }),
                addr: Operand::Vector { reg: areg, .. },
                soffset,
                offset,
                offen,
                idxen,
                ..
            } if name.starts_with("buffer_load_dword") && !*idxen => {
                let k = (0..*dwords).find(|&k| inst.write(Reg::V(reg + k)) == Some(v))?;
                let src = self.desc_map.get(&site).copied()?;
                let bindings = self.resolve_buffer_bindings(src);
                let &(binding, built_off) = bindings.first()?;
                let off_c = if *offen {
                    match self.eval_operand(OperandVal::Val(inst.read(Reg::V(*areg))?)) {
                        OperandVal::Const(c) => c,
                        OperandVal::Val(_) => return None,
                    }
                } else {
                    0
                };
                let OperandVal::Const(so_c) = self.eval_operand(self.operand_val(inst, soffset)?)
                else {
                    return None;
                };
                let byte_offset = u32::from(*offset)
                    .wrapping_add(built_off)
                    .wrapping_add(off_c)
                    .wrapping_add(so_c)
                    .wrapping_add(4 * u32::from(k));
                Some(ScalarSource::BufferWord {
                    binding,
                    byte_offset,
                })
            }
            _ => None,
        }
    }

    fn resolve_buffer_bindings(&self, src: &Source) -> Vec<(BufferBinding, u32)> {
        let mut out = Vec::new();
        for leaf in src.leaves() {
            match leaf {
                Source::Loaded { load, first } | Source::Fetched { load, first } => {
                    let Some(a) = self.scalar_map.get(load) else {
                        continue;
                    };
                    if a.dynamic.is_some() {
                        continue;
                    }
                    let Some(rp) = self.native.layout.get(a.param) else {
                        continue;
                    };
                    if rp.root == RootType::Table {
                        let binding = BufferBinding::Table {
                            slot: rp.slot,
                            table_offset_dwords: (a.offset / 4) + u32::from(*first),
                        };
                        if !out.contains(&(binding, 0)) {
                            out.push((binding, 0));
                        }
                    }
                }
                Source::Built { address, .. } => {
                    if address.dynamic.is_some() {
                        continue;
                    }
                    let Some(rp) = self.native.layout.get(address.param) else {
                        continue;
                    };
                    if rp.root == RootType::Descriptor {
                        let binding = BufferBinding::RootDescriptor { slot: rp.slot };
                        if !out.contains(&(binding, address.offset)) {
                            out.push((binding, address.offset));
                        }
                    }
                }
                _ => {}
            }
        }
        out
    }

    fn is_same_or_copy(&self, mut v: Value, target: Value) -> bool {
        for _ in 0..4 {
            if v == target {
                return true;
            }
            let Some(site) = self.def(v) else {
                return false;
            };
            let inst = self.inst(site);
            match self.copy_source(inst, v) {
                Some(OperandVal::Val(src)) => v = src,
                _ => return false,
            }
        }
        v == target
    }

    fn operand_val(&self, inst: &Inst, op: &Operand) -> Option<OperandVal> {
        match *op {
            Operand::Scalar { code: NULL, .. } => Some(OperandVal::Const(0)),
            Operand::Scalar { code, .. } => inst.read(Reg::S(code)).map(OperandVal::Val),
            Operand::Vector { reg, .. } => inst.read(Reg::V(reg)).map(OperandVal::Val),
            Operand::Literal(v) => Some(OperandVal::Const(v)),
            Operand::Const(c) => inline_const(c).map(OperandVal::Const),
            Operand::Attr { .. } => None,
        }
    }

    fn eval_operand(&self, op: OperandVal) -> OperandVal {
        let OperandVal::Val(v) = op else {
            return op;
        };
        match self.resolve_const(v, 0) {
            Some(c) => OperandVal::Const(c),
            None => op,
        }
    }

    fn resolve_const(&self, v: Value, depth: usize) -> Option<u32> {
        if depth > 6 {
            return None;
        }
        let site = self.def(v)?;
        let inst = self.inst(site);
        let name = inst.name();
        match &inst.fields {
            Fields::Sop { src, imm, .. } => match name {
                "s_mov_b32" => match self.operand_val(inst, src.first()?)? {
                    OperandVal::Const(c) => Some(c),
                    OperandVal::Val(x) => self.resolve_const(x, depth + 1),
                },
                "s_movk_i32" => Some(i32::from((*imm)? as i16) as u32),
                "s_add_u32" | "s_add_i32" => {
                    let a = self.resolve_operand_const(inst, src.first()?, depth + 1)?;
                    let b = self.resolve_operand_const(inst, src.get(1)?, depth + 1)?;
                    Some(a.wrapping_add(b))
                }
                "s_sub_u32" | "s_sub_i32" => {
                    let a = self.resolve_operand_const(inst, src.first()?, depth + 1)?;
                    let b = self.resolve_operand_const(inst, src.get(1)?, depth + 1)?;
                    Some(a.wrapping_sub(b))
                }
                "s_lshl_b32" => {
                    let a = self.resolve_operand_const(inst, src.first()?, depth + 1)?;
                    let b = self.resolve_operand_const(inst, src.get(1)?, depth + 1)?;
                    Some(a.wrapping_shl(b & 31))
                }
                "s_mul_i32" => {
                    let a = self.resolve_operand_const(inst, src.first()?, depth + 1)?;
                    let b = self.resolve_operand_const(inst, src.get(1)?, depth + 1)?;
                    Some(a.wrapping_mul(b))
                }
                _ => None,
            },
            Fields::Valu(f) if f.abs == 0 && f.neg == 0 && f.sdwa.is_none() && f.dpp.is_none() => {
                match name {
                    "v_mov_b32" | "v_readfirstlane_b32" | "v_readlane_b32" => {
                        self.resolve_operand_const(inst, f.src.first()?, depth + 1)
                    }
                    _ => None,
                }
            }
            _ => None,
        }
    }

    fn resolve_operand_const(&self, inst: &Inst, op: &Operand, depth: usize) -> Option<u32> {
        match self.operand_val(inst, op)? {
            OperandVal::Const(c) => Some(c),
            OperandVal::Val(v) => self.resolve_const(v, depth),
        }
    }

    fn copy_source(&self, inst: &Inst, v: Value) -> Option<OperandVal> {
        let name = inst.name();
        match &inst.fields {
            Fields::Sop { src, .. } if name == "s_mov_b32" => {
                self.operand_val(inst, src.first()?)
            }
            Fields::Sop {
                dst: Some(Operand::Scalar { code, dwords: 2 }),
                src,
                ..
            } if name == "s_mov_b64" && inst.write(Reg::S(*code)) == Some(v) => {
                self.operand_val(inst, src.first()?)
            }
            Fields::Valu(f)
                if matches!(name, "v_mov_b32" | "v_readfirstlane_b32" | "v_readlane_b32")
                    && f.abs == 0
                    && f.neg == 0
                    && f.sdwa.is_none()
                    && f.dpp.is_none() =>
            {
                self.operand_val(inst, f.src.first()?)
            }
            _ => None,
        }
    }

    fn binary_operands(&self, inst: &Inst) -> Option<(OperandVal, OperandVal)> {
        match &inst.fields {
            Fields::Sop { src, .. } => Some((
                self.operand_val(inst, src.first()?)?,
                self.operand_val(inst, src.get(1)?)?,
            )),
            Fields::Valu(f) if f.abs == 0 && f.neg == 0 && f.sdwa.is_none() && f.dpp.is_none() => {
                Some((
                    self.operand_val(inst, f.src.first()?)?,
                    self.operand_val(inst, f.src.get(1)?)?,
                ))
            }
            _ => None,
        }
    }

    fn select_operands(&self, inst: &Inst, v: Value) -> Option<(OperandVal, OperandVal)> {
        let name = inst.name();
        match &inst.fields {
            Fields::Sop {
                dst: Some(Operand::Scalar { code, .. }),
                src,
                ..
            } if name == "s_cselect_b32" => Some((
                self.eval_operand(self.operand_val(inst, src.first()?)?),
                self.eval_operand(self.operand_val(inst, src.get(1)?)?),
            )),
            Fields::Sop {
                dst: Some(Operand::Scalar { code, .. }),
                src,
                ..
            } if name == "s_cmov_b32" && inst.write(Reg::S(*code)) == Some(v) => {
                let a = self.eval_operand(self.operand_val(inst, src.first()?)?);
                let old = OperandVal::Val(inst.read(Reg::S(*code))?);
                Some((a, self.eval_operand(old)))
            }
            Fields::Valu(f)
                if name == "v_cndmask_b32"
                    && f.abs == 0
                    && f.neg == 0
                    && f.sdwa.is_none()
                    && f.dpp.is_none() =>
            {
                Some((
                    self.eval_operand(self.operand_val(inst, f.src.first()?)?),
                    self.eval_operand(self.operand_val(inst, f.src.get(1)?)?),
                ))
            }
            _ => None,
        }
    }

    fn add_operands(&self, v: Value) -> Option<(OperandVal, OperandVal)> {
        let site = self.def(v)?;
        self.add_operands_inst(self.inst(site))
    }

    fn add_operands_inst(&self, inst: &Inst) -> Option<(OperandVal, OperandVal)> {
        match inst.name() {
            "s_add_u32" | "s_add_i32" | "v_add_nc_u32" | "v_add_co_u32" => {
                self.binary_operands(inst)
            }
            "s_addk_i32" => {
                let Fields::Sop {
                    dst: Some(Operand::Scalar { code, .. }),
                    imm: Some(k),
                    ..
                } = inst.fields
                else {
                    return None;
                };
                let old = inst.read(Reg::S(code))?;
                Some((OperandVal::Val(old), OperandVal::Const(i32::from(k as i16) as u32)))
            }
            _ => None,
        }
    }

    fn sub_operands(&self, v: Value) -> Option<(OperandVal, OperandVal)> {
        let site = self.def(v)?;
        self.sub_operands_inst(self.inst(site))
    }

    fn sub_operands_inst(&self, inst: &Inst) -> Option<(OperandVal, OperandVal)> {
        match inst.name() {
            "s_sub_u32" | "s_sub_i32" | "v_sub_nc_u32" | "v_sub_co_u32" => {
                self.binary_operands(inst)
            }
            "v_subrev_nc_u32" | "v_subrev_co_u32" => {
                let (a, b) = self.binary_operands(inst)?;
                Some((b, a))
            }
            _ => None,
        }
    }

    fn const_add_sub_val(&self, v: Value) -> Option<(Value, u32)> {
        let site = self.def(v)?;
        self.const_add_sub(self.inst(site), v)
    }

    /// If `inst` computes `base + delta` or `base - delta` with constant `delta`,
    /// returns `(base, signed_delta_as_u32)`.
    fn const_add_sub(&self, inst: &Inst, _v: Value) -> Option<(Value, u32)> {
        if let Some((a, b)) = self.add_operands_inst(inst) {
            return match (self.eval_operand(a), self.eval_operand(b)) {
                (OperandVal::Val(x), OperandVal::Const(c))
                | (OperandVal::Const(c), OperandVal::Val(x)) => Some((x, c)),
                _ => None,
            };
        }
        if let Some((a, b)) = self.sub_operands_inst(inst) {
            return match (self.eval_operand(a), self.eval_operand(b)) {
                (OperandVal::Val(x), OperandVal::Const(c)) => Some((x, 0u32.wrapping_sub(c))),
                _ => None,
            };
        }
        if inst.name() == "v_add3_u32"
            && let Fields::Valu(f) = &inst.fields
            && f.abs == 0
            && f.neg == 0
            && f.sdwa.is_none()
            && f.dpp.is_none()
        {
            let ops = [
                self.eval_operand(self.operand_val(inst, f.src.first()?)?),
                self.eval_operand(self.operand_val(inst, f.src.get(1)?)?),
                self.eval_operand(self.operand_val(inst, f.src.get(2)?)?),
            ];
            let mut var = None;
            let mut sum = 0u32;
            for o in ops {
                match o {
                    OperandVal::Const(c) => sum = sum.wrapping_add(c),
                    OperandVal::Val(x) => {
                        if var.is_some() {
                            return None;
                        }
                        var = Some(x);
                    }
                }
            }
            return var.map(|x| (x, sum));
        }
        None
    }

    fn const_shl(&self, inst: &Inst) -> Option<(Value, u32)> {
        match inst.name() {
            "s_lshl_b32" | "s_lshl_b64" | "v_lshl_b32" => {
                let (a, b) = self.binary_operands(inst)?;
                match (self.eval_operand(a), self.eval_operand(b)) {
                    (OperandVal::Val(x), OperandVal::Const(s)) => Some((x, s)),
                    _ => None,
                }
            }
            "v_lshlrev_b32" | "v_lshlrev_b64" => {
                let (a, b) = self.binary_operands(inst)?;
                match (self.eval_operand(a), self.eval_operand(b)) {
                    (OperandVal::Const(s), OperandVal::Val(x)) => Some((x, s)),
                    _ => None,
                }
            }
            _ => None,
        }
    }

    fn const_mul(&self, inst: &Inst) -> Option<(Value, u32)> {
        match inst.name() {
            "s_mul_i32" | "v_mul_lo_u32" | "v_mul_lo_i32" | "v_mul_u32_u24" | "v_mul_i32_i24" => {
                let (a, b) = self.binary_operands(inst)?;
                match (self.eval_operand(a), self.eval_operand(b)) {
                    (OperandVal::Val(x), OperandVal::Const(c))
                    | (OperandVal::Const(c), OperandVal::Val(x)) => Some((x, c)),
                    _ => None,
                }
            }
            "s_mulk_i32" => {
                let Fields::Sop {
                    dst: Some(Operand::Scalar { code, .. }),
                    imm: Some(k),
                    ..
                } = inst.fields
                else {
                    return None;
                };
                let old = inst.read(Reg::S(code))?;
                Some((old, i32::from(k as i16) as u32))
            }
            _ => None,
        }
    }

    fn const_mad_or_lshl_add(&self, inst: &Inst) -> Option<(Value, u32, u32)> {
        let Fields::Valu(f) = &inst.fields else {
            return None;
        };
        if f.abs != 0 || f.neg != 0 || f.sdwa.is_some() || f.dpp.is_some() {
            return None;
        }
        let a = self.eval_operand(self.operand_val(inst, f.src.first()?)?);
        let b = self.eval_operand(self.operand_val(inst, f.src.get(1)?)?);
        let c = self.eval_operand(self.operand_val(inst, f.src.get(2)?)?);
        match inst.name() {
            "v_mad_u32_u24" | "v_mad_i32_i24" | "v_mad_u64_u32" | "v_mad_i64_i32" => {
                match (a, b, c) {
                    (OperandVal::Val(x), OperandVal::Const(m), OperandVal::Const(add))
                    | (OperandVal::Const(m), OperandVal::Val(x), OperandVal::Const(add)) => {
                        Some((x, m, add))
                    }
                    _ => None,
                }
            }
            "v_lshl_add_u32" => match (a, b, c) {
                (OperandVal::Val(x), OperandVal::Const(sh), OperandVal::Const(add)) if sh < 31 => {
                    Some((x, 1u32 << sh, add))
                }
                _ => None,
            },
            _ => None,
        }
    }
}

trait SopDstRead {
    fn fields_sop_dst_read(&self, inst: &Inst) -> Option<Value>;
}

impl SopDstRead for Inst {
    fn fields_sop_dst_read(&self, inst: &Inst) -> Option<Value> {
        let Fields::Sop {
            dst: Some(Operand::Scalar { code, .. }),
            ..
        } = self.fields
        else {
            return None;
        };
        inst.read(Reg::S(code))
    }
}

fn is_texture_read_op(inst: &Inst) -> bool {
    if !matches!(inst.fields, Fields::Image { .. }) {
        return false;
    }
    let name = inst.name();
    !name.starts_with("image_store")
        && !name.starts_with("image_atomic_")
        && !name.starts_with("image_bvh")
}

fn inline_const(code: u8) -> Option<u32> {
    match code {
        128..=192 => Some(u32::from(code - 128)),
        193..=208 => Some((-(i32::from(code) - 192)) as u32),
        _ => None,
    }
}

fn parse_scalar_cmp(name: &str) -> Option<CmpOp> {
    let rest = name
        .strip_prefix("s_cmp_")
        .or_else(|| name.strip_prefix("s_cmpk_"))?;
    let op = rest
        .strip_suffix("_u32")
        .or_else(|| rest.strip_suffix("_i32"))?;
    match op {
        "lt" => Some(CmpOp::Lt),
        "le" => Some(CmpOp::Le),
        "gt" => Some(CmpOp::Gt),
        "ge" => Some(CmpOp::Ge),
        "eq" => Some(CmpOp::Eq),
        "lg" => Some(CmpOp::Ne),
        _ => None,
    }
}

fn parse_vector_cmp(name: &str) -> Option<CmpOp> {
    let rest = name
        .strip_prefix("v_cmp_")
        .or_else(|| name.strip_prefix("v_cmpx_"))?;
    let op = rest
        .strip_suffix("_u32")
        .or_else(|| rest.strip_suffix("_i32"))?;
    match op {
        "lt" => Some(CmpOp::Lt),
        "le" => Some(CmpOp::Le),
        "gt" => Some(CmpOp::Gt),
        "ge" => Some(CmpOp::Ge),
        "eq" => Some(CmpOp::Eq),
        "ne" | "lg" => Some(CmpOp::Ne),
        _ => None,
    }
}

/// Computes natural loops in `blocks` using Cooper-Harvey-Kennedy dominators.
fn find_natural_loops(blocks: &[Block]) -> HashMap<usize, NaturalLoop> {
    let n = blocks.len();
    if n == 0 {
        return HashMap::new();
    }
    let rpo = crate::ir::reverse_postorder(blocks);
    let mut rpo_idx = vec![usize::MAX; n];
    for (idx, &b) in rpo.iter().enumerate() {
        rpo_idx[b] = idx;
    }

    // Immediate dominator in RPO indices (usize::MAX = undefined).
    let mut idom = vec![usize::MAX; n];
    idom[0] = 0;

    let intersect = |mut b1: usize, mut b2: usize, idom: &[usize]| -> usize {
        while b1 != b2 {
            while rpo_idx[b1] > rpo_idx[b2] {
                b1 = idom[b1];
            }
            while rpo_idx[b2] > rpo_idx[b1] {
                b2 = idom[b2];
            }
        }
        b1
    };

    let mut changed = true;
    while changed {
        changed = false;
        for &b in rpo.iter().skip(1) {
            let mut new_idom = usize::MAX;
            for &p in &blocks[b].preds {
                if rpo_idx[p] == usize::MAX || idom[p] == usize::MAX {
                    continue;
                }
                new_idom = if new_idom == usize::MAX {
                    p
                } else {
                    intersect(p, new_idom, &idom)
                };
            }
            if new_idom != usize::MAX && idom[b] != new_idom {
                idom[b] = new_idom;
                changed = true;
            }
        }
    }

    let dominates = |a: usize, mut b: usize| -> bool {
        if rpo_idx[a] == usize::MAX || rpo_idx[b] == usize::MAX {
            return false;
        }
        while b != a {
            if b == 0 || idom[b] == usize::MAX || idom[b] == b {
                return false;
            }
            b = idom[b];
        }
        true
    };

    let mut back_edges: HashMap<usize, Vec<usize>> = HashMap::new();
    for &b in &rpo {
        for succ in blocks[b].exit.successors() {
            if dominates(succ, b) {
                back_edges.entry(succ).or_default().push(b);
            }
        }
    }

    let mut loops = HashMap::new();
    for (header, latches) in back_edges {
        let mut body = HashSet::new();
        body.insert(header);
        let mut work = Vec::new();
        for &latch in &latches {
            if body.insert(latch) {
                work.push(latch);
            }
        }
        while let Some(curr) = work.pop() {
            for &p in &blocks[curr].preds {
                if body.insert(p) {
                    work.push(p);
                }
            }
        }
        let mut entry_preds = Vec::new();
        let mut back_preds = Vec::new();
        for &p in &blocks[header].preds {
            if body.contains(&p) {
                back_preds.push(p);
            } else if rpo_idx[p] != usize::MAX {
                entry_preds.push(p);
            }
        }
        loops.insert(
            header,
            NaturalLoop {
                header,
                body,
                entry_preds,
                back_preds,
            },
        );
    }
    loops
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::native::{HwStage, NativeProgram, ResourceType, RootParam, RootType, StageIo, UsedResource};
    use crate::{ir, resource, structure};

    const S_ENDPGM: u32 = 0xBF81_0000;
    const NULL_CODE: u32 = 125;

    /// Inline constant 0..=64 operand code.
    const fn c(n: u32) -> u32 {
        128 + n
    }

    /// `s_mov_b32 sdst, ssrc0`
    fn s_mov_b32(sdst: u32, ssrc0: u32) -> u32 {
        0xBE80_0300 | (sdst << 16) | ssrc0
    }

    /// `s_add_u32 sdst, ssrc0, ssrc1`
    fn s_add_u32(sdst: u32, ssrc0: u32, ssrc1: u32) -> u32 {
        (0b10 << 30) | (sdst << 16) | (ssrc1 << 8) | ssrc0
    }

    /// `s_sub_u32 sdst, ssrc0, ssrc1`
    fn s_sub_u32(sdst: u32, ssrc0: u32, ssrc1: u32) -> u32 {
        (0b10 << 30) | (1 << 23) | (sdst << 16) | (ssrc1 << 8) | ssrc0
    }

    /// `s_cmp_lg_u32 ssrc0, ssrc1` (op 0x07)
    fn s_cmp_lg_u32(ssrc0: u32, ssrc1: u32) -> u32 {
        0xBF00_0000 | (0x07 << 16) | (ssrc1 << 8) | ssrc0
    }

    /// `s_lshl_b32 sdst, ssrc0, ssrc1`
    fn s_lshl_b32(sdst: u32, ssrc0: u32, ssrc1: u32) -> u32 {
        (0b10 << 30) | (0x1e << 23) | (sdst << 16) | (ssrc1 << 8) | ssrc0
    }

    /// `s_mul_i32 sdst, ssrc0, ssrc1` (with optional 32-bit literal if `ssrc1 == 0xFF`).
    fn s_mul_i32_lit(sdst: u32, ssrc0: u32, lit: u32) -> [u32; 2] {
        [(0b10 << 30) | (0x26 << 23) | (sdst << 16) | (0xFF << 8) | ssrc0, lit]
    }

    /// `s_cmp_lt_u32 ssrc0, ssrc1` (op 0x0a)
    fn s_cmp_lt_u32(ssrc0: u32, ssrc1: u32) -> u32 {
        0xBF00_0000 | (0x0a << 16) | (ssrc1 << 8) | ssrc0
    }

    /// `s_cmp_le_u32 ssrc0, ssrc1` (op 0x0b)
    fn s_cmp_le_u32(ssrc0: u32, ssrc1: u32) -> u32 {
        0xBF00_0000 | (0x0b << 16) | (ssrc1 << 8) | ssrc0
    }

    /// `s_cbranch_scc1 simm16` (op 0x05)
    fn s_cbranch_scc1(simm16: i16) -> u32 {
        0xBF80_0000 | (0x05 << 16) | (simm16 as u16 as u32)
    }

    /// `s_cbranch_scc0 simm16` (op 0x04)
    fn s_cbranch_scc0(simm16: i16) -> u32 {
        0xBF80_0000 | (0x04 << 16) | (simm16 as u16 as u32)
    }

    /// `s_load_dwordx4` (op 2) / `s_load_dwordx8` (op 3) from SGPR pair `sbase`
    /// with signed `offset` and `soffset`.
    fn s_load(op: u32, sdata: u32, sbase: u32, offset: u32, soffset: u32) -> [u32; 2] {
        [
            (0x3D << 26) | (op << 18) | (sdata << 6) | (sbase >> 1),
            (soffset << 25) | (offset & 0x1F_FFFF),
        ]
    }

    /// `s_buffer_load_dword` (op 8) / `x2` (9) / `x4` (10) from 4-SGPR descriptor `sbase`.
    fn s_buffer_load(op: u32, sdata: u32, sbase: u32, offset: u32, soffset: u32) -> [u32; 2] {
        [
            (0x3D << 26) | (op << 18) | (sdata << 6) | (sbase >> 1),
            (soffset << 25) | (offset & 0x1F_FFFF),
        ]
    }

    /// `v_mov_b32 vdst, src0` (VOP1 op 1)
    fn v_mov_b32(vdst: u32, src0: u32) -> u32 {
        (0x3F << 25) | (vdst << 17) | (1 << 9) | src0
    }

    /// `v_readfirstlane_b32 sdst, vsrc0` (VOP1 op 2)
    fn v_readfirstlane_b32(sdst: u32, vsrc0: u32) -> u32 {
        (0x3F << 25) | (sdst << 17) | (2 << 9) | (256 + vsrc0)
    }

    /// `buffer_load_dword vdata, vaddr, srsrc, soffset [idxen] [offen] offset:imm` (MUBUF op 0x0c)
    fn buffer_load_dword(
        vdata: u32,
        vaddr: u32,
        srsrc: u32,
        soffset: u32,
        offset: u16,
        idxen: bool,
        offen: bool,
    ) -> [u32; 2] {
        [
            (0x38 << 26)
                | (0x0c << 18)
                | (u32::from(idxen) << 13)
                | (u32::from(offen) << 12)
                | u32::from(offset),
            (soffset << 24) | ((srsrc >> 2) << 16) | (vdata << 8) | vaddr,
        ]
    }

    /// `image_load vdata, vaddr, srsrc dmask:0xf dim:SQ_RSRC_IMG_2D` (MIMG op 0)
    fn image_load(vdata: u32, vaddr: u32, srsrc: u32) -> [u32; 2] {
        // MIMG: [31:26]=0x3C, [24:18]=op(0), [11:8]=dmask(0xf), [6:3]=dim(1=2D)
        [
            (0x3C << 26) | (0xf << 8) | (1 << 3),
            ((srsrc >> 2) << 16) | (vdata << 8) | vaddr,
        ]
    }

    fn run_extract(
        words: &[u32],
        layout: Vec<RootParam>,
        used: Vec<UsedResource>,
    ) -> Vec<HeapIndexSource> {
        let code: Vec<u8> = words.iter().flat_map(|w| w.to_le_bytes()).collect();
        let native = NativeProgram {
            stage: HwStage::Ps,
            wave64: false,
            vgprs: 16,
            layout,
            used,
            io: StageIo::None,
            code,
        };
        let p = ir::build(&native.code, false, 16).unwrap();
        let st = structure::structure(&p).unwrap();
        let res = resource::track(&p, &native.layout, native.stage, Some(&st)).unwrap();
        extract(&native, &p, &res)
    }

    fn ps_layout() -> (Vec<RootParam>, Vec<UsedResource>) {
        // s0 = Table (slot 0), s1 = ResourceHeap (slot 1)
        let layout = vec![
            RootParam {
                root: RootType::Table,
                table_offset_dwords: 0,
                sgprs: 1,
                base_register: 0,
                user_data: 0,
                slot: 0,
            },
            RootParam {
                root: RootType::ResourceHeap,
                table_offset_dwords: 0,
                sgprs: 1,
                base_register: 0,
                user_data: 1,
                slot: 1,
            },
        ];
        let used = vec![
            UsedResource {
                kind: ResourceType::Cbv,
                root: RootType::Table,
                table_offset_dwords: 0,
                lower_bound: 0,
                stride: 0,
                dwords: 4,
                user_data: 0,
                slot: 0,
            },
            UsedResource {
                kind: ResourceType::Srv,
                root: RootType::Table,
                table_offset_dwords: 8,
                lower_bound: 0,
                stride: 64,
                dwords: 4,
                user_data: 0,
                slot: 0,
            },
        ];
        (layout, used)
    }

    #[test]
    fn extracts_direct_cbv_word_heap_index() {
        let (layout, used) = ps_layout();
        // s[2:3] = (s0, 0) table base; s[4:5] = (s1, 0) heap base
        // s_load_dwordx4 s[8:11], s[2:3], 0x0          ; load CBV descriptor at table slot 0
        // s_buffer_load_dword s12, s[8:11], 0x24       ; load texture index from CBV byte 0x24
        // s_lshl_b32 s13, s12, 5                       ; idx * 32
        // s_load_dwordx8 s[16:23], s[4:5], s13         ; load texture descriptor from ResourceHeap
        // image_load v[0:3], v[0:1], s[16:23]          ; sample/load texture
        let mut w = vec![
            s_mov_b32(2, 0),
            s_mov_b32(3, c(0)),
            s_mov_b32(4, 1),
            s_mov_b32(5, c(0)),
        ];
        w.extend(s_load(2, 8, 2, 0, NULL_CODE));
        w.extend(s_buffer_load(8, 12, 8, 0x24, NULL_CODE));
        w.push(s_lshl_b32(13, 12, c(5)));
        w.extend(s_load(3, 16, 4, 0, 13));
        w.extend(image_load(0, 0, 16));
        w.push(S_ENDPGM);

        let sources = run_extract(&w, layout, used);
        assert_eq!(
            sources,
            vec![HeapIndexSource::CbvWord {
                binding: BufferBinding::Table {
                    slot: 0,
                    table_offset_dwords: 0,
                },
                byte_offset: 0x24,
                index_offset: 0,
            }]
        );
    }

    #[test]
    fn extracts_structured_loop_field_with_explicit_stride_and_cbv_bounds() {
        let (layout, used) = ps_layout();
        // Table slot 0 (offset 0): CBV with start at +0x10, end at +0x14
        // Table slot 1 (offset 32 = 8 dwords): Structured SRV with arbitrary 88-byte records,
        // texture index at field offset 0x1c.
        let mut w = vec![
            // Prologue (block 0):
            s_mov_b32(2, 0),    // table lo
            s_mov_b32(3, c(0)), // table hi
            s_mov_b32(4, 1),    // heap lo
            s_mov_b32(5, c(0)), // heap hi
        ];
        w.extend(s_load(2, 8, 2, 0, NULL_CODE)); // s[8:11] = CBV desc
        w.extend(s_load(2, 12, 2, 32, NULL_CODE)); // s[12:15] = SRV desc (table_offset_dwords=8)
        w.extend(s_buffer_load(8, 20, 8, 0x10, NULL_CODE)); // s20 = start
        w.extend(s_buffer_load(8, 21, 8, 0x14, NULL_CODE)); // s21 = end
        // Guard: if !(s20 < s21) skip loop
        w.push(s_cmp_lt_u32(20, 21));
        let guard_idx = w.len();
        w.push(0); // placeholder s_cbranch_scc0 -> exit

        // Loop body (header):
        let loop_start = w.len();
        w.extend(s_mul_i32_lit(22, 20, 88)); // s22 = i * 88
        w.extend(s_buffer_load(8, 23, 12, 0x1c, 22)); // s23 = records[i].field(0x1c)
        w.push(s_lshl_b32(24, 23, c(5))); // s24 = tex_idx * 32
        w.extend(s_load(3, 32, 4, 0, 24)); // s[32:39] = ResourceHeap[tex_idx]
        w.extend(image_load(0, 0, 32));
        w.push(s_add_u32(20, 20, c(1))); // i += 1
        w.push(s_cmp_lt_u32(20, 21)); // i < end
        let back_idx = w.len();
        let back_delta = (loop_start as i32 - (back_idx as i32 + 1)) as i16;
        w.push(s_cbranch_scc1(back_delta));

        let exit_idx = w.len();
        w.push(S_ENDPGM);
        let guard_delta = (exit_idx as i32 - (guard_idx as i32 + 1)) as i16;
        w[guard_idx] = s_cbranch_scc0(guard_delta);

        let sources = run_extract(&w, layout, used);
        assert_eq!(
            sources,
            vec![HeapIndexSource::StructuredLoopField {
                buffer: BufferBinding::Table {
                    slot: 0,
                    table_offset_dwords: 8,
                },
                stride: Some(88),
                field_byte_offset: 0x1c,
                start: ScalarSource::BufferWord {
                    binding: BufferBinding::Table {
                        slot: 0,
                        table_offset_dwords: 0,
                    },
                    byte_offset: 0x10,
                },
                end: ScalarSource::BufferWord {
                    binding: BufferBinding::Table {
                        slot: 0,
                        table_offset_dwords: 0,
                    },
                    byte_offset: 0x14,
                },
                end_is_count: false,
                inclusive_extra: 0,
                index_offset: 0,
            }]
        );
    }

    #[test]
    fn extracts_idxen_structured_loop_with_inclusive_bound() {
        let (layout, used) = ps_layout();
        // Same setup, but uses hardware `buffer_load_dword` with `idxen=true`
        // and an inclusive `s_cmp_le_u32` loop bound.
        let mut w = vec![
            s_mov_b32(2, 0),
            s_mov_b32(3, c(0)),
            s_mov_b32(4, 1),
            s_mov_b32(5, c(0)),
        ];
        w.extend(s_load(2, 8, 2, 0, NULL_CODE)); // CBV
        w.extend(s_load(2, 12, 2, 32, NULL_CODE)); // SRV
        w.extend(s_buffer_load(8, 20, 8, 0x08, NULL_CODE)); // s20 = start
        w.extend(s_buffer_load(8, 21, 8, 0x0c, NULL_CODE)); // s21 = end
        w.push(s_cmp_le_u32(20, 21));
        let guard_idx = w.len();
        w.push(0);

        let loop_start = w.len();
        w.push(v_mov_b32(2, 20)); // v2 = i
        w.extend(buffer_load_dword(3, 2, 12, c(0), 0x10, true, false)); // v3 = records[i].field(0x10)
        w.push(v_readfirstlane_b32(23, 3)); // s23 = v3
        w.push(s_lshl_b32(24, 23, c(5))); // s24 = tex_idx * 32
        w.extend(s_load(3, 32, 4, 0, 24));
        w.extend(image_load(0, 0, 32));
        w.push(s_add_u32(20, 20, c(1)));
        w.push(s_cmp_le_u32(20, 21));
        let back_idx = w.len();
        let back_delta = (loop_start as i32 - (back_idx as i32 + 1)) as i16;
        w.push(s_cbranch_scc1(back_delta));

        let exit_idx = w.len();
        w.push(S_ENDPGM);
        let guard_delta = (exit_idx as i32 - (guard_idx as i32 + 1)) as i16;
        w[guard_idx] = s_cbranch_scc0(guard_delta);

        let sources = run_extract(&w, layout, used);
        assert_eq!(
            sources,
            vec![HeapIndexSource::StructuredLoopField {
                buffer: BufferBinding::Table {
                    slot: 0,
                    table_offset_dwords: 8,
                },
                stride: None,
                field_byte_offset: 0x10,
                start: ScalarSource::BufferWord {
                    binding: BufferBinding::Table {
                        slot: 0,
                        table_offset_dwords: 0,
                    },
                    byte_offset: 0x08,
                },
                end: ScalarSource::BufferWord {
                    binding: BufferBinding::Table {
                        slot: 0,
                        table_offset_dwords: 0,
                    },
                    byte_offset: 0x0c,
                },
                end_is_count: false,
                inclusive_extra: 1,
                index_offset: 0,
            }]
        );
    }

    #[test]
    fn extracts_strength_reduced_byte_iv_loop() {
        let (layout, used) = ps_layout();
        // Strength-reduced loop: `off = start * 72`, `off += 72`, compared against `end * 72`.
        let mut w = vec![
            s_mov_b32(2, 0),
            s_mov_b32(3, c(0)),
            s_mov_b32(4, 1),
            s_mov_b32(5, c(0)),
        ];
        w.extend(s_load(2, 8, 2, 0, NULL_CODE));
        w.extend(s_load(2, 12, 2, 32, NULL_CODE));
        w.extend(s_buffer_load(8, 20, 8, 0x00, NULL_CODE)); // start
        w.extend(s_buffer_load(8, 21, 8, 0x04, NULL_CODE)); // end
        w.extend(s_mul_i32_lit(22, 20, 72)); // off = start * 72
        w.extend(s_mul_i32_lit(23, 21, 72)); // end_off = end * 72
        w.push(s_cmp_lt_u32(22, 23));
        let guard_idx = w.len();
        w.push(0);

        let loop_start = w.len();
        w.extend(s_buffer_load(8, 24, 12, 0x08, 22)); // records[i].field(0x08)
        w.push(s_lshl_b32(25, 24, c(5)));
        w.extend(s_load(3, 32, 4, 0, 25));
        w.extend(image_load(0, 0, 32));
        // s22 += 72 (using literal 0xFF)
        w.push(s_add_u32(22, 22, 0xFF));
        w.push(72);
        w.push(s_cmp_lt_u32(22, 23));
        let back_idx = w.len();
        let back_delta = (loop_start as i32 - (back_idx as i32 + 1)) as i16;
        w.push(s_cbranch_scc1(back_delta));

        let exit_idx = w.len();
        w.push(S_ENDPGM);
        let guard_delta = (exit_idx as i32 - (guard_idx as i32 + 1)) as i16;
        w[guard_idx] = s_cbranch_scc0(guard_delta);

        let sources = run_extract(&w, layout, used);
        assert_eq!(
            sources,
            vec![HeapIndexSource::StructuredLoopField {
                buffer: BufferBinding::Table {
                    slot: 0,
                    table_offset_dwords: 8,
                },
                stride: Some(72),
                field_byte_offset: 0x08,
                start: ScalarSource::BufferWord {
                    binding: BufferBinding::Table {
                        slot: 0,
                        table_offset_dwords: 0,
                    },
                    byte_offset: 0x00,
                },
                end: ScalarSource::BufferWord {
                    binding: BufferBinding::Table {
                        slot: 0,
                        table_offset_dwords: 0,
                    },
                    byte_offset: 0x04,
                },
                end_is_count: false,
                inclusive_extra: 0,
                index_offset: 0,
            }]
        );
    }

    #[test]
    fn extracts_shift_add_stride_chain_and_direct_record_access() {
        let (layout, used) = ps_layout();
        // Stride 76 = ((i << 4) + (i << 1) + i) << 2 computed purely via shift-add
        // chains (no multiply instruction), plus a direct non-loop access at
        // `records[cb.end].field(0x0c)`.
        let mut w = vec![
            s_mov_b32(2, 0),
            s_mov_b32(3, c(0)),
            s_mov_b32(4, 1),
            s_mov_b32(5, c(0)),
        ];
        w.extend(s_load(2, 8, 2, 0, NULL_CODE));
        w.extend(s_load(2, 12, 2, 32, NULL_CODE));
        w.extend(s_buffer_load(8, 20, 8, 0x04, NULL_CODE)); // s20 = start
        w.extend(s_buffer_load(8, 21, 8, 0x08, NULL_CODE)); // s21 = end
        w.push(s_cmp_lt_u32(20, 21));
        let guard_idx = w.len();
        w.push(0);

        let loop_start = w.len();
        w.push(s_lshl_b32(22, 20, c(4))); // 16 * i
        w.push(s_lshl_b32(23, 20, c(1))); // 2 * i
        w.push(s_add_u32(22, 22, 23)); // 18 * i
        w.push(s_add_u32(22, 22, 20)); // 19 * i
        w.push(s_lshl_b32(22, 22, c(2))); // 76 * i
        w.extend(s_buffer_load(8, 24, 12, 0x0c, 22));
        w.push(s_lshl_b32(25, 24, c(5)));
        w.extend(s_load(3, 32, 4, 0, 25));
        w.extend(image_load(0, 0, 32));
        w.push(s_add_u32(20, 20, c(1)));
        w.push(s_cmp_lt_u32(20, 21));
        let back_idx = w.len();
        let back_delta = (loop_start as i32 - (back_idx as i32 + 1)) as i16;
        w.push(s_cbranch_scc1(back_delta));

        let exit_idx = w.len();
        // Direct single-record access at `records[end].field(0x0c)` after loop.
        w.extend(s_mul_i32_lit(26, 21, 76));
        w.extend(s_buffer_load(8, 27, 12, 0x0c, 26));
        w.push(s_lshl_b32(28, 27, c(5)));
        w.extend(s_load(3, 40, 4, 0, 28));
        w.extend(image_load(0, 0, 40));
        w.push(S_ENDPGM);
        let guard_delta = (exit_idx as i32 - (guard_idx as i32 + 1)) as i16;
        w[guard_idx] = s_cbranch_scc0(guard_delta);

        let cbv = BufferBinding::Table {
            slot: 0,
            table_offset_dwords: 0,
        };
        let srv = BufferBinding::Table {
            slot: 0,
            table_offset_dwords: 8,
        };
        let sources = run_extract(&w, layout, used);
        assert_eq!(sources.len(), 2);
        assert!(sources.contains(&HeapIndexSource::StructuredLoopField {
            buffer: srv,
            stride: Some(76),
            field_byte_offset: 0x0c,
            start: ScalarSource::BufferWord {
                binding: cbv,
                byte_offset: 0x04,
            },
            end: ScalarSource::BufferWord {
                binding: cbv,
                byte_offset: 0x08,
            },
            end_is_count: false,
            inclusive_extra: 0,
            index_offset: 0,
        }));
        assert!(sources.contains(&HeapIndexSource::StructuredLoopField {
            buffer: srv,
            stride: Some(76),
            field_byte_offset: 0x0c,
            start: ScalarSource::BufferWord {
                binding: cbv,
                byte_offset: 0x08,
            },
            end: ScalarSource::Const(1),
            end_is_count: true,
            inclusive_extra: 0,
            index_offset: 0,
        }));
    }

    #[test]
    fn extracts_countdown_loop_with_strength_reduced_byte_iv() {
        let (layout, used) = ps_layout();
        // Countdown loop: `rem = end - start; off = start * 96; do { ... off += 96; rem -= 1; } while (rem != 0);`
        let mut w = vec![
            s_mov_b32(2, 0),
            s_mov_b32(3, c(0)),
            s_mov_b32(4, 1),
            s_mov_b32(5, c(0)),
        ];
        w.extend(s_load(2, 8, 2, 0, NULL_CODE));
        w.extend(s_load(2, 12, 2, 32, NULL_CODE));
        w.extend(s_buffer_load(8, 20, 8, 0x18, NULL_CODE)); // start
        w.extend(s_buffer_load(8, 21, 8, 0x1c, NULL_CODE)); // end
        w.push(s_sub_u32(29, 21, 20)); // rem = end - start
        w.push(s_cmp_lg_u32(29, c(0)));
        let guard_idx = w.len();
        w.push(0);
        w.extend(s_mul_i32_lit(22, 20, 96)); // off = start * 96

        let loop_start = w.len();
        w.extend(s_buffer_load(8, 24, 12, 0x20, 22));
        w.push(s_lshl_b32(25, 24, c(5)));
        w.extend(s_load(3, 32, 4, 0, 25));
        w.extend(image_load(0, 0, 32));
        w.push(s_add_u32(22, 22, 0xFF));
        w.push(96);
        w.push(s_sub_u32(29, 29, c(1)));
        w.push(s_cmp_lg_u32(29, c(0)));
        let back_idx = w.len();
        let back_delta = (loop_start as i32 - (back_idx as i32 + 1)) as i16;
        w.push(s_cbranch_scc1(back_delta));

        let exit_idx = w.len();
        w.push(S_ENDPGM);
        let guard_delta = (exit_idx as i32 - (guard_idx as i32 + 1)) as i16;
        w[guard_idx] = s_cbranch_scc0(guard_delta);

        let sources = run_extract(&w, layout, used);
        assert_eq!(
            sources,
            vec![HeapIndexSource::StructuredLoopField {
                buffer: BufferBinding::Table {
                    slot: 0,
                    table_offset_dwords: 8,
                },
                stride: Some(96),
                field_byte_offset: 0x20,
                start: ScalarSource::BufferWord {
                    binding: BufferBinding::Table {
                        slot: 0,
                        table_offset_dwords: 0,
                    },
                    byte_offset: 0x18,
                },
                end: ScalarSource::BufferWord {
                    binding: BufferBinding::Table {
                        slot: 0,
                        table_offset_dwords: 0,
                    },
                    byte_offset: 0x1c,
                },
                end_is_count: false,
                inclusive_extra: 0,
                index_offset: 0,
            }]
        );
    }
}
