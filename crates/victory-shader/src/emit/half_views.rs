//! Typed buffer loads that move 16-bit halves without the d16 form: a
//! format load of whole dwords whose result the program reads only one
//! half of, as a half float, moves 16-bit integers, like a d16 access
//! ([`super::super::HostShader::half_views`]).

use std::collections::{BTreeSet, HashSet};

use super::super::bitcode::{Attr, Bin, Term, V, pred};
use super::{
    Fields, Inst, Lower, Operand, Reg, Role, RootType, Site, Source, TYPED_F32_PROPS,
    TYPED_I32_PROPS, TYPED_U32_PROPS, VIEW_INTEGER, VIEW_SIGNED, VIEW_STRIDE_BITS, VIEW_STRIDES,
    Value, Why,
};

impl<'a> Lower<'a> {
    /// Whether every use of `value`, a typed buffer load's result, reads
    /// one 16-bit half of it as a half float: a `v_fma_mix*` source with
    /// its `op_sel_hi` bit set, or `v_cvt_f32_f16`. A program that takes a
    /// format load apart like that reads a 16-bit integer view on the
    /// console (the load gives the zero-extended integer, whose bits are a
    /// half; a float view's value it would not take apart). A use is a
    /// read as an operand ([`crate::ir::operand_reads`]): a later write of
    /// the register reads the value only to keep it in the lanes EXEC
    /// leaves alone, and a phi carries it on, its own uses counting. False
    /// without a use.
    pub(super) fn read_only_as_halves(&self, value: Value) -> bool {
        let p = self.x.p;
        let mut used = false;
        let mut seen = HashSet::from([value]);
        let mut todo = vec![value];
        while let Some(value) = todo.pop() {
            for b in &p.blocks {
                for phi in b.phis.iter().filter(|phi| phi.args.contains(&value)) {
                    if seen.insert(phi.value) {
                        todo.push(phi.value);
                    }
                }
                for i in &b.insts {
                    if !i.reads.iter().any(|r| r.1 == value) {
                        continue;
                    }
                    let operands = crate::ir::operand_reads(p, self.x.native.vgprs, i);
                    let regs: Vec<Reg> = i
                        .reads
                        .iter()
                        .filter(|r| r.1 == value && operands.contains(&r.0))
                        .map(|r| r.0)
                        .collect();
                    if regs.is_empty() {
                        continue;
                    }
                    used = true;
                    let Fields::Valu(v) = &i.fields else {
                        return false;
                    };
                    match i.name() {
                        "v_cvt_f32_f16" => {}
                        "v_fma_mix_f32" | "v_fma_mixlo_f16" | "v_fma_mixhi_f16" => {
                            let whole = v.src.iter().enumerate().any(|(k, op)| {
                                matches!(*op, Operand::Vector { reg, .. } if regs.contains(&Reg::V(reg)))
                                    && v.opsel_hi >> k & 1 == 0
                            });
                            if whole {
                                return false;
                            }
                        }
                        _ => return false,
                    }
                }
            }
        }
        used
    }

    /// Whether `inst` is a format load of whole dwords (not d16) whose
    /// every channel is read only as halves ([`Self::read_only_as_halves`]).
    pub(super) fn format_load_read_as_halves(&self, inst: &Inst) -> bool {
        let Fields::Buffer {
            data: Some(Operand::Vector { reg, .. }),
            ..
        } = inst.fields
        else {
            return false;
        };
        let name = inst.name();
        let count = match name
            .strip_prefix("buffer_load_format_")
            .or(name.strip_prefix("tbuffer_load_format_"))
        {
            Some("x") => 1,
            Some("xy") => 2,
            Some("xyz") => 3,
            Some("xyzw") => 4,
            _ => return false,
        };
        (0..count).all(|k| {
            inst.write(Reg::V(reg + k))
                .is_some_and(|v| self.read_only_as_halves(v))
        })
    }

    /// The table places ([`Self::table_place`]) every access through which
    /// moves halves: a d16 access, or a format load read only as halves
    /// ([`Self::format_load_read_as_halves`]). A place the program reaches
    /// otherwise too is left out (a format load of numbers beside one of
    /// halves, a raw or scalar load, a descriptor chosen by control flow),
    /// and so is every place of a table the program also picks from at run
    /// time: what the program makes of such a view is not halves alone.
    pub(super) fn places_read_as_halves(&self) -> BTreeSet<(u8, u32)> {
        let mut halves = BTreeSet::new();
        let mut other = BTreeSet::new();
        let mut picked = HashSet::new();
        for d in &self.x.res.descriptors {
            if d.role == Role::Sampler {
                continue;
            }
            let inst = self
                .x
                .p
                .blocks
                .get(d.site.0)
                .and_then(|b| b.insts.get(d.site.1));
            let half = matches!(d.source, Source::Loaded { .. })
                && inst.is_some_and(|i| {
                    (i.name().starts_with("buffer_") && i.name().contains("_format_d16_"))
                        || self.format_load_read_as_halves(i)
                });
            for leaf in d.source.leaves() {
                let Source::Loaded { load, .. } = leaf else {
                    continue;
                };
                let Some(a) = self.scalar.get(load) else {
                    continue;
                };
                if a.dynamic.is_some() {
                    if let Some(rp) = self.x.native.layout.get(a.param)
                        && rp.root == RootType::Table
                    {
                        picked.insert(rp.slot);
                    }
                } else if let Some(place) = self.table_place(leaf) {
                    if half {
                        halves.insert(place);
                    } else {
                        other.insert(place);
                    }
                }
            }
        }
        halves.retain(|p| !other.contains(p) && !picked.contains(&p.0));
        halves
    }

    /// The table descriptor `src` is, as (root parameter, dword in the
    /// table), when the code loads it from a fixed place in a table: the
    /// load's own place, not the start of the range its used-resource
    /// entry describes (a range of several descriptors starts before most
    /// of them). `None` for a descriptor picked at run time, built in the
    /// code or chosen by control flow.
    pub(super) fn table_place(&self, src: &Source) -> Option<(u8, u32)> {
        let Source::Loaded { load, first } = src else {
            return None;
        };
        let a = self.scalar.get(load)?;
        let rp = self.x.native.layout.get(a.param)?;
        (rp.root == RootType::Table && a.dynamic.is_none())
            .then(|| (rp.slot, a.offset / 4 + u32::from(*first)))
    }

    /// The byte stride of the buffer view at heap index `index`, from the
    /// view-stride table (`t`[`VIEW_STRIDES`]).
    pub(super) fn stride_at(&mut self, index: V) -> V {
        let word = self.view_word(index);
        let bits = self.s.m.i32(VIEW_STRIDE_BITS);
        self.s.m.bin(self.cur, Bin::And, word, bits)
    }

    /// Dword `index` of the view-stride table: the stride and the format
    /// kind bits ([`VIEW_INTEGER`], [`VIEW_SIGNED`]).
    pub(super) fn view_word(&mut self, index: V) -> V {
        let c = self.cur;
        let four = self.s.m.i32(4);
        let at = self.s.m.bin(c, Bin::Mul, index, four);
        let table = self.s.root_srv_handle(VIEW_STRIDES);
        self.raw_dword(table, at)
    }

    /// Whether the typed buffer view a handle names has an integer format,
    /// and whether a signed one, from the view-stride table; `None` if the
    /// handle's heap index is not known.
    pub(super) fn integer_view(&mut self, h: V) -> Option<(V, V)> {
        let index = *self.s.view_index.get(&h)?;
        let word = self.view_word(index);
        let c = self.cur;
        let zero = self.s.m.i32(0);
        let [int, signed] = [VIEW_INTEGER, VIEW_SIGNED].map(|bit| {
            let bit = self.s.m.i32(bit);
            let x = self.s.m.bin(c, Bin::And, word, bit);
            self.s.m.cmp(c, pred::NE, x, zero)
        });
        Some((int, signed))
    }

    /// A 32-bit typed load through independently annotated float and integer
    /// handles. The view's format is dynamic, so the overloads must be in
    /// mutually exclusive branches: executing `bufferLoad.i32` on a float view
    /// (or the reverse) is not a bitcast. Each result channel is an `i32` phi;
    /// the float path bitcasts only after the float load has taken place.
    pub(super) fn typed_buffer_load(
        &mut self,
        float_h: V,
        integer_h: V,
        index: V,
        integer: V,
        count: u32,
    ) -> Vec<V> {
        let head = self.cur;
        let (integer_block, float_block, join) = (
            self.s.m.new_block(),
            self.s.m.new_block(),
            self.s.m.new_block(),
        );
        self.s
            .m
            .terminate(head, Term::CondBr(integer, integer_block, float_block));
        let iu = self.s.m.undef(self.s.t.i32);
        let mut integer_values = Vec::new();
        self.cur = integer_block;
        let r = self.s.op(
            integer_block,
            "bufferLoad.i32",
            Attr::ReadOnly,
            self.s.t.res_ret_i32,
            68,
            &[integer_h, index, iu],
        );
        for ch in 0..count {
            integer_values.push(self.s.m.extract(integer_block, r, ch));
        }
        self.s.m.terminate(integer_block, Term::Br(join));
        let mut float_values = Vec::new();
        self.cur = float_block;
        let r = self.s.op(
            float_block,
            "bufferLoad.f32",
            Attr::ReadOnly,
            self.s.t.res_ret,
            68,
            &[float_h, index, iu],
        );
        for ch in 0..count {
            let x = self.s.m.extract(float_block, r, ch);
            float_values.push(self.as_int(x));
        }
        self.s.m.terminate(float_block, Term::Br(join));
        self.cur = join;
        integer_values
            .into_iter()
            .zip(float_values)
            .map(|(integer, float)| {
                let phi = self.s.m.phi(join, self.s.t.i32);
                self.s
                    .m
                    .set_phi(phi, vec![(integer, integer_block), (float, float_block)]);
                phi
            })
            .collect()
    }

    /// Unpacks packed 16-bit channels for a `d16` typed buffer store.
    pub(super) fn d16_store_values(
        &mut self,
        vals: &[V],
        count: u32,
        halves: bool,
        typed_h: Option<V>,
    ) -> Vec<V> {
        let integer = if halves {
            None
        } else {
            typed_h.and_then(|h| self.integer_view(h))
        };
        let c = self.cur;
        let mut wide = Vec::new();
        for k in 0..count {
            let mut x = vals[(k / 2) as usize];
            if k % 2 == 1 {
                let k16 = self.s.m.i32(16);
                x = self.s.m.bin(c, Bin::LShr, x, k16);
            }
            if halves {
                let low = self.s.m.i32(0xFFFF);
                wide.push(self.s.m.bin(c, Bin::And, x, low));
                continue;
            }
            let f = self.f16_to_f32(c, x);
            let mut w = self.as_int(f);
            if let Some((int, signed)) = integer {
                let (k16, low) = (self.s.m.i32(16), self.s.m.i32(0xFFFF));
                let zero_ext = self.s.m.bin(c, Bin::And, x, low);
                let up = self.s.m.bin(c, Bin::Shl, x, k16);
                let sign_ext = self.s.m.bin(c, Bin::AShr, up, k16);
                let n = self.s.m.select(c, signed, sign_ext, zero_ext);
                w = self.s.m.select(c, int, n, w);
            }
            wide.push(w);
        }
        wide
    }

    /// Emits a 32-bit typed buffer store with dynamic float/integer dispatch.
    #[allow(clippy::too_many_arguments)]
    pub(super) fn typed_buffer_store_32(
        &mut self,
        site: Site,
        src: &Source,
        u: u32,
        typed_h: Option<V>,
        index: Option<V>,
        vals: &[V],
        guard: Option<V>,
    ) -> Result<(), Why> {
        let h = typed_h.ok_or("a typed buffer store")?;
        let element = index.unwrap_or(self.s.m.i32(0));
        let (byte, bytes) = if self.x.view_marks {
            let at = self
                .s
                .view_index
                .get(&h)
                .copied()
                .ok_or("a typed buffer store")?;
            let stride = self.stride_at(at);
            (self.s.m.bin(self.cur, Bin::Mul, element, stride), stride)
        } else {
            (element, self.s.m.i32(0))
        };
        let integer = self.integer_view(h).map(|(integer, _)| integer);
        let integer_h = if integer.is_some() {
            Some(self.handle(site, src, false, (10 | u, TYPED_I32_PROPS))?)
        } else {
            None
        };
        let iu = self.s.m.undef(self.s.t.i32);
        let outer = self.begin_guarded(guard);
        self.mark_view_write(h, byte, bytes, None);
        if let (Some(integer), Some(integer_h)) = (integer, integer_h) {
            let head = self.cur;
            let (integer_block, float_block, join) = (
                self.s.m.new_block(),
                self.s.m.new_block(),
                self.s.m.new_block(),
            );
            self.s
                .m
                .terminate(head, Term::CondBr(integer, integer_block, float_block));
            let mask = self.s.m.i8(0xF);
            self.cur = integer_block;
            let mut args = vec![integer_h, element, iu];
            args.extend_from_slice(vals);
            args.push(mask);
            self.s.op(
                integer_block,
                "bufferStore.i32",
                Attr::NoUnwind,
                self.s.t.void,
                69,
                &args,
            );
            self.s.m.terminate(integer_block, Term::Br(join));
            self.cur = float_block;
            let mut args = vec![h, element, iu];
            args.extend(vals.iter().map(|&x| self.as_float(x)));
            args.push(mask);
            self.s.op(
                float_block,
                "bufferStore.f32",
                Attr::NoUnwind,
                self.s.t.void,
                69,
                &args,
            );
            self.s.m.terminate(float_block, Term::Br(join));
            self.cur = join;
        } else {
            let mut args = vec![h, element, iu];
            args.extend(vals.iter().map(|&x| self.as_float(x)));
            args.push(self.s.m.i8(0xF));
            self.s.op(
                self.cur,
                "bufferStore.f32",
                Attr::NoUnwind,
                self.s.t.void,
                69,
                &args,
            );
        }
        self.end_guarded(outer, &[]);
        Ok(())
    }

    /// Emits a 32-bit typed buffer load (`halves` unsigned twin or dynamic float/integer).
    #[allow(clippy::too_many_arguments)]
    pub(super) fn typed_buffer_load_32(
        &mut self,
        site: Site,
        src: &Source,
        u: u32,
        index: Option<V>,
        count: u32,
        halves: bool,
    ) -> Result<Vec<V>, Why> {
        let iu = self.s.m.undef(self.s.t.i32);
        let index = index.unwrap_or(self.s.m.i32(0));
        if halves {
            let h = self.handle(site, src, false, (10 | u, TYPED_U32_PROPS))?;
            let i32t = self.s.t.i32;
            let ret = self.s.m.named_struct("dx.types.ResRet.i32", vec![i32t; 5]);
            let r = self.s.op(
                self.cur,
                "bufferLoad.i32",
                Attr::ReadOnly,
                ret,
                68,
                &[h, index, iu],
            );
            Ok((0..count)
                .map(|ch| self.s.m.extract(self.cur, r, ch))
                .collect())
        } else {
            let h = self.handle(site, src, false, (10 | u, TYPED_F32_PROPS))?;
            if let Some((integer, _)) = self.integer_view(h) {
                let integer_h = self.handle(site, src, false, (10 | u, TYPED_I32_PROPS))?;
                Ok(self.typed_buffer_load(h, integer_h, index, integer, count))
            } else {
                let r = self.s.op(
                    self.cur,
                    "bufferLoad.f32",
                    Attr::ReadOnly,
                    self.s.t.res_ret,
                    68,
                    &[h, index, iu],
                );
                Ok((0..count)
                    .map(|ch| {
                        let x = self.s.m.extract(self.cur, r, ch);
                        self.as_int(x)
                    })
                    .collect())
            }
        }
    }

    /// Emits a `d16` typed buffer load and returns `(result_struct, float_handle)`.
    pub(super) fn typed_buffer_load_d16(
        &mut self,
        site: Site,
        src: &Source,
        u: u32,
        index: Option<V>,
        halves: bool,
    ) -> Result<(V, Option<V>), Why> {
        let c = self.cur;
        let iu = self.s.m.undef(self.s.t.i32);
        let index = index.unwrap_or(self.s.m.i32(0));
        if halves {
            let h = self.handle(site, src, false, (10 | u, TYPED_U32_PROPS))?;
            let i32t = self.s.t.i32;
            let ret = self.s.m.named_struct("dx.types.ResRet.i32", vec![i32t; 5]);
            let r = self
                .s
                .op(c, "bufferLoad.i32", Attr::ReadOnly, ret, 68, &[h, index, iu]);
            Ok((r, None))
        } else {
            let h = self.handle(site, src, false, (10 | u, TYPED_F32_PROPS))?;
            let r = self.s.op(
                c,
                "bufferLoad.f32",
                Attr::ReadOnly,
                self.s.t.res_ret,
                68,
                &[h, index, iu],
            );
            Ok((r, Some(h)))
        }
    }

    /// Packs `d16` loaded channels two per VGPR and writes them to `reg..`.
    #[allow(clippy::too_many_arguments)]
    pub(super) fn write_d16_channels(
        &mut self,
        inst: &Inst,
        reg: u8,
        r: V,
        count: u32,
        halves: bool,
        typed_h: Option<V>,
        guard: Option<V>,
    ) {
        let integer = typed_h.and_then(|h| self.integer_view(h));
        let c = self.cur;
        let k16 = self.s.m.i32(16);
        for k in 0..count.div_ceil(2) {
            let mut x = None;
            for ch in (2 * k..(2 * k + 2).min(count)).rev() {
                let f = self.s.m.extract(c, r, ch);
                let mut h = if halves {
                    let low = self.s.m.i32(0xFFFF);
                    self.s.m.bin(c, Bin::And, f, low)
                } else {
                    self.f32_to_f16(c, f)
                };
                if let Some((int, _)) = integer {
                    let bits = self.as_int(f);
                    let low = self.s.m.i32(0xFFFF);
                    let low = self.s.m.bin(c, Bin::And, bits, low);
                    h = self.s.m.select(c, int, low, h);
                }
                x = Some(match x {
                    Some(hi) => {
                        let hi = self.s.m.bin(c, Bin::Shl, hi, k16);
                        self.s.m.bin(c, Bin::Or, h, hi)
                    }
                    None => h,
                });
            }
            self.write(inst, Reg::V(reg + k as u8), x.expect("a channel"), guard);
        }
    }
}
