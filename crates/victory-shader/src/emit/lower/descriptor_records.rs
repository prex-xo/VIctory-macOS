//! A buffer descriptor the program loads only to read its record count
//! (dword 2): no instruction of the program accesses the buffer, so its
//! kind comes from the used-resource entry of the table slot it is loaded
//! from.

use super::*;

impl Lower<'_> {
    /// The handle properties of the buffer descriptor that `site` loads
    /// whole from a table slot, when the program never accesses it: by its
    /// used-resource entry's stride, structured (the record count in
    /// elements) or raw (in bytes), as for a buffer read with scalar loads
    /// ([`Lower::descriptor_data`]). None for anything else: a load that
    /// does not start at a buffer descriptor of the entry, a constant
    /// buffer, or an odd stride.
    pub(super) fn unused_buffer(&self, site: Site) -> Option<(u32, u32)> {
        // Only the record count: a program that also reads the address
        // takes it from the view addresses ([`Lower::view_addresses`]).
        let inst = &self.x.p.blocks[site.0].insts[site.1];
        let read = |v: Value| {
            self.x.p.blocks.iter().any(|b| {
                b.phis.iter().any(|f| f.args.contains(&v))
                    || b.insts.iter().any(|i| i.reads.iter().any(|r| r.1 == v))
            })
        };
        let Fields::Smem {
            data: Some(Operand::Scalar { code, .. }),
            ..
        } = inst.fields
        else {
            return None;
        };
        let others = inst
            .writes
            .iter()
            .filter(|w| w.0 != Reg::S(code + 2))
            .any(|w| read(w.1));
        if others {
            return None;
        }
        let a = self.scalar.get(&site)?;
        let used = self.used_kind(&Source::Loaded {
            load: site,
            first: 0,
        })?;
        let dword = a.offset / 4;
        if !a.offset.is_multiple_of(4)
            || used.dwords != 4
            || !(dword - used.table_offset_dwords).is_multiple_of(4)
        {
            return None;
        }
        let u = match used.kind {
            ResourceType::Srv => 0,
            ResourceType::Uav => 0x1000,
            _ => return None,
        };
        match used.stride {
            0 | 1 => Some((11 | u, 0)),
            s if s.is_multiple_of(4) => Some((12 | u, s)),
            _ => None,
        }
    }

    /// Dword 2 of the buffer descriptor `site` loads ([`Lower::unused_buffer`],
    /// `props`): the record count the host view gives; the other dwords
    /// stop the program when read, as in [`Lower::descriptor_data`].
    pub(super) fn buffer_records(
        &mut self,
        inst: &Inst,
        site: Site,
        code: u8,
        count: u32,
        props: (u32, u32),
    ) -> Result<(), Why> {
        let src = Source::Loaded {
            load: site,
            first: 0,
        };
        let h = self.handle(site, &src, false, props)?;
        let c = self.cur;
        let i32t = self.s.t.i32;
        let dims = self.s.m.named_struct("dx.types.Dimensions", vec![i32t; 4]);
        let mip = self.s.m.undef(i32t);
        let d = self
            .s
            .op(c, "getDimensions", Attr::ReadOnly, dims, 72, &[h, mip]);
        let records = self.s.m.extract(c, d, 0);
        for k in 0..count {
            if k == 2 {
                self.write(inst, Reg::S(code + 2), records, None);
            } else if let Some(v) = inst.write(Reg::S(code + k as u8)) {
                self.vals.insert(
                    v,
                    Val::Missing("a descriptor's address or format read as data"),
                );
            }
        }
        Ok(())
    }
}
