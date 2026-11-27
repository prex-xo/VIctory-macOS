//! Mesh shader output signatures, counts and lane indices.

use super::{Attr, Bin, Export, HashMap, Lower, SigElement, V, Why, pred};
use crate::emit::shader::Mesh;

/// Semantic key `(UPPERCASE_NAME, index)` -> 4-bit absolute channel mask of
/// arbitrary (`kind == 0`) inputs read by the paired pixel shader.
pub(crate) type KeptOutputs = HashMap<(String, u32), u8>;

/// Keeps only arbitrary (`kind == 0`) signature elements whose semantic has at
/// least one channel read in `keep`, trimming unread leading and trailing
/// channels and ordering arbitrary elements by descending component size
/// (`4`, `2`, `1`, `3`) so Metal's aligned mesh-output layout leaves no padding
/// holes between scalar/pair and `float4` attributes. System values (`kind != 0`)
/// stay untouched ahead of arbitrary elements, except clip and cull distances:
/// Metal lays them out as values in element order too, so they go by size
/// among the arbitrary ones (one right after the position leaves a hole of up
/// to 12 bytes before the first `float4`).
pub(super) fn prune_signature(elements: &mut Vec<SigElement>, keep: &KeptOutputs) {
    elements.retain_mut(|e| {
        if e.kind != 0 {
            return true;
        }
        if e.row.is_none() {
            return false;
        }
        let key = (e.name.to_ascii_uppercase(), e.index);
        let used = keep.get(&key).copied().unwrap_or(0);
        let rel = (used >> e.col) & ((1u8 << e.cols) - 1);
        if rel == 0 {
            return false;
        }
        let first = rel.trailing_zeros() as u8;
        let last = 7 - rel.leading_zeros() as u8;
        e.col += first;
        e.cols = last - first + 1;
        true
    });
    elements.sort_by_key(|e| {
        if !matches!(e.kind, 0 | 6 | 7) {
            (0u8, 0u8, String::new(), 0u32, 0u8)
        } else {
            let size_rank = match e.cols {
                4 => 0,
                2 => 1,
                1 => 2,
                _ => 3,
            };
            (1u8, size_rank, e.name.to_ascii_uppercase(), e.index, e.col)
        }
    });
}

impl Lower<'_> {
    /// Mesh: the counts and the primitives no lane exports. The console's
    /// counts come from one lane's allocation request, which DXIL cannot
    /// express (one call, dominating every output write).
    // shortcut: the counts are the declared maxima and each lane first culls
    // its primitive (all indices 0, zero area), so primitives past the
    // program's count are culled and vertices past it unused; this costs
    // the host the unused vertex and primitive slots. Pass the program's own
    // counts if that shows in frame time.
    pub(super) fn mesh_prologue(&mut self, mesh: Mesh) -> Result<(), Why> {
        let threads = self.s.threads.expect("mesh")[0];
        if mesh.topology != 2 {
            return Err("a mesh shader not drawing triangles");
        }
        if threads > 128 || mesh.max_vertices > 256 || mesh.max_primitives > 256 {
            return Err("a mesh shader beyond the host's limits");
        }
        let e = self.s.entry;
        let void = self.s.t.void;
        // A mesh with zero vertices cannot validly index any primitive, so
        // force the primitive count to 0 as well.
        let prim_count = if mesh.max_vertices == 0 {
            0
        } else {
            mesh.max_primitives
        };
        let (v, p) = (self.s.m.i32(mesh.max_vertices), self.s.m.i32(prim_count));
        self.s
            .op(e, "setMeshOutputCounts", Attr::NoUnwind, void, 168, &[v, p]);
        if prim_count == 0 {
            return Ok(());
        }
        let flat = self.mesh_thread();
        let guard = (threads > prim_count).then(|| self.s.m.cmp(e, pred::IULT, flat, p));
        let at = self.begin_guarded(guard);
        let z = self.s.m.i32(0);
        self.s.op(
            self.cur,
            "emitIndices",
            Attr::NoUnwind,
            void,
            169,
            &[flat, z, z, z],
        );
        self.end_guarded(at, &[]);
        Ok(())
    }

    /// Mesh: `exp prim` as the lane's primitive. Each corner takes ten
    /// bits: a 9-bit vertex index and, above it, its edge flag (bits 9, 19
    /// and 29), which only the console's edge-flag rasterisation reads and
    /// which is no part of the index. Bit 31 culls it (all indices 0: no
    /// area). Non-culled corner indices are clamped below the mesh's vertex
    /// count so out-of-bounds 9-bit indices cannot read past the vertex
    /// array or wrap modulo 256 on hosts with 8-bit mesh index storage.
    pub(super) fn emit_indices(&mut self, e: &Export, guard: Option<V>) -> Result<(), Why> {
        let v = e.values[0].ok_or("a primitive export without its indices")?;
        let x = self.i32_of(v)?;
        let mesh = self.s.mesh.expect("mesh");
        if mesh.max_vertices == 0 || mesh.max_primitives == 0 {
            return Ok(());
        }
        let threads = self.s.threads.expect("mesh")[0];
        let flat = self.mesh_thread();
        let guard = if threads > mesh.max_primitives {
            let p = self.s.m.i32(mesh.max_primitives);
            let in_range = self.s.m.cmp(self.cur, pred::IULT, flat, p);
            Some(match guard {
                Some(g) => self.s.m.bin(self.cur, Bin::And, g, in_range),
                None => in_range,
            })
        } else {
            guard
        };
        let at = self.begin_guarded(guard);
        let c = self.cur;
        let z = self.s.m.i32(0);
        let last = self.s.m.i32(mesh.max_vertices - 1);
        let culled = self.s.m.cmp(c, pred::SLT, x, z);
        let mut args = vec![flat];
        for k in 0..3 {
            let (shift, mask) = (self.s.m.i32(10 * k), self.s.m.i32(0x1FF));
            let i = self.s.m.bin(c, Bin::LShr, x, shift);
            let i = self.s.m.bin(c, Bin::And, i, mask);
            let in_bounds = self.s.m.cmp(c, pred::IULT, i, last);
            let clamped = self.s.m.select(c, in_bounds, i, last);
            args.push(self.s.m.select(c, culled, z, clamped));
        }
        let void = self.s.t.void;
        self.s
            .op(c, "emitIndices", Attr::NoUnwind, void, 169, &args);
        self.end_guarded(at, &[]);
        Ok(())
    }

    /// Mesh: the thread's index in the group, which is its vertex and
    /// primitive index.
    pub(super) fn mesh_thread(&mut self) -> V {
        if let Some(f) = self.flat {
            return f;
        }
        let e = self.s.entry;
        let f = self.s.op(
            e,
            "flattenedThreadIdInGroup.i32",
            Attr::ReadNone,
            self.s.t.i32,
            96,
            &[],
        );
        self.flat = Some(f);
        f
    }
}
