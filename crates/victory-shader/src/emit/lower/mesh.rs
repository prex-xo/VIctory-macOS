//! Mesh shader output signatures, counts and lane indices.

use super::{Attr, Bin, Export, HashMap, Lower, SigElement, V, Why, pred};
use crate::emit::mesh_size;
use crate::emit::shader::Mesh;

/// A semantic: `(UPPERCASE_NAME, index)`.
pub(crate) type Semantic = (String, u32);

/// What the paired pixel shader of a linked mesh shader reads, and the
/// layout both stages declare ([`plan_layout`]).
#[derive(Debug, Clone, Default)]
pub(crate) struct KeptOutputs {
    /// Semantic -> 4-bit absolute channel mask of the arbitrary (`kind ==
    /// 0`) inputs the pixel shader reads.
    pub reads: HashMap<Semantic, u8>,
    /// Planned from both stages' signatures; without it, arbitrary elements
    /// and distances go by size (`4`, `2`, `1`, `3`).
    pub layout: Option<Layout>,
}

impl KeptOutputs {
    pub(crate) fn new() -> Self {
        Self::default()
    }
}

/// How the two stages of a linked pair declare the mesh shader's vertex
/// outputs so that Metal's size of them ([`mesh_size`]) is smallest without
/// a value changing. Made from the mesh program's outputs and the paired
/// pixel shader's inputs by [`plan_layout`]; applied to each by
/// [`prune_signature`], so both declare the same elements in the same order.
///
/// After halfwave50's `VICTORY_MERGE_MESH_ROWS` levels 6 and 8
/// (halfwave50/VIctory-macOS-halfwave50 df9aab4, dc3e48b and 2beb020,
/// GPL-3.0), keyed by semantic rather than by register here: the stages'
/// signatures name the same semantics, and the registers are given anew in
/// element order when a linked shader is finished (`container::repack_rows`).
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub(crate) struct Layout {
    /// Elements that share a console row, of one component type and mode,
    /// each group declared as one element `VROW<n>` (`n` its place here)
    /// over their columns. An `xyz` and a `w` element take 16 + 4 bytes
    /// after a `float4` and up to 28 elsewhere, one `float4` 16.
    merged: Vec<Vec<Semantic>>,
    /// Elements (after merging) the pixel shader reads with a gap, as the
    /// absolute columns it reads: declared as one element `VPART<4j + c>`
    /// per run of them (`j` the place here, `c` the run's first column). A
    /// component nothing reads costs 4 bytes only while an element spans it.
    split: Vec<(Semantic, u8)>,
    /// Clip distance elements with distances that never clip
    /// ([`idle_clip`]), as the absolute columns that can: declared as those
    /// only, or not at all where none can.
    live_clip: Vec<(Semantic, u8)>,
    /// The order of the elements that may go anywhere after the position
    /// (arbitrary ones and distances), smallest by the measured rule
    /// ([`fitted_order`]).
    order: Vec<Semantic>,
}

fn semantic(e: &SigElement) -> Semantic {
    (e.name.to_ascii_uppercase(), e.index)
}

/// The columns an element spans, as an absolute mask.
fn span(e: &SigElement) -> u8 {
    ((1u8 << e.cols) - 1) << e.col
}

/// Elements a layout may place anywhere after the position: arbitrary ones,
/// and clip and cull distances, which Metal lays out as values too.
fn movable(e: &SigElement) -> bool {
    matches!(e.kind, 0 | 6 | 7) && e.row.is_some()
}

/// Keeps only arbitrary (`kind == 0`) signature elements whose semantic has at
/// least one channel read in `keep`, trimming unread leading and trailing
/// channels. System values (`kind != 0`) stay untouched ahead of arbitrary
/// elements. Then, with a [`Layout`], declares the elements as it says;
/// without one, orders arbitrary elements by descending component size (`4`,
/// `2`, `1`, `3`) so Metal's aligned mesh-output layout leaves few padding
/// holes, clip and cull distances among them (one right after the position
/// leaves a hole of up to 12 bytes before the first `float4`).
pub(super) fn prune_signature(elements: &mut Vec<SigElement>, keep: &KeptOutputs) {
    prune_unread(elements, &keep.reads);
    match &keep.layout {
        Some(layout) => layout.apply(elements),
        None => elements.sort_by_key(by_size),
    }
}

fn prune_unread(elements: &mut Vec<SigElement>, reads: &HashMap<Semantic, u8>) {
    elements.retain_mut(|e| {
        if e.kind != 0 {
            return true;
        }
        if e.row.is_none() {
            return false;
        }
        let used = reads.get(&semantic(e)).copied().unwrap_or(0);
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
}

/// The order without a [`Layout`]: system values first (stable), then
/// arbitrary elements and distances by size.
fn by_size(e: &SigElement) -> (u8, u8, String, u32, u8) {
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
}

impl Layout {
    /// Declares `elements` (pruned) as the layout says. Elements it does
    /// not name (system values, a pixel shader's per-primitive inputs) keep
    /// their places, system values first.
    fn apply(&self, elements: &mut Vec<SigElement>) {
        for (n, group) in self.merged.iter().enumerate() {
            let members: Vec<usize> = (0..elements.len())
                .filter(|&i| elements[i].kind == 0 && group.contains(&semantic(&elements[i])))
                .collect();
            if members.len() != group.len() {
                continue;
            }
            let col = members.iter().map(|&i| elements[i].col).min().unwrap_or(0);
            let end = members
                .iter()
                .map(|&i| elements[i].col + elements[i].cols)
                .max()
                .unwrap_or(0);
            let usage = members
                .iter()
                .fold(0, |u, &i| u | elements[i].usage << (elements[i].col - col));
            let first = &mut elements[members[0]];
            first.name = MERGED_NAME.to_string();
            first.index = n as u32;
            (first.col, first.cols, first.usage) = (col, end - col, usage);
            for &i in members[1..].iter().rev() {
                elements.remove(i);
            }
        }
        for (j, (key, read)) in self.split.iter().enumerate() {
            let Some(at) = elements
                .iter()
                .position(|e| e.kind == 0 && &semantic(e) == key)
            else {
                continue;
            };
            let e = elements.remove(at);
            let read = read & span(&e);
            let mut parts = Vec::new();
            let mut col = e.col;
            while col < e.col + e.cols {
                if read >> col & 1 == 0 {
                    col += 1;
                    continue;
                }
                let cols = (read >> col).trailing_ones() as u8;
                let mut part = e.clone();
                part.name = PART_NAME.to_string();
                part.index = 4 * j as u32 + u32::from(col);
                (part.col, part.cols) = (col, cols);
                part.usage = (e.usage >> (col - e.col)) & ((1 << cols) - 1);
                parts.push(part);
                col += cols;
            }
            elements.splice(at..at, parts);
        }
        elements.retain_mut(|e| {
            let live = self
                .live_clip
                .iter()
                .find(|c| e.kind == 6 && c.0 == semantic(e));
            let Some(live) = live.map(|c| c.1 & span(e)) else {
                return true;
            };
            if live == 0 {
                return false;
            }
            let col = live.trailing_zeros() as u8;
            let cols = (live >> col).trailing_ones() as u8;
            e.usage = (e.usage >> (col - e.col)) & ((1 << cols) - 1);
            (e.col, e.cols) = (col, cols);
            true
        });
        // Elements the order does not name (a pixel shader's per-primitive
        // inputs, a mesh shader's primitive outputs) by size after it.
        let rank = |e: &SigElement| {
            let named = movable(e).then(|| self.order.iter().position(|k| *k == semantic(e)));
            match named {
                None => (0, 0, by_size(e)),
                Some(Some(at)) => (1, at, by_size(e)),
                Some(None) => (2, 0, by_size(e)),
            }
        };
        elements.sort_by_key(rank);
    }
}

/// The semantic of merged elements; its index is the group's place.
const MERGED_NAME: &str = "VROW";
/// The semantic of a part of a split element; its index is four times the
/// element's place plus the part's first column.
const PART_NAME: &str = "VPART";

/// The layout of a linked pair: the mesh program's `outputs` and the pixel
/// program's `inputs` (their console signatures), pruned to what the pixel
/// shader reads (`reads`), with `live_clip` from [`idle_clip`] (empty where
/// the pixel shader reads a clip distance). Each step is taken only where
/// both stages declare the elements alike, so they declare the same thing.
pub(crate) fn plan_layout(
    reads: &HashMap<Semantic, u8>,
    outputs: &[SigElement],
    inputs: &[SigElement],
    live_clip: &[(Semantic, u8)],
) -> Layout {
    let (mut mesh, mut pixel) = (outputs.to_vec(), inputs.to_vec());
    prune_unread(&mut mesh, reads);
    prune_unread(&mut pixel, reads);
    // The pixel shader declares an element as the mesh shader does.
    let alike = |e: &SigElement, pixel: &[SigElement]| {
        pixel.iter().any(|p| {
            semantic(p) == semantic(e)
                && (p.kind, p.comp, p.half, p.col, p.cols)
                    == (e.kind, e.comp, e.half, e.col, e.cols)
        })
    };
    let mut layout = Layout::default();
    let mut rows: Vec<u32> = mesh
        .iter()
        .filter(|e| e.kind == 0)
        .filter_map(|e| e.row)
        .collect();
    rows.sort_unstable();
    rows.dedup();
    for r in rows {
        let members: Vec<&SigElement> = mesh
            .iter()
            .filter(|e| e.kind == 0 && e.row == Some(r))
            .collect();
        let first = members[0];
        let same = members
            .iter()
            .all(|e| (e.comp, e.half, e.interp) == (first.comp, first.half, first.interp));
        // In the pixel shader too, the members share one row.
        let pixel_rows: Vec<Option<u32>> = members
            .iter()
            .map(|m| {
                pixel
                    .iter()
                    .find(|p| semantic(p) == semantic(m))
                    .and_then(|p| p.row)
            })
            .collect();
        let one_row = pixel_rows
            .iter()
            .all(|r| r.is_some() && *r == pixel_rows[0]);
        if members.len() > 1 && same && one_row && members.iter().all(|m| alike(m, &pixel)) {
            layout
                .merged
                .push(members.iter().map(|m| semantic(m)).collect());
        }
    }
    layout.apply(&mut mesh);
    layout.apply(&mut pixel);
    // The columns read of each arbitrary element, merged ones by their
    // members' semantics.
    for e in mesh.iter().filter(|e| e.kind == 0) {
        let read = if e.name == MERGED_NAME {
            let group = &layout.merged[e.index as usize];
            let console = |k: &Semantic| outputs.iter().find(|o| o.kind == 0 && semantic(o) == *k);
            group.iter().fold(0, |m, k| {
                m | console(k).map_or(0, |o| reads.get(k).copied().unwrap_or(0) & span(o))
            })
        } else {
            reads.get(&semantic(e)).copied().unwrap_or(0)
        } & span(e);
        if read != span(e) && read != 0 && alike(e, &pixel) {
            layout.split.push((semantic(e), read));
        }
    }
    let split = Layout {
        split: layout.split.clone(),
        ..Layout::default()
    };
    split.apply(&mut mesh);
    // Distances that never clip, where the pixel shader declares the
    // element alike or not at all.
    for (key, live) in live_clip {
        let e = mesh.iter().find(|e| e.kind == 6 && semantic(e) == *key);
        if e.is_some_and(|e| alike(e, &pixel) || !pixel.iter().any(|p| semantic(p) == *key)) {
            layout.live_clip.push((key.clone(), *live));
        }
    }
    let trim = Layout {
        live_clip: layout.live_clip.clone(),
        ..Layout::default()
    };
    trim.apply(&mut mesh);
    let mut free: Vec<&SigElement> = mesh.iter().filter(|e| movable(e)).collect();
    free.sort_by_key(|e| by_size(e));
    let widths: Vec<u8> = free.iter().map(|e| e.cols).collect();
    let start = mesh_size::fixed_bytes(mesh.iter().filter(|e| e.row.is_some()).map(|e| e.kind));
    layout.order = fitted_order(start, &widths)
        .into_iter()
        .map(|i| semantic(free[i]))
        .collect();
    layout
}

/// The order of elements of these `widths` (components, in a preferred
/// order) in which a vertex is smallest by the measured rule, placed from
/// `start`: indices into `widths`. Only an element's width matters, and
/// alignments are 4, 8, 12 and 16 bytes, so the offset modulo 48; among
/// equal orders the one that keeps the given order longest is taken.
fn fitted_order(start: u32, widths: &[u8]) -> Vec<usize> {
    fn least(at: u32, left: &mut [u8; 5], memo: &mut HashMap<(u32, [u8; 5]), u32>) -> u32 {
        if left.iter().all(|&n| n == 0) {
            return 0;
        }
        let key = (at % 48, *left);
        if let Some(&known) = memo.get(&key) {
            return known;
        }
        let mut best = u32::MAX;
        for w in 1..=4 {
            if left[w] == 0 {
                continue;
            }
            left[w] -= 1;
            let n = mesh_size::after(at % 48, w as u8) - at % 48;
            best = best.min(n + least(at + n, left, memo));
            left[w] += 1;
        }
        memo.insert(key, best);
        best
    }
    let mut left = [0u8; 5];
    for &w in widths {
        left[usize::from(w.clamp(1, 4))] += 1;
    }
    let mut memo = HashMap::new();
    let mut taken = vec![false; widths.len()];
    let mut order = Vec::with_capacity(widths.len());
    let mut at = start;
    while order.len() < widths.len() {
        let goal = least(at, &mut left, &mut memo);
        // The first element, in the given order, of a width that reaches it.
        let next = (0..widths.len()).filter(|&i| !taken[i]).find(|&i| {
            let w = usize::from(widths[i].clamp(1, 4));
            left[w] -= 1;
            let n = mesh_size::after(at % 48, w as u8) - at % 48;
            let reaches = n + least(at + n, &mut left, &mut memo) == goal;
            left[w] += 1;
            reaches
        });
        let i = next.expect("an element reaches the least size");
        let w = usize::from(widths[i].clamp(1, 4));
        left[w] -= 1;
        taken[i] = true;
        order.push(i);
        at = mesh_size::after(at, w as u8);
    }
    order
}

/// The clip distance elements of a mesh program's console signature with
/// distances that never clip, as (semantic, absolute columns that can clip):
/// a distance every export of it writes as a constant that is not negative
/// (a `v_mov_b32` of +0.0 or more for every lane, a VGPR nothing set on
/// entry, which is lowered as zero, or a phi of such values). The clipper
/// removes only what is under zero, so such a distance removes nothing. The
/// title's ground materials write three of their four so. Only elements
/// alone in their row, whose other columns are one run (or none), are
/// named. After halfwave50's `lower::idle_clip` (2beb020).
pub(crate) fn idle_clip(
    p: &crate::ir::Program,
    io: &crate::io::Io,
    outputs: &[SigElement],
) -> Vec<(Semantic, u8)> {
    use crate::gfx10::Fields;
    use crate::io::Target;
    use crate::ir::{Reg, Value};
    use crate::native::Usage;
    fn idle(p: &crate::ir::Program, io: &crate::io::Io, v: Value, seen: &mut Vec<Value>) -> bool {
        if seen.contains(&v) {
            return true;
        }
        seen.push(v);
        if let Some(phi) = p.blocks.iter().flat_map(|b| &b.phis).find(|f| f.value == v) {
            return seen.len() < 64 && phi.args.iter().all(|&a| idle(p, io, a, seen));
        }
        if p.inputs
            .iter()
            .any(|i| i.1 == v && matches!(i.0, Reg::V(_)))
        {
            return !io.inputs.iter().any(|i| i.0 == v);
        }
        let writer = p
            .blocks
            .iter()
            .flat_map(|b| &b.insts)
            .find(|i| i.writes.iter().any(|w| w.1 == v));
        writer.is_some_and(|i| match &i.fields {
            Fields::Valu(f) if i.name() == "v_mov_b32" && f.neg == 0 && f.abs == 0 => f
                .src
                .first()
                .and_then(|&o| super::scalar_constant(o))
                .is_some_and(|bits| bits <= 0x7F80_0000),
            _ => false,
        })
    }
    // A parameter export whose row the lowering finds elsewhere (a mesh
    // program's unnamed first entry) could write a distance unseen here.
    if io.exports.iter().any(|x| {
        matches!(
            x.target,
            Target::Param {
                usage: Usage::Other(0),
                ..
            }
        )
    }) {
        return Vec::new();
    }
    // The distance channels in the order the position exports carry them.
    let mut dists: Vec<&SigElement> = outputs
        .iter()
        .filter(|e| matches!(e.kind, 6 | 7) && e.row.is_some())
        .collect();
    dists.sort_by_key(|e| (e.kind, e.index));
    let flat: Vec<(u32, u8)> = dists
        .iter()
        .flat_map(|e| (0..e.cols).map(move |c| (e.row.unwrap_or(0), e.col + c)))
        .collect();
    let mut found = Vec::new();
    for e in dists.iter().filter(|e| e.kind == 6) {
        let Some(r) = e.row else { continue };
        if outputs.iter().filter(|o| o.row == Some(r)).count() != 1 {
            continue;
        }
        let mut live = 0u8;
        for col in e.col..e.col + e.cols {
            let Some(k) = flat.iter().position(|&at| at == (r, col)) else {
                continue;
            };
            // What writes the channel: its position export, and a parameter
            // export of the same row.
            let (mut writes, mut all_idle) = (0, true);
            for x in &io.exports {
                let ch = match x.target {
                    Target::Position(n) if usize::from(n) == 1 + k / 4 => k % 4,
                    Target::Param { row, .. } if u32::from(row) == r => usize::from(col),
                    _ => continue,
                };
                if let Some(v) = x.values[ch] {
                    writes += 1;
                    all_idle &= idle(p, io, v, &mut Vec::new());
                }
            }
            if writes == 0 || !all_idle {
                live |= 1 << col;
            }
        }
        let low = live >> live.trailing_zeros().min(7);
        let run = live == 0 || low & (low + 1) == 0;
        if live != span(e) && run {
            found.push((semantic(e), live));
        }
    }
    found
}

/// Where the clip and cull distance channel `k` of a mesh or vertex
/// program's position exports (pos1, pos2) is declared: the element of
/// `declared` and its column. The channels are the console signature's
/// (`console`) distances in semantic order; one a [`Layout`] left out (a
/// distance that never clips) is declared nowhere.
pub(super) fn distance_channel(
    console: &[SigElement],
    declared: &[SigElement],
    k: usize,
) -> Option<(usize, u8)> {
    let mut dists: Vec<&SigElement> = console.iter().filter(|o| matches!(o.kind, 6 | 7)).collect();
    dists.sort_by_key(|o| (o.kind, o.index));
    let (e, c) = dists
        .iter()
        .flat_map(|o| (0..o.cols).map(move |c| (*o, o.col + c)))
        .nth(k)?;
    let id = declared
        .iter()
        .position(|o| (o.kind, o.index) == (e.kind, e.index) && c >= o.col && c < o.col + o.cols)?;
    Some((id, c - declared[id].col))
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

#[cfg(test)]
mod tests {
    use super::*;

    fn element(name: &str, index: u32, kind: u8, row: u32, col: u8, cols: u8) -> SigElement {
        SigElement {
            name: name.to_string(),
            index,
            kind,
            comp: 9,
            interp: 0,
            row: Some(row),
            col,
            cols,
            usage: (1 << cols) - 1,
            half: false,
        }
    }

    /// Metal's bytes for one vertex of these elements, in their order.
    fn vertex_bytes(elements: &[SigElement]) -> u32 {
        let list: Vec<(u8, u8)> = elements
            .iter()
            .filter(|e| e.row.is_some())
            .map(|e| (e.kind, e.cols))
            .collect();
        mesh_size::record_bytes(&list)
    }

    /// The outputs of the title's two 4-layer ground mesh programs (as
    /// halfwave50 found them): the position, four clip distances, and 16
    /// rows of arbitrary elements (three elements in row 5, two in row 7).
    /// The pixel shader reads all but the third component of row 15 and,
    /// with `seven` false, the fourth of row 7.
    fn ground_shape(seven: bool) -> (Vec<SigElement>, HashMap<Semantic, u8>) {
        let mut outputs = vec![
            element("SV_Position", 0, 3, 0, 0, 4),
            element("SV_ClipDistance", 0, 6, 1, 0, 4),
        ];
        let mut shape: Vec<(u32, u8, u8)> = vec![
            (2, 0, 4),
            (3, 0, 4),
            (4, 0, 4),
            (5, 0, 1),
            (5, 1, 2),
            (5, 3, 1),
            (6, 0, 4),
            (7, 0, 3),
            (7, 3, 1),
        ];
        shape.extend((8..14).map(|r| (r, 0, 3)));
        shape.extend([(14, 0, 2), (15, 0, 4), (16, 0, 3), (17, 0, 3)]);
        let mut reads = HashMap::new();
        for (n, (row, col, cols)) in shape.into_iter().enumerate() {
            let e = element("TEXCOORD", n as u32, 0, row, col, cols);
            let read = match (row, col) {
                (15, _) => 0xB,
                (7, 3) if !seven => 0,
                _ => span(&e),
            };
            if read != 0 {
                reads.insert(semantic(&e), read);
            }
            outputs.push(e);
        }
        (outputs, reads)
    }

    /// Both stages of a pair as a [`Layout`] declares them: the mesh
    /// shader's outputs, and the pixel shader's inputs, which list the same
    /// elements backwards and without the position.
    fn linked(seven: bool, live_clip: &[(Semantic, u8)]) -> (Vec<SigElement>, Vec<SigElement>) {
        let (outputs, reads) = ground_shape(seven);
        let pixel: Vec<SigElement> = outputs[1..].iter().rev().cloned().collect();
        let keep = KeptOutputs {
            layout: Some(plan_layout(&reads, &outputs, &pixel, live_clip)),
            reads,
        };
        let (mut mesh, mut ps) = (outputs, pixel);
        prune_signature(&mut mesh, &keep);
        prune_signature(&mut ps, &keep);
        (mesh, ps)
    }

    fn shape(elements: &[SigElement]) -> Vec<(String, u32, u8, u8, u8)> {
        elements
            .iter()
            .map(|e| (e.name.clone(), e.index, e.kind, e.col, e.cols))
            .collect()
    }

    /// The ground shapes as halfwave50's levels 6 and 8 make them, by the
    /// measured rule: the rows of several elements merged, the element read
    /// at x, y and w declared as its two parts, the smallest order. The
    /// first then fits Metal's 256 bytes a vertex exactly, the second is 4
    /// over (the probe measured both: `mesh_size::MEASURED`, 256 and 260);
    /// with the three distances their programs write as 0 left out, they
    /// are 244 and 248, and with none that can clip 16 + 4 * 53. The pixel
    /// shader declares the same elements in the same order.
    #[test]
    fn ground_shapes_fit_as_the_measured_rule_counts_them() {
        let clip = ("SV_CLIPDISTANCE".to_string(), 0);
        for (seven, live, bytes) in [
            (false, None, 256),
            (true, None, 260),
            (false, Some(0b0001), 244),
            (true, Some(0b0001), 248),
            (true, Some(0), 16 + 4 * 53),
        ] {
            let live_clip: Vec<(Semantic, u8)> =
                live.map(|m| (clip.clone(), m)).into_iter().collect();
            let (mesh, pixel) = linked(seven, &live_clip);
            assert_eq!(vertex_bytes(&mesh), bytes, "seven {seven}, live {live:?}");
            assert_eq!(
                shape(&pixel),
                shape(&mesh[1..]),
                "seven {seven}, live {live:?}"
            );
            let distances: Vec<(u8, u8)> = mesh
                .iter()
                .filter(|e| e.kind == 6)
                .map(|e| (e.col, e.cols))
                .collect();
            let expected = match live {
                None => vec![(0, 4)],
                Some(0) => vec![],
                Some(_) => vec![(0, 1)],
            };
            assert_eq!(distances, expected);
        }
        // Row 5's three elements are one, row 15's element two parts in
        // its register: x and y, and w.
        let (mesh, _) = linked(false, &[]);
        let merged: Vec<(u8, u8, u8)> = mesh
            .iter()
            .filter(|e| e.name == MERGED_NAME)
            .map(|e| (e.col, e.cols, e.usage))
            .collect();
        assert_eq!(merged, vec![(0, 4, 0xF)]);
        let parts: Vec<(u8, u8, Option<u32>)> = mesh
            .iter()
            .filter(|e| e.name == PART_NAME)
            .map(|e| (e.col, e.cols, e.row))
            .collect();
        assert_eq!(parts, vec![(0, 2, Some(15)), (3, 1, Some(15))]);
        // Without a layout the outputs only go by size: 264 bytes for both.
        for seven in [false, true] {
            let (mut mesh, reads) = ground_shape(seven);
            prune_signature(
                &mut mesh,
                &KeptOutputs {
                    reads,
                    layout: None,
                },
            );
            assert_eq!(vertex_bytes(&mesh), 264, "seven {seven}");
        }
    }

    /// Elements of one row are merged only where the pixel shader declares
    /// them alike in one row too, and of one component type.
    #[test]
    fn rows_merge_only_where_both_stages_share_them() {
        let outputs = vec![
            element("SV_Position", 0, 3, 0, 0, 4),
            element("TEXCOORD", 0, 0, 1, 0, 3),
            element("TEXCOORD", 1, 0, 1, 3, 1),
            element("TEXCOORD", 2, 0, 2, 0, 3),
            element("COLOR", 0, 0, 2, 3, 1),
        ];
        let mut outputs = outputs;
        outputs[4].comp = 5;
        let reads: HashMap<Semantic, u8> = outputs[1..]
            .iter()
            .map(|e| (semantic(e), span(e)))
            .collect();
        let layout = plan_layout(&reads, &outputs, &outputs[1..], &[]);
        assert_eq!(
            layout.merged,
            vec![vec![
                ("TEXCOORD".to_string(), 0),
                ("TEXCOORD".to_string(), 1)
            ]]
        );
        // The pixel shader has TEXCOORD1 in a row of its own.
        let mut pixel = outputs[1..].to_vec();
        pixel[1].row = Some(7);
        pixel[1].col = 3;
        assert!(plan_layout(&reads, &outputs, &pixel, &[]).merged.is_empty());
    }

    /// The order found is the smallest of all orders, and among those it
    /// keeps the given order of equal widths.
    #[test]
    fn the_fitted_order_is_the_smallest() {
        fn orders(n: usize) -> Vec<Vec<usize>> {
            fn go(k: usize, a: &mut Vec<usize>, out: &mut Vec<Vec<usize>>) {
                if k <= 1 {
                    out.push(a.clone());
                    return;
                }
                for i in 0..k {
                    go(k - 1, a, out);
                    a.swap(if k.is_multiple_of(2) { i } else { 0 }, k - 1);
                }
            }
            let mut out = Vec::new();
            go(n, &mut (0..n).collect(), &mut out);
            out
        }
        let size = |start: u32, widths: &[u8], order: &[usize]| {
            order
                .iter()
                .fold(start, |at, &i| mesh_size::after(at, widths[i]))
        };
        let cases: [(u32, &[u8]); 6] = [
            (16, &[4, 3, 3, 2]),
            (32, &[4, 4, 3, 3, 3, 2, 1, 2]),
            (32, &[4, 3, 3, 1, 2, 1, 1]),
            (16, &[3, 4, 4, 1]),
            (0, &[3, 4, 1, 2, 3, 4]),
            (32, &[3, 3, 3, 4, 1, 2, 2]),
        ];
        for (start, widths) in cases {
            let order = fitted_order(start, widths);
            let mut sorted = order.clone();
            sorted.sort_unstable();
            assert_eq!(
                sorted,
                (0..widths.len()).collect::<Vec<usize>>(),
                "{order:?}"
            );
            let least = orders(widths.len())
                .iter()
                .map(|o| size(start, widths, o))
                .min()
                .expect("an order");
            assert_eq!(
                size(start, widths, &order),
                least,
                "{start} {widths:?}: {order:?}"
            );
            for (i, a) in order.iter().enumerate() {
                for b in &order[i + 1..] {
                    assert!(widths[*a] != widths[*b] || a < b, "{order:?}");
                }
            }
        }
        // Widths that waste nothing in the given order keep it.
        assert_eq!(fitted_order(16, &[4, 4, 2, 1, 1]), vec![0, 1, 2, 3, 4]);
    }
}
