//! The size of a mesh shader's outputs as D3DMetal and Metal count it: the
//! number Metal tests against its 32,768 bytes
//! ([`super::METAL_MESH_OUTPUT_BYTES`]).
//!
//! Measured, not derived, by halfwave50 (halfwave50/VIctory-macOS-halfwave50
//! b50fd5b, GPL-3.0): a probe made pipelines of hand-assembled mesh and
//! pixel shaders under Wine and read the total D3DMetal prints when Metal
//! refuses one (D3DMetal 4.0 beta 3 on an M4 Pro). [`MEASURED`] holds what
//! it measured; the rule below gives every row of it, the 21 shapes of
//! `mesh_outputs_are_laid_out_as_metal_lays_them_out` (measured on an M5
//! Pro), and every size the title logged on that fork.
//!
//! The rule. One vertex is laid out as a struct:
//!
//! 1. `SV_Position` first, 16 bytes, wherever it is in the signature;
//! 2. 16 more bytes if the signature has any `SV_ClipDistance` element,
//!    however many distances there are;
//! 3. then every other element, the clip and cull distance elements among
//!    them, in the order of the signature (the order of the elements and of
//!    `PSV0`, not of the registers or the names). An element of n components
//!    takes 4n bytes and starts at the next multiple of 4n: 4, 8, 12 or 16.
//!    Component type and interpolation mode change nothing, and neither does
//!    anything in the pixel shader. Two elements in one register are two
//!    members. Nothing is added at the end.
//!
//! A primitive's elements are laid out the same way from 0. The total is the
//! vertex's bytes times the declared vertices rounded up to a multiple of 4,
//! plus the primitive's bytes times the declared primitives (not rounded).
//! 32,768 is allowed.

/// DXIL semantic kinds the rule names.
const POSITION: u8 = 3;
const CLIP_DISTANCE: u8 = 6;

/// The offset after an element of `cols` components placed at or after `at`.
pub(crate) fn after(at: u32, cols: u8) -> u32 {
    let size = 4 * u32::from(cols);
    at.next_multiple_of(size) + size
}

/// The bytes ahead of the elements that may be placed in any order: the
/// position and, with clip distances, 16 more. `kinds` are the semantic
/// kinds of a record's elements.
pub(crate) fn fixed_bytes(kinds: impl IntoIterator<Item = u8>) -> u32 {
    let (mut position, mut clip) = (false, false);
    for kind in kinds {
        position |= kind == POSITION;
        clip |= kind == CLIP_DISTANCE;
    }
    16 * (u32::from(position) + u32::from(clip))
}

/// The bytes of one vertex, or of one primitive, with these elements
/// `(DXIL semantic kind, columns)` in signature order.
pub(crate) fn record_bytes(elements: &[(u8, u8)]) -> u32 {
    elements
        .iter()
        .filter(|e| e.0 != POSITION && e.1 > 0)
        .fold(fixed_bytes(elements.iter().map(|e| e.0)), |at, e| {
            after(at, e.1)
        })
}

/// The total Metal tests against its limit.
pub(crate) fn total_bytes(vertex: u32, primitive: u32, vertices: u32, primitives: u32) -> u32 {
    vertex * vertices.next_multiple_of(4) + primitive * primitives
}

/// What halfwave50's probe measured: a signature in the probe's notation
/// ([`parse`]), the bytes of one vertex and of one primitive. Lines that
/// start with `#` are headings; `VICTORY_MERGE_MESH_ROWS` levels are that
/// fork's names for the layouts `lower::mesh` now plans.
#[cfg(test)]
pub(crate) const MEASURED: &[(&str, u32, u32)] = &[
    (
        "# the total is a*v + b*p: one signature at several counts",
        0,
        0,
    ),
    ("P f4*16 v=128", 272, 0),
    ("P f4*16 v=129", 272, 0),
    ("P f4*16 v=200", 272, 0),
    ("P f4*16 v=256", 272, 0),
    ("P f4*16 v=128 p=1", 272, 0),
    ("P f4*16 v=128 p=256", 272, 0),
    ("P f4*15 v=128", 256, 0),
    ("P f4*15 qf4 v=128 p=128", 256, 16),
    ("P f4*15 qf4 v=128 p=129", 256, 16),
    ("P f4*7 qf4*8 v=256 p=256", 128, 128),
    ("# what the pixel shader does with them", 0, 0),
    ("P f4*16 ps=decl", 272, 0),
    ("P f4*16 ps=none", 272, 0),
    ("P f3*16 ps=read", 216, 0),
    ("P f3*16 ps=none", 216, 0),
    (
        "# the position alone, and one row of each width after it",
        0,
        0,
    ),
    ("P", 16, 0),
    ("P f1", 20, 0),
    ("P f2", 24, 0),
    ("P f3", 36, 0),
    ("P f4", 32, 0),
    ("# runs of one width", 0, 0),
    ("P f1*2", 24, 0),
    ("P f1*3", 28, 0),
    ("P f1*4", 32, 0),
    ("P f1*5", 36, 0),
    ("P f2*2", 32, 0),
    ("P f2*3", 40, 0),
    ("P f2*4", 48, 0),
    ("P f3*2", 48, 0),
    ("P f3*3", 60, 0),
    ("P f3*4", 72, 0),
    ("P f3*5", 84, 0),
    ("P f3*8", 120, 0),
    ("P f4*2", 48, 0),
    ("P f4*3", 64, 0),
    ("P f4*8", 144, 0),
    ("# two widths, both orders", 0, 0),
    ("P f2 f4", 48, 0),
    ("P f4 f2", 40, 0),
    ("P f3 f4", 64, 0),
    ("P f4 f3", 48, 0),
    ("P f1 f4", 48, 0),
    ("P f4 f1", 36, 0),
    ("P f3 f1", 40, 0),
    ("P f1 f3", 36, 0),
    ("P f3 f2", 48, 0),
    ("P f2 f3", 36, 0),
    ("P f1 f2", 32, 0),
    ("P f2 f1", 28, 0),
    ("P f3 f4 f3 f4", 112, 0),
    ("P f3 f3 f4 f4", 80, 0),
    ("P f4 f4 f3 f3", 72, 0),
    ("P f3 f1 f3 f1", 64, 0),
    ("P f3 f3 f1 f1", 56, 0),
    ("P f2 f2 f4", 48, 0),
    ("P f2 f4 f2", 56, 0),
    ("P f1 f1 f1 f1 f4", 48, 0),
    ("P f1 f4 f1 f4", 80, 0),
    (
        "# registers: gaps, order in the signature against register order, the position's place",
        0,
        0,
    ),
    ("P f4@1 f4@2", 48, 0),
    ("P f4@1 f4@5", 48, 0),
    ("P f4@2 f4@1", 48, 0),
    ("P f3@2 f4@1", 64, 0),
    ("P f4@1 f3@2", 48, 0),
    ("f4@0 P@1 f3@2", 48, 0),
    ("f4@0 f3@1 P@2", 48, 0),
    ("f3@0 f4@1 P@2", 64, 0),
    ("P@2 f3@0 f4@1", 64, 0),
    ("# several elements in one register", 0, 0),
    ("P f3 +f1", 40, 0),
    ("P f2 +f2", 32, 0),
    ("P f1 +f1 +f1 +f1", 32, 0),
    ("P f1 +f2 +f1", 36, 0),
    ("P f2 +f1", 28, 0),
    ("P f3 +f1 f3 +f1", 64, 0),
    ("P f1@1.3 f3@1.0", 36, 0),
    ("P f3@1.1", 36, 0),
    ("P f1@1.3", 20, 0),
    ("P f2@1.2", 24, 0),
    (
        "# clip distances: how many, and in the first, a middle and the last register",
        0,
        0,
    ),
    ("P C1", 36, 0),
    ("P C2", 40, 0),
    ("P C3", 48, 0),
    ("P C4", 48, 0),
    ("P C4 f4", 64, 0),
    ("P C4 f3", 60, 0),
    ("P C1 f4", 64, 0),
    ("P C1 f3", 48, 0),
    ("P C4 f4*4", 112, 0),
    ("P f4*2 C4 f4*2", 112, 0),
    ("P f4*4 C4", 112, 0),
    ("P C4 f3*4", 96, 0),
    ("P f3*2 C4 f3*2", 108, 0),
    ("P f3*4 C4", 112, 0),
    ("P C1 f4*4", 112, 0),
    ("P f4*4 C1", 100, 0),
    ("P C4 f4*4 pd=0", 112, 0),
    ("P f4*4 C4 pd=0", 112, 0),
    ("C4@0 P@1 f4*4", 112, 0),
    ("P C4 C4 f4*2", 96, 0),
    ("# cull distances", 0, 0),
    ("P K4 f4*4", 96, 0),
    ("P f4*4 K4", 96, 0),
    ("P K1 f4*4", 96, 0),
    ("P C4 K4 f4*4", 128, 0),
    ("P C2 +K2 f4*4", 112, 0),
    ("# integers against floats, flat against interpolated", 0, 0),
    ("P u4", 32, 0),
    ("P u3", 36, 0),
    ("P u2", 24, 0),
    ("P u1", 20, 0),
    ("P i4", 32, 0),
    ("P i1", 20, 0),
    ("P u1 f4", 48, 0),
    ("P f4 u1", 36, 0),
    ("P u1 +u1 f4", 48, 0),
    ("P u1 +u1 +u1 f4", 48, 0),
    ("P u3 f4", 64, 0),
    ("P u3 u3", 48, 0),
    ("P c4", 32, 0),
    ("P c3", 36, 0),
    ("P c3 c3", 48, 0),
    ("P c3 f4", 64, 0),
    ("P n3 n3", 48, 0),
    ("P n3 f4", 64, 0),
    ("P f3 c3 f3 c3", 72, 0),
    ("P f3 f3 c3 c3", 72, 0),
    ("P f3 u3 f3 u3", 72, 0),
    ("P e3 e3", 48, 0),
    ("P s3 s3", 48, 0),
    ("P f3 +c1", 40, 0),
    ("P f2 +u2", 32, 0),
    ("# per-primitive rows", 0, 0),
    ("P f4 qf4", 32, 16),
    ("P f4 qf3", 32, 12),
    ("P f4 qf3*2", 32, 24),
    ("P f4 qf3 qf4", 32, 32),
    ("P f4 qf1", 32, 4),
    ("P f4 qu1", 32, 4),
    ("P f4 qf2 qf4", 32, 32),
    ("P f4 qf4*4", 32, 64),
    ("P f3*3 qf3*3", 60, 36),
    (
        "# the counts: vertices are rounded up to a multiple of 4, primitives are not",
        0,
        0,
    ),
    ("P f4*16 v=125", 272, 0),
    ("P f4*16 v=126", 272, 0),
    ("P f4*16 v=127", 272, 0),
    ("P f4*16 v=130", 272, 0),
    ("P f4*16 v=131", 272, 0),
    ("P f4*16 v=132", 272, 0),
    ("P f4*16 v=133", 272, 0),
    ("P f4*15 qf4 p=127", 256, 16),
    ("P f4*15 qf4 p=130", 256, 16),
    ("P f4*15 qf4 p=131", 256, 16),
    (
        "# where the 16 bytes of a shader with clip distances sit",
        0,
        0,
    ),
    ("P f1 C1", 40, 0),
    ("P f3 C1", 52, 0),
    ("P f2 C1", 44, 0),
    ("P f1 C4", 64, 0),
    ("P f3 C4", 64, 0),
    ("P f3 C4 f3", 84, 0),
    ("f3@0 P@1 C1@2", 52, 0),
    ("f3@0 C1@1 P@2", 52, 0),
    ("C1@0 f3@1 P@2", 48, 0),
    ("P f3 f3 C1 f1", 68, 0),
    ("P C1 C1", 40, 0),
    ("P C1 +C1", 40, 0),
    ("P C1 +C1 +C1 +C1", 48, 0),
    ("P C2 C2", 48, 0),
    ("P C4 C1", 52, 0),
    ("P C4 C4", 64, 0),
    ("P C4 C4 C4", 80, 0),
    ("P f3 K1", 40, 0),
    ("P f3 K4 f3", 84, 0),
    (
        "# clip distances: interpolation mode, and components never written (in another element, P f4!b, D3DMetal ends the process)",
        0,
        0,
    ),
    ("P C4^4 f4", 64, 0),
    ("P C4^1 f4", 64, 0),
    ("P C4^0 f4", 64, 0),
    ("P C4!1 f4", 64, 0),
    ("P C4!3 f4", 64, 0),
    ("P C4!0 f4", 64, 0),
    ("P f3 C4!1", 64, 0),
    (
        "# order: the signature's, the registers' or the names'",
        0,
        0,
    ),
    ("P f3:TEXCOORD0@2 f4:TEXCOORD1@1", 64, 0),
    ("P f3:TEXCOORD1@2 f4:TEXCOORD0@1", 64, 0),
    ("P f3:TEXCOORD1@1 f4:TEXCOORD0@2", 64, 0),
    ("P f4:TEXCOORD0@2 f3:TEXCOORD1@1", 48, 0),
    ("P f4:TEXCOORD1@2 f3:TEXCOORD0@1", 48, 0),
    ("P f4:TEXCOORD1@1 f3:TEXCOORD0@2", 48, 0),
    ("P f3:COLOR0 f4:TEXCOORD0", 64, 0),
    ("P f3:TEXCOORD0 f4:COLOR0", 64, 0),
    ("P f4:COLOR0 f3:TEXCOORD0", 48, 0),
    ("P f4:TEXCOORD0 f3:COLOR0", 48, 0),
    ("P f3:B5 f4:A7 f3:A2 f4:C0", 112, 0),
    ("P f3:A2 f3:B5 f4:A7 f4:C0", 80, 0),
    ("# one register, several elements", 0, 0),
    ("P f2@1 f1@1.3", 28, 0),
    ("P f2@1 f1@1.2", 28, 0),
    ("P f1@1 f2@1.2", 32, 0),
    ("P f2@1.2 f2@1.0", 32, 0),
    ("P f1@1.1 f1@1.3 f1@1.0 f1@1.2", 32, 0),
    ("# many small elements", 0, 0),
    ("P f1*52", 224, 0),
    ("P f1*60", 256, 0),
    ("P f2*30", 256, 0),
    ("P f1*16 f2*8 f3*4 f4*2", 224, 0),
    (
        "# shapes of 17 to 19 registers, each in three orders (the sizes a game on this port logged for them: 264 to 272, and 260 against a fit)",
        0,
        0,
    ),
    (
        "P@0 C4@1 f4@2 f4@3 f4@4 f4@5 f4@6 f4@7 f3@8 f3@9 f3@10 f3@11 f3@12 f3@13 f3@14 f3@15 f3@16 f2@17",
        264,
        0,
    ),
    (
        "P@0 C4@1 f4@2 f4@3 f4@4 f4@5 f4@6 f4@7 f4@8 f3@9 f3@10 f3@11 f3@12 f3@13 f3@14 f3@15 f3@16 f2@17",
        272,
        0,
    ),
    (
        "P@0 f4@1 f4@2 f4@3 f4@4 f4@5 f4@6 f3@7 f3@8 f3@9 f3@10 f3@11 f3@12 f3@13 f3@14 f3@15 f2@16 C4@17",
        272,
        0,
    ),
    (
        "P@0 f4@1 f4@2 f4@3 f4@4 f4@5 f4@6 f4@7 f3@8 f3@9 f3@10 f3@11 f3@12 f3@13 f3@14 f3@15 f2@16 C4@17",
        272,
        0,
    ),
    (
        "P@0 f3@1 f3@2 f3@3 f3@4 f3@5 f3@6 f3@7 f3@8 f3@9 f2@10 f4@11 f4@12 f4@13 f4@14 f4@15 f4@16 C4@17",
        272,
        0,
    ),
    (
        "P@0 f3@1 f3@2 f3@3 f3@4 f3@5 f3@6 f3@7 f3@8 f2@9 f4@10 f4@11 f4@12 f4@13 f4@14 f4@15 f4@16 C4@17",
        272,
        0,
    ),
    (
        "P@0 f4@1 f4@2 f4@3 f4@4 f4@5 f4@6 f3@7 f3@8 f3@9 f3@10 f3@11 f3@12 f3@13 f3@14 f2@15 f1@16 C1@17",
        248,
        0,
    ),
    (
        "P@0 f3@1 f3@2 f3@3 f3@4 f3@5 f3@6 f3@7 f3@8 f2@9 f1@10 f4@11 f4@12 f4@13 f4@14 f4@15 f4@16 C1@17",
        260,
        0,
    ),
    (
        "C1@0 P@1 f4@2 f4@3 f4@4 f4@5 f4@6 f4@7 f3@8 f3@10 f3@11 f3@12 f3@13 f3@14 f3@15 f3@16 f2@17.2 u1@18",
        252,
        0,
    ),
    (
        "f3@0 P@1 f3@2 f3@3 f3@4 f3@5 f3@6 f3@7 f3@8 f2@10.2 u1@11 f4@12 f4@13 f4@14 f4@15 f4@16 f4@17 C1@18",
        260,
        0,
    ),
    (
        "P@0 f4@1 f4@2 f4@3 f4@4 f4@5 f4@6 f4@7 f4@8 f4@9 f4@10 f4@11 f4@12 f3@13 f1@14.3 u1@15 C1@16",
        252,
        0,
    ),
    (
        "P@0 f3@1 f1@2.3 u1@3 f4@4 f4@5 f4@6 f4@7 f4@8 f4@9 f4@10 f4@11 f4@12 f4@13 f4@14 f4@15 C1@16",
        260,
        0,
    ),
    (
        "# the two ground-material shapes as VICTORY_MERGE_MESH_ROWS=6 makes them (`lower::mesh`'s test): the first fits exactly",
        0,
        0,
    ),
    (
        "P@0 f4@1 f4@2 f4@3 f4@4 f4@5 f2@6 f2@7 f1@7.3 f3@8 f3@9 f3@10 f3@11 f3@12 f3@13 f3@14 f3@15 f3@16 C4@17",
        256,
        0,
    ),
    (
        "P@0 f4@1 f4@2 f4@3 f4@4 f4@5 f4@6 C4@7 f3@8 f3@9 f3@10 f3@11 f3@12 f3@13 f3@14 f3@15 f2@16 f2@17 f1@17.3",
        260,
        0,
    ),
    (
        "# the second with its four clip distances as an ordinary element (the 16 bytes go)",
        0,
        0,
    ),
    (
        "P@0 f4@1 f4@2 f4@3 f4@4 f4@5 f4@6 f4@7 f3@8 f3@9 f3@10 f3@11 f3@12 f3@13 f3@14 f3@15 f2@16 f2@17 f1@17.3",
        252,
        0,
    ),
    (
        "P@0 f4@1 f4@2 f4@3 f4@4 f4@5 f4@6 f4@7 f2@8 f2@9 f1@9.3 f3@10 f3@11 f3@12 f3@13 f3@14 f3@15 f3@16 f3@17",
        252,
        0,
    ),
    ("P C3 C1 f4", 80, 0),
    ("# the second as VICTORY_MERGE_MESH_ROWS=7 makes it", 0, 0),
    (
        "P@0 f4@1 f4@2 f4@3 f4@4 f4@5 f3@6 f3@7 f3@8 f3@9 f3@10 f3@11 f3@12 f3@13 f4@14 f4@15 f2@16 f2@17 f1@17.3",
        244,
        0,
    ),
    (
        "# the two ground shapes as VICTORY_MERGE_MESH_ROWS=7 and 8 make them now: the three distances that never clip left out",
        0,
        0,
    ),
    (
        "P@0 f4@1 f4@2 f4@3 f4@4 f4@5 f2@6 f2@7 f1@7.3 f3@8 f3@9 f3@10 f3@11 f3@12 f3@13 f3@14 f3@15 f3@16 C1@17",
        244,
        0,
    ),
    (
        "P@0 f4@1 f4@2 f4@3 f4@4 f4@5 f4@6 f2@7 f1@7.3 C1@8 f3@9 f3@10 f3@11 f3@12 f3@13 f3@14 f3@15 f3@16 f2@17",
        248,
        0,
    ),
];

/// A vertex's elements and a primitive's, `(kind, columns)` each.
#[cfg(test)]
type Records = (Vec<(u8, u8)>, Vec<(u8, u8)>);

/// A line of [`MEASURED`] as the elements `(kind, columns)` of a vertex and
/// of a primitive, in signature order. The notation (halfwave50's probe):
/// `P` the position, `f1`..`f4` (also `c n e s i u`) an element of that many
/// components, `C`/`K` clip and cull distances, a `q` prefix per primitive,
/// `*n` repeats; registers (`@`), names (`:`), modes (`^`), masks (`!`), a
/// `+` prefix and `v=`/`p=`/`ps=`/`pd=` options change no size and are
/// skipped.
#[cfg(test)]
fn parse(line: &str) -> Records {
    let (mut vertex, mut primitive) = (Vec::new(), Vec::new());
    for token in line.split_whitespace().filter(|t| !t.contains('=')) {
        let (token, times) = match token.split_once('*') {
            Some((t, n)) => (t, n.parse().expect("a count")),
            None => (token, 1),
        };
        let token = token.trim_start_matches('+');
        let (prim, token) = match token.strip_prefix('q') {
            Some(t) => (true, t),
            None => (false, token),
        };
        let mut chars = token.chars();
        let letter = chars.next().expect("an element");
        let cols = chars
            .next()
            .and_then(|c| c.to_digit(10))
            .map_or(4, |c| c as u8);
        let kind = match letter {
            'P' => POSITION,
            'C' => CLIP_DISTANCE,
            'K' => 7,
            'f' | 'c' | 'n' | 'e' | 's' | 'i' | 'u' => 0,
            _ => panic!("{line}: {token}"),
        };
        let list = if prim { &mut primitive } else { &mut vertex };
        list.extend(std::iter::repeat_n((kind, cols), times));
    }
    (vertex, primitive)
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The rule gives every size the probe measured.
    #[test]
    fn the_rule_gives_every_measured_size() {
        let mut rows = 0;
        for &(line, vertex, primitive) in MEASURED.iter().filter(|m| !m.0.starts_with('#')) {
            let (v, p) = parse(line);
            assert_eq!(
                (record_bytes(&v), record_bytes(&p)),
                (vertex, primitive),
                "{line}"
            );
            rows += 1;
        }
        assert!(rows >= 200, "{rows} measured rows");
    }

    /// Vertices count in fours, primitives singly, and the limit itself is
    /// allowed (measured: `P f4*16` at 125 to 133 vertices is refused for
    /// 34816, 35904 and 36992 bytes; `P f4*15 qf4` at 127 primitives for
    /// 34800; `P f4*15` at 128 vertices is made).
    #[test]
    fn totals_round_vertices_up_to_fours() {
        for (vertices, total) in [
            (125, 34816),
            (128, 34816),
            (129, 35904),
            (132, 35904),
            (133, 36992),
            (256, 69632),
        ] {
            assert_eq!(
                total_bytes(272, 0, vertices, 128),
                total,
                "{vertices} vertices"
            );
        }
        assert_eq!(total_bytes(256, 16, 128, 127), 34800);
        assert_eq!(total_bytes(256, 16, 128, 129), 34832);
        assert_eq!(
            total_bytes(256, 0, 128, 128),
            super::super::METAL_MESH_OUTPUT_BYTES
        );
    }
}
