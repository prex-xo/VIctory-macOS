//! The 16/64-bit round mode (MODE bits 3:2) each block starts with, moved
//! out of `lower.rs` to keep it under the size limit.

use crate::gfx10::{Fields, Operand};
use crate::ir::Inst;

/// The 16/64-bit round mode (MODE bits 3:2: 0 nearest even, 3 toward zero)
/// after `inst`, from `mode` before it (`None`: not known).
pub(in crate::emit) fn round16_after(inst: &Inst, mode: Option<u8>) -> Option<u8> {
    match (inst.name(), &inst.fields) {
        ("s_round_mode", Fields::Sopp { imm }) => Some((imm >> 2 & 3) as u8),
        (
            "s_setreg_imm32_b32" | "s_setreg_b32",
            Fields::Sop {
                src, imm: Some(h), ..
            },
        ) => {
            let (id, off, size) = (h & 0x3F, h >> 6 & 0x1F, (h >> 11 & 0x1F) + 1);
            if id != 1 || off >= 4 || off + size <= 2 {
                return mode;
            }
            match src.first() {
                Some(&Operand::Literal(x)) if off <= 2 && off + size >= 4 => {
                    Some((x >> (2 - off) & 3) as u8)
                }
                _ => None,
            }
        }
        _ => mode,
    }
}

/// The 16/64-bit round mode each block starts with: nearest even at the
/// program's start, then as the MODE writes on the way set it; `None` where
/// paths that set it differently join (or a write is not a constant).
pub(super) fn round16(p: &crate::ir::Program) -> Vec<Option<u8>> {
    // None: not reached yet; Some(m): the mode (m None: not known).
    let mut entry: Vec<Option<Option<u8>>> = vec![None; p.blocks.len()];
    entry[0] = Some(Some(0));
    let order = crate::ir::reverse_postorder(&p.blocks);
    loop {
        let mut changed = false;
        for &b in &order {
            let Some(mut m) = entry[b] else { continue };
            for inst in &p.blocks[b].insts {
                m = round16_after(inst, m);
            }
            for s in p.blocks[b].exit.successors() {
                let new = match entry[s] {
                    None => Some(m),
                    Some(old) if old == m => continue,
                    Some(_) => Some(None),
                };
                if entry[s] != new {
                    entry[s] = new;
                    changed = true;
                }
            }
        }
        if !changed {
            return entry.into_iter().map(Option::flatten).collect();
        }
    }
}
