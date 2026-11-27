//! Finds the AVX masked stores in a title's code (`vpmaskmovd`,
//! `vpmaskmovq`, `vmaskmovps` and `vmaskmovpd` to memory).
//!
//! Rosetta cannot resume one of these that faults partway (#14), so
//! fault-based write tracking is off under Rosetta and every submission
//! copies title memory whole: about 129 GB per frame once the world
//! streams in (#50). Knowing where the title uses these stores is the
//! first step to patching them so tracking can come back on. The scan
//! reads bytes, not decoded instructions, so a match inside another
//! instruction's bytes is possible, though rare.

use victory_pe::ImageInfo;
use victory_pe::layout::section_flags;

/// One masked store found.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct MaskedStore {
    pub rva: u32,
    pub kind: MaskedKind,
    /// 256-bit (`ymm`) rather than 128-bit.
    pub wide: bool,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum MaskedKind {
    Vpmaskmovd,
    Vpmaskmovq,
    Vmaskmovps,
    Vmaskmovpd,
    /// Not masked, but a vector store Rosetta also cannot resume (#50).
    Vcvtps2ph,
    Vextractf128,
    Vextracti128,
}

/// The masked store whose encoding starts at `at`, if one does: a
/// three-byte VEX prefix (`c4`) in the 0F38 map with the 66 prefix, then
/// opcode 8e (`vpmaskmov`, W picks d or q), 2e (`vmaskmovps`) or 2f
/// (`vmaskmovpd`), and a memory operand.
fn decode(code: &[u8], at: usize) -> Option<(MaskedKind, bool)> {
    let &[c4, b1, b2, opcode, modrm] = code.get(at..at + 5)? else {
        return None;
    };
    if c4 != 0xc4 || b2 & 3 != 1 || modrm >> 6 == 3 {
        return None;
    }
    let w = b2 & 0x80 != 0;
    let kind = match (b1 & 0x1f, opcode) {
        (2, 0x8e) if w => MaskedKind::Vpmaskmovq,
        (2, 0x8e) => MaskedKind::Vpmaskmovd,
        (2, 0x2e) => MaskedKind::Vmaskmovps,
        (2, 0x2f) => MaskedKind::Vmaskmovpd,
        (3, 0x1d) if !w => MaskedKind::Vcvtps2ph,
        (3, 0x19) if !w && b2 & 4 != 0 => MaskedKind::Vextractf128,
        (3, 0x39) if !w && b2 & 4 != 0 => MaskedKind::Vextracti128,
        _ => return None,
    };
    Some((kind, b2 & 4 != 0))
}

/// Every masked store in the executable sections of `bytes` (the image
/// file `info` was parsed from), by RVA.
pub fn masked_stores(info: &ImageInfo, bytes: &[u8]) -> Vec<MaskedStore> {
    let mut found = Vec::new();
    for s in info
        .sections
        .iter()
        .filter(|s| s.characteristics & section_flags::MEM_EXECUTE != 0)
    {
        let start = (s.file_offset as usize).min(bytes.len());
        let end = start.saturating_add(s.file_size as usize).min(bytes.len());
        let code = &bytes[start..end];
        for at in 0..code.len() {
            if let Some((kind, wide)) = decode(code, at) {
                // An offset past 4 GiB has no RVA; skip the store rather
                // than wrap.
                if let Some(rva) = u32::try_from(at)
                    .ok()
                    .and_then(|off| s.rva.checked_add(off))
                {
                    found.push(MaskedStore { rva, kind, wide });
                }
            }
        }
    }
    found
}

/// One line for the start-up report: how many of each kind, and the first
/// few RVAs.
pub fn summary(found: &[MaskedStore]) -> String {
    if found.is_empty() {
        return "none in code".to_owned();
    }
    let count = |k| found.iter().filter(|f| f.kind == k).count();
    let first: Vec<String> = found
        .iter()
        .take(16)
        .map(|f| format!("{:#x}", f.rva))
        .collect();
    format!(
        "{} in code (vpmaskmovd {}, vpmaskmovq {}, vmaskmovps {}, vmaskmovpd {}, vcvtps2ph {}, vextractf128 {}, vextracti128 {}), first at {}{}",
        found.len(),
        count(MaskedKind::Vpmaskmovd),
        count(MaskedKind::Vpmaskmovq),
        count(MaskedKind::Vmaskmovps),
        count(MaskedKind::Vmaskmovpd),
        count(MaskedKind::Vcvtps2ph),
        count(MaskedKind::Vextractf128),
        count(MaskedKind::Vextracti128),
        first.join(" "),
        if found.len() > 16 { " ..." } else { "" }
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    /// An `ImageInfo` whose one executable section covers `len` bytes of
    /// file offset 0 at `rva`.
    fn exec_info(rva: u32, len: u32) -> ImageInfo {
        ImageInfo {
            machine: victory_pe::MACHINE_AMD64,
            characteristics: 0,
            timestamp: 0,
            subsystem: 0,
            dll_characteristics: 0,
            image_base: 0,
            size_of_image: 0,
            entry_rva: 0,
            stack_reserve: 0,
            stack_commit: 0,
            sections: vec![victory_pe::SectionInfo {
                name: ".text".into(),
                rva,
                virtual_size: len,
                file_offset: 0,
                file_size: len,
                characteristics: section_flags::MEM_EXECUTE,
            }],
            imports: Vec::new(),
            delay_imports: Vec::new(),
            exports: Vec::new(),
            tls: None,
            has_relocations: false,
            has_exception_table: false,
            has_load_config: false,
            codeview: None,
        }
    }

    #[test]
    fn stores_whose_rva_would_overflow_are_skipped() {
        // vpmaskmovd at offset 0 (its RVA fits) and at offset 5 (past the
        // 4 GiB limit: a panic in debug builds before).
        let code = [0xc4, 0xe2, 0x45, 0x8e, 0x30, 0xc4, 0xe2, 0x45, 0x8e, 0x30];
        let info = exec_info(u32::MAX - 2, code.len() as u32);
        assert_eq!(
            masked_stores(&info, &code),
            [MaskedStore {
                rva: u32::MAX - 2,
                kind: MaskedKind::Vpmaskmovd,
                wide: true,
            }]
        );
    }

    #[test]
    fn decodes_the_store_seen_in_the_rosetta_hang() {
        // `vpmaskmovd [rax], ymm6, ymm4` from #1 and #50: c4 e2 45 8e 30.
        assert_eq!(
            decode(&[0xc4, 0xe2, 0x45, 0x8e, 0x30], 0),
            Some((MaskedKind::Vpmaskmovd, true))
        );
        // W set: vpmaskmovq; 128-bit.
        assert_eq!(
            decode(&[0xc4, 0xe2, 0xc1, 0x8e, 0x30], 0),
            Some((MaskedKind::Vpmaskmovq, false))
        );
        assert_eq!(
            decode(&[0xc4, 0xe2, 0x45, 0x2e, 0x08], 0),
            Some((MaskedKind::Vmaskmovps, true))
        );
        assert_eq!(
            decode(&[0xc4, 0xe2, 0x45, 0x2f, 0x08], 0),
            Some((MaskedKind::Vmaskmovpd, true))
        );
    }

    #[test]
    fn leaves_loads_registers_and_other_maps_alone() {
        // The load forms (8c, 2c) are not stores.
        assert_eq!(decode(&[0xc4, 0xe2, 0x45, 0x8c, 0x30], 0), None);
        // A register operand (mod 3) is not a store to memory.
        assert_eq!(decode(&[0xc4, 0xe2, 0x45, 0x8e, 0xf0], 0), None);
        // The 0F map, or no 66 prefix.
        assert_eq!(decode(&[0xc4, 0xe1, 0x45, 0x8e, 0x30], 0), None);
        assert_eq!(decode(&[0xc4, 0xe2, 0x44, 0x8e, 0x30], 0), None);
        // Too short.
        assert_eq!(decode(&[0xc4, 0xe2, 0x45, 0x8e], 0), None);
    }

    #[test]
    fn the_summary_counts_and_lists() {
        let f = |rva, kind| MaskedStore {
            rva,
            kind,
            wide: true,
        };
        let found = [
            f(0x1000, MaskedKind::Vpmaskmovd),
            f(0x2000, MaskedKind::Vmaskmovps),
        ];
        assert_eq!(
            summary(&found),
            "2 in code (vpmaskmovd 1, vpmaskmovq 0, vmaskmovps 1, vmaskmovpd 0, vcvtps2ph 0, vextractf128 0, vextracti128 0), first at 0x1000 0x2000"
        );
        assert_eq!(summary(&[]), "none in code");
    }
}

/// A decoded masked store, enough to redo it with plain stores.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) struct Store {
    /// Instruction length in bytes.
    pub len: usize,
    /// Lanes of 8 bytes (`q`, `pd`) rather than 4.
    pub wide_lanes: bool,
    /// 256-bit registers.
    pub ymm: bool,
    /// The mask register (VEX.vvvv) and the stored register (ModRM.reg).
    pub mask: u8,
    pub src: u8,
    /// The memory operand: REX X and B, then ModRM, an optional SIB and
    /// the displacement.
    pub x: bool,
    pub b: bool,
    pub modrm: u8,
    pub sib: Option<u8>,
    pub disp: i32,
    /// For an unmasked store (`vcvtps2ph`, `vextract*128`, a plain 256-bit
    /// move): the three-byte VEX prefix's two bytes, the opcode, the
    /// immediate if there is one, and the bytes stored.
    pub plain: Option<(u8, u8, u8, Option<u8>, u8)>,
}

impl Store {
    fn lanes(&self) -> usize {
        match (self.ymm, self.wide_lanes) {
            (true, false) => 8,
            (true, true) | (false, false) => 4,
            (false, true) => 2,
        }
    }
    fn lane_bytes(&self) -> u8 {
        if self.wide_lanes { 8 } else { 4 }
    }
    /// RIP-relative: ModRM mod 0, r/m 5.
    fn rip_relative(&self) -> bool {
        self.modrm >> 6 == 0 && self.modrm & 7 == 5
    }
    /// A SIB with no base (mod 0, base 5): an absolute disp32.
    fn no_base(&self) -> bool {
        self.modrm >> 6 == 0 && self.sib.is_some_and(|s| s & 7 == 5)
    }
    /// The base register is rsp.
    fn rsp_base(&self) -> bool {
        !self.b && self.sib.is_some_and(|s| s & 7 == 4)
    }
}

/// The memory operand whose ModRM byte is at `at`: ModRM, the SIB byte if
/// any, the displacement, and where the operand ends.
fn operand(code: &[u8], at: usize) -> Option<(u8, Option<u8>, i32, usize)> {
    let modrm = *code.get(at)?;
    let mut end = at + 1;
    let (md, rm) = (modrm >> 6, modrm & 7);
    let sib = if rm == 4 {
        end += 1;
        Some(*code.get(at + 1)?)
    } else {
        None
    };
    let disp_len = match md {
        1 => 1,
        2 => 4,
        _ if rm == 5 || sib.is_some_and(|s| s & 7 == 5) => 4,
        _ => 0,
    };
    let bytes = code.get(end..end + disp_len)?;
    let disp = match disp_len {
        1 => i32::from(bytes[0] as i8),
        4 => i32::from_le_bytes(bytes.try_into().ok()?),
        _ => 0,
    };
    Some((modrm, sib, disp, end + disp_len))
}

/// Decodes the whole masked store at `at`: its operands and length.
pub(crate) fn decode_store(code: &[u8], at: usize) -> Option<Store> {
    let (kind, ymm) = decode(code, at)?;
    let (b1, b2) = (code[at + 1], code[at + 2]);
    let (modrm, sib, disp, mut end) = operand(code, at + 4)?;
    let plain = match kind {
        MaskedKind::Vcvtps2ph | MaskedKind::Vextractf128 | MaskedKind::Vextracti128 => {
            let imm = *code.get(end)?;
            end += 1;
            let bytes = if kind == MaskedKind::Vcvtps2ph && !ymm {
                8
            } else {
                16
            };
            Some((b1, b2, code[at + 3], Some(imm), bytes))
        }
        _ => None,
    };
    Some(Store {
        plain,
        len: end - at,
        wide_lanes: matches!(kind, MaskedKind::Vpmaskmovq | MaskedKind::Vmaskmovpd),
        ymm,
        mask: !(b2 >> 3) & 0xf,
        src: ((modrm >> 3) & 7) | if b1 & 0x80 == 0 { 8 } else { 0 },
        x: b1 & 0x40 == 0,
        b: b1 & 0x20 == 0,
        modrm,
        sib,
        disp,
    })
}

/// Decodes the plain 256-bit store at `at` that need not be aligned:
/// `vmovups`, `vmovupd` or `vmovdqu` to memory, with a two- or three-byte
/// VEX prefix. Rosetta can fail these the same way when they fault
/// partway (#144). They are far more common than
/// masked stores and the scan does not look for them, so only RVAs named
/// in `VICTORY_PATCH_MASKED` are decoded this way. Under 5 bytes there is
/// no room for the jump, so those are left alone.
pub(crate) fn decode_wide_move(code: &[u8], at: usize) -> Option<Store> {
    let (b1, b2, vex) = match *code.get(at)? {
        0xc4 => (*code.get(at + 1)?, *code.get(at + 2)?, 3),
        // The two-byte form as the three-byte one: its R, no X or B
        // (set bits), the 0F map, W 0.
        0xc5 => {
            let p = *code.get(at + 1)?;
            (p & 0x80 | 0x61, p & 0x7f, 2)
        }
        _ => return None,
    };
    let opcode = *code.get(at + vex)?;
    // The 0F map, no second source (vvvv 1111), 256-bit; then 11 with no
    // prefix or 66 (`vmovups`, `vmovupd`) or 7f with F3 (`vmovdqu`).
    let moves = matches!((opcode, b2 & 3), (0x11, 0 | 1) | (0x7f, 2));
    if b1 & 0x1f != 1 || b2 & 0x7c != 0x7c || !moves {
        return None;
    }
    let (modrm, sib, disp, end) = operand(code, at + vex + 1)?;
    if modrm >> 6 == 3 || end - at < 5 {
        return None;
    }
    Some(Store {
        plain: Some((b1, b2, opcode, None, 32)),
        len: end - at,
        wide_lanes: false,
        ymm: true,
        mask: 0,
        src: ((modrm >> 3) & 7) | if b1 & 0x80 == 0 { 8 } else { 0 },
        x: b1 & 0x40 == 0,
        b: b1 & 0x20 == 0,
        modrm,
        sib,
        disp,
    })
}

/// Bytes the stack moves before the trampoline's `lea`: rax, rcx, flags.
const PUSHED: i32 = 24;

/// The trampoline for `s`, placed at `at`, for the store at `site`: the
/// same stores done one lane at a time with plain `mov`s (which Rosetta
/// can resume after a fault), then a jump back past the store. Flags,
/// rax, rcx and the vector registers are kept.
pub(crate) fn trampoline(s: &Store, site: u64, at: u64) -> Option<Vec<u8>> {
    let mut c = vec![0x50, 0x51, 0x9c]; // push rax; push rcx; pushfq
    // lea rax, [operand]
    c.push(0x48 | u8::from(s.x) << 1 | u8::from(s.b));
    c.push(0x8d);
    let rm = s.modrm & 7;
    if s.rip_relative() {
        c.push(rm); // mod 0, reg 0 (rax), r/m 5
        let target = (site as i64) + s.len as i64 + i64::from(s.disp);
        let next = at as i64 + c.len() as i64 + 4;
        c.extend_from_slice(&i32::try_from(target - next).ok()?.to_le_bytes());
    } else if s.no_base() {
        c.push(rm);
        c.push(s.sib?);
        c.extend_from_slice(&s.disp.to_le_bytes());
    } else {
        // Always a disp32 form (mod 2), so rsp's shift always fits.
        c.push(0x80 | rm);
        if let Some(sib) = s.sib {
            c.push(sib);
        }
        let disp = if s.rsp_base() {
            s.disp.checked_add(PUSHED)?
        } else {
            s.disp
        };
        c.extend_from_slice(&disp.to_le_bytes());
    }
    if let Some((b1, b2, opcode, imm, bytes)) = s.plain {
        // The same instruction into [rsp] (X and B cleared, ModRM.reg
        // kept), then the bytes copied out 8 at a time with plain moves.
        c.extend_from_slice(&[0x48, 0x83, 0xec, 0x20]); // sub rsp, 32
        c.extend_from_slice(&[
            0xc4,
            b1 | 0x60,
            b2,
            opcode,
            0x40 | (s.modrm & 0x38) | 4,
            0x24,
            0,
        ]);
        c.extend(imm);
        for k in 0..bytes / 8 {
            c.extend_from_slice(&[0x48, 0x8b, 0x4c, 0x24, k * 8]); // mov rcx, [rsp + 8k]
            c.extend_from_slice(&[0x48, 0x89, 0x48, k * 8]); // mov [rax + 8k], rcx
        }
        c.extend_from_slice(&[0x48, 0x83, 0xc4, 0x20]); // add rsp, 32
        return tail(c, site, s.len, at);
    }
    c.extend_from_slice(&[0x48, 0x83, 0xec, 0x40]); // sub rsp, 64
    // vmovdqu [rsp], mask; vmovdqu [rsp + 32], src (F3 0F 7F /r).
    for (reg, disp) in [(s.mask, None), (s.src, Some(0x20u8))] {
        c.push(0xc4);
        c.push(if reg & 8 == 0 { 0x80 } else { 0 } | 0x60 | 0x01);
        c.push(0x78 | if s.ymm { 4 } else { 0 } | 0x02);
        c.push(0x7f);
        c.push(if disp.is_some() { 0x40 } else { 0 } | (reg & 7) << 3 | 4);
        c.push(0x24);
        if let Some(d) = disp {
            c.push(d);
        }
    }
    let (n, size) = (s.lanes(), s.lane_bytes());
    let rex = if size == 8 { Some(0x48) } else { None };
    for i in 0..n as u8 {
        let off = i * size;
        // mov ecx/rcx, [rsp + off]; test; jns skip
        c.extend(rex);
        c.extend_from_slice(&[0x8b, 0x4c, 0x24, off]);
        c.extend(rex);
        c.extend_from_slice(&[0x85, 0xc9]);
        // The two moves after the jump.
        let skip = 2 * u8::from(rex.is_some()) + 7;
        c.extend_from_slice(&[0x79, skip]);
        // mov ecx/rcx, [rsp + 32 + off]; mov [rax + off], ecx/rcx
        c.extend(rex);
        c.extend_from_slice(&[0x8b, 0x4c, 0x24, 0x20 + off]);
        c.extend(rex);
        c.extend_from_slice(&[0x89, 0x48, off]);
    }
    c.extend_from_slice(&[0x48, 0x83, 0xc4, 0x40]); // add rsp, 64
    tail(c, site, s.len, at)
}

/// Restores flags, rcx and rax, and jumps back past the store.
fn tail(mut c: Vec<u8>, site: u64, len: usize, at: u64) -> Option<Vec<u8>> {
    c.extend_from_slice(&[0x9d, 0x59, 0x58]); // popfq; pop rcx; pop rax
    let back = (site as i64) + len as i64;
    let next = at as i64 + c.len() as i64 + 5;
    c.push(0xe9);
    c.extend_from_slice(&i32::try_from(back - next).ok()?.to_le_bytes());
    Some(c)
}

/// The bytes that replace the store at `site`: a jump to `to`, then
/// `nop`s to the store's length.
pub(crate) fn jump(site: u64, len: usize, to: u64) -> Option<Vec<u8>> {
    let rel = i32::try_from(to as i64 - (site as i64 + 5)).ok()?;
    let mut c = vec![0xe9];
    c.extend_from_slice(&rel.to_le_bytes());
    c.resize(len.max(5), 0x90);
    Some(c)
}

/// The bytes of the executable section that holds `rva`, in the image
/// file, and where `rva` is in them.
#[cfg(windows)]
fn code_at<'a>(info: &ImageInfo, file: &'a [u8], rva: u32) -> Option<(&'a [u8], usize)> {
    let s = info.sections.iter().find(|s| {
        s.characteristics & section_flags::MEM_EXECUTE != 0
            && s.rva <= rva
            && rva - s.rva < s.file_size
    })?;
    let start = s.file_offset as usize;
    let code = file.get(start..start.checked_add(s.file_size as usize)?)?;
    Some((code, (rva - s.rva) as usize))
}

/// Patches the stores of the image mapped at `base` that `wanted` picks
/// to jump to trampolines of plain stores, and returns how many it
/// patched. Before any title thread runs.
#[cfg(windows)]
pub fn patch(base: u64, info: &ImageInfo, file: &[u8], wanted: &Wanted) -> usize {
    use windows_sys::Win32::System::Diagnostics::Debug::FlushInstructionCache;
    use windows_sys::Win32::System::Memory::{
        MEM_COMMIT, MEM_RESERVE, PAGE_EXECUTE_READ, PAGE_EXECUTE_READWRITE, PAGE_READWRITE,
        VirtualAlloc, VirtualProtect,
    };
    use windows_sys::Win32::System::Threading::GetCurrentProcess;
    let mut stores: Vec<(u32, Store)> = masked_stores(info, file)
        .into_iter()
        .filter(|m| wanted.picks(m.rva))
        .filter_map(|m| {
            let (code, at) = code_at(info, file, m.rva)?;
            Some((m.rva, decode_store(code, at)?))
        })
        .collect();
    // Named RVAs the scan did not find: plain 256-bit stores (#144).
    for &rva in &wanted.rvas {
        if stores.iter().any(|(r, _)| *r == rva) {
            continue;
        }
        match code_at(info, file, rva).and_then(|(code, at)| decode_wide_move(code, at)) {
            Some(store) => stores.push((rva, store)),
            None => tracing::warn!(
                rva = format_args!("{rva:#x}"),
                "not a masked store or a plain 256-bit store; left as it is"
            ),
        }
    }
    if stores.is_empty() {
        return 0;
    }
    // Trampolines within reach of a rel32 jump: below the image first.
    const SIZE: usize = 1 << 20;
    let mut pool = std::ptr::null_mut();
    for step in 1..2048u64 {
        let Some(want) = base.checked_sub(step << 20) else {
            break;
        };
        // SAFETY: a plain reservation at a hint; failure returns null.
        pool = unsafe {
            VirtualAlloc(
                want as *const _,
                SIZE,
                MEM_RESERVE | MEM_COMMIT,
                PAGE_READWRITE,
            )
        };
        if !pool.is_null() {
            break;
        }
    }
    if pool.is_null() {
        tracing::warn!("no memory near the image for masked store trampolines");
        return 0;
    }
    let mut used = 0usize;
    let mut patched = 0;
    for (rva, store) in &stores {
        let site = base + u64::from(*rva);
        let at = pool as u64 + used as u64;
        let (Some(code), Some(jump)) = (trampoline(store, site, at), jump(site, store.len, at))
        else {
            tracing::warn!(rva = format_args!("{rva:#x}"), "masked store out of reach");
            continue;
        };
        if used + code.len() > SIZE {
            break;
        }
        let mut old = 0;
        // SAFETY: our pool has room for the code; the site is the title's
        // code, writable for the moment of the write, before any title
        // thread runs.
        unsafe {
            std::ptr::copy_nonoverlapping(code.as_ptr(), at as *mut u8, code.len());
            if VirtualProtect(
                site as *const _,
                jump.len(),
                PAGE_EXECUTE_READWRITE,
                &mut old,
            ) == 0
            {
                continue;
            }
            std::ptr::copy_nonoverlapping(jump.as_ptr(), site as *mut u8, jump.len());
            VirtualProtect(site as *const _, jump.len(), old, &mut old);
        }
        used = (used + code.len() + 15) & !15;
        patched += 1;
    }
    let mut old = 0;
    // SAFETY: our pool; the whole image range for the cache flush.
    unsafe {
        VirtualProtect(pool, SIZE, PAGE_EXECUTE_READ, &mut old);
        FlushInstructionCache(GetCurrentProcess(), std::ptr::null(), 0);
    }
    patched
}

/// What to patch, from `VICTORY_PATCH_MASKED`: its value (`all` or hex
/// RVAs, for [`wanted_from`]), nothing for `none`, and `all` when it is
/// unset and `needed` (fault tracking under Rosetta).
pub fn choice(value: Option<String>, needed: bool) -> Option<String> {
    match value.as_deref() {
        Some("none") => None,
        Some(_) => value,
        None => needed.then(|| "all".to_owned()),
    }
}

/// The stores `VICTORY_PATCH_MASKED` asks to patch.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct Wanted {
    /// Every masked store the scan finds.
    pub all: bool,
    /// Stores named by RVA: masked ones, or plain 256-bit ones the scan
    /// does not look for (#144).
    pub rvas: Vec<u32>,
}

impl Wanted {
    /// Whether the masked store at `rva` is wanted.
    pub fn picks(&self, rva: u32) -> bool {
        self.all || self.rvas.contains(&rva)
    }
}

/// What `VICTORY_PATCH_MASKED` asks for: a comma list of `all` and hex
/// RVAs, such as `all,1e09b31`. Nothing when unset.
pub fn wanted_from(value: Option<&str>) -> Wanted {
    let mut wanted = Wanted::default();
    for item in value.unwrap_or_default().split(',').map(str::trim) {
        if item == "all" {
            wanted.all = true;
        } else if let Ok(rva) = u32::from_str_radix(item.trim_start_matches("0x"), 16) {
            wanted.rvas.push(rva);
        }
    }
    wanted
}

#[cfg(test)]
mod patch_tests {
    use super::*;

    #[test]
    fn stores_decode_with_their_operands_and_length() {
        // The store from the Rosetta hang (#1): VEX.vvvv 1000 is mask ymm7,
        // ModRM.reg 6 the stored ymm6.
        let s = decode_store(&[0xc4, 0xe2, 0x45, 0x8e, 0x30], 0).unwrap();
        assert_eq!(
            (s.len, s.mask, s.src, s.ymm, s.wide_lanes),
            (5, 7, 6, true, false)
        );
        // vpmaskmovd [rsp+0x10], ymm1, ymm2: SIB and disp8.
        let s = decode_store(&[0xc4, 0xe2, 0x75, 0x8e, 0x54, 0x24, 0x10], 0).unwrap();
        assert_eq!((s.len, s.disp, s.rsp_base()), (7, 0x10, true));
        // vmaskmovpd [rip+0x100], xmm1, xmm2: RIP-relative disp32.
        let s = decode_store(&[0xc4, 0xe2, 0x71, 0x2f, 0x15, 0, 1, 0, 0], 0).unwrap();
        assert_eq!(
            (s.len, s.disp, s.rip_relative(), s.lanes()),
            (9, 0x100, true, 2)
        );
    }

    #[test]
    fn the_store_from_the_second_hang_decodes() {
        // vcvtps2ph [r8+rdx*2], ymm0, 3 from run 11 in #50.
        let s = decode_store(&[0xc4, 0xc3, 0x7d, 0x1d, 0x04, 0x50, 0x03], 0).unwrap();
        assert_eq!(
            (s.len, s.b, s.x, s.sib, s.plain.map(|p| (p.3, p.4))),
            (7, true, false, Some(0x50), Some((Some(3), 16)))
        );
    }

    #[test]
    fn the_plain_store_from_144_decodes() {
        // vmovups [r8+0x34], ymm0 at 0x1E09B31 in #144.
        let s = decode_wide_move(&[0xc4, 0xc1, 0x7c, 0x11, 0x40, 0x34], 0).unwrap();
        assert_eq!(
            (s.len, s.b, s.x, s.src, s.disp, s.plain),
            (6, true, false, 0, 0x34, Some((0xc1, 0x7c, 0x11, None, 32)))
        );
        // vmovdqu [rcx+8], ymm9 with the two-byte prefix: R comes along.
        let s = decode_wide_move(&[0xc5, 0x7e, 0x7f, 0x49, 0x08], 0).unwrap();
        assert_eq!(
            (s.len, s.b, s.src, s.plain),
            (5, false, 9, Some((0x61, 0x7e, 0x7f, None, 32)))
        );
        // vmovupd [rsp+0x20], ymm1: the SIB byte.
        let s = decode_wide_move(&[0xc5, 0xfd, 0x11, 0x4c, 0x24, 0x20], 0).unwrap();
        assert_eq!((s.len, s.disp, s.rsp_base()), (6, 0x20, true));
    }

    #[test]
    fn only_unaligned_256_bit_stores_are_plain_wide_moves() {
        let no = |bytes: &[u8]| assert_eq!(decode_wide_move(bytes, 0), None, "{bytes:02x?}");
        // 128-bit: one store Rosetta can resume.
        no(&[0xc4, 0xc1, 0x78, 0x11, 0x40, 0x34]);
        // Aligned: vmovaps (29), vmovdqa (66 7f); 32 aligned bytes never
        // cross a page.
        no(&[0xc4, 0xc1, 0x7c, 0x29, 0x40, 0x34]);
        no(&[0xc4, 0xc1, 0x7d, 0x7f, 0x40, 0x34]);
        // Loads (10, F3 6f) and a register operand.
        no(&[0xc4, 0xc1, 0x7c, 0x10, 0x40, 0x34]);
        no(&[0xc4, 0xc1, 0x7e, 0x6f, 0x40, 0x34]);
        no(&[0xc4, 0xc1, 0x7c, 0x11, 0xc0, 0x90]);
        // vvvv not 1111, or not the 0F map.
        no(&[0xc4, 0xc1, 0x74, 0x11, 0x40, 0x34]);
        no(&[0xc4, 0xc2, 0x7c, 0x11, 0x40, 0x34]);
        // vmovups [rax], ymm0: 4 bytes, no room for the jump.
        no(&[0xc5, 0xfc, 0x11, 0x00]);
        // Cut short.
        no(&[0xc4, 0xc1, 0x7c, 0x11, 0x40]);
    }

    #[test]
    fn the_jump_pads_to_the_store_length() {
        assert_eq!(
            jump(0x1000, 7, 0x2000).unwrap(),
            vec![0xe9, 0xfb, 0x0f, 0, 0, 0x90, 0x90]
        );
        assert!(jump(0, 5, 1 << 40).is_none());
    }

    #[test]
    fn the_list_picks_rvas() {
        let w = wanted_from(Some("1dac734,0x10"));
        assert!(w.picks(0x1dac734) && w.picks(0x10) && !w.picks(0x11));
        assert!(wanted_from(Some("all")).picks(5));
        assert!(!wanted_from(None).picks(5));
        assert_eq!(wanted_from(Some("")), Wanted::default());
        // `all` and RVAs together.
        assert_eq!(
            wanted_from(Some("all, 1e09b31,1E09B37")),
            Wanted {
                all: true,
                rvas: vec![0x1e09b31, 0x1e09b37]
            }
        );
    }

    #[test]
    fn stores_are_patched_by_default_only_when_needed() {
        let s = |v: &str| Some(v.to_owned());
        assert_eq!(choice(None, true), s("all"));
        assert_eq!(choice(None, false), None);
        assert_eq!(choice(s("none"), true), None);
        assert_eq!(choice(s("1e09b31"), true), s("1e09b31"));
        assert_eq!(choice(s("all"), false), s("all"));
    }

    /// Real code with a masked store, run as written and then patched:
    /// both write the same lanes. Needs a CPU (or Rosetta) with AVX2.
    #[cfg(all(windows, target_arch = "x86_64"))]
    #[test]
    fn a_patched_store_writes_what_the_store_did() {
        use windows_sys::Win32::System::Memory::*;
        // f(dst rcx, mask rdx, src r8): vmovdqu ymm1, [rdx]; vmovdqu ymm2,
        // [r8]; <store>; vzeroupper; ret.
        let cases: [&[u8]; 10] = [
            &[0xc4, 0xe2, 0x75, 0x8e, 0x11],       // vpmaskmovd [rcx], ymm1, ymm2
            &[0xc4, 0xe2, 0xf5, 0x8e, 0x51, 0x08], // vpmaskmovq [rcx+8], ymm1, ymm2
            &[0xc4, 0xe2, 0x71, 0x2e, 0x54, 0x01, 0x04], // vmaskmovps [rcx+rax+4], xmm1, xmm2 (rax 0)
            &[0xc4, 0xe3, 0x7d, 0x1d, 0x09, 0x00],       // vcvtps2ph [rcx], ymm1, 0
            &[0xc4, 0xe3, 0x79, 0x1d, 0x49, 0x04, 0x03], // vcvtps2ph [rcx+4], xmm1, 3
            &[0xc4, 0xe3, 0x7d, 0x19, 0x49, 0x10, 0x01], // vextractf128 [rcx+16], ymm1, 1
            &[0xc4, 0xe3, 0x7d, 0x39, 0x09, 0x01],       // vextracti128 [rcx], ymm1, 1
            &[0xc4, 0xe1, 0x7c, 0x11, 0x51, 0x04],       // vmovups [rcx+4], ymm2
            &[0xc4, 0xe1, 0x7d, 0x11, 0x11],             // vmovupd [rcx], ymm2
            &[0xc5, 0xfe, 0x7f, 0x49, 0x0c],             // vmovdqu [rcx+12], ymm1
        ];
        for (k, store) in cases.into_iter().enumerate() {
            let mut code = vec![
                0x31, 0xc0, 0xc5, 0xfe, 0x6f, 0x0a, 0xc4, 0xc1, 0x7e, 0x6f, 0x10,
            ];
            let site_at = code.len();
            code.extend_from_slice(store);
            code.extend_from_slice(&[0xc5, 0xf8, 0x77, 0xc3]);
            let run = |patch: bool| -> [u32; 12] {
                // SAFETY: our own executable page, written then run.
                unsafe {
                    let page = VirtualAlloc(
                        std::ptr::null(),
                        1 << 16,
                        MEM_RESERVE | MEM_COMMIT,
                        PAGE_EXECUTE_READWRITE,
                    ) as *mut u8;
                    std::ptr::copy_nonoverlapping(code.as_ptr(), page, code.len());
                    if patch {
                        let s = decode_store(&code, site_at)
                            .or_else(|| decode_wide_move(&code, site_at))
                            .unwrap();
                        let site = page as u64 + site_at as u64;
                        let at = page as u64 + 0x1000;
                        let t = trampoline(&s, site, at).unwrap();
                        std::ptr::copy_nonoverlapping(t.as_ptr(), at as *mut u8, t.len());
                        let j = jump(site, s.len, at).unwrap();
                        std::ptr::copy_nonoverlapping(j.as_ptr(), site as *mut u8, j.len());
                    }
                    let f: extern "win64" fn(*mut u32, *const u32, *const u32) =
                        std::mem::transmute(page);
                    let mask: [u32; 8] = [
                        0x8000_0000,
                        0,
                        0xffff_ffff,
                        0,
                        0,
                        0x8000_0000,
                        0x8000_0000,
                        1,
                    ];
                    let src: [u32; 8] = [11, 22, 33, 44, 55, 66, 77, 88];
                    let mut dst = [7u32; 12];
                    f(dst.as_mut_ptr(), mask.as_ptr(), src.as_ptr());
                    VirtualFree(page.cast(), 0, MEM_RELEASE);
                    dst
                }
            };
            let (plain, patched) = (run(false), run(true));
            assert_eq!(plain, patched, "case {k}");
            assert_ne!(plain, [7u32; 12], "case {k} wrote something");
        }
    }

    /// The point of the patch: a patched store into a read-only page
    /// faults on a plain store, a handler makes the page writable, and the
    /// store resumes. The original masked store would hang Rosetta here
    /// (#14), so it is never run this way.
    #[cfg(all(windows, target_arch = "x86_64"))]
    #[test]
    fn a_patched_store_resumes_after_a_write_fault() {
        use windows_sys::Win32::Foundation::EXCEPTION_ACCESS_VIOLATION;
        use windows_sys::Win32::System::Diagnostics::Debug::*;
        use windows_sys::Win32::System::Memory::*;
        static PAGE: std::sync::atomic::AtomicUsize = std::sync::atomic::AtomicUsize::new(0);
        unsafe extern "system" fn unguard(info: *mut EXCEPTION_POINTERS) -> i32 {
            // SAFETY: the system passes valid pointers.
            unsafe {
                let r = &*(*info).ExceptionRecord;
                let page = PAGE.load(std::sync::atomic::Ordering::Relaxed);
                if r.ExceptionCode == EXCEPTION_ACCESS_VIOLATION
                    && (r.ExceptionInformation[1] & !0xfff) == page
                {
                    let mut old = 0;
                    VirtualProtect(page as *const _, 0x1000, PAGE_READWRITE, &mut old);
                    return -1; // EXCEPTION_CONTINUE_EXECUTION
                }
            }
            0
        }
        let store = [0xc4, 0xe2, 0x75, 0x8e, 0x11]; // vpmaskmovd [rcx], ymm1, ymm2
        let mut code = vec![0xc5, 0xfe, 0x6f, 0x0a, 0xc4, 0xc1, 0x7e, 0x6f, 0x10];
        let site_at = code.len();
        code.extend_from_slice(&store);
        code.extend_from_slice(&[0xc5, 0xf8, 0x77, 0xc3]);
        // SAFETY: our own pages and handler.
        unsafe {
            let exec = VirtualAlloc(
                std::ptr::null(),
                1 << 16,
                MEM_RESERVE | MEM_COMMIT,
                PAGE_EXECUTE_READWRITE,
            ) as *mut u8;
            std::ptr::copy_nonoverlapping(code.as_ptr(), exec, code.len());
            let s = decode_store(&code, site_at).unwrap();
            let site = exec as u64 + site_at as u64;
            let at = exec as u64 + 0x1000;
            let t = trampoline(&s, site, at).unwrap();
            std::ptr::copy_nonoverlapping(t.as_ptr(), at as *mut u8, t.len());
            let j = jump(site, s.len, at).unwrap();
            std::ptr::copy_nonoverlapping(j.as_ptr(), site as *mut u8, j.len());
            let data = VirtualAlloc(
                std::ptr::null(),
                0x1000,
                MEM_RESERVE | MEM_COMMIT,
                PAGE_READONLY,
            ) as *mut u32;
            PAGE.store(data as usize, std::sync::atomic::Ordering::Relaxed);
            let handler = AddVectoredExceptionHandler(1, Some(unguard));
            let f: extern "win64" fn(*mut u32, *const u32, *const u32) = std::mem::transmute(exec);
            let mask: [u32; 8] = [0, 0x8000_0000, 0, 0, 0, 0, 0, 0x8000_0000];
            let src: [u32; 8] = [1, 2, 3, 4, 5, 6, 7, 8];
            f(data, mask.as_ptr(), src.as_ptr());
            RemoveVectoredExceptionHandler(handler);
            let got = std::slice::from_raw_parts(data, 8).to_vec();
            assert_eq!(got, vec![0, 2, 0, 0, 0, 0, 0, 8]);
            VirtualFree(data.cast(), 0, MEM_RELEASE);
            VirtualFree(exec.cast(), 0, MEM_RELEASE);
        }
    }

    /// A patched `vmovups` like the one in #144, 32 bytes that start in a
    /// writable page and end in a read-only one. A plain `mov` faults
    /// instead, a handler makes the page writable, and the store resumes
    /// with every byte written. One of the `mov`s itself crosses the page
    /// boundary.
    #[cfg(all(windows, target_arch = "x86_64"))]
    #[test]
    fn a_patched_wide_store_resumes_across_a_page() {
        use windows_sys::Win32::Foundation::EXCEPTION_ACCESS_VIOLATION;
        use windows_sys::Win32::System::Diagnostics::Debug::*;
        use windows_sys::Win32::System::Memory::*;
        static PAGE: std::sync::atomic::AtomicUsize = std::sync::atomic::AtomicUsize::new(0);
        unsafe extern "system" fn unguard(info: *mut EXCEPTION_POINTERS) -> i32 {
            // SAFETY: the system passes valid pointers.
            unsafe {
                let r = &*(*info).ExceptionRecord;
                let page = PAGE.load(std::sync::atomic::Ordering::Relaxed);
                // A write that crosses into the page may report its start,
                // in the page before.
                let at = r.ExceptionInformation[1];
                if r.ExceptionCode == EXCEPTION_ACCESS_VIOLATION
                    && at.wrapping_sub(page - 0x1000) < 0x2000
                {
                    let mut old = 0;
                    VirtualProtect(page as *const _, 0x1000, PAGE_READWRITE, &mut old);
                    return -1; // EXCEPTION_CONTINUE_EXECUTION
                }
            }
            0
        }
        // f(dst rcx, _, src r8): vmovdqu ymm2, [r8]; vmovups [rcx+4], ymm2;
        // vzeroupper; ret.
        let mut code = vec![0xc4, 0xc1, 0x7e, 0x6f, 0x10];
        let site_at = code.len();
        code.extend_from_slice(&[0xc4, 0xe1, 0x7c, 0x11, 0x51, 0x04]);
        code.extend_from_slice(&[0xc5, 0xf8, 0x77, 0xc3]);
        // SAFETY: our own pages and handler.
        unsafe {
            let exec = VirtualAlloc(
                std::ptr::null(),
                1 << 16,
                MEM_RESERVE | MEM_COMMIT,
                PAGE_EXECUTE_READWRITE,
            ) as *mut u8;
            std::ptr::copy_nonoverlapping(code.as_ptr(), exec, code.len());
            let s = decode_wide_move(&code, site_at).unwrap();
            let site = exec as u64 + site_at as u64;
            let at = exec as u64 + 0x1000;
            let t = trampoline(&s, site, at).unwrap();
            std::ptr::copy_nonoverlapping(t.as_ptr(), at as *mut u8, t.len());
            let j = jump(site, s.len, at).unwrap();
            std::ptr::copy_nonoverlapping(j.as_ptr(), site as *mut u8, j.len());
            let data = VirtualAlloc(
                std::ptr::null(),
                0x2000,
                MEM_RESERVE | MEM_COMMIT,
                PAGE_READWRITE,
            ) as *mut u8;
            let guarded = data.add(0x1000);
            let mut old = 0;
            VirtualProtect(guarded.cast(), 0x1000, PAGE_READONLY, &mut old);
            PAGE.store(guarded as usize, std::sync::atomic::Ordering::Relaxed);
            let handler = AddVectoredExceptionHandler(1, Some(unguard));
            let f: extern "win64" fn(*mut u8, *const u32, *const u32) = std::mem::transmute(exec);
            // The store covers 12 bytes before the boundary and 20 after.
            let dst = guarded.sub(16);
            let src: [u32; 8] = [1, 2, 3, 4, 5, 6, 7, 8];
            f(dst, std::ptr::null(), src.as_ptr());
            RemoveVectoredExceptionHandler(handler);
            let got = std::slice::from_raw_parts(dst.add(4).cast::<u32>(), 8).to_vec();
            assert_eq!(got, src.to_vec());
            // The handler ran: the page is writable now.
            VirtualProtect(guarded.cast(), 0x1000, PAGE_READWRITE, &mut old);
            assert_eq!(old, PAGE_READWRITE);
            VirtualFree(data.cast(), 0, MEM_RELEASE);
            VirtualFree(exec.cast(), 0, MEM_RELEASE);
        }
    }
}

/// Patched stores against the stores themselves, in every form the patch
/// takes (#184: a store the patch redid wrongly would lose or misplace a
/// write of the title's).
#[cfg(all(test, windows, target_arch = "x86_64"))]
mod equivalence_tests {
    use super::*;
    use std::mem::offset_of;
    use windows_sys::Win32::System::Memory::*;

    /// Registers before and after one store.
    #[repr(C, align(32))]
    struct Ctx {
        gpr: [u64; 16],
        flags: u64,
        _pad: [u64; 3],
        ymm: [[u8; 32]; 16],
        stack: [u8; 128],
        out_gpr: [u64; 16],
        out_flags: u64,
        _pad2: [u64; 3],
        out_ymm: [[u8; 32]; 16],
        out_stack: [u8; 128],
    }

    /// The arithmetic flags and bit 1: no trap, direction or alignment
    /// check flags.
    const FLAGS: u64 = 0x8d5;
    /// Where the harness keeps the context pointer, the trampoline goes
    /// and RIP-relative stores land, in the code allocation.
    const SLOT: usize = 0x3000;
    const TRAMPOLINE: usize = 0x1000;
    const RIP_AREA: usize = 0x2000;
    /// The harness's stack frame: a 128-byte area stores relative to rsp
    /// land in, then xmm6 to xmm15.
    const FRAME: u32 = 0x128;

    struct Rng(u64);
    impl Rng {
        fn next(&mut self) -> u64 {
            self.0 ^= self.0 << 13;
            self.0 ^= self.0 >> 7;
            self.0 ^= self.0 << 17;
            self.0
        }
        fn below(&mut self, n: u64) -> u64 {
            self.next() % n
        }
        fn coin(&mut self) -> bool {
            self.next() & 1 != 0
        }
    }

    /// A memory operand, as the generator places it.
    #[derive(Debug, Clone, Copy)]
    enum Mem {
        Rip(i32),
        /// `[index * scale + disp32]` with no base.
        Absolute {
            index: Option<(u8, u8)>,
            disp: i32,
        },
        /// `[base + index * scale + disp]`; `wide_disp` asks for a disp32.
        Based {
            base: u8,
            index: Option<(u8, u8)>,
            disp: i32,
            wide_disp: bool,
        },
    }

    /// ModRM (with `reg`), SIB, displacement and the REX X and B bits.
    fn modrm_bytes(reg: u8, m: Mem) -> (Vec<u8>, bool, bool) {
        let r = (reg & 7) << 3;
        let sib = |index: Option<(u8, u8)>, base: u8| {
            let (i, s) = index.map_or((4, 0), |(i, s)| (i & 7, s));
            s << 6 | i << 3 | base
        };
        let x = |index: Option<(u8, u8)>| index.is_some_and(|(i, _)| i >= 8);
        match m {
            Mem::Rip(disp) => {
                let mut v = vec![r | 5];
                v.extend_from_slice(&disp.to_le_bytes());
                (v, false, false)
            }
            Mem::Absolute { index, disp } => {
                let mut v = vec![r | 4, sib(index, 5)];
                v.extend_from_slice(&disp.to_le_bytes());
                (v, x(index), false)
            }
            Mem::Based {
                base,
                index,
                disp,
                wide_disp,
            } => {
                let needs_sib = base & 7 == 4 || index.is_some();
                let md = if disp == 0 && base & 7 != 5 && !wide_disp {
                    0
                } else if i8::try_from(disp).is_ok() && !wide_disp {
                    1
                } else {
                    2
                };
                let mut v = vec![md << 6 | r | if needs_sib { 4 } else { base & 7 }];
                if needs_sib {
                    v.push(sib(index, base & 7));
                }
                match md {
                    1 => v.push(disp as i8 as u8),
                    2 => v.extend_from_slice(&disp.to_le_bytes()),
                    _ => {}
                }
                (v, x(index), base >= 8)
            }
        }
    }

    /// A store to generate: its kind, its registers and its operand.
    #[derive(Debug, Clone, Copy, PartialEq)]
    enum Kind {
        Masked { opcode: u8, w: bool },
        Cvt,
        Extract(u8),
        Move { opcode: u8, pp: u8, two_byte: bool },
    }

    /// The instruction's bytes: `kind` storing `src` (masked by `mask`) to `m`.
    fn encode(kind: Kind, ymm: bool, mask: u8, src: u8, m: Mem, imm: u8) -> Vec<u8> {
        let (body, x, b) = modrm_bytes(src, m);
        let rbar = if src & 8 == 0 { 0x80 } else { 0 };
        let xb = if x { 0 } else { 0x40 } | if b { 0 } else { 0x20 };
        let l = if ymm { 4 } else { 0 };
        let mut v = match kind {
            Kind::Masked { opcode, w } => vec![
                0xc4,
                rbar | xb | 2,
                if w { 0x80 } else { 0 } | (!mask & 0xf) << 3 | l | 1,
                opcode,
            ],
            Kind::Cvt => vec![0xc4, rbar | xb | 3, 0x78 | l | 1, 0x1d],
            Kind::Extract(opcode) => vec![0xc4, rbar | xb | 3, 0x78 | 4 | 1, opcode],
            Kind::Move {
                opcode,
                pp,
                two_byte: true,
            } => vec![0xc5, rbar | 0x78 | 4 | pp, opcode],
            Kind::Move {
                opcode,
                pp,
                two_byte: false,
            } => {
                vec![0xc4, rbar | xb | 1, 0x78 | 4 | pp, opcode]
            }
        };
        v.extend(body);
        if matches!(kind, Kind::Cvt | Kind::Extract(_)) {
            v.push(imm);
        }
        v
    }

    /// A little assembler for the harness: `mov`s between registers and
    /// `[base + disp32]`, and vector moves.
    struct Asm(Vec<u8>);
    impl Asm {
        /// REX.W `op` /r with `reg` and `[base + disp32]` (base not rsp or r12).
        fn rm64(&mut self, op: u8, reg: u8, base: u8, disp: usize) {
            self.0.push(0x48 | (reg >> 3) << 2 | (base >> 3));
            self.0.push(op);
            self.0.push(0x80 | (reg & 7) << 3 | (base & 7));
            self.0.extend_from_slice(&(disp as u32).to_le_bytes());
        }
        /// `vmovdqu` (`op` 6f load, 7f store) of `ymm reg` with
        /// `[base + disp32]`; base 4 is rsp.
        fn vmov(&mut self, op: u8, reg: u8, base: u8, disp: usize) {
            let rbar = if reg & 8 == 0 { 0x80 } else { 0 };
            self.0.extend_from_slice(&[0xc4, rbar | 0x60 | 1, 0x7e, op]);
            self.0.push(0x80 | (reg & 7) << 3 | (base & 7));
            if base == 4 {
                self.0.push(0x24);
            }
            self.0.extend_from_slice(&(disp as u32).to_le_bytes());
        }
        /// `movdqu` (`op` 6f load, 7f store) of `xmm reg` with `[rsp + disp32]`.
        fn xmm_rsp(&mut self, op: u8, reg: u8, disp: u32) {
            self.0.push(0xf3);
            if reg >= 8 {
                self.0.push(0x44);
            }
            self.0
                .extend_from_slice(&[0x0f, op, 0x84 | (reg & 7) << 3, 0x24]);
            self.0.extend_from_slice(&disp.to_le_bytes());
        }
        /// `mov [rip + to - next], reg` (89) or `mov reg, [rip + ...]` (8b)
        /// for code placed at offset 0 of the allocation.
        fn rip64(&mut self, op: u8, reg: u8, to: usize) {
            self.0
                .extend_from_slice(&[0x48 | (reg >> 3) << 2, op, (reg & 7) << 3 | 5]);
            let next = self.0.len() + 4;
            self.0
                .extend_from_slice(&((to as i64 - next as i64) as i32).to_le_bytes());
        }
    }

    /// The harness: loads `Ctx` (pointer in rcx), runs `store`, saves the
    /// registers back, keeping what the Windows ABI asks to keep. Returns
    /// the code and where the store starts in it.
    fn harness(store: &[u8]) -> (Vec<u8>, usize) {
        let mut a = Asm(Vec::new());
        // push rbx, rbp, rsi, rdi, r12..r15
        a.0.extend_from_slice(&[
            0x53, 0x55, 0x56, 0x57, 0x41, 0x54, 0x41, 0x55, 0x41, 0x56, 0x41, 0x57,
        ]);
        a.0.extend_from_slice(&[0x48, 0x81, 0xec]); // sub rsp, FRAME
        a.0.extend_from_slice(&FRAME.to_le_bytes());
        for i in 0..10 {
            a.xmm_rsp(0x7f, 6 + i, 0x80 + 16 * u32::from(i));
        }
        a.rip64(0x89, 1, SLOT); // mov [rip + slot], rcx
        for k in 0..4 {
            a.vmov(0x6f, 0, 1, offset_of!(Ctx, stack) + 32 * k);
            a.vmov(0x7f, 0, 4, 32 * k);
        }
        for i in 0..16 {
            a.vmov(0x6f, i, 1, offset_of!(Ctx, ymm) + 32 * usize::from(i));
        }
        // push qword [rcx + flags]; popfq
        a.0.extend_from_slice(&[0xff, 0xb1]);
        a.0.extend_from_slice(&(offset_of!(Ctx, flags) as u32).to_le_bytes());
        a.0.push(0x9d);
        for r in (0..16u8).filter(|&r| r != 1 && r != 4) {
            a.rm64(0x8b, r, 1, offset_of!(Ctx, gpr) + 8 * usize::from(r));
        }
        a.rm64(0x8b, 1, 1, offset_of!(Ctx, gpr) + 8);
        let site = a.0.len();
        a.0.extend_from_slice(store);
        a.0.extend_from_slice(&[0x9c, 0x50]); // pushfq; push rax
        a.rip64(0x8b, 0, SLOT); // mov rax, [rip + slot]
        for r in (1..16u8).filter(|&r| r != 4) {
            a.rm64(0x89, r, 0, offset_of!(Ctx, out_gpr) + 8 * usize::from(r));
        }
        a.0.push(0x59); // pop rcx: rax as the store left it
        a.rm64(0x89, 1, 0, offset_of!(Ctx, out_gpr));
        a.0.push(0x59); // pop rcx: the flags
        a.rm64(0x89, 1, 0, offset_of!(Ctx, out_flags));
        for i in 0..16 {
            a.vmov(0x7f, i, 0, offset_of!(Ctx, out_ymm) + 32 * usize::from(i));
        }
        for k in 0..4 {
            a.vmov(0x6f, 0, 4, 32 * k);
            a.vmov(0x7f, 0, 0, offset_of!(Ctx, out_stack) + 32 * k);
        }
        for i in 0..10 {
            a.xmm_rsp(0x6f, 6 + i, 0x80 + 16 * u32::from(i));
        }
        a.0.extend_from_slice(&[0x48, 0x81, 0xc4]); // add rsp, FRAME
        a.0.extend_from_slice(&FRAME.to_le_bytes());
        a.0.extend_from_slice(&[
            0x41, 0x5f, 0x41, 0x5e, 0x41, 0x5d, 0x41, 0x5c, 0x5f, 0x5e, 0x5d, 0x5b,
        ]);
        a.0.extend_from_slice(&[0xfc, 0xc5, 0xf8, 0x77, 0xc3]); // cld; vzeroupper; ret
        (a.0, site)
    }

    /// What one run leaves: registers, flags, vector registers, the stack
    /// area, the data buffer, the RIP area and the low buffer.
    #[derive(Debug, PartialEq, Eq)]
    struct After {
        gpr: Vec<u64>,
        flags: u64,
        ymm: Vec<[u8; 32]>,
        stack: [u8; 128],
        data: Vec<u8>,
        rip: Vec<u8>,
        low: Vec<u8>,
    }

    /// A 4 KiB buffer below 2 GB for absolute operands, if one can be had.
    fn low_buffer() -> Option<*mut u8> {
        (0x1000u64..0x7000).map(|k| k << 16).find_map(|at| {
            // SAFETY: a plain allocation at a hint; null when refused.
            let p = unsafe {
                VirtualAlloc(
                    at as *const _,
                    0x1000,
                    MEM_RESERVE | MEM_COMMIT,
                    PAGE_READWRITE,
                )
            };
            (!p.is_null() && (p as u64) < 0x7000_0000).then_some(p.cast())
        })
    }

    /// Every masked and plain store the patch redoes, in every operand form,
    /// with random registers, masks and data: the patched store leaves
    /// memory, registers and flags as the store itself does, the masked
    /// ones and the moves write exactly what their definition says, and
    /// each decodes to its own length.
    #[test]
    fn patched_stores_match_the_stores_in_every_form() {
        const CASES: u32 = 4000;
        let mut rng = Rng(0x9e37_79b9_7f4a_7c15);
        // SAFETY: our own allocations, freed at the end.
        let page = unsafe {
            VirtualAlloc(
                std::ptr::null(),
                0x4000,
                MEM_RESERVE | MEM_COMMIT,
                PAGE_EXECUTE_READWRITE,
            )
        } as *mut u8;
        assert!(!page.is_null());
        let mut data = vec![0u8; 4096];
        let low = low_buffer();
        let mut ran = [0u32; 4]; // masked, cvt, extract, move
        let mut forms = std::collections::BTreeSet::new();
        for case in 0..CASES {
            let kind = match rng.below(9) {
                0 => Kind::Masked {
                    opcode: 0x8e,
                    w: false,
                },
                1 => Kind::Masked {
                    opcode: 0x8e,
                    w: true,
                },
                2 => Kind::Masked {
                    opcode: 0x2e,
                    w: false,
                },
                3 => Kind::Masked {
                    opcode: 0x2f,
                    w: false,
                },
                4 => Kind::Cvt,
                5 => Kind::Extract(if rng.coin() { 0x19 } else { 0x39 }),
                n => {
                    let (opcode, pp) = [(0x11, 0), (0x11, 1), (0x7f, 2)][rng.below(3) as usize];
                    Kind::Move {
                        opcode,
                        pp,
                        two_byte: n == 8,
                    }
                }
            };
            let ymm = matches!(kind, Kind::Extract(_) | Kind::Move { .. }) || rng.coin();
            let (mask, src) = (rng.below(16) as u8, rng.below(16) as u8);
            let imm = rng.below(256) as u8;
            // SAFETY: every field is an integer or bytes, valid as zeros.
            let mut ctx: Box<Ctx> = Box::new(unsafe { std::mem::zeroed() });
            for g in ctx.gpr.iter_mut() {
                *g = rng.next();
            }
            ctx.flags = rng.next() & FLAGS | 2;
            for y in ctx.ymm.iter_mut() {
                y.iter_mut().for_each(|b| *b = rng.next() as u8);
            }
            ctx.stack.iter_mut().for_each(|b| *b = rng.next() as u8);
            // The operand: a target 64 bytes from the edges of its buffer.
            let target = 64 + rng.below(4096 - 160) as i64;
            let scale = rng.below(4) as u8;
            let pick_index = |rng: &mut Rng, base: Option<u8>| loop {
                let i = rng.below(16) as u8;
                if i != 4 && Some(i) != base {
                    break i;
                }
            };
            let two_byte = matches!(kind, Kind::Move { two_byte: true, .. });
            let form = rng.below(if two_byte { 2 } else { 6 });
            let (m, desc) = match form {
                // A base, maybe an index (a low register for the two-byte
                // prefix, which has no X or B).
                0 | 1 => {
                    let base = loop {
                        let b = rng.below(16) as u8;
                        if b != 4 && (!two_byte || b < 8) {
                            break b;
                        }
                    };
                    let index = (form == 1).then(|| {
                        loop {
                            let i = pick_index(&mut rng, Some(base));
                            if !two_byte || i < 8 {
                                break (i, scale);
                            }
                        }
                    });
                    let disp = match rng.below(3) {
                        0 => 0,
                        1 => rng.below(256) as i32 - 128,
                        _ => rng.below(1 << 20) as i32 - (1 << 19),
                    };
                    let iv = rng.below(64);
                    if let Some((i, s)) = index {
                        ctx.gpr[usize::from(i)] = iv;
                        let ea = data.as_ptr() as i64 + target;
                        ctx.gpr[usize::from(base)] =
                            (ea - (iv << s) as i64 - i64::from(disp)) as u64;
                    } else {
                        ctx.gpr[usize::from(base)] =
                            (data.as_ptr() as i64 + target - i64::from(disp)) as u64;
                    }
                    let wide_disp = rng.coin();
                    (
                        Mem::Based {
                            base,
                            index,
                            disp,
                            wide_disp,
                        },
                        format!("base {base} index {index:?}"),
                    )
                }
                // The same register as base and index.
                2 => {
                    let r = loop {
                        let r = rng.below(16) as u8;
                        if r != 4 {
                            break r;
                        }
                    };
                    let k = 1 + (1i64 << scale);
                    let v = (data.as_ptr() as i64 + target) / k;
                    let disp = (data.as_ptr() as i64 + target - v * k) as i32;
                    ctx.gpr[usize::from(r)] = v as u64;
                    (
                        Mem::Based {
                            base: r,
                            index: Some((r, scale)),
                            disp,
                            wide_disp: false,
                        },
                        format!("base and index {r}"),
                    )
                }
                // rsp as the base: into the harness's stack area.
                3 => {
                    let index = rng.coin().then(|| (pick_index(&mut rng, Some(4)), scale));
                    let at = rng.below(128 - 32) as i64;
                    let iv = rng.below(8) as i64;
                    let disp = match index {
                        Some((i, s)) => {
                            ctx.gpr[usize::from(i)] = iv as u64;
                            (at - (iv << s)) as i32
                        }
                        None => at as i32,
                    };
                    (
                        Mem::Based {
                            base: 4,
                            index,
                            disp,
                            wide_disp: rng.coin(),
                        },
                        "rsp".into(),
                    )
                }
                // RIP-relative, into the RIP area of the code allocation.
                4 => (Mem::Rip(0), "rip".into()),
                // No base: an absolute address in the low buffer.
                _ => match low {
                    Some(low) => {
                        let index = rng.coin().then(|| (pick_index(&mut rng, None), scale));
                        let iv = rng.below(16) as i64;
                        if let Some((i, _)) = index {
                            ctx.gpr[usize::from(i)] = iv as u64;
                        }
                        let shift = index.map_or(0, |(_, s)| iv << s);
                        let disp = (low as i64 + target - shift) as i32;
                        (Mem::Absolute { index, disp }, "absolute".into())
                    }
                    None => continue,
                },
            };
            // RIP-relative: the displacement depends on where the store ends.
            let mut store = encode(kind, ymm, mask, src, m, imm);
            let (mut code, site) = harness(&store);
            if let Mem::Rip(_) = m {
                let next = site + store.len();
                let disp = (RIP_AREA as i64 + target - next as i64) as i32;
                store = encode(kind, ymm, mask, src, Mem::Rip(disp), imm);
                code[site..site + store.len()].copy_from_slice(&store);
            }
            forms.insert(desc);
            let decoded = match kind {
                Kind::Move { .. } => decode_wide_move(&code, site),
                _ => decode_store(&code, site),
            };
            let Some(s) = decoded else {
                // Only plain moves too short for the jump may be left alone.
                assert!(
                    matches!(kind, Kind::Move { .. }) && store.len() < 5,
                    "case {case}: {store:02x?} ({kind:?}) did not decode"
                );
                continue;
            };
            assert_eq!(
                s.len,
                store.len(),
                "case {case}: {store:02x?} decodes to the wrong length"
            );
            let run = |patch: bool, ctx: &mut Ctx, data: &mut Vec<u8>| -> After {
                // SAFETY: our own code allocation, written and then run; the
                // buffers the operands point at are live.
                unsafe {
                    std::ptr::write_bytes(page, 0xcc, 0x4000);
                    std::ptr::copy_nonoverlapping(code.as_ptr(), page, code.len());
                    for k in 0..0x1000 {
                        *page.add(RIP_AREA + k) = (k * 7 + 3) as u8;
                    }
                    if patch {
                        let site_at = page as u64 + site as u64;
                        let at = page as u64 + TRAMPOLINE as u64;
                        let t = trampoline(&s, site_at, at).expect("a trampoline");
                        std::ptr::copy_nonoverlapping(t.as_ptr(), at as *mut u8, t.len());
                        let j = jump(site_at, s.len, at).expect("a jump");
                        std::ptr::copy_nonoverlapping(j.as_ptr(), site_at as *mut u8, j.len());
                    }
                    data.iter_mut()
                        .enumerate()
                        .for_each(|(k, b)| *b = (k * 13 + 5) as u8);
                    if let Some(low) = low {
                        for k in 0..0x1000 {
                            *low.add(k) = (k * 11 + 1) as u8;
                        }
                    }
                    let f: extern "win64" fn(*mut Ctx) = std::mem::transmute(page);
                    f(ctx);
                    After {
                        gpr: ctx
                            .out_gpr
                            .iter()
                            .enumerate()
                            .filter(|(i, _)| *i != 4)
                            .map(|(_, g)| *g)
                            .collect(),
                        flags: ctx.out_flags & FLAGS,
                        ymm: ctx.out_ymm.to_vec(),
                        stack: ctx.out_stack,
                        data: data.clone(),
                        rip: std::slice::from_raw_parts(page.add(RIP_AREA), 0x1000).to_vec(),
                        low: low
                            .map(|l| std::slice::from_raw_parts(l, 0x1000).to_vec())
                            .unwrap_or_default(),
                    }
                }
            };
            let before_regs = (ctx.gpr, ctx.flags, ctx.ymm, ctx.stack);
            let plain = run(false, &mut ctx, &mut data);
            (ctx.gpr, ctx.flags, ctx.ymm, ctx.stack) = before_regs;
            let patched = run(true, &mut ctx, &mut data);
            let what = format!(
                "case {case}: {store:02x?} ({kind:?}, ymm {ymm}, mask {mask}, src {src}, {m:?})"
            );
            assert_eq!(patched, plain, "{what}: the patched store differs");
            // The store's own definition, for the masked stores and moves:
            // only the bytes it stores change, to the stored register's.
            let lane = match kind {
                Kind::Masked {
                    opcode: 0x8e,
                    w: true,
                }
                | Kind::Masked { opcode: 0x2f, .. } => Some(8),
                Kind::Masked { .. } => Some(4),
                _ => None,
            };
            let (written, region): (Vec<Option<u8>>, &[u8]) = {
                let reg = &before_regs.2[usize::from(src)];
                let size = if ymm { 32 } else { 16 };
                let expect: Vec<Option<u8>> = match kind {
                    Kind::Masked { .. } => {
                        let lane = lane.unwrap();
                        let maskr = &before_regs.2[usize::from(mask)];
                        (0..size)
                            .map(|k| {
                                (maskr[(k / lane) * lane + lane - 1] & 0x80 != 0).then_some(reg[k])
                            })
                            .collect()
                    }
                    Kind::Extract(_) => (0..16)
                        .map(|k| Some(reg[(usize::from(imm & 1)) * 16 + k]))
                        .collect(),
                    Kind::Move { .. } => (0..32).map(|k| Some(reg[k])).collect(),
                    Kind::Cvt => Vec::new(),
                };
                let region: &[u8] = match m {
                    Mem::Rip(_) => &plain.rip,
                    Mem::Absolute { .. } => &plain.low,
                    Mem::Based { base: 4, .. } => &plain.stack,
                    Mem::Based { .. } => &plain.data,
                };
                (expect, region)
            };
            if !written.is_empty() {
                let at = match m {
                    Mem::Based {
                        base: 4,
                        index,
                        disp,
                        ..
                    } => {
                        (i64::from(disp)
                            + index.map_or(0, |(i, s)| (before_regs.0[usize::from(i)] as i64) << s))
                            as usize
                    }
                    _ => target as usize,
                };
                for (k, want) in written.iter().enumerate() {
                    if let Some(want) = want {
                        assert_eq!(region[at + k], *want, "{what}: byte {k} not the stored one");
                    }
                }
            }
            ran[match kind {
                Kind::Masked { .. } => 0,
                Kind::Cvt => 1,
                Kind::Extract(_) => 2,
                Kind::Move { .. } => 3,
            }] += 1;
        }
        // SAFETY: our own allocations.
        unsafe {
            VirtualFree(page.cast(), 0, MEM_RELEASE);
            if let Some(low) = low {
                VirtualFree(low.cast(), 0, MEM_RELEASE);
            }
        }
        eprintln!(
            "ran masked {} cvt {} extract {} move {} in {} operand forms; low buffer {}",
            ran[0],
            ran[1],
            ran[2],
            ran[3],
            forms.len(),
            low.is_some()
        );
        assert!(ran.iter().all(|&n| n > 100), "every kind ran: {ran:?}");
        // Base and index registers, the same register as both, rsp, rip and
        // (where a low buffer could be had) no base at all.
        assert!(forms.len() > 200, "operand forms: {}", forms.len());
    }
}
