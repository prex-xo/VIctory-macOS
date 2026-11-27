//! 256-bit vector stores redone as two 128-bit stores (Mac port).
//!
//! Rosetta 2 cannot take a fault in the second page of a 32-byte store that
//! began in a writable page: it aborts with "unexpectedly need to
//! EmulateForward on a synchronous exception" before any handler runs
//! (measured in ~/victory-mac/rosetta-test/t2.c: `vmovdqu ymm`, `vmovups ymm`
//! and `vmaskmov ymm` die; 128-bit stores, `vextractf128` to memory and
//! smaller stores fault and resume normally). Page guards over title memory
//! therefore cannot be used while the title has such stores.
//!
//! `VICTORY_PATCH_WIDE=<file>` names a list of sites found offline with a
//! disassembler (one line each: RVA in hex, bytes replaced, then pieces:
//! `sN` a 256-bit store of N bytes, split; `cN` N bytes copied as they are;
//! `jN:T` a jump of N bytes to RVA T, which ends the run). Each site becomes
//! a jump to a trampoline that stores the low half with the same instruction
//! at 128 bits and the high half with `vextractf128 [m+16], ymm, 1`. Every
//! store piece is checked against the image's bytes before it is patched.

/// A store piece decoded: VEX fields and its memory operand.
#[derive(Debug, Clone, Copy)]
struct Wide {
    rxb: u8, // inverted R, X, B bits as in the VEX prefix (bits 2..0)
    map: u8,
    w: bool,
    vvvv: u8,
    pp: u8,
    opcode: u8,
    modrm: u8,
    sib: Option<u8>,
    disp: i32,
}

fn decode(b: &[u8]) -> Option<Wide> {
    let (rxb, map, w, vvvv, l, pp, rest) = match *b.first()? {
        0xc5 => {
            let p = *b.get(1)?;
            (
                (p >> 7) << 2 | 0b011,
                1,
                false,
                (p >> 3) & 15,
                (p >> 2) & 1,
                p & 3,
                2,
            )
        }
        0xc4 => {
            let (p1, p2) = (*b.get(1)?, *b.get(2)?);
            (
                p1 >> 5,
                p1 & 31,
                p2 & 0x80 != 0,
                (p2 >> 3) & 15,
                (p2 >> 2) & 1,
                p2 & 3,
                3,
            )
        }
        _ => return None,
    };
    let opcode = *b.get(rest)?;
    let modrm = *b.get(rest + 1)?;
    if map != 1 || l != 1 || !matches!(opcode, 0x11 | 0x29 | 0x7f | 0x2b | 0xe7) || modrm >> 6 == 3
    {
        return None;
    }
    let (md, rm) = (modrm >> 6, modrm & 7);
    let mut at = rest + 2;
    let sib = if rm == 4 {
        at += 1;
        Some(*b.get(at - 1)?)
    } else {
        None
    };
    let disp_len = match md {
        1 => 1,
        2 => 4,
        _ if rm == 5 || sib.is_some_and(|s| s & 7 == 5) => 4,
        _ => 0,
    };
    let d = b.get(at..at + disp_len)?;
    let disp = match disp_len {
        1 => i32::from(d[0] as i8),
        4 => i32::from_le_bytes(d.try_into().ok()?),
        _ => 0,
    };
    if at + disp_len != b.len() {
        return None;
    }
    Some(Wide {
        rxb,
        map,
        w,
        vvvv,
        pp,
        opcode,
        modrm,
        sib,
        disp,
    })
}

/// The memory operand of `s` with `extra` added to its displacement, always
/// with a 32-bit displacement. `rip`: the original instruction's end, for a
/// RIP-relative operand; `end`: where the new instruction will end.
fn memory(
    c: &mut Vec<u8>,
    s: &Wide,
    extra: i32,
    rip: u64,
    end_after: usize,
    at: u64,
) -> Option<()> {
    let (md, rm, reg) = (s.modrm >> 6, s.modrm & 7, s.modrm & 0x38);
    if md == 0 && rm == 5 {
        c.push(reg | 5);
        let target = rip as i64 + i64::from(s.disp) + i64::from(extra);
        let end = at as i64 + c.len() as i64 + 4 + end_after as i64;
        c.extend_from_slice(&i32::try_from(target - end).ok()?.to_le_bytes());
    } else if md == 0 && s.sib.is_some_and(|x| x & 7 == 5) {
        c.push(reg | 4);
        c.push(s.sib?);
        c.extend_from_slice(&s.disp.checked_add(extra)?.to_le_bytes());
    } else {
        c.push(0x80 | reg | rm);
        if let Some(x) = s.sib {
            c.push(x);
        }
        c.extend_from_slice(&s.disp.checked_add(extra)?.to_le_bytes());
    }
    Some(())
}

/// The two 128-bit stores for `s`, appended to `c` (whose first byte will lie at `base_at`).
fn split(c: &mut Vec<u8>, s: &Wide, rip: u64, base_at: u64) -> Option<()> {
    // Low half: the same store at 128 bits.
    c.extend_from_slice(&[
        0xc4,
        s.rxb << 5 | s.map,
        u8::from(s.w) << 7 | s.vvvv << 3 | s.pp,
        s.opcode,
    ]);
    memory(c, s, 0, rip, 0, base_at)?;
    // High half: vextractf128 [m+16], ymm, 1.
    c.extend_from_slice(&[0xc4, s.rxb << 5 | 3, 0x7d, 0x19]);
    memory(c, s, 16, rip, 1, base_at)?;
    c.push(1);
    Some(())
}

#[derive(Debug, Clone)]
enum Piece {
    Store(usize),
    Copy(usize),
    Jump(usize, u32),
    /// A conditional jump of N bytes, its condition, its target RVA.
    Cond(usize, u8, u32),
    /// N bytes copied whose 32-bit displacement at K is RIP-relative to RVA T.
    Rip(usize, usize, u32),
}

/// Where a 4-byte site jumps first: padding slots (RVAs) jumped through
/// with 2-byte jumps, the last of 5 bytes holding the jump to the trampoline.
type Route = Vec<u32>;

fn parse(line: &str) -> Option<(u32, usize, Vec<Piece>, Route)> {
    let mut f = line.split_whitespace();
    let rva = u32::from_str_radix(f.next()?, 16).ok()?;
    let total: usize = f.next()?.parse().ok()?;
    let mut pieces = Vec::new();
    let mut route = Route::new();
    let hex = |t: &str| u32::from_str_radix(t, 16).ok();
    for p in f {
        let (k, v) = p.split_at(1);
        let parts: Vec<&str> = v.split(':').collect();
        pieces.push(match (k, parts.as_slice()) {
            ("s", [n]) => Piece::Store(n.parse().ok()?),
            ("c", [n]) => Piece::Copy(n.parse().ok()?),
            ("j", [n, t]) => Piece::Jump(n.parse().ok()?, hex(t)?),
            ("k", [n, cc, t]) => {
                Piece::Cond(n.parse().ok()?, u8::from_str_radix(cc, 16).ok()?, hex(t)?)
            }
            ("r", [n, at, t]) => Piece::Rip(n.parse().ok()?, at.parse().ok()?, hex(t)?),
            ("@", _) => {
                route = v.split(',').map(hex).collect::<Option<_>>()?;
                continue;
            }
            _ => return None,
        });
    }
    let sum: Option<usize> = pieces
        .iter()
        .map(|p| match p {
            Piece::Store(n)
            | Piece::Copy(n)
            | Piece::Jump(n, _)
            | Piece::Cond(n, ..)
            | Piece::Rip(n, ..) => *n,
        })
        .try_fold(0usize, |acc, n| acc.checked_add(n));
    let sum = sum?;
    (sum == total && (total >= 5 || !route.is_empty() && total >= 2))
        .then_some((rva, total, pieces, route))
}

/// The trampoline for one site, its first byte at `at`; `code` the image's bytes at the site.
fn trampoline(
    base: u64,
    rva: u32,
    total: usize,
    pieces: &[Piece],
    code: &[u8],
    at: u64,
) -> Option<Vec<u8>> {
    let site = base + u64::from(rva);
    let mut c = Vec::with_capacity(64);
    let mut off = 0usize;
    for p in pieces {
        match *p {
            Piece::Store(n) => {
                let s = decode(code.get(off..off + n)?)?;
                let rip = site + (off + n) as u64;
                split(&mut c, &s, rip, at)?;
                off += n;
            }
            Piece::Copy(n) => {
                let b = code.get(off..off + n)?;
                // Nothing that moves with its address may be copied.
                if b.iter().all(|&x| x == 0xcc) {
                    return None;
                }
                c.extend_from_slice(b);
                off += n;
            }
            Piece::Cond(n, cc, target) => {
                let b = code.get(off..off + n)?;
                let ok = n == 2 && b[0] == 0x70 | cc || n == 6 && b[0] == 0x0f && b[1] == 0x80 | cc;
                if !ok || cc > 15 {
                    return None;
                }
                let to = base + u64::from(target);
                let next = at as i64 + c.len() as i64 + 6;
                c.extend_from_slice(&[0x0f, 0x80 | cc]);
                c.extend_from_slice(&i32::try_from(to as i64 - next).ok()?.to_le_bytes());
                off += n;
            }
            Piece::Rip(n, k, target) => {
                let b = code.get(off..off + n)?;
                let old = i32::from_le_bytes(b.get(k..k + 4)?.try_into().ok()?);
                // The displacement must be the one the list was made with.
                if (site + (off + n) as u64) as i64 + i64::from(old)
                    != (base + u64::from(target)) as i64
                {
                    return None;
                }
                let end = at as i64 + (c.len() + n) as i64;
                let new = i32::try_from((base + u64::from(target)) as i64 - end).ok()?;
                let from = c.len();
                c.extend_from_slice(b);
                c[from + k..from + k + 4].copy_from_slice(&new.to_le_bytes());
                off += n;
            }
            Piece::Jump(n, target) => {
                let b = code.get(off..off + n)?;
                if !(b[0] == 0xe9 && n == 5 || b[0] == 0xeb && n == 2) {
                    return None;
                }
                let to = base + u64::from(target);
                let next = at as i64 + c.len() as i64 + 5;
                c.push(0xe9);
                c.extend_from_slice(&i32::try_from(to as i64 - next).ok()?.to_le_bytes());
                return (off + n == total).then_some(c);
            }
        }
    }
    let back = site + total as u64;
    let next = at as i64 + c.len() as i64 + 5;
    c.push(0xe9);
    c.extend_from_slice(&i32::try_from(back as i64 - next).ok()?.to_le_bytes());
    Some(c)
}

#[cfg(windows)]
pub fn patch(base: u64, list: &std::path::Path) -> (usize, usize) {
    use windows_sys::Win32::System::Diagnostics::Debug::FlushInstructionCache;
    use windows_sys::Win32::System::Memory::{
        MEM_COMMIT, MEM_RESERVE, PAGE_EXECUTE_READ, PAGE_EXECUTE_READWRITE, PAGE_READWRITE,
        VirtualAlloc, VirtualProtect,
    };
    use windows_sys::Win32::System::Threading::GetCurrentProcess;
    let Ok(text) = std::fs::read_to_string(list) else {
        tracing::warn!(?list, "VICTORY_PATCH_WIDE: the site list cannot be read");
        return (0, 0);
    };
    let sites: Vec<_> = text.lines().filter_map(parse).collect();
    const SIZE: usize = 16 << 20;
    let mut pool = std::ptr::null_mut();
    // Below the masked stores' pool (base - n MiB), within reach of rel32.
    for step in 64..2000u64 {
        let Some(want) = base.checked_sub(step << 20) else {
            break;
        };
        // SAFETY: a plain reservation of free address space.
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
        tracing::warn!("no memory near the image for wide store trampolines");
        return (0, sites.len());
    }
    let (mut used, mut patched, mut refused) = (0usize, 0usize, 0usize);
    // Pages of the image made writable once each.
    let mut opened: std::collections::BTreeMap<u64, u32> = std::collections::BTreeMap::new();
    for (rva, total, pieces, route) in &sites {
        let site = base + u64::from(*rva);
        // SAFETY: the site lies in the mapped image (its list was made from it).
        let code = unsafe { std::slice::from_raw_parts(site as *const u8, *total) };
        let at = pool as u64 + used as u64;
        let Some(t) = trampoline(base, *rva, *total, pieces, code, at) else {
            refused += 1;
            continue;
        };
        if used + t.len() > SIZE {
            refused += sites.len() - patched - refused;
            break;
        }
        // What is written: (address, bytes), the site's first.
        let mut writes: Vec<(u64, Vec<u8>)> = Vec::new();
        if route.is_empty() {
            let Ok(rel) = i32::try_from(at as i64 - (site as i64 + 5)) else {
                refused += 1;
                continue;
            };
            let mut jump = vec![0xe9];
            jump.extend_from_slice(&rel.to_le_bytes());
            jump.resize(*total, 0xcc);
            writes.push((site, jump));
        } else {
            let mut from = site;
            let mut ok = true;
            for (k, hop) in route.iter().enumerate() {
                let to = base + u64::from(*hop);
                let last = k + 1 == route.len();
                // SAFETY: padding inside the mapped image.
                let pad = unsafe {
                    std::slice::from_raw_parts(to as *const u8, if last { 5 } else { 2 })
                };
                let Ok(rel) = i8::try_from(to as i64 - (from as i64 + 2)) else {
                    ok = false;
                    break;
                };
                if pad.iter().any(|&x| x != 0xcc) {
                    ok = false;
                    break;
                }
                let mut j = vec![0xeb, rel as u8];
                if from == site {
                    j.resize(*total, 0xcc);
                }
                writes.push((from, j));
                if last {
                    let Ok(rel) = i32::try_from(at as i64 - (to as i64 + 5)) else {
                        ok = false;
                        break;
                    };
                    let mut j = vec![0xe9];
                    j.extend_from_slice(&rel.to_le_bytes());
                    writes.push((to, j));
                }
                from = to;
            }
            if !ok {
                refused += 1;
                continue;
            }
        }
        // SAFETY: the pool is ours and writable; the site and its route lie in the mapped image, opened here.
        unsafe {
            std::ptr::copy_nonoverlapping(t.as_ptr(), at as *mut u8, t.len());
            for (to, bytes) in &writes {
                for page in (to & !0xfff..=(to + bytes.len() as u64 - 1) & !0xfff).step_by(0x1000) {
                    if let std::collections::btree_map::Entry::Vacant(e) = opened.entry(page) {
                        let mut old = 0;
                        if VirtualProtect(
                            page as *const _,
                            0x1000,
                            PAGE_EXECUTE_READWRITE,
                            &mut old,
                        ) == 0
                        {
                            continue;
                        }
                        e.insert(old);
                    }
                }
            }
            // The hops first, the site last: a thread at the site sees a whole route.
            for (to, bytes) in writes.iter().rev() {
                std::ptr::copy_nonoverlapping(bytes.as_ptr(), *to as *mut u8, bytes.len());
            }
        }
        used = (used + t.len() + 15) & !15;
        patched += 1;
    }
    // SAFETY: the image pages opened above get their protection back; the pool is ours.
    unsafe {
        for (page, old) in opened {
            let mut was = 0;
            VirtualProtect(page as *const _, 0x1000, old, &mut was);
        }
        let mut old = 0;
        VirtualProtect(pool, SIZE, PAGE_EXECUTE_READ, &mut old);
        FlushInstructionCache(GetCurrentProcess(), std::ptr::null(), 0);
    }
    tracing::info!(
        patched,
        refused,
        bytes = used,
        "wide stores split into 128-bit halves (VICTORY_PATCH_WIDE)"
    );
    (patched, refused)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_list_whose_piece_sizes_overflow_is_rejected() {
        // The declared total is the pieces' wrapped sum, so a sum that wraps
        // instead of overflowing (a plain `+` in a release build) would make
        // this line parse: the assertion fails in release builds too, not
        // only as a panic in debug ones. A total of usize::MAX is not asserted
        // to parse: patch() would hand it to slice::from_raw_parts, whose
        // length must stay within isize::MAX.
        let max = usize::MAX;
        let line = format!("1000 {} s{max} s6", max.wrapping_add(6));
        assert!(parse(&line).is_none());
        // Control: an ordinary line.
        assert!(parse("1000 10 s6 c4").is_some());
    }

    #[test]
    fn splits_vmovups_rdi() {
        // vmovups [rdi], ymm0
        let s = decode(&[0xc5, 0xfc, 0x11, 0x07]).unwrap();
        let mut c = Vec::new();
        split(&mut c, &s, 0, 0).unwrap();
        // vmovups [rdi+0], xmm0 ; vextractf128 [rdi+16], ymm0, 1
        assert_eq!(
            c,
            [
                0xc4, 0xe1, 0x78, 0x11, 0x87, 0, 0, 0, 0, 0xc4, 0xe3, 0x7d, 0x19, 0x87, 16, 0, 0,
                0, 1
            ]
        );
    }

    #[test]
    fn splits_r12_with_sib_and_high_register() {
        // vmovups [r12], ymm0 = c4 c1 7c 11 04 24 ; vmovdqu [rax+rcx*8+0x40], ymm9 = c5 7e 7f 4c c8 40
        let s = decode(&[0xc4, 0xc1, 0x7c, 0x11, 0x04, 0x24]).unwrap();
        let mut c = Vec::new();
        split(&mut c, &s, 0, 0).unwrap();
        assert_eq!(&c[..10], &[0xc4, 0xc1, 0x78, 0x11, 0x84, 0x24, 0, 0, 0, 0]);
        let s = decode(&[0xc5, 0x7e, 0x7f, 0x4c, 0xc8, 0x40]).unwrap();
        let mut c = Vec::new();
        split(&mut c, &s, 0, 0).unwrap();
        assert_eq!(
            &c[..10],
            &[0xc4, 0x61, 0x7a, 0x7f, 0x8c, 0xc8, 0x40, 0, 0, 0]
        );
        assert_eq!(
            &c[10..],
            &[0xc4, 0x63, 0x7d, 0x19, 0x8c, 0xc8, 0x50, 0, 0, 0, 1]
        );
    }

    #[test]
    fn refuses_loads_and_xmm() {
        assert!(decode(&[0xc5, 0xfc, 0x10, 0x07]).is_none()); // load
        assert!(decode(&[0xc5, 0xf8, 0x11, 0x07]).is_none()); // xmm store
        assert!(decode(&[0xc5, 0xfc, 0x11, 0xc1]).is_none()); // register form
    }
}

#[cfg(test)]
mod offline {
    use super::*;
    /// `VICTORY_WIDE_IMAGE=<pe file> VICTORY_WIDE_LIST=<list> cargo test offline -- --nocapture`
    #[test]
    fn replay() {
        let (Some(img), Some(list)) = (
            std::env::var_os("VICTORY_WIDE_IMAGE"),
            std::env::var_os("VICTORY_WIDE_LIST"),
        ) else {
            return;
        };
        let file = std::fs::read(img).unwrap();
        let pe = u32::from_le_bytes(file[0x3c..0x40].try_into().unwrap()) as usize;
        let nsec = u16::from_le_bytes(file[pe + 6..pe + 8].try_into().unwrap()) as usize;
        let opt = u16::from_le_bytes(file[pe + 20..pe + 22].try_into().unwrap()) as usize;
        let mut secs = Vec::new();
        for i in 0..nsec {
            let o = pe + 24 + opt + 40 * i;
            let r =
                |k: usize| u32::from_le_bytes(file[o + k..o + k + 4].try_into().unwrap()) as usize;
            secs.push((r(12), r(16), r(20)));
        }
        let at_rva = |rva: usize, n: usize| {
            let &(va, _, roff) = secs.iter().find(|s| s.0 <= rva && rva < s.0 + s.1).unwrap();
            &file[roff + rva - va..roff + rva - va + n]
        };
        let base = 0x6ffff1de0000u64;
        let pool = base - (64 << 20);
        let mut used = 0usize;
        let mut reasons = std::collections::BTreeMap::<String, usize>::new();
        let text = std::fs::read_to_string(list).unwrap();
        let want = std::env::var("VICTORY_WIDE_AT")
            .ok()
            .and_then(|v| usize::from_str_radix(&v, 16).ok());
        for (rva, total, pieces, _route) in text.lines().filter_map(parse) {
            let code = at_rva(rva as usize, total);
            let at = pool + used as u64;
            match trampoline(base, rva, total, &pieces, code, at) {
                Some(t) => {
                    if want.is_some_and(|w| (used..used + t.len()).contains(&w)) {
                        println!(
                            "site {rva:#x} {pieces:?} code {code:02x?} tramp@{used:#x} {t:02x?}"
                        );
                    }
                    used = (used + t.len() + 15) & !15;
                }
                None => {
                    let first = match pieces[0] {
                        Piece::Store(n) => n,
                        _ => 0,
                    };
                    let k = if decode(&code[..first]).is_none() {
                        format!("decode {:02x?}", &code[..first.min(3)])
                    } else {
                        "other".into()
                    };
                    *reasons.entry(k).or_default() += 1;
                }
            }
        }
        let mut r: Vec<_> = reasons.into_iter().collect();
        r.sort_by_key(|x| std::cmp::Reverse(x.1));
        println!("used {used:#x} refused {:?}", &r[..r.len().min(12)]);
    }
}
