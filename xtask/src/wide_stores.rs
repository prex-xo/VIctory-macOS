//! `cargo xtask wide-stores <image>`: lists the title's 256-bit vector stores
//! for `VICTORY_PATCH_WIDE` (`victory-loader/src/wide.rs`).
//!
//! Rosetta 2 stops ("unexpectedly need to EmulateForward on a synchronous
//! exception") when a 32-byte store begins in writable memory and faults in
//! the next page, which fault tracking makes happen (docs/macos.md, "Plain
//! 256-bit stores into protected pages stop Rosetta too"). The loader
//! rewrites each listed store as a jump to a trampoline that stores the two
//! 128-bit halves. This finds the stores with `llvm-objdump` and prints the
//! list; nothing of the title is in it but offsets and lengths.
//!
//! The image is the mapped copy VIctory writes to its cache
//! (`<cache_dir>/images/victory-image-<hash>.dll`), whose bytes are the
//! title's.
//!
//! A store of 5 bytes or more becomes a 5-byte jump in place. A 4-byte store
//! has no room for one: its run takes in the instructions after it (or
//! before it) that can be moved, until 5 bytes are covered, and only where
//! no branch lands inside the run. A line: RVA (hex), bytes replaced, then
//! the pieces of the run: `sN` a 256-bit store, `cN` bytes copied, `rN:K:T`
//! bytes copied with their RIP-relative displacement at K redone for RVA T,
//! `kN:CC:T` a conditional jump to T, `jN:T` a jump to T that ends the run.

use std::collections::{BTreeMap, HashMap, HashSet, VecDeque};
use std::io::{BufRead, BufReader, Write};
use std::path::Path;
use std::process::{Command, Stdio};

use anyhow::{Context, Result};

/// One decoded instruction: its RVA, length and text (`mnemonic operands`,
/// with any `# 0x...` comment `llvm-objdump` adds).
type Insn = (u32, usize, String);

struct Store {
    insn: Insn,
    /// The three instructions before it, nearest first.
    before: Vec<Insn>,
    /// The four after it.
    after: Vec<Insn>,
}

struct Sections {
    base: u64,
    by_name: HashMap<String, (u32, u32, u32, u32)>,
}

fn sections(data: &[u8]) -> Result<Sections> {
    let u16_at = |at: usize| u16::from_le_bytes([data[at], data[at + 1]]) as usize;
    let u32_at = |at: usize| u32::from_le_bytes(data[at..at + 4].try_into().unwrap());
    anyhow::ensure!(data.len() > 0x40 && &data[..2] == b"MZ", "not a PE image");
    let pe = u32_at(0x3c) as usize;
    let count = u16_at(pe + 6);
    let optional = u16_at(pe + 20);
    let base = u64::from_le_bytes(data[pe + 48..pe + 56].try_into().unwrap());
    let mut by_name = HashMap::new();
    for i in 0..count {
        let at = pe + 24 + optional + 40 * i;
        let name = String::from_utf8_lossy(&data[at..at + 8])
            .trim_end_matches('\0')
            .to_string();
        by_name.insert(
            name,
            (
                u32_at(at + 12),
                u32_at(at + 8),
                u32_at(at + 20),
                u32_at(at + 16),
            ),
        );
    }
    Ok(Sections { base, by_name })
}

fn starts_with_word(text: &str, word: &str) -> bool {
    text.starts_with(word)
        && !text[word.len()..]
            .chars()
            .next()
            .is_some_and(|c| c.is_ascii_alphanumeric() || c == '_')
}

/// A 256-bit store to memory: `v<op> ymmword ptr [...], ymmN`.
fn is_store(text: &str) -> bool {
    let Some((mnemonic, rest)) = text.split_once(' ') else {
        return false;
    };
    let Some(inside) = rest.strip_prefix("ymmword ptr [") else {
        return false;
    };
    let Some((address, register)) = inside.split_once("], ") else {
        return false;
    };
    mnemonic.starts_with('v')
        && !address.contains(']')
        && register
            .strip_prefix("ymm")
            .is_some_and(|n| !n.is_empty() && n.bytes().all(|b| b.is_ascii_digit()))
}

fn is_masked(text: &str) -> bool {
    text.starts_with("vmaskmov") || text.starts_with("vpmaskmov")
}

/// What cannot be moved out of place (or is patched by `VICTORY_PATCH_MASKED`).
fn is_fixed(text: &str) -> bool {
    const PREFIXES: [&str; 19] = [
        "vextract",
        "vcvtps2ph",
        "vpmaskmov",
        "vmaskmov",
        "call",
        "ret",
        "loop",
        "bnd",
        "int3",
        "syscall",
        "xbegin",
        "xend",
        "push",
        "pop",
        "enter",
        "leave",
        "rep",
        "lock",
        "j",
    ];
    PREFIXES
        .iter()
        .any(|p| text.starts_with(p) && (*p != "j" || text.len() > 1))
        || starts_with_word(text, "int")
}

/// The hexadecimal number at the start of `text` after `0x`.
fn hex_after_0x(text: &str) -> Option<u64> {
    let digits = text.strip_prefix("0x")?;
    let end = digits
        .find(|c: char| !c.is_ascii_hexdigit())
        .unwrap_or(digits.len());
    u64::from_str_radix(&digits[..end], 16).ok()
}

/// `(mnemonic, target)` of a direct branch.
fn branch(text: &str) -> Option<(&str, u64)> {
    let (mnemonic, rest) = text.split_once(' ')?;
    let is_branch = (mnemonic.starts_with('j') && mnemonic.len() > 1)
        || mnemonic == "call"
        || mnemonic.starts_with("loop")
        || mnemonic == "xbegin";
    if !is_branch {
        return None;
    }
    Some((mnemonic, hex_after_0x(rest)?))
}

/// The target of the RIP-relative operand `llvm-objdump` notes as `# 0x...`.
fn noted_target(text: &str) -> Option<u64> {
    hex_after_0x(&text[text.find("# 0x")? + 2..])
}

pub(crate) fn run(image: &Path) -> Result<()> {
    let objdump = ["llvm-objdump", "/opt/homebrew/opt/llvm/bin/llvm-objdump"]
        .into_iter()
        .find(|o| {
            Command::new(o)
                .arg("--version")
                .stdout(Stdio::null())
                .status()
                .is_ok_and(|s| s.success())
        })
        .context("llvm-objdump not found (brew install llvm)")?;
    let data = std::fs::read(image).with_context(|| format!("reading {}", image.display()))?;
    let secs = sections(&data)?;
    let base = secs.base;
    let section = |name: &str| {
        secs.by_name
            .get(name)
            .copied()
            .with_context(|| format!("no {name} section"))
    };
    let (tva, tsize, traw, _) = section(".text")?;
    let code = |rva: u32, n: usize| {
        let at = (traw + rva - tva) as usize;
        &data[at..at + n]
    };

    // Where branches land: direct targets, function starts and ends, and
    // anything in the data that looks like an address in the code (switch tables).
    let mut targets: HashSet<u32> = HashSet::new();
    // A small DLL may lack any of these sections.
    if let Ok((_, vsize, raw, rsize)) = section(".pdata") {
        for entry in data[raw as usize..][..vsize.min(rsize) as usize]
            .as_chunks::<12>()
            .0
        {
            targets.insert(u32::from_le_bytes([entry[0], entry[1], entry[2], entry[3]]));
            targets.insert(u32::from_le_bytes([entry[4], entry[5], entry[6], entry[7]]));
        }
    }
    for name in [".rdata", ".data"] {
        let Ok((_, _, raw, rsize)) = section(name) else {
            continue;
        };
        for word in data[raw as usize..][..rsize as usize].as_chunks::<4>().0 {
            let value = u32::from_le_bytes(*word);
            if (tva..tva + tsize).contains(&value) {
                targets.insert(value);
            }
        }
    }

    // One pass: every 256-bit store, with the three instructions before it
    // and the four after.
    let mut child = Command::new(objdump)
        .args(["-d", "--x86-asm-syntax=intel", "-j", ".text"])
        .arg(image)
        .stdout(Stdio::piped())
        .spawn()
        .context("running llvm-objdump")?;
    let reader =
        BufReader::with_capacity(1 << 20, child.stdout.take().context("llvm-objdump output")?);
    let mut stores: Vec<Store> = Vec::new();
    let mut recent: VecDeque<Insn> = VecDeque::with_capacity(4);
    let mut waiting: Vec<usize> = Vec::new();
    for line in reader.lines() {
        let line = line?;
        let Some(insn) = parse(&line, base) else {
            if line.trim().is_empty() || line.ends_with(':') {
                recent.clear();
            }
            continue;
        };
        if let Some((_, target)) = branch(&insn.2) {
            targets.insert(target.wrapping_sub(base) as u32);
        }
        for &w in &waiting {
            stores[w].after.push(insn.clone());
        }
        waiting.retain(|&w| stores[w].after.len() < 4);
        if is_store(command(&insn.2)) {
            stores.push(Store {
                insn: insn.clone(),
                before: recent.iter().rev().cloned().collect(),
                after: Vec::new(),
            });
            waiting.push(stores.len() - 1);
        }
        if recent.len() == 3 {
            recent.pop_front();
        }
        recent.push_back(insn);
    }
    anyhow::ensure!(child.wait()?.success(), "llvm-objdump failed");

    let piece = |a: u32, m: usize, text: &str| -> Option<String> {
        let t = command(text);
        if is_store(t) {
            return (!t.contains("rip") && !is_masked(t)).then(|| format!("s{m}"));
        }
        if let Some((mnemonic, target)) = branch(t)
            && mnemonic.starts_with('j')
            && !mnemonic.starts_with("jmp")
        {
            let b = code(a, m);
            let cc = if m == 2 { b[0] } else { b[1] } & 0xf;
            return Some(format!("k{m}:{cc:x}:{:x}", target - base));
        }
        if t.contains("rip") {
            if is_fixed(t) {
                return None;
            }
            let target = noted_target(text)?.checked_sub(base)? as i64;
            let b = code(a, m);
            for k in 1..m.saturating_sub(3) {
                let disp = i32::from_le_bytes(b[k..k + 4].try_into().unwrap());
                if i64::from(disp) == target - (i64::from(a) + m as i64) {
                    return Some(format!("r{m}:{k}:{target:x}"));
                }
            }
            return None;
        }
        if is_fixed(t) || t.is_empty() {
            return None;
        }
        Some(format!("c{m}"))
    };

    let mut sites: BTreeMap<u32, (usize, Vec<String>)> = BTreeMap::new();
    let mut short: Vec<&Store> = Vec::new();
    let mut rip_short = 0;
    for store in &stores {
        let (rva, n, text) = &store.insn;
        let t = command(text);
        if is_masked(t) {
            continue;
        }
        if *n >= 5 {
            sites.entry(*rva).or_insert((*n, vec![format!("s{n}")]));
        } else if t.contains("rip") {
            rip_short += 1;
        } else {
            short.push(store);
        }
    }
    // 4-byte stores: the instructions after first, then those before.
    let mut left: Vec<&Store> = Vec::new();
    let mut taken: HashSet<u32> = HashSet::new();
    for store in short {
        let rva = store.insn.0;
        if taken.contains(&rva) {
            continue;
        }
        let (mut pieces, mut cur) = (vec!["s4".to_string()], rva + 4);
        for (a, m, text) in &store.after {
            if *a != cur || targets.contains(a) {
                break;
            }
            let t = command(text);
            if let Some(("jmp", target)) = branch(t) {
                pieces.push(format!("j{m}:{:x}", target - base));
                cur += *m as u32;
                break;
            }
            let Some(p) = piece(*a, *m, text) else {
                break;
            };
            pieces.push(p);
            cur += *m as u32;
            if cur - rva >= 5 {
                break;
            }
        }
        if cur - rva >= 5 {
            for k in rva + 1..cur {
                sites.remove(&k);
                taken.insert(k);
            }
            sites.insert(rva, ((cur - rva) as usize, pieces));
        } else {
            left.push(store);
        }
    }
    let free = |sites: &BTreeMap<u32, (usize, Vec<String>)>, lo: u32, hi: u32| {
        !sites
            .range(lo.saturating_sub(63)..hi)
            .any(|(&r, &(t, _))| r + t as u32 > lo)
    };
    let mut ends: HashMap<u32, u32> = sites
        .iter()
        .filter(|(_, (_, p))| !p.iter().any(|x| x.starts_with('j')))
        .map(|(&r, &(t, _))| (r + t as u32, r))
        .collect();
    let mut unpatched = 0;
    for store in left {
        let rva = store.insn.0;
        // A site that ends where this store begins takes it in.
        if !targets.contains(&rva)
            && let Some(r0) = ends.remove(&rva)
        {
            let site = sites.get_mut(&r0).expect("ends names sites");
            site.0 += 4;
            site.1.push("s4".into());
            ends.insert(rva + 4, r0);
            continue;
        }
        let (mut run, mut start) = (Vec::new(), rva);
        for (a, m, text) in &store.before {
            if a + *m as u32 != start || targets.contains(&start) {
                break;
            }
            let Some(p) = piece(*a, *m, text) else {
                break;
            };
            run.insert(0, p);
            start = *a;
            if rva + 4 - start >= 5 {
                break;
            }
        }
        if rva + 4 - start >= 5 && free(&sites, start, rva + 4) {
            run.push("s4".into());
            sites.insert(start, ((rva + 4 - start) as usize, run));
        } else {
            unpatched += 1;
        }
    }
    let mut out = std::io::BufWriter::new(std::io::stdout().lock());
    let (mut last, mut listed) = (0u32, 0usize);
    for (&rva, (total, pieces)) in &sites {
        if rva < last {
            continue;
        }
        writeln!(out, "{rva:x} {total} {}", pieces.join(" "))?;
        last = rva + *total as u32;
        listed += 1;
    }
    out.flush()?;
    eprintln!(
        "{} 256-bit stores; {listed} sites listed; {unpatched} 4-byte stores left (no room and nothing that can \
         be moved){}",
        stores.len(),
        if rip_short > 0 {
            format!("; {rip_short} 4-byte RIP-relative left")
        } else {
            String::new()
        }
    );
    if stores.is_empty() {
        eprintln!("no 256-bit stores in this image");
    }
    Ok(())
}

/// An instruction without the comment `llvm-objdump` adds.
fn command(text: &str) -> &str {
    text.split(" #").next().unwrap_or(text)
}

/// `addr: bytes \t mnemonic \t operands` as `(rva, length, text)`.
fn parse(line: &str, base: u64) -> Option<Insn> {
    let (address, rest) = line.split_once(':')?;
    let address = u64::from_str_radix(address.trim(), 16).ok()?;
    let mut fields = rest.split('\t');
    let bytes = fields.next()?.split_whitespace().count();
    let mnemonic = fields.next()?.trim();
    // An instruction listed as one field with spaces (some prefixed forms) is
    // passed over, as one that cannot be moved.
    if bytes == 0 || mnemonic.is_empty() || mnemonic.contains(char::is_whitespace) {
        return None;
    }
    let operands = fields.collect::<Vec<_>>().join(" ");
    let text = format!("{mnemonic} {operands}")
        .split_whitespace()
        .collect::<Vec<_>>()
        .join(" ");
    Some((address.checked_sub(base)? as u32, bytes, text))
}
