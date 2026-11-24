//! Embeds the argument clamp bytecode that the clamp-bytecode workflow
//! builds with Microsoft's compilers, since Wine cannot compile the shader.

use std::fmt::Write as _;
use std::path::Path;

use anyhow::{Context, Result};

/// FNV-1a, the same hash `victory-gpu` takes of the source.
fn fnv(data: &[u8]) -> u64 {
    data.iter().fold(0xcbf2_9ce4_8422_2325, |h, &b| {
        (h ^ u64::from(b)).wrapping_mul(0x100_0000_01b3)
    })
}

fn array(data: &[u8]) -> String {
    let mut out = String::from("&[\n");
    for row in data.chunks(16) {
        out.push_str("    ");
        for b in row {
            let _ = write!(out, "0x{b:02x}, ");
        }
        out.push('\n');
    }
    out.push(']');
    out
}

pub fn run(root: &Path, folder: &Path) -> Result<()> {
    let src = root.join("crates/victory-gpu/src");
    let source = std::fs::read(src.join("clamp.hlsl")).context("reading clamp.hlsl")?;
    let dxil = std::fs::read(folder.join("clamp.dxil")).context("reading clamp.dxil")?;
    let dxbc = std::fs::read(folder.join("clamp.dxbc")).context("reading clamp.dxbc")?;
    let text = format!(
        "//! The argument clamp shader (`clamp.hlsl`) compiled by Microsoft's\n\
         //! compilers on a Windows runner. Written by `cargo xtask embed-clamp`.\n\
         //! `SOURCE_HASH` names the source it was built from, and other bytes are\n\
         //! never used.\n\n\
         pub(crate) const SOURCE_HASH: u64 = 0x{:016x};\n\
         pub(crate) const DXIL: &[u8] = {};\n\
         pub(crate) const DXBC: &[u8] = {};\n",
        fnv(&source),
        array(&dxil),
        array(&dxbc)
    );
    std::fs::write(src.join("clamp_bytecode.rs"), text).context("writing clamp_bytecode.rs")?;
    println!(
        "embedded {} bytes of DXIL and {} of DXBC",
        dxil.len(),
        dxbc.len()
    );
    Ok(())
}
