//! Embeds the test shaders that the test-shaders workflow compiles with
//! Microsoft's FXC, since Wine's compiler cannot build all of them.

use std::fmt::Write as _;
use std::path::Path;

use anyhow::{Context, Result};

pub fn run(root: &Path, folder: &Path) -> Result<()> {
    let mut shaders = Vec::new();
    for entry in std::fs::read_dir(folder).context("reading the folder")? {
        let path = entry?.path();
        let Some(stem) = path.file_stem().and_then(|s| s.to_str()) else {
            continue;
        };
        if path.extension().and_then(|e| e.to_str()) != Some("bin") {
            continue;
        }
        let key = u64::from_str_radix(stem, 16)
            .with_context(|| format!("{} is not a key", path.display()))?;
        shaders.push((key, std::fs::read(&path)?));
    }
    shaders.sort();
    let mut text = String::from(
        "//! Shaders compiled by Microsoft's FXC on a Windows runner, for use\n\
         //! under Wine (`fxc`). Written by `cargo xtask\n\
         //! embed-test-shaders`.\n\n\
         pub(crate) const SHADERS: &[(u64, &[u8])] = &[\n",
    );
    for (key, bytes) in &shaders {
        let _ = writeln!(text, "    (0x{key:016x}, &[");
        for row in bytes.chunks(16) {
            text.push_str("        ");
            for b in row {
                let _ = write!(text, "0x{b:02x}, ");
            }
            text.push('\n');
        }
        text.push_str("    ]),\n");
    }
    text.push_str("];\n");
    let out = root.join("crates/victory-gpu/src/fxc_bytecode.rs");
    std::fs::write(&out, text).context("writing fxc_bytecode.rs")?;
    println!("embedded {} test shaders", shaders.len());
    Ok(())
}
