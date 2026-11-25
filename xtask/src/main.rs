//! Repository maintenance. Run with `cargo xtask <command>`.

mod embed_clamp;
mod embed_test_shaders;
mod policy;

use std::path::{Path, PathBuf};
use std::process::Command;

use anyhow::{Context, Result};
use clap::{Parser, Subcommand};

#[derive(Parser)]
#[command(name = "xtask")]
struct Cli {
    #[command(subcommand)]
    command: Cmd,
}

#[derive(Subcommand)]
enum Cmd {
    /// Enforce the asset policy: the repository holds only small text files
    /// we wrote. Checks tracked and untracked-but-not-ignored files.
    Policy {
        /// Check only what is staged for commit (used by the pre-commit hook).
        #[arg(long)]
        staged: bool,
    },
    /// Point git at .githooks/ so every commit runs the policy check.
    InstallHooks,
    /// Write crates/victory-gpu/src/clamp_bytecode.rs from the output of
    /// the clamp-bytecode workflow (a folder with clamp.dxil and clamp.dxbc).
    EmbedClamp { folder: PathBuf },
    /// Write crates/victory-gpu/src/fxc_bytecode.rs from the output of the
    /// test-shaders workflow (a folder of `<key>.bin` files).
    EmbedTestShaders { folder: PathBuf },
}

fn main() -> Result<()> {
    let root = workspace_root();
    match Cli::parse().command {
        Cmd::Policy { staged } => policy::run(&root, staged),
        Cmd::EmbedClamp { folder } => embed_clamp::run(&root, &folder),
        Cmd::EmbedTestShaders { folder } => embed_test_shaders::run(&root, &folder),
        Cmd::InstallHooks => {
            let status = Command::new("git")
                .args(["config", "core.hooksPath", ".githooks"])
                .current_dir(&root)
                .status()
                .context("running git")?;
            anyhow::ensure!(
                status.success(),
                "git config failed (is this a git repository?)"
            );
            println!("hooks installed: commits now run `cargo xtask policy --staged`");
            Ok(())
        }
    }
}

fn workspace_root() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR"))
        .parent()
        .expect("xtask lives in the workspace")
        .into()
}
