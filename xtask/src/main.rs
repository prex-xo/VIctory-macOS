//! Repository maintenance. Run with `cargo xtask <command>`.

mod bench_compare;
mod policy;
mod wide_stores;

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
    /// List the title's 256-bit stores for `VICTORY_PATCH_WIDE` (macOS):
    /// `cargo xtask wide-stores <cached title image> > wide-sites.txt`, or
    /// `cargo xtask wide-stores --cache <cache dir>` for every cached image
    /// at once, into `<cache dir>/wide`, where VIctory finds them.
    WideStores {
        /// The mapped title image in VIctory's cache (`images/victory-image-<hash>.dll`).
        #[arg(required_unless_present = "cache", conflicts_with = "cache")]
        image: Option<PathBuf>,
        /// VIctory's cache folder (as a Mac path): lists every image in its
        /// `images` folder that has none in its `wide` folder yet.
        #[arg(long)]
        cache: Option<PathBuf>,
        /// With `--cache`, lists the images that have a list again.
        #[arg(long, requires = "cache")]
        again: bool,
    },
    /// Compare the A and B runs of `tools/macos/bench-ab.sh`: medians,
    /// B - A and a bootstrap 95 % interval of it.
    BenchCompare {
        /// The folder of `A<n>.json` and `B<n>.json` reports.
        dir: PathBuf,
    },
}

fn main() -> Result<()> {
    let root = workspace_root();
    match Cli::parse().command {
        Cmd::Policy { staged } => policy::run(&root, staged),
        Cmd::WideStores {
            image,
            cache,
            again,
        } => match (image, cache) {
            (_, Some(cache)) => wide_stores::run_cache(&cache, again),
            (Some(image), None) => wide_stores::run(&image),
            (None, None) => unreachable!("clap requires one"),
        },
        Cmd::BenchCompare { dir } => bench_compare::run(&dir),
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
