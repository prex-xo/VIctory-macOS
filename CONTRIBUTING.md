# Contributing to VIctory for macOS

Thank you for wanting to help. The goal is simple to state: GTA VI on an Apple Silicon Mac, from a copy that the player owns. There is work here for people who know Rust, graphics, Wine, macOS or none of these yet.

## The one rule

Nothing from the game is ever committed. That covers executables, DLLs, shaders, textures, audio, keys, dumps, decrypted files and screenshots of game content. Everything is built from files that the user supplies on their own machine. A check called `cargo xtask policy` enforces this on every commit, and pull requests that break it are closed. Please also do not post or link game files in issues.

D3DMetal and anything else from Apple's Game Porting Toolkit stays out of the repository too, because Apple's licence does not allow sharing it.

## Getting started

Follow the three steps in the [README](README.md) to set up your Mac. Then check that both test runs pass before you change anything.

```bash
cargo test --workspace
```

```bash
WINEDLLOVERRIDES="d3d12,d3d12core,dxgi=b" tools/macos/test-wine.sh --workspace --no-fail-fast
```

Read [docs/macos.md](docs/macos.md) for how the port works and what was already found, and [docs/architecture.md](docs/architecture.md) for VIctory itself.

## Where help is needed

These are the open areas, roughly from most to least useful.

- **World rendering.** The game reaches its world but draws little of it, on Windows too. Shader translation gaps and missing GPU features are listed in VIctory's own status table. Work on the shared crates helps every platform.
- **D3DMetal differences.** Metal and the console GPU disagree in small ways, as `sin(NaN)` showed. A test that passes on Windows and fails under D3DMetal usually points at one of these. Find it, fix it in the translator or the GPU layer, and add a test.
- **Speed.** Rosetta, Wine and D3DMetal each cost time. Profiling the frame loop on a Mac and finding where it goes is very valuable.
- **A native Mac host.** The long-term route replaces Wine with a native macOS host and a Metal backend. It is described at the end of `docs/macos.md`, and design discussion in an issue is welcome before any code.
- **Setup and packaging.** Making `setup.sh` faster or more robust, or turning it into a Mac app, helps everyone who is not a developer.
- **Testing on other Macs.** Results from M1, M2 and M3 machines, and from different macOS versions, help a lot. Open an issue with your Mac model, your macOS version and the output of both test runs. The [Mac test fleet](docs/fleet.md) does this for you: it takes a task from the issues, runs both suites and posts the counts.

## Making a change

1. Fork the repository and make a branch for your change.
2. Keep each commit small, with a short subject line that says what it changes, such as `fix(gpu): keep the clamp's budget per list`.
3. Add or update a test when you change behaviour. GPU behaviour is best tested under Wine with D3DMetal, as in the second command above.
4. Run clippy, format your changes, and run both test runs. Lint the Windows build, with and without `--all-features`, as CI does; on a Mac, plain `cargo clippy` lints the macOS build, where the Windows-only code is unused and fails as dead code:

   ```bash
   cargo clippy --workspace --all-targets --target x86_64-pc-windows-gnullvm -- -D warnings
   ```

   ```bash
   cargo clippy --workspace --all-targets --all-features --target x86_64-pc-windows-gnullvm -- -D warnings
   ```

   Not all of the code is `cargo fmt`-clean yet and CI does not check formatting, so format the code you change rather than running `cargo fmt` over everything, which would put unrelated changes in your pull request.
5. Open a pull request that says what changed, why, and how you tested it.

## Reporting a problem

Open an issue and include your Mac model, your macOS version, the version of the Game Porting Toolkit you used, and the full output of the command that failed, inside a code block. If `./setup.sh` failed, the log files it names are the most useful thing to attach.

## Licence

This project is licensed under GPL-3.0, like VIctory, which it is built on. By contributing, you agree that your work is shared under the same licence.
