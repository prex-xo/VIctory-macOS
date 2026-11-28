#!/bin/zsh
# One-step setup of VIctory for macOS on an Apple Silicon Mac.
#
#   ./setup.sh
#
# It installs the tools it needs with Homebrew, builds an x86_64 Wine from
# CodeWeavers' CrossOver source, adds Apple's D3DMetal from the Game Porting
# Toolkit in your Downloads folder, and builds VIctory. Run it again at any
# time: finished steps are skipped. The first run takes about an hour.
set -eu
ROOT="${0:A:h}"
# Resolve an explicit image path relative to the caller before changing folders.
dmg="${GPTK_DMG:-}"
case "$dmg" in
  ''|/*) ;;
  *) dmg="$PWD/$dmg" ;;
esac
cd "$ROOT"
say() { print -P "%F{cyan}==>%f $*"; }
die() { print -P "%F{red}error:%f $*" >&2; exit 1; }

# 1. The Mac itself.
[[ $(uname -m) == arm64 ]] || die "this needs an Apple Silicon Mac (M1 or newer)"
(( ${$(sw_vers -productVersion)%%.*} >= 15 )) || die "this needs macOS 15 Sequoia or newer"
arch -x86_64 /usr/bin/true 2> /dev/null || die "Rosetta 2 is missing. Install it with: softwareupdate --install-rosetta"
xcode-select -p > /dev/null 2>&1 || die "the Xcode command line tools are missing. Install them with: xcode-select --install"
command -v brew > /dev/null || die "Homebrew is missing. Install it from https://brew.sh and run ./setup.sh again"
# The first run builds Wine and needs about 15 GB. Later runs reuse that build,
# so only check the space before Wine exists.
if [[ ! -x $ROOT/target/wine-cx/bin/wine ]]; then
  free_gb=$(( $(df -Pk "$ROOT" | awk 'NR == 2 { print $4 }') / 1024 / 1024 ))
  (( free_gb >= 15 )) || die "the first setup needs about 15 GB free and this disk has ${free_gb} GB. Free some space and run ./setup.sh again"
fi

# 2. Tools from Homebrew.
BREW="$(brew --prefix)"
wanted=(rustup llvm lld mingw-w64 bison flex pkgconf)
missing=()
for f in $wanted; do brew list --formula "$f" > /dev/null 2>&1 || missing+=("$f"); done
if (( ${#missing} )); then
  say "installing ${missing[*]} with Homebrew"
  # brew exits with an error when a post-install step fails, even if the
  # formula itself installed (rustup's can fail on a completion link it may
  # not write). Ask Homebrew what is installed instead of trusting the exit.
  brew install "${missing[@]}" || true
  failed=()
  for f in $missing; do brew list --formula "$f" > /dev/null 2>&1 || failed+=("$f"); done
  (( ${#failed} == 0 )) || die "Homebrew could not install ${failed[*]}. See its output above, fix that and run ./setup.sh again"
fi
export PATH="$BREW/opt/rustup/bin:$PATH"

# 3. Rust with the Windows target VIctory is built for on a Mac.
say "setting up Rust"
rustup show active-toolchain > /dev/null 2>&1 || rustup toolchain install "$(sed -n 's/^channel = "\(.*\)"/\1/p' rust-toolchain.toml)" --profile minimal
rustup target add x86_64-pc-windows-gnullvm > /dev/null

# Wine's build breaks on paths with spaces, so the large build folders live
# under a space-free link in the cache folder. There is one link for every
# checkout, so it names whichever checkout ran setup last (a fleet task's
# too): the build goes through it, and tools/macos/victory.sh and
# test-wine.sh take this checkout's own target/wine-cx first.
ALIAS="$HOME/Library/Caches/victory-macos"
mkdir -p "$ALIAS" "$ROOT/target"
ln -sfn "$ROOT/target" "$ALIAS/target"
WINE_DIR="$ALIAS/target/wine-cx"

# 4. x86_64 libraries for Wine.
say "building the x86_64 libraries Wine needs (about 10 minutes the first time)"
tools/macos/build-deps.sh "$ALIAS/target/deps-x86"

# 5. CrossOver Wine, with the patches in tools/macos/patches and SDL2 for game
# controllers. The builder decides whether it can update just the affected
# libraries or needs to reinstall the full release.
if [[ -x $WINE_DIR/bin/wine && -f $WINE_DIR/.victory-patches && -f $WINE_DIR/.victory-patch-paths && "$(cat "$WINE_DIR/.victory-patches" 2> /dev/null)" == "$(tools/macos/build-crossover-wine.sh --stamp)" ]]; then
  say "Wine already built"
elif [[ -x $WINE_DIR/bin/wine ]]; then
  say "updating the installed CrossOver Wine"
  DEPS="$ALIAS/target/deps-x86" tools/macos/build-crossover-wine.sh
else
  say "building CrossOver Wine (about 30 minutes the first time)"
  DEPS="$ALIAS/target/deps-x86" tools/macos/build-crossover-wine.sh
fi

# 6. D3DMetal from Apple's Game Porting Toolkit.
if [[ -d $WINE_DIR/lib/external/D3DMetal.framework ]]; then
  say "D3DMetal already added"
else
  if [[ -z $dmg ]]; then
    # (N) keeps zsh quiet when nothing matches, (om) puts the newest first.
    downloads=("$HOME"/Downloads/Game_Porting_Toolkit*.dmg(Nom))
    dmg="${downloads[1]:-}"
  elif [[ ! -f $dmg ]]; then
    printf 'GPTK_DMG does not point to a file: %s\n' "$dmg" >&2
    exit 1
  fi
  if [[ -z $dmg || ! -f $dmg ]]; then
    print
    print -P "%F{yellow}One manual step is left.%f Apple only gives D3DMetal to people who sign in, so you have to download it yourself:"
    print "  1. Open https://developer.apple.com/download/all/?q=game%20porting%20toolkit"
    print "  2. Sign in with any Apple account (a free one works) and download the newest Game Porting Toolkit .dmg"
    print "  3. Leave it in your Downloads folder and run ./setup.sh again"
    exit 1
  fi
  say "adding D3DMetal from ${dmg:t}"
  outer="$(mktemp -d)"; inner="$(mktemp -d)"
  # Detach both images however this step ends, so a failed run does not leave
  # the toolkit mounted for the next one.
  unmount() {
    hdiutil detach -quiet "$inner" 2> /dev/null || true
    hdiutil detach -quiet "$outer" 2> /dev/null || true
    rmdir "$inner" "$outer" 2> /dev/null || true
  }
  trap unmount EXIT
  if ! hdiutil attach -quiet -nobrowse -readonly -mountpoint "$outer" "$dmg"; then
    # An intact download that macOS would not mount is not fixed by
    # downloading it again: a sandboxed shell, such as an AI agent's, failed
    # with "Device not configured" while the disk-image service was not running.
    if hdiutil verify -quiet "$dmg" 2> /dev/null; then
      die "macOS could not mount ${dmg:t}, though the file is intact. Run ./setup.sh again; if an AI agent or another sandboxed shell runs it and it fails again, run it from your own Terminal"
    fi
    die "could not open ${dmg:t}. It may be an incomplete download: delete it, download it again and run ./setup.sh again"
  fi
  eval_dmg=("$outer"/Evaluation*.dmg(N))
  (( ${#eval_dmg} )) || die "${dmg:t} has no Evaluation .dmg inside. Check that it is Apple's Game Porting Toolkit"
  hdiutil attach -quiet -nobrowse -readonly -mountpoint "$inner" "$eval_dmg[1]" \
    || die "could not open ${eval_dmg[1]:t} inside ${dmg:t}"
  [[ -d $inner/redist/lib/external/D3DMetal.framework ]] \
    || die "${dmg:t} has no D3DMetal.framework where setup expects it (redist/lib/external). Download the newest Game Porting Toolkit and run ./setup.sh again"
  # A half-copied framework would make the next run skip this step, so remove
  # it if the copy fails.
  ditto "$inner/redist/lib/" "$WINE_DIR/lib/" || {
    rm -rf "$WINE_DIR/lib/external/D3DMetal.framework"
    die "copying D3DMetal into Wine failed. Check the free space on this disk and run ./setup.sh again"
  }
  unmount
  trap - EXIT
fi

# 7. VIctory itself.
say "building VIctory"
[[ -f victory.toml ]] || cp victory.example.toml victory.toml
# victory.sh owns the single release build: it sets VICTORY_HELPER_DXC when
# D3DMetal's DXC is there, and a separate build here without it would make
# victory-gpu's build.rs see the variable appear and rebuild and relink
# victory a second time. --version starts VIctory under Wine without reading
# victory.toml, whose example game_dir does not exist yet, so a fresh setup
# does not fail here. Stderr stays visible, and captured stdout is printed on
# failure so diagnostics from either Cargo or a custom Wine wrapper are kept.
version_out=$(tools/macos/victory.sh --version) || {
  [[ -n "$version_out" ]] && print -r -- "$version_out" >&2
  die "VIctory failed to build or start under Wine. Run tools/macos/victory.sh --version to see why"
}

print
print -P "%F{green}Setup finished.%f"
print "Put your own copy of the game in a folder, set game_dir in victory.toml, check it with"
print "  tools/macos/victory.sh config"
print "and start it with:"
print "  tools/macos/victory.sh run --execute <the game's executable>"
