#!/bin/zsh
# Builds CodeWeavers' CrossOver Wine (GPL source, published by CodeWeavers)
# for x86_64 under Rosetta and installs it into target/wine-cx. D3DMetal is
# made for this Wine, and its memory APIs are new enough for VIctory.
#
# DEPS names an x86_64 prefix with FreeType, GnuTLS and SDL2, as
# tools/macos/build-deps.sh builds it.
#
# The patches in tools/macos/patches are applied to the source. A patch-only
# update can rebuild just the server, ntdll and HID bus (see below), and
#   tools/macos/build-crossover-wine.sh --stamp
# prints the release and patch/SDL2 stamp that setup.sh compares with the
# install's. The cached source tree is recreated when its stamp differs,
# so neither a previous release nor an old patch can stay in the cache.
set -eu
VERSION="${CROSSOVER_VERSION:-26.3.0}"
ROOT="${0:A:h:h:h}"
PATCHES=("${0:A:h}"/patches/*.patch(N))
# Keep the release readable so the partial install can check it separately.
# "with SDL2" marks the installs whose HID bus has SDL.
stamp() {
  printf '%s:' "$VERSION"
  { cat "${PATCHES[@]}" /dev/null; print 'with SDL2'; } | shasum -a 256 | cut -c1-64
}
build_stamp="$(stamp)"
if [[ ${1-} == --stamp ]]; then print -r -- "$build_stamp"; exit 0; fi
DEPS_REAL="${DEPS:?set DEPS to an x86_64 prefix with FreeType, GnuTLS and SDL2}"
# Wine's build splits flags at spaces, and both the project and the deps
# can live under folders with spaces in their names. Space-free links to
# them keep every path in the build free of spaces. The build's own link is
# keyed by a short hash of the checkout's path: the plain target link is
# shared (setup.sh points it at whichever checkout ran setup last, and
# native-build-tools.sh keeps its wrappers behind it), so a build folder
# reached through it could belong to another checkout and this script's
# rm -rf must not go through it.
ALIAS="$HOME/Library/Caches/victory-macos"
mkdir -p "$ALIAS" "$ROOT/target"
ln -sfn "$ROOT/target" "$ALIAS/target"
BUILD="$(printf '%s' "$ROOT" | shasum -a 256 | cut -c1-12)"
ln -sfn "$ROOT/target" "$ALIAS/target-$BUILD"
BUILD="$ALIAS/target-$BUILD"
# The deps are copied, and each library's recorded path is pointed at the
# copy, because a prefix built elsewhere may record a folder that is gone.
DEPS="$BUILD/deps"
# install_name_tool warns on every call that the code signature is now
# invalid (each library is signed again below). Its output is shown only when
# it fails, so a new path that does not fit in a library's header stops the
# script with the reason instead of without a word.
rename_in() {
  local out
  out="$(install_name_tool "$@" 2>&1)" || {
    print -r -- "$out" >&2
    echo "could not relink ${@[-1]:t}: libraries built before -headerpad_max_install_names must be built again (empty ${DEPS_REAL} and run tools/macos/build-deps.sh)" >&2
    exit 1
  }
}
# A copy from before SDL2 was in the deps is made again.
if [[ ! -f $DEPS/.relinked || ! -f $DEPS/lib/libSDL2-2.0.0.dylib ]]; then
  rm -rf "$DEPS"
  ditto "$DEPS_REAL" "$DEPS"
  for lib in "$DEPS"/lib/*.dylib(.N); do
    old_dir="$(otool -D "$lib" | sed -n 2p)"
    old_dir="${old_dir:h}"
    rename_in -id "$DEPS/lib/${lib:t}" "$lib"
    for ref in $(otool -L "$lib" | awk 'NR > 2 {print $1}' | grep -F "$old_dir/" || true); do
      rename_in -change "$ref" "$DEPS/lib/${ref:t}" "$lib"
    done
    codesign -f -s - "$lib" 2> /dev/null || true
  done
  touch "$DEPS/.relinked"
fi
[[ -f $DEPS/lib/libSDL2-2.0.0.dylib ]] || { echo "$DEPS_REAL has no SDL2: build it with tools/macos/build-deps.sh" >&2; exit 1; }
WORK="$BUILD/crossover"
OBJ="$WORK/build"
PREFIX="$BUILD/wine-cx"
BREW="${BREW:-/opt/homebrew}"
. "${0:A:h}/native-build-tools.sh"
x86() { arch -x86_64 /usr/bin/env PATH="$NATIVE_TOOLS:$BREW/opt/bison/bin:$BREW/opt/flex/bin:$BREW/bin:/usr/bin:/bin:/usr/sbin:/sbin" "$@"; }

mkdir -p "$WORK" "$OBJ"
TAR="$WORK/crossover-sources-$VERSION.tar.gz"
verify_source_tar() {
  [[ $VERSION != 26.3.0 ]] ||
    printf '%s  %s\n' ac99c8ca4b3848f3e81784135f023df266b61c2345726ea55a50b3e030dd6872 "$1" | shasum -a 256 -c --quiet
}
fetch_source_tar() {
  if [[ ! -f $TAR ]] || ! verify_source_tar "$TAR"; then
    (
      partial="$(mktemp "$TAR.part.XXXXXX")"
      trap 'rm -f "$partial"' EXIT
      curl -fL --retry 3 -o "$partial" "https://media.codeweavers.com/pub/crossover/source/crossover-sources-$VERSION.tar.gz"
      verify_source_tar "$partial" || { echo "checksum mismatch for ${TAR:t}" >&2; exit 1; }
      mv "$partial" "$TAR"
    )
  fi
}
SRC="$WORK/sources/wine"
# The stamp lives inside the source tree. When the release or patches
# change, fetch and verify the selected tarball before dropping the source
# and build folders. A stale build is also dropped if its source is gone.
SRC_STAMP="$SRC/.victory-patches"
if [[ -d $SRC && -f $SRC_STAMP && "$(< "$SRC_STAMP")" == "$build_stamp" ]]; then
  :
else
  if [[ -d $SRC ]]; then
    echo "the CrossOver release or patches changed: recreating the source tree"
  fi
  fetch_source_tar
  rm -rf "$OBJ" "$WORK/sources"
  tar -xzf "$TAR" -C "$WORK"
  mkdir -p "$OBJ"
fi

# CrossOver's win32u names libvulkan even when Wine is built without
# Vulkan. A fallback name lets it compile, and loading it then fails
# cleanly at run time. D3DMetal does not use Vulkan.
VK="$SRC/dlls/win32u/vulkan.c"
if ! grep -q 'victory-macos fallback' "$VK"; then
  printf '%s\n' '/* victory-macos fallback */' '#ifndef SONAME_LIBVULKAN' '#define SONAME_LIBVULKAN "libvulkan.1.dylib"' '#endif' | cat - "$VK" > "$VK.new"
  mv "$VK.new" "$VK"
fi

# The sorted paths the patches touch.
patch_paths() { grep -h '^+++ ' "$@" /dev/null | awk '{sub(/^b\//, "", $2); print $2}' | sort -u; }

# An installed Wine has D3DMetal's DLLs copied over Wine's own, and make
# install would put Wine's back. So it gets only its server, ntdll's Unix
# library and the HID bus's (winebus.so, where SDL2 support lives) rebuilt,
# which D3DMetal does not replace. That is enough only while neither the
# patches baked into the installed Wine nor the current ones change anything
# but the server and ntdll, and the installed release matches. The paths of
# the applied patches are recorded next to the install stamp.
installed=0
PATCH_PATHS="$PREFIX/.victory-patch-paths"
if [[ -x $PREFIX/bin/wine && -f $PREFIX/.victory-patches && -f $PATCH_PATHS ]]; then
  if [[ "$(< "$PREFIX/.victory-patches")" != "$VERSION:"* ]]; then
    echo "the installed CrossOver release changed or was not recorded: reinstalling Wine fully"
  else
    outside=($( { patch_paths "${PATCHES[@]}"; cat "$PATCH_PATHS"; } | grep -v -e '^server/' -e '^dlls/ntdll/unix/' -e '^$' | sort -u || true))
    if (( ${#outside} > 0 )); then
      echo "the patches change ${outside[*]}: reinstalling Wine fully"
    else
      installed=1
    fi
  fi
elif [[ -x $PREFIX/bin/wine ]]; then
  echo "the installed Wine metadata is incomplete: reinstalling it fully"
fi

# Each patch goes in once: one that already reverses cleanly is in. A patch
# is applied only after a dry run shows all of it applies, and one that
# neither applies nor reverses stops the build.
for p in $PATCHES; do
  patch -d "$SRC" -p1 -R -f -s --dry-run < "$p" > /dev/null 2>&1 && continue
  patch -d "$SRC" -p1 -f -s --dry-run < "$p" > /dev/null 2>&1 || { echo "${p:t} does not apply to CrossOver $VERSION" >&2; exit 1; }
  echo "applying ${p:t}"
  patch -d "$SRC" -p1 -f -s < "$p"
done
print -r -- "$build_stamp" > "$SRC_STAMP"

cd "$OBJ"
# A build configured before SDL2 was in the deps is configured again.
if [[ ! -f Makefile ]] || ! grep -q '^#define SONAME_LIBSDL2 ' include/config.h 2> /dev/null; then
  echo "configuring CrossOver Wine"
  # The prefix has spaces in it, so autoconf's default site file path would
  # be split at the first one. No site file is needed.
  # pipe2 is new in macOS 27. With a macOS 27 SDK configure finds it, and
  # on macOS 26 Wine then calls a missing pipe2 and crashes at its first
  # start (#50). Wine's fallback, pipe and fcntl, works on every macOS.
  CONFIG_SITE=/dev/null x86 "$SRC/configure" --prefix="$PREFIX" --enable-archs=x86_64 \
    ac_cv_func_pipe2=no \
    CC="clang -arch x86_64" CXX="clang++ -arch x86_64" \
    CFLAGS="-g -O2 -I$DEPS/include -Wno-error=incompatible-function-pointer-types -Wno-error=int-conversion" \
    LDFLAGS="-L$DEPS/lib -Wl,-rpath,@loader_path/../.." \
    PKG_CONFIG_PATH="$DEPS/lib/pkgconfig" \
    FREETYPE_CFLAGS="-I$DEPS/include/freetype2" FREETYPE_LIBS="-L$DEPS/lib -lfreetype" \
    GNUTLS_CFLAGS="-I$DEPS/include" GNUTLS_LIBS="-L$DEPS/lib -lgnutls" \
    SDL2_CFLAGS="-I$DEPS/include/SDL2 -D_THREAD_SAFE" SDL2_LIBS="-L$DEPS/lib -lSDL2" \
    --without-gstreamer --without-cups --without-sane --without-krb5 \
    --without-netapi --without-pcap --without-usb --without-v4l2 --without-capi \
    --without-gphoto --without-x --without-wayland --without-pulse --without-alsa \
    --without-oss --without-udev --without-inotify --without-vulkan --without-opencl \
    --without-ffmpeg > configure.log 2>&1 || { tail -20 configure.log; exit 1; }
  # Without SDL2 the HID bus drops game controllers: their IOHID devices
  # are left to SDL, so XInput would see none.
  grep -q '^#define SONAME_LIBSDL2 ' include/config.h || { echo "configure did not find SDL2 in $DEPS, see $OBJ/configure.log" >&2; exit 1; }
fi
if (( installed )); then
  echo "rebuilding the server, ntdll and HID bus of the installed CrossOver Wine"
  x86 make -j"$(sysctl -n hw.ncpu)" server/wineserver dlls/ntdll/ntdll.so dlls/winebus.sys/winebus.so > make.log 2>&1 || { grep -E 'error:|\*\*\*' make.log | grep -v Wno-error | head -5; exit 1; }
  rm -f "$PREFIX/.victory-patches" "$PATCH_PATHS"
  install -m 755 server/wineserver "$PREFIX/bin/wineserver"
  install -m 755 dlls/ntdll/ntdll.so "$PREFIX/lib/wine/x86_64-unix/ntdll.so"
  install -m 755 dlls/winebus.sys/winebus.so "$PREFIX/lib/wine/x86_64-unix/winebus.so"
  # install removes the old file first. A dylib written over in place keeps
  # its old code signature in the kernel, and Rosetta then fails to load it.
  install -m 755 "$DEPS/lib/libSDL2-2.0.0.dylib" "$PREFIX/lib/"
else
  echo "building CrossOver Wine"
  x86 make -j"$(sysctl -n hw.ncpu)" > make.log 2>&1 || { grep -E 'error:|\*\*\*' make.log | grep -v Wno-error | head -5; exit 1; }
  # Mark the prefix incomplete before modifying it, and invalidate any
  # D3DMetal overlay so setup.sh reapplies it after make install overwrites
  # Wine's own DLLs.
  rm -f "$PREFIX/.victory-patches" "$PATCH_PATHS"
  rm -rf "$PREFIX/lib/external/D3DMetal.framework"
  x86 make install > install.log 2>&1
  # A full reinstall can run over an existing prefix, whose libraries must
  # be replaced with install (see above), not written over in place.
  for lib in gmp.10 nettle.8 hogweed.6 gnutls.30 freetype.6 SDL2-2.0.0; do
    install -m 755 "$DEPS/lib/lib$lib.dylib" "$PREFIX/lib/"
  done
fi
patch_paths "${PATCHES[@]}" > "$PATCH_PATHS"
print -r -- "$build_stamp" > "$PREFIX/.victory-patches"
echo "installed in $PREFIX"
