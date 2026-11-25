# macOS port status

This file tracks the work to run VIctory on Apple Silicon Macs. No game files are used or needed for any step that is marked as done.

## Done on 24 November 2026

The whole workspace builds natively on macOS (aarch64, Homebrew Rust 1.100). The Windows-only parts are gated with `cfg(windows)`, and a few small stubs stand in for them. The text entry dialog is cancelled on other hosts, a relaunch request stops the process, and the `translate` command reports that it needs Windows. On a Mac M4 the test suite gives 162 passed, 0 failed and 10 ignored. The ignored tests read their own test binary as a PE file or use Windows paths, so they only run on Windows.

A native build proves that the platform code (formats, shader recompiler, PE analysis and most of the HLE layer) is portable. It does not run a title, because the loader maps a Windows PE image into the process and calls Windows APIs.

## Done on 24 November 2026: the Windows build under Wine

VIctory now cross builds on the Mac for `x86_64-pc-windows-gnullvm` and runs under Wine 11.14. The `windows-gnu` target cannot be used, because Rust keeps its thread-locals in OS TLS slots there and the loader only saves and restores static TLS, so std aborted when a thread started. Homebrew has no llvm-mingw, so `tools/macos/gnullvm-cc.sh` links with Homebrew clang and lld against the mingw-w64 runtime.

Run the tests under Wine with this command.

```sh
WINE=/path/to/wine tools/macos/test-wine.sh --workspace --no-fail-fast
```

The loader probe passes, so the stand-in title is mapped, bound, delay loaded and run with its thread callbacks. The HLE tests pass (86 of 86), and `victory.exe --help` runs. Wine lacks three network exports (`WSAConnectByList`, `GetAddrInfoExA` and `WinHttpGetProxySettingsEx`), and VIctory now fails those calls cleanly. Two kinds of test still fail under Wine.

- Seven shader tests need a D3D12 device with shader model 6.6. The Wine build has no D3D12 that works on Metal, so this is the GPU gap that D3DMetal has to fill.
- The xtask policy test starts git from inside Wine. Run `cargo xtask policy` natively instead.

## D3DMetal results, 24 November 2026

With the GPTK cask (`brew install --cask gcenx/wine/game-porting-toolkit`, Wine 7.7 with the D3DMetal of GPTK 1.1), VIctory gets a D3D12 device on Metal, and all 69 shader tests pass on an M4. Metal's sin and cos return 0 for NaN and infinity, where the console GPU returns NaN, so the translator now passes that NaN through. D3DMetal does not check the DXIL container hash.

Wine 7.7 is too old for the rest of VIctory. The console memory tests (xmem and dstorage) and the display test fail on it, and it also lacks `WSAConnectByNameA`. Wine 11.14 passes all of them, but D3DMetal from GPTK 4.0 beta 3 crashes in libsystem_pthread on Wine 11.14, because D3DMetal needs Apple's or CrossOver's patched Wine. A Wine that is new enough for VIctory and also works with D3DMetal is the open problem.

```sh
WINE="/Applications/Game Porting Toolkit.app/Contents/Resources/wine/bin/wine64" WINEDLLOVERRIDES=winedbg.exe=d tools/macos/test-wine.sh --workspace --no-fail-fast
```

## CrossOver Wine with D3DMetal, 24 November 2026

`tools/macos/build-crossover-wine.sh` builds CodeWeavers' CrossOver 26.3 Wine (Wine 11.0) from its published source into `target/wine-cx`, using x86_64 FreeType and GnuTLS from `DEPS`. Copy the `redist/lib` folder of Apple's "Evaluation environment for Windows games" over `target/wine-cx/lib`, and run the tests with this command.

```sh
WINE="$HOME/Library/Caches/victory-macos/target/wine-cx/bin/wine" WINEDLLOVERRIDES="d3d12,d3d12core,dxgi=b" tools/macos/test-wine.sh --workspace --no-fail-fast
```

On an M4 with D3DMetal 4.0 beta 3 this gives 260 passed, 4 failed and 4 ignored. Every loader, HLE, console memory and shader test passes on the same Wine. The policy test fails only because it runs git inside Wine. Three GPU tests show real D3DMetal differences that still need work.

- `buffer_data_reaches_the_host_texture`: fixed. The upload was queued as a copy into title memory, and that check uses VirtualQuery, which cannot see D3DMetal's mappings under Wine, so the copy was skipped. It is now queued as a fill of the list's upload memory.
- `argument_clamp_builds`: fixed. Wine's HLSL compiler (vkd3d-shader) cannot compile the clamp shader, so the `clamp bytecode` workflow compiles `clamp.hlsl` with FXC and DXC on a Windows runner, and `cargo xtask embed-clamp` embeds the result with a hash of the source.
- `a_rejected_blob_gives_an_empty_library`: fixed. D3DMetal accepts any pipeline library bytes, so the cache file now carries a mark and a checksum that VIctory checks itself.

After these fixes the whole suite passed under CrossOver Wine with D3DMetal on an M4: 343 passed, 0 failed and 4 ignored. After the merge of 25 November 2026 and the fixes below it gives 481 passed, 0 failed and 7 ignored.

## Run-time HLSL and textures under Wine and D3DMetal

Under Wine, `D3DCompile` is Wine's HLSL compiler (vkd3d-shader). In CrossOver 26.3 Wine it differs from Microsoft's compilers in four ways that matter here:

- It ignores `[RootSignature]` (warning W5302) and writes no root signature into the bytecode, so `CreateRootSignature` on that bytecode fails with `E_FAIL`. Build run-time compute shaders with `compute::pipeline` in `victory-gpu`, which serializes the root signature itself.
- `RWByteAddressBuffer` has no `Load`, and `RWStructuredBuffer` stores are not implemented. The depth converter in `twin.rs` reads its scratch buffer through a structured view and writes it through a raw one.
- Shader model 5.1 resource arrays are not implemented, bounded or unbounded (`E_NOTIMPL`), and `NonUniformResourceIndex` is not defined. Shaders that need them have to be compiled on Windows, as the clamp shader is. The one test that needs them (`lists_share_a_table_and_copies_keep_their_views`) prints that it was skipped.
- It still compiles plain `cs_5_1` shaders, and D3DMetal runs their bytecode.

D3DMetal has no Metal format for `R10G10B10_XR_BIAS_A2_UNORM`. It refuses any castable format list that names it, and creating a texture in that format stops the process in Metal's validation. `castable` leaves it out, so 4-byte colour textures of one memory share one host texture under D3DMetal too.

D3DMetal rounds float depth down when it stores 24-bit depth (a clear to 0.5 stores 8388607, not 8388608), where Windows GPUs round to nearest.

## Route A: next steps

The first playable route is the upstream Windows build running under Wine, with D3D12 going to Metal through Apple's D3DMetal. DXMT cannot be used because it covers D3D11 only. Rosetta 2 runs the title's x86-64 code and has supported AVX2 since macOS 15.

The steps are as follows.

1. Install rustup and add the `x86_64-pc-windows-gnu` or `x86_64-pc-windows-msvc` target. The MSVC route through `cargo-xwin` downloads the Microsoft CRT and SDK under Microsoft's licence.
2. Build `apps/victory` and the `tests/fixtures/hle-probe` stand-in title for Windows.
3. Run the probe under Wine to test the loader and HLE layer without any game files.
4. Add D3DMetal from Apple's Game Porting Toolkit and check that VIctory creates a D3D12 device with mesh shaders and shader model 6.6.

## Route B: a native macOS host

Later work replaces the Windows host underneath the title. That needs a Rosetta x86-64 process that maps the PE image, a macOS version of the host pass-through in `victory-hle`, a Metal backend for `victory-gpu`, and DXIL to Metal conversion with Apple's Metal shader converter. This only makes sense after Route A shows where things break.

## Running VIctory on a Mac

Run `./setup.sh` from the top of the repository. It checks the Mac, installs the Homebrew tools, builds checksum-verified x86_64 GMP, Nettle, GnuTLS and FreeType, builds CrossOver Wine, adds D3DMetal from the Game Porting Toolkit in Downloads, and builds VIctory. The only manual step is downloading the toolkit from Apple. A clean run on an M4 on 24 November 2026 finished without help, and the full suite then passed (343 passed, 0 failed).

## D3DMetal's shader cache needs unique shader hashes

D3DMetal keeps a compiled-shader cache for each executable, in `$(getconf DARWIN_USER_CACHE_DIR)d3dm/<exe name>/shaders.cache`. It finds shaders by the DXIL container's shader hash (the `HASH` part), apparently together with the shader's length: two shaders collided only when their lengths matched. VIctory used to write the same all-zero shader hash into every container. So two different shaders of the same length shared a cache entry, and the one compiled second ran the first one's code. The cache is on disk, so the wrong code also came back on every later run of that executable.

This was the "signed zero in minimums" difference listed here before. Each `v_min_f32` and `v_min3_f32` test shader is the same length as the maximum built just before it, so the minimum ran the maximum's code and gave +0 where the console gives -0. Selecting on the integer bits could not help, because the shader that ran was never the one being changed. It also made `v_cmp_class_f16_matches_rdna2` fail whenever the max/min test ran before it, as it does with `--test-threads=1`. The `v_max3_f32` shader and that test's first shader are both 2316 bytes.

`container::shader_hash` now writes the MD5 of the program into the `HASH` part. The minimum cases run under Wine again, and both tests pass in every order on an M5 Pro with D3DMetal 4.0 beta 3. Cache entries written under the old zero hash are never looked up again, so old caches need no cleaning.

## Open differences under D3DMetal

The merge of VIctory's changes up to 70564ae on 25 November 2026 brought new shader tests, and three of them find real differences in D3DMetal 4.0 beta 3 on an M4. The tests skip only those checks under Wine, so the rest of each test still runs.

- 64-bit image atomics. D3DMetal reports `AtomicInt64OnTypedResourceSupported` and `AtomicInt64OnDescriptorHeapResourceSupported`, so `victory_gpu` builds pipelines that use them, but the atomics in `image_atomics_of_64_bits` leave every texel at zero. The game's alpha prepasses use them, so those prepasses will be wrong on a Mac until the translator emulates them or D3DMetal is fixed. The test host now also checks the two features, as the runtime does.
- Sample bias in compute programs. In `image_addresses_of_16_bits_and_clamps`, a sample with a bias of 1 reads level 1 where the console reads level 2, so the bias is ignored. The gather and clamp cases of that test pass.
- 24-bit depth. Apple GPUs have no 24-bit depth format, so D3DMetal keeps it as 32-bit float. A depth clear of 0.5 then reads back one step lower once VIctory converts it to the console's 16 bits (32767 where Windows gives 32768), so `a_depth_texture_written_through_unordered_views_reads_as_depth` allows one step there.
- Gather positions. D3DMetal's gather truncates the position where the console rounds it, as WARP's does, so `gathers_at_a_level_match_the_hardware` skips the same comparison it already skipped for WARP.

The seven victory-gpu tests that compile HLSL at run time now take the bytes that Microsoft's FXC made on a Windows runner (the test-shaders workflow and `cargo xtask embed-test-shaders`), because Wine's compiler cannot build them. The depth conversion in `twin.rs` goes through the same table, so depth textures written through unordered views are converted on the Mac too.
