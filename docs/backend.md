# Host GPU backends

`victory-gpu` translates the console's D3D12.X onto a host GPU. Until now that host was always D3D12: on Windows directly, and on a Mac through D3DMetal. Issue #221 proposes a second host, Metal, reached from Windows code through a Wine unixlib. The point is that Apple Silicon's memory is unified, as the console's is, so Metal can put buffers over title memory itself (`newBufferWithBytesNoCopy`). That removes the host copies of title memory, the guards that track them and the copies back, which D3DMetal cannot do (#197).

The step 0 probe for that work, which needs no game files, is in issue #221.

## The traits

`crates/victory-gpu/src/backend.rs` names what the platform code asks of the host GPU, as traits. `crate::host::Host`, the D3D12 host, implements them. D3D12 stays the first implementation and the reference that tests compare against. Each step below moves code behind a trait unchanged, so nothing changes for Windows or for the Wine and D3DMetal route.

| Trait | What it covers | Status |
| --- | --- | --- |
| `HostGpu` | What the host can run (mesh shaders, waves of 32, 64-bit heap atomics), new command lists, holding submissions off, and running a submission | Done |
| `HostMemory` | Host buffers by how the CPU and GPU share them: GPU-only, upload, readback, and shared (host copies of title memory) | Done |

## What comes next, in order

1. **Recording commands.** `HostList::list` is still a D3D12 command list, and about 200 places record into it directly, most of them in `draw.rs`, `memory.rs` and `widened.rs`. The first commands to move are copies and barriers, because the memory path is what Metal changes first. Draws, dispatches and bindings come after them.
2. **Title memory in place.** `HostMemory` gains a buffer over title memory itself, plus residency for it. On D3D12 that is `OpenExistingHeapFromAddress` where the host has it, the `mach_vm_remap` alias of #207 under D3DMetal, or nothing. On Metal it is a no-copy buffer, with all its buffers in a residency set. The probe in #221 showed that the GPU does not keep two of its addresses for one physical page coherent within a submission. Memory the title maps at several addresses therefore has to reach the GPU through one buffer, and `title_memory_aliased` already reports where that happens.
3. **Descriptors and root signatures**, as Metal argument buffers in the layout Apple's Metal Shader Converter expects.
4. **Pipelines.** DXIL from `victory-shader` goes through the Metal Shader Converter, and pipelines are kept in an `MTLBinaryArchive`, built ahead of time by `victory translate --cache --pipelines`.
5. **Textures** (detiling on the GPU from title memory) **and presentation.**
6. **Choosing the host** with a switch. D3D12 stays the default.

The Metal backend itself starts after step 2. It is first useful for title buffers, which is where the copies are.
