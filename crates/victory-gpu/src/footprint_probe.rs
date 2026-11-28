//! A probe of which rows of macOS's `footprint` hold which D3DMetal
//! allocations, and whether memory released through D3DMetal leaves them.
//!
//! The probe makes and releases textures, heaps and buffers of 1 GiB at a
//! time, in steps. Before each step's reading it writes `phase-<n>` (its
//! name inside) into `VICTORY_FOOTPRINT_DIR` and waits for `ack-<n>`, which
//! `tools/macos/footprint-probe.sh` writes once it has read `footprint -p`
//! of the Wine process. Without the folder it runs the steps without
//! waiting (and reads nothing).
//!
//! ```sh
//! tools/macos/footprint-probe.sh
//! ```

use std::path::PathBuf;
use std::time::{Duration, Instant};

use windows::Win32::Foundation::{CloseHandle, E_FAIL, HANDLE};
use windows::Win32::Graphics::Direct3D12::*;
use windows::Win32::Graphics::Dxgi::Common::{DXGI_FORMAT_R8G8B8A8_UNORM, DXGI_SAMPLE_DESC};
use windows::Win32::System::Threading::{CreateEventW, WaitForSingleObject};
use windows::core::Interface;

const MB: u64 = 1 << 20;
/// One texture: 2048 x 2048 RGBA8, 16 MiB.
const SIDE: u32 = 2048;
/// Textures (or 64 MiB buffers, heaps) in one GiB.
const TEXTURES: usize = 64;
const CHUNK: u64 = 64 * MB;
const CHUNKS: usize = 16;

struct Steps {
    dir: Option<PathBuf>,
    n: u32,
}

impl Steps {
    /// Lets the reader take `footprint` now, named `what`.
    fn mark(&mut self, what: &str) {
        self.n += 1;
        let used = crate::host::video_memory().map_or(-1.0, |[(used, _), _]| used);
        eprintln!(
            "FOOTPRINT step {}: {what}; D3DMetal's usage {used:.0} MB",
            self.n
        );
        let Some(dir) = &self.dir else { return };
        let what = format!("{what} (usage {used:.0} MB)");
        let ack = dir.join(format!("ack-{}", self.n));
        std::fs::write(dir.join(format!("phase-{}", self.n)), &what).expect("phase file");
        let start = Instant::now();
        while !ack.exists() {
            assert!(
                start.elapsed() < Duration::from_secs(120),
                "no reader acked step {}",
                self.n
            );
            std::thread::sleep(Duration::from_millis(100));
        }
    }
}

struct Rig {
    device: ID3D12Device,
    queue: ID3D12CommandQueue,
    allocator: ID3D12CommandAllocator,
    list: ID3D12GraphicsCommandList,
    fence: ID3D12Fence,
    done: u64,
    event: HANDLE,
}

impl Drop for Rig {
    fn drop(&mut self) {
        // SAFETY: our event.
        unsafe {
            let _ = CloseHandle(self.event);
        }
    }
}

impl Rig {
    fn new(device: &ID3D12Device) -> windows::core::Result<Rig> {
        // SAFETY: plain creation of objects of our own.
        unsafe {
            let queue: ID3D12CommandQueue =
                device.CreateCommandQueue(&D3D12_COMMAND_QUEUE_DESC {
                    Type: D3D12_COMMAND_LIST_TYPE_DIRECT,
                    ..Default::default()
                })?;
            let allocator: ID3D12CommandAllocator =
                device.CreateCommandAllocator(D3D12_COMMAND_LIST_TYPE_DIRECT)?;
            let list: ID3D12GraphicsCommandList =
                device.CreateCommandList(0, D3D12_COMMAND_LIST_TYPE_DIRECT, &allocator, None)?;
            list.Close()?;
            let fence: ID3D12Fence = device.CreateFence(0, D3D12_FENCE_FLAG_NONE)?;
            let event = CreateEventW(None, false, false, None)?;
            Ok(Rig {
                device: device.clone(),
                queue,
                allocator,
                list,
                fence,
                done: 0,
                event,
            })
        }
    }

    fn run(
        &mut self,
        record: impl FnOnce(&ID3D12GraphicsCommandList),
    ) -> windows::core::Result<()> {
        // SAFETY: our own list and queue, one run at a time.
        unsafe {
            self.allocator.Reset()?;
            self.list.Reset(&self.allocator, None)?;
            record(&self.list);
            self.list.Close()?;
            self.queue.ExecuteCommandLists(&[Some(self.list.cast()?)]);
            self.done += 1;
            self.queue.Signal(&self.fence, self.done)?;
            self.fence.SetEventOnCompletion(self.done, self.event)?;
            if WaitForSingleObject(self.event, 30_000).0 != 0 {
                return Err(windows::core::Error::from_hresult(E_FAIL));
            }
            self.device.GetDeviceRemovedReason()
        }
    }
}

fn texture_desc(flags: D3D12_RESOURCE_FLAGS) -> D3D12_RESOURCE_DESC {
    D3D12_RESOURCE_DESC {
        Dimension: D3D12_RESOURCE_DIMENSION_TEXTURE2D,
        Width: SIDE.into(),
        Height: SIDE,
        DepthOrArraySize: 1,
        MipLevels: 1,
        Format: DXGI_FORMAT_R8G8B8A8_UNORM,
        SampleDesc: DXGI_SAMPLE_DESC {
            Count: 1,
            Quality: 0,
        },
        Flags: flags,
        ..Default::default()
    }
}

fn buffer_desc(size: u64) -> D3D12_RESOURCE_DESC {
    D3D12_RESOURCE_DESC {
        Dimension: D3D12_RESOURCE_DIMENSION_BUFFER,
        Width: size,
        Height: 1,
        DepthOrArraySize: 1,
        MipLevels: 1,
        SampleDesc: DXGI_SAMPLE_DESC {
            Count: 1,
            Quality: 0,
        },
        Layout: D3D12_TEXTURE_LAYOUT_ROW_MAJOR,
        ..Default::default()
    }
}

fn committed(
    device: &ID3D12Device,
    kind: D3D12_HEAP_TYPE,
    desc: &D3D12_RESOURCE_DESC,
    state: D3D12_RESOURCE_STATES,
) -> ID3D12Resource {
    let heap = D3D12_HEAP_PROPERTIES {
        Type: kind,
        ..Default::default()
    };
    let mut out: Option<ID3D12Resource> = None;
    // SAFETY: plain creation.
    unsafe {
        device.CreateCommittedResource(&heap, D3D12_HEAP_FLAG_NONE, desc, state, None, &mut out)
    }
    .expect("committed resource");
    out.expect("resource")
}

fn heap(device: &ID3D12Device, flags: D3D12_HEAP_FLAGS) -> ID3D12Heap {
    let desc = D3D12_HEAP_DESC {
        SizeInBytes: CHUNK,
        Properties: D3D12_HEAP_PROPERTIES {
            Type: D3D12_HEAP_TYPE_DEFAULT,
            ..Default::default()
        },
        Alignment: 64 << 10,
        Flags: flags,
    };
    let mut out: Option<ID3D12Heap> = None;
    // SAFETY: plain creation.
    unsafe { device.CreateHeap(&desc, &mut out) }.expect("heap");
    out.expect("heap")
}

fn placed(
    device: &ID3D12Device,
    heap: &ID3D12Heap,
    offset: u64,
    desc: &D3D12_RESOURCE_DESC,
    state: D3D12_RESOURCE_STATES,
) -> ID3D12Resource {
    let mut out: Option<ID3D12Resource> = None;
    // SAFETY: inside the heap.
    unsafe { device.CreatePlacedResource(heap, offset, desc, state, None, &mut out) }
        .expect("placed");
    out.expect("placed")
}

/// Writes every byte of `textures` (render targets) with a clear.
fn clear(rig: &mut Rig, textures: &[ID3D12Resource]) {
    // SAFETY: a descriptor heap of our own, one view per texture.
    let rtv: ID3D12DescriptorHeap = unsafe {
        rig.device
            .CreateDescriptorHeap(&D3D12_DESCRIPTOR_HEAP_DESC {
                Type: D3D12_DESCRIPTOR_HEAP_TYPE_RTV,
                NumDescriptors: textures.len() as u32,
                ..Default::default()
            })
    }
    .expect("rtv heap");
    // SAFETY: as above.
    let step = unsafe {
        rig.device
            .GetDescriptorHandleIncrementSize(D3D12_DESCRIPTOR_HEAP_TYPE_RTV)
    } as usize;
    // SAFETY: as above.
    let base = unsafe { rtv.GetCPUDescriptorHandleForHeapStart() };
    let device = rig.device.clone();
    rig.run(|list| {
        for (i, t) in textures.iter().enumerate() {
            let at = D3D12_CPU_DESCRIPTOR_HANDLE {
                ptr: base.ptr + i * step,
            };
            // SAFETY: live textures, their own slots.
            unsafe {
                device.CreateRenderTargetView(t, None, at);
                list.ClearRenderTargetView(at, &[0.25, 0.5, 0.75, 1.0], None);
            }
        }
    })
    .expect("clear");
}

/// Writes every byte of `targets` (textures or buffers in the copy-dest
/// state) with copies from a 64 MiB upload buffer of nonzero bytes.
fn fill(rig: &mut Rig, targets: &[ID3D12Resource], textures: bool) {
    let upload = committed(
        &rig.device,
        D3D12_HEAP_TYPE_UPLOAD,
        &buffer_desc(CHUNK),
        D3D12_RESOURCE_STATE_GENERIC_READ,
    );
    let mut at = std::ptr::null_mut();
    // SAFETY: a whole map of our upload buffer.
    unsafe {
        upload.Map(0, None, Some(&mut at)).expect("map");
        std::ptr::write_bytes(at.cast::<u8>(), 0x5a, CHUNK as usize);
        upload.Unmap(0, None);
    }
    rig.run(|list| {
        for t in targets {
            // SAFETY: live resources.
            unsafe {
                if textures {
                    let d = t.GetDesc();
                    let src = D3D12_TEXTURE_COPY_LOCATION {
                        pResource: std::mem::ManuallyDrop::new(Some(upload.clone())),
                        Type: D3D12_TEXTURE_COPY_TYPE_PLACED_FOOTPRINT,
                        Anonymous: D3D12_TEXTURE_COPY_LOCATION_0 {
                            PlacedFootprint: D3D12_PLACED_SUBRESOURCE_FOOTPRINT {
                                Offset: 0,
                                Footprint: D3D12_SUBRESOURCE_FOOTPRINT {
                                    Format: DXGI_FORMAT_R8G8B8A8_UNORM,
                                    Width: d.Width as u32,
                                    Height: d.Height,
                                    Depth: 1,
                                    RowPitch: d.Width as u32 * 4,
                                },
                            },
                        },
                    };
                    let dst = D3D12_TEXTURE_COPY_LOCATION {
                        pResource: std::mem::ManuallyDrop::new(Some(t.clone())),
                        Type: D3D12_TEXTURE_COPY_TYPE_SUBRESOURCE_INDEX,
                        Anonymous: D3D12_TEXTURE_COPY_LOCATION_0 {
                            SubresourceIndex: 0,
                        },
                    };
                    list.CopyTextureRegion(&dst, 0, 0, 0, &src, None);
                    drop(std::mem::ManuallyDrop::into_inner(src.pResource));
                    drop(std::mem::ManuallyDrop::into_inner(dst.pResource));
                } else {
                    list.CopyBufferRegion(t, 0, &upload, 0, CHUNK);
                }
            }
        }
    })
    .expect("fill");
}

fn settle() {
    std::thread::sleep(Duration::from_secs(3));
}

#[test]
#[ignore = "probe: run under Wine on a Mac with tools/macos/footprint-probe.sh"]
fn footprint_probe() {
    let Some(device) = crate::host::host().map(|h| h.device.clone()) else {
        eprintln!("FOOTPRINT skipped: no host GPU here");
        return;
    };
    let mut steps = Steps {
        dir: std::env::var_os("VICTORY_FOOTPRINT_DIR").map(PathBuf::from),
        n: 0,
    };
    let mut rig = Rig::new(&device).expect("rig");
    let rt = texture_desc(D3D12_RESOURCE_FLAG_ALLOW_RENDER_TARGET);
    let plain = texture_desc(D3D12_RESOURCE_FLAG_NONE);
    settle();
    steps.mark("baseline");

    // Committed render targets: made, written, released, made again.
    let mut ts: Vec<_> = (0..TEXTURES)
        .map(|_| {
            committed(
                &device,
                D3D12_HEAP_TYPE_DEFAULT,
                &rt,
                D3D12_RESOURCE_STATE_RENDER_TARGET,
            )
        })
        .collect();
    settle();
    steps.mark("1 GiB committed render targets made");
    clear(&mut rig, &ts);
    settle();
    steps.mark("1 GiB committed render targets cleared");
    ts.clear();
    settle();
    steps.mark("committed render targets released");
    rig.run(|_| {}).expect("empty");
    rig.run(|_| {}).expect("empty");
    settle();
    steps.mark("two empty submissions later");
    ts = (0..TEXTURES)
        .map(|_| {
            committed(
                &device,
                D3D12_HEAP_TYPE_DEFAULT,
                &rt,
                D3D12_RESOURCE_STATE_RENDER_TARGET,
            )
        })
        .collect();
    clear(&mut rig, &ts);
    settle();
    steps.mark("1 GiB committed render targets made again and cleared");
    ts.clear();
    settle();
    steps.mark("released again");

    // Committed sampled textures filled by copies (as host textures are).
    ts = (0..TEXTURES)
        .map(|_| {
            committed(
                &device,
                D3D12_HEAP_TYPE_DEFAULT,
                &plain,
                D3D12_RESOURCE_STATE_COPY_DEST,
            )
        })
        .collect();
    fill(&mut rig, &ts, true);
    settle();
    steps.mark("1 GiB committed plain textures filled");
    ts.clear();
    settle();
    steps.mark("committed plain textures released");

    // The same textures placed in heaps of 64 MiB (`crate::placement`).
    let heaps: Vec<_> = (0..CHUNKS)
        .map(|_| heap(&device, D3D12_HEAP_FLAG_ALLOW_ONLY_NON_RT_DS_TEXTURES))
        .collect();
    // SAFETY: a plain query.
    let size = unsafe { device.GetResourceAllocationInfo(0, &[plain]) }
        .SizeInBytes
        .next_multiple_of(64 << 10);
    let per_heap = CHUNK / size;
    eprintln!("FOOTPRINT a plain texture takes {size} bytes in a heap: {per_heap} a heap");
    settle();
    steps.mark("16 heaps of 64 MiB made");
    ts = heaps
        .iter()
        .flat_map(|h| (0..per_heap).map(move |i| (h, i * size)))
        .map(|(h, off)| placed(&device, h, off, &plain, D3D12_RESOURCE_STATE_COPY_DEST))
        .collect();
    fill(&mut rig, &ts, true);
    settle();
    steps.mark("1 GiB placed plain textures filled");
    ts.clear();
    settle();
    steps.mark("placed textures released, heaps kept");
    ts = heaps
        .iter()
        .flat_map(|h| (0..per_heap).map(move |i| (h, i * size)))
        .map(|(h, off)| placed(&device, h, off, &plain, D3D12_RESOURCE_STATE_COPY_DEST))
        .collect();
    fill(&mut rig, &ts, true);
    settle();
    steps.mark("1 GiB placed again in the same heaps and filled");
    ts.clear();
    drop(heaps);
    settle();
    steps.mark("placed textures and heaps released");

    // Default buffers filled by copies.
    let mut bs: Vec<_> = (0..CHUNKS)
        .map(|_| {
            committed(
                &device,
                D3D12_HEAP_TYPE_DEFAULT,
                &buffer_desc(CHUNK),
                D3D12_RESOURCE_STATE_COPY_DEST,
            )
        })
        .collect();
    fill(&mut rig, &bs, false);
    settle();
    steps.mark("1 GiB default buffers filled");
    bs.clear();
    settle();
    steps.mark("default buffers released");

    // Upload buffers written by the CPU.
    bs = (0..CHUNKS)
        .map(|_| {
            let b = committed(
                &device,
                D3D12_HEAP_TYPE_UPLOAD,
                &buffer_desc(CHUNK),
                D3D12_RESOURCE_STATE_GENERIC_READ,
            );
            let mut at = std::ptr::null_mut();
            // SAFETY: a whole map of our buffer.
            unsafe {
                b.Map(0, None, Some(&mut at)).expect("map");
                std::ptr::write_bytes(at.cast::<u8>(), 0x33, CHUNK as usize);
                b.Unmap(0, None);
            }
            b
        })
        .collect();
    settle();
    steps.mark("1 GiB upload buffers written");
    bs.clear();
    settle();
    steps.mark("upload buffers released");

    rig.run(|_| {}).expect("empty");
    rig.run(|_| {}).expect("empty");
    settle();
    steps.mark("two empty submissions later");

    // The peak: 2 GiB written and released, then 1 GiB.
    ts = (0..2 * TEXTURES)
        .map(|_| {
            committed(
                &device,
                D3D12_HEAP_TYPE_DEFAULT,
                &plain,
                D3D12_RESOURCE_STATE_COPY_DEST,
            )
        })
        .collect();
    fill(&mut rig, &ts, true);
    settle();
    steps.mark("2 GiB committed plain textures filled");
    ts.clear();
    for _ in 0..4 {
        rig.run(|_| {}).expect("empty");
    }
    settle();
    steps.mark("released, four empty submissions later");
    ts = (0..TEXTURES)
        .map(|_| {
            committed(
                &device,
                D3D12_HEAP_TYPE_DEFAULT,
                &plain,
                D3D12_RESOURCE_STATE_COPY_DEST,
            )
        })
        .collect();
    fill(&mut rig, &ts, true);
    settle();
    steps.mark("1 GiB committed plain textures filled");
    ts.clear();
    for _ in 0..4 {
        rig.run(|_| {}).expect("empty");
    }
    // Wait longer, for anything given back late.
    std::thread::sleep(Duration::from_secs(15));
    steps.mark("released, 15 s later");
}

/// One heap of the model below: its id, heap and free ranges.
type ModelHeap = (u64, ID3D12Heap, std::collections::BTreeMap<u64, u64>);

/// `crate::heap_fit`'s model (`emptying_sparse_heaps_holds_fewer_heaps_in_the_model`)
/// on real heaps of 64 MiB and placed textures written by copies: a
/// world's worth of textures (2 GiB), then fewer (0.7 GiB), each used in
/// half the passes and given up after three idle ones, with or without one
/// sparse heap emptied a pass (`crate::heap_fit::to_empty`). Empty heaps
/// past two are released, as `crate::placement` does. Returns the heaps
/// held at the end.
fn sparse_heaps(rig: &mut Rig, emptying: bool) -> Vec<ModelHeap> {
    use crate::heap_fit::{HeapUse, give, pick, take, to_empty};
    const K: u64 = 64 << 10;
    let device = rig.device.clone();
    let mut rng = 42u64;
    let mut next = move || {
        rng ^= rng << 13;
        rng ^= rng >> 7;
        rng ^= rng << 17;
        rng
    };
    let mut heaps: Vec<ModelHeap> = Vec::new();
    let mut ids = 0;
    // (heap id, offset, size, last pass used, texture).
    let mut held: Vec<(u64, u64, u64, u64, ID3D12Resource)> = Vec::new();
    let mut live = 0;
    let at = |heaps: &[ModelHeap], id| heaps.iter().position(|h| h.0 == id).unwrap();
    // Places a texture of `side` x `side`, written later; its heap and offset.
    let mut place = |heaps: &mut Vec<ModelHeap>, side: u32, avoid: Option<u64>| {
        let desc = texture_desc_sized(side);
        // SAFETY: a plain query.
        let size = unsafe { device.GetResourceAllocationInfo(0, &[desc]) }
            .SizeInBytes
            .next_multiple_of(K);
        let skip = avoid.map(|a| at(heaps, a));
        let i = pick(heaps.iter().map(|h| &h.2), size, K, skip).unwrap_or_else(|| {
            heaps.push((
                ids,
                heap(&device, D3D12_HEAP_FLAG_ALLOW_ONLY_NON_RT_DS_TEXTURES),
                [(0, CHUNK)].into(),
            ));
            ids += 1;
            heaps.len() - 1
        });
        let offset = take(&mut heaps[i].2, size, K).unwrap();
        let t = placed(
            &device,
            &heaps[i].1,
            offset,
            &desc,
            D3D12_RESOURCE_STATE_COPY_DEST,
        );
        (heaps[i].0, offset, size, t)
    };
    let rounds = 60;
    for round in 0..=rounds {
        let target = if round < rounds / 2 {
            2048
        } else {
            2048 - 1300 * (round - rounds / 2) / (rounds / 2)
        } * MB;
        let mut i = 0;
        while i < held.len() {
            if next() % 2 == 0 {
                held[i].3 = round;
            }
            if round > 0 && round - held[i].3 >= 3 {
                let (h, offset, size, ..) = held.swap_remove(i);
                let h = at(&heaps, h);
                give(&mut heaps[h].2, offset, size);
                live -= size;
            } else {
                i += 1;
            }
        }
        let mut emptied = None;
        let mut fresh = Vec::new();
        if emptying {
            let uses: Vec<HeapUse> = heaps
                .iter()
                .map(|(id, _, free)| HeapUse {
                    id: *id,
                    free: free.values().sum(),
                    may_go: held
                        .iter()
                        .filter(|t| t.0 == *id && t.3 < round)
                        .map(|t| t.2)
                        .sum(),
                })
                .collect();
            emptied = to_empty(&uses, CHUNK);
            if let Some(e) = emptied {
                let (gone, kept): (Vec<_>, Vec<_>) = held.drain(..).partition(|t| t.0 == e);
                held = kept;
                for (h, offset, size, last, t) in gone {
                    let side = t_side(&t);
                    drop(t);
                    let h = at(&heaps, h);
                    give(&mut heaps[h].2, offset, size);
                    if next() % 100 < 85 {
                        let (h, offset, size, t) = place(&mut heaps, side, emptied);
                        fresh.push(t.clone());
                        held.push((h, offset, size, last, t));
                    } else {
                        live -= size;
                    }
                }
            }
        }
        while live < target {
            // 128 to 2048 texels square: 64 KiB to 16 MiB, small ones most often.
            let side = 128u32 << (next() % 5).min(next() % 5);
            let (h, offset, size, t) = place(&mut heaps, side, emptied);
            fresh.push(t.clone());
            held.push((h, offset, size, round, t));
            live += size;
        }
        for chunk in fresh.chunks(256) {
            fill(rig, chunk, true);
        }
        let mut spare = 0;
        heaps.retain(|(_, _, free)| {
            if free.get(&0) == Some(&CHUNK) {
                spare += 1;
                spare <= 2
            } else {
                true
            }
        });
    }
    drop(held);
    heaps
}

fn texture_desc_sized(side: u32) -> D3D12_RESOURCE_DESC {
    D3D12_RESOURCE_DESC {
        Width: side.into(),
        Height: side,
        ..texture_desc(D3D12_RESOURCE_FLAG_NONE)
    }
}

fn t_side(t: &ID3D12Resource) -> u32 {
    // SAFETY: a live resource.
    unsafe { t.GetDesc() }.Height
}

#[test]
#[ignore = "probe: run under Wine on a Mac with tools/macos/footprint-probe.sh footprint_of_sparse_heaps"]
fn footprint_of_sparse_heaps() {
    let Some(device) = crate::host::host().map(|h| h.device.clone()) else {
        eprintln!("FOOTPRINT skipped: no host GPU here");
        return;
    };
    let mut steps = Steps {
        dir: std::env::var_os("VICTORY_FOOTPRINT_DIR").map(PathBuf::from),
        n: 0,
    };
    let mut rig = Rig::new(&device).expect("rig");
    let quiet = |rig: &mut Rig| {
        for _ in 0..2 {
            rig.run(|_| {}).expect("empty");
        }
        settle();
    };
    quiet(&mut rig);
    steps.mark("baseline");
    for emptying in [false, true] {
        let heaps = sparse_heaps(&mut rig, emptying);
        let free: u64 = heaps.iter().flat_map(|h| h.2.values()).sum();
        quiet(&mut rig);
        steps.mark(&format!(
            "model ended, {}: {} heaps, {} MB of them free",
            if emptying {
                "emptying sparse heaps"
            } else {
                "first fit only"
            },
            heaps.len(),
            free / MB
        ));
        drop(heaps);
        quiet(&mut rig);
        steps.mark("heaps released");
    }
}
