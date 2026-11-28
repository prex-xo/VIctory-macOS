//! Host copies that read the title's physical pages in place
//! (`VICTORY_PHYSICAL_TILES=1`, off by default).
//!
//! The title maps its physical pages (`XMemAllocatePhysicalPages`,
//! `XMemMapPhysicalPages`) into its buffers page by page. With the switch,
//! each section of those pages is a CPU-visible GPU heap
//! ([`crate::alias::PhysicalHeap`]) and the title's views of it are remapped
//! onto the heap, so the title's CPU writes land in GPU memory. A title
//! buffer whose pages all lie on such heaps then needs no copy: its host
//! copy is a reserved buffer whose 64 KiB tiles are those pages
//! (`UpdateTileMappings`), in the title's order, wherever they lie in the
//! heaps. The copy is zero-copy ([`Shadow::new_aliased`]): nothing is
//! refreshed into it, nothing copied back, and what shaders write the
//! title reads at once, as on the console.
//!
//! When the title maps other pages there, or unmaps them, the runtime
//! passes the new tiles ([`physical_retiled`]) and the copy's tiles follow
//! at once, on the host queue: work the GPU runs from then on reads the new
//! pages, as the console's GPU reads its page tables when it runs (a title
//! does not remap pages work in flight reads).
//!
//! The run on D3DMetal is the inplace probe's `remap_cpu_visible_tiles`:
//! reserved buffers over CPU-visible heap tiles in any order, remapped,
//! both ways coherent.

use std::sync::{Arc, Mutex, OnceLock, Weak};

use windows::Win32::Graphics::Direct3D12::*;

use crate::alias::Tile;
use crate::shadow::Shadow;
use crate::watch::PAGE;

/// The runtime's lookup of the tiles showing title memory `[at, at + len)`
/// (page aligned), one per 64 KiB page in order, `None` where a page is not
/// on a GPU heap; `false` if it cannot tell.
pub type TileLookup = fn(usize, usize, &mut Vec<Option<Tile>>) -> bool;

static LOOKUP: OnceLock<TileLookup> = OnceLock::new();

/// Sets the runtime's [`TileLookup`].
pub fn on_physical_tiles(lookup: TileLookup) {
    let _ = LOOKUP.set(lookup);
}

/// A tiled copy: its first title page (page number), its reserved buffer,
/// and the tiles its pages show now (held, so their heaps outlive the
/// mapping).
struct Tiled {
    first: usize,
    copy: Weak<Shadow>,
    buffer: ID3D12Resource,
    tiles: Vec<Option<Tile>>,
}

static TILED: Mutex<Vec<Tiled>> = Mutex::new(Vec::new());

fn tiled() -> std::sync::MutexGuard<'static, Vec<Tiled>> {
    TILED.lock().unwrap_or_else(|e| e.into_inner())
}

/// A zero-copy host copy of title buffer `[start, start + len)` over its
/// physical pages, if every page of it is on a GPU heap; `None` otherwise
/// (it is copied as before). Call without the buffers' lock: the lookup
/// takes the runtime's.
pub(crate) fn make(start: u64, len: u64) -> Option<Arc<Shadow>> {
    if len == 0 || !crate::alias::physical_tiles() {
        return None;
    }
    let lookup = LOOKUP.get()?;
    let first = start as usize / PAGE;
    let pages = (start.checked_add(len)? as usize).div_ceil(PAGE) - first;
    let mut tiles = Vec::with_capacity(pages);
    if !lookup(first * PAGE, pages * PAGE, &mut tiles)
        || tiles.len() != pages
        || tiles.iter().any(Option::is_none)
    {
        return None;
    }
    let host = crate::host::host()?;
    let desc = D3D12_RESOURCE_DESC {
        Dimension: D3D12_RESOURCE_DIMENSION_BUFFER,
        Width: (pages * PAGE) as u64,
        Height: 1,
        DepthOrArraySize: 1,
        MipLevels: 1,
        SampleDesc: windows::Win32::Graphics::Dxgi::Common::DXGI_SAMPLE_DESC { Count: 1, Quality: 0 },
        Layout: D3D12_TEXTURE_LAYOUT_ROW_MAJOR,
        Flags: D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS,
        ..Default::default()
    };
    let mut buffer: Option<ID3D12Resource> = None;
    // SAFETY: plain creation with an output slot on this stack.
    if let Err(error) = unsafe {
        host.device
            .CreateReservedResource(&desc, D3D12_RESOURCE_STATE_COMMON, None, &mut buffer)
    } {
        tracing::warn!(%error, len, "no reserved buffer over physical pages; copying them");
        return None;
    }
    let buffer = buffer?;
    map(&host.queue(), &buffer, 0, &tiles);
    crate::host::name(&buffer, || format!("title buffer {start:#x}+{len:#x} on physical pages"));
    // SAFETY: a live buffer.
    let gpu = unsafe { buffer.GetGPUVirtualAddress() };
    let base = start - (first * PAGE) as u64;
    let copy = Shadow::new_aliased(buffer.clone(), start as *mut u8, gpu + base, start, len, base);
    {
        let mut list = tiled();
        list.retain(|t| t.copy.strong_count() > 0);
        list.push(Tiled {
            first,
            copy: Arc::downgrade(&copy),
            buffer,
            tiles,
        });
    }
    static FIRST: std::sync::Once = std::sync::Once::new();
    FIRST.call_once(|| tracing::info!(start = format_args!("{start:#x}"), len, "first title buffer read on its physical pages in place"));
    Some(copy)
}

/// The title's memory `[at, at + len)` now shows `tiles` (one per page; `None`
/// unmapped, or not on a GPU heap): copies over it map them at once.
pub fn physical_retiled(at: usize, len: usize, tiles: &[Option<Tile>]) {
    let mut list = tiled();
    list.retain(|t| t.copy.strong_count() > 0);
    if list.is_empty() {
        return;
    }
    let Some(host) = crate::host::host() else {
        return;
    };
    let (from, to) = (at / PAGE, at.saturating_add(len).div_ceil(PAGE));
    let queue = host.queue();
    for t in list.iter_mut() {
        let (a, b) = (from.max(t.first), to.min(t.first + t.tiles.len()));
        if a >= b {
            continue;
        }
        for page in a..b {
            t.tiles[page - t.first] = tiles.get(page - from).cloned().flatten();
        }
        if t.tiles[a - t.first..b - t.first].iter().any(Option::is_none) {
            static WARNED: std::sync::Once = std::sync::Once::new();
            WARNED.call_once(|| {
                tracing::warn!(
                    at = format_args!("{at:#x}"),
                    len,
                    "a title buffer read in place lost pages of a GPU heap: they read zeros"
                );
            });
        }
        map(&queue, &t.buffer, (a - t.first) as u32, &t.tiles[a - t.first..b - t.first]);
    }
}

/// A tile of zeros that unmapped pages show (one heap, made once).
fn zero_tile() -> Option<Tile> {
    static ZERO: OnceLock<Option<crate::alias::PhysicalHeap>> = OnceLock::new();
    ZERO.get_or_init(|| {
        let heap = crate::alias::PhysicalHeap::new(&crate::host::host()?.device, PAGE).ok()?;
        // SAFETY: the heap's own CPU view of one page.
        unsafe { (heap.cpu_address() as *mut u8).write_bytes(0, PAGE) };
        Some(heap)
    })
    .as_ref()?
    .tile(0)
}

/// Maps `buffer`'s tiles from `first` on to `tiles` (zeros where `None`), in
/// runs of one heap's consecutive tiles.
fn map(queue: &ID3D12CommandQueue, buffer: &ID3D12Resource, first: u32, tiles: &[Option<Tile>]) {
    let mut k = 0;
    while k < tiles.len() {
        let mut n = 1;
        match &tiles[k] {
            Some(tile) => {
                while tiles.get(k + n).is_some_and(|next| {
                    next.as_ref().is_some_and(|next| {
                        next.heap == tile.heap && next.index == tile.index + n as u32
                    })
                }) {
                    n += 1;
                }
                let start = D3D12_TILED_RESOURCE_COORDINATE { X: first + k as u32, ..Default::default() };
                let size = D3D12_TILE_REGION_SIZE { NumTiles: n as u32, ..Default::default() };
                let flags = D3D12_TILE_RANGE_FLAG_NONE;
                let count = n as u32;
                // SAFETY: one region of our reserved buffer onto a live heap's tiles.
                unsafe {
                    queue.UpdateTileMappings(
                        buffer,
                        1,
                        Some(&start),
                        Some(&size),
                        &tile.heap,
                        1,
                        Some(&flags),
                        Some(&tile.index),
                        Some(&count),
                        D3D12_TILE_MAPPING_FLAG_NONE,
                    );
                }
            }
            None => {
                // D3DMetal leaves a tile mapped as it was when asked to map
                // it null (`a_buffer_on_gpu_physical_pages_is_read_and_written_in_place`):
                // each goes to a tile of zeros of our own instead, as D3D12
                // reads a null tile. Shaders writing there (a title bug, as
                // the memory is unmapped) can leave other bytes in it.
                let Some(zero) = zero_tile() else {
                    k += 1;
                    continue;
                };
                let start = D3D12_TILED_RESOURCE_COORDINATE { X: first + k as u32, ..Default::default() };
                let size = D3D12_TILE_REGION_SIZE { NumTiles: 1, ..Default::default() };
                let flags = D3D12_TILE_RANGE_FLAG_NONE;
                // SAFETY: one tile of our reserved buffer onto the live zero heap.
                unsafe {
                    queue.UpdateTileMappings(
                        buffer,
                        1,
                        Some(&start),
                        Some(&size),
                        &zero.heap,
                        1,
                        Some(&flags),
                        Some(&zero.index),
                        Some(&1),
                        D3D12_TILE_MAPPING_FLAG_NONE,
                    );
                }
            }
        }
        k += n;
    }
}
