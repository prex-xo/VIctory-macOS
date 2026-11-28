//! Host textures placed in heaps made ahead instead of committed one by
//! one: a new shot brings hundreds of textures, and the host spent most of
//! a frame making them (300 textures, 430 MB on the RX 6700 XT: committed
//! 255 ms; one heap 32 ms, then placed 3 ms).
//!
//! Textures without render-target, depth or unordered-access use, up to a
//! quarter of a heap, go into heaps of [`HEAP`] bytes, first fit. A
//! texture's range is free again once nothing else holds its resource
//! (heap slots keep views' resources until the running submission has run,
//! lists until they are reset), so no view or list sees its memory reused.
//! Empty heaps past [`SPARE`] are released.
//!
//! A heap is memory of the process whole until it is released, and holes
//! left by textures given up are refilled first fit in every heap, so few
//! empty. The eviction pass empties a sparse heap now and then
//! ([`choose_to_empty`], `crate::heap_fit::to_empty`): it gives up the
//! heap's textures, and nothing new is placed there while another heap has
//! room.

use std::collections::BTreeMap;
use std::sync::Mutex;

use crate::heap_fit::{HeapUse, give, pick, take};

use windows::Win32::Graphics::Direct3D12::*;
use windows::core::Interface;

/// Bytes of one heap.
const HEAP: u64 = 64 << 20;
/// Empty heaps kept for the next textures.
const SPARE: usize = 2;
/// Placement alignment of textures that are not render targets.
const ALIGN: u64 = 64 << 10;

/// Where a texture lies: heap, offset and bytes.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) struct Placement {
    heap: u64,
    offset: u64,
    size: u64,
}

/// A heap and its free ranges (offset → bytes).
struct Heap {
    heap: ID3D12Heap,
    id: u64,
    free: BTreeMap<u64, u64>,
}

#[derive(Default)]
struct Heaps {
    heaps: Vec<Heap>,
    next: u64,
    /// Ranges whose texture may still be held elsewhere, with a holder of
    /// our own.
    pending: Vec<(ID3D12Resource, Placement)>,
    /// The heap being emptied ([`choose_to_empty`]): new textures go
    /// elsewhere while another heap has room.
    emptying: Option<u64>,
}

// SAFETY: D3D12 heaps and resources are free-threaded; used under the lock.
unsafe impl Send for Heaps {}

static HEAPS: Mutex<Option<Heaps>> = Mutex::new(None);

/// Whether a resource has no holder but the one passed.
fn last_holder(resource: &ID3D12Resource) -> bool {
    // SAFETY: a live COM object; one reference taken and given back.
    unsafe {
        let raw = resource.as_raw();
        let unknown = &*(raw as *const *const windows::core::IUnknown_Vtbl);
        ((**unknown).AddRef)(raw);
        ((**unknown).Release)(raw) == 1
    }
}

impl Heaps {
    /// Frees the pending ranges nothing else holds now, and releases empty
    /// heaps past [`SPARE`].
    fn reap(&mut self) {
        let mut i = 0;
        while i < self.pending.len() {
            if last_holder(&self.pending[i].0) {
                let (_, p) = self.pending.swap_remove(i);
                if let Some(h) = self.heaps.iter_mut().find(|h| h.id == p.heap) {
                    give(&mut h.free, p.offset, p.size);
                }
            } else {
                i += 1;
            }
        }
        let empty = |h: &Heap| h.free.get(&0) == Some(&HEAP);
        // Emptied (or released): an ordinary heap again.
        if !self.heaps.iter().any(|h| Some(h.id) == self.emptying && !empty(h)) {
            self.emptying = None;
        }
        let mut spare = 0;
        let mut i = 0;
        while i < self.heaps.len() {
            if !empty(&self.heaps[i]) {
                i += 1;
                continue;
            }
            spare += 1;
            if spare <= SPARE {
                i += 1;
            } else {
                let removed = self.heaps.remove(i);
                crate::host::retire_any(removed.heap);
            }
        }
    }

    /// A range of `size` bytes: in the first heap with room, the one being
    /// emptied last, else in a new heap.
    fn allocate(&mut self, device: &ID3D12Device, size: u64, align: u64) -> Option<(ID3D12Heap, Placement)> {
        let emptying = self.heaps.iter().position(|h| Some(h.id) == self.emptying);
        if let Some(i) = pick(self.heaps.iter().map(|h| &h.free), size, align, emptying) {
            let h = &mut self.heaps[i];
            let offset = take(&mut h.free, size, align)?;
            return Some((h.heap.clone(), Placement { heap: h.id, offset, size }));
        }
        let desc = D3D12_HEAP_DESC {
            SizeInBytes: HEAP,
            Properties: D3D12_HEAP_PROPERTIES { Type: D3D12_HEAP_TYPE_DEFAULT, ..Default::default() },
            Alignment: ALIGN,
            Flags: D3D12_HEAP_FLAG_ALLOW_ONLY_NON_RT_DS_TEXTURES,
        };
        let mut heap: Option<ID3D12Heap> = None;
        // SAFETY: plain creation, our out pointer.
        unsafe { device.CreateHeap(&desc, &mut heap) }.ok()?;
        let heap = heap?;
        crate::host::name(&heap, || "texture heap".into());
        let id = self.next;
        self.next += 1;
        let mut free = BTreeMap::new();
        free.insert(0, HEAP);
        let offset = take(&mut free, size, align)?;
        self.heaps.push(Heap { heap: heap.clone(), id, free });
        Some((heap, Placement { heap: id, offset, size }))
    }
}

/// Whether a texture of `desc` goes into a heap: no render-target, depth or
/// unordered-access use, and at most a quarter of a heap.
pub(crate) fn fits(device: &ID3D12Device, desc: &D3D12_RESOURCE_DESC) -> bool {
    const USES: i32 = D3D12_RESOURCE_FLAG_ALLOW_RENDER_TARGET.0
        | D3D12_RESOURCE_FLAG_ALLOW_DEPTH_STENCIL.0
        | D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS.0
        | D3D12_RESOURCE_FLAG_ALLOW_SIMULTANEOUS_ACCESS.0;
    if desc.Dimension == D3D12_RESOURCE_DIMENSION_BUFFER || desc.Flags.0 & USES != 0 || desc.SampleDesc.Count > 1 {
        return false;
    }
    // SAFETY: a plain query.
    let info = unsafe { device.GetResourceAllocationInfo(0, &[*desc]) };
    info.SizeInBytes != u64::MAX && info.SizeInBytes <= HEAP / 4 && info.Alignment <= ALIGN
}

/// Room for a texture of `desc` in a heap ([`fits`] first): the heap and
/// the placement, which the texture gives back with [`free`].
pub(crate) fn place(device: &ID3D12Device, desc: &D3D12_RESOURCE_DESC) -> Option<(ID3D12Heap, Placement)> {
    // SAFETY: a plain query.
    let info = unsafe { device.GetResourceAllocationInfo(0, &[*desc]) };
    let size = info.SizeInBytes.next_multiple_of(ALIGN);
    let mut heaps = HEAPS.lock().unwrap_or_else(|e| e.into_inner());
    let heaps = heaps.get_or_insert_with(Heaps::default);
    heaps.reap();
    heaps.allocate(device, size, info.Alignment.max(ALIGN))
}

/// A range [`place`] gave whose texture could not be made there.
pub(crate) fn unplace(p: Placement) {
    let mut heaps = HEAPS.lock().unwrap_or_else(|e| e.into_inner());
    if let Some(h) = heaps.as_mut().and_then(|hs| hs.heaps.iter_mut().find(|h| h.id == p.heap)) {
        give(&mut h.free, p.offset, p.size);
    }
}

/// The texture at `p` is let go of: its range is free again once nothing
/// else holds `resource` (checked at the next placements and passes).
pub(crate) fn free(resource: ID3D12Resource, p: Placement) {
    let mut heaps = HEAPS.lock().unwrap_or_else(|e| e.into_inner());
    if let Some(h) = heaps.as_mut() {
        h.pending.push((resource, p));
        h.reap();
    }
}

/// Bytes of heap space free for textures (after freeing what nothing
/// holds now): the card's memory they take is not in use.
pub(crate) fn free_bytes() -> u64 {
    let mut heaps = HEAPS.lock().unwrap_or_else(|e| e.into_inner());
    let Some(h) = heaps.as_mut() else { return 0 };
    h.reap();
    h.heaps.iter().flat_map(|h| h.free.values()).sum()
}

/// Heaps of textures held, and their used and free bytes (after freeing
/// what nothing holds now), for the statistics line.
pub(crate) fn stats() -> (usize, u64, u64) {
    let mut heaps = HEAPS.lock().unwrap_or_else(|e| e.into_inner());
    let Some(h) = heaps.as_mut() else { return (0, 0, 0) };
    h.reap();
    let free: u64 = h.heaps.iter().flat_map(|h| h.free.values()).sum();
    (h.heaps.len(), h.heaps.len() as u64 * HEAP - free, free)
}

/// Picks the heap to empty (`crate::heap_fit::to_empty`), with
/// `may_go(heap)` the bytes of its textures that may be given up now, and
/// places nothing new there while another heap has room. Returns its id,
/// whose textures the caller gives up.
pub(crate) fn choose_to_empty(may_go: impl Fn(u64) -> u64) -> Option<u64> {
    let mut heaps = HEAPS.lock().unwrap_or_else(|e| e.into_inner());
    let h = heaps.as_mut()?;
    h.reap();
    let uses: Vec<HeapUse> = h
        .heaps
        .iter()
        .map(|heap| HeapUse { id: heap.id, free: heap.free.values().sum(), may_go: may_go(heap.id) })
        .collect();
    h.emptying = crate::heap_fit::to_empty(&uses, HEAP);
    h.emptying
}

impl Placement {
    /// Its offset in its heap.
    pub(crate) fn offset(&self) -> u64 {
        self.offset
    }

    /// Its heap's id ([`choose_to_empty`]).
    pub(crate) fn heap(&self) -> u64 {
        self.heap
    }

    /// Its bytes in the heap.
    pub(crate) fn size(&self) -> u64 {
        self.size
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A texture's range is not handed out again while anything else holds
    /// its resource (a view's slot, a list), and is once nothing does.
    #[test]
    fn a_range_is_reused_only_once_nothing_holds_its_texture() {
        use windows::Win32::Graphics::Dxgi::Common::*;
        let Some(host) = crate::host::host() else {
            return; // no host GPU here
        };
        let desc = D3D12_RESOURCE_DESC {
            Dimension: D3D12_RESOURCE_DIMENSION_TEXTURE2D,
            Width: 256,
            Height: 256,
            DepthOrArraySize: 1,
            MipLevels: 9,
            Format: DXGI_FORMAT_BC1_UNORM,
            SampleDesc: DXGI_SAMPLE_DESC { Count: 1, Quality: 0 },
            ..Default::default()
        };
        assert!(fits(&host.device, &desc));
        let mut heaps = Heaps::default();
        let (heap, a) = heaps.allocate(&host.device, ALIGN, ALIGN).unwrap();
        let mut texture: Option<ID3D12Resource> = None;
        // SAFETY: a range of the heap, our out pointer.
        unsafe { host.device.CreatePlacedResource(&heap, a.offset(), &desc, D3D12_RESOURCE_STATE_COMMON, None, &mut texture) }.unwrap();
        let texture = texture.unwrap();
        let held = texture.clone();
        heaps.pending.push((texture, a));
        heaps.reap();
        let (_, b) = heaps.allocate(&host.device, ALIGN, ALIGN).unwrap();
        assert_ne!(b, a, "still held: not handed out");
        drop(held);
        heaps.reap();
        let (_, c) = heaps.allocate(&host.device, ALIGN, ALIGN).unwrap();
        assert_eq!(c, a, "nothing holds it: free again");
    }

    /// New ranges go elsewhere while a heap is emptied, there only when no
    /// other heap has room, and the heap is an ordinary one once empty.
    #[test]
    fn nothing_new_goes_into_the_heap_being_emptied() {
        let Some(host) = crate::host::host() else {
            return; // no host GPU here
        };
        let mut heaps = Heaps::default();
        let (_, a) = heaps.allocate(&host.device, HEAP / 2, ALIGN).unwrap();
        let (_, b) = heaps.allocate(&host.device, HEAP * 3 / 4, ALIGN).unwrap();
        assert_ne!(a.heap, b.heap, "no room in the first: a second heap");
        heaps.emptying = Some(a.heap);
        let (_, c) = heaps.allocate(&host.device, HEAP / 8, ALIGN).unwrap();
        assert_eq!(c.heap, b.heap, "not into the heap emptied");
        let (_, d) = heaps.allocate(&host.device, HEAP / 4, ALIGN).unwrap();
        assert_eq!(
            d.heap, a.heap,
            "no other heap has room: there, not a new heap"
        );
        assert_eq!(heaps.heaps.len(), 2);
        heaps.reap();
        assert_eq!(heaps.emptying, Some(a.heap), "not empty yet");
        for p in [a, d] {
            give(&mut heaps.heaps[0].free, p.offset, p.size);
        }
        heaps.reap();
        assert_eq!(heaps.emptying, None, "empty: an ordinary heap again");
    }
}
