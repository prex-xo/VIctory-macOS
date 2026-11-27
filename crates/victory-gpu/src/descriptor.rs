//! Descriptor heaps: zeroed memory, with handles that are plain addresses
//! (CPU and GPU alike, as on the console).
//!
//! Descriptor sizes: views 32 bytes and samplers 16, the size of RDNA 2's
//! image and sampler descriptors, so a title that copies descriptors itself
//! moves whole ones. Render target and depth views are ours alone; 128 bytes
//! leaves room for whatever stride a title assumes.
#![allow(clippy::not_unsafe_ptr_arg_deref)]

#[cfg(windows)]
use crate::backend::HostGpu;
use crate::object::{self, DESCRIPTOR_HEAP, E_INVALIDARG, HRESULT, Iid, Object, Table, This, guid};
use crate::resource::{Memory, PAGEABLE_IIDS};

#[cfg(windows)]
pub(crate) mod host;
// Also built for tests, which run where `host` cannot.
#[cfg(any(windows, test))]
mod paged;

const HEAP_IIDS: [Iid; 4] = [
    guid(
        0x8efb471d,
        0x616c,
        0x4f49,
        [0x90, 0xf7, 0x12, 0x7b, 0xb7, 0x63, 0xfa, 0x51],
    ),
    PAGEABLE_IIDS[0],
    PAGEABLE_IIDS[1],
    PAGEABLE_IIDS[2],
];

#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub(crate) struct HeapDesc {
    kind: u32,
    count: u32,
    flags: u32,
    nodes: u32,
}

/// Newer titles (Grand Theft Auto VI) pass the description followed by
/// memory of their own for the descriptors: an address and its size in
/// bytes, or a null address to have the heap allocate (inferred from the
/// title's call sites; older titles pass the plain description).
#[repr(C)]
#[derive(Debug, Clone, Copy)]
struct Placement {
    address: u64,
    size: u64,
}

#[repr(C)]
pub(crate) struct Heap {
    object: Object,
    desc: HeapDesc,
    /// The first descriptor: in `memory`, or in the title's own memory.
    pub(crate) start: u64,
    memory: Option<Memory>,
    /// Shader-visible view and sampler heaps get a host heap with the same
    /// descriptor order when a draw first uses them (decision 0007, step
    /// 3); slots written before wait in `pending`.
    #[cfg(windows)]
    host: std::sync::OnceLock<Option<host::HostHeap>>,
    #[cfg(windows)]
    pending: std::sync::Mutex<Vec<u32>>,
    /// A shader-visible heap in the title's memory, tracked: the title also
    /// copies descriptors into it itself, as bytes (`sync_heaps`).
    #[cfg(windows)]
    written: Option<crate::watch::Tracked>,
}

#[cfg(windows)]
pub(crate) use host::fallback_samplers;

/// A title heap's host copy.
pub(crate) fn heap<'a>(this: This) -> &'a Heap {
    // SAFETY: called from descriptor heap methods only.
    unsafe { &*this.cast::<Heap>() }
}

impl Heap {
    /// Whether this is a sampler heap.
    pub(crate) fn samplers(&self) -> bool {
        self.desc.kind == SAMPLERS
    }

    /// The host heap, made on first use from what the title wrote so far.
    /// Made late because a host view heap the size of Grand Theft Auto VI's
    /// costs over 100 MB of commit, which its boot needs for its page pool.
    #[cfg(windows)]
    pub(crate) fn host(&self) -> Option<&host::HostHeap> {
        if let Some(h) = self.host.get() {
            return h.as_ref();
        }
        if self.desc.flags & SHADER_VISIBLE == 0 || !matches!(self.desc.kind, VIEWS_KIND | SAMPLERS)
        {
            return None;
        }
        // Submissions held off while a view heap is made and filled
        // (`Host::quiet`), from before `pending` is taken: `mirror` waits
        // for that with `SHADER_HEAPS` held, which a submission waits for.
        let _quiet = (self.desc.kind == VIEWS_KIND)
            .then(|| crate::host::host()?.quiet())
            .flatten();
        let mut pending = self.pending.lock().unwrap_or_else(|e| e.into_inner());
        // A recorder can take the fast path as soon as the OnceLock is
        // initialized. Publish only after replaying the queued writes.
        self.host
            .get_or_init(|| {
                let step = u64::from(increment(self.desc.kind)?);
                let h = host::HostHeap::new(self.samplers(), self.desc.count)?;
                for i in pending.drain(..) {
                    h.mirror(i, self.start + u64::from(i) * step, false);
                }
                Some(h)
            })
            .as_ref()
    }

    /// Number of descriptors in this heap.
    pub(crate) fn count(&self) -> u32 {
        self.desc.count
    }

    /// The descriptor index of `handle` in this heap, if it lies in it.
    pub(crate) fn index(&self, handle: u64) -> Option<u32> {
        let step = u64::from(increment(self.desc.kind)?);
        let at = handle.checked_sub(self.start)?;
        let index = at / step;
        (index < u64::from(self.desc.count)).then_some(index as u32)
    }

    /// The descriptor handle for title slot `index` in this heap, if in range.
    #[cfg(windows)]
    pub(crate) fn handle(&self, index: u32) -> Option<u64> {
        let step = u64::from(increment(self.desc.kind)?);
        (index < self.desc.count).then(|| self.start + u64::from(index) * step)
    }

    /// The host heap if it has already been created (never allocates or calls
    /// `Host::quiet`).
    #[cfg(windows)]
    pub(crate) fn made_host(&self) -> Option<&host::HostHeap> {
        self.host.get()?.as_ref()
    }
}

const VIEWS_KIND: u32 = 0;
const SAMPLERS: u32 = 1;
const SHADER_VISIBLE: u32 = 1;

/// Shader-visible heaps by address range, for mirroring descriptor writes:
/// start, end, heap.
#[cfg(windows)]
static SHADER_HEAPS: std::sync::RwLock<Vec<(u64, u64, usize)>> = std::sync::RwLock::new(Vec::new());

#[cfg(any(windows, test))]
type Registration = (u64, u64, usize);

/// Registers a heap, retiring any range it overlaps. Descriptor handles carry
/// no heap identity, so two live heaps cannot unambiguously own one address.
#[cfg(any(windows, test))]
fn register_heap(heaps: &mut Vec<Registration>, (start, end): (u64, u64), at: usize) {
    heaps.retain(|&(old_start, old_end, _)| old_end <= start || end <= old_start);
    heaps.push((start, end, at));
}

/// Removes only this heap's registration.
#[cfg(any(windows, test))]
fn unregister_heap(heaps: &mut Vec<Registration>, at: usize) {
    heaps.retain(|&(_, _, heap)| heap != at);
}

#[cfg(any(windows, test))]
static AMBIGUOUS_RANGE: std::sync::Once = std::sync::Once::new();

/// The heap owning `handle`, if exactly one registered range contains it.
#[cfg(any(windows, test))]
fn heap_at(heaps: &[Registration], handle: u64) -> Option<usize> {
    let mut found = None;
    for &(start, end, at) in heaps {
        if (start..end).contains(&handle) && found.replace(at).is_some() {
            AMBIGUOUS_RANGE.call_once(|| {
                tracing::warn!(
                    handle,
                    "overlapping descriptor heaps: descriptor write refused"
                )
            });
            return None;
        }
    }
    found
}

/// Makes the host descriptor for what the title wrote at `handle`, if it
/// is in a shader-visible heap. Keeps `SHADER_HEAPS` locked for the duration
/// so `Heap::drop` cannot free the heap concurrently.
#[cfg(windows)]
fn mirror(handle: u64) {
    #[cfg(feature = "diagnostics")]
    let waited = crate::watch::Timed(std::time::Instant::now(), crate::copystat::C::MirrorWaitUs);
    let heaps = SHADER_HEAPS.read().unwrap_or_else(|e| e.into_inner());
    #[cfg(feature = "diagnostics")]
    drop(waited);
    let Some(at) = heap_at(&heaps, handle) else {
        return;
    };
    // SAFETY: registered heaps stay alive until `Heap::drop`, which takes
    // the `SHADER_HEAPS` lock we hold here.
    let heap = unsafe { &*(at as *const Heap) };
    let Some(index) = heap.index(handle) else {
        return;
    };
    let mut pending = heap.pending.lock().unwrap_or_else(|e| e.into_inner());
    match heap.host.get() {
        None => pending.push(index),
        Some(host) => {
            drop(pending);
            if let Some(host) = host {
                host.mirror(index, handle, false);
            }
        }
    }
}

/// Mirrors what the title wrote into its shader-visible heaps without a
/// call: Grand Theft Auto VI copies descriptors (our records) into its heap
/// as bytes, so only a change of its memory shows them. Each slot of a page
/// written since the last sync is made again if its record changed (in the
/// world 1,300-14,400 slots a frame, against a few hundred views made by
/// calls). Called before each submission: shaders take descriptors from the
/// heap at indices in their data, where a slot never mirrored reads nothing.
#[cfg(windows)]
pub(crate) fn sync_heaps() {
    /// A heap's start, its address, and the ranges written since the last sync.
    type Written = (u64, usize, Vec<(usize, usize)>);
    // The written ranges of every heap, taken in one short hold of the lock.
    let work: Vec<Written> = {
        let heaps = SHADER_HEAPS.read().unwrap_or_else(|e| e.into_inner());
        heaps
            .iter()
            .filter_map(|&(start, _, at)| {
                // SAFETY: registered heaps are live (removed when dropped).
                let heap = unsafe { &*(at as *const Heap) };
                // As before: ranges are taken only once the heap has its
                // host heap, so writes before that are not lost.
                let (Some(Some(_)), Some(written), Some(_)) =
                    (heap.host.get(), &heap.written, increment(heap.desc.kind))
                else {
                    return None;
                };
                Some((start, at, written.take(start as usize, usize::MAX)))
            })
            .collect()
    };
    // Then page by page, the lock held for one page at a time: title
    // threads writing views (`mirror`) wait on it, and held across every
    // page of every heap they waited ~100 ms a frame in the world (some
    // 650,000 slots checked a frame, 29 submissions). A heap is not dropped
    // while its page is made: `Heap::drop` takes the write lock to leave the
    // list, so a heap still listed under the lock is alive.
    #[cfg(windows)]
    thread_local! {
        static SYNC_PAGE_BUF: std::cell::RefCell<Vec<u8>> = const { std::cell::RefCell::new(Vec::new()) };
    }
    let sync_range = |host: &host::HostHeap,
                      count: u32,
                      start: u64,
                      step: u64,
                      range_from: usize,
                      range_to: usize,
                      bytes: &[u8]| {
        let (first, last) = (
            (range_from as u64 - start) / step,
            (range_to as u64 - start).div_ceil(step),
        );
        let last = last.min(u64::from(count));
        if first >= last {
            return;
        }
        #[cfg(feature = "diagnostics")]
        crate::copystat::add(crate::copystat::C::SyncSlots, last.saturating_sub(first));
        host.sync_slots_bytes(
            first as u32,
            last as u32,
            start,
            step,
            range_from as u64,
            bytes,
        );
    };

    const SUB_PAGE: usize = 4 << 10;
    let page_size = crate::watch::PAGE;
    let mut page_buf = SYNC_PAGE_BUF.with(|b| std::mem::take(&mut *b.borrow_mut()));
    if page_buf.len() < page_size {
        page_buf.resize(page_size, 0);
    }
    for (start, at, ranges) in work {
        for (from, to) in ranges {
            for page in (from & !(page_size - 1)..to).step_by(page_size) {
                let heaps = SHADER_HEAPS.read().unwrap_or_else(|e| e.into_inner());
                if !heaps.iter().any(|&(s, _, h)| (s, h) == (start, at)) {
                    break;
                }
                // SAFETY: listed under the lock we hold, so not yet dropped.
                let heap = unsafe { &*(at as *const Heap) };
                let (Some(Some(host)), Some(step)) = (heap.host.get(), increment(heap.desc.kind))
                else {
                    break;
                };
                let step = u64::from(step);
                let heap_end = start.saturating_add(u64::from(heap.desc.count).saturating_mul(step));
                let (page, end) = (
                    page.max(from),
                    (page + page_size).min(to).min(heap_end as usize),
                );
                if page >= end {
                    continue;
                }
                let len = end - page;
                // By page: read the whole mapped range once; if part of a 64 KiB
                // page is unmapped, fall back to 4 KiB OS pages so mapped slots
                // are never skipped.
                if crate::host::read_into(page as *const u8, &mut page_buf[..len]) {
                    sync_range(host, heap.desc.count, start, step, page, end, &page_buf[..len]);
                } else {
                    for sub in (page & !(SUB_PAGE - 1)..end).step_by(SUB_PAGE) {
                        let (sub_from, sub_to) = (sub.max(page), (sub + SUB_PAGE).min(end));
                        let sub_len = sub_to - sub_from;
                        if crate::host::read_into(sub_from as *const u8, &mut page_buf[..sub_len]) {
                            sync_range(
                                host,
                                heap.desc.count,
                                start,
                                step,
                                sub_from,
                                sub_to,
                                &page_buf[..sub_len],
                            );
                        }
                    }
                }
            }
        }
    }
    SYNC_PAGE_BUF.with(|b| {
        *b.borrow_mut() = page_buf;
    });
}

/// Makes the views of slots `slots` that were left null (textures the title
/// placed itself, textures given up) in every shader-visible view heap, now
/// that shaders read them (`crate::shadow::take_reads`), and stamps what
/// they view as used (`crate::resource::evict`).
#[cfg(windows)]
pub(crate) fn make_read(slots: &[u32]) {
    if crate::heap_worker::on() {
        let frame = crate::queue::presented();
        let mut kick_heaps: Vec<usize> = Vec::new();
        let mut sync_uavs: Vec<(usize, u32)> = Vec::new();
        {
            let heaps = SHADER_HEAPS.read().unwrap_or_else(|e| e.into_inner());
            for &(_, _, at) in heaps.iter() {
                // SAFETY: registered heaps are live (removed when dropped).
                let heap = unsafe { &*(at as *const Heap) };
                let Some(host) = heap.made_host() else {
                    continue;
                };
                if heap.desc.kind != VIEWS_KIND {
                    continue;
                }
                let mut unmade = Vec::new();
                for &i in slots.iter().filter(|&&i| i < heap.desc.count) {
                    host.stamp_read(i);
                    match host.unmade_read_kind(i) {
                        host::UnmadeRead::None => {}
                        host::UnmadeRead::Background => unmade.push(i),
                        host::UnmadeRead::Sync => sync_uavs.push((at, i)),
                    }
                }
                if !unmade.is_empty() {
                    host.prioritize_texture_slots(&unmade, frame);
                    kick_heaps.push(at);
                }
            }
            for (at, i) in sync_uavs {
                // SAFETY: registered heaps are live under `SHADER_HEAPS`.
                let heap = unsafe { &*(at as *const Heap) };
                if let (Some(host), Some(handle)) = (heap.made_host(), heap.handle(i)) {
                    host.mirror(i, handle, true);
                    host.stamp_read(i);
                }
            }
            // Kick while holding `SHADER_HEAPS.read()` so each heap stays live
            // while `kick` increments its refcount (never locks `Host::submitted`).
            for at in kick_heaps {
                // SAFETY: registered heaps are live under `SHADER_HEAPS`.
                crate::heap_worker::kick(unsafe { &*(at as *const Heap) });
            }
        }
        return;
    }
    // The placed textures those slots view are made first, without the
    // heaps' lock, which every title thread writing a view waits on (making
    // a texture takes up to milliseconds; title threads waited 1-6 s a frame,
    // runs evict2, noevict1). They are never released, so they outlive it;
    // the lock is held only to look the slots up and to make the views.
    let placed: Vec<View> = {
        let heaps = SHADER_HEAPS.read().unwrap_or_else(|e| e.into_inner());
        heaps
            .iter()
            // SAFETY: registered heaps are live (removed when dropped).
            .map(|&(_, _, at)| unsafe { &*(at as *const Heap) })
            .filter(|heap| heap.desc.kind == VIEWS_KIND)
            .filter_map(|heap| heap.host.get()?.as_ref().map(|host| (heap, host)))
            .flat_map(|(heap, host)| slots.iter().filter(|&&i| i < heap.desc.count).filter_map(|&i| host.unmade_placed(i)))
            .collect()
    };
    #[cfg(test)]
    tests::while_making(slots);
    // Several at a time (each a committed texture; hundreds in a new shot,
    // 7 % of the submitting thread when made one by one, run evict5).
    crate::texture::in_parallel(placed.len(), |i| {
        let _ = host::placed_texture(&placed[i]);
    });
    let heaps = SHADER_HEAPS.read().unwrap_or_else(|e| e.into_inner());
    for &(start, _, at) in heaps.iter() {
        // SAFETY: registered heaps are live (removed when dropped).
        let heap = unsafe { &*(at as *const Heap) };
        let (Some(Some(host)), Some(step)) = (heap.host.get(), increment(heap.desc.kind)) else {
            continue;
        };
        if heap.desc.kind != VIEWS_KIND {
            continue;
        }
        for &i in slots.iter().filter(|&&i| i < heap.desc.count) {
            host.mirror(i, start + u64::from(i) * u64::from(step), true);
            host.stamp_read(i);
        }
    }
}

/// Wakes the background heap worker for any shader-visible view heap that has
/// queued texture slots or tracked bindless textures.
#[cfg(windows)]
pub(crate) fn kick_worker_heaps() {
    if !crate::heap_worker::on() {
        return;
    }
    let heaps = SHADER_HEAPS.read().unwrap_or_else(|e| e.into_inner());
    for &(_, _, at) in heaps.iter() {
        // SAFETY: registered heaps are live under `SHADER_HEAPS`.
        let heap = unsafe { &*(at as *const Heap) };
        if heap.desc.kind != VIEWS_KIND {
            continue;
        }
        let Some(host) = heap.made_host() else {
            continue;
        };
        if host.has_queued_textures() || host.viewed.has_textures() {
            crate::heap_worker::kick(heap);
        }
    }
}

/// Makes again, in every shader-visible view heap, the unordered texture
/// views left null when their textures were given up
/// (`VICTORY_EVICT_HEAP_UAV`): from now on shaders write through heaps.
#[cfg(windows)]
pub(crate) fn remake_unordered() {
    let heaps = SHADER_HEAPS.read().unwrap_or_else(|e| e.into_inner());
    let mut made = 0;
    for &(start, _, at) in heaps.iter() {
        // SAFETY: registered heaps are live (removed when dropped).
        let heap = unsafe { &*(at as *const Heap) };
        if let (Some(Some(host)), Some(step), VIEWS_KIND) = (heap.host.get(), increment(heap.desc.kind), heap.desc.kind) {
            made += host.remake_unordered(start, u64::from(step));
        }
    }
    tracing::info!(made, "shaders write through heaps: unordered texture views made again");
}

/// The host textures (raw pointers) that slots `slots` of the
/// shader-visible view heaps view.
#[cfg(windows)]
pub(crate) fn read_textures(slots: &[u32]) -> std::collections::HashSet<usize> {
    let heaps = SHADER_HEAPS.read().unwrap_or_else(|e| e.into_inner());
    let mut textures = std::collections::HashSet::new();
    for &(_, _, at) in heaps.iter() {
        // SAFETY: registered heaps are live (removed when dropped).
        let heap = unsafe { &*(at as *const Heap) };
        if let (Some(Some(host)), VIEWS_KIND) = (heap.host.get(), heap.desc.kind) {
            textures.extend(host.textures_at(slots));
        }
    }
    textures
}

/// Every shader-visible view heap's title slots viewing one of the host
/// textures `textures` (raw pointers) are left null, made again when read
/// (`crate::resource::evict` gives those textures up). Dynamic slots stay
/// made for the lists that recorded them.
#[cfg(windows)]
pub(crate) fn evict(textures: &std::collections::HashSet<usize>) {
    let heaps = SHADER_HEAPS.read().unwrap_or_else(|e| e.into_inner());
    for &(_, _, at) in heaps.iter() {
        // SAFETY: registered heaps are live (removed when dropped).
        let heap = unsafe { &*(at as *const Heap) };
        if let (Some(Some(host)), VIEWS_KIND) = (heap.host.get(), heap.desc.kind) {
            host.evict(textures);
        }
    }
}

/// Gives back dynamic slots of the host heap behind the title heap at
/// `heap`, only if that heap still exists and its identity matches `id`.
#[cfg(windows)]
pub(crate) fn release_dynamic(heap: usize, id: u64, first: u32, n: u32) {
    let heaps = SHADER_HEAPS.read().unwrap_or_else(|e| e.into_inner());
    if heaps.iter().any(|&(_, _, h)| h == heap)
        // SAFETY: registered heaps are live (removed when dropped).
        && let Some(Some(host)) = unsafe { &*(heap as *const Heap) }.host.get()
    {
        host.release(id, first, n);
    }
}

/// Runs `f` on the host heap behind the title heap at `heap` while that
/// heap cannot be released; None if it is gone or has no host heap.
#[cfg(windows)]
pub(crate) fn with_host<R>(heap: usize, f: impl FnOnce(&host::HostHeap) -> R) -> Option<R> {
    let heaps = SHADER_HEAPS.read().unwrap_or_else(|e| e.into_inner());
    if !heaps.iter().any(|&(_, _, h)| h == heap) {
        return None;
    }
    // SAFETY: registered heaps are live (removed when dropped, which waits
    // for the lock we hold).
    let host = unsafe { &*(heap as *const Heap) }.host.get()?.as_ref()?;
    Some(f(host))
}

#[cfg(windows)]
impl Drop for Heap {
    fn drop(&mut self) {
        let at = std::ptr::from_ref::<Heap>(self) as usize;
        let mut heaps = SHADER_HEAPS.write().unwrap_or_else(|e| e.into_inner());
        unregister_heap(&mut heaps, at);
        // Nothing can find the heap by now. The table's lock is taken
        // inside the heaps' lock (`memory::bind`), never the other way
        // round, so it is taken here after the heaps' lock is let go.
        drop(heaps);
        crate::memory::release(at);
    }
}

/// Bytes per descriptor, by heap type (views, samplers, render targets,
/// depth).
fn increment(kind: u32) -> Option<u32> {
    [32, 16, 128, 128].get(kind as usize).copied()
}

/// Device method.
pub(crate) extern "system" fn increment_size(_: This, kind: u32) -> u32 {
    increment(kind).unwrap_or(0)
}

/// What our render target views hold: the resource and the view
/// description, turned into a host view when a command uses them.
#[repr(C)]
#[derive(Clone, Copy)]
pub(crate) struct RtvRecord {
    tag: u32,
    described: u32,
    pub(crate) resource: This,
    desc: [u32; 6],
}
const RTV_TAG: u32 = u32::from_le_bytes(*b"vRTV");
const _: () = assert!(size_of::<RtvRecord>() <= 112);

impl RtvRecord {
    /// All the view it records is made from: records with equal keys have
    /// equal views (of the same host texture).
    #[cfg(any(windows, test))]
    pub(crate) fn key(&self) -> (usize, [u32; 7]) {
        let mut words = [self.described; 7];
        words[1..].copy_from_slice(&self.desc);
        (self.resource as usize, words)
    }

    /// The view description in the host's type (the same layout).
    #[cfg(windows)]
    pub(crate) fn desc(
        &self,
    ) -> Option<windows::Win32::Graphics::Direct3D12::D3D12_RENDER_TARGET_VIEW_DESC> {
        const TEXTURE_2D: u32 = 4;
        self.host_desc(TEXTURE_2D)
    }

    /// The depth view description in the host's type (the same layout).
    #[cfg(windows)]
    pub(crate) fn depth_desc(
        &self,
    ) -> Option<windows::Win32::Graphics::Direct3D12::D3D12_DEPTH_STENCIL_VIEW_DESC> {
        const TEXTURE_2D: u32 = 3;
        self.host_desc(TEXTURE_2D)
    }

    /// The description words as a host type, the format as the host has it.
    /// Without one, a typeless host copy gets a 2D view (`dimension`) in
    /// the title's format, which the host needs spelled out.
    #[cfg(windows)]
    fn host_desc<T>(&self, dimension: u32) -> Option<T> {
        assert!(size_of::<T>() <= size_of::<[u32; 6]>());
        let mut words = if self.described != 0 {
            self.desc
        } else {
            [
                crate::resource::view_format(self.resource)?,
                dimension,
                0,
                0,
                0,
                0,
            ]
        };
        // Render target and depth views: a packed 16-bit target may be
        // R8G8B8A8_UNORM on this GPU (`resource::target_format`).
        words[0] = crate::resource::target_format(words[0]);
        // SAFETY: both host view descriptions are 24 bytes in this layout.
        Some(unsafe { (&raw const words).cast::<T>().read_unaligned() })
    }
}

#[cfg(windows)]
const _: () = {
    use windows::Win32::Graphics::Direct3D12::{
        D3D12_DEPTH_STENCIL_VIEW_DESC, D3D12_RENDER_TARGET_VIEW_DESC,
        D3D12_SHADER_RESOURCE_VIEW_DESC, D3D12_UNORDERED_ACCESS_VIEW_DESC,
    };
    assert!(size_of::<D3D12_RENDER_TARGET_VIEW_DESC>() == 24);
    assert!(std::mem::offset_of!(D3D12_RENDER_TARGET_VIEW_DESC, Anonymous) == 8);
    assert!(size_of::<D3D12_DEPTH_STENCIL_VIEW_DESC>() == 24);
    assert!(std::mem::offset_of!(D3D12_DEPTH_STENCIL_VIEW_DESC, Anonymous) == 12);
    assert!(size_of::<D3D12_SHADER_RESOURCE_VIEW_DESC>() == 40);
    assert!(std::mem::offset_of!(D3D12_SHADER_RESOURCE_VIEW_DESC, Anonymous) == 16);
    assert!(size_of::<D3D12_UNORDERED_ACCESS_VIEW_DESC>() == 40);
    assert!(std::mem::offset_of!(D3D12_UNORDERED_ACCESS_VIEW_DESC, Anonymous) == 8);
};

/// Zeroes the union tail words an RTV or DSV dimension does not use so
/// uninitialized caller stack bytes do not cause false target-cache misses.
/// Unknown dimensions and dimension 0 keep every word unchanged.
fn sanitize_target_words(words: &mut [u32; 6], depth: bool) {
    let (start, used) = if depth {
        // D3D12_DSV_DIMENSION: 1D, 1D array, 2D, 2D array, 2D MS, 2D MS array.
        const USED: [usize; 7] = [0, 1, 3, 1, 3, 0, 2];
        (3, USED.get(words[1] as usize).copied())
    } else {
        // D3D12_RTV_DIMENSION: buffer, 1D, 1D array, 2D, 2D array, 2D MS,
        // 2D MS array, 3D.
        const USED: [usize; 9] = [0, 3, 1, 3, 2, 4, 0, 2, 3];
        (2, USED.get(words[1] as usize).copied())
    };
    if let Some(used) = used.filter(|_| words[1] != 0) {
        words[start + used..6].fill(0);
    }
}

fn write_target_record(
    tag: u32,
    depth: bool,
    resource: This,
    desc: *const [u32; 6],
    handle: u64,
) {
    if handle == 0 {
        return;
    }
    let desc_words = words(desc).map(|mut w| {
        sanitize_target_words(&mut w, depth);
        w
    });
    let record = RtvRecord {
        tag,
        described: u32::from(desc_words.is_some()),
        resource,
        desc: desc_words.unwrap_or_default(),
    };
    // SAFETY: the handle is a descriptor slot, at least 112 bytes.
    unsafe { (handle as *mut RtvRecord).write_unaligned(record) };
}

/// Device method: records the view at `handle` (descriptor memory).
pub(crate) extern "system" fn create_rtv(
    _: This,
    resource: This,
    desc: *const [u32; 6],
    handle: u64,
) {
    write_target_record(RTV_TAG, false, resource, desc, handle);
}

/// Device method: a depth view, recorded as render target views are.
pub(crate) extern "system" fn create_dsv(
    _: This,
    resource: This,
    desc: *const [u32; 6],
    handle: u64,
) {
    write_target_record(DSV_TAG, true, resource, desc, handle);
}
const DSV_TAG: u32 = u32::from_le_bytes(*b"vDSV");

/// A shader view or sampler as the title described it (descriptions in the
/// host's layout, kept as words), for the host to build its own from.
#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord)]
pub(crate) enum View {
    /// Constant buffer: GPU address and size.
    Constant(u64, u32),
    /// Shader resource: resource, description (none: the resource's own).
    Shader(usize, Option<[u32; 10]>),
    /// Unordered access: resource, counter resource, description.
    Unordered(usize, usize, Option<[u32; 10]>),
    Sampler([u32; 13]),
    /// The console's own sampler description: flags, filters per axis,
    /// addressing, LOD and anisotropy settings, border color, then
    /// console-only tuning words.
    ConsoleSampler([u32; 27]),
    /// A view of memory without a resource object: whether it is unordered
    /// access, the resource description, then the view's (the plain one
    /// followed by the memory address and console fields).
    Placed(bool, [u32; 14], [u32; 16]),
}

/// Every distinct view in use; a descriptor holds a tag, an index into it
/// and the entry's generation, which survive titles copying descriptors as
/// bytes. A view of a resource lives as long as the resource: when the
/// title releases it, its entries are freed for reuse under a new
/// generation, so descriptors (and host slots made from them) naming the
/// old one go stale instead of naming whatever is made at the same address
/// next.
/// shortcut: views without a resource object are kept for the run. None
/// can be freed by age: a descriptor naming one may lie anywhere in title
/// memory (a table written once at load, or copied as bytes), and freeing
/// the oldest left the world's tables null. Samplers are bounded by the
/// title's sampler states. Constant buffer views and views of placed
/// memory are not: a title writes thousands of constant buffer views a
/// frame, each at the address its constants have that frame, and as
/// entries they grew the table without end (about 0.28 GB a minute, found
/// in 64bitMango's fork). So those two kinds are not entries where their
/// record can name them whole ([`inline_words`]); the table keeps only the
/// shapes of placed views ([`SHAPES`]), as many as the title has kinds of
/// textures and buffers. `VICTORY_VIEW_INLINE=0` makes them entries again.
#[cfg_attr(test, derive(Default))]
struct Views {
    /// Each entry's view and generation.
    list: Vec<(View, u32)>,
    /// Live entries in [`index`](Self::index) holding constant buffer views
    /// (`[0]`) and placed memory views (`[1]`), kept O(1) as entries change.
    address_entries: [u64; 2],
    index: std::collections::BTreeMap<View, u32>,
    /// Entries naming each resource.
    by_resource: std::collections::BTreeMap<usize, Vec<u32>>,
    free: Vec<u32>,
}

/// Read by every view written and every slot a draw binds, on the
/// title's threads at once; written when a view is first seen or its
/// resource released.
static VIEWS: std::sync::RwLock<Views> = std::sync::RwLock::new(Views {
    list: Vec::new(),
    address_entries: [0, 0],
    index: std::collections::BTreeMap::new(),
    by_resource: std::collections::BTreeMap::new(),
    free: Vec::new(),
});
const VIEW_TAG: u32 = u32::from_le_bytes(*b"vVIE");

// A record is the tag and two words, `i` and `generation` (the host keeps
// them as one 64-bit key per slot). The top two bits of `generation` say
// what the other 62 bits are: 0, entry `i` of the table at that
// generation; [`CONSTANT`], a constant buffer view, whole; [`PLACED`], a
// view of placed memory: its address and the number of its shape.
const KIND_SHIFT: u32 = 30;
/// Table generations stay below the kind bits (and above 0).
const GENERATION_MASK: u32 = (1 << KIND_SHIFT) - 1;
const CONSTANT: u32 = 1;
const PLACED: u32 = 2;
/// Bits of a constant buffer view's address in its record: any address a
/// program can have. The 15 bits above them are its size.
const CONSTANT_ADDRESS_BITS: u32 = 47;
/// Bits of a placed view's address in its record: placed memory is title
/// memory, which lies below 1 TiB. The 22 bits above them number its shape.
const PLACED_ADDRESS_BITS: u32 = 40;
const SHAPE_LIMIT: usize = 1 << (62 - PLACED_ADDRESS_BITS);

/// The next generation of a table entry after `generation`: never 0 (no
/// view) and never into the kind bits.
fn next_generation(generation: u32) -> u32 {
    (generation.wrapping_add(1) & GENERATION_MASK).max(1)
}

/// A view of placed memory without its address: whether it is unordered
/// access, the resource description, and the view's words but the two that
/// hold the address (10 and 11).
type Shape = (bool, [u32; 14], [u32; 14]);

/// The shapes of placed views named whole in their records, by number, and
/// where each is. Only ever added to: as many as the title has kinds of
/// textures and buffers, however often it moves them. A lock of its own:
/// nearly every placed view has a shape seen before, which readers share.
struct Shapes {
    list: Vec<Shape>,
    index: std::collections::BTreeMap<Shape, u32>,
}

static SHAPES: std::sync::RwLock<Shapes> = std::sync::RwLock::new(Shapes {
    list: Vec::new(),
    index: std::collections::BTreeMap::new(),
});

/// The number of shapes in [`SHAPES`], for [`view_gen_current`] to know a
/// record's shape without the lock (stored after the shape is there).
static SHAPES_KNOWN: std::sync::atomic::AtomicUsize = std::sync::atomic::AtomicUsize::new(0);

/// Whether constant buffer views and views of placed memory are named whole
/// in their records ([`Views`]): unless `VICTORY_VIEW_INLINE=0`.
fn view_inline() -> bool {
    #[cfg(test)]
    if tests::ENTRIES_ONLY.with(|only| only.get()) {
        return false;
    }
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| std::env::var_os("VICTORY_VIEW_INLINE").is_none_or(|v| v != "0"))
}

/// The two record words of the 62 bits `payload` of kind `kind`.
fn kind_words(kind: u32, payload: u64) -> [u32; 2] {
    [payload as u32, kind << KIND_SHIFT | (payload >> 32) as u32]
}

/// The 62 bits two record words carry beside their kind.
fn payload(i: u32, generation: u32) -> u64 {
    u64::from(generation & GENERATION_MASK) << 32 | u64::from(i)
}

/// The record words that name the constant buffer view of `size` bytes at
/// `address` whole; `None` if they cannot: an address past 47 bits, or a
/// size of 16 KiB or more that is no multiple of 256 or reaches 4 MiB (the
/// size has 15 bits: bytes below 16 KiB, else units of 256 bytes). Such a
/// view is a table entry.
fn constant_words(address: u64, size: u32) -> Option<[u32; 2]> {
    if address >> CONSTANT_ADDRESS_BITS != 0 {
        return None;
    }
    let code = if size < 1 << 14 {
        u64::from(size)
    } else if size.is_multiple_of(256) && size >> 8 < 1 << 14 {
        1 << 14 | u64::from(size >> 8)
    } else {
        return None;
    };
    Some(kind_words(CONSTANT, code << CONSTANT_ADDRESS_BITS | address))
}

/// The constant buffer view [`constant_words`] named.
fn constant_of(i: u32, generation: u32) -> View {
    let payload = payload(i, generation);
    let code = (payload >> CONSTANT_ADDRESS_BITS) as u32;
    let size = if code >> 14 == 0 { code } else { (code & 0x3fff) << 8 };
    View::Constant(payload & ((1 << CONSTANT_ADDRESS_BITS) - 1), size)
}

/// A placed view's shape and its address.
fn shape_of(unordered: bool, resource: &[u32; 14], desc: &[u32; 16]) -> (Shape, u64) {
    let mut rest = [0u32; 14];
    rest[..10].copy_from_slice(&desc[..10]);
    rest[10..].copy_from_slice(&desc[12..]);
    ((unordered, *resource, rest), u64::from(desc[10]) | u64::from(desc[11]) << 32)
}

/// The number of `shape`, a new one if it is not there yet; `None` when the
/// shapes are full (the view is then a table entry).
fn shape_number(shape: Shape) -> Option<u32> {
    if let Some(&at) = SHAPES.read().unwrap_or_else(|e| e.into_inner()).index.get(&shape) {
        return Some(at);
    }
    let mut shapes = SHAPES.write().unwrap_or_else(|e| e.into_inner());
    // Looked up again: another thread may have entered it meanwhile.
    if let Some(&at) = shapes.index.get(&shape) {
        return Some(at);
    }
    if shapes.list.len() >= SHAPE_LIMIT {
        return None;
    }
    let at = shapes.list.len() as u32;
    shapes.list.push(shape);
    shapes.index.insert(shape, at);
    SHAPES_KNOWN.store(shapes.list.len(), std::sync::atomic::Ordering::Release);
    Some(at)
}

/// The view of placed memory named by the record words, if its shape is known.
fn placed_of(i: u32, generation: u32) -> Option<View> {
    let payload = payload(i, generation);
    let address = payload & ((1 << PLACED_ADDRESS_BITS) - 1);
    let shapes = SHAPES.read().unwrap_or_else(|e| e.into_inner());
    let &(unordered, resource, rest) = shapes.list.get((payload >> PLACED_ADDRESS_BITS) as usize)?;
    let mut desc = [0u32; 16];
    desc[..10].copy_from_slice(&rest[..10]);
    (desc[10], desc[11]) = (address as u32, (address >> 32) as u32);
    desc[12..].copy_from_slice(&rest[10..]);
    Some(View::Placed(unordered, resource, desc))
}

/// The two words that name `view` whole in a record, where its kind allows
/// and `VICTORY_VIEW_INLINE` is not `0` ([`Views`]); else `None`, and the
/// view is a table entry.
fn inline_words(view: &View) -> Option<[u32; 2]> {
    if !view_inline() {
        return None;
    }
    match view {
        View::Constant(address, size) => constant_words(*address, *size),
        View::Placed(unordered, resource, desc) => {
            let (shape, address) = shape_of(*unordered, resource, desc);
            if address >> PLACED_ADDRESS_BITS != 0 {
                return None;
            }
            let payload = u64::from(shape_number(shape)?) << PLACED_ADDRESS_BITS | address;
            Some(kind_words(PLACED, payload))
        }
        _ => None,
    }
}

/// The view record words of a kind other than a table entry name whole;
/// `None` for a table entry's words or a placed view of an unknown shape.
fn inline_view(i: u32, generation: u32) -> Option<View> {
    match generation >> KIND_SHIFT {
        CONSTANT => Some(constant_of(i, generation)),
        PLACED => placed_of(i, generation),
        _ => None,
    }
}

/// Whether record words of a kind other than a table entry name a view: a
/// constant buffer view always does, a placed view if its shape is known.
fn inline_current(i: u32, generation: u32) -> bool {
    match generation >> KIND_SHIFT {
        CONSTANT => true,
        PLACED => {
            let shapes = SHAPES_KNOWN.load(std::sync::atomic::Ordering::Acquire);
            ((payload(i, generation) >> PLACED_ADDRESS_BITS) as usize) < shapes
        }
        _ => false,
    }
}

impl View {
    /// The resource objects the view names.
    fn resources(&self) -> [usize; 2] {
        match *self {
            View::Shader(r, _) => [r, 0],
            View::Unordered(r, counter, _) => [r, counter],
            _ => [0, 0],
        }
    }
}

impl Views {
    /// Counts a new entry of a view without a resource object, which stays.
    fn count_address(&mut self, view: &View) {
        match view {
            View::Constant(..) => self.address_entries[0] += 1,
            View::Placed(..) => self.address_entries[1] += 1,
            _ => {}
        }
    }

    /// Looks up `view`, or inserts it: its `(index, generation)`. The
    /// lock-free mirror of generations ([`set_view_gen`]) is the caller's to
    /// set: it mirrors [`VIEWS`] only, and a test's own table set there
    /// named [`VIEWS`]'s low entries stale while another test synced them.
    fn entry_kept(&mut self, view: View) -> (u32, u32) {
        if let Some(&i) = self.index.get(&view) {
            return (i, self.list[i as usize].1);
        }
        let i = match self.free.pop() {
            Some(i) => {
                self.list[i as usize].0 = view.clone();
                i
            }
            None => {
                self.list.push((view.clone(), 1));
                self.list.len() as u32 - 1
            }
        };
        let generation = self.list[i as usize].1;
        self.count_address(&view);
        for r in view.resources().into_iter().filter(|&r| r != 0) {
            self.by_resource.entry(r).or_default().push(i);
        }
        self.index.insert(view, i);
        (i, generation)
    }
}

/// O(1) view-table counts for diagnostics: `[slots_total, live, cbv_live, placed_live]`.
#[cfg(any(all(windows, feature = "diagnostics"), test))]
pub(crate) fn views_kept() -> [u64; 4] {
    let views = VIEWS.read().unwrap_or_else(|e| e.into_inner());
    [
        views.list.len() as u64,
        views.index.len() as u64,
        views.address_entries[0],
        views.address_entries[1],
    ]
}

fn record(handle: u64, view: View) {
    if handle == 0 {
        return;
    }
    // A view its record names whole needs neither the table nor its lock.
    if let Some([i, generation]) = inline_words(&view) {
        // SAFETY: the handle is a descriptor slot, at least 16 bytes.
        unsafe { (handle as *mut [u32; 3]).write_unaligned([VIEW_TAG, i, generation]) };
        #[cfg(windows)]
        mirror(handle);
        return;
    }
    // A view seen before, as most are: its entry is found under the shared
    // lock, which keeps `forget` from making it stale while it is written.
    let known = {
        let views = VIEWS.read().unwrap_or_else(|e| e.into_inner());
        let found = views.index.get(&view).map(|&i| (i, views.list[i as usize].1));
        if let Some((i, generation)) = found {
            // SAFETY: the handle is a descriptor slot, at least 16 bytes.
            unsafe { (handle as *mut [u32; 3]).write_unaligned([VIEW_TAG, i, generation]) };
        }
        found.is_some()
    };
    if known {
        #[cfg(windows)]
        mirror(handle);
        return;
    }
    let mut views = VIEWS.write().unwrap_or_else(|e| e.into_inner());
    let (i, generation) = views.entry_kept(view);
    // Before the record names it, as the heap syncs check it without the lock.
    set_view_gen(i, generation);
    // SAFETY: the handle is a descriptor slot, at least 16 bytes.
    unsafe { (handle as *mut [u32; 3]).write_unaligned([VIEW_TAG, i, generation]) };
    drop(views);
    #[cfg(windows)]
    mirror(handle);
}

/// Lock-free mirror of each recorded view entry's active generation (0 when
/// free), chunked in 65,536-entry blocks up to 16M entries. Updated under
/// `VIEWS.write()` before publishing records or bumping `VIEWS_FORGOT`, so
/// heap syncs can check record liveness without locking `VIEWS`.
const VIEW_GEN_CHUNK_SHIFT: u32 = 16;
const VIEW_GEN_CHUNK_LEN: usize = 1 << VIEW_GEN_CHUNK_SHIFT;
const VIEW_GEN_CHUNKS: usize = 256;
static VIEW_GENS: [std::sync::OnceLock<Box<[std::sync::atomic::AtomicU32]>>; VIEW_GEN_CHUNKS] =
    [const { std::sync::OnceLock::new() }; VIEW_GEN_CHUNKS];

fn set_view_gen(i: u32, generation: u32) {
    let chunk = (i >> VIEW_GEN_CHUNK_SHIFT) as usize;
    let offset = (i as usize) & (VIEW_GEN_CHUNK_LEN - 1);
    if let Some(slot) = VIEW_GENS.get(chunk) {
        let table = slot.get_or_init(|| {
            (0..VIEW_GEN_CHUNK_LEN)
                .map(|_| std::sync::atomic::AtomicU32::new(0))
                .collect()
        });
        table[offset].store(generation, std::sync::atomic::Ordering::Release);
    }
}

/// Lock-free check of whether view entry `i` is still live at `generation`.
#[cfg(any(windows, test))]
pub(crate) fn view_gen_current(i: u32, generation: u32) -> bool {
    if generation == 0 {
        return false;
    }
    if generation >> KIND_SHIFT != 0 {
        return inline_current(i, generation);
    }
    let chunk = (i >> VIEW_GEN_CHUNK_SHIFT) as usize;
    let offset = (i as usize) & (VIEW_GEN_CHUNK_LEN - 1);
    if let Some(slot) = VIEW_GENS.get(chunk) {
        slot.get().is_some_and(|table| {
            table[offset].load(std::sync::atomic::Ordering::Acquire) == generation
        })
    } else {
        let views = VIEWS.read().unwrap_or_else(|e| e.into_inner());
        views
            .list
            .get(i as usize)
            .is_some_and(|(_, current)| *current == generation)
    }
}

/// Bumped whenever [`forget`] invalidates a recorded view while holding `VIEWS`.
static VIEWS_FORGOT: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(1);

#[cfg(any(windows, test))]
pub(crate) fn views_forgot() -> u64 {
    VIEWS_FORGOT.load(std::sync::atomic::Ordering::Acquire)
}

/// The title released `resource`: the views naming it go stale.
pub(crate) fn forget(resource: usize) {
    let mut views = VIEWS.write().unwrap_or_else(|e| e.into_inner());
    let Some(entries) = views.by_resource.remove(&resource) else {
        return;
    };
    let mut removed = false;
    for i in entries {
        let (view, generation) = views.list[i as usize].clone();
        // Entries freed through a view's other resource may hold another
        // view by now.
        if views.index.get(&view) != Some(&i) || !view.resources().contains(&resource) {
            continue;
        }
        views.index.remove(&view);
        views.list[i as usize].1 = next_generation(generation);
        set_view_gen(i, 0);
        // The view's other resource (a counter) keeps a stale entry, which
        // `forget` of that resource skips.
        views.free.push(i);
        removed = true;
    }
    if removed {
        VIEWS_FORGOT.fetch_add(1, std::sync::atomic::Ordering::Release);
    }
}

/// A current view whose resources cannot be freed while it is used: it holds
/// a reference on each resource object the view names (and none is made for
/// a resource the title's last release had already reached). The title's
/// release never waits for it. A release that would have been the last runs
/// at the pin's drop instead, on whichever thread drops it, so a pin is
/// dropped outside the locks the destructor takes (`VIEWS`, `bindless`,
/// `shadow`).
#[cfg(any(windows, test))]
pub(crate) struct ViewPin {
    view: View,
    held: [usize; 2],
}

#[cfg(any(windows, test))]
impl std::ops::Deref for ViewPin {
    type Target = View;

    fn deref(&self) -> &Self::Target {
        &self.view
    }
}

#[cfg(any(windows, test))]
impl std::fmt::Debug for ViewPin {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        self.view.fmt(f)
    }
}

#[cfg(any(windows, test))]
impl Drop for ViewPin {
    fn drop(&mut self) {
        for r in self.held.into_iter().filter(|&r| r != 0) {
            crate::object::release_ref(r as This);
        }
    }
}

/// The view of entry `i` if `generation` is still its own, pinned until the
/// returned value is dropped.
///
/// The references are taken under the shared lock. `Resource::drop` takes
/// the exclusive one (`forget`) before its memory goes, so the address of a
/// current entry's resource is still there to count on. A resource whose
/// count already reached zero is on its way out, and the view is not current
/// for it.
#[cfg(any(windows, test))]
fn view_at(i: u32, generation: u32) -> Option<ViewPin> {
    if generation >> KIND_SHIFT != 0 {
        // Named whole: no resource object to hold.
        return inline_view(i, generation).map(|view| ViewPin { view, held: [0, 0] });
    }
    let views = VIEWS.read().unwrap_or_else(|e| e.into_inner());
    let (view, current) = views.list.get(i as usize)?;
    if *current != generation {
        return None;
    }
    let view = view.clone();
    let mut held = [0usize; 2];
    let mut all = true;
    for (slot, r) in view.resources().into_iter().enumerate() {
        if r == 0 {
            continue;
        }
        if crate::object::try_add_ref(r as This) {
            held[slot] = r;
        } else {
            all = false;
        }
    }
    drop(views);
    // Dropped here, with the lock released, if a resource was already gone:
    // the references taken on the others may be their last.
    let pin = ViewPin { view, held };
    all.then_some(pin)
}

/// Batched [`view_at`] for a slice of packed `(generation << 32 | index)` keys
/// under one shared hold of `VIEWS`. Any partial pin whose other resource is
/// already dying is dropped after releasing `VIEWS`.
#[cfg(any(windows, test))]
pub(crate) fn views_at_batch(keys: &[u64]) -> Vec<Option<ViewPin>> {
    let mut results = Vec::with_capacity(keys.len());
    let mut failed = Vec::new();
    {
        let views = VIEWS.read().unwrap_or_else(|e| e.into_inner());
        for &key in keys {
            let i = key as u32;
            let generation = (key >> 32) as u32;
            if generation >> KIND_SHIFT != 0 {
                results.push(inline_view(i, generation).map(|view| ViewPin { view, held: [0, 0] }));
                continue;
            }
            let Some((view, current)) = views.list.get(i as usize) else {
                results.push(None);
                continue;
            };
            if *current != generation {
                results.push(None);
                continue;
            }
            let view = view.clone();
            let mut held = [0usize; 2];
            let mut all = true;
            for (slot, r) in view.resources().into_iter().enumerate() {
                if r == 0 {
                    continue;
                }
                if crate::object::try_add_ref(r as This) {
                    held[slot] = r;
                } else {
                    all = false;
                }
            }
            let pin = ViewPin { view, held };
            if all {
                results.push(Some(pin));
            } else {
                failed.push(pin);
                results.push(None);
            }
        }
    }
    drop(failed);
    results
}

/// For each record (tag, index, generation): whether it names a view that
/// is current, as [`view_at`] would find it (one lock for them all).
#[cfg(windows)]
fn views_current_into(records: &[[u32; 3]], out: &mut Vec<bool>) {
    out.clear();
    let views = VIEWS.read().unwrap_or_else(|e| e.into_inner());
    out.extend(records.iter().map(|&[tag, i, generation]| {
        tag == VIEW_TAG
            && if generation >> KIND_SHIFT != 0 {
                inline_current(i, generation)
            } else {
                views.list.get(i as usize).is_some_and(|(_, current)| *current == generation)
            }
    }));
}

/// The view recorded at `handle`, if one is (and its resource is alive),
/// pinned until the returned value is dropped.
#[cfg(any(windows, test))]
pub(crate) fn view(handle: u64) -> Option<ViewPin> {
    if handle == 0 {
        return None;
    }
    // SAFETY: the handle is a descriptor slot the title passes back.
    let [tag, i, generation] = unsafe { (handle as *const [u32; 3]).read_unaligned() };
    (tag == VIEW_TAG).then(|| view_at(i, generation)).flatten()
}

/// Like [`view`], but reads the title descriptor slot through
/// [`crate::host::read_into`] so unmapped heap pages never fault.
#[cfg(windows)]
pub(crate) fn view_safe(handle: u64) -> Option<ViewPin> {
    if handle == 0 {
        return None;
    }
    let mut raw = [0u8; 12];
    if !crate::host::read_into(handle as *const u8, &mut raw) {
        return None;
    }
    let tag = u32::from_le_bytes([raw[0], raw[1], raw[2], raw[3]]);
    let i = u32::from_le_bytes([raw[4], raw[5], raw[6], raw[7]]);
    let generation = u32::from_le_bytes([raw[8], raw[9], raw[10], raw[11]]);
    (tag == VIEW_TAG).then(|| view_at(i, generation)).flatten()
}

/// The view recorded at `handle` and the `(generation << 32) | index` key that
/// identifies that specific record (`host::HostHeap::publish`).
#[cfg(any(windows, test))]
pub(crate) fn view_and_key(handle: u64) -> Option<(ViewPin, u64)> {
    if handle == 0 {
        return None;
    }
    // SAFETY: as in `view`.
    let [tag, i, generation] = unsafe { (handle as *const [u32; 3]).read_unaligned() };
    if tag != VIEW_TAG {
        return None;
    }
    Some((
        view_at(i, generation)?,
        u64::from(generation) << 32 | u64::from(i),
    ))
}

/// A description of `N` words, if the title passes one.
fn words<const N: usize>(desc: *const [u32; N]) -> Option<[u32; N]> {
    // SAFETY: the title passes a description or null.
    (!desc.is_null()).then(|| unsafe { desc.read_unaligned() })
}

/// Device method.
///
/// A null `desc` writes a null CBV (`Constant(0, 0)`), matching D3D12's
/// `CreateConstantBufferView` with a null description. Leaving the slot
/// untouched would keep a stale view active.
pub(crate) extern "system" fn create_cbv(_: This, desc: *const [u32; 4], handle: u64) {
    let view = match words(desc) {
        Some([lo, hi, size, _]) => {
            let address = u64::from(lo) | u64::from(hi) << 32;
            View::Constant(address, size)
        }
        None => View::Constant(0, 0),
    };
    record(handle, view);
}

/// Copies only the defined fields of a 56-byte `ResourceDesc`, leaving struct
/// padding words 1 (after `Dimension`) and 13 (after `Flags`) zeroed so
/// uninitialized caller stack bytes do not mint duplicate `VIEWS` entries.
fn placed_resource_words(desc: *const [u32; 14]) -> Option<[u32; 14]> {
    if desc.is_null() {
        return None;
    }
    let src = desc.cast::<u32>();
    let mut fields = [0u32; 14];
    // SAFETY: the caller supplies a 56-byte `ResourceDesc`, possibly unaligned.
    // `Dimension` occupies word 0; `Alignment` through `Flags` occupy words 2..13.
    unsafe {
        fields[0] = src.read_unaligned();
        fields[2..13].copy_from_slice(&src.add(2).cast::<[u32; 11]>().read_unaligned());
    }
    Some(fields)
}

/// Zeroes SRV alignment padding (word 3) and the unused tail of the view
/// union for known non-zero SRV/UAV dimensions so uninitialized caller bytes
/// do not defeat `VIEWS` deduplication. Unknown dimensions and dimension 0
/// keep every word unchanged.
fn sanitize_view_words(words: &mut [u32; 10], unordered: bool) {
    let (start, used) = if unordered {
        // D3D12_UAV_DIMENSION: buffer, 1D, 1D array, 2D, 2D array, 2D MS,
        // 2D MS array, 3D.
        const USED: [usize; 9] = [0, 7, 1, 3, 2, 4, 0, 2, 3];
        (2, USED.get(words[1] as usize).copied())
    } else {
        // D3D12_SRV_DIMENSION: buffer, 1D, 1D array, 2D, 2D array, 2D MS,
        // 2D MS array, 3D, cube, cube array, acceleration structure.
        const USED: [usize; 12] = [0, 5, 3, 5, 4, 6, 0, 2, 3, 3, 5, 2];
        (4, USED.get(words[1] as usize).copied())
    };
    if let Some(used) = used.filter(|_| words[1] != 0) {
        if !unordered {
            words[3] = 0;
        }
        words[start + used..10].fill(0);
    }
}

fn view_words(desc: *const [u32; 10], unordered: bool) -> Option<[u32; 10]> {
    let mut d = words(desc)?;
    sanitize_view_words(&mut d, unordered);
    Some(d)
}

fn placed_view_words(desc: *const [u32; 16], unordered: bool) -> Option<[u32; 16]> {
    let mut d = words(desc)?;
    let head: &mut [u32; 10] = (&mut d[..10]).try_into().expect("10-word prefix");
    sanitize_view_words(head, unordered);
    Some(d)
}

/// Device method.
pub(crate) extern "system" fn create_srv(
    _: This,
    resource: This,
    desc: *const [u32; 10],
    handle: u64,
) {
    record(handle, View::Shader(resource as usize, view_words(desc, false)));
}

/// Device method.
pub(crate) extern "system" fn create_uav(
    _: This,
    resource: This,
    counter: This,
    desc: *const [u32; 10],
    handle: u64,
) {
    let view = View::Unordered(resource as usize, counter as usize, view_words(desc, true));
    record(handle, view);
}

/// Device methods 0x230 and 0x238: shader resource and unordered access
/// views of memory the title placed itself, `(resource desc, view desc,
/// handle)`.
pub(crate) extern "system" fn create_placed_srv(
    _: This,
    resource: *const [u32; 14],
    desc: *const [u32; 16],
    handle: u64,
) {
    if let (Some(resource), Some(desc)) =
        (placed_resource_words(resource), placed_view_words(desc, false))
    {
        record(handle, View::Placed(false, resource, desc));
    }
}

pub(crate) extern "system" fn create_placed_uav(
    _: This,
    resource: *const [u32; 14],
    desc: *const [u32; 16],
    handle: u64,
) {
    if let (Some(resource), Some(desc)) =
        (placed_resource_words(resource), placed_view_words(desc, true))
    {
        record(handle, View::Placed(true, resource, desc));
    }
}

/// Device method.
pub(crate) extern "system" fn create_sampler(_: This, desc: *const [u32; 13], handle: u64) {
    if let Some(desc) = words(desc) {
        record(handle, View::Sampler(desc));
    }
}

/// Device method 0x1e8.
pub(crate) extern "system" fn create_sampler_x(_: This, desc: *const [u32; 27], handle: u64) {
    if let Some(desc) = words(desc) {
        record(handle, View::ConsoleSampler(desc));
    }
}

/// Device method: `count` descriptors of `kind` from `src` to `dst`.
pub(crate) extern "system" fn copy_simple(_: This, count: u32, dst: u64, src: u64, kind: u32) {
    let Some(step) = increment(kind).map(u64::from) else {
        return;
    };
    let len = u64::from(count) * step;
    if dst != 0 && src != 0 {
        // SAFETY: both ranges are descriptor slots the title owns.
        unsafe { std::ptr::copy(src as *const u8, dst as *mut u8, len as usize) };
        #[cfg(windows)]
        for i in 0..u64::from(count) {
            mirror(dst + i * step);
        }
    }
}

/// Device method: ranges of descriptors (start and count each) gathered
/// from the sources and scattered to the destinations, in order.
#[allow(clippy::too_many_arguments)] // the method's own arguments
pub(crate) extern "system" fn copy(
    _: This,
    dst_ranges: u32,
    dst_starts: *const u64,
    dst_sizes: *const u32,
    src_ranges: u32,
    src_starts: *const u64,
    src_sizes: *const u32,
    kind: u32,
) {
    let Some(step) = increment(kind) else {
        return;
    };
    // Each range as (start, count); a null size array means one descriptor
    // per range.
    let ranges = |n: u32, starts: *const u64, sizes: *const u32| -> Vec<(u64, u32)> {
        (0..n as usize)
            .map(|i| {
                // SAFETY: the title passes `n` starts and sizes (or null sizes).
                unsafe {
                    (
                        *starts.add(i),
                        if sizes.is_null() { 1 } else { *sizes.add(i) },
                    )
                }
            })
            .collect()
    };
    if dst_starts.is_null() || src_starts.is_null() {
        return;
    }
    let src = ranges(src_ranges, src_starts, src_sizes);
    let mut dst = ranges(dst_ranges, dst_starts, dst_sizes).into_iter();
    let (mut to, mut room) = dst.next().unwrap_or((0, 0));
    for (mut from, mut left) in src {
        while left > 0 {
            while room == 0 {
                let Some(next) = dst.next() else { return };
                (to, room) = next;
            }
            let n = left.min(room);
            copy_simple(std::ptr::null(), n, to, from, kind);
            let bytes = u64::from(n) * u64::from(step);
            (to, from, room, left) = (to + bytes, from + bytes, room - n, left - n);
        }
    }
}

/// The view recorded at `handle`, if one is.
pub(crate) fn rtv_record(handle: u64) -> Option<RtvRecord> {
    tagged_record(handle, RTV_TAG)
}

/// The depth view recorded at `handle`, if one is.
pub(crate) fn dsv_record(handle: u64) -> Option<RtvRecord> {
    tagged_record(handle, DSV_TAG)
}

fn tagged_record(handle: u64, tag: u32) -> Option<RtvRecord> {
    if handle == 0 {
        return None;
    }
    // SAFETY: the handle is a descriptor slot the title passes back.
    let record = unsafe { (handle as *const RtvRecord).read_unaligned() };
    (record.tag == tag).then_some(record)
}

/// Device method.
pub(crate) extern "system" fn create_heap(
    _: This,
    desc_ptr: *const HeapDesc,
    iid: *const Iid,
    out: *mut This,
) -> HRESULT {
    // SAFETY: the title passes a description (null is tolerated).
    let Some(&desc) = (unsafe { desc_ptr.as_ref() }) else {
        return E_INVALIDARG;
    };
    let Some(size) = increment(desc.kind).map(|i| u64::from(i) * u64::from(desc.count.max(1)))
    else {
        return E_INVALIDARG;
    };
    // SAFETY: read only as a hint: a title that passes the plain
    // description has its own stack or data after it.
    let placement = unsafe { desc_ptr.add(1).cast::<Placement>().read_unaligned() };
    // shortcut: a placement is recognised by its size matching the count,
    // since the plain and extended descriptions share the slot and the
    // interface version; find the flag that tells them apart if a title
    // trips over this.
    let placed = placement.address != 0 && placement.size == size;
    tracing::debug!(?desc, placed, "descriptor heap");
    let (start, memory) = if placed {
        if placement.address.checked_add(size).is_none() {
            return E_INVALIDARG;
        }
        (placement.address, None)
    } else {
        match Memory::zeroed(size, 64) {
            Ok(memory) => (memory.address(), Some(memory)),
            Err(hr) => return hr,
        }
    };
    let visible = desc.flags & SHADER_VISIBLE != 0 && matches!(desc.kind, VIEWS_KIND | SAMPLERS);
    let heap = Box::new(Heap {
        object: Object::new(&FUNCTIONS),
        desc,
        start,
        memory,
        #[cfg(windows)]
        host: std::sync::OnceLock::new(),
        #[cfg(windows)]
        pending: Default::default(),
        #[cfg(windows)]
        written: (placed && visible)
            .then(|| crate::watch::track(start as usize, size as usize, true))
            .flatten(),
    });
    #[cfg(windows)]
    let at = std::ptr::from_ref::<Heap>(&heap) as usize;
    #[cfg(not(windows))]
    let _ = visible;
    let result = object::hand_out(DESCRIPTOR_HEAP, heap, iid, &HEAP_IIDS, out);
    #[cfg(windows)]
    if visible && result == crate::object::S_OK {
        let mut heaps = SHADER_HEAPS.write().unwrap_or_else(|e| e.into_inner());
        register_heap(&mut heaps, (start, start + size), at);
    }
    result
}

extern "system" fn query(this: This, iid: *const Iid, out: *mut This) -> HRESULT {
    let found = object::is(iid, &HEAP_IIDS).then_some(this);
    object::answer(DESCRIPTOR_HEAP, iid, found, out)
}
extern "system" fn release(this: This) -> u32 {
    object::release_boxed::<Heap>(this)
}
extern "system" fn desc(this: This, out: *mut HeapDesc) {
    if !out.is_null() {
        // SAFETY: checked non-null; the title passes room for a description.
        unsafe { out.write(heap(this).desc) };
    }
}
/// The same address for the CPU and the GPU start.
extern "system" fn start(this: This) -> u64 {
    heap(this).start
}

/// IUnknown 3, object 4, device child 1, heap 3.
static FUNCTIONS: Table<24> = {
    let mut f = object::d3d12_object(object::missing_table::<DESCRIPTOR_HEAP, 24>());
    f[0] = query as *const ();
    f[2] = release as *const ();
    f[7] = object::parent_device as *const (); // 0x38
    f[8] = desc as *const (); // 0x40
    f[9] = start as *const (); // 0x48 CPU
    f[10] = start as *const (); // 0x50 GPU
    Table(f)
};

#[cfg(test)]
pub(crate) mod tests {
    /// Work to run while `make_read` makes the textures of a read slot (the
    /// slot), outside the heaps' lock.
    pub(crate) static WHILE_MAKING: std::sync::Mutex<Option<(u32, Work)>> = std::sync::Mutex::new(None);

    /// The work [`WHILE_MAKING`] runs.
    pub(crate) type Work = Box<dyn FnOnce() + Send>;

    pub(crate) fn while_making(slots: &[u32]) {
        let mut hook = WHILE_MAKING.lock().unwrap_or_else(|e| e.into_inner());
        if hook.as_ref().is_some_and(|(slot, _)| slots.contains(slot))
            && let Some((_, work)) = hook.take()
        {
            drop(hook);
            work();
        }
    }

    use super::*;
    use crate::object::{S_OK, slot};

    /// What newer titles pass; tests use it throughout, so that reading the
    /// placement stays inside the value.
    #[repr(C)]
    struct Extended(HeapDesc, Placement);

    fn create(desc: HeapDesc, placement: Placement, out: &mut This) -> HRESULT {
        let extended = Extended(desc, placement);
        create_heap(std::ptr::null(), &extended.0, &HEAP_IIDS[0], out)
    }

    /// A released view heap takes its title memory page table with it, also
    /// when no heap is made at its address afterwards (else the table, its
    /// 16.8 MiB mapping and the copies it names live on).
    #[cfg(windows)]
    #[test]
    fn a_released_view_heap_drops_its_page_table() {
        let Some(host) = crate::host::host() else {
            return; // no host GPU here
        };
        let desc = HeapDesc {
            kind: VIEWS_KIND,
            count: 1000,
            flags: SHADER_VISIBLE,
            nodes: 0,
        };
        let none = Placement {
            address: 0,
            size: 0,
        };
        let mut h = std::ptr::null();
        assert_eq!(create(desc, none, &mut h), S_OK);
        let key = h as usize;
        assert!(heap(h).host().is_some(), "a host heap");
        let list = host.new_list().unwrap();
        assert!(crate::memory::bind(&list, key).is_some(), "a page table");
        assert!(crate::memory::has_table(key));
        assert_eq!(slot::<extern "system" fn(This) -> u32>(h, 0x10)(h), 0);
        assert!(!crate::memory::has_table(key), "released with its heap");
    }

    #[test]
    fn registering_a_heap_retires_overlapping_ranges_but_keeps_adjacent_ones() {
        let mut heaps = vec![(10, 20, 1), (20, 30, 2), (30, 40, 3), (40, 50, 4)];
        register_heap(&mut heaps, (15, 35), 5);
        assert_eq!(heaps, [(40, 50, 4), (15, 35, 5)]);
        register_heap(&mut heaps, (35, 40), 6);
        assert_eq!(heaps, [(40, 50, 4), (15, 35, 5), (35, 40, 6)]);
    }

    #[test]
    fn lookup_is_half_open_and_refuses_ambiguous_ranges() {
        let heaps = [(10, 20, 1), (20, 30, 2)];
        assert_eq!(heap_at(&heaps, 10), Some(1));
        assert_eq!(heap_at(&heaps, 19), Some(1));
        assert_eq!(heap_at(&heaps, 20), Some(2));
        assert_eq!(heap_at(&heaps, 30), None);
        assert_eq!(heap_at(&[(10, 30, 1), (20, 40, 2)], 25), None);
    }

    #[test]
    fn unregistering_a_retired_heap_keeps_its_replacement() {
        let mut heaps = vec![(10, 20, 1)];
        register_heap(&mut heaps, (10, 20), 2);
        unregister_heap(&mut heaps, 1);
        assert_eq!(heaps, [(10, 20, 2)]);
    }

    #[cfg(windows)]
    struct HeapOwner(This);

    #[cfg(windows)]
    impl Drop for HeapOwner {
        fn drop(&mut self) {
            if !self.0.is_null() {
                slot::<extern "system" fn(This) -> u32>(self.0, 0x10)(self.0);
            }
        }
    }

    #[cfg(windows)]
    fn owner(desc: HeapDesc, placement: Placement) -> HeapOwner {
        let mut this = std::ptr::null();
        assert_eq!(create(desc, placement, &mut this), S_OK);
        HeapOwner(this)
    }

    #[test]
    fn heaps_hold_count_descriptors_from_their_start() {
        let null = std::ptr::null();
        let desc = HeapDesc {
            kind: 0,
            count: 1000,
            flags: 1,
            nodes: 0,
        };
        let none = Placement {
            address: 0,
            size: 0,
        };
        let mut h = null;
        assert_eq!(create(desc, none, &mut h), S_OK);
        let cpu = slot::<extern "system" fn(This) -> u64>(h, 0x48)(h);
        assert_eq!(cpu % 64, 0);
        assert_eq!(slot::<extern "system" fn(This) -> u64>(h, 0x50)(h), cpu);
        assert_eq!(increment_size(null, 0), 32);
        // SAFETY: the heap holds 1000 descriptors of 32 bytes.
        unsafe { (cpu as *mut u8).add(999 * 32 + 31).write(1) };
        assert_eq!(increment_size(null, 9), 0);
        let bad = HeapDesc { kind: 9, ..desc };
        assert_eq!(create(bad, none, &mut h), E_INVALIDARG);
    }

    #[test]
    fn views_survive_copies() {
        let null = std::ptr::null();
        let mut heap = vec![0u64; 4 * 8]; // four 32-byte slots
        let at = |i: usize| heap.as_ptr() as u64 + 32 * i as u64;
        let (s0, s1, s2, s3) = (at(0), at(1), at(2), at(3));
        create_cbv(null, &[0x1000, 1, 256, 0], s0);
        assert_eq!(view(s0).as_deref(), Some(&View::Constant(0x1_0000_1000, 256)));
        create_srv(null, null, std::ptr::null(), s1);
        assert_eq!(view(s1).as_deref(), Some(&View::Shader(0, None)));
        // The same view twice is one entry.
        create_srv(null, null, std::ptr::null(), s2);
        // SAFETY: both slots are in `heap`.
        assert_eq!(unsafe { *(s1 as *const [u32; 3]) }, unsafe {
            *(s2 as *const [u32; 3])
        });
        // Two ranges of one into one range of two.
        let (dst, dst_n, src, src_n) = ([s2], [2u32], [s0, s1], [1u32, 1]);
        copy(
            null,
            1,
            dst.as_ptr(),
            dst_n.as_ptr(),
            2,
            src.as_ptr(),
            src_n.as_ptr(),
            0,
        );
        assert_eq!(view(s2).as_deref(), view(s0).as_deref());
        assert_eq!(view(s3).as_deref(), view(s1).as_deref());
        assert!(view(at(0) + 8).is_none());
        heap[0] = 0;
    }

    /// An object of ours that frees itself at its last release, as a
    /// resource does: its views go stale first (`forget`), and `FREED` notes
    /// its address.
    static FREED: std::sync::Mutex<Vec<usize>> = std::sync::Mutex::new(Vec::new());

    fn freed(address: usize) -> bool {
        FREED.lock().unwrap().contains(&address)
    }

    extern "system" fn release_resource_like(this: This) -> u32 {
        if crate::object::refs(this) == 1 {
            forget(this as usize);
            FREED.lock().unwrap().push(this as usize);
        }
        crate::object::release_boxed::<crate::object::Object>(this)
    }

    /// A null CBV description overwrites the slot (D3D12 null CBV), so a
    /// later read does not see the previous view.
    #[test]
    fn null_cbv_clears_a_stale_view() {
        let null = std::ptr::null();
        let slots = [0u64; 2 * 4];
        let a = slots.as_ptr() as u64;
        create_cbv(null, &[0x1000, 1, 256, 0], a);
        assert_eq!(view(a).as_deref(), Some(&View::Constant(0x1_0000_1000, 256)));
        create_cbv(null, std::ptr::null(), a);
        assert_eq!(view(a).as_deref(), Some(&View::Constant(0, 0)));
        create_srv(null, null, std::ptr::null(), a);
        assert_eq!(view(a).as_deref(), Some(&View::Shader(0, None)));
        create_cbv(null, std::ptr::null(), a);
        assert_eq!(view(a).as_deref(), Some(&View::Constant(0, 0)));
    }

    /// A released resource's views go stale; the same view of a new
    /// resource is a new record.
    #[test]
    fn views_die_with_their_resource() {
        let null = std::ptr::null();
        let slots = [0u64; 2 * 4];
        let (a, b) = (slots.as_ptr() as u64, slots.as_ptr() as u64 + 32);
        let resource = crate::object::test_object(release_resource_like);
        let address = resource as usize;
        create_srv(null, resource, std::ptr::null(), a);
        assert_eq!(view(a).as_deref(), Some(&View::Shader(address, None)));
        // The last release frees the object, and its views go stale.
        assert_eq!(crate::object::release_ref(resource), 0);
        assert!(view(a).is_none(), "stale after release");
        let again = crate::object::test_object(release_resource_like);
        create_srv(null, again, std::ptr::null(), b);
        assert_eq!(
            view(b).as_deref(),
            Some(&View::Shader(again as usize, None))
        );
        // SAFETY: both slots are in `slots`.
        let (old, new) = unsafe { (*(a as *const [u32; 3]), *(b as *const [u32; 3])) };
        assert_ne!(old[1..], new[1..], "a new entry or generation");
        assert_eq!(crate::object::release_ref(again), 0);
    }

    /// A final release with the view pinned does not wait: the title's call
    /// returns at once, the resource is kept alive, and the last pin's drop
    /// does the freeing (which makes the views stale then).
    #[test]
    fn a_pinned_resource_is_freed_by_the_last_pin() {
        let null = std::ptr::null();
        let slots = [0u64; 2 * 4];
        let at = slots.as_ptr() as u64;
        let resource = crate::object::test_object(release_resource_like);
        let address = resource as usize;
        FREED.lock().unwrap().retain(|&a| a != address);
        create_srv(null, resource, std::ptr::null(), at);
        let first = view(at).expect("a current view");
        let second = view(at).expect("the same current view again");

        // The title releases its last reference with both pins held; the
        // call returns while the object is held by them.
        assert_eq!(crate::object::release_ref(resource), 2);
        assert!(!freed(address));
        assert!(view(at).is_some(), "the entry stays current while pinned");
        drop(first);
        assert!(!freed(address));
        drop(second);
        assert!(freed(address), "the last pin freed it");
        assert!(view(at).is_none(), "stale after the last pin");
    }

    /// A record the title copies into its shader-visible heap as bytes (no
    /// call) is made on the host at the next sync, and only then.
    #[cfg(windows)]
    #[test]
    fn records_copied_as_bytes_are_made_at_the_next_sync() {
        let memory = crate::watch::tests::pool(1);
        let desc = HeapDesc {
            kind: SAMPLERS,
            count: 64,
            flags: SHADER_VISIBLE,
            nodes: 1,
        };
        let mut h = std::ptr::null();
        let placement = Placement {
            address: memory as u64,
            size: 64 * 16,
        };
        assert_eq!(create(desc, placement, &mut h), S_OK);
        if let Some(host) = heap(h).host() {
            // Our record, written where no heap sees it, then copied in.
            let mut elsewhere = [0u64; 2];
            let at = elsewhere.as_mut_ptr() as u64;
            create_sampler(std::ptr::null(), &[21, 3, 3, 3, 0, 1, 0, 0, 0, 0, 0, 0x7f7f_ffff, 0], at);
            let slot = memory as u64 + 5 * 16;
            // SAFETY: 16 bytes of each, ours; the heap's page is guarded
            // (our handler opens it).
            unsafe { std::ptr::copy_nonoverlapping(at as *const u8, slot as *mut u8, 16) };
            assert!(!host.made(5));
            sync_heaps();
            assert!(host.made(5), "made at the sync");
        }
        slot::<extern "system" fn(This) -> u32>(h, 0x10)(h);
        crate::watch::tests::free(memory);
    }

    #[cfg(windows)]
    #[test]
    fn sync_heaps_reads_partial_64k_pages_and_small_heaps() {
        use windows_sys::Win32::System::Memory::{MEM_COMMIT, MEM_RELEASE, MEM_RESERVE, PAGE_READWRITE, VirtualAlloc, VirtualFree};
        // Reserve 64 KiB with write-watch, commit only the first 4 KiB so a full
        // heap-range read_into fails and sync_heaps must fall back to 4 KiB pages.
        // SAFETY: fresh reservation freed at the end of the test.
        let base = unsafe {
            VirtualAlloc(
                std::ptr::null(),
                64 << 10,
                MEM_RESERVE | crate::watch::tests::MEM_WRITE_WATCH,
                PAGE_READWRITE,
            )
        } as usize;
        assert_ne!(base, 0);
        // SAFETY: committing the first 4 KiB of our reservation.
        let committed = unsafe { VirtualAlloc(base as *const _, 4 << 10, MEM_COMMIT, PAGE_READWRITE) };
        assert!(!committed.is_null());

        let desc = HeapDesc {
            kind: SAMPLERS,
            count: 512, // 512 * 16 = 8 KiB: spans committed 4 KiB + uncommitted 4 KiB
            flags: SHADER_VISIBLE,
            nodes: 1,
        };
        let mut h = std::ptr::null();
        let placement = Placement {
            address: base as u64,
            size: 512 * 16,
        };
        assert_eq!(create(desc, placement, &mut h), S_OK);
        if let Some(host) = heap(h).host() {
            let mut elsewhere = [0u64; 2];
            let at = elsewhere.as_mut_ptr() as u64;
            create_sampler(std::ptr::null(), &[21, 3, 3, 3, 0, 1, 0, 0, 0, 0, 0, 0x7f7f_ffff, 0], at);
            let slot1 = base as u64 + 16;
            // SAFETY: 16 bytes inside the committed first 4 KiB.
            unsafe { std::ptr::copy_nonoverlapping(at as *const u8, slot1 as *mut u8, 16) };
            assert!(!host.made(1));
            sync_heaps();
            assert!(host.made(1), "partial 64 KiB page / small heap must still sync");
        }
        slot::<extern "system" fn(This) -> u32>(h, 0x10)(h);
        // SAFETY: our reservation.
        unsafe { VirtualFree(base as *mut _, 0, MEM_RELEASE) };
    }

    /// A placed heap reused at the same address replaces the stale range;
    /// descriptor writes go only to the new heap.
    #[cfg(windows)]
    #[test]
    fn overlapping_placed_heaps_mirror_into_the_replacement() {
        let mut memory = vec![0u8; 2 * 32];
        let address = memory.as_mut_ptr() as u64;
        let desc = HeapDesc {
            kind: VIEWS_KIND,
            count: 2,
            flags: SHADER_VISIBLE,
            nodes: 0,
        };
        let placement = Placement {
            address,
            size: 2 * 32,
        };
        let old = owner(desc, placement);
        let new = owner(desc, placement);
        {
            let heaps = SHADER_HEAPS.read().unwrap_or_else(|e| e.into_inner());
            assert_eq!(
                heaps
                    .iter()
                    .filter(|&&(start, end, _)| (start..end).contains(&address))
                    .count(),
                1
            );
            assert_eq!(heap_at(&heaps, address), Some(new.0 as usize));
        }

        create_cbv(std::ptr::null(), &[0x1000, 1, 256, 0], address);
        assert!(heap(old.0).pending.lock().unwrap().is_empty());
        assert_eq!(heap(new.0).pending.lock().unwrap().as_slice(), [0]);

        drop(old);
        create_cbv(std::ptr::null(), &[0x2000, 1, 256, 0], address + 32);
        assert_eq!(heap(new.0).pending.lock().unwrap().as_slice(), [0, 1]);
    }

    /// A creation that does not hand out a heap must not retire an existing
    /// registration for the same memory.
    #[cfg(windows)]
    #[test]
    fn failed_overlapping_creation_keeps_the_existing_heap() {
        let mut memory = vec![0u8; 32];
        let address = memory.as_mut_ptr() as u64;
        let desc = HeapDesc {
            kind: VIEWS_KIND,
            count: 1,
            flags: SHADER_VISIBLE,
            nodes: 0,
        };
        let placement = Placement { address, size: 32 };
        let existing = owner(desc, placement);
        let extended = Extended(desc, placement);
        let bad_iid = [0xff; 16];
        let mut out = std::ptr::null();
        assert_ne!(
            create_heap(std::ptr::null(), &extended.0, &bad_iid, &mut out),
            S_OK
        );
        assert!(out.is_null());
        assert_eq!(
            create_heap(
                std::ptr::null(),
                &extended.0,
                &HEAP_IIDS[0],
                std::ptr::null_mut()
            ),
            crate::object::S_FALSE
        );
        {
            let heaps = SHADER_HEAPS.read().unwrap_or_else(|e| e.into_inner());
            assert_eq!(heap_at(&heaps, address), Some(existing.0 as usize));
        }
        create_cbv(std::ptr::null(), &[0x3000, 1, 256, 0], address);
        assert_eq!(heap(existing.0).pending.lock().unwrap().as_slice(), [0]);
    }

    /// Grand Theft Auto VI places its shader-visible heap in its own memory
    /// and checks that the heap starts there.
    #[test]
    fn placed_heaps_start_at_the_title_memory() {
        let mut memory = vec![0u8; 64 * 32];
        let address = memory.as_mut_ptr() as u64;
        let desc = HeapDesc {
            kind: 0,
            count: 64,
            flags: 1,
            nodes: 1,
        };
        let mut h = std::ptr::null();
        let placement = Placement { address, size: 64 * 32 };
        assert_eq!(create(desc, placement, &mut h), S_OK);
        assert_eq!(slot::<extern "system" fn(This) -> u64>(h, 0x48)(h), address);
        assert_eq!(slot::<extern "system" fn(This) -> u64>(h, 0x50)(h), address);
        // Remove the placed heap from the registry before its memory can
        // be reused by another heap.
        release(h);
        // A size that does not match the count is not a placement.
        assert_eq!(create(desc, Placement { address, size: 7 }, &mut h), S_OK);
        assert_ne!(slot::<extern "system" fn(This) -> u64>(h, 0x48)(h), address);
        release(h);

        #[cfg(windows)]
        {
            // Reuse the address without relying on the allocator to give it
            // to another test. The descriptor write must reach the new heap.
            assert_eq!(create(desc, placement, &mut h), S_OK);
            create_srv(std::ptr::null(), std::ptr::null(), std::ptr::null(), address);
            let reused = heap(h);
            let queued = reused.pending.lock().unwrap().as_slice() == [0];
            release(h);
            assert!(queued, "the descriptor write reached the heap at the reused address");
        }
    }

    #[test]
    fn view_gen_current_tracks_record_forget_and_reuse() {
        #[repr(C, align(8))]
        struct AlignedDesc([u32; 14]);
        let mut slot_a = [0u32; 4];
        let mut slot_b = [0u32; 4];
        let mut res_a = std::ptr::null();
        let mut res_b = std::ptr::null();
        let desc = AlignedDesc([1, 0, 0, 0, 64, 0, 1, 1 | 1 << 16, 0, 1, 0, 0, 0, 0]);
        let properties = [1u32, 0, 0, 0, 0];
        let iid = crate::object::guid(
            0x696442be,
            0xa72e,
            0x4059,
            [0xbc, 0x79, 0x5b, 0x5c, 0x98, 0x04, 0x0f, 0xad],
        );
        assert_eq!(
            crate::resource::create_committed(
                std::ptr::null(),
                properties.as_ptr().cast(),
                0,
                desc.0.as_ptr().cast(),
                0,
                std::ptr::null(),
                &iid,
                &mut res_a,
            ),
            S_OK
        );
        assert_eq!(
            crate::resource::create_committed(
                std::ptr::null(),
                properties.as_ptr().cast(),
                0,
                desc.0.as_ptr().cast(),
                0,
                std::ptr::null(),
                &iid,
                &mut res_b,
            ),
            S_OK
        );

        create_srv(std::ptr::null(), res_a, std::ptr::null(), slot_a.as_mut_ptr() as u64);
        let [tag, idx, gen_a] = [slot_a[0], slot_a[1], slot_a[2]];
        assert_eq!(tag, VIEW_TAG);
        assert!(gen_a >= 1);
        assert!(view_gen_current(idx, gen_a));
        assert!(!view_gen_current(idx, 0));
        assert!(!view_gen_current(idx, gen_a.wrapping_add(1).max(1)));

        let key_a = u64::from(gen_a) << 32 | u64::from(idx);
        let batch = views_at_batch(&[key_a, 0]);
        assert!(batch[0].is_some());
        assert!(batch[1].is_none());
        drop(batch);

        forget(res_a as usize);
        assert!(!view_gen_current(idx, gen_a));
        assert!(views_at_batch(&[key_a])[0].is_none());

        create_srv(std::ptr::null(), res_b, std::ptr::null(), slot_b.as_mut_ptr() as u64);
        let [tag_b, idx_b, gen_b] = [slot_b[0], slot_b[1], slot_b[2]];
        assert_eq!(tag_b, VIEW_TAG);
        assert!(view_gen_current(idx_b, gen_b));
        assert!(!view_gen_current(idx, gen_a));

        crate::object::release(res_a);
        crate::object::release(res_b);
    }

    #[test]
    fn placed_views_ignore_resource_desc_padding() {
        let mut slot = [0u32; 8];
        let handle = slot.as_mut_ptr() as u64;
        let view_desc = [0u32; 16];
        let null = std::ptr::null();
        let expected = [3, 0, 0, 0, 8, 0, 8, 0x1_0001, 28, 1, 0, 0, 0, 0];
        for (unordered, create) in [
            (false, create_placed_srv as extern "system" fn(_, _, _, _)),
            (true, create_placed_uav),
        ] {
            create(null, &expected, &view_desc, handle);
            let first = slot;
            let mut padded = expected;
            padded[1] = 0xaaaa_aaaa;
            padded[13] = 0xbbbb_bbbb;
            create(null, &padded, &view_desc, handle);
            assert_eq!(slot, first, "padding words 1 and 13 must deduplicate");
            assert_eq!(
                view(handle).as_deref(),
                Some(&View::Placed(unordered, expected, view_desc))
            );
            let mut other = expected;
            other[12] = 4; // Flags
            create(null, &other, &view_desc, handle);
            assert_ne!(slot, first, "non-padding Flags word must still differentiate");
        }
    }

    #[test]
    fn views_ignore_unused_srv_and_uav_words() {
        let mut slot = [0u32; 8];
        let handle = slot.as_mut_ptr() as u64;
        let null = std::ptr::null();
        let resource = [1, 0, 0, 0, 0x4000, 0, 1, 0x1_0001, 0, 1, 0, 1, 0, 0];

        // Every known SRV dimension (1..=11): word 3 and words[4+used..10] are zeroed.
        let srv_used: [(u32, usize); 11] = [
            (1, 5),
            (2, 3),
            (3, 5),
            (4, 4),
            (5, 6),
            (6, 0),
            (7, 2),
            (8, 3),
            (9, 3),
            (10, 5),
            (11, 2),
        ];
        for (dim, used) in srv_used {
            let mut clean = [0u32; 16];
            clean[0] = 28;
            clean[1] = dim;
            clean[2] = 0x1688;
            for k in 0..used {
                clean[4 + k] = 10 + k as u32;
            }
            clean[10] = 0x1000 + dim;
            clean[11] = 0x6;
            let mut noisy = clean;
            noisy[3] = 0xdead_beef;
            for w in &mut noisy[4 + used..10] {
                *w = 0xfeed_face;
            }

            create_placed_srv(null, &resource, &clean, handle);
            let first = slot;
            create_placed_srv(null, &resource, &noisy, handle);
            assert_eq!(slot, first, "placed SRV dim {dim} unused words");
            assert_eq!(
                view(handle).as_deref(),
                Some(&View::Placed(false, resource, clean))
            );

            let clean10: [u32; 10] = clean[..10].try_into().unwrap();
            let noisy10: [u32; 10] = noisy[..10].try_into().unwrap();
            create_srv(null, null, &clean10, handle);
            let first_srv = slot;
            create_srv(null, null, &noisy10, handle);
            assert_eq!(slot, first_srv, "resource SRV dim {dim} unused words");
            assert_eq!(view(handle).as_deref(), Some(&View::Shader(0, Some(clean10))));

            if used > 0 {
                let mut diff = clean10;
                diff[4 + used - 1] += 1;
                create_srv(null, null, &diff, handle);
                assert_ne!(slot, first_srv, "SRV dim {dim} last used word must count");
            }
        }

        // Every known UAV dimension (1..=8): words[2+used..10] are zeroed.
        let uav_used: [(u32, usize); 8] = [
            (1, 7),
            (2, 1),
            (3, 3),
            (4, 2),
            (5, 4),
            (6, 0),
            (7, 2),
            (8, 3),
        ];
        for (dim, used) in uav_used {
            let mut clean = [0u32; 16];
            clean[0] = 28;
            clean[1] = dim;
            for k in 0..used {
                clean[2 + k] = 20 + k as u32;
            }
            clean[10] = 0x2000 + dim;
            clean[11] = 0x6;
            let mut noisy = clean;
            for w in &mut noisy[2 + used..10] {
                *w = 0xbadc_0ffe;
            }

            create_placed_uav(null, &resource, &clean, handle);
            let first = slot;
            create_placed_uav(null, &resource, &noisy, handle);
            assert_eq!(slot, first, "placed UAV dim {dim} unused words");
            assert_eq!(
                view(handle).as_deref(),
                Some(&View::Placed(true, resource, clean))
            );

            let clean10: [u32; 10] = clean[..10].try_into().unwrap();
            let noisy10: [u32; 10] = noisy[..10].try_into().unwrap();
            create_uav(null, null, null, &clean10, handle);
            let first_uav = slot;
            create_uav(null, null, null, &noisy10, handle);
            assert_eq!(slot, first_uav, "resource UAV dim {dim} unused words");
            assert_eq!(
                view(handle).as_deref(),
                Some(&View::Unordered(0, 0, Some(clean10)))
            );

            if used > 0 {
                let mut diff = clean10;
                diff[2 + used - 1] += 1;
                create_uav(null, null, null, &diff, handle);
                assert_ne!(slot, first_uav, "UAV dim {dim} last used word must count");
            }
        }

        // Dimension 0 and unknown dimensions preserve every word untouched.
        for (unordered, dim) in [(false, 0u32), (false, 99), (true, 0), (true, 99)] {
            let raw = [28, dim, 0x1688, 0x11, 0x22, 0x33, 0x44, 0x55, 0x66, 0x77];
            let mut out = raw;
            sanitize_view_words(&mut out, unordered);
            assert_eq!(out, raw, "unknown/zero dim {dim} (uav={unordered}) preserved");
        }
    }

    /// Views without a resource object are never freed, however many are
    /// made after them: a slot written once still names its view (the
    /// world's tables went null once #414's cap of them was reached).
    #[test]
    fn ownerless_entries_are_kept() {
        let mut views = Views::default();
        let placed = |n: u32| View::Placed(false, [n; 14], [n; 16]);
        let live_counts = |v: &Views| {
            let count = |f: fn(&View) -> bool| v.index.keys().filter(|k| f(k)).count() as u64;
            [
                count(|k| matches!(k, View::Constant(..))),
                count(|k| matches!(k, View::Placed(..))),
            ]
        };
        let named = |views: &Views, (i, generation): (u32, u32), view: &View| {
            views
                .list
                .get(i as usize)
                .is_some_and(|(v, g)| *g == generation && v == view)
        };
        let first = views.entry_kept(placed(0));
        let constant = views.entry_kept(View::Constant(0x5678, 256));
        for n in 1..5000u32 {
            views.entry_kept(placed(n));
            views.entry_kept(View::Constant(0x1_0000 + 256 * u64::from(n), 256));
        }
        assert!(named(&views, first, &placed(0)), "the first placed view");
        assert!(
            named(&views, constant, &View::Constant(0x5678, 256)),
            "the first constant view"
        );
        // Recording it again finds the same entry.
        assert_eq!(views.entry_kept(placed(0)), first);
        assert_eq!(views.index.len(), 2 * 5000);
        assert_eq!(views.address_entries, [5000, 5000]);
        assert_eq!(views.address_entries, live_counts(&views));
        let _ = views_kept();
    }

    #[test]
    fn rtv_and_dsv_records_ignore_unused_union_words() {
        let mut slot = [0u64; 16];
        let handle = slot.as_mut_ptr() as u64;
        let null = std::ptr::null();

        // RTV dims 1..=8: union starts at word 2.
        for (dim, used) in [
            (1u32, 3usize),
            (2, 1),
            (3, 3),
            (4, 2),
            (5, 4),
            (6, 0),
            (7, 2),
            (8, 3),
        ] {
            let mut clean = [28, dim, 0, 0, 0, 0];
            for k in 0..used {
                clean[2 + k] = 1 + k as u32;
            }
            let mut noisy = clean;
            for w in &mut noisy[2 + used..6] {
                *w = 0xdead_beef;
            }
            create_rtv(null, null, &clean, handle);
            let k_clean = rtv_record(handle).unwrap().key();
            create_rtv(null, null, &noisy, handle);
            let k_noisy = rtv_record(handle).unwrap().key();
            assert_eq!(k_noisy, k_clean, "RTV dim {dim} unused tail");
            if used > 0 {
                let mut diff = clean;
                diff[2 + used - 1] += 1;
                create_rtv(null, null, &diff, handle);
                assert_ne!(rtv_record(handle).unwrap().key(), k_clean, "RTV dim {dim} used word");
            }
        }

        // DSV dims 1..=6: Flags at word 2, union starts at word 3.
        for (dim, used) in [(1u32, 1usize), (2, 3), (3, 1), (4, 3), (5, 0), (6, 2)] {
            let mut clean = [40, dim, 1, 0, 0, 0];
            for k in 0..used {
                clean[3 + k] = 5 + k as u32;
            }
            let mut noisy = clean;
            for w in &mut noisy[3 + used..6] {
                *w = 0xfeed_face;
            }
            create_dsv(null, null, &clean, handle);
            let k_clean = dsv_record(handle).unwrap().key();
            create_dsv(null, null, &noisy, handle);
            let k_noisy = dsv_record(handle).unwrap().key();
            assert_eq!(k_noisy, k_clean, "DSV dim {dim} unused tail");
            if used > 0 {
                let mut diff = clean;
                diff[3 + used - 1] += 1;
                create_dsv(null, null, &diff, handle);
                assert_ne!(dsv_record(handle).unwrap().key(), k_clean, "DSV dim {dim} used word");
            }
            // DSV Flags (word 2) must always be preserved.
            let mut flags_diff = clean;
            flags_diff[2] ^= 2;
            create_dsv(null, null, &flags_diff, handle);
            assert_ne!(dsv_record(handle).unwrap().key(), k_clean, "DSV dim {dim} Flags");
        }

        // Unknown / zero dimensions keep every word; null desc records zeroed words.
        for (depth, dim) in [(false, 0u32), (false, 99), (true, 0), (true, 99)] {
            let raw = [28, dim, 1, 2, 3, 4];
            let mut out = raw;
            sanitize_target_words(&mut out, depth);
            assert_eq!(out, raw);
        }
        create_rtv(null, null, std::ptr::null(), handle);
        assert_eq!(rtv_record(handle).unwrap().key(), (0, [0; 7]));
        create_dsv(null, null, std::ptr::null(), handle);
        assert_eq!(dsv_record(handle).unwrap().key(), (0, [0; 7]));
    }

    /// A table made afresh for a draw holds the descriptors its shader
    /// moves halves through (`HostShader::half_views`), also past the range
    /// starts its used-resource entries give: with one entry for a range
    /// from dword 0 and a 16-bit float buffer view at dword 8, the table
    /// takes two dynamic slots, so the next allocation does not land on the
    /// view's.
    #[cfg(windows)]
    #[test]
    fn a_fresh_table_holds_the_half_views_past_its_range_starts() {
        use victory_shader::{ResourceType, RootType, UsedResource};
        let Some(device) = crate::host::host() else {
            return; // no host GPU
        };
        let list = device.new_list().expect("a host list");
        let desc = HeapDesc {
            kind: VIEWS_KIND,
            count: 2,
            flags: SHADER_VISIBLE,
            nodes: 1,
        };
        let none = Placement {
            address: 0,
            size: 0,
        };
        let mut h: This = std::ptr::null();
        assert_eq!(create(desc, none, &mut h), S_OK);
        let title = heap(h);
        let host_heap = title.host().expect("a host view heap");
        let (first, second) = (title.start, title.start + 32);
        // An R16G16B16A16_FLOAT buffer view of memory in the second slot.
        let memory = [0u64; 64];
        let address = memory.as_ptr() as u64;
        let resource = [1, 0, 0, 0, 512, 0, 1, 0x1_0001, 0, 1, 0, 1, 0, 0];
        let mut words = [0u32; 16];
        words[..10].copy_from_slice(&[10, 1, 5768, 0, 0, 0, 64, 0, 0, 0]);
        words[10] = address as u32;
        words[11] = (address >> 32) as u32;
        create_placed_srv(std::ptr::null(), &resource, &words, second);
        assert!(view(second).as_deref().and_then(host::half_view).is_some());
        let mut state = crate::draw::State::default();
        state.set_heaps(&[h]);
        state.set_table(true, 1, first);
        let range = UsedResource {
            kind: ResourceType::Srv,
            root: RootType::Table,
            table_offset_dwords: 0,
            lower_bound: 0,
            stride: 8,
            dwords: 4,
            user_data: 4,
            slot: 1,
        };
        let made = state.tables(&list, &[range], &[], true, &[(1, 8)], true);
        let [(1, offset)] = made[..] else {
            panic!("one fresh table for root parameter 1, not {made:?}");
        };
        assert_ne!(offset, u32::MAX, "the table found no dynamic slots");
        let next = host_heap.dynamic(1).expect("a dynamic slot");
        host_heap.release(host_heap.id(), next, 1);
        drop(state);
        slot::<extern "system" fn(This) -> u32>(h, 0x10)(h);
        assert_eq!(
            next,
            offset / 32 + 2,
            "the next allocation took the table's second slot"
        );
    }

    thread_local! {
        /// Set in a test: every view of that test's thread is a table entry,
        /// as with `VICTORY_VIEW_INLINE=0`.
        pub(crate) static ENTRIES_ONLY: std::cell::Cell<bool> =
            const { std::cell::Cell::new(false) };
    }

    /// The record of a view written to a fresh slot, and the view it reads back as.
    fn recorded(view: View) -> ([u32; 3], Option<View>) {
        let mut slot = [0u32; 4];
        let handle = slot.as_mut_ptr() as u64;
        record(handle, view);
        let read = super::view(handle).map(|pin| (*pin).clone());
        ([slot[0], slot[1], slot[2]], read)
    }

    /// Constant buffer views at a new address each, as a title writes them
    /// every frame, are named whole in their records: the table does not
    /// grow with them (it grew 0.28 GB a minute in a title when it did),
    /// and they read back as themselves. As entries, each is one more. Kept
    /// small: the heap sync tests beside it are sensitive to load.
    #[test]
    fn constant_buffer_views_at_new_addresses_add_no_table_entries() {
        // Constant buffer entries are never freed, so their count only
        // grows (the table's length need not: entries freed by other tests
        // are taken again).
        let constant_entries = || views_kept()[2];
        let n = 3_000u64;
        let before = constant_entries();
        for k in 0..n {
            let mut slot = [0u32; 4];
            let address = 0x3_0000_0000 + 256 * k;
            let desc = [address as u32, (address >> 32) as u32, 256, 0];
            create_cbv(std::ptr::null(), &desc, slot.as_mut_ptr() as u64);
            assert_eq!(slot[2] >> KIND_SHIFT, CONSTANT);
            if k % 100 == 0 {
                assert_eq!(
                    view(slot.as_mut_ptr() as u64).as_deref(),
                    Some(&View::Constant(address, 256))
                );
            }
        }
        // Other tests may add entries meanwhile; these 3,000 add none.
        assert!(constant_entries() - before < 100, "the table grew with constant buffer views");
        ENTRIES_ONLY.with(|only| only.set(true));
        let before = constant_entries();
        for k in 0..100 {
            let (record, read) = recorded(View::Constant(0x4_0000_0000 + 256 * k, 256));
            assert_eq!(record[2] >> KIND_SHIFT, 0, "an entry with VICTORY_VIEW_INLINE=0");
            assert_eq!(read, Some(View::Constant(0x4_0000_0000 + 256 * k, 256)));
        }
        assert!(constant_entries() - before >= 100);
        ENTRIES_ONLY.with(|only| only.set(false));
    }

    /// Every size below 16 KiB and the multiples of 256 below 4 MiB fit in a
    /// record; other sizes and addresses past 47 bits are table entries, and
    /// read back the same either way.
    #[test]
    fn constant_buffer_views_read_back_whole_or_as_entries() {
        for (address, size, whole) in [
            (0, 0, true),
            (0x1000, 1, true),
            (0x7fff_ffff_ff00, 256, true),
            (0x2000, (1 << 14) - 1, true),
            (0x2000, 1 << 14, true),
            (0x2000, (4 << 20) - 256, true),
            (0x2000, 4 << 20, false),
            (0x2000, (1 << 14) + 4, false),
            (1 << 47, 256, false),
        ] {
            assert_eq!(constant_words(address, size).is_some(), whole, "{address:#x} {size}");
            let (record, read) = recorded(View::Constant(address, size));
            assert_eq!(record[2] >> KIND_SHIFT == CONSTANT, whole, "{address:#x} {size}");
            assert_eq!(read, Some(View::Constant(address, size)), "{address:#x} {size}");
        }
    }

    /// Views of placed memory are named by their address and the number of
    /// their shape: moved, the same shape is not entered again; past 40
    /// address bits the view is a table entry.
    #[test]
    fn placed_views_are_named_by_address_and_shape() {
        let resource = [7u32, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 0];
        let desc = |address: u64| {
            let mut d = [0x5151u32; 16];
            (d[10], d[11]) = (address as u32, (address >> 32) as u32);
            d
        };
        let (first, read) = recorded(View::Placed(false, resource, desc(0x10_0000)));
        assert_eq!(first[2] >> KIND_SHIFT, PLACED);
        assert_eq!(read, Some(View::Placed(false, resource, desc(0x10_0000))));
        let shape = |record: [u32; 3]| payload(record[1], record[2]) >> PLACED_ADDRESS_BITS;
        for k in 1..500u64 {
            let address = 0x10_0000 + (k << 16);
            let (record, read) = recorded(View::Placed(false, resource, desc(address)));
            assert_eq!(record[2] >> KIND_SHIFT, PLACED);
            assert_eq!(shape(record), shape(first), "one shape for all 500");
            assert_eq!(read, Some(View::Placed(false, resource, desc(address))));
        }
        let (other, read) = recorded(View::Placed(true, resource, desc(0x10_0000)));
        assert_ne!(other[1..], first[1..], "unordered is another shape");
        assert_eq!(read, Some(View::Placed(true, resource, desc(0x10_0000))));
        let far = 1u64 << 40;
        let (record, read) = recorded(View::Placed(false, resource, desc(far)));
        assert_eq!(record[2] >> KIND_SHIFT, 0, "an entry past 40 address bits");
        assert_eq!(read, Some(View::Placed(false, resource, desc(far))));
    }

    /// The heap syncs' checks take whole records as current and find their
    /// views by key; a placed record of a shape not entered names nothing.
    #[test]
    fn records_named_whole_are_current_for_the_heap_syncs() {
        let ([_, i, generation], _) = recorded(View::Constant(0x9000, 512));
        assert!(view_gen_current(i, generation));
        let key = u64::from(generation) << 32 | u64::from(i);
        let pins = views_at_batch(&[key]);
        assert_eq!(pins[0].as_deref(), Some(&View::Constant(0x9000, 512)));
        let unknown = kind_words(PLACED, ((SHAPE_LIMIT - 1) as u64) << PLACED_ADDRESS_BITS);
        assert!(!view_gen_current(unknown[0], unknown[1]));
        assert!(view_at(unknown[0], unknown[1]).is_none());
        #[cfg(windows)]
        {
            let mut current = Vec::new();
            views_current_into(
                &[[VIEW_TAG, i, generation], [VIEW_TAG, unknown[0], unknown[1]]],
                &mut current,
            );
            assert_eq!(current, [true, false]);
        }
    }

    /// A table entry's generation never reaches the kind bits or 0.
    #[test]
    fn table_generations_stay_below_the_kind_bits() {
        assert_eq!(next_generation(1), 2);
        assert_eq!(next_generation(GENERATION_MASK), 1);
        assert_eq!(next_generation(u32::MAX), 1);
    }
}
