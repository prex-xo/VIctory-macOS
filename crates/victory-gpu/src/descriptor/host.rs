//! Host descriptor heaps behind the title's shader-visible heaps (decision
//! 0007, step 3). Translated shaders index the heaps directly at the
//! title's descriptor positions, so each host heap keeps the title heap's
//! order: slot `i` is made from what the title wrote at slot `i`, when it
//! writes it (view creation and descriptor copies) and again when a draw
//! reads it through a table and finds the slot changed.

use std::collections::HashMap;
use std::sync::Mutex;
use std::sync::atomic::{AtomicU64, Ordering};

use windows::Win32::Graphics::Direct3D12::*;
use windows::Win32::Graphics::Dxgi::Common::*;

use super::paged::Paged;
use super::{
    VIEW_TAG, View, ViewPin, view_at, view_gen_current, views_at_batch, views_current_into,
    views_forgot,
};
use crate::backend::{HostGpu, HostMemory};
use crate::host::host;

const SYNC_BLOCK_SLOTS: u32 = 128;
/// The widest span of stale slots a heap sync's batch reads in one call
/// (`mirror_synced_batch`); its callers pass a page's slots (64 KiB).
const SPAN_READ_MAX: u64 = 1 << 20;

thread_local! {
    /// True while this thread is inside [`HostHeap::publish`], preventing
    /// `mirror(..., true)` from re-enqueuing the slot being published.
    static PUBLISHING: std::cell::Cell<bool> = const { std::cell::Cell::new(false) };
}

struct PublishingGuard(bool);

impl Drop for PublishingGuard {
    fn drop(&mut self) {
        PUBLISHING.set(self.0);
    }
}

/// Whether due texture slots in [`UploadQueue`] alternate retries and fresh
/// slots 1:1 (`VICTORY_FAIR_DUE=0` restores retries-first ordering).
fn fair_due() -> bool {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| std::env::var_os("VICTORY_FAIR_DUE").is_none_or(|v| v != "0"))
}

/// Maximum backoff interval in milliseconds for re-checking unmapped slots
/// (`VICTORY_RELOOK_MS`, default 20,000 ms).
pub(crate) fn relook_ms() -> u64 {
    static MS: std::sync::OnceLock<u64> = std::sync::OnceLock::new();
    *MS.get_or_init(|| {
        std::env::var("VICTORY_RELOOK_MS")
            .ok()
            .and_then(|v| v.trim().parse().ok())
            .unwrap_or(20_000)
    })
}

/// Backoff cap in presented frames for a frame interval `frame_ms` and relook
/// ceiling `relook_ms`.
pub(crate) fn relook_frames(frame_ms: u64, relook_ms: u64) -> u64 {
    (relook_ms / frame_ms.max(1)).clamp(16, 1024)
}

/// Classification of an unmade slot reported by GPU read-marks when the
/// background heap worker is enabled.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum UnmadeRead {
    None,
    Background,
    Sync,
}

#[derive(Clone, Copy)]
struct QueuedSlot {
    write: u64,
    due_frame: u64,
    looks: u32,
    urgent: bool,
}

/// Queue of shader-visible texture slots waiting for background creation and
/// detiling (`crate::heap_worker`).
///
/// Serves GPU read-marked (`urgent`) slots first, then alternates previously
/// retried slots (`again`, newest write first) 1:1 with never-looked-at slots
/// (`fresh`, alternating newest/oldest write) so streaming backlogs never
/// starve newly visible textures.
#[derive(Default)]
pub(crate) struct UploadQueue {
    next_write: u64,
    slots: HashMap<u32, QueuedSlot>,
}

impl UploadQueue {
    pub(crate) fn written(&mut self, index: u32, frame: u64) {
        self.next_write += 1;
        self.slots.insert(
            index,
            QueuedSlot {
                write: self.next_write,
                due_frame: frame,
                looks: 0,
                urgent: false,
            },
        );
    }

    pub(crate) fn prioritize(&mut self, indices: &[u32], frame: u64) {
        for &index in indices {
            match self.slots.get_mut(&index) {
                Some(entry) => {
                    entry.urgent = true;
                    entry.due_frame = entry.due_frame.min(frame);
                }
                None => {
                    self.next_write += 1;
                    self.slots.insert(
                        index,
                        QueuedSlot {
                            write: self.next_write,
                            due_frame: frame,
                            looks: 0,
                            urgent: true,
                        },
                    );
                }
            }
        }
    }

    #[allow(dead_code)] // hook for urgent bake-source pre-upload
    pub(crate) fn requeue(&mut self, indices: &[u32], frame: u64) {
        for &index in indices {
            if !self.slots.contains_key(&index) {
                self.next_write += 1;
                self.slots.insert(
                    index,
                    QueuedSlot {
                        write: self.next_write,
                        due_frame: frame,
                        looks: 0,
                        urgent: true,
                    },
                );
            }
        }
    }

    #[allow(dead_code)] // hook for urgent bake-source pre-upload
    pub(crate) fn queued(&self, indices: &[u32]) -> Vec<(u32, u64)> {
        indices
            .iter()
            .filter_map(|&index| self.slots.get(&index).map(|w| (index, w.write)))
            .collect()
    }

    pub(crate) fn remove(&mut self, index: u32) {
        self.slots.remove(&index);
    }

    pub(crate) fn done(&mut self, index: u32, write: u64) {
        if self.slots.get(&index).is_some_and(|w| w.write == write) {
            self.slots.remove(&index);
        }
    }

    pub(crate) fn later(&mut self, index: u32, write: u64, frame: u64, max_wait: u64) {
        if let Some(entry) = self.slots.get_mut(&index)
            && entry.write == write
        {
            entry.looks = entry.looks.saturating_add(1);
            entry.urgent = false;
            let backoff = (1u64 << entry.looks.saturating_sub(1).min(10)).min(max_wait.max(1));
            entry.due_frame = frame.saturating_add(backoff);
        }
    }

    pub(crate) fn due(&self, frame: u64, max: usize) -> Vec<(u32, u64)> {
        self.due_with(frame, max, fair_due())
    }

    pub(crate) fn due_with(&self, frame: u64, max: usize, fair: bool) -> Vec<(u32, u64)> {
        if max == 0 || self.slots.is_empty() {
            return Vec::new();
        }
        let mut urgent: Vec<(u32, u64)> = Vec::new();
        let mut again: Vec<(u32, u64)> = Vec::new();
        let mut fresh: Vec<(u32, u64)> = Vec::new();
        for (&index, w) in &self.slots {
            if w.due_frame > frame {
                continue;
            }
            if w.urgent {
                urgent.push((index, w.write));
            } else if w.looks > 0 {
                again.push((index, w.write));
            } else {
                fresh.push((index, w.write));
            }
        }
        let newest_first = |a: &(u32, u64), b: &(u32, u64)| b.1.cmp(&a.1);
        urgent.sort_unstable_by(newest_first);
        again.sort_unstable_by(newest_first);
        fresh.sort_unstable_by(newest_first);

        let mut order = Vec::with_capacity(max.min(urgent.len() + again.len() + fresh.len()));
        order.extend(urgent.into_iter().take(max));
        if order.len() >= max {
            return order;
        }
        let remaining = max - order.len();
        let mut fresh_order: Vec<(u32, u64)> = Vec::with_capacity(fresh.len().min(remaining));
        let (mut new, mut old) = (0, fresh.len());
        while new < old && fresh_order.len() < remaining {
            fresh_order.push(fresh[new]);
            new += 1;
            if new < old && fresh_order.len() < remaining {
                old -= 1;
                fresh_order.push(fresh[old]);
            }
        }
        if !fair {
            order.extend(again.into_iter().chain(fresh_order).take(remaining));
            return order;
        }
        let mut again_iter = again.into_iter();
        let mut fresh_iter = fresh_order.into_iter();
        while order.len() < max {
            let (a, b) = (again_iter.next(), fresh_iter.next());
            if a.is_none() && b.is_none() {
                break;
            }
            order.extend(a);
            if order.len() < max {
                order.extend(b);
            }
        }
        order
    }

    pub(crate) fn is_empty(&self) -> bool {
        self.slots.is_empty()
    }
}


/// Pre-populated null descriptor templates in [`StagingHeap`] (slot 0 for
/// sampler heaps; slots `0..17` covering every null SRV/UAV kind for view
/// heaps). Remaining slots hold per-batch templates for repeated views.
const STAGING_NULL_SLOTS: u32 = 17;
const STAGING_TOTAL_SLOTS: u32 = 256;

fn fast_heap_sync_value(value: Option<&std::ffi::OsStr>) -> bool {
    value != Some(std::ffi::OsStr::new("0"))
}

fn fast_heap_sync() -> bool {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| fast_heap_sync_value(std::env::var_os("VICTORY_FAST_HEAP_SYNC").as_deref()))
}

/// Policy for mirroring placed-texture slots during non-draw updates
/// (`VICTORY_PLACED_AT_ONCE`, off by default).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum PlacedAtOnce {
    /// Unset, empty, or `0`: leave placed-texture slots null until a draw or
    /// shader read triggers creation.
    Off,
    /// `1`: make the slot at once when the placed texture's host texture
    /// already exists (creating no new host texture).
    Made,
    /// `keep`: like `1`, and when the new texture is not made yet, keep any
    /// existing host view (and its write mark) in the slot instead of writing
    /// a null view until a shader read or draw remakes it on demand.
    Keep,
    /// `all`: make every placed-texture slot at once.
    All,
}

impl PlacedAtOnce {
    fn parse(value: Option<&str>) -> Self {
        match value.map(str::trim) {
            Some("1") => Self::Made,
            Some("keep") => Self::Keep,
            Some("all") => Self::All,
            _ => Self::Off,
        }
    }

    fn makes_existing(self) -> bool {
        self != Self::Off
    }
}

#[cfg(test)]
thread_local! {
    static TEST_PLACED_AT_ONCE: std::cell::Cell<Option<PlacedAtOnce>> = const { std::cell::Cell::new(None) };
}

fn placed_at_once() -> PlacedAtOnce {
    #[cfg(test)]
    if let Some(mode) = TEST_PLACED_AT_ONCE.get() {
        return mode;
    }
    static MODE: std::sync::OnceLock<PlacedAtOnce> = std::sync::OnceLock::new();
    *MODE.get_or_init(|| {
        let mode = PlacedAtOnce::parse(std::env::var("VICTORY_PLACED_AT_ONCE").ok().as_deref());
        match mode {
            PlacedAtOnce::Off => {}
            PlacedAtOnce::Made => tracing::info!(
                "VICTORY_PLACED_AT_ONCE=1: making rewritten placed-texture slots at once when their host texture already exists"
            ),
            PlacedAtOnce::Keep => tracing::info!(
                "VICTORY_PLACED_AT_ONCE=keep: making existing placed textures at once and keeping prior host views until read"
            ),
            PlacedAtOnce::All => tracing::info!(
                "VICTORY_PLACED_AT_ONCE=all: making every rewritten placed-texture slot at once"
            ),
        }
        mode
    })
}

fn made_at_once(view: &View, mode: PlacedAtOnce) -> bool {
    if !mode.makes_existing() {
        return false;
    }
    let View::Placed(_, resource, desc) = view else {
        return false;
    };
    crate::resource::placed_made(u64::from(desc[10]) | u64::from(desc[11]) << 32, resource)
}

fn defer_placed(view: &View, mode: PlacedAtOnce) -> bool {
    is_placed_texture(view) && mode != PlacedAtOnce::All && !made_at_once(view, mode)
}

/// CPU-only (`D3D12_DESCRIPTOR_HEAP_FLAG_NONE`) staging heap holding null
/// templates (`0..STAGING_NULL_SLOTS`) and scratch template slots for
/// `CopyDescriptorsSimple` during batched heap syncs.
struct StagingHeap {
    _heap: ID3D12DescriptorHeap,
    start: usize,
    step: u32,
    kind: D3D12_DESCRIPTOR_HEAP_TYPE,
    count: u32,
}

impl StagingHeap {
    fn new(sampler: bool, step: u32) -> Option<Self> {
        let host = host()?;
        let kind = if sampler {
            D3D12_DESCRIPTOR_HEAP_TYPE_SAMPLER
        } else {
            D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV
        };
        // SAFETY: CPU-only staging descriptor heap; valid source for
        // `CopyDescriptorsSimple` into shader-visible heaps.
        let heap: ID3D12DescriptorHeap = unsafe {
            host.device
                .CreateDescriptorHeap(&D3D12_DESCRIPTOR_HEAP_DESC {
                    Type: kind,
                    NumDescriptors: STAGING_TOTAL_SLOTS,
                    Flags: D3D12_DESCRIPTOR_HEAP_FLAG_NONE,
                    NodeMask: 0,
                })
        }
        .ok()?;
        // SAFETY: live heap.
        let start = unsafe { heap.GetCPUDescriptorHandleForHeapStart() }.ptr;
        let staging = Self {
            _heap: heap,
            start,
            step,
            kind,
            count: STAGING_TOTAL_SLOTS,
        };
        if sampler {
            null_sampler(staging.cpu(0));
        } else {
            null_view(staging.cpu(0));
            null_buffer_view(staging.cpu(1));
            for dim in 2..=10u32 {
                let mut words = [0u32; 10];
                words[1] = dim;
                null_like(&words, staging.cpu(dim));
            }
            let buf_uav = D3D12_UNORDERED_ACCESS_VIEW_DESC {
                Format: DXGI_FORMAT_R32_UINT,
                ViewDimension: D3D12_UAV_DIMENSION_BUFFER,
                ..Default::default()
            };
            let uav_1d = null_uav_desc(Some(D3D12_UAV_DIMENSION_TEXTURE1D.0 as u32));
            let uav_1d_arr = null_uav_desc(Some(D3D12_UAV_DIMENSION_TEXTURE1DARRAY.0 as u32));
            let uav_2d_arr = null_uav_desc(Some(D3D12_UAV_DIMENSION_TEXTURE2DARRAY.0 as u32));
            let uav_3d = null_uav_desc(Some(D3D12_UAV_DIMENSION_TEXTURE3D.0 as u32));
            let uav_2d = null_uav_desc(None);
            // SAFETY: null resources with explicit descriptions in our staging heap.
            unsafe {
                host.device.CreateUnorderedAccessView(
                    None::<&ID3D12Resource>,
                    None::<&ID3D12Resource>,
                    Some(&buf_uav),
                    staging.cpu(11),
                );
                host.device.CreateUnorderedAccessView(
                    None::<&ID3D12Resource>,
                    None::<&ID3D12Resource>,
                    Some(&uav_1d),
                    staging.cpu(12),
                );
                host.device.CreateUnorderedAccessView(
                    None::<&ID3D12Resource>,
                    None::<&ID3D12Resource>,
                    Some(&uav_1d_arr),
                    staging.cpu(13),
                );
                host.device.CreateUnorderedAccessView(
                    None::<&ID3D12Resource>,
                    None::<&ID3D12Resource>,
                    Some(&uav_2d_arr),
                    staging.cpu(14),
                );
                host.device.CreateUnorderedAccessView(
                    None::<&ID3D12Resource>,
                    None::<&ID3D12Resource>,
                    Some(&uav_3d),
                    staging.cpu(15),
                );
                host.device.CreateUnorderedAccessView(
                    None::<&ID3D12Resource>,
                    None::<&ID3D12Resource>,
                    Some(&uav_2d),
                    staging.cpu(16),
                );
            }
        }
        Some(staging)
    }

    fn cpu(&self, index: u32) -> D3D12_CPU_DESCRIPTOR_HANDLE {
        D3D12_CPU_DESCRIPTOR_HANDLE {
            ptr: self.start + index as usize * self.step as usize,
        }
    }
}

/// Slot index in [`StagingHeap`] matching [`null_for`] / [`null_view`] /
/// [`null_sampler`].
fn null_slot_index(sampler: bool, view: Option<&View>) -> u32 {
    if sampler {
        return 0;
    }
    let Some(view) = view else {
        return 0;
    };
    let is_buffer = |r: &usize| crate::resource::buffer_bytes(*r as crate::object::This).is_some();
    let (buffer, dimension) = match view {
        View::Unordered(r, _, desc) => (
            is_buffer(r) || desc.is_some_and(|d| d[1] == 1),
            desc.map(|d| d[1]),
        ),
        View::Placed(true, resource, desc) => (resource[0] == 1, Some(desc[1])),
        View::Shader(r, desc) if is_buffer(r) || desc.is_some_and(|d| d[1] == 1) => return 1,
        View::Placed(false, resource, _) if resource[0] == 1 => return 1,
        View::Shader(_, Some(desc)) => {
            return if (2..=10).contains(&desc[1]) { desc[1] } else { 0 };
        }
        View::Placed(false, _, desc) => {
            return if (2..=10).contains(&desc[1]) { desc[1] } else { 0 };
        }
        View::Constant(..) => return 1,
        _ => return 0,
    };
    if buffer {
        return 11;
    }
    match dimension.map(|d| D3D12_UAV_DIMENSION(d as i32)) {
        Some(D3D12_UAV_DIMENSION_TEXTURE1D) => 12,
        Some(D3D12_UAV_DIMENSION_TEXTURE1DARRAY) => 13,
        Some(D3D12_UAV_DIMENSION_TEXTURE2DARRAY) => 14,
        Some(D3D12_UAV_DIMENSION_TEXTURE3D) => 15,
        _ => 16,
    }
}
fn sync_fast_enabled() -> bool {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| std::env::var_os("VICTORY_SYNC_FAST").as_deref() != Some(std::ffi::OsStr::new("0")))
}

/// Whether heap syncs leave a made slot as it is when the title wrote bytes
/// there that are no record of ours (zeros, or its own data between its
/// descriptors): `VICTORY_STALE_LEFT=1`. Most slots a sync made were such
/// in a title's world (32,000 of 47,000 a frame in 64bitMango's fork, where
/// leaving them took heap syncs from 35.4 to 23.4 ms a frame), and no draw
/// read one through a table. Left, the slot keeps the view it had: a draw
/// that reads it through a table makes it null then ([`HostHeap::mirror`]
/// finds no record of ours), a title thread writing a view there makes it,
/// and the title writing back the record it was made from costs nothing. A
/// record of ours that names no view (of a released resource) is still
/// made null.
/// shortcut: a shader that takes such a slot from the heap at an index from
/// its data reads the view the slot had, not nothing; and what the slot
/// views stays entered (`crate::bindless`), so its memory is kept current
/// until the slot is written again.
fn stale_left() -> bool {
    // Tests choose for themselves, so the tests of making such slots null
    // still run whatever the default.
    #[cfg(test)]
    {
        tests::STALE_LEFT.with(std::cell::Cell::get)
    }
    #[cfg(not(test))]
    {
        static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
        *ON.get_or_init(|| crate::switch::on("VICTORY_STALE_LEFT"))
    }
}

/// Monotonic identity for each [`HostHeap`] so stale dynamic-slot releases
/// targeting a released heap's recycled address never free a newer heap's slots.
static NEXT_ID: AtomicU64 = AtomicU64::new(1);

pub(crate) struct HostHeap {
    id: u64,
    pub(crate) heap: ID3D12DescriptorHeap,
    cpu: usize,
    step: u32,
    /// Slots mirroring the title's; view heaps have `dynamic` more.
    count: u32,
    /// Dynamic slots past `count` (none in sampler heaps): [`DYNAMIC`]
    /// where the host takes them, else [`MIN_DYNAMIC`].
    dynamic: u32,
    sampler: bool,
    /// The record (view index and generation) each slot was last made from.
    made: Mutex<Vec<u64>>,
    /// Lock-free mirror of `made`, including texture views left null until
    /// drawn or evicted: published last under `made` and cleared first, so
    /// heap syncs skip already-synced slots without locking `made` or `VIEWS`.
    synced: Box<[AtomicU64]>,
    /// One bit per slot: whether `made[index] != 0`, so empty slots skip
    /// `made` and `VIEWS` locks during `sync_slots`.
    made_nonzero: Box<[AtomicU64]>,
    /// Per `SYNC_BLOCK_SLOTS` block, the `views_forgot()` epoch last verified.
    block_forgot: Box<[AtomicU64]>,
    /// Per title slot, its record once the slot is made from it and holds
    /// what a draw reads (a texture's view not left null), else 0: what
    /// [`HostHeap::mirror`] finds without the `made` lock, which every
    /// title thread binding or writing views would otherwise wait on. Set
    /// last, under that lock, once the slot is made; cleared first.
    ready: Box<[AtomicU64]>,
    /// CPU-only staging heap for null and repeated-view template copies during
    /// batched heap syncs, created on first use.
    staging: std::sync::OnceLock<Option<StagingHeap>>,
    /// The host resource each slot views, held while the slot does: shaders
    /// index heaps directly, so a slot the title has not rewritten since it
    /// released the resource may still be read, and a freed host resource
    /// read there is a GPU page fault (the console reads stale memory).
    /// With it the view's own copy of title memory, if it has one. Paged:
    /// the title views through a fraction of the slots, and a dense table
    /// of them all was 37.7 MB, written when the heap was made.
    kept: Mutex<Paged<Held>>,
    /// Unordered views of textures left null by [`HostHeap::evict`]
    /// (`VICTORY_EVICT_HEAP_UAV`), made again by
    /// [`HostHeap::remake_unordered`] once shaders write through heaps.
    nulled_unordered: Mutex<Vec<u32>>,
    /// Where the search for free dynamic slots starts, and which are taken
    /// (by lists not yet reset), a bit for each.
    next: Mutex<(u32, Vec<u64>)>,
    /// Slot pairs kept for host copies' views (`crate::memory`), past the
    /// dynamic slots and below the null buffer views: the next never
    /// taken, those given back, and where they end.
    pairs: Mutex<(u32, Vec<u32>)>,
    pairs_end: u32,
    /// Null raw SRV for explicitly bound root descriptors. Its UAV is one slot below.
    null_root_srv: Option<u32>,
    /// View heaps: the view-stride table translated shaders read
    /// (`victory_shader::emit::VIEW_STRIDES`), a dword per title slot: the
    /// byte stride of the buffer view there, 0 for none. Mapped at the
    /// address kept beside it.
    strides: Option<(ID3D12Resource, usize)>,
    /// View heaps: the view-mark table translated shaders read
    /// (`victory_shader::emit`, "Write marks"), two dwords per slot as the
    /// stride table has one: the mark bit and phase of the unordered buffer
    /// view there, `VIEW_MARK_NONE` for none. Mapped at the address kept
    /// beside it.
    marks: Option<(ID3D12Resource, usize)>,
    /// Per title slot and per dynamic slot (tables made afresh lie there,
    /// and shaders index the view addresses at the slots their tables are
    /// in), the console descriptor's dwords 0-3 as far as the host knows
    /// them (shaders reading a descriptor as data, `crate::memory`;
    /// [`write_address`]): the buffer, its mapping and host address, made on
    /// first use.
    addresses: std::sync::OnceLock<Option<(ID3D12Resource, usize, u64)>>,
    /// What the title slots view that title memory feeds (`crate::bindless`).
    pub(crate) viewed: std::sync::Arc<crate::bindless::Viewed>,
    /// Live widened SRV buffers, by host address, with the number of slots
    /// showing each. Heap-indexing shaders can read any of them.
    widened: Mutex<HashMap<usize, (usize, std::sync::Weak<crate::shadow::Widened>)>>,
    /// Slots queued for background texture upload (`crate::heap_worker`).
    uploads: Mutex<UploadQueue>,
    /// Slots left showing what they showed before, because the texture
    /// their new record views was made while a submission may still run
    /// ([`show_after`]): slot, and the title's handle and record. Made at
    /// the next submission ([`HostHeap::show_held`]).
    held: Mutex<HashMap<u32, (u64, u64)>>,
}

/// Host heaps' GPU ranges (start, bytes, bytes per slot, sampler), for
/// device-removed reports: a shader indexing a heap past its end faults.
static GPU_RANGES: Mutex<Vec<(u64, u64, u64, bool)>> = Mutex::new(Vec::new());

/// For device-removed reports: where host GPU address `va` lies against
/// the host descriptor heaps.
pub(crate) fn near_heaps(va: u64) -> String {
    let ranges = GPU_RANGES.lock().unwrap_or_else(|e| e.into_inner());
    let mut text = String::new();
    for &(start, bytes, step, sampler) in ranges.iter() {
        let kind = if sampler { "sampler" } else { "view" };
        let place = if va < start {
            format!("{:#x} below", start - va)
        } else if va < start + bytes {
            format!("inside, slot {}", (va - start) / step)
        } else {
            format!(
                "{:#x} past its end (slot {})",
                va - start - bytes,
                (va - start) / step
            )
        };
        text.push_str(&format!("; {kind} heap {start:#x}+{bytes:#x}: {place}"));
    }
    text
}

/// The host's limit for shader-visible view heaps on resource binding
/// tiers 1 and 2; tier 3 hardware may take more.
const PORTABLE_VIEWS: u32 = 1_000_000;

/// Slots past the title's in a view heap, for tables a draw needs made
/// afresh (constant buffers copied per draw). A list holds the slots it
/// took until it is reset (its submissions have run) or released.
/// 2^16 ran out seconds into the world (every world run on an RX 9070 XT,
/// each a device removal shortly after; #59).
const DYNAMIC: u32 = 1 << 18;

/// The dynamic slots of a heap whose host would not take [`DYNAMIC`] more
/// (a capped heap, NVIDIA): the title's slots keep their room first.
const MIN_DYNAMIC: u32 = 1 << 16;

/// Counts a [`HostHeap::mirror_synced_batch`] into `crate::descriptor::SYNC_STATS` when dropped.
struct MakeTimer(std::time::Instant, u64);

impl Drop for MakeTimer {
    fn drop(&mut self) {
        use crate::descriptor::{SyncStat, sync_stat};
        sync_stat(SyncStat::MakeUs, self.0.elapsed().as_micros() as u64);
        sync_stat(SyncStat::Made, self.1);
    }
}

impl HostHeap {
    pub(crate) fn new(sampler: bool, count: u32) -> Option<Self> {
        let host = host()?;
        let kind = if sampler {
            D3D12_DESCRIPTOR_HEAP_TYPE_SAMPLER
        } else {
            D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV
        };
        let make = |count: u32| -> windows::core::Result<ID3D12DescriptorHeap> {
            // SAFETY: plain heap creation.
            unsafe {
                host.device
                    .CreateDescriptorHeap(&D3D12_DESCRIPTOR_HEAP_DESC {
                        Type: kind,
                        NumDescriptors: count,
                        Flags: D3D12_DESCRIPTOR_HEAP_FLAG_SHADER_VISIBLE,
                        NodeMask: 0,
                    })
            }
        };
        // Every slot a shader can index (title slots, also past a capped
        // heap, and dynamic ones): shaders read the table as a root view,
        // which has no bounds, at indices clamped below `HEAP_VIEWS`.
        let strides = (!sampler)
            .then(|| host.upload_buffer(u64::from(count.max(victory_shader::emit::HEAP_VIEWS)) * 4))
            .flatten()
            .map(|(buffer, data)| (buffer, data as usize));
        let marks = (!sampler)
            .then(|| {
                let bytes = u64::from(count.max(victory_shader::emit::HEAP_VIEWS)) * 8;
                let (buffer, data) = host.upload_buffer(bytes)?;
                crate::host::name(&buffer, || "view marks".into());
                // Every slot unmarked (all ones) until a view is made there.
                // SAFETY: our mapped buffer of `bytes` bytes.
                unsafe { std::ptr::write_bytes(data, 0xFF, bytes as usize) };
                Some((buffer, data as usize))
            })
            .flatten();
        // Other threads' submissions held off from here until the heap is
        // filled (`Host::quiet`; a title heap's `host` holds them already).
        let _quiet = (!sampler).then(|| host.quiet()).flatten();
        // The larger dynamic range only where the host takes it on top of
        // every title slot; else the smaller one, and the cap after that.
        // Translated shaders clamp their indices below `HEAP_VIEWS` or
        // `HEAP_SAMPLERS`: slots up to there exist and stay null.
        let (views, samplers) = (victory_shader::emit::HEAP_VIEWS, victory_shader::emit::HEAP_SAMPLERS);
        let (heap, count, extra) = if sampler {
            match make(count.max(samplers)) {
                Ok(heap) => (heap, count, 0),
                Err(e) => {
                    tracing::warn!(error = %e, count, sampler, "no host descriptor heap");
                    return None;
                }
            }
        } else if let Ok(heap) = make((count + DYNAMIC).max(views)) {
            (heap, count, DYNAMIC)
        } else {
            // shortcut: a host that caps its heaps below `HEAP_VIEWS` (NVIDIA)
            // faults on an index past them, as before the clamp.
            match make(count + MIN_DYNAMIC) {
                Ok(heap) => (heap, count, MIN_DYNAMIC),
                Err(e) if count + MIN_DYNAMIC > PORTABLE_VIEWS => {
                    // shortcut: descriptors past the host's limit are left
                    // out; a title that reaches them needs remapping.
                    tracing::warn!(error = %e, count, "host view heap capped at {PORTABLE_VIEWS}");
                    (
                        make(PORTABLE_VIEWS).ok()?,
                        PORTABLE_VIEWS - MIN_DYNAMIC,
                        MIN_DYNAMIC,
                    )
                }
                Err(e) => {
                    tracing::warn!(error = %e, count, sampler, "no host descriptor heap");
                    return None;
                }
            }
        };
        // SAFETY: a live heap.
        let (cpu, step) = unsafe {
            (
                heap.GetCPUDescriptorHandleForHeapStart().ptr,
                host.device.GetDescriptorHandleIncrementSize(kind),
            )
        };
        tracing::debug!(count, sampler, "host descriptor heap");
        // SAFETY: a live shader-visible heap.
        let (gpu, total) = unsafe {
            (heap.GetGPUDescriptorHandleForHeapStart().ptr, heap.GetDesc().NumDescriptors)
        };
        GPU_RANGES.lock().unwrap_or_else(|e| e.into_inner()).push((
            gpu,
            u64::from(total) * u64::from(step),
            u64::from(step),
            sampler,
        ));
        let mut h = HostHeap {
            id: NEXT_ID.fetch_add(1, Ordering::Relaxed),
            heap,
            cpu,
            step,
            count,
            dynamic: extra,
            sampler,
            made: Mutex::new(vec![0; count as usize]),
            synced: (0..count).map(|_| AtomicU64::new(0)).collect(),
            made_nonzero: (0..(count as usize).div_ceil(64))
                .map(|_| AtomicU64::new(0))
                .collect(),
            block_forgot: (0..(count as usize).div_ceil(SYNC_BLOCK_SLOTS as usize))
                .map(|_| AtomicU64::new(0))
                .collect(),
            ready: (0..count).map(|_| AtomicU64::new(0)).collect(),
            staging: std::sync::OnceLock::new(),
            kept: Mutex::new(Paged::new((count + extra) as usize)),
            nulled_unordered: Mutex::new(Vec::new()),
            next: Mutex::new((0, vec![0; (extra as usize).div_ceil(64)])),
            pairs: Mutex::new((count + extra, Vec::new())),
            null_root_srv: None,
            pairs_end: match victory_shader::emit::NULL_BUFFER_VIEW {
                null if !sampler && null > count + extra && null < total => null - 1,
                _ => total,
            },
            strides,
            marks,
            addresses: std::sync::OnceLock::new(),
            viewed: Default::default(),
            widened: Default::default(),
            uploads: Default::default(),
            held: Default::default(),
        };
        if !sampler {
            crate::bindless::register(&h.viewed);
        }
        // A slot never written is undefined: shaders reading one (a slot
        // the title wrote in a way we do not know, or a table entry no
        // used-resource entry names) read a wild address. Every slot starts
        // null instead.
        let started = std::time::Instant::now();
        if let Err(e) = h.fill_null(kind, total) {
            tracing::warn!(error = %e, "host descriptor heap not made null");
        }
        tracing::debug!(ms = started.elapsed().as_millis(), "host descriptor heap made null");
        // What shaders read instead of a slot without a buffer view where
        // they take one at a computed index (`victory_shader::emit`, "Buffer
        // views at computed heap indices"): a null raw SRV, and UAV below it.
        if !sampler {
            let fixed = victory_shader::emit::NULL_BUFFER_VIEW;
            let null = if fixed > count + extra && fixed < total {
                Some(fixed)
            } else {
                tracing::warn!(count, total, "no room for the null buffer views: computed indices of slots without a buffer view read what is there");
                // Root descriptors receive a heap index at bind time and can
                // use a pair inside a capped heap. Keep it for the heap's life.
                h.pair().map(|first| first + 1)
            };
            if let Some(srv) = null {
                h.raw_view_at(srv, None, 0, 0, false);
                h.raw_view_at(srv - 1, None, 0, 0, true);
                h.null_root_srv = Some(srv);
            }
        }
        Some(h)
    }

    pub(crate) fn null_root_view(&self, uav: bool) -> Option<u32> {
        self.null_root_srv.map(|srv| srv - u32::from(uav))
    }

    /// Makes the first `total` slots null: one null slot in a CPU-side heap,
    /// doubled by copies to a block, the block copied over the heap.
    fn fill_null(&self, kind: D3D12_DESCRIPTOR_HEAP_TYPE, total: u32) -> windows::core::Result<()> {
        const BLOCK: u32 = 1 << 16;
        let Some(host) = host() else { return Ok(()) };
        // SAFETY: plain heap creation and copies inside live heaps.
        unsafe {
            let block: ID3D12DescriptorHeap =
                host.device
                    .CreateDescriptorHeap(&D3D12_DESCRIPTOR_HEAP_DESC {
                        Type: kind,
                        NumDescriptors: BLOCK,
                        ..Default::default()
                    })?;
            let start = block.GetCPUDescriptorHandleForHeapStart();
            let at = |i: u32| D3D12_CPU_DESCRIPTOR_HANDLE {
                ptr: start.ptr + (i * self.step) as usize,
            };
            if self.sampler {
                null_sampler(start);
            } else {
                null_view(start);
            }
            let mut filled = 1;
            while filled < BLOCK {
                let n = filled.min(BLOCK - filled);
                host.device
                    .CopyDescriptorsSimple(n, at(filled), start, kind);
                filled += n;
            }
            let mut done = 0;
            while done < total {
                let n = BLOCK.min(total - done);
                host.device
                    .CopyDescriptorsSimple(n, self.cpu(done), start, kind);
                done += n;
            }
        }
        Ok(())
    }

    /// A plain point sampler at slot `index`, for sampler slots without a
    /// view.
    fn null_sampler(&self, index: u32) {
        null_sampler(self.cpu(index));
    }
}

/// A plain point sampler at `at`.
fn null_sampler(at: D3D12_CPU_DESCRIPTOR_HANDLE) {
    let Some(host) = host() else { return };
    let desc = D3D12_SAMPLER_DESC {
        Filter: D3D12_FILTER_MIN_MAG_MIP_POINT,
        AddressU: D3D12_TEXTURE_ADDRESS_MODE_CLAMP,
        AddressV: D3D12_TEXTURE_ADDRESS_MODE_CLAMP,
        AddressW: D3D12_TEXTURE_ADDRESS_MODE_CLAMP,
        MaxAnisotropy: 1,
        ComparisonFunc: D3D12_COMPARISON_FUNC_NEVER,
        ..Default::default()
    };
    // SAFETY: a slot of a sampler heap of ours.
    unsafe { host.device.CreateSampler(&desc, at) };
}

/// A sampler heap of our own, for the lists that set a root signature whose
/// sampler heap is indexed directly (`SAMPLER_HEAP_DIRECTLY_INDEXED`) while
/// the title binds no sampler heap: D3D12 requires one to be bound first
/// (the debug layer's id 1419), and breaking that rule removes the device on
/// NVIDIA. Slot 0 holds a point sampler, so an indexed read of it returns
/// something defined rather than whatever the memory held.
pub(crate) fn fallback_samplers() -> Option<&'static HostHeap> {
    static HEAP: std::sync::OnceLock<Option<HostHeap>> = std::sync::OnceLock::new();
    HEAP.get_or_init(|| {
        let heap = HostHeap::new(true, 1)?;
        null_sampler(heap.cpu(0));
        Some(heap)
    })
    .as_ref()
}

/// The first of `n` clear bits in a row within `[from, to)` of `taken`
/// (a set of slots, a bit each), if there are any. A word's bits are
/// looked at together.
fn free_run(taken: &[u64], from: usize, to: usize, n: usize) -> Option<usize> {
    let (mut at, mut run) = (from, 0);
    while at < to {
        let room = (64 - at % 64).min(to - at);
        let window = u64::MAX >> (64 - room);
        // Bit p: slot `at + p` is clear.
        let free = !(taken[at / 64] >> (at % 64)) & window;
        // The clear slots from `at` on, which go on a run begun before `at`.
        let lead = ((!free).trailing_zeros() as usize).min(room);
        if run + lead >= n {
            return Some(at - run);
        }
        if n <= 64 {
            let fits = starts(free, n);
            if fits != 0 {
                return Some(at + fits.trailing_zeros() as usize);
            }
        }
        if lead == room {
            run += room;
        } else {
            run = (free << (64 - room)).leading_ones() as usize;
        }
        at += room;
    }
    None
}

/// Bit p set when bits p to p + n - 1 of `free` are all set (n <= 64):
/// runs of 1, 2, 4... bits are joined until they make n.
fn starts(free: u64, n: usize) -> u64 {
    let (mut acc, mut len, mut cur, mut pow, mut rem) = (u64::MAX, 0usize, free, 1usize, n);
    while rem > 0 {
        if rem & 1 == 1 {
            acc = if len == 0 { cur } else { acc & (cur >> len) };
            len += pow;
        }
        rem >>= 1;
        if rem > 0 {
            cur &= cur >> pow;
            pow *= 2;
        }
    }
    acc
}

/// Sets (`on`) or clears the bits of `taken` from `from` to `to`.
fn set_taken(taken: &mut [u64], from: usize, to: usize, on: bool) {
    let mut at = from;
    while at < to {
        let (bit, room) = (at % 64, (64 - at % 64).min(to - at));
        let mask = (u64::MAX >> (64 - room)) << bit;
        if on {
            taken[at / 64] |= mask;
        } else {
            taken[at / 64] &= !mask;
        }
        at += room;
    }
}

impl HostHeap {
    /// `n` consecutive free dynamic slots (view heaps only), taken until
    /// [`HostHeap::release`]: the first's index.
    pub(crate) fn dynamic(&self, n: u32) -> Option<u32> {
        if self.sampler || n == 0 || n > self.dynamic {
            return None;
        }
        let mut guard = self.next.lock().unwrap_or_else(|e| e.into_inner());
        let (next, taken) = &mut *guard;
        let n = n as usize;
        // Search from `next` on, then from the start (extending past cursor by
        // n - 1 so contiguous runs crossing the cursor are not missed).
        let cursor = *next as usize;
        let end = self.dynamic as usize;
        let second_end = cursor.saturating_add(n.saturating_sub(1)).min(end);
        let Some(first) =
            free_run(taken, cursor, end, n).or_else(|| free_run(taken, 0, second_end, n))
        else {
            static ONCE: std::sync::Once = std::sync::Once::new();
            ONCE.call_once(|| {
                tracing::warn!(n, "no free dynamic descriptor slots: a table is not made")
            });
            return None;
        };
        set_taken(taken, first, first + n, true);
        *next = ((first + n) % self.dynamic as usize) as u32;
        Some(self.count + first as u32)
    }

    /// Two consecutive slots (view heaps only) kept until
    /// [`HostHeap::free_pair`]: past the dynamic slots where the heap has
    /// room (a host copy's views, `crate::memory`, for as long as it
    /// lives), else dynamic ones.
    pub(crate) fn pair(&self) -> Option<u32> {
        if self.sampler {
            return None;
        }
        {
            let mut pairs = self.pairs.lock().unwrap_or_else(|e| e.into_inner());
            if let Some(index) = pairs.1.pop() {
                return Some(index);
            }
            if pairs.0 + 2 <= self.pairs_end {
                pairs.0 += 2;
                return Some(pairs.0 - 2);
            }
        }
        self.dynamic(2)
    }

    /// Gives back a pair [`HostHeap::pair`] returned.
    pub(crate) fn free_pair(&self, index: u32) {
        if index >= self.count + self.dynamic {
            self.pairs.lock().unwrap_or_else(|e| e.into_inner()).1.push(index);
        } else {
            self.release(self.id, index, 2);
        }
    }

    /// The title slots this heap has host slots for; fewer than the title's
    /// when the host caps its size.
    pub(crate) fn count(&self) -> u32 {
        self.count
    }

    /// This heap's unique identity ([`NEXT_ID`]): dynamic slots are returned
    /// only when the caller's identity matches this heap.
    pub(crate) fn id(&self) -> u64 {
        self.id
    }

    /// Gives back `n` dynamic slots from `first` (an index [`dynamic`]
    /// returned), only if `id` matches this heap's [`Self::id`].
    ///
    /// [`dynamic`]: HostHeap::dynamic
    pub(crate) fn release(&self, id: u64, first: u32, n: u32) {
        if id != self.id {
            return;
        }
        let Some(at) = first.checked_sub(self.count) else {
            return;
        };
        let mut guard = self.next.lock().unwrap_or_else(|e| e.into_inner());
        let end = (at + n).min(self.dynamic) as usize;
        set_taken(&mut guard.1, at as usize, end, false);
    }

    /// Makes slot `index` (a dynamic one) from `view`; a null view if the
    /// host cannot.
    pub(crate) fn make_at(&self, index: u32, view: &View) {
        let view_is_sampler = is_sampler(view);
        if view_is_sampler != self.sampler {
            warn_heap_mismatch(index, self.sampler, view_is_sampler);
            self.make_null(index);
            return;
        }
        let (kept, mark) = make_marked(view, self.cpu(index));
        if kept.is_none() {
            null_for(view, self.cpu(index));
        }
        self.keep(index, kept.flatten());
        self.set_stride(index, Some(view));
        self.set_mark(index, mark);
        // A table made afresh: shaders reading a view's address there (e.g.
        // the BVH builder's base) index the view addresses at this slot.
        self.note_address(index, Some(view));
    }

    /// Makes slot `index` (a dynamic one) a raw view, shader resource or
    /// unordered access (`uav`), of `bytes` of `buffer` from byte `first`
    /// (both multiples of 4), which it holds; a null one for no buffer, or
    /// for unordered access to a buffer shaders cannot write.
    pub(crate) fn raw_view_at(&self, index: u32, buffer: Option<&ID3D12Resource>, first: u64, bytes: u64, uav: bool) {
        let Some(host) = host() else { return };
        let writable = |b: &ID3D12Resource| {
            // SAFETY: a live buffer.
            unsafe { b.GetDesc() }.Flags.0 & D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS.0 != 0
        };
        let buffer = buffer.filter(|&b| !uav || writable(b));
        // shortcut: a buffer view holds at most 2^27 elements (512 MiB raw).
        let (first, elements) = match buffer {
            Some(_) => (first / 4, (bytes / 4).clamp(1, 1 << 27) as u32),
            // A null view still names its size.
            None => (0, 1),
        };
        // SAFETY: a slot of our heap; the buffer is live (or none: a null view).
        unsafe {
            if uav {
                let desc = D3D12_UNORDERED_ACCESS_VIEW_DESC {
                    Format: DXGI_FORMAT_R32_TYPELESS,
                    ViewDimension: D3D12_UAV_DIMENSION_BUFFER,
                    Anonymous: D3D12_UNORDERED_ACCESS_VIEW_DESC_0 {
                        Buffer: D3D12_BUFFER_UAV {
                            FirstElement: first,
                            NumElements: elements,
                            Flags: D3D12_BUFFER_UAV_FLAG_RAW,
                            ..Default::default()
                        },
                    },
                };
                host.device
                    .CreateUnorderedAccessView(buffer, None, Some(&desc), self.cpu(index));
            } else {
                let desc = D3D12_SHADER_RESOURCE_VIEW_DESC {
                    Format: DXGI_FORMAT_R32_TYPELESS,
                    ViewDimension: D3D12_SRV_DIMENSION_BUFFER,
                    Shader4ComponentMapping: D3D12_DEFAULT_SHADER_4_COMPONENT_MAPPING,
                    Anonymous: D3D12_SHADER_RESOURCE_VIEW_DESC_0 {
                        Buffer: D3D12_BUFFER_SRV {
                            FirstElement: first,
                            NumElements: elements,
                            StructureByteStride: 0,
                            Flags: D3D12_BUFFER_SRV_FLAG_RAW,
                        },
                    },
                };
                host.device
                    .CreateShaderResourceView(buffer, Some(&desc), self.cpu(index));
            }
        }
        self.keep(index, buffer.map(|b| (b.clone(), None)));
        self.write_stride(index, 4);
        self.set_mark(index, None);
        // Not a title view: no entry an earlier table left there stays.
        self.note_address(index, None);
    }

    /// Makes slot `index` read nothing: a null view, or a plain sampler.
    pub(crate) fn make_null(&self, index: u32) {
        if self.sampler {
            self.null_sampler(index);
        } else {
            null_view(self.cpu(index));
        }
        self.keep(index, None);
        self.set_stride(index, None);
        self.set_mark(index, None);
        self.note_address(index, None);
        if index < self.count {
            self.viewed.set(index, None, false);
        }
    }

    /// Enters the byte stride of `view` (0 for none or not a buffer view)
    /// for slot `index` in the view-stride table, with the integer-format
    /// bits of a typed buffer view ([`view_kind`]).
    fn set_stride(&self, index: u32, view: Option<&View>) {
        self.write_stride(index, view.map_or(0, |v| view_stride(v) | view_kind(v)));
    }

    fn write_stride(&self, index: u32, stride: u32) {
        let Some((_, data)) = &self.strides else { return };
        if index >= self.count.max(victory_shader::emit::HEAP_VIEWS) {
            return;
        }
        // SAFETY: a dword of the mapped table, which has one per slot below
        // the larger of the title's count and `HEAP_VIEWS`.
        unsafe { (*data as *mut u32).add(index as usize).write_volatile(stride) };
    }

    /// The view-stride table, for root SRV `VIEW_STRIDES`.
    pub(crate) fn strides(&self) -> Option<&ID3D12Resource> {
        self.strides.as_ref().map(|(buffer, _)| buffer)
    }

    /// Enters slot `index`'s view-mark entry: the mark bit and phase of the
    /// unordered buffer view there ([`crate::shadow::Shadow::mark_entry`]),
    /// none for anything else.
    pub(crate) fn set_mark(&self, index: u32, entry: Option<(u32, u32)>) {
        let Some((_, data)) = &self.marks else { return };
        if index >= self.count.max(victory_shader::emit::HEAP_VIEWS) {
            return;
        }
        let entry = entry.unwrap_or((victory_shader::emit::VIEW_MARK_NONE, 0));
        // SAFETY: two dwords of the mapped table, which has two per slot
        // below the larger of the title's count and `HEAP_VIEWS`.
        unsafe {
            let at = (*data as *mut u32).add(2 * index as usize);
            at.write_volatile(entry.0);
            at.add(1).write_volatile(entry.1);
        }
    }

    /// The view-mark table, for the root SRV of shaders marking their writes.
    pub(crate) fn marks(&self) -> Option<&ID3D12Resource> {
        self.marks.as_ref().map(|(buffer, _)| buffer)
    }

    /// Slot `index` now views `resource` (or nothing of the host's).
    fn keep(&self, index: u32, resource: Kept) {
        let mut kept = self.kept.lock().unwrap_or_else(|e| e.into_inner());
        // A slot that held nothing and gets nothing is left alone (no page).
        if let Some(slot) = kept.slot(index as usize, resource.is_some()) {
            let old_widened = slot.as_ref().and_then(|(_, source)| match source {
                Some(HeldSource::Widened(view, _)) => Some(view.clone()),
                _ => None,
            });
            let new_widened = resource.as_ref().and_then(|(_, source)| match source {
                Some(HeldSource::Widened(view, _)) => Some(view.clone()),
                _ => None,
            });
            let old = std::mem::replace(slot, resource);
            self.note_widened(old_widened.as_ref(), false);
            self.note_widened(new_widened.as_ref(), true);
            #[cfg(feature = "diagnostics")]
            if let Some((old, _)) = &old
                && !self.sampler
                && slot.as_ref().map(|s| windows::core::Interface::as_raw(&s.0)) != Some(windows::core::Interface::as_raw(old))
                // SAFETY: a live resource.
                && unsafe { old.GetDesc() }.Dimension != D3D12_RESOURCE_DIMENSION_BUFFER
            {
                use windows::core::Interface;
                let left = crate::resource::life::holders(old);
                crate::resource::life::note(old.as_raw() as usize, crate::resource::life::SLOT, u64::from(index), left);
            }
            // Title threads rewrite slots while the GPU runs a submission;
            // one of its shaders may hold the old descriptor (or read the
            // slot at an index from wrong data): what it views is freed
            // only once that work has run.
            if let Some((old, _)) = old {
                crate::host::retire(old);
            }
        }
    }

    /// Batched [`HostHeap::keep`]: applies all slot updates under one hold of
    /// `self.kept` and pushes displaced host resources into `retired` for the
    /// caller to retire outside locks.
    fn keep_batch(&self, updates: Vec<(u32, Kept)>, retired: &mut Vec<ID3D12Resource>) {
        if updates.is_empty() {
            return;
        }
        let mut widened_changes: Vec<(std::sync::Arc<crate::shadow::Widened>, bool)> = Vec::new();
        {
            let mut kept = self.kept.lock().unwrap_or_else(|e| e.into_inner());
            for (index, resource) in updates {
                let Some(slot) = kept.slot(index as usize, resource.is_some()) else {
                    continue;
                };
                if let Some((_, Some(HeldSource::Widened(view, _)))) = slot.as_ref() {
                    widened_changes.push((view.clone(), false));
                }
                if let Some((_, Some(HeldSource::Widened(view, _)))) = resource.as_ref() {
                    widened_changes.push((view.clone(), true));
                }
                let same_resource = match (slot.as_ref(), resource.as_ref()) {
                    (Some((cur, _)), Some((new, _))) => {
                        windows::core::Interface::as_raw(cur)
                            == windows::core::Interface::as_raw(new)
                    }
                    _ => false,
                };
                if same_resource {
                    if let (Some(cur), Some((_, new_src))) = (slot.as_mut(), resource) {
                        cur.1 = new_src;
                    }
                    continue;
                }
                let old = std::mem::replace(slot, resource);
                #[cfg(feature = "diagnostics")]
                if let Some((old, _)) = &old
                    && !self.sampler
                    && slot.as_ref().map(|s| windows::core::Interface::as_raw(&s.0))
                        != Some(windows::core::Interface::as_raw(old))
                    // SAFETY: a live resource.
                    && unsafe { old.GetDesc() }.Dimension != D3D12_RESOURCE_DIMENSION_BUFFER
                {
                    use windows::core::Interface;
                    let left = crate::resource::life::holders(old);
                    crate::resource::life::note(
                        old.as_raw() as usize,
                        crate::resource::life::SLOT,
                        u64::from(index),
                        left,
                    );
                }
                if let Some((old, _)) = old {
                    retired.push(old);
                }
            }
        }
        if !widened_changes.is_empty() {
            let mut widened = self.widened.lock().unwrap_or_else(|e| e.into_inner());
            for (view, add) in widened_changes {
                let key = std::sync::Arc::as_ptr(&view) as usize;
                if add {
                    let entry = widened
                        .entry(key)
                        .or_insert_with(|| (0, std::sync::Arc::downgrade(&view)));
                    entry.0 += 1;
                } else if let Some((count, _)) = widened.get_mut(&key) {
                    *count = count.saturating_sub(1);
                    if *count == 0 {
                        widened.remove(&key);
                    }
                }
            }
        }
    }

    fn note_widened(&self, view: Option<&std::sync::Arc<crate::shadow::Widened>>, add: bool) {
        let Some(view) = view else { return };
        let key = std::sync::Arc::as_ptr(view) as usize;
        let mut widened = self.widened.lock().unwrap_or_else(|e| e.into_inner());
        if add {
            let entry = widened.entry(key).or_insert_with(|| (0, std::sync::Arc::downgrade(view)));
            entry.0 += 1;
        } else if let Some((count, _)) = widened.get_mut(&key) {
            *count = count.saturating_sub(1);
            if *count == 0 {
                widened.remove(&key);
            }
        }
    }

    /// Widened SRV buffers a heap-indexing shader may read.
    pub(crate) fn widened_buffers(&self) -> Vec<ID3D12Resource> {
        self.widened
            .lock()
            .unwrap_or_else(|e| e.into_inner())
            .values()
            .filter_map(|(_, view)| view.upgrade().map(|view| view.buffer.clone()))
            .collect()
    }

    /// The host address of the slots' view addresses (see the field), made
    /// on first use from the slots made so far.
    pub(crate) fn view_addresses(&self) -> Option<u64> {
        let made = self.addresses.get_or_init(|| {
            // The view is a root view, which has no bounds: an entry for
            // every slot a table can lie in, the dynamic ones too (a table
            // made afresh on a host whose heap is smaller than the title's
            // read past the title slots' entries: whatever followed the
            // buffer, or a GPU fault).
            let (buffer, data) = host()?.upload_buffer(u64::from(self.address_slots()) * 16)?;
            crate::host::name(&buffer, || "view addresses".into());
            let data = data as usize;
            let made = self.made.lock().unwrap_or_else(|e| e.into_inner());
            for (i, &key) in made.iter().enumerate() {
                let view = (key != 0).then(|| view_at(key as u32, (key >> 32) as u32)).flatten();
                write_address(data, i, view.as_deref());
            }
            // Dynamic slots are entered as they are made ([`Self::make_at`],
            // [`Self::make_null`], raw views).
            for i in self.count..self.address_slots() {
                write_address(data, i as usize, None);
            }
            // SAFETY: a live buffer.
            let gpu = unsafe { buffer.GetGPUVirtualAddress() };
            Some((buffer, data, gpu))
        });
        made.as_ref().map(|m| m.2)
    }

    /// The slots the view addresses have an entry for: the title's and the
    /// dynamic ones.
    fn address_slots(&self) -> u32 {
        self.count + self.dynamic
    }

    /// Slot `index` now holds `view` (or no view): its view address.
    fn note_address(&self, index: u32, view: Option<&View>) {
        if index < self.address_slots()
            && let Some(Some((_, data, _))) = self.addresses.get()
        {
            write_address(*data, index as usize, view);
        }
    }

    /// The view address entered for slot `index`; none for a slot past the
    /// view addresses' buffer (tests).
    #[cfg(test)]
    fn address_at(&self, index: u32) -> Option<u64> {
        let (buffer, data, _) = self.addresses.get()?.as_ref()?;
        // SAFETY: a live buffer.
        if 16 * u64::from(index) + 16 > unsafe { buffer.GetDesc() }.Width {
            return None;
        }
        // SAFETY: an entry of the mapped view addresses (checked).
        let [lo, hi] = unsafe { ((data + 16 * index as usize) as *const [u32; 2]).read_unaligned() };
        Some(u64::from(hi) << 32 | u64::from(lo))
    }

    /// Makes slots `index` and `index + 1` (dynamic ones) a raw shader
    /// resource view and a raw unordered access view of the first `bytes`
    /// (a multiple of 4) of `buffer`, which they hold; null views for no
    /// buffer.
    pub(crate) fn raw_pair_at(&self, index: u32, buffer: Option<&ID3D12Resource>, first: u64, bytes: u64) {
        let Some(host) = host() else { return };
        // A buffer view holds at most 2^27 elements (512 MiB raw): the page
        // table names a longer copy in pieces (`crate::memory::pieces`).
        let elements = (bytes / 4).min(1 << 27) as u32;
        let srv = D3D12_SHADER_RESOURCE_VIEW_DESC {
            Format: DXGI_FORMAT_R32_TYPELESS,
            ViewDimension: D3D12_SRV_DIMENSION_BUFFER,
            Shader4ComponentMapping: D3D12_DEFAULT_SHADER_4_COMPONENT_MAPPING,
            Anonymous: D3D12_SHADER_RESOURCE_VIEW_DESC_0 {
                Buffer: D3D12_BUFFER_SRV {
                    FirstElement: first / 4,
                    NumElements: elements,
                    StructureByteStride: 0,
                    Flags: D3D12_BUFFER_SRV_FLAG_RAW,
                },
            },
        };
        let uav = D3D12_UNORDERED_ACCESS_VIEW_DESC {
            Format: DXGI_FORMAT_R32_TYPELESS,
            ViewDimension: D3D12_UAV_DIMENSION_BUFFER,
            Anonymous: D3D12_UNORDERED_ACCESS_VIEW_DESC_0 {
                Buffer: D3D12_BUFFER_UAV {
                    FirstElement: first / 4,
                    NumElements: elements,
                    Flags: D3D12_BUFFER_UAV_FLAG_RAW,
                    ..Default::default()
                },
            },
        };
        // SAFETY: slots of our heap; the buffer is live (or none: null views).
        unsafe {
            host.device
                .CreateShaderResourceView(buffer, Some(&srv), self.cpu(index));
            host.device
                .CreateUnorderedAccessView(buffer, None, Some(&uav), self.cpu(index + 1));
        }
        self.keep(index, buffer.map(|b| (b.clone(), None)));
        self.keep(index + 1, buffer.map(|b| (b.clone(), None)));
        self.write_stride(index, 4);
        self.write_stride(index + 1, 4);
        // Writes through these are title memory's, marked in its page table.
        self.set_mark(index, None);
        self.set_mark(index + 1, None);
        // Not a title view: no entry an earlier table left there stays.
        self.note_address(index, None);
        self.note_address(index + 1, None);
    }

    /// The shader-visible handle of slot `index`.
    pub(crate) fn gpu(&self, index: u32) -> D3D12_GPU_DESCRIPTOR_HANDLE {
        // SAFETY: a live shader-visible heap.
        let start = unsafe { self.heap.GetGPUDescriptorHandleForHeapStart() };
        D3D12_GPU_DESCRIPTOR_HANDLE {
            ptr: start.ptr + u64::from(index) * u64::from(self.step),
        }
    }

    fn cpu(&self, index: u32) -> D3D12_CPU_DESCRIPTOR_HANDLE {
        D3D12_CPU_DESCRIPTOR_HANDLE {
            ptr: self.cpu + index as usize * self.step as usize,
        }
    }

    fn set_made_nonzero(&self, index: u32, nonzero: bool) {
        let word = &self.made_nonzero[(index / 64) as usize];
        let mask = 1u64 << (index % 64);
        if nonzero {
            word.fetch_or(mask, Ordering::Release);
        } else {
            word.fetch_and(!mask, Ordering::Release);
        }
    }

    fn made_is_nonzero(&self, index: u32) -> bool {
        let word = self.made_nonzero[(index / 64) as usize].load(Ordering::Acquire);
        (word >> (index % 64)) & 1 != 0
    }

    /// A heap sync of slots `first..last`, whose records are at `start` +
    /// index × `step`: [`HostHeap::mirror`] of each slot whose record
    /// differs from what the slot holds, skipping empty and already-verified
    /// slots locklessly when no views were forgotten.
    #[cfg(test)]
    pub(crate) fn sync_slots(&self, first: u32, last: u32, start: u64, step: u64) {
        self.sync_slots_bytes(first, last, start, step, 0, &[]);
    }

    /// Like [`HostHeap::sync_slots`], reading records that lie inside
    /// `buf` (`[buf_start, buf_start + buf.len())`) without re-reading title memory.
    pub(crate) fn sync_slots_bytes(
        &self,
        first: u32,
        last: u32,
        start: u64,
        step: u64,
        buf_start: u64,
        buf: &[u8],
    ) {
        self.sync_slots_bytes_with(first..last, start, step, buf_start, buf, fast_heap_sync());
    }

    fn staging(&self) -> Option<&StagingHeap> {
        self.staging
            .get_or_init(|| StagingHeap::new(self.sampler, self.step))
            .as_ref()
    }

    fn sync_slots_bytes_with(
        &self,
        slots: std::ops::Range<u32>,
        start: u64,
        step: u64,
        buf_start: u64,
        buf: &[u8],
        fast: bool,
    ) {
        let first = slots.start;
        let last = slots.end.min(self.count);
        if first >= last {
            return;
        }
        let at = |i: u32| start + u64::from(i) * step;
        let forgot = views_forgot();
        let b_first = first / SYNC_BLOCK_SLOTS;
        let b_last = (last - 1) / SYNC_BLOCK_SLOTS;
        let buf_end = buf_start.saturating_add(buf.len() as u64);
        let mut legacy_verify = false;

        let left = stale_left();
        if fast {
            let mut stale: Vec<(u32, [u32; 3])> = Vec::new();
            let all_in_buf =
                at(first) >= buf_start && at(last - 1).saturating_add(12) <= buf_end;
            let step_usize = step as usize;
            for b in b_first..=b_last {
                let block_verified =
                    self.block_forgot[b as usize].load(Ordering::Acquire) == forgot;
                let s_first = (b * SYNC_BLOCK_SLOTS).max(first);
                let s_last = ((b + 1) * SYNC_BLOCK_SLOTS).min(last);
                // Load `made_nonzero` once per 64-slot word so empty slots avoid
                // per-slot Acquire atomic loads.
                let mut i = s_first;
                while i < s_last {
                    let word_idx = (i / 64) as usize;
                    let word_end = ((word_idx as u32 + 1) * 64).min(s_last);
                    let nonzero_word = self.made_nonzero[word_idx].load(Ordering::Acquire);
                    if all_in_buf {
                        let mut ptr = buf
                            .as_ptr()
                            .wrapping_add((at(i) - buf_start) as usize);
                        while i < word_end {
                            // SAFETY: `all_in_buf` checks `[at(first), at(last - 1) + 12)`
                            // lies inside `[buf_start, buf_start + buf.len())`.
                            let record = unsafe { ptr.cast::<[u32; 3]>().read_unaligned() };
                            ptr = ptr.wrapping_add(step_usize);
                            let [tag, view, generation] = record;
                            if tag != VIEW_TAG || generation == 0 {
                                if (nonzero_word >> (i & 63)) & 1 != 0
                                    && !(left && tag != VIEW_TAG)
                                {
                                    stale.push((i, record));
                                }
                                i += 1;
                                continue;
                            }
                            let key = u64::from(generation) << 32 | u64::from(view);
                            let synced = self.synced[i as usize].load(Ordering::Acquire);
                            if synced == key {
                                if !block_verified && !view_gen_current(view, generation) {
                                    stale.push((i, record));
                                }
                                i += 1;
                                continue;
                            }
                            if view_gen_current(view, generation)
                                || synced != 0
                                || (nonzero_word >> (i & 63)) & 1 != 0
                            {
                                stale.push((i, record));
                            }
                            i += 1;
                        }
                    } else {
                        while i < word_end {
                            let addr = at(i);
                            let record =
                                if addr >= buf_start && addr.saturating_add(12) <= buf_end {
                                    let off = (addr - buf_start) as usize;
                                    // SAFETY: `off + 12 <= buf.len()`, unaligned copy of 12 bytes.
                                    unsafe {
                                        buf.as_ptr().add(off).cast::<[u32; 3]>().read_unaligned()
                                    }
                                } else if !buf.is_empty() {
                                    let mut raw = [0u8; 12];
                                    if !crate::host::read_into(addr as *const u8, &mut raw) {
                                        i += 1;
                                        continue;
                                    }
                                    [
                                        u32::from_le_bytes([raw[0], raw[1], raw[2], raw[3]]),
                                        u32::from_le_bytes([raw[4], raw[5], raw[6], raw[7]]),
                                        u32::from_le_bytes([raw[8], raw[9], raw[10], raw[11]]),
                                    ]
                                } else {
                                    // SAFETY: direct test path (`sync_slots`) in committed memory.
                                    unsafe { (addr as *const [u32; 3]).read_unaligned() }
                                };
                            let [tag, view, generation] = record;
                            if tag != VIEW_TAG || generation == 0 {
                                if (nonzero_word >> (i & 63)) & 1 != 0
                                    && !(left && tag != VIEW_TAG)
                                {
                                    stale.push((i, record));
                                }
                                i += 1;
                                continue;
                            }
                            let key = u64::from(generation) << 32 | u64::from(view);
                            let synced = self.synced[i as usize].load(Ordering::Acquire);
                            if synced == key {
                                if !block_verified && !view_gen_current(view, generation) {
                                    stale.push((i, record));
                                }
                                i += 1;
                                continue;
                            }
                            if view_gen_current(view, generation)
                                || synced != 0
                                || (nonzero_word >> (i & 63)) & 1 != 0
                            {
                                stale.push((i, record));
                            }
                            i += 1;
                        }
                    }
                }
            }
            if !stale.is_empty() {
                self.mirror_synced_batch(&stale, at);
            }
        } else {
            let legacy_fast = sync_fast_enabled();
            let blocks_verified = legacy_fast
                && (b_first..=b_last)
                    .all(|b| self.block_forgot[b as usize].load(Ordering::Acquire) == forgot);

            let mut stale: Vec<u32> = Vec::new();
            let mut candidates: Vec<(u32, [u32; 3])> = Vec::new();

            for i in first..last {
                let addr = at(i);
                let record = if addr >= buf_start && addr.saturating_add(12) <= buf_end {
                    let off = (addr - buf_start) as usize;
                    // SAFETY: `off + 12 <= buf.len()`, unaligned copy of 12 bytes.
                    unsafe { buf.as_ptr().add(off).cast::<[u32; 3]>().read_unaligned() }
                } else if !buf.is_empty() {
                    let mut raw = [0u8; 12];
                    if !crate::host::read_into(addr as *const u8, &mut raw) {
                        continue;
                    }
                    [
                        u32::from_le_bytes([raw[0], raw[1], raw[2], raw[3]]),
                        u32::from_le_bytes([raw[4], raw[5], raw[6], raw[7]]),
                        u32::from_le_bytes([raw[8], raw[9], raw[10], raw[11]]),
                    ]
                } else {
                    // SAFETY: direct test path (`sync_slots`) in committed memory.
                    unsafe { (addr as *const [u32; 3]).read_unaligned() }
                };
                let [tag, view, generation] = record;
                if tag != VIEW_TAG {
                    if !left && (!legacy_fast || self.made_is_nonzero(i)) {
                        stale.push(i);
                    }
                    continue;
                }
                let key = u64::from(generation) << 32 | u64::from(view);
                if blocks_verified && self.ready[i as usize].load(Ordering::Acquire) == key {
                    continue;
                }
                candidates.push((i, record));
            }

            if !candidates.is_empty() {
                let records: Vec<[u32; 3]> = candidates.iter().map(|&(_, r)| r).collect();
                let mut current = Vec::with_capacity(records.len());
                views_current_into(&records, &mut current);
                let made = self.made.lock().unwrap_or_else(|e| e.into_inner());
                for (&(i, [_, view, generation]), is_current) in candidates.iter().zip(current) {
                    let changed = match made.get(i as usize) {
                        Some(&m) if is_current => {
                            m != (u64::from(generation) << 32 | u64::from(view))
                        }
                        Some(&m) => m != 0,
                        None => true,
                    };
                    if changed {
                        stale.push(i);
                    }
                }
            }

            for i in stale {
                self.mirror(i, at(i), false);
            }
            legacy_verify = legacy_fast;
        }

        if fast || legacy_verify {
            for b in b_first..=b_last {
                let b_start = b * SYNC_BLOCK_SLOTS;
                let b_end = ((b + 1) * SYNC_BLOCK_SLOTS).min(self.count);
                if first <= b_start && last >= b_end {
                    self.block_forgot[b as usize].store(forgot, Ordering::Release);
                }
            }
        }
    }

    /// Mirrors a page's stale slots (`for_draw == false`) under one hold each
    /// of `VIEWS`, `self.made`, `self.kept`, `self.viewed`, and `shadow::READS`,
    /// deduplicating `prepare` and descriptor creation via [`StagingHeap`] +
    /// `CopyDescriptorsSimple`.
    fn mirror_synced_batch(&self, stale: &[(u32, [u32; 3])], at: impl Fn(u32) -> u64) {
        let _making = MakeTimer(std::time::Instant::now(), stale.len() as u64);
        struct CachedTemplate {
            src: D3D12_CPU_DESCRIPTOR_HANDLE,
            kept: Option<Kept>,
            mark: Option<(u32, u32)>,
        }

        let mut unique_keys: Vec<u64> = Vec::new();
        let mut key_counts: Vec<u32> = Vec::new();
        let mut key_to_idx: HashMap<u64, usize> = HashMap::new();
        let mut slot_key_idx: Vec<Option<usize>> = Vec::with_capacity(stale.len());

        for &(_, [tag, view, generation]) in stale {
            if tag == VIEW_TAG && generation != 0 {
                let key = u64::from(generation) << 32 | u64::from(view);
                let idx = match key_to_idx.get(&key) {
                    Some(&idx) => {
                        key_counts[idx] += 1;
                        idx
                    }
                    None => {
                        let idx = unique_keys.len();
                        unique_keys.push(key);
                        key_counts.push(1);
                        key_to_idx.insert(key, idx);
                        idx
                    }
                };
                slot_key_idx.push(Some(idx));
            } else {
                slot_key_idx.push(None);
            }
        }

        use crate::descriptor::{SyncStat as S, sync_stat};
        let lap = |stat: S, since: &mut std::time::Instant| {
            let now = std::time::Instant::now();
            sync_stat(stat, (now - *since).as_micros() as u64);
            *since = now;
        };
        let mut clock = std::time::Instant::now();
        let at_once = placed_at_once();
        let pins = views_at_batch(&unique_keys);
        let mut defer: Vec<bool> = Vec::with_capacity(unique_keys.len());
        let mut is_tex: Vec<bool> = Vec::with_capacity(unique_keys.len());
        let mut strides: Vec<u32> = Vec::with_capacity(unique_keys.len());
        for pin in &pins {
            if let Some(view) = pin.as_deref() {
                let later = defer_placed(view, at_once);
                if !later {
                    prepare(view);
                }
                defer.push(later);
                is_tex.push(is_texture(view));
                strides.push(view_stride(view) | view_kind(view));
            } else {
                defer.push(false);
                is_tex.push(false);
                strides.push(0);
            }
        }

        lap(S::PrepareUs, &mut clock);
        // Initialize the staging heap before acquiring `self.made`.
        let staging = self.staging();
        let host = host();

        let mut again: Vec<u32> = Vec::new();
        let mut retired: Vec<ID3D12Resource> = Vec::new();

        {
            let mut made = self.made.lock().unwrap_or_else(|e| e.into_inner());
            let mut templates: Vec<Option<CachedTemplate>> =
                (0..unique_keys.len()).map(|_| None).collect();
            let mut next_template = if self.sampler { 1 } else { STAGING_NULL_SLOTS };
            let mut kept_updates: Vec<(u32, Kept)> = Vec::with_capacity(stale.len());
            let mut copy_dsts: Vec<D3D12_CPU_DESCRIPTOR_HANDLE> = Vec::with_capacity(stale.len());
            let mut copy_srcs: Vec<D3D12_CPU_DESCRIPTOR_HANDLE> = Vec::with_capacity(stale.len());
            let mut viewed_updates: Vec<(u32, Option<&View>, bool)> = if self.sampler {
                Vec::new()
            } else {
                Vec::with_capacity(stale.len())
            };
            let mut forgot_reads: Vec<u32> = Vec::new();
            let mut finalize: Vec<(u32, u64, u64)> = Vec::with_capacity(stale.len());
            // What the slots took, for the frame report's line.
            let (mut unchanged, mut from_template, mut deferred, mut nulled) = (0u64, 0u64, 0u64, 0u64);
            let mut created: Vec<u64> = Vec::new();
            let mut create_us = 0u64;
            let mut make_marked = |view: &View, at: D3D12_CPU_DESCRIPTOR_HANDLE, key: u64| {
                let started = std::time::Instant::now();
                let made = make_marked(view, at);
                create_us += started.elapsed().as_micros() as u64;
                created.push(key);
                made
            };
            let addr_data = self
                .addresses
                .get()
                .and_then(Option::as_ref)
                .map(|&(_, data, _)| data);
            let addr_slots = self.address_slots();
            // The live records of every stale slot, read once under
            // `self.made` (one `ReadProcessMemory` for the span, a page's at
            // most): read a slot at a time it was a call per slot, ~0.5 us
            // each under Wine, most of the batch's time for ~39,000 slots.
            // A span that cannot be read whole is read a slot at a time.
            let span = stale
                .iter()
                .map(|&(i, _)| at(i))
                .fold((u64::MAX, 0u64), |(lo, hi), a| (lo.min(a), hi.max(a + 12)));
            let mut live_span: Vec<u8> = Vec::new();
            let span_read = !stale.is_empty() && span.1 - span.0 <= SPAN_READ_MAX && {
                live_span.resize((span.1 - span.0) as usize, 0);
                crate::host::read_into(span.0 as *const u8, &mut live_span)
            };
            for (entry_idx, &(index, snapshot)) in stale.iter().enumerate() {
                let addr = at(index);
                // Re-read live title memory under `self.made`: if a title
                // thread overwrote the slot since the page snapshot (including
                // during `prepare`), defer to `mirror` after dropping locks.
                // SAFETY: the caller holds the heap alive until the sync completes.
                let mut raw = [0u8; 12];
                if span_read {
                    let off = (addr - span.0) as usize;
                    raw.copy_from_slice(&live_span[off..off + 12]);
                } else if !crate::host::read_into(addr as *const u8, &mut raw) {
                    continue;
                }
                let live = [
                    u32::from_le_bytes([raw[0], raw[1], raw[2], raw[3]]),
                    u32::from_le_bytes([raw[4], raw[5], raw[6], raw[7]]),
                    u32::from_le_bytes([raw[8], raw[9], raw[10], raw[11]]),
                ];
                if live != snapshot {
                    again.push(index);
                    continue;
                }
                let resolved = slot_key_idx[entry_idx].and_then(|kidx| {
                    let [_, view_i, generation] = snapshot;
                    if !view_gen_current(view_i, generation) {
                        return None;
                    }
                    pins[kidx]
                        .as_deref()
                        .filter(|view| is_sampler(view) == self.sampler)
                        .map(|view| (kidx, unique_keys[kidx], view))
                });
                let Some((kidx, key, view)) = resolved else {
                    if made[index as usize] != 0 {
                        nulled += 1;
                        self.ready[index as usize].store(0, Ordering::Relaxed);
                        self.synced[index as usize].store(0, Ordering::Relaxed);
                        made[index as usize] = 0;
                        if let (Some(_), Some(st)) = (host, staging) {
                            copy_dsts.push(self.cpu(index));
                            copy_srcs.push(st.cpu(0));
                        } else if self.sampler {
                            self.null_sampler(index);
                        } else {
                            null_view(self.cpu(index));
                        }
                        self.write_stride(index, 0);
                        self.set_mark(index, None);
                        if let Some(data) = addr_data
                            && index < addr_slots
                        {
                            write_address(data, index as usize, None);
                        }
                        kept_updates.push((index, None));
                        if !self.sampler {
                            viewed_updates.push((index, None, false));
                        }
                        finalize.push((index, 0, 0));
                    }
                    continue;
                };
                if made[index as usize] == key {
                    unchanged += 1;
                    continue;
                }
                self.ready[index as usize].store(0, Ordering::Relaxed);
                self.synced[index as usize].store(0, Ordering::Relaxed);
                self.write_stride(index, strides[kidx]);
                if let Some(data) = addr_data
                    && index < addr_slots
                {
                    write_address(data, index as usize, Some(view));
                }
                let later = defer[kidx];
                let keep_old =
                    later && at_once == PlacedAtOnce::Keep && self.kept(index);
                if later {
                    forgot_reads.push(index);
                }
                made[index as usize] = key;

                let (kept, mark) = if later {
                    deferred += 1;
                    if !self.sampler && !keep_old {
                        if let (Some(_), Some(st)) = (host, staging) {
                            let null_slot = null_slot_index(false, Some(view));
                            copy_dsts.push(self.cpu(index));
                            copy_srcs.push(st.cpu(null_slot));
                        } else {
                            null_for(view, self.cpu(index));
                        }
                    }
                    (None, None)
                } else if key_counts[kidx] >= 2
                    && let (Some(_), Some(st)) = (host, staging)
                {
                    if let Some(tpl) = &templates[kidx] {
                        from_template += 1;
                        copy_dsts.push(self.cpu(index));
                        copy_srcs.push(tpl.src);
                        (tpl.kept.clone(), tpl.mark)
                    } else if next_template < st.count {
                        let t = next_template;
                        next_template += 1;
                        let (k, m) = make_marked(view, st.cpu(t), key);
                        let src = if k.is_none() && !self.sampler {
                            st.cpu(null_slot_index(false, Some(view)))
                        } else {
                            st.cpu(t)
                        };
                        copy_dsts.push(self.cpu(index));
                        copy_srcs.push(src);
                        templates[kidx] = Some(CachedTemplate {
                            src,
                            kept: k.clone(),
                            mark: m,
                        });
                        (k, m)
                    } else {
                        let (k, m) = make_marked(view, self.cpu(index), key);
                        if k.is_none() && !self.sampler {
                            let null_slot = null_slot_index(false, Some(view));
                            copy_dsts.push(self.cpu(index));
                            copy_srcs.push(st.cpu(null_slot));
                        }
                        (k, m)
                    }
                } else {
                    let (k, m) = make_marked(view, self.cpu(index), key);
                    if k.is_none() && !self.sampler {
                        if let (Some(_), Some(st)) = (host, staging) {
                            let null_slot = null_slot_index(false, Some(view));
                            copy_dsts.push(self.cpu(index));
                            copy_srcs.push(st.cpu(null_slot));
                        } else {
                            null_for(view, self.cpu(index));
                        }
                    }
                    (k, m)
                };

                if !keep_old {
                    self.set_mark(index, mark);
                }
                if !self.sampler {
                    viewed_updates.push((index, Some(view), kept.is_some()));
                }
                let kept = kept.flatten();
                let ready = if is_tex[kidx] && kept.is_none() {
                    0
                } else {
                    key
                };
                if !keep_old {
                    kept_updates.push((index, kept));
                }
                finalize.push((index, key, ready));
            }

            lap(S::LoopUs, &mut clock);
            {
                sync_stat(S::Unchanged, unchanged);
                sync_stat(S::Created, created.len() as u64);
                sync_stat(S::CreateUs, create_us);
                sync_stat(S::FromTemplate, from_template);
                sync_stat(S::Deferred, deferred);
                sync_stat(S::Nulled, nulled);
                if crate::descriptor::view_probe() {
                    sync_stat(S::SeenBefore, crate::descriptor::seen_before(&created));
                }
            }
            let copying = std::time::Instant::now();
            if !copy_dsts.is_empty()
                && let (Some(h), Some(st)) = (host, staging)
            {
                let count = copy_dsts.len() as u32;
                // SAFETY: `copy_dsts` and `copy_srcs` have length `count`; each
                // source handle lives in our CPU-only staging heap (already
                // populated above) and each dest handle in our shader-visible
                // heap; `None` range sizes specify 1 descriptor per range.
                unsafe {
                    h.device.CopyDescriptors(
                        count,
                        copy_dsts.as_ptr(),
                        None,
                        count,
                        copy_srcs.as_ptr(),
                        None,
                        st.kind,
                    );
                }
            }

            sync_stat(S::CopyUs, copying.elapsed().as_micros() as u64);
            clock = std::time::Instant::now();
            self.keep_batch(kept_updates, &mut retired);
            lap(S::KeepUs, &mut clock);
            if !self.sampler {
                self.viewed.set_batch(&viewed_updates);
            }
            lap(S::ViewedUs, &mut clock);
            crate::shadow::forget_reads(&forgot_reads);
            // Coalesce `made_nonzero` bit updates per 64-slot word (stale indices
            // arrive in ascending order): set nonzero bits before publishing
            // `synced`/`ready`, and clear nonzero bits after clearing them.
            let mut set_word = usize::MAX;
            let mut set_mask = 0u64;
            for &(index, new_synced, _) in &finalize {
                if new_synced == 0 {
                    continue;
                }
                let word = (index / 64) as usize;
                if word != set_word {
                    if set_word != usize::MAX && set_mask != 0 {
                        self.made_nonzero[set_word].fetch_or(set_mask, Ordering::Release);
                    }
                    set_word = word;
                    set_mask = 0;
                }
                set_mask |= 1u64 << (index & 63);
            }
            if set_word != usize::MAX && set_mask != 0 {
                self.made_nonzero[set_word].fetch_or(set_mask, Ordering::Release);
            }
            for &(index, new_synced, new_ready) in &finalize {
                self.synced[index as usize].store(new_synced, Ordering::Release);
                self.ready[index as usize].store(new_ready, Ordering::Release);
            }
            let mut clr_word = usize::MAX;
            let mut clr_mask = 0u64;
            for &(index, new_synced, _) in &finalize {
                if new_synced != 0 {
                    continue;
                }
                let word = (index / 64) as usize;
                if word != clr_word {
                    if clr_word != usize::MAX && clr_mask != 0 {
                        self.made_nonzero[clr_word].fetch_and(!clr_mask, Ordering::Release);
                    }
                    clr_word = word;
                    clr_mask = 0;
                }
                clr_mask |= 1u64 << (index & 63);
            }
            if clr_word != usize::MAX && clr_mask != 0 {
                self.made_nonzero[clr_word].fetch_and(!clr_mask, Ordering::Release);
            }
        }

        crate::host::retire_all(retired);
        drop(pins);
        for index in again {
            self.mirror(index, at(index), false);
        }
    }

    /// Whether the view made in slot `index` must be made again for a draw
    /// that reads it: a placed texture's view left null is made when a draw
    /// reads the slot through a table (heap syncs revisit it at every
    /// submission), a texture's unordered view left null when the texture was
    /// given up (`VICTORY_EVICT_HEAP_UAV`) is made again like a shader view (a
    /// table copying the null slot loses writes), and a later unordered or
    /// renderable view that upgraded the memory's `HostTexture` leaves the
    /// made view on the old one.
    fn needs_remake(&self, index: u32, view: &View, for_draw: bool) -> bool {
        if !for_draw || !(is_texture(view) || is_unordered_texture(view)) {
            return false;
        }
        if self.ready[index as usize].load(Ordering::Acquire) == 0 {
            return true;
        }
        let held = self
            .kept
            .lock()
            .unwrap_or_else(|e| e.into_inner())
            .get(index as usize)
            .map(|(res, _)| windows::core::Interface::as_raw(res));
        let Some(held) = held else {
            return true;
        };
        is_placed_texture(view)
            && placed_texture(view).is_some_and(|p| {
                let (View::Shader(r, _) | View::Unordered(r, ..)) = p else {
                    return false;
                };
                !crate::resource::host_copy(r as crate::object::This).is_some_and(|copy| {
                    windows::core::Interface::as_raw(&copy.texture) == held
                        || copy.twin.get().and_then(Option::as_ref).is_some_and(|t| {
                            t.planes
                                .iter()
                                .any(|p| windows::core::Interface::as_raw(&p.texture) == held)
                        })
                })
            })
    }

    /// Makes host slot `index` from the record the title has at `handle`
    /// if it changed since, and returns the view recorded there, pinned for
    /// as long as the caller uses it. Buffer views are made only for draws
    /// (`for_draw`), which also copy their data: a host copy of a buffer is
    /// made on first use.
    /// Whether host slot `index` already holds the mirrored view currently
    /// named by `handle`, without taking any lock.
    pub(crate) fn slot_ready(&self, index: u32, handle: u64) -> bool {
        let Some(slot) = self.ready.get(index as usize) else {
            return false;
        };
        // SAFETY: `handle` is a descriptor slot in a live title heap.
        let [tag, i, generation] = unsafe { (handle as *const [u32; 3]).read_unaligned() };
        let key = u64::from(generation) << 32 | u64::from(i);
        tag == VIEW_TAG && key != 0 && slot.load(Ordering::Acquire) == key
    }

    pub(crate) fn mirror(&self, index: u32, handle: u64, for_draw: bool) -> Option<ViewPin> {
        if index >= self.count {
            // Diagnostic (NVIDIA heap cap): the highest title slot past the
            // host heap seen so far.
            static HIGHEST: std::sync::atomic::AtomicU32 = std::sync::atomic::AtomicU32::new(0);
            if HIGHEST.fetch_max(index, std::sync::atomic::Ordering::Relaxed) < index {
                tracing::warn!(
                    index,
                    count = self.count,
                    for_draw,
                    "title slot past the host heap"
                );
            }
            return None;
        }
        // SAFETY: the handle is a descriptor slot in title memory.
        let [tag, i, generation] = unsafe { (handle as *const [u32; 3]).read_unaligned() };
        // Generations start at 1, so a made slot's key is never 0.
        let key = u64::from(generation) << 32 | u64::from(i);
        // Made from this record already and readable as it is (what the
        // checks below find under the lock): nothing to make.
        if tag == VIEW_TAG
            && self.ready[index as usize].load(Ordering::Acquire) == key
            && let Some(view) = view_at(i, generation)
            && is_sampler(&view) == self.sampler
            && !self.needs_remake(index, &view, for_draw)
        {
            return Some(view);
        }
        let mut made = self.made.lock().unwrap_or_else(|e| e.into_inner());
        // A second recorder may have replaced the slot while this thread waited.
        // SAFETY: the caller holds the heap alive until the update completes.
        let [tag, i, generation] = unsafe { (handle as *const [u32; 3]).read_unaligned() };
        let key = u64::from(generation) << 32 | u64::from(i);
        let view = (tag == VIEW_TAG).then(|| view_at(i, generation)).flatten();
        let Some(view) = view else {
            // Not a view of ours (never written, or of a resource the
            // title released): nothing to read.
            if made[index as usize] != 0 {
                self.ready[index as usize].store(0, Ordering::Relaxed);
                self.synced[index as usize].store(0, Ordering::Relaxed);
                made[index as usize] = 0;
                self.make_null(index);
                self.note_address(index, None);
                self.set_made_nonzero(index, false);
            }
            return None;
        };
        let view_is_sampler = is_sampler(&view);
        if view_is_sampler != self.sampler {
            warn_heap_mismatch(index, self.sampler, view_is_sampler);
            if made[index as usize] != 0 {
                self.ready[index as usize].store(0, Ordering::Relaxed);
                made[index as usize] = 0;
                self.set_made_nonzero(index, false);
                self.make_null(index);
                self.note_address(index, None);
            }
            drop(made);
            if !self.sampler && crate::heap_worker::on() {
                self.uploads
                    .lock()
                    .unwrap_or_else(|e| e.into_inner())
                    .remove(index);
            }
            return None;
        }
        // A placed texture's view left null below is made when a draw reads
        // the slot through a table; until then the slot is not made again
        // (heap syncs revisit it at every submission). Also re-mirror if a
        // later unordered or renderable view upgraded the memory's HostTexture.
        let needs_remake = || self.needs_remake(index, &view, for_draw);
        if made[index as usize] == key && !needs_remake() {
            return Some(view);
        }
        // Shaders index the heap itself (`crate::bindless`): buffer views
        // are made when written.
        // shortcut: views of textures the title placed itself are made
        // when a draw reads them through a table: a host texture (and a
        // staging copy) for each of the ~20,000 the title has views of
        // ran the host out of memory (run b2).
        let at_once = placed_at_once();
        let later = !for_draw && defer_placed(&view, at_once);
        if !later {
            // What the view needs (the texture it views, the copy of the
            // memory) is made without the slot lock, which title threads
            // binding and writing views wait on (run evict5).
            drop(made);
            prepare(&view);
            made = self.made.lock().unwrap_or_else(|e| e.into_inner());
            // The title may have written another record there meanwhile (and
            // another thread made it): that one is the slot's.
            // SAFETY: as above.
            if unsafe { (handle as *const [u32; 3]).read_unaligned() } != [tag, i, generation] {
                drop(made);
                drop(view);
                return self.mirror(index, handle, for_draw);
            }
            if made[index as usize] == key && !needs_remake() {
                return Some(view);
            }
            // Lists already on the GPU read this slot by index: a texture
            // made while they may run is shown once they have ([`show_after`]).
            if !self.sampler && show_after() && fresh_texture(&view) {
                drop(made);
                self.hold(index, handle, key);
                return Some(view);
            }
        }
        {
            let keep_old = later && at_once == PlacedAtOnce::Keep && self.kept(index);
            self.ready[index as usize].store(0, Ordering::Relaxed);
            self.synced[index as usize].store(0, Ordering::Relaxed);
            self.set_stride(index, Some(&view));
            self.note_address(index, Some(&view));
            if later {
                // Made once a shader reads it (`crate::shadow::take_reads`).
                crate::shadow::forget_read(index);
            }
            made[index as usize] = key;
            self.set_made_nonzero(index, true);
            let (kept, mark) = if later {
                (None, None)
            } else {
                make_marked(&view, self.cpu(index))
            };
            if !keep_old {
                self.set_mark(index, mark);
            }
            if kept.is_none() && !self.sampler && !keep_old {
                null_for(&view, self.cpu(index));
            }
            if !self.sampler {
                self.viewed.set(index, Some(&view), kept.is_some());
            }
            let kept = kept.flatten();
            // A texture's view left null (or keeping a prior view) is made again when a draw reads it.
            let ready = if is_texture(&view) && kept.is_none() { 0 } else { key };
            if !keep_old {
                self.keep(index, kept);
            }
            self.synced[index as usize].store(key, Ordering::Release);
            self.ready[index as usize].store(ready, Ordering::Release);
        }
        drop(made);
        if !self.sampler && !PUBLISHING.get() && crate::heap_worker::on() {
            let mut uploads = self.uploads.lock().unwrap_or_else(|e| e.into_inner());
            if !for_draw && is_texture_view(&view) {
                uploads.written(index, crate::queue::presented());
            } else {
                uploads.remove(index);
            }
        }
        Some(view)
    }

    /// Publishes host slot `index` from `handle` if the title slot still holds
    /// record `key`. Called inside `Host::execute` once the texture's upload is
    /// recorded in the submission's upload list (`crate::heap_worker`).
    pub(crate) fn publish(&self, index: u32, handle: u64, key: u64) -> bool {
        if handle == 0 || index >= self.count {
            return false;
        }
        // SAFETY: the handle is a descriptor slot in title memory.
        let [tag, i, generation] = unsafe { (handle as *const [u32; 3]).read_unaligned() };
        if tag != VIEW_TAG || (u64::from(generation) << 32 | u64::from(i)) != key {
            return false;
        }
        let _guard = PublishingGuard(PUBLISHING.replace(true));
        self.mirror(index, handle, true).is_some()
    }

    /// Leaves slot `index` showing what it showed, to be made from the
    /// record `key` at `handle` at the next submission ([`show_after`]).
    fn hold(&self, index: u32, handle: u64, key: u64) {
        let fresh = self
            .held
            .lock()
            .unwrap_or_else(|e| e.into_inner())
            .insert(index, (handle, key))
            .is_none();
        if fresh {
            let n = SHOWN_AFTER[0].fetch_add(1, Ordering::Relaxed) + 1;
            if n == 1 {
                tracing::info!(
                    "a heap slot's new texture was made while a submission may still run: \
                     the slot is shown after it (VICTORY_SHOW_AFTER=1)"
                );
            }
        }
    }

    /// Makes the slots [`HostHeap::hold`] left as they were, if the title
    /// still has the same record there; returns how many were made. Called
    /// while preparing a submission, when no submission runs any more
    /// (`Host::execute`, after `crate::descriptor::sync_heaps`).
    pub(crate) fn show_held(&self) -> usize {
        let held: Vec<(u32, (u64, u64))> = {
            let mut held = self.held.lock().unwrap_or_else(|e| e.into_inner());
            if held.is_empty() {
                return 0;
            }
            held.drain().collect()
        };
        let mut shown = 0;
        for (index, (handle, key)) in held {
            // SAFETY: a descriptor slot in the live title heap the caller
            // holds (`SHADER_HEAPS`).
            let [tag, i, generation] = unsafe { (handle as *const [u32; 3]).read_unaligned() };
            if tag != VIEW_TAG || (u64::from(generation) << 32 | u64::from(i)) != key {
                // Written again since: that write made (or held) the slot.
                SHOWN_AFTER[2].fetch_add(1, Ordering::Relaxed);
                continue;
            }
            if self.mirror(index, handle, true).is_some() {
                shown += 1;
            }
        }
        let n = SHOWN_AFTER[1].fetch_add(shown as u64, Ordering::Relaxed) + shown as u64;
        if shown > 0 && (n == shown as u64 || n / 5000 != (n - shown as u64) / 5000) {
            tracing::info!(
                held = SHOWN_AFTER[0].load(Ordering::Relaxed),
                shown = n,
                rewritten = SHOWN_AFTER[2].load(Ordering::Relaxed),
                "heap slots shown after the submissions that ran while their texture was made \
                 (VICTORY_SHOW_AFTER=1), since the start"
            );
        }
        shown
    }

    /// Whether slot `index` waits for [`HostHeap::show_held`] (tests).
    #[cfg(test)]
    pub(crate) fn is_held(&self, index: u32) -> bool {
        self.held
            .lock()
            .unwrap_or_else(|e| e.into_inner())
            .contains_key(&index)
    }

    /// Whether slot `index` currently shows a non-null host view made from the
    /// exact record at `handle`.
    pub(crate) fn texture_slot_made_from(&self, index: u32, handle: u64) -> bool {
        if handle == 0 || index >= self.count {
            return false;
        }
        // SAFETY: descriptor slot in title memory.
        let [tag, i, generation] = unsafe { (handle as *const [u32; 3]).read_unaligned() };
        if tag != VIEW_TAG {
            return false;
        }
        let key = u64::from(generation) << 32 | u64::from(i);
        self.ready[index as usize].load(Ordering::Acquire) == key
    }

    pub(crate) fn texture_slots_due(&self, frame: u64, max: usize) -> Vec<(u32, u64)> {
        self.uploads
            .lock()
            .unwrap_or_else(|e| e.into_inner())
            .due(frame, max)
    }

    pub(crate) fn texture_slot_done(&self, index: u32, write: u64) {
        self.uploads
            .lock()
            .unwrap_or_else(|e| e.into_inner())
            .done(index, write);
    }

    pub(crate) fn texture_slot_later(&self, index: u32, write: u64, frame: u64, max_wait: u64) {
        self.uploads
            .lock()
            .unwrap_or_else(|e| e.into_inner())
            .later(index, write, frame, max_wait);
    }

    pub(crate) fn prioritize_texture_slots(&self, indices: &[u32], frame: u64) {
        self.uploads
            .lock()
            .unwrap_or_else(|e| e.into_inner())
            .prioritize(indices, frame);
    }

    #[allow(dead_code)] // hook for urgent bake-source pre-upload
    pub(crate) fn requeue_texture_slots(&self, indices: &[u32], frame: u64) {
        self.uploads
            .lock()
            .unwrap_or_else(|e| e.into_inner())
            .requeue(indices, frame);
    }

    #[allow(dead_code)] // hook for urgent bake-source pre-upload
    pub(crate) fn texture_slots_queued(&self, indices: &[u32]) -> Vec<(u32, u64)> {
        self.uploads
            .lock()
            .unwrap_or_else(|e| e.into_inner())
            .queued(indices)
    }

    pub(crate) fn has_queued_textures(&self) -> bool {
        !self
            .uploads
            .lock()
            .unwrap_or_else(|e| e.into_inner())
            .is_empty()
    }

    /// The view host slot `index` was made from, in a few words, with the
    /// title and host addresses of a buffer view (diagnostics). "null" for
    /// a slot never made or left null.
    #[cfg(feature = "diagnostics")]
    pub(crate) fn describe_slot(&self, index: u32) -> Option<String> {
        let key = *self.made.lock().unwrap_or_else(|e| e.into_inner()).get(index as usize)?;
        // A view the host could not make leaves a null view, its record
        // noted all the same; it keeps no resource.
        let kept = self.kept.lock().unwrap_or_else(|e| e.into_inner()).get(index as usize).is_some();
        if key == 0 || (!kept && !self.sampler) {
            return Some("null".into());
        }
        let view = view_at(key as u32, (key >> 32) as u32)?;
        let mut text: String = format!("{view:?}").chars().take(120).collect();
        if let Some((address, len)) = buffer_range(&view) {
            let host = crate::shadow::find(address, len.max(1)).map(|s| s.gpu(address));
            text.push_str(&format!(" title {address:#x}+{len:#x} host {host:#x?}"));
        }
        Some(text)
    }

    /// Title slots viewing one of the host textures `textures` (raw pointers),
    /// which are given up: each gets a null view of its kind and holds the
    /// texture no more (retired until the running submission has run), its
    /// record stays noted so heap syncs leave it, and the next draw reading
    /// it through a table or shader reading it through the heap (its read
    /// mark, cleared) makes it again. Slots past `count` stay made: recorded
    /// lists and host copies already name them, and there is no title record
    /// from which to make them again.
    /// The slots are found by a walk of the heap, once per pass that gives
    /// textures up: a map of each texture's slots, kept at every slot the
    /// title rewrote, cost ~25 us a slot (a texture viewed in thousands of
    /// slots had its list searched each time; 62,000 slots a frame, run
    /// evict3).
    pub(crate) fn evict(&self, textures: &std::collections::HashSet<usize>) {
        // Keep a title slot from being rebound between testing its held texture
        // and clearing the descriptor. Mirroring takes made before kept as well.
        let made = self.made.lock().unwrap_or_else(|e| e.into_inner());
        let slots: Vec<u32> = {
            let kept = self.kept.lock().unwrap_or_else(|e| e.into_inner());
            kept.iter()
                .filter(|&(i, k)| {
                    i < self.count as usize
                        && k.1.is_none()
                        && textures.contains(&(windows::core::Interface::as_raw(&k.0) as usize))
                })
                .map(|(i, _)| i as u32)
                .collect()
        };
        for index in slots {
            if let Some(ready) = self.ready.get(index as usize) {
                ready.store(0, Ordering::Relaxed);
            }
            let key = made.get(index as usize).copied().unwrap_or(0);
            let view = (key != 0).then(|| view_at(key as u32, (key >> 32) as u32)).flatten();
            match &view {
                Some(v) => null_for(v, self.cpu(index)),
                None => null_view(self.cpu(index)),
            }
            self.keep(index, None);
            if index < self.count {
                self.viewed.set(index, view.as_deref(), false);
                crate::shadow::forget_read(index);
                if view.as_deref().is_some_and(is_unordered_texture) {
                    self.nulled_unordered.lock().unwrap_or_else(|e| e.into_inner()).push(index);
                }
            }
        }
    }

    /// Makes again the unordered texture views [`HostHeap::evict`] left
    /// null, the title's heap starting at `start` with `step` bytes a slot:
    /// shaders indexing the heap may write through them now.
    pub(crate) fn remake_unordered(&self, start: u64, step: u64) -> usize {
        let slots = std::mem::take(&mut *self.nulled_unordered.lock().unwrap_or_else(|e| e.into_inner()));
        for &index in &slots {
            {
                let mut made = self.made.lock().unwrap_or_else(|e| e.into_inner());
                self.ready[index as usize].store(0, Ordering::Relaxed);
                self.synced[index as usize].store(0, Ordering::Release);
                made[index as usize] = 0;
                self.set_made_nonzero(index, false);
            }
            self.mirror(index, start + u64::from(index) * step, true);
        }
        slots.len()
    }

    /// The host textures (raw pointers) slots `slots` view.
    pub(crate) fn textures_at(&self, slots: &[u32]) -> Vec<usize> {
        let kept = self.kept.lock().unwrap_or_else(|e| e.into_inner());
        slots
            .iter()
            .filter_map(|&i| kept.get(i as usize).filter(|k| k.1.is_none()))
            .map(|k| windows::core::Interface::as_raw(&k.0) as usize)
            .collect()
    }

    /// The view of slot `index` if it is a placed texture's left unmade (either
    /// nulled or keeping a prior view under `VICTORY_PLACED_AT_ONCE=keep`).
    pub(crate) fn unmade_placed(&self, index: u32) -> Option<View> {
        let key = *self.made.lock().unwrap_or_else(|e| e.into_inner()).get(index as usize)?;
        let view = (key != 0).then(|| view_at(key as u32, (key >> 32) as u32)).flatten()?;
        let ready = self.ready[index as usize].load(Ordering::Acquire) == key;
        let kept = self.kept.lock().unwrap_or_else(|e| e.into_inner()).get(index as usize).is_some();
        (is_placed_texture(&view) && (!kept || !ready)).then(|| (*view).clone())
    }

    /// How `make_read` should handle an unmade slot when `heap_worker::on()`.
    pub(crate) fn unmade_read_kind(&self, index: u32) -> UnmadeRead {
        let Some(key) = self
            .made
            .lock()
            .unwrap_or_else(|e| e.into_inner())
            .get(index as usize)
            .copied()
        else {
            return UnmadeRead::None;
        };
        if key == 0 {
            return UnmadeRead::None;
        }
        let ready = self.ready[index as usize].load(Ordering::Acquire) == key;
        let kept = self
            .kept
            .lock()
            .unwrap_or_else(|e| e.into_inner())
            .get(index as usize)
            .is_some();
        if kept && ready {
            return UnmadeRead::None;
        }
        let Some(view) = view_at(key as u32, (key >> 32) as u32) else {
            return UnmadeRead::None;
        };
        if is_texture_view(&view) {
            UnmadeRead::Background
        } else if is_placed_texture(&view) {
            UnmadeRead::Sync
        } else {
            UnmadeRead::None
        }
    }

    /// A shader read slot `index` through the heap: the texture it views
    /// was used now.
    pub(crate) fn stamp_read(&self, index: u32) {
        let key = self.made.lock().unwrap_or_else(|e| e.into_inner()).get(index as usize).copied().unwrap_or(0);
        let Some(view) = (key != 0).then(|| view_at(key as u32, (key >> 32) as u32)).flatten() else {
            return;
        };
        let placed = placed_texture(&view);
        let current = placed.as_ref().unwrap_or(&view);
        if let View::Shader(resource, _) = current
            && let Some(t) = crate::resource::made_copy(*resource as crate::object::This)
        {
            t.stamp();
        }
    }

    /// Whether slot `index` was made from a record (tests).
    #[cfg(test)]
    pub(crate) fn made(&self, index: u32) -> bool {
        self.made.lock().unwrap_or_else(|e| e.into_inner())[index as usize] != 0
    }

    /// Whether slot `index` holds a host view of a resource.
    pub(crate) fn kept(&self, index: u32) -> bool {
        self.kept.lock().unwrap_or_else(|e| e.into_inner()).get(index as usize).is_some()
    }
}

/// Logs `what` the first time it happens.
macro_rules! once {
    ($($arg:tt)*) => {{
        static ONCE: std::sync::Once = std::sync::Once::new();
        ONCE.call_once(|| tracing::warn!($($arg)*));
    }};
}

/// A view description from its words, with the format as the host has it.
fn srv_desc(words: &[u32; 10]) -> D3D12_SHADER_RESOURCE_VIEW_DESC {
    let mut words = *words;
    words[0] = crate::resource::pc_format(words[0]);
    // SAFETY: 40 bytes in the host's layout.
    unsafe {
        std::ptr::from_ref(&words)
            .cast::<D3D12_SHADER_RESOURCE_VIEW_DESC>()
            .read_unaligned()
    }
}

/// The unsigned integer format with the layout of a 16-bit float or
/// signed integer format (one, two or four channels), if `format` is one.
pub(crate) fn half_format(format: u32) -> Option<u32> {
    match format {
        // R16G16B16A16_FLOAT and _SINT: R16G16B16A16_UINT.
        10 | 14 => Some(12),
        // R16G16_FLOAT and _SINT: R16G16_UINT.
        34 | 38 => Some(36),
        // R16_FLOAT and _SINT: R16_UINT.
        54 | 59 => Some(57),
        _ => None,
    }
}

/// `view` as a shader is shown it where it moves a buffer's 16-bit
/// channels as unsigned integers
/// (`victory_shader::emit::HostShader::half_views`): a buffer view in a
/// 16-bit float or signed format, in the unsigned format of the same
/// layout, through which every 16-bit pattern is read and stored as it
/// is. `None` for every other view: the title's own serves (an unsigned
/// one already carries the bits; a texture can only be viewed in the
/// formats it was made for).
pub(crate) fn half_view(view: &View) -> Option<View> {
    // Buffer views: dimension 1 in the title's description of either kind.
    const BUFFER: u32 = 1;
    match view {
        View::Shader(resource, Some(desc)) if desc[1] == BUFFER => {
            let mut desc = *desc;
            desc[0] = half_format(desc[0])?;
            Some(View::Shader(*resource, Some(desc)))
        }
        View::Unordered(resource, counter, Some(desc)) if desc[1] == BUFFER => {
            let mut desc = *desc;
            desc[0] = half_format(desc[0])?;
            Some(View::Unordered(*resource, *counter, Some(desc)))
        }
        View::Placed(unordered, resource, desc) if resource[0] == BUFFER && desc[1] == BUFFER => {
            let mut desc = *desc;
            desc[0] = half_format(desc[0])?;
            Some(View::Placed(*unordered, *resource, desc))
        }
        _ => None,
    }
}

/// The format of `view`, a buffer view through which a shader moves
/// 16-bit channels as unsigned integers
/// (`victory_shader::emit::HostShader::half_views`), if those integers
/// are not what the console gives: a format of wider channels, a
/// normalised one or a signed 8-bit one, whose values the console converts
/// to and from halves. `None` for the formats the integers serve exactly
/// (16 bits a channel of any kind, through [`half_view`] where needed, and
/// unsigned channels of fewer bits) and for any other view.
pub(crate) fn half_inexact(view: &View) -> Option<u32> {
    // Buffer views: dimension 1 in the title's description of either kind.
    const BUFFER: u32 = 1;
    // R16G16B16A16, R16G16 and R16 (float, unsigned, signed), then
    // R10G10B10A2, R8G8B8A8, R8G8 and R8 unsigned.
    const EXACT: [u32; 13] = [10, 12, 14, 34, 36, 38, 54, 57, 59, 25, 30, 50, 62];
    let format = match view {
        View::Shader(_, Some(desc)) | View::Unordered(_, _, Some(desc)) if desc[1] == BUFFER => {
            desc[0]
        }
        View::Placed(_, resource, desc) if resource[0] == BUFFER && desc[1] == BUFFER => desc[0],
        _ => return None,
    };
    (!EXACT.contains(&format)).then_some(format)
}

fn uav_desc(words: &[u32; 10]) -> D3D12_UNORDERED_ACCESS_VIEW_DESC {
    let mut words = *words;
    words[0] = crate::resource::pc_format(words[0]);
    // SAFETY: 40 bytes in the host's layout.
    unsafe {
        std::ptr::from_ref(&words)
            .cast::<D3D12_UNORDERED_ACCESS_VIEW_DESC>()
            .read_unaligned()
    }
}

/// The byte stride of a buffer view (its element size, at least 1), 0 for
/// other views: what the view-stride table holds for it.
pub(crate) fn view_stride(view: &View) -> u32 {
    let (unordered, words): (bool, &[u32; 10]) = match view {
        // Read as raw buffers (`make`).
        View::Constant(..) => return 4,
        View::Shader(_, Some(desc)) => (false, desc),
        View::Unordered(_, _, Some(desc)) => (true, desc),
        View::Placed(unordered, resource, desc) if resource[0] == 1 => {
            match desc[..10].try_into() {
                Ok(words) => (*unordered, words),
                Err(_) => return 0,
            }
        }
        _ => return 0,
    };
    // The view dimension: buffer (1) for shader and unordered views alike.
    if words[1] != 1 {
        return 0;
    }
    let element = if unordered {
        let desc = uav_desc(words);
        // SAFETY: a buffer view's union member.
        let buffer = unsafe { desc.Anonymous.Buffer };
        let raw = buffer.Flags.0 & D3D12_BUFFER_UAV_FLAG_RAW.0 != 0;
        buffer_element(buffer.StructureByteStride, raw, desc.Format.0).unwrap_or(0)
    } else {
        element_size(&srv_desc(words))
    };
    // Nonzero: shaders taking views at computed indices read the null buffer
    // view where the table holds 0 (`victory_shader::emit`).
    element.max(1) as u32
}

/// The view-stride table's format-kind bits of a buffer view: a typed view
/// (no structure stride, not raw) of an integer format gets
/// `VIEW_INTEGER`, a signed one also `VIEW_SIGNED`; anything else none.
/// Translated shaders select integer typed-buffer overloads for such views,
/// and pass their 16-bit typed data as bits (`victory_shader::emit`, "d16").
fn view_kind(view: &View) -> u32 {
    use victory_shader::emit::{VIEW_INTEGER, VIEW_SIGNED};
    let (unordered, words): (bool, &[u32; 10]) = match view {
        View::Shader(_, Some(desc)) => (false, desc),
        View::Unordered(_, _, Some(desc)) => (true, desc),
        View::Placed(unordered, resource, desc) if resource[0] == 1 => match desc[..10].try_into() {
            Ok(words) => (*unordered, words),
            Err(_) => return 0,
        },
        _ => return 0,
    };
    // Buffer views only (dimension 1); a structured or raw one is no typed view.
    let (format, typed) = if unordered {
        let desc = uav_desc(words);
        // SAFETY: a buffer view's union member.
        let b = unsafe { desc.Anonymous.Buffer };
        let raw = b.Flags.0 & D3D12_BUFFER_UAV_FLAG_RAW.0 != 0;
        (desc.Format.0 as u32, typed_buffer(b.StructureByteStride, raw))
    } else {
        let desc = srv_desc(words);
        // SAFETY: a buffer view's union member.
        let b = unsafe { desc.Anonymous.Buffer };
        let raw = b.Flags.0 & D3D12_BUFFER_SRV_FLAG_RAW.0 != 0;
        (desc.Format.0 as u32, typed_buffer(b.StructureByteStride, raw))
    };
    match (words[1] == 1 && typed).then(|| integer_format(format)).flatten() {
        Some(true) => VIEW_INTEGER | VIEW_SIGNED,
        Some(false) => VIEW_INTEGER,
        None => 0,
    }
}

/// The four-channel host format for a three-channel 32-bit typed SRV.
fn widened_format(format: i32) -> Option<DXGI_FORMAT> {
    match format {
        6 => Some(DXGI_FORMAT_R32G32B32A32_FLOAT),
        7 => Some(DXGI_FORMAT_R32G32B32A32_UINT),
        8 => Some(DXGI_FORMAT_R32G32B32A32_SINT),
        _ => None,
    }
}

/// The fourth component's bits for a widened three-channel SRV. Integer
/// formats must not receive the bit pattern of floating-point 1.0.
fn widened_w(format: i32) -> Option<u32> {
    match format {
        6 => Some(1.0f32.to_bits()),
        7 | 8 => Some(1),
        _ => None,
    }
}

/// Whether `view` is a typed three-channel 32-bit SRV backed by a widened
/// host copy under Wine.
pub(crate) fn three_channel_srv(view: &View) -> bool {
    let words: &[u32; 10] = match view {
        View::Shader(_, Some(desc)) => desc,
        View::Placed(false, resource, desc) if resource[0] == 1 => match desc[..10].try_into() {
            Ok(words) => words,
            Err(_) => return false,
        },
        _ => return false,
    };
    let desc = srv_desc(words);
    // SAFETY: a buffer view's union member.
    let buffer = unsafe { desc.Anonymous.Buffer };
    let raw = buffer.Flags.0 & D3D12_BUFFER_SRV_FLAG_RAW.0 != 0;
    desc.ViewDimension == D3D12_SRV_DIMENSION_BUFFER
        && matches!(desc.Format.0, 6..=8)
        && typed_buffer(buffer.StructureByteStride, raw)
}

/// Whether a DXGI format is an integer one, and if so whether signed:
/// `Some(false)` for `UINT`, `Some(true)` for `SINT`, `None` otherwise.
fn integer_format(format: u32) -> Option<bool> {
    match format {
        3 | 7 | 12 | 17 | 25 | 30 | 36 | 42 | 50 | 57 | 62 => Some(false),
        4 | 8 | 14 | 18 | 32 | 38 | 43 | 52 | 59 | 64 => Some(true),
        _ => None,
    }
}

/// Bytes per element of a buffer view: its structure stride if it has one,
/// 4 for a raw view, else its format's element size (`None` if the format
/// has none).
fn buffer_element(stride: u32, raw: bool, format: i32) -> Option<u64> {
    if stride > 0 {
        Some(u64::from(stride))
    } else if raw {
        Some(4)
    } else {
        crate::resource::element(format as u32).map(|(bytes, _)| bytes)
    }
}

/// Whether a buffer view is typed: no structure stride, not raw.
fn typed_buffer(stride: u32, raw: bool) -> bool {
    stride == 0 && !raw
}

/// Bytes per element of a shader resource buffer view, 0 for a format with
/// none.
fn element_size(desc: &D3D12_SHADER_RESOURCE_VIEW_DESC) -> u64 {
    // SAFETY: a buffer view's union member.
    let buffer = unsafe { desc.Anonymous.Buffer };
    let raw = buffer.Flags.0 & D3D12_BUFFER_SRV_FLAG_RAW.0 != 0;
    buffer_element(buffer.StructureByteStride, raw, desc.Format.0).unwrap_or(0)
}

/// Writes slot `index`'s words into the view addresses at `data` (the
/// caller keeps `index` below [`HostHeap::address_slots`]), 16 bytes a
/// slot: a buffer view's title address (dword 0 low, dword 1 bits 0-15
/// high), a texture view's size fields ([`texture_words`]), zeros for
/// anything else.
fn write_address(data: usize, index: usize, view: Option<&View>) {
    let address = view.and_then(buffer_range).map_or(0, |(a, _)| a);
    let words = match view.and_then(texture_words) {
        Some(words) => words,
        None => [address as u32, (address >> 32) as u32 & 0xFFFF, 0, 0],
    };
    // SAFETY: the mapped view addresses hold 16 bytes per title and dynamic
    // slot (below `address_slots`).
    unsafe { ((data + 16 * index) as *mut [u32; 4]).write_unaligned(words) };
}

/// A texture view's console image descriptor dwords 0-3 as far as its size
/// goes: width - 1 and height - 1 of the resource's level 0 (dword 1 bits
/// 30-31, dword 2 bits 0-13 and 14-29), dword 3 an identity swizzle, the
/// view's first and last level (bits 12-15, 16-19) and the image type (bits
/// 28-31). `None` for buffers and samplers.
// shortcut: the address, format and tiling fields read 0 (no translated
// shader reads them: the shader refuses such reads where it can tell),
// and the swizzle is the identity whatever the view's component mapping.
fn texture_words(view: &View) -> Option<[u32; 4]> {
    let (resource, desc, unordered) = match view {
        View::Shader(r, desc) => (crate::resource::texture_desc(*r as crate::object::This)?, *desc, false),
        View::Unordered(r, _, desc) => (crate::resource::texture_desc(*r as crate::object::This)?, *desc, true),
        View::Placed(unordered, resource, desc) if resource[0] != 1 => {
            (*resource, desc[..10].try_into().ok(), *unordered)
        }
        _ => return None,
    };
    Some(image_words(&resource, desc.as_ref(), unordered))
}

/// [`texture_words`] of a texture of resource description `resource` (as
/// words) through a view of description `desc` (none: the resource's own).
fn image_words(resource: &[u32; 14], desc: Option<&[u32; 10]>, unordered: bool) -> [u32; 4] {
    let width = u64::from(resource[4]) | u64::from(resource[5]) << 32;
    let height = resource[6];
    let (depth, mips) = (resource[7] & 0xFFFF, resource[7] >> 16);
    let (samples, dimension) = (resource[9], resource[0]);
    // Mip count 0: the whole chain.
    let deepest = if dimension == 4 { depth } else { 1 };
    let chain = 64 - (width.max(u64::from(height)).max(u64::from(deepest)) | 1).leading_zeros();
    let mips = if mips == 0 { chain } else { mips };
    // Image types: 8 1D, 9 2D, 10 3D, 11 cube, 12 1D array, 13 2D array,
    // 14 2D MSAA, 15 2D MSAA array.
    let (ty, first, count) = match desc {
        // Shader and unordered view dimensions number alike up to 3D (8).
        Some(d) => {
            let ty = [0, 0, 8, 12, 9, 13, 14, 15, 10, 11, 11].get(d[1] as usize).copied().unwrap_or(0);
            match (unordered, d[1]) {
                (_, 6 | 7) => (ty, 0, 1),
                // An unordered view's one level is its word 2 (`MipSlice`:
                // its union starts at byte 8); word 4 is an array's size or
                // a 3D view's depth. A shader view's first level and count
                // are words 4 and 5 (its union starts at byte 16).
                (true, _) => (ty, d[2], 1),
                (false, _) => (ty, d[4], if d[5] == u32::MAX { mips.saturating_sub(d[4]) } else { d[5] }),
            }
        }
        None => {
            let array = depth > 1;
            let ty = match dimension {
                2 if array => 12,
                2 => 8,
                3 if samples > 1 => 14 + u32::from(array),
                3 => 9 + 4 * u32::from(array),
                _ => 10,
            };
            (ty, 0, if unordered { 1 } else { mips })
        }
    };
    let (w, h) = (width.saturating_sub(1) as u32, height.saturating_sub(1));
    let last = first + count.max(1) - 1;
    [
        0,
        (w & 3) << 30,
        (w >> 2 & 0x3FFF) | (h & 0xFFFF) << 14,
        0xFAC | (first & 0xF) << 12 | (last & 0xF) << 16 | ty << 28,
    ]
}

/// The title memory a buffer shader or unordered access view reads: its
/// address and length.
pub(crate) fn buffer_range(view: &View) -> Option<(u64, u64)> {
    let (base, words): (u64, &[u32; 10]) = match view {
        View::Shader(r, Some(desc)) => (
            crate::resource::buffer_bytes(*r as crate::object::This)?.0 as u64,
            desc,
        ),
        View::Placed(false, resource, desc) if resource[0] == 1 => (
            u64::from(desc[10]) | u64::from(desc[11]) << 32,
            desc[..10].try_into().ok()?,
        ),
        View::Unordered(r, _, Some(desc)) => {
            let base = crate::resource::buffer_bytes(*r as crate::object::This)?.0 as u64;
            return unordered_range(base, desc);
        }
        View::Placed(true, resource, desc) if resource[0] == 1 => {
            let base = u64::from(desc[10]) | u64::from(desc[11]) << 32;
            return unordered_range(base, desc[..10].try_into().ok()?);
        }
        _ => return None,
    };
    let desc = srv_desc(words);
    let element = element_size(&desc);
    // SAFETY: a buffer view's union member.
    let buffer = unsafe { desc.Anonymous.Buffer };
    Some((
        base + buffer.FirstElement * element,
        u64::from(buffer.NumElements) * element,
    ))
}

/// The host copy a buffer view's bytes are in (the one its host view
/// reads, [`crate::shadow::for_view`]) and the view's title range.
pub(crate) fn view_copy(view: &View) -> Option<(std::sync::Arc<crate::shadow::Shadow>, u64, u64)> {
    let (address, len) = buffer_range(view)?;
    let (shadow, _) = crate::shadow::for_view(address, len, u64::from(view_stride(view)))?;
    Some((shadow, address, len))
}

/// [`buffer_range`] of an unordered access view of memory at `base`.
fn unordered_range(base: u64, words: &[u32; 10]) -> Option<(u64, u64)> {
    let desc = uav_desc(words);
    // SAFETY: a buffer view's union member.
    let buffer = unsafe { desc.Anonymous.Buffer };
    let raw = buffer.Flags.0 & D3D12_BUFFER_UAV_FLAG_RAW.0 != 0;
    let element = buffer_element(buffer.StructureByteStride, raw, desc.Format.0)?;
    Some((
        base + buffer.FirstElement * element,
        u64::from(buffer.NumElements) * element,
    ))
}

/// The host resource a slot made by [`make`] views, and the view's own copy
/// of title memory ([`crate::shadow::for_view`]): the slot holds it, so that
/// draws reading the view find and refresh that copy.
type Held = (ID3D12Resource, Option<HeldSource>);

/// The fields keep the normal and widened copies alive for as long as the
/// slot shows them.
#[allow(dead_code)]
#[derive(Clone)]
enum HeldSource {
    Shadow(std::sync::Arc<crate::shadow::Shadow>),
    /// A widened three-channel view and the normal shadow it reads.
    Widened(
        std::sync::Arc<crate::shadow::Widened>,
        std::sync::Arc<crate::shadow::Shadow>,
    ),
}

/// What a slot made by [`make`] holds, if anything.
type Kept = Option<Held>;

/// Buffer views refused so far (diagnostics: the first are logged).
#[cfg(feature = "diagnostics")]
static REFUSED: std::sync::atomic::AtomicU32 = std::sync::atomic::AtomicU32::new(0);

fn buffer_srv(
    address: u64,
    len: u64,
    mut desc: D3D12_SHADER_RESOURCE_VIEW_DESC,
    at: D3D12_CPU_DESCRIPTOR_HANDLE,
) -> Option<Kept> {
    let element = element_size(&desc);
    // SAFETY: a buffer view's union member.
    let buffer = unsafe { &mut desc.Anonymous.Buffer };
    if element == 0 {
        once!(len, "buffer view of elements without a size");
        return None;
    }
    // The copy its own bytes are in, its first element counted from there.
    let start = address + buffer.FirstElement * element;
    let Some((shadow, own)) =
        crate::shadow::for_view(start, u64::from(buffer.NumElements) * element, element)
    else {
        #[cfg(feature = "diagnostics")]
        if REFUSED.fetch_add(1, std::sync::atomic::Ordering::Relaxed) < 200 {
            tracing::info!(start = format_args!("{start:#x}"), len, element, count = buffer.NumElements, "refused buffer view: no host copy");
        }
        return None;
    };
    buffer.FirstElement = shadow.offset(start) / element;
    let (first, count) = (buffer.FirstElement, buffer.NumElements);
    if !in_buffer(&shadow.buffer, first, count, element) {
        #[cfg(feature = "diagnostics")]
        if REFUSED.fetch_add(1, std::sync::atomic::Ordering::Relaxed) < 200 {
            tracing::info!(start = format_args!("{start:#x}"), first, count, element, "refused buffer view: empty or past its copy");
        }
        once!(first, count, element, "buffer view empty or past its host copy");
        return None;
    }
    let host = host()?;
    if crate::under_wine()
        && matches!(desc.Format.0, 6..=8)
        && let Some(w) = widened_w(desc.Format.0)
        && let Some(widened) = shadow.widened(start, u64::from(count) * 12, w)
    {
        desc.Format = widened_format(desc.Format.0).expect("a three-channel format");
        buffer.FirstElement = 0;
        buffer.NumElements = count;
        // SAFETY: a live widened buffer and a slot of our heap.
        unsafe {
            host.device
                .CreateShaderResourceView(&widened.buffer, Some(&desc), at)
        };
        return Some(Some((
            widened.buffer.clone(),
            Some(HeldSource::Widened(widened, shadow.clone())),
        )));
    }
    let stride = u64::from(buffer.StructureByteStride);
    if stride > 0 && raw_structured() {
        // Scalar and offset-only buffer reads emit `rawBufferLoad` on a raw
        // handle (`victory_shader::emit`), which D3D12 leaves undefined on a
        // structured SRV (NVIDIA treats the byte offset as an element index,
        // hanging `WindNew` loops). NVIDIA, WARP and D3DMetal all read both
        // raw handles and structured handles accurately from a raw view of the
        // same bytes; under Wine/D3DMetal any dword-aligned start (`+8B` or
        // `+680B`) works too, while native D3D12 requires 16-byte alignment.
        let (byte, dwords) = (first * stride, u64::from(count) * stride / 4);
        let align = if crate::shader_cache::under_wine() { 4 } else { 16 };
        match u32::try_from(dwords) {
            Ok(dwords) if byte.is_multiple_of(align) && stride.is_multiple_of(4) => {
                desc.Format = DXGI_FORMAT_R32_TYPELESS;
                desc.Anonymous.Buffer = D3D12_BUFFER_SRV {
                    FirstElement: byte / 4,
                    NumElements: dwords,
                    StructureByteStride: 0,
                    Flags: D3D12_BUFFER_SRV_FLAG_RAW,
                };
            }
            _ => once!(byte, stride, "structured buffer view not made raw: unaligned"),
        }
    }
    // SAFETY: a live buffer and a slot of our heap.
    unsafe {
        host.device
            .CreateShaderResourceView(&shadow.buffer, Some(&desc), at)
    };
    Some(Some((
        shadow.buffer.clone(),
        own.then(|| HeldSource::Shadow(shadow)),
    )))
}

/// Whether structured buffer views are made raw ([`buffer_srv`]): under Wine
/// (D3DMetal) and on NVIDIA hosts, both of which serve structured-handle and
/// raw-handle `rawBufferLoad`s from a raw view of the same bytes.
/// `VICTORY_RAW_STRUCTURED=0` or `=1` overrides.
fn raw_structured() -> bool {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| {
        let on = match std::env::var("VICTORY_RAW_STRUCTURED").as_deref() {
            Ok("0" | "false" | "off") => false,
            Ok("1" | "true" | "on") => true,
            _ => crate::shader_cache::under_wine() || crate::host::host_vendor() == 0x10DE,
        };
        tracing::info!(on, "structured buffer views made raw");
        on
    })
}

/// Whether `count` elements of `element` bytes from element `first` lie in
/// `buffer`. The console takes a view that is empty or runs past its
/// memory; the host removes the device for one.
fn in_buffer(buffer: &ID3D12Resource, first: u64, count: u32, element: u64) -> bool {
    // SAFETY: a live buffer.
    within(unsafe { buffer.GetDesc() }.Width, first, count, element)
}

fn within(width: u64, first: u64, count: u32, element: u64) -> bool {
    count > 0 && first.saturating_add(u64::from(count)).saturating_mul(element) <= width
}

/// An unordered access view of title buffer memory at `address` (`len`
/// bytes) on its host copy, as [`buffer_srv`]; false if the copy cannot be
/// written by shaders (upload memory).
/// shortcut: what shaders write stays in the host copy; title memory (and
/// so the CPU, and copies run on the CPU) does not see it.
fn buffer_uav(
    address: u64,
    len: u64,
    mut desc: D3D12_UNORDERED_ACCESS_VIEW_DESC,
    at: D3D12_CPU_DESCRIPTOR_HANDLE,
) -> Option<Kept> {
    // SAFETY: a buffer view's union member.
    let buffer = unsafe { &mut desc.Anonymous.Buffer };
    let raw = buffer.Flags.0 & D3D12_BUFFER_UAV_FLAG_RAW.0 != 0;
    let element = buffer_element(buffer.StructureByteStride, raw, desc.Format.0).unwrap_or(0);
    if element == 0 {
        once!(len, "unordered buffer view of elements without a size");
        return None;
    }
    let format = desc.Format.0;
    let typed_three_channel =
        typed_buffer(buffer.StructureByteStride, raw) && matches!(format, 6..=8);
    // D3DMetal 3.0 reports typed-UAV support for these formats, then tries
    // to make an unsupported three-channel Metal format and aborts inside the
    // void CreateUnorderedAccessView call. D3D12 exposes no D3DMetal version,
    // so keep the Wine-wide guard until 4.0 is characterized. The caller
    // installs a null UAV and keeps no resource or mark.
    if typed_three_channel && crate::under_wine() {
        once!(format, "three-channel typed unordered buffer view under Wine");
        return None;
    }
    let start = address + buffer.FirstElement * element;
    let (shadow, own) = crate::shadow::for_view(start, u64::from(buffer.NumElements) * element, element)?;
    // SAFETY: a live buffer.
    let flags = unsafe { shadow.buffer.GetDesc() }.Flags;
    if flags.0 & D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS.0 == 0 {
        once!("unordered access to a buffer whose host copy shaders cannot write");
        return None;
    }
    buffer.FirstElement = shadow.offset(start) / element;
    buffer.CounterOffsetInBytes = 0;
    let (first, count) = (buffer.FirstElement, buffer.NumElements);
    if !in_buffer(&shadow.buffer, first, count, element) {
        once!(first, count, element, "unordered buffer view empty or past its host copy");
        return None;
    }
    let host = host()?;
    if typed_three_channel {
        // The COM method has no HRESULT. On native hosts, reject a format
        // the device does not advertise and a device already removed.
        // SAFETY: plain device queries.
        if unsafe { host.device.GetDeviceRemovedReason() }.is_err() || !typed_store(format as u32) {
            once!(format, "three-channel typed unordered buffer view the host cannot make");
            return None;
        }
    }
    // SAFETY: a live buffer and a slot of our heap.
    unsafe {
        host.device.CreateUnorderedAccessView(
            &shadow.buffer,
            None::<&ID3D12Resource>,
            Some(&desc),
            at,
        )
    };
    // CreateUnorderedAccessView is void and cannot report failure directly.
    // If it removed the native device, do not keep its descriptor or mark.
    // SAFETY: a plain device query.
    if typed_three_channel && unsafe { host.device.GetDeviceRemovedReason() }.is_err() {
        once!(format, "device removed making a three-channel typed unordered buffer view");
        return None;
    }
    MARK.with(|m| m.set(shadow.mark_entry(start)));
    Some(Some((
        shadow.buffer.clone(),
        own.then(|| HeldSource::Shadow(shadow)),
    )))
}

thread_local! {
    /// The view-mark entry of the unordered buffer view [`make`] last made
    /// on this thread ([`make_marked`]).
    static MARK: std::cell::Cell<Option<(u32, u32)>> = const { std::cell::Cell::new(None) };
}

/// [`make`], and the view-mark entry of the view if it is an unordered
/// buffer view whose writes shaders mark (`victory_shader::emit`, "Write
/// marks").
fn make_marked(view: &View, at: D3D12_CPU_DESCRIPTOR_HANDLE) -> (Option<Kept>, Option<(u32, u32)>) {
    MARK.with(|m| m.set(None));
    let kept = make(view, at);
    (kept, MARK.with(std::cell::Cell::take))
}

/// Whether the host can make a shader (or unordered access) view of
/// `texture` in `format` (none: the texture's own). The console allows any
/// view; the host refuses (and removes the device for) views without the
/// resource's permission, of typed depth, and in a format neither of the
/// resource's family nor among the formats it was made castable to
/// (`castable`, `resource::castable`). A view of a typed texture in another
/// format of its family takes the texture's format unless it is castable.
/// shortcut: such a view (sRGB of UNORM, say, on a texture the host made
/// without castable formats) reads without its conversion.
fn fits(copy: &crate::resource::HostTexture, mut format: Option<&mut DXGI_FORMAT>, unordered: bool) -> bool {
    use crate::resource::{DEPTH_FORMATS, family};
    // SAFETY: a live resource.
    let d = unsafe { copy.texture.GetDesc() };
    let castable = &copy.castable;
    let allowed = if unordered {
        d.Flags.0 & D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS.0 != 0
    } else {
        d.Flags.0 & D3D12_RESOURCE_FLAG_DENY_SHADER_RESOURCE.0 == 0
    };
    let own = d.Format.0 as u32;
    if !allowed || DEPTH_FORMATS.iter().any(|&(typed, _)| typed == own) {
        once!(
            unordered,
            format = own,
            flags = d.Flags.0,
            "view the host does not allow"
        );
        return false;
    }
    // Every family has more than one format, its typeless one first.
    let is_typeless = |f: u32| family(f) == f && family(f + 1) == f;
    let typeless = is_typeless(own);
    if let Some(format) = format.as_deref_mut()
        && unordered
        && let Some(&(_, linear)) = crate::resource::SRGB
            .iter()
            .find(|&&(s, _)| s == format.0 as u32)
    {
        format.0 = linear as i32;
    }
    let view = format.as_ref().map_or(0, |f| f.0 as u32);
    // A typed unordered view in a format the host cannot store to
    // (B5G5R5A1_UNORM on NVIDIA) removes the device when made, in the
    // texture's own format too: the slot stays null, its writes dropped.
    // The format checked is the one the view is made in: a typed
    // texture's own unless the requested one is castable (below).
    let typed = if view == 0 || (!typeless && !castable.contains(&view)) { own } else { view };
    if unordered && !(view == 0 && typeless) && !typed_store(typed) {
        once!(view = typed, own, "an unordered view in a format the host cannot store");
        return false;
    }
    if view == 0 {
        return !typeless && (!unordered || typed_store(own));
    }
    // A packed 16-bit target made R8G8B8A8_UNORM on this GPU
    // (`resource::target_format`): views name the format it has.
    if let Some(packed) = copy.promoted_from {
        if packed != view {
            return false;
        }
        if let Some(format) = format {
            *format = d.Format;
        }
        return true;
    }
    if DEPTH_FORMATS.iter().any(|&(typed, _)| typed == view) || is_typeless(view) {
        once!(unordered, view, own, "view the host does not allow");
        return false;
    }
    let cast = castable.contains(&view);
    if family(view) != family(own) && !cast {
        once!(view, own, "view format of another family");
        return false;
    }
    let effective = if typeless || cast { view } else { own };
    if unordered && !typed_store(effective) {
        once!(
            view = effective,
            own,
            "an unordered view in a format the host cannot store"
        );
        return false;
    }
    if let Some(format) = format
        && !typeless
        && !cast
    {
        *format = d.Format;
    }
    true
}

/// Whether a view without a description of `resource` (a title texture)
/// may take its host texture's defaults: not when the host texture has
/// another format than the title's (typeless, or shared with a title
/// texture of another format of the same memory).
/// shortcut: such views are null; spell the description out (dimension,
/// mips) if the title makes them.
fn default_view(resource: crate::object::This) -> bool {
    let other = crate::resource::view_format(resource);
    if other.is_some() {
        once!(format = other, "a view without a description of a texture whose host texture has another format");
    }
    other.is_none()
}

/// Whether the host stores to typed unordered views of `format`.
fn typed_store(format: u32) -> bool {
    static KNOWN: Mutex<Vec<(u32, bool)>> = Mutex::new(Vec::new());
    let mut known = KNOWN.lock().unwrap_or_else(|e| e.into_inner());
    if let Some(&(_, ok)) = known.iter().find(|(f, _)| *f == format) {
        return ok;
    }
    let mut support = D3D12_FEATURE_DATA_FORMAT_SUPPORT {
        Format: DXGI_FORMAT(format as i32),
        ..Default::default()
    };
    let ok = host().is_some_and(|h| {
        // SAFETY: a plain query into a struct of its size.
        unsafe {
            h.device.CheckFeatureSupport(
                D3D12_FEATURE_FORMAT_SUPPORT,
                (&raw mut support).cast(),
                size_of_val(&support) as u32,
            )
        }
        .is_ok()
            && support.Support1.0 & D3D12_FORMAT_SUPPORT1_TYPED_UNORDERED_ACCESS_VIEW.0 != 0
    });
    known.push((format, ok));
    ok
}

/// Makes the host descriptor for `view` at `at` (in any host view heap):
/// the host resource it views, if it can.
pub(crate) fn make_view(view: &View, at: D3D12_CPU_DESCRIPTOR_HANDLE) -> Option<ID3D12Resource> {
    make(view, at).flatten().map(|k| k.0)
}

/// Makes the host descriptor for `view` at `at`: the host resource it
/// views, or `None` if it cannot.
fn make(view: &View, at: D3D12_CPU_DESCRIPTOR_HANDLE) -> Option<Kept> {
    let host = host()?;
    let device = &host.device;
    match view {
        &View::Constant(address, size) => {
            // Constant buffers in the title's slots, through the heap itself
            // or a table, are read as raw buffers by translated shaders
            // (`victory_shader::emit`, "Constant buffers in the heap"): the
            // console needs 16-byte alignment, a host constant buffer view
            // 256, which most of the title's do not have on the host.
            let shadow = crate::shadow::find(address, u64::from(size))?;
            let bytes = u64::from(size).min(shadow.left(address));
            let desc = D3D12_SHADER_RESOURCE_VIEW_DESC {
                Format: DXGI_FORMAT_R32_TYPELESS,
                ViewDimension: D3D12_SRV_DIMENSION_BUFFER,
                Shader4ComponentMapping: D3D12_DEFAULT_SHADER_4_COMPONENT_MAPPING,
                Anonymous: D3D12_SHADER_RESOURCE_VIEW_DESC_0 {
                    Buffer: D3D12_BUFFER_SRV {
                        NumElements: (bytes / 4) as u32,
                        Flags: D3D12_BUFFER_SRV_FLAG_RAW,
                        ..Default::default()
                    },
                },
            };
            buffer_srv(address, bytes, desc, at)
        }
        View::Shader(resource, desc) => {
            let resource = *resource as crate::object::This;
            if let Some(copy) = crate::resource::host_copy(resource) {
                copy.viewed.store(true, std::sync::atomic::Ordering::Relaxed);
                let texture = &copy.texture;
                let mut desc = desc.as_ref().map(srv_desc);
                if desc.is_none() && !default_view(resource) {
                    return None;
                }
                if let Some(desc) = desc.as_mut()
                    && desc.Format.0 == 0
                    && let Some(format) = crate::resource::view_format(resource)
                {
                    desc.Format = DXGI_FORMAT(format as i32);
                }
                if !fits(&copy, desc.as_mut().map(|d| &mut d.Format), false) {
                    return None;
                }
                // SAFETY: a live texture and a slot of our heap.
                unsafe {
                    device.CreateShaderResourceView(
                        texture,
                        desc.as_ref().map(std::ptr::from_ref),
                        at,
                    )
                };
                Some(Some((texture.clone(), None)))
            } else if let (Some((data, len)), Some(desc)) =
                (crate::resource::buffer_bytes(resource), desc)
            {
                buffer_srv(data as u64, len, srv_desc(desc), at)
            } else {
                None
            }
        }
        View::Unordered(resource, _counter, desc) => {
            let resource = *resource as crate::object::This;
            let Some(copy) = crate::resource::host_copy(resource) else {
                return match (crate::resource::buffer_bytes(resource), desc) {
                    (Some((data, len)), Some(desc)) => {
                        buffer_uav(data as u64, len, uav_desc(desc), at)
                    }
                    _ => None,
                };
            };
            copy.viewed.store(true, std::sync::atomic::Ordering::Relaxed);
            // Shaders indexing the heap write through this view without
            // moving the texture to a write state: kept from now on, unless
            // `VICTORY_EVICT_HEAP_UAV` lets it go until shaders write
            // through heaps (`crate::resource::evictable`).
            if crate::resource::evict_heap_uav() {
                copy.note_heap_writable();
            } else {
                copy.note_state(0x8);
            }
            let texture = &copy.texture;
            // A depth texture's unordered views write its twin.
            // SAFETY: a live resource.
            if unsafe { texture.GetDesc() }.Flags.0 & D3D12_RESOURCE_FLAG_ALLOW_DEPTH_STENCIL.0 != 0 {
                return crate::twin::uav(resource, &copy, desc.as_ref(), at)
                    .map(|twin| Some((twin, None)));
            }
            let mut desc = desc.as_ref().map(uav_desc);
            if desc.is_none() && !default_view(resource) {
                return None;
            }
            if let Some(desc) = desc.as_mut()
                && desc.Format.0 == 0
                && let Some(format) = crate::resource::view_format(resource)
            {
                desc.Format = DXGI_FORMAT(format as i32);
            }
            if !fits(&copy, desc.as_mut().map(|d| &mut d.Format), true) {
                return None;
            }
            // SAFETY: a live texture and a slot of our heap.
            unsafe {
                device.CreateUnorderedAccessView(
                    texture,
                    None::<&ID3D12Resource>,
                    desc.as_ref().map(std::ptr::from_ref),
                    at,
                )
            };
            Some(Some((texture.clone(), None)))
        }
        View::Sampler(words) => {
            // SAFETY: 13 words in the host's layout.
            let desc = unsafe {
                std::ptr::from_ref(words)
                    .cast::<D3D12_SAMPLER_DESC>()
                    .read_unaligned()
            };
            // SAFETY: a slot of our sampler heap.
            unsafe { device.CreateSampler(&desc, at) };
            Some(None)
        }
        View::ConsoleSampler(words) => {
            let desc = crate::sampler::console(words, crate::sampler::aniso_point_mip(device));
            // SAFETY: a slot of our sampler heap.
            unsafe { device.CreateSampler(&desc, at) };
            Some(None)
        }
        View::Placed(unordered, resource, desc) => {
            if let Some(view) = placed_texture(view) {
                return make(&view, at);
            }
            const BUFFER: u32 = 1;
            if resource[0] != BUFFER {
                return None;
            }
            // The resource description's width, after the alignment.
            let len = u64::from(resource[4]) | u64::from(resource[5]) << 32;
            let address = u64::from(desc[10]) | u64::from(desc[11]) << 32;
            let words: &[u32; 10] = desc[..10].try_into().unwrap_or(&[0; 10]);
            if *unordered {
                buffer_uav(address, len, uav_desc(words), at)
            } else {
                buffer_srv(address, len, srv_desc(words), at)
            }
        }
    }
}

/// Whether `view` is a sampler descriptor (must only be created on a sampler
/// heap, never on a CBV/SRV/UAV heap).
fn is_sampler(view: &View) -> bool {
    matches!(view, View::Sampler(_) | View::ConsoleSampler(_))
}

fn warn_heap_mismatch(index: u32, sampler_heap: bool, sampler_view: bool) {
    static WARNED: std::sync::atomic::AtomicBool = std::sync::atomic::AtomicBool::new(false);
    if !WARNED.swap(true, Ordering::Relaxed) {
        tracing::warn!(
            index,
            sampler_heap,
            sampler_view,
            "descriptor record kind mismatches host heap type; clearing slot"
        );
    }
}

/// Whether `view` is an unordered view of a resource object's texture.
fn is_unordered_texture(view: &View) -> bool {
    matches!(view, View::Unordered(r, _, _) if crate::resource::texture_memory(*r as crate::object::This).is_some())
}

/// Whether `view` is a view of a texture the title placed in memory itself.
pub(crate) fn is_placed_texture(view: &View) -> bool {
    matches!(view, View::Placed(_, resource, _) if resource[0] != 1)
}

/// Makes what a host view of `view` needs, if it is not made yet: the
/// texture it views (made on first use) or the host copy of the buffer
/// memory it views. The slot's view is then quick to make.
fn prepare(view: &View) {
    #[cfg(test)]
    tests::while_preparing();
    match view {
        View::Placed(..) if is_placed_texture(view) => {
            let _ = placed_texture(view);
        }
        View::Shader(r, _) | View::Unordered(r, _, _) if crate::resource::texture_memory(*r as crate::object::This).is_some() => {
            let _ = crate::resource::host_resource(*r as crate::object::This);
        }
        _ => {
            let _ = view_copy(view);
        }
    }
}

/// Whether `view` is a shader view of a texture (one placed in memory, or a
/// resource object's): such a slot left null is made when read.
/// Slots held for [`show_after`] since the start: held, made later, and
/// written again by the title before that.
static SHOWN_AFTER: [AtomicU64; 3] = [const { AtomicU64::new(0) }; 3];

#[cfg(test)]
thread_local! {
    /// Test-only per-thread override for [`show_after`].
    static TEST_SHOW_AFTER: std::cell::Cell<Option<bool>> = const { std::cell::Cell::new(None) };
}

/// Forces [`show_after`] on or off on this thread (`None`: the switch).
#[cfg(test)]
pub(crate) fn set_show_after(on: Option<bool>) {
    TEST_SHOW_AFTER.set(on);
}

/// Whether a heap slot whose new record views a host texture made while a
/// submission may still run keeps showing what it showed until the next
/// submission (`VICTORY_SHOW_AFTER=1`; off by default).
///
/// Lists on the GPU read heap slots by index. Changed at once, a slot shows
/// those lists a texture made after they were handed to the queue: under
/// D3D12 on an RX 7900 XTX every device loss with a page-fault address was at
/// a texture made and shown in the last tenth of a second (ported from
/// avatarmanz/VIctory-7900XTX's `VICTORY_SHOW_AFTER` and
/// `VICTORY_SHOW_NEW_LATER`). Here the held slot is made in the next
/// submission's preparation, when no submission runs, and before the
/// uploads of the textures heap slots view are collected.
pub(crate) fn show_after() -> bool {
    #[cfg(test)]
    if let Some(on) = TEST_SHOW_AFTER.get() {
        return on;
    }
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| crate::switch::on("VICTORY_SHOW_AFTER"))
}

/// Whether `view` views a host texture made while a submission may still
/// run ([`crate::resource::HostTexture::fresh`]).
fn fresh_texture(view: &View) -> bool {
    if !is_texture(view) {
        return false;
    }
    let resource = match view {
        View::Shader(r, _) => *r,
        _ => match placed_texture(view) {
            Some(View::Shader(r, _)) => r,
            _ => return false,
        },
    };
    crate::resource::host_copy(resource as crate::object::This).is_some_and(|copy| copy.fresh())
}

pub(crate) fn is_texture(view: &View) -> bool {
    is_placed_texture(view)
        || matches!(view, View::Shader(r, _) if crate::resource::texture_memory(*r as crate::object::This).is_some())
}

/// Whether `view` is a read-only (SRV) texture view suitable for background
/// creation and upload (`View::Placed(false, ..)` texture or `View::Shader` of
/// a title texture).
pub(crate) fn is_texture_view(view: &View) -> bool {
    matches!(view, View::Placed(false, resource, _) if resource[0] != 1)
        || matches!(view, View::Shader(r, _) if crate::resource::texture_memory(*r as crate::object::This).is_some())
}

/// A view of a texture the title placed in memory itself, as a view of the
/// resource made for that memory (on first use).
pub(crate) fn placed_texture(view: &View) -> Option<View> {
    const BUFFER: u32 = 1;
    let View::Placed(unordered, resource, desc) = view else {
        return None;
    };
    if resource[0] == BUFFER {
        return None;
    }
    let address = u64::from(desc[10]) | u64::from(desc[11]) << 32;
    let texture = crate::resource::texture_at(address, resource)?;
    let view_desc = desc[..10].try_into().ok();
    Some(if *unordered {
        View::Unordered(texture as usize, 0, view_desc)
    } else {
        View::Shader(texture as usize, view_desc)
    })
}

/// A null view of the kind `view` is, for a slot whose view the host
/// cannot make: unordered access (to a buffer or a 2D texture) or a shader
/// view. Writes to it are dropped, reads give zeros.
fn null_for(view: &View, at: D3D12_CPU_DESCRIPTOR_HANDLE) {
    let Some(host) = host() else { return };
    let is_buffer = |r: &usize| crate::resource::buffer_bytes(*r as crate::object::This).is_some();
    // Unordered views: whether a buffer, and the title's view dimension.
    let (buffer, dimension) = match view {
        View::Unordered(r, _, desc) => (
            is_buffer(r) || desc.is_some_and(|d| d[1] == 1),
            desc.map(|d| d[1]),
        ),
        View::Placed(true, resource, desc) => (resource[0] == 1, Some(desc[1])),
        // Shader views of buffers read nothing through a null buffer view:
        // a null 2D view where the shader expects a buffer is a mismatch
        // the hardware does not survive in general (GPU-based validation
        // reports it).
        View::Shader(r, desc) if is_buffer(r) || desc.is_some_and(|d| d[1] == 1) => {
            return null_buffer_view(at);
        }
        View::Placed(false, resource, _) if resource[0] == 1 => return null_buffer_view(at),
        // Textures: a null view of the title's dimension (1D, arrays, 3D,
        // cubes), not always 2D.
        View::Shader(_, Some(desc)) => return null_like(desc, at),
        View::Placed(false, _, desc) => {
            return null_like(desc[..10].try_into().unwrap_or(&[0; 10]), at);
        }
        // Read as raw buffers, as made (`make`).
        View::Constant(..) => return null_buffer_view(at),
        _ => return null_view(at),
    };
    let desc = if buffer {
        D3D12_UNORDERED_ACCESS_VIEW_DESC {
            Format: DXGI_FORMAT_R32_UINT,
            ViewDimension: D3D12_UAV_DIMENSION_BUFFER,
            ..Default::default()
        }
    } else {
        null_uav_desc(dimension)
    };
    // SAFETY: a slot of our heap; a null resource is allowed with a desc.
    unsafe {
        host.device.CreateUnorderedAccessView(
            None::<&ID3D12Resource>,
            None::<&ID3D12Resource>,
            Some(&desc),
            at,
        )
    };
}

/// A null texture UAV of the title's view dimension (1D, 1D or 2D arrays,
/// 3D), 2D if it names none: a 2D null view where the shader writes a 3D
/// texture is a mismatch GPU-based validation reports (the world's wind
/// field, CS_CopyWindFieldBeforeUpdateWithSmooth, 2026-11-24).
fn null_uav_desc(dimension: Option<u32>) -> D3D12_UNORDERED_ACCESS_VIEW_DESC {
    let mut desc = D3D12_UNORDERED_ACCESS_VIEW_DESC {
        Format: DXGI_FORMAT_R32_UINT,
        ViewDimension: D3D12_UAV_DIMENSION_TEXTURE2D,
        ..Default::default()
    };
    match dimension.map(|d| D3D12_UAV_DIMENSION(d as i32)) {
        Some(D3D12_UAV_DIMENSION_TEXTURE1D) => {
            desc.ViewDimension = D3D12_UAV_DIMENSION_TEXTURE1D;
        }
        Some(D3D12_UAV_DIMENSION_TEXTURE1DARRAY) => {
            desc.ViewDimension = D3D12_UAV_DIMENSION_TEXTURE1DARRAY;
            desc.Anonymous.Texture1DArray = D3D12_TEX1D_ARRAY_UAV {
                ArraySize: 1,
                ..Default::default()
            };
        }
        Some(D3D12_UAV_DIMENSION_TEXTURE2DARRAY) => {
            desc.ViewDimension = D3D12_UAV_DIMENSION_TEXTURE2DARRAY;
            desc.Anonymous.Texture2DArray = D3D12_TEX2D_ARRAY_UAV {
                ArraySize: 1,
                ..Default::default()
            };
        }
        Some(D3D12_UAV_DIMENSION_TEXTURE3D) => {
            desc.ViewDimension = D3D12_UAV_DIMENSION_TEXTURE3D;
            desc.Anonymous.Texture3D = D3D12_TEX3D_UAV {
                WSize: u32::MAX,
                ..Default::default()
            };
        }
        _ => {}
    }
    desc
}

/// A buffer view that reads zeros (raw, as translated shaders mostly read
/// buffers).
fn null_buffer_view(at: D3D12_CPU_DESCRIPTOR_HANDLE) {
    let Some(host) = host() else { return };
    let desc = D3D12_SHADER_RESOURCE_VIEW_DESC {
        Format: DXGI_FORMAT_R32_TYPELESS,
        ViewDimension: D3D12_SRV_DIMENSION_BUFFER,
        Shader4ComponentMapping: D3D12_DEFAULT_SHADER_4_COMPONENT_MAPPING,
        Anonymous: D3D12_SHADER_RESOURCE_VIEW_DESC_0 {
            Buffer: D3D12_BUFFER_SRV {
                NumElements: 1,
                Flags: D3D12_BUFFER_SRV_FLAG_RAW,
                ..Default::default()
            },
        },
    };
    // SAFETY: a slot of our heap; a null resource is allowed with a desc.
    unsafe {
        host.device
            .CreateShaderResourceView(None::<&ID3D12Resource>, Some(&desc), at)
    };
}

/// A texture view that reads zeros with the dimension (and mip and array
/// counts) of the title's description `words`; 2D if it names none.
fn null_like(words: &[u32; 10], at: D3D12_CPU_DESCRIPTOR_HANDLE) {
    const TEXTURES: std::ops::RangeInclusive<u32> = 2..=10;
    if !TEXTURES.contains(&words[1]) {
        return null_view(at);
    }
    let Some(host) = host() else { return };
    // One mip, one slice or cube, plane 0: fields a null view still checks.
    let mut w = [0u32; 10];
    w[1] = words[1];
    match words[1] {
        6 => {}
        7 => w[5] = 1,
        3 | 5 | 10 => (w[5], w[7]) = (1, 1),
        _ => w[5] = 1,
    }
    let mut desc = srv_desc(&w);
    desc.Format = DXGI_FORMAT_R8G8B8A8_UNORM;
    desc.Shader4ComponentMapping = D3D12_DEFAULT_SHADER_4_COMPONENT_MAPPING;
    // SAFETY: a slot of our heap; a null resource is allowed with a desc.
    unsafe {
        host.device
            .CreateShaderResourceView(None::<&ID3D12Resource>, Some(&desc), at)
    };
}

/// A view that reads zeros, for slots whose view the host cannot make.
/// shortcut: slots with no view at all (kind unknown) get this null 2D
/// view; null buffer views there did not stop the world's low-address
/// page faults (2026-11-24, run w10), so they are not the cause.
fn null_view(at: D3D12_CPU_DESCRIPTOR_HANDLE) {
    let Some(host) = host() else { return };
    let desc = D3D12_SHADER_RESOURCE_VIEW_DESC {
        Format: DXGI_FORMAT_R8G8B8A8_UNORM,
        ViewDimension: D3D12_SRV_DIMENSION_TEXTURE2D,
        Shader4ComponentMapping: D3D12_DEFAULT_SHADER_4_COMPONENT_MAPPING,
        Anonymous: D3D12_SHADER_RESOURCE_VIEW_DESC_0 {
            Texture2D: D3D12_TEX2D_SRV {
                MipLevels: 1,
                ..Default::default()
            },
        },
    };
    // SAFETY: a slot of our heap; a null resource is allowed with a desc.
    unsafe {
        host.device
            .CreateShaderResourceView(None::<&ID3D12Resource>, Some(&desc), at)
    };
}

#[cfg(test)]
mod tests {
    use super::*;

    thread_local! {
        /// Set in a test: heap syncs of its thread leave slots of foreign
        /// bytes as they are, as with `VICTORY_STALE_LEFT=1`.
        pub(crate) static STALE_LEFT: std::cell::Cell<bool> = const { std::cell::Cell::new(false) };
    }

    /// With `VICTORY_STALE_LEFT=1` a heap sync leaves a made slot the title
    /// wrote zeros or other bytes of its own to, and the record written back
    /// costs nothing; a draw reading the slot makes it null, and a record of
    /// ours naming no view is still made null by the sync. Off, the sync
    /// makes the slot null. Both the fast and the legacy sync.
    #[test]
    fn syncs_leave_slots_the_title_wrote_foreign_bytes_to() {
        const SLOTS: u32 = 2_048;
        let srv_desc = [62u32, 1, 0x1688, 0, 0, 0, 256, 0, 0, 0];
        let buffer = crate::resource::test_buffer(256);
        let (ptr, len) = crate::resource::buffer_bytes(buffer).expect("buffer bytes");
        crate::shadow::register(buffer as usize, ptr as u64, len);
        let mut record = [0u32; 3];
        let at = record.as_mut_ptr() as u64;
        crate::descriptor::create_srv(std::ptr::null(), buffer, &srv_desc, at);
        let write = |mem: &mut [u8], slot: usize, words: [u32; 3]| {
            for (k, w) in words.iter().enumerate() {
                mem[slot * 32 + 4 * k..slot * 32 + 4 * k + 4].copy_from_slice(&w.to_le_bytes());
            }
        };
        for fast in [false, true] {
            let Some(host) = HostHeap::new(false, SLOTS) else {
                break; // no host GPU here
            };
            let mut mem = vec![0u8; SLOTS as usize * 32];
            for slot in 0..4 {
                write(&mut mem, slot, record);
            }
            let sync = |mem: &[u8]| {
                let start = mem.as_ptr() as u64;
                host.sync_slots_bytes_with(0..SLOTS, start, 32, start, mem, fast);
            };
            sync(&mem);
            assert!((0..4).all(|i| host.made(i)), "fast {fast}: made from the records");
            STALE_LEFT.with(|left| left.set(true));
            write(&mut mem, 0, [0, 0, 0]);
            write(&mut mem, 1, [0xdead_beef, 0x1234_5678, 7]);
            write(&mut mem, 2, [VIEW_TAG, 0, 0]);
            sync(&mem);
            assert!(host.made(0), "fast {fast}: zeros are left");
            assert!(host.made(1), "fast {fast}: other bytes are left");
            assert!(!host.made(2), "fast {fast}: a record of ours naming no view is made null");
            assert!(host.made(3), "fast {fast}: an unchanged record stays");
            // The record written back: the slot is still made from it.
            write(&mut mem, 0, record);
            sync(&mem);
            assert!(host.made(0), "fast {fast}: written back");
            // A draw reading a left slot makes it null.
            let handle = mem.as_ptr() as u64 + 32;
            assert!(host.mirror(1, handle, true).is_none());
            assert!(!host.made(1), "fast {fast}: a draw makes the left slot null");
            STALE_LEFT.with(|left| left.set(false));
            write(&mut mem, 3, [0, 0, 0]);
            sync(&mem);
            assert!(!host.made(3), "fast {fast}: off, zeros are made null");
        }
        crate::object::release(buffer);
    }

    /// Typed buffer views of integer formats carry their kind in the
    /// view-stride table; float, normalized, raw and structured ones none.
    #[test]
    fn integer_typed_buffer_views_are_marked_in_the_stride_table() {
        use victory_shader::emit::{VIEW_INTEGER, VIEW_SIGNED};
        // Format, dimension (buffer), first element, count, stride, counter, flags.
        let uav = |format, stride, flags| View::Unordered(0, 0, Some([format, 1, 0, 0, 64, stride, 0, 0, flags, 0]));
        assert_eq!(view_kind(&uav(12, 0, 0)), VIEW_INTEGER, "R16G16B16A16_UINT");
        assert_eq!(view_kind(&uav(7, 0, 0)), VIEW_INTEGER, "R32G32B32_UINT");
        assert_eq!(view_kind(&uav(8, 0, 0)), VIEW_INTEGER | VIEW_SIGNED, "R32G32B32_SINT");
        assert_eq!(view_kind(&uav(57, 0, 0)), VIEW_INTEGER, "R16_UINT");
        assert_eq!(view_kind(&uav(59, 0, 0)), VIEW_INTEGER | VIEW_SIGNED, "R16_SINT");
        assert_eq!(view_kind(&uav(64, 0, 0)), VIEW_INTEGER | VIEW_SIGNED, "R8_SINT");
        assert_eq!(view_kind(&uav(10, 0, 0)), 0, "R16G16B16A16_FLOAT");
        assert_eq!(view_kind(&uav(58, 0, 0)), 0, "R16_SNORM");
        assert_eq!(view_kind(&uav(39, 0, 1)), 0, "raw");
        assert_eq!(view_kind(&uav(0, 16, 0)), 0, "structured");
        // Format, dimension, mapping, first element, count, stride, flags.
        let srv = View::Shader(0, Some([62, 1, 5768, 0, 0, 0, 64, 0, 0, 0]));
        assert_eq!(view_kind(&srv), VIEW_INTEGER, "R8_UINT");
        assert_eq!(view_stride(&uav(12, 0, 0)) | view_kind(&uav(12, 0, 0)), 8 | VIEW_INTEGER);
        assert_eq!(widened_format(6), Some(DXGI_FORMAT_R32G32B32A32_FLOAT));
        assert_eq!(widened_format(7), Some(DXGI_FORMAT_R32G32B32A32_UINT));
        assert_eq!(widened_format(8), Some(DXGI_FORMAT_R32G32B32A32_SINT));
        assert_eq!(widened_format(2), None);
        assert_eq!(widened_w(6), Some(0x3f80_0000));
        assert_eq!(widened_w(7), Some(1));
        assert_eq!(widened_w(8), Some(1));
        assert_eq!(widened_w(2), None);
    }

    /// Three-channel 32-bit SRVs are backed by a widened copy on hosts that
    /// read R32G32B32 as R32G32. The copy keeps 12-byte source elements,
    /// supplies w in the source format and follows later title writes.
    #[test]
    fn three_channel_srvs_use_a_widened_refreshed_copy() {
        if !crate::under_wine() {
            return;
        }
        let Some(host) = host() else { return };
        let p = crate::watch::tests::pool(1);
        let len = 4 * 12;
        let first: [u32; 12] = [
            1000, 2000, 3000, 1003, 2003, 3003, 1006, 2006, 3006, 1009, 2009, 3009,
        ];
        // SAFETY: 48 bytes of our pool page.
        unsafe { std::ptr::copy_nonoverlapping(first.as_ptr().cast::<u8>(), p as *mut u8, len) };
        crate::shadow::register(p, p as u64, len as u64);
        let shadow = crate::shadow::find(p as u64, len as u64).expect("a host copy");
        shadow.update(p, len);

        // SAFETY: plain descriptor-heap creation.
        let heap: ID3D12DescriptorHeap = unsafe {
            host.device.CreateDescriptorHeap(&D3D12_DESCRIPTOR_HEAP_DESC {
                Type: D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV,
                NumDescriptors: 4,
                ..Default::default()
            })
        }
        .expect("a descriptor heap");
        // SAFETY: a live heap.
        let at = unsafe { heap.GetCPUDescriptorHandleForHeapStart() };
        let desc = D3D12_SHADER_RESOURCE_VIEW_DESC {
            Format: DXGI_FORMAT_R32G32B32_FLOAT,
            ViewDimension: D3D12_SRV_DIMENSION_BUFFER,
            Shader4ComponentMapping: D3D12_DEFAULT_SHADER_4_COMPONENT_MAPPING,
            Anonymous: D3D12_SHADER_RESOURCE_VIEW_DESC_0 {
                Buffer: D3D12_BUFFER_SRV {
                    NumElements: 4,
                    ..Default::default()
                },
            },
        };
        let (resource, source) = buffer_srv(p as u64, len as u64, desc, at)
            .expect("a view")
            .expect("a kept view");
        let Some(HeldSource::Widened(widened, _)) = source else {
            panic!("a widened source");
        };
        // SAFETY: a live resource.
        let made = unsafe { resource.GetDesc() };
        assert_eq!(made.Width, 4 * 16);
        let float_one = 0x3f80_0000;
        let want = |w| {
            [
                1000, 2000, 3000, w, 1003, 2003, 3003, w, 1006, 2006, 3006, w, 1009, 2009, 3009, w,
            ]
        };
        assert_eq!(widened.read_host(16), want(float_one));

        // SAFETY: a live heap; each later slot is one descriptor stride on.
        let step = unsafe {
            host.device
                .GetDescriptorHandleIncrementSize(D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV)
        };
        let at_slot = |slot: usize| D3D12_CPU_DESCRIPTOR_HANDLE {
            ptr: at.ptr + slot * step as usize,
        };
        let (second_resource, second_source) =
            buffer_srv(p as u64, len as u64, desc, at_slot(1))
                .expect("a second view")
                .expect("a second kept view");
        let Some(HeldSource::Widened(second, _)) = second_source else {
            panic!("a second widened source");
        };
        assert!(
            std::sync::Arc::ptr_eq(&widened, &second),
            "one widened copy per range and w"
        );
        drop((second_resource, second));

        let uint_desc = D3D12_SHADER_RESOURCE_VIEW_DESC {
            Format: DXGI_FORMAT_R32G32B32_UINT,
            ..desc
        };
        let (uint_resource, uint_source) = buffer_srv(p as u64, len as u64, uint_desc, at_slot(2))
            .expect("a uint view")
            .expect("a kept uint view");
        let Some(HeldSource::Widened(uint, _)) = uint_source else {
            panic!("a widened uint source");
        };
        let sint_desc = D3D12_SHADER_RESOURCE_VIEW_DESC {
            Format: DXGI_FORMAT_R32G32B32_SINT,
            ..desc
        };
        let (sint_resource, sint_source) = buffer_srv(p as u64, len as u64, sint_desc, at_slot(3))
            .expect("a sint view")
            .expect("a kept sint view");
        let Some(HeldSource::Widened(sint, _)) = sint_source else {
            panic!("a widened sint source");
        };
        assert!(
            std::sync::Arc::ptr_eq(&uint, &sint),
            "integer formats share their widened copy"
        );
        assert!(
            !std::sync::Arc::ptr_eq(&widened, &uint),
            "float and integer w bits need distinct copies"
        );
        assert_eq!(uint.read_host(16), want(1));
        assert_eq!(sint.read_host(16), want(1));

        let changed = [0x1111_1111, 0x2222_2222, 0x3333_3333];
        // SAFETY: one 12-byte element of our pool page.
        unsafe { std::ptr::copy_nonoverlapping(changed.as_ptr().cast::<u8>(), (p + 12) as *mut u8, 12) };
        assert!(shadow.update(p, len) > 0, "the title write is copied");
        let changed_want = |w| {
            [
                1000, 2000, 3000, w, 0x1111_1111, 0x2222_2222, 0x3333_3333, w, 1006, 2006, 3006, w,
                1009, 2009, 3009, w,
            ]
        };
        assert_eq!(widened.read_host(16), changed_want(float_one));
        assert_eq!(uint.read_host(16), changed_want(1));
        assert_eq!(sint.read_host(16), changed_want(1));

        drop((
            resource,
            widened,
            uint_resource,
            uint,
            sint_resource,
            sint,
            heap,
            shadow,
        ));
        crate::shadow::unregister(p, p as u64);
        crate::watch::tests::free(p);
    }

    /// A placed texture's view the title writes into a heap is left null
    /// and noted once: heap syncs (every submission) pass it by, a draw
    /// reading it through a table makes it.
    #[test]
    fn a_placed_texture_slot_left_null_is_made_once_a_draw_reads_it() {
        if host().is_none() {
            return; // no host GPU here
        }
        let null = std::ptr::null();
        let heap_desc: [u32; 8] = [0, 4, 1, 0, 0, 0, 0, 0];
        let mut heap = null;
        let iid = crate::object::guid(0x8efb471d, 0x616c, 0x4f49, [0x90, 0xf7, 0x12, 0x7b, 0xb7, 0x63, 0xfa, 0x51]);
        assert_eq!(crate::descriptor::create_heap(null, heap_desc.as_ptr().cast(), &iid, &mut heap), 0);
        let title_heap = crate::descriptor::heap(heap);
        // A 4x1 RGBA8 texture, linear, in memory of its own, and a 2D view.
        let memory = vec![0u8; 1 << 16];
        let address = memory.as_ptr() as u64;
        let words: [u32; 14] = [3, 0, 0, 0, 4, 0, 1, 1 | 1 << 16, 28, 1, 0, 0x100, 0, 0];
        let mut view = [0u32; 16];
        view[..6].copy_from_slice(&[28, 4, 0x1688, 0, 0, 1]);
        (view[10], view[11]) = (address as u32, (address >> 32) as u32);
        crate::descriptor::create_placed_srv(null, &words, &view, title_heap.start);
        let host = title_heap.host().expect("a host view heap");
        for _ in 0..2 {
            assert!(host.mirror(0, title_heap.start, false).is_some());
            assert!(host.made(0), "the record is noted");
            assert!(!host.kept(0), "no host texture for it yet");
        }
        assert!(host.mirror(0, title_heap.start, true).is_some());
        assert!(host.kept(0), "made for the draw");
    }

    /// A slot viewing a texture given up (`crate::resource::evict`) holds it
    /// no more: heap syncs pass it by, a draw reading it through a table
    /// makes it again.
    #[test]
    fn a_slot_viewing_a_texture_given_up_is_made_again_when_read() {
        if host().is_none() {
            return; // no host GPU here
        }
        let null = std::ptr::null();
        let heap_desc: [u32; 8] = [0, 4, 1, 0, 0, 0, 0, 0];
        let mut heap = null;
        let iid = crate::object::guid(0x8efb471d, 0x616c, 0x4f49, [0x90, 0xf7, 0x12, 0x7b, 0xb7, 0x63, 0xfa, 0x51]);
        assert_eq!(crate::descriptor::create_heap(null, heap_desc.as_ptr().cast(), &iid, &mut heap), 0);
        let title_heap = crate::descriptor::heap(heap);
        // A 4x1 RGBA8 texture, linear, in memory of its own, and a 2D view.
        let memory = vec![0u8; 1 << 16];
        let address = memory.as_ptr() as u64;
        let words: [u32; 14] = [3, 0, 0, 0, 4, 0, 1, 1 | 1 << 16, 28, 1, 0, 0x100, 0, 0];
        let mut view = [0u32; 16];
        view[..6].copy_from_slice(&[28, 4, 0x1688, 0, 0, 1]);
        (view[10], view[11]) = (address as u32, (address >> 32) as u32);
        crate::descriptor::create_placed_srv(null, &words, &view, title_heap.start);
        let host = title_heap.host().expect("a host view heap");
        assert!(host.mirror(0, title_heap.start, true).is_some());
        assert!(host.kept(0), "made for the draw");
        let texture = crate::resource::texture_at(address, &words).expect("its resource");
        assert!(crate::resource::give_up_now(texture));
        assert!(!host.kept(0), "the slot holds it no more");
        assert!(host.made(0), "its record stays noted");
        assert!(host.mirror(0, title_heap.start, false).is_some());
        assert!(!host.kept(0), "heap syncs pass it by");
        assert!(host.mirror(0, title_heap.start, true).is_some());
        assert!(host.kept(0), "made again for the draw");
        assert!(crate::resource::made_copy(texture).is_some(), "with a host texture again");
    }

    std::thread_local! {
        /// Work to run while a view's texture is prepared on this thread,
        /// once (tests). Each test thread has its own: tests run at once,
        /// and every test's heap syncs and mirrors pass through `prepare`.
        /// One hook for the process ran a test's work on whichever thread
        /// prepared a view first (the bench's syncs, thousands a second),
        /// so the test went on without waiting for it.
        static WHILE_PREPARING: std::cell::RefCell<Option<crate::descriptor::tests::Work>> =
            const { std::cell::RefCell::new(None) };
    }

    /// Sets the work the next `prepare` on this thread runs.
    fn when_preparing(work: impl FnOnce() + Send + 'static) {
        WHILE_PREPARING.set(Some(Box::new(work)));
    }

    /// Whether this thread's `prepare` ran the work set (or none was set).
    fn prepared() -> bool {
        WHILE_PREPARING.with_borrow(Option::is_none)
    }

    pub(crate) fn while_preparing() {
        if let Some(work) = WHILE_PREPARING.take() {
            work();
        }
    }

    /// A resource's final release during preparation returns at once, and the
    /// resource stays alive for the make: the view's pin frees it when the
    /// caller drops the view.
    #[test]
    fn a_resource_released_while_its_view_is_prepared_outlives_the_make() {
        if host().is_none() {
            return; // no host GPU here
        }
        let resource = crate::resource::test_buffer(256);
        let mut record = [0u32; 12];
        let at = record.as_mut_ptr() as u64;
        // R8_UINT buffer SRV, all 256 elements.
        let desc = [62, 1, 0x1688, 0, 0, 256, 0, 0, 0, 0];
        crate::descriptor::create_srv(std::ptr::null(), resource, &desc, at);
        let Some(heap) = HostHeap::new(false, 1) else {
            crate::object::slot::<extern "system" fn(crate::object::This) -> u32>(resource, 0x10)(
                resource,
            );
            return;
        };

        let address = resource as usize;
        let release = crate::object::slot::<extern "system" fn(crate::object::This) -> u32>(
            resource,
            0x10,
        );
        when_preparing(move || {
            // The title's last reference goes while the view is being made.
            // The call must not wait for the make: this very thread is the
            // one making the view, so a wait would never end.
            let released = std::thread::spawn(move || release(address as crate::object::This));
            let deadline = std::time::Instant::now() + std::time::Duration::from_secs(5);
            while !released.is_finished() {
                assert!(
                    std::time::Instant::now() < deadline,
                    "the title's release waited for the view's make"
                );
                std::thread::yield_now();
            }
            assert_eq!(released.join().unwrap(), 1, "the pin still holds it");
        });

        let view = heap.mirror(0, at, false).expect("the current view");
        assert!(prepared(), "the prepare ran");
        assert!(heap.made(0), "the view was made from the live resource");
        assert!(
            crate::descriptor::view(at).is_some(),
            "still current while the caller holds the view"
        );
        // The last pin's drop frees the resource and makes its views stale.
        drop(view);
        assert!(crate::descriptor::view(at).is_none(), "stale once it is freed");
    }

    /// A record the title writes into a slot while another thread prepares
    /// the slot's old record is the one the slot ends up viewing.
    #[test]
    fn a_slot_rewritten_while_its_view_is_prepared_views_the_new_record() {
        if host().is_none() {
            return; // no host GPU here
        }
        let null = std::ptr::null();
        let heap_desc: [u32; 8] = [0, 1024, 1, 0, 0, 0, 0, 0];
        let mut heap = null;
        let iid = crate::object::guid(0x8efb471d, 0x616c, 0x4f49, [0x90, 0xf7, 0x12, 0x7b, 0xb7, 0x63, 0xfa, 0x51]);
        assert_eq!(crate::descriptor::create_heap(null, heap_desc.as_ptr().cast(), &iid, &mut heap), 0);
        let title_heap = crate::descriptor::heap(heap);
        let s = 960u32;
        let at = title_heap.start + u64::from(s) * 32;
        let memory = vec![0u8; 1 << 17];
        let words: [u32; 14] = [3, 0, 0, 0, 4, 0, 1, 1 | 1 << 16, 28, 1, 0, 0x100, 0, 0];
        let view_of = |address: u64| {
            let mut view = [0u32; 16];
            view[..6].copy_from_slice(&[28, 4, 0x1688, 0, 0, 1]);
            (view[10], view[11]) = (address as u32, (address >> 32) as u32);
            view
        };
        let (first, second) = (memory.as_ptr() as u64, memory.as_ptr() as u64 + (1 << 16));
        crate::descriptor::create_placed_srv(null, &words, &view_of(first), at);
        let host = title_heap.host().expect("a host view heap");
        let (done, waited) = std::sync::mpsc::channel();
        let second_view = view_of(second);
        when_preparing(move || {
            // The title writes the other record into the slot meanwhile.
            std::thread::spawn(move || {
                crate::descriptor::create_placed_srv(std::ptr::null(), &words, &second_view, at);
                done.send(()).unwrap();
            });
            assert!(waited.recv_timeout(std::time::Duration::from_secs(5)).is_ok(), "the write waits");
        });
        // A draw reads the slot: its first record is prepared.
        assert!(host.mirror(s, at, true).is_some());
        let second_texture = crate::resource::texture_at(second, &words).expect("its resource");
        let raw = windows::core::Interface::as_raw(&crate::resource::made_copy(second_texture).expect("made").texture) as usize;
        assert_eq!(host.textures_at(&[s]), [raw], "the slot views the record written last");
    }

    /// What a view needs is made without the slot lock: another thread
    /// binding a view of the same heap meanwhile is not held up.
    #[test]
    fn views_bound_while_another_slot_gets_its_texture_do_not_wait() {
        if host().is_none() {
            return; // no host GPU here
        }
        let null = std::ptr::null();
        let heap_desc: [u32; 8] = [0, 1024, 1, 0, 0, 0, 0, 0];
        let mut heap = null;
        let iid = crate::object::guid(0x8efb471d, 0x616c, 0x4f49, [0x90, 0xf7, 0x12, 0x7b, 0xb7, 0x63, 0xfa, 0x51]);
        assert_eq!(crate::descriptor::create_heap(null, heap_desc.as_ptr().cast(), &iid, &mut heap), 0);
        let title_heap = crate::descriptor::heap(heap);
        let (first, second) = (950u32, 951u32);
        let memory = vec![0u8; 1 << 17];
        let words: [u32; 14] = [3, 0, 0, 0, 4, 0, 1, 1 | 1 << 16, 28, 1, 0, 0x100, 0, 0];
        let slot = |i: u32| title_heap.start + u64::from(i) * 32;
        for (i, at) in [(first, 0u64), (second, 1 << 16)] {
            let address = memory.as_ptr() as u64 + at;
            let mut view = [0u32; 16];
            view[..6].copy_from_slice(&[28, 4, 0x1688, 0, 0, 1]);
            (view[10], view[11]) = (address as u32, (address >> 32) as u32);
            crate::descriptor::create_placed_srv(null, &words, &view, slot(i));
        }
        let host = title_heap.host().expect("a host view heap");
        let heap_at = title_heap.start;
        let heap = heap as usize;
        let (done, waited) = std::sync::mpsc::channel();
        when_preparing(move || {
            std::thread::spawn(move || {
                let h = crate::descriptor::heap(heap as crate::object::This);
                h.host().unwrap().mirror(second, heap_at + u64::from(second) * 32, true);
                done.send(()).unwrap();
            });
            assert!(
                waited.recv_timeout(std::time::Duration::from_secs(5)).is_ok(),
                "a view bound while another slot's texture is made waits"
            );
        });
        assert!(host.mirror(first, slot(first), true).is_some());
        assert!(prepared(), "the prepare ran");
        assert!(host.kept(first) && host.kept(second), "both made");
    }

    /// The textures of slots shaders read are made without the heaps' lock:
    /// a title thread writing a view meanwhile is not held up by the make.
    #[test]
    fn views_written_while_read_slots_get_their_textures_do_not_wait() {
        if host().is_none() {
            return; // no host GPU here
        }
        let null = std::ptr::null();
        let heap_desc: [u32; 8] = [0, 1024, 1, 0, 0, 0, 0, 0];
        let mut heap = null;
        let iid = crate::object::guid(0x8efb471d, 0x616c, 0x4f49, [0x90, 0xf7, 0x12, 0x7b, 0xb7, 0x63, 0xfa, 0x51]);
        assert_eq!(crate::descriptor::create_heap(null, heap_desc.as_ptr().cast(), &iid, &mut heap), 0);
        let title_heap = crate::descriptor::heap(heap);
        let (read, written) = (900u32, 901u32);
        // A 4x1 RGBA8 texture, linear, in memory of its own, and a 2D view.
        let memory = vec![0u8; 1 << 16];
        let address = memory.as_ptr() as u64;
        let words: [u32; 14] = [3, 0, 0, 0, 4, 0, 1, 1 | 1 << 16, 28, 1, 0, 0x100, 0, 0];
        let mut view = [0u32; 16];
        view[..6].copy_from_slice(&[28, 4, 0x1688, 0, 0, 1]);
        (view[10], view[11]) = (address as u32, (address >> 32) as u32);
        let slot = |i: u32| title_heap.start + u64::from(i) * 32;
        crate::descriptor::create_placed_srv(null, &words, &view, slot(read));
        let host = title_heap.host().expect("a host view heap");
        assert!(!host.kept(read), "left null until read");
        let (done, waited) = std::sync::mpsc::channel();
        let at = slot(written);
        *crate::descriptor::tests::WHILE_MAKING.lock().unwrap() = Some((
            read,
            Box::new(move || {
                std::thread::spawn(move || {
                    crate::descriptor::create_placed_srv(std::ptr::null(), &words, &view, at);
                    done.send(()).unwrap();
                });
                assert!(
                    waited.recv_timeout(std::time::Duration::from_secs(5)).is_ok(),
                    "a view written while a read slot's texture is made waits"
                );
            }),
        ));
        crate::descriptor::make_read(&[read]);
        assert!(crate::descriptor::tests::WHILE_MAKING.lock().unwrap().is_none(), "the make ran");
        assert!(host.kept(read), "made for the read");
    }

    /// A heap sync makes the slots whose records the title changed as bytes
    /// (a copy of another slot's record, a record cleared) and passes the
    /// others by.
    #[test]
    fn heap_syncs_make_the_slots_whose_records_changed() {
        if host().is_none() {
            return; // no host GPU here
        }
        let null = std::ptr::null();
        let heap_desc: [u32; 8] = [0, 4, 1, 0, 0, 0, 0, 0];
        let mut heap = null;
        let iid = crate::object::guid(0x8efb471d, 0x616c, 0x4f49, [0x90, 0xf7, 0x12, 0x7b, 0xb7, 0x63, 0xfa, 0x51]);
        assert_eq!(crate::descriptor::create_heap(null, heap_desc.as_ptr().cast(), &iid, &mut heap), 0);
        let title_heap = crate::descriptor::heap(heap);
        let start = title_heap.start;
        // A view of no memory: a view of the test's own heap memory would get
        // a host copy whose 4 MiB window the guards may cover (the process
        // heap guarded read-only hung the suite).
        crate::descriptor::create_cbv(null, &[0, 0, 256, 0], start);
        let host = title_heap.host().expect("a host view heap");
        // SAFETY: slots 0 and 1 of the heap's memory, 32 bytes each.
        unsafe { std::ptr::copy_nonoverlapping(start as *const u8, (start + 32) as *mut u8, 32) };
        assert!(!host.made(1), "copied as bytes: not seen yet");
        host.sync_slots(0, 4, start, 32);
        assert!(host.made(0) && host.made(1) && !host.made(2));
        // SAFETY: as above.
        unsafe { std::ptr::write_bytes((start + 32) as *mut u8, 0, 32) };
        host.sync_slots(0, 4, start, 32);
        assert!(host.made(0) && !host.made(1), "a cleared record leaves its slot");
    }

    /// The world's 16-byte view of 0 elements removed the device.
    #[test]
    fn empty_and_overlong_buffer_views_are_refused() {
        assert!(within(64, 0, 4, 16));
        assert!(within(64, 3, 1, 16));
        assert!(!within(64, 0, 0, 16));
        assert!(!within(64, 3, 2, 16));
        assert!(!within(64, u64::MAX, 1, 16));
    }

    /// A null texture UAV keeps the title's dimension (3D for the world's
    /// wind field), and is 2D when the title names none.
    #[test]
    fn null_uavs_keep_the_titles_dimension() {
        let d = null_uav_desc(Some(8));
        assert_eq!(d.ViewDimension, D3D12_UAV_DIMENSION_TEXTURE3D);
        // SAFETY: the member the dimension names.
        assert_eq!(unsafe { d.Anonymous.Texture3D.WSize }, u32::MAX);
        let d = null_uav_desc(Some(5));
        assert_eq!(d.ViewDimension, D3D12_UAV_DIMENSION_TEXTURE2DARRAY);
        // SAFETY: the member the dimension names.
        assert_eq!(unsafe { d.Anonymous.Texture2DArray.ArraySize }, 1);
        let dimension = |d| null_uav_desc(d).ViewDimension;
        assert_eq!(dimension(Some(2)), D3D12_UAV_DIMENSION_TEXTURE1D);
        assert_eq!(dimension(Some(4)), D3D12_UAV_DIMENSION_TEXTURE2D);
        assert_eq!(dimension(None), D3D12_UAV_DIMENSION_TEXTURE2D);
        // Multisampled UAVs need a cap; they stay 2D.
        assert_eq!(dimension(Some(6)), D3D12_UAV_DIMENSION_TEXTURE2D);
    }

    /// A null 3D UAV made on the real device leaves it alive.
    #[test]
    fn null_3d_uav_is_accepted_by_the_device() {
        let Some(host) = host() else {
            return; // no host GPU
        };
        let Some(heap) = HostHeap::new(false, 4) else {
            return;
        };
        let view = View::Unordered(0, 0, Some([28, 8, 0, 0, 0, 0, 0, 0, 0, 0]));
        null_for(&view, heap.cpu(0));
        // SAFETY: a plain device query.
        assert!(unsafe { host.device.GetDeviceRemovedReason() }.is_ok());
    }

    /// A table made afresh lies in dynamic slots, and a shader reading a
    /// view's address (e.g. the BVH builder's base) indexes the view
    /// addresses there: they have an entry for each dynamic slot, entered
    /// whenever the slot is made (a view, null, or a raw view of ours).
    /// Giving a slot back leaves its entry: every table made afresh makes
    /// all its slots. (Hosts whose view heaps are smaller than the title's,
    /// NVIDIA, make such tables; the shader read past the buffer.)
    ///
    /// Any heap size the host takes will do: a title heap of 4 slots leaves
    /// dynamic slots past it either way ([`DYNAMIC`] or [`MIN_DYNAMIC`]).
    #[test]
    fn view_addresses_cover_tables_made_afresh() {
        let Some(heap) = HostHeap::new(false, 4) else {
            return; // no host GPU
        };
        let end = heap.count() + heap.dynamic;
        assert!(heap.view_addresses().is_some());
        let first = heap.dynamic(2).unwrap();
        assert!(first >= heap.count() && first + 1 < end);
        // A buffer the title placed at 0xd_6013_e000: 16 elements of 4 bytes.
        let address = 0xd_6013_e000u64;
        let mut resource = [0u32; 14];
        (resource[0], resource[4]) = (1, 64);
        let mut desc = [0u32; 16];
        (desc[1], desc[2], desc[6], desc[7]) = (1, D3D12_DEFAULT_SHADER_4_COMPONENT_MAPPING, 16, 4);
        (desc[10], desc[11]) = (address as u32, (address >> 32) as u32);
        let view = View::Placed(false, resource, desc);
        assert_eq!(buffer_range(&view), Some((address, 64)));
        assert_eq!(heap.address_at(first), Some(0), "not made yet");
        heap.make_at(first, &view);
        assert_eq!(heap.address_at(first), Some(address));
        assert_eq!(heap.address_at(first + 1), Some(0), "not made yet");
        // The last dynamic slot has its entry too.
        heap.make_at(end - 1, &view);
        assert_eq!(heap.address_at(end - 1), Some(address));
        // A slot given back keeps its entry until it is made again; made
        // null, it has its entry cleared.
        heap.release(heap.id(), first, 2);
        assert_eq!(
            heap.address_at(first),
            Some(address),
            "given back, not made"
        );
        heap.make_null(first);
        assert_eq!(heap.address_at(first), Some(0));
        assert_eq!(heap.address_at(end - 1), Some(address), "still made");
        // Raw views of ours (root views, host copies' view pairs) in dynamic
        // slots clear what an earlier table entered there.
        heap.make_at(first, &view);
        heap.raw_view_at(first, None, 0, 0, false);
        assert_eq!(heap.address_at(first), Some(0), "a root view");
        heap.make_at(first, &view);
        heap.make_at(first + 1, &view);
        heap.raw_pair_at(first, None, 0, 0);
        assert_eq!(heap.address_at(first), Some(0), "a view pair");
        assert_eq!(heap.address_at(first + 1), Some(0), "a view pair");
        // Slots past the dynamic ones (host copies' view pairs) have no
        // entry, and nothing is written past the last: a mark there (the
        // buffer's allocation reaches past it) stays.
        assert_eq!(heap.address_at(end), None);
        let host = host().unwrap();
        let (buffer, data, _) = heap.addresses.get().unwrap().as_ref().unwrap();
        // SAFETY: plain queries of a live buffer.
        let (desc, size) = unsafe {
            let desc = buffer.GetDesc();
            let info = host.device.GetResourceAllocationInfo(0, &[desc]);
            (desc, info.SizeInBytes)
        };
        assert_eq!(desc.Width, 16 * u64::from(end));
        assert!(size >= desc.Width + 8, "room for the mark");
        let past = (data + 16 * end as usize) as *mut u64;
        let mark = 0x5a5a_5a5a_5a5a_5a5a;
        // SAFETY: mapped memory of the buffer's allocation (checked).
        unsafe { past.write_unaligned(mark) };
        heap.note_address(end, Some(&view));
        // SAFETY: as above.
        assert_eq!(unsafe { past.read_unaligned() }, mark);
        // SAFETY: a plain device query.
        assert!(unsafe { host.device.GetDeviceRemovedReason() }.is_ok());
    }

    /// Dynamic slots stay taken until given back, and runs do not wrap.
    #[test]
    fn dynamic_slots_are_held_until_released() {
        let Some(heap) = HostHeap::new(false, 4) else {
            return; // no host GPU
        };
        // Every index a translated shader can reach exists (clamped below
        // `HEAP_VIEWS`, `HEAP_SAMPLERS`).
        // SAFETY: plain queries of live heaps.
        unsafe {
            assert!(heap.heap.GetDesc().NumDescriptors >= victory_shader::emit::HEAP_VIEWS);
            let samplers = HostHeap::new(true, 128).unwrap();
            assert!(samplers.heap.GetDesc().NumDescriptors >= victory_shader::emit::HEAP_SAMPLERS);
        }
        // A slot that stops viewing a resource leaves it alive until the GPU
        // work in flight has run (a read of a freed host texture otherwise).
        // SAFETY: a plain committed buffer; one reference taken and given back.
        unsafe {
            let host = host().unwrap();
            let mut buffer: Option<ID3D12Resource> = None;
            host.device
                .CreateCommittedResource(
                    &D3D12_HEAP_PROPERTIES { Type: D3D12_HEAP_TYPE_DEFAULT, ..Default::default() },
                    D3D12_HEAP_FLAG_NONE,
                    &D3D12_RESOURCE_DESC {
                        Dimension: D3D12_RESOURCE_DIMENSION_BUFFER,
                        Width: 256,
                        Height: 1,
                        DepthOrArraySize: 1,
                        MipLevels: 1,
                        SampleDesc: DXGI_SAMPLE_DESC { Count: 1, Quality: 0 },
                        Layout: D3D12_TEXTURE_LAYOUT_ROW_MAJOR,
                        ..Default::default()
                    },
                    D3D12_RESOURCE_STATE_COMMON,
                    None,
                    &mut buffer,
                )
                .unwrap();
            let buffer = buffer.unwrap();
            let references = |b: &ID3D12Resource| {
                let raw = windows::core::Interface::as_raw(b);
                let unknown = &*(raw as *const *const windows::core::IUnknown_Vtbl);
                ((**unknown).AddRef)(raw);
                ((**unknown).Release)(raw)
            };
            let slot = heap.count + 2;
            heap.raw_view_at(slot, Some(&buffer), 0, 256, false);
            assert_eq!(references(&buffer), 2, "ours and the slot's");
            heap.make_null(slot);
            assert!(references(&buffer) >= 2, "the slot's, retired");
            crate::host::release_retired();
            // Released on a thread of its own.
            let since = std::time::Instant::now();
            while references(&buffer) > 1 && since.elapsed() < std::time::Duration::from_secs(5) {
                std::thread::yield_now();
            }
            assert_eq!(references(&buffer), 1, "ours");
        }
        let a = heap.dynamic(heap.dynamic - 100).unwrap();
        let b = heap.dynamic(60).unwrap();
        assert_eq!((a, b), (4, 4 + heap.dynamic - 100));
        // 40 left at the end: 50 do not fit until something is given back.
        assert_eq!(heap.dynamic(50), None);
        heap.release(heap.id(), a, 50);
        assert_eq!(heap.dynamic(50), Some(4));
        assert_eq!(heap.dynamic(40), Some(4 + heap.dynamic - 40));
    }

    #[test]
    fn free_runs_are_found_across_words() {
        // Slots 0-69, 100-109 and 201-255 taken: free runs 70-99, 110-200
        // and 256-300 (the last one past a word's end, `to` at 301).
        let mut taken = vec![0u64; 5];
        set_taken(&mut taken, 0, 70, true);
        set_taken(&mut taken, 100, 110, true);
        set_taken(&mut taken, 201, 256, true);
        assert_eq!(free_run(&taken, 0, 301, 30), Some(70));
        assert_eq!(free_run(&taken, 0, 301, 31), Some(110));
        assert_eq!(free_run(&taken, 0, 301, 91), Some(110));
        assert_eq!(free_run(&taken, 0, 301, 92), None);
        assert_eq!(free_run(&taken, 0, 200, 91), None, "a run ends before `to`");
        assert_eq!(free_run(&taken, 0, 201, 91), Some(110));
        assert_eq!(free_run(&taken, 120, 301, 81), Some(120));
        assert_eq!(free_run(&taken, 120, 301, 82), None);
        // Every other slot taken: no two clear in a row.
        let alternate = [0x5555_5555_5555_5555u64; 2];
        assert_eq!(free_run(&alternate, 0, 128, 2), None);
        assert_eq!(free_run(&alternate, 0, 128, 1), Some(1));
        assert_eq!(free_run(&alternate, 3, 128, 1), Some(3));
        // A word's last (or first) slot taken ends the run there.
        assert_eq!(free_run(&[1 << 63, 0], 0, 128, 63), Some(0));
        assert_eq!(free_run(&[1 << 63, 0], 0, 128, 64), Some(64));
        assert_eq!(free_run(&[0, 1], 0, 128, 64), Some(0));
        assert_eq!(free_run(&[0, 1], 0, 128, 65), None);
        // Runs longer than a word, from a clear word on.
        assert_eq!(free_run(&[0u64; 2], 10, 128, 118), Some(10));
        assert_eq!(free_run(&[0u64; 2], 10, 128, 119), None);
        assert_eq!(free_run(&[0u64; 2], 10, 128, 100), Some(10));
        // 128-191 is a whole free word between taken ones.
        set_taken(&mut taken, 128, 192, true);
        assert_eq!(free_run(&taken, 0, 301, 45), Some(256));
        assert_eq!(free_run(&taken, 0, 301, 46), None);
        assert_eq!(free_run(&taken, 100, 301, 20), Some(256));
        set_taken(&mut taken, 128, 192, false);
        assert_eq!(free_run(&taken, 128, 301, 64), Some(128));
        set_taken(&mut taken, 128, 192, true);
        assert_eq!(free_run(&taken, 128, 301, 1), Some(192));
    }

    #[test]
    fn dynamic_search_can_find_a_run_across_the_cursor() {
        let Some(heap) = HostHeap::new(false, 4) else {
            eprintln!("no host descriptor heap; skipped");
            return;
        };
        let first = heap.dynamic(1).unwrap();
        let left = heap.dynamic(2).unwrap();
        let right = heap.dynamic(2).unwrap();
        let rest = heap.dynamic(heap.dynamic - 5).unwrap();
        assert_eq!((first, left, right, rest), (4, 5, 7, 9));

        // Reallocating the left half puts the cursor back in the middle.
        heap.release(heap.id(), left, 2);
        assert_eq!(heap.dynamic(2), Some(left));
        heap.release(heap.id(), left, 2);
        heap.release(heap.id(), right, 2);
        assert_eq!(heap.dynamic(4), Some(left));
        assert_eq!(heap.dynamic(1), None);

        // A free slot at each physical end does not form a two-slot range.
        heap.release(heap.id(), first, 1);
        let last = first + heap.dynamic - 1;
        heap.release(heap.id(), last, 1);
        assert_eq!(heap.dynamic(2), None);
    }

    /// A range of dynamic slots is returned only to the heap that allocated it:
    /// a stale release carrying another heap's identity must be a no-op.
    #[test]
    fn dynamic_slots_are_given_back_only_to_the_heap_that_took_them() {
        let Some(a) = HostHeap::new(false, 4) else {
            return; // no host GPU
        };
        let Some(b) = HostHeap::new(false, 4) else {
            return;
        };
        assert_ne!(a.id(), b.id());
        let first = a.dynamic(2).expect("two dynamic slots");
        let rest = a.dynamic(a.dynamic - 2).expect("remaining dynamic slots");
        assert_eq!(rest, first + 2);
        assert_eq!(a.dynamic(1), None, "heap is full");

        // Releasing with another heap's identity must not free A's slots.
        a.release(b.id(), first, 2);
        assert_eq!(a.dynamic(1), None, "foreign heap id must not free A's slots");

        // Releasing with A's own identity gives the range back.
        a.release(a.id(), first, 2);
        assert_eq!(a.dynamic(2), Some(first), "own heap id returns the range");
    }

    /// A texture view's size words are the console image descriptor's
    /// fields: level 0's size of the resource, the view's levels, its type.
    #[test]
    fn a_texture_view_gives_its_image_descriptor_size_fields() {
        // A 1000x600 2D texture of 10 levels, then an array of 6 with the
        // whole chain (mip count 0, 1024 wide: 11 levels).
        let mut texture = [0u32; 14];
        (texture[0], texture[4], texture[6], texture[7], texture[9]) = (3, 1000, 600, 1 | 10 << 16, 1);
        let mut array = texture;
        (array[4], array[7]) = (1024, 6);
        let mut view = [0u32; 10];
        // Levels 2-4 of the texture: width - 1 = 999 (0b11_1110_0111).
        (view[1], view[4], view[5]) = (4, 2, 3);
        let words = image_words(&texture, Some(&view), false);
        assert_eq!(words, [0, 3 << 30, 249 | 599 << 14, 0xFAC | 2 << 12 | 4 << 16 | 9 << 28]);
        // Its whole chain from level 1.
        (view[4], view[5]) = (1, u32::MAX);
        assert_eq!(image_words(&texture, Some(&view), false)[3], 0xFAC | 1 << 12 | 9 << 16 | 9 << 28);
        // An unordered view of level 3 (`MipSlice`, word 2): that level alone.
        let uav = [0, 4, 3, 0, 0, 0, 0, 0, 0, 0];
        assert_eq!(image_words(&texture, Some(&uav), true)[3], 0xFAC | 3 << 12 | 3 << 16 | 9 << 28);
        // An unordered view of level 2 of six array slices (`ArraySize`,
        // word 4): the level is not the slice count.
        let uav = [0, 5, 2, 1, 6, 0, 0, 0, 0, 0];
        assert_eq!(
            image_words(&array, Some(&uav), true)[3],
            0xFAC | 2 << 12 | 2 << 16 | 13 << 28
        );
        // The array's own view: a 2D array, levels 0-10.
        let words = image_words(&array, None, false);
        assert_eq!(words[1..], [3 << 30, 255 | 599 << 14, 0xFAC | 10 << 16 | 13 << 28]);
    }

    fn placed_memory() -> u64 {
        let layout = std::alloc::Layout::from_size_align(1 << 16, 1 << 16).unwrap();
        // SAFETY: non-zero size and power-of-two alignment; leaked for the test.
        unsafe { std::alloc::alloc_zeroed(layout) as u64 }
    }

    /// `len` zeroed bytes that no other test's memory comes near for the
    /// rest of the run: committed at the start of a reservation of 4 MiB
    /// windows of its own (the windows `crate::shadow` copies loose memory
    /// in), never freed. Memory from the process heap is handed to the next
    /// test at the same address, which finds what the tables keyed by address
    /// keep of the last one's (its placed textures, its views, the window
    /// copies of its memory and their guards), and its memory shares those
    /// windows with the tests running beside it.
    fn private_memory(len: usize) -> &'static mut [u8] {
        use windows_sys::Win32::System::Memory::{
            MEM_COMMIT, MEM_RESERVE, PAGE_READWRITE, VirtualAlloc,
        };
        const WINDOW: usize = 4 << 20;
        let span = len.next_multiple_of(WINDOW);
        // SAFETY: a fresh reservation, never freed: one window more than
        // `span`, so a whole aligned `span` lies inside it.
        let base =
            unsafe { VirtualAlloc(std::ptr::null(), span + WINDOW, MEM_RESERVE, PAGE_READWRITE) }
                as usize;
        assert_ne!(base, 0, "address space for the test's memory");
        let start = base.next_multiple_of(WINDOW);
        // SAFETY: inside the reservation above.
        let at =
            unsafe { VirtualAlloc(start as *const _, len, MEM_COMMIT, PAGE_READWRITE) } as usize;
        assert_eq!(at, start, "the test's memory committed");
        // SAFETY: `len` committed bytes (zeroed), used by the caller alone.
        unsafe { std::slice::from_raw_parts_mut(start as *mut u8, len) }
    }

    fn placed_tex_desc(width: u64, height: u32, format: u32, flags: u32) -> [u32; 14] {
        [
            3,
            0,
            0,
            0,
            width as u32,
            (width >> 32) as u32,
            height,
            1 | (1 << 16),
            format,
            1,
            0,
            0,
            flags,
            0,
        ]
    }

    fn placed_tex_uav_desc(va: u64, view_format: u32) -> [u32; 16] {
        let mut desc = [0u32; 16];
        desc[0] = view_format;
        desc[1] = 4; // D3D12_UAV_DIMENSION_TEXTURE2D
        desc[10] = va as u32;
        desc[11] = (va >> 32) as u32;
        desc
    }

    fn placed_tex_srv_desc(va: u64, view_format: u32) -> [u32; 16] {
        let mut desc = [0u32; 16];
        desc[0] = view_format;
        desc[1] = 4; // D3D12_SRV_DIMENSION_TEXTURE2D
        desc[2] = D3D12_DEFAULT_SHADER_4_COMPONENT_MAPPING;
        desc[5] = 1; // MipLevels
        desc[10] = va as u32;
        desc[11] = (va >> 32) as u32;
        desc
    }

    #[test]
    fn placed_uav_textures_create_real_host_uavs() {
        let Some(host) = host() else {
            return;
        };
        let Some(heap) = HostHeap::new(false, 32) else {
            return;
        };
        let _ = host.log_messages();
        let bgrx_uav = typed_store(88);
        let cases = [
            (28u32, 28u32, 4u32, true),     // R8G8B8A8_UNORM
            (27u32, 28u32, 4u32, true),     // R8G8B8A8_TYPELESS -> R8G8B8A8_UNORM
            (29u32, 29u32, 4u32, true),     // R8G8B8A8_UNORM_SRGB -> linear R8G8B8A8_UNORM
            (29u32, 0u32, 4u32, true),      // R8G8B8A8_UNORM_SRGB + UNKNOWN -> linear 28
            (91u32, 91u32, 4u32, true),     // B8G8R8A8_UNORM_SRGB -> linear B8G8R8A8_UNORM
            (91u32, 0u32, 4u32, true),      // B8G8R8A8_UNORM_SRGB + UNKNOWN -> linear 87
            (28u32, 28u32, 0u32, false),    // missing ALLOW_UNORDERED_ACCESS -> null fallback
            (93u32, 93u32, 4u32, bgrx_uav), // B8G8R8X8_UNORM_SRGB -> linear 88 if supported
            (93u32, 0u32, 4u32, bgrx_uav),  // B8G8R8X8_UNORM_SRGB + UNKNOWN -> linear 88
            (92u32, 88u32, 4u32, bgrx_uav), // B8G8R8X8_TYPELESS -> B8G8R8X8_UNORM
            (92u32, 93u32, 4u32, bgrx_uav), // B8G8R8X8_TYPELESS -> B8G8R8X8_UNORM_SRGB
            (86u32, 86u32, 4u32, true),     // B5G5R5A1_UNORM, widened if unsupported
            (115u32, 115u32, 4u32, true),   // B4G4R4A4_UNORM, widened if unsupported
            (40u32, 40u32, 4u32, false),    // D32_FLOAT + UAV flag (no depth flag) -> null fallback
            (55u32, 55u32, 4u32, false),    // D16_UNORM + UAV flag (no depth flag) -> null fallback
            (39u32, 40u32, 4u32, false),    // R32_TYPELESS + D32_FLOAT view -> null fallback
            (53u32, 55u32, 4u32, false),    // R16_TYPELESS + D16_UNORM view -> null fallback
            (27u32, 27u32, 4u32, false),    // R8G8B8A8_TYPELESS + typeless view -> null fallback
        ];
        let mut slots = [[0u32; 8]; 24];
        for (idx, (res_fmt, view_fmt, flags, expect_real)) in cases.into_iter().enumerate() {
            let va = placed_memory();
            let resource = placed_tex_desc(4, 4, res_fmt, flags);
            let desc = placed_tex_uav_desc(va, view_fmt);
            let slot_addr = slots[idx].as_mut_ptr() as u64;
            crate::descriptor::create_placed_uav(std::ptr::null(), &resource, &desc, slot_addr);
            let view = crate::descriptor::view(slot_addr).unwrap();
            assert!(matches!(
                placed_texture(&view),
                Some(View::Unordered(_, 0, Some(_)))
            ));
            assert!(heap.mirror(idx as u32, slot_addr, false).is_some());
            assert!(
                heap.kept.lock().unwrap().get(idx).is_none(),
                "case {idx} mirror(false) should defer host creation"
            );
            let mirrored = heap.mirror(idx as u32, slot_addr, true).unwrap();
            let dyn_slot = heap.count + idx as u32;
            heap.make_at(dyn_slot, &mirrored);
            let kept = heap.kept.lock().unwrap();
            assert_eq!(kept.get(idx).is_some(), expect_real, "case {idx} mirror");
            assert_eq!(
                kept.get(dyn_slot as usize).is_some(),
                expect_real,
                "case {idx} make_at"
            );
        }
        for (offset, (res_fmt, view_fmt, expect_real)) in [
            (39u32, 40u32, false), // R32_TYPELESS + D32_FLOAT SRV -> null fallback
            (53u32, 55u32, false), // R16_TYPELESS + D16_UNORM SRV -> null fallback
            (29u32, 0u32, true),   // R8G8B8A8_UNORM_SRGB + UAV + UNKNOWN SRV -> 29
            (91u32, 0u32, true),   // B8G8R8A8_UNORM_SRGB + UAV + UNKNOWN SRV -> 91
        ]
        .into_iter()
        .enumerate()
        {
            let idx = cases.len() + offset;
            let va = placed_memory();
            let resource = placed_tex_desc(4, 4, res_fmt, 4);
            let desc = placed_tex_srv_desc(va, view_fmt);
            let slot_addr = slots[idx].as_mut_ptr() as u64;
            crate::descriptor::create_placed_srv(std::ptr::null(), &resource, &desc, slot_addr);
            heap.mirror(idx as u32, slot_addr, true).unwrap();
            assert_eq!(
                heap.kept.lock().unwrap().get(idx).is_some(),
                expect_real,
                "srv case {offset}"
            );
        }
        // SAFETY: plain device query.
        assert!(unsafe { host.device.GetDeviceRemovedReason() }.is_ok());
        assert_eq!(host.log_messages(), None);
    }

    /// Wine's D3DMetal aborts while creating these typed UAVs. The production
    /// path must leave a null descriptor, no kept resource and no write mark.
    fn three_channel_uav_falls_back_under_wine(format: DXGI_FORMAT) {
        let Some(host) = host() else { return };
        let Some(heap) = HostHeap::new(false, 4) else { return };
        let p = crate::watch::tests::pool(1);
        crate::shadow::register(p, p as u64, 64);
        let mut resource = [0u32; 14];
        resource[0] = 1; // buffer
        resource[4] = 64;
        let mut words = [0u32; 16];
        words[0] = format.0 as u32;
        words[1] = 1; // buffer
        words[4] = 1; // one 12-byte element
        (words[10], words[11]) = (p as u32, (p >> 32) as u32);
        let slot = heap.dynamic(1).expect("a dynamic slot");
        heap.make_at(slot, &View::Placed(true, resource, words));
        assert!(!heap.kept(slot), "a refused UAV keeps no resource");
        let marks = heap.marks.as_ref().expect("the write-mark table").1;
        // SAFETY: two dwords of the mapped table for this live slot.
        let mark = unsafe { std::slice::from_raw_parts((marks as *const u32).add(slot as usize * 2), 2) };
        assert_eq!(mark, [victory_shader::emit::VIEW_MARK_NONE, 0], "a refused UAV has no write mark");
        // SAFETY: a plain device query.
        assert!(unsafe { host.device.GetDeviceRemovedReason() }.is_ok(), "the null fallback leaves the device alive");
        crate::shadow::unregister(p, p as u64);
        crate::watch::tests::free(p);
    }

    /// Synthetic hardware characterization: create a typed R32G32B32 UAV,
    /// store one three-component element, and copy the raw bytes back. Wine
    /// exercises the safe fallback instead, because raw creation aborts.
    fn characterize_r32g32b32(format: DXGI_FORMAT, scalar: &str, value: &str, expected: [u32; 3]) {
        eprintln!("typed UAV characterization: {format:?}");
        if crate::under_wine() {
            three_channel_uav_falls_back_under_wine(format);
            eprintln!("typed UAV characterization: Wine fallback passed");
            return;
        }
        let Some(host) = host() else {
            eprintln!("typed UAV characterization: no host device");
            return;
        };
        let device = &host.device;
        let mut support = D3D12_FEATURE_DATA_FORMAT_SUPPORT { Format: format, ..Default::default() };
        // SAFETY: a plain query into a struct of its size.
        let support_result = unsafe {
            device.CheckFeatureSupport(
                D3D12_FEATURE_FORMAT_SUPPORT,
                (&raw mut support).cast(),
                size_of_val(&support) as u32,
            )
        };
        eprintln!("typed UAV characterization: format support {support_result:?}, Support1={:#x}", support.Support1.0);
        if support_result.is_err()
            || support.Support1.0 & D3D12_FORMAT_SUPPORT1_TYPED_UNORDERED_ACCESS_VIEW.0 == 0
        {
            eprintln!("typed UAV characterization: host does not advertise typed UAV support; skipped");
            return;
        }
        let mut made: Option<ID3D12Resource> = None;
        // SAFETY: a plain committed buffer description.
        unsafe {
            device
                .CreateCommittedResource(
                    &D3D12_HEAP_PROPERTIES { Type: D3D12_HEAP_TYPE_DEFAULT, ..Default::default() },
                    D3D12_HEAP_FLAG_NONE,
                    &D3D12_RESOURCE_DESC {
                        Dimension: D3D12_RESOURCE_DIMENSION_BUFFER,
                        Width: 256,
                        Height: 1,
                        DepthOrArraySize: 1,
                        MipLevels: 1,
                        Format: DXGI_FORMAT_UNKNOWN,
                        SampleDesc: DXGI_SAMPLE_DESC { Count: 1, Quality: 0 },
                        Layout: D3D12_TEXTURE_LAYOUT_ROW_MAJOR,
                        Flags: D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS,
                        ..Default::default()
                    },
                    D3D12_RESOURCE_STATE_UNORDERED_ACCESS,
                    None,
                    &mut made,
                )
                .unwrap();
        }
        let buffer = made.unwrap();
        // SAFETY: a plain shader-visible descriptor heap.
        let heap: ID3D12DescriptorHeap = unsafe {
            device.CreateDescriptorHeap(&D3D12_DESCRIPTOR_HEAP_DESC {
                Type: D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV,
                NumDescriptors: 1,
                Flags: D3D12_DESCRIPTOR_HEAP_FLAG_SHADER_VISIBLE,
                ..Default::default()
            })
        }
        .unwrap();
        let desc = D3D12_UNORDERED_ACCESS_VIEW_DESC {
            Format: format,
            ViewDimension: D3D12_UAV_DIMENSION_BUFFER,
            Anonymous: D3D12_UNORDERED_ACCESS_VIEW_DESC_0 {
                Buffer: D3D12_BUFFER_UAV { NumElements: 1, ..Default::default() },
            },
        };
        // SAFETY: a live buffer and one slot of our heap.
        unsafe {
            let cpu = heap.GetCPUDescriptorHandleForHeapStart();
            device.CreateUnorderedAccessView(&buffer, None::<&ID3D12Resource>, Some(&desc), cpu);
        }
        // SAFETY: a plain device query.
        let removed = unsafe { device.GetDeviceRemovedReason() };
        eprintln!("typed UAV characterization: after creation {removed:?}");
        removed.expect("creating the typed UAV leaves the device alive");
        let source = format!(
            "// {format:?}\nRWBuffer<{scalar}3> output : register(u0);\n[numthreads(1, 1, 1)]\nvoid main() {{ output[0] = {scalar}3({value}); }}\n"
        );
        let (root, pso) = crate::compute::pipeline(
            &source,
            windows::core::s!("typed uav characterization"),
            0,
            &[crate::compute::Param::Table(&[crate::compute::uav(0, 1)])],
        )
        .expect("the typed-store shader compiles");
        let list = host.new_list().unwrap();
        let (readback, data) = list.readback(host, 256).unwrap();
        // SAFETY: an open list, live objects, and a descriptor in our heap.
        unsafe {
            let gpu = heap.GetGPUDescriptorHandleForHeapStart();
            list.list.SetDescriptorHeaps(&[Some(heap.clone())]);
            list.list.SetComputeRootSignature(&root);
            list.list.SetComputeRootDescriptorTable(0, gpu);
            list.list.SetPipelineState(&pso);
            list.list.Dispatch(1, 1, 1);
            list.list.ResourceBarrier(&[crate::host::transition(
                &buffer,
                D3D12_RESOURCE_STATE_UNORDERED_ACCESS,
                D3D12_RESOURCE_STATE_COPY_SOURCE,
            )]);
            list.list.CopyResource(&readback, &buffer);
        }
        list.close();
        host.execute(&[&list]);
        // SAFETY: execute waited for the copy; the mapped readback is 256 bytes.
        let bytes = unsafe { std::slice::from_raw_parts(data.cast::<u8>(), 36) };
        eprintln!("typed UAV characterization: raw readback {bytes:?}");
        let want: Vec<u8> = expected.iter().flat_map(|word| word.to_le_bytes()).collect();
        assert_eq!(bytes, want, "typed UAV store/raw readback");
    }

    #[test]
    fn characterize_r32g32b32_float_typed_uav() {
        characterize_r32g32b32(
            DXGI_FORMAT_R32G32B32_FLOAT,
            "float",
            "1.25, -2.5, 3.75",
            [1.25f32.to_bits(), (-2.5f32).to_bits(), 3.75f32.to_bits()],
        );
    }

    #[test]
    fn characterize_r32g32b32_uint_typed_uav() {
        characterize_r32g32b32(
            DXGI_FORMAT_R32G32B32_UINT,
            "uint",
            "0x11223344u, 0x55667788u, 0x99aabbccu",
            [0x1122_3344, 0x5566_7788, 0x99aa_bbcc],
        );
    }

    #[test]
    fn characterize_r32g32b32_sint_typed_uav() {
        characterize_r32g32b32(
            DXGI_FORMAT_R32G32B32_SINT,
            "int",
            "-1, -2, -3",
            [(-1i32) as u32, (-2i32) as u32, (-3i32) as u32],
        );
    }

    #[test]
    fn sync_slots_skips_empty_and_clears_after_forget() {
        use crate::object::{HRESULT, S_OK};
        let Some(host) = HostHeap::new(false, 256) else {
            return;
        };
        let desc: [u32; 14] = [1, 0, 0, 0, 64, 0, 1, 1 | 1 << 16, 0, 1, 0, 0, 0, 0];
        let properties = [1u32, 0, 0, 0, 0];
        let mut resource = std::ptr::null();
        let iid = crate::object::guid(
            0x696442be,
            0xa72e,
            0x4059,
            [0xbc, 0x79, 0x5b, 0x5c, 0x98, 0x04, 0x0f, 0xad],
        );
        let made: HRESULT = crate::resource::create_committed(
            std::ptr::null(),
            properties.as_ptr().cast(),
            0,
            desc.as_ptr().cast(),
            0,
            std::ptr::null(),
            &iid,
            &mut resource,
        );
        assert_eq!(made, S_OK);

        let mut title_heap = vec![0u8; 256 * 32];
        let start = title_heap.as_ptr() as u64;
        crate::descriptor::create_srv(
            std::ptr::null(),
            resource,
            std::ptr::null(),
            start,
        );
        host.sync_slots_bytes(0, 128, start, 32, start, &title_heap[..128 * 32]);
        assert!(host.made(0));
        assert!(host.made_is_nonzero(0));
        assert!(!host.made(1));
        assert!(!host.made_is_nonzero(1));

        // Second sync over the verified block keeps slot 0 live and skips empty slots.
        host.sync_slots_bytes(0, 128, start, 32, start, &title_heap[..128 * 32]);
        assert!(host.made(0));

        // Forgetting the resource bumps VIEWS_FORGOT so the verified block re-checks
        // and clears slot 0 even though the title memory bytes didn't change.
        crate::descriptor::forget(resource as usize);
        host.sync_slots_bytes(0, 128, start, 32, start, &title_heap[..128 * 32]);
        assert!(!host.made(0));
        assert!(!host.made_is_nonzero(0));

        crate::object::release(resource);
        let _ = title_heap.as_mut_ptr();
    }

    /// PR #1246 fair queue: previously retried slots and never-looked-at fresh
    /// slots alternate 1:1 (a retried slot first), so a large retry backlog
    /// cannot starve fresh slots; `fair = false` restores retries-first order.
    #[test]
    fn upload_queue_interleaves_retries_and_fresh_slots() {
        let mut q = UploadQueue::default();
        for i in 0..10u32 {
            q.written(100 + i, 1);
        }
        for i in 0..10u32 {
            let w = q
                .due_with(u64::MAX, 100, true)
                .into_iter()
                .find(|d| d.0 == 100 + i)
                .unwrap()
                .1;
            q.later(100 + i, w, 1, 1);
        }
        for i in 1..4u32 {
            q.written(i, 2);
        }
        let fair: Vec<u32> = q.due_with(10, 6, true).into_iter().map(|d| d.0).collect();
        assert_eq!(fair.len(), 6);
        assert!(fair[0] >= 100, "first slot is a retry: {fair:?}");
        assert!(fair[1] < 100, "second slot is fresh: {fair:?}");
        assert!(fair[2] >= 100, "third slot is a retry: {fair:?}");
        assert!(fair[3] < 100, "fourth slot is fresh: {fair:?}");
        assert_eq!(
            fair.iter().filter(|&&i| i < 100).count(),
            3,
            "all fresh slots reached in pass of 6: {fair:?}"
        );
        let legacy: Vec<u32> = q.due_with(10, 6, false).into_iter().map(|d| d.0).collect();
        assert!(
            legacy.iter().all(|&i| i >= 100),
            "legacy mode puts retries ahead of fresh: {legacy:?}"
        );
    }

    /// Tier-0 read-marked slots preempt retries and fresh slots, `requeue`
    /// restores unqueued slots without disturbing queued ones, and relook
    /// backoff clamps to `(VICTORY_RELOOK_MS / frame_ms).clamp(16, 1024)`.
    #[test]
    fn upload_queue_prioritizes_tier0_reads_and_respects_relook_cap() {
        assert_eq!(relook_frames(1000, 20_000), 20);
        assert_eq!(relook_frames(5000, 20_000), 16, "clamps low at 16");
        assert_eq!(relook_frames(1, 20_000), 1024, "clamps high at 1024");
        assert_eq!(relook_frames(250, 20_000), 80);

        let mut q = UploadQueue::default();
        q.written(10, 1);
        q.written(11, 1);
        let w10 = q.due_with(1, 10, true).into_iter().find(|d| d.0 == 10).unwrap().1;
        q.later(10, w10, 1, 16);
        // After 1 look, due_frame = 2; after repeated looks, backoff caps at 16.
        for f in [2u64, 4, 8, 16, 32, 64] {
            let w = q.queued(&[10])[0].1;
            q.later(10, w, f, 16);
        }
        assert!(
            !q.due_with(79, 10, true).iter().any(|d| d.0 == 10),
            "capped at +16 frames (due at 80)"
        );
        assert!(q.due_with(80, 10, true).iter().any(|d| d.0 == 10));

        // Tier-0 prioritize puts slot 50 ahead of both retries and fresh slots.
        q.prioritize(&[50], 1);
        let first = q.due_with(80, 3, true);
        assert_eq!(first[0].0, 50, "Tier-0 read-marked slot comes first");

        // `requeue` adds unqueued slot 60 as urgent without resetting slot 11.
        let before11 = q.queued(&[11]);
        q.requeue(&[11, 60], 80);
        assert_eq!(q.queued(&[11]), before11);
        let due = q.due_with(80, 4, true);
        assert_eq!(due[0].0, 60);
        assert_eq!(due[1].0, 50);
    }

    /// PR #1007 publication invariant: a slot published with its matching key
    /// becomes live without re-entering the upload queue, and a stale key is
    /// rejected after the title overwrites the slot.
    #[test]
    fn published_slot_stays_out_of_queue_and_rejects_stale_key() {
        let Some(heap) = HostHeap::new(false, 8) else {
            return; // no host GPU
        };
        let _worker = crate::heap_worker::EnableForTest::new();
        let mem = crate::watch::tests::pool(1);
        let placed = |at: u64| {
            let mut desc = [0u32; 16];
            // RGBA8 4x1 linear, 1 mip
            (desc[0], desc[1], desc[2], desc[5]) = (28, 4, 0x1688, 1);
            (desc[10], desc[11]) = (at as u32, (at >> 32) as u32);
            View::Placed(
                false,
                [3, 0, 0, 0, 4, 0, 1, 1 | 1 << 16, 28, 1, 0, 0x100, 0, 0],
                desc,
            )
        };
        let mut title = [0u32; 8];
        let handle = title.as_mut_ptr() as u64;
        crate::descriptor::record(handle, placed(mem as u64));
        let (_, key1) = crate::descriptor::view_and_key(handle).unwrap();
        assert!(heap.mirror(1, handle, false).is_some());
        assert!(!heap.kept(1), "placed texture starts null until published");
        let due = heap.texture_slots_due(u64::MAX, 8);
        assert_eq!(due.len(), 1);
        assert_eq!(due[0].0, 1);
        heap.texture_slot_done(1, due[0].1);
        assert!(heap.publish(1, handle, key1));
        assert!(heap.kept(1), "slot now shows the published texture");
        assert!(
            heap.texture_slots_due(u64::MAX, 8).is_empty(),
            "publish must not re-enqueue the slot"
        );
        // Overwrite the slot with another record: stale `key1` must be rejected.
        crate::descriptor::record(handle, placed((mem + 256) as u64));
        let (_, key2) = crate::descriptor::view_and_key(handle).unwrap();
        assert_ne!(key1, key2);
        assert!(!heap.publish(1, handle, key1));
        crate::watch::tests::free(mem);
    }

    #[test]
    fn fast_heap_sync_env_toggle() {
        use std::ffi::OsStr;
        assert!(fast_heap_sync_value(None));
        assert!(fast_heap_sync_value(Some(OsStr::new("1"))));
        assert!(fast_heap_sync_value(Some(OsStr::new(""))));
        assert!(!fast_heap_sync_value(Some(OsStr::new("0"))));
    }

    struct PlacedAtOnceGuard(Option<PlacedAtOnce>);

    impl PlacedAtOnceGuard {
        fn set(mode: PlacedAtOnce) -> Self {
            Self(TEST_PLACED_AT_ONCE.replace(Some(mode)))
        }
    }

    impl Drop for PlacedAtOnceGuard {
        fn drop(&mut self) {
            TEST_PLACED_AT_ONCE.set(self.0);
        }
    }

    #[test]
    fn placed_at_once_env_parsing() {
        for off in [None, Some(""), Some("0"), Some("yes"), Some("true"), Some("2"), Some("Keep")] {
            assert_eq!(PlacedAtOnce::parse(off), PlacedAtOnce::Off, "{off:?}");
            assert!(!PlacedAtOnce::parse(off).makes_existing());
        }
        assert_eq!(PlacedAtOnce::parse(Some("1")), PlacedAtOnce::Made);
        assert_eq!(PlacedAtOnce::parse(Some(" 1 ")), PlacedAtOnce::Made);
        assert_eq!(PlacedAtOnce::parse(Some("keep")), PlacedAtOnce::Keep);
        assert_eq!(PlacedAtOnce::parse(Some(" keep ")), PlacedAtOnce::Keep);
        assert_eq!(PlacedAtOnce::parse(Some("all")), PlacedAtOnce::All);
        assert!(
            [PlacedAtOnce::Made, PlacedAtOnce::Keep, PlacedAtOnce::All]
                .iter()
                .all(|m| m.makes_existing())
        );
        if std::env::var_os("VICTORY_PLACED_AT_ONCE").is_none() {
            assert_eq!(placed_at_once(), PlacedAtOnce::Off);
        }
    }

    #[test]
    fn placed_at_once_keep_preserves_prior_view_until_draw_or_read() {
        let Some(host_gpu) = host() else {
            return; // no host GPU here
        };
        let Some(heap) = HostHeap::new(false, 128) else {
            return;
        };
        let _guard = PlacedAtOnceGuard::set(PlacedAtOnce::Keep);

        // Four distinct linear RGBA8 4x1 placed textures in separate addresses.
        let mem = vec![0u8; 4 << 16];
        let base = mem.as_ptr() as u64;
        let addr_a = base;
        let addr_b = base + (1 << 16);
        let addr_c = base + (2 << 16);
        let addr_d = base + (3 << 16);
        let tex_words: [u32; 14] = [3, 0, 0, 0, 4, 0, 1, 1 | 1 << 16, 28, 1, 0, 0x100, 0, 0];
        let make_view = |addr: u64| {
            let mut v = [0u32; 16];
            v[..6].copy_from_slice(&[28, 4, 0x1688, 0, 0, 1]);
            (v[10], v[11]) = (addr as u32, (addr >> 32) as u32);
            v
        };

        let mut title_heap = vec![0u8; 128 * 32];
        let start = title_heap.as_ptr() as u64;
        let slot_addr = |i: u32| start + u64::from(i) * 32;
        let write_placed = |i: u32, addr: u64| -> u64 {
            let view = make_view(addr);
            crate::descriptor::create_placed_srv(std::ptr::null(), &tex_words, &view, slot_addr(i));
            crate::descriptor::view_and_key(slot_addr(i)).unwrap().1
        };

        // Prime slots 0 and 1 with placed texture A (`for_draw = true` creates its host texture).
        let key_a0 = write_placed(0, addr_a);
        let key_a1 = write_placed(1, addr_a);
        assert!(heap.mirror(0, slot_addr(0), true).is_some());
        assert!(heap.mirror(1, slot_addr(1), true).is_some());
        assert_eq!(heap.ready[0].load(Ordering::Acquire), key_a0);
        assert_eq!(heap.ready[1].load(Ordering::Acquire), key_a1);
        let tex_a = crate::resource::texture_at(addr_a, &tex_words).expect("texture A");
        let raw_a = windows::core::Interface::as_raw(
            &crate::resource::made_copy(tex_a).expect("made A").texture,
        ) as usize;
        assert_eq!(heap.textures_at(&[0]), [raw_a]);
        assert!(crate::resource::placed_made(addr_a, &tex_words));
        assert!(!crate::resource::placed_made(addr_b, &tex_words));

        // Batched sync path (`mirror_synced_batch`): rewrite slot 0 to unmade placed texture B,
        // and write fresh empty slot 2 with unmade placed texture D.
        let key_b = write_placed(0, addr_b);
        let key_d = write_placed(2, addr_d);
        heap.sync_slots_bytes_with(0..128, start, 32, start, &title_heap, true);
        assert!(heap.made(0));
        assert_eq!(heap.synced[0].load(Ordering::Acquire), key_b);
        assert_eq!(heap.ready[0].load(Ordering::Acquire), 0);
        assert!(!heap.slot_ready(0, slot_addr(0)));
        assert!(heap.kept(0), "slot 0 keeps prior host view instead of null");
        assert_eq!(heap.textures_at(&[0]), [raw_a], "slot 0 still shows texture A");
        assert!(heap.unmade_placed(0).is_some(), "make_read sees unmade kept slot");
        assert_eq!(
            heap.unmade_read_kind(0),
            UnmadeRead::Background,
            "heap_worker sees unmade kept slot"
        );
        assert!(!crate::resource::placed_made(addr_b, &tex_words), "B not allocated eagerly");
        assert!(heap.made(2));
        assert_eq!(heap.synced[2].load(Ordering::Acquire), key_d);
        assert!(!heap.kept(2), "empty slot with no prior view stays null");

        // Drawing / shader-read remakes slot 0 to texture B cleanly on demand.
        assert!(heap.mirror(0, slot_addr(0), true).is_some());
        let tex_b = crate::resource::texture_at(addr_b, &tex_words).expect("texture B");
        let raw_b = windows::core::Interface::as_raw(
            &crate::resource::made_copy(tex_b).expect("made B").texture,
        ) as usize;
        assert_ne!(raw_a, raw_b);
        assert_eq!(heap.textures_at(&[0]), [raw_b], "slot 0 remade to texture B");
        assert_eq!(heap.ready[0].load(Ordering::Acquire), key_b);
        assert!(heap.unmade_placed(0).is_none());
        assert_eq!(heap.unmade_read_kind(0), UnmadeRead::None);

        // Rewriting slot 0 back to already-made texture A updates immediately on sync (no null frame).
        let key_a_again = write_placed(0, addr_a);
        heap.sync_slots_bytes_with(0..128, start, 32, start, &title_heap, true);
        assert_eq!(heap.textures_at(&[0]), [raw_a], "existing host texture made at once");
        assert_eq!(heap.ready[0].load(Ordering::Acquire), key_a_again);

        // Direct `mirror(..., false)` path: rewrite slot 1 from A to unmade C -> keeps A until draw.
        let key_c = write_placed(1, addr_c);
        assert!(heap.mirror(1, slot_addr(1), false).is_some());
        assert_eq!(heap.synced[1].load(Ordering::Acquire), key_c);
        assert_eq!(heap.ready[1].load(Ordering::Acquire), 0);
        assert_eq!(heap.textures_at(&[1]), [raw_a], "mirror(false) keeps prior view");
        assert!(heap.mirror(1, slot_addr(1), true).is_some());
        let tex_c = crate::resource::texture_at(addr_c, &tex_words).expect("texture C");
        let raw_c = windows::core::Interface::as_raw(
            &crate::resource::made_copy(tex_c).expect("made C").texture,
        ) as usize;
        assert_eq!(heap.textures_at(&[1]), [raw_c], "mirror(true) remakes to texture C");
        assert_eq!(heap.ready[1].load(Ordering::Acquire), key_c);

        // SAFETY: plain device health check.
        assert!(unsafe { host_gpu.device.GetDeviceRemovedReason() }.is_ok());
        let _ = title_heap.as_mut_ptr();
        let _ = mem.as_ptr();
    }

    #[test]
    fn fast_heap_sync_handles_placed_null_duplicates_forget_and_concurrent_rewrite() {
        let Some(host) = HostHeap::new(false, 512) else {
            return;
        };
        let make_shadowed_buffer = || {
            let b = crate::resource::test_buffer(256);
            let (ptr, len) = crate::resource::buffer_bytes(b).expect("buffer bytes");
            crate::shadow::register(b as usize, ptr as u64, len);
            b
        };
        let buf_a = make_shadowed_buffer();
        let buf_b = make_shadowed_buffer();

        let mut title_heap = vec![0u8; 512 * 32];
        let start = title_heap.as_ptr() as u64;
        let slot_addr = |i: u32| start + u64::from(i) * 32;
        // R8_UINT buffer SRV, all 256 elements (valid host buffer SRV; word 6 = NumElements).
        let srv_desc = [62u32, 1, 0x1688, 0, 0, 0, 256, 0, 0, 0];

        // 10 duplicate copies of buffer SRV A (exercises template copy via StagingHeap).
        for i in 0..10u32 {
            crate::descriptor::create_srv(std::ptr::null(), buf_a, &srv_desc, slot_addr(i));
        }

        // 10 duplicate copies of a placed 2D texture (left null on sync, ready == 0, synced == key).
        let tex_mem = vec![0u8; 1 << 16];
        let tex_addr = tex_mem.as_ptr() as u64;
        let tex_words: [u32; 14] = [3, 0, 0, 0, 4, 0, 1, 1 | 1 << 16, 28, 1, 0, 0x100, 0, 0];
        let mut tex_view = [0u32; 16];
        tex_view[..6].copy_from_slice(&[28, 4, 0x1688, 0, 0, 1]);
        (tex_view[10], tex_view[11]) = (tex_addr as u32, (tex_addr >> 32) as u32);
        for i in 20..30u32 {
            crate::descriptor::create_placed_srv(
                std::ptr::null(),
                &tex_words,
                &tex_view,
                slot_addr(i),
            );
        }

        // First fast sync populates buffer slots via staging template and placed textures via null template.
        host.sync_slots_bytes_with(0..512, start, 32, start, &title_heap, true);
        for i in 0..10u32 {
            assert!(host.made(i), "buffer slot {i} made");
            assert!(host.kept(i), "buffer slot {i} kept");
            assert_ne!(host.synced[i as usize].load(Ordering::Acquire), 0);
            assert_eq!(
                host.ready[i as usize].load(Ordering::Acquire),
                host.synced[i as usize].load(Ordering::Acquire)
            );
        }
        for i in 20..30u32 {
            assert!(host.made(i), "placed texture slot {i} noted");
            assert!(!host.kept(i), "placed texture slot {i} left null until drawn");
            assert_ne!(host.synced[i as usize].load(Ordering::Acquire), 0);
            assert_eq!(host.ready[i as usize].load(Ordering::Acquire), 0);
        }

        // Second fast sync skips all slots locklessly and preserves placed texture ready == 0.
        host.sync_slots_bytes_with(0..512, start, 32, start, &title_heap, true);
        for i in 20..30u32 {
            assert!(!host.kept(i));
            assert_eq!(host.ready[i as usize].load(Ordering::Acquire), 0);
        }
        // Drawing slot 20 still remakes the placed texture on demand.
        assert!(host.mirror(20, slot_addr(20), true).is_some());
        assert!(host.kept(20), "placed texture made on draw");

        // Forgetting buffer A immediately invalidates all 10 buffer slots on next sync
        // even though title heap bytes did not change.
        crate::descriptor::forget(buf_a as usize);
        host.sync_slots_bytes_with(0..512, start, 32, start, &title_heap, true);
        for i in 0..10u32 {
            assert!(!host.made(i), "forgotten slot {i} cleared");
            assert!(!host.made_is_nonzero(i), "forgotten slot {i} nonzero bit cleared");
            assert!(!host.kept(i), "forgotten slot {i} no longer kept");
            assert_eq!(host.synced[i as usize].load(Ordering::Acquire), 0);
            assert_eq!(host.ready[i as usize].load(Ordering::Acquire), 0);
        }

        // Concurrent title rewrite during `prepare` leaves the slot holding the latest view.
        let buf_c = make_shadowed_buffer();
        let race_slot = 64u32;
        let race_addr = slot_addr(race_slot);
        crate::descriptor::create_srv(std::ptr::null(), buf_b, &srv_desc, race_addr);
        let buf_c_addr = buf_c as usize;
        let mut expected_c = [0u32; 4];
        crate::descriptor::create_srv(
            std::ptr::null(),
            buf_c,
            &srv_desc,
            expected_c.as_mut_ptr() as u64,
        );
        let expected_c_key = u64::from(expected_c[2]) << 32 | u64::from(expected_c[1]);
        when_preparing(move || {
            crate::descriptor::create_srv(
                std::ptr::null(),
                buf_c_addr as crate::object::This,
                &srv_desc,
                race_addr,
            );
        });
        host.sync_slots_bytes_with(0..128, start, 32, start, &title_heap[..128 * 32], true);
        assert!(prepared());
        assert_eq!(
            host.synced[race_slot as usize].load(Ordering::Acquire),
            expected_c_key,
            "concurrent rewrite during prepare won"
        );
        assert!(host.kept(race_slot));
        // SAFETY: plain device health check after batched CopyDescriptors.
        assert!(unsafe { crate::host::host().unwrap().device.GetDeviceRemovedReason() }.is_ok());

        crate::object::release(buf_a);
        crate::object::release(buf_b);
        crate::object::release(buf_c);
        let _ = title_heap.as_mut_ptr();
        let _ = tex_mem.as_ptr();
    }

    /// Slots made again as a ring of descriptor tables does it: every slot
    /// takes another view each pass. Times a slot made again and prints
    /// where its time goes ("heap slots made again" in the frame report).
    /// `cargo test --release -p victory-gpu --lib bench_made_again --
    /// --ignored --nocapture`.
    #[test]
    #[ignore = "times heap slots made again"]
    fn bench_made_again() {
        const SLOTS: u32 = 16_384;
        const PAGE_SLOTS: u32 = 2_048;
        let Some(host) = HostHeap::new(false, SLOTS) else {
            return;
        };
        let srv_desc = [62u32, 1, 0x1688, 0, 0, 0, 256, 0, 0, 0];
        let records: Vec<[u32; 3]> = (0..256)
            .map(|_| {
                let b = crate::resource::test_buffer(256);
                let (ptr, len) = crate::resource::buffer_bytes(b).expect("buffer bytes");
                crate::shadow::register(b as usize, ptr as u64, len);
                let mut rec = [0u32; 3];
                crate::descriptor::create_srv(std::ptr::null(), b, &srv_desc, rec.as_mut_ptr() as u64);
                rec
            })
            .collect();
        let mem = private_memory(SLOTS as usize * 32);
        let start = mem.as_ptr() as u64;
        let write = |mem: &mut [u8], pass: usize| {
            for i in 0..SLOTS as usize {
                let rec = records[(i * 7 + pass * 13) % records.len()];
                for (k, w) in rec.iter().enumerate() {
                    mem[i * 32 + 4 * k..i * 32 + 4 * k + 4].copy_from_slice(&w.to_le_bytes());
                }
            }
        };
        let sync = |mem: &[u8]| {
            for p in 0..SLOTS / PAGE_SLOTS {
                let (first, last) = (p * PAGE_SLOTS, (p + 1) * PAGE_SLOTS);
                let (b0, b1) = (first as usize * 32, last as usize * 32);
                host.sync_slots_bytes_with(first..last, start, 32, start + b0 as u64, &mem[b0..b1], true);
            }
        };
        write(mem, 0);
        sync(mem);
        let _ = crate::descriptor::SYNC_STATS.each_ref().map(|c| c.swap(0, Ordering::Relaxed));
        let passes = 20;
        let started = std::time::Instant::now();
        for pass in 1..=passes {
            write(mem, pass);
            sync(mem);
        }
        let each = started.elapsed().as_secs_f64() * 1e6 / (passes * SLOTS as usize) as f64;
        let st = crate::descriptor::SYNC_STATS.each_ref().map(|c| c.swap(0, Ordering::Relaxed));
        eprintln!(
            "a slot made again: {each:.2} us; made again {}: unchanged {}, created {} ({} us), from a template {}; \
             us: prepare {}, loop {}, copy {}, keep {}, viewed {}, all making {}",
            st[7], st[8], st[9], st[10], st[11], st[16], st[17], st[15], st[18], st[19], st[6]
        );
    }

    #[test]
    fn the_view_probe_counts_descriptors_an_earlier_sync_created() {
        // Keys no other test makes (generation 0xffff_fff0..).
        let keys: Vec<u64> = (0..4).map(|i| (0xffff_fff0u64 + i) << 32 | 7).collect();
        assert_eq!(crate::descriptor::seen_before(&keys[..2]), 0);
        assert_eq!(crate::descriptor::seen_before(&keys), 2, "the first two again");
    }

    /// What a 12-byte read through the kernel costs (one a slot made
    /// again, `mirror_synced_batch`) against reading the page once.
    #[test]
    #[ignore = "times kernel reads of heap records"]
    fn bench_record_reads() {
        let mem = private_memory(1 << 16);
        let rounds = 100_000;
        let mut raw = [0u8; 12];
        let started = std::time::Instant::now();
        for i in 0..rounds {
            assert!(crate::host::read_into(mem[(i % 2048) * 32..].as_ptr(), &mut raw));
        }
        let each = started.elapsed().as_secs_f64() * 1e6 / rounds as f64;
        let mut page = vec![0u8; 1 << 16];
        let started = std::time::Instant::now();
        for _ in 0..1000 {
            assert!(crate::host::read_into(mem.as_ptr(), &mut page));
        }
        let whole = started.elapsed().as_secs_f64() * 1e6 / 1000.0;
        eprintln!("12-byte kernel read {each:.2} us; a 64 KiB page {whole:.1} us (2,048 records)");
    }

    /// What syncing a heap page the title rewrote with the same records
    /// costs (every slot scanned, none stale), against comparing the page
    /// with its bytes from the last sync. `cargo test --release -p
    /// victory-gpu --lib bench_unchanged_page -- --ignored --nocapture`.
    #[test]
    #[ignore = "times a sync of unchanged heap pages"]
    fn bench_unchanged_page_scan_vs_compare() {
        const SLOTS: u32 = 65_536;
        const PAGE_SLOTS: u32 = 2_048;
        let Some(host) = HostHeap::new(false, SLOTS) else {
            return;
        };
        let srv_desc = [62u32, 1, 0x1688, 0, 0, 0, 256, 0, 0, 0];
        let records: Vec<[u32; 3]> = (0..64)
            .map(|_| {
                let b = crate::resource::test_buffer(256);
                let (ptr, len) = crate::resource::buffer_bytes(b).expect("buffer bytes");
                crate::shadow::register(b as usize, ptr as u64, len);
                let mut rec = [0u32; 3];
                crate::descriptor::create_srv(std::ptr::null(), b, &srv_desc, rec.as_mut_ptr() as u64);
                rec
            })
            .collect();
        let mem = private_memory(SLOTS as usize * 32);
        for i in 0..SLOTS as usize {
            let rec = records[i % records.len()];
            for (k, w) in rec.iter().enumerate() {
                mem[i * 32 + 4 * k..i * 32 + 4 * k + 4].copy_from_slice(&w.to_le_bytes());
            }
        }
        let start = mem.as_ptr() as u64;
        let sync = |host: &HostHeap| {
            for p in 0..SLOTS / PAGE_SLOTS {
                let (first, last) = (p * PAGE_SLOTS, (p + 1) * PAGE_SLOTS);
                let (b0, b1) = (first as usize * 32, last as usize * 32);
                host.sync_slots_bytes_with(first..last, start, 32, start + b0 as u64, &mem[b0..b1], true);
            }
        };
        sync(&host);
        let pages = (SLOTS / PAGE_SLOTS) as usize;
        let rounds = 20;
        let started = std::time::Instant::now();
        for _ in 0..rounds {
            sync(&host);
        }
        let scan = started.elapsed().as_secs_f64() * 1e6 / (rounds * pages) as f64;
        let saved = mem.to_vec();
        let started = std::time::Instant::now();
        let mut same = 0;
        for _ in 0..rounds {
            for p in 0..pages {
                let r = p * 65_536..(p + 1) * 65_536;
                same += usize::from(std::hint::black_box(&mem[r.clone()]) == &saved[r]);
            }
        }
        let compare = started.elapsed().as_secs_f64() * 1e6 / (rounds * pages) as f64;
        assert_eq!(same, rounds * pages);
        eprintln!("unchanged 64 KiB heap page ({PAGE_SLOTS} slots): scan {scan:.1} us, compare {compare:.1} us");
    }

    #[test]
    fn bench_sync_slots_1_5m_fast_vs_legacy() {
        const SLOTS: u32 = 65_536; // 32 x 64 KiB pages (2,048 32-byte slots/page)
        const PASSES: usize = 23; // 23 * 65,536 = 1,507,328 slots scanned
        const PAGE_SLOTS: u32 = 2_048;

        let Some(host_legacy) = HostHeap::new(false, SLOTS) else {
            return;
        };
        let Some(host_fast) = HostHeap::new(false, SLOTS) else {
            return;
        };

        // Valid R8_UINT buffer SRV covering all 256 bytes (word 6 = NumElements).
        let srv_desc = [62u32, 1, 0x1688, 0, 0, 0, 256, 0, 0, 0];

        // 150 distinct buffer resources (10x duplication across active/stale slots),
        // registered with shadow so every buffer SRV creates a real D3D12 view + bindless hull.
        let make_shadowed_buffer = || {
            let b = crate::resource::test_buffer(256);
            let (ptr, len) = crate::resource::buffer_bytes(b).expect("buffer bytes");
            crate::shadow::register(b as usize, ptr as u64, len);
            b
        };
        let buffers: Vec<_> = (0..150).map(|_| make_shadowed_buffer()).collect();
        // Extra resource forgotten after benchmark to verify Views/HostHeap invalidation.
        let ephemeral = make_shadowed_buffer();

        let tex_addr = private_memory(1 << 16).as_ptr() as u64;
        let tex_words: [u32; 14] = [3, 0, 0, 0, 4, 0, 1, 1 | 1 << 16, 28, 1, 0, 0x100, 0, 0];
        let mut placed_records: Vec<[u32; 3]> = Vec::with_capacity(40);
        for k in 0..40u64 {
            let mut view = [0u32; 16];
            view[..6].copy_from_slice(&[28, 4, 0x1688, 0, 0, 1]);
            let addr = tex_addr + k * 256;
            (view[10], view[11]) = (addr as u32, (addr >> 32) as u32);
            let mut rec = [0u32; 3];
            crate::descriptor::create_placed_srv(
                std::ptr::null(),
                &tex_words,
                &view,
                rec.as_mut_ptr() as u64,
            );
            placed_records.push(rec);
        }

        let mut buf_records: Vec<[u32; 3]> = Vec::with_capacity(buffers.len());
        for &b in &buffers {
            let mut rec = [0u32; 3];
            crate::descriptor::create_srv(
                std::ptr::null(),
                b,
                &srv_desc,
                rec.as_mut_ptr() as u64,
            );
            buf_records.push(rec);
        }
        let mut eph_record = [0u32; 3];
        crate::descriptor::create_srv(
            std::ptr::null(),
            ephemeral,
            &srv_desc,
            eph_record.as_mut_ptr() as u64,
        );

        let write_rec = |mem: &mut [u8], slot: usize, rec: [u32; 3]| {
            let off = slot * 32;
            mem[off..off + 4].copy_from_slice(&rec[0].to_le_bytes());
            mem[off + 4..off + 8].copy_from_slice(&rec[1].to_le_bytes());
            mem[off + 8..off + 12].copy_from_slice(&rec[2].to_le_bytes());
        };

        let build_heap = || {
            let mem = private_memory(SLOTS as usize * 32);
            // 10,000 steady-state buffer views (real host SRVs), 20,000 steady-state
            // placed-texture-left-null slots (ready == 0, synced == key), 16 ephemeral
            // slots, remainder empty.
            for i in 0..10_000usize {
                write_rec(mem, i, buf_records[i % 75]);
            }
            for i in 0..20_000usize {
                write_rec(mem, 12_288 + i, placed_records[i % 20]);
            }
            for i in 0..16usize {
                write_rec(mem, 34_816 + i, eph_record);
            }
            mem
        };

        let mem_legacy = build_heap();
        let mem_fast = build_heap();

        let sync_all_pages = |host: &HostHeap, mem: &[u8], fast: bool| {
            let start = mem.as_ptr() as u64;
            for p in 0..(SLOTS / PAGE_SLOTS) {
                let first = p * PAGE_SLOTS;
                let last = first + PAGE_SLOTS;
                let byte_start = first as usize * 32;
                let byte_end = last as usize * 32;
                host.sync_slots_bytes_with(
                    first..last,
                    start,
                    32,
                    start + byte_start as u64,
                    &mem[byte_start..byte_end],
                    fast,
                );
            }
        };

        // Initial warm sync so steady-state slots (including 20,000 placed-left-null) are populated.
        sync_all_pages(&host_legacy, mem_legacy, false);
        sync_all_pages(&host_fast, mem_fast, true);
        assert!(host_fast.kept(0), "benchmark buffer SRVs are real host views");
        assert!(!host_fast.kept(12_288), "placed textures remain null on sync");

        // Mutate ~1,700 slots/pass (740 real buffer views of 75 keys + 960 placed-null/cleared)
        // across 23 passes = ~39,100 stale slots + 1,507,328 total scanned slots.
        let mutate_pass = |mem: &mut [u8], pass: usize| {
            let toggle = pass & 1;
            for j in 0..740usize {
                write_rec(mem, 40_960 + j, buf_records[toggle * 75 + (j % 75)]);
            }
            for j in 0..960usize {
                let rec = if toggle == 0 {
                    placed_records[20 + (j % 20)]
                } else {
                    [0, 0, 0]
                };
                write_rec(mem, 45_056 + j, rec);
            }
        };

        // Two extra resources whose views are forgotten mid-run so each pass
        // exercises a real `VIEWS_FORGOT` epoch bump without affecting live slots.
        let bump_legacy = crate::resource::test_buffer(256);
        let bump_fast = crate::resource::test_buffer(256);
        for &r in &[bump_legacy, bump_fast] {
            let mut scratch = [0u32; 4];
            crate::descriptor::create_srv(
                std::ptr::null(),
                r,
                &srv_desc,
                scratch.as_mut_ptr() as u64,
            );
        }

        // Run legacy across 23 passes (1.5M scanned + ~39k stale) with a mid-run forget() epoch bump.
        let t_legacy = std::time::Instant::now();
        for pass in 0..PASSES {
            mutate_pass(mem_legacy, pass);
            if pass == PASSES / 2 {
                crate::descriptor::forget(bump_legacy as usize);
            }
            sync_all_pages(&host_legacy, mem_legacy, false);
        }
        let legacy_ms = t_legacy.elapsed().as_secs_f64() * 1000.0;

        // Run fast across identical 23 passes (1.5M scanned + ~39k stale) with identical forget() epoch bump.
        let t_fast = std::time::Instant::now();
        for pass in 0..PASSES {
            mutate_pass(mem_fast, pass);
            if pass == PASSES / 2 {
                crate::descriptor::forget(bump_fast as usize);
            }
            sync_all_pages(&host_fast, mem_fast, true);
        }
        let fast_ms = t_fast.elapsed().as_secs_f64() * 1000.0;

        // Both heaps' memory took the same writes: records that differ were
        // written by something else, and the heaps' states differ with them.
        if let Some(i) = mem_fast
            .chunks(32)
            .zip(mem_legacy.chunks(32))
            .position(|(f, l)| f != l)
        {
            panic!(
                "slot {i}'s record changed behind the bench: {:?} (fast) vs {:?} (legacy)",
                &mem_fast[i * 32..i * 32 + 12],
                &mem_legacy[i * 32..i * 32 + 12]
            );
        }

        // Verify identical final made/kept state across all 65,536 slots.
        for i in 0..SLOTS {
            assert_eq!(host_fast.made(i), host_legacy.made(i), "made mismatch at slot {i}");
            assert_eq!(host_fast.kept(i), host_legacy.kept(i), "kept mismatch at slot {i}");
        }

        // Now actually forget ephemeral and verify both clear slots 34,816..34,832.
        crate::descriptor::forget(ephemeral as usize);
        sync_all_pages(&host_legacy, mem_legacy, false);
        sync_all_pages(&host_fast, mem_fast, true);
        for i in 34_816..34_832u32 {
            assert!(!host_legacy.made(i));
            assert!(!host_fast.made(i));
            assert!(!host_fast.kept(i));
        }

        let speedup = legacy_ms / fast_ms.max(0.001);
        eprintln!(
            "bench_sync_slots_1_5m_fast_vs_legacy: slots={} stale=~39100 legacy={:.2}ms fast={:.2}ms speedup={:.2}x",
            SLOTS as usize * PASSES,
            legacy_ms,
            fast_ms,
            speedup
        );

        crate::object::release(bump_legacy);
        crate::object::release(bump_fast);
        crate::object::release(ephemeral);
        for b in buffers {
            crate::object::release(b);
        }
    }

    /// Mismatched descriptor kinds (a sampler record in a CBV/SRV/UAV heap or
    /// a CBV/SRV record in a sampler heap) must be cleared/skipped instead of
    /// calling `CreateSampler` / `CreateShaderResourceView` on the wrong heap
    /// handle type.
    #[test]
    fn mismatched_heap_kind_views_are_cleared_not_created() {
        let Some(host) = host() else {
            return; // no host GPU here
        };
        let Some(view_heap) = HostHeap::new(false, 4) else {
            return;
        };
        let Some(sampler_heap) = HostHeap::new(true, 4) else {
            return;
        };
        let null = std::ptr::null();
        let sampler_desc: [u32; 13] = [21, 3, 3, 3, 0, 1, 0, 0, 0, 0, 0, 0x7f7f_ffff, 0];
        let mut console_sampler = [0u32; 27];
        console_sampler[6..9].copy_from_slice(&[3, 3, 3]);

        // View heap: valid view in slot 0, then overwritten with sampler records.
        let mut view_slots = [[0u32; 8]; 4];
        let v0 = view_slots[0].as_mut_ptr() as u64;
        let v1 = view_slots[1].as_mut_ptr() as u64;
        crate::descriptor::create_cbv(null, &[0, 0, 256, 0], v0);
        assert!(view_heap.mirror(0, v0, false).is_some());
        assert!(view_heap.made(0));
        assert!(view_heap.made_is_nonzero(0));

        crate::descriptor::create_sampler(null, &sampler_desc, v0);
        assert!(view_heap.mirror(0, v0, false).is_none());
        assert!(!view_heap.made(0));
        assert!(!view_heap.made_is_nonzero(0));

        crate::descriptor::create_sampler_x(null, &console_sampler, v1);
        assert!(view_heap.mirror(1, v1, true).is_none());
        assert!(!view_heap.made(1));

        let dyn_slot = view_heap.dynamic(1).expect("a dynamic slot");
        view_heap.make_at(dyn_slot, &View::Sampler(sampler_desc));
        assert!(!view_heap.kept(dyn_slot));
        view_heap.release(view_heap.id(), dyn_slot, 1);

        // Sampler heap: valid sampler in slot 0, then overwritten with CBV/SRV records.
        let mut sampler_slots = [[0u32; 4]; 4];
        let s0 = sampler_slots[0].as_mut_ptr() as u64;
        let s1 = sampler_slots[1].as_mut_ptr() as u64;
        crate::descriptor::create_sampler(null, &sampler_desc, s0);
        assert!(sampler_heap.mirror(0, s0, false).is_some());
        assert!(sampler_heap.made(0));
        assert!(sampler_heap.made_is_nonzero(0));

        crate::descriptor::create_cbv(null, &[0, 0, 256, 0], s0);
        assert!(sampler_heap.mirror(0, s0, false).is_none());
        assert!(!sampler_heap.made(0));
        assert!(!sampler_heap.made_is_nonzero(0));

        crate::descriptor::create_srv(null, null, std::ptr::null(), s1);
        assert!(sampler_heap.mirror(1, s1, true).is_none());
        assert!(!sampler_heap.made(1));

        // SAFETY: plain device query.
        assert!(unsafe { host.device.GetDeviceRemovedReason() }.is_ok());
    }

    /// The unsigned twin of a 16-bit float or signed format has its layout,
    /// and only buffer views in such a format are shown as another view.
    #[test]
    fn only_16_bit_float_and_signed_buffer_views_get_an_unsigned_twin() {
        for (format, unsigned) in [(10, 12), (14, 12), (34, 36), (38, 36), (54, 57), (59, 57)] {
            assert_eq!(half_format(format), Some(unsigned));
            assert_eq!(
                crate::resource::element(format),
                crate::resource::element(unsigned),
                "format {format}"
            );
        }
        // The unsigned formats themselves, normalised 16-bit formats, and
        // wider and narrower ones.
        for other in [0, 12, 36, 57, 11, 13, 35, 37, 56, 58, 42, 17, 30, 61] {
            assert_eq!(half_format(other), None, "format {other}");
        }
        const BUFFER: u32 = 1;
        let words = |format: u32, dimension: u32| [format, dimension, 5768, 0, 0, 0, 64, 0, 0, 0];
        // R16G16B16A16_FLOAT: a buffer of halves a program moves as they are.
        assert_eq!(
            half_view(&View::Unordered(7, 0, Some(words(10, BUFFER)))),
            Some(View::Unordered(7, 0, Some(words(12, BUFFER))))
        );
        assert_eq!(
            half_view(&View::Shader(7, Some(words(54, BUFFER)))),
            Some(View::Shader(7, Some(words(57, BUFFER))))
        );
        let mut placed = [0u32; 16];
        placed[..10].copy_from_slice(&words(38, BUFFER));
        placed[10] = 0x1234_0000;
        let mut twin = placed;
        twin[0] = 36;
        let memory = [BUFFER, 0, 0, 0, 512, 0, 1, 0x1_0001, 0, 1, 0, 1, 0, 0];
        assert_eq!(
            half_view(&View::Placed(true, memory, placed)),
            Some(View::Placed(true, memory, twin))
        );
        // Unsigned 16-bit buffers, texture views (dimension 4: 2D), views
        // without a description, and constant buffers keep their own view.
        assert_eq!(
            half_view(&View::Unordered(7, 0, Some(words(12, BUFFER)))),
            None
        );
        assert_eq!(half_view(&View::Shader(7, Some(words(57, BUFFER)))), None);
        assert_eq!(half_view(&View::Unordered(7, 0, Some(words(10, 4)))), None);
        assert_eq!(half_view(&View::Shader(7, None)), None);
        assert_eq!(half_view(&View::Constant(0x1000, 256)), None);
    }

    /// Buffer views whose values the console would convert to and from halves
    /// are reported by `half_inexact`; 16-bit formats and narrow unsigned
    /// integer formats are not.
    #[test]
    fn views_whose_values_the_console_would_convert_are_named() {
        const BUFFER: u32 = 1;
        let words = |format: u32, dimension: u32| [format, dimension, 5768, 0, 0, 0, 64, 0, 0, 0];
        for exact in [10, 12, 14, 34, 36, 38, 54, 57, 59, 25, 30, 50, 62] {
            let view = View::Unordered(7, 0, Some(words(exact, BUFFER)));
            assert_eq!(half_inexact(&view), None, "format {exact}");
        }
        for inexact in [41, 42, 11, 28, 64, 0] {
            let view = View::Shader(7, Some(words(inexact, BUFFER)));
            assert_eq!(half_inexact(&view), Some(inexact));
        }
        let mut placed = [0u32; 16];
        placed[..10].copy_from_slice(&words(41, BUFFER));
        let memory = [BUFFER, 0, 0, 0, 512, 0, 1, 0x1_0001, 0, 1, 0, 1, 0, 0];
        assert_eq!(half_inexact(&View::Placed(true, memory, placed)), Some(41));
        // Not a buffer view.
        assert_eq!(half_inexact(&View::Shader(7, Some(words(41, 4)))), None);
        assert_eq!(half_inexact(&View::Constant(0x1000, 256)), None);
    }

    /// Moving all 65,536 16-bit patterns through an `R16G16B16A16_UINT`
    /// buffer view (the unsigned twin `half_view` gives a 16-bit float view)
    /// preserves every pattern bit-exactly when accessed as unsigned integers.
    #[test]
    fn halves_cross_an_unsigned_buffer_view_unchanged() {
        const MOVES: &str = r#"
RWBuffer<uint4> from : register(u0);
RWBuffer<uint4> to : register(u1);
[numthreads(64, 1, 1)]
void main(uint3 id : SV_DispatchThreadID) {
    uint4 h = from[id.x] & 0xffff;
    uint low = h.x | h.y << 16;
    uint high = h.z | h.w << 16;
    to[id.x] = uint4(low & 0xffff, low >> 16, high & 0xffff, high >> 16);
}
"#;
        const ELEMENTS: usize = 1 << 14;
        const BYTES: u64 = ELEMENTS as u64 * 8;
        const UNSIGNED: u32 = 12; // R16G16B16A16_UINT
        let Some(host) = host() else {
            return; // no host GPU
        };
        assert_eq!(half_format(10), Some(UNSIGNED));
        let uav_range = [crate::compute::uav(0, 2)];
        let (root, pso) = crate::compute::pipeline(
            MOVES,
            windows::core::s!("half_views_test"),
            0,
            &[crate::compute::Param::Table(&uav_range)],
        )
        .expect("half-views compute pipeline");
        let heap: ID3D12DescriptorHeap = {
            let desc = D3D12_DESCRIPTOR_HEAP_DESC {
                Type: D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV,
                NumDescriptors: 2,
                Flags: D3D12_DESCRIPTOR_HEAP_FLAG_SHADER_VISIBLE,
                NodeMask: 0,
            };
            // SAFETY: plain heap creation.
            unsafe { host.device.CreateDescriptorHeap(&desc) }.expect("a view heap")
        };
        let (from, filled) = host.shared_buffer(BYTES).expect("a buffer");
        let (to, result) = host.shared_buffer(BYTES).expect("a buffer");
        // SAFETY: both mappings hold `BYTES` bytes; the views are of live
        // buffers in slots of the heap made above; an open list.
        let unsigned = unsafe {
            let filled = std::slice::from_raw_parts_mut(filled.cast::<u16>(), ELEMENTS * 4);
            for (pattern, half) in filled.iter_mut().enumerate() {
                *half = pattern as u16;
            }
            std::ptr::write_bytes(result, 0xee, BYTES as usize);
            let view = D3D12_UNORDERED_ACCESS_VIEW_DESC {
                Format: DXGI_FORMAT(UNSIGNED as i32),
                ViewDimension: D3D12_UAV_DIMENSION_BUFFER,
                Anonymous: D3D12_UNORDERED_ACCESS_VIEW_DESC_0 {
                    Buffer: D3D12_BUFFER_UAV {
                        NumElements: ELEMENTS as u32,
                        ..Default::default()
                    },
                },
            };
            let start = heap.GetCPUDescriptorHandleForHeapStart();
            let step = host
                .device
                .GetDescriptorHandleIncrementSize(D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV)
                as usize;
            host.device
                .CreateUnorderedAccessView(&from, None::<&ID3D12Resource>, Some(&view), start);
            let second = D3D12_CPU_DESCRIPTOR_HANDLE {
                ptr: start.ptr + step,
            };
            host.device
                .CreateUnorderedAccessView(&to, None::<&ID3D12Resource>, Some(&view), second);
            let list = host.new_list().expect("a host list");
            list.list.SetDescriptorHeaps(&[Some(heap.clone())]);
            list.list.SetComputeRootSignature(&root);
            list.list.SetPipelineState(&pso);
            list.list
                .SetComputeRootDescriptorTable(0, heap.GetGPUDescriptorHandleForHeapStart());
            list.list.Dispatch(ELEMENTS as u32 / 64, 1, 1);
            list.close();
            host.execute(&[&list]);
            assert!(
                host.device.GetDeviceRemovedReason().is_ok(),
                "the device, after views in format {UNSIGNED}"
            );
            let result = std::slice::from_raw_parts(result.cast::<u16>(), ELEMENTS * 4);
            (0..=u16::MAX)
                .filter(|&p| result[usize::from(p)] == p)
                .count()
        };
        assert_eq!(
            unsigned, 65_536,
            "patterns unchanged, moved as unsigned integers"
        );
    }

    /// End-to-end test of [`buffer_srv`] converting a structured buffer SRV
    /// (`StructureByteStride = 680`, starting at `FirstElement = 1` = `+680`
    /// bytes, `8 mod 16`) into a raw SRV and reading it through a translated
    /// DXIL compute shader via MUBUF raw byte offset (`0xC4`), SMEM scalar
    /// byte offset (`0xC4`), and MUBUF structured element+field (`[2] + 12`).
    #[test]
    fn structured_buffer_srv_made_raw_reads_byte_and_element_offsets() {
        use crate::pipeline::{Shader, Translated, host::HostPipeline};
        use victory_shader::native::{ComputeIo, StageIo};
        use victory_shader::{
            Container, HwStage, NativeProgram, ResourceType, RootParam, RootType, UsedResource,
            emit, io, ir, resource, structure,
        };

        let Some(host) = host() else {
            return;
        };
        let mut sm = D3D12_FEATURE_DATA_SHADER_MODEL {
            HighestShaderModel: D3D_SHADER_MODEL_6_6,
        };
        // SAFETY: valid feature support query.
        let ok = unsafe {
            host.device.CheckFeatureSupport(
                D3D12_FEATURE_SHADER_MODEL,
                (&raw mut sm).cast(),
                size_of_val(&sm) as u32,
            )
        };
        if ok.is_err() || sm.HighestShaderModel.0 < D3D_SHADER_MODEL_6_6.0 {
            return;
        }

        let code_words: &[u32] = &[
            0xBE84_0300, // s_mov_b32 s4, s0
            0xBE85_0380, // s_mov_b32 s5, 0
            0xF408_0202,
            0xFA00_0000, // s_load_dwordx4 s[8:11], s[4:5], 0x0 (SRV slot 0)
            0xBE86_0301, // s_mov_b32 s6, s1
            0xBE87_0380, // s_mov_b32 s7, 0
            0xF408_0303,
            0xFA00_0000, // s_load_dwordx4 s[12:15], s[6:7], 0x0 (UAV slot 1)
            0xBF8C_C07F, // s_waitcnt lgkmcnt(0)
            0xF420_0404,
            0xFA00_00C4, // s_buffer_load_dword s16, s[8:11], 0xC4
            0xBF8C_C07F, // s_waitcnt lgkmcnt(0)
            0x7E08_0210, // v_mov_b32 v4, s16
            0xE030_00C4,
            0x8002_0100, // buffer_load_dword v1, off, s[8:11], 0 offset:196
            0x7E04_0282, // v_mov_b32 v2, 2
            0xE030_200C,
            0x8002_0302, // buffer_load_dword v3, v2, s[8:11], 0 idxen offset:12
            0xBF8C_3F70, // s_waitcnt vmcnt(0)
            0xE070_0000,
            0x8003_0100, // buffer_store_dword v1, off, s[12:15], 0 offset:0
            0xE070_0004,
            0x8003_0400, // buffer_store_dword v4, off, s[12:15], 0 offset:4
            0xE070_0008,
            0x8003_0300, // buffer_store_dword v3, off, s[12:15], 0 offset:8
            0xBF81_0000, // s_endpgm
        ];
        let native = NativeProgram {
            stage: HwStage::Cs,
            wave64: false,
            vgprs: 8,
            layout: (0..2)
                .map(|slot| RootParam {
                    root: RootType::Table,
                    table_offset_dwords: 0,
                    sgprs: 1,
                    base_register: 0,
                    user_data: slot,
                    slot,
                })
                .collect(),
            used: vec![
                UsedResource {
                    kind: ResourceType::Srv,
                    root: RootType::Table,
                    table_offset_dwords: 0,
                    lower_bound: 0,
                    stride: 680,
                    dwords: 4,
                    user_data: 0,
                    slot: 0,
                },
                UsedResource {
                    kind: ResourceType::Uav,
                    root: RootType::Table,
                    table_offset_dwords: 0,
                    lower_bound: 0,
                    stride: 0,
                    dwords: 4,
                    user_data: 1,
                    slot: 1,
                },
            ],
            io: StageIo::Cs(ComputeIo {
                rsrc2: 2 << 1,
                threads: [1, 1, 1],
            }),
            code: code_words.iter().flat_map(|w| w.to_le_bytes()).collect(),
        };
        let mut container = vec![0; 32];
        container[..4].copy_from_slice(b"DXBC");
        container[20..24].copy_from_slice(&1u32.to_le_bytes());
        container[24..28].copy_from_slice(&32u32.to_le_bytes());
        let c = Container::parse(&container).unwrap();
        let p = ir::build(&native.code, native.wave64, native.vgprs).unwrap();
        let st = structure::structure(&p).unwrap();
        let res = resource::track(&p, &native.layout, native.stage, Some(&st)).unwrap();
        let io = io::raise(&p, &native).unwrap();
        let hs = emit::emit(&c, &native, &p, &st, &io, &res).unwrap();
        let ud_spec = hs.user_data.clone();
        let shader = Shader::for_test(
            "compute",
            Translated {
                host: hs,
                used: native.used,
                used_written: Vec::new(),
                write_unknown: true,
                threads: None,
            },
        );
        let Some(hp) = HostPipeline::new(&[shader], None, |_, _| {}) else {
            panic!("synthetic structured-SRV pipeline");
        };

        // Allocate 8 KiB of title memory (2048 dwords) filled with distinct words.
        let bytes_len = 8192usize;
        let mut backing: Vec<u32> = (0..(bytes_len / 4) as u32)
            .map(|i| 0x2000_0000 + i)
            .collect();
        let mem = backing.as_mut_ptr();
        crate::shadow::register(mem as usize, mem as u64, bytes_len as u64);

        // SAFETY: shader-visible heap with 2 slots.
        let d3d_heap: ID3D12DescriptorHeap = unsafe {
            host.device
                .CreateDescriptorHeap(&D3D12_DESCRIPTOR_HEAP_DESC {
                    Type: D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV,
                    NumDescriptors: 2,
                    Flags: D3D12_DESCRIPTOR_HEAP_FLAG_SHADER_VISIBLE,
                    ..Default::default()
                })
                .unwrap()
        };
        // SAFETY: live descriptor heap.
        let cpu = unsafe { d3d_heap.GetCPUDescriptorHandleForHeapStart() };
        // SAFETY: plain device query.
        let step = unsafe {
            host.device
                .GetDescriptorHandleIncrementSize(D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV)
        } as usize;

        // Structured SRV: stride = 680, FirstElement = 1 (+680B = +170 dwords), NumElements = 4.
        let srv_desc = D3D12_SHADER_RESOURCE_VIEW_DESC {
            Format: DXGI_FORMAT_UNKNOWN,
            ViewDimension: D3D12_SRV_DIMENSION_BUFFER,
            Shader4ComponentMapping: D3D12_DEFAULT_SHADER_4_COMPONENT_MAPPING,
            Anonymous: D3D12_SHADER_RESOURCE_VIEW_DESC_0 {
                Buffer: D3D12_BUFFER_SRV {
                    FirstElement: 1,
                    NumElements: 4,
                    StructureByteStride: 680,
                    Flags: D3D12_BUFFER_SRV_FLAG_NONE,
                },
            },
        };
        let _kept = buffer_srv(mem as u64, bytes_len as u64, srv_desc, cpu)
            .expect("buffer_srv succeeds")
            .expect("holds shadow buffer");

        let list = host.new_list().unwrap();
        let shadow = crate::shadow::find(mem as u64, bytes_len as u64).expect("registered shadow");
        shadow.refresh(&list, mem as u64, bytes_len as u64);

        let mut uav_buf: Option<ID3D12Resource> = None;
        // SAFETY: valid default UAV buffer description.
        unsafe {
            host.device
                .CreateCommittedResource(
                    &D3D12_HEAP_PROPERTIES {
                        Type: D3D12_HEAP_TYPE_DEFAULT,
                        ..Default::default()
                    },
                    D3D12_HEAP_FLAG_NONE,
                    &D3D12_RESOURCE_DESC {
                        Dimension: D3D12_RESOURCE_DIMENSION_BUFFER,
                        Width: 256,
                        Height: 1,
                        DepthOrArraySize: 1,
                        MipLevels: 1,
                        SampleDesc: DXGI_SAMPLE_DESC {
                            Count: 1,
                            Quality: 0,
                        },
                        Layout: D3D12_TEXTURE_LAYOUT_ROW_MAJOR,
                        Flags: D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS,
                        ..Default::default()
                    },
                    D3D12_RESOURCE_STATE_COMMON,
                    None,
                    &mut uav_buf,
                )
                .unwrap();
        }
        let uav_buf = uav_buf.unwrap();
        let uav_desc = D3D12_UNORDERED_ACCESS_VIEW_DESC {
            Format: DXGI_FORMAT_R32_TYPELESS,
            ViewDimension: D3D12_UAV_DIMENSION_BUFFER,
            Anonymous: D3D12_UNORDERED_ACCESS_VIEW_DESC_0 {
                Buffer: D3D12_BUFFER_UAV {
                    NumElements: 8,
                    Flags: D3D12_BUFFER_UAV_FLAG_RAW,
                    ..Default::default()
                },
            },
        };
        // SAFETY: valid UAV slot 1.
        unsafe {
            host.device.CreateUnorderedAccessView(
                &uav_buf,
                None,
                Some(&uav_desc),
                D3D12_CPU_DESCRIPTOR_HANDLE {
                    ptr: cpu.ptr + step,
                },
            );
        }

        let mut ud = vec![0u32; hp.stages[0].user_data.map_or(1, |(_, n)| n as usize)];
        for u in &ud_spec {
            ud[u.index as usize] = 32 * u32::from(u.slot);
        }
        let (readback, rb_ptr) = list.readback(host, 256).unwrap();
        // SAFETY: open list with heap, root signature, constants, and PSO bound.
        unsafe {
            list.list.SetDescriptorHeaps(&[Some(d3d_heap.clone())]);
            list.list.SetComputeRootSignature(&hp.root);
            list.list
                .SetComputeRoot32BitConstants(0, ud.len() as u32, ud.as_ptr().cast(), 0);
            list.list.SetPipelineState(&hp.pso);
            list.list.Dispatch(1, 1, 1);
            list.list.ResourceBarrier(&[crate::host::transition(
                &uav_buf,
                D3D12_RESOURCE_STATE_UNORDERED_ACCESS,
                D3D12_RESOURCE_STATE_COPY_SOURCE,
            )]);
            list.list.CopyResource(&readback, &uav_buf);
        }
        list.close();
        host.execute(&[&list]);

        // SAFETY: GPU execution finished; read 3 dwords from mapped readback.
        let got = unsafe { *rb_ptr.cast::<[u32; 3]>() };
        // SAFETY: plain device query.
        assert!(unsafe { host.device.GetDeviceRemovedReason() }.is_ok());
        crate::shadow::unregister(mem as usize, mem as u64);
        drop(backing);

        let expected = [
            0x2000_0000 + 170 + 49,  // MUBUF raw @ 0xC4
            0x2000_0000 + 170 + 49,  // SMEM scalar raw @ 0xC4
            0x2000_0000 + 170 + 343, // MUBUF structured [2] + 12
        ];
        assert_eq!(got, expected);
    }
}
