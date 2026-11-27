//! GPU-to-CPU tiled texture readback (`VICTORY_TEXTURE_BACK=1`).
//!
//! On unified-memory consoles, textures written by GPU shaders (compute UAVs
//! or render targets) reside directly in CPU-visible tiled memory. On a
//! discrete or translated host, GPU writes remain in the texture's host copy.
//! When enabled via `VICTORY_TEXTURE_BACK`, command lists that write eligible
//! 2D textures copy them into a host readback buffer at `HostList::close()`,
//! and after the host GPU fence completes, tile the texels back into the
//! texture's title CPU memory in the console's native tiled layout.
//!
//! Each pending readback carries an allocation [`Ticket`] invalidated if the
//! title releases the texture, replaces the allocation with a new texture, or
//! remaps the underlying title memory range before the readback lands. Only
//! matching allocations are invalidated; unrelated pending readbacks still
//! land cleanly.

use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Mutex, OnceLock, Weak};

#[cfg(windows)]
use std::sync::atomic::AtomicU64;
#[cfg(windows)]
use windows::Win32::Graphics::Direct3D12::*;

#[cfg(windows)]
use crate::object::This;

/// Default texture kinds enabled by `VICTORY_TEXTURE_BACK=1`:
/// `(width, height, dxgi_format)` for 2D height/displacement float surfaces.
const DEFAULT_SPEC: &str = "256x256f41,256x256f16,64x64f16,128x128f36";

/// Parses a `VICTORY_TEXTURE_BACK` value (`"1"` for defaults, `"0"` or empty
/// for disabled, or comma-separated `<width>x<height>f<format>` specs).
pub(crate) fn parse_kinds(spec: &str) -> Vec<(u64, u32, u32)> {
    let trimmed = spec.trim();
    if trimmed.is_empty() || trimmed == "0" {
        return Vec::new();
    }
    let list = if trimmed == "1" {
        DEFAULT_SPEC
    } else {
        trimmed
    };
    list.split(',')
        .filter_map(|entry| {
            let (size, format) = entry.trim().split_once('f')?;
            let (w, h) = size.split_once('x')?;
            Some((w.parse().ok()?, h.parse().ok()?, format.parse().ok()?))
        })
        .collect()
}

#[cfg(test)]
static TEST_KINDS: Mutex<Option<Vec<(u64, u32, u32)>>> = Mutex::new(None);
#[cfg(test)]
static TEST_KINDS_LOCK: Mutex<()> = Mutex::new(());

/// Active texture kinds copied back (`(width, height, format)`).
fn active_kinds() -> Vec<(u64, u32, u32)> {
    #[cfg(test)]
    if let Some(overridden) = TEST_KINDS
        .lock()
        .unwrap_or_else(|e| e.into_inner())
        .as_ref()
    {
        return overridden.clone();
    }
    static KINDS: OnceLock<Vec<(u64, u32, u32)>> = OnceLock::new();
    KINDS
        .get_or_init(|| {
            let kinds = std::env::var("VICTORY_TEXTURE_BACK")
                .map(|s| parse_kinds(&s))
                .unwrap_or_default();
            if !kinds.is_empty() {
                tracing::info!(
                    ?kinds,
                    "GPU-written 2D textures matching these kinds will be read back to title CPU memory (VICTORY_TEXTURE_BACK)"
                );
            }
            kinds
        })
        .clone()
}

/// Whether title texture `resource` matches an enabled readback kind.
#[cfg(windows)]
pub(crate) fn wanted(resource: This) -> bool {
    let kinds = active_kinds();
    !kinds.is_empty() && crate::resource::texture_kind(resource).is_some_and(|k| kinds.contains(&k))
}

/// An outstanding readback targeting a specific title allocation `[at, end)`.
/// Resource release, replacement, or memory remap invalidates only tickets
/// matching the changed allocation.
struct Ticket {
    owner: usize,
    at: usize,
    end: usize,
    valid: AtomicBool,
    landed: AtomicBool,
}

static TICKETS: Mutex<Vec<Weak<Ticket>>> = Mutex::new(Vec::new());
static TRACKING: AtomicBool = AtomicBool::new(false);

fn track(owner: usize, at: usize, end: usize) -> Arc<Ticket> {
    let ticket = Arc::new(Ticket {
        owner,
        at,
        end,
        valid: AtomicBool::new(true),
        landed: AtomicBool::new(false),
    });
    let mut tickets = TICKETS.lock().unwrap_or_else(|e| e.into_inner());
    tickets.retain(|weak| weak.strong_count() != 0);
    tickets.push(Arc::downgrade(&ticket));
    TRACKING.store(true, Ordering::Release);
    ticket
}

fn invalidate(matches: impl Fn(&Ticket) -> bool) {
    if !TRACKING.load(Ordering::Acquire) {
        return;
    }
    let mut tickets = TICKETS.lock().unwrap_or_else(|e| e.into_inner());
    tickets.retain(|weak| {
        let Some(ticket) = weak.upgrade() else {
            return false;
        };
        if matches(&ticket) {
            ticket.valid.store(false, Ordering::Release);
        }
        true
    });
}

/// Invalidates pending texture readbacks owned by the released title resource.
#[cfg(any(windows, test))]
pub(crate) fn released(owner: usize) {
    invalidate(|ticket| ticket.owner == owner);
}

/// Invalidates pending texture readbacks whose title allocation is replaced or
/// overlapped by a newly created texture at `[at, at + len)`.
#[cfg(any(windows, test))]
pub(crate) fn replaced(at: usize, len: usize) {
    if len == 0 {
        invalidate(|ticket| ticket.at == at);
    } else {
        let end = at.saturating_add(len);
        invalidate(|ticket| ticket.at == at || (at < ticket.end && ticket.at < end));
    }
}

/// Invalidates pending texture readbacks overlapping a remapped or unmapped
/// title memory region `[at, at + len)`.
#[cfg(any(windows, test))]
pub(crate) fn remapped(at: usize, len: usize) {
    let end = at.saturating_add(len);
    invalidate(|ticket| at < ticket.end && ticket.at < end);
}

/// Returns the sorted, coalesced title-memory destination ranges `[start, end)`
/// for all currently valid, unlanded texture readbacks. Async or late
/// submissions can query this to keep CPU reads/writes coherent with in-flight
/// readbacks.
#[allow(dead_code)]
pub(crate) fn pending_ranges() -> Vec<(usize, usize)> {
    if !TRACKING.load(Ordering::Acquire) {
        return Vec::new();
    }
    let mut ranges = Vec::new();
    let mut tickets = TICKETS.lock().unwrap_or_else(|e| e.into_inner());
    tickets.retain(|weak| {
        let Some(ticket) = weak.upgrade() else {
            return false;
        };
        if ticket.valid.load(Ordering::Acquire) && !ticket.landed.load(Ordering::Acquire) {
            ranges.push((ticket.at, ticket.end));
        }
        true
    });
    ranges.sort_unstable();
    let mut merged: Vec<(usize, usize)> = Vec::with_capacity(ranges.len());
    for (start, end) in ranges {
        if let Some(last) = merged.last_mut()
            && start <= last.1
        {
            last.1 = last.1.max(end);
        } else {
            merged.push((start, end));
        }
    }
    merged
}

/// A texture scheduled to be copied into a readback buffer when its command
/// list closes.
#[cfg(windows)]
pub(crate) struct Target {
    pub(crate) copy: Arc<crate::resource::HostTexture>,
    layout: victory_formats::texture::Layout,
    ticket: Arc<Ticket>,
}

#[cfg(windows)]
impl Target {
    pub(crate) fn new(resource: This) -> Option<Self> {
        let copy = crate::resource::host_copy(resource)?;
        let (layout, at, size) = crate::resource::title_texture(resource)?;
        let end = at.checked_add(size).filter(|&end| at != 0 && end > at)?;
        let ticket = track(resource as usize, at, end);
        Some(Self {
            copy,
            layout,
            ticket,
        })
    }

    pub(crate) fn owner(&self) -> usize {
        self.ticket.owner
    }

    pub(crate) fn is_valid(&self) -> bool {
        self.ticket.valid.load(Ordering::Acquire)
    }
}

/// Maximum number of idle readback buffers retained in the pool.
#[cfg(any(windows, test))]
const MAX_IDLE_BUFFERS: usize = 64;
/// Maximum total bytes of idle readback buffers retained in the pool (16 MiB).
#[cfg(any(windows, test))]
const MAX_IDLE_BYTES: u64 = 16 << 20;
/// Readback buffer size alignment (64 KiB, matching D3D12 default placement alignment).
#[cfg(any(windows, test))]
const READBACK_ALIGNMENT: u64 = 64 << 10;
#[cfg(windows)]
const _: () = assert!(READBACK_ALIGNMENT == D3D12_DEFAULT_RESOURCE_PLACEMENT_ALIGNMENT as u64);

/// Rounds a readback allocation request up to [`READBACK_ALIGNMENT`].
#[cfg(any(windows, test))]
fn align_readback_bytes(bytes: u64) -> Option<u64> {
    bytes.max(1).checked_next_multiple_of(READBACK_ALIGNMENT)
}

#[cfg(any(windows, test))]
struct IdleEntry<T> {
    resource: T,
    bytes: u64,
}

/// Bounded best-fit pool of idle readback buffers.
#[cfg(any(windows, test))]
struct IdlePool<T> {
    entries: Vec<IdleEntry<T>>,
    total_bytes: u64,
}

#[cfg(any(windows, test))]
impl<T> IdlePool<T> {
    const fn new() -> Self {
        Self {
            entries: Vec::new(),
            total_bytes: 0,
        }
    }

    /// Takes the smallest idle buffer in `[aligned, 2 * aligned]` so small
    /// readbacks never hijack idle multi-megabyte buffers.
    fn take(&mut self, aligned: u64) -> Option<(T, u64)> {
        let max_bytes = aligned.saturating_mul(2);
        let best = self
            .entries
            .iter()
            .enumerate()
            .filter(|(_, entry)| entry.bytes >= aligned && entry.bytes <= max_bytes)
            .min_by_key(|(_, entry)| entry.bytes)
            .map(|(idx, _)| idx)?;
        let entry = self.entries.swap_remove(best);
        self.total_bytes = self.total_bytes.saturating_sub(entry.bytes);
        Some((entry.resource, entry.bytes))
    }

    /// Stores `resource` if within count and byte caps; otherwise returns it
    /// to the caller for deferred retirement outside the pool lock.
    fn put(&mut self, resource: T, bytes: u64) -> Option<T> {
        if self.entries.len() < MAX_IDLE_BUFFERS
            && self
                .total_bytes
                .checked_add(bytes)
                .is_some_and(|next| next <= MAX_IDLE_BYTES)
        {
            self.total_bytes += bytes;
            self.entries.push(IdleEntry { resource, bytes });
            None
        } else {
            Some(resource)
        }
    }

    /// Drains all idle entries and resets tracked pool bytes to zero.
    fn drain(&mut self) -> Vec<T> {
        self.total_bytes = 0;
        self.entries.drain(..).map(|entry| entry.resource).collect()
    }

    #[cfg(test)]
    fn stats(&self) -> (usize, u64) {
        (self.entries.len(), self.total_bytes)
    }
}

#[cfg(windows)]
static IDLE: Mutex<IdlePool<ID3D12Resource>> = Mutex::new(IdlePool::new());

/// Pooled `D3D12_HEAP_TYPE_READBACK` buffer paired with its aligned capacity.
#[cfg(windows)]
struct Readback {
    resource: ID3D12Resource,
    bytes: u64,
}

#[cfg(windows)]
impl Readback {
    /// Acquires a pooled readback buffer using bounded best-fit, or allocates
    /// a fresh 64 KiB-aligned readback buffer on a pool miss.
    fn acquire(host: &crate::host::Host, bytes: u64) -> Option<Self> {
        let aligned = align_readback_bytes(bytes)?;
        if let Some((resource, capacity)) = IDLE
            .lock()
            .unwrap_or_else(|e| e.into_inner())
            .take(aligned)
        {
            return Some(Self {
                resource,
                bytes: capacity,
            });
        }
        let resource = host.read_back_buffer(aligned)?;
        Some(Self {
            resource,
            bytes: aligned,
        })
    }

    /// Returns this buffer to the idle pool if within caps, else retires it
    /// through [`crate::host::retire`] so D3DMetal `bury_lag` is respected.
    fn recycle(self) {
        let overflow = IDLE
            .lock()
            .unwrap_or_else(|e| e.into_inner())
            .put(self.resource, self.bytes);
        if let Some(resource) = overflow {
            crate::host::retire(resource);
        }
    }

    fn resource(&self) -> &ID3D12Resource {
        &self.resource
    }
}

/// Drains idle pooled readback buffers and retires them through
/// [`crate::host::retire_all`] so D3DMetal `bury_lag` is respected.
#[cfg(windows)]
pub(crate) fn trim_idle() {
    let evicted = IDLE.lock().unwrap_or_else(|e| e.into_inner()).drain();
    crate::host::retire_all(evicted);
}

/// Recorded host readback buffer and console layout metadata for one texture.
#[cfg(windows)]
pub(crate) struct Back {
    buffer: Readback,
    footprints: Vec<D3D12_PLACED_SUBRESOURCE_FOOTPRINT>,
    rows: Vec<u32>,
    mips: u32,
    layout: victory_formats::texture::Layout,
    ticket: Arc<Ticket>,
}

#[cfg(windows)]
impl Back {
    /// Destination title-memory byte range `[start, end)`.
    pub(crate) fn range(&self) -> (usize, usize) {
        (self.ticket.at, self.ticket.end)
    }

    /// Whether the destination allocation is still valid (not released,
    /// replaced, or remapped).
    #[allow(dead_code)]
    pub(crate) fn is_valid(&self) -> bool {
        self.ticket.valid.load(Ordering::Acquire)
    }

    /// Returns the readback buffer to the idle pool (or retires it on cap
    /// overflow). Call only after the command list has finished executing and
    /// its allocator and list have been reset.
    pub(crate) fn recycle(self) {
        self.buffer.recycle();
    }

    #[cfg(test)]
    pub(crate) fn raw_resource_ptr(&self) -> usize {
        windows::core::Interface::as_raw(self.buffer.resource()) as usize
    }
}

/// Records `CopyTextureRegion` calls from `target.copy` (already transitioned
/// to `D3D12_RESOURCE_STATE_COPY_SOURCE`) into a pooled or newly allocated
/// host readback buffer.
#[cfg(windows)]
pub(crate) fn record(list: &ID3D12GraphicsCommandList, target: Target) -> Option<Back> {
    if !target.ticket.valid.load(Ordering::Acquire) {
        return None;
    }
    let host = crate::host::host()?;
    let texture = &target.copy.texture;
    // SAFETY: `target.copy` holds a live `ID3D12Resource`.
    let desc = unsafe { texture.GetDesc() };
    let mips = u32::from(desc.MipLevels.max(1));
    let subresources = mips * u32::from(desc.DepthOrArraySize.max(1));
    let n = subresources as usize;
    let mut footprints = vec![D3D12_PLACED_SUBRESOURCE_FOOTPRINT::default(); n];
    let mut rows = vec![0u32; n];
    let mut total = 0u64;
    // SAFETY: `footprints` and `rows` have `n == subresources` elements.
    unsafe {
        host.device.GetCopyableFootprints(
            &desc,
            0,
            subresources,
            0,
            Some(footprints.as_mut_ptr()),
            Some(rows.as_mut_ptr()),
            None,
            Some(&mut total),
        );
    }
    // Ensure the readback buffer includes full row-end padding for the final
    // subresource so `Layout::tile` can read pitched slices safely.
    for (footprint, &row_count) in footprints.iter().zip(&rows) {
        let f = footprint.Footprint;
        let pitched = u64::from(f.RowPitch)
            .checked_mul(u64::from(row_count))?
            .checked_mul(u64::from(f.Depth))?;
        total = total.max(footprint.Offset.checked_add(pitched)?);
    }
    let buffer = Readback::acquire(host, total)?;
    for (i, footprint) in footprints.iter().enumerate() {
        let to = D3D12_TEXTURE_COPY_LOCATION {
            // SAFETY: borrowed COM pointer; `buffer` outlives the call.
            pResource: unsafe { std::mem::transmute_copy(buffer.resource()) },
            Type: D3D12_TEXTURE_COPY_TYPE_PLACED_FOOTPRINT,
            Anonymous: D3D12_TEXTURE_COPY_LOCATION_0 {
                PlacedFootprint: *footprint,
            },
        };
        let from = D3D12_TEXTURE_COPY_LOCATION {
            // SAFETY: borrowed COM pointer; `texture` outlives the call.
            pResource: unsafe { std::mem::transmute_copy(texture) },
            Type: D3D12_TEXTURE_COPY_TYPE_SUBRESOURCE_INDEX,
            Anonymous: D3D12_TEXTURE_COPY_LOCATION_0 {
                SubresourceIndex: i as u32,
            },
        };
        // SAFETY: open command list; `texture` is in `COPY_SOURCE` and `buffer`
        // is in `COPY_DEST`.
        unsafe { list.CopyTextureRegion(&to, 0, 0, 0, &from, None) };
    }
    Some(Back {
        buffer,
        footprints,
        rows,
        mips,
        layout: target.layout,
        ticket: target.ticket,
    })
}

/// After the host GPU fence completes, retiles the readback buffer into the
/// texture's title CPU allocation unless that allocation was invalidated.
#[cfg(windows)]
pub(crate) fn land(back: &Back) {
    static COPIED: AtomicU64 = AtomicU64::new(0);
    static STALE: AtomicU64 = AtomicU64::new(0);
    // Lock order: `TICKETS` strictly before `watch::guards()` (acquired inside
    // `host::copy_mapped` via `watch::writing`). `watch::title_memory_remapped`
    // calls `texback::remapped` before locking `guards()`, preserving this DAG.
    let _ticket_guard = TICKETS.lock().unwrap_or_else(|e| e.into_inner());
    if !back.ticket.valid.load(Ordering::Acquire) {
        let skipped = STALE.fetch_add(1, Ordering::Relaxed) + 1;
        if skipped.is_power_of_two() {
            tracing::info!(
                skipped,
                "skipping stale texture readback whose title allocation changed"
            );
        }
        return;
    }
    let (at, end) = back.range();
    let size = end - at;
    let mut tiled = vec![0u8; size];
    // Preserve title bytes outside the tiled subresource footprint (tail padding).
    if !crate::host::read_into(at as *const u8, &mut tiled) {
        return;
    }
    let mut data = std::ptr::null_mut();
    // SAFETY: our readback buffer written by the completed GPU submission.
    unsafe {
        if back.buffer.resource().Map(0, None, Some(&mut data)).is_err() || data.is_null() {
            return;
        }
    }
    for (i, (footprint, &row_count)) in back.footprints.iter().zip(&back.rows).enumerate() {
        let (mip, slice) = (i as u32 % back.mips, i as u32 / back.mips);
        let f = footprint.Footprint;
        let pitch = f.RowPitch as usize;
        let len = pitch * row_count as usize * f.Depth as usize;
        // SAFETY: subresource slice within the mapped readback buffer.
        let linear = unsafe {
            std::slice::from_raw_parts(data.cast::<u8>().add(footprint.Offset as usize), len)
        };
        if let Err(e) = back
            .layout
            .tile(&mut tiled, mip, slice, linear, pitch, pitch * row_count as usize)
        {
            static ONCE: std::sync::Once = std::sync::Once::new();
            ONCE.call_once(|| {
                tracing::warn!(?e, mip, slice, "texture readback could not be retiled")
            });
            // SAFETY: mapped above.
            unsafe { back.buffer.resource().Unmap(0, None) };
            return;
        }
    }
    // SAFETY: mapped above.
    unsafe { back.buffer.resource().Unmap(0, None) };
    // `copy_mapped(..., true)` guards the destination range with
    // `watch::writing` so write-watched pages stay coherent without faulting.
    crate::host::copy_mapped(at as *mut u8, tiled.as_ptr(), size, true);
    back.ticket.landed.store(true, Ordering::Release);
    let copied = COPIED.fetch_add(1, Ordering::Relaxed) + 1;
    if copied == 1 || copied.is_multiple_of(128) {
        tracing::info!(copied, bytes = size, "texture readback landed");
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[cfg(windows)]
    use crate::backend::{HostGpu, HostMemory};

    struct KindGuard {
        _lock: std::sync::MutexGuard<'static, ()>,
    }

    impl KindGuard {
        fn set(kinds: Vec<(u64, u32, u32)>) -> Self {
            let lock = TEST_KINDS_LOCK.lock().unwrap_or_else(|e| e.into_inner());
            *TEST_KINDS.lock().unwrap_or_else(|e| e.into_inner()) = Some(kinds);
            Self { _lock: lock }
        }
    }

    impl Drop for KindGuard {
        fn drop(&mut self) {
            *TEST_KINDS.lock().unwrap_or_else(|e| e.into_inner()) = None;
        }
    }

    #[test]
    fn env_spec_parsing_and_defaults() {
        assert!(parse_kinds("").is_empty());
        assert!(parse_kinds("0").is_empty());
        assert_eq!(
            parse_kinds("1"),
            vec![
                (256, 256, 41),
                (256, 256, 16),
                (64, 64, 16),
                (128, 128, 36),
            ]
        );
        assert_eq!(
            parse_kinds(" 64x64f41 , bad , 128x32f16 "),
            vec![(64, 64, 41), (128, 32, 16)]
        );
        let _off = KindGuard::set(Vec::new());
        assert!(active_kinds().is_empty());
        drop(_off);
        let _on = KindGuard::set(parse_kinds("1"));
        assert_eq!(active_kinds().len(), 4);
    }

    #[test]
    fn only_readbacks_for_changed_allocations_are_invalidated() {
        let first = track(901, 0x10_0000, 0x10_1000);
        let next = track(902, 0x10_1000, 0x10_2000);
        let overlap = track(903, 0x10_2000, 0x10_4000);
        let far = track(904, 0x10_8000, 0x10_9000);

        let pending = pending_ranges();
        assert!(pending.contains(&(0x10_0000, 0x10_4000)));
        assert!(pending.contains(&(0x10_8000, 0x10_9000)));

        // Remap overlaps only `first`.
        remapped(0x0f_ff00, 0x200);
        assert!(!first.valid.load(Ordering::Acquire));
        assert!(next.valid.load(Ordering::Acquire));
        assert!(overlap.valid.load(Ordering::Acquire));
        assert!(far.valid.load(Ordering::Acquire));

        // Exact-start replacement invalidates only `next`.
        replaced(0x10_1000, 0);
        assert!(!next.valid.load(Ordering::Acquire));
        assert!(overlap.valid.load(Ordering::Acquire));
        assert!(far.valid.load(Ordering::Acquire));

        // Overlapping range replacement invalidates `overlap` without touching `far`.
        replaced(0x10_3000, 0x1000);
        assert!(!overlap.valid.load(Ordering::Acquire));
        assert!(far.valid.load(Ordering::Acquire));

        // Release invalidates `far` while a freshly tracked ticket stays valid.
        released(904);
        assert!(!far.valid.load(Ordering::Acquire));
        let fresh = track(905, 0x10_1000, 0x10_2000);
        assert!(fresh.valid.load(Ordering::Acquire));
        let pending_after = pending_ranges();
        assert!(pending_after.contains(&(0x10_1000, 0x10_2000)));
        assert!(!pending_after.iter().any(|&(a, b)| a <= 0x10_8000 && 0x10_9000 <= b));
    }

    #[cfg(windows)]
    fn make_placed_2d_r32f(width: u64, height: u32, backing: *mut u8) -> This {
        let mut out: This = std::ptr::null();
        let hr = crate::resource::create_placed_2d_for_test(
            backing,
            width,
            height,
            41,  // DXGI_FORMAT_R32_FLOAT
            0x4, // ALLOW_UNORDERED_ACCESS
            &mut out,
        );
        assert_eq!(hr, crate::object::S_OK, "create placed 2D R32_FLOAT texture");
        out
    }

    #[cfg(windows)]
    fn release_resource(res: This) {
        type Release = extern "system" fn(This) -> u32;
        crate::object::slot::<Release>(res, 0x10)(res);
    }

    #[cfg(windows)]
    fn record_gpu_fill_r32f(
        host: &crate::host::Host,
        list: &crate::host::HostList,
        res: This,
        pixels: &[f32],
    ) {
        let copy = crate::resource::host_copy(res).expect("host copy");
        // SAFETY: live host texture.
        let desc = unsafe { copy.texture.GetDesc() };
        let mut fp = D3D12_PLACED_SUBRESOURCE_FOOTPRINT::default();
        let mut rows = 0u32;
        let mut total = 0u64;
        // SAFETY: single subresource footprint query.
        unsafe {
            host.device.GetCopyableFootprints(
                &desc,
                0,
                1,
                0,
                Some(&mut fp),
                Some(&mut rows),
                None,
                Some(&mut total),
            );
        }
        let pitch = fp.Footprint.RowPitch as usize;
        let w = desc.Width as usize;
        let h = desc.Height as usize;
        let (upload, ptr) = host.upload_buffer(total).expect("upload buffer");
        // SAFETY: `ptr` points to `total` mapped bytes of `upload`.
        unsafe {
            for y in 0..h {
                let row_dst = ptr.add(fp.Offset as usize + y * pitch).cast::<f32>();
                std::ptr::copy_nonoverlapping(pixels[y * w..(y + 1) * w].as_ptr(), row_dst, w);
            }
        }
        if let Some(barrier) = list.move_texture(&copy, 0x400 /* COPY_DEST */) {
            // SAFETY: open list, valid barrier.
            unsafe { list.list.ResourceBarrier(&[barrier]) };
        }
        let dst_loc = D3D12_TEXTURE_COPY_LOCATION {
            // SAFETY: borrowed COM pointer.
            pResource: unsafe { std::mem::transmute_copy(&copy.texture) },
            Type: D3D12_TEXTURE_COPY_TYPE_SUBRESOURCE_INDEX,
            Anonymous: D3D12_TEXTURE_COPY_LOCATION_0 {
                SubresourceIndex: 0,
            },
        };
        let src_loc = D3D12_TEXTURE_COPY_LOCATION {
            // SAFETY: borrowed COM pointer.
            pResource: unsafe { std::mem::transmute_copy(&upload) },
            Type: D3D12_TEXTURE_COPY_TYPE_PLACED_FOOTPRINT,
            Anonymous: D3D12_TEXTURE_COPY_LOCATION_0 {
                PlacedFootprint: fp,
            },
        };
        // SAFETY: open list; texture is in `COPY_DEST`, upload is in `GENERIC_READ`.
        unsafe { list.list.CopyTextureRegion(&dst_loc, 0, 0, 0, &src_loc, None) };
        list.keep_upload_for_test(upload);
    }

    #[cfg(windows)]
    #[test]
    fn gpu_written_tiled_texture_lands_exact_texels_in_title_memory() {
        let Some(host) = crate::host::host() else {
            return;
        };
        let _guard = KindGuard::set(vec![(64, 64, 41)]);
        let mut backing = vec![0xCCu8; 64 * 1024];
        let res = make_placed_2d_r32f(64, 64, backing.as_mut_ptr());
        let (layout, at, size) = crate::resource::title_texture(res).expect("title texture layout");
        assert_eq!(at, backing.as_ptr() as usize);
        assert!(size <= backing.len());

        let expected: Vec<f32> = (0..64 * 64).map(|i| (i as f32) * 0.25 + 1.5).collect();
        let list = host.new_list().expect("new list");
        record_gpu_fill_r32f(host, &list, res, &expected);
        list.texture_written(res);
        list.close();

        assert!(list.has_texture_backs());
        assert_eq!(list.texture_back_ranges(), vec![(at, at + size)]);
        assert!(pending_ranges().iter().any(|&(a, b)| a <= at && at + size <= b));

        host.execute(&[&list]);

        let mut detiled = vec![0u8; 64 * 64 * 4];
        layout
            .detile(&backing[..size], 0, 0, &mut detiled, 64 * 4, 64 * 64 * 4)
            .expect("detile landed title memory");
        let got: Vec<f32> = detiled
            .as_chunks::<4>()
            .0
            .iter()
            .map(|&c| f32::from_le_bytes(c))
            .collect();
        assert_eq!(got, expected);

        list.reset();
        release_resource(res);
    }

    #[cfg(windows)]
    #[test]
    fn stale_texture_readback_is_skipped_without_dropping_unrelated_readback() {
        let Some(host) = crate::host::host() else {
            return;
        };
        let _guard = KindGuard::set(vec![(64, 64, 41)]);

        let mut buf_released = vec![0xAAu8; 64 * 1024];
        let mut buf_replaced = vec![0xDDu8; 64 * 1024];
        let mut buf_remapped = vec![0xBBu8; 64 * 1024];
        let mut buf_live = vec![0xCCu8; 64 * 1024];

        let tex_released = make_placed_2d_r32f(64, 64, buf_released.as_mut_ptr());
        let tex_replaced = make_placed_2d_r32f(64, 64, buf_replaced.as_mut_ptr());
        let tex_remapped = make_placed_2d_r32f(64, 64, buf_remapped.as_mut_ptr());
        let tex_live = make_placed_2d_r32f(64, 64, buf_live.as_mut_ptr());
        let (layout_live, _, size_live) =
            crate::resource::title_texture(tex_live).expect("live texture layout");

        let pat_a: Vec<f32> = (0..64 * 64).map(|i| (i as f32) + 10.0).collect();
        let pat_b: Vec<f32> = (0..64 * 64).map(|i| (i as f32) + 20.0).collect();
        let pat_c: Vec<f32> = (0..64 * 64).map(|i| (i as f32) + 30.0).collect();
        let pat_live: Vec<f32> = (0..64 * 64).map(|i| (i as f32) * 0.5 + 3.25).collect();

        let list = host.new_list().expect("new list");
        record_gpu_fill_r32f(host, &list, tex_released, &pat_a);
        list.texture_written(tex_released);
        record_gpu_fill_r32f(host, &list, tex_replaced, &pat_b);
        list.texture_written(tex_replaced);
        record_gpu_fill_r32f(host, &list, tex_remapped, &pat_c);
        list.texture_written(tex_remapped);
        record_gpu_fill_r32f(host, &list, tex_live, &pat_live);
        list.texture_written(tex_live);
        list.close();
        assert_eq!(list.texture_back_ranges().len(), 4);

        // Invalidate the first three (release, replacement, remap) before GPU execution lands.
        release_resource(tex_released);
        let tex_replacement = make_placed_2d_r32f(64, 64, buf_replaced.as_mut_ptr());
        crate::watch::title_memory_remapped(buf_remapped.as_ptr() as usize, buf_remapped.len());

        host.execute(&[&list]);

        // Stale destinations were skipped cleanly; unrelated live texture landed.
        assert!(buf_released.iter().all(|&b| b == 0xAA));
        assert!(buf_replaced.iter().all(|&b| b == 0xDD));
        assert!(buf_remapped.iter().all(|&b| b == 0xBB));

        let mut detiled = vec![0u8; 64 * 64 * 4];
        layout_live
            .detile(&buf_live[..size_live], 0, 0, &mut detiled, 64 * 4, 64 * 64 * 4)
            .expect("detile live texture");
        let got: Vec<f32> = detiled
            .as_chunks::<4>()
            .0
            .iter()
            .map(|&c| f32::from_le_bytes(c))
            .collect();
        assert_eq!(got, pat_live);

        list.reset();
        release_resource(tex_replacement);
        release_resource(tex_replaced);
        release_resource(tex_remapped);
        release_resource(tex_live);
    }

    #[cfg(windows)]
    #[test]
    fn disabled_by_default_does_not_record_readbacks() {
        let Some(host) = crate::host::host() else {
            return;
        };
        let _guard = KindGuard::set(Vec::new());
        let mut backing = vec![0u8; 64 * 1024];
        let res = make_placed_2d_r32f(64, 64, backing.as_mut_ptr());
        let list = host.new_list().expect("new list");
        list.texture_written(res);
        list.close();
        assert!(!list.has_texture_backs());
        assert!(list.texture_back_ranges().is_empty());
        list.reset();
        release_resource(res);
    }

    #[test]
    fn idle_pool_bounded_best_fit_caps_and_drain() {
        assert_eq!(align_readback_bytes(0), Some(64 << 10));
        assert_eq!(align_readback_bytes(1), Some(64 << 10));
        assert_eq!(align_readback_bytes(64 << 10), Some(64 << 10));
        assert_eq!(align_readback_bytes((64 << 10) + 1), Some(128 << 10));
        assert_eq!(align_readback_bytes(u64::MAX), None);

        let mut pool = IdlePool::<u32>::new();
        assert_eq!(pool.put(1, 4 << 20), None);
        assert_eq!(pool.stats(), (1, 4 << 20));

        // A 64 KiB request must NOT hijack an idle 4 MiB buffer.
        assert_eq!(pool.take(64 << 10), None);
        assert_eq!(pool.stats(), (1, 4 << 20));

        // Bounded best-fit picks the smallest entry in [aligned, 2 * aligned].
        assert_eq!(pool.put(2, 128 << 10), None);
        assert_eq!(pool.put(3, 64 << 10), None);
        assert_eq!(pool.put(4, 256 << 10), None);
        assert_eq!(pool.take(64 << 10), Some((3, 64 << 10)));
        assert_eq!(pool.take(64 << 10), Some((2, 128 << 10)));
        // 256 KiB exceeds 2 * 64 KiB, so a third 64 KiB request misses.
        assert_eq!(pool.take(64 << 10), None);
        // 192 KiB matches 256 KiB (256 KiB <= 2 * 192 KiB).
        assert_eq!(pool.take(192 << 10), Some((4, 256 << 10)));
        // 3 MiB matches the 4 MiB buffer (4 MiB <= 2 * 3 MiB).
        assert_eq!(pool.take(3 << 20), Some((1, 4 << 20)));
        assert_eq!(pool.stats(), (0, 0));

        // Count cap (`MAX_IDLE_BUFFERS = 64`): 65th entry overflows.
        for i in 0..MAX_IDLE_BUFFERS as u32 {
            assert_eq!(pool.put(i, 64 << 10), None);
        }
        assert_eq!(pool.put(999, 64 << 10), Some(999));
        let drained = pool.drain();
        assert_eq!(drained.len(), MAX_IDLE_BUFFERS);
        assert_eq!(pool.stats(), (0, 0));

        // Byte cap (`MAX_IDLE_BYTES = 16 MiB`): 5th 4 MiB entry overflows.
        for i in 0..4u32 {
            assert_eq!(pool.put(i, 4 << 20), None);
        }
        assert_eq!(pool.put(100, 64 << 10), Some(100));
        assert_eq!(pool.drain().len(), 4);
        assert_eq!(pool.stats(), (0, 0));
    }

    #[cfg(windows)]
    fn idle_stats() -> (usize, u64) {
        IDLE.lock().unwrap_or_else(|e| e.into_inner()).stats()
    }

    #[cfg(windows)]
    #[test]
    fn readback_buffer_is_reused_after_list_reset_across_distinct_fills() {
        let Some(host) = crate::host::host() else {
            return;
        };
        let _guard = KindGuard::set(vec![(64, 64, 41)]);
        trim_idle();
        assert_eq!(idle_stats(), (0, 0));

        let mut backing = vec![0xCCu8; 64 * 1024];
        let res = make_placed_2d_r32f(64, 64, backing.as_mut_ptr());
        let (layout, _, size) = crate::resource::title_texture(res).expect("title texture layout");

        let fill_a: Vec<f32> = (0..64 * 64).map(|i| (i as f32) * 0.25 + 1.5).collect();
        let fill_b: Vec<f32> = (0..64 * 64).map(|i| 10_000.0 - (i as f32) * 0.75).collect();

        let list = host.new_list().expect("new list");

        // Pass 1: allocates a 64 KiB readback buffer and lands fill A.
        record_gpu_fill_r32f(host, &list, res, &fill_a);
        list.texture_written(res);
        list.close();
        let ptrs_a = list.texture_back_raw_ptrs_for_test();
        assert_eq!(ptrs_a.len(), 1);
        assert_eq!(idle_stats(), (0, 0));

        host.execute(&[&list]);
        let mut detiled = vec![0u8; 64 * 64 * 4];
        layout
            .detile(&backing[..size], 0, 0, &mut detiled, 64 * 4, 64 * 64 * 4)
            .expect("detile pass 1");
        let got_a: Vec<f32> = detiled
            .as_chunks::<4>()
            .0
            .iter()
            .map(|&c| f32::from_le_bytes(c))
            .collect();
        assert_eq!(got_a, fill_a);
        // Buffer stays owned by the list until `reset()` completes.
        assert_eq!(idle_stats(), (0, 0));

        list.reset();
        assert_eq!(idle_stats(), (1, READBACK_ALIGNMENT));

        // Pass 2: reuses the exact same pooled readback buffer and lands fill B.
        record_gpu_fill_r32f(host, &list, res, &fill_b);
        list.texture_written(res);
        list.close();
        let ptrs_b = list.texture_back_raw_ptrs_for_test();
        assert_eq!(ptrs_b, ptrs_a, "expected pooled readback buffer reuse");
        assert_eq!(idle_stats(), (0, 0));

        host.execute(&[&list]);
        detiled.fill(0);
        layout
            .detile(&backing[..size], 0, 0, &mut detiled, 64 * 4, 64 * 64 * 4)
            .expect("detile pass 2");
        let got_b: Vec<f32> = detiled
            .as_chunks::<4>()
            .0
            .iter()
            .map(|&c| f32::from_le_bytes(c))
            .collect();
        assert_eq!(got_b, fill_b);

        list.reset();
        assert_eq!(idle_stats(), (1, READBACK_ALIGNMENT));
        trim_idle();
        assert_eq!(idle_stats(), (0, 0));
        release_resource(res);
    }

    #[cfg(windows)]
    #[test]
    fn bounded_best_fit_does_not_hand_4mib_idle_buffer_to_64kib_request() {
        let Some(host) = crate::host::host() else {
            return;
        };
        let _guard = KindGuard::set(Vec::new());
        trim_idle();
        assert_eq!(idle_stats(), (0, 0));

        let big = Readback::acquire(host, 4 << 20).expect("4 MiB readback");
        let big_ptr = windows::core::Interface::as_raw(big.resource()) as usize;
        assert_eq!(big.bytes, 4 << 20);
        big.recycle();
        assert_eq!(idle_stats(), (1, 4 << 20));

        let small = Readback::acquire(host, 64 << 10).expect("64 KiB readback");
        let small_ptr = windows::core::Interface::as_raw(small.resource()) as usize;
        assert_ne!(small_ptr, big_ptr, "64 KiB request must not hijack 4 MiB buffer");
        assert_eq!(small.bytes, 64 << 10);
        assert_eq!(idle_stats(), (1, 4 << 20));

        // A 3 MiB request is within 2x of 4 MiB, so it reuses the idle 4 MiB buffer.
        let reused_big = Readback::acquire(host, 3 << 20).expect("3 MiB readback");
        let reused_ptr = windows::core::Interface::as_raw(reused_big.resource()) as usize;
        assert_eq!(reused_ptr, big_ptr);
        assert_eq!(reused_big.bytes, 4 << 20);
        assert_eq!(idle_stats(), (0, 0));

        small.recycle();
        reused_big.recycle();
        trim_idle();
        assert_eq!(idle_stats(), (0, 0));
    }

    #[cfg(windows)]
    #[test]
    fn trim_idle_empties_readback_pool_cleanly() {
        let Some(host) = crate::host::host() else {
            return;
        };
        let _guard = KindGuard::set(Vec::new());
        trim_idle();
        assert_eq!(idle_stats(), (0, 0));

        let a = Readback::acquire(host, 32 << 10).expect("64 KiB aligned readback");
        let b = Readback::acquire(host, 100 << 10).expect("128 KiB aligned readback");
        assert_eq!(a.bytes, 64 << 10);
        assert_eq!(b.bytes, 128 << 10);
        a.recycle();
        b.recycle();
        assert_eq!(idle_stats(), (2, 192 << 10));

        trim_idle();
        assert_eq!(idle_stats(), (0, 0));

        // Idempotent when already empty.
        trim_idle();
        assert_eq!(idle_stats(), (0, 0));
    }
}
