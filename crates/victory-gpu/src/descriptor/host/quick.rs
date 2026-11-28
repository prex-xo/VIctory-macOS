//! Per-page deduplicated buffer-SRV and CBV resolution for fast heap syncs
//! (`VICTORY_QUICK_SYNC=1` or `VICTORY_QUICK_SYNC=views`, off by default).
//!
//! During `HostHeap::mirror_synced_batch`, most stale title heap slots hold
//! buffer shader-resource views or constant buffer views, often repeated
//! across many slots in the same 64 KiB page. On the standard path, each
//! unique record resolves its host shadow copy three times (`prepare`,
//! `make_marked`/`buffer_srv`, and `bindless::seen`), with the latter two
//! running under `HostHeap::made`.
//!
//! When enabled, [`resolve`] resolves each unique buffer-SRV / CBV record once
//! per page outside `HostHeap::made` — computing the host buffer, final D3D12
//! SRV descriptor (including 3-channel widening and `raw_structured` lowering),
//! held resource source, view-stride entry, view-address word, and
//! `bindless::Seen` entry. `mirror_synced_batch` then stamps every matching
//! slot through its `StagingHeap` templates, `keep_batch`, and
//! `Viewed::set_batch_resolved`. All non-buffer views, textures, UAVs,
//! samplers, placed textures, and refused/edge-case views return `None` and
//! stay on the standard `prepare` + `make_marked` path.

use std::sync::Arc;

use windows::Win32::Graphics::Direct3D12::*;
use windows::Win32::Graphics::Dxgi::Common::*;

use super::{
    HeldSource, element_size, in_buffer, raw_structured, srv_desc, view_kind, view_stride,
    widened_format, widened_w,
};
use crate::bindless::Seen;
use crate::descriptor::View;
use crate::shadow::{Shadow, Widened};

/// Whether per-page quick buffer-SRV / CBV resolution is enabled
/// (`VICTORY_QUICK_SYNC=1` or `=views`).
pub(super) fn on() -> bool {
    crate::census_maps::quick_views()
}

/// Pre-resolved host buffer view and side-table data for one unique buffer
/// SRV or CBV record within a synced page.
pub(super) struct ResolvedBuffer {
    pub(super) buffer: ID3D12Resource,
    pub(super) desc: D3D12_SHADER_RESOURCE_VIEW_DESC,
    shadow: Arc<Shadow>,
    own: bool,
    widened: Option<Arc<Widened>>,
    pub(super) stride: u32,
    /// Title address stored in `HostHeap::addresses` (`0` for `View::Constant`,
    /// matching `write_address` / `buffer_range`).
    pub(super) address: u64,
    pub(super) seen: Option<Seen>,
    pub(super) epoch: u64,
}

impl ResolvedBuffer {
    pub(super) fn current(&self) -> bool {
        self.epoch == crate::shadow::lookup_epoch()
    }

    /// Returns the secondary lifetime handle held beside the host buffer in
    /// `HostHeap::kept`, matching `buffer_srv`.
    pub(super) fn held(&self) -> Option<HeldSource> {
        match &self.widened {
            Some(widened) => Some(HeldSource::Widened(widened.clone(), self.shadow.clone())),
            None => self.own.then(|| HeldSource::Shadow(self.shadow.clone())),
        }
    }
}

#[cfg(test)]
fn run_resolve_hooks() {
    super::tests::while_preparing();
    tests::run_local_resolve_hook();
}

#[cfg(test)]
pub(super) fn run_topology_hook(publishing: bool) {
    tests::run_topology_hook(publishing);
}

/// Resolves a buffer SRV, placed buffer SRV, or constant buffer view outside
/// `HostHeap::made`, producing the exact host descriptor, held source, stride,
/// view address, and `bindless::Seen` entry that `make` + `buffer_srv` +
/// `bindless::seen` would produce.
///
/// Returns `None` for any texture, UAV, sampler, placed texture, refused view,
/// or edge case (zero/unaligned CBV, unaligned structured view) so the caller
/// falls back to the standard path identically.
pub(super) fn resolve(view: &View) -> Option<ResolvedBuffer> {
    // A lookup can itself register a shadow. That changes the epoch too;
    // the caller can then use the warmed standard path for this first view.
    let epoch = crate::shadow::lookup_epoch();
    let (address, mut desc, constant, has_view_address) = match view {
        &View::Constant(address, size) => {
            let len = u64::from(size);
            if size == 0 || !size.is_multiple_of(4) {
                return None;
            }
            #[cfg(test)]
            run_resolve_hooks();
            let shadow = crate::shadow::find(address, len)?;
            if shadow.left(address) < len {
                return None;
            }
            let desc = D3D12_SHADER_RESOURCE_VIEW_DESC {
                Format: DXGI_FORMAT_R32_TYPELESS,
                ViewDimension: D3D12_SRV_DIMENSION_BUFFER,
                Shader4ComponentMapping: D3D12_DEFAULT_SHADER_4_COMPONENT_MAPPING,
                Anonymous: D3D12_SHADER_RESOURCE_VIEW_DESC_0 {
                    Buffer: D3D12_BUFFER_SRV {
                        NumElements: size / 4,
                        Flags: D3D12_BUFFER_SRV_FLAG_RAW,
                        ..Default::default()
                    },
                },
            };
            (address, desc, Some(shadow), false)
        }
        View::Shader(resource, Some(words)) => {
            let resource = *resource as crate::object::This;
            let (data, _) = crate::resource::buffer_bytes(resource)?;
            if crate::resource::made_copy(resource).is_some() {
                return None;
            }
            let desc = srv_desc(words);
            if desc.ViewDimension != D3D12_SRV_DIMENSION_BUFFER {
                return None;
            }
            #[cfg(test)]
            run_resolve_hooks();
            (data as u64, desc, None, true)
        }
        View::Placed(false, resource, words) if resource[0] == 1 => {
            let desc = srv_desc(words[..10].try_into().ok()?);
            if desc.ViewDimension != D3D12_SRV_DIMENSION_BUFFER {
                return None;
            }
            let address = u64::from(words[10]) | u64::from(words[11]) << 32;
            #[cfg(test)]
            run_resolve_hooks();
            (address, desc, None, true)
        }
        _ => return None,
    };

    let element = element_size(&desc);
    if element == 0 {
        return None;
    }
    // SAFETY: `desc.ViewDimension == D3D12_SRV_DIMENSION_BUFFER` verified above.
    let (first_elem, count, struct_stride) = unsafe {
        let b = desc.Anonymous.Buffer;
        (
            b.FirstElement,
            b.NumElements,
            u64::from(b.StructureByteStride),
        )
    };
    let start = address.checked_add(first_elem.checked_mul(element)?)?;
    let len = u64::from(count).checked_mul(element)?;

    let (shadow, own) = match &constant {
        Some(found) if found.offset(start).is_multiple_of(element) => (found.clone(), false),
        _ => crate::shadow::for_view(start, len, element)?,
    };
    let first = shadow.offset(start) / element;
    if !in_buffer(&shadow.buffer, first, count, element) {
        return None;
    }

    let widened = if crate::under_wine()
        && matches!(desc.Format.0, 6..=8)
        && let Some(w) = widened_w(desc.Format.0)
    {
        shadow.widened(start, u64::from(count) * 12, w)
    } else {
        None
    };

    let buffer = match &widened {
        Some(widened) => {
            desc.Format = widened_format(desc.Format.0)?;
            desc.Anonymous.Buffer.FirstElement = 0;
            desc.Anonymous.Buffer.NumElements = count;
            widened.buffer.clone()
        }
        None => {
            desc.Anonymous.Buffer.FirstElement = first;
            if struct_stride > 0 && raw_structured() {
                let (byte, dwords) = (first * struct_stride, u64::from(count) * struct_stride / 4);
                let align = if crate::under_wine() { 4 } else { 16 };
                match u32::try_from(dwords) {
                    Ok(dwords) if byte.is_multiple_of(align) && struct_stride.is_multiple_of(4) => {
                        desc.Format = DXGI_FORMAT_R32_TYPELESS;
                        desc.Anonymous.Buffer = D3D12_BUFFER_SRV {
                            FirstElement: byte / 4,
                            NumElements: dwords,
                            StructureByteStride: 0,
                            Flags: D3D12_BUFFER_SRV_FLAG_RAW,
                        };
                    }
                    // Fall back to `buffer_srv` for unaligned structured views and their warning.
                    _ => return None,
                }
            }
            shadow.buffer.clone()
        }
    };

    let seen = match constant {
        Some(found) => Seen::shader_buffer(found, address, len),
        None => Seen::shader_buffer(shadow.clone(), start, len),
    };

    Some(ResolvedBuffer {
        buffer,
        desc,
        shadow,
        own,
        widened,
        stride: view_stride(view) | view_kind(view),
        address: if has_view_address { start } else { 0 },
        seen,
        epoch,
    })
}

#[cfg(test)]
mod tests {
    use std::sync::atomic::Ordering;

    use super::*;
    use crate::backend::HostGpu;
    use crate::bindless::Viewed;
    use crate::descriptor::host::{HostHeap, host};

    type TopologyHook = (bool, Box<dyn FnOnce()>);

    thread_local! {
        static LOCAL_RESOLVE_HOOK: std::cell::RefCell<Option<Box<dyn FnOnce()>>> =
            const { std::cell::RefCell::new(None) };
        static TOPOLOGY_HOOK: std::cell::RefCell<Option<TopologyHook>> =
            const { std::cell::RefCell::new(None) };
    }

    pub(super) fn run_local_resolve_hook() {
        let hook = LOCAL_RESOLVE_HOOK.with(|h| h.borrow_mut().take());
        if let Some(hook) = hook {
            hook();
        }
    }

    pub(super) fn run_topology_hook(publishing: bool) {
        let hook = TOPOLOGY_HOOK.with(|h| {
            let mut hook = h.borrow_mut();
            if hook.as_ref().is_some_and(|(when, _)| *when == publishing) {
                hook.take()
            } else {
                None
            }
        });
        if let Some((_, hook)) = hook {
            hook();
        }
    }

    type SlotState = (
        bool,
        bool,
        u64,
        u64,
        u32,
        (u32, u32),
        Option<u64>,
        Option<crate::bindless::Range>,
    );

    fn slot_state(heap: &HostHeap, index: u32) -> SlotState {
        let stride = heap.strides.as_ref().map_or(0, |(_, data)| {
            // SAFETY: one dword per slot in the mapped stride table.
            unsafe { (*data as *const u32).add(index as usize).read_volatile() }
        });
        let mark = heap.marks.as_ref().map_or((0, 0), |(_, data)| {
            // SAFETY: two dwords per slot in the mapped mark table.
            unsafe {
                let p = (*data as *const u32).add(2 * index as usize);
                (p.read_volatile(), p.add(1).read_volatile())
            }
        });
        let synced = heap.synced[index as usize].load(Ordering::Acquire);
        let ready = heap.ready[index as usize].load(Ordering::Acquire);
        (
            heap.made(index),
            heap.kept(index),
            synced,
            ready,
            stride,
            mark,
            heap.address_at(index),
            heap.viewed.range_at(index),
        )
    }

    fn normalize_writes(writes: &[(Arc<Shadow>, usize, usize)]) -> Vec<(usize, usize, usize)> {
        let mut out: Vec<_> = writes
            .iter()
            .map(|(s, a, l)| (Arc::as_ptr(s) as usize, *a, *l))
            .collect();
        out.sort_unstable();
        out
    }

    fn make_texture_resource() -> crate::object::This {
        let desc: [u32; 14] = [3, 0, 0, 0, 4, 0, 1, 1 | 1 << 16, 28, 1, 0, 0x100, 0, 0];
        let properties = [1u32, 0, 0, 0, 0];
        let mut out = std::ptr::null();
        let iid = crate::object::guid(
            0x696442be,
            0xa72e,
            0x4059,
            [0xbc, 0x79, 0x5b, 0x5c, 0x98, 0x04, 0x0f, 0xad],
        );
        let hr = crate::resource::create_committed(
            std::ptr::null(),
            properties.as_ptr().cast(),
            0,
            desc.as_ptr().cast(),
            0,
            std::ptr::null(),
            &iid,
            &mut out,
        );
        assert_eq!(hr, crate::object::S_OK);
        out
    }

    /// Compares `quick=true` (+ `VICTORY_CENSUS_FAST=1` + `quick_maps=true`)
    /// against `quick=false` (standard maps) across CBVs, buffer SRVs,
    /// structured SRVs, placed buffer SRVs, 3-channel widened SRVs, textures,
    /// placed textures, UAVs, refused views, duplicates, overwrites, and clears.
    #[test]
    fn quick_sync_matches_standard_across_buffer_views_textures_uavs_and_duplicates() {
        if host().is_none() {
            return; // no host GPU here
        }
        let buf = crate::resource::test_buffer(4096);
        let (buf_ptr, buf_len) = crate::resource::buffer_bytes(buf).expect("buffer bytes");
        crate::shadow::register(buf as usize, buf_ptr as u64, buf_len);
        let memory = buf_ptr as u64;

        let tex = make_texture_resource();
        let placed_tex_mem = vec![0u8; 4096];
        let placed_tex_addr = placed_tex_mem.as_ptr() as u64;

        const SLOTS: u32 = 32;
        let mut title_heap = vec![0u8; SLOTS as usize * 32];
        let start = title_heap.as_ptr() as u64;
        let at = |i: u32| start + u64::from(i) * 32;
        let null = std::ptr::null();

        // 0..=2: duplicate typed R8_UINT buffer SRVs (tests StagingHeap template stamping).
        let srv_r8 = [62u32, 1, 0x1688, 0, 0, 0, 256, 0, 0, 0];
        for i in 0..=2 {
            crate::descriptor::create_srv(null, buf, &srv_r8, at(i));
        }
        // 3: subrange typed R8_UINT buffer SRV.
        crate::descriptor::create_srv(null, buf, &[62, 1, 0x1688, 0, 16, 0, 48, 0, 0, 0], at(3));
        // 4..=5: duplicate structured SRVs (16-byte stride, FirstElement=1, exercises raw_structured).
        let srv_struct = [0u32, 1, 0x1688, 0, 1, 0, 8, 16, 0, 0];
        crate::descriptor::create_srv(null, buf, &srv_struct, at(4));
        crate::descriptor::create_srv(null, buf, &srv_struct, at(5));
        // 6: placed buffer SRV off element alignment (12-byte stride at +4B -> own exact shadow copy).
        let mut placed_buf = [0u32; 16];
        placed_buf[..10].copy_from_slice(&[0, 1, 0x1688, 0, 0, 0, 8, 12, 0, 0]);
        (placed_buf[10], placed_buf[11]) = ((memory + 4) as u32, ((memory + 4) >> 32) as u32);
        let mut placed_buf_res = [0u32; 14];
        (placed_buf_res[0], placed_buf_res[4]) = (1, 256);
        crate::descriptor::create_placed_srv(null, &placed_buf_res, &placed_buf, at(6));
        // 7..=8: duplicate constant buffers (aligned dword multiple).
        let cbv = [memory as u32, (memory >> 32) as u32, 64, 0];
        crate::descriptor::create_cbv(null, &cbv, at(7));
        crate::descriptor::create_cbv(null, &cbv, at(8));
        // 9: unaligned-size CBV (falls back to standard path).
        crate::descriptor::create_cbv(null, &[memory as u32, (memory >> 32) as u32, 62, 0], at(9));
        // 10..=11: duplicate 3-channel R32G32B32_FLOAT SRVs (widened under Wine).
        let srv_rgb32f = [6u32, 1, 0x1688, 0, 0, 0, 8, 0, 0, 0];
        crate::descriptor::create_srv(null, buf, &srv_rgb32f, at(10));
        crate::descriptor::create_srv(null, buf, &srv_rgb32f, at(11));
        // 12: 3-channel R32G32B32_UINT SRV.
        crate::descriptor::create_srv(null, buf, &[7, 1, 0x1688, 0, 0, 0, 8, 0, 0, 0], at(12));
        // 13: regular 2D texture SRV (standard path).
        crate::descriptor::create_srv(null, tex, std::ptr::null(), at(13));
        // 14: placed 2D texture SRV (left null on sync, ready == 0).
        let tex_res = [3u32, 0, 0, 0, 4, 0, 1, 1 | 1 << 16, 28, 1, 0, 0x100, 0, 0];
        let mut placed_tex = [0u32; 16];
        placed_tex[..6].copy_from_slice(&[28, 4, 0x1688, 0, 0, 1]);
        (placed_tex[10], placed_tex[11]) = (placed_tex_addr as u32, (placed_tex_addr >> 32) as u32);
        crate::descriptor::create_placed_srv(null, &tex_res, &placed_tex, at(14));
        // 15: raw buffer UAV (standard path, sets mark & enters UnorderedIndex).
        crate::descriptor::create_uav(null, buf, null, &[39, 1, 0, 0, 64, 0, 0, 0, 1, 0], at(15));
        // 16: refused empty buffer SRV (NumElements == 0 -> standard path).
        crate::descriptor::create_srv(null, buf, &[62, 1, 0x1688, 0, 0, 0, 0, 0, 0, 0], at(16));
        // 17..32: empty slots.

        // Verify quick::resolve classification explicitly.
        for (slot, want_quick) in [
            (0, true),
            (3, true),
            (4, true),
            (6, true),
            (7, true),
            (9, false),
            (10, true),
            (12, true),
            (13, false),
            (14, false),
            (15, false),
            (16, false),
        ] {
            let view = crate::descriptor::view(at(slot)).expect("recorded view");
            assert_eq!(
                resolve(&view).is_some(),
                want_quick,
                "unexpected quick::resolve classification for slot {slot}"
            );
        }

        let (Some(mut standard), Some(mut quick)) =
            (HostHeap::new(false, SLOTS), HostHeap::new(false, SLOTS))
        else {
            crate::object::release_ref(tex);
            crate::object::release_ref(buf);
            return;
        };
        standard.viewed = Arc::new(Viewed::with_options(
            false,
            false,
            crate::ranges::disjoint_spans_enabled(),
        ));
        quick.viewed = Arc::new(Viewed::with_options(
            true,
            true,
            crate::ranges::disjoint_spans_enabled(),
        ));
        // Initialize view_addresses buffers before sync so write_buffer_address is verified too.
        assert!(standard.view_addresses().is_some());
        assert!(quick.view_addresses().is_some());

        standard.sync_slots_bytes_mode(0..SLOTS, start, 32, (start, &title_heap), true, false);
        quick.sync_slots_bytes_mode(0..SLOTS, start, 32, (start, &title_heap), true, true);

        for i in 0..SLOTS {
            assert_eq!(
                slot_state(&quick, i),
                slot_state(&standard, i),
                "round 0 mismatch at slot {i}"
            );
        }
        assert_eq!(
            quick.widened_buffers().len(),
            standard.widened_buffers().len(),
            "widened buffer count mismatch"
        );
        if crate::under_wine() {
            assert_eq!(
                quick.widened_buffers().len(),
                2,
                "float + uint widened copies"
            );
        }

        let check_uav_writes = |label: &str| {
            let mut want = Vec::new();
            crate::bindless::widened_writes_into(&standard.viewed, &mut want);
            let mut got_scan = Vec::new();
            crate::bindless::widened_writes_into(&quick.viewed, &mut got_scan);
            let got_fast = crate::bindless::unordered_writes(&quick.viewed).expect("fast cache");
            assert_eq!(
                normalize_writes(&got_scan),
                normalize_writes(&want),
                "{label}: widened_writes_into mismatch"
            );
            assert_eq!(
                normalize_writes(&got_fast),
                normalize_writes(&want),
                "{label}: unordered_writes mismatch"
            );
        };
        check_uav_writes("round 0");

        // Round 1: clear slots 1, 10, 15; overwrite slot 3 with CBV and slot 7 with subrange SRV.
        title_heap[32..64].fill(0);
        title_heap[10 * 32..11 * 32].fill(0);
        title_heap[15 * 32..16 * 32].fill(0);
        crate::descriptor::create_cbv(null, &cbv, at(3));
        crate::descriptor::create_srv(null, buf, &[62, 1, 0x1688, 0, 32, 0, 64, 0, 0, 0], at(7));

        standard.sync_slots_bytes_mode(0..SLOTS, start, 32, (start, &title_heap), true, false);
        quick.sync_slots_bytes_mode(0..SLOTS, start, 32, (start, &title_heap), true, true);

        for i in 0..SLOTS {
            assert_eq!(
                slot_state(&quick, i),
                slot_state(&standard, i),
                "round 1 mismatch at slot {i}"
            );
        }
        assert!(!quick.made(1) && !quick.made(10) && !quick.made(15));
        assert_eq!(
            quick.widened_buffers().len(),
            standard.widened_buffers().len()
        );
        check_uav_writes("round 1");

        // SAFETY: plain device health check.
        assert!(unsafe { host().unwrap().device.GetDeviceRemovedReason() }.is_ok());

        drop((standard, quick));
        crate::object::release_ref(tex);
        crate::object::release_ref(buf);
        let _ = placed_tex_mem.as_ptr();
    }

    /// End-to-end GPU read test comparing a heap synced via `quick=true`
    /// against `quick=false` across CBV, typed SRV, structured SRV, placed
    /// raw SRV, 3-channel widened SRV (4-channel on native D3D12), and a
    /// duplicate template slot.
    #[test]
    fn quick_sync_gpu_reads_match_standard_for_cbv_typed_structured_placed_and_widened_srvs() {
        const SHADER: &str = r#"
ByteAddressBuffer cbv_raw : register(t0);
Buffer<uint> typed_u32 : register(t1);
ByteAddressBuffer struct_raw : register(t2);
ByteAddressBuffer placed_raw : register(t3);
Buffer<uint4> widened_u32 : register(t4);
Buffer<uint> dup_typed : register(t5);
RWByteAddressBuffer dst : register(u0);
[numthreads(1, 1, 1)]
void main(uint3 id : SV_DispatchThreadID) {
    dst.Store(0, cbv_raw.Load(8));
    dst.Store(4, typed_u32[3]);
    dst.Store(8, struct_raw.Load(4));
    dst.Store(12, placed_raw.Load(0));
    uint4 w = widened_u32[1];
    dst.Store(16, w.x);
    dst.Store(20, w.y);
    dst.Store(24, w.z);
    dst.Store(28, dup_typed[3]);
}
"#;
        let Some(host) = host() else {
            return;
        };
        let src = crate::resource::test_buffer(4096);
        let dst = crate::resource::test_buffer(256);
        let (src_ptr, src_len) = crate::resource::buffer_bytes(src).expect("src bytes");
        let (dst_ptr, dst_len) = crate::resource::buffer_bytes(dst).expect("dst bytes");
        // SAFETY: 4096 bytes of test buffer memory.
        unsafe {
            let words = std::slice::from_raw_parts_mut(src_ptr.cast::<u32>(), 1024);
            for (i, w) in words.iter_mut().enumerate() {
                *w = 0xA000_0000 + i as u32;
            }
        }
        crate::shadow::register(src as usize, src_ptr as u64, src_len);
        crate::shadow::register(dst as usize, dst_ptr as u64, dst_len);
        let src_shadow = crate::shadow::find(src_ptr as u64, src_len).expect("src shadow");
        src_shadow.update(src_ptr as usize, src_len as usize);
        let dst_shadow = crate::shadow::find(dst_ptr as u64, dst_len).expect("dst shadow");

        const SLOTS: u32 = 8;
        let mut title_heap = vec![0u8; SLOTS as usize * 32];
        let start = title_heap.as_mut_ptr() as u64;
        let at = |i: u32| start + u64::from(i) * 32;
        let null = std::ptr::null();
        let mem = src_ptr as u64;
        // Native D3D12 (including WARP) requires 16-byte alignment on raw SRVs
        // and does not run the Wine-only 3-channel widening pass.
        let native = !crate::under_wine();
        let placed_at: u64 = if native { 32 } else { 28 };

        // Slot 0: CBV starting at +16 bytes (dword 4), size 64. Load(8) -> dword 6.
        crate::descriptor::create_cbv(
            null,
            &[(mem + 16) as u32, ((mem + 16) >> 32) as u32, 64, 0],
            at(0),
        );
        // Slot 1: typed R32_UINT SRV (format 42), FirstElement=2, NumElements=16. [3] -> dword 5.
        let typed_desc = [42u32, 1, 0x1688, 0, 2, 0, 16, 0, 0, 0];
        crate::descriptor::create_srv(null, src, &typed_desc, at(1));
        // Slot 2: structured SRV (stride=16, FirstElement=1 -> +16B = dword 4). Load(4) -> dword 5.
        // Only raw-accessible if raw_structured() is enabled; use a raw SRV otherwise so the shader matches.
        let slot2_desc = if raw_structured() {
            [0u32, 1, 0x1688, 0, 1, 0, 8, 16, 0, 0]
        } else {
            [39u32, 1, 0x1688, 0, 4, 0, 32, 0, 1, 0]
        };
        crate::descriptor::create_srv(null, src, &slot2_desc, at(2));
        // Slot 3: placed raw buffer SRV starting at +32B (native, dword 8) or +28B (Wine, dword 7).
        let mut placed_raw = [0u32; 16];
        placed_raw[..10].copy_from_slice(&[39, 1, 0x1688, 0, 0, 0, 16, 0, 1, 0]);
        (placed_raw[10], placed_raw[11]) =
            ((mem + placed_at) as u32, ((mem + placed_at) >> 32) as u32);
        let mut placed_res = [0u32; 14];
        (placed_res[0], placed_res[4]) = (1, 256);
        crate::descriptor::create_placed_srv(null, &placed_res, &placed_raw, at(3));
        // Slot 4: R32G32B32A32_UINT (format 3) on native D3D12 ([1] -> dwords 4, 5, 6), or
        // 3-channel R32G32B32_UINT (format 7, widened under Wine, [1] -> dwords 3, 4, 5).
        let slot4_fmt: u32 = if native { 3 } else { 7 };
        crate::descriptor::create_srv(
            null,
            src,
            &[slot4_fmt, 1, 0x1688, 0, 0, 0, 8, 0, 0, 0],
            at(4),
        );
        // Slot 5: duplicate of Slot 1 (exercises StagingHeap copy). [3] -> dword 5.
        crate::descriptor::create_srv(null, src, &typed_desc, at(5));
        // Slot 6: output raw UAV.
        crate::descriptor::create_uav(null, dst, null, &[39, 1, 0, 0, 16, 0, 0, 0, 1, 0], at(6));

        use crate::compute::{Param, srv, uav};
        let params = &[Param::Table(&[srv(0, 6), uav(0, 1)])];
        let (root, pso) =
            crate::compute::pipeline(SHADER, windows::core::s!("quick_sync_eq"), 0, params)
                .expect("compute pipeline");

        let run_mode = |quick: bool| -> [u32; 8] {
            let mut heap = HostHeap::new(false, SLOTS).expect("host heap");
            heap.viewed = Arc::new(Viewed::with_options(
                quick,
                quick,
                crate::ranges::disjoint_spans_enabled(),
            ));
            heap.sync_slots_bytes_mode(0..SLOTS, start, 32, (start, &title_heap), true, quick);
            let list = host.new_list().expect("list");
            // SAFETY: open list, live heap/root/pso.
            unsafe {
                list.list.SetDescriptorHeaps(&[Some(heap.heap.clone())]);
                list.list.SetComputeRootSignature(&root);
                list.list.SetComputeRootDescriptorTable(0, heap.gpu(0));
                list.list.SetPipelineState(&pso);
                list.list.Dispatch(1, 1, 1);
            }
            list.defer_heap(&heap.viewed);
            list.close();
            host.execute(&[&list]);
            let words = dst_shadow.read_host(dst_ptr as u64, 8);
            words.try_into().expect("8 dwords")
        };

        let standard_out = run_mode(false);
        let quick_out = run_mode(true);
        assert_eq!(quick_out, standard_out, "quick vs standard GPU output");
        let expected = if native {
            [
                0xA000_0006, // cbv_raw.Load(8)
                0xA000_0005, // typed_u32[3]
                0xA000_0005, // struct_raw.Load(4)
                0xA000_0008, // placed_raw.Load(0) at +32B
                0xA000_0004, // R32G32B32A32_UINT[1].x
                0xA000_0005, // R32G32B32A32_UINT[1].y
                0xA000_0006, // R32G32B32A32_UINT[1].z
                0xA000_0005, // dup_typed[3]
            ]
        } else {
            [
                0xA000_0006, // cbv_raw.Load(8)
                0xA000_0005, // typed_u32[3]
                0xA000_0005, // struct_raw.Load(4)
                0xA000_0007, // placed_raw.Load(0) at +28B
                0xA000_0003, // widened[1].x
                0xA000_0004, // widened[1].y
                0xA000_0005, // widened[1].z
                0xA000_0005, // dup_typed[3]
            ]
        };
        assert_eq!(quick_out, expected);
        // SAFETY: plain device health check.
        assert!(unsafe { host.device.GetDeviceRemovedReason() }.is_ok());

        crate::object::release_ref(src);
        crate::object::release_ref(dst);
    }

    /// Concurrent title rewrite occurring while `quick::resolve` runs outside
    /// `HostHeap::made` must be detected on the live re-read and land the
    /// newer view cleanly.
    #[test]
    fn quick_sync_handles_concurrent_rewrite_during_resolve() {
        let Some(mut heap) = HostHeap::new(false, 64) else {
            return;
        };
        heap.viewed = Arc::new(Viewed::with_options(
            true,
            true,
            crate::ranges::disjoint_spans_enabled(),
        ));
        let make_buf = || {
            let b = crate::resource::test_buffer(256);
            let (ptr, len) = crate::resource::buffer_bytes(b).expect("buffer bytes");
            crate::shadow::register(b as usize, ptr as u64, len);
            b
        };
        let buf_a = make_buf();
        let buf_b = make_buf();
        let mut title_heap = vec![0u8; 64 * 32];
        let start = title_heap.as_mut_ptr() as u64;
        let slot_addr = start + 8 * 32;
        let srv_desc = [62u32, 1, 0x1688, 0, 0, 0, 256, 0, 0, 0];
        crate::descriptor::create_srv(std::ptr::null(), buf_a, &srv_desc, slot_addr);

        let mut expected_b = [0u32; 4];
        crate::descriptor::create_srv(
            std::ptr::null(),
            buf_b,
            &srv_desc,
            expected_b.as_mut_ptr() as u64,
        );
        let expected_b_key = u64::from(expected_b[2]) << 32 | u64::from(expected_b[1]);
        let buf_b_raw = buf_b as usize;
        LOCAL_RESOLVE_HOOK.with(|h| {
            *h.borrow_mut() = Some(Box::new(move || {
                crate::descriptor::create_srv(
                    std::ptr::null(),
                    buf_b_raw as crate::object::This,
                    &srv_desc,
                    slot_addr,
                );
            }));
        });

        heap.sync_slots_bytes_mode(0..64, start, 32, (start, &title_heap), true, true);
        assert!(LOCAL_RESOLVE_HOOK.with(|h| h.borrow().is_none()));
        assert_eq!(heap.synced[8].load(Ordering::Acquire), expected_b_key);
        assert!(heap.kept(8));

        drop(heap);
        crate::object::release_ref(buf_a);
        crate::object::release_ref(buf_b);
    }

    #[test]
    fn quick_sync_resolves_again_when_the_shadow_mapping_changes() {
        for publishing in [false, true] {
            shadow_mapping_change(publishing);
        }
    }

    fn shadow_mapping_change(publishing: bool) {
        let Some(heap) = HostHeap::new(false, 4) else {
            return;
        };
        let resource = crate::resource::test_buffer(4096);
        let (start, len) = crate::resource::buffer_bytes(resource).expect("buffer bytes");
        let start = start as u64;
        crate::shadow::register(resource as usize, start, len);
        let outer = crate::shadow::find(start, len).expect("outer shadow");
        let address = start + 256;
        let mut records = [0u8; 4 * 32];
        let heap_start = records.as_mut_ptr() as u64;
        for i in 0..2 {
            crate::descriptor::create_cbv(
                std::ptr::null(),
                &[address as u32, (address >> 32) as u32, 64, 0],
                heap_start + 32 * i,
            );
        }

        // The record and its generation stay the same, but the innermost
        // registered buffer changes after the quick candidate was prepared.
        let inner_key = address as usize;
        TOPOLOGY_HOOK.with(|hook| {
            *hook.borrow_mut() = Some((
                publishing,
                Box::new(move || {
                    crate::shadow::register(inner_key, address, 256);
                }),
            ));
        });
        heap.sync_slots_bytes_mode(0..4, heap_start, 32, (heap_start, &records), true, true);
        assert!(TOPOLOGY_HOOK.with(|hook| hook.borrow().is_none()));
        let inner = crate::shadow::find(address, 64).expect("inner shadow");
        assert!(!Arc::ptr_eq(&inner, &outer));
        let expected = Some((Arc::as_ptr(&inner) as usize, address, 64, false));
        for index in 0..2 {
            assert!(heap.kept(index));
            assert_eq!(heap.viewed.range_at(index), expected);
        }

        drop(heap);
        crate::shadow::unregister(inner_key, address);
        crate::object::release_ref(resource);
    }

    #[test]
    fn rewriting_a_view_keeps_its_displaced_arena_slice_for_retirement() {
        let Some(heap) = HostHeap::new(false, 4) else {
            return;
        };
        let resource = crate::resource::test_buffer(4096);
        let (start, len) = crate::resource::buffer_bytes(resource).expect("buffer bytes");
        let start = start as u64;
        crate::shadow::register(resource as usize, start, len);
        let (old, old_own) = crate::shadow::for_view(start + 4, 80, 16).expect("old view copy");
        let (new, new_own) = crate::shadow::for_view(start + 20, 80, 16).expect("new view copy");
        assert!(old_own && new_own);
        assert_eq!(
            windows::core::Interface::as_raw(&old.buffer),
            windows::core::Interface::as_raw(&new.buffer)
        );
        let old = (old.buffer.clone(), HeldSource::Shadow(old));
        let new = (new.buffer.clone(), HeldSource::Shadow(new));
        let mut retired = Vec::new();

        heap.keep_batch(vec![(0, Some((old.0, Some(old.1))))], &mut retired);
        let duplicate = Some((new.0.clone(), Some(new.1.clone())));
        heap.keep_batch(vec![(0, Some((new.0, Some(new.1))))], &mut retired);
        assert_eq!(retired.len(), 1);
        let displaced = retired[0].1.as_ref().expect("old slice retained");
        assert!(
            crate::shadow::for_view(start + 4, 80, 16)
                .is_some_and(|(from_cache, _)| Arc::ptr_eq(displaced, &from_cache))
        );
        heap.keep_batch(vec![(0, duplicate)], &mut retired);
        assert_eq!(
            retired.len(),
            1,
            "unchanged slice needs no duplicate retirement"
        );
        drop(heap);
        crate::object::release_ref(resource);
    }
}
