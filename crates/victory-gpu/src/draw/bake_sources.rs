//! Urgent pre-upload of bindless texture sources for one-time composite/bake
//! pixel passes.
//!
//! Pixel shaders that bind no texture tables and instead index
//! `ResourceDescriptorHeap[idx]` from CBV words and/or CBV-bounded structured
//! buffer loops can sample placed textures whose host slots are still null
//! (`for_draw = false` leaves placed textures unmade until a GPU read-mark is
//! observed). Because a one-time bake writes its render target once, sampling
//! null slots bakes black/missing content permanently.
//!
//! Using the generic [`HeapIndexSource`] list extracted from pixel-shader IR,
//! [`prepare_bake_sources`] reads the active CBV words and bounded structured
//! records at draw time (never faulting on unmapped memory), synchronously
//! mirrors each referenced placed texture into the bound host heap, and
//! schedules [`crate::texture::upload`] so `Host::execute` uploads those
//! textures ahead of this draw.

use std::collections::HashSet;
use std::sync::OnceLock;
use std::sync::atomic::{AtomicU64, Ordering};

use victory_shader::emit::{BufferBinding, HeapIndexSource, ScalarSource};
use victory_shader::{ResourceType, RootType};

use super::{Root, State};
use crate::descriptor::View;
use crate::host::HostList;
use crate::object::This;
use crate::pipeline::host::HostPipeline;

/// Hard cap on records scanned per `StructuredLoopField` per draw.
const MAX_RECORDS: u32 = 256;
/// Default per-draw pre-upload budget in MiB (`VICTORY_BAKE_SOURCES_MB`).
const DEFAULT_BUDGET_MIB: u64 = 256;
/// Default fallback byte length when reading a root descriptor not in shadow.
const FALLBACK_ROOT_LEN: u64 = 64 << 10;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum BakeMode {
    Off,
    Active,
    Count,
}

fn parse_mode(value: Option<&str>) -> BakeMode {
    match value.map(str::trim) {
        Some(v)
            if v.eq_ignore_ascii_case("0")
                || v.eq_ignore_ascii_case("off")
                || v.eq_ignore_ascii_case("false") =>
        {
            BakeMode::Off
        }
        Some(v) if v.eq_ignore_ascii_case("count") => BakeMode::Count,
        _ => BakeMode::Active,
    }
}

fn mode() -> BakeMode {
    static MODE: OnceLock<BakeMode> = OnceLock::new();
    *MODE.get_or_init(|| {
        let var = std::env::var("VICTORY_BAKE_SOURCES").ok();
        let m = parse_mode(var.as_deref());
        if m != BakeMode::Active {
            tracing::info!(mode = ?m, "VICTORY_BAKE_SOURCES");
        }
        m
    })
}

fn parse_budget_bytes(value: Option<&str>) -> u64 {
    let mib = value
        .map(str::trim)
        .and_then(|v| v.parse::<u64>().ok())
        .unwrap_or(DEFAULT_BUDGET_MIB);
    mib.saturating_mul(1024 * 1024)
}

fn budget_bytes() -> u64 {
    static BUDGET: OnceLock<u64> = OnceLock::new();
    *BUDGET.get_or_init(|| {
        let var = std::env::var("VICTORY_BAKE_SOURCES_MB").ok();
        parse_budget_bytes(var.as_deref())
    })
}

static DRAWS: AtomicU64 = AtomicU64::new(0);
static SOURCES_CHECKED: AtomicU64 = AtomicU64::new(0);
static UPLOADED: AtomicU64 = AtomicU64::new(0);
static ALREADY_READY: AtomicU64 = AtomicU64::new(0);
static UNMAPPED: AtomicU64 = AtomicU64::new(0);

/// Pure, fault-free extraction of deduplicated `ResourceDescriptorHeap` indices
/// from `sources`.
fn collect_heap_indices(
    sources: &[HeapIndexSource],
    read_root_const: impl Fn(u8, u32) -> Option<u32>,
    resolve_buffer: impl Fn(BufferBinding) -> Option<(u64, u64, Option<u32>)>,
    read_u32: impl Fn(u64) -> Option<u32>,
) -> Vec<u32> {
    let read_buffer_word = |binding: BufferBinding, byte_offset: u32| -> Option<u32> {
        let (addr, len, _) = resolve_buffer(binding)?;
        let end = u64::from(byte_offset).checked_add(4)?;
        if end > len {
            return None;
        }
        read_u32(addr.checked_add(u64::from(byte_offset))?)
    };

    let read_scalar = |scalar: ScalarSource| -> Option<u32> {
        match scalar {
            ScalarSource::Const(v) => Some(v),
            ScalarSource::BufferWord {
                binding,
                byte_offset,
            } => read_buffer_word(binding, byte_offset),
            ScalarSource::RootConstant { slot, dword } => read_root_const(slot, dword),
        }
    };

    let mut out = Vec::new();
    let mut seen = HashSet::new();
    let mut push = |raw: u32, index_offset: u32| {
        if let Some(idx) = raw.checked_add(index_offset)
            && seen.insert(idx)
        {
            out.push(idx);
        }
    };

    for source in sources {
        match *source {
            HeapIndexSource::CbvWord {
                binding,
                byte_offset,
                index_offset,
            } => {
                if let Some(word) = read_buffer_word(binding, byte_offset) {
                    push(word, index_offset);
                }
            }
            HeapIndexSource::StructuredLoopField {
                buffer,
                stride,
                field_byte_offset,
                start,
                end,
                end_is_count,
                inclusive_extra,
                index_offset,
            } => {
                let Some((buf_addr, buf_len, view_stride)) = resolve_buffer(buffer) else {
                    continue;
                };
                let Some(stride) = stride.or(view_stride).filter(|&s| s > 0) else {
                    continue;
                };
                let (Some(start_idx), Some(end_raw)) = (read_scalar(start), read_scalar(end))
                else {
                    continue;
                };
                let end_excl = if end_is_count {
                    start_idx
                        .saturating_add(end_raw)
                        .saturating_add(inclusive_extra)
                } else {
                    end_raw.saturating_add(inclusive_extra)
                };
                if end_excl <= start_idx {
                    continue;
                }
                let count = (end_excl - start_idx).min(MAX_RECORDS);
                for k in 0..count {
                    let rec_idx = start_idx.wrapping_add(k);
                    let Some(off) = u64::from(rec_idx)
                        .checked_mul(u64::from(stride))
                        .and_then(|base| base.checked_add(u64::from(field_byte_offset)))
                    else {
                        break;
                    };
                    let Some(word_end) = off.checked_add(4) else {
                        break;
                    };
                    if word_end > buf_len {
                        break;
                    }
                    let Some(addr) = buf_addr.checked_add(off) else {
                        break;
                    };
                    let Some(word) = read_u32(addr) else {
                        break;
                    };
                    push(word, index_offset);
                }
            }
        }
    }
    out
}

/// Synchronously mirrors and queues texture uploads for urgent bindless texture
/// sources referenced by a pure-bindless pixel shader (no table-bound texture
/// SRVs) ahead of the draw.
pub(super) fn prepare_bake_sources(state: &State, list: &HostList, p: &HostPipeline) {
    let bake_mode = mode();
    if bake_mode == BakeMode::Off || state.heaps[0] == 0 {
        return;
    }
    let Some(ps) = p.stages.iter().find(|s| s.shader.stage == "pixel") else {
        return;
    };
    let t = ps.translated();
    if !t.host.heap || t.host.heap_index_sources.is_empty() {
        return;
    }
    // Generic trigger: only pixel draws that bind no texture SRVs through tables.
    let has_table_texture = t.used.iter().any(|u| {
        if u.root != RootType::Table || u.kind != ResourceType::Srv {
            return false;
        }
        let Some(Root::Table(table)) = state.roots_ref(false).get(usize::from(u.slot)) else {
            return false;
        };
        let handle = table + u64::from(u.table_offset_dwords) * 4;
        crate::descriptor::view_safe(handle)
            .as_deref()
            .is_some_and(crate::descriptor::host::is_texture)
    });
    if has_table_texture {
        return;
    }
    let heap = crate::descriptor::heap(state.heaps[0] as This);
    let Some(host) = heap.host() else {
        return;
    };

    let indices = collect_heap_indices(
        &t.host.heap_index_sources,
        |slot, dword| Some(state.user_value(false, slot, RootType::Constants, dword)),
        |binding| resolve_buffer(state, binding),
        read_title_u32,
    );
    if indices.is_empty() {
        return;
    }

    const SHADER_RESOURCE: u32 = 0x40 | 0x80;
    let budget = budget_bytes();
    let max_slot = heap.count().min(host.count());
    let mut draw_checked = 0u64;
    let mut draw_uploaded = 0u64;
    let mut draw_ready = 0u64;
    let mut draw_unmapped = 0u64;
    let mut uploaded_bytes = 0u64;

    for index in indices {
        if index >= max_slot {
            continue;
        }
        let handle = heap.start + u64::from(index) * 32;
        let Some(view) = crate::descriptor::view_safe(handle) else {
            continue;
        };
        let (addr, size) = match &*view {
            View::Placed(false, resource, desc) if resource[0] != 1 => {
                let addr = u64::from(desc[10]) | u64::from(desc[11]) << 32;
                let size = crate::resource::placed_size(resource).unwrap_or(0);
                (addr, size)
            }
            View::Shader(r, _) => {
                let Some((ptr, desc)) = crate::resource::texture_memory(*r as This) else {
                    continue;
                };
                let size = crate::resource::texture_layout(&desc)
                    .map_or(4096, |l| l.planes[0].layout.size);
                (ptr as u64, size)
            }
            _ => continue,
        };

        draw_checked += 1;

        // Already mirrored in the host heap AND already uploaded/host-written?
        if host.kept(index) {
            let existing = match &*view {
                View::Shader(r, _) => Some(*r as This),
                View::Placed(..) => crate::descriptor::host::placed_texture(&view).and_then(|v| {
                    match v {
                        View::Shader(r, _) => Some(r as This),
                        _ => None,
                    }
                }),
                _ => None,
            };
            if let Some(res) = existing
                && let Some(copy) = crate::resource::made_copy(res)
            {
                let host_owned = copy.host_written.load(Ordering::Relaxed) || !copy.title_written();
                let uploaded_clean = copy
                    .staging
                    .lock()
                    .unwrap_or_else(|e| e.into_inner())
                    .as_ref()
                    .is_some_and(|s| s.is_ready());
                if host_owned || uploaded_clean {
                    copy.stamp();
                    draw_ready += 1;
                    continue;
                }
            }
        }

        if !crate::host::has_mapped_memory(addr, size) {
            draw_unmapped += 1;
            continue;
        }

        if budget == 0 || (uploaded_bytes > 0 && uploaded_bytes.saturating_add(size) > budget) {
            break;
        }

        if bake_mode == BakeMode::Count {
            draw_uploaded += 1;
            uploaded_bytes = uploaded_bytes.saturating_add(size);
            continue;
        }

        let Some(pinned) = host.mirror(index, handle, true) else {
            continue;
        };
        let placed = crate::descriptor::host::placed_texture(&pinned);
        let resolved = placed.as_ref().unwrap_or(&pinned);
        if let View::Shader(r, _) = resolved {
            let resource = *r as This;
            if let Some(a) = crate::resource::alias_source(resource) {
                crate::texture::alias_copy(list, a.get(), resource);
            }
            if let Some(copy) = crate::resource::host_copy(resource) {
                copy.stamp();
                if copy.title_written() {
                    crate::texture::upload(list, resource, &copy.texture);
                }
                crate::list::ensure_state(list, resource, SHADER_RESOURCE);
                draw_uploaded += 1;
                uploaded_bytes = uploaded_bytes.saturating_add(size);
            }
        }
    }

    let draws = DRAWS.fetch_add(1, Ordering::Relaxed) + 1;
    let total_checked = SOURCES_CHECKED.fetch_add(draw_checked, Ordering::Relaxed) + draw_checked;
    let total_uploaded = UPLOADED.fetch_add(draw_uploaded, Ordering::Relaxed) + draw_uploaded;
    let total_ready = ALREADY_READY.fetch_add(draw_ready, Ordering::Relaxed) + draw_ready;
    let total_unmapped = UNMAPPED.fetch_add(draw_unmapped, Ordering::Relaxed) + draw_unmapped;

    let should_log = {
        #[cfg(feature = "diagnostics")]
        {
            draws.is_power_of_two() || (draw_uploaded > 0 && draws <= 16)
        }
        #[cfg(not(feature = "diagnostics"))]
        {
            bake_mode == BakeMode::Count && draws.is_power_of_two()
        }
    };
    if should_log {
        tracing::info!(
            pipeline = %p.name,
            mode = ?bake_mode,
            draws,
            draw_checked,
            draw_uploaded,
            draw_ready,
            draw_unmapped,
            uploaded_kib = uploaded_bytes / 1024,
            total_checked,
            total_uploaded,
            total_ready,
            total_unmapped,
            "bake sources"
        );
    }
}

fn resolve_buffer(state: &State, binding: BufferBinding) -> Option<(u64, u64, Option<u32>)> {
    match binding {
        BufferBinding::Table {
            slot,
            table_offset_dwords,
        } => {
            let Root::Table(table) = state.roots_ref(false).get(usize::from(slot))? else {
                return None;
            };
            let handle = table.checked_add(u64::from(table_offset_dwords) * 4)?;
            let view = crate::descriptor::view_safe(handle)?;
            match &*view {
                View::Constant(addr, size) => Some((*addr, u64::from(*size), None)),
                v => {
                    let (addr, len) = crate::descriptor::host::buffer_range(v)?;
                    let stride = crate::descriptor::host::view_stride(v);
                    Some((addr, len, (stride > 0).then_some(stride)))
                }
            }
        }
        BufferBinding::RootDescriptor { slot } => {
            let Root::View(addr) = *state.roots_ref(false).get(usize::from(slot))? else {
                return None;
            };
            if addr == 0 {
                return None;
            }
            let len = crate::shadow::find(addr, 1)
                .map_or(FALLBACK_ROOT_LEN, |s| s.left(addr));
            Some((addr, len, None))
        }
    }
}

fn read_title_u32(addr: u64) -> Option<u32> {
    let mut buf = [0u8; 4];
    crate::host::read_into(addr as *const u8, &mut buf).then(|| u32::from_le_bytes(buf))
}

#[cfg(test)]
mod tests {
    #[cfg(windows)]
    use crate::backend::HostGpu;
    use super::*;
    use std::collections::HashMap;

    #[test]
    fn mode_and_budget_parsing() {
        assert_eq!(parse_mode(None), BakeMode::Active);
        assert_eq!(parse_mode(Some("1")), BakeMode::Active);
        assert_eq!(parse_mode(Some("0")), BakeMode::Off);
        assert_eq!(parse_mode(Some("off")), BakeMode::Off);
        assert_eq!(parse_mode(Some("false")), BakeMode::Off);
        assert_eq!(parse_mode(Some("count")), BakeMode::Count);
        assert_eq!(parse_mode(Some("COUNT")), BakeMode::Count);

        assert_eq!(parse_budget_bytes(None), 256 * 1024 * 1024);
        assert_eq!(parse_budget_bytes(Some("64")), 64 * 1024 * 1024);
        assert_eq!(parse_budget_bytes(Some("0")), 0);
        assert_eq!(parse_budget_bytes(Some("bad")), 256 * 1024 * 1024);
    }

    #[test]
    fn collects_cbv_and_clamped_deduped_structured_records() {
        let cbv = BufferBinding::Table {
            slot: 0,
            table_offset_dwords: 0,
        };
        let srv = BufferBinding::Table {
            slot: 0,
            table_offset_dwords: 8,
        };
        let cbv_base = 0x1000u64;
        let srv_base = 0x1_0000u64;
        let stride = 16u32;

        let mut mem: HashMap<u64, u32> = HashMap::new();
        // CBV words:
        mem.insert(cbv_base + 0x04, 10); // direct texture index
        mem.insert(cbv_base + 0x08, 10); // duplicate direct index
        mem.insert(cbv_base + 0x10, 2); // loop start
        mem.insert(cbv_base + 0x14, 500); // loop end (> MAX_RECORDS=256!)

        // Populate 300 records, leaving record 260 unreadable to verify both
        // clamping and clean stop on unreadable memory.
        for rec in 0..300u32 {
            let addr = srv_base + u64::from(rec) * u64::from(stride) + 8;
            // Deliberately create duplicates (10 at rec 2, and rec % 10 + 100).
            let tex_idx = if rec == 2 { 10 } else { 100 + (rec % 300) };
            mem.insert(addr, tex_idx);
        }

        let sources = vec![
            HeapIndexSource::CbvWord {
                binding: cbv,
                byte_offset: 0x04,
                index_offset: 0,
            },
            HeapIndexSource::CbvWord {
                binding: cbv,
                byte_offset: 0x08,
                index_offset: 0,
            },
            HeapIndexSource::StructuredLoopField {
                buffer: srv,
                stride: Some(stride),
                field_byte_offset: 8,
                start: ScalarSource::BufferWord {
                    binding: cbv,
                    byte_offset: 0x10,
                },
                end: ScalarSource::BufferWord {
                    binding: cbv,
                    byte_offset: 0x14,
                },
                end_is_count: false,
                inclusive_extra: 0,
                index_offset: 0,
            },
        ];

        let indices = collect_heap_indices(
            &sources,
            |_, _| None,
            |b| match b {
                x if x == cbv => Some((cbv_base, 256, None)),
                x if x == srv => Some((srv_base, 300 * u64::from(stride), Some(stride))),
                _ => None,
            },
            |addr| mem.get(&addr).copied(),
        );

        // Expect: 10 (from CBV, deduplicated with rec 2), plus recs 3..258 (255 unique values)
        // because 2..500 is clamped to MAX_RECORDS=256 (records 2..258).
        assert_eq!(indices.len(), 256);
        assert_eq!(indices[0], 10);
        assert_eq!(indices[1], 103);
        assert_eq!(*indices.last().unwrap(), 100 + 257);
    }

    #[test]
    fn stops_cleanly_on_unreadable_memory_and_buffer_end() {
        let cbv = BufferBinding::Table {
            slot: 0,
            table_offset_dwords: 0,
        };
        let srv = BufferBinding::Table {
            slot: 0,
            table_offset_dwords: 8,
        };
        let cbv_base = 0x1000u64;
        let srv_base = 0x2_0000u64;
        let stride = 32u32;

        let mut mem: HashMap<u64, u32> = HashMap::new();
        mem.insert(cbv_base, 0); // start = 0
        mem.insert(cbv_base + 4, 10); // end = 10
        // Only records 0, 1, 2 are mapped; record 3 is unmapped!
        mem.insert(srv_base + 4, 41);
        mem.insert(srv_base + 32 + 4, 42);
        mem.insert(srv_base + 64 + 4, 43);
        // Record 4 is mapped, but we should stop cleanly at unreadable record 3.
        mem.insert(srv_base + 128 + 4, 44);

        let sources = vec![HeapIndexSource::StructuredLoopField {
            buffer: srv,
            stride: None, // resolved from view_stride!
            field_byte_offset: 4,
            start: ScalarSource::BufferWord {
                binding: cbv,
                byte_offset: 0,
            },
            end: ScalarSource::BufferWord {
                binding: cbv,
                byte_offset: 4,
            },
            end_is_count: false,
            inclusive_extra: 0,
            index_offset: 1,
        }];

        let indices = collect_heap_indices(
            &sources,
            |_, _| None,
            |b| match b {
                x if x == cbv => Some((cbv_base, 64, None)),
                x if x == srv => Some((srv_base, 10 * u64::from(stride), Some(stride))),
                _ => None,
            },
            |addr| mem.get(&addr).copied(),
        );

        assert_eq!(indices, vec![42, 43, 44]);
    }

    /// Builds a minimal DXBC container carrying an `OSG1` part with a single
    /// `SV_Target0` float4 output so `victory_shader::emit::emit` can emit a
    /// real SM 6.6 pixel shader.
    fn sv_target0_container() -> Vec<u8> {
        let mut osg1 = Vec::new();
        // count = 1, base = 8
        osg1.extend(1u32.to_le_bytes());
        osg1.extend(8u32.to_le_bytes());
        // 32-byte element: stream(0), name_off(40), index(0), kind(64=SV_Target),
        // comp(3=float32), register(0), mask(0x0f), rw(0x0f), pad(0,0), min_prec(0)
        osg1.extend(0u32.to_le_bytes()); // stream
        osg1.extend(40u32.to_le_bytes()); // name offset
        osg1.extend(0u32.to_le_bytes()); // semantic index
        osg1.extend(64u32.to_le_bytes()); // D3D_NAME_TARGET -> DXIL kind 16
        osg1.extend(3u32.to_le_bytes()); // float32
        osg1.extend(0u32.to_le_bytes()); // register 0
        osg1.extend([0x0f, 0x0f, 0, 0]); // mask, rw, pad
        osg1.extend(0u32.to_le_bytes()); // min_prec
        osg1.extend(b"SV_Target\0\0\0");

        let mut dxbc = Vec::new();
        dxbc.extend(b"DXBC");
        dxbc.extend([0u8; 16]); // hash placeholder
        dxbc.extend(1u32.to_le_bytes()); // version
        let total_size = 32 + 4 + 8 + osg1.len();
        dxbc.extend((total_size as u32).to_le_bytes());
        dxbc.extend(1u32.to_le_bytes()); // 1 part
        dxbc.extend(36u32.to_le_bytes()); // part offset
        dxbc.extend(b"OSG1");
        dxbc.extend((osg1.len() as u32).to_le_bytes());
        dxbc.extend(&osg1);
        dxbc
    }

    /// End-to-end GPU test: a pure-bindless pixel shader whose `HeapIndexSource`
    /// points to a not-yet-made placed texture in the bound view heap renders
    /// real texels on its very first execution.
    #[test]
    fn bindless_bake_draw_pre_uploads_unmade_placed_texture_on_first_execution() {
        use crate::object::{S_OK, guid};
        use crate::pipeline::host::{GraphicsState, HostPipeline};
        use crate::pipeline::{Shader, Translated};
        use victory_shader::native::StageIo;
        use victory_shader::{Container, HwStage, NativeProgram, RootParam, UsedResource};
        use windows::Win32::Foundation::RECT;
        use windows::Win32::Graphics::Direct3D::D3D_PRIMITIVE_TOPOLOGY_TRIANGLELIST;
        use windows::Win32::Graphics::Direct3D::{Fxc::D3DCompile, ID3DBlob};
        use windows::Win32::Graphics::Direct3D12::*;
        use windows::Win32::Graphics::Dxgi::Common::*;
        use windows::core::{PCSTR, s};

        let Some(host) = crate::host::host() else {
            return; // no host GPU here
        };
        let mut sm = D3D12_FEATURE_DATA_SHADER_MODEL {
            HighestShaderModel: D3D_SHADER_MODEL_6_6,
        };
        // SAFETY: an output value of the size required for this feature query.
        let shader_model = unsafe {
            host.device.CheckFeatureSupport(
                D3D12_FEATURE_SHADER_MODEL,
                (&raw mut sm).cast(),
                size_of_val(&sm) as u32,
            )
        };
        if shader_model.is_err() || sm.HighestShaderModel.0 < D3D_SHADER_MODEL_6_6.0 {
            eprintln!("skipped: the bindless bake shader needs SM 6.6");
            return;
        }

        // 1. Compile fullscreen VS via FXC (`vs_5_1`).
        let vs_src = b"float4 main(uint id : SV_VertexID) : SV_Position {
            return float4(id == 1 ? 3 : -1, id == 2 ? -3 : 1, 0.0, 1.0);
        }";
        let mut vs_blob: Option<ID3DBlob> = None;
        // SAFETY: valid static HLSL source and output blob pointer.
        unsafe {
            D3DCompile(
                vs_src.as_ptr().cast(),
                vs_src.len(),
                PCSTR::null(),
                None,
                None,
                s!("main"),
                s!("vs_5_1"),
                0,
                0,
                &mut vs_blob,
                None,
            )
            .unwrap();
        }
        let vs_blob = vs_blob.unwrap();
        // SAFETY: `vs_blob` owns the compiled bytecode.
        let vs_bytes = unsafe {
            std::slice::from_raw_parts(
                vs_blob.GetBufferPointer().cast::<u8>(),
                vs_blob.GetBufferSize(),
            )
        };
        let vs = Shader::for_test(
            "vertex",
            Translated {
                host: victory_shader::emit::HostShader {
                    stage: HwStage::Vs,
                    dxil: vs_bytes.to_vec(),
                    user_data: Vec::new(),
                    samplers: Vec::new(),
                    static_samplers: Vec::new(),
                    root_cbvs: Vec::new(),
                    root_srvs: Vec::new(),
                    root_uavs: Vec::new(),
                    root_srv_bytes: Vec::new(),
                    memory: false,
                    memory_writes: false,
                    view_addresses: false,
                    heap: false,
                    heap_writes: false,
                    heap_index_sources: Vec::new(),
                    view_marks: false,
                    half_views: Vec::new(),
                },
                used: Vec::new(),
                used_written: Vec::new(),
                write_unknown: true,
                threads: None,
            },
        );

        // 2. Translate a synthetic GFX10 pure-bindless pixel shader that loads
        //    `texIdx` from table slot 0 CBV byte 0, loads `ResourceDescriptorHeap[texIdx]`,
        //    reads texel (0, 0) via `image_load`, and exports `mrt0`.
        let null_code = 125u32;
        let c0 = 128u32;
        let c5 = 128 + 5u32;
        let s_mov = |d: u32, s: u32| 0xBE80_0300 | (d << 16) | s;
        let s_lshl = |d: u32, a: u32, b: u32| (0b10 << 30) | (0x1e << 23) | (d << 16) | (b << 8) | a;
        let s_load = |op: u32, d: u32, b: u32, off: u32, soff: u32| {
            [
                (0x3D << 26) | (op << 18) | (d << 6) | (b >> 1),
                (soff << 25) | (off & 0x1F_FFFF),
            ]
        };
        let mut ps_words = vec![
            s_mov(2, 0),  // s2 = table base lo
            s_mov(3, c0), // s3 = 0
            s_mov(4, 1),  // s4 = ResourceHeap base lo
            s_mov(5, c0), // s5 = 0
        ];
        ps_words.extend(s_load(2, 8, 2, 0, null_code)); // s[8:11] = CBV desc
        ps_words.extend(s_load(8, 12, 8, 0, null_code)); // s12 = cb[0] (tex heap index)
        ps_words.push(s_lshl(13, 12, c5)); // s13 = texIdx * 32
        ps_words.extend(s_load(3, 16, 4, 0, 13)); // s[16:23] = ResourceHeap[texIdx]
        ps_words.push(0x7E00_0280); // v_mov_b32 v0, 0
        ps_words.push(0x7E02_0280); // v_mov_b32 v1, 0
        ps_words.push(0xF000_1F08); // image_load v[2:5], v[0:1], s[16:23] dmask:0xf dim:2D unorm
        ps_words.push(0x0004_0200);
        ps_words.push(0xF800_180F); // exp mrt0 v2, v3, v4, v5 done vm
        ps_words.push(0x0504_0302);
        ps_words.push(0xBF81_0000); // s_endpgm

        let ps_used = vec![UsedResource {
            kind: ResourceType::Cbv,
            root: RootType::Table,
            table_offset_dwords: 0,
            lower_bound: 0,
            stride: 0,
            dwords: 4,
            user_data: 0,
            slot: 0,
        }];
        let ps_native = NativeProgram {
            stage: HwStage::Ps,
            wave64: false,
            vgprs: 8,
            layout: vec![
                RootParam {
                    root: RootType::Table,
                    table_offset_dwords: 0,
                    sgprs: 1,
                    base_register: 0,
                    user_data: 0,
                    slot: 0,
                },
                RootParam {
                    root: RootType::ResourceHeap,
                    table_offset_dwords: 0,
                    sgprs: 1,
                    base_register: 0,
                    user_data: 1,
                    slot: 1,
                },
            ],
            used: ps_used.clone(),
            io: StageIo::Ps {
                inputs: Vec::new(),
                input_addr: 0,
                db_shader_control: 0,
                patches: Vec::new(),
            },
            code: ps_words.iter().flat_map(|w| w.to_le_bytes()).collect(),
        };
        let dxbc = sv_target0_container();
        let container = Container::parse(&dxbc).expect("synthetic container");
        let ir_prog = victory_shader::ir::build(&ps_native.code, false, ps_native.vgprs).unwrap();
        let structured = victory_shader::structure::structure(&ir_prog).unwrap();
        let tracked = victory_shader::resource::track(
            &ir_prog,
            &ps_native.layout,
            ps_native.stage,
            Some(&structured),
        )
        .unwrap();
        let raised_io = victory_shader::io::raise(&ir_prog, &ps_native).unwrap();
        let ps_host = victory_shader::emit::emit(
            &container,
            &ps_native,
            &ir_prog,
            &structured,
            &raised_io,
            &tracked,
        )
        .expect("emit bindless PS");

        assert!(ps_host.heap, "translated PS must index ResourceDescriptorHeap");
        assert_eq!(
            ps_host.heap_index_sources,
            vec![HeapIndexSource::CbvWord {
                binding: BufferBinding::Table {
                    slot: 0,
                    table_offset_dwords: 0,
                },
                byte_offset: 0,
                index_offset: 0,
            }],
            "IR extractor must discover the CBV heap index source"
        );

        let ps = Shader::for_test(
            "pixel",
            Translated {
                host: ps_host,
                used: ps_used,
                used_written: Vec::new(),
                write_unknown: true,
                threads: None,
            },
        );
        let mut gfx = GraphicsState::default();
        gfx.topology = D3D12_PRIMITIVE_TOPOLOGY_TYPE_TRIANGLE;
        gfx.targets.NumRenderTargets = 1;
        gfx.targets.RTFormats[0] = DXGI_FORMAT_R32G32B32A32_FLOAT;
        let refusal = std::cell::Cell::new(None);
        let Some(pipeline) = HostPipeline::new(&[vs, ps], Some(&gfx), |reason, code| {
            refusal.set(Some((reason, code)));
        }) else {
            let (reason, code) = refusal.get().unwrap_or(("unknown", None));
            eprintln!("host bake pipeline refused: {reason} ({code:?})");
            // A refusal that carries a driver error code means this host GPU
            // cannot build the pipeline (the CI runner's software adapter).
            assert!(code.is_some(), "host bake pipeline: {reason}");
            return;
        };

        // 3. Create shader-visible title descriptor heap with 16 slots.
        let desc = [0u32, 16, 1, 0];
        let mut heap_ptr = std::ptr::null();
        let heap_iid = guid(
            0x8efb471d,
            0x616c,
            0x4f49,
            [0x90, 0xf7, 0x12, 0x7b, 0xb7, 0x63, 0xfa, 0x51],
        );
        assert_eq!(
            crate::descriptor::create_heap(std::ptr::null(), desc.as_ptr().cast(), &heap_iid, &mut heap_ptr),
            S_OK
        );
        let title_heap = crate::descriptor::heap(heap_ptr);
        let host_heap = title_heap.host().expect("host view heap");

        // 4. Allocate a 64x64 linear RGBA8 placed texture filled with [0x40, 0x80, 0xC0, 0xFF],
        //    and record a placed SRV at heap slot 5. Because `for_draw = false` defers placed
        //    textures, slot 5 MUST start unmade (`kept(5) == false`).
        let tex_layout = std::alloc::Layout::from_size_align(64 * 64 * 4, 1 << 16).unwrap();
        // SAFETY: non-zero aligned allocation, leaked for the duration of the test.
        let tex_ptr = unsafe { std::alloc::alloc_zeroed(tex_layout) };
        let tex_addr = tex_ptr as u64;
        for px in 0..(64 * 64) {
            // SAFETY: inside `64 * 64 * 4` byte allocation.
            unsafe {
                std::ptr::copy_nonoverlapping(
                    [0x40u8, 0x80, 0xC0, 0xFF].as_ptr(),
                    tex_ptr.add(px * 4),
                    4,
                );
            }
        }
        let tex_words: [u32; 14] = [3, 0, 0, 0, 64, 0, 64, 1 | (1 << 16), 28, 1, 0, 0x100, 0, 0];
        let mut srv_words = [0u32; 16];
        srv_words[..6].copy_from_slice(&[28, 4, 0x1688, 0, 0, 1]);
        srv_words[10] = tex_addr as u32;
        srv_words[11] = (tex_addr >> 32) as u32;
        let tex_slot = 5u32;
        crate::descriptor::create_placed_srv(
            std::ptr::null(),
            &tex_words,
            &srv_words,
            title_heap.start + u64::from(tex_slot) * 32,
        );
        assert!(
            !host_heap.kept(tex_slot),
            "placed texture slot must start unmade (null) before the bake draw"
        );

        // 5. Create CBV buffer containing `tex_slot` (5) at byte offset 0 and record it at slot 0.
        let cb_buf = super::super::tests::buffer(256);
        let (cb_ptr, _) = crate::resource::buffer_bytes(cb_buf).unwrap();
        // SAFETY: 256-byte live title buffer.
        unsafe { (cb_ptr as *mut u32).write(tex_slot) };
        let cb_addr = cb_ptr as u64;
        crate::descriptor::create_cbv(
            std::ptr::null(),
            &[cb_addr as u32, (cb_addr >> 32) as u32, 256, 0],
            title_heap.start,
        );

        // 6. Create 4x4 RGBA32F render target and readback buffer.
        let device = &host.device;
        let rt_desc = D3D12_RESOURCE_DESC {
            Dimension: D3D12_RESOURCE_DIMENSION_TEXTURE2D,
            Width: 4,
            Height: 4,
            DepthOrArraySize: 1,
            MipLevels: 1,
            Format: DXGI_FORMAT_R32G32B32A32_FLOAT,
            SampleDesc: DXGI_SAMPLE_DESC {
                Count: 1,
                Quality: 0,
            },
            Flags: D3D12_RESOURCE_FLAG_ALLOW_RENDER_TARGET,
            ..Default::default()
        };
        let default_heap = D3D12_HEAP_PROPERTIES {
            Type: D3D12_HEAP_TYPE_DEFAULT,
            ..Default::default()
        };
        let mut rt: Option<ID3D12Resource> = None;
        // SAFETY: valid resource desc and out pointer.
        unsafe {
            device
                .CreateCommittedResource(
                    &default_heap,
                    D3D12_HEAP_FLAG_NONE,
                    &rt_desc,
                    D3D12_RESOURCE_STATE_RENDER_TARGET,
                    None,
                    &mut rt,
                )
                .unwrap();
        }
        let rt = rt.unwrap();
        let rtv = host.rtv(&rt, None);

        let rb_desc = D3D12_RESOURCE_DESC {
            Dimension: D3D12_RESOURCE_DIMENSION_BUFFER,
            Width: 256 * 4,
            Height: 1,
            DepthOrArraySize: 1,
            MipLevels: 1,
            SampleDesc: DXGI_SAMPLE_DESC {
                Count: 1,
                Quality: 0,
            },
            Layout: D3D12_TEXTURE_LAYOUT_ROW_MAJOR,
            ..Default::default()
        };
        let rb_heap = D3D12_HEAP_PROPERTIES {
            Type: D3D12_HEAP_TYPE_READBACK,
            ..Default::default()
        };
        let mut readback: Option<ID3D12Resource> = None;
        // SAFETY: valid readback buffer desc and out pointer.
        unsafe {
            device
                .CreateCommittedResource(
                    &rb_heap,
                    D3D12_HEAP_FLAG_NONE,
                    &rb_desc,
                    D3D12_RESOURCE_STATE_COPY_DEST,
                    None,
                    &mut readback,
                )
                .unwrap();
        }
        let readback = readback.unwrap();

        // 7. Record and execute the pure-bindless draw ONCE.
        let list = host.new_list().unwrap();
        let mut state = State::default();
        state.heaps[0] = heap_ptr as usize;
        state.graphics = vec![Root::Table(title_heap.start)];

        // SAFETY: open command list with live render target.
        unsafe {
            list.list.ClearRenderTargetView(rtv, &[0.0, 0.0, 0.0, 0.0], None);
            list.list.OMSetRenderTargets(1, Some(&rtv), false, None);
            list.list.RSSetViewports(&[D3D12_VIEWPORT {
                Width: 4.0,
                Height: 4.0,
                MaxDepth: 1.0,
                ..Default::default()
            }]);
            list.list.RSSetScissorRects(&[RECT {
                left: 0,
                top: 0,
                right: 4,
                bottom: 4,
            }]);
            list.list
                .IASetPrimitiveTopology(D3D_PRIMITIVE_TOPOLOGY_TRIANGLELIST);
        }

        assert!(
            state.bind(&list, &pipeline, false),
            "bake draw bind must succeed"
        );
        assert!(
            host_heap.kept(tex_slot),
            "prepare_bake_sources must synchronously mirror the placed texture slot before the draw"
        );

        // SAFETY: open list with pipeline, root signature, heaps, and RTV bound.
        unsafe {
            list.list.DrawInstanced(3, 1, 0, 0);
            list.list.ResourceBarrier(&[crate::host::transition(
                &rt,
                D3D12_RESOURCE_STATE_RENDER_TARGET,
                D3D12_RESOURCE_STATE_COPY_SOURCE,
            )]);
            let from = D3D12_TEXTURE_COPY_LOCATION {
                pResource: std::mem::transmute_copy(&rt),
                Type: D3D12_TEXTURE_COPY_TYPE_SUBRESOURCE_INDEX,
                Anonymous: D3D12_TEXTURE_COPY_LOCATION_0 {
                    SubresourceIndex: 0,
                },
            };
            let to = D3D12_TEXTURE_COPY_LOCATION {
                pResource: std::mem::transmute_copy(&readback),
                Type: D3D12_TEXTURE_COPY_TYPE_PLACED_FOOTPRINT,
                Anonymous: D3D12_TEXTURE_COPY_LOCATION_0 {
                    PlacedFootprint: D3D12_PLACED_SUBRESOURCE_FOOTPRINT {
                        Offset: 0,
                        Footprint: D3D12_SUBRESOURCE_FOOTPRINT {
                            Format: DXGI_FORMAT_R32G32B32A32_FLOAT,
                            Width: 4,
                            Height: 4,
                            Depth: 1,
                            RowPitch: 256,
                        },
                    },
                },
            };
            list.list.CopyTextureRegion(&to, 0, 0, 0, &from, None);
        }
        list.close();
        host.execute(&[&list]);

        // 8. Verify the very first execution rendered the real placed-texture texels!
        let mut mapped = std::ptr::null_mut();
        // SAFETY: GPU execution finished; readback buffer holds 4x4 RGBA32F pixels.
        let pixel = unsafe {
            readback.Map(0, None, Some(&mut mapped)).unwrap();
            let px = (mapped as *const [f32; 4]).read();
            readback.Unmap(0, None);
            px
        };
        let want = [0x40 as f32 / 255.0, 0x80 as f32 / 255.0, 0xC0 as f32 / 255.0, 1.0];
        for (c, (g, w)) in pixel.iter().zip(want).enumerate() {
            assert!(
                (g - w).abs() < 1e-2,
                "channel {c}: got {g}, expected {w} (full pixel {pixel:?})"
            );
        }
    }
}
