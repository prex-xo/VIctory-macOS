//! GPU refreshes of widened three-channel buffer views after UAV writes.
//!
//! D3DMetal reads `R32G32B32` SRVs as `R32G32`. Their host views therefore use
//! a separate 16-byte-element copy. A raw or typed UAV still writes the normal
//! 12-byte shadow, so a buffer barrier also widens the overlapping elements
//! before a later SRV in the same submission reads them.

use std::collections::HashSet;
use std::sync::{Arc, OnceLock};

use windows::Win32::Graphics::Direct3D12::*;
use windows::core::{Interface, s};

use crate::backend::HostMemory;
use crate::compute::{Param, pipeline};
use crate::host::{HostList, transition, uav_barrier};
use crate::shadow::{Shadow, Widened};

const SOURCE: &str = r#"
cbuffer Copy : register(b0) { uint first; uint count; uint w; uint unused; };
ByteAddressBuffer src : register(t0);
RWByteAddressBuffer dst : register(u0);
[numthreads(64, 1, 1)]
void main(uint3 id : SV_DispatchThreadID) {
    uint i = id.x;
    if (i < count) {
        uint element = first + i;
        uint3 value = src.Load3(element * 12);
        dst.Store4(element * 16, uint4(value, w));
    }
}
"#;

const PATCH_SOURCE: &str = r#"
cbuffer Copy : register(b0) { uint first; uint count; uint segments; uint w; };
ByteAddressBuffer origin : register(t0);
ByteAddressBuffer patches : register(t1);
ByteAddressBuffer metadata : register(t2);
RWByteAddressBuffer dst : register(u0);
[numthreads(64, 1, 1)]
void main(uint3 id : SV_DispatchThreadID) {
    uint i = id.x;
    if (i < count) {
        uint element = first + i;
        uint4 value = uint4(origin.Load3(element * 12), w);
        for (uint j = 0; j < segments; ++j) {
            uint destination = metadata.Load(j * 12);
            uint source = metadata.Load(j * 12 + 4);
            uint bytes = metadata.Load(j * 12 + 8);
            uint begin = max(element * 12, destination);
            uint end = min(element * 12 + 12, destination + bytes);
            for (uint at = begin; at < end; ++at) {
                uint b = at - element * 12;
                uint component = b / 4;
                uint word;
                if (component == 0) word = value.x;
                else if (component == 1) word = value.y;
                else if (component == 2) word = value.z;
                else word = value.w;
                uint shift = (b % 4) * 8;
                uint address = source + at - destination;
                uint loaded = patches.Load(address & ~3u);
                uint byte = (loaded >> ((address & 3u) * 8)) & 0xffu;
                word = (word & ~(0xffu << shift)) | (byte << shift);
                if (component == 0) value.x = word;
                else if (component == 1) value.y = word;
                else if (component == 2) value.z = word;
                else value.w = word;
            }
        }
        dst.Store4(element * 16, value);
    }
}
"#;

struct Copier {
    root: ID3D12RootSignature,
    pso: ID3D12PipelineState,
    patch: ID3D12PipelineState,
}

// SAFETY: D3D12 objects are free-threaded.
unsafe impl Send for Copier {}
// SAFETY: as above.
unsafe impl Sync for Copier {}

fn copier() -> Option<&'static Copier> {
    static COPIER: OnceLock<Option<Copier>> = OnceLock::new();
    COPIER
        .get_or_init(|| {
            let params = [
                Param::Constants {
                    register: 0,
                    count: 4,
                },
                Param::Srv {
                    register: 0,
                    space: 0,
                },
                Param::Uav { register: 0 },
                Param::Srv {
                    register: 1,
                    space: 0,
                },
                Param::Srv {
                    register: 2,
                    space: 0,
                },
            ];
            let made = pipeline(SOURCE, s!("widen rgb32"), 0, &params).and_then(|(root, pso)| {
                let patch = pipeline(PATCH_SOURCE, s!("patch rgb32"), 0, &params)?.1;
                Ok((root, pso, patch))
            });
            match made {
                Ok((root, pso, patch)) => Some(Copier { root, pso, patch }),
                Err(e) => {
                    tracing::warn!(error = %e, "no three-channel widening copy");
                    None
                }
            }
        })
        .as_ref()
}

#[derive(Clone, Copy, PartialEq, Eq)]
enum JobKind {
    /// Whole elements from `first` to the exclusive `last` index.
    Widen { first: usize, last: usize },
    /// Only the bytes of a write that cross a shadow boundary.
    Patch { from: usize, to: usize },
}

struct Job {
    shadow: Arc<Shadow>,
    view: Arc<Widened>,
    kind: JobKind,
}

struct Segment {
    shadow: Arc<Shadow>,
    from: usize,
    to: usize,
    /// Byte offset from the widened view's start in the source's 12-byte layout.
    destination: u32,
    staging: u32,
    skip: u32,
    copy_offset: u64,
    copy_len: u32,
}

struct Prepared {
    segments: Vec<Segment>,
    patches: ID3D12Resource,
    metadata: ID3D12Resource,
}

enum GroupKind {
    Widen {
        shadow: Arc<Shadow>,
        first: usize,
        last: usize,
    },
    Patch(Prepared),
}

struct Group {
    view: Arc<Widened>,
    kind: GroupKind,
}

/// Refreshes the widened views touched by writes before a buffer barrier:
/// `new`, the writes since the last one, after `history`, those of the
/// list's earlier barriers (two slices, not one list made of both: in the
/// world each barrier copied a history of ~90,000 writes and grew the copy,
/// ~55 blocks of 2-4 MB a frame). History is replayed for new views and
/// views touched by this barrier, so a partial patch keeps widened-only
/// bytes from earlier barriers.
pub(crate) fn sync(
    list: &HostList,
    history: &[(Arc<Shadow>, usize, usize)],
    new: &[(Arc<Shadow>, usize, usize)],
    ready: &HashSet<usize>,
) -> Option<Vec<usize>> {
    let views = crate::shadow::widened_all();
    let keys: Vec<_> = views
        .iter()
        .map(|view| view.buffer.as_raw() as usize)
        .collect();
    if views.is_empty() {
        return Some(keys);
    }
    // Ready keys are host addresses. Keep every view considered alive until
    // the list resets so a released view's address cannot be reused for a
    // different, not-yet-current cache entry.
    for view in &views {
        list.keep(&view.buffer);
    }
    let pending_keys: HashSet<_> = new
        .iter()
        .flat_map(|(_, from, len)| {
            let to = from.saturating_add(*len);
            views.iter().filter_map(move |view| {
                view.overlap(*from, to)
                    .map(|_| view.buffer.as_raw() as usize)
            })
        })
        .collect();
    let history_views: Vec<_> = views
        .iter()
        .filter(|view| {
            !ready.contains(&(view.buffer.as_raw() as usize))
                || pending_keys.contains(&(view.buffer.as_raw() as usize))
        })
        .cloned()
        .collect();
    // With no view to replay it to, the history makes no job.
    let replayed = if history_views.is_empty() {
        &[][..]
    } else {
        history
    };
    let writes = replayed
        .iter()
        .map(|write| (write, &history_views))
        .chain(new.iter().map(|write| (write, &views)));
    let mut jobs = Vec::new();
    for ((shadow, from, len), candidates) in writes {
        let to = from.saturating_add(*len);
        if *from >= to {
            continue;
        }
        let (start, size) = shadow.range();
        let end = start.saturating_add(size);
        for view in candidates {
            let Some((first, last)) = view.overlap(*from, to) else {
                continue;
            };
            let (Some(element_first), Some(element_last)) = (
                view.start.checked_add(first as u64 * 12),
                view.start.checked_add(last as u64 * 12),
            ) else {
                continue;
            };
            if view.start >= start && element_first >= start && element_last <= end {
                jobs.push(Job {
                    shadow: shadow.clone(),
                    view: view.clone(),
                    kind: JobKind::Widen { first, last },
                });
            } else {
                let view_end = view.start.saturating_add(view.len);
                let patch_from = (*from).max(view.start as usize);
                let patch_to = to.min(view_end as usize);
                if patch_from < patch_to && patch_from as u64 >= start && patch_to as u64 <= end {
                    jobs.push(Job {
                        shadow: shadow.clone(),
                        view: view.clone(),
                        kind: JobKind::Patch {
                            from: patch_from,
                            to: patch_to,
                        },
                    });
                }
            }
        }
    }
    let mut grouped: Vec<(Arc<Widened>, Vec<Job>)> = Vec::new();
    for job in jobs {
        if let Some((_, jobs)) = grouped
            .iter_mut()
            .find(|(view, _)| Arc::ptr_eq(view, &job.view))
        {
            jobs.push(job);
        } else {
            grouped.push((job.view.clone(), vec![job]));
        }
    }
    let mut groups = Vec::new();
    for (view, jobs) in grouped {
        // At this barrier repeated writes of the same whole elements are
        // all read from the same current shadow. Use the direct GPU copy.
        let job = &jobs[0];
        let kind = if let JobKind::Widen { first, last } = job.kind
            && jobs[1..]
                .iter()
                .all(|other| Arc::ptr_eq(&job.shadow, &other.shadow) && job.kind == other.kind)
        {
            if u32::try_from(first).is_err()
                || u32::try_from(last).is_err()
                || last as u64 * 12 > u64::from(u32::MAX)
            {
                return None;
            }
            GroupKind::Widen {
                shadow: job.shadow.clone(),
                first,
                last,
            }
        } else {
            if view.len > u64::from(u32::MAX) || u32::try_from(view.len / 12).is_err() {
                return None;
            }
            let segments = jobs
                .iter()
                .map(|job| {
                    let (from, to) = match job.kind {
                        JobKind::Widen { first, last } => (
                            usize::try_from(job.view.start.checked_add(first as u64 * 12)?).ok()?,
                            usize::try_from(job.view.start.checked_add(last as u64 * 12)?).ok()?,
                        ),
                        JobKind::Patch { from, to } => (from, to),
                    };
                    let relative = from.checked_sub(job.view.start as usize)?;
                    let destination = u32::try_from(relative).ok()?;
                    Some(Segment {
                        shadow: job.shadow.clone(),
                        from,
                        to,
                        destination,
                        staging: 0,
                        skip: 0,
                        copy_offset: 0,
                        copy_len: 0,
                    })
                })
                .collect::<Option<Vec<_>>>()?;
            GroupKind::Patch(prepare(segments)?)
        };
        groups.push(Group { view, kind });
    }
    if groups.is_empty() {
        return Some(keys);
    }
    let copier = copier()?;
    let mut buffers = Vec::new();
    let mut unique = HashSet::new();
    for group in &groups {
        let key = group.view.buffer.as_raw() as usize;
        if unique.insert(key) {
            buffers.push(group.view.buffer.clone());
        }
    }
    list.widen_begin_all(&buffers);
    for group in groups {
        match group.kind {
            GroupKind::Widen {
                shadow,
                first,
                last,
            } => {
                record(list, copier, &shadow, &group.view, first, last);
            }
            GroupKind::Patch(prepared) => {
                patch(list, copier, &group.view, prepared);
            }
        }
    }
    // SAFETY: an open list and the widened buffers the copies just wrote.
    unsafe {
        list.list.ResourceBarrier(&[uav_barrier()]);
    }
    list.widen_end_all(&buffers);
    list.need_rebind();
    Some(keys)
}

fn prepare(mut segments: Vec<Segment>) -> Option<Prepared> {
    let host = crate::host::host()?;
    let mut bytes = 0u64;
    for segment in &mut segments {
        let len = u64::try_from(segment.to.checked_sub(segment.from)?).ok()?;
        u32::try_from(len).ok()?.checked_add(segment.destination)?;
        let source = segment.shadow.offset(segment.from as u64);
        let copy_offset = source & !3;
        let skip = u32::try_from(source - copy_offset).ok()?;
        let copy_len = (u64::from(skip) + len + 3) & !3;
        let copy_len = u32::try_from(copy_len).ok()?;
        // SAFETY: a live source buffer.
        let width = unsafe { segment.shadow.buffer.GetDesc() }.Width;
        if copy_offset
            .checked_add(u64::from(copy_len))
            .is_none_or(|end| end > width)
        {
            return None;
        }
        bytes = bytes.checked_add(u64::from(copy_len))?;
        if bytes > u64::from(u32::MAX) {
            return None;
        }
        segment.staging = u32::try_from(bytes - u64::from(copy_len)).ok()?;
        segment.skip = skip;
        segment.copy_offset = copy_offset;
        segment.copy_len = copy_len;
    }
    let count = u32::try_from(segments.len()).ok()?;
    let metadata_bytes = u64::from(count) * 12;
    let patches = host.default_buffer(bytes)?;
    let (metadata, data) = host.upload_buffer(metadata_bytes)?;
    for (i, segment) in segments.iter().enumerate() {
        let len = u32::try_from(segment.to - segment.from).ok()?;
        let source = segment.staging.checked_add(segment.skip)?;
        let record = [segment.destination, source, len];
        // SAFETY: three dwords per segment inside the mapped metadata buffer.
        unsafe { data.cast::<[u32; 3]>().add(i).write_unaligned(record) };
    }
    Some(Prepared {
        segments,
        patches,
        metadata,
    })
}

fn record(
    list: &HostList,
    copier: &Copier,
    shadow: &Arc<Shadow>,
    view: &Widened,
    first: usize,
    last: usize,
) {
    let (Ok(first), Ok(last)) = (u32::try_from(first), u32::try_from(last)) else {
        return;
    };
    let count = last - first;
    list.keep_shadow(shadow);
    list.keep(&shadow.buffer);
    list.keep(&view.buffer);
    list.keep(&copier.root);
    list.keep(&copier.pso);
    // SAFETY: a live widened buffer.
    let destination = unsafe { view.buffer.GetGPUVirtualAddress() };
    // One dispatch dimension is at most 65,535 groups.
    const MAX_ELEMENTS: usize = 65_535 * 64;
    let mut done = 0usize;
    while done < count as usize {
        let n = (count as usize - done).min(MAX_ELEMENTS) as u32;
        let constants = [first + done as u32, n, view.w, 0];
        // SAFETY: an open list, our pipeline and live buffers. The caller
        // checked that the source range starts inside the writer shadow.
        unsafe {
            list.list.SetComputeRootSignature(&copier.root);
            list.list.SetPipelineState(&copier.pso);
            list.list
                .SetComputeRoot32BitConstants(0, 4, constants.as_ptr().cast(), 0);
            list.list
                .SetComputeRootShaderResourceView(1, shadow.gpu(view.start));
            list.list.SetComputeRootUnorderedAccessView(2, destination);
            list.list.Dispatch(n.div_ceil(64), 1, 1);
        }
        done += n as usize;
    }
    #[cfg(feature = "diagnostics")]
    list.note_op(|| format!("widen rgb32 {count} elements at {first}"));
}

fn patch(list: &HostList, copier: &Copier, view: &Widened, prepared: Prepared) {
    let origin = view.source();
    list.keep_shadow(&origin);
    list.keep(&origin.buffer);
    list.keep(&prepared.patches);
    list.keep(&prepared.metadata);
    list.keep(&view.buffer);
    list.keep(&copier.root);
    list.keep(&copier.patch);
    for segment in &prepared.segments {
        list.keep_shadow(&segment.shadow);
        // SAFETY: an open list and live buffers; the source range is inside
        // the shadow and the destination inside the staging buffer.
        unsafe {
            list.list.CopyBufferRegion(
                &prepared.patches,
                u64::from(segment.staging),
                &segment.shadow.buffer,
                segment.copy_offset,
                u64::from(segment.copy_len),
            );
        }
    }
    // SAFETY: an open list and the staging buffer the copies just wrote.
    unsafe {
        list.list.ResourceBarrier(&[transition(
            &prepared.patches,
            D3D12_RESOURCE_STATE_COPY_DEST,
            D3D12_RESOURCE_STATE_ALL_SHADER_RESOURCE,
        )]);
    }
    // SAFETY: live buffers.
    let (destination, patches, metadata) = unsafe {
        (
            view.buffer.GetGPUVirtualAddress(),
            prepared.patches.GetGPUVirtualAddress(),
            prepared.metadata.GetGPUVirtualAddress(),
        )
    };
    let segments = prepared.segments.len() as u32;
    let count = (view.len / 12) as usize;
    let mut first = 0u32;
    // One dispatch dimension is at most 65,535 groups.
    const MAX_ELEMENTS: usize = 65_535 * 64;
    while (first as usize) < count {
        let n = (count - first as usize).min(MAX_ELEMENTS) as u32;
        let constants = [first, n, segments, view.w];
        // SAFETY: an open list, our pipeline and live buffers. Each thread
        // rebuilds one element from the origin and the ordered patch stream.
        unsafe {
            list.list.SetComputeRootSignature(&copier.root);
            list.list.SetPipelineState(&copier.patch);
            list.list
                .SetComputeRoot32BitConstants(0, 4, constants.as_ptr().cast(), 0);
            list.list
                .SetComputeRootShaderResourceView(1, origin.gpu(view.start));
            list.list.SetComputeRootUnorderedAccessView(2, destination);
            list.list.SetComputeRootShaderResourceView(3, patches);
            list.list.SetComputeRootShaderResourceView(4, metadata);
            list.list.Dispatch(n.div_ceil(64), 1, 1);
        }
        first += n;
    }
    #[cfg(feature = "diagnostics")]
    list.note_op(|| format!("patch rgb32 {segments} ranges over {count} elements"));
}
