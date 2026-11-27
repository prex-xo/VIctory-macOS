//! Opt-in single-frame HDR luminance and non-finite pass probe.
//!
//! Enabled only when `VICTORY_HDR_PROBE=<world_offset>` is set (e.g.
//! `VICTORY_HDR_PROBE=24`); zero overhead when unset and zero overhead on
//! every frame other than the single targeted world frame.
//!
//! On the target world frame (counting from the first frame that executes
//! at least 500 host draws), each render pass (on render-target switch or
//! command-list close) and each compute UAV texture pass copies slice 0 of
//! its written HDR/color target into a temporary readback buffer and logs
//! one compact line with format, dimensions, shader hashes, finite min/p50/
//! p95/max, `nan_count`, `inf_count`, `neg_count`, `gt100_count`, and the
//! center-pixel RGBA. Bounded to [`MAX_PROBES`] targets per frame so it
//! completes quickly without exhausting memory.

use std::sync::OnceLock;
use std::sync::atomic::{AtomicBool, AtomicU64, AtomicUsize, Ordering};

/// Default maximum number of target readbacks recorded during the probed frame.
pub(crate) const MAX_PROBES: usize = 80;

/// Maximum extra frames to keep waiting if the target frame executes only
/// pre-recorded command lists (recorded one frame ahead by pipelined workers).
const MAX_WAIT_FRAMES: u64 = 4;

/// Maximum probes configurable via `VICTORY_HDR_PROBE_MAX`.
const MAX_PROBES_LIMIT: usize = 512;

/// Draw-count threshold identifying the first rendered 3D world frame
/// (matches `capture::presented`).
const WORLD_FRAME_MIN_DRAWS: u64 = 500;

/// Maximum samples collected along each axis for percentile (`p50`/`p95`)
/// estimation; `min`/`max`/`nan`/`inf`/`neg`/`gt100` always scan 100% of
/// pixels so localized hot pixels or non-finites are never missed.
const PERCENTILE_GRID: usize = 128;

/// Metadata for a supported HDR or color DXGI format.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) struct FormatInfo {
    pub(crate) name: &'static str,
    pub(crate) bytes_per_pixel: usize,
    /// Number of color channels included in luminance/range statistics (1..=3).
    pub(crate) stat_channels: usize,
}

/// Returns decoding metadata for supported HDR/color DXGI format numbers, or
/// `None` for depth/stencil/block-compressed/integer formats.
pub(crate) const fn format_info(format: u32) -> Option<FormatInfo> {
    Some(match format {
        1 | 2 => FormatInfo {
            name: "R32G32B32A32_FLOAT",
            bytes_per_pixel: 16,
            stat_channels: 3,
        },
        9 | 10 => FormatInfo {
            name: "R16G16B16A16_FLOAT",
            bytes_per_pixel: 8,
            stat_channels: 3,
        },
        15 | 16 => FormatInfo {
            name: "R32G32_FLOAT",
            bytes_per_pixel: 8,
            stat_channels: 2,
        },
        23 | 24 => FormatInfo {
            name: "R10G10B10A2_UNORM",
            bytes_per_pixel: 4,
            stat_channels: 3,
        },
        26 => FormatInfo {
            name: "R11G11B10_FLOAT",
            bytes_per_pixel: 4,
            stat_channels: 3,
        },
        27 | 28 => FormatInfo {
            name: "R8G8B8A8_UNORM",
            bytes_per_pixel: 4,
            stat_channels: 3,
        },
        29 => FormatInfo {
            name: "R8G8B8A8_UNORM_SRGB",
            bytes_per_pixel: 4,
            stat_channels: 3,
        },
        33 | 34 => FormatInfo {
            name: "R16G16_FLOAT",
            bytes_per_pixel: 4,
            stat_channels: 2,
        },
        39 | 41 => FormatInfo {
            name: "R32_FLOAT",
            bytes_per_pixel: 4,
            stat_channels: 1,
        },
        53 | 54 => FormatInfo {
            name: "R16_FLOAT",
            bytes_per_pixel: 2,
            stat_channels: 1,
        },
        87 => FormatInfo {
            name: "B8G8R8A8_UNORM",
            bytes_per_pixel: 4,
            stat_channels: 3,
        },
        88 => FormatInfo {
            name: "B8G8R8X8_UNORM",
            bytes_per_pixel: 4,
            stat_channels: 3,
        },
        91 => FormatInfo {
            name: "B8G8R8A8_UNORM_SRGB",
            bytes_per_pixel: 4,
            stat_channels: 3,
        },
        93 => FormatInfo {
            name: "B8G8R8X8_UNORM_SRGB",
            bytes_per_pixel: 4,
            stat_channels: 3,
        },
        _ => return None,
    })
}

/// Decodes an IEEE-754 binary16 half-precision float.
pub(crate) fn decode_f16(bits: u16) -> f32 {
    let sign = u32::from(bits >> 15);
    let exp = i32::from((bits >> 10) & 0x1f);
    let man = f32::from(bits & 0x3ff);
    let mag = match exp {
        0 => man * 2f32.powi(-24),
        31 if man != 0.0 => f32::NAN,
        31 => f32::INFINITY,
        e => (1.0 + man / 1024.0) * 2f32.powi(e - 15),
    };
    if sign != 0 { -mag } else { mag }
}

/// Decodes an unsigned 11-bit (`mantissa_bits = 6`) or 10-bit
/// (`mantissa_bits = 5`) floating-point channel from `R11G11B10_FLOAT`.
pub(crate) fn decode_ufloat(bits: u32, mantissa_bits: u32) -> f32 {
    let mask = (1u32 << mantissa_bits) - 1;
    let exp = (bits >> mantissa_bits) & 0x1f;
    let man = (bits & mask) as f32;
    let scale = (1u32 << mantissa_bits) as f32;
    match exp {
        0 => (man / scale) * 2f32.powi(-14),
        31 if man != 0.0 => f32::NAN,
        31 => f32::INFINITY,
        e => (1.0 + man / scale) * 2f32.powi(e as i32 - 15),
    }
}

/// Decodes one pixel at `px` into `[r, g, b, a]` for a supported DXGI format.
pub(crate) fn decode_pixel(format: u32, px: &[u8]) -> Option<[f32; 4]> {
    let u16_at = |i: usize| u16::from_le_bytes([px[2 * i], px[2 * i + 1]]);
    let u32_at = |i: usize| u32::from_le_bytes([px[4 * i], px[4 * i + 1], px[4 * i + 2], px[4 * i + 3]]);
    Some(match format {
        1 | 2 => [
            f32::from_bits(u32_at(0)),
            f32::from_bits(u32_at(1)),
            f32::from_bits(u32_at(2)),
            f32::from_bits(u32_at(3)),
        ],
        9 | 10 => [
            decode_f16(u16_at(0)),
            decode_f16(u16_at(1)),
            decode_f16(u16_at(2)),
            decode_f16(u16_at(3)),
        ],
        15 | 16 => [f32::from_bits(u32_at(0)), f32::from_bits(u32_at(1)), 0.0, 1.0],
        23 | 24 => {
            let v = u32_at(0);
            [
                (v & 0x3ff) as f32 / 1023.0,
                ((v >> 10) & 0x3ff) as f32 / 1023.0,
                ((v >> 20) & 0x3ff) as f32 / 1023.0,
                ((v >> 30) & 0x3) as f32 / 3.0,
            ]
        }
        26 => {
            let v = u32_at(0);
            [
                decode_ufloat(v & 0x7ff, 6),
                decode_ufloat((v >> 11) & 0x7ff, 6),
                decode_ufloat((v >> 22) & 0x3ff, 5),
                1.0,
            ]
        }
        27..=29 => [
            f32::from(px[0]) / 255.0,
            f32::from(px[1]) / 255.0,
            f32::from(px[2]) / 255.0,
            f32::from(px[3]) / 255.0,
        ],
        33 | 34 => [decode_f16(u16_at(0)), decode_f16(u16_at(1)), 0.0, 1.0],
        39 | 41 => [f32::from_bits(u32_at(0)), 0.0, 0.0, 1.0],
        53 | 54 => [decode_f16(u16_at(0)), 0.0, 0.0, 1.0],
        87 | 91 => [
            f32::from(px[2]) / 255.0,
            f32::from(px[1]) / 255.0,
            f32::from(px[0]) / 255.0,
            f32::from(px[3]) / 255.0,
        ],
        88 | 93 => [
            f32::from(px[2]) / 255.0,
            f32::from(px[1]) / 255.0,
            f32::from(px[0]) / 255.0,
            1.0,
        ],
        _ => return None,
    })
}

/// Summary statistics for one probed texture slice.
#[derive(Clone, Copy, Debug, PartialEq)]
pub(crate) struct HdrStats {
    pub(crate) min: f32,
    pub(crate) p50: f32,
    pub(crate) p95: f32,
    pub(crate) max: f32,
    pub(crate) nan_count: u64,
    pub(crate) inf_count: u64,
    pub(crate) neg_count: u64,
    pub(crate) gt100_count: u64,
    pub(crate) center: [f32; 4],
}

/// Computes HDR statistics over a 2D pixel buffer (`width x height`, `pitch`
/// bytes per row) in `format`.
///
/// `min`, `max`, `nan_count`, `inf_count`, `neg_count`, and `gt100_count` scan
/// every pixel's color channels (`stat_channels`); `p50` and `p95` are computed
/// over a bounded uniform grid of finite channel samples so large targets stay
/// fast and allocation-light.
pub(crate) fn compute_stats(
    format: u32,
    width: usize,
    height: usize,
    pitch: usize,
    rows: &[u8],
) -> Option<HdrStats> {
    let info = format_info(format)?;
    if width == 0 || height == 0 || pitch < width.checked_mul(info.bytes_per_pixel)? {
        return None;
    }
    let required = pitch.checked_mul(height)?;
    if rows.len() < required {
        return None;
    }

    let step_x = (width / PERCENTILE_GRID).max(1);
    let step_y = (height / PERCENTILE_GRID).max(1);
    let est_samples =
        width.div_ceil(step_x) * height.div_ceil(step_y) * info.stat_channels;
    let mut samples = Vec::with_capacity(est_samples);

    let mut min = f32::INFINITY;
    let mut max = f32::NEG_INFINITY;
    let mut nan_count = 0u64;
    let mut inf_count = 0u64;
    let mut neg_count = 0u64;
    let mut gt100_count = 0u64;

    for y in 0..height {
        let row = &rows[y * pitch..][..width * info.bytes_per_pixel];
        let sample_y = y % step_y == 0;
        for x in 0..width {
            let px = &row[x * info.bytes_per_pixel..][..info.bytes_per_pixel];
            let rgba = decode_pixel(format, px)?;
            let take = sample_y && (x % step_x == 0);
            for &v in &rgba[..info.stat_channels] {
                if v.is_nan() {
                    nan_count += 1;
                } else if v.is_infinite() {
                    inf_count += 1;
                } else {
                    if v < min {
                        min = v;
                    }
                    if v > max {
                        max = v;
                    }
                    if v < 0.0 {
                        neg_count += 1;
                    }
                    if v > 100.0 {
                        gt100_count += 1;
                    }
                    if take {
                        samples.push(v);
                    }
                }
            }
        }
    }

    let (min, max) = if min <= max { (min, max) } else { (0.0, 0.0) };
    let (p50, p95) = if samples.is_empty() {
        (0.0, 0.0)
    } else {
        let nth = |buf: &mut [f32], q: f64| -> f32 {
            let k = ((buf.len() - 1) as f64 * q).round() as usize;
            *buf.select_nth_unstable_by(k, f32::total_cmp).1
        };
        let p50 = nth(&mut samples, 0.50);
        let p95 = nth(&mut samples, 0.95);
        (p50, p95)
    };

    let cx = width / 2;
    let cy = height / 2;
    let cpx = &rows[cy * pitch + cx * info.bytes_per_pixel..][..info.bytes_per_pixel];
    let center = decode_pixel(format, cpx)?;

    Some(HdrStats {
        min,
        p50,
        p95,
        max,
        nan_count,
        inf_count,
        neg_count,
        gt100_count,
        center,
    })
}

/// Parses `VICTORY_HDR_PROBE=<world_offset>`.
fn parse_offset(raw: &str) -> Option<u64> {
    let trimmed = raw.trim();
    if trimmed.is_empty() {
        return None;
    }
    trimmed.parse::<u64>().ok()
}

/// Configured world-frame offset (`VICTORY_HDR_PROBE`), evaluated once.
fn configured_offset() -> Option<u64> {
    static OFFSET: OnceLock<Option<u64>> = OnceLock::new();
    *OFFSET.get_or_init(|| {
        std::env::var("VICTORY_HDR_PROBE")
            .ok()
            .and_then(|v| parse_offset(&v))
    })
}

/// Whether `VICTORY_HDR_PROBE` was enabled via environment variable.
#[inline]
pub(crate) fn enabled() -> bool {
    configured_offset().is_some()
}

/// Configured probe cap (`VICTORY_HDR_PROBE_MAX`, defaults to [`MAX_PROBES`]).
fn configured_max_probes() -> usize {
    static MAX: OnceLock<usize> = OnceLock::new();
    *MAX.get_or_init(|| {
        std::env::var("VICTORY_HDR_PROBE_MAX")
            .ok()
            .and_then(|v| v.trim().parse::<usize>().ok())
            .map_or(MAX_PROBES, |n| n.clamp(1, MAX_PROBES_LIMIT))
    })
}

/// Sentinel indicating the target frame has not yet been armed.
const UNARMED: u64 = u64::MAX;

/// Single-frame probe state machine (also directly unit-tested).
struct ProbeClock {
    last_drawn: AtomicU64,
    world_start: AtomicU64,
    target_frame: AtomicU64,
    done: AtomicBool,
    probed: AtomicUsize,
    skipped_no_copy: AtomicUsize,
    skipped_twin: AtomicUsize,
    skipped_dim_or_msaa: AtomicUsize,
    skipped_format: AtomicUsize,
    skipped_copy: AtomicUsize,
}

impl ProbeClock {
    const fn new() -> Self {
        Self {
            last_drawn: AtomicU64::new(0),
            world_start: AtomicU64::new(UNARMED),
            target_frame: AtomicU64::new(UNARMED),
            done: AtomicBool::new(false),
            probed: AtomicUsize::new(0),
            skipped_no_copy: AtomicUsize::new(0),
            skipped_twin: AtomicUsize::new(0),
            skipped_dim_or_msaa: AtomicUsize::new(0),
            skipped_format: AtomicUsize::new(0),
            skipped_copy: AtomicUsize::new(0),
        }
    }

    /// Updates state on presentation of `frame` with cumulative `total_drawn`.
    fn on_present(&self, offset: u64, max_probes: usize, frame: u64, total_drawn: u64) {
        if self.done.load(Ordering::Relaxed) {
            return;
        }
        let prev = self.last_drawn.swap(total_drawn, Ordering::Relaxed);
        let draws = total_drawn.saturating_sub(prev);
        let target = self.target_frame.load(Ordering::Relaxed);
        if target == UNARMED {
            if draws >= WORLD_FRAME_MIN_DRAWS {
                // `frame` was just presented; offset 0 targets the immediately
                // following frame so inline recording hooks can capture it.
                let armed = frame.saturating_add(offset.max(1));
                self.world_start.store(frame, Ordering::Relaxed);
                self.target_frame.store(armed, Ordering::Relaxed);
                tracing::info!(
                    world_first = frame,
                    target_frame = armed,
                    world_offset = offset,
                    max_probes,
                    "VICTORY_HDR_PROBE armed"
                );
            }
            return;
        }
        if frame < target {
            return;
        }
        let probed = self.probed.load(Ordering::Relaxed).min(max_probes);
        // Pipelined titles can finish recording frame N's command lists during
        // frame N-1, leaving 0 newly recorded passes on frame N itself. Keep
        // the probe armed until at least one pass is recorded (bounded by
        // `MAX_WAIT_FRAMES`).
        if probed == 0 && frame < target.saturating_add(MAX_WAIT_FRAMES) {
            tracing::info!(
                frame,
                target_frame = target,
                "VICTORY_HDR_PROBE waiting for newly recorded passes (frame used pre-recorded command lists)"
            );
            return;
        }
        if !self.done.swap(true, Ordering::Relaxed) {
            tracing::info!(
                frame,
                world_first = self.world_start.load(Ordering::Relaxed),
                probed,
                max_probes,
                skipped_no_copy = self.skipped_no_copy.load(Ordering::Relaxed),
                skipped_twin = self.skipped_twin.load(Ordering::Relaxed),
                skipped_dim_or_msaa = self.skipped_dim_or_msaa.load(Ordering::Relaxed),
                skipped_format = self.skipped_format.load(Ordering::Relaxed),
                skipped_copy = self.skipped_copy.load(Ordering::Relaxed),
                "VICTORY_HDR_PROBE complete"
            );
        }
    }

    /// Whether probes should be recorded while building `next_frame`.
    #[inline]
    fn active(&self, next_frame: u64, max_probes: usize) -> bool {
        if self.done.load(Ordering::Relaxed) {
            return false;
        }
        let target = self.target_frame.load(Ordering::Relaxed);
        target != UNARMED && next_frame >= target && self.probed.load(Ordering::Relaxed) < max_probes
    }

    /// Reserves the next pass index (`0..max_probes`), or `None` if capped.
    fn reserve(&self, max_probes: usize) -> Option<usize> {
        let idx = self.probed.fetch_add(1, Ordering::Relaxed);
        (idx < max_probes).then_some(idx)
    }

    fn world_offset(&self, frame: u64) -> u64 {
        let start = self.world_start.load(Ordering::Relaxed);
        if start == UNARMED {
            0
        } else {
            frame.saturating_sub(start)
        }
    }
}

static CLOCK: ProbeClock = ProbeClock::new();

/// Called at the end of each presented frame; zero overhead when unset.
#[cfg(windows)]
pub(crate) fn on_present(frame: u64) {
    let Some(offset) = configured_offset() else {
        return;
    };
    CLOCK.on_present(offset, configured_max_probes(), frame, crate::draw::drawn());
}

/// Fast check: `true` only while recording the single targeted world frame
/// and still under the probe cap.
#[cfg(windows)]
#[inline]
pub(crate) fn active() -> bool {
    enabled() && CLOCK.active(crate::queue::presented() + 1, configured_max_probes())
}

/// Records an inline slice-0 readback of `resource` into `list` and logs one
/// compact HDR stats line once the submission completes on the host GPU.
#[cfg(windows)]
pub(crate) fn probe_texture(
    list: &crate::host::HostList,
    resource: crate::object::This,
    restore_state: u32,
    kind: &'static str,
    shaders: &str,
) {
    if resource.is_null() || !active() {
        return;
    }
    probe_texture_with_clock(
        &CLOCK,
        configured_max_probes(),
        list,
        resource,
        restore_state,
        kind,
        shaders,
        None,
    );
}

#[cfg(windows)]
type ProbeSink = std::sync::Arc<std::sync::Mutex<Vec<(&'static str, HdrStats)>>>;

#[cfg(windows)]
#[allow(clippy::too_many_arguments)]
fn probe_texture_with_clock(
    clock: &ProbeClock,
    max_probes: usize,
    list: &crate::host::HostList,
    resource: crate::object::This,
    restore_state: u32,
    kind: &'static str,
    shaders: &str,
    sink: Option<ProbeSink>,
) {
    use windows::Win32::Graphics::Direct3D12::*;
    use windows::Win32::Graphics::Dxgi::Common::DXGI_SAMPLE_DESC;

    if resource.is_null() {
        return;
    }
    let Some(host) = crate::host::host() else {
        return;
    };
    let Some(copy) = crate::resource::host_copy(resource) else {
        clock.skipped_no_copy.fetch_add(1, Ordering::Relaxed);
        return;
    };
    // Skip depth textures with writable twins so twin state routing stays untouched.
    if copy.twin.get().and_then(Option::as_ref).is_some() {
        clock.skipped_twin.fetch_add(1, Ordering::Relaxed);
        return;
    }
    let texture = &copy.texture;
    // SAFETY: `texture` is a live D3D12 resource held by `copy`.
    let desc = unsafe { texture.GetDesc() };
    if desc.Dimension == D3D12_RESOURCE_DIMENSION_BUFFER || desc.SampleDesc.Count > 1 {
        clock.skipped_dim_or_msaa.fetch_add(1, Ordering::Relaxed);
        return;
    }
    let Some(info) = format_info(desc.Format.0 as u32) else {
        clock.skipped_format.fetch_add(1, Ordering::Relaxed);
        return;
    };
    let Some(pass_idx) = clock.reserve(max_probes) else {
        return;
    };

    // For 3D textures, read back slice 0 only to keep readback memory bounded.
    let is_3d = desc.Dimension == D3D12_RESOURCE_DIMENSION_TEXTURE3D;
    let slice_desc = if is_3d {
        D3D12_RESOURCE_DESC {
            DepthOrArraySize: 1,
            ..desc
        }
    } else {
        desc
    };
    let mut footprint = D3D12_PLACED_SUBRESOURCE_FOOTPRINT::default();
    let mut total = 0u64;
    // SAFETY: `slice_desc` describes subresource 0 (or its first depth slice).
    unsafe {
        host.device.GetCopyableFootprints(
            &slice_desc,
            0,
            1,
            0,
            Some(&mut footprint),
            None,
            None,
            Some(&mut total),
        );
    }
    if total == 0 {
        clock.skipped_copy.fetch_add(1, Ordering::Relaxed);
        return;
    }

    let heap = D3D12_HEAP_PROPERTIES {
        Type: D3D12_HEAP_TYPE_READBACK,
        ..Default::default()
    };
    let buf_desc = D3D12_RESOURCE_DESC {
        Dimension: D3D12_RESOURCE_DIMENSION_BUFFER,
        Width: total,
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
    let mut buffer: Option<ID3D12Resource> = None;
    // SAFETY: plain readback buffer creation on the live host device.
    let created = unsafe {
        host.device.CreateCommittedResource(
            &heap,
            D3D12_HEAP_FLAG_NONE,
            &buf_desc,
            D3D12_RESOURCE_STATE_COPY_DEST,
            None,
            &mut buffer,
        )
    };
    let (Ok(()), Some(buffer)) = (created, buffer) else {
        clock.skipped_copy.fetch_add(1, Ordering::Relaxed);
        return;
    };

    const COPY_SOURCE: u32 = 0x800;
    const UNORDERED: u32 = 0x8;
    if restore_state == UNORDERED {
        // SAFETY: `list.list` is open; flush prior UAV writes before copy.
        unsafe { list.list.ResourceBarrier(&[crate::host::uav_barrier()]) };
    }
    crate::list::ensure_state(list, resource, COPY_SOURCE);
    let src_box = is_3d.then_some(D3D12_BOX {
        left: 0,
        top: 0,
        front: 0,
        right: desc.Width as u32,
        bottom: desc.Height,
        back: 1,
    });
    // SAFETY: `list.list` is open; `texture` is in `COPY_SOURCE` and `buffer` is in `COPY_DEST`.
    unsafe {
        let to = D3D12_TEXTURE_COPY_LOCATION {
            pResource: std::mem::transmute_copy(&buffer),
            Type: D3D12_TEXTURE_COPY_TYPE_PLACED_FOOTPRINT,
            Anonymous: D3D12_TEXTURE_COPY_LOCATION_0 {
                PlacedFootprint: footprint,
            },
        };
        let from = D3D12_TEXTURE_COPY_LOCATION {
            pResource: std::mem::transmute_copy(texture),
            Type: D3D12_TEXTURE_COPY_TYPE_SUBRESOURCE_INDEX,
            Anonymous: D3D12_TEXTURE_COPY_LOCATION_0 {
                SubresourceIndex: 0,
            },
        };
        list.list
            .CopyTextureRegion(&to, 0, 0, 0, &from, src_box.as_ref().map(std::ptr::from_ref));
    }
    crate::list::ensure_state(list, resource, restore_state);
    list.keep(&buffer);

    struct Readback(ID3D12Resource);
    // SAFETY: D3D12 COM objects are free-threaded.
    unsafe impl Send for Readback {}
    // SAFETY: as above.
    unsafe impl Sync for Readback {}

    let readback = Readback(buffer);
    let shaders = shaders.to_owned();
    let format = desc.Format.0 as u32;
    let width = footprint.Footprint.Width as usize;
    let height = footprint.Footprint.Height as usize;
    let depth = desc.DepthOrArraySize;
    let pitch = footprint.Footprint.RowPitch as usize;
    let res_addr = resource as usize;
    let frame = crate::queue::presented() + 1;
    let world_offset = clock.world_offset(frame);

    list.defer_after(move || {
        let mut data = std::ptr::null_mut();
        // SAFETY: submission finished on GPU; readback buffer holds `pitch * height` bytes.
        unsafe {
            if readback.0.Map(0, None, Some(&mut data)).is_err() {
                return;
            }
            let len = pitch.saturating_mul(height);
            let rows = std::slice::from_raw_parts(data.cast::<u8>(), len);
            if let Some(s) = compute_stats(format, width, height, pitch, rows) {
                if let Some(sink) = &sink {
                    sink.lock().unwrap_or_else(|e| e.into_inner()).push((kind, s));
                }
                tracing::info!(
                    target: "victory_gpu::hdr_probe",
                    pass_idx,
                    kind,
                    frame,
                    world_offset,
                    res = format_args!("{res_addr:#x}"),
                    format = info.name,
                    format_id = format,
                    width,
                    height,
                    depth,
                    shaders = shaders.as_str(),
                    min = format_args!("{:.4}", s.min),
                    p50 = format_args!("{:.4}", s.p50),
                    p95 = format_args!("{:.4}", s.p95),
                    max = format_args!("{:.4}", s.max),
                    nan_count = s.nan_count,
                    inf_count = s.inf_count,
                    neg_count = s.neg_count,
                    gt100_count = s.gt100_count,
                    center = format_args!(
                        "[{:.4},{:.4},{:.4},{:.4}]",
                        s.center[0], s.center[1], s.center[2], s.center[3]
                    ),
                    "hdr pass probe"
                );
            }
            readback.0.Unmap(0, None);
        }
    });
}

#[cfg(test)]
mod tests {
    use super::*;
    #[cfg(windows)]
    use crate::backend::HostGpu;

    fn approx(a: f32, b: f32, eps: f32) {
        assert!((a - b).abs() <= eps, "expected {a} ~= {b} (eps {eps})");
    }

    #[test]
    fn decode_f16_handles_normal_subnormal_negative_and_specials() {
        assert_eq!(decode_f16(0x0000).to_bits(), 0.0f32.to_bits());
        assert_eq!(decode_f16(0x8000).to_bits(), (-0.0f32).to_bits());
        assert_eq!(decode_f16(0x3c00), 1.0);
        assert_eq!(decode_f16(0xbe00), -1.5);
        assert_eq!(decode_f16(0x0001), 2f32.powi(-24));
        assert_eq!(decode_f16(0x7bff), 65504.0);
        assert_eq!(decode_f16(0x7c00), f32::INFINITY);
        assert_eq!(decode_f16(0xfc00), f32::NEG_INFINITY);
        assert!(decode_f16(0x7c01).is_nan());
        assert!(decode_f16(0xfe00).is_nan());
    }

    #[test]
    fn decode_ufloat_handles_11bit_and_10bit_channels() {
        for mantissa in [6u32, 5u32] {
            let exp_inf = 31u32 << mantissa;
            assert_eq!(decode_ufloat(0, mantissa).to_bits(), 0.0f32.to_bits());
            assert_eq!(decode_ufloat(1, mantissa), 2f32.powi(-14 - mantissa as i32));
            assert_eq!(decode_ufloat(15 << mantissa, mantissa), 1.0);
            assert_eq!(decode_ufloat(exp_inf, mantissa), f32::INFINITY);
            assert!(decode_ufloat(exp_inf | 1, mantissa).is_nan());
        }
        // R11G11B10_FLOAT packed pixel: R=2.0, G=0.5, B=4.0.
        let r11 = 16u32 << 6;
        let g11 = 14u32 << 6;
        let b10 = 17u32 << 5;
        let packed = (r11 | (g11 << 11) | (b10 << 22)).to_le_bytes();
        assert_eq!(decode_pixel(26, &packed), Some([2.0, 0.5, 4.0, 1.0]));
    }

    #[test]
    fn decode_pixel_supports_all_requested_formats() {
        // R32G32B32A32_FLOAT
        let f32x4: Vec<u8> = [1.5f32, -2.0, 150.0, 0.25]
            .into_iter()
            .flat_map(f32::to_le_bytes)
            .collect();
        assert_eq!(decode_pixel(2, &f32x4), Some([1.5, -2.0, 150.0, 0.25]));

        // R16G16B16A16_FLOAT
        let f16x4: Vec<u8> = [0x3c00u16, 0x4000, 0x4200, 0x3800]
            .into_iter()
            .flat_map(u16::to_le_bytes)
            .collect();
        assert_eq!(decode_pixel(10, &f16x4), Some([1.0, 2.0, 3.0, 0.5]));

        // R16G16_FLOAT
        let f16x2: Vec<u8> = [0x3c00u16, 0xbc00]
            .into_iter()
            .flat_map(u16::to_le_bytes)
            .collect();
        assert_eq!(decode_pixel(34, &f16x2), Some([1.0, -1.0, 0.0, 1.0]));

        // R32_FLOAT & R16_FLOAT
        assert_eq!(decode_pixel(41, &42.5f32.to_le_bytes()), Some([42.5, 0.0, 0.0, 1.0]));
        assert_eq!(decode_pixel(54, &0x4000u16.to_le_bytes()), Some([2.0, 0.0, 0.0, 1.0]));

        // R8G8B8A8_UNORM / SRGB & B8G8R8A8_UNORM
        let rgba8 = decode_pixel(28, &[255, 0, 128, 255]).unwrap();
        approx(rgba8[0], 1.0, 1e-6);
        approx(rgba8[1], 0.0, 1e-6);
        approx(rgba8[2], 128.0 / 255.0, 1e-6);
        approx(rgba8[3], 1.0, 1e-6);
        assert_eq!(decode_pixel(29, &[255, 0, 128, 255]), Some(rgba8));

        let bgra8 = decode_pixel(87, &[0, 128, 255, 255]).unwrap();
        approx(bgra8[0], 1.0, 1e-6);
        approx(bgra8[1], 128.0 / 255.0, 1e-6);
        approx(bgra8[2], 0.0, 1e-6);
        approx(bgra8[3], 1.0, 1e-6);

        // R10G10B10A2_UNORM
        let packed10 = (1023u32 | (512 << 10) | (3 << 30)).to_le_bytes();
        let rgb10 = decode_pixel(24, &packed10).unwrap();
        approx(rgb10[0], 1.0, 1e-6);
        approx(rgb10[1], 512.0 / 1023.0, 1e-6);
        approx(rgb10[2], 0.0, 1e-6);
        approx(rgb10[3], 1.0, 1e-6);

        // Unsupported depth/BC format returns None.
        assert!(format_info(45).is_none());
        assert!(format_info(98).is_none());
    }

    #[test]
    fn compute_stats_tracks_nan_inf_neg_gt100_percentiles_and_center() {
        // 3x2 R32G32B32A32_FLOAT with padded pitch (64 bytes/row vs 48 bytes payload).
        let pixels: [[f32; 4]; 6] = [
            [-5.0, 0.0, 1.0, 1.0],
            [2.0, 4.0, 6.0, 1.0],
            [10.0, 20.0, 50.0, 1.0],
            [150.0, 250.0, f32::NAN, 1.0],
            // Center pixel (x=1, y=1):
            [3.0, 5.0, 7.0, 0.5],
            [f32::INFINITY, f32::NEG_INFINITY, 101.0, 1.0],
        ];
        let pitch = 64;
        let mut rows = vec![0xAAu8; pitch * 2];
        for y in 0..2 {
            for x in 0..3 {
                let p = pixels[y * 3 + x];
                let bytes: Vec<u8> = p.into_iter().flat_map(f32::to_le_bytes).collect();
                rows[y * pitch + x * 16..][..16].copy_from_slice(&bytes);
            }
        }

        let stats = compute_stats(2, 3, 2, pitch, &rows).expect("stats");
        assert_eq!(stats.nan_count, 1);
        assert_eq!(stats.inf_count, 2);
        assert_eq!(stats.neg_count, 1);
        assert_eq!(stats.gt100_count, 3); // 150, 250, 101
        assert_eq!(stats.min, -5.0);
        assert_eq!(stats.max, 250.0);
        assert_eq!(stats.center, [3.0, 5.0, 7.0, 0.5]);
        // Finite RGB values (15 total), sorted:
        // [-5, 0, 1, 2, 3, 4, 5, 6, 7, 10, 20, 50, 101, 150, 250]
        // k(0.50) = round(14 * 0.50) = 7 -> 6.0
        // k(0.95) = round(14 * 0.95) = 13 -> 150.0
        assert_eq!(stats.p50, 6.0);
        assert_eq!(stats.p95, 150.0);
    }

    #[test]
    fn compute_stats_handles_all_nan_and_r16_float_targets() {
        // 2x1 R16_FLOAT containing [NaN, +Inf].
        let bytes: Vec<u8> = [0x7c01u16, 0x7c00]
            .into_iter()
            .flat_map(u16::to_le_bytes)
            .collect();
        let stats = compute_stats(54, 2, 1, 4, &bytes).expect("stats");
        assert_eq!(stats.nan_count, 1);
        assert_eq!(stats.inf_count, 1);
        assert_eq!(stats.neg_count, 0);
        assert_eq!(stats.gt100_count, 0);
        assert_eq!((stats.min, stats.p50, stats.p95, stats.max), (0.0, 0.0, 0.0, 0.0));
        assert!(stats.center[0].is_infinite());
    }

    #[test]
    fn probe_clock_arms_on_world_frame_runs_once_and_caps_probes() {
        assert_eq!(parse_offset(""), None);
        assert_eq!(parse_offset("abc"), None);
        assert_eq!(parse_offset(" 24 "), Some(24));
        assert_eq!(parse_offset("0"), Some(0));

        let clock = ProbeClock::new();
        // Loading frames (< 500 draws): never armed.
        clock.on_present(24, MAX_PROBES, 100, 120);
        assert!(!clock.active(101, MAX_PROBES));

        // First world frame (frame 3406, +800 draws) arms target = 3430.
        clock.on_present(24, MAX_PROBES, 3406, 920);
        assert!(!clock.active(3407, MAX_PROBES));
        assert!(!clock.active(3429, MAX_PROBES));
        assert!(clock.active(3430, MAX_PROBES));
        assert_eq!(clock.world_offset(3430), 24);

        // Pipelined frame 3430 executed pre-recorded lists (0 newly recorded
        // probes): clock must remain armed for frame 3431 instead of finishing
        // with `probed=0`.
        clock.on_present(24, MAX_PROBES, 3430, 1500);
        assert!(clock.active(3431, MAX_PROBES));

        // Cap enforces MAX_PROBES.
        for i in 0..MAX_PROBES {
            assert_eq!(clock.reserve(MAX_PROBES), Some(i));
        }
        assert_eq!(clock.reserve(MAX_PROBES), None);
        assert!(!clock.active(3431, MAX_PROBES));

        // Presenting frame 3431 with probes recorded marks done permanently.
        clock.on_present(24, MAX_PROBES, 3431, 50_000);
        assert!(!clock.active(3431, MAX_PROBES));
        assert!(!clock.active(3432, MAX_PROBES));

        // Safety bound: even if a scene never records any probes, clock disarms
        // after `MAX_WAIT_FRAMES` frames past target.
        let idle = ProbeClock::new();
        idle.on_present(2, 10, 100, 600);
        for f in 102..102 + MAX_WAIT_FRAMES {
            idle.on_present(2, 10, f, 600);
            assert!(idle.active(f + 1, 10), "still waiting after frame {f}");
        }
        idle.on_present(2, 10, 102 + MAX_WAIT_FRAMES, 600);
        assert!(!idle.active(103 + MAX_WAIT_FRAMES, 10));
    }

    /// End-to-end live GPU test (D3DMetal under Wine): executes a real graphics
    /// pass (`OMSetRenderTargets` + `DrawInstanced` into an `R16G16B16A16_FLOAT`
    /// RT) AND a real compute pass (`Dispatch` writing an `R16G16B16A16_FLOAT`
    /// UAV with `NaN`, `+Inf`, and `>100` radiance), verifying both probes fire
    /// (`probed >= 2`) and report exact HDR stats.
    #[cfg(windows)]
    #[test]
    fn d3dmetal_probes_graphics_rtv_and_compute_uav_hdr_passes() {
        use std::sync::{Arc, Mutex};
        use windows::Win32::Foundation::RECT;
        use windows::Win32::Graphics::Direct3D::Fxc::D3DCompile;
        use windows::Win32::Graphics::Direct3D::*;
        use windows::Win32::Graphics::Direct3D12::*;
        use windows::Win32::Graphics::Dxgi::Common::*;
        use windows::core::s;

        let Some(host) = crate::host::host() else {
            return; // no host GPU here
        };

        const GFX_HLSL: &str = r#"
float4 vs(uint id : SV_VertexID) : SV_Position {
    float2 uv = float2((id << 1) & 2, id & 2);
    return float4(uv * float2(2, -2) + float2(-1, 1), 0, 1);
}
float4 ps() : SV_Target {
    return float4(2.0, 200.0, -1.5, 0.5);
}
"#;
        const CS_HLSL: &str = r#"
RWTexture2D<float4> out_tex : register(u0);
[numthreads(4, 4, 1)]
void cs(uint3 tid : SV_DispatchThreadID) {
    if (tid.x == 0 && tid.y == 0) {
        out_tex[tid.xy] = float4(asfloat(0x7fc00000u), asfloat(0x7f800000u), 150.0, 1.0);
    } else {
        out_tex[tid.xy] = float4(1.0, 4.0, 12.0, 0.25);
    }
}
"#;
        let compile = |src: &str, entry: windows::core::PCSTR, target: windows::core::PCSTR| {
            let mut blob: Option<ID3DBlob> = None;
            let mut errs: Option<ID3DBlob> = None;
            // SAFETY: valid source slice and NUL-terminated entry/target strings.
            let hr = unsafe {
                D3DCompile(
                    src.as_ptr().cast(),
                    src.len(),
                    s!("hdr_probe_test"),
                    None,
                    None,
                    entry,
                    target,
                    0,
                    0,
                    &mut blob,
                    Some(&mut errs),
                )
            };
            assert!(hr.is_ok(), "D3DCompile failed: {hr:?}");
            blob.unwrap()
        };

        let vs = compile(GFX_HLSL, s!("vs"), s!("vs_5_1"));
        let ps = compile(GFX_HLSL, s!("ps"), s!("ps_5_1"));
        let cs = compile(CS_HLSL, s!("cs"), s!("cs_5_1"));

        // SAFETY: reading bytes from live compiled ID3DBlobs.
        let bytecode = |b: &ID3DBlob| unsafe {
            D3D12_SHADER_BYTECODE {
                pShaderBytecode: b.GetBufferPointer(),
                BytecodeLength: b.GetBufferSize(),
            }
        };

        // Graphics root signature (empty) and PSO targeting R16G16B16A16_FLOAT.
        let gfx_root = crate::compute::root_signature(&host.device, &[]).unwrap();
        let mut gfx_desc = D3D12_GRAPHICS_PIPELINE_STATE_DESC {
            // SAFETY: borrowed for CreateGraphicsPipelineState.
            pRootSignature: unsafe { std::mem::transmute_copy(&gfx_root) },
            VS: bytecode(&vs),
            PS: bytecode(&ps),
            SampleMask: u32::MAX,
            PrimitiveTopologyType: D3D12_PRIMITIVE_TOPOLOGY_TYPE_TRIANGLE,
            NumRenderTargets: 1,
            SampleDesc: DXGI_SAMPLE_DESC {
                Count: 1,
                Quality: 0,
            },
            ..Default::default()
        };
        gfx_desc.RTVFormats[0] = DXGI_FORMAT_R16G16B16A16_FLOAT;
        gfx_desc.BlendState.RenderTarget[0].RenderTargetWriteMask = 0xf;
        gfx_desc.RasterizerState.FillMode = D3D12_FILL_MODE_SOLID;
        gfx_desc.RasterizerState.CullMode = D3D12_CULL_MODE_NONE;
        // SAFETY: valid graphics PSO description on live host device.
        let gfx_pso: ID3D12PipelineState =
            unsafe { host.device.CreateGraphicsPipelineState(&gfx_desc) }.unwrap();

        // Compute root signature (serialized via `crate::compute`) and PSO.
        let cs_root = crate::compute::root_signature(
            &host.device,
            &[crate::compute::Param::Table(&[crate::compute::uav(0, 1)])],
        )
        .unwrap();
        // SAFETY: valid compute PSO description with live root signature and CS blob.
        let cs_pso: ID3D12PipelineState = unsafe {
            let desc = D3D12_COMPUTE_PIPELINE_STATE_DESC {
                pRootSignature: std::mem::transmute_copy(&cs_root),
                CS: bytecode(&cs),
                ..Default::default()
            };
            host.device.CreateComputePipelineState(&desc).unwrap()
        };

        // Real title textures (R16G16B16A16_FLOAT = 10; ALLOW_RENDER_TARGET = 1, ALLOW_UAV = 4).
        let rt_res = crate::resource::test_texture_2d(4, 4, 10, 1);
        let uav_res = crate::resource::test_texture_2d(4, 4, 10, 4);
        let rt_host = crate::resource::host_resource(rt_res).expect("rt host texture");
        let uav_host = crate::resource::host_resource(uav_res).expect("uav host texture");

        // Descriptor heaps for the RTV and the compute UAV.
        // SAFETY: plain descriptor heap creation on live host device.
        let (rtv_heap, uav_heap): (ID3D12DescriptorHeap, ID3D12DescriptorHeap) = unsafe {
            let rtv: ID3D12DescriptorHeap = host
                .device
                .CreateDescriptorHeap(&D3D12_DESCRIPTOR_HEAP_DESC {
                    Type: D3D12_DESCRIPTOR_HEAP_TYPE_RTV,
                    NumDescriptors: 1,
                    ..Default::default()
                })
                .unwrap();
            let uav: ID3D12DescriptorHeap = host
                .device
                .CreateDescriptorHeap(&D3D12_DESCRIPTOR_HEAP_DESC {
                    Type: D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV,
                    NumDescriptors: 1,
                    Flags: D3D12_DESCRIPTOR_HEAP_FLAG_SHADER_VISIBLE,
                    ..Default::default()
                })
                .unwrap();
            host.device
                .CreateRenderTargetView(&rt_host, None, rtv.GetCPUDescriptorHandleForHeapStart());
            let uav_desc = D3D12_UNORDERED_ACCESS_VIEW_DESC {
                Format: DXGI_FORMAT_R16G16B16A16_FLOAT,
                ViewDimension: D3D12_UAV_DIMENSION_TEXTURE2D,
                ..Default::default()
            };
            host.device.CreateUnorderedAccessView(
                &uav_host,
                None,
                Some(&uav_desc),
                uav.GetCPUDescriptorHandleForHeapStart(),
            );
            (rtv, uav)
        };

        let clock = ProbeClock::new();
        let results: Arc<Mutex<Vec<(&'static str, HdrStats)>>> = Arc::new(Mutex::new(Vec::new()));
        let list = host.new_list().unwrap();

        const RENDER_TARGET: u32 = 0x4;
        const UNORDERED: u32 = 0x8;
        crate::list::ensure_state(&list, rt_res, RENDER_TARGET);
        // SAFETY: open command list, live PSO/root/RTV heap.
        unsafe {
            let rtv = rtv_heap.GetCPUDescriptorHandleForHeapStart();
            let vp = D3D12_VIEWPORT {
                TopLeftX: 0.0,
                TopLeftY: 0.0,
                Width: 4.0,
                Height: 4.0,
                MinDepth: 0.0,
                MaxDepth: 1.0,
            };
            let sc = RECT {
                left: 0,
                top: 0,
                right: 4,
                bottom: 4,
            };
            list.list.OMSetRenderTargets(1, Some(&rtv), false, None);
            list.list.RSSetViewports(&[vp]);
            list.list.RSSetScissorRects(&[sc]);
            list.list.IASetPrimitiveTopology(D3D_PRIMITIVE_TOPOLOGY_TRIANGLELIST);
            list.list.SetGraphicsRootSignature(&gfx_root);
            list.list.SetPipelineState(&gfx_pso);
            list.list.DrawInstanced(3, 1, 0, 0);
        }
        probe_texture_with_clock(
            &clock,
            MAX_PROBES,
            &list,
            rt_res,
            RENDER_TARGET,
            "rtv",
            "[vs synthetic, ps synthetic]",
            Some(Arc::clone(&results)),
        );

        crate::list::ensure_state(&list, uav_res, UNORDERED);
        // SAFETY: open command list, live compute PSO/root/UAV descriptor heap.
        unsafe {
            list.list.SetDescriptorHeaps(&[Some(uav_heap.clone())]);
            list.list.SetComputeRootSignature(&cs_root);
            list.list
                .SetComputeRootDescriptorTable(0, uav_heap.GetGPUDescriptorHandleForHeapStart());
            list.list.SetPipelineState(&cs_pso);
            list.list.Dispatch(1, 1, 1);
        }
        probe_texture_with_clock(
            &clock,
            MAX_PROBES,
            &list,
            uav_res,
            UNORDERED,
            "uav",
            "[cs synthetic]",
            Some(Arc::clone(&results)),
        );

        list.close();
        host.execute(&[&list]);

        assert!(
            clock.probed.load(Ordering::Relaxed) >= 2,
            "expected >= 2 probes (rtv + uav)"
        );
        let captured = results.lock().unwrap_or_else(|e| e.into_inner());
        assert_eq!(captured.len(), 2, "both graphics RTV and compute UAV passes must report stats");

        let (rt_kind, rt_stats) = captured[0];
        assert_eq!(rt_kind, "rtv");
        assert_eq!(rt_stats.nan_count, 0);
        assert_eq!(rt_stats.inf_count, 0);
        assert_eq!(rt_stats.neg_count, 16);
        assert_eq!(rt_stats.gt100_count, 16);
        approx(rt_stats.min, -1.5, 1e-3);
        approx(rt_stats.p50, 2.0, 1e-3);
        approx(rt_stats.p95, 200.0, 1e-2);
        approx(rt_stats.max, 200.0, 1e-2);
        approx(rt_stats.center[0], 2.0, 1e-3);
        approx(rt_stats.center[1], 200.0, 1e-2);
        approx(rt_stats.center[2], -1.5, 1e-3);
        approx(rt_stats.center[3], 0.5, 1e-3);

        let (uav_kind, uav_stats) = captured[1];
        assert_eq!(uav_kind, "uav");
        assert_eq!(uav_stats.nan_count, 1);
        assert_eq!(uav_stats.inf_count, 1);
        assert_eq!(uav_stats.neg_count, 0);
        assert_eq!(uav_stats.gt100_count, 1);
        approx(uav_stats.min, 1.0, 1e-3);
        approx(uav_stats.p50, 4.0, 1e-3);
        approx(uav_stats.p95, 12.0, 1e-2);
        approx(uav_stats.max, 150.0, 1e-2);
        approx(uav_stats.center[0], 1.0, 1e-3);
        approx(uav_stats.center[1], 4.0, 1e-3);
        approx(uav_stats.center[2], 12.0, 1e-2);
        approx(uav_stats.center[3], 0.25, 1e-3);

        let release = |o: crate::object::This| {
            crate::object::slot::<extern "system" fn(crate::object::This) -> u32>(o, 0x10)(o)
        };
        assert_eq!(release(rt_res), 0);
        assert_eq!(release(uav_res), 0);
    }
}
