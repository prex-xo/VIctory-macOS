//! Draws a presented overlay plane over the picture plane (#15).
//!
//! On Series S Grand Theft Auto VI presents two planes: a 1920x1080
//! overlay and the 3840x2160 picture. The console's display hardware blends
//! the overlay over the picture. Here a compute pass does it into a texture
//! of the picture's size, which is then copied to the swap chain:
//! `picture * (1 - alpha) + overlay * alpha`, the overlay scaled to the
//! picture by its nearest texel. The overlay's alpha is taken as straight
//! (not premultiplied) until a run with the game shows otherwise.
//!
//! The pass is off unless `VICTORY_PRESENT_COMPOSITE=1`: if the overlay
//! turns out opaque where the picture should show, compositing would bring
//! the black window back.

use windows::Win32::Graphics::Direct3D12::*;
use windows::Win32::Graphics::Dxgi::Common::*;
use windows::core::{Result, s};

/// Whether `VICTORY_PRESENT_COMPOSITE=1` asks for the pass.
pub(crate) fn wanted() -> bool {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| std::env::var("VICTORY_PRESENT_COMPOSITE").is_ok_and(|v| v == "1"))
}

const SOURCE: &str = r#"
cbuffer C : register(b0) { uint width; uint height; uint overWidth; uint overHeight; };
Texture2D<float4> picture : register(t0);
Texture2D<float4> overlay : register(t1);
RWTexture2D<float4> result : register(u0);
[numthreads(8, 8, 1)]
void main(uint3 id : SV_DispatchThreadID) {
    if (id.x >= width || id.y >= height) return;
    float4 p = picture.Load(int3(id.xy, 0));
    float4 o = overlay.Load(int3(id.x * overWidth / width, id.y * overHeight / height, 0));
    result[id.xy] = float4(lerp(p.rgb, o.rgb, o.a), 1.0);
}
"#;

/// The pass's pipeline and its shader-visible heap of three views
/// (picture, overlay, result). Presents run one at a time, under the
/// presenter's lock, and each waits for its list, so one heap serves all.
pub(crate) struct Compositor {
    root: ID3D12RootSignature,
    pso: ID3D12PipelineState,
    heap: ID3D12DescriptorHeap,
    step: u32,
}

impl Compositor {
    pub(crate) fn new(device: &ID3D12Device) -> Result<Self> {
        use crate::compute::{Param, srv, uav};
        let ranges = [srv(0, 2), uav(0, 1)];
        let params = [
            Param::Table(&ranges),
            Param::Constants {
                register: 0,
                count: 4,
            },
        ];
        let (root, pso) = crate::compute::pipeline(SOURCE, s!("composite"), 0, &params)?;
        // SAFETY: plain object creation.
        unsafe {
            let heap = device.CreateDescriptorHeap(&D3D12_DESCRIPTOR_HEAP_DESC {
                Type: D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV,
                NumDescriptors: 3,
                Flags: D3D12_DESCRIPTOR_HEAP_FLAG_SHADER_VISIBLE,
                ..Default::default()
            })?;
            let step =
                device.GetDescriptorHandleIncrementSize(D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV);
            Ok(Self {
                root,
                pso,
                heap,
                step,
            })
        }
    }

    /// Records the pass. `picture` and `overlay` must be in a non-pixel
    /// shader resource state and `result` (picture-sized, in `format`) in
    /// the unordered access state; they stay so.
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn record(
        &self,
        device: &ID3D12Device,
        list: &ID3D12GraphicsCommandList,
        picture: (&ID3D12Resource, DXGI_FORMAT),
        overlay: (&ID3D12Resource, DXGI_FORMAT),
        result: &ID3D12Resource,
        format: DXGI_FORMAT,
    ) {
        // SAFETY: live resources and our heap of three views.
        unsafe {
            let at = self.heap.GetCPUDescriptorHandleForHeapStart();
            let slot = |i: u32| D3D12_CPU_DESCRIPTOR_HANDLE {
                ptr: at.ptr + (i * self.step) as usize,
            };
            for (i, (texture, format)) in [picture, overlay].into_iter().enumerate() {
                let view = D3D12_SHADER_RESOURCE_VIEW_DESC {
                    Format: format,
                    ViewDimension: D3D12_SRV_DIMENSION_TEXTURE2D,
                    Shader4ComponentMapping: D3D12_DEFAULT_SHADER_4_COMPONENT_MAPPING,
                    Anonymous: D3D12_SHADER_RESOURCE_VIEW_DESC_0 {
                        Texture2D: D3D12_TEX2D_SRV {
                            MipLevels: 1,
                            ..Default::default()
                        },
                    },
                };
                device.CreateShaderResourceView(texture, Some(&view), slot(i as u32));
            }
            let view = D3D12_UNORDERED_ACCESS_VIEW_DESC {
                Format: format,
                ViewDimension: D3D12_UAV_DIMENSION_TEXTURE2D,
                ..Default::default()
            };
            device.CreateUnorderedAccessView(result, None, Some(&view), slot(2));
            let (p, o) = (picture.0.GetDesc(), overlay.0.GetDesc());
            let constants = [p.Width as u32, p.Height, o.Width as u32, o.Height];
            list.SetDescriptorHeaps(&[Some(self.heap.clone())]);
            list.SetComputeRootSignature(&self.root);
            list.SetPipelineState(&self.pso);
            list.SetComputeRootDescriptorTable(0, self.heap.GetGPUDescriptorHandleForHeapStart());
            list.SetComputeRoot32BitConstants(1, 4, constants.as_ptr().cast(), 0);
            list.Dispatch(constants[0].div_ceil(8), constants[1].div_ceil(8), 1);
        }
    }
}

/// A picture-sized texture the pass writes, in `format`.
pub(crate) fn result_texture(
    device: &ID3D12Device,
    width: u64,
    height: u32,
    format: DXGI_FORMAT,
) -> Result<ID3D12Resource> {
    let mut texture: Option<ID3D12Resource> = None;
    // SAFETY: plain creation.
    unsafe {
        device.CreateCommittedResource(
            &D3D12_HEAP_PROPERTIES {
                Type: D3D12_HEAP_TYPE_DEFAULT,
                ..Default::default()
            },
            D3D12_HEAP_FLAG_NONE,
            &D3D12_RESOURCE_DESC {
                Dimension: D3D12_RESOURCE_DIMENSION_TEXTURE2D,
                Width: width,
                Height: height,
                DepthOrArraySize: 1,
                MipLevels: 1,
                Format: format,
                SampleDesc: DXGI_SAMPLE_DESC {
                    Count: 1,
                    Quality: 0,
                },
                Flags: D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS,
                ..Default::default()
            },
            D3D12_RESOURCE_STATE_UNORDERED_ACCESS,
            None,
            &mut texture,
        )?;
    }
    texture.ok_or_else(windows::core::Error::empty)
}

/// The format the pass writes and the swap chain shows for a picture whose
/// swap format is `format`: typed stores to BGRA are optional in D3D12, so
/// BGRA becomes RGBA (the shader reads and writes colors by name, so the
/// picture keeps its colors).
pub(crate) fn result_format(format: DXGI_FORMAT) -> DXGI_FORMAT {
    if format == DXGI_FORMAT_B8G8R8A8_UNORM {
        DXGI_FORMAT_R8G8B8A8_UNORM
    } else {
        format
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::host::transition;

    /// A buffer of `len` bytes in `heap`, in `state`.
    fn buffer(heap: D3D12_HEAP_TYPE, len: u64, state: D3D12_RESOURCE_STATES) -> ID3D12Resource {
        let device = &crate::host::host().unwrap().device;
        let mut buffer: Option<ID3D12Resource> = None;
        // SAFETY: plain creation.
        unsafe {
            device
                .CreateCommittedResource(
                    &D3D12_HEAP_PROPERTIES {
                        Type: heap,
                        ..Default::default()
                    },
                    D3D12_HEAP_FLAG_NONE,
                    &D3D12_RESOURCE_DESC {
                        Dimension: D3D12_RESOURCE_DIMENSION_BUFFER,
                        Width: len,
                        Height: 1,
                        DepthOrArraySize: 1,
                        MipLevels: 1,
                        SampleDesc: DXGI_SAMPLE_DESC {
                            Count: 1,
                            Quality: 0,
                        },
                        Layout: D3D12_TEXTURE_LAYOUT_ROW_MAJOR,
                        ..Default::default()
                    },
                    state,
                    None,
                    &mut buffer,
                )
                .unwrap();
        }
        buffer.unwrap()
    }

    /// A copy location for `texture`'s first subresource.
    fn whole(texture: &ID3D12Resource) -> D3D12_TEXTURE_COPY_LOCATION {
        D3D12_TEXTURE_COPY_LOCATION {
            // SAFETY: a borrowed pointer the location never releases.
            pResource: unsafe { std::mem::transmute_copy(texture) },
            Type: D3D12_TEXTURE_COPY_TYPE_SUBRESOURCE_INDEX,
            Anonymous: D3D12_TEXTURE_COPY_LOCATION_0 {
                SubresourceIndex: 0,
            },
        }
    }

    /// A copy location for 8-bit RGBA rows at a 256-byte pitch in `buffer`.
    fn rows(buffer: &ID3D12Resource, width: u32, height: u32) -> D3D12_TEXTURE_COPY_LOCATION {
        D3D12_TEXTURE_COPY_LOCATION {
            // SAFETY: a borrowed pointer the location never releases.
            pResource: unsafe { std::mem::transmute_copy(buffer) },
            Type: D3D12_TEXTURE_COPY_TYPE_PLACED_FOOTPRINT,
            Anonymous: D3D12_TEXTURE_COPY_LOCATION_0 {
                PlacedFootprint: D3D12_PLACED_SUBRESOURCE_FOOTPRINT {
                    Offset: 0,
                    Footprint: D3D12_SUBRESOURCE_FOOTPRINT {
                        Format: DXGI_FORMAT_R8G8B8A8_UNORM,
                        Width: width,
                        Height: height,
                        Depth: 1,
                        RowPitch: 256,
                    },
                },
            },
        }
    }

    /// An 8-bit RGBA texture of `width` x `height` holding `texels`, left
    /// in the non-pixel shader resource state; its upload buffer is kept.
    fn texture(
        width: u32,
        height: u32,
        texels: &[[u8; 4]],
        list: &ID3D12GraphicsCommandList,
        keep: &mut Vec<ID3D12Resource>,
    ) -> ID3D12Resource {
        let device = &crate::host::host().unwrap().device;
        let texture =
            result_texture(device, width.into(), height, DXGI_FORMAT_R8G8B8A8_UNORM).unwrap();
        let upload = buffer(
            D3D12_HEAP_TYPE_UPLOAD,
            256 * u64::from(height),
            D3D12_RESOURCE_STATE_GENERIC_READ,
        );
        let mut data: *mut std::ffi::c_void = std::ptr::null_mut();
        // SAFETY: a mapped upload buffer written in bounds; an open list and
        // copy locations naming live resources.
        unsafe {
            upload.Map(0, None, Some(&mut data)).unwrap();
            for y in 0..height {
                for x in 0..width {
                    let at = data.cast::<u8>().add((y * 256 + x * 4) as usize);
                    std::ptr::copy_nonoverlapping(texels[(y * width + x) as usize].as_ptr(), at, 4);
                }
            }
            list.ResourceBarrier(&[transition(
                &texture,
                D3D12_RESOURCE_STATE_UNORDERED_ACCESS,
                D3D12_RESOURCE_STATE_COPY_DEST,
            )]);
            list.CopyTextureRegion(
                &whole(&texture),
                0,
                0,
                0,
                &rows(&upload, width, height),
                None,
            );
            list.ResourceBarrier(&[transition(
                &texture,
                D3D12_RESOURCE_STATE_COPY_DEST,
                D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE,
            )]);
        }
        keep.push(upload);
        texture
    }

    /// A 4x4 red picture under a 2x2 overlay whose top-left texel is opaque
    /// green, top-right half-transparent blue and the rest clear: each
    /// overlay texel covers a 2x2 block of the picture.
    #[test]
    fn the_overlay_is_blended_over_the_picture() {
        let Some(host) = crate::host::host() else {
            return; // no host GPU here
        };
        let device = &host.device;
        let compositor = Compositor::new(device).expect("the pass compiles");
        let run = host.new_list().unwrap();
        let list = &run.list;
        let mut keep = Vec::new();
        let picture = texture(4, 4, &[[255, 0, 0, 255]; 16], list, &mut keep);
        let clear = [9, 9, 9, 0];
        let overlay = texture(
            2,
            2,
            &[[0, 255, 0, 255], [0, 0, 255, 128], clear, clear],
            list,
            &mut keep,
        );
        let format = DXGI_FORMAT_R8G8B8A8_UNORM;
        let result = result_texture(device, 4, 4, format).unwrap();
        compositor.record(
            device,
            list,
            (&picture, format),
            (&overlay, format),
            &result,
            format,
        );
        let back = buffer(
            D3D12_HEAP_TYPE_READBACK,
            256 * 4,
            D3D12_RESOURCE_STATE_COPY_DEST,
        );
        // SAFETY: an open list and live resources.
        unsafe {
            list.ResourceBarrier(&[transition(
                &result,
                D3D12_RESOURCE_STATE_UNORDERED_ACCESS,
                D3D12_RESOURCE_STATE_COPY_SOURCE,
            )]);
            list.CopyTextureRegion(&rows(&back, 4, 4), 0, 0, 0, &whole(&result), None);
        }
        run.close();
        host.execute(&[&run]);
        let mut data: *mut std::ffi::c_void = std::ptr::null_mut();
        // SAFETY: a finished readback buffer.
        unsafe { back.Map(0, None, Some(&mut data)) }.unwrap();
        // SAFETY: inside the mapped buffer: 4 rows at a 256-byte pitch.
        let texel = |x: usize, y: usize| unsafe { *data.cast::<[u8; 4]>().add(y * 64 + x) };
        for (x, y) in [(0, 0), (1, 1)] {
            assert_eq!(
                texel(x, y),
                [0, 255, 0, 255],
                "opaque overlay at ({x}, {y})"
            );
        }
        let half = texel(3, 0);
        assert!(
            half[0].abs_diff(127) <= 1 && half[1] == 0 && half[2].abs_diff(128) <= 1,
            "half blue over red: {half:?}"
        );
        for (x, y) in [(0, 2), (3, 3)] {
            assert_eq!(texel(x, y), [255, 0, 0, 255], "clear overlay at ({x}, {y})");
        }
    }
}
