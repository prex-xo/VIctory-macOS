use super::{PIPELINE_IIDS, Shader, Translated, hand_out, host_pipeline};
use crate::backend::HostGpu;
use crate::object::{S_OK, This, guid, slot};
use std::sync::{Arc, OnceLock};
use victory_shader::native::{ComputeIo, StageIo};
use victory_shader::{
    Container, HwStage, NativeProgram, ResourceType, RootParam, RootType, UsedResource, emit, io,
    ir, resource, structure,
};
use windows::Win32::Graphics::Direct3D12::*;
use windows::Win32::Graphics::Dxgi::Common::{
    DXGI_FORMAT_B4G4R4A4_UNORM, DXGI_FORMAT_B5G5R5A1_UNORM, DXGI_FORMAT_R8G8B8A8_UNORM,
    DXGI_FORMAT_R32G32B32A32_FLOAT,
};
use windows::core::Interface;

/// Synthetic compute shader storing four root constants at texel (0, 0).
fn store_shader() -> Arc<Shader> {
    let words: [u32; 12] = [
        0xBE85_0380, // s_mov_b32 s5, 0: high word of the table address
        0xF40C_0202,
        0xFA00_0000, // s_load_dwordx8 s[8:15], s[4:5], 0
        0x7E02_0200,
        0x7E04_0201,
        0x7E06_0202,
        0x7E08_0203, // v_mov_b32 v[1:4], s[0:3]
        0x7E0A_0280,
        0x7E0C_0280, // v_mov_b32 v5/v6, 0: texel coordinates
        0xF020_1F08,
        0x0002_0105, // image_store v[1:4], v[5:6], s[8:15], dim:2D, dmask:0xf
        0xBF81_0000, // s_endpgm
    ];
    let native = NativeProgram {
        stage: HwStage::Cs,
        wave64: false,
        vgprs: 8,
        layout: vec![
            RootParam {
                root: RootType::Constants,
                table_offset_dwords: 0,
                sgprs: 4,
                base_register: 0,
                user_data: 0,
                slot: 0,
            },
            RootParam {
                root: RootType::Table,
                table_offset_dwords: 0,
                sgprs: 1,
                base_register: 0,
                user_data: 4,
                slot: 1,
            },
        ],
        used: vec![UsedResource {
            kind: ResourceType::Uav,
            root: RootType::Table,
            table_offset_dwords: 0,
            lower_bound: 0,
            stride: 0,
            dwords: 8,
            user_data: 4,
            slot: 1,
        }],
        io: StageIo::Cs(ComputeIo {
            rsrc2: 5 << 1,
            threads: [1, 1, 1],
        }),
        code: words.into_iter().flat_map(u32::to_le_bytes).collect(),
    };
    // Compute has no input/output signatures. Supply just the DXBC header.
    let mut container = vec![0; 32];
    container[..4].copy_from_slice(b"DXBC");
    container[20..24].copy_from_slice(&1u32.to_le_bytes());
    container[24..28].copy_from_slice(&32u32.to_le_bytes());
    let c = Container::parse(&container).unwrap();
    let p = ir::build(&native.code, native.wave64, native.vgprs).unwrap();
    let st = structure::structure(&p).unwrap();
    let res = resource::track(&p, &native.layout, native.stage, Some(&st)).unwrap();
    let io = io::raise(&p, &native).unwrap();
    let host = emit::emit(&c, &native, &p, &st, &io, &res).unwrap();
    Arc::new(Shader {
        stage: "compute",
        hash: blake3::hash(&native.code).to_hex()[..16].to_owned(),
        full_hash: *blake3::hash(&native.code).as_bytes(),
        size: container.len(),
        container: std::sync::Mutex::new(container),
        spilled: OnceLock::new(),
        translated: OnceLock::from(Some(Translated {
            host,
            used: native.used,
            used_written: Vec::new(),
            write_unknown: true,
            threads: Some([1, 1, 1]),
        })),
    })
}

struct Guest(This);

impl Drop for Guest {
    fn drop(&mut self) {
        slot::<extern "system" fn(This) -> u32>(self.0, 0x10)(self.0);
    }
}

// The cached placement must stay live for the process. Align within the array.
const TEXTURE_ALIGNMENT: usize = 64 << 10;
static BACKING: [u8; 2 * TEXTURE_ALIGNMENT] = [0; 2 * TEXTURE_ALIGNMENT];
static PACKED_BACKING: [[u8; 2 * TEXTURE_ALIGNMENT]; 2] = [[0; 2 * TEXTURE_ALIGNMENT]; 2];

#[test]
fn placed_uav_dispatch_writes_its_texture() {
    let value = [1.0f32, 0.5, -0.25, 2.0];
    let expected = value.map(f32::to_le_bytes).concat();
    check_placed_store(
        &BACKING,
        DXGI_FORMAT_R32G32B32A32_FLOAT.0 as u32,
        16,
        value,
        &expected,
    );
}

#[test]
fn widened_packed_uavs_store_through_the_guest_format() {
    let formats = [DXGI_FORMAT_B5G5R5A1_UNORM, DXGI_FORMAT_B4G4R4A4_UNORM];
    for (format, backing) in formats.into_iter().zip(PACKED_BACKING.iter()) {
        let format = format.0 as u32;
        if crate::resource::target_format(format) != DXGI_FORMAT_R8G8B8A8_UNORM.0 as u32 {
            continue;
        }
        eprintln!("testing a widened UAV for format {format}");
        check_placed_store(
            backing,
            format,
            128,
            [1.0, 0.0, 1.0, 1.0],
            &[255, 0, 255, 255],
        );
    }
}

fn check_placed_store(
    backing: &'static [u8],
    format: u32,
    width: u32,
    value: [f32; 4],
    expected: &[u8],
) {
    let Some(host) = crate::host::host() else {
        return;
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
        eprintln!("placed UAV dispatch needs SM 6.6");
        return;
    }

    let mut pipeline = std::ptr::null();
    assert_eq!(
        hand_out(vec![store_shader()], None, &[], &PIPELINE_IIDS[0], &mut pipeline),
        S_OK
    );
    let pipeline = Guest(pipeline);
    assert!(
        host_pipeline(pipeline.0).is_some(),
        "synthetic host pipeline"
    );

    #[repr(C)]
    struct HeapDescWithPlacement(D3D12_DESCRIPTOR_HEAP_DESC, [u64; 2]);
    let desc = HeapDescWithPlacement(
        D3D12_DESCRIPTOR_HEAP_DESC {
            Type: D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV,
            NumDescriptors: 2,
            Flags: D3D12_DESCRIPTOR_HEAP_FLAG_SHADER_VISIBLE,
            NodeMask: 0,
        },
        [0; 2],
    );
    let iid = ID3D12DescriptorHeap::IID;
    let iid = guid(iid.data1, iid.data2, iid.data3, iid.data4);
    let mut views = std::ptr::null();
    assert_eq!(
        crate::descriptor::create_heap(
            std::ptr::null(),
            std::ptr::from_ref(&desc).cast(),
            &iid,
            &mut views
        ),
        S_OK
    );
    let views = Guest(views);
    let handle = crate::descriptor::heap(views.0).start + 32;

    let address = (backing.as_ptr() as usize).next_multiple_of(TEXTURE_ALIGNMENT) as u64;
    let mut resource = [0; 14];
    resource[0] = D3D12_RESOURCE_DIMENSION_TEXTURE2D.0 as u32;
    resource[4] = width; // width of one guest 256-byte row
    resource[6] = 1; // height
    resource[7] = 1 | (1 << 16); // one slice and one mip
    resource[8] = format;
    resource[9] = 1; // sample count
    resource[11] = 1; // linear layout
    resource[12] = D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS.0 as u32;
    let mut view = [0; 16];
    view[0] = format;
    view[1] = D3D12_UAV_DIMENSION_TEXTURE2D.0 as u32;
    view[10] = address as u32;
    view[11] = (address >> 32) as u32;
    let placed = crate::resource::texture_at(address, &resource).expect("placed texture");
    let texture = crate::resource::host_resource(placed).expect("host texture");
    crate::descriptor::create_placed_uav(std::ptr::null(), &resource, &view, handle);

    let list = host.new_list().unwrap();
    crate::texture::upload(&list, placed, &texture);
    let mut state = crate::draw::State::default();
    state.set_pipeline(pipeline.0);
    state.set_heaps(&[views.0]);
    state.set_constants(true, 0, &value.map(f32::to_bits), 0);
    state.set_table(true, 1, handle);
    let dispatched = crate::draw::dispatched();
    state.dispatch(&list, 1, 1, 1);
    assert!(
        crate::draw::dispatched() > dispatched,
        "dispatch did not record"
    );
    crate::list::ensure_state(&list, placed, D3D12_RESOURCE_STATE_COPY_SOURCE.0 as u32);
    let (readback, footprint) =
        crate::capture::read_back(&host.device, &list.list, &texture).unwrap();
    list.close();
    host.execute(&[&list]);

    // SAFETY: read two host texels after the synchronous submission.
    let texels = unsafe {
        let mut data = std::ptr::null_mut();
        readback.Map(0, None, Some(&mut data)).unwrap();
        let start = data.cast::<u8>().add(footprint.Offset as usize);
        let texels = std::slice::from_raw_parts(start, 2 * expected.len()).to_vec();
        readback.Unmap(0, None);
        assert!(host.device.GetDeviceRemovedReason().is_ok());
        texels
    };
    let (first, next) = texels.split_at(expected.len());
    assert_eq!(first, expected, "placed UAV dispatch readback");
    assert!(
        next.iter().all(|&byte| byte == 0),
        "neighboring texel changed"
    );
}
