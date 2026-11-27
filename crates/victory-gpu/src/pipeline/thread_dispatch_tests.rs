//! `DispatchX` whose counts are threads (`draw::State::dispatch_x`): a
//! title compute pipeline made from a console container, dispatched
//! through a list's state, and what it wrote read back.

use super::{PIPELINE_IIDS, hand_out, host_pipeline, shader};
use crate::backend::HostGpu;
use crate::object::{S_OK, This, guid, slot};
use victory_shader::{FourCC, RootType, emit};
use windows::Win32::Graphics::Direct3D12::*;
use windows::Win32::Graphics::Dxgi::Common::DXGI_FORMAT_R32G32B32A32_FLOAT;
use windows::core::Interface;

/// A title object of ours, released as the title releases it.
struct Guest(This);

impl Drop for Guest {
    fn drop(&mut self) {
        slot::<extern "system" fn(This) -> u32>(self.0, 0x10)(self.0);
    }
}

/// RGBA32_FLOAT texels in the row written: two groups of 64 threads.
const TEXELS: usize = 128;
const TEXTURE_ALIGNMENT: usize = 64 << 10;
/// Room for a placed texture per dispatch below (placements stay live for
/// the process).
static BACKING: [u8; 6 * TEXTURE_ALIGNMENT] = [0; 6 * TEXTURE_ALIGNMENT];

/// A console container (an `ACCC` part, the layout `native::fspx_program`
/// reads) of a compute program of groups of 64 whose thread `x` (along the
/// dispatch) stores `[1, 0, 0, 0]` at texel `x` of the 2D UAV at entry 0
/// of the table at user data 4 (root parameter 0).
fn texel_per_thread_container() -> Vec<u8> {
    let words: [u32; 15] = [
        0xBE86_0304, // s_mov_b32 s6, s4: the table's address
        0xBE87_0380, // s_mov_b32 s7, 0
        0xF40C_0203,
        0xFA00_0000, // s_load_dwordx8 s[8:15], s[6:7], 0
        0x7E0E_0205, // v_mov_b32 v7, s5: the group id
        0x340E_0E86, // v_lshlrev_b32 v7, 6, v7
        0x4A0A_0107, // v_add_nc_u32 v5, v7, v0: the thread's x
        0x7E0C_0280, // v_mov_b32 v6, 0
        0x7E02_02F2, // v_mov_b32 v1, 1.0
        0x7E04_0280,
        0x7E06_0280,
        0x7E08_0280, // v_mov_b32 v[2:4], 0
        0xF020_1F08,
        0x0002_0105, // image_store v[1:4], v[5:6], s[8:15], dim:2D, dmask:0xf
        0xBF81_0000, // s_endpgm
    ];
    let code: Vec<u8> = words.into_iter().flat_map(u32::to_le_bytes).collect();
    let text: Vec<String> = victory_shader::gfx10::decode(&code)
        .unwrap()
        .iter()
        .map(|i| i.text(true))
        .collect();
    for want in [
        "s_load_dwordx8 s[8:15], s[6:7]",
        "v_mov_b32 v7, s5",
        "v_lshlrev_b32 v7, 6, v7",
        "v_add_nc_u32 v5, v7, v0",
        "image_store v[1:4], v[5:6], s[8:15]",
    ] {
        assert!(
            text.iter().any(|t| t.contains(want)),
            "{want:?} not in {text:?}"
        );
    }
    let le = |w: &[u32]| w.iter().flat_map(|x| x.to_le_bytes()).collect::<Vec<u8>>();
    // Wave32, 64 x 1 x 1 threads, 16 VGPRs; five user SGPRs, then the group
    // id in x (RSRC2 bit 7).
    let packet = le(&[
        0,
        0,
        0,
        64 | 1 << 16,
        1 | (1 << 4) << 16,
        1,
        5 << 1 | 1 << 7,
    ]);
    // The UAV: a table entry of 8 dwords, 2D, user data 4, root slot 0.
    let mut uav = [0u8; 32];
    uav[0] = 0x41;
    uav[4..6].copy_from_slice(&[3, 3]);
    uav[16] = 4;
    uav[20..23].copy_from_slice(&[8, 4, 0]);
    uav[24..28].copy_from_slice(&u32::MAX.to_le_bytes());
    let mut hash = [0u8; 32];
    hash[0] = 0x43;
    let mut meta = le(&[0x403, 0, words.len() as u32, 0, 0]);
    meta.extend_from_slice(&[0x40, 4, 1, 0]);
    meta.extend_from_slice(&uav);
    meta.extend_from_slice(&hash);
    let len = meta.len() as u32 - 0x14;
    meta[4..8].copy_from_slice(&len.to_le_bytes());
    let blobs: [(u32, &[u8]); 4] = [(0, &packet), (1, &[0; 0x14]), (6, &code), (7, &meta)];
    let mut part = vec![0u8; 0x10 + 16 * blobs.len()];
    for (i, (kind, blob)) in blobs.iter().enumerate() {
        let at = part.len().next_multiple_of(16);
        part.resize(at, 0);
        let entry = le(&[*kind, at as u32, blob.len() as u32, 0]);
        part[0x10 + 16 * i..][..16].copy_from_slice(&entry);
        part.extend_from_slice(blob);
    }
    let header = le(&[u32::from_le_bytes(*b"FSPX"), 1, 4, part.len() as u32]);
    part[..16].copy_from_slice(&header);
    // A DXBC container of that one part: header, its offset, the part.
    let total = 0x20 + 4 + 8 + part.len();
    let mut out = b"DXBC".to_vec();
    out.extend([0; 16]);
    out.extend(le(&[1, total as u32, 1, 0x24]));
    out.extend(FourCC::ACCC.0);
    out.extend((part.len() as u32).to_le_bytes());
    out.extend(part);
    out
}

/// `DispatchX` with the flag that its counts are threads (0x20): 70 threads
/// in groups of 64 run two groups, and threads 70 to 127 of the second
/// write nothing (taken for 70 groups, as before, 4,480 threads ran and
/// every texel was written). Only that dispatch, whose last group is a part
/// of one, takes the form whose lanes past the count leave; the pipeline's
/// own translation reads no count. Without the flag the counts are groups.
#[test]
fn a_dispatch_counted_in_threads_runs_only_those_threads() {
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
        eprintln!("the dispatch needs SM 6.6");
        return;
    }
    let kind = RootType::Other(emit::USER_DISPATCH_THREADS_X);
    let compute = shader("compute", &texel_per_thread_container());
    let t = compute.translated().expect("translates");
    assert_eq!(t.threads, Some([64, 1, 1]));
    assert!(
        !t.host.user_data.iter().any(|u| u.root == kind),
        "its own form reads no count"
    );
    let mut pipeline = std::ptr::null();
    assert_eq!(
        hand_out(vec![compute], None, &[], &PIPELINE_IIDS[0], &mut pipeline),
        S_OK
    );
    let pipeline = Guest(pipeline);
    let p = host_pipeline(pipeline.0).expect("host pipeline");

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

    let base = (BACKING.as_ptr() as usize).next_multiple_of(TEXTURE_ALIGNMENT) as u64;
    let format = DXGI_FORMAT_R32G32B32A32_FLOAT.0 as u32;
    let mut resource = [0; 14];
    resource[0] = D3D12_RESOURCE_DIMENSION_TEXTURE2D.0 as u32;
    resource[4] = TEXELS as u32; // width, one 2 KiB row
    resource[6] = 1; // height
    resource[7] = 1 | (1 << 16); // one slice and one mip
    resource[8] = format;
    resource[9] = 1; // sample count
    resource[11] = 1; // linear layout
    resource[12] = D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS.0 as u32;
    let counted = crate::draw::thread_counted();
    // Which texels the `n`th dispatch, of `x` with `flags`, writes, each into
    // a texture of its own.
    let written = |n: u64, x: u32, flags: u32| -> Vec<bool> {
        let address = base + n * TEXTURE_ALIGNMENT as u64;
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
        state.set_table(true, 0, handle);
        let dispatched = crate::draw::dispatched();
        state.dispatch_x(&list, x, 1, 1, flags);
        assert!(
            crate::draw::dispatched() > dispatched,
            "dispatch did not record"
        );
        crate::list::ensure_state(&list, placed, D3D12_RESOURCE_STATE_COPY_SOURCE.0 as u32);
        let row = (TEXELS * 16) as u32;
        let (readback, data) = list.readback(host, u64::from(row)).unwrap();
        // SAFETY: an open list; a live texture in the copy-source state and
        // a readback buffer the list holds, large enough for the row.
        unsafe {
            let to = D3D12_TEXTURE_COPY_LOCATION {
                pResource: std::mem::transmute_copy(&readback),
                Type: D3D12_TEXTURE_COPY_TYPE_PLACED_FOOTPRINT,
                Anonymous: D3D12_TEXTURE_COPY_LOCATION_0 {
                    PlacedFootprint: D3D12_PLACED_SUBRESOURCE_FOOTPRINT {
                        Offset: 0,
                        Footprint: D3D12_SUBRESOURCE_FOOTPRINT {
                            Format: DXGI_FORMAT_R32G32B32A32_FLOAT,
                            Width: TEXELS as u32,
                            Height: 1,
                            Depth: 1,
                            RowPitch: row,
                        },
                    },
                },
            };
            let from = D3D12_TEXTURE_COPY_LOCATION {
                pResource: std::mem::transmute_copy(&texture),
                Type: D3D12_TEXTURE_COPY_TYPE_SUBRESOURCE_INDEX,
                Anonymous: D3D12_TEXTURE_COPY_LOCATION_0 {
                    SubresourceIndex: 0,
                },
            };
            list.list.CopyTextureRegion(&to, 0, 0, 0, &from, None);
        }
        list.close();
        host.execute(&[&list]);
        // SAFETY: a plain query; the mapped row, read after the synchronous
        // submission.
        let texels: Vec<[f32; 4]> = unsafe {
            assert!(host.device.GetDeviceRemovedReason().is_ok());
            (0..TEXELS)
                .map(|i| data.cast::<[f32; 4]>().add(i).read_unaligned())
                .collect()
        };
        for t in &texels {
            assert!(*t == [0.0; 4] || *t == [1.0, 0.0, 0.0, 0.0], "texel {t:?}");
        }
        texels.iter().map(|t| t[0] == 1.0).collect()
    };
    let first = |n: usize| (0..TEXELS).map(|i| i < n).collect::<Vec<bool>>();
    assert_eq!(
        written(0, 64, 0x20),
        first(64),
        "one whole group of threads"
    );
    assert_eq!(written(1, 70, 0x20), first(70), "70 threads");
    assert_eq!(written(2, 1, 0x20), first(1), "one thread");
    assert!(crate::draw::thread_counted() >= counted + 3);
    assert_eq!(written(3, 2, 0), first(128), "two groups");
    assert_eq!(written(4, 1, 0), first(64), "one group");
    // The form the part of a group took reads the count.
    let limited = super::thread_limited(p).expect("the form whose lanes past the count leave");
    let word = limited.stages[0]
        .translated()
        .host
        .user_data
        .iter()
        .find(|u| u.root == kind);
    assert_eq!(word.map(|u| u.dword), Some(64));
}
