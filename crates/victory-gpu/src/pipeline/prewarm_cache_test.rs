//! Original shader pack and verified runtime workload for the Mac experiment.
//! Both phases run in fresh Wine processes. See tools/macos/prewarm-cache.sh.

use std::fs;
use std::path::{Path, PathBuf};
use std::ptr::null;
use std::time::{Duration, Instant};

use victory_game::pipelines::{self, Named};
use victory_game::shaderpack::{self, Pack};
use victory_shader::{Container, FourCC, native_programs};
use windows::Win32::Foundation::RECT;
use windows::Win32::Graphics::Direct3D12::*;
use windows::Win32::Graphics::Dxgi::Common::DXGI_FORMAT_R32G32B32A32_FLOAT;
use windows::core::Interface;

use super::{Bytecode, ComputeHead, PIPELINE_IIDS, StreamDesc, create_compute, create_stream};
use crate::backend::HostGpu;
use crate::list_bench::gfx10;
use crate::object::{S_OK, This, guid, slot};
use crate::pipeline::thread_dispatch_tests::read_texture_row;

const TEXELS: u32 = 128;
const ALIGN: usize = 64 << 10;
const NONE: u32 = u32::MAX;
const STORE_BYTES: usize = 8 * ALIGN;
static BACKING: [u8; STORE_BYTES] = [0; STORE_BYTES];

fn w(out: &mut Vec<u8>, words: &[u32]) {
    out.extend(words.iter().flat_map(|n| n.to_le_bytes()));
}

fn q(out: &mut Vec<u8>, words: &[u64]) {
    out.extend(words.iter().flat_map(|n| n.to_le_bytes()));
}

fn table(out: &mut Vec<u8>, bytes: &[Vec<u8>], alignment: usize) -> Vec<u8> {
    let mut blocks = Vec::new();
    for item in bytes {
        blocks.resize(blocks.len().next_multiple_of(alignment), 0);
        w(out, &[blocks.len() as u32, item.len() as u32]);
        blocks.extend(item);
    }
    blocks
}

/// Add the program IDs from the test's original containers to the synthetic
/// pack format used by victory-game's shaderpack tests. The prebuild must
/// match the pack's IDs to the actual shader containers beside it.
fn pack_bytes(programs: Vec<Vec<u8>>, compute: u32) -> Vec<u8> {
    let hash = victory_game::cityhash::city64;
    let format = DXGI_FORMAT_R32G32B32A32_FLOAT.0 as u8;
    let packet = |mask: u32| {
        let mut p = vec![0; 0xf0];
        p[4] = 4;
        p[0xcc..0xd0].copy_from_slice(&mask.to_le_bytes());
        p
    };
    let mut meta = vec![0; 0x50];
    meta[0x38] = 4; // four bytes holding an empty input layout
    meta[0x10] = format;
    meta[0x1c..0x20].copy_from_slice(&u32::MAX.to_le_bytes());
    meta[0x25..0x27].copy_from_slice(&[1, 3]); // one target, triangles
    meta[0x28] = 1;
    let blobs = [packet(15), packet(1), meta, Vec::new()];
    let depth = [
        0, 0, 1, 0, 0xff, 0xff, 1, 1, 1, 8, 1, 1, 1, 8, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1,
    ];
    let raster = [3, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0]; // solid, no culling
    let blend = |mask| {
        let mut b = [0; 68];
        for target in b[4..].as_chunks_mut::<8>().0 {
            *target = [0, 2, 1, 1, 2, 1, 1, 15];
        }
        b[11] = mask;
        b
    };

    let mut out = shaderpack::MAGIC.to_vec();
    out.extend(b"original synthetic pack");
    out.resize(0x80, 0);
    for (id, mask) in [(0, 15), (1, 1), (2, 15)] {
        let blob = u32::from(mask == 1);
        q(&mut out, &[id]);
        w(&mut out, &[blob, 2]);
        q(
            &mut out,
            &[0, 0, hash(&depth), hash(&raster), hash(&blend(mask))],
        );
        w(&mut out, &[compute + 1, NONE, compute, NONE]); // PS and VS
        w(&mut out, &[NONE; 10]);
    }
    // Two identical records for compute 0, and one missing container.
    for (record, program) in (0..compute).chain([0, compute + 2]).enumerate() {
        q(&mut out, &[record as u64]);
        w(&mut out, &[3, 3, program, NONE]);
    }
    let blob_data = table(&mut out, &blobs, 1);
    let program_data = table(&mut out, &programs, 256);
    let mut counts = Vec::new();
    w(
        &mut counts,
        &[
            3,
            compute + 2,
            blobs.len() as u32,
            programs.len() as u32,
            blob_data.len() as u32,
            program_data.len() as u32,
        ],
    );
    out[0x50..0x68].copy_from_slice(&counts);
    out.extend(blob_data);
    out.extend(program_data);
    out
}

fn filename(id: u32) -> String {
    format!("compute-{id:05}.dxbc")
}

fn generate(dir: &Path, compute: u32, rounds: u32) {
    fs::create_dir_all(dir).unwrap();
    let mut programs = Vec::new();
    let mut add = |name: &str, bytes: Vec<u8>| {
        let c = Container::parse(&bytes).unwrap();
        let part = c.part(FourCC::DXIL).unwrap();
        let shaders = native_programs(part.data).unwrap();
        let id = shaderpack::program_id(&shaders[0].code).unwrap();
        programs.push(id.to_vec());
        fs::write(dir.join(name), bytes).unwrap();
    };
    for id in 0..compute {
        add(&filename(id), gfx10::prewarm_compute(id, rounds));
    }
    add("vertex.dxbc", gfx10::prewarm_vertex());
    add("pixel.dxbc", gfx10::prewarm_pixel());
    programs.push(gfx10::pack_id_bytes(compute));
    let bytes = pack_bytes(programs, compute);
    let pack = Pack::parse(&bytes).unwrap();
    assert_eq!(pack.compute().count(), compute as usize + 2);
    let states = Named::find(&pack, &[]);
    assert_eq!(
        (states.depth.len(), states.raster.len(), states.blend.len()),
        (1, 1, 2)
    );
    fs::write(dir.join("synthetic-pack"), bytes).unwrap();
    eprintln!(
        "fixture {compute} compute shaders, 2 graphics shaders, 2 graphics states, 2 duplicate records, 1 missing container"
    );
}

struct Guest(This);

impl Drop for Guest {
    fn drop(&mut self) {
        slot::<extern "system" fn(This) -> u32>(self.0, 0x10)(self.0);
    }
}

fn bytecode(bytes: &[u8]) -> Bytecode {
    Bytecode {
        code: bytes.as_ptr(),
        size: bytes.len(),
    }
}

fn make_compute(bytes: &[u8]) -> Guest {
    let mut pipeline = null();
    let desc = ComputeHead {
        root: null(),
        stage: bytecode(bytes),
    };
    assert_eq!(
        create_compute(null(), &desc, &PIPELINE_IIDS[0], &mut pipeline),
        S_OK
    );
    Guest(pipeline)
}

fn make_graphics(stream: &[u8]) -> Guest {
    let mut pipeline = null();
    let desc = StreamDesc {
        size: stream.len(),
        stream: stream.as_ptr(),
    };
    assert_eq!(
        create_stream(null(), &desc, &PIPELINE_IIDS[0], &mut pipeline),
        S_OK
    );
    Guest(pipeline)
}

/// Do not let a slow pipeline silently drop the dispatch or draw. The
/// production pipeline is requested on demand and may use a worker.
fn wait_for_pipeline(pipeline: &Guest) {
    let started = Instant::now();
    while super::host_pipeline(pipeline.0).is_none() {
        assert!(
            started.elapsed() < Duration::from_secs(30),
            "pipeline did not become ready"
        );
        std::thread::sleep(Duration::from_millis(5));
    }
}

fn texture(index: usize, flags: D3D12_RESOURCE_FLAGS) -> (This, ID3D12Resource, [u32; 14], u64) {
    let base = (BACKING.as_ptr() as usize).next_multiple_of(ALIGN);
    let address = (base + index * ALIGN) as u64;
    assert!(index < STORE_BYTES / ALIGN - 1);
    let mut desc = [0; 14];
    desc[0] = D3D12_RESOURCE_DIMENSION_TEXTURE2D.0 as u32;
    desc[4] = TEXELS;
    desc[6] = 1;
    desc[7] = 1 | 1 << 16; // one slice, one mip
    desc[8] = DXGI_FORMAT_R32G32B32A32_FLOAT.0 as u32;
    desc[9] = 1; // one sample
    desc[11] = 1; // linear
    desc[12] = flags.0 as u32;
    let placed = crate::resource::texture_at(address, &desc).expect("placed texture");
    let texture = crate::resource::host_resource(placed).expect("host texture");
    (placed, texture, desc, address)
}

fn dispatch(host: &crate::host::Host, pso: &Guest, index: usize) -> Vec<[f32; 4]> {
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
    let mut heap = null();
    assert_eq!(
        crate::descriptor::create_heap(
            null(),
            std::ptr::from_ref(&desc).cast(),
            &guid(iid.data1, iid.data2, iid.data3, iid.data4),
            &mut heap,
        ),
        S_OK
    );
    let heap = Guest(heap);
    let handle = crate::descriptor::heap(heap.0).start + 32;
    let (placed, texture, desc, address) =
        texture(index, D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS);
    let mut view = [0; 16];
    view[0] = DXGI_FORMAT_R32G32B32A32_FLOAT.0 as u32;
    view[1] = D3D12_UAV_DIMENSION_TEXTURE2D.0 as u32;
    view[10] = address as u32;
    view[11] = (address >> 32) as u32;
    crate::descriptor::create_placed_uav(null(), &desc, &view, handle);
    let list = host.new_list().unwrap();
    crate::texture::upload(&list, placed, &texture);
    let mut state = crate::draw::State::default();
    state.set_pipeline(pso.0);
    state.set_heaps(&[heap.0]);
    state.set_table(true, 0, handle);
    let dispatched = crate::draw::dispatched();
    state.dispatch(&list, 2, 1, 1);
    assert_eq!(crate::draw::dispatched(), dispatched + 1);
    crate::list::ensure_state(&list, placed, D3D12_RESOURCE_STATE_COPY_SOURCE.0 as u32);
    read_texture_row(host, &list, &texture, TEXELS)
}

fn draw(host: &crate::host::Host, pso: &Guest, index: usize) -> Vec<[f32; 4]> {
    let (placed, texture, _, _) = texture(index, D3D12_RESOURCE_FLAG_ALLOW_RENDER_TARGET);
    let list = host.new_list().unwrap();
    crate::texture::upload(&list, placed, &texture);
    let mut view = [0u64; 16];
    let handle = view.as_mut_ptr() as u64;
    crate::descriptor::create_rtv(null(), placed, null(), handle);
    let mut state = crate::draw::State::default();
    state.set_pipeline(pso.0);
    state.set_topology(&list, 4); // triangles
    state.set_targets(&list, &[handle], 0);
    state.viewports(
        &list,
        &[D3D12_VIEWPORT {
            Width: TEXELS as f32,
            Height: 1.0,
            MaxDepth: 1.0,
            ..Default::default()
        }],
    );
    state.scissors(
        &list,
        &[RECT {
            left: 0,
            top: 0,
            right: TEXELS as i32,
            bottom: 1,
        }],
    );
    let drawn = crate::draw::drawn();
    state.draw(&list, 3, 1, 0, 0);
    assert_eq!(crate::draw::drawn(), drawn + 1);
    crate::list::ensure_state(&list, placed, D3D12_RESOURCE_STATE_COPY_SOURCE.0 as u32);
    read_texture_row(host, &list, &texture, TEXELS)
}

fn sample(phase: &str) {
    let signals = std::env::var_os("VICTORY_PREWARM_TEST_SIGNALS").map(PathBuf::from);
    eprintln!("sampling {phase}");
    if let Some(signals) = signals {
        let resume = signals.join(phase);
        let wait = Instant::now();
        while !resume.is_file() {
            assert!(
                wait.elapsed() < Duration::from_secs(60),
                "sampling {phase} timed out"
            );
            std::thread::sleep(Duration::from_millis(10));
        }
    }
}

fn run(pack_dir: &Path, cache: &Path, rounds: u32, use_excluded: bool) {
    let _ = tracing_subscriber::fmt()
        .with_writer(std::io::stderr)
        .try_init();
    let pipeline_cache = cache.join("pipelines");
    fs::create_dir_all(&pipeline_cache).unwrap();
    super::use_pipeline_cache(pipeline_cache);

    let started = Instant::now();
    let host = crate::host::host().expect("real D3D12 device required");
    assert!(
        crate::under_wine(),
        "this experiment runs with CrossOver Wine and D3DMetal"
    );
    eprintln!(
        "device ready in {:.3} ms",
        started.elapsed().as_secs_f64() * 1000.0
    );
    sample("device");

    crate::shader_cache::use_shader_cache(cache.join("shaders")).expect("shader cache");
    let mut mismatches = 0;
    let (made0, loaded0) = super::pipeline_counts();
    let (hits0, translated0) = crate::shader_cache::counts();
    let shader_ids: &[u32] = if use_excluded { &[0, 1, 2] } else { &[0, 1] };
    for (i, &id) in shader_ids.iter().enumerate() {
        let bytes = fs::read(pack_dir.join(filename(id))).unwrap();
        let (made, loaded) = super::pipeline_counts();
        let pso = make_compute(&bytes);
        wait_for_pipeline(&pso);
        let texels = dispatch(host, &pso, i);
        for (x, pixel) in texels.iter().enumerate() {
            let mut mixed = x as u32;
            for round in 0..rounds {
                mixed = mixed.wrapping_add(gfx10::mix_constant(id, round));
                mixed ^= mixed >> 13;
            }
            let want = f32::from_bits(1f32.to_bits() | (mixed & 0x007f_ffff));
            if *pixel != [want, 0.0, 0.0, 0.0] {
                if mismatches == 0 {
                    eprintln!("first mismatch: compute {id} texel {x}, {pixel:?} vs {want}");
                }
                mismatches += 1;
            }
        }
        let (now_made, now_loaded) = super::pipeline_counts();
        let key = super::Prebuild::compute(&bytes).key();
        eprintln!(
            "compute {id} {key}: {} made, {} loaded",
            now_made - made,
            now_loaded - loaded
        );
    }

    let bytes = fs::read(pack_dir.join("synthetic-pack")).unwrap();
    let pack = Pack::parse(&bytes).unwrap();
    let named = Named::find(&pack, &[]);
    let vs = fs::read(pack_dir.join("vertex.dxbc")).unwrap();
    let ps = fs::read(pack_dir.join("pixel.dxbc")).unwrap();
    for (i, graphics) in pack.graphics().take(2).enumerate() {
        let stream = pipelines::stream(&graphics, &named, &vs, false, Some(&ps)).unwrap();
        let pso = make_graphics(stream.bytes());
        wait_for_pipeline(&pso);
        let want = if i == 0 {
            [1.0, 0.0, 0.5, 1.0]
        } else {
            [1.0, 0.0, 0.0, 0.0]
        };
        for (x, pixel) in draw(host, &pso, i + 3).iter().enumerate() {
            if *pixel != want {
                if mismatches == 0 {
                    eprintln!("first mismatch: graphics {i} texel {x}, {pixel:?} vs {want:?}");
                }
                mismatches += 1;
            }
        }
    }

    // SAFETY: a plain query on the live, previously used device.
    let errors = usize::from(unsafe { host.device.GetDeviceRemovedReason() }.is_err());
    let (made, loaded) = super::pipeline_counts();
    let (hits, translated) = crate::shader_cache::counts();
    eprintln!(
        "shader translations: {} cached, {} translated",
        hits - hits0,
        translated - translated0
    );
    eprintln!(
        "workload: {} dispatches, 2 draws, {} made, {} loaded, {mismatches} mismatches, {errors} device errors",
        shader_ids.len(),
        made - made0,
        loaded - loaded0,
    );
    assert_eq!((mismatches, errors), (0, 0));
    sample("workload");
    super::save_pipeline_cache();
}

#[test]
#[ignore = "explicit hardware experiment: tools/macos/prewarm-cache.sh"]
fn synthetic_prewarm_cache() {
    let path = |name| PathBuf::from(std::env::var_os(name).expect(name));
    let number = |name| std::env::var(name).expect(name).parse::<u32>().expect(name);
    let pack = path("VICTORY_PREWARM_TEST_PACK");
    let rounds = number("VICTORY_PREWARM_TEST_ROUNDS");
    assert!((1..=128).contains(&rounds));
    match std::env::var("VICTORY_PREWARM_TEST_PHASE")
        .unwrap()
        .as_str()
    {
        "generate" => {
            let count = number("VICTORY_PREWARM_TEST_COUNT");
            assert!((3..=4096).contains(&count));
            generate(&pack, count, rounds);
        }
        "run" => run(&pack, &path("VICTORY_CACHE_DIR"), rounds, false),
        "fallback" => run(&pack, &path("VICTORY_CACHE_DIR"), rounds, true),
        _ => panic!("unknown synthetic phase"),
    }
}
