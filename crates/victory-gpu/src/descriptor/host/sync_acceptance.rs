//! The title writes descriptor records as bytes, and the host consumes them
//! for the first time while preparing a real GPU submission.

use super::tests::{placed_tex_desc, placed_tex_srv_desc, prepared, when_preparing};
use super::*;
use crate::compute::{self, Param, srv, uav};
use crate::descriptor::{create_cbv, create_heap, create_placed_srv, create_srv, create_uav, heap};
use crate::object::{S_OK, This, release_ref};
use std::ptr::null;

const SOURCE: &str = r#"
ByteAddressBuffer raw_view : register(t0);
ByteAddressBuffer duplicate_view : register(t1);
Buffer<uint> typed_view : register(t2);
ByteAddressBuffer structured_view : register(t3);
ByteAddressBuffer constant_view : register(t4);
Buffer<uint4> widened_view : register(t5);
Texture2D<float4> texture_view : register(t6);
ByteAddressBuffer far_view : register(t7);
RWByteAddressBuffer result : register(u0);
[numthreads(1, 1, 1)]
void main() {
    result.Store(0, raw_view.Load(0));
    result.Store(4, duplicate_view.Load(0));
    result.Store(8, typed_view[0]);
    result.Store(12, structured_view.Load(4));
    result.Store(16, constant_view.Load(0));
    result.Store4(20, widened_view[1]);
    result.Store(36, asuint(texture_view.Load(int3(0, 0, 0)).x));
    result.Store(40, far_view.Load(0));
}
"#;

thread_local! {
    static QUICK_PUBLICATION: std::cell::Cell<Option<bool>> = const { std::cell::Cell::new(None) };
}

pub(super) fn note_publication(
    updates: &[(u32, crate::bindless::BatchUpdate<'_>)],
    finalize: &[(u32, u64, u64)],
) {
    QUICK_PUBLICATION.with(|published| {
        if published.get() != Some(false) {
            return;
        }
        published.set(Some(updates.iter().any(|(index, update)| {
            matches!(update, crate::bindless::BatchUpdate::Quick(..))
                && finalize
                    .iter()
                    .any(|&(slot, _, ready)| slot == *index && ready != 0)
        })));
    });
}

const FAR: usize = 512 << 10;
const LEN: usize = FAR + 4096;
const BENCH_WARMUP: usize = 3;
const BENCH_PASSES: usize = 23;
const RAW: [u32; 10] = [39, 1, 0x1688, 0, 0, 0, 4, 0, 1, 0];

fn locate(resource: This, len: usize) -> u64 {
    let (address, size) = crate::resource::buffer_bytes(resource).expect("title buffer bytes");
    assert_eq!(size, len as u64);
    address as u64
}

fn srv_record(resource: This, words: &[u32; 10]) -> [u32; 4] {
    let mut record = [0u32; 4];
    create_srv(null(), resource, words, record.as_mut_ptr() as u64);
    record
}

fn write_slot(at: u64, record: [u32; 4]) {
    // SAFETY: the caller supplies a slot in the test's tracked heap; the
    // title writes these 16 bytes without using descriptor API copy calls.
    unsafe { (at as *mut [u32; 4]).write_volatile(record) };
}

fn write_data(address: u64, offset: usize, value: u32) {
    // SAFETY: the caller supplies a dword in a live test title buffer.
    unsafe { (address as *mut u32).add(offset / 4).write_volatile(value) };
}

#[test]
#[ignore = "requires Wine with D3DMetal, a D3D12 device, and the HLSL compiler"]
fn watched_descriptor_records_reach_gpu_readback() {
    run(false);
}

#[test]
#[ignore = "measure in release mode under Wine with D3DMetal"]
fn bench_watched_descriptor_submissions() {
    run(true);
}

fn run(benchmark: bool) {
    assert!(
        crate::under_wine(),
        "acceptance requires Wine with D3DMetal"
    );
    assert!(
        raw_structured(),
        "acceptance expects structured SRVs lowered to raw"
    );
    assert!(sync_fast_enabled(), "acceptance requires batched heap sync");
    let gpu = host().expect("acceptance requires a D3D12 device");
    let (root, pso) = compute::pipeline(
        SOURCE,
        windows::core::s!("watched_descriptor_records"),
        0,
        &[
            Param::Table(&[srv(0, 8).volatile()]),
            Param::Table(&[uav(0, 1).volatile()]),
        ],
    )
    .expect("acceptance compute pipeline");

    let source = crate::resource::test_buffer(LEN as u64);
    let other = crate::resource::test_buffer(256);
    let output = crate::resource::test_buffer(256);
    let a = locate(source, LEN);
    let b = locate(other, 256);
    let result = locate(output, 256);
    for (offset, value) in (0..32).step_by(4).zip(101..109) {
        write_data(a, offset, value);
    }
    write_data(a, 64, 210);
    write_data(a, FAR, 301);
    write_data(b, 0, 401);
    let result_copy = crate::shadow::find(result, 256).expect("output shadow");
    // Populate the shadow registry before resolving descriptors, so its
    // first insertions do not invalidate the page's prepared quick views.
    assert!(
        crate::shadow::find(a, LEN as u64).is_some(),
        "source shadow"
    );
    assert!(crate::shadow::find(b, 256).is_some(), "replacement shadow");

    const BENCH_PAGES: usize = 32;
    let pages = if benchmark { BENCH_PAGES } else { 1 };
    let slots = if benchmark {
        pages * crate::watch::PAGE / 32
    } else {
        16
    };
    let page = crate::watch::tests::pool(pages);
    let heap_desc = [
        0u32,
        slots as u32,
        1,
        0,
        page as u32,
        (page as u64 >> 32) as u32,
        (slots * 32) as u32,
        0,
    ];
    let iid = crate::object::guid(
        0x8efb471d,
        0x616c,
        0x4f49,
        [0x90, 0xf7, 0x12, 0x7b, 0xb7, 0x63, 0xfa, 0x51],
    );
    let mut title = null();
    assert_eq!(
        create_heap(null(), heap_desc.as_ptr().cast(), &iid, &mut title),
        S_OK
    );
    let title_heap = heap(title);
    assert_eq!(title_heap.start, page as u64);
    let at = |slot| title_heap.handle(slot).expect("test descriptor slot");
    let h = title_heap.host().expect("acceptance host descriptor heap");

    let raw = srv_record(source, &RAW);
    let raw_other = srv_record(other, &RAW);
    let typed = srv_record(source, &[42, 1, 0x1688, 0, 2, 0, 1, 0, 0, 0]);
    // The translator reads this 16-byte structured view through a raw
    // handle. A nonzero FirstElement makes the offset observable.
    let structured = srv_record(source, &[0, 1, 0x1688, 0, 1, 0, 1, 16, 0, 0]);
    let mut constant = [0u32; 4];
    let ca = a + 64;
    create_cbv(
        null(),
        &[ca as u32, (ca >> 32) as u32, 16, 0],
        constant.as_mut_ptr() as u64,
    );
    let widened = srv_record(source, &[7, 1, 0x1688, 0, 0, 0, 2, 0, 0, 0]);
    let pixels = crate::watch::tests::pool(1);
    let mut tex_desc = placed_tex_desc(4, 1, 28, 0);
    tex_desc[11] = 1; // Linear title texture memory.
    let tex_view = placed_tex_srv_desc(pixels as u64, 28);
    let mut texture = [0u32; 4];
    create_placed_srv(null(), &tex_desc, &tex_view, texture.as_mut_ptr() as u64);
    let far = srv_record(source, &[39, 1, 0x1688, 0, FAR as u32 / 4, 0, 1, 0, 1, 0]);
    let mut output_view = [0u32; 4];
    create_uav(
        null(),
        output,
        null(),
        &[39, 1, 0, 0, 64, 0, 0, 0, 1, 0],
        output_view.as_mut_ptr() as u64,
    );
    for (slot, record) in [
        raw,
        raw,
        typed,
        structured,
        constant,
        widened,
        texture,
        far,
        output_view,
    ]
    .into_iter()
    .enumerate()
    {
        write_slot(at(slot as u32), record);
        assert!(
            !h.made(slot as u32),
            "title byte write must not mirror slot {slot}"
        );
    }

    let record_dispatch = |output_slot, copy_to_title: Option<usize>| {
        let list = gpu.new_list().expect("acceptance command list");
        list.defer_heap(&h.viewed);
        if let Some(offset) = copy_to_title {
            list.defer_gpu_write(&result_copy, result as usize + offset, 44, false);
        }
        // SAFETY: live host heap, pipeline and command list. Used slots are
        // populated by submission preparation before the list is executed.
        unsafe {
            list.list.SetDescriptorHeaps(&[Some(h.heap.clone())]);
            list.list.SetComputeRootSignature(&root);
            list.list.SetComputeRootDescriptorTable(0, h.gpu(0));
            list.list
                .SetComputeRootDescriptorTable(1, h.gpu(output_slot));
            list.list.SetPipelineState(&pso);
            list.list.Dispatch(1, 1, 1);
        }
        list.close();
        list
    };
    let dispatch = |label: &str, expected: [u32; 11]| {
        result_copy.write_host(result, &[0xa5; 44]);
        let list = record_dispatch(8, None);
        let started = std::time::Instant::now();
        gpu.execute_with_barriers(&[&list], &[]);
        let elapsed = started.elapsed();
        // SAFETY: plain device health query after the fence completes.
        assert!(unsafe { gpu.device.GetDeviceRemovedReason() }.is_ok());
        let got = result_copy.read_host(result, expected.len());
        assert_eq!(got, expected, "GPU readback after {label}");
        if !benchmark {
            eprintln!("dispatch=1 stage={label} checked={got:?}");
        }
        elapsed
    };

    let initial = [101, 101, 103, 106, 210, 104, 105, 106, 1, 0, 301];
    if !benchmark {
        QUICK_PUBLICATION.set(Some(false));
    }
    dispatch("initial_and_duplicate", initial);
    if !benchmark {
        let used = QUICK_PUBLICATION
            .replace(None)
            .expect("publication tracking");
        assert_eq!(
            used,
            quick::on(),
            "binding a quick view must publish its live descriptor"
        );
        eprintln!("quick_resolved_published={used}");
    }
    assert!(
        h.made(6) && !h.kept(6),
        "placed texture bound with the null texture fallback"
    );
    assert!(h.kept(8), "UAV fallback created a real output descriptor");
    if benchmark {
        let views = [0, 64].map(|start| {
            (start..start + 60)
                .map(|first| {
                    let mut words = RAW;
                    words[4] = first;
                    words[6] = 1;
                    srv_record(source, &words)
                })
                .collect::<Vec<_>>()
        });
        let clear = [0; 4];
        let snapshot = || {
            crate::descriptor::SYNC_STATS
                .each_ref()
                .map(|counter| counter.load(Ordering::Relaxed))
        };
        let mut elapsed = std::time::Duration::ZERO;
        let mut before = snapshot();
        for pass in 0..BENCH_WARMUP + BENCH_PASSES {
            if pass == BENCH_WARMUP {
                before = snapshot();
            }
            let toggle = (pass + 1) & 1;
            // Dirty all 32 pages. The unused first slot on each later page
            // stays zero, so the scanner sees the real unchanged-page work.
            for i in 1..32 {
                write_slot(at(i * 2048), clear);
            }
            for i in 0..740 {
                let table = &views[toggle];
                write_slot(at(4096 + i), table[i as usize % table.len()]);
            }
            for i in 0..960 {
                write_slot(at(6144 + i), if toggle == 0 { texture } else { clear });
            }
            write_slot(at(0), if toggle == 0 { raw } else { raw_other });
            let mut expected = initial;
            expected[0] = if toggle == 0 { 101 } else { 401 };
            let time = dispatch("benchmark", expected);
            if pass >= BENCH_WARMUP {
                elapsed += time;
            }
        }
        use crate::descriptor::SyncStat as S;
        let after = snapshot();
        let stat = |i: S| after[i as usize] - before[i as usize];
        let heap_sync_us = stat(S::TakeUs) + stat(S::ReadUs) + stat(S::CheckUs);
        assert_eq!(stat(S::Syncs), BENCH_PASSES as u64);
        assert_eq!(stat(S::Slots), 32 * 2048 * BENCH_PASSES as u64);
        assert!(stat(S::Created) > 0 && stat(S::FromTemplate) > 0);
        eprintln!(
            "bench_watched_descriptor_submissions: submissions={BENCH_PASSES} slots={} total_us={} heap_sync_parts_us={heap_sync_us} made={} creations={} template_copies={} readback_words={}",
            stat(S::Slots),
            elapsed.as_micros(),
            stat(S::Made),
            stat(S::Created),
            stat(S::FromTemplate),
            BENCH_PASSES * initial.len(),
        );
    }

    let before = h.viewed.hull_generation();
    write_slot(at(0), raw);
    dispatch("unchanged_record", initial);
    assert_eq!(h.viewed.hull_generation(), before);

    // A writer changes a candidate after taking the page snapshot, during
    // preparation of the first slot. Only this writer accesses title heap
    // bytes while the syncing thread waits in the test hook.
    write_slot(at(0), raw_other);
    let race_slot = at(0);
    when_preparing(move || {
        std::thread::spawn(move || write_slot(race_slot, raw))
            .join()
            .expect("title writer");
    });
    dispatch("concurrent_rewrite", initial);
    assert!(prepared(), "rewrite hook ran during the page batch");

    write_slot(at(0), raw_other);
    write_slot(at(9), raw_other);
    write_data(a, 16, 555);
    write_data(a, FAR, 302);
    let changed = [401, 101, 103, 106, 210, 104, 555, 106, 1, 0, 302];
    dispatch("changed_record_and_widened_source", changed);

    assert!(h.kept(9), "the old resource is live in two duplicate slots");
    assert_eq!(release_ref(other), 0);
    assert!(
        crate::descriptor::view(at(0)).is_none(),
        "removed resource's view is stale"
    );
    let replacement = crate::resource::test_buffer(256);
    let replacement_address = locate(replacement, 256);
    write_data(replacement_address, 0, 501);
    let replacement_raw = srv_record(replacement, &RAW);
    assert_eq!(
        raw_other[1], replacement_raw[1],
        "the view index was reused"
    );
    assert_ne!(
        raw_other[2], replacement_raw[2],
        "the reused index received a new generation"
    );
    assert!(
        !crate::descriptor::view_gen_current(raw_other[1], raw_other[2]),
        "old record remains stale after reusing the view table",
    );
    // Slot 6 is consumed as a texture (the generic null slot is a null 2D
    // SRV). Slot 9 is unread by the shader and can hold a stale raw record.
    write_slot(at(6), [0; 4]);
    write_slot(at(9), raw_other);
    write_slot(at(0), replacement_raw);
    let mut replaced = changed;
    replaced[0] = 501;
    dispatch("cleared_texture_and_replaced_resource", replaced);
    assert!(
        !h.made(6) && !h.kept(6),
        "cleared texture descriptor is no longer retained"
    );
    assert!(
        !h.made(9) && !h.kept(9),
        "old generation was not republished"
    );
    let mut submissions = if benchmark {
        BENCH_PASSES + BENCH_WARMUP + 5
    } else {
        5
    };
    if !benchmark {
        // A placed texture that a draw first reads leaves the null fallback.
        // Its host upload and the following read still run through submission.
        write_data(pixels as u64, 0, 255);
        write_slot(at(6), texture);
        assert!(h.mirror(6, at(6), true).is_some() && h.kept(6));
        replaced[9] = 1f32.to_bits();
        dispatch("texture_made_for_draw", replaced);
        submissions += 1;

        // The placed buffer uses a GPU-backed alias of the title mapping.
        // A subsequent CPU edit reaches the same host SRV without rewriting
        // its record. Use a separate pool from the watched descriptor heap.
        let alias = crate::watch::tests::pool(1);
        write_data(alias as u64, 0, 601);
        assert!(
            crate::alias::alias_region(alias, crate::watch::PAGE, true),
            "GPU memory alias"
        );
        let placed_buffer = [1, 0, 0, 0, 16, 0, 1, 1 | 1 << 16, 0, 1, 0, 1, 0, 0];
        let mut alias_desc = [0; 16];
        alias_desc[..10].copy_from_slice(&RAW);
        alias_desc[10] = alias as u32;
        alias_desc[11] = (alias as u64 >> 32) as u32;
        let mut alias_record = [0; 4];
        create_placed_srv(
            null(),
            &placed_buffer,
            &alias_desc,
            alias_record.as_mut_ptr() as u64,
        );
        write_slot(at(0), alias_record);
        replaced[0] = 601;
        dispatch("aliased_placed_buffer", replaced);
        write_data(alias as u64, 0, 602);
        replaced[0] = 602;
        dispatch("unchanged_alias_descriptor", replaced);
        submissions += 2;

        if crate::switch::on("VICTORY_SYNC_EARLY") {
            assert!(
                crate::host::flight1_enabled(),
                "early sync requires VICTORY_FLIGHT1=1"
            );
            assert!(
                !matches!(
                    std::env::var("VICTORY_PREP_AHEAD").as_deref(),
                    Ok("1" | "ranges")
                ),
                "early sync uses wait-before-prep"
            );
            // Use separate UAV ranges, and update slots outside flight 1's
            // live descriptor set while the GPU queue still waits on a gate.
            let mut second_output = [0; 4];
            create_uav(
                null(),
                output,
                null(),
                &[39, 1, 16, 0, 48, 0, 0, 0, 1, 0],
                second_output.as_mut_ptr() as u64,
            );
            result_copy.write_host(result, &[0xa5; 128]);
            let first = std::sync::Arc::new(record_dispatch(8, Some(0)));
            let second = std::sync::Arc::new(record_dispatch(10, Some(64)));
            let mut gate = crate::host::Gate::hold(gpu);
            gpu.execute_flight1(vec![first], vec![]);
            assert!(
                gpu.has_in_flight(),
                "first dispatch is held on the GPU queue"
            );
            write_slot(at(9), raw);
            write_slot(at(10), second_output);
            when_preparing(move || {
                assert!(
                    gpu.has_in_flight(),
                    "second heap update occurs before first landing"
                );
                gate.open();
            });
            gpu.execute_flight1(vec![second], vec![]);
            gpu.finish_flight_now();
            assert!(prepared());
            for offset in [0, 64] {
                // SAFETY: both lists landed, and the title buffer holds both
                // output ranges. Landing copied the writes back to the title.
                let got = unsafe { ((result + offset) as *const [u32; 11]).read_volatile() };
                assert_eq!(
                    got, replaced,
                    "early-sync GPU output at byte offset {offset}"
                );
                eprintln!("dispatch=1 stage=early_sync_output_{offset} checked={got:?}");
            }
            submissions += 2;
        }

        // The final completed submission leaves the alias unused from now on.
        assert!(crate::alias::unalias_region(alias));
        crate::watch::tests::free(alias);
    }

    // SAFETY: the device remains live after the submissions.
    let adapter = unsafe { gpu.device.GetAdapterLuid() };
    eprintln!(
        "backend=Wine/D3D12 adapter_luid={:08x}:{:08x} submissions={submissions} dispatches={submissions} record_generations={}->{}",
        adapter.HighPart, adapter.LowPart, raw_other[2], replacement_raw[2],
    );

    release_ref(title);
    for resource in [source, replacement, output] {
        assert_eq!(release_ref(resource), 0);
    }
    crate::watch::tests::free(page);
    crate::watch::tests::free(pixels);
}
