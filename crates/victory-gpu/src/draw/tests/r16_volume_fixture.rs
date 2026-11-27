//! The generated R16_FLOAT volume from #242. Keep the DXC executable and
//! its DLLs together. Run this ignored test in a fresh process with
//! `VICTORY_PC_DXC` set and `VICTORY_GPU_TIMING` unset.

use std::path::Path;

use super::*;

const WIDTH: usize = 60;
const HEIGHT: usize = 34;
const DEPTH: usize = 512;
const SLICES: [usize; 8] = [0, 7, 8, 15, 16, 40, 255, 511];
const R16_FLOAT: u32 = 54;
const R32_FLOAT: u32 = 41;
const SENTINEL: u32 = 123456789;
const COMMON: u32 = D3D12_RESOURCE_STATE_COMMON.0 as u32;
const UAV: u32 = D3D12_RESOURCE_STATE_UNORDERED_ACCESS.0 as u32;
const SRV: u32 = (D3D12_RESOURCE_STATE_PIXEL_SHADER_RESOURCE.0
    | D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE.0) as u32;

#[derive(Clone, Copy)]
enum Layout {
    SameList,
    SplitBatch,
    SplitSeparate,
    SplitTail,
}

impl Layout {
    fn name(self) -> &'static str {
        match self {
            Self::SameList => "same-list",
            Self::SplitBatch => "split-batch",
            Self::SplitSeparate => "split-separate",
            Self::SplitTail => "split-tail",
        }
    }

    /// The number of lists in each native call, grouped by logical fence.
    /// The last list in every native call is the completion mark.
    fn native_calls(self, split: bool) -> &'static [&'static [usize]] {
        match self {
            Self::SameList => &[&[3]],
            Self::SplitBatch if split => &[&[3, 3]],
            Self::SplitBatch => &[&[5]],
            Self::SplitSeparate => &[&[3], &[3]],
            Self::SplitTail => &[&[4]],
        }
    }
}

#[derive(Default)]
struct Mismatches {
    untouched: usize,
    read_zero: usize,
    wrong_nonzero: usize,
}

impl Mismatches {
    fn count(&self) -> usize {
        self.untouched + self.read_zero + self.wrong_nonzero
    }

    fn add(&mut self, observed: u32) {
        match observed {
            SENTINEL => self.untouched += 1,
            0 => self.read_zero += 1,
            _ => self.wrong_nonzero += 1,
        }
    }
}

const PRODUCE: &str = include_str!("r16_volume_produce.hlsl");
const CONSUME: &str = include_str!("r16_volume_consume.hlsl");

fn compile(dxc: &Path, dir: &Path, name: &str, source: &str) -> Vec<u8> {
    let input = dir.join(format!("{name}.hlsl"));
    let output = dir.join(format!("{name}.dxil"));
    std::fs::write(&input, source).expect("write the generated fixture shader");
    let result = std::process::Command::new(dxc)
        .args(["-T", "cs_6_0", "-E", "main", "-Fo"])
        .arg(&output)
        .arg(&input)
        .output()
        .expect("start the PC DXC executable");
    assert!(
        result.status.success(),
        "DXC failed for {name}: {}",
        String::from_utf8_lossy(&result.stderr)
    );
    let shader = std::fs::read(output).expect("read DXC's generated shader");
    assert!(
        shader.starts_with(b"DXBC")
            && shader
                .get(4..20)
                .is_some_and(|digest| digest.iter().any(|&byte| byte != 0)),
        "{name}: expected signed DXIL; keep the matching dxil.dll beside dxc.exe"
    );
    shader
}

fn pipeline(
    host: &crate::host::Host,
    shader: &[u8],
    params: &[Param],
) -> (ID3D12RootSignature, ID3D12PipelineState) {
    let root = crate::compute::root_signature(&host.device, params).unwrap();
    // SAFETY: the bytecode and root signature live through this call.
    let pso = unsafe {
        host.device
            .CreateComputePipelineState(&D3D12_COMPUTE_PIPELINE_STATE_DESC {
                pRootSignature: std::mem::transmute_copy(&root),
                CS: D3D12_SHADER_BYTECODE {
                    pShaderBytecode: shader.as_ptr().cast(),
                    BytecodeLength: shader.len(),
                },
                ..Default::default()
            })
    }
    .expect("validated generated compute pipeline");
    (root, pso)
}

fn expected(x: usize, y: usize, z: usize) -> u32 {
    let lane = (x % 2) + 2 * (y % 2) + 4 * (z % 16);
    let cross = (lane & 62) ^ 32;
    let value = 1 + (3 * x + 5 * y + 7 * z + cross) % 1536;
    (value as f32).to_bits()
}

#[test]
#[ignore = "requires PC DXC and hardware D3D12; see #242"]
fn r16_volume_validate_uav_srv_prelude_split() {
    run_volume_fixture(R16_FLOAT);
}

#[test]
#[ignore = "requires PC DXC and hardware D3D12"]
fn r32_volume_validate_uav_srv_prelude_split() {
    run_volume_fixture(R32_FLOAT);
}

fn run_volume_fixture(format: u32) {
    assert!(
        !crate::switch::on("VICTORY_GPU_TIMING"),
        "unset VICTORY_GPU_TIMING to exercise batched submissions"
    );
    let dxc = std::env::var_os("VICTORY_PC_DXC").expect("set VICTORY_PC_DXC to dxc.exe");
    let shaders = tempfile::tempdir().expect("temporary fixture shader directory");
    let produce = compile(Path::new(&dxc), shaders.path(), "produce", PRODUCE);
    let consume = compile(Path::new(&dxc), shaders.path(), "consume", CONSUME);
    let host = crate::host::host().expect("hardware host required");
    let write = pipeline(host, &produce, WRITES);
    let read = pipeline(host, &consume, READS);
    let split = crate::switch::on("VICTORY_GPU_SPLIT_UAV_SRV_PRELUDE");
    let words_count = WIDTH * HEIGHT * SLICES.len();
    use Layout::*;
    let forward = [SameList, SplitBatch, SplitSeparate, SplitTail];
    let reverse = [SplitTail, SplitSeparate, SplitBatch, SameList];
    let mut failed_cases = 0;
    for (round, layout) in forward.into_iter().chain(reverse).enumerate() {
        let iteration = round / forward.len() + 1;
        let name = layout.name();
        let memory = title_memory(1 << 23);
        let desc = [
            4,
            0,
            0,
            0,
            WIDTH as u32,
            0,
            HEIGHT as u32,
            DEPTH as u32 | 1 << 16,
            format,
            1,
            0,
            0,
            4,
            0,
        ];
        let texture = texture_at_address(memory, &desc);
        let uav = View::Unordered(
            texture as usize,
            0,
            Some([format, 8, 0, 0, u32::MAX, 0, 0, 0, 0, 0]),
        );
        let srv = View::Shader(
            texture as usize,
            Some([format, 8, 0x1688, 0, 0, 1, 0, 0, 0, 0]),
        );
        let slots = Slots::new(2);
        slots.make(0, &uav);
        slots.make(1, &srv);
        let (output, mapped) = host.shared_buffer((words_count * 4) as u64).unwrap();
        // SAFETY: the test owns the aligned mapped range, which has not been submitted yet.
        unsafe { std::slice::from_raw_parts_mut(mapped.cast::<u32>(), words_count) }.fill(SENTINEL);

        let resource = crate::resource::host_resource(texture).unwrap();
        let resource_ptr = resource.as_raw() as usize;
        let barriers = || {
            crate::host::tests::issued()
                .into_iter()
                .filter_map(|(resource, before, after)| {
                    (resource == resource_ptr).then_some((before, after))
                })
                .collect::<Vec<_>>()
        };
        barriers();

        let producer = host.new_list().unwrap();
        readable(&producer, &used(ResourceType::Uav), &uav, false);
        // SAFETY: an open list and live objects.
        unsafe {
            producer.list.SetDescriptorHeaps(&[Some(slots.0.clone())]);
            producer.list.SetComputeRootSignature(&write.0);
            producer.list.SetPipelineState(&write.1);
            producer.list.SetComputeRootDescriptorTable(0, slots.gpu(0));
            producer
                .list
                .Dispatch(WIDTH as u32 / 2, HEIGHT as u32 / 2, DEPTH as u32 / 16);
        }
        if matches!(layout, SplitTail) {
            readable(&producer, &used(ResourceType::Srv), &srv, false);
        }
        let consumer = (!matches!(layout, SameList)).then(|| host.new_list().unwrap());
        let read_list = consumer.as_ref().unwrap_or(&producer);
        readable(read_list, &used(ResourceType::Srv), &srv, false);
        run(read_list, &slots, &read, 1, Some(&output), (8, 5));
        producer.close();
        if let Some(consumer) = &consumer {
            consumer.close();
        }
        let record_barriers = barriers();
        crate::host::tests::start_native_call_trace();
        let started = std::time::Instant::now();
        match (&consumer, layout) {
            (None, _) => host.execute(&[&producer]),
            (Some(consumer), SplitSeparate) => {
                host.execute(&[&producer]);
                host.execute(&[consumer]);
            }
            (Some(consumer), _) => host.execute(&[&producer, consumer]),
        }
        let host_wall_us = started.elapsed().as_micros();
        let native_call_groups = crate::host::tests::take_native_call_trace();
        let submit_barriers = barriers();
        let expected_calls = layout.native_calls(split);
        assert_eq!(
            native_call_groups.len(),
            expected_calls.len(),
            "{name}: logical fence count"
        );
        for ((calls, waited), &expected) in native_call_groups.iter().zip(expected_calls) {
            assert!(*waited, "{name}: the fence did not complete");
            assert_eq!(
                calls.as_slice(),
                expected,
                "{name}: native command-list counts"
            );
        }
        if matches!(layout, SameList | SplitTail) {
            assert_eq!(record_barriers, [(UAV, SRV)], "{name}: barriers recorded");
            assert_eq!(
                submit_barriers,
                [(COMMON, UAV)],
                "{name}: barriers deferred"
            );
        } else {
            assert!(record_barriers.is_empty(), "{name}: barriers recorded");
            assert_eq!(
                submit_barriers,
                [(COMMON, UAV), (UAV, SRV)],
                "{name}: barriers deferred"
            );
        }

        // SAFETY: the dispatch and host submission have finished. The mapped
        // range has one initialized u32 for each requested texel.
        let got = unsafe { std::slice::from_raw_parts(mapped.cast::<u32>(), words_count) };
        let mut total = Mismatches::default();
        for (i, &z) in SLICES.iter().enumerate() {
            let mut slice = Mismatches::default();
            for y in 0..HEIGHT {
                for x in 0..WIDTH {
                    let at = (i * HEIGHT + y) * WIDTH + x;
                    let observed = got[at];
                    let want = expected(x, y, z);
                    if observed == want {
                        continue;
                    }
                    total.add(observed);
                    slice.add(observed);
                    if total.count() <= 8 {
                        println!(
                            "R16_VOLUME_DIAG_MISMATCH iteration={iteration} layout={name} x={x} y={y} z={z} got={observed} expected={want}"
                        );
                    }
                }
            }
            if slice.count() != 0 {
                println!(
                    "R16_VOLUME_DIAG_SLICE iteration={iteration} layout={name} z={z} untouched={} read_zero={} wrong_nonzero={}",
                    slice.untouched, slice.read_zero, slice.wrong_nonzero
                );
            }
        }
        println!(
            "R16_VOLUME_DIAG iteration={iteration} layout={name} texels={words_count} mismatches={} untouched={} read_zero={} wrong_nonzero={} host_wall_us={host_wall_us} record_barriers={record_barriers:?} submit_barriers={submit_barriers:?} native_call_groups={native_call_groups:?}",
            total.count(),
            total.untouched,
            total.read_zero,
            total.wrong_nonzero
        );
        failed_cases += usize::from(total.count() != 0);
    }
    assert_eq!(failed_cases, 0, "submission layouts with incorrect texels");
}
