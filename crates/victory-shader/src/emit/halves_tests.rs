//! Console wave64 compute programs translated for a host of 32-lane waves
//! (`lower::halves`): what reads within a 32-lane half runs exactly in two
//! host waves of 32; what reads across them passes between the host waves
//! through group-shared memory where every lane of the group runs it once,
//! and is refused with its reason elsewhere.
//!
//! The GPU test runs one program on every hardware adapter: in halves
//! everywhere (`WaveSize(32)`), and also as one wave64 (`WaveSize(64)`)
//! where the adapter has 64-lane waves (AMD), which is the console's own
//! form; the results must be the console's, computed here, in both.

#[cfg(windows)]
use super::host::{Dispatch, Host, Root, Stride, View};

use super::lower;

#[cfg(windows)]
use super::validator;

use crate::native::{
    ComputeIo, HwStage, NativeProgram, ResourceType, RootParam, RootType, StageIo, UsedResource,
};
use crate::{gfx10, io, ir, resource, structure};

const S_ENDPGM: u32 = 0xBF81_0000;

const S_WAITCNT_LGKM: u32 = 0xBF8C_C07F; // s_waitcnt lgkmcnt(0)

/// `s[40:43]`: the raw UAV at table dword 16 (s0 is the table's address).
const PROLOGUE: [u32; 5] = [
    0xBE84_0300, // s_mov_b32 s4, s0
    0xBE85_0380, // s_mov_b32 s5, 0
    0xF408_0A02,
    0xFA00_0040, // s_load_dwordx4 s[40:43], s[4:5], 0x40
    S_WAITCNT_LGKM,
];

/// `buffer_store_dword v<vdata>, v16, s[40:43], 0 offen offset:<offset>`.
const fn store(vdata: u8, offset: u32) -> [u32; 2] {
    [0xE070_1000 | offset, 0x800A_0010 | (vdata as u32) << 8]
}

/// Per thread `l` of a group of 64 (one console wave64), eight dwords at
/// `l * 32`: its thread id, `v_mbcnt` of all lanes (its lane), `l ^ 5`
/// through `ds_swizzle` (xor mode, within 32 lanes), `l ^ 9` through
/// `ds_bpermute_b32` (within the half), `0x777` written by `v_writelane` to
/// lane 40 only, whether the constant mask -4 (every lane but 0 and 1; lane
/// 32 is bit 0 of its high dword) has its lane, and `l ^ 1` through DPP
/// (`quad_perm:[1,0,3,2]`).
fn in_halves() -> Vec<u32> {
    let mut w = PROLOGUE.to_vec();
    w.extend([
        0x3420_0085, // v_lshlrev_b32 v16, 5, v0
        0xD765_0001,
        0x0001_00C1, // v_mbcnt_lo_u32_b32 v1, -1, 0
        0xD766_0001,
        0x0002_02C1, // v_mbcnt_hi_u32_b32 v1, -1, v1
        0xD8D4_141F,
        0x0200_0000, // ds_swizzle_b32 v2, v0 offset:swizzle(BITMASK_PERM xor 5)
        0x3A08_0089, // v_xor_b32 v4, 9, v0
        0x3408_0882, // v_lshlrev_b32 v4, 2, v4
        0xDACC_0000,
        0x0300_0004, // ds_bpermute_b32 v3, v4, v0
        0xBE88_03FF,
        0x0000_0777, // s_mov_b32 s8, 0x777
        0x7E0A_0280, // v_mov_b32 v5, 0
        0xD761_0005,
        0x0001_5008, // v_writelane_b32 v5, s8, 40
        0xBE94_04C4, // s_mov_b64 s[20:21], -4
        0xD501_0006,
        0x0051_0280, // v_cndmask_b32 v6, 0, 1, s[20:21]
        0x7E0E_02FA,
        0xFF00_B100, // v_mov_b32 v7, v0 quad_perm:[1,0,3,2] (DPP)
        S_WAITCNT_LGKM,
    ]);
    for (vdata, slot) in [(0, 0), (1, 1), (2, 2), (3, 3), (5, 4), (6, 5), (7, 6)] {
        w.extend(store(vdata, 4 * slot));
    }
    w.push(S_ENDPGM);
    w
}

/// What the console gives [`in_halves`]'s lane `l`.
#[cfg(windows)]
fn console(l: u32) -> [u32; 8] {
    [
        l,
        l,
        l ^ 5,
        l ^ 9,
        if l == 40 { 0x777 } else { 0 },
        u32::from(l >= 2),
        l ^ 1,
        0,
    ]
}

fn native(words: &[u32], wave64: bool, threads: u32, lds_bytes: u32) -> NativeProgram {
    let range = |kind, table_offset_dwords, dwords| UsedResource {
        kind,
        root: RootType::Table,
        table_offset_dwords,
        lower_bound: 0,
        stride: 0,
        dwords,
        user_data: 0,
        slot: 0,
    };
    NativeProgram {
        stage: HwStage::Cs,
        wave64,
        vgprs: 32,
        layout: vec![RootParam {
            root: RootType::Table,
            table_offset_dwords: 0,
            sgprs: 1,
            base_register: 0,
            user_data: 0,
            slot: 0,
        }],
        used: vec![
            range(ResourceType::Srv, 0, 8),
            range(ResourceType::Srv, 8, 4),
            range(ResourceType::Uav, 16, 4),
        ],
        io: StageIo::Cs(ComputeIo {
            // One user-data SGPR; LDS in blocks of 512 bytes (bits 15-23).
            rsrc2: 1 << 1 | lds_bytes.div_ceil(512) << 15,
            threads: [threads, 1, 1],
        }),
        code: words.iter().flat_map(|w| w.to_le_bytes()).collect(),
    }
}

fn lowered(
    native: &NativeProgram,
    host_wave32: bool,
) -> Result<lower::Lowered, crate::ShaderError> {
    lowered_with(native, host_wave32, false)
}

/// [`lowered`], with LDS synced for narrow host waves where `narrow_waves`
/// (`emit::Options::narrow_waves`, D3DMetal).
fn lowered_with(
    native: &NativeProgram,
    host_wave32: bool,
    narrow_waves: bool,
) -> Result<lower::Lowered, crate::ShaderError> {
    lower_halves(native, host_wave32, narrow_waves, false)
}

fn lower_halves(
    native: &NativeProgram,
    host_wave32: bool,
    narrow_waves: bool,
    literal: bool,
) -> Result<lower::Lowered, crate::ShaderError> {
    lower_halves_limited(native, host_wave32, narrow_waves, literal, false)
}

/// [`lower_halves`]; `limits`: lanes past a dispatch's count of threads
/// leave where the program can (`emit::Options::thread_limits`). With
/// `limits` the program goes through the translation's own entry, where
/// `literal` retries a refusal in whole-wave halves; without, `literal`
/// takes a 64-thread wave64 program straight to the whole-wave translation.
fn lower_halves_limited(
    native: &NativeProgram,
    host_wave32: bool,
    narrow_waves: bool,
    literal: bool,
    limits: bool,
) -> Result<lower::Lowered, crate::ShaderError> {
    let p = ir::build(&native.code, native.wave64, native.vgprs).expect("ir::build");
    let st = structure::structure(&p).expect("structure");
    let res = resource::track(&p, &native.layout, native.stage, Some(&st)).expect("track");
    let io = io::raise(&p, native).expect("io::raise");
    let parts = lower::Parts {
            native,
            p: &p,
            st: &st,
            io: &io,
            res: &res,
            inputs: Vec::new(),
            outputs: Vec::new(),
            prims: Vec::new(),
            mesh_stride: 0,
            view_marks: false,
            atomic_locks: false,
            atomic_lock_rounds: false,
            atomic_umax: false,
            bias_as_gradients: false,
            narrow_waves,
            uav_size_level0: false,
            halves_rendezvous: true,
            host_wave32,
    };
    let threads = match &native.io {
        StageIo::Cs(cs) => cs.threads.iter().product::<u32>(),
        _ => 0,
    };
    if literal && !limits && threads == 64 {
        lower::lower_literal(&parts, None, false)
    } else {
        lower::lower_limited(&parts, None, false, literal, limits)
    }
}

#[cfg(windows)]
fn shader(l: lower::Lowered) -> super::HostShader {
    super::HostShader {
        stage: HwStage::Cs,
        dxil: l.dxil,
        user_data: l.user_data,
        static_samplers: Vec::new(),
        samplers: l.samplers,
        root_cbvs: l.root_cbvs,
        root_srvs: l.root_srvs,
        root_uavs: l.root_uavs,
        root_srv_bytes: l.root_srv_bytes,
        memory: l.memory,
        memory_writes: l.memory_writes,
        view_addresses: l.view_addresses,
        heap: l.heap,
        heap_writes: l.heap_writes,
        heap_index_sources: Vec::new(),
        view_marks: l.view_marks,
        half_views: l.half_views,
    }
}

fn refusal(words: &[u32], threads: u32, lds_bytes: u32) -> String {
    match lowered(&native(words, true, threads, lds_bytes), true) {
        Ok(_) => "translated".into(),
        Err(e) => e.to_string(),
    }
}

/// In halves the program declares waves of 32; without, the console's 64.
#[test]
fn a_wave64_program_reading_lanes_declares_32_lane_waves_in_halves() {
    let n = native(&in_halves(), true, 64, 0);
    let halves = lowered(&n, true).expect("halves");
    let whole = lowered(&n, false).expect("one wave64");
    assert_eq!(halves.wave_size, Some(32));
    assert_eq!(whole.wave_size, Some(64));
    #[cfg(windows)]
    if let Ok(mut v) = validator::Validator::load(validator::Validator::default_path().as_deref()) {
        v.validate(&mut halves.dxil.clone())
            .expect("DXIL validation failed");
    }
    // A wave32 program is the console's either way.
    let n32 = native(&in_halves(), false, 32, 0);
    assert_eq!(lowered(&n32, true).expect("wave32").wave_size, Some(32));
}

/// `s_ff1_i32_b64 s12, exec` at the start: the console wave's first active
/// lane, 0 in every lane.
fn first_active_lane() -> Vec<u32> {
    [
        PROLOGUE.as_slice(),
        &[
            0x3420_0085, // v_lshlrev_b32 v16, 5, v0
            0xBE8C_147E, // s_ff1_i32_b64 s12, exec
            0x7E02_020C, // v_mov_b32 v1, s12
        ],
        &store(1, 0),
        &[S_ENDPGM],
    ]
    .concat()
}

/// The lanes below `limit` (`v_cmp_gt_u32 vcc, limit, v0`) as data: per
/// lane, three dwords at `l * 32`: the console wave's count of them
/// (`s_bcnt1_i32_b64`), their high dword (`vcc_hi`, lanes 32-63), and how
/// many are below the lane (`v_mbcnt` of the mask, as a compaction counts
/// its place).
fn lanes_below(limit: u32) -> Vec<u32> {
    let mut w = PROLOGUE.to_vec();
    w.extend([
        0x3420_0085, // v_lshlrev_b32 v16, 5, v0
        0x7D88_00FF,
        limit, // v_cmp_gt_u32 vcc, limit, v0
        0xBE8C_106A, // s_bcnt1_i32_b64 s12, vcc
        0x7E02_020C, // v_mov_b32 v1, s12
        0x7E04_026B, // v_mov_b32 v2, vcc_hi
        0xD765_0003,
        0x0001_006A, // v_mbcnt_lo_u32_b32 v3, vcc_lo, 0
        0xD766_0003,
        0x0002_066B, // v_mbcnt_hi_u32_b32 v3, vcc_hi, v3
    ]);
    w.extend(store(1, 0));
    w.extend(store(2, 4));
    w.extend(store(3, 8));
    w.push(S_ENDPGM);
    w
}

/// What the console gives [`lanes_below`]`(limit)`'s lane `l` in a group of
/// `threads`.
#[cfg(windows)]
fn console_lanes_below(limit: u32, threads: u32, l: u32) -> [u32; 3] {
    let wave = l & !63;
    let below = |end: u32| limit.min(threads).min(end).saturating_sub(wave);
    let high = below(wave + 64).saturating_sub(32);
    [
        below(wave + 64),
        if high == 32 { u32::MAX } else { (1 << high) - 1 },
        below(l),
    ]
}

/// The lanes below 40, as in [`lanes_below`], counted in a scalar loop
/// that runs twice: the mask is written in a block each lane runs more than
/// once, as often as every other lane of the group.
fn lanes_below_in_a_loop() -> Vec<u32> {
    let mut w = PROLOGUE.to_vec();
    w.extend([
        0x3420_0085, // v_lshlrev_b32 v16, 5, v0
        0xBE94_0380, // s_mov_b32 s20, 0
        0x7D88_00A8, // v_cmp_gt_u32 vcc, 40, v0
        0xBE8C_106A, // s_bcnt1_i32_b64 s12, vcc
        0x8014_8114, // s_add_u32 s20, s20, 1
        0xBF0A_8214, // s_cmp_lt_u32 s20, 2
        0xBF85_FFFB, // s_cbranch_scc1 (back to the v_cmp)
        0x7E02_020C, // v_mov_b32 v1, s12
    ]);
    w.extend(store(1, 0));
    w.push(S_ENDPGM);
    w
}

/// Per thread `l`, two dwords at `l * 32`: `v_readlane_b32` of the thread
/// id at console lane `k` (a constant) and at lane `s` (in an SGPR).
fn lane_reads(k: u32, s: u32) -> Vec<u32> {
    let mut w = PROLOGUE.to_vec();
    w.extend([
        0x3420_0085, // v_lshlrev_b32 v16, 5, v0
        0xD760_000C,
        0x0001_0100 | (128 + k) << 9, // v_readlane_b32 s12, v0, k
        0xBE8D_03FF,
        s, // s_mov_b32 s13, s
        0xD760_000E,
        0x0000_1B00, // v_readlane_b32 s14, v0, s13
        0x7E02_020C, // v_mov_b32 v1, s12
        0x7E04_020E, // v_mov_b32 v2, s14
    ]);
    w.extend(store(1, 0));
    w.extend(store(2, 4));
    w.push(S_ENDPGM);
    w
}

/// How many passes [`in_a_loop`] makes.
#[derive(Clone, Copy)]
enum Count {
    /// Two: a count every host wave computes alike.
    Two,
    /// Up to the thread id of the host wave's first lane
    /// (`v_readfirstlane_b32`), at least one: a count the host waves need
    /// not share.
    FirstLane,
}

/// `body` in a scalar loop (so each lane runs its block more than once),
/// then the scalar `s<result>` stored by every lane.
fn in_a_loop(body: &[u32], result: u32, count: Count) -> Vec<u32> {
    let mut w = PROLOGUE.to_vec();
    w.extend([
        0x3420_0085, // v_lshlrev_b32 v16, 5, v0
        0xD765_0001,
        0x0001_00C1, // v_mbcnt_lo_u32_b32 v1, -1, 0
        0xBE94_0380, // s_mov_b32 s20, 0
    ]);
    let compare = match count {
        Count::Two => 0xBF0A_8214, // s_cmp_lt_u32 s20, 2
        Count::FirstLane => {
            w.push(0x7E2A_0500); // v_readfirstlane_b32 s21, v0
            0xBF0A_1514 // s_cmp_lt_u32 s20, s21
        }
    };
    w.extend(body);
    let back = -(body.len() as i32 + 3);
    w.extend([
        0x8014_8114,                          // s_add_u32 s20, s20, 1
        compare,                              // s_cmp_lt_u32 s20, <count>
        0xBF85_0000 | (back as u32 & 0xFFFF), // s_cbranch_scc1 (back to the body)
        0x7E02_0200 | result,                 // v_mov_b32 v1, s<result>
    ]);
    w.extend(store(1, 0));
    w.push(S_ENDPGM);
    w
}

/// VOP2 opcodes of the operations [`reduction`] takes.
const V_ADD_F32: u32 = 0x03;

const V_MIN_U32: u32 = 0x13;

const V_MAX_U32: u32 = 0x14;

/// `v_mov_b32 v3, v0`.
const V_MOV_V3_V0: u32 = 0x7E06_0300;

/// The console compiler's whole-wave reduction of `v3` (`input`, from the
/// thread id) by the VOP2 operation `op`, with the inline constant `ident`
/// in inactive lanes: within rows of 16 by DPP, across rows by
/// `v_permlanex16`, across the halves by `v_readlane` of lane 32, and read
/// back from lane 0. Per thread, one dword at `l * 32`: its console wave's
/// result.
fn reduction(op: u32, ident: u32, input: u32) -> Vec<u32> {
    let vop2 = |word: u32| word & !(0x3F << 25) | op << 25;
    let mut w = PROLOGUE.to_vec();
    w.extend([
        0x3420_0085, // v_lshlrev_b32 v16, 5, v0
        input,       // v3 from v0
        0xBE8C_25C1, // s_or_saveexec_b64 s[12:13], -1
        0xD501_0001,
        0x0032_0600 | ident, // v_cndmask_b32 v1, ident, v3, s[12:13]
    ]);
    for control in [0x161, 0x162, 0x164, 0x168] {
        w.extend([
            vop2(0x0602_02FA),
            0xFF00_0001 | control << 8, // op v1, v1, v1 row_xmask
        ]);
    }
    w.extend([
        0xD778_0002,
        0x0201_0101,       // v_permlanex16_b32 v2, v1, 0, 0
        vop2(0x0602_0302), // op v1, v2, v1
        0xD760_000E,
        0x0001_4101,       // v_readlane_b32 s14, v1, 32
        vop2(0x0602_020E), // op v1, s14, v1
        0xD760_000F,
        0x0001_0101, // v_readlane_b32 s15, v1, 0
        0xBEFE_040C, // s_mov_b64 exec, s[12:13]
        0x7E02_020F, // v_mov_b32 v1, s15
    ]);
    w.extend(store(1, 0));
    w.push(S_ENDPGM);
    w
}

/// A console wave32 waterfall loop over the low two bits of the thread id
/// (`s_andn2_wrexec_b32` takes each value's lanes out of the pending mask
/// and sets SCC to whether lanes remain), then the compiler's select on
/// that SCC after the loop: `s_cselect_b32` of EXEC or 0, OR-ed with the
/// lanes below 20. The loop leaves when no lanes remain, so SCC is 0 there
/// and the mask is the compare's. Per thread, one dword at `l * 32`: 9
/// below 20, else 7.
fn mask_selected_after_a_waterfall() -> Vec<u32> {
    let mut w = PROLOGUE.to_vec();
    w.extend([
        0x3420_0085, // v_lshlrev_b32 v16, 5, v0
        0x360A_0083, // v_and_b32 v5, 3, v0
        0xBE83_037E, // s_mov_b32 s3, exec_lo
        0xBE86_037E, // s_mov_b32 s6, exec_lo
        0x7E0E_0505, // v_readfirstlane_b32 s7, v5
        0x7DA4_0A07, // v_cmpx_eq_u32 s7, v5
        0xBE86_4706, // s_andn2_wrexec_b32 s6, s6
        0xBF89_FFFC, // s_cbranch_execnz (back to the v_readfirstlane)
        0xBEFE_0303, // s_mov_b32 exec_lo, s3
        0x7D88_0094, // v_cmp_gt_u32 vcc_lo, 20, v0
        0x8504_807E, // s_cselect_b32 s4, exec_lo, 0
        0x8804_046A, // s_or_b32 s4, vcc_lo, s4
        0xD501_0001,
        0x0011_1287, // v_cndmask_b32 v1, 7, 9, s4
    ]);
    w.extend(store(1, 0));
    w.push(S_ENDPGM);
    w
}

/// The console's low-half f16 whole-wave64 sum of `v0`: DPP within rows,
/// `v_permlanex16` across rows, then lanes 31 and 63 (each 32-lane half's
/// sum) read and added. Its high half is kept from lane 31, as the
/// compiler's `v_add_f16` lowering does.
fn half_reduction() -> Vec<u32> {
    let vop2 = |src0: u16| (0x32u32 << 25) | (1 << 17) | (1 << 9) | u32::from(src0);
    let dpp16 = |ctrl: u16| (0xFu32 << 28) | (0xFu32 << 24) | (u32::from(ctrl) << 8) | 1;
    let vop3 = |dst: u8, op: u16, src0: u16, src1: u16, src2: u16| {
        [
            0xD400_0000 | u32::from(op) << 16 | u32::from(dst),
            u32::from(src0) | u32::from(src1) << 9 | u32::from(src2) << 18,
        ]
    };
    let mut w = PROLOGUE.to_vec();
    w.extend([
        0x3420_0085, // v_lshlrev_b32 v16, 5, v0
        0x7E06_0300, // v_mov_b32 v3, v0
        0xBE8C_25C1, // s_or_saveexec_b64 s[12:13], -1
        0xD501_0001,
        0x0032_0680, // v_cndmask_b32 v1, 0, v3, s[12:13]
        vop2(128),   // v_add_f16 v1, 0, v1
    ]);
    for ctrl in [0x00B1, 0x004E, 0x0164, 0x0168] {
        w.push(vop2(0x00FA));
        w.push(dpp16(ctrl));
    }
    w.extend(vop3(2, 0x378, 257, 128, 128)); // v_permlanex16_b32 v2, v1, 0, 0
    w.push(vop2(258)); // v_add_f16 v1, v2, v1
    // Lane 31 holds the sum of lanes 0-31, lane 63 that of lanes 32-63.
    w.extend(vop3(1, 0x360, 257, 159, 0)); // v_readlane_b32 s1, v1, 31
    w.extend(vop3(2, 0x360, 257, 191, 0)); // v_readlane_b32 s2, v1, 63
    w.push(0x7E06_0202); // v_mov_b32 v3, s2
    w.push((0x32 << 25) | (1 << 17) | (3 << 9) | 1); // v_add_f16 v1, s1, v3
    w.extend(vop3(1, 0x360, 257, 159, 0)); // v_readlane_b32 s1, v1, 31
    w.extend([
        0xBEFE_040C, // s_mov_b64 exec, s[12:13]
        0x7E02_0201, // v_mov_b32 v1, s1
    ]);
    w.extend(store(1, 0));
    w.push(S_ENDPGM);
    w
}

/// What the console gives [`reduction`] of `f` (folding the inputs of a
/// console wave's lanes) in lane `l` of a group of `threads`.
#[cfg(windows)]
fn console_reduction(
    threads: u32,
    l: u32,
    f: impl Fn(u32, u32) -> u32,
    input: impl Fn(u32) -> u32,
) -> u32 {
    let wave = l & !63;
    (wave + 1..(wave + 64).min(threads)).fold(input(wave), |a, t| f(a, input(t)))
}

/// The console compiler's exclusive prefix sum of the thread id over the
/// wave64 (`WavePrefixSum`): inclusive within rows of 16 by DPP, rows
/// joined by `v_readlane` of each row's last lane under EXEC of the rows
/// after it, shifted one lane on (the first lane of each row from the
/// previous row's last), lane 0 the identity. One dword per thread at
/// `l * 32`.
fn exclusive_prefix() -> Vec<u32> {
    const EXEC: u32 = 126;
    let sop1 = |sdst: u32, op: u32, ssrc0: u32| 0xBE80_0000 | sdst << 16 | op << 8 | ssrc0;
    let vop1 = |vdst: u32, src0: u32| 0x7E00_0200 | vdst << 17 | src0;
    let add = |src0: u32| 0x4A02_0200 | src0; // v_add_nc_u32 v1, src0, v1
    let dpp = |ctrl: u32| 0xFF00_0001 | ctrl << 8; // of v1, row and bank masks 0xf
    let readlane = |lane: u32| [0xD760_0002, 0x0000_0101 | (128 + lane) << 9]; // s2, v1
    let mut w = PROLOGUE.to_vec();
    w.extend([
        0x3420_0085,        // v_lshlrev_b32 v16, 5, v0
        sop1(0, 0x25, 193), // s_or_saveexec_b64 s[0:1], -1
        0xD501_0001,
        0x0002_0080, // v_cndmask_b32 v1, 0, v0, s[0:1]
    ]);
    for shift in [1, 2, 4, 8] {
        w.extend([add(0xFA), dpp(0x110 + shift)]); // v_add_nc_u32 v1, v1, v1 row_shr
    }
    // Each row's sum so far into the rows after it.
    for (last, exec_lo, exec_hi) in [
        (15, 0xFFFF_0000, 0),
        (31, 0, 0x0000_FFFF),
        (47, 0, 0xFFFF_0000),
    ] {
        w.extend(readlane(last));
        w.extend([
            sop1(EXEC, 0x03, 255),
            exec_lo,
            sop1(EXEC + 1, 0x03, 255),
            exec_hi,
        ]);
        w.push(add(2)); // v_add_nc_u32 v1, s2, v1
    }
    w.extend([
        sop1(EXEC, 0x04, 193), // s_mov_b64 exec, -1
        vop1(3, 257),          // v_mov_b32 v3, v1
        vop1(3, 0xFA),
        dpp(0x111), // v_mov_b32 v3, v1 row_shr:1
    ]);
    // The first lane of each row from the last of the row before.
    for (last, exec_lo, exec_hi) in [(15, 0x0001_0000, 0), (31, 0, 1), (47, 0, 0x0001_0000)] {
        w.extend(readlane(last));
        w.extend([
            sop1(EXEC, 0x03, 255),
            exec_lo,
            sop1(EXEC + 1, 0x03, 255),
            exec_hi,
        ]);
        w.push(vop1(3, 2)); // v_mov_b32 v3, s2
    }
    w.extend([
        sop1(EXEC, 0x04, 193), // s_mov_b64 exec, -1
        vop1(2, 128),          // v_mov_b32 v2, 0
        sop1(EXEC, 0x04, 194), // s_mov_b64 exec, -2
        vop1(2, 259),          // v_mov_b32 v2, v3
        sop1(EXEC, 0x04, 0),   // s_mov_b64 exec, s[0:1]
    ]);
    w.extend(store(2, 0));
    w.push(S_ENDPGM);
    w
}

/// A lane mask read as data in a group of 33 to 64 lanes or more holds
/// lanes of both host waves: they pass their halves to each other where
/// every lane of the group runs the block writing it together (once, or in
/// a loop on a count every host wave computes alike), and it is refused in
/// a loop counted to a lane's value. In a group of 32 or fewer, one host
/// wave holds the whole mask.
#[test]
fn lane_masks_read_as_data_pass_between_halves() {
    assert_eq!(refusal(&first_active_lane(), 32, 0), "translated");
    assert_eq!(refusal(&first_active_lane(), 64, 0), "translated");
    for threads in [64, 96, 128] {
        assert_eq!(refusal(&lanes_below(40), threads, 0), "translated");
        assert_eq!(refusal(&lanes_below_in_a_loop(), threads, 0), "translated");
    }
    // v_cmp_gt_u32 vcc, 40, v0; s_bcnt1_i32_b64 s12, vcc
    let counted = in_a_loop(&[0x7D88_00A8, 0xBE8C_106A], 12, Count::FirstLane);
    let why = refusal(&counted, 64, 0);
    assert!(why.contains("a lane mask read as data (wave64 on a 32-lane host)"), "{why}");
    assert_eq!(refusal(&counted, 32, 0), "translated");
    // For 64-lane hosts it is read in one wave, as before.
    assert!(lowered(&native(&counted, true, 64, 0), false).is_ok());

    let halves = lowered(&native(&lanes_below(40), true, 64, 0), true).expect("halves");
    assert_eq!(halves.wave_size, Some(32));
    #[cfg(windows)]
    if let Ok(mut v) = validator::Validator::load(validator::Validator::default_path().as_deref()) {
        v.validate(&mut halves.dxil.clone())
            .expect("DXIL validation failed");
    }
}

/// [`lanes_below`] and [`first_active_lane`] on every hardware adapter give
/// the console's masks: in one console wave of 64 (both halves' lanes), in
/// two (one mask each), and in a group of 96, whose second console wave has
/// no upper half (a high dword of 0).
#[test]
#[cfg(windows)]
fn lane_masks_passed_between_halves_match_the_console_on_every_gpu() {
    let mut failed = differences(&first_active_lane(), 64, 1, &[0; 64]);
    for (limit, threads) in [(40, 64), (100, 128), (80, 96), (96, 96)] {
        let want: Vec<u32> = (0..threads)
            .flat_map(|l| console_lanes_below(limit, threads, l))
            .collect();
        failed.extend(differences(&lanes_below(limit), threads, 3, &want));
    }
    // Counted in a loop: the count of lanes below 40, each pass.
    for threads in [64, 96, 128] {
        let want: Vec<u32> = (0..threads)
            .map(|l| console_lanes_below(40, threads, l)[0])
            .collect();
        failed.extend(differences(&lanes_below_in_a_loop(), threads, 1, &want));
    }
    assert!(failed.is_empty(), "{}", failed.join("\n"));
}

/// Whole-wave reductions and prefix sums pass between the halves where
/// every lane of the group runs them together (once, or in a loop on a
/// count every host wave computes alike); in a group of 32 or fewer, one
/// host wave holds the whole console wave.
#[test]
fn reductions_and_prefixes_pass_between_halves() {
    let sum = reduction(V_ADD_F32, 0x80, 0x7E06_0D00);
    for threads in [32, 64, 96, 128] {
        assert_eq!(refusal(&sum, threads, 0), "translated");
        assert_eq!(refusal(&exclusive_prefix(), threads, 0), "translated");
    }
    let body = &sum[PROLOGUE.len() + 1..sum.len() - 3];
    for threads in [64, 96, 128] {
        assert_eq!(refusal(&in_a_loop(body, 15, Count::Two), threads, 0), "translated");
    }
    let looped = in_a_loop(body, 15, Count::FirstLane);
    let why = refusal(&looped, 64, 0);
    assert!(
        why.contains("a wave reduction over both 32-lane halves"),
        "{why}"
    );
    assert_eq!(refusal(&looped, 32, 0), "translated");

    for words in [sum, exclusive_prefix()] {
        let halves = lowered(&native(&words, true, 96, 0), true).expect("halves");
        assert_eq!(halves.wave_size, Some(32));
        #[cfg(windows)]
        if let Ok(mut v) =
            validator::Validator::load(validator::Validator::default_path().as_deref())
        {
            v.validate(&mut halves.dxil.clone())
                .expect("DXIL validation failed");
        }
    }

    let half_sum = half_reduction();
    let p = ir::build(&native(&half_sum, true, 96, 0).code, true, 32).expect("f16 IR");
    assert!(
        super::wave::sections(&p, 0).values().any(|section| {
            section
                .shapes
                .values()
                .any(|shape| matches!(shape, super::wave::Shape::Half(..)))
        }),
        "f16 reduction fixture was not recognized"
    );
    for threads in [32, 64, 96, 128] {
        assert_eq!(refusal(&half_sum, threads, 0), "translated");
    }
    let halves = lowered(&native(&half_sum, true, 96, 0), true).expect("f16 halves");
    assert_eq!(halves.wave_size, Some(32));
}

/// [`reduction`]s and [`exclusive_prefix`] on every hardware adapter give
/// the console's results: in a group of 32 (one host wave), of 64 (one
/// console wave in two halves), of 128 (two), and of 96, whose second
/// console wave has no upper half.
#[test]
#[cfg(windows)]
fn reductions_and_prefixes_across_halves_match_the_console_on_every_gpu() {
    let mut failed = Vec::new();
    for threads in [32, 64, 96, 128] {
        let want: Vec<u32> = (0..threads)
            .map(|l| {
                let sum = console_reduction(threads, l, |a, b| a + b, |t| t);
                (sum as f32).to_bits()
            })
            .collect();
        failed.extend(differences(
            &reduction(V_ADD_F32, 0x80, 0x7E06_0D00),
            threads,
            1,
            &want,
        ));
        let want: Vec<u32> = (0..threads)
            .map(|l| console_reduction(threads, l, u32::min, |t| t ^ 0x2A))
            .collect();
        let xor = 0x3A06_00AA; // v_xor_b32 v3, 42, v0
        failed.extend(differences(
            &reduction(V_MIN_U32, 0xC1, xor),
            threads,
            1,
            &want,
        ));
        let want: Vec<u32> = (0..threads)
            .map(|l| console_reduction(threads, l, u32::max, |t| t ^ 0x2A))
            .collect();
        failed.extend(differences(
            &reduction(V_MAX_U32, 0x80, xor),
            threads,
            1,
            &want,
        ));
        let want: Vec<u32> = (0..threads)
            .map(|l| {
                let (wave, k) = (l & !63, l & 63);
                wave * k + k * k.saturating_sub(1) / 2
            })
            .collect();
        failed.extend(differences(&exclusive_prefix(), threads, 1, &want));
    }
    // The sum in a loop: the same each pass.
    let sum = reduction(V_ADD_F32, 0x80, 0x7E06_0D00);
    let looped = in_a_loop(&sum[PROLOGUE.len() + 1..sum.len() - 3], 15, Count::Two);
    for threads in [64, 96, 128] {
        let want: Vec<u32> = (0..threads)
            .map(|l| (console_reduction(threads, l, |a, b| a + b, |t| t) as f32).to_bits())
            .collect();
        failed.extend(differences(&looped, threads, 1, &want));
    }
    // Each input lane is a positive f16 subnormal whose bit pattern is its
    // lane number. These sums are exactly representable, so per-step f16
    // arithmetic and the translator's f32 accumulation round to the same
    // result. The 96-thread case also checks a missing upper half.
    let half_sum = half_reduction();
    for threads in [32, 64, 96, 128] {
        let want: Vec<u32> = (0..threads)
            .map(|l| match (threads, l & !63) {
                (32, _) => 0x01F0,
                (_, 0) => 0x07E0,
                (96, 64) => 0x08F8,
                (128, 64) => 0x0DF8,
                _ => unreachable!(),
            })
            .collect();
        failed.extend(differences(&half_sum, threads, 1, &want));
    }
    assert!(failed.is_empty(), "{}", failed.join("\n"));
}

/// [`mask_selected_after_a_waterfall`]: the SCC the waterfall loop's exit
/// shows is 0, so the select is the compare's mask and no lane mask is
/// read as data.
#[test]
fn a_mask_selected_on_a_waterfall_exits_scc_translates() {
    let words = mask_selected_after_a_waterfall();
    let code: Vec<u8> = words.iter().flat_map(|w| w.to_le_bytes()).collect();
    let all = gfx10::decode(&code)
        .expect("decodes")
        .iter()
        .map(|i| i.text(false))
        .collect::<Vec<_>>()
        .join("\n");
    for want in [
        "v_and_b32 v5, 3, v0",
        "s_mov_b32 s3, exec_lo",
        "s_mov_b32 s6, exec_lo",
        "v_readfirstlane_b32 s7, v5",
        "v_cmpx_eq_u32 exec_lo, s7, v5",
        "s_andn2_wrexec_b32 s6, s6",
        "s_cbranch_execnz",
        "s_mov_b32 exec_lo, s3",
        "v_cmp_gt_u32 vcc_lo, 20, v0",
        "s_cselect_b32 s4, exec_lo, 0",
        "s_or_b32 s4, vcc_lo, s4",
        "v_cndmask_b32 v1, 7, 9, s4",
    ] {
        assert!(all.contains(want), "{want:?} not in:\n{all}");
    }
    let n = native(&words, false, 32, 0);
    lowered(&n, false).expect("wave32");
}

/// [`mask_selected_after_a_waterfall`] on every hardware adapter gives the
/// console's values.
#[test]
#[cfg(windows)]
fn a_mask_selected_on_a_waterfall_exits_scc_matches_the_console_on_every_gpu() {
    let want: Vec<u32> = (0..32).map(|l| if l < 20 { 9 } else { 7 }).collect();
    let words = mask_selected_after_a_waterfall();
    let failed = differences_in(false, &words, 32, 1, &want, 0, false, false);
    assert!(failed.is_empty(), "{}", failed.join("\n"));
}

/// Lane reads and first-lane reads that may be in the other half pass
/// between the halves where every lane of the group runs them together
/// (once, or in a loop on a count every host wave computes alike), and are
/// refused with a reason naming the host's waves in a loop counted to a
/// lane's value; the same programs translate for 64-lane hosts.
#[test]
fn lane_reads_pass_between_halves() {
    for threads in [32, 64, 96, 128] {
        assert_eq!(refusal(&lane_reads(40, 33), threads, 0), "translated");
        assert_eq!(
            refusal(&first_lane_of_the_thread_id(), threads, 0),
            "translated"
        );
    }
    assert_eq!(
        refusal(&first_lane_of_a_buffer_load(0), 64, 0),
        "translated"
    );

    let readlane = [0xD760_000C, 0x0001_5100]; // v_readlane_b32 s12, v0, 40
    let first = [0x7E18_0500]; // v_readfirstlane_b32 s12, v0
    for threads in [64, 96, 128] {
        for body in [&readlane[..], &first] {
            let words = in_a_loop(body, 12, Count::Two);
            assert_eq!(refusal(&words, threads, 0), "translated");
        }
    }

    let readlane = in_a_loop(&readlane, 12, Count::FirstLane);
    let why = refusal(&readlane, 64, 0);
    assert!(
        why.contains("a lane read that may cross 32-lane halves (wave64 on a 32-lane host)"),
        "{why}"
    );
    assert_eq!(refusal(&readlane, 32, 0), "translated");
    assert!(lowered(&native(&readlane, true, 64, 0), false).is_ok());

    let first = in_a_loop(&first, 12, Count::FirstLane);
    let why = refusal(&first, 64, 0);
    assert!(
        why.contains("a first-lane read of a value that may differ between 32-lane halves"),
        "{why}"
    );
    assert_eq!(refusal(&first, 32, 0), "translated");
    assert!(lowered(&native(&first, true, 64, 0), false).is_ok());

    // A first-lane read alone needs no console lanes: the host wave's
    // first lane, in halves or not.
    let first_alone = [
        PROLOGUE.as_slice(),
        &[
            0x3420_0085, // v_lshlrev_b32 v16, 5, v0
            0x7E18_0500, // v_readfirstlane_b32 s12, v0
            0x7E02_020C, // v_mov_b32 v1, s12
        ],
        &store(1, 0),
        &[S_ENDPGM],
    ]
    .concat();
    assert_eq!(refusal(&first_alone, 64, 0), "translated");
    // A buffer dword loaded at an offset the same in every lane, through a
    // scalar descriptor, with a store present (`store(5, 0)` below): not
    // proven the same in every lane, so the first-lane read passes the
    // value between the halves in the entry block, where every lane of the
    // group runs it together.
    assert_eq!(
        refusal(&first_lane_of_a_buffer_load(3), 64, 0),
        "translated"
    );

    for words in [lane_reads(20, 5), first_lane_of_the_thread_id()] {
        let halves = lowered(&native(&words, true, 96, 0), true).expect("halves");
        assert_eq!(halves.wave_size, Some(32));
        #[cfg(windows)]
        if let Ok(mut v) =
            validator::Validator::load(validator::Validator::default_path().as_deref())
        {
            v.validate(&mut halves.dxil.clone())
                .expect("DXIL validation failed");
        }
    }
}

/// [`lane_reads`] and [`first_lane_of_the_thread_id`] on every hardware
/// adapter give the console's lanes: in a group of one console wave, of
/// two, of 96 (the second console wave has no upper half), and of 32.
#[test]
#[cfg(windows)]
fn lane_reads_across_halves_match_the_console_on_every_gpu() {
    let mut failed = Vec::new();
    for (k, s, threads) in [
        (40, 33, 64),
        (40, 63, 128),
        (7, 50, 128),
        (20, 5, 96),
        (20, 5, 32),
    ] {
        let want: Vec<u32> = (0..threads)
            .flat_map(|l| [(l & !63) + k, (l & !63) + s])
            .collect();
        failed.extend(differences(&lane_reads(k, s), threads, 2, &want));
    }
    for threads in [64, 96, 128] {
        let want: Vec<u32> = (0..threads).map(|l| l & !63).collect();
        failed.extend(differences(
            &first_lane_of_the_thread_id(),
            threads,
            1,
            &want,
        ));
        // In a loop, the same each pass.
        let first = in_a_loop(&[0x7E18_0500], 12, Count::Two); // v_readfirstlane_b32 s12, v0
        failed.extend(differences(&first, threads, 1, &want));
    }
    for threads in [64, 128] {
        let want: Vec<u32> = (0..threads).map(|l| (l & !63) + 40).collect();
        // v_readlane_b32 s12, v0, 40
        let readlane = in_a_loop(&[0xD760_000C, 0x0001_5100], 12, Count::Two);
        failed.extend(differences(&readlane, threads, 1, &want));
    }
    assert!(failed.is_empty(), "{}", failed.join("\n"));
}

/// LDS shared by a group that is one console wave64 (no barriers between
/// its lanes) cannot be split into two host waves; one of 32 threads or
/// fewer is one host wave, and a group of several console waves keeps the
/// console's barriers.
#[test]
fn lds_of_a_one_wave64_group_is_refused_in_halves() {
    let lds = [
        PROLOGUE.as_slice(),
        &[
            0x3420_0085, // v_lshlrev_b32 v16, 5, v0
            0x3404_0082, // v_lshlrev_b32 v2, 2, v0
            0xD765_0001,
            0x0001_00C1, // v_mbcnt_lo_u32_b32 v1, -1, 0
            0xD834_0000,
            0x0000_0102, // ds_write_b32 v2, v1
            S_WAITCNT_LGKM,
            0xD8D8_0000,
            0x0300_0002, // ds_read_b32 v3, v2
            S_WAITCNT_LGKM,
        ],
        &store(3, 0),
        &[S_ENDPGM],
    ]
    .concat();
    let why = refusal(&lds, 64, 256);
    assert!(
        why.contains("LDS one console wave64 shares without barriers"),
        "{why}"
    );
    assert_eq!(refusal(&lds, 32, 256), "translated");
    assert_eq!(refusal(&lds, 128, 512), "translated");
}

/// Runs `words`, one group of `threads` of a wave64 program whose lanes store
/// `per_lane` dwords at `l * 32`, on every hardware adapter: in halves, and
/// as one wave64 where the adapter runs 64-lane waves. A line per lane whose
/// dwords differ from `want` (the console's, `per_lane` per lane).
#[cfg(windows)]
fn differences(words: &[u32], threads: u32, per_lane: usize, want: &[u32]) -> Vec<String> {
    differences_with(words, threads, per_lane, want, 0, false)
}

/// [`differences`] for a program with `lds_bytes` of LDS, lowered with
/// `narrow_waves` ([`lowered_with`]).
#[cfg(windows)]
fn differences_with(
    words: &[u32],
    threads: u32,
    per_lane: usize,
    want: &[u32],
    lds_bytes: u32,
    narrow_waves: bool,
) -> Vec<String> {
    differences_in(
        true,
        words,
        threads,
        per_lane,
        want,
        lds_bytes,
        narrow_waves,
        false,
    )
}

/// [`differences_with`] for a program of console waves of 64 lanes
/// (`wave64`) or of 32, which a host of any wave size runs as it is.
#[cfg(windows)]
#[allow(clippy::too_many_arguments)]
fn differences_in(
    wave64: bool,
    words: &[u32],
    threads: u32,
    per_lane: usize,
    want: &[u32],
    lds_bytes: u32,
    narrow_waves: bool,
    literal: bool,
) -> Vec<String> {
    differences_with_buffer(
        wave64,
        words,
        threads,
        per_lane,
        want,
        lds_bytes,
        narrow_waves,
        literal,
        &[0; 4],
    )
}

/// [`differences_in`] with custom contents for the structured SRV buffer at
/// table dword 8.
#[allow(clippy::too_many_arguments)]
#[cfg(windows)]
fn differences_with_buffer(
    wave64: bool,
    words: &[u32],
    threads: u32,
    per_lane: usize,
    want: &[u32],
    lds_bytes: u32,
    narrow_waves: bool,
    literal: bool,
    buf_data: &[u32],
) -> Vec<String> {
    let n = native(words, wave64, threads, lds_bytes);
    let forms = if literal {
        vec![(
            "literal halves",
            shader(lower_halves(&n, true, narrow_waves, true).expect("literal halves")),
        )]
    } else if wave64 {
        vec![
            (
                "halves",
                shader(lower_halves(&n, true, narrow_waves, literal).expect("halves")),
            ),
            (
                "wave64",
                shader(lowered_with(&n, false, narrow_waves).expect("one wave64")),
            ),
        ]
    } else {
        vec![(
            "wave32",
            shader(lowered_with(&n, false, narrow_waves).expect("one wave32")),
        )]
    };
    let views = [
        View::Texture(1, 1, &[[0.0; 4]]),
        View::Buffer {
            data: buf_data,
            stride: Stride::Structured(16),
        },
        View::RawUav(threads * 8),
    ];
    let _one = super::valu_tests::ONE_AT_A_TIME
        .lock()
        .unwrap_or_else(|e| e.into_inner());
    let hosts = Host::each_hardware();
    if hosts.is_empty() {
        eprintln!("no SM 6.6 hardware device; skipped");
    }
    let mut failed = Vec::new();
    for (adapter, host) in &hosts {
        let wave32 = host.wave32().unwrap_or(false);
        for (form, shader) in &forms {
            let label = format!("{adapter} ({form})");
            // A host of 32-lane waves only ever runs the halves. NVIDIA
            // refuses the console's own form; D3DMetal makes its pipeline
            // and then runs nothing.
            if *form == "wave64" && wave32 {
                eprintln!("{label}: one wave64 on a host of 32-lane waves; skipped");
                continue;
            }
            let Ok((root, pso)) = host.compute_pipeline(shader) else {
                panic!("{label}: no pipeline");
            };
            let ud = vec![0u32; shader.user_data_dwords().max(1) as usize];
            let out = host
                .dispatch(
                    &root,
                    &pso,
                    &Dispatch {
                        roots: vec![(0, Root::Constants(ud))],
                        views: &views,
                        samplers: &[],
                        groups: [1, 1, 1],
                    },
                )
                .unwrap_or_else(|e| panic!("{label}: dispatch failed: {e}"));
            let wrong: Vec<String> = out[0]
                .chunks(8)
                .zip(want.chunks(per_lane))
                .enumerate()
                .filter(|(_, (got, want))| got[..per_lane] != **want)
                .map(|(l, (got, want))| {
                    format!(
                        "{label}: lane {l}: {:?}, the console {want:?}",
                        &got[..per_lane]
                    )
                })
                .collect();
            eprintln!(
                "{label}: {} lanes of {threads} differ from the console",
                wrong.len()
            );
            failed.extend(wrong);
        }
    }
    failed
}

/// A DPP row mask names the console's rows 0-3: with `row_mask:0xc` only
/// rows 2 and 3 (lanes 32-63) take `l ^ 1`, rows 0 and 1 keep -1.
fn dpp_row_mask() -> Vec<u32> {
    let mut w = PROLOGUE.to_vec();
    w.extend([
        0x3420_0085, // v_lshlrev_b32 v16, 5, v0
        0x7E0E_02C1, // v_mov_b32 v7, -1
        0x7E0E_02FA,
        0xCF00_B100, // v_mov_b32 v7, v0 quad_perm:[1,0,3,2] row_mask:0xc (DPP)
    ]);
    w.extend(store(7, 0));
    w.push(S_ENDPGM);
    w
}

#[test]
#[cfg(windows)]
fn dpp_row_masks_name_the_console_rows_on_every_gpu() {
    let want: Vec<u32> = (0..64)
        .map(|l| if l >= 32 { l ^ 1 } else { u32::MAX })
        .collect();
    let failed = differences(&dpp_row_mask(), 64, 1, &want);
    assert!(failed.is_empty(), "{}", failed.join("\n"));
}

/// `v_readfirstlane_b32` of the thread id, with console lanes (`v_mbcnt`):
/// the console's first lane is lane 0 of the wave.
fn first_lane_of_the_thread_id() -> Vec<u32> {
    let mut w = PROLOGUE.to_vec();
    w.extend([
        0x3420_0085, // v_lshlrev_b32 v16, 5, v0
        0xD765_0001,
        0x0001_00C1, // v_mbcnt_lo_u32_b32 v1, -1, 0
        0x7E18_0500, // v_readfirstlane_b32 s12, v0
        0x7E04_020C, // v_mov_b32 v2, s12
    ]);
    w.extend(store(2, 0));
    w.push(S_ENDPGM);
    w
}

/// `v_readfirstlane_b32` of a dword loaded from the buffer at table dword 8,
/// at the byte offset in `v<offset>`: `v3` holds 4 in every lane, `v0` the
/// thread id.
fn first_lane_of_a_buffer_load(offset: u32) -> Vec<u32> {
    let mut w = PROLOGUE.to_vec();
    w.extend([
        0xF408_0B02,
        0xFA00_0020, // s_load_dwordx4 s[44:47], s[4:5], 0x20
        S_WAITCNT_LGKM,
        0x3420_0085, // v_lshlrev_b32 v16, 5, v0
        0xD765_0001,
        0x0001_00C1, // v_mbcnt_lo_u32_b32 v1, -1, 0
        0x7E06_0284, // v_mov_b32 v3, 4
        0xE030_1000,
        0x800B_0400 | offset, // buffer_load_dword v4, v<offset>, s[44:47], 0 offen
        0xBF8C_3F70,         // s_waitcnt vmcnt(0)
        0x7E18_0504,         // v_readfirstlane_b32 s12, v4
        0x7E0A_020C,         // v_mov_b32 v5, s12
    ]);
    w.extend(store(5, 0));
    w.push(S_ENDPGM);
    w
}

/// Per thread `l` of a group of 128 (two console wave64s), two dwords at
/// `l * 32`: the first lane of `l >> 6` (one value per console wave, the
/// same in both its halves) and the lane `v_mbcnt` counts.
fn first_lane_alike() -> Vec<u32> {
    let mut w = PROLOGUE.to_vec();
    w.extend([
        0x3420_0085, // v_lshlrev_b32 v16, 5, v0
        0xD765_0001,
        0x0001_00C1, // v_mbcnt_lo_u32_b32 v1, -1, 0
        0xD766_0001,
        0x0002_02C1, // v_mbcnt_hi_u32_b32 v1, -1, v1
        0x2C04_0086, // v_lshrrev_b32 v2, 6, v0
        0x7E18_0502, // v_readfirstlane_b32 s12, v2
        0x7E06_020C, // v_mov_b32 v3, s12
    ]);
    w.extend(store(3, 0));
    w.extend(store(1, 4));
    w.push(S_ENDPGM);
    w
}

/// First-lane reads kept in halves give the console's value on every
/// hardware adapter: the thread id's in a group of 32 (one host wave holds
/// the whole console wave), and one value per console wave in a group of
/// 128, where each half reads its console wave's.
#[test]
#[cfg(windows)]
fn first_lanes_kept_in_halves_match_the_console_on_every_gpu() {
    let mut failed = differences(&first_lane_of_the_thread_id(), 32, 1, &[0; 32]);
    let want: Vec<u32> = (0..128).flat_map(|l| [l >> 6, l & 63]).collect();
    failed.extend(differences(&first_lane_alike(), 128, 2, &want));
    assert!(failed.is_empty(), "{}", failed.join("\n"));
}

/// [`in_halves`] on every hardware adapter: in halves, and as one wave64
/// where the adapter runs 64-lane waves; both give the console's results.
#[test]
#[cfg(windows)]
fn wave64_programs_in_halves_match_the_console_on_every_gpu() {
    let want: Vec<u32> = (0..64).flat_map(console).collect();
    let failed = differences(&in_halves(), 64, 8, &want);
    assert!(
        failed.is_empty(),
        "{}",
        failed.join(
            "
"
        )
    );
}

/// Each lane of a one-wave group of 64 writes its `v_mbcnt` count to LDS and
/// reads the dword of the lane 32 away, in the other half: the console's
/// lockstep orders the two without a barrier.
fn lds_across_halves() -> Vec<u32> {
    let mut w = PROLOGUE.to_vec();
    w.extend([
        0x3420_0085, // v_lshlrev_b32 v16, 5, v0
        0x3404_0082, // v_lshlrev_b32 v2, 2, v0
        0xD765_0001,
        0x0001_00C1, // v_mbcnt_lo_u32_b32 v1, -1, 0
        0xD834_0000,
        0x0000_0102, // ds_write_b32 v2, v1
        S_WAITCNT_LGKM,
        0x3A08_00A0, // v_xor_b32 v4, 32, v0
        0x3408_0882, // v_lshlrev_b32 v4, 2, v4
        0xD8D8_0000,
        0x0300_0004, // ds_read_b32 v3, v4
        S_WAITCNT_LGKM,
    ]);
    w.extend(store(3, 0));
    w.push(S_ENDPGM);
    w
}

#[test]
fn the_lds_program_is_what_its_comments_say() {
    let code: Vec<u8> = lds_across_halves()
        .iter()
        .flat_map(|w| w.to_le_bytes())
        .collect();
    let all = gfx10::decode(&code)
        .expect("decodes")
        .iter()
        .map(|i| i.text(true))
        .collect::<Vec<_>>()
        .join("\n");
    for want in [
        "v_mbcnt_lo_u32_b32 v1, -1, 0",
        "ds_write_b32 v2, v1",
        "v_xor_b32 v4, 32, v0",
        "v_lshlrev_b32 v4, 2, v4",
        "ds_read_b32 v3, v4",
    ] {
        assert!(all.contains(want), "{want:?} not in:\n{all}");
    }
}

/// With narrow host waves (#148, D3DMetal), a one-wave group whose LDS gets
/// its barrier runs in halves instead of being refused; without them it is
/// refused, as before.
#[test]
fn lds_of_a_one_wave64_group_synced_by_barriers_runs_in_halves() {
    let n = native(&lds_across_halves(), true, 64, 256);
    let synced = lowered_with(&n, true, true).expect("synced, in halves");
    assert_eq!(synced.wave_size, Some(32));
    assert_eq!(synced.split_barriers, 1);
    let why = lowered_with(&n, true, false)
        .err()
        .expect("refused")
        .to_string();
    assert!(
        why.contains("LDS one console wave64 shares without barriers"),
        "{why}"
    );
}

/// The synced program on every hardware adapter, in halves: each lane reads
/// the count the lane in the other half wrote (32 for the low half, the
/// lane less 32 for the high one).
#[test]
#[cfg(windows)]
fn lds_synced_in_halves_matches_the_console_on_every_gpu() {
    let want: Vec<u32> = (0..64u32).map(|l| (l ^ 32).min(32)).collect();
    let failed = differences_with(&lds_across_halves(), 64, 1, &want, 256, true);
    assert!(failed.is_empty(), "{}", failed.join("\n"));
}

/// Per thread `l` of a one-wave group of 64: four passes of a loop on a
/// scalar count, each writing the lane's sum to LDS and adding the sum of
/// the lane 32 away, in the other half (`v_mbcnt` asks for the console's
/// lanes). The console's lockstep orders the accesses; split, each pass
/// needs a barrier before its read, and one before its write, which the
/// other half's read of the last pass must not miss.
fn lds_in_a_loop() -> Vec<u32> {
    let body = [
        0xD834_0000,
        0x0000_0102, // ds_write_b32 v2, v1
        S_WAITCNT_LGKM,
        0xD8D8_0000,
        0x0300_0004, // ds_read_b32 v3, v4
        S_WAITCNT_LGKM,
        0x4A02_0701, // v_add_nc_u32 v1, v1, v3
    ];
    let mut w = PROLOGUE.to_vec();
    w.extend([
        0x3420_0085, // v_lshlrev_b32 v16, 5, v0
        0xD765_0005,
        0x0001_00C1, // v_mbcnt_lo_u32_b32 v5, -1, 0
        0x3404_0082, // v_lshlrev_b32 v2, 2, v0
        0x3A08_00A0, // v_xor_b32 v4, 32, v0
        0x3408_0882, // v_lshlrev_b32 v4, 2, v4
        0x7E02_0300, // v_mov_b32 v1, v0
        0xBE94_0380, // s_mov_b32 s20, 0
    ]);
    w.extend(body);
    let back = -(body.len() as i32 + 3);
    w.extend([
        0x8014_8114,                          // s_add_u32 s20, s20, 1
        0xBF0A_8414,                          // s_cmp_lt_u32 s20, 4
        0xBF85_0000 | (back as u32 & 0xFFFF), // s_cbranch_scc1 (back to the write)
    ]);
    w.extend(store(1, 0));
    w.push(S_ENDPGM);
    w
}

/// What the console gives [`lds_in_a_loop`]: each pass, a lane's sum and
/// that of the lane 32 away both become their total.
#[cfg(windows)]
fn console_lds_in_a_loop() -> Vec<u32> {
    let mut sum: Vec<u32> = (0..64).collect();
    for _ in 0..4 {
        sum = (0..64).map(|l| sum[l] + sum[l ^ 32]).collect();
    }
    sum
}

#[test]
fn the_lds_loop_is_what_its_comments_say() {
    let code: Vec<u8> = lds_in_a_loop()
        .iter()
        .flat_map(|w| w.to_le_bytes())
        .collect();
    let all = gfx10::decode(&code)
        .expect("decodes")
        .iter()
        .map(|i| i.text(true))
        .collect::<Vec<_>>()
        .join("\n");
    for want in [
        "v_mbcnt_lo_u32_b32 v5, -1, 0",
        "v_xor_b32 v4, 32, v0",
        "v_mov_b32 v1, v0",
        "s_mov_b32 s20, 0",
        "ds_write_b32 v2, v1",
        "ds_read_b32 v3, v4",
        "v_add_nc_u32 v1, v1, v3",
        "s_add_u32 s20, s20, 1",
        "s_cmp_lt_u32 s20, 4",
        "s_cbranch_scc1",
    ] {
        assert!(all.contains(want), "{want:?} not in:\n{all}");
    }
}

/// With narrow host waves (D3DMetal), LDS a one-wave group shares in a loop
/// on a scalar count gets its barriers inside the loop and runs in halves;
/// without them it is refused, as before.
#[test]
fn lds_of_a_one_wave64_group_in_a_loop_runs_in_halves() {
    let n = native(&lds_in_a_loop(), true, 64, 256);
    let synced = lowered_with(&n, true, true).expect("synced, in halves");
    assert_eq!(synced.wave_size, Some(32));
    assert_eq!(synced.split_barriers, 2);
    #[cfg(windows)]
    if let Ok(mut v) = validator::Validator::load(validator::Validator::default_path().as_deref()) {
        v.validate(&mut synced.dxil.clone())
            .expect("DXIL validation failed");
    }
    let why = lowered_with(&n, true, false)
        .err()
        .expect("refused")
        .to_string();
    assert!(
        why.contains("LDS one console wave64 shares without barriers"),
        "{why}"
    );
}

/// The loop on every hardware adapter, in halves: every pass gives the
/// console's sums.
#[test]
#[cfg(windows)]
fn lds_in_a_loop_in_halves_matches_the_console_on_every_gpu() {
    let failed = differences_with(&lds_in_a_loop(), 64, 1, &console_lds_in_a_loop(), 256, true);
    assert!(failed.is_empty(), "{}", failed.join("\n"));
}

/// The selectors: every nibble of `s8` is odd and every one of `s9` is even.
const PERM_LOW: u32 = 0x1357_9BDF;

const PERM_HIGH: u32 = 0x2468_ACE0;

/// What `v1` holds before an instruction that may leave it alone.
const PERM_OLD: u32 = 0xAAAA;

/// A single result in v3, stored after the body restores EXEC.
fn literal_program(body: &[u32]) -> Vec<u32> {
    [
        PROLOGUE.as_slice(),
        &[0x3420_0085, 0x7E06_0280], // v16 = v0 * 32; v3 = 0
        body,
        &store(3, 0),
        &[S_ENDPGM],
    ]
    .concat()
}

/// A nonempty mixed EXEC takes the direct edge to the join. Inactive
/// lanes remain inactive there. The first popcount forces a numeric mask
/// in the narrowed block, and the joined popcount supplies the result.
fn literal_mask_join(high_only: bool) -> Vec<u32> {
    literal_program(&[
        0x7D86_00A8,                                       // v_cmp_le_u32 vcc, 40, v0
        0xBE94_246A,                                       // s_and_saveexec_b64 s[20:21], vcc
        0xBF88_0006,                                       // s_cbranch_execz restore
        0xBE8C_107E,                                       // s_bcnt1_i32_b64 s12, exec
        0xBF89_0001,                                       // s_cbranch_execnz join
        if high_only { 0xBEFF_0380 } else { 0xBEFE_04C1 }, // s_mov_b32 exec_hi, 0 or s_mov_b64 exec, -1
        0xBE8D_107E,                                       // join: s_bcnt1_i32_b64 s13, exec
        0x7E1A_020D,                                       // v_mov_b32 v13, s13
        0x4A06_1A0C,                                       // v_add_nc_u32 v3, s12, v13
        0xBEFE_0414,                                       // restore: s_mov_b64 exec, s[20:21]
    ])
}

/// Both scalar lane reads happen under an EXEC mask with no low lanes.
fn literal_lane_reads(first: bool) -> Vec<u32> {
    let read: &[u32] = if first {
        &[0x7E14_0500] // v_readfirstlane_b32 s10, v0
    } else {
        &[0xD760_000A, 0x0201_6500] // v_readlane_b32 s10, v0, 50
    };
    let mut body = vec![
        0x7D86_00A8,                           // v_cmp_le_u32 vcc, 40, v0
        0xBE94_246A,                           // s_and_saveexec_b64 s[20:21], vcc
        0xBF88_0000 | (read.len() as u32 + 1), // s_cbranch_execz restore
    ];
    body.extend(read);
    body.extend([
        0x7E06_020A, // v_mov_b32 v3, s10
        0xBEFE_0414, // restore: s_mov_b64 exec, s[20:21]
    ]);
    // The first-lane case also names the console lanes.
    body.extend([0xD765_0001, 0x0001_00C1]); // v_mbcnt_lo_u32_b32 v1, -1, 0
    literal_program(&body)
}

/// Separate writes select the upper half before reading its first lane.
fn literal_exec_dwords() -> Vec<u32> {
    literal_program(&[
        0xBE94_047E, // s_mov_b64 s[20:21], exec
        0xBEFE_0380, // s_mov_b32 exec_lo, 0
        0xBEFF_03C1, // s_mov_b32 exec_hi, -1
        0xBF88_0002, // s_cbranch_execz restore
        0x7E14_0500, // v_readfirstlane_b32 s10, v0
        0x7E06_020A, // v_mov_b32 v3, s10
        0xBEFE_0414, // restore: s_mov_b64 exec, s[20:21]
    ])
}

/// Lane 40 is inactive when the scalar write targets it.
fn literal_lane_write() -> Vec<u32> {
    literal_program(&[
        0x7D88_0094, // v_cmp_gt_u32 vcc, 20, v0
        0xBE94_246A, // s_and_saveexec_b64 s[20:21], vcc
        0xBF88_0005, // s_cbranch_execz restore
        0xD761_0003,
        0x0001_50B7, // v_writelane_b32 v3, 55, 40
        0xD760_000A,
        0x0201_5103, // v_readlane_b32 s10, v3, 40
        0x7E06_020A, // v_mov_b32 v3, s10
        0xBEFE_0414, // restore: s_mov_b64 exec, s[20:21]
    ])
}

/// Lanes leave independently as their counter reaches zero. SCC is set
/// only when every bit of EXEC compares equal to zero.
fn literal_mask_loop(mask_eq: bool) -> Vec<u32> {
    let mut body = vec![
        0x3602_0083, // v_and_b32 v1, 3, v0
        0xBE98_047E, // s_mov_b64 s[24:25], exec
        0x7D82_0280, // v_cmp_lt_u32 vcc, 0, v1
        0x87FE_6A7E, // s_and_b64 exec, exec, vcc
        0xBF88_0008, // s_cbranch_execz done
        0xD760_000A,
        0x0201_0F00, // top: v_readlane_b32 s10, v0, 7
        0x4A06_060A, // v_add_nc_u32 v3, s10, v3
        0x4E02_0281, // v_subrev_nc_u32 v1, 1, v1
        0x7D82_0280, // v_cmp_lt_u32 vcc, 0, v1
        0x87FE_6A7E, // s_and_b64 exec, exec, vcc
        0xBF12_807E, // s_cmp_eq_u64 exec, 0
        0xBF84_FFF8, // s_cbranch_scc0 top
        0xBEFE_0418, // done: s_mov_b64 exec, s[24:25]
    ];
    if !mask_eq {
        body.remove(11);
        body[4] = 0xBF88_0007; // s_cbranch_execz done
        body[11] = 0xBF89_FFF9; // s_cbranch_execnz top
    }
    literal_program(&body)
}

/// Repeated swizzles keep the first lane number of each half.
fn literal_swizzles() -> Vec<u32> {
    let mut body = vec![V_MOV_V3_V0];
    for _ in 0..12 {
        body.extend([
            0xD8D4_80B1,
            0x0300_0003, // ds_swizzle_b32 v3, v3 quad_perm:[1,0,3,2]
            S_WAITCNT_LGKM,
            0xD8D4_0020,
            0x0300_0003, // ds_swizzle_b32 v3, v3 broadcast lane 1 of each half
            S_WAITCNT_LGKM,
        ]);
    }
    body.extend([
        0x7D86_00A8, // v_cmp_le_u32 vcc, 40, v0
        0xBE94_246A, // s_and_saveexec_b64 s[20:21], vcc
        0xBF88_0003, // s_cbranch_execz restore
        0xD760_000A,
        0x0201_6503, // v_readlane_b32 s10, v3, 50
        0x7E06_020A, // v_mov_b32 v3, s10
        0xBEFE_0414, // restore: s_mov_b64 exec, s[20:21]
    ]);
    literal_program(&body)
}

/// Active lanes in both halves write under EXEC and then read their peers.
fn literal_lds() -> Vec<u32> {
    literal_program(&[
        0x3404_0082, // v_lshlrev_b32 v2, 2, v0
        0x3A08_00A0, // v_xor_b32 v4, 32, v0
        0x3408_0882, // v_lshlrev_b32 v4, 2, v4
        0x3602_009F, // v_and_b32 v1, 31, v0
        0x7D86_0290, // v_cmp_le_u32 vcc, 16, v1
        0xBE94_246A, // s_and_saveexec_b64 s[20:21], vcc
        0xBF88_0006, // s_cbranch_execz restore
        0xD834_0000,
        0x0000_0002, // ds_write_b32 v2, v0
        S_WAITCNT_LGKM,
        0xD8D8_0000,
        0x0300_0004, // ds_read_b32 v3, v4
        S_WAITCNT_LGKM,
        0xBEFE_0414, // restore: s_mov_b64 exec, s[20:21]
    ])
}

/// The compiler's whole-wave sum of lanes 20 through 63.
fn literal_sum() -> Vec<u32> {
    let sum = reduction(V_ADD_F32, 0x80, 0x7E06_0D00);
    let steps = &sum[PROLOGUE.len() + 1..sum.len() - 3];
    let mut body = vec![
        0x7D86_0094,                            // v_cmp_le_u32 vcc, 20, v0
        0xBE94_246A,                            // s_and_saveexec_b64 s[20:21], vcc
        0xBF88_0000 | (steps.len() as u32 + 1), // s_cbranch_execz restore
    ];
    body.extend(steps);
    body.extend([
        0x7E06_0301, // v_mov_b32 v3, v1
        0xBEFE_0414, // restore: s_mov_b64 exec, s[20:21]
    ]);
    literal_program(&body)
}

/// Lane 0 adds 2048 to 1 twice, rounding each addition to 2048 in f16.
fn literal_half_sum() -> Vec<u32> {
    let sum = reduction(0x32, 0x80, V_MOV_V3_V0);
    let mut body = vec![
        0x7D86_0083, // v_cmp_le_u32 vcc, 3, v0
        0xBE94_047E, // s_mov_b64 s[20:21], exec
        0x8AFE_6A7E, // s_andn2_b64 exec, exec, vcc
        0xBF88_0000, // s_cbranch_execz restore
        0x7E06_02FF,
        0x0000_3C00, // v_mov_b32 v3, 1.0 in f16
        0x7D84_0080, // v_cmp_eq_u32 vcc, 0, v0
        0xD501_0003,
        0x01A9_FF03, // v_cndmask_b32 v3, v3, 2048.0 in f16, vcc
        0x0000_6800,
    ];
    body.extend(&sum[PROLOGUE.len() + 2..sum.len() - 3]);
    body.extend([
        0x7E06_0301, // v_mov_b32 v3, v1
        0xBEFE_0414, // restore: s_mov_b64 exec, s[20:21]
    ]);
    body[3] |= (body.len() - 5) as u32;
    literal_program(&body)
}

#[test]
fn literal_wave64_recovers_refusals_in_exactly_one_wave() {
    for (words, lds, reason) in [
        (literal_mask_join(false), 0, "a lane mask read as data"),
        (
            literal_mask_join(true),
            0,
            "a lane mask whose lanes depend on its own value",
        ),
        (literal_lane_reads(false), 0, "a lane read that may cross"),
        (literal_lane_reads(true), 0, "a first-lane read"),
        (literal_exec_dwords(), 0, "a lane mask holding lanes elsewhere read as data"),
        (literal_lane_write(), 0, "a lane write in a predicated block"),
        (literal_mask_loop(false), 0, "a lane read that may cross"),
        (literal_mask_loop(true), 0, "a lane read that may cross"),
        (literal_swizzles(), 0, "a lane read that may cross"),
        (
            literal_lds(),
            512,
            "LDS one console wave64 shares without barriers",
        ),
        (
            literal_sum(),
            0,
            "a wave reduction over both 32-lane halves",
        ),
        (
            literal_half_sum(),
            0,
            "a wave reduction over both 32-lane halves",
        ),
    ] {
        let n = native(&words, true, 64, lds);
        // The per-lane translation may refuse the program (main's has
        // since learned some of these); the literal one must translate it
        // either way.
        if let Err(why) = lowered_with(&n, true, true) {
            assert!(why.to_string().contains(reason), "{why}");
        }
        let l = lower_halves(&n, true, true, true).unwrap_or_else(|e| panic!("{reason}: {e}"));
        assert_eq!(l.wave_size, Some(32));
        if lds != 0 {
            assert_eq!(l.split_barriers, 1);
        }
        #[cfg(windows)]
        if let Ok(mut v) =
            validator::Validator::load(validator::Validator::default_path().as_deref())
        {
            v.validate(&mut l.dxil.clone()).expect(reason);
        }
    }
    let words = literal_lane_reads(false);
    for threads in [48, 96, 128] {
        let n = native(&words, true, threads, 0);
        // The retry is limited to 64 threads: other sizes translate as the
        // per-lane translation does, or are refused as it refuses them.
        let plain = lower_halves(&n, true, true, false).map(|_| ());
        let retried = lower_halves(&n, true, true, true).map(|_| ());
        assert_eq!(
            plain.map_err(|e| e.to_string()),
            retried.map_err(|e| e.to_string()),
            "{threads} threads"
        );
    }
}

#[test]
fn literal_wave64_refuses_a_selector_from_hardware_state() {
    let words = literal_program(&[
        0xB909_0004, // s_getreg_b32 s9, hwreg(HW_ID)
        0xD760_000A,
        0x0200_1300, // v_readlane_b32 s10, v0, s9
        0x7E06_020A, // v_mov_b32 v3, s10
    ]);
    let n = native(&words, true, 64, 512);
    let why = lower_halves(&n, true, false, true)
        .err()
        .expect("unproven lane selector");
    assert!(
        why.to_string().contains("a lane read that may cross"),
        "{why}"
    );
}

#[test]
fn literal_wave64_refuses_a_barrier_under_an_unproven_branch() {
    let mut words = literal_lds();
    let at = words.iter().position(|&w| w == 0xBF88_0006).unwrap();
    words.splice(
        at..=at,
        [
            0xB909_0004, // s_getreg_b32 s9, hwreg(HW_ID)
            0xBF06_8009, // s_cmp_eq_u32 s9, 0
            0xBF84_0006, // s_cbranch_scc0 restore
        ],
    );
    let n = native(&words, true, 64, 512);
    let why = lower_halves(&n, true, true, true)
        .err()
        .expect("unproven group convergence");
    assert!(
        why.to_string().contains("uniform group control flow"),
        "{why}"
    );
}

#[test]
#[cfg(windows)]
fn literal_wave64_programs_match_the_console_on_every_gpu() {
    let cases: [(Vec<u32>, u32, Vec<u32>); 12] = [
        (
            literal_mask_join(false),
            0,
            (0..64).map(|l| 48 * u32::from(l >= 40)).collect(),
        ),
        (
            literal_mask_join(true),
            0,
            (0..64).map(|l| 48 * u32::from(l >= 40)).collect(),
        ),
        (
            literal_lane_reads(false),
            0,
            (0..64).map(|l| 50 * u32::from(l >= 40)).collect(),
        ),
        (
            literal_lane_reads(true),
            0,
            (0..64).map(|l| 40 * u32::from(l >= 40)).collect(),
        ),
        (
            literal_exec_dwords(),
            0,
            (0..64).map(|l| 32 * u32::from(l >= 32)).collect(),
        ),
        (
            literal_lane_write(),
            0,
            (0..64).map(|l| 55 * u32::from(l < 20 || l == 40)).collect(),
        ),
        (
            literal_mask_loop(false),
            0,
            (0..64).map(|l| 7 * (l & 3)).collect(),
        ),
        (
            literal_mask_loop(true),
            0,
            (0..64).map(|l| 7 * (l & 3)).collect(),
        ),
        (literal_swizzles(), 0, (0..64).map(|l| l & 32).collect()),
        (
            literal_lds(),
            512,
            (0..64)
                .map(|l| if (l & 31) >= 16 { l ^ 32 } else { 0 })
                .collect(),
        ),
        (
            literal_sum(),
            0,
            (0..64)
                .map(|l| if l >= 20 { 1826f32.to_bits() } else { 0 })
                .collect(),
        ),
        (
            literal_half_sum(),
            0,
            (0..64).map(|l| if l < 3 { 0x6800 } else { 0 }).collect(),
        ),
    ];
    for (words, lds, want) in cases {
        let failed = differences_in(true, &words, 64, 1, &want, lds, true, true);
        assert!(failed.is_empty(), "{}", failed.join("\n"));
    }
}

/// A number kept in VCC_HI on its way into a scalar load's offset: the
/// console wave's first thread id, plus one, times 16 bytes, read from the
/// structured buffer, and a branch on the dword read. Every lane stores the
/// dword when it is nonzero.
fn literal_vcc_hi_offset() -> Vec<u32> {
    literal_program(&[
        0xF408_0B02,
        0xFA00_0020, // s_load_dwordx4 s[44:47], s[4:5], 0x20
        S_WAITCNT_LGKM,
        0x7E2A_0500, // v_readfirstlane_b32 s21, v0
        0x8015_8115, // s_add_u32 s21, s21, 1
        0x8F6B_8415, // s_lshl_b32 vcc_hi, s21, 4
        0x8817_806B, // s_or_b32 s23, vcc_hi, 0
        0xF420_0616,
        0x2E00_0000, // s_buffer_load_dword s24, s[44:47], s23
        S_WAITCNT_LGKM,
        0xBF07_8018, // s_cmp_lg_u32 s24, 0
        0xBF84_0001, // s_cbranch_scc0 1
        0x7E06_0218, // v_mov_b32 v3, s24
    ])
}

/// VCC_HI read as a number keeps the whole-wave translation's group
/// control flow uniform.
#[test]
fn literal_wave64_reads_a_number_in_vcc_hi_as_data() {
    let words = literal_vcc_hi_offset();
    let code: Vec<u8> = words.iter().flat_map(|w| w.to_le_bytes()).collect();
    let p = ir::build(&code, true, 256).unwrap();
    let names: Vec<_> = p
        .blocks
        .iter()
        .flat_map(|b| b.insts.iter().map(|i| i.name()))
        .collect();
    for want in [
        "v_readfirstlane_b32",
        "s_add_u32",
        "s_lshl_b32",
        "s_or_b32",
        "s_buffer_load_dword",
        "s_cmp_lg_u32",
        "s_cbranch_scc0",
    ] {
        assert!(names.contains(&want), "{want} missing from {names:?}");
    }
    let n = native(&words, true, 64, 0);
    let l = lower_halves(&n, true, true, true).expect("literal halves");
    assert_eq!(l.wave_size, Some(32));
}

#[test]
#[cfg(windows)]
fn literal_wave64_vcc_hi_offset_matches_the_console_on_every_gpu() {
    let failed = differences_with_buffer(
        true,
        &literal_vcc_hi_offset(),
        64,
        1,
        &[7; 64],
        0,
        true,
        true,
        &[0, 0, 0, 0, 7, 0, 0, 0],
    );
    assert!(failed.is_empty(), "{}", failed.join("\n"));
}

/// How a full-LDS program reaches its halves exchange.
#[derive(Clone, Copy, Debug)]
enum Exchanges {
    /// One ballot read as data.
    One,
    /// Two back to back.
    Two,
    /// One in a loop of 3 trips, each adding 1 to the dwords the exchange
    /// borrows (the first 1,024) first.
    Loop,
}

/// What the program adds to dwords 0..1023 before reading them back.
const fn full_lds_bump(e: Exchanges) -> u32 {
    match e {
        Exchanges::Loop => 3,
        _ => 0,
    }
}

/// A 1,024-thread wave64 program that fills all 8,192 dwords of its LDS
/// (thread t writes dword `t + 1024k` = `t + 1024k + 0xA500_0000` for k
/// 0-7), passes a ballot between the halves (`e`), and reads all eight of
/// another wave's dwords back (thread t reads those of `(t + 32) % 1024`).
/// Per lane: the ballots' lane count, the OR of every difference from what
/// was written (0 when the LDS came back byte for byte), and its own dword
/// `t` as it is at the end.
fn full_lds_program(e: Exchanges) -> Vec<u32> {
    const BASE: u32 = 0xA500_0000;
    let bump = full_lds_bump(e);
    let ds_write = |offset: u32| [0xD834_0000 | offset, 0x0000_0401]; // ds_write_b32 v1, v4 offset:
    let ds_read = |offset: u32| [0xD8D8_0000 | offset, 0x0E00_000A]; // ds_read_b32 v14, v10 offset:
    let mut w = PROLOGUE.to_vec();
    w.extend([
        0x3420_0085, // v_lshlrev_b32 v16, 5, v0
        0x3402_0082, // v_lshlrev_b32 v1, 2, v0
        0x4A04_00FF,
        BASE, // v_add_nc_u32 v2, BASE, v0
    ]);
    for k in 0..8u32 {
        w.extend([0x4A08_04FF, 1024 * k]); // v_add_nc_u32 v4, 1024k, v2
        w.extend(ds_write(4096 * k));
    }
    w.extend([
        S_WAITCNT_LGKM,
        0xBF8A_0000, // s_barrier
        0x360E_00BF, // v_and_b32 v7, 63, v0
    ]);
    // Lanes 0-39 of each console wave: 40 a ballot.
    let ballot = [
        0x7D88_0EA8, // v_cmp_gt_u32 vcc, 40, v7
        0xBE8C_106A, // s_bcnt1_i32_b64 s12, vcc
    ];
    match e {
        Exchanges::One => w.extend(ballot),
        Exchanges::Two => {
            w.extend(ballot);
            w.extend([
                0x7D88_0E94, // v_cmp_gt_u32 vcc, 20, v7
                0xBE8D_106A, // s_bcnt1_i32_b64 s13, vcc
                0x800C_0D0C, // s_add_u32 s12, s12, s13
            ]);
        }
        Exchanges::Loop => {
            w.extend([
                0x7E12_0281, // v_mov_b32 v9, 1
                0xBE94_0380, // s_mov_b32 s20, 0
                0xBE8E_0380, // s_mov_b32 s14, 0
            ]);
            let top = w.len();
            w.extend([0xD800_0000, 0x0000_0901]); // ds_add_u32 v1, v9
            w.extend(ballot);
            w.extend([
                0x800E_0C0E, // s_add_u32 s14, s14, s12
                0x8014_8114, // s_add_u32 s20, s20, 1
                0xBF0A_8314, // s_cmp_lt_u32 s20, 3
            ]);
            let back = (top as i32 - (w.len() as i32 + 1)) as u16;
            w.push(0xBF85_0000 | u32::from(back)); // s_cbranch_scc1 top
            w.push(0xBE8C_030E); // s_mov_b32 s12, s14
        }
    }
    w.extend([
        S_WAITCNT_LGKM,
        0xBF8A_0000, // s_barrier
        0x4A1A_00A0, // v_add_nc_u32 v13, 32, v0
        0x361A_1AFF,
        0x0000_03FF, // v_and_b32 v13, 1023, v13
        0x3414_1A82, // v_lshlrev_b32 v10, 2, v13
        0x4A18_1AFF,
        BASE,        // v_add_nc_u32 v12, BASE, v13
        0x7E16_0280, // v_mov_b32 v11, 0
    ]);
    for k in 0..8u32 {
        w.extend(ds_read(4096 * k));
        w.extend([
            S_WAITCNT_LGKM,
            0x4A1E_18FF,
            1024 * k + if k == 0 { bump } else { 0 }, // v_add_nc_u32 v15, what was written, v12
            0x3A1C_1F0E,                              // v_xor_b32 v14, v14, v15
            0x3816_1D0B,                              // v_or_b32 v11, v11, v14
        ]);
    }
    w.extend([
        0xD8D8_0000,
        0x1100_0001, // ds_read_b32 v17, v1
        S_WAITCNT_LGKM,
        0x7E06_020C, // v_mov_b32 v3, s12
    ]);
    w.extend(store(3, 0));
    w.extend(store(11, 4));
    w.extend(store(17, 8));
    w.push(S_ENDPGM);
    w
}

/// What each lane of [`full_lds_program`] stores.
fn full_lds_want(e: Exchanges) -> Vec<u32> {
    let count = match e {
        Exchanges::One => 40,
        Exchanges::Two => 60,
        Exchanges::Loop => 120,
    };
    (0..1024u32)
        .flat_map(|t| [count, 0, t + 0xA500_0000 + full_lds_bump(e)])
        .collect()
}

/// The program's words are the instructions their comments name.
#[test]
fn the_full_lds_programs_are_what_their_comments_say() {
    for e in [Exchanges::One, Exchanges::Two, Exchanges::Loop] {
        let words = full_lds_program(e);
        let code: Vec<u8> = words.iter().flat_map(|w| w.to_le_bytes()).collect();
        let p = ir::build(&code, true, 256).unwrap();
        let insts: Vec<_> = p.blocks.iter().flat_map(|b| b.insts.iter()).collect();
        let count = |name: &str| insts.iter().filter(|i| i.name() == name).count();
        assert_eq!(count("ds_write_b32"), 8, "{e:?}");
        assert_eq!(count("ds_read_b32"), 9, "{e:?}");
        assert_eq!(count("s_barrier"), 2, "{e:?}");
        for name in [
            "v_lshlrev_b32",
            "v_add_nc_u32",
            "v_and_b32",
            "v_xor_b32",
            "v_or_b32",
            "v_cmp_gt_u32",
            "s_bcnt1_i32_b64",
            "buffer_store_dword",
        ] {
            assert!(count(name) > 0, "{e:?}: no {name}");
        }
        let offsets: Vec<i32> = insts
            .iter()
            .filter(|i| i.name() == "ds_write_b32")
            .map(|i| match &i.fields {
                gfx10::Fields::Ds {
                    offset0, offset1, ..
                } => i32::from(*offset1) << 8 | i32::from(*offset0),
                _ => -1,
            })
            .collect();
        assert_eq!(offsets, (0..8).map(|k| 4096 * k).collect::<Vec<_>>());
        if let Exchanges::Loop = e {
            assert_eq!(count("ds_add_u32"), 1);
            assert_eq!(count("s_cbranch_scc1"), 1);
            assert!(p.blocks.len() >= 3);
        }
    }
}

/// A program whose own LDS is the host's whole 32 KiB passes values between
/// its halves through dwords it borrows from that LDS.
#[test]
fn halves_exchange_in_a_program_that_fills_the_lds() {
    for e in [Exchanges::One, Exchanges::Two, Exchanges::Loop] {
        let words = full_lds_program(e);
        let full = lowered_with(&native(&words, true, 1024, 32768), true, true)
            .unwrap_or_else(|why| panic!("{e:?}: {why}"));
        assert_eq!(full.wave_size, Some(32));
    }
}

#[test]
#[cfg(windows)]
fn halves_give_back_borrowed_lds_on_every_gpu() {
    let mut failed = Vec::new();
    for e in [Exchanges::One, Exchanges::Two, Exchanges::Loop] {
        failed.extend(
            differences_with(
                &full_lds_program(e),
                1024,
                3,
                &full_lds_want(e),
                32768,
                true,
            )
            .into_iter()
            .map(|f| format!("{e:?}: {f}")),
        );
    }
    assert!(failed.is_empty(), "{}", failed.join("\n"));
}

/// VOP2 opcode for `v_add_f16`.
const V_ADD_F16: u32 = 0x32;

/// Appends the console compiler's whole-wave reduction of `v3` by VOP2 `op`
/// (identity inline constant `ident`), leaving the wave result in `s15`.
fn emit_reduction_body(w: &mut Vec<u32>, op: u32, ident: u32) {
    let vop2 = |word: u32| word & !(0x3F << 25) | op << 25;
    w.extend([
        0xBE8C_25C1, // s_or_saveexec_b64 s[12:13], -1
        0xD501_0001,
        0x0032_0600 | ident, // v_cndmask_b32 v1, ident, v3, s[12:13]
    ]);
    for control in [0x161, 0x162, 0x164, 0x168] {
        w.extend([
            vop2(0x0602_02FA),
            0xFF00_0001 | control << 8, // op v1, v1, v1 row_xmask
        ]);
    }
    w.extend([
        0xD778_0002,
        0x0201_0101,       // v_permlanex16_b32 v2, v1, 0, 0
        vop2(0x0602_0302), // op v1, v2, v1
        0xD760_000E,
        0x0001_4101,       // v_readlane_b32 s14, v1, 32
        vop2(0x0602_020E), // op v1, s14, v1
        0xD760_000F,
        0x0001_0101, // v_readlane_b32 s15, v1, 0
        0xBEFE_040C, // s_mov_b64 exec, s[12:13]
    ]);
}

/// Test 1: early-out bounds check (`keep_below`: `if (tid >= limit) return;`;
/// `!keep_below`: `if (tid < limit) return;`) followed by cross-half
/// sum/min/max reductions, ballot/`s_bcnt1`/`v_mbcnt`, `v_readfirstlane_b32`,
/// `v_readlane_b32`, and split-wave LDS write-then-read.
fn early_out_program(keep_below: bool, limit: u32) -> Vec<u32> {
    let mut w = PROLOGUE.to_vec();
    // Initialize output offset and zero-initialize LDS[v0 * 4].
    w.extend([
        0x3420_0085, // v_lshlrev_b32 v16, 5, v0
        0x3404_0082, // v_lshlrev_b32 v2, 2, v0
        0x7E10_0280, // v_mov_b32 v8, 0
        0xD834_0000,
        0x0000_0802, // ds_write_b32 v2, v8
        S_WAITCNT_LGKM,
        // `keep_below`: v_cmp_gt_u32 vcc, limit, v0 (v0 < limit)
        // `!keep_below`: v_cmp_le_u32 vcc, limit, v0 (v0 >= limit)
        if keep_below { 0x7D88_00FF } else { 0x7D86_00FF },
        limit,
        0xBE8A_246A, // s_and_saveexec_b64 s[10:11], vcc
    ]);
    let branch_at = w.len();
    w.push(0); // patched to `s_cbranch_execz end`

    let mut body = Vec::new();
    // Slot 0: f32 sum reduction of (float)v0 across active lanes.
    body.push(0x7E06_0D00); // v_cvt_f32_u32 v3, v0
    emit_reduction_body(&mut body, V_ADD_F32, 0x80);
    body.push(0x7E0E_020F); // v_mov_b32 v7, s15

    // Slot 1: u32 min reduction of v0 across active lanes.
    body.push(V_MOV_V3_V0);
    emit_reduction_body(&mut body, V_MIN_U32, 0xC1);
    body.push(0x7E10_020F); // v_mov_b32 v8, s15

    // Slot 2: u32 max reduction of v0 across active lanes.
    body.push(V_MOV_V3_V0);
    emit_reduction_body(&mut body, V_MAX_U32, 0x80);
    body.push(0x7E12_020F); // v_mov_b32 v9, s15

    // Slots 3 & 4: cross-half ballot (`v0 < 50`) count & `v_mbcnt` prefix.
    body.extend([
        0x7D88_00FF,
        50,          // v_cmp_gt_u32 vcc, 50, v0
        0xBE8C_106A, // s_bcnt1_i32_b64 s12, vcc
        0x7E14_020C, // v_mov_b32 v10, s12
        0xD765_000B,
        0x0001_006A, // v_mbcnt_lo_u32_b32 v11, vcc_lo, 0
        0xD766_000B,
        0x0002_166B, // v_mbcnt_hi_u32_b32 v11, vcc_hi, v11
    ]);

    // Slot 5: `v_readfirstlane_b32` of v0 (upper half when lower half inactive).
    body.extend([
        0x7E18_0500, // v_readfirstlane_b32 s12, v0
        0x7E18_020C, // v_mov_b32 v12, s12
    ]);

    // Slot 6: `v_readlane_b32` of v0 at lane 38.
    body.extend([
        0xD760_000C,
        0x0001_0100 | (128 + 38) << 9, // v_readlane_b32 s12, v0, 38
        0x7E1A_020C,                   // v_mov_b32 v13, s12
    ]);

    // Slot 7: split-wave LDS write (`v0 + 100` at `v0 * 4`) then read at `(v0 ^ 32) * 4`.
    body.extend([
        0x3404_0082, // v_lshlrev_b32 v2, 2, v0
        0x4A1C_00FF,
        100, // v_add_nc_u32 v14, 100, v0
        0xD834_0000,
        0x0000_0E02, // ds_write_b32 v2, v14
        S_WAITCNT_LGKM,
        0x3A08_00A0, // v_xor_b32 v4, 32, v0
        0x3408_0882, // v_lshlrev_b32 v4, 2, v4
        0xD8D8_0000,
        0x0F00_0004, // ds_read_b32 v15, v4
        S_WAITCNT_LGKM,
    ]);

    for (vdata, slot) in [
        (7, 0),
        (8, 1),
        (9, 2),
        (10, 3),
        (11, 4),
        (12, 5),
        (13, 6),
        (15, 7),
    ] {
        body.extend(store(vdata, 4 * slot));
    }

    w[branch_at] = 0xBF88_0000 | (body.len() as u32 & 0xFFFF); // s_cbranch_execz end
    w.extend(body);
    w.push(S_ENDPGM);
    w
}

fn console_early_out(keep_below: bool, limit: u32, l: u32) -> [u32; 8] {
    let active = |t: u32| if keep_below { t < limit } else { t >= limit };
    if !active(l) {
        return [0; 8];
    }
    let active_lanes: Vec<u32> = (0..64).filter(|&t| active(t)).collect();
    let sum = active_lanes.iter().sum::<u32>() as f32;
    let min = *active_lanes.iter().min().unwrap();
    let max = *active_lanes.iter().max().unwrap();
    let below50: Vec<u32> = active_lanes.iter().copied().filter(|&t| t < 50).collect();
    let bcnt = below50.len() as u32;
    let mbcnt = below50.iter().filter(|&&t| t < l).count() as u32;
    let first = active_lanes[0];
    let peer = l ^ 32;
    let lds = if active(peer) { peer + 100 } else { 0 };
    [sum.to_bits(), min, max, bcnt, mbcnt, first, 38, lds]
}

/// Test 2: acyclic `if (tid < 36) { ... } else { ... }` diamond containing
/// cross-half ballot + f32 reduction + LDS exchange in the `then` branch and
/// cross-half ballot + f32 reduction + `v_readfirstlane_b32` (upper half only)
/// in the `else` branch, merging `v8..v10` at the join.
fn diamond_program() -> Vec<u32> {
    let mut w = PROLOGUE.to_vec();
    w.extend([
        0x3420_0085, // v_lshlrev_b32 v16, 5, v0
        0x3404_0082, // v_lshlrev_b32 v2, 2, v0
        0x7E10_0280, // v_mov_b32 v8, 0
        0x7E12_0280, // v_mov_b32 v9, 0
        0x7E14_0280, // v_mov_b32 v10, 0
        0xD834_0000,
        0x0000_0802, // ds_write_b32 v2, v8
        S_WAITCNT_LGKM,
        0x7D88_00FF,
        36,          // v_cmp_gt_u32 vcc, 36, v0 (v0 < 36)
        0xBE8A_246A, // s_and_saveexec_b64 s[10:11], vcc
    ]);
    let then_branch = w.len();
    w.push(0); // patched to `s_cbranch_execz else_check`

    let mut then_body = Vec::new();
    then_body.extend([
        0x7D88_00FF,
        64,          // v_cmp_gt_u32 vcc, 64, v0
        0xBE8C_106A, // s_bcnt1_i32_b64 s12, vcc
        0x7E10_020C, // v_mov_b32 v8, s12
        0x7E06_0D00, // v_cvt_f32_u32 v3, v0
    ]);
    emit_reduction_body(&mut then_body, V_ADD_F32, 0x80);
    then_body.extend([
        0x7E12_020F, // v_mov_b32 v9, s15
        0x3404_0082, // v_lshlrev_b32 v2, 2, v0
        0x4A16_008A, // v_add_nc_u32 v11, 10, v0
        0xD834_0000,
        0x0000_0B02, // ds_write_b32 v2, v11
        S_WAITCNT_LGKM,
        0x3A08_00A0, // v_xor_b32 v4, 32, v0
        0x3408_0882, // v_lshlrev_b32 v4, 2, v4
        0xD8D8_0000,
        0x0A00_0004, // ds_read_b32 v10, v4
        S_WAITCNT_LGKM,
    ]);
    w[then_branch] = 0xBF88_0000 | (then_body.len() as u32 & 0xFFFF);
    w.extend(then_body);

    // Else check: flip EXEC to `saved & !exec` (v0 >= 36).
    w.push(0x8AFE_7E0A); // s_andn2_b64 exec, s[10:11], exec
    let else_branch = w.len();
    w.push(0); // patched to `s_cbranch_execz join`

    let mut else_body = Vec::new();
    else_body.extend([
        0x7D88_00FF,
        64,          // v_cmp_gt_u32 vcc, 64, v0
        0xBE8C_106A, // s_bcnt1_i32_b64 s12, vcc
        0x7E10_020C, // v_mov_b32 v8, s12
        0x7E06_0D00, // v_cvt_f32_u32 v3, v0
    ]);
    emit_reduction_body(&mut else_body, V_ADD_F32, 0x80);
    else_body.extend([
        0x7E12_020F, // v_mov_b32 v9, s15
        0x7E1A_0500, // v_readfirstlane_b32 s13, v0
        0x7E14_020D, // v_mov_b32 v10, s13
    ]);
    w[else_branch] = 0xBF88_0000 | (else_body.len() as u32 & 0xFFFF);
    w.extend(else_body);

    // Join: restore EXEC and store merged v8..v10.
    w.push(0xBEFE_040A); // s_mov_b64 exec, s[10:11]
    w.extend(store(8, 0));
    w.extend(store(9, 4));
    w.extend(store(10, 8));
    w.push(S_ENDPGM);
    w
}

fn console_diamond(l: u32) -> [u32; 3] {
    if l < 36 {
        let sum: u32 = (0..36).sum();
        let peer = l ^ 32;
        let lds = if peer < 36 { peer + 10 } else { 0 };
        [36, (sum as f32).to_bits(), lds]
    } else {
        let sum: u32 = (36..64).sum();
        [28, (sum as f32).to_bits(), 36]
    }
}

/// Test 3: `Shape::Half(Op::AddH)` 16-bit float wave64 reduction across both
/// 32-lane halves. Low 16 bits of `v3` hold `(f16)v0`, high 16 bits hold
/// `v0 + 7`; the reduction sums the low f16 across the wave and keeps lane 0's
/// high 16 bits (`(l & !63) + 7`).
fn half_add_reduction() -> Vec<u32> {
    let mut w = PROLOGUE.to_vec();
    w.extend([
        0x3420_0085, // v_lshlrev_b32 v16, 5, v0
        0x7E0A_0D00, // v_cvt_f32_u32 v5, v0
        0x7E06_0280, // v_mov_b32 v3, 0
        0x7E06_1505, // v_cvt_f16_f32 v3, v5
        0x4A08_0087, // v_add_nc_u32 v4, 7, v0
        0x3408_0890, // v_lshlrev_b32 v4, 16, v4
        0x3806_0903, // v_or_b32 v3, v3, v4
    ]);
    emit_reduction_body(&mut w, V_ADD_F16, 0x80);
    w.push(0x7E02_020F); // v_mov_b32 v1, s15
    w.extend(store(1, 0));
    w.push(S_ENDPGM);
    w
}

/// Exact IEEE-754 f16 encoding of an exact non-negative integer `n` in `0..=65504`.
fn u32_to_f16_bits(n: u32) -> u32 {
    if n == 0 {
        return 0;
    }
    let msb = 31 - n.leading_zeros();
    let exp = msb + 15;
    let frac = if msb >= 10 {
        (n >> (msb - 10)) & 0x3FF
    } else {
        (n << (10 - msb)) & 0x3FF
    };
    (exp << 10) | frac
}

/// Test 4: uniform loop whose trip count comes from a group-uniform
/// `buffer_load_dword` -> `v_readfirstlane_b32`, containing a cross-half
/// ballot (`v0 < 40`) and cross-half `v_add_f32` reduction accumulated across
/// every iteration.
fn buffer_load_uniform_loop() -> Vec<u32> {
    let mut w = PROLOGUE.to_vec();
    w.extend([
        0xF408_0B02,
        0xFA00_0020, // s_load_dwordx4 s[44:47], s[4:5], 0x20
        S_WAITCNT_LGKM,
        0x3420_0085, // v_lshlrev_b32 v16, 5, v0
        0x7E06_0280, // v_mov_b32 v3, 0
        0xE030_1000,
        0x800B_0403, // buffer_load_dword v4, v3, s[44:47], 0 offen
        0xBF8C_3F70, // s_waitcnt vmcnt(0)
        0x7E2A_0504, // v_readfirstlane_b32 s21, v4
        0x7E0C_0280, // v_mov_b32 v6, 0 (ballot accumulator)
        0x7E0E_0280, // v_mov_b32 v7, 0 (f32 reduction accumulator)
        0xBE94_0380, // s_mov_b32 s20, 0
    ]);
    let mut body = Vec::new();
    body.extend([
        0x7D88_00A8, // v_cmp_gt_u32 vcc, 40, v0
        0xBE8C_106A, // s_bcnt1_i32_b64 s12, vcc
        0x4A0C_0C0C, // v_add_nc_u32 v6, s12, v6
        0x7E06_0D00, // v_cvt_f32_u32 v3, v0
    ]);
    emit_reduction_body(&mut body, V_ADD_F32, 0x80);
    body.push(0x060E_0E0F); // v_add_f32 v7, s15, v7
    let back = -(body.len() as i32 + 3);
    w.extend(body);
    w.extend([
        0x8014_8114,                          // s_add_u32 s20, s20, 1
        0xBF0A_1514,                          // s_cmp_lt_u32 s20, s21
        0xBF85_0000 | (back as u32 & 0xFFFF), // s_cbranch_scc1
    ]);
    w.extend(store(6, 0));
    w.extend(store(7, 4));
    w.push(S_ENDPGM);
    w
}

#[test]
fn the_predication_test_programs_are_what_their_comments_say() {
    let programs: Vec<(Vec<u32>, &[&str])> = vec![
        (
            early_out_program(true, 40),
            &[
                "v_cmp_gt_u32 vcc, lit(0x00000028), v0",
                "s_and_saveexec_b64 s[10:11], vcc",
                "s_cbranch_execz",
                "v_readfirstlane_b32 s12, v0",
                "v_readlane_b32 s12, v0, 38",
                "ds_write_b32 v2, v14",
                "ds_read_b32 v15, v4",
            ],
        ),
        (
            early_out_program(false, 35),
            &[
                "v_cmp_le_u32 vcc, lit(0x00000023), v0",
                "s_and_saveexec_b64 s[10:11], vcc",
                "s_cbranch_execz",
            ],
        ),
        (
            diamond_program(),
            &[
                "v_cmp_gt_u32 vcc, lit(0x00000024), v0",
                "s_and_saveexec_b64 s[10:11], vcc",
                "s_andn2_b64 exec, s[10:11], exec",
                "v_readfirstlane_b32 s13, v0",
                "s_mov_b64 exec, s[10:11]",
            ],
        ),
        (
            half_add_reduction(),
            &[
                "v_cvt_f16_f32 v3, v5",
                "v_add_nc_u32 v4, 7, v0",
                "v_lshlrev_b32 v4, 16, v4",
                "v_or_b32 v3, v3, v4",
                "v_add_f16 v1, v2, v1",
                "v_add_f16 v1, s14, v1",
            ],
        ),
        (
            buffer_load_uniform_loop(),
            &[
                "buffer_load_dword v4, v3, s[44:47], 0 offen",
                "v_readfirstlane_b32 s21, v4",
                "v_add_nc_u32 v6, s12, v6",
                "v_add_f32 v7, s15, v7",
                "s_cmp_lt_u32 s20, s21",
            ],
        ),
    ];
    for (words, wants) in programs {
        let code: Vec<u8> = words.iter().flat_map(|w| w.to_le_bytes()).collect();
        let all = gfx10::decode(&code)
            .expect("decodes")
            .iter()
            .map(|i| i.text(true))
            .collect::<Vec<_>>()
            .join("\n");
        for want in wants {
            assert!(all.contains(want), "{want:?} not in:\n{all}");
        }
    }
}

#[test]
#[cfg(windows)]
fn predicated_early_out_matches_the_console_on_every_gpu() {
    let mut failed = Vec::new();
    for (keep_below, limit) in [(true, 40), (false, 35)] {
        let words = early_out_program(keep_below, limit);
        let n = native(&words, true, 64, 256);
        let lowered = lowered_with(&n, true, true).expect("predicated early-out lowers");
        assert_eq!(lowered.wave_size, Some(32));
        if let Ok(mut v) =
            validator::Validator::load(validator::Validator::default_path().as_deref())
        {
            v.validate(&mut lowered.dxil.clone())
                .expect("DXIL validation failed");
        }
        let want: Vec<u32> = (0..64)
            .flat_map(|l| console_early_out(keep_below, limit, l))
            .collect();
        failed.extend(differences_with(&words, 64, 8, &want, 256, true));
    }
    assert!(failed.is_empty(), "{}", failed.join("\n"));
}

#[test]
#[cfg(windows)]
fn predicated_diamond_matches_the_console_on_every_gpu() {
    let words = diamond_program();
    let n = native(&words, true, 64, 256);
    let lowered = lowered_with(&n, true, true).expect("predicated diamond lowers");
    assert_eq!(lowered.wave_size, Some(32));
    if let Ok(mut v) = validator::Validator::load(validator::Validator::default_path().as_deref()) {
        v.validate(&mut lowered.dxil.clone())
            .expect("DXIL validation failed");
    }
    let want: Vec<u32> = (0..64).flat_map(console_diamond).collect();
    let failed = differences_with(&words, 64, 3, &want, 256, true);
    assert!(failed.is_empty(), "{}", failed.join("\n"));
}

#[test]
#[cfg(windows)]
fn half_add_reduction_across_halves_matches_the_console_on_every_gpu() {
    let words = half_add_reduction();
    let mut failed = Vec::new();
    for threads in [32, 64, 96, 128] {
        let n = native(&words, true, threads, 0);
        let lowered = lowered_with(&n, true, true).expect("half add reduction lowers");
        assert_eq!(lowered.wave_size, Some(32));
        if let Ok(mut v) =
            validator::Validator::load(validator::Validator::default_path().as_deref())
        {
            v.validate(&mut lowered.dxil.clone())
                .expect("DXIL validation failed");
        }
        let want: Vec<u32> = (0..threads)
            .map(|l| {
                let wave = l & !63;
                let sum = console_reduction(threads, l, |a, b| a + b, |t| t);
                let hi = (wave + 7) << 16;
                hi | u32_to_f16_bits(sum)
            })
            .collect();
        failed.extend(differences_with(&words, threads, 1, &want, 0, true));
    }
    assert!(failed.is_empty(), "{}", failed.join("\n"));
}

#[test]
#[cfg(windows)]
fn uniform_buffer_load_loop_across_halves_matches_the_console_on_every_gpu() {
    let words = buffer_load_uniform_loop();
    let mut failed = Vec::new();
    let trips = 3u32;
    for threads in [64, 96, 128] {
        let n = native(&words, true, threads, 0);
        let lowered = lowered_with(&n, true, true).expect("buffer-load loop lowers");
        assert_eq!(lowered.wave_size, Some(32));
        if let Ok(mut v) =
            validator::Validator::load(validator::Validator::default_path().as_deref())
        {
            v.validate(&mut lowered.dxil.clone())
                .expect("DXIL validation failed");
        }
        let want: Vec<u32> = (0..threads)
            .flat_map(|l| {
                let bcnt = console_lanes_below(40, threads, l)[0];
                let sum = console_reduction(threads, l, |a, b| a + b, |t| t) as f32;
                [trips * bcnt, ((trips as f32) * sum).to_bits()]
            })
            .collect();
        failed.extend(differences_with_buffer(
            true,
            &words,
            threads,
            2,
            &want,
            0,
            true,
            false,
            &[trips, 0, 0, 0],
        ));
    }
    assert!(failed.is_empty(), "{}", failed.join("\n"));
}

/// Skip-guard (`if (tid >= keep_from) { for (i = 0..3) { ... } }`) followed by
/// a tail store: exercises `SkipGuard` predication around a uniform loop
/// containing cross-half ballot, `v_readfirstlane_b32` (including upper-half
/// only), and f32 wave reduction, plus `SkipGuard` join-phi preservation (`999`
/// for inactive lanes).
fn skip_guard_uniform_loop_program(keep_from: u32) -> Vec<u32> {
    let mut w = PROLOGUE.to_vec();
    w.extend([
        0x3420_0085, // v_lshlrev_b32 v16, 5, v0
        0x7E0C_02FF,
        999, // v_mov_b32 v6, 999
        0x7E0E_02FF,
        999, // v_mov_b32 v7, 999
        0x7E10_02FF,
        999, // v_mov_b32 v8, 999
        0x7D86_00FF,
        keep_from,   // v_cmp_le_u32 vcc, lit(keep_from), v0 (v0 >= keep_from)
        0xBE8A_246A, // s_and_saveexec_b64 s[10:11], vcc
    ]);
    let skip_at = w.len();
    w.push(0); // patched to `s_cbranch_execz join`

    let mut body = vec![
        0x7E0C_0280, // v_mov_b32 v6, 0
        0x7E0E_0280, // v_mov_b32 v7, 0
        0x7E10_0280, // v_mov_b32 v8, 0
        0xBE94_0380, // s_mov_b32 s20, 0
    ];
    let mut loop_body = vec![
        0x7D88_00FF,
        50,          // v_cmp_gt_u32 vcc, 50, v0
        0xBE8C_106A, // s_bcnt1_i32_b64 s12, vcc
        0x4A0C_0C0C, // v_add_nc_u32 v6, s12, v6
        0x7E1A_0500, // v_readfirstlane_b32 s13, v0
        0x7E0E_020D, // v_mov_b32 v7, s13
        0x7E06_0D00, // v_cvt_f32_u32 v3, v0
    ];
    emit_reduction_body(&mut loop_body, V_ADD_F32, 0x80);
    loop_body.push(0x0610_100F); // v_add_f32 v8, s15, v8
    let back = -(loop_body.len() as i32 + 3);
    body.extend(loop_body);
    body.extend([
        0x8014_8114,                          // s_add_u32 s20, s20, 1
        0xBF0A_8314,                          // s_cmp_lt_u32 s20, 3
        0xBF85_0000 | (back as u32 & 0xFFFF), // s_cbranch_scc1
    ]);

    w[skip_at] = 0xBF88_0000 | (body.len() as u32 & 0xFFFF); // s_cbranch_execz join
    w.extend(body);
    w.push(0xBEFE_040A); // s_mov_b64 exec, s[10:11]
    w.extend(store(6, 0));
    w.extend(store(7, 4));
    w.extend(store(8, 8));
    w.push(S_ENDPGM);
    w
}

/// Wave64 scalarization loop (`while (exec != 0)`): partitions lanes by a key
/// (`(v0 >> 3) & 3` plus `4` for `v0 >= 48`) so wave 0 runs 6 iterations
/// (including cross-half keys, lower-half-only keys, and upper-half-only keys)
/// while wave 1 runs 4 iterations, using `v_readfirstlane_b32`, `s_bcnt1`,
/// `s_ff1`, and `v_mbcnt` across both halves inside a `LockstepLoop`.
fn wave_scalarization_loop_program() -> Vec<u32> {
    let mut w = PROLOGUE.to_vec();
    w.extend([
        0x3420_0085, // v_lshlrev_b32 v16, 5, v0
        0x2C0A_0083, // v_lshrrev_b32 v5, 3, v0
        0x360A_0A83, // v_and_b32 v5, 3, v5
        0x7D86_00FF,
        48,          // v_cmp_le_u32 vcc, 48, v0 (v0 >= 48)
        0x4A08_0A84, // v_add_nc_u32 v4, 4, v5
        0xD501_0005,
        0x01AA_0905, // v_cndmask_b32 v5, v5, v4, vcc
        0x7E0C_0280, // v_mov_b32 v6, 0 (matched key)
        0x7E0E_0280, // v_mov_b32 v7, 0 (matched count)
        0x7E10_0280, // v_mov_b32 v8, 0 (matched first lane)
        0x7E12_0280, // v_mov_b32 v9, 0 (matched mbcnt)
        0xBE8A_047E, // s_mov_b64 s[10:11], exec
    ]);
    let loop_body: Vec<u32> = vec![
        0x7E18_0505, // v_readfirstlane_b32 s12, v5
        0x7D84_0A0C, // v_cmp_eq_u32 vcc, s12, v5
        0xBE8D_106A, // s_bcnt1_i32_b64 s13, vcc
        0xBE8E_146A, // s_ff1_i32_b64 s14, vcc
        0xD765_000A,
        0x0001_006A, // v_mbcnt_lo_u32_b32 v10, vcc_lo, 0
        0xD766_000A,
        0x0002_146B, // v_mbcnt_hi_u32_b32 v10, vcc_hi, v10
        0xD501_0006,
        0x01A8_1906, // v_cndmask_b32 v6, v6, s12, vcc
        0xD501_0007,
        0x01A8_1B07, // v_cndmask_b32 v7, v7, s13, vcc
        0xD501_0008,
        0x01A8_1D08, // v_cndmask_b32 v8, v8, s14, vcc
        0xD501_0009,
        0x01AA_1509, // v_cndmask_b32 v9, v9, v10, vcc
        0x8AFE_6A7E, // s_andn2_b64 exec, exec, vcc
    ];
    let back = -(loop_body.len() as i32 + 1);
    w.extend(loop_body);
    w.push(0xBF89_0000 | (back as u32 & 0xFFFF)); // s_cbranch_execnz loop
    w.push(0xBEFE_040A); // s_mov_b64 exec, s[10:11]
    w.extend(store(6, 0));
    w.extend(store(7, 4));
    w.extend(store(8, 8));
    w.extend(store(9, 12));
    w.push(S_ENDPGM);
    w
}

/// Early-out branching to a shared epilogue trampoline (`s_mov_b64 exec, -1;
/// s_branch end`) followed by 64-bit lane-mask arithmetic (`s_and_b64`,
/// `s_bcnt1_i32_b64`, `s_ff1_i32_b64`, and high-dword `s[15]` extraction).
fn trampoline_early_out_mask_program() -> Vec<u32> {
    let mut w = PROLOGUE.to_vec();
    w.extend([
        0x3420_0085, // v_lshlrev_b32 v16, 5, v0
        0x7D88_00FF,
        45,          // v_cmp_gt_u32 vcc, 45, v0 (v0 < 45)
        0xBE8A_246A, // s_and_saveexec_b64 s[10:11], vcc
    ]);
    let branch_at = w.len();
    w.push(0); // patched to `s_cbranch_execz trampoline`

    let mut body = vec![
        0x7D86_00FF,
        10,          // v_cmp_le_u32 vcc, 10, v0 (v0 >= 10)
        0xBE8E_046A, // s_mov_b64 s[14:15], vcc
        0x7D88_00FF,
        40,          // v_cmp_gt_u32 vcc, 40, v0 (v0 < 40)
        0x878E_6A0E, // s_and_b64 s[14:15], s[14:15], vcc (10 <= v0 < 40)
        0xBE8C_100E, // s_bcnt1_i32_b64 s12, s[14:15]
        0xBE8D_140E, // s_ff1_i32_b64 s13, s[14:15]
        0x7E0C_020C, // v_mov_b32 v6, s12
        0x7E0E_020D, // v_mov_b32 v7, s13
        0x7E10_020F, // v_mov_b32 v8, s15 (high 32-bit mask word: lanes 32..39)
    ];
    body.extend(store(6, 0));
    body.extend(store(7, 4));
    body.extend(store(8, 8));
    body.push(0xBF82_0002); // s_branch +2 (skip trampoline to S_ENDPGM)

    w[branch_at] = 0xBF88_0000 | (body.len() as u32 & 0xFFFF); // s_cbranch_execz trampoline
    w.extend(body);
    // Epilogue trampoline: restores EXEC and branches to S_ENDPGM.
    w.push(0xBEFE_04C1); // s_mov_b64 exec, -1
    w.push(0xBF82_0000); // s_branch +0 (next word is S_ENDPGM)
    w.push(S_ENDPGM);
    w
}

#[test]
fn the_extended_halves_programs_are_what_their_comments_say() {
    let programs: Vec<(Vec<u32>, &[&str])> = vec![
        (
            skip_guard_uniform_loop_program(36),
            &[
                "v_cmp_le_u32 vcc, lit(0x00000024), v0",
                "s_and_saveexec_b64 s[10:11], vcc",
                "s_cbranch_execz",
                "v_readfirstlane_b32 s13, v0",
                "s_cmp_lt_u32 s20, 3",
                "s_mov_b64 exec, s[10:11]",
            ],
        ),
        (
            wave_scalarization_loop_program(),
            &[
                "v_lshrrev_b32 v5, 3, v0",
                "v_and_b32 v5, 3, v5",
                "v_readfirstlane_b32 s12, v5",
                "v_cmp_eq_u32 vcc, s12, v5",
                "s_bcnt1_i32_b64 s13, vcc",
                "s_ff1_i32_b64 s14, vcc",
                "s_andn2_b64 exec, exec, vcc",
                "s_cbranch_execnz",
            ],
        ),
        (
            trampoline_early_out_mask_program(),
            &[
                "v_cmp_gt_u32 vcc, lit(0x0000002d), v0",
                "s_and_b64 s[14:15], s[14:15], vcc",
                "s_bcnt1_i32_b64 s12, s[14:15]",
                "s_ff1_i32_b64 s13, s[14:15]",
                "v_mov_b32 v8, s15",
            ],
        ),
    ];
    for (words, wants) in programs {
        let code: Vec<u8> = words.iter().flat_map(|w| w.to_le_bytes()).collect();
        let all = gfx10::decode(&code)
            .expect("decodes")
            .iter()
            .map(|i| i.text(true))
            .collect::<Vec<_>>()
            .join("\n");
        for want in wants {
            assert!(all.contains(want), "{want:?} not in:\n{all}");
        }
    }
}

#[test]
fn extended_wave64_programs_lower_to_32_lane_waves() {
    for (words, lds, sizes) in [
        (early_out_program(true, 40), 256, &[64][..]),
        (early_out_program(false, 35), 256, &[64]),
        (diamond_program(), 256, &[64]),
        (half_add_reduction(), 0, &[32, 64, 96, 128]),
        (buffer_load_uniform_loop(), 0, &[64, 96, 128]),
        (skip_guard_uniform_loop_program(20), 0, &[64]),
        (skip_guard_uniform_loop_program(36), 0, &[64]),
        (wave_scalarization_loop_program(), 0, &[64, 96, 128]),
        (trampoline_early_out_mask_program(), 0, &[64]),
    ] {
        for &threads in sizes {
            let n = native(&words, true, threads, lds);
            let lowered = lowered_with(&n, true, true).expect("extended wave64 program lowers");
            assert_eq!(lowered.wave_size, Some(32));
        }
    }
}

#[test]
#[cfg(windows)]
fn skip_guard_with_uniform_loop_matches_the_console_on_every_gpu() {
    let mut failed = Vec::new();
    for keep_from in [20u32, 36u32] {
        let words = skip_guard_uniform_loop_program(keep_from);
        let n = native(&words, true, 64, 0);
        let lowered = lowered_with(&n, true, true).expect("skip-guard uniform loop lowers");
        assert_eq!(lowered.wave_size, Some(32));
        if let Ok(mut v) =
            validator::Validator::load(validator::Validator::default_path().as_deref())
        {
            v.validate(&mut lowered.dxil.clone())
                .expect("DXIL validation failed");
        }
        let bcnt = (keep_from..50).count() as u32;
        let sum = (keep_from..64).sum::<u32>() as f32;
        let want: Vec<u32> = (0..64)
            .flat_map(|l| {
                if l >= keep_from {
                    [3 * bcnt, keep_from, (3.0 * sum).to_bits()]
                } else {
                    [999, 999, 999]
                }
            })
            .collect();
        failed.extend(differences_with(&words, 64, 3, &want, 0, true));
    }
    assert!(failed.is_empty(), "{}", failed.join("\n"));
}

#[test]
#[cfg(windows)]
fn wave_scalarization_lockstep_loop_matches_the_console_on_every_gpu() {
    let words = wave_scalarization_loop_program();
    let mut failed = Vec::new();
    let key_of = |l: u32| ((l >> 3) & 3) + if l >= 48 { 4 } else { 0 };
    for threads in [64, 96, 128] {
        let n = native(&words, true, threads, 0);
        let lowered = lowered_with(&n, true, true).expect("scalarization loop lowers");
        assert_eq!(lowered.wave_size, Some(32));
        if let Ok(mut v) =
            validator::Validator::load(validator::Validator::default_path().as_deref())
        {
            v.validate(&mut lowered.dxil.clone())
                .expect("DXIL validation failed");
        }
        let want: Vec<u32> = (0..threads)
            .flat_map(|l| {
                let wave = l & !63;
                let end = (wave + 64).min(threads);
                let k = key_of(l);
                let peers: Vec<u32> = (wave..end).filter(|&t| key_of(t) == k).collect();
                let count = peers.len() as u32;
                let ff1 = peers[0] - wave;
                let mbcnt = peers.iter().filter(|&&t| t < l).count() as u32;
                [k, count, ff1, mbcnt]
            })
            .collect();
        failed.extend(differences_with(&words, threads, 4, &want, 0, true));
    }
    assert!(failed.is_empty(), "{}", failed.join("\n"));
}

#[test]
#[cfg(windows)]
fn trampoline_early_out_and_mask_data_match_the_console_on_every_gpu() {
    let words = trampoline_early_out_mask_program();
    let n = native(&words, true, 64, 0);
    let lowered = lowered_with(&n, true, true).expect("trampoline early-out lowers");
    assert_eq!(lowered.wave_size, Some(32));
    if let Ok(mut v) = validator::Validator::load(validator::Validator::default_path().as_deref()) {
        v.validate(&mut lowered.dxil.clone())
            .expect("DXIL validation failed");
    }
    // Active lanes 0..45; intersection with 10 <= v0 < 40 is lanes 10..40 (30 lanes,
    // first lane = 10, high dword = bits 0..7 = 0xFF). Inactive lanes 45..64 stay 0.
    let want: Vec<u32> = (0..64)
        .flat_map(|l| if l < 45 { [30, 10, 0xFF] } else { [0, 0, 0] })
        .collect();
    let failed = differences_with(&words, 64, 3, &want, 0, true);
    assert!(failed.is_empty(), "{}", failed.join("\n"));
}

/// `VICTORY_THREAD_TICKET` defaults on and turns off with `"0"`.
#[test]
fn thread_ticket_env_toggle_defaults_on() {
    use std::ffi::OsStr;
    assert!(super::thread_ticket_value(None));
    assert!(super::thread_ticket_value(Some(OsStr::new("1"))));
    assert!(!super::thread_ticket_value(Some(OsStr::new("0"))));
    super::with_thread_ticket(false, || {
        assert!(!super::thread_ticket());
        super::with_thread_ticket(true, || {
            assert!(super::thread_ticket());
        });
        assert!(!super::thread_ticket());
    });
}

/// Multi-wave 2-D compute groups (32x2 = 64 threads) and partial-tail groups
/// (40x1 = 32 + 8 threads) with console lanes (`v_mbcnt`) claim aligned thread
/// blocks from the group-shared ticket counter so every console thread id in
/// the group is executed once on 32-lane hardware (issue #1243).
#[test]
fn thread_ticket_relabels_2d_and_partial_tail_groups_bijectively() {
    // Console wave32 program: triggers `console_lanes` via `v_mbcnt_lo_u32_b32`,
    // computes `slot = (y * 32 + x) * 4`, and stores `1` to raw UAV `s[40:43]`.
    let mut words_2d = PROLOGUE.to_vec();
    words_2d.extend([
        0xD765_0003,
        0x0001_007E, // v_mbcnt_lo_u32_b32 v3, exec_lo, 0
        0x3420_0285, // v_lshlrev_b32 v16, 5, v1 (y * 32)
        0x4A20_0110, // v_add_nc_u32 v16, v16, v0 (y * 32 + x)
        0x3420_2082, // v_lshlrev_b32 v16, 2, v16 (* 4)
        0x7E0A_0281, // v_mov_b32 v5, 1
        S_WAITCNT_LGKM,
    ]);
    words_2d.extend(store(5, 0));
    words_2d.push(S_ENDPGM);

    let mut prog_2d = native(&words_2d, false, 64, 0);
    prog_2d.io = StageIo::Cs(ComputeIo {
        // 1 user SGPR, thread id X and Y in v0, v1 (bit 11 = tid_comp_cnt >= 1).
        rsrc2: 1 << 1 | 1 << 11,
        threads: [32, 2, 1],
    });

    let lower_prog = |prog: &NativeProgram, ticket: bool| -> lower::Lowered {
        let p = ir::build(&prog.code, prog.wave64, prog.vgprs).expect("ir::build");
        let st = structure::structure(&p).expect("structure");
        let res = resource::track(&p, &prog.layout, prog.stage, Some(&st)).expect("resource");
        let io = io::raise(&p, prog).expect("io");
        super::with_thread_ticket(ticket, || {
            lower::lower(&lower::Parts {
                halves_rendezvous: false,
                native: prog,
                p: &p,
                st: &st,
                io: &io,
                res: &res,
                inputs: Vec::new(),
                outputs: Vec::new(),
                prims: Vec::new(),
                mesh_stride: 0,
                view_marks: false,
                atomic_locks: false,
                atomic_lock_rounds: false,
                atomic_umax: false,
                bias_as_gradients: false,
                host_wave32: true,
                narrow_waves: false,
                uav_size_level0: false,
            })
            .expect("lower")
        })
    };

    let with_ticket = lower_prog(&prog_2d, true);
    let without_ticket = lower_prog(&prog_2d, false);
    assert_ne!(
        with_ticket.dxil, without_ticket.dxil,
        "multi-wave console-lane group should emit the TGSM ticket counter"
    );
    #[cfg(windows)]
    if let Ok(mut v) = validator::Validator::load(validator::Validator::default_path().as_deref()) {
        v.validate(&mut with_ticket.dxil.clone()).expect("DXIL validation (2-D ticket)");
    }

    // Partial tail group: 40x1 (1 full wave of 32 + 1 partial wave of 8).
    let mut words_40 = PROLOGUE.to_vec();
    words_40.extend([
        0xD765_0003,
        0x0001_007E, // v_mbcnt_lo_u32_b32 v3, exec_lo, 0
        0x3420_0082, // v_lshlrev_b32 v16, 2, v0 (x * 4)
        0x7E0A_0281, // v_mov_b32 v5, 1
        S_WAITCNT_LGKM,
    ]);
    words_40.extend(store(5, 0));
    words_40.push(S_ENDPGM);
    let prog_40 = native(&words_40, false, 40, 0);
    let lowered_40 = lower_prog(&prog_40, true);
    #[cfg(windows)]
    if let Ok(mut v) = validator::Validator::load(validator::Validator::default_path().as_deref()) {
        v.validate(&mut lowered_40.dxil.clone()).expect("DXIL validation (40-thread partial tail)");
    }

    assert!(!lowered_40.dxil.is_empty());
    #[cfg(windows)]
    check_ticket_dispatches(with_ticket, lowered_40);
}

#[cfg(windows)]
fn check_ticket_dispatches(with_ticket: lower::Lowered, lowered_40: lower::Lowered) {
    let _one = super::valu_tests::ONE_AT_A_TIME
        .lock()
        .unwrap_or_else(|e| e.into_inner());
    let Ok(host) = Host::new(false) else {
        return; // no hardware D3D12 device
    };
    let run = |l: lower::Lowered, count: usize| -> Vec<u32> {
        let s = shader(l);
        let (root, pso) = host.compute_pipeline(&s).unwrap();
        let ud = vec![0u32; s.user_data_dwords().max(1) as usize];
        let out = host
            .dispatch(
                &root,
                &pso,
                &Dispatch {
                    roots: vec![(0, Root::Constants(ud))],
                    views: &[
                        View::Texture(1, 1, &[[0.0; 4]]),
                        View::Buffer {
                            data: &[0; 4],
                            stride: Stride::Structured(16),
                        },
                        View::RawUav(count as u32),
                    ],
                    samplers: &[],
                    groups: [1, 1, 1],
                },
            )
            .expect("dispatch");
        out[0][..count].to_vec()
    };

    let got_2d = run(with_ticket, 64);
    assert_eq!(got_2d, vec![1u32; 64], "all 64 threads of 32x2 group must run once");

    let got_40 = run(lowered_40, 40);
    assert_eq!(got_40, vec![1u32; 40], "all 40 threads (32 full + 8 partial) must run once");
}

// ---- the halves meet in divergent code (`lower::meet`) ----

/// Which lanes take a divergent branch: the even ones
/// (`v_and_b32 v3, 1, v0; v_cmp_eq_u32 vcc, 0, v3`), or those below a
/// thread id (`v_cmp_gt_u32 vcc, n, v0`).
#[derive(Clone, Copy)]
enum Side {
    Even,
    Below(u32),
}

impl Side {
    fn words(self) -> Vec<u32> {
        match self {
            Side::Even => vec![0x3606_0081, 0x7D84_0680],
            Side::Below(n) => vec![0x7D88_00FF, n],
        }
    }

    fn takes(self, l: u32) -> bool {
        match self {
            Side::Even => l.is_multiple_of(2),
            Side::Below(n) => l < n,
        }
    }
}

/// `v_cmp_gt_u32 vcc, limit, v0`, the console wave's count of those lanes
/// (`s_bcnt1_i32_b64 s12, vcc`) into `v1` and their high dword (`vcc_hi`)
/// into `v2`: a lane mask read as data where it is written.
fn ballot_below(limit: u32) -> [u32; 5] {
    [
        0x7D88_00FF,
        limit,       // v_cmp_gt_u32 vcc, limit, v0
        0xBE8C_106A, // s_bcnt1_i32_b64 s12, vcc
        0x7E02_020C, // v_mov_b32 v1, s12
        0x7E04_026B, // v_mov_b32 v2, vcc_hi
    ]
}

const S_AND_SAVEEXEC_S20_VCC: u32 = 0xBE94_246A; // s_and_saveexec_b64 s[20:21], vcc
const S_MOV_EXEC_S20: u32 = 0xBEFE_0414; // s_mov_b64 exec, s[20:21]
const S_ANDN2_EXEC_S20_EXEC: u32 = 0x8AFE_7E14; // s_andn2_b64 exec, s[20:21], exec
const fn s_cbranch_execz(skip: usize) -> u32 {
    0xBF88_0000 | skip as u32
}

/// `v1`, `v2` start at -1; in the lanes `side` takes, [`ballot_below`]`
/// (limit)` behind `s_cbranch_execz`; both stored. Per lane two dwords.
fn ballot_in_an_if(side: Side, limit: u32) -> Vec<u32> {
    let body = ballot_below(limit);
    let mut w = PROLOGUE.to_vec();
    w.extend([
        0x3420_0085, // v_lshlrev_b32 v16, 5, v0
        0x7E02_02C1, // v_mov_b32 v1, -1
        0x7E04_02C1, // v_mov_b32 v2, -1
    ]);
    w.extend(side.words());
    w.extend([S_AND_SAVEEXEC_S20_VCC, s_cbranch_execz(body.len())]);
    w.extend(body);
    w.push(S_MOV_EXEC_S20);
    w.extend(store(1, 0));
    w.extend(store(2, 4));
    w.push(S_ENDPGM);
    w
}

/// [`ballot_in_an_if`] with an else: the lanes `side` does not take run
/// [`ballot_below`]`(other)`, behind a branch of their own.
fn ballots_in_an_if_and_else(side: Side, limit: u32, other: u32) -> Vec<u32> {
    let (then, otherwise) = (ballot_below(limit), ballot_below(other));
    let mut w = PROLOGUE.to_vec();
    w.extend([
        0x3420_0085, // v_lshlrev_b32 v16, 5, v0
        0x7E02_02C1, // v_mov_b32 v1, -1
        0x7E04_02C1, // v_mov_b32 v2, -1
    ]);
    w.extend(side.words());
    w.extend([S_AND_SAVEEXEC_S20_VCC, s_cbranch_execz(then.len())]);
    w.extend(then);
    w.extend([S_ANDN2_EXEC_S20_EXEC, s_cbranch_execz(otherwise.len())]);
    w.extend(otherwise);
    w.push(S_MOV_EXEC_S20);
    w.extend(store(1, 0));
    w.extend(store(2, 4));
    w.push(S_ENDPGM);
    w
}

/// [`ballot_in_an_if`] in a scalar loop that runs twice (the branch block
/// runs twice, in step), each pass adding the count to `v1` (from 0):
/// `v1` is twice the count where the lane takes the branch.
fn ballot_in_an_if_in_a_loop(side: Side, limit: u32) -> Vec<u32> {
    let body = [
        0x7D88_00FF,
        limit,       // v_cmp_gt_u32 vcc, limit, v0
        0xBE8C_106A, // s_bcnt1_i32_b64 s12, vcc
        0x4A02_020C, // v_add_nc_u32 v1, s12, v1
    ];
    let mut w = PROLOGUE.to_vec();
    w.extend([
        0x3420_0085, // v_lshlrev_b32 v16, 5, v0
        0x7E02_0280, // v_mov_b32 v1, 0
        0xBE96_0380, // s_mov_b32 s22, 0
    ]);
    let top = w.len();
    w.extend(side.words());
    w.extend([S_AND_SAVEEXEC_S20_VCC, s_cbranch_execz(body.len())]);
    w.extend(body);
    w.extend([
        S_MOV_EXEC_S20,
        0x8016_8116, // s_add_u32 s22, s22, 1
        0xBF0A_8216, // s_cmp_lt_u32 s22, 2
    ]);
    let back = top as i32 - (w.len() as i32 + 1);
    w.push(0xBF85_0000 | (back as u32 & 0xFFFF)); // s_cbranch_scc1 (back to the top)
    w.extend(store(1, 0));
    w.push(S_ENDPGM);
    w
}

/// A scalar loop of two passes inside the branch: the region is a cycle,
/// where the halves cannot meet.
fn ballot_in_a_loop_in_an_if(side: Side, limit: u32) -> Vec<u32> {
    let mut body = vec![0xBE96_0380]; // s_mov_b32 s22, 0
    let top = body.len();
    body.extend([
        0x7D88_00FF,
        limit,       // v_cmp_gt_u32 vcc, limit, v0
        0xBE8C_106A, // s_bcnt1_i32_b64 s12, vcc
        0x4A02_020C, // v_add_nc_u32 v1, s12, v1
        0x8016_8116, // s_add_u32 s22, s22, 1
        0xBF0A_8216, // s_cmp_lt_u32 s22, 2
    ]);
    let back = top as i32 - (body.len() as i32 + 1);
    body.push(0xBF85_0000 | (back as u32 & 0xFFFF)); // s_cbranch_scc1
    let mut w = PROLOGUE.to_vec();
    w.extend([
        0x3420_0085, // v_lshlrev_b32 v16, 5, v0
        0x7E02_0280, // v_mov_b32 v1, 0
    ]);
    w.extend(side.words());
    w.extend([S_AND_SAVEEXEC_S20_VCC, s_cbranch_execz(body.len())]);
    w.extend(body);
    w.push(S_MOV_EXEC_S20);
    w.extend(store(1, 0));
    w.push(S_ENDPGM);
    w
}

/// The console's [`ballot_below`]`(limit)` in lane `l` of a group of
/// `threads` where the lanes `active` runs it: the count of its console
/// wave's active lanes below `limit`, and their high dword.
fn console_ballot(threads: u32, l: u32, active: impl Fn(u32) -> bool, limit: u32) -> [u32; 2] {
    let wave = l & !63;
    let on = |t: u32| t < threads && active(t) && t < limit;
    let count = (wave..wave + 64).filter(|&t| on(t)).count() as u32;
    let high = (0..32)
        .filter(|&k| on(wave + 32 + k))
        .fold(0u32, |m, k| m | 1 << k);
    [count, high]
}

/// The divergent test programs are what their comments say.
#[test]
fn the_divergent_ballot_programs_are_what_their_comments_say() {
    let programs: Vec<(Vec<u32>, &[&str])> = vec![
        (
            ballot_in_an_if(Side::Even, 50),
            &[
                "v_and_b32 v3, 1, v0",
                "v_cmp_eq_u32 vcc, 0, v3",
                "s_and_saveexec_b64 s[20:21], vcc",
                "s_cbranch_execz",
                "s_bcnt1_i32_b64 s12, vcc",
                "s_mov_b64 exec, s[20:21]",
            ],
        ),
        (
            ballots_in_an_if_and_else(Side::Even, 50, 40),
            &["s_andn2_b64 exec, s[20:21], exec", "v_mov_b32 v2, vcc_hi"],
        ),
        (
            ballot_in_an_if_in_a_loop(Side::Even, 50),
            &[
                "s_mov_b32 s22, 0",
                "v_add_nc_u32 v1, s12, v1",
                "s_add_u32 s22, s22, 1",
                "s_cmp_lt_u32 s22, 2",
                "s_cbranch_scc1",
            ],
        ),
    ];
    for (words, wants) in programs {
        let code: Vec<u8> = words.iter().flat_map(|w| w.to_le_bytes()).collect();
        let all = gfx10::decode(&code)
            .expect("decodes")
            .iter()
            .map(|i| i.text(true))
            .collect::<Vec<_>>()
            .join("\n");
        for want in wants {
            assert!(all.contains(want), "{want:?} not in:\n{all}");
        }
    }
}

/// A lane mask read as data in divergent code: the halves meet there
/// (`lower::meet`) in groups of more than one host wave, and the program
/// validates; a loop inside the branch keeps it refused.
#[test]
fn lane_masks_in_divergent_code_meet_between_halves() {
    let programs = [
        ballot_in_an_if(Side::Even, 50),
        ballot_in_an_if(Side::Below(20), 64),
        ballots_in_an_if_and_else(Side::Even, 50, 40),
        ballot_in_an_if_in_a_loop(Side::Below(40), 64),
    ];
    for words in &programs {
        for threads in [32, 64, 96, 128] {
            assert_eq!(refusal(words, threads, 0), "translated", "{threads} threads");
        }
        let halves = lowered(&native(words, true, 128, 0), true).expect("halves");
        assert_eq!(halves.wave_size, Some(32));
        if let Ok(mut v) = validator::Validator::load(validator::Validator::default_path().as_deref())
        {
            v.validate(&mut halves.dxil.clone())
                .expect("DXIL validation failed");
        }
    }
    let looped = ballot_in_a_loop_in_an_if(Side::Even, 50);
    // Refused before main's scalar mask numbers (`Lower::mask_number`),
    // which now give such a mask as the console does: either is right.
    let why = refusal(&looped, 64, 0);
    assert!(
        why == "translated" || why.contains("a lane mask read as data (wave64 on a 32-lane host)"),
        "{why}"
    );
    assert_eq!(refusal(&looped, 32, 0), "translated");
    // Without the rendezvous, refused as before.
    let n = native(&programs[0], true, 64, 0);
    let p = ir::build(&n.code, n.wave64, n.vgprs).expect("ir::build");
    let st = structure::structure(&p).expect("structure");
    let res = resource::track(&p, &n.layout, n.stage, Some(&st)).expect("track");
    let io = io::raise(&p, &n).expect("io::raise");
    let off = lower::lower(&lower::Parts {
        uav_size_level0: false,
        native: &n,
        p: &p,
        st: &st,
        io: &io,
        res: &res,
        inputs: Vec::new(),
        outputs: Vec::new(),
        prims: Vec::new(),
        mesh_stride: 0,
        view_marks: false,
        atomic_locks: false,
        atomic_lock_rounds: false,
        atomic_umax: false,
        bias_as_gradients: false,
        narrow_waves: false,
        halves_rendezvous: false,
        host_wave32: true,
    });
    let why = off.err().map(|e| e.to_string()).unwrap_or_else(|| "translated".into());
    assert!(
        why == "translated" || why.contains("a lane mask read as data (wave64 on a 32-lane host)"),
        "{why}"
    );
}

/// The divergent ballots on every hardware adapter give the console's
/// masks: some lanes of each half in the branch, only the lower half's
/// (the upper half meets no one and must not wait), both sides of an if
/// and else, two console waves diverging differently (128 threads), a
/// console wave with no upper half (96), and the branch in a loop run
/// twice (the epochs).
#[test]
fn lane_masks_in_divergent_code_match_the_console_on_every_gpu() {
    let mut failed = Vec::new();
    let cases = [
        (Side::Even, 50, 64),
        (Side::Below(20), 64, 64),
        (Side::Below(20), 64, 128),
        (Side::Below(80), 100, 128),
        (Side::Even, 90, 128),
        (Side::Even, 90, 96),
        (Side::Below(40), 70, 96),
    ];
    for (side, limit, threads) in cases {
        let want: Vec<u32> = (0..threads)
            .flat_map(|l| {
                if side.takes(l) {
                    console_ballot(threads, l, |t| side.takes(t), limit)
                } else {
                    [u32::MAX; 2]
                }
            })
            .collect();
        failed.extend(differences(&ballot_in_an_if(side, limit), threads, 2, &want));
    }
    for (side, limit, other, threads) in [
        (Side::Even, 50, 40, 64),
        (Side::Below(20), 64, 40, 64),
        (Side::Below(80), 100, 110, 128),
        (Side::Even, 90, 70, 96),
    ] {
        let want: Vec<u32> = (0..threads)
            .flat_map(|l| {
                if side.takes(l) {
                    console_ballot(threads, l, |t| side.takes(t), limit)
                } else {
                    console_ballot(threads, l, |t| !side.takes(t), other)
                }
            })
            .collect();
        failed.extend(differences(
            &ballots_in_an_if_and_else(side, limit, other),
            threads,
            2,
            &want,
        ));
    }
    for (side, limit, threads) in [
        (Side::Below(40), 64, 64),
        (Side::Even, 90, 128),
        (Side::Below(20), 64, 96),
    ] {
        let want: Vec<u32> = (0..threads)
            .map(|l| {
                if side.takes(l) {
                    2 * console_ballot(threads, l, |t| side.takes(t), limit)[0]
                } else {
                    0
                }
            })
            .collect();
        failed.extend(differences(
            &ballot_in_an_if_in_a_loop(side, limit),
            threads,
            1,
            &want,
        ));
    }
    assert!(failed.is_empty(), "{}", failed.join("\n"));
}

/// `v_readfirstlane_b32 s12, v0` inside an `s_and_saveexec_b64` region.
///
/// * `mask_words`: computes `vcc` selecting the active lanes (e.g. lanes 8..=9,
///   40..=41, 12 & 44, or empty).
/// * `with_mbcnt`: include `v_mbcnt_lo_u32_b32` so the program requires
///   console lanes (`halves` mode on 32-lane hosts).
/// * `with_branch`: wrap the `v_readfirstlane_b32` in `s_cbranch_execz` so
///   the read sits in a divergent basic block rather than a predicated line.
fn readfirstlane_in_branch(mask_words: &[u32], with_mbcnt: bool, with_branch: bool) -> Vec<u32> {
    let mut w = PROLOGUE.to_vec();
    w.push(0x3420_0085); // v_lshlrev_b32 v16, 5, v0
    if with_mbcnt {
        w.extend([
            0xD765_0001,
            0x0001_00C1, // v_mbcnt_lo_u32_b32 v1, -1, 0
        ]);
    }
    w.extend(mask_words);
    w.push(0xBE80_246A); // s_and_saveexec_b64 s[0:1], vcc
    if with_branch {
        w.push(0xBE8C_0380); // s_mov_b32 s12, 0
        w.push(0xBF88_0001); // s_cbranch_execz +1 (skip v_readfirstlane_b32)
    }
    w.push(0x7E18_0500); // v_readfirstlane_b32 s12, v0
    w.push(0xBEFE_0400); // s_mov_b64 exec, s[0:1]
    w.push(0x7E04_020C); // v_mov_b32 v2, s12
    w.extend(store(2, 0));
    w.push(S_ENDPGM);
    w
}

/// Whole-wave `v_add_f64` active-lane count (`mask ? 1.0_f64 : 0.0_f64`)
/// followed by `v_cvt_f32_f64`: counts lanes in `[lo, hi)` within each 64-lane
/// console wave, setting `HW_REG_MODE` bits 3:2 to `round_mode` inside the
/// section, and writes `[f64_lo, f64_hi, f32_bits, 0]` per lane.
#[cfg(windows)]
fn f64_count_reduction(lo: u32, hi: u32, round_mode: u32) -> Vec<u32> {
    assert!(lo <= 64 && hi <= 64);
    let mut w = PROLOGUE.to_vec();
    w.extend([
        0x3420_0085, // v_lshlrev_b32 v16, 5, v0 (byte offset = lane * 32)
        0x3612_00BF, // v_and_b32 v9, 63, v0
        0xD4C6_000A,
        ((128 + lo) << 9) | 265, // v_cmp_ge_u32 s[10:11], v9, lo
        0xD4C1_000C,
        ((128 + hi) << 9) | 265, // v_cmp_lt_u32 s[12:13], v9, hi
        0x878A_0C0A,             // s_and_b64 s[10:11], s[10:11], s[12:13]
        0x7E26_0280,             // v_mov_b32 v19, 0
        0x7E28_02FF,
        0x3FF0_0000, // v_mov_b32 v20, lit(0x3ff00000) -> 1.0_f64
        0xBE86_25C1, // s_or_saveexec_b64 s[6:7], -1
        0x7E04_0280, // v_mov_b32 v2, 0
        0x7E06_0280, // v_mov_b32 v3, 0
        0x8788_060A, // s_and_b64 s[8:9], s[10:11], s[6:7]
        0xD501_0002,
        0x0022_2702, // v_cndmask_b32 v2, v2, v19, s[8:9]
        0xD501_0003,
        0x0022_2903, // v_cndmask_b32 v3, v3, v20, s[8:9]
        0x7E08_02E9,
        0xDE54_C102, // v_mov_b32 v4, v2 dpp8:[1,0,3,2,5,4,7,6]
        0x7E0A_02E9,
        0xDE54_C103, // v_mov_b32 v5, v3 dpp8:[1,0,3,2,5,4,7,6]
        0xBA80_0881,
        round_mode, // s_setreg_imm32_b32 hwreg(HW_REG_MODE, 2, 2), round_mode
        0xD564_0002,
        0x0002_0504, // v_add_f64 v[2:3], v[4:5], v[2:3]
        0x7E08_02E9,
        0x9360_1202, // v_mov_b32 v4, v2 dpp8:[2,2,0,0,6,6,4,4]
        0x7E0A_02E9,
        0x9360_1203, // v_mov_b32 v5, v3 dpp8:[2,2,0,0,6,6,4,4]
        0xD564_0002,
        0x0002_0504, // v_add_f64 v[2:3], v[4:5], v[2:3]
        0x7E08_02E9,
        0x0009_2402, // v_mov_b32 v4, v2 dpp8:[4,4,4,4,0,0,0,0]
        0x7E0A_02E9,
        0x0009_2403, // v_mov_b32 v5, v3 dpp8:[4,4,4,4,0,0,0,0]
        0xD564_0002,
        0x0002_0504, // v_add_f64 v[2:3], v[4:5], v[2:3]
        0x7E08_02FA,
        0xFF09_4002, // v_mov_b32 v4, v2 row_mirror bound_ctrl:1
        0x7E0A_02FA,
        0xFF09_4003, // v_mov_b32 v5, v3 row_mirror bound_ctrl:1
        0xD564_0002,
        0x0002_0504, // v_add_f64 v[2:3], v[4:5], v[2:3]
        0xD778_0004,
        0x0201_0102, // v_permlanex16_b32 v4, v2, 0, 0
        0xD778_0005,
        0x0201_0103, // v_permlanex16_b32 v5, v3, 0, 0
        0xD564_0002,
        0x0002_0504, // v_add_f64 v[2:3], v[4:5], v[2:3]
        0xD760_0008,
        0x0001_4102, // v_readlane_b32 s8, v2, 32
        0xD760_0009,
        0x0001_4103, // v_readlane_b32 s9, v3, 32
        0xD564_0002,
        0x0002_0408, // v_add_f64 v[2:3], s[8:9], v[2:3]
        0xD760_0008,
        0x0001_0102, // v_readlane_b32 s8, v2, 0
        0xD760_0009,
        0x0001_0103, // v_readlane_b32 s9, v3, 0
        0xBE8C_0308, // s_mov_b32 s12, s8
        0xBE8D_0309, // s_mov_b32 s13, s9
        0xBEFE_0406, // s_mov_b64 exec, s[6:7]
        0x7E0C_020C, // v_mov_b32 v6, s12
        0x7E0E_020D, // v_mov_b32 v7, s13
        0x7E10_1E0C, // v_cvt_f32_f64 v8, s[12:13]
        0x7E12_0280, // v_mov_b32 v9, 0
    ]);
    for (vdata, slot) in [(6, 0), (7, 1), (8, 2), (9, 3)] {
        w.extend(store(vdata, 4 * slot));
    }
    w.push(S_ENDPGM);
    w
}

/// Active-lane predicates for [`readfirstlane_in_branch`]:
/// - lower half only: `(v0 >> 1) == 4` -> lanes 8, 9 (first = 8)
/// - upper half only: `(v0 >> 1) == 20` -> lanes 40, 41 (first = 40)
/// - both halves: `(v0 & 31) == 12` -> lanes 12, 44 (first = 12)
/// - empty EXEC: `(v0 >> 1) == 64` -> no lanes active (first = lane 0 = 0, or
///   with `v0 + 7` -> 7, distinguishing lane 0 from 0 and from lane 63)
const MASK_LANES_8_9: [u32; 2] = [
    0x2C06_0081, // v_lshrrev_b32 v3, 1, v0
    0x7D84_0684, // v_cmp_eq_u32 vcc, 4, v3
];

const MASK_LANES_40_41: [u32; 2] = [
    0x2C06_0081, // v_lshrrev_b32 v3, 1, v0
    0x7D84_0694, // v_cmp_eq_u32 vcc, 20, v3
];

const MASK_LANES_12_44: [u32; 2] = [
    0x3606_009F, // v_and_b32 v3, 31, v0
    0x7D84_068C, // v_cmp_eq_u32 vcc, 12, v3
];

const MASK_EMPTY: [u32; 2] = [
    0x2C06_0081, // v_lshrrev_b32 v3, 1, v0
    0x7D84_06C0, // v_cmp_eq_u32 vcc, 64, v3
];

/// Same as [`readfirstlane_in_branch`] but reads `v4 = v0 + 7` so lane 0 has
/// value `7` (distinguishable from zero and from lane 31/63).
fn readfirstlane_offset_in_branch(
    mask_words: &[u32],
    with_mbcnt: bool,
    with_branch: bool,
) -> Vec<u32> {
    let mut w = PROLOGUE.to_vec();
    w.push(0x3420_0085); // v_lshlrev_b32 v16, 5, v0
    w.push(0x4A08_0087); // v_add_nc_u32 v4, 7, v0
    if with_mbcnt {
        w.extend([
            0xD765_0001,
            0x0001_00C1, // v_mbcnt_lo_u32_b32 v1, -1, 0
        ]);
    }
    w.extend(mask_words);
    w.push(0xBE80_246A); // s_and_saveexec_b64 s[0:1], vcc
    if with_branch {
        w.push(0xBE8C_0387); // s_mov_b32 s12, 7
        w.push(0xBF88_0001); // s_cbranch_execz +1 (skip v_readfirstlane_b32)
    }
    w.push(0x7E18_0504); // v_readfirstlane_b32 s12, v4
    w.push(0xBEFE_0400); // s_mov_b64 exec, s[0:1]
    w.push(0x7E04_020C); // v_mov_b32 v2, s12
    w.extend(store(2, 0));
    w.push(S_ENDPGM);
    w
}

#[test]
fn the_readfirstlane_branch_programs_are_what_their_comments_say() {
    let programs: Vec<(Vec<u32>, &[&str])> = vec![
        (
            readfirstlane_in_branch(&MASK_LANES_8_9, false, false),
            &[
                "v_lshlrev_b32 v16, 5, v0",
                "v_lshrrev_b32 v3, 1, v0",
                "v_cmp_eq_u32 vcc, 4, v3",
                "s_and_saveexec_b64 s[0:1], vcc",
                "v_readfirstlane_b32 s12, v0",
                "s_mov_b64 exec, s[0:1]",
                "v_mov_b32 v2, s12",
            ],
        ),
        (
            readfirstlane_in_branch(&MASK_LANES_40_41, true, true),
            &[
                "v_mbcnt_lo_u32_b32 v1, -1, 0",
                "v_cmp_eq_u32 vcc, 20, v3",
                "s_cbranch_execz",
                "v_readfirstlane_b32 s12, v0",
            ],
        ),
        (
            readfirstlane_offset_in_branch(&MASK_LANES_12_44, true, false),
            &[
                "v_add_nc_u32 v4, 7, v0",
                "v_and_b32 v3, 31, v0",
                "v_cmp_eq_u32 vcc, 12, v3",
                "v_readfirstlane_b32 s12, v4",
            ],
        ),
        (
            readfirstlane_offset_in_branch(&MASK_EMPTY, true, false),
            &[
                "v_cmp_eq_u32 vcc, 64, v3",
                "v_readfirstlane_b32 s12, v4",
            ],
        ),
    ];
    for (words, wants) in programs {
        let code: Vec<u8> = words.iter().flat_map(|w| w.to_le_bytes()).collect();
        let all = gfx10::decode(&code)
            .expect("decodes")
            .iter()
            .map(|i| i.text(true))
            .collect::<Vec<_>>()
            .join("\n");
        for want in wants {
            assert!(all.contains(want), "{want:?} not in:\n{all}");
        }
    }
}

#[test]
#[cfg(windows)]
fn readfirstlane_in_branch_matches_the_console_on_every_gpu() {
    let mut failed = Vec::new();
    for (label, mask, want_val) in [
        ("lower_8_9", &MASK_LANES_8_9, 8u32),
        ("upper_40_41", &MASK_LANES_40_41, 40u32),
        ("both_12_44", &MASK_LANES_12_44, 12u32),
        ("empty_exec", &MASK_EMPTY, 0u32),
    ] {
        for with_mbcnt in [false, true] {
            for with_branch in [false, true] {
                let words = readfirstlane_offset_in_branch(mask, with_mbcnt, with_branch);
                let n = native(&words, true, 64, 0);
                let l = lowered(&n, true).expect("readfirstlane branch lowers in halves");
                assert_eq!(l.wave_size, Some(32));
                if let Ok(mut v) =
                    validator::Validator::load(validator::Validator::default_path().as_deref())
                {
                    v.validate(&mut l.dxil.clone())
                        .expect("DXIL validation failed");
                }
                let want = vec![want_val + 7; 64];
                for msg in differences(&words, 64, 1, &want) {
                    failed.push(format!(
                        "{label}(mbcnt={with_mbcnt},branch={with_branch}): {msg}"
                    ));
                }
            }
        }
    }
    // Direct `v_readfirstlane_b32 s12, v0` without `v_mbcnt` or `s_cbranch_execz`:
    for (label, mask, want_val) in [
        ("direct_v0_8_9", &MASK_LANES_8_9, 8u32),
        ("direct_v0_40_41", &MASK_LANES_40_41, 40u32),
        ("direct_v0_12_44", &MASK_LANES_12_44, 12u32),
        ("direct_v0_empty", &MASK_EMPTY, 0u32),
    ] {
        let words = readfirstlane_in_branch(mask, false, false);
        let want = vec![want_val; 64];
        for msg in differences(&words, 64, 1, &want) {
            failed.push(format!("{label}: {msg}"));
        }
    }
    // Single-host-wave (32), partial-tail (96), and multi-wave (128) groups:
    for (label, mask, threads, expected) in [
        ("wave32_8_9", &MASK_LANES_8_9, 32u32, (0..32).map(|_| 8 + 7).collect::<Vec<_>>()),
        ("wave32_empty", &MASK_EMPTY, 32u32, (0..32).map(|_| 7).collect::<Vec<_>>()),
        (
            "wave96_both_12_44",
            &MASK_LANES_12_44,
            96u32,
            (0..96).map(|l| (l & !63) + 12 + 7).collect::<Vec<_>>(),
        ),
        (
            "wave96_empty",
            &MASK_EMPTY,
            96u32,
            (0..96).map(|l| (l & !63) + 7).collect::<Vec<_>>(),
        ),
        (
            "wave128_both_12_44",
            &MASK_LANES_12_44,
            128u32,
            (0..128).map(|l| (l & !63) + 12 + 7).collect::<Vec<_>>(),
        ),
        (
            "wave128_empty",
            &MASK_EMPTY,
            128u32,
            (0..128).map(|l| (l & !63) + 7).collect::<Vec<_>>(),
        ),
    ] {
        let words = readfirstlane_offset_in_branch(mask, false, false);
        for msg in differences(&words, threads, 1, &expected) {
            failed.push(format!("{label}: {msg}"));
        }
    }
    // Literal whole-wave halves with empty EXEC must also return lane 0's value.
    let empty_lit = readfirstlane_offset_in_branch(&MASK_EMPTY, true, false);
    for msg in differences_in(true, &empty_lit, 64, 1, &[7; 64], 0, true, true) {
        failed.push(format!("literal_empty: {msg}"));
    }
    assert!(failed.is_empty(), "{}", failed.join("\n"));
}

/// Whole-wave `v_add_f64` active-lane count + `v_cvt_f32_f64` matches exact
/// IEEE-754 `f64` `(lo, hi)` and `f32` results across empty, lower-half-only,
/// upper-half-only, boundary-straddling, and full-wave masks (including a
/// 96-thread group whose second console wave has no upper half), and refuses
/// non-nearest rounding modes inside the whole-wave section.
#[test]
#[cfg(windows)]
fn v_add_f64_count_reduction_and_cvt_f32_f64_match_console_on_every_gpu() {
    // Non-nearest rounding mode (RTZ = 3) inside the whole-wave section must be refused.
    assert_ne!(
        refusal(&f64_count_reduction(0, 64, 3), 64, 0),
        "translated",
        "non-nearest rounding mode in v_add_f64 whole-wave section must be refused"
    );

    let cases: &[(u32, u32, u32)] = &[
        (0, 0, 64),   // empty mask (0.0)
        (0, 13, 64),  // lower-half-only (13.0)
        (41, 64, 64), // upper-half-only (23.0)
        (25, 40, 64), // straddling 32-lane boundary (15.0)
        (0, 47, 64),  // straddling boundary (47.0)
        (0, 64, 64),  // full wave (64.0)
        (10, 50, 96), // wave 0: 40.0, wave 1 (32 threads, no upper half): 22.0
    ];
    let mut failed = Vec::new();
    for &(lo, hi, threads) in cases {
        let words = f64_count_reduction(lo, hi, 0);
        let want: Vec<u32> = (0..threads)
            .flat_map(|l| {
                let wave_base = l & !63;
                let wave_lanes = (threads - wave_base).min(64);
                let count = hi.min(wave_lanes).saturating_sub(lo.min(wave_lanes));
                let bits = f64::from(count).to_bits();
                [bits as u32, (bits >> 32) as u32, (count as f32).to_bits(), 0]
            })
            .collect();
        failed.extend(differences(&words, threads, 4, &want));
        if threads == 64 {
            // Also verify under literal wave64 halves (`self.x.st.literal == true`).
            failed.extend(differences_in(true, &words, threads, 4, &want, 0, false, true));
        }
    }
    assert!(failed.is_empty(), "{}", failed.join("\n"));
}

/// Each lane that runs adds 1 to `counter[local x]` (the raw UAV).
fn counted_lanes(extra: &[u32]) -> Vec<u32> {
    let mut w = PROLOGUE.to_vec();
    w.extend(extra);
    w.extend([
        0x3402_0082, // v_lshlrev_b32 v1, 2, v0
        0x7E04_0281, // v_mov_b32 v2, 1
        0xE0C8_1000,
        0x800A_0201, // buffer_atomic_add v2, v1, s[40:43], 0 offen
        S_ENDPGM,
    ]);
    w
}

/// The user-data word a translation reads for a dispatch counted in
/// threads, if any.
fn thread_limit(l: &lower::Lowered) -> Option<super::UserData> {
    let code = RootType::Other(super::USER_DISPATCH_THREADS_X);
    l.user_data.iter().copied().find(|u| u.root == code)
}

/// Which programs leave the lanes past a dispatch's count of threads:
/// those of one dimension whose lanes never wait for each other, not in
/// 32-lane halves that pass values nor in literal halves, and only when the
/// translation asks for it; the word is past the console's user data and
/// names the group's width.
#[test]
fn only_programs_whose_lanes_never_wait_leave_past_a_count_of_threads() {
    let limited = |words: &[u32], wave64, threads, host_wave32| {
        let n = native(words, wave64, threads, 0);
        let l = lower_halves_limited(&n, host_wave32, true, true, true).expect("translates");
        thread_limit(&l)
    };
    let plain = counted_lanes(&[]);
    let word = limited(&plain, false, 64, true).expect("a word for the count");
    assert_eq!(
        (word.index, word.dword),
        (1, 64),
        "past the one user-data SGPR"
    );
    assert_eq!(
        limited(&plain, true, 64, true).map(|u| u.dword),
        Some(64),
        "wave64 in 32-lane waves"
    );
    let n = native(&plain, false, 64, 0);
    assert_eq!(
        thread_limit(&lower_halves(&n, true, true, false).unwrap()),
        None,
        "not asked for"
    );
    assert_eq!(
        limited(&plain, false, 1, true),
        None,
        "a group of one thread"
    );
    let mut flat = native(&plain, false, 64, 0);
    flat.io = StageIo::Cs(ComputeIo {
        rsrc2: 1 << 1,
        threads: [8, 8, 1],
    });
    let l = lower_halves_limited(&flat, true, true, true, true).unwrap();
    assert_eq!(thread_limit(&l), None, "two dimensions");
    // s_barrier: lanes that left would never reach it.
    assert_eq!(
        limited(&counted_lanes(&[0xBF8A_0000]), false, 64, true),
        None,
        "a group barrier"
    );
    // v_mbcnt reads console lanes: a wave64 group of 64 runs in halves,
    // which pass values between them at barriers; one of 32 does not.
    let lanes = counted_lanes(&[0xD765_0003, 0x0001_00C1]);
    assert_eq!(limited(&lanes, true, 64, true), None, "in halves");
    assert!(limited(&lanes, true, 32, true).is_some(), "one host wave");
    assert!(
        limited(&lanes, true, 64, false).is_some(),
        "one host wave64"
    );
    // A lane read across the halves translates only as literal halves,
    // whose group must take one path: no word there.
    let n = native(&literal_lane_reads(false), true, 64, 0);
    let l = lower_halves_limited(&n, true, true, true, true).expect("literal halves");
    assert_eq!(thread_limit(&l), None, "literal halves");
    let code: Vec<u8> = lanes.iter().flat_map(|w| w.to_le_bytes()).collect();
    let text: Vec<String> = gfx10::decode(&code)
        .unwrap()
        .iter()
        .map(|i| i.text(true))
        .collect();
    for want in [
        "v_mbcnt_lo_u32_b32 v3, -1, 0",
        "v_lshlrev_b32 v1, 2, v0",
        "v_mov_b32 v2, 1",
        "buffer_atomic_add v2, v1, s[40:43], 0 offen",
    ] {
        assert!(
            text.iter().any(|t| t.contains(want)),
            "{want:?} not in {text:?}"
        );
    }
    let code: Vec<u8> = [0xBF8A_0000u32, S_ENDPGM]
        .iter()
        .flat_map(|w| w.to_le_bytes())
        .collect();
    assert_eq!(gfx10::decode(&code).unwrap()[0].text(true), "s_barrier");
}

/// A dispatch counted in threads (`DispatchX`, after DexNom/VIctory
/// bc3cd4b0): of two groups of 64 only the lanes below the count run, in
/// wave32 and wave64 programs, and a count of 0 runs them all. Each lane
/// that runs adds 1 to the counter of its local x. On hardware devices
/// only, as the other halves tests that run programs.
#[cfg(windows)]
#[test]
fn the_lanes_past_a_count_of_threads_do_not_run() {
    let _one = super::valu_tests::ONE_AT_A_TIME
        .lock()
        .unwrap_or_else(|e| e.into_inner());
    let hosts = Host::each_hardware();
    if hosts.is_empty() {
        eprintln!("no SM 6.6 hardware device; skipped");
        return;
    }
    let views = [
        View::Texture(1, 1, &[[0.0; 4]]),
        View::Buffer {
            data: &[0; 4],
            stride: Stride::Structured(16),
        },
        View::RawUav(64),
    ];
    for (adapter, host) in &hosts {
        let wave32 = host.wave32().unwrap_or(false);
        for wave64 in [false, true] {
            let n = native(&counted_lanes(&[]), wave64, 64, 0);
            let l = lower_halves_limited(&n, wave32, wave32, true, true).expect("translates");
            let limit = thread_limit(&l).expect("a word for the count");
            let shader = shader(l);
            if let Ok(mut v) =
                validator::Validator::load(validator::Validator::default_path().as_deref())
            {
                v.validate(&mut shader.dxil.clone())
                    .expect("DXIL validation failed");
            }
            let label = format!("{adapter} (wave64 {wave64})");
            let Ok((root, pso)) = host.compute_pipeline(&shader) else {
                panic!("{label}: no pipeline");
            };
            for count in [0u32, 1, 7, 31, 32, 33, 63, 64, 65, 74, 127, 128] {
                let mut ud = vec![0; shader.user_data_dwords() as usize];
                ud[limit.index as usize] = count;
                let out = host
                    .dispatch(
                        &root,
                        &pso,
                        &Dispatch {
                            roots: vec![(0, Root::Constants(ud))],
                            views: &views,
                            samplers: &[],
                            groups: [2, 1, 1],
                        },
                    )
                    .unwrap_or_else(|e| panic!("{label}: dispatch failed: {e}"));
                let runs = if count == 0 { 128 } else { count };
                let want: Vec<u32> = (0..64)
                    .map(|i| u32::from(i < runs) + u32::from(i + 64 < runs))
                    .collect();
                assert_eq!(out[0], want, "{label}: count {count}");
            }
            eprintln!("{label}: the lanes past every count stayed out");
        }
    }
}

/// A program whose only cross-half operations are first-lane reads goes to
/// halves for them (#522). Where halves cannot pass the read between them,
/// here in a loop counted to the console wave's first lane, it is refused,
/// with or without console lanes asked for as well: each 32-lane host wave
/// reading its own first lane would count the upper half's loop to thread
/// 32 and store 32 there, where the console stores 0
/// ([`a_first_lane_read_in_a_loop_matches_the_console_on_every_gpu`]).
#[test]
fn first_lane_reads_halves_cannot_pass_are_refused() {
    let words = first_lane_in_a_counted_loop();
    let with_mbcnt = in_a_loop(&[0x7E18_0500], 12, Count::FirstLane);
    for threads in [64, 128, 256] {
        for program in [&words, &with_mbcnt] {
            let why = refusal(program, threads, 0);
            assert!(
                why.contains("a first-lane read of a value that may differ between 32-lane halves"),
                "{threads} threads: {why}"
            );
        }
    }
}

/// [`in_a_loop`] of `v_readfirstlane_b32 s12, v0` counted to the first
/// lane, without its `v_mbcnt`: a program whose only cross-half operations
/// are first-lane reads, one of them in a loop.
fn first_lane_in_a_counted_loop() -> Vec<u32> {
    let mbcnt = [0xD765_0001, 0x0001_00C1]; // v_mbcnt_lo_u32_b32 v1, -1, 0
    let first = [0x7E18_0500]; // v_readfirstlane_b32 s12, v0
    let with_mbcnt = in_a_loop(&first, 12, Count::FirstLane);
    let at = with_mbcnt
        .windows(2)
        .position(|w| w == mbcnt)
        .expect("in_a_loop's v_mbcnt");
    [&with_mbcnt[..at], &with_mbcnt[at + 2..]].concat()
}

/// A first-lane read the halves cannot pass between them is refused, never
/// made by each 32-lane host wave from its own first lane: on the console
/// every lane of a wave64 reads the wave's first lane, thread `l & !63`,
/// where the upper host wave's own first lane is thread `l & !31`.
#[cfg(windows)]
#[test]
fn a_first_lane_read_in_a_loop_matches_the_console_on_every_gpu() {
    let words = first_lane_in_a_counted_loop();
    let mut failed = Vec::new();
    for threads in [64, 128] {
        match lowered(&native(&words, true, threads, 0), true) {
            Err(e) => {
                let why = e.to_string();
                let across = "a first-lane read of a value that may differ between 32-lane halves";
                assert!(why.contains(across), "{threads} threads: {why}");
            }
            Ok(_) => {
                let want: Vec<u32> = (0..threads).map(|l| l & !63).collect();
                failed.extend(differences(&words, threads, 1, &want));
            }
        }
    }
    assert!(failed.is_empty(), "{}", failed.join("\n"));
}

/// `v_readfirstlane_b32 s12, v4` inside an `s_and_saveexec_b64` region
/// whose EXEC is `mask_words`'s, of `v4 = f32(s14) + 1.0` made there, with
/// `s14` the first dword of the SRV buffer at table dword 8 (a scalar load,
/// as in GTA VI's reciprocals of constant-buffer divisors): the same in
/// every lane EXEC has. `inside`: `v2 = s12` under the region's EXEC (only
/// lanes that ran the read see it), else after EXEC is restored (every
/// lane).
fn readfirstlane_of_a_scalar_in_branch(
    mask_words: &[u32],
    with_mbcnt: bool,
    inside: bool,
) -> Vec<u32> {
    let mut w = PROLOGUE.to_vec();
    w.extend([
        0xF408_0B02,
        0xFA00_0020, // s_load_dwordx4 s[44:47], s[4:5], 0x20
        S_WAITCNT_LGKM,
        0xF420_0396,
        0xFA00_0000, // s_buffer_load_dword s14, s[44:47], 0x0
        S_WAITCNT_LGKM,
        0x3420_0085, // v_lshlrev_b32 v16, 5, v0
        0x7E04_0280, // v_mov_b32 v2, 0
    ]);
    if with_mbcnt {
        w.extend([
            0xD765_0001,
            0x0001_00C1, // v_mbcnt_lo_u32_b32 v1, -1, 0
        ]);
    }
    w.extend(mask_words);
    w.push(0xBE80_246A); // s_and_saveexec_b64 s[0:1], vcc
    let region = [
        0x7E08_0C0E, // v_cvt_f32_u32 v4, s14
        0x0608_08F2, // v_add_f32 v4, 1.0, v4
        0x7E18_0504, // v_readfirstlane_b32 s12, v4
    ];
    let skip = region.len() as u32 + u32::from(inside);
    w.push(0xBF88_0000 | skip); // s_cbranch_execz (past the region)
    w.extend(region);
    if inside {
        w.push(0x7E04_020C); // v_mov_b32 v2, s12
    }
    w.push(0xBEFE_0400); // s_mov_b64 exec, s[0:1]
    if !inside {
        w.push(0x7E04_020C); // v_mov_b32 v2, s12
    }
    w.extend(store(2, 0));
    w.push(S_ENDPGM);
    w
}

#[test]
fn the_scalar_readfirstlane_programs_are_what_their_comments_say() {
    for inside in [false, true] {
        let words = readfirstlane_of_a_scalar_in_branch(&MASK_LANES_40_41, true, inside);
        let code: Vec<u8> = words.iter().flat_map(|w| w.to_le_bytes()).collect();
        let all = gfx10::decode(&code)
            .expect("decodes")
            .iter()
            .map(|i| i.text(true))
            .collect::<Vec<_>>()
            .join("\n");
        for want in [
            "s_load_dwordx4 s[44:47], s[4:5], 0x000020",
            "s_buffer_load_dword s14, s[44:47]",
            "v_cmp_eq_u32 vcc, 20, v3",
            "s_and_saveexec_b64 s[0:1], vcc",
            "s_cbranch_execz",
            "v_cvt_f32_u32 v4, s14",
            "v_add_f32 v4, 1.0, v4",
            "v_readfirstlane_b32 s12, v4",
            "v_mov_b32 v2, s12",
        ] {
            assert!(all.contains(want), "{want:?} not in:\n{all}");
        }
    }
}

/// A first-lane read of a value every lane EXEC has computed alike, whose
/// result only those lanes see, is made by each host half by itself: its
/// own first active lane, with no value passed between the halves (none of
/// an exchange's group barriers), so it translates also where the halves
/// cannot meet (#522 refused it there). Seen by lanes outside the read's
/// EXEC, it still passes the halves' first lanes and ballots between them.
#[test]
fn a_first_lane_read_only_active_lanes_see_is_made_in_each_half() {
    for mask in [&MASK_LANES_8_9, &MASK_LANES_40_41, &MASK_LANES_12_44] {
        let inside = readfirstlane_of_a_scalar_in_branch(mask, true, true);
        let alone = lowered(&native(&inside, true, 64, 0), true).expect("read alone in halves");
        assert_eq!(alone.wave_size, Some(32));
        let outside = readfirstlane_of_a_scalar_in_branch(mask, true, false);
        let across = lowered(&native(&outside, true, 64, 0), true).expect("read across in halves");
        // At least two exchanges (the value, the ballot) of two barriers.
        assert!(
            across.barriers >= alone.barriers + 4,
            "{} {}",
            across.barriers,
            alone.barriers
        );
    }
}

/// [`readfirstlane_of_a_scalar_in_branch`] read inside its region gives the
/// console's lanes on every hardware adapter: `f32(41) + 1.0` in the lanes
/// EXEC has, 0 elsewhere, with the active lanes in the lower half only, in
/// the upper half only (the lower host wave has none), in both, or none (the
/// region is skipped); with console lanes needed elsewhere (halves) and not.
#[test]
#[cfg(windows)]
fn a_first_lane_read_only_active_lanes_see_matches_the_console_on_every_gpu() {
    let value = (41.0f32 + 1.0).to_bits();
    let lower: fn(u32) -> bool = |l| l >> 1 == 4;
    let upper: fn(u32) -> bool = |l| l >> 1 == 20;
    let both: fn(u32) -> bool = |l| l & 31 == 12;
    let none: fn(u32) -> bool = |_| false;
    let mut failed = Vec::new();
    for (label, mask, active) in [
        ("lower_8_9", &MASK_LANES_8_9, lower),
        ("upper_40_41", &MASK_LANES_40_41, upper),
        ("both_12_44", &MASK_LANES_12_44, both),
        ("empty", &MASK_EMPTY, none),
    ] {
        for with_mbcnt in [false, true] {
            let words = readfirstlane_of_a_scalar_in_branch(mask, with_mbcnt, true);
            let want: Vec<u32> = (0..64).map(|l| if active(l) { value } else { 0 }).collect();
            let found = differences_with_buffer(
                true,
                &words,
                64,
                1,
                &want,
                0,
                false,
                false,
                &[41, 0, 0, 0],
            );
            failed.extend(
                found
                    .into_iter()
                    .map(|msg| format!("{label}(mbcnt={with_mbcnt}): {msg}")),
            );
        }
    }
    assert!(failed.is_empty(), "{}", failed.join("\n"));
}

/// A first-lane read of a value a multiply-accumulate made, in an
/// `s_and_saveexec_b64` region whose EXEC is `mask_words`'s, its result
/// used in the region only (as GTA VI's `ef154ebc9e03d1b1` reads a function
/// of a constant-buffer value): `v4 = f32(s14)`, `s14` the first dword of
/// the SRV buffer at table dword 8, then `v_mac_f32 v4, v7, v8` with
/// `v7 = 2.0`, so `v4 += 2.0 * v8`. `v8` is 2.0 (the same in every lane of
/// the region), or with `per_lane` the lane's thread id. The region's lanes
/// store the read; the others 0.
fn mac_read_in_a_region(mask_words: &[u32], per_lane: bool) -> Vec<u32> {
    let mut w = PROLOGUE.to_vec();
    w.extend([
        0xF408_0B02,
        0xFA00_0020, // s_load_dwordx4 s[44:47], s[4:5], 0x20
        S_WAITCNT_LGKM,
        0xF420_0396,
        0xFA00_0000, // s_buffer_load_dword s14, s[44:47], 0x0
        S_WAITCNT_LGKM,
        0x3420_0085, // v_lshlrev_b32 v16, 5, v0
        0x7E04_0280, // v_mov_b32 v2, 0
    ]);
    w.extend(mask_words);
    w.extend([
        0xBE80_246A, // s_and_saveexec_b64 s[0:1], vcc
        0xBF88_0006, // s_cbranch_execz (past the region)
        0x7E08_0C0E, // v_cvt_f32_u32 v4, s14
        0x7E0E_02F4, // v_mov_b32 v7, 2.0
        if per_lane {
            0x7E10_0D00 // v_cvt_f32_u32 v8, v0
        } else {
            0x7E10_02F4 // v_mov_b32 v8, 2.0
        },
        0x3E08_1107, // v_mac_f32 v4, v7, v8
        0x7E18_0504, // v_readfirstlane_b32 s12, v4
        0x7E04_020C, // v_mov_b32 v2, s12
        0xBEFE_0400, // s_mov_b64 exec, s[0:1]
    ]);
    w.extend(store(2, 0));
    w.push(S_ENDPGM);
    w
}

#[test]
fn the_mac_read_programs_are_what_their_comments_say() {
    for per_lane in [false, true] {
        let words = mac_read_in_a_region(&MASK_LANES_12_44, per_lane);
        let code: Vec<u8> = words.iter().flat_map(|w| w.to_le_bytes()).collect();
        let all = gfx10::decode(&code)
            .expect("decodes")
            .iter()
            .map(|i| i.text(true))
            .collect::<Vec<_>>()
            .join("\n");
        for want in [
            "s_buffer_load_dword s14, s[44:47]",
            "s_and_saveexec_b64 s[0:1], vcc",
            "s_cbranch_execz",
            "v_cvt_f32_u32 v4, s14",
            "v_mov_b32 v7, 2.0",
            if per_lane {
                "v_cvt_f32_u32 v8, v0"
            } else {
                "v_mov_b32 v8, 2.0"
            },
            "v_mac_f32 v4, v7, v8",
            "v_readfirstlane_b32 s12, v4",
            "v_mov_b32 v2, s12",
            "s_mov_b64 exec, s[0:1]",
        ] {
            assert!(all.contains(want), "per_lane={per_lane}: {want:?} not in:\n{all}");
        }
    }
}

/// A multiply-accumulate adds to its old destination. Where that, like its
/// operands, is the same in every lane of the read's EXEC, so is its
/// result, and each half makes the read by itself: no halves for it, no
/// exchange. Before, any multiply-accumulate in the value's chain sent the
/// read to the halves. With a per-lane operand the read still takes them.
#[test]
fn a_first_lane_read_of_a_multiply_accumulate_is_made_in_each_half() {
    for threads in [64, 128] {
        let of = |per_lane: bool| {
            let words = mac_read_in_a_region(&MASK_LANES_12_44, per_lane);
            lowered(&native(&words, true, threads, 0), true)
                .unwrap_or_else(|e| panic!("per_lane={per_lane} {threads} threads: {e}"))
        };
        let alike = of(false);
        assert_eq!(alike.barriers, 0, "{threads} threads: no exchange");
        let per_lane = of(true);
        assert_eq!(per_lane.wave_size, Some(32));
        assert!(per_lane.barriers > 0, "{threads} threads: the halves pass the first lane");
    }
}

/// Both give the console's lanes on every GPU: the region's lanes store
/// `41.0 + 2.0 * 2.0`, or with a per-lane operand `41.0 + 2.0 *` the thread
/// id of the console wave's first lane in the region.
#[cfg(windows)]
#[test]
fn a_first_lane_read_of_a_multiply_accumulate_matches_the_console_on_every_gpu() {
    type InRegion = fn(u32) -> bool;
    let masks: [(&str, &[u32; 2], InRegion, &[u32]); 3] = [
        ("lower_8_9", &MASK_LANES_8_9, |t| t >> 1 == 4, &[64]),
        ("upper_40_41", &MASK_LANES_40_41, |t| t >> 1 == 20, &[64]),
        ("both_12_44", &MASK_LANES_12_44, |t| t & 31 == 12, &[64, 128]),
    ];
    let mut failed = Vec::new();
    for (label, mask, in_region, groups) in masks {
        for &threads in groups {
            for per_lane in [false, true] {
                let words = mac_read_in_a_region(mask, per_lane);
                let want: Vec<u32> = (0..threads)
                    .map(|t| {
                        if !in_region(t) {
                            return 0;
                        }
                        let base = t & !63;
                        let first = (base..base + 64).find(|&l| in_region(l)).expect("in it");
                        let operand = if per_lane { first as f32 } else { 2.0 };
                        (41.0 + 2.0 * operand).to_bits()
                    })
                    .collect();
                let found = differences_with_buffer(
                    true, &words, threads, 1, &want, 0, false, false, &[41, 0, 0, 0],
                );
                failed.extend(
                    found
                        .into_iter()
                        .map(|m| format!("{label}(per_lane={per_lane}) {threads} threads: {m}")),
                );
            }
        }
    }
    assert!(failed.is_empty(), "{}", failed.join("\n"));
}

/// Two first-lane reads of values the same in every lane that makes them,
/// in an `s_and_saveexec_b64` region whose EXEC is `mask_words`'s:
/// `v4 = 42` then `v4 = 1.0 + v4` (1.0, read into `s12`) and `v5 = 1.0 + v4`
/// (2.0, read into `s13`). The lanes outside the region never make `v4` or
/// `v5`. Every lane stores `v2` and `v6`, which take the two reads after
/// EXEC is restored (every lane), or with `inside` in the region (its lanes
/// only, the others keeping 0: reads only the lanes that ran them see).
/// `with_mbcnt`: a `v_mbcnt` first, so the program runs in halves on a
/// 32-lane host whatever else it holds.
fn alike_reads_in_a_region(mask_words: &[u32], with_mbcnt: bool, inside: bool) -> Vec<u32> {
    let mut w = PROLOGUE.to_vec();
    w.extend([
        0x3420_0085, // v_lshlrev_b32 v16, 5, v0
        0x7E04_0280, // v_mov_b32 v2, 0
        0x7E0C_0280, // v_mov_b32 v6, 0
    ]);
    if with_mbcnt {
        w.extend([
            0xD765_0001,
            0x0001_00C1, // v_mbcnt_lo_u32_b32 v1, -1, 0
        ]);
    }
    w.extend(mask_words);
    let moves = [
        0x7E04_020C, // v_mov_b32 v2, s12
        0x7E0C_020D, // v_mov_b32 v6, s13
    ];
    w.extend([
        0xBE80_246A,                               // s_and_saveexec_b64 s[0:1], vcc
        0xBF88_0005 + 2 * u32::from(inside), // s_cbranch_execz (past the region)
        0x7E08_02AA,                               // v_mov_b32 v4, 42
        0x0608_08F2,                               // v_add_f32 v4, 1.0, v4
        0x7E18_0504,                               // v_readfirstlane_b32 s12, v4
        0x060A_08F2,                               // v_add_f32 v5, 1.0, v4
        0x7E1A_0505,                               // v_readfirstlane_b32 s13, v5
    ]);
    if inside {
        w.extend(moves);
    }
    w.push(0xBEFE_0400); // s_mov_b64 exec, s[0:1]
    if !inside {
        w.extend(moves);
    }
    w.extend(store(2, 0));
    w.extend(store(6, 4));
    w.push(S_ENDPGM);
    w
}

#[test]
fn the_alike_read_programs_are_what_their_comments_say() {
    let words = alike_reads_in_a_region(&MASK_LANES_40_41, true, false);
    let code: Vec<u8> = words.iter().flat_map(|w| w.to_le_bytes()).collect();
    let all = gfx10::decode(&code)
        .expect("decodes")
        .iter()
        .map(|i| i.text(true))
        .collect::<Vec<_>>()
        .join("\n");
    for want in [
        "v_mov_b32 v6, 0",
        "v_mbcnt_lo_u32_b32 v1, -1, 0",
        "v_cmp_eq_u32 vcc, 20, v3",
        "s_and_saveexec_b64 s[0:1], vcc",
        "s_cbranch_execz",
        "v_mov_b32 v4, 42",
        "v_add_f32 v4, 1.0, v4",
        "v_readfirstlane_b32 s12, v4",
        "v_add_f32 v5, 1.0, v4",
        "v_readfirstlane_b32 s13, v5",
        "s_mov_b64 exec, s[0:1]",
        "v_mov_b32 v2, s12",
        "v_mov_b32 v6, s13",
    ] {
        assert!(all.contains(want), "{want:?} not in:\n{all}");
    }
}

/// Lanes 8 and 9 of every console wave (its lower half).
const MASK_WAVE_LANES_8_9: [u32; 2] = [
    0x3606_00BE, // v_and_b32 v3, 62, v0
    0x7D84_0688, // v_cmp_eq_u32 vcc, 8, v3
];

/// Lanes 40 and 41 of every console wave (its upper half).
const MASK_WAVE_LANES_40_41: [u32; 2] = [
    0x3606_00BE, // v_and_b32 v3, 62, v0
    0x7D84_06A8, // v_cmp_eq_u32 vcc, 40, v3
];

/// A first-lane read of a value the same in every lane that makes it reads
/// a lane that made it. The halves run such a region in every lane, under a
/// predicate: the host wave's first lane may be outside it and hold an old
/// value (the read took that one), and a half with no lane of the region
/// holds none at all (it takes the other half's). `inside`: only the
/// region's lanes see the reads, each half making them by itself
/// (`first_lane::read_alone`), from a lane of the region too.
#[cfg(windows)]
#[test]
fn a_first_lane_read_of_an_alike_value_in_a_region_matches_the_console_on_every_gpu() {
    let mut failed = Vec::new();
    type InRegion = fn(u32) -> bool;
    let masks: [(&str, &[u32; 2], InRegion); 3] = [
        ("lower_8_9", &MASK_WAVE_LANES_8_9, |t| t & 62 == 8),
        ("upper_40_41", &MASK_WAVE_LANES_40_41, |t| t & 62 == 40),
        ("both_12_44", &MASK_LANES_12_44, |t| t & 31 == 12),
    ];
    for (label, mask, in_region) in masks {
        for with_mbcnt in [false, true] {
            for threads in [64, 128] {
                for inside in [false, true] {
                    let words = alike_reads_in_a_region(mask, with_mbcnt, inside);
                    let want: Vec<u32> = (0..threads)
                        .flat_map(|t| {
                            if !inside || in_region(t) {
                                [1.0f32.to_bits(), 2.0f32.to_bits()]
                            } else {
                                [0, 0]
                            }
                        })
                        .collect();
                    let found = differences_with_buffer(
                        true, &words, threads, 2, &want, 0, false, false, &[0; 4],
                    );
                    failed.extend(found.into_iter().map(|m| {
                        format!("{label}(mbcnt={with_mbcnt}, inside={inside}) {threads} threads: {m}")
                    }));
                }
            }
        }
    }
    assert!(failed.is_empty(), "{}", failed.join("\n"));
}

/// The masks are what their comments say.
#[test]
fn the_wave_lane_masks_are_what_their_comments_say() {
    for (mask, want) in [
        (&MASK_WAVE_LANES_8_9, "v_cmp_eq_u32 vcc, 8, v3"),
        (&MASK_WAVE_LANES_40_41, "v_cmp_eq_u32 vcc, 40, v3"),
    ] {
        let code: Vec<u8> = mask
            .iter()
            .chain(&[S_ENDPGM])
            .flat_map(|w| w.to_le_bytes())
            .collect();
        let text: Vec<String> = gfx10::decode(&code)
            .expect("decodes")
            .iter()
            .map(|i| i.text(true))
            .collect();
        assert_eq!(text[..2], ["v_and_b32 v3, 62, v0", want]);
    }
}

/// The hand-encoded words are the instructions their comments name.
#[test]
fn the_test_programs_are_what_their_comments_say() {
    let programs: Vec<(Vec<u32>, &[&str])> = vec![
        (
            in_halves(),
            &[
                "v_mbcnt_lo_u32_b32 v1, -1, 0",
                "v_mbcnt_hi_u32_b32 v1, -1, v1",
                "ds_swizzle_b32 v2, v0",
                "v_xor_b32 v4, 9, v0",
                "ds_bpermute_b32 v3, v4, v0",
                "v_writelane_b32 v5, s8, 40",
                "v_cndmask_b32 v6, 0, 1, s[20:21]",
                "v_mov_b32 v7, v0 quad_perm:[1,0,3,2]",
                "buffer_store_dword v7, v16, s[40:43], 0 offen offset:24",
            ],
        ),
        (
            sum_after_a_loop_lanes_leave(),
            &[
                "v_and_b32 v5, 3, v0",
                "v_mov_b32 v6, 0",
                "s_mov_b64 s[4:5], exec",
                "v_cmp_lt_u32 vcc, v6, v5",
                "s_and_b64 exec, exec, vcc",
                "s_cbranch_execz",
                "v_add_nc_u32 v6, 1, v6",
                "s_branch",
                "s_mov_b64 exec, s[4:5]",
                "v_cvt_f32_u32 v3, v6",
            ],
        ),
        (
            loops_sharing_a_latch(),
            &[
                "s_add_u32 s22, s22, 1",
                "s_cmp_lt_u32 s22, 3",
                "s_add_u32 s20, s20, 1",
                "s_cmp_gt_u32 s20, 2",
                "s_mov_b32 s22, 0",
                "s_add_u32 s21, s21, 1",
                "v_mov_b32 v1, s21",
            ],
        ),
        (
            dpp_row_mask(),
            &[
                "v_mov_b32 v7, -1",
                "v_mov_b32 v7, v0 quad_perm:[1,0,3,2] row_mask:0xc",
                "buffer_store_dword v7, v16, s[40:43], 0 offen",
            ],
        ),
        (
            first_lane_of_the_thread_id(),
            &[
                "v_mbcnt_lo_u32_b32 v1, -1, 0",
                "v_readfirstlane_b32 s12, v0",
                "v_mov_b32 v2, s12",
            ],
        ),
        (
            first_lane_of_a_buffer_load(3),
            &[
                "s_load_dwordx4 s[44:47], s[4:5], 0x000020",
                "v_mov_b32 v3, 4",
                "buffer_load_dword v4, v3, s[44:47], 0 offen",
                "s_waitcnt vmcnt(0)",
                "v_readfirstlane_b32 s12, v4",
                "v_mov_b32 v5, s12",
            ],
        ),
        (
            first_lane_alike(),
            &[
                "v_lshrrev_b32 v2, 6, v0",
                "v_readfirstlane_b32 s12, v2",
                "v_mov_b32 v3, s12",
                "buffer_store_dword v1, v16, s[40:43], 0 offen offset:4",
            ],
        ),
        (
            first_active_lane(),
            &["s_ff1_i32_b64 s12, exec", "v_mov_b32 v1, s12"],
        ),
        (
            lanes_below(100),
            &[
                "v_cmp_gt_u32 vcc, lit(0x00000064), v0",
                "s_bcnt1_i32_b64 s12, vcc",
                "v_mov_b32 v2, vcc_hi",
                "v_mbcnt_lo_u32_b32 v3, vcc_lo, 0",
                "v_mbcnt_hi_u32_b32 v3, vcc_hi, v3",
                "buffer_store_dword v3, v16, s[40:43], 0 offen offset:8",
            ],
        ),
        (
            lanes_below_in_a_loop(),
            &[
                "s_mov_b32 s20, 0",
                "v_cmp_gt_u32 vcc, 40, v0",
                "s_add_u32 s20, s20, 1",
                "s_cmp_lt_u32 s20, 2",
                "s_cbranch_scc1",
            ],
        ),
        (
            lane_reads(40, 33),
            &[
                "v_readlane_b32 s12, v0, 40",
                "v_readlane_b32 s14, v0, s13",
                "v_mov_b32 v1, s12",
                "v_mov_b32 v2, s14",
                "buffer_store_dword v2, v16, s[40:43], 0 offen offset:4",
            ],
        ),
        (
            reduction(V_ADD_F32, 0x80, 0x7E06_0D00),
            &[
                "v_cvt_f32_u32 v3, v0",
                "s_or_saveexec_b64 s[12:13], -1",
                "v_cndmask_b32 v1, 0, v3, s[12:13]",
                "v_permlanex16_b32 v2, v1, 0, 0",
                "v_add_f32 v1, v2, v1",
                "v_readlane_b32 s14, v1, 32",
                "v_add_f32 v1, s14, v1",
                "v_readlane_b32 s15, v1, 0",
                "s_mov_b64 exec, s[12:13]",
                "v_mov_b32 v1, s15",
            ],
        ),
        (
            reduction(V_MIN_U32, 0xC1, V_MOV_V3_V0),
            &[
                "v_mov_b32 v3, v0",
                "v_cndmask_b32 v1, -1, v3, s[12:13]",
                "v_min_u32 v1, v2, v1",
                "v_min_u32 v1, s14, v1",
            ],
        ),
        (
            reduction(V_MAX_U32, 0x80, V_MOV_V3_V0),
            &["v_max_u32 v1, v2, v1", "v_max_u32 v1, s14, v1"],
        ),
        (
            exclusive_prefix(),
            &[
                "s_or_saveexec_b64 s[0:1], -1",
                "v_cndmask_b32 v1, 0, v0, s[0:1]",
                "v_add_nc_u32 v1, v1, v1 row_shr:8",
                "v_readlane_b32 s2, v1, 47",
                "v_mov_b32 v3, v1 row_shr:1",
                "s_mov_b64 exec, -2",
                "v_mov_b32 v2, v3",
                "s_mov_b64 exec, s[0:1]",
                "buffer_store_dword v2, v16, s[40:43], 0 offen",
            ],
        ),
        (
            in_a_loop(&[0xD760_000C, 0x0001_5100], 12, Count::Two),
            &[
                "v_readlane_b32 s12, v0, 40",
                "s_add_u32 s20, s20, 1",
                "s_cmp_lt_u32 s20, 2",
                "s_cbranch_scc1",
                "v_mov_b32 v1, s12",
            ],
        ),
        (
            in_a_loop(&[0x7D88_00A8, 0xBE8C_106A], 12, Count::FirstLane),
            &[
                "v_readfirstlane_b32 s21, v0",
                "v_cmp_gt_u32 vcc, 40, v0",
                "s_bcnt1_i32_b64 s12, vcc",
                "s_cmp_lt_u32 s20, s21",
                "s_cbranch_scc1",
            ],
        ),
    ];
    for (words, wants) in programs {
        let code: Vec<u8> = words.iter().flat_map(|w| w.to_le_bytes()).collect();
        let all = gfx10::decode(&code)
            .expect("decodes")
            .iter()
            .map(|i| i.text(true))
            .collect::<Vec<_>>()
            .join("\n");
        for want in wants {
            assert!(all.contains(want), "{want:?} not in:\n{all}");
        }
    }
}

/// Each lane counts to the low two bits of its thread id in a loop it leaves
/// on its own, whose back edge is unconditional (`s_branch`): lowering caps
/// that back edge at the loop iteration limit. Then the console compiler's
/// whole-wave sum of the counts ([`reduction`] of `v_cvt_f32_u32 v3, v6`).
/// Per thread, one dword at `l * 32`: its console wave's sum.
fn sum_after_a_loop_lanes_leave() -> Vec<u32> {
    let sum = reduction(V_ADD_F32, 0x80, 0x7E06_0D06);
    let mut w = PROLOGUE.to_vec();
    w.extend([
        0x3420_0085, // v_lshlrev_b32 v16, 5, v0
        0x360A_0083, // v_and_b32 v5, 3, v0
        0x7E0C_0280, // v_mov_b32 v6, 0
        0xBE84_047E, // s_mov_b64 s[4:5], exec
        0x7D82_0B06, // v_cmp_lt_u32 vcc, v6, v5
        0x87FE_6A7E, // s_and_b64 exec, exec, vcc
        0xBF88_0002, // s_cbranch_execz (past the s_branch)
        0x4A0C_0C81, // v_add_nc_u32 v6, 1, v6
        0xBF82_FFFB, // s_branch (back to the v_cmp)
        0xBEFE_0404, // s_mov_b64 exec, s[4:5]
    ]);
    w.extend(&sum[PROLOGUE.len() + 1..]);
    w
}

/// Two scalar loops whose latches share a block, as the console compiler
/// lays them out: the latch branches back to the inner loop's header three
/// times and otherwise falls through to the outer loop's header, which runs
/// twice. Both targets of that branch are back edges. Per thread, one dword
/// at `l * 32`: the inner loop's passes, 6.
fn loops_sharing_a_latch() -> Vec<u32> {
    let mut w = PROLOGUE.to_vec();
    w.extend([
        0x3420_0085, // v_lshlrev_b32 v16, 5, v0
        0xBE94_0380, // s_mov_b32 s20, 0
        0xBE95_0380, // s_mov_b32 s21, 0
        0xBF82_0003, // s_branch (to the outer header)
        0x8016_8116, // s_add_u32 s22, s22, 1 (the shared latch)
        0xBF0A_8316, // s_cmp_lt_u32 s22, 3
        0xBF85_0004, // s_cbranch_scc1 (back to the inner header)
        0x8014_8114, // s_add_u32 s20, s20, 1 (the outer header)
        0xBF08_8214, // s_cmp_gt_u32 s20, 2
        0xBF85_0003, // s_cbranch_scc1 (out of both loops)
        0xBE96_0380, // s_mov_b32 s22, 0
        0x8015_8115, // s_add_u32 s21, s21, 1 (the inner header)
        0xBF82_FFF7, // s_branch (to the shared latch)
        0x7E02_0215, // v_mov_b32 v1, s21
    ]);
    w.extend(store(1, 0));
    w.push(S_ENDPGM);
    w
}

/// A reduction after a loop the lanes leave one by one, whose back edge is
/// capped at the loop iteration limit ([`sum_after_a_loop_lanes_leave`]):
/// the console runs the loop to its end, so with the cap taken as never
/// reached every lane of the group runs the reduction together and the
/// halves pass each other their sums. Only with the cap's barriers refused
/// (`VICTORY_CAP_BARRIERS=1`) is a lane taken to have left at the cap, and
/// the reduction refused.
#[test]
fn a_reduction_after_a_loop_lanes_leave_passes_between_halves() {
    let words = sum_after_a_loop_lanes_leave();
    for threads in [64, 96, 128] {
        // The lockstep loops of #591 may now translate this shape even with
        // the cap's barriers, so the refusal is no longer required.
        let why = super::with_cap_barriers(true, || refusal(&words, threads, 0));
        assert!(
            why == "translated" || why.contains("a wave reduction over both 32-lane halves"),
            "{threads}: {why}"
        );
        let translated = super::with_cap_barriers(false, || refusal(&words, threads, 0));
        assert_eq!(translated, "translated", "{threads}");
    }
    assert_eq!(refusal(&words, 32, 0), "translated");
}

/// [`sum_after_a_loop_lanes_leave`] on every hardware adapter gives the
/// console's sums, in a group of 64 (one console wave in two halves), of 96
/// (the second console wave has no upper half) and of 128 (two).
#[test]
#[cfg(windows)]
fn a_reduction_after_a_loop_lanes_leave_matches_the_console_on_every_gpu() {
    let words = sum_after_a_loop_lanes_leave();
    let mut failed = Vec::new();
    for threads in [64, 96, 128] {
        let want: Vec<u32> = (0..threads)
            .map(|l| (console_reduction(threads, l, |a, b| a + b, |t| t & 3) as f32).to_bits())
            .collect();
        failed.extend(super::with_cap_barriers(false, || {
            differences(&words, threads, 1, &want)
        }));
    }
    assert!(failed.is_empty(), "{}", failed.join("\n"));
}

/// [`loops_sharing_a_latch`] in whole-wave halves: each side of the shared
/// latch's branch counts its own loop's passes under the iteration limit.
#[test]
fn a_branch_back_to_two_loops_runs_in_whole_wave_halves() {
    let n = native(&loops_sharing_a_latch(), true, 64, 0);
    let literal = lower_halves(&n, true, false, true).expect("whole-wave halves");
    assert_eq!(literal.wave_size, Some(32));
}

/// [`loops_sharing_a_latch`] in whole-wave halves on every hardware adapter
/// gives the console's count in every lane.
#[test]
#[cfg(windows)]
fn a_branch_back_to_two_loops_matches_the_console_on_every_gpu() {
    let words = loops_sharing_a_latch();
    let failed = differences_in(true, &words, 64, 1, &[6; 64], 0, false, true);
    assert!(failed.is_empty(), "{}", failed.join("\n"));
}

/// `v_permlane16_b32` or `v_permlanex16_b32` with the FI and BOUND_CTRL bits
/// (`op_sel` bits 0 and 1, word bits 11 and 12): lane `l` takes the value
/// of `v0` from its own row of 16 (or the other row of its half), the lane
/// its nibble of `s8` (lanes 0 to 7) or `s9` (lanes 8 to 15) names.
const fn permlane16(other_row: bool, fetch_inactive: bool, bound_ctrl: bool) -> [u32; 2] {
    let opcode = if other_row { 0xD778_0001 } else { 0xD777_0001 };
    [
        opcode | (fetch_inactive as u32) << 11 | (bound_ctrl as u32) << 12,
        0x0024_1100,
    ]
}

/// Keep arbitrary high bits in the same-row fixture's source data.
const PERM_PAYLOAD: u32 = 0xDEAD_BE00;

/// Per workgroup lane: `v0` (its thread id, or a raw value made from the
/// workgroup and thread ids for the same-row form), permuted first with all
/// lanes active, then with only even lanes active and every FI/BOUND_CTRL
/// combination. The same-row form also checks full EXEC with BOUND_CTRL
/// set. Each masked or potentially out-of-bounds read starts with `v1`
/// holding [`PERM_OLD`]. EXEC is whole again for every store.
#[cfg(windows)]
fn permlane16_program(lanes: u32, other_row: bool) -> Vec<u32> {
    let wave64 = lanes > 32;
    let full = |w: &mut Vec<u32>| {
        w.push(if wave64 {
            0xBEFE_04C1 // s_mov_b64 exec, -1
        } else {
            0xBEFE_03C1 // s_mov_b32 exec_lo, -1
        });
    };
    let even = |w: &mut Vec<u32>| {
        w.push(if wave64 {
            0xBEFE_040A // s_mov_b64 exec, s[10:11]
        } else {
            0xBEFE_030A // s_mov_b32 exec_lo, s10
        });
    };
    let mut w = PROLOGUE.to_vec();
    if other_row {
        w.push(0x3420_0085); // v_lshlrev_b32 v16, 5, v0
    } else {
        w.extend([
            0x8F02_8801, // s_lshl_b32 s2, s1 (group id X), 8
            0x4A22_0002, // v_add_nc_u32 v17, s2, v0
            0x3420_2285, // v_lshlrev_b32 v16, 5, v17
            0x3A00_22FF,
            PERM_PAYLOAD, // v_xor_b32 v0, PERM_PAYLOAD, v17
        ]);
    }
    w.extend([
        0xBE88_03FF,
        PERM_LOW, // s_mov_b32 s8, PERM_LOW
        0xBE89_03FF,
        PERM_HIGH, // s_mov_b32 s9, PERM_HIGH
        0xBE8A_03FF,
        0x5555_5555, // s_mov_b32 s10, 0x55555555
        0xBE8B_03FF,
        0x5555_5555, // s_mov_b32 s11, 0x55555555
    ]);
    if !other_row {
        w.extend([0x7E02_02FF, PERM_OLD]); // v_mov_b32 v1, PERM_OLD
    }
    w.extend(permlane16(other_row, false, false));
    w.extend(store(1, 0));
    for (slot, (fi, bc)) in [(false, false), (false, true), (true, false), (true, true)]
        .into_iter()
        .enumerate()
    {
        w.extend([0x7E02_02FF, PERM_OLD]); // v_mov_b32 v1, PERM_OLD
        even(&mut w);
        w.extend(permlane16(other_row, fi, bc));
        full(&mut w);
        w.extend(store(1, 4 * (slot as u32 + 1)));
    }
    if !other_row {
        w.extend([0x7E02_02FF, PERM_OLD]); // v_mov_b32 v1, PERM_OLD
        w.extend(permlane16(false, false, true));
        w.extend(store(1, 20));
    }
    w.push(S_ENDPGM);
    w
}

/// What the console gives [`permlane16_program`]'s lane `l`. An FI read
/// from a lane that was never launched has no documented input value.
fn permlane16_console(l: u32, group: u32, threads: u32, other_row: bool) -> [Option<u32>; 6] {
    let in_row = l % 16;
    let selectors = if in_row >= 8 { PERM_HIGH } else { PERM_LOW };
    let row = if other_row { (l / 16) ^ 1 } else { l / 16 };
    let source = row * 16 + ((selectors >> (4 * (in_row % 8))) & 15);
    let value = if other_row {
        source
    } else {
        (256 * group + source) ^ PERM_PAYLOAD
    };
    let active = |lane: u32| lane.is_multiple_of(2);
    let result = |masked: bool, fi: bool, bc: bool| {
        if masked && !active(l) {
            Some(PERM_OLD)
        } else if fi {
            (source < threads).then_some(value)
        } else if source < threads && (!masked || active(source)) {
            Some(value)
        } else if bc {
            Some(0)
        } else {
            Some(PERM_OLD)
        }
    };
    [
        result(false, false, false),
        result(true, false, false),
        result(true, false, true),
        result(true, true, false),
        result(true, true, true),
        result(false, false, true),
    ]
}

/// The hand-encoded words are the instructions their comments name.
#[test]
fn the_permlane16_programs_are_what_their_comments_say() {
    for (words, opcode) in [
        (permlane16(true, false, false), "v_permlanex16_b32"),
        (permlane16(true, true, true), "v_permlanex16_b32"),
        (permlane16(false, false, false), "v_permlane16_b32"),
        (permlane16(false, true, true), "v_permlane16_b32"),
    ] {
        let p = ir::build(
            &[words[0], words[1], S_ENDPGM]
                .iter()
                .flat_map(|w| w.to_le_bytes())
                .collect::<Vec<u8>>(),
            false,
            32,
        )
        .expect("ir::build");
        assert_eq!(p.blocks[0].insts[0].name(), opcode);
    }
}

/// The lanes of each row read the other row of their 32-lane half through
/// their selector nibbles, and an inactive source lane gives the old value,
/// zero or its register as FI and BOUND_CTRL say, on every hardware
/// adapter, in console waves of 32 and of 64 (in halves and whole).
#[test]
#[cfg(windows)]
fn permlanex16_picks_lanes_as_the_console_does_on_every_gpu() {
    let mut failed = Vec::new();
    for lanes in [32, 64] {
        let want: Vec<u32> = (0..lanes)
            .flat_map(|l| {
                permlane16_console(l, 0, lanes, true)
                    .into_iter()
                    .take(5)
                    .map(|value| value.expect("full cross-row fixtures"))
            })
            .collect();
        failed.extend(differences_in(
            lanes == 64,
            &permlane16_program(lanes, true),
            lanes,
            5,
            &want,
            0,
            false,
            false,
        ));
    }
    assert!(failed.is_empty(), "{}", failed.join("\n"));
}

/// Lane 0 selects an odd lane (nibble 15) and lane 8 selects an even lane
/// (nibble 0). Lane 32 stays in the same half for either row choice.
#[test]
fn the_permlane16_model_follows_the_instructions() {
    assert_eq!(
        permlane16_console(0, 0, 64, true),
        [31, PERM_OLD, 0, 31, 31, 31].map(Some)
    );
    assert_eq!(permlane16_console(8, 0, 64, true), [Some(16); 6]);
    assert_eq!(
        permlane16_console(1, 0, 64, true),
        [29, PERM_OLD, PERM_OLD, PERM_OLD, PERM_OLD, 29].map(Some)
    );
    assert_eq!(permlane16_console(16, 0, 64, true)[0], Some(15));
    assert_eq!(permlane16_console(32, 0, 64, true)[0], Some(63));
    assert_eq!(permlane16_console(48, 0, 64, true)[0], Some(47));
    assert_eq!(
        permlane16_console(0, 0, 64, false),
        [
            0xDEAD_BE0F,
            PERM_OLD,
            0,
            0xDEAD_BE0F,
            0xDEAD_BE0F,
            0xDEAD_BE0F
        ]
        .map(Some)
    );
    assert_eq!(permlane16_console(8, 0, 64, false), [Some(PERM_PAYLOAD); 6]);
    assert_eq!(permlane16_console(32, 1, 64, false)[0], Some(0xDEAD_BF2F));
    assert_eq!(
        permlane16_console(64, 0, 79, false),
        [Some(PERM_OLD), Some(PERM_OLD), Some(0), None, None, Some(0)]
    );
}

fn same_row_permlane16_native(threads: u32) -> NativeProgram {
    let words = permlane16_program(threads, false);
    let mut n = native(&words, threads > 32, threads, 0);
    let StageIo::Cs(cs) = &mut n.io else {
        unreachable!()
    };
    cs.rsrc2 |= 1 << 7; // group id X in s1, after the user-data SGPR
    n
}

#[test]
fn same_row_permlane16_uses_host_lanes_without_an_exchange() {
    let n = same_row_permlane16_native(64);
    let l = super::with_thread_ticket(false, || lowered(&n, true))
        .expect("a live same-row permute lowers in wave64 halves");
    assert_eq!(l.wave_size, Some(32));
    assert!(l.uses_lane_index);
    assert_eq!(l.barriers, 0);
}

/// The last 16-lane row is complete at 80 and 112 threads. At 79 threads
/// some selectors name the missing lane. Check both BOUND_CTRL outcomes
/// for those reads with FI clear.
#[test]
#[cfg(windows)]
fn same_row_permlane16_matches_the_console_on_hardware() {
    let _one = super::valu_tests::ONE_AT_A_TIME
        .lock()
        .unwrap_or_else(|e| e.into_inner());
    let hosts = Host::each_hardware();
    if hosts.is_empty() {
        eprintln!("no SM 6.6 hardware device; skipped same-row permlane16");
        return;
    }
    let workgroups = 3;
    let mut completed = 0;
    let mut checked_lanes = 0;
    let mut checked_dwords = 0;
    for (adapter, host) in &hosts {
        for threads in [32, 64, 79, 80, 96, 112, 128, 256] {
            let n = same_row_permlane16_native(threads);
            let bytes = crate::dxbc::build(&[]);
            let container = crate::dxbc::Container::parse(&bytes).expect("empty test container");
            let (p, st) =
                crate::range::build(&n.patched_code(), n.wave64, n.vgprs).expect("range::build");
            let st = st.expect("structure");
            let res = resource::track(&p, &n.layout, n.stage, Some(&st)).expect("track");
            let io = io::raise(&p, &n).expect("io::raise");
            let shader = super::emit_with(
                &container,
                &n,
                &p,
                &st,
                &io,
                &res,
                super::Options {
                    host_wave32: true,
                    narrow_waves: true,
                    ..Default::default()
                },
            )
            .expect("production translation of same-row permlane16");
            let label = format!("{adapter}: {threads} threads");
            if let Ok(mut v) =
                validator::Validator::load(validator::Validator::default_path().as_deref())
            {
                v.validate(&mut shader.dxil.clone())
                    .unwrap_or_else(|e| panic!("{label}: DXIL validation: {e}"));
            }
            let (root, pso) = host
                .compute_pipeline(&shader)
                .unwrap_or_else(|e| panic!("{label}: pipeline creation: {e}"));
            let ud = vec![0; shader.user_data_dwords().max(1) as usize];
            let out = host
                .dispatch(
                    &root,
                    &pso,
                    &Dispatch {
                        roots: vec![(0, Root::Constants(ud))],
                        views: &[
                            View::Texture(1, 1, &[[0.0; 4]]),
                            View::Buffer {
                                data: &[0; 4],
                                stride: Stride::Structured(16),
                            },
                            View::RawUav(workgroups * 256 * 8),
                        ],
                        samplers: &[],
                        groups: [workgroups, 1, 1],
                    },
                )
                .unwrap_or_else(|e| panic!("{label}: dispatch failed: {e}"));
            assert_eq!(out.len(), 1, "{label}: readback UAV count");
            assert_eq!(
                out[0].len(),
                (workgroups * 256 * 8) as usize,
                "{label}: readback length"
            );
            for group in 0..workgroups {
                for lane in 0..threads {
                    let offset = ((group * 256 + lane) * 8) as usize;
                    for (slot, expected) in permlane16_console(lane, group, threads, false)
                        .into_iter()
                        .enumerate()
                    {
                        if let Some(expected) = expected {
                            assert_eq!(
                                out[0][offset + slot],
                                expected,
                                "{label}: workgroup {group}, lane {lane}, slot {slot}"
                            );
                            checked_dwords += 1;
                        }
                    }
                }
            }
            completed += 1;
            checked_lanes += threads * workgroups;
            eprintln!("{label}: pipeline created, {workgroups} workgroups completed and checked");
        }
    }
    assert!(completed > 0 && checked_lanes > 0 && checked_dwords > 0);
    eprintln!(
        "same-row permlane16: {completed} hardware dispatches, {checked_lanes} lanes and {checked_dwords} defined dwords checked"
    );
}

/// A `while` loop as the console compiler lays one out: the entry jumps to
/// the test, which branches back to the body while lanes stay; the body
/// falls into the test. The body reads the first lane's thread id
/// (`v_readfirstlane_b32 s6, v0`) into an accumulator `v2`, and counts its
/// passes in `v1`. `narrow`: a lane stays while `v1 < (tid & 7)`, and the
/// test drops the lanes done from EXEC (`s_and_b64 exec, exec, vcc;
/// s_cbranch_execnz`). Else every lane stays while `v1 < 7`, a test the
/// same in every lane, and EXEC stays (`s_and_b64 vcc, exec, vcc;
/// s_cbranch_vccnz`, as GTA VI's loops test). Every lane stores `v2`.
fn while_loop_first_lane(narrow: bool) -> Vec<u32> {
    let mut w = PROLOGUE.to_vec();
    w.extend([
        0x3420_0085, // v_lshlrev_b32 v16, 5, v0
        0x7E02_0280, // v_mov_b32 v1, 0
        0x7E04_0280, // v_mov_b32 v2, 0
        if narrow {
            0x360A_0087 // v_and_b32 v5, 7, v0
        } else {
            0x7E0A_0287 // v_mov_b32 v5, 7
        },
        0xBEA4_047E, // s_mov_b64 s[36:37], exec
        0xBF82_0003, // s_branch (the test)
        0x7E0C_0500, // v_readfirstlane_b32 s6, v0
        0x4A04_0406, // v_add_nc_u32 v2, s6, v2
        0x4A02_0281, // v_add_nc_u32 v1, 1, v1
        0x7D82_0B01, // v_cmp_lt_u32 vcc, v1, v5
    ]);
    if narrow {
        w.extend([
            0x87FE_6A7E, // s_and_b64 exec, exec, vcc
            0xBF89_FFFA, // s_cbranch_execnz (the body)
        ]);
    } else {
        w.extend([
            0x87EA_6A7E, // s_and_b64 vcc, exec, vcc
            0xBF87_FFFA, // s_cbranch_vccnz (the body)
        ]);
    }
    w.push(0xBEFE_0424); // s_mov_b64 exec, s[36:37]
    w.extend(store(2, 0));
    w.push(S_ENDPGM);
    w
}

/// What every lane of [`while_loop_first_lane`] stores on the console: the
/// first lane still in the loop in each pass is the console wave's thread
/// `64 * wave + pass + 1` (narrow), or its first, `64 * wave`, in all 7
/// passes.
fn console_while_loop_first_lane(narrow: bool, t: u32) -> u32 {
    let base = t & !63;
    if narrow {
        let n = t & 7;
        n * base + n * (n + 1) / 2
    } else {
        7 * base
    }
}

#[test]
fn the_while_loop_programs_are_what_their_comments_say() {
    for narrow in [false, true] {
        let words = while_loop_first_lane(narrow);
        let code: Vec<u8> = words.iter().flat_map(|w| w.to_le_bytes()).collect();
        let all = gfx10::decode(&code)
            .expect("decodes")
            .iter()
            .map(|i| i.text(true))
            .collect::<Vec<_>>()
            .join("\n");
        let tail: &[&str] = if narrow {
            &["s_and_b64 exec, exec, vcc", "s_cbranch_execnz"]
        } else {
            &["s_and_b64 vcc, exec, vcc", "s_cbranch_vccnz"]
        };
        let bound = if narrow {
            "v_and_b32 v5, 7, v0"
        } else {
            "v_mov_b32 v5, 7"
        };
        for want in [
            bound,
            "s_mov_b64 s[36:37], exec",
            "s_branch",
            "v_readfirstlane_b32 s6, v0",
            "v_add_nc_u32 v2, s6, v2",
            "v_add_nc_u32 v1, 1, v1",
            "v_cmp_lt_u32 vcc, v1, v5",
            "s_mov_b64 exec, s[36:37]",
        ]
        .iter()
        .chain(tail)
        {
            assert!(all.contains(want), "{want:?} not in:\n{all}");
        }
    }
}

/// A first-lane read in a `while` loop (its exit test before the latch's
/// work, as GTA VI's `23d507cbb6a22e86` and `b0095eb50222e808` have) runs in
/// halves with the loop in lockstep: the group stays while any lane stays,
/// so the halves meet in every pass and pass the console wave's first lane
/// between them. Before, the read was refused there.
#[test]
fn a_first_lane_read_in_a_while_loop_translates_in_halves() {
    for narrow in [false, true] {
        for threads in [64, 128, 256] {
            let words = while_loop_first_lane(narrow);
            let l = lowered(&native(&words, true, threads, 0), true)
                .unwrap_or_else(|e| panic!("narrow={narrow} {threads} threads: {e}"));
            assert_eq!(l.wave_size, Some(32));
            assert!(l.barriers > 0, "the halves meet in the loop");
        }
    }
}

/// The read in a `while` loop gives the console's lanes on every GPU.
#[cfg(windows)]
#[test]
fn a_first_lane_read_in_a_while_loop_matches_the_console_on_every_gpu() {
    let mut failed = Vec::new();
    for narrow in [false, true] {
        for threads in [64, 128] {
            let words = while_loop_first_lane(narrow);
            let want: Vec<u32> = (0..threads)
                .map(|t| console_while_loop_first_lane(narrow, t))
                .collect();
            failed.extend(
                differences(&words, threads, 1, &want)
                    .into_iter()
                    .map(|m| format!("narrow={narrow} {threads} threads: {m}")),
            );
        }
    }
    assert!(failed.is_empty(), "{}", failed.join("\n"));
}

/// [`while_loop_first_lane`] (the form lanes leave one by one: a lane stays
/// while `v1 < (tid & 7)`) with a store in its body: each pass a lane makes
/// stores its count of passes `v1` at dword `v1` of its own eight. A lane
/// that has left the loop stores no more.
fn while_loop_with_a_store() -> Vec<u32> {
    let mut w = PROLOGUE.to_vec();
    w.extend([
        0x3420_0085, // v_lshlrev_b32 v16, 5, v0
        0x7E02_0280, // v_mov_b32 v1, 0
        0x7E04_0280, // v_mov_b32 v2, 0
        0x360A_0087, // v_and_b32 v5, 7, v0
        0xBEA4_047E, // s_mov_b64 s[36:37], exec
        0xBF82_0007, // s_branch (the test)
        0x7E0C_0500, // v_readfirstlane_b32 s6, v0
        0x4A04_0406, // v_add_nc_u32 v2, s6, v2
        0x4A02_0281, // v_add_nc_u32 v1, 1, v1
        0x3422_0282, // v_lshlrev_b32 v17, 2, v1
        0x4A22_2310, // v_add_nc_u32 v17, v16, v17
        0xE070_1000,
        0x800A_0111, // buffer_store_dword v1, v17, s[40:43], 0 offen
        0x7D82_0B01, // v_cmp_lt_u32 vcc, v1, v5
        0x87FE_6A7E, // s_and_b64 exec, exec, vcc
        0xBF89_FFF6, // s_cbranch_execnz (the body)
        0xBEFE_0424, // s_mov_b64 exec, s[36:37]
    ]);
    w.extend(store(2, 0));
    w.push(S_ENDPGM);
    w
}

#[test]
fn the_while_loop_with_a_store_is_what_its_comments_say() {
    let words = while_loop_with_a_store();
    let code: Vec<u8> = words.iter().flat_map(|w| w.to_le_bytes()).collect();
    let text: Vec<String> = gfx10::decode(&code)
        .expect("decodes")
        .iter()
        .map(|i| i.text(true))
        .collect();
    let want = [
        "v_lshlrev_b32 v16, 5, v0",
        "v_mov_b32 v1, 0",
        "v_mov_b32 v2, 0",
        "v_and_b32 v5, 7, v0",
        "s_mov_b64 s[36:37], exec",
        "s_branch",
        "v_readfirstlane_b32 s6, v0",
        "v_add_nc_u32 v2, s6, v2",
        "v_add_nc_u32 v1, 1, v1",
        "v_lshlrev_b32 v17, 2, v1",
        "v_add_nc_u32 v17, v16, v17",
        "buffer_store_dword v1, v17, s[40:43], 0 offen",
        "v_cmp_lt_u32 vcc, v1, v5",
        "s_and_b64 exec, exec, vcc",
        "s_cbranch_execnz",
        "s_mov_b64 exec, s[36:37]",
    ];
    // The prologue's five words are four instructions.
    for (n, want) in want.iter().enumerate() {
        let got = &text[PROLOGUE.len() - 1 + n];
        assert!(got.starts_with(want), "{n}: {got:?}, not {want:?}");
    }
}

/// A store in a `while` loop run in lockstep is made by the lanes still in
/// the loop only: a lane that left keeps running the body with the group,
/// inactive, and stores nothing. Each lane's dwords 1 to `tid & 7` hold
/// their own index, the rest stay 0; dword 0 is the accumulator of
/// [`console_while_loop_first_lane`].
#[cfg(windows)]
#[test]
fn a_store_in_a_while_loop_is_made_by_the_lanes_still_in_it_on_every_gpu() {
    let words = while_loop_with_a_store();
    let mut failed = Vec::new();
    for threads in [64, 128] {
        let want: Vec<u32> = (0..threads)
            .flat_map(|t| {
                let n = t & 7;
                let mut lane = [0; 8];
                lane[0] = console_while_loop_first_lane(true, t);
                for p in 1..=n {
                    lane[p as usize] = p;
                }
                lane
            })
            .collect();
        failed.extend(
            differences(&words, threads, 8, &want)
                .into_iter()
                .map(|m| format!("{threads} threads: {m}")),
        );
    }
    assert!(failed.is_empty(), "{}", failed.join("\n"));
}

/// [`native`] for a group of two dimensions: thread ids x and y in `v0` and
/// `v1`, the group's x in `s1` (after the one user-data SGPR).
fn native_2d(words: &[u32], threads: [u32; 2]) -> NativeProgram {
    let mut n = native(words, true, threads[0] * threads[1], 0);
    n.io = StageIo::Cs(ComputeIo {
        // One user-data SGPR, the group's x (bit 7), thread ids x and y
        // (bits 11-12: 1).
        rsrc2: 1 << 1 | 1 << 7 | 1 << 11,
        threads: [threads[0], threads[1], 1],
    });
    n
}

/// [`while_loop_with_a_store`] in a group `width` threads wide (8 or 16), of
/// several rows, in a dispatch of several groups: the thread's flat id
/// `width y + x` is in `v0`, and a lane stays while
/// `v1 < ((flat + group) & 7)`, so the lanes leave at passes that differ
/// from group to group. Each lane has 16 dwords at
/// `(256 group + flat) * 64`: the accumulator of first lanes, its passes
/// at dwords 1 to 7, `flat + 1` added (atomically, so a thread id two
/// host lanes hold shows) at dword 8, and its count of passes at dword 9. `forever`: no lane ever leaves (the
/// loop then runs to the translation's limit of passes), and no pass stores.
fn while_loop_in_rows(forever: bool, width: u32) -> Vec<u32> {
    let mut w = PROLOGUE.to_vec();
    w.extend([
        0x3406_0280 | width.trailing_zeros(), // v_lshlrev_b32 v3, 3 or 4, v1
        0x4A00_0700,                          // v_add_nc_u32 v0, v0, v3
        0x3420_0086,                          // v_lshlrev_b32 v16, 6, v0
        0x7E08_0201,                          // v_mov_b32 v4, s1
        0x3408_088E,                          // v_lshlrev_b32 v4, 14, v4
        0x4A20_0910,                          // v_add_nc_u32 v16, v16, v4
        0x4A0C_0001,                          // v_add_nc_u32 v6, s1, v0
        0x7E02_0280,                          // v_mov_b32 v1, 0
        0x7E04_0280,                          // v_mov_b32 v2, 0
        if forever {
            0x7E0A_02C1 // v_mov_b32 v5, -1
        } else {
            0x360A_0C87 // v_and_b32 v5, 7, v6
        },
        0xBEA4_047E, // s_mov_b64 s[36:37], exec
    ]);
    if forever {
        w.extend([
            0xBF82_0003, // s_branch (the test)
            0x7E0C_0500, // v_readfirstlane_b32 s6, v0
            0x4A04_0406, // v_add_nc_u32 v2, s6, v2
            0x4A02_0281, // v_add_nc_u32 v1, 1, v1
            0x7D82_0B01, // v_cmp_lt_u32 vcc, v1, v5
            0x87FE_6A7E, // s_and_b64 exec, exec, vcc
            0xBF89_FFFA, // s_cbranch_execnz (the body)
        ]);
    } else {
        w.extend([
            0xBF82_0007, // s_branch (the test)
            0x7E0C_0500, // v_readfirstlane_b32 s6, v0
            0x4A04_0406, // v_add_nc_u32 v2, s6, v2
            0x4A02_0281, // v_add_nc_u32 v1, 1, v1
            0x3422_0282, // v_lshlrev_b32 v17, 2, v1
            0x4A22_2310, // v_add_nc_u32 v17, v16, v17
            0xE070_1000,
            0x800A_0111, // buffer_store_dword v1, v17, s[40:43], 0 offen
            0x7D82_0B01, // v_cmp_lt_u32 vcc, v1, v5
            0x87FE_6A7E, // s_and_b64 exec, exec, vcc
            0xBF89_FFF6, // s_cbranch_execnz (the body)
        ]);
    }
    w.extend([
        0xBEFE_0424, // s_mov_b64 exec, s[36:37]
        0x4A0E_0081, // v_add_nc_u32 v7, 1, v0
        0xE0C8_1020,
        0x800A_0710, // buffer_atomic_add v7, v16, s[40:43], 0 offen offset:32
    ]);
    w.extend(store(1, 36));
    w.extend(store(2, 0));
    w.push(S_ENDPGM);
    w
}

#[test]
fn the_while_loop_in_rows_is_what_its_comments_say() {
    for (forever, width) in [(false, 8), (true, 8), (false, 16)] {
        let words = while_loop_in_rows(forever, width);
        let code: Vec<u8> = words.iter().flat_map(|w| w.to_le_bytes()).collect();
        let text: Vec<String> = gfx10::decode(&code)
            .expect("decodes")
            .iter()
            .map(|i| i.text(true))
            .collect();
        let bound = if forever {
            "v_mov_b32 v5, -1"
        } else {
            "v_and_b32 v5, 7, v6"
        };
        let rows = if width == 8 {
            "v_lshlrev_b32 v3, 3, v1"
        } else {
            "v_lshlrev_b32 v3, 4, v1"
        };
        let mut want = vec![
            rows,
            "v_add_nc_u32 v0, v0, v3",
            "v_lshlrev_b32 v16, 6, v0",
            "v_mov_b32 v4, s1",
            "v_lshlrev_b32 v4, 14, v4",
            "v_add_nc_u32 v16, v16, v4",
            "v_add_nc_u32 v6, s1, v0",
            "v_mov_b32 v1, 0",
            "v_mov_b32 v2, 0",
            bound,
            "s_mov_b64 s[36:37], exec",
            "s_branch",
            "v_readfirstlane_b32 s6, v0",
            "v_add_nc_u32 v2, s6, v2",
            "v_add_nc_u32 v1, 1, v1",
        ];
        if !forever {
            want.extend([
                "v_lshlrev_b32 v17, 2, v1",
                "v_add_nc_u32 v17, v16, v17",
                "buffer_store_dword v1, v17, s[40:43], 0 offen",
            ]);
        }
        want.extend([
            "v_cmp_lt_u32 vcc, v1, v5",
            "s_and_b64 exec, exec, vcc",
            "s_cbranch_execnz",
            "s_mov_b64 exec, s[36:37]",
            "v_add_nc_u32 v7, 1, v0",
            "buffer_atomic_add v7, v16, s[40:43], 0 offen offset:32",
            "buffer_store_dword v1, v16, s[40:43], 0 offen offset:36",
        ]);
        // The prologue's five words are four instructions.
        for (n, want) in want.iter().enumerate() {
            let got = &text[PROLOGUE.len() - 1 + n];
            assert!(got.starts_with(want), "{n}: {got:?}, not {want:?}");
        }
    }
}

/// What the console stores for [`while_loop_in_rows`]'s lanes of group
/// `group`, `lanes` threads in console waves of 64: 16 dwords each.
#[cfg(windows)]
fn console_while_loop_in_rows(lanes: u32, group: u32) -> Vec<[u32; 16]> {
    let stays = |t: u32| (t + group) & 7;
    (0..lanes)
        .map(|t| {
            let mut lane = [0; 16];
            let wave = (t & !63)..lanes.min((t & !63) + 64);
            for pass in 0..stays(t) {
                // The first lane of its wave still in the loop in this pass.
                lane[0] += wave.clone().find(|&f| stays(f) > pass).expect("this one");
                lane[pass as usize + 1] = pass + 1;
            }
            lane[8] = t + 1;
            lane[9] = stays(t);
            lane
        })
        .collect()
}

/// A `while` loop in lockstep translates in halves in groups of rows.
#[test]
fn a_while_loop_in_rows_translates_in_halves() {
    for threads in [[8, 8], [8, 6], [16, 16]] {
        for forever in [false, true] {
            let n = native_2d(&while_loop_in_rows(forever, threads[0]), threads);
            let l =
                lowered(&n, true).unwrap_or_else(|e| panic!("{threads:?} forever={forever}: {e}"));
            assert_eq!(l.wave_size, Some(32));
            assert!(l.barriers > 0, "the halves meet in the loop");
        }
    }
}

/// A `while` loop in lockstep in groups of rows (8x8: two full host waves;
/// 8x6: 48 threads, the console wave's upper half 16 lanes; 16x16: four
/// console waves, as GTA VI's programs with such a loop have), several groups
/// to a dispatch, lanes leaving at different passes: every lane stores what
/// the console's does, its count of passes too, and each thread id is held
/// by one host lane. Left by no lane, the loop ends at the translation's
/// limit of passes in every lane of every group.
///
/// A group that stayed in the loop after its last lane left would store
/// nothing more, so that shows only in time. Time does not tell here: on an
/// M5 Pro 2,048 groups of 8x8 took 22 to 25 ms, and 56 to 62 ms with every
/// group at the limit (16,383 passes, two exchanges each), so the times are
/// printed, not checked.
#[cfg(windows)]
#[test]
fn a_while_loop_in_rows_matches_the_console_and_leaves_on_every_gpu() {
    let _one = super::valu_tests::ONE_AT_A_TIME
        .lock()
        .unwrap_or_else(|e| e.into_inner());
    let hosts = Host::each_hardware();
    if hosts.is_empty() {
        eprintln!("no SM 6.6 hardware device; skipped");
        return;
    }
    let mut failed = Vec::new();
    for (adapter, host) in &hosts {
        for threads in [[8u32, 8], [8, 6], [16, 16]] {
            let lanes = threads[0] * threads[1];
            for narrow_waves in [false, true] {
                let label = format!("{adapter} {threads:?} narrow_waves={narrow_waves}");
                let make = |forever: bool| {
                    let n = native_2d(&while_loop_in_rows(forever, threads[0]), threads);
                    let s = shader(lower_halves(&n, true, narrow_waves, false).expect("halves"));
                    if let Ok(mut v) =
                        validator::Validator::load(validator::Validator::default_path().as_deref())
                    {
                        v.validate(&mut s.dxil.clone())
                            .expect("DXIL validation failed");
                    }
                    let Ok((root, pso)) = host.compute_pipeline(&s) else {
                        panic!("{label}: no pipeline");
                    };
                    (s, root, pso)
                };
                let leaves = make(false);
                let stays = make(true);
                let run = |forever: bool, groups: u32| {
                    let (s, root, pso) = if forever { &stays } else { &leaves };
                    let ud = vec![0u32; s.user_data_dwords().max(1) as usize];
                    let views = [
                        View::Texture(1, 1, &[[0.0; 4]]),
                        View::Buffer {
                            data: &[0; 4],
                            stride: Stride::Structured(16),
                        },
                        View::RawUav(groups * 256 * 16),
                    ];
                    let start = std::time::Instant::now();
                    let out = host
                        .dispatch(
                            root,
                            pso,
                            &Dispatch {
                                roots: vec![(0, Root::Constants(ud))],
                                views: &views,
                                samplers: &[],
                                groups: [groups, 1, 1],
                            },
                        )
                        .unwrap_or_else(|e| panic!("{label}: dispatch failed: {e}"));
                    (out, start.elapsed())
                };
                for groups in [1u32, 5, 37] {
                    let (out, _) = run(false, groups);
                    let mut wrong = 0;
                    for g in 0..groups {
                        let want = console_while_loop_in_rows(lanes, g);
                        for (t, want) in want.iter().enumerate() {
                            let at = (g as usize * 256 + t) * 16;
                            let got = &out[0][at..at + 16];
                            if got != want {
                                wrong += 1;
                                failed.push(format!(
                                    "{label}: {groups} groups: group {g} lane {t}: {got:?}, the console {want:?}"
                                ));
                            }
                        }
                        // Nothing stored where the group has no thread.
                        let past = (g as usize * 256 + lanes as usize) * 16;
                        let end = (g as usize + 1) * 256 * 16;
                        if out[0][past..end].iter().any(|&d| d != 0) {
                            wrong += 1;
                            failed.push(format!(
                                "{label}: {groups} groups: group {g}: stores past its {lanes} threads"
                            ));
                        }
                    }
                    eprintln!("{label}: {groups} groups: {wrong} lanes differ from the console");
                }
                // Many groups, the best of three, and the loop no lane leaves
                // (every group at the limit of passes).
                let groups = 512;
                let best = (0..3).map(|_| run(false, groups).1).min().expect("three");
                let (out, limit) = run(true, groups);
                // Every lane of every group ran to the limit of passes.
                let passes: std::collections::BTreeSet<u32> = (0..groups as usize * 256)
                    .filter(|t| t % 256 < lanes as usize)
                    .map(|t| out[0][t * 16 + 9])
                    .collect();
                eprintln!("{label}: passes at the limit: {passes:?}");
                if passes != [super::DEFAULT_LOOP_LIMIT - 1].into() {
                    failed.push(format!("{label}: passes at the limit: {passes:?}"));
                }
                eprintln!(
                    "{label}: {groups} groups: {best:?}; with every group at the limit of passes {limit:?}"
                );
            }
        }
    }
    assert!(failed.is_empty(), "{}", failed.join("\n"));
}

/// A sum over the group's earlier console waves, in a `while` loop whose
/// exit test is in its header and whose latch reads the first lane (the
/// shape of the loops in GTA VI's acceleration-structure builder): every
/// lane writes its wave's number plus 5 to the wave's LDS dword; after a
/// barrier, the lanes of wave `w > 0` add dwords 0 to `w - 1` up, carrying
/// the sum from pass to pass through the first lane (`s17`), and leave when
/// the pass count `s18` is their wave. Each lane stores its sum.
fn sum_of_earlier_waves() -> Vec<u32> {
    let mut w = PROLOGUE.to_vec();
    w.extend([
        0x3420_0085, // v_lshlrev_b32 v16, 5, v0
        0x2C0E_0086, // v_lshrrev_b32 v7, 6, v0
        0x3410_0E82, // v_lshlrev_b32 v8, 2, v7
        0x4A12_0E85, // v_add_nc_u32 v9, 5, v7
        0xD834_0000,
        0x0000_0908, // ds_write_b32 v8, v9
        S_WAITCNT_LGKM,
        0xBF8A_0000, // s_barrier
        0x7E10_0280, // v_mov_b32 v8, 0
        0xBE80_047E, // s_mov_b64 s[0:1], exec
        0x7DAA_0E80, // v_cmpx_ne_u32 exec, 0, v7
        0xBF88_0012, // s_cbranch_execz (the end)
        0xBE82_047E, // s_mov_b64 s[2:3], exec
        0xBE86_047E, // s_mov_b64 s[6:7], exec
        0xBE92_0380, // s_mov_b32 s18, 0
        0xBE91_0380, // s_mov_b32 s17, 0
        0xBF82_0002, // s_branch (the header)
        0x87FE_0610, // s_and_b64 exec, s[16:17], s[6:7]
        0x7E22_0508, // v_readfirstlane_b32 s17, v8
        0x8F10_8212, // s_lshl_b32 s16, s18, 2
        0x8112_8112, // s_add_i32 s18, s18, 1
        0x7E10_0210, // v_mov_b32 v8, s16
        0xD8D8_0000,
        0x0800_0008, // ds_read_b32 v8, v8
        S_WAITCNT_LGKM,
        0x4A10_1011, // v_add_nc_u32 v8, s17, v8
        0xBE90_047E, // s_mov_b64 s[16:17], exec
        0x7DA4_0E12, // v_cmpx_eq_u32 exec, s18, v7
        0x8A86_7E06, // s_andn2_b64 s[6:7], s[6:7], exec
        0xBF85_FFF3, // s_cbranch_scc1 (the latch)
        0xBEFE_0400, // s_mov_b64 exec, s[0:1]
    ]);
    w.extend(store(8, 0));
    w.push(S_ENDPGM);
    w
}

#[test]
fn the_sum_of_earlier_waves_is_what_its_comments_say() {
    let words = sum_of_earlier_waves();
    let code: Vec<u8> = words.iter().flat_map(|w| w.to_le_bytes()).collect();
    let text: Vec<String> = gfx10::decode(&code)
        .expect("decodes")
        .iter()
        .map(|i| i.text(true))
        .collect();
    let want = [
        "v_lshlrev_b32 v16, 5, v0",
        "v_lshrrev_b32 v7, 6, v0",
        "v_lshlrev_b32 v8, 2, v7",
        "v_add_nc_u32 v9, 5, v7",
        "ds_write_b32 v8, v9",
        "s_waitcnt lgkmcnt(0)",
        "s_barrier",
        "v_mov_b32 v8, 0",
        "s_mov_b64 s[0:1], exec",
        "v_cmpx_ne_u32 exec, 0, v7",
        "s_cbranch_execz",
        "s_mov_b64 s[2:3], exec",
        "s_mov_b64 s[6:7], exec",
        "s_mov_b32 s18, 0",
        "s_mov_b32 s17, 0",
        "s_branch",
        "s_and_b64 exec, s[16:17], s[6:7]",
        "v_readfirstlane_b32 s17, v8",
        "s_lshl_b32 s16, s18, 2",
        "s_add_i32 s18, s18, 1",
        "v_mov_b32 v8, s16",
        "ds_read_b32 v8, v8",
        "s_waitcnt lgkmcnt(0)",
        "v_add_nc_u32 v8, s17, v8",
        "s_mov_b64 s[16:17], exec",
        "v_cmpx_eq_u32 exec, s18, v7",
        "s_andn2_b64 s[6:7], s[6:7], exec",
        "s_cbranch_scc1",
        "s_mov_b64 exec, s[0:1]",
    ];
    // The prologue's five words are four instructions.
    for (n, want) in want.iter().enumerate() {
        let got = &text[PROLOGUE.len() - 1 + n];
        assert!(got.starts_with(want), "{n}: {got:?}, not {want:?}");
    }
    // The three branches go to the end, the header and the latch.
    let at = |name: &str| {
        let i = gfx10::decode(&code).expect("decodes");
        i.iter()
            .find(|i| i.text(true).starts_with(name))
            .map(|i| i.text(true))
            .expect("there")
    };
    let offset = |n: usize| {
        let i = gfx10::decode(&code).expect("decodes");
        i[PROLOGUE.len() - 1 + n].offset
    };
    assert!(
        at("s_cbranch_execz").contains(&format!("{:04X}", offset(28))),
        "{}",
        at("s_cbranch_execz")
    );
    assert!(
        at("s_branch").contains(&format!("{:04X}", offset(18))),
        "{}",
        at("s_branch")
    );
    assert!(
        at("s_cbranch_scc1").contains(&format!("{:04X}", offset(16))),
        "{}",
        at("s_cbranch_scc1")
    );
}

/// The sum translates in halves, its loop in lockstep.
#[test]
fn a_sum_of_earlier_waves_translates_in_halves() {
    for threads in [128, 256, 512] {
        for narrow_waves in [false, true] {
            let n = native(&sum_of_earlier_waves(), true, threads, 512);
            let l = lowered_with(&n, true, narrow_waves)
                .unwrap_or_else(|e| panic!("{threads} threads narrow_waves={narrow_waves}: {e}"));
            assert_eq!(l.wave_size, Some(32));
        }
    }
}

/// Each lane of wave `w` stores `5 + 6 + ... + (w + 4)`, as on the console.
#[cfg(windows)]
#[test]
fn a_sum_of_earlier_waves_matches_the_console_on_every_gpu() {
    let words = sum_of_earlier_waves();
    let mut failed = Vec::new();
    for threads in [128u32, 256, 512] {
        let want: Vec<u32> = (0..threads)
            .map(|t| (0..t >> 6).map(|k| k + 5).sum())
            .collect();
        for narrow_waves in [false, true] {
            failed.extend(
                differences_with(&words, threads, 1, &want, 512, narrow_waves)
                    .into_iter()
                    .map(|m| format!("{threads} threads narrow_waves={narrow_waves}: {m}")),
            );
        }
    }
    assert!(failed.is_empty(), "{}", failed.join("\n"));
}

/// LDS written in a `while` loop of a group of one console wave (64
/// threads), the shape of the loops in the builder's `28918b7150a7aad5` and
/// `caa78b8ca62bc3d6`: each lane clears its eight LDS dwords; the lanes whose
/// count `tid & 7` is not 0 then loop, writing the pass number `s28` at
/// dword `s28 - 1` of theirs in each pass, and leave when `s28` is their
/// count (the latch restores EXEC from the mask of lanes left, `s[16:17]`).
/// After the loop each lane reads the eight dwords of lane `tid ^ 63`, in
/// the wave's other half, and stores them.
fn lds_written_in_a_while_loop() -> Vec<u32> {
    let mut w = PROLOGUE.to_vec();
    w.extend([
        0x3420_0085, // v_lshlrev_b32 v16, 5, v0
        0x3614_0087, // v_and_b32 v10, 7, v0
        0x3418_0085, // v_lshlrev_b32 v12, 5, v0
        0x7E16_0280, // v_mov_b32 v11, 0
    ]);
    for k in 0..8 {
        w.extend([0xD834_0000 | (4 * k), 0x0000_0B0C]); // ds_write_b32 v12, v11 offset:4k
    }
    w.extend([
        0xBE8C_047E, // s_mov_b64 s[12:13], exec
        0x7DAA_1480, // v_cmpx_ne_u32 exec, 0, v10
        0xBF88_000F, // s_cbranch_execz (the end)
        0xBE8E_047E, // s_mov_b64 s[14:15], exec
        0xBE90_047E, // s_mov_b64 s[16:17], exec
        0xBE9C_0380, // s_mov_b32 s28, 0
        0xBF82_0001, // s_branch (the header)
        0x87FE_1012, // s_and_b64 exec, s[18:19], s[16:17]
        0x8F12_821C, // s_lshl_b32 s18, s28, 2
        0x811C_811C, // s_add_i32 s28, s28, 1
        0x4A1A_1812, // v_add_nc_u32 v13, s18, v12
        0x7E1E_021C, // v_mov_b32 v15, s28
        0xBE92_047E, // s_mov_b64 s[18:19], exec
        0xD834_0000,
        0x0000_0F0D, // ds_write_b32 v13, v15
        0x7DA4_141C, // v_cmpx_eq_u32 exec, s28, v10
        0x8A90_7E10, // s_andn2_b64 s[16:17], s[16:17], exec
        0xBF85_FFF5, // s_cbranch_scc1 (the latch)
        0xBEFE_040C, // s_mov_b64 exec, s[12:13]
        0x3A1C_00BF, // v_xor_b32 v14, 63, v0
        0x341C_1C85, // v_lshlrev_b32 v14, 5, v14
    ]);
    for k in 0..8 {
        w.extend([0xD8D8_0000 | (4 * k), 0x0000_000E | (20 + k) << 24]); // ds_read_b32 v(20+k), v14 offset:4k
    }
    w.push(S_WAITCNT_LGKM);
    for k in 0..8 {
        w.extend(store(20 + k as u8, 4 * k));
    }
    w.push(S_ENDPGM);
    w
}

#[test]
fn the_lds_while_loop_is_what_its_comments_say() {
    let words = lds_written_in_a_while_loop();
    let code: Vec<u8> = words.iter().flat_map(|w| w.to_le_bytes()).collect();
    let list = gfx10::decode(&code).expect("decodes");
    let text: Vec<String> = list.iter().map(|i| i.text(true)).collect();
    let all = text.join("\n");
    for want in [
        "v_lshlrev_b32 v16, 5, v0",
        "v_and_b32 v10, 7, v0",
        "v_lshlrev_b32 v12, 5, v0",
        "v_mov_b32 v11, 0",
        "ds_write_b32 v12, v11\n",
        "ds_write_b32 v12, v11 offset:28",
        "s_mov_b64 s[12:13], exec",
        "v_cmpx_ne_u32 exec, 0, v10",
        "s_mov_b64 s[14:15], exec",
        "s_mov_b64 s[16:17], exec",
        "s_mov_b32 s28, 0",
        "s_and_b64 exec, s[18:19], s[16:17]",
        "s_lshl_b32 s18, s28, 2",
        "s_add_i32 s28, s28, 1",
        "v_add_nc_u32 v13, s18, v12",
        "v_mov_b32 v15, s28",
        "s_mov_b64 s[18:19], exec",
        "ds_write_b32 v13, v15",
        "v_cmpx_eq_u32 exec, s28, v10",
        "s_andn2_b64 s[16:17], s[16:17], exec",
        "s_mov_b64 exec, s[12:13]",
        "v_xor_b32 v14, 63, v0",
        "v_lshlrev_b32 v14, 5, v14",
        "ds_read_b32 v20, v14\n",
        "ds_read_b32 v27, v14 offset:28",
    ] {
        assert!(all.contains(want), "{want:?} not in:\n{all}");
    }
    // The branches: to the end, the header and the latch.
    let at = |name: &str| {
        list.iter()
            .position(|i| i.text(true).starts_with(name))
            .expect("there")
    };
    let target = |n: usize| format!("{:04X}", list[n].offset);
    let (end, header, latch) = (
        at("s_mov_b64 exec, s[12:13]"),
        at("s_lshl_b32 s18"),
        at("s_and_b64 exec, s[18:19]"),
    );
    assert!(
        text[at("s_cbranch_execz")].contains(&target(end)),
        "{}",
        text[at("s_cbranch_execz")]
    );
    assert!(
        text[at("s_branch")].contains(&target(header)),
        "{}",
        text[at("s_branch")]
    );
    assert!(
        text[at("s_cbranch_scc1")].contains(&target(latch)),
        "{}",
        text[at("s_cbranch_scc1")]
    );
}

/// Each lane stores the dwords of lane `tid ^ 63`: 1 to that lane's count,
/// then zeros, as on the console.
#[cfg(windows)]
#[test]
fn lds_written_in_a_while_loop_matches_the_console_on_every_gpu() {
    let words = lds_written_in_a_while_loop();
    let want: Vec<u32> = (0..64u32)
        .flat_map(|t| {
            let n = (t ^ 63) & 7;
            (1..=8).map(move |p| if p <= n { p } else { 0 })
        })
        .collect();
    let mut failed = Vec::new();
    for narrow_waves in [true, false] {
        let n = native(&words, true, 64, 2048);
        match lowered_with(&n, true, narrow_waves) {
            Ok(l) => eprintln!(
                "narrow_waves={narrow_waves}: translates, {} barriers",
                l.barriers
            ),
            Err(e) => {
                eprintln!("narrow_waves={narrow_waves}: refused: {e}");
                continue;
            }
        }
        failed.extend(
            differences_with(&words, 64, 8, &want, 2048, narrow_waves)
                .into_iter()
                .map(|m| format!("narrow_waves={narrow_waves}: {m}")),
        );
    }
    assert!(failed.is_empty(), "{}", failed.join("\n"));
}

/// A register-index waterfall as the console compiler writes one for a
/// per-lane `v[3 + index]`: `v3..v10` hold `k * 1000 + thread id`, the index
/// is `(thread id >> 2) & 7`, and the loop reads one index left
/// (`v_readfirstlane_b32 s38, v28`), runs the lanes holding it with
/// `m0 = s38` and `v_movrels_b32 v31, v3`, drops them, and loops while any
/// lane is left. Every lane stores `v31`: `index * 1000 + thread id`.
/// `with_mbcnt` asks for console lanes as well (so the program runs in
/// halves); `after` is put after the loop, `body` before the loop's
/// `s_xor_b64`.
fn index_waterfall(with_mbcnt: bool, body: &[u32], after: &[u32]) -> Vec<u32> {
    let mut w = PROLOGUE.to_vec();
    w.push(0x3420_0085); // v_lshlrev_b32 v16, 5, v0
    if with_mbcnt {
        w.extend([
            0xD765_0001,
            0x0001_00C1, // v_mbcnt_lo_u32_b32 v1, -1, 0
        ]);
    }
    for k in 0..8u32 {
        // v_add_nc_u32 v(3 + k), k * 1000, v0
        w.extend([0x4A00_00FF | (3 + k) << 17, k * 1000]);
    }
    w.extend([
        0x2C38_0082, // v_lshrrev_b32 v28, 2, v0
        0x3638_3887, // v_and_b32 v28, 7, v28
        0xBEA4_047E, // s_mov_b64 s[36:37], exec
    ]);
    let looped = [
        &[
            0x7E4C_051C, // v_readfirstlane_b32 s38, v28
            0x7D84_3826, // v_cmp_eq_u32 vcc, s38, v28
            0xBE86_246A, // s_and_saveexec_b64 s[6:7], vcc
            0xBEFC_0326, // s_mov_b32 m0, s38
            0x7E3E_8703, // v_movrels_b32 v31, v3
        ][..],
        body,
        &[0x89FE_067E], // s_xor_b64 exec, exec, s[6:7]
    ]
    .concat();
    let back = -(looped.len() as i32 + 1);
    w.extend(&looped);
    w.push(0xBF89_0000 | (back as u32 & 0xFFFF)); // s_cbranch_execnz (the read)
    w.push(0xBEFE_0424); // s_mov_b64 exec, s[36:37]
    w.extend(after);
    w.extend(store(31, 0));
    w.push(S_ENDPGM);
    w
}

#[test]
fn the_index_waterfall_programs_are_what_their_comments_say() {
    let words = index_waterfall(true, &[], &[]);
    let code: Vec<u8> = words.iter().flat_map(|w| w.to_le_bytes()).collect();
    let all = gfx10::decode(&code)
        .expect("decodes")
        .iter()
        .map(|i| i.text(true))
        .collect::<Vec<_>>()
        .join("\n");
    for want in [
        "v_add_nc_u32 v3, lit(0x00000000), v0",
        "v_add_nc_u32 v10, lit(0x00001b58), v0",
        "v_lshrrev_b32 v28, 2, v0",
        "v_and_b32 v28, 7, v28",
        "s_mov_b64 s[36:37], exec",
        "v_readfirstlane_b32 s38, v28",
        "v_cmp_eq_u32 vcc, s38, v28",
        "s_and_saveexec_b64 s[6:7], vcc",
        "s_mov_b32 m0, s38",
        "v_movrels_b32 v31, v3",
        "s_xor_b64 exec, exec, s[6:7]",
        "s_cbranch_execnz",
        "s_mov_b64 exec, s[36:37]",
    ] {
        assert!(all.contains(want), "{want:?} not in:\n{all}");
    }
}

/// The first-lane read opening a register-index waterfall is made by each
/// host half by itself: the program needs no halves for it, and translates
/// in halves too although the read is in a loop (#522 sent GTA VI's
/// `11eeb27c2d15980d` to halves for it, which refused it). Where the read's
/// result is seen after the loop, or the loop body does scalar work, whose
/// iterations the halves would see, the read still takes halves (refused
/// where they cannot meet).
#[test]
fn a_register_index_waterfall_is_made_in_each_half() {
    let across = "a first-lane read of a value that may differ between 32-lane halves";
    let halves = |words: &[u32], threads: u32| match lowered(&native(words, true, threads, 0), true)
    {
        Ok(l) => l.wave_size == Some(32),
        Err(e) => e.to_string().contains(across),
    };
    for threads in [64, 128] {
        assert!(!halves(&index_waterfall(false, &[], &[]), threads));
        assert!(halves(&index_waterfall(true, &[], &[]), threads));
        // v_mov_b32 v31, s38 after the loop
        assert!(halves(
            &index_waterfall(false, &[], &[0x7E3E_0226]),
            threads
        ));
        // s_add_u32 s50, s50, 1 in the body
        assert!(halves(
            &index_waterfall(false, &[0x8032_8132], &[]),
            threads
        ));
    }
}

/// The register-index waterfall gives the console's lanes on every GPU, in
/// halves and not: `((thread id >> 2) & 7) * 1000 + thread id` in every
/// lane.
#[cfg(windows)]
#[test]
fn a_register_index_waterfall_matches_the_console_on_every_gpu() {
    let mut failed = Vec::new();
    for threads in [64, 128] {
        let want: Vec<u32> = (0..threads).map(|t| ((t >> 2) & 7) * 1000 + t).collect();
        for with_mbcnt in [false, true] {
            let words = index_waterfall(with_mbcnt, &[], &[]);
            failed.extend(
                differences(&words, threads, 1, &want)
                    .into_iter()
                    .map(|m| format!("{threads} threads (mbcnt={with_mbcnt}): {m}")),
            );
        }
    }
    assert!(failed.is_empty(), "{}", failed.join("\n"));
}
