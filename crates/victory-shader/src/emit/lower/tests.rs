use super::{
    Lowered, Numeric, Parts, barycentric_signature, buffer_format, code_table_start, half_shift,
    lower, packed_half,
};
use crate::native::{
    ComputeIo, HwStage, NativeProgram, ResourceType, RootParam, RootType, StageIo, UsedResource,
};
use crate::{io, ir, resource, structure};

const S_ENDPGM: u32 = 0xBF81_0000;

#[test]
fn direct_barycentric_modes_match_dxil_semantics() {
    use crate::io::Barycentric::*;

    assert_eq!(barycentric_signature(PerspectiveSample), Some((0, 6)));
    assert_eq!(barycentric_signature(PerspectiveCenter), Some((0, 2)));
    assert_eq!(barycentric_signature(PerspectiveCentroid), Some((0, 3)));
    assert_eq!(barycentric_signature(LinearSample), Some((1, 7)));
    assert_eq!(barycentric_signature(LinearCenter), Some((1, 4)));
    assert_eq!(barycentric_signature(LinearCentroid), Some((1, 5)));
    assert_eq!(barycentric_signature(PerspectivePullModel), None);
}

/// A program's constant table starts after its instructions, unless a
/// load at a fixed offset reads below that.
#[test]
fn the_code_table_starts_after_the_instructions() {
    // s_nop 0, s_endpgm, then 1,000 dwords of constants.
    let mut words = vec![0xBF80_0000, S_ENDPGM];
    words.extend(0..1000u32);
    let code: Vec<u8> = words.iter().flat_map(|w| w.to_le_bytes()).collect();
    let p = ir::build(&code, false, 256).unwrap();
    let at = |offset: u32, dynamic: bool| resource::Address {
        param: resource::CODE,
        offset,
        dynamic: dynamic.then_some(ir::Value(0)),
    };
    let (computed, fixed_after, fixed_inside) = (at(0, true), at(12, false), at(4, false));
    // A load at a computed offset from the code's start: the table is
    // the constants only, not the whole program.
    assert_eq!(code_table_start(&p, &[&computed], 0), 8);
    assert_eq!(code_table_start(&p, &[&computed, &fixed_after], 0), 8);
    // A load at a fixed offset inside the instructions keeps the start.
    assert_eq!(code_table_start(&p, &[&computed, &fixed_inside], 0), 0);
    // Loads past the instructions only: from the lowest of them.
    assert_eq!(code_table_start(&p, &[&fixed_after], 12), 12);
}

/// Whether a synthetic compute program translates with these input registers.
fn lowers(words: &[u32], rsrc2: u32) -> Result<(), crate::ShaderError> {
    let code: Vec<u8> = words.iter().flat_map(|w| w.to_le_bytes()).collect();
    let native = NativeProgram {
        stage: HwStage::Cs,
        wave64: false,
        vgprs: 256,
        layout: vec![],
        used: vec![],
        io: StageIo::Cs(ComputeIo {
            rsrc2,
            threads: [1, 1, 1],
        }),
        code,
    };
    let p = ir::build(&native.code, false, 256).unwrap();
    let st = structure::structure(&p).unwrap();
    let res = resource::track(&p, &native.layout, native.stage, Some(&st)).unwrap();
    let io = io::raise(&p, &native).unwrap();
    lower(&Parts {
        halves_rendezvous: false,
        native: &native,
        p: &p,
        st: &st,
        io: &io,
        res: &res,
        inputs: vec![],
        outputs: vec![],
        prims: vec![],
        mesh_stride: 0,
        atomic_locks: false,
        atomic_lock_rounds: false,
            atomic_umax: false,
        bias_as_gradients: false,
        host_wave32: false,
        narrow_waves: false,
        uav_size_level0: false,
        view_marks: false,
    })
    .map(|_| ())
}

/// Lowers a compute program with the root parameters `layout` (in user
/// SGPRs from s0 on) and the used resources `used`.
fn lower_with(
    words: &[u32],
    layout: Vec<RootParam>,
    used: Vec<UsedResource>,
) -> Result<Lowered, crate::ShaderError> {
    lower_wave(words, layout, used, false)
}

/// [`lower_with`] in wave64 if `wave64`.
fn lower_wave(
    words: &[u32],
    layout: Vec<RootParam>,
    used: Vec<UsedResource>,
    wave64: bool,
) -> Result<Lowered, crate::ShaderError> {
    let code: Vec<u8> = words.iter().flat_map(|w| w.to_le_bytes()).collect();
    let sgprs: u32 = layout.iter().map(|p| p.sgprs).sum();
    let native = NativeProgram {
        stage: HwStage::Cs,
        wave64,
        vgprs: 256,
        layout,
        used,
        io: StageIo::Cs(ComputeIo {
            // The layout's user SGPRs (at least two), 512 bytes of LDS.
            rsrc2: 1 << 15 | sgprs.max(2) << 1,
            threads: [1, 1, 1],
        }),
        code,
    };
    let p = ir::build(&native.code, wave64, 256).unwrap();
    let st = structure::structure(&p).unwrap();
    let res = resource::track(&p, &native.layout, native.stage, Some(&st))?;
    let io = io::raise(&p, &native).unwrap();
    lower(&Parts {
        halves_rendezvous: false,
        native: &native,
        p: &p,
        st: &st,
        io: &io,
        res: &res,
        inputs: vec![],
        outputs: vec![],
        prims: vec![],
        mesh_stride: 0,
        atomic_locks: false,
        atomic_lock_rounds: false,
        atomic_umax: false,
        bias_as_gradients: false,
        host_wave32: false,
        narrow_waves: false,
        uav_size_level0: false,
        view_marks: false,
    })
}

/// Root parameter `slot` in user SGPRs from `user_data` on.
fn root(root: RootType, user_data: u8, slot: u8) -> RootParam {
    RootParam {
        root,
        table_offset_dwords: 0,
        sgprs: if root == RootType::Descriptor { 2 } else { 1 },
        base_register: 0,
        user_data,
        slot,
    }
}

/// Lowers a one-thread compute program that copies four root constants
/// (s0..s3) into v1..v4, runs `body`, and stores v0 to a raw buffer from a
/// descriptor table, so the vector instructions in `body` have an effect.
fn lower_valu(body: &[u32]) -> Result<Lowered, crate::ShaderError> {
    let mut words = vec![
        0xBE85_0380, // s_mov_b32 s5, 0
        0xF408_0202, // s_load_dwordx4 s[8:11], s[4:5], 0
        0xFA00_0000,
    ];
    // v_mov_b32 v1..v4, s0..s3
    words.extend((0..4u32).map(|i| 0x7E00_0200 | (i + 1) << 17 | i));
    words.extend_from_slice(body);
    words.extend_from_slice(&[
        0xE070_0000, // buffer_store_dword v0, off, s[8:11], 0
        0x8002_0000,
        S_ENDPGM,
    ]);
    let constants = RootParam {
        sgprs: 4,
        ..root(RootType::Constants, 0, 0)
    };
    let buffer = UsedResource {
        kind: ResourceType::Uav,
        root: RootType::Table,
        table_offset_dwords: 0,
        lower_bound: 0,
        stride: 0,
        dwords: 4,
        user_data: 4,
        slot: 1,
    };
    lower_with(
        &words,
        vec![constants, root(RootType::Table, 4, 1)],
        vec![buffer],
    )
}

fn lowers_valu(body: &[u32]) -> Result<(), crate::ShaderError> {
    lower_valu(body).map(|_| ())
}

#[test]
fn vop3_conversion_modifiers_affect_the_lowering() {
    for (op, first, second, modifier) in [
        (0x0Au32, 1 << 15, 0, "clamp"),
        (0x0B, 1 << 8, 0, "abs"),
        (0x0B, 0, 1 << 29, "neg"),
    ] {
        let vop1 = lower_valu(&[0x7E00_0000 | op << 9 | 0x101])
            .unwrap()
            .dxil;
        let vop3 = |first: u32, second: u32| {
            lower_valu(&[
                0xD400_0000 | (0x180 + op) << 16 | first,
                0x101 | second,
            ])
            .unwrap()
            .dxil
        };
        assert_eq!(vop1, vop3(0, 0), "VOP1/VOP3 control for op {op:#x}");
        assert_ne!(
            vop1,
            vop3(first, second),
            "VOP3 {modifier} was dropped for op {op:#x}"
        );
    }
}

/// The f32 rounding ops that read a denormal operand as a zero of its sign
/// (`v_fract_f32`, `v_floor_f32`, `v_ceil_f32`) flush it on the bits before
/// the DXIL unary op; `v_trunc_f32` and `v_rndne_f32` already round
/// subnormals to signed zero, so they take the operand as is.
#[test]
fn fract_floor_and_ceil_flush_denormal_operands() {
    // VOP1 `v_<op>_f32 v0, v1`, its DXIL unary opcode, and whether the
    // operand is flushed first.
    for (op, unary, flushed) in [
        (0x20u32, 22, true), // v_fract_f32
        (0x24, 27, true),    // v_floor_f32
        (0x22, 28, true),    // v_ceil_f32
        (0x23, 26, false),   // v_rndne_f32
        (0x21, 29, false),   // v_trunc_f32
    ] {
        let l = lower_valu(&[0x7E00_0000 | op << 9 | 0x101]).unwrap();
        assert_eq!(
            l.unary_f32,
            [(unary, flushed)],
            "VOP1 {op:#04x} lowers to DXIL unary {unary}"
        );
    }
}

#[test]
fn the_24_bit_multiplies_lower_both_halves() {
    // VOP2 `op v0, v1, v2`: v_mul_i32_i24 and v_mul_u32_u24 (the low
    // dword) and v_mul_hi_i32_i24 and v_mul_hi_u32_u24 (the high one).
    for op in [0x09u32, 0x0A, 0x0B, 0x0C] {
        let word = op << 25 | 2 << 9 | 0x101;
        lowers_valu(&[word]).unwrap_or_else(|e| panic!("VOP2 {op:#x}: {e}"));
    }
}

#[test]
fn sums_of_absolute_differences_lower() {
    // VOP3 `op v0, v1, v2, v3`: v_sad_u8, v_sad_hi_u8, v_sad_u16,
    // v_sad_u32 and v_msad_u8.
    for op in [0x15Au32, 0x15B, 0x15C, 0x15D, 0x171] {
        let words = [0xD400_0000 | op << 16, 0x101 | 0x102 << 9 | 0x103 << 18];
        lowers_valu(&words).unwrap_or_else(|e| panic!("VOP3 {op:#x}: {e}"));
    }
}

#[test]
fn buffer_cache_invalidations_lower_as_no_ops() {
    // MUBUF opcode 0x071/0x072 with no data operand: the title asks the
    // host to invalidate its cached view of a buffer.
    assert!(lowers(&[0xE1C4_0000, 0, S_ENDPGM], 0).is_ok(), "buffer_gl0_inv");
    assert!(lowers(&[0xE1C8_0000, 0, S_ENDPGM], 0).is_ok(), "buffer_gl1_inv");
}

/// A full typed store to element `s0` through the table at `s1`, with
/// `soffset` in the MUBUF instruction's scalar-source field.
fn typed_store_words(soffset: u8) -> Vec<u32> {
    let mut words = vec![
        0xBE84_0301, // s_mov_b32 s4, s1
        0xBE85_0380, // s_mov_b32 s5, 0
        0xF408_0202,
        0xFA00_0000, // s_load_dwordx4 s[8:11], s[4:5], 0
        0x7E00_0200, // v_mov_b32 v0, s0
        0xBF8C_C07F, // s_waitcnt lgkmcnt(0)
    ];
    words.extend((1..=4).map(|v| 0x7E00_0000 | (v as u32) << 17 | 0x01 << 9 | 0x80));
    words.extend([
        0xE01C_2000,
        0x8002_0100 | (soffset as u32) << 24, // buffer_store_format_xyzw v[1:4], v0, s[8:11], 0 idxen
        S_ENDPGM,
    ]);
    words
}

#[test]
fn typed_stores_refuse_register_and_immediate_soffsets() {
    let layout = || vec![root(RootType::Constants, 0, 0), root(RootType::Table, 1, 1)];
    // RGBA32 and RGB32 use the same full-store MUBUF form. The mark width
    // comes from the host's stride table; the used-resource stride here
    // keeps both descriptor layouts in the regression matrix.
    for (format, stride) in [("RGBA32", 16), ("RGB32", 12)] {
        let used = || {
            vec![UsedResource {
                kind: ResourceType::Uav,
                root: RootType::Table,
                table_offset_dwords: 0,
                lower_bound: 0,
                stride,
                dwords: 4,
                user_data: 1,
                slot: 1,
            }]
        };
        // A register and an inline constant both make an instruction-valued
        // offset. Neither may be silently dropped until offsets can fold into
        // the typed element index. Inline zero is the accepted no-offset form.
        for (source, soffset) in [("register", 6), ("immediate", 132)] {
            let Err(crate::ShaderError::Unsupported { what, .. }) =
                lower_with(&typed_store_words(soffset), layout(), used())
            else {
                panic!("translated a {format} typed store with a {source} soffset");
            };
            assert_eq!(
                what, "a typed buffer store at an offset",
                "{format} {source} soffset"
            );
        }
        lower_with(&typed_store_words(128), layout(), used())
            .unwrap_or_else(|e| panic!("{format} inline zero soffset: {e}"));
    }
}

#[test]
fn a_scalar_carry_chain_reaches_the_select() {
    // The high dword of a 64-bit add or subtract, then a select on the
    // carry (borrow) the high half leaves in SCC, and a branch on it
    // (only code with an effect needs its operands).
    let add = [
        0xBE80_03C1, // s_mov_b32 s0, -1
        0xBE81_0380, // s_mov_b32 s1, 0
        0x8000_8100, // s_add_u32 s0, s0, 1
        0x8201_8001, // s_addc_u32 s1, s1, 0
        0x8502_0001, // s_cselect_b32 s2, s1, s0
        0xBF07_8002, // s_cmp_lg_u32 s2, 0
        0xBF85_0001, // s_cbranch_scc1 -> s_endpgm
        0xBE83_0380, // s_mov_b32 s3, 0
        S_ENDPGM,
    ];
    let sub = [
        0xBE80_0380, // s_mov_b32 s0, 0
        0xBE81_0380, // s_mov_b32 s1, 0
        0x8080_8100, // s_sub_u32 s0, s0, 1
        0x8281_8001, // s_subb_u32 s1, s1, 0
        0x8502_0001, // s_cselect_b32 s2, s1, s0
        0xBF07_8002, // s_cmp_lg_u32 s2, 0
        0xBF85_0001, // s_cbranch_scc1 -> s_endpgm
        0xBE83_0380, // s_mov_b32 s3, 0
        S_ENDPGM,
    ];

    for words in [add, sub] {
        lowers(&words, 0).unwrap();
    }
}

#[test]
fn v_cndmask_accepts_uniform_and_constant_masks() {
    // Uniform select (`s_cselect_b32`) and inline constant masks in `v_cndmask_b32`
    // (resolved in `Structured::other` or directly as constants).
    let words = [
        0xBF06_8000u32, // s_cmp_eq_u32 s0, 0
        0x8500_80C1,    // s_cselect_b32 s0, -1, 0
        0xD501_0001,
        0x0001_0080,    // v_cndmask_b32 v1, 0, 1, s0
        0xD501_0002,
        0x0201_02C1,    // v_cndmask_b32 v2, 1, v1, -1
        0xD834_0000,
        0x0000_0200,    // ds_write_b32 v0, v2
        S_ENDPGM,
    ];
    lowers(&words, 0x8001).unwrap();
}

fn perm_features(selector: u32) -> u64 {
    // Synthetic compute shader:
    // v0 is the thread ID and can supply a dynamic selector.
    // Store the result in LDS to keep the permutation live.
    let words = [
        0x7E02_02FFu32,
        0x817F_34AB,
        0x7E04_02FF,
        0xFE01_56CD,
        0xD744_0003,
        257 | (258 << 9) | (selector << 18),
        0x3408_0082,
        0xD834_0000,
        0x0000_0304,
        S_ENDPGM,
    ];

    let code: Vec<u8> = words.iter().flat_map(|w| w.to_le_bytes()).collect();

    let native = NativeProgram {
        stage: HwStage::Cs,
        wave64: false,
        vgprs: 256,
        layout: vec![],
        used: vec![],
        io: StageIo::Cs(ComputeIo {
            rsrc2: 1 << 15,
            threads: [64, 1, 1],
        }),
        code,
    };

    let p = ir::build(&native.code, false, 256).unwrap();

    assert!(
        p.blocks
            .iter()
            .flat_map(|b| &b.insts)
            .any(|i| i.name() == "v_perm_b32")
    );

    let st = structure::structure(&p).unwrap();
    let res = resource::track(&p, &native.layout, native.stage, Some(&st)).unwrap();

    let io = io::raise(&p, &native).unwrap();

    let emitted = lower(&Parts {
        halves_rendezvous: false,
        native: &native,
        p: &p,
        st: &st,
        io: &io,
        res: &res,
        inputs: vec![],
        outputs: vec![],
        prims: vec![],
        mesh_stride: 0,
        atomic_locks: false,
        atomic_lock_rounds: false,
        atomic_umax: false,
        bias_as_gradients: false,
        host_wave32: false,
        narrow_waves: false,
        uav_size_level0: false,
        view_marks: false,
    })
    .unwrap();

    let container = crate::Container::parse(&emitted.dxil).unwrap();

    let features = container.part(crate::FourCC::SFI0).unwrap();

    u64::from_le_bytes(features.data.try_into().unwrap())
}

#[test]
fn perm_constant_selector_needs_no_int64() {
    assert_eq!(perm_features(128) & 0x8000, 0);
}

#[test]
fn perm_dynamic_selector_needs_no_int64() {
    assert_eq!(
        perm_features(256) & 0x8000,
        0,
        "dynamic v_perm_b32 must not require INT64"
    );
}

#[test]
fn a_descriptor_selected_by_scc_picks_its_heap_index() {
    // Two buffer descriptors loaded from the table at s0, one picked by
    // a root constant (s1) with s_cselect_b64 per half, and loaded
    // through by index and stored to LDS so the load is needed.
    let words = [
        0xBE84_0300, // s_mov_b32 s4, s0
        0xBE85_0380, // s_mov_b32 s5, 0
        0xF408_0202,
        0xFA00_0000, // s_load_dwordx4 s[8:11], s[4:5], null
        0xF408_0302,
        0xFA00_0010, // s_load_dwordx4 s[12:15], s[4:5], 0x10
        0xBF06_8001, // s_cmp_eq_u32 s1, 0
        0x8590_0C08, // s_cselect_b64 s[16:17], s[8:9], s[12:13]
        0x8592_0E0A, // s_cselect_b64 s[18:19], s[10:11], s[14:15]
        0xE030_2000,
        0x8004_0100, // buffer_load_dword v1, v0, s[16:19], 0 idxen
        0xD834_0000,
        0x0000_0100, // ds_write_b32 v0, v1
        S_ENDPGM,
    ];
    let layout = || vec![root(RootType::Table, 0, 0), root(RootType::Constants, 1, 1)];
    // Both are structured buffers of dwords in the table's range.
    let used = || {
        vec![UsedResource {
            kind: ResourceType::Srv,
            root: RootType::Table,
            table_offset_dwords: 0,
            lower_bound: 0,
            stride: 4,
            dwords: 4,
            user_data: 0,
            slot: 0,
        }]
    };
    lower_with(&words, layout(), used()).unwrap();
}

/// Tables of 2D textures at s0 and samplers at s1.
fn texture_tables() -> (Vec<RootParam>, Vec<UsedResource>) {
    let range = |kind, user_data, slot| UsedResource {
        kind,
        root: RootType::Table,
        table_offset_dwords: 0,
        lower_bound: 0,
        stride: 0,
        dwords: if kind == ResourceType::Sampler { 4 } else { 8 },
        user_data,
        slot,
    };
    (
        vec![root(RootType::Table, 0, 0), root(RootType::Table, 1, 1)],
        vec![
            range(ResourceType::Srv, 0, 0),
            range(ResourceType::Sampler, 1, 1),
        ],
    )
}

#[test]
fn a_descriptor_loaded_only_for_its_size_takes_the_kind_of_its_reload() {
    // The record count of the buffer at s0 (count - 1 as an index), then
    // the same descriptor loaded again for the indexed read.
    let words = [
        0xBE84_0300, // s_mov_b32 s4, s0
        0xBE85_0380, // s_mov_b32 s5, 0
        0xF408_0202,
        0xFA00_0000, // s_load_dwordx4 s[8:11], s[4:5], null
        0x810C_C10A, // s_add_i32 s12, s10, -1
        0xF408_0402,
        0xFA00_0000, // s_load_dwordx4 s[16:19], s[4:5], null
        0x7E04_020C, // v_mov_b32 v2, s12
        0xE030_2000,
        0x8004_0102, // buffer_load_dword v1, v2, s[16:19], 0 idxen
        0xD834_0000,
        0x0000_0100, // ds_write_b32 v0, v1
        S_ENDPGM,
    ];
    let layout = vec![root(RootType::Table, 0, 0), root(RootType::Constants, 1, 1)];
    let used = vec![UsedResource {
        kind: ResourceType::Srv,
        root: RootType::Table,
        table_offset_dwords: 0,
        lower_bound: 0,
        stride: 4,
        dwords: 4,
        user_data: 0,
        slot: 0,
    }];
    lower_with(&words, layout, used).unwrap();
}

#[test]
fn a_waterfall_loop_that_only_sizes_takes_the_kind_of_the_sampled_one() {
    // Each lane names table entry (lane & 1) by its byte offset in v7.
    // A first waterfall loop only reads the width bits of the entry
    // (`v_alignbit_b32`), a second one samples it; the width goes to the
    // buffer at dword 16 of the table.
    let words = [
        0xBE84_0300, // s_mov_b32 s4, s0
        0xBE85_0380, // s_mov_b32 s5, 0
        0xBE86_0301, // s_mov_b32 s6, s1
        0xBE87_0380, // s_mov_b32 s7, 0
        0xF408_0203,
        0xFA00_0000, // s_load_dwordx4 s[8:11], s[6:7], 0x0
        0xF408_0A02,
        0xFA00_0040, // s_load_dwordx4 s[40:43], s[4:5], 0x40
        0x7E04_0D00, // v_cvt_f32_u32 v2, v0
        0x7E06_0302, // v_mov_b32 v3, v2
        0x3420_0082, // v_lshlrev_b32 v16, 2, v0
        0x360E_0081, // v_and_b32 v7, 1, v0
        0x340E_0E85, // v_lshlrev_b32 v7, 5, v7
        0xBF8C_C07F, // s_waitcnt lgkmcnt(0)
        0xBE97_037E, // s_mov_b32 s23, exec_lo
        0xBE95_037E, // s_mov_b32 s21, exec_lo
        0x7E28_0507, // L1: v_readfirstlane_b32 s20, v7
        0x7DA4_0E14, // v_cmpx_eq_u32 exec_lo, s20, v7
        0xF40C_0302,
        0x2800_0000, // s_load_dwordx8 s[12:19], s[4:5], s20
        0xBF8C_C07F, // s_waitcnt lgkmcnt(0)
        0xD54E_0008,
        0x0278_1A0E, // v_alignbit_b32 v8, s14, s13, 30
        0xBE95_4715, // s_andn2_wrexec_b32 s21, s21
        0xBF89_FFF7, // s_cbranch_execnz L1
        0xBEFE_0317, // s_mov_b32 exec_lo, s23
        0xBE95_037E, // s_mov_b32 s21, exec_lo
        0x7E28_0507, // L2: v_readfirstlane_b32 s20, v7
        0x7DA4_0E14, // v_cmpx_eq_u32 exec_lo, s20, v7
        0xF40C_0302,
        0x2800_0000, // s_load_dwordx8 s[12:19], s[4:5], s20
        0xBF8C_C07F, // s_waitcnt lgkmcnt(0)
        0xF09C_0F08,
        0x0043_0C02, // image_sample_lz v[12:15], v[2:3], s[12:19], s[8:11] dmask:0xf dim:2D
        0xBF8C_3F70, // s_waitcnt vmcnt(0)
        0xBE95_4715, // s_andn2_wrexec_b32 s21, s21
        0xBF89_FFF6, // s_cbranch_execnz L2
        0xBEFE_0317, // s_mov_b32 exec_lo, s23
        0xE070_1000,
        0x800A_0810, // buffer_store_dword v8, v16, s[40:43], 0 offen
        S_ENDPGM,
    ];
    let (layout, used) = texture_and_uav_tables();
    lower_wave(&words, layout, used, false).unwrap();
}

#[test]
fn permlane16_and_permlanex16_lower_in_wave32_and_wave64() {
    let layout = vec![RootParam {
        sgprs: 2,
        ..root(RootType::Constants, 0, 0)
    }];
    for opcode in [0xD777_0001, 0xD778_0001] {
        let words = [
            opcode,
            0x0004_0100, // v_permlane[x]16_b32 v1, v0, s0, s1
            0xD834_0000,
            0x0000_0100, // ds_write_b32 v0, v1
            S_ENDPGM,
        ];
        for wave64 in [false, true] {
            let result = lower_wave(&words, layout.clone(), vec![], wave64).unwrap();
            assert!(result.wave_ops);
            assert_eq!(result.barriers, 0);
        }
    }
}

#[test]
fn compute_spills_beside_a_descriptor_word_read_back() {
    // Dword 0 of a buffer descriptor (no host value) and a root constant
    // spilled to lanes 0 and 1 of v5 in a compute program; reading lane
    // 1 back is the constant (the descriptor's lane is never read).
    let words = [
        0xBE84_0300, // s_mov_b32 s4, s0
        0xBE85_0380, // s_mov_b32 s5, 0
        0xF408_0202,
        0xFA00_0000, // s_load_dwordx4 s[8:11], s[4:5], null
        0xD761_0005,
        0x0001_0008, // v_writelane_b32 v5, s8, 0
        0xD761_0005,
        0x0001_0201, // v_writelane_b32 v5, s1, 1
        0xD760_000C,
        0x0001_0305, // v_readlane_b32 s12, v5, 1
        0xE030_2000,
        0x8002_0100, // buffer_load_dword v1, v0, s[8:11], 0 idxen
        0x7E06_020C, // v_mov_b32 v3, s12
        0xD834_0000,
        0x0000_0300, // ds_write_b32 v0, v3
        0xD834_0000,
        0x0000_0100, // ds_write_b32 v0, v1
        S_ENDPGM,
    ];
    let layout = vec![root(RootType::Table, 0, 0), root(RootType::Constants, 1, 1)];
    let used = vec![UsedResource {
        kind: ResourceType::Srv,
        root: RootType::Table,
        table_offset_dwords: 0,
        lower_bound: 0,
        stride: 4,
        dwords: 4,
        user_data: 0,
        slot: 0,
    }];
    lower_with(&words, layout, used).unwrap();
}

#[test]
fn waterfall_loops_over_per_lane_descriptors_sample_with_each_lanes_own() {
    use crate::resource::tests::{
        WATERFALL_FETCHED_WAVE32, WATERFALL_FETCHED_WAVE64, WATERFALL_PICKED_WAVE32,
    };
    for (words, wave64) in [
        (WATERFALL_FETCHED_WAVE32, false),
        (WATERFALL_FETCHED_WAVE64, true),
        (WATERFALL_PICKED_WAVE32, false),
    ] {
        let (layout, used) = texture_tables();
        let l = lower_wave(words, layout, used, wave64).unwrap();
        // The loop's reads of the first lane became each lane's own
        // dwords: no wave operation is left.
        assert_eq!(l.nonuniform_handles, 1);
        assert!(!l.wave_ops);
    }
}

/// [`texture_tables`] with a raw buffer UAV at dword 16 of the first.
pub(crate) fn texture_and_uav_tables() -> (Vec<RootParam>, Vec<UsedResource>) {
    let (layout, mut used) = texture_tables();
    used.push(UsedResource {
        kind: ResourceType::Uav,
        root: RootType::Table,
        table_offset_dwords: 16,
        lower_bound: 0,
        stride: 0,
        dwords: 4,
        user_data: 0,
        slot: 0,
    });
    (layout, used)
}

#[test]
fn waterfall_loops_over_descriptors_joined_per_lane_sample_with_each_lanes_own() {
    use crate::resource::tests::{WATERFALL_JOINED_MIXED_WAVE32, WATERFALL_JOINED_WAVE32};
    for words in [WATERFALL_JOINED_WAVE32, WATERFALL_JOINED_MIXED_WAVE32] {
        let (layout, used) = texture_and_uav_tables();
        let l = lower_wave(words, layout, used, false).unwrap();
        // One image handle, from the join's per-lane heap index.
        assert_eq!(l.nonuniform_handles, 1);
        assert!(!l.wave_ops);
    }
}

#[test]
fn an_address_half_set_where_only_lanes_that_never_run_skip_is_known() {
    // As `indirect_light_probes:CS_Alpha_Schedule` after its texture: a
    // table address whose high half (s25 = 0) is set inside the region
    // that only a skip no lane takes bypasses; the skip edge would bring
    // the stale input s25.
    let words = [
        0xBE9A_037E, // s_mov_b32 s26, exec_lo
        0xBEFE_097E, // s_wqm_b32 exec_lo, exec_lo
        0xBE98_0300, // s_mov_b32 s24, s0
        0xBE86_3C1A, // s_and_saveexec_b32 s6, s26
        0xBF88_0001, // s_cbranch_execz -> join
        0xBE99_0380, // s_mov_b32 s25, 0
        0xBEFE_0306, // join: s_mov_b32 exec_lo, s6
        0xF408_0A0C,
        0xFA00_0040, // s_load_dwordx4 s[40:43], s[24:25], 0x40
        0x3420_0082, // v_lshlrev_b32 v16, 2, v0
        0x7E06_0280, // v_mov_b32 v3, 0
        0xBF8C_C07F, // s_waitcnt lgkmcnt(0)
        0xE070_1000,
        0x800A_0310, // buffer_store_dword v3, v16, s[40:43], 0 offen
        S_ENDPGM,
    ];
    let (_, mut used) = texture_and_uav_tables();
    used.retain(|u| u.user_data == 0);
    let layout = vec![root(RootType::Table, 0, 0)];
    lower_with(&words, layout, used).expect("the high half is 0 wherever a lane runs");
}

#[test]
fn a_descriptor_skipped_only_by_lanes_that_never_run_needs_no_source() {
    // As `indirect_light_probes:CS_Alpha_Schedule`: the texture is
    // loaded under the lanes that ran before `s_wqm`, skipped when there
    // are none, and sampled with the helper lanes after the join. With
    // every lane running from the start nothing takes the skip, so the
    // registers it would bring (no descriptor) are never read.
    let words = [
        0xBE9A_037E, // s_mov_b32 s26, exec_lo
        0xBEFE_097E, // s_wqm_b32 exec_lo, exec_lo
        0xBE84_0300, // s_mov_b32 s4, s0
        0xBE85_0380, // s_mov_b32 s5, 0
        0xF408_0A02,
        0xFA00_0040, // s_load_dwordx4 s[40:43], s[4:5], 0x40
        0xBE86_3C1A, // s_and_saveexec_b32 s6, s26
        0xBF88_0002, // s_cbranch_execz -> join
        0xF40C_0302,
        0xFA00_0000, // s_load_dwordx8 s[12:19], s[4:5], 0x0
        0xBEFE_0306, // join: s_mov_b32 exec_lo, s6
        0xBE88_03FF,
        0x0000_0092, // s_mov_b32 s8, 0x92 (an immediate sampler)
        0xBE89_0380, // s_mov_b32 s9, 0
        0xBE8A_0380, // s_mov_b32 s10, 0
        0xBE8B_0380, // s_mov_b32 s11, 0
        0x7E04_02F0, // v_mov_b32 v2, 0.5
        0x7E06_02F0, // v_mov_b32 v3, 0.5
        0x3420_0082, // v_lshlrev_b32 v16, 2, v0
        0xBF8C_C07F, // s_waitcnt lgkmcnt(0)
        0xF09C_0F08,
        0x0043_0C02, // image_sample_lz v[12:15], v[2:3], s[12:19], s[8:11] dmask:0xf dim:SQ_RSRC_IMG_2D
        0xBF8C_3F70, // s_waitcnt vmcnt(0)
        0xE070_1000,
        0x800A_0C10, // buffer_store_dword v12, v16, s[40:43], 0 offen
        S_ENDPGM,
    ];
    let (_, mut used) = texture_and_uav_tables();
    used.retain(|u| u.user_data == 0);
    let layout = vec![root(RootType::Table, 0, 0)];
    let l = lower_with(&words, layout, used).unwrap();
    assert_eq!(l.nonuniform_handles, 0);
    #[cfg(windows)]
    if let Ok(mut v) = super::super::validator::Validator::load(
        super::super::validator::Validator::default_path().as_deref(),
    ) {
        v.validate(&mut l.dxil.clone())
            .expect("DXIL validation failed");
    }
}

#[test]
fn descriptors_joined_by_nested_control_flow_lower_to_one_uniform_handle() {
    // The view table at s0 (2D textures at entries 0 and 1, a raw UAV at
    // dword 16) and a root constant at s1: entry 0 if s1 is 0, else
    // entry 1 if it is 1, else entry 0 again, the inner if/else joined
    // before the outer one. The join's edges are a load and a join.
    let words = [
        0xBE84_0300, // s_mov_b32 s4, s0
        0xBE85_0380, // s_mov_b32 s5, 0
        0xF408_0A02,
        0xFA00_0040, // s_load_dwordx4 s[40:43], s[4:5], 0x40
        0xBE88_03FF,
        0x0000_0092, // s_mov_b32 s8, 0x92 (an immediate sampler)
        0xBE89_0380, // s_mov_b32 s9, 0
        0xBE8A_0380, // s_mov_b32 s10, 0
        0xBE8B_0380, // s_mov_b32 s11, 0
        0xBF06_8001, // s_cmp_eq_u32 s1, 0
        0xBF84_0003, // s_cbranch_scc0 -> 0x38
        0xF40C_0302,
        0xFA00_0000, // s_load_dwordx8 s[12:19], s[4:5], 0x0
        0xBF82_0008, // s_branch -> 0x58
        0xBF06_8101, // 0x38: s_cmp_eq_u32 s1, 1
        0xBF84_0003, // s_cbranch_scc0 -> 0x4c
        0xF40C_0302,
        0xFA00_0020, // s_load_dwordx8 s[12:19], s[4:5], 0x20
        0xBF82_0002, // s_branch -> 0x54
        0xF40C_0302,
        0xFA00_0000, // 0x4c: s_load_dwordx8 s[12:19], s[4:5], 0x0
        0xBE86_0301, // 0x54: s_mov_b32 s6, s1
        0x7E04_02F0, // 0x58: v_mov_b32 v2, 0.5
        0x7E06_02F0, // v_mov_b32 v3, 0.5
        0x3420_0082, // v_lshlrev_b32 v16, 2, v0
        0xBF8C_C07F, // s_waitcnt lgkmcnt(0)
        0xF09C_0F08,
        0x0043_0C02, // image_sample_lz v[12:15], v[2:3], s[12:19], s[8:11] dmask:0xf dim:SQ_RSRC_IMG_2D
        0xE070_1000,
        0x800A_0C10, // buffer_store_dword v12, v16, s[40:43], 0 offen
        S_ENDPGM,
    ];
    let (_, mut used) = texture_and_uav_tables();
    used.retain(|u| u.user_data == 0);
    let layout = vec![root(RootType::Table, 0, 0), root(RootType::Constants, 1, 1)];
    let l = lower_with(&words, layout, used).unwrap();
    // One image handle from the joins' heap index, the same in all lanes.
    assert_eq!(l.nonuniform_handles, 0);
    #[cfg(windows)]
    if let Ok(mut v) = super::super::validator::Validator::load(
        super::super::validator::Validator::default_path().as_deref(),
    ) {
        v.validate(&mut l.dxil.clone())
            .expect("DXIL validation failed");
    }
}

#[test]
fn descriptors_reloaded_from_a_root_table_share_one_handle() {
    // The view table at s0 (2D textures at entries 0 and 1, a raw UAV at
    // dword 16) and a root constant at s1. Entry 0 is loaded and sampled
    // in a branch, then loaded again and sampled after it (the console
    // compiler reloads a descriptor before each use rather than keep it).
    let words = [
        0xBE84_0300, // s_mov_b32 s4, s0
        0xBE85_0380, // s_mov_b32 s5, 0
        0xF408_0A02,
        0xFA00_0040, // s_load_dwordx4 s[40:43], s[4:5], 0x40
        0xBE88_03FF,
        0x0000_0092, // s_mov_b32 s8, 0x92 (an immediate sampler)
        0xBE89_0380, // s_mov_b32 s9, 0
        0xBE8A_0380, // s_mov_b32 s10, 0
        0xBE8B_0380, // s_mov_b32 s11, 0
        0x7E04_02F0, // v_mov_b32 v2, 0.5
        0x7E06_02F0, // v_mov_b32 v3, 0.5
        0x3420_0082, // v_lshlrev_b32 v16, 2, v0
        0xBF06_8001, // s_cmp_eq_u32 s1, 0
        0xBF84_0005, // s_cbranch_scc0 -> the reload
        0xF40C_0302,
        0xFA00_0000, // s_load_dwordx8 s[12:19], s[4:5], 0x0
        0xBF8C_C07F, // s_waitcnt lgkmcnt(0)
        0xF09C_0F08,
        0x0043_0C02, // image_sample_lz v[12:15], v[2:3], s[12:19], s[8:11] dmask:0xf dim:SQ_RSRC_IMG_2D
        0xF40C_0302,
        0xFA00_0000, // s_load_dwordx8 s[12:19], s[4:5], 0x0 (again)
        0xBF8C_C07F, // s_waitcnt lgkmcnt(0)
        0xF09C_0F08,
        0x0043_0C02, // image_sample_lz v[12:15], v[2:3], s[12:19], s[8:11]
        0xE070_1000,
        0x800A_0C10, // buffer_store_dword v12, v16, s[40:43], 0 offen
        S_ENDPGM,
    ];
    let (_, mut used) = texture_and_uav_tables();
    used.retain(|u| u.user_data == 0);
    let layout = vec![root(RootType::Table, 0, 0), root(RootType::Constants, 1, 1)];
    let l = lower_with(&words, layout, used).unwrap();
    // One image handle made in the entry block for both loads, and the
    // UAV's.
    assert_eq!(l.heap_handles, 2);
    #[cfg(windows)]
    if let Ok(mut v) = super::super::validator::Validator::load(
        super::super::validator::Validator::default_path().as_deref(),
    ) {
        v.validate(&mut l.dxil.clone())
            .expect("DXIL validation failed");
    }
}

#[test]
fn scalar_bit_tests_lower() {
    // Each tests one bit of a constant into SCC and branches on it (only
    // code with an effect needs its operands).
    let tail = [
        0xBF85_0001, // s_cbranch_scc1 -> s_endpgm
        0xBE84_0380, // s_mov_b32 s4, 0
        S_ENDPGM,
    ];
    let bit_tests: [&[u32]; 4] = [
        &[0xBE81_0387, 0xBF0C_8301], // s_mov_b32 s1, 7; s_bitcmp0_b32 s1, 3
        &[0xBE81_0387, 0xBF0D_8301], // s_mov_b32 s1, 7; s_bitcmp1_b32 s1, 3
        &[
            0xBE80_0387, // s_mov_b32 s0, 7
            0xBE81_0380, // s_mov_b32 s1, 0
            0xBF0E_8300, // s_bitcmp0_b64 s[0:1], 3
        ],
        &[
            0xBE80_0387, // s_mov_b32 s0, 7
            0xBE81_0380, // s_mov_b32 s1, 0
            0xBF0F_8300, // s_bitcmp1_b64 s[0:1], 3
        ],
    ];
    for head in bit_tests {
        let words: Vec<u32> = head.iter().chain(&tail).copied().collect();
        lowers(&words, 0).unwrap();
    }
}

#[test]
fn a_missing_value_stays_missing_through_two_live_joins() {
    let mut words = [
        0xF490_0180,
        0,           // s_memtime s[6:7]
        0xBF06_8000, // s_cmp_eq_u32 s0, 0 (s0 = group id X)
        0xBF85_0001, // skip the first overwrite
        0xBE86_0380, // s_mov_b32 s6, 0
        0xBF06_8101, // s_cmp_eq_u32 s1, 1 (s1 = group id Y)
        0xBF85_0001, // skip the second overwrite
        0xBE86_0381, // s_mov_b32 s6, 1
        0xBF07_8006, // s_cmp_lg_u32 s6, 0
        0xBF85_0001, // use the result to branch
        0xBE83_0380,
        S_ENDPGM,
    ];
    let group_ids_xy = 3 << 7;
    // Compare to a direct read so future timestamp support works in both cases.
    let refusal = |code: &[u32]| match lowers(code, group_ids_xy) {
        Ok(()) => None,
        Err(crate::ShaderError::Unsupported { what, .. }) => Some(what),
        Err(error) => panic!("{error}"),
    };
    let direct = [&words[..2], &words[8..]].concat();
    assert_eq!(refusal(&words), refusal(&direct));
    words[8] = 0xBF07_8000; // read the group id instead; the timestamp is dead
    lowers(&words, group_ids_xy).expect("dead timestamp");
}

#[test]
fn a_value_no_lane_from_its_edge_reads_needs_no_form() {
    // s6 is a timestamp (no DXIL form) where the load is skipped and 5
    // where it runs; a flag in v1, set only with the load, guards the
    // read after the join (the console compiler's form, where the
    // skipped path leaves a stale lane mask in the register).
    let mut words = [
        0xF490_0180,
        0,           // s_memtime s[6:7]
        0x7E02_0280, // v_mov_b32 v1, 0
        0xBF06_8000, // s_cmp_eq_u32 s0, 0 (s0 = group id X)
        0xBF85_0002, // s_cbranch_scc1 -> join
        0xBE86_0385, // s_mov_b32 s6, 5
        0x7E02_02C1, // v_mov_b32 v1, -1
        0x7D84_02C1, // join: v_cmp_eq_u32 vcc_lo, -1, v1
        0xBF86_0003, // s_cbranch_vccz -> end
        0xBF07_8006, // s_cmp_lg_u32 s6, 0
        0xBF85_0001, // s_cbranch_scc1 -> end
        0xBE84_0380, // s_mov_b32 s4, 0
        S_ENDPGM,
    ];
    let group_ids_xy = 3 << 7;
    lowers(&words, group_ids_xy).expect("the timestamp is never read");
    // With the flag set on both paths the timestamp is read: as refused
    // (or accepted) as a direct read.
    words[2] = 0x7E02_02C1; // v_mov_b32 v1, -1
    let refusal = |code: &[u32]| match lowers(code, group_ids_xy) {
        Ok(()) => None,
        Err(crate::ShaderError::Unsupported { what, .. }) => Some(what),
        Err(error) => panic!("{error}"),
    };
    let direct = [&words[..2], &words[9..]].concat();
    assert_eq!(refusal(&words), refusal(&direct));
}

#[test]
fn a_mask_joined_with_data_needs_a_uniform_join() {
    // s6: lanes below 24, then shifted (data) on each back edge; read
    // after the loop. Lanes leaving the loop at different iterations
    // (per lane) see different numbers than the console's register:
    // refused. The uniform loop is `synthetic_tests::
    // a_mask_joined_with_data_at_a_uniform_loop_is_its_number`.
    let mut words = [
        0x7D88_0098, // v_cmp_gt_u32 vcc_lo, 24, v0
        0x8706_7E6A, // s_and_b32 s6, vcc_lo, exec_lo
        0xBE87_0380, // s_mov_b32 s7, 0
        0x9006_8406, // header: s_lshr_b32 s6, s6, 4
        0x8007_8107, // s_add_u32 s7, s7, 1
        0x7D88_0007, // v_cmp_gt_u32 vcc_lo, s7, v0
        0x8A7E_6A7E, // s_andn2_b32 exec_lo, exec_lo, vcc_lo
        0xBF89_FFFB, // s_cbranch_execnz header
        0xBF07_8006, // s_cmp_lg_u32 s6, 0
        0xBF85_0001, // s_cbranch_scc1 +1
        0xBE84_0380, // s_mov_b32 s4, 0
        S_ENDPGM,
    ];
    let Err(crate::ShaderError::Unsupported { what, .. }) = lowers(&words, 1 << 7) else {
        panic!("a per-lane loop's mask join translated");
    };
    assert!(what.contains("lane mask"), "{what}");
    // Leaving on a uniform condition instead: the mask's number.
    words[5] = 0xBF0A_8207; // s_cmp_lt_u32 s7, 2
    words[6] = 0xBF80_0000; // s_nop 0
    words[7] = 0xBF85_FFFB; // s_cbranch_scc1 header
    lowers(&words, 1 << 7).expect("a uniform loop's mask join");
}

#[test]
fn a_scalar_result_needs_no_exec() {
    // s6 is 0 where every lane skips and the saved EXEC where some run:
    // no lane mask at the per-lane join. EXEC taken from it guards only
    // a `v_readlane` (written whatever EXEC holds) before it is restored
    // (`CS_IrradianceCache_Shade`).
    let words = [
        0xBE86_0380, // s_mov_b32 s6, 0
        0x7D88_0090, // v_cmp_gt_u32 vcc_lo, 16, v0
        0xBE84_3C6A, // s_and_saveexec_b32 s4, vcc_lo
        0xBF88_0001, // s_cbranch_execz +1
        0xBE86_0304, // s_mov_b32 s6, s4
        0xBEFE_0306, // s_mov_b32 exec_lo, s6
        0xD760_0002, // v_readlane_b32 s2, v0, 3
        0x0001_0700,
        0xBEFE_0304, // s_mov_b32 exec_lo, s4
        S_ENDPGM,
    ];
    lowers(&words, 1 << 7).expect("a lane read under an EXEC with no mask");
}

#[test]
fn guarded_or_modified_v_mov_b32_is_not_folded_as_constant() {
    let group_ids_xy = 3 << 7;
    let refusal = |code: &[u32]| match lowers(code, group_ids_xy) {
        Ok(()) => None,
        Err(crate::ShaderError::Unsupported { what, .. }) => Some(what),
        Err(error) => panic!("{error}"),
    };
    let direct = [
        0xF490_0180,
        0,           // s_memtime s[6:7]
        0xBF07_8006, // s_cmp_lg_u32 s6, 0
        0xBF85_0001, // s_cbranch_scc1 -> end
        0xBE84_0380, // s_mov_b32 s4, 0
        S_ENDPGM,
    ];
    assert!(refusal(&direct).is_some());

    // Guarded `v_mov_b32 v1, 0`: inactive lanes (v0 != 0) preserve v1 = -1,
    // so they pass the compare after the join and read s6 (`s_memtime`).
    let mut guarded = [
        0xF490_0180,
        0,           // s_memtime s[6:7]
        0x7E02_02C1, // v_mov_b32 v1, -1
        0x7DA4_0080, // v_cmpx_eq_u32 exec, 0, v0
        0x7E02_0280, // v_mov_b32 v1, 0 (guarded)
        0xBEFE_03C1, // s_mov_b32 exec_lo, -1
        0xBF06_8000, // s_cmp_eq_u32 s0, 0
        0xBF85_0002, // s_cbranch_scc1 -> join
        0xBE86_0385, // s_mov_b32 s6, 5
        0x7E02_02C1, // v_mov_b32 v1, -1
        0x7D84_02C1, // join: v_cmp_eq_u32 vcc_lo, -1, v1
        0xBF86_0003, // s_cbranch_vccz -> end
        0xBF07_8006, // s_cmp_lg_u32 s6, 0
        0xBF85_0001, // s_cbranch_scc1 -> end
        0xBE84_0380, // s_mov_b32 s4, 0
        S_ENDPGM,
    ];
    assert_eq!(refusal(&guarded), refusal(&direct));
    // Without the EXEC guard, `v_mov_b32 v1, 0` writes every lane and s6 is dead.
    guarded[3] = 0xBF80_0000; // s_nop 0
    guarded[5] = 0xBF80_0000; // s_nop 0
    lowers(&guarded, group_ids_xy).expect("unguarded v_mov_b32 still folds");

    // VOP3 `v_mov_b32_e64 v1, -(1.0)`: neg modifier flips 1.0f to -1.0f,
    // matching the compare after the join so s6 is live.
    let mut negated = [
        0xF490_0180,
        0,           // s_memtime s[6:7]
        0xD581_0001,
        0x2000_00F2, // v_mov_b32_e64 v1, -(1.0)
        0xBF06_8000, // s_cmp_eq_u32 s0, 0
        0xBF85_0002, // s_cbranch_scc1 -> join
        0xBE86_0385, // s_mov_b32 s6, 5
        0x7E02_02F3, // v_mov_b32 v1, -1.0
        0x7D84_02F3, // join: v_cmp_eq_u32 vcc_lo, -1.0, v1
        0xBF86_0003, // s_cbranch_vccz -> end
        0xBF07_8006, // s_cmp_lg_u32 s6, 0
        0xBF85_0001, // s_cbranch_scc1 -> end
        0xBE84_0380, // s_mov_b32 s4, 0
        S_ENDPGM,
    ];
    assert_eq!(refusal(&negated), refusal(&direct));
    // Clearing the neg bit yields unmodified 1.0f (!= -1.0f), so s6 is dead.
    negated[3] = 0x0000_00F2; // v_mov_b32_e64 v1, 1.0
    lowers(&negated, group_ids_xy).expect("unmodified v_mov_b32_e64 still folds");
}

#[test]
fn a_missing_value_on_a_path_no_lane_takes_is_dead() {
    let refusal = |code: &[u32]| match lowers(code, 1 << 7) {
        Ok(()) => None,
        Err(crate::ShaderError::Unsupported { what, .. }) => Some(what),
        Err(error) => panic!("{error}"),
    };
    // The timestamp stays in s6 only where s0 == 0, and those lanes
    // overwrite it before the second join: it never reaches the read.
    let mut joins = [
        0xF490_0180,
        0,           // s_memtime s[6:7]
        0xBF06_8000, // s_cmp_eq_u32 s0, 0 (s0 = group id X)
        0xBF85_0001, // s_cbranch_scc1: s0 == 0 keeps the timestamp
        0xBE86_0380, // s_mov_b32 s6, 0
        0xBF84_0001, // s_cbranch_scc0: s0 != 0 keeps that 0
        0xBE86_0381, // s_mov_b32 s6, 1
        0xBF07_8006, // s_cmp_lg_u32 s6, 0
        0xBF85_0001,
        0xBE83_0380,
        S_ENDPGM,
    ];
    assert_eq!(refusal(&joins), None);
    // Tested on the other side, the timestamp reaches the read.
    joins[5] = 0xBF85_0001; // s_cbranch_scc1
    assert!(refusal(&joins).is_some());
    // The same through a VGPR whose old value a guarded write keeps
    // (lanes outside it get 0 for a value with no DXIL form, as for any
    // such write): keeping does not make the join live.
    let kept = [
        0xF490_0180,
        0,           // s_memtime s[6:7]
        0x7E02_0206, // v_mov_b32 v1, s6
        0xBF06_8000, // s_cmp_eq_u32 s0, 0
        0xBF85_0001, // s_cbranch_scc1: s0 == 0 keeps the timestamp
        0x7E02_0280, // v_mov_b32 v1, 0
        0xBF84_0004, // s_cbranch_scc0: s0 != 0 keeps that 0
        0x7DA4_0080, // v_cmpx_eq_u32 exec, 0, v0
        0x7E02_0282, // v_mov_b32 v1, 2 (the other lanes keep v1)
        0xBEFE_03C1, // s_mov_b32 exec_lo, -1
        0x7E02_0283, // v_mov_b32 v1, 3
        0x7D84_0280, // v_cmp_eq_u32 vcc, 0, v1
        0xBF87_0001, // s_cbranch_vccnz
        0xBE83_0380,
        S_ENDPGM,
    ];
    assert_eq!(refusal(&kept), None);
}

#[test]
fn a_guarded_scalar_write_kills_a_stale_sgpr() {
    // s6 holds a missing value (timestamp) where s0 == 0 and 5 where
    // s0 != 0; the flag in v1 skips the first read on the s0 == 0 path.
    // After that, inside a guarded region (EXEC narrowed without a
    // branch), `v_readfirstlane_b32` overwrites s6 unconditionally before
    // s6 is read again.
    let mut words = [
        0xF490_0180,
        0,           // s_memtime s[6:7]
        0x7E02_0280, // v_mov_b32 v1, 0
        0xBF06_8000, // s_cmp_eq_u32 s0, 0 (s0 = group id X)
        0xBF85_0002, // s_cbranch_scc1 -> join
        0xBE86_0385, // s_mov_b32 s6, 5
        0x7E02_02C1, // v_mov_b32 v1, -1
        0x7D84_02C1, // join: v_cmp_eq_u32 vcc_lo, -1, v1
        0xBF86_0003, // s_cbranch_vccz -> after
        0xBF07_8006, // s_cmp_lg_u32 s6, 0
        0xBF85_0001, // s_cbranch_scc1 -> after
        0xBE84_0380, // s_mov_b32 s4, 0
        0x7DA4_0080, // after: v_cmpx_eq_u32 exec_lo, 0, v0 (narrows EXEC)
        0x7E0C_0500, // v_readfirstlane_b32 s6, v0 (guarded, writes SGPR)
        0x7DA4_0006, // v_cmpx_eq_u32 exec_lo, s6, v0 (reads new s6)
        0xBEFE_03C1, // s_mov_b32 exec_lo, -1
        S_ENDPGM,
    ];
    let group_ids_xy = 3 << 7;
    lowers(&words, group_ids_xy).expect("guarded v_readfirstlane overwrites stale s6");
    // Without the overwrite, the guarded compare reads the stale s6.
    let mut no_kill = words;
    no_kill[13] = 0xBF80_0000; // s_nop 0
    assert!(lowers(&no_kill, group_ids_xy).is_err());
    // With the flag set on both paths, the earlier read sees the stale s6.
    words[2] = 0x7E02_02C1; // v_mov_b32 v1, -1
    assert!(lowers(&words, group_ids_xy).is_err());
}

#[test]
fn buffer_formats_by_number() {
    // The numbers the disassembler names 8_8_8_8_UNORM, 16_16_SNORM,
    // 2_10_10_10_UNORM, 16_FLOAT, 32_32_32_32_FLOAT.
    assert_eq!(buffer_format(56), Some((&[8, 8, 8, 8][..], Numeric::Unorm)));
    assert_eq!(buffer_format(24), Some((&[16, 16][..], Numeric::Snorm)));
    assert_eq!(
        buffer_format(50),
        Some((&[10, 10, 10, 2][..], Numeric::Unorm))
    );
    assert_eq!(buffer_format(13), Some((&[16][..], Numeric::Float)));
    assert_eq!(
        buffer_format(77),
        Some((&[32, 32, 32, 32][..], Numeric::Float))
    );
    // Invalid, past the table.
    assert_eq!(buffer_format(0), None);
    assert_eq!(buffer_format(78), None);
}

#[test]
fn packed_buffer_formats_by_number() {
    // LLVM's `UfmtGFX10`: 10_11_11 is 30-36, 11_11_10 37-43,
    // 10_10_10_2 44-49. Channels from the lowest bits: R11G11B10_FLOAT
    // is 10_11_11 (Mesa), 10_10_10_2 has the 2-bit channel first.
    assert_eq!(buffer_format(30), Some((&[11, 11, 10][..], Numeric::Unorm)));
    assert_eq!(buffer_format(36), Some((&[11, 11, 10][..], Numeric::Float)));
    assert_eq!(buffer_format(37), Some((&[10, 11, 11][..], Numeric::Unorm)));
    assert_eq!(buffer_format(43), Some((&[10, 11, 11][..], Numeric::Float)));
    assert_eq!(
        buffer_format(44),
        Some((&[2, 10, 10, 10][..], Numeric::Unorm))
    );
    assert_eq!(
        buffer_format(45),
        Some((&[2, 10, 10, 10][..], Numeric::Snorm))
    );
    assert_eq!(
        buffer_format(46),
        Some((&[2, 10, 10, 10][..], Numeric::Uscaled))
    );
    assert_eq!(
        buffer_format(47),
        Some((&[2, 10, 10, 10][..], Numeric::Sscaled))
    );
    assert_eq!(
        buffer_format(48),
        Some((&[2, 10, 10, 10][..], Numeric::Uint))
    );
    assert_eq!(
        buffer_format(49),
        Some((&[2, 10, 10, 10][..], Numeric::Sint))
    );
}

/// A half's value (sign, 5-bit exponent biased 15, 10-bit mantissa), as
/// `legacyF16ToF32` widens it.
fn half(h: u32) -> f32 {
    let (s, e, m) = ((h >> 15) & 1, (h >> 10) & 31, h & 1023);
    let v = match e {
        0 => m as f32 * 2f32.powi(-24),
        31 if m == 0 => f32::INFINITY,
        31 => f32::NAN,
        _ => (1024 + m) as f32 * 2f32.powi(e as i32 - 25),
    };
    if s == 1 { -v } else { v }
}

#[test]
fn small_floats_widen_as_halves() {
    // Unsigned 11- and 10-bit floats: a 5-bit exponent biased 15 over a
    // 6- or 5-bit mantissa, no sign; values from the D3D/GL spec.
    let f11 = |x: u32| half(x << half_shift(11));
    let f10 = |x: u32| half(x << half_shift(10));
    assert_eq!(f11(0), 0.0);
    assert_eq!(f11(0x001), 2f32.powi(-20)); // smallest denormal
    assert_eq!(f11(0x03F), 63.0 * 2f32.powi(-20)); // largest denormal
    assert_eq!(f11(0x040), 2f32.powi(-14)); // smallest normal
    assert_eq!(f11(0x3C0), 1.0);
    assert_eq!(f11(0x3E0), 1.5);
    assert_eq!(f11(0x7BF), 65024.0); // largest
    assert_eq!(f11(0x7C0), f32::INFINITY);
    assert!(f11(0x7C1).is_nan());
    assert_eq!(f10(0x001), 2f32.powi(-19));
    assert_eq!(f10(0x020), 2f32.powi(-14));
    assert_eq!(f10(0x1E0), 1.0);
    assert_eq!(f10(0x3DF), 64512.0);
    assert_eq!(f10(0x3E0), f32::INFINITY);
    assert!(f10(0x3FF).is_nan());
    // A 16-bit float is a half already, sign included.
    assert_eq!(half(0xBC00 << half_shift(16)), -1.0);
}

#[test]
fn packed_halves_fill_parameters_first_fit() {
    // 16-bit rows as a title's signature has them: row 0 holds two
    // elements (x y and z w), row 3 is 32-bit, row 4 has one channel at w.
    let el = |row: u32, col: u8, cols: u8, half: bool| super::SigElement {
        name: "TEXCOORD".into(),
        index: row,
        kind: 0,
        comp: 9,
        interp: 0,
        row: Some(row),
        col,
        cols,
        usage: 0,
        half,
    };
    let sig = [
        el(0, 0, 2, true),
        el(0, 2, 2, true),
        el(1, 0, 4, true),
        el(2, 0, 3, true),
        el(3, 0, 4, false),
        el(4, 0, 3, true),
        el(4, 3, 1, true),
        el(5, 0, 3, true),
    ];
    // Each half as row and channel ("-": padding or past the end).
    let param = |a| {
        (0..8)
            .map(|h| match packed_half(&sig, a, h) {
                Some((r, c)) => format!("{r}{}", ["x", "y", "z", "w"][usize::from(c)]),
                None => "-".into(),
            })
            .collect::<Vec<_>>()
            .join(" ")
    };
    // By first channel, then row: row 0's x y, all of row 1, then (last,
    // as it starts at z) row 0's z w.
    assert_eq!(param(0), "0x 0y 1x 1y 1z 1w 0z 0w");
    // Three channels take two dwords; row 4's w fills the first
    // parameter with room, after row 5.
    assert_eq!(param(2), "2x 2y 2z - 4x 4y 4z -");
    assert_eq!(param(5), "5x 5y 5z - 4w - - -");
    // No parameter starts in rows 1, 3 or 4.
    for a in [1, 3, 4] {
        assert_eq!(param(a), "- - - - - - - -");
    }
}

// ---- one console wave split across host waves ----

const V_LSHLREV_V1_2_V0: u32 = 0x3402_0082; // v_lshlrev_b32 v1, 2, v0
const DS_WRITE_V1_V0: [u32; 2] = [0xD834_0000, 0x0000_0001]; // ds_write_b32 v1, v0
const DS_READ_V2_V1_4: [u32; 2] = [0xD8D8_0004, 0x0200_0001]; // ds_read_b32 v2, v1 offset:4
const SAVE_EXEC: u32 = 0xBE80_047E; // s_mov_b64 s[0:1], exec
const CMPX_V0_EQ_0: u32 = 0x7DA4_0080; // v_cmpx_eq_u32 exec, 0, v0
const RESTORE_EXEC: u32 = 0xBEFE_0400; // s_mov_b64 exec, s[0:1]

/// `s_cbranch_execz` over the next `words` words.
const fn skip(words: u32) -> u32 {
    0xBF88_0000 | words
}

fn wave64(words: &[u32]) -> ir::Program {
    let code: Vec<u8> = words.iter().flat_map(|w| w.to_le_bytes()).collect();
    ir::build(&code, true, 256).unwrap()
}

/// [`super::split_wave_barriers`] of a program.
fn barriers_at(p: &ir::Program) -> Option<Vec<(usize, usize)>> {
    let st = structure::structure(p).unwrap();
    let mut at: Vec<_> = super::split_wave_barriers(p, &st)?.into_iter().collect();
    at.sort_unstable();
    Some(at)
}

/// Lowers a wave64 compute program of `lanes` threads with 512 bytes of
/// LDS: the barriers it got for narrow host waves.
fn split_barriers(words: &[u32], lanes: u32, narrow_waves: bool) -> usize {
    lowered(words, lanes, narrow_waves).split_barriers
}

/// The same program, lowered.
fn lowered(words: &[u32], lanes: u32, narrow_waves: bool) -> super::Lowered {
    lowered_result(words, lanes, narrow_waves).expect("lower")
}

/// The same program, lowered, as a result (a refusal is an `Err`).
fn lowered_result(
    words: &[u32],
    lanes: u32,
    narrow_waves: bool,
) -> Result<super::Lowered, crate::ShaderError> {
    let code: Vec<u8> = words.iter().flat_map(|w| w.to_le_bytes()).collect();
    let native = NativeProgram {
        stage: HwStage::Cs,
        wave64: true,
        vgprs: 256,
        layout: vec![],
        used: vec![],
        io: StageIo::Cs(ComputeIo {
            rsrc2: 1 << 15 | 2 << 1,
            threads: [lanes, 1, 1],
        }),
        code,
    };
    let p = ir::build(&native.code, true, 256).unwrap();
    let st = structure::structure(&p).unwrap();
    let res = resource::track(&p, &native.layout, native.stage, Some(&st)).unwrap();
    let io = io::raise(&p, &native).unwrap();
    lower(&Parts {
        halves_rendezvous: false,
        native: &native,
        p: &p,
        st: &st,
        io: &io,
        res: &res,
        inputs: vec![],
        outputs: vec![],
        prims: vec![],
        mesh_stride: 0,
        atomic_locks: false,
        atomic_lock_rounds: false,
        atomic_umax: false,
        bias_as_gradients: false,
        narrow_waves,
        uav_size_level0: false,
        host_wave32: false,
        view_marks: false,
    })
}

#[test]
fn a_read_after_another_lanes_write_syncs_in_a_split_wave() {
    let mut w = vec![V_LSHLREV_V1_2_V0];
    w.extend(DS_WRITE_V1_V0);
    w.extend(DS_READ_V2_V1_4);
    w.push(S_ENDPGM);
    assert_eq!(barriers_at(&wave64(&w)).unwrap(), [(0, 2)]);
    assert_eq!(split_barriers(&w, 64, true), 1);
    // Option off, or a group one host wave holds: as before.
    assert_eq!(split_barriers(&w, 64, false), 0);
    assert_eq!(split_barriers(&w, 32, true), 0);
    // Synced, it no longer asks for 64-lane waves a 32-lane host lacks
    // (D3DMetal runs nothing then); without the option it still does.
    assert_eq!(lowered(&w, 64, true).wave_size, None);
    assert_eq!(lowered(&w, 64, false).wave_size, Some(64));
    assert_eq!(lowered(&w, 32, true).wave_size, None);
}

#[test]
fn a_divergent_write_syncs_at_the_join_before_the_read() {
    // 34d88edf583f3778's shape: lanes with v0 = 0 write, all read.
    let mut w = vec![V_LSHLREV_V1_2_V0, SAVE_EXEC, CMPX_V0_EQ_0, skip(2)];
    w.extend(DS_WRITE_V1_V0);
    w.push(RESTORE_EXEC);
    w.extend(DS_READ_V2_V1_4);
    w.push(S_ENDPGM);
    let p = wave64(&w);
    let at = barriers_at(&p).unwrap();
    assert_eq!(at.len(), 1);
    let (b, i) = at[0];
    assert!(p.blocks[b].insts[i].name().starts_with("ds_read"));
    assert_eq!(p.blocks[b].preds.len(), 2, "the join");
    assert_eq!(split_barriers(&w, 64, true), 1);
}

#[test]
fn a_write_before_divergent_reads_syncs_where_all_lanes_run() {
    let mut w = vec![V_LSHLREV_V1_2_V0];
    w.extend(DS_WRITE_V1_V0);
    w.extend([SAVE_EXEC, CMPX_V0_EQ_0, skip(2)]);
    w.extend(DS_READ_V2_V1_4);
    w.push(RESTORE_EXEC);
    w.push(S_ENDPGM);
    let p = wave64(&w);
    let at = barriers_at(&p).unwrap();
    assert_eq!(at, [(0, p.blocks[0].insts.len())], "the end of the entry");
    assert_eq!(split_barriers(&w, 64, true), 1);
}

#[test]
fn no_barrier_in_divergent_code() {
    // Write and read both under a branch: without predication no barrier can
    // go in divergent code, while lowering with `narrow_waves` predicates the
    // acyclic diamond and places one uniform barrier between them.
    let mut w = vec![V_LSHLREV_V1_2_V0, SAVE_EXEC, CMPX_V0_EQ_0, skip(4)];
    w.extend(DS_WRITE_V1_V0);
    w.extend(DS_READ_V2_V1_4);
    w.push(RESTORE_EXEC);
    w.push(S_ENDPGM);
    assert!(barriers_at(&wave64(&w)).is_none());
    assert_eq!(split_barriers(&w, 64, true), 1);
}

const S_MOV_S20_0: u32 = 0xBE94_0380; // s_mov_b32 s20, 0
const S_ADD_S20_1: u32 = 0x8014_8114; // s_add_u32 s20, s20, 1
const S_CMP_LT_S20_4: u32 = 0xBF0A_8414; // s_cmp_lt_u32 s20, 4
const S_CMP_LT_S20_S21: u32 = 0xBF0A_1514; // s_cmp_lt_u32 s20, s21
const READFIRSTLANE_S21_V0: u32 = 0x7E2A_0500; // v_readfirstlane_b32 s21, v0

/// `body` in a loop on s20, ending at the compare `cmp` (`s_cbranch_scc1`
/// back to the body's start).
fn looped(before: &[u32], body: &[u32], cmp: u32) -> Vec<u32> {
    let mut w = before.to_vec();
    w.push(S_MOV_S20_0);
    w.extend(body);
    let back = -(body.len() as i32 + 3);
    w.extend([S_ADD_S20_1, cmp, 0xBF85_0000 | (back as u32 & 0xFFFF)]);
    w.push(S_ENDPGM);
    w
}

#[test]
fn the_loop_programs_are_what_their_comments_say() {
    let w = looped(
        &[V_LSHLREV_V1_2_V0, READFIRSTLANE_S21_V0],
        &[DS_WRITE_V1_V0, DS_READ_V2_V1_4].concat(),
        S_CMP_LT_S20_S21,
    );
    let code: Vec<u8> = w.iter().flat_map(|w| w.to_le_bytes()).collect();
    let all = crate::gfx10::decode(&code)
        .unwrap()
        .iter()
        .map(|i| i.text(true))
        .collect::<Vec<_>>()
        .join("\n");
    for want in [
        "v_readfirstlane_b32 s21, v0",
        "s_mov_b32 s20, 0",
        "ds_write_b32 v1, v0",
        "ds_read_b32 v2, v1 offset:4",
        "s_add_u32 s20, s20, 1",
        "s_cmp_lt_u32 s20, s21",
        "s_cbranch_scc1",
    ] {
        assert!(all.contains(want), "{want:?} not in:\n{all}");
    }
    let w = looped(&[], &[], S_CMP_LT_S20_4);
    let code: Vec<u8> = w.iter().flat_map(|w| w.to_le_bytes()).collect();
    let all = crate::gfx10::decode(&code)
        .unwrap()
        .iter()
        .map(|i| i.text(true))
        .collect::<Vec<_>>()
        .join("\n");
    assert!(all.contains("s_cmp_lt_u32 s20, 4"), "{all}");
}

#[test]
fn lds_in_a_loop_on_a_scalar_count_syncs_inside_the_loop() {
    // Each pass writes, then reads what another lane wrote: a barrier
    // before the read, and one before the write, which the last pass's read
    // of another lane must not miss.
    let w = looped(
        &[V_LSHLREV_V1_2_V0],
        &[DS_WRITE_V1_V0, DS_READ_V2_V1_4].concat(),
        S_CMP_LT_S20_4,
    );
    let p = wave64(&w);
    assert!(p.blocks[1].insts[0].name().starts_with("ds_write"));
    assert_eq!(barriers_at(&p).unwrap(), [(1, 0), (1, 1)]);
    assert_eq!(split_barriers(&w, 64, true), 2);
    assert_eq!(lowered(&w, 64, true).wave_size, None);
}

#[test]
fn lds_in_a_loop_on_a_lane_value_gets_no_barrier() {
    // The count comes from a lane (`v_readfirstlane`): the host waves need
    // not agree on it, so no barrier can go in the loop.
    let w = looped(
        &[V_LSHLREV_V1_2_V0, READFIRSTLANE_S21_V0],
        &[DS_WRITE_V1_V0, DS_READ_V2_V1_4].concat(),
        S_CMP_LT_S20_S21,
    );
    assert!(barriers_at(&wave64(&w)).is_none());
    assert_eq!(split_barriers(&w, 64, true), 0);
    // A loop on a lane's own condition (`s_cbranch_vccnz`) neither.
    let mut w = vec![V_LSHLREV_V1_2_V0];
    w.extend(DS_WRITE_V1_V0);
    w.extend(DS_READ_V2_V1_4);
    w.extend([
        0x7D88_0480, // v_cmp_gt_u32 vcc, 0, v2
        0xBF87_FFFA, // s_cbranch_vccnz (back to the write)
        S_ENDPGM,
    ]);
    assert!(barriers_at(&wave64(&w)).is_none());
}

#[test]
fn a_write_after_another_lanes_read_syncs_before_it() {
    let mut w = vec![V_LSHLREV_V1_2_V0];
    w.extend(DS_WRITE_V1_V0);
    w.extend(DS_READ_V2_V1_4);
    w.extend(DS_WRITE_V1_V0);
    w.extend(DS_READ_V2_V1_4);
    w.push(S_ENDPGM);
    assert_eq!(barriers_at(&wave64(&w)).unwrap(), [(0, 2), (0, 3), (0, 4)]);
}

#[test]
fn a_read_before_divergent_writes_syncs_where_all_lanes_run() {
    let mut w = vec![V_LSHLREV_V1_2_V0];
    w.extend(DS_READ_V2_V1_4);
    w.extend([SAVE_EXEC, CMPX_V0_EQ_0, skip(2)]);
    w.extend(DS_WRITE_V1_V0);
    w.push(RESTORE_EXEC);
    w.push(S_ENDPGM);
    let p = wave64(&w);
    assert_eq!(barriers_at(&p).unwrap(), [(0, p.blocks[0].insts.len())]);
    // A write after a read, both under the branch: no barrier can go
    // there, and the program runs as before.
    let mut w = vec![V_LSHLREV_V1_2_V0, SAVE_EXEC, CMPX_V0_EQ_0, skip(4)];
    w.extend(DS_READ_V2_V1_4);
    w.extend(DS_WRITE_V1_V0);
    w.push(RESTORE_EXEC);
    w.push(S_ENDPGM);
    assert_eq!(barriers_at(&wave64(&w)).unwrap(), []);
}

/// Lowers a wave64 compute program of `lanes` threads in halves
/// (`host_wave32`, a host of 32-lane waves): the console wave runs as two
/// host waves, which pass values between them behind group barriers.
fn lowered_halves(words: &[u32], lanes: u32) -> Result<Lowered, crate::ShaderError> {
    lowered_halves_lds(words, lanes, 0)
}

/// [`lowered_halves`] of a program declaring `lds_bytes` of LDS (a
/// multiple of 512).
fn lowered_halves_lds(
    words: &[u32],
    lanes: u32,
    lds_bytes: u32,
) -> Result<Lowered, crate::ShaderError> {
    let code: Vec<u8> = words.iter().flat_map(|w| w.to_le_bytes()).collect();
    let native = NativeProgram {
        stage: HwStage::Cs,
        wave64: true,
        vgprs: 256,
        layout: vec![],
        used: vec![],
        io: StageIo::Cs(ComputeIo {
            // Two user-data SGPRs, LDS in 512-byte granules.
            rsrc2: 2 << 1 | (lds_bytes / 512) << 15,
            threads: [lanes, 1, 1],
        }),
        code,
    };
    let p = ir::build(&native.code, true, 256).unwrap();
    let st = structure::structure(&p).unwrap();
    let res = resource::track(&p, &native.layout, native.stage, Some(&st)).unwrap();
    let io = io::raise(&p, &native).unwrap();
    lower(&Parts {
        halves_rendezvous: false,
        native: &native,
        p: &p,
        st: &st,
        io: &io,
        res: &res,
        inputs: vec![],
        outputs: vec![],
        prims: vec![],
        mesh_stride: 0,
        atomic_locks: false,
        atomic_lock_rounds: false,
        atomic_umax: false,
        bias_as_gradients: false,
        narrow_waves: false,
        uav_size_level0: false,
        host_wave32: true,
        view_marks: false,
    })
}

/// A ballot read as data in a 1,024-thread wave64 program whose own LDS is
/// the host's whole 32 KiB: the halves pass it through dwords borrowed from
/// that LDS, saved and put back behind three more group barriers. Before,
/// the program was refused.
#[test]
fn a_halves_exchange_borrows_a_full_lds() {
    let w = [
        0x3402_0082, // v_lshlrev_b32 v1, 2, v0
        0xD834_0000,
        0x0000_0001, // ds_write_b32 v1, v0
        0xBF8C_C07F, // s_waitcnt lgkmcnt(0)
        0xBF8A_0000, // s_barrier
        0x7D88_00A8, // v_cmp_gt_u32 vcc, 40, v0
        0xBE8C_106A, // s_bcnt1_i32_b64 s12, vcc
        0x7E06_020C, // v_mov_b32 v3, s12
        0xD834_0000,
        0x0000_0301, // ds_write_b32 v1, v3
        S_ENDPGM,
    ];
    // (The IR leaves out `s_waitcnt`.)
    let p = wave64(&w);
    let names: Vec<_> = p.blocks[0].insts.iter().map(|i| i.name()).collect();
    assert_eq!(
        names,
        [
            "v_lshlrev_b32",
            "ds_write_b32",
            "s_barrier",
            "v_cmp_gt_u32",
            "s_bcnt1_i32_b64",
            "v_mov_b32",
            "ds_write_b32",
            "s_endpgm",
        ]
    );
    lowered_halves_lds(&w, 1024, 16384).expect("LDS with room");
    let full = lowered_halves_lds(&w, 1024, 32768).expect("a full LDS");
    assert_eq!(full.wave_size, Some(32));
    // The program's own barrier, the exchange's two, and three for the
    // borrowed dwords (no thread ticket: that needs a dword of LDS too).
    assert_eq!(full.barriers, 1 + 2 + 3);
}

/// A uniform outer loop (on the scalar `s20`) holding a divergent inner
/// loop — entered on `vcc`, the lanes with thread id below 40 — whose latch
/// is an unconditional `s_branch` back to its header: the back edge that
/// lowering caps with an early return at the loop iteration limit. After
/// the inner loop, `v_readlane_b32` of console lane 40, which in halves
/// passes between the 32-lane halves behind group barriers (its result is
/// only read after the outer loop, so no branch in the loop depends on it).
fn capped_back_edge_in_a_divergent_loop() -> Vec<u32> {
    vec![
        0xBE94_0380, // s_mov_b32 s20, 0
        0x7D88_00FF,
        40,          // v_cmp_gt_u32 vcc, 40, v0
        0xBF87_0001, // s_cbranch_vccnz -> the inner loop
        0xBF82_0005, // s_branch -> the join
        0xBE95_0380, // s_mov_b32 s21, 0
        0x8015_8115, // s_add_u32 s21, s21, 1
        0xBF06_8315, // s_cmp_eq_u32 s21, 3
        0xBF85_0001, // s_cbranch_scc1 -> the join
        0xBF82_FFFC, // s_branch -> the inner loop's header (unconditional back edge)
        0xD760_000C,
        0x0001_5100, // v_readlane_b32 s12, v0, 40
        0x8014_8114, // s_add_u32 s20, s20, 1
        0xBF0A_8214, // s_cmp_lt_u32 s20, 2
        0xBF85_FFF2, // s_cbranch_scc1 -> the outer loop's header
        0xBF07_800C, // s_cmp_lg_u32 s12, 0 (after the outer loop)
        0xBF85_0001, // s_cbranch_scc1 -> s_endpgm
        0xBF80_0000, // s_nop
        S_ENDPGM,
    ]
}

/// The same outer loop with an inner loop every lane enters together: its
/// unconditional back edge is capped too, but every lane of the group
/// leaves the loop through it together, so nothing after it is skipped.
fn capped_back_edge_in_a_uniform_loop() -> Vec<u32> {
    vec![
        0xBE94_0380, // s_mov_b32 s20, 0
        0xBE95_0380, // s_mov_b32 s21, 0
        0x8015_8115, // s_add_u32 s21, s21, 1
        0xBF06_8315, // s_cmp_eq_u32 s21, 3
        0xBF85_0001, // s_cbranch_scc1 -> the join
        0xBF82_FFFC, // s_branch -> the inner loop's header (unconditional back edge)
        0xD760_000C,
        0x0001_5100, // v_readlane_b32 s12, v0, 40
        0x8014_8114, // s_add_u32 s20, s20, 1
        0xBF0A_8214, // s_cmp_lt_u32 s20, 2
        0xBF85_FFF6, // s_cbranch_scc1 -> the outer loop's header
        0xBF07_800C, // s_cmp_lg_u32 s12, 0 (after the outer loop)
        0xBF85_0001, // s_cbranch_scc1 -> s_endpgm
        0xBF80_0000, // s_nop
        S_ENDPGM,
    ]
}

/// The divergent inner loop with a conditional back edge instead: the
/// iteration cap then only ends the loop through its normal exit, no lane
/// leaves the program early.
fn conditional_back_edge_in_a_divergent_loop() -> Vec<u32> {
    vec![
        0xBE94_0380, // s_mov_b32 s20, 0
        0x7D88_00FF,
        40,          // v_cmp_gt_u32 vcc, 40, v0
        0xBF87_0001, // s_cbranch_vccnz -> the inner loop
        0xBF82_0004, // s_branch -> the join
        0xBE95_0380, // s_mov_b32 s21, 0
        0x8015_8115, // s_add_u32 s21, s21, 1
        0xBF0A_8315, // s_cmp_lt_u32 s21, 3
        0xBF85_FFFD, // s_cbranch_scc1 -> the inner loop's header (conditional back edge)
        0xD760_000C,
        0x0001_5100, // v_readlane_b32 s12, v0, 40
        0x8014_8114, // s_add_u32 s20, s20, 1
        0xBF0A_8214, // s_cmp_lt_u32 s20, 2
        0xBF85_FFF3, // s_cbranch_scc1 -> the outer loop's header
        0xBF07_800C, // s_cmp_lg_u32 s12, 0 (after the outer loop)
        0xBF85_0001, // s_cbranch_scc1 -> s_endpgm
        0xBF80_0000, // s_nop
        S_ENDPGM,
    ]
}

/// The back-edge programs have the back edges their comments say: one
/// unconditional `s_branch` to a loop header where they say so, and the
/// words decode to the instructions their comments name.
#[test]
fn the_back_edge_programs_are_what_their_comments_say() {
    let back_edges = |w: &[u32]| {
        let p = wave64(w);
        let order = crate::ir::reverse_postorder(&p.blocks);
        let mut rpo = vec![usize::MAX; p.blocks.len()];
        for (r, &b) in order.iter().enumerate() {
            rpo[b] = r;
        }
        p.blocks
            .iter()
            .enumerate()
            .filter(|(b, blk)| match blk.exit {
                ir::Exit::Jump(t) => rpo[*b] != usize::MAX && rpo[t] <= rpo[*b],
                _ => false,
            })
            .count()
    };
    assert_eq!(back_edges(&capped_back_edge_in_a_divergent_loop()), 1);
    assert_eq!(back_edges(&capped_back_edge_in_a_uniform_loop()), 1);
    assert_eq!(back_edges(&conditional_back_edge_in_a_divergent_loop()), 0);
    for w in [
        capped_back_edge_in_a_divergent_loop(),
        capped_back_edge_in_a_uniform_loop(),
        conditional_back_edge_in_a_divergent_loop(),
    ] {
        let code: Vec<u8> = w.iter().flat_map(|w| w.to_le_bytes()).collect();
        let all = crate::gfx10::decode(&code)
            .unwrap()
            .iter()
            .map(|i| i.text(true))
            .collect::<Vec<_>>()
            .join("\n");
        for want in [
            "s_mov_b32 s20, 0",
            "s_add_u32 s21, s21, 1",
            "v_readlane_b32 s12, v0, 40",
            "s_cmp_lg_u32 s12, 0",
            "s_add_u32 s20, s20, 1",
            "s_cmp_lt_u32 s20, 2",
            "s_cbranch_scc1",
            "s_branch",
            "s_endpgm",
        ] {
            assert!(all.contains(want), "{want:?} not in:\n{all}");
        }
    }
    let code: Vec<u8> = capped_back_edge_in_a_divergent_loop()
        .iter()
        .flat_map(|w| w.to_le_bytes())
        .collect();
    let all = crate::gfx10::decode(&code)
        .unwrap()
        .iter()
        .map(|i| i.text(true))
        .collect::<Vec<_>>()
        .join("\n");
    for want in ["v_cmp_gt_u32 vcc", "s_cbranch_vccnz", "s_cmp_eq_u32 s21, 3"] {
        assert!(all.contains(want), "{want:?} not in:\n{all}");
    }
}

/// A half-wave exchange after a divergent loop whose unconditional back
/// edge lowering caps with an early return: one wave may leave the program
/// at the cap while another reaches the exchange's group barrier with
/// missing participants, so the exchange is refused in halves. As one
/// wave64 there are no halves and it translates.
#[test]
fn an_exchange_after_a_divergent_capped_back_edge_is_refused() {
    let w = capped_back_edge_in_a_divergent_loop();
    let why = lowered_halves(&w, 64)
        .err()
        .expect("refused in halves")
        .to_string();
    assert!(
        why.contains("a lane read that may cross 32-lane halves"),
        "{why}"
    );
    // One wave64: no halves, no exchange, and it translates.
    let whole = lowered(&w, 64, false);
    assert_eq!(whole.wave_size, Some(64));
}

/// The same exchange after a loop whose unconditional back edge is capped
/// but which every lane of the group runs together: all lanes leave at the
/// cap together, so the exchange stands in halves.
#[test]
fn an_exchange_after_a_uniform_capped_back_edge_translates() {
    let w = capped_back_edge_in_a_uniform_loop();
    let l = lowered_halves(&w, 64).expect("translated in halves");
    assert_eq!(l.wave_size, Some(32));
    assert!(l.wave_ops);
    // The exchange really is there: each half's store between two group
    // barriers.
    assert!(l.barriers > 0, "no dx.op.barrier written");
}

/// The same exchange after a divergent loop whose back edge is conditional:
/// the cap then ends the loop through its normal exit, so no lane leaves
/// early and the exchange stands in halves.
#[test]
fn an_exchange_after_a_divergent_conditional_back_edge_translates() {
    let w = conditional_back_edge_in_a_divergent_loop();
    let l = lowered_halves(&w, 64).expect("translated in halves");
    assert_eq!(l.wave_size, Some(32));
    assert!(l.wave_ops);
    assert!(l.barriers > 0, "no dx.op.barrier written");
}

/// A closed loop: its header's conditional back edge branches on `scc` from
/// comparing the loop-carried `s12` (a divergent condition), and the latch's
/// unconditional `s_branch` back to the header is the loop's only other
/// back edge — and the one lowering caps with an early return. Every back
/// edge to the header shares its iteration counter, so a lane whose compare
/// holds runs the conditional back edge and drives the shared count up; it
/// can reach the cap at the latch and leave the program there while another
/// lane goes on. The latch is still run by every lane of the group (it
/// post-dominates the branch), so only the shared-counter check can see the
/// escape: `v_readlane_b32` of console lane 40 in the loop, whose result the
/// header's branch reads, is refused in halves.
fn shared_counter_with_a_divergent_conditional_back_edge() -> Vec<u32> {
    vec![
        0xBE94_0380, // s_mov_b32 s20, 0
        0xBF82_0000, // s_branch -> the loop header
        0xBF07_800C, // s_cmp_lg_u32 s12, 0 (the loop-carried exchange's result)
        0xBF85_FFFE, // s_cbranch_scc1 -> the loop header (conditional back edge)
        0xBF82_0000, // s_branch -> the exchange
        0xD760_000C,
        0x0001_5100, // v_readlane_b32 s12, v0, 40
        0xBF82_0000, // s_branch -> the latch
        0xBF82_FFF9, // s_branch -> the loop header (unconditional latch, capped)
    ]
}

/// Blocker 1: the loop above shares one counter between its conditional and
/// unconditional back edges. The latch is in step, but the header is not
/// (the divergent branch's region) and the conditional back edge is not
/// uniform, so a lane may return early at the latch's cap while another
/// continues — the exchange in the loop is refused in halves.
#[test]
fn an_exchange_in_a_loop_whose_counter_a_divergent_back_edge_shares_is_refused() {
    let w = shared_counter_with_a_divergent_conditional_back_edge();
    // The words decode to what the comments say: one conditional back edge
    // to the header on vcc, one unconditional capped one from the latch.
    let p = wave64(&w);
    let order = crate::ir::reverse_postorder(&p.blocks);
    let mut rpo = vec![usize::MAX; p.blocks.len()];
    for (r, &b) in order.iter().enumerate() {
        rpo[b] = r;
    }
    let mut unconditional = 0;
    let mut conditional = 0;
    for (b, blk) in p.blocks.iter().enumerate() {
        let back = |t: usize| rpo[b] != usize::MAX && rpo[t] <= rpo[b];
        match blk.exit {
            ir::Exit::Jump(t) if back(t) => unconditional += 1,
            ir::Exit::Branch { taken, not_taken } if back(taken) || back(not_taken) => {
                conditional += 1;
            }
            _ => {}
        }
    }
    assert_eq!(unconditional, 1, "one capped unconditional back edge");
    assert_eq!(conditional, 1, "one conditional back edge to the same header");
    assert_eq!(
        p.blocks
            .iter()
            .filter(|b| b.insts.iter().any(|i| i.name() == "s_barrier"))
            .count(),
        0
    );
    let why = lowered_halves(&w, 64)
        .err()
        .expect("refused in halves")
        .to_string();
    assert!(
        why.contains("a lane read that may cross 32-lane halves"),
        "{why}"
    );
}

/// A lane-dependent branch into a closed self-loop (its only edge is the
/// unconditional `s_branch` back to itself, which lowering caps with an
/// early return) against a sibling path holding `v_readlane_b32` of console
/// lane 40. The lanes in the self-loop never reach the sibling path — they
/// leave at the cap — so the exchange there is refused in halves.
fn closed_self_loop_next_to_a_lane_read() -> Vec<u32> {
    vec![
        0x7D88_00FF,
        40,          // v_cmp_gt_u32 vcc, 40, v0 (vcc = v0 < 40)
        0xBF87_0001, // s_cbranch_vccnz -> the self-loop
        0xBF82_0001, // s_branch -> the sibling path
        0xBF82_FFFF, // s_branch -> itself: the closed self-loop (capped back edge)
        0xD760_000C,
        0x0001_5100, // v_readlane_b32 s12, v0, 40
        0xBF07_800C, // s_cmp_lg_u32 s12, 0
        0xBF85_0001, // s_cbranch_scc1 -> s_endpgm
        0xBF80_0000, // s_nop
        S_ENDPGM,
    ]
}

/// Blocker 2: the closed self-loop has no exit in the console's CFG, but
/// lowering's early return at its capped back edge is one, so the sibling
/// path with the lane read does not post-dominate the lane-dependent branch
/// and is not a reconvergence point — the exchange is refused in halves.
#[test]
fn an_exchange_on_the_sibling_of_a_closed_capped_self_loop_is_refused() {
    let w = closed_self_loop_next_to_a_lane_read();
    let p = wave64(&w);
    let order = crate::ir::reverse_postorder(&p.blocks);
    let mut rpo = vec![usize::MAX; p.blocks.len()];
    for (r, &b) in order.iter().enumerate() {
        rpo[b] = r;
    }
    assert_eq!(
        p.blocks
            .iter()
            .enumerate()
            .filter(|(b, blk)| match blk.exit {
                ir::Exit::Jump(t) => rpo[*b] != usize::MAX && rpo[t] <= rpo[*b],
                _ => false,
            })
            .count(),
        1,
        "one capped unconditional back edge (the self-loop)"
    );
    let why = lowered_halves(&w, 64)
        .err()
        .expect("refused in halves")
        .to_string();
    assert!(
        why.contains("a lane read that may cross 32-lane halves"),
        "{why}"
    );
}

/// The closed self-loop of [`closed_self_loop_next_to_a_lane_read`], with
/// `s_barrier; s_endpgm` on the sibling path instead of the lane read. The
/// lane read stays in the entry block (its result unused) so that halves
/// really runs.
fn closed_self_loop_next_to_an_s_barrier() -> Vec<u32> {
    vec![
        0x7D88_00FF,
        40,          // v_cmp_gt_u32 vcc, 40, v0 (vcc = v0 < 40)
        0xD760_000C,
        0x0001_5100, // v_readlane_b32 s12, v0, 40 (unused; activates halves)
        0xBF87_0001, // s_cbranch_vccnz -> the self-loop (word 6)
        0xBF82_0001, // s_branch -> the sibling path (word 7)
        0xBF82_FFFF, // s_branch -> itself: the closed self-loop (capped back edge)
        0xBF8A_0000, // s_barrier on the sibling path
        S_ENDPGM,
    ]
}

/// Blocker 3: an explicit `s_barrier` on the sibling path of a closed capped
/// self-loop is refused. The sibling is not reachable from the loop's
/// header, but the lane-dependent branch leads to the latch, so the escape
/// at the cap still reaches the sibling — in halves and as one wave64 alike.
#[test]
fn an_s_barrier_on_the_sibling_of_a_closed_capped_self_loop_is_refused() {
    let w = closed_self_loop_next_to_an_s_barrier();
    // The words decode to what the comments say: one capped back edge (the
    // self-loop) and the s_barrier on the sibling path.
    let p = wave64(&w);
    let order = crate::ir::reverse_postorder(&p.blocks);
    let mut rpo = vec![usize::MAX; p.blocks.len()];
    for (r, &b) in order.iter().enumerate() {
        rpo[b] = r;
    }
    assert_eq!(
        p.blocks
            .iter()
            .enumerate()
            .filter(|(b, blk)| match blk.exit {
                ir::Exit::Jump(t) => rpo[*b] != usize::MAX && rpo[t] <= rpo[*b],
                _ => false,
            })
            .count(),
        1,
        "one capped unconditional back edge (the self-loop)"
    );
    assert_eq!(
        p.blocks
            .iter()
            .filter(|b| b.insts.iter().any(|i| i.name() == "s_barrier"))
            .count(),
        1
    );
    for why in [
        lowered_halves(&w, 64).err().expect("refused in halves"),
        lowered_result(&w, 64, false)
            .err()
            .expect("refused as one wave64"),
    ] {
        let why = why.to_string();
        assert!(
            why.contains("an s_barrier where a lane may have left the program early"),
            "{why}"
        );
    }
}

/// The same closed self-loop, but the `s_barrier` sits before the
/// lane-dependent branch: every lane runs that block once before any lane
/// can have left at the cap, so the barrier is kept.
fn an_s_barrier_before_the_branch_of_a_closed_self_loop() -> Vec<u32> {
    vec![
        0xBF8A_0000, // s_barrier, before the branch
        0xD760_000C,
        0x0001_5100, // v_readlane_b32 s12, v0, 40 (unused; activates halves)
        0x7D88_00FF,
        40,          // v_cmp_gt_u32 vcc, 40, v0
        0xBF87_0001, // s_cbranch_vccnz -> the self-loop (word 7)
        0xBF82_0001, // s_branch -> the sibling path (word 8)
        0xBF82_FFFF, // s_branch -> itself: the closed self-loop (capped back edge)
        S_ENDPGM,    // the sibling path
    ]
}

/// An `s_barrier` before the divergent branch that leads to a closed capped
/// self-loop is not marked escaped: it translates, in halves and as one
/// wave64 alike.
#[test]
fn an_s_barrier_before_the_branch_of_a_closed_capped_self_loop_translates() {
    let w = an_s_barrier_before_the_branch_of_a_closed_self_loop();
    let l = lowered_halves(&w, 64).expect("translated in halves");
    assert_eq!(l.wave_size, Some(32));
    assert!(l.barriers > 0, "no dx.op.barrier written in halves");
    let whole = lowered(&w, 64, false);
    assert_eq!(whole.wave_size, Some(64));
    assert!(whole.barriers > 0, "no dx.op.barrier written as one wave64");
    // The explicit s_barrier is one barrier beyond the same program with an
    // s_nop in its place.
    let mut nop = w.clone();
    nop[0] = 0xBF80_0000; // s_nop instead of the s_barrier
    let without = lowered_halves(&nop, 64).expect("translated in halves");
    assert_eq!(
        l.barriers,
        without.barriers + 1,
        "the s_barrier adds exactly one dx.op.barrier"
    );
}

/// An earlier divergent branch whose arms reconverge at a join holding an
/// `s_barrier`, then a later divergent branch into a closed capped self-loop
/// (its only edge the unconditional `s_branch` back to itself, which
/// lowering caps with an early return). The lane read stays in the entry
/// block (its result unused) so that halves really runs.
fn an_s_barrier_at_an_earlier_join_before_a_closed_capped_self_loop() -> Vec<u32> {
    vec![
        0xD760_000C,
        0x0001_5100, // v_readlane_b32 s12, v0, 40 (unused; activates halves)
        0x7D88_00FF,
        40,          // v_cmp_gt_u32 vcc, 40, v0 (the earlier divergent branch)
        0xBF87_0001, // s_cbranch_vccnz -> arm A (word 6)
        0xBF82_0001, // s_branch -> the join (word 7)
        0xBF80_0000, // s_nop, arm A -> the join (word 7)
        0xBF8A_0000, // s_barrier at the join (word 7)
        0x7D88_00FF,
        40,          // v_cmp_gt_u32 vcc, 40, v0 (the later divergent branch)
        0xBF87_0001, // s_cbranch_vccnz -> the self-loop (word 12)
        0xBF82_0001, // s_branch -> the sibling path (word 13)
        0xBF82_FFFF, // s_branch -> itself: the closed self-loop (capped back edge)
        S_ENDPGM,    // the sibling path (word 13)
    ]
}

/// An `s_barrier` at the join of an earlier divergent branch, before a later
/// divergent branch into a closed capped self-loop, is not marked escaped:
/// every lane runs the join before any lane can have left at the self-loop's
/// cap, so the escape at the cap must not reach back past the reconvergence
/// at the join. It translates in halves and as one wave64 alike.
#[test]
fn an_s_barrier_at_an_earlier_join_before_a_closed_capped_self_loop_translates() {
    let w = an_s_barrier_at_an_earlier_join_before_a_closed_capped_self_loop();
    // The words give one capped back edge (the self-loop) and one s_barrier
    // at the earlier join.
    let p = wave64(&w);
    let order = crate::ir::reverse_postorder(&p.blocks);
    let mut rpo = vec![usize::MAX; p.blocks.len()];
    for (r, &b) in order.iter().enumerate() {
        rpo[b] = r;
    }
    assert_eq!(
        p.blocks
            .iter()
            .enumerate()
            .filter(|(b, blk)| match blk.exit {
                ir::Exit::Jump(t) => rpo[*b] != usize::MAX && rpo[t] <= rpo[*b],
                _ => false,
            })
            .count(),
        1,
        "one capped unconditional back edge (the self-loop)"
    );
    assert_eq!(
        p.blocks
            .iter()
            .filter(|b| b.insts.iter().any(|i| i.name() == "s_barrier"))
            .count(),
        1
    );
    let l = lowered_halves(&w, 64).expect("translated in halves");
    assert_eq!(l.wave_size, Some(32));
    assert!(l.barriers > 0, "no dx.op.barrier written in halves");
    let whole = lowered(&w, 64, false);
    assert_eq!(whole.wave_size, Some(64));
    assert!(whole.barriers > 0, "no dx.op.barrier written as one wave64");
    // The explicit s_barrier is one barrier beyond the same program with an
    // s_nop at the join.
    let mut nop = w.clone();
    nop[7] = 0xBF80_0000; // s_nop instead of the s_barrier at the join
    let without = lowered_halves(&nop, 64).expect("translated in halves");
    assert_eq!(
        l.barriers,
        without.barriers + 1,
        "the s_barrier adds exactly one dx.op.barrier"
    );
}

/// The same shape with a uniform branch (on `scc` from comparing the
/// constant `s20`): every lane takes the same arm, so no lane can be at the
/// sibling's barrier while another left at the cap, and the barrier is
/// kept.
fn an_s_barrier_on_the_sibling_behind_a_uniform_branch() -> Vec<u32> {
    vec![
        0xBE94_0380, // s_mov_b32 s20, 0
        0xD760_000C,
        0x0001_5100, // v_readlane_b32 s12, v0, 40 (unused; activates halves)
        0xBF07_8014, // s_cmp_lg_u32 s20, 0 (uniform: s20 is the constant 0)
        0xBF85_0001, // s_cbranch_scc1 -> the self-loop (word 6)
        0xBF82_0001, // s_branch -> the sibling path (word 7)
        0xBF82_FFFF, // s_branch -> itself: the closed self-loop (capped back edge)
        0xBF8A_0000, // s_barrier on the sibling path
        S_ENDPGM,
    ]
}

/// An `s_barrier` on the sibling path behind a uniform branch is not marked
/// escaped either: the branch sends every lane the same way.
#[test]
fn an_s_barrier_on_the_sibling_behind_a_uniform_branch_translates() {
    let w = an_s_barrier_on_the_sibling_behind_a_uniform_branch();
    let l = lowered_halves(&w, 64).expect("translated in halves");
    assert_eq!(l.wave_size, Some(32));
    assert!(l.barriers > 0, "no dx.op.barrier written in halves");
    let whole = lowered(&w, 64, false);
    assert_eq!(whole.wave_size, Some(64));
    assert!(whole.barriers > 0, "no dx.op.barrier written as one wave64");
    // The explicit s_barrier is one barrier beyond the same program with an
    // s_nop in its place.
    let mut nop = w.clone();
    nop[7] = 0xBF80_0000; // s_nop instead of the s_barrier on the sibling path
    let without = lowered_halves(&nop, 64).expect("translated in halves");
    assert_eq!(
        l.barriers,
        without.barriers + 1,
        "the s_barrier adds exactly one dx.op.barrier"
    );
}

/// The divergent capped loop of [`capped_back_edge_in_a_divergent_loop`],
/// with an explicit `s_barrier` at the join instead of the lane read: a lane
/// may have left the program early at the inner loop's cap, so the barrier
/// is refused even without any exchange.
fn a_divergent_capped_loop_then_an_s_barrier_at_the_join() -> Vec<u32> {
    vec![
        0xBE94_0380, // s_mov_b32 s20, 0
        0x7D88_00FF,
        40,          // v_cmp_gt_u32 vcc, 40, v0
        0xBF87_0001, // s_cbranch_vccnz -> the inner loop
        0xBF82_0005, // s_branch -> the join
        0xBE95_0380, // s_mov_b32 s21, 0
        0x8015_8115, // s_add_u32 s21, s21, 1
        0xBF06_8315, // s_cmp_eq_u32 s21, 3
        0xBF85_0001, // s_cbranch_scc1 -> the join
        0xBF82_FFFC, // s_branch -> the inner loop's header (capped back edge)
        0xBF8A_0000, // s_barrier at the join
        0xBF80_0000, // s_nop
        0x8014_8114, // s_add_u32 s20, s20, 1
        0xBF0A_8214, // s_cmp_lt_u32 s20, 2
        0xBF85_FFF2, // s_cbranch_scc1 -> the outer loop's header
        0xBF07_8000, // s_cmp_lg_u32 s0, 0
        0xD760_000C,
        0x0001_5100, // v_readlane_b32 s12, v0, 40 (unused; activates halves)
        S_ENDPGM,
    ]
}

/// Blocker 3: an explicit `s_barrier` at the join after a divergent loop
/// whose capped back edge may have let a lane leave the program early is
/// refused, in halves and as one wave64 alike.
#[test]
fn an_s_barrier_after_a_divergent_capped_loop_is_refused() {
    let w = a_divergent_capped_loop_then_an_s_barrier_at_the_join();
    for why in [
        lowered_halves(&w, 64).err().expect("refused in halves"),
        lowered_result(&w, 64, false)
            .err()
            .expect("refused as one wave64"),
    ] {
        let why = why.to_string();
        assert!(
            why.contains("an s_barrier where a lane may have left the program early"),
            "{why}"
        );
    }
}

/// The same `s_barrier` where no lane can have left early — a uniform loop
/// with a capped back edge, every lane leaving at the cap together — is
/// kept: the cap analysis does not reject legitimate barriers. The lane
/// read at the join stays (it is what makes halves run), and the explicit
/// `s_barrier` goes just before `s_endpgm`.
#[test]
fn an_s_barrier_after_a_uniform_capped_loop_translates() {
    let mut w = capped_back_edge_in_a_uniform_loop();
    // Keep the v_readlane at the join (word 6), so halves really runs, and
    // put the explicit s_barrier where the s_nop before s_endpgm was.
    w[13] = 0xBF8A_0000; // s_barrier
    let l = lowered_halves(&w, 64).expect("translated in halves");
    assert_eq!(l.wave_size, Some(32));
    assert!(l.barriers > 0, "no dx.op.barrier written in halves");
    // And as one wave64, where the barrier is the program's own.
    let whole = lowered(&w, 64, false);
    assert_eq!(whole.wave_size, Some(64));
    assert!(
        whole.barriers > 0,
        "no dx.op.barrier written as one wave64"
    );
    // The explicit s_barrier is one barrier beyond the same program without
    // it (the fixture itself, whose word 13 is the s_nop).
    let without = lowered_halves(&capped_back_edge_in_a_uniform_loop(), 64)
        .expect("translated in halves");
    assert_eq!(
        l.barriers,
        without.barriers + 1,
        "the s_barrier adds exactly one dx.op.barrier"
    );
}

#[test]
fn ds_bpermute_wave64_preserves_32_lane_half() {
    // Without reading v0 (`thread_id`), only the wave64 half-wave base
    // (`lane_index & !31`) introduces `dx.op.waveGetLaneIndex`.
    let words = [
        0x7E02_02FF,
        0x0000_0104, // v_mov_b32 v1, 0x104
        0x7E04_0285, // v_mov_b32 v2, 5
        0xDACC_000C,
        0x0300_0201, // ds_bpermute_b32 v3, v1, v2 offset:12
        0x7E00_0503, // v_readfirstlane_b32 s0, v3
        0xBF07_8000, // s_cmp_lg_u32 s0, 0
        0xBF85_0001, // s_cbranch_scc1 -> s_endpgm
        0xBE83_0380, // s_mov_b32 s3, 0
        S_ENDPGM,
    ];
    let code: Vec<u8> = words.iter().flat_map(|w| w.to_le_bytes()).collect();
    for wave64 in [false, true] {
        let native = NativeProgram {
            stage: HwStage::Cs,
            wave64,
            vgprs: 8,
            layout: vec![],
            used: vec![],
            io: StageIo::Cs(ComputeIo {
                rsrc2: 0,
                threads: [if wave64 { 64 } else { 32 }, 1, 1],
            }),
            code: code.clone(),
        };
        let p = ir::build(&native.code, wave64, native.vgprs).unwrap();
        let st = structure::structure(&p).unwrap();
        let res = resource::track(&p, &native.layout, native.stage, Some(&st)).unwrap();
        let io = io::raise(&p, &native).unwrap();
        let mut lowered = lower(&Parts {
            halves_rendezvous: false,
            native: &native,
            p: &p,
            st: &st,
            io: &io,
            res: &res,
            inputs: vec![],
            outputs: vec![],
            prims: vec![],
            mesh_stride: 0,
            atomic_locks: false,
            atomic_lock_rounds: false,
        atomic_umax: false,
            bias_as_gradients: false,
            narrow_waves: false,
            uav_size_level0: false,
            host_wave32: false,
            view_marks: false,
        })
        .unwrap();
        assert!(lowered.wave_ops);
        assert_eq!(lowered.uses_lane_index, wave64, "wave64={wave64}");
        assert_eq!(lowered.has_half_wave_mask, wave64, "wave64={wave64}");
        #[cfg(windows)]
        if let Ok(mut v) = crate::emit::validator::Validator::load(
            crate::emit::validator::Validator::default_path().as_deref(),
        ) {
            v.validate(&mut lowered.dxil).unwrap();
        }
    }
}

#[test]
fn ds_bpermute_refuses_gds() {
    // ds_bpermute_b32 with gds=1 (bit 17), consumed by a branch, must fail with GDS.
    let words = [
        0x7E02_0285,           // v_mov_b32 v1, 5
        0xDACC_0000 | 1 << 17, // ds_bpermute_b32 gds
        2 << 24 | 1 << 8,
        0x7E00_0502, // v_readfirstlane_b32 s0, v2
        0xBF07_8000, // s_cmp_lg_u32 s0, 0
        0xBF85_0001, // s_cbranch_scc1 -> s_endpgm
        0xBE83_0380, // s_mov_b32 s3, 0
        S_ENDPGM,
    ];
    assert!(matches!(
        lowers(&words, 0),
        Err(crate::ShaderError::Unsupported { what: "GDS", .. })
    ));
}

fn lower_ps_for_early_z(
    extra: &[u32],
    export: [u32; 2],
    outputs: Vec<crate::emit::container::SigElement>,
    db_shader_control: u32,
    view_marks: bool,
) -> Lowered {
    use crate::native::{HwStage, NativeProgram, RootParam, RootType, StageIo};
    const S_WAITCNT_ALL: u32 = 0xBF8C_0000;
    let mut w = vec![
        0xBE85_0380, // s_mov_b32 s5, 0
        0xF40C_0302, // s_load_dwordx8 s[12:19], s[4:5], 32
        0xFA00_0020,
        S_WAITCNT_ALL,
        0x7E00_0280, // v_mov_b32 v0, 0
        0x7E02_0280, // v_mov_b32 v1, 0
        0xF000_1F08, // image_load v[2:5], v[0:1], s[12:19] dmask:0xf dim:2D unorm
        0x0003_0200,
        S_WAITCNT_ALL,
    ];
    w.extend_from_slice(extra);
    w.extend_from_slice(&export);
    w.push(S_ENDPGM);
    let native = NativeProgram {
        stage: HwStage::Ps,
        wave64: false,
        vgprs: 8,
        layout: vec![RootParam {
            root: RootType::ResourceHeap,
            table_offset_dwords: 0,
            sgprs: 1,
            base_register: 0,
            user_data: 4,
            slot: 0,
        }],
        used: Vec::new(),
        io: StageIo::Ps {
            inputs: Vec::new(),
            input_addr: 0,
            db_shader_control,
            patches: Vec::new(),
        },
        code: w.iter().flat_map(|w| w.to_le_bytes()).collect(),
    };
    let p = crate::ir::build(&native.code, native.wave64, native.vgprs).expect("ir::build");
    let st = crate::structure::structure(&p).expect("structure");
    let res = crate::resource::track(&p, &native.layout, native.stage, Some(&st)).expect("track");
    let io = crate::io::raise(&p, &native).expect("io::raise");
    lower(&Parts {
        halves_rendezvous: false,
        native: &native,
        p: &p,
        st: &st,
        io: &io,
        res: &res,
        inputs: Vec::new(),
        outputs,
        prims: Vec::new(),
        mesh_stride: 0,
        atomic_locks: false,
        atomic_lock_rounds: false,
        atomic_umax: false,
        bias_as_gradients: false,
        narrow_waves: false,
        uav_size_level0: false,
        host_wave32: false,
        view_marks,
    })
    .expect("lower")
}

#[test]
fn pixel_shader_with_only_read_marks_enables_early_depth_stencil() {
    let lowered = lower_ps_for_early_z(&[], [0xF800_180F, 0x0504_0302], Vec::new(), 0, true);
    assert!(lowered.view_marks);
    assert!(lowered.early_depth_stencil);
    #[cfg(windows)]
    if let Ok(mut v) = crate::emit::validator::Validator::load(
        crate::emit::validator::Validator::default_path().as_deref(),
    ) {
        v.validate(&mut lowered.dxil.clone()).unwrap();
    }
}

#[test]
fn pixel_shader_skips_early_depth_stencil_with_discard_depth_or_uav() {
    // 1. No view_marks -> no injected UAV, no need for ForceEarlyDepthStencil.
    let no_marks = lower_ps_for_early_z(&[], [0xF800_180F, 0x0504_0302], Vec::new(), 0, false);
    assert!(!no_marks.early_depth_stencil);

    // 2. Depth export (exp mrtz v2, off, off, off done vm).
    let depth_out = vec![crate::emit::container::SigElement {
        name: "SV_Depth".into(),
        index: 0,
        kind: 17,
        comp: 9,
        interp: 0,
        row: None,
        col: 0,
        cols: 1,
        usage: 1,
        half: false,
    }];
    let depth = lower_ps_for_early_z(&[], [0xF800_1881, 0x0000_0002], depth_out, 0, true);
    assert!(!depth.early_depth_stencil);

    // 3. Stencil / POPS (db_shader_control bit 16).
    let pops = lower_ps_for_early_z(&[], [0xF800_180F, 0x0504_0302], Vec::new(), 1 << 16, true);
    assert!(!pops.early_depth_stencil);

    // 4. Conditional discard (divergent forward exit to `exp null done`).
    let discard = lower_ps_for_early_z(
        &[
            0xBE88_037E, // s_mov_b32 s8, exec_lo
            0x7C22_0480, // v_cmpx_lt_f32 exec, 0, v2
            0x8A08_7E08, // s_andn2_b32 s8, s8, exec_lo
            0xBF84_0004, // s_cbranch_scc0 -> kill (past s_mov + exp mrt0 + s_endpgm)
            0xBEFE_0308, // s_mov_b32 exec_lo, s8
            0xF800_180F, // exp mrt0 v2, v3, v4, v5 done vm
            0x0504_0302,
            S_ENDPGM,
        ],
        [0xF800_1890, 0x0000_0000], // kill: exp null done vm
        Vec::new(),
        0,
        true,
    );
    assert!(!discard.early_depth_stencil);

    // 5. Title UAV store (`buffer_store_dword` through heap slot 2).
    let uav = lower_ps_for_early_z(
        &[
            0xF408_0202, // s_load_dwordx4 s[8:11], s[4:5], 64
            0xFA00_0040,
            0xBF8C_0000, // s_waitcnt lgkmcnt(0)
            0xE070_0000, // buffer_store_dword v2, off, s[8:11], 0
            0x8002_0000,
        ],
        [0xF800_180F, 0x0504_0302],
        Vec::new(),
        0,
        true,
    );
    assert!(!uav.early_depth_stencil);
}

/// `DB_SHADER_CONTROL.DEPTH_BEFORE_SHADER`: the console tests depth before
/// the pixel shader, so a pixel behind closer geometry never makes the
/// shader's writes. The host shader then tests depth first too, title UAV
/// writes and discards notwithstanding; a depth export still keeps it late.
#[test]
fn console_depth_before_shader_keeps_early_depth_stencil_with_uav_writes() {
    const DEPTH_BEFORE_SHADER: u32 = 1 << 12;
    let uav_store = [
        0xF408_0202, // s_load_dwordx4 s[8:11], s[4:5], 64
        0xFA00_0040,
        0xBF8C_0000, // s_waitcnt lgkmcnt(0)
        0xE070_0000, // buffer_store_dword v2, off, s[8:11], 0
        0x8002_0000,
    ];
    let color = [0xF800_180F, 0x0504_0302]; // exp mrt0 v2, v3, v4, v5 done vm
    for view_marks in [false, true] {
        let uav = lower_ps_for_early_z(
            &uav_store,
            color,
            Vec::new(),
            DEPTH_BEFORE_SHADER,
            view_marks,
        );
        assert!(
            uav.early_depth_stencil,
            "title UAV store, view marks {view_marks}"
        );
        #[cfg(windows)]
        if let Ok(mut v) = crate::emit::validator::Validator::load(
            crate::emit::validator::Validator::default_path().as_deref(),
        ) {
            v.validate(&mut uav.dxil.clone()).unwrap();
        }
    }

    // Conditional discard (as in `pixel_shader_skips_..._or_uav`).
    let discard = lower_ps_for_early_z(
        &[
            0xBE88_037E, // s_mov_b32 s8, exec_lo
            0x7C22_0480, // v_cmpx_lt_f32 exec, 0, v2
            0x8A08_7E08, // s_andn2_b32 s8, s8, exec_lo
            0xBF84_0004, // s_cbranch_scc0 -> kill
            0xBEFE_0308, // s_mov_b32 exec_lo, s8
            0xF800_180F, // exp mrt0 v2, v3, v4, v5 done vm
            0x0504_0302,
            S_ENDPGM,
        ],
        [0xF800_1890, 0x0000_0000], // kill: exp null done vm
        Vec::new(),
        DEPTH_BEFORE_SHADER,
        true,
    );
    assert!(discard.early_depth_stencil, "discard");

    // Depth export (exp mrtz v2, off, off, off done vm).
    let depth_out = vec![crate::emit::container::SigElement {
        name: "SV_Depth".into(),
        index: 0,
        kind: 17,
        comp: 9,
        interp: 0,
        row: None,
        col: 0,
        cols: 1,
        usage: 1,
        half: false,
    }];
    let depth = lower_ps_for_early_z(
        &[],
        [0xF800_1881, 0x0000_0002],
        depth_out,
        DEPTH_BEFORE_SHADER,
        true,
    );
    assert!(!depth.early_depth_stencil, "depth export");
}

/// GLOBAL `op` with address `s[saddr:saddr+1] + vaddr + offset`.
const fn global(op: u32, vdst: u8, vaddr: u8, saddr: u8, offset: u32) -> [u32; 2] {
    [
        0x37 << 26 | op << 18 | 2 << 14 | (offset & 0xFFF),
        vaddr as u32 | (saddr as u32) << 16 | (vdst as u32) << 24,
    ]
}

/// Lowers a one-thread compute program with a pointer to title memory in
/// root constants s[0:1] (and s2, s3), the raw UAV of table entry 0 (s4) in
/// s[8:11], around `body`.
fn lower_memory(body: &[u32]) -> Lowered {
    let mut words = vec![
        0xBE85_0380, // s_mov_b32 s5, 0
        0xF408_0202, // s_load_dwordx4 s[8:11], s[4:5], 0
        0xFA00_0000,
    ];
    words.extend_from_slice(body);
    words.push(S_ENDPGM);
    let constants = RootParam {
        sgprs: 4,
        ..root(RootType::Constants, 0, 0)
    };
    let buffer = UsedResource {
        kind: ResourceType::Uav,
        root: RootType::Table,
        table_offset_dwords: 0,
        lower_bound: 0,
        stride: 0,
        dwords: 4,
        user_data: 4,
        slot: 1,
    };
    let l = lower_with(
        &words,
        vec![constants, root(RootType::Table, 4, 1)],
        vec![buffer],
    )
    .expect("lower");
    assert!(l.memory);
    l
}

/// Each title-memory access reads the page table twice, its level-1 entry
/// and then the page's entry (view and start in one load), and the null
/// pair's view, the same for all, is read once in the entry block: here
/// two loads behind a branch from it.
#[test]
fn the_null_pair_is_read_once_in_the_entry_block() {
    let mut w = vec![
        0xBF06_8002, // s_cmp_eq_u32 s2, 0
        0xBF85_000A, // s_cbranch_scc1 -> s_endpgm
        0x7E02_0280, // v_mov_b32 v1, 0
    ];
    w.extend(global(0x0C, 2, 1, 0, 0)); // global_load_dword v2, v1, s[0:1]
    w.extend(global(0x0C, 3, 1, 0, 0x100));
    w.push(0xBF8C_0000); // s_waitcnt
    w.extend([0xE070_0000, 0x8002_0200]); // buffer_store_dword v2, off, s[8:11], 0
    w.extend([0xE070_0004, 0x8002_0300]); // v3 at 4
    let l = lower_memory(&w);
    let [loads, table] = &l.raw_loads;
    assert_eq!(table.len(), 1 + 2 * 2, "{table:?}");
    assert_eq!(loads.len(), 1 + 2 * 3, "{loads:?}");
    assert_eq!(table.iter().filter(|&&b| b == 0).count(), 1, "{table:?}");
    assert_eq!(loads.iter().filter(|&&b| b == 0).count(), 1, "{loads:?}");
}

/// A box32 node's last 48 bytes are read through its first 64 bytes'
/// page-table lookup, and looked up again only in lanes whose node starts
/// in the last 64 bytes of a page (`raw_load_past`): the null pair's view
/// in the entry block, the head's lookup in the `ty < 6` block (so sentinel
/// `INVALID` and unhandled types 6/7 read nothing), then the head's 16
/// dwords and the tail's 12 through that lookup, and the second lookup in a
/// block of its own.
#[test]
fn a_box32_nodes_tail_reuses_its_heads_lookup() {
    // The descriptor in s[12:15], based at the pointer.
    let mut w = vec![
        0x908C_8800, // s_lshr_b64 s[12:13], s[0:1], 8
        0x880D_FF0D, // s_or_b32 s13, s13, 0x80000000
        0x8000_0000,
        0xBE8E_03C1, // s_mov_b32 s14, -1
        0xBE8F_03FF, // s_mov_b32 s15: type 8, barycentrics, no bound
        8 << 28 | 1 << 24 | 1 << 27 | 0x3FF,
    ];
    // v2..v12: the node pointer and the ray, from s2 and s3.
    w.extend((0..11u32).map(|k| 0x7E00_0200 | (2 + k) << 17 | (2 + k % 2)));
    // image_bvh_intersect_ray v[20:23], v[2:12], s[12:15]
    w.extend([0xF198_9F01, 20 << 8 | (12 >> 2) << 16 | 2]);
    // buffer_store_dwordx4 v[20:23], off, s[8:11], 0
    w.extend([0xE078_0000, 0x8002_1400]);
    let l = lower_memory(&w);
    let [loads, table] = &l.raw_loads;
    let at = |b: usize| loads.iter().filter(|&&x| x == b).count();
    assert_eq!(table.len(), 1 + 2 + 2, "{table:?}");
    assert_eq!(loads.len(), 1 + (2 + 4) + 3 + (2 + 3), "{loads:?}");
    assert_eq!(table[0], 0, "{table:?}");
    let head = table[1];
    assert!(head > 0 && table[2] == head, "{table:?}");
    let again = table[3];
    assert!(again > head && table[4] == again && at(again) == 2, "{loads:?} {table:?}");
    // Excluding the 2 page-table loads in `head`, the 4 head node loads and
    // 3 non-crossing tail node loads all reuse `head`'s lookup.
    let through_head = loads.iter().filter(|&&b| b >= head && b < again).count() - 2;
    assert_eq!(through_head, 4 + 3, "{loads:?} {table:?}");
}

#[test]
fn buffer_load_readfirstlane_loop_is_in_step() {
    let w: Vec<u32> = vec![
        0xBE84_0300, // s_mov_b32 s4, s0
        0xBE85_0380, // s_mov_b32 s5, 0
        0xF408_0B02,
        0xFA00_0020, // s_load_dwordx4 s[44:47], s[4:5], 0x20
        0xBF8C_C07F, // s_waitcnt lgkmcnt(0)
        0x7E06_0280, // v_mov_b32 v3, 0
        0xE030_1000,
        0x800B_0403, // buffer_load_dword v4, v3, s[44:47], 0 offen
        0xBF8C_3F70, // s_waitcnt vmcnt(0)
        0x7E2A_0504, // v_readfirstlane_b32 s21, v4
        0xBE94_0380, // s_mov_b32 s20, 0
        // loop body (block 1):
        0x8014_8114, // s_add_u32 s20, s20, 1
        0xBF0A_1514, // s_cmp_lt_u32 s20, s21
        0xBF85_FFFD, // s_cbranch_scc1 -3
        S_ENDPGM,
    ];
    let p = wave64(&w);
    let st = structure::structure(&p).unwrap();
    let step = super::in_step(&p, &st);
    assert!(
        step.iter().all(|&s| s),
        "expected all blocks in_step, got {step:?}; exits={:?}; blocks={:?}",
        st.exits,
        p.blocks
            .iter()
            .map(|b| b
                .insts
                .iter()
                .map(|i| (i.name(), i.reads.clone(), i.writes.clone()))
                .collect::<Vec<_>>())
            .collect::<Vec<_>>()
    );
}

/// A uniform branch on a number kept in VCC_HI: `source` puts a group-wide
/// value in s21, the program shifts it into VCC_HI, adds that to the
/// offset of an `s_buffer_load_dword`, and branches on the loaded dword.
fn vcc_hi_offset(source: &[u32]) -> Vec<u32> {
    let mut w = vec![
        0xBE84_0300, // s_mov_b32 s4, s0
        0xBE85_0380, // s_mov_b32 s5, 0
    ];
    w.extend(smem(2, 44, 4, 0x20, None)); // s_load_dwordx4 s[44:47], s[4:5], 0x20
    w.push(0xBF8C_C07F); // s_waitcnt lgkmcnt(0)
    w.extend(source);
    w.extend([
        0x8F6B_8215, // s_lshl_b32 vcc_hi, s21, 2
        0x8817_806B, // s_or_b32 s23, vcc_hi, 0
    ]);
    w.extend(smem(8, 24, 44, 0, Some(23))); // s_buffer_load_dword s24, s[44:47], s23
    w.extend([
        0xBF8C_C07F, // s_waitcnt lgkmcnt(0)
        0xBF07_8018, // s_cmp_lg_u32 s24, 0
        0xBF84_0001, // s_cbranch_scc0 1
        0x7E06_0281, // v_mov_b32 v3, 1
        S_ENDPGM,
    ]);
    w
}

/// A branch on VCC_HI after `def` puts something in VCC: `s_cmp_lg_u32
/// vcc_hi, 0` and an `s_cbranch_scc0` over one vector move.
fn vcc_hi_branch(def: &[u32]) -> Vec<u32> {
    let mut w = vec![
        0xBE84_0300, // s_mov_b32 s4, s0
        0xBE85_0300, // s_mov_b32 s5, s0
    ];
    w.extend(def);
    w.extend([
        0xBF07_806B, // s_cmp_lg_u32 vcc_hi, 0
        0xBF84_0001, // s_cbranch_scc0 1
        0x7E06_0281, // v_mov_b32 v3, 1
        S_ENDPGM,
    ]);
    w
}

/// The (a)/(b)/(c)/(d) programs below, by their instructions' names.
fn names(words: &[u32]) -> Vec<&'static str> {
    wave64(words)
        .blocks
        .iter()
        .flat_map(|b| b.insts.iter().map(|i| i.name()))
        .collect()
}

#[test]
fn the_vcc_scratch_programs_are_what_their_comments_say() {
    let first = vcc_hi_offset(&[
        0x7E06_0280, // v_mov_b32 v3, 0
        0xE030_1000,
        0x800B_0403, // buffer_load_dword v4, v3, s[44:47], 0 offen
        0xBF8C_3F70, // s_waitcnt vmcnt(0)
        0x7E2A_0504, // v_readfirstlane_b32 s21, v4
    ]);
    let n = names(&first);
    for want in [
        "s_lshl_b32",
        "s_or_b32",
        "s_buffer_load_dword",
        "s_cmp_lg_u32",
        "s_cbranch_scc0",
        "v_readfirstlane_b32",
    ] {
        assert!(n.contains(&want), "{want} missing from {n:?}");
    }
    for (def, want) in [
        (vec![0x7D88_0085], "v_cmp_gt_u32"),
        (vec![0x87EA_047E], "s_and_b64"),
        (vec![0xBE8A_047E, 0xBEEA_040A], "s_mov_b64"),
    ] {
        let n = names(&vcc_hi_branch(&def));
        assert!(n.contains(&want), "{want} missing from {n:?}");
    }
    let p = wave64(&vcc_hi_branch(&[0x87EA_047E]));
    let and = p.blocks[0].insts.iter().find(|i| i.name() == "s_and_b64").unwrap();
    assert!(and.reads.iter().any(|r| r.0 == ir::Reg::S(ir::EXEC)));
    assert!(and.writes.iter().any(|w| w.0 == ir::Reg::S(ir::VCC + 1)));
}

/// (a) A first-lane value the compiler keeps in VCC_HI as a spare SGPR, on
/// its way into a scalar load's offset, is data, not a lane mask: in literal
/// wave64 halves the branch on the loaded dword is group-uniform and every
/// block is in step.
#[test]
fn a_first_lane_number_in_vcc_hi_keeps_literal_halves_in_step() {
    let w = vcc_hi_offset(&[
        0x7E06_0280, // v_mov_b32 v3, 0
        0xE030_1000,
        0x800B_0403, // buffer_load_dword v4, v3, s[44:47], 0 offen
        0xBF8C_3F70, // s_waitcnt vmcnt(0)
        0x7E2A_0504, // v_readfirstlane_b32 s21, v4
    ]);
    let p = wave64(&w);
    let st = structure::split_literal(&p).unwrap();
    let step = super::in_step(&p, &st);
    assert!(step.iter().all(|&s| s), "{step:?}");
}

/// A number from user data kept in VCC_HI is group-uniform in per-lane
/// halves too.
#[test]
fn a_uniform_number_in_vcc_hi_is_group_uniform() {
    let w = vcc_hi_offset(&[0xBE95_0300]); // s_mov_b32 s21, s0
    let p = wave64(&w);
    let st = structure::structure(&p).unwrap();
    let step = super::in_step(&p, &st);
    assert!(step.iter().all(|&s| s), "{step:?}");
}

/// Per-lane wave64 halves: whether every block of `def` then a branch on
/// VCC_HI is in step.
fn vcc_hi_branch_in_step(def: &[u32]) -> bool {
    let p = wave64(&vcc_hi_branch(def));
    let st = structure::structure(&p).unwrap();
    assert!(!st.literal);
    super::in_step(&p, &st).iter().all(|&s| s)
}

/// (b) A `v_cmp` result in VCC is a lane mask, whatever scalar op reads it.
#[test]
fn a_compare_mask_in_vcc_is_not_group_uniform() {
    assert!(!vcc_hi_branch_in_step(&[
        0x7D88_0085, // v_cmp_gt_u32 vcc, 5, v0
    ]));
    assert!(!vcc_hi_branch_in_step(&[
        0x7D88_0085, // v_cmp_gt_u32 vcc, 5, v0
        0x8F6B_816B, // s_lshl_b32 vcc_hi, vcc_hi, 1
    ]));
}

/// (c) A 64-bit scalar write of VCC from EXEC: VCC_HI holds the upper 32
/// lanes' bits, which differ between the host halves.
#[test]
fn exec_anded_into_vcc_is_not_group_uniform() {
    assert!(!vcc_hi_branch_in_step(&[
        0x87EA_047E, // s_and_b64 vcc, exec, s[4:5]
    ]));
}

/// (d) A copy of EXEC saved early and moved into VCC later is still EXEC.
#[test]
fn a_saved_exec_moved_into_vcc_is_not_group_uniform() {
    assert!(!vcc_hi_branch_in_step(&[
        0xBE8A_047E, // s_mov_b64 s[10:11], exec
        0x8006_8104, // s_add_u32 s6, s4, 1
        0xBEEA_040A, // s_mov_b64 vcc, s[10:11]
    ]));
}

/// A 64-bit scalar write of VCC is left out even from uniform sources: the
/// rule admits 32-bit writes of one VCC dword only.
#[test]
fn a_64_bit_write_of_vcc_is_not_taken_as_a_number() {
    assert!(!vcc_hi_branch_in_step(&[
        0xBEEA_0404, // s_mov_b64 vcc, s[4:5]
    ]));
    assert!(vcc_hi_branch_in_step(&[
        0xBEEB_0304, // s_mov_b32 vcc_hi, s4
    ]));
}

/// SMEM `op` (`s_load_dword*` 0-4, `s_buffer_load_dword*` 8-12) of
/// `s[sdata..]` from `s[sbase..]` at `offset`, plus SGPR `soffset` if any.
fn smem(op: u32, sdata: u32, sbase: u32, offset: u32, soffset: Option<u32>) -> [u32; 2] {
    [
        0xF400_0000 | op << 18 | sdata << 6 | sbase >> 1,
        soffset.unwrap_or(125) << 25 | offset,
    ]
}

/// A compute program that loads the raw UAV of the table at `s[uav]` into
/// s[12:15], runs `body`, and stores `n` registers from `first` on (SGPRs
/// through v1, or VGPRs if `vector`) to its dwords in order, so every
/// dword `body` loads is needed.
fn stored(uav: u32, body: &[u32], first: u32, n: u32, vector: bool) -> Vec<u32> {
    let mut w = vec![0xBE9A_0300 | uav, 0xBE9B_0380]; // s_mov_b32 s26, s[uav]; s_mov_b32 s27, 0
    w.extend(smem(2, 12, 26, 0, None)); // s_load_dwordx4 s[12:15], s[26:27], 0
    w.extend(body);
    w.push(0xBF8C_0000); // s_waitcnt vmcnt(0) lgkmcnt(0)
    for k in 0..n {
        let v = if vector {
            first + k
        } else {
            w.push(0x7E02_0200 | (first + k)); // v_mov_b32 v1, s[first + k]
            1
        };
        // buffer_store_dword v, off, s[12:15], 0 offset:4k
        w.extend([0xE070_0000 | (4 * k), 0x8003_0000 | v << 8]);
    }
    w.push(S_ENDPGM);
    w
}

fn used(kind: ResourceType, root: RootType, user_data: u8, slot: u8) -> UsedResource {
    UsedResource {
        kind,
        root,
        table_offset_dwords: 0,
        lower_bound: 0,
        stride: 0,
        dwords: if root == RootType::Descriptor { 2 } else { 4 },
        user_data,
        slot,
    }
}

/// `s_buffer_load_dword*` `op` of s[32..] at `offset` (plus s2 if
/// `soffset`) through the constant buffer in the table at s0; the UAV's
/// table at s1.
fn table_constants(op: u32, offset: u32, soffset: bool) -> Lowered {
    let mut body = vec![0xBE84_0300, 0xBE85_0380]; // s_mov_b32 s4, s0; s_mov_b32 s5, 0
    body.extend(smem(2, 8, 4, 0, None)); // s_load_dwordx4 s[8:11], s[4:5], 0
    body.push(0xBF8C_0000);
    body.extend(smem(op, 32, 8, offset, soffset.then_some(2)));
    let layout = vec![
        root(RootType::Table, 0, 0),
        root(RootType::Table, 1, 1),
        root(RootType::Constants, 2, 2),
    ];
    let used = vec![
        used(ResourceType::Cbv, RootType::Table, 0, 0),
        used(ResourceType::Uav, RootType::Table, 1, 1),
    ];
    lower_with(&stored(1, &body, 32, 1 << (op - 8), false), layout, used).unwrap()
}

/// `s_buffer_load_dword*` `op` of s[32..] at 0 through the constant buffer
/// at the heap index in s0 (the resource heap at s1); the UAV's table at s2.
fn heap_constants(op: u32) -> Lowered {
    let mut body = vec![0x8F04_8500, 0xBE94_0301, 0xBE95_0380]; // s_lshl_b32 s4, s0, 5; s[20:21] = s1, 0
    body.extend(smem(2, 8, 20, 0, Some(4))); // s_load_dwordx4 s[8:11], s[20:21], s4
    body.push(0xBF8C_0000);
    body.extend(smem(op, 32, 8, 0, None));
    let layout = vec![
        root(RootType::Constants, 0, 0),
        root(RootType::ResourceHeap, 1, 1),
        root(RootType::Table, 2, 2),
    ];
    let used = vec![used(ResourceType::Uav, RootType::Table, 2, 2)];
    lower_with(&stored(2, &body, 32, 1 << (op - 8), false), layout, used).unwrap()
}

/// A buffer descriptor built around the root CBV at s[0:1], in s[8:11].
const ROOT_CBV_DESCRIPTOR: [u32; 5] = [
    0xBE88_0300, // s_mov_b32 s8, s0
    0xBE89_0301, // s_mov_b32 s9, s1
    0xBE8A_03C1, // s_mov_b32 s10, -1
    0xBE8B_03FF,
    0x2100_0FAC, // s_mov_b32 s11, 0x21000fac
];

fn root_cbv_layout() -> (Vec<RootParam>, Vec<UsedResource>) {
    (
        vec![
            root(RootType::Descriptor, 0, 0),
            root(RootType::Constants, 2, 1),
            root(RootType::Table, 3, 2),
        ],
        vec![
            used(ResourceType::Cbv, RootType::Descriptor, 0, 0),
            used(ResourceType::Uav, RootType::Table, 3, 2),
        ],
    )
}

/// `s_buffer_load_dword*` `op` of s[32..] through the root CBV at s[0:1],
/// at 4 plus s2 if `soffset`; the UAV's table at s3.
fn root_constants(op: u32, soffset: bool) -> Lowered {
    let mut body = ROOT_CBV_DESCRIPTOR.to_vec();
    body.extend(smem(op, 32, 8, 4, soffset.then_some(2)));
    let (layout, used) = root_cbv_layout();
    lower_with(&stored(3, &body, 32, 1 << (op - 8), false), layout, used).unwrap()
}

/// `buffer_load_dwordx4 v[4:7], v0, s[8:11], 0 offen offset:4`: a row of
/// constants per lane through the constant buffer in the table at s0, or
/// through the root CBV at s[0:1] (the UAV's table at s3).
fn lane_constants(table: bool) -> Lowered {
    let mut body = if table {
        let mut b = vec![0xBE84_0300, 0xBE85_0380];
        b.extend(smem(2, 8, 4, 0, None));
        b.push(0xBF8C_0000);
        b
    } else {
        ROOT_CBV_DESCRIPTOR.to_vec()
    };
    body.extend([0xE038_1004, 0x8002_0400]);
    let (layout, used) = if table {
        (
            vec![root(RootType::Table, 0, 0), root(RootType::Table, 3, 2)],
            vec![
                used(ResourceType::Cbv, RootType::Table, 0, 0),
                used(ResourceType::Uav, RootType::Table, 3, 2),
            ],
        )
    } else {
        root_cbv_layout()
    };
    lower_with(&stored(3, &body, 4, 4, true), layout, used).unwrap()
}

/// `global_load_dword*` `op` (12, 13, 15, 14: one to four dwords) of v[4..]
/// at 8 through the root SRV at s[0:1]; the UAV's table at s2.
fn root_view_load(op: u32, n: u32) -> Lowered {
    let body = [
        0x7E02_0280, // v_mov_b32 v1, 0
        0xDC00_8008 | op << 18,
        0x0400_0001, // global_load_dword* v[4..], v1, s[0:1] offset:8
    ];
    let layout = vec![
        root(RootType::Descriptor, 0, 0),
        root(RootType::Table, 2, 1),
    ];
    let used = vec![
        used(ResourceType::Srv, RootType::Descriptor, 0, 0),
        used(ResourceType::Uav, RootType::Table, 2, 1),
    ];
    lower_with(&stored(2, &body, 4, n, true), layout, used).unwrap()
}

/// `s_load_dword*` `op` of s[32..] from title memory at the pointer in
/// root constants s[0:1]; the UAV's table at s2.
fn memory_constants(op: u32) -> Lowered {
    let body = smem(op, 32, 0, 0x10, None);
    let layout = vec![
        RootParam {
            sgprs: 2,
            ..root(RootType::Constants, 0, 0)
        },
        root(RootType::Table, 2, 1),
    ];
    let used = vec![used(ResourceType::Uav, RootType::Table, 2, 1)];
    lower_with(&stored(2, &body, 32, 1 << op, false), layout, used).unwrap()
}

/// `l`'s raw buffer loads of more than one dword (the page table's and the
/// stride guard's read one).
fn wide_loads(l: &Lowered) -> Vec<(Option<i64>, Option<i64>)> {
    l.raw_load_args
        .iter()
        .copied()
        .filter(|&(_, m)| m != Some(1))
        .collect()
}

/// Constants through a raw view are read four dwords a load, at any
/// 4-byte offset (raw loads need no more alignment), the last load masked
/// to what is left.
#[test]
fn scalar_buffer_loads_read_four_dwords_a_load() {
    let four = |at: i64, n: i64| {
        (0..n)
            .map(|k| (Some(at + 16 * k), Some(15)))
            .collect::<Vec<_>>()
    };
    // In a table, at a fixed offset.
    assert_eq!(
        table_constants(8, 0, false).raw_load_args,
        [(Some(0), Some(1))]
    );
    assert_eq!(
        table_constants(9, 0, false).raw_load_args,
        [(Some(0), Some(3))]
    );
    assert_eq!(table_constants(10, 0, false).raw_load_args, four(0, 1));
    assert_eq!(table_constants(11, 4, false).raw_load_args, four(4, 2));
    assert_eq!(table_constants(12, 4, false).raw_load_args, four(4, 4));
    // At a computed one.
    assert_eq!(
        table_constants(12, 4, true).raw_load_args,
        [(None, Some(15)); 4]
    );
    // Through the heap itself, beside the stride guard's load.
    let heap = heap_constants(12);
    assert_eq!(heap.raw_load_args.len(), 5);
    assert_eq!(wide_loads(&heap), four(0, 4));
    // Per lane through a table, and global loads through a root view.
    assert_eq!(lane_constants(true).raw_load_args, [(None, Some(15))]);
    for (op, n, mask) in [(13, 2, 3), (15, 3, 7), (14, 4, 15)] {
        assert_eq!(root_view_load(op, n).raw_load_args, [(None, Some(mask))]);
    }
    // From title memory: the page table's entry (two dwords) for the run's
    // first byte and, in a block that only lanes whose run crosses onto the
    // next page run, for its last; the constants four dwords a load, each
    // dword also read through the next page's copy (a load each); then a
    // dword a load again, the same way, in a block of their own, which only
    // lanes whose run wraps (it starts before its copy) run.
    let memory = memory_constants(4);
    assert_eq!(
        wide_loads(&memory),
        [
            (None, Some(3)),
            (None, Some(3)),
            (None, Some(15)),
            (None, Some(15)),
            (None, Some(15)),
            (None, Some(15)),
        ]
    );
    let [blocks, table] = &memory.raw_loads;
    let (first, crossing) = (table[1], table[3]);
    assert!(first == table[2] && crossing == table[4] && crossing != first, "{table:?}");
    let run = blocks.iter().position(|&b| b != first && b != crossing).expect("the run's loads");
    let wraps = *blocks.last().expect("loads");
    assert!(wraps != blocks[run], "{blocks:?}");
    // Four loads of four with four of one each, then sixteen pairs of one.
    assert_eq!(blocks.iter().filter(|&&b| b == blocks[run]).count(), 4 * 5, "{blocks:?}");
    assert_eq!(blocks.iter().filter(|&&b| b == wraps).count(), 16 * 2, "{blocks:?}");
    let at_wraps = blocks.len() - 32;
    assert_eq!(memory.raw_load_args[at_wraps..], [(None, Some(1)); 32]);
}

/// Constants at a computed offset of a constant buffer view load each row
/// the run touches once: one more row than its aligned rows, which an
/// unaligned run reaches into. The user data take a row, and one more with
/// the offset in a root constant.
#[test]
fn constants_at_a_computed_offset_load_each_row_once() {
    let rows = [8, 9, 10, 11, 12].map(|op| root_constants(op, true).cbuffer_rows - 2);
    assert_eq!(rows, [1, 2, 2, 3, 5]);
    // Per lane: four dwords, two rows.
    assert_eq!(lane_constants(false).cbuffer_rows - 1, 2);
}
/// Lowers a wave64 compute program of 64 threads as a host of 32-lane waves
/// runs it (`host_wave32`: the console wave is two halves), with root
/// constants in s[0:3] and no LDS (halves refuse one console wave's LDS).
fn lower_halves(words: &[u32]) -> Result<Lowered, crate::ShaderError> {
    lower_halves_with(
        words,
        vec![RootParam {
            sgprs: 4,
            ..root(RootType::Constants, 0, 0)
        }],
        vec![],
    )
}

/// [`lower_halves`] with the root parameters `layout` (in user SGPRs from s0
/// on) and the used resources `used`.
fn lower_halves_with(
    words: &[u32],
    layout: Vec<RootParam>,
    used: Vec<UsedResource>,
) -> Result<Lowered, crate::ShaderError> {
    let sgprs: u32 = layout.iter().map(|p| p.sgprs).sum();
    lower_halves_io(words, layout, used, sgprs.max(2) << 1, [64, 1, 1])
}

/// [`lower_halves_with`] with a custom `rsrc2` and thread count — a program
/// that allocates LDS in a group of two console waves, say, where LDS keeps
/// its barriers in halves and only the value analysis can refuse.
fn lower_halves_io(
    words: &[u32],
    layout: Vec<RootParam>,
    used: Vec<UsedResource>,
    rsrc2: u32,
    threads: [u32; 3],
) -> Result<Lowered, crate::ShaderError> {
    let code: Vec<u8> = words.iter().flat_map(|w| w.to_le_bytes()).collect();
    let native = NativeProgram {
        stage: HwStage::Cs,
        wave64: true,
        vgprs: 256,
        layout,
        used,
        io: StageIo::Cs(ComputeIo { rsrc2, threads }),
        code,
    };
    let p = ir::build(&native.code, true, 256).unwrap();
    let st = structure::structure(&p).unwrap();
    let res = resource::track(&p, &native.layout, native.stage, Some(&st)).unwrap();
    let io = io::raise(&p, &native).unwrap();
    // These tests probe the value analysis behind each refusal, which
    // selective predication of a divergent branch would route around.
    super::NO_PREDICATION.with(|c| c.set(true));
    // The wave ticket's own barrier would count among the exchanges these
    // tests look for, so they lower without it.
    let lowered = crate::emit::with_thread_ticket(false, || {
        lower(&Parts {
            halves_rendezvous: false,
            native: &native,
            p: &p,
            st: &st,
            io: &io,
            res: &res,
            inputs: vec![],
            outputs: vec![],
            prims: vec![],
            mesh_stride: 0,
            atomic_locks: false,
            atomic_lock_rounds: false,
            atomic_umax: false,
            bias_as_gradients: false,
            host_wave32: true,
            narrow_waves: false,
            uav_size_level0: false,
            view_marks: false,
        })
    });
    super::NO_PREDICATION.with(|c| c.set(false));
    lowered
}

/// A `v_writelane_b32` asks for the console's lane layout (so the program
/// runs in halves), and a branch on a compare of the thread id makes the
/// block after it divergent. `read` (a first-lane read) goes there, and its
/// SGPR result is branched on, so a refusal in it surfaces.
const WRITELANE_V4_V1_5: [u32; 2] = [0xD761_0004, 0x0001_0B01]; // v_writelane_b32 v4, v1, 5

fn divergent_read(w: &mut Vec<u32>, read: u32, cmp: u32) {
    w.extend(WRITELANE_V4_V1_5);
    w.extend([CMPX_V0_EQ_0, skip(3)]);
    w.extend([read, cmp, 0xBF85_0001]); // s_cbranch_scc1 over the s_mov
    w.extend([0xBE87_0380, S_ENDPGM]); // s_mov_b32 s7, 0; s_endpgm
}

/// A `global_load_dword` at an address the whole console wave holds alike
/// loads one value, so a `v_readfirstlane_b32` of it inside a divergent
/// block — where a cross-half exchange would be refused — lowers without
/// one.
#[test]
fn a_uniform_global_loads_first_lane_read_needs_no_exchange() {
    // v_mov_b32 v1, 0x1000; global_load_dword v2, v1, s[0:1], 0
    let mut w = vec![0x7E02_02FF, 0x0000_1000];
    w.extend(global(0x0C, 2, 1, 0, 0));
    divergent_read(&mut w, 0x7E0C_0502, 0xBF07_8006); // v_readfirstlane_b32 s6, v2; s_cmp_lg_u32 s6, 0
    let l = lower_halves(&w).expect("a uniform global load lowers in halves");
    assert_eq!(l.wave_size, Some(32), "the console wave ran as two halves");
    assert_eq!(
        l.shape.barriers, 0,
        "no cross-half exchange was emitted (where one would be refused)"
    );
    // The VGPR-pair address form (saddr = null): both words alike.
    let mut w = vec![0x7E02_02FF, 0x0000_1000]; // v_mov_b32 v1, 0x1000
    w.push(0x7E04_0280); // v_mov_b32 v2, 0
    w.extend(global(0x0C, 3, 1, 0x7D, 0)); // global_load_dword v3, v[1:2], 0
    divergent_read(&mut w, 0x7E0E_0503, 0xBF07_8007); // v_readfirstlane_b32 s7, v3; s_cmp_lg_u32 s7, 0
    let l = lower_halves(&w).expect("a uniform VGPR-pair global load lowers in halves");
    assert_eq!(l.wave_size, Some(32));
    assert_eq!(l.shape.barriers, 0, "no cross-half exchange was emitted");
}

/// A `global_load_dword` at a lane-varying address does not give a value the
/// same in every lane: the first-lane read of it in a divergent block keeps
/// its cross-half refusal, and in a block every lane runs together keeps
/// its cross-half exchange.
#[test]
fn a_lane_varying_global_load_keeps_the_first_lane_refusal() {
    let first_across = super::halves::FIRST_ACROSS;
    let refused = |w: &[u32]| matches!(lower_halves(w), Err(crate::ShaderError::Unsupported { what, .. }) if what == first_across);
    // The VGPR offset is the thread id.
    let mut w = vec![0x7E02_0300]; // v_mov_b32 v1, v0
    w.extend(global(0x0C, 2, 1, 0, 0)); // global_load_dword v2, v1, s[0:1], 0
    divergent_read(&mut w, 0x7E0C_0502, 0xBF07_8006); // v_readfirstlane_b32 s6, v2; s_cmp_lg_u32 s6, 0
    assert!(refused(&w), "a lane-varying VGPR offset keeps the refusal");
    // The high dword of a VGPR-pair address is the thread id.
    let mut w = vec![0x7E02_02FF, 0x0000_1000]; // v_mov_b32 v1, 0x1000
    w.push(0x7E04_0300); // v_mov_b32 v2, v0
    w.extend(global(0x0C, 3, 1, 0x7D, 0)); // global_load_dword v3, v[1:2], 0
    divergent_read(&mut w, 0x7E0E_0503, 0xBF07_8007); // v_readfirstlane_b32 s7, v3; s_cmp_lg_u32 s7, 0
    assert!(
        refused(&w),
        "a lane-varying second VGPR word keeps the refusal"
    );
    // The same lane-varying load, read first in the entry block every lane
    // runs together: the cross-half exchange is kept.
    let mut w = vec![0x7E02_0300]; // v_mov_b32 v1, v0
    w.extend(global(0x0C, 2, 1, 0, 0)); // global_load_dword v2, v1, s[0:1], 0
    w.extend(WRITELANE_V4_V1_5);
    w.extend([0x7E0C_0502, 0xBF07_8006, 0xBF85_0001, 0xBE87_0380, S_ENDPGM]);
    let l = lower_halves(&w).expect("an exchange can go in the entry block");
    assert!(l.shape.barriers > 0, "the cross-half exchange was emitted");
}

/// GLOBAL `global_store_dword` of `vdata` at `s[saddr:saddr+1] + vaddr +
/// offset`.
const fn global_store(vdata: u8, vaddr: u8, saddr: u8, offset: u32) -> [u32; 2] {
    [
        0x37 << 26 | 0x1C << 18 | 2 << 14 | (offset & 0xFFF),
        vaddr as u32 | (vdata as u32) << 8 | (saddr as u32) << 16,
    ]
}

/// The first-lane read of a `global_load_dword*`'s result is refused
/// (`FIRST_ACROSS`) in a divergent block.
fn refused_in_halves(w: &[u32]) -> bool {
    let first_across = super::halves::FIRST_ACROSS;
    matches!(lower_halves(w), Err(crate::ShaderError::Unsupported { what, .. }) if what == first_across)
}

/// A `global_load_dword` at an address every lane of the console wave holds
/// alike, in a divergent block whose active lanes are in both halves (a
/// branch true everywhere but lane 5, and a `v_writelane_b32` into the
/// upper half): the first-lane read of it is observable (branched on) and
/// still needs no cross-half exchange.
#[test]
fn a_uniform_global_load_read_first_with_lanes_active_in_both_halves() {
    let mut w = vec![0x7E02_02FF, 0x0000_1000]; // v_mov_b32 v1, 0x1000
    w.extend(global(0x0C, 2, 1, 0, 0)); // global_load_dword v2, v1, s[0:1], 0
    w.extend(WRITELANE_V4_V1_5); // v_writelane_b32 v4, v1, 5
    w.extend([0xD761_0005, 0x0001_5101]); // v_writelane_b32 v5, v1, 40 (the upper half)
    w.extend([0x7DAA_0085, skip(3)]); // v_cmpx_ne_u32 exec, 5, v0; s_cbranch_scc1
    w.extend([0x7E0C_0502, 0xBF07_8006, 0xBF85_0001]); // v_readfirstlane_b32 s6, v2; s_cmp_lg_u32 s6, 0; s_cbranch_scc1
    w.extend([0xBE87_0380, S_ENDPGM]); // s_mov_b32 s7, 0; s_endpgm
    let l =
        lower_halves(&w).expect("a uniform global load lowers with lanes active in both halves");
    assert_eq!(l.wave_size, Some(32), "the console wave ran as two halves");
    assert_eq!(
        l.shape.barriers, 0,
        "no cross-half exchange was emitted for the observable first-lane read"
    );
}

/// The last destination word of a `global_load_dwordx2`, `x3` or `x4` at an
/// alike address is one value in every lane too: its first-lane read in a
/// divergent block lowers without a cross-half exchange.
#[test]
fn a_uniform_global_loads_last_word_of_x2_x3_x4_needs_no_exchange() {
    // op, the read of the last destination word (s6, v6+N-1).
    for (op, last) in [
        (0x0D, 0x7E0C_0507), // global_load_dwordx2 v[6:7]; v_readfirstlane_b32 s6, v7
        (0x0F, 0x7E0C_0508), // global_load_dwordx3 v[6:8]; v_readfirstlane_b32 s6, v8
        (0x0E, 0x7E0C_0509), // global_load_dwordx4 v[6:9]; v_readfirstlane_b32 s6, v9
    ] {
        let mut w = vec![0x7E02_02FF, 0x0000_1000]; // v_mov_b32 v1, 0x1000
        w.extend(global(op, 6, 1, 0, 0));
        divergent_read(&mut w, last, 0xBF07_8006); // s_cmp_lg_u32 s6, 0
        let l = lower_halves(&w).unwrap_or_else(|e| panic!("global_load_dwordx op {op:#x}: {e}"));
        assert_eq!(l.wave_size, Some(32));
        assert_eq!(
            l.shape.barriers, 0,
            "no cross-half exchange for the last word of op {op:#x}"
        );
    }
}

/// Equal addresses do not prove equal load results where memory can be
/// written: a `global_store_dword` in the same shader (here at the loaded
/// address) means the console wave's two host waves can reach the load on
/// opposite sides of another lane's store, so the load is not lane-uniform
/// and its first-lane read in a divergent block keeps the refusal.
#[test]
fn an_aliasing_global_store_keeps_the_first_lane_refusal() {
    let mut w = vec![0x7E02_02FF, 0x0000_1000]; // v_mov_b32 v1, 0x1000
    w.extend([0x7E04_02FF, 0x0000_0777]); // v_mov_b32 v2, 0x777
    w.extend(global_store(2, 1, 0, 0)); // global_store_dword v2, v1, s[0:1], 0
    w.extend(global(0x0C, 3, 1, 0, 0)); // global_load_dword v3, v1, s[0:1], 0
    divergent_read(&mut w, 0x7E0E_0503, 0xBF07_8007); // v_readfirstlane_b32 s7, v3; s_cmp_lg_u32 s7, 0
    assert!(
        refused_in_halves(&w),
        "an aliasing global store keeps the refusal"
    );
}

/// A root SRV and a root UAV can bind the same title-memory address, so a
/// `global_store_dword` through the root UAV can alias a `global_load_dword`
/// through the root SRV: the load is not lane-uniform, and its first-lane
/// read in a divergent block keeps the refusal.
#[test]
fn a_root_srv_load_with_a_root_uav_store_keeps_the_first_lane_refusal() {
    // s[0:1]: the root SRV's address; s[2:3]: the root UAV's address.
    let srv = root(RootType::Descriptor, 0, 0);
    let uav = RootParam {
        sgprs: 2,
        ..root(RootType::Descriptor, 2, 1)
    };
    let used = vec![
        UsedResource {
            kind: ResourceType::Srv,
            root: RootType::Descriptor,
            table_offset_dwords: 0,
            lower_bound: 0,
            stride: 0,
            dwords: 4,
            user_data: 0,
            slot: 0,
        },
        UsedResource {
            kind: ResourceType::Uav,
            root: RootType::Descriptor,
            table_offset_dwords: 0,
            lower_bound: 0,
            stride: 0,
            dwords: 4,
            user_data: 2,
            slot: 1,
        },
    ];
    let mut w = vec![
        0x7E02_02FF,
        0x0000_1000, // v_mov_b32 v1, 0x1000
        0x7E04_02FF,
        0x0000_0777, // v_mov_b32 v2, 0x777
    ];
    w.extend(global_store(2, 1, 2, 0)); // global_store_dword v2, v1, s[2:3], 0 (the root UAV)
    w.extend(global(0x0C, 3, 1, 0, 0)); // global_load_dword v3, v1, s[0:1], 0 (the root SRV)
    divergent_read(&mut w, 0x7E0E_0503, 0xBF07_8007); // v_readfirstlane_b32 s7, v3; s_cmp_lg_u32 s7, 0
    let first_across = super::halves::FIRST_ACROSS;
    assert!(
        matches!(lower_halves_with(&w, vec![srv, uav], used), Err(crate::ShaderError::Unsupported { what, .. }) if what == first_across),
        "a root SRV a root UAV store can alias keeps the refusal"
    );
}

/// A `buffer_atomic_add` writes title memory a `global_load_dword` can read,
/// so the load is not lane-uniform: its first-lane read in a divergent block
/// keeps the refusal.
#[test]
fn a_buffer_atomic_with_a_global_load_keeps_the_first_lane_refusal() {
    // s[0:3]: root constants; s[4:5]: the root table the buffer's descriptor
    // is loaded from into s[8:11].
    let constants = RootParam {
        sgprs: 4,
        ..root(RootType::Constants, 0, 0)
    };
    let buffer = UsedResource {
        kind: ResourceType::Uav,
        root: RootType::Table,
        table_offset_dwords: 0,
        lower_bound: 0,
        stride: 0,
        dwords: 4,
        user_data: 4,
        slot: 1,
    };
    let mut w = vec![
        0x7E02_02FF,
        0x0000_1000, // v_mov_b32 v1, 0x1000
        0xBE85_0380, // s_mov_b32 s5, 0
        0xF408_0202, // s_load_dwordx4 s[8:11], s[4:5], 0
        0xFA00_0000,
        0xBF8C_0000, // s_waitcnt lgkmcnt(0)
        0xE0C8_1000, // buffer_atomic_add v2, v1, s[8:11], 0 offen
        0x8002_0201,
    ];
    w.extend(global(0x0C, 3, 1, 0, 0)); // global_load_dword v3, v1, s[0:1], 0
    divergent_read(&mut w, 0x7E0E_0503, 0xBF07_8007); // v_readfirstlane_b32 s7, v3; s_cmp_lg_u32 s7, 0
    let first_across = super::halves::FIRST_ACROSS;
    assert!(
        matches!(lower_halves_with(&w, vec![constants, root(RootType::Table, 4, 1)], vec![buffer]), Err(crate::ShaderError::Unsupported { what, .. }) if what == first_across),
        "a global load a buffer atomic can alias keeps the refusal"
    );
}

/// A `buffer_atomic_add` at the same UAV byte offset a `buffer_load_dword`
/// reads can be observed at different stages by the console wave's two host
/// halves — one half can reach the load on the other side of the atomic —
/// so the load's result is not the same in every lane of the console wave:
/// its first-lane read in a divergent block keeps the refusal. The load's
/// descriptor is a root one held in SGPRs (s[0:1] the root descriptor's
/// address, s2/s3 constants), so every register it reads is positively
/// proven uniform and only the atomic's aliasing keeps the refusal; without
/// the atomic the same load translates with no cross-half exchange.
#[test]
fn a_buffer_load_aliased_by_a_buffer_atomic_keeps_the_first_lane_refusal() {
    // s[0:1]: the root UAV descriptor's 64-bit address; s2/s3: the rest of
    // the buffer descriptor the load and atomic take as srsrc s[0:3].
    let uav = root(RootType::Descriptor, 0, 0);
    let buffer = UsedResource {
        kind: ResourceType::Uav,
        root: RootType::Descriptor,
        table_offset_dwords: 0,
        lower_bound: 0,
        stride: 0,
        dwords: 4,
        user_data: 0,
        slot: 0,
    };
    let head = [
        0xBE82_0380, // s_mov_b32 s2, 0 (srsrc dword 2)
        0xBE83_0380, // s_mov_b32 s3, 0 (srsrc dword 3)
        0x7E02_02FF,
        0x0000_0000, // v_mov_b32 v1, 0 (the load's and the atomic's byte offset)
        0xE030_1000,
        0x8000_0201, // buffer_load_dword v2, v1, s[0:3], 0 offen
    ];
    // With the atomic at the same UAV dword: the load can be observed on
    // opposite sides of it in the two host halves, so it is not lane-uniform.
    let mut w = Vec::from(head);
    w.extend([
        0xBF8C_3F70, // s_waitcnt vmcnt(0)
        0x7E06_02FF,
        0x0000_0001, // v_mov_b32 v3, 1 (non-zero addend so the atomic modifies the dword)
        0xE0C8_1000,
        0x8000_0301, // buffer_atomic_add v3, v1, s[0:3], 0 offen (the same UAV dword)
    ]);
    divergent_read(&mut w, 0x7E0C_0502, 0xBF07_8006); // v_readfirstlane_b32 s6, v2; s_cmp_lg_u32 s6, 0
    let first_across = super::halves::FIRST_ACROSS;
    assert!(
        matches!(lower_halves_with(&w, vec![uav], vec![buffer]), Err(crate::ShaderError::Unsupported { what, .. }) if what == first_across),
        "a buffer load a buffer atomic at the same UAV offset can alias keeps the refusal"
    );
    // Without the atomic: no write the shader performs can alias the load,
    // and every register it reads is proven uniform (the root descriptor's
    // SGPRs, the constants, the constant VGPR offset), so the load is
    // lane-uniform and its first-lane read needs no cross-half exchange.
    let mut w = Vec::from(head);
    divergent_read(&mut w, 0x7E0C_0502, 0xBF07_8006); // v_readfirstlane_b32 s6, v2; s_cmp_lg_u32 s6, 0
    let l = lower_halves_with(&w, vec![uav], vec![buffer])
        .expect("a buffer load with a proven root descriptor and no aliasing atomic lowers in halves");
    assert_eq!(l.wave_size, Some(32), "the console wave ran as two halves");
    assert_eq!(
        l.shape.barriers, 0,
        "no cross-half exchange was emitted for the proven-descriptor load"
    );
}

/// A `buffer_load_dword`'s scalar offset (`soffset`) is an `s_load_dword`'s
/// result, which is not whitelisted ([`SCALAR_ALOP`]) and so has no positive
/// proof of uniformity ([`same_s`]): the two host halves can read different
/// stages of the memory the `s_load_dword` read, so the buffer offset — and
/// with it the loaded dword — can differ between them. The first-lane read
/// of the load's result in a divergent block keeps the refusal. The load's
/// descriptor is a root one held in SGPRs (s[0:1] the root descriptor's
/// address, s2/s3 constants), so only the unproven `soffset` keeps the
/// refusal; with a proven root-constant `soffset` (s4) the same load
/// translates with no cross-half exchange.
#[test]
fn a_buffer_load_with_an_s_load_soffset_keeps_the_first_lane_refusal() {
    // s[0:1]: the root SRV descriptor's 64-bit address (s2/s3 the rest of
    // srsrc s[0:3]); s[4:7]: root constants (s[4:5] the s_load's base, s4 a
    // proven soffset).
    let srv = root(RootType::Descriptor, 0, 0);
    let constants = RootParam {
        sgprs: 4,
        ..root(RootType::Constants, 4, 1)
    };
    let buffer = UsedResource {
        kind: ResourceType::Srv,
        root: RootType::Descriptor,
        table_offset_dwords: 0,
        lower_bound: 0,
        stride: 4,
        dwords: 4,
        user_data: 0,
        slot: 0,
    };
    let head = [
        0xBE82_0380, // s_mov_b32 s2, 0 (srsrc dword 2)
        0xBE83_0380, // s_mov_b32 s3, 0 (srsrc dword 3)
        0x7E02_02FF,
        0x0000_0000, // v_mov_b32 v1, 0 (the load's index)
    ];
    // With the `s_load_dword`-derived soffset s8: unproven, so the load is
    // not lane-uniform.
    let mut w = Vec::from(head);
    w.extend([
        0xF400_0202,
        0xFA00_0040, // s_load_dword s8, s[4:5], 0x40 (an unproven scalar read)
        0xBF8C_0000, // s_waitcnt lgkmcnt(0)
        0xE030_1000,
        0x0800_0201, // buffer_load_dword v2, v1, s[0:3], s8 offen
    ]);
    divergent_read(&mut w, 0x7E0C_0502, 0xBF07_8006); // v_readfirstlane_b32 s6, v2; s_cmp_lg_u32 s6, 0
    let first_across = super::halves::FIRST_ACROSS;
    assert!(
        matches!(lower_halves_with(&w, vec![srv, constants], vec![buffer]), Err(crate::ShaderError::Unsupported { what, .. }) if what == first_across),
        "a buffer load with an s_load_dword-derived soffset keeps the refusal"
    );
    // With the root-constant soffset s4 (in [`same_s`] from entry): proven,
    // so the load is lane-uniform and its first-lane read needs no
    // cross-half exchange.
    let mut w = Vec::from(head);
    w.extend([
        0xE030_1000,
        0x0400_0201, // buffer_load_dword v2, v1, s[0:3], s4 offen
    ]);
    divergent_read(&mut w, 0x7E0C_0502, 0xBF07_8006); // v_readfirstlane_b32 s6, v2; s_cmp_lg_u32 s6, 0
    let l = lower_halves_with(&w, vec![srv, constants], vec![buffer])
        .expect("a buffer load with a proven root descriptor and root-constant soffset lowers in halves");
    assert_eq!(l.wave_size, Some(32), "the console wave ran as two halves");
    assert_eq!(
        l.shape.barriers, 0,
        "no cross-half exchange was emitted for the proven-soffset load"
    );
}

/// An overlapping `buffer_load_dword v1, v1, s[0:3], 0 offen` reads its
/// destination as the address (`vaddr`) too: with a lane-varying `v1` (the
/// thread id) the loaded dword differs between lanes, so the result is not
/// the same in every lane of the console wave and its first-lane read in a
/// divergent block keeps the refusal. The descriptor is a root one held in
/// SGPRs (proven uniform); with a uniform `v1` the overlapping load
/// translates with no cross-half exchange.
#[test]
fn an_overlapping_buffer_load_with_a_lane_varying_vaddr_keeps_the_first_lane_refusal() {
    // s[0:1]: the root SRV descriptor's 64-bit address; s2/s3: the rest of
    // the buffer descriptor the load takes as srsrc s[0:3].
    let srv = root(RootType::Descriptor, 0, 0);
    let buffer = UsedResource {
        kind: ResourceType::Srv,
        root: RootType::Descriptor,
        table_offset_dwords: 0,
        lower_bound: 0,
        stride: 4,
        dwords: 4,
        user_data: 0,
        slot: 0,
    };
    let head = |v1: &[u32]| {
        let mut w = vec![
            0xBE82_0380, // s_mov_b32 s2, 0 (srsrc dword 2)
            0xBE83_0380, // s_mov_b32 s3, 0 (srsrc dword 3)
        ];
        w.extend_from_slice(v1);
        w.extend([
            0xE030_1000,
            0x8000_0101, // buffer_load_dword v1, v1, s[0:3], 0 offen (dst == vaddr)
        ]);
        w
    };
    // The address/destination v1 is the thread id: lane-varying, so the
    // load is not lane-uniform.
    let mut w = head(&[0x7E02_0300]); // v_mov_b32 v1, v0
    divergent_read(&mut w, 0x7E0C_0501, 0xBF07_8006); // v_readfirstlane_b32 s6, v1; s_cmp_lg_u32 s6, 0
    let first_across = super::halves::FIRST_ACROSS;
    assert!(
        matches!(lower_halves_with(&w, vec![srv], vec![buffer]), Err(crate::ShaderError::Unsupported { what, .. }) if what == first_across),
        "an overlapping buffer load with a lane-varying vaddr keeps the refusal"
    );
    // The same overlapping load with a uniform v1: the address is proven
    // uniform, so the load is lane-uniform and needs no cross-half exchange.
    let mut w = head(&[0x7E02_02FF, 0x0000_0000]); // v_mov_b32 v1, 0
    divergent_read(&mut w, 0x7E0C_0501, 0xBF07_8006); // v_readfirstlane_b32 s6, v1; s_cmp_lg_u32 s6, 0
    let l = lower_halves_with(&w, vec![srv], vec![buffer])
        .expect("an overlapping buffer load with a uniform vaddr lowers in halves");
    assert_eq!(l.wave_size, Some(32), "the console wave ran as two halves");
    assert_eq!(
        l.shape.barriers, 0,
        "no cross-half exchange was emitted for the uniform overlapping load"
    );
}

/// A `ds_read_u16_d16` writes only the low half of its destination VGPR and
/// keeps the old high half as live data: with a lane-varying old
/// destination (the thread id) the result is lane-varying too, so it is not
/// the same in every lane of the console wave and its first-lane read in a
/// divergent block keeps the refusal. With a uniform old destination the
/// load is lane-uniform and translates with no cross-half exchange. (The
/// group is two console waves, so LDS keeps its barriers in halves and only
/// the value analysis can refuse.)
#[test]
fn a_ds_read_u16_d16_into_a_lane_varying_destination_keeps_the_first_lane_refusal() {
    // 512 bytes of LDS in a group of 128 threads.
    let lower = |w: &[u32]| {
        lower_halves_io(
            w,
            vec![RootParam {
                sgprs: 4,
                ..root(RootType::Constants, 0, 0)
            }],
            vec![],
            1 << 15 | 4 << 1,
            [128, 1, 1],
        )
    };
    let head = |v1: &[u32]| {
        let mut w = Vec::from(v1);
        w.extend([
            0x7E04_02FF,
            0x0000_1000, // v_mov_b32 v2, 0x1000 (uniform out-of-bounds LDS address)
            0xDA98_0000,
            0x0100_0002, // ds_read_u16_d16 v1, v2 offset:0 (keeps the old high half of v1)
        ]);
        w
    };
    // The old destination v1 holds tid << 16: lane-varying in its preserved
    // high 16 bits, so the _d16 load's result is lane-varying.
    let mut w = head(&[0x3402_0090]); // v_lshlrev_b32 v1, 16, v0
    divergent_read(&mut w, 0x7E0C_0501, 0xBF07_8006); // v_readfirstlane_b32 s6, v1; s_cmp_lg_u32 s6, 0
    let first_across = super::halves::FIRST_ACROSS;
    assert!(
        matches!(lower(&w), Err(crate::ShaderError::Unsupported { what, .. }) if what == first_across),
        "a ds_read_u16_d16 into a lane-varying destination keeps the refusal"
    );
    // The same load with a uniform old destination: proven uniform, so the
    // load is lane-uniform and needs no cross-half exchange.
    let mut w = head(&[0x7E02_02FF, 0x0000_0000]); // v_mov_b32 v1, 0
    divergent_read(&mut w, 0x7E0C_0501, 0xBF07_8006); // v_readfirstlane_b32 s6, v1; s_cmp_lg_u32 s6, 0
    let l = lower(&w).expect("a ds_read_u16_d16 into a uniform destination lowers in halves");
    assert_eq!(l.wave_size, Some(32), "the console wave ran as two halves");
    assert_eq!(
        l.shape.barriers, 0,
        "no cross-half exchange was emitted for the uniform-destination _d16 read"
    );
}

/// A `ds_read_b32` at a uniform address stays lane-uniform even when the
/// shader writes LDS, provided every LDS write before and after it is
/// separated by an `s_barrier`. Without a barrier before (RAW) or after (WAR)
/// the read, it stays lane-varying and the divergent `v_readfirstlane_b32`
/// keeps the cross-half refusal.
#[test]
fn a_barrier_isolated_ds_read_stays_uniform_with_lds_writes_present() {
    let lower = |words: &[u32]| {
        lower_halves_io(
            words,
            vec![RootParam {
                sgprs: 4,
                ..root(RootType::Constants, 0, 0)
            }],
            vec![],
            1 << 15 | 4 << 1,
            [128, 1, 1],
        )
    };
    const S_BARRIER: u32 = 0xBF8A_0000;
    const DS_WRITE_B32_V2_V0: [u32; 2] = [0xD834_0000, 0x0000_0002];
    const DS_READ_B32_V1_V2: [u32; 2] = [0xD8D8_0000, 0x0100_0002];
    let preamble = [
        0x7E04_02FF,
        0x0000_0000, // v_mov_b32 v2, 0 (uniform LDS address)
    ];
    let first_across = super::halves::FIRST_ACROSS;

    // 1. ds_write -> s_barrier -> ds_read (no write after): uniform.
    let mut w = Vec::from(preamble);
    w.extend(DS_WRITE_B32_V2_V0);
    w.push(S_BARRIER);
    w.extend(DS_READ_B32_V1_V2);
    divergent_read(&mut w, 0x7E0C_0501, 0xBF07_8006); // v_readfirstlane_b32 s6, v1; s_cmp_lg_u32 s6, 0
    let l = lower(&w).expect("barrier before ds_read makes uniform LDS read lower in halves");
    assert_eq!(l.wave_size, Some(32));
    assert_eq!(l.shape.barriers, 1);

    // 2. ds_write -> s_barrier -> ds_read -> s_barrier -> ds_write: uniform.
    let mut w = Vec::from(preamble);
    w.extend(DS_WRITE_B32_V2_V0);
    w.push(S_BARRIER);
    w.extend(DS_READ_B32_V1_V2);
    w.push(S_BARRIER);
    w.extend(DS_WRITE_B32_V2_V0);
    divergent_read(&mut w, 0x7E0C_0501, 0xBF07_8006);
    let l = lower(&w).expect("barriers on both sides make uniform LDS read lower in halves");
    assert_eq!(l.wave_size, Some(32));
    assert_eq!(l.shape.barriers, 2);

    // 3. ds_write -> ds_read (RAW hazard, no barrier): refused.
    let mut w = Vec::from(preamble);
    w.extend(DS_WRITE_B32_V2_V0);
    w.extend(DS_READ_B32_V1_V2);
    divergent_read(&mut w, 0x7E0C_0501, 0xBF07_8006);
    assert!(
        matches!(lower(&w), Err(crate::ShaderError::Unsupported { what, .. }) if what == first_across),
        "RAW hazard without s_barrier keeps the cross-half refusal"
    );

    // 4. ds_write -> s_barrier -> ds_read -> ds_write (WAR hazard, no barrier after): refused.
    let mut w = Vec::from(preamble);
    w.extend(DS_WRITE_B32_V2_V0);
    w.push(S_BARRIER);
    w.extend(DS_READ_B32_V1_V2);
    w.extend(DS_WRITE_B32_V2_V0);
    divergent_read(&mut w, 0x7E0C_0501, 0xBF07_8006);
    assert!(
        matches!(lower(&w), Err(crate::ShaderError::Unsupported { what, .. }) if what == first_across),
        "WAR hazard without s_barrier after ds_read keeps the cross-half refusal"
    );
}

/// A `global_load_dword` from the code's own table stays lane-uniform even
/// where the shader performs stores that could alias other loads (here a
/// `global_store_dword`): the first-lane read of it in a divergent block
/// needs no cross-half exchange.
#[test]
fn a_global_load_from_the_code_table_stays_uniform_with_a_store_present() {
    let mut w = vec![
        0xBE80_1F00, // s_getpc_b64 s[0:1] (the code's table base)
        0x7E02_02FF,
        0x0000_0000, // v_mov_b32 v1, 0 (the code table sits right after the code)
        0x7E04_02FF,
        0x0000_0777, // v_mov_b32 v2, 0x777
    ];
    w.extend(global_store(2, 1, 2, 0)); // global_store_dword v2, v1, s[2:3], 0
    w.extend(global(0x0C, 3, 1, 0, 0x40)); // global_load_dword v3, v1, s[0:1], 0x40 (the code table)
    divergent_read(&mut w, 0x7E0E_0503, 0xBF07_8007); // v_readfirstlane_b32 s7, v3; s_cmp_lg_u32 s7, 0
    let l = lower_halves(&w).expect("a load from the code's table stays lane-uniform with a store present");
    assert_eq!(l.wave_size, Some(32));
    assert_eq!(
        l.shape.barriers, 0,
        "no cross-half exchange was emitted for the code table's load"
    );
}

/// Where the shader also stores (here a `global_store_dword`), a
/// `buffer_load_dword` result is not lane-uniform: the console wave's two
/// host waves can observe the store to the loaded buffer at different times
/// and load different dwords. As the VGPR offset of a code-table
/// `global_load_dword` it would let each half read a different code entry
/// while a `v_readfirstlane_b32` of the result skips the cross-half
/// exchange: the first-lane read in a divergent block keeps its refusal.
#[test]
fn a_buffer_load_offset_to_code_with_a_store_keeps_the_first_lane_refusal() {
    // s[0:3]: root constants (s[2:3] the store's address); s[4:5]: the root
    // table the buffer's descriptor is loaded from into s[8:11].
    let constants = RootParam {
        sgprs: 4,
        ..root(RootType::Constants, 0, 0)
    };
    let buffer = UsedResource {
        kind: ResourceType::Srv,
        root: RootType::Table,
        table_offset_dwords: 0,
        lower_bound: 0,
        stride: 4,
        dwords: 4,
        user_data: 4,
        slot: 1,
    };
    let mut w = vec![
        0xBE80_1F00, // s_getpc_b64 s[0:1] (the code's table base)
        0x7E02_02FF,
        0x0000_0000, // v_mov_b32 v1, 0 (the buffer load's address)
        0xBE85_0380, // s_mov_b32 s5, 0
        0xF408_0202,
        0xFA00_0000, // s_load_dwordx4 s[8:11], s[4:5], 0
        0xBF8C_0000, // s_waitcnt lgkmcnt(0)
        0xE030_1000,
        0x8002_0201, // buffer_load_dword v2, v1, s[8:11], 0 offen
        0x7E06_02FF,
        0x0000_0777, // v_mov_b32 v3, 0x777
    ];
    w.extend(global_store(3, 1, 2, 0)); // global_store_dword v3, v1, s[2:3], 0
    w.extend(global(0x0C, 5, 2, 0, 0x40)); // global_load_dword v5, v2, s[0:1], 0x40 (the code table)
    divergent_read(&mut w, 0x7E12_0505, 0xBF07_8009); // v_readfirstlane_b32 s9, v5; s_cmp_lg_u32 s9, 0
    w.extend([0x1111_1111, 0x2222_2222]); // the code's constant table, past the instructions
    let first_across = super::halves::FIRST_ACROSS;
    assert!(
        matches!(lower_halves_with(&w, vec![constants, root(RootType::Table, 4, 1)], vec![buffer]), Err(crate::ShaderError::Unsupported { what, .. }) if what == first_across),
        "a buffer-load offset into the code table with a store present keeps the refusal"
    );
}

/// A phi selecting between two constants counts only where every lane of
/// both halves reaches its block together. Here the branch choosing between
/// the constants tests a `buffer_load_dword` result, which is not
/// lane-uniform where the shader stores (here a `global_store_dword` is
/// present) — the two host halves can observe a store to the loaded buffer
/// at different times and take different sides — so the phi can pick a
/// different constant in each half. As the VGPR offset of a code-table
/// `global_load_dword`, each half would then read a different code entry
/// while a first-lane read of the load's result skips the cross-half
/// exchange: the first-lane read in a divergent block keeps its refusal.
#[test]
fn a_phi_over_a_load_controlled_branch_keeps_the_first_lane_refusal() {
    // s[0:3]: root constants (s[2:3] the store's address); s[4:5]: the root
    // table the buffer's descriptor is loaded from into s[8:11].
    let constants = RootParam {
        sgprs: 4,
        ..root(RootType::Constants, 0, 0)
    };
    let buffer = UsedResource {
        kind: ResourceType::Srv,
        root: RootType::Table,
        table_offset_dwords: 0,
        lower_bound: 0,
        stride: 4,
        dwords: 4,
        user_data: 4,
        slot: 1,
    };
    let mut w = vec![
        0xBE80_1F00, // s_getpc_b64 s[0:1] (the code's table base)
        0x7E02_02FF,
        0x0000_0000, // v_mov_b32 v1, 0 (the buffer load's address)
        0x7E0C_02FF,
        0x0000_0777, // v_mov_b32 v6, 0x777 (the store's data)
        0xBE85_0380, // s_mov_b32 s5, 0
        0xF408_0202,
        0xFA00_0000, // s_load_dwordx4 s[8:11], s[4:5], 0
        0xBF8C_0000, // s_waitcnt lgkmcnt(0)
        0xE030_1000,
        0x8002_0201, // buffer_load_dword v2, v1, s[8:11], 0 offen
        0x7E06_02FF,
        0x0000_0011, // v_mov_b32 v3, 0x11 (the phi's first constant)
        0x7D84_0480, // v_cmp_eq_u32 vcc_lo, 0, v2 (the load controls the branch)
        0xBF86_0002, // s_cbranch_vccz -> the merge (over the second constant)
        0x7E06_02FF,
        0x0000_0022, // v_mov_b32 v3, 0x22 (the phi's second constant)
    ];
    w.extend(global_store(6, 1, 2, 0)); // global_store_dword v6, v1, s[2:3], 0
    w.extend(global(0x0C, 5, 3, 0, 0x40)); // global_load_dword v5, v3 (the phi), s[0:1], 0x40 (the code table)
    divergent_read(&mut w, 0x7E12_0505, 0xBF07_8009); // v_readfirstlane_b32 s9, v5; s_cmp_lg_u32 s9, 0
    w.extend([0x1111_1111, 0x2222_2222]); // the code's constant table, past the instructions
    let first_across = super::halves::FIRST_ACROSS;
    assert!(
        matches!(lower_halves_with(&w, vec![constants, root(RootType::Table, 4, 1)], vec![buffer]), Err(crate::ShaderError::Unsupported { what, .. }) if what == first_across),
        "a phi over a load-controlled branch with a store present keeps the refusal"
    );
}

/// A `buffer_load_dword` comparison is not alike where the shader stores
/// (here a `global_store_dword` is present): the two host halves can
/// observe a store to the loaded buffer at different times and compute
/// different carries. If it narrows EXEC around a code-table
/// `global_load_dword`, one half can take the load while the other keeps
/// the old value, so the guarded load must not enter `same` and a
/// first-lane read after EXEC is restored keeps its refusal.
#[test]
fn a_code_load_under_a_buffer_load_guard_keeps_the_first_lane_refusal() {
    let constants = RootParam {
        sgprs: 4,
        ..root(RootType::Constants, 0, 0)
    };
    let buffer = UsedResource {
        kind: ResourceType::Srv,
        root: RootType::Table,
        table_offset_dwords: 0,
        lower_bound: 0,
        stride: 4,
        dwords: 4,
        user_data: 4,
        slot: 1,
    };
    let mut w = vec![
        0xBE80_1F00, // s_getpc_b64 s[0:1] (the code's table base)
        0x7E02_02FF,
        0x0000_0000, // v_mov_b32 v1, 0 (the buffer load's and code load's offset)
        0x7E0A_02FF,
        0x0000_0000, // v_mov_b32 v5, 0 (pure old value of destination)
        0x7E0C_02FF,
        0x0000_0777, // v_mov_b32 v6, 0x777 (the store's data)
        0xBE85_0380, // s_mov_b32 s5, 0
        0xF408_0202,
        0xFA00_0000, // s_load_dwordx4 s[8:11], s[4:5], 0
        0xBF8C_0000, // s_waitcnt lgkmcnt(0)
        0xE030_1000,
        0x8002_0201, // buffer_load_dword v2, v1, s[8:11], 0 offen
        0x7D84_0480, // v_cmp_eq_u32 vcc, 0, v2 (not lane-uniform with a store present)
        0xBE90_246A, // s_and_saveexec_b64 s[16:17], vcc
    ];
    w.extend(global_store(6, 1, 2, 0)); // global_store_dword v6, v1, s[2:3], 0
    w.extend(global(0x0C, 5, 1, 0, 0x40)); // guarded global_load_dword v5, v1, s[0:1], 0x40
    w.push(0xBEFE_0410); // s_mov_b64 exec, s[16:17]
    divergent_read(&mut w, 0x7E12_0505, 0xBF07_8009); // v_readfirstlane_b32 s9, v5; s_cmp_lg_u32 s9, 0
    w.extend([0x1111_1111, 0x2222_2222]); // the code's constant table, past the instructions
    let first_across = super::halves::FIRST_ACROSS;
    assert!(
        matches!(lower_halves_with(&w, vec![constants, root(RootType::Table, 4, 1)], vec![buffer]), Err(crate::ShaderError::Unsupported { what, .. }) if what == first_across),
        "a code-table load under a buffer-load guard with a store present keeps the refusal"
    );
}

/// `v_dot4c_i32_i8` accumulates into its destination register, so an old
/// destination loaded from a buffer must be alike for the result to be:
/// where the shader stores (here a `global_store_dword` is present) the
/// `buffer_load_dword` result is not lane-uniform, so the dot4c's result is
/// not either, and the code-table load at its offset is not lane-uniform.
#[test]
fn a_dot4c_accumulator_from_a_buffer_load_keeps_the_first_lane_refusal() {
    let constants = RootParam {
        sgprs: 4,
        ..root(RootType::Constants, 0, 0)
    };
    let buffer = UsedResource {
        kind: ResourceType::Srv,
        root: RootType::Table,
        table_offset_dwords: 0,
        lower_bound: 0,
        stride: 4,
        dwords: 4,
        user_data: 4,
        slot: 1,
    };
    let mut w = vec![
        0xBE80_1F00, // s_getpc_b64 s[0:1] (the code's table base)
        0x7E02_02FF,
        0x0000_0000, // v_mov_b32 v1, 0
        0x7E06_02FF,
        0x0000_0000, // v_mov_b32 v3, 0 (pure src0)
        0x7E08_02FF,
        0x0000_0000, // v_mov_b32 v4, 0 (pure src1)
        0x7E0C_02FF,
        0x0000_0777, // v_mov_b32 v6, 0x777 (the store's data)
        0xBE85_0380, // s_mov_b32 s5, 0
        0xF408_0202,
        0xFA00_0000, // s_load_dwordx4 s[8:11], s[4:5], 0
        0xBF8C_0000, // s_waitcnt lgkmcnt(0)
        0xE030_1000,
        0x8002_0201, // buffer_load_dword v2, v1, s[8:11], 0 offen (not lane-uniform with a store present)
        0x1A04_0903, // v_dot4c_i32_i8 v2, v3, v4 (reads old v2 as accumulator)
    ];
    w.extend(global_store(6, 1, 2, 0)); // global_store_dword v6, v1, s[2:3], 0
    w.extend(global(0x0C, 5, 2, 0, 0x40)); // global_load_dword v5, v2, s[0:1], 0x40
    divergent_read(&mut w, 0x7E12_0505, 0xBF07_8009); // v_readfirstlane_b32 s9, v5; s_cmp_lg_u32 s9, 0
    w.extend([0x1111_1111, 0x2222_2222]);
    let first_across = super::halves::FIRST_ACROSS;
    assert!(
        matches!(lower_halves_with(&w, vec![constants, root(RootType::Table, 4, 1)], vec![buffer]), Err(crate::ShaderError::Unsupported { what, .. }) if what == first_across),
        "a v_dot4c_i32_i8 with a buffer-loaded accumulator must not enter pure"
    );
}

/// A 64-bit VALU op reads both dwords of its scalar operand, so a pure one
/// needs both in `same_s`: here `v_lshlrev_b64`'s high word is an
/// `s_load_dword` result (unproven), which the operand's low word alone
/// would not check. As the VGPR offset of a code-table `global_load_dword`
/// with a store present the shift's result must not count as pure: the
/// first-lane read in a divergent block keeps its refusal.
#[test]
fn a_64bit_shift_with_an_unproven_high_scalar_keeps_the_first_lane_refusal() {
    // s[0:7]: root constants (s[0:1] overwritten by s_getpc with the code's
    // table base, s[2:3] the store's address, s6 the shift's low word).
    let constants = RootParam {
        sgprs: 8,
        ..root(RootType::Constants, 0, 0)
    };
    let mut w = vec![
        0xBE80_1F00, // s_getpc_b64 s[0:1] (the code's table base)
        0xF400_01C1,
        0xFA00_0040, // s_load_dword s7, s[2:3], 0x40 (an unproven scalar read)
        0xBF8C_0000, // s_waitcnt lgkmcnt(0)
        0xD6FF_0004,
        0x0000_0C86, // v_lshlrev_b64 v[4:5], 6, s[6:7] (the high word unproven)
        0x7E02_02FF,
        0x0000_0000, // v_mov_b32 v1, 0 (the store's address)
        0x7E0C_02FF,
        0x0000_0777, // v_mov_b32 v6, 0x777 (the store's data)
    ];
    w.extend(global_store(6, 1, 2, 0)); // global_store_dword v6, v1, s[2:3], 0
    w.extend(global(0x0C, 5, 4, 0, 0x40)); // global_load_dword v5, v4, s[0:1], 0x40 (the code table)
    divergent_read(&mut w, 0x7E12_0505, 0xBF07_8009); // v_readfirstlane_b32 s9, v5; s_cmp_lg_u32 s9, 0
    w.extend([0x1111_1111, 0x2222_2222]); // the code's constant table, past the instructions
    let first_across = super::halves::FIRST_ACROSS;
    assert!(
        matches!(lower_halves_with(&w, vec![constants], vec![]), Err(crate::ShaderError::Unsupported { what, .. }) if what == first_across),
        "a 64-bit shift with an unproven high scalar word keeps the refusal"
    );
}

/// An unproven scalar base word copied through `s_mov_b32` carries its data
/// in a new SSA value the copy does not prove uniform — an `s_load_dword`'s
/// result is a memory read, which is not whitelisted, and a waterfall loop's
/// reads are each lane's own — so the load at that base is not lane-uniform
/// and its first-lane read in a divergent block keeps the refusal.
#[test]
fn a_copied_unproven_base_word_keeps_the_first_lane_refusal() {
    let mut w = vec![
        0x7E02_02FF,
        0x0000_1000, // v_mov_b32 v1, 0x1000
        0xF400_0080, // s_load_dword s2, s[0:1], 0x40 (an unproven scalar read)
        0xFA00_0040,
        0xBE84_0302, // s_mov_b32 s4, s2 (the copy)
        0xBE85_0380, // s_mov_b32 s5, 0
    ];
    w.extend(global(0x0C, 2, 1, 4, 0)); // global_load_dword v2, v1, s[4:5], 0
    divergent_read(&mut w, 0x7E0C_0502, 0xBF07_8006); // v_readfirstlane_b32 s6, v2; s_cmp_lg_u32 s6, 0
    assert!(
        refused_in_halves(&w),
        "a copied unproven base word keeps the refusal"
    );
    // The same with the copy's source a first-lane read of the thread id:
    // in halves each host wave reads its own first lane, so the source holds
    // per-lane data, and the copy does not prove it uniform either.
    let mut w = vec![0x7E02_02FF, 0x0000_1000]; // v_mov_b32 v1, 0x1000
    w.push(0x7E04_0500); // v_readfirstlane_b32 s2, v0 (the host wave's first thread id)
    w.push(0xBE84_0302); // s_mov_b32 s4, s2 (the copy)
    w.push(0xBE85_0380); // s_mov_b32 s5, 0
    w.extend(global(0x0C, 2, 1, 4, 0)); // global_load_dword v2, v1, s[4:5], 0
    divergent_read(&mut w, 0x7E0C_0502, 0xBF07_8006); // v_readfirstlane_b32 s6, v2; s_cmp_lg_u32 s6, 0
    assert!(
        refused_in_halves(&w),
        "a copied first-lane read keeps the refusal"
    );
}

/// A `global_load_dword` under a lane-dependent guard runs only in some
/// lanes of each half, so its result is not one value per console wave: the
/// first-lane read of it in a divergent block keeps the refusal.
#[test]
fn a_lane_dependent_load_guard_keeps_the_first_lane_refusal() {
    let mut w = vec![0x7E02_02FF, 0x0000_1000]; // v_mov_b32 v1, 0x1000
    w.extend(WRITELANE_V4_V1_5);
    w.extend([CMPX_V0_EQ_0, skip(5)]);
    w.extend(global(0x0C, 2, 1, 0, 0)); // global_load_dword v2, v1, s[0:1], 0 (under the guard)
    w.extend([0x7E0C_0502, 0xBF07_8006, 0xBF85_0001]); // v_readfirstlane_b32 s6, v2; s_cmp_lg_u32 s6, 0; s_cbranch_scc1
    w.extend([0xBE87_0380, S_ENDPGM]); // s_mov_b32 s7, 0; s_endpgm
    assert!(
        refused_in_halves(&w),
        "a lane-dependent load guard keeps the refusal"
    );
}

/// A `global_load_dword` with a `glc` or `slc` cache flag can observe a
/// per-lane write through the cache, so it is not lane-uniform: its
/// first-lane read in a divergent block keeps the refusal.
#[test]
fn glc_and_slc_global_loads_keep_the_first_lane_refusal() {
    for (bit, what) in [(16, "glc"), (17, "slc")] {
        let mut w = vec![0x7E02_02FF, 0x0000_1000]; // v_mov_b32 v1, 0x1000
        let mut g = global(0x0C, 2, 1, 0, 0);
        g[0] |= 1 << bit;
        w.extend(g); // global_load_dword v2, v1, s[0:1], 0 with the cache flag
        divergent_read(&mut w, 0x7E0C_0502, 0xBF07_8006); // v_readfirstlane_b32 s6, v2; s_cmp_lg_u32 s6, 0
        assert!(
            refused_in_halves(&w),
            "a {what} global load keeps the refusal"
        );
    }
}

/// A `v_cndmask_b32` reads its select mask per lane (bit `lane_id` of the
/// 64-bit mask), so an inline mask of `1` — or a uniform root SGPR mask,
/// whose scalar value is one value but whose per-lane bit is not — selects
/// different sources for lane 0 than for lanes 1..63. As the VGPR offset of
/// a code-table `global_load_dword` with a store present (where the offset
/// must be pure), each half would read a different code entry while a
/// `v_readfirstlane_b32` of the load's result skips the cross-half
/// exchange: the first-lane read in a divergent block keeps its refusal.
#[test]
fn a_cndmask_with_a_per_lane_mask_bit_keeps_the_first_lane_refusal() {
    // s[0:1]: the code's table base (s_getpc); s[2:3]: root constants (the
    // store's address).
    let head = [
        0xBE80_1F00, // s_getpc_b64 s[0:1] (the code's table base)
        0x7E02_02FF,
        0x0000_0000, // v_mov_b32 v1, 0 (the store's address)
        0x7E04_02FF,
        0x0000_0777, // v_mov_b32 v2, 0x777 (the store's data)
    ];
    // v_cndmask_b32 v3, 0, 0x40, <mask> (VOP3 opcode 0x101: the third source
    // is the 64-bit mask).
    let cndmask = |mask: u32| [0xD501_0003, 128 | 192 << 9 | mask << 18];
    let tail = |w: &mut Vec<u32>, cnd: [u32; 2]| {
        w.extend(head);
        w.extend(cnd);
        w.extend(global_store(2, 1, 2, 0)); // global_store_dword v2, v1, s[2:3], 0
        w.extend(global(0x0C, 5, 3, 0, 0x40)); // global_load_dword v5, v3, s[0:1], 0x40 (the code table)
        divergent_read(w, 0x7E12_0505, 0xBF07_8009); // v_readfirstlane_b32 s9, v5; s_cmp_lg_u32 s9, 0
        w.extend([0x1111_1111, 0x2222_2222]); // the code's constant table
    };
    // An inline mask of 1: lane 0 takes 0x40, lanes 1..63 take 0.
    let mut w = Vec::new();
    tail(&mut w, cndmask(129)); // inline constant 1
    assert!(
        refused_in_halves(&w),
        "an inline mask of 1 keeps the first-lane refusal"
    );
    // A root SGPR mask s[2:3]: one scalar value, but bit `lane_id` per lane.
    let mut w = Vec::new();
    tail(&mut w, cndmask(2)); // s[2:3]
    assert!(
        refused_in_halves(&w),
        "a root SGPR select mask keeps the first-lane refusal"
    );
    // An all-ones mask selects the same source in every lane: no refusal.
    let mut w = Vec::new();
    tail(&mut w, cndmask(193)); // inline constant -1
    let l = lower_halves(&w)
        .expect("an all-ones select mask stays lane-uniform with a store present");
    assert_eq!(l.wave_size, Some(32), "the console wave ran as two halves");
    assert_eq!(
        l.shape.barriers, 0,
        "no cross-half exchange was emitted for the all-ones mask"
    );
    // A VOP2 `v_cndmask_b32` on VCC from a compare of alike values: the
    // mask's per-lane bit is the same in every lane, so no refusal either.
    let mut w = vec![
        0x7E0A_02FF,
        0x0000_0040, // v_mov_b32 v5, 0x40
        0x7E0C_02FF,
        0x0000_0000, // v_mov_b32 v6, 0
        0x7D84_0C80, // v_cmp_eq_u32 vcc, 0, v6 (alike operands)
        0x0206_0A80, // v_cndmask_b32 v3, 0, v5, vcc (VOP2: the implicit VCC)
    ];
    w.extend(global(0x0C, 2, 3, 0, 0)); // global_load_dword v2, v3, s[0:1], 0
    divergent_read(&mut w, 0x7E0C_0502, 0xBF07_8006); // v_readfirstlane_b32 s6, v2; s_cmp_lg_u32 s6, 0
    let l = lower_halves(&w)
        .expect("a VOP2 cndmask on an alike compare's VCC stays lane-uniform");
    assert_eq!(l.wave_size, Some(32), "the console wave ran as two halves");
    assert_eq!(
        l.shape.barriers, 0,
        "no cross-half exchange was emitted for the alike-compare VCC mask"
    );
}

/// A `v_dot4c_i32_i8` reads its destination as an implicit accumulator (in
/// `inst.reads`, though not among its operands): with a lane-varying old
/// destination (the thread id) and zero inputs it preserves the
/// lane-varying value, so its result is not the same in every lane. As the
/// VGPR offset of a `global_load_dword` in a shader without stores (where
/// the offset only needs to be lane-uniform), each half would load from a
/// different address while a `v_readfirstlane_b32` of the result skips the
/// cross-half exchange: the first-lane read in a divergent block keeps its
/// refusal.
#[test]
fn a_dot4c_with_a_lane_varying_accumulator_keeps_the_first_lane_refusal() {
    // v_dot4c_i32_i8 v1, v2, v3 (VOP2 opcode 0x00d): v1 = old v1 + dot4(v2, v3).
    const DOT4C_V1_V2_V3: u32 = 0x1A02_0702;
    // The old v1 is the thread id: lane-varying.
    let mut w = vec![
        0x7E02_0300, // v_mov_b32 v1, v0 (the thread id)
        0x7E04_02FF,
        0x0000_0000, // v_mov_b32 v2, 0
        0x7E06_02FF,
        0x0000_0000, // v_mov_b32 v3, 0
        DOT4C_V1_V2_V3,
    ];
    w.extend(global(0x0C, 5, 1, 0, 0)); // global_load_dword v5, v1, s[0:1], 0
    divergent_read(&mut w, 0x7E12_0505, 0xBF07_8009); // v_readfirstlane_b32 s9, v5; s_cmp_lg_u32 s9, 0
    assert!(
        refused_in_halves(&w),
        "a v_dot4c_i32_i8 over a lane-varying accumulator keeps the refusal"
    );
    // The same with a uniform old destination (0): the result is 0 in every
    // lane, so the load is lane-uniform and the first-lane read of it needs
    // no cross-half exchange.
    let mut w = vec![
        0x7E02_02FF,
        0x0000_0000, // v_mov_b32 v1, 0 (a uniform old accumulator)
        0x7E04_02FF,
        0x0000_0000, // v_mov_b32 v2, 0
        0x7E06_02FF,
        0x0000_0000, // v_mov_b32 v3, 0
        DOT4C_V1_V2_V3,
    ];
    w.extend(global(0x0C, 5, 1, 0, 0)); // global_load_dword v5, v1, s[0:1], 0
    divergent_read(&mut w, 0x7E12_0505, 0xBF07_8009); // v_readfirstlane_b32 s9, v5; s_cmp_lg_u32 s9, 0
    let l = lower_halves(&w)
        .expect("a v_dot4c_i32_i8 over a uniform accumulator lowers in halves");
    assert_eq!(l.wave_size, Some(32), "the console wave ran as two halves");
    assert_eq!(
        l.shape.barriers, 0,
        "no cross-half exchange was emitted for the uniform accumulator"
    );
}

/// An `s_load_dword` that can alias a store (`global_store_dword` at the same
/// address) reads memory whose stages the console wave's two host halves can
/// observe differently, so its result has no positive proof of uniformity
/// (not in [`same_s`]). An `s_cmp_*` of it writes an SCC that can differ
/// between the halves, and a `v_mov_b32 ..., src_scc` of that carry is one
/// value per host wave, not per console wave: a `v_readfirstlane_b32` of it
/// in a divergent block keeps its first-lane refusal.
#[test]
fn an_scc_from_an_aliasing_s_load_keeps_the_first_lane_refusal() {
    // s[0:3]: root constants (s[0:1] the s_load's and the store's base).
    let mut w = vec![
        0xF400_0080, 0xFA00_0040, // s_load_dword s2, s[0:1], 0x40 (can alias the store)
        0xBF8C_0000,              // s_waitcnt lgkmcnt(0)
        0xBF06_8002,              // s_cmp_eq_u32 s2, 0 (writes SCC from the load)
        0x7E02_02FD,              // v_mov_b32 v1, scc
        0x7E04_02FF, 0x0000_0000, // v_mov_b32 v2, 0 (the store's address)
        0x7E06_02FF, 0x0000_0777, // v_mov_b32 v3, 0x777 (the store's data)
    ];
    w.extend(global_store(3, 2, 0, 0x40)); // global_store_dword v3, v2, s[0:1], 0x40 (aliases the load)
    divergent_read(&mut w, 0x7E0C_0501, 0xBF07_8006); // v_readfirstlane_b32 s6, v1; s_cmp_lg_u32 s6, 0
    assert!(
        refused_in_halves(&w),
        "a v_mov_b32 of an s_load-derived SCC keeps the first-lane refusal"
    );
}

/// A `v_cmp_*` of an unproven scalar (an `s_load_dword` result that can alias
/// a store) writes a VCC whose per-lane bit can differ between the two host
/// halves; a `v_cndmask_b32` selecting on that VCC is then not alike, and a
/// `v_readfirstlane_b32` of its result in a divergent block keeps its
/// first-lane refusal.
#[test]
fn a_cndmask_on_a_vcc_from_an_aliasing_s_load_keeps_the_first_lane_refusal() {
    let mut w = vec![
        0xF400_0080, 0xFA00_0040, // s_load_dword s2, s[0:1], 0x40 (can alias the store)
        0xBF8C_0000,              // s_waitcnt lgkmcnt(0)
        0x7E04_02FF, 0x0000_0000, // v_mov_b32 v2, 0 (the store's address)
        0x7E0A_02FF, 0x0000_0040, // v_mov_b32 v5, 0x40
        0x7E0C_02FF, 0x0000_0000, // v_mov_b32 v6, 0
        0x7D84_0C02,              // v_cmp_eq_u32 vcc, s2, v6 (the unproven scalar)
        0xD501_0003, 128 | 192 << 9 | 106 << 18, // v_cndmask_b32 v3, 0, 0x40, vcc
    ];
    w.extend(global_store(2, 2, 0, 0x40)); // global_store_dword v2, v2, s[0:1], 0x40 (aliases the load)
    divergent_read(&mut w, 0x7E0E_0503, 0xBF07_8007); // v_readfirstlane_b32 s7, v3; s_cmp_lg_u32 s7, 0
    assert!(
        refused_in_halves(&w),
        "a v_cndmask_b32 on a vcc from an s_load compare keeps the first-lane refusal"
    );
}

/// A guarded `v_mov_b32 v5, 1` over an old `0`, under a guard a
/// `buffer_load_dword` comparison controls, with a `global_store_dword`
/// present: the load is not lane-uniform with a store present, so the guard
/// is not alike and one host half can run the write while the other keeps
/// the `0`. The guarded write must not enter `same`, and a
/// `v_readfirstlane_b32` of `v5` in a divergent block keeps its refusal.
#[test]
fn a_guarded_v_mov_under_a_buffer_load_guard_with_a_store_keeps_the_first_lane_refusal() {
    // s[0:3]: root constants (s[2:3] the store's address); s[4:5]: the root
    // table the buffer's descriptor is loaded from into s[8:11].
    let constants = RootParam {
        sgprs: 4,
        ..root(RootType::Constants, 0, 0)
    };
    let buffer = UsedResource {
        kind: ResourceType::Srv,
        root: RootType::Table,
        table_offset_dwords: 0,
        lower_bound: 0,
        stride: 4,
        dwords: 4,
        user_data: 4,
        slot: 1,
    };
    let mut w = vec![
        0x7E02_02FF,
        0x0000_0000, // v_mov_b32 v1, 0 (the buffer load's address)
        0x7E0A_02FF,
        0x0000_0000, // v_mov_b32 v5, 0 (the old value of the guarded write)
        0x7E0C_02FF,
        0x0000_0777, // v_mov_b32 v6, 0x777 (the store's data)
        0xBE85_0380, // s_mov_b32 s5, 0
        0xF408_0202,
        0xFA00_0000, // s_load_dwordx4 s[8:11], s[4:5], 0
        0xBF8C_0000, // s_waitcnt lgkmcnt(0)
        0xE030_1000,
        0x8002_0201, // buffer_load_dword v2, v1, s[8:11], 0 offen
        0x7D84_0480, // v_cmp_eq_u32 vcc, 0, v2 (not lane-uniform with a store present)
        0xBE90_246A, // s_and_saveexec_b64 s[16:17], vcc
    ];
    w.extend(global_store(6, 1, 2, 0)); // global_store_dword v6, v1, s[2:3], 0
    w.extend([0x7E0A_02FF, 0x0000_0001]); // v_mov_b32 v5, 1 (under the guard)
    w.push(0xBEFE_0410); // s_mov_b64 exec, s[16:17]
    divergent_read(&mut w, 0x7E12_0505, 0xBF07_8009); // v_readfirstlane_b32 s9, v5; s_cmp_lg_u32 s9, 0
    let first_across = super::halves::FIRST_ACROSS;
    assert!(
        matches!(lower_halves_with(&w, vec![constants, root(RootType::Table, 4, 1)], vec![buffer]), Err(crate::ShaderError::Unsupported { what, .. }) if what == first_across),
        "a guarded v_mov under a buffer-load guard with a store present keeps the refusal"
    );
}

/// `vccz` (inline constant 251) is a one-bit zero test (`0` or `1`), not an
/// all-zeros 64-bit lane mask: as a `v_cndmask_b32` select mask, `1` picks
/// the source only for lane 0. A first-lane read of the cndmask's result in
/// a divergent block keeps its refusal; an all-zeros literal mask still
/// selects the same source in every lane and needs no exchange.
#[test]
fn a_cndmask_with_vccz_as_its_mask_keeps_the_first_lane_refusal() {
    // v_cndmask_b32 v3, 0, 0x40, <mask> (VOP3 opcode 0x101: the third source
    // is the 64-bit mask).
    let cndmask = |mask: u32| [0xD501_0003, 128 | 192 << 9 | mask << 18];
    let head = [
        0x7E0A_02FF,
        0x0000_0040, // v_mov_b32 v5, 0x40
        0x7E0C_02FF,
        0x0000_0000, // v_mov_b32 v6, 0
        0x7D84_0C80, // v_cmp_eq_u32 vcc, 0, v6 (alike operands, so vcc is in `same`)
    ];
    // vccz as the select mask.
    let mut w = Vec::new();
    w.extend(head);
    w.extend(cndmask(251)); // v_cndmask_b32 v3, 0, 0x40, vccz
    divergent_read(&mut w, 0x7E0E_0503, 0xBF07_8007); // v_readfirstlane_b32 s7, v3; s_cmp_lg_u32 s7, 0
    assert!(
        refused_in_halves(&w),
        "vccz as a select mask (1 selects only lane 0) keeps the first-lane refusal"
    );
    // An all-zeros literal mask selects the same source in every lane.
    let mut w = Vec::new();
    w.extend(head);
    w.extend(cndmask(128)); // v_cndmask_b32 v3, 0, 0x40, 0
    divergent_read(&mut w, 0x7E0E_0503, 0xBF07_8007);
    let l = lower_halves(&w).expect("an all-zeros select mask stays lane-uniform");
    assert_eq!(l.wave_size, Some(32), "the console wave ran as two halves");
    assert_eq!(
        l.shape.barriers, 0,
        "no cross-half exchange was emitted for the all-zeros mask"
    );
}


