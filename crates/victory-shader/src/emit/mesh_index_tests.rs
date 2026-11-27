//! Clean-room regression tests for mesh shader primitive index bounds and
//! zero-vertex meshes:
//! 1. Culled primitives carrying out-of-bounds corner indices (and edge flags).
//! 2. Non-culled primitives with corner indices >= vertex count (both constant
//!    and dynamically loaded from root constants).
//! 3. Meshes declaring zero vertices (with zero or non-zero primitives) and
//!    empty meshlets that exit without exporting vertices.

use super::container::{self, SigElement};
use super::{HostShader, emit};
use crate::dxbc::{Container, FourCC};
use crate::native::{HwStage, MeshIo, NativeProgram, RootParam, RootType, StageIo};
use crate::{ShaderError, io, ir, resource, structure};

const S_ENDPGM: u32 = 0xBF81_0000;
const EXEC_LO: u8 = 0x7E;
const VCC_LO: u8 = 0x6A;
const M0: u8 = 0x7C;

/// Packs three 9-bit corner indices into the GFX10 `exp prim` word.
const fn pack_prim(i0: u32, i1: u32, i2: u32) -> u32 {
    (i0 & 0x1FF) | ((i1 & 0x1FF) << 10) | ((i2 & 0x1FF) << 20)
}

/// Triangle `(0, 1, 2)` covering the full 16x16 render target.
const TRIANGLE: u32 = pack_prim(0, 1, 2);
/// Edge flags above each 9-bit corner index (bits 9, 19, 29).
const EDGE_FLAGS: u32 = (1 << 9) | (1 << 19) | (1 << 29);
/// Primitive cull bit (bit 31).
const CULL: u32 = 1 << 31;

const fn s_mov_b32(sdst: u8, src0: u8) -> u32 {
    0xBE80_0000 | (sdst as u32) << 16 | 0x03 << 8 | src0 as u32
}

const fn s_and_saveexec_b32(sdst: u8, src0: u8) -> u32 {
    0xBE80_0000 | (sdst as u32) << 16 | 0x3C << 8 | src0 as u32
}

const fn s_cbranch_execz(simm16: i16) -> u32 {
    0xBF88_0000 | (simm16 as u16 as u32)
}

/// `v_cmp_eq_u32_e32 vcc, imm, vsrc1` (true when `vsrc1 == imm`, `imm <= 64`).
const fn v_cmp_eq_u32_imm(imm: u8, vsrc1: u8) -> u32 {
    0x7C00_0000 | (0xC2 << 17) | (vsrc1 as u32) << 9 | (0x80 + imm as u32)
}

/// `v_cmp_gt_u32_e32 vcc, imm, vsrc1` (true when `vsrc1 < imm`, `imm <= 64`).
const fn v_cmp_gt_u32_imm(imm: u8, vsrc1: u8) -> u32 {
    0x7C00_0000 | (0xC4 << 17) | (vsrc1 as u32) << 9 | (0x80 + imm as u32)
}

/// `s_sendmsg sendmsg(MSG_GS_ALLOC_REQ)` (id 9).
const S_SENDMSG_GS_ALLOC_REQ: u32 = 0xBF90_0009;

#[derive(Clone, Copy)]
enum PrimSpec {
    /// Emit a literal primitive word on the given lane.
    Literal { lane: u8, word: u32 },
    /// Emit the primitive word from root constant `s8` on the given lane.
    RootConst { lane: u8 },
    /// Emit a literal primitive word unconditionally on all 32 lanes.
    AllLanesLiteral { word: u32 },
}

/// Builds a synthetic 32-lane triangle mesh shader with configurable
/// `max_vertices`, `max_primitives`, vertex export count `emit_vertices`
/// (lanes `< emit_vertices` export `pos0`), and per-lane primitive exports.
fn build_mesh(
    max_vertices: u32,
    max_primitives: u32,
    emit_vertices: u8,
    prims: &[PrimSpec],
    gs_alloc_zero: bool,
) -> Result<HostShader, ShaderError> {
    let uses_root = prims
        .iter()
        .any(|p| matches!(p, PrimSpec::RootConst { .. }));

    let mut words: Vec<u32> = Vec::new();
    if gs_alloc_zero {
        words.push(s_mov_b32(M0, 0x80)); // s_mov_b32 m0, 0
        words.push(S_SENDMSG_GS_ALLOC_REQ);
    }

    // The position export is always present, because the DXIL validator
    // wants every declared output written. With `emit_vertices` 0 it is
    // guarded by a test no lane passes.
    {
        words.extend([
            0xD765_0001u32, // v_mbcnt_lo_u32_b32 v1, -1, 0
            0x0001_00C1,
        ]);
    }

    {
        words.extend([
            0x3604_0281, // v_and_b32 v2, 1, v1
            0x3606_0282, // v_and_b32 v3, 2, v1
            0x7E04_0D02, // v_cvt_f32_u32 v2, v2
            0x7E06_0D03, // v_cvt_f32_u32 v3, v3
            0x1008_04F6, // v_mul_f32 v4, 4.0, v2
            0x0608_08F3, // v_add_f32 v4, -1.0, v4
            0x100A_06F4, // v_mul_f32 v5, 2.0, v3
            0x060A_0AF3, // v_add_f32 v5, -1.0, v5
            0x7E0C_02F0, // v_mov_b32 v6, 0.5
            0x7E0E_02F2, // v_mov_b32 v7, 1.0
        ]);
        if emit_vertices < 32 {
            words.push(v_cmp_gt_u32_imm(emit_vertices, 1));
            words.push(s_and_saveexec_b32(12, VCC_LO));
            words.push(s_cbranch_execz(2));
            words.extend([
                0xF800_08CF, // exp pos0 v4, v5, v6, v7 done
                0x0706_0504,
            ]);
            words.push(s_mov_b32(EXEC_LO, 12));
        } else {
            words.extend([
                0xF800_08CF, // exp pos0 v4, v5, v6, v7 done
                0x0706_0504,
            ]);
        }
    }

    for &spec in prims {
        match spec {
            PrimSpec::AllLanesLiteral { word } => {
                words.extend([
                    0x7E12_02FF,
                    word, // v_mov_b32 v9, word
                    0xF800_0941, // exp prim v9, off, off, off done
                    0x0000_0009,
                ]);
            }
            PrimSpec::Literal { lane, word } => {
                words.extend([0x7E12_02FF, word]); // v_mov_b32 v9, word
                words.push(v_cmp_eq_u32_imm(lane, 1));
                words.push(s_and_saveexec_b32(12, VCC_LO));
                words.push(s_cbranch_execz(2));
                words.extend([
                    0xF800_0941, // exp prim v9, off, off, off done
                    0x0000_0009,
                ]);
                words.push(s_mov_b32(EXEC_LO, 12));
            }
            PrimSpec::RootConst { lane } => {
                // HwStage::Ms maps user_data 0 to s8 (`user_sgpr(0) == 8`).
                words.push(0x7E12_0208); // v_mov_b32 v9, s8
                words.push(v_cmp_eq_u32_imm(lane, 1));
                words.push(s_and_saveexec_b32(12, VCC_LO));
                words.push(s_cbranch_execz(2));
                words.extend([
                    0xF800_0941, // exp prim v9, off, off, off done
                    0x0000_0009,
                ]);
                words.push(s_mov_b32(EXEC_LO, 12));
            }
        }
    }

    words.push(S_ENDPGM);

    let layout = if uses_root {
        vec![RootParam {
            root: RootType::Constants,
            table_offset_dwords: 0,
            sgprs: 1,
            base_register: 0,
            user_data: 0,
            slot: 0,
        }]
    } else {
        vec![]
    };
    let native = NativeProgram {
        stage: HwStage::Ms,
        wave64: false,
        vgprs: 10,
        layout,
        used: vec![],
        io: StageIo::Ms(MeshIo {
            outputs: vec![],
            clip_cntl: 0,
            threads: 32,
            max_vertices,
            max_primitives,
            topology: 2,
            lds_bytes: 0,
        }),
        code: words.iter().flat_map(|w| w.to_le_bytes()).collect(),
    };
    let position = SigElement {
        name: "SV_Position".into(),
        index: 0,
        kind: 3,
        comp: 9,
        interp: 0,
        row: Some(0),
        col: 0,
        cols: 4,
        usage: 0xF,
        half: false,
    };
    let osg1 = container::signature(&[position], false);
    let bytes = crate::dxbc::build(&[(FourCC::OSG1, &osg1)]);
    let c = Container::parse(&bytes)?;
    let p = ir::build(&native.patched_code(), native.wave64, native.vgprs)?;
    let st = structure::structure(&p)?;
    let res = resource::track(&p, &native.layout, native.stage, Some(&st))?;
    let io = io::raise(&p, &native)?;
    emit(&c, &native, &p, &st, &io, &res)
}

/// Reads `(max_vertices, max_primitives)` from an emitted mesh DXIL container's
/// `PSV0` part.
fn psv0_mesh_counts(dxil: &[u8]) -> Option<(u16, u16)> {
    let c = Container::parse(dxil).ok()?;
    let psv = c.part(FourCC::PSV0)?;
    // `PSV0` begins with the 4-byte runtime-info size followed by PSVRuntimeInfo0
    // (24 bytes), where mesh shaders store max_vertices at +12 and
    // max_primitives at +14.
    let info = psv.data.get(4..28)?;
    let max_v = u16::from_le_bytes([info[12], info[13]]);
    let max_p = u16::from_le_bytes([info[14], info[15]]);
    Some((max_v, max_p))
}

#[test]
fn mesh_index_cases_translate() {
    #[cfg(windows)]
    let mut validator = super::validator::Validator::load(
        super::validator::Validator::default_path().as_deref(),
    )
    .ok();

    let cases = [
        (
            "culled_oob_32v",
            build_mesh(
                32,
                32,
                32,
                &[PrimSpec::Literal {
                    lane: 0,
                    word: CULL | EDGE_FLAGS | pack_prim(300, 400, 511),
                }],
                false,
            ),
            (32u16, 32u16),
        ),
        (
            "non_culled_oob_3v",
            build_mesh(
                3,
                1,
                3,
                &[PrimSpec::Literal {
                    lane: 0,
                    word: pack_prim(0, 1, 3),
                }],
                false,
            ),
            (3, 1),
        ),
        (
            "non_culled_oob_dynamic_3v",
            build_mesh(3, 1, 3, &[PrimSpec::RootConst { lane: 0 }], false),
            (3, 1),
        ),
        (
            "prim_slot_beyond_max_prims_1",
            build_mesh(
                3,
                1,
                3,
                &[
                    PrimSpec::AllLanesLiteral { word: TRIANGLE },
                    PrimSpec::Literal {
                        lane: 0,
                        word: CULL | TRIANGLE,
                    },
                ],
                false,
            ),
            (3, 1),
        ),
        (
            "zero_verts_zero_prims",
            build_mesh(0, 0, 0, &[], false),
            (0, 0),
        ),
        (
            "zero_verts_one_prim",
            build_mesh(0, 1, 0, &[], false),
            (0, 0),
        ),
        (
            "zero_verts_with_exp_prim",
            build_mesh(
                0,
                32,
                0,
                &[PrimSpec::Literal {
                    lane: 0,
                    word: TRIANGLE,
                }],
                false,
            ),
            (0, 0),
        ),
        (
            "runtime_empty_meshlet",
            build_mesh(32, 32, 0, &[], true),
            (32, 32),
        ),
    ];
    for (name, res, expected_counts) in cases {
        let ms = res.unwrap_or_else(|e| panic!("{name}: translate failed: {e}"));
        assert!(!ms.dxil.is_empty(), "{name}: empty DXIL");
        assert_eq!(
            psv0_mesh_counts(&ms.dxil),
            Some(expected_counts),
            "{name}: unexpected PSV0 mesh counts"
        );
        #[cfg(windows)]
        if let Some(v) = validator.as_mut() {
            v.validate(&mut ms.dxil.clone())
                .unwrap_or_else(|e| panic!("{name}: DXIL validation failed: {e}"));
        }
    }
}

#[cfg(windows)]
mod gpu {
    use super::*;
    use crate::emit::bitcode::{Attr, Term};
    use crate::emit::host::{Draw, Host, Root};
    use crate::emit::shader::Shader;
    use windows::Win32::Graphics::Direct3D12::{
        D3D12_FEATURE_D3D12_OPTIONS7, D3D12_FEATURE_DATA_D3D12_OPTIONS7,
        D3D12_MESH_SHADER_TIER_NOT_SUPPORTED, D3D12_SHADER_VISIBILITY_MESH,
    };

    const LIT_COLOR: [f32; 4] = [1.0, 0.5, 0.25, 1.0];

    fn constant_color_ps() -> Vec<u8> {
        let mut s = Shader::new(true);
        s.outputs.push(SigElement {
            name: "SV_Target".into(),
            index: 0,
            kind: 16,
            comp: 9,
            interp: 0,
            row: Some(0),
            col: 0,
            cols: 4,
            usage: 0xF,
            half: false,
        });
        let b = s.entry;
        for (c, v) in LIT_COLOR.into_iter().enumerate() {
            let args = [
                s.m.i32(0),
                s.m.i32(0),
                s.m.i8(c as u8),
                s.m.f32(v.to_bits()),
            ];
            s.op(b, "storeOutput.f32", Attr::NoUnwind, s.t.void, 5, &args);
        }
        s.m.terminate(b, Term::Ret);
        s.finish()
    }

    fn run_mesh(
        host: &Host,
        ms: &HostShader,
        ps: &[u8],
        root_const: u32,
    ) -> Result<Vec<[f32; 4]>, String> {
        let root = host
            .root_signature_for(&[(ms, D3D12_SHADER_VISIBILITY_MESH)])
            .map_err(|e| format!("root_signature: {e}"))?;
        let pso = host
            .mesh_pipeline_with(&root, &ms.dxil, ps)
            .map_err(|e| format!("mesh_pipeline: {e}"))?;
        host.draw(
            &root,
            &pso,
            &Draw {
                roots: vec![(0, Root::Constants(vec![root_const]))],
                textures: &[],
                samplers: &[],
                vertices: &[],
                stride: 0,
                size: (16, 16),
                mesh_groups: 1,
            },
        )
        .map_err(|e| format!("draw: {e}"))
    }

    /// Diagnostic & regression probe on live hardware (Apple Silicon D3DMetal
    /// under Wine, or native D3D12) covering all three mesh-index cases:
    /// culled OOB indices, non-culled indices >= vertex count, and zero-vertex
    /// meshes. After each case, executes a healthy sanity triangle draw on the
    /// same device to detect silent device removal or queue corruption.
    #[test]
    fn mesh_index_bounds_and_zero_vertex_on_gpu() {
        let _one = crate::emit::valu_tests::ONE_AT_A_TIME
            .lock()
            .unwrap_or_else(|e| e.into_inner());
        let ps = constant_color_ps();
        let sanity_ms =
            build_mesh(3, 1, 3, &[PrimSpec::Literal { lane: 0, word: TRIANGLE }], false).unwrap();

        let cases: &[(&str, HostShader, u32, Option<usize>)] = &[
            (
                "1a_culled_oob_32v",
                build_mesh(
                    32,
                    32,
                    32,
                    &[PrimSpec::Literal {
                        lane: 0,
                        word: CULL | EDGE_FLAGS | pack_prim(300, 400, 511),
                    }],
                    false,
                )
                .unwrap(),
                0,
                Some(0),
            ),
            (
                "1b_culled_oob_alongside_valid_3v",
                build_mesh(
                    3,
                    2,
                    3,
                    &[
                        PrimSpec::Literal {
                            lane: 0,
                            word: TRIANGLE,
                        },
                        PrimSpec::Literal {
                            lane: 1,
                            word: CULL | EDGE_FLAGS | pack_prim(3, 300, 511),
                        },
                    ],
                    false,
                )
                .unwrap(),
                0,
                Some(256),
            ),
            (
                "2a_non_culled_oob_const_0_1_3_in_3v",
                build_mesh(
                    3,
                    1,
                    3,
                    &[PrimSpec::Literal {
                        lane: 0,
                        word: pack_prim(0, 1, 3),
                    }],
                    false,
                )
                .unwrap(),
                0,
                Some(256),
            ),
            (
                "2b_non_culled_oob_const_3_300_511_in_3v",
                build_mesh(
                    3,
                    1,
                    3,
                    &[PrimSpec::Literal {
                        lane: 0,
                        word: pack_prim(3, 300, 511),
                    }],
                    false,
                )
                .unwrap(),
                0,
                Some(0),
            ),
            (
                "2c_non_culled_oob_alongside_valid_3v",
                build_mesh(
                    3,
                    2,
                    3,
                    &[
                        PrimSpec::Literal {
                            lane: 0,
                            word: TRIANGLE,
                        },
                        PrimSpec::Literal {
                            lane: 1,
                            word: pack_prim(3, 300, 511),
                        },
                    ],
                    false,
                )
                .unwrap(),
                0,
                Some(256),
            ),
            (
                "2d_non_culled_oob_const_300_400_511_in_32v",
                build_mesh(
                    32,
                    32,
                    32,
                    &[PrimSpec::Literal {
                        lane: 0,
                        word: pack_prim(300, 400, 511),
                    }],
                    false,
                )
                .unwrap(),
                0,
                Some(0),
            ),
            (
                "2e_non_culled_oob_dynamic_0_1_3_in_3v",
                build_mesh(3, 1, 3, &[PrimSpec::RootConst { lane: 0 }], false).unwrap(),
                pack_prim(0, 1, 3),
                Some(256),
            ),
            (
                "2f_non_culled_oob_dynamic_3_300_511_in_3v",
                build_mesh(3, 1, 3, &[PrimSpec::RootConst { lane: 0 }], false).unwrap(),
                pack_prim(3, 300, 511),
                Some(0),
            ),
            (
                "2g_non_culled_oob_dynamic_0_1_511_in_32v",
                build_mesh(32, 32, 32, &[PrimSpec::RootConst { lane: 0 }], false).unwrap(),
                pack_prim(0, 1, 511),
                None,
            ),
            (
                "2h_non_culled_oob_const_256_257_258_in_3v",
                build_mesh(
                    3,
                    1,
                    3,
                    &[PrimSpec::Literal {
                        lane: 0,
                        word: pack_prim(256, 257, 258),
                    }],
                    false,
                )
                .unwrap(),
                0,
                Some(0),
            ),
            (
                "2i_non_culled_oob_dynamic_256_257_258_in_3v",
                build_mesh(3, 1, 3, &[PrimSpec::RootConst { lane: 0 }], false).unwrap(),
                pack_prim(256, 257, 258),
                Some(0),
            ),
            (
                "2j_non_culled_oob_dynamic_256_257_258_in_32v",
                build_mesh(32, 32, 32, &[PrimSpec::RootConst { lane: 0 }], false).unwrap(),
                pack_prim(256, 257, 258),
                Some(0),
            ),
            (
                "2k_prim_slot_beyond_max_prims_1",
                build_mesh(
                    3,
                    1,
                    3,
                    &[
                        PrimSpec::AllLanesLiteral { word: TRIANGLE },
                        PrimSpec::Literal {
                            lane: 0,
                            word: CULL | TRIANGLE,
                        },
                    ],
                    false,
                )
                .unwrap(),
                0,
                Some(0),
            ),
            (
                "3a_zero_verts_zero_prims",
                build_mesh(0, 0, 0, &[], false).unwrap(),
                0,
                Some(0),
            ),
            (
                "3b_zero_verts_one_prim_no_exp",
                build_mesh(0, 1, 0, &[], false).unwrap(),
                0,
                Some(0),
            ),
            (
                "3c_zero_verts_32_prims_non_culled_exp",
                build_mesh(
                    0,
                    32,
                    0,
                    &[PrimSpec::Literal {
                        lane: 0,
                        word: TRIANGLE,
                    }],
                    false,
                )
                .unwrap(),
                0,
                Some(0),
            ),
            (
                "3d_zero_verts_32_prims_culled_exp",
                build_mesh(
                    0,
                    32,
                    0,
                    &[PrimSpec::Literal {
                        lane: 0,
                        word: TRIANGLE | CULL,
                    }],
                    false,
                )
                .unwrap(),
                0,
                Some(0),
            ),
            (
                "3e_runtime_empty_meshlet_gs_alloc_zero",
                build_mesh(32, 32, 0, &[], true).unwrap(),
                0,
                Some(0),
            ),
        ];

        let mut ran = 0;
        let mut failures: Vec<String> = Vec::new();
        for (label, warp) in [("hardware", false), ("warp", true)] {
            let Ok(host) = Host::new(warp) else {
                eprintln!("no SM 6.6 {label} device; skipped");
                continue;
            };
            let mut options = D3D12_FEATURE_DATA_D3D12_OPTIONS7::default();
            // SAFETY: queries D3D12_OPTIONS7 into an appropriately sized struct.
            let queried = unsafe {
                host.device.CheckFeatureSupport(
                    D3D12_FEATURE_D3D12_OPTIONS7,
                    (&raw mut options).cast(),
                    size_of_val(&options) as u32,
                )
            };
            if queried.is_err() || options.MeshShaderTier == D3D12_MESH_SHADER_TIER_NOT_SUPPORTED {
                eprintln!("no mesh shaders on {label} device; skipped");
                continue;
            }
            ran += 1;

            for &(name, ref ms, rc, expected_lit) in cases {
                match run_mesh(&host, ms, &ps, rc) {
                    Ok(texels) => {
                        let lit = texels.iter().filter(|t| t[3] != 0.0).count();
                        let exact_lit = texels.iter().filter(|t| **t == LIT_COLOR).count();
                        let sanity = run_mesh(&host, &sanity_ms, &ps, 0).map(|t| {
                            t.iter().filter(|p| **p == LIT_COLOR).count()
                        });
                        eprintln!(
                            "[{label}] {name}: OK lit={lit}/256 exact={exact_lit}/256 first={:?} sanity={sanity:?}",
                            texels[0]
                        );
                        if let Some(exp) = expected_lit
                            && (lit != exp || (exp > 0 && exact_lit != exp))
                        {
                            failures.push(format!(
                                "[{label}] {name}: expected {exp} lit, got lit={lit} exact={exact_lit}"
                            ));
                        }
                        if sanity != Ok(256) {
                            failures.push(format!(
                                "[{label}] {name}: post-draw sanity failed: {sanity:?}"
                            ));
                        }
                    }
                    Err(e) => {
                        let sanity = run_mesh(&host, &sanity_ms, &ps, 0).map(|t| {
                            t.iter().filter(|p| **p == LIT_COLOR).count()
                        });
                        eprintln!("[{label}] {name}: ERR {e} | sanity={sanity:?}");
                        failures.push(format!("[{label}] {name}: ERR {e}"));
                    }
                }
            }
        }
        eprintln!("mesh_index_bounds_and_zero_vertex_on_gpu ran on {ran} device(s)");
        assert!(
            failures.is_empty(),
            "GPU mesh index failures:\n{}",
            failures.join("\n")
        );
    }
}
