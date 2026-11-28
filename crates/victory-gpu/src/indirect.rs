//! Indirect execution's arguments, bounded before the GPU reads them. The
//! title's own compute passes write most of them; a pass we drop or
//! translate wrongly leaves whatever the memory held, and a draw of four
//! billion vertices or a dispatch past the host's limits runs until the
//! host resets the display driver. So every record is clamped on the GPU,
//! just before its execution, by a compute shader of ours (HLSL compiled by
//! the system's compiler; a 5.1 shader with its root signature inside).
//!
//! Each list also has a budget (2^27 vertex instances or threads, groups
//! counted as 64 threads, mesh groups as 128): records past it run empty,
//! so a list of thousands of garbage records cannot run for seconds
//! either. By default (`VICTORY_INDIRECT_REST=0`), a record that does not
//! fit in the remaining budget runs empty without taking the rest, so
//! smaller draws after it still run.
//!
//! With a count buffer ExecuteIndirect runs only the first min(max, count)
//! records, so the clamp reads the count word and stops there: the maximum
//! can be far above the count, and the records between never run.
//!
//! shortcut: the bounds (below 2^23 vertices per instance, 2^26 vertex
//! instances, 65,535 groups per dimension except in x where [`widest`]
//! allows more, 2^20 per dispatch, one-dimensional mesh dispatches up to
//! 65,535 groups, the budget) cut
//! legitimate work beyond them; raise them if the title needs more.

use windows::Win32::Graphics::Direct3D::Fxc::D3DCompile;
use windows::Win32::Graphics::Direct3D::ID3DBlob;
use windows::Win32::Graphics::Direct3D12::*;
use windows::core::s;

use std::sync::atomic::{AtomicU64, Ordering};

use crate::host::{HostList, global_barrier, host, transition, uav_barrier};

/// Whether indirect argument clamps without an earlier conflicting GPU write
/// in the same host list are hoisted into the list's prelude as a single
/// compute batch (`VICTORY_BATCH_CLAMPS=0` disables and clamps inline).
pub(crate) fn batch_clamps_enabled_value(value: Option<&std::ffi::OsStr>) -> bool {
    value != Some(std::ffi::OsStr::new("0"))
}

/// Whether indirect argument clamps use scoped Enhanced Global Barriers and
/// bump-allocated disjoint inline argument slices (`VICTORY_CLAMP_SYNC=1`,
/// default off until benchmarked).
pub(crate) fn clamp_sync_enabled_value(value: Option<&std::ffi::OsStr>) -> bool {
    crate::switch::value_is_on(value)
}

#[cfg(test)]
std::thread_local! {
    static TEST_CLAMP_SYNC: std::cell::Cell<Option<bool>> = const { std::cell::Cell::new(None) };
}

#[cfg(test)]
pub(crate) fn with_clamp_sync<R>(enabled: bool, f: impl FnOnce() -> R) -> R {
    struct Restore(Option<bool>);
    impl Drop for Restore {
        fn drop(&mut self) {
            TEST_CLAMP_SYNC.with(|c| c.set(self.0));
        }
    }
    let previous = TEST_CLAMP_SYNC.with(|c| c.replace(Some(enabled)));
    let _restore = Restore(previous);
    f()
}

pub(crate) fn clamp_sync_enabled() -> bool {
    #[cfg(test)]
    if let Some(enabled) = TEST_CLAMP_SYNC.with(|c| c.get()) {
        return enabled;
    }
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| {
        clamp_sync_enabled_value(std::env::var_os("VICTORY_CLAMP_SYNC").as_deref())
    })
}

#[cfg(test)]
std::thread_local! {
    static TEST_BATCH_CLAMPS: std::cell::Cell<Option<bool>> = const { std::cell::Cell::new(None) };
}

#[cfg(test)]
pub(crate) fn with_batch_clamps<R>(enabled: bool, f: impl FnOnce() -> R) -> R {
    struct Restore(Option<bool>);
    impl Drop for Restore {
        fn drop(&mut self) {
            TEST_BATCH_CLAMPS.with(|c| c.set(self.0));
        }
    }
    let previous = TEST_BATCH_CLAMPS.with(|c| c.replace(Some(enabled)));
    let _restore = Restore(previous);
    f()
}

pub(crate) fn batch_clamps_enabled() -> bool {
    #[cfg(test)]
    if let Some(enabled) = TEST_BATCH_CLAMPS.with(|c| c.get()) {
        return enabled;
    }
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| {
        batch_clamps_enabled_value(std::env::var_os("VICTORY_BATCH_CLAMPS").as_deref())
    })
}

#[cfg(test)]
std::thread_local! {
    static TEST_CLAMP_CUTS: std::cell::Cell<Option<bool>> = const { std::cell::Cell::new(None) };
}

#[cfg(test)]
pub(crate) fn with_clamp_cuts<R>(enabled: bool, f: impl FnOnce() -> R) -> R {
    struct Restore(Option<bool>);
    impl Drop for Restore {
        fn drop(&mut self) {
            TEST_CLAMP_CUTS.with(|c| c.set(self.0));
        }
    }
    let previous = TEST_CLAMP_CUTS.with(|c| c.replace(Some(enabled)));
    let _restore = Restore(previous);
    f()
}

/// Whether a clamp a title list's host list cannot hoist (a shader of the
/// list may have written its arguments) starts a new host list for the rest
/// of the title list, whose prelude then takes it and the clamps after it,
/// until the next write (`crate::list::cut_for_clamp`).
/// `VICTORY_CLAMP_CUTS=0` clamps them inline, as before.
///
/// Measured (Metal System Trace, M5 Pro, D3DMetal, 2026-11-27): inline, a
/// world frame's main command buffer held ~3,160 encoders, compute and
/// render in strict turn, one 3 µs clamp dispatch and one restarted render
/// pass per draw: D3DMetal ends the render pass for every dispatch, and
/// every clamp after the list's first unhoistable one ran inline (4,582
/// inline, 362 hoisted a frame in that run's log).
pub(crate) fn clamp_cuts_enabled() -> bool {
    #[cfg(test)]
    if let Some(enabled) = TEST_CLAMP_CUTS.with(|c| c.get()) {
        return enabled;
    }
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| crate::switch::on_by_default("VICTORY_CLAMP_CUTS"))
}

/// Whether the argument clamp compute pipeline is available on this host.
pub(crate) fn available() -> bool {
    clamp().is_some()
}

/// One indirect execution's clamp hoisted into a host list's prelude.
#[derive(Clone, Copy, Debug)]
pub(crate) struct PendingClamp {
    pub(crate) src_gpu: u64,
    pub(crate) dst_gpu: u64,
    pub(crate) count_gpu: Option<u64>,
    pub(crate) kind: u32,
    pub(crate) stride: u32,
    pub(crate) records: u32,
}

const SOURCE: &str = include_str!("indirect-clamp.hlsl");

/// The most groups in x the clamp lets a dispatch run (2^20 groups in all
/// still bound it). The title's tile passes are dispatches of one group per
/// 8x4 tile of the picture, 67,440 and 47,760 for its two tile classes in a
/// frame at 2560x1440 (measured on Windows in VIctory-7900XTX, 84eae35);
/// cut to 65,535, the last tiles, the lowest of the picture, never run.
/// The console runs what it is asked, and so does D3DMetal: an M5 Pro ran
/// every group of indirect dispatches up to 2^20 wide. D3D12's own limit,
/// 65,535, stays for other hosts. `VICTORY_INDIRECT_WIDE=0` or `1`
/// overrides.
pub(crate) fn widest() -> u32 {
    static WIDEST: std::sync::OnceLock<u32> = std::sync::OnceLock::new();
    *WIDEST.get_or_init(|| {
        let wide = match std::env::var("VICTORY_INDIRECT_WIDE").as_deref() {
            Ok("0") => false,
            Ok("1") => true,
            _ => crate::shader_cache::under_wine(),
        };
        if wide { 1 << 20 } else { 65_535 }
    })
}

struct Clamp {
    root: ID3D12RootSignature,
    pso: ID3D12PipelineState,
    /// Built from [`WORDS_SOURCE`] or [`WORDS_1_SOURCE`], which also reads
    /// the budget as words through root view 5 (the other views are those of
    /// [`SOURCE`]).
    words: bool,
}

// SAFETY: D3D12 objects are free-threaded.
unsafe impl Send for Clamp {}
// SAFETY: as above.
unsafe impl Sync for Clamp {}

fn clamp() -> Option<&'static Clamp> {
    static CLAMP: std::sync::OnceLock<Option<Clamp>> = std::sync::OnceLock::new();
    CLAMP
        .get_or_init(|| match make() {
            Ok(c) => Some(c),
            Err(e) => {
                tracing::warn!(error = %e, "no argument clamp: indirect execution dropped");
                None
            }
        })
        .as_ref()
}

fn make() -> windows::core::Result<Clamp> {
    #[cfg(victory_dxc_clamp)]
    match precompiled() {
        Ok(clamp) => {
            tracing::info!("argument clamp: using precompiled 64-thread shader");
            return Ok(clamp);
        }
        Err(e) => tracing::warn!(error = %e, "precompiled argument clamp unavailable"),
    }
    make_fxc()
        .or_else(|e| {
            tracing::info!(error = %e, "argument clamp: building the 64-lane words form");
            make_words()
        })
        .or_else(|e| {
            tracing::info!(error = %e, "argument clamp: building the one-thread words form");
            make_words_1()
        })
}

/// Whether a record that does not fit in what is left of its list's budget
/// takes the rest (`VICTORY_INDIRECT_REST=1`), emptying every record after
/// it. Off (`0`) by default so one oversized record zeroes only itself.
fn indirect_rest() -> bool {
    #[cfg(test)]
    if let Some(rest) = TEST_INDIRECT_REST.with(std::cell::Cell::get) {
        return rest;
    }
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| crate::switch::on("VICTORY_INDIRECT_REST"))
}

#[cfg(test)]
thread_local! {
    static TEST_INDIRECT_REST: std::cell::Cell<Option<bool>> = const { std::cell::Cell::new(None) };
}

#[cfg(test)]
fn with_indirect_rest<R>(rest: bool, f: impl FnOnce() -> R) -> R {
    struct Guard(Option<bool>);
    impl Drop for Guard {
        fn drop(&mut self) {
            TEST_INDIRECT_REST.with(|cell| cell.set(self.0));
        }
    }
    let prev = TEST_INDIRECT_REST.with(|cell| cell.replace(Some(rest)));
    let _guard = Guard(prev);
    f()
}

/// The clamp compiled by DXC at build time. D3DMetal finds a shader by the
/// `HASH` part of its container, and DXC leaves that and the container
/// digest as zeros when it cannot find `dxil.dll` to sign with, which would
/// make this shader share a cache entry with every other unsigned one (issue
/// #4). So both are checked here and set when they are zero.
#[cfg(victory_dxc_clamp)]
fn precompiled() -> windows::core::Result<Clamp> {
    let mut bytes = include_bytes!(concat!(env!("OUT_DIR"), "/indirect-clamp.dxil")).to_vec();
    match victory_shader::emit::sign_container(&mut bytes) {
        Ok(true) => tracing::info!("argument clamp: the compiled shader was unsigned, its digests are set"),
        Ok(false) => {}
        Err(e) => tracing::warn!(error = %e, "argument clamp: could not check the compiled shader's digests"),
    }
    from_bytes(&bytes)
}

/// The 64-lane clamp for Wine's HLSL compiler (vkd3d-shader), which has no
/// `Load` on `RWByteAddressBuffer`, no stores to indexed group-shared memory
/// and no `[RootSignature]`: [`SOURCE`]'s 64-lane tile clamp, read through
/// structured views of the same buffers, with per-lane scratch slots in
/// `budget` bytes `64..320` (`budgetWords[16..80]`).
const WORDS_SOURCE: &str = r#"
cbuffer Record : register(b0) { uint kind; uint stride; uint records; uint flags; uint widest; };
StructuredBuffer<uint> argWords : register(t0);
RWByteAddressBuffer budget : register(u1);
RWByteAddressBuffer dest : register(u0);
StructuredBuffer<uint> countWords : register(t1);
RWStructuredBuffer<uint> budgetWords : register(u2);
static const uint BUDGET = 1u << 27;
static const uint MOST_DRAW = 1u << 23;
static const uint GROUPS = 1u << 20;
static const uint PAST = 0xffffffffu;
uint3 load3(uint at) { return uint3(argWords[at / 4u], argWords[at / 4u + 1u], argWords[at / 4u + 2u]); }
uint clamped(uint at) {
    uint cost;
    uint base_bytes;
    if (kind <= 1u) {
        uint raw_count = argWords[at / 4u];
        uint raw_instances = argWords[at / 4u + 1u];
        uint count = raw_count;
        uint instances = 0u;
        if (raw_count >= MOST_DRAW) {
            count = 0u;
        } else {
            instances = min(raw_instances, (1u << 26) / max(count, 1u));
        }
        cost = count * instances;
        dest.Store2(at, uint2(count, instances));
        if (kind == 1u) {
            dest.Store3(at + 8u, uint3(argWords[at / 4u + 2u], argWords[at / 4u + 3u], argWords[at / 4u + 4u]));
            base_bytes = 20u;
        } else {
            dest.Store2(at + 8u, uint2(argWords[at / 4u + 2u], argWords[at / 4u + 3u]));
            base_bytes = 16u;
        }
    } else if (kind == 2u) {
        uint3 g = load3(at);
        g = uint3(g.x <= widest ? g.x : 65535u, min(g.y, 65535u), min(g.z, 65535u));
        if ((g.y != 0u && g.x > GROUPS / g.y) || (g.z != 0u && g.x * g.y > GROUPS / g.z)) {
            g = uint3(0u, 0u, 0u);
        }
        cost = g.x * g.y * g.z * 64u;
        dest.Store3(at, g);
        base_bytes = 12u;
    } else {
        uint3 raw_g = load3(at);
        uint3 g = uint3(raw_g.x, min(raw_g.y, 1u), min(raw_g.z, 1u));
        if (raw_g.x > 65535u) {
            g = uint3(0u, 0u, 0u);
        }
        cost = g.x * g.y * g.z * 128u;
        dest.Store3(at, g);
        base_bytes = 12u;
    }
    for (uint extra = base_bytes; extra < stride; extra += 4u) {
        dest.Store(at + extra, argWords[(at + extra) / 4u]);
    }
    return cost;
}
[numthreads(64, 1, 1)]
void main(uint3 id : SV_DispatchThreadID) {
    uint lane = id.x;
    uint left = 0u;
    if (lane == 0u) {
        left = BUDGET - min(budgetWords[0], BUDGET);
    }
    uint walked = records;
    if ((flags & 1u) != 0u) {
        walked = min(records, countWords[0]);
    }
    uint take_rest = flags & 2u;
    uint tiles = (records + 63u) / 64u;
    for (uint tile = 0u; tile < tiles; tile++) {
        uint record = tile * 64u + lane;
        uint at = record * stride;
        uint cost = 0u;
        if (record < walked) {
            cost = clamped(at);
        }
        budget.Store(64u + lane * 4u, cost);
        AllMemoryBarrierWithGroupSync();
        if (lane == 0u && tile * 64u < walked) {
            for (uint i = 0u; i < 64u; i++) {
                uint c = budgetWords[16u + i];
                if (c > left) {
                    budget.Store(64u + i * 4u, PAST);
                    if (take_rest != 0u) {
                        left = 0u;
                    }
                } else {
                    left -= c;
                }
            }
        }
        AllMemoryBarrierWithGroupSync();
        if (record < walked && budgetWords[16u + lane] == PAST) {
            if (kind <= 1u) {
                dest.Store2(at, uint2(0u, 0u));
            } else {
                dest.Store3(at, uint3(0u, 0u, 0u));
            }
        }
        if (tile + 1u < tiles) {
            AllMemoryBarrierWithGroupSync();
        }
    }
    if (lane == 0u) {
        budget.Store(0u, BUDGET - left);
    }
}
"#;

/// One-thread fallback for compilers that refuse [`WORDS_SOURCE`].
const WORDS_1_SOURCE: &str = r#"
cbuffer Record : register(b0) { uint kind; uint stride; uint records; uint flags; uint widest; };
StructuredBuffer<uint> argWords : register(t0);
RWByteAddressBuffer budget : register(u1);
RWByteAddressBuffer dest : register(u0);
StructuredBuffer<uint> countWords : register(t1);
RWStructuredBuffer<uint> budgetWords : register(u2);
static const uint BUDGET = 1u << 27;
static const uint MOST_DRAW = 1u << 23;
static const uint GROUPS = 1u << 20;
uint3 load3(uint at) { return uint3(argWords[at / 4u], argWords[at / 4u + 1u], argWords[at / 4u + 2u]); }
uint clamped(uint at) {
    uint cost;
    uint base_bytes;
    if (kind <= 1u) {
        uint raw_count = argWords[at / 4u];
        uint raw_instances = argWords[at / 4u + 1u];
        uint count = raw_count;
        uint instances = 0u;
        if (raw_count >= MOST_DRAW) {
            count = 0u;
        } else {
            instances = min(raw_instances, (1u << 26) / max(count, 1u));
        }
        cost = count * instances;
        dest.Store2(at, uint2(count, instances));
        if (kind == 1u) {
            dest.Store3(at + 8u, uint3(argWords[at / 4u + 2u], argWords[at / 4u + 3u], argWords[at / 4u + 4u]));
            base_bytes = 20u;
        } else {
            dest.Store2(at + 8u, uint2(argWords[at / 4u + 2u], argWords[at / 4u + 3u]));
            base_bytes = 16u;
        }
    } else if (kind == 2u) {
        uint3 g = load3(at);
        g = uint3(g.x <= widest ? g.x : 65535u, min(g.y, 65535u), min(g.z, 65535u));
        if ((g.y != 0u && g.x > GROUPS / g.y) || (g.z != 0u && g.x * g.y > GROUPS / g.z)) {
            g = uint3(0u, 0u, 0u);
        }
        cost = g.x * g.y * g.z * 64u;
        dest.Store3(at, g);
        base_bytes = 12u;
    } else {
        uint3 raw_g = load3(at);
        uint3 g = uint3(raw_g.x, min(raw_g.y, 1u), min(raw_g.z, 1u));
        if (raw_g.x > 65535u) {
            g = uint3(0u, 0u, 0u);
        }
        cost = g.x * g.y * g.z * 128u;
        dest.Store3(at, g);
        base_bytes = 12u;
    }
    for (uint extra = base_bytes; extra < stride; extra += 4u) {
        dest.Store(at + extra, argWords[(at + extra) / 4u]);
    }
    return cost;
}
[numthreads(1, 1, 1)]
void main() {
    uint left = BUDGET - min(budgetWords[0], BUDGET);
    uint walked = records;
    if ((flags & 1u) != 0u) {
        walked = min(records, countWords[0]);
    }
    uint take_rest = flags & 2u;
    for (uint record = 0u; record < walked; record++) {
        uint at = record * stride;
        uint cost = clamped(at);
        if (cost > left) {
            if (kind <= 1u) {
                dest.Store2(at, uint2(0u, 0u));
            } else {
                dest.Store3(at, uint3(0u, 0u, 0u));
            }
            if (take_rest != 0u) {
                left = 0u;
            }
        } else {
            left -= cost;
        }
    }
    budget.Store(0u, BUDGET - left);
}
"#;

fn make_words() -> windows::core::Result<Clamp> {
    make_words_from(WORDS_SOURCE)
}

fn make_words_1() -> windows::core::Result<Clamp> {
    make_words_from(WORDS_1_SOURCE)
}

fn make_words_from(source: &str) -> windows::core::Result<Clamp> {
    use crate::compute::Param;
    let params = [
        Param::Constants { register: 0, count: 5 },
        Param::Srv { register: 0, space: 0 },
        Param::Uav { register: 1 },
        Param::Uav { register: 0 },
        Param::Srv { register: 1, space: 0 },
        Param::Uav { register: 2 },
    ];
    let (root, pso) = crate::compute::pipeline(source, s!("clamp"), 0, &params)?;
    Ok(Clamp { root, pso, words: true })
}

fn make_fxc() -> windows::core::Result<Clamp> {
    let mut code: Option<ID3DBlob> = None;
    let mut errors: Option<ID3DBlob> = None;
    // SAFETY: the source and names live through the call; out pointers ours.
    let compiled = unsafe {
        D3DCompile(
            SOURCE.as_ptr().cast(),
            SOURCE.len(),
            s!("clamp"),
            None,
            None,
            s!("main"),
            s!("cs_5_1"),
            0,
            0,
            &mut code,
            Some(&mut errors),
        )
    };
    if let Err(e) = compiled {
        if let Some(errors) = errors {
            // SAFETY: the blob's own bytes.
            let text = unsafe {
                std::slice::from_raw_parts(errors.GetBufferPointer().cast::<u8>(), errors.GetBufferSize())
            };
            tracing::warn!("argument clamp: {}", String::from_utf8_lossy(text));
        }
        return Err(e);
    }
    let code = code.ok_or_else(windows::core::Error::empty)?;
    // SAFETY: the compiler owns the blob until this call returns.
    let bytes = unsafe { std::slice::from_raw_parts(code.GetBufferPointer().cast::<u8>(), code.GetBufferSize()) };
    from_bytes(bytes)
}

fn from_bytes(bytes: &[u8]) -> windows::core::Result<Clamp> {
    let device = &host().ok_or_else(windows::core::Error::empty)?.device;
    // SAFETY: D3D12 validates the bytecode and its embedded root signature.
    unsafe {
        let root: ID3D12RootSignature = device.CreateRootSignature(0, bytes)?;
        let pso = device.CreateComputePipelineState(&D3D12_COMPUTE_PIPELINE_STATE_DESC {
            pRootSignature: std::mem::transmute_copy(&root),
            CS: D3D12_SHADER_BYTECODE {
                pShaderBytecode: bytes.as_ptr().cast(),
                BytecodeLength: bytes.len(),
            },
            ..Default::default()
        })?;
        Ok(Clamp { root, pso, words: false })
    }
}

/// Records, in `list`, the clamp of `records` records of `kind` (PC
/// argument numbering) `stride` bytes apart at host address `gpu` (a title
/// buffer's copy, only read) into the list's own argument buffer, left
/// ready to be read as arguments: that buffer, for [`done`] after the
/// execution. Replaces the list's compute root signature and pipeline
/// (callers bind theirs after). `None` if the clamp is not available.
///
/// `count` is the host address of the title's record count (a title
/// buffer's copy, only read): the clamp then stops at the lesser of it and
/// `records`, which is where ExecuteIndirect stops. A count that is not
/// 4-byte aligned is not read: every record is clamped.
///
/// Title buffers' copies are never transitioned: they are common between
/// submissions and every use promotes them (shader and root views, the
/// input assembler, these reads), with unordered-access barriers for order.
/// Only the list's argument buffer moves, common, unordered access,
/// indirect argument, common: its states are known whatever order lists
/// run in.
// shortcut: a buffer a shader writes through an unordered view and the
// input assembler then reads in one submission is, by the letter of D3D12,
// promoted to a write state the input assembler may not use (the debug
// layer cannot see view accesses; GPU-based validation might); upgrade to
// per-list tracked buffer states if a driver or that validation minds.
pub(crate) fn bound(
    list: &HostList,
    gpu: u64,
    (kind, stride, records): (u32, u32, u32),
    count: Option<u64>,
) -> Option<(ID3D12Resource, u64)> {
    bound_with(clamp()?, list, gpu, (kind, stride, records), count)
}

/// [`bound`] with a given clamp.
fn bound_with(
    c: &Clamp,
    list: &HostList,
    gpu: u64,
    (kind, stride, records): (u32, u32, u32),
    count: Option<u64>,
) -> Option<(ID3D12Resource, u64)> {
    let budget = list.budget()?;
    if !gpu.is_multiple_of(4) || !stride.is_multiple_of(4) {
        return None;
    }
    let count = count.filter(|at| at.is_multiple_of(4));
    RECORDS[0].fetch_add(u64::from(records), Ordering::Relaxed);
    if count.is_some() {
        RECORDS[1].fetch_add(u64::from(records), Ordering::Relaxed);
    }
    CLAMPS[1].fetch_add(1, Ordering::Relaxed);
    let bytes = u64::from(stride) * u64::from(records.max(1));
    let cmd7 = list.narrow_clamp_sync_list();
    let (out, dst_offset, dst_gpu) = if cmd7.is_some() {
        // Bump-allocated disjoint slice: zero WAR hazard against earlier
        // `ExecuteIndirect` calls, so pre-clamp stays pure COMPUTE -> COMPUTE.
        list.inline_argument_slice(bytes)?
    } else {
        let out = list.arguments(bytes)?;
        // SAFETY: a live buffer.
        let dst_gpu = unsafe { out.GetGPUVirtualAddress() };
        (out, 0, dst_gpu)
    };
    let l = &list.list;
    let flags = u32::from(count.is_some()) | (u32::from(indirect_rest()) << 1);
    let constants = [kind, stride, records, flags, widest()];
    // SAFETY: an open list; our root signature, pipeline and buffers, and
    // title buffers' copies the caller's list holds.
    unsafe {
        if let Some(cmd7) = cmd7 {
            // Earlier compute writes (title argument/count passes and previous
            // clamp's budget update) -> this clamp's SRV (args/count) and UAV
            // (budget/disjoint dest) accesses.
            global_barrier(
                cmd7,
                D3D12_BARRIER_SYNC_COMPUTE_SHADING,
                D3D12_BARRIER_SYNC_COMPUTE_SHADING,
                D3D12_BARRIER_ACCESS_UNORDERED_ACCESS,
                D3D12_BARRIER_ACCESS_SHADER_RESOURCE | D3D12_BARRIER_ACCESS_UNORDERED_ACCESS,
            );
        } else {
            // Earlier shader writes (the title's argument passes) first.
            l.ResourceBarrier(&[
                uav_barrier(),
                transition(&out, D3D12_RESOURCE_STATE_COMMON, D3D12_RESOURCE_STATE_UNORDERED_ACCESS),
            ]);
        }
        l.SetComputeRootSignature(&c.root);
        l.SetPipelineState(&c.pso);
        l.SetComputeRoot32BitConstants(0, constants.len() as u32, constants.as_ptr().cast(), 0);
        l.SetComputeRootShaderResourceView(1, gpu);
        l.SetComputeRootUnorderedAccessView(2, budget);
        l.SetComputeRootUnorderedAccessView(3, dst_gpu);
        // Set even when not read: a root view has to be set.
        l.SetComputeRootShaderResourceView(4, count.unwrap_or(gpu));
        if c.words {
            l.SetComputeRootUnorderedAccessView(5, budget);
        }
        // One group walks every record (up to the count).
        l.Dispatch(records.min(1), 1, 1);
        if let Some(cmd7) = cmd7 {
            global_barrier(
                cmd7,
                D3D12_BARRIER_SYNC_COMPUTE_SHADING,
                D3D12_BARRIER_SYNC_EXECUTE_INDIRECT,
                D3D12_BARRIER_ACCESS_UNORDERED_ACCESS,
                D3D12_BARRIER_ACCESS_INDIRECT_ARGUMENT,
            );
        } else {
            l.ResourceBarrier(&[
                uav_barrier(),
                transition(&out, D3D12_RESOURCE_STATE_UNORDERED_ACCESS, D3D12_RESOURCE_STATE_INDIRECT_ARGUMENT),
            ]);
        }
    }
    Some((out, dst_offset))
}

/// Queues an indirect argument clamp to run in `list`'s prelude before
/// `list.list` starts, allocating a disjoint slice from `list`'s hoisted
/// argument arena. Returns the arena buffer and byte offset ready for
/// `ExecuteIndirect` (the prelude transitions the buffer into
/// `INDIRECT_ARGUMENT`, and `list.close()` transitions it back to `COMMON`).
pub(crate) fn enqueue(
    list: &HostList,
    gpu: u64,
    (kind, stride, records): (u32, u32, u32),
    count: Option<u64>,
) -> Option<(ID3D12Resource, u64)> {
    let _ = clamp()?;
    if !gpu.is_multiple_of(4) || !stride.is_multiple_of(4) {
        return None;
    }
    let count_gpu = count.filter(|at| at.is_multiple_of(4));
    let bytes = u64::from(stride) * u64::from(records.max(1));
    let out = list.enqueue_hoisted_clamp(bytes, |dst_gpu| PendingClamp {
        src_gpu: gpu,
        dst_gpu,
        count_gpu,
        kind,
        stride,
        records,
    })?;
    RECORDS[0].fetch_add(u64::from(records), Ordering::Relaxed);
    if count_gpu.is_some() {
        RECORDS[1].fetch_add(u64::from(records), Ordering::Relaxed);
    }
    CLAMPS[0].fetch_add(1, Ordering::Relaxed);
    Some(out)
}

/// Records all hoisted argument clamps for a host list into its prelude
/// command list as one compute batch: one pre-batch barrier, one root
/// signature + PSO + budget bind, one dispatch per non-empty clamp (with a
/// UAV barrier between consecutive dispatches so the shared list budget is
/// charged in exact recorded order), and one post-batch transition into
/// `INDIRECT_ARGUMENT`.
pub(crate) fn record_batch(
    cmd: &ID3D12GraphicsCommandList,
    cmd7: Option<&ID3D12GraphicsCommandList7>,
    budget: u64,
    buffers: &[ID3D12Resource],
    clamps: &[PendingClamp],
) {
    let Some(c) = clamp() else {
        return;
    };
    if buffers.is_empty() || clamps.is_empty() {
        return;
    }
    let (pre, post) = if cmd7.is_some() {
        (Vec::new(), Vec::new())
    } else {
        let mut pre = Vec::with_capacity(buffers.len() + 1);
        pre.push(uav_barrier());
        for buf in buffers {
            pre.push(transition(
                buf,
                D3D12_RESOURCE_STATE_COMMON,
                D3D12_RESOURCE_STATE_UNORDERED_ACCESS,
            ));
        }
        let mut post = Vec::with_capacity(buffers.len() + 1);
        post.push(uav_barrier());
        for buf in buffers {
            post.push(transition(
                buf,
                D3D12_RESOURCE_STATE_UNORDERED_ACCESS,
                D3D12_RESOURCE_STATE_INDIRECT_ARGUMENT,
            ));
        }
        (pre, post)
    };
    let widest = widest();
    let rest = indirect_rest();
    // SAFETY: `cmd` is the host list's open prelude command list; our clamp
    // pipeline, budget buffer, hoisted argument buffers, and title shadow
    // copies are kept alive by the host list until reset.
    unsafe {
        if let Some(cmd7) = cmd7 {
            global_barrier(
                cmd7,
                D3D12_BARRIER_SYNC_COMPUTE_SHADING,
                D3D12_BARRIER_SYNC_COMPUTE_SHADING,
                D3D12_BARRIER_ACCESS_UNORDERED_ACCESS,
                D3D12_BARRIER_ACCESS_SHADER_RESOURCE | D3D12_BARRIER_ACCESS_UNORDERED_ACCESS,
            );
        } else {
            cmd.ResourceBarrier(&pre);
        }
        cmd.SetComputeRootSignature(&c.root);
        cmd.SetPipelineState(&c.pso);
        cmd.SetComputeRootUnorderedAccessView(2, budget);
        if c.words {
            cmd.SetComputeRootUnorderedAccessView(5, budget);
        }
        let mut dispatched = false;
        for clamp in clamps {
            if clamp.records == 0 {
                continue;
            }
            if dispatched {
                // Serialize budget read-modify-write between consecutive
                // clamps in recorded order while staying in one compute pass.
                if let Some(cmd7) = cmd7 {
                    global_barrier(
                        cmd7,
                        D3D12_BARRIER_SYNC_COMPUTE_SHADING,
                        D3D12_BARRIER_SYNC_COMPUTE_SHADING,
                        D3D12_BARRIER_ACCESS_UNORDERED_ACCESS,
                        D3D12_BARRIER_ACCESS_UNORDERED_ACCESS,
                    );
                } else {
                    cmd.ResourceBarrier(&[uav_barrier()]);
                }
            }
            let constants = [
                clamp.kind,
                clamp.stride,
                clamp.records,
                u32::from(clamp.count_gpu.is_some()) | (u32::from(rest) << 1),
                widest,
            ];
            cmd.SetComputeRoot32BitConstants(
                0,
                constants.len() as u32,
                constants.as_ptr().cast(),
                0,
            );
            cmd.SetComputeRootShaderResourceView(1, clamp.src_gpu);
            cmd.SetComputeRootUnorderedAccessView(3, clamp.dst_gpu);
            cmd.SetComputeRootShaderResourceView(4, clamp.count_gpu.unwrap_or(clamp.src_gpu));
            cmd.Dispatch(1, 1, 1);
            dispatched = true;
        }
        if let Some(cmd7) = cmd7 {
            global_barrier(
                cmd7,
                D3D12_BARRIER_SYNC_COMPUTE_SHADING,
                D3D12_BARRIER_SYNC_EXECUTE_INDIRECT,
                D3D12_BARRIER_ACCESS_UNORDERED_ACCESS,
                D3D12_BARRIER_ACCESS_INDIRECT_ARGUMENT,
            );
        } else {
            cmd.ResourceBarrier(&post);
        }
    }
}

/// Records the clamps were asked to walk (the maximum of each execution)
/// since the last call, and the part of them under a count buffer, where
/// the walk stops at the count.
pub(crate) fn take_records() -> [u64; 2] {
    RECORDS.each_ref().map(|n| n.swap(0, Ordering::Relaxed))
}
static RECORDS: [AtomicU64; 2] = [AtomicU64::new(0), AtomicU64::new(0)];

/// Clamp invocations since the last call: `[hoisted, inline]`.
pub(crate) fn take_clamps() -> [u64; 2] {
    CLAMPS.each_ref().map(|n| n.swap(0, Ordering::Relaxed))
}
static CLAMPS: [AtomicU64; 2] = [AtomicU64::new(0), AtomicU64::new(0)];

/// Notes an inline clamp and what kept it out of the prelude (`None`:
/// batching is off, `VICTORY_BATCH_CLAMPS=0`).
pub(crate) fn note_inline(why: Option<crate::host::ClampBlock>) {
    use crate::host::ClampBlock;
    let at = match why {
        Some(ClampBlock::Unknown) => 0,
        Some(ClampBlock::Overlap) => 1,
        Some(ClampBlock::AfterInline) => 2,
        None => 3,
    };
    INLINE_WHY[at].fetch_add(1, Ordering::Relaxed);
}

/// Inline clamps since the last call by what kept them out of the
/// prelude: `[a write anywhere, a write of their arguments, an inline clamp
/// before them, batching off]`.
pub(crate) fn take_inline_why() -> [u64; 4] {
    INLINE_WHY.each_ref().map(|n| n.swap(0, Ordering::Relaxed))
}
static INLINE_WHY: [AtomicU64; 4] = [const { AtomicU64::new(0) }; 4];

/// A title list went on in a new host list for a clamp
/// ([`clamp_cuts_enabled`]).
pub(crate) fn note_cut() {
    CUTS.fetch_add(1, Ordering::Relaxed);
}

/// Host lists started for clamps since the last call.
pub(crate) fn take_cuts() -> u64 {
    CUTS.swap(0, Ordering::Relaxed)
}
static CUTS: AtomicU64 = AtomicU64::new(0);

/// After the execution: the list's argument buffer back to common.
/// No-op when narrow clamp sync is active (argument buffers stay in `COMMON`).
pub(crate) fn done(list: &HostList, arguments: &ID3D12Resource) {
    if list.narrow_clamp_sync() {
        return;
    }
    // SAFETY: an open list and a live buffer.
    unsafe {
        list.list.ResourceBarrier(&[transition(
            arguments,
            D3D12_RESOURCE_STATE_INDIRECT_ARGUMENT,
            D3D12_RESOURCE_STATE_COMMON,
        )])
    };
}

#[cfg(test)]
mod tests {
    use crate::backend::{HostGpu, HostMemory};

    /// The clamp compiles with the system's compiler (including both 64-lane
    /// and 1-lane words forms) and the host takes it.
    #[test]
    fn argument_clamp_builds() {
        if super::host().is_none() {
            return; // no host GPU here
        }
        assert!(super::clamp().is_some());
        assert!(super::make_words().is_ok());
        assert!(super::make_words_1().is_ok());
    }

    /// Every word the clamp `c` writes for `records` records of `kind` made
    /// from `seed` by a small generator, in one list with no count buffer.
    /// `None` without a host GPU.
    fn clamp_all(c: &super::Clamp, kind: u32, stride_words: usize, seed: u32, records: u32) -> Option<Vec<u32>> {
        let host = super::host()?;
        let all = stride_words * records as usize;
        let (buffer, data) = host.shared_buffer(all as u64 * 4 + 16).unwrap();
        let data = data.cast::<u32>();
        let list = host.new_list().unwrap();
        let mut state = seed;
        // SAFETY: our mapped buffer, written before the list runs and read
        // after it has; a live buffer.
        unsafe {
            for at in 0..all {
                state = state.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
                // Mostly small values, now and then a wild or all-ones one.
                let word = match state >> 28 {
                    0 => state,
                    1 => u32::MAX,
                    _ => (state >> 8) % 9000,
                };
                data.add(at).write_volatile(word);
            }
            let spec = (kind, stride_words as u32 * 4, records);
            let (out, offset) = super::bound_with(c, &list, buffer.GetGPUVirtualAddress(), spec, None).unwrap();
            super::done(&list, &out);
            list.close();
            host.execute(&[&list]);
            let mut out_data = std::ptr::null_mut();
            out.Map(0, None, Some(&mut out_data)).unwrap();
            let base = out_data.cast::<u8>().add(offset as usize).cast::<u32>();
            Some((0..all).map(|i| base.add(i).read_volatile()).collect())
        }
    }

    /// The precompiled clamp is the one in use when it builds, and it writes
    /// what the clamp compiled at run time writes for the same records of
    /// every kind.
    #[cfg(victory_dxc_clamp)]
    #[test]
    #[ignore = "requires a hardware D3D12 queue and the build-time DXC shader"]
    fn hardware_uses_precompiled_clamp() {
        if super::host().is_none() {
            return; // no host GPU here
        }
        let precompiled = super::precompiled().expect("create pipeline from the compiled clamp");
        assert!(!precompiled.words);
        assert!(!super::clamp().expect("argument clamp").words);
        let reference = super::make_fxc().or_else(|_| super::make_words()).expect("the existing clamp");
        for (kind, per, seed, records) in [(0, 4, 1, 200), (1, 5, 2, 200), (2, 3, 3, 130), (10, 3, 4, 70), (0, 8, 5, 135)] {
            let new = clamp_all(&precompiled, kind, per, seed, records).unwrap();
            let old = clamp_all(&reference, kind, per, seed, records).unwrap();
            assert_eq!(new, old, "kind {kind}, {records} records");
        }
    }

    /// The 64-lane `WORDS_SOURCE` clamp and the 1-lane `WORDS_1_SOURCE`
    /// fallback produce identical words across multi-tile batches and padded
    /// strides, under both `VICTORY_INDIRECT_REST=0` and `=1`.
    #[test]
    fn words_64_lane_matches_1_lane_across_tiles_and_strides() {
        if super::host().is_none() {
            return; // no host GPU here
        }
        let wide = super::make_words().expect("64-lane words clamp");
        let single = super::make_words_1().expect("1-lane words clamp");
        for rest in [false, true] {
            super::with_indirect_rest(rest, || {
                for (kind, per, seed, records) in [
                    (0, 4, 11, 150),
                    (0, 8, 12, 135),
                    (1, 5, 13, 140),
                    (2, 3, 14, 130),
                    (10, 3, 15, 145),
                    (10, 8, 16, 132),
                ] {
                    let a = clamp_all(&wide, kind, per, seed, records).unwrap();
                    let b = clamp_all(&single, kind, per, seed, records).unwrap();
                    assert_eq!(a, b, "rest {rest}, kind {kind}, stride_words {per}, records {records}");
                }
            });
        }
    }

    /// Clamps `records` records of `kind`, filled by repeating `words`,
    /// `calls` times in one list: each record's first word in the list's
    /// argument buffer after (zero when it runs empty). The title's copy,
    /// one record more included, stays as written. `None` without a host
    /// GPU.
    fn clamped(kind: u32, words: &[u32], records: u32, calls: u32) -> Option<Vec<u32>> {
        clamped_under(kind, words, records, calls, None)
    }

    /// As [`clamped`], with the title's record count `count` (a word in a
    /// buffer of its own) if there is one.
    fn clamped_under(
        kind: u32,
        words: &[u32],
        records: u32,
        calls: u32,
        count: Option<u32>,
    ) -> Option<Vec<u32>> {
        let host = super::host()?;
        // Words in a record of `kind`.
        let per = match kind {
            0 => 4,
            1 => 5,
            _ => 3,
        };
        let spec = (kind, per as u32 * 4, records);
        let all = per * (records as usize + 1);
        let (buffer, data) = host.shared_buffer(all as u64 * 4).unwrap();
        let data = data.cast::<u32>();
        let list = host.new_list().unwrap();
        // SAFETY: our mapped buffer, written before the list runs and read
        // after it has; a live buffer.
        unsafe {
            for at in 0..all {
                data.add(at).write_volatile(words[at % words.len()]);
            }
            let gpu = buffer.GetGPUVirtualAddress();
            let (counter, word) = host.shared_buffer(4).unwrap();
            word.cast::<u32>().write_volatile(count.unwrap_or(0));
            let counted = count.map(|_| counter.GetGPUVirtualAddress());
            let mut out = None;
            for _ in 0..calls {
                let (clamped, offset) = super::bound(&list, gpu, spec, counted).unwrap();
                super::done(&list, &clamped);
                out = Some((clamped, offset));
            }
            list.close();
            host.execute(&[&list]);
            for at in 0..all {
                assert_eq!(data.add(at).read_volatile(), words[at % words.len()], "the title's copy as written");
            }
            let (out, offset) = out?;
            let mut out_data = std::ptr::null_mut();
            out.Map(0, None, Some(&mut out_data)).unwrap();
            let base = out_data.cast::<u8>().add(offset as usize).cast::<u32>();
            let first = |i: usize| base.add(i * per).read_volatile();
            Some((0..records as usize).map(first).collect())
        }
    }

    /// Unfilled draw records (`>= 1 << 23` vertices/indices) and unfilled
    /// mesh dispatch records (`> 65535` groups, including `0xffffffff`) are
    /// zeroed with trailing words preserved and spend zero budget.
    #[test]
    fn unfilled_draw_and_mesh_records_run_empty_without_spending_budget() {
        let Some(host) = super::host() else {
            return; // no host GPU here
        };
        // 80 draw records (`stride = 32` bytes = 8 words): 76 unfilled
        // (`1 << 23` and `u32::MAX`) followed by 4 valid records.
        const COUNT: usize = 80;
        let (draw_buf, draw_ptr) = host.shared_buffer((COUNT * 32) as u64).unwrap();
        let (mesh_buf, mesh_ptr) = host.shared_buffer((COUNT * 32) as u64).unwrap();
        // SAFETY: our mapped buffers, written before execution and read after.
        unsafe {
            let draws = std::slice::from_raw_parts_mut(draw_ptr.cast::<[u32; 8]>(), COUNT);
            for (i, d) in draws.iter_mut().enumerate() {
                let k = i as u32;
                let head = match i {
                    0..=37 => [u32::MAX, u32::MAX],
                    38..=75 => [1 << 23, 4],
                    76 => [(1 << 23) - 1, 1],
                    _ => [12, 2],
                };
                *d = [head[0], head[1], 100 + k, 200 + k, 0xa000 + k, 0xb000 + k, 0xc000 + k, 0xd000 + k];
            }
            let meshes = std::slice::from_raw_parts_mut(mesh_ptr.cast::<[u32; 8]>(), COUNT);
            for (i, m) in meshes.iter_mut().enumerate() {
                let k = i as u32;
                let gx = match i {
                    0..=37 => u32::MAX,
                    38..=75 => 65_536,
                    76 => 65_535,
                    _ => 7,
                };
                *m = [gx, 9, 5, 0x1000 + k, 0x2000 + k, 0x3000 + k, 0x4000 + k, 0x5000 + k];
            }

            let list = host.new_list().unwrap();
            let (out_d, off_d) = super::bound(&list, draw_buf.GetGPUVirtualAddress(), (0, 32, COUNT as u32), None).unwrap();
            super::done(&list, &out_d);
            list.close();
            host.execute(&[&list]);
            let mut p = std::ptr::null_mut();
            out_d.Map(0, None, Some(&mut p)).unwrap();
            let got_d = std::slice::from_raw_parts(p.cast::<u8>().add(off_d as usize).cast::<[u32; 8]>(), COUNT);
            for (i, r) in got_d.iter().enumerate() {
                let k = i as u32;
                let expected_head = match i {
                    0..=75 => [0, 0],
                    76 => [(1 << 23) - 1, 1],
                    _ => [12, 2],
                };
                assert_eq!(
                    *r,
                    [expected_head[0], expected_head[1], 100 + k, 200 + k, 0xa000 + k, 0xb000 + k, 0xc000 + k, 0xd000 + k],
                    "draw record {i}"
                );
            }
            out_d.Unmap(0, None);

            // On a fresh list, 76 unfilled `0xffffffff` mesh records before 16
            // max-cost `65535` mesh records spend nothing so all 16 run.
            let (mesh16_buf, mesh16_ptr) = host.shared_buffer(((76 + 16) * 12) as u64).unwrap();
            let m16 = std::slice::from_raw_parts_mut(mesh16_ptr.cast::<[u32; 3]>(), 76 + 16);
            for (i, m) in m16.iter_mut().enumerate() {
                *m = if i < 76 { [u32::MAX; 3] } else { [65_535, 4, 4] };
            }
            list.reset();
            let (out_m16, off_m16) = super::bound(&list, mesh16_buf.GetGPUVirtualAddress(), (10, 12, (76 + 16) as u32), None).unwrap();
            super::done(&list, &out_m16);
            list.close();
            host.execute(&[&list]);
            out_m16.Map(0, None, Some(&mut p)).unwrap();
            let got_m16 = std::slice::from_raw_parts(p.cast::<u8>().add(off_m16 as usize).cast::<[u32; 3]>(), 76 + 16);
            assert!(got_m16[..76].iter().all(|&m| m == [0, 0, 0]));
            assert!(got_m16[76..].iter().all(|&m| m == [65_535, 1, 1]));
            out_m16.Unmap(0, None);

            list.reset();
            let (out_m, off_m) = super::bound(&list, mesh_buf.GetGPUVirtualAddress(), (10, 32, COUNT as u32), None).unwrap();
            super::done(&list, &out_m);
            list.close();
            host.execute(&[&list]);
            out_m.Map(0, None, Some(&mut p)).unwrap();
            let got_m = std::slice::from_raw_parts(p.cast::<u8>().add(off_m as usize).cast::<[u32; 8]>(), COUNT);
            for (i, r) in got_m.iter().enumerate() {
                let k = i as u32;
                let head = match i {
                    0..=75 => [0, 0, 0],
                    76 => [65_535, 1, 1],
                    _ => [7, 1, 1],
                };
                assert_eq!(
                    *r,
                    [head[0], head[1], head[2], 0x1000 + k, 0x2000 + k, 0x3000 + k, 0x4000 + k, 0x5000 + k],
                    "mesh record {i}"
                );
            }
            out_m.Unmap(0, None);
        }
    }

    /// Rejected records do not wrap the budget: of 128 max-cost draws (2^26
    /// vertex instances each once clamped) two run, where a wrapping add
    /// let four; of 600 max-cost mesh dispatches 16, not 31.
    #[test]
    fn rejected_records_do_not_wrap_the_budget() {
        let Some(draws) = clamped(0, &[1 << 22, u32::MAX, 0, 0], 128, 1) else {
            return; // no host GPU here
        };
        assert_eq!(draws.iter().filter(|&&count| count != 0).count(), 2);
        let meshes = clamped(10, &[65_535, 1, 1], 600, 1).unwrap();
        assert_eq!(meshes.iter().filter(|&&groups| groups != 0).count(), 16);
    }

    /// The budget goes to the first records, in order: of 200 draws of
    /// 3 * 2^18 vertices the first 170 run (it ends inside the third tile);
    /// small records that all fit all run.
    #[test]
    fn the_budget_goes_to_the_first_records() {
        let Some(draws) = clamped(0, &[3 << 18, 1, 0, 0], 200, 1) else {
            return; // no host GPU here
        };
        assert!(draws[..170].iter().all(|&count| count != 0));
        assert!(draws[170..].iter().all(|&count| count == 0));
        let small = clamped(0, &[1], 1 << 16, 1).unwrap();
        assert!(small.iter().all(|&count| count == 1));
    }

    /// By default (`VICTORY_INDIRECT_REST=0`), a record that does not fit in
    /// the remaining budget runs empty while smaller records after it still
    /// run; with `VICTORY_INDIRECT_REST=1`, it takes what is left so later
    /// records run empty too.
    #[test]
    fn a_record_that_does_not_fit_respects_indirect_rest() {
        let mut words = [1 << 22, 12, 0, 0].repeat(3);
        words.extend([3, 1, 0, 0].repeat(5));
        let Some(draws_default) = clamped(0, &words, 8, 1) else {
            return; // no host GPU here
        };
        let ran_default: Vec<bool> = draws_default.iter().map(|&count| count != 0).collect();
        assert_eq!(
            ran_default,
            [true, true, false, true, true, true, true, true],
            "VICTORY_INDIRECT_REST=0 keeps budget for later fitting records"
        );
        let draws_rest = super::with_indirect_rest(true, || clamped(0, &words, 8, 1).unwrap());
        let ran_rest: Vec<bool> = draws_rest.iter().map(|&count| count != 0).collect();
        assert_eq!(
            ran_rest,
            [true, true, false, false, false, false, false, false],
            "VICTORY_INDIRECT_REST=1 ends the budget on the first overflow"
        );
    }

    /// Two lists clamping into their own argument buffers, run in either
    /// order: every barrier starts from the buffer's current state (the
    /// debug layer's rule; runs light19 and light21 had the title's GPU
    /// heap moved to unordered access under the input assembler), the
    /// title's copy is never transitioned, and all end common.
    #[test]
    fn argument_buffers_move_from_their_current_states() {
        use crate::host::tests::{issued, replay};
        use windows::core::Interface;
        let Some(host) = super::host() else {
            return; // no host GPU here
        };
        let (title, data) = host.shared_buffer(1 << 16).unwrap();
        // SAFETY: our mapped buffer; a live buffer.
        let gpu = unsafe {
            data.cast::<[u32; 3]>().write_volatile([70_000, 1, 1]);
            title.GetGPUVirtualAddress()
        };
        let key = |r: &super::ID3D12Resource| r.as_raw() as usize;
        for a_first in [true, false] {
            let lists = [host.new_list().unwrap(), host.new_list().unwrap()];
            issued();
            let recorded: Vec<_> = lists
                .iter()
                .map(|l| {
                    let (out, off) = super::bound(l, gpu, (2, 12, 1), None).unwrap();
                    super::done(l, &out);
                    l.close();
                    (out, off, issued())
                })
                .collect();
            let order = if a_first { [0, 1] } else { [1, 0] };
            host.execute(&[&lists[order[0]], &lists[order[1]]]);
            let mut run = issued();
            run.extend(order.iter().flat_map(|&i| recorded[i].2.clone()));
            let mut states: std::collections::HashMap<usize, u32> =
                [key(&title), key(&recorded[0].0), key(&recorded[1].0)].into_iter().map(|k| (k, 0)).collect();
            replay(&run, &mut states).unwrap();
            assert!(run.iter().all(|t| t.0 != key(&title)), "the title's copy is never transitioned");
            assert!(states.values().all(|&s| s == 0), "all common again");
            for (out, off, _) in &recorded {
                let mut p = std::ptr::null_mut();
                // SAFETY: a mapped buffer of ours, the lists have run.
                let clamped = unsafe {
                    out.Map(0, None, Some(&mut p)).unwrap();
                    p.cast::<u8>().add(*off as usize).cast::<[u32; 3]>().read_volatile()
                };
                assert_eq!(clamped, [70_000.min(super::widest()), 1, 1]);
            }
            // SAFETY: as above.
            assert_eq!(unsafe { data.cast::<[u32; 3]>().read_volatile() }, [70_000, 1, 1], "the title's copy as written");
        }
    }

    /// The budget is the list's: a second execution of the same records
    /// finds it spent, and they all run empty.
    #[test]
    fn a_spent_budget_stays_spent() {
        let Some(twice) = clamped(1, &[1 << 22, u32::MAX, 0, 0, 0], 64, 2) else {
            return; // no host GPU here
        };
        assert!(twice.iter().all(|&count| count == 0));
    }

    /// The walk stops at the count: of 128 max-cost draws under a count of 3
    /// the first two run and the third runs empty, as they do without one,
    /// and the others are never read (the argument buffer's fresh words stay
    /// zero). A count above the records changes nothing; a count of zero
    /// walks none.
    #[test]
    fn the_walk_stops_at_the_count() {
        let words = [1 << 22, u32::MAX, 0, 0];
        let Some(under) = clamped_under(0, &words, 128, 1, Some(3)) else {
            return; // no host GPU here
        };
        assert_eq!(under[..3], [1 << 22, 1 << 22, 0]);
        assert!(under[3..].iter().all(|&count| count == 0));
        let all = clamped(0, &words, 128, 1).unwrap();
        assert_eq!(under[..3], all[..3]);
        let above = clamped_under(0, &words, 128, 1, Some(1 << 20)).unwrap();
        assert_eq!(above, all);
        let none = clamped_under(0, &words, 128, 1, Some(0)).unwrap();
        assert!(none.iter().all(|&count| count == 0));
    }

    /// Only the records that run are charged to the budget: after a call
    /// under a count of 1 a second finds room for that record again, where
    /// the same draws with no count spent it all on records that never ran.
    #[test]
    fn records_past_the_count_are_not_charged() {
        let words = [1 << 22, u32::MAX, 0, 0];
        let Some(twice) = clamped_under(0, &words, 128, 2, Some(1)) else {
            return; // no host GPU here
        };
        assert_eq!(twice[0], 1 << 22);
        let spent = clamped(0, &words, 128, 2).unwrap();
        assert!(spent.iter().all(|&count| count == 0));
    }

    /// The groups of one indirect dispatch of `groups` x 1 x 1 that run on
    /// the host, through the clamp when `clamped`, else straight from the
    /// record. Each group marks its own word. `None` without a host GPU.
    fn groups_that_run(groups: u32, clamped: bool) -> Option<usize> {
        use super::*;
        use crate::compute::Param;
        const MARK: &str = r#"
RWByteAddressBuffer marks : register(u0);
[numthreads(1, 1, 1)]
void main(uint3 g : SV_GroupID) {
    marks.Store(g.x * 4, 1);
}
"#;
        let host = host()?;
        let (root, pso) =
            crate::compute::pipeline(MARK, s!("marks"), 0, &[Param::Uav { register: 0 }]).unwrap();
        let (record, words) = host.shared_buffer(12).unwrap();
        let (marks, seen) = host.shared_buffer(u64::from(groups) * 4).unwrap();
        let argument = D3D12_INDIRECT_ARGUMENT_DESC {
            Type: D3D12_INDIRECT_ARGUMENT_TYPE_DISPATCH,
            ..Default::default()
        };
        let desc = D3D12_COMMAND_SIGNATURE_DESC {
            ByteStride: 12,
            NumArgumentDescs: 1,
            pArgumentDescs: &argument,
            NodeMask: 0,
        };
        let list = host.new_list().unwrap();
        // SAFETY: our mapped buffers, written before the list runs and read
        // after it has; our pipeline, signature and buffers.
        unsafe {
            let mut signature: Option<ID3D12CommandSignature> = None;
            host.device
                .CreateCommandSignature(&desc, None::<&ID3D12RootSignature>, &mut signature)
                .unwrap();
            let signature = signature.unwrap();
            words.cast::<[u32; 3]>().write_volatile([groups, 1, 1]);
            std::ptr::write_bytes(seen, 0, groups as usize * 4);
            let (arguments, offset) = if clamped {
                bound(&list, record.GetGPUVirtualAddress(), (2, 12, 1), None).unwrap()
            } else {
                list.list.ResourceBarrier(&[transition(
                    &record,
                    D3D12_RESOURCE_STATE_COMMON,
                    D3D12_RESOURCE_STATE_INDIRECT_ARGUMENT,
                )]);
                (record.clone(), 0)
            };
            let l = &list.list;
            l.SetComputeRootSignature(&root);
            l.SetPipelineState(&pso);
            l.SetComputeRootUnorderedAccessView(0, marks.GetGPUVirtualAddress());
            l.ExecuteIndirect(&signature, 1, &arguments, offset, None, 0);
            if clamped {
                done(&list, &arguments);
            } else {
                l.ResourceBarrier(&[transition(
                    &record,
                    D3D12_RESOURCE_STATE_INDIRECT_ARGUMENT,
                    D3D12_RESOURCE_STATE_COMMON,
                )]);
            }
            list.close();
            host.execute(&[&list]);
            let seen = seen.cast::<u32>();
            Some(
                (0..groups as usize)
                    .filter(|&i| seen.add(i).read_volatile() != 0)
                    .count(),
            )
        }
    }

    /// A tile pass of the title (67,440 groups, one class of its 8x4 tiles at
    /// 2560x1440) runs every group where the host runs dispatches that wide,
    /// and its first 65,535 elsewhere. The host runs the widest dispatch the
    /// clamp lets through, so a host that stops running them fails here.
    #[test]
    fn a_wide_tile_pass_runs_every_group_the_host_runs() {
        let Some(ran) = groups_that_run(67_440, true) else {
            return; // no host GPU here
        };
        let widest = super::widest();
        assert_eq!(ran, 67_440.min(widest) as usize);
        assert_eq!(
            groups_that_run(widest, false).unwrap(),
            widest as usize,
            "the host runs the widest dispatch"
        );
        assert_eq!(
            groups_that_run(widest, true).unwrap(),
            widest as usize,
            "the clamp lets it through"
        );
    }

    /// Dispatches wider than [`super::widest`] are cut to 65,535 groups in x
    /// as before, and one whose groups pass 2^20 in all runs empty, also
    /// where x times y would wrap 32 bits.
    #[test]
    fn dispatches_past_the_bounds_are_cut() {
        let widest = super::widest();
        let Some(past) = clamped(2, &[widest + 1, 1, 1], 1, 1) else {
            return; // no host GPU here
        };
        assert_eq!(past, [65_535]);
        assert_eq!(clamped(2, &[widest, 1, 1], 1, 1).unwrap(), [widest]);
        // Past 2^20 groups in all: 65,535 by 17, 2^10 by 2^10 by 2, and
        // 2^20 by 4,096, whose x * y wraps 32 bits to zero.
        assert_eq!(clamped(2, &[65_535, 17, 1], 1, 1).unwrap(), [0]);
        assert_eq!(clamped(2, &[1 << 10, 1 << 10, 2], 1, 1).unwrap(), [0]);
        assert_eq!(clamped(2, &[1 << 20, 4_096, 1], 1, 1).unwrap(), [0]);
    }

    #[test]
    fn batch_clamps_env_toggle_defaults_on() {
        use std::ffi::OsStr;
        assert!(super::batch_clamps_enabled_value(None));
        assert!(super::batch_clamps_enabled_value(Some(OsStr::new("1"))));
        assert!(super::batch_clamps_enabled_value(Some(OsStr::new("yes"))));
        assert!(!super::batch_clamps_enabled_value(Some(OsStr::new("0"))));
        super::with_batch_clamps(false, || {
            assert!(!super::batch_clamps_enabled());
            super::with_batch_clamps(true, || {
                assert!(super::batch_clamps_enabled());
            });
            assert!(!super::batch_clamps_enabled());
        });
    }

    /// What keeps a clamp out of the prelude is named, most binding first:
    /// an inline clamp before it, then a write anywhere, then a write of
    /// its own arguments or count. `VICTORY_CLAMP_CUTS` is on unless `0`.
    #[test]
    fn write_hazard_names_what_blocks_a_clamp() {
        use crate::host::ClampBlock;
        let mut h = crate::host::WriteHazard::default();
        assert_eq!(h.blocker(0x1000, 16, None), None);
        h.note_write(0x1000, 16);
        assert_eq!(h.blocker(0x1000, 16, None), Some(ClampBlock::Overlap));
        assert_eq!(
            h.blocker(0x3000, 16, Some(0x1004)),
            Some(ClampBlock::Overlap)
        );
        assert_eq!(h.blocker(0x3000, 16, None), None);
        h.note_unknown();
        assert_eq!(h.blocker(0x3000, 16, None), Some(ClampBlock::Unknown));
        h.note_inline();
        assert_eq!(h.blocker(0x3000, 16, None), Some(ClampBlock::AfterInline));
        h.clear();
        assert_eq!(h.blocker(0x1000, 16, None), None);

        super::with_clamp_cuts(false, || {
            assert!(!super::clamp_cuts_enabled());
            super::with_clamp_cuts(true, || assert!(super::clamp_cuts_enabled()));
            assert!(!super::clamp_cuts_enabled());
        });
        assert!(crate::switch::on_by_default("VICTORY_CLAMP_CUTS_UNSET_FOR_TEST"));

        let _ = super::take_inline_why();
        super::note_inline(Some(ClampBlock::Unknown));
        super::note_inline(Some(ClampBlock::AfterInline));
        super::note_inline(None);
        let why = super::take_inline_why();
        // Other tests may clamp at the same time: at least these.
        assert!(why[0] >= 1 && why[2] >= 1 && why[3] >= 1, "{why:?}");
    }

    #[test]
    fn write_hazard_tracks_overlaps_coalescing_and_order() {
        let mut h = crate::host::WriteHazard::default();
        assert!(h.can_hoist(0x1000, 32, None));
        h.note_write(0x1000, 0); // zero-length writes nothing
        assert!(h.can_hoist(0x1000, 32, None));

        // Disjoint writes and boundary checks.
        h.note_write(0x2000, 16);
        h.note_write(0x1000, 16);
        assert!(h.can_hoist(0x1010, 0x0FF0, None), "adjacent ranges do not overlap");
        assert!(!h.can_hoist(0x100C, 16, None), "partial head overlap");
        assert!(!h.can_hoist(0x0FF8, 16, None), "partial tail overlap");
        assert!(!h.can_hoist(0x3000, 16, Some(0x200C)), "count word overlap");
        assert!(h.can_hoist(0x3000, 16, Some(0x2010)), "count word right after write");

        // Coalescing bridging write merges [0x1000, 0x1010) and [0x2000, 0x2010).
        h.note_write(0x1010, 0x0FF0);
        assert!(!h.can_hoist(0x1800, 4, None));

        // Inline clamp blocks later hoisting to preserve sequential budget order.
        let mut h2 = crate::host::WriteHazard::default();
        assert!(h2.can_hoist(0x4000, 16, None));
        h2.note_inline();
        assert!(!h2.can_hoist(0x4000, 16, None));
        h2.clear();
        assert!(h2.can_hoist(0x4000, 16, None));

        // Unknown GPU write (bindless heap UAV / memory writes) blocks everything until clear.
        h2.note_unknown();
        assert!(!h2.can_hoist(0x9000, 16, None));
        h2.clear();
        assert!(h2.can_hoist(0x9000, 16, None));
    }

    /// A batch of hoisted clamps in one prelude produces bit-for-bit identical
    /// clamped words and sequential budget accounting as inline clamps across
    /// kinds (draw, indexed, dispatch, mesh), count buffers, and budget
    /// exhaustion.
    #[test]
    fn hoisted_batch_matches_inline_across_kinds_counts_and_budget() {
        let Some(host) = super::host() else {
            return; // no host GPU here
        };
        struct Case {
            kind: u32,
            stride: u32,
            records: u32,
            words: Vec<u32>,
            count: Option<u32>,
        }
        let cases = vec![
            Case {
                kind: 0,
                stride: 16,
                records: 3,
                words: vec![
                    12, 2, 0, 0,
                    1 << 25, 10, 1, 2, // vertex count clamped to 2^24, instances clamped to 4
                    5, 1, 3, 4,
                ],
                count: None,
            },
            Case {
                kind: 1,
                stride: 20,
                records: 4,
                words: vec![
                    100, 3, 7, 8, 9,
                    u32::MAX, u32::MAX, 1, 2, 3, // past-count: untouched
                    u32::MAX, u32::MAX, 4, 5, 6,
                    u32::MAX, u32::MAX, 7, 8, 9,
                ],
                count: Some(1),
            },
            Case {
                kind: 2,
                stride: 12,
                records: 2,
                words: vec![
                    super::widest() + 5, 1, 1, // cut to 65535
                    128, 2, 1,
                ],
                count: None,
            },
            Case {
                kind: 10,
                stride: 12,
                records: 2,
                words: vec![
                    100, 5, 3, // y,z clamped to 1
                    70_000, 1, 1, // x clamped to 65535
                ],
                count: None,
            },
            // Exhausts the remaining list budget across the two records.
            Case {
                kind: 0,
                stride: 16,
                records: 4,
                words: vec![
                    1 << 22, 16, 0, 0,
                    1 << 22, 16, 0, 0,
                    3, 1, 0, 0,
                    3, 1, 0, 0,
                ],
                count: None,
            },
        ];

        let mut inputs = Vec::new();
        for case in &cases {
            let (src, src_ptr) = host.shared_buffer((case.words.len() * 4) as u64).unwrap();
            let (cnt, cnt_ptr) = host.shared_buffer(4).unwrap();
            // SAFETY: our mapped buffers, written before submission.
            unsafe {
                for (i, &w) in case.words.iter().enumerate() {
                    src_ptr.cast::<u32>().add(i).write_volatile(w);
                }
                cnt_ptr.cast::<u32>().write_volatile(case.count.unwrap_or(0));
            }
            inputs.push((src, cnt));
        }

        // 1. Run hoisted in one prelude.
        let _ = super::take_clamps();
        let hoisted_list = host.new_list().unwrap();
        let mut hoisted_outs = Vec::new();
        for (case, (src, cnt)) in cases.iter().zip(&inputs) {
            // SAFETY: live buffers.
            let (src_gpu, cnt_gpu) = unsafe {
                (
                    src.GetGPUVirtualAddress(),
                    case.count.map(|_| cnt.GetGPUVirtualAddress()),
                )
            };
            let out = super::enqueue(
                &hoisted_list,
                src_gpu,
                (case.kind, case.stride, case.records),
                cnt_gpu,
            )
            .expect("hoisted clamp enqueued");
            hoisted_outs.push(out);
        }
        hoisted_list.close();
        host.execute(&[&hoisted_list]);
        let [hoisted_n, inline_n] = super::take_clamps();
        // The counters are global, so a test running at the same time may
        // add to them: the words compared below are the exact check.
        assert!(hoisted_n >= cases.len() as u64, "{hoisted_n} {inline_n}");

        let mut hoisted_words: Vec<Vec<u32>> = Vec::new();
        for (case, (buf, offset)) in cases.iter().zip(&hoisted_outs) {
            let mut ptr = std::ptr::null_mut();
            // SAFETY: mapped shared buffer, list has run.
            unsafe {
                buf.Map(0, None, Some(&mut ptr)).unwrap();
                let base = ptr.cast::<u8>().add(*offset as usize).cast::<u32>();
                let n = case.count.map_or(case.records, |c| c.min(case.records)) as usize
                    * (case.stride as usize / 4);
                hoisted_words.push((0..n).map(|i| base.add(i).read_volatile()).collect());
                buf.Unmap(0, None);
            }
        }

        // 2. Run inline in one list, copying each clamped slice out before the
        //    next inline clamp overwrites offset 0.
        let inline_list = host.new_list().unwrap();
        let mut readbacks = Vec::new();
        for (case, (src, cnt)) in cases.iter().zip(&inputs) {
            // SAFETY: live buffers and open list.
            unsafe {
                let src_gpu = src.GetGPUVirtualAddress();
                let cnt_gpu = case.count.map(|_| cnt.GetGPUVirtualAddress());
                let (out, off) = super::bound(
                    &inline_list,
                    src_gpu,
                    (case.kind, case.stride, case.records),
                    cnt_gpu,
                )
                .expect("inline clamp");
                let bytes = u64::from(case.stride) * u64::from(case.records);
                let (copy, copy_ptr) = host.shared_buffer(bytes).unwrap();
                inline_list.list.ResourceBarrier(&[super::transition(
                    &out,
                    super::D3D12_RESOURCE_STATE_INDIRECT_ARGUMENT,
                    super::D3D12_RESOURCE_STATE_COPY_SOURCE,
                )]);
                inline_list.list.CopyBufferRegion(&copy, 0, &out, off, bytes);
                inline_list.list.ResourceBarrier(&[super::transition(
                    &out,
                    super::D3D12_RESOURCE_STATE_COPY_SOURCE,
                    super::D3D12_RESOURCE_STATE_INDIRECT_ARGUMENT,
                )]);
                super::done(&inline_list, &out);
                readbacks.push((copy, copy_ptr));
            }
        }
        inline_list.close();
        host.execute(&[&inline_list]);
        let [hoisted_n, inline_n] = super::take_clamps();
        assert!(inline_n >= cases.len() as u64, "{hoisted_n} {inline_n}");

        for (i, (case, (_buf, ptr))) in cases.iter().zip(&readbacks).enumerate() {
            let n = case.count.map_or(case.records, |c| c.min(case.records)) as usize
                * (case.stride as usize / 4);
            // SAFETY: mapped shared buffer, list has run.
            let inline: Vec<u32> =
                unsafe { (0..n).map(|k| ptr.cast::<u32>().add(k).read_volatile()).collect() };
            assert_eq!(hoisted_words[i], inline, "case {i} kind {}", case.kind);
        }
        assert_eq!(host.log_messages(), None);
    }

    /// Hoisted argument buffers transition cleanly COMMON -> UAV ->
    /// INDIRECT_ARGUMENT in prelude and INDIRECT_ARGUMENT -> COMMON at close,
    /// in any submission order, across re-submission of a closed list, and
    /// across list reset and reuse.
    #[test]
    fn hoisted_argument_buffers_transition_resubmit_and_reset_cleanly() {
        use crate::host::tests::{issued, replay};
        use windows::core::Interface;
        let Some(host) = super::host() else {
            return; // no host GPU here
        };
        let (title, data) = host.shared_buffer(64).unwrap();
        // SAFETY: mapped buffer.
        let gpu = unsafe {
            data.cast::<[u32; 3]>().write_volatile([123, 1, 1]);
            title.GetGPUVirtualAddress()
        };
        let key = |r: &super::ID3D12Resource| r.as_raw() as usize;

        let list = host.new_list().unwrap();
        let _ = issued();
        let (out1, off1) = super::enqueue(&list, gpu, (2, 12, 1), None).unwrap();
        let (out2, off2) = super::enqueue(&list, gpu, (2, 12, 1), None).unwrap();
        assert_eq!(key(&out1), key(&out2), "small clamps pack into one chunk");
        assert_ne!(off1, off2, "hoisted clamps get disjoint offsets");
        list.close();
        let close_transitions = issued();

        // Execute once: prelude transitions happen during execute().
        host.execute(&[&list]);
        let prelude_transitions = issued();
        let mut run = prelude_transitions;
        run.extend_from_slice(&close_transitions);
        let mut states: std::collections::HashMap<usize, u32> =
            [(key(&title), 0), (key(&out1), 0)].into_iter().collect();
        replay(&run, &mut states).unwrap();
        assert!(run.iter().all(|t| t.0 != key(&title)), "title copy never transitioned");
        assert_eq!(states[&key(&out1)], 0, "hoisted buffer ends COMMON");

        // Update source and re-submit the SAME closed list without reset.
        // SAFETY: mapped buffer.
        unsafe {
            data.cast::<[u32; 3]>().write_volatile([456, 2, 1]);
        }
        host.execute(&[&list]);
        let mut ptr = std::ptr::null_mut();
        // SAFETY: mapped buffer, list has run.
        unsafe {
            out1.Map(0, None, Some(&mut ptr)).unwrap();
            let got1 = ptr.cast::<u8>().add(off1 as usize).cast::<[u32; 3]>().read_volatile();
            let got2 = ptr.cast::<u8>().add(off2 as usize).cast::<[u32; 3]>().read_volatile();
            out1.Unmap(0, None);
            assert_eq!(got1, [456, 2, 1]);
            assert_eq!(got2, [456, 2, 1]);
        }

        // Reset the list, reuse the chunk for new data, and execute again.
        list.reset();
        // SAFETY: mapped buffer.
        unsafe {
            data.cast::<[u32; 3]>().write_volatile([789, 1, 1]);
        }
        let (reused, off_reused) = super::enqueue(&list, gpu, (2, 12, 1), None).unwrap();
        assert_eq!(key(&reused), key(&out1), "chunk is reused after reset");
        assert_eq!(off_reused, 0, "bump cursor resets to 0");
        list.close();
        host.execute(&[&list]);
        // SAFETY: mapped buffer, list has run.
        unsafe {
            reused.Map(0, None, Some(&mut ptr)).unwrap();
            let got = ptr.cast::<[u32; 3]>().read_volatile();
            reused.Unmap(0, None);
            assert_eq!(got, [789, 1, 1]);
        }
        assert_eq!(host.log_messages(), None);
    }

    /// Multiple hoisted indirect dispatches execute in one list, while a
    /// same-list GPU write into an argument buffer disables hoisting for that
    /// buffer (falling back to inline clamping) without blocking disjoint
    /// buffers, and a mixed hoisted-prefix + inline-suffix list shares its
    /// budget sequentially.
    #[test]
    fn hoisted_indirect_execution_and_same_list_write_hazard() {
        use super::*;
        use crate::compute::Param;
        const MARK: &str = r#"
RWByteAddressBuffer marks : register(u0);
cbuffer Offset : register(b0) { uint base; };
[numthreads(1, 1, 1)]
void main(uint3 g : SV_GroupID) {
    marks.Store((base + g.x) * 4, 1);
}
"#;
        const WRITE_ARGS: &str = r#"
RWByteAddressBuffer outArgs : register(u0);
cbuffer Val : register(b0) { uint gx; uint gy; uint gz; };
[numthreads(1, 1, 1)]
void main() {
    outArgs.Store3(0, uint3(gx, gy, gz));
}
"#;
        let Some(host) = host() else {
            return; // no host GPU here
        };
        let (mark_root, mark_pso) = crate::compute::pipeline(
            MARK,
            s!("mark_batch"),
            0,
            &[Param::Uav { register: 0 }, Param::Constants { register: 0, count: 1 }],
        )
        .unwrap();
        let (write_root, write_pso) = crate::compute::pipeline(
            WRITE_ARGS,
            s!("write_args"),
            0,
            &[Param::Uav { register: 0 }, Param::Constants { register: 0, count: 3 }],
        )
        .unwrap();
        let argument = D3D12_INDIRECT_ARGUMENT_DESC {
            Type: D3D12_INDIRECT_ARGUMENT_TYPE_DISPATCH,
            ..Default::default()
        };
        let desc = D3D12_COMMAND_SIGNATURE_DESC {
            ByteStride: 12,
            NumArgumentDescs: 1,
            pArgumentDescs: &argument,
            NodeMask: 0,
        };
        // SAFETY: plain command signature creation.
        let signature = unsafe {
            let mut sig: Option<ID3D12CommandSignature> = None;
            host.device
                .CreateCommandSignature(&desc, None::<&ID3D12RootSignature>, &mut sig)
                .unwrap();
            sig.unwrap()
        };

        // Part 1: 4 hoisted indirect dispatches in one list, no PSO re-bind between them.
        let (args_buf, args_ptr) = host.shared_buffer(64).unwrap();
        let (marks_buf, marks_ptr) = host.shared_buffer(64 * 4).unwrap();
        let group_counts = [3u32, 5, 2, 4];
        let bases = [0u32, 8, 16, 24];
        let list = host.new_list().unwrap();
        // SAFETY: mapped buffers, open list, live PSO/root/signature.
        unsafe {
            std::ptr::write_bytes(marks_ptr, 0, 64 * 4);
            for (i, &g) in group_counts.iter().enumerate() {
                args_ptr.cast::<u32>().add(i * 4).write_volatile(g);
                args_ptr.cast::<u32>().add(i * 4 + 1).write_volatile(1);
                args_ptr.cast::<u32>().add(i * 4 + 2).write_volatile(1);
            }
            let args_gpu = args_buf.GetGPUVirtualAddress();
            let l = &list.list;
            l.SetComputeRootSignature(&mark_root);
            l.SetPipelineState(&mark_pso);
            l.SetComputeRootUnorderedAccessView(0, marks_buf.GetGPUVirtualAddress());
            for (i, &base) in bases.iter().enumerate() {
                let (clamped, off) =
                    enqueue(&list, args_gpu + (i as u64) * 16, (2, 12, 1), None).unwrap();
                l.SetComputeRoot32BitConstants(1, 1, (&raw const base).cast(), 0);
                l.ExecuteIndirect(&signature, 1, &clamped, off, None, 0);
            }
            list.close();
            host.execute(&[&list]);
            for (&g, &base) in group_counts.iter().zip(&bases) {
                for k in 0..8u32 {
                    let expected = u32::from(k < g);
                    assert_eq!(
                        marks_ptr.cast::<u32>().add((base + k) as usize).read_volatile(),
                        expected,
                        "hoisted dispatch base={base} k={k}"
                    );
                }
            }
        }

        // Part 2: Same-list GPU write into shadow buffer A blocks hoisting for A
        // (and runs inline with the freshly written args), while disjoint B is
        // still hoistable before the inline clamp runs.
        let backing = vec![0u8; 4096];
        let addr_a = backing.as_ptr() as u64;
        let addr_b = addr_a + 1024;
        crate::shadow::register(addr_a as usize, addr_a, 4096);
        let shadow_a = crate::shadow::find(addr_a, 12).expect("registered shadow");
        let list2 = host.new_list().unwrap();
        // SAFETY: mapped buffers and open list.
        unsafe {
            std::ptr::write_bytes(marks_ptr, 0, 64 * 4);
            // Pre-populate shadow A with stale [1, 1, 1] and shadow B with [4, 1, 1].
            let mut ptr = std::ptr::null_mut();
            shadow_a.buffer.Map(0, None, Some(&mut ptr)).unwrap();
            ptr.cast::<[u32; 3]>().write_volatile([1, 1, 1]);
            ptr.cast::<u8>().add(1024).cast::<[u32; 3]>().write_volatile([4, 1, 1]);
            shadow_a.buffer.Unmap(0, None);

            // GPU dispatch writes [7, 1, 1] into shadow A.
            shadow_a.gpu_writes(&list2, addr_a, 12, false);
            let vals = [7u32, 1, 1];
            let l = &list2.list;
            l.SetComputeRootSignature(&write_root);
            l.SetPipelineState(&write_pso);
            l.SetComputeRootUnorderedAccessView(0, shadow_a.gpu(addr_a));
            l.SetComputeRoot32BitConstants(1, 3, vals.as_ptr().cast(), 0);
            l.Dispatch(1, 1, 1);
            l.ResourceBarrier(&[uav_barrier()]);

            // Disjoint buffer B can still hoist into prelude; buffer A cannot!
            assert!(list2.can_hoist_clamp(addr_b, 12, None));
            assert!(!list2.can_hoist_clamp(addr_a, 12, None));

            // Hoist B first (prefix), then run A inline (suffix).
            let (clamped_b, off_b) =
                enqueue(&list2, shadow_a.gpu(addr_b), (2, 12, 1), None).unwrap();
            l.SetComputeRootSignature(&mark_root);
            l.SetPipelineState(&mark_pso);
            l.SetComputeRootUnorderedAccessView(0, marks_buf.GetGPUVirtualAddress());
            let base_b = 0u32;
            l.SetComputeRoot32BitConstants(1, 1, (&raw const base_b).cast(), 0);
            l.ExecuteIndirect(&signature, 1, &clamped_b, off_b, None, 0);

            let (clamped_a, off_a) = bound(&list2, shadow_a.gpu(addr_a), (2, 12, 1), None).unwrap();
            list2.note_inline_clamp();
            // After an inline clamp, even disjoint B must stay inline for budget order.
            assert!(!list2.can_hoist_clamp(addr_b, 12, None));
            l.SetComputeRootSignature(&mark_root);
            l.SetPipelineState(&mark_pso);
            l.SetComputeRootUnorderedAccessView(0, marks_buf.GetGPUVirtualAddress());
            let base_a = 16u32;
            l.SetComputeRoot32BitConstants(1, 1, (&raw const base_a).cast(), 0);
            l.ExecuteIndirect(&signature, 1, &clamped_a, off_a, None, 0);
            done(&list2, &clamped_a);

            list2.close();
            host.execute(&[&list2]);
            for k in 0..8u32 {
                assert_eq!(
                    marks_ptr.cast::<u32>().add(k as usize).read_volatile(),
                    u32::from(k < 4),
                    "hoisted disjoint B k={k}"
                );
                assert_eq!(
                    marks_ptr.cast::<u32>().add((16 + k) as usize).read_volatile(),
                    u32::from(k < 7),
                    "inline hazard A saw GPU-written [7,1,1] k={k}"
                );
            }
        }
        crate::shadow::unregister(addr_a as usize, addr_a);
        assert_eq!(host.log_messages(), None);
    }

    /// The ordering a clamp cut (`crate::list::cut_for_clamp`) relies on: a
    /// host list writes an argument record on the GPU, so it cannot hoist
    /// that record's clamp; the next host list, started for it, hoists it
    /// (twice) into its prelude, and submitted together the prelude reads
    /// what the list before it wrote, not the stale record. A cut list that
    /// hoisted only one clamp does not pay for another cut.
    #[test]
    fn a_clamp_cut_prelude_sees_the_write_of_the_list_before() {
        use super::*;
        use crate::compute::Param;
        use crate::host::ClampBlock;
        const MARK: &str = r#"
RWByteAddressBuffer marks : register(u0);
cbuffer Offset : register(b0) { uint base; };
[numthreads(1, 1, 1)]
void main(uint3 g : SV_GroupID) {
    marks.Store((base + g.x) * 4, 1);
}
"#;
        const WRITE_ARGS: &str = r#"
RWByteAddressBuffer outArgs : register(u0);
cbuffer Val : register(b0) { uint gx; uint gy; uint gz; };
[numthreads(1, 1, 1)]
void main() {
    outArgs.Store3(0, uint3(gx, gy, gz));
}
"#;
        let Some(host) = host() else {
            return; // no host GPU here
        };
        let (mark_root, mark_pso) = crate::compute::pipeline(
            MARK,
            s!("cut_mark"),
            0,
            &[Param::Uav { register: 0 }, Param::Constants { register: 0, count: 1 }],
        )
        .unwrap();
        let (write_root, write_pso) = crate::compute::pipeline(
            WRITE_ARGS,
            s!("cut_write_args"),
            0,
            &[Param::Uav { register: 0 }, Param::Constants { register: 0, count: 3 }],
        )
        .unwrap();
        let argument = D3D12_INDIRECT_ARGUMENT_DESC {
            Type: D3D12_INDIRECT_ARGUMENT_TYPE_DISPATCH,
            ..Default::default()
        };
        let desc = D3D12_COMMAND_SIGNATURE_DESC {
            ByteStride: 12,
            NumArgumentDescs: 1,
            pArgumentDescs: &argument,
            NodeMask: 0,
        };
        // SAFETY: plain command signature creation.
        let signature = unsafe {
            let mut sig: Option<ID3D12CommandSignature> = None;
            host.device
                .CreateCommandSignature(&desc, None::<&ID3D12RootSignature>, &mut sig)
                .unwrap();
            sig.unwrap()
        };
        let (marks_buf, marks_ptr) = host.shared_buffer(64 * 4).unwrap();
        let backing = vec![0u8; 4096];
        let addr = backing.as_ptr() as u64;
        crate::shadow::register(addr as usize, addr, 4096);
        let shadow = crate::shadow::find(addr, 12).expect("registered shadow");
        let writer = host.new_list().unwrap();
        let cut = host.new_list().unwrap();
        // SAFETY: mapped buffers, open lists, live pipelines and signature.
        unsafe {
            std::ptr::write_bytes(marks_ptr, 0, 64 * 4);
            // A stale record of one group in the buffer's host copy.
            let mut ptr = std::ptr::null_mut();
            shadow.buffer.Map(0, None, Some(&mut ptr)).unwrap();
            ptr.cast::<[u32; 3]>().write_volatile([1, 1, 1]);
            shadow.buffer.Unmap(0, None);

            // The first list writes [6, 1, 1] over it.
            shadow.gpu_writes(&writer, addr, 12, false);
            let vals = [6u32, 1, 1];
            let l = &writer.list;
            l.SetComputeRootSignature(&write_root);
            l.SetPipelineState(&write_pso);
            l.SetComputeRootUnorderedAccessView(0, shadow.gpu(addr));
            l.SetComputeRoot32BitConstants(1, 3, vals.as_ptr().cast(), 0);
            l.Dispatch(1, 1, 1);
            assert_eq!(writer.clamp_blocker(addr, 12, None), Some(ClampBlock::Overlap));
            assert!(writer.clamp_cut_pays(), "a list not started for a clamp may cut");
            writer.close();

            // The next list, started for the clamp, hoists it and one more.
            cut.mark_clamp_cut();
            assert_eq!(cut.clamp_blocker(addr, 12, None), None);
            let l = &cut.list;
            l.SetComputeRootSignature(&mark_root);
            l.SetPipelineState(&mark_pso);
            l.SetComputeRootUnorderedAccessView(0, marks_buf.GetGPUVirtualAddress());
            for (k, base) in [0u32, 16].into_iter().enumerate() {
                let (clamped, off) = enqueue(&cut, shadow.gpu(addr), (2, 12, 1), None).unwrap();
                assert_eq!(cut.clamp_cut_pays(), k == 1, "pays once it hoisted two");
                l.SetComputeRoot32BitConstants(1, 1, (&raw const base).cast(), 0);
                l.ExecuteIndirect(&signature, 1, &clamped, off, None, 0);
            }
            cut.close();
            host.execute(&[&writer, &cut]);
            for base in [0u32, 16] {
                for k in 0..8u32 {
                    assert_eq!(
                        marks_ptr.cast::<u32>().add((base + k) as usize).read_volatile(),
                        u32::from(k < 6),
                        "the cut list's prelude read the written [6, 1, 1]: base {base} group {k}"
                    );
                }
            }
        }
        cut.reset();
        assert!(cut.clamp_cut_pays(), "a reset list is no longer a cut one");
        writer.reset();
        crate::shadow::unregister(addr as usize, addr);
        assert_eq!(host.log_messages(), None);
    }

    /// A list that mixes a hoisted prefix (in prelude) and an inline suffix
    /// (in the main list) shares the same list budget sequentially across the
    /// prelude -> main list boundary.
    #[test]
    fn hoisted_prefix_and_inline_suffix_share_budget_in_order() {
        let Some(host) = super::host() else {
            return; // no host GPU here
        };
        // Each record asks for 2^22 * 16 = 2^26 vertex instances (half the 2^27
        // list budget). Two hoisted records exhaust the budget completely, so
        // an inline record after them in the same list must be zeroed.
        let (src, ptr) = host.shared_buffer(32).unwrap();
        let (inline_src, inline_ptr) = host.shared_buffer(16).unwrap();
        // SAFETY: mapped buffers.
        let (src_gpu, inline_gpu) = unsafe {
            let p = ptr.cast::<u32>();
            for i in 0..2 {
                p.add(i * 4).write_volatile(1 << 22);
                p.add(i * 4 + 1).write_volatile(16);
                p.add(i * 4 + 2).write_volatile(0);
                p.add(i * 4 + 3).write_volatile(0);
            }
            let ip = inline_ptr.cast::<u32>();
            ip.add(0).write_volatile(3);
            ip.add(1).write_volatile(1);
            ip.add(2).write_volatile(0);
            ip.add(3).write_volatile(0);
            (src.GetGPUVirtualAddress(), inline_src.GetGPUVirtualAddress())
        };
        let list = host.new_list().unwrap();
        let (hoisted_buf, hoisted_off) = super::enqueue(&list, src_gpu, (0, 16, 2), None).unwrap();
        let (inline_buf, inline_off) = super::bound(&list, inline_gpu, (0, 16, 1), None).unwrap();
        list.note_inline_clamp();
        super::done(&list, &inline_buf);
        list.close();
        host.execute(&[&list]);

        let mut p = std::ptr::null_mut();
        // SAFETY: mapped shared buffers, list has run.
        unsafe {
            hoisted_buf.Map(0, None, Some(&mut p)).unwrap();
            let base = p.cast::<u8>().add(hoisted_off as usize).cast::<u32>();
            assert_eq!(base.add(0).read_volatile(), 1 << 22, "hoisted record 0 fits");
            assert_eq!(base.add(4).read_volatile(), 1 << 22, "hoisted record 1 fits");
            hoisted_buf.Unmap(0, None);

            inline_buf.Map(0, None, Some(&mut p)).unwrap();
            let inline_base = p.cast::<u8>().add(inline_off as usize).cast::<u32>();
            assert_eq!(
                inline_base.read_volatile(),
                0,
                "inline suffix sees budget spent by hoisted prelude"
            );
            inline_buf.Unmap(0, None);
        }
        assert_eq!(host.log_messages(), None);
    }

    /// `D3D12_FEATURE_D3D12_OPTIONS12::EnhancedBarriersSupported` and cached
    /// `ID3D12GraphicsCommandList7` on both `list` and `prelude` are available
    /// on D3DMetal 4.0b3, and `VICTORY_CLAMP_SYNC` defaults off until enabled.
    #[test]
    fn enhanced_barriers_supported_and_command_list7_available() {
        use std::ffi::OsStr;
        assert!(!super::clamp_sync_enabled_value(None));
        assert!(!super::clamp_sync_enabled_value(Some(OsStr::new("0"))));
        assert!(super::clamp_sync_enabled_value(Some(OsStr::new("1"))));
        super::with_clamp_sync(true, || {
            assert!(super::clamp_sync_enabled());
            super::with_clamp_sync(false, || {
                assert!(!super::clamp_sync_enabled());
            });
            assert!(super::clamp_sync_enabled());
        });

        let Some(host) = super::host() else {
            return; // no host GPU here
        };
        assert!(
            host.enhanced_barriers(),
            "host must report D3D12_OPTIONS12::EnhancedBarriersSupported"
        );
        let list = host.new_list().unwrap();
        assert!(
            list.list7.is_some(),
            "HostList::list7 must be cached when enhanced barriers are supported"
        );
        assert!(
            list.prelude7.is_some(),
            "HostList::prelude7 must be cached when enhanced barriers are supported"
        );
        assert_eq!(host.log_messages(), None);
    }

    /// Narrow clamp sync (`VICTORY_CLAMP_SYNC=1`) produces bit-exact hoisted
    /// and inline clamped words and exact budget accounting matching legacy
    /// barriers across all indirect kinds, count buffers, and budget
    /// exhaustion, while emitting zero legacy resource transitions and giving
    /// each inline clamp a disjoint bump-allocated slice.
    #[test]
    fn narrow_clamp_sync_inline_and_hoisted_match_legacy() {
        use crate::host::tests::issued;
        use windows::core::Interface;
        let Some(host) = super::host() else {
            return; // no host GPU here
        };
        assert!(host.enhanced_barriers());

        struct Case {
            kind: u32,
            stride: u32,
            records: u32,
            words: Vec<u32>,
            count: Option<u32>,
        }
        let cases = vec![
            Case {
                kind: 0,
                stride: 16,
                records: 3,
                words: vec![
                    12, 2, 0, 0,
                    1 << 25, 10, 1, 2,
                    5, 1, 3, 4,
                ],
                count: None,
            },
            Case {
                kind: 1,
                stride: 20,
                records: 4,
                words: vec![
                    100, 3, 7, 8, 9,
                    u32::MAX, u32::MAX, 1, 2, 3,
                    u32::MAX, u32::MAX, 4, 5, 6,
                    u32::MAX, u32::MAX, 7, 8, 9,
                ],
                count: Some(1),
            },
            Case {
                kind: 2,
                stride: 12,
                records: 2,
                words: vec![
                    super::widest() + 5, 1, 1,
                    128, 2, 1,
                ],
                count: None,
            },
            Case {
                kind: 10,
                stride: 12,
                records: 2,
                words: vec![
                    100, 5, 3,
                    70_000, 1, 1,
                ],
                count: None,
            },
            Case {
                kind: 0,
                stride: 16,
                records: 4,
                words: vec![
                    1 << 22, 16, 0, 0,
                    1 << 22, 16, 0, 0,
                    3, 1, 0, 0,
                    3, 1, 0, 0,
                ],
                count: None,
            },
        ];

        let mut inputs = Vec::new();
        for case in &cases {
            let (src, src_ptr) = host.shared_buffer((case.words.len() * 4) as u64).unwrap();
            let (cnt, cnt_ptr) = host.shared_buffer(4).unwrap();
            // SAFETY: our mapped buffers, written before submission.
            unsafe {
                for (i, &w) in case.words.iter().enumerate() {
                    src_ptr.cast::<u32>().add(i).write_volatile(w);
                }
                cnt_ptr.cast::<u32>().write_volatile(case.count.unwrap_or(0));
            }
            inputs.push((src, cnt));
        }

        let run_pair = |narrow: bool| -> (Vec<Vec<u32>>, Vec<Vec<u32>>, u32, u32) {
            super::with_clamp_sync(narrow, || {
                // 1. Hoisted batch in prelude.
                let _ = issued();
                let hoisted_list = host.new_list().unwrap();
                let mut hoisted_outs = Vec::new();
                for (case, (src, cnt)) in cases.iter().zip(&inputs) {
                    // SAFETY: live buffers.
                    let (src_gpu, cnt_gpu) = unsafe {
                        (
                            src.GetGPUVirtualAddress(),
                            case.count.map(|_| cnt.GetGPUVirtualAddress()),
                        )
                    };
                    let out = super::enqueue(
                        &hoisted_list,
                        src_gpu,
                        (case.kind, case.stride, case.records),
                        cnt_gpu,
                    )
                    .expect("hoisted clamp enqueued");
                    hoisted_outs.push(out);
                }
                hoisted_list.close();
                let close_t = issued();
                host.execute(&[&hoisted_list]);
                let exec_t = issued();
                if narrow {
                    assert!(close_t.is_empty(), "narrow hoisted close emitted legacy transitions: {close_t:?}");
                    assert!(exec_t.is_empty(), "narrow hoisted prelude emitted legacy transitions: {exec_t:?}");
                } else {
                    assert!(!close_t.is_empty() && !exec_t.is_empty());
                }
                let hoisted_budget = hoisted_list.budget_used().expect("hoisted budget written");
                let mut hoisted_words = Vec::new();
                for (case, (buf, offset)) in cases.iter().zip(&hoisted_outs) {
                    let mut ptr = std::ptr::null_mut();
                    // SAFETY: mapped shared buffer, list has run.
                    unsafe {
                        buf.Map(0, None, Some(&mut ptr)).unwrap();
                        let base = ptr.cast::<u8>().add(*offset as usize).cast::<u32>();
                        let n = case.count.map_or(case.records, |c| c.min(case.records)) as usize
                            * (case.stride as usize / 4);
                        hoisted_words.push((0..n).map(|i| base.add(i).read_volatile()).collect());
                        buf.Unmap(0, None);
                    }
                }

                // 2. Inline clamps in one list.
                let _ = issued();
                let inline_list = host.new_list().unwrap();
                let mut inline_slices = Vec::new();
                let mut legacy_readbacks = Vec::new();
                for (case, (src, cnt)) in cases.iter().zip(&inputs) {
                    // SAFETY: live buffers and open list.
                    unsafe {
                        let src_gpu = src.GetGPUVirtualAddress();
                        let cnt_gpu = case.count.map(|_| cnt.GetGPUVirtualAddress());
                        let (out, off) = super::bound(
                            &inline_list,
                            src_gpu,
                            (case.kind, case.stride, case.records),
                            cnt_gpu,
                        )
                        .expect("inline clamp");
                        if !narrow {
                            let bytes = u64::from(case.stride) * u64::from(case.records);
                            let (copy, copy_ptr) = host.shared_buffer(bytes).unwrap();
                            inline_list.list.ResourceBarrier(&[super::transition(
                                &out,
                                super::D3D12_RESOURCE_STATE_INDIRECT_ARGUMENT,
                                super::D3D12_RESOURCE_STATE_COPY_SOURCE,
                            )]);
                            inline_list.list.CopyBufferRegion(&copy, 0, &out, off, bytes);
                            inline_list.list.ResourceBarrier(&[super::transition(
                                &out,
                                super::D3D12_RESOURCE_STATE_COPY_SOURCE,
                                super::D3D12_RESOURCE_STATE_INDIRECT_ARGUMENT,
                            )]);
                            legacy_readbacks.push((copy, copy_ptr));
                        }
                        super::done(&inline_list, &out);
                        inline_slices.push((out, off));
                    }
                }
                inline_list.close();
                let inline_rec_t = issued();
                host.execute(&[&inline_list]);
                let inline_exec_t = issued();
                if narrow {
                    assert!(
                        inline_rec_t.is_empty(),
                        "narrow inline clamps emitted legacy transitions: {inline_rec_t:?}"
                    );
                    assert!(
                        inline_exec_t.is_empty(),
                        "narrow inline execute emitted legacy transitions: {inline_exec_t:?}"
                    );
                    // All inline slices pack into one chunk at strictly increasing
                    // 16-byte-aligned disjoint offsets (zero WAR hazard).
                    let chunk0 = inline_slices[0].0.as_raw() as usize;
                    for pair in inline_slices.windows(2) {
                        assert_eq!(pair[0].0.as_raw() as usize, chunk0);
                        assert_eq!(pair[1].0.as_raw() as usize, chunk0);
                        assert_eq!(pair[1].1 % 16, 0);
                        assert!(pair[1].1 > pair[0].1, "disjoint inline offsets");
                    }
                }
                let inline_budget = inline_list.budget_used().expect("inline budget written");
                let mut inline_words = Vec::new();
                if narrow {
                    for (case, (buf, off)) in cases.iter().zip(&inline_slices) {
                        let mut ptr = std::ptr::null_mut();
                        // SAFETY: mapped shared buffer, list has run; disjoint slice preserved.
                        unsafe {
                            buf.Map(0, None, Some(&mut ptr)).unwrap();
                            let base = ptr.cast::<u8>().add(*off as usize).cast::<u32>();
                            let n = case.count.map_or(case.records, |c| c.min(case.records)) as usize
                                * (case.stride as usize / 4);
                            inline_words.push((0..n).map(|k| base.add(k).read_volatile()).collect());
                            buf.Unmap(0, None);
                        }
                    }
                } else {
                    for (case, (_buf, ptr)) in cases.iter().zip(&legacy_readbacks) {
                        let n = case.count.map_or(case.records, |c| c.min(case.records)) as usize
                            * (case.stride as usize / 4);
                        // SAFETY: mapped shared buffer, list has run.
                        let got: Vec<u32> = unsafe {
                            (0..n).map(|k| ptr.cast::<u32>().add(k).read_volatile()).collect()
                        };
                        inline_words.push(got);
                    }
                }
                (hoisted_words, inline_words, hoisted_budget, inline_budget)
            })
        };

        let (leg_h, leg_i, leg_hb, leg_ib) = run_pair(false);
        let (nar_h, nar_i, nar_hb, nar_ib) = run_pair(true);
        assert_eq!(leg_h, leg_i, "legacy hoisted vs inline");
        assert_eq!(nar_h, nar_i, "narrow hoisted vs inline");
        assert_eq!(nar_h, leg_h, "narrow hoisted vs legacy hoisted");
        assert_eq!(nar_i, leg_i, "narrow inline vs legacy inline");
        assert_eq!(leg_hb, leg_ib);
        assert_eq!(nar_hb, leg_hb);
        assert_eq!(nar_ib, leg_ib);
        assert_eq!(host.log_messages(), None);
    }

    /// 32-iteration back-to-back loop in ONE command list under narrow clamp
    /// sync: compute producer writing distinct args + count -> narrow inline
    /// clamp -> `ExecuteIndirect` compute consumer writing per-iteration
    /// output, interleaved with a graphics render pass in the same command
    /// list; verifies exact output across all 32 iterations, exact budget
    /// accounting, and exact render target pixels.
    #[test]
    fn narrow_clamp_sync_compute_producer_to_execute_indirect_chain() {
        use super::*;
        use crate::compute::Param;
        use windows::Win32::Graphics::Direct3D::D3D_PRIMITIVE_TOPOLOGY_TRIANGLELIST;
        use windows::Win32::Graphics::Dxgi::Common::{DXGI_FORMAT_R8G8B8A8_UNORM, DXGI_SAMPLE_DESC};

        const PRODUCER: &str = r#"
RWByteAddressBuffer argsOut : register(u0);
RWByteAddressBuffer countOut : register(u1);
cbuffer Params : register(b0) { uint argsOff; uint countOff; uint groups; };
[numthreads(1, 1, 1)]
void main() {
    argsOut.Store3(argsOff, uint3(groups, 1u, 1u));
    argsOut.Store3(argsOff + 12u, uint3(0xffffffffu, 0xffffffffu, 0xffffffffu));
    countOut.Store(countOff, 1u);
}
"#;
        const CONSUMER: &str = r#"
RWByteAddressBuffer output : register(u0);
cbuffer Params : register(b0) { uint iter; };
[numthreads(1, 1, 1)]
void main(uint3 g : SV_GroupID) {
    output.Store((iter * 16u + g.x) * 4u, (iter + 1u) * 100u + g.x + 1u);
}
"#;
        const RENDER_PASS: &str = r#"
float4 vs(uint id : SV_VertexID) : SV_Position {
    float2 p = float2((id == 1) ? 3.0 : -1.0, (id == 2) ? 3.0 : -1.0);
    return float4(p, 0.0, 1.0);
}
float4 ps() : SV_Target {
    return float4(1.0, 0.0, 1.0, 1.0);
}
"#;

        let Some(host) = host() else {
            return; // no host GPU here
        };
        assert!(host.enhanced_barriers());

        let (prod_root, prod_pso) = crate::compute::pipeline(
            PRODUCER,
            s!("narrow_producer"),
            0,
            &[
                Param::Uav { register: 0 },
                Param::Uav { register: 1 },
                Param::Constants { register: 0, count: 3 },
            ],
        )
        .unwrap();
        let (cons_root, cons_pso) = crate::compute::pipeline(
            CONSUMER,
            s!("narrow_consumer"),
            0,
            &[
                Param::Uav { register: 0 },
                Param::Constants { register: 0, count: 1 },
            ],
        )
        .unwrap();

        let compile_gfx = |entry: windows::core::PCSTR, target: windows::core::PCSTR| {
            let mut blob: Option<ID3DBlob> = None;
            // SAFETY: static HLSL source and entry/target names outlive the call.
            unsafe {
                D3DCompile(
                    RENDER_PASS.as_ptr().cast(),
                    RENDER_PASS.len(),
                    s!("narrow_gfx"),
                    None,
                    None,
                    entry,
                    target,
                    0,
                    0,
                    &mut blob,
                    None,
                )
                .unwrap();
            }
            blob.unwrap()
        };
        let vs = compile_gfx(s!("vs"), s!("vs_5_1"));
        let ps = compile_gfx(s!("ps"), s!("ps_5_1"));
        let gfx_root = crate::compute::root_signature(&host.device, &[]).unwrap();
        // SAFETY: compiled shader blobs and root signature are live.
        let (gfx_pso, rt, rtv_heap, rtv, rt_readback, rt_footprint) = unsafe {
            let mut gfx_desc = D3D12_GRAPHICS_PIPELINE_STATE_DESC {
                pRootSignature: std::mem::transmute_copy(&gfx_root),
                VS: D3D12_SHADER_BYTECODE {
                    pShaderBytecode: vs.GetBufferPointer(),
                    BytecodeLength: vs.GetBufferSize(),
                },
                PS: D3D12_SHADER_BYTECODE {
                    pShaderBytecode: ps.GetBufferPointer(),
                    BytecodeLength: ps.GetBufferSize(),
                },
                SampleMask: u32::MAX,
                PrimitiveTopologyType: D3D12_PRIMITIVE_TOPOLOGY_TYPE_TRIANGLE,
                NumRenderTargets: 1,
                SampleDesc: DXGI_SAMPLE_DESC { Count: 1, Quality: 0 },
                ..Default::default()
            };
            gfx_desc.RTVFormats[0] = DXGI_FORMAT_R8G8B8A8_UNORM;
            gfx_desc.BlendState.RenderTarget[0].RenderTargetWriteMask = 0xf;
            gfx_desc.RasterizerState.FillMode = D3D12_FILL_MODE_SOLID;
            gfx_desc.RasterizerState.CullMode = D3D12_CULL_MODE_NONE;
            let gfx_pso: ID3D12PipelineState =
                host.device.CreateGraphicsPipelineState(&gfx_desc).unwrap();

            let tex_desc = D3D12_RESOURCE_DESC {
                Dimension: D3D12_RESOURCE_DIMENSION_TEXTURE2D,
                Width: 16,
                Height: 16,
                DepthOrArraySize: 1,
                MipLevels: 1,
                Format: DXGI_FORMAT_R8G8B8A8_UNORM,
                SampleDesc: DXGI_SAMPLE_DESC { Count: 1, Quality: 0 },
                Flags: D3D12_RESOURCE_FLAG_ALLOW_RENDER_TARGET,
                ..Default::default()
            };
            let mut rt: Option<ID3D12Resource> = None;
            host.device
                .CreateCommittedResource(
                    &D3D12_HEAP_PROPERTIES {
                        Type: D3D12_HEAP_TYPE_DEFAULT,
                        ..Default::default()
                    },
                    D3D12_HEAP_FLAG_NONE,
                    &tex_desc,
                    D3D12_RESOURCE_STATE_RENDER_TARGET,
                    None,
                    &mut rt,
                )
                .unwrap();
            let rt = rt.unwrap();
            let rtv_heap: ID3D12DescriptorHeap = host
                .device
                .CreateDescriptorHeap(&D3D12_DESCRIPTOR_HEAP_DESC {
                    Type: D3D12_DESCRIPTOR_HEAP_TYPE_RTV,
                    NumDescriptors: 1,
                    ..Default::default()
                })
                .unwrap();
            let rtv = rtv_heap.GetCPUDescriptorHandleForHeapStart();
            host.device.CreateRenderTargetView(&rt, None, rtv);
            let mut footprint = D3D12_PLACED_SUBRESOURCE_FOOTPRINT::default();
            let mut total = 0u64;
            host.device.GetCopyableFootprints(
                &tex_desc,
                0,
                1,
                0,
                Some(&mut footprint),
                None,
                None,
                Some(&mut total),
            );
            let (rt_readback, _) = host.shared_buffer(total).unwrap();
            (gfx_pso, rt, rtv_heap, rtv, rt_readback, footprint)
        };
        let _keep_rtv_heap = rtv_heap;

        let argument = D3D12_INDIRECT_ARGUMENT_DESC {
            Type: D3D12_INDIRECT_ARGUMENT_TYPE_DISPATCH,
            ..Default::default()
        };
        let sig_desc = D3D12_COMMAND_SIGNATURE_DESC {
            ByteStride: 12,
            NumArgumentDescs: 1,
            pArgumentDescs: &argument,
            NodeMask: 0,
        };
        // SAFETY: plain command signature creation.
        let signature = unsafe {
            let mut sig: Option<ID3D12CommandSignature> = None;
            host.device
                .CreateCommandSignature(&sig_desc, None::<&ID3D12RootSignature>, &mut sig)
                .unwrap();
            sig.unwrap()
        };

        const ITERS: u32 = 32;
        let (prod_args, prod_args_ptr) = host.shared_buffer(u64::from(ITERS) * 32).unwrap();
        let (prod_count, prod_count_ptr) = host.shared_buffer(u64::from(ITERS) * 16).unwrap();
        let (output, output_ptr) = host.shared_buffer(u64::from(ITERS) * 16 * 4).unwrap();

        with_clamp_sync(true, || {
            let list = host.new_list().unwrap();
            let mut expected_budget = 0u32;
            // SAFETY: mapped buffers, open list, live PSOs/roots/signature.
            unsafe {
                // Zero-fill inputs so stale reads would dispatch 0 groups.
                std::ptr::write_bytes(prod_args_ptr, 0, ITERS as usize * 32);
                std::ptr::write_bytes(prod_count_ptr, 0, ITERS as usize * 16);
                std::ptr::write_bytes(output_ptr, 0, ITERS as usize * 16 * 4);

                let prod_args_gpu = prod_args.GetGPUVirtualAddress();
                let prod_count_gpu = prod_count.GetGPUVirtualAddress();
                let output_gpu = output.GetGPUVirtualAddress();
                let l = &list.list;

                for iter in 0..ITERS {
                    let groups = (iter % 7) + 1;
                    expected_budget += groups * 64;
                    let args_off = iter * 32;
                    let count_off = iter * 16;

                    // 1. Compute producer writes distinct args (2 records, record 1
                    //    poisoned) and count = 1.
                    l.SetComputeRootSignature(&prod_root);
                    l.SetPipelineState(&prod_pso);
                    l.SetComputeRootUnorderedAccessView(0, prod_args_gpu);
                    l.SetComputeRootUnorderedAccessView(1, prod_count_gpu);
                    let params = [args_off, count_off, groups];
                    l.SetComputeRoot32BitConstants(2, 3, params.as_ptr().cast(), 0);
                    l.Dispatch(1, 1, 1);

                    // 2. Narrow inline clamp reads GPU-written args + count into a
                    //    disjoint bump-allocated slice.
                    let (clamped, dst_off) = bound(
                        &list,
                        prod_args_gpu + u64::from(args_off),
                        (2, 12, 2),
                        Some(prod_count_gpu + u64::from(count_off)),
                    )
                    .expect("narrow inline clamp");

                    // 3. ExecuteIndirect compute consumer writes iteration output.
                    l.SetComputeRootSignature(&cons_root);
                    l.SetPipelineState(&cons_pso);
                    l.SetComputeRootUnorderedAccessView(0, output_gpu);
                    l.SetComputeRoot32BitConstants(1, 1, (&raw const iter).cast(), 0);
                    l.ExecuteIndirect(
                        &signature,
                        2,
                        &clamped,
                        dst_off,
                        Some(&prod_count),
                        u64::from(count_off),
                    );
                    done(&list, &clamped);

                    // Interleave a graphics render pass in the same command list.
                    if iter == 15 {
                        let vp = D3D12_VIEWPORT {
                            TopLeftX: 0.0,
                            TopLeftY: 0.0,
                            Width: 16.0,
                            Height: 16.0,
                            MinDepth: 0.0,
                            MaxDepth: 1.0,
                        };
                        let sc = windows::Win32::Foundation::RECT {
                            left: 0,
                            top: 0,
                            right: 16,
                            bottom: 16,
                        };
                        l.OMSetRenderTargets(1, Some(&rtv), false, None);
                        l.RSSetViewports(&[vp]);
                        l.RSSetScissorRects(&[sc]);
                        l.ClearRenderTargetView(rtv, &[0.0, 0.0, 0.0, 1.0], None);
                        l.SetGraphicsRootSignature(&gfx_root);
                        l.SetPipelineState(&gfx_pso);
                        l.IASetPrimitiveTopology(D3D_PRIMITIVE_TOPOLOGY_TRIANGLELIST);
                        l.DrawInstanced(3, 1, 0, 0);
                    }
                }

                // Read back the interleaved render target.
                l.ResourceBarrier(&[transition(
                    &rt,
                    D3D12_RESOURCE_STATE_RENDER_TARGET,
                    D3D12_RESOURCE_STATE_COPY_SOURCE,
                )]);
                let to = D3D12_TEXTURE_COPY_LOCATION {
                    pResource: std::mem::transmute_copy(&rt_readback),
                    Type: D3D12_TEXTURE_COPY_TYPE_PLACED_FOOTPRINT,
                    Anonymous: D3D12_TEXTURE_COPY_LOCATION_0 {
                        PlacedFootprint: rt_footprint,
                    },
                };
                let from = D3D12_TEXTURE_COPY_LOCATION {
                    pResource: std::mem::transmute_copy(&rt),
                    Type: D3D12_TEXTURE_COPY_TYPE_SUBRESOURCE_INDEX,
                    Anonymous: D3D12_TEXTURE_COPY_LOCATION_0 { SubresourceIndex: 0 },
                };
                l.CopyTextureRegion(&to, 0, 0, 0, &from, None);

                list.close();
                host.execute(&[&list]);

                let out_words = output_ptr.cast::<u32>();
                for iter in 0..ITERS {
                    let groups = (iter % 7) + 1;
                    for k in 0..16u32 {
                        let got = out_words.add((iter * 16 + k) as usize).read_volatile();
                        let expected = if k < groups { (iter + 1) * 100 + k + 1 } else { 0 };
                        assert_eq!(got, expected, "iter={iter} k={k} groups={groups}");
                    }
                }
                assert_eq!(
                    list.budget_used(),
                    Some(expected_budget),
                    "exact 32-iteration budget accounting"
                );

                let mut rt_ptr = std::ptr::null_mut();
                rt_readback.Map(0, None, Some(&mut rt_ptr)).unwrap();
                let pitch = rt_footprint.Footprint.RowPitch as usize;
                for y in 0..16usize {
                    let row = std::slice::from_raw_parts(rt_ptr.cast::<u8>().add(y * pitch), 16 * 4);
                    for x in 0..16usize {
                        assert_eq!(
                            &row[x * 4..x * 4 + 4],
                            &[255, 0, 255, 255],
                            "interleaved render pass pixel ({x},{y})"
                        );
                    }
                }
                rt_readback.Unmap(0, None);
            }
            assert_eq!(host.log_messages(), None);
        });
    }

    /// Mixed hoisted prelude batch + inline clamp suffix + `ExecuteIndirect` +
    /// closed-list re-submission + list `reset()` reuse under
    /// `VICTORY_CLAMP_SYNC=1`, with zero legacy resource transitions.
    #[test]
    fn narrow_clamp_sync_mixed_hoisted_prelude_and_inline_suffix() {
        use super::*;
        use crate::compute::Param;
        use crate::host::tests::issued;
        use windows::core::Interface;

        const MARK: &str = r#"
RWByteAddressBuffer marks : register(u0);
cbuffer Offset : register(b0) { uint base; };
[numthreads(1, 1, 1)]
void main(uint3 g : SV_GroupID) {
    marks.Store((base + g.x) * 4u, 1u);
}
"#;

        let Some(host) = host() else {
            return; // no host GPU here
        };
        assert!(host.enhanced_barriers());

        let (mark_root, mark_pso) = crate::compute::pipeline(
            MARK,
            s!("narrow_mixed_mark"),
            0,
            &[
                Param::Uav { register: 0 },
                Param::Constants { register: 0, count: 1 },
            ],
        )
        .unwrap();
        let argument = D3D12_INDIRECT_ARGUMENT_DESC {
            Type: D3D12_INDIRECT_ARGUMENT_TYPE_DISPATCH,
            ..Default::default()
        };
        let sig_desc = D3D12_COMMAND_SIGNATURE_DESC {
            ByteStride: 12,
            NumArgumentDescs: 1,
            pArgumentDescs: &argument,
            NodeMask: 0,
        };
        // SAFETY: plain command signature creation.
        let signature = unsafe {
            let mut sig: Option<ID3D12CommandSignature> = None;
            host.device
                .CreateCommandSignature(&sig_desc, None::<&ID3D12RootSignature>, &mut sig)
                .unwrap();
            sig.unwrap()
        };

        let (args_buf, args_ptr) = host.shared_buffer(64).unwrap();
        let (marks_buf, marks_ptr) = host.shared_buffer(64 * 4).unwrap();

        with_clamp_sync(true, || {
            let list = host.new_list().unwrap();
            let _ = issued();

            // SAFETY: mapped buffers, open list, live PSO/root/signature.
            let (first_inline_key, first_inline_off) = unsafe {
                let args_gpu = args_buf.GetGPUVirtualAddress();
                let marks_gpu = marks_buf.GetGPUVirtualAddress();
                let write_args = |counts: [u32; 4]| {
                    for (i, &g) in counts.iter().enumerate() {
                        args_ptr.cast::<u32>().add(i * 4).write_volatile(g);
                        args_ptr.cast::<u32>().add(i * 4 + 1).write_volatile(1);
                        args_ptr.cast::<u32>().add(i * 4 + 2).write_volatile(1);
                    }
                };
                let check_marks = |counts: [u32; 4], label: &str| {
                    for (i, &g) in counts.iter().enumerate() {
                        let base = (i as u32) * 8;
                        for k in 0..8u32 {
                            assert_eq!(
                                marks_ptr.cast::<u32>().add((base + k) as usize).read_volatile(),
                                u32::from(k < g),
                                "{label}: slot={i} k={k} g={g}"
                            );
                        }
                    }
                };

                // Pass 1: 2 hoisted in prelude + 2 inline in main list.
                let counts1 = [3u32, 5, 2, 4];
                write_args(counts1);
                std::ptr::write_bytes(marks_ptr, 0, 64 * 4);

                let l = &list.list;
                l.SetComputeRootSignature(&mark_root);
                l.SetPipelineState(&mark_pso);
                l.SetComputeRootUnorderedAccessView(0, marks_gpu);
                for i in 0..2u32 {
                    let (clamped, off) =
                        enqueue(&list, args_gpu + u64::from(i) * 16, (2, 12, 1), None).unwrap();
                    let base = i * 8;
                    l.SetComputeRoot32BitConstants(1, 1, (&raw const base).cast(), 0);
                    l.ExecuteIndirect(&signature, 1, &clamped, off, None, 0);
                }

                let mut inline_key = 0usize;
                let mut inline_off0 = u64::MAX;
                for i in 2..4u32 {
                    let (clamped, off) =
                        bound(&list, args_gpu + u64::from(i) * 16, (2, 12, 1), None).unwrap();
                    if i == 2 {
                        inline_key = clamped.as_raw() as usize;
                        inline_off0 = off;
                    } else {
                        assert_eq!(clamped.as_raw() as usize, inline_key);
                        assert!(off > inline_off0);
                    }
                    list.note_inline_clamp();
                    l.SetComputeRootSignature(&mark_root);
                    l.SetPipelineState(&mark_pso);
                    l.SetComputeRootUnorderedAccessView(0, marks_gpu);
                    let base = i * 8;
                    l.SetComputeRoot32BitConstants(1, 1, (&raw const base).cast(), 0);
                    l.ExecuteIndirect(&signature, 1, &clamped, off, None, 0);
                    done(&list, &clamped);
                }
                list.close();
                assert!(issued().is_empty(), "mixed narrow close must emit zero legacy transitions");
                host.execute(&[&list]);
                assert!(issued().is_empty(), "mixed narrow execute must emit zero legacy transitions");
                check_marks(counts1, "first submit");
                assert_eq!(list.budget_used(), Some(counts1.iter().sum::<u32>() * 64));

                // Pass 2: mutate source args and re-submit the SAME closed list.
                let counts2 = [6u32, 1, 7, 2];
                write_args(counts2);
                std::ptr::write_bytes(marks_ptr, 0, 64 * 4);
                host.execute(&[&list]);
                assert!(issued().is_empty(), "narrow re-submit must emit zero legacy transitions");
                check_marks(counts2, "re-submit");
                assert_eq!(list.budget_used(), Some(counts2.iter().sum::<u32>() * 64));

                (inline_key, inline_off0)
            };

            // Pass 3: reset() list, verify inline arena chunk reuse & cursor
            // reset to 0, record new hoisted + inline dispatches, and run.
            list.reset();
            assert_eq!(first_inline_off, 0, "first inline clamp started at offset 0");
            // SAFETY: mapped buffers, freshly reset open list.
            unsafe {
                let args_gpu = args_buf.GetGPUVirtualAddress();
                let marks_gpu = marks_buf.GetGPUVirtualAddress();
                let counts3 = [4u32, 0, 6, 0];
                for (i, &g) in counts3.iter().enumerate() {
                    args_ptr.cast::<u32>().add(i * 4).write_volatile(g);
                    args_ptr.cast::<u32>().add(i * 4 + 1).write_volatile(1);
                    args_ptr.cast::<u32>().add(i * 4 + 2).write_volatile(1);
                }
                std::ptr::write_bytes(marks_ptr, 0, 64 * 4);

                let l = &list.list;
                let (h_buf, h_off) = enqueue(&list, args_gpu, (2, 12, 1), None).unwrap();
                assert_eq!(h_off, 0, "hoisted arena cursor resets to 0");
                l.SetComputeRootSignature(&mark_root);
                l.SetPipelineState(&mark_pso);
                l.SetComputeRootUnorderedAccessView(0, marks_gpu);
                let base0 = 0u32;
                l.SetComputeRoot32BitConstants(1, 1, (&raw const base0).cast(), 0);
                l.ExecuteIndirect(&signature, 1, &h_buf, h_off, None, 0);

                let (i_buf, i_off) = bound(&list, args_gpu + 32, (2, 12, 1), None).unwrap();
                assert_eq!(i_buf.as_raw() as usize, first_inline_key, "inline chunk reused after reset");
                assert_eq!(i_off, 0, "inline arena cursor resets to 0");
                list.note_inline_clamp();
                l.SetComputeRootSignature(&mark_root);
                l.SetPipelineState(&mark_pso);
                l.SetComputeRootUnorderedAccessView(0, marks_gpu);
                let base2 = 16u32;
                l.SetComputeRoot32BitConstants(1, 1, (&raw const base2).cast(), 0);
                l.ExecuteIndirect(&signature, 1, &i_buf, i_off, None, 0);
                done(&list, &i_buf);

                list.close();
                assert!(issued().is_empty(), "post-reset narrow close must emit zero legacy transitions");
                host.execute(&[&list]);
                assert!(issued().is_empty(), "post-reset narrow execute must emit zero legacy transitions");

                for k in 0..8u32 {
                    assert_eq!(
                        marks_ptr.cast::<u32>().add(k as usize).read_volatile(),
                        u32::from(k < 4),
                        "post-reset hoisted k={k}"
                    );
                    assert_eq!(
                        marks_ptr.cast::<u32>().add((16 + k) as usize).read_volatile(),
                        u32::from(k < 6),
                        "post-reset inline k={k}"
                    );
                }
                assert_eq!(list.budget_used(), Some((4 + 6) * 64));
            }
            assert_eq!(host.log_messages(), None);
        });
    }
}
