//! Draws on the host GPU (decision 0007, step 3). A list keeps the state
//! the title sets for its draws (pipeline, descriptor heaps, root
//! arguments, vertex and index buffers, targets); each draw then brings the
//! host list up to date: the host pipeline and our root signature, the
//! user data the translated shaders read, the host copies of the buffers
//! and descriptors they read, and the states of the textures involved.
//! State the host keeps across draws (viewports, scissors, blend factor,
//! stencil reference, depth bounds, topology, targets) is set on the host
//! list as the title sets it.

use std::sync::atomic::{AtomicU64, Ordering};

use victory_shader::{ResourceType, RootType, UsedResource};

use crate::descriptor::View;
use windows::Win32::Foundation::RECT;
use windows::Win32::Graphics::Direct3D::D3D_PRIMITIVE_TOPOLOGY;
use windows::Win32::Graphics::Direct3D12::*;
use windows::Win32::Graphics::Dxgi::Common::DXGI_FORMAT;
use windows::core::Interface;

use crate::host::HostList;
use crate::object::This;
use crate::pipeline::host::HostPipeline;

mod bake_sources;

/// A root argument as the title set it.
#[derive(Debug, Clone, Default)]
enum Root {
    #[default]
    None,
    /// A descriptor table's first descriptor (an address in a heap).
    Table(u64),
    Constants(Vec<u32>),
    /// A root view's GPU address.
    View(u64),
}

#[repr(C)]
#[derive(Debug, Clone, Copy, Default)]
pub(crate) struct VertexView {
    address: u64,
    size: u32,
    stride: u32,
}

#[repr(C)]
#[derive(Debug, Clone, Copy, Default)]
pub(crate) struct IndexView {
    address: u64,
    size: u32,
    format: u32,
}

/// What one title list has set, and what its host list was last given.
#[derive(Default)]
pub(crate) struct State {
    pipeline: usize,
    /// View heap, sampler heap.
    heaps: [usize; 2],
    graphics: Vec<Root>,
    compute: Vec<Root>,
    vertex: [VertexView; 32],
    index: IndexView,
    /// Bound render target and depth resources, for their host states.
    targets: Vec<usize>,
    depth: usize,
    /// Last render target and depth keys set on the host list.
    host_targets: Option<(Vec<TargetKey>, Option<TargetKey>)>,
    /// The formats of the views [`State::set_targets`] bound on the host
    /// list: a draw's pipeline is one for them
    /// (`crate::pipeline::host_pipeline_for`).
    target_formats: crate::pipeline::host::TargetFormats,
    /// None leaves the host list's default bounds (0..1) in place.
    depth_bounds: Option<(f32, f32)>,
    /// The title's primitive topology (the host's may differ).
    topology: u32,
    /// What the host list has: pipeline, root signature, heaps.
    host_pipeline: usize,
    /// Graphics, compute.
    host_roots: [usize; 2],
    host_heaps: [usize; 2],
    /// Tables made afresh in this list: (table, what it holds) → first
    /// dynamic slot.
    fresh_tables: std::collections::HashMap<(u64, u64), u32>,
    /// The dynamic slots the list holds: title heap, host heap identity
    /// (`HostHeap::id`), first, count; given back when the list is reset or
    /// released.
    dynamic: Vec<(usize, u64, u32, u32)>,
    /// A fresh table of the stage being bound could not be made.
    table_failed: bool,
    /// The title-memory page table made for this list (`crate::memory`)
    /// and its view heap.
    memory: Option<(usize, crate::memory::Map)>,
    /// Views made for root SRVs and UAVs in this list ([`State::root_view`]):
    /// (view heap, host copy, unordered access) → dynamic slot.
    root_slots: std::collections::HashMap<(usize, usize, bool), u32>,
    /// The run of dynamic slots root views take from (view heap, next
    /// slot, end): taken [`ROOT_VIEW_RUN`] at a time, so that single slots
    /// do not break up the runs fresh tables need.
    root_run: (usize, u32, u32),
    /// What was set straight onto the host list, for the next segment of
    /// the list ([`State::cut`]).
    fixed: Fixed,
    /// Per-list descriptor preparation memo (`VICTORY_TABLE_MEMO`): maps a
    /// table entry key to `(descriptor_bytes, required_epoch, read_tested_depth, seen)`.
    entry_memo: std::collections::HashMap<
        EntryKey,
        EntryValue,
        std::hash::BuildHasherDefault<MemoHasher>,
    >,
    /// Pending RTVs drawn while `VICTORY_HDR_PROBE` is active, flushed on
    /// target switch or list close: `(resources, last_pipeline_name)`.
    hdr_rtvs: Option<(Vec<usize>, String)>,
    /// Last `(pipeline_ptr, uav_resource)` probed by a compute dispatch on
    /// this list, collapsing consecutive tile/mip dispatches of the same pass.
    hdr_last_compute: (usize, usize),
}

/// Fast FNV-1a hasher for per-list descriptor memo keys.
#[derive(Default)]
struct MemoHasher(u64);

impl std::hash::Hasher for MemoHasher {
    fn finish(&self) -> u64 {
        self.0
    }

    fn write(&mut self, bytes: &[u8]) {
        let mut h = if self.0 == 0 {
            0xcbf2_9ce4_8422_2325
        } else {
            self.0
        };
        for &b in bytes {
            h ^= u64::from(b);
            h = h.wrapping_mul(0x0000_0100_0000_01b3);
        }
        self.0 = h;
    }

    fn write_u64(&mut self, i: u64) {
        let mut h = if self.0 == 0 {
            0xcbf2_9ce4_8422_2325
        } else {
            self.0
        };
        h ^= i;
        h = h.wrapping_mul(0x0000_0100_0000_01b3);
        self.0 = h;
    }

    fn write_usize(&mut self, i: usize) {
        self.write_u64(i as u64);
    }
}

/// Key for per-list table-entry memoization: `(handle, kind, tested_depth_resource)`.
type EntryKey = (u64, ResourceType, usize);

/// Value stored in [`State::entry_memo`]: `(descriptor_bytes, required_epoch,
/// read_tested_depth, seen)`.
type EntryValue = ([u64; 4], u64, bool, Seen);

/// Decision from [`State::entry_begin`] before preparing a table entry.
enum EntryMemo {
    Off,
    Skip,
    Do {
        key: EntryKey,
        bytes: [u64; 4],
        hit: bool,
        epoch: u64,
        deferred_epoch: u64,
        steps: usize,
        depth_before: (usize, bool),
        depth_hit: bool,
    },
}

/// External state a memoized table entry depended on beyond the command list's
/// epoch: nothing (buffers, constants, samplers), a title texture and its
/// current host copy pointer, or a placed texture and its host copy pointer.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
enum Seen {
    Nothing,
    Texture(usize, usize),
    Placed(usize, usize),
}

fn table_memo_mode_value(value: Option<&std::ffi::OsStr>) -> u8 {
    match value.and_then(|v| v.to_str()) {
        Some("1") => 1,
        Some("2" | "verify") => 2,
        _ => 0,
    }
}

#[cfg(test)]
thread_local! {
    static TABLE_MEMO_OVERRIDE: std::cell::Cell<Option<u8>> = const { std::cell::Cell::new(None) };
}

/// Scoped test guard to override `VICTORY_TABLE_MEMO` on the current thread.
#[cfg(test)]
struct TableMemoGuard(Option<u8>);

#[cfg(test)]
impl TableMemoGuard {
    fn set(mode: u8) -> Self {
        Self(TABLE_MEMO_OVERRIDE.with(|c| c.replace(Some(mode))))
    }
}

#[cfg(test)]
impl Drop for TableMemoGuard {
    fn drop(&mut self) {
        TABLE_MEMO_OVERRIDE.with(|c| c.set(self.0));
    }
}

/// `VICTORY_TABLE_MEMO`: `0` off (default), `1` active skip, `2` (or `verify`)
/// verification mode that still runs the full path on hits and counts any
/// epoch, deferred-step, or depth-test mismatch.
fn table_memo() -> u8 {
    #[cfg(test)]
    if let Some(v) = TABLE_MEMO_OVERRIDE.with(std::cell::Cell::get) {
        return v;
    }
    static MODE: std::sync::OnceLock<u8> = std::sync::OnceLock::new();
    *MODE.get_or_init(|| table_memo_mode_value(std::env::var_os("VICTORY_TABLE_MEMO").as_deref()))
}

thread_local! {
    /// Currently bound graphics depth target `(resource, read_by_table)` while
    /// preparing descriptor tables.
    static TESTED_DEPTH: std::cell::Cell<(usize, bool)> = const { std::cell::Cell::new((0, false)) };
}

fn copy_of(r: usize) -> usize {
    crate::resource::made_copy(r as This).map_or(0, |c| std::sync::Arc::as_ptr(&c) as usize)
}

/// Returns what a memoized table entry for `(view, kind)` depends on, or `None`
/// if the entry must not be memoized (e.g., buffer UAVs whose write step is
/// recorded every time, 3-channel widened SRVs that check `widened_ready`, or
/// aliased textures whose source may be written outside this list).
fn memo_copy(view: &View, kind: ResourceType) -> Option<Seen> {
    let is_uav = kind == ResourceType::Uav;
    if kind == ResourceType::Srv && crate::descriptor::host::three_channel_srv(view) {
        return None;
    }
    let texture = |r: usize| -> Option<(usize, usize)> {
        if crate::resource::buffer_bytes(r as This).is_some() {
            None
        } else {
            Some((r, copy_of(r)))
        }
    };
    let aliased = |r: usize| crate::resource::alias_source(r as This).is_some();
    match view {
        View::Constant(..) => (!is_uav).then_some(Seen::Nothing),
        View::Sampler(..) | View::ConsoleSampler(..) => Some(Seen::Nothing),
        View::Shader(r, _) | View::Unordered(r, _, _) => match texture(*r) {
            Some((r, copy)) => (!aliased(r)).then_some(Seen::Texture(r, copy)),
            None => (!is_uav).then_some(Seen::Nothing),
        },
        View::Placed(..) => match crate::descriptor::host::placed_texture(view) {
            Some(View::Shader(r, _) | View::Unordered(r, _, _)) => match texture(r) {
                Some((r, copy)) => (!aliased(r)).then_some(Seen::Placed(r, copy)),
                None => (!is_uav).then_some(Seen::Nothing),
            },
            Some(_) => None,
            None => (!is_uav).then_some(Seen::Nothing),
        },
    }
}

/// Checks whether the external resource state observed when `handle` was
/// memoized still stands.
fn copy_stands(seen: Seen, handle: u64) -> bool {
    match seen {
        Seen::Nothing => true,
        Seen::Texture(r, copy) => crate::descriptor::view(handle).is_some_and(|view| {
            matches!(
                &*view,
                View::Shader(p, _) | View::Unordered(p, _, _) if *p == r
            ) && copy_of(r) == copy
        }),
        Seen::Placed(r, copy) => crate::descriptor::view(handle).is_some_and(|view| {
            matches!(
                crate::descriptor::host::placed_texture(&view),
                Some(View::Shader(p, _) | View::Unordered(p, _, _)) if p == r
            ) && copy_of(r) == copy
        }),
    }
}

/// Table memo counters:
/// 0: lookups, 1: hits, 2: verify epoch mismatches, 3: verify deferred_steps
/// mismatches, 4: verify depth mismatches, 5: first-in-list misses, 6: epoch
/// misses, 7: host-copy misses, 8: unmemoized entries.
const MEMO_COUNTERS: usize = 9;
static MEMO: [AtomicU64; MEMO_COUNTERS] = [const { AtomicU64::new(0) }; MEMO_COUNTERS];

#[derive(Default)]
struct MemoCounts([u64; MEMO_COUNTERS]);

impl MemoCounts {
    fn flush(&mut self) {
        for (i, v) in self.0.iter_mut().enumerate() {
            let n = std::mem::take(v);
            if n != 0 {
                MEMO[i].fetch_add(n, Ordering::Relaxed);
            }
        }
    }
}

impl Drop for MemoCounts {
    fn drop(&mut self) {
        self.flush();
    }
}

thread_local! {
    static MEMO_LOCAL: std::cell::RefCell<(MemoCounts, u32)> =
        std::cell::RefCell::new((MemoCounts::default(), 0));
}

fn memo_count(i: usize) {
    MEMO_LOCAL.with_borrow_mut(|(c, n)| {
        c.0[i] += 1;
        *n += 1;
        if *n >= 4096 {
            *n = 0;
            c.flush();
        }
    });
}

fn flush_memo_local() {
    MEMO_LOCAL.with_borrow_mut(|(c, n)| {
        *n = 0;
        c.flush();
    });
}

/// Logs compact per-frame descriptor memo stats when `VICTORY_TABLE_MEMO != 0`.
pub(crate) fn table_memo_line(frames: f64) {
    let mode = table_memo();
    if mode == 0 {
        return;
    }
    flush_memo_local();
    let v: [u64; MEMO_COUNTERS] = std::array::from_fn(|i| MEMO[i].swap(0, Ordering::Relaxed));
    let f = frames.max(1.0);
    let hit_pct = 100.0 * v[1] as f64 / v[0].max(1) as f64;
    if mode == 2 {
        tracing::info!(
            "table memo (VICTORY_TABLE_MEMO=2) per frame: lookups {:.0}, hits {:.0} ({hit_pct:.1}%), \
             verify mismatches: epoch {:.1}, steps {:.1}, depth {:.1} (must be 0); \
             misses: first {:.0}, epoch {:.0}, copy {:.0}, unmemoized {:.0}",
            v[0] as f64 / f,
            v[1] as f64 / f,
            v[2] as f64 / f,
            v[3] as f64 / f,
            v[4] as f64 / f,
            v[5] as f64 / f,
            v[6] as f64 / f,
            v[7] as f64 / f,
            v[8] as f64 / f,
        );
    } else {
        tracing::info!(
            "table memo (VICTORY_TABLE_MEMO=1) per frame: lookups {:.0}, hits {:.0} ({hit_pct:.1}%); \
             misses: first {:.0}, epoch {:.0}, copy {:.0}, unmemoized {:.0}",
            v[0] as f64 / f,
            v[1] as f64 / f,
            v[5] as f64 / f,
            v[6] as f64 / f,
            v[7] as f64 / f,
            v[8] as f64 / f,
        );
    }
}

/// Fixed-function state the title set, as it set it.
#[derive(Default)]
struct Fixed {
    viewports: Option<Vec<D3D12_VIEWPORT>>,
    scissors: Option<Vec<RECT>>,
    blend_factor: Option<Option<[f32; 4]>>,
    stencil_ref: Option<u32>,
    /// Render target and depth view handles.
    targets: Option<(Vec<u64>, u64)>,
    topology: bool,
}

impl Drop for State {
    fn drop(&mut self) {
        for &(heap, id, first, n) in &self.dynamic {
            match (heap, crate::memory::heap()) {
                (MEMORY_HEAP, Some(ours)) => ours.release(id, first, n),
                _ => crate::descriptor::release_dynamic(heap, id, first, n),
            }
        }
    }
}

/// The key of our own view heap (`crate::memory::heap`) among the title's.
const MEMORY_HEAP: usize = crate::memory::OURS;

/// The sampler heap we bind when the title binds none (a value no title heap
/// address takes).
const SAMPLERS_OURS: usize = usize::MAX;

/// Dynamic slots taken at a time for root views.
const ROOT_VIEW_RUN: u32 = 256;

/// Dynamic slots taken for root views ([`State::root_view`]) since the last
/// call.
pub(crate) fn take_root_view_slots() -> u64 {
    ROOT_VIEW_SLOTS.swap(0, Ordering::Relaxed)
}
static ROOT_VIEW_SLOTS: AtomicU64 = AtomicU64::new(0);

/// Draws recorded on the host GPU since the last frame line.
static DRAWN: AtomicU64 = AtomicU64::new(0);

/// Draws run on the host so far.
pub(crate) fn drawn() -> u64 {
    DRAWN.load(Ordering::Relaxed)
}

/// The draws traced since the last present (diagnostics builds, with
/// debug logging for this module): logged when the frame comes out black.
#[cfg(feature = "diagnostics")]
static FRAME: std::sync::Mutex<Vec<String>> = std::sync::Mutex::new(Vec::new());

/// The draws traced since the last call.
#[cfg(feature = "diagnostics")]
pub(crate) fn take_frame() -> Vec<String> {
    std::mem::take(&mut *FRAME.lock().unwrap_or_else(|e| e.into_inner()))
}

/// Draws, mesh dispatches and dispatches recorded but not run on the host
/// (no host pipeline) since the last call.
pub(crate) fn take_dropped() -> u64 {
    DROPPED.swap(0, Ordering::Relaxed)
}
static DROPPED: AtomicU64 = AtomicU64::new(0);

/// Marks the last traced draw as not run on the host, counts it by why
/// (the reason [`not_run`] gave, if its pipeline has a host one) and notes
/// it in the list (diagnostics).
#[cfg(feature = "diagnostics")]
fn dropped(list: &HostList, pipeline: usize, kind: &str) {
    let why = NOT_RUN.with(|w| w.replace(""));
    if why.is_empty() || crate::pipeline::host_pipeline(pipeline as This).is_none() {
        crate::pipeline::note_drop(pipeline as This);
    } else {
        crate::pipeline::note_why(format!("{kind}: {why} {}", crate::pipeline::shaders_of(pipeline as This)));
    }
    list.note_op(|| format!("DROPPED {kind} ({why}) {}", crate::pipeline::shaders_of(pipeline as This)));
    if let Some(last) = FRAME.lock().unwrap_or_else(|e| e.into_inner()).last_mut() {
        last.push_str(" DROPPED");
    }
}

#[cfg(feature = "diagnostics")]
thread_local! {
    /// Why the draw or dispatch this thread last prepared cannot run.
    static NOT_RUN: std::cell::Cell<&'static str> = const { std::cell::Cell::new("") };
}

/// Notes why the draw or dispatch being prepared cannot run (diagnostics).
fn not_run(why: &'static str) {
    #[cfg(feature = "diagnostics")]
    NOT_RUN.with(|w| w.set(why));
    #[cfg(not(feature = "diagnostics"))]
    let _ = why;
}

/// Dispatches run on the host so far.
static DISPATCHED: AtomicU64 = AtomicU64::new(0);

pub(crate) fn dispatched() -> u64 {
    DISPATCHED.load(Ordering::Relaxed)
}

/// What a render target or depth view of the host is made from: the host
/// texture (0 for a null view) and [`crate::descriptor::RtvRecord::key`].
type TargetKey = (usize, (usize, [u32; 7]));

fn bind_cache_enabled_value(value: Option<&std::ffi::OsStr>) -> bool {
    value != Some(std::ffi::OsStr::new("0"))
}

fn bind_cache_enabled() -> bool {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| bind_cache_enabled_value(std::env::var_os("VICTORY_BIND_CACHE").as_deref()))
}

/// The host address of a root view: its buffer's host copy, or for a
/// constant buffer not 256-byte aligned there (the console needs 16) a
/// copy in upload memory. What shaders write through a root UAV goes back
/// to title memory after the submission.
/// shortcut: a root view has no size; constant buffers are taken as 64 KiB,
/// other views as 1 MiB (both cut at their buffer's end).
fn root_view(list: &HostList, address: u64, srv: bool, uav: bool) -> Option<u64> {
    let shadow = crate::shadow::find(address, 1)?;
    let len = shadow
        .left(address)
        .min(if srv || uav { 1 << 20 } else { 1 << 16 });
    let gpu = shadow.gpu(address);
    if srv || uav || gpu % 256 == 0 {
        shadow.refresh(list, address, len);
        if uav {
            // A host root descriptor: no heap view, so its writes are not
            // marked.
            shadow.gpu_writes(list, address, len, false);
        }
        return Some(gpu);
    }
    list.fill_scratch(address as *const u8, len)
}

macro_rules! once {
    ($($arg:tt)*) => {{
        static ONCE: std::sync::Once = std::sync::Once::new();
        ONCE.call_once(|| tracing::warn!($($arg)*));
    }};
}

fn set(roots: &mut Vec<Root>, slot: u32, root: Root) {
    let slot = slot as usize;
    if roots.len() <= slot {
        roots.resize(slot + 1, Root::None);
    }
    roots[slot] = root;
}

/// The format of a host view of `texture` whose description names
/// `described` (none without a description): that one, or the texture's
/// own, which is what the host gives a view without a format.
fn bound_format(texture: &ID3D12Resource, described: Option<DXGI_FORMAT>) -> u32 {
    match described {
        Some(format) if format.0 != 0 => format.0 as u32,
        // SAFETY: a live texture.
        _ => unsafe { texture.GetDesc() }.Format.0 as u32,
    }
}

impl State {
    pub(crate) fn reset(&mut self, pipeline: This) {
        *self = State::default();
        self.pipeline = pipeline as usize;
    }

    /// The title's topology; console-only ones become host lists (see
    /// [`crate::primitive`]).
    pub(crate) fn set_topology(&mut self, list: &HostList, topology: u32) {
        self.topology = topology;
        self.fixed.topology = true;
        let host = match topology {
            RECTANGLES => 4,
            TRIANGLE_FAN | LINE_LOOP | QUADS | QUAD_STRIP => {
                once!(topology, "console topology drawn as a triangle list");
                4
            }
            t => t,
        };
        // SAFETY: an open list.
        unsafe {
            list.list
                .IASetPrimitiveTopology(D3D_PRIMITIVE_TOPOLOGY(host as i32))
        };
    }

    pub(crate) fn set_pipeline(&mut self, pipeline: This) {
        self.pipeline = pipeline as usize;
    }

    /// Fixed-function state, straight onto the host list (and kept for the
    /// list's next segment).
    pub(crate) fn viewports(&mut self, list: &HostList, viewports: &[D3D12_VIEWPORT]) {
        self.fixed.viewports = Some(viewports.to_vec());
        // SAFETY: an open list.
        unsafe { list.list.RSSetViewports(viewports) };
    }

    pub(crate) fn scissors(&mut self, list: &HostList, rects: &[RECT]) {
        self.fixed.scissors = Some(rects.to_vec());
        // SAFETY: an open list.
        unsafe { list.list.RSSetScissorRects(rects) };
    }

    pub(crate) fn blend_factor(&mut self, list: &HostList, factor: Option<&[f32; 4]>) {
        self.fixed.blend_factor = Some(factor.copied());
        // SAFETY: an open list.
        unsafe { list.list.OMSetBlendFactor(factor) };
    }

    pub(crate) fn stencil_ref(&mut self, list: &HostList, reference: u32) {
        self.fixed.stencil_ref = Some(reference);
        // SAFETY: an open list.
        unsafe { list.list.OMSetStencilRef(reference) };
    }

    /// The title list goes on in a fresh host list (a segment after a wait,
    /// `crate::list`): the state it set is given to `list` again.
    ///
    /// Replayed: topology, viewports, scissors, blend factor, stencil
    /// reference, render targets and depth (set at once); heaps, root
    /// signatures, pipelines, root arguments, vertex and index buffers are
    /// set by the next draw or dispatch from what this state holds (the
    /// host caches are cleared). Made again in the new segment, so their
    /// submission-time work (refreshes, uploads) runs with it: fresh tables,
    /// root views and the title-memory page table. Not replayed, as nothing
    /// of it is on the host list: depth bounds and shading rate (ignored),
    /// queries (resolved with plain values), the 0x4a0 upload memory (the
    /// title list's own), texture states and twins (each segment has its
    /// own, made from the real states at its submission). Clears and copies
    /// are commands, recorded once in the segment they belong to.
    pub(crate) fn cut(&mut self, list: &HostList) {
        self.host_state_lost();
        self.fresh_tables.clear();
        self.entry_memo.clear();
        self.root_slots.clear();
        self.memory = None;
        let fixed = std::mem::take(&mut self.fixed);
        if fixed.topology {
            self.set_topology(list, self.topology);
        }
        if let Some(v) = &fixed.viewports {
            self.viewports(list, v);
        }
        if let Some(r) = &fixed.scissors {
            self.scissors(list, r);
        }
        if let Some(f) = fixed.blend_factor {
            self.blend_factor(list, f.as_ref());
        }
        if let Some(s) = fixed.stencil_ref {
            self.stencil_ref(list, s);
        }
        if let Some((targets, depth)) = &fixed.targets {
            self.set_targets(list, targets, *depth);
        }
    }

    /// Others recorded into the host list (`crate::external`): its heaps,
    /// root signatures and pipeline are set again before the next draw or
    /// dispatch.
    pub(crate) fn host_state_lost(&mut self) {
        // No title heaps (0) differ from what the host list has, too.
        (self.host_pipeline, self.host_roots, self.host_heaps) = (0, [0; 2], [usize::MAX; 2]);
        self.host_targets = None;
    }

    pub(crate) fn set_heaps(&mut self, heaps: &[This]) {
        self.heaps = [0; 2];
        for &h in heaps {
            if !h.is_null() {
                let samplers = crate::descriptor::heap(h).samplers();
                self.heaps[usize::from(samplers)] = h as usize;
            }
        }
    }

    fn roots(&mut self, compute: bool) -> &mut Vec<Root> {
        if compute {
            &mut self.compute
        } else {
            &mut self.graphics
        }
    }

    pub(crate) fn set_table(&mut self, compute: bool, slot: u32, handle: u64) {
        set(self.roots(compute), slot, Root::Table(handle));
    }

    pub(crate) fn set_view(&mut self, compute: bool, slot: u32, address: u64) {
        set(self.roots(compute), slot, Root::View(address));
    }

    pub(crate) fn set_constants(&mut self, compute: bool, slot: u32, values: &[u32], offset: u32) {
        let roots = self.roots(compute);
        let s = slot as usize;
        if !matches!(roots.get(s), Some(Root::Constants(_))) {
            set(roots, slot, Root::Constants(Vec::new()));
        }
        if let Root::Constants(c) = &mut roots[s] {
            let end = offset as usize + values.len();
            if c.len() < end {
                c.resize(end, 0);
            }
            c[offset as usize..end].copy_from_slice(values);
        }
    }

    pub(crate) fn set_vertex_buffers(&mut self, start: u32, views: &[VertexView]) {
        for (i, v) in views.iter().enumerate() {
            if let Some(slot) = self.vertex.get_mut(start as usize + i) {
                *slot = *v;
            }
        }
    }

    pub(crate) fn clear_vertex_buffers(&mut self, start: u32, count: u32) {
        for slot in self
            .vertex
            .iter_mut()
            .skip(start as usize)
            .take(count as usize)
        {
            *slot = VertexView::default();
        }
    }

    pub(crate) fn set_index_buffer(&mut self, view: Option<&IndexView>) {
        self.index = view.copied().unwrap_or_default();
    }

    /// Render targets and depth, set on the host list at once.
    pub(crate) fn set_targets(&mut self, list: &HostList, handles: &[u64], depth: u64) {
        self.fixed.targets = Some((handles.to_vec(), depth));
        let Some(host) = crate::host::host() else {
            return;
        };
        self.targets.clear();
        // The formats of the host views, for the draws' pipelines.
        let mut formats = [0; 9];
        let mut rtv_views = Vec::with_capacity(handles.len());
        let mut rtv_keys = Vec::with_capacity(handles.len());
        for &h in handles {
            let Some(record) = crate::descriptor::rtv_record(h) else {
                once!(
                    handle = format_args!("{h:#x}"),
                    "render target without a recorded view"
                );
                continue;
            };
            let Some(texture) = crate::resource::host_resource(record.resource) else {
                continue;
            };
            self.targets.push(record.resource as usize);
            #[cfg(feature = "diagnostics")]
            crate::capture::note_target(list, record.resource);
            let desc = record.desc();
            if rtv_views.len() < 8 {
                formats[rtv_views.len()] = bound_format(&texture, desc.as_ref().map(|d| d.Format));
            }
            rtv_keys.push((texture.as_raw() as usize, record.key()));
            rtv_views.push((texture, desc));
        }
        if self
            .hdr_rtvs
            .as_ref()
            .is_some_and(|(prev, _)| prev != &self.targets)
        {
            self.flush_hdr_probe(list);
        }
        self.depth = 0;
        let mut dsv_view = None;
        let mut dsv_key = None;
        if let Some(record) = crate::descriptor::dsv_record(depth)
            && let Some(texture) = crate::resource::host_resource(record.resource)
        {
            self.depth = record.resource as usize;
            let desc = record.depth_desc();
            formats[8] = bound_format(&texture, desc.as_ref().map(|d| d.Format));
            #[cfg(feature = "diagnostics")]
            crate::capture::note_target(list, record.resource);
            dsv_key = Some((texture.as_raw() as usize, record.key()));
            dsv_view = Some((texture, desc));
        }
        self.target_formats = formats;
        if bind_cache_enabled()
            && self
                .host_targets
                .as_ref()
                .is_some_and(|(r, d)| r == &rtv_keys && d == &dsv_key)
        {
            return;
        }
        let rtvs: Vec<_> = rtv_views
            .iter()
            .map(|(texture, desc)| host.rtv(texture, desc.as_ref()))
            .collect();
        let dsv = dsv_view
            .as_ref()
            .map(|(texture, desc)| host.dsv(texture, desc.as_ref()));
        // SAFETY: an open list; the views were just made.
        unsafe {
            list.list.OMSetRenderTargets(
                rtvs.len() as u32,
                (!rtvs.is_empty()).then_some(rtvs.as_ptr()),
                false,
                dsv.as_ref().map(std::ptr::from_ref),
            )
        };
        self.host_targets = Some((rtv_keys, dsv_key));
    }

    /// The user-data value of a root argument, as translated shaders read
    /// it: a table's byte offset in its heap, a constant's value, a root
    /// view's address (title memory).
    fn user_value(&self, compute: bool, slot: u8, root: RootType, dword: u32) -> u32 {
        match (self.roots_ref(compute).get(usize::from(slot)), root) {
            // Every lane of every group, unless the dispatch counts threads
            // (`dispatch_x`, which sets it after).
            (_, RootType::Other(victory_shader::emit::USER_DISPATCH_THREADS_X)) => 0,
            (Some(Root::View(address)), RootType::Descriptor) if dword < 2 => {
                (address >> (32 * dword)) as u32
            }
            (Some(Root::Table(handle)), RootType::Table) => match self.heap_of(*handle) {
                Some(heap) => (handle - heap.start) as u32,
                None => {
                    once!(
                        handle = format_args!("{handle:#x}"),
                        "descriptor table outside the bound heaps"
                    );
                    0
                }
            },
            (Some(Root::Constants(c)), RootType::Constants) => {
                c.get(dword as usize).copied().unwrap_or(0)
            }
            _ => {
                once!(
                    slot,
                    ?root,
                    "a shader reads a root argument the title did not set"
                );
                0
            }
        }
    }

    /// The host addresses of the bound view heap's title-memory page table
    /// and the list's page bitmap for it, set up on first use (the table
    /// names every host copy at each submission).
    fn memory_map(&mut self, list: &HostList) -> Option<(u64, u64)> {
        let heap = match self.heaps[0] {
            0 => MEMORY_HEAP,
            h => h,
        };
        if let Some((h, map)) = &self.memory
            && *h == heap
        {
            return Some((map.table, map.dirty));
        }
        let map = crate::memory::bind(list, heap)?;
        let out = (map.table, map.dirty);
        self.memory = Some((heap, map));
        Some(out)
    }

    /// The bound heap holding `handle`.
    fn heap_of(&self, handle: u64) -> Option<&'static crate::descriptor::Heap> {
        self.heaps
            .iter()
            .filter(|&&h| h != 0)
            .map(|&h| crate::descriptor::heap(h as This))
            .find(|h| h.index(handle).is_some())
    }

    pub(crate) fn set_depth_bounds(&mut self, list: &HostList, min: f32, max: f32) {
        self.depth_bounds = Some((min, max));
        if list.depth_bounds.is_none() {
            crate::host::warn_depth_bounds();
        }
        self.apply_depth_bounds(list);
    }

    fn apply_depth_bounds(&self, list: &HostList) {
        if let (Some(list), Some((min, max))) = (&list.depth_bounds, self.depth_bounds) {
            // SAFETY: the host supports depth bounds and the list is open.
            unsafe { list.OMSetDepthBounds(min, max) };
        }
    }

    /// Brings the host list up to date for a draw (`mesh` false) or a mesh
    /// dispatch; false if it cannot run on the host.
    fn prepare(&mut self, list: &HostList, indexed: bool, mesh: bool) -> bool {
        #[cfg(feature = "diagnostics")]
        if crate::pipeline::is_skipped(self.pipeline as This) {
            return false;
        }
        let Some(p) = crate::pipeline::host_pipeline_for(self.pipeline as This, &self.target_formats)
        else {
            return false;
        };
        if p.compute || p.mesh != mesh {
            once!(mesh, "a draw or mesh dispatch with another kind of pipeline");
            not_run("another kind of pipeline");
            return false;
        }
        if !self.bind(list, p, false) {
            return false;
        }
        // The sky dome the ambient light follows (`crate::ambient`).
        if crate::ambient::on()
            && crate::pipeline::ambient_role(self.pipeline as This) & crate::ambient::DRAWS_SKY != 0
            && let Some(&sky) = self.targets.first()
        {
            crate::ambient::sky_drawn(sky);
        }
        let l = &list.list;
        // Vertex buffers the input layout reads.
        let mut views = [D3D12_VERTEX_BUFFER_VIEW::default(); 32];
        let mut count = 0;
        for (slot, (view, &v)) in views.iter_mut().zip(&self.vertex).enumerate() {
            if p.slots & 1 << slot == 0 {
                continue;
            }
            count = slot + 1;
            if v.address == 0 || v.size == 0 {
                continue;
            }
            let Some(shadow) = crate::shadow::find(v.address, u64::from(v.size)) else {
                continue;
            };
            shadow.refresh(list, v.address, u64::from(v.size));
            *view = D3D12_VERTEX_BUFFER_VIEW {
                BufferLocation: shadow.gpu(v.address),
                SizeInBytes: v.size,
                StrideInBytes: v.stride,
            };
        }
        // SAFETY: an open list; the views name live host buffers.
        unsafe {
            if count > 0 {
                l.IASetVertexBuffers(0, Some(&views[..count]));
            }
        }
        if indexed {
            let v = self.index;
            let Some(shadow) = crate::shadow::find(v.address, u64::from(v.size)) else {
                not_run("index buffer outside title buffers");
                return false;
            };
            shadow.refresh(list, v.address, u64::from(v.size));
            let view = D3D12_INDEX_BUFFER_VIEW {
                BufferLocation: shadow.gpu(v.address),
                SizeInBytes: v.size,
                Format: DXGI_FORMAT(v.format as i32),
            };
            // SAFETY: an open list; the view names a live host buffer.
            unsafe { l.IASetIndexBuffer(Some(&view)) };
        }
        const RENDER_TARGET: u32 = 0x4;
        const DEPTH_WRITE: u32 = 0x10;
        for &t in &self.targets {
            list.texture_written(t as This);
            crate::list::ensure_state(list, t as This, RENDER_TARGET);
        }
        if self.depth != 0 {
            crate::list::ensure_state(list, self.depth as This, DEPTH_WRITE);
        }
        // A depth copy of ours (`crate::twin`) ran a compute pipeline.
        if list.take_rebind() {
            // SAFETY: an open list and a live pipeline.
            unsafe { l.SetPipelineState(&p.pso) };
            (self.host_pipeline, self.host_roots[1]) = (p.pso.as_raw() as usize, 0);
        }
        true
    }

    /// A descriptor a shader reads through a table: its host slot made if
    /// it changed, constant data copied, textures made readable.
    /// What draws and dispatches share: heaps, our root signature, the
    /// pipeline, and per stage the user data, root views and the
    /// descriptors its tables hold.
    /// False if the draw or dispatch must not run: a table it reads could
    /// not be made.
    fn bind(&mut self, list: &HostList, p: &HostPipeline, compute: bool) -> bool {
        // What its shaders write is withheld only if the program uses title
        // memory and a list using it misses (`crate::memory::withholding`).
        let _writer = list.writer_uses_memory(p.stages.iter().any(|s| s.memory.is_some()));
        let l = &list.list;
        // A depth copy of ours (`crate::twin`) since the last one ran a
        // compute pipeline.
        if list.take_rebind() {
            (self.host_pipeline, self.host_roots[1]) = (0, 0);
        }
        // Heaps first: a root signature with directly indexed heaps must be
        // set after them. Title memory's views need a view heap: ours where
        // the title binds none (the BVH builder's dispatches may not).
        let memory_heap = self.heaps[0] == 0
            && p.stages.iter().any(|s| s.memory.is_some() || has_root_views(&s.translated().host));
        self.bind_heaps(list, memory_heap);
        let root = p.root.as_raw() as usize;
        if self.host_roots[usize::from(compute)] != root {
            // SAFETY: an open list and a live root signature.
            unsafe {
                if compute {
                    l.SetComputeRootSignature(&p.root);
                } else {
                    l.SetGraphicsRootSignature(&p.root);
                }
            }
            self.host_roots[usize::from(compute)] = root;
        }
        // By host object: a title pipeline released and another made at its
        // address is another host pipeline (the list keeps the old one).
        if self.host_pipeline != p.pso.as_raw() as usize {
            // SAFETY: an open list and a live pipeline.
            unsafe { l.SetPipelineState(&p.pso) };
            // Also reached after the indirect clamp invalidates the pipeline.
            if !compute {
                self.apply_depth_bounds(list);
            }
            self.host_pipeline = p.pso.as_raw() as usize;
            // The title may release the pipeline before it submits the list.
            list.bind_pipeline(&p.name, &p.pso, &p.root);
            list.mark_pipeline(&p.name);
        }
        #[cfg(feature = "diagnostics")]
        crate::capture::binding(&p.name);
        // The bound view heap's mark table and the bitmap shaders mark
        // their writes through views in (`crate::shadow::copy_back_marks`).
        let marks = if let Some((table_param, bitmap_param)) = p.marks {
            let heap = match self.heaps[0] {
                0 => crate::memory::heap(),
                h => crate::descriptor::heap(h as This).host(),
            };
            // Shaders using title memory mark in a bitmap of their own, held
            // back while their results are (`crate::memory::withholding`).
            let memory = p.stages.iter().any(|s| s.memory.is_some());
            let (Some(table), Some(bitmap)) = (heap.and_then(|h| h.marks()), crate::shadow::view_writes(memory)) else {
                once!("a shader marking its writes without a view heap: dropped");
                not_run("write marks without a view heap");
                return false;
            };
            list.keep(table);
            list.note_marks();
            // SAFETY: an open list; the table lives while the list holds it,
            // the bitmap for good.
            unsafe {
                let table = table.GetGPUVirtualAddress();
                if compute {
                    l.SetComputeRootShaderResourceView(table_param, table);
                    l.SetComputeRootUnorderedAccessView(bitmap_param, bitmap);
                } else {
                    l.SetGraphicsRootShaderResourceView(table_param, table);
                    l.SetGraphicsRootUnorderedAccessView(bitmap_param, bitmap);
                }
                Some((table_param, table, bitmap_param, bitmap))
            }
        } else {
            None
        };
        // Shaders taking views from the heap itself read (or write) views no
        // table names: everything the bound heap's views reach is brought up
        // to date at submission (`crate::bindless`).
        if p.stages.iter().any(|s| s.translated().host.heap) {
            let writes = p.stages.iter().any(|s| s.translated().host.heap_writes);
            if writes {
                list.note_unknown_gpu_write();
            }
            if self.heaps[0] != 0 {
                let title_heap = crate::descriptor::heap(self.heaps[0] as This);
                if let Some(host) = title_heap.host() {
                    list.defer_heap(&host.viewed);
                    for buffer in host.widened_buffers() {
                        list.mark_widened_read(&buffer);
                    }
                    if writes {
                        list.mark_widened_heap(&host.viewed);
                    }
                    if crate::heap_worker::on() {
                        static KICKED_FRAME: AtomicU64 = AtomicU64::new(u64::MAX);
                        let frame = crate::queue::presented();
                        if KICKED_FRAME.load(Ordering::Relaxed) != frame
                            && KICKED_FRAME.swap(frame, Ordering::Relaxed) != frame
                        {
                            crate::heap_worker::kick(title_heap);
                        }
                    }
                }
            }
            crate::bindless::note_draw(writes);
        }
        for stage in &p.stages {
            let t = stage.translated();
            self.table_failed = false;
            // The view addresses exist before the stage's tables are made:
            // a table made afresh enters its views' addresses there.
            if stage.view_addresses.is_some() && self.heaps[0] != 0 {
                let _ = crate::descriptor::heap(self.heaps[0] as This)
                    .host()
                    .and_then(|h| h.view_addresses());
            }
            let made = self.tables(
                list,
                &t.used,
                &t.used_written,
                t.write_unknown,
                &t.host.half_views,
                compute,
            );
            // A depth copy of ours made readable a texture the tables hold
            // (`crate::twin`), with a compute pipeline: this one's again
            // (root arguments are set below).
            if list.take_rebind() {
                // SAFETY: an open list, a live root signature and pipeline.
                unsafe {
                    if compute {
                        l.SetComputeRootSignature(&p.root);
                        if let Some((tp, table, bp, bitmap)) = marks {
                            l.SetComputeRootShaderResourceView(tp, table);
                            l.SetComputeRootUnorderedAccessView(bp, bitmap);
                        }
                    }
                    l.SetPipelineState(&p.pso);
                }
                self.host_pipeline = p.pso.as_raw() as usize;
                self.host_roots[1] = if compute { root } else { 0 };
            }
            if self.table_failed {
                static DROPPED_TABLES: AtomicU64 = AtomicU64::new(0);
                let n = DROPPED_TABLES.fetch_add(1, Ordering::Relaxed) + 1;
                if n.is_power_of_two() {
                    tracing::warn!(
                        dropped = n,
                        "a draw or dispatch dropped: no dynamic slots for its tables"
                    );
                }
                not_run("no dynamic slots for its tables");
                return false;
            }
            // Root views a shader reads as title memory: the copies holding
            // their addresses go into the page table.
            let mut roots = Vec::new();
            if let Some((param, dwords)) = stage.user_data {
                let count = dwords as usize;
                let mut stack = [0u32; 64];
                let mut overflow = Vec::new();
                let values: &mut [u32] = if count <= stack.len() {
                    &mut stack[..count]
                } else {
                    overflow.resize(count, 0);
                    &mut overflow
                };
                for u in &t.host.user_data {
                    if u.root == RootType::Descriptor
                        && u.dword == 0
                        && let Some(Root::View(address)) =
                            self.roots_ref(compute).get(usize::from(u.slot))
                        && *address != 0
                        && let Some(shadow) = crate::shadow::find(*address, 1)
                    {
                        let uav = t.used.iter().any(|r| {
                            r.slot == u.slot
                                && r.root == RootType::Descriptor
                                && r.kind == ResourceType::Uav
                        });
                        roots.push((*address, uav.then_some(shadow)));
                    }
                    let fresh = made
                        .iter()
                        .find(|m| m.0 == u.slot && u.root == RootType::Table);
                    values[u.index as usize] = match fresh {
                        Some(&(_, offset)) => offset,
                        None if is_root_view_offset(u) => {
                            // Where the root address lies in the copy its view shows.
                            let offset = match self.roots_ref(compute).get(usize::from(u.slot)) {
                                Some(&Root::View(0)) => Some(0),
                                Some(&Root::View(address)) => crate::shadow::find(address, 1)
                                    .and_then(|s| u32::try_from(s.offset(address)).ok()),
                                _ => None,
                            };
                            match offset {
                                Some(offset) => offset,
                                None => {
                                    not_run("a root view's offset: no root view, or outside title buffers");
                                    return false;
                                }
                            }
                        }
                        None if is_root_view(u) => {
                            let uav = u.dword == victory_shader::emit::ROOT_UAV_VIEW;
                            let read_end = t
                                .host
                                .root_srv_bytes
                                .iter()
                                .find_map(|&(slot, end)| (slot == u.slot).then_some(end));
                            match self.root_view(list, compute, u.slot, uav, read_end) {
                                Some((slot, _)) => slot,
                                None => return false,
                            }
                        }
                        None => self.user_value(compute, u.slot, u.root, u.dword),
                    };
                }
                // SAFETY: an open list; `values` covers the parameter.
                unsafe {
                    if compute {
                        l.SetComputeRoot32BitConstants(param, dwords, values.as_ptr().cast(), 0);
                    } else {
                        l.SetGraphicsRoot32BitConstants(param, dwords, values.as_ptr().cast(), 0);
                    }
                }
            }
            for (views, kind) in [(&stage.root_cbvs, 0), (&stage.root_srvs, 1), (&stage.root_uavs, 2)] {
                for &(slot, param) in views {
                    if kind == 1 && slot == victory_shader::emit::VIEW_STRIDES {
                        // The bound view heap's stride table; without one
                        // the shader would read an unbound root view.
                        let heap = (self.heaps[0] != 0)
                            .then(|| crate::descriptor::heap(self.heaps[0] as This).host())
                            .flatten();
                        let Some(table) = heap.and_then(|h| h.strides()) else {
                            once!("a shader reads view strides without a view heap: dropped");
                            not_run("view strides without a view heap");
                            return false;
                        };
                        list.keep(table);
                        // SAFETY: an open list; the table lives while the list
                        // holds it.
                        unsafe {
                            let gpu = table.GetGPUVirtualAddress();
                            if compute {
                                l.SetComputeRootShaderResourceView(param, gpu);
                            } else {
                                l.SetGraphicsRootShaderResourceView(param, gpu);
                            }
                        }
                        continue;
                    }
                    let gpu = match self.roots_ref(compute).get(usize::from(slot)) {
                        Some(Root::View(address)) if *address != 0 => {
                            let found = root_view(list, *address, kind == 1, kind == 2);
                            if found.is_none() {
                                once!(
                                    slot,
                                    address = format_args!("{:#x}", *address),
                                    "a shader uses a direct root view outside title memory"
                                );
                            }
                            found
                        }
                        Some(Root::View(_)) => None,
                        _ => {
                            once!(slot, "a shader reads a root view the title did not set");
                            None
                        }
                    };
                    let Some(gpu) = gpu.or_else(|| {
                        let host = crate::host::host()?;
                        if kind == 2 {
                            host.sink_root_uav()
                        } else {
                            host.zero_root_read()
                        }
                    }) else {
                        not_run("no fallback buffer for a null or unmapped root view");
                        return false;
                    };
                    // SAFETY: an open list; `gpu` lies in a live host buffer.
                    unsafe {
                        match (compute, kind) {
                            (false, 0) => l.SetGraphicsRootConstantBufferView(param, gpu),
                            (false, 1) => l.SetGraphicsRootShaderResourceView(param, gpu),
                            (false, _) => l.SetGraphicsRootUnorderedAccessView(param, gpu),
                            (true, 0) => l.SetComputeRootConstantBufferView(param, gpu),
                            (true, 1) => l.SetComputeRootShaderResourceView(param, gpu),
                            (true, _) => l.SetComputeRootUnorderedAccessView(param, gpu),
                        }
                    }
                }
            }
            if let Some(param) = stage.view_addresses {
                let gpu = (self.heaps[0] != 0)
                    .then(|| crate::descriptor::heap(self.heaps[0] as This).host())
                    .flatten()
                    .and_then(|h| h.view_addresses());
                let Some(gpu) = gpu else {
                    once!("a shader reading a view's address dropped: no view heap");
                    not_run("view addresses without a view heap");
                    return false;
                };
                // SAFETY: an open list; the heap's buffer lives with the heap.
                unsafe {
                    if compute {
                        l.SetComputeRootShaderResourceView(param, gpu);
                    } else {
                        l.SetGraphicsRootShaderResourceView(param, gpu);
                    }
                }
            }
            if let Some(param) = stage.memory {
                let Some((table, dirty)) = self.memory_map(list) else {
                    once!("a shader reading title memory dropped: no view heap or page table");
                    not_run("title memory without a view heap or page table");
                    return false;
                };
                crate::memory::note_use(&p.name);
                // What shaders write through a root UAV goes back by its
                // bytes, as through root views ([`root_view`]).
                for (address, shadow) in &roots {
                    if let (Some(shadow), Some((_, map))) = (shadow, &self.memory) {
                        let len = shadow.left(*address).min(1 << 20);
                        shadow.refresh(list, *address, len);
                        shadow.gpu_writes(list, *address, len, false);
                        map.precise(*address, len);
                    }
                }
                // SAFETY: an open list; list memory alive until it is reset.
                unsafe {
                    if compute {
                        l.SetComputeRootShaderResourceView(param, table);
                    } else {
                        l.SetGraphicsRootShaderResourceView(param, table);
                    }
                    if let Some(param) = stage.memory_dirty {
                        list.note_unknown_gpu_write();
                        if compute {
                            l.SetComputeRootUnorderedAccessView(param, dirty);
                        } else {
                            l.SetGraphicsRootUnorderedAccessView(param, dirty);
                        }
                    }
                }
            }
        }
        if !compute {
            bake_sources::prepare_bake_sources(self, list, p);
        }
        #[cfg(feature = "diagnostics")]
        crate::capture::binding("");
        true
    }

    /// The heap index of a raw view (`uav`: unordered access) of the host
    /// copy holding the root view the title set at root parameter `slot`,
    /// and the root address's byte offset in it: a dynamic slot of the view
    /// heap the shader indexes, made once per list and copy; the range from
    /// the root address brought up to date at submission
    /// (`victory_shader::emit`, "Root SRVs and root UAVs"). None if the
    /// draw must be dropped: no root view set, or no dynamic slot.
    /// The copy is brought up to date from the root address to its end: a
    /// shader can follow offsets the root view itself contains. If the
    /// translator proves all read offsets, only the corresponding prefix
    /// needs refreshing. The host view still covers the whole copy.
    /// shortcut: what shaders write through a root UAV goes back to title
    /// memory for 1 MiB from its address, as before the views.
    fn root_view(
        &mut self,
        list: &HostList,
        compute: bool,
        slot: u8,
        uav: bool,
        read_end: Option<u32>,
    ) -> Option<(u32, u32)> {
        let Some(&Root::View(address)) = self.roots_ref(compute).get(usize::from(slot)) else {
            once!(slot, "a shader reads a root view the title did not set: dropped");
            not_run("a root view the title did not set");
            return None;
        };
        let (key, host) = match self.heaps[0] {
            0 => (MEMORY_HEAP, crate::memory::heap()),
            h => (
                std::ptr::from_ref(crate::descriptor::heap(h as This)) as usize,
                crate::descriptor::heap(h as This).host(),
            ),
        };
        let Some(host) = host else {
            not_run("a root view without a host view heap");
            return None;
        };
        if address == 0 {
            let index = host.null_root_view(uav);
            if index.is_none() {
                not_run("no null root buffer view in the host heap");
            }
            return index.map(|slot| (slot, 0));
        }
        let Some(shadow) = crate::shadow::find(address, 1) else {
            not_run("a root view outside title buffers");
            return None;
        };
        let Some(offset) = u32::try_from(shadow.offset(address)).ok() else {
            not_run("a root view 4 GiB or more into its copy");
            return None;
        };
        let len = shadow.left(address);
        let (refresh_at, refresh_len) = if uav && shadow.marked() {
            shadow.page_range(address, len)
        } else if !uav
            && offset.is_multiple_of(4)
            && let Some(end) = read_end.filter(|&end| offset.checked_add(end).is_some())
        {
            (address, len.min(u64::from(end)))
        } else {
            (address, len)
        };
        shadow.refresh(list, refresh_at, refresh_len);
        if uav {
            // Writes through the view are marked wherever they land in the
            // copy; else declared for 1 MiB.
            let len = if shadow.marked() { shadow.left(address) } else { shadow.left(address).min(1 << 20) };
            shadow.gpu_writes(list, address, len, true);
        }
        let copy = std::sync::Arc::as_ptr(&shadow) as usize;
        if let Some(&index) = self.root_slots.get(&(key, copy, uav)) {
            return Some((index, offset));
        }
        let (run_heap, next, end) = self.root_run;
        if run_heap != key || next >= end {
            let Some(first) = host.dynamic(ROOT_VIEW_RUN) else {
                once!("no dynamic slots for root views: dropped");
                not_run("no dynamic slots for root views");
                return None;
            };
            self.dynamic.push((key, host.id(), first, ROOT_VIEW_RUN));
            self.root_run = (key, first, first + ROOT_VIEW_RUN);
        }
        let index = self.root_run.1;
        self.root_run.1 += 1;
        ROOT_VIEW_SLOTS.fetch_add(1, Ordering::Relaxed);
        // To the copy's end in its buffer: `offset` counts from the buffer's start.
        let (start, len) = shadow.range();
        host.raw_view_at(
            index,
            Some(&shadow.buffer),
            0,
            shadow.offset(start + len) & !3,
            uav,
        );
        if uav {
            host.set_mark(index, shadow.mark_entry(shadow.range().0));
        }
        self.root_slots.insert((key, copy, uav), index);
        Some((index, offset))
    }

    /// Gives the host list the title's heaps if it has others.
    /// The heaps a list must have bound before a root signature that indexes
    /// them directly is set: the title's of each kind, and ours for a kind it
    /// binds none of — D3D12 requires a heap of both kinds to be bound before
    /// such a signature (the debug layer's id 1419), and breaking that rule
    /// removes the device on NVIDIA. Compute lists that only read views are
    /// the case: they bind no sampler heap, and every signature we make says
    /// its sampler heap is indexed directly.
    fn wanted(title: [usize; 2], memory_heap: bool) -> [usize; 2] {
        let sampler = if title[1] != 0 {
            title[1]
        } else {
            SAMPLERS_OURS
        };
        let views = if memory_heap { MEMORY_HEAP } else { title[0] };
        [views, sampler]
    }

    /// Gives the host list the title's heaps if it has others.
    fn bind_heaps(&mut self, list: &HostList, memory_heap: bool) {
        let want = Self::wanted(self.heaps, memory_heap);
        if self.host_heaps != want {
            let mut heaps: Vec<Option<ID3D12DescriptorHeap>> = self
                .heaps
                .iter()
                .filter(|&&h| h != 0 && h != SAMPLERS_OURS)
                .filter_map(|&h| crate::descriptor::heap(h as This).host())
                .map(|h| Some(h.heap.clone()))
                .collect();
            if memory_heap && let Some(h) = crate::memory::heap() {
                heaps.insert(0, Some(h.heap.clone()));
            }
            if want[1] == SAMPLERS_OURS {
                match crate::descriptor::fallback_samplers() {
                    Some(h) => {
                        tracing::debug!("the title binds no sampler heap: binding ours");
                        heaps.push(Some(h.heap.clone()));
                    }
                    None => {
                        once!("no sampler heap of ours to bind");
                    }
                }
            }
            heaps.iter().flatten().for_each(|h| list.keep(h));
            // SAFETY: an open list and shader-visible heaps.
            unsafe { list.list.SetDescriptorHeaps(&heaps) };
            self.host_heaps = want;
            self.host_roots = [0; 2];
        }
    }

    /// `ClearUnorderedAccessViewUint` or `...Float` (`float`): the view the
    /// title made at `cpu` (its CPU handle; the GPU one names the same view
    /// in the bound heap) cleared to `values` on the host, through a
    /// dynamic slot of the bound view heap that the list holds until reset.
    pub(crate) fn clear_unordered(
        &mut self,
        list: &HostList,
        cpu: u64,
        values: [u32; 4],
        float: bool,
        rects: &[RECT],
    ) {
        let Some(view) = crate::descriptor::view(cpu) else {
            once!("an unordered access clear of a slot without a view");
            return;
        };
        let heap = (self.heaps[0] != 0).then(|| crate::descriptor::heap(self.heaps[0] as This));
        let (Some(heap), Some(host)) = (heap, crate::host::host()) else {
            once!("an unordered access clear without a bound view heap");
            return;
        };
        let Some(visible) = heap.host() else { return };
        let Some(slot) = visible.dynamic(1) else {
            return;
        };
        self.dynamic
            .push((std::ptr::from_ref(heap) as usize, visible.id(), slot, 1));
        visible.make_at(slot, &view);
        let at = host.cpu_view();
        let Some(resource) = crate::descriptor::host::make_view(&view, at) else {
            once!("an unordered access clear of a view the host cannot make");
            return;
        };
        let written = UsedResource {
            kind: ResourceType::Uav,
            root: RootType::Table,
            table_offset_dwords: 0,
            lower_bound: 0,
            stride: 0,
            dwords: 8,
            user_data: 0,
            slot: 0,
        };
        readable(list, &written, &view, false);
        self.bind_heaps(list, false);
        list.keep(&resource);
        let gpu = visible.gpu(slot);
        // SAFETY: an open list; both slots view the live resource.
        unsafe {
            if float {
                let values = values.map(f32::from_bits);
                list.list
                    .ClearUnorderedAccessViewFloat(gpu, at, &resource, &values, rects);
            } else {
                list.list
                    .ClearUnorderedAccessViewUint(gpu, at, &resource, &values, rects);
            }
        }
    }

    /// Runs a dispatch on the host.
    pub(crate) fn dispatch(&mut self, list: &HostList, x: u32, y: u32, z: u32) {
        self.dispatch_x(list, x, y, z, 0);
    }

    /// Runs a dispatch with the console's dispatch flags (`DispatchX`), of
    /// which the host follows [`THREAD_DIMENSIONS`]: the counts are then
    /// threads, run as the groups that hold them. Where the last group
    /// along x is a part of one, the pipeline's form whose lanes past the
    /// count leave runs instead (`crate::pipeline::thread_limited`).
    pub(crate) fn dispatch_x(&mut self, list: &HostList, x: u32, y: u32, z: u32, flags: u32) {
        #[cfg(feature = "diagnostics")]
        if crate::pipeline::is_skipped(self.pipeline as This) {
            DROPPED.fetch_add(1, Ordering::Relaxed);
            dropped(list, self.pipeline, "dispatch");
            return;
        }
        // The acceleration-structure builder runs whole or not at all
        // (`pipeline::builder`).
        if crate::pipeline::held_back(self.pipeline as This) {
            once!("a dispatch of the BVH builder held back");
            DROPPED.fetch_add(1, Ordering::Relaxed);
            #[cfg(feature = "diagnostics")]
            dropped(list, self.pipeline, "dispatch");
            return;
        }
        // A ray pass with no structure to work on (`VICTORY_RAY_SKIP`,
        // `crate::ray_skip`): left out, and counted there, not as dropped.
        if crate::pipeline::ray_pass_left_out(self.pipeline as This) {
            #[cfg(feature = "diagnostics")]
            list.note_op(|| {
                format!(
                    "LEFT OUT dispatch {x}x{y}x{z} (VICTORY_RAY_SKIP) {}",
                    crate::pipeline::shaders_of(self.pipeline as This)
                )
            });
            return;
        }
        let Some(mut p) = crate::pipeline::host_pipeline(self.pipeline as This) else {
            once!(
                pipeline = format_args!("{:#x}", self.pipeline),
                "a dispatch without a host pipeline dropped"
            );
            DROPPED.fetch_add(1, Ordering::Relaxed);
            #[cfg(feature = "diagnostics")]
            dropped(list, self.pipeline, "dispatch");
            return;
        };
        if !p.compute {
            once!("a dispatch with a graphics pipeline");
            return;
        }
        let threads = [x, y, z];
        let counted = flags & THREAD_DIMENSIONS != 0 && thread_dimensions();
        let size = counted
            .then(|| p.stages.first().and_then(|s| s.translated().threads))
            .flatten();
        if counted && size.is_none() {
            once!(pipeline = %p.name, "a dispatch counted in threads without the program's group size: run as groups");
        }
        let [x, y, z] = size.map_or(threads, |size| dispatch_groups(threads, size));
        // The word that tells such a form's lanes how many threads run.
        let mut limit = None;
        if let Some(size) = size {
            let partial = (0..3).any(|i| !threads[i].is_multiple_of(size[i]));
            let limited = (partial && size[1..] == [1, 1])
                .then(|| crate::pipeline::thread_limited(p))
                .flatten();
            note_counted_in_threads(&p.name, threads, size, partial, limited.is_some());
            if let Some(limited) = limited {
                p = limited;
                limit = compute_word(p, victory_shader::emit::USER_DISPATCH_THREADS_X);
            }
        }
        if !self.bind(list, p, true) {
            DROPPED.fetch_add(1, Ordering::Relaxed);
            #[cfg(feature = "diagnostics")]
            dropped(list, self.pipeline, "dispatch");
            return;
        }
        if let Some((param, word)) = limit {
            // SAFETY: an open list; the root signature has the word (`bind`
            // set it to 0, all of every group).
            unsafe {
                list.list
                    .SetComputeRoot32BitConstant(param, threads[0], word)
            };
        }
        #[cfg(feature = "diagnostics")]
        list.note_op(|| {
            self.op_text(
                format!("dispatch {x}x{y}x{z} (asked {threads:?}, flags {flags:#x})"),
                true,
            )
        });
        #[cfg(feature = "diagnostics")]
        self.probe_small(list, true);
        crate::census::direct(&p.name, [x, y, z], p.stages.first().and_then(|s| s.translated().threads));
        // SAFETY: an open list with everything set.
        unsafe { list.list.Dispatch(x, y, z) };
        self.probe_hdr_compute(list);
        #[cfg(feature = "diagnostics")]
        self.probe_written(list);
        DISPATCHED.fetch_add(1, Ordering::Relaxed);
        self.ambient(list);
    }

    /// After a dispatch: the stand-in for ray-traced indirect light
    /// (`crate::ambient`, `VICTORY_AMBIENT=1`; from Belobug 0603123), if the
    /// program is one it follows. Its pass sets a view heap, a compute root
    /// signature and a pipeline of its own: the title's are set again by the
    /// next draw or dispatch.
    fn ambient(&mut self, list: &HostList) {
        if !crate::ambient::on() {
            return;
        }
        let role = crate::pipeline::ambient_role(self.pipeline as This);
        let estimate = crate::ambient_estimate::roles();
        if role & (crate::ambient::RESOLVES | crate::ambient::ENDS_REFLECTIONS | estimate) == 0 {
            return;
        }
        let Some(stage) =
            crate::pipeline::host_pipeline(self.pipeline as This).and_then(|p| p.stages.first())
        else {
            return;
        };
        // The texture viewed at a place of one of the program's tables, if
        // its code uses that place as `kind`; the view stays pinned while
        // the pass is recorded.
        let image = |kind: ResourceType, (slot, offset): (u8, u32)| {
            stage.translated().used.iter().find(|u| {
                u.kind == kind
                    && u.root == RootType::Table
                    && (u.slot, u.table_offset_dwords) == (slot, offset)
            })?;
            let Some(Root::Table(table)) = self.roots_ref(true).get(usize::from(slot)) else {
                return None;
            };
            let pin = crate::descriptor::view(table + u64::from(offset) * 4)?;
            // A texture the title placed in memory itself: the resource made
            // for that memory.
            let placed = crate::descriptor::host::placed_texture(&pin);
            let texture = match (placed.as_ref().unwrap_or(&pin), kind) {
                (&View::Unordered(r, ..), ResourceType::Uav) => r,
                (&View::Shader(r, _), ResourceType::Srv) => r,
                _ => 0,
            };
            (texture != 0).then_some((pin, texture as This))
        };
        let recorded = if crate::ambient_estimate::on() {
            // The estimate in the constant's place (`VICTORY_AMBIENT=estimate`,
            // from avatarmanz 82ca1347).
            use crate::ambient_estimate as estimate;
            if role & estimate::KEEPS != 0 {
                match &image(ResourceType::Uav, estimate::RADIANCE) {
                    Some((_, radiance)) => estimate::keep(list, *radiance),
                    None => {
                        once!("no ambient estimate: the light visible surfaces send out is not at its place in the table");
                        false
                    }
                }
            } else if role & estimate::LIGHTS_AIR != 0 {
                // (Nothing unless `VICTORY_AMBIENT_AIR` is set.)
                match &image(ResourceType::Uav, estimate::AIR) {
                    Some((_, scattered)) => estimate::air(list, *scattered),
                    None => false,
                }
            } else if role & crate::ambient::RESOLVES != 0 {
                let light = image(ResourceType::Uav, crate::ambient::LIGHT);
                let depth = image(ResourceType::Srv, crate::ambient::DEPTH);
                match &light {
                    Some((_, light)) => estimate::raise(list, *light, depth.as_ref().map(|d| d.1)),
                    None => {
                        once!("no ambient estimate: the resolve's light image is not at its place in the table");
                        false
                    }
                }
            } else {
                // The reflections, held to the light around.
                match &image(ResourceType::Uav, crate::ambient::REFLECTIONS) {
                    Some((_, reflections)) => estimate::hold(list, *reflections),
                    None => false,
                }
            }
        } else if role & crate::ambient::RESOLVES != 0 {
            let light = image(ResourceType::Uav, crate::ambient::LIGHT);
            let depth = image(ResourceType::Srv, crate::ambient::DEPTH);
            match &light {
                Some((_, light)) => {
                    crate::ambient::raise(list, *light, depth.as_ref().map(|d| d.1))
                }
                None => {
                    once!("no ambient light: the resolve's light image is not at its place in the table");
                    false
                }
            }
        } else {
            // The reflections, held to the light around.
            match &image(ResourceType::Uav, crate::ambient::REFLECTIONS) {
                Some((_, reflections)) => crate::ambient::hold(list, *reflections),
                None => false,
            }
        };
        if recorded {
            #[cfg(feature = "diagnostics")]
            list.note_op(|| "the ambient light's pass (VICTORY_AMBIENT)".into());
            self.host_state_lost();
        }
    }

    /// The title pipeline the next draw or dispatch runs (null: none set).
    #[cfg(feature = "diagnostics")]
    pub(crate) fn title_pipeline(&self) -> usize {
        self.pipeline
    }

    /// Flushes any pending RTV probes recorded during `VICTORY_HDR_PROBE`.
    pub(crate) fn flush_hdr_probe(&mut self, list: &HostList) {
        let Some((targets, shaders)) = self.hdr_rtvs.take() else {
            return;
        };
        if !crate::hdr_probe::active() {
            return;
        }
        const RENDER_TARGET: u32 = 0x4;
        for resource in targets {
            crate::hdr_probe::probe_texture(list, resource as This, RENDER_TARGET, "rtv", &shaders);
        }
    }

    /// Marks the currently bound RTVs dirty for `VICTORY_HDR_PROBE`; zero cost
    /// outside the single targeted world frame.
    fn note_hdr_draw(&mut self) {
        if !crate::hdr_probe::active() || self.targets.is_empty() {
            return;
        }
        let Some(p) = crate::pipeline::host_pipeline(self.pipeline as This) else {
            return;
        };
        self.hdr_rtvs = Some((self.targets.clone(), p.name.clone()));
    }

    /// Probes HDR/color UAV textures written by the just-recorded compute
    /// dispatch when `VICTORY_HDR_PROBE` is active.
    fn probe_hdr_compute(&mut self, list: &HostList) {
        if !crate::hdr_probe::active() {
            return;
        }
        let Some(p) = crate::pipeline::host_pipeline(self.pipeline as This) else {
            return;
        };
        const UNORDERED: u32 = 0x8;
        for stage in &p.stages {
            for u in stage.translated().used.iter().filter(|u| u.kind == ResourceType::Uav) {
                let (RootType::Table, Some(Root::Table(table))) =
                    (u.root, self.compute.get(usize::from(u.slot)))
                else {
                    continue;
                };
                let Some(view) =
                    crate::descriptor::view(table + u64::from(u.table_offset_dwords) * 4)
                else {
                    continue;
                };
                let placed = crate::descriptor::host::placed_texture(&view);
                let view = placed.as_ref().unwrap_or(&view);
                let View::Unordered(r, ..) = view else {
                    continue;
                };
                let key = (self.pipeline, *r);
                if self.hdr_last_compute == key {
                    continue;
                }
                self.hdr_last_compute = key;
                crate::hdr_probe::probe_texture(list, *r as This, UNORDERED, "uav", &p.name);
            }
        }
    }

    /// What a draw or dispatch runs, for the device-removed report
    /// (diagnostics): the pipeline, then per stage the user data it gets
    /// (with the view at that heap index, if one is), the host addresses
    /// of its root views and the views its tables name.
    #[cfg(feature = "diagnostics")]
    fn op_text(&self, kind: String, compute: bool) -> String {
        use std::fmt::Write;
        let Some(p) = crate::pipeline::host_pipeline(self.pipeline as This) else {
            return kind;
        };
        let mut text = format!("{kind} {}", p.name);
        let views = (self.heaps[0] != 0)
            .then(|| crate::descriptor::heap(self.heaps[0] as This).host())
            .flatten();
        for stage in &p.stages {
            let t = stage.translated();
            let _ = write!(text, " | {}:", stage.shader.hash);
            for u in t.host.user_data.iter().filter(|u| !is_root_view(u)) {
                let v = self.user_value(compute, u.slot, u.root, u.dword);
                let _ = write!(text, " u{}={v:#x}", u.index);
                let index = if u.root == RootType::Table { v / 32 } else { v };
                if let Some(slot) = views.and_then(|h| h.describe_slot(index)) {
                    let _ = write!(text, " [heap {index}: {slot}]");
                    // A slot the host never made: what the title's memory
                    // holds there (our record, zeros, or words it wrote itself).
                    if slot == "null" && self.heaps[0] != 0 {
                        let at = crate::descriptor::heap(self.heaps[0] as This).start + u64::from(index) * 32;
                        let mut words = [0u8; 32];
                        if crate::host::read_into(at as *const u8, &mut words) {
                            let w: Vec<u32> =
                                words.chunks(4).map(|c| u32::from_le_bytes([c[0], c[1], c[2], c[3]])).collect();
                            let _ = write!(text, " (title words {w:x?}");
                            if let Some(view) = crate::descriptor::view(at) {
                                let _ = write!(text, ": {}", describe(Some(&*view)));
                            }
                            let _ = write!(text, ")");
                        }
                    }
                }
            }
            let viewed = t.host.user_data.iter().filter(|u| is_root_view(u)).map(|u| u.slot);
            let slots: std::collections::BTreeSet<u8> =
                stage.root_cbvs.iter().chain(&stage.root_srvs).map(|r| r.0).chain(viewed).collect();
            for slot in slots {
                if let Some(Root::View(address)) = self.roots_ref(compute).get(usize::from(slot)) {
                    let host = crate::shadow::find(*address, 1)
                        .map(|s| (s.gpu(*address), s.left(*address)));
                    // Its first dwords (a vertex stream header, say).
                    let mut head = [0u8; 16];
                    let head = crate::host::read_into(*address as *const u8, &mut head)
                        .then(|| head.chunks(4).map(|c| u32::from_le_bytes([c[0], c[1], c[2], c[3]])).collect::<Vec<_>>());
                    let _ = write!(text, " root{slot}={address:#x} host {host:x?} head {head:x?}");
                }
            }
        }
        let _ = write!(text, " tables {:?}", self.reads(compute));
        if !compute {
            let mem = |r: usize| crate::resource::texture_memory(r as This).map_or(0, |(p, _)| p as u64);
            let targets: Vec<String> = self.targets.iter().map(|&t| format!("{t:#x} mem {:#x}", mem(t))).collect();
            let _ = write!(text, " targets {targets:?} depth {:#x} mem {:#x}", self.depth, mem(self.depth));
        }
        text
    }


    /// Runs a mesh dispatch of `x` groups on the host (the title's mesh
    /// programs are one-dimensional).
    pub(crate) fn dispatch_mesh(&mut self, list: &HostList, x: u32) {
        #[cfg(feature = "diagnostics")]
        self.trace("mesh", x, 0);
        let ran = self.prepare(list, false, true)
            && match list.list.cast::<ID3D12GraphicsCommandList6>() {
                Ok(l6) => {
                    #[cfg(feature = "diagnostics")]
                    list.note_op(|| self.op_text(format!("mesh {x}"), false));
                    // SAFETY: an open list with everything set.
                    unsafe { l6.DispatchMesh(x, 1, 1) };
                    self.note_hdr_draw();
                    DRAWN.fetch_add(1, Ordering::Relaxed);
                    true
                }
                Err(_) => false,
            };
        if !ran {
            DROPPED.fetch_add(1, Ordering::Relaxed);
            #[cfg(feature = "diagnostics")]
            dropped(list, self.pipeline, "mesh");
        }
    }

    fn roots_ref(&self, compute: bool) -> &Vec<Root> {
        if compute {
            &self.compute
        } else {
            &self.graphics
        }
    }

    fn entry_begin(
        &mut self,
        list: &HostList,
        ready_slot: Option<(&crate::descriptor::host::HostHeap, u32)>,
        handle: u64,
        kind: ResourceType,
    ) -> EntryMemo {
        let mode = table_memo();
        if mode == 0 {
            return EntryMemo::Off;
        }
        // SAFETY: `handle` has been validated as belonging to a live title
        // descriptor heap by the caller.
        let bytes = unsafe { (handle as *const [u64; 4]).read_unaligned() };
        let tested = TESTED_DEPTH.with(std::cell::Cell::get);
        let key = (handle, kind, tested.0);
        let epoch = list.epoch();
        let deferred_epoch = list.deferred_epoch();
        memo_count(0);
        let (hit, depth_hit) = match self.entry_memo.get(&key) {
            None => {
                memo_count(5);
                (false, false)
            }
            Some(&(saved_bytes, _, _, _)) if saved_bytes != bytes => {
                memo_count(5);
                (false, false)
            }
            Some(&(_, saved_epoch, _, seen))
                if saved_epoch
                    != if matches!(seen, Seen::Nothing) {
                        deferred_epoch
                    } else {
                        epoch
                    } =>
            {
                memo_count(6);
                (false, false)
            }
            Some(&(_, _, depth_hit, seen)) => {
                let slot_ok = ready_slot.is_none_or(|(host, index)| host.slot_ready(index, handle));
                if slot_ok && copy_stands(seen, handle) {
                    (true, depth_hit)
                } else {
                    memo_count(7);
                    (false, false)
                }
            }
        };
        if hit {
            memo_count(1);
            if mode == 1 {
                if depth_hit {
                    TESTED_DEPTH.with(|d| d.set((tested.0, true)));
                }
                return EntryMemo::Skip;
            }
        }
        TESTED_DEPTH.with(|d| d.set((tested.0, false)));
        EntryMemo::Do {
            key,
            bytes,
            hit,
            epoch,
            deferred_epoch,
            steps: list.deferred_steps(),
            depth_before: tested,
            depth_hit,
        }
    }

    fn entry_end(&mut self, list: &HostList, memo: EntryMemo, view: Option<&View>) {
        let EntryMemo::Do {
            key,
            bytes,
            hit,
            epoch,
            deferred_epoch,
            steps,
            depth_before,
            depth_hit,
        } = memo
        else {
            return;
        };
        let depth = TESTED_DEPTH.with(std::cell::Cell::get).1;
        TESTED_DEPTH.with(|d| d.set((depth_before.0, depth_before.1 || depth)));
        if hit {
            if list.epoch() != epoch || list.deferred_epoch() != deferred_epoch {
                memo_count(2);
                static SAID: AtomicU64 = AtomicU64::new(0);
                if SAID.fetch_add(1, Ordering::Relaxed) < 20 {
                    tracing::warn!(
                        handle = format_args!("{:#x}", key.0),
                        kind = ?key.1,
                        tested = format_args!("{:#x}", key.2),
                        ?view,
                        "table memo (verify): remembered entry changed list epoch"
                    );
                }
            }
            if list.deferred_steps() != steps {
                memo_count(3);
                static SAID_STEPS: AtomicU64 = AtomicU64::new(0);
                if SAID_STEPS.fetch_add(1, Ordering::Relaxed) < 20 {
                    tracing::warn!(
                        handle = format_args!("{:#x}", key.0),
                        kind = ?key.1,
                        steps_before = steps,
                        steps_after = list.deferred_steps(),
                        "table memo (verify): remembered entry changed deferred steps"
                    );
                }
            }
            if depth != depth_hit {
                memo_count(4);
                static SAID_DEPTH: AtomicU64 = AtomicU64::new(0);
                if SAID_DEPTH.fetch_add(1, Ordering::Relaxed) < 20 {
                    tracing::warn!(
                        handle = format_args!("{:#x}", key.0),
                        kind = ?key.1,
                        depth_before = depth_hit,
                        depth_after = depth,
                        "table memo (verify): remembered entry changed depth test"
                    );
                }
            }
        }
        if self.entry_memo.len() >= 1 << 16 {
            self.entry_memo.clear();
        }
        match view.and_then(|v| memo_copy(v, key.1)) {
            Some(seen) => {
                let required_epoch = if matches!(seen, Seen::Nothing) {
                    list.deferred_epoch()
                } else {
                    list.epoch()
                };
                self.entry_memo
                    .insert(key, (bytes, required_epoch, depth, seen));
            }
            None => {
                memo_count(8);
                self.entry_memo.remove(&key);
            }
        }
    }

    fn table_entry(&mut self, list: &HostList, u: &UsedResource, compute: bool, writes: bool) {
        let Some(Root::Table(table)) = self.roots_ref(compute).get(usize::from(u.slot)) else {
            return;
        };
        let handle = table + u64::from(u.table_offset_dwords) * 4;
        let Some(heap) = self.heap_of(handle) else {
            return;
        };
        let (Some(host), Some(index)) = (heap.host(), heap.index(handle)) else {
            return;
        };
        let samplers = heap.samplers();
        let memo = if writes {
            self.entry_begin(list, Some((host, index)), handle, u.kind)
        } else {
            EntryMemo::Off
        };
        if matches!(memo, EntryMemo::Skip) {
            return;
        }
        let view = host.mirror(index, handle, true);
        if let Some(view) = &view {
            readable_entry(list, u, view, true, writes);
        }
        if !samplers {
            list.defer_slot(&host.viewed, index);
        }
        self.entry_end(list, memo, view.as_deref());
    }

    /// The descriptors a shader reads through tables, made from the
    /// title's slots. Constant buffers there are raw views of their data
    /// (translated shaders read them so: the host wants constant buffer
    /// views 256-byte aligned, the console only 16), brought up to date at
    /// submission like other buffer views. Returns the byte offsets of
    /// tables made afresh in dynamic slots, by root parameter.
    ///
    /// Within a list, the same table holding the same descriptors is made
    /// once.
    ///
    /// `half`: the table descriptors the shader moves sixteen bits a channel
    /// as unsigned integers (`HostShader::half_views`). A table with a 16-bit
    /// float or signed buffer view at one of them is made afresh as well,
    /// with that view in the unsigned format of the same layout
    /// (`descriptor::host::half_view`): the title's own slot keeps the
    /// title's format for shaders that read it as floats.
    pub(crate) fn tables(
        &mut self,
        list: &HostList,
        used: &[UsedResource],
        written: &[bool],
        write_unknown: bool,
        half: &[(u8, u32)],
        compute: bool,
    ) -> Vec<(u8, u32)> {
        let writes = |u: &UsedResource| -> bool {
            if write_unknown || !table_uav_write_skip() {
                return true;
            }
            used.iter()
                .position(|x| std::ptr::eq(x, u))
                .and_then(|i| written.get(i).copied())
                .unwrap_or(true)
        };
        let prev_depth = TESTED_DEPTH.with(|d| d.replace((if compute { 0 } else { self.depth }, false)));
        let tables = |u: &&UsedResource| u.root == RootType::Table;
        let mut fresh: Vec<u8> = Vec::new();
        // The view shown in place of the title's at dword `at` of the table
        // in root parameter `slot`, if the shader moves halves there and the
        // title's view is a 16-bit float or signed buffer view.
        let halved = |state: &Self, slot: u8, at: u32| -> Option<View> {
            if !half.contains(&(slot, at)) {
                return None;
            }
            let Some(&Root::Table(table)) = state.roots_ref(compute).get(usize::from(slot)) else {
                return None;
            };
            let view = crate::descriptor::view(table + u64::from(at) * 4)?;
            crate::descriptor::host::half_view(&view)
        };
        for &(slot, at) in half {
            let Some(&Root::Table(table)) = self.roots_ref(compute).get(usize::from(slot)) else {
                continue;
            };
            let Some(view) = crate::descriptor::view(table + u64::from(at) * 4) else {
                continue;
            };
            if crate::descriptor::host::half_view(&view).is_some() {
                if !fresh.contains(&slot) {
                    fresh.push(slot);
                }
            } else if let Some(format) = crate::descriptor::host::half_inexact(&view) {
                once!(
                    format,
                    "a shader moves a buffer's 16-bit channels as integers through a view whose values the console converts: they are read and written unconverted"
                );
            }
        }
        // Tables reaching past a host heap smaller than the title's (the
        // host's limit, 1,000,000 views on NVIDIA) are made afresh too: the
        // title's slots there have no host slot of their own.
        for u in used.iter().filter(tables) {
            if fresh.contains(&u.slot) {
                continue;
            }
            let Some(&Root::Table(table)) = self.roots_ref(compute).get(usize::from(u.slot)) else {
                continue;
            };
            let end = table + u64::from(u.table_offset_dwords) * 4;
            let Some(heap) = self.heap_of(table) else {
                continue;
            };
            let (Some(host), Some(index)) = (heap.host(), heap.index(end)) else {
                continue;
            };
            if index >= host.count() {
                fresh.push(u.slot);
            }
        }
        fresh.sort_unstable();
        fresh.dedup();
        for u in used.iter().filter(tables) {
            if !fresh.contains(&u.slot) {
                self.table_entry(list, u, compute, writes(u));
            }
        }
        let mut made = Vec::new();
        for slot in fresh {
            let Some(&Root::Table(table)) = self.roots_ref(compute).get(usize::from(slot)) else {
                continue;
            };
            let entries: Vec<&UsedResource> = used
                .iter()
                .filter(tables)
                .filter(|u| u.slot == slot)
                .collect();
            let Some(heap) = self.heap_of(table) else {
                continue;
            };
            let Some(host) = heap.host() else {
                continue;
            };
            // The descriptors the shader moves halves through are made too,
            // and the table reaches them: they may lie past the range starts
            // the entries give (a range of several views names only its first).
            let n = entries
                .iter()
                .map(|u| u.table_offset_dwords / 8 + 1)
                .chain(
                    half.iter()
                        .filter(|&&(s, _)| s == slot)
                        .map(|&(_, at)| at / 8 + 1),
                )
                .max()
                .unwrap_or(1);
            // The table and what its entries hold (record words).
            let key = {
                use std::hash::{Hash, Hasher};
                let mut h = std::collections::hash_map::DefaultHasher::new();
                n.hash(&mut h);
                for k in 0..n {
                    let at = table + u64::from(k) * 32;
                    // SAFETY: a descriptor slot in title memory.
                    unsafe { (at as *const [u32; 3]).read_unaligned() }.hash(&mut h);
                    // Another table than the same slots made for a shader
                    // that reads them in the title's formats.
                    half.contains(&(slot, k * 8)).hash(&mut h);
                }
                (table, h.finish())
            };
            if let Some(&first) = self.fresh_tables.get(&key) {
                for u in entries.iter().filter(|u| u.kind != ResourceType::Cbv) {
                    let handle = table + u64::from(u.table_offset_dwords) * 4;
                    let memo = if writes(u) {
                        self.entry_begin(list, None, handle, u.kind)
                    } else {
                        EntryMemo::Off
                    };
                    if matches!(memo, EntryMemo::Skip) {
                        continue;
                    }
                    let view = crate::descriptor::view(handle);
                    if let Some(view) = &view {
                        readable_entry(list, u, view, true, writes(u));
                    }
                    self.entry_end(list, memo, view.as_deref());
                }
                made.push((slot, first * 32));
                continue;
            }
            let Some(first) = host.dynamic(n) else {
                // Without its fresh table the shader would read the title's
                // slots, which for constant buffer tables hold no constant
                // view: a texture or null descriptor read as a buffer gives
                // a wild address on AMD and the device is removed (#59).
                // The draw or dispatch is dropped instead.
                self.table_failed = true;
                continue;
            };
            self.fresh_tables.insert(key, first);
            self.dynamic
                .push((std::ptr::from_ref(heap) as usize, host.id(), first, n));
            // The whole range: a shader may index past the entries its
            // used-resource list names (descriptor arrays).
            for k in 0..n {
                let index = first + k;
                let used_entry = entries.iter().find(|u| u.table_offset_dwords / 8 == k);
                let view = crate::descriptor::view(table + u64::from(k) * 32);
                match view.as_deref() {
                    Some(view) => {
                        match halved(self, slot, k * 8) {
                            Some(twin) => host.make_at(index, &twin),
                            None => host.make_at(index, view),
                        }
                        if let Some(u) = used_entry {
                            readable_entry(list, u, view, true, writes(u));
                        }
                    }
                    None => {
                        host.make_null(index);
                        if used_entry.is_some() {
                            once!(slot, "a table entry without a view");
                        }
                    }
                }
            }
            made.push((slot, first * 32));
        }
        TESTED_DEPTH.with(|d| d.set(prev_depth));
        made
    }

    /// The first draws, at debug level: what they bind and read
    /// (diagnostics builds).
    #[cfg(feature = "diagnostics")]
    fn trace(&self, kind: &str, count: u32, first: u32) {
        static TRACED: AtomicU64 = AtomicU64::new(0);
        if !tracing::enabled!(tracing::Level::DEBUG) {
            return;
        }
        let listed = crate::capture::listed(crate::queue::presented() + 1);
        let logged = listed || TRACED.fetch_add(1, Ordering::Relaxed) < 64;
        let targets: Vec<String> = self
            .targets
            .iter()
            .filter_map(|&t| crate::resource::host_resource(t as This))
            .map(|t| {
                // SAFETY: a live resource.
                let d = unsafe { t.GetDesc() };
                format!("{}x{} f{}", d.Width, d.Height, d.Format.0)
            })
            .collect();
        let v = self.vertex[0];
        let line = format!(
            "{kind} {count} from {first} topology {} pipeline {:#x} targets {targets:?} vb {:#x}+{} stride {} reads {:?}",
            self.topology,
            self.pipeline,
            v.address,
            v.size,
            v.stride,
            self.reads(false)
        );
        if logged {
            tracing::debug!("draw {line}");
        }
        let mut frame = FRAME.lock().unwrap_or_else(|e| e.into_inner());
        if frame.len() < 256 {
            frame.push(line);
        }
    }

    /// In a listed frame: each small buffer (64 bytes at most) the draw's
    /// tables name, as title memory now and as its host copy when the list
    /// is submitted (before the GPU runs it) and after it has run.
    #[cfg(feature = "diagnostics")]
    fn probe_small(&self, list: &HostList, compute: bool) {
        if !crate::capture::listed(crate::queue::presented() + 1) {
            return;
        }
        let Some(p) = crate::pipeline::host_pipeline(self.pipeline as This) else { return };
        for stage in &p.stages {
            for u in &stage.translated().used {
                let (RootType::Table, Some(Root::Table(table))) =
                    (u.root, self.roots_ref(compute).get(usize::from(u.slot)))
                else {
                    continue;
                };
                let Some(view) = crate::descriptor::view(table + u64::from(u.table_offset_dwords) * 4) else {
                    continue;
                };
                let range = match &*view {
                    View::Constant(address, size) => Some((*address, u64::from(*size))),
                    v => crate::descriptor::host::buffer_range(v),
                };
                let Some((at, len)) = range.filter(|r| r.1 <= 64) else {
                    continue;
                };
                let Some(shadow) = crate::shadow::find(at, len) else { continue };
                let who = format!("{} {:?} s{}+{} {at:#x}+{len}", stage.shader.hash, u.kind, u.slot, u.table_offset_dwords);
                let (s, w) = (shadow.clone(), who.clone());
                list.defer_host(move || tracing::info!(target: "victory_gpu::trace", "small {w} at submission: {}", s.peek(at, len)));
                list.defer_after(move || tracing::info!(target: "victory_gpu::trace", "small {who} after the list: {}", shadow.peek(at, len)));
            }
        }
    }

    /// In a listed frame, right after a dispatch: each 1D or 3D texture it
    /// writes through a table, read back in the list itself (what the
    /// dispatch left there, before anything later in the frame).
    #[cfg(feature = "diagnostics")]
    fn probe_written(&self, list: &HostList) {
        if !crate::capture::listed(crate::queue::presented() + 1) {
            return;
        }
        let (Some(p), Some(host)) = (crate::pipeline::host_pipeline(self.pipeline as This), crate::host::host()) else {
            return;
        };
        for stage in &p.stages {
            for u in stage.translated().used.iter().filter(|u| u.kind == ResourceType::Uav) {
                let (RootType::Table, Some(Root::Table(table))) = (u.root, self.compute.get(usize::from(u.slot))) else {
                    continue;
                };
                let Some(view) = crate::descriptor::view(table + u64::from(u.table_offset_dwords) * 4) else {
                    continue;
                };
                let View::Unordered(r, ..) = &*view else {
                    continue;
                };
                let resource = *r;
                let Some(t) = crate::resource::host_resource(resource as This) else { continue };
                // SAFETY: a live resource.
                let d = unsafe { t.GetDesc() };
                if d.Dimension != D3D12_RESOURCE_DIMENSION_TEXTURE1D && d.Dimension != D3D12_RESOURCE_DIMENSION_TEXTURE3D {
                    continue;
                }
                const COPY_SOURCE: u32 = 0x800;
                const UNORDERED: u32 = 0x8;
                // All UAV writes done before the copy.
                // SAFETY: an open list.
                unsafe { list.list.ResourceBarrier(&[crate::host::uav_barrier()]) };
                crate::list::ensure_state(list, resource as This, COPY_SOURCE);
                let read = crate::capture::read_back(&host.device, &list.list, &t);
                crate::list::ensure_state(list, resource as This, UNORDERED);
                let Some((buffer, f)) = read else { continue };
                list.keep(&buffer);
                struct Readback(ID3D12Resource);
                // SAFETY: D3D12 objects are free-threaded.
                unsafe impl Send for Readback {}
                // SAFETY: as above.
                unsafe impl Sync for Readback {}
                let buffer = Readback(buffer);
                let who = format!("{} s{}+{} {resource:#x} {}x{}x{} f{}", stage.shader.hash, u.slot, u.table_offset_dwords, d.Width, d.Height, d.DepthOrArraySize, d.Format.0);
                list.defer_after(move || {
                    let mut data = std::ptr::null_mut();
                    // SAFETY: the list has run; the mapping spans the first slice.
                    unsafe {
                        if buffer.0.Map(0, None, Some(&mut data)).is_err() {
                            return;
                        }
                        let n = (f.Footprint.RowPitch * f.Footprint.Height) as usize / 4;
                        let w = std::slice::from_raw_parts(data.cast::<u32>(), n);
                        let set = w.iter().filter(|&&x| x != 0).count();
                        let first: Vec<f32> = w.iter().take(8).map(|&x| f32::from_bits(x)).collect();
                        tracing::info!(target: "victory_gpu::trace", "written {who} right after the dispatch: {set} of {n} dwords set, first {first:?}");
                        buffer.0.Unmap(0, None);
                    }
                });
            }
        }
    }

    /// What a draw's shaders read, one line per used resource (diagnostics
    /// builds).
    #[cfg(feature = "diagnostics")]
    fn reads(&self, compute: bool) -> Vec<String> {
        let Some(p) = crate::pipeline::host_pipeline(self.pipeline as This) else {
            return vec!["no host pipeline".into()];
        };
        let mut out = Vec::new();
        for stage in &p.stages {
            let t = stage.translated();
            for u in &t.used {
                let what = match (u.root, self.roots_ref(compute).get(usize::from(u.slot))) {
                    (RootType::Table, Some(Root::Table(table))) => {
                        let handle = table + u64::from(u.table_offset_dwords) * 4;
                        let view = crate::descriptor::view(handle);
                        let mut text = describe(view.as_deref());
                        // The title's view words, and whether the host slot
                        // the shader reads was made from a view.
                        if let Some(View::Shader(_, Some(w)) | View::Unordered(_, _, Some(w))) = view.as_deref() {
                            text.push_str(&format!(" words {w:?}"));
                        }
                        let slot = (self.heaps[0] != 0)
                            .then(|| crate::descriptor::heap(self.heaps[0] as This))
                            .and_then(|h| {
                                let i = handle.checked_sub(h.start)? / 32;
                                h.host()?.describe_slot(u32::try_from(i).ok()?)
                            });
                        let current: String = view.as_deref().map(|v| format!("{v:?}")).unwrap_or_default().chars().take(120).collect();
                        text.push_str(match slot.as_deref() {
                            Some("null") => " slot null",
                            Some(s) if s.starts_with(&current) => " slot made",
                            Some(_) => " slot stale",
                            None => " slot unknown",
                        });
                        text
                    }
                    (root, _) => format!("{root:?}"),
                };
                out.push(format!(
                    "{} {:?} s{}+{}: {what}",
                    stage.shader.hash, u.kind, u.slot, u.table_offset_dwords
                ));
            }
        }
        out
    }

    pub(crate) fn draw(
        &mut self,
        list: &HostList,
        vertices: u32,
        instances: u32,
        first: u32,
        first_instance: u32,
    ) {
        #[cfg(feature = "diagnostics")]
        self.trace("draw", vertices, first);
        // Rectangles: completed by the pipeline's geometry shader, else
        // expanded into vertex buffers of their own.
        let rects = self.topology == RECTANGLES && !self.completes_rectangles();
        let ran = self.prepare(list, false, false)
            && (!rects || self.rectangles(list, vertices, first) || {
                not_run("rectangle vertices outside their buffer");
                false
            });
        if ran {
            // Rectangles were expanded into their own vertex buffers.
            let (count, from) = if rects {
                (vertices / 3 * 6, 0)
            } else {
                (vertices, first)
            };
            #[cfg(feature = "diagnostics")]
            list.note_op(|| self.op_text(format!("draw {count}x{instances}"), false));
            #[cfg(feature = "diagnostics")]
            self.probe_small(list, false);
            // SAFETY: an open list with everything set.
            unsafe {
                list.list
                    .DrawInstanced(count, instances, from, first_instance)
            };
            self.note_hdr_draw();
            DRAWN.fetch_add(1, Ordering::Relaxed);
        }
        if !ran {
            DROPPED.fetch_add(1, Ordering::Relaxed);
            #[cfg(feature = "diagnostics")]
            dropped(list, self.pipeline, "draw");
        }
    }

    /// Whether the bound pipeline's geometry shader completes rectangles.
    fn completes_rectangles(&self) -> bool {
        crate::pipeline::host_pipeline(self.pipeline as This).is_some_and(|p| p.rectangles)
    }

    /// Binds vertex buffers holding the rectangles of vertices `first..first
    /// + count` as triangles, made when the list is submitted.
    fn rectangles(&self, list: &HostList, count: u32, first: u32) -> bool {
        let Some(p) = crate::pipeline::host_pipeline_for(self.pipeline as This, &self.target_formats)
        else {
            return false;
        };
        let rects = u64::from(count / 3);
        for slot in 0..32 {
            let elements: Vec<(u32, u32)> = p
                .inputs
                .iter()
                .filter(|e| e.slot == slot && !e.per_instance)
                .map(|e| (e.offset, e.format))
                .collect();
            if elements.is_empty() {
                continue;
            }
            let v = self.vertex[slot as usize];
            let stride = u64::from(v.stride);
            let from = v.address + u64::from(first) * stride;
            if v.address == 0
                || stride == 0
                || u64::from(first + count) * stride > u64::from(v.size)
                || crate::shadow::find(from, rects * 3 * stride).is_none()
            {
                once!(slot, "rectangle vertices outside their buffer");
                return false;
            }
            let Some((gpu, to)) = list.scratch(rects * 6 * stride) else {
                return false;
            };
            let (from, to, stride) = (from as usize, to as usize, stride as usize);
            let n = rects as usize;
            list.defer_host(move || {
                // SAFETY: the vertices lie in a title buffer (checked when
                // recorded) and the scratch memory is the list's.
                let (src, dst) = unsafe {
                    (
                        std::slice::from_raw_parts(from as *const u8, n * 3 * stride),
                        std::slice::from_raw_parts_mut(to as *mut u8, n * 6 * stride),
                    )
                };
                crate::primitive::expand_rects(src, dst, stride, &elements);
            });
            let view = D3D12_VERTEX_BUFFER_VIEW {
                BufferLocation: gpu,
                SizeInBytes: (rects * 6 * stride as u64) as u32,
                StrideInBytes: v.stride,
            };
            // SAFETY: an open list; the view names live scratch memory.
            unsafe { list.list.IASetVertexBuffers(slot, Some(&[view])) };
        }
        true
    }

    pub(crate) fn draw_indexed(
        &mut self,
        list: &HostList,
        indices: u32,
        instances: u32,
        first: u32,
        base: i32,
        first_instance: u32,
    ) {
        #[cfg(feature = "diagnostics")]
        self.trace("indexed", indices, first);
        let rects = self.topology == RECTANGLES && !self.completes_rectangles();
        if rects {
            once!("indexed rectangles not drawn");
            not_run("indexed rectangles");
        }
        let ran = !rects && self.prepare(list, true, false);
        if ran {
            #[cfg(feature = "diagnostics")]
            list.note_op(|| self.op_text(format!("indexed {indices}x{instances}"), false));
            // SAFETY: an open list with everything set.
            unsafe {
                list.list
                    .DrawIndexedInstanced(indices, instances, first, base, first_instance)
            };
            self.note_hdr_draw();
            DRAWN.fetch_add(1, Ordering::Relaxed);
        }
        if !ran {
            DROPPED.fetch_add(1, Ordering::Relaxed);
            #[cfg(feature = "diagnostics")]
            dropped(list, self.pipeline, "indexed");
        }
    }
}

/// What [`State::execute_indirect`] did.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum Indirect {
    Ran,
    /// Not run on the host (dropped, logged where it was).
    NotRun,
    /// Nothing recorded: go on in a new host list and call again.
    CutFirst,
}

impl State {
    /// Indirect execution of a one-argument signature of `kind` (PC
    /// numbering: 0 draw, 1 indexed draw, 2 dispatch, 10 mesh dispatch) with
    /// records `stride` bytes apart at title address `arguments`, and the
    /// record count at `count` if the title gives one: the GPU reads them
    /// from the buffer's host copy, where earlier dispatches may have
    /// written them. False if it cannot run on the host.
    ///
    /// `may_cut`: if the clamp cannot be hoisted into `list`'s prelude
    /// (a command before it may have written the arguments) and a new host
    /// list would let it be, nothing is recorded and [`Indirect::CutFirst`]
    /// asks the caller to go on in one (`crate::list::cut_for_clamp`) and
    /// call again without `may_cut`.
    pub(crate) fn execute_indirect(
        &mut self,
        list: &HostList,
        (kind, stride, max): (u32, u32, u32),
        arguments: u64,
        count: Option<u64>,
        may_cut: bool,
    ) -> Indirect {
        match self.execute_indirect_in(list, (kind, stride, max), arguments, count, may_cut) {
            Some(true) => Indirect::Ran,
            Some(false) => Indirect::NotRun,
            None => Indirect::CutFirst,
        }
    }

    /// [`State::execute_indirect`]: whether it ran, `None` for
    /// [`Indirect::CutFirst`].
    fn execute_indirect_in(
        &mut self,
        list: &HostList,
        (kind, stride, max): (u32, u32, u32),
        arguments: u64,
        count: Option<u64>,
        may_cut: bool,
    ) -> Option<bool> {
        // Skip before the argument clamp can mutate the indirect buffer.
        #[cfg(feature = "diagnostics")]
        if crate::pipeline::is_skipped(self.pipeline as This) {
            crate::pipeline::note_drop(self.pipeline as This);
            list.note_op(|| format!("DROPPED indirect kind {kind} (skipped) {}", crate::pipeline::shaders_of(self.pipeline as This)));
            return Some(false);
        }
        const DRAW: u32 = 0;
        const INDEXED: u32 = 1;
        const DISPATCH: u32 = 2;
        const MESH: u32 = 10;
        #[cfg(feature = "diagnostics")]
        let pipeline = self.pipeline as This;
        #[cfg(feature = "diagnostics")]
        let why = |what: &str| {
            crate::pipeline::note_why(format!("indirect kind {kind}: {what}"));
            list.note_op(|| format!("DROPPED indirect kind {kind} max {max} ({what}) {}", crate::pipeline::shaders_of(pipeline)));
        };
        #[cfg(not(feature = "diagnostics"))]
        let why = |_: &str| {};
        // A call with no host pipeline cannot run, and that is the common
        // case while the world builds its pipelines: #400 measures 1 400-1 750
        // indirect calls a frame dropped this way. Drop it here, before the
        // argument refresh and the clamp dispatch, which would otherwise copy
        // the arguments and dispatch a compute shader for records nothing
        // reads. Same order as the `is_skipped` check above, for the same
        // reason.
        if crate::pipeline::held_back(self.pipeline as This) {
            why("the BVH builder held back");
            return Some(false);
        }
        // A ray pass with no structure to work on (`VICTORY_RAY_SKIP`,
        // `crate::ray_skip`): left out before its arguments are refreshed
        // and clamped, and counted there, not as a dropped execution.
        if kind == DISPATCH && crate::pipeline::ray_pass_left_out(self.pipeline as This) {
            #[cfg(feature = "diagnostics")]
            list.note_op(|| {
                format!(
                    "LEFT OUT indirect kind {kind} max {max} (VICTORY_RAY_SKIP) {}",
                    crate::pipeline::shaders_of(pipeline)
                )
            });
            return Some(true);
        }
        if crate::pipeline::host_pipeline(self.pipeline as This).is_none() {
            #[cfg(feature = "diagnostics")]
            crate::pipeline::note_drop(self.pipeline as This);
            why("no host pipeline");
            return Some(false);
        }
        let len = u64::from(stride) * u64::from(max.max(1));
        let Some(args) = crate::shadow::find(arguments, len) else {
            why("arguments outside title buffers");
            return Some(false);
        };
        let counter = match count {
            Some(at) => match crate::shadow::find(at, 4) {
                Some(s) => Some((s, at)),
                None => {
                    why("count outside title buffers");
                    return Some(false);
                }
            },
            None => None,
        };
        let Some(signature) = matches!(kind, DRAW | INDEXED | DISPATCH | MESH)
            .then(|| host_signature(kind, stride))
            .flatten()
        else {
            why("no host signature");
            return Some(false);
        };
        let args_gpu = args.gpu(arguments);
        if !crate::indirect::available() || !args_gpu.is_multiple_of(4) || !stride.is_multiple_of(4)
        {
            why("not bounded (no clamp, or arguments not 4-byte aligned)");
            return Some(false);
        }
        // Decide hoisting before `prepare`/`bind`, which record *this* call's
        // own GPU writes (those only constrain subsequent indirect calls),
        // and before anything of this call is recorded, so a cut leaves
        // nothing of it behind.
        let blocker = if crate::indirect::batch_clamps_enabled() {
            list.clamp_blocker(arguments, len, count)
        } else {
            None
        };
        let hoist = crate::indirect::batch_clamps_enabled() && blocker.is_none();
        if !hoist
            && may_cut
            && blocker.is_some()
            && crate::indirect::clamp_cuts_enabled()
            && list.clamp_cut_pays()
        {
            return None;
        }
        args.refresh(list, arguments, len);
        if let Some((s, at)) = &counter {
            s.refresh(list, *at, 4);
        }
        let counted = counter.as_ref().map(|(s, at)| s.gpu(*at));
        let inline_clamped = if hoist {
            None
        } else {
            // Bounded first: the inline clamp binds a compute pipeline of its
            // own. It reads the count, so that it stops where execution does.
            let Some((clamped, dst_offset)) =
                crate::indirect::bound(list, args_gpu, (kind, stride, max), counted)
            else {
                why("not bounded (no clamp, or arguments not 4-byte aligned)");
                return Some(false);
            };
            list.note_inline_clamp();
            crate::indirect::note_inline(blocker);
            self.host_pipeline = 0;
            self.host_roots[1] = 0;
            Some((clamped, dst_offset))
        };
        let ran = match kind {
            DRAW | INDEXED => {
                (self.topology != RECTANGLES || self.completes_rectangles() || {
                    not_run("rectangles");
                    false
                }) && self.prepare(list, kind == INDEXED, false)
            }
            MESH => self.prepare(list, false, true),
            DISPATCH => match crate::pipeline::host_pipeline(self.pipeline as This) {
                Some(p) if p.compute => self.bind(list, p, true),
                _ => false,
            },
            _ => unreachable!("the signature check above admits only these kinds"),
        };
        if !ran {
            if let Some((clamped, _)) = &inline_clamped {
                crate::indirect::done(list, clamped);
            }
            #[cfg(feature = "diagnostics")]
            match crate::pipeline::host_pipeline(self.pipeline as This) {
                None => {
                    crate::pipeline::note_drop(self.pipeline as This);
                    list.note_op(|| format!("DROPPED indirect kind {kind} (no host pipeline) {}", crate::pipeline::shaders_of(pipeline)));
                }
                Some(p) => why(&format!(
                    "prepare failed (compute {}, mesh {}): {}",
                    p.compute,
                    p.mesh,
                    NOT_RUN.with(|w| w.replace(""))
                )),
            }
            return Some(false);
        }
        let (clamped, dst_offset, needs_done) = match inline_clamped {
            Some((clamped, dst_offset)) => (clamped, dst_offset, true),
            None => {
                let Some((clamped, dst_offset)) =
                    crate::indirect::enqueue(list, args_gpu, (kind, stride, max), counted)
                else {
                    why("not bounded (no hoisted argument buffer)");
                    return Some(false);
                };
                (clamped, dst_offset, false)
            }
        };
        #[cfg(feature = "diagnostics")]
        list.note_op(|| {
            self.op_text(
                format!("indirect kind {kind} max {max} at {arguments:#x}"),
                kind == DISPATCH,
            )
        });
        if kind == DISPATCH
            && crate::census::on()
            && let Some(p) = crate::pipeline::host_pipeline(self.pipeline as This)
        {
            crate::census::indirect(&p.name, max);
        }
        // SAFETY: an open list with everything set; the buffers are host
        // copies the list keeps alive (refresh), the signature is cached.
        unsafe {
            list.list.ExecuteIndirect(
                &signature,
                max,
                &clamped,
                dst_offset,
                counter.as_ref().map(|(s, _)| &s.buffer),
                counter.as_ref().map_or(0, |(s, at)| s.offset(*at)),
            )
        };
        // A listed frame, or any frame for a traced shader: what the records
        // held (after the clamp).
        #[cfg(feature = "diagnostics")]
        {
            let always = !crate::host::traced_shaders().is_empty()
                && crate::host::is_traced(&crate::pipeline::shaders_of(pipeline));
            if always || crate::capture::dump_on() {
                let count = counter.as_ref().map(|(s, at)| (&s.buffer, s.offset(*at)));
                let name = crate::pipeline::shaders_of(pipeline);
                crate::capture::read_indirect(
                    list,
                    (&clamped, dst_offset),
                    count,
                    (kind, stride, max),
                    name,
                    always,
                );
            }
        }
        if needs_done {
            crate::indirect::done(list, &clamped);
        }
        if kind == DISPATCH {
            self.probe_hdr_compute(list);
            DISPATCHED.fetch_add(1, Ordering::Relaxed);
            self.ambient(list);
        } else {
            self.note_hdr_draw();
            DRAWN.fetch_add(1, Ordering::Relaxed);
        }
        Some(true)
    }
}

/// The host command signature for one argument of `kind` with records
/// `stride` bytes apart (no root arguments, so no root signature), made
/// once.
fn host_signature(kind: u32, stride: u32) -> Option<ID3D12CommandSignature> {
    static MADE: std::sync::Mutex<Vec<((u32, u32), ID3D12CommandSignature)>> =
        std::sync::Mutex::new(Vec::new());
    let mut made = MADE.lock().unwrap_or_else(|e| e.into_inner());
    if let Some((_, s)) = made.iter().find(|(k, _)| *k == (kind, stride)) {
        return Some(s.clone());
    }
    let argument = D3D12_INDIRECT_ARGUMENT_DESC {
        Type: D3D12_INDIRECT_ARGUMENT_TYPE(kind as i32),
        ..Default::default()
    };
    let desc = D3D12_COMMAND_SIGNATURE_DESC {
        ByteStride: stride,
        NumArgumentDescs: 1,
        pArgumentDescs: &argument,
        NodeMask: 0,
    };
    let mut signature: Option<ID3D12CommandSignature> = None;
    // SAFETY: a description of one argument; no root arguments.
    let result = unsafe {
        crate::host::host()?.device.CreateCommandSignature(
            &desc,
            None::<&ID3D12RootSignature>,
            &mut signature,
        )
    };
    if let Err(e) = result {
        once!(error = %e, kind, stride, "host command signature refused");
        return None;
    }
    let signature = signature?;
    made.push(((kind, stride), signature.clone()));
    Some(signature)
}

/// Whether table UAV buffer writebacks are skipped for descriptors a shader
/// never stores to (`VICTORY_TABLE_UAV_WRITE_SKIP=0` disables; on by default).
fn table_uav_write_skip() -> bool {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| match std::env::var_os("VICTORY_TABLE_UAV_WRITE_SKIP") {
        Some(v) => crate::switch::value_is_on(Some(&v)),
        None => true,
    })
}

/// Gets what a shader reads or writes through a view ready: buffer data
/// copied, textures moved to the state they need. `marked`: a shader's
/// writes through the view mark their pages (every translated shader's do);
/// the host's own (clears) do not.
fn readable(list: &HostList, u: &UsedResource, view: &View, marked: bool) {
    readable_entry(list, u, view, marked, true);
}

fn readable_entry(list: &HostList, u: &UsedResource, view: &View, marked: bool, writes: bool) {
    const SHADER_RESOURCE: u32 = 0x40 | 0x80;
    const UNORDERED_ACCESS: u32 = 0x8;
    let placed = crate::descriptor::host::placed_texture(view);
    let view = placed.as_ref().unwrap_or(view);
    // Buffer views read title memory: their range is copied at submission;
    // what shaders write through unordered access goes back to it after.
    // Constant buffers (raw views) to their copy's end at most; others in
    // the copy their host view reads.
    let range = match *view {
        View::Constant(address, size) => crate::shadow::find(address, 1).map(|s| {
            let len = u64::from(size).min(s.left(address));
            (s, address, len)
        }),
        _ => crate::descriptor::host::view_copy(view),
    };
    if let Some((shadow, address, len)) = range
        && len > 0
    {
        let uav_writes = u.kind == ResourceType::Uav && writes;
        let (refresh_at, refresh_len) = if uav_writes && marked && shadow.marked() {
            shadow.page_range(address, len)
        } else {
            (address, len)
        };
        shadow.refresh(list, refresh_at, refresh_len);
        if uav_writes {
            shadow.gpu_writes(list, address, len, marked);
        } else if u.kind == ResourceType::Srv && crate::descriptor::host::three_channel_srv(view) {
            for widened in
                crate::shadow::widened_in(address as usize, (address + len) as usize)
            {
                list.mark_widened_read(&widened.buffer);
            }
        }
    }
    match (view, u.kind) {
        (View::Shader(resource, _), ResourceType::Srv) => {
            if *resource != 0 {
                let tested = TESTED_DEPTH.with(std::cell::Cell::get);
                if tested.0 == *resource {
                    TESTED_DEPTH.with(|d| d.set((*resource, true)));
                }
            }
            // A view of memory the GPU writes through another title texture
            // whose host copy this one could not share reads that copy.
            let copied = crate::resource::alias_source(*resource as This)
                .is_some_and(|a| crate::texture::alias_copy(list, a.get(), *resource as This));
            if !copied
                && let Some(t) = crate::resource::host_copy(*resource as This)
                && t.title_written()
            {
                crate::texture::upload(list, *resource as This, &t.texture);
            }
            crate::list::ensure_state(list, *resource as This, SHADER_RESOURCE);
            #[cfg(feature = "diagnostics")]
            if let Some(t) = crate::resource::host_resource(*resource as This) {
                // SAFETY: a live resource.
                let d = unsafe { t.GetDesc() };
                if d.Flags.0 & 7 == 0 {
                    tracing::debug!(
                        resource = format_args!("{resource:#x}"),
                        ?d,
                        "a shader reads a texture the title fills"
                    );
                }
            }
        }
        (View::Unordered(resource, _, _), ResourceType::Uav) => {
            #[cfg(feature = "diagnostics")]
            crate::capture::note_target(list, *resource as This);
            list.texture_written(*resource as This);
            if writes {
                crate::list::ensure_state(list, *resource as This, UNORDERED_ACCESS);
            } else {
                // No shader stores through it (`crate::twin`: a depth
                // texture's twin stays as current as it was).
                crate::twin::reading(|| crate::list::ensure_state(list, *resource as This, UNORDERED_ACCESS));
            }
        }
        _ => {}
    }
}

/// A view in a few words (diagnostics builds).
#[cfg(feature = "diagnostics")]
fn describe(view: Option<&View>) -> String {
    let texture = |r: This| {
        crate::resource::host_resource(r).map(|t| {
            // SAFETY: a live resource.
            let d = unsafe { t.GetDesc() };
            // Its console memory: another view of it (a placed one) is
            // another host texture.
            let at = crate::resource::texture_memory(r).map_or(0, |(p, _)| p as u64);
            format!("{}x{} f{} flags {:#x} mem {at:#x}", d.Width, d.Height, d.Format.0, d.Flags.0)
        })
    };
    // Small buffers (exposure, counters): their title memory now, as floats.
    let small = |range: Option<(u64, u64)>| match range {
        Some((at, len)) if len <= 64 && crate::texture::committed(at as *const u8, len as usize) => {
            // SAFETY: `len` committed bytes of title memory.
            let b = unsafe { std::slice::from_raw_parts(at as *const u8, len as usize) };
            let f: Vec<f32> = b.as_chunks::<4>().0.iter().map(|&c| f32::from_le_bytes(c)).collect();
            format!(" = {f:?}")
        }
        _ => String::new(),
    };
    match view {
        None => "no view".into(),
        Some(v @ View::Shader(r, _)) => texture(*r as This).unwrap_or_else(|| {
            let range = crate::descriptor::host::buffer_range(v);
            format!("buffer {range:?}{}", small(range))
        }),
        // A buffer has no host texture: its title range, as for a shader view.
        Some(v @ View::Unordered(r, ..)) => texture(*r as This).map_or_else(
            || {
                let range = crate::descriptor::host::buffer_range(v);
                format!("uav buffer {range:?}{}", small(range))
            },
            |t| format!("uav {t}"),
        ),
        Some(View::Placed(u, r, d)) => format!(
            "placed uav={u} dim {} {}x{} f{} layout {:#x} at {:#x}",
            r[0],
            u64::from(r[4]) | u64::from(r[5]) << 32,
            r[6],
            r[8],
            r[11],
            u64::from(d[10]) | u64::from(d[11]) << 32
        ),
        // The first four values, as a colour rectangle's constants are, read
        // through the kernel (asking `VirtualQuery` first cost milliseconds
        // per draw in the title's pools, with every draw noted).
        Some(View::Constant(address, size)) if *size >= 16 => {
            let mut bytes = [0u8; 16];
            if crate::host::read_into(*address as *const u8, &mut bytes) {
                let v: [f32; 4] = std::array::from_fn(|i| f32::from_le_bytes([0, 1, 2, 3].map(|k| bytes[4 * i + k])));
                format!("Constant({address:#x}, {size}) {v:?}")
            } else {
                format!("Constant({address:#x}, {size})")
            }
        }
        Some(v) => format!("{v:?}").chars().take(60).collect(),
    }
}

/// A user-data entry that takes the heap index of a root view's view.
fn is_root_view(u: &victory_shader::emit::UserData) -> bool {
    u.root == RootType::Descriptor && u.dword >= victory_shader::emit::ROOT_SRV_VIEW
}

/// Whether `u` is a root view's offset in the copy its view shows: a root
/// descriptor's dword 4. A root constant's dword 4 is a value.
fn is_root_view_offset(u: &victory_shader::emit::UserData) -> bool {
    u.root == RootType::Descriptor && u.dword == victory_shader::emit::ROOT_VIEW_OFFSET
}

/// Whether a shader reads root views through views the host makes.
fn has_root_views(t: &victory_shader::emit::HostShader) -> bool {
    t.user_data.iter().any(is_root_view)
}

/// `DispatchX`'s flag that the counts are threads, not groups (the console's
/// dispatch initiator bit "use thread dimensions"; DexNom/VIctory bc3cd4b0).
/// Taken for groups, a stage of the ray-tracing structure builder ran up to
/// 64 times its threads and stored over the structures after its own.
const THREAD_DIMENSIONS: u32 = 0x20;

/// `VICTORY_THREAD_DIMENSIONS=0`: such counts are taken for groups, as before.
fn thread_dimensions() -> bool {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| std::env::var_os("VICTORY_THREAD_DIMENSIONS").is_none_or(|v| v != "0"))
}

/// The groups of `size` threads that hold `threads`.
fn dispatch_groups(threads: [u32; 3], size: [u32; 3]) -> [u32; 3] {
    std::array::from_fn(|i| threads[i].div_ceil(size[i]))
}

/// Where a compute pipeline's user data has a word of `kind` the host
/// computes (`RootType::Other`): the root parameter and its dword there.
fn compute_word(p: &HostPipeline, kind: u32) -> Option<(u32, u32)> {
    let stage = p.stages.first().filter(|_| p.compute)?;
    let (param, _) = stage.user_data?;
    let host = &stage.translated().host;
    let word = host
        .user_data
        .iter()
        .find(|u| u.root == RootType::Other(kind))?;
    Some((param, word.index))
}

/// Logs each pipeline's first dispatch counted in threads: what it asked
/// for, its group, whether the last group is a part of one (`partial`) and
/// whether the form whose lanes past the count leave runs it (`limited`);
/// a part of a group without it runs whole (a warning).
fn note_counted_in_threads(
    name: &str,
    threads: [u32; 3],
    size: [u32; 3],
    partial: bool,
    limited: bool,
) {
    static SEEN: std::sync::Mutex<std::collections::BTreeSet<String>> =
        std::sync::Mutex::new(std::collections::BTreeSet::new());
    THREAD_COUNTED.fetch_add(1, Ordering::Relaxed);
    if !SEEN
        .lock()
        .unwrap_or_else(|e| e.into_inner())
        .insert(name.to_owned())
    {
        return;
    }
    if partial && !limited {
        tracing::warn!(
            pipeline = name,
            ?threads,
            group = ?size,
            "a dispatch counted in threads whose program cannot leave the lanes past the count: its last group runs whole"
        );
    } else {
        tracing::info!(pipeline = name, ?threads, group = ?size, partial, limited, "a dispatch counted in threads");
    }
}

/// Dispatches counted in threads (`DispatchX`) so far.
static THREAD_COUNTED: AtomicU64 = AtomicU64::new(0);

/// Dispatches counted in threads so far.
#[cfg(test)]
pub(crate) fn thread_counted() -> u64 {
    THREAD_COUNTED.load(Ordering::Relaxed)
}

/// The console's own primitive topologies.
const TRIANGLE_FAN: u32 = 6;
const RECTANGLES: u32 = 17;
const LINE_LOOP: u32 = 18;
const QUADS: u32 = 19;
const QUAD_STRIP: u32 = 20;

#[cfg(test)]
mod tests {
    mod flight;
    mod tracked_rows;

    use super::*;
    use crate::backend::{HostGpu, HostMemory};

    /// A root signature that indexes the heaps directly needs one of each
    /// kind bound before it is set: our sampler heap stands in for a title
    /// that binds none, which a compute list that only reads views does.
    #[test]
    fn a_list_without_a_sampler_heap_gets_ours() {
        // The title binds both: nothing of ours.
        assert_eq!(State::wanted([0x10, 0x20], false), [0x10, 0x20]);
        // Views only: our sampler heap after them.
        assert_eq!(State::wanted([0x10, 0], false), [0x10, SAMPLERS_OURS]);
        // Neither, with title memory's views needed: ours of both kinds.
        assert_eq!(
            State::wanted([0, 0], true),
            [MEMORY_HEAP, SAMPLERS_OURS]
        );
        // A title sampler heap stays whatever the views are.
        assert_eq!(State::wanted([0, 0x20], true), [MEMORY_HEAP, 0x20]);
    }
    use crate::object::{HRESULT, S_OK};
    use crate::compute::{Param, srv, uav};
    use windows::core::s;

    /// A root constant's dword 4 is a value: only a root descriptor's dword
    /// 4 names its view's offset.
    #[test]
    fn only_a_root_descriptors_dword_four_is_a_view_offset() {
        let u = |root, dword| victory_shader::emit::UserData {
            index: 0,
            slot: 0,
            root,
            dword,
        };
        let offset = victory_shader::emit::ROOT_VIEW_OFFSET;
        assert!(is_root_view_offset(&u(RootType::Descriptor, offset)));
        assert!(!is_root_view_offset(&u(RootType::Constants, offset)));
        assert!(!is_root_view_offset(&u(RootType::Table, offset)));
    }

    /// Reads 16 dwords 64 bytes apart from `offset` past the root view
    /// (heap index in the table) as the translated shaders do: the root
    /// address's offset in the view plus their own.
    const SOURCE: &str = r#"
ByteAddressBuffer src : register(t0);
RWByteAddressBuffer dst : register(u0);
cbuffer c : register(b0) { uint root; uint offset; }
[numthreads(16, 1, 1)]
void main(uint3 id : SV_DispatchThreadID) {
    dst.Store(id.x * 4, src.Load(root + offset + id.x * 64));
}
"#;

    const MARKED_WRITE_SOURCE: &str = r#"
RWByteAddressBuffer dst : register(u0);
ByteAddressBuffer marks : register(t0);
RWByteAddressBuffer writes : register(u1);
cbuffer c : register(b0) { uint slot; uint byte_offset; uint value; }
[numthreads(1, 1, 1)]
void main() {
    dst.Store(byte_offset, value);
    uint2 m = marks.Load2(slot * 8);
    uint bit = m.x + ((m.y + byte_offset) >> 16);
    uint old;
    writes.InterlockedOr((bit >> 5) * 4, 1u << (bit & 31), old);
}
"#;

    /// The marked write shader's root signature: a table of one unordered
    /// view, the marks, the write bitmap and three constants.
    const MARKED_WRITES: &[Param] = &[
        Param::Table(&[uav(0, 1)]),
        Param::Srv { register: 0, space: 0 },
        Param::Uav { register: 1 },
        Param::Constants { register: 0, count: 3 },
    ];

    /// Whether the host's HLSL compiler builds the marked write shader. Wine's
    /// (vkd3d-shader) fails on its interlocked write to the bitmap, so these
    /// tests skip there.
    fn marked_writes_build() -> bool {
        match crate::compute::pipeline(MARKED_WRITE_SOURCE, s!("root"), 0, MARKED_WRITES) {
            Ok(_) => true,
            Err(e) if crate::under_wine() => {
                eprintln!("skipped: Wine's HLSL compiler cannot build the marked write shader ({e})");
                false
            }
            Err(e) => panic!("the marked write shader does not build: {e}"),
        }
    }

    fn buffer_at_address(address: u64, bytes: u64) -> This {
        let desc: [u32; 14] = [
            1,
            0,
            0,
            0,
            bytes as u32,
            (bytes >> 32) as u32,
            1,
            1 | 1 << 16,
            0,
            1,
            0,
            1,
            0,
            0,
        ];
        let mut out = std::ptr::null();
        let iid = crate::object::guid(
            0x696442be,
            0xa72e,
            0x4059,
            [0xbc, 0x79, 0x5b, 0x5c, 0x98, 0x04, 0x0f, 0xad],
        );
        let made = crate::resource::create_at_address(
            std::ptr::null(),
            address,
            desc.as_ptr().cast(),
            0,
            std::ptr::null(),
            &iid,
            &mut out,
        );
        assert_eq!(made, S_OK);
        out
    }

    fn view_heap(
        count: u32,
    ) -> (
        This,
        &'static crate::descriptor::Heap,
        &'static crate::descriptor::host::HostHeap,
    ) {
        let null = std::ptr::null();
        let heap_desc: [u32; 8] = [0, count, 1, 0, 0, 0, 0, 0];
        let mut heap = null;
        let iid = crate::object::guid(
            0x8efb471d,
            0x616c,
            0x4f49,
            [0x90, 0xf7, 0x12, 0x7b, 0xb7, 0x63, 0xfa, 0x51],
        );
        assert_eq!(
            crate::descriptor::create_heap(null, heap_desc.as_ptr().cast(), &iid, &mut heap),
            S_OK
        );
        let title_heap = crate::descriptor::heap(heap);
        let heap_host = title_heap.host().expect("a host view heap");
        (heap, title_heap, heap_host)
    }

    fn dispatch_marked_uav(
        list: &HostList,
        heap_host: &crate::descriptor::host::HostHeap,
        (root, pso): &(ID3D12RootSignature, ID3D12PipelineState),
        table_slot: u32,
        mark_slot: u32,
        byte_offset: u32,
        value: u32,
    ) {
        let marks = heap_host.marks().expect("marks table");
        let writes = crate::shadow::view_writes(false).expect("view_writes bitmap");
        // SAFETY: live resource.
        let marks_gpu = unsafe { marks.GetGPUVirtualAddress() };
        list.keep(marks);
        let constants = [mark_slot, byte_offset, value];
        // SAFETY: an open list and live objects.
        unsafe {
            list.list
                .SetDescriptorHeaps(&[Some(heap_host.heap.clone())]);
            list.list.SetComputeRootSignature(root);
            list.list
                .SetComputeRootDescriptorTable(0, heap_host.gpu(table_slot));
            list.list.SetComputeRootShaderResourceView(1, marks_gpu);
            list.list.SetComputeRootUnorderedAccessView(2, writes);
            list.list
                .SetComputeRoot32BitConstants(3, 3, constants.as_ptr().cast(), 0);
            list.list.SetPipelineState(pso);
            list.list.Dispatch(1, 1, 1);
        }
    }

    /// Sub-64 KiB marked UAV writes must expand `Shadow::refresh` to the
    /// enclosing 64 KiB page so `Tracked::take` guards it before execution
    /// and `copy_back_marks` writes GPU bytes back while keeping neighbor
    /// guest bytes intact.
    #[test]
    fn marked_subpage_uav_writes_copy_back_and_keep_neighbor_bytes() {
        if !marked_writes_build() {
            return;
        }
        let Some(host) = crate::host::host() else {
            return;
        };
        let p = crate::watch::tests::pool(1);
        // SAFETY: committed write-watched page of 64 KiB.
        unsafe {
            ((p + 0x100) as *mut u32).write_volatile(0x1122_3344);
            ((p + 0x2000) as *mut u32).write_volatile(0x5566_7788);
        }
        let buf = buffer_at_address(p as u64, crate::watch::PAGE as u64);
        let (_, title_heap, heap_host) = view_heap(8);
        let _ = host.log_messages();

        let uav_desc: [u32; 10] = [39, 1, 1024, 0, 64, 0, 0, 0, 1, 0];
        let slot0 = title_heap.start;
        crate::descriptor::create_uav(std::ptr::null(), buf, std::ptr::null(), &uav_desc, slot0);
        let view = crate::descriptor::view(slot0).expect("subpage uav view");
        let _ = heap_host.mirror(0, slot0, true);

        let shader = compile_cs(MARKED_WRITE_SOURCE, MARKED_WRITES);
        let list = host.new_list().unwrap();
        readable(&list, &used(ResourceType::Uav), &view, true);
        dispatch_marked_uav(&list, heap_host, &shader, 0, 0, 0, 0xDEAD_BEEF);
        list.close();
        host.execute(&[&list]);

        assert_eq!(host.log_messages(), None);
        // SAFETY: bytes inside our page.
        unsafe {
            assert_eq!(((p + 4096) as *const u32).read_volatile(), 0xDEAD_BEEF);
            assert_eq!(((p + 0x100) as *const u32).read_volatile(), 0x1122_3344);
            assert_eq!(((p + 0x2000) as *const u32).read_volatile(), 0x5566_7788);
        }
        crate::shadow::unregister(buf as usize, p as u64);
    }

    /// A title copy to a disjoint offset on the same 64 KiB page while a
    /// sub-64 KiB marked UAV is pending arms `Tracked::saved`, and
    /// `copy_back_marks` 3-way merges GPU, CPU, and pre-existing bytes.
    #[test]
    fn marked_subpage_uav_and_cpu_disjoint_write_three_way_merge() {
        if !marked_writes_build() {
            return;
        }
        let Some(host) = crate::host::host() else {
            return;
        };
        let p = crate::watch::tests::pool(1);
        // SAFETY: committed write-watched page of 64 KiB.
        unsafe {
            ((p + 0x100) as *mut u32).write_volatile(0x1122_3344);
        }
        let buf = buffer_at_address(p as u64, crate::watch::PAGE as u64);
        let (_, title_heap, heap_host) = view_heap(8);
        let _ = host.log_messages();

        let uav_desc: [u32; 10] = [39, 1, 1024, 0, 64, 0, 0, 0, 1, 0];
        let slot0 = title_heap.start;
        crate::descriptor::create_uav(std::ptr::null(), buf, std::ptr::null(), &uav_desc, slot0);
        let view = crate::descriptor::view(slot0).expect("subpage uav view");
        let _ = heap_host.mirror(0, slot0, true);

        let cpu_val = 0xCAFE_BABEu32.to_le_bytes();
        let shader = compile_cs(MARKED_WRITE_SOURCE, MARKED_WRITES);
        let list = host.new_list().unwrap();
        readable(&list, &used(ResourceType::Uav), &view, true);
        list.defer_copy((p + 0x2000) as *mut u8, cpu_val.as_ptr(), cpu_val.len());
        dispatch_marked_uav(&list, heap_host, &shader, 0, 0, 0, 0xDEAD_BEEF);
        list.close();
        host.execute(&[&list]);

        assert_eq!(host.log_messages(), None);
        // SAFETY: bytes inside our page.
        unsafe {
            assert_eq!(((p + 4096) as *const u32).read_volatile(), 0xDEAD_BEEF);
            assert_eq!(((p + 0x2000) as *const u32).read_volatile(), 0xCAFE_BABE);
            assert_eq!(((p + 0x100) as *const u32).read_volatile(), 0x1122_3344);
        }
        crate::shadow::unregister(buf as usize, p as u64);
    }

    /// Live GPU regression test covering write-watched (`MEM_WRITE_WATCH`)
    /// pages (unarmed marked UAV + declared UAV) AND a mixed tracked +
    /// untracked (`PAGE_READWRITE`) shadow: when a compute shader writes
    /// `[0..4)` and the CPU writes `[16..20)` right before copy-back, the
    /// CPU's bytes must never be clobbered and untracked plain memory must
    /// stay `PAGE_READWRITE`.
    #[test]
    fn writewatch_and_untracked_uav_copy_back_preserve_cpu_writes() {
        use windows_sys::Win32::System::Memory::{
            MEM_COMMIT, MEM_RELEASE, MEM_RESERVE, MEMORY_BASIC_INFORMATION, PAGE_READWRITE, VirtualAlloc,
            VirtualFree, VirtualQuery,
        };
        let Some(host) = crate::host::host() else {
            return;
        };
        // Single-thread marked UAV shader using SM 5.0 `RWByteAddressBuffer::Store`
        // so Wine's `vkd3d-shader` compiles and runs it live on D3DMetal.
        const LIVE_MARKED_WRITE: &str = r#"
RWByteAddressBuffer dst : register(u0);
RWByteAddressBuffer writes : register(u1);
cbuffer c : register(b0) { uint byte_offset; uint value; uint mark_word_bytes; uint mark_mask; }
[numthreads(1, 1, 1)]
void main() {
    dst.Store(byte_offset, value);
    if (mark_mask != 0) {
        writes.Store(mark_word_bytes, mark_mask);
    }
}
"#;
        const LIVE_PARAMS: &[Param] = &[
            Param::Table(&[uav(0, 1)]),
            Param::Uav { register: 1 },
            Param::Constants { register: 0, count: 4 },
        ];
        let (root, pso) = compile_cs(LIVE_MARKED_WRITE, LIVE_PARAMS);
        let dispatch = |list: &HostList,
                        heap_host: &crate::descriptor::host::HostHeap,
                        table_slot: u32,
                        byte_offset: u32,
                        value: u32,
                        mark_bit: Option<u32>| {
            let writes = crate::shadow::view_writes(false).expect("view_writes bitmap");
            let (mark_word_bytes, mark_mask) = match mark_bit {
                Some(bit) => ((bit >> 5) * 4, 1u32 << (bit & 31)),
                None => (0, 0),
            };
            let constants = [byte_offset, value, mark_word_bytes, mark_mask];
            // SAFETY: an open list and live objects.
            unsafe {
                list.list.SetDescriptorHeaps(&[Some(heap_host.heap.clone())]);
                list.list.SetComputeRootSignature(&root);
                list.list.SetComputeRootDescriptorTable(0, heap_host.gpu(table_slot));
                list.list.SetComputeRootUnorderedAccessView(1, writes);
                list.list.SetComputeRoot32BitConstants(2, 4, constants.as_ptr().cast(), 0);
                list.list.SetPipelineState(&pso);
                list.list.Dispatch(1, 1, 1);
            }
        };
        let (_, title_heap, heap_host) = view_heap(8);
        let _ = host.log_messages();

        // Case 1: Write-watched pool page + unarmed marked UAV (bindless/heap
        // path where `gpu_writes` was NOT declared): CPU write to `[16..20)`
        // before copy-back must NOT be clobbered by `copy_back_marks`.
        let p = crate::watch::tests::pool(1);
        // SAFETY: committed write-watched page of 64 KiB.
        unsafe {
            (p as *mut u32).write_volatile(0x1111_1111);
            ((p + 16) as *mut u32).write_volatile(0x2222_2222);
        }
        let buf_pool = buffer_at_address(p as u64, crate::watch::PAGE as u64);
        let shadow_pool = crate::shadow::find(p as u64, crate::watch::PAGE as u64).expect("pool shadow");
        let pool_mark = shadow_pool.mark_entry(p as u64).map(|(bit, _)| bit);
        assert!(pool_mark.is_some(), "write-watched pool shadow must be marked");
        let uav_at_0: [u32; 10] = [39, 1, 0, 0, 64, 0, 0, 0, 1, 0];
        let slot0 = title_heap.start;
        crate::descriptor::create_uav(std::ptr::null(), buf_pool, std::ptr::null(), &uav_at_0, slot0);
        let view0 = crate::descriptor::view(slot0).expect("pool uav view");
        let _ = heap_host.mirror(0, slot0, true);

        let cpu_val = 0xCAFE_BABEu32.to_le_bytes();
        let list = host.new_list().unwrap();
        // Refresh as read-only (no `gpu_writes` arming), then title writes
        // `[16..20)` before the GPU shader writes `[0..4)` and marks the page.
        readable(&list, &used(ResourceType::Srv), &view0, false);
        list.defer_copy((p + 16) as *mut u8, cpu_val.as_ptr(), cpu_val.len());
        dispatch(&list, heap_host, 0, 0, 0xDEAD_BEEF, pool_mark);
        list.close();
        host.execute(&[&list]);
        assert_eq!(host.log_messages(), None);
        // SAFETY: bytes inside our page.
        let cpu_after = unsafe { ((p + 16) as *const u32).read_volatile() };
        assert_eq!(
            cpu_after,
            0xCAFE_BABE,
            "unarmed marked copy-back must keep CPU write on write-watched page"
        );

        // Case 2: Write-watched pool page + armed marked UAV: 3-way merge
        // preserves BOTH GPU `[0..4)` and CPU `[16..20)`.
        // SAFETY: committed write-watched page of 64 KiB.
        unsafe {
            (p as *mut u32).write_volatile(0x1111_1111);
            ((p + 16) as *mut u32).write_volatile(0x2222_2222);
        }
        let list_armed = host.new_list().unwrap();
        readable(&list_armed, &used(ResourceType::Uav), &view0, true);
        list_armed.defer_copy((p + 16) as *mut u8, cpu_val.as_ptr(), cpu_val.len());
        dispatch(&list_armed, heap_host, 0, 0, 0xDEAD_BEEF, pool_mark);
        list_armed.close();
        host.execute(&[&list_armed]);
        assert_eq!(host.log_messages(), None);
        // SAFETY: bytes inside our page.
        unsafe {
            assert_eq!((p as *const u32).read_volatile(), 0xDEAD_BEEF, "armed GPU write copied back");
            assert_eq!(((p + 16) as *const u32).read_volatile(), 0xCAFE_BABE, "armed copy-back preserved CPU write");
        }

        // Case 3: Write-watched pool page + declared `gpu_writes(..., false)`
        // (snapshot 3-way merge): both GPU `[0..4)` and CPU `[16..20)` survive.
        // SAFETY: committed write-watched page of 64 KiB.
        unsafe {
            (p as *mut u32).write_volatile(0x1111_1111);
            ((p + 16) as *mut u32).write_volatile(0x2222_2222);
        }
        let list2 = host.new_list().unwrap();
        shadow_pool.refresh(&list2, p as u64, crate::watch::PAGE as u64);
        shadow_pool.gpu_writes(&list2, p as u64, crate::watch::PAGE as u64, false);
        list2.defer_copy((p + 16) as *mut u8, cpu_val.as_ptr(), cpu_val.len());
        dispatch(&list2, heap_host, 0, 0, 0xDEAD_BEEF, None);
        list2.close();
        host.execute(&[&list2]);
        assert_eq!(host.log_messages(), None);
        // SAFETY: bytes inside our page.
        unsafe {
            assert_eq!((p as *const u32).read_volatile(), 0xDEAD_BEEF, "declared GPU write copied back");
            assert_eq!(((p + 16) as *const u32).read_volatile(), 0xCAFE_BABE, "declared copy-back preserved CPU write");
        }
        crate::shadow::unregister(buf_pool as usize, p as u64);

        // Case 4: Mixed tracked (`MEM_WRITE_WATCH`) + untracked plain
        // (`PAGE_READWRITE`) shadow: UAV targets the untracked plain page,
        // GPU writes `[0..4)`, CPU writes `[16..20)` right before copy-back.
        let page = crate::watch::PAGE;
        let (pool, plain) = (0..8)
            .find_map(|_| {
                // SAFETY: fresh reservation released immediately, then two adjacent pages.
                unsafe {
                    let base = VirtualAlloc(std::ptr::null(), 2 * page, MEM_RESERVE, PAGE_READWRITE) as usize;
                    VirtualFree(base as *mut _, 0, MEM_RELEASE);
                    let at = |q: usize, kind: u32| {
                        VirtualAlloc(q as *const _, page, MEM_RESERVE | MEM_COMMIT | kind, PAGE_READWRITE) as usize
                    };
                    let (a, b) = (at(base, crate::watch::tests::MEM_WRITE_WATCH), at(base + page, 0));
                    if base != 0 && a == base && b == base + page {
                        return Some((a, b));
                    }
                    [a, b].into_iter().filter(|&q| q != 0).for_each(crate::watch::tests::free);
                    None
                }
            })
            .expect("two adjacent pages");
        // SAFETY: bytes inside our plain page.
        unsafe {
            (plain as *mut u32).write_volatile(0x1111_1111);
            ((plain + 16) as *mut u32).write_volatile(0x2222_2222);
        }
        let buf_mixed = buffer_at_address(pool as u64, (2 * page) as u64);
        let shadow_mixed = crate::shadow::find(pool as u64, (2 * page) as u64).expect("mixed shadow");
        let plain_mark = shadow_mixed.mark_entry(plain as u64).map(|(bit, _)| bit);
        // Raw UAV starting at `plain` (offset `page` = 16384 dwords).
        let uav_plain: [u32; 10] = [39, 1, (page / 4) as u32, 0, 64, 0, 0, 0, 1, 0];
        let slot1 = title_heap.start + 32;
        crate::descriptor::create_uav(std::ptr::null(), buf_mixed, std::ptr::null(), &uav_plain, slot1);
        let view1 = crate::descriptor::view(slot1).expect("mixed plain uav view");
        let _ = heap_host.mirror(1, slot1, true);

        let list3 = host.new_list().unwrap();
        readable(&list3, &used(ResourceType::Uav), &view1, true);
        list3.defer_copy((plain + 16) as *mut u8, cpu_val.as_ptr(), cpu_val.len());
        dispatch(&list3, heap_host, 1, 0, 0xDEAD_BEEF, plain_mark);
        list3.close();
        host.execute(&[&list3]);
        assert_eq!(host.log_messages(), None);

        // SAFETY: bytes inside our plain page.
        unsafe {
            assert_eq!(
                (plain as *const u32).read_volatile(),
                0xDEAD_BEEF,
                "GPU write on untracked page of mixed shadow must copy back"
            );
            assert_eq!(
                ((plain + 16) as *const u32).read_volatile(),
                0xCAFE_BABE,
                "CPU write on untracked page of mixed shadow must not be clobbered"
            );
        }
        // SAFETY: plain data filled in by the query.
        let mut info: MEMORY_BASIC_INFORMATION = unsafe { std::mem::zeroed() };
        // SAFETY: valid out pointer.
        unsafe { VirtualQuery(plain as *const _, &mut info, size_of_val(&info)) };
        assert_eq!(
            info.Protect, PAGE_READWRITE,
            "untracked plain page must remain PAGE_READWRITE after copy-back"
        );
        crate::shadow::unregister(buf_mixed as usize, pool as u64);
    }

    /// Unaligned and `< 64 KiB` / cross-page `Shadow`s clamp `page_range`
    /// cleanly and copy back marked sub-range UAV writes across the page
    /// boundary without disturbing surrounding title memory.
    #[test]
    fn marked_subpage_uav_unaligned_and_cross_page_shadow_bounds() {
        if !marked_writes_build() {
            return;
        }
        let Some(host) = crate::host::host() else {
            return;
        };
        let p = crate::watch::tests::pool(2);
        let shadow_addr = p as u64 + 0xFF00;
        let shadow_len = 0x300u64;
        // Pure `Shadow::page_range` checks: single-page unaligned `< 64 KiB`.
        let small = p as u64 + 0x1000;
        crate::shadow::register(1, small, 0x2000);
        let small_copy = crate::shadow::find(small, 0x2000).expect("small shadow");
        assert_eq!(small_copy.page_range(small + 0x400, 0x100), (small, 0x2000));
        assert_eq!(small_copy.page_range(small, 0), (small, 0));
        assert_eq!(
            small_copy.page_range(small + 0x3000, 0x100),
            (small + 0x2000, 0)
        );
        assert_eq!(
            small_copy.page_range(u64::MAX - 8, 128),
            (small + 0x2000, 0)
        );
        crate::shadow::unregister(1, small);

        // SAFETY: sentinel bytes outside `[shadow_addr, shadow_addr + shadow_len)`.
        unsafe {
            ((p + 0xFE00) as *mut u32).write_volatile(0xA1A2_A3A4);
            ((p + 0x1_0300) as *mut u32).write_volatile(0xB1B2_B3B4);
        }
        let buf = buffer_at_address(shadow_addr, shadow_len);
        let shadow = crate::shadow::find(shadow_addr, shadow_len).expect("cross-page shadow");
        assert_eq!(shadow.page_range(shadow_addr, 0x80), (shadow_addr, 0x100));
        assert_eq!(
            shadow.page_range(p as u64 + 0x1_0000, 0x80),
            (p as u64 + 0x1_0000, 0x200)
        );
        assert_eq!(
            shadow.page_range(shadow_addr + 0x80, 0x100),
            (shadow_addr, shadow_len)
        );

        let (_, title_heap, heap_host) = view_heap(8);
        let _ = host.log_messages();
        // UAV `[p + 0xFF80 .. p + 0x1_0080)` crosses the 64 KiB page boundary.
        let uav_desc: [u32; 10] = [39, 1, 32, 0, 64, 0, 0, 0, 1, 0];
        let slot0 = title_heap.start;
        crate::descriptor::create_uav(std::ptr::null(), buf, std::ptr::null(), &uav_desc, slot0);
        let view = crate::descriptor::view(slot0).expect("cross-page subrange uav");
        let _ = heap_host.mirror(0, slot0, true);

        let shader = compile_cs(MARKED_WRITE_SOURCE, MARKED_WRITES);
        let list = host.new_list().unwrap();
        readable(&list, &used(ResourceType::Uav), &view, true);
        dispatch_marked_uav(&list, heap_host, &shader, 0, 0, 0, 0x1234_5678);
        dispatch_marked_uav(&list, heap_host, &shader, 0, 0, 0x80, 0x9ABC_DEF0);
        list.close();
        host.execute(&[&list]);

        assert_eq!(host.log_messages(), None);
        // SAFETY: bytes inside our 2-page pool.
        unsafe {
            assert_eq!(((p + 0xFF80) as *const u32).read_volatile(), 0x1234_5678);
            assert_eq!(((p + 0x1_0000) as *const u32).read_volatile(), 0x9ABC_DEF0);
            assert_eq!(((p + 0xFE00) as *const u32).read_volatile(), 0xA1A2_A3A4);
            assert_eq!(((p + 0x1_0300) as *const u32).read_volatile(), 0xB1B2_B3B4);
        }
        crate::shadow::unregister(buf as usize, shadow_addr);
    }

    /// Multiple sub-64 KiB marked UAVs and a preceding `Step::Copy` on the
    /// same 64 KiB page within one command list all land in title memory.
    #[test]
    fn marked_subpage_uavs_after_copy_in_same_list() {
        if !marked_writes_build() {
            return;
        }
        let Some(host) = crate::host::host() else {
            return;
        };
        let p = crate::watch::tests::pool(1);
        let buf = buffer_at_address(p as u64, crate::watch::PAGE as u64);
        let (_, title_heap, heap_host) = view_heap(8);
        let _ = host.log_messages();

        let uav1_desc: [u32; 10] = [39, 1, 1024, 0, 64, 0, 0, 0, 1, 0];
        let uav2_desc: [u32; 10] = [39, 1, 8192, 0, 64, 0, 0, 0, 1, 0];
        let slot0 = title_heap.start;
        let slot1 = title_heap.start + 32;
        crate::descriptor::create_uav(std::ptr::null(), buf, std::ptr::null(), &uav1_desc, slot0);
        crate::descriptor::create_uav(std::ptr::null(), buf, std::ptr::null(), &uav2_desc, slot1);
        let view1 = crate::descriptor::view(slot0).expect("uav1");
        let view2 = crate::descriptor::view(slot1).expect("uav2");
        let _ = heap_host.mirror(0, slot0, true);
        let _ = heap_host.mirror(1, slot1, true);

        let copy_src = 0x1111_1111u32.to_le_bytes();
        let shader = compile_cs(MARKED_WRITE_SOURCE, MARKED_WRITES);
        let list = host.new_list().unwrap();
        list.defer_copy((p + 0x200) as *mut u8, copy_src.as_ptr(), copy_src.len());
        readable(&list, &used(ResourceType::Uav), &view1, true);
        dispatch_marked_uav(&list, heap_host, &shader, 0, 0, 0, 0x2222_2222);
        readable(&list, &used(ResourceType::Uav), &view2, true);
        dispatch_marked_uav(&list, heap_host, &shader, 1, 1, 0, 0x3333_3333);
        list.close();
        host.execute(&[&list]);

        assert_eq!(host.log_messages(), None);
        // SAFETY: bytes inside our page.
        unsafe {
            assert_eq!(((p + 0x200) as *const u32).read_volatile(), 0x1111_1111);
            assert_eq!(((p + 0x1000) as *const u32).read_volatile(), 0x2222_2222);
            assert_eq!(((p + 0x8000) as *const u32).read_volatile(), 0x3333_3333);
        }
        crate::shadow::unregister(buf as usize, p as u64);
    }

    /// A root UAV bound at a non-64 KiB-aligned offset (`State::root_view`)
    /// and a bindless heap UAV hull (`crate::bindless::refresh`) both expand
    /// their refresh to enclosing 64 KiB page bounds.
    #[test]
    fn marked_unaligned_root_uav_and_bindless_subpage_refresh() {
        if !marked_writes_build() {
            return;
        }
        let Some(host) = crate::host::host() else {
            return;
        };
        let p = crate::watch::tests::pool(2);
        // SAFETY: committed write-watched 2-page buffer.
        unsafe {
            ((p + 0x100) as *mut u32).write_volatile(0x1122_3344);
            ((p + 0x1_0100) as *mut u32).write_volatile(0x5566_7788);
        }
        let buf = buffer_at_address(p as u64, (2 * crate::watch::PAGE) as u64);
        let (heap, title_heap, heap_host) = view_heap(8);
        let _ = host.log_messages();

        let shader = compile_cs(MARKED_WRITE_SOURCE, MARKED_WRITES);
        let list = host.new_list().unwrap();
        let mut state = State::default();
        state.set_heaps(&[heap]);
        state.set_view(true, 0, p as u64 + 4096);
        let (index, offset) = state.root_view(&list, true, 0, true, None).expect("root uav");
        assert_eq!(offset, 4096);
        dispatch_marked_uav(&list, heap_host, &shader, index, index, offset, 0xBEEF_CAFE);
        list.close();
        host.execute(&[&list]);

        assert_eq!(host.log_messages(), None);
        // SAFETY: bytes inside our 2-page pool.
        unsafe {
            assert_eq!(((p + 4096) as *const u32).read_volatile(), 0xBEEF_CAFE);
            assert_eq!(((p + 0x100) as *const u32).read_volatile(), 0x1122_3344);
            assert_eq!(((p + 0x1_0100) as *const u32).read_volatile(), 0x5566_7788);
        }
        crate::shadow::unregister(buf as usize, p as u64);

        // Bindless refresh expands marked UAV hulls to `page_range` while keeping SRV hulls exact.
        let p2 = crate::watch::tests::pool(1);
        let buf2 = buffer_at_address(p2 as u64, crate::watch::PAGE as u64);
        let uav_desc: [u32; 10] = [39, 1, 1024, 0, 64, 0, 0, 0, 1, 0];
        let srv_desc: [u32; 10] = [39, 1, 0x1688, 0, 2048, 0, 64, 0, 1, 0];
        crate::descriptor::create_uav(
            std::ptr::null(),
            buf2,
            std::ptr::null(),
            &uav_desc,
            title_heap.start + 32,
        );
        crate::descriptor::create_srv(std::ptr::null(), buf2, &srv_desc, title_heap.start + 64);
        let uav_view = crate::descriptor::view(title_heap.start + 32).expect("bindless uav");
        let srv_view = crate::descriptor::view(title_heap.start + 64).expect("bindless srv");
        let viewed = crate::bindless::Viewed::default();
        viewed.set(0, Some(&uav_view), true);
        viewed.set(1, Some(&srv_view), true);
        let mut refreshed = Vec::new();
        crate::bindless::refresh(&viewed, |_, a, l| refreshed.push((a, l)));
        refreshed.sort_unstable();
        assert_eq!(refreshed, vec![(p2, crate::watch::PAGE), (p2 + 8192, 256)]);
        crate::shadow::unregister(buf2 as usize, p2 as u64);
    }

    fn compile() -> (ID3D12RootSignature, ID3D12PipelineState) {
        let constants = Param::Constants { register: 0, count: 2 };
        compile_cs(SOURCE, &[Param::Table(&[srv(0, 1)]), Param::Table(&[uav(0, 1)]), constants])
    }

    /// A compute shader with a root signature of `params`.
    fn compile_cs(source: &str, params: &[Param]) -> (ID3D12RootSignature, ID3D12PipelineState) {
        crate::compute::pipeline(source, s!("root"), 0, params).unwrap()
    }

    /// Which pixels of a 64x64 host D32S8 `texture` (title texture `depth`)
    /// pass an EQUAL depth test against a square (two triangles, as the
    /// title's rectangles become) drawn over pixel `k - 1` (row by row) with
    /// z = k times the title's 1/65535 (`0x37800080`), the depth
    /// `VS_TiledLighting` gives draw k: entry `k - 1`.
    fn equal_test_passes(depth: This, texture: &ID3D12Resource) -> Vec<bool> {
        use windows::Win32::Graphics::Direct3D::{Fxc::D3DCompile, ID3DBlob};
        use windows::Win32::Graphics::Dxgi::Common::*;
        const SOURCE: &str = r#"
static const float2 CORNERS[6] = { float2(0, 0), float2(1, 0), float2(0, 1), float2(1, 0), float2(1, 1), float2(0, 1) };
float4 vs(uint v : SV_VertexID) : SV_Position {
    uint i = v / 6;
    float2 p = float2(i % 64, i / 64) + CORNERS[v % 6];
    return float4(p.x / 32.0 - 1.0, 1.0 - p.y / 32.0, (float)(i + 1) * asfloat(0x37800080), 1.0);
}
uint ps() : SV_Target { return 1; }
"#;
        let host = crate::host::host().unwrap();
        let blob = |entry: windows::core::PCSTR, target: windows::core::PCSTR| {
            let mut code: Option<ID3DBlob> = None;
            // SAFETY: the source and names outlive the call.
            unsafe { D3DCompile(SOURCE.as_ptr().cast(), SOURCE.len(), s!("equal"), None, None, entry, target, 0, 0, &mut code, None) }.unwrap();
            code.unwrap()
        };
        let (vs, ps) = (blob(s!("vs"), s!("vs_5_1")), blob(s!("ps"), s!("ps_5_1")));
        // SAFETY: the blobs' own bytes.
        let bytes = |b: &ID3DBlob| unsafe { D3D12_SHADER_BYTECODE { pShaderBytecode: b.GetBufferPointer(), BytecodeLength: b.GetBufferSize() } };
        // The empty root signature, serialized (not embedded in the shader:
        // Wine's HLSL compiler ignores `[RootSignature]`).
        let root = crate::compute::root_signature(&host.device, &[]).unwrap();
        let mut desc = D3D12_GRAPHICS_PIPELINE_STATE_DESC {
            // SAFETY: borrowed for the call.
            pRootSignature: unsafe { std::mem::transmute_copy(&root) },
            VS: bytes(&vs),
            PS: bytes(&ps),
            SampleMask: u32::MAX,
            PrimitiveTopologyType: D3D12_PRIMITIVE_TOPOLOGY_TYPE_TRIANGLE,
            NumRenderTargets: 1,
            DSVFormat: DXGI_FORMAT_D32_FLOAT_S8X24_UINT,
            SampleDesc: DXGI_SAMPLE_DESC { Count: 1, Quality: 0 },
            ..Default::default()
        };
        desc.RTVFormats[0] = DXGI_FORMAT_R32_UINT;
        desc.BlendState.RenderTarget[0].RenderTargetWriteMask = 0xf;
        desc.RasterizerState.FillMode = D3D12_FILL_MODE_SOLID;
        desc.RasterizerState.CullMode = D3D12_CULL_MODE_NONE;
        desc.RasterizerState.DepthClipEnable = true.into();
        desc.DepthStencilState.DepthEnable = true.into();
        desc.DepthStencilState.DepthWriteMask = D3D12_DEPTH_WRITE_MASK_ZERO;
        desc.DepthStencilState.DepthFunc = D3D12_COMPARISON_FUNC_EQUAL;
        // SAFETY: a complete description.
        let pso: ID3D12PipelineState = unsafe { host.device.CreateGraphicsPipelineState(&desc) }.unwrap();
        let target_desc = D3D12_RESOURCE_DESC {
            Dimension: D3D12_RESOURCE_DIMENSION_TEXTURE2D,
            Width: 64,
            Height: 64,
            DepthOrArraySize: 1,
            MipLevels: 1,
            Format: DXGI_FORMAT_R32_UINT,
            SampleDesc: DXGI_SAMPLE_DESC { Count: 1, Quality: 0 },
            Flags: D3D12_RESOURCE_FLAG_ALLOW_RENDER_TARGET,
            ..Default::default()
        };
        let heap = D3D12_HEAP_PROPERTIES { Type: D3D12_HEAP_TYPE_DEFAULT, ..Default::default() };
        let mut target: Option<ID3D12Resource> = None;
        // SAFETY: plain creation.
        unsafe {
            host.device.CreateCommittedResource(&heap, D3D12_HEAP_FLAG_NONE, &target_desc, D3D12_RESOURCE_STATE_RENDER_TARGET, None, &mut target)
        }
        .unwrap();
        let target = target.unwrap();
        let mut f = D3D12_PLACED_SUBRESOURCE_FOOTPRINT::default();
        let mut total = 0;
        // SAFETY: outputs for one subresource.
        unsafe { host.device.GetCopyableFootprints(&target_desc, 0, 1, 0, Some(&mut f), None, None, Some(&mut total)) };
        let (raw, data) = host.shared_buffer(total).unwrap();
        let list = host.new_list().unwrap();
        crate::list::ensure_state(&list, depth, 0x10);
        let dsv = D3D12_DEPTH_STENCIL_VIEW_DESC {
            Format: DXGI_FORMAT_D32_FLOAT_S8X24_UINT,
            ViewDimension: D3D12_DSV_DIMENSION_TEXTURE2D,
            ..Default::default()
        };
        let (rtv, dsv) = (host.rtv(&target, None), host.dsv(texture, Some(&dsv)));
        // SAFETY: an open list; the target in the render target state, the
        // depth texture in the depth write state; copies into our buffer.
        unsafe {
            let l = &list.list;
            l.ClearRenderTargetView(rtv, &[0.0; 4], None);
            l.OMSetRenderTargets(1, Some(&rtv), false, Some(&dsv));
            l.RSSetViewports(&[D3D12_VIEWPORT { Width: 64.0, Height: 64.0, MaxDepth: 1.0, ..Default::default() }]);
            l.RSSetScissorRects(&[windows::Win32::Foundation::RECT { right: 64, bottom: 64, ..Default::default() }]);
            l.SetGraphicsRootSignature(&root);
            l.SetPipelineState(&pso);
            l.IASetPrimitiveTopology(windows::Win32::Graphics::Direct3D::D3D_PRIMITIVE_TOPOLOGY_TRIANGLELIST);
            l.DrawInstanced(64 * 64 * 6, 1, 0, 0);
            l.ResourceBarrier(&[crate::host::transition(&target, D3D12_RESOURCE_STATE_RENDER_TARGET, D3D12_RESOURCE_STATE_COPY_SOURCE)]);
            let to = D3D12_TEXTURE_COPY_LOCATION {
                pResource: std::mem::transmute_copy(&raw),
                Type: D3D12_TEXTURE_COPY_TYPE_PLACED_FOOTPRINT,
                Anonymous: D3D12_TEXTURE_COPY_LOCATION_0 { PlacedFootprint: f },
            };
            let from = D3D12_TEXTURE_COPY_LOCATION {
                pResource: std::mem::transmute_copy(&target),
                Type: D3D12_TEXTURE_COPY_TYPE_SUBRESOURCE_INDEX,
                Anonymous: D3D12_TEXTURE_COPY_LOCATION_0 { SubresourceIndex: 0 },
            };
            l.CopyTextureRegion(&to, 0, 0, 0, &from, None);
        }
        list.close();
        host.execute(&[&list]);
        (0..64 * 64)
            .map(|i| {
                // SAFETY: inside the mapped buffer, by the footprint.
                unsafe { (data.add(f.Offset as usize + (i / 64) * f.Footprint.RowPitch as usize + (i % 64) * 4) as *const u32).read() == 1 }
            })
            .collect()
    }

    /// A title buffer of `bytes` in memory of its own.
    pub(super) fn buffer(bytes: u64) -> This {
        let desc: [u32; 14] = [1, 0, 0, 0, bytes as u32, (bytes >> 32) as u32, 1, 1 | 1 << 16, 0, 1, 0, 1, 0, 0];
        let properties = [1u32, 0, 0, 0, 0];
        let mut out = std::ptr::null();
        let iid = crate::object::guid(0x696442be, 0xa72e, 0x4059, [0xbc, 0x79, 0x5b, 0x5c, 0x98, 0x04, 0x0f, 0xad]);
        let made: HRESULT = crate::resource::create_committed(
            std::ptr::null(),
            properties.as_ptr().cast(),
            0,
            desc.as_ptr().cast(),
            0,
            std::ptr::null(),
            &iid,
            &mut out,
        );
        assert_eq!(made, S_OK);
        out
    }

    /// Three-channel typed SRVs keep their 12-byte element stride and a
    /// format-correct w through the host descriptor.
    #[test]
    fn three_channel_typed_srvs_reach_shaders_with_their_own_stride() {
        if !crate::under_wine() {
            return;
        }
        let Some(host) = crate::host::host() else { return };
        let (src, dst) = (buffer(4 * 12), buffer(256));
        let (src_bytes, dst_bytes) = (
            crate::resource::buffer_bytes(src).unwrap().0,
            crate::resource::buffer_bytes(dst).unwrap().0,
        );
        let data = [
            1000u32, 2000, 3000, 1003, 2003, 3003, 1006, 2006, 3006, 1009, 2009, 3009,
        ];
        // SAFETY: the source buffer's 48 bytes.
        unsafe { std::ptr::copy_nonoverlapping(data.as_ptr().cast::<u8>(), src_bytes, 48) };
        let null = std::ptr::null();
        let heap_desc: [u32; 8] = [0, 2, 1, 0, 0, 0, 0, 0];
        let iid = crate::object::guid(0x8efb471d, 0x616c, 0x4f49, [0x90, 0xf7, 0x12, 0x7b, 0xb7, 0x63, 0xfa, 0x51]);
        let uav_words = [39, 1, 0, 0, 64, 0, 0, 0, 1, 0];
        let out = crate::shadow::find(dst_bytes as u64, 256).expect("the output's host copy");
        for (format, kind, w) in [(6, "float", 0x3f80_0000), (7, "uint", 1), (8, "int", 1)] {
            let mut heap = null;
            assert_eq!(
                crate::descriptor::create_heap(null, heap_desc.as_ptr().cast(), &iid, &mut heap),
                S_OK
            );
            let title_heap = crate::descriptor::heap(heap);
            let srv_words = [format, 1, 0x1688, 0, 0, 0, 4, 0, 0, 0];
            crate::descriptor::create_srv(null, src, &srv_words, title_heap.start);
            crate::descriptor::create_uav(null, dst, null, &uav_words, title_heap.start + 32);
            let heap_host = title_heap.host().expect("a host view heap");
            assert!(heap_host.made(0) && heap_host.kept(0), "the source view is made");
            assert!(heap_host.made(1) && heap_host.kept(1), "the output view is made");
            let source = format!(
                r#"Buffer<{kind}4> src : register(t0);
RWByteAddressBuffer dst : register(u0);
[numthreads(1, 1, 1)]
void main(uint3 id : SV_DispatchThreadID) {{
    dst.Store4(id.x * 16, asuint(src[id.x]));
}}
"#
            );
            let (root, pso) = compile_cs(
                &source,
                &[Param::Table(&[srv(0, 1)]), Param::Table(&[uav(0, 1)])],
            );
            let list = host.new_list().unwrap();
            // SAFETY: an open list and live objects of ours.
            unsafe {
                list.list.SetDescriptorHeaps(&[Some(heap_host.heap.clone())]);
                list.list.SetComputeRootSignature(&root);
                list.list.SetComputeRootDescriptorTable(0, heap_host.gpu(0));
                list.list.SetComputeRootDescriptorTable(1, heap_host.gpu(1));
                list.list.SetPipelineState(&pso);
                list.list.Dispatch(4, 1, 1);
            }
            list.close();
            crate::shadow::find(src_bytes as u64, 48)
                .expect("the source's host copy")
                .refresh(&list, src_bytes as u64, 48);
            host.execute(&[&list]);
            assert_eq!(
                out.read_host(dst_bytes as u64, 16),
                [
                    1000, 2000, 3000, w, 1003, 2003, 3003, w, 1006, 2006, 3006, w, 1009, 2009,
                    3009, w,
                ],
                "{kind} w"
            );
        }
    }

    /// A widened RGB32 SRV sees a raw UAV write made earlier in the same
    /// submission when the title issues a buffer barrier between them.
    #[test]
    fn a_widened_srv_sees_an_earlier_uav_write_in_the_same_submission() {
        if !crate::under_wine() {
            return;
        }
        let Some(host) = crate::host::host() else { return };
        let (src, dst) = (buffer(48), buffer(256));
        let (src_bytes, dst_bytes) = (
            crate::resource::buffer_bytes(src).unwrap().0,
            crate::resource::buffer_bytes(dst).unwrap().0,
        );
        let initial = [1000u32, 2000, 3000, 1003, 2003, 3003, 1006, 2006, 3006, 1009, 2009, 3009];
        // SAFETY: the source buffer's 48 bytes.
        unsafe { std::ptr::copy_nonoverlapping(initial.as_ptr().cast::<u8>(), src_bytes, 48) };

        let null = std::ptr::null();
        let heap_desc: [u32; 8] = [0, 5, 1, 0, 0, 0, 0, 0];
        let iid = crate::object::guid(0x8efb471d, 0x616c, 0x4f49, [0x90, 0xf7, 0x12, 0x7b, 0xb7, 0x63, 0xfa, 0x51]);
        let mut heap = null;
        assert_eq!(crate::descriptor::create_heap(null, heap_desc.as_ptr().cast(), &iid, &mut heap), S_OK);
        let title_heap = crate::descriptor::heap(heap);
        // The raw view starts inside the second RGB32 element. Its range is
        // dword-aligned, so it shares the source shadow, while the widening
        // copy has to refresh the whole touched elements.
        let source_raw = [39, 1, 5, 0, 6, 0, 0, 0, 1, 0];
        let output_raw = [39, 1, 0, 0, 64, 0, 0, 0, 1, 0];
        let srv_words = [6, 1, 0x1688, 0, 0, 0, 4, 0, 0, 0];
        crate::descriptor::create_uav(null, src, null, &source_raw, title_heap.start);
        crate::descriptor::create_srv(null, src, &srv_words, title_heap.start + 32);
        crate::descriptor::create_uav(null, dst, null, &output_raw, title_heap.start + 64);
        let first_result = [39, 1, 4, 0, 4, 0, 0, 0, 1, 0];
        crate::descriptor::create_uav(null, dst, null, &first_result, title_heap.start + 128);
        let heap_host = title_heap.host().expect("a host view heap");
        assert!(heap_host.made(0) && heap_host.kept(0), "the source UAV is made");
        assert!(heap_host.made(1) && heap_host.kept(1), "the widened SRV is made");
        assert!(heap_host.made(2) && heap_host.kept(2), "the output UAV is made");
        assert_eq!(heap_host.widened_buffers().len(), 1, "one live widened source view");

        let writer = compile_cs(
            r#"
RWByteAddressBuffer src : register(u0);
[numthreads(1, 1, 1)]
void main() { src.Store4(0, uint4(9, 8, 7, 0)); }
"#,
            &[Param::Table(&[uav(0, 1)])],
        );
        let writer_again = compile_cs(
            r#"
RWByteAddressBuffer src : register(u0);
[numthreads(1, 1, 1)]
void main() { src.Store4(0, uint4(4, 5, 6, 7)); }
"#,
            &[Param::Table(&[uav(0, 1)])],
        );
        let reader = compile_cs(
            r#"
Buffer<uint4> src : register(t0);
RWByteAddressBuffer dst : register(u0);
[numthreads(1, 1, 1)]
void main() { dst.Store4(0, asuint(src[1])); }
"#,
            &[Param::Table(&[srv(0, 1)]), Param::Table(&[uav(0, 1)])],
        );
        let source_copy = crate::shadow::find(src_bytes as u64, 48).expect("the source copy");
        source_copy.update(src_bytes as usize, 48);
        let list = host.new_list().unwrap();
        let source_view = crate::descriptor::view(title_heap.start).expect("the source UAV record");
        readable(&list, &used(ResourceType::Uav), &source_view, false);
        // SAFETY: an open list, live objects and descriptors in our heap.
        unsafe {
            list.list.SetDescriptorHeaps(&[Some(heap_host.heap.clone())]);
            list.list.SetComputeRootSignature(&writer.0);
            list.list.SetComputeRootDescriptorTable(0, heap_host.gpu(0));
            list.list.SetPipelineState(&writer.1);
            list.list.Dispatch(1, 1, 1);
            list.list.ResourceBarrier(&[crate::host::uav_barrier()]);
        }
        list.sync_widened();
        // The title makes another RGB32 view after the barrier. Its CPU copy
        // predates the UAV write, so its first read must also consume the
        // writes retained from earlier in this list.
        let late_srv_words = [7, 1, 0x1688, 0, 0, 0, 4, 0, 0, 0];
        crate::descriptor::create_srv(null, src, &late_srv_words, title_heap.start + 96);
        assert!(heap_host.made(3) && heap_host.kept(3), "the late SRV is made");
        assert_eq!(heap_host.widened_buffers().len(), 2, "a distinct late widened view");
        let late_view = crate::descriptor::view(title_heap.start + 96).expect("the late SRV record");
        readable(&list, &used(ResourceType::Srv), &late_view, false);
        // SAFETY: the widening copy finished and the descriptors are live.
        unsafe {
            list.list.SetComputeRootSignature(&reader.0);
            list.list.SetComputeRootDescriptorTable(0, heap_host.gpu(3));
            list.list.SetComputeRootDescriptorTable(1, heap_host.gpu(4));
            list.list.SetPipelineState(&reader.1);
            list.list.Dispatch(1, 1, 1);
        }
        // A second write to the same range after the view is current must be
        // pending work, not deduplicated against the retained history.
        readable(&list, &used(ResourceType::Uav), &source_view, false);
        // SAFETY: an open list, live objects and descriptors in our heap.
        unsafe {
            list.list.SetComputeRootSignature(&writer_again.0);
            list.list.SetComputeRootDescriptorTable(0, heap_host.gpu(0));
            list.list.SetPipelineState(&writer_again.1);
            list.list.Dispatch(1, 1, 1);
            list.list.ResourceBarrier(&[crate::host::uav_barrier()]);
        }
        list.sync_widened();
        // SAFETY: the second widening copy finished.
        unsafe {
            list.list.SetComputeRootSignature(&reader.0);
            list.list.SetComputeRootDescriptorTable(0, heap_host.gpu(3));
            list.list.SetComputeRootDescriptorTable(1, heap_host.gpu(2));
            list.list.SetPipelineState(&reader.1);
            list.list.Dispatch(1, 1, 1);
        }
        list.close();
        host.execute(&[&list]);
        assert_eq!(
            source_copy.read_host(src_bytes as u64 + 20, 4),
            [4, 5, 6, 7],
            "the second raw UAV write reaches its shadow"
        );
        let out = crate::shadow::find(dst_bytes as u64, 256).expect("the output copy");
        assert_eq!(
            out.read_host(dst_bytes as u64 + 16, 4),
            [1003, 2003, 9, 1],
            "first write"
        );
        assert_eq!(
            out.read_host(dst_bytes as u64, 4),
            [1003, 2003, 4, 1],
            "second write"
        );
    }

    /// A heap-writing draw is sampled at its barrier. A widened view made
    /// after that barrier must see the heap's then-current UAV range.
    #[test]
    fn a_late_widened_view_follows_a_bindless_write() {
        late_widened_view(false, 1);
    }

    /// Same end-to-end widening path with `VICTORY_CENSUS_FAST=1`'s cached UAV
    /// snapshot and multiple heap-writing draws before the barrier.
    #[test]
    fn a_late_widened_view_follows_a_bindless_write_with_fast_census() {
        late_widened_view(true, 3);
    }

    fn late_widened_view(fast_census: bool, marks: usize) {
        if !crate::under_wine() {
            return;
        }
        let Some(host) = crate::host::host() else { return };
        let (src, dst) = (buffer(48), buffer(256));
        let (src_bytes, dst_bytes) = (
            crate::resource::buffer_bytes(src).unwrap().0,
            crate::resource::buffer_bytes(dst).unwrap().0,
        );
        let initial = [1000u32, 2000, 3000, 1003, 2003, 3003, 1006, 2006, 3006, 1009, 2009, 3009];
        // SAFETY: the source buffer's 48 bytes.
        unsafe { std::ptr::copy_nonoverlapping(initial.as_ptr().cast::<u8>(), src_bytes, 48) };

        let old_raw = [39, 1, 0, 0, 12, 0, 0, 0, 1, 0];
        let new_raw = [39, 1, 3, 0, 9, 0, 0, 0, 1, 0];
        let output_raw = [39, 1, 0, 0, 64, 0, 0, 0, 1, 0];
        let old = View::Unordered(src as usize, 0, Some(old_raw));
        let new = View::Unordered(src as usize, 0, Some(new_raw));
        let srv_view = View::Shader(src as usize, Some([7, 1, 0x1688, 0, 0, 0, 4, 0, 0, 0]));
        let null = std::ptr::null();
        let heap_desc: [u32; 8] = [0, 3, 1, 0, 0, 0, 0, 0];
        let iid = crate::object::guid(0x8efb471d, 0x616c, 0x4f49, [0x90, 0xf7, 0x12, 0x7b, 0xb7, 0x63, 0xfa, 0x51]);
        let mut heap = null;
        assert_eq!(crate::descriptor::create_heap(null, heap_desc.as_ptr().cast(), &iid, &mut heap), S_OK);
        let title_heap = crate::descriptor::heap(heap);
        crate::descriptor::create_uav(null, src, null, &old_raw, title_heap.start);
        crate::descriptor::create_uav(null, dst, null, &output_raw, title_heap.start + 64);
        let heap_host = title_heap.host().expect("a host view heap");
        let viewed = std::sync::Arc::new(crate::bindless::Viewed::new(fast_census));
        viewed.set(0, Some(&old), true);
        crate::shadow::find(src_bytes as u64, 48).expect("the source copy").update(src_bytes as usize, 48);

        let writer = compile_cs(
            r#"
RWByteAddressBuffer src : register(u0);
[numthreads(1, 1, 1)]
void main() { src.Store4(0, uint4(9, 8, 7, 0)); }
"#,
            &[Param::Table(&[uav(0, 1)])],
        );
        let reader = compile_cs(
            r#"
Buffer<uint4> src : register(t0);
RWByteAddressBuffer dst : register(u0);
[numthreads(1, 1, 1)]
void main() { dst.Store4(0, asuint(src[1])); }
"#,
            &[Param::Table(&[srv(0, 1)]), Param::Table(&[uav(0, 1)])],
        );
        let list = host.new_list().unwrap();
        list.mark_widened_heap(&viewed);
        // SAFETY: an open list and live descriptors.
        unsafe {
            list.list.SetDescriptorHeaps(&[Some(heap_host.heap.clone())]);
            list.list.SetComputeRootSignature(&writer.0);
            list.list.SetComputeRootDescriptorTable(0, heap_host.gpu(0));
            list.list.SetPipelineState(&writer.1);
            list.list.Dispatch(1, 1, 1);
        }
        // The title rewrites the heap slot before its barrier. Both the GPU
        // descriptor and the range sampled for widening must name the new view.
        crate::descriptor::create_uav(null, src, null, &new_raw, title_heap.start);
        viewed.set(0, Some(&new), true);
        for _ in 1..marks {
            list.mark_widened_heap(&viewed);
        }
        // SAFETY: an open list.
        unsafe {
            list.list.ResourceBarrier(&[crate::host::uav_barrier()]);
        }
        list.sync_widened();
        // The range must be frozen at the barrier even when no widened view
        // exists yet; the title can remove the heap slot before the first SRV.
        viewed.set(0, None, true);
        let srv_words = [7, 1, 0x1688, 0, 0, 0, 4, 0, 0, 0];
        crate::descriptor::create_srv(null, src, &srv_words, title_heap.start + 32);
        readable(&list, &used(ResourceType::Srv), &srv_view, false);
        // SAFETY: the widening copy finished and the descriptors are live.
        unsafe {
            list.list.SetComputeRootSignature(&reader.0);
            list.list.SetComputeRootDescriptorTable(0, heap_host.gpu(1));
            list.list.SetComputeRootDescriptorTable(1, heap_host.gpu(2));
            list.list.SetPipelineState(&reader.1);
            list.list.Dispatch(1, 1, 1);
        }
        list.close();
        host.execute(&[&list]);
        let source = crate::shadow::find(src_bytes as u64, 48).expect("the source copy");
        assert_eq!(source.read_host(src_bytes as u64 + 12, 4), [9, 8, 7, 0]);
        let out = crate::shadow::find(dst_bytes as u64, 256).expect("the output copy");
        assert_eq!(out.read_host(dst_bytes as u64, 4), [9, 8, 7, 1]);
    }

    /// Root UAVs declare their writes outside `readable`; they must refresh a
    /// widened SRV after a barrier just like table UAVs do.
    #[test]
    fn a_root_uav_write_reaches_a_widened_srv_in_the_same_submission() {
        if !crate::under_wine() {
            return;
        }
        let Some(host) = crate::host::host() else { return };
        let (src, dst) = (buffer(48), buffer(256));
        let (src_bytes, dst_bytes) = (
            crate::resource::buffer_bytes(src).unwrap().0,
            crate::resource::buffer_bytes(dst).unwrap().0,
        );
        let initial = [1000u32, 2000, 3000, 1003, 2003, 3003, 1006, 2006, 3006, 1009, 2009, 3009];
        // SAFETY: the source buffer's 48 bytes.
        unsafe { std::ptr::copy_nonoverlapping(initial.as_ptr().cast::<u8>(), src_bytes, 48) };

        let null = std::ptr::null();
        let heap_desc: [u32; 8] = [0, 2, 1, 0, 0, 0, 0, 0];
        let iid = crate::object::guid(0x8efb471d, 0x616c, 0x4f49, [0x90, 0xf7, 0x12, 0x7b, 0xb7, 0x63, 0xfa, 0x51]);
        let mut heap = null;
        assert_eq!(crate::descriptor::create_heap(null, heap_desc.as_ptr().cast(), &iid, &mut heap), S_OK);
        let title_heap = crate::descriptor::heap(heap);
        let srv_words = [6, 1, 0x1688, 0, 0, 0, 4, 0, 0, 0];
        let output_raw = [39, 1, 0, 0, 64, 0, 0, 0, 1, 0];
        crate::descriptor::create_srv(null, src, &srv_words, title_heap.start);
        crate::descriptor::create_uav(null, dst, null, &output_raw, title_heap.start + 32);
        let heap_host = title_heap.host().expect("a host view heap");
        let source_view = crate::descriptor::view(title_heap.start).expect("the source SRV");

        let writer = compile_cs(
            r#"
RWByteAddressBuffer src : register(u0);
[numthreads(1, 1, 1)]
void main() { src.Store4(0, uint4(9, 8, 7, 0)); }
"#,
            &[Param::Table(&[uav(0, 1)])],
        );
        let reader = compile_cs(
            r#"
Buffer<float4> src : register(t0);
RWByteAddressBuffer dst : register(u0);
[numthreads(1, 1, 1)]
void main() { dst.Store4(0, asuint(src[0])); }
"#,
            &[Param::Table(&[srv(0, 1)]), Param::Table(&[uav(0, 1)])],
        );
        let list = host.new_list().unwrap();
        let mut state = State::default();
        state.set_heaps(&[heap]);
        state.set_view(true, 0, src_bytes as u64);
        let (writer_slot, _) = state.root_view(&list, true, 0, true, None).expect("a root UAV");
        // SAFETY: an open list, live objects and descriptors in our heap.
        unsafe {
            list.list.SetDescriptorHeaps(&[Some(heap_host.heap.clone())]);
            list.list.SetComputeRootSignature(&writer.0);
            list.list.SetComputeRootDescriptorTable(0, heap_host.gpu(writer_slot));
            list.list.SetPipelineState(&writer.1);
            list.list.Dispatch(1, 1, 1);
            list.list.ResourceBarrier(&[crate::host::uav_barrier()]);
        }
        list.sync_widened();
        readable(&list, &used(ResourceType::Srv), &source_view, false);
        // SAFETY: the widening copy finished and the descriptors are live.
        unsafe {
            list.list.SetComputeRootSignature(&reader.0);
            list.list.SetComputeRootDescriptorTable(0, heap_host.gpu(0));
            list.list.SetComputeRootDescriptorTable(1, heap_host.gpu(1));
            list.list.SetPipelineState(&reader.1);
            list.list.Dispatch(1, 1, 1);
        }
        list.close();
        host.execute(&[&list]);
        let out = crate::shadow::find(dst_bytes as u64, 256).expect("the output copy");
        assert_eq!(out.read_host(dst_bytes as u64, 4), [9, 8, 7, 1.0f32.to_bits()]);
    }

    /// Root views (`victory_shader::emit`, "Root SRVs and root UAVs") through
    /// our device: a raw view of the host copy holding the root address,
    /// brought up to date from there to the copy's end, so data 1.5 MiB past
    /// the root address (past the 1 MiB the old root views brought up to
    /// date) reads right; reads past the copy's end read 0 (a host root
    /// descriptor read whatever lay there). Two root addresses in one copy
    /// share its view.
    #[test]
    fn root_views_reach_their_copy_and_read_zeros_past_it() {
        let Some(host) = crate::host::host() else {
            return; // no host GPU here
        };
        const MIB: u64 = 1 << 20;
        let (src, dst) = (buffer(4 * MIB), buffer(256));
        let (src_bytes, _) = crate::resource::buffer_bytes(src).unwrap();
        let (dst_bytes, _) = crate::resource::buffer_bytes(dst).unwrap();
        let null = std::ptr::null();
        let heap_desc: [u32; 8] = [0, 8, 1, 0, 0, 0, 0, 0];
        let mut heap = null;
        let iid = crate::object::guid(0x8efb471d, 0x616c, 0x4f49, [0x90, 0xf7, 0x12, 0x7b, 0xb7, 0x63, 0xfa, 0x51]);
        assert_eq!(crate::descriptor::create_heap(null, heap_desc.as_ptr().cast(), &iid, &mut heap), S_OK);
        let title_heap = crate::descriptor::heap(heap);
        let uav: [u32; 10] = [39, 1, 0, 0, 64, 0, 0, 0, 1, 0];
        crate::descriptor::create_uav(null, dst, null, &uav, title_heap.start + 32);
        let heap_host = title_heap.host().expect("a host view heap");
        let (root, pso) = compile();
        let list = host.new_list().unwrap();
        let out = crate::shadow::find(dst_bytes as u64, 256).expect("the output's host copy");
        let src = src_bytes as u64;
        let source = crate::shadow::find(src, 4 * MIB).expect("the input's host copy");
        let sentinel = src + 4 * MIB - 0x400;
        source.write_host(sentinel, &[0x5a; 4]);
        let cases = [
            // Data 1.5 MiB past the root address.
            (src + 0x100, 0x18_0000u32, (0..16).map(|k| 0x5000 + k).collect::<Vec<u32>>()),
            // The copy's last 256 bytes: four lanes inside, the rest past it.
            (src + 4 * MIB - 256, 0, (0..16).map(|k| if k < 4 { 0x7000 + k } else { 0 }).collect()),
        ];
        for (round, (address, offset, want)) in cases.into_iter().enumerate() {
            if round > 0 {
                list.reset();
            }
            // The title writes after recording, before submitting.
            let mut state = State::default();
            state.set_heaps(&[heap]);
            state.set_view(true, 0, address);
            let read_end = (round == 0).then_some(offset + 15 * 64 + 4);
            let (index, at) = state.root_view(&list, true, 0, false, read_end).expect("a root view");
            assert_eq!(u64::from(at), address - src, "the root address's offset in the copy");
            state.set_view(true, 1, address - 0x40);
            let other = state.root_view(&list, true, 1, false, Some(4)).expect("a root view");
            assert_eq!(other, (index, at - 0x40), "one view per copy");
            let constants = [at, offset];
            // SAFETY: an open list, live objects of ours.
            unsafe {
                list.list.SetDescriptorHeaps(&[Some(heap_host.heap.clone())]);
                list.list.SetComputeRootSignature(&root);
                list.list.SetComputeRootDescriptorTable(0, heap_host.gpu(index));
                list.list.SetComputeRootDescriptorTable(1, heap_host.gpu(1));
                list.list.SetComputeRoot32BitConstants(2, 2, constants.as_ptr().cast(), 0);
                list.list.SetPipelineState(&pso);
                list.list.Dispatch(1, 1, 1);
            }
            list.close();
            for k in 0..16u64 {
                let at = address + u64::from(offset) + 64 * k;
                if at + 4 <= src + 4 * MIB {
                    let value = if round == 0 { 0x5000 + k as u32 } else { 0x7000 + k as u32 };
                    // SAFETY: inside the title buffer.
                    unsafe { (at as *mut u32).write(value) };
                }
            }
            host.execute(&[&list]);
            assert_eq!(out.read_host(dst_bytes as u64, 16), want, "round {round}");
            assert_eq!(source.read_host(sentinel, 1), [0x5a5a_5a5a], "bytes outside the refresh");
        }

        // Rebind the same slots to a null SRV, then to a null UAV. The
        // previous round left nonzero words in the destination.
        for null_uav in [false, true] {
            list.reset();
            let mut state = State::default();
            state.set_heaps(&[heap]);
            assert!(state.root_view(&list, true, 0, false, None).is_none(), "unset root");
            state.set_view(true, 0, if null_uav { src + 0x18_0100 } else { 0 });
            state.set_view(true, 1, if null_uav { 0 } else { dst_bytes as u64 });
            let (input, at) = state.root_view(&list, true, 0, false, None).unwrap();
            let (output, out_at) = state.root_view(&list, true, 1, true, None).unwrap();
            assert_ne!(input, output);
            assert_eq!(state.root_slots.len(), 1, "only the real buffer needs a dynamic view");
            assert_eq!(out_at, 0);
            if !null_uav {
                assert_eq!(at, 0);
            }
            let constants = [at, 0];
            // SAFETY: all descriptors are in the bound heap and held through execution.
            unsafe {
                list.list.SetDescriptorHeaps(&[Some(heap_host.heap.clone())]);
                list.list.SetComputeRootSignature(&root);
                list.list.SetComputeRootDescriptorTable(0, heap_host.gpu(input));
                list.list.SetComputeRootDescriptorTable(1, heap_host.gpu(output));
                list.list.SetComputeRoot32BitConstants(2, 2, constants.as_ptr().cast(), 0);
                list.list.SetPipelineState(&pso);
                list.list.Dispatch(1, 1, 1);
            }
            list.close();
            host.execute(&[&list]);
            assert_eq!(out.read_host(dst_bytes as u64, 16), [0; 16]);
        }
    }

    /// Title memory of `bytes` (64 KiB aligned) for textures at an address.
    fn title_memory(bytes: usize) -> u64 {
        let layout = std::alloc::Layout::from_size_align(bytes, 1 << 16).unwrap();
        // SAFETY: a non-zero size; leaked, as title memory outlives a test.
        unsafe { std::alloc::alloc_zeroed(layout) as u64 }
    }

    /// A 2D resource description as words: size, format, layout, flags.
    fn texture_words(width: u32, height: u32, format: u32, layout: u32, flags: u32) -> [u32; 14] {
        [3, 0, 0, 0, width, 0, height, 1 | 1 << 16, format, 1, 0, layout, flags, 0]
    }

    /// A title texture (resource object) at `address`.
    fn texture_at_address(address: u64, words: &[u32; 14]) -> This {
        let iid = crate::object::guid(0x696442be, 0xa72e, 0x4059, [0xbc, 0x79, 0x5b, 0x5c, 0x98, 0x04, 0x0f, 0xad]);
        let mut out = std::ptr::null();
        let made = crate::resource::create_at_address(std::ptr::null(), address, words.as_ptr().cast(), 0, std::ptr::null(), &iid, &mut out);
        assert_eq!(made, S_OK);
        out
    }

    /// A placed shader view (device 0x230) of a 2D texture at `address`.
    fn placed_view(address: u64, words: [u32; 14]) -> View {
        let mut desc = [0u32; 16];
        desc[..6].copy_from_slice(&[words[8], 4, 0x1688, 0, 0, 1]);
        (desc[10], desc[11]) = (address as u32, (address >> 32) as u32);
        View::Placed(false, words, desc)
    }

    fn used(kind: ResourceType) -> UsedResource {
        UsedResource {
            kind,
            root: RootType::Table,
            table_offset_dwords: 0,
            lower_bound: 0,
            stride: 0,
            dwords: 8,
            user_data: 0,
            slot: 0,
        }
    }

    #[derive(Clone, Copy, Debug, PartialEq, Eq)]
    enum Rewrite {
        Mirrored,
        Copied,
        DuringSubmission,
    }

    /// A table slot rewritten after its draw is recorded must refresh the
    /// buffer its host descriptor names at submission, not the old one.
    #[test]
    fn a_table_slot_rewritten_before_submission_refreshes_its_new_view() {
        for mode in [Rewrite::Mirrored, Rewrite::Copied, Rewrite::DuringSubmission] {
            check_table_slot_rewrite(mode);
        }
    }

    /// `Copied` leaves the host mirror to the submission's heap sync;
    /// `DuringSubmission` rewrites after that sync, before deferred work.
    fn check_table_slot_rewrite(mode: Rewrite) {
        let Some(host) = crate::host::host() else {
            return; // no host GPU here
        };
        let (a, b, out) = (buffer(256), buffer(256), buffer(256));
        let (a_bytes, _) = crate::resource::buffer_bytes(a).unwrap();
        let (b_bytes, _) = crate::resource::buffer_bytes(b).unwrap();
        let (out_bytes, _) = crate::resource::buffer_bytes(out).unwrap();
        let memory = (mode == Rewrite::Copied).then(|| crate::watch::tests::pool(1));
        #[repr(C)]
        struct Extended {
            desc: [u32; 4],
            address: u64,
            size: u64,
        }
        let extended = Extended {
            desc: [0, 4, 1, 0],
            address: memory.map_or(0, |memory| memory as u64),
            size: memory.map_or(0, |_| 4 * 32),
        };
        let mut heap = std::ptr::null();
        let iid = crate::object::guid(
            0x8efb471d,
            0x616c,
            0x4f49,
            [0x90, 0xf7, 0x12, 0x7b, 0xb7, 0x63, 0xfa, 0x51],
        );
        assert_eq!(
            crate::descriptor::create_heap(
                std::ptr::null(),
                std::ptr::from_ref(&extended.desc).cast(),
                &iid,
                &mut heap,
            ),
            S_OK
        );
        let title_heap = crate::descriptor::heap(heap);
        if let Some(memory) = memory {
            assert_eq!(
                title_heap.start,
                memory as u64,
                "the placed heap starts at its memory"
            );
        }
        let host_heap = title_heap
            .host()
            .unwrap_or_else(|| panic!("a host view heap ({mode:?})"));
        let (a_bytes, b_bytes, out_bytes) = (a_bytes as u64, b_bytes as u64, out_bytes as u64);
        let out_copy = crate::shadow::find(out_bytes, 256).expect("the output's host copy");
        let raw: [u32; 10] = [39, 1, 0, 0, 64, 0, 0, 0, 1, 0];
        crate::descriptor::create_cbv(
            std::ptr::null(),
            &[a_bytes as u32, (a_bytes >> 32) as u32, 256, 0],
            title_heap.start,
        );
        crate::descriptor::create_uav(
            std::ptr::null(),
            out,
            std::ptr::null(),
            &raw,
            title_heap.start + 32,
        );
        let (root, pso) = compile();
        let b_desc = [b_bytes as u32, (b_bytes >> 32) as u32, 256, 0];
        let list = host.new_list().unwrap();
        if mode == Rewrite::DuringSubmission {
            let handle = title_heap.start;
            list.defer_host(move || {
                crate::descriptor::create_cbv(std::ptr::null(), &b_desc, handle)
            });
        }
        let mut state = State::default();
        state.set_heaps(&[heap]);
        state.set_table(true, 0, title_heap.start);
        assert!(
            state
                .tables(&list, &[used(ResourceType::Srv)], &[false], false, &[], true)
                .is_empty(),
            "an in-place table"
        );

        // The title replaces the record after recording, before submission.
        if mode == Rewrite::Copied {
            let mut record = [0u32; 8];
            let at = record.as_mut_ptr() as u64;
            crate::descriptor::create_cbv(std::ptr::null(), &b_desc, at);
            // SAFETY: one 32-byte record in each of our allocations.
            unsafe {
                std::ptr::copy_nonoverlapping(
                    record.as_ptr().cast::<u8>(),
                    title_heap.start as *mut u8,
                    32,
                )
            };
        } else if mode == Rewrite::Mirrored {
            crate::descriptor::create_cbv(std::ptr::null(), &b_desc, title_heap.start);
        }
        // SAFETY: inside the title buffers.
        unsafe {
            (a_bytes as *mut u32).write(0x1111_1111);
            (b_bytes as *mut u32).write(0x2222_2222);
        }
        // SAFETY: an open list and live objects of ours.
        unsafe {
            list.list
                .SetDescriptorHeaps(&[Some(host_heap.heap.clone())]);
            list.list.SetComputeRootSignature(&root);
            list.list.SetComputeRootDescriptorTable(0, host_heap.gpu(0));
            list.list.SetComputeRootDescriptorTable(1, host_heap.gpu(1));
            list.list.SetComputeRoot32BitConstants(2, 2, [0u32; 2].as_ptr().cast(), 0);
            list.list.SetPipelineState(&pso);
            list.list.Dispatch(1, 1, 1);
        }
        list.close();
        host.execute(&[&list]);
        assert_eq!(
            out_copy.read_host(out_bytes, 1),
            [0x2222_2222],
            "the new descriptor's data ({mode:?})"
        );
        assert_eq!(
            crate::object::slot::<extern "system" fn(This) -> u32>(heap, 0x10)(heap),
            0
        );
        if let Some(memory) = memory {
            crate::watch::tests::free(memory);
        }
    }

    /// A table slot the title wrote once (as it writes its materials' tables
    /// at load) keeps its view, a constant buffer view or a placed view,
    /// however many views without a resource object the title makes after it
    /// (the world makes ~5,000 more a second). A view table that freed the
    /// oldest of them past a cap left such a slot null: shaders read zeros
    /// and the world drew black once the cap was reached (#414, ~284 s in).
    /// #414's cap failed this with `VICTORY_OWNERLESS_VIEWS=64`.
    #[test]
    fn a_slot_written_once_keeps_its_view_past_newer_ownerless_views() {
        slot_keeps_its_view_past(4096, 0x7ff0_0000_0000);
    }

    /// The same past #414's default cap (2^20), which it failed.
    #[test]
    #[ignore = "makes 2^20 views (~10 s under Wine, ~300 MB kept for the run): run by name"]
    fn a_slot_written_once_keeps_its_view_past_two_to_the_twenty_ownerless_views() {
        slot_keeps_its_view_past((1 << 20) + 64, 0x7e00_0000_0000);
    }

    /// Writes a constant buffer view and a placed buffer view of the same
    /// title buffer once, each into a table of its own, reads both through
    /// a dispatch, makes `newer` constant buffer views of other memory from
    /// address `base` on, and reads both again.
    fn slot_keeps_its_view_past(newer: u64, base: u64) {
        let Some(host) = crate::host::host() else {
            return; // no host GPU here
        };
        let null = std::ptr::null();
        let (a, out) = (buffer(256), buffer(256));
        let (a_bytes, _) = crate::resource::buffer_bytes(a).unwrap();
        let (out_bytes, _) = crate::resource::buffer_bytes(out).unwrap();
        let (a_bytes, out_bytes) = (a_bytes as u64, out_bytes as u64);
        let out_copy = crate::shadow::find(out_bytes, 256).expect("the output's host copy");
        let raw: [u32; 10] = [39, 1, 0, 0, 64, 0, 0, 0, 1, 0];
        // A raw shader view of `a` placed by address: R32_TYPELESS, buffer,
        // the default mapping, 64 elements, raw; then the address.
        let mut placed = [0u32; 16];
        placed[..10].copy_from_slice(&[39, 1, 0x1688, 0, 0, 0, 64, 0, 1, 0]);
        (placed[10], placed[11]) = (a_bytes as u32, (a_bytes >> 32) as u32);
        let buffer_desc: [u32; 14] = [1, 0, 0, 0, 256, 0, 1, 1 | 1 << 16, 0, 1, 0, 1, 0, 0];
        let (root, pso) = compile();
        let iid = crate::object::guid(
            0x8efb471d,
            0x616c,
            0x4f49,
            [0x90, 0xf7, 0x12, 0x7b, 0xb7, 0x63, 0xfa, 0x51],
        );
        // A shader-visible view heap of 4 slots (no memory of its own) for
        // each view: the view in slot 0, the output in slot 1.
        let heaps = [false, true].map(|placed_view| {
            let heap_desc: [u32; 8] = [0, 4, 1, 0, 0, 0, 0, 0];
            let mut heap = null;
            assert_eq!(
                crate::descriptor::create_heap(null, heap_desc.as_ptr().cast(), &iid, &mut heap),
                S_OK
            );
            let start = crate::descriptor::heap(heap).start;
            if placed_view {
                crate::descriptor::create_placed_srv(null, &buffer_desc, &placed, start);
            } else {
                let desc = [a_bytes as u32, (a_bytes >> 32) as u32, 256, 0];
                crate::descriptor::create_cbv(null, &desc, start);
            }
            crate::descriptor::create_uav(null, out, null, &raw, start + 32);
            heap
        });
        let read = |heap: This, value: u32| {
            let title_heap = crate::descriptor::heap(heap);
            let host_heap = title_heap.host().expect("a host view heap");
            // SAFETY: inside the title buffer.
            unsafe { (a_bytes as *mut u32).write(value) };
            let list = host.new_list().unwrap();
            let mut state = State::default();
            state.set_heaps(&[heap]);
            state.set_table(true, 0, title_heap.start);
            assert!(
                state
                    .tables(&list, &[used(ResourceType::Srv)], &[], true, &[], true)
                    .is_empty(),
                "an in-place table"
            );
            // SAFETY: an open list and live objects of ours.
            unsafe {
                list.list
                    .SetDescriptorHeaps(&[Some(host_heap.heap.clone())]);
                list.list.SetComputeRootSignature(&root);
                list.list.SetComputeRootDescriptorTable(0, host_heap.gpu(0));
                list.list.SetComputeRootDescriptorTable(1, host_heap.gpu(1));
                list.list
                    .SetComputeRoot32BitConstants(2, 2, [0u32; 2].as_ptr().cast(), 0);
                list.list.SetPipelineState(&pso);
                list.list.Dispatch(1, 1, 1);
            }
            list.close();
            host.execute(&[&list]);
            out_copy.read_host(out_bytes, 1)[0]
        };
        // The constant buffer view's table, then the placed view's.
        let before = heaps.map(|heap| read(heap, 0x1111_1111));
        assert_eq!(before, [0x1111_1111; 2], "before");
        // Views of other memory, written elsewhere.
        let mut record = [0u32; 8];
        for k in 0..newer {
            let address = base + 256 * k;
            let desc = [address as u32, (address >> 32) as u32, 256, 0];
            crate::descriptor::create_cbv(null, &desc, record.as_mut_ptr() as u64);
        }
        let after = heaps.map(|heap| read(heap, 0x2222_2222));
        assert_eq!(after, [0x2222_2222; 2], "after {newer} newer views");
        for heap in heaps {
            assert_eq!(
                crate::object::slot::<extern "system" fn(This) -> u32>(heap, 0x10)(heap),
                0
            );
        }
    }

    /// Slots of a shader-visible view heap of ours.
    struct Slots(ID3D12DescriptorHeap, u32);

    impl Slots {
        fn new(n: u32) -> Self {
            let host = crate::host::host().unwrap();
            let desc = D3D12_DESCRIPTOR_HEAP_DESC {
                Type: D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV,
                NumDescriptors: n,
                Flags: D3D12_DESCRIPTOR_HEAP_FLAG_SHADER_VISIBLE,
                NodeMask: 0,
            };
            // SAFETY: plain creation.
            let heap = unsafe { host.device.CreateDescriptorHeap(&desc) }.unwrap();
            // SAFETY: a plain query.
            let step = unsafe { host.device.GetDescriptorHandleIncrementSize(D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV) };
            Slots(heap, step)
        }

        fn make(&self, i: u32, view: &View) {
            // SAFETY: a live heap.
            let mut at = unsafe { self.0.GetCPUDescriptorHandleForHeapStart() };
            at.ptr += (i * self.1) as usize;
            assert!(crate::descriptor::host::make_view(view, at).is_some(), "slot {i} made");
        }

        fn gpu(&self, i: u32) -> D3D12_GPU_DESCRIPTOR_HANDLE {
            // SAFETY: a live shader-visible heap.
            let mut at = unsafe { self.0.GetGPUDescriptorHandleForHeapStart() };
            at.ptr += u64::from(i * self.1);
            at
        }
    }

    /// Records a dispatch of `pso` over `groups` with the table at slot
    /// `table` as parameter 0 and, if given, root UAV `out` as parameter 1.
    fn run(
        list: &HostList,
        slots: &Slots,
        (root, pso): &(ID3D12RootSignature, ID3D12PipelineState),
        table: u32,
        out: Option<&ID3D12Resource>,
        groups: (u32, u32),
    ) {
        // SAFETY: an open list and live objects.
        unsafe {
            list.list.SetDescriptorHeaps(&[Some(slots.0.clone())]);
            list.list.SetComputeRootSignature(root);
            list.list.SetPipelineState(pso);
            list.list.SetComputeRootDescriptorTable(0, slots.gpu(table));
            if let Some(out) = out {
                list.list.SetComputeRootUnorderedAccessView(1, out.GetGPUVirtualAddress());
            }
            list.list.Dispatch(groups.0, groups.1, 1);
        }
    }

    const READ: &str = r#"
Texture2D<float4> t : register(t0);
RWByteAddressBuffer o : register(u1);
[numthreads(8, 8, 1)]
void main(uint3 id : SV_DispatchThreadID) {
    o.Store4((id.y * 64 + id.x) * 16, asuint(t.Load(int3(id.xy, 0))));
}
"#;
    /// The read shaders' root signature: a table of one shader view, and
    /// the output buffer in `u1`.
    const READS: &[Param] = &[Param::Table(&[srv(0, 1)]), Param::Uav { register: 1 }];
    /// The write shaders' root signature: a table of one unordered view.
    const WRITES: &[Param] = &[Param::Table(&[uav(0, 1)])];

    /// One texture memory written through an unordered view of a resource
    /// object in one format and read through a placed view in another: a
    /// castable pair shares one host texture (no copy), a pair of other
    /// shapes over the same bytes (block-compressed against plain) is two,
    /// bridged by a copy at the read.
    #[test]
    fn a_texture_written_in_one_format_reads_in_another_through_a_placed_view() {
        let Some(host) = crate::host::host() else {
            return; // no host GPU here
        };
        let slots = Slots::new(4);
        let read = compile_cs(READ, READS);
        let (out, data) = host.shared_buffer(64 * 64 * 16).unwrap();
        // Format written and size, what the shader writes, format read and
        // size, the texel expected, whether one host texture.
        type Case = (u32, u32, &'static str, u32, u32, [f32; 4], bool);
        let cases: [Case; 2] = [
            // R32_UINT holding R9G9B9E5 (1, 0.5, 0.25): the sky's tables.
            (
                42,
                64,
                "RWTexture2D<uint> o : register(u0); void put(uint2 p) { o[p] = 256u | 128u << 9 | 64u << 18 | 16u << 27; }",
                67,
                64,
                [1.0, 0.5, 0.25, 1.0],
                true,
            ),
            // R32G32_UINT blocks read as BC1: solid red blocks.
            (
                17,
                16,
                "RWTexture2D<uint2> o : register(u0); void put(uint2 p) { o[p] = uint2(0xF800, 0); }",
                71,
                64,
                [1.0, 0.0, 0.0, 1.0],
                false,
            ),
        ];
        for (round, (written, size, put, format, read_size, want, shared)) in cases.into_iter().enumerate() {
            let address = title_memory(1 << 20);
            let object = texture_at_address(address, &texture_words(size, size, written, 0x105, 4));
            let view = placed_view(address, texture_words(read_size, read_size, format, 0x105, 0));
            let Some(View::Shader(placed, _)) = crate::descriptor::host::placed_texture(&view) else {
                panic!("round {round}: a placed texture");
            };
            let a = crate::resource::host_copy(object).unwrap();
            let b = crate::resource::host_copy(placed as This).unwrap();
            assert_eq!(std::sync::Arc::ptr_eq(&a, &b), shared, "round {round}: one host texture");
            let unordered = View::Unordered(object as usize, 0, Some([written, 4, 0, 0, 0, 0, 0, 0, 0, 0]));
            slots.make(0, &unordered);
            slots.make(1, &view);
            let write = compile_cs(
                &format!("{put}\n[numthreads(8, 8, 1)]\nvoid main(uint3 id : SV_DispatchThreadID) {{ put(id.xy); }}"),
                WRITES,
            );
            let list = host.new_list().unwrap();
            let copies = crate::texture::ALIAS_COPIES.load(Ordering::Relaxed);
            readable(&list, &used(ResourceType::Uav), &unordered, false);
            run(&list, &slots, &write, 0, None, (size / 8, size / 8));
            readable(&list, &used(ResourceType::Srv), &view, false);
            run(&list, &slots, &read, 1, Some(&out), (read_size / 8, read_size / 8));
            list.close();
            host.execute(&[&list]);
            let copied = crate::texture::ALIAS_COPIES.load(Ordering::Relaxed) > copies;
            assert_eq!(copied, !shared, "round {round}: a copy only where not shared");
            let n = read_size as usize;
            for i in [0usize, 7, n * 9 + 33, n * n - 1] {
                let (x, y) = (i % n, i / n);
                // SAFETY: inside the mapped output.
                let got = unsafe { (data.add((y * 64 + x) * 16) as *const [f32; 4]).read() };
                assert_eq!(got, want, "round {round} texel {x},{y}");
            }
        }
    }

    /// Packed targets and their sampled aliases may have different host
    /// row pitches before promotion is accounted for (5120 vs 10240).
    #[test]
    fn a_packed_target_is_sampled_through_its_placed_view() {
        let Some(host) = crate::host::host() else {
            return;
        };
        let slots = Slots::new(2);
        let read = compile_cs(READ, READS);
        let write = compile_cs(
            r#"
RWTexture2D<float4> target : register(u0);
[numthreads(8, 8, 1)]
void main(uint3 id : SV_DispatchThreadID) { target[id.xy] = float4(1, 0, 1, 1); }
"#,
            WRITES,
        );
        let expected = [1.0f32, 0.0, 1.0, 1.0];
        let mut samples = Vec::new();
        for format in [86, 115] {
            for flags in [1, 4] {
                let address = title_memory(1 << 20);
                let view = placed_view(address, texture_words(2560, 8, format, 0x105, 0));
                if flags == 4 {
                    // A sampled view created before the target starts in its native format.
                    let Some(View::Shader(early, _)) =
                        crate::descriptor::host::placed_texture(&view)
                    else {
                        panic!("missing initial packed texture");
                    };
                    let native = crate::resource::host_resource(early as This).unwrap();
                    // SAFETY: a live texture.
                    assert_eq!(unsafe { native.GetDesc() }.Format.0 as u32, format);
                }
                let target =
                    texture_at_address(address, &texture_words(2560, 8, format, 0x105, flags));
                let gpu = crate::resource::host_resource(target).expect("host render target");
                let Some(View::Shader(placed, _)) = crate::descriptor::host::placed_texture(&view)
                else {
                    panic!("format {format}, flags {flags}: placed texture missing");
                };
                let sampled =
                    crate::resource::host_resource(placed as This).expect("host sample texture");
                if crate::resource::target_format(format) != format {
                    assert!(
                        crate::descriptor::host::make_view(
                            &View::Shader(placed, None),
                            host.cpu_view()
                        )
                        .is_some()
                    );
                    let other = if format == 86 { 115 } else { 86 };
                    let wrong = View::Shader(placed, Some([other, 4, 0x1688, 0, 0, 1, 0, 0, 0, 0]));
                    assert!(crate::descriptor::host::make_view(&wrong, host.cpu_view()).is_none());
                }
                slots.make(1, &view);
                let list = host.new_list().unwrap();
                if flags == 1 {
                    crate::list::ensure_state(&list, target, 0x4);
                    let rtv = host.rtv(&gpu, None);
                    // SAFETY: a live render target in the writable state.
                    unsafe { list.list.ClearRenderTargetView(rtv, &expected, None) };
                } else {
                    let uav = View::Unordered(
                        target as usize,
                        0,
                        Some([format, 4, 0, 0, 0, 0, 0, 0, 0, 0]),
                    );
                    slots.make(0, &uav);
                    readable(&list, &used(ResourceType::Uav), &uav, false);
                    run(&list, &slots, &write, 0, None, (1, 1));
                }
                let (out, data) = host.shared_buffer(64 * 8 * 16).unwrap();
                readable(&list, &used(ResourceType::Srv), &view, false);
                run(&list, &slots, &read, 1, Some(&out), (1, 1));
                list.close();
                host.execute(&[&list]);
                // SAFETY: execute waits for the single dispatch to fill the mapped output.
                let got = unsafe { data.add((3 * 64 + 5) * 16).cast::<[f32; 4]>().read() };
                // SAFETY: both resources are live.
                let (written_format, sampled_format) =
                    unsafe { (gpu.GetDesc().Format.0, sampled.GetDesc().Format.0) };
                samples.push((format, flags, written_format, sampled_format, got));
            }
        }
        eprintln!("packed target GPU samples {samples:?}");
        assert!(samples.iter().all(|s| s.4 == expected), "{samples:?}");
    }

    /// A texture the title wrote, read through tables by two lists of one
    /// submission, is uploaded once; the next submission uploads it again
    /// only after the title writes it (run perf7 uploaded the same textures
    /// up to 24 times a frame).
    #[test]
    fn a_texture_is_uploaded_once_per_submission_and_only_when_written() {
        let Some(host) = crate::host::host() else {
            return; // no host GPU here
        };
        let slots = Slots::new(2);
        let read = compile_cs(READ, READS);
        let (out, data) = host.shared_buffer(64 * 64 * 16).unwrap();
        // Runtime memory the change journal follows; RGBA8 64x64, linear.
        let address = crate::watch::tests::pool(1) as u64;
        let object = texture_at_address(address, &texture_words(64, 64, 28, 0x100, 0));
        let mut desc = [0u32; 10];
        desc[..6].copy_from_slice(&[28, 4, 0x1688, 0, 0, 1]);
        let view = View::Shader(object as usize, Some(desc));
        slots.make(0, &view);
        let submit = |byte: Option<u8>| {
            if let Some(byte) = byte {
                // SAFETY: the texture's memory, 256-byte rows of 64 texels.
                unsafe { std::ptr::write_bytes(address as *mut u8, byte, 64 * 64 * 4) };
            }
            let lists = [host.new_list().unwrap(), host.new_list().unwrap()];
            for list in &lists {
                readable(list, &used(ResourceType::Srv), &view, false);
                run(list, &slots, &read, 0, Some(&out), (8, 8));
                list.close();
            }
            let before = crate::texture::uploads(object);
            host.execute(&[&lists[0], &lists[1]]);
            // SAFETY: inside the mapped output.
            let texel = unsafe { (data as *const [f32; 4]).read() };
            (crate::texture::uploads(object) - before, texel[0])
        };
        let (n, x) = submit(Some(0x40));
        assert_eq!(n, 1, "one upload for two lists");
        assert!((x - 64.0 / 255.0).abs() < 1e-6);
        assert_eq!(submit(None).0, 0, "not written: not uploaded");
        let (n, x) = submit(Some(0x80));
        assert_eq!(n, 1, "written: uploaded again");
        assert!((x - 128.0 / 255.0).abs() < 1e-6);
        assert_eq!(submit(Some(0x80)).0, 0, "the same bytes in a shared page: not uploaded");
    }

    /// A title write into one page of a large mip detiles and copies the
    /// rows of that page alone: the rows written arrive, the other rows
    /// keep what the host texture held.
    #[test]
    fn a_write_into_one_page_uploads_its_rows_alone() {
        let Some(host) = crate::host::host() else {
            return; // no host GPU here
        };
        let slots = Slots::new(2);
        let read = compile_cs(READ, READS);
        let (out, data) = host.shared_buffer(64 * 64 * 16).unwrap();
        // RGBA8 512x512, one mip, linear, in memory the journal follows.
        let address = crate::watch::tests::pool(32) as u64;
        let object = texture_at_address(address, &texture_words(512, 512, 28, 0x100, 0));
        let (_, desc) = crate::resource::texture_memory(object).unwrap();
        let layout = crate::resource::texture_layout(&desc)
            .unwrap()
            .planes[0]
            .layout
            .clone();
        let row = |y: u32| layout.offset(0, 0, y, 0).unwrap() as usize;
        assert_eq!(row(32) % crate::watch::PAGE, 0, "rows 32.. start a page");
        assert!(
            row(64) - row(32) <= crate::watch::PAGE,
            "rows 32..64 within one page"
        );
        // SAFETY: the texture's memory (its layout's size).
        unsafe { std::ptr::write_bytes(address as *mut u8, 0x40, layout.size as usize) };
        let mut view = [0u32; 10];
        view[..6].copy_from_slice(&[28, 4, 0x1688, 0, 0, 1]);
        let view = View::Shader(object as usize, Some(view));
        slots.make(0, &view);
        let submit = || {
            let list = host.new_list().unwrap();
            readable(&list, &used(ResourceType::Srv), &view, false);
            run(&list, &slots, &read, 0, Some(&out), (8, 8));
            list.close();
            let before = crate::texture::copied(object);
            host.execute(&[&list]);
            // SAFETY: inside the mapped output, 64x64 texels of 16 bytes.
            let rows: Vec<f32> = (0..64)
                .map(|y| unsafe { (data as *const [f32; 4]).add(y * 64).read()[0] })
                .collect();
            (crate::texture::copied(object) - before, rows)
        };
        let is = |x: f32, b: u8| (x - f32::from(b) / 255.0).abs() < 1e-6;
        let (n, rows) = submit();
        assert_eq!(n, 1);
        assert!(rows.iter().all(|&x| is(x, 0x40)), "{rows:?}");
        let page = crate::watch::PAGE;
        // SAFETY: one page of the texture's memory, rows 32.. .
        unsafe { std::ptr::write_bytes((address as usize + row(32)) as *mut u8, 0x80, page) };
        let (n, rows) = submit();
        assert_eq!(n, 1);
        for (y, &x) in rows.iter().enumerate() {
            assert!(is(x, if y < 32 { 0x40 } else { 0x80 }), "row {y}: {x}");
        }
        // SAFETY: the first page, rows ..32.
        unsafe { std::ptr::write_bytes(address as *mut u8, 0xc0, page) };
        let (n, rows) = submit();
        assert_eq!(n, 1);
        for (y, &x) in rows.iter().enumerate() {
            assert!(is(x, if y < 32 { 0xc0 } else { 0x80 }), "row {y}: {x}");
        }
    }

    /// A list recorded before another whose host copy writes a texture it
    /// reads, but run after it, reads the host's texels: whether its upload
    /// from title memory is still due is decided when its submission is
    /// made, after the writing list's (`texture::host_writes`), not when it
    /// was recorded.
    #[test]
    fn a_list_run_after_a_host_write_keeps_the_host_texels() {
        let Some(host) = crate::host::host() else {
            return; // no host GPU here
        };
        let slots = Slots::new(2);
        let read = compile_cs(READ, READS);
        let (out, data) = host.shared_buffer(64 * 64 * 16).unwrap();
        // Runtime memory the change journal follows; RGBA8 64x64, linear.
        let address = crate::watch::tests::pool(1) as u64;
        let object = texture_at_address(address, &texture_words(64, 64, 28, 0x100, 0));
        let mut desc = [0u32; 10];
        desc[..6].copy_from_slice(&[28, 4, 0x1688, 0, 0, 1]);
        let view = View::Shader(object as usize, Some(desc));
        slots.make(0, &view);
        let reading = || {
            let list = host.new_list().unwrap();
            readable(&list, &used(ResourceType::Srv), &view, false);
            run(&list, &slots, &read, 0, Some(&out), (8, 8));
            list.close();
            list
        };
        // SAFETY: inside the mapped output.
        let texel = || unsafe { (data as *const [f32; 4]).read() }[0];
        // SAFETY: the texture's memory, 256-byte rows of 64 texels.
        let write = |byte: u8| unsafe { std::ptr::write_bytes(address as *mut u8, byte, 64 * 64 * 4) };
        write(0x40);
        host.execute(&[&reading()]);
        assert!((texel() - 64.0 / 255.0).abs() < 1e-6, "uploaded: {}", texel());
        // The title writes the memory again and a list reading the texture
        // is recorded: its upload is due.
        write(0x80);
        let late = reading();
        // A list recorded after it, holding a host copy into the texture
        // (its mark; what it copies is the host's business), runs first.
        let copying = host.new_list().unwrap();
        crate::texture::host_writes(&copying, object);
        copying.close();
        host.execute(&[&copying]);
        host.execute(&[&late]);
        assert!(
            (texel() - 64.0 / 255.0).abs() < 1e-6,
            "the host's texels kept, not title memory uploaded over them: {}",
            texel()
        );
    }

    /// A texture fed from title memory given up (`resource::evict`) while a
    /// recorded list names it: the list still reads it, and the next use
    /// makes it again with its bytes from title memory.
    #[test]
    fn a_texture_given_up_is_made_again_with_its_bytes_and_recorded_lists_keep_it() {
        let Some(host) = crate::host::host() else {
            return; // no host GPU here
        };
        let slots = Slots::new(2);
        let read = compile_cs(READ, READS);
        let (out, data) = host.shared_buffer(64 * 64 * 16).unwrap();
        let address = crate::watch::tests::pool(1) as u64;
        let object = texture_at_address(address, &texture_words(64, 64, 28, 0x100, 0));
        // SAFETY: the texture's memory, 256-byte rows of 64 texels.
        unsafe { std::ptr::write_bytes(address as *mut u8, 0x40, 64 * 64 * 4) };
        let mut desc = [0u32; 10];
        desc[..6].copy_from_slice(&[28, 4, 0x1688, 0, 0, 1]);
        let view = View::Shader(object as usize, Some(desc));
        let record = || {
            slots.make(0, &view);
            let list = host.new_list().unwrap();
            readable(&list, &used(ResourceType::Srv), &view, false);
            run(&list, &slots, &read, 0, Some(&out), (8, 8));
            list.close();
            list
        };
        // SAFETY: inside the mapped output.
        let texel = || unsafe { (data as *const [f32; 4]).read() }[0];
        let list = record();
        assert!(crate::resource::give_up_now(object), "fed from title memory");
        assert!(crate::resource::made_copy(object).is_none(), "its title texture let go");
        host.execute(&[&list]);
        assert!((texel() - 64.0 / 255.0).abs() < 1e-6, "the recorded list read it");
        // SAFETY: inside the mapped output.
        unsafe { std::ptr::write_bytes(data, 0, 16) };
        let list = record();
        host.execute(&[&list]);
        assert!((texel() - 64.0 / 255.0).abs() < 1e-6, "made again with its bytes");
        assert_eq!(crate::texture::uploads(object), 1, "uploaded into the new one");
    }

    /// A dynamic texture descriptor a recorded list names stays made through
    /// an eviction pass: unlike a title slot, it has no descriptor record
    /// from which to make it again.
    #[test]
    fn a_recorded_list_keeps_its_dynamic_texture_descriptor() {
        let Some(host) = crate::host::host() else {
            return; // no host GPU here
        };
        let address = crate::watch::tests::pool(1) as u64;
        let object = texture_at_address(address, &texture_words(64, 64, 42, 0x100, 4));
        let view = View::Unordered(object as usize, 0, Some([42, 4, 0, 0, 0, 0, 0, 0, 0, 0]));

        let null = std::ptr::null();
        let heap_desc: [u32; 8] = [0, 8, 1, 0, 0, 0, 0, 0];
        let mut heap = null;
        let iid = crate::object::guid(0x8efb471d, 0x616c, 0x4f49, [0x90, 0xf7, 0x12, 0x7b, 0xb7, 0x63, 0xfa, 0x51]);
        assert_eq!(crate::descriptor::create_heap(null, heap_desc.as_ptr().cast(), &iid, &mut heap), S_OK);
        let slots = crate::descriptor::heap(heap).host().expect("a host view heap");
        let dynamic = slots.dynamic(1).expect("a dynamic slot");
        assert!(dynamic >= slots.count(), "past the title's recorded slots");
        slots.make_at(dynamic, &view);
        assert!(slots.kept(dynamic), "made for the list");
        let texture = crate::resource::made_copy(object).expect("its host texture");
        let raw = windows::core::Interface::as_raw(&texture.texture) as usize;

        // A clear through the dynamic descriptor, then a copy of what it left.
        let mut footprint = D3D12_PLACED_SUBRESOURCE_FOOTPRINT::default();
        let mut total = 0;
        // SAFETY: outputs for the texture's one subresource.
        unsafe {
            let desc = texture.texture.GetDesc();
            host.device.GetCopyableFootprints(&desc, 0, 1, 0, Some(&mut footprint), None, None, Some(&mut total));
        }
        let (readback, data) = host.shared_buffer(total).unwrap();
        let list = host.new_list().unwrap();
        readable(&list, &used(ResourceType::Uav), &view, false);
        let at = host.cpu_view();
        let resource = crate::descriptor::host::make_view(&view, at).expect("a matching CPU descriptor");
        let values = [0x1122_3344, 0x5566_7788, 0x99aa_bbcc, 0xddee_ff00];
        // SAFETY: an open list and live objects; the clear names the dynamic slot.
        unsafe {
            list.list.SetDescriptorHeaps(&[Some(slots.heap.clone())]);
            list.list.ClearUnorderedAccessViewUint(slots.gpu(dynamic), at, &resource, &values, &[]);
        }
        crate::list::ensure_state(&list, object, 0x800);
        // SAFETY: borrowed resources; the copy is recorded before they are released.
        unsafe {
            let to = D3D12_TEXTURE_COPY_LOCATION {
                pResource: std::mem::transmute_copy(&readback),
                Type: D3D12_TEXTURE_COPY_TYPE_PLACED_FOOTPRINT,
                Anonymous: D3D12_TEXTURE_COPY_LOCATION_0 { PlacedFootprint: footprint },
            };
            let from = D3D12_TEXTURE_COPY_LOCATION {
                pResource: std::mem::transmute_copy(&texture.texture),
                Type: D3D12_TEXTURE_COPY_TYPE_SUBRESOURCE_INDEX,
                Anonymous: D3D12_TEXTURE_COPY_LOCATION_0 { SubresourceIndex: 0 },
            };
            list.list.CopyTextureRegion(&to, 0, 0, 0, &from, None);
        }
        list.close();

        // Force the eviction pass directly: an unordered texture is not a
        // normal candidate, but the dynamic-slot rule applies to every view.
        crate::descriptor::evict(&std::collections::HashSet::from([raw]));
        assert!(slots.kept(dynamic), "the recorded list's descriptor is retained");
        assert_eq!(slots.textures_at(&[dynamic]), [raw], "still views the held texture");

        host.execute(&[&list]);
        // SAFETY: inside the mapped readback, at the first and last texels.
        unsafe {
            let word = |x: u32, y: u32| {
                (data.add(y as usize * footprint.Footprint.RowPitch as usize + x as usize * 4) as *const u32).read()
            };
            assert_eq!(word(0, 0), values[0], "the recorded clear used the descriptor");
            assert_eq!(word(63, 63), values[0], "the whole texture was cleared");
        }

        slots.make_null(dynamic);
        slots.release(slots.id(), dynamic, 1);
        drop(list);
        assert_eq!(
            crate::object::slot::<extern "system" fn(crate::object::This) -> u32>(heap, 0x10)(heap),
            0
        );
        assert_eq!(
            crate::object::slot::<extern "system" fn(crate::object::This) -> u32>(object, 0x10)(object),
            0
        );
        drop(texture);
        crate::watch::tests::free(address as usize);
    }

    /// Textures brought up in one submission share staging space (parts
    /// of one upload buffer): each reads its own bytes.
    #[test]
    fn textures_uploaded_together_each_read_their_own_bytes() {
        let Some(host) = crate::host::host() else {
            return; // no host GPU here
        };
        let slots = Slots::new(2);
        let read = compile_cs(READ, READS);
        let outs: Vec<_> = (0..2).map(|_| host.shared_buffer(64 * 64 * 16).unwrap()).collect();
        let pool = crate::watch::tests::pool(2);
        let views: Vec<View> = (0..2usize)
            .map(|k| {
                let address = (pool + k * crate::watch::PAGE) as u64;
                // SAFETY: the texture's memory in our pool, 256-byte rows of 64 texels.
                unsafe { std::ptr::write_bytes(address as *mut u8, 0x40 * (k as u8 + 1), 64 * 64 * 4) };
                let object = texture_at_address(address, &texture_words(64, 64, 28, 0x100, 0));
                let mut desc = [0u32; 10];
                desc[..6].copy_from_slice(&[28, 4, 0x1688, 0, 0, 1]);
                View::Shader(object as usize, Some(desc))
            })
            .collect();
        let lists: Vec<_> = (0..2usize)
            .map(|k| {
                slots.make(k as u32, &views[k]);
                let list = host.new_list().unwrap();
                readable(&list, &used(ResourceType::Srv), &views[k], false);
                run(&list, &slots, &read, k as u32, Some(&outs[k].0), (8, 8));
                list.close();
                list
            })
            .collect();
        host.execute(&[&lists[0], &lists[1]]);
        for (k, (_, data)) in outs.iter().enumerate() {
            // SAFETY: inside the mapped output.
            let texel = unsafe { (*data as *const [f32; 4]).read() };
            assert!((texel[0] - (0x40 * (k + 1)) as f32 / 255.0).abs() < 1e-6, "texture {k}: {}", texel[0]);
        }
    }

    /// A texture written in a page that holds only its own bytes is uploaded
    /// without hashing them (a movie frame rewritten every frame), so even
    /// the same bytes again are uploaded.
    #[test]
    fn a_texture_owning_its_pages_is_uploaded_on_any_write() {
        let Some(host) = crate::host::host() else {
            return; // no host GPU here
        };
        let slots = Slots::new(2);
        let read = compile_cs(READ, READS);
        let (out, data) = host.shared_buffer(64 * 64 * 16).unwrap();
        // RGBA8 128x128, linear: one whole 64 KiB page.
        let address = crate::watch::tests::pool(1) as u64;
        let object = texture_at_address(address, &texture_words(128, 128, 28, 0x100, 0));
        let mut desc = [0u32; 10];
        desc[..6].copy_from_slice(&[28, 4, 0x1688, 0, 0, 1]);
        let view = View::Shader(object as usize, Some(desc));
        slots.make(0, &view);
        let submit = |byte: u8| {
            // SAFETY: the texture's memory, one page.
            unsafe { std::ptr::write_bytes(address as *mut u8, byte, crate::watch::PAGE) };
            let list = host.new_list().unwrap();
            readable(&list, &used(ResourceType::Srv), &view, false);
            run(&list, &slots, &read, 0, Some(&out), (8, 8));
            list.close();
            let before = crate::texture::uploads(object);
            host.execute(&[&list]);
            // SAFETY: inside the mapped output.
            let texel = unsafe { (data as *const [f32; 4]).read() };
            (crate::texture::uploads(object) - before, texel[0])
        };
        assert_eq!(submit(0x40).0, 1);
        assert_eq!(submit(0x40).0, 1, "its own page written: uploaded");
        let (n, x) = submit(0x80);
        assert_eq!(n, 1);
        assert!((x - 128.0 / 255.0).abs() < 1e-6, "a reused staging buffer carries the new bytes: {x}");
    }

    /// A title write into one mip of a texture uploads that mip alone (the
    /// pages of the others did not change), and the others keep their
    /// contents.
    #[test]
    fn a_write_into_one_mip_uploads_that_mip_alone() {
        let Some(host) = crate::host::host() else {
            return; // no host GPU here
        };
        let slots = Slots::new(2);
        let read = compile_cs(READ, READS);
        let (out, data) = host.shared_buffer(64 * 64 * 16).unwrap();
        // RGBA8 512x512 with two mips, linear, in memory the journal follows.
        let address = crate::watch::tests::pool(32) as u64;
        let mut words = texture_words(512, 512, 28, 0x100, 0);
        words[7] = 1 | 2 << 16;
        let object = texture_at_address(address, &words);
        let (_, desc) = crate::resource::texture_memory(object).unwrap();
        let layout = crate::resource::texture_layout(&desc).unwrap().planes[0].layout.clone();
        let mip1 = layout.mips[1].offset as usize;
        // SAFETY: the texture's memory (its layout's size).
        unsafe { std::ptr::write_bytes(address as *mut u8, 0x40, layout.size as usize) };
        let mut view = [0u32; 10];
        view[..6].copy_from_slice(&[28, 4, 0x1688, 0, 0, 2]);
        let view = View::Shader(object as usize, Some(view));
        slots.make(0, &view);
        let submit = || {
            let list = host.new_list().unwrap();
            readable(&list, &used(ResourceType::Srv), &view, false);
            run(&list, &slots, &read, 0, Some(&out), (8, 8));
            list.close();
            let before = crate::texture::copied(object);
            host.execute(&[&list]);
            // SAFETY: inside the mapped output.
            let texel = unsafe { (data as *const [f32; 4]).read() };
            (crate::texture::copied(object) - before, texel[0])
        };
        assert_eq!(submit().0, 2, "both mips at first");
        // SAFETY: mip 1's bytes (256x256 texels of 4 bytes).
        unsafe { std::ptr::write_bytes((address as usize + mip1) as *mut u8, 0x80, 256 * 256 * 4) };
        let (n, x) = submit();
        assert_eq!(n, 1, "mip 1 alone");
        assert!((x - 64.0 / 255.0).abs() < 1e-6, "mip 0 kept: {x}");
    }

    /// A depth texture with stencil (the console's 16-bit depth, the
    /// host's 32-bit float) written through unordered views of its planes,
    /// then read as depth and stencil: the 16-bit depth arrives as the
    /// float the title's shaders compute for it, the stencil as written,
    /// and a draw at that depth passes an EQUAL test. Rendered as depth (a
    /// clear), then used through the unordered view: back in 16 bits.
    #[test]
    fn a_depth_texture_written_through_unordered_views_reads_as_depth() {
        let Some(host) = crate::host::host() else {
            return; // no host GPU here
        };
        let slots = Slots::new(4);
        let address = title_memory(1 << 22);
        let depth = texture_at_address(address, &texture_words(64, 64, 118, 0, 2));
        let copy = crate::resource::host_copy(depth).expect("a host depth texture");
        let planes = [
            View::Unordered(depth as usize, 0, Some([56, 4, 0, 0, 0, 0, 0, 0, 0, 0])),
            View::Unordered(depth as usize, 0, Some([62, 4, 0, 1, 0, 0, 0, 0, 0, 0])),
        ];
        slots.make(0, &planes[0]);
        slots.make(1, &planes[1]);
        let reads = [
            View::Shader(depth as usize, Some([119, 4, 0x1688, 0, 0, 1, 0, 0, 0, 0])),
            View::Shader(depth as usize, Some([120, 4, 0x1688, 0, 0, 1, 1, 0, 0, 0])),
        ];
        slots.make(2, &reads[0]);
        slots.make(3, &reads[1]);
        let write = compile_cs(
            r#"
RWTexture2D<float> d : register(u0);
RWTexture2D<uint> s : register(u1);
[numthreads(8, 8, 1)]
void main(uint3 id : SV_DispatchThreadID) {
    d[id.xy] = (1 + id.x + 64 * id.y) * (1.0 / 65535.0);
    s[id.xy] = (id.x ^ id.y) & 0xff;
}
"#,
            &[Param::Table(&[uav(0, 2)])],
        );
        let read = compile_cs(
            r#"
Texture2D<float> d : register(t0);
Texture2D<uint2> s : register(t1);
RWByteAddressBuffer o : register(u1);
[numthreads(8, 8, 1)]
void main(uint3 id : SV_DispatchThreadID) {
    o.Store2((id.y * 64 + id.x) * 8, uint2(asuint(d.Load(int3(id.xy, 0))), s.Load(int3(id.xy, 0)).g));
}
"#,
            &[Param::Table(&[srv(0, 2)]), Param::Uav { register: 1 }],
        );
        let (out, data) = host.shared_buffer(64 * 64 * 8).unwrap();
        // The depth texture's planes as bytes, read back after.
        // SAFETY: a live resource.
        let dd = unsafe { copy.texture.GetDesc() };
        let mut f = [D3D12_PLACED_SUBRESOURCE_FOOTPRINT::default(); 2];
        let mut total = 0;
        // SAFETY: outputs for two subresources.
        unsafe { host.device.GetCopyableFootprints(&dd, 0, 2, 0, Some(f.as_mut_ptr()), None, None, Some(&mut total)) };
        let (raw, raw_data) = host.shared_buffer(total).unwrap();
        let list = host.new_list().unwrap();
        for v in &planes {
            readable(&list, &used(ResourceType::Uav), v, false);
        }
        run(&list, &slots, &write, 0, None, (8, 8));
        for v in &reads {
            readable(&list, &used(ResourceType::Srv), v, false);
        }
        assert!(list.take_rebind(), "the conversion ran in the list");
        run(&list, &slots, &read, 2, Some(&out), (8, 8));
        crate::list::ensure_state(&list, depth, 0x800);
        let at = |r: &ID3D12Resource, sub: u32| D3D12_TEXTURE_COPY_LOCATION {
            // SAFETY: a borrowed pointer the copy does not release.
            pResource: unsafe { std::mem::transmute_copy(r) },
            Type: D3D12_TEXTURE_COPY_TYPE_SUBRESOURCE_INDEX,
            Anonymous: D3D12_TEXTURE_COPY_LOCATION_0 { SubresourceIndex: sub },
        };
        let to = |fp: D3D12_PLACED_SUBRESOURCE_FOOTPRINT| D3D12_TEXTURE_COPY_LOCATION {
            // SAFETY: as above.
            pResource: unsafe { std::mem::transmute_copy(&raw) },
            Type: D3D12_TEXTURE_COPY_TYPE_PLACED_FOOTPRINT,
            Anonymous: D3D12_TEXTURE_COPY_LOCATION_0 { PlacedFootprint: fp },
        };
        // SAFETY: an open list, the texture in the copy source state.
        unsafe {
            list.list.CopyTextureRegion(&to(f[0]), 0, 0, 0, &at(&copy.texture, 0), None);
            list.list.CopyTextureRegion(&to(f[1]), 0, 0, 0, &at(&copy.texture, 1), None);
        }
        list.close();
        host.execute(&[&list]);
        // The title's float for 16-bit value u (its 1/65535 is 0x37800080).
        let float = |u: u32| (u as f32 * f32::from_bits(0x3780_0080)).to_bits();
        for (x, y) in [(0usize, 0usize), (5, 0), (63, 2), (17, 40), (63, 63)] {
            let k = (1 + x + 64 * y) as u32;
            // SAFETY: inside the mapped buffers.
            let [depth_bits, stencil] = unsafe { (data.add((y * 64 + x) * 8) as *const [u32; 2]).read() };
            let got = f32::from_bits(depth_bits);
            assert!((got - k as f32 / 65535.0).abs() < 1e-7, "depth at {x},{y}: {got}");
            assert_eq!(stencil, ((x ^ y) & 0xff) as u32, "stencil at {x},{y}");
            // SAFETY: inside the mapped buffer, by the footprints.
            let (word, byte) = unsafe {
                (
                    (raw_data.add(f[0].Offset as usize + y * f[0].Footprint.RowPitch as usize + x * 4) as *const u32).read(),
                    raw_data.add(f[1].Offset as usize + y * f[1].Footprint.RowPitch as usize + x).read(),
                )
            };
            assert_eq!(word, float(k), "depth at {x},{y}");
            assert_eq!(u32::from(byte), ((x ^ y) & 0xff) as u32);
        }
        // Tested EQUAL against a point at the depth the title's tiled
        // lighting gives value k (k times its 1/65535), as the console's
        // 16-bit depth passes it for every k. In 24-bit depth on the RX 6700
        // XT only k = 258 of 1..=4096 passed.
        let passed = equal_test_passes(depth, &copy.texture);
        let failed: Vec<u32> = (1..=4096).filter(|&k| !passed[k as usize - 1]).collect();
        assert!(failed.is_empty(), "EQUAL fails at {failed:?}");
        // Rendered as depth (a clear of both planes), then used through
        // the unordered view: copied to the twin, converted back. The clear
        // is 0.75 (12582911.25 in 24 bits), not on a rounding tie: D3DMetal
        // rounds float depth down to 24 bits, so a tie such as 0.5
        // (8388607.5) stores 8388607 there and 8388608 on Windows GPUs, and
        // converts to 32767 or 32768.
        list.reset();
        crate::list::ensure_state(&list, depth, 0x10);
        let dsv = D3D12_DEPTH_STENCIL_VIEW_DESC {
            Format: windows::Win32::Graphics::Dxgi::Common::DXGI_FORMAT_D32_FLOAT_S8X24_UINT,
            ViewDimension: D3D12_DSV_DIMENSION_TEXTURE2D,
            ..Default::default()
        };
        let handle = host.dsv(&copy.texture, Some(&dsv));
        // SAFETY: an open list, the texture in the depth write state.
        unsafe {
            list.list.ClearDepthStencilView(handle, D3D12_CLEAR_FLAG_DEPTH | D3D12_CLEAR_FLAG_STENCIL, 0.75, 0x11, None)
        };
        readable(&list, &used(ResourceType::Uav), &planes[0], false);
        let twin = copy.twin.get().and_then(Option::as_ref).expect("a twin");
        let mut g = [D3D12_PLACED_SUBRESOURCE_FOOTPRINT::default(); 2];
        for (p, plane) in twin.planes.iter().enumerate() {
            // SAFETY: a live resource.
            let pd = unsafe { plane.texture.GetDesc() };
            // SAFETY: an output for one subresource.
            unsafe { host.device.GetCopyableFootprints(&pd, 0, 1, f[p].Offset, Some(&mut g[p]), None, None, None) };
            if let Some(b) = list.move_texture(plane, 0x800) {
                // SAFETY: an open list.
                unsafe { list.list.ResourceBarrier(&[b]) };
            }
        }
        // SAFETY: an open list, the twin's planes in the copy source state.
        unsafe {
            list.list.CopyTextureRegion(&to(g[0]), 0, 0, 0, &at(&twin.planes[0].texture, 0), None);
            list.list.CopyTextureRegion(&to(g[1]), 0, 0, 0, &at(&twin.planes[1].texture, 0), None);
        }
        list.close();
        host.execute(&[&list]);
        for (x, y) in [(0usize, 0usize), (63, 63)] {
            // SAFETY: inside the mapped buffer, by the footprints.
            let (half, byte) = unsafe {
                (
                    (raw_data.add(g[0].Offset as usize + y * g[0].Footprint.RowPitch as usize + x * 2) as *const u16).read(),
                    raw_data.add(g[1].Offset as usize + y * g[1].Footprint.RowPitch as usize + x).read(),
                )
            };
            assert_eq!(half, 49151, "16-bit depth at {x},{y}");
            assert_eq!(byte, 0x11, "stencil at {x},{y}");
        }
    }

    #[test]
    fn compute_view_marks_rebind_after_depth_twin_conversion() {
        use victory_shader::emit::{Options, emit_with};
        use victory_shader::native::{ComputeIo, HwStage, NativeProgram, RootParam, StageIo};
        use victory_shader::{Container, io, ir, resource, structure};

        let Some(host) = crate::host::host() else {
            return;
        };
        let _ = host.log_messages();

        // Synthetic Wave32 compute shader: loads texel (0, 0) of a 2D depth SRV
        // at table+0 (`s[8:15]`) and stores the float bits to byte offset
        // `0x1_0010` (page 1) of a raw buffer UAV at table+0x20 (`s[16:19]`).
        let code: &[u32] = &[
            0xBE84_0300, // s_mov_b32 s4, s0
            0xBE85_0380, // s_mov_b32 s5, 0
            0xF40C_0202,
            0xFA00_0000, // s_load_dwordx8 s[8:15], s[4:5], 0x0
            0xF408_0402,
            0xFA00_0020, // s_load_dwordx4 s[16:19], s[4:5], 0x20
            0x7E00_0280, // v_mov_b32 v0, 0
            0x7E02_0280, // v_mov_b32 v1, 0
            0xBF8C_C07F, // s_waitcnt lgkmcnt(0)
            0xF000_0108,
            0x0002_0200, // image_load v2, v[0:1], s[8:15] dmask:0x1 dim:2D
            0xBF8C_3F70, // s_waitcnt vmcnt(0)
            0x7E06_02FF,
            0x0001_0010, // v_mov_b32 v3, 0x10010
            0xE070_1000,
            0x8004_0203, // buffer_store_dword v2, v3, s[16:19], 0 offen
            0xBF81_0000, // s_endpgm
        ];
        let used_entries = vec![
            UsedResource {
                kind: ResourceType::Srv,
                root: RootType::Table,
                table_offset_dwords: 0,
                lower_bound: 0,
                stride: 0,
                dwords: 8,
                user_data: 0,
                slot: 0,
            },
            UsedResource {
                kind: ResourceType::Uav,
                root: RootType::Table,
                table_offset_dwords: 8,
                lower_bound: 0,
                stride: 0,
                dwords: 4,
                user_data: 0,
                slot: 0,
            },
        ];
        let native = NativeProgram {
            stage: HwStage::Cs,
            wave64: false,
            vgprs: 8,
            layout: vec![RootParam {
                root: RootType::Table,
                table_offset_dwords: 0,
                sgprs: 1,
                base_register: 0,
                user_data: 0,
                slot: 0,
            }],
            used: used_entries.clone(),
            io: StageIo::Cs(ComputeIo {
                rsrc2: 3 << 1,
                threads: [1, 1, 1],
            }),
            code: code.iter().flat_map(|w| w.to_le_bytes()).collect(),
        };
        let p = ir::build(&native.code, native.wave64, native.vgprs).unwrap();
        let st = structure::structure(&p).unwrap();
        let res = resource::track(&p, &native.layout, native.stage, Some(&st)).unwrap();
        let io = io::raise(&p, &native).unwrap();
        let host_shader = emit_with(
            &Container { parts: Vec::new() },
            &native,
            &p,
            &st,
            &io,
            &res,
            Options {
                view_marks: true,
                uav_size_level0: false,
                ..Options::default()
            },
        )
        .unwrap();
        assert!(host_shader.view_marks);
        let shader = crate::pipeline::Shader::for_test(
            "compute",
            crate::pipeline::Translated {
                host: host_shader,
                used: used_entries,
                used_written: Vec::new(),
                write_unknown: true,
                threads: None,
            },
        );
        let hp = HostPipeline::new(&[shader], None, |_, _| {}).expect("compute host pipeline");
        assert!(hp.marks.is_some());

        // 16-bit console depth-stencil texture (`format = 118`) whose twin
        // requires the compute `converter()` pass when transitioning to SRV.
        let depth_mem = title_memory(1 << 22);
        let depth = texture_at_address(depth_mem, &texture_words(64, 64, 118, 0, 2));

        // 2-page tracked guest buffer (`marks.is_some()`): UAV writes go back
        // only through `VIEW_MARKS` / `VIEW_WRITES` (`copy_back_marks`).
        let pool_ptr = crate::watch::tests::pool(2);
        let buf_words: [u32; 14] = [1, 0, 0, 0, 2 << 16, 0, 1, 1 | 1 << 16, 0, 1, 0, 1, 4, 0];
        let dst_buf = texture_at_address(pool_ptr as u64, &buf_words);
        let dst_shadow =
            crate::shadow::find(pool_ptr as u64, 2 << 16).expect("tracked shadow buffer");
        assert!(dst_shadow.marked());

        let null = std::ptr::null();
        let heap_iid = crate::object::guid(
            0x8efb471d,
            0x616c,
            0x4f49,
            [0x90, 0xf7, 0x12, 0x7b, 0xb7, 0x63, 0xfa, 0x51],
        );
        let heap_desc: [u32; 8] = [0, 8, 1, 0, 0, 0, 0, 0];
        let mut heap = null;
        assert_eq!(
            crate::descriptor::create_heap(null, heap_desc.as_ptr().cast(), &heap_iid, &mut heap),
            S_OK
        );
        let title_heap = crate::descriptor::heap(heap);
        let depth_srv: [u32; 10] = [119, 4, 0x1688, 0, 0, 1, 0, 0, 0, 0];
        let raw_uav: [u32; 10] = [39, 1, 0, 0, 32768, 0, 0, 0, 1, 0];
        crate::descriptor::create_srv(null, depth, &depth_srv, title_heap.start);
        crate::descriptor::create_uav(null, dst_buf, null, &raw_uav, title_heap.start + 32);

        // Step 1: Write `49152 / 65535.0` into the 16-bit depth texture's
        // writable twin UAV in the same command list.
        let twin_slots = Slots::new(1);
        let twin_uav = View::Unordered(depth as usize, 0, Some([56, 4, 0, 0, 0, 0, 0, 0, 0, 0]));
        twin_slots.make(0, &twin_uav);
        let write_twin = compile_cs(
            r#"
RWTexture2D<float> d : register(u0);
[numthreads(8, 8, 1)]
void main(uint3 id : SV_DispatchThreadID) {
    d[id.xy] = 49152.0 / 65535.0;
}
"#,
            WRITES,
        );

        let list = host.new_list().unwrap();
        let copies_before = crate::twin::COPIES.load(Ordering::Relaxed);
        readable(&list, &used(ResourceType::Uav), &twin_uav, false);
        run(&list, &twin_slots, &write_twin, 0, None, (8, 8));

        // Step 2: `State::bind` binds `hp.marks`, then `self.tables` makes
        // `depth` readable as SRV — running `Twin::copy`'s compute converter
        // and triggering `take_rebind()`, which must restore `hp.marks`.
        let mut state = State::default();
        state.set_heaps(&[heap]);
        state.set_table(true, 0, title_heap.start);
        assert!(state.bind(&list, &hp, true));
        assert!(
            crate::twin::COPIES.load(Ordering::Relaxed) > copies_before,
            "depth twin conversion ran during State::bind"
        );
        // SAFETY: open command list with compute pipeline and roots bound.
        unsafe { list.list.Dispatch(1, 1, 1) };
        list.close();
        host.execute(&[&list]);

        assert_eq!(
            host.log_messages(),
            None,
            "D3D12 debug layer error after depth-twin compute rebind"
        );
        // SAFETY: plain device health query.
        assert!(unsafe { host.device.GetDeviceRemovedReason() }.is_ok());

        // SAFETY: `pool_ptr` holds 2 pages of live tracked title memory.
        let bits = unsafe { ((pool_ptr + 0x1_0010) as *const u32).read_volatile() };
        let got = f32::from_bits(bits);
        assert!(
            (got - 49152.0 / 65535.0).abs() < 1e-6,
            "marked page 1 copied back converted 16-bit depth: {got} ({bits:#x})"
        );

        type Release = extern "system" fn(This) -> u32;
        let _ = crate::object::slot::<Release>(heap, 0x10)(heap);
        let _ = crate::object::slot::<Release>(dst_buf, 0x10)(dst_buf);
        let _ = crate::object::slot::<Release>(depth, 0x10)(depth);
    }


    /// Unordered-access-only 1D and 3D title textures (the colour grading
    /// bakes: a 4096 R32_FLOAT curve, a 64^3 RGBA16F table) keep what a
    /// dispatch writes through their unordered views, whole (a 3D view of
    /// every depth slice, -1), and a shader view reads it.
    #[test]
    fn unordered_1d_and_3d_textures_keep_their_writes() {
        let Some(host) = crate::host::host() else {
            return; // no host GPU here
        };
        let slots = Slots::new(2);
        let (out, data) = host.shared_buffer(64 * 64 * 16).unwrap();
        // Description words (dimension, width, height, depth | mips << 16,
        // format), the unordered view's and shader view's words, the
        // shaders' bodies, groups, a texel to check and its value.
        type Case = ([u32; 5], [u32; 10], [u32; 10], &'static str, &'static str, (u32, u32), [f32; 4]);
        let cases: [Case; 2] = [
            (
                [2, 4096, 1, 1 | 1 << 16, 41],
                [41, 2, 0, 0, 0, 0, 0, 0, 0, 0],
                [41, 2, 0x1688, 0, 0, 1, 0, 0, 0, 0],
                "RWTexture1D<float> o : register(u0); void put(uint3 p) { if (p.y == 0 && p.z == 0) o[p.x] = p.x * 0.25; }",
                "Texture1D<float4> t : register(t0); float4 get(uint3 p) { return t.Load(int2(p.y * 64 + p.x, 0)); }",
                (512, 1),
                [(2 * 64 + 7) as f32 * 0.25, 0.0, 0.0, 1.0],
            ),
            (
                [4, 64, 64, 64 | 1 << 16, 10],
                [10, 8, 0, 0, u32::MAX, 0, 0, 0, 0, 0],
                [10, 8, 0x1688, 0, 0, 1, 0, 0, 0, 0],
                "RWTexture3D<float4> o : register(u0); void put(uint3 p) { o[p] = float4(p, 1) / 64; }",
                "Texture3D<float4> t : register(t0); float4 get(uint3 p) { return t.Load(int4(p.x, p.y, 40, 0)); }",
                (8, 8),
                [7.0 / 64.0, 2.0 / 64.0, 40.0 / 64.0, 1.0 / 64.0],
            ),
        ];
        for (round, (d, uav, srv, put, get, groups, want)) in cases.into_iter().enumerate() {
            let address = title_memory(1 << 22);
            let words = [d[0], 0, 0, 0, d[1], 0, d[2], d[3], d[4], 1, 0, 0, 4, 0];
            let texture = texture_at_address(address, &words);
            let unordered = View::Unordered(texture as usize, 0, Some(uav));
            let shader = View::Shader(texture as usize, Some(srv));
            slots.make(0, &unordered);
            slots.make(1, &shader);
            let write = compile_cs(
                &format!("{put}\n[numthreads(8, 8, 1)]\nvoid main(uint3 id : SV_DispatchThreadID) {{ for (uint z = 0; z < 64; z++) put(uint3(id.xy, z)); }}"),
                WRITES,
            );
            let read = compile_cs(
                &format!("{get}\nRWByteAddressBuffer o : register(u1);\n[numthreads(8, 8, 1)]\nvoid main(uint3 id : SV_DispatchThreadID) {{ o.Store4((id.y * 64 + id.x) * 16, asuint(get(id))); }}"),
                READS,
            );
            let list = host.new_list().unwrap();
            readable(&list, &used(ResourceType::Uav), &unordered, false);
            run(&list, &slots, &write, 0, None, groups);
            readable(&list, &used(ResourceType::Srv), &shader, false);
            run(&list, &slots, &read, 1, Some(&out), (8, 8));
            list.close();
            host.execute(&[&list]);
            // SAFETY: inside the mapped output: texel (7, 2) of the read.
            let got = unsafe { (data.add((2 * 64 + 7) * 16) as *const [f32; 4]).read() };
            assert_eq!(got, want, "round {round}");
        }
    }

    #[path = "r16_volume_fixture.rs"]
    mod r16_volume_fixture;

    /// A texture written through an unordered view and copied into a title
    /// buffer (a read back) reaches the buffer's title memory once the
    /// submission has run (#50: the title waited on such a copy).
    #[test]
    fn a_texture_copied_to_a_buffer_reaches_title_memory() {
        let Some(host) = crate::host::host() else {
            return; // no host GPU here
        };
        let slots = Slots::new(1);
        let address = title_memory(1 << 22);
        let words = [3, 0, 0, 0, 64, 0, 64, 1 | 1 << 16, 41, 1, 0, 0, 4, 0];
        let texture = texture_at_address(address, &words);
        let unordered = View::Unordered(texture as usize, 0, Some([41, 4, 0, 0, 0, 0, 0, 0, 0, 0]));
        slots.make(0, &unordered);
        let write = compile_cs(
            "RWTexture2D<float> o : register(u0);\n[numthreads(8, 8, 1)]\nvoid main(uint3 id : SV_DispatchThreadID) { o[id.xy] = id.x + id.y * 64; }",
            WRITES,
        );
        let dst = buffer(256 * 64);
        let list = host.new_list().unwrap();
        readable(&list, &used(ResourceType::Uav), &unordered, false);
        run(&list, &slots, &write, 0, None, (8, 8));
        let footprint = crate::resource::PlacedFootprint::new(0, 41, 64, 64, 256);
        let to = crate::list::CopyLocation::placed(dst, footprint);
        let from = crate::list::CopyLocation::whole(texture);
        crate::list::read_back(&list, &to, [0, 0, 0], &from, std::ptr::null());
        list.close();
        host.execute(&[&list]);
        let (bytes, _) = crate::resource::buffer_bytes(dst).unwrap();
        // SAFETY: inside the buffer: texel (7, 2) at a 256-byte pitch.
        let got = unsafe { (bytes.add(2 * 256 + 7 * 4) as *const f32).read() };
        assert_eq!(got, (7 + 2 * 64) as f32);
    }

    /// A structured view the title places off its buffer's elements (the
    /// culling constants at a ring position) is made over a copy of its own
    /// bytes; one on an element views the buffer's copy.
    #[test]
    fn structured_views_off_their_buffers_elements_get_their_own_copy() {
        let Some(host) = crate::host::host() else {
            return; // no host GPU here
        };
        let src = buffer(1 << 16);
        let (bytes, _) = crate::resource::buffer_bytes(src).unwrap();
        let start = bytes as u64;
        // SAFETY: a plain CPU-side heap of one view.
        let cpu: ID3D12DescriptorHeap = unsafe {
            host.device.CreateDescriptorHeap(&D3D12_DESCRIPTOR_HEAP_DESC {
                Type: D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV,
                NumDescriptors: 1,
                ..Default::default()
            })
        }
        .unwrap();
        // SAFETY: a live heap.
        let at = unsafe { cpu.GetCPUDescriptorHandleForHeapStart() };
        // Four 456-byte elements placed at `address` (resource and view
        // descriptions as the title passes them to device 0x230).
        let view = |address: u64| {
            let resource: [u32; 14] = [1, 0, 0, 0, 4 * 456, 0, 1, 1 | 1 << 16, 0, 1, 0, 1, 0, 0];
            let mut desc = [0u32; 16];
            desc[..10].copy_from_slice(&[0, 1, 5768, 0, 0, 0, 4, 456, 0, 0]);
            (desc[10], desc[11]) = (address as u32, (address >> 32) as u32);
            View::Placed(false, resource, desc)
        };
        let width = |address: u64| {
            let made = crate::descriptor::host::make_view(&view(address), at).expect("a view");
            // SAFETY: a live buffer.
            unsafe { made.GetDesc() }.Width
        };
        assert_eq!(width(start + 456), 1 << 16, "on an element: the buffer's copy");
        assert_ne!(width(start + 100), 1 << 16, "off: a copy of its own bytes (in a shared buffer)");
        let (copy, address, len) = crate::descriptor::host::view_copy(&view(start + 100)).unwrap();
        assert_eq!((copy.range(), address, len), ((start + 100, 4 * 456), start + 100, 4 * 456));
        drop(copy);
        // A slot made from the view holds its copy: after more than a
        // thousand other views' copies, draws still find (and refresh) the
        // copy the slot shows; rewritten, the slot lets it go.
        let null = std::ptr::null();
        let heap_desc: [u32; 8] = [0, 8, 1, 0, 0, 0, 0, 0];
        let mut heap = null;
        let iid = crate::object::guid(0x8efb471d, 0x616c, 0x4f49, [0x90, 0xf7, 0x12, 0x7b, 0xb7, 0x63, 0xfa, 0x51]);
        assert_eq!(crate::descriptor::create_heap(null, heap_desc.as_ptr().cast(), &iid, &mut heap), S_OK);
        let slots = crate::descriptor::heap(heap).host().expect("a host view heap");
        slots.make_at(0, &view(start + 100));
        let shown = std::sync::Arc::downgrade(&crate::descriptor::host::view_copy(&view(start + 100)).unwrap().0);
        for k in 1..1200 {
            if (100 + 4 * k) % 456 != 0 {
                crate::descriptor::host::view_copy(&view(start + 100 + 4 * k)).unwrap();
            }
        }
        let found = crate::descriptor::host::view_copy(&view(start + 100)).unwrap().0;
        assert!(shown.upgrade().is_some_and(|s| std::sync::Arc::ptr_eq(&s, &found)), "the slot's copy");
        drop(found);
        slots.make_null(0);
        assert!(shown.upgrade().is_none(), "released with the slot");
    }

    #[test]
    fn placed_unordered_textures_write_and_read_back_through_placed_views() {
        let Some(host) = crate::host::host() else {
            return; // no host GPU here
        };
        let slots = Slots::new(4);
        let _ = host.log_messages();
        let va1 = title_memory(1 << 20);
        let va2 = title_memory(1 << 20);
        let tex1_res = texture_words(8, 8, 28, 0, 4); // R8G8B8A8_UNORM + UAV
        let tex2_res = texture_words(8, 8, 29, 0, 4); // R8G8B8A8_UNORM_SRGB + UAV
        let placed_uav = |va: u64, res: [u32; 14], view_fmt: u32| {
            let mut desc = [0u32; 16];
            desc[0] = view_fmt;
            desc[1] = 4; // D3D12_UAV_DIMENSION_TEXTURE2D
            (desc[10], desc[11]) = (va as u32, (va >> 32) as u32);
            View::Placed(true, res, desc)
        };
        let uav1 = placed_uav(va1, tex1_res, 28);
        let srv1 = placed_view(va1, tex1_res);
        // DXGI_FORMAT_UNKNOWN (0) on sRGB + UAV resolves through view_format -> linear 28.
        let uav2 = placed_uav(va2, tex2_res, 0);
        let srv2 = placed_view(va2, tex2_res);
        slots.make(0, &uav1);
        slots.make(1, &srv1);
        slots.make(2, &uav2);
        slots.make(3, &srv2);

        let write = compile_cs(
            r#"
RWTexture2D<float4> outTex : register(u0);
[numthreads(8, 8, 1)]
void main(uint3 id : SV_DispatchThreadID) {
    outTex[id.xy] = float4((id.x + 1.0) / 8.0, (id.y + 1.0) / 8.0, 0.5, 1.0);
}
"#,
            WRITES,
        );
        let copy_srv_to_uav = compile_cs(
            r#"
Texture2D<float4> inTex : register(t0);
RWTexture2D<float4> outTex : register(u0);
[numthreads(8, 8, 1)]
void main(uint3 id : SV_DispatchThreadID) {
    outTex[id.xy] = inTex.Load(int3(id.xy, 0));
}
"#,
            &[Param::Table(&[srv(0, 1)]), Param::Table(&[uav(0, 1)])],
        );
        let read = compile_cs(READ, READS);
        let (out, data) = host.shared_buffer(64 * 64 * 16).unwrap();

        let list = host.new_list().unwrap();
        readable(&list, &used(ResourceType::Uav), &uav1, false);
        let tex1 = crate::resource::texture_at(va1, &tex1_res).unwrap();
        assert_eq!(
            list.transition(tex1, 0x8).map(|(_, before)| before),
            None,
            "placed UAV moved to UNORDERED_ACCESS"
        );
        run(&list, &slots, &write, 0, None, (1, 1));

        readable(&list, &used(ResourceType::Srv), &srv1, false);
        readable(&list, &used(ResourceType::Uav), &uav2, false);
        assert!(!crate::texture::title_written(
            &crate::resource::host_resource(tex1).unwrap()
        ));
        // SAFETY: an open list and live objects.
        unsafe {
            list.list.SetDescriptorHeaps(&[Some(slots.0.clone())]);
            list.list.SetComputeRootSignature(&copy_srv_to_uav.0);
            list.list.SetPipelineState(&copy_srv_to_uav.1);
            list.list.SetComputeRootDescriptorTable(0, slots.gpu(1));
            list.list.SetComputeRootDescriptorTable(1, slots.gpu(2));
            list.list.Dispatch(1, 1, 1);
        }

        readable(&list, &used(ResourceType::Srv), &srv2, false);
        run(&list, &slots, &read, 3, Some(&out), (1, 1));
        list.close();
        host.execute(&[&list]);

        for (x, y) in [(0usize, 0usize), (3, 5), (7, 7)] {
            // SAFETY: inside the mapped output buffer.
            let got = unsafe { (data.add((y * 64 + x) * 16) as *const [f32; 4]).read() };
            // `srv2` is R8G8B8A8_UNORM_SRGB, so reading back the UNORM bytes written
            // through `uav2` applies sRGB-to-linear decoding on R, G, B.
            let srgb_to_linear = |c: f32| {
                let u = (c * 255.0).round() / 255.0;
                if u <= 0.04045 {
                    u / 12.92
                } else {
                    ((u + 0.055) / 1.055).powf(2.4)
                }
            };
            let want = [
                srgb_to_linear((x as f32 + 1.0) / 8.0),
                srgb_to_linear((y as f32 + 1.0) / 8.0),
                srgb_to_linear(0.5),
                1.0,
            ];
            for ch in 0..4 {
                assert!(
                    (got[ch] - want[ch]).abs() < 0.015,
                    "texel ({x},{y}) ch {ch}: got {got:?}, want {want:?}"
                );
            }
        }
        assert_eq!(host.log_messages(), None);
    }

    #[test]
    fn explicit_and_placed_rgba8_textures_at_one_address_share_compute_writes() {
        let Some(host) = crate::host::host() else {
            return; // no host GPU here
        };
        let null = std::ptr::null();
        let heap_desc: [u32; 8] = [0, 16, 1, 0, 0, 0, 0, 0];
        let mut raw_heap = null;
        let iid = crate::object::guid(
            0x8efb471d,
            0x616c,
            0x4f49,
            [0x90, 0xf7, 0x12, 0x7b, 0xb7, 0x63, 0xfa, 0x51],
        );
        assert_eq!(
            crate::descriptor::create_heap(null, heap_desc.as_ptr().cast(), &iid, &mut raw_heap),
            S_OK
        );
        let title_heap = crate::descriptor::heap(raw_heap);
        let heap_host = title_heap.host().expect("a host view heap");
        let _ = host.log_messages();

        let write = compile_cs(
            r#"
RWTexture2D<unorm float4> outTex : register(u0);
[numthreads(2, 2, 1)]
void main(uint3 id : SV_DispatchThreadID) {
    outTex[id.xy] = float4(64.0 / 255.0, 0.0, 1.0, 1.0);
}
"#,
            WRITES,
        );
        let read_bytes = compile_cs(
            r#"
Texture2D<unorm float4> inTex : register(t0);
RWByteAddressBuffer outBuf : register(u1);
[numthreads(2, 2, 1)]
void main(uint3 id : SV_DispatchThreadID) {
    uint4 px = uint4(round(inTex.Load(int3(id.xy, 0)) * 255.0));
    outBuf.Store4((id.y * 2 + id.x) * 16, px);
}
"#,
            READS,
        );
        let (out, data) = host.shared_buffer(4 * 16).unwrap();

        // Four pairings at one guest address in a single command list:
        // 0: placed UAV -> placed SRV;
        // 1: explicit address-backed UAV -> placed SRV (same resource desc);
        // 2: placed SRV (`flags = 0`) mirrored before explicit UAV (`flags = 4`) upgrades host copy;
        // 3: explicit SRV (`flags = 0`) created before placed UAV (`flags = 4`) upgrades host copy.
        for case in 0..4u32 {
            let va = title_memory(1 << 20);
            let res_rw = texture_words(2, 2, 28, 0, 4);
            let res_ro = texture_words(2, 2, 28, 0, 0);
            let uav_slot = (2 * case) as u64;
            let srv_slot = uav_slot + 1;
            let uav_handle = title_heap.start + uav_slot * 32;
            let srv_handle = title_heap.start + srv_slot * 32;

            let mut placed_uav = [0u32; 16];
            placed_uav[0] = 28;
            placed_uav[1] = 4;
            (placed_uav[10], placed_uav[11]) = (va as u32, (va >> 32) as u32);

            let mut placed_srv = [0u32; 16];
            placed_srv[0] = 28;
            placed_srv[1] = 4;
            placed_srv[2] = 0x1688;
            placed_srv[5] = 1;
            (placed_srv[10], placed_srv[11]) = (va as u32, (va >> 32) as u32);

            let uav_words: [u32; 10] = [28, 4, 0, 0, 0, 0, 0, 0, 0, 0];
            let srv_words: [u32; 10] = [28, 4, 0x1688, 0, 0, 1, 0, 0, 0, 0];

            let mut kept_obj = std::ptr::null();
            match case {
                0 => {
                    crate::descriptor::create_placed_uav(null, &res_rw, &placed_uav, uav_handle);
                    crate::descriptor::create_placed_srv(null, &res_rw, &placed_srv, srv_handle);
                }
                1 => {
                    kept_obj = texture_at_address(va, &res_rw);
                    crate::descriptor::create_uav(null, kept_obj, null, &uav_words, uav_handle);
                    crate::descriptor::create_placed_srv(null, &res_rw, &placed_srv, srv_handle);
                }
                2 => {
                    crate::descriptor::create_placed_srv(null, &res_ro, &placed_srv, srv_handle);
                    // Pre-mirror the placed SRV before the explicit UAV upgrades the memory's host texture.
                    heap_host.mirror(srv_slot as u32, srv_handle, true).unwrap();
                    kept_obj = texture_at_address(va, &res_rw);
                    crate::descriptor::create_uav(null, kept_obj, null, &uav_words, uav_handle);
                }
                _ => {
                    kept_obj = texture_at_address(va, &res_ro);
                    crate::descriptor::create_srv(null, kept_obj, &srv_words, srv_handle);
                    crate::descriptor::create_placed_uav(null, &res_rw, &placed_uav, uav_handle);
                }
            }

            if case == 1 {
                let placed_v = crate::descriptor::view(srv_handle).unwrap();
                let Some(View::Shader(placed_r, _)) =
                    crate::descriptor::host::placed_texture(&placed_v)
                else {
                    panic!("expected placed texture view");
                };
                assert!(
                    std::sync::Arc::ptr_eq(
                        &crate::resource::host_copy(kept_obj).unwrap(),
                        &crate::resource::host_copy(placed_r as This).unwrap(),
                    ),
                    "case 1 placed SRV shares the explicit texture's host copy"
                );
            }

            // SAFETY: inside the mapped output buffer.
            unsafe { std::ptr::write_bytes(data, 0, 4 * 16) };

            let list = host.new_list().unwrap();
            let uav_view = heap_host.mirror(uav_slot as u32, uav_handle, true).unwrap();
            readable(&list, &used(ResourceType::Uav), &uav_view, false);
            // SAFETY: an open list and live objects.
            unsafe {
                list.list
                    .SetDescriptorHeaps(&[Some(heap_host.heap.clone())]);
                list.list.SetComputeRootSignature(&write.0);
                list.list.SetPipelineState(&write.1);
                list.list
                    .SetComputeRootDescriptorTable(0, heap_host.gpu(uav_slot as u32));
                list.list.Dispatch(1, 1, 1);
            }

            let srv_view = heap_host.mirror(srv_slot as u32, srv_handle, true).unwrap();
            readable(&list, &used(ResourceType::Srv), &srv_view, false);
            // SAFETY: an open list and live objects.
            unsafe {
                list.list
                    .SetDescriptorHeaps(&[Some(heap_host.heap.clone())]);
                list.list.SetComputeRootSignature(&read_bytes.0);
                list.list.SetPipelineState(&read_bytes.1);
                list.list
                    .SetComputeRootDescriptorTable(0, heap_host.gpu(srv_slot as u32));
                list.list
                    .SetComputeRootUnorderedAccessView(1, out.GetGPUVirtualAddress());
                list.list.Dispatch(1, 1, 1);
            }
            list.close();
            host.execute(&[&list]);

            for texel in 0..4usize {
                // SAFETY: inside the mapped output buffer.
                let got = unsafe { (data.add(texel * 16) as *const [u32; 4]).read() };
                assert_eq!(got, [64, 0, 255, 255], "case {case} texel {texel}");
            }
            assert_eq!(host.log_messages(), None, "case {case} D3D12 debug errors");
            let _ = kept_obj;
        }
    }

    #[test]
    fn bind_cache_switch_defaults_on_and_disables_with_zero() {
        use std::ffi::OsStr;
        assert!(bind_cache_enabled_value(None));
        assert!(bind_cache_enabled_value(Some(OsStr::new("1"))));
        assert!(!bind_cache_enabled_value(Some(OsStr::new("0"))));
    }

    #[test]
    fn set_targets_caches_keys_and_clears_on_host_state_lost() {
        let Some(host) = crate::host::host() else { return };
        let Ok(list) = host.new_list() else { return };
        let mut state = State::default();
        state.set_targets(&list, &[], 0);
        assert!(state.host_targets.is_some());
        let cached = state.host_targets.clone();
        state.set_targets(&list, &[], 0);
        assert_eq!(state.host_targets, cached);
        state.host_state_lost();
        assert!(state.host_targets.is_none());
        list.close();
    }

    /// Direct root CBV/SRV/UAV bindings (`stage.root_cbvs`, `stage.root_srvs`,
    /// `stage.root_uavs`) bind zeroed/sink host buffers instead of leaving
    /// `0x0` or a previous dispatch's stale GPU VA bound when a slot is null,
    /// unset, or outside committed title memory, and shader writes into a null
    /// root UAV never corrupt the shared zero-read fallback buffer.
    #[test]
    fn null_and_unmapped_direct_root_views_bind_safe_fallbacks() {
        use victory_shader::emit::{Options, emit_with};
        use victory_shader::native::{ComputeIo, HwStage, NativeProgram, StageIo};
        use victory_shader::{Container, io, ir, resource, structure};

        let Some(host) = crate::host::host() else {
            return; // no host GPU here
        };

        const DIRECT_ROOT_HLSL: &str = r#"
cbuffer RootCbv : register(b0) { uint4 cbv_words; };
StructuredBuffer<uint> root_srv : register(t0);
RWByteAddressBuffer sink_uav : register(u0);
RWByteAddressBuffer out_uav : register(u1);
[numthreads(1, 1, 1)]
void main() {
    uint sum = cbv_words.x + root_srv[0];
    sink_uav.Store(0, 0xdeadbeefu);
    out_uav.Store(0, sum ^ 0x5a5a0000u);
}
"#;
        let params = [
            Param::Cbv {
                register: 0,
                space: 0,
            },
            Param::Srv {
                register: 0,
                space: 0,
            },
            Param::Uav { register: 0 },
            Param::Uav { register: 1 },
        ];
        let (root, pso) = crate::compute::pipeline(DIRECT_ROOT_HLSL, s!("direct_roots"), 0, &params)
            .expect("direct root view test pipeline");

        let native = NativeProgram {
            stage: HwStage::Cs,
            wave64: false,
            vgprs: 4,
            layout: Vec::new(),
            used: Vec::new(),
            io: StageIo::Cs(ComputeIo {
                rsrc2: 0,
                threads: [1, 1, 1],
            }),
            code: 0xBF81_0000u32.to_le_bytes().to_vec(),
        };
        let p = ir::build(&native.code, native.wave64, native.vgprs).unwrap();
        let st = structure::structure(&p).unwrap();
        let res = resource::track(&p, &native.layout, native.stage, Some(&st)).unwrap();
        let io = io::raise(&p, &native).unwrap();
        let host_shader = emit_with(
            &Container { parts: Vec::new() },
            &native,
            &p,
            &st,
            &io,
            &res,
            Options::default(),
        )
        .unwrap();
        let shader = crate::pipeline::Shader::for_test(
            "compute",
            crate::pipeline::Translated {
                host: host_shader,
                used: Vec::new(),
                used_written: Vec::new(),
                write_unknown: true,
                threads: None,
            },
        );
        let hp = HostPipeline {
            compute: true,
            mesh: false,
            pso,
            root,
            stages: vec![crate::pipeline::host::Stage {
                shader,
                user_data: None,
                root_cbvs: vec![(0, 0)],
                root_srvs: vec![(1, 1)],
                root_uavs: vec![(2, 2), (3, 3)],
                memory: None,
                memory_dirty: None,
                view_addresses: None,
            }],
            rectangles: false,
            slots: 0,
            inputs: Vec::new(),
            name: "direct_roots_test".into(),
            marks: None,
        };

        let cbv_buf = buffer(256);
        let srv_buf = buffer(256);
        let uav_sink_valid = buffer(256);
        let out1 = buffer(256);
        let out2 = buffer(256);
        let out3 = buffer(256);
        let (cbv_ptr, _) = crate::resource::buffer_bytes(cbv_buf).unwrap();
        let (srv_ptr, _) = crate::resource::buffer_bytes(srv_buf).unwrap();
        let (sink_ptr, _) = crate::resource::buffer_bytes(uav_sink_valid).unwrap();
        let (out1_ptr, _) = crate::resource::buffer_bytes(out1).unwrap();
        let (out2_ptr, _) = crate::resource::buffer_bytes(out2).unwrap();
        let (out3_ptr, _) = crate::resource::buffer_bytes(out3).unwrap();

        // SAFETY: inside our 256-byte title buffers.
        unsafe {
            cbv_ptr.cast::<u32>().write_volatile(0x0011_0022);
            srv_ptr.cast::<u32>().write_volatile(0x0033_0044);
            sink_ptr.cast::<u32>().write_volatile(0);
            out1_ptr.cast::<u32>().write_volatile(0);
            out2_ptr.cast::<u32>().write_volatile(0);
            out3_ptr.cast::<u32>().write_volatile(0);
        }

        let list = host.new_list().unwrap();
        let mut state = State::default();

        // Dispatch 1: valid non-zero CBV, SRV, and UAVs bind title shadow VAs.
        state.set_view(true, 0, cbv_ptr as u64);
        state.set_view(true, 1, srv_ptr as u64);
        state.set_view(true, 2, sink_ptr as u64);
        state.set_view(true, 3, out1_ptr as u64);
        assert!(state.bind(&list, &hp, true));
        // SAFETY: bound compute pipeline and root views on open `list`.
        unsafe { list.list.Dispatch(1, 1, 1) };

        // Dispatch 2 on the same command list: slot 0 = null (`0`), slot 1 =
        // invalid address (`0x1000`), slot 2 = null UAV (`0`). Without the fix,
        // `continue;` left Dispatch 1's `cbv_buf` / `srv_buf` / `uav_sink_valid`
        // VAs bound (or faulted on `0x0`).
        state.set_view(true, 0, 0);
        state.set_view(true, 1, 0x1000);
        state.set_view(true, 2, 0);
        state.set_view(true, 3, out2_ptr as u64);
        assert!(state.bind(&list, &hp, true));
        // SAFETY: fallback buffers bound on open `list`.
        unsafe { list.list.Dispatch(1, 1, 1) };

        // Dispatch 3: unset slot 0 (`State::default()`), out-of-user-space
        // slot 1 (`1 << 47`), unset slot 2 -> `zero_root_read` must still read
        // 0 despite Dispatch 2 writing `0xdeadbeef` to `sink_root_uav`.
        let mut fresh_state = State::default();
        fresh_state.set_view(true, 1, 1u64 << 47);
        fresh_state.set_view(true, 3, out3_ptr as u64);
        assert!(fresh_state.bind(&list, &hp, true));
        // SAFETY: fallback buffers bound on open `list`.
        unsafe { list.list.Dispatch(1, 1, 1) };

        list.close();
        host.execute(&[&list]);

        // SAFETY: `execute` waited for the GPU fence and completed copy-back.
        unsafe {
            assert_eq!(
                out1_ptr.cast::<u32>().read_volatile(),
                (0x0011_0022 + 0x0033_0044) ^ 0x5a5a_0000,
                "valid direct root views ran"
            );
            assert_eq!(
                sink_ptr.cast::<u32>().read_volatile(),
                0xdead_beef,
                "Dispatch 2 did not overwrite Dispatch 1's UAV after slot 2 became null, and Dispatch 1 wrote 0xdeadbeef"
            );
            assert_eq!(
                out2_ptr.cast::<u32>().read_volatile(),
                0x5a5a_0000,
                "null CBV + unmapped SRV read zeros instead of stale Dispatch 1 VAs"
            );
            assert_eq!(
                out3_ptr.cast::<u32>().read_volatile(),
                0x5a5a_0000,
                "null UAV writes into sink_root_uav never corrupt zero_root_read"
            );
        }
        assert_eq!(host.log_messages(), None, "no D3D12 debug errors");
    }

    fn local_memo_counts() -> [u64; MEMO_COUNTERS] {
        MEMO_LOCAL.with_borrow(|(c, _)| c.0)
    }

    #[test]
    fn table_memo_mode_parsing() {
        use std::ffi::OsStr;
        assert_eq!(table_memo_mode_value(None), 0);
        assert_eq!(table_memo_mode_value(Some(OsStr::new("0"))), 0);
        assert_eq!(table_memo_mode_value(Some(OsStr::new(""))), 0);
        assert_eq!(table_memo_mode_value(Some(OsStr::new("invalid"))), 0);
        assert_eq!(table_memo_mode_value(Some(OsStr::new("1"))), 1);
        assert_eq!(table_memo_mode_value(Some(OsStr::new("2"))), 2);
        assert_eq!(table_memo_mode_value(Some(OsStr::new("verify"))), 2);
    }

    #[test]
    fn table_memo_skips_repeat_entries_and_verifies_zero_mismatches() {
        let Some(host) = crate::host::host() else { return };
        let Ok(list) = host.new_list() else { return };
        let p = crate::watch::tests::pool(1);
        let buf = buffer_at_address(p as u64, crate::watch::PAGE as u64);
        let tex_mem = crate::watch::tests::pool(1) as u64;
        let tex = texture_at_address(tex_mem, &texture_words(32, 32, 28, 0x100, 0));
        let (heap_raw, title_heap, _heap_host) = view_heap(8);

        let slot0 = title_heap.start;
        let slot1 = title_heap.start + 32;
        let cbv_desc: [u32; 4] = [p as u32, (p as u64 >> 32) as u32, 256, 0];
        crate::descriptor::create_cbv(std::ptr::null(), &cbv_desc, slot0);
        crate::descriptor::create_srv(std::ptr::null(), tex, std::ptr::null(), slot1);

        let mut state = State::default();
        state.set_heaps(&[heap_raw]);
        state.set_table(false, 0, title_heap.start);
        state.depth = tex as usize;

        let u_cbv = used(ResourceType::Cbv);
        let u_srv = UsedResource {
            table_offset_dwords: 8,
            ..used(ResourceType::Srv)
        };

        // Verify mode (=2): repeat hits run the full path and must report 0
        // epoch, deferred-step, or tested-depth mismatches.
        {
            let _guard = TableMemoGuard::set(2);
            flush_memo_local();
            let prev = TESTED_DEPTH.with(|d| d.replace((tex as usize, false)));

            state.table_entry(&list, &u_cbv, false, true);
            let epoch_after_cbv = list.epoch();
            let steps_after_cbv = list.deferred_steps();
            state.table_entry(&list, &u_cbv, false, true);
            assert_eq!(list.epoch(), epoch_after_cbv);
            assert_eq!(list.deferred_steps(), steps_after_cbv);

            state.table_entry(&list, &u_srv, false, true);
            assert!(TESTED_DEPTH.with(std::cell::Cell::get).1, "first SRV read marks tested depth");
            let epoch_after_srv = list.epoch();
            let steps_after_srv = list.deferred_steps();
            TESTED_DEPTH.with(|d| d.set((tex as usize, false)));
            state.table_entry(&list, &u_srv, false, true);
            assert!(TESTED_DEPTH.with(std::cell::Cell::get).1, "verified SRV hit preserves tested depth");
            assert_eq!(list.epoch(), epoch_after_srv);
            assert_eq!(list.deferred_steps(), steps_after_srv);

            let c = local_memo_counts();
            assert_eq!(c[0], 4, "4 lookups");
            assert_eq!(c[1], 2, "2 verify-mode hits");
            assert_eq!(c[2], 0, "0 verify epoch mismatches");
            assert_eq!(c[3], 0, "0 verify deferred_steps mismatches");
            assert_eq!(c[4], 0, "0 verify depth mismatches");
            TESTED_DEPTH.with(|d| d.set(prev));
        }

        // Active skip mode (=1): hits return immediately while preserving
        // `TESTED_DEPTH`.
        {
            let _guard = TableMemoGuard::set(1);
            flush_memo_local();
            let prev = TESTED_DEPTH.with(|d| d.replace((tex as usize, false)));
            let epoch_before = list.epoch();
            let steps_before = list.deferred_steps();

            state.table_entry(&list, &u_srv, false, true);
            assert!(
                TESTED_DEPTH.with(std::cell::Cell::get).1,
                "active skip hit must restore tested depth flag"
            );
            assert_eq!(list.epoch(), epoch_before);
            assert_eq!(list.deferred_steps(), steps_before);

            let c = local_memo_counts();
            assert_eq!(c[0], 1, "1 lookup");
            assert_eq!(c[1], 1, "1 active skip hit");
            TESTED_DEPTH.with(|d| d.set(prev));
        }

        list.close();
        crate::shadow::unregister(buf as usize, p as u64);
        let release = crate::object::slot::<extern "system" fn(This) -> u32>;
        assert_eq!(release(tex, 0x10)(tex), 0);
        assert_eq!(release(buf, 0x10)(buf), 0);
        assert_eq!(release(heap_raw, 0x10)(heap_raw), 0);
    }

    #[test]
    fn table_memo_invalidates_on_descriptor_change_epoch_bump_stores_and_reset() {
        let Some(host) = crate::host::host() else { return };
        let Ok(list) = host.new_list() else { return };
        let p = crate::watch::tests::pool(1);
        let buf = buffer_at_address(p as u64, crate::watch::PAGE as u64);
        let (heap_raw, title_heap, _heap_host) = view_heap(8);

        let slot0 = title_heap.start;
        let cbv_a: [u32; 4] = [p as u32, (p as u64 >> 32) as u32, 256, 0];
        let cbv_b: [u32; 4] = [p as u32, (p as u64 >> 32) as u32, 512, 0];
        crate::descriptor::create_cbv(std::ptr::null(), &cbv_a, slot0);

        let mut state = State::default();
        state.set_heaps(&[heap_raw]);
        state.set_table(false, 0, title_heap.start);
        let u_cbv = used(ResourceType::Cbv);

        let _guard = TableMemoGuard::set(1);
        flush_memo_local();

        // Initial miss + repeat hit.
        state.table_entry(&list, &u_cbv, false, true);
        state.table_entry(&list, &u_cbv, false, true);
        let c = local_memo_counts();
        assert_eq!((c[0], c[1], c[5]), (2, 1, 1), "first miss then hit");

        // (a) Descriptor bytes change at the same handle -> miss, including ABA
        // restoration back to `cbv_a`.
        crate::descriptor::create_cbv(std::ptr::null(), &cbv_b, slot0);
        state.table_entry(&list, &u_cbv, false, true);
        let c = local_memo_counts();
        assert_eq!((c[0], c[1], c[5]), (3, 1, 2), "changed descriptor bytes miss");
        state.table_entry(&list, &u_cbv, false, true);
        assert_eq!(local_memo_counts()[1], 2, "updated descriptor hits on repeat");
        crate::descriptor::create_cbv(std::ptr::null(), &cbv_a, slot0);
        state.table_entry(&list, &u_cbv, false, true);
        let c = local_memo_counts();
        assert_eq!((c[0], c[1], c[5]), (5, 2, 3), "ABA descriptor restoration misses");

        // (b) `list.deferred_epoch()` bump (e.g., deferred copy) -> epoch miss.
        let src = [0x5Au8; 16];
        list.defer_copy(p as *mut u8, src.as_ptr(), src.len());
        state.table_entry(&list, &u_cbv, false, true);
        let c = local_memo_counts();
        assert_eq!(c[6], 1, "deferred_epoch bump causes epoch miss");
        assert_eq!(c[1], 2, "no hit after deferred copy");

        // (c) Texture-only `list.epoch()` bump does NOT invalidate buffer CBVs.
        list.changed_something();
        state.table_entry(&list, &u_cbv, false, true);
        let c = local_memo_counts();
        assert_eq!(c[1], 3, "texture-only epoch bump preserves CBV memo hit");

        // (d) `State::reset` & `HostList::reset` clear `entry_memo` and bump epochs.
        assert!(!state.entry_memo.is_empty());
        let epoch_before_reset = list.epoch();
        let def_epoch_before_reset = list.deferred_epoch();
        state.reset(std::ptr::null());
        list.close();
        list.reset();
        assert!(state.entry_memo.is_empty(), "State::reset clears entry_memo");
        assert_ne!(list.epoch(), epoch_before_reset, "HostList::reset advances epoch");
        assert_ne!(
            list.deferred_epoch(),
            def_epoch_before_reset,
            "HostList::reset advances deferred_epoch"
        );

        state.set_heaps(&[heap_raw]);
        state.set_table(false, 0, title_heap.start);
        state.table_entry(&list, &u_cbv, false, true);
        assert_eq!(local_memo_counts()[1], 3, "post-reset call misses cleanly");

        list.close();
        crate::shadow::unregister(buf as usize, p as u64);
        let release = crate::object::slot::<extern "system" fn(This) -> u32>;
        assert_eq!(release(buf, 0x10)(buf), 0);
        assert_eq!(release(heap_raw, 0x10)(heap_raw), 0);
    }

    fn run_shadow_preserve_gpu_write_repro(p: usize, len: usize) {
        let _keep = crate::shadow::KeepGuard::enable();
        let Some(host) = crate::host::host() else {
            return;
        };
        // SAFETY: committed buffer of `len` bytes.
        unsafe {
            (p as *mut u32).write_volatile(0x1111_1111);
            ((p + 0x200) as *mut u32).write_volatile(0x2222_2222);
        }
        let buf = buffer_at_address(p as u64, len as u64);
        let copy = crate::shadow::find(p as u64, len as u64).expect("shadow copy");
        assert!(!copy.is_aliased(), "test requires a non-aliased shadow buffer");
        let (out_buf, out_data) = host.shared_buffer(256).expect("output buffer");
        // SAFETY: live output buffer.
        let out_gpu = unsafe { out_buf.GetGPUVirtualAddress() };
        // SAFETY: inside mapped output buffer.
        unsafe { std::ptr::write_bytes(out_data, 0, 256) };

        let writer = compile_cs(
            r#"
RWByteAddressBuffer dst : register(u0);
[numthreads(1, 1, 1)]
void main() { dst.Store(0, 0xDEADBEEFu); }
"#,
            &[Param::Uav { register: 0 }],
        );
        let reader = compile_cs(
            r#"
ByteAddressBuffer src : register(t0);
RWByteAddressBuffer dst : register(u0);
[numthreads(1, 1, 1)]
void main() {
    dst.Store(0, src.Load(0));
    dst.Store(4, src.Load(0x200));
}
"#,
            &[
                Param::Srv {
                    register: 0,
                    space: 0,
                },
                Param::Uav { register: 0 },
            ],
        );

        // Submission 1: initial refresh uploads CPU values, then GPU writes
        // 0xDEADBEEF to word 0 (kept in shadow, not copied back to CPU).
        let list1 = host.new_list().unwrap();
        copy.refresh(&list1, p as u64, len as u64);
        // SAFETY: open command list and live root/PSO/buffers.
        unsafe {
            list1.list.SetComputeRootSignature(&writer.0);
            list1.list.SetComputeRootUnorderedAccessView(0, copy.gpu(p as u64));
            list1.list.SetPipelineState(&writer.1);
            list1.list.Dispatch(1, 1, 1);
        }
        list1.close();
        host.execute(&[&list1]);
        assert_eq!(copy.read_host(p as u64, 1), [0xDEAD_BEEF]);
        // SAFETY: committed title memory.
        assert_eq!(unsafe { (p as *const u32).read_volatile() }, 0x1111_1111);

        // Between submissions: sibling CPU write in a different 256-byte piece
        // on the same 64 KiB page marks the page dirty while word 0 stays unchanged.
        // SAFETY: committed title memory.
        unsafe { ((p + 0x200) as *mut u32).write_volatile(0x3333_3333) };

        // Submission 2: refresh runs, then GPU reads both words into `out_buf`.
        let list2 = host.new_list().unwrap();
        copy.refresh(&list2, p as u64, len as u64);
        // SAFETY: open command list and live root/PSO/buffers.
        unsafe {
            list2.list.SetComputeRootSignature(&reader.0);
            list2.list.SetComputeRootShaderResourceView(0, copy.gpu(p as u64));
            list2.list.SetComputeRootUnorderedAccessView(1, out_gpu);
            list2.list.SetPipelineState(&reader.1);
            list2.list.Dispatch(1, 1, 1);
        }
        list2.close();
        host.execute(&[&list2]);

        // SAFETY: inside mapped output buffer after GPU completion.
        let got = unsafe { std::slice::from_raw_parts(out_data.cast::<u32>(), 2).to_vec() };
        crate::shadow::unregister(buf as usize, p as u64);
        assert_eq!(
            got,
            [0xDEAD_BEEF, 0x3333_3333],
            "GPU write at word 0 must survive refresh when sibling CPU write dirties the same 64 KiB page"
        );
    }

    #[test]
    fn shadow_refresh_preserves_gpu_writes_on_same_tracked_page() {
        let len = crate::watch::PAGE;
        let p = crate::watch::tests::pool(1);
        run_shadow_preserve_gpu_write_repro(p, len);
    }

    #[test]
    fn shadow_refresh_preserves_gpu_writes_on_untracked_buffer() {
        use windows_sys::Win32::System::Memory::{
            MEM_COMMIT, MEM_RESERVE, PAGE_READWRITE, VirtualAlloc,
        };
        let len = crate::watch::PAGE;
        // SAFETY: fresh plain committed page (untracked by fault tracking).
        let p = unsafe {
            VirtualAlloc(std::ptr::null(), len, MEM_RESERVE | MEM_COMMIT, PAGE_READWRITE)
        } as usize;
        assert_ne!(p, 0);
        run_shadow_preserve_gpu_write_repro(p, len);
        crate::watch::tests::free(p);
    }
}
