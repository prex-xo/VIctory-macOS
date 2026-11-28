//! Depth textures the title also writes through unordered access (decision
//! 0007, "One host texture per texture memory"). The console allows both
//! uses of one texture (Grand Theft Auto VI's shadow-map copies and its
//! material depth, written by compute and then tested as depth); the host
//! does not: a depth-stencil resource takes no unordered access, in any
//! format. Such a texture gets a writable twin, a colour texture per plane
//! (depth, stencil) that its unordered access views write, and the newer
//! contents move between the two by copies where the use changes: within
//! a list where it records the change (`HostList::transition`), and at
//! submission for each list's first use, from the side the lists before it
//! left newer. A 16-bit depth plane in the host's 32-bit float one (the
//! console's 16-bit depth with stencil) is converted on the way, so a
//! value written as 16-bit depth compares equal to the same depth
//! rasterised.
//!
//! shortcut: uses the lists do not name (shaders indexing the heap
//! themselves) move nothing; a whole twin is copied for any use of either
//! plane, also when the next pass overwrites all of it; console 24-bit
//! depth gets no twin. Upgrade when a frame trace shows such a use, or the
//! copies (counted on the frame-rate line) cost.

use std::cell::Cell;
use std::sync::Arc;
use std::sync::OnceLock;
use std::sync::atomic::{AtomicBool, AtomicU8, AtomicU64, Ordering::Relaxed};

use windows::Win32::Graphics::Direct3D12::*;
use windows::Win32::Graphics::Dxgi::Common::*;
use windows::core::s;

use crate::host::{HostList, host, transition, uav_barrier};
use crate::object::This;
use crate::resource::HostTexture;

/// Which side holds the newer contents.
pub(crate) const BOTH: u8 = 0;
pub(crate) const DEPTH: u8 = 1;
pub(crate) const TWIN: u8 = 2;

const RENDER_TARGET: u32 = 0x4;
const UNORDERED_ACCESS: u32 = 0x8;
const DEPTH_WRITE: u32 = 0x10;
const COPY_DEST: u32 = 0x400;
const COPY_SOURCE: u32 = 0x800;
const RESOLVE_DEST: u32 = 0x1000;

/// Copies between a depth texture and its twin since the last frame-rate
/// line (either way, within lists and at submission).
pub(crate) static COPIES: AtomicU64 = AtomicU64::new(0);

/// Since the last frame-rate line: copies into the depth texture and into
/// the twin, the bytes they moved (the twin's side), and the copies into
/// the depth texture a whole clear made pointless (`CLEARED`: owed before
/// a clear of every plane of a one-subresource texture; `SKIPPED`: of
/// those, left out, with `VICTORY_TWIN_CLEAR_SKIP=1`).
static INTO_DEPTH: AtomicU64 = AtomicU64::new(0);
static INTO_TWIN: AtomicU64 = AtomicU64::new(0);
static BYTES: AtomicU64 = AtomicU64::new(0);
static CLEARED: AtomicU64 = AtomicU64::new(0);
static SKIPPED: AtomicU64 = AtomicU64::new(0);
/// Copies into the depth texture from a twin no shader stored to since the
/// two were last the same (every unordered use since stored nothing): the
/// bytes they move are unchanged.
static UNCHANGED: AtomicU64 = AtomicU64::new(0);

/// `VICTORY_TWIN_UAV_READS=1`: an unordered use of a depth texture whose
/// shaders store nothing through it (`crate::draw`'s per-slot write bits)
/// reads the twin and leaves it as current as it was, instead of making it
/// the newer side: the next depth use copies nothing back.
fn uav_reads() -> bool {
    static ON: OnceLock<bool> = OnceLock::new();
    *ON.get_or_init(|| crate::switch::on("VICTORY_TWIN_UAV_READS"))
}

/// `VICTORY_TWIN_CLEAR_SKIP=1`: a clear that overwrites a whole depth
/// texture needs none of its older contents, so the copy into it from a
/// newer twin is left out.
fn clear_skip() -> bool {
    static ON: OnceLock<bool> = OnceLock::new();
    *ON.get_or_init(|| crate::switch::on("VICTORY_TWIN_CLEAR_SKIP"))
}

thread_local! {
    /// While the use being routed is a clear without rectangles
    /// (`overwriting`): its flags (1 depth, 2 stencil); else 0.
    static WHOLE: Cell<u32> = const { Cell::new(0) };
    /// While the use being routed is an unordered one whose shaders store
    /// nothing through the view (`reading`).
    static READING: Cell<bool> = const { Cell::new(false) };
}

/// Runs `f`, the transition for an unordered view no shader of the draw
/// or dispatch stores through.
pub(crate) fn reading<R>(f: impl FnOnce() -> R) -> R {
    READING.with(|r| r.set(true));
    let r = f();
    READING.with(|r| r.set(false));
    r
}

/// Runs `f`, the transition for a clear without rectangles of the planes
/// `flags` names (1 depth, 2 stencil): if those are all the texture has and
/// it has one subresource per plane, the use overwrites all of it.
pub(crate) fn overwriting<R>(flags: u32, f: impl FnOnce() -> R) -> R {
    WHOLE.with(|w| w.set(flags & 3));
    let r = f();
    WHOLE.with(|w| w.set(0));
    r
}

/// Whether a clear of the planes `flags` names overwrites a texture whose
/// twin has `planes` planes and `routes` subresources in all.
fn covers(flags: u32, planes: usize, routes: usize) -> bool {
    let need = if planes > 1 { 3 } else { 1 };
    flags & need == need && routes == planes
}

/// The twin copy line, after the frame-rate line's total.
pub(crate) fn log(frames: f64) {
    let [depth, twin, bytes, cleared, skipped, unchanged] =
        [&INTO_DEPTH, &INTO_TWIN, &BYTES, &CLEARED, &SKIPPED, &UNCHANGED].map(|c| c.swap(0, Relaxed));
    if depth + twin + cleared == 0 {
        return;
    }
    tracing::info!(
        "depth twin copies: {:.1}/frame into depth, {:.1}/frame into twins, {:.1} MB/frame; \
         {:.1}/frame into depth before a whole clear, {:.1}/frame of them left out (VICTORY_TWIN_CLEAR_SKIP); \
         {:.1}/frame into depth from a twin no shader stored to (VICTORY_TWIN_UAV_READS)",
        depth as f64 / frames,
        twin as f64 / frames,
        bytes as f64 / frames / 1e6,
        cleared as f64 / frames,
        skipped as f64 / frames,
        unchanged as f64 / frames,
    );
}

pub(crate) struct Twin {
    /// The colour texture of each plane the depth texture has (depth, then
    /// stencil), shader-writable.
    pub(crate) planes: Vec<Arc<HostTexture>>,
    /// Which side is newer as of the lists submitted so far.
    pub(crate) newer: AtomicU8,
    /// With the twin newer: whether a shader stored to it since the two
    /// were last the same, as of the lists submitted so far.
    stored: AtomicBool,
    /// Per plane and subresource: how its bytes go through `scratch`.
    routes: Vec<Route>,
    /// A shader-writable buffer the copies go through, in the common state
    /// between them.
    scratch: ID3D12Resource,
}

// SAFETY: D3D12 objects are free-threaded; the rest is atomics and data.
unsafe impl Send for Twin {}
// SAFETY: as above.
unsafe impl Sync for Twin {}

/// One subresource of one plane.
struct Route {
    plane: usize,
    /// Its index in the twin's plane texture and in the depth texture.
    twin_sub: u32,
    depth_sub: u32,
    /// Its place in the scratch buffer as the twin's texture has it.
    twin: D3D12_PLACED_SUBRESOURCE_FOOTPRINT,
    /// As the depth texture has it, where that differs: a 16-bit depth
    /// plane the host keeps as 32-bit floats, converted.
    depth: Option<D3D12_PLACED_SUBRESOURCE_FOOTPRINT>,
}

/// A depth texture's use in one list: the side its first use needs (made
/// newer at submission, `at_submission`) and the side the list left newer
/// (`None`: unchanged from where it started).
pub(crate) struct Use {
    first: u8,
    last: Option<u8>,
    /// The first use overwrites the whole depth texture (`Use::new`).
    whole: bool,
    /// A shader stored to the twin in this list since its last copy (or
    /// its start), and whether a copy was recorded in it.
    stored: bool,
    synced: bool,
    depth: Arc<HostTexture>,
}

/// The twin's plane formats for a console depth format (resource or
/// view format): 16- or 32-bit depth, then 8-bit stencil if it has one.
/// `None` for 24-bit depth, whose host plane takes no unordered view.
fn plane_formats(console: u32) -> Option<&'static [u32]> {
    const R16: u32 = 53;
    const R32: u32 = 39;
    const R8: u32 = 60;
    match console {
        53 | 55 => Some(&[R16]),
        39..=41 => Some(&[R32]),
        118..=120 => Some(&[R16, R8]),
        19..=22 => Some(&[R32, R8]),
        _ => None,
    }
}

/// The twin of the depth texture `depth`, whose title resource has format
/// `console`; `None` if the host makes none.
fn make(console: u32, depth: &ID3D12Resource) -> Option<Twin> {
    let host = host()?;
    let Some(formats) = plane_formats(console) else {
        once(console, 0, 0, "a depth texture of this format gets no writable twin: its unordered writes are dropped");
        return None;
    };
    // SAFETY: a live resource.
    let dd = unsafe { depth.GetDesc() };
    let subs = u32::from(dd.MipLevels) * u32::from(dd.DepthOrArraySize).max(1);
    let heap = D3D12_HEAP_PROPERTIES {
        Type: D3D12_HEAP_TYPE_DEFAULT,
        ..Default::default()
    };
    let mut planes = Vec::new();
    for &f in formats {
        let desc = D3D12_RESOURCE_DESC {
            Format: DXGI_FORMAT(f as i32),
            Flags: D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS,
            Layout: D3D12_TEXTURE_LAYOUT_UNKNOWN,
            Alignment: 0,
            ..dd
        };
        let mut texture: Option<ID3D12Resource> = None;
        // SAFETY: plain creation; our out pointer.
        let made = unsafe {
            host.device.CreateCommittedResource(&heap, D3D12_HEAP_FLAG_NONE, &desc, D3D12_RESOURCE_STATE_COMMON, None, &mut texture)
        };
        if let Err(e) = made {
            tracing::warn!(error = %e, ?desc, "no writable twin for a depth texture");
            return None;
        }
        let texture = texture?;
        crate::host::name(&texture, || format!("depth twin {}x{} f{f}", dd.Width, dd.Height));
        planes.push(Arc::new(HostTexture::new(texture, 0, Vec::new())));
    }
    // The depth texture's footprints, every plane, and the twin's.
    let n = subs * planes.len() as u32;
    let mut df = vec![D3D12_PLACED_SUBRESOURCE_FOOTPRINT::default(); n as usize];
    let mut drows = vec![0u32; n as usize];
    // SAFETY: outputs sized for `n` subresources.
    unsafe { host.device.GetCopyableFootprints(&dd, 0, n, 0, Some(df.as_mut_ptr()), Some(drows.as_mut_ptr()), None, None) };
    let mut routes = Vec::new();
    let mut at = 0u64;
    let mut place = |mut f: D3D12_PLACED_SUBRESOURCE_FOOTPRINT, rows: u32| {
        f.Offset = at;
        let bytes = u64::from(f.Footprint.RowPitch) * u64::from(rows) * u64::from(f.Footprint.Depth);
        at = (at + bytes).next_multiple_of(512);
        f
    };
    for (p, plane) in planes.iter().enumerate() {
        // SAFETY: a live resource.
        let td = unsafe { plane.texture.GetDesc() };
        let mut tf = vec![D3D12_PLACED_SUBRESOURCE_FOOTPRINT::default(); subs as usize];
        let mut trows = vec![0u32; subs as usize];
        // SAFETY: outputs sized for `subs` subresources.
        unsafe { host.device.GetCopyableFootprints(&td, 0, subs, 0, Some(tf.as_mut_ptr()), Some(trows.as_mut_ptr()), None, None) };
        for s in 0..subs {
            let d = (p as u32 * subs + s) as usize;
            let same = df[d].Footprint.Format == td.Format;
            // The same bytes: both sides copy through the depth texture's
            // footprint; else each its own, converted between.
            let route = if same {
                Route { plane: p, twin_sub: s, depth_sub: d as u32, twin: place(df[d], drows[d]), depth: None }
            } else {
                let twin = place(tf[s as usize], trows[s as usize]);
                Route { plane: p, twin_sub: s, depth_sub: d as u32, twin, depth: Some(place(df[d], drows[d])) }
            };
            if route.depth.is_some() && (td.Format != DXGI_FORMAT_R16_TYPELESS || df[d].Footprint.Format != DXGI_FORMAT_R32_TYPELESS) {
                once(console, td.Format.0 as u32, df[d].Footprint.Format.0 as u32, "a depth twin plane with no conversion to the host's: not made");
                return None;
            }
            routes.push(route);
        }
    }
    let scratch = shared_scratch(at)?;
    tracing::debug!(console, width = dd.Width, height = dd.Height, planes = planes.len(), "a writable twin for a depth texture");
    Some(Twin { planes, newer: AtomicU8::new(DEPTH), stored: AtomicBool::new(false), routes, scratch })
}

/// Scratch buffers shared by the twins, one per power-of-two size: a
/// buffer each held as much video memory as its twin's planes for good
/// (about 325 MB for Grand Theft Auto VI's 315 twins in the menu).
/// Sharing is safe: every copy through one ([`Twin::copy`]) starts in the
/// common state and returns the buffer to it, so copies in one list are
/// ordered by those transitions, and lists run one after another on the
/// host's single queue.
struct Scratch(Vec<(u64, ID3D12Resource)>);

// SAFETY: D3D12 resources are free-threaded.
unsafe impl Send for Scratch {}

static SCRATCH: std::sync::Mutex<Scratch> = std::sync::Mutex::new(Scratch(Vec::new()));

/// The shared scratch buffer of at least `size` bytes: the one of `size`'s
/// power-of-two class, made on first use. All of them together hold at
/// most twice the largest.
fn shared_scratch(size: u64) -> Option<ID3D12Resource> {
    let class = size.max(1 << 16).next_power_of_two();
    let mut all = SCRATCH.lock().unwrap_or_else(|e| e.into_inner());
    if let Some((_, buffer)) = all.0.iter().find(|(c, _)| *c == class) {
        return Some(buffer.clone());
    }
    let buffer = scratch_buffer(class)?;
    crate::host::name(&buffer, || format!("depth twin copies {class:#x}"));
    all.0.push((class, buffer.clone()));
    Some(buffer)
}

/// A default-heap buffer of `size` bytes shaders may write, in the common
/// state.
fn scratch_buffer(size: u64) -> Option<ID3D12Resource> {
    let host = host()?;
    let heap = D3D12_HEAP_PROPERTIES {
        Type: D3D12_HEAP_TYPE_DEFAULT,
        ..Default::default()
    };
    let desc = D3D12_RESOURCE_DESC {
        Dimension: D3D12_RESOURCE_DIMENSION_BUFFER,
        Width: size.max(256),
        Height: 1,
        DepthOrArraySize: 1,
        MipLevels: 1,
        SampleDesc: DXGI_SAMPLE_DESC { Count: 1, Quality: 0 },
        Layout: D3D12_TEXTURE_LAYOUT_ROW_MAJOR,
        Flags: D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS,
        ..Default::default()
    };
    let mut buffer: Option<ID3D12Resource> = None;
    // SAFETY: plain creation.
    let made = unsafe {
        host.device.CreateCommittedResource(&heap, D3D12_HEAP_FLAG_NONE, &desc, D3D12_RESOURCE_STATE_COMMON, None, &mut buffer)
    };
    made.ok().and(buffer)
}

/// Logs once per (console format, view format, plane) what happened.
fn once(console: u32, view: u32, plane: u32, what: &str) {
    static SEEN: std::sync::Mutex<Vec<(u32, u32, u32, String)>> = std::sync::Mutex::new(Vec::new());
    let mut seen = SEEN.lock().unwrap_or_else(|e| e.into_inner());
    let key = (console, view, plane, what.to_owned());
    if seen.len() < 64 && !seen.contains(&key) {
        tracing::info!(console, view, plane, "{what}");
        seen.push(key);
    }
}

/// An unordered access view at `at` of `resource`, a depth texture whose
/// host copy is `copy`, made on its twin's plane the title's view names
/// (`words`: the view description), in a format of that plane's family
/// (the title's own if it is one, else the plane's integer or normalised
/// one, as the title's format is). The twin's plane texture, which the
/// slot then views.
pub(crate) fn uav(
    resource: This,
    copy: &Arc<HostTexture>,
    words: Option<&[u32; 10]>,
    at: D3D12_CPU_DESCRIPTOR_HANDLE,
) -> Option<ID3D12Resource> {
    let console = crate::resource::title_format(resource)?;
    let twin = copy.twin.get_or_init(|| make(console, &copy.texture)).as_ref()?;
    let mut words = *words?;
    const TEXTURE_2D: u32 = 4;
    const TEXTURE_2D_ARRAY: u32 = 5;
    // The plane: last of the 2D view's fields (mip, plane; mip, first
    // slice, slices, plane).
    let plane_word = match words[1] {
        TEXTURE_2D => 3,
        TEXTURE_2D_ARRAY => 5,
        other => {
            once(console, words[0], other, "an unordered view of a depth texture of this dimension: not made");
            return None;
        }
    };
    let plane = words[plane_word] as usize;
    let Some(texture) = twin.planes.get(plane) else {
        once(console, words[0], plane as u32, "an unordered view of a depth plane the twin does not have: not made");
        return None;
    };
    // SAFETY: a live resource.
    let own = unsafe { texture.texture.GetDesc() }.Format.0 as u32;
    let view = crate::resource::pc_format(words[0]);
    let format = if crate::resource::family(view) == own {
        view
    } else {
        // R16_TYPELESS: unorm or uint; R32_TYPELESS: float or uint; R8: uint.
        match (own, integer(view)) {
            (53, false) => 56,
            (53, true) => 57,
            (39, false) => 41,
            (39, true) => 42,
            _ => 62,
        }
    };
    once(console, words[0], plane as u32, "an unordered view of a depth texture, made on its writable twin");
    words[0] = format;
    words[plane_word] = 0;
    // SAFETY: 40 bytes in the host's layout.
    let desc = unsafe { std::ptr::from_ref(&words).cast::<D3D12_UNORDERED_ACCESS_VIEW_DESC>().read_unaligned() };
    // SAFETY: a live texture and a slot of our heap.
    unsafe { host()?.device.CreateUnorderedAccessView(&texture.texture, None, Some(&desc), at) };
    Some(texture.texture.clone())
}

/// Whether a format holds integers (`_UINT`, `_SINT`).
fn integer(format: u32) -> bool {
    matches!(format, 3 | 4 | 7 | 8 | 12 | 14 | 17 | 18 | 25 | 30 | 32 | 36 | 38 | 42 | 43 | 50 | 52 | 57 | 59 | 62 | 64)
}

/// Before a use in `list` of the depth texture `depth` (with `twin`) in
/// host state `wanted`: the side the use needs is brought up to date (a
/// copy recorded now, or by the list's prelude for its first use). For
/// unordered access the twin's planes are moved there and true returned:
/// the depth texture itself is not used.
pub(crate) fn route(list: &HostList, depth: &Arc<HostTexture>, twin: &Twin, wanted: u32) -> bool {
    let side = if wanted & UNORDERED_ACCESS != 0 { TWIN } else { DEPTH };
    let writes = side == TWIN || wanted & (DEPTH_WRITE | RENDER_TARGET | COPY_DEST | RESOLVE_DEST) != 0;
    // A whole clear: one subresource per plane, so its view covers it all.
    let whole = side == DEPTH && writes && covers(WHOLE.with(Cell::get), twin.planes.len(), twin.routes.len());
    let stores = side == TWIN && !READING.with(Cell::get);
    let writes = writes && (side == DEPTH || stores || !uav_reads());
    if let Some(to_depth) = list.twin_use(depth, side, writes, whole, stores)
        && twin.copy(&list.list, depth, to_depth, &mut |t, w| list.move_texture(t, w))
    {
        list.need_rebind();
    }
    if side == DEPTH {
        return false;
    }
    let barriers: Vec<_> = twin.planes.iter().filter_map(|p| list.move_texture(p, UNORDERED_ACCESS)).collect();
    if !barriers.is_empty() {
        // SAFETY: an open list and live textures.
        unsafe { list.list.ResourceBarrier(&barriers) };
    }
    true
}

impl Use {
    /// `whole`: the use overwrites every byte of the depth texture (a
    /// whole clear), so a copy owed into it is pointless.
    pub(crate) fn new(depth: &Arc<HostTexture>, side: u8, writes: bool, whole: bool) -> Self {
        Use { first: side, last: writes.then_some(side), whole, stored: false, synced: false, depth: depth.clone() }
    }

    /// A later use in the list needing `side`: whether a copy must be
    /// recorded first (true: into the depth texture).
    pub(crate) fn again(&mut self, side: u8, writes: bool, whole: bool) -> Option<bool> {
        let copy = again(self.first, &mut self.last, side, writes);
        if copy == Some(true) && self.synced && !self.stored {
            UNCHANGED.fetch_add(1, Relaxed);
        }
        let copy = cleared(copy, whole, clear_skip());
        if copy.is_some() {
            self.synced = true;
            self.stored = false;
        }
        copy
    }

    /// A use of the twin by a shader that stores to it.
    pub(crate) fn store(&mut self) {
        self.stored = true;
    }
}

/// A copy owed before a use: left out (`None`) if the use overwrites it
/// all and `skip` is on; counted as pointless either way.
fn cleared(copy: Option<bool>, whole: bool, skip: bool) -> Option<bool> {
    if copy == Some(true) && whole {
        CLEARED.fetch_add(1, Relaxed);
        if skip {
            SKIPPED.fetch_add(1, Relaxed);
            return None;
        }
    }
    copy
}

/// [`Use::again`] on its first side and the side it left newer.
fn again(first: u8, last: &mut Option<u8>, side: u8, writes: bool) -> Option<bool> {
    // Unchanged since the list's start: only the first use's side is
    // known to be current there.
    let current = match *last {
        None => first == side,
        Some(newer) => newer == BOTH || newer == side,
    };
    let copy = (!current).then_some(side == DEPTH);
    if copy.is_some() {
        *last = Some(BOTH);
    }
    if writes {
        *last = Some(side);
    }
    copy
}

/// Where a list's uses leave a twin's newer side, from `newer` before it:
/// the copy its prelude makes first (true: into the depth texture), and the
/// newer side after the list.
fn submitted(first: u8, last: Option<u8>, newer: u8) -> (Option<bool>, u8) {
    let copy = match (first, newer) {
        (DEPTH, TWIN) => Some(true),
        (TWIN, DEPTH) => Some(false),
        _ => None,
    };
    let start = if copy.is_some() { BOTH } else { newer };
    (copy, last.unwrap_or(start))
}

/// At submission, for a list's twin uses in order: the copies its prelude
/// records first (the depth texture and whether into it), the twins'
/// newer sides moved on to where the list leaves them.
pub(crate) fn at_submission<'a>(uses: impl Iterator<Item = &'a Use>) -> Vec<(Arc<HostTexture>, bool)> {
    let mut copies = Vec::new();
    for u in uses {
        let Some(Some(twin)) = u.depth.twin.get() else { continue };
        let (to_depth, after) = submitted(u.first, u.last, twin.newer.load(Relaxed));
        let (prior, stored) = (twin.newer.load(Relaxed), twin.stored.load(Relaxed));
        if to_depth == Some(true) && !stored {
            UNCHANGED.fetch_add(1, Relaxed);
        }
        twin.newer.store(after, Relaxed);
        let carried = !u.synced && to_depth.is_none() && prior == TWIN && stored;
        twin.stored.store(after == TWIN && (u.stored || carried), Relaxed);
        if let Some(to_depth) = cleared(to_depth, u.whole, clear_skip()) {
            copies.push((u.depth.clone(), to_depth));
        }
    }
    copies
}

/// Moves a texture to a state for a copy: the barrier, if any.
pub(crate) type Mover<'a> = dyn FnMut(&Arc<HostTexture>, u32) -> Option<D3D12_RESOURCE_BARRIER> + 'a;

impl Twin {
    /// Records into `l` the copy of the newer side into the other: from
    /// the twin into `depth` (`to_depth`) or back. `mv` moves textures to
    /// the copy states (tracked by the list, or from their real states in
    /// a prelude). True if it ran our conversion, which replaces the
    /// list's compute root signature and pipeline.
    pub(crate) fn copy(&self, l: &ID3D12GraphicsCommandList, depth: &Arc<HostTexture>, to_depth: bool, mv: &mut Mover) -> bool {
        let (src, dst) = if to_depth { (COPY_SOURCE, COPY_DEST) } else { (COPY_DEST, COPY_SOURCE) };
        let mut barriers: Vec<_> = self.planes.iter().filter_map(|p| mv(p, src)).collect();
        barriers.extend(mv(depth, dst));
        let at = |r: &ID3D12Resource, sub: u32| D3D12_TEXTURE_COPY_LOCATION {
            // SAFETY: a borrowed pointer the copy does not release.
            pResource: unsafe { std::mem::transmute_copy(r) },
            Type: D3D12_TEXTURE_COPY_TYPE_SUBRESOURCE_INDEX,
            Anonymous: D3D12_TEXTURE_COPY_LOCATION_0 { SubresourceIndex: sub },
        };
        let placed = |f: D3D12_PLACED_SUBRESOURCE_FOOTPRINT| D3D12_TEXTURE_COPY_LOCATION {
            // SAFETY: as above.
            pResource: unsafe { std::mem::transmute_copy(&self.scratch) },
            Type: D3D12_TEXTURE_COPY_TYPE_PLACED_FOOTPRINT,
            Anonymous: D3D12_TEXTURE_COPY_LOCATION_0 { PlacedFootprint: f },
        };
        let converts = self.routes.iter().any(|r| r.depth.is_some());
        let state = |from, to| transition(&self.scratch, D3D12_RESOURCE_STATES(from), D3D12_RESOURCE_STATES(to));
        let mut converted = false;
        // SAFETY: an open list; live resources in the states named (the
        // scratch buffer starts common: its first copy promotes it).
        unsafe {
            if !barriers.is_empty() {
                l.ResourceBarrier(&barriers);
            }
            for r in &self.routes {
                let plane = &self.planes[r.plane].texture;
                if to_depth {
                    l.CopyTextureRegion(&placed(r.twin), 0, 0, 0, &at(plane, r.twin_sub), None);
                } else {
                    l.CopyTextureRegion(&placed(r.depth.unwrap_or(r.twin)), 0, 0, 0, &at(&depth.texture, r.depth_sub), None);
                }
            }
            if converts && let Some(c) = converter() {
                l.ResourceBarrier(&[state(COPY_DEST as i32, UNORDERED_ACCESS as i32)]);
                l.SetComputeRootSignature(&c.root);
                l.SetPipelineState(&c.pso);
                l.SetComputeRootUnorderedAccessView(1, self.scratch.GetGPUVirtualAddress());
                l.SetComputeRootUnorderedAccessView(2, self.scratch.GetGPUVirtualAddress());
                for r in &self.routes {
                    let Some(d) = r.depth else { continue };
                    let (t, d) = (r.twin.Footprint, d);
                    let constants = [t.Width, t.Height, t.RowPitch, r.twin.Offset as u32, d.Footprint.RowPitch, d.Offset as u32, u32::from(to_depth), 0];
                    l.SetComputeRoot32BitConstants(0, 8, constants.as_ptr().cast(), 0);
                    l.Dispatch(t.Width.div_ceil(2).div_ceil(64), t.Height, 1);
                }
                l.ResourceBarrier(&[uav_barrier(), state(UNORDERED_ACCESS as i32, COPY_SOURCE as i32)]);
                converted = true;
            } else {
                l.ResourceBarrier(&[state(COPY_DEST as i32, COPY_SOURCE as i32)]);
            }
            for r in &self.routes {
                let plane = &self.planes[r.plane].texture;
                if to_depth {
                    l.CopyTextureRegion(&at(&depth.texture, r.depth_sub), 0, 0, 0, &placed(r.depth.unwrap_or(r.twin)), None);
                } else {
                    l.CopyTextureRegion(&at(plane, r.twin_sub), 0, 0, 0, &placed(r.twin), None);
                }
            }
            l.ResourceBarrier(&[state(COPY_SOURCE as i32, 0)]);
        }
        COPIES.fetch_add(1, Relaxed);
        if to_depth { &INTO_DEPTH } else { &INTO_TWIN }.fetch_add(1, Relaxed);
        let moved: u64 = self
            .routes
            .iter()
            .map(|r| {
                u64::from(r.twin.Footprint.RowPitch)
                    * u64::from(r.twin.Footprint.Height)
                    * u64::from(r.twin.Footprint.Depth)
            })
            .sum();
        BYTES.fetch_add(moved, Relaxed);
        #[cfg(feature = "diagnostics")]
        {
            use crate::copystat::{C, add};
            // The bytes of the twin's planes the copy moves (the depth
            // texture's side may be half as big: 16-bit depth).
            let bytes: u64 = self
                .routes
                .iter()
                .map(|r| {
                    u64::from(r.twin.Footprint.RowPitch)
                        * u64::from(r.twin.Footprint.Height)
                        * u64::from(r.twin.Footprint.Depth)
                })
                .sum();
            let (copies, moved) = if to_depth {
                (C::TwinToDepth, C::TwinToDepthBytes)
            } else {
                (C::TwinToTwin, C::TwinToTwinBytes)
            };
            add(copies, 1);
            add(moved, bytes);
        }
        converted
    }
}

/// 16-bit depth (unorm, two per dword) to the host's 32-bit float depth
/// plane and back, in the scratch buffer: `u` becomes `u` times
/// `0x37800080` (the title's 1/65535; its tiled-lighting draws rasterise
/// that very float and test it EQUAL), and a float back the nearest `u`.
/// The scratch buffer is bound twice, read through `words` and written
/// through `buf`, because Wine's HLSL compiler (vkd3d-shader) loads only
/// from structured buffers and stores only to raw ones. The reads and the
/// writes are in different parts of it.
const SOURCE: &str = r#"
cbuffer C : register(b0) { uint width; uint height; uint pitch16; uint at16; uint pitch32; uint at32; uint toDepth; uint unused; };
RWStructuredBuffer<uint> words : register(u0);
RWByteAddressBuffer buf : register(u1);
uint depth(uint u) { return asuint((float)u * asfloat(0x37800080)); }
uint u16(uint d) { return (uint)(saturate(asfloat(d)) * 65535.0 + 0.5); }
[numthreads(64, 1, 1)]
void main(uint3 id : SV_DispatchThreadID) {
    uint x = id.x * 2;
    uint y = id.y;
    if (x >= width || y >= height) return;
    uint a = at16 + y * pitch16 + x * 2;
    uint b = at32 + y * pitch32 + x * 4;
    bool two = x + 1 < width;
    if (toDepth != 0) {
        uint pair = words[a / 4];
        buf.Store(b, depth(pair & 0xffff));
        if (two) buf.Store(b + 4, depth(pair >> 16));
    } else {
        uint lo = u16(words[b / 4]);
        uint hi = two ? u16(words[b / 4 + 1]) : 0;
        buf.Store(a, lo | hi << 16);
    }
}
"#;

struct Converter {
    root: ID3D12RootSignature,
    pso: ID3D12PipelineState,
}

// SAFETY: D3D12 objects are free-threaded.
unsafe impl Send for Converter {}
// SAFETY: as above.
unsafe impl Sync for Converter {}

fn converter() -> Option<&'static Converter> {
    static MADE: std::sync::OnceLock<Option<Converter>> = std::sync::OnceLock::new();
    MADE.get_or_init(|| match compile() {
        Ok(c) => Some(c),
        Err(e) => {
            tracing::warn!(error = %e, "no depth conversion: 16-bit depth written through unordered views does not reach the depth texture");
            None
        }
    })
    .as_ref()
}

fn compile() -> windows::core::Result<Converter> {
    use crate::compute::Param;
    let params = [
        Param::Constants { register: 0, count: 8 },
        Param::Uav { register: 0 },
        Param::Uav { register: 1 },
    ];
    let (root, pso) = crate::compute::pipeline(SOURCE, s!("depth"), 0, &params)?;
    Ok(Converter { root, pso })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::backend::HostGpu;

    /// The material-depth frame of Grand Theft Auto VI: written through
    /// unordered views, read as a shader resource in the same list, written
    /// again; read in the next list; tested as depth in a later one; the
    /// next frame writes it through unordered views again.
    #[test]
    fn copies_follow_the_side_each_use_needs() {
        // List 1: unordered write first (the prelude copies the depth side
        // over, newer since the last frame), a read (copy back now), a
        // write again.
        let mut last = Some(TWIN);
        assert_eq!(again(TWIN, &mut last, DEPTH, false), Some(true));
        assert_eq!(last, Some(BOTH));
        assert_eq!(again(TWIN, &mut last, TWIN, true), None);
        assert_eq!(submitted(TWIN, last, DEPTH), (Some(false), TWIN));
        // List 2 only reads: its prelude copies into the depth texture.
        assert_eq!(submitted(DEPTH, None, TWIN), (Some(true), BOTH));
        // List 3 renders to it: nothing to copy, the depth side is newer.
        assert_eq!(submitted(DEPTH, Some(DEPTH), BOTH), (None, DEPTH));
        // A list that starts reading the depth side, unchanged since, does
        // not know the twin is current: a use of the twin copies.
        let mut last = None;
        assert_eq!(again(DEPTH, &mut last, DEPTH, false), None);
        assert_eq!(again(DEPTH, &mut last, TWIN, true), Some(false));
        assert_eq!(last, Some(TWIN));
    }

    /// Probe: whether a colour texture placed on the same heap bytes as a
    /// 32-bit depth texture sees what the depth side holds and the other
    /// way round, under this host (D3D12 leaves it undefined; a twin that
    /// aliases its depth texture would need no copies). Prints the result;
    /// fails only if the host cannot run it.
    #[test]
    fn probe_aliased_twin_sees_depth() {
        let Some(host) = host() else { return };
        let dev = &host.device;
        const W: u32 = 64;
        const H: u32 = 64;
        // SAFETY: plain object creation and recording on our own queue;
        // every resource outlives the wait below.
        unsafe {
            let mut opts = D3D12_FEATURE_DATA_D3D12_OPTIONS::default();
            dev.CheckFeatureSupport(D3D12_FEATURE_D3D12_OPTIONS, std::ptr::from_mut(&mut opts).cast(), size_of_val(&opts) as u32).unwrap();
            if opts.ResourceHeapTier.0 < 2 {
                println!("alias probe: resource heap tier 1, no mixed heap");
                return;
            }
            let ddesc = D3D12_RESOURCE_DESC {
                Dimension: D3D12_RESOURCE_DIMENSION_TEXTURE2D,
                Width: u64::from(W),
                Height: H,
                DepthOrArraySize: 1,
                MipLevels: 1,
                Format: DXGI_FORMAT_R32_TYPELESS,
                SampleDesc: DXGI_SAMPLE_DESC { Count: 1, Quality: 0 },
                Flags: D3D12_RESOURCE_FLAG_ALLOW_DEPTH_STENCIL,
                ..Default::default()
            };
            let cdesc = D3D12_RESOURCE_DESC { Format: DXGI_FORMAT_R32_FLOAT, Flags: D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS, ..ddesc };
            let size = dev.GetResourceAllocationInfo(0, &[ddesc]).SizeInBytes.max(dev.GetResourceAllocationInfo(0, &[cdesc]).SizeInBytes);
            let hdesc = D3D12_HEAP_DESC {
                SizeInBytes: size,
                Properties: D3D12_HEAP_PROPERTIES { Type: D3D12_HEAP_TYPE_DEFAULT, ..Default::default() },
                Alignment: 0,
                Flags: D3D12_HEAP_FLAG_NONE,
            };
            let mut heap: Option<ID3D12Heap> = None;
            dev.CreateHeap(&hdesc, &mut heap).unwrap();
            let heap = heap.unwrap();
            let mut depth: Option<ID3D12Resource> = None;
            let mut colour: Option<ID3D12Resource> = None;
            dev.CreatePlacedResource(&heap, 0, &ddesc, D3D12_RESOURCE_STATE_DEPTH_WRITE, None, &mut depth).unwrap();
            dev.CreatePlacedResource(&heap, 0, &cdesc, D3D12_RESOURCE_STATE_COPY_SOURCE, None, &mut colour).unwrap();
            let (depth, colour) = (depth.unwrap(), colour.unwrap());
            let rb_heap = D3D12_HEAP_PROPERTIES { Type: D3D12_HEAP_TYPE_READBACK, ..Default::default() };
            let up_heap = D3D12_HEAP_PROPERTIES { Type: D3D12_HEAP_TYPE_UPLOAD, ..Default::default() };
            let bytes = u64::from(W * 4 * H);
            let bdesc = D3D12_RESOURCE_DESC {
                Dimension: D3D12_RESOURCE_DIMENSION_BUFFER,
                Width: bytes,
                Height: 1,
                DepthOrArraySize: 1,
                MipLevels: 1,
                SampleDesc: DXGI_SAMPLE_DESC { Count: 1, Quality: 0 },
                Layout: D3D12_TEXTURE_LAYOUT_ROW_MAJOR,
                ..Default::default()
            };
            let mut rb: Option<ID3D12Resource> = None;
            let mut up: Option<ID3D12Resource> = None;
            dev.CreateCommittedResource(&rb_heap, D3D12_HEAP_FLAG_NONE, &bdesc, D3D12_RESOURCE_STATE_COPY_DEST, None, &mut rb).unwrap();
            dev.CreateCommittedResource(&up_heap, D3D12_HEAP_FLAG_NONE, &bdesc, D3D12_RESOURCE_STATE_GENERIC_READ, None, &mut up).unwrap();
            let (rb, up) = (rb.unwrap(), up.unwrap());
            let mut p = std::ptr::null_mut();
            up.Map(0, None, Some(&mut p)).unwrap();
            std::slice::from_raw_parts_mut(p.cast::<f32>(), (W * H) as usize).fill(0.5);
            let dsv_heap: ID3D12DescriptorHeap = dev
                .CreateDescriptorHeap(&D3D12_DESCRIPTOR_HEAP_DESC { Type: D3D12_DESCRIPTOR_HEAP_TYPE_DSV, NumDescriptors: 1, ..Default::default() })
                .unwrap();
            let dsv = dsv_heap.GetCPUDescriptorHandleForHeapStart();
            let vdesc = D3D12_DEPTH_STENCIL_VIEW_DESC { Format: DXGI_FORMAT_D32_FLOAT, ViewDimension: D3D12_DSV_DIMENSION_TEXTURE2D, ..Default::default() };
            dev.CreateDepthStencilView(&depth, Some(&vdesc), dsv);
            let queue: ID3D12CommandQueue = dev.CreateCommandQueue(&D3D12_COMMAND_QUEUE_DESC::default()).unwrap();
            let alloc: ID3D12CommandAllocator = dev.CreateCommandAllocator(D3D12_COMMAND_LIST_TYPE_DIRECT).unwrap();
            let list: ID3D12GraphicsCommandList = dev.CreateCommandList(0, D3D12_COMMAND_LIST_TYPE_DIRECT, &alloc, None).unwrap();
            let fence: ID3D12Fence = dev.CreateFence(0, D3D12_FENCE_FLAG_NONE).unwrap();
            let fp = D3D12_PLACED_SUBRESOURCE_FOOTPRINT {
                Offset: 0,
                Footprint: D3D12_SUBRESOURCE_FOOTPRINT { Format: DXGI_FORMAT_R32_FLOAT, Width: W, Height: H, Depth: 1, RowPitch: W * 4 },
            };
            let tex = |r: &ID3D12Resource| D3D12_TEXTURE_COPY_LOCATION {
                pResource: std::mem::transmute_copy(r),
                Type: D3D12_TEXTURE_COPY_TYPE_SUBRESOURCE_INDEX,
                Anonymous: D3D12_TEXTURE_COPY_LOCATION_0 { SubresourceIndex: 0 },
            };
            let buf = |r: &ID3D12Resource, f: D3D12_SUBRESOURCE_FOOTPRINT| D3D12_TEXTURE_COPY_LOCATION {
                pResource: std::mem::transmute_copy(r),
                Type: D3D12_TEXTURE_COPY_TYPE_PLACED_FOOTPRINT,
                Anonymous: D3D12_TEXTURE_COPY_LOCATION_0 { PlacedFootprint: D3D12_PLACED_SUBRESOURCE_FOOTPRINT { Offset: 0, Footprint: f } },
            };
            let st = |r: &ID3D12Resource, a: D3D12_RESOURCE_STATES, b: D3D12_RESOURCE_STATES| transition(r, a, b);
            let aliasing = || D3D12_RESOURCE_BARRIER { Type: D3D12_RESOURCE_BARRIER_TYPE_ALIASING, ..Default::default() };
            let run = |n: u64| {
                list.Close().unwrap();
                queue.ExecuteCommandLists(&[Some(list.cast().unwrap())]);
                queue.Signal(&fence, n).unwrap();
                while fence.GetCompletedValue() < n {
                    std::thread::yield_now();
                }
            };
            let read = || {
                let mut q = std::ptr::null_mut();
                rb.Map(0, None, Some(&mut q)).unwrap();
                let v = std::slice::from_raw_parts(q.cast::<f32>(), (W * H) as usize).to_vec();
                rb.Unmap(0, None);
                v
            };
            use windows::core::Interface;
            // Depth written by a clear, read through the colour texture.
            list.ClearDepthStencilView(dsv, D3D12_CLEAR_FLAG_DEPTH, 0.25, 0, None);
            list.ResourceBarrier(&[aliasing()]);
            list.CopyTextureRegion(&buf(&rb, fp.Footprint), 0, 0, 0, &tex(&colour), None);
            run(1);
            let v = read();
            let same = v.iter().filter(|&&x| x == 0.25).count();
            println!("alias probe: depth clear 0.25 seen through colour in {same} of {} texels (first {:?})", v.len(), &v[..4]);
            // Colour written by a copy, read through the depth texture.
            alloc.Reset().unwrap();
            list.Reset(&alloc, None).unwrap();
            list.ResourceBarrier(&[st(&colour, D3D12_RESOURCE_STATE_COPY_SOURCE, D3D12_RESOURCE_STATE_COPY_DEST)]);
            list.CopyTextureRegion(&tex(&colour), 0, 0, 0, &buf(&up, fp.Footprint), None);
            list.ResourceBarrier(&[aliasing(), st(&depth, D3D12_RESOURCE_STATE_DEPTH_WRITE, D3D12_RESOURCE_STATE_COPY_SOURCE)]);
            list.CopyTextureRegion(&buf(&rb, fp.Footprint), 0, 0, 0, &tex(&depth), None);
            run(2);
            let v = read();
            let same = v.iter().filter(|&&x| x == 0.5).count();
            println!("alias probe: colour 0.5 seen through depth in {same} of {} texels (first {:?})", v.len(), &v[..4]);
            // The clear read through the depth texture itself.
            alloc.Reset().unwrap();
            list.Reset(&alloc, None).unwrap();
            list.ResourceBarrier(&[st(&depth, D3D12_RESOURCE_STATE_COPY_SOURCE, D3D12_RESOURCE_STATE_DEPTH_WRITE)]);
            list.ClearDepthStencilView(dsv, D3D12_CLEAR_FLAG_DEPTH, 0.25, 0, None);
            list.ResourceBarrier(&[st(&depth, D3D12_RESOURCE_STATE_DEPTH_WRITE, D3D12_RESOURCE_STATE_COPY_SOURCE)]);
            list.CopyTextureRegion(&buf(&rb, fp.Footprint), 0, 0, 0, &tex(&depth), None);
            run(3);
            let v = read();
            println!("alias probe: depth clear 0.25 seen through depth in {} texels", v.iter().filter(|&&x| x == 0.25).count());
            // Then the colour side again, after that clear was copied out.
            alloc.Reset().unwrap();
            list.Reset(&alloc, None).unwrap();
            list.ResourceBarrier(&[aliasing(), st(&colour, D3D12_RESOURCE_STATE_COPY_DEST, D3D12_RESOURCE_STATE_COPY_SOURCE)]);
            list.CopyTextureRegion(&buf(&rb, fp.Footprint), 0, 0, 0, &tex(&colour), None);
            run(4);
            let v = read();
            println!("alias probe: after it, colour sees 0.25 in {} texels", v.iter().filter(|&&x| x == 0.25).count());
            // A buffer copy into depth, read through colour.
            std::slice::from_raw_parts_mut(p.cast::<f32>(), (W * H) as usize).fill(0.75);
            alloc.Reset().unwrap();
            list.Reset(&alloc, None).unwrap();
            list.ResourceBarrier(&[st(&depth, D3D12_RESOURCE_STATE_COPY_SOURCE, D3D12_RESOURCE_STATE_COPY_DEST)]);
            list.CopyTextureRegion(&tex(&depth), 0, 0, 0, &buf(&up, fp.Footprint), None);
            list.ResourceBarrier(&[aliasing()]);
            list.CopyTextureRegion(&buf(&rb, fp.Footprint), 0, 0, 0, &tex(&colour), None);
            run(5);
            let v = read();
            println!("alias probe: buffer copy 0.75 into depth seen through colour in {} texels", v.iter().filter(|&&x| x == 0.75).count());
        }
    }

    /// Probe, run by hand under `xctrace record --attach`: named command
    /// lists of copies for `VICTORY_LABEL_PROBE_S` seconds (5), to see in a
    /// Metal System Trace whether D3DMetal labels their command buffers or
    /// encoders with `SetName`'s text ("victory-label-probe N").
    #[test]
    #[ignore = "a probe for a trace recorded by hand"]
    fn probe_list_names_reach_metal() {
        use windows::core::{HSTRING, Interface};
        let Some(host) = host() else { return };
        let dev = &host.device;
        let secs: u64 = std::env::var("VICTORY_LABEL_PROBE_S").ok().and_then(|s| s.parse().ok()).unwrap_or(5);
        // SAFETY: plain creation and recording on our own queue; the
        // buffers outlive every wait.
        unsafe {
            let heap = D3D12_HEAP_PROPERTIES { Type: D3D12_HEAP_TYPE_DEFAULT, ..Default::default() };
            let desc = D3D12_RESOURCE_DESC {
                Dimension: D3D12_RESOURCE_DIMENSION_BUFFER,
                Width: 64 << 20,
                Height: 1,
                DepthOrArraySize: 1,
                MipLevels: 1,
                SampleDesc: DXGI_SAMPLE_DESC { Count: 1, Quality: 0 },
                Layout: D3D12_TEXTURE_LAYOUT_ROW_MAJOR,
                ..Default::default()
            };
            let mut a: Option<ID3D12Resource> = None;
            let mut b: Option<ID3D12Resource> = None;
            dev.CreateCommittedResource(&heap, D3D12_HEAP_FLAG_NONE, &desc, D3D12_RESOURCE_STATE_COMMON, None, &mut a).unwrap();
            dev.CreateCommittedResource(&heap, D3D12_HEAP_FLAG_NONE, &desc, D3D12_RESOURCE_STATE_COMMON, None, &mut b).unwrap();
            let (a, b) = (a.unwrap(), b.unwrap());
            let queue: ID3D12CommandQueue = dev.CreateCommandQueue(&D3D12_COMMAND_QUEUE_DESC::default()).unwrap();
            let _ = queue.SetName(&HSTRING::from("victory-label-probe queue"));
            let alloc: ID3D12CommandAllocator = dev.CreateCommandAllocator(D3D12_COMMAND_LIST_TYPE_DIRECT).unwrap();
            let fence: ID3D12Fence = dev.CreateFence(0, D3D12_FENCE_FLAG_NONE).unwrap();
            let end = std::time::Instant::now() + std::time::Duration::from_secs(secs);
            let mut n = 0u64;
            while std::time::Instant::now() < end {
                n += 1;
                alloc.Reset().unwrap();
                let list: ID3D12GraphicsCommandList = dev.CreateCommandList(0, D3D12_COMMAND_LIST_TYPE_DIRECT, &alloc, None).unwrap();
                let _ = list.SetName(&HSTRING::from(format!("victory-label-probe {n}")));
                list.CopyBufferRegion(&b, 0, &a, 0, 64 << 20);
                list.Close().unwrap();
                queue.ExecuteCommandLists(&[Some(list.cast().unwrap())]);
                queue.Signal(&fence, n).unwrap();
                while fence.GetCompletedValue() < n {
                    std::thread::yield_now();
                }
            }
            println!("label probe: {n} named lists");
        }
    }

    /// Run H's round trip: depth rendered, then an unordered use, then a
    /// depth use. Counted as a write, the unordered use makes the twin
    /// newer and the depth use copies it back; one that stores nothing
    /// (`VICTORY_TWIN_UAV_READS`) leaves both sides current.
    #[test]
    fn unordered_reads_do_not_copy_back() {
        let mut last = Some(DEPTH);
        assert_eq!(again(DEPTH, &mut last, TWIN, true), Some(false));
        assert_eq!(again(DEPTH, &mut last, DEPTH, false), Some(true), "the copy back");
        let mut last = Some(DEPTH);
        assert_eq!(again(DEPTH, &mut last, TWIN, false), Some(false));
        assert_eq!(last, Some(BOTH));
        assert_eq!(again(DEPTH, &mut last, DEPTH, false), None, "nothing to copy back");
        // Across lists: a list that only reads the twin leaves the newer
        // side where it was.
        assert_eq!(submitted(TWIN, None, DEPTH), (Some(false), BOTH));
        assert_eq!(submitted(DEPTH, None, BOTH), (None, BOTH));
    }

    /// A clear overwrites a texture only with every plane it has and one
    /// subresource per plane; only then is a copy into depth left out, and
    /// only with the switch on.
    #[test]
    fn whole_clears_leave_out_the_copy_into_depth() {
        assert!(covers(1, 1, 1));
        assert!(!covers(1, 2, 2), "depth only, stencil kept");
        assert!(covers(3, 2, 2));
        assert!(!covers(3, 1, 4), "mips or slices the view may not cover");
        assert!(!covers(0, 1, 1), "not a clear");
        assert_eq!(cleared(Some(true), true, false), Some(true));
        assert_eq!(cleared(Some(true), true, true), None);
        assert_eq!(cleared(Some(true), false, true), Some(true));
        assert_eq!(cleared(Some(false), true, true), Some(false), "a copy into the twin stays");
        assert_eq!(cleared(None, true, true), None);
        // Left out, the list still leaves the depth side newer.
        let (copy, after) = submitted(DEPTH, Some(DEPTH), TWIN);
        assert_eq!((cleared(copy, true, true), after), (None, DEPTH));
    }

    /// A list that writes a depth texture through its twin and then reads
    /// its depth side: its prelude owes a copy into the twin first. Every
    /// barrier the prelude and the list issue starts from the state the
    /// texture is in, as the debug layer checks (run light21: the prelude's
    /// copy started from the states the list leaves).
    #[test]
    fn prelude_copies_start_from_the_real_states() {
        use crate::host::tests::{issued, replay};
        use windows::core::Interface;
        let Some(host) = host() else {
            return; // no host GPU here
        };
        let desc = D3D12_RESOURCE_DESC {
            Dimension: D3D12_RESOURCE_DIMENSION_TEXTURE2D,
            Width: 64,
            Height: 64,
            DepthOrArraySize: 1,
            MipLevels: 1,
            Format: DXGI_FORMAT_R16_TYPELESS,
            SampleDesc: DXGI_SAMPLE_DESC { Count: 1, Quality: 0 },
            Flags: D3D12_RESOURCE_FLAG_ALLOW_DEPTH_STENCIL,
            ..Default::default()
        };
        let heap = D3D12_HEAP_PROPERTIES { Type: D3D12_HEAP_TYPE_DEFAULT, ..Default::default() };
        let mut texture: Option<ID3D12Resource> = None;
        // SAFETY: plain creation; our out pointer.
        unsafe { host.device.CreateCommittedResource(&heap, D3D12_HEAP_FLAG_NONE, &desc, D3D12_RESOURCE_STATE_DEPTH_WRITE, None, &mut texture) }.unwrap();
        let depth = Arc::new(HostTexture::new(texture.unwrap(), DEPTH_WRITE, Vec::new()));
        let twin = depth.twin.get_or_init(|| make(53, &depth.texture)).as_ref().unwrap();
        let list = host.new_list().unwrap();
        issued();
        assert!(route(&list, &depth, twin, UNORDERED_ACCESS));
        const PIXEL_SHADER_RESOURCE: u32 = 0x80;
        assert!(!route(&list, &depth, twin, PIXEL_SHADER_RESOURCE));
        if let Some(b) = list.move_texture(&depth, PIXEL_SHADER_RESOURCE) {
            // SAFETY: an open list and a live texture.
            unsafe { list.list.ResourceBarrier(&[b]) };
        }
        list.close();
        let recorded = issued();
        host.execute(&[&list]);
        // The GPU runs the prelude, then the list.
        let mut run = issued();
        run.extend(recorded);
        let key = |t: &HostTexture| t.texture.as_raw() as usize;
        let mut states = std::collections::HashMap::from([(key(&depth), DEPTH_WRITE), (key(&twin.planes[0]), 0)]);
        replay(&run, &mut states).unwrap();
        // What the list left is what the next prelude starts from.
        assert_eq!(states[&key(&depth)], depth.state.load(Relaxed));
        assert_eq!(states[&key(&twin.planes[0])], twin.planes[0].state.load(Relaxed));
    }
}
