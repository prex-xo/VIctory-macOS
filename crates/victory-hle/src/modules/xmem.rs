//! `xmem.dll`: console memory, served from host memory. The GPU side works
//! from CPU addresses (unified memory), so memory types and bandwidth
//! classes change nothing here; what constrains addresses (alignment,
//! 2 MiB pages, address-bit limits) is honoured.
#![allow(non_snake_case)]
// Title code calls these through its import table, passing pointers the real
// API would trust.
#![allow(clippy::not_unsafe_ptr_arg_deref)]

use std::alloc::{Layout, alloc, dealloc};
use std::collections::{BTreeMap, HashMap};
use std::ffi::c_void;
use std::sync::atomic::{AtomicBool, AtomicU64, AtomicUsize, Ordering};
use std::sync::{
    Arc, LazyLock, Mutex, MutexGuard, OnceLock, RwLock, RwLockReadGuard, RwLockWriteGuard,
};

use windows_sys::Win32::Foundation::{
    CloseHandle, ERROR_COMMITMENT_LIMIT, HANDLE, INVALID_HANDLE_VALUE,
};
use windows_sys::Win32::System::Memory::{
    CreateFileMappingW, MEM_ADDRESS_REQUIREMENTS, MEM_COMMIT, MEM_EXTENDED_PARAMETER,
    MEM_PRESERVE_PLACEHOLDER, MEM_RELEASE, MEM_REPLACE_PLACEHOLDER, MEM_RESERVE,
    MEM_RESERVE_PLACEHOLDER, MEMORY_BASIC_INFORMATION, MEMORY_MAPPED_VIEW_ADDRESS, MapViewOfFile3,
    MemExtendedParameterAddressRequirements, PAGE_NOACCESS, PAGE_READWRITE, SEC_COMMIT,
    SEC_RESERVE, UnmapViewOfFile2, VirtualAlloc, VirtualAlloc2, VirtualFree, VirtualQueryEx,
};
use windows_sys::Win32::System::ProcessStatus::{
    K32GetPerformanceInfo, K32GetProcessMemoryInfo, PERFORMANCE_INFORMATION,
    PROCESS_MEMORY_COUNTERS,
};
use windows_sys::Win32::System::Threading::GetCurrentProcess;
use windows_sys::core::BOOL;

use crate::{HleModule, hle_export};

const MIB: usize = 1 << 20;

/// Whether we run under Wine (its ntdll exports `wine_get_version`).
pub(crate) fn on_wine() -> bool {
    static WINE: LazyLock<bool> = LazyLock::new(|| {
        use windows_sys::Win32::System::LibraryLoader::{GetModuleHandleA, GetProcAddress};
        // SAFETY: plain lookups in ntdll, always loaded.
        unsafe {
            let ntdll = GetModuleHandleA(c"ntdll.dll".as_ptr().cast());
            !ntdll.is_null() && GetProcAddress(ntdll, c"wine_get_version".as_ptr().cast()).is_some()
        }
    });
    *WINE
}

/// Standard memory the title has handed to the system.
static TRANSFERRED: AtomicUsize = AtomicUsize::new(0);

// Console additions to the host's flags.
/// 2 MiB pages; the host reads this bit as large pages, which need a
/// privilege.
const PAGES_2M: u32 = 0x2000_0000;
/// With `PAGES_2M`: 64 KiB pages; the host reads it as AWE memory.
const PAGES_64K_BIT: u32 = 0x40_0000;
const PAGES_64K: u32 = PAGES_2M | PAGES_64K_BIT;
/// The host's protection constants and `PAGE_GUARD`. Above them the console
/// adds GPU access bits (0x800-0x40000). Write-combining and uncached are
/// dropped too: our GPU side reads this memory through the CPU.
const HOST_PROTECTION: u32 = 0x1ff;
/// XMem flags of memory both processors reach: all of ours.
const GRAPHICS: u64 = 3;
/// XMem flag bit of 32-bit graphics memory.
const ADDRESS_32BIT: u64 = 4;
/// The host's write watching (not in windows-sys' memory module).
const MEM_WRITE_WATCH: u32 = 0x20_0000;

/// The addresses XMem `flags` allow, first and last: the low 4 GiB for
/// 32-bit memory; else from 4 GiB (kept for 32-bit requests, which the
/// title's big pools would otherwise crowd out) up to 46 down to 40 bits
/// from the field at bit 12, or 40 bits without one: the console keeps
/// title memory there, and its ray-tracing structures must be (Grand Theft
/// Auto VI's BVH builder refuses one above; the title asks for its GPU heap
/// with no field). Decision 0001.
fn address_range(flags: u64) -> (usize, usize) {
    let bits = match (flags & ADDRESS_32BIT, flags >> 12 & 7) {
        (0, 0) => 40,
        (0, field) => 47 - field,
        _ => return (0, u32::MAX as usize),
    };
    (1 << 32, (1 << bits) - 1)
}

pub extern "system" fn XMemVirtualAlloc(
    base: *mut c_void,
    size: usize,
    kind: u32,
    flags: u64,
    protect: u32,
) -> *mut c_void {
    let (lowest, highest) = address_range(flags);
    let mut requirements = MEM_ADDRESS_REQUIREMENTS {
        LowestStartingAddress: lowest as *mut c_void,
        HighestEndingAddress: highest as *mut c_void,
        Alignment: if kind & (PAGES_2M | PAGES_64K_BIT) == PAGES_2M {
            2 * MIB
        } else {
            0
        },
    };
    let mut protect = match protect & HOST_PROTECTION {
        0 => PAGE_NOACCESS,
        p => p,
    };
    // A reservation of 64 KiB pages takes physical pages later
    // (`XMemMapPhysicalPages`); on the host only a placeholder can.
    // shortcut: committing part of such a reservation later would fail
    // (Grand Theft Auto VI's 23 calls never do); replace that part of the
    // placeholder then.
    let placeholder = kind & PAGES_64K == PAGES_64K && kind & MEM_COMMIT == 0;
    if placeholder {
        if kind & MEM_REPLACE_PLACEHOLDER != 0 {
            // Part of a placeholder the title split: it stays one.
            return base;
        }
        protect = PAGE_NOACCESS;
    }
    let mut kind = kind & !(PAGES_2M | PAGES_64K_BIT);
    if placeholder {
        kind |= MEM_RESERVE | MEM_RESERVE_PLACEHOLDER;
    }
    let mut parameter = MEM_EXTENDED_PARAMETER::default();
    parameter.Anonymous1._bitfield = MemExtendedParameterAddressRequirements as u64;
    parameter.Anonymous2.Pointer = (&raw mut requirements).cast();
    // Address requirements apply to reservations at an address of our choice.
    let constrained = base.is_null();
    if constrained {
        kind |= MEM_RESERVE;
        // The GPU side copies only the pages the title wrote since its last
        // copy (`victory_gpu`, decision 0007); placeholders cannot be
        // watched (their physical-page views are copied whole). Nor is
        // anything under Rosetta, where Wine's watch faults on first writes
        // Rosetta cannot resume (`victory_gpu::fault_tracking`).
        if kind & MEM_RESERVE_PLACEHOLDER == 0 && victory_gpu::fault_tracking() {
            kind |= MEM_WRITE_WATCH;
        }
    }
    // SAFETY: the title's own arguments, cleaned of console bits; the
    // parameter points at `requirements`, which outlives the call.
    let allocate = |parameter: &mut MEM_EXTENDED_PARAMETER| unsafe {
        VirtualAlloc2(
            GetCurrentProcess(),
            base,
            size,
            kind,
            protect,
            if constrained {
                parameter
            } else {
                std::ptr::null_mut()
            },
            constrained.into(),
        )
    };
    let mut at = allocate(&mut parameter);
    // A 32-bit request that found no room below 4 GiB: the guards there
    // make way for it.
    if at.is_null() && constrained && highest < FOUR_GIB {
        at = low_guards::retry(size, || allocate(&mut parameter));
    }
    if at.is_null() {
        let error = std::io::Error::last_os_error();
        if kind & MEM_COMMIT != 0 {
            commit_refused(size, "virtual memory", error);
        } else {
            tracing::warn!(
                size,
                kind = format_args!("{kind:#x}"),
                flags = format_args!("{flags:#x}"),
                %error,
                "XMemVirtualAlloc refused"
            );
        }
    }
    // Committed title memory: aliased zero-copy where it can be, else
    // counted by why not (the frame report: what host copies refresh).
    if !at.is_null() && kind & MEM_COMMIT != 0 {
        use victory_gpu::Unaliased as U;
        let why = if !victory_gpu::alias_buffers() {
            Some(U::Off)
        } else if kind & MEM_RESERVE_PLACEHOLDER != 0 {
            Some(U::Placeholder)
        } else if protect != PAGE_READWRITE {
            Some(U::Protection)
        } else if !(at as usize).is_multiple_of(PAGE) || size < PAGE || !size.is_multiple_of(PAGE) {
            Some(U::Unaligned)
        } else if !victory_gpu::alias_region(at as usize, size, false) {
            Some(U::Failed)
        } else {
            None
        };
        if let Some(why) = why {
            victory_gpu::note_unaliased(size, why);
        }
    }
    if base.is_null() {
        tracing::debug!(
            size,
            kind = format_args!("{kind:#x}"),
            flags = format_args!("{flags:#x}"),
            at = format_args!("{at:p}"),
            "XMemVirtualAlloc"
        );
    }
    at
}

/// `MEMORY_BASIC_INFORMATION` (the console's first 0x30 bytes are the
/// same), then the XMem flags and tag.
#[repr(C)]
pub struct BasicInformation {
    host: MEMORY_BASIC_INFORMATION,
    flags: u64,
    tag_code: u32,
    tag: [u8; 4],
}

// shortcut: every committed page reports graphics memory and no tag, since
// all of it is GPU-reachable here (Grand Theft Auto VI's BVH builder, the
// only caller, wants graphics memory); remember the flags of
// `XMemVirtualAlloc` regions if a title tells CPU-only memory apart.
pub extern "system" fn XMemVirtualQuery(
    process: HANDLE,
    address: *const c_void,
    out: *mut BasicInformation,
) -> BOOL {
    let size = size_of::<MEMORY_BASIC_INFORMATION>();
    // SAFETY: the title passes room for the whole structure; the host
    // fails on a null one.
    unsafe {
        if VirtualQueryEx(process, address, out.cast(), size) == 0 {
            return 0;
        }
        let info = &mut *out;
        info.flags = if info.host.State == MEM_COMMIT {
            GRAPHICS
        } else {
            0
        };
        info.tag_code = 0;
        info.tag = [0; 4];
    }
    1
}

// `XMemAlloc` attributes: alignment as log2 (5 bits at 24, 4 bytes and up
// from 2), then the memory type, then bit 33: a 32-bit address.
const ALIGNMENT_SHIFT: u64 = 24;
const ALLOC_32BIT: u64 = 1 << 33;
/// Our block header: the block's size and alignment, just below the
/// pointer handed out.
const HEADER: usize = 2 * size_of::<usize>();

fn layout(size: usize, attributes: u64) -> Option<Layout> {
    // An alignment of at least 16 leaves room for the header.
    let align = 1usize << (attributes >> ALIGNMENT_SHIFT & 0x1f).max(4);
    Layout::from_size_align(size.checked_add(align)?, align).ok()
}

type AllocRoutine = extern "system" fn(usize, u64) -> *mut c_void;
type FreeRoutine = extern "system" fn(*mut c_void, u64);

/// The allocation and free routines the title's exe exports as
/// `__XMemAllocationHooks__` (Grand Theft Auto VI does): `XMemAlloc` and
/// `XMemFree` call them, for the title's own calls and its middleware's;
/// the `…Default` exports do not.
fn hooks() -> [usize; 2] {
    static AT: std::sync::OnceLock<usize> = std::sync::OnceLock::new();
    let at = match AT.get() {
        Some(&at) => at,
        // Looked up once the title is mapped, which is before it runs.
        None => match crate::runtime::main_image() {
            Some(base) => *AT.get_or_init(|| {
                use windows_sys::Win32::System::LibraryLoader::GetProcAddress;
                // SAFETY: a mapped image and a NUL-terminated name.
                let at = unsafe {
                    GetProcAddress(base as _, c"__XMemAllocationHooks__".as_ptr().cast())
                }
                .map_or(0, |f| f as usize);
                tracing::info!(found = at != 0, "xmem allocation hooks");
                at
            }),
            None => 0,
        },
    };
    if at == 0 {
        return [0; 2];
    }
    // SAFETY: the export is two routine pointers.
    unsafe { (at as *const [usize; 2]).read() }
}

/// The title's general allocator, for middleware too: its hook if it has
/// one, else [`XMemAllocDefault`].
pub extern "system" fn XMemAlloc(size: usize, attributes: u64) -> *mut c_void {
    match hooks()[0] {
        0 => XMemAllocDefault(size, attributes),
        hook => {
            // SAFETY: the title's own routine of this type.
            let alloc: AllocRoutine = unsafe { std::mem::transmute(hook) };
            alloc(size, attributes)
        }
    }
}

/// See [`XMemAlloc`].
pub extern "system" fn XMemFree(address: *mut c_void, attributes: u64) {
    match hooks()[1] {
        0 => XMemFreeDefault(address, attributes),
        hook => {
            // SAFETY: the title's own routine of this type.
            let free: FreeRoutine = unsafe { std::mem::transmute(hook) };
            free(address, attributes)
        }
    }
}

/// The system allocator behind [`XMemAlloc`].
pub extern "system" fn XMemAllocDefault(size: usize, attributes: u64) -> *mut c_void {
    if attributes & ALLOC_32BIT != 0 {
        return XMemVirtualAlloc(
            std::ptr::null_mut(),
            size,
            MEM_RESERVE | MEM_COMMIT,
            GRAPHICS | ADDRESS_32BIT,
            PAGE_READWRITE,
        );
    }
    let Some(layout) = layout(size, attributes) else {
        return std::ptr::null_mut();
    };
    // shortcut: the host heap's addresses lie above 2^40, unlike the
    // console's; nothing seen builds ray-tracing structures in this memory
    // (Grand Theft Auto VI uses its GPU heap). Sub-allocate from a
    // reservation below 2^40 if one does.
    // SAFETY: the layout has a non-zero size.
    let block = unsafe { alloc(layout) };
    if block.is_null() {
        tracing::warn!(size, "XMemAllocDefault: the host heap is out of memory");
        return block.cast();
    }
    // SAFETY: the block holds the alignment (at least the header) and then
    // `size` bytes.
    unsafe {
        let user = block.add(layout.align());
        user.sub(HEADER)
            .cast::<[usize; 2]>()
            .write([layout.size(), layout.align()]);
        user.cast()
    }
}

/// Frees what [`XMemAllocDefault`] returned, given the same attributes.
pub extern "system" fn XMemFreeDefault(address: *mut c_void, attributes: u64) {
    if address.is_null() {
        return;
    }
    // SAFETY: the address came from `XMemAlloc` with these attributes: a
    // whole reservation, or a header-prefixed block.
    unsafe {
        if attributes & ALLOC_32BIT != 0 {
            victory_gpu::unalias_region(address as usize);
            let _going = victory_gpu::title_memory_unmapping(address as usize, 0);
            VirtualFree(address, 0, MEM_RELEASE);
            return;
        }
        let user = address.cast::<u8>();
        let [size, align] = user.sub(HEADER).cast::<[usize; 2]>().read();
        dealloc(
            user.sub(align),
            Layout::from_size_align_unchecked(size, align),
        );
    }
}

/// A retail console has no tools memory.
pub extern "system" fn XMemIsToolsMemAvailable() -> BOOL {
    0
}

// Physical pages: on the console, 64 KiB frames a title maps into its own
// reservations, at more than one address if it likes. Here each allocation
// is a pagefile-backed section, its pages are 64 KiB slices of it, and a
// mapping is a view of consecutive slices in place of a host placeholder.

const PAGE: usize = 64 << 10;
/// Minimum starting address for our own whole-section and fallback views under
/// Wine (`2 TiB`): above the title's 40-bit address space (`< 1 TiB`) and
/// above Metal's `~1 TiB` host GPU heaps on macOS, so an unpatched Wine's
/// `try_map_free_area` does not step 64 KiB at a time across `< 4 GiB` or
/// gigabytes of host mappings while holding `virtual_mutex`.
const OWN_VIEW_FLOOR: usize = 2 << 40;

/// Maps `[offset, offset + size)` of `section` writable for our own CPU writes
/// (DirectStorage destinations), starting at or above [`OWN_VIEW_FLOOR`] under
/// Wine and falling back to [`MapViewOfFile`] if the host refuses the hint.
unsafe fn map_own_view(section: HANDLE, offset: u64, size: usize) -> MEMORY_MAPPED_VIEW_ADDRESS {
    use windows_sys::Win32::System::Memory::{FILE_MAP_WRITE, MapViewOfFile};
    if on_wine() {
        let mut requirements = MEM_ADDRESS_REQUIREMENTS {
            LowestStartingAddress: OWN_VIEW_FLOOR as *mut c_void,
            HighestEndingAddress: std::ptr::null_mut(),
            Alignment: 0,
        };
        let mut parameter = MEM_EXTENDED_PARAMETER::default();
        parameter.Anonymous1._bitfield = MemExtendedParameterAddressRequirements as u64;
        parameter.Anonymous2.Pointer = (&raw mut requirements).cast();
        // SAFETY: caller owns `section`; `parameter` points at `requirements`,
        // which outlives the call.
        let view = unsafe {
            MapViewOfFile3(
                section,
                GetCurrentProcess(),
                std::ptr::null(),
                offset,
                size,
                0,
                PAGE_READWRITE,
                &mut parameter,
                1,
            )
        };
        if !view.Value.is_null() {
            return view;
        }
    }
    // SAFETY: caller owns `section`.
    unsafe {
        MapViewOfFile(
            section,
            FILE_MAP_WRITE,
            (offset >> 32) as u32,
            offset as u32,
            size,
        )
    }
}

struct Section {
    handle: HANDLE,
    size: usize,
    /// The pages as GPU memory (`VICTORY_PHYSICAL_TILES=1`): the title's
    /// views are remapped onto it, and copies of title buffers on them read
    /// it in place (`victory_gpu::physical_retiled`). The file still
    /// supplies Wine's placeholder-view metadata.
    shared: Option<victory_gpu::PhysicalHeap>,
    /// Our own view of the whole section, mapped the first time a
    /// DirectStorage request writes its pages and kept while the section
    /// lives (0: it could not be mapped). A view per request cost a map
    /// and an unmap, and under Wine each holds the process's one
    /// virtual-memory lock across a wineserver request: every page fault
    /// and protection change on other threads waits for it (#50).
    whole: OnceLock<usize>,
    /// Its pages committed so far, a bit each ([`Section::commit_shown`]).
    committed: Box<[AtomicU64]>,
}

/// Whether a mapping asks the host only for what it needs: no split of a
/// placeholder that is already the view's own (left by unmapping a view of
/// that exact size), and no commit of section pages already committed
/// (`VICTORY_MAP_LEAN=0`: split and commit every time, as before).
fn lean() -> bool {
    #[cfg(test)]
    if let Some(on) = tests::TEST_LEAN.with(std::cell::Cell::get) {
        return on;
    }
    LEAN.load(Ordering::Relaxed)
}

static LEAN: LazyLock<AtomicBool> = LazyLock::new(|| {
    AtomicBool::new(std::env::var_os("VICTORY_MAP_LEAN").is_none_or(|v| v != "0"))
});

/// Whether contiguous physical and virtual 64 KiB pages are coalesced into
/// multi-page `MapViewOfFile3` views (`VICTORY_MAP_RUNS=1`). Off by default:
/// on Xbox each 64 KiB page has an independent PTE, whereas unmapping or
/// remapping a single 64 KiB page inside a coalesced multi-page host view has
/// to `UnmapViewOfFile2` the whole view before re-mapping its outer halves,
/// briefly unmapping sibling pages that streaming threads may be reading.
fn map_runs() -> bool {
    #[cfg(test)]
    if let Some(on) = tests::TEST_MAP_RUNS.with(std::cell::Cell::get) {
        return on;
    }
    MAP_RUNS.load(Ordering::Relaxed)
}

static MAP_RUNS: LazyLock<AtomicBool> = LazyLock::new(|| {
    let on = matches!(
        std::env::var("VICTORY_MAP_RUNS").as_deref(),
        Ok("1" | "true" | "on")
    );
    AtomicBool::new(on)
});

// SAFETY: a section handle may be used from any thread, and the view is
// plain memory.
unsafe impl Send for Section {}
// SAFETY: as above.
unsafe impl Sync for Section {}

impl Section {
    fn new(handle: HANDLE, pages: usize) -> Section {
        let size = pages * PAGE;
        let committed = (0..pages.div_ceil(64)).map(|_| AtomicU64::new(0)).collect();
        Section {
            handle,
            size,
            shared: victory_gpu::physical_heap(size),
            whole: OnceLock::new(),
            committed,
        }
    }

    /// The address of our view of the whole section, mapped the first time;
    /// `None` if the host would not map it.
    fn whole(&self) -> Option<usize> {
        if let Some(shared) = &self.shared {
            return Some(shared.cpu_address());
        }
        let at = *self.whole.get_or_init(|| {
            // SAFETY: our section; the view is ours until the section drops.
            unsafe { map_own_view(self.handle, 0, self.size) }.Value as usize
        });
        (at != 0).then_some(at)
    }

    /// Commits the pages from `index` on that a view at `base` of `len`
    /// bytes shows, unless all of them were committed already ([`lean`]).
    fn commit_shown(&self, index: usize, base: usize, len: usize) -> bool {
        let pages = index..index + len / PAGE;
        let bit = |p: usize| (self.committed.get(p / 64), 1u64 << (p % 64));
        let done = |p: usize| match bit(p) {
            (Some(word), mask) => word.load(Ordering::Acquire) & mask != 0,
            (None, _) => false,
        };
        if lean() && pages.clone().all(done) {
            return true;
        }
        if !commit(base, len) {
            return false;
        }
        for p in pages {
            if let (Some(word), mask) = bit(p) {
                word.fetch_or(mask, Ordering::AcqRel);
            }
        }
        true
    }
}

impl Drop for Section {
    fn drop(&mut self) {
        use windows_sys::Win32::System::Memory::UnmapViewOfFile;
        if let Some(&at) = self.whole.get()
            && at != 0
        {
            // SAFETY: our view of the section; nothing borrows it any more
            // (views of it hold the section).
            unsafe {
                UnmapViewOfFile(MEMORY_MAPPED_VIEW_ADDRESS {
                    Value: at as *mut c_void,
                })
            };
        }
        // SAFETY: our handle; views keep the section alive on their own.
        unsafe { CloseHandle(self.handle) };
    }
}

/// Where a page lives, or what a view starts with: a section and a page
/// index in it.
#[derive(Clone)]
struct Slice {
    section: Arc<Section>,
    index: usize,
}

impl Slice {
    fn key(&self, k: usize) -> (usize, usize) {
        (Arc::as_ptr(&self.section) as usize, self.index + k)
    }
}

#[derive(Default)]
struct Physical {
    /// Mapped views by address: their length and what they show.
    views: BTreeMap<usize, (usize, Slice)>,
    /// Unmapped placeholders left by whole-view unmaps: `(address, len)`
    /// ([`lean`]). If the title coalesces one with neighbours, the direct
    /// replace fails harmlessly and splits as before.
    holes: HashMap<usize, usize>,
}

/// The title's allocated physical pages by page number.
struct Pages {
    next_page: usize,
    pages: HashMap<usize, Slice>,
}

/// Where the title sees its pages: [`Physical::views`] indexed the other way.
/// Locked separately from [`PHYSICAL`] for brief in-memory updates/queries so
/// DirectStorage workers never wait behind host map/unmap syscalls.
#[derive(Default)]
struct Shown {
    /// Addresses showing `(section_ptr, page_index)` for views up to [`SHORT`]
    /// pages long.
    pages: HashMap<(usize, usize), Vec<usize>>,
    /// Views longer than [`SHORT`] pages: `(base, len, section_ptr, first_index)`.
    long: Vec<(usize, usize, usize, usize)>,
}

/// Views up to this many pages record each page in [`Shown::pages`]; larger
/// runs stay compact in [`Shown::long`].
const SHORT: usize = 64;

impl Shown {
    /// Appends every title view address currently showing `(section, index)`.
    fn push_at(&self, section: usize, index: usize, out: &mut Vec<usize>) {
        if let Some(addresses) = self.pages.get(&(section, index)) {
            out.extend_from_slice(addresses);
        }
        for &(base, len, of, first) in &self.long {
            if of == section && (first..first + len / PAGE).contains(&index) {
                out.push(base + (index - first) * PAGE);
            }
        }
    }

    /// Records `[base, base + len)` showing `slice..`, returning `(new_page,
    /// existing_others)` for every page that was already shown elsewhere.
    fn note(&mut self, base: usize, len: usize, slice: &Slice) -> Vec<(usize, Vec<usize>)> {
        let section = Arc::as_ptr(&slice.section) as usize;
        let count = len / PAGE;
        let mut aliased = Vec::new();
        for k in 0..count {
            let mut existing = Vec::new();
            self.push_at(section, slice.index + k, &mut existing);
            if !existing.is_empty() {
                aliased.push((base + k * PAGE, existing));
            }
        }
        if count > SHORT {
            self.long.push((base, len, section, slice.index));
        } else {
            for k in 0..count {
                self.pages
                    .entry((section, slice.index + k))
                    .or_default()
                    .push(base + k * PAGE);
            }
        }
        aliased
    }

    fn forget(&mut self, base: usize, len: usize, slice: &Slice) {
        if len / PAGE > SHORT {
            self.long.retain(|view| view.0 != base);
            return;
        }
        let section = Arc::as_ptr(&slice.section) as usize;
        for k in 0..len / PAGE {
            let key = (section, slice.index + k);
            if let Some(addresses) = self.pages.get_mut(&key) {
                addresses.retain(|&a| a != base + k * PAGE);
                if addresses.is_empty() {
                    self.pages.remove(&key);
                }
            }
        }
    }
}

/// Every address of a title view showing a page that title memory at
/// `[at, at + len)` shows, except the page's own address: pushed to `out`
/// as (page address, other address). Tests read `shown` back with it; the
/// GPU side's change tracking (`victory_gpu::watch`) marks writes there too.
#[cfg(test)]
fn aliases(at: usize, len: usize, out: &mut Vec<(usize, usize)>) {
    let physical = physical();
    let end = at.saturating_add(len);
    let first = physical
        .views
        .range(..=at)
        .next_back()
        .filter(|(base, (size, _))| *base + size > at)
        .map(|(base, _)| *base);
    let starts: Vec<usize> = first
        .into_iter()
        .chain(
            physical
                .views
                .range(at.saturating_add(1)..end)
                .map(|(b, _)| *b),
        )
        .collect();
    if starts.is_empty() {
        return;
    }
    let shown = shown();
    let mut addresses = Vec::new();
    for base in starts {
        let (size, slice) = &physical.views[&base];
        let section = Arc::as_ptr(&slice.section) as usize;
        let (from, to) = (at.max(base), end.min(base + size));
        for page in (from / PAGE..to.div_ceil(PAGE)).map(|p| p * PAGE) {
            let k = (page - base) / PAGE;
            addresses.clear();
            shown.push_at(section, slice.index + k, &mut addresses);
            out.extend(addresses.iter().filter(|&&a| a != page).map(|&a| (page, a)));
        }
    }
}

/// How title memory `[at, at + len)` lies on the title's physical pages,
/// for the GPU side's frame report (`victory_gpu::on_physical_runs`): the
/// 64 KiB pages of it shown by physical-page views, the runs they make
/// (pages one after another in one section, as a GPU buffer over that
/// memory would need them), and the sections. `None` if none is shown so.
pub(crate) fn physical_runs(at: usize, len: usize) -> Option<(usize, usize, usize)> {
    let physical = physical();
    let end = at.saturating_add(len);
    let first = physical
        .views
        .range(..=at)
        .next_back()
        .filter(|(base, (size, _))| *base + size > at)
        .map(|(base, _)| *base)
        .unwrap_or(at);
    let (mut pages, mut runs) = (0, 0);
    let mut sections = std::collections::HashSet::new();
    // The (section, index) the next page would continue a run with.
    let mut next: Option<(usize, usize)> = None;
    let mut last_end = usize::MAX;
    for (&base, (size, slice)) in physical.views.range(first..end) {
        let section = Arc::as_ptr(&slice.section) as usize;
        sections.insert(section);
        let (from, to) = (at.max(base), end.min(base + size));
        for page in (from / PAGE..to.div_ceil(PAGE)).map(|p| p * PAGE) {
            let here = (section, slice.index + (page - base) / PAGE);
            if next != Some(here) || last_end != page {
                runs += 1;
            }
            pages += 1;
            next = Some((here.0, here.1 + 1));
            last_end = page + PAGE;
        }
    }
    (pages > 0).then_some((pages, runs, sections.len()))
}

/// The tiles showing title memory `[at, at + len)` (page aligned), one a
/// page, for the GPU side's copies reading them in place
/// (`victory_gpu::on_physical_tiles`): `None` where a page shows no section
/// of GPU memory.
pub(crate) fn physical_tiles(at: usize, len: usize, out: &mut Vec<Option<victory_gpu::Tile>>) -> bool {
    let physical = physical();
    out.clear();
    out.extend((at / PAGE..at.saturating_add(len).div_ceil(PAGE)).map(|page| {
        let slice = slice_at(&physical, page * PAGE)?;
        slice.section.shared.as_ref()?.tile(slice.index * PAGE)
    }));
    true
}

/// The addresses of title views showing `pages` (page numbers): writes into
/// them through our own views must be reported there.
fn shown_at(pages: &[usize]) -> Vec<usize> {
    let slices: Vec<Slice> = {
        let table = page_table();
        pages
            .iter()
            .filter_map(|p| table.pages.get(p))
            .cloned()
            .collect()
    };
    if slices.is_empty() {
        return Vec::new();
    }
    let shown = shown();
    let mut out = Vec::new();
    for slice in &slices {
        shown.push_at(Arc::as_ptr(&slice.section) as usize, slice.index, &mut out);
    }
    out
}

/// Reports writes into physical pages through our own views at every title
/// view showing them (DirectStorage's page destinations).
fn report_written(shown: &[usize]) {
    for &page in shown {
        #[cfg(test)]
        REPORTED.with(|r| r.borrow_mut().push(page));
        drop(victory_gpu::kernel_will_write(page, PAGE));
    }
}

#[cfg(test)]
thread_local! {
    /// The title addresses this thread's writes into physical pages reported.
    static REPORTED: std::cell::RefCell<Vec<usize>> = const { std::cell::RefCell::new(Vec::new()) };
}

/// Mapped views, changed by the title's mapping calls (which hold it across
/// host map/commit/unmap syscalls).
static PHYSICAL: LazyLock<Mutex<Physical>> = LazyLock::new(Default::default);

fn physical() -> MutexGuard<'static, Physical> {
    PHYSICAL.lock().unwrap_or_else(|e| e.into_inner())
}

/// Allocated physical pages by number. Kept apart from [`PHYSICAL`] so
/// DirectStorage workers resolving page destinations never wait behind title
/// `MapViewOfFile3` / `UnmapViewOfFile2` calls. Never held across [`PHYSICAL`]
/// or [`SHOWN`].
static PAGES: LazyLock<RwLock<Pages>> = LazyLock::new(|| {
    RwLock::new(Pages {
        // Page numbers are opaque to titles; 0 stays unused.
        next_page: 0x10_0000,
        pages: HashMap::new(),
    })
});

fn page_table() -> RwLockReadGuard<'static, Pages> {
    PAGES.read().unwrap_or_else(|e| e.into_inner())
}

fn page_table_mut() -> RwLockWriteGuard<'static, Pages> {
    PAGES.write().unwrap_or_else(|e| e.into_inner())
}

/// Reverse slice-to-address index. Taken after [`PHYSICAL`] when both are
/// held, and dropped before calling into `victory_gpu`.
static SHOWN: LazyLock<RwLock<Shown>> = LazyLock::new(Default::default);

fn shown() -> RwLockReadGuard<'static, Shown> {
    SHOWN.read().unwrap_or_else(|e| e.into_inner())
}

fn shown_mut() -> RwLockWriteGuard<'static, Shown> {
    SHOWN.write().unwrap_or_else(|e| e.into_inner())
}

/// `count` is in and out (pages wanted, pages given); all or nothing here.
pub extern "system" fn XMemAllocatePhysicalPages(
    flags: u64,
    count: *mut usize,
    pages: *mut usize,
) -> BOOL {
    if count.is_null() || pages.is_null() {
        return 0;
    }
    // SAFETY: checked non-null.
    let n = unsafe { *count };
    let Some(size) = n.checked_mul(PAGE).filter(|&s| s > 0) else {
        return 0;
    };
    // A pagefile-backed section of `size` bytes, its pages committed when
    // first mapped or written (`commit`): Grand Theft Auto VI asks for 6.1
    // GiB at once, which a busy host often cannot commit (error 1450).
    // SAFETY: plain section creation.
    let handle = unsafe {
        CreateFileMappingW(
            INVALID_HANDLE_VALUE,
            std::ptr::null(),
            // Under Wine a committed section is a sparse file, free until
            // touched, and committing a reserved one page by page costs a
            // wineserver request each that scans every range committed so
            // far: commit it whole there.
            PAGE_READWRITE | if on_wine() { SEC_COMMIT } else { SEC_RESERVE },
            (size as u64 >> 32) as u32,
            size as u32,
            std::ptr::null(),
        )
    };
    if handle.is_null() {
        let error = std::io::Error::last_os_error();
        tracing::warn!(n, %error, "XMemAllocatePhysicalPages failed");
        // SAFETY: checked non-null.
        unsafe { count.write(0) };
        return 0;
    }
    tracing::debug!(
        n,
        flags = format_args!("{flags:#x}"),
        "XMemAllocatePhysicalPages"
    );
    let section = Arc::new(Section::new(handle, n));
    let mut table = page_table_mut();
    let first = table.next_page;
    table.next_page += n * FRAMES;
    for index in 0..n {
        let slice = Slice {
            section: section.clone(),
            index,
        };
        table.pages.insert(first + index * FRAMES, slice);
    }
    // Touch title memory outside our locks: a guard fault there may wait for
    // a mapping call on another thread.
    drop(table);
    for index in 0..n {
        // SAFETY: the title passes room for `n` page numbers.
        unsafe { pages.add(index).write(first + index * FRAMES) };
    }
    1
}

/// Page numbers count 4 KiB frames, so a 64 KiB page's is a multiple of
/// 16: Grand Theft Auto VI keeps them as 21-bit `number >> 4` and rebuilds
/// them with `<< 4` (128 GiB of pages at most).
const FRAMES: usize = PAGE >> 12;

/// How many of `pages` are no allocated page (diagnostics: a DirectStorage
/// destination that could not be written).
pub(crate) fn unknown_pages(pages: &[usize]) -> usize {
    let table = page_table();
    pages
        .iter()
        .filter(|p| !table.pages.contains_key(p))
        .count()
}

/// Pages still mapped stay readable through their views until unmapped.
pub extern "system" fn XMemFreePhysicalPages(n: usize, pages: *const usize) -> BOOL {
    if pages.is_null() {
        return 0;
    }
    // SAFETY: the title passes `n` page numbers.
    let numbers = unsafe { std::slice::from_raw_parts(pages, n) }.to_vec();
    let mut table = page_table_mut();
    // Drop sections outside `PAGES`: the last slice unmaps `Section::whole`.
    let gone: Vec<Slice> = numbers
        .iter()
        .filter_map(|number| table.pages.remove(number))
        .collect();
    drop(table);
    drop(gone);
    1
}

/// Consecutive slices of one section, committed and writable: through our
/// view of the whole section, or through a view of their own if the host
/// would not map all of it, unmapped when dropped.
struct View {
    at: usize,
    len: usize,
    /// The view this maps for itself, if any.
    own: Option<MEMORY_MAPPED_VIEW_ADDRESS>,
    /// Keeps the section, and with it the whole view, while this lives.
    _section: Arc<Section>,
}

impl View {
    /// `n` slices from `slice` on.
    fn map(slice: &Slice, n: usize) -> Option<View> {
        let (offset, len) = ((slice.index * PAGE) as u64, n * PAGE);
        let section = slice.section.clone();
        if let Some(whole) = section.whole() {
            let at = whole + offset as usize;
            return section.commit_shown(slice.index, at, len).then_some(View {
                at,
                len,
                own: None,
                _section: section,
            });
        }
        // SAFETY: slices of a section we own; the view is ours until dropped.
        let own = unsafe { map_own_view(section.handle, offset, len) };
        if own.Value.is_null() {
            return None;
        }
        let view = View {
            at: own.Value as usize,
            len,
            own: Some(own),
            _section: section.clone(),
        };
        section
            .commit_shown(slice.index, view.at, len)
            .then_some(view)
    }

    fn bytes(&mut self) -> &mut [u8] {
        // SAFETY: the view holds `len` committed read-write bytes, ours
        // alone while `self` lives (other views of the slices only alias
        // them, as the console's do).
        unsafe { std::slice::from_raw_parts_mut(self.at as *mut u8, self.len) }
    }
}

impl Drop for View {
    fn drop(&mut self) {
        use windows_sys::Win32::System::Memory::UnmapViewOfFile;
        if let Some(own) = self.own {
            // SAFETY: the view this mapped.
            unsafe { UnmapViewOfFile(own) };
        }
    }
}

/// The pages behind bytes `[offset, offset + len)` of `pages` (64 KiB
/// each), as runs of consecutive slices of one section: each run's first
/// slice and its length in pages. One read lock on [`PAGES`], independent
/// of [`PHYSICAL`]. `None` if a page is unknown or `pages` ends early.
fn runs(pages: &[usize], offset: usize, len: usize) -> Option<Vec<(Slice, usize)>> {
    let end = offset.checked_add(len)?;
    let pages = pages.get(offset / PAGE..end.div_ceil(PAGE))?;
    let table = page_table();
    let mut runs: Vec<(Slice, usize)> = Vec::new();
    for page in pages {
        let slice = table.pages.get(page)?;
        match runs.last_mut() {
            Some((first, n))
                if Arc::ptr_eq(&first.section, &slice.section)
                    && slice.index == first.index + *n =>
            {
                *n += 1;
            }
            _ => runs.push((slice.clone(), 1)),
        }
    }
    Some(runs)
}

/// Copies `data` into physical pages, from `offset` bytes into the first,
/// whether or not they are mapped anywhere (DirectStorage destinations).
/// Through each section's whole view ([`View`]), one run of consecutive
/// pages at a time. False if a page is unknown or the host fails.
pub(crate) fn write_physical(pages: &[usize], offset: usize, data: &[u8]) -> bool {
    let Some(runs) = runs(pages, offset, data.len()) else {
        return false;
    };
    let (mut within, mut done) = (offset % PAGE, 0);
    for (slice, n) in &runs {
        let Some(mut view) = View::map(slice, *n) else {
            return false;
        };
        let k = (n * PAGE - within).min(data.len() - done);
        view.bytes()[within..within + k].copy_from_slice(&data[done..done + k]);
        (within, done) = (0, done + k);
    }
    // Where the pages show once written (see `PageWindow`).
    report_written(&shown_at(
        pages
            .get(offset / PAGE..(offset + data.len()).div_ceil(PAGE))
            .unwrap_or(&[]),
    ));
    true
}

/// Bytes `[offset, offset + len)` of physical pages, writable in place
/// until dropped: DirectStorage decodes into it instead of a buffer it
/// would then copy.
pub(crate) struct PageWindow {
    view: View,
    within: usize,
    len: usize,
    /// The pages (page numbers) the bytes reach: the title views showing
    /// them when the window is done are told. Looked up then, not when the
    /// window opens: a decode takes milliseconds, and the title may map a
    /// page at another address meanwhile (its GPU heap); a copy that took
    /// the page there before the bytes landed would never hear of them. An
    /// address that stops showing a page meanwhile was told by the unmap.
    pages: Vec<usize>,
}

impl Drop for PageWindow {
    fn drop(&mut self) {
        report_written(&shown_at(&self.pages));
    }
}

impl PageWindow {
    pub(crate) fn bytes(&mut self) -> &mut [u8] {
        let (within, len) = (self.within, self.len);
        &mut self.view.bytes()[within..within + len]
    }
}

/// A window on bytes `[offset, offset + len)` of `pages` when they are
/// consecutive pages of one allocation (how titles usually hand them out);
/// `None` otherwise (pages of several allocations, out of order, or
/// unknown, or a host failure), which `write_physical` then handles or
/// reports.
pub(crate) fn page_window(pages: &[usize], offset: usize, len: usize) -> Option<PageWindow> {
    let runs = runs(pages, offset, len)?;
    let [(slice, n)] = runs.as_slice() else {
        return None;
    };
    Some(PageWindow {
        view: View::map(slice, *n)?,
        within: offset % PAGE,
        len,
        pages: pages
            .get(offset / PAGE..(offset + len).div_ceil(PAGE))
            .unwrap_or(&[])
            .to_vec(),
    })
}

/// Maps `n` pages at `address`, in place of what is mapped there; null
/// `pages` unmaps. Returns `address`, or null if a page or the host fails.
pub extern "system" fn XMemMapPhysicalPages(
    address: *mut c_void,
    n: usize,
    pages: *const usize,
) -> *mut c_void {
    if n == 0 {
        return address;
    }
    let base = address as usize;
    // Read title memory and resolve slices before taking `PHYSICAL`: a fault
    // in title memory may wait for a mapping call on another thread.
    // SAFETY: the title passes `n` page numbers, or null.
    let pages =
        (!pages.is_null()).then(|| unsafe { std::slice::from_raw_parts(pages, n) }.to_vec());
    let slices: Option<Vec<Option<Slice>>> = pages.as_ref().map(|pages| {
        let table = page_table();
        pages
            .iter()
            .map(|page| table.pages.get(page).cloned())
            .collect()
    });
    let mut physical = physical();
    let (Some(pages), Some(slices)) = (pages, slices) else {
        unmap(&mut physical, base, n * PAGE);
        return address;
    };
    let mut i = 0;
    while i < n {
        let Some(first) = slices[i].clone() else {
            unmap(&mut physical, base + i * PAGE, (n - i) * PAGE);
            tracing::warn!(page = pages[i], "XMemMapPhysicalPages: unknown page");
            return std::ptr::null_mut();
        };
        if shows(&physical, base + i * PAGE, &first) {
            i += 1;
            continue;
        }
        // One view per run of consecutive changed slices of one section when
        // `VICTORY_MAP_RUNS=1` is enabled; page by page otherwise so a later
        // single-page remap never tears down sibling pages.
        let run = if map_runs() {
            1 + slices[i + 1..]
                .iter()
                .zip(1..)
                .take_while(|&(slice, k)| {
                    slice.as_ref().is_some_and(|s| {
                        Arc::ptr_eq(&s.section, &first.section)
                            && s.index == first.index + k
                            && !shows(&physical, base + (i + k) * PAGE, s)
                    })
                })
                .count()
        } else {
            1
        };
        unmap(&mut physical, base + i * PAGE, run * PAGE);
        if !map_run(&mut physical, base + i * PAGE, run, &first) {
            return std::ptr::null_mut();
        }
        i += run;
    }
    address
}

/// One page at each address; null `pages`, or a page number 0, unmaps
/// (Grand Theft Auto VI passes 0 for addresses to clear). When
/// `VICTORY_MAP_RUNS` is enabled, consecutive virtual addresses are grouped
/// into a single `XMemMapPhysicalPages` call.
pub extern "system" fn XMemMapPhysicalPagesScatter(
    addresses: *const *mut c_void,
    n: usize,
    pages: *const usize,
) -> BOOL {
    if addresses.is_null() {
        return 0;
    }
    if !map_runs() {
        for i in 0..n {
            // SAFETY: the title passes `n` addresses and, unless null, `n` pages.
            let (address, page) = unsafe {
                (
                    *addresses.add(i),
                    if pages.is_null() || *pages.add(i) == 0 {
                        std::ptr::null()
                    } else {
                        pages.add(i)
                    },
                )
            };
            if XMemMapPhysicalPages(address, 1, page).is_null() {
                return 0;
            }
        }
        return 1;
    }
    let mut i = 0;
    while i < n {
        // SAFETY: the title passes `n` addresses and, unless null, `n` pages.
        let (first_addr, first_unmap) = unsafe {
            let addr = *addresses.add(i);
            let unmap = pages.is_null() || *pages.add(i) == 0;
            (addr, unmap)
        };
        let mut run = 1;
        while i + run < n {
            // SAFETY: `i + run < n` is within the title's arrays.
            let (next_addr, next_unmap) = unsafe {
                let addr = *addresses.add(i + run);
                let unmap = pages.is_null() || *pages.add(i + run) == 0;
                (addr, unmap)
            };
            if next_addr as usize != (first_addr as usize).wrapping_add(run * PAGE)
                || next_unmap != first_unmap
            {
                break;
            }
            run += 1;
        }
        let run_pages = if first_unmap {
            std::ptr::null()
        } else {
            // SAFETY: `pages` is non-null when `!first_unmap`, and `i + run <= n`.
            unsafe { pages.add(i) }
        };
        if XMemMapPhysicalPages(first_addr, run, run_pages).is_null() {
            return 0;
        }
        i += run;
    }
    1
}

/// Maps the physical pages shown at `[from, from + len)` at `to` as well,
/// in place of what `to` shows; `from` 0 unmaps. The GPU's page-mapping
/// copies (`victory_gpu`, queue `CopyPageMappingsBatchX`): Grand Theft Auto
/// VI maps pages into a pool view, then has the GPU map them into its tiled
/// heap. Coalesced into multi-page views when `VICTORY_MAP_RUNS` is on;
/// false if a source page is not mapped physical memory.
pub fn map_alias(to: u64, from: u64, len: u64) -> bool {
    let (to, from, len) = (to as usize, from as usize, len as usize);
    let mut physical = physical();
    if map_runs() {
        alias_in_runs(&mut physical, to, from, len)
    } else {
        alias_pages(&mut physical, to, from, len)
    }
}

fn follows(a: &Slice, b: &Slice) -> bool {
    Arc::ptr_eq(&a.section, &b.section) && a.index + 1 == b.index
}

fn alias_in_runs(physical: &mut Physical, to: usize, from: usize, len: usize) -> bool {
    if len == 0 {
        return true;
    }
    if from == 0 {
        unmap(physical, to, len);
        return true;
    }
    let pages = len / PAGE;
    let wanted: Vec<Slice> = (0..pages)
        .map_while(|k| slice_at(physical, from + k * PAGE))
        .collect();
    let same: Vec<bool> = wanted
        .iter()
        .enumerate()
        .map(|(k, w)| shows(physical, to + k * PAGE, w))
        .collect();
    let mut k = 0;
    while k < wanted.len() {
        let mut end = k + 1;
        while end < wanted.len()
            && same[end] == same[k]
            && (same[k] || follows(&wanted[end - 1], &wanted[end]))
        {
            end += 1;
        }
        if !same[k] {
            let (at, bytes) = (to + k * PAGE, (end - k) * PAGE);
            unmap(physical, at, bytes);
            if !map_run(physical, at, end - k, &wanted[k]) {
                return false;
            }
        }
        k = end;
    }
    if wanted.len() < pages {
        unmap(
            physical,
            to + wanted.len() * PAGE,
            len - wanted.len() * PAGE,
        );
        return false;
    }
    true
}

fn alias_pages(physical: &mut Physical, to: usize, from: usize, len: usize) -> bool {
    if len == 0 {
        return true;
    }
    if from == 0 {
        unmap(physical, to, len);
        return true;
    }
    for k in (0..len).step_by(PAGE) {
        let Some(slice) = slice_at(physical, from + k) else {
            unmap(physical, to + k, len - k);
            return false;
        };
        if shows(physical, to + k, &slice) {
            continue;
        }
        unmap(physical, to + k, PAGE);
        if !map(physical, to + k, PAGE, slice) {
            return false;
        }
    }
    true
}

/// Maps `pages` consecutive slices from `slice` on at `base`: as one view
/// (coalescing previously split placeholders if needed), or page by page if
/// coalescing is disabled or refused.
fn map_run(physical: &mut Physical, base: usize, pages: usize, slice: &Slice) -> bool {
    const MEM_COALESCE_PLACEHOLDERS: u32 = 1;
    if pages == 0 {
        return true;
    }
    if pages > 1 && map_runs() {
        if let Some(view) = try_map_placeholder(physical, base, pages * PAGE, slice) {
            return finish_map(physical, base, pages * PAGE, slice.clone(), view);
        }
        for k in 0..pages {
            physical.holes.remove(&(base + k * PAGE));
        }
        // SAFETY: `[base, base + pages * PAGE)` contains only unmapped
        // placeholders; splitting both outer edges ensures placeholder
        // boundaries before coalescing into one `pages * PAGE` placeholder.
        let coalesced = unsafe {
            VirtualFree(
                base as *mut c_void,
                PAGE,
                MEM_RELEASE | MEM_PRESERVE_PLACEHOLDER,
            );
            VirtualFree(
                (base + (pages - 1) * PAGE) as *mut c_void,
                PAGE,
                MEM_RELEASE | MEM_PRESERVE_PLACEHOLDER,
            );
            VirtualFree(
                base as *mut c_void,
                pages * PAGE,
                MEM_RELEASE | MEM_COALESCE_PLACEHOLDERS,
            ) != 0
        };
        if coalesced {
            physical.holes.insert(base, pages * PAGE);
            if map(physical, base, pages * PAGE, slice.clone()) {
                return true;
            }
        }
    }
    (0..pages).all(|k| {
        let single = Slice {
            index: slice.index + k,
            section: slice.section.clone(),
        };
        map(physical, base + k * PAGE, PAGE, single)
    })
}

/// The page a title view shows at `at`, if any.
fn slice_at(physical: &Physical, at: usize) -> Option<Slice> {
    physical
        .views
        .range(..=at)
        .next_back()
        .filter(|(base, (size, _))| at < *base + size)
        .map(|(base, (_, s))| Slice {
            index: s.index + (at - base) / PAGE,
            section: s.section.clone(),
        })
}

/// Whether a title view shows `slice` at `at` already. Mapping it there
/// again would change nothing, and would cost an unmap and a map, which
/// under Wine each hold the process's one virtual-memory lock across a
/// wineserver request: every page fault and protection change on other
/// threads waits for them (#50).
fn shows(physical: &Physical, at: usize, slice: &Slice) -> bool {
    slice_at(physical, at).is_some_and(|s| s.key(0) == slice.key(0))
}

fn try_map_placeholder(
    physical: &mut Physical,
    base: usize,
    len: usize,
    slice: &Slice,
) -> Option<MEMORY_MAPPED_VIEW_ADDRESS> {
    let map_view = || {
        // SAFETY: maps the section over a placeholder of `len` bytes, or
        // fails harmlessly if `[base, base + len)` is not one placeholder.
        unsafe {
            MapViewOfFile3(
                slice.section.handle,
                GetCurrentProcess(),
                base as *const c_void,
                (slice.index * PAGE) as u64,
                len,
                MEM_REPLACE_PLACEHOLDER,
                PAGE_READWRITE,
                std::ptr::null_mut(),
                0,
            )
        }
    };
    let own = physical.holes.remove(&base) == Some(len) && lean();
    let mut view = MEMORY_MAPPED_VIEW_ADDRESS {
        Value: std::ptr::null_mut(),
    };
    if own {
        view = map_view();
    }
    if view.Value as usize != base {
        // SAFETY: splits an enclosing placeholder around `[base, base + len)`
        // (failing harmlessly when already exact or split inside).
        unsafe {
            VirtualFree(
                base as *mut c_void,
                len,
                MEM_RELEASE | MEM_PRESERVE_PLACEHOLDER,
            );
        }
        view = map_view();
    }
    (view.Value as usize == base).then_some(view)
}

fn finish_map(
    physical: &mut Physical,
    base: usize,
    len: usize,
    slice: Slice,
    view: MEMORY_MAPPED_VIEW_ADDRESS,
) -> bool {
    #[cfg(test)]
    tests::MAPS.with(|m| m.set(m.get() + 1));
    let mapped = slice.section.commit_shown(slice.index, base, len);
    let mapped = mapped
        && slice.section.shared.as_ref().is_none_or(|shared| {
            // SAFETY: this live view is ours. The slice will hold the
            // backing heap until the view is unmapped.
            let aliased = unsafe { shared.alias_view(base, slice.index * PAGE, len) };
            if !aliased {
                tracing::warn!(
                    address = format_args!("{base:#x}"),
                    len,
                    "sharing the mapped XMem view with the host heap failed"
                );
            }
            aliased
        });
    if !mapped {
        // SAFETY: the view just mapped, back to a placeholder.
        let gone = unsafe { UnmapViewOfFile2(GetCurrentProcess(), view, MEM_PRESERVE_PLACEHOLDER) };
        if gone != 0 {
            physical.holes.insert(base, len);
        }
        return false;
    }
    // Record in `SHOWN` before notifying `victory_gpu` (and drop `SHOWN`
    // before acquiring `GUARDS`) so concurrent DirectStorage completions never
    // miss a newly mapped view.
    let aliased = shown_mut().note(base, len, &slice);
    // New contents, read-write: the GPU side's copies no longer guard it.
    victory_gpu::title_memory_remapped(base, len);
    for (at, others) in aliased {
        // Pages shown elsewhere already: guarded here too if they are there.
        victory_gpu::title_memory_aliased(at, &others);
    }
    // Copies of title buffers reading these pages in place map them now.
    if victory_gpu::physical_tiles() {
        let tiles: Vec<Option<victory_gpu::Tile>> = (0..len / PAGE)
            .map(|k| slice.section.shared.as_ref()?.tile((slice.index + k) * PAGE))
            .collect();
        victory_gpu::physical_retiled(base, len, &tiles);
    }
    physical.views.insert(base, (len, slice));
    true
}

/// Maps `slice` and the slices after it over `[base, base + len)`, which
/// must lie in one placeholder.
fn map(physical: &mut Physical, base: usize, len: usize, slice: Slice) -> bool {
    let Some(view) = try_map_placeholder(physical, base, len, &slice) else {
        #[cfg(test)]
        tests::MAPS.with(|m| m.set(m.get() + 1));
        tracing::warn!(
            address = format_args!("{base:#x}"),
            len,
            error = %std::io::Error::last_os_error(),
            "mapping physical pages failed (not a placeholder?)"
        );
        return false;
    };
    finish_map(physical, base, len, slice, view)
}

/// Commits the section pages a view shows (once per page, whichever view).
fn commit(base: usize, len: usize) -> bool {
    if on_wine() {
        return true; // sections are committed whole there
    }
    // SAFETY: `[base, base + len)` is a view of a reserved section.
    if !unsafe { VirtualAlloc(base as *const c_void, len, MEM_COMMIT, PAGE_READWRITE) }.is_null() {
        return true;
    }
    commit_refused(len, "physical pages", std::io::Error::last_os_error());
    false
}

/// Reports memory the host would not give: at the Windows commit limit to
/// `crashes.log`, since the title usually faults soon after on the null it
/// gets; otherwise as a warning.
fn commit_refused(len: usize, what: &str, error: std::io::Error) {
    if error.raw_os_error() == Some(ERROR_COMMITMENT_LIMIT as i32) {
        victory_loader::log_crash(&commit_limit_message(
            len,
            what,
            &error,
            system_commit().as_ref(),
        ));
    } else {
        tracing::warn!(len, %error, "committing {what} failed");
    }
}

/// Reports to `crashes.log` that the host allocator refused `len_bytes` of
/// APU heap memory. There is no Windows error code to give: `std::alloc`
/// allocates with `HeapAlloc`, which does not set one when it fails.
pub(crate) fn report_allocator_refusal(len_bytes: usize) {
    victory_loader::log_crash(&allocator_refusal_message(
        len_bytes,
        system_commit().as_ref(),
    ));
}

/// The system commit, if Windows reports it.
fn system_commit() -> Option<PERFORMANCE_INFORMATION> {
    let mut info = PERFORMANCE_INFORMATION {
        cb: size_of::<PERFORMANCE_INFORMATION>() as u32,
        ..Default::default()
    };
    // SAFETY: the buffer and size describe a writable PERFORMANCE_INFORMATION.
    (unsafe { K32GetPerformanceInfo(&mut info, info.cb) } != 0).then_some(info)
}

fn commit_limit_message(
    len: usize,
    what: &str,
    error: &std::io::Error,
    info: Option<&PERFORMANCE_INFORMATION>,
) -> String {
    refusal_message(
        format_args!(
            "Windows commit limit reached while committing {len} bytes of {what}: {error}"
        ),
        info,
    )
}

fn allocator_refusal_message(len_bytes: usize, info: Option<&PERFORMANCE_INFORMATION>) -> String {
    refusal_message(
        format_args!("the host allocator refused {len_bytes} bytes of APU heap memory"),
        info,
    )
}

/// What the host refused, then the system commit and what the user can do.
fn refusal_message(
    refusal: std::fmt::Arguments<'_>,
    info: Option<&PERFORMANCE_INFORMATION>,
) -> String {
    let mut message = format!("VIctory: {refusal}.\n");
    if let Some(info) = info {
        use std::fmt::Write;

        let committed = info.CommitTotal * info.PageSize / MIB;
        let limit = info.CommitLimit * info.PageSize / MIB;
        let _ = writeln!(
            message,
            "System commit: {committed} MiB of {limit} MiB in use."
        );
    }
    message.push_str(
        "Close other applications or increase the Windows paging file size, then retry.\n",
    );
    message
}

/// Returns `[base, base + len)` to placeholders; views reaching past it
/// are mapped again outside it.
fn unmap(physical: &mut Physical, base: usize, len: usize) {
    // No bytes: no view to return. The search below would still find the
    // view holding `base` and cut it in two there, and its range of views
    // starting inside would run backwards (which `BTreeMap` panics on).
    if len == 0 {
        return;
    }
    let end = base + len;
    // Views do not overlap: the one holding `base`, then those starting
    // inside (the GPU's page mappings make one view per page).
    let first = physical
        .views
        .range(..=base)
        .next_back()
        .filter(|(at, (size, _))| *at + size > base)
        .map(|(at, _)| *at);
    let hit: Vec<usize> = first
        .into_iter()
        .chain(physical.views.range(base + 1..end).map(|(at, _)| *at))
        .collect();
    for at in hit {
        let Some((size, slice)) = physical.views.remove(&at) else {
            continue;
        };
        shown_mut().forget(at, size, &slice);
        if victory_gpu::physical_tiles() {
            victory_gpu::physical_retiled(at, size, &vec![None; size / PAGE]);
        }
        let going = victory_gpu::title_memory_unmapping(at, size);
        // SAFETY: a view we mapped; it becomes a placeholder again.
        let gone = unsafe {
            UnmapViewOfFile2(
                GetCurrentProcess(),
                MEMORY_MAPPED_VIEW_ADDRESS {
                    Value: at as *mut c_void,
                },
                MEM_PRESERVE_PLACEHOLDER,
            )
        };
        if gone != 0 && at >= base && at + size <= end {
            physical.holes.insert(at, size);
        }
        drop(going);
        if at < base {
            map(physical, at, base - at, slice.clone());
        }
        if at + size > end {
            let index = slice.index + (end - at) / PAGE;
            map(physical, end, at + size - end, Slice { index, ..slice });
        }
    }
}

/// The oldest layout we know; newer titles pass other flag values (Minecraft
/// 0x20000001) and read the same leading fields.
#[repr(C)]
#[derive(Debug, Default)]
pub struct WorkingSet {
    pub title_limit: usize,
    pub title_used: usize,
    pub tools_limit: usize,
    pub tools_used: usize,
    pub stack_used: u32,
    pub physical_pages: u32,
    pub gpu_optimal_limit: usize,
    pub gpu_optimal_used: usize,
    pub tools_gpu_optimal_limit: usize,
    pub tools_gpu_optimal_used: usize,
    pub kernel_pool_used: u32,
    pub kernel_pool_limit: u32,
}

/// The title's budget and our process's commit as what it uses. No tools
/// memory: a retail console.
// shortcut: all fields are filled whatever the flags ask for, and use is
// not split by bandwidth class (GPU-optimal use reads 0); track
// allocations per class if a title budgets from that field.
pub extern "system" fn XMemGetWorkingSetStatistics(_which: u32, out: *mut WorkingSet) -> BOOL {
    if out.is_null() {
        return 0;
    }
    let mut counters = PROCESS_MEMORY_COUNTERS {
        cb: size_of::<PROCESS_MEMORY_COUNTERS>() as u32,
        ..Default::default()
    };
    // SAFETY: our own process and a sized structure.
    unsafe { K32GetProcessMemoryInfo(GetCurrentProcess(), &mut counters, counters.cb) };
    let (gpu_optimal, standard) = victory_gpu::console::current().title_memory();
    let stats = WorkingSet {
        title_limit: gpu_optimal + standard - TRANSFERRED.load(Ordering::Relaxed),
        title_used: counters.PagefileUsage,
        gpu_optimal_limit: gpu_optimal,
        ..Default::default()
    };
    // SAFETY: checked non-null; the title passes room for the structure.
    unsafe { out.write(stats) };
    1
}

/// Hands title memory (standard, in 2 MiB pages) to the system. The other
/// transfers involve tools memory, which a retail console has none of.
pub extern "system" fn XMemTransferMemory(kind: u32, size: usize) -> BOOL {
    const TITLE_TO_SYSTEM: u32 = 0;
    tracing::info!(kind, size, "XMemTransferMemory");
    if size == 0 {
        return 1;
    }
    let Some(size) = size.checked_next_multiple_of(2 * MIB) else {
        return 0;
    };
    let standard = victory_gpu::console::current().title_memory().1;
    let moved = kind == TITLE_TO_SYSTEM
        && TRANSFERRED
            .fetch_update(Ordering::Relaxed, Ordering::Relaxed, |t| {
                t.checked_add(size).filter(|&t| t <= standard)
            })
            .is_ok();
    moved.into()
}

/// The title's own `VirtualFree`: the GPU side's guards let go of the memory
/// first, and keep it unguarded until it is gone
/// (`victory_gpu::title_memory_unmapping`). Once free, the address may be
/// other memory's at once, which a guard noted of the old memory would
/// leave unwatched.
extern "system" fn title_virtual_free(address: *mut c_void, size: usize, kind: u32) -> BOOL {
    const MEM_COALESCE_PLACEHOLDERS: u32 = 1;
    // Splitting and joining placeholders changes nothing they show.
    let going = (kind & (MEM_PRESERVE_PLACEHOLDER | MEM_COALESCE_PLACEHOLDERS) == 0).then(|| {
        // A release takes the whole allocation; its size argument is 0.
        let len = if kind & MEM_RELEASE != 0 { 0 } else { size };
        victory_gpu::title_memory_unmapping(address as usize, len)
    });
    // SAFETY: the title's own arguments.
    let freed = unsafe { VirtualFree(address, size, kind) };
    drop(going);
    freed
}

/// [`title_virtual_free`] for the title's imports of `VirtualFree`, which
/// the rest of the system's DLLs serve as they are.
pub(crate) fn virtual_free_module() -> HleModule {
    HleModule {
        name: "xmem-virtual-free",
        dlls: &["kernel32.dll", "kernelbase.dll", "api-ms-win-core-memory-*"],
        exports: vec![crate::HleExport {
            name: "VirtualFree",
            address: title_virtual_free as *const () as usize,
        }],
    }
}

pub(crate) fn module() -> HleModule {
    HleModule {
        name: "xmem",
        dlls: &["xmem.dll"],
        exports: vec![
            hle_export!(XMemAlloc),
            hle_export!(XMemFree),
            hle_export!(XMemAllocDefault),
            hle_export!(XMemFreeDefault),
            hle_export!(XMemIsToolsMemAvailable),
            hle_export!(XMemAllocatePhysicalPages),
            hle_export!(XMemFreePhysicalPages),
            hle_export!(XMemMapPhysicalPages),
            hle_export!(XMemMapPhysicalPagesScatter),
            hle_export!(XMemVirtualAlloc),
            hle_export!(XMemVirtualQuery),
            hle_export!(XMemGetWorkingSetStatistics),
            hle_export!(XMemTransferMemory),
        ],
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The title's imports of `VirtualFree`, however it names their DLL,
    /// reach [`title_virtual_free`], which frees as the system does and
    /// passes placeholder splits through.
    #[test]
    fn the_titles_virtual_free_is_ours() {
        for dll in [
            "kernel32.dll",
            "KERNELBASE.dll",
            "api-ms-win-core-memory-l1-1-0.dll",
        ] {
            assert_eq!(
                crate::registry::lookup(dll, "VirtualFree"),
                Some(title_virtual_free as *const () as usize),
                "{dll}"
            );
        }
        // SAFETY: a fresh reservation of ours, freed through the export.
        let p = unsafe {
            VirtualAlloc(
                std::ptr::null(),
                2 * PAGE,
                MEM_RESERVE | MEM_COMMIT,
                PAGE_READWRITE,
            )
        };
        assert!(!p.is_null());
        assert_ne!(title_virtual_free(p, 0, MEM_RELEASE), 0);
        assert_eq!(title_virtual_free(p, 0, MEM_RELEASE), 0, "freed already");
    }

    fn query(address: *const c_void) -> BasicInformation {
        // SAFETY: plain data; zero is a valid value.
        let mut info: BasicInformation = unsafe { std::mem::zeroed() };
        // SAFETY: a pseudo handle needs no closing.
        let process = unsafe { GetCurrentProcess() };
        assert_eq!(XMemVirtualQuery(process, address, &mut info), 1);
        info
    }

    #[test]
    fn alloc_honours_alignment_and_frees() {
        for (field, align) in [(0, 16), (4, 16), (0xc, 4096), (0x10, 64 * 1024)] {
            let attributes = field << ALIGNMENT_SHIFT | 1 << 29; // a graphics type
            let p = XMemAlloc(100, attributes);
            assert!(!p.is_null());
            assert_eq!(p as usize % align, 0, "field {field:#x}");
            // SAFETY: 100 bytes were allocated.
            unsafe { p.cast::<u8>().write_bytes(0xab, 100) };
            XMemFree(p, attributes);
        }
        let p = XMemAlloc(4096, ALLOC_32BIT);
        assert!(!p.is_null() && (p as usize) < 1 << 32);
        XMemFree(p, ALLOC_32BIT);
    }

    #[test]
    fn virtual_alloc_meets_console_constraints() {
        const GPU_READ_WRITE: u32 = 0x2000;
        let p = XMemVirtualAlloc(
            std::ptr::null_mut(),
            4 * MIB,
            PAGES_2M | MEM_RESERVE | MEM_COMMIT,
            GRAPHICS | 7 << 12, // 40-bit addresses
            PAGE_READWRITE | GPU_READ_WRITE,
        );
        assert!(!p.is_null());
        assert_eq!(p as usize % (2 * MIB), 0);
        assert!((p as usize) < 1 << 40);
        let info = query(p);
        assert_eq!(info.host.State, MEM_COMMIT);
        assert_eq!(info.host.Protect, PAGE_READWRITE);
        assert_eq!(info.flags & GRAPHICS, GRAPHICS);
        // Write-watched, for the GPU side's copies.
        let (mut found, mut count, mut granularity) = ([0usize; 4], 4, 0);
        // SAFETY: room for `count` addresses.
        let watched = unsafe {
            windows_sys::Win32::System::Memory::GetWriteWatch(
                0,
                p,
                4 * MIB,
                found.as_mut_ptr().cast(),
                &mut count,
                &mut granularity,
            )
        };
        // Not under Rosetta (`victory_gpu::fault_tracking`).
        assert_eq!(watched == 0, victory_gpu::fault_tracking());
        // SAFETY: our own reservation.
        assert_ne!(unsafe { VirtualFree(p, 0, MEM_RELEASE) }, 0);
        assert_eq!(query(p).flags, 0);
        // No address field: above the 32-bit range and below 2^40, where
        // the console keeps it.
        let p = XMemVirtualAlloc(
            std::ptr::null_mut(),
            16 << 30,
            MEM_RESERVE,
            GRAPHICS,
            PAGE_NOACCESS,
        );
        assert!(!p.is_null() && (p as usize) >= 1 << 32, "{p:p}");
        assert!((p as usize) + (16 << 30) <= 1 << 40, "{p:p}");
        // SAFETY: our own reservation.
        assert_ne!(unsafe { VirtualFree(p, 0, MEM_RELEASE) }, 0);
    }

    #[test]
    fn transfers_shrink_the_title_budget() {
        let mut before = WorkingSet::default();
        assert_eq!(XMemGetWorkingSetStatistics(0x2000_0001, &mut before), 1);
        assert!(before.title_used > 0);
        assert_eq!(XMemTransferMemory(0, 128 * MIB - 1), 1);
        assert_eq!(XMemTransferMemory(2, MIB), 0, "no tools memory");
        let standard = victory_gpu::console::current().title_memory().1;
        assert_eq!(XMemTransferMemory(0, standard), 0, "over budget");
        let mut after = WorkingSet::default();
        XMemGetWorkingSetStatistics(0x2000_0001, &mut after);
        assert_eq!(before.title_limit - after.title_limit, 128 * MIB);
    }

    #[test]
    fn statistics_have_the_console_layout() {
        assert_eq!(size_of::<WorkingSet>(), 0x50);
        assert_eq!(size_of::<BasicInformation>(), 0x40);
        assert_eq!(std::mem::offset_of!(WorkingSet, gpu_optimal_limit), 0x28);
    }

    #[test]
    fn commit_limit_report_includes_request_and_system_commit() {
        let error = std::io::Error::from_raw_os_error(ERROR_COMMITMENT_LIMIT as i32);
        let info = PERFORMANCE_INFORMATION {
            CommitTotal: 512,
            CommitLimit: 768,
            PageSize: 4096,
            ..Default::default()
        };
        let report = commit_limit_message(4 * MIB, "physical pages", &error, Some(&info));
        assert!(report.contains("4194304 bytes of physical pages"));
        assert!(report.contains("2 MiB of 3 MiB in use"));
        let fallback = commit_limit_message(4 * MIB, "virtual memory", &error, None);
        assert!(fallback.contains("4194304 bytes of virtual memory"));
        for report in [&report, &fallback] {
            assert!(report.contains("os error 1455"));
            assert!(report.contains("Windows paging file size"));
        }
        assert!(!fallback.contains("System commit:"));
    }

    thread_local! {
        /// Views this thread mapped for the title (`map`).
        pub(super) static MAPS: std::cell::Cell<usize> = const { std::cell::Cell::new(0) };
        /// Per-thread test override for [`lean`].
        pub(super) static TEST_LEAN: std::cell::Cell<Option<bool>> = const { std::cell::Cell::new(None) };
        /// Per-thread test override for [`map_runs`].
        pub(super) static TEST_MAP_RUNS: std::cell::Cell<Option<bool>> = const { std::cell::Cell::new(None) };
    }

    struct ModeGuard {
        prev_lean: Option<bool>,
        prev_runs: Option<bool>,
    }

    impl ModeGuard {
        fn set(lean: Option<bool>, runs: Option<bool>) -> Self {
            let prev_lean = TEST_LEAN.with(|c| c.replace(lean));
            let prev_runs = TEST_MAP_RUNS.with(|c| c.replace(runs));
            Self {
                prev_lean,
                prev_runs,
            }
        }
    }

    impl Drop for ModeGuard {
        fn drop(&mut self) {
            TEST_LEAN.with(|c| c.set(self.prev_lean));
            TEST_MAP_RUNS.with(|c| c.set(self.prev_runs));
        }
    }

    /// `VICTORY_PHYSICAL_TILES=1`: mapped physical pages are GPU memory, and
    /// the lookup names their tiles in the title's order; unmapped, none.
    #[test]
    fn physical_tiles_name_the_gpu_tiles_a_view_shows() {
        let pages = allocate(2);
        if page_table().pages[&pages[0]].section.shared.is_none() {
            assert!(!victory_gpu::physical_tiles(), "the switch is on, the pages are not GPU memory");
            return; // VICTORY_PHYSICAL_TILES off, or not Rosetta and D3DMetal
        }
        let at = XMemVirtualAlloc(std::ptr::null_mut(), 2 * PAGE, MEM_RESERVE | PAGES_64K, 0, PAGE_READWRITE);
        assert!(!at.is_null());
        let order = [pages[1], pages[0]];
        assert_eq!(XMemMapPhysicalPages(at, 2, order.as_ptr()), at);
        let mut tiles = Vec::new();
        assert!(physical_tiles(at as usize, 2 * PAGE, &mut tiles));
        let index = |k: usize| tiles[k].as_ref().map(victory_gpu::Tile::index);
        let first = page_table().pages[&pages[0]].index as u32;
        assert_eq!([index(0), index(1)], [Some(first + 1), Some(first)], "in the title's order");
        // The title's writes land in the GPU heap: its own view shows them.
        // SAFETY: mapped read-write.
        unsafe { at.cast::<u32>().write(0x5eed) };
        let whole = page_table().pages[&pages[1]].section.whole().unwrap();
        // SAFETY: the section's whole view, the heap's CPU address.
        assert_eq!(unsafe { ((whole + (first as usize + 1) * PAGE) as *const u32).read() }, 0x5eed);
        assert_eq!(XMemMapPhysicalPages(at, 2, std::ptr::null()), at);
        assert!(physical_tiles(at as usize, 2 * PAGE, &mut tiles));
        assert!(tiles.iter().all(Option::is_none), "unmapped");
        assert_eq!(XMemFreePhysicalPages(2, pages.as_ptr()), 1);
    }

    #[test]
    fn physical_runs_count_pages_one_after_another_in_a_section() {
        let pages = allocate(4);
        let at = XMemVirtualAlloc(std::ptr::null_mut(), 4 * PAGE, MEM_RESERVE | PAGES_64K, 0, PAGE_READWRITE);
        assert!(!at.is_null());
        // Pages 0, 1, then 3, 2: a run of two, then two runs of one.
        let order = [pages[0], pages[1], pages[3], pages[2]];
        assert_eq!(XMemMapPhysicalPages(at, 4, order.as_ptr()), at);
        assert_eq!(physical_runs(at as usize, 4 * PAGE), Some((4, 3, 1)));
        assert_eq!(physical_runs(at as usize, 2 * PAGE), Some((2, 1, 1)), "the first two in order");
        assert_eq!(physical_runs(at as usize + 3 * PAGE + 100, 8), Some((1, 1, 1)));
        // Not physical memory.
        let plain = XMemVirtualAlloc(std::ptr::null_mut(), PAGE, MEM_RESERVE | MEM_COMMIT, 0, PAGE_READWRITE);
        assert_eq!(physical_runs(plain as usize, PAGE), None);
    }

    #[test]
    fn mapping_what_shows_already_maps_nothing() {
        let pages = allocate(2);
        let reserve = || {
            XMemVirtualAlloc(
                std::ptr::null_mut(),
                2 * PAGE,
                MEM_RESERVE | PAGES_64K,
                0,
                PAGE_READWRITE,
            )
        };
        let (a, b) = (reserve(), reserve());
        assert!(!a.is_null() && !b.is_null());
        let maps = || MAPS.with(std::cell::Cell::get);
        assert_eq!(XMemMapPhysicalPages(a, 2, pages.as_ptr()), a);
        let before = maps();
        // The same pages at the same address: nothing to do.
        assert_eq!(XMemMapPhysicalPages(a, 2, pages.as_ptr()), a);
        let scattered = [a, a.wrapping_byte_add(PAGE)];
        assert_eq!(
            XMemMapPhysicalPagesScatter(scattered.as_ptr(), 2, pages.as_ptr()),
            1
        );
        assert_eq!(maps(), before);
        // The GPU's page-mapping copy, made twice: the second maps nothing.
        assert!(map_alias(b as u64, a as u64, 2 * PAGE as u64));
        assert_eq!(maps(), before + 2);
        assert!(map_alias(b as u64, a as u64, 2 * PAGE as u64));
        assert!(map_alias(
            b.wrapping_byte_add(PAGE) as u64,
            a.wrapping_byte_add(PAGE) as u64,
            PAGE as u64
        ));
        assert_eq!(maps(), before + 2);
        // Another page, or one page of the range changed, maps only that page.
        assert!(map_alias(
            b as u64,
            a.wrapping_byte_add(PAGE) as u64,
            PAGE as u64
        ));
        assert_eq!(maps(), before + 3);
        // SAFETY: both mapped read-write, showing the second page.
        unsafe {
            a.wrapping_byte_add(PAGE).cast::<u8>().write(6);
            assert_eq!(b.cast::<u8>().read(), 6);
        }
        // Changing only the second page of `a` leaves the first untouched.
        let one_changed = [pages[0], pages[0]];
        assert_eq!(XMemMapPhysicalPages(a, 2, one_changed.as_ptr()), a);
        assert_eq!(maps(), before + 4);
        // Pages swapped from `[pages[0], pages[0]]` to `[pages[1], pages[0]]`:
        // only the first page differs now, so only 1 view maps again.
        let swapped = [pages[1], pages[0]];
        assert_eq!(XMemMapPhysicalPages(a, 2, swapped.as_ptr()), a);
        assert_eq!(maps(), before + 5);
        // SAFETY: mapped read-write; `a` shows the second page now.
        assert_eq!(unsafe { a.cast::<u8>().read() }, 6);
        for at in [a, b] {
            assert_eq!(XMemMapPhysicalPages(at, 2, std::ptr::null()), at);
        }
        assert_eq!(XMemFreePhysicalPages(2, pages.as_ptr()), 1);
    }

    #[test]
    fn allocator_refusal_report_names_apu_heap_and_system_commit() {
        let info = PERFORMANCE_INFORMATION {
            CommitTotal: 512,
            CommitLimit: 768,
            PageSize: 4096,
            ..Default::default()
        };
        let report = allocator_refusal_message(4 * MIB, Some(&info));
        assert!(report.contains("host allocator refused 4194304 bytes of APU heap memory"));
        assert!(report.contains("System commit: 2 MiB of 3 MiB in use"));
        assert!(report.contains("Windows paging file size"));
        let fallback = allocator_refusal_message(4 * MIB, None);
        assert!(fallback.contains("4194304 bytes of APU heap memory"));
        assert!(!fallback.contains("System commit:"));
    }

    #[test]
    fn physical_pages_map_alias_and_unmap() {
        let mut pages = [0usize; 3];
        let mut n = 3;
        assert_eq!(XMemAllocatePhysicalPages(0, &mut n, pages.as_mut_ptr()), 1);
        assert_eq!(n, 3);
        if victory_gpu::physical_tiles() {
            let table = page_table();
            assert!(
                table.pages[&pages[0]].section.shared.is_some(),
                "the test must use shared physical pages"
            );
        }
        // Packed into 21 bits as `number >> 4`, as the title keeps them.
        for p in pages {
            assert_eq!((p >> 4 & 0x1f_ffff) << 4, p);
        }
        // Reservations of 64 KiB pages, as titles make them for mapping.
        let reserve = || {
            XMemVirtualAlloc(
                std::ptr::null_mut(),
                3 * PAGE,
                MEM_RESERVE | PAGES_64K,
                0,
                PAGE_READWRITE,
            )
        };
        let (a, b) = (reserve(), reserve());
        assert!(!a.is_null() && !b.is_null());
        let at = |p: *mut c_void, page: usize| p.wrapping_byte_add(page * PAGE);
        assert_eq!(XMemMapPhysicalPages(a, 3, pages.as_ptr()), a);

        // The same page at a second address shows the same bytes.
        assert_eq!(
            XMemMapPhysicalPages(at(b, 1), 1, pages[2..].as_ptr()),
            at(b, 1)
        );
        // SAFETY: both pages are mapped read-write.
        unsafe {
            at(a, 2).cast::<u8>().write(7);
            assert_eq!(at(b, 1).cast::<u8>().read(), 7);
            a.cast::<u8>().write(1);
        }
        // Each mapping knows the other (the GPU side's change tracking).
        let others = |p: *mut c_void| {
            let mut out = Vec::new();
            aliases(p as usize, PAGE, &mut out);
            out.into_iter().map(|(_, a)| a).collect::<Vec<_>>()
        };
        assert_eq!(others(at(a, 2)), [at(b, 1) as usize]);
        assert_eq!(others(at(b, 1)), [at(a, 2) as usize]);
        assert!(others(a).is_empty());
        // Writes into pages (DirectStorage's) show through their views.
        assert!(write_physical(&pages[2..], 1, &[9]));
        // SAFETY: mapped read-write.
        assert_eq!(unsafe { at(b, 1).cast::<u8>().add(1).read() }, 9);

        // Unmapping the middle page keeps its neighbours.
        assert_eq!(
            XMemMapPhysicalPages(at(a, 1), 1, std::ptr::null()),
            at(a, 1)
        );
        assert_eq!(query(at(a, 1)).host.State, MEM_RESERVE);
        assert_eq!(query(a).host.State, MEM_COMMIT);
        assert_eq!(query(at(a, 2)).host.State, MEM_COMMIT);
        // SAFETY: still mapped.
        assert_eq!(unsafe { at(a, 2).cast::<u8>().read() }, 7);

        // A scattered page lands in the hole.
        let addresses = [at(a, 1)];
        assert_eq!(
            XMemMapPhysicalPagesScatter(addresses.as_ptr(), 1, pages.as_ptr()),
            1
        );
        // SAFETY: page 0 is now mapped at both `a` and `a + PAGE`.
        assert_eq!(unsafe { at(a, 1).cast::<u8>().read() }, 1);
        // Page number 0 unmaps.
        assert_eq!(
            XMemMapPhysicalPagesScatter(addresses.as_ptr(), 1, [0usize].as_ptr()),
            1
        );
        assert_eq!(query(at(a, 1)).host.State, MEM_RESERVE);
        // The GPU's page-mapping copy: what `a + 2 pages` shows, at `b`.
        assert!(map_alias(b as u64, at(a, 2) as u64, PAGE as u64));
        // SAFETY: mapped read-write.
        assert_eq!(unsafe { b.cast::<u8>().read() }, 7);
        let mut both = others(at(a, 2));
        both.sort_unstable();
        assert_eq!(both, [b as usize, at(b, 1) as usize]);
        assert!(map_alias(b as u64, 0, PAGE as u64));
        assert_eq!(
            others(at(a, 2)),
            [at(b, 1) as usize],
            "unmapped views are forgotten"
        );
        assert_eq!(query(b).host.State, MEM_RESERVE);
        assert!(!map_alias(b as u64, at(a, 1) as u64, PAGE as u64));
        assert_eq!(XMemFreePhysicalPages(3, pages.as_ptr()), 1);
        assert_eq!(XMemIsToolsMemAvailable(), 0);
    }

    #[test]
    fn mapping_no_pages_changes_nothing() {
        let pages = allocate(2);
        let a = XMemVirtualAlloc(
            std::ptr::null_mut(),
            2 * PAGE,
            MEM_RESERVE | PAGES_64K,
            0,
            PAGE_READWRITE,
        );
        assert!(!a.is_null());
        assert_eq!(XMemMapPhysicalPages(a, 2, pages.as_ptr()), a);
        // SAFETY: both pages are mapped read-write.
        unsafe { a.cast::<u8>().write(5) };
        let second = a.wrapping_byte_add(PAGE);

        // A count of 0 at the start of a view, inside it and past it: with
        // pages, without (an unmap), and as the GPU's page-mapping copy.
        for at in [a, second, a.wrapping_byte_add(2 * PAGE)] {
            assert_eq!(XMemMapPhysicalPages(at, 0, pages.as_ptr()), at);
            assert_eq!(XMemMapPhysicalPages(at, 0, std::ptr::null()), at);
            assert!(map_alias(at as u64, a as u64, 0));
            assert!(map_alias(at as u64, 0, 0));
        }

        // Each page's view is still intact.
        assert_eq!(physical().views.get(&(a as usize)).map(|v| v.0), Some(PAGE));
        assert_eq!(
            physical().views.get(&(second as usize)).map(|v| v.0),
            Some(PAGE)
        );
        assert_eq!(query(a).host.State, MEM_COMMIT);
        assert_eq!(query(second).host.State, MEM_COMMIT);
        // SAFETY: still mapped.
        assert_eq!(unsafe { a.cast::<u8>().read() }, 5);

        // A count of 1 still unmaps that page alone.
        assert_eq!(XMemMapPhysicalPages(second, 1, std::ptr::null()), second);
        assert_eq!(query(second).host.State, MEM_RESERVE);
        assert_eq!(query(a).host.State, MEM_COMMIT);
        assert_eq!(XMemMapPhysicalPages(a, 1, std::ptr::null()), a);
        assert_eq!(XMemFreePhysicalPages(2, pages.as_ptr()), 1);
    }

    /// `n` pages of one new allocation.
    fn allocate(n: usize) -> Vec<usize> {
        let (mut pages, mut given) = (vec![0; n], n);
        assert_eq!(
            XMemAllocatePhysicalPages(0, &mut given, pages.as_mut_ptr()),
            1
        );
        assert_eq!(given, n);
        pages
    }

    /// The bytes of one page, through our view of its section.
    fn page_bytes(page: usize) -> Vec<u8> {
        let slice = page_table().pages.get(&page).cloned().expect("known page");
        View::map(&slice, 1).expect("mapped").bytes().to_vec()
    }

    #[test]
    fn runs_group_consecutive_pages_of_one_allocation() {
        let (a, b) = (allocate(3), allocate(1));
        let lengths = |pages: &[usize], offset, len| {
            runs(pages, offset, len).map(|r| r.iter().map(|&(_, n)| n).collect::<Vec<_>>())
        };
        assert_eq!(lengths(&a, 0, 3 * PAGE), Some(vec![3]));
        // Only the pages the bytes reach.
        assert_eq!(lengths(&a, PAGE + 5, 10), Some(vec![1]));
        assert_eq!(lengths(&a, PAGE - 1, 2), Some(vec![2]));
        assert_eq!(lengths(&a, 0, 0), Some(vec![]));
        // Out of order, or across allocations: one run each.
        assert_eq!(
            lengths(&[a[0], a[2], a[1]], 0, 3 * PAGE),
            Some(vec![1, 1, 1])
        );
        assert_eq!(lengths(&[a[1], a[2], b[0]], 0, 3 * PAGE), Some(vec![2, 1]));
        // A slice that follows on in another section starts a run.
        assert_eq!(lengths(&[b[0], a[1]], 0, 2 * PAGE), Some(vec![1, 1]));
        // Too few pages for the bytes, or an unknown one.
        assert_eq!(lengths(&a, 0, 3 * PAGE + 1), None);
        assert_eq!(lengths(&[a[0], 0x10], 0, 2 * PAGE), None);
        assert_eq!(lengths(&a, usize::MAX, 2), None);
        assert_eq!(XMemFreePhysicalPages(3, a.as_ptr()), 1);
        assert_eq!(XMemFreePhysicalPages(1, b.as_ptr()), 1);
    }

    #[test]
    fn writes_cross_runs_and_allocations() {
        let (a, b) = (allocate(2), allocate(1));
        let pages = [a[0], a[1], b[0]];
        let data: Vec<u8> = (0..2 * PAGE + 64).map(|i| (i * 7 % 251) as u8).collect();
        let offset = PAGE - 32;
        // Past the third page: the second run is too short.
        assert!(!write_physical(&pages, offset, &data));
        assert!(write_physical(&pages, offset, &data[..2 * PAGE]));
        let seen: Vec<u8> = pages.iter().flat_map(|&p| page_bytes(p)).collect();
        assert!(seen[..offset].iter().all(|&b| b == 0), "before the offset");
        assert_eq!(seen[offset..offset + 2 * PAGE], data[..2 * PAGE]);
        assert!(seen[offset + 2 * PAGE..].iter().all(|&b| b == 0), "after");
        assert!(
            !write_physical(&[a[0], 0x10], 0, &data[..PAGE + 1]),
            "unknown page"
        );
        assert_eq!(XMemFreePhysicalPages(2, a.as_ptr()), 1);
        assert_eq!(XMemFreePhysicalPages(1, b.as_ptr()), 1);
    }

    /// A page mapped at a second address while a window writes it is
    /// reported there too when the window is done (the title maps streamed
    /// pages into its GPU heap while their reads decode); an address that
    /// stopped showing it meanwhile is not.
    #[test]
    fn page_windows_report_where_the_pages_show_when_done() {
        let pages = allocate(1);
        let reserve = || {
            XMemVirtualAlloc(
                std::ptr::null_mut(),
                PAGE,
                MEM_RESERVE | PAGES_64K,
                0,
                PAGE_READWRITE,
            )
        };
        let (a, b) = (reserve(), reserve());
        assert!(!a.is_null() && !b.is_null());
        assert_eq!(XMemMapPhysicalPages(a, 1, pages.as_ptr()), a);
        REPORTED.with(|r| r.borrow_mut().clear());
        let mut window = page_window(&pages, 0, 16).expect("one run");
        // Mapped at `b`, unmapped from `a`, while the bytes are written.
        assert_eq!(XMemMapPhysicalPages(b, 1, pages.as_ptr()), b);
        assert_eq!(XMemMapPhysicalPages(a, 1, std::ptr::null()), a);
        window.bytes().fill(3);
        drop(window);
        assert_eq!(REPORTED.with(|r| r.borrow().clone()), [b as usize]);
        // SAFETY: mapped read-write.
        assert_eq!(unsafe { b.cast::<u8>().read() }, 3);
        // `write_physical` the same way.
        REPORTED.with(|r| r.borrow_mut().clear());
        assert!(write_physical(&pages, 0, &[4]));
        assert_eq!(REPORTED.with(|r| r.borrow().clone()), [b as usize]);
        assert_eq!(XMemMapPhysicalPages(b, 1, std::ptr::null()), b);
        assert_eq!(XMemFreePhysicalPages(1, pages.as_ptr()), 1);
    }

    #[test]
    fn page_windows_cover_one_run_in_place() {
        let (a, b) = (allocate(3), allocate(1));
        let mut window = page_window(&a, PAGE + 3, PAGE).expect("one run");
        assert_eq!(window.bytes().len(), PAGE);
        window.bytes().fill(0x5a);
        drop(window);
        let (second, third) = (page_bytes(a[1]), page_bytes(a[2]));
        assert!(second[..3].iter().all(|&b| b == 0));
        assert!(second[3..].iter().all(|&b| b == 0x5a));
        assert!(third[..3].iter().all(|&b| b == 0x5a));
        assert!(third[3..].iter().all(|&b| b == 0));
        // Several runs, or an unknown page: no window.
        assert!(page_window(&[a[0], b[0]], 0, PAGE + 1).is_none());
        assert!(page_window(&[a[1], a[0]], 0, PAGE + 1).is_none());
        assert!(page_window(&[0x10], 0, 1).is_none());
        // One page of several is still one run.
        assert!(page_window(&[a[0], b[0]], 0, PAGE).is_some());
        assert_eq!(XMemFreePhysicalPages(3, a.as_ptr()), 1);
        assert_eq!(XMemFreePhysicalPages(1, b.as_ptr()), 1);
    }

    #[test]
    fn page_writes_share_one_view_of_their_section() {
        let a = allocate(3);
        let slice = |p| page_table().pages.get(&p).cloned().expect("known page");
        let section = Arc::downgrade(&slice(a[0]).section);
        // Every request on the section goes through the same view, each
        // page at its place in it, with no view of its own to map and unmap.
        let (first, third) = (
            View::map(&slice(a[0]), 1).expect("mapped"),
            View::map(&slice(a[2]), 1).expect("mapped"),
        );
        assert_eq!(third.at - first.at, 2 * PAGE);
        assert!(first.own.is_none() && third.own.is_none());
        let at = first.at;
        drop((first, third));
        assert_eq!(View::map(&slice(a[1]), 2).expect("mapped").at, at + PAGE);
        assert!(write_physical(&a, 2 * PAGE + 3, &[4, 5]));
        assert_eq!(page_bytes(a[2])[3..5], [4, 5]);
        // The view lives as long as the section: until its pages are freed.
        assert_eq!(XMemFreePhysicalPages(3, a.as_ptr()), 1);
        assert!(section.upgrade().is_none());
    }

    #[test]
    fn long_views_track_aliases_and_directstorage_writes() {
        let _modes = ModeGuard::set(Some(true), Some(true));
        let count = SHORT + 8;
        let pages = allocate(count);
        let reserve = |n: usize| {
            XMemVirtualAlloc(
                std::ptr::null_mut(),
                n * PAGE,
                MEM_RESERVE | PAGES_64K,
                0,
                PAGE_READWRITE,
            )
        };
        let (big, single) = (reserve(count), reserve(1));
        assert!(!big.is_null() && !single.is_null());
        assert_eq!(XMemMapPhysicalPages(big, count, pages.as_ptr()), big);
        let target = count - 3;
        assert_eq!(
            XMemMapPhysicalPages(single, 1, pages[target..].as_ptr()),
            single
        );

        let big_page = big.wrapping_byte_add(target * PAGE);
        let mut out = Vec::new();
        aliases(big_page as usize, PAGE, &mut out);
        assert_eq!(out, [(big_page as usize, single as usize)]);
        out.clear();
        aliases(single as usize, PAGE, &mut out);
        assert_eq!(out, [(single as usize, big_page as usize)]);

        REPORTED.with(|r| r.borrow_mut().clear());
        assert!(write_physical(&pages[target..target + 1], 0, &[0x77]));
        let mut reported = REPORTED.with(|r| r.borrow().clone());
        reported.sort_unstable();
        let mut expected = [big_page as usize, single as usize];
        expected.sort_unstable();
        assert_eq!(reported, expected);

        assert_eq!(XMemMapPhysicalPages(single, 1, std::ptr::null()), single);
        assert_eq!(XMemMapPhysicalPages(big, count, std::ptr::null()), big);
        // SAFETY: releasing test reservations.
        unsafe {
            VirtualFree(single, 0, MEM_RELEASE);
            VirtualFree(big, 0, MEM_RELEASE);
        }
        assert_eq!(XMemFreePhysicalPages(count, pages.as_ptr()), 1);
    }

    #[test]
    fn concurrent_page_lookup_under_mapping_contention() {
        use std::sync::atomic::AtomicU64;
        use std::time::{Duration, Instant};

        let pages = allocate(256);
        let mapped = XMemVirtualAlloc(
            std::ptr::null_mut(),
            256 * PAGE,
            MEM_RESERVE | PAGES_64K,
            0,
            PAGE_READWRITE,
        ) as usize;
        assert_ne!(mapped, 0);

        for mapping in [false, true] {
            let (done, waited_ns, worst_ns, maps) = (
                AtomicU64::new(0),
                AtomicU64::new(0),
                AtomicU64::new(0),
                AtomicU64::new(0),
            );
            let until = Instant::now() + Duration::from_millis(250);
            std::thread::scope(|s| {
                if mapping {
                    s.spawn(|| {
                        let at: Vec<*mut c_void> = (0..64)
                            .map(|i| (mapped + i * PAGE) as *mut c_void)
                            .collect();
                        while Instant::now() < until {
                            assert_eq!(
                                XMemMapPhysicalPagesScatter(at.as_ptr(), 64, pages.as_ptr()),
                                1
                            );
                            assert_eq!(
                                XMemMapPhysicalPagesScatter(at.as_ptr(), 64, std::ptr::null()),
                                1
                            );
                            maps.fetch_add(128, Ordering::Relaxed);
                        }
                    });
                }
                for t in 0..16 {
                    let (pages, done, waited_ns, worst_ns) = (&pages, &done, &waited_ns, &worst_ns);
                    s.spawn(move || {
                        let base_idx = 64 + t * 8;
                        let mut i = base_idx;
                        while Instant::now() < until {
                            let page = &pages[i..=i];
                            i = if i == base_idx + 7 { base_idx } else { i + 1 };
                            let t0 = Instant::now();
                            let mut window = page_window(page, 0, PAGE).unwrap();
                            let lookup_t = t0.elapsed();
                            window.bytes()[0] = t as u8;
                            let t1 = Instant::now();
                            drop(window);
                            let total_ns = (lookup_t + t1.elapsed()).as_nanos() as u64;
                            waited_ns.fetch_add(total_ns, Ordering::Relaxed);
                            worst_ns.fetch_max(total_ns, Ordering::Relaxed);
                            done.fetch_add(1, Ordering::Relaxed);
                        }
                    });
                }
            });
            let n = done.load(Ordering::Relaxed).max(1);
            println!(
                "[xmem lookup bench] mapping={mapping}: {:.0} req/s, {:.2} us/req lookup+notice (worst {:.2} ms), {:.0} maps/s",
                n as f64 / 0.25,
                waited_ns.load(Ordering::Relaxed) as f64 / n as f64 / 1e3,
                worst_ns.load(Ordering::Relaxed) as f64 / 1e6,
                maps.load(Ordering::Relaxed) as f64 / 0.25,
            );
            assert!(n > 100);
        }

        // SAFETY: releasing unmapped test placeholder reservation.
        unsafe { VirtualFree(mapped as *mut c_void, 0, MEM_RELEASE) };
        assert_eq!(XMemFreePhysicalPages(256, pages.as_ptr()), 1);
    }

    #[test]
    fn pages_mapped_again_hold_what_they_held() {
        let pages = allocate(3);
        let a = XMemVirtualAlloc(
            std::ptr::null_mut(),
            8 * PAGE,
            MEM_RESERVE | PAGES_64K,
            0,
            PAGE_READWRITE,
        );
        assert!(!a.is_null());
        let at = |k: usize| a.wrapping_byte_add(k * PAGE);
        for round in 0..6_usize {
            for (k, page) in pages.iter().enumerate() {
                let to = at((round + 2 * k) % 8);
                assert_eq!(XMemMapPhysicalPages(to, 1, page), to);
                let first = to.cast::<u64>();
                // SAFETY: mapped read-write; the far end of the page too.
                unsafe {
                    if round > 0 {
                        assert_eq!(first.read(), (round * 16 + k - 16) as u64);
                        assert_eq!(first.add(PAGE / 8 - 1).read(), k as u64);
                    }
                    first.write((round * 16 + k) as u64);
                    first.add(PAGE / 8 - 1).write(k as u64);
                }
            }
            for k in 0..pages.len() {
                let to = at((round + 2 * k) % 8);
                assert_eq!(XMemMapPhysicalPages(to, 1, std::ptr::null()), to);
                assert_eq!(query(to).host.State, MEM_RESERVE);
            }
            if round == 3 {
                const MEM_COALESCE_PLACEHOLDERS: u32 = 1;
                // SAFETY: placeholders of ours, nothing mapped in them.
                unsafe { VirtualFree(a, 8 * PAGE, MEM_RELEASE | MEM_COALESCE_PLACEHOLDERS) };
            }
        }
        // SAFETY: releasing test reservation.
        unsafe { VirtualFree(a, 0, MEM_RELEASE) };
        assert_eq!(XMemFreePhysicalPages(3, pages.as_ptr()), 1);
    }

    #[test]
    fn a_page_mapped_again_takes_its_own_placeholder() {
        let pages = allocate(1);
        let a = XMemVirtualAlloc(
            std::ptr::null_mut(),
            2 * PAGE,
            MEM_RESERVE | PAGES_64K,
            0,
            PAGE_READWRITE,
        );
        assert!(!a.is_null());
        assert_eq!(XMemMapPhysicalPages(a, 1, pages.as_ptr()), a);
        let section = page_table().pages[&pages[0]].section.clone();
        assert_eq!(section.committed[0].load(Ordering::Relaxed), 1);
        assert_eq!(XMemMapPhysicalPages(a, 1, std::ptr::null()), a);
        assert_eq!(physical().holes.get(&(a as usize)), Some(&PAGE));
        assert_eq!(XMemMapPhysicalPages(a, 1, pages.as_ptr()), a);
        assert_eq!(physical().holes.get(&(a as usize)), None);
        assert_eq!(query(a).host.State, MEM_COMMIT);
        // SAFETY: mapped read-write.
        unsafe { a.cast::<u32>().write(7) };

        // If the title coalesces the placeholder with its neighbour, the
        // direct replace fails and splits cleanly as before.
        assert_eq!(XMemMapPhysicalPages(a, 1, std::ptr::null()), a);
        assert_eq!(physical().holes.get(&(a as usize)), Some(&PAGE));
        const MEM_COALESCE_PLACEHOLDERS: u32 = 1;
        // SAFETY: placeholders of ours, nothing mapped in them.
        let joined = unsafe { VirtualFree(a, 2 * PAGE, MEM_RELEASE | MEM_COALESCE_PLACEHOLDERS) };
        assert_ne!(joined, 0);
        assert_eq!(XMemMapPhysicalPages(a, 1, pages.as_ptr()), a);
        // SAFETY: mapped read-write.
        assert_eq!(unsafe { a.cast::<u32>().read() }, 7);
        let second = a.wrapping_byte_add(PAGE);
        assert_eq!(XMemMapPhysicalPages(second, 1, pages.as_ptr()), second);
        assert_eq!(XMemMapPhysicalPages(a, 2, std::ptr::null()), a);
        // SAFETY: releasing test reservation.
        unsafe { VirtualFree(a, 0, MEM_RELEASE) };
        assert_eq!(XMemFreePhysicalPages(1, pages.as_ptr()), 1);
    }

    #[test]
    fn batched_runs_coalesce_split_placeholders_and_preserve_watch_aliases() {
        let _modes = ModeGuard::set(Some(true), Some(true));
        let pages = allocate(4);
        let reserve = || {
            XMemVirtualAlloc(
                std::ptr::null_mut(),
                4 * PAGE,
                MEM_RESERVE | PAGES_64K,
                0,
                PAGE_READWRITE,
            )
        };
        let (a, b) = (reserve(), reserve());
        assert!(!a.is_null() && !b.is_null());
        let at = |p: *mut c_void, k: usize| p.wrapping_byte_add(k * PAGE);

        // Split both `a` and `b` into 4 individual 1-page placeholders first
        // using reversed scatter mappings.
        let rev_pages = [pages[3], pages[2], pages[1], pages[0]];
        assert_eq!(XMemMapPhysicalPages(a, 4, rev_pages.as_ptr()), a);
        assert_eq!(XMemMapPhysicalPages(b, 4, rev_pages.as_ptr()), b);
        assert_eq!(XMemMapPhysicalPages(a, 4, std::ptr::null()), a);
        assert_eq!(XMemMapPhysicalPages(b, 4, std::ptr::null()), b);

        // Now map all 4 contiguous pages over `a` and alias `a -> b`: both
        // must coalesce the 4 split placeholders into a single 4-page view!
        let before = MAPS.with(std::cell::Cell::get);
        let scatter_addrs = [at(a, 0), at(a, 1), at(a, 2), at(a, 3)];
        assert_eq!(
            XMemMapPhysicalPagesScatter(scatter_addrs.as_ptr(), 4, pages.as_ptr()),
            1
        );
        assert_eq!(
            MAPS.with(std::cell::Cell::get),
            before + 1,
            "contiguous scatter coalesced into 1 MapViewOfFile3 call"
        );
        assert!(map_alias(b as u64, a as u64, 4 * PAGE as u64));
        assert_eq!(
            MAPS.with(std::cell::Cell::get),
            before + 2,
            "4-page map_alias coalesced into 1 MapViewOfFile3 call"
        );

        // Verify alias links and byte sharing across all 4 pages, plus partial
        // source unmap handling in `alias_in_runs`.
        for k in 0..4 {
            // SAFETY: both `a` and `b` are mapped read-write across 4 pages.
            unsafe {
                at(a, k).cast::<u8>().add(32).write(0x30 + k as u8);
                assert_eq!(at(b, k).cast::<u8>().add(32).read(), 0x30 + k as u8);
            }
            let mut linked = Vec::new();
            aliases(at(a, k) as usize, PAGE, &mut linked);
            assert_eq!(linked, [(at(a, k) as usize, at(b, k) as usize)]);
        }

        // Unmap middle page of `a`; `map_alias` across a gap maps the valid
        // prefix, unmaps the missing tail, and returns false.
        assert!(map_alias(at(a, 2) as u64, 0, PAGE as u64));
        assert!(!map_alias(b as u64, a as u64, 3 * PAGE as u64));
        assert_eq!(query(at(b, 0)).host.State, MEM_COMMIT);
        assert_eq!(query(at(b, 1)).host.State, MEM_COMMIT);
        assert_eq!(query(at(b, 2)).host.State, MEM_RESERVE);
        assert_eq!(query(at(b, 3)).host.State, MEM_COMMIT);

        assert_eq!(XMemMapPhysicalPages(a, 4, std::ptr::null()), a);
        assert_eq!(XMemMapPhysicalPages(b, 4, std::ptr::null()), b);
        // SAFETY: releasing test reservations.
        unsafe {
            VirtualFree(a, 0, MEM_RELEASE);
            VirtualFree(b, 0, MEM_RELEASE);
        }
        assert_eq!(XMemFreePhysicalPages(4, pages.as_ptr()), 1);
    }

    #[test]
    fn partial_remap_keeps_sibling_pages_mapped_for_concurrent_readers() {
        use std::sync::Barrier;
        use std::sync::atomic::{AtomicBool, AtomicU64};

        let pages = allocate(5);
        let words = PAGE / size_of::<u64>();
        for (idx, &page) in pages.iter().enumerate() {
            let fill = 0x1111_1111_1111_1111_u64 * (idx as u64 + 1);
            let slice = page_table().pages[&page].clone();
            let mut view = View::map(&slice, 1).expect("mapped");
            // SAFETY: `view` holds one committed 64 KiB page (`words` u64s).
            let buf = unsafe { std::slice::from_raw_parts_mut(view.bytes().as_mut_ptr().cast::<u64>(), words) };
            buf.fill(fill);
        }

        let base = XMemVirtualAlloc(
            std::ptr::null_mut(),
            3 * PAGE,
            MEM_RESERVE | PAGES_64K,
            0,
            PAGE_READWRITE,
        );
        assert!(!base.is_null());
        let initial = [pages[0], pages[1], pages[2]];
        assert_eq!(XMemMapPhysicalPages(base, 3, initial.as_ptr()), base);

        let base_addr = base as usize;
        let stop = AtomicBool::new(false);
        let reads = AtomicU64::new(0);
        let ready = Barrier::new(3);

        std::thread::scope(|s| {
            for reader in 0..2_usize {
                let (stop, reads, ready) = (&stop, &reads, &ready);
                s.spawn(move || {
                    let p0 = base_addr as *const u64;
                    let p2 = (base_addr + 2 * PAGE) as *const u64;
                    let mut step = reader * 97;
                    ready.wait();
                    while !stop.load(Ordering::Acquire) {
                        let off = step % words;
                        // SAFETY: pages 0 and 2 stay mapped to `pages[0]` and
                        // `pages[2]` across every remap of page 1.
                        let (v0, v2) = unsafe {
                            (p0.add(off).read_volatile(), p2.add(off).read_volatile())
                        };
                        assert_eq!(v0, 0x1111_1111_1111_1111);
                        assert_eq!(v2, 0x3333_3333_3333_3333);
                        step = step.wrapping_add(131);
                        reads.fetch_add(1, Ordering::Relaxed);
                    }
                });
            }

            ready.wait();
            let middle = base.wrapping_byte_add(PAGE);
            for _ in 0..400 {
                // 3-page `XMemMapPhysicalPages` where pages 0 and 2 already match:
                // only page 1 is unmapped and remapped (`MAPS` + 1).
                let before = MAPS.with(std::cell::Cell::get);
                let next_a = [pages[0], pages[3], pages[2]];
                assert_eq!(XMemMapPhysicalPages(base, 3, next_a.as_ptr()), base);
                assert_eq!(MAPS.with(std::cell::Cell::get), before + 1);
                // SAFETY: page 1 now shows `pages[3]`.
                assert_eq!(unsafe { middle.cast::<u64>().read_volatile() }, 0x4444_4444_4444_4444);

                let next_b = [pages[0], pages[4], pages[2]];
                assert_eq!(XMemMapPhysicalPages(base, 3, next_b.as_ptr()), base);
                assert_eq!(MAPS.with(std::cell::Cell::get), before + 2);
                // SAFETY: page 1 now shows `pages[4]`.
                assert_eq!(unsafe { middle.cast::<u64>().read_volatile() }, 0x5555_5555_5555_5555);

                // Single-page `XMemMapPhysicalPages` on page 1 also leaves sibling
                // pages 0 and 2 untouched because each 64 KiB page has its own view.
                assert_eq!(XMemMapPhysicalPages(middle, 1, &pages[1]), middle);
                assert_eq!(MAPS.with(std::cell::Cell::get), before + 3);
                // SAFETY: page 1 now shows `pages[1]`.
                assert_eq!(unsafe { middle.cast::<u64>().read_volatile() }, 0x2222_2222_2222_2222);
            }
            stop.store(true, Ordering::Release);
        });

        assert!(reads.load(Ordering::Relaxed) > 0);
        assert_eq!(XMemMapPhysicalPages(base, 3, std::ptr::null()), base);
        // SAFETY: releasing test reservation.
        unsafe { VirtualFree(base, 0, MEM_RELEASE) };
        assert_eq!(XMemFreePhysicalPages(5, pages.as_ptr()), 1);
    }

    #[test]
    fn lean_and_batched_runs_remap_benchmark() {
        const PAGES_N: usize = 64;
        const SINGLE_ROUNDS: usize = 60;
        const BATCH_ROUNDS: usize = 120;

        let pages = allocate(PAGES_N);
        let reserve = || {
            XMemVirtualAlloc(
                std::ptr::null_mut(),
                PAGES_N * PAGE,
                MEM_RESERVE | PAGES_64K,
                0,
                PAGE_READWRITE,
            )
        };
        let (a, b) = (reserve(), reserve());
        assert!(!a.is_null() && !b.is_null());

        // 1) Single-page remap benchmark: VICTORY_MAP_LEAN=0 vs VICTORY_MAP_LEAN=1.
        for lean_on in [false, true] {
            let _modes = ModeGuard::set(Some(lean_on), Some(false));
            let start = std::time::Instant::now();
            for round in 0..SINGLE_ROUNDS {
                for (k, page) in pages.iter().enumerate() {
                    let to = a.wrapping_byte_add(((k + round) % PAGES_N) * PAGE);
                    assert_eq!(XMemMapPhysicalPages(to, 1, page), to);
                }
                for k in 0..PAGES_N {
                    let to = a.wrapping_byte_add(k * PAGE);
                    assert_eq!(XMemMapPhysicalPages(to, 1, std::ptr::null()), to);
                }
            }
            let elapsed = start.elapsed();
            let us_per_page = elapsed.as_secs_f64() * 1e6 / (SINGLE_ROUNDS * PAGES_N) as f64;
            println!(
                "[xmem single-page remap] lean={lean_on}: {us_per_page:.2} us/page (map+unmap)"
            );
        }

        // 2) Streaming pool -> tiled-heap `XMemMapPhysicalPagesScatter` + `map_alias`
        //    benchmark: baseline (lean=false, runs=false) vs default (lean=true, runs=false)
        //    vs opt-in runs (lean=true, runs=true).
        let scatter_addrs: Vec<*mut c_void> = (0..PAGES_N)
            .map(|k| a.wrapping_byte_add(k * PAGE))
            .collect();
        for (label, lean_on, runs_on) in [
            ("baseline (lean=0, runs=0)", false, false),
            ("default  (lean=1, runs=0)", true, false),
            ("opt-in   (lean=1, runs=1)", true, true),
        ] {
            let _modes = ModeGuard::set(Some(lean_on), Some(runs_on));
            let maps_before = MAPS.with(std::cell::Cell::get);
            let start = std::time::Instant::now();
            for _ in 0..BATCH_ROUNDS {
                assert_eq!(
                    XMemMapPhysicalPagesScatter(scatter_addrs.as_ptr(), PAGES_N, pages.as_ptr()),
                    1
                );
                assert!(map_alias(b as u64, a as u64, (PAGES_N * PAGE) as u64));
                assert!(map_alias(b as u64, 0, (PAGES_N * PAGE) as u64));
                assert_eq!(
                    XMemMapPhysicalPagesScatter(scatter_addrs.as_ptr(), PAGES_N, std::ptr::null()),
                    1
                );
            }
            let elapsed = start.elapsed();
            let view_maps = MAPS.with(std::cell::Cell::get) - maps_before;
            let total_pages = BATCH_ROUNDS * PAGES_N * 2;
            let us_per_page = elapsed.as_secs_f64() * 1e6 / total_pages as f64;
            let pages_per_sec = total_pages as f64 / elapsed.as_secs_f64().max(1e-6);
            println!(
                "[xmem batch remap] {label}: {us_per_page:.2} us/page ({pages_per_sec:.0} pages/s, MapViewOfFile3 calls={view_maps})"
            );
        }

        // SAFETY: releasing test reservations.
        unsafe {
            VirtualFree(a, 0, MEM_RELEASE);
            VirtualFree(b, 0, MEM_RELEASE);
        }
        assert_eq!(XMemFreePhysicalPages(PAGES_N, pages.as_ptr()), 1);
    }

    #[test]
    fn own_section_views_start_above_two_tib_under_wine() {
        use windows_sys::Win32::System::Memory::UnmapViewOfFile;
        let pages = allocate(2);
        let slice = page_table().pages.get(&pages[1]).cloned().expect("known page");
        let whole = slice.section.whole().expect("whole view");
        // SAFETY: our section's second page, mapped and unmapped here to cover
        // the per-slice `View::map` fallback path as well.
        let fallback = unsafe { map_own_view(slice.section.handle, PAGE as u64, PAGE) };
        assert!(!fallback.Value.is_null());
        // With `VICTORY_PHYSICAL_TILES=1` the whole view is the section's
        // GPU heap, not a view of ours: no floor.
        if on_wine() && slice.section.shared.is_none() {
            assert!(
                whole >= OWN_VIEW_FLOOR,
                "whole view {whole:#x} below {OWN_VIEW_FLOOR:#x}"
            );
            assert!(
                fallback.Value as usize >= OWN_VIEW_FLOOR,
                "fallback view {:p} below {OWN_VIEW_FLOOR:#x}",
                fallback.Value
            );
        }
        // SAFETY: mapped just above.
        unsafe { UnmapViewOfFile(fallback) };
        assert_eq!(XMemFreePhysicalPages(2, pages.as_ptr()), 1);
    }
}

/// The space below 4 GiB, which the title's 32-bit memory (`XMemVirtualAlloc`
/// with the 32-bit address flag) has to fit in.
const FOUR_GIB: usize = 1 << 32;

/// Guards over the space below 4 GiB. Windows places allocations bottom up,
/// so the host's own memory (the driver's, heaps, the shader and pipeline
/// caches read at startup) filled it first: with a large pipeline cache only
/// 22 MB was left, in pieces, when Grand Theft Auto VI asked for a 185 MB
/// 32-bit placeholder at startup, the request was refused
/// (`ERROR_NOT_ENOUGH_MEMORY`) and the title stopped at an assertion. The
/// debug layer's memory did the same. So the free space below 4 GiB is
/// reserved at startup, past [`low_guards::SLACK`] (or
/// [`low_guards::WINE_SLACK`] under Wine) left for the host, and a 32-bit
/// request that finds no room releases guards and tries again.
/// `VICTORY_LOW_GUARDS=0` (`false`, `off`) disables these guards.
pub mod low_guards {
    use std::sync::Mutex;

    use windows_sys::Win32::System::Memory::{
        MEM_FREE, MEM_RELEASE, MEM_RESERVE, MEMORY_BASIC_INFORMATION, PAGE_NOACCESS, VirtualAlloc,
        VirtualFree, VirtualQuery,
    };

    use super::FOUR_GIB;

    /// Free space left to the host below 4 GiB on native Windows (the lowest
    /// free space).
    pub(crate) const SLACK: usize = 256 << 20;
    /// Free space left to the host below 4 GiB under Wine (`2.75 GiB`). Host
    /// GPU drivers such as D3DMetal/Metal allocate outside Wine above 4 GiB,
    /// while Wine's own heap segments and thread stacks search bottom-up from
    /// `0x10000`: leaving 2.75 GiB at the bottom keeps them out of gigabytes
    /// of host mappings on unpatched Wine (`try_map_free_area` under
    /// `virtual_mutex`), while reserving the top `~1 GiB` below 4 GiB for the
    /// title's 32-bit `XMemVirtualAlloc` requests.
    pub(crate) const WINE_SLACK: usize = 2816 << 20;
    /// The allocation granularity of Windows.
    const GRANULE: usize = 64 << 10;

    /// The guards, `(base, bytes)`.
    static GUARDS: Mutex<Vec<(usize, usize)>> = Mutex::new(Vec::new());

    fn guards() -> std::sync::MutexGuard<'static, Vec<(usize, usize)>> {
        GUARDS.lock().unwrap_or_else(|e| e.into_inner())
    }

    /// Whether `VICTORY_LOW_GUARDS` leaves low-address guards enabled.
    fn enabled_from(value: Option<&str>) -> bool {
        !value.is_some_and(|v| {
            let v = v.trim();
            v == "0" || v.eq_ignore_ascii_case("false") || v.eq_ignore_ascii_case("off")
        })
    }

    /// Low-address slack to leave unguarded at the bottom of `< 4 GiB`.
    const fn slack_for(wine: bool) -> usize {
        if wine { WINE_SLACK } else { SLACK }
    }

    /// Skips the lowest `slack` free bytes across ascending `runs` so the
    /// free headroom stays at the bottom of `< 4 GiB` and the guarded region
    /// sits at the top.
    fn plan_guards(runs: &[(usize, usize)], mut slack: usize) -> Vec<(usize, usize)> {
        let mut planned = Vec::new();
        for &(mut base, mut bytes) in runs {
            let left = slack.min(bytes);
            (base, bytes, slack) = (base + left, bytes - left, slack - left);
            if bytes > 0 {
                planned.push((base, bytes));
            }
        }
        planned
    }

    /// The free runs in `[from, to)`, granule-aligned.
    fn free_runs(from: usize, to: usize) -> Vec<(usize, usize)> {
        let mut runs = Vec::new();
        let mut at = from.max(GRANULE);
        while at < to {
            // SAFETY: plain data filled in by the query.
            let mut m: MEMORY_BASIC_INFORMATION = unsafe { std::mem::zeroed() };
            // SAFETY: a valid out pointer of the right size.
            if unsafe { VirtualQuery(at as *const _, &mut m, size_of_val(&m)) } == 0 {
                break;
            }
            let base = m.BaseAddress as usize;
            let end = (base + m.RegionSize).min(to);
            if m.State == MEM_FREE {
                let (start, end) = (
                    base.max(from).next_multiple_of(GRANULE),
                    end & !(GRANULE - 1),
                );
                if end > start {
                    runs.push((start, end - start));
                }
            }
            at = base + m.RegionSize;
        }
        runs
    }

    /// Reserves `[base, base + bytes)` as a guard; false if something took it.
    fn guard(base: usize, bytes: usize) -> bool {
        // SAFETY: a reservation of free address space; nothing refers to it.
        let at = unsafe { VirtualAlloc(base as *const _, bytes, MEM_RESERVE, PAGE_NOACCESS) };
        at as usize == base
    }

    /// Reserves the free space below 4 GiB past the lowest [`SLACK`] (or
    /// [`WINE_SLACK`] under Wine). Call once, before the host allocates much
    /// (the GPU device, the caches).
    pub fn install() {
        if !enabled_from(std::env::var("VICTORY_LOW_GUARDS").ok().as_deref()) {
            tracing::info!("space below 4 GiB left unguarded (VICTORY_LOW_GUARDS)");
            return;
        }
        let mut guards = guards();
        let slack = slack_for(super::on_wine());
        for (base, bytes) in plan_guards(&free_runs(0, FOUR_GIB), slack) {
            if guard(base, bytes) {
                guards.push((base, bytes));
            }
        }
        let total: usize = guards.iter().map(|g| g.1).sum();
        tracing::info!(
            guards = guards.len(),
            mb = total >> 20,
            slack_mb = slack >> 20,
            "space below 4 GiB kept for the title"
        );
    }

    /// Try the smallest fitting guards first, then try smaller guards in
    /// descending size order. A replacement guard cannot be retried here.
    pub(crate) fn retry(
        bytes: usize,
        mut allocate: impl FnMut() -> *mut std::ffi::c_void,
    ) -> *mut std::ffi::c_void {
        use windows_sys::Win32::Foundation::{GetLastError, SetLastError};

        // Preserve the allocation error across address queries and reservations.
        // SAFETY: reads this thread's last-error value.
        let mut error = unsafe { GetLastError() };
        let mut guards = guards();
        let mut candidates = guards.clone();
        candidates.sort_unstable_by_key(|&(_, size)| {
            let fits = size >= bytes;
            (!fits, if fits { size } else { usize::MAX - size })
        });
        for (base, size) in candidates {
            // SAFETY: our own reservation, with no allocations inside it.
            if unsafe { VirtualFree(base as *mut _, 0, MEM_RELEASE) } == 0 {
                continue;
            }
            guards.retain(|&(b, _)| b != base);
            let at = allocate();
            // SAFETY: reads this thread's last-error value.
            error = unsafe { GetLastError() };
            for (start, len) in free_runs(base, base + size) {
                if guard(start, len) {
                    guards.push((start, len));
                }
            }
            if !at.is_null() {
                return at;
            }
        }
        // SAFETY: sets this thread's last-error value.
        unsafe { SetLastError(error) };
        std::ptr::null_mut()
    }

    #[cfg(test)]
    mod tests {
        use super::*;

        /// Set in the child a test below relaunches itself in. These tests
        /// reserve and release address space of the whole process, so in the
        /// parallel suite another test can take it between their steps; the
        /// child runs the one test alone (`--exact`), in a clean address
        /// space.
        const CHILD: &str = "VICTORY_TEST_LOW_GUARDS_CHILD";

        /// Runs the test `name` in a child process, returning true in the
        /// parent (the test then returns); in the child returns false, so
        /// the test's body runs there.
        fn alone(name: &str) -> bool {
            if std::env::var_os(CHILD).is_some() {
                return false;
            }
            let result = std::process::Command::new(std::env::current_exe().unwrap())
                .args(["--exact", name, "--nocapture"])
                .env(CHILD, "1")
                .output()
                .expect("run low-guards test in a child");
            let stdout = String::from_utf8_lossy(&result.stdout);
            assert!(
                result.status.success() && stdout.contains("1 passed"),
                "low-guards child {}: {stdout}{}",
                result.status,
                String::from_utf8_lossy(&result.stderr)
            );
            true
        }

        #[test]
        fn switch_and_wine_slack_leave_bottom_free_and_guard_top_gib() {
            assert!(enabled_from(None));
            assert!(enabled_from(Some("")));
            assert!(enabled_from(Some("1")));
            assert!(enabled_from(Some("true")));
            assert!(enabled_from(Some("on")));
            for off in ["0", "false", "FALSE", "off", "Off", " 0 "] {
                assert!(!enabled_from(Some(off)), "{off:?}");
            }
            assert_eq!(slack_for(false), 256 << 20);
            assert_eq!(slack_for(true), 2816 << 20);

            // Across two free runs below 4 GiB (~4,073 MiB total, as at Wine
            // startup), `plan_guards` leaves the lowest 2,816 MiB unguarded
            // for bottom-up allocations and guards the top >1 GiB up to 4 GiB.
            let runs = [
                (GRANULE, (2047 << 20) - GRANULE),
                (2048 << 20, (2048 << 20) - GRANULE),
            ];
            let planned = plan_guards(&runs, WINE_SLACK);
            assert_eq!(planned.len(), 1);
            let (guard_base, guard_len) = planned[0];
            assert_eq!(guard_base + guard_len, FOUR_GIB - GRANULE);
            assert!(guard_base > (2816 << 20));
            assert!(guard_len >= (1024 << 20));
        }

        #[test]
        fn a_refused_retry_leaves_the_guards_in_place() {
            if alone("modules::xmem::low_guards::tests::a_refused_retry_leaves_the_guards_in_place")
            {
                return;
            }
            // A known hole of four granules, reserved and registered as a
            // guard directly: releasing and re-reserving it would leave a
            // window another allocation could take.
            // SAFETY: a fresh reservation; nothing refers to it.
            let hole =
                unsafe { VirtualAlloc(std::ptr::null(), 4 * GRANULE, MEM_RESERVE, PAGE_NOACCESS) };
            assert!(!hole.is_null());
            let first = hole as usize;
            guards().push((first, 4 * GRANULE));
            let mut attempts = 0;
            let result = retry(4 * GRANULE, || {
                attempts += 1;
                std::ptr::null_mut()
            });
            assert!(result.is_null());
            assert_eq!(attempts, 1);
            assert!(guards().contains(&(first, 4 * GRANULE)));
            guards().retain(|&(base, _)| base != first);
            // SAFETY: the guard the retry put back, ours.
            assert_ne!(unsafe { VirtualFree(hole, 0, MEM_RELEASE) }, 0);
        }

        /// A released guard is guarded again around what took part of it.
        #[test]
        fn a_released_guard_is_guarded_again_around_what_took_it() {
            if alone(
                "modules::xmem::low_guards::tests::a_released_guard_is_guarded_again_around_what_took_it",
            ) {
                return;
            }
            // A known hole of four granules, reserved and registered as a
            // guard directly.
            // SAFETY: a fresh reservation; nothing refers to it.
            let hole =
                unsafe { VirtualAlloc(std::ptr::null(), 4 * GRANULE, MEM_RESERVE, PAGE_NOACCESS) }
                    as usize;
            assert_ne!(hole, 0);
            guards().push((hole, 4 * GRANULE));
            let taken = retry(GRANULE, || {
                // SAFETY: a reservation of the first granule at a known address.
                unsafe { VirtualAlloc(hole as *const _, GRANULE, MEM_RESERVE, PAGE_NOACCESS) }
            }) as usize;
            assert_eq!(taken, hole);
            assert!(guards().contains(&(hole + GRANULE, 3 * GRANULE)));
            guards().retain(|&(b, _)| b != hole + GRANULE);
            // SAFETY: ours.
            unsafe {
                VirtualFree(taken as *mut _, 0, MEM_RELEASE);
                VirtualFree((hole + GRANULE) as *mut _, 0, MEM_RELEASE);
            }
        }
    }
}
