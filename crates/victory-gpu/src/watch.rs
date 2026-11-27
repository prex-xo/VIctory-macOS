//! Which title memory changed since a host copy took it (decision 0007,
//! "Copies of title memory" and "Change tracking"). Each host copy of title
//! memory (`crate::shadow`) and each texture's staging (`crate::texture`)
//! tracks its memory ([`Tracked`]): the 64 KiB pages it has not taken since
//! they last changed. Taking pages makes their blocks read-only (guards);
//! the first write to a guarded block faults into our vectored handler,
//! which gives write access back and marks the block changed in every copy
//! holding it. Writers that do not fault say so: our own and the kernel's
//! writes into title memory ([`writing`], before they write), and the
//! runtime mapping or unmapping memory ([`title_memory_remapped`]). So
//! bringing a copy up to date costs what changed since, not how much memory
//! it covers: no system call for memory nobody wrote.
//!
//! Only the runtime's GPU memory is tracked: its write-watched pools, its
//! physical-page views and their placeholders. Other memory a copy takes
//! (the C heap, where the kernel writes system call results into buffers a
//! guard would make fail) is copied whole every time.
//!
//! Blocks that cannot be made read-only (placeholders, nothing mapped)
//! read as zeros and count as taken: only a mapping changes them, which the
//! runtime reports.
//!
//! Physical pages may be mapped at several addresses (the title fills
//! pages through a pool view and maps them into its heaps), so a write at
//! one is a change at all of them: the runtime says where else memory is
//! mapped whenever it maps it ([`title_memory_aliased`]); taking a page
//! guards every address showing it, and a write at any of them marks the
//! trackers at all. (Asking the runtime at each take instead waited for
//! its lock, which the title's page mappings hold: 4 s per world frame.)
//! DirectStorage writing pages through views of the runtime's own reports
//! the addresses showing them as written.
//!
//! shortcut: a system call writing into a guarded block fails; the known
//! ones (file reads, DirectStorage) open their blocks first. Memory freed
//! and allocated again while a tracker holds it is not seen.
//!
//! Under Rosetta (Wine on macOS) memory is tracked only when asked for, with
//! the stores Rosetta cannot resume rewritten at load time: see
//! [`fault_tracking`].

use std::collections::{BTreeMap, BTreeSet};
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Arc, Condvar, LazyLock, Mutex};

/// Whether title writes may fault into a tracker: the guards here and the
/// write watch of the runtime's pools (`victory_hle`'s `XMemVirtualAlloc`,
/// which Wine emulates with page protection). Every store that hits a
/// tracked page must be resumed after the fault. Rosetta cannot resume
/// some: it prints `unexpectedly need to EmulateForward on a synchronous
/// exception` and the process hangs (VIctory-macOS issue #1). On macOS 27
/// those are AVX masked stores that start in writable memory and end in a
/// protected page, and `vcvtps2ph` stores anywhere in one; the loader
/// rewrites both as plain stores whenever tracking runs under Rosetta
/// ([`stores_need_patching`]). Earlier Rosetta also stops on a plain 32-byte
/// `vmovups` across pages (macOS 26.6, #1), and #144 hung on an M1 with
/// both switches on, so under Rosetta tracking stays off unless
/// `VICTORY_FAULT_TRACKING=1` asks for it. Without tracking, copies take
/// their memory whole every time, as they do for memory that is not the
/// runtime's (tens of GB per world frame, #50). `0` turns tracking off
/// anywhere. Tests keep it on: their stores do not cross pages.
pub fn fault_tracking() -> bool {
    static ON: LazyLock<bool> = LazyLock::new(|| {
        decide(
            std::env::var("VICTORY_FAULT_TRACKING").ok().as_deref(),
            cfg!(test) || !rosetta(),
        )
    });
    *ON
}

/// Whether tracking runs under Rosetta, so the stores Rosetta cannot resume
/// after a fault must be rewritten at load time (`victory_loader::masked`).
pub fn stores_need_patching() -> bool {
    fault_tracking() && rosetta()
}

/// `fault_tracking` from the variable's value and the default.
fn decide(var: Option<&str>, default: bool) -> bool {
    match var {
        Some("1") => true,
        Some("0") => false,
        _ => default,
    }
}

/// Whether x86-64 code here runs under Rosetta: Wine on macOS, whose
/// x86-64 Wine only runs that way on the Apple silicon Macs D3DMetal needs.
pub fn rosetta() -> bool {
    use std::ffi::{CStr, c_char};
    use windows_sys::Win32::System::LibraryLoader::{GetModuleHandleA, GetProcAddress};
    type HostVersion = unsafe extern "C" fn(*mut *const c_char, *mut *const c_char);
    // SAFETY: ntdll is always loaded and both names end with a NUL. Wine's
    // `wine_get_host_version` (cdecl, two out pointers) sets them to static
    // strings: the host's `uname` system name and release.
    unsafe {
        let Some(f) = GetProcAddress(
            GetModuleHandleA(c"ntdll.dll".as_ptr().cast()),
            c"wine_get_host_version".as_ptr().cast(),
        ) else {
            return false;
        };
        let f: HostVersion = std::mem::transmute(f);
        let (mut sysname, mut release) = (std::ptr::null(), std::ptr::null());
        f(&mut sysname, &mut release);
        !sysname.is_null() && CStr::from_ptr(sysname) == c"Darwin"
    }
}

/// The unit tracked and copied: 64 KiB, the console's page. Finer units
/// made the checks of the world's big buffers cost more than the copies
/// they saved.
pub(crate) const PAGE: usize = 64 << 10;

/// The pages a tracker has not taken since they changed, what it saves of
/// pages it armed, its bit in [`DIRTY`] (none: always taken as dirty),
/// whether it is a host buffer copy's (the only trackers [`retake`] marks),
/// and any pages retaken/remapped whose baseline hashes must be discarded.
struct Changed(
    Mutex<BTreeSet<usize>>,
    Mutex<Saved>,
    Option<u32>,
    bool,
    std::sync::atomic::AtomicBool,
    Mutex<BTreeSet<usize>>,
);

/// A bit per tracker, set while it holds changed pages: set and cleared
/// under the tracker's lock, read without one ([`Tracked::clean`]). Dense,
/// so that a submission asking about every copy its heaps reach (~14,000)
/// reads a few KB instead of a cache line per copy.
static DIRTY: [AtomicU64; DIRTY_WORDS] = [const { AtomicU64::new(0) }; DIRTY_WORDS];
const DIRTY_WORDS: usize = 1 << 14;

/// Bits of [`DIRTY`] given back, and the first never handed out.
static FREE_BITS: Mutex<(Vec<u32>, u32)> = Mutex::new((Vec::new(), 0));

fn dirty_bit() -> Option<u32> {
    let mut free = FREE_BITS.lock().unwrap_or_else(|e| e.into_inner());
    if let Some(bit) = free.0.pop() {
        return Some(bit);
    }
    let bit = free.1;
    (bit < (DIRTY_WORDS * 64) as u32).then(|| {
        free.1 += 1;
        bit
    })
}

fn set_dirty(bit: Option<u32>, dirty: bool) {
    if let Some(bit) = bit {
        let (word, mask) = (&DIRTY[bit as usize / 64], 1u64 << (bit % 64));
        if dirty {
            word.fetch_or(mask, Ordering::Release);
        } else {
            word.fetch_and(!mask, Ordering::Release);
        }
    }
}

/// Whether the tracker with bit `bit` holds no changed page (see
/// [`Tracked::clean`]).
pub(crate) fn clean_bit(bit: Option<u32>) -> bool {
    bit.is_some_and(|bit| DIRTY[bit as usize / 64].load(Ordering::Acquire) & 1 << (bit % 64) == 0)
}

impl Drop for Changed {
    fn drop(&mut self) {
        if let Some(bit) = self.2 {
            set_dirty(Some(bit), false);
            FREE_BITS.lock().unwrap_or_else(|e| e.into_inner()).0.push(bit);
        }
    }
}

/// Pages (page numbers) whose bytes a tracker keeps when they are first
/// written after it took them, and those bytes: while shaders may be
/// writing its copy of them, what it held before (`crate::shadow`'s
/// copy-back of marked pages), with what the copy took of the page since
/// laid over them ([`Tracked::refreshed`]).
#[derive(Default)]
struct Saved {
    /// The armed pages as runs `[first, last)`, sorted, apart and not
    /// touching (runs armed that meet merge): a root view declares its
    /// copy from its address to the end, the world's 174 MB heap many
    /// times a submission, and a set entry per page cost 30 microseconds a
    /// declaration; descriptor tables arm many sub-ranges end to end, and
    /// the write handler checks a page under the guards' lock.
    armed: Vec<(usize, usize)>,
    /// Pages of those runs that are not armed any more.
    unarmed: BTreeSet<usize>,
    bytes: std::collections::HashMap<usize, Box<[u8]>>,
}

impl Saved {
    /// Whether page `page` is armed.
    fn armed(&self, page: usize) -> bool {
        // The first run ending after `page` is the only one that can hold it.
        let i = self.armed.partition_point(|&(_, last)| last <= page);
        self.armed.get(i).is_some_and(|&(first, _)| first <= page) && !self.unarmed.contains(&page)
    }

    /// Arms pages `[first, last)`.
    fn arm(&mut self, first: usize, last: usize) {
        if first >= last {
            return;
        }
        if !self.unarmed.is_empty() {
            let pages: Vec<usize> = self.unarmed.range(first..last).copied().collect();
            for page in pages {
                self.unarmed.remove(&page);
            }
        }
        // Runs `[lo, hi)` overlap or touch the new one: it replaces them.
        let lo = self.armed.partition_point(|&(_, b)| b < first);
        let hi = self.armed.partition_point(|&(a, _)| a <= last);
        if lo + 1 == hi && self.armed[lo].0 <= first && last <= self.armed[lo].1 {
            return;
        }
        let run = if lo < hi {
            (self.armed[lo].0.min(first), self.armed[hi - 1].1.max(last))
        } else {
            (first, last)
        };
        self.armed.splice(lo..hi, [run]);
    }
}

impl Changed {
    /// `pages` were taken: anything saved of them is stale.
    fn taken_many(&self, pages: &[usize]) {
        if pages.is_empty() {
            return;
        }
        let mut saved = self.1.lock().unwrap_or_else(|e| e.into_inner());
        if !saved.bytes.is_empty() {
            for &page in pages {
                saved.bytes.remove(&page);
            }
        }
    }

    /// Page `page` is about to change for the first time since it was
    /// taken: its bytes now, if armed for it.
    fn save(&self, page: usize) {
        let mut saved = self.1.lock().unwrap_or_else(|e| e.into_inner());
        if !saved.armed(page) || saved.bytes.contains_key(&page) {
            return;
        }
        let mut bytes = vec![0u8; PAGE].into_boxed_slice();
        if crate::host::read_into((page * PAGE) as *const u8, &mut bytes) {
            saved.bytes.insert(page, bytes);
            #[cfg(feature = "diagnostics")]
            crate::copystat::add(crate::copystat::C::SavedPages, 1);
        }
    }
}

/// Byte runs `[start, end)` in order.
pub(crate) type Runs = Vec<(usize, usize)>;

/// Title memory a host copy or texture staging takes: bytes `[start, end)`.
pub(crate) struct Tracked {
    start: usize,
    end: usize,
    changed: Arc<Changed>,
    /// Parts of the range on pages that are not the runtime's memory (in
    /// order): not tracked, copied whole at each use ([`Tracked::untracked`]).
    untracked: Vec<(usize, usize)>,
}

/// State of a page currently being transitioned to `PAGE_READONLY` outside
/// [`GUARDS`].
#[derive(Clone, Copy)]
struct GuardOp {
    /// Identifies the owning `protect_with_aliases` batch.
    op_id: u64,
    /// A concurrent write fault, kernel write, remap, or tracker drop raced
    /// with this transition: do not commit the page as taken, and reopen it to
    /// `PAGE_READWRITE` if `VirtualProtect(PAGE_READONLY)` succeeded.
    reopen: bool,
    /// A concurrent alias write or `retake` raced with this transition: keep
    /// the page read-only if `VirtualProtect(PAGE_READONLY)` succeeded, but do
    /// not clear the page from the caller's dirty set.
    skip_commit: bool,
}

struct Guards {
    /// Blocks (pages) we made read-only.
    protected: BTreeSet<usize>,
    /// Blocks currently being transitioned to `PAGE_READONLY` outside
    /// [`GUARDS`]. Removed only by the owning batch once it re-acquires
    /// [`GUARDS`]; any cancelled page is moved atomically into `opening`
    /// before releasing [`GUARDS`] to reopen it.
    guarding: BTreeMap<usize, GuardOp>,
    next_guard_op: u64,
    /// Page runs `[first, last)` currently being transitioned to
    /// `PAGE_READWRITE` outside [`GUARDS`].
    opening: Vec<(usize, usize)>,
    /// The trackers holding each page.
    holders: BTreeMap<usize, Vec<Arc<Changed>>>,
    /// Pages `[first, last)` of writes the guards do not see that are under
    /// way ([`Writing`]): not guarded until they end, or the write fails.
    busy: Vec<(usize, usize)>,
    /// Pages showing the same physical pages elsewhere, both ways, as the
    /// runtime told when it mapped them: a write at one marks the others,
    /// with no question to the runtime (whose lock its page mappings hold
    /// for long).
    alias_of: BTreeMap<usize, Vec<usize>>,
    /// Tracked pages written this frame and the frame before (diagnostics).
    #[cfg(feature = "diagnostics")]
    written: BTreeSet<usize>,
    #[cfg(feature = "diagnostics")]
    written_before: BTreeSet<usize>,
}

static GUARDS: Mutex<Guards> = Mutex::new(Guards {
    protected: BTreeSet::new(),
    guarding: BTreeMap::new(),
    next_guard_op: 1,
    opening: Vec::new(),
    holders: BTreeMap::new(),
    busy: Vec::new(),
    alias_of: BTreeMap::new(),
    #[cfg(feature = "diagnostics")]
    written: BTreeSet::new(),
    #[cfg(feature = "diagnostics")]
    written_before: BTreeSet::new(),
});

/// Wakes threads waiting for an unlocked `VirtualProtect(..., PAGE_READONLY)`
/// transition (and any cancellation reopen) to finish on a page.
static GUARDING_CV: Condvar = Condvar::new();
/// Wakes threads waiting for an unlocked `VirtualProtect(..., PAGE_READWRITE)`
/// transition to finish on a page.
static OPENING_CV: Condvar = Condvar::new();
/// Serializes unlocked `VirtualProtect(..., PAGE_READONLY)` / `VirtualQuery`
/// batches across parallel refresh workers so they do not stampede Wine's
/// `virtual_mutex` while write-fault handlers remain completely independent of
/// it. Never acquired while holding [`GUARDS`], and [`GUARDS`] is never
/// acquired while holding this lock.
static PROTECT_SYSCALL: Mutex<()> = Mutex::new(());

thread_local! {
    /// Whether this thread holds the guard lock. A guarded block it faults
    /// on then is one the lock's own work touched (#50: an allocation made
    /// under the lock landed in a guarded block of the host heap, which
    /// title XMem blocks share), and taking the lock again would hang it.
    static HOLDING: std::cell::Cell<bool> = const { std::cell::Cell::new(false) };
}

/// Blocks opened by faults taken under the guard lock, for its next holder
/// to mark changed (0: free slot). Fixed: the handler must not allocate.
static OPENED_UNDER_LOCK: [std::sync::atomic::AtomicUsize; 64] =
    [const { std::sync::atomic::AtomicUsize::new(0) }; 64];

/// Whether any block waits in them: the next holder of the lock scans
/// [`OPENED_UNDER_LOCK`] only when one was opened, not on every take. Only
/// touched while the lock is held, so a clear scan leaves no block behind.
static OPENED: std::sync::atomic::AtomicBool = std::sync::atomic::AtomicBool::new(false);

/// Records `page` in [`OPENED_UNDER_LOCK`] for the current or next holder of
/// [`GUARDS`] to mark changed. Lock-free and allocation-free.
// shortcut: with all 64 slots taken the block stays open without being marked
// (counted); a bigger table if that shows up.
fn note_opened_under_lock(page: usize) {
    use std::sync::atomic::Ordering::{AcqRel, Relaxed, Release};
    if OPENED_UNDER_LOCK.iter().any(|s| s.compare_exchange(0, page, AcqRel, Relaxed).is_ok()) {
        OPENED.store(true, Release);
    } else {
        static LOST: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);
        LOST.fetch_add(1, Relaxed);
    }
}

/// A fault on the block `page` by the thread holding the guard lock: write
/// access back now, marked changed before the lock is released.
// shortcut: assumes a read-only block this thread writes while holding the
// lock is one of ours.
fn open_under_lock(page: usize) {
    use windows_sys::Win32::System::Memory::{PAGE_READWRITE, VirtualProtect};
    let mut old = 0;
    // SAFETY: a block of this process; fails harmlessly if it is gone.
    unsafe { VirtualProtect((page * PAGE) as *const _, PAGE, PAGE_READWRITE, &mut old) };
    note_opened_under_lock(page);
}

/// The guard lock, held while this lives. With diagnostics, timed for the
/// line that took it ([`held_times`]).
struct Held {
    guard: Option<std::sync::MutexGuard<'static, Guards>>,
    #[cfg(feature = "diagnostics")]
    since: std::time::Instant,
    #[cfg(feature = "diagnostics")]
    line: u32,
}

#[track_caller]
fn guards() -> Held {
    let guard = GUARDS.lock().unwrap_or_else(|e| e.into_inner());
    HOLDING.with(|h| h.set(true));
    let mut held = Held {
        guard: Some(guard),
        #[cfg(feature = "diagnostics")]
        since: std::time::Instant::now(),
        #[cfg(feature = "diagnostics")]
        line: std::panic::Location::caller().line(),
    };
    held.drain_opened_under_lock();
    held
}

impl Held {
    fn record_hold_segment(&mut self) {
        #[cfg(feature = "diagnostics")]
        {
            use std::sync::atomic::Ordering::Relaxed;
            let us = self.since.elapsed().as_micros() as u64;
            let line = u64::from(self.line);
            if let Some(slot) = HELD
                .iter()
                .find(|s| s[0].load(Relaxed) == line || s[0].compare_exchange(0, line, Relaxed, Relaxed).is_ok())
            {
                slot[1].fetch_add(us, Relaxed);
                slot[2].fetch_max(us, Relaxed);
            }
        }
    }

    /// Reconciles any pages opened by re-entrant faults while [`GUARDS`] was
    /// held, without calling `VirtualProtect` or releasing [`GUARDS`].
    fn drain_opened_under_lock(&mut self) {
        use std::sync::atomic::Ordering::{AcqRel, Acquire};
        while OPENED.load(Acquire) && OPENED.swap(false, AcqRel) {
            for slot in &OPENED_UNDER_LOCK {
                let page = slot.swap(0, AcqRel);
                if page != 0 {
                    self.protected.remove(&page);
                    if let Some(op) = self.guarding.get_mut(&page) {
                        op.reopen = true;
                        op.skip_commit = true;
                    }
                    let others: Vec<usize> = self
                        .alias_of
                        .get(&page)
                        .into_iter()
                        .flatten()
                        .copied()
                        .collect();
                    cancel_guarding_aliases(self, &others);
                    mark_changed(self, page, page + 1, false);
                    for alias in others {
                        mark_changed(self, alias, alias + 1, false);
                    }
                }
            }
        }
    }

    /// Releases [`GUARDS`] for the duration of `f` (used around `VirtualProtect`,
    /// `VirtualQuery`, and `GetWriteWatch` syscalls) and re-acquires it
    /// afterwards.
    fn unlock_for<R>(&mut self, f: impl FnOnce() -> R) -> R {
        self.drain_opened_under_lock();
        self.record_hold_segment();
        drop(self.guard.take());
        HOLDING.with(|h| h.set(false));
        let out = f();
        let guard = GUARDS.lock().unwrap_or_else(|e| e.into_inner());
        HOLDING.with(|h| h.set(true));
        self.guard = Some(guard);
        #[cfg(feature = "diagnostics")]
        {
            self.since = std::time::Instant::now();
        }
        self.drain_opened_under_lock();
        out
    }

    /// Waits on `cv`, clearing [`HOLDING`] and hold-time accounting while
    /// [`GUARDS`] is released.
    fn wait(&mut self, cv: &'static Condvar) {
        self.drain_opened_under_lock();
        self.record_hold_segment();
        let guard = self.guard.take().expect("held until dropped");
        HOLDING.with(|h| h.set(false));
        let guard = cv.wait(guard).unwrap_or_else(|e| e.into_inner());
        HOLDING.with(|h| h.set(true));
        self.guard = Some(guard);
        #[cfg(feature = "diagnostics")]
        {
            self.since = std::time::Instant::now();
        }
        self.drain_opened_under_lock();
    }
}

impl std::ops::Deref for Held {
    type Target = Guards;
    fn deref(&self) -> &Guards {
        self.guard.as_ref().expect("held until dropped")
    }
}

impl std::ops::DerefMut for Held {
    fn deref_mut(&mut self) -> &mut Guards {
        self.guard.as_mut().expect("held until dropped")
    }
}

/// Per line taking the guard lock: the line, microseconds held in all, the
/// longest hold (no allocation: the fault handler takes the lock).
#[cfg(feature = "diagnostics")]
static HELD: [[std::sync::atomic::AtomicU64; 3]; 16] =
    [const { [const { std::sync::atomic::AtomicU64::new(0) }; 3] }; 16];

impl Drop for Held {
    fn drop(&mut self) {
        if self.guard.is_some() {
            self.drain_opened_under_lock();
            self.record_hold_segment();
            drop(self.guard.take());
            HOLDING.with(|h| h.set(false));
        }
    }
}

/// Since the last call, per line of this file taking the guard lock: (line,
/// milliseconds held, longest hold in milliseconds), most held first.
#[cfg(feature = "diagnostics")]
pub(crate) fn held_times() -> Vec<(u64, f64, f64)> {
    use std::sync::atomic::Ordering::Relaxed;
    let mut out: Vec<(u64, f64, f64)> = HELD
        .iter()
        .filter(|s| s[0].load(Relaxed) != 0)
        .map(|s| (s[0].load(Relaxed), s[1].swap(0, Relaxed) as f64 / 1e3, s[2].swap(0, Relaxed) as f64 / 1e3))
        .collect();
    out.sort_by(|a, b| b.1.total_cmp(&a.1));
    out
}

fn is_busy(g: &Guards, page: usize) -> bool {
    g.busy.iter().any(|&(a, b)| (a..b).contains(&page))
}

fn is_opening(g: &Guards, page: usize) -> bool {
    g.opening.iter().any(|&(a, b)| (a..b).contains(&page))
}

fn overlaps_opening(g: &Guards, first: usize, last: usize) -> bool {
    first < last && g.opening.iter().any(|&(a, b)| a < last && first < b)
}

/// Prevents any in-flight `protect_with_aliases` on `others` (or their
/// aliases) from clearing tracker dirty state when it re-acquires [`GUARDS`],
/// without reopening their read-only guards.
fn cancel_guarding_aliases(g: &mut Guards, others: &[usize]) {
    if g.guarding.is_empty() || others.is_empty() {
        return;
    }
    for &alias in others {
        if let Some(op) = g.guarding.get_mut(&alias) {
            op.skip_commit = true;
        }
        if let Some(hop) = g.alias_of.get(&alias) {
            for &p in hop {
                if let Some(op) = g.guarding.get_mut(&p) {
                    op.skip_commit = true;
                }
            }
        }
    }
}

/// (page, other page) for the pages (page numbers) showing the same memory
/// as `pages` elsewhere.
fn aliases(g: &Guards, pages: &[usize]) -> Vec<(usize, usize)> {
    pages
        .iter()
        .flat_map(|&p| g.alias_of.get(&p).into_iter().flatten().map(move |&a| (p, a)))
        .collect()
}

/// Tracks `[start, start + len)` if it is the runtime's GPU memory (every
/// page changed at first); `None` for other memory, and with `views` false
/// for anything but its write-watched pools (physical-page views are
/// written through other views showing the same pages: the title fills
/// streamed textures and video planes through a pool view and maps the
/// pages into its tiled heap, where a guard sees none of it).
///
/// Only the pages of the range that are such memory are tracked; the rest is
/// never guarded and is copied whole at each use, as memory with no tracker.
pub(crate) fn track(start: usize, len: usize, views: bool) -> Option<Tracked> {
    track_as(start, len, views, false)
}

/// [`track`] for a host buffer copy: what shaders write lands in it, so a
/// [`retake`] of its memory marks it.
pub(crate) fn track_copy(start: usize, len: usize, views: bool) -> Option<Tracked> {
    track_as(start, len, views, true)
}

fn track_as(start: usize, len: usize, views: bool, copy: bool) -> Option<Tracked> {
    if len == 0 || !fault_tracking() {
        return None;
    }
    let end = start + len;
    let pages = runtime_pages(start, end, views);
    if pages.is_empty() {
        return None;
    }
    install();
    // The rest of the range, as byte runs.
    let mut untracked: Vec<(usize, usize)> = Vec::new();
    let mut at = start;
    for &page in pages.iter().chain([&usize::MAX]) {
        let (a, b) = ((page.saturating_mul(PAGE)).max(start).min(end), (page.saturating_add(1).saturating_mul(PAGE)).min(end));
        if at < a {
            untracked.push((at, a));
        }
        at = at.max(b);
    }
    // A tracker with untracked parts is never clean: no bit.
    let bit = if untracked.is_empty() { dirty_bit() } else { None };
    set_dirty(bit, true);
    let changed = Arc::new(Changed(
        Mutex::new(pages.iter().copied().collect()),
        Mutex::default(),
        bit,
        copy,
        std::sync::atomic::AtomicBool::new(false),
        Mutex::default(),
    ));
    let mut g = guards();
    for &page in &pages {
        g.holders.entry(page).or_default().push(changed.clone());
    }
    drop(g);
    Some(Tracked {
        start,
        end,
        changed,
        untracked,
    })
}

/// Whether every byte of `[at, at + len)` lies in a single write-watched pool
/// (one lookup).
fn write_watched_range(at: usize, len: usize) -> bool {
    use windows_sys::Win32::System::Memory::GetWriteWatch;
    if len == 0 {
        return false;
    }
    let (mut found, mut count, mut granularity) = (0usize, 1usize, 0u32);
    // SAFETY: room for one answer; no reset, so nothing changes.
    unsafe {
        GetWriteWatch(
            0,
            at as *const _,
            len,
            (&raw mut found).cast(),
            &mut count,
            &mut granularity,
        ) == 0
    }
}

/// The pages of `[start, end)` that are the runtime's GPU memory: a
/// write-watched pool, or (with `views`) anything but plain committed memory
/// and images (views, placeholders, nothing mapped yet). Decided for every
/// region the range covers, and a page touching any other memory is left
/// out: a range can run past its allocation (a 4 MiB window of memory
/// outside buffers, `crate::shadow`), and when the first page alone decided,
/// a window starting in free memory had the process heap's pages after it
/// guarded; the heap's own code then faulted while holding its lock, and the
/// guard handler waited for the guard lock, whose holder waited for the heap
/// lock (the crate's tests hung).
///
/// shortcut: memory free or reserved when the range is tracked counts as the
/// runtime's; another allocator committing it before the first take would
/// have it guarded.
fn runtime_pages(start: usize, end: usize, views: bool) -> Vec<usize> {
    use windows_sys::Win32::System::Memory::{
        MEM_COMMIT, MEM_IMAGE, MEM_PRIVATE, MEMORY_BASIC_INFORMATION, VirtualQuery,
    };
    let (first, last) = (start / PAGE, end.div_ceil(PAGE));
    // One pool holds it all (the common case; asking `VirtualQuery` costs
    // milliseconds in the runtime's big pools).
    if write_watched_range(start & !(4096 - 1), end - (start & !(4096 - 1))) {
        return (first..last).collect();
    }
    let (mut runtime, mut other) = (std::collections::BTreeSet::new(), std::collections::BTreeSet::new());
    let mut at = start & !(4096 - 1);
    while at < end {
        // SAFETY: plain data filled in by the query.
        let mut info: MEMORY_BASIC_INFORMATION = unsafe { std::mem::zeroed() };
        // SAFETY: a valid out pointer of the right size.
        if unsafe { VirtualQuery(at as *const _, &mut info, size_of_val(&info)) } == 0 {
            break;
        }
        let next = (info.BaseAddress as usize + info.RegionSize).max(at + 1).min(end);
        let ours = write_watched_range(at, next - at)
            || views && info.Type != MEM_IMAGE && !(info.State == MEM_COMMIT && info.Type == MEM_PRIVATE);
        (if ours { &mut runtime } else { &mut other }).extend(at / PAGE..next.div_ceil(PAGE));
        at = next;
    }
    runtime.difference(&other).copied().collect()
}

impl Drop for Tracked {
    /// Pages no tracker holds any more get write access back, so that the
    /// guard set never names memory freed and allocated again; unless another
    /// mapping of the page is held: the guard here is that tracker's too (it
    /// was set when that tracker took its page), and with it opened a write
    /// here went unseen there.
    fn drop(&mut self) {
        let mut g = guards();
        let mut free = Vec::new();
        for page in self.start / PAGE..self.end.div_ceil(PAGE) {
            if let Some(holders) = g.holders.get_mut(&page) {
                holders.retain(|c| !Arc::ptr_eq(c, &self.changed));
                if holders.is_empty() {
                    g.holders.remove(&page);
                    free.push(page);
                }
            }
        }
        let mut to_open = Vec::new();
        let mut wait_guarding = Vec::new();
        let mut wait_opening = Vec::new();
        for page in free {
            let watched_elsewhere = g
                .alias_of
                .get(&page)
                .is_some_and(|o| o.iter().any(|p| g.holders.contains_key(p)));
            if !watched_elsewhere {
                if g.protected.remove(&page) {
                    to_open.push(page);
                }
                if let Some(op) = g.guarding.get_mut(&page) {
                    op.reopen = true;
                    op.skip_commit = true;
                    wait_guarding.push(page);
                } else if is_opening(&g, page) {
                    wait_opening.push(page);
                }
            }
        }
        open_sorted_pages(&mut g, &to_open);
        while wait_guarding.iter().any(|p| g.guarding.contains_key(p)) {
            g.wait(&GUARDING_CV);
        }
        while wait_guarding.iter().chain(&wait_opening).any(|&p| is_opening(&g, p)) {
            g.wait(&OPENING_CV);
        }
    }
}

/// Coalesces sorted `pages` (already removed from `g.protected`/`g.guarding`)
/// into contiguous runs and calls `VirtualProtect(..., PAGE_READWRITE)`
/// outside [`GUARDS`], holding [`Guards::opening`] across the syscalls so
/// concurrent takes and faults serialize cleanly on only those pages.
fn open_sorted_pages(g: &mut Held, pages: &[usize]) {
    if pages.is_empty() {
        return;
    }
    let mut runs: Vec<(usize, usize)> = Vec::new();
    let mut i = 0;
    while i < pages.len() {
        let mut j = i + 1;
        while j < pages.len() && pages[j] == pages[j - 1] + 1 {
            j += 1;
        }
        runs.push((pages[i], pages[j - 1] + 1));
        i = j;
    }
    g.opening.extend(runs.iter().copied());
    let opened = g.unlock_for(|| {
        use windows_sys::Win32::System::Memory::{PAGE_READWRITE, VirtualProtect};
        debug_assert!(
            !HOLDING.with(std::cell::Cell::get),
            "VirtualProtect(PAGE_READWRITE) must not run while holding GUARDS"
        );
        let mut count = 0u64;
        for &(first, last) in &runs {
            let mut old = 0;
            if last - first == 1 {
                // SAFETY: a block we made read-only (fails harmlessly if gone).
                unsafe { VirtualProtect((first * PAGE) as *const _, PAGE, PAGE_READWRITE, &mut old) };
                count += 1;
                continue;
            }
            // SAFETY: blocks we made read-only.
            if unsafe {
                VirtualProtect((first * PAGE) as *const _, (last - first) * PAGE, PAGE_READWRITE, &mut old)
            } != 0
            {
                count += (last - first) as u64;
            } else {
                for page in first..last {
                    // SAFETY: as above.
                    unsafe { VirtualProtect((page * PAGE) as *const _, PAGE, PAGE_READWRITE, &mut old) };
                    count += 1;
                }
            }
        }
        count
    });
    for run in &runs {
        if let Some(pos) = g.opening.iter().position(|r| r == run) {
            g.opening.swap_remove(pos);
        }
    }
    OPENING_CV.notify_all();
    #[cfg(feature = "diagnostics")]
    crate::copystat::add(crate::copystat::C::OpenPages, opened);
    #[cfg(not(feature = "diagnostics"))]
    let _ = opened;
}

impl Tracked {
    /// Takes the changed pages of `[from, to)`: guards them (writes from
    /// now on mark them again) and returns the byte ranges to copy, cut to
    /// `[from, to)`, in order. A page whose part in the tracked memory lies
    /// only partly in `[from, to)` is copied in part and stays changed.
    pub(crate) fn take(&self, from: usize, to: usize) -> Vec<(usize, usize)> {
        let (copy, zero) = self.take_split(from, to);
        let mut runs: Vec<(usize, usize)> = copy.into_iter().chain(zero).collect();
        runs.sort_unstable();
        let mut merged: Vec<(usize, usize)> = Vec::with_capacity(runs.len());
        for (a, b) in runs {
            match merged.last_mut() {
                Some((_, end)) if *end == a => *end = b,
                _ => merged.push((a, b)),
            }
        }
        merged
    }

    /// Requeues the tracked pages returned by `take` when their copy could
    /// not be staged. Writes since the take stay in the set too.
    pub(crate) fn retry(&self, taken: &[(usize, usize)]) {
        let mut changed = self.changed.0.lock().unwrap_or_else(|e| e.into_inner());
        for &(from, to) in taken {
            changed.extend(from / PAGE..to.div_ceil(PAGE));
        }
        set_dirty(self.changed.2, !changed.is_empty());
    }

    /// Drains any retaken or remapped pages overlapping `[from, to)` so a host
    /// buffer copy discards cached baseline piece hashes before refreshing.
    pub(crate) fn take_retaken(&self, from: usize, to: usize) -> Runs {
        if !self.changed.4.load(Ordering::Acquire) {
            return Vec::new();
        }
        let (from, to) = (from.max(self.start), to.min(self.end));
        if from >= to {
            return Vec::new();
        }
        let mut retaken = self.changed.5.lock().unwrap_or_else(|e| e.into_inner());
        let pages: Vec<usize> = retaken.range(from / PAGE..to.div_ceil(PAGE)).copied().collect();
        for &p in &pages {
            retaken.remove(&p);
        }
        if retaken.is_empty() {
            self.changed.4.store(false, Ordering::Release);
        }
        drop(retaken);
        let mut runs: Runs = Vec::new();
        for p in pages {
            let (a, b) = ((p * PAGE).max(from), ((p + 1) * PAGE).min(to));
            match runs.last_mut() {
                Some((_, end)) if *end == a => *end = b,
                _ => runs.push((a, b)),
            }
        }
        runs
    }

    /// [`Tracked::take_split`] of several ranges at once: one split pair per
    /// range, in order, with the process's guard lock taken once for all of
    /// them instead of once a range (a copy of many ranges took it as many
    /// times). A page two ranges share is taken by the first of them, as the
    /// first of as many calls would have taken it, leaving the others to see
    /// it changed no more.
    pub(crate) fn take_split_many(&self, ranges: &[(usize, usize)]) -> Vec<(Runs, Runs)> {
        let clipped: Vec<(usize, usize)> = ranges
            .iter()
            .map(|&(from, to)| (from.max(self.start), to.min(self.end)))
            .collect();
        let pages: Vec<Vec<usize>> = {
            #[cfg(feature = "diagnostics")]
            let waited = Timed(std::time::Instant::now(), crate::copystat::C::GuardWaitUs);
            let changed = self.changed.0.lock().unwrap_or_else(|e| e.into_inner());
            #[cfg(feature = "diagnostics")]
            drop(waited);
            clipped
                .iter()
                .map(|&(from, to)| {
                    if from >= to {
                        return Vec::new();
                    }
                    changed.range(from / PAGE..to.div_ceil(PAGE)).copied().collect()
                })
                .collect()
        };
        if pages.iter().all(Vec::is_empty) {
            return vec![(Vec::new(), Vec::new()); ranges.len()];
        }
        let whole: Vec<usize> = pages
            .iter()
            .flatten()
            .copied()
            .filter(|&p| {
                clipped.iter().any(|&(from, to)| {
                    (p * PAGE).max(self.start) >= from && ((p + 1) * PAGE).min(self.end) <= to
                })
            })
            .collect();
        let push = |out: &mut Vec<(Runs, Runs)>, pages: &[usize], from: usize, to: usize, unmapped: &[usize]| {
            let (mut copy, mut zero): (Runs, Runs) = (Vec::new(), Vec::new());
            for &p in pages {
                let (a, b) = ((p * PAGE).max(from), ((p + 1) * PAGE).min(to));
                let runs = if unmapped.binary_search(&p).is_ok() { &mut zero } else { &mut copy };
                match runs.last_mut() {
                    Some((_, end)) if *end == a => *end = b,
                    _ => runs.push((a, b)),
                }
            }
            out.push((copy, zero));
        };
        let mut unmapped = Vec::new();
        let mut taken: BTreeSet<usize> = BTreeSet::new();
        let mut out: Vec<(Runs, Runs)> = Vec::with_capacity(ranges.len());
        if whole.is_empty() {
            for (i, &(from, to)) in clipped.iter().enumerate() {
                push(&mut out, &pages[i], from, to, &unmapped);
            }
            return out;
        }
        // Every address showing the same pages is guarded too. A write in
        // between marks them changed again; the copy, made after, has it.
        #[cfg(feature = "diagnostics")]
        let waited = Timed(std::time::Instant::now(), crate::copystat::C::GuardWaitUs);
        let mut g = guards();
        #[cfg(feature = "diagnostics")]
        drop(waited);
        #[cfg(feature = "diagnostics")]
        let asked = Timed(std::time::Instant::now(), crate::copystat::C::AliasUs);
        let others = aliases(&g, &whole);
        #[cfg(feature = "diagnostics")]
        drop(asked);
        for (i, &(from, to)) in clipped.iter().enumerate() {
            let pages: Vec<usize> = pages[i]
                .iter()
                .copied()
                .filter(|p| !taken.contains(p))
                .collect();
            // Partial pages are copied but stay dirty for a later request.
            let whole: Vec<usize> = pages
                .iter()
                .copied()
                .filter(|&p| {
                    (p * PAGE).max(self.start) >= from && ((p + 1) * PAGE).min(self.end) <= to
                })
                .collect();
            protect_with_aliases(&mut g, &whole, &others, true, Some(&mut unmapped), |pages| {
                let mut changed = self.changed.0.lock().unwrap_or_else(|e| e.into_inner());
                for &page in pages {
                    changed.remove(&page);
                    taken.insert(page);
                }
                set_dirty(self.changed.2, !changed.is_empty());
                drop(changed);
                self.changed.taken_many(pages);
            });
            unmapped.sort_unstable();
            push(&mut out, &pages, from, to, &unmapped);
        }
        drop(g);
        out
    }

    /// [`Tracked::take`], the runs on pages with nothing mapped apart: they
    /// read as zeros, with no copy to try (each such page cost a dozen
    /// failing kernel reads, 7-30 ms a frame in run perf4).
    pub(crate) fn take_split(&self, from: usize, to: usize) -> (Runs, Runs) {
        self.take_split_many(&[(from, to)]).pop().unwrap()
    }

    /// Whether page `page` (a page number) changed since it was taken.
    pub(crate) fn changed(&self, page: usize) -> bool {
        self.changed
            .0
            .lock()
            .unwrap_or_else(|e| e.into_inner())
            .contains(&page)
    }

    /// The caller made pages (page numbers) of its copy equal to title
    /// memory itself (what shaders wrote went back): taken and guarded.
    /// Untracked pages (never guarded) are skipped.
    pub(crate) fn taken(&self, pages: &[usize]) {
        let tracked_pages: Vec<usize> = pages
            .iter()
            .copied()
            .filter(|&p| !self.is_untracked(p))
            .collect();
        if tracked_pages.is_empty() {
            return;
        }
        let mut g = guards();
        let others = aliases(&g, &tracked_pages);
        protect_with_aliases(&mut g, &tracked_pages, &others, true, None, |taken| {
            {
                let mut changed = self.changed.0.lock().unwrap_or_else(|e| e.into_inner());
                for &page in taken {
                    changed.remove(&page);
                }
                set_dirty(self.changed.2, !changed.is_empty());
            }
            self.changed.taken_many(taken);
        });
    }

    /// Whether any byte of page `page` (a page number) lies in the tracker's
    /// untracked parts (never guarded, copied whole at each use).
    pub(crate) fn is_untracked(&self, page: usize) -> bool {
        let (a, b) = (
            page.saturating_mul(PAGE),
            page.saturating_add(1).saturating_mul(PAGE),
        );
        self.untracked.iter().any(|&(u, v)| u < b && a < v)
    }

    /// Whether no page changed since it was taken: then [`Tracked::take`]
    /// finds nothing, and asking costs no lock. A write racing this answer
    /// is found by the next asking, as one racing a take.
    pub(crate) fn clean(&self) -> bool {
        clean_bit(self.changed.2)
    }

    /// The tracker's bit for [`clean_bit`], for callers that ask about
    /// many trackers at once without touching each (none: never clean).
    pub(crate) fn bit(&self) -> Option<u32> {
        self.changed.2
    }

    /// The parts of `[from, to)` that are not tracked (see
    /// [`Tracked::untracked`] field): to be copied whole at each use.
    pub(crate) fn untracked(&self, from: usize, to: usize) -> Vec<(usize, usize)> {
        self.untracked
            .iter()
            .map(|&(a, b)| (a.max(from), b.min(to)))
            .filter(|&(a, b)| a < b)
            .collect()
    }

    /// Shaders may write the copy's bytes `[from, to)` until [`Tracked::disarm`]:
    /// pages there written next after they were taken have their bytes kept
    /// first ([`Tracked::saved`]).
    pub(crate) fn arm(&self, from: usize, to: usize) {
        let (from, to) = (from.max(self.start), to.min(self.end));
        if from < to {
            self.changed.1.lock().unwrap_or_else(|e| e.into_inner()).arm(from / PAGE, to.div_ceil(PAGE));
        }
    }

    /// Page `page` (a page number) needs no saving any more: its copy-back
    /// is under way.
    pub(crate) fn unarm(&self, page: usize) {
        let mut saved = self.changed.1.lock().unwrap_or_else(|e| e.into_inner());
        if saved.armed(page) {
            saved.unarmed.insert(page);
        }
    }

    /// No shader writes the copy any more: nothing is armed or kept.
    pub(crate) fn disarm(&self) {
        let mut saved = self.changed.1.lock().unwrap_or_else(|e| e.into_inner());
        saved.armed.clear();
        saved.unarmed.clear();
        saved.bytes.clear();
    }

    /// The bytes page `page` (a page number) held when it was first written
    /// since it was taken, if it was armed then (and so changed), with what
    /// the copy took of it since: the copy's bytes before the submission.
    pub(crate) fn saved(&self, page: usize) -> Option<Box<[u8]>> {
        self.changed.1.lock().unwrap_or_else(|e| e.into_inner()).bytes.remove(&page)
    }

    /// The copy took `bytes` at title address `from`: any saved bytes of pages
    /// it covers take them too, so marked copy-back neither takes a partial
    /// refresh for the GPU's nor drops a shader write back to the pre-fault byte.
    pub(crate) fn refreshed(&self, from: usize, bytes: &[u8]) {
        if bytes.is_empty() {
            return;
        }
        let mut saved = self.changed.1.lock().unwrap_or_else(|e| e.into_inner());
        if saved.bytes.is_empty() {
            return;
        }
        let to = from + bytes.len();
        for page in from / PAGE..to.div_ceil(PAGE) {
            if let Some(before) = saved.bytes.get_mut(&page) {
                let (a, b) = (from.max(page * PAGE), to.min((page + 1) * PAGE));
                before[a - page * PAGE..b - page * PAGE]
                    .copy_from_slice(&bytes[a - from..b - from]);
            }
        }
    }
}

/// Makes `pages` read-only where not already (in runs, with [`GUARDS`]
/// released across syscalls), and returns those now guarded or with nothing
/// mapped (taken either way); a page that is mapped but cannot be guarded or
/// was concurrently opened/cancelled is left out (it stays changed).
fn protect(g: &mut Held, pages: &[usize]) -> Vec<usize> {
    protect_with_aliases(g, pages, &[], false, None, |_| {})
}

/// Guards `primary` (and, when `check_aliases` is true, every alias mapping of
/// each primary page) in a single unlocked `VirtualProtect` / `VirtualQuery` /
/// `GetWriteWatch` pass, committing tracker dirty-set clears atomically under
/// [`GUARDS`] via `on_commit` before releasing [`GUARDS`] to reopen any
/// concurrently cancelled pages.
fn protect_with_aliases(
    g: &mut Held,
    primary: &[usize],
    alias_pairs: &[(usize, usize)],
    check_aliases: bool,
    mut unmapped_pages: Option<&mut Vec<usize>>,
    mut on_commit: impl FnMut(&[usize]),
) -> Vec<usize> {
    use windows_sys::Win32::System::Memory::{
        MEM_COMMIT, MEMORY_BASIC_INFORMATION, PAGE_READONLY, VirtualProtect, VirtualQuery,
    };
    if primary.is_empty() {
        return Vec::new();
    }

    let mut prim: Vec<usize> = primary.to_vec();
    prim.sort_unstable();
    prim.dedup();

    // Partition into pages already fully guarded (which we commit immediately
    // under `GUARDS` before unlocking, so a concurrent fault while unlocked is
    // never lost) and pages that need at least one unlocked `VirtualProtect`.
    let mut already_ready = Vec::new();
    let mut pending_prim = Vec::new();
    let mut candidates = Vec::new();

    for &p in &prim {
        if is_busy(g, p) || is_opening(g, p) || g.guarding.contains_key(&p) {
            continue;
        }
        let mut alias_blocked = false;
        let mut all_aliases_protected = true;
        if check_aliases && let Some(links) = g.alias_of.get(&p) {
            for &a in links {
                if is_busy(g, a) || is_opening(g, a) || g.guarding.contains_key(&a) {
                    alias_blocked = true;
                    break;
                }
                if !g.protected.contains(&a) {
                    all_aliases_protected = false;
                }
            }
        }
        if alias_blocked {
            continue;
        }
        if g.protected.contains(&p) && all_aliases_protected {
            already_ready.push(p);
            continue;
        }
        pending_prim.push(p);
        if !g.protected.contains(&p) {
            candidates.push(p);
        }
        if check_aliases && let Some(links) = g.alias_of.get(&p) {
            for &a in links {
                if !g.protected.contains(&a) {
                    candidates.push(a);
                }
            }
        }
    }
    if check_aliases {
        for &(p, a) in alias_pairs {
            if pending_prim.binary_search(&p).is_ok()
                && !g.protected.contains(&a)
                && !is_busy(g, a)
                && !is_opening(g, a)
                && !g.guarding.contains_key(&a)
            {
                candidates.push(a);
            }
        }
    }

    if !already_ready.is_empty() {
        on_commit(&already_ready);
    }
    let mut all_taken = already_ready;

    candidates.sort_unstable();
    candidates.dedup();
    if candidates.is_empty() {
        return all_taken;
    }

    let op_id = g.next_guard_op;
    g.next_guard_op = g.next_guard_op.wrapping_add(1).max(1);
    for &page in &candidates {
        g.guarding.insert(
            page,
            GuardOp {
                op_id,
                reopen: false,
                skip_commit: false,
            },
        );
    }

    // Only the runtime's GPU memory is guarded. A tracked range can start in
    // free or reserved memory and run on into the host's own pages (#50).
    // Filter and `VirtualProtect` run completely outside `GUARDS`, preserving
    // contiguous `GetWriteWatch` batching (#510) and uncommitted-region skips
    // (#523).
    let (succeeded, unmapped_found) = g.unlock_for(|| {
        let _syscall = PROTECT_SYSCALL.lock().unwrap_or_else(|e| e.into_inner());
        #[cfg(feature = "diagnostics")]
        let _timed = Timed(std::time::Instant::now(), crate::copystat::C::ProtectUs);
        debug_assert!(
            !HOLDING.with(std::cell::Cell::get),
            "VirtualProtect(PAGE_READONLY) must not run while holding GUARDS"
        );
        let set = |from: usize, to: usize| {
            #[cfg(feature = "diagnostics")]
            {
                crate::copystat::add(crate::copystat::C::ProtectCalls, 1);
                crate::copystat::add(crate::copystat::C::ProtectPages, (to - from) as u64);
            }
            #[cfg(test)]
            tests::PROTECTS.with(|n| n.set(n.get() + 1));
            let mut old = 0;
            // SAFETY: title memory; only its protection changes.
            unsafe { VirtualProtect((from * PAGE) as *const _, (to - from) * PAGE, PAGE_READONLY, &mut old) != 0 }
        };
        let unmapped_region = |page: usize| -> Option<usize> {
            // SAFETY: plain data filled in by the query.
            let mut info: MEMORY_BASIC_INFORMATION = unsafe { std::mem::zeroed() };
            // SAFETY: a valid out pointer of the right size.
            let n = unsafe { VirtualQuery((page * PAGE) as *const _, &mut info, size_of_val(&info)) };
            if n == 0 {
                return Some(page + 1);
            }
            if info.State != MEM_COMMIT {
                let end = ((info.BaseAddress as usize).saturating_add(info.RegionSize) / PAGE)
                    .max(page + 1);
                return Some(end);
            }
            None
        };

        let empty = BTreeSet::new();
        let (filtered, ends) = filter_runtime_pages(
            &candidates,
            &empty,
            |first, count| write_watched_range(first * PAGE, count * PAGE),
            query_runtime_region,
        );

        let mut ok = BTreeSet::new();
        let mut unmapped_out = Vec::new();
        let mut run_taken = Vec::with_capacity(filtered.len());
        let mut i = 0;
        while i < filtered.len() {
            let first = i;
            while i + 1 < filtered.len()
                && filtered[i + 1] == filtered[i] + 1
                && filtered[i + 1] < ends[first]
            {
                i += 1;
            }
            let run = &filtered[first..=i];
            i += 1;
            protect_run_pages(
                run,
                &mut ok,
                &mut run_taken,
                &mut unmapped_out,
                set,
                unmapped_region,
            );
        }
        (ok, unmapped_out)
    });

    // Reconcile under `GUARDS`: any candidate marked `reopen` (or now
    // busy/opening) was concurrently faulted, written, remapped, or dropped
    // while unlocked and must be reopened to `PAGE_READWRITE` before waking
    // `GUARDING_CV`. Any candidate marked `skip_commit` stays dirty.
    let mut reopen_set = BTreeSet::new();
    let mut skip_commit_set = BTreeSet::new();
    for &page in &candidates {
        if let Some(&op) = g.guarding.get(&page)
            && op.op_id == op_id
        {
            g.guarding.remove(&page);
            if op.reopen || is_busy(g, page) || is_opening(g, page) {
                reopen_set.insert(page);
                skip_commit_set.insert(page);
            } else if op.skip_commit {
                skip_commit_set.insert(page);
            }
        } else {
            reopen_set.insert(page);
            skip_commit_set.insert(page);
        }
    }

    let mut reopen_cancelled = Vec::new();
    for &page in &succeeded {
        if reopen_set.contains(&page) {
            reopen_cancelled.push(page);
        } else {
            g.protected.insert(page);
        }
    }
    let unmapped_ok: BTreeSet<usize> = unmapped_found
        .into_iter()
        .filter(|p| !skip_commit_set.contains(p))
        .collect();

    let mut newly_taken = Vec::with_capacity(pending_prim.len());
    for &p in &pending_prim {
        if skip_commit_set.contains(&p)
            || is_busy(g, p)
            || is_opening(g, p)
            || g.guarding.contains_key(&p)
        {
            continue;
        }
        let p_unmapped = unmapped_ok.contains(&p);
        if !g.protected.contains(&p) && !p_unmapped {
            continue;
        }
        if check_aliases
            && let Some(links) = g.alias_of.get(&p)
            && links.iter().any(|a| {
                skip_commit_set.contains(a)
                    || is_busy(g, *a)
                    || is_opening(g, *a)
                    || g.guarding.contains_key(a)
                    || (!g.protected.contains(a) && !unmapped_ok.contains(a))
            })
        {
            continue;
        }
        newly_taken.push(p);
        if p_unmapped && let Some(out) = unmapped_pages.as_deref_mut() {
            out.push(p);
        }
    }

    // Commit tracker state atomically under `GUARDS` before `open_sorted_pages`
    // unlocks `GUARDS` to reopen cancelled pages: otherwise a newly guarded
    // page in `newly_taken` could fault and reopen while unlocked, only for
    // the caller to clear its dirty flag afterwards.
    if !newly_taken.is_empty() {
        on_commit(&newly_taken);
        all_taken.extend(newly_taken);
    }

    if !reopen_cancelled.is_empty() {
        open_sorted_pages(g, &reopen_cancelled);
    }
    GUARDING_CV.notify_all();
    all_taken
}

/// Guards a contiguous run of unprotected pages, falling back to per-page
/// protection on failure and skipping whole uncommitted `VirtualQuery` regions
/// without probing each unmapped page individually under `Guards`.
fn protect_run_pages(
    run: &[usize],
    protected: &mut BTreeSet<usize>,
    taken: &mut Vec<usize>,
    unmapped_pages: &mut Vec<usize>,
    mut set: impl FnMut(usize, usize) -> bool,
    mut unmapped_region: impl FnMut(usize) -> Option<usize>,
) {
    if set(run[0], run[run.len() - 1] + 1) {
        protected.extend(run);
        taken.extend(run);
        return;
    }
    let mut unmapped_until = 0usize;
    for &page in run {
        if page < unmapped_until {
            taken.push(page);
            unmapped_pages.push(page);
            continue;
        }
        if set(page, page + 1) {
            protected.insert(page);
            taken.push(page);
        } else if let Some(end) = unmapped_region(page) {
            taken.push(page);
            unmapped_pages.push(page);
            unmapped_until = end;
        }
    }
}

/// Asks `VirtualQuery` for the region containing `page` and whether it is
/// runtime memory (anything other than plain committed private memory, since
/// write-watched pools are already handled by `write_watched_range`).
fn query_runtime_region(page: usize) -> Option<(usize, usize, bool)> {
    use windows_sys::Win32::System::Memory::{
        MEM_COMMIT, MEM_PRIVATE, MEMORY_BASIC_INFORMATION, VirtualQuery,
    };
    // SAFETY: plain data filled in by the query.
    let mut info: MEMORY_BASIC_INFORMATION = unsafe { std::mem::zeroed() };
    // SAFETY: a valid out pointer of the right size.
    let n = unsafe { VirtualQuery((page * PAGE) as *const _, &mut info, size_of_val(&info)) };
    if n == 0 {
        return None;
    }
    let yes = !(info.State == MEM_COMMIT && info.Type == MEM_PRIVATE);
    let (base, size) = (info.BaseAddress as usize, info.RegionSize);
    // In our pages, rounded inward so the region never claims a page it
    // only shares.
    let (a, b) = (base.div_ceil(PAGE).min(page), ((base + size) / PAGE).max(page + 1));
    Some((a, b, yes))
}

/// Filters `pages` to those already guarded or in the runtime's GPU memory,
/// returning each kept page together with the exclusive end of its known
/// region.
///
/// Contiguous unprotected runs are probed as a whole with `watched_range`
/// (one `GetWriteWatch` call for an entire write-watched pool run instead of
/// one per page under the global `Guards` lock, #460). When a run crosses a
/// pool boundary, an exponential + binary search finds the exact prefix in
/// `O(log n)` calls without falling back to `VirtualQuery` on the pool.
fn filter_runtime_pages(
    pages: &[usize],
    protected: &BTreeSet<usize>,
    mut watched_range: impl FnMut(usize, usize) -> bool,
    mut query_region: impl FnMut(usize) -> Option<(usize, usize, bool)>,
) -> (Vec<usize>, Vec<usize>) {
    let mut out_pages = Vec::with_capacity(pages.len());
    let mut out_ends = Vec::with_capacity(pages.len());
    let mut last: Option<(usize, usize, bool)> = None;
    let mut i = 0;
    while i < pages.len() {
        if protected.contains(&pages[i]) {
            out_pages.push(pages[i]);
            out_ends.push(usize::MAX);
            i += 1;
            continue;
        }
        // Gather a maximal contiguous ascending run of unprotected pages.
        let first = i;
        let mut j = i + 1;
        while j < pages.len() && pages[j] == pages[j - 1] + 1 && !protected.contains(&pages[j]) {
            j += 1;
        }
        let run = &pages[first..j];
        i = j;

        let mut k = 0;
        while k < run.len() {
            let p = run[k];
            let rem = run.len() - k;
            if let Some((a, b, yes)) = last
                && (a..b).contains(&p)
            {
                let take = (b - p).min(rem);
                if yes {
                    out_pages.extend_from_slice(&run[k..k + take]);
                    out_ends.extend(std::iter::repeat_n(b, take));
                }
                k += take;
                continue;
            }
            // Fast path: the entire remaining run lies in one write-watched pool.
            if watched_range(p, rem) {
                let end = p + rem;
                out_pages.extend_from_slice(&run[k..]);
                out_ends.extend(std::iter::repeat_n(end, rem));
                last = Some((p, end, true));
                break;
            }
            // If the run starts in a write-watched pool and crosses a boundary,
            // find the exact watched prefix in O(log rem) probes so we neither
            // call VirtualQuery on a pool nor probe page by page.
            if rem > 1 && watched_range(p, 1) {
                let mut lo = 1usize;
                let mut hi = 2usize.min(rem);
                while hi < rem && watched_range(p, hi) {
                    lo = hi;
                    hi = (hi * 2).min(rem);
                }
                while lo + 1 < hi {
                    let mid = lo + (hi - lo) / 2;
                    if watched_range(p, mid) {
                        lo = mid;
                    } else {
                        hi = mid;
                    }
                }
                let end = p + lo;
                out_pages.extend_from_slice(&run[k..k + lo]);
                out_ends.extend(std::iter::repeat_n(end, lo));
                k += lo;
                continue;
            }
            if let Some((a, b, yes)) = query_region(p) {
                last = Some((a, b, yes));
                let take = b.saturating_sub(p).max(1).min(rem);
                if yes {
                    out_pages.extend_from_slice(&run[k..k + take]);
                    out_ends.extend(std::iter::repeat_n(b, take));
                }
                k += take;
            } else {
                k += 1;
            }
        }
    }
    (out_pages, out_ends)
}

/// Pages `[first, last)` changed, and so the pages showing the same memory
/// elsewhere: gives write access back where we guarded them (outside
/// [`GUARDS`]) and marks them in every tracker holding them. True if one of
/// `[first, last)` was guarded.
fn changed(g: &mut Held, first: usize, last: usize, save: bool) -> bool {
    let to_open: Vec<usize> = g.protected.range(first..last).copied().collect();
    for &page in &to_open {
        g.protected.remove(&page);
    }
    for (_, op) in g.guarding.range_mut(first..last) {
        op.reopen = true;
        op.skip_commit = true;
    }
    let guarded = !to_open.is_empty();
    let others: Vec<usize> = g.alias_of.range(first..last).flat_map(|(_, o)| o.iter().copied()).collect();
    cancel_guarding_aliases(g, &others);
    #[cfg(feature = "diagnostics")]
    {
        let held = others.iter().filter(|p| g.holders.contains_key(p)).count();
        crate::copystat::add(crate::copystat::C::AliasMarks, held as u64);
    }
    // Every page's bytes are saved before any is opened: once a page is
    // open, a write by another thread lands without a fault, and a save
    // after it would keep that write as the page's old bytes.
    mark_changed(g, first, last, save);
    // The pages showing the same memory are marked, not opened: nothing
    // writes through them now, and opening them only made the next take
    // guard them again, one Wine call each (two thirds of all protection
    // changes in #50's world).
    for page in others {
        mark_changed(g, page, page + 1, save);
    }
    open_sorted_pages(g, &to_open);
    guarded
}

/// Marks pages `[first, last)` changed in every tracker holding them,
/// leaving their protection as it is. With `save`, before the write lands:
/// trackers armed for a page they had taken keep its bytes.
fn mark_changed(g: &mut Guards, first: usize, last: usize, save: bool) {
    for (&page, holders) in g.holders.range(first..last) {
        for c in holders {
            let first_write = {
                let mut set = c.0.lock().unwrap_or_else(|e| e.into_inner());
                set_dirty(c.2, true);
                set.insert(page)
            };
            if first_write && save {
                c.save(page);
            }
        }
        #[cfg(feature = "diagnostics")]
        g.written.insert(page);
    }
}

/// At each presented frame (diagnostics): counts the tracked pages written
/// in it, and those written the frame before too.
#[cfg(feature = "diagnostics")]
pub(crate) fn frame_done() {
    let mut g = guards();
    let written = std::mem::take(&mut g.written);
    let again = written.intersection(&g.written_before).count();
    crate::copystat::add(crate::copystat::C::WrittenPages, written.len() as u64);
    crate::copystat::add(crate::copystat::C::WrittenAgain, again as u64);
    g.written_before = written;
}

/// A write into title memory the guards do not see, under way: its pages
/// stay open (a tracker taking them meanwhile would guard them, and the
/// kernel's write would fail) until it is dropped, which marks them changed
/// once more.
#[must_use = "the write must happen while this lives"]
pub struct Writing {
    first: usize,
    last: usize,
}

impl Drop for Writing {
    fn drop(&mut self) {
        // An unmapping with tracking off: nothing was noted.
        if self.first == self.last {
            return;
        }
        let mut g = guards();
        if let Some(i) = g.busy.iter().position(|&r| r == (self.first, self.last)) {
            g.busy.swap_remove(i);
        }
        changed(&mut g, self.first, self.last, false);
    }
}

/// Before our own write into title memory `[at, at + len)` through the
/// kernel (which does not fault into our handler) or another write the
/// guards do not see: opens its guarded blocks, marks them changed and
/// keeps them open while the result lives. The runtime calls it (as
/// `kernel_will_write`) around the title's kernel reads into its own
/// memory and DirectStorage's writes to addresses.
pub fn writing(at: usize, len: usize) -> Writing {
    // The length is the title's (`fread`'s size times count, saturated).
    let (first, last) = (at / PAGE, at.saturating_add(len).div_ceil(PAGE));
    let mut g = guards();
    g.busy.push((first, last));
    loop {
        changed(&mut g, first, last, true);
        if g.guarding.range(first..last).next().is_some() {
            g.wait(&GUARDING_CV);
            continue;
        }
        if overlaps_opening(&g, first, last) {
            g.wait(&OPENING_CV);
            continue;
        }
        break;
    }
    Writing { first, last }
}

/// The runtime mapped or unmapped title memory `[at, at + len)`: what it
/// shows is new, and read-write.
pub fn title_memory_remapped(at: usize, len: usize) {
    crate::texback::remapped(at, len);
    let (first, last) = (at / PAGE, at.saturating_add(len).div_ceil(PAGE));
    let mut g = guards();
    remapped(&mut g, first, last);
    let cancelled_ops: Vec<(usize, u64)> = g
        .guarding
        .range(first..last)
        .filter(|(_, op)| op.reopen)
        .map(|(&p, op)| (p, op.op_id))
        .collect();
    while cancelled_ops
        .iter()
        .any(|&(p, id)| g.guarding.get(&p).is_some_and(|op| op.op_id == id))
    {
        g.wait(&GUARDING_CV);
    }
    drop(g);
    crate::shadow::forget_range(at, len);
}

/// Before title memory `[at, at + len)` is unmapped or freed (`len` 0: the
/// whole allocation at `at`, as `VirtualFree` with `MEM_RELEASE` frees):
/// as [`title_memory_remapped`], and its pages stay unguarded until the
/// result is dropped, after the memory went.
///
/// A page guarded while it goes stays in the guard set after it went.
/// Memory allocated at that address later (the title's pools are freed
/// and allocated again; nothing tells the guards of an allocation) then
/// counts as guarded already and is never made read-only: its writes go
/// unseen, and copies of it keep stale bytes. So the runtime calls this
/// around its unmappings and the title's frees, and a tracker taking the
/// pages meanwhile leaves them changed, as during a kernel write.
pub fn title_memory_unmapping(at: usize, len: usize) -> Writing {
    if !fault_tracking() || at == 0 {
        return Writing { first: 0, last: 0 };
    }
    let len = if len == 0 { allocation_len(at) } else { len };
    let (first, last) = (at / PAGE, at.saturating_add(len).div_ceil(PAGE));
    let mut g = guards();
    g.busy.push((first, last));
    loop {
        remapped(&mut g, first, last);
        if g.guarding.range(first..last).next().is_some() {
            g.wait(&GUARDING_CV);
            continue;
        }
        if overlaps_opening(&g, first, last) {
            g.wait(&OPENING_CV);
            continue;
        }
        break;
    }
    drop(g);
    crate::shadow::forget_range(at, len);
    Writing { first, last }
}

/// The bytes of the allocation that starts at `at`: its regions while they
/// share that allocation base (guarded and open pages are regions of their
/// own). 0 if nothing is allocated there.
fn allocation_len(at: usize) -> usize {
    use windows_sys::Win32::System::Memory::{MEMORY_BASIC_INFORMATION, VirtualQuery};
    let mut len = 0usize;
    loop {
        // SAFETY: plain data filled in by the query.
        let mut info: MEMORY_BASIC_INFORMATION = unsafe { std::mem::zeroed() };
        // SAFETY: a valid out pointer of the right size.
        let n = unsafe { VirtualQuery((at + len) as *const _, &mut info, size_of_val(&info)) };
        if n == 0 || info.AllocationBase as usize != at || info.RegionSize == 0 {
            return len;
        }
        len += info.RegionSize;
    }
}

/// [`title_memory_remapped`] of pages `[first, last)`, under the guard lock.
fn remapped(g: &mut Held, first: usize, last: usize) {
    // What it showed is gone already: nothing to keep.
    changed(g, first, last, false);
    mark_retaken(g, first, last);
    // What it showed is not what it shows now: its links go, both ways.
    let stale: Vec<usize> = g.alias_of.range(first..last).map(|(&p, _)| p).collect();
    for page in stale {
        for other in g.alias_of.remove(&page).unwrap_or_default() {
            if let Some(links) = g.alias_of.get_mut(&other) {
                links.retain(|&p| p != page);
                if links.is_empty() {
                    g.alias_of.remove(&other);
                }
            }
        }
    }
}

/// Reports GPU shader writes into aliased title memory `[at, at + len)` so
/// any other trackers over those pages (such as textures' staging) know they
/// changed and re-upload on next use. Fast-paths out when no tracker or alias
/// covers the range.
pub(crate) fn notify_written(at: usize, len: usize) {
    if len == 0 {
        return;
    }
    let (first, last) = (at / PAGE, at.saturating_add(len).div_ceil(PAGE));
    let mut g = guards();
    if g.holders.range(first..last).next().is_none()
        && g.alias_of.range(first..last).next().is_none()
    {
        return;
    }
    changed(&mut g, first, last, false);
}

/// What shaders wrote into host copies of `[at, at + len)` is dropped:
/// every host buffer copy holding the memory takes it from title memory
/// again at its next refresh (the pages count as changed in each, aliases
/// too). Title memory itself did not change, so its guards stay (opened page
/// by page, they cost ~15 % of the submitting thread at each frame end, run
/// perf15), and other trackers (textures' staging, heaps) are left alone:
/// shaders' writes never reach them (marked, textures over withheld ranges
/// were uploaded again for nothing: 19 % of the uploads, run evict3).
pub(crate) fn retake(at: usize, len: usize) {
    let (first, last) = (at / PAGE, at.saturating_add(len).div_ceil(PAGE));
    let mut g = guards();
    for (_, op) in g.guarding.range_mut(first..last) {
        op.skip_commit = true;
    }
    let others: Vec<usize> = g.alias_of.range(first..last).flat_map(|(_, o)| o.iter().copied()).collect();
    cancel_guarding_aliases(&mut g, &others);
    mark_copies(&g, first, last);
    for &page in &others {
        mark_copies(&g, page, page + 1);
    }
    drop(g);
    crate::shadow::forget_range(at, len);
    for page in others {
        crate::shadow::forget_range(page * PAGE, PAGE);
    }
}

/// Marks pages `[first, last)` as retaken in every host buffer copy's tracker
/// holding them so cached baseline piece hashes are discarded before refresh.
fn mark_retaken(g: &Guards, first: usize, last: usize) {
    for (&page, holders) in g.holders.range(first..last) {
        for c in holders.iter().filter(|c| c.3) {
            let mut set = c.5.lock().unwrap_or_else(|e| e.into_inner());
            set.insert(page);
            c.4.store(true, Ordering::Release);
        }
    }
}

/// Pages `[first, last)` count as changed in every host buffer copy's
/// tracker holding them.
fn mark_copies(g: &Guards, first: usize, last: usize) {
    for (&page, holders) in g.holders.range(first..last) {
        for c in holders.iter().filter(|c| c.3) {
            let mut set = c.0.lock().unwrap_or_else(|e| e.into_inner());
            set_dirty(c.2, true);
            set.insert(page);
            drop(set);
            let mut retaken = c.5.lock().unwrap_or_else(|e| e.into_inner());
            retaken.insert(page);
            c.4.store(true, Ordering::Release);
        }
    }
}

/// The runtime mapped the page at `at` where `others` show it already: the
/// pages are linked both ways, and if we guard it there, it is guarded here
/// too (a write through the new mapping must not go unseen by the trackers
/// there).
pub fn title_memory_aliased(at: usize, others: &[usize]) {
    if others.is_empty() {
        return;
    }
    let mut g = guards();
    let page = at / PAGE;
    for &other in others {
        for (x, y) in [(other / PAGE, page), (page, other / PAGE)] {
            let links = g.alias_of.entry(x).or_default();
            if !links.contains(&y) {
                links.push(y);
            }
        }
    }
    // If the new mapping cannot be guarded (a write the guards do not see is
    // under way through it), the page is changed; its bytes now may hold that
    // write already, so nothing is saved of them.
    let need_guard = others.iter().any(|&o| {
        let op = o / PAGE;
        g.protected.contains(&op) || g.guarding.contains_key(&op)
    });
    if need_guard && protect(&mut g, &[page]).is_empty() {
        changed(&mut g, page, page + 1, false);
    }
}

/// Whether a page of `[at, at + len)` is shown elsewhere too, as the
/// runtime told when it mapped it (tracked or not): a write here changes
/// the bytes there. Asked at submission, by a batch of refreshes that
/// keeps what a copy into title memory cannot have changed; it takes the
/// guard lock for one lookup and allocates nothing under it.
pub(crate) fn aliased(at: usize, len: usize) -> bool {
    let (first, last) = (at / PAGE, at.saturating_add(len).div_ceil(PAGE));
    guards().alias_of.range(first..last).next().is_some()
}

/// Alias pages (page numbers) mapping the same physical memory as `[at, at + len)`.
pub(crate) fn alias_pages(at: usize, len: usize) -> Vec<usize> {
    if len == 0 {
        return Vec::new();
    }
    let (first, last) = (at / PAGE, at.saturating_add(len).div_ceil(PAGE));
    guards()
        .alias_of
        .range(first..last)
        .flat_map(|(_, o)| o.iter().copied())
        .collect()
}

/// Handles a write access violation at `at`. Returns `true` when execution
/// should resume (`EXCEPTION_CONTINUE_EXECUTION`), `false` when the fault does
/// not belong to our guards (`EXCEPTION_CONTINUE_SEARCH`).
fn handle_write_fault(at: usize) -> bool {
    let page = at / PAGE;
    #[cfg(feature = "diagnostics")]
    let _timed = Timed(std::time::Instant::now(), crate::copystat::C::HandlerUs);
    if HOLDING.with(|h| h.get()) {
        if read_only(at) {
            open_under_lock(page);
            return true;
        }
        // Another thread's `open_sorted_pages` can concurrently restore
        // `PAGE_READWRITE` outside `GUARDS` before `read_only(at)` runs.
        if writable(at) {
            note_opened_under_lock(page);
            return true;
        }
        return false;
    }
    let mut g = guards();
    let mut waited_opening = false;
    let mut cancelled_guard = false;
    while is_opening(&g, page) || g.guarding.contains_key(&page) {
        if is_opening(&g, page) {
            waited_opening = true;
            g.wait(&OPENING_CV);
        } else {
            mark_changed(&mut g, page, page + 1, true);
            let others: Vec<usize> = g
                .alias_of
                .get(&page)
                .into_iter()
                .flatten()
                .copied()
                .collect();
            cancel_guarding_aliases(&mut g, &others);
            for alias in others {
                mark_changed(&mut g, alias, alias + 1, true);
            }
            if let Some(op) = g.guarding.get_mut(&page) {
                op.reopen = true;
                op.skip_commit = true;
                cancelled_guard = true;
            }
            g.wait(&GUARDING_CV);
        }
    }
    if changed(&mut g, page, page + 1, true) {
        drop(g);
        #[cfg(feature = "diagnostics")]
        crate::copystat::add(crate::copystat::C::GuardFaults, 1);
        return true;
    }
    if waited_opening {
        drop(g);
        return true;
    }
    // Check `writable(at)` while still holding `GUARDS` so a concurrent
    // `protect_with_aliases` cannot flip `page` back to `PAGE_READONLY`
    // between dropping `GUARDS` and querying the page's protection, which
    // would otherwise turn a stale fault on an already-opened guarded page
    // into an unhandled access violation.
    let handled = writable(at);
    drop(g);
    #[cfg(feature = "diagnostics")]
    if cancelled_guard && handled {
        crate::copystat::add(crate::copystat::C::GuardFaults, 1);
    }
    #[cfg(not(feature = "diagnostics"))]
    let _ = cancelled_guard;
    handled
}

/// Registers the handler that takes writes to guarded blocks, once, ahead
/// of the crash reporter's.
fn install() {
    use windows_sys::Win32::Foundation::{EXCEPTION_ACCESS_VIOLATION, NTSTATUS};
    use windows_sys::Win32::System::Diagnostics::Debug::{
        AddVectoredExceptionHandler, EXCEPTION_CONTINUE_EXECUTION, EXCEPTION_CONTINUE_SEARCH,
        EXCEPTION_POINTERS,
    };
    unsafe extern "system" fn on_write(info: *mut EXCEPTION_POINTERS) -> i32 {
        // SAFETY: the system passes the exception being dispatched.
        let r = unsafe { &*(*info).ExceptionRecord };
        const WRITE: usize = 1;
        if r.ExceptionCode != EXCEPTION_ACCESS_VIOLATION as NTSTATUS
            || r.NumberParameters < 2
            || r.ExceptionInformation[0] != WRITE
        {
            return EXCEPTION_CONTINUE_SEARCH;
        }
        if handle_write_fault(r.ExceptionInformation[1]) {
            EXCEPTION_CONTINUE_EXECUTION
        } else {
            EXCEPTION_CONTINUE_SEARCH
        }
    }
    static ONCE: std::sync::Once = std::sync::Once::new();
    // SAFETY: the handler only touches our guard state and protections.
    ONCE.call_once(|| unsafe {
        AddVectoredExceptionHandler(1, Some(on_write));
    });
}

/// Whether `at` is committed memory that is read-only, as our guards make
/// it (ignoring cache-modifier flags, rejecting `PAGE_GUARD`).
fn read_only(at: usize) -> bool {
    use windows_sys::Win32::System::Memory::{
        MEM_COMMIT, MEMORY_BASIC_INFORMATION, PAGE_GUARD, PAGE_READONLY, VirtualQuery,
    };
    // SAFETY: plain data filled in by the query.
    let mut info: MEMORY_BASIC_INFORMATION = unsafe { std::mem::zeroed() };
    // SAFETY: a valid out pointer of the right size.
    let n = unsafe { VirtualQuery(at as *const _, &mut info, size_of_val(&info)) };
    n != 0
        && info.State == MEM_COMMIT
        && info.Protect & PAGE_GUARD == 0
        && info.Protect & 0xff == PAGE_READONLY
}

/// Whether `at` is committed read-write memory now (ignoring cache-modifier
/// flags, rejecting `PAGE_GUARD`).
fn writable(at: usize) -> bool {
    use windows_sys::Win32::System::Memory::{
        MEM_COMMIT, MEMORY_BASIC_INFORMATION, PAGE_EXECUTE_READWRITE, PAGE_GUARD, PAGE_READWRITE,
        VirtualQuery,
    };
    // SAFETY: plain data filled in by the query.
    let mut info: MEMORY_BASIC_INFORMATION = unsafe { std::mem::zeroed() };
    // SAFETY: a valid out pointer of the right size.
    let n = unsafe { VirtualQuery(at as *const _, &mut info, size_of_val(&info)) };
    n != 0
        && info.State == MEM_COMMIT
        && info.Protect & PAGE_GUARD == 0
        && matches!(info.Protect & 0xff, PAGE_READWRITE | PAGE_EXECUTE_READWRITE)
}

/// Adds the time since it was made to a counter when dropped (diagnostics).
#[cfg(feature = "diagnostics")]
pub(crate) struct Timed(pub(crate) std::time::Instant, pub(crate) crate::copystat::C);

#[cfg(feature = "diagnostics")]
impl Drop for Timed {
    fn drop(&mut self) {
        let c = std::mem::replace(&mut self.1, crate::copystat::C::Submissions);
        crate::copystat::add(c, self.0.elapsed().as_micros() as u64);
    }
}

#[cfg(test)]
pub(crate) mod tests {
    use super::*;

    #[test]
    fn a_fault_under_the_guard_lock_does_not_hang() {
        use windows_sys::Win32::System::Memory::{PAGE_READONLY, VirtualProtect};
        install();
        // SAFETY: a fresh block of our own.
        let block = unsafe { VirtualAlloc(std::ptr::null(), PAGE, MEM_RESERVE | MEM_COMMIT, PAGE_READWRITE) }.cast::<u8>();
        assert!(!block.is_null());
        let mut old = 0;
        // SAFETY: as above.
        assert_ne!(unsafe { VirtualProtect(block.cast(), PAGE, PAGE_READONLY, &mut old) }, 0);
        {
            let _g = guards();
            // SAFETY: inside the block; the handler gives write access back.
            unsafe { block.add(100).write_volatile(7) };
        }
        // SAFETY: as above.
        assert_eq!(unsafe { block.add(100).read_volatile() }, 7);
        assert!(writable(block as usize));
        drop(guards());
        assert!(OPENED_UNDER_LOCK.iter().all(|s| s.load(std::sync::atomic::Ordering::Relaxed) != block as usize / PAGE));
        // SAFETY: allocated above.
        assert_ne!(unsafe { VirtualFree(block.cast(), 0, MEM_RELEASE) }, 0);
    }

    use windows_sys::Win32::System::Memory::{
        MEM_COMMIT, MEM_RELEASE, MEM_RESERVE, PAGE_READWRITE, VirtualAlloc, VirtualFree,
    };
    pub(crate) const MEM_WRITE_WATCH: u32 = 0x20_0000;

    #[test]
    fn fault_tracking_follows_the_variable() {
        assert!(decide(Some("1"), false));
        assert!(!decide(Some("0"), true));
        assert!(decide(None, true));
        assert!(!decide(None, false));
        assert!(!decide(Some("yes"), false));
        // Tests keep tracking on, under Rosetta too.
        assert!(fault_tracking() || std::env::var("VICTORY_FAULT_TRACKING").as_deref() == Ok("0"));
    }

    #[test]
    fn tracking_is_opt_in_under_rosetta() {
        // The variable decides when it is set; otherwise tracking is on
        // except under Rosetta (`fault_tracking`'s default is `!rosetta()`).
        assert!(decide(Some("1"), false));
        assert!(!decide(Some("0"), true));
        assert!(decide(None, true));
        assert!(!decide(None, false));
        assert!(!decide(Some("2"), false));
        // Stores are patched exactly when tracking runs under Rosetta.
        assert_eq!(stores_need_patching(), fault_tracking() && rosetta());
    }

    /// A write-watched allocation of `pages` pages (as the runtime's pools),
    /// on a 64 KiB boundary (the allocation granularity).
    pub(crate) fn pool(pages: usize) -> usize {
        // SAFETY: a fresh allocation, freed by the test.
        let p = unsafe {
            VirtualAlloc(
                std::ptr::null(),
                pages * PAGE,
                MEM_RESERVE | MEM_COMMIT | MEM_WRITE_WATCH,
                PAGE_READWRITE,
            )
        } as usize;
        assert!(p != 0 && p.is_multiple_of(PAGE));
        p
    }

    /// Frees a test's allocation at `p`, unless the guards still guard or
    /// hold a page of it: then it stays allocated, as the other tests'
    /// pools do.
    ///
    /// A copy that a batch, a page table or a widened view still holds
    /// keeps its tracker after the test drops its own handle. Freed under
    /// it, the pages stayed in `protected`, and a test running at the same
    /// time that was given the same address got them unguarded:
    /// [`protect`] skips a page it has guarded already, so the new
    /// allocation stayed writable and its writes went unseen (#31: a guard
    /// test failed in 7 of 150 full runs on an M5 Pro, whichever drew the
    /// address; 2 of 300 with this). The title's own memory does not go
    /// this way: the runtime tells the guards when it unmaps it
    /// ([`title_memory_remapped`]).
    pub(crate) fn free(p: usize) {
        if still_watched(p) {
            return;
        }
        // SAFETY: an allocation of ours.
        unsafe { VirtualFree(p as *mut _, 0, MEM_RELEASE) };
    }

    /// Whether the guards guard or hold a page of the allocation at `p`.
    fn still_watched(p: usize) -> bool {
        use windows_sys::Win32::System::Memory::{MEMORY_BASIC_INFORMATION, VirtualQuery};
        // The allocation's length: its regions, while they start from `p`
        // (guarded and open pages are regions of their own).
        let mut len = 0;
        loop {
            // SAFETY: plain data filled in by the query.
            let mut info: MEMORY_BASIC_INFORMATION = unsafe { std::mem::zeroed() };
            // SAFETY: a valid out pointer of the right size.
            let n = unsafe { VirtualQuery((p + len) as *const _, &mut info, size_of_val(&info)) };
            if n == 0 || info.AllocationBase as usize != p || info.RegionSize == 0 {
                break;
            }
            len += info.RegionSize;
        }
        let (first, last) = (p / PAGE, (p + len).div_ceil(PAGE));
        let g = guards();
        g.protected.range(first..last).next().is_some()
            || g.guarding.range(first..last).next().is_some()
            || overlaps_opening(&g, first, last)
            || g.holders.range(first..last).next().is_some()
    }

    /// Everything is changed at first; after taking, only what is written
    /// (by faults, or reported) is, and every tracker of a page sees it.
    /// A tracked range that starts in reserved memory and runs into plain
    /// committed memory (the host's, as the process heap was in #50) never
    /// makes that memory read-only; it is left out of the tracker instead
    /// (copied whole at each use, so the tracker is never clean).
    #[test]
    fn plain_committed_memory_is_never_guarded() {
        use windows_sys::Win32::System::Memory::{
            MEM_COMMIT, MEM_RELEASE, MEM_RESERVE, MEMORY_BASIC_INFORMATION, PAGE_READWRITE, VirtualAlloc,
            VirtualFree, VirtualQuery,
        };
        // SAFETY: plain reservation and commit of our own memory.
        let base = unsafe { VirtualAlloc(std::ptr::null(), 4 * PAGE, MEM_RESERVE, PAGE_READWRITE) } as usize;
        assert_ne!(base, 0);
        // The last two blocks committed as plain private memory.
        // SAFETY: inside our reservation.
        assert!(!unsafe { VirtualAlloc((base + 2 * PAGE) as *const _, 2 * PAGE, MEM_COMMIT, PAGE_READWRITE) }.is_null());
        let t = track(base, 4 * PAGE, true).expect("reserved memory counts as runtime memory");
        assert!(!t.is_untracked(base / PAGE) && !t.is_untracked(base / PAGE + 1));
        assert!(t.is_untracked(base / PAGE + 2) && t.is_untracked(base / PAGE + 3));
        let _ = t.take(base, base + 4 * PAGE);
        // SAFETY: plain data filled in by the query.
        let mut info: MEMORY_BASIC_INFORMATION = unsafe { std::mem::zeroed() };
        // SAFETY: a valid out pointer of the right size.
        unsafe { VirtualQuery((base + 2 * PAGE) as *const _, &mut info, size_of_val(&info)) };
        assert_eq!(info.Protect, PAGE_READWRITE, "still writable");
        assert!(!t.changed(base / PAGE + 2) && !t.changed(base / PAGE + 3), "not tracked");
        assert_eq!(t.untracked(base, base + 4 * PAGE), [(base + 2 * PAGE, base + 4 * PAGE)], "copied whole");
        assert!(!t.clean(), "never clean");
        // SAFETY: writable, so no fault.
        unsafe { ((base + 2 * PAGE + 8) as *mut u8).write_volatile(1) };
        drop(t);
        // SAFETY: our reservation.
        unsafe { VirtualFree(base as *mut _, 0, MEM_RELEASE) };
    }

    #[test]
    fn writes_mark_every_tracker_of_their_page() {
        let p = pool(4);
        let a = track(p, 4 * PAGE, true).expect("pool memory is tracked");
        let b = track(p + PAGE, 2 * PAGE, true).expect("tracked");
        assert!(!a.clean() && !b.clean(), "changed at first");
        assert_eq!(a.take(p, p + 4 * PAGE), [(p, p + 4 * PAGE)]);
        assert!(a.clean() && !b.clean());
        assert_eq!(b.take(p, p + 4 * PAGE), [(p + PAGE, p + 3 * PAGE)]);
        assert!(a.take(p, p + 4 * PAGE).is_empty() && b.take(p, p + 4 * PAGE).is_empty());
        assert!(a.clean() && b.clean());
        // SAFETY: committed pages of ours, now read-only: the handler opens
        // the block; the second write needs no fault.
        unsafe {
            ((p + PAGE + 8) as *mut u8).write_volatile(1);
            ((p + PAGE + 9) as *mut u8).write_volatile(2);
        }
        assert!(!a.clean() && !b.clean(), "a write is seen by every tracker");
        assert_eq!(a.take(p, p + 4 * PAGE), [(p + PAGE, p + 2 * PAGE)]);
        assert!(a.clean());
        // `a` guarded it again; `b` still has it changed.
        assert!(b.changed(p / PAGE + 1));
        assert_eq!(b.take(p, p + 4 * PAGE), [(p + PAGE, p + 2 * PAGE)]);
        // SAFETY: as above: a fault again, for both.
        unsafe { ((p + PAGE + 10) as *mut u8).write_volatile(3) };
        assert!(a.changed(p / PAGE + 1) && b.changed(p / PAGE + 1));
        assert_eq!(a.take(p, p + 4 * PAGE), [(p + PAGE, p + 2 * PAGE)]);
        drop(writing(p + 3 * PAGE + 5, 1));
        assert_eq!(a.take(p, p + 4 * PAGE), [(p + 3 * PAGE, p + 4 * PAGE)]);
        title_memory_remapped(p, PAGE);
        assert!(a.changed(p / PAGE) && !a.changed(p / PAGE + 1));
        drop((a, b));
        free(p);
    }

    /// A range running from free memory over another allocation into a pool
    /// (a 4 MiB window of memory outside buffers) tracks the pool's pages
    /// and leaves the other allocation alone: taking guards the pool only.
    #[test]
    fn a_range_over_foreign_memory_guards_only_the_runtimes() {
        use windows_sys::Win32::System::Memory::{MEMORY_BASIC_INFORMATION, PAGE_READONLY, VirtualQuery};
        // This test tracks a mostly free address range. In the parallel suite,
        // other tests can map their views in its holes even after both of our
        // allocations succeed. Keep those views out of this address space.
        const CHILD: &str = "VICTORY_TEST_FOREIGN_WINDOW_CHILD";
        if std::env::var_os(CHILD).is_none() {
            let name = "watch::tests::a_range_over_foreign_memory_guards_only_the_runtimes";
            let result = std::process::Command::new(std::env::current_exe().unwrap())
                .args(["--exact", name, "--nocapture"])
                .env(CHILD, "1")
                .output()
                .expect("run foreign-window test in a child");
            let stdout = String::from_utf8_lossy(&result.stdout);
            assert!(
                result.status.success() && stdout.contains("1 passed"),
                "foreign-window child {}: {stdout}{}",
                result.status,
                String::from_utf8_lossy(&result.stderr)
            );
            return;
        }
        const WINDOW: usize = 4 << 20;
        // In a free 4 MiB-aligned window: plain committed memory (as the
        // process heap's) 1 MiB in, a pool 2 MiB in.
        let (base, foreign, pool) = (0..8)
            .find_map(|_| {
                // SAFETY: a fresh reservation, released at once; then
                // allocations in the window it left free, freed below.
                unsafe {
                    let p = VirtualAlloc(std::ptr::null(), 2 * WINDOW, MEM_RESERVE, PAGE_READWRITE) as usize;
                    VirtualFree(p as *mut _, 0, MEM_RELEASE);
                    let base = p.next_multiple_of(WINDOW);
                    let at = |offset: usize, kind: u32| {
                        VirtualAlloc((base + offset) as *const _, PAGE, MEM_RESERVE | MEM_COMMIT | kind, PAGE_READWRITE) as usize
                    };
                    let (foreign, pool) = (at(1 << 20, 0), at(2 << 20, MEM_WRITE_WATCH));
                    if p != 0 && foreign == base + (1 << 20) && pool == base + (2 << 20) {
                        return Some((base, foreign, pool));
                    }
                    [foreign, pool].into_iter().filter(|&q| q != 0).for_each(free);
                    None
                }
            })
            .expect("a free window");
        let t = track(base, WINDOW, true).expect("the pool is tracked");
        let taken = t.take(base, base + WINDOW);
        assert!(taken.iter().any(|&(a, b)| a <= pool && pool + PAGE <= b), "{taken:x?}");
        assert!(!taken.iter().any(|&(a, b)| a < foreign + PAGE && foreign < b), "{taken:x?}");
        let protection = |p: usize| {
            // SAFETY: plain data filled in by the query.
            let mut info: MEMORY_BASIC_INFORMATION = unsafe { std::mem::zeroed() };
            // SAFETY: a valid out pointer of the right size.
            unsafe { VirtualQuery(p as *const _, &mut info, size_of_val(&info)) };
            info.Protect
        };
        assert_eq!(protection(pool), PAGE_READONLY, "the pool is guarded");
        assert_eq!(protection(foreign), PAGE_READWRITE, "the other allocation is not");
        drop(t);
        free(foreign);
        free(pool);
    }

    /// A tracker's dirty bit goes back when it goes, and the next tracker
    /// starts dirty on it (everything changed at first).
    #[test]
    fn a_new_tracker_starts_dirty_on_a_bit_given_back() {
        let p = pool(1);
        let a = track(p, PAGE, true).expect("tracked");
        assert_eq!(a.take(p, p + PAGE).len(), 1);
        assert!(a.clean() && clean_bit(a.bit()));
        drop(a);
        let b = track(p, PAGE, true).expect("tracked");
        assert!(!b.clean() && !clean_bit(b.bit()));
        assert!(!clean_bit(None), "no bit: never clean");
        drop(b);
        free(p);
    }

    /// A page with nothing mapped is taken as zeros: `take_split` lists it
    /// apart from the pages to copy.
    #[test]
    fn unmapped_pages_are_taken_apart() {
        // SAFETY: a fresh write-watched reservation of two pages, the first
        // committed; freed below.
        let p = unsafe {
            let p = VirtualAlloc(std::ptr::null(), 2 * PAGE, MEM_RESERVE | MEM_WRITE_WATCH, PAGE_READWRITE) as usize;
            assert_ne!(p, 0);
            assert_ne!(VirtualAlloc(p as *const _, PAGE, MEM_COMMIT, PAGE_READWRITE) as usize, 0);
            p
        };
        let t = track(p, 2 * PAGE, true).expect("tracked");
        assert_eq!(t.take_split(p, p + 2 * PAGE), (vec![(p, p + PAGE)], vec![(p + PAGE, p + 2 * PAGE)]));
        assert!(t.take(p, p + 2 * PAGE).is_empty(), "both taken");
        drop(t);
        free(p);
    }

    /// A page only partly in the range asked is copied in part and stays
    /// changed; edge pages count whole when the tracked part is covered.
    #[test]
    fn partial_pages_stay_changed() {
        let p = pool(3);
        let t = track(p + 100, 2 * PAGE, true).expect("tracked");
        assert_eq!(t.take(p + 100, p + PAGE + 10), [(p + 100, p + PAGE + 10)]);
        assert!(!t.changed(p / PAGE) && t.changed(p / PAGE + 1));
        assert!(!t.clean(), "a page taken in part stays changed");
        assert_eq!(t.take(p, p + 3 * PAGE), [(p + PAGE, p + 2 * PAGE + 100)]);
        assert!(t.take(p, p + 3 * PAGE).is_empty());
        assert!(t.clean());
        drop(t);
        free(p);
    }

    #[test]
    fn batched_partial_ranges_match_sequential_takes() {
        let p = pool(3);
        let ranges = [
            (p + 100, p + PAGE + 10),
            (p + PAGE + 20, p + 2 * PAGE),
            (p + PAGE, p + 3 * PAGE),
            (p, p),
            (p, p + 3 * PAGE),
        ];
        let sequential = track(p, 3 * PAGE, true).expect("tracked");
        let expected: Vec<_> = ranges.iter().map(|&(a, b)| sequential.take_split(a, b)).collect();
        drop(sequential);
        let batched = track(p, 3 * PAGE, true).expect("tracked");
        assert_eq!(batched.take_split_many(&ranges), expected);
        assert!(batched.clean(), "the final range takes the remaining partial page");
        drop(batched);
        free(p);
    }

    /// Pages of the ranges armed keep their bytes from before the first
    /// write after they were taken, whichever overlapping ranges armed
    /// them; a page unarmed keeps none until it is armed again, and none
    /// does once the tracker is disarmed.
    #[test]
    fn armed_ranges_keep_their_pages_bytes_from_before_the_first_write() {
        let p = pool(4);
        let t = track(p, 4 * PAGE, true).expect("pool memory is tracked");
        let all = || t.take(p, p + 4 * PAGE);
        // SAFETY: (every write below) a byte of our pool; a guarded page
        // faults into the handler, which saves it first where it is armed.
        let write = |page: usize, value: u8| unsafe { ((p + page * PAGE + 8) as *mut u8).write_volatile(value) };
        let saved = |page: usize| t.saved(p / PAGE + page).map(|bytes| bytes[8]);
        all();
        write(0, 1);
        assert_eq!(saved(0), None, "nothing armed yet");
        all();
        // Bytes 100 of page 0 to 1 of page 2: pages 0 to 2, page 3 not.
        t.arm(p + 100, p + 2 * PAGE + 1);
        t.arm(p + PAGE, p + 2 * PAGE);
        t.arm(p + PAGE + 5, p + 2 * PAGE + 1);
        t.unarm(p / PAGE + 1);
        for page in 0..4 {
            write(page, 2);
        }
        assert_eq!([saved(0), saved(1), saved(2), saved(3)], [Some(1), None, Some(0), None]);
        assert_eq!(saved(0), None, "handed out once");
        all();
        // Armed again by a range holding it.
        t.arm(p + PAGE, p + 2 * PAGE);
        write(1, 3);
        write(1, 4);
        assert_eq!(saved(1), Some(2), "from before the first write");
        t.disarm();
        all();
        write(2, 5);
        assert_eq!(saved(2), None, "disarmed");
        drop(t);
        free(p);
    }

    /// Runs armed that touch or overlap merge into one, so a page's check
    /// stays a search however many sub-ranges are armed; pages unarmed in
    /// them stay unarmed until a range holding them is armed again.
    #[test]
    fn adjacent_and_overlapping_armed_runs_coalesce() {
        let mut s = Saved::default();
        // Adjacent: [2k, 2k + 2) end to end, in order and out of it.
        for k in 0..200 {
            s.arm(2 * k, 2 * k + 2);
        }
        assert_eq!(s.armed, [(0, 400)]);
        let mut s = Saved::default();
        for k in (0..200).rev() {
            s.arm(2 * k, 2 * k + 2);
        }
        assert_eq!(s.armed, [(0, 400)]);
        // Overlapping, nested and disjoint.
        let mut s = Saved::default();
        s.arm(10, 20);
        s.arm(15, 25);
        assert_eq!(s.armed, [(10, 25)]);
        s.arm(12, 14);
        assert_eq!(s.armed, [(10, 25)], "nested");
        s.arm(40, 50);
        s.arm(0, 5);
        assert_eq!(s.armed, [(0, 5), (10, 25), (40, 50)], "disjoint, sorted");
        s.arm(5, 10);
        assert_eq!(s.armed, [(0, 25), (40, 50)], "touching both sides");
        s.arm(30, 35);
        s.arm(20, 45);
        assert_eq!(s.armed, [(0, 50)], "spanning several");
        s.arm(60, 70);
        s.arm(55, 55);
        assert_eq!(s.armed, [(0, 50), (60, 70)], "an empty run arms nothing");
        assert!(s.armed(0) && s.armed(49) && !s.armed(50) && !s.armed(59));
        assert!(s.armed(60) && s.armed(69) && !s.armed(70) && !s.armed(1000));
        // Unarmed pages stay out of a merged run until armed again.
        s.unarmed.insert(45);
        s.unarmed.insert(65);
        s.arm(50, 60);
        assert_eq!(s.armed, [(0, 70)]);
        assert!(!s.armed(45) && !s.armed(65) && s.armed(55));
        s.arm(44, 46);
        assert!(s.armed(45) && !s.armed(65));
        s.arm(0, 70);
        assert!(s.armed(65) && s.unarmed.is_empty());
    }

    /// Plain committed memory (the C heap) is not tracked; reserved memory
    /// is, reads as taken, and changes when the runtime maps it.
    #[test]
    fn heap_memory_is_not_tracked_and_placeholders_wait_for_mappings() {
        let plain = vec![0u8; 2 * PAGE];
        assert!(track(plain.as_ptr() as usize, plain.len(), true).is_none());
        // SAFETY: a reservation of ours, freed below.
        let r = unsafe { VirtualAlloc(std::ptr::null(), 2 * PAGE, MEM_RESERVE, PAGE_READWRITE) } as usize;
        assert_ne!(r, 0);
        let t = track(r, 2 * PAGE, true).expect("reserved memory is tracked");
        assert_eq!(t.take(r, r + 2 * PAGE).len(), 1, "copied once (as zeros)");
        assert!(t.take(r, r + 2 * PAGE).is_empty());
        title_memory_remapped(r, PAGE);
        assert_eq!(t.take(r, r + 2 * PAGE), [(r, r + PAGE)]);
        drop(t);
        // SAFETY: our reservation.
        unsafe { VirtualFree(r as *mut _, 0, MEM_RELEASE) };
    }

    /// A kernel read into a guarded view block fails; opened first, it
    /// lands and the block counts as changed.
    #[test]
    fn kernel_reads_into_guarded_blocks_after_opening() {
        use std::io::{Read, Seek, Write};
        use windows_sys::Win32::System::Memory::{
            CreateFileMappingW, FILE_MAP_WRITE, MapViewOfFile, UnmapViewOfFile,
        };
        // SAFETY: a pagefile section of two blocks, mapped read-write.
        let view = unsafe {
            let section = CreateFileMappingW(
                -1isize as _,
                std::ptr::null(),
                PAGE_READWRITE,
                0,
                (2 * PAGE) as u32,
                std::ptr::null(),
            );
            assert!(!section.is_null());
            MapViewOfFile(section, FILE_MAP_WRITE, 0, 0, 2 * PAGE).Value as usize
        };
        assert_ne!(view, 0);
        assert!(track(view, 2 * PAGE, false).is_none(), "views only where asked");
        let t = track(view, 2 * PAGE, true).expect("views are tracked");
        assert_eq!(t.take(view, view + 2 * PAGE), [(view, view + 2 * PAGE)]); // now read-only
        let mut file = tempfile::tempfile().unwrap();
        file.write_all(b"from the disk").unwrap();
        // `ReadFile` inside: the kernel writes the view.
        let read = |file: &mut std::fs::File| {
            file.rewind().unwrap();
            // SAFETY: 13 bytes of our view, nothing else refers to them.
            let into = unsafe { std::slice::from_raw_parts_mut((view + 100) as *mut u8, 13) };
            file.read(into).is_ok_and(|n| n == 13)
        };
        assert!(!read(&mut file), "the kernel does not fault into our handler");
        let write = writing(view + 100, 13);
        // A tracker taking the page meanwhile leaves it open.
        assert_eq!(t.take(view, view + 2 * PAGE), [(view, view + PAGE)]);
        assert!(read(&mut file));
        drop(write);
        // SAFETY: the bytes just read.
        assert_eq!(unsafe { std::slice::from_raw_parts((view + 100) as *const u8, 13) }, b"from the disk");
        assert_eq!(t.take(view, view + 2 * PAGE), [(view, view + PAGE)]);
        drop(t);
        // As the runtime does before it unmaps: the guards let go of the
        // view's pages. Dropping the tracker opens a page only when nothing
        // else holds it; otherwise the pages stayed in the guard set after
        // the unmap, and memory a test running at the same time was given
        // there counted as guarded already, so its writes went unseen (#31).
        title_memory_remapped(view, 2 * PAGE);
        // SAFETY: our view.
        unsafe { UnmapViewOfFile(windows_sys::Win32::System::Memory::MEMORY_MAPPED_VIEW_ADDRESS { Value: view as *mut _ }) };
    }

    /// `pages` pages of a fresh pagefile section mapped twice (as the
    /// title's pool view and tiled heap show the same physical pages),
    /// reported as the runtime does when it maps them.
    pub(crate) fn twin_views(pages: usize) -> (usize, usize) {
        use windows_sys::Win32::System::Memory::{CreateFileMappingW, FILE_MAP_WRITE, MapViewOfFile};
        let len = pages * PAGE;
        // SAFETY: a pagefile section mapped read-write twice; the test
        // keeps both views.
        let (a, b) = unsafe {
            let section = CreateFileMappingW(
                -1isize as _,
                std::ptr::null(),
                PAGE_READWRITE,
                0,
                len as u32,
                std::ptr::null(),
            );
            assert!(!section.is_null());
            (
                MapViewOfFile(section, FILE_MAP_WRITE, 0, 0, len).Value as usize,
                MapViewOfFile(section, FILE_MAP_WRITE, 0, 0, len).Value as usize,
            )
        };
        assert!(a != 0 && b != 0);
        for k in (0..len).step_by(PAGE) {
            title_memory_aliased(b + k, &[a + k]);
        }
        (a, b)
    }

    thread_local! {
        /// `VirtualProtect` calls this thread made in `protect`.
        pub(super) static PROTECTS: std::cell::Cell<usize> = const { std::cell::Cell::new(0) };
    }

    /// `single` views of one page each side by side, as the runtime's page
    /// mappings make them, then one view of `single` pages: the address of
    /// the first, and of the long one.
    fn views_side_by_side(single: usize) -> (usize, usize) {
        use windows_sys::Win32::System::Memory::{
            CreateFileMappingW, MEM_PRESERVE_PLACEHOLDER, MEM_REPLACE_PLACEHOLDER,
            MEM_RESERVE_PLACEHOLDER, MapViewOfFile3, PAGE_NOACCESS, VirtualAlloc2,
        };
        use windows_sys::Win32::System::Threading::GetCurrentProcess;
        let len = 2 * single * PAGE;
        // SAFETY: a pagefile section and a placeholder of our own, split
        // and filled with views of the section; the test keeps them.
        unsafe {
            let section = CreateFileMappingW(
                -1isize as _,
                std::ptr::null(),
                PAGE_READWRITE,
                0,
                len as u32,
                std::ptr::null(),
            );
            assert!(!section.is_null());
            let base = VirtualAlloc2(
                GetCurrentProcess(),
                std::ptr::null(),
                len,
                MEM_RESERVE | MEM_RESERVE_PLACEHOLDER,
                PAGE_NOACCESS,
                std::ptr::null_mut(),
                0,
            ) as usize;
            assert_ne!(base, 0);
            let map = |at: usize, offset: usize, n: usize| {
                VirtualFree(at as *mut _, n * PAGE, MEM_RELEASE | MEM_PRESERVE_PLACEHOLDER);
                let view = MapViewOfFile3(
                    section,
                    GetCurrentProcess(),
                    at as *const _,
                    offset as u64,
                    n * PAGE,
                    MEM_REPLACE_PLACEHOLDER,
                    PAGE_READWRITE,
                    std::ptr::null_mut(),
                    0,
                );
                assert_eq!(view.Value as usize, at);
            };
            for k in 0..single {
                map(base + k * PAGE, k * PAGE, 1);
            }
            let long = base + single * PAGE;
            map(long, single * PAGE, single);
            (base, long)
        }
    }

    /// Guarding pages of separate views takes one call per view, not a
    /// call over all of them that fails first and then one per page.
    #[test]
    fn takes_guard_one_view_per_call() {
        let calls = || PROTECTS.with(std::cell::Cell::get);
        // Four one-page views and a four-page view, taken together.
        let (base, _) = views_side_by_side(4);
        let t = track(base, 8 * PAGE, true).expect("views are tracked");
        let before = calls();
        assert_eq!(t.take(base, base + 8 * PAGE), [(base, base + 8 * PAGE)]);
        assert_eq!(calls() - before, 5);
        assert!((0..8).all(|k| read_only(base + k * PAGE)));
        drop(t);
        // Each kind alone, written after: the guards catch it.
        let (base, long) = views_side_by_side(4);
        for (at, want) in [(base, 4), (long, 1)] {
            let t = track(at, 4 * PAGE, true).expect("views are tracked");
            let before = calls();
            assert_eq!(t.take(at, at + 4 * PAGE), [(at, at + 4 * PAGE)]);
            assert_eq!(calls() - before, want, "calls for the views at {at:#x}");
            for k in 0..4 {
                assert!(read_only(at + k * PAGE), "page {k} at {at:#x} guarded");
            }
            // Guarded, they mark the tracker when written.
            // SAFETY: a page of the views; the handler opens it.
            unsafe { ((at + 2 * PAGE + 8) as *mut u8).write_volatile(1) };
            assert!(t.changed(at / PAGE + 2));
            assert!(!t.changed(at / PAGE + 1));
        }
    }

    /// A write through another mapping of the same pages marks the tracker
    /// (the guard there faults, the runtime says where else it is mapped);
    /// so does a write the runtime reports at the other mapping.
    #[test]
    fn writes_through_another_mapping_mark_the_tracker() {
        let (a, b) = twin_views(2);
        let t = track(a, 2 * PAGE, true).expect("views are tracked");
        assert_eq!(t.take(a, a + 2 * PAGE), [(a, a + 2 * PAGE)]);
        // SAFETY: the second view of the pages, guarded by the take.
        unsafe { ((b + PAGE + 4) as *mut u32).write_volatile(7) };
        assert!(read_only(a + PAGE), "marked, still guarded: no call to guard it again");
        assert_eq!(t.take(a, a + 2 * PAGE), [(a + PAGE, a + 2 * PAGE)]);
        // SAFETY: the tracked view, which shows the write.
        assert_eq!(unsafe { ((a + PAGE + 4) as *const u32).read_volatile() }, 7);
        drop(writing(b, 8));
        assert_eq!(t.take(a, a + 2 * PAGE), [(a, a + PAGE)]);
        assert!(t.take(a, a + 2 * PAGE).is_empty());
        // A third mapping made later starts guarded, so its writes count.
        let (c, _) = twin_views(1);
        title_memory_aliased(c, &[a]);
        // SAFETY: a view of ours (another section: only the tracking
        // takes it for `a`'s pages here).
        unsafe { (c as *mut u32).write_volatile(1) };
        assert_eq!(t.take(a, a + 2 * PAGE), [(a, a + PAGE)]);
    }

    /// Pages are reported aliased while the runtime says they are shown
    /// twice, and not after either mapping changes.
    #[test]
    fn aliased_pages_are_found_by_range() {
        let (a, b) = twin_views(3);
        assert!(
            aliased(a, 1) && aliased(b + 2 * PAGE + 5, 10),
            "either view, any byte"
        );
        assert!(aliased(a + PAGE - 1, 2), "a range that reaches a page");
        assert!(!aliased(a, 0), "no bytes");
        let plain = pool(1);
        assert!(!aliased(plain, PAGE));
        free(plain);
        // One page shown once only again: its link goes, the others stay.
        title_memory_remapped(b + PAGE, PAGE);
        assert!(!aliased(a + PAGE, PAGE) && !aliased(b + PAGE, PAGE));
        assert!(aliased(a, PAGE) && aliased(b + 2 * PAGE, PAGE));
        assert!(aliased(a, usize::MAX), "no overflow");
    }

    /// A tracker of one mapping going away leaves the guard its page shares
    /// with a tracker of another mapping, so a write there still counts (it
    /// was opened, and the pages it went unseen in stayed clean).
    #[test]
    fn a_tracker_dropped_on_one_mapping_leaves_the_other_guarded() {
        let (a, b) = twin_views(1);
        let t = track(a, PAGE, true).expect("views are tracked");
        let u = track(b, PAGE, true).expect("views are tracked");
        assert_eq!(t.take(a, a + PAGE), [(a, a + PAGE)]);
        assert_eq!(u.take(b, b + PAGE), [(b, b + PAGE)]);
        drop(u);
        // SAFETY: the second view, guarded for `t`'s page too.
        unsafe { ((b + 4) as *mut u32).write_volatile(7) };
        assert_eq!(t.take(a, a + PAGE), [(a, a + PAGE)]);
    }

    /// A retake has every copy of the memory, and of its other mappings,
    /// take it again, and leaves it guarded: a later write still counts.
    #[test]
    fn a_retake_marks_the_copies_and_keeps_the_guard() {
        use windows_sys::Win32::System::Memory::{MEMORY_BASIC_INFORMATION, PAGE_READONLY, VirtualQuery};
        let protection = |at: usize| {
            // SAFETY: plain data filled in by the query.
            let mut info: MEMORY_BASIC_INFORMATION = unsafe { std::mem::zeroed() };
            // SAFETY: a valid out pointer of the right size.
            unsafe { VirtualQuery(at as *const _, &mut info, size_of_val(&info)) };
            info.Protect
        };
        let (a, b) = twin_views(2);
        let t = track_copy(a, 2 * PAGE, true).expect("views are tracked");
        let u = track_copy(b, 2 * PAGE, true).expect("views are tracked");
        assert_eq!(t.take(a, a + 2 * PAGE), [(a, a + 2 * PAGE)]);
        assert_eq!(u.take(b, b + 2 * PAGE), [(b, b + 2 * PAGE)]);
        assert!(t.clean() && u.clean());
        retake(a + PAGE + 16, 32);
        // Marked in both copies, the second mapping's too, still guarded.
        assert!(!t.clean() && !u.clean());
        assert!(t.changed((a + PAGE) / PAGE) && u.changed((b + PAGE) / PAGE));
        assert!(!t.changed(a / PAGE));
        assert_eq!(protection(a + PAGE), PAGE_READONLY);
        assert_eq!(protection(b + PAGE), PAGE_READONLY);
        // Taken again (the copies re-copy that page), then guarded and clean.
        assert_eq!(t.take(a, a + 2 * PAGE), [(a + PAGE, a + 2 * PAGE)]);
        assert_eq!(u.take(b, b + 2 * PAGE), [(b + PAGE, b + 2 * PAGE)]);
        assert!(t.clean() && u.clean());
        assert_eq!(protection(a + PAGE), PAGE_READONLY);
        // SAFETY: the second view, guarded all along.
        unsafe { ((b + PAGE + 8) as *mut u32).write_volatile(3) };
        assert_eq!(t.take(a, a + 2 * PAGE), [(a + PAGE, a + 2 * PAGE)]);
        assert_eq!(u.take(b, b + 2 * PAGE), [(b + PAGE, b + 2 * PAGE)]);
        // A retake while marked and guarded, then a drop: nothing breaks.
        retake(a, 2 * PAGE);
        drop(u);
        assert_eq!(t.take(a, a + 2 * PAGE), [(a, a + 2 * PAGE)]);
        assert!(t.clean());
    }

    /// A retake marks host buffer copies holding the memory, not other
    /// trackers of it (a texture's staging): shaders' writes reach only the
    /// copies. A title write still marks both.
    #[test]
    fn a_retake_marks_buffer_copies_alone() {
        let p = pool(2);
        let copy = track_copy(p, 2 * PAGE, true).expect("pool memory is tracked");
        let texture = track(p, 2 * PAGE, true).expect("pool memory is tracked");
        assert_eq!(copy.take(p, p + 2 * PAGE), [(p, p + 2 * PAGE)]);
        assert_eq!(texture.take(p, p + 2 * PAGE), [(p, p + 2 * PAGE)]);
        retake(p + PAGE, 16);
        assert!(copy.changed(p / PAGE + 1), "the copy takes it again");
        assert!(!texture.changed(p / PAGE + 1) && texture.clean(), "the texture's tracker is left alone");
        assert_eq!(copy.take(p, p + 2 * PAGE), [(p + PAGE, p + 2 * PAGE)]);
        // SAFETY: a page of our pool, guarded: the handler opens it.
        unsafe { ((p + PAGE + 8) as *mut u8).write_volatile(1) };
        assert!(copy.changed(p / PAGE + 1) && texture.changed(p / PAGE + 1), "a title write marks both");
    }

    /// A write under way on another mapping of a page keeps the page open
    /// there, so taking the page (or marking it taken) through its tracked
    /// mapping must leave it changed and keep what was saved of it; aliasing
    /// a busy mapping to a guarded one must mark the guarded page changed.
    #[test]
    fn a_busy_alias_leaves_the_page_changed_and_keeps_saved_bytes() {
        let (a, b) = twin_views(1);
        let t = track(a, PAGE, true).expect("views are tracked");
        assert_eq!(t.take(a, a + PAGE), [(a, a + PAGE)]);
        t.arm(a, a + PAGE);
        let write = writing(b, 16);
        assert!(t.changed(a / PAGE));
        assert_eq!(t.take(a, a + PAGE), [(a, a + PAGE)]);
        assert!(t.changed(a / PAGE), "take must leave the page changed");
        t.taken(&[a / PAGE]);
        assert!(t.changed(a / PAGE), "taken must leave the page changed");
        // SAFETY: the second view of the page, kept open by `write`.
        unsafe { (b as *mut u32).write_volatile(0x1234) };
        assert_eq!(t.take(a, a + PAGE), [(a, a + PAGE)]);
        drop(write);
        let saved = t.saved(a / PAGE).expect("saved bytes survive");
        assert_eq!(&saved[..4], &[0, 0, 0, 0]);
        assert_eq!(t.take(a, a + PAGE), [(a, a + PAGE)]);
        assert!(t.take(a, a + PAGE).is_empty(), "taken once the write ended");
        let (c, _) = twin_views(1);
        let busy_c = writing(c, 16);
        title_memory_aliased(c, &[a]);
        assert!(
            t.changed(a / PAGE),
            "aliasing a busy page marks the tracked one"
        );
        drop(busy_c);
    }

    /// Memory given up while a tracker holds it guarded (a pool page
    /// decommitted here; the title frees and allocates its pools again,
    /// which no one tells the guards of) is guarded when it comes back: the
    /// guard goes with it, and a take while it goes leaves it unguarded. A
    /// guard set that still named the page would skip it as guarded already,
    /// and the new memory's writes would go unseen.
    #[test]
    fn memory_given_up_under_a_guard_is_guarded_when_it_comes_back() {
        use windows_sys::Win32::System::Memory::MEM_DECOMMIT;
        let p = pool(2);
        let page = p + PAGE;
        let t = track(p, 2 * PAGE, true).expect("pool memory is tracked");
        assert_eq!(t.take(p, p + 2 * PAGE), [(p, p + 2 * PAGE)]);
        assert!(read_only(page));
        let going = title_memory_unmapping(page, PAGE);
        assert!(writable(page), "given up: no guard while it goes");
        assert_eq!(
            t.take(p, p + 2 * PAGE),
            [(page, p + 2 * PAGE)],
            "changed, and taken"
        );
        assert!(writable(page), "a take while it goes guards nothing");
        // SAFETY: the second page of our pool; the reservation stays.
        let freed = unsafe { VirtualFree(page as *mut _, PAGE, MEM_DECOMMIT) };
        assert_ne!(freed, 0);
        drop(going);
        // SAFETY: the same page of our reservation, committed again.
        let back =
            unsafe { VirtualAlloc(page as *const _, PAGE, MEM_COMMIT, PAGE_READWRITE) } as usize;
        assert_eq!(back, page);
        assert_eq!(t.take(p, p + 2 * PAGE), [(page, p + 2 * PAGE)]);
        assert!(read_only(page), "the memory that came back is guarded");
        // SAFETY: a byte of the page, guarded: the handler opens it.
        unsafe { ((page + 8) as *mut u8).write_volatile(1) };
        assert!(t.changed(page / PAGE), "and its writes are seen");
        drop(t);
        free(p);
    }

    /// With length 0, the whole allocation at the address goes (as
    /// `VirtualFree` with `MEM_RELEASE` frees it).
    #[test]
    fn unmapping_with_no_length_takes_the_whole_allocation() {
        let p = pool(3);
        let t = track(p, 3 * PAGE, true).expect("pool memory is tracked");
        assert_eq!(t.take(p, p + 3 * PAGE), [(p, p + 3 * PAGE)]);
        let going = title_memory_unmapping(p, 0);
        assert!((0..3).all(|k| writable(p + k * PAGE)));
        assert_eq!(t.take(p, p + 3 * PAGE), [(p, p + 3 * PAGE)]);
        drop(going);
        drop(t);
        free(p);
    }

    /// A write or mapping that runs past the end of the address space (a
    /// title's `fread` with a huge count) covers up to the end; the sum used
    /// to overflow. Up there no test has guarded blocks.
    #[test]
    fn ranges_past_the_address_space_end_do_not_overflow() {
        let top = usize::MAX - 16;
        drop(writing(top, PAGE));
        title_memory_remapped(top, usize::MAX);
    }

    /// `Tracked::taken` must never guard pages outside the tracker's runtime
    /// pages (such as plain committed memory inside a wider shadow window).
    #[test]
    fn taken_never_guards_untracked_pages() {
        use windows_sys::Win32::System::Memory::{
            MEM_COMMIT, MEM_RELEASE, MEM_RESERVE, MEMORY_BASIC_INFORMATION, PAGE_READWRITE, VirtualAlloc,
            VirtualFree, VirtualQuery,
        };
        // SAFETY: plain reservation and commit of our own memory.
        let base = unsafe { VirtualAlloc(std::ptr::null(), 4 * PAGE, MEM_RESERVE, PAGE_READWRITE) } as usize;
        assert_ne!(base, 0);
        // SAFETY: inside our reservation.
        assert!(!unsafe { VirtualAlloc((base + 2 * PAGE) as *const _, 2 * PAGE, MEM_COMMIT, PAGE_READWRITE) }.is_null());
        let t = track(base, 4 * PAGE, true).expect("reserved memory counts as runtime memory");
        assert!(!t.is_untracked(base / PAGE) && !t.is_untracked(base / PAGE + 1));
        assert!(t.is_untracked(base / PAGE + 2) && t.is_untracked(base / PAGE + 3));
        let _ = t.take(base, base + 4 * PAGE);
        t.taken(&[base / PAGE + 2, base / PAGE + 3]);
        // SAFETY: plain data filled in by the query.
        let mut info: MEMORY_BASIC_INFORMATION = unsafe { std::mem::zeroed() };
        // SAFETY: a valid out pointer of the right size.
        unsafe { VirtualQuery((base + 2 * PAGE) as *const _, &mut info, size_of_val(&info)) };
        assert_eq!(info.Protect, PAGE_READWRITE, "untracked plain memory must stay writable");
        drop(t);
        // SAFETY: our reservation.
        unsafe { VirtualFree(base as *mut _, 0, MEM_RELEASE) };
    }

    /// A contiguous run inside one write-watched pool is checked with a single
    /// `GetWriteWatch` range probe instead of one per page (#460), and a run
    /// crossing a pool boundary splits in `O(log n)` probes without querying
    /// `VirtualQuery` on the pool.
    #[test]
    fn filter_runtime_pages_batches_watched_runs_and_splits_boundaries() {
        let protected = BTreeSet::new();

        // Single pool of 4096 pages: 1 watched_range call, 0 query_region calls.
        let pages: Vec<usize> = (100..4196).collect();
        let (mut watch_calls, mut query_calls) = (0usize, 0usize);
        let (kept, ends) = filter_runtime_pages(
            &pages,
            &protected,
            |first, count| {
                watch_calls += 1;
                first >= 100 && first + count <= 4196
            },
            |_| {
                query_calls += 1;
                None
            },
        );
        assert_eq!(watch_calls, 1);
        assert_eq!(query_calls, 0);
        assert_eq!(kept, pages);
        assert!(ends.iter().all(|&e| e == 4196));

        // Mixed run: pool A [0..1000), plain private [1000..1500), view [1500..2000).
        let mixed: Vec<usize> = (0..2000).collect();
        let (mut watch_calls, mut query_calls) = (0usize, 0usize);
        let (kept, ends) = filter_runtime_pages(
            &mixed,
            &protected,
            |first, count| {
                watch_calls += 1;
                first < 1000 && first + count <= 1000
            },
            |p| {
                query_calls += 1;
                if (1000..1500).contains(&p) {
                    Some((1000, 1500, false))
                } else if (1500..2000).contains(&p) {
                    Some((1500, 2000, true))
                } else {
                    None
                }
            },
        );
        assert!(watch_calls <= 32, "expected O(log n) probes, got {watch_calls}");
        assert_eq!(query_calls, 2, "one query per non-pool region");
        let expected: Vec<usize> = (0..1000).chain(1500..2000).collect();
        assert_eq!(kept, expected);
        assert!(ends[..1000].iter().all(|&e| e == 1000));
        assert!(ends[1000..].iter().all(|&e| e == 2000));
    }

    /// When a bulk `VirtualProtect` fails on a run containing uncommitted pages,
    /// `protect_run_pages` skips the rest of each uncommitted `VirtualQuery`
    /// region without calling `VirtualProtect` or `VirtualQuery` on every
    /// unmapped page individually.
    #[test]
    fn protect_run_pages_skips_uncommitted_regions_without_per_page_probes() {
        let run: Vec<usize> = (100..164).collect(); // 64 pages
        let mut protected = BTreeSet::new();
        let mut taken = Vec::new();
        let mut unmapped_pages = Vec::new();
        let (mut protect_calls, mut query_calls) = (0usize, 0usize);

        // Pages 100..110 are committed; 110..164 are one uncommitted reservation.
        protect_run_pages(
            &run,
            &mut protected,
            &mut taken,
            &mut unmapped_pages,
            |from, to| {
                protect_calls += 1;
                to <= 110 && from >= 100
            },
            |page| {
                query_calls += 1;
                (page >= 110).then_some(164)
            },
        );

        // 1 bulk call + 10 committed per-page calls + 1 failing call at 110 = 12.
        assert_eq!(protect_calls, 12);
        assert_eq!(query_calls, 1, "one VirtualQuery for the whole uncommitted tail");
        assert_eq!(taken, run);
        assert_eq!(protected, (100..110).collect::<BTreeSet<_>>());
        assert_eq!(unmapped_pages, (110..164).collect::<Vec<_>>());
    }

    /// Stale write faults (where another thread already restored `PAGE_READWRITE`
    /// before the fault handler ran, both outside and inside `HOLDING`) must
    /// resume execution and mark tracked pages dirty, while genuinely
    /// non-writable or `PAGE_GUARD` pages must remain unhandled.
    #[test]
    fn stale_write_fault_and_protection_flags_handled_cleanly() {
        use windows_sys::Win32::System::Memory::{
            PAGE_EXECUTE_READWRITE, PAGE_GUARD, PAGE_NOACCESS, PAGE_NOCACHE, PAGE_READONLY,
            VirtualProtect,
        };
        let p = pool(1);
        let t = track(p, PAGE, true).expect("pool memory is tracked");
        assert_eq!(t.take(p, p + PAGE), [(p, p + PAGE)]);
        assert!(read_only(p));
        assert!(!t.changed(p / PAGE));

        // Simulate another thread opening the page before a racing thread
        // enters the fault handler.
        {
            let mut g = guards();
            assert!(g.protected.remove(&(p / PAGE)));
            open_sorted_pages(&mut g, &[p / PAGE]);
        }
        assert!(writable(p));
        assert!(
            handle_write_fault(p + 0x488),
            "stale fault on opened tracked page must resume execution"
        );
        assert!(
            t.changed(p / PAGE),
            "stale fault on tracked page must mark tracker dirty"
        );

        // Stale fault while `HOLDING == true` on an already-writable page must
        // also resume execution and queue the page for marking on lock release.
        assert_eq!(t.take(p, p + PAGE), [(p, p + PAGE)]);
        {
            let mut g = guards();
            assert!(g.protected.remove(&(p / PAGE)));
            open_sorted_pages(&mut g, &[p / PAGE]);
            assert!(
                handle_write_fault(p + 0x488),
                "stale fault under HOLDING on writable page must resume"
            );
        }
        assert!(
            t.changed(p / PAGE),
            "re-entrant stale fault must drain and mark tracker dirty on lock drop"
        );

        // Negative controls: genuinely non-writable pages must stay unhandled.
        // SAFETY: fresh reserved page owned by this test, freed below.
        let unmapped =
            unsafe { VirtualAlloc(std::ptr::null(), PAGE, MEM_RESERVE, PAGE_NOACCESS) } as usize;
        assert_ne!(unmapped, 0);
        assert!(!handle_write_fault(unmapped + 16));
        {
            let _g = guards();
            assert!(!handle_write_fault(unmapped + 16));
        }
        // SAFETY: reservation allocated above.
        assert_ne!(unsafe { VirtualFree(unmapped as *mut _, 0, MEM_RELEASE) }, 0);

        // Protection-flag matrix on a private scratch page.
        // SAFETY: fresh private page owned by this test, freed below.
        let scratch = unsafe {
            VirtualAlloc(
                std::ptr::null(),
                PAGE,
                MEM_RESERVE | MEM_COMMIT,
                PAGE_READWRITE,
            )
        } as usize;
        assert_ne!(scratch, 0);
        let set_prot = |prot: u32| {
            let mut old = 0;
            // SAFETY: `scratch` is a live committed private page.
            unsafe { VirtualProtect(scratch as *const _, PAGE, prot, &mut old) != 0 }
        };
        assert!(set_prot(PAGE_NOACCESS));
        assert!(!writable(scratch) && !read_only(scratch));
        assert!(!handle_write_fault(scratch + 8));

        assert!(set_prot(PAGE_READONLY));
        assert!(read_only(scratch) && !writable(scratch));
        assert!(!handle_write_fault(scratch + 8));

        if set_prot(PAGE_READWRITE | PAGE_NOCACHE) {
            assert!(writable(scratch));
        }
        assert!(set_prot(PAGE_EXECUTE_READWRITE));
        assert!(writable(scratch));
        assert!(set_prot(PAGE_READWRITE | PAGE_GUARD));
        assert!(!writable(scratch) && !read_only(scratch));
        assert!(set_prot(PAGE_READWRITE));
        // SAFETY: private page allocated above.
        assert_ne!(unsafe { VirtualFree(scratch as *mut _, 0, MEM_RELEASE) }, 0);

        drop(t);
        free(p);
    }

    /// Concurrent writers hammering real hardware faults, `Tracked::take_split`,
    /// and `watch::writing` must never lose a CPU write or re-protect a busy
    /// page while `Writing` is held.
    #[test]
    fn concurrent_writers_take_and_writing_never_lose_updates() {
        use std::sync::atomic::AtomicBool;

        const PAGES: usize = 6;
        let p = pool(PAGES);
        let t = track(p, PAGES * PAGE, true).expect("pool memory is tracked");
        let _ = t.take(p, p + PAGES * PAGE);
        let shadow = Mutex::new(vec![0u8; PAGES * PAGE]);
        let done = AtomicBool::new(false);

        let sync_into_shadow = |shadow: &Mutex<Vec<u8>>| {
            let (copy, zero) = t.take_split(p, p + PAGES * PAGE);
            assert!(zero.is_empty(), "all pool pages are committed");
            if copy.is_empty() {
                return;
            }
            let mut dst = shadow.lock().unwrap_or_else(|e| e.into_inner());
            for (from, to) in copy {
                // SAFETY: `[from, to)` lies inside our live committed pool `p`.
                let src = unsafe { std::slice::from_raw_parts(from as *const u8, to - from) };
                dst[from - p..to - p].copy_from_slice(src);
            }
        };

        std::thread::scope(|s| {
            // Taker continuously re-guards and refreshes the shadow copy.
            let shadow_ref = &shadow;
            let done_ref = &done;
            s.spawn(move || {
                while !done_ref.load(Ordering::Acquire) {
                    sync_into_shadow(shadow_ref);
                    std::thread::yield_now();
                }
            });

            // Kernel-writer continuously opens the last page via `writing` and
            // asserts it stays writable across concurrent `take_split` calls.
            s.spawn(move || {
                let busy_addr = p + (PAGES - 1) * PAGE;
                for iter in 0..80usize {
                    let guard = writing(busy_addr, PAGE);
                    assert!(
                        writable(busy_addr),
                        "busy page must stay writable while Writing is held"
                    );
                    // SAFETY: inside our committed pool; `guard` holds it open.
                    unsafe { ((busy_addr + (iter % 1024)) as *mut u8).write_volatile(iter as u8) };
                    assert!(
                        writable(busy_addr),
                        "concurrent take_split must not re-protect a busy page"
                    );
                    drop(guard);
                    std::thread::yield_now();
                }
            });

            // Three writer threads fault across pages 0..PAGES-1.
            let writers: Vec<_> = (0..3usize)
                .map(|worker| {
                    s.spawn(move || {
                        for iter in 0..150usize {
                            let page_idx = (iter + worker) % (PAGES - 1);
                            let off = (worker * 16 + (iter % 64) * 64) % (PAGE - 8);
                            let addr = p + page_idx * PAGE + off;
                            let val = ((worker as u64 + 1) << 32) | (iter as u64 + 1);
                            // SAFETY: aligned 8-byte slot inside our live pool;
                            // each worker owns disjoint offsets (`worker * 16`).
                            unsafe { (addr as *mut u64).write_volatile(val) };
                        }
                    })
                })
                .collect();

            for w in writers {
                w.join().expect("writer thread panicked");
            }
            done.store(true, Ordering::Release);
        });

        // Final take must observe every write made while concurrent takes ran.
        sync_into_shadow(&shadow);
        let dst = shadow.lock().unwrap_or_else(|e| e.into_inner());
        // SAFETY: `p` is our live committed pool and no writer is running now.
        let live = unsafe { std::slice::from_raw_parts(p as *const u8, PAGES * PAGE) };
        assert_eq!(
            &dst[..],
            live,
            "shadow copy missed at least one concurrent CPU write"
        );

        drop(t);
        free(p);
    }

    /// Stale write faults racing with continuous unlocked re-protection must
    /// never return unhandled.
    #[test]
    fn concurrent_stale_faults_and_reprotect_never_go_unhandled() {
        use std::sync::atomic::AtomicBool;

        let p = pool(2);
        let t = track(p, 2 * PAGE, true).expect("pool memory is tracked");
        let _ = t.take(p, p + 2 * PAGE);
        let done = AtomicBool::new(false);

        std::thread::scope(|s| {
            let t_ref = &t;
            let done_ref = &done;
            s.spawn(move || {
                while !done_ref.load(Ordering::Acquire) {
                    let _ = t_ref.take(p, p + 2 * PAGE);
                    std::thread::yield_now();
                }
            });
            let writers: Vec<_> = (0..3usize)
                .map(|worker| {
                    s.spawn(move || {
                        for iter in 0..150usize {
                            let off = (iter * 64 + worker * 8) % PAGE;
                            // SAFETY: live 2-page pool owned by this test.
                            unsafe { ((p + off) as *mut u8).write_volatile(iter as u8) };
                            assert!(
                                handle_write_fault(p + PAGE + off),
                                "concurrent fault handler on tracked page must never return unhandled"
                            );
                        }
                    })
                })
                .collect();
            for w in writers {
                w.join().expect("writer thread panicked");
            }
            done.store(true, Ordering::Release);
        });

        drop(t);
        free(p);
    }

    /// Copy-back preservation under unlocked guards:
    /// - an armed page written by CPU keeps exact pre-write bytes in `saved()`,
    /// - an unarmed page written by CPU stays `changed()` with no `saved()`,
    /// - `taken()` racing with concurrent CPU writes never leaves a writable
    ///   page invisible (`!changed()`).
    #[test]
    fn concurrent_armed_copy_back_preserves_cpu_writes_and_saved_baselines() {
        use std::sync::atomic::AtomicBool;

        let p = pool(3);
        let t = track_copy(p, 3 * PAGE, true).expect("pool memory is tracked");

        // Seed distinct pre-write bytes on pages 0 and 1, then guard them.
        // SAFETY: live pool pages owned by this test.
        unsafe {
            ((p + 16) as *mut u32).write_volatile(0xAAAA_1111);
            ((p + PAGE + 16) as *mut u32).write_volatile(0xBBBB_2222);
        }
        assert_eq!(t.take(p, p + 3 * PAGE), [(p, p + 3 * PAGE)]);

        // Arm pages 0 and 1, then unarm page 1 (simulating copy-back setup).
        t.arm(p, p + 2 * PAGE);
        t.unarm(p / PAGE + 1);

        // CPU writes both pages while armed/unarmed.
        // SAFETY: live pool pages; handler saves page 0 before opening it.
        unsafe {
            ((p + 16) as *mut u32).write_volatile(0xDEAD_BEEF);
            ((p + PAGE + 16) as *mut u32).write_volatile(0xCAFE_BABE);
        }
        let saved0 = t.saved(p / PAGE).expect("armed page must save pre-write bytes");
        assert_eq!(
            u32::from_le_bytes(saved0[16..20].try_into().unwrap()),
            0xAAAA_1111,
            "saved baseline must capture pre-write bytes before PAGE_READWRITE"
        );
        assert!(t.changed(p / PAGE));
        assert!(t.saved(p / PAGE + 1).is_none(), "unarmed page saves nothing");
        assert!(
            t.changed(p / PAGE + 1),
            "unarmed page written by CPU must stay changed so copy-back keeps CPU bytes"
        );

        // Race `taken()` against a concurrent CPU writer on page 2: a writable
        // page must NEVER be left invisible (`!t.changed()`).
        let page2 = p / PAGE + 2;
        let done = AtomicBool::new(false);
        std::thread::scope(|s| {
            let done_ref = &done;
            s.spawn(move || {
                let mut v = 1u32;
                while !done_ref.load(Ordering::Acquire) {
                    // SAFETY: live pool page owned by this test.
                    unsafe { ((p + 2 * PAGE + 32) as *mut u32).write_volatile(v) };
                    v = v.wrapping_add(1);
                }
            });
            for _ in 0..120 {
                t.taken(&[page2]);
                // Under `GUARDS`, check the fundamental invariant: if page 2 is
                // not guarded (not in `protected`), it MUST be marked changed.
                let g = guards();
                let guarded = g.protected.contains(&page2);
                let dirty = t.changed(page2);
                drop(g);
                assert!(
                    guarded || dirty,
                    "a writable page must never be invisible to Tracked::changed"
                );
            }
            done.store(true, Ordering::Release);
        });

        drop(t);
        free(p);
    }

    /// Concurrent writes through an alias mapping racing with unlocked
    /// `protect_with_aliases` on the primary tracker must never miss an update.
    #[test]
    fn concurrent_alias_writes_and_take_never_miss_updates() {
        use std::sync::atomic::AtomicBool;

        let (a, b) = twin_views(2);
        let t = track(a, 2 * PAGE, true).expect("views are tracked");
        let _ = t.take(a, a + 2 * PAGE);
        let shadow = Mutex::new(vec![0u8; 2 * PAGE]);
        let done = AtomicBool::new(false);

        let sync = |shadow: &Mutex<Vec<u8>>| {
            let runs = t.take(a, a + 2 * PAGE);
            if runs.is_empty() {
                return;
            }
            let mut dst = shadow.lock().unwrap_or_else(|e| e.into_inner());
            for (from, to) in runs {
                // SAFETY: `[from, to)` lies inside our mapped view `a`.
                let src = unsafe { std::slice::from_raw_parts(from as *const u8, to - from) };
                dst[from - a..to - a].copy_from_slice(src);
            }
        };

        std::thread::scope(|s| {
            let shadow_ref = &shadow;
            let done_ref = &done;
            s.spawn(move || {
                while !done_ref.load(Ordering::Acquire) {
                    sync(shadow_ref);
                    std::thread::yield_now();
                }
            });
            for iter in 0..150usize {
                let page_idx = iter % 2;
                let off = (iter * 16) % (PAGE - 8);
                // SAFETY: write through alias view `b`; backed by same section as `a`.
                unsafe { ((b + page_idx * PAGE + off) as *mut u64).write_volatile(iter as u64 + 1) };
            }
            done.store(true, Ordering::Release);
        });

        sync(&shadow);
        let dst = shadow.lock().unwrap_or_else(|e| e.into_inner());
        // SAFETY: `a` is our live mapped view and no writer is running now.
        let live = unsafe { std::slice::from_raw_parts(a as *const u8, 2 * PAGE) };
        assert_eq!(
            &dst[..],
            live,
            "shadow copy missed at least one concurrent write through alias view"
        );

        drop(t);
        title_memory_remapped(a, 2 * PAGE);
        title_memory_remapped(b, 2 * PAGE);
    }
}
