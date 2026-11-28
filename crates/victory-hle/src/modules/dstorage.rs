//! The console's DirectStorage (`XCRDAPI.dll`): a factory that opens files
//! and creates request queues and status arrays the title polls. Its
//! interfaces have grown since 2020 (new interface ids, new slots); the
//! slots here are the ones Grand Theft Auto VI calls, read from its call
//! sites. Slots nobody has needed yet log their offset.
//!
//! Submitted requests of every queue run on one pool of workers, as the
//! console's hardware runs them: in parallel, started in queue order,
//! queues served by weighted round robin over their priorities (about 10x
//! per level, counted in bytes), with at most the staging buffer's worth of
//! compressed source in flight. Each request reads its source (a file or
//! memory), inflates zlib data and/or unpacks BCPack data, and writes the
//! destination (an address, or physical pages). Completion is what stays
//! in order: a status entry is written once every request enqueued before
//! it on its queue has finished, with the first failure (in queue order)
//! since the previous status entry, or success.
#![allow(non_snake_case)]
#![allow(clippy::not_unsafe_ptr_arg_deref)]

use std::borrow::Cow;
use std::collections::{BTreeMap, BTreeSet, HashMap, VecDeque};
use std::ffi::{c_char, c_void};
use std::os::windows::fs::FileExt;
use std::os::windows::io::FromRawHandle;
use std::sync::atomic::{AtomicBool, AtomicI32, AtomicU32, AtomicU64, Ordering};
use std::sync::{Condvar, Mutex, MutexGuard, OnceLock};

use victory_formats::bcpack::{self, Tables};

use super::xasync::{E_INVALIDARG, E_NOINTERFACE, E_NOTIMPL, E_OUTOFMEMORY, E_PENDING};
use super::xgameruntime::{Guid, HRESULT, Methods, Object, S_OK, add_ref, missing, put, text};
use crate::{HleModule, hle_export};

/// The factory interface Grand Theft Auto VI asks for.
const FACTORY: Guid = Guid::new(
    0x6295_9d11,
    0xd45f,
    0x4884,
    [0xa0, 0xa3, 0xf7, 0x86, 0x2e, 0xa2, 0x43, 0xae],
);
/// `HRESULT_FROM_WIN32(ERROR_HANDLE_EOF)`: the source ends early.
const END_OF_FILE: HRESULT = 0x8007_0026_u32 as i32;
/// `HRESULT_FROM_WIN32(ERROR_INVALID_DATA)`: a bad compressed stream.
const INVALID_DATA: HRESULT = 0x8007_000D_u32 as i32;

type Out = *mut *mut c_void;

static FACTORY_OBJECT: Object = Object::new("dstorage factory", &FACTORY_VTABLE);

pub extern "system" fn DStorageGetFactory(iid: *const Guid, out: Out) -> HRESULT {
    // SAFETY: the title passes a GUID (null tolerated).
    match unsafe { iid.as_ref() } {
        Some(&iid) if iid == FACTORY => {
            put(out, std::ptr::from_ref(&FACTORY_OBJECT).cast_mut().cast())
        }
        iid => {
            tracing::warn!(iid = ?iid.map(Guid::to_string), "DStorageGetFactory: unknown interface");
            E_NOINTERFACE
        }
    }
}

/// One of our objects with state: the runtime's object header (method
/// table first), a reference count, then `T`.
#[repr(C)]
struct Instance<T> {
    base: Object,
    refs: AtomicU32,
    state: T,
}

impl<T> Instance<T> {
    fn create(base: Object, state: T) -> *mut c_void {
        Box::into_raw(Box::new(Instance {
            base,
            refs: AtomicU32::new(1),
            state,
        }))
        .cast()
    }
}

extern "system" fn add_ref_instance<T>(this: *const Instance<T>) -> u32 {
    // SAFETY: an instance we created and the title still holds.
    unsafe { &*this }.refs.fetch_add(1, Ordering::Relaxed) + 1
}

extern "system" fn release<T>(this: *mut Instance<T>) -> u32 {
    // SAFETY: as above.
    let left = unsafe { &*this }.refs.fetch_sub(1, Ordering::AcqRel) - 1;
    if left == 0 {
        // SAFETY: the last reference; created by `Instance::create`.
        drop(unsafe { Box::from_raw(this) });
    }
    left
}

fn iid_text(iid: *const Guid) -> String {
    // SAFETY: the title passes a GUID (null tolerated).
    unsafe { iid.as_ref() }.map_or_else(|| "null".into(), Guid::to_string)
}

/// `HRESULT_FROM_WIN32` of a host error.
fn from_win32(error: &std::io::Error) -> HRESULT {
    error
        .raw_os_error()
        .map_or(E_INVALIDARG, |e| (0x8007_0000 | (e as u32 & 0xffff)) as i32)
}

// The factory.

type Factory = *const Object;

/// Opens `path` for reading; the file object owns the handle.
extern "system" fn open_file(_: Factory, path: *const u16, iid: *const Guid, out: Out) -> HRESULT {
    const GENERIC_READ: u32 = 0x8000_0000;
    const FILE_SHARE_ALL: u32 = 1 | 2 | 4;
    const OPEN_EXISTING: u32 = 3;
    if path.is_null() {
        return E_INVALIDARG;
    }
    // Goes through `files`, so paths on our virtual drives work too.
    let handle = super::files::CreateFileW(
        path,
        GENERIC_READ,
        FILE_SHARE_ALL,
        std::ptr::null(),
        OPEN_EXISTING,
        0,
        std::ptr::null_mut(),
    );
    // SAFETY: the title passes a NUL-terminated path.
    let (name, wide_path) = unsafe {
        let len = (0..).take_while(|&i| *path.add(i) != 0).count();
        let slice = std::slice::from_raw_parts(path, len);
        let with_nul: Box<[u16]> = std::slice::from_raw_parts(path, len + 1).into();
        (String::from_utf16_lossy(slice), with_nul)
    };
    if handle as isize == -1 {
        let error = std::io::Error::last_os_error();
        tracing::warn!(path = name, %error, "dstorage: opening a file failed");
        return from_win32(&error);
    }
    tracing::info!(path = name, iid = iid_text(iid), "dstorage: file opened");
    // SAFETY: a handle we just opened and nobody else owns.
    let file = unsafe { std::fs::File::from_raw_handle(handle) };
    let file = Instance::create(
        Object::new("dstorage file", &FILE_VTABLE),
        File {
            files: Files::new_with_path(file, Some(wide_path), Io::chosen()),
            path: name,
        },
    );
    put(out, file)
}

/// `(capacity, name, iid, out)`.
extern "system" fn create_status_array(
    _: Factory,
    capacity: u32,
    name: *const c_char,
    iid: *const Guid,
    out: Out,
) -> HRESULT {
    tracing::info!(
        capacity,
        name = text(name).unwrap_or_default(),
        iid = iid_text(iid),
        "dstorage: status array"
    );
    let entries = (0..capacity).map(|_| AtomicI32::new(S_OK)).collect();
    let array = Instance::create(
        Object::new("dstorage status array", &STATUS_VTABLE),
        StatusArray { entries },
    );
    put(out, array)
}

/// The queue's description: as in 2020 (source type, capacity, priority,
/// name), plus a field Grand Theft Auto VI sets to 1.
#[repr(C)]
struct QueueDesc {
    source_type: u64,
    capacity: u16,
    priority: i8,
    name: *const c_char,
    unknown: u64,
}

extern "system" fn create_queue(
    _: Factory,
    desc: *const QueueDesc,
    iid: *const Guid,
    out: Out,
) -> HRESULT {
    // SAFETY: the title passes its description (null tolerated).
    let Some(desc) = (unsafe { desc.as_ref() }) else {
        return E_INVALIDARG;
    };
    let name = text(desc.name).unwrap_or_default();
    tracing::info!(
        source = desc.source_type,
        capacity = desc.capacity,
        priority = desc.priority,
        name,
        unknown = desc.unknown,
        iid = iid_text(iid),
        "dstorage: queue"
    );
    let Some(scheduler) = scheduler() else {
        return E_OUTOFMEMORY;
    };
    let mut lanes = scheduler.lock();
    let id = lanes.next_id;
    lanes.next_id += 1;
    let lane = Lane::new(desc.priority, desc.capacity);
    lanes.lanes.insert(id, lane);
    drop(lanes);
    put(
        out,
        Instance::create(Object::new("dstorage queue", &QUEUE_VTABLE), Queue { id }),
    )
}

/// Grand Theft Auto VI asks for 32 MiB and stops if this fails.
extern "system" fn set_staging_buffer_size(_: Factory, size: u32) -> HRESULT {
    tracing::info!(size, "dstorage: staging buffer size");
    if let Some(scheduler) = scheduler() {
        scheduler.lock().staging = u64::from(size);
    }
    S_OK
}

static FACTORY_VTABLE: Methods<16> = Methods([
    missing::<0> as *const (),
    add_ref as *const (),
    add_ref as *const (),
    missing::<3> as *const (),            // 0x18 (2020: queue creation)
    open_file as *const (),               // 0x20
    create_status_array as *const (),     // 0x28
    missing::<6> as *const (),            // 0x30 (2020: CPU affinity)
    missing::<7> as *const (),            // 0x38 (2020: debug flags)
    set_staging_buffer_size as *const (), // 0x40
    missing::<9> as *const (),
    create_queue as *const (), // 0x50
    missing::<11> as *const (),
    missing::<12> as *const (),
    missing::<13> as *const (),
    missing::<14> as *const (),
    missing::<15> as *const (),
]);

// Files.

struct File {
    files: Files,
    /// As the title named it (diagnostics).
    path: String,
}

/// How files are read (`VICTORY_DSTORAGE_IO`, `VICTORY_DSTORAGE_HANDLES`).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Io {
    /// `one` (or `VICTORY_DSTORAGE_HANDLES=0`): all workers read through the
    /// one handle the title opened, as before. The host serialises synchronous
    /// reads on a single file object.
    One,
    /// `handles` (default): each worker lazily opens its own synchronous
    /// handle with `ReOpenFile`, falling back to `first` if refused.
    Handles,
    /// `overlapped`: one reopened `FILE_FLAG_OVERLAPPED` handle shared across
    /// workers via per-thread completion events.
    Overlapped,
    /// `direct`: a handle per worker opened with `FILE_FLAG_OVERLAPPED |
    /// FILE_FLAG_NO_BUFFERING`, bypassing the host file cache.
    Direct,
}

impl Io {
    const DEFAULT: Io = Io::Handles;

    fn chosen() -> Io {
        static CHOSEN: OnceLock<Io> = OnceLock::new();
        *CHOSEN.get_or_init(|| {
            let io = match std::env::var("VICTORY_DSTORAGE_IO")
                .as_deref()
                .map(str::trim)
            {
                Ok("one") => Io::One,
                Ok("handles") => Io::Handles,
                Ok("overlapped") => Io::Overlapped,
                Ok("direct") => Io::Direct,
                _ if std::env::var_os("VICTORY_DSTORAGE_HANDLES").is_some_and(|v| v == "0") => {
                    Io::One
                }
                _ => Io::DEFAULT,
            };
            tracing::info!(?io, "dstorage: how files are read");
            io
        })
    }
}

/// Per-slot reopened file handle: synchronous handles carry an uncontended
/// per-slot `Mutex` so two threads can never race `seek_read` (`lseek` +
/// `read`) on the same synchronous handle under Wine, while `FILE_FLAG_OVERLAPPED`
/// handles issue true offset-addressed overlapped reads.
enum SlotFile {
    Sync(Mutex<std::fs::File>),
    Overlapped(std::fs::File),
}

/// A file as the workers read it: the handle the title opened, and handles
/// made from it when each worker first reads. Synchronous reads on one handle
/// serialize inside Windows' file-object lock (185 -> 990 MB/s across 24
/// workers) and race `lseek` + `read` on a shared cursor under macOS Wine
/// (`hr=0x8007000d` zlib corruption). Each DirectStorage pool worker `i`
/// indexes `others[i]` directly via [`WORKER_SLOT`] (zero modulo collisions,
/// zero lock contention), with `CreateFileW` fallback on `wide_path` if
/// `ReOpenFile` fails and mutex-serialized fallback to `first` if all reopen
/// paths fail or `Io::One` (`VICTORY_DSTORAGE_HANDLES=0`) is selected.
struct Files {
    first: Mutex<std::fs::File>,
    wide_path: Option<Box<[u16]>>,
    others: Box<[OnceLock<Option<SlotFile>>]>,
    io: Io,
}

impl Files {
    #[cfg(test)]
    fn new(first: std::fs::File, io: Io) -> Files {
        Files::new_with_path(first, None, io)
    }

    fn new_with_path(first: std::fs::File, wide_path: Option<Box<[u16]>>, io: Io) -> Files {
        let others = match io {
            Io::One => 0,
            Io::Overlapped => 1,
            Io::Handles | Io::Direct => workers(),
        };
        Files::with_path(first, wide_path, io, others)
    }

    #[cfg(test)]
    fn with(first: std::fs::File, io: Io, others: usize) -> Files {
        Files::with_path(first, None, io, others)
    }

    /// With `others` reopened handle slots besides `first` (0: `first` alone).
    fn with_path(
        first: std::fs::File,
        wide_path: Option<Box<[u16]>>,
        io: Io,
        others: usize,
    ) -> Files {
        Files {
            first: Mutex::new(first),
            wide_path,
            others: (0..others).map(|_| OnceLock::new()).collect(),
            io,
        }
    }

    /// Reads at `offset`, guaranteeing two threads never invoke `seek_read`
    /// concurrently on the same synchronous `std::fs::File`.
    fn read_at(&self, out: &mut [u8], offset: u64) -> std::io::Result<usize> {
        use windows_sys::Win32::Storage::FileSystem::{
            FILE_FLAG_NO_BUFFERING, FILE_FLAG_OVERLAPPED,
        };
        let flags = match self.io {
            Io::One | Io::Handles => 0,
            Io::Overlapped => FILE_FLAG_OVERLAPPED,
            Io::Direct => FILE_FLAG_OVERLAPPED | FILE_FLAG_NO_BUFFERING,
        };
        let other = match self.others.len() {
            0 => None,
            n => {
                let slot = WORKER_SLOT.get().unwrap_or_else(non_worker_slot) % n;
                self.others[slot]
                    .get_or_init(|| {
                        let raw = {
                            use std::os::windows::io::AsRawHandle;
                            let guard = self.first.lock().unwrap_or_else(|e| e.into_inner());
                            guard.as_raw_handle()
                        };
                        let file = reopen(raw, self.wide_path.as_deref(), flags)?;
                        Some(if flags & FILE_FLAG_OVERLAPPED != 0 {
                            SlotFile::Overlapped(file)
                        } else {
                            SlotFile::Sync(Mutex::new(file))
                        })
                    })
                    .as_ref()
            }
        };
        match (other, self.io) {
            (Some(SlotFile::Overlapped(file)), Io::Overlapped) => {
                read_overlapped(file, out, offset)
            }
            (Some(SlotFile::Overlapped(file)), Io::Direct) => read_direct(file, out, offset),
            (Some(SlotFile::Sync(file)), _) => {
                let guard = file.lock().unwrap_or_else(|e| e.into_inner());
                guard.seek_read(out, offset)
            }
            _ => {
                let guard = self.first.lock().unwrap_or_else(|e| e.into_inner());
                guard.seek_read(out, offset)
            }
        }
    }
}

/// Worker pool size (`VICTORY_DSTORAGE_WORKERS`, else logical CPU count).
fn workers() -> usize {
    std::env::var("VICTORY_DSTORAGE_WORKERS")
        .ok()
        .and_then(|v| v.trim().parse::<usize>().ok())
        .filter(|&n| (1..=256).contains(&n))
        .unwrap_or_else(|| std::thread::available_parallelism().map_or(4, |n| n.get()))
}

thread_local! {
    /// Exact `0..workers()` slot index for DirectStorage worker pool threads;
    /// `None` on title or test threads outside the pool.
    static WORKER_SLOT: std::cell::Cell<Option<usize>> = const { std::cell::Cell::new(None) };
}

/// Fallback slot counter for non-pool threads (e.g. unit tests calling `Files` directly).
fn non_worker_slot() -> usize {
    static NEXT: AtomicU32 = AtomicU32::new(0);
    thread_local! {
        static NUMBER: usize = NEXT.fetch_add(1, Ordering::Relaxed) as usize;
    }
    NUMBER.with(|n| *n)
}

/// Opens an independent handle for reading (`ReOpenFile`, falling back to
/// `super::files::CreateFileW` on `wide_path` when `ReOpenFile` fails or on
/// virtual drives if needed); returns `None` only if all attempts fail.
fn reopen(
    raw_handle: std::os::windows::io::RawHandle,
    wide_path: Option<&[u16]>,
    flags: u32,
) -> Option<std::fs::File> {
    use windows_sys::Win32::Foundation::{GENERIC_READ, INVALID_HANDLE_VALUE};
    use windows_sys::Win32::Storage::FileSystem::{FILE_SHARE_READ, OPEN_EXISTING, ReOpenFile};
    const FILE_SHARE_ALL: u32 = 1 | 2 | 4;
    static WARNED: AtomicBool = AtomicBool::new(false);

    // Under Wine, prefer opening a fresh OS file description via `CreateFileW`
    // whenever `wide_path` is known so two handles never share a `dup()`d Unix
    // `fileglob` file offset across concurrent `lseek` + `read` calls.
    if super::xmem::on_wine()
        && let Some(path) = wide_path
    {
        let reopened = super::files::CreateFileW(
            path.as_ptr(),
            GENERIC_READ,
            FILE_SHARE_ALL,
            std::ptr::null(),
            OPEN_EXISTING,
            flags,
            std::ptr::null_mut(),
        );
        if reopened != INVALID_HANDLE_VALUE {
            // SAFETY: freshly opened handle owned exclusively by this slot.
            return Some(unsafe { std::fs::File::from_raw_handle(reopened) });
        }
    }

    // SAFETY: `raw_handle` is kept open by `Files::first` for the duration of this call.
    let mut handle = unsafe { ReOpenFile(raw_handle, GENERIC_READ, FILE_SHARE_ALL, flags) };
    if handle == INVALID_HANDLE_VALUE {
        // SAFETY: retry with `FILE_SHARE_READ` if the file rejected wider sharing.
        handle = unsafe { ReOpenFile(raw_handle, GENERIC_READ, FILE_SHARE_READ, flags) };
    }
    if handle == INVALID_HANDLE_VALUE
        && let Some(path) = wide_path
    {
        handle = super::files::CreateFileW(
            path.as_ptr(),
            GENERIC_READ,
            FILE_SHARE_ALL,
            std::ptr::null(),
            OPEN_EXISTING,
            flags,
            std::ptr::null_mut(),
        );
    }
    if handle == INVALID_HANDLE_VALUE {
        let error = std::io::Error::last_os_error();
        if !WARNED.swap(true, Ordering::Relaxed) {
            tracing::warn!(%error, flags, "dstorage: reopen failed; falling back to mutex-guarded shared handle");
        } else {
            tracing::debug!(%error, flags, "dstorage: reopen failed; falling back to mutex-guarded shared handle");
        }
        return None;
    }
    // SAFETY: `handle` was freshly opened and belongs solely to this slot.
    Some(unsafe { std::fs::File::from_raw_handle(handle) })
}

/// A read through a handle opened with `FILE_FLAG_OVERLAPPED`, waited for on
/// a per-thread manual-reset event.
fn read_overlapped(file: &std::fs::File, out: &mut [u8], offset: u64) -> std::io::Result<usize> {
    use std::os::windows::io::AsRawHandle;
    use windows_sys::Win32::Foundation::{CloseHandle, ERROR_HANDLE_EOF, ERROR_IO_PENDING, HANDLE};
    use windows_sys::Win32::Storage::FileSystem::ReadFile;
    use windows_sys::Win32::System::IO::{GetOverlappedResult, OVERLAPPED};
    use windows_sys::Win32::System::Threading::CreateEventW;
    struct Event(HANDLE);
    impl Drop for Event {
        fn drop(&mut self) {
            if !self.0.is_null() {
                // SAFETY: `self.0` is a valid event handle created for this thread.
                unsafe { CloseHandle(self.0) };
            }
        }
    }
    thread_local! {
        static DONE: Event = {
            // SAFETY: creates an unnamed manual-reset event with default security.
            Event(unsafe { CreateEventW(std::ptr::null(), 1, 0, std::ptr::null()) })
        };
    }
    DONE.with(|done| {
        if done.0.is_null() {
            return Err(std::io::Error::last_os_error());
        }
        let mut overlapped = OVERLAPPED {
            hEvent: done.0,
            ..Default::default()
        };
        overlapped.Anonymous.Anonymous.Offset = offset as u32;
        overlapped.Anonymous.Anonymous.OffsetHigh = (offset >> 32) as u32;
        let len = out.len().min(1 << 30) as u32;
        let mut n = 0;
        // SAFETY: `out` and `overlapped` remain live on the stack until `GetOverlappedResult` completes.
        let mut read = unsafe {
            ReadFile(
                file.as_raw_handle(),
                out.as_mut_ptr(),
                len,
                std::ptr::null_mut(),
                &mut overlapped,
            )
        } != 0;
        let mut error = std::io::Error::last_os_error();
        if read || error.raw_os_error() == Some(ERROR_IO_PENDING as i32) {
            // SAFETY: waits synchronously on `overlapped` before `out` or `overlapped` leave scope.
            read =
                unsafe { GetOverlappedResult(file.as_raw_handle(), &overlapped, &mut n, 1) } != 0;
            error = std::io::Error::last_os_error();
        }
        match error.raw_os_error() {
            _ if read => Ok(n as usize),
            Some(code) if code == ERROR_HANDLE_EOF as i32 => Ok(0),
            _ => Err(error),
        }
    })
}

/// Unbuffered sector-aligned read (`FILE_FLAG_NO_BUFFERING`).
fn read_direct(file: &std::fs::File, out: &mut [u8], offset: u64) -> std::io::Result<usize> {
    const SECTOR: usize = 4096;
    const MOST: usize = 1 << 20;
    let sector = SECTOR as u64;
    if offset.is_multiple_of(sector)
        && out.len().is_multiple_of(SECTOR)
        && (out.as_ptr() as usize).is_multiple_of(SECTOR)
    {
        return read_overlapped(file, out, offset);
    }
    thread_local! {
        static BUFFER: std::cell::RefCell<Vec<u8>> = std::cell::RefCell::new(vec![0; MOST + SECTOR]);
    }
    BUFFER.with_borrow_mut(|buffer| {
        let skip = buffer.as_ptr().align_offset(SECTOR);
        let before = (offset % sector) as usize;
        let len = (before + out.len()).next_multiple_of(SECTOR).min(MOST);
        let aligned = &mut buffer[skip..skip + len];
        let read = read_overlapped(file, aligned, offset - before as u64)?;
        let n = read.saturating_sub(before).min(out.len());
        out[..n].copy_from_slice(&aligned[before..before + n]);
        Ok(n)
    })
}

static FILE_VTABLE: Methods<8> = Methods([
    missing::<0> as *const (),
    add_ref_instance::<File> as *const (),
    release::<File> as *const (),
    missing::<3> as *const (),
    missing::<4> as *const (),
    missing::<5> as *const (),
    missing::<6> as *const (),
    missing::<7> as *const (),
]);

// Status arrays.

struct StatusArray {
    entries: Box<[AtomicI32]>,
}

type Array = *mut Instance<StatusArray>;

/// The result of the requests before entry `index`: pending, the first
/// failure, or success.
extern "system" fn status_result(this: Array, index: u32) -> HRESULT {
    // SAFETY: an array we created and the title still holds.
    let entries = &unsafe { &*this }.state.entries;
    let result = entries
        .get(index as usize)
        .map_or(E_INVALIDARG, |e| e.load(Ordering::Acquire));
    if result == E_PENDING
        && let Some(scheduler) = scheduler()
    {
        let released = scheduler.lock().note_poll(index);
        if released > 0 {
            // The queue had held work; the workers must wake for it or the
            // title polls this entry forever.
            scheduler.work.notify_all();
        }
    }
    result
}

/// Grand Theft Auto VI polls this (entry 0, a `Sleep(1)` loop) and never
/// reads the result itself.
extern "system" fn is_complete(this: Array, index: u32) -> bool {
    status_result(this, index) != E_PENDING
}

static STATUS_VTABLE: Methods<5> = Methods([
    missing::<0> as *const (),
    add_ref_instance::<StatusArray> as *const (),
    release::<StatusArray> as *const (),
    is_complete as *const (),   // 0x18
    status_result as *const (), // 0x20
]);

// Queues.

/// A queue object: its lane in the scheduler.
struct Queue {
    id: u64,
}

impl Drop for Queue {
    fn drop(&mut self) {
        close_lane(self.id);
    }
}

type QueueRef = *const Instance<Queue>;

/// Every queue's requests, run by a pool of workers.
struct Scheduler {
    lanes: Mutex<Lanes>,
    /// Requests to start, or staging space freed.
    work: Condvar,
    /// A request finished (for full queues and `close`).
    finished: Condvar,
}

/// Priority weights per class (low, normal, high and real-time), as the
/// console's: about 10x per level, counted against request sizes.
const COST: [u64; 3] = [100, 10, 1];
/// The console's default staging buffer.
const STAGING: u64 = 32 << 20;

struct Lanes {
    lanes: HashMap<u64, Lane>,
    next_id: u64,
    /// Submission order over all queues.
    next_seq: u64,
    /// Bytes of staging held by running requests, and the limit.
    staged: u64,
    staging: u64,
    /// A start waited for staging space since the last finish.
    starved: bool,
    /// Weighted round robin: each class's virtual time (bytes × cost), and
    /// the time of the last start; within a class, the lane clock of the
    /// last start.
    clock: [u64; 3],
    now: u64,
    lane_now: [u64; 3],
    /// Since the last debug summary: requests, bytes written, time spent
    /// running them.
    done: (u64, u64, std::time::Duration),
    refused: u64,
    /// BCPack bytes written and time spent on them.
    bcpack: (u64, std::time::Duration),
    /// Requests into physical pages.
    pages: u64,
    since: Option<std::time::Instant>,
    /// When a request last started or finished, or a command was enqueued
    /// or submitted, and whether the queues being idle since then was said.
    moved: std::time::Instant,
    said_idle: Option<std::time::Instant>,

    /// Status entries the title polled while pending, and since when.
    polls: u64,
    polling_since: Option<std::time::Instant>,
    said_polling: bool,
}

/// A queue's requests in the scheduler.
struct Lane {
    class: usize,
    capacity: usize,
    /// Bytes started, for round robin among the lanes of a class.
    clock: u64,
    /// Enqueued, not yet submitted.
    unsubmitted: Vec<Command>,
    /// Left out when the queue submitted itself, until the title's Submit:
    /// reads into physical pages and the status entries behind them.
    held: Vec<Command>,
    /// Submitted, by sequence number: reads not started, status entries
    /// not written, reads running, and failures not yet reported.
    reads: VecDeque<(u64, Request)>,
    statuses: VecDeque<(u64, Status)>,
    running: BTreeSet<u64>,
    failures: BTreeMap<u64, HRESULT>,
    closed: bool,
    /// Since when reads into physical pages wait for the title's Submit.
    held_since: Option<std::time::Instant>,
    /// A batch of 64 requests or more waiting, since when: said once it
    /// finishes, so a load's streaming shows its ends.
    batch: Option<(usize, std::time::Instant)>,
}

// SAFETY: requests point into title memory and our own objects, which stay
// valid until the title has seen them complete; status arrays count their
// references atomically.
unsafe impl Send for Lane {}

/// A read a worker runs.
struct Job {
    lane: u64,
    seq: u64,
    request: Request,
    staging: u64,
}

// SAFETY: as for `Lane`.
unsafe impl Send for Job {}

/// The scheduler, with its workers started on first use; `None` if not
/// even one worker could start.
fn scheduler() -> Option<&'static Scheduler> {
    static SCHEDULER: OnceLock<Option<Scheduler>> = OnceLock::new();
    static STARTED: std::sync::Once = std::sync::Once::new();
    let scheduler = SCHEDULER
        .get_or_init(|| {
            Some(Scheduler {
                lanes: Mutex::new(Lanes {
                    lanes: HashMap::new(),
                    next_id: 0,
                    next_seq: 0,
                    staged: 0,
                    staging: STAGING,
                    starved: false,
                    clock: [0; 3],
                    now: 0,
                    lane_now: [0; 3],
                    done: Default::default(),
                    refused: 0,
                    bcpack: Default::default(),
                    pages: 0,
                    since: None,
                    moved: std::time::Instant::now(),
                    said_idle: None,
                    polls: 0,
                    polling_since: None,
                    said_polling: false,
                }),
                work: Condvar::new(),
                finished: Condvar::new(),
            })
        })
        .as_ref()?;
    STARTED.call_once(|| {
        // `VICTORY_DSTORAGE_WORKERS`: more reads in flight for storage
        // with a long wait per read (a network share), else one per core.
        let started = (0..workers())
            .filter(|&i| {
                std::thread::Builder::new()
                    .name(format!("dstorage {i}"))
                    .spawn(move || {
                        WORKER_SLOT.set(Some(i));
                        work(scheduler);
                    })
                    .is_ok()
            })
            .count();
        tracing::info!(workers = started, "dstorage: workers");
        // A queue the title waits on and nothing fills is invisible
        // otherwise: requests only log when they finish.
        let _ = std::thread::Builder::new()
            .name("dstorage watch".into())
            .spawn(|| watch(scheduler));
    });
    Some(scheduler)
}

/// Says what the queues hold when nothing has started or finished for a
/// while (a wait nothing ends).
fn watch(scheduler: &'static Scheduler) {
    loop {
        std::thread::sleep(std::time::Duration::from_secs(1));
        let mut lanes = scheduler.lock();
        let released = lanes.release_held(hold_delay());
        lanes.report_stuck();
        drop(lanes);
        // A Submit that never came: the reads started anyway, so the
        // workers must be woken for them.
        if released > 0 {
            scheduler.work.notify_all();
        }
    }
}

impl Scheduler {
    fn lock(&self) -> MutexGuard<'_, Lanes> {
        self.lanes.lock().unwrap_or_else(|e| e.into_inner())
    }
}

/// How long the queues may hold work with nothing starting or finishing
/// before the watchdog says what they hold.
const STUCK: std::time::Duration = std::time::Duration::from_secs(5);
/// How long the title may poll a pending status entry before that is said.
const POLLING: std::time::Duration = std::time::Duration::from_secs(2);

impl Lanes {
    /// A request started or finished, or a command was enqueued or
    /// submitted: the queues are moving.
    fn moved(&mut self) {
        self.moved = std::time::Instant::now();
        self.said_idle = None;
        self.polls = 0;
        self.polling_since = None;
        self.said_polling = false;
    }

    /// The title asked for a status entry still pending. Grand Theft Auto VI
    /// polls entry 0 in a `Sleep(1)` loop and never reads the result, so a
    /// poll that goes on means the load waits for these reads.
    /// The title asked for a status entry still pending. Grand Theft Auto VI
    /// polls entry 0 in a `Sleep(1)` loop and never reads the result, so a
    /// poll that goes on means the load waits for these reads. Returns how
    /// many queues' work it released (the caller wakes the workers).
    fn note_poll(&mut self, index: u32) -> usize {
        self.polls += 1;
        let since = *self
            .polling_since
            .get_or_insert_with(std::time::Instant::now);
        let held = since.elapsed();
        if held < POLLING || self.said_polling {
            return 0;
        }
        self.said_polling = true;
        tracing::info!(
            index,
            polls = self.polls,
            seconds = format_args!("{:.1}", held.as_secs_f64()),
            "dstorage: the title polls a status entry that is still pending"
        );
        self.say_lanes("a queue holds what the title polls for");
        // The title is waiting on this entry: whatever reads stand in front
        // of it must run, or nothing after them ever will.
        self.release_held(std::time::Duration::ZERO)
    }

    /// Says what the queues hold when nothing has moved for [`STUCK`], every
    /// 30 s while it lasts.
    fn report_stuck(&mut self) {
        if !self.lanes.values().any(|l| l.entries() > 0) {
            self.said_idle = None;
            return;
        }
        if self.moved.elapsed() < STUCK {
            return;
        }
        if self
            .said_idle
            .is_some_and(|at| at.elapsed() < std::time::Duration::from_secs(30))
        {
            return;
        }
        self.said_idle = Some(std::time::Instant::now());
        self.say_lanes("no request started or finished");
    }

    /// One info line per queue with work left, with the ages that say what
    /// it waits for: `why` names the wait.
    fn say_lanes(&self, why: &str) {
        let idle = self.moved.elapsed().as_secs_f64() * 1e3;
        for (&id, lane) in &self.lanes {
            if lane.entries() == 0 {
                continue;
            }
            let held_ms = lane
                .held_since
                .map_or(0.0, |t| t.elapsed().as_secs_f64() * 1e3);
            let batch = lane.batch.map(|(n, _)| n).unwrap_or(0);
            tracing::info!(
                queue = id,
                class = lane.class,
                capacity = lane.capacity,
                unsubmitted = lane.unsubmitted.len(),
                held = lane.held.len(),
                reads = lane.reads.len(),
                statuses = lane.statuses.len(),
                running = lane.running.len(),
                failures = lane.failures.len(),
                batch,
                held_ms,
                staged_mb = format_args!("{:.1}", self.staged as f64 / 1e6),
                idle_ms = format_args!("{:.0}", idle),
                "dstorage: {why}"
            );
        }
    }
}

/// A worker: runs the next read of the most deserving queue.
fn work(scheduler: &'static Scheduler) {
    let mut lanes = scheduler.lock();
    loop {
        let Some(job) = lanes.start() else {
            lanes = scheduler
                .work
                .wait(lanes)
                .unwrap_or_else(|e| e.into_inner());
            continue;
        };
        drop(lanes);
        let started = std::time::Instant::now();
        let result = execute(&job.request);
        lanes = scheduler.lock();
        lanes.finish(&job, result);
        lanes.count(&job, started);
        if std::mem::take(&mut lanes.starved) {
            scheduler.work.notify_all();
        }
        scheduler.finished.notify_all();
    }
}

impl Lanes {
    /// Counts a finished request; every 5 s a debug line sums them up.
    fn count(&mut self, job: &Job, started: std::time::Instant) {
        if super::read_stats::on() {
            super::read_stats::add(
                source_name(&job.request),
                u64::from(job.request.source_size),
            );
        }
        let (requests, bytes, busy) = &mut self.done;
        *requests += 1;
        *bytes += u64::from(job.request.destination_size);
        *busy += started.elapsed();
        if job.request.options & BCPACK != 0 {
            self.bcpack.0 += u64::from(job.request.destination_size);
            self.bcpack.1 += started.elapsed();
        }
        if job.request.options & PAGE_DESTINATION != 0 {
            self.pages += 1;
        }
        let since = *self.since.get_or_insert(started);
        let elapsed = since.elapsed();
        if elapsed.as_secs() >= 5 {
            let waiting: usize = self.lanes.values().map(|l| l.reads.len()).sum();
            let reading = READING.swap(0, Ordering::Relaxed) as f64 / 1e9;
            let inflating = INFLATING.swap(0, Ordering::Relaxed) as f64 / 1e9;
            let inflated = INFLATED.swap(0, Ordering::Relaxed);
            tracing::debug!(
                requests = *requests,
                mb_per_s = format_args!("{:.1}", *bytes as f64 / elapsed.as_secs_f64() / 1e6),
                busy_workers = format_args!("{:.2}", busy.as_secs_f64() / elapsed.as_secs_f64()),
                reading_workers = format_args!("{:.2}", reading / elapsed.as_secs_f64()),
                staging_waits = std::mem::take(&mut self.refused),
                bcpack_mb = self.bcpack.0 / 1_000_000,
                bcpack_workers =
                    format_args!("{:.2}", self.bcpack.1.as_secs_f64() / elapsed.as_secs_f64()),
                zlib_mb = inflated / 1_000_000,
                zlib_workers = format_args!("{:.2}", inflating / elapsed.as_secs_f64()),
                page_destinations = std::mem::take(&mut self.pages),
                waiting,
                "dstorage: last {:.1} s",
                elapsed.as_secs_f64()
            );
            self.done = Default::default();
            self.bcpack = Default::default();
            self.since = None;
        }
    }

    /// Starts the next read of the class with the least virtual time, from
    /// its lane that has started the fewest bytes, if the staging buffer
    /// has room for it (or nothing else holds any).
    fn start(&mut self) -> Option<Job> {
        let busy = || self.lanes.iter().filter(|(_, l)| !l.reads.is_empty());
        let (_, class) = busy().map(|(_, l)| (self.clock[l.class], l.class)).min()?;
        let (_, _, id) = busy()
            .filter(|(_, l)| l.class == class)
            .filter_map(|(&id, l)| Some((l.clock, l.reads.front()?.0, id)))
            .min()?;
        let lane = self.lanes.get_mut(&id)?;
        let &(seq, request) = lane.reads.front()?;
        let staging = staging_bytes(&request);
        if self.staged > 0 && self.staged + staging > self.staging {
            self.starved = true;
            self.refused += 1;
            return None;
        }
        lane.reads.pop_front();
        lane.running.insert(seq);
        self.staged += staging;
        let size = u64::from(request.destination_size.max(request.source_size)).max(1);
        self.now = self.clock[class];
        self.clock[class] += size * COST[class];
        self.lane_now[class] = lane.clock;
        lane.clock += size;
        self.moved();
        Some(Job {
            lane: id,
            seq,
            request,
            staging,
        })
    }

    fn finish(&mut self, job: &Job, result: Result<(), HRESULT>) {
        self.staged -= job.staging;
        if let Some(lane) = self.lanes.get_mut(&job.lane) {
            lane.running.remove(&job.seq);
            if let Err(hr) = result {
                lane.failures.insert(job.seq, hr);
            }
            lane.flush();
            if lane.entries() == 0
                && let Some((requests, at)) = lane.batch.take()
            {
                tracing::info!(
                    requests,
                    seconds = format_args!("{:.1}", at.elapsed().as_secs_f64()),
                    "dstorage: a batch of reads finished"
                );
            }
        }
        self.moved();
    }

    /// Moves a queue's enqueued commands to the workers (the title's
    /// Submit).
    fn submit(&mut self, id: u64) {
        self.submit_as(id, By::Title);
    }

    /// Starts the reads a queue has held for the title's Submit for longer
    /// than `older`: a Submit that never comes is a load that never
    /// finishes, and the console's hardware would have run them already.
    /// Returns how many queues it released; the caller must wake the
    /// workers with [`Scheduler::work`] when it is more than zero.
    fn release_held(&mut self, older: std::time::Duration) -> usize {
        let ids: Vec<u64> = self
            .lanes
            .iter()
            .filter(|(_, l)| {
                !l.held.is_empty() && l.held_since.is_some_and(|t| t.elapsed() >= older)
            })
            .map(|(&id, _)| id)
            .collect();
        let released = ids.len();
        for id in ids {
            self.submit_as(id, By::Late);
        }
        released
    }

    /// As [`Lanes::submit`]; [`By::Auto`] when a half-full queue submits
    /// itself, which leaves reads into physical pages for the title's
    /// Submit: their page lists point into the title's page table, which
    /// Grand Theft Auto VI fills for the batch in its own submit path, right
    /// before Submit (`sub_1400e88f0`), so until then they name no page or a
    /// page that now holds something else. A status entry behind a held
    /// read waits with it, so a Submit that never comes holds the status the
    /// title polls. [`By::Late`] starts them anyway once
    /// [`VICTORY_DSTORAGE_HOLD_MS`] has passed; a list the title has not
    /// filled yet is read again there (`into_destination`).
    fn submit_as(&mut self, id: u64, by: By) {
        let Some(class) = self.lanes.get(&id).map(|l| l.class) else {
            return;
        };
        // A class that had nothing waiting starts from now, not from the
        // credit it left behind.
        if !self
            .lanes
            .values()
            .any(|l| l.class == class && !l.reads.is_empty())
        {
            self.clock[class] = self.clock[class].max(self.now);
        }
        let Some(lane) = self.lanes.get_mut(&id) else {
            return;
        };
        if lane.reads.is_empty() {
            lane.clock = lane.clock.max(self.lane_now[class]);
        }
        let explicit = by != By::Auto;
        let mut commands = if explicit {
            std::mem::take(&mut lane.held)
        } else {
            Vec::new()
        };
        if !commands.is_empty() {
            let held = commands.len();
            let waited = lane
                .held_since
                .take()
                .map_or(0.0, |t| t.elapsed().as_secs_f64() * 1e3);
            tracing::info!(
                held,
                waited_ms = waited,
                by = by.name(),
                "dstorage: reads held for a Submit start"
            );
        }
        commands.append(&mut lane.unsubmitted);
        let hold_page_reads = by == By::Auto && hold_delay() > std::time::Duration::ZERO;
        for command in commands {
            let hold = hold_page_reads
                && match &command {
                    Command::Read(r) => r.options & PAGE_DESTINATION != 0,
                    Command::Status(_) => !lane.held.is_empty(),
                };
            if hold {
                lane.held_since.get_or_insert_with(std::time::Instant::now);
                lane.held.push(command);
                continue;
            }
            let seq = self.next_seq;
            self.next_seq += 1;
            match command {
                Command::Read(request) => lane.reads.push_back((seq, request)),
                Command::Status(status) => lane.statuses.push_back((seq, status)),
            }
        }
        lane.flush();
        lane.note_batch();
        self.moved();
    }
}

impl Lane {
    fn new(priority: i8, capacity: u16) -> Lane {
        Lane {
            class: priority.clamp(-1, 1).wrapping_add(1) as usize,
            capacity: usize::from(capacity),
            clock: 0,
            unsubmitted: Vec::new(),
            held: Vec::new(),
            reads: VecDeque::new(),
            statuses: VecDeque::new(),
            running: BTreeSet::new(),
            failures: BTreeMap::new(),
            closed: false,
            held_since: None,
            batch: None,
        }
    }

    /// Requests and status entries held, submitted or not.
    fn entries(&self) -> usize {
        self.unsubmitted.len()
            + self.held.len()
            + self.reads.len()
            + self.statuses.len()
            + self.running.len()
    }

    /// Remembers a batch of 64 entries or more waiting, to say when it ends.
    fn note_batch(&mut self) {
        let entries = self.entries();
        match &mut self.batch {
            Some(batch) => batch.0 = batch.0.max(entries),
            None if entries >= 64 => self.batch = Some((entries, std::time::Instant::now())),
            None => {}
        }
    }

    /// Writes the status entries whose earlier reads have all finished.
    fn flush(&mut self) {
        while let Some(&(seq, _)) = self.statuses.front() {
            let waiting = self.reads.front().map(|r| r.0);
            if waiting
                .into_iter()
                .chain(self.running.first().copied())
                .any(|s| s < seq)
            {
                return;
            }
            let Some((_, status)) = self.statuses.pop_front() else {
                return;
            };
            let later = self.failures.split_off(&seq);
            let first = std::mem::replace(&mut self.failures, later)
                .into_values()
                .next();
            status
                .entry()
                .store(first.unwrap_or(S_OK), Ordering::Release);
        }
    }
}

/// Staging bytes a request holds while it runs: what it buffers between
/// the file and the destination. Raw reads go straight to their
/// destination, memory sources need none (as on the console).
fn staging_bytes(request: &Request) -> u64 {
    let buffered = request.options & (ZLIB | BCPACK | PAGE_DESTINATION) != 0;
    if buffered && request.options & MEMORY_SOURCE == 0 {
        u64::from(request.source_size)
    } else {
        0
    }
}

/// Ends a queue: what was not started is dropped, and this waits for the
/// reads already running (they write title memory).
fn close_lane(id: u64) {
    let Some(scheduler) = scheduler() else {
        return;
    };
    let mut lanes = scheduler.lock();
    let Some(lane) = lanes.lanes.get_mut(&id) else {
        return;
    };
    lane.closed = true;
    lane.unsubmitted.clear();
    lane.held.clear();
    lane.reads.clear();
    lane.statuses.clear(); // releases their arrays
    while lanes.lanes.get(&id).is_some_and(|l| !l.running.is_empty()) {
        lanes = scheduler
            .finished
            .wait(lanes)
            .unwrap_or_else(|e| e.into_inner());
    }
    lanes.lanes.remove(&id);
    scheduler.finished.notify_all();
}

/// A read request as the title builds it, 0x48 bytes.
#[repr(C)]
#[derive(Clone, Copy)]
struct Request {
    options: u64,
    /// An address, or with `PAGE_DESTINATION` an array of 64 KiB page
    /// numbers.
    destination: *mut c_void,
    /// With `PAGE_DESTINATION`, the low 16 bits are where in the first page
    /// the destination starts.
    page_offset: u64,
    destination_size: u32,
    /// Our file object, or with `MEMORY_SOURCE` an address.
    source: *const c_void,
    offset: u64,
    source_size: u32,
    /// Compressed size between two decompression stages; unused here.
    intermediate_size: u32,
    /// Matched by cancellation; Grand Theft Auto VI leaves it 0.
    tag: u64,
    name: *const c_char,
}

// Request options.
/// Inflate zlib (RFC 1950) data.
const ZLIB: u64 = 1;
/// Bits 1-4: decode BCPack data of this BC format number (0: none).
const BCPACK: u64 = 0xf << 1;
const PAGE_DESTINATION: u64 = 1 << 21;
const MEMORY_SOURCE: u64 = 1 << 22;
// Bits 5-20 select a further transform, bit 23 a source in physical pages;
// Grand Theft Auto VI's packages need neither, so requests with them fail.

const PAGE: usize = 64 * 1024;

enum Command {
    Read(Request),
    Status(Status),
}

/// What moved a queue's commands to the workers.
#[derive(Clone, Copy, PartialEq, Eq)]
enum By {
    /// The title's `Submit`.
    Title,
    /// The queue submitting itself once half full.
    Auto,
    /// Held for a Submit that did not come; started once the hold passed.
    Late,
}

impl By {
    fn name(self) -> &'static str {
        match self {
            Self::Title => "the title's Submit",
            Self::Auto => "the queue itself",
            Self::Late => "the hold's delay",
        }
    }
}

/// How long reads into physical pages wait for the title's Submit before
/// they start anyway. `VICTORY_DSTORAGE_HOLD_MS`, 0 for as the queue
/// submits itself.
fn hold_delay() -> std::time::Duration {
    static DELAY: OnceLock<std::time::Duration> = OnceLock::new();
    *DELAY.get_or_init(|| {
        std::env::var("VICTORY_DSTORAGE_HOLD_MS")
            .ok()
            .and_then(|v| v.trim().parse::<u64>().ok())
            .map_or(
                std::time::Duration::from_secs(2),
                std::time::Duration::from_millis,
            )
    })
}

/// A status entry to write, holding a reference on its array.
struct Status {
    array: Array,
    index: u32,
}

impl Drop for Status {
    fn drop(&mut self) {
        release(self.array);
    }
}

impl Status {
    fn entry(&self) -> &AtomicI32 {
        // SAFETY: we hold a reference; the index was checked when enqueued.
        &unsafe { &*self.array }.state.entries[self.index as usize]
    }
}

extern "system" fn enqueue_request(this: QueueRef, request: *const Request) {
    // SAFETY: the title passes its request (null tolerated).
    let Some(&request) = (unsafe { request.as_ref() }) else {
        return;
    };
    queue(this, Command::Read(request));
}

extern "system" fn enqueue_status(this: QueueRef, array: Array, index: u32) {
    // SAFETY: the title passes an array we created (null tolerated).
    let Some(entries) = (unsafe { array.as_ref() }).map(|a| &a.state.entries) else {
        return;
    };
    let Some(entry) = entries.get(index as usize) else {
        tracing::warn!(index, "dstorage: status entry out of range");
        return;
    };
    entry.store(E_PENDING, Ordering::Release);
    add_ref_instance(array);
    queue(this, Command::Status(Status { array, index }));
}

/// Adds a command to the queue; blocks while the queue is full, and
/// submits on its own once half of it is enqueued, as the console does.
fn queue(this: QueueRef, command: Command) {
    // SAFETY: a queue we created and the title still holds.
    let id = unsafe { &*this }.state.id;
    let Some(scheduler) = scheduler() else {
        return;
    };
    let mut lanes = scheduler.lock();
    loop {
        let Some(lane) = lanes.lanes.get(&id).filter(|l| !l.closed) else {
            return; // closed: dropped (a status releases its array)
        };
        // One slot stays free, as the console's count of empty slots says.
        if lane.entries() + 1 < lane.capacity.max(2) {
            break;
        }
        // Full with reads held for a Submit the title cannot make while it
        // waits here: they go now, page lists as they are.
        if !lane.held.is_empty() {
            static WARNED: AtomicBool = AtomicBool::new(false);
            if !WARNED.swap(true, Ordering::Relaxed) {
                tracing::warn!(
                    "dstorage: a full queue's reads into physical pages start before the title's Submit"
                );
            }
            lanes.submit(id);
            scheduler.work.notify_all();
        }
        lanes = scheduler
            .finished
            .wait(lanes)
            .unwrap_or_else(|e| e.into_inner());
    }
    let Some(lane) = lanes.lanes.get_mut(&id) else {
        return;
    };
    lane.unsubmitted.push(command);
    lane.note_batch();
    if lane.unsubmitted.len() >= (lane.capacity / 2).max(1) {
        let capacity = lane.capacity;
        lanes.submit_as(id, By::Auto);
        scheduler.work.notify_all();
        let held = lanes.lanes.get(&id).map_or(0, |l| l.held.len());
        let n = AUTO_SUBMITS.fetch_add(1, Ordering::Relaxed) + 1;
        if n.is_power_of_two() {
            tracing::info!(
                n,
                capacity,
                held,
                "dstorage: a queue half full submitted itself; reads into physical pages wait for the title's Submit"
            );
        }
    }
    // Enqueuing is progress too: a batch of 2,000 takes a while to build.
    lanes.moved();
}

extern "system" fn submit(this: QueueRef) {
    // SAFETY: as above.
    let id = unsafe { &*this }.state.id;
    let Some(scheduler) = scheduler() else {
        return;
    };
    let mut lanes = scheduler.lock();
    let pending = lanes
        .lanes
        .get(&id)
        .map_or(0, |l| l.unsubmitted.len() + l.held.len());
    tracing::debug!(commands = pending, "dstorage: submit");
    lanes.submit(id);
    scheduler.work.notify_all();
}

/// Ends the queue whatever its reference count: what was not started never
/// completes, and this returns once the reads already running have
/// finished. Grand Theft Auto VI calls this, not `Release`.
extern "system" fn close(this: QueueRef) {
    // SAFETY: as above.
    close_lane(unsafe { &*this }.state.id);
}

/// Reads, inflates and/or unpacks, and writes one request.
fn execute(request: &Request) -> Result<(), HRESULT> {
    let options = request.options;
    let fail = |hr: HRESULT, why: &str| {
        tracing::warn!(
            options = format_args!("{options:#x}"),
            offset = request.offset,
            source_size = request.source_size,
            destination_size = request.destination_size,
            intermediate_size = request.intermediate_size,
            tag = request.tag,
            name = text(request.name).unwrap_or_default(),
            hr = format_args!("{:#010x}", hr as u32),
            "dstorage: request failed: {why}"
        );
        hr
    };
    if options & !(ZLIB | BCPACK | PAGE_DESTINATION | MEMORY_SOURCE) != 0 {
        return Err(fail(E_NOTIMPL, "unsupported options"));
    }
    if request.source.is_null() || request.destination.is_null() {
        return Err(fail(E_INVALIDARG, "no source or destination"));
    }
    let bc = ((options & BCPACK) >> 1) as u8;
    let size = request.destination_size as usize;
    if options & (ZLIB | BCPACK) == 0 {
        if request.source_size > request.destination_size {
            return Err(fail(E_INVALIDARG, "source larger than destination"));
        }
        let size = request.source_size as usize;
        return into_destination(request, size, |out| read_into(request, out).map(|()| size))
            .map_err(|hr| fail(hr, "copying"));
    }
    let tables = if bc != 0 {
        Some(bcpack_tables()?)
    } else {
        None
    };
    let source = read_source(request).map_err(|hr| fail(hr, "reading the source"))?;
    // The console's order: zlib first, then BCPack.
    let source = if bc != 0 && options & ZLIB != 0 {
        let mut packed = vec![0; request.intermediate_size as usize];
        let n = inflate(&source, &mut packed).map_err(|hr| fail(hr, "zlib"))?;
        packed.truncate(n);
        Cow::Owned(packed)
    } else {
        source
    };
    into_destination(request, size, |out| match tables {
        Some(tables) => bcpack::decode(bc, &source, out, tables)
            .map(|()| out.len())
            .map_err(|e| fail(INVALID_DATA, &e.to_string())),
        None => inflate(&source, out).map_err(|hr| fail(hr, "zlib")),
    })?;
    tracing::trace!(
        options = format_args!("{options:#x}"),
        offset = request.offset,
        size,
        "dstorage: request done"
    );
    Ok(())
}

/// Inflates a zlib stream into `out`, returning the bytes written. A stream
/// that ends short of `out` passes: the title's chunks fill it exactly, and
/// nothing says that is required.
fn inflate(source: &[u8], out: &mut [u8]) -> Result<usize, HRESULT> {
    let started = std::time::Instant::now();
    let inflated = miniz_oxide::inflate::decompress_slice_iter_to_slice(
        out,
        std::iter::once(source),
        true,
        false,
    )
    .map_err(|status| {
        tracing::debug!(?status, "dstorage: zlib stream");
        INVALID_DATA
    });
    INFLATING.fetch_add(started.elapsed().as_nanos() as u64, Ordering::Relaxed);
    if let Ok(n) = inflated {
        INFLATED.fetch_add(n as u64, Ordering::Relaxed);
    }
    inflated
}

/// Nanoseconds the workers spent inflating zlib streams, and the bytes they
/// inflated, since the last debug summary: whether inflating is a part of
/// streaming worth making faster (under Rosetta `miniz_oxide` inflates about
/// 380 MB/s a worker on an M5).
static INFLATING: AtomicU64 = AtomicU64::new(0);
static INFLATED: AtomicU64 = AtomicU64::new(0);

/// The BCPack format's constant tables, parsed once for the process. If
/// they do not parse every BCPack request fails: one warning, then debug.
fn bcpack_tables() -> Result<&'static Tables, HRESULT> {
    static TABLES: OnceLock<Result<Tables, String>> = OnceLock::new();
    static WARNED: AtomicBool = AtomicBool::new(false);
    match TABLES.get_or_init(load_bcpack_tables) {
        Ok(tables) => Ok(tables),
        Err(why) if !WARNED.swap(true, Ordering::Relaxed) => {
            tracing::warn!("dstorage: BCPack requests fail: {why} (more at debug)");
            Err(E_NOTIMPL)
        }
        Err(_) => {
            tracing::debug!("dstorage: BCPack request failed: no tables");
            Err(E_NOTIMPL)
        }
    }
}

/// The tables VIctory ships with.
#[cfg(not(test))]
fn load_bcpack_tables() -> Result<Tables, String> {
    Tables::builtin().map_err(|e| format!("the built-in BCPack tables: {e}"))
}

/// Tests decode streams made against tables of their own, from this folder.
#[cfg(test)]
static BCPACK_DIR: OnceLock<std::path::PathBuf> = OnceLock::new();

#[cfg(test)]
fn use_bcpack_tables(dir: std::path::PathBuf) {
    let _ = BCPACK_DIR.set(dir);
}

#[cfg(test)]
fn load_bcpack_tables() -> Result<Tables, String> {
    let dir = BCPACK_DIR
        .get()
        .ok_or("no folder set for the test tables")?;
    let bytes = std::fs::read(dir.join("tables.bin")).map_err(|e| e.to_string())?;
    Tables::parse(&bytes).map_err(|e| e.to_string())
}

/// A compressed request's source: borrowed from memory, or read from the
/// file.
fn read_source(request: &Request) -> Result<Cow<'static, [u8]>, HRESULT> {
    let size = request.source_size as usize;
    if request.options & MEMORY_SOURCE != 0 {
        // SAFETY: the title's source holds `size` bytes until completion.
        let bytes = unsafe { std::slice::from_raw_parts(request.source.cast(), size) };
        return Ok(Cow::Borrowed(bytes));
    }
    let mut bytes = vec![0; size];
    read_into(request, &mut bytes)?;
    Ok(Cow::Owned(bytes))
}

/// Nanoseconds workers spent reading files, for the debug summary.
static READING: AtomicU64 = AtomicU64::new(0);

/// Fills `out` (the source size) from the request's file or memory.
fn read_into(request: &Request, out: &mut [u8]) -> Result<(), HRESULT> {
    if request.options & MEMORY_SOURCE != 0 {
        // SAFETY: the title's source holds `out.len()` bytes until completion.
        out.copy_from_slice(unsafe {
            std::slice::from_raw_parts(request.source.cast(), out.len())
        });
        return Ok(());
    }
    // SAFETY: a file object we created; the title keeps it open.
    let files = &unsafe { &*request.source.cast::<Instance<File>>() }
        .state
        .files;
    let started = std::time::Instant::now();
    let mut done = 0;
    let result = loop {
        if done == out.len() {
            break Ok(());
        }
        match files.read_at(&mut out[done..], request.offset + done as u64) {
            Ok(0) => break Err(END_OF_FILE),
            Ok(n) => done += n,
            Err(error) => break Err(from_win32(&error)),
        }
    };
    READING.fetch_add(started.elapsed().as_nanos() as u64, Ordering::Relaxed);
    result
}

/// Runs `fill` on the destination's first `len` bytes; `fill` returns how
/// many it produced. GPU-visible destinations are host memory like the
/// rest (GPU address = CPU address), so an address is written in place;
/// physical pages are too when they are consecutive pages of one
/// allocation, and otherwise go through a buffer.
fn into_destination(
    request: &Request,
    len: usize,
    fill: impl FnOnce(&mut [u8]) -> Result<usize, HRESULT>,
) -> Result<(), HRESULT> {
    if request.options & PAGE_DESTINATION == 0 {
        // Raw reads land through the kernel, which does not fault into the
        // GPU side's guard on title buffers; open them first.
        let _writing = victory_gpu::kernel_will_write(request.destination as usize, len);
        // SAFETY: the title's destination holds `destination_size` bytes, at
        // least `len`, and nothing else writes it until completion.
        let out = unsafe { std::slice::from_raw_parts_mut(request.destination.cast::<u8>(), len) };
        return fill(out).map(drop);
    }
    let offset = (request.page_offset & 0xffff) as usize;
    let count = (offset + len).div_ceil(PAGE);
    // The title's page list, copied: the title may still be filling it
    // (below), so each attempt is judged by the list it wrote through, not
    // by a later read of it.
    let list = || {
        // SAFETY: the title passes a page number per 64 KiB of destination,
        // which holds at least `len` bytes from `offset`.
        unsafe { std::slice::from_raw_parts(request.destination.cast::<usize>(), count) }.to_vec()
    };
    // Consecutive pages of one allocation (the usual case): written in
    // place through one view.
    if let Some(mut window) = super::xmem::page_window(&list(), offset, len) {
        return fill(window.bytes()).map(drop);
    }
    let mut data = vec![0; len];
    let n = fill(&mut data)?;
    let mut pages = list();
    if super::xmem::write_physical(&pages, offset, &data[..n]) {
        return Ok(());
    }
    // Page number 0 is no page (numbers start at 0x10_0000): a list naming
    // it is one the title has not filled yet. The title writes a request's
    // page list after enqueueing it, and a queue half full is submitted on
    // its own, so a read can start first. Read the list again for a few
    // milliseconds before failing: the title never looks at a request's
    // result, so a failed one is a model or texture left unwritten.
    let mut rereads = 0;
    while pages.contains(&0) && rereads < LATE_LIST_TRIES {
        rereads += 1;
        std::thread::sleep(std::time::Duration::from_millis(1));
        pages = list();
        if !pages.contains(&0) && super::xmem::write_physical(&pages, offset, &data[..n]) {
            let late = LATE_LISTS.fetch_add(1, Ordering::Relaxed) + 1;
            if late.is_power_of_two() {
                tracing::info!(
                    late,
                    rereads,
                    "dstorage: a page list filled after its request started: written late"
                );
            }
            return Ok(());
        }
    }
    // `still_empty`: the list still named page 0 after the last read;
    // else the pages it names could not be written.
    tracing::warn!(
        offset,
        len,
        pages = pages.len(),
        still_empty = pages.contains(&0),
        rereads,
        unknown = super::xmem::unknown_pages(&pages),
        list = format_args!("{:x?}", &pages[..pages.len().min(8)]),
        options = format_args!("{:#x}", request.options),
        file = source_name(request),
        file_offset = request.offset,
        "dstorage: a destination's physical pages could not be written; its request fails"
    );
    Err(E_INVALIDARG)
}

/// The file a request reads, as the title named it, or "memory".
fn source_name(request: &Request) -> &str {
    if request.options & MEMORY_SOURCE != 0 {
        return "memory";
    }
    // SAFETY: a file object we created; the title keeps it open.
    &unsafe { &*request.source.cast::<Instance<File>>() }
        .state
        .path
}

/// Queues that submitted themselves, half full (diagnostics).
static AUTO_SUBMITS: AtomicU32 = AtomicU32::new(0);

/// Milliseconds a page list naming page 0 is read again for.
const LATE_LIST_TRIES: u32 = 8;
/// Page lists filled after their request started (written late).
static LATE_LISTS: AtomicU32 = AtomicU32::new(0);

/// Slots by the 2020 order, which the argument shapes at Grand Theft Auto
/// VI's call sites match (0x18, 0x20, 0x30 and 0x40 are called).
static QUEUE_VTABLE: Methods<12> = Methods([
    missing::<0> as *const (),
    add_ref_instance::<Queue> as *const (),
    release::<Queue> as *const (),
    enqueue_request as *const (), // 0x18
    enqueue_status as *const (),  // 0x20
    missing::<5> as *const (),    // 0x28 (2020: fence signal)
    submit as *const (),          // 0x30
    missing::<7> as *const (),    // 0x38 (2020: cancel by tag)
    close as *const (),           // 0x40
    missing::<9> as *const (),    // 0x48 (2020: error event)
    missing::<10> as *const (),   // 0x50 (2020: error record)
    missing::<11> as *const (),   // 0x58 (2020: queue info)
]);

pub(crate) fn module() -> HleModule {
    HleModule {
        name: "dstorage",
        dlls: &["xcrdapi.dll"],
        exports: vec![hle_export!(DStorageGetFactory)],
    }
}

#[cfg(test)]
mod tests {
    use std::os::windows::ffi::OsStrExt;
    use std::time::{Duration, Instant};

    use super::*;
    use crate::modules::xgameruntime::method;

    fn call<F: Copy>(object: *mut c_void, slot: usize) -> F {
        // SAFETY: every object here starts with the runtime header.
        method(unsafe { &*object.cast::<Object>() }, slot)
    }

    fn factory() -> *mut c_void {
        let mut factory = std::ptr::null_mut();
        assert_eq!(DStorageGetFactory(&FACTORY, &mut factory), S_OK);
        factory
    }

    fn open(path: &std::path::Path) -> *mut c_void {
        let wide: Vec<u16> = path.as_os_str().encode_wide().chain([0]).collect();
        let open: extern "system" fn(*mut c_void, *const u16, *const Guid, Out) -> HRESULT =
            call(factory(), 0x20);
        let mut file = std::ptr::null_mut();
        assert_eq!(
            open(factory(), wide.as_ptr(), std::ptr::null(), &mut file),
            S_OK
        );
        file
    }

    fn status_array(capacity: u32) -> *mut c_void {
        let create: extern "system" fn(
            *mut c_void,
            u32,
            *const c_char,
            *const Guid,
            Out,
        ) -> HRESULT = call(factory(), 0x28);
        let mut array = std::ptr::null_mut();
        let name = c"test".as_ptr();
        assert_eq!(
            create(factory(), capacity, name, std::ptr::null(), &mut array),
            S_OK
        );
        array
    }

    fn new_queue(source_type: u64) -> *mut c_void {
        let create: extern "system" fn(*mut c_void, *const QueueDesc, *const Guid, Out) -> HRESULT =
            call(factory(), 0x50);
        let desc = QueueDesc {
            source_type,
            capacity: 0x80,
            priority: 0,
            name: c"test".as_ptr(),
            unknown: 1,
        };
        let mut queue = std::ptr::null_mut();
        assert_eq!(create(factory(), &desc, std::ptr::null(), &mut queue), S_OK);
        queue
    }

    fn request(
        options: u64,
        destination: *mut c_void,
        destination_size: usize,
        source: *const c_void,
        offset: u64,
        source_size: usize,
    ) -> Request {
        Request {
            options,
            destination,
            page_offset: 0,
            destination_size: destination_size as u32,
            source,
            offset,
            source_size: source_size as u32,
            intermediate_size: 0,
            tag: 0,
            name: std::ptr::null(),
        }
    }

    fn enqueue(queue: *mut c_void, request: &Request) {
        let f: extern "system" fn(*mut c_void, *const Request) = call(queue, 0x18);
        f(queue, request);
    }

    fn enqueue_status(queue: *mut c_void, array: *mut c_void, index: u32) {
        let f: extern "system" fn(*mut c_void, *mut c_void, u32) = call(queue, 0x20);
        f(queue, array, index);
    }

    fn void(object: *mut c_void, slot: usize) {
        let f: extern "system" fn(*mut c_void) = call(object, slot);
        f(object);
    }

    fn result(array: *mut c_void, index: u32) -> HRESULT {
        let f: extern "system" fn(*mut c_void, u32) -> HRESULT = call(array, 0x20);
        f(array, index)
    }

    /// Polls as the title does, then returns the entry's result.
    fn wait(array: *mut c_void, index: u32) -> HRESULT {
        let done: extern "system" fn(*mut c_void, u32) -> bool = call(array, 0x18);
        let start = Instant::now();
        while !done(array, index) {
            assert!(start.elapsed() < Duration::from_secs(10), "never completed");
            std::thread::sleep(Duration::from_millis(1));
        }
        result(array, index)
    }

    fn file_with(bytes: &[u8]) -> (tempfile::TempDir, *mut c_void) {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("data.bin");
        std::fs::write(&path, bytes).unwrap();
        let file = open(&path);
        (dir, file)
    }

    fn pattern(n: usize) -> Vec<u8> {
        (0..n).map(|i| (i * 7 + i / 251) as u8).collect()
    }

    fn read_all(files: &Files, out: &mut [u8], at: u64) {
        let mut done = 0;
        while done < out.len() {
            let n = files.read_at(&mut out[done..], at + done as u64).unwrap();
            assert_ne!(n, 0, "{at} {}", out.len());
            done += n;
        }
    }

    #[test]
    fn workers_read_a_file_through_handles_of_their_own() {
        let plain = pattern(300_001);
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("data.bin");
        std::fs::write(&path, &plain).unwrap();
        for others in [0, 3] {
            let files = Files::with(std::fs::File::open(&path).unwrap(), Io::Handles, others);
            std::thread::scope(|s| {
                for _ in 0..6 {
                    s.spawn(|| {
                        let reads = [
                            (0, 4096),
                            (1, 100),
                            (4095, 2),
                            (12_345, 250_000),
                            (299_990, 11),
                        ];
                        for (at, len) in reads {
                            let mut out = vec![0; len];
                            read_all(&files, &mut out, at as u64);
                            assert!(out == plain[at..at + len], "{others} {at} {len}");
                        }
                        let mut out = [0; 16];
                        assert_eq!(files.read_at(&mut out, 299_999).unwrap(), 2);
                        assert_eq!(files.read_at(&mut out, 300_001).unwrap(), 0);
                        assert_eq!(files.read_at(&mut out, 400_000).unwrap(), 0);
                    });
                }
            });
            let made = files
                .others
                .iter()
                .filter(|o| o.get().is_some_and(Option::is_some))
                .count();
            assert_eq!(
                made, others,
                "a reopened handle for each worker slot under Windows/Wine"
            );
        }
    }

    #[test]
    fn every_way_of_reading_gives_the_same_bytes() {
        let plain = pattern(300_001);
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("data.bin");
        std::fs::write(&path, &plain).unwrap();
        for mode in [Io::One, Io::Handles, Io::Overlapped, Io::Direct] {
            let files = Files::new(std::fs::File::open(&path).unwrap(), mode);
            for (at, len) in [
                (0, 4096),
                (1, 100),
                (4095, 2),
                (8192, 200_000),
                (12_345, 250_000),
                (299_990, 11),
                (300_000, 1),
            ] {
                let mut out = vec![0u8; len];
                read_all(&files, &mut out, at as u64);
                assert!(out == plain[at..at + len], "{mode:?} {at} {len}");
            }
            let mut out = [0u8; 16];
            assert_eq!(files.read_at(&mut out, 299_999).unwrap(), 2, "{mode:?}");
            assert_eq!(files.read_at(&mut out, 300_001).unwrap(), 0, "{mode:?}");
            assert_eq!(files.read_at(&mut out, 400_000).unwrap(), 0, "{mode:?}");
        }
    }

    #[test]
    fn multi_worker_reopen_handles_throughput() {
        const THREADS: usize = 16;
        const FILE_LEN: usize = 16 << 20;
        const READ_SIZE: usize = 48 << 10;
        const READS: usize = 4_000;
        let plain = pattern(FILE_LEN);
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("archive.bin");
        std::fs::write(&path, &plain).unwrap();
        let mut seed = 0x9e37_79b9_7f4a_7c15_u64;
        let offsets: Vec<u64> = (0..READS)
            .map(|_| {
                seed ^= seed << 13;
                seed ^= seed >> 7;
                seed ^= seed << 17;
                (seed % (FILE_LEN - READ_SIZE) as u64) & !15
            })
            .collect();

        for (label, mode, slots) in [
            ("shared handle (Io::One)", Io::One, 0),
            ("per-worker ReOpenFile (Io::Handles)", Io::Handles, THREADS),
            ("shared overlapped (Io::Overlapped)", Io::Overlapped, 1),
            ("per-worker unbuffered (Io::Direct)", Io::Direct, THREADS),
        ] {
            let wide: Box<[u16]> = path.as_os_str().encode_wide().chain([0]).collect();
            let files =
                Files::with_path(std::fs::File::open(&path).unwrap(), Some(wide), mode, slots);
            // Warm all worker slots before timing steady-state throughput.
            let next = std::sync::atomic::AtomicUsize::new(0);
            let busy_us = AtomicU64::new(0);
            let start = Instant::now();
            std::thread::scope(|s| {
                for tid in 0..THREADS {
                    let (files, offsets, next, busy_us, plain) =
                        (&files, &offsets, &next, &busy_us, &plain);
                    s.spawn(move || {
                        WORKER_SLOT.set(Some(tid));
                        let mut out = vec![0u8; READ_SIZE];
                        while let Some(&at) = offsets.get(next.fetch_add(1, Ordering::Relaxed)) {
                            let t0 = Instant::now();
                            read_all(files, &mut out, at);
                            busy_us.fetch_add(t0.elapsed().as_micros() as u64, Ordering::Relaxed);
                            let idx = at as usize;
                            assert_eq!(out[0], plain[idx]);
                            assert_eq!(out[READ_SIZE - 1], plain[idx + READ_SIZE - 1]);
                        }
                        WORKER_SLOT.set(None);
                    });
                }
            });
            let elapsed = start.elapsed().as_secs_f64().max(1e-6);
            let mb_s = (READS * READ_SIZE) as f64 / elapsed / 1e6;
            let avg_ms = busy_us.load(Ordering::Relaxed) as f64 / READS as f64 / 1e3;
            let reopened = files
                .others
                .iter()
                .filter(|o| o.get().is_some_and(Option::is_some))
                .count();
            println!(
                "[dstorage bench] {label}: {mb_s:.1} MB/s ({avg_ms:.3} ms/read, reopened={reopened}/{slots})"
            );
            if slots > 0 {
                assert!(reopened > 0, "worker handles should open");
            }
        }
    }

    /// Reproduces the live 16-worker M5 Max `offset=69614334544` (>4 GiB)
    /// concurrent `seek_read` zlib corruption scenario across 8 threads on a
    /// virtual `G:` drive: tests `ReOpenFile` handles, `CreateFileW` virtual
    /// drive handles, `Io::One` mutex fallback, modulo-colliding non-worker
    /// callers, and full DirectStorage `ZLIB` queue submissions simultaneously.
    #[test]
    fn concurrent_virtual_drive_high_offset_zlib_reads_never_corrupt() {
        const CHUNKS: usize = 8;
        const PLAIN_LEN: usize = 65_536;
        const BASE_4GIB_PLUS: u64 = 5_000_000_123; // > 4 GiB, unaligned
        const STRIDE: u64 = 128 << 20;

        let dir = tempfile::tempdir().unwrap();
        let sub = dir.path().join("levels");
        std::fs::create_dir_all(&sub).unwrap();
        let archive_path = sub.join("sparse_archive.rpf");

        // Build 8 distinct 64 KiB payloads compressed with zlib at >4 GiB offsets.
        let mut expected_plain = Vec::with_capacity(CHUNKS);
        let mut compressed_chunks = Vec::with_capacity(CHUNKS);
        let mut chunk_offsets = Vec::with_capacity(CHUNKS);
        {
            let f = std::fs::OpenOptions::new()
                .create(true)
                .write(true)
                .read(true)
                .truncate(true)
                .open(&archive_path)
                .unwrap();
            let total_sparse_len = BASE_4GIB_PLUS + (CHUNKS as u64) * STRIDE + (1 << 20);
            f.set_len(total_sparse_len).unwrap();
            for c in 0..CHUNKS {
                let plain: Vec<u8> = (0..PLAIN_LEN)
                    .map(|i| ((i * 13 + c * 53 + (i >> 8) * 19) & 0xff) as u8)
                    .collect();
                let z = miniz_oxide::deflate::compress_to_vec_zlib(&plain, 6);
                let off = BASE_4GIB_PLUS + (c as u64) * STRIDE + (c as u64) * 317;
                let mut written = 0;
                while written < z.len() {
                    written += f.seek_write(&z[written..], off + written as u64).unwrap();
                }
                expected_plain.push(plain);
                compressed_chunks.push(z);
                chunk_offsets.push(off);
            }
        }

        // 1) Mount `dir` on a virtual drive letter and open via `open_file` (`G:\levels\...`).
        let drive = crate::modules::files::mount(dir.path()).expect("virtual drive letter");
        let virtual_str = format!("{drive}:\\levels\\sparse_archive.rpf");
        let virtual_wide: Vec<u16> = virtual_str.encode_utf16().chain([0]).collect();
        let open_fn: extern "system" fn(*mut c_void, *const u16, *const Guid, Out) -> HRESULT =
            call(factory(), 0x20);
        let mut dstorage_file = std::ptr::null_mut();
        assert_eq!(
            open_fn(
                factory(),
                virtual_wide.as_ptr(),
                std::ptr::null(),
                &mut dstorage_file
            ),
            S_OK
        );

        // 2) Test 8-thread concurrent reads + zlib inflate at >4 GiB offsets across:
        //    - raw `ReOpenFile` (`wide_path = None`, 8 slots AND 3 modulo-colliding slots),
        //    - virtual-drive `Files` (`wide_path = Some(virtual_wide)`, 8 slots),
        //    - `Io::One` fallback (`0` other slots -> shared `first: Mutex<File>`).
        let virtual_box: Box<[u16]> = virtual_wide.clone().into_boxed_slice();
        for (case_name, wide_opt, mode, slots) in [
            ("ReOpenFile 8 slots", None, Io::Handles, 8usize),
            ("ReOpenFile 3 colliding slots", None, Io::Handles, 3usize),
            (
                "Virtual-drive path 8 slots",
                Some(virtual_box.clone()),
                Io::Handles,
                8usize,
            ),
            ("Io::One mutex fallback", None, Io::One, 0usize),
        ] {
            let raw_file = std::fs::File::open(&archive_path).unwrap();
            let files = Files::with_path(raw_file, wide_opt, mode, slots);
            let barrier = std::sync::Barrier::new(8);
            std::thread::scope(|s| {
                for tid in 0..8usize {
                    let (files, barrier, expected_plain, compressed_chunks, chunk_offsets) = (
                        &files,
                        &barrier,
                        &expected_plain,
                        &compressed_chunks,
                        &chunk_offsets,
                    );
                    s.spawn(move || {
                        if slots == 8 {
                            WORKER_SLOT.set(Some(tid));
                        } else {
                            WORKER_SLOT.set(None);
                        }
                        barrier.wait();
                        let mut comp_buf = vec![0u8; PLAIN_LEN * 2];
                        let mut out_plain = vec![0u8; PLAIN_LEN];
                        for iter in 0..250usize {
                            let c = (tid * 3 + iter * 5) % CHUNKS;
                            let comp_len = compressed_chunks[c].len();
                            read_all(files, &mut comp_buf[..comp_len], chunk_offsets[c]);
                            let n = inflate(&comp_buf[..comp_len], &mut out_plain).unwrap_or_else(
                                |hr| {
                                    panic!(
                                        "{case_name} tid={tid} iter={iter} chunk={c} offset={} zlib corrupted: hr={hr:#010x}",
                                        chunk_offsets[c]
                                    )
                                },
                            );
                            assert_eq!(n, PLAIN_LEN);
                            assert_eq!(out_plain, expected_plain[c], "{case_name} byte mismatch");
                        }
                        WORKER_SLOT.set(None);
                    });
                }
            });
            if slots > 0 {
                let opened = files
                    .others
                    .iter()
                    .filter(|o| o.get().is_some_and(Option::is_some))
                    .count();
                assert_eq!(opened, slots, "{case_name}: all slots reopened");
            }
        }

        // 3) End-to-end DirectStorage queue submission of 64 concurrent >4 GiB
        //    zlib requests through the virtual-drive `dstorage_file` handle!
        let (queue, array) = (new_queue(0), status_array(1));
        let mut outputs: Vec<Vec<u8>> = (0..64).map(|_| vec![0u8; PLAIN_LEN]).collect();
        for (req_idx, out_buf) in outputs.iter_mut().enumerate() {
            let c = req_idx % CHUNKS;
            let r = request(
                ZLIB,
                out_buf.as_mut_ptr().cast(),
                PLAIN_LEN,
                dstorage_file,
                chunk_offsets[c],
                compressed_chunks[c].len(),
            );
            enqueue(queue, &r);
        }
        enqueue_status(queue, array, 0);
        void(queue, 0x30);
        assert_eq!(wait(array, 0), S_OK);
        for (req_idx, out_buf) in outputs.iter().enumerate() {
            assert_eq!(out_buf, &expected_plain[req_idx % CHUNKS]);
        }
        void(queue, 0x40);
        void(dstorage_file, 0x10);
    }

    /// Benchmark against a real archive at `VICTORY_BENCH_FILE`:
    /// `VICTORY_BENCH_FILE=... cargo test --release -p victory-hle --lib read_bench -- --ignored --nocapture`
    #[test]
    #[ignore = "a benchmark"]
    fn read_bench() {
        const SIZE: usize = 48 << 10;
        const READS: usize = 12_000;
        let threads = workers();
        let path = std::env::var("VICTORY_BENCH_FILE").expect("VICTORY_BENCH_FILE");
        let len = std::fs::metadata(&path).unwrap().len();
        let mut seed = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_nanos() as u64
            | 1;
        let mut random = move || {
            seed ^= seed << 13;
            seed ^= seed >> 7;
            seed ^= seed << 17;
            seed
        };
        for mode in [Io::One, Io::Handles, Io::Overlapped, Io::Direct] {
            let files = Files::new(std::fs::File::open(&path).unwrap(), mode);
            let at: Vec<u64> = (0..READS)
                .map(|_| (random() % (len - SIZE as u64)) & !15)
                .collect();
            for pass in ["cold", "again"] {
                let next = std::sync::atomic::AtomicUsize::new(0);
                let busy = AtomicU64::new(0);
                let start = Instant::now();
                std::thread::scope(|s| {
                    for _ in 0..threads {
                        s.spawn(|| {
                            let mut out = vec![0u8; SIZE];
                            while let Some(&at) = at.get(next.fetch_add(1, Ordering::Relaxed)) {
                                let began = Instant::now();
                                read_all(&files, &mut out, at);
                                busy.fetch_add(
                                    began.elapsed().as_micros() as u64,
                                    Ordering::Relaxed,
                                );
                            }
                        });
                    }
                });
                let took = start.elapsed().as_secs_f64();
                println!(
                    "{threads} threads, {mode:?}, {pass}: {:.0} MB/s, {:.2} ms a read",
                    (READS * SIZE) as f64 / took / 1e6,
                    busy.load(Ordering::Relaxed) as f64 / READS as f64 / 1e3
                );
            }
        }
    }

    #[test]
    fn request_layout_is_the_titles() {
        assert_eq!(size_of::<Request>(), 0x48);
        assert_eq!(std::mem::offset_of!(Request, destination_size), 0x18);
        assert_eq!(std::mem::offset_of!(Request, source), 0x20);
        assert_eq!(std::mem::offset_of!(Request, source_size), 0x30);
        assert_eq!(std::mem::offset_of!(Request, name), 0x40);
        assert_eq!(std::mem::offset_of!(QueueDesc, unknown), 0x18);
    }

    #[test]
    fn raw_and_zlib_reads_from_a_file() {
        let plain = pattern(300_000);
        let packed = miniz_oxide::deflate::compress_to_vec_zlib(&plain, 6);
        let mut contents = plain.clone();
        contents.extend_from_slice(&packed);
        let (_dir, file) = file_with(&contents);
        let (queue, array) = (new_queue(0), status_array(1));

        let mut raw = vec![0u8; 1000];
        let mut inflated = vec![0u8; plain.len()];
        let at = |b: &mut Vec<u8>| b.as_mut_ptr().cast();
        enqueue(queue, &request(0, at(&mut raw), 1000, file, 5000, 1000));
        let offset = plain.len() as u64;
        let zlib = request(
            ZLIB,
            at(&mut inflated),
            plain.len(),
            file,
            offset,
            packed.len(),
        );
        enqueue(queue, &zlib);
        enqueue_status(queue, array, 0);
        assert_eq!(result(array, 0), E_PENDING, "pending until done");
        void(queue, 0x30);
        assert_eq!(wait(array, 0), S_OK);
        assert_eq!(raw, plain[5000..6000]);
        assert_eq!(inflated, plain);
        void(queue, 0x40);
    }

    #[test]
    fn memory_sources_inflate_too() {
        let plain = pattern(70_000);
        let packed = miniz_oxide::deflate::compress_to_vec_zlib(&plain, 6);
        let (queue, array) = (new_queue(1), status_array(1));
        let mut out = vec![0u8; plain.len()];
        let options = ZLIB | MEMORY_SOURCE;
        let r = request(
            options,
            out.as_mut_ptr().cast(),
            out.len(),
            packed.as_ptr().cast(),
            0,
            packed.len(),
        );
        enqueue(queue, &r);
        enqueue_status(queue, array, 0);
        void(queue, 0x30);
        assert_eq!(wait(array, 0), S_OK);
        assert_eq!(out, plain);
    }

    #[test]
    fn statuses_complete_in_order_with_the_first_failure() {
        let contents = pattern(4096);
        let (_dir, file) = file_with(&contents);
        let (queue, array) = (new_queue(0), status_array(3));
        // Requests in flight together run in any order, so none of them
        // shares a destination (failed ones leave theirs undefined).
        let mut buffers = [[0u8; 16]; 5];
        let [failed, failed_too, out, other, bcpack] =
            buffers.each_mut().map(|b| b.as_mut_ptr().cast());

        // Segment 0: a read past the end, then a bad stream; the first wins.
        enqueue(queue, &request(0, failed, 16, file, 4090, 16));
        enqueue(queue, &request(ZLIB, failed_too, 16, file, 0, 16));
        enqueue_status(queue, array, 0);
        // Segment 1: two reads.
        enqueue(queue, &request(0, other, 16, file, 0, 16));
        enqueue(queue, &request(0, out, 16, file, 100, 16));
        enqueue_status(queue, array, 1);
        void(queue, 0x30);
        // Segment 2: a BCPack request (BC7) whose stream is empty fails.
        with_test_tables();
        enqueue(queue, &request(7 << 1, bcpack, 16, file, 0, 0));
        enqueue_status(queue, array, 2);
        void(queue, 0x30);

        assert_eq!(wait(array, 2), INVALID_DATA);
        assert_eq!(result(array, 0), END_OF_FILE);
        assert_eq!(result(array, 1), S_OK);
        assert_eq!(buffers[2], contents[100..116]);
        assert_eq!(buffers[3], contents[..16]);
    }

    #[test]
    fn half_a_queue_submits_itself() {
        let (_dir, file) = file_with(&pattern(64));
        let create: extern "system" fn(*mut c_void, *const QueueDesc, *const Guid, Out) -> HRESULT =
            call(factory(), 0x50);
        let desc = QueueDesc {
            source_type: 0,
            capacity: 4,
            priority: 1,
            name: c"small".as_ptr(),
            unknown: 1,
        };
        let mut queue = std::ptr::null_mut();
        assert_eq!(create(factory(), &desc, std::ptr::null(), &mut queue), S_OK);
        let array = status_array(1);
        let mut out = [0u8; 16];
        enqueue(queue, &request(0, out.as_mut_ptr().cast(), 16, file, 8, 16));
        enqueue_status(queue, array, 0); // the second of four: no submit needed
        assert_eq!(wait(array, 0), S_OK);
        assert_eq!(out, pattern(64)[8..24]);
    }

    /// Lanes without workers: what `start` picks, and when statuses land.
    fn lanes(staging: u64) -> Lanes {
        Lanes {
            lanes: HashMap::new(),
            next_id: 0,
            next_seq: 0,
            staged: 0,
            staging,
            starved: false,
            clock: [0; 3],
            now: 0,
            lane_now: [0; 3],
            done: Default::default(),
            refused: 0,
            bcpack: Default::default(),
            pages: 0,
            since: None,
            moved: std::time::Instant::now(),
            said_idle: None,
            polls: 0,
            polling_since: None,
            said_polling: false,
        }
    }

    fn read(lanes: &mut Lanes, id: u64, options: u64, size: u32) {
        let mut r = request(options, std::ptr::null_mut(), 0, std::ptr::null(), 0, 0);
        (r.destination_size, r.source_size) = (size, size);
        let lane = lanes.lanes.get_mut(&id).unwrap();
        lane.unsubmitted.push(Command::Read(r));
    }

    #[test]
    fn priorities_share_by_bytes_about_tenfold() {
        let mut l = lanes(STAGING);
        l.lanes.insert(0, Lane::new(0, 4096));
        l.lanes.insert(1, Lane::new(1, 4096));
        l.lanes.insert(2, Lane::new(2, 4096)); // real-time: as high
        for _ in 0..400 {
            for id in 0..3 {
                read(&mut l, id, 0, 4096);
            }
        }
        (0..3).for_each(|id| l.submit(id));
        let mut started = [0u32; 3];
        for _ in 0..330 {
            let job = l.start().unwrap();
            started[job.lane as usize] += 1;
            l.finish(&job, Ok(()));
        }
        // Normal gets about a tenth of what high and real-time get together.
        assert_eq!(started[1] + started[2], 300, "{started:?}");
        assert_eq!(started[0], 30, "{started:?}");
        assert!(started[1].abs_diff(started[2]) <= 1, "{started:?}");
    }

    #[test]
    fn staging_bounds_compressed_bytes_in_flight() {
        let mut l = lanes(100);
        l.lanes.insert(0, Lane::new(0, 64));
        read(&mut l, 0, ZLIB, 60);
        read(&mut l, 0, ZLIB, 60);
        read(&mut l, 0, 0, 1000); // raw: straight to its destination
        l.submit(0);
        let first = l.start().unwrap();
        assert!(l.start().is_none(), "60 + 60 > 100");
        l.finish(&first, Ok(()));
        let second = l.start().unwrap();
        let raw = l.start().unwrap();
        assert_eq!((second.staging, raw.staging), (60, 0));
        // Alone, a request larger than the buffer still runs.
        read(&mut l, 0, ZLIB, 500);
        l.submit(0);
        assert!(l.start().is_none());
        l.finish(&second, Ok(()));
        assert!(l.start().is_some());
    }

    #[test]
    fn a_status_waits_for_every_earlier_read() {
        let mut l = lanes(STAGING);
        l.lanes.insert(0, Lane::new(0, 64));
        let array = status_array(2).cast::<Instance<StatusArray>>();
        let status = |index| {
            add_ref_instance(array);
            Command::Status(Status { array, index })
        };
        read(&mut l, 0, 0, 16);
        read(&mut l, 0, 0, 16);
        l.lanes.get_mut(&0).unwrap().unsubmitted.push(status(0));
        read(&mut l, 0, 0, 16);
        l.lanes.get_mut(&0).unwrap().unsubmitted.push(status(1));
        // SAFETY: an array we created.
        for entry in unsafe { &(*array).state.entries }.iter() {
            entry.store(E_PENDING, Ordering::Release);
        }
        l.submit(0);
        let jobs: Vec<Job> = (0..3).map(|_| l.start().unwrap()).collect();
        // The last read and the first finish; the second then fails.
        l.finish(&jobs[2], Ok(()));
        l.finish(&jobs[0], Ok(()));
        let results = || (result(array.cast(), 0), result(array.cast(), 1));
        assert_eq!(results(), (E_PENDING, E_PENDING));
        l.finish(&jobs[1], Err(INVALID_DATA));
        assert_eq!(results(), (INVALID_DATA, S_OK));
    }

    #[test]
    fn zlib_then_bcpack() {
        with_test_tables();
        let packed = miniz_oxide::deflate::compress_to_vec_zlib(&BC4_STREAM, 6);
        let (_dir, file) = file_with(&packed);
        let (queue, array) = (new_queue(0), status_array(1));
        let mut out = [0u8; 16];
        let options = ZLIB | 4 << 1;
        let mut r = request(options, out.as_mut_ptr().cast(), 16, file, 0, packed.len());
        r.intermediate_size = BC4_STREAM.len() as u32;
        enqueue(queue, &r);
        enqueue_status(queue, array, 0);
        void(queue, 0x30);
        assert_eq!(wait(array, 0), S_OK);
        assert_eq!(out, bc4_blocks());
    }

    /// BCPack tables we made (uniform models; BC4 index pairs: low 3 bits,
    /// next 3 bits), in a folder kept for the test process.
    fn with_test_tables() {
        static DIR: OnceLock<()> = OnceLock::new();
        DIR.get_or_init(|| {
            let mut t = Vec::new();
            for m in 0..128 {
                let g = m / 16;
                for b in 0..9 {
                    let f: u16 = if b < g + 2 { 1 << (15 - g) } else { 0 };
                    t.extend_from_slice(&f.to_le_bytes());
                }
            }
            t.extend([0; 256 + 64 + 256]);
            t.extend((0..64u8).map(|v| (v & 7) | (v >> 3) << 4));
            t.resize(bcpack::TABLES_SIZE, 0);
            let dir = tempfile::tempdir().unwrap().keep();
            std::fs::write(dir.join("tables.bin"), t).unwrap();
            use_bcpack_tables(dir);
        });
    }

    /// Two BC4 blocks from the format's test encoder: endpoints 10, -3 with
    /// indices p -> p & 7, then the same block plus 1 everywhere.
    const BC4_STREAM: [u8; 88] = [
        0, 0, 0, 0, 0, 0, 0, 0, 5, 5, 5, 5, 4, 4, 4, 4, 1, 1, 1, 1, 0x10, 0x10, 0x10, 0x10, 2, 1,
        0, 0, 0, 0, 0, 0, 0, 0, 0x14, 5, 0x20, 0x68, 0xb0, 0xf8, 3, 3, 2, 2, 0x24, 0x24, 0x24,
        0x24, 2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 4, 4, 4, 4, 0x10, 0x10, 0x10, 0x10, 0, 0, 0, 0,
        0x20, 0x68, 0xb0, 0xf8, 0x24, 0x24, 0x24, 0x24, 0, 0, 0, 0, 0, 0, 0, 0,
    ];

    /// What `BC4_STREAM` unpacks to.
    fn bc4_blocks() -> [u8; 16] {
        let code = [0u64, 2, 3, 4, 5, 6, 7, 1];
        let block = |e0: u64, e1: u64, add: usize| {
            let bits = (0..16).fold(0, |a, p| a | code[(p + add) & 7] << (16 + 3 * p));
            (e0 | e1 << 8 | bits).to_le_bytes()
        };
        let mut out = [0; 16];
        out[..8].copy_from_slice(&block(138, 125, 0));
        out[8..].copy_from_slice(&block(139, 126, 1));
        out
    }

    #[test]
    fn bcpack_requests_unpack_into_the_destination() {
        with_test_tables();
        let mut contents = pattern(100);
        contents.extend_from_slice(&BC4_STREAM);
        let (_dir, file) = file_with(&contents);
        let (queue, array) = (new_queue(0), status_array(1));
        let mut out = [0u8; 16];
        enqueue(
            queue,
            &request(4 << 1, out.as_mut_ptr().cast(), 16, file, 100, 88),
        );
        enqueue_status(queue, array, 0);
        void(queue, 0x30);
        assert_eq!(wait(array, 0), S_OK);
        assert_eq!(out, bc4_blocks());
    }

    #[test]
    fn a_closed_queue_completes_nothing() {
        let (_dir, file) = file_with(&pattern(64));
        let (queue, array) = (new_queue(0), status_array(1));
        let mut out = [0u8; 16];
        enqueue(queue, &request(0, out.as_mut_ptr().cast(), 16, file, 0, 16));
        enqueue_status(queue, array, 0);
        void(queue, 0x40);
        void(queue, 0x30);
        std::thread::sleep(Duration::from_millis(50));
        assert_eq!(result(array, 0), E_PENDING);
        assert_eq!(out, [0; 16]);
    }

    #[test]
    fn physical_page_destinations() {
        use crate::modules::xmem;
        let mut pages = [0usize; 2];
        let mut n = 2;
        assert_eq!(
            xmem::XMemAllocatePhysicalPages(0, &mut n, pages.as_mut_ptr()),
            1
        );
        // Where the title would see them: a reservation of 64 KiB pages.
        let mapped = xmem::XMemVirtualAlloc(
            std::ptr::null_mut(),
            2 * PAGE,
            0x2000 | 0x2040_0000, // reserve, 64 KiB pages
            0,
            0x04,
        );
        assert!(!mapped.is_null());
        assert_eq!(
            xmem::XMemMapPhysicalPages(mapped, 2, pages.as_ptr()),
            mapped
        );

        let plain = pattern(100);
        let (queue, array) = (new_queue(1), status_array(1));
        let mut r = request(
            PAGE_DESTINATION | MEMORY_SOURCE,
            pages.as_mut_ptr().cast(),
            plain.len(),
            plain.as_ptr().cast(),
            0,
            plain.len(),
        );
        r.page_offset = (PAGE - 40) as u64; // across the page boundary
        enqueue(queue, &r);
        enqueue_status(queue, array, 0);
        void(queue, 0x30);
        assert_eq!(wait(array, 0), S_OK);
        // SAFETY: both pages are mapped read-write.
        let seen = unsafe { std::slice::from_raw_parts(mapped.cast::<u8>().add(PAGE - 40), 100) };
        assert_eq!(seen, plain);
    }
    #[test]
    fn a_page_list_filled_after_its_request_starts() {
        use crate::modules::xmem;
        let (mut given, mut page) = (1usize, [0usize; 1]);
        assert_eq!(
            xmem::XMemAllocatePhysicalPages(0, &mut given, page.as_mut_ptr()),
            1
        );
        let mapped = xmem::XMemVirtualAlloc(
            std::ptr::null_mut(),
            PAGE,
            0x2000 | 0x2040_0000, // reserve, 64 KiB pages
            0,
            0x04,
        );
        assert!(!mapped.is_null());
        assert_eq!(xmem::XMemMapPhysicalPages(mapped, 1, page.as_ptr()), mapped);

        // The title's list, still empty when the request is submitted.
        let list = Box::leak(Box::new([0usize; 1]));
        let at = std::ptr::from_mut(list) as usize;
        let plain = pattern(4096);
        let (queue, array) = (new_queue(1), status_array(1));
        let r = request(
            PAGE_DESTINATION | MEMORY_SOURCE,
            list.as_mut_ptr().cast(),
            plain.len(),
            plain.as_ptr().cast(),
            0,
            plain.len(),
        );
        let number = page[0];
        let filler = std::thread::spawn(move || {
            std::thread::sleep(std::time::Duration::from_millis(2));
            // SAFETY: the leaked list, alive for the whole test.
            unsafe { std::ptr::write_volatile(at as *mut usize, number) };
        });
        enqueue(queue, &r);
        enqueue_status(queue, array, 0);
        void(queue, 0x30);
        assert_eq!(wait(array, 0), S_OK);
        filler.join().unwrap();
        // SAFETY: the page is mapped read-write.
        let seen = unsafe { std::slice::from_raw_parts(mapped.cast::<u8>(), plain.len()) };
        assert_eq!(seen, plain);
    }

    #[test]
    fn a_page_list_never_filled_fails_its_request() {
        let list = [0usize; 1];
        let plain = pattern(4096);
        let (queue, array) = (new_queue(1), status_array(1));
        let r = request(
            PAGE_DESTINATION | MEMORY_SOURCE,
            list.as_ptr().cast_mut().cast(),
            plain.len(),
            plain.as_ptr().cast(),
            0,
            plain.len(),
        );
        enqueue(queue, &r);
        enqueue_status(queue, array, 0);
        void(queue, 0x30);
        assert_eq!(wait(array, 0), E_INVALIDARG);
    }
    #[test]
    fn a_queue_that_submits_itself_holds_reads_into_pages() {
        let mut l = lanes(STAGING);
        l.lanes.insert(0, Lane::new(0, 64));
        let array = status_array(1).cast::<Instance<StatusArray>>();
        read(&mut l, 0, PAGE_DESTINATION, 16);
        read(&mut l, 0, 0, 16);
        add_ref_instance(array);
        l.lanes
            .get_mut(&0)
            .unwrap()
            .unsubmitted
            .push(Command::Status(Status { array, index: 0 }));
        l.submit_as(0, By::Auto);
        // Only the plain read starts; the status waits behind the held one.
        let plain = l.start().unwrap();
        assert_eq!(plain.request.options, 0);
        assert!(l.start().is_none());
        l.finish(&plain, Ok(()));
        assert_eq!(l.lanes[&0].held.len(), 2);
        l.submit(0);
        let paged = l.start().unwrap();
        assert_eq!(paged.request.options, PAGE_DESTINATION);
        assert!(l.lanes[&0].held.is_empty());
        assert!(!l.lanes[&0].statuses.is_empty());
        l.finish(&paged, Ok(()));
        assert!(l.lanes[&0].statuses.is_empty());
    }

    /// A queue that submitted itself, holding a read into pages and the
    /// status behind it, after its plain read ran.
    fn held_page_read() -> Lanes {
        let mut l = lanes(STAGING);
        l.lanes.insert(0, Lane::new(0, 64));
        let array = status_array(1).cast::<Instance<StatusArray>>();
        read(&mut l, 0, PAGE_DESTINATION, 16);
        read(&mut l, 0, 0, 16);
        add_ref_instance(array);
        l.lanes
            .get_mut(&0)
            .unwrap()
            .unsubmitted
            .push(Command::Status(Status { array, index: 0 }));
        l.submit_as(0, By::Auto);
        let plain = l.start().unwrap();
        l.finish(&plain, Ok(()));
        assert_eq!(l.lanes[&0].held.len(), 2);
        assert!(l.lanes[&0].held_since.is_some());
        l
    }

    #[test]
    fn held_reads_start_once_the_hold_has_passed() {
        let mut l = held_page_read();
        // Not yet: the hold has not passed.
        l.release_held(Duration::from_secs(3600));
        assert_eq!(l.lanes[&0].held.len(), 2);
        assert!(l.start().is_none());
        // The Submit never came: they start anyway.
        l.release_held(Duration::ZERO);
        assert!(l.lanes[&0].held.is_empty());
        let paged = l.start().unwrap();
        assert_eq!(paged.request.options, PAGE_DESTINATION);
        l.finish(&paged, Ok(()));
        assert!(l.lanes[&0].statuses.is_empty());
    }

    #[test]
    fn polling_a_pending_status_releases_held_reads() {
        let mut l = held_page_read();
        // A first poll only starts the clock.
        l.note_poll(0);
        assert_eq!(l.lanes[&0].held.len(), 2);
        // Polled for longer than POLLING: the title waits for these reads.
        l.polling_since = Some(Instant::now() - POLLING);
        l.note_poll(0);
        assert!(l.lanes[&0].held.is_empty());
        let paged = l.start().unwrap();
        assert_eq!(paged.request.options, PAGE_DESTINATION);
    }

    #[test]
    fn a_page_list_the_title_changes_before_its_submit() {
        use crate::modules::xmem;
        let (mut given, mut pages) = (2usize, [0usize; 2]);
        assert_eq!(
            xmem::XMemAllocatePhysicalPages(0, &mut given, pages.as_mut_ptr()),
            1
        );
        let mapped = xmem::XMemVirtualAlloc(
            std::ptr::null_mut(),
            2 * PAGE,
            0x2000 | 0x2040_0000, // reserve, 64 KiB pages
            0,
            0x04,
        );
        assert!(!mapped.is_null());
        assert_eq!(
            xmem::XMemMapPhysicalPages(mapped, 2, pages.as_ptr()),
            mapped
        );
        // The title's page table: still the page's previous owner when the
        // request is enqueued, the right page by its Submit.
        let mut list = [pages[0]];
        let plain = pattern(4096);
        let create: extern "system" fn(*mut c_void, *const QueueDesc, *const Guid, Out) -> HRESULT =
            call(factory(), 0x50);
        let desc = QueueDesc {
            source_type: 0,
            capacity: 4,
            priority: 1,
            name: c"pages".as_ptr(),
            unknown: 1,
        };
        let mut queue = std::ptr::null_mut();
        assert_eq!(create(factory(), &desc, std::ptr::null(), &mut queue), S_OK);
        let array = status_array(1);
        let r = request(
            PAGE_DESTINATION | MEMORY_SOURCE,
            list.as_mut_ptr().cast(),
            plain.len(),
            plain.as_ptr().cast(),
            0,
            plain.len(),
        );
        enqueue(queue, &r);
        let mut out = [0u8; 16];
        enqueue(
            queue,
            &request(
                MEMORY_SOURCE,
                out.as_mut_ptr().cast(),
                16,
                plain.as_ptr().cast(),
                0,
                16,
            ),
        );
        // Half of four: the queue submitted itself. Time for a read started
        // then to land in the old page.
        std::thread::sleep(std::time::Duration::from_millis(50));
        // SAFETY: the list outlives the request, which reads it when it runs.
        unsafe { std::ptr::write_volatile(list.as_mut_ptr(), pages[1]) };
        enqueue_status(queue, array, 0);
        void(queue, 0x30);
        assert_eq!(wait(array, 0), S_OK);
        assert_eq!(out, plain[..16]);
        // SAFETY: both pages are mapped read-write.
        let (first, second) = unsafe {
            (
                std::slice::from_raw_parts(mapped.cast::<u8>(), plain.len()),
                std::slice::from_raw_parts(mapped.cast::<u8>().add(PAGE), plain.len()),
            )
        };
        assert_eq!(second, plain);
        assert!(first.iter().all(|&b| b == 0), "the old page was written");
    }

    #[test]
    fn zlib_into_pages_in_place_and_across_allocations() {
        use crate::modules::xmem;
        let allocate = |n: usize| {
            let (mut pages, mut given) = (vec![0usize; n], n);
            assert_eq!(
                xmem::XMemAllocatePhysicalPages(0, &mut given, pages.as_mut_ptr()),
                1
            );
            pages
        };
        let (a, b) = (allocate(3), allocate(1));
        let plain = pattern(2 * PAGE + 100);
        let packed = miniz_oxide::deflate::compress_to_vec_zlib(&plain, 6);
        // One allocation's pages in order (one window), then pages of two
        // (through a buffer), both from 50 bytes into the first.
        let mut destinations = [a.clone(), vec![a[0], a[1], b[0]]];
        let (queue, array) = (new_queue(1), status_array(1));
        for pages in &mut destinations {
            let mut r = request(
                ZLIB | PAGE_DESTINATION | MEMORY_SOURCE,
                pages.as_mut_ptr().cast(),
                plain.len(),
                packed.as_ptr().cast(),
                0,
                packed.len(),
            );
            r.page_offset = 50;
            enqueue(queue, &r);
        }
        enqueue_status(queue, array, 0);
        void(queue, 0x30);
        assert_eq!(wait(array, 0), S_OK);
        for pages in &destinations {
            let mapped = xmem::XMemVirtualAlloc(
                std::ptr::null_mut(),
                3 * PAGE,
                0x2000 | 0x2040_0000, // reserve, 64 KiB pages
                0,
                0x04,
            );
            assert!(!mapped.is_null());
            assert_eq!(
                xmem::XMemMapPhysicalPages(mapped, 3, pages.as_ptr()),
                mapped
            );
            // SAFETY: all three pages are mapped read-write.
            let seen = unsafe { std::slice::from_raw_parts(mapped.cast::<u8>(), 3 * PAGE) };
            assert!(seen[..50].iter().all(|&b| b == 0));
            assert_eq!(seen[50..50 + plain.len()], plain);
            assert_eq!(
                xmem::XMemMapPhysicalPages(mapped, 3, std::ptr::null()),
                mapped
            );
        }
        assert_eq!(xmem::XMemFreePhysicalPages(3, a.as_ptr()), 1);
        assert_eq!(xmem::XMemFreePhysicalPages(1, b.as_ptr()), 1);
    }
}
