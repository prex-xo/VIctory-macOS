//! Background worker for heap-indexed texture creation and detiling
//! (`VICTORY_HEAP_WORKER=1`, off by default).
//!
//! Moves host texture creation, staging allocation, memory comparison/hashing,
//! and CPU detiling for bindless heap-indexed textures off `Host::execute` onto
//! a single dedicated thread (`victory-heap`).
//!
//! D3DMetal / single-queue invariants:
//! - No second D3D12 command queue: the worker never submits commands and never
//!   acquires `Host::submitted`, preserving `Host::quiet()` (shader-visible heap
//!   creation serialization), `Host::lost()` / queue replacement, `Mark` checks,
//!   and linear `HostTexture::state` tracking.
//! - Isolated staging pool ([`WorkerStaging`]): worker-owned upload chunks and
//!   oversize buffers are transferred to [`ReadyBatch`] and only returned when
//!   `Host::execute` finishes waiting on its fence, so `crate::texture::recycle`
//!   can never rewind or overwrite staging memory in use by the worker.
//! - Delayed slot publication ([`Publish`]): a host descriptor slot is only
//!   mirrored to its new texture inside `Host::execute` once the upload copy is
//!   recorded ahead of the title's lists; if the title overwrote the slot or the
//!   texture became host/GPU-written meanwhile, the stale upload is skipped.
//! - No rayon contention: the worker runs detiles sequentially on itself and
//!   never calls `crate::texture::in_parallel`, avoiding work-stealing deadlocks
//!   and contention with `Host::execute`.
//! - Table-bound textures (`Step::Texture`) stay synchronous in `Host::execute`
//!   so UI and explicit draws never flicker.

use std::collections::VecDeque;
use std::sync::atomic::{AtomicBool, AtomicU64, AtomicUsize, Ordering::Relaxed};
use std::sync::{Arc, Condvar, Mutex, MutexGuard, OnceLock};
use std::time::Instant;

use windows::Win32::Graphics::Direct3D12::ID3D12Resource;

use crate::backend::HostMemory;
use crate::descriptor::{Heap, View};
use crate::object::This;
use crate::resource::HostTexture;
use crate::texture::{Staging, Upload, WorkerRefresh};

#[cfg(test)]
thread_local! {
    /// Test-only per-thread override for [`on`]: `None` uses `VICTORY_HEAP_WORKER`,
    /// `Some(b)` forces `b` on this thread without disturbing parallel tests.
    static TEST_ON: std::cell::Cell<Option<bool>> = const { std::cell::Cell::new(None) };
}

/// Whether background heap-indexed texture uploads are enabled
/// (`VICTORY_HEAP_WORKER=1`).
pub(crate) fn on() -> bool {
    #[cfg(test)]
    if let Some(v) = TEST_ON.get() {
        return v;
    }
    static ON: OnceLock<bool> = OnceLock::new();
    *ON.get_or_init(|| crate::switch::on("VICTORY_HEAP_WORKER"))
}

/// Frame-log counters since the last report:
/// `[busy_ns, detile_ns, batches, shown, stale_slot, skipped_written]`.
pub(crate) static STATS: [AtomicU64; 6] = [const { AtomicU64::new(0) }; 6];

/// Flush a ready batch to `Host::execute` once it accumulates this many staging
/// bytes or this much wall time, so newly streamed textures appear promptly.
const BATCH_BYTES: u64 = 48 << 20;
const BATCH_MS: u128 = 40;

/// Soft cap on pre-staged bytes waiting in [`READY`] so unified memory does not
/// grow unboundedly if submissions stall.
const MAX_READY_BYTES: u64 = 192 << 20;

/// Worker-owned staging chunk size and retention bounds.
const WORKER_CHUNK: u64 = 16 << 20;
const WORKER_CHUNKS_KEPT: usize = 16;
const WORKER_OVERSIZE_MIN: u64 = 32 << 20;
const WORKER_OVERSIZE_MAX: u64 = 128 << 20;

/// A descriptor slot to publish once its texture's upload copy is recorded into
/// the submission's upload list.
pub(crate) struct Publish {
    pub(crate) heap: usize,
    pub(crate) index: u32,
    pub(crate) handle: u64,
    pub(crate) key: u64,
}

/// One pre-staged texture upload ready for `Host::execute` to record into its
/// upload list and publish.
pub(crate) struct ReadyUpload {
    pub(crate) resource: usize,
    pub(crate) staging: Arc<Staging>,
    pub(crate) copy: Arc<HostTexture>,
    pub(crate) upload: Option<Upload>,
    pub(crate) publish: Option<Publish>,
}

// SAFETY: `ID3D12Resource`, `Staging`, and `HostTexture` are free-threaded;
// `resource` and `Publish::heap` are kept alive by `Held` / `HeldHeap`.
unsafe impl Send for ReadyUpload {}

/// Owns a COM reference to a title descriptor [`Heap`].
struct HeldHeap(This);

// SAFETY: only the atomic reference count and locked/immutable heap fields are
// touched from the worker and submitting threads.
unsafe impl Send for HeldHeap {}

impl HeldHeap {
    fn new(heap: &Heap) -> Self {
        let this = std::ptr::from_ref(heap).cast();
        crate::object::add_ref(this);
        HeldHeap(this)
    }

    fn ptr(&self) -> This {
        self.0
    }

    fn heap(&self) -> &Heap {
        crate::descriptor::heap(self.0)
    }
}

impl Drop for HeldHeap {
    fn drop(&mut self) {
        crate::object::release_ref(self.0);
    }
}

/// A batch of pre-staged uploads and the worker staging buffers backing them.
pub(crate) struct ReadyBatch {
    pub(crate) uploads: Vec<ReadyUpload>,
    chunks: Vec<(ID3D12Resource, usize)>,
    oversize: Vec<(u64, ID3D12Resource, usize)>,
    _heap: HeldHeap,
}

// SAFETY: D3D12 resources and persistent upload-buffer mappings are owned
// exclusively by this batch until returned to `WORKER_STAGING` after the
// submission's fence completes.
unsafe impl Send for ReadyBatch {}

/// Drained batches owned by a submission's `Uploads` until `Host::execute`
/// finishes waiting on the GPU fence; dropping returns staging buffers to
/// [`WORKER_STAGING`].
pub(crate) struct ReadyDrain {
    batches: Vec<ReadyBatch>,
}

impl ReadyDrain {
    pub(crate) fn is_empty(&self) -> bool {
        self.batches.is_empty()
    }

    pub(crate) fn iter_uploads(&self) -> impl Iterator<Item = &ReadyUpload> {
        self.batches.iter().flat_map(|b| b.uploads.iter())
    }
}

impl Drop for ReadyDrain {
    fn drop(&mut self) {
        let mut chunks = Vec::new();
        let mut oversize = Vec::new();
        for mut batch in self.batches.drain(..) {
            batch.uploads.clear();
            chunks.append(&mut batch.chunks);
            oversize.append(&mut batch.oversize);
        }
        if !chunks.is_empty() || !oversize.is_empty() {
            return_staging(chunks, oversize);
        }
    }
}

/// Worker-private staging buffer pool, completely separate from
/// `crate::texture`'s synchronous `CHUNKS` and `BUFFERS`.
struct WorkerStaging {
    free_chunks: Vec<(ID3D12Resource, usize)>,
    chunk_peak: crate::recent::Peak,
    free_oversize: VecDeque<(u64, ID3D12Resource, usize)>,
    free_oversize_bytes: u64,
    oversize_peak: crate::recent::Peak,
}

// SAFETY: upload buffers and their persistent CPU mappings are only ever owned
// by one active `Batch`, one `ReadyBatch`, or this pool's free lists.
unsafe impl Send for WorkerStaging {}

static WORKER_STAGING: Mutex<WorkerStaging> = Mutex::new(WorkerStaging {
    free_chunks: Vec::new(),
    chunk_peak: crate::recent::Peak::new(),
    free_oversize: VecDeque::new(),
    free_oversize_bytes: 0,
    oversize_peak: crate::recent::Peak::new(),
});

fn take_worker_chunk() -> Option<(ID3D12Resource, usize)> {
    let existing = lock(&WORKER_STAGING).free_chunks.pop();
    if let Some(chunk) = existing {
        return Some(chunk);
    }
    let (buffer, data) = crate::host::host()?.upload_buffer(WORKER_CHUNK)?;
    crate::host::name(&buffer, || "heap worker staging chunk".into());
    Some((buffer, data as usize))
}

fn take_worker_oversize(size: u64) -> Option<(u64, ID3D12Resource, usize)> {
    let class = size.max(1 << 16).next_power_of_two();
    let existing = {
        let mut pool = lock(&WORKER_STAGING);
        let pos = pool.free_oversize.iter().position(|e| e.0 == class);
        pos.and_then(|i| pool.free_oversize.remove(i))
            .inspect(|_| pool.free_oversize_bytes -= class)
    };
    if let Some(entry) = existing {
        return Some(entry);
    }
    let (buffer, data) = crate::host::host()?.upload_buffer(class)?;
    crate::host::name(&buffer, || "heap worker staging oversize".into());
    Some((class, buffer, data as usize))
}

fn return_staging(
    chunks: Vec<(ID3D12Resource, usize)>,
    oversize: Vec<(u64, ID3D12Resource, usize)>,
) {
    let now = Instant::now();
    let chunk_count = chunks.len();
    let oversize_bytes: u64 = oversize.iter().map(|e| e.0).sum();
    let mut dropped_chunks = Vec::new();
    let mut dropped_oversize = Vec::new();
    {
        let mut pool = lock(&WORKER_STAGING);
        let keep_chunks = pool
            .chunk_peak
            .begin(now, chunk_count)
            .clamp(1, WORKER_CHUNKS_KEPT);
        pool.free_chunks.extend(chunks);
        if pool.free_chunks.len() > keep_chunks {
            dropped_chunks = pool.free_chunks.split_off(keep_chunks);
        }
        let keep_oversize = (pool.oversize_peak.begin(now, oversize_bytes as usize) as u64)
            .clamp(WORKER_OVERSIZE_MIN, WORKER_OVERSIZE_MAX);
        for entry in oversize {
            pool.free_oversize_bytes += entry.0;
            pool.free_oversize.push_back(entry);
        }
        while pool.free_oversize_bytes > keep_oversize
            && let Some(old) = pool.free_oversize.pop_front()
        {
            pool.free_oversize_bytes -= old.0;
            dropped_oversize.push(old);
        }
    }
    // Drop freed COM resources outside `WORKER_STAGING`.
    drop(dropped_chunks);
    drop(dropped_oversize);
}

static READY: Mutex<Vec<ReadyBatch>> = Mutex::new(Vec::new());
static READY_COUNT: AtomicUsize = AtomicUsize::new(0);
static READY_BYTES: AtomicU64 = AtomicU64::new(0);

fn lock<T>(m: &Mutex<T>) -> MutexGuard<'_, T> {
    m.lock().unwrap_or_else(|e| e.into_inner())
}

/// Fast lock-free check used by `Host::execute` to see if pre-staged uploads
/// are waiting to be recorded and published.
pub(crate) fn has_ready() -> bool {
    READY_COUNT.load(Relaxed) > 0
}

/// Drains all pre-staged batches ready for the submission about to run.
/// Called by `bindless::uploads` under `Host::submitted`.
pub(crate) fn take_ready() -> ReadyDrain {
    if READY_COUNT.load(Relaxed) == 0 {
        return ReadyDrain { batches: Vec::new() };
    }
    let batches = {
        let mut ready = lock(&READY);
        READY_COUNT.store(0, Relaxed);
        READY_BYTES.store(0, Relaxed);
        std::mem::take(&mut *ready)
    };
    STATS[2].fetch_add(batches.len() as u64, Relaxed);
    ReadyDrain { batches }
}

/// In-progress batch built on `victory-heap`.
struct Batch {
    heap: This,
    uploads: Vec<ReadyUpload>,
    active: Option<(ID3D12Resource, usize, u64)>,
    chunks: Vec<(ID3D12Resource, usize)>,
    oversize: Vec<(u64, ID3D12Resource, usize)>,
    bytes: u64,
    began: Instant,
}

impl Batch {
    fn new(heap: &Heap) -> Self {
        Batch {
            heap: std::ptr::from_ref(heap).cast(),
            uploads: Vec::new(),
            active: None,
            chunks: Vec::new(),
            oversize: Vec::new(),
            bytes: 0,
            began: Instant::now(),
        }
    }

    /// Allocates 512-byte-aligned staging space owned exclusively by this batch.
    fn alloc(&mut self, size: u64) -> Option<(ID3D12Resource, u64, *mut u8)> {
        if size > WORKER_CHUNK {
            let entry = take_worker_oversize(size)?;
            let out = (entry.1.clone(), 0, entry.2 as *mut u8);
            self.oversize.push(entry);
            return Some(out);
        }
        loop {
            if let Some((buffer, data, used)) = &mut self.active {
                let at = used.next_multiple_of(512);
                if at + size <= WORKER_CHUNK {
                    let out = (buffer.clone(), at, (*data + at as usize) as *mut u8);
                    *used = at + size;
                    return Some(out);
                }
                if let Some((old_buf, old_data, _)) = self.active.take() {
                    self.chunks.push((old_buf, old_data));
                }
            }
            let (buffer, data) = take_worker_chunk()?;
            self.active = Some((buffer, data, 0));
        }
    }

    fn push(&mut self, upload: ReadyUpload, staging_bytes: u64) {
        if self.uploads.is_empty() {
            self.began = Instant::now();
        }
        self.uploads.push(upload);
        self.bytes += staging_bytes;
        if self.bytes >= BATCH_BYTES || self.began.elapsed().as_millis() >= BATCH_MS {
            self.flush();
        }
    }

    fn flush(&mut self) {
        if let Some((buf, data, _)) = self.active.take() {
            self.chunks.push((buf, data));
        }
        let uploads = std::mem::take(&mut self.uploads);
        let chunks = std::mem::take(&mut self.chunks);
        let oversize = std::mem::take(&mut self.oversize);
        let bytes = std::mem::replace(&mut self.bytes, 0);
        if uploads.is_empty() {
            if !chunks.is_empty() || !oversize.is_empty() {
                return_staging(chunks, oversize);
            }
            return;
        }
        let heap = HeldHeap::new(crate::descriptor::heap(self.heap));
        let mut ready = lock(&READY);
        ready.push(ReadyBatch {
            uploads,
            chunks,
            oversize,
            _heap: heap,
        });
        READY_COUNT.store(ready.len(), Relaxed);
        READY_BYTES.fetch_add(bytes, Relaxed);
    }
}

impl Drop for Batch {
    fn drop(&mut self) {
        self.flush();
    }
}

/// Pending heaps for the worker to sweep, plus its wakeup condvar.
struct Job {
    heaps: Mutex<Vec<HeldHeap>>,
    wake: Condvar,
}

static JOB: OnceLock<Job> = OnceLock::new();
/// Held by whoever runs the texture pass (the worker, or an urgent recording
/// thread preempting it via [`urgent`]).
static PASS: Mutex<()> = Mutex::new(());
/// Set while an urgent caller is waiting for [`PASS`].
static URGENT: AtomicBool = AtomicBool::new(false);

/// Wakes the background worker to sweep `heap`. Returns `false` if the worker
/// is disabled or could not be spawned (the caller falls back synchronously).
///
/// Only accepts heaps whose `HostHeap` already exists (`made_host()`), so the
/// worker can never trigger `HostHeap::new()` or `Host::quiet()`.
pub(crate) fn kick(heap: &Heap) -> bool {
    if !on() || heap.made_host().is_none() {
        return false;
    }
    static STARTED: OnceLock<bool> = OnceLock::new();
    let started = *STARTED.get_or_init(|| {
        JOB.get_or_init(|| Job {
            heaps: Mutex::new(Vec::new()),
            wake: Condvar::new(),
        });
        match std::thread::Builder::new()
            .name("victory-heap".into())
            .spawn(worker)
        {
            Ok(_) => {
                tracing::info!(
                    "heap-indexed textures staged on background thread (VICTORY_HEAP_WORKER)"
                );
                true
            }
            Err(e) => {
                tracing::warn!(error = %e, "could not start victory-heap worker thread");
                false
            }
        }
    });
    let Some(job) = JOB.get().filter(|_| started) else {
        return false;
    };
    let this: This = std::ptr::from_ref(heap).cast();
    let mut heaps = lock(&job.heaps);
    if !heaps.iter().any(|h| h.ptr() == this) {
        heaps.push(HeldHeap::new(heap));
    }
    job.wake.notify_one();
    true
}

/// Preempts the worker for an urgent synchronous caller (e.g. a one-time bake
/// draw on a recording thread): the worker yields [`PASS`] and flushes what it
/// has staged so far.
#[allow(dead_code)] // used by tests and Phase 2 bake-source pre-upload
pub(crate) fn urgent() -> MutexGuard<'static, ()> {
    if let Ok(pass) = PASS.try_lock() {
        return pass;
    }
    URGENT.store(true, Relaxed);
    let pass = lock(&PASS);
    URGENT.store(false, Relaxed);
    pass
}

/// Whether the worker must pause and hand [`PASS`] to an urgent caller.
pub(crate) fn make_way() -> bool {
    URGENT.load(Relaxed)
}

/// Periodic frame-log summary for `VICTORY_HEAP_WORKER`.
#[cfg(feature = "diagnostics")]
pub(crate) fn log(frames: f64) {
    let [busy, detiling, batches, shown, stale, skipped] =
        std::array::from_fn::<f64, 6, _>(|i| STATS[i].swap(0, Relaxed) as f64 / frames);
    if busy > 0.0 || batches > 0.0 || shown > 0.0 {
        tracing::info!(
            "heap worker (VICTORY_HEAP_WORKER), per frame: {:.1} ms busy ({:.1} ms detiling); \
             {batches:.1} batches, {shown:.0} slots published, {stale:.1} overwritten before publish, \
             {skipped:.1} skipped (host/GPU written or evicted)",
            busy / 1e6,
            detiling / 1e6,
        );
    }
}

fn worker() {
    let Some(job) = JOB.get() else {
        return;
    };
    loop {
        let heap = {
            let mut heaps = lock(&job.heaps);
            loop {
                if let Some(next) = heaps.pop() {
                    break next;
                }
                heaps = job.wake.wait(heaps).unwrap_or_else(|e| e.into_inner());
            }
        };
        if READY_BYTES.load(Relaxed) >= MAX_READY_BYTES {
            std::thread::sleep(std::time::Duration::from_millis(4));
            let mut heaps = lock(&job.heaps);
            if !heaps.iter().any(|h| h.ptr() == heap.ptr()) {
                heaps.push(heap);
            }
            continue;
        }
        let started = Instant::now();
        {
            let _pass = lock(&PASS);
            process_heap(heap.heap());
        }
        STATS[0].fetch_add(started.elapsed().as_nanos() as u64, Relaxed);
        while make_way() {
            std::thread::yield_now();
        }
    }
}

/// Sweeps due slots from `heap`'s `UploadQueue` and dirty `Viewed::textures`,
/// staging uploads into `Batch` without touching `Host::submitted`,
/// `SHADER_HEAPS`, or `crate::texture::workers`.
fn process_heap(heap: &Heap) {
    let Some(host) = heap.made_host() else {
        return;
    };
    let frame = crate::queue::presented();
    static LAST_PASS_MS: AtomicU64 = AtomicU64::new(0);
    let now_ms = now_millis();
    let frame_ms = now_ms
        .saturating_sub(LAST_PASS_MS.swap(now_ms, Relaxed))
        .max(100);
    let max_wait = crate::descriptor::host::relook_frames(
        frame_ms,
        crate::descriptor::host::relook_ms(),
    );
    let mut batch = Batch::new(heap);
    let mut handled_resources = std::collections::HashSet::new();

    // 1. Queued texture slots: Tier-0 GPU read-marks first, then PR #1246 fair
    //    1:1 interleaving of retries and fresh slots.
    let due = host.texture_slots_due(frame, 4096);
    for (index, write) in due {
        if make_way() || READY_BYTES.load(Relaxed) >= MAX_READY_BYTES {
            break;
        }
        let Some(handle) = heap.handle(index) else {
            host.texture_slot_done(index, write);
            continue;
        };
        let Some((view, key)) = crate::descriptor::view_and_key(handle) else {
            host.texture_slot_done(index, write);
            continue;
        };
        let (resource, _held) = match &*view {
            View::Placed(false, ..) => {
                let Some(View::Shader(r, _)) = crate::descriptor::host::placed_texture(&view) else {
                    host.texture_slot_done(index, write);
                    continue;
                };
                // Placed textures made by `texture_at` are never freed, but
                // holding them keeps the resource invariant uniform.
                let held = crate::resource::hold(r as This);
                (r as This, held)
            }
            View::Shader(object, _) => {
                if crate::resource::buffer_bytes(*object as This).is_some() {
                    host.texture_slot_done(index, write);
                    continue;
                }
                let Some(held) = crate::resource::hold(*object as This) else {
                    host.texture_slot_done(index, write);
                    continue;
                };
                (*object as This, Some(held))
            }
            _ => {
                host.texture_slot_done(index, write);
                continue;
            }
        };
        let Some(copy) = crate::resource::host_copy(resource) else {
            host.texture_slot_done(index, write);
            continue;
        };
        if !copy.title_written() || copy.host_written.load(Relaxed) || copy.is_gpu_written() {
            // Host/GPU-written texture: publish slot if unmade, no CPU upload.
            if !host.texture_slot_made_from(index, handle)
                && let Some(staging) = crate::texture::staging(resource, &copy.texture)
            {
                batch.push(
                    ReadyUpload {
                        resource: resource as usize,
                        staging,
                        copy,
                        upload: None,
                        publish: Some(Publish {
                            heap: std::ptr::from_ref(heap) as usize,
                            index,
                            handle,
                            key,
                        }),
                    },
                    0,
                );
            }
            host.texture_slot_done(index, write);
            continue;
        }
        let Some(staging) = crate::texture::staging(resource, &copy.texture) else {
            host.texture_slot_done(index, write);
            continue;
        };
        let detile_start = Instant::now();
        let refreshed = staging.refresh_for_worker(&mut |sz| batch.alloc(sz));
        STATS[1].fetch_add(detile_start.elapsed().as_nanos() as u64, Relaxed);
        match refreshed {
            WorkerRefresh::Uploaded(upload) => {
                handled_resources.insert(resource as usize);
                let sz = staging.size_bytes();
                batch.push(
                    ReadyUpload {
                        resource: resource as usize,
                        staging,
                        copy,
                        upload: Some(upload),
                        publish: Some(Publish {
                            heap: std::ptr::from_ref(heap) as usize,
                            index,
                            handle,
                            key,
                        }),
                    },
                    sz,
                );
                host.texture_slot_done(index, write);
            }
            WorkerRefresh::Unchanged => {
                handled_resources.insert(resource as usize);
                if !host.texture_slot_made_from(index, handle) {
                    batch.push(
                        ReadyUpload {
                            resource: resource as usize,
                            staging,
                            copy,
                            upload: None,
                            publish: Some(Publish {
                                heap: std::ptr::from_ref(heap) as usize,
                                index,
                                handle,
                                key,
                            }),
                        },
                        0,
                    );
                }
                host.texture_slot_done(index, write);
            }
            WorkerRefresh::Unreadable => {
                host.texture_slot_later(index, write, frame, max_wait);
            }
        }
    }

    // 2. Refresh already-published `used_recently()` textures whose title memory
    //    changed (e.g. streamed mip arrivals or dynamic CPU-updated textures).
    for r in host.viewed.texture_resources() {
        if make_way() || READY_BYTES.load(Relaxed) >= MAX_READY_BYTES {
            break;
        }
        if !handled_resources.insert(r) {
            continue;
        }
        let Some(_held) = crate::resource::hold(r as This) else {
            continue;
        };
        let Some(copy) = crate::resource::made_copy(r as This) else {
            continue;
        };
        if !copy.used_recently()
            || !copy.title_written()
            || copy.host_written.load(Relaxed)
            || copy.is_gpu_written()
        {
            continue;
        }
        let Some(staging) = crate::texture::staging(r as This, &copy.texture) else {
            continue;
        };
        if !staging.maybe_changed() {
            continue;
        }
        let detile_start = Instant::now();
        let refreshed = staging.refresh_for_worker(&mut |sz| batch.alloc(sz));
        STATS[1].fetch_add(detile_start.elapsed().as_nanos() as u64, Relaxed);
        if let WorkerRefresh::Uploaded(upload) = refreshed {
            let sz = staging.size_bytes();
            batch.push(
                ReadyUpload {
                    resource: r,
                    staging,
                    copy,
                    upload: Some(upload),
                    publish: None,
                },
                sz,
            );
        }
    }
}

fn now_millis() -> u64 {
    static START: OnceLock<Instant> = OnceLock::new();
    START.get_or_init(Instant::now).elapsed().as_millis() as u64
}

/// Held by a test that enables the worker, and by GPU tests that must not run
/// beside one: `kick_worker_heaps` wakes the worker for every shader-visible
/// heap, so a test's heap swept by it loses its tracked page changes.
#[cfg(test)]
pub(crate) fn serial() -> MutexGuard<'static, ()> {
    static TEST_LOCK: Mutex<()> = Mutex::new(());
    lock(&TEST_LOCK)
}

/// Test helper: temporarily enables `heap_worker` on the calling thread and
/// resets its queues cleanly on drop.
#[cfg(test)]
pub(crate) struct EnableForTest {
    prev: Option<bool>,
    _guard: MutexGuard<'static, ()>,
}

#[cfg(test)]
impl EnableForTest {
    pub(crate) fn new() -> Self {
        let guard = serial();
        let prev = TEST_ON.replace(Some(true));
        // Drain any stale entries left by an earlier test.
        drop(take_ready());
        EnableForTest {
            prev,
            _guard: guard,
        }
    }

    /// Runs `process_heap` synchronously on the caller thread (under `PASS`) so
    /// GPU integration tests can assert exact publication/upload ordering
    /// without racing wall-clock sleeps.
    pub(crate) fn step_heap(&self, heap: &Heap) {
        let _pass = lock(&PASS);
        process_heap(heap);
    }
}

#[cfg(test)]
impl Drop for EnableForTest {
    fn drop(&mut self) {
        drop(take_ready());
        TEST_ON.set(self.prev);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::atomic::AtomicU32;
    use std::time::Duration;

    /// Urgent preemption hands `PASS` from a busy worker loop to the urgent
    /// caller promptly without deadlock.
    #[test]
    fn urgent_preempts_worker_without_deadlock() {
        let stop = Arc::new(AtomicBool::new(false));
        let yields = Arc::new(AtomicU32::new(0));
        let stop_w = stop.clone();
        let yields_w = yields.clone();
        let handle = std::thread::spawn(move || {
            while !stop_w.load(Relaxed) {
                {
                    let _pass = lock(&PASS);
                    for _ in 0..500 {
                        if make_way() {
                            yields_w.fetch_add(1, Relaxed);
                            break;
                        }
                        std::thread::sleep(Duration::from_micros(50));
                    }
                }
                while make_way() {
                    std::thread::yield_now();
                }
            }
        });
        // Give the worker a moment to enter `PASS`.
        std::thread::sleep(Duration::from_millis(5));
        for _ in 0..8 {
            let started = Instant::now();
            let _guard = urgent();
            assert!(
                started.elapsed() < Duration::from_secs(2),
                "urgent() took too long to preempt worker"
            );
            drop(_guard);
            std::thread::sleep(Duration::from_millis(2));
        }
        stop.store(true, Relaxed);
        handle.join().unwrap();
        assert!(
            yields.load(Relaxed) > 0,
            "worker should have observed make_way()"
        );
    }

    /// Worker staging chunks are isolated from `crate::texture::recycle`:
    /// recycling the synchronous pools never resets or drops a worker chunk
    /// held by an active `ReadyDrain`, and returning chunks shrinks back after
    /// `recent::WINDOW`.
    #[test]
    fn worker_staging_leases_isolated_from_recycle() {
        let Some(_host) = crate::host::host() else {
            return;
        };
        let chunk = take_worker_chunk().expect("worker chunk");
        let raw = windows::core::Interface::as_raw(&chunk.0) as usize;
        // Rewinding the synchronous pools must not affect `WORKER_STAGING`.
        crate::texture::recycle();
        {
            let pool = lock(&WORKER_STAGING);
            assert!(
                !pool
                    .free_chunks
                    .iter()
                    .any(|(b, _)| windows::core::Interface::as_raw(b) as usize == raw),
                "checked-out worker chunk must stay exclusively owned across texture::recycle()"
            );
        }
        return_staging(vec![chunk], Vec::new());
        {
            let pool = lock(&WORKER_STAGING);
            assert!(
                pool.free_chunks
                    .iter()
                    .any(|(b, _)| windows::core::Interface::as_raw(b) as usize == raw),
                "returned worker chunk should be available for reuse"
            );
        }
    }
}
