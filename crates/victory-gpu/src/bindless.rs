//! Views shaders reach through the resource heap itself (decision 0007,
//! "Heap-indexed views"). Grand Theft Auto VI is largely bindless: its
//! shaders load descriptors at indices they take from root constants or
//! memory (`HostShader::heap`), so no draw names the views it reads. Every
//! view the title writes into a shader-visible heap gets its host view when
//! it is written (`descriptor::host`), and each heap keeps here what its
//! slots view that title memory feeds: ranges of host buffer copies, and
//! textures whose contents the title writes. A submission whose lists run
//! such shaders first brings all of them up to date (what changed since,
//! `crate::watch`).

use std::collections::{BTreeMap, BTreeSet, HashMap};
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering::{Acquire, Relaxed, Release}};
use std::sync::{Arc, Mutex};

use crate::backend::HostGpu;
use crate::census_fast::{Snapshot, SnapshotCache, UnorderedIndex};
use crate::descriptor::View;
use crate::host::HostList;
use crate::object::This;
use crate::shadow::Shadow;

/// What a title heap slot views that title memory feeds.
#[derive(Clone)]
enum Seen {
    /// `len` bytes at `address` of a buffer's host copy; unordered access.
    Buffer(Arc<Shadow>, u64, u64, bool),
    /// A title texture (resource object) whose contents the title writes.
    Texture(usize),
    /// A view left without a host view (textures the title placed itself),
    /// by kind (diagnostics: `KINDS`), with a placed texture's address and
    /// console bytes.
    Left(#[cfg_attr(not(feature = "diagnostics"), allow(dead_code))] (usize, u64, u64)),
}

/// Kinds of views left without a host view, for the census.
#[cfg(feature = "diagnostics")]
const KINDS: [&str; 4] = ["buffer", "unordered buffer", "placed texture", "other"];

/// A heap's slots that view title memory, and the distinct buffer ranges
/// and textures among them (with how many slots view each).
pub(crate) struct Viewed {
    inner: Mutex<Inner>,
    /// Generation-cached widening snapshot when `VICTORY_CENSUS_FAST=1`.
    fast: Option<SnapshotCache<Shadow>>,
    /// Monotonic generation bumped whenever `inner.hulls` changes, so
    /// submission batching can skip repeat `refresh` walks of an unchanged heap.
    hull_generation: AtomicU64,
}

impl Default for Viewed {
    fn default() -> Self {
        Self::new(crate::census_fast::on())
    }
}

impl Viewed {
    /// Creates a bindless registry, optionally enabling the fast UAV range
    /// index and snapshot cache (`VICTORY_CENSUS_FAST=1`).
    pub(crate) fn new(fast: bool) -> Self {
        Viewed {
            inner: Mutex::new(Inner::new(fast)),
            fast: fast.then(SnapshotCache::default),
            hull_generation: AtomicU64::new(0),
        }
    }

    /// Monotonic generation bumped whenever `inner.hulls` changes.
    pub(crate) fn hull_generation(&self) -> u64 {
        self.hull_generation.load(Acquire)
    }
}

type Range = (usize, u64, u64, bool);

/// A merged range: copy, start, bytes, unordered access, the copy's bit in
/// the change tracker (`crate::watch::clean_bit`).
type Hull = (Arc<Shadow>, u64, u64, bool, Option<u32>);

/// One non-empty 64-tracker word of `watch::DIRTY` referenced by `Inner::hulls`,
/// holding the bitmask of active tracker bits and the sorted `(bit_offset, hull_idx)`
/// entries indexing directly into `Inner::hulls`.
struct ActiveWord {
    word: u32,
    mask: u64,
    entries: Vec<(u8, u32)>,
}

struct Inner {
    slots: HashMap<u32, Seen>,
    ranges: BTreeMap<Range, (u32, Arc<Shadow>)>,
    /// Per title texture, the slots viewing it: an entry names its slots, so
    /// releasing the texture ([`Inner::forget`]) does not search the heap.
    textures: BTreeMap<usize, BTreeSet<u32>>,
    /// `ranges` merged per copy and kind ([`Viewed::ranges`]), kept current
    /// as ranges come and go (each list running heap shaders asks for all
    /// of them; rebuilt whole after every change, they cost 13-27 ms a frame,
    /// run perf4). Stored densely so `refresh` indexes directly without hashing.
    hulls: Vec<Hull>,
    /// Maps `(copy_ptr, uav)` to its dense index in `hulls`.
    hull_pos: HashMap<(usize, bool), u32>,
    /// Sorted by `word`; lets `refresh` test 64 copies per atomic word load
    /// and index `hulls` directly with zero hash lookups.
    active_words: Vec<ActiveWord>,
    /// Indices into `hulls` whose `Shadow` has no `watch::DIRTY` bit
    /// (`bit == None`: untracked buffers and zero-copy aliased shadows).
    untracked_hulls: Vec<u32>,
    /// Per copy and kind (unordered access): the copy, and how many ranges
    /// start and end at each offset, so that the hulls are made from the
    /// copies, not from every range (the title rewrites slots between most
    /// lists, and remerging the world's 130,000 ranges each time cost 0.15
    /// s per frame).
    ends: HashMap<(usize, bool), Ends>,
    /// Secondary index of active UAV ranges only (`VICTORY_CENSUS_FAST=1`).
    unordered: Option<UnorderedIndex<Shadow>>,
}

impl Default for Inner {
    fn default() -> Self {
        Self::new(false)
    }
}

impl Inner {
    fn new(fast: bool) -> Self {
        Inner {
            slots: HashMap::new(),
            ranges: BTreeMap::new(),
            textures: BTreeMap::new(),
            hulls: Vec::new(),
            hull_pos: HashMap::new(),
            active_words: Vec::new(),
            untracked_hulls: Vec::new(),
            ends: HashMap::new(),
            unordered: fast.then(UnorderedIndex::new),
        }
    }
}

/// A copy, and the counts of its ranges' starts and ends.
type Ends = (Arc<Shadow>, BTreeMap<u64, u32>, BTreeMap<u64, u32>);

/// One more (`add`) or one fewer of `at` in `counts`.
fn tally(counts: &mut BTreeMap<u64, u32>, at: u64, add: bool) {
    let n = counts.entry(at).or_insert(0);
    *n = if add { *n + 1 } else { n.saturating_sub(1) };
    if *n == 0 {
        counts.remove(&at);
    }
}

impl Inner {
    fn insert_hull_index(&mut self, hull_idx: u32, bit: Option<u32>) {
        match bit {
            Some(bit) => {
                let word = bit / 64;
                let off = (bit % 64) as u8;
                let mask = 1u64 << off;
                match self.active_words.binary_search_by_key(&word, |w| w.word) {
                    Ok(i) => {
                        let aw = &mut self.active_words[i];
                        aw.mask |= mask;
                        let pos = aw.entries.partition_point(|&(o, _)| o < off);
                        aw.entries.insert(pos, (off, hull_idx));
                    }
                    Err(i) => {
                        self.active_words.insert(
                            i,
                            ActiveWord {
                                word,
                                mask,
                                entries: vec![(off, hull_idx)],
                            },
                        );
                    }
                }
            }
            None => self.untracked_hulls.push(hull_idx),
        }
    }

    fn remove_hull_index(&mut self, hull_idx: u32, bit: Option<u32>) {
        match bit {
            Some(bit) => {
                let word = bit / 64;
                let off = (bit % 64) as u8;
                if let Ok(i) = self.active_words.binary_search_by_key(&word, |w| w.word) {
                    let aw = &mut self.active_words[i];
                    if let Some(pos) = aw
                        .entries
                        .iter()
                        .position(|&(o, idx)| o == off && idx == hull_idx)
                    {
                        aw.entries.remove(pos);
                    }
                    if !aw.entries.iter().any(|&(o, _)| o == off) {
                        aw.mask &= !(1u64 << off);
                    }
                    if aw.entries.is_empty() {
                        self.active_words.remove(i);
                    }
                }
            }
            None => {
                if let Some(pos) = self.untracked_hulls.iter().position(|&idx| idx == hull_idx) {
                    self.untracked_hulls.swap_remove(pos);
                }
            }
        }
    }

    fn move_hull_index(&mut self, bit: Option<u32>, old_idx: u32, new_idx: u32) {
        match bit {
            Some(bit) => {
                let word = bit / 64;
                let off = (bit % 64) as u8;
                if let Ok(i) = self.active_words.binary_search_by_key(&word, |w| w.word) {
                    let aw = &mut self.active_words[i];
                    let start = aw.entries.partition_point(|&(o, _)| o < off);
                    for entry in &mut aw.entries[start..] {
                        if entry.0 != off {
                            break;
                        }
                        if entry.1 == old_idx {
                            entry.1 = new_idx;
                            break;
                        }
                    }
                }
            }
            None => {
                if let Some(slot) = self.untracked_hulls.iter_mut().find(|&&mut idx| idx == old_idx)
                {
                    *slot = new_idx;
                }
            }
        }
    }

    /// Updates range, hull, texture, and (when enabled) UAV-index counts for
    /// one slot transition. Returns `(unordered_changed, hulls_changed)`.
    fn count(&mut self, index: u32, seen: &Seen, add: bool) -> (bool, bool) {
        let step = |n: &mut u32| if add { *n += 1 } else { *n = n.saturating_sub(1) };
        let mut unordered_changed = false;
        let mut hulls_changed = false;
        match seen {
            Seen::Buffer(shadow, address, len, uav) => {
                let key = (Arc::as_ptr(shadow) as usize, *address, address + len, *uav);
                let e = self.ranges.entry(key).or_insert_with(|| (0, shadow.clone()));
                let was = e.0;
                step(&mut e.0);
                let now = e.0;
                if now == 0 {
                    self.ranges.remove(&key);
                }
                if (was == 0) != (now == 0)
                    && key.3
                    && let Some(unordered) = &mut self.unordered
                {
                    if now > 0 {
                        unordered.insert(key, shadow);
                    } else {
                        unordered.remove(&key);
                    }
                    unordered_changed = true;
                }
                // A range came or went: its copy's hull may change.
                if (was == 0) != (now == 0) {
                    let hull_key = (key.0, key.3);
                    let ends = self
                        .ends
                        .entry(hull_key)
                        .or_insert_with(|| (shadow.clone(), BTreeMap::new(), BTreeMap::new()));
                    tally(&mut ends.1, key.1, now > 0);
                    tally(&mut ends.2, key.2, now > 0);
                    let hull = match (ends.1.keys().next(), ends.2.keys().next_back()) {
                        (Some(&from), Some(&to)) => {
                            Some((shadow.clone(), from, to - from, key.3, shadow.dirty_bit()))
                        }
                        _ => None,
                    };
                    match hull {
                        Some(hull) => {
                            if let Some(&idx) = self.hull_pos.get(&hull_key) {
                                self.hulls[idx as usize] = hull;
                            } else {
                                let bit = hull.4;
                                let idx = self.hulls.len() as u32;
                                self.hulls.push(hull);
                                self.hull_pos.insert(hull_key, idx);
                                self.insert_hull_index(idx, bit);
                            }
                        }
                        None => {
                            self.ends.remove(&hull_key);
                            if let Some(idx) = self.hull_pos.remove(&hull_key) {
                                let last_idx = (self.hulls.len() - 1) as u32;
                                let removed = self.hulls.swap_remove(idx as usize);
                                self.remove_hull_index(idx, removed.4);
                                if idx != last_idx {
                                    let moved = &self.hulls[idx as usize];
                                    let moved_key = (Arc::as_ptr(&moved.0) as usize, moved.3);
                                    let moved_bit = moved.4;
                                    self.hull_pos.insert(moved_key, idx);
                                    self.move_hull_index(moved_bit, last_idx, idx);
                                }
                            }
                        }
                    }
                    hulls_changed = true;
                }
            }
            Seen::Texture(r) => {
                let slots = self.textures.entry(*r).or_default();
                if add {
                    slots.insert(index);
                } else {
                    slots.remove(&index);
                }
                if slots.is_empty() {
                    self.textures.remove(r);
                }
            }
            Seen::Left(_) => {}
        }
        (unordered_changed, hulls_changed)
    }

    /// The title released the texture `resource`: the slots viewing it go.
    fn forget(&mut self, resource: usize) {
        if let Some(slots) = self.textures.remove(&resource) {
            #[cfg(feature = "diagnostics")]
            {
                FORGOTTEN[0].fetch_add(1, Relaxed);
                FORGOTTEN[1].fetch_add(slots.len() as u64, Relaxed);
            }
            for index in slots {
                self.slots.remove(&index);
            }
        }
    }
}

impl Viewed {
    /// Title slot `index` now holds `view` (none: no view), made on the
    /// host (`made`) or left without a host view.
    pub(crate) fn set(&self, index: u32, view: Option<&View>, made: bool) {
        let seen = view.and_then(|v| if made { seen(v) } else { Some(Seen::Left(left_kind(v))) });
        let mut inner = self.inner.lock().unwrap_or_else(|e| e.into_inner());
        let (mut unordered_changed, mut hulls_changed) = (false, false);
        if let Some(old) = inner.slots.remove(&index) {
            let (u, h) = inner.count(index, &old, false);
            unordered_changed |= u;
            hulls_changed |= h;
        }
        if let Some(seen) = seen {
            let (u, h) = inner.count(index, &seen, true);
            unordered_changed |= u;
            hulls_changed |= h;
            inner.slots.insert(index, seen);
        }
        if hulls_changed {
            self.hull_generation.fetch_add(1, Release);
        }
        if unordered_changed
            && let Some(fast) = &self.fast
        {
            fast.invalidate();
        }
    }

    /// Batched [`Viewed::set`]: computes each unique view's entry outside the
    /// lock (deduplicating repeated view pointers) and applies all slot
    /// updates under one hold of `self.inner`.
    #[cfg(windows)]
    pub(crate) fn set_batch(&self, updates: &[(u32, Option<&View>, bool)]) {
        if updates.is_empty() {
            return;
        }
        let mut cache: HashMap<(usize, bool), Option<Seen>> = HashMap::new();
        let mut entries: Vec<(u32, Option<Seen>)> = Vec::with_capacity(updates.len());
        for &(index, view, made) in updates {
            let entry = match view {
                None => None,
                Some(v) => {
                    let key = (std::ptr::from_ref(v) as usize, made);
                    cache
                        .entry(key)
                        .or_insert_with(|| {
                            if made {
                                seen(v)
                            } else {
                                Some(Seen::Left(left_kind(v)))
                            }
                        })
                        .clone()
                }
            };
            entries.push((index, entry));
        }
        let mut inner = self.inner.lock().unwrap_or_else(|e| e.into_inner());
        let (mut unordered_changed, mut hulls_changed) = (false, false);
        for (index, entry) in entries {
            if let Some(old) = inner.slots.remove(&index) {
                let (u, h) = inner.count(index, &old, false);
                unordered_changed |= u;
                hulls_changed |= h;
            }
            if let Some(entry) = entry {
                let (u, h) = inner.count(index, &entry, true);
                unordered_changed |= u;
                hulls_changed |= h;
                inner.slots.insert(index, entry);
            }
        }
        if hulls_changed {
            self.hull_generation.fetch_add(1, Release);
        }
        if unordered_changed
            && let Some(fast) = &self.fast
        {
            fast.invalidate();
        }
    }

    /// A slot whose unordered view reaches `len` bytes of `shadow` at
    /// `address` (tests: no view or resource needed).
    #[cfg(test)]
    pub(crate) fn add_unordered(&self, shadow: &Arc<Shadow>, address: u64, len: u64) {
        let seen = Seen::Buffer(shadow.clone(), address, len, true);
        let (unordered_changed, hulls_changed) = self
            .inner
            .lock()
            .unwrap_or_else(|e| e.into_inner())
            .count(u32::MAX, &seen, true); // no slot: only textures name theirs
        if hulls_changed {
            self.hull_generation.fetch_add(1, Release);
        }
        if unordered_changed
            && let Some(fast) = &self.fast
        {
            fast.invalidate();
        }
    }

    /// The buffer ranges views reach, merged per host copy and kind
    /// (shader resource, unordered access): copy, start, bytes.
    #[cfg(any(test, feature = "diagnostics"))]
    fn ranges(&self) -> Vec<Hull> {
        self.inner.lock().unwrap_or_else(|e| e.into_inner()).hulls.clone()
    }

    /// Snapshot of distinct title texture resources currently viewed by this
    /// heap, for the background heap worker.
    pub(crate) fn texture_resources(&self) -> Vec<usize> {
        self.inner
            .lock()
            .unwrap_or_else(|e| e.into_inner())
            .textures
            .keys()
            .copied()
            .collect()
    }

    /// Whether any title-written texture is currently viewed by this heap.
    pub(crate) fn has_textures(&self) -> bool {
        !self
            .inner
            .lock()
            .unwrap_or_else(|e| e.into_inner())
            .textures
            .is_empty()
    }
}

/// What a made view reads that title memory feeds.
fn seen(view: &View) -> Option<Seen> {
    let placed = crate::descriptor::host::placed_texture(view);
    let view = placed.as_ref().unwrap_or(view);
    if let View::Shader(resource, _) = view
        && let Some(texture) = crate::resource::made_copy(*resource as This)
    {
        return texture.title_written().then_some(Seen::Texture(*resource));
    }
    // The copy the host view reads.
    let (shadow, address, len) = match *view {
        View::Constant(address, size) => {
            let len = u64::from(size);
            ((len > 0).then(|| crate::shadow::find(address, len)).flatten()?, address, len)
        }
        _ => crate::descriptor::host::view_copy(view)?,
    };
    let uav = matches!(view, View::Unordered(..) | View::Placed(true, ..));
    if len == 0 {
        return None;
    }
    let len = len.min(shadow.left(address));
    Some(Seen::Buffer(shadow, address, len, uav))
}

fn left_kind(view: &View) -> (usize, u64, u64) {
    match view {
        View::Placed(_, resource, desc) if resource[0] != 1 => {
            let address = u64::from(desc[10]) | u64::from(desc[11]) << 32;
            (2, address, crate::resource::placed_size(resource).unwrap_or(0))
        }
        View::Unordered(..) | View::Placed(true, ..) => (1, 0, 0),
        View::Shader(..) | View::Placed(false, ..) | View::Constant(..) => (0, 0, 0),
        _ => (3, 0, 0),
    }
}

/// At submission, before anything else of it: the textures the heaps of its
/// heap-indexing lists view, and `textures` (staging and host texture of
/// those its lists read through tables, `crate::texture::upload`), are
/// brought up to date, those whose title memory changed copied into their
/// host textures by a list of ours, which the submission runs first (and
/// leaves them readable by shaders): each at most once.
///
/// shortcut: a title copy into texture memory recorded in the same
/// submission shows from the next one on.
pub(crate) fn uploads(
    heaps: &[Arc<Viewed>],
    textures: &[(Arc<crate::texture::Staging>, Arc<crate::resource::HostTexture>)],
) -> Option<Uploads> {
    static LIST: Mutex<Holder> = Mutex::new(Holder(None, false));
    let list = {
        let mut guard = LIST.lock().unwrap_or_else(|e| e.into_inner());
        crate::texture::recycle();
        // Nothing to bring up, and the list holds nothing of the last
        // submission to let go of: it stays as it is.
        if heaps.is_empty() && textures.is_empty() && !guard.1 && !crate::heap_worker::has_ready() {
            return None;
        }
        let list = match guard.0.take() {
            // Closed by the last submission, which has run.
            Some(list) => {
                list.reset();
                list
            }
            // Made open.
            None => {
                let list = crate::host::host()?.new_list().ok()?;
                list.set_ours();
                list
            }
        };
        guard.1 = false;
        list
    };
    let mut uploads = Uploads {
        list: Some(list),
        dirty: true,
        pool: &LIST,
        worker: Mutex::new(None),
    };
    let list = uploads.list();
    // What to bring up: the textures lists read through tables, and those
    // the heaps view that shaders read lately. Collected under the heaps'
    // locks and brought up without them, several at a time: reading title
    // memory, detiling and making staging buffers took half the submitting
    // thread in a new shot, and title threads setting views waited on the
    // lock meanwhile (run evict5).
    let mut work: Vec<(Arc<crate::texture::Staging>, Arc<crate::resource::HostTexture>)> = Vec::new();
    let mut staged = std::collections::HashSet::new();
    for (staging, copy) in textures {
        // A host copy wrote it in a submission made before this one (it
        // was not when the list was recorded).
        if copy.host_written.load(Relaxed) {
            continue;
        }
        if staged.insert(Arc::as_ptr(staging)) {
            work.push((staging.clone(), copy.clone()));
        }
    }
    let worker_on = crate::heap_worker::on();
    let mut done = std::collections::HashSet::new();
    if !worker_on {
        for viewed in heaps {
            let inner = viewed.inner.lock().unwrap_or_else(|e| e.into_inner());
            for &r in inner.textures.keys() {
                if !done.insert(r) {
                    continue;
                }
                // Not made (given up, or never read): nothing to bring up.
                let Some(texture) = crate::resource::made_copy(r as This) else {
                    continue;
                };
                // Only what shaders read lately (the eviction pass's stamps):
                // the heaps view thousands of textures a frame never reads, and
                // bringing each up whenever its pages change re-uploaded about
                // 4 GB in 30-46 world frames (runs evict2, noevict1).
                // shortcut: a texture read again after a gap is brought up after
                // the submission that read it, which can see bytes one refresh
                // old (a slot the title rewrites is made fresh anyway); upload on
                // the mark itself if that shows.
                if !texture.used_recently() {
                    continue;
                }
                // Its contents are the host's since a host copy wrote it.
                if texture.host_written.load(Relaxed) {
                    continue;
                }
                if let Some(staging) = crate::texture::staging(r as This, &texture.texture)
                    && staged.insert(Arc::as_ptr(&staging))
                {
                    work.push((staging, texture));
                }
            }
        }
    }
    let mut recorded = false;
    // The textures copied into stay alive in `work` until recorded.
    let mut readable = Vec::new();
    for ((staging, copy), upload) in work.iter().zip(crate::texture::refresh_all(&work)) {
        if let Some(upload) = upload {
            readable.extend(crate::texture::record_upload(list, copy, staging, &upload));
            recorded = true;
            #[cfg(feature = "diagnostics")]
            UPLOADED.fetch_add(1, Relaxed);
        }
    }
    let ready = if worker_on {
        let ready = crate::heap_worker::take_ready();
        for item in ready.iter_uploads() {
            let still_made = crate::resource::made_copy(item.resource as This)
                .is_some_and(|c| Arc::ptr_eq(&c, &item.copy));
            if !still_made {
                crate::heap_worker::STATS[5].fetch_add(1, Relaxed);
                continue;
            }
            if let Some(p) = &item.publish {
                let key_matches =
                    crate::descriptor::view_and_key(p.handle).is_some_and(|(_, k)| k == p.key);
                if !key_matches {
                    crate::heap_worker::STATS[4].fetch_add(1, Relaxed);
                    if item.upload.is_some()
                        && !item.copy.host_written.load(Relaxed)
                        && !item.copy.is_gpu_written()
                    {
                        item.staging.invalidate();
                    }
                    continue;
                }
            }
            if let Some(upload) = &item.upload {
                if item.copy.host_written.load(Relaxed) || item.copy.is_gpu_written() {
                    crate::heap_worker::STATS[5].fetch_add(1, Relaxed);
                } else if staged.insert(Arc::as_ptr(&item.staging)) {
                    readable.extend(crate::texture::record_upload(
                        list,
                        &item.copy,
                        &item.staging,
                        upload,
                    ));
                    item.staging.mark_uploaded_once();
                    recorded = true;
                    #[cfg(feature = "diagnostics")]
                    UPLOADED.fetch_add(1, Relaxed);
                }
            }
            if let Some(p) = &item.publish
                && let Some(host) = crate::descriptor::heap(p.heap as This).made_host()
            {
                let shown = host.publish(p.index, p.handle, p.key);
                crate::heap_worker::STATS[if shown { 3 } else { 4 }].fetch_add(1, Relaxed);
            }
        }
        if !heaps.is_empty() {
            crate::descriptor::kick_worker_heaps();
        }
        (!ready.is_empty() && recorded).then_some(ready)
    } else {
        None
    };
    if !readable.is_empty() {
        // SAFETY: an open list; the barriers name live textures.
        unsafe { list.list.ResourceBarrier(&readable) };
    }
    list.close();
    uploads.dirty = recorded;
    *uploads.worker.get_mut().unwrap_or_else(|e| e.into_inner()) = ready;
    // A list recording nothing is not run (dropping `uploads` returns it to
    // `LIST` with `dirty = false` for the next submission).
    recorded.then_some(uploads)
}

/// The upload list, reused across submissions, and whether it holds what it
/// recorded (the staging it names) until the next reset.
pub(crate) struct Holder(Option<HostList>, bool);
// SAFETY: D3D12 objects are free-threaded; the list is used by one
// submission at a time, checked out under `Host::submitted`.
unsafe impl Send for Holder {}

/// The upload list and worker staging drain of a submission, owned until the
/// submission's GPU fence completes (so Flight-1 overlap can move it to the
/// background finisher without holding a `MutexGuard` across threads or a GPU
/// wait); dropping returns the upload list to its reuse slot.
pub(crate) struct Uploads {
    list: Option<HostList>,
    dirty: bool,
    pool: &'static Mutex<Holder>,
    worker: Mutex<Option<crate::heap_worker::ReadyDrain>>,
}

impl Uploads {
    pub(crate) fn list(&self) -> &HostList {
        self.list.as_ref().expect("made before it is handed out")
    }

    /// Returns worker-owned staging chunks immediately once the submission's
    /// GPU fence has completed.
    pub(crate) fn release_worker(&self) {
        drop(self.worker.lock().unwrap_or_else(|e| e.into_inner()).take());
    }
}

impl Drop for Uploads {
    fn drop(&mut self) {
        self.release_worker();
        if let Some(list) = self.list.take() {
            let mut guard = self.pool.lock().unwrap_or_else(|e| e.into_inner());
            if guard.0.is_none() {
                guard.0 = Some(list);
                guard.1 = self.dirty;
            }
        }
    }
}

/// Per submission: brings the buffer ranges a heap's views reach up to date
/// (in the submission's batch of refreshes): the pages the title changed
/// since their copies took them.
///
/// shortcut: what shaders write through the heap's unordered access views
/// stays in the host copies (later draws read it there) and does not go
/// back to title memory, so the CPU does not see it, and a page the title
/// also writes is copied over from title memory, losing the GPU's bytes
/// there. Copying back the whole unordered ranges after every submission
/// (about 1 GB of them in the world) cost seconds per frame; the upgrade
/// is shaders marking the pages they write (as `crate::memory` does).
///
/// Copies the title has not written since they took it are passed by: a
/// world submission's heaps reach ~9,000 copies, a few of them written.
pub(crate) fn refresh(viewed: &Viewed, mut add: impl FnMut(&Arc<Shadow>, usize, usize)) {
    let emit = |(shadow, address, len, uav, _): &Hull,
                add: &mut dyn FnMut(&Arc<Shadow>, usize, usize)| {
        // Aliased zero-copy shadows have `dirty_bit() == None` and live in
        // `untracked_hulls`, yet report `clean() == true` unless widened:
        // skip `page_range` and `Batch::add` for clean shadows.
        if shadow.clean() {
            return;
        }
        let (address, len) = if *uav && shadow.marked() {
            shadow.page_range(*address, *len)
        } else {
            (*address, *len)
        };
        add(shadow, address as usize, len as usize);
    };
    let inner = viewed.inner.lock().unwrap_or_else(|e| e.into_inner());
    for aw in &inner.active_words {
        let mut dirty = crate::watch::dirty_word(aw.word as usize) & aw.mask;
        if dirty == 0 {
            continue;
        }
        if dirty == aw.mask {
            for &(_, idx) in &aw.entries {
                emit(&inner.hulls[idx as usize], &mut add);
            }
            continue;
        }
        if dirty.count_ones() <= 4 {
            while dirty != 0 {
                let off = dirty.trailing_zeros() as u8;
                dirty &= dirty - 1;
                let start = aw.entries.partition_point(|&(o, _)| o < off);
                for &(o, idx) in &aw.entries[start..] {
                    if o != off {
                        break;
                    }
                    emit(&inner.hulls[idx as usize], &mut add);
                }
            }
        } else {
            for &(off, idx) in &aw.entries {
                if dirty & (1u64 << off) != 0 {
                    emit(&inner.hulls[idx as usize], &mut add);
                }
            }
        }
    }
    for &idx in &inner.untracked_hulls {
        emit(&inner.hulls[idx as usize], &mut add);
    }
}

/// Per submission: the buffer range of the view currently in `index`. A
/// table draw may have been recorded with another view in that slot.
pub(crate) fn refresh_slot(
    viewed: &Viewed,
    index: u32,
    mut add: impl FnMut(&Arc<Shadow>, usize, usize),
) {
    let inner = viewed.inner.lock().unwrap_or_else(|e| e.into_inner());
    let Some(Seen::Buffer(shadow, address, len, _)) = inner.slots.get(&index) else {
        return;
    };
    if !shadow.clean() && !crate::watch::clean_bit(shadow.dirty_bit()) {
        add(shadow, *address as usize, *len as usize);
    }
}

/// Heap UAV ranges that may need to refresh a widened SRV after a later
/// buffer barrier, added to `out` (the caller's buffer, kept from barrier to
/// barrier: the world's heaps have tens of thousands). Computed heap indices
/// do not name ordinary write steps.
pub(crate) fn widened_writes_into(viewed: &Viewed, out: &mut Vec<(Arc<Shadow>, usize, usize)>) {
    if !crate::under_wine() {
        return;
    }
    out.extend(
        viewed
            .inner
            .lock()
            .unwrap_or_else(|e| e.into_inner())
            .ranges
            .iter()
            .filter(|((_, address, end, uav), _)| *uav && address < end)
            .map(|((_, address, end, _), (_, shadow))| {
                (shadow.clone(), *address as usize, (end - address) as usize)
            }),
    );
}

/// Cached UAV buffer ranges for widening (`VICTORY_CENSUS_FAST=1`). Returns
/// `None` when the fast cache is disabled on `viewed`, letting callers fall
/// back to [`widened_writes`].
pub(crate) fn unordered_writes(viewed: &Viewed) -> Option<Snapshot<Shadow>> {
    let fast = viewed.fast.as_ref()?;
    if !crate::under_wine() {
        return Some(Arc::from([]));
    }
    if let Some(cached) = fast.get_cached() {
        return Some(cached);
    }
    let inner = viewed.inner.lock().unwrap_or_else(|e| e.into_inner());
    inner
        .unordered
        .as_ref()
        .map(|unordered| fast.get_or_build(unordered))
}

/// Draws and dispatches run with heap-indexing shaders, those of them
/// writing through the heap, textures uploaded for heaps, released textures
/// a heap viewed and the slots taken with them (diagnostics).
static HEAP_DRAWS: [AtomicU64; 2] = [AtomicU64::new(0), AtomicU64::new(0)];
#[cfg(feature = "diagnostics")]
static UPLOADED: AtomicU64 = AtomicU64::new(0);
#[cfg(feature = "diagnostics")]
static FORGOTTEN: [AtomicU64; 2] = [AtomicU64::new(0), AtomicU64::new(0)];

/// A draw or dispatch runs a shader that indexes the heap (and writes
/// through it).
pub(crate) fn note_draw(writes: bool) {
    HEAP_DRAWS[0].fetch_add(1, Relaxed);
    if writes {
        HEAP_DRAWS[1].fetch_add(1, Relaxed);
        if !HEAP_WRITTEN.swap(true, Relaxed) {
            REMAKE.store(true, Relaxed);
        }
    }
}

/// A shader wrote through a heap's views (ever): textures viewed for
/// unordered access there may hold what the GPU wrote, which no state move
/// shows (`crate::resource::evictable`).
static HEAP_WRITTEN: AtomicBool = AtomicBool::new(false);
/// The unordered views left null are to be made again before the next
/// submission ([`take_remake`]).
static REMAKE: AtomicBool = AtomicBool::new(false);

/// Whether any shader wrote through a heap yet.
pub(crate) fn heap_written() -> bool {
    HEAP_WRITTEN.load(Relaxed)
}

/// Whether the null unordered views are to be made again now (once).
pub(crate) fn take_remake() -> bool {
    REMAKE.swap(false, Relaxed)
}

/// Every view heap's registry.
static HEAPS: Mutex<Vec<std::sync::Weak<Viewed>>> = Mutex::new(Vec::new());

pub(crate) fn register(viewed: &Arc<Viewed>) {
    let mut heaps = HEAPS.lock().unwrap_or_else(|e| e.into_inner());
    heaps.retain(|h| h.strong_count() > 0);
    heaps.push(Arc::downgrade(viewed));
}

/// The title released `resource`: slots viewing it as a texture view
/// nothing to upload any more (its memory may be gone).
pub(crate) fn forget(resource: usize) {
    let heaps = HEAPS.lock().unwrap_or_else(|e| e.into_inner()).clone();
    for viewed in heaps.iter().filter_map(std::sync::Weak::upgrade) {
        let mut inner = viewed.inner.lock().unwrap_or_else(|e| e.into_inner());
        inner.forget(resource);
    }
}

/// Logs, per frame over `frames`, the heap-indexed draws, and what the
/// view heaps' slots view (diagnostics).
#[cfg(feature = "diagnostics")]
pub(crate) fn log(frames: f64) {
    let [draws, writes] = [0, 1].map(|i| HEAP_DRAWS[i].swap(0, Relaxed) as f64 / frames);
    let uploaded = UPLOADED.swap(0, Relaxed) as f64 / frames;
    let [forgotten, taken] = [0, 1].map(|i| FORGOTTEN[i].swap(0, Relaxed) as f64 / frames);
    let heaps = HEAPS.lock().unwrap_or_else(|e| e.into_inner()).clone();
    for viewed in heaps.iter().filter_map(std::sync::Weak::upgrade) {
        let inner = viewed.inner.lock().unwrap_or_else(|e| e.into_inner());
        let mut left = [0u32; KINDS.len()];
        let (mut buffers, mut textures) = (0u32, 0u32);
        // Placed textures left null: the largest at each address.
        let mut placed = HashMap::new();
        for seen in inner.slots.values() {
            match seen {
                Seen::Buffer(..) => buffers += 1,
                Seen::Texture(_) => textures += 1,
                Seen::Left((k, address, bytes)) => {
                    left[*k] += 1;
                    if *k == 2 {
                        let b = placed.entry(*address).or_insert(0u64);
                        *b = (*b).max(*bytes);
                    }
                }
            }
        }
        let placed_mb = placed.values().sum::<u64>() as f64 / 1e6;
        let bytes: u64 = inner.ranges.keys().map(|&(_, a, b, _)| b - a).sum();
        drop(inner);
        let hulls = viewed.ranges();
        let uav_hulls: u64 = hulls.iter().filter(|r| r.3).map(|r| r.2).sum();
        let all_hulls: u64 = hulls.iter().map(|r| r.2).sum();
        let inner = viewed.inner.lock().unwrap_or_else(|e| e.into_inner());
        let copies = {
            let mut c: Vec<usize> = inner.ranges.keys().map(|k| k.0).collect();
            c.dedup();
            c.len()
        };
        let left: Vec<String> = KINDS.iter().zip(left).map(|(k, n)| format!("{k} {n}")).collect();
        let all = crate::shadow::all();
        let all_bytes: u64 = all.iter().map(|s| s.range().1).sum();
        tracing::info!(
            "heap views: {draws:.0} draws/frame index the heap ({writes:.0} writing through it), \
             {uploaded:.0} textures uploaded/frame, {forgotten:.1} released textures/frame viewed in a heap \
             ({taken:.1} slots); slots: {buffers} buffer views ({} ranges, \
             {:.1} MB, {copies} copies), {textures} title-written textures ({} distinct); \
             left null: {} (placed textures at {} addresses, {placed_mb:.0} MB); \
             host copies of title memory: {} ({:.1} MB); hulls {:.1} MB, unordered {:.1} MB",
            inner.ranges.len(),
            bytes as f64 / 1e6,
            inner.textures.len(),
            left.join(", "),
            placed.len(),
            all.len(),
            all_bytes as f64 / 1e6,
            all_hulls as f64 / 1e6,
            uav_hulls as f64 / 1e6
        );
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::object::{HRESULT, S_OK};
    use windows::Win32::Graphics::Direct3D12::*;
    use windows::core::s;

    /// Reads slots 0 (raw buffer) and 2 (texture) of the heap and writes
    /// slot 1 (raw unordered buffer): through a table over the heap's
    /// start, the same slots a shader indexing the heap reads.
    const SOURCE: &str = r#"
ByteAddressBuffer src : register(t0);
RWByteAddressBuffer dst : register(u0);
Texture2D<float4> tex : register(t1);
[numthreads(16, 1, 1)]
void main(uint3 id : SV_DispatchThreadID) {
    uint texel = (uint)round(tex.Load(int3(id.x % 4, 0, 0)).x * 255.0);
    dst.Store(id.x * 4, src.Load(id.x * 4) + texel);
}
"#;

    fn compile() -> (ID3D12RootSignature, ID3D12PipelineState) {
        use crate::compute::{Param, srv, uav};
        let params = &[Param::Table(&[srv(0, 1), uav(0, 1), srv(1, 1)])];
        crate::compute::pipeline(SOURCE, s!("heap"), 0, params).unwrap()
    }

    /// A title resource from a description in the console's layout (56
    /// bytes: dimension, alignment, width, height, depth and mips, format,
    /// samples, layout, flags).
    fn resource(dimension: u32, width: u64, height: u32, format: u32, layout: u32) -> This {
        let desc: [u32; 14] = [
            dimension, 0, 0, 0, width as u32, (width >> 32) as u32, height, 1 | 1 << 16, format, 1, 0,
            layout, 0, 0,
        ];
        let properties = [1u32, 0, 0, 0, 0];
        let mut out = std::ptr::null();
        let iid = crate::object::guid(
            0x696442be,
            0xa72e,
            0x4059,
            [0xbc, 0x79, 0x5b, 0x5c, 0x98, 0x04, 0x0f, 0xad],
        );
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

    /// A title resource at `address` (title memory of its own), as the
    /// title places buffers and textures in its heaps.
    fn resource_at(address: usize, dimension: u32, width: u64, height: u32, format: u32, layout: u32) -> This {
        let desc: [u32; 14] = [
            dimension, 0, 0, 0, width as u32, (width >> 32) as u32, height, 1 | 1 << 16, format, 1, 0,
            layout, 0, 0,
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
            address as u64,
            desc.as_ptr().cast(),
            0,
            std::ptr::null(),
            &iid,
            &mut out,
        );
        assert_eq!(made, S_OK);
        out
    }

    const LINEAR: u32 = 0x100;

    /// A copy's hull is its live ranges' first start to last end, kept as
    /// slots come and go; a copy no slot views any more has none.
    #[test]
    fn hulls_follow_the_ranges() {
        let memory = vec![0u8; 0x1000];
        let at = memory.as_ptr() as u64;
        crate::shadow::register(at as usize, at, 0x1000);
        let Some(copy) = crate::shadow::find(at, 0x1000) else {
            crate::shadow::unregister(at as usize, at);
            return; // no host GPU here
        };
        let viewed = Viewed::default();
        let set = |slot: u32, range: Option<(u64, u64)>| {
            let mut inner = viewed.inner.lock().unwrap();
            if let Some(old) = inner.slots.remove(&slot) {
                inner.count(slot, &old, false);
            }
            if let Some((from, len)) = range {
                let seen = Seen::Buffer(copy.clone(), at + from, len, false);
                inner.count(slot, &seen, true);
                inner.slots.insert(slot, seen);
            }
        };
        let hull = || viewed.ranges().iter().map(|h| (h.1 - at, h.2)).collect::<Vec<_>>();
        set(0, Some((0x100, 0x100)));
        set(1, Some((0x800, 0x200)));
        set(2, Some((0x100, 0x100)));
        assert_eq!(hull(), [(0x100, 0x900)]);
        set(1, None);
        assert_eq!(hull(), [(0x100, 0x100)], "the far range gone");
        set(0, None);
        assert_eq!(hull(), [(0x100, 0x100)], "one slot still views it");
        set(2, None);
        assert!(hull().is_empty());
        crate::shadow::unregister(at as usize, at);
    }

    /// `refresh` scans `watch::DIRTY` words for tracked hulls (covering SRV and
    /// UAV hulls sharing one copy bit), emits dirty untracked hulls while
    /// skipping clean aliased shadows (`dirty_bit() == None`), cleans up its
    /// word index on removal, and bumps `hull_generation` in `set`,
    /// `set_batch`, and `add_unordered`.
    #[test]
    fn refresh_word_scan_covers_dirty_srv_uav_untracked_aliased_and_generations() {
        use crate::backend::HostMemory;
        let Some(host) = crate::host::host() else {
            return;
        };
        let p1 = crate::watch::tests::pool(1);
        let p2 = crate::watch::tests::pool(1);
        let untracked_mem = vec![0u8; 0x1000];
        let u_at = untracked_mem.as_ptr() as u64;
        crate::shadow::register(p1, p1 as u64, PAGE_BYTES as u64);
        crate::shadow::register(p2, p2 as u64, PAGE_BYTES as u64);
        crate::shadow::register(u_at as usize, u_at, 0x1000);
        let (Some(c1), Some(c2), Some(cu)) = (
            crate::shadow::find(p1 as u64, PAGE_BYTES as u64),
            crate::shadow::find(p2 as u64, PAGE_BYTES as u64),
            crate::shadow::find(u_at, 0x1000),
        ) else {
            crate::shadow::unregister(p1, p1 as u64);
            crate::shadow::unregister(p2, p2 as u64);
            crate::shadow::unregister(u_at as usize, u_at);
            crate::watch::tests::free(p1);
            crate::watch::tests::free(p2);
            return;
        };
        assert!(c1.dirty_bit().is_some() && c2.dirty_bit().is_some());
        assert!(cu.dirty_bit().is_none(), "plain heap buffer is untracked");
        // Make c2 clean before adding it.
        c2.update(p2, PAGE_BYTES);
        assert!(c2.clean());

        // Clean aliased shadow: `dirty_bit() == None` (lives in `untracked_hulls`)
        // yet `clean() == true`, so `refresh` must skip it.
        let (alias_buf, alias_data) = host.shared_buffer(0x1000).unwrap();
        // SAFETY: live D3D12 buffer.
        let alias_gpu = unsafe { alias_buf.GetGPUVirtualAddress() };
        let alias_base = 0x7000_0000_0000u64;
        let ca = Arc::new(Shadow::new_aliased(
            alias_buf, alias_data, alias_gpu, alias_base, 0x1000,
        ));
        assert!(ca.dirty_bit().is_none());
        assert!(ca.clean(), "fresh aliased shadow without widened SRVs is clean");

        let viewed = Viewed::default();
        let set_buf = |slot: u32, entry: Option<(&Arc<Shadow>, u64, u64, bool)>| {
            let mut inner = viewed.inner.lock().unwrap();
            let mut hulls_changed = false;
            if let Some(old) = inner.slots.remove(&slot) {
                hulls_changed |= inner.count(slot, &old, false).1;
            }
            if let Some((copy, addr, len, uav)) = entry {
                let seen = Seen::Buffer(copy.clone(), addr, len, uav);
                hulls_changed |= inner.count(slot, &seen, true).1;
                inner.slots.insert(slot, seen);
            }
            if hulls_changed {
                viewed.hull_generation.fetch_add(1, Release);
            }
        };
        let collect_dirty = || {
            let mut out = Vec::new();
            refresh(&viewed, |s, a, l| out.push((Arc::as_ptr(s) as usize, a, l)));
            out.sort_unstable();
            out
        };

        let g0 = viewed.hull_generation();
        set_buf(0, Some((&c1, p1 as u64 + 64, 128, false)));
        let g1 = viewed.hull_generation();
        assert!(g1 > g0, "first hull insert bumps generation");
        // Duplicate exact range keeps generation steady.
        set_buf(1, Some((&c1, p1 as u64 + 64, 128, false)));
        assert_eq!(viewed.hull_generation(), g1, "duplicate range keeps generation");
        // UAV hull on the same copy shares c1's dirty bit.
        set_buf(2, Some((&c1, p1 as u64 + 256, 128, true)));
        set_buf(3, Some((&c2, p2 as u64, 128, false)));
        set_buf(4, Some((&cu, u_at, 128, false)));
        set_buf(5, Some((&ca, alias_base, 128, false)));
        let g2 = viewed.hull_generation();
        assert!(g2 > g1);

        let p1_ptr = Arc::as_ptr(&c1) as usize;
        let cu_ptr = Arc::as_ptr(&cu) as usize;
        let mut expected = vec![
            (p1_ptr, p1 + 64, 128),
            (p1_ptr, p1, PAGE_BYTES), // UAV marked copy expands to page_range
            (cu_ptr, u_at as usize, 128),
        ];
        expected.sort_unstable();
        assert_eq!(
            collect_dirty(),
            expected,
            "dirty SRV+UAV and untracked emitted; clean c2 and clean aliased ca skipped"
        );

        // Taking c1 makes it clean: only untracked non-aliased copy remains.
        c1.update(p1, PAGE_BYTES);
        assert!(c1.clean());
        assert_eq!(collect_dirty(), [(cu_ptr, u_at as usize, 128)]);

        // Writing to c1 faults it dirty again: both SRV and UAV hulls reappear.
        // SAFETY: committed pool page of ours.
        unsafe { (p1 as *mut u32).write_volatile(0x1234_5678) };
        assert!(!c1.clean());
        assert_eq!(collect_dirty(), expected);

        // Remove both SRV slots of c1; UAV slot still keeps c1's bit indexed.
        set_buf(0, None);
        set_buf(1, None);
        let mut expected_uav = vec![(p1_ptr, p1, PAGE_BYTES), (cu_ptr, u_at as usize, 128)];
        expected_uav.sort_unstable();
        assert_eq!(collect_dirty(), expected_uav);

        // Remove all slots: word index and untracked list clean up completely.
        for slot in [2, 3, 4, 5] {
            set_buf(slot, None);
        }
        assert!(collect_dirty().is_empty());
        {
            let inner = viewed.inner.lock().unwrap();
            assert!(inner.hulls.is_empty());
            assert!(inner.hull_pos.is_empty());
            assert!(inner.active_words.is_empty());
            assert!(inner.untracked_hulls.is_empty());
        }

        // Verify `Viewed::set`, `Viewed::set_batch`, and `Viewed::add_unordered`
        // all bump `hull_generation`.
        let buf_res = resource(1, 0x1000, 1, 0, 1);
        let (buf_ptr, buf_size) = crate::resource::buffer_bytes(buf_res).unwrap();
        assert!(crate::shadow::find(buf_ptr as u64, buf_size).is_some());
        let srv_view = View::Shader(
            buf_res as usize,
            Some([39, 1, 0x1688, 0, 0, 0, 64, 0, 1, 0]),
        );
        let uav_view = View::Unordered(
            buf_res as usize,
            0,
            Some([39, 1, 0, 0, 64, 0, 0, 0, 1, 0]),
        );
        let v_gen = Viewed::new(true);
        let before_set = v_gen.hull_generation();
        v_gen.set(0, Some(&srv_view), true);
        let after_set = v_gen.hull_generation();
        assert!(after_set > before_set, "Viewed::set must bump hull_generation");
        v_gen.set_batch(&[(1, Some(&uav_view), true)]);
        let after_batch = v_gen.hull_generation();
        assert!(after_batch > after_set, "Viewed::set_batch must bump hull_generation");
        v_gen.set_batch(&[(2, Some(&uav_view), true)]);
        assert_eq!(
            v_gen.hull_generation(),
            after_batch,
            "duplicate range in set_batch keeps hull_generation"
        );
        v_gen.add_unordered(&c1, p1 as u64, 64);
        assert!(
            v_gen.hull_generation() > after_batch,
            "Viewed::add_unordered must bump hull_generation"
        );

        drop((v_gen, viewed));
        crate::object::release_ref(buf_res);
        crate::shadow::unregister(p1, p1 as u64);
        crate::shadow::unregister(p2, p2 as u64);
        crate::shadow::unregister(u_at as usize, u_at);
        drop((c1, c2, cu, ca));
        crate::watch::tests::free(p1);
        crate::watch::tests::free(p2);
    }

    /// A released texture takes the slots viewing it, and only those; a
    /// texture no slot views has nothing to take.
    #[test]
    fn forget_takes_the_texture_slots_only() {
        let mut inner = Inner::default();
        for (slot, seen) in [
            (0, Seen::Texture(0x10)),
            (1, Seen::Texture(0x20)),
            (2, Seen::Texture(0x10)),
            (3, Seen::Left((0, 0, 0))),
        ] {
            inner.count(slot, &seen, true);
            inner.slots.insert(slot, seen);
        }
        inner.forget(0x10);
        assert_eq!(inner.slots.len(), 2);
        assert!(matches!(inner.slots.get(&1), Some(Seen::Texture(0x20))));
        assert!(matches!(inner.slots.get(&3), Some(Seen::Left(_))));
        assert_eq!(inner.textures.keys().copied().collect::<Vec<_>>(), [0x20]);
        inner.forget(0x30);
        assert_eq!(inner.slots.len(), 2);
    }

    /// With `VICTORY_CENSUS_FAST=1`, `unordered_writes` matches the standard
    /// `widened_writes` full-map scan across added, replaced, cleared,
    /// unmade, batched (`set_batch`), and high/out-of-bounds slot indices,
    /// while hitting the lock-free snapshot cache when UAV ranges are unchanged.
    #[test]
    fn fast_census_matches_standard_across_added_replaced_cleared_and_batched_slots() {
        let Some(_host) = crate::host::host() else {
            return; // no host GPU here
        };
        let bufs = [
            resource(1, 0x1000, 1, 0, 1),
            resource(1, 0x1000, 1, 0, 1),
            resource(1, 0x1000, 1, 0, 1),
        ];
        // Ensure each buffer has a registered shadow copy.
        for &buf in &bufs {
            let (ptr, size) = crate::resource::buffer_bytes(buf).unwrap();
            assert!(crate::shadow::find(ptr as u64, size).is_some());
        }
        let uav_view = |buf: This, first_dw: u32, count_dw: u32| {
            View::Unordered(
                buf as usize,
                0,
                Some([39, 1, first_dw, 0, count_dw, 0, 0, 0, 1, 0]),
            )
        };
        let srv_view = |buf: This, first_dw: u32, count_dw: u32| {
            View::Shader(
                buf as usize,
                Some([39, 1, 0x1688, first_dw, 0, 0, count_dw, 0, 1, 0]),
            )
        };
        let normalize_writes = |writes: &[(Arc<Shadow>, usize, usize)]| {
            let mut out: Vec<_> = writes
                .iter()
                .map(|(s, a, l)| (Arc::as_ptr(s) as usize, *a, *l))
                .collect();
            out.sort_unstable();
            out
        };
        let normalize_hulls = |viewed: &Viewed| {
            let mut out: Vec<_> = viewed
                .ranges()
                .iter()
                .map(|h| (Arc::as_ptr(&h.0) as usize, h.1, h.2, h.3))
                .collect();
            out.sort_unstable();
            out
        };

        let standard = Viewed::new(false);
        let fast = Viewed::new(true);
        assert!(unordered_writes(&standard).is_none());

        let check_eq = |label: &str| {
            let got = unordered_writes(&fast).expect("fast cache enabled");
            let mut want = Vec::new();
            widened_writes_into(&standard, &mut want);
            assert_eq!(
                normalize_writes(&got),
                normalize_writes(&want),
                "{label}: widened writes mismatch"
            );
            assert_eq!(
                normalize_hulls(&fast),
                normalize_hulls(&standard),
                "{label}: hulls mismatch"
            );
            got
        };

        let u0 = uav_view(bufs[0], 0, 64);
        let u1 = uav_view(bufs[1], 16, 32);
        let u2 = uav_view(bufs[2], 32, 48);
        let s0 = srv_view(bufs[0], 0, 64);
        let s1 = srv_view(bufs[1], 0, 64);

        // 1. Added slots (UAV + SRV + high/out-of-bounds slot index).
        for viewed in [&standard, &fast] {
            viewed.set(0, Some(&u0), true);
            viewed.set(1, Some(&s0), true);
            viewed.set(u32::MAX, Some(&u1), true);
        }
        let first = check_eq("after adds");
        // Adding a non-UAV SRV must keep the cached snapshot intact under Wine.
        for viewed in [&standard, &fast] {
            viewed.set(2, Some(&s1), true);
        }
        if crate::under_wine() {
            assert!(
                fast.fast.as_ref().unwrap().get_cached().is_some(),
                "non-UAV SRV write must not invalidate UAV snapshot cache"
            );
            let cached = check_eq("after SRV-only add");
            assert!(Arc::ptr_eq(&first, &cached));
        }

        // 2. Replaced slots: UAV -> UAV, UAV -> SRV, SRV -> UAV, made=false.
        for viewed in [&standard, &fast] {
            viewed.set(0, Some(&u2), true);
            viewed.set(u32::MAX, Some(&s1), true);
            viewed.set(1, Some(&u0), true);
            viewed.set(3, Some(&u1), false); // unmade: Left, not a live UAV
        }
        check_eq("after replacements");

        // 3. Batched updates (`set_batch`) mixing adds, duplicates, replacements,
        //    clears, unmade views, and high slot indices.
        let batch: &[(u32, Option<&View>, bool)] = &[
            (10, Some(&u0), true),
            (11, Some(&u1), true),
            (12, Some(&u1), true), // duplicate UAV range
            (13, Some(&s0), true),
            (14, Some(&u2), false),
            (0, None, false), // clear existing UAV
            (2_000_000, Some(&u2), true),
            (2_000_001, None, false), // clear nonexistent high slot
        ];
        standard.set_batch(batch);
        fast.set_batch(batch);
        check_eq("after set_batch");

        // 4. Clearing one duplicate keeps range; clearing the second drops it.
        for viewed in [&standard, &fast] {
            viewed.set(11, None, false);
        }
        check_eq("after clearing first duplicate");
        for viewed in [&standard, &fast] {
            viewed.set(12, None, false);
        }
        check_eq("after clearing second duplicate");

        // 5. Randomized mix of single `set` and `set_batch` calls.
        let pool_views = [u0, u1, u2, s0, s1];
        let mut rng = 0x517C_C1B7_2722_0A95u64;
        let mut next = || {
            rng ^= rng << 13;
            rng ^= rng >> 7;
            rng ^= rng << 17;
            rng
        };
        let mut cache_hits = 0usize;
        for step in 0..500usize {
            if step.is_multiple_of(4) {
                let mut ops: Vec<(u32, Option<usize>, bool)> = Vec::new();
                for _ in 0..8 {
                    let slot = if next().is_multiple_of(11) {
                        1_500_000 + (next() % 8) as u32
                    } else {
                        (next() % 64) as u32
                    };
                    let view_idx = (!next().is_multiple_of(5)).then(|| (next() % 5) as usize);
                    let made = !next().is_multiple_of(6);
                    ops.push((slot, view_idx, made));
                }
                let refs: Vec<(u32, Option<&View>, bool)> = ops
                    .iter()
                    .map(|&(s, vi, m)| (s, vi.map(|i| &pool_views[i]), m))
                    .collect();
                standard.set_batch(&refs);
                fast.set_batch(&refs);
            } else {
                let slot = (next() % 64) as u32;
                let view = (!next().is_multiple_of(5)).then(|| &pool_views[(next() % 5) as usize]);
                let made = !next().is_multiple_of(6);
                standard.set(slot, view, made);
                fast.set(slot, view, made);
            }
            if step.is_multiple_of(2) {
                if fast.fast.as_ref().unwrap().get_cached().is_some() {
                    cache_hits += 1;
                }
                check_eq("randomized step");
            }
        }
        assert!(
            !crate::under_wine() || cache_hits > 20,
            "expected fast cache hits under Wine, saw {cache_hits}"
        );

        drop((standard, fast));
        for buf in bufs {
            crate::object::release_ref(buf);
        }
    }

    /// The title writes a buffer and a texture, and views of them and of an
    /// output buffer into its heap; a shader reaching them only through the
    /// heap (no table entry names them) reads what the title wrote at
    /// submission, every round (the title rewrites both each time, as Bink
    /// rewrites its planes every frame). The output buffer is not read back
    /// into title memory (`refresh`'s shortcut): it is read from its host
    /// copy.
    #[test]
    fn heap_views_are_current() {
        let (src, dst) = (resource(1, 256, 1, 0, 1), resource(1, 256, 1, 0, 1));
        let tex = resource(3, 4, 1, 28, LINEAR);
        let write = |base: u32, texel: u8| {
            let (src_bytes, _) = crate::resource::buffer_bytes(src).unwrap();
            let (tex_bytes, tex_desc) = crate::resource::texture_memory(tex).unwrap();
            let size = crate::resource::texture_layout(&tex_desc).unwrap().planes[0].layout.size as usize;
            // SAFETY: the resources' own memory, 256 bytes and the texture's.
            unsafe {
                let words = std::slice::from_raw_parts_mut(src_bytes.cast::<u32>(), 64);
                for (i, w) in words.iter_mut().enumerate() {
                    *w = base + i as u32;
                }
                std::ptr::write_bytes(tex_bytes.cast_mut(), texel, size);
            }
        };
        rounds(src, dst, tex, write);
    }

    /// The same with the buffer and the texture in one view of physical
    /// pages and the title writing them through another (the title fills its
    /// heaps through a pool view): every round must reach the shader.
    #[test]
    fn heap_views_follow_writes_through_another_mapping() {
        let (a, b) = crate::watch::tests::twin_views(2);
        let src = resource_at(a, 1, 256, 1, 0, 1);
        let tex = resource_at(a + PAGE_BYTES, 3, 4, 1, 28, LINEAR);
        let dst = resource(1, 256, 1, 0, 1);
        let (_, tex_desc) = crate::resource::texture_memory(tex).unwrap();
        let size = crate::resource::texture_layout(&tex_desc).unwrap().planes[0].layout.size as usize;
        let write = |base: u32, texel: u8| {
            // SAFETY: the second view of the buffer's and the texture's pages.
            unsafe {
                let words = std::slice::from_raw_parts_mut(b as *mut u32, 64);
                for (i, w) in words.iter_mut().enumerate() {
                    *w = base + i as u32;
                }
                std::ptr::write_bytes((b + PAGE_BYTES) as *mut u8, texel, size);
            }
        };
        rounds(src, dst, tex, write);
    }

    const PAGE_BYTES: usize = crate::watch::PAGE;

    /// Three rounds of: record a dispatch reading slots 0 and 2 and writing
    /// slot 1, let the title `write` (after recording, before submitting),
    /// submit, check the output's host copy.
    fn rounds(src: This, dst: This, tex: This, write: impl Fn(u32, u8)) {
        let _serial = crate::heap_worker::serial();
        let Some(host) = crate::host::host() else {
            return; // no host GPU here
        };
        let null = std::ptr::null();
        // A shader-visible view heap of 8 slots, in memory of its own.
        let heap_desc: [u32; 8] = [0, 8, 1, 0, 0, 0, 0, 0];
        let mut heap = null;
        let iid = crate::object::guid(
            0x8efb471d,
            0x616c,
            0x4f49,
            [0x90, 0xf7, 0x12, 0x7b, 0xb7, 0x63, 0xfa, 0x51],
        );
        assert_eq!(crate::descriptor::create_heap(null, heap_desc.as_ptr().cast(), &iid, &mut heap), S_OK);
        let title_heap = crate::descriptor::heap(heap);
        let slot = |i: u64| title_heap.start + 32 * i;
        // Raw views of 64 dwords: format R32_TYPELESS, buffer dimension,
        // the default mapping, then first element, count, stride, flags.
        let srv: [u32; 10] = [39, 1, 0x1688, 0, 0, 0, 64, 0, 1, 0];
        let uav: [u32; 10] = [39, 1, 0, 0, 64, 0, 0, 0, 1, 0];
        crate::descriptor::create_srv(null, src, &srv, slot(0));
        crate::descriptor::create_uav(null, dst, null, &uav, slot(1));
        crate::descriptor::create_srv(null, tex, std::ptr::null(), slot(2));
        let heap_host = title_heap.host().expect("a host view heap");
        let (root, pso) = compile();
        let list = host.new_list().unwrap();
        let (dst_bytes, _) = crate::resource::buffer_bytes(dst).unwrap();
        let out = crate::shadow::find(dst_bytes as u64, 256).expect("the output's host copy");
        for (round, (base, texel)) in [(1000u32, 0x40u8), (5000, 0x10), (9000, 0x22)].into_iter().enumerate() {
            if round > 0 {
                list.reset();
            }
            // SAFETY: an open list, live objects of ours.
            unsafe {
                list.list.SetDescriptorHeaps(&[Some(heap_host.heap.clone())]);
                list.list.SetComputeRootSignature(&root);
                list.list.SetComputeRootDescriptorTable(0, heap_host.gpu(0));
                list.list.SetPipelineState(&pso);
                list.list.Dispatch(4, 1, 1);
            }
            list.defer_heap(&heap_host.viewed);
            list.close();
            // The title writes after recording, before submitting.
            write(base, texel);
            host.execute(&[&list]);
            let words = out.read_host(dst_bytes as u64, 64);
            for (i, &v) in words.iter().enumerate() {
                assert_eq!(v, base + i as u32 + u32::from(texel), "round {round}, dword {i}");
            }
        }
    }

    /// With `VICTORY_HEAP_WORKER=1`:
    /// - a placed bindless texture `A` stays null until the worker stages it
    ///   and `Host::execute` publishes it atomically ahead of the dispatch;
    /// - a bindless texture `B` whose `host_written` flips before submission
    ///   has its staged CPU upload skipped so host-written contents are kept;
    /// - a table-bound texture `C` (`Step::Texture`) uploads synchronously in
    ///   the very first submission without waiting for the worker.
    #[test]
    fn worker_upload_publishes_atomically_and_skips_host_written() {
        let Some(host) = crate::host::host() else {
            return;
        };
        let _worker = crate::heap_worker::EnableForTest::new();
        let null = std::ptr::null();
        let heap_desc: [u32; 8] = [0, 16, 1, 0, 0, 0, 0, 0];
        let mut heap = null;
        let heap_iid = crate::object::guid(
            0x8efb471d,
            0x616c,
            0x4f49,
            [0x90, 0xf7, 0x12, 0x7b, 0xb7, 0x63, 0xfa, 0x51],
        );
        assert_eq!(
            crate::descriptor::create_heap(null, heap_desc.as_ptr().cast(), &heap_iid, &mut heap),
            S_OK
        );
        let title_heap = crate::descriptor::heap(heap);
        let slot = |i: u64| title_heap.start + 32 * i;
        let heap_host = title_heap.host().expect("host view heap");

        let pool = crate::watch::tests::pool(2);
        let addr_a = pool as u64;
        let addr_b = (pool + PAGE_BYTES) as u64;
        // SAFETY: two dedicated pool pages.
        unsafe {
            std::ptr::write_bytes(addr_a as *mut u8, 0x33, PAGE_BYTES);
            std::ptr::write_bytes(addr_b as *mut u8, 0x77, PAGE_BYTES);
        }
        let placed_res = [3u32, 0, 0, 0, 4, 0, 1, 1 | 1 << 16, 28, 1, 0, LINEAR, 0, 0];
        let mut desc_a = [0u32; 16];
        (desc_a[0], desc_a[1], desc_a[2], desc_a[5]) = (28, 4, 0x1688, 1);
        (desc_a[10], desc_a[11]) = (addr_a as u32, (addr_a >> 32) as u32);
        let mut desc_b = desc_a;
        (desc_b[10], desc_b[11]) = (addr_b as u32, (addr_b >> 32) as u32);

        let src = resource(1, 256, 1, 0, 1);
        let dst = resource(1, 256, 1, 0, 1);
        let tex_c = resource(3, 4, 1, 28, LINEAR);
        let (src_bytes, _) = crate::resource::buffer_bytes(src).unwrap();
        let (dst_bytes, _) = crate::resource::buffer_bytes(dst).unwrap();
        let (c_bytes, c_desc) = crate::resource::texture_memory(tex_c).unwrap();
        let c_size = crate::resource::texture_layout(&c_desc).unwrap().planes[0].layout.size as usize;
        // SAFETY: owned buffers/texture memory.
        unsafe {
            std::ptr::write_bytes(src_bytes, 0, 256);
            std::ptr::write_bytes(c_bytes.cast_mut(), 0x55, c_size);
        }

        let srv: [u32; 10] = [39, 1, 0x1688, 0, 0, 0, 64, 0, 1, 0];
        let uav: [u32; 10] = [39, 1, 0, 0, 64, 0, 0, 0, 1, 0];
        crate::descriptor::create_srv(null, src, &srv, slot(0));
        crate::descriptor::create_uav(null, dst, null, &uav, slot(1));
        crate::descriptor::create_placed_srv(null, &placed_res, &desc_a, slot(2));
        crate::descriptor::create_placed_srv(null, &placed_res, &desc_b, slot(3));
        assert!(!heap_host.kept(2) && !heap_host.kept(3), "placed views stay null before worker");

        // Stage A and B on the worker pass, then flip B to `host_written` before submission.
        _worker.step_heap(title_heap);
        assert!(crate::heap_worker::has_ready());
        assert!(!heap_host.kept(2), "not published until Host::execute");
        let res_b = crate::resource::texture_at(addr_b, &placed_res).unwrap();
        let copy_b = crate::resource::made_copy(res_b).unwrap();
        let uploads_b_before = crate::texture::uploads(res_b);
        copy_b.host_written.store(true, Relaxed);

        // Also attach table-bound texture C synchronously on this list.
        let copy_c = crate::resource::host_copy(tex_c).unwrap();
        let (root, pso) = compile();
        let list = host.new_list().unwrap();
        crate::texture::upload(&list, tex_c, &copy_c.texture);
        // SAFETY: open list, live objects.
        unsafe {
            list.list.SetDescriptorHeaps(&[Some(heap_host.heap.clone())]);
            list.list.SetComputeRootSignature(&root);
            list.list.SetComputeRootDescriptorTable(0, heap_host.gpu(0));
            list.list.SetPipelineState(&pso);
            list.list.Dispatch(4, 1, 1);
        }
        list.defer_heap(&heap_host.viewed);
        list.close();
        host.execute(&[&list]);

        // A was published and read 0x33 in that exact submission.
        assert!(heap_host.kept(2), "slot 2 published at submission");
        let out = crate::shadow::find(dst_bytes as u64, 256).unwrap();
        let words = out.read_host(dst_bytes as u64, 4);
        assert_eq!(words, [0x33, 0x33, 0x33, 0x33], "shader saw A's uploaded texels");
        // B's CPU upload was skipped because `host_written` flipped before submission.
        assert_eq!(
            crate::texture::uploads(res_b),
            uploads_b_before,
            "host_written texture upload was skipped"
        );
        // Table-bound texture C uploaded synchronously in the first submission.
        assert_eq!(
            crate::texture::uploads(tex_c),
            1,
            "table-bound texture uploaded synchronously"
        );

        crate::object::release_ref(heap);
        crate::object::release_ref(src);
        crate::object::release_ref(dst);
        crate::object::release_ref(tex_c);
        crate::watch::tests::free(pool);
    }
}
