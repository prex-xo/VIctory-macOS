//! Work a queue holds back (decision 0007, "Queues wait on memory values").
//!
//! The console's lists wait on memory values (`Wait32BitValueX`,
//! `Wait64BitValueX`) that lists on other queues write at their end
//! (`Write32/64BitValueBottomOfPipeX`): Grand Theft Auto VI's four compute
//! queues start each list with a wait for the frame's value of a word that
//! a graphics list writes, and some lists wait again after their first
//! commands. Our submissions run to completion when the title makes them,
//! so a queue here is a FIFO of what the title submitted to it: segments of
//! lists (a list is cut where it waits after commands, `crate::list`), its
//! Signals and its fence Waits, in their order. The head of a queue runs
//! once its waits hold; consecutive heads that may run go to the host GPU
//! together (a piece); after every piece the heads of all queues are tested
//! again until none can move. Order within a queue never changes, only the
//! interleaving across queues.

use std::collections::VecDeque;
use std::time::{Duration, Instant};

/// A wait for a memory value, as a list records it: the value at `address`
/// (`bytes` 4 or 8, the 32-bit form masked) compared with `reference` by a
/// D3D12 comparison function.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) struct ValueWait {
    pub(crate) address: u64,
    pub(crate) bytes: u8,
    pub(crate) comparison: u32,
    pub(crate) reference: u64,
    pub(crate) mask: u64,
}

/// D3D12 comparison functions (`D3D12_COMPARISON_FUNC`).
const NEVER: u32 = 1;
const LESS: u32 = 2;
const EQUAL: u32 = 3;
const LESS_EQUAL: u32 = 4;
const GREATER: u32 = 5;
const NOT_EQUAL: u32 = 6;
const GREATER_EQUAL: u32 = 7;
const ALWAYS: u32 = 8;

impl ValueWait {
    /// Whether `raw` (masked) compares true against `reference`. Returns `None`
    /// for a comparison function we do not know.
    pub(crate) fn holds_for(&self, raw: u64) -> Option<bool> {
        if !(NEVER..=ALWAYS).contains(&self.comparison) {
            return None;
        }
        let value = raw & self.mask;
        let reference = self.reference;
        Some(match self.comparison {
            NEVER => false,
            LESS => value < reference,
            EQUAL => value == reference,
            LESS_EQUAL => value <= reference,
            GREATER => value > reference,
            NOT_EQUAL => value != reference,
            GREATER_EQUAL => value >= reference,
            _ => true,
        })
    }

    /// Whether it holds with `read` giving the memory's value: `None` for a
    /// comparison we do not know (counted by the caller, taken as holding)
    /// and for memory that cannot be read (taken as not holding).
    pub(crate) fn holds(&self, read: &dyn Fn(u64, u8) -> Option<u64>) -> Option<bool> {
        if !(NEVER..=ALWAYS).contains(&self.comparison) {
            return None;
        }
        let raw = read(self.address, self.bytes)?;
        self.holds_for(raw)
    }
}

/// Summary of a segment's deferred CPU/GPU side effects used to avoid merging
/// across submission-phase ordering hazards.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub(crate) struct BatchFlags {
    /// Writes into title CPU memory during deferred prep (`Step::Copy`, `Step::Move`).
    pub(crate) has_title_copy: bool,
    /// Reads CPU title memory during deferred prep (`Step::Copy`, `Step::Move`, `Step::Fill`).
    pub(crate) has_cpu_read: bool,
    /// Records GPU shader writes to a shadow buffer (`Step::GpuWrite`).
    pub(crate) has_gpu_write: bool,
    /// Runs post-execution callbacks (`defer_after`, e.g., readbacks).
    pub(crate) has_after: bool,
    /// Explicitly uploads title-written textures before deferred prep (`Step::Texture`).
    pub(crate) has_textures: bool,
}

impl BatchFlags {
    pub(crate) fn merge(self, other: Self) -> Self {
        Self {
            has_title_copy: self.has_title_copy || other.has_title_copy,
            has_cpu_read: self.has_cpu_read || other.has_cpu_read,
            has_gpu_write: self.has_gpu_write || other.has_gpu_write,
            has_after: self.has_after || other.has_after,
            has_textures: self.has_textures || other.has_textures,
        }
    }
}

/// Whether a segment with `next` flags can join a non-empty batch with
/// accumulated `existing` flags and `existing_len` segments without exceeding
/// `max_lists` or violating host submission phase ordering:
/// - Explicit texture uploads run before deferred title copies, so a prior
///   title copy or post-execution callback must complete before a later
///   segment uploads textures from title memory.
/// - Deferred CPU reads (`Copy`/`Move`/`Fill`) run before GPU execution and
///   copy-back, so a prior GPU write or post-execution callback must complete
///   before a later segment reads CPU title memory.
pub(crate) fn can_join_batch(
    existing: BatchFlags,
    next: BatchFlags,
    existing_len: usize,
    max_lists: usize,
) -> bool {
    if existing_len >= max_lists {
        return false;
    }
    if (existing.has_title_copy || existing.has_after) && next.has_textures {
        return false;
    }
    if (existing.has_gpu_write || existing.has_after) && next.has_cpu_read {
        return false;
    }
    true
}

/// What a queue holds, in its order.
pub(crate) enum Item<S> {
    /// A list segment and the waits at its start.
    Segment(S, Vec<ValueWait>),
    /// A fence Signal: the fence and its value.
    Signal(usize, u64),
    /// A fence Wait: until the fence reaches the value.
    FenceWait(usize, u64),
    /// Work of ours in the queue's order (a page mapping batch): it runs once
    /// everything before it on its queue was handed to the host GPU, and
    /// nothing after it on its queue is taken before it ran. Other queues
    /// are not ordered with it.
    Action(Action),
}

/// What an [`Item::Action`] runs.
pub(crate) type Action = Box<dyn FnOnce() + Send>;

/// What a pump did, for the frame's accounting.
#[derive(Default, Debug, PartialEq, Eq)]
pub(crate) struct Pumped {
    /// Pieces given to the host GPU, and the segments in them.
    pub(crate) pieces: u32,
    pub(crate) segments: u32,
    /// Of those, segments that had waited (not run when submitted): their
    /// queue and how long.
    pub(crate) waited: Vec<(usize, Duration)>,
    /// Waits with a comparison we do not know (taken as holding).
    pub(crate) unknown: u32,
    /// Waits satisfied by an earlier segment's deferred bottom-of-pipe write
    /// in the same coalesced piece.
    pub(crate) coalesced_bop_waits: u32,
    /// Coalesced pieces flushed early because the next ready segment hit a
    /// submission-phase ordering hazard or the batch list cap.
    pub(crate) hazard_splits: u32,
}

/// One ready step popped from [`Held`] so the caller can release the `HELD`
/// lock while running it on the host GPU.
#[cfg(any(windows, test))]
pub(crate) enum Ready<S> {
    /// Ready segments popped as one host submission (from one queue, or
    /// coalesced across queues and bottom-of-pipe dependencies when
    /// `coalesce` is enabled), paired with whether a global UAV barrier must
    /// precede each segment, plus any mid-batch `Item::Signal`s to attach
    /// behind the submission. Every queue in `queues` stays marked running
    /// until [`Held::finish_running`] is called so later submissions and
    /// signals on those queues wait behind it.
    Segments {
        queues: Vec<usize>,
        segments: Vec<(S, bool)>,
        signals: Vec<(usize, u64)>,
        waited: Vec<(usize, Duration)>,
        unknown: u32,
        coalesced_bop_waits: u32,
        hazard_splits: u32,
    },
    /// A fence signal ready at the head of a queue with no preceding segments.
    Signal { fence: usize, value: u64 },
    /// An [`Item::Action`] at the head of a queue, nothing taken before it in
    /// this call: the caller runs it before taking anything else.
    Action(Action),
}

/// Every queue's held work.
pub(crate) struct Held<S> {
    queues: Vec<(usize, Fifo<S>)>,
    /// Queues whose popped segments are currently launching outside the lock.
    running: Vec<usize>,
}

/// A queue's items, each with when it was pushed and whether it is fresh
/// (not yet found waiting).
type Fifo<S> = VecDeque<(Item<S>, Instant, bool)>;

impl<S> Default for Held<S> {
    fn default() -> Self {
        Self::new()
    }
}

impl<S> Held<S> {
    pub(crate) const fn new() -> Self {
        Held {
            queues: Vec::new(),
            running: Vec::new(),
        }
    }
}

/// Predicate checking whether a ready segment can join the in-progress coalesced piece.
pub(crate) type CanJoin<'a, S> = &'a dyn Fn(&[(S, bool)], &S) -> bool;

/// What the pump needs from the outside world.
pub(crate) struct World<'a, S> {
    /// A memory value (address, bytes).
    pub(crate) read: &'a dyn Fn(u64, u8) -> Option<u64>,
    /// A fence's value.
    pub(crate) fence: &'a dyn Fn(usize) -> u64,
    /// Whether a segment's deferred bottom-of-pipe writes satisfy `wait`.
    pub(crate) will_write: &'a dyn Fn(&S, &ValueWait) -> bool,
    /// Whether `next` can join a non-empty coalesced `piece` without violating
    /// ordering hazards or batch limits.
    pub(crate) can_join: CanJoin<'a, S>,
    /// Whether cross-queue and bottom-of-pipe coalescing is active.
    pub(crate) coalesce: bool,
    /// Runs a piece of `(segment, barrier_before)` pairs in order to
    /// completion, the bottom-of-pipe writes included.
    pub(crate) run: &'a mut dyn FnMut(Vec<(S, bool)>),
    /// Signals a fence.
    pub(crate) signal: &'a mut dyn FnMut(usize, u64),
}

impl<S> Held<S> {
    /// Appends `item` to `queue`'s FIFO, marked fresh: whether it waited is
    /// decided by its first test.
    pub(crate) fn push(&mut self, queue: usize, item: Item<S>) {
        let now = Instant::now();
        match self.queues.iter_mut().find(|(q, _)| *q == queue) {
            Some((_, items)) => items.push_back((item, now, true)),
            None => self.queues.push((queue, VecDeque::from([(item, now, true)]))),
        }
    }

    /// Runs every queue's heads that may run, until none can.
    pub(crate) fn pump(&mut self, w: &mut World<S>) -> Pumped {
        if w.coalesce {
            self.pump_coalesced(w)
        } else {
            self.pump_per_queue(w)
        }
    }

    /// Legacy per-queue pump: flushes each queue's consecutive ready segments
    /// separately and flushes before every `Item::Signal`.
    fn pump_per_queue(&mut self, w: &mut World<S>) -> Pumped {
        let mut done = Pumped::default();
        loop {
            let mut moved = false;
            for (queue, items) in &mut self.queues {
                let mut piece = Vec::new();
                loop {
                    let ready = match items.front_mut() {
                        None => false,
                        Some((Item::Segment(_, waits), since, fresh)) => {
                            // Earlier segments of this queue run before its
                            // waits are tested: their bottom-of-pipe writes
                            // can change what the waits read.
                            if !waits.is_empty() && !piece.is_empty() {
                                (w.run)(std::mem::take(&mut piece));
                                done.pieces += 1;
                            }
                            let mut ok = true;
                            for wait in waits.iter() {
                                match wait.holds(w.read) {
                                    Some(true) => {}
                                    Some(false) => ok = false,
                                    None => done.unknown += 1,
                                }
                            }
                            if !ok {
                                *fresh = false;
                            } else if !*fresh {
                                done.waited.push((*queue, since.elapsed()));
                            }
                            ok
                        }
                        // A Signal or Wait keeps its place: the piece before
                        // it runs first.
                        Some((Item::Signal(..) | Item::Action(_), ..)) => {
                            if !piece.is_empty() {
                                (w.run)(std::mem::take(&mut piece));
                                done.pieces += 1;
                            }
                            true
                        }
                        Some((Item::FenceWait(fence, value), _, fresh)) => {
                            let ok = (w.fence)(*fence) >= *value;
                            if !ok {
                                *fresh = false;
                            }
                            ok
                        }
                    };
                    if !ready {
                        break;
                    }
                    moved = true;
                    match items.pop_front().map(|(item, ..)| item) {
                        Some(Item::Segment(s, _)) => {
                            piece.push((s, false));
                            done.segments += 1;
                        }
                        Some(Item::Signal(fence, value)) => (w.signal)(fence, value),
                        Some(Item::Action(act)) => act(),
                        Some(Item::FenceWait(..)) | None => {}
                    }
                }
                if !piece.is_empty() {
                    (w.run)(piece);
                    done.pieces += 1;
                }
            }
            if !moved {
                break;
            }
        }
        self.queues.retain(|(_, items)| !items.is_empty());
        done
    }

    /// Coalesced pump: accumulates ready segments across queues into one host
    /// submission, allowing a segment whose memory waits are satisfied by an
    /// earlier segment's deferred bottom-of-pipe write in the same piece to
    /// join behind a global UAV barrier (subject to `can_join`), and deferring
    /// mid-piece `Item::Signal`s until immediately after the piece completes.
    fn pump_coalesced(&mut self, w: &mut World<S>) -> Pumped {
        enum Decision {
            Blocked,
            FlushFirst,
            TakeSegment(bool),
            TakeSignal,
            TakeFenceWait(bool),
            TakeAction,
        }

        let flush = |piece: &mut Vec<(S, bool)>,
                     pending_signals: &mut Vec<(usize, u64)>,
                     last_queue: &mut Option<usize>,
                     w: &mut World<S>,
                     done: &mut Pumped| {
            if !piece.is_empty() {
                (w.run)(std::mem::take(piece));
                done.pieces += 1;
            }
            *last_queue = None;
            for (fence, value) in pending_signals.drain(..) {
                (w.signal)(fence, value);
            }
        };

        let mut done = Pumped::default();
        let mut piece: Vec<(S, bool)> = Vec::new();
        let mut pending_signals: Vec<(usize, u64)> = Vec::new();
        let mut last_queue: Option<usize> = None;
        loop {
            let mut moved = false;
            for (queue, items) in &mut self.queues {
                let mut pending_fence_barrier = false;
                loop {
                    let decision = match items.front_mut() {
                        None => Decision::Blocked,
                        Some((Item::Segment(seg, waits), since, fresh)) => {
                            let mut ok = true;
                            let mut via_bop = 0u32;
                            let mut unknown = 0u32;
                            for wait in waits.iter() {
                                match wait.holds(w.read) {
                                    Some(true) => {}
                                    Some(false) => {
                                        if piece.iter().rev().any(|(prev, _)| (w.will_write)(prev, wait)) {
                                            via_bop += 1;
                                        } else {
                                            ok = false;
                                        }
                                    }
                                    None => unknown += 1,
                                }
                            }
                            // Only check `can_join` when crossing a join boundary (another queue,
                            // a BoP-satisfied wait, or a deferred-signal fence wait). Consecutive
                            // ready segments on the same queue already ran together in one host
                            // submission in legacy mode and must never be shredded by `can_join`.
                            let is_join_boundary = !piece.is_empty()
                                && (last_queue != Some(*queue)
                                    || via_bop > 0
                                    || pending_fence_barrier);
                            if !ok {
                                *fresh = false;
                                done.unknown += unknown;
                                Decision::Blocked
                            } else if is_join_boundary && !(w.can_join)(&piece, seg) {
                                Decision::FlushFirst
                            } else {
                                if !*fresh {
                                    done.waited.push((*queue, since.elapsed()));
                                }
                                done.unknown += unknown;
                                done.coalesced_bop_waits += via_bop;
                                let barrier = via_bop > 0 || std::mem::take(&mut pending_fence_barrier);
                                Decision::TakeSegment(barrier)
                            }
                        }
                        Some((Item::Signal(..), ..)) => Decision::TakeSignal,
                        Some((Item::Action(_), ..)) => Decision::TakeAction,
                        Some((Item::FenceWait(fence, value), _, fresh)) => {
                            let mem_ok = (w.fence)(*fence) >= *value;
                            let pending_ok = !mem_ok
                                && pending_signals
                                    .iter()
                                    .any(|&(f, v)| f == *fence && v >= *value);
                            if !mem_ok && !pending_ok {
                                *fresh = false;
                                Decision::Blocked
                            } else {
                                Decision::TakeFenceWait(pending_ok)
                            }
                        }
                    };
                    match decision {
                        Decision::Blocked => break,
                        Decision::FlushFirst => {
                            done.hazard_splits += 1;
                            flush(&mut piece, &mut pending_signals, &mut last_queue, w, &mut done);
                            pending_fence_barrier = false;
                            moved = true;
                            continue;
                        }
                        Decision::TakeSegment(barrier_before) => {
                            moved = true;
                            if let Some((Item::Segment(s, _), ..)) = items.pop_front() {
                                piece.push((s, barrier_before));
                                last_queue = Some(*queue);
                                done.segments += 1;
                            }
                        }
                        Decision::TakeSignal => {
                            moved = true;
                            if let Some((Item::Signal(fence, value), ..)) = items.pop_front() {
                                if piece.is_empty() {
                                    (w.signal)(fence, value);
                                } else {
                                    pending_signals.push((fence, value));
                                }
                            }
                        }
                        Decision::TakeFenceWait(via_pending) => {
                            moved = true;
                            if via_pending {
                                pending_fence_barrier = true;
                            }
                            items.pop_front();
                        }
                        Decision::TakeAction => {
                            // What was taken before it runs first, its
                            // signals included; nothing joins across it.
                            moved = true;
                            flush(&mut piece, &mut pending_signals, &mut last_queue, w, &mut done);
                            pending_fence_barrier = false;
                            if let Some((Item::Action(act), ..)) = items.pop_front() {
                                act();
                            }
                        }
                    }
                }
            }
            if !moved {
                if piece.is_empty() && pending_signals.is_empty() {
                    break;
                }
                flush(&mut piece, &mut pending_signals, &mut last_queue, w, &mut done);
            }
        }
        self.queues.retain(|(_, items)| !items.is_empty());
        done
    }

    /// The queues still holding work: (queue, items, how long the head
    /// waited, the head's waits that do not hold).
    pub(crate) fn blocked(&self, read: &dyn Fn(u64, u8) -> Option<u64>) -> Vec<(usize, usize, Duration, Vec<ValueWait>)> {
        self.queues
            .iter()
            .filter_map(|(q, items)| {
                let (head, since, _) = items.front()?;
                let unmet = match head {
                    Item::Segment(_, waits) => waits.iter().filter(|w| w.holds(read) == Some(false)).copied().collect(),
                    _ => Vec::new(),
                };
                Some((*q, items.len(), since.elapsed(), unmet))
            })
            .collect()
    }

    /// Pops the next ready batch of segments or ready fence signal so the
    /// caller can drop the `HELD` lock before invoking the host GPU.
    /// When `coalesce` is `true`, accumulates ready segments across all
    /// non-running queues (including segments whose memory waits are satisfied
    /// by an earlier segment's deferred bottom-of-pipe writes via `will_write`,
    /// gated at cross-queue / BoP / fence-wait boundaries by `can_join`) and
    /// defers mid-batch `Item::Signal`s into `Ready::Segments::signals`.
    /// When [`Ready::Segments`] is returned, the caller must call
    /// [`Self::finish_running`] with `queues` once the submission call returns.
    #[cfg(any(windows, test))]
    pub(crate) fn take_ready_with(
        &mut self,
        read: &dyn Fn(u64, u8) -> Option<u64>,
        fence: &dyn Fn(usize) -> u64,
        will_write: &dyn Fn(&S, &ValueWait) -> bool,
        can_join: CanJoin<'_, S>,
        coalesce: bool,
    ) -> Option<Ready<S>> {
        if !coalesce {
            return self.take_ready_per_queue(read, fence);
        }
        let mut piece: Vec<(S, bool)> = Vec::new();
        let mut signals: Vec<(usize, u64)> = Vec::new();
        let mut queues_touched: Vec<usize> = Vec::new();
        let mut waited: Vec<(usize, Duration)> = Vec::new();
        let mut unknown = 0u32;
        let mut coalesced_bop_waits = 0u32;
        let mut hazard_splits = 0u32;
        let mut last_queue: Option<usize> = None;
        let mut split_now = false;

        let mark_touched = |touched: &mut Vec<usize>, q: usize| {
            if !touched.contains(&q) {
                touched.push(q);
            }
        };

        loop {
            let mut moved = false;
            for (queue, items) in &mut self.queues {
                if self.running.contains(queue) {
                    continue;
                }
                let mut pending_fence_barrier = false;
                let mut popped_pending_waits: Vec<(Item<S>, Instant, bool)> = Vec::new();
                loop {
                    match items.front_mut() {
                        None => break,
                        Some((Item::FenceWait(f, v), _, fresh)) => {
                            let mem_ok = fence(*f) >= *v;
                            let pending_ok = !mem_ok
                                && signals.iter().any(|&(sf, sv)| sf == *f && sv >= *v);
                            if !mem_ok && !pending_ok {
                                *fresh = false;
                                break;
                            }
                            if let Some(entry) = items.pop_front() {
                                if pending_ok {
                                    pending_fence_barrier = true;
                                    popped_pending_waits.push(entry);
                                }
                                moved = true;
                            }
                        }
                        Some((Item::Action(_), ..)) => {
                            // Only at an empty piece: what this queue put in
                            // the piece goes to the host GPU first, and the
                            // queue takes nothing more in this call.
                            if piece.is_empty()
                                && signals.is_empty()
                                && let Some((Item::Action(act), ..)) = items.pop_front()
                            {
                                self.queues.retain(|(_, q)| !q.is_empty());
                                return Some(Ready::Action(act));
                            }
                            for entry in popped_pending_waits.drain(..).rev() {
                                items.push_front(entry);
                            }
                            break;
                        }
                        Some((Item::Signal(..), ..)) => {
                            if piece.is_empty() && signals.is_empty() {
                                if let Some((Item::Signal(f, v), ..)) = items.pop_front() {
                                    self.queues.retain(|(_, q)| !q.is_empty());
                                    return Some(Ready::Signal { fence: f, value: v });
                                }
                                break;
                            }
                            if let Some((Item::Signal(f, v), ..)) = items.pop_front() {
                                popped_pending_waits.clear();
                                signals.push((f, v));
                                mark_touched(&mut queues_touched, *queue);
                                moved = true;
                            }
                        }
                        Some((Item::Segment(seg, waits), since, fresh)) => {
                            let mut ok = true;
                            let mut via_bop = 0u32;
                            let mut seg_unknown = 0u32;
                            for wait in waits.iter() {
                                match wait.holds(read) {
                                    Some(true) => {}
                                    Some(false) => {
                                        if piece
                                            .iter()
                                            .rev()
                                            .any(|(prev, _)| will_write(prev, wait))
                                        {
                                            via_bop += 1;
                                        } else {
                                            ok = false;
                                        }
                                    }
                                    None => seg_unknown += 1,
                                }
                            }
                            let is_join_boundary = !piece.is_empty()
                                && (last_queue != Some(*queue)
                                    || via_bop > 0
                                    || pending_fence_barrier);
                            if !ok {
                                *fresh = false;
                                unknown += seg_unknown;
                                for entry in popped_pending_waits.drain(..).rev() {
                                    items.push_front(entry);
                                }
                                break;
                            }
                            if is_join_boundary && !can_join(&piece, seg) {
                                hazard_splits += 1;
                                split_now = true;
                                for entry in popped_pending_waits.drain(..).rev() {
                                    items.push_front(entry);
                                }
                                break;
                            }
                            if !*fresh {
                                waited.push((*queue, since.elapsed()));
                            }
                            unknown += seg_unknown;
                            coalesced_bop_waits += via_bop;
                            let barrier_before =
                                via_bop > 0 || std::mem::take(&mut pending_fence_barrier);
                            popped_pending_waits.clear();
                            if let Some((Item::Segment(s, _), ..)) = items.pop_front() {
                                piece.push((s, barrier_before));
                                last_queue = Some(*queue);
                                mark_touched(&mut queues_touched, *queue);
                                moved = true;
                            }
                        }
                    }
                }
                for entry in popped_pending_waits.drain(..).rev() {
                    items.push_front(entry);
                }
                if split_now {
                    break;
                }
            }
            if split_now || !moved {
                break;
            }
        }
        self.queues.retain(|(_, items)| !items.is_empty());
        if !piece.is_empty() {
            for &q in &queues_touched {
                if !self.running.contains(&q) {
                    self.running.push(q);
                }
            }
            return Some(Ready::Segments {
                queues: queues_touched,
                segments: piece,
                signals,
                waited,
                unknown,
                coalesced_bop_waits,
                hazard_splits,
            });
        }
        None
    }

    #[cfg(any(windows, test))]
    fn take_ready_per_queue(
        &mut self,
        read: &dyn Fn(u64, u8) -> Option<u64>,
        fence: &dyn Fn(usize) -> u64,
    ) -> Option<Ready<S>> {
        for (queue, items) in &mut self.queues {
            if self.running.contains(queue) {
                continue;
            }
            let mut segments = Vec::new();
            let mut waited = Vec::new();
            let mut unknown = 0u32;
            loop {
                match items.front_mut() {
                    None => break,
                    Some((Item::FenceWait(f, v), _, fresh)) => {
                        if fence(*f) >= *v {
                            items.pop_front();
                            continue;
                        }
                        *fresh = false;
                        break;
                    }
                    Some((Item::Signal(..), ..)) => {
                        if !segments.is_empty() {
                            break;
                        }
                        if let Some((Item::Signal(f, v), ..)) = items.pop_front() {
                            self.queues.retain(|(_, q)| !q.is_empty());
                            return Some(Ready::Signal { fence: f, value: v });
                        }
                        break;
                    }
                    Some((Item::Action(_), ..)) => {
                        if !segments.is_empty() {
                            break;
                        }
                        if let Some((Item::Action(act), ..)) = items.pop_front() {
                            self.queues.retain(|(_, q)| !q.is_empty());
                            return Some(Ready::Action(act));
                        }
                        break;
                    }
                    Some((Item::Segment(_, waits), since, fresh)) => {
                        let mut ok = true;
                        for wait in waits.iter() {
                            match wait.holds(read) {
                                Some(true) => {}
                                Some(false) => ok = false,
                                None => unknown += 1,
                            }
                        }
                        if !ok {
                            *fresh = false;
                            break;
                        }
                        if !*fresh {
                            waited.push((*queue, since.elapsed()));
                        }
                        if let Some((Item::Segment(s, _), ..)) = items.pop_front() {
                            segments.push((s, false));
                        }
                    }
                }
            }
            if !segments.is_empty() {
                let q = *queue;
                self.running.push(q);
                return Some(Ready::Segments {
                    queues: vec![q],
                    segments,
                    signals: Vec::new(),
                    waited,
                    unknown,
                    coalesced_bop_waits: 0,
                    hazard_splits: 0,
                });
            }
        }
        self.queues.retain(|(_, items)| !items.is_empty());
        None
    }

    /// Clears the running markers set by [`Self::take_ready_with`] once the
    /// popped batch has been handed to the host GPU.
    #[cfg(any(windows, test))]
    pub(crate) fn finish_running(&mut self, queues: &[usize]) {
        self.running.retain(|q| !queues.contains(q));
        self.queues.retain(|(_, items)| !items.is_empty());
    }

    /// Whether no queue holds anything and none is currently running.
    pub(crate) fn is_empty(&self) -> bool {
        self.running.is_empty() && self.queues.is_empty()
    }

    /// Whether `queue` holds anything or has a popped batch still launching.
    pub(crate) fn holds_work(&self, queue: usize) -> bool {
        self.running.contains(&queue)
            || self.queues.iter().any(|(q, items)| *q == queue && !items.is_empty())
    }

    /// Releases work held longer than `limit` at the heads of blocked
    /// queues, as if its waits held: the count of segments forced.
    // shortcut: a safety valve against waits nothing will satisfy (a write
    // we do not emulate), which would hold the title's fences back until
    // its 30 s watchdog ends the run; remove once runs show it never fires.
    pub(crate) fn force_older_than(&mut self, limit: Duration) -> Vec<(usize, Vec<ValueWait>)> {
        let mut forced = Vec::new();
        for (q, items) in &mut self.queues {
            if let Some((Item::Segment(_, waits), since, _)) = items.front_mut()
                && since.elapsed() >= limit
            {
                forced.push((*q, std::mem::take(waits)));
            }
        }
        forced
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::cell::RefCell;
    use std::collections::HashMap;

    /// Memory, fences and a log of what ran, for a recording executor.
    #[derive(Default)]
    struct Fake {
        memory: HashMap<u64, u64>,
        fences: HashMap<usize, u64>,
        log: Vec<String>,
        /// Bottom-of-pipe writes of each segment: (address, value).
        writes: HashMap<&'static str, Vec<(u64, u64)>>,
        /// Per-segment hazard flags for `can_join_batch`.
        flags: HashMap<&'static str, BatchFlags>,
    }

    fn ge(address: u64, reference: u64) -> ValueWait {
        ValueWait { address, bytes: 8, comparison: GREATER_EQUAL, reference, mask: u64::MAX }
    }

    fn pump_with(
        held: &mut Held<&'static str>,
        fake: &RefCell<Fake>,
        coalesce: bool,
        max_lists: usize,
    ) -> Pumped {
        let read = |a: u64, _| Some(*fake.borrow().memory.get(&a).unwrap_or(&0));
        let fence = |f: usize| *fake.borrow().fences.get(&f).unwrap_or(&0);
        let will_write = |seg: &&'static str, wait: &ValueWait| {
            fake.borrow()
                .writes
                .get(seg)
                .and_then(|writes| {
                    writes
                        .iter()
                        .rev()
                        .find(|&&(addr, _)| addr == wait.address)
                        .map(|&(_, v)| wait.holds_for(v) == Some(true))
                })
                .unwrap_or(false)
        };
        let can_join = |piece: &[(&'static str, bool)], next: &&'static str| {
            let f = fake.borrow();
            let existing = piece.iter().fold(BatchFlags::default(), |acc, (s, _)| {
                acc.merge(f.flags.get(s).copied().unwrap_or_default())
            });
            let next_flags = f.flags.get(next).copied().unwrap_or_default();
            can_join_batch(existing, next_flags, piece.len(), max_lists)
        };
        let mut run = |segments: Vec<(&'static str, bool)>| {
            let mut f = fake.borrow_mut();
            let mut names = Vec::with_capacity(segments.len());
            for (s, barrier) in &segments {
                for (a, v) in f.writes.get(s).cloned().unwrap_or_default() {
                    f.memory.insert(a, v);
                }
                if *barrier {
                    names.push(format!("|{s}"));
                } else {
                    names.push((*s).to_owned());
                }
            }
            f.log.push(format!("run: {}", names.join(" ")));
        };
        let mut signal = |fence: usize, value: u64| {
            let mut f = fake.borrow_mut();
            f.fences.insert(fence, value);
            f.log.push(format!("signal {fence} {value}"));
        };
        held.pump(&mut World {
            read: &read,
            fence: &fence,
            will_write: &will_write,
            can_join: &can_join,
            coalesce,
            run: &mut run,
            signal: &mut signal,
        })
    }

    fn pump(held: &mut Held<&'static str>, fake: &RefCell<Fake>) -> Pumped {
        pump_with(held, fake, false, 64)
    }

    fn pump_coalesced(held: &mut Held<&'static str>, fake: &RefCell<Fake>) -> Pumped {
        pump_with(held, fake, true, 64)
    }

    #[test]
    fn comparisons_test_the_masked_value() {
        let read = |_, _| Some(0x1_0000_0005);
        let w = |comparison, reference| ValueWait { address: 8, bytes: 4, comparison, reference, mask: 0xffff_ffff };
        let expect = [(NEVER, false), (LESS, false), (EQUAL, true), (LESS_EQUAL, true), (GREATER, false), (NOT_EQUAL, false), (GREATER_EQUAL, true), (ALWAYS, true)];
        for (cmp, holds) in expect {
            assert_eq!(w(cmp, 5).holds(&read), Some(holds), "comparison {cmp}");
            assert_eq!(w(cmp, 5).holds_for(0x1_0000_0005), Some(holds), "holds_for {cmp}");
        }
        assert_eq!(w(LESS, 6).holds(&read), Some(true));
        assert_eq!(w(GREATER, 4).holds(&read), Some(true));
        assert_eq!(w(0, 5).holds(&read), None, "unknown comparison");
        assert_eq!(w(9, 5).holds(&read), None);
        assert_eq!(w(0, 5).holds_for(5), None);
        assert_eq!(w(EQUAL, 5).holds(&|_, _| None), None, "unreadable memory");
    }

    /// A wait is tested after the earlier segments of its own queue ran:
    /// one that sets the value back holds the next segment's wait.
    #[test]
    fn a_wait_reads_what_earlier_segments_of_its_queue_wrote() {
        let fake = RefCell::new(Fake::default());
        fake.borrow_mut().memory.insert(0x100, 1);
        fake.borrow_mut().writes.insert("reset", vec![(0x100, 0)]);
        let eq = |reference| ValueWait { address: 0x100, bytes: 8, comparison: EQUAL, reference, mask: u64::MAX };
        let mut held = Held::default();
        held.push(1, Item::Segment("reset", vec![]));
        held.push(1, Item::Segment("after", vec![eq(1)]));
        pump(&mut held, &fake);
        assert_eq!(fake.borrow().log, ["run: reset"], "the wait reads the 0 the reset wrote");
        assert!(held.holds_work(1));
        fake.borrow_mut().memory.insert(0x100, 1);
        pump(&mut held, &fake);
        assert_eq!(fake.borrow().log, ["run: reset", "run: after"]);
    }

    /// The compute queue's list waits for a value the graphics queue's list
    /// writes at its end: it runs after it.
    #[test]
    fn a_list_waits_for_another_queues_write() {
        let fake = RefCell::new(Fake::default());
        fake.borrow_mut().writes.insert("draw", vec![(0x100, 7)]);
        let mut held = Held::default();
        held.push(2, Item::Segment("depth", vec![ge(0x100, 7)]));
        let p = pump(&mut held, &fake);
        assert_eq!((p.pieces, p.segments), (0, 0), "held: the value is not there");
        assert!(held.holds_work(2));
        held.push(1, Item::Segment("draw", vec![]));
        let p = pump(&mut held, &fake);
        assert_eq!(fake.borrow().log, ["run: draw", "run: depth"]);
        assert_eq!((p.pieces, p.segments, p.waited.len()), (2, 2, 1));
        assert_eq!(p.waited[0].0, 2);
        assert!(!held.holds_work(2));
    }

    #[test]
    fn a_wait_that_holds_defers_nothing() {
        let fake = RefCell::new(Fake::default());
        fake.borrow_mut().memory.insert(0x100, 9);
        let mut held = Held::default();
        held.push(2, Item::Segment("a", vec![ge(0x100, 7)]));
        held.push(2, Item::Segment("b", vec![]));
        let p = pump(&mut held, &fake);
        assert_eq!(fake.borrow().log, ["run: a b"], "one piece");
        assert_eq!((p.pieces, p.waited.len()), (1, 0));
    }

    /// A list's second segment waits for a value the other queue writes
    /// after reading what the first segment wrote: the cut is what lets
    /// both make progress.
    #[test]
    fn chained_waits_across_two_queues() {
        let fake = RefCell::new(Fake::default());
        {
            let mut f = fake.borrow_mut();
            f.writes.insert("c1", vec![(0x10, 1)]);
            f.writes.insert("g1", vec![(0x20, 1)]);
            f.writes.insert("c2", vec![(0x30, 1)]);
        }
        let mut held = Held::default();
        // Compute: [wait 0x08][c1, writes 0x10][wait 0x20][c2, writes 0x30].
        held.push(2, Item::Segment("c1", vec![ge(0x08, 1)]));
        held.push(2, Item::Segment("c2", vec![ge(0x20, 1)]));
        // Graphics: [g0, writes 0x08], [wait 0x10][g1, writes 0x20], [wait 0x30][g2].
        held.push(1, Item::Segment("g0", vec![]));
        fake.borrow_mut().writes.insert("g0", vec![(0x08, 1)]);
        held.push(1, Item::Segment("g1", vec![ge(0x10, 1)]));
        held.push(1, Item::Segment("g2", vec![ge(0x30, 1)]));
        let p = pump(&mut held, &fake);
        assert_eq!(fake.borrow().log, ["run: g0", "run: c1", "run: g1", "run: c2", "run: g2"]);
        assert_eq!(p.pieces, 5, "legacy per-queue mode keeps 5 pieces");
        assert!(held.blocked(&|_, _| Some(0)).is_empty());
    }

    /// Coalesced mode joins the entire cross-queue BoP chain into a single
    /// host submission with a UAV barrier before every BoP-unblocked segment.
    #[test]
    fn coalesced_chained_bop_waits_join_into_one_piece_with_barriers() {
        let fake = RefCell::new(Fake::default());
        {
            let mut f = fake.borrow_mut();
            f.writes.insert("g0", vec![(0x08, 1)]);
            f.writes.insert("c1", vec![(0x10, 1)]);
            f.writes.insert("g1", vec![(0x20, 1)]);
            f.writes.insert("c2", vec![(0x30, 1)]);
        }
        let mut held = Held::default();
        held.push(2, Item::Segment("c1", vec![ge(0x08, 1)]));
        held.push(2, Item::Segment("c2", vec![ge(0x20, 1)]));
        // First pump marks c1 as waiting (not fresh).
        assert_eq!(pump_coalesced(&mut held, &fake).pieces, 0);
        held.push(1, Item::Segment("g0", vec![]));
        held.push(1, Item::Segment("g1", vec![ge(0x10, 1)]));
        held.push(1, Item::Segment("g2", vec![ge(0x30, 1)]));
        let p = pump_coalesced(&mut held, &fake);
        assert_eq!(fake.borrow().log, ["run: g0 |c1 |g1 |c2 |g2"]);
        assert_eq!(p.pieces, 1);
        assert_eq!(p.segments, 5);
        assert_eq!(p.coalesced_bop_waits, 4);
        assert_eq!(p.hazard_splits, 0);
        assert_eq!(p.waited.len(), 4, "c1, g1, c2, g2 each waited once before BoP unblocked them");
        assert!(held.is_empty());
    }

    /// A submission-phase hazard (e.g., GPU write before CPU read, or title
    /// copy before explicit texture upload) flushes the piece and its deferred
    /// signals before running the conflicting segment.
    #[test]
    fn coalesced_hazard_split_flushes_piece_and_signals_first() {
        let fake = RefCell::new(Fake::default());
        {
            let mut f = fake.borrow_mut();
            f.writes.insert("gpu_prod", vec![(0x100, 1)]);
            f.flags.insert("gpu_prod", BatchFlags { has_gpu_write: true, ..Default::default() });
            f.flags.insert("cpu_reader", BatchFlags { has_cpu_read: true, ..Default::default() });
            f.flags.insert("title_copier", BatchFlags { has_title_copy: true, ..Default::default() });
            f.flags.insert("tex_user", BatchFlags { has_textures: true, ..Default::default() });
        }
        let mut held = Held::default();
        held.push(2, Item::Segment("cpu_reader", vec![ge(0x100, 1)]));
        held.push(1, Item::Segment("gpu_prod", vec![]));
        held.push(1, Item::Signal(7, 42));
        let p = pump_coalesced(&mut held, &fake);
        assert_eq!(
            fake.borrow().log,
            ["run: gpu_prod", "signal 7 42", "run: cpu_reader"],
            "hazard split flushed gpu_prod and its deferred signal before cpu_reader"
        );
        assert_eq!(p.pieces, 2);
        assert_eq!(p.hazard_splits, 1);
        assert_eq!(p.coalesced_bop_waits, 0);

        // Title copy followed by explicit texture upload also splits.
        fake.borrow_mut().log.clear();
        held.push(1, Item::Segment("title_copier", vec![]));
        held.push(2, Item::Segment("tex_user", vec![]));
        let p2 = pump_coalesced(&mut held, &fake);
        assert_eq!(fake.borrow().log, ["run: title_copier", "run: tex_user"]);
        assert_eq!(p2.pieces, 2);
        assert_eq!(p2.hazard_splits, 1);
    }

    /// Batch list cap (`max_lists`) splits oversized coalesced pieces across
    /// queue boundaries without breaking a single queue's own submission.
    #[test]
    fn coalesced_max_lists_cap_splits_piece() {
        let fake = RefCell::new(Fake::default());
        let mut held = Held::default();
        held.push(1, Item::Segment("a", vec![]));
        held.push(1, Item::Segment("b", vec![]));
        held.push(2, Item::Segment("c", vec![]));
        held.push(2, Item::Segment("d", vec![]));
        held.push(3, Item::Segment("e", vec![]));
        let p = pump_with(&mut held, &fake, true, 2);
        assert_eq!(fake.borrow().log, ["run: a b", "run: c d", "run: e"]);
        assert_eq!(p.pieces, 3);
        assert_eq!(p.segments, 5);
        assert_eq!(p.hazard_splits, 2);
    }

    /// Consecutive ready segments on the same queue (e.g., one multi-list
    /// submission containing mixed GPU writes, CPU copies/fills, and texture
    /// uploads) are never split by `can_join`, while a cross-queue BoP-dependent
    /// compute segment still coalesces onto that piece.
    #[test]
    fn coalesced_never_splits_same_queue_consecutive_segments() {
        let fake = RefCell::new(Fake::default());
        {
            let mut f = fake.borrow_mut();
            f.flags.insert("g0", BatchFlags { has_gpu_write: true, ..Default::default() });
            f.flags.insert("g1", BatchFlags { has_title_copy: true, has_cpu_read: true, ..Default::default() });
            f.flags.insert("g2", BatchFlags { has_textures: true, has_gpu_write: true, ..Default::default() });
            f.writes.insert("g2", vec![(0x200, 1)]);
            f.flags.insert("c0", BatchFlags { has_gpu_write: true, ..Default::default() });
        }
        let mut held = Held::default();
        held.push(2, Item::Segment("c0", vec![ge(0x200, 1)]));
        assert_eq!(pump_coalesced(&mut held, &fake).pieces, 0);
        held.push(1, Item::Segment("g0", vec![]));
        held.push(1, Item::Segment("g1", vec![]));
        held.push(1, Item::Segment("g2", vec![]));
        // Even with max_lists=2, same-queue g0..g2 are never shredded, and with max_lists=64 c0 joins too.
        let p = pump_coalesced(&mut held, &fake);
        assert_eq!(fake.borrow().log, ["run: g0 g1 g2 |c0"]);
        assert_eq!(p.pieces, 1);
        assert_eq!(p.segments, 4);
        assert_eq!(p.coalesced_bop_waits, 1);
        assert_eq!(p.hazard_splits, 0);

        // Even when a single queue has more segments than `max_lists`, it stays one piece.
        fake.borrow_mut().log.clear();
        held.push(1, Item::Segment("g0", vec![]));
        held.push(1, Item::Segment("g1", vec![]));
        held.push(1, Item::Segment("g2", vec![]));
        let p2 = pump_with(&mut held, &fake, true, 2);
        assert_eq!(fake.borrow().log, ["run: g0 g1 g2"]);
        assert_eq!(p2.pieces, 1);
        assert_eq!(p2.hazard_splits, 0);
    }

    /// Deferred `Item::Signal` stays after its piece completes, while another
    /// queue's `Item::FenceWait` for that pending signal joins behind a barrier.
    #[test]
    fn coalesced_deferred_signal_unblocks_fence_wait_with_barrier() {
        let fake = RefCell::new(Fake::default());
        let mut held = Held::default();
        held.push(1, Item::Segment("work", vec![]));
        held.push(1, Item::Signal(5, 3));
        held.push(1, Item::Segment("more", vec![]));
        held.push(2, Item::FenceWait(5, 3));
        held.push(2, Item::Segment("consumer", vec![]));
        let p = pump_coalesced(&mut held, &fake);
        assert_eq!(
            fake.borrow().log,
            ["run: work more |consumer", "signal 5 3"]
        );
        assert_eq!(p.pieces, 1);
        assert_eq!(p.segments, 3);
        assert_eq!(fake.borrow().fences.get(&5), Some(&3));
    }

    /// Order within a queue never changes in coalesced mode: a later segment
    /// whose waits hold does not pass a held one on the same queue.
    #[test]
    fn coalesced_preserves_per_queue_fifo_order() {
        let fake = RefCell::new(Fake::default());
        let mut held = Held::default();
        held.push(2, Item::Segment("first", vec![ge(0x100, 1)]));
        held.push(2, Item::Segment("second", vec![]));
        held.push(1, Item::Segment("independent", vec![]));
        let p1 = pump_coalesced(&mut held, &fake);
        assert_eq!(fake.borrow().log, ["run: independent"]);
        assert_eq!(p1.pieces, 1);
        assert!(held.holds_work(2));
        fake.borrow_mut().memory.insert(0x100, 1);
        let p2 = pump_coalesced(&mut held, &fake);
        assert_eq!(fake.borrow().log, ["run: independent", "run: first second"]);
        assert_eq!(p2.pieces, 1);
        assert!(held.is_empty());
    }

    /// Order within a queue never changes: a later segment whose waits hold
    /// does not pass a held one.
    #[test]
    fn a_queue_keeps_its_order() {
        let fake = RefCell::new(Fake::default());
        let mut held = Held::default();
        held.push(2, Item::Segment("first", vec![ge(0x100, 1)]));
        held.push(2, Item::Segment("second", vec![]));
        pump(&mut held, &fake);
        assert!(fake.borrow().log.is_empty());
        fake.borrow_mut().memory.insert(0x100, 1);
        pump(&mut held, &fake);
        assert_eq!(fake.borrow().log, ["run: first second"]);
    }

    /// A Signal behind held work is delivered after it: the title's fence
    /// (and its memory) does not move before the work ran.
    #[test]
    fn a_signal_waits_for_the_work_before_it() {
        let fake = RefCell::new(Fake::default());
        let mut held = Held::default();
        held.push(2, Item::Segment("work", vec![ge(0x100, 1)]));
        held.push(2, Item::Signal(5, 3));
        held.push(2, Item::Segment("after", vec![]));
        pump(&mut held, &fake);
        assert_eq!(fake.borrow().fences.get(&5), None, "not signalled yet");
        fake.borrow_mut().memory.insert(0x100, 1);
        pump(&mut held, &fake);
        assert_eq!(fake.borrow().log, ["run: work", "signal 5 3", "run: after"]);
    }

    #[test]
    fn a_fence_wait_holds_its_queue() {
        let fake = RefCell::new(Fake::default());
        let mut held = Held::default();
        held.push(2, Item::FenceWait(5, 2));
        held.push(2, Item::Segment("after", vec![]));
        pump(&mut held, &fake);
        assert!(fake.borrow().log.is_empty());
        held.push(1, Item::Signal(5, 2));
        pump(&mut held, &fake);
        assert_eq!(fake.borrow().log, ["signal 5 2", "run: after"]);
    }

    /// The frame ends with work held: it stays held (and is reported), and
    /// the next frame's write releases it.
    #[test]
    fn work_held_at_the_frames_end_stays_held() {
        let fake = RefCell::new(Fake::default());
        let mut held = Held::default();
        held.push(2, Item::Segment("late", vec![ge(0x100, 2)]));
        held.push(2, Item::Signal(5, 1));
        pump(&mut held, &fake);
        let read = |a: u64, _| Some(*fake.borrow().memory.get(&a).unwrap_or(&0));
        assert!(!held.is_empty());
        let blocked = held.blocked(&read);
        assert_eq!(blocked.len(), 1);
        assert_eq!((blocked[0].0, blocked[0].1, blocked[0].3.clone()), (2, 2, vec![ge(0x100, 2)]));
        fake.borrow_mut().memory.insert(0x100, 2);
        let p = pump(&mut held, &fake);
        assert_eq!(p.waited.len(), 1);
        assert!(held.is_empty());
        assert_eq!(fake.borrow().log, ["run: late", "signal 5 1"]);
    }

    #[test]
    fn unknown_comparisons_are_counted_and_do_not_hold_work() {
        let fake = RefCell::new(Fake::default());
        let mut held = Held::default();
        let odd = ValueWait { address: 0x100, bytes: 4, comparison: 12, reference: 1, mask: 0xffff_ffff };
        held.push(2, Item::Segment("odd", vec![odd]));
        let p = pump(&mut held, &fake);
        assert_eq!((p.segments, p.unknown), (1, 1));
    }

    #[test]
    fn the_valve_releases_only_old_heads() {
        let fake = RefCell::new(Fake::default());
        let mut held = Held::default();
        held.push(2, Item::Segment("stuck", vec![ge(0x100, 1)]));
        assert!(held.force_older_than(Duration::from_secs(60)).is_empty());
        let forced = held.force_older_than(Duration::ZERO);
        assert_eq!(forced, [(2, vec![ge(0x100, 1)])]);
        pump(&mut held, &fake);
        assert_eq!(fake.borrow().log, ["run: stuck"]);
    }

    #[test]
    fn batch_flags_and_can_join_enforce_phase_hazards_and_caps() {
        let clean = BatchFlags::default();
        let gpu_write = BatchFlags { has_gpu_write: true, ..Default::default() };
        let title_copy = BatchFlags { has_title_copy: true, has_cpu_read: true, ..Default::default() };
        let cpu_read = BatchFlags { has_cpu_read: true, ..Default::default() };
        let after = BatchFlags { has_after: true, ..Default::default() };
        let tex = BatchFlags { has_textures: true, ..Default::default() };

        assert!(can_join_batch(clean, clean, 1, 64));
        assert!(can_join_batch(gpu_write, tex, 1, 64), "GPU write -> texture upload is safe");
        assert!(can_join_batch(gpu_write, gpu_write, 1, 64), "GPU write -> GPU write is safe");
        assert!(!can_join_batch(gpu_write, cpu_read, 1, 64), "GPU write -> CPU read splits");
        assert!(!can_join_batch(after, cpu_read, 1, 64), "defer_after -> CPU read splits");
        assert!(!can_join_batch(title_copy, tex, 1, 64), "title copy -> texture upload splits");
        assert!(!can_join_batch(after, tex, 1, 64), "defer_after -> texture upload splits");
        assert!(!can_join_batch(clean, clean, 64, 64), "cap reached splits");
    }

    thread_local! {
        /// What ran, in order, for tests whose actions log too (an action
        /// is `Send + 'static`, so it cannot borrow a test's `Fake`).
        static ORDER: RefCell<Vec<String>> = const { RefCell::new(Vec::new()) };
    }

    fn logged(what: &str) {
        ORDER.with(|o| o.borrow_mut().push(what.to_owned()));
    }

    fn act(name: &'static str) -> Item<&'static str> {
        Item::Action(Box::new(move || logged(&format!("act {name}"))))
    }

    /// Pumps with no waits, logging pieces and actions to [`ORDER`].
    fn pump_ordered(held: &mut Held<&'static str>, coalesce: bool) -> Vec<String> {
        ORDER.with(|o| o.borrow_mut().clear());
        let read = |_: u64, _| Some(0u64);
        let fence = |_: usize| 0u64;
        let no_write = |_: &&'static str, _: &ValueWait| false;
        let join = |_: &[(&'static str, bool)], _: &&'static str| true;
        let mut run = |segments: Vec<(&'static str, bool)>| {
            let names: Vec<&str> = segments.iter().map(|(s, _)| *s).collect();
            logged(&format!("run: {}", names.join(" ")));
        };
        let mut signal = |f: usize, v: u64| logged(&format!("signal {f} {v}"));
        held.pump(&mut World {
            read: &read,
            fence: &fence,
            will_write: &no_write,
            can_join: &join,
            coalesce,
            run: &mut run,
            signal: &mut signal,
        });
        ORDER.with(|o| o.borrow().clone())
    }

    /// An action keeps its place in its queue in both pumps: the work before
    /// it runs first, and nothing after it joins that work. Other queues
    /// still join.
    #[test]
    fn an_action_runs_between_its_queues_work() {
        for coalesce in [false, true] {
            let mut held = Held::default();
            held.push(1, Item::Segment("a", vec![]));
            held.push(1, act("map"));
            held.push(1, Item::Segment("b", vec![]));
            let order = pump_ordered(&mut held, coalesce);
            assert_eq!(order, ["run: a", "act map", "run: b"], "coalesce {coalesce}");
            assert!(held.is_empty());
        }
        // A signal before the action is delivered before it.
        let mut held = Held::default();
        held.push(1, Item::Segment("a", vec![]));
        held.push(1, Item::Signal(9, 1));
        held.push(1, act("map"));
        held.push(1, Item::Segment("b", vec![]));
        assert_eq!(
            pump_ordered(&mut held, true),
            ["run: a", "signal 9 1", "act map", "run: b"]
        );
    }

    /// Taken for the submit worker, an action ends its queue's part of a
    /// piece and comes out on its own once that piece was handed over;
    /// coalescing, another queue's ready work still joins the piece. The
    /// work after the action comes next.
    #[test]
    fn take_ready_hands_out_an_action_after_its_queues_work() {
        let read = |_: u64, _| Some(0u64);
        let fence = |_: usize| 0u64;
        let no_write = |_: &&str, _: &ValueWait| false;
        let join = |_: &[(&str, bool)], _: &&str| true;
        for coalesce in [true, false] {
            let mut held = Held::default();
            held.push(1, Item::Segment("a", vec![]));
            held.push(1, act("map"));
            held.push(1, Item::Segment("b", vec![]));
            held.push(2, Item::Segment("c", vec![]));
            let take = |held: &mut Held<&'static str>| {
                held.take_ready_with(&read, &fence, &no_write, &join, coalesce)
            };

            let Some(Ready::Segments { queues, segments, .. }) = take(&mut held) else {
                panic!("expected segments first (coalesce {coalesce})");
            };
            let expected: &[(&str, bool)] = if coalesce { &[("a", false), ("c", false)] } else { &[("a", false)] };
            assert_eq!(segments, expected);
            assert!(held.holds_work(1), "running until finished");
            // While queue 1 is handed over, its action stays; per queue,
            // queue 2's work comes out on its own.
            match take(&mut held) {
                None => assert!(coalesce),
                Some(Ready::Segments { queues, segments, .. }) => {
                    assert!(!coalesce);
                    assert_eq!(segments, [("c", false)]);
                    held.finish_running(&queues);
                }
                Some(_) => panic!("the action came out while its queue's work was being handed over"),
            }
            held.finish_running(&queues);
            let Some(Ready::Action(action)) = take(&mut held) else {
                panic!("expected the action (coalesce {coalesce})");
            };
            ORDER.with(|o| o.borrow_mut().clear());
            action();
            assert_eq!(ORDER.with(|o| o.borrow().clone()), ["act map"]);
            let mut rest = Vec::new();
            while let Some(Ready::Segments { queues, segments, .. }) = take(&mut held) {
                rest.extend(segments.into_iter().map(|(s, _)| s));
                held.finish_running(&queues);
            }
            assert_eq!(rest, ["b"], "coalesce {coalesce}");
            assert!(held.is_empty(), "coalesce {coalesce}");
        }
    }

    #[test]
    fn take_ready_marks_queue_running_until_finished() {
        let mut held = Held::default();
        held.push(1, Item::Segment("a", vec![]));
        held.push(1, Item::Signal(9, 1));
        held.push(1, Item::Segment("b", vec![]));
        let read = |_: u64, _| Some(0u64);
        let fence = |_: usize| 0u64;
        let no_write = |_: &&str, _: &ValueWait| false;
        let no_join = |_: &[(&str, bool)], _: &&str| false;

        match held.take_ready_with(&read, &fence, &no_write, &no_join, false) {
            Some(Ready::Segments {
                queues, segments, ..
            }) => {
                assert_eq!(queues, [1]);
                assert_eq!(segments, [("a", false)]);
            }
            _ => panic!("expected segments"),
        }
        // Queue 1 is marked running: holds_work stays true and take_ready_with
        // must not pop its queued Signal before finish_running(&[1]).
        assert!(held.holds_work(1));
        assert!(!held.is_empty());
        assert!(held
            .take_ready_with(&read, &fence, &no_write, &no_join, false)
            .is_none());

        held.finish_running(&[1]);
        match held.take_ready_with(&read, &fence, &no_write, &no_join, false) {
            Some(Ready::Signal { fence: f, value }) => assert_eq!((f, value), (9, 1)),
            _ => panic!("expected signal"),
        }
        match held.take_ready_with(&read, &fence, &no_write, &no_join, false) {
            Some(Ready::Segments {
                queues, segments, ..
            }) => {
                assert_eq!(queues, [1]);
                assert_eq!(segments, [("b", false)]);
            }
            _ => panic!("expected trailing segment"),
        }
        held.finish_running(&[1]);
        assert!(!held.holds_work(1));
        assert!(held.is_empty());
    }

    #[test]
    fn take_ready_coalesced_merges_cross_queue_bop_and_deferred_signals() {
        let mut held = Held::default();
        // Queue 2 waits on BoP 0x100 >= 1 written by Queue 1's segment "g1".
        held.push(2, Item::Segment("c1", vec![ge(0x100, 1)]));
        // Queue 3 waits on fence 9 >= 1 signalled after "g1".
        held.push(3, Item::FenceWait(9, 1));
        held.push(3, Item::Segment("c2", vec![]));
        // Queue 1 produces "g1" (writes 0x100 = 1), signals fence 9, then "g2".
        held.push(1, Item::Segment("g1", vec![]));
        held.push(1, Item::Signal(9, 1));
        held.push(1, Item::Segment("g2", vec![]));

        let read = |_: u64, _| Some(0u64);
        let fence = |_: usize| 0u64;
        let will_write = |seg: &&str, wait: &ValueWait| *seg == "g1" && wait.address == 0x100;
        let can_join = |_: &[(&str, bool)], next: &&str| *next != "c2";

        // First coalesced batch picks up Queue 1 ("g1", deferred Signal(9,1), "g2")
        // AND Queue 2 ("c1" via BoP with barrier_before=true), then splits before
        // Queue 3 ("c2") because `can_join` rejects it at the join boundary.
        match held.take_ready_with(&read, &fence, &will_write, &can_join, true) {
            Some(Ready::Segments {
                queues,
                segments,
                signals,
                coalesced_bop_waits,
                hazard_splits,
                ..
            }) => {
                assert_eq!(queues, [1, 2]);
                assert_eq!(segments, [("g1", false), ("g2", false), ("c1", true)]);
                assert_eq!(signals, [(9, 1)]);
                assert_eq!(coalesced_bop_waits, 1);
                assert_eq!(hazard_splits, 1);
            }
            _ => panic!("expected coalesced batch"),
        }
        assert!(held.holds_work(1));
        assert!(held.holds_work(2));
        held.finish_running(&[1, 2]);

        // Queue 3's FenceWait(9, 1) was restored when "c2" split out, so "c2"
        // stays blocked until fence 9 actually reaches 1.
        assert!(held
            .take_ready_with(&read, &fence, &will_write, &can_join, true)
            .is_none());

        // Once the first batch's signal (9, 1) has landed, Queue 3's "c2" runs.
        let fence_after = |f: usize| if f == 9 { 1 } else { 0 };
        match held.take_ready_with(&read, &fence_after, &will_write, &can_join, true) {
            Some(Ready::Segments {
                queues, segments, ..
            }) => {
                assert_eq!(queues, [3]);
                assert_eq!(segments, [("c2", false)]);
            }
            _ => panic!("expected c2 after fence"),
        }
        held.finish_running(&[3]);
        assert!(held.is_empty());
    }
}
