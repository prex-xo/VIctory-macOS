//! `VICTORY_GPU_TIMING_CUTS`: GPU time per pipeline from wall time alone,
//! for hosts whose D3D12 timestamps do not advance.
//!
//! `VICTORY_GPU_TIMING` runs each host list on its own and waits for it, so
//! a list's wall time is its GPU time; which command of the list took it
//! comes from timestamp queries, and under D3DMetal those all read 0. With
//! the cuts on, a title list goes on in a fresh host list (a segment, as
//! after a wait: `crate::list`) wherever the pipeline changes, so every host
//! list runs one pipeline and its wall time is that pipeline's.
//!
//! Two modes, and both leave nearly all of the frame as it is: only a few
//! pipelines at a time get lists of their own, everything else stays
//! together. A list of shader hashes (`<hash>,<hash>`): the pipelines with a
//! listed shader. `sweep` (or `sweep:<n>`, 16 by default): `n` pipelines at
//! a time of those the title drew with since the last frame-rate line, in
//! the order it first drew with them, the next `n` after every such line,
//! round and round, so that a few minutes in one place time every pipeline
//! of the scene, draws included. A pipeline's figures are per frame it had
//! lists in, as its turn starts and ends inside a report's frames.
//!
//! Cutting at every change of pipeline instead was tried and is not here: a
//! world frame of Grand Theft Auto VI became about 2,000 host lists, video
//! memory grew by 12 GB, single lists stalled for seconds, and a pipeline
//! that costs 0.7 ms a frame listed alone was blamed for 2 to 15 s. The
//! report gives the lists' floor (the tenth percentile of all lists' wall
//! time) and each pipeline's time above it.
//!
//! Plain std: the decisions and the sums, tested everywhere. The host list
//! holds a [`Timed`], `crate::list` cuts, `crate::host` times and logs.
#![allow(dead_code)]

use std::collections::HashMap;

/// What the switch asks for.
#[derive(Clone, Debug, PartialEq, Eq)]
pub(crate) enum Mode {
    Off,
    /// Lists of their own for the pipelines with one of these shader
    /// hashes (lower case).
    Listed(Vec<String>),
    /// Lists of their own for this many pipelines at a time, the next ones
    /// after every report.
    Sweep(usize),
}

/// Pipelines a sweep times at once, unless the switch says.
pub(crate) const SWEEP: usize = 16;

/// The mode a value of `VICTORY_GPU_TIMING_CUTS` selects: off when unset or
/// `0` (as every switch); a comma-separated list of 16-digit shader hashes
/// for those pipelines; `sweep:<n>` for `n` pipelines at a time; anything
/// else that is on, a sweep of [`SWEEP`].
pub(crate) fn mode_of(value: Option<&str>) -> Mode {
    let Some(value) = value.map(str::trim) else {
        return Mode::Off;
    };
    if value == "0" {
        return Mode::Off;
    }
    let hashes: Vec<String> = value
        .split(',')
        .map(|h| h.trim().to_ascii_lowercase())
        .filter(|h| !h.is_empty())
        .collect();
    let hash = |h: &String| h.len() == 16 && h.bytes().all(|b| b.is_ascii_hexdigit());
    if !hashes.is_empty() && hashes.iter().all(hash) {
        return Mode::Listed(hashes);
    }
    let n = value
        .to_ascii_lowercase()
        .strip_prefix("sweep:")
        .and_then(|n| n.trim().parse().ok())
        .filter(|&n| n > 0);
    Mode::Sweep(n.unwrap_or(SWEEP))
}

/// Whether a pipeline named by its shaders (`[compute 0123..]`) has one of
/// the `listed` hashes.
pub(crate) fn lists(listed: &[String], shaders: &str) -> bool {
    listed.iter().any(|h| shaders.contains(h.as_str()))
}

/// The pipelines a sweep has seen, in the order the title first drew with
/// them, the ones it times now, and where it goes on.
#[derive(Default)]
pub(crate) struct Sweep {
    /// Each pipeline's place in the order.
    order: std::collections::HashMap<usize, usize>,
    /// The pipelines by place.
    seen: Vec<usize>,
    /// The places drawn with since the last report.
    live: std::collections::BTreeSet<usize>,
    /// The pipelines timed now.
    window: std::collections::HashSet<usize>,
    /// The place the next window starts at.
    next: usize,
}

impl Sweep {
    /// Notes a draw or dispatch of `pipeline`; whether the sweep times it
    /// now (a pipeline not seen before joins the end of the order, and has
    /// its turn after the next report at the earliest).
    pub(crate) fn times(&mut self, pipeline: usize) -> bool {
        let place = *self.order.entry(pipeline).or_insert_with(|| {
            self.seen.push(pipeline);
            self.seen.len() - 1
        });
        self.live.insert(place);
        self.window.contains(&pipeline)
    }

    /// After a report: the next `n` of the pipelines drawn with since the
    /// one before, from where the last window ended, round the end of the
    /// order. Pipelines the title no longer draws with (a menu's, a loading
    /// screen's) have no turn. Returns how many it times now and how many
    /// were drawn with.
    pub(crate) fn advance(&mut self, n: usize) -> (usize, usize) {
        let live = std::mem::take(&mut self.live);
        let from = self.next;
        let turn: Vec<usize> = live
            .range(from..)
            .chain(live.range(..from))
            .take(n)
            .copied()
            .collect();
        self.next = turn.last().map_or(0, |&place| place + 1);
        self.window = turn.iter().map(|&place| self.seen[place]).collect();
        (turn.len(), live.len())
    }
}

/// The key of the commands no list is cut between (those of pipelines not
/// listed or not in the sweep's window): no title pipeline has it (a null
/// pipeline draws nothing).
pub(crate) const REST: usize = 0;

/// What tells a command's list from another's, and whether as a dispatch:
/// the title pipeline and its kind if it gets lists of its own (`own`),
/// or [`REST`].
pub(crate) fn key(pipeline: usize, compute: bool, own: bool) -> (usize, bool) {
    if own && pipeline != REST {
        (pipeline, compute)
    } else {
        (REST, false)
    }
}

/// What a host list runs, once its first draw or dispatch is recorded.
#[derive(Clone, Debug, PartialEq, Eq)]
pub(crate) struct Timed {
    pub(crate) key: usize,
    pub(crate) compute: bool,
    /// Draws and dispatches recorded.
    pub(crate) commands: u32,
    /// The title pipeline's shaders, as the log names them.
    pub(crate) name: String,
}

/// Whether a command of `key` goes into a new host list: the list has
/// commands of another kind or pipeline. A list with none takes it.
pub(crate) fn cuts(current: Option<&Timed>, key: usize, compute: bool) -> bool {
    current.is_some_and(|t| t.key != key || t.compute != compute)
}

/// A timed list's line in the report: its kind and pipeline.
pub(crate) fn label(timed: Option<&Timed>) -> String {
    match timed {
        None => "no draw or dispatch (uploads, copies, clears)".into(),
        Some(t) if t.key == REST => "the draws and dispatches not cut apart".into(),
        Some(t) => format!("{} {}", if t.compute { "dispatch" } else { "draw" }, t.name),
    }
}

/// Wall time of the lists since the last report, by [`label`].
#[derive(Default)]
pub(crate) struct Walls {
    by: HashMap<String, Wall>,
    /// Every list's milliseconds, for the floor.
    all: Vec<f32>,
}

#[derive(Default)]
struct Wall {
    ms: f64,
    lists: u64,
    commands: u64,
    /// Frames it had lists in, and the last of them.
    frames: u64,
    last: Option<u64>,
}

/// One pipeline's share of the frames it had lists in.
#[derive(Clone, Debug, PartialEq)]
pub(crate) struct Row {
    pub(crate) label: String,
    /// Milliseconds a frame it had lists in.
    pub(crate) ms: f64,
    /// Of them, above the floor its lists would cost empty.
    pub(crate) above: f64,
    /// Lists and commands a frame it had lists in.
    pub(crate) lists: f64,
    pub(crate) commands: f64,
    /// The frames it had lists in.
    pub(crate) frames: u64,
}

impl Walls {
    /// A list that took `ms`, submitted in frame `frame` (frames do not go
    /// back).
    pub(crate) fn add(&mut self, ms: f64, timed: Option<&Timed>, frame: u64) {
        let e = self.by.entry(label(timed)).or_default();
        e.ms += ms;
        e.lists += 1;
        e.commands += u64::from(timed.map_or(0, |t| t.commands));
        if e.last != Some(frame) {
            e.last = Some(frame);
            e.frames += 1;
        }
        self.all.push(ms as f32);
    }

    /// What a list costs whatever it runs (submitting it and waiting for
    /// it): the tenth percentile of the lists' milliseconds.
    pub(crate) fn floor(&self) -> f64 {
        if self.all.is_empty() {
            return 0.0;
        }
        let mut all = self.all.clone();
        let at = all.len() / 10;
        let (_, floor, _) = all.select_nth_unstable_by(at, f32::total_cmp);
        f64::from(*floor)
    }

    /// Over `frames` frames: lists and milliseconds a frame, the floor, and
    /// the pipelines, each per frame it had lists in, costliest above the
    /// floor first.
    pub(crate) fn report(&self, frames: f64) -> (f64, f64, f64, Vec<Row>) {
        let frames = frames.max(1.0);
        let floor = self.floor();
        let mut rows: Vec<Row> = self
            .by
            .iter()
            .map(|(label, w)| {
                let ran = w.frames.max(1) as f64;
                Row {
                    label: label.clone(),
                    ms: w.ms / ran,
                    above: (w.ms - floor * w.lists as f64).max(0.0) / ran,
                    lists: w.lists as f64 / ran,
                    commands: w.commands as f64 / ran,
                    frames: w.frames,
                }
            })
            .collect();
        rows.sort_by(|a, b| {
            b.above
                .total_cmp(&a.above)
                .then_with(|| a.label.cmp(&b.label))
        });
        let total: f64 = self.by.values().map(|w| w.ms).sum();
        (self.all.len() as f64 / frames, total / frames, floor, rows)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn timed(key: usize, compute: bool, commands: u32) -> Timed {
        Timed {
            key,
            compute,
            commands,
            name: format!("[{key:x}]"),
        }
    }

    #[test]
    fn the_switch_is_off_unset_or_zero_and_names_its_modes() {
        assert_eq!(mode_of(None), Mode::Off);
        assert_eq!(mode_of(Some("0")), Mode::Off);
        for sweep in [
            "1",
            "sweep",
            " Sweep ",
            "",
            "true",
            "0123",
            "0123456789abcdeg",
            "sweep:0",
            "sweep:x",
        ] {
            assert_eq!(mode_of(Some(sweep)), Mode::Sweep(SWEEP), "{sweep:?}");
        }
        assert_eq!(mode_of(Some("sweep:8")), Mode::Sweep(8));
        assert_eq!(mode_of(Some("SWEEP: 40")), Mode::Sweep(40));
        let Mode::Listed(hashes) = mode_of(Some("B6DF66AF105D8B2D, 63381c3250b287fa,")) else {
            panic!("two hashes are a list");
        };
        assert_eq!(hashes, ["b6df66af105d8b2d", "63381c3250b287fa"]);
        assert!(lists(&hashes, "[compute 63381c3250b287fa]"));
        assert!(lists(
            &hashes,
            "[mesh 0000000000000001, pixel b6df66af105d8b2d]"
        ));
        assert!(!lists(&hashes, "[compute 0000000000000002]"));
    }

    /// Only the pipelines that get lists of their own leave the list the
    /// rest shares, draws and dispatches alike.
    #[test]
    fn only_a_pipeline_with_lists_of_its_own_is_cut_from_the_rest() {
        assert_eq!(key(0x1000, true, false), (REST, false));
        assert_eq!(key(0x1000, true, false), key(0x2000, false, false));
        assert_eq!(key(0x3000, true, true), (0x3000, true));
        assert_eq!(key(REST, false, true), (REST, false));
        let rest = timed(REST, false, 7);
        assert!(!cuts(Some(&rest), REST, false));
        assert!(cuts(Some(&rest), 0x3000, true));
        let own = timed(0x3000, true, 1);
        assert!(!cuts(Some(&own), 0x3000, true));
        assert!(cuts(Some(&own), REST, false));
    }

    /// A list takes its first command; then only its own pipeline's, and a
    /// compute pipeline's dispatch is not a draw of the same pointer.
    #[test]
    fn a_list_is_cut_where_the_pipeline_or_the_kind_changes() {
        assert!(!cuts(None, 0x1000, true));
        let list = timed(0x1000, true, 1);
        assert!(!cuts(Some(&list), 0x1000, true));
        assert!(cuts(Some(&list), 0x2000, true));
        assert!(cuts(Some(&list), 0x1000, false));
    }

    /// A sweep times a window of the pipelines drawn with since the last
    /// report, in the order they were first seen, moves on at each report,
    /// wraps, takes in pipelines seen later and passes over those the title
    /// stopped drawing with.
    #[test]
    fn a_sweep_goes_round_the_live_pipelines_a_window_at_a_time() {
        let mut sweep = Sweep::default();
        let all: Vec<usize> = (1..=5).map(|i| i * 0x100).collect();
        let draw = |sweep: &mut Sweep, with: &[usize]| -> Vec<usize> {
            with.iter().copied().filter(|&p| sweep.times(p)).collect()
        };
        // Nothing is timed before the first report.
        assert_eq!(draw(&mut sweep, &all), []);
        assert_eq!(sweep.advance(2), (2, 5));
        assert_eq!(draw(&mut sweep, &all), [0x100, 0x200]);
        assert_eq!(sweep.advance(2), (2, 5));
        assert_eq!(draw(&mut sweep, &all), [0x300, 0x400]);
        // The window wraps round the end.
        assert_eq!(sweep.advance(2), (2, 5));
        assert_eq!(draw(&mut sweep, &all), [0x100, 0x500]);
        // The title stops drawing with 0x200 and 0x300 and starts with
        // 0x600: the sweep goes on after 0x100 with what is drawn with.
        let now = [0x100, 0x400, 0x500, 0x600];
        assert_eq!(sweep.advance(2), (2, 5));
        assert_eq!(draw(&mut sweep, &now), []);
        assert_eq!(sweep.advance(2), (2, 4));
        assert_eq!(draw(&mut sweep, &now), [0x400, 0x500]);
        assert_eq!(sweep.advance(2), (2, 4));
        assert_eq!(draw(&mut sweep, &now), [0x100, 0x600]);
        // A window as wide as the scene times it all; no draws, no window.
        assert_eq!(sweep.advance(16), (4, 4));
        assert_eq!(draw(&mut sweep, &now), now);
        assert_eq!(sweep.advance(16), (4, 4));
        assert_eq!(sweep.advance(16), (0, 0));
        assert_eq!(draw(&mut sweep, &now), []);
    }

    /// The floor is what the cheap lists cost; a pipeline's time above it
    /// leaves that out for each of its lists, and ranks the report.
    #[test]
    fn the_report_ranks_pipelines_by_their_time_above_the_floor() {
        let mut walls = Walls::default();
        // Two frames: 18 cheap lists of pipeline A (0.25 ms each), two
        // heavy lists of B (10 ms each), one upload list.
        for i in 0..18 {
            walls.add(0.25, Some(&timed(0xa, false, 2)), i / 9);
        }
        for frame in 0..2 {
            walls.add(10.0, Some(&timed(0xb, true, 1)), frame);
        }
        walls.add(0.25, None, 1);
        assert_eq!(walls.floor(), 0.25);
        let (lists, ms, floor, rows) = walls.report(2.0);
        assert_eq!((lists, floor), (10.5, 0.25));
        assert!((ms - (18.0 * 0.25 + 20.0 + 0.25) / 2.0).abs() < 1e-9);
        assert_eq!(rows[0].label, "dispatch [b]");
        assert!((rows[0].ms - 10.0).abs() < 1e-9 && (rows[0].above - 9.75).abs() < 1e-9);
        assert_eq!(
            (rows[0].lists, rows[0].commands, rows[0].frames),
            (1.0, 1.0, 2)
        );
        let a = rows
            .iter()
            .find(|r| r.label == "draw [a]")
            .expect("pipeline A");
        assert_eq!(
            (a.above, a.lists, a.commands, a.frames),
            (0.0, 9.0, 18.0, 2)
        );
        assert!(
            rows.iter()
                .any(|r| r.label.starts_with("no draw or dispatch"))
        );
    }

    /// A pipeline whose turn covered three of a report's ten frames has
    /// its cost per frame it had lists in, not a third of it.
    #[test]
    fn a_pipeline_timed_in_part_of_the_frames_is_not_diluted() {
        let mut walls = Walls::default();
        for frame in 0..10 {
            walls.add(1.0, Some(&timed(REST, false, 100)), frame);
            if (4..7).contains(&frame) {
                walls.add(20.0, Some(&timed(0xc, true, 1)), frame);
            }
        }
        let (_, ms, _, rows) = walls.report(10.0);
        assert!((ms - 7.0).abs() < 1e-9);
        let c = &rows[0];
        assert_eq!(
            (c.label.as_str(), c.frames, c.lists),
            ("dispatch [c]", 3, 1.0)
        );
        assert!((c.ms - 20.0).abs() < 1e-9);
        let rest = rows
            .iter()
            .find(|r| r.label.starts_with("the draws"))
            .expect("the rest");
        assert_eq!((rest.frames, rest.ms), (10, 1.0));
    }

    #[test]
    fn the_commands_left_together_have_one_line() {
        let rest = timed(REST, false, 40);
        assert_eq!(label(Some(&rest)), "the draws and dispatches not cut apart");
        assert_eq!(Walls::default().report(0.0), (0.0, 0.0, 0.0, Vec::new()));
    }
}
