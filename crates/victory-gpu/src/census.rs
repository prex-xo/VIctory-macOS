//! `VICTORY_DISPATCH_CENSUS=1`: the compute work each host pipeline is given
//! a frame, logged after the frame-rate line for the 20 that run the most
//! lanes. D3DMetal labels no Metal encoder with the host's names (neither
//! `SetName` nor `BeginEvent` reaches a trace; `twin::tests::
//! probe_list_names_reach_metal`), so a Metal System Trace's expensive
//! compute encoders are matched to programs by this census instead.
//! Indirect dispatches are counted by their records: their group counts are
//! the GPU's.

use std::collections::HashMap;
use std::sync::atomic::{AtomicBool, Ordering::Relaxed};
use std::sync::{Mutex, OnceLock};

/// Per pipeline name: direct dispatches, their groups, their lanes, and
/// indirect dispatch records.
static WORK: Mutex<Option<HashMap<String, [u64; 4]>>> = Mutex::new(None);

pub(crate) fn on() -> bool {
    static ON: OnceLock<bool> = OnceLock::new();
    *ON.get_or_init(|| crate::switch::on("VICTORY_DISPATCH_CENSUS"))
}

fn add(name: &str, i: usize, n: u64, j: usize, m: u64) {
    let mut work = WORK.lock().unwrap_or_else(|e| e.into_inner());
    let row = work.get_or_insert_with(HashMap::new).entry(name.to_owned()).or_default();
    row[i] += n;
    row[j] += m;
}

/// A direct dispatch of `groups` groups of `size` lanes each.
pub(crate) fn direct(name: &str, groups: [u32; 3], size: Option<[u32; 3]>) {
    if !on() {
        return;
    }
    let g: u64 = groups.iter().map(|&v| u64::from(v)).product();
    let lanes = g * size.map_or(1, |s| s.iter().map(|&v| u64::from(v)).product());
    add(name, 0, 1, 1, g);
    add(name, 2, lanes, 2, 0);
}

/// An indirect dispatch of at most `records` records.
pub(crate) fn indirect(name: &str, records: u32) {
    if on() {
        add(name, 3, u64::from(records), 3, 0);
    }
}

/// The rows, most lanes first (then most indirect records), at most `n`.
fn top(work: HashMap<String, [u64; 4]>, n: usize) -> Vec<(String, [u64; 4])> {
    let mut rows: Vec<_> = work.into_iter().collect();
    rows.sort_by(|a, b| b.1[2].cmp(&a.1[2]).then(b.1[3].cmp(&a.1[3])).then(a.0.cmp(&b.0)));
    rows.truncate(n);
    rows
}

pub(crate) fn log(frames: f64) {
    static SAID: AtomicBool = AtomicBool::new(false);
    let Some(work) = WORK.lock().unwrap_or_else(|e| e.into_inner()).take() else {
        return;
    };
    if !SAID.swap(true, Relaxed) {
        tracing::info!("dispatch census (VICTORY_DISPATCH_CENSUS): per frame, by host pipeline, most lanes first");
    }
    for (name, [n, groups, lanes, records]) in top(work, 20) {
        tracing::info!(
            "  census {:.1} dispatches, {:.0} groups, {:.2} M lanes, {:.1} indirect records: {name}",
            n as f64 / frames,
            groups as f64 / frames,
            lanes as f64 / frames / 1e6,
            records as f64 / frames,
        );
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn most_lanes_first() {
        let mut w = HashMap::new();
        w.insert("a".to_owned(), [1, 10, 640, 0]);
        w.insert("b".to_owned(), [2, 100, 6400, 0]);
        w.insert("c".to_owned(), [0, 0, 0, 7]);
        let names: Vec<_> = top(w, 2).into_iter().map(|r| r.0).collect();
        assert_eq!(names, ["b", "a"]);
    }
}
