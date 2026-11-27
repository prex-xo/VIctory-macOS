//! A sampled texture's refresh under Rosetta, where no memory is
//! tracked (VIctory-macOS issue #1): the refresh reads and compares the
//! whole memory — the compare is the change detector — then detiles.
//! The compare is timed alone, both ways: every chunk walked byte by
//! byte, as it was before the slice fast path (`walk_changed`), and
//! with the fast path (`changed_runs`, the compare production uses),
//! which answers an equal chunk with one slice comparison and walks
//! only the chunks that differ. Almost every chunk of the hot path is
//! equal: the title makes about 80 sampled-texture copies a frame,
//! each touching a few KiB, and every other chunk of every sampled
//! texture is unchanged. The whole refresh — read, compare, and detile
//! — is timed with the detile of every subresource and with the span
//! detile, only the subresources the changed bytes touch; the detile is
//! timed alone too. The read is a memcpy of resident memory standing in
//! for the kernel's read of the title's memory, which costs more under
//! Wine than this copy, so the whole-refresh numbers understate the
//! read and overstate the compare's share of it. The compare is the
//! production function (`victory_formats::texture::changed_runs`), so
//! the bench and the refresh it measures cannot drift. Streamed
//! textures change mip by mip, so each submission changes one 64 KiB
//! page, cycling across the subresources. Run with
//! `cargo bench -p victory-formats --bench tex_refresh`.

use std::hint::black_box;
use std::time::{Duration, Instant};

use victory_formats::texture::{Desc, Dimension, Layout, Swizzle, changed_runs, walk_changed};

/// One page a submission changes.
const PAGE: usize = 64 * 1024;
/// Two rounds over a 13-mip texture's subresources; the median is over all.
const SUBMISSIONS: usize = 26;
/// The size the untracked refresh reads and compares at a time, as
/// victory-gpu's texture.rs (`COMPARE_CHUNK`).
const COMPARE_CHUNK: usize = 1 << 20;

/// The bytes of the texture's memory subresource `i` lies in, as the GPU's
/// staging computes them from the layout (rounded out to its blocks).
fn span(layout: &Layout, i: u32, mips: u32) -> (usize, usize) {
    let d = &layout.desc;
    let block = layout.alignment.max(256);
    let (mip, slice) = (i % mips, i / mips);
    let m = &layout.mips[mip as usize];
    let (z0, z1) = if d.dimension == Dimension::Tex3d {
        (0, m.depth.max(1) - 1)
    } else {
        (slice, slice)
    };
    let first = layout.offset(mip, 0, 0, z0).unwrap();
    let last = layout
        .offset(mip, m.width.max(1) - 1, m.height.max(1) - 1, z1)
        .unwrap()
        + u64::from(d.element_bytes);
    let (from, to) = (first / block * block, last.div_ceil(block) * block);
    (from as usize, to.min(layout.size) as usize)
}

/// Detiles `subs` of `tiled` into `linear` rows, as a refresh does.
fn detile(layout: &Layout, linear: &mut [Vec<u8>], tiled: &[u8], subs: &[u32], mips: u32) {
    for &i in subs {
        let (mip, slice) = (i % mips, i / mips);
        let m = &layout.mips[mip as usize];
        let pitch = m.pitch as usize * layout.desc.element_bytes as usize;
        layout
            .detile(
                tiled,
                mip,
                slice,
                &mut linear[i as usize],
                pitch,
                pitch * m.rows as usize,
            )
            .unwrap();
    }
}

/// Writes one changed page into `tiled`, returning where it lies.
fn write_page(tiled: &mut [u8], at: usize) -> (usize, usize) {
    let len = PAGE.min(tiled.len() - at);
    for b in &mut tiled[at..at + len] {
        *b = b.wrapping_add(1);
    }
    (at, len)
}

/// The compare of one whole untracked refresh: `tiled`, the texture's
/// memory, compared with `kept`, the bytes last compared equal, a
/// `COMPARE_CHUNK` at a time — every chunk walked byte by byte when
/// `bytewalk` (the compare as it was before the slice fast path), the
/// production compare otherwise. The answer is the runs of bytes that
/// differ, and `kept` ends the memory's. The read is not part of this
/// number: the whole-refresh timings below carry it as the memcpy
/// stand-in for the kernel's read.
fn compare_all(bytewalk: bool, kept: &mut [u8], tiled: &[u8]) -> Vec<(usize, usize)> {
    let size = tiled.len();
    let mut runs: Vec<(usize, usize)> = Vec::new();
    let mut at = 0;
    while at < size {
        let n = COMPARE_CHUNK.min(size - at);
        if bytewalk {
            walk_changed(at, &mut kept[at..at + n], &tiled[at..at + n], &mut runs);
        } else {
            changed_runs(at, &mut kept[at..at + n], &tiled[at..at + n], &mut runs);
        }
        at += n;
    }
    runs
}

/// One whole untracked refresh of `tiled` into `linear` with `kept`, the
/// bytes last compared equal: the memory read a `COMPARE_CHUNK` at a
/// time — a memcpy of resident memory standing in for the kernel's read
/// of the title's memory, which costs more under Wine — and compared
/// with `kept` (`changed_runs`, the production compare), then the
/// detile of every subresource (`whole`) or of only the subresources
/// the differing runs touch.
fn refresh(
    layout: &Layout,
    linear: &mut [Vec<u8>],
    tiled: &[u8],
    kept: &mut [u8],
    scratch: &mut [u8],
    spans: &[(usize, usize)],
    whole: bool,
) {
    let mips = layout.mips.len() as u32;
    let size = tiled.len();
    let mut runs: Vec<(usize, usize)> = Vec::new();
    let mut at = 0;
    while at < size {
        let n = COMPARE_CHUNK.min(size - at);
        // The read: a memcpy stand-in for the kernel's read of the
        // title's memory, which costs more under Wine than this copy.
        scratch[..n].copy_from_slice(&tiled[at..at + n]);
        changed_runs(at, &mut kept[at..at + n], &scratch[..n], &mut runs);
        at += n;
    }
    let subs: Vec<u32> = if whole {
        (0..spans.len() as u32).collect()
    } else {
        (0..spans.len() as u32)
            .filter(|&i| {
                let (a, b) = spans[i as usize];
                runs.iter().any(|&(from, len)| from < b && a < from + len)
            })
            .collect()
    };
    detile(layout, linear, black_box(tiled), &subs, mips);
}

fn median(mut samples: Vec<Duration>) -> Duration {
    samples.sort_unstable();
    samples[samples.len() / 2]
}

fn compare_changes() {
    let changed: &[(&str, &[(usize, usize)])] = &[
        ("equal", &[]),
        ("single", &[(64 * 1024, 128 * 1024)]),
        (
            "separated",
            &[(64 * 1024, 128 * 1024), (896 * 1024, 960 * 1024)],
        ),
        ("dense", &[(0, COMPARE_CHUNK)]),
    ];
    let mut kept = vec![0; COMPARE_CHUNK];
    let mut runs = Vec::new();
    for (name, regions) in changed {
        let mut fresh = vec![0; COMPARE_CHUNK];
        for &(start, end) in *regions {
            fresh[start..end].fill(1);
        }
        let mut expected = vec![0; COMPARE_CHUNK];
        let mut expected_runs = Vec::new();
        walk_changed(37, &mut expected, &fresh, &mut expected_runs);
        let mut samples = Vec::new();
        for i in 0..(64 + 8) {
            kept.fill(0);
            runs.clear();
            let start = Instant::now();
            changed_runs(37, black_box(&mut kept), black_box(&fresh), &mut runs);
            let elapsed = start.elapsed();
            assert_eq!(runs, expected_runs);
            assert_eq!(kept, expected);
            if i >= 8 {
                samples.push(elapsed.as_nanos());
            }
        }
        samples.sort_unstable();
        println!(
            "bench_tex_compare_{name}_ns {:?} runs={}",
            samples,
            expected_runs.len()
        );
    }
}

fn main() {
    if std::env::var_os("VICTORY_BENCH_COMPARE_ONLY").is_some() {
        compare_changes();
        return;
    }
    let cases = [
        (
            "bc7",
            Desc {
                dimension: Dimension::Tex2d,
                width: 4096,
                height: 4096,
                depth: 1,
                mips: 13,
                element_bytes: 16,
                block: 4,
                samples: 1,
                swizzle: Swizzle::SW_64KB_D,
            },
        ),
        (
            "rgba8",
            Desc {
                dimension: Dimension::Tex2d,
                width: 4096,
                height: 4096,
                depth: 1,
                mips: 13,
                element_bytes: 4,
                block: 1,
                samples: 1,
                swizzle: Swizzle::SW_64KB_D,
            },
        ),
    ];
    for (name, desc) in cases {
        let layout = Layout::new(&desc).unwrap();
        let size = layout.size as usize;
        let mips = layout.mips.len() as u32;
        let count = mips * layout.slices;
        let spans: Vec<(usize, usize)> = (0..count).map(|i| span(&layout, i, mips)).collect();
        let mut linear: Vec<Vec<u8>> = (0..count)
            .map(|i| {
                let m = &layout.mips[(i % mips) as usize];
                let pitch = m.pitch as usize * desc.element_bytes as usize;
                vec![0u8; pitch * m.rows as usize]
            })
            .collect();
        let mut tiled = vec![0u8; size];
        for (i, b) in tiled.iter_mut().enumerate() {
            *b = (i % 251) as u8;
        }
        let every: Vec<u32> = (0..count).collect();
        let mut checksum = 0u64;
        // One 64 KiB page changes per submission, cycling across the
        // subresources (streamed textures change mip by mip).
        let pages: Vec<usize> = (0..SUBMISSIONS)
            .map(|i| spans[i % count as usize].0.min(size - 1))
            .collect();
        // Before the span detile: a differing byte anywhere re-detiles
        // every subresource.
        let mut full = Vec::new();
        for &at in &pages {
            let (_, len) = write_page(&mut tiled, at);
            let start = Instant::now();
            detile(&layout, &mut linear, black_box(&tiled), &every, mips);
            full.push(start.elapsed());
            checksum += u64::from(linear[len % count as usize][0]);
        }
        // The span detile: only the subresources the changed page touches.
        let mut partial = Vec::new();
        for &at in &pages {
            let (at, len) = write_page(&mut tiled, at);
            let fresh: Vec<u32> = (0..count)
                .filter(|&i| at < spans[i as usize].1 && spans[i as usize].0 < at + len)
                .collect();
            let start = Instant::now();
            detile(&layout, &mut linear, black_box(&tiled), &fresh, mips);
            partial.push(start.elapsed());
            checksum += u64::from(linear[len % count as usize][0]);
        }
        // The compare alone, every chunk walked byte by byte: the shape
        // before the slice fast path, whose cost the review asked about.
        let mut kept = tiled.clone();
        let mut bytewalk = Vec::new();
        for &at in &pages {
            write_page(&mut tiled, at);
            let start = Instant::now();
            let runs = compare_all(true, &mut kept, black_box(&tiled));
            bytewalk.push(start.elapsed());
            checksum += runs.iter().map(|&(_, len)| len as u64).sum::<u64>();
        }
        // The compare alone with the fast path: the production compare.
        let mut fast = Vec::new();
        kept.copy_from_slice(&tiled);
        for &at in &pages {
            write_page(&mut tiled, at);
            let start = Instant::now();
            let runs = compare_all(false, &mut kept, black_box(&tiled));
            fast.push(start.elapsed());
            checksum += runs.iter().map(|&(_, len)| len as u64).sum::<u64>();
        }
        // The whole refresh, read (the memcpy stand-in) + compare + detile:
        // every subresource, then only the touched ones. Each keeps its
        // own `kept`, the bytes last compared equal, as the refresh
        // keeps `last.read`.
        let mut scratch = vec![0u8; COMPARE_CHUNK];
        let mut whole_full = Vec::new();
        kept.copy_from_slice(&tiled);
        for &at in &pages {
            let (_, len) = write_page(&mut tiled, at);
            let start = Instant::now();
            refresh(
                &layout,
                &mut linear,
                black_box(&tiled),
                &mut kept,
                &mut scratch,
                &spans,
                true,
            );
            whole_full.push(start.elapsed());
            checksum += u64::from(linear[len % count as usize][0]);
        }
        let mut whole_partial = Vec::new();
        kept.copy_from_slice(&tiled);
        for &at in &pages {
            let (_, len) = write_page(&mut tiled, at);
            let start = Instant::now();
            refresh(
                &layout,
                &mut linear,
                black_box(&tiled),
                &mut kept,
                &mut scratch,
                &spans,
                false,
            );
            whole_partial.push(start.elapsed());
            checksum += u64::from(linear[len % count as usize][0]);
        }
        black_box(checksum);
        println!(
            "bench_tex_refresh_us_full_{name} {:.3}",
            median(full).as_secs_f64() * 1e6
        );
        println!(
            "bench_tex_refresh_us_partial_{name} {:.3}",
            median(partial).as_secs_f64() * 1e6
        );
        println!(
            "bench_tex_refresh_us_compare_bytewalk_{name} {:.3}",
            median(bytewalk).as_secs_f64() * 1e6
        );
        println!(
            "bench_tex_refresh_us_compare_fast_{name} {:.3}",
            median(fast).as_secs_f64() * 1e6
        );
        println!(
            "bench_tex_refresh_us_whole_full_{name} {:.3}",
            median(whole_full).as_secs_f64() * 1e6
        );
        println!(
            "bench_tex_refresh_us_whole_partial_{name} {:.3}",
            median(whole_partial).as_secs_f64() * 1e6
        );
    }
}
