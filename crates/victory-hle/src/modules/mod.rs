//! Implementations, one module per area of the console API.
//!
//! Adding an API: write an `extern "system"` function named exactly like
//! the import, then list it with `hle_export!` in the module's `module()`.
//! Run `victory census` over your titles to see what is still missing.
//!
//! Planned areas (driven by census results, not by completeness):
//! `thread` (TLS callbacks for title threads), `xaudio2`.

#[cfg(windows)]
pub mod apu;
pub mod audio;
pub mod d3d12x;
pub mod display;
pub mod dsp;
#[cfg(windows)]
pub mod dstorage;
#[cfg(windows)]
pub mod fidelityfx;
#[cfg(windows)]
pub mod files;
pub mod gameinput;
pub mod gameinput_script;
pub mod kernel;
#[cfg(windows)]
pub mod lifecycle;
#[cfg(windows)]
pub mod media;
#[cfg(windows)]
pub mod net;
pub mod read_stats;
pub mod rpcrt4;
pub mod speech;
pub mod storage;
pub mod xasync;
pub mod xg;
pub mod xgame;
pub mod xgameruntime;
pub mod xgamesave;
pub mod xgameui;
#[cfg(windows)]
pub mod xmem;
pub mod xnetworking;
pub mod xpackage;
pub mod xstore;
pub mod xsystem;
pub mod xuser;

use std::sync::{Mutex, MutexGuard};

use crate::HleModule;

/// Locks `mutex`, taking the data even if a thread panicked holding it.
pub(crate) fn lock<T: ?Sized>(mutex: &Mutex<T>) -> MutexGuard<'_, T> {
    mutex.lock().unwrap_or_else(|e| e.into_inner())
}

pub(crate) fn all() -> Vec<HleModule> {
    let mut all = vec![
        kernel::module(),
        xgameruntime::module(),
        gameinput::module(),
        d3d12x::module(),
        audio::module(),
        dsp::module(),
        xg::module(),
    ];
    #[cfg(windows)]
    all.push(apu::module());
    #[cfg(windows)]
    all.push(audio::silent::module());
    #[cfg(windows)]
    all.push(media::module());
    #[cfg(windows)]
    all.push(dstorage::module());
    #[cfg(windows)]
    all.push(files::module());
    #[cfg(windows)]
    all.push(lifecycle::module());
    #[cfg(windows)]
    all.push(net::module());
    all.push(rpcrt4::module());
    #[cfg(windows)]
    all.push(xmem::module());
    #[cfg(windows)]
    all.push(xmem::virtual_free_module());
    #[cfg(windows)]
    all.push(xsystem::module());
    #[cfg(windows)]
    if fidelityfx::serving() {
        all.push(fidelityfx::module());
    }
    all
}
