//! The macOS kernel's own memory pressure, read past Wine.
//!
//! What Wine gives Windows code as free memory (`GlobalMemoryStatusEx`)
//! stays high while macOS compresses and swaps: with incompressible memory
//! filled step by step on a 24 GB Mac it never went under 17 % free while
//! macOS compressed 4-6 GB a second and swapped 2.2 GB, and with half of each
//! page compressible it said 21 % free with 22 GB held. The kernel's pressure
//! level (`kern.memorystatus_vm_pressure_level`) went to its warning level
//! as the heavy compression started, before the swapping. Under Rosetta the
//! x86-64 BSD syscall reaches the kernel directly, as the Mach traps of
//! `crate::alias` do.

use std::sync::OnceLock;

/// What the kernel says of the computer's memory.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum Pressure {
    Normal,
    /// The kernel warns: it compresses heavily and swapping is near.
    Warn,
    /// The kernel is short: it swaps and may end processes.
    Critical,
}

impl Pressure {
    /// From the value of `kern.memorystatus_vm_pressure_level`
    /// (`DISPATCH_MEMORYPRESSURE_*`: 1 normal, 2 warn, 4 critical).
    pub(crate) fn from_level(level: i32) -> Option<Pressure> {
        match level {
            1 => Some(Pressure::Normal),
            2 => Some(Pressure::Warn),
            4 => Some(Pressure::Critical),
            _ => None,
        }
    }
}

/// Whether [`now`] asks the kernel: under Rosetta (Wine on macOS) unless
/// `VICTORY_MAC_PRESSURE=0`, and only once a first read worked.
fn enabled() -> bool {
    static ON: OnceLock<bool> = OnceLock::new();
    *ON.get_or_init(|| {
        if std::env::var_os("VICTORY_MAC_PRESSURE").is_some_and(|v| v == "0")
            || !crate::watch::rosetta()
        {
            return false;
        }
        let first = read();
        match first {
            Some(p) => tracing::info!(pressure = ?p, "reading the macOS kernel's memory pressure"),
            None => tracing::info!("the macOS kernel's memory pressure could not be read"),
        }
        first.is_some()
    })
}

/// The kernel's memory pressure now, or None off macOS or where it cannot
/// be read.
pub(crate) fn now() -> Option<Pressure> {
    if !enabled() {
        return None;
    }
    read()
}

fn read() -> Option<Pressure> {
    let mut value = [0u8; 4];
    let len = sysctl_by_name(b"kern.memorystatus_vm_pressure_level", &mut value)?;
    (len == 4)
        .then(|| Pressure::from_level(i32::from_le_bytes(value)))
        .flatten()
}

/// `sysctlbyname` (BSD syscall 274) for `name`, into `out`: the length
/// written, or None if the kernel refused. Only under Rosetta, where the
/// x86-64 XNU syscall ABI is what `syscall` reaches.
#[cfg(target_arch = "x86_64")]
fn sysctl_by_name(name: &[u8], out: &mut [u8]) -> Option<usize> {
    use core::arch::asm;
    const BSD: u64 = 0x0200_0000;
    const SYSCTLBYNAME: u64 = 274;
    let mut len = out.len();
    let ret: u64;
    let failed: u8;
    // SAFETY: XNU's x86-64 BSD syscall ABI under Rosetta 2: `name` is
    // `name.len()` bytes (no NUL needed, the length is passed), `out` is
    // `len` bytes and `len` is written back; nothing is set (null, 0). The
    // kernel returns a second value in rdx and flags an error with carry.
    unsafe {
        asm!(
            "syscall",
            "setc {failed}",
            failed = out(reg_byte) failed,
            inlateout("rax") BSD | SYSCTLBYNAME => ret,
            in("rdi") name.as_ptr(),
            in("rsi") name.len(),
            inlateout("rdx") out.as_mut_ptr() => _,
            in("r10") &raw mut len,
            in("r8") 0usize,
            in("r9") 0usize,
            lateout("rcx") _,
            lateout("r11") _,
        );
    }
    (failed == 0 && ret == 0).then_some(len)
}

#[cfg(not(target_arch = "x86_64"))]
fn sysctl_by_name(_: &[u8], _: &mut [u8]) -> Option<usize> {
    None
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The kernel's three levels, and nothing else.
    #[test]
    fn pressure_levels_are_the_kernels_three() {
        assert_eq!(Pressure::from_level(1), Some(Pressure::Normal));
        assert_eq!(Pressure::from_level(2), Some(Pressure::Warn));
        assert_eq!(Pressure::from_level(4), Some(Pressure::Critical));
        for other in [0, 3, 5, 8, -1] {
            assert_eq!(Pressure::from_level(other), None, "{other}");
        }
    }

    /// Under Wine on macOS the kernel answers with one of its levels.
    #[test]
    fn under_rosetta_the_kernel_answers() {
        if !crate::watch::rosetta() {
            return;
        }
        assert!(read().is_some());
    }
}
