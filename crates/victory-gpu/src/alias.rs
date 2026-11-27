//! Zero-copy title buffers on Apple Silicon through D3DMetal (`mach_vm_remap`
//! aliasing, issue #204).
//!
//! D3DMetal refuses `OpenExistingHeapFromAddress*`, so VIctory normally keeps
//! host copies of title buffers (`crate::shadow`) and copies pages back and
//! forth. On Apple Silicon (unified memory), D3DMetal's own mapped write-back
//! L0 buffers are backed by the same physical RAM. Under Wine + Rosetta 2, raw
//! `mach_msg2_trap` MIG calls (`mach_vm_remap` with `copy = FALSE`,
//! `VM_FLAGS_FIXED | VM_FLAGS_OVERWRITE`) point the title's `VirtualAlloc`
//! virtual address range directly at the D3DMetal buffer's physical pages.
//!
//! On by default on Rosetta + D3DMetal (never WARP); `VICTORY_ALIAS_BUFFERS=0`
//! keeps the host copies. The probe at start-up ([`supported`]) still falls
//! back to the copies where the remap does not work.

use std::sync::{Mutex, OnceLock};

use windows_sys::Win32::System::Memory::{
    VirtualAlloc, VirtualFree, MEM_COMMIT, MEM_RELEASE, MEM_RESERVE, PAGE_READWRITE,
};

use crate::backend::HostMemory;
use crate::host::host;
use crate::watch::PAGE;

#[cfg(any(test, feature = "diagnostics"))]
use windows::Win32::Graphics::Direct3D12::*;

#[cfg(target_arch = "x86_64")]
mod mach {
    use core::arch::asm;
    use std::sync::{Mutex, OnceLock};

    const MACH: u64 = 0x0100_0000;
    const TRAP_REPLY_PORT: u64 = 26;
    const TRAP_TASK_SELF: u64 = 28;
    /// `mach_msg2_trap` (8 arguments). `mach_msg_trap` (31) is fatal on
    /// macOS 27+ for kernel-object calls (`EXC_GUARD`).
    const TRAP_MSG2: u64 = 47;
    const MACH64_SEND_KOBJECT_CALL: u64 = 0x0000_0002_0000_0000;

    pub const VM_FLAGS_FIXED: i32 = 0;
    pub const VM_FLAGS_OVERWRITE: i32 = 0x4000;

    unsafe fn trap(n: u64, a: [u64; 6]) -> u64 {
        // SAFETY: forwarded to `trap8`.
        unsafe { trap8(n, a, 0, 0) }
    }

    unsafe fn trap8(n: u64, a: [u64; 6], a7: u64, a8: u64) -> u64 {
        let ret: u64;
        // SAFETY: x86-64 XNU Mach trap ABI under Rosetta 2; 7th and 8th args
        // sit at `[rsp + 8]` and `[rsp + 16]`.
        unsafe {
            asm!(
                "sub rsp, 32",
                "mov qword ptr [rsp], 0",
                "mov qword ptr [rsp + 8], {a7}",
                "mov qword ptr [rsp + 16], {a8}",
                "syscall",
                "add rsp, 32",
                inlateout("rax") MACH | n => ret,
                in("rdi") a[0],
                in("rsi") a[1],
                in("rdx") a[2],
                in("r10") a[3],
                in("r8") a[4],
                in("r9") a[5],
                a7 = in(reg) a7,
                a8 = in(reg) a8,
                lateout("rcx") _,
                lateout("r11") _,
            );
        }
        ret
    }

    pub fn task_self() -> u32 {
        // SAFETY: argument-less Mach trap.
        unsafe { trap(TRAP_TASK_SELF, [0; 6]) as u32 }
    }

    fn reply_port() -> u32 {
        static PORT: OnceLock<u32> = OnceLock::new();
        *PORT.get_or_init(|| {
            // SAFETY: argument-less Mach trap.
            unsafe { trap(TRAP_REPLY_PORT, [0; 6]) as u32 }
        })
    }

    #[repr(C, packed(4))]
    #[derive(Default, Clone, Copy)]
    struct Header {
        bits: u32,
        size: u32,
        remote: u32,
        local: u32,
        voucher: u32,
        id: i32,
    }

    /// `mach_vm_remap` MIG request (`msgh_id = 4813`, `#pragma pack(4)`).
    #[repr(C, packed(4))]
    #[derive(Default, Clone, Copy)]
    struct RemapRequest {
        head: Header,
        descriptors: u32,
        src_task: u32,
        pad1: u32,
        disp_type: u32,
        ndr: [u8; 8],
        target_address: u64,
        size: u64,
        mask: u64,
        flags: i32,
        src_address: u64,
        copy: u32,
        inheritance: u32,
    }

    #[repr(C, packed(4))]
    #[derive(Clone, Copy)]
    struct RemapReply {
        head: Header,
        ndr: [u8; 8],
        ret: i32,
        target_address: u64,
        cur_protection: i32,
        max_protection: i32,
        trailer: [u8; 128],
    }

    #[derive(Debug)]
    pub struct Remapped {
        pub msg: u32,
        pub kr: i32,
        pub address: u64,
    }

    static REMAP_LOCK: Mutex<()> = Mutex::new(());

    /// Maps `[src, src + size)` of this task at `target`, sharing physical
    /// pages (`copy = FALSE`).
    ///
    /// # Safety
    /// Replaces whatever is mapped at `target`; both ranges must be page
    /// aligned and owned by the caller.
    pub unsafe fn remap_shared(target: u64, src: u64, size: u64, flags: i32) -> Remapped {
        let _guard = REMAP_LOCK.lock().unwrap_or_else(|e| e.into_inner());
        let task = task_self();
        let reply = reply_port();
        let mut msg = [0u8; size_of::<RemapReply>()];
        let req = RemapRequest {
            head: Header {
                // MACH_MSGH_BITS(COPY_SEND=19, MAKE_SEND_ONCE=21) | MACH_MSGH_BITS_COMPLEX
                bits: 19 | (21 << 8) | 0x8000_0000,
                size: size_of::<RemapRequest>() as u32,
                remote: task,
                local: reply,
                voucher: 0,
                id: 4813,
            },
            descriptors: 1,
            src_task: task,
            pad1: 0,
            disp_type: 19 << 16,
            ndr: [0, 0, 0, 0, 1, 0, 0, 0],
            target_address: target,
            size,
            mask: 0,
            flags,
            src_address: src,
            copy: 0,
            inheritance: 1,
        };
        // SAFETY: `req` and `msg` are plain-old-data structs of known size.
        unsafe {
            std::ptr::copy_nonoverlapping(
                (&raw const req).cast::<u8>(),
                msg.as_mut_ptr(),
                size_of::<RemapRequest>(),
            );
        }
        let pack = |lo: u32, hi: u32| (u64::from(hi) << 32) | u64::from(lo);
        let args = [
            msg.as_mut_ptr() as u64,
            3 | MACH64_SEND_KOBJECT_CALL,
            pack(req.head.bits, size_of::<RemapRequest>() as u32),
            pack(task, reply),
            pack(0, 4813),
            pack(1, reply),
        ];
        // SAFETY: `msg` holds the request and space for the reply and trailer.
        let r = unsafe { trap8(TRAP_MSG2, args, pack(msg.len() as u32, 0), 0) } as u32;
        // SAFETY: the kernel wrote at most `msg.len()` bytes into `msg`.
        let rep: RemapReply = unsafe { std::ptr::read_unaligned(msg.as_ptr().cast()) };
        Remapped {
            msg: r,
            kr: rep.ret,
            address: rep.target_address,
        }
    }
}

#[cfg(not(target_arch = "x86_64"))]
mod mach {
    pub const VM_FLAGS_FIXED: i32 = 0;
    pub const VM_FLAGS_OVERWRITE: i32 = 0x4000;

    #[derive(Debug)]
    pub struct Remapped {
        pub msg: u32,
        pub kr: i32,
        pub address: u64,
    }

    pub fn task_self() -> u32 {
        0
    }

    pub unsafe fn remap_shared(_target: u64, _src: u64, _size: u64, _flags: i32) -> Remapped {
        Remapped {
            msg: 1,
            kr: -1,
            address: 0,
        }
    }
}

/// Minimum physical memory (20 GiB) required for aliasing to be enabled by default.
/// Below this (e.g. 8/16/18 GB Macs), pinned/wired aliased regions can trigger
/// Metal out-of-memory errors; `VICTORY_ALIAS_BUFFERS=1` still opts in.
const DEFAULT_MIN_RAM_BYTES: u64 = 20 << 30;

/// The computer's memory, as Wine reports it (`ullTotalPhys`).
pub(crate) fn total_ram_bytes() -> Option<u64> {
    use windows_sys::Win32::System::SystemInformation::{GlobalMemoryStatusEx, MEMORYSTATUSEX};
    // SAFETY: `MEMORYSTATUSEX` is plain-old-data with `dwLength` initialized.
    let mut status: MEMORYSTATUSEX = unsafe { std::mem::zeroed() };
    status.dwLength = size_of::<MEMORYSTATUSEX>() as u32;
    // SAFETY: `status` is valid and `dwLength` is set.
    if unsafe { GlobalMemoryStatusEx(&mut status) } == 0 || status.ullTotalPhys == 0 {
        None
    } else {
        Some(status.ullTotalPhys)
    }
}

/// `asked`: `Some(on)` when `VICTORY_ALIAS_BUFFERS` is set (`0` is off, any
/// other value on), `None` when it is not: then on only with >= 20 GiB RAM.
fn decide_enabled(asked: Option<bool>, rosetta: bool, warp: bool, total_ram: Option<u64>) -> bool {
    let default_on = total_ram.is_some_and(|ram| ram >= DEFAULT_MIN_RAM_BYTES);
    asked.unwrap_or(default_on) && rosetta && !warp
}

/// Whether title buffers are aliased on this host: Rosetta + D3DMetal,
/// on by default with >= 20 GiB RAM unless `VICTORY_ALIAS_BUFFERS` overrides it.
pub fn enabled() -> bool {
    static ON: OnceLock<bool> = OnceLock::new();
    *ON.get_or_init(|| {
        let rosetta = crate::watch::rosetta();
        let warp = cfg!(feature = "diagnostics") && crate::switch::on("VICTORY_D3D12_WARP");
        let asked = std::env::var_os("VICTORY_ALIAS_BUFFERS")
            .map(|_| crate::switch::on("VICTORY_ALIAS_BUFFERS"));
        let ram = total_ram_bytes();
        let on = decide_enabled(asked, rosetta, warp, ram);
        if asked.is_none() && rosetta && !warp && !on {
            tracing::info!(
                ram_gib = ram.map(|b| b / (1 << 30)),
                "zero-copy title buffer aliasing off by default below 20 GiB RAM (set VICTORY_ALIAS_BUFFERS=1 to enable)"
            );
        }
        on
    })
}

/// Probes once whether `mach_vm_remap` onto a D3DMetal write-back L0 buffer
/// works and whether fresh buffers arrive zero-initialized.
fn supported() -> Option<bool> {
    static PROBE: OnceLock<Option<bool>> = OnceLock::new();
    *PROBE.get_or_init(|| {
        let warp = cfg!(feature = "diagnostics") && crate::switch::on("VICTORY_D3D12_WARP");
        if !crate::watch::rosetta() || warp || mach::task_self() == 0 {
            return None;
        }
        let h = host()?;
        let (buffer, gpu_mem) = h.shared_buffer(PAGE as u64)?;
        // SAFETY: `gpu_mem` is our `PAGE`-byte mapped buffer.
        let zeroed = unsafe {
            gpu_mem.cast::<u64>().read_volatile() == 0
                && gpu_mem.add(PAGE - 8).cast::<u64>().read_volatile() == 0
        };
        // SAFETY: standalone `PAGE`-byte allocation freed below.
        let title = unsafe {
            VirtualAlloc(
                std::ptr::null(),
                PAGE,
                MEM_RESERVE | MEM_COMMIT,
                PAGE_READWRITE,
            )
        }
        .cast::<u8>();
        if title.is_null() {
            return None;
        }
        // SAFETY: `title` and `gpu_mem` are both `PAGE`-aligned and `PAGE` bytes.
        let r = unsafe {
            mach::remap_shared(
                title as u64,
                gpu_mem as u64,
                PAGE as u64,
                mach::VM_FLAGS_FIXED | mach::VM_FLAGS_OVERWRITE,
            )
        };
        let ok = if r.msg == 0 && r.kr == 0 && r.address == title as u64 {
            // SAFETY: both pointers point to the same `PAGE` of RAM.
            unsafe {
                title.add(128).write_volatile(0x5a);
                let seen_gpu = gpu_mem.add(128).read_volatile() == 0x5a;
                gpu_mem.add(256).write_volatile(0xa5);
                let seen_title = title.add(256).read_volatile() == 0xa5;
                seen_gpu && seen_title
            }
        } else {
            false
        };
        // SAFETY: `title` was allocated with `VirtualAlloc`.
        unsafe { VirtualFree(title.cast(), 0, MEM_RELEASE) };
        drop(buffer);
        if ok {
            tracing::info!(
                zeroed,
                "zero-copy title buffer aliasing active (mach_vm_remap)"
            );
            Some(zeroed)
        } else {
            tracing::warn!(
                ?r,
                "mach_vm_remap aliasing probe failed; falling back to shadow copies"
            );
            None
        }
    })
}

/// CPU-visible backing for XMem physical pages. The placed buffer keeps a
/// CPU mapping of the heap from which reserved resources can take tiles.
#[cfg(any(test, feature = "diagnostics"))]
pub struct PhysicalHeap {
    backing: ID3D12Resource,
    // A placed buffer must keep its heap alive.
    pub(crate) _heap: ID3D12Heap,
    address: usize,
    len: usize,
}

#[cfg(any(test, feature = "diagnostics"))]
impl PhysicalHeap {
    pub(crate) fn new(device: &ID3D12Device, len: usize) -> windows::core::Result<Self> {
        let desc = D3D12_HEAP_DESC {
            SizeInBytes: len as u64,
            Properties: D3D12_HEAP_PROPERTIES {
                Type: D3D12_HEAP_TYPE_CUSTOM,
                CPUPageProperty: D3D12_CPU_PAGE_PROPERTY_WRITE_BACK,
                MemoryPoolPreference: D3D12_MEMORY_POOL_L0,
                CreationNodeMask: 1,
                VisibleNodeMask: 1,
            },
            Flags: D3D12_HEAP_FLAG_ALLOW_ONLY_BUFFERS,
            ..Default::default()
        };
        let buffer = D3D12_RESOURCE_DESC {
            Dimension: D3D12_RESOURCE_DIMENSION_BUFFER,
            Width: len as u64,
            Height: 1,
            DepthOrArraySize: 1,
            MipLevels: 1,
            SampleDesc: windows::Win32::Graphics::Dxgi::Common::DXGI_SAMPLE_DESC {
                Count: 1,
                Quality: 0,
            },
            Layout: D3D12_TEXTURE_LAYOUT_ROW_MAJOR,
            ..Default::default()
        };
        let mut heap = None;
        let mut backing = None;
        let mut address = std::ptr::null_mut();
        // SAFETY: plain D3D12 creation with output slots on this stack.
        let (heap, backing) = unsafe {
            device.CreateHeap(&desc, &mut heap)?;
            let heap: ID3D12Heap = heap.ok_or_else(windows::core::Error::empty)?;
            device.CreatePlacedResource(
                &heap,
                0,
                &buffer,
                D3D12_RESOURCE_STATE_COMMON,
                None,
                &mut backing,
            )?;
            let backing: ID3D12Resource = backing.ok_or_else(windows::core::Error::empty)?;
            backing.Map(0, None, Some(&mut address))?;
            (heap, backing)
        };
        if address.is_null() {
            return Err(windows::core::Error::empty());
        }
        Ok(Self {
            backing,
            _heap: heap,
            address: address as usize,
            len,
        })
    }

    /// The CPU view, valid for the lifetime of this heap.
    pub fn cpu_address(&self) -> usize {
        self.address
    }

    /// Replaces a Windows file view with the corresponding heap pages.
    ///
    /// # Safety
    /// The caller owns the destination view and keeps this heap alive as
    /// long as the view is mapped.
    pub unsafe fn alias_view(&self, at: usize, offset: usize, len: usize) -> bool {
        if at == 0
            || len == 0
            || [at, offset, len].iter().any(|n| !n.is_multiple_of(PAGE))
            || offset.checked_add(len).is_none_or(|end| end > self.len)
        {
            return false;
        }
        // SAFETY: the caller owns the view; the source range is in the
        // live placed buffer and checked above.
        let remap = unsafe {
            mach::remap_shared(
                at as u64,
                (self.address + offset) as u64,
                len as u64,
                mach::VM_FLAGS_FIXED | mach::VM_FLAGS_OVERWRITE,
            )
        };
        remap.msg == 0 && remap.kr == 0 && remap.address == at as u64
    }
}

#[cfg(any(test, feature = "diagnostics"))]
impl Drop for PhysicalHeap {
    fn drop(&mut self) {
        // SAFETY: unmapped once, while the placed buffer still owns the view.
        unsafe { self.backing.Unmap(0, None) };
    }
}

/// Allocates CPU-visible physical tiles for the Rosetta diagnostics trial.
#[cfg(feature = "diagnostics")]
pub fn physical_heap(len: usize) -> Option<PhysicalHeap> {
    static ENABLED: OnceLock<bool> = OnceLock::new();
    if !*ENABLED.get_or_init(|| crate::switch::on("VICTORY_PHYSICAL_TILES") && supported().is_some())
        || len == 0
        || !len.is_multiple_of(PAGE)
    {
        return None;
    }
    let host = host()?;
    match PhysicalHeap::new(&host.device, len) {
        Ok(heap) => Some(heap),
        Err(error) => {
            tracing::warn!(len, %error, "physical tile heap unavailable; using a file section");
            None
        }
    }
}

/// Aliases `[base, base + len)` onto a new write-back L0 D3D12 buffer so CPU
/// and GPU share physical pages with zero copy.
pub fn alias_region(base: usize, len: usize, preserve_contents: bool) -> bool {
    static LOCK: Mutex<()> = Mutex::new(());
    let Some(zeroed) = supported() else {
        return false;
    };
    if base == 0 || len < PAGE || !base.is_multiple_of(PAGE) || !len.is_multiple_of(PAGE) {
        return false;
    }
    let Some(end) = base.checked_add(len) else {
        return false;
    };
    let _guard = LOCK.lock().unwrap_or_else(|e| e.into_inner());
    if crate::shadow::has_aliased_overlap(base as u64, end as u64) {
        return false;
    }
    let Some(h) = host() else {
        return false;
    };
    let Some((buffer, gpu_mem)) = h.shared_buffer(len as u64) else {
        return false;
    };
    // SAFETY: `base` and `gpu_mem` are `len`-byte owned mappings.
    unsafe {
        if preserve_contents {
            std::ptr::copy_nonoverlapping(base as *const u8, gpu_mem, len);
        } else if !zeroed {
            std::ptr::write_bytes(gpu_mem, 0, len);
        }
        let r = mach::remap_shared(
            base as u64,
            gpu_mem as u64,
            len as u64,
            mach::VM_FLAGS_FIXED | mach::VM_FLAGS_OVERWRITE,
        );
        if r.msg != 0 || r.kr != 0 || r.address != base as u64 {
            tracing::warn!(
                base = format_args!("{base:#x}"),
                len,
                ?r,
                "mach_vm_remap failed; keeping plain memory"
            );
            return false;
        }
    }
    crate::host::name(&buffer, || {
        format!("aliased title buffer {base:#x}+{len:#x}")
    });
    // SAFETY: `buffer` is alive.
    let gpu = unsafe { buffer.GetGPUVirtualAddress() };
    let shadow = crate::shadow::Shadow::new_aliased(buffer, gpu_mem, gpu, base as u64, len as u64);
    crate::shadow::register_aliased(shadow);
    if len >= 16 << 20 {
        tracing::info!(
            base = format_args!("{base:#x}"),
            len,
            "large title buffer aliased zero-copy"
        );
    } else {
        tracing::debug!(
            base = format_args!("{base:#x}"),
            len,
            "title buffer aliased zero-copy"
        );
    }
    true
}

/// Removes an aliased region registered at `base`.
pub fn unalias_region(base: usize) -> bool {
    crate::shadow::unregister_aliased(base as u64)
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};
    use windows::Win32::Graphics::Dxgi::Common::DXGI_SAMPLE_DESC;
    use windows::Win32::System::Threading::{CreateEventW, WaitForSingleObject};
    use windows_sys::Win32::System::Diagnostics::Debug::{
        AddVectoredExceptionHandler, RemoveVectoredExceptionHandler, EXCEPTION_POINTERS,
    };
    use windows_sys::Win32::System::Memory::{
        VirtualProtect, VirtualQuery, MEMORY_BASIC_INFORMATION, PAGE_READONLY,
    };

    struct AliasedAlloc {
        ptr: *mut u8,
        len: usize,
        aliased: bool,
    }

    impl AliasedAlloc {
        fn new(len: usize, init: Option<&dyn Fn(*mut u8)>, preserve: bool) -> Self {
            // SAFETY: allocating `len` bytes of RW memory for the test.
            let ptr = unsafe {
                VirtualAlloc(
                    std::ptr::null(),
                    len,
                    MEM_RESERVE | MEM_COMMIT,
                    PAGE_READWRITE,
                )
            }
            .cast::<u8>();
            assert!(!ptr.is_null());
            if let Some(f) = init {
                f(ptr);
            }
            let aliased = alias_region(ptr as usize, len, preserve);
            assert!(aliased, "alias_region failed");
            Self { ptr, len, aliased }
        }
    }

    impl Drop for AliasedAlloc {
        fn drop(&mut self) {
            if self.aliased {
                unalias_region(self.ptr as usize);
            }
            // As the runtime does when the title unmaps memory: the guards
            // let go of its pages. A tracker some test or copy still holds
            // left them in the guard set otherwise, and a test running at
            // the same time that was given the address had them skipped as
            // guarded already, so they stayed writable (#31:
            // `takes_guard_one_view_per_call` after
            // `aliased_gpu_writes_notify_overlapping_texture_tracker`).
            crate::watch::title_memory_remapped(self.ptr as usize, self.len);
            // SAFETY: `self.ptr` was allocated by `VirtualAlloc`.
            unsafe { VirtualFree(self.ptr.cast(), 0, MEM_RELEASE) };
        }
    }

    fn buffer_desc(size: u64) -> D3D12_RESOURCE_DESC {
        D3D12_RESOURCE_DESC {
            Dimension: D3D12_RESOURCE_DIMENSION_BUFFER,
            Width: size.max(1),
            Height: 1,
            DepthOrArraySize: 1,
            MipLevels: 1,
            SampleDesc: DXGI_SAMPLE_DESC {
                Count: 1,
                Quality: 0,
            },
            Layout: D3D12_TEXTURE_LAYOUT_ROW_MAJOR,
            Flags: D3D12_RESOURCE_FLAG_NONE,
            ..Default::default()
        }
    }

    fn committed(
        device: &ID3D12Device,
        heap_type: D3D12_HEAP_TYPE,
        size: u64,
        state: D3D12_RESOURCE_STATES,
    ) -> ID3D12Resource {
        let props = D3D12_HEAP_PROPERTIES {
            Type: heap_type,
            ..Default::default()
        };
        let mut res = None;
        // SAFETY: valid committed buffer description.
        unsafe {
            device
                .CreateCommittedResource(
                    &props,
                    D3D12_HEAP_FLAG_NONE,
                    &buffer_desc(size),
                    state,
                    None,
                    &mut res,
                )
                .unwrap();
        }
        res.unwrap()
    }

    fn run_copy(device: &ID3D12Device, f: impl FnOnce(&ID3D12GraphicsCommandList)) {
        // SAFETY: standalone queue/list/fence for test verification.
        unsafe {
            let queue: ID3D12CommandQueue = device
                .CreateCommandQueue(&D3D12_COMMAND_QUEUE_DESC {
                    Type: D3D12_COMMAND_LIST_TYPE_DIRECT,
                    ..Default::default()
                })
                .unwrap();
            let alloc: ID3D12CommandAllocator = device
                .CreateCommandAllocator(D3D12_COMMAND_LIST_TYPE_DIRECT)
                .unwrap();
            let list: ID3D12GraphicsCommandList = device
                .CreateCommandList(0, D3D12_COMMAND_LIST_TYPE_DIRECT, &alloc, None)
                .unwrap();
            f(&list);
            list.Close().unwrap();
            let cmd: ID3D12CommandList = windows::core::Interface::cast(&list).unwrap();
            queue.ExecuteCommandLists(&[Some(cmd)]);
            let fence: ID3D12Fence = device.CreateFence(0, D3D12_FENCE_FLAG_NONE).unwrap();
            let event = CreateEventW(None, false, false, None).unwrap();
            queue.Signal(&fence, 1).unwrap();
            if fence.GetCompletedValue() < 1 {
                fence.SetEventOnCompletion(1, event).unwrap();
                WaitForSingleObject(event, 5000);
            }
            let _ = windows::Win32::Foundation::CloseHandle(event);
        }
    }

    fn gpu_read(device: &ID3D12Device, src: &ID3D12Resource, offset: u64, len: usize) -> Vec<u8> {
        let back = committed(
            device,
            D3D12_HEAP_TYPE_READBACK,
            len as u64,
            D3D12_RESOURCE_STATE_COPY_DEST,
        );
        // SAFETY: `back` and `src` are valid buffers of sufficient size.
        run_copy(device, |l| unsafe {
            l.CopyBufferRegion(&back, 0, src, offset, len as u64)
        });
        let mut out = vec![0u8; len];
        // SAFETY: `back` is a readback buffer of `len` bytes.
        unsafe {
            let mut ptr = std::ptr::null_mut();
            back.Map(0, None, Some(&mut ptr)).unwrap();
            std::ptr::copy_nonoverlapping(ptr.cast::<u8>(), out.as_mut_ptr(), len);
            back.Unmap(0, None);
        }
        out
    }

    fn gpu_write(device: &ID3D12Device, dst: &ID3D12Resource, offset: u64, bytes: &[u8]) {
        let up = committed(
            device,
            D3D12_HEAP_TYPE_UPLOAD,
            bytes.len() as u64,
            D3D12_RESOURCE_STATE_GENERIC_READ,
        );
        // SAFETY: `up` is an upload buffer of `bytes.len()` bytes.
        unsafe {
            let mut ptr = std::ptr::null_mut();
            up.Map(0, None, Some(&mut ptr)).unwrap();
            std::ptr::copy_nonoverlapping(bytes.as_ptr(), ptr.cast::<u8>(), bytes.len());
            up.Unmap(0, None);
        }
        // SAFETY: `dst` and `up` are valid buffers of sufficient size.
        run_copy(device, |l| unsafe {
            l.CopyBufferRegion(dst, offset, &up, 0, bytes.len() as u64)
        });
    }

    #[test]
    fn switch_respects_rosetta_and_warp() {
        assert!(!decide_enabled(Some(false), true, false, Some(32 << 30)));
        assert!(!decide_enabled(Some(true), false, false, Some(32 << 30)));
        assert!(!decide_enabled(Some(true), true, true, Some(32 << 30)));
        assert!(decide_enabled(Some(true), true, false, Some(32 << 30)));
        assert!(decide_enabled(Some(true), true, false, Some(16 << 30)));
    }

    #[test]
    fn unset_is_on_under_rosetta_only() {
        assert!(decide_enabled(None, true, false, Some(24 << 30)));
        assert!(!decide_enabled(None, false, false, Some(24 << 30)));
        assert!(!decide_enabled(None, true, true, Some(24 << 30)));
    }

    #[test]
    fn unset_requires_at_least_20_gib_ram() {
        assert!(!decide_enabled(None, true, false, None));
        assert!(!decide_enabled(None, true, false, Some(8 << 30)));
        assert!(!decide_enabled(None, true, false, Some(16 << 30)));
        assert!(!decide_enabled(None, true, false, Some(18 << 30)));
        assert!(decide_enabled(None, true, false, Some(20 << 30)));
        assert!(decide_enabled(None, true, false, Some(32 << 30)));
        assert!(total_ram_bytes().is_some());
    }

    #[test]
    fn aliased_buffer_is_coherent_with_gpu_copies() {
        if !crate::watch::rosetta() {
            return;
        }
        let Some(h) = host() else {
            return;
        };
        let len = 4 * PAGE;
        let alloc = AliasedAlloc::new(
            len,
            Some(&|p| {
                for i in (0..len).step_by(4096) {
                    // SAFETY: `p` has `len` bytes.
                    unsafe { p.add(i).write((i >> 12) as u8) };
                }
            }),
            true,
        );
        // Initial contents survived remap.
        for i in (0..len).step_by(4096) {
            // SAFETY: `alloc.ptr` has `len` bytes.
            assert_eq!(unsafe { alloc.ptr.add(i).read_volatile() }, (i >> 12) as u8);
        }

        let shadow = crate::shadow::find(alloc.ptr as u64, len as u64).expect("aliased shadow");
        assert!(shadow.is_aliased());
        assert!(shadow.clean());
        assert_eq!(shadow.update(alloc.ptr as usize, len), 0);
        assert_eq!(shadow.copy_back(alloc.ptr as usize, len), 0);

        for round in 0..8u8 {
            let off = len - 2 * PAGE;
            for i in 0..PAGE {
                let b = (i as u8).wrapping_mul(31) ^ round;
                // SAFETY: inside `alloc`.
                unsafe { alloc.ptr.add(off + i).write_volatile(b) };
            }
            let got = gpu_read(&h.device, &shadow.buffer, off as u64, PAGE);
            for (i, &b) in got.iter().enumerate() {
                assert_eq!(b, (i as u8).wrapping_mul(31) ^ round);
            }

            let new: Vec<u8> = (0..PAGE).map(|i| (i as u8) ^ round ^ 0xc3).collect();
            gpu_write(&h.device, &shadow.buffer, (len - PAGE) as u64, &new);
            // SAFETY: inside `alloc`.
            let cpu = unsafe { std::slice::from_raw_parts(alloc.ptr.add(len - PAGE), PAGE) };
            assert_eq!(cpu, &new[..]);
        }
        // SAFETY: querying device health.
        assert!(unsafe { h.device.GetDeviceRemovedReason() }.is_ok());
    }

    #[test]
    fn aliased_gpu_writes_notify_overlapping_texture_tracker() {
        if !crate::watch::rosetta() || !crate::watch::fault_tracking() {
            return;
        }
        const MEM_WRITE_WATCH: u32 = 0x20_0000;
        let len = 2 * PAGE;
        // SAFETY: allocating `len` bytes with `MEM_WRITE_WATCH` for tracker test.
        let ptr = unsafe {
            VirtualAlloc(
                std::ptr::null(),
                len,
                MEM_RESERVE | MEM_COMMIT | MEM_WRITE_WATCH,
                PAGE_READWRITE,
            )
        }
        .cast::<u8>();
        assert!(!ptr.is_null());
        assert!(alias_region(ptr as usize, len, false));
        let alloc = AliasedAlloc {
            ptr,
            len,
            aliased: true,
        };
        let shadow = crate::shadow::find(alloc.ptr as u64, len as u64).expect("aliased shadow");
        let base = alloc.ptr as usize;

        let tex = crate::watch::track(base, len, false).expect("texture tracker");
        let _ = tex.take(base, base + len);
        assert!(tex.clean());

        assert_eq!(shadow.copy_back(base, PAGE), 0);
        assert!(tex.changed(base / PAGE));
        assert!(!tex.changed((base + PAGE) / PAGE));

        let _ = tex.take(base, base + len);
        assert!(tex.clean());
        assert_eq!(shadow.copy_back_marked(&[base + PAGE]), 0);
        assert!(!tex.changed(base / PAGE));
        assert!(tex.changed((base + PAGE) / PAGE));

        let _ = tex.take(base, base + len);
        assert_eq!(shadow.copy_back_page(base), 0);
        assert!(tex.changed(base / PAGE));
    }

    #[test]
    fn aliased_shadow_refreshes_widened_rgb32_view() {
        if !crate::watch::rosetta() {
            return;
        }
        let len = PAGE;
        let alloc = AliasedAlloc::new(len, None, false);
        let shadow = crate::shadow::find(alloc.ptr as u64, len as u64).expect("aliased shadow");
        let widened = shadow
            .widened(alloc.ptr as u64, 48, 0x3f80_0000)
            .expect("widened view");
        assert!(!shadow.clean());

        let words = [1u32, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12];
        // SAFETY: writing 48 bytes at the start of `alloc.ptr`.
        unsafe {
            std::ptr::copy_nonoverlapping(words.as_ptr(), alloc.ptr.cast::<u32>(), words.len());
        }
        assert_eq!(shadow.update(alloc.ptr as usize, 48), 0);
        assert_eq!(
            widened.read_host(16),
            vec![
                1,
                2,
                3,
                0x3f80_0000,
                4,
                5,
                6,
                0x3f80_0000,
                7,
                8,
                9,
                0x3f80_0000,
                10,
                11,
                12,
                0x3f80_0000,
            ]
        );
    }

    #[test]
    fn aliased_memory_supports_virtual_query_and_write_guard() {
        if !crate::watch::rosetta() {
            return;
        }
        let Some(h) = host() else {
            return;
        };
        static GUARD_AT: AtomicUsize = AtomicUsize::new(0);
        static GUARD_HIT: AtomicBool = AtomicBool::new(false);

        unsafe extern "system" fn on_fault(info: *mut EXCEPTION_POINTERS) -> i32 {
            // SAFETY: `info` is provided by Windows/Wine exception dispatch.
            unsafe {
                let rec = &*(*info).ExceptionRecord;
                let page = GUARD_AT.load(Ordering::SeqCst);
                if rec.ExceptionCode as u32 == 0xC000_0005 && page != 0 {
                    let addr = rec.ExceptionInformation[1];
                    if addr >= page && addr < page + PAGE {
                        let mut old = 0;
                        VirtualProtect(page as *const _, PAGE, PAGE_READWRITE, &mut old);
                        GUARD_HIT.store(true, Ordering::SeqCst);
                        return -1;
                    }
                }
                0
            }
        }

        let len = 2 * PAGE;
        let alloc = AliasedAlloc::new(len, None, false);
        let shadow = crate::shadow::find(alloc.ptr as u64, len as u64).expect("aliased shadow");
        // SAFETY: querying our own allocation.
        unsafe {
            let mut info: MEMORY_BASIC_INFORMATION = std::mem::zeroed();
            let q = VirtualQuery(alloc.ptr.add(PAGE).cast(), &mut info, size_of_val(&info));
            assert_ne!(q, 0);
            assert_eq!(info.State, MEM_COMMIT);
            assert_eq!(info.Protect, PAGE_READWRITE);

            let handler = AddVectoredExceptionHandler(1, Some(on_fault));
            let page = alloc.ptr.add(PAGE);
            GUARD_AT.store(page as usize, Ordering::SeqCst);
            GUARD_HIT.store(false, Ordering::SeqCst);
            let mut old = 0;
            assert_ne!(
                VirtualProtect(page.cast(), PAGE, PAGE_READONLY, &mut old),
                0
            );
            page.add(100).write_volatile(0x77);
            let hit = GUARD_HIT.load(Ordering::SeqCst);
            GUARD_AT.store(0, Ordering::SeqCst);
            RemoveVectoredExceptionHandler(handler);
            assert!(hit, "VirtualProtect write guard did not fault");
        }
        assert_eq!(
            gpu_read(&h.device, &shadow.buffer, (PAGE + 100) as u64, 1),
            vec![0x77]
        );
    }

    #[test]
    fn unalias_region_removes_shadow_and_allows_virtual_free() {
        if !crate::watch::rosetta() {
            return;
        }
        let Some(h) = host() else {
            return;
        };
        let len = PAGE;
        // SAFETY: allocating `len` bytes for explicit unalias + free test.
        let ptr = unsafe {
            VirtualAlloc(
                std::ptr::null(),
                len,
                MEM_RESERVE | MEM_COMMIT,
                PAGE_READWRITE,
            )
        }
        .cast::<u8>();
        assert!(!ptr.is_null());
        assert!(alias_region(ptr as usize, len, false));
        let shadow = crate::shadow::find(ptr as u64, len as u64).expect("aliased shadow");
        assert!(shadow.is_aliased());
        assert!(unalias_region(ptr as usize));
        assert!(crate::shadow::find(ptr as u64, len as u64).is_none_or(|s| !s.is_aliased()));
        // SAFETY: `ptr` was allocated by `VirtualAlloc`.
        assert_ne!(unsafe { VirtualFree(ptr.cast(), 0, MEM_RELEASE) }, 0);
        drop(shadow);
        // SAFETY: querying device health.
        assert!(unsafe { h.device.GetDeviceRemovedReason() }.is_ok());
    }
}
