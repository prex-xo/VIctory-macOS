//! A Flight-1 flight (`VICTORY_FLIGHT1`, `crate::host::flight`) lands
//! what its shaders wrote with no list declaring it, whatever the title
//! wrote into the same pages while it was in the air.

use std::sync::Arc;

use super::*;
use crate::host::{Gate, ModeGuard};

/// A raw view of a buffer's 256 bytes.
const RAW_256: [u32; 10] = [39, 1, 0, 0, 64, 0, 0, 0, 1, 0];

/// A shader writes a dword through a heap's unordered view (marked, and
/// declared by no list: the bindless path), and while its flight is in the
/// air the title writes another dword of the same 64 KiB page from the CPU,
/// as its render thread goes on writing a frame's data with Flight-1: both
/// writes are in title memory once the flight has landed, as on the
/// console, where memory is written byte by byte.
#[test]
fn a_heap_view_write_and_a_title_write_on_one_page_during_a_flight_both_land() {
    let Some(host) = crate::host::host() else {
        return;
    };
    const LIVE_MARKED_WRITE: &str = r#"
RWByteAddressBuffer dst : register(u0);
RWByteAddressBuffer writes : register(u1);
cbuffer c : register(b0) { uint byte_offset; uint value; uint mark_word_bytes; uint mark_mask; }
[numthreads(1, 1, 1)]
void main() {
    dst.Store(byte_offset, value);
    writes.Store(mark_word_bytes, mark_mask);
}
"#;
    let (root, pso) = compile_cs(
        LIVE_MARKED_WRITE,
        &[
            Param::Table(&[uav(0, 1)]),
            Param::Uav { register: 1 },
            Param::Constants {
                register: 0,
                count: 4,
            },
        ],
    );
    let (_, title_heap, heap_host) = view_heap(8);
    let p = crate::watch::tests::pool(1);
    // SAFETY: a committed write-watched page of ours.
    unsafe {
        (p as *mut u32).write_volatile(0x1111_1111);
        ((p + 16) as *mut u32).write_volatile(0x2222_2222);
    }
    let buf = buffer_at_address(p as u64, crate::watch::PAGE as u64);
    let copy = crate::shadow::find(p as u64, crate::watch::PAGE as u64).expect("the page's copy");
    let Some((bit, _)) = copy.mark_entry(p as u64) else {
        crate::shadow::unregister(buf as usize, p as u64);
        return; // no write marks on this host
    };
    crate::descriptor::create_uav(
        std::ptr::null(),
        buf,
        std::ptr::null(),
        &RAW_256,
        title_heap.start,
    );
    let _ = heap_host.mirror(0, title_heap.start, true);
    let _mode = ModeGuard::new(true, false);
    let mut gate = Gate::hold(host);
    let list = Arc::new(host.new_list().unwrap());
    // The heap's views brought up to date (its unordered views included),
    // as for a list whose shaders index the heap: no write declared.
    list.defer_heap(&heap_host.viewed);
    let writes = crate::shadow::view_writes(false).expect("the write marks bitmap");
    let constants = [0u32, 0xDEAD_BEEF, (bit >> 5) * 4, 1 << (bit & 31)];
    // SAFETY: an open list, live objects and descriptors in our heap.
    unsafe {
        list.list
            .SetDescriptorHeaps(&[Some(heap_host.heap.clone())]);
        list.list.SetComputeRootSignature(&root);
        list.list.SetComputeRootDescriptorTable(0, heap_host.gpu(0));
        list.list.SetComputeRootUnorderedAccessView(1, writes);
        list.list
            .SetComputeRoot32BitConstants(2, 4, constants.as_ptr().cast(), 0);
        list.list.SetPipelineState(&pso);
        list.list.Dispatch(1, 1, 1);
    }
    list.close();
    host.execute_flight1(vec![list], vec![]);
    // The title's render thread goes on, and writes the page meanwhile.
    // SAFETY: a dword of our page.
    unsafe { ((p + 16) as *mut u32).write_volatile(0xCAFE_BABE) };
    gate.open();
    host.finish_flight_now();
    // SAFETY: dwords of our page.
    let (gpu, cpu) = unsafe {
        (
            (p as *const u32).read_volatile(),
            ((p + 16) as *const u32).read_volatile(),
        )
    };
    assert_eq!(cpu, 0xCAFE_BABE, "the title's write");
    assert_eq!(
        gpu, 0xDEAD_BEEF,
        "the shader's write, on the page the title wrote during the flight"
    );
    crate::shadow::unregister(buf as usize, p as u64);
}
