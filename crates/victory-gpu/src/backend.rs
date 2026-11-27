//! What the platform code asks of the host GPU, as traits, so that more than
//! one graphics API can sit under it (VIctory-macOS #221).
//!
//! The console's D3D12.X runs on host D3D12, and [`crate::host::Host`] is
//! that host: one device, its direct queue and the memory it makes. These
//! traits name the parts of it the rest of the crate uses, in terms any host
//! GPU can provide. D3D12 is their first implementation and, for now, the
//! only one: its code moved behind them unchanged. A Metal host, reached from
//! Windows code through a Wine unixlib and able to put buffers over title
//! memory itself (Apple Silicon's memory is unified), is meant to be the
//! second.
//!
//! What is behind the traits so far:
//!
//! - [`HostGpu`]: what the host GPU can run, command lists, and submission.
//! - [`HostMemory`]: host buffers, by how the CPU and the GPU share them.
//!
//! Not yet: recording commands into a list (`HostList::list` is still a
//! D3D12 command list, recorded directly), descriptor heaps and views, root
//! signatures and pipelines, textures, and presentation. `docs/backend.md`
//! lists the order they are meant to follow in.

/// The host GPU: what it can run, its command lists, and its submissions.
pub(crate) trait HostGpu: Send + Sync {
    /// A host command list, recorded on one thread and then run by one
    /// submission at a time.
    type List;
    /// Why the host could not make a list.
    type Error;
    /// Submissions held off for as long as it lives ([`HostGpu::quiet`]).
    type Quiet<'a>
    where
        Self: 'a;

    /// Whether the host runs translated mesh shaders, and what it lacks if
    /// not.
    fn mesh_support(&self) -> crate::host::MeshSupport;

    /// Whether the host's waves have at most 32 lanes, so that wave64
    /// compute programs are translated for waves of 32.
    fn wave32(&self) -> bool;

    /// Whether the host runs 64-bit atomics on typed resources that shaders
    /// reach through the descriptor heap.
    fn heap_atomics64(&self) -> bool;

    /// A new list, open for recording.
    fn new_list(&self) -> Result<Self::List, Self::Error>;

    /// Holds submissions off until the result is dropped, where the host
    /// needs that while a view heap is made and filled. `None` where it does
    /// not, or where this thread holds them off already.
    fn quiet(&self) -> Option<Self::Quiet<'_>>;

    /// Runs closed lists, in order, as one submission, and waits for them.
    fn execute(&self, runs: &[&Self::List]);
}

/// Host buffers, by how the CPU and the GPU share them. Each mapped buffer
/// comes with the CPU address of its first byte, valid while the buffer
/// lives.
pub(crate) trait HostMemory: Send + Sync {
    /// A host buffer.
    type Buffer;

    /// A buffer only the GPU reads and writes, ready to be copied into.
    fn default_buffer(&self, size: u64) -> Option<Self::Buffer>;

    /// A buffer the CPU writes and the GPU reads.
    fn upload_buffer(&self, size: u64) -> Option<(Self::Buffer, *mut u8)>;

    /// A buffer the GPU copies into and the CPU reads.
    fn readback_buffer(&self, size: u64) -> Option<(Self::Buffer, *mut u8)>;

    /// A buffer in memory that the CPU reads and writes quickly and that
    /// shaders read and write too: for host copies of title memory.
    fn shared_buffer(&self, size: u64) -> Option<(Self::Buffer, *mut u8)>;
}
