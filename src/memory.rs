//! Fixed-size chunk allocation over a single, statically-sized backing
//! region.
//!
//! [`MemoryManagement`] hands out equal-sized `ALLOC_SIZE` chunks one at a
//! time and never reclaims them individually - the whole region is released
//! at once when the implementor is dropped. `allocate` takes `&self` rather
//! than `&mut self`, so implementations are expected to make concurrent
//! calls safe on their own (see [`imp::Memory`]'s atomic bump cursor).
//!
//! A concrete implementation, [`Memory`], backed by an anonymous `mmap`'d
//! region, is available with the `std` feature enabled, since it goes
//! through `rustix` for the underlying syscalls.

extern crate alloc;

use core::{ffi::c_void, ptr::NonNull};

use alloc::alloc::AllocError;

/// A fixed-size chunk allocator over a region of `total_size` bytes.
///
/// Every chunk handed out by [`allocate`](Self::allocate) is exactly
/// `ALLOC_SIZE` bytes; there is no way to grow, shrink, or individually free
/// a chunk once allocated. The entire region is expected to be released
/// together, typically via `Drop`.
pub trait MemoryManagement<const ALLOC_SIZE: usize> {
    /// Reserve a backing region large enough for `total_size` bytes.
    fn new(total_size: usize) -> Self;

    fn start(&self) -> NonNull<c_void>;

    /// Hand out one more `ALLOC_SIZE`-byte chunk from the region, or
    /// `Err(AllocError)` once the region has no room left for another one.
    fn allocate(&self) -> Result<NonNull<c_void>, AllocError>;
}

/// A [`MemoryManagement`] implementation backed by a single anonymous
/// `mmap`'d region.
///
/// Requires the `std` feature: mapping and unmapping memory goes through
/// `rustix`, which needs an underlying OS (this is not available in a
/// `no_std` build).
#[cfg(feature = "std")]
mod imp {
    extern crate std;

    use std::{
        alloc::AllocError,
        ffi::c_void,
        ptr::{self, NonNull},
        sync::atomic::{AtomicUsize, Ordering as AtomicOrdering},
    };

    use rustix::mm::{MapFlags, ProtFlags, mmap_anonymous, munmap};

    use crate::memory::MemoryManagement;

    /// A single anonymous `mmap`'d region, bump-allocated in fixed-size
    /// chunks.
    ///
    /// The region is requested up front in [`new`](Self::new) and never
    /// resized. [`allocate`](Self::allocate) never reclaims individual
    /// chunks; the whole region is unmapped at once when `Memory` is
    /// dropped.
    pub struct Memory {
        /// Start of the mapped region.
        start: NonNull<c_void>,
        /// Total size of the mapped region, in bytes; also the value
        /// passed to `munmap` on drop.
        total_size: usize,
        /// Byte offset of the next chunk to hand out, bumped with a CAS
        /// loop so concurrent `allocate` calls each get a disjoint chunk.
        offset: AtomicUsize,
    }

    impl<const ALLOC_SIZE: usize> MemoryManagement<ALLOC_SIZE> for Memory {
        /// Map a fresh, zero-filled region of `total_size` bytes.
        ///
        /// # Panics
        ///
        /// Panics if the underlying `mmap` call fails. Note that the
        /// requested mapping uses `MAP_HUGETLB`, so this will fail unless
        /// the system has huge pages reserved (e.g. via
        /// `/proc/sys/vm/nr_hugepages`).
        fn new(total_size: usize) -> Self {
            let start = match unsafe {
                mmap_anonymous(
                    ptr::null_mut(),
                    total_size,
                    ProtFlags::READ | ProtFlags::WRITE,
                    MapFlags::PRIVATE,
                )
            } {
                Ok(start) => NonNull::new(start).unwrap(),
                Err(error) => {
                    panic!("Unable to allocate {:#x} byte(s): {}", total_size, error);
                }
            };

            Self {
                start,
                total_size,
                offset: AtomicUsize::new(0),
            }
        }

        /// Bump the offset by `ALLOC_SIZE` and return a pointer to the
        /// chunk it used to point to, or `Err(AllocError)` if fewer than
        /// `ALLOC_SIZE` bytes remain in the region.
        ///
        /// Lock-free: concurrent callers race on a `compare_exchange_weak`
        /// loop over `offset`, so each successful call claims a distinct,
        /// non-overlapping chunk.
        fn allocate(&self) -> Result<NonNull<c_void>, std::alloc::AllocError> {
            let mut offset = self.offset.load(AtomicOrdering::Relaxed);

            let offset = loop {
                if offset + ALLOC_SIZE > self.total_size {
                    return Err(AllocError);
                }

                match self.offset.compare_exchange_weak(
                    offset,
                    offset + ALLOC_SIZE,
                    AtomicOrdering::Relaxed,
                    AtomicOrdering::Relaxed,
                ) {
                    Ok(offset) => break offset,
                    Err(new_offset) => {
                        offset = new_offset;
                    }
                };
            };

            unsafe { Ok(self.start.add(offset)) }
        }

        fn start(&self) -> NonNull<c_void> {
            self.start
        }
    }

    impl Drop for Memory {
        /// Unmap the whole region. Errors from `munmap` are ignored, since
        /// there is nothing a `drop` impl can do about them.
        fn drop(&mut self) {
            let _ = unsafe { munmap(self.start.as_ptr(), self.total_size) };
        }
    }
}

#[cfg(feature = "std")]
pub use imp::*;
