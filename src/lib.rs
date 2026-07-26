//! `integrity-rs`: runtime memory-integrity protection with Merkle trees.
//!
//! The crate layers three pieces:
//!
//! - [`merkle_tree`] — an address-indexed, BLAKE3-backed [`MerkleTree`] that
//!   records and verifies the contents of a power-of-two memory region.
//! - [`buddy_allocator`] — a fixed-capacity [`BuddyAllocator`] that hands out
//!   blocks from a backing region.
//! - [`mountable_merkle_tree`] — a two-level ("scalable") tree that composes
//!   the two: many per-sub-region sub-trees under one root tree, so integrity
//!   metadata scales with the memory actually in use.
//!
//! [`MerkleTree`]: merkle_tree::MerkleTree
//! [`BuddyAllocator`]: buddy_allocator::BuddyAllocator
//!
//! # The guard interface
//!
//! [`LockedMountableMerkleTree`] is the high-level entry point: it stores
//! values in integrity-protected memory and hands out
//! [`MountableMerkleTreeGuard`]s. Read a value with
//! [`with`](mountable_merkle_tree::MountableMerkleTreeGuard::with) and mutate
//! it with
//! [`with_mut`](mountable_merkle_tree::MountableMerkleTreeGuard::with_mut);
//! both re-validate the value's hash before handing it to your closure, and
//! `with_mut` re-hashes afterwards so the change is recorded. As long as
//! every write goes through `with_mut`, reads always succeed:
//!
//! ```
//! # #![feature(allocator_api)]
//! # use core::ffi::c_void;
//! # use core::ptr::NonNull;
//! # use std::alloc::AllocError;
//! # use integrity_rs::memory::MemoryManagement;
//! # use integrity_rs::merkle_tree::NodeAllocator;
//! # use integrity_rs::mountable_merkle_tree::{LockedMountableMerkleTree, MEM_PER_SUBTREE};
//! # use parking_lot::RawMutex;
//! # struct HeapMemory { _buf: Box<[u8]>, start: NonNull<u8> }
//! # impl<const A: usize> MemoryManagement<A> for HeapMemory {
//! #     fn new(total: usize) -> Self {
//! #         let buf = vec![0u8; total + MEM_PER_SUBTREE].into_boxed_slice();
//! #         let aligned = (buf.as_ptr() as usize).next_multiple_of(MEM_PER_SUBTREE);
//! #         HeapMemory { start: NonNull::new(aligned as *mut u8).unwrap(), _buf: buf }
//! #     }
//! #     fn start(&self) -> NonNull<c_void> { self.start.cast() }
//! #     fn allocate(&self) -> Result<NonNull<c_void>, AllocError> { unreachable!() }
//! # }
//! let tree = LockedMountableMerkleTree::<RawMutex, NodeAllocator, HeapMemory>::new();
//!
//! let counter = tree.create(0u64);
//! counter.with_mut(|n| *n += 42);
//!
//! // A later read re-validates and returns the recorded value.
//! assert_eq!(counter.with(|n| *n), 42);
//! ```
//!
//! # Misuse: tampering behind the guard's back
//!
//! The guard tracks integrity, not ownership: it also hands out a raw
//! [`as_ptr`](mountable_merkle_tree::MountableMerkleTreeGuard::as_ptr), and
//! writing through it bypasses the re-hash that
//! [`with_mut`](mountable_merkle_tree::MountableMerkleTreeGuard::with_mut)
//! would have done. The recorded hash then no longer matches memory — which
//! is exactly what the tree exists to catch — so the next access panics:
//!
//! ```should_panic
//! # #![feature(allocator_api)]
//! # use core::ffi::c_void;
//! # use core::ptr::NonNull;
//! # use std::alloc::AllocError;
//! # use integrity_rs::memory::MemoryManagement;
//! # use integrity_rs::merkle_tree::NodeAllocator;
//! # use integrity_rs::mountable_merkle_tree::{LockedMountableMerkleTree, MEM_PER_SUBTREE};
//! # use parking_lot::RawMutex;
//! # struct HeapMemory { _buf: Box<[u8]>, start: NonNull<u8> }
//! # impl<const A: usize> MemoryManagement<A> for HeapMemory {
//! #     fn new(total: usize) -> Self {
//! #         let buf = vec![0u8; total + MEM_PER_SUBTREE].into_boxed_slice();
//! #         let aligned = (buf.as_ptr() as usize).next_multiple_of(MEM_PER_SUBTREE);
//! #         HeapMemory { start: NonNull::new(aligned as *mut u8).unwrap(), _buf: buf }
//! #     }
//! #     fn start(&self) -> NonNull<c_void> { self.start.cast() }
//! #     fn allocate(&self) -> Result<NonNull<c_void>, AllocError> { unreachable!() }
//! # }
//! let tree = LockedMountableMerkleTree::<RawMutex, NodeAllocator, HeapMemory>::new();
//! let secret = tree.create(0u64);
//!
//! // Tamper with the value directly instead of going through `with_mut`,
//! // so the tree is never told the memory changed.
//! unsafe { (secret.as_ptr() as *mut u64).write(0xbad) };
//!
//! // The recorded hash no longer matches memory: this panics with
//! // "sub-tree 0 failed integrity check".
//! secret.with(|n| *n);
//! ```
//!
//! [`LockedMountableMerkleTree`]: mountable_merkle_tree::LockedMountableMerkleTree
//! [`MountableMerkleTreeGuard`]: mountable_merkle_tree::MountableMerkleTreeGuard

#![cfg_attr(not(test), no_std)]
#![feature(allocator_api)]

pub mod buddy_allocator;
pub mod merkle_tree;
pub mod memory;
pub mod mountable_merkle_tree;
