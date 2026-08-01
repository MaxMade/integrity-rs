//! A two-level ("scalable") Merkle tree over an allocator, after the scheme
//! in *Scalable Memory Protection in the PENGLAI Enclave*.
//!
//! The protected region ([`TOTAL_MEM`] bytes) is divided into
//! [`NUM_SUBTREES`] equal, [`MEM_PER_SUBTREE`]-sized sub-regions, each with
//! its own [`MerkleTree`] (a `sub_trees` entry) and its own
//! [`BuddyAllocator`] serving allocations out of that sub-region.
//! A single `root_tree` then covers an array of the sub-trees' *root
//! digests* ([`MerkleTree::root_digest`]): whenever a sub-tree changes, its
//! new root digest is republished into that array and hashed up through the
//! root tree, so the root digest transitively attests to every sub-tree, and
//! therefore to every byte any of them cover. (Hashing the digests, rather
//! than the `MerkleTree` structs, is essential — a re-hash changes a
//! sub-tree's nodes, not its struct bytes.) Integrity metadata thus scales
//! with the memory actually in use (sub-trees only materialize nodes for
//! regions that have been touched — see [`MerkleTree::leaf_node`]) rather
//! than with [`TOTAL_MEM`].
//!
//! What gets published is [`root_digest`](MerkleTree::root_digest), not
//! [`root_hash`](MerkleTree::root_hash), so that a sub-tree's *version* is
//! attested along with its contents. A sub-tree's major counter lives in the
//! `MerkleTree` struct, which no node hashes, so publishing the bare root
//! hash would let a whole sub-region be rolled back to a state it held under
//! an earlier major without the digest moving.
//!
//! That recursion has to stop somewhere: the `root_tree`'s own major has no
//! level above it to be published into, so nothing here attests it. Closing
//! that last step needs a freshness anchor outside this structure — storage
//! an attacker cannot roll back, holding the last-seen top-level digest.
//! Until then the tree detects tampering, but a rollback of the *entire*
//! structure — protected memory, nodes and digests together — still
//! validates.
//!
//! [`LockedMountableMerkleTree`] wraps the whole structure in a mutex and
//! hands out [`MountableMerkleTreeGuard`]s that re-validate on read and
//! re-hash on write, so callers interact with protected values without
//! touching the tree machinery directly.

extern crate alloc;

use core::ffi::c_void;
use core::ptr::NonNull;

use crate::{
    buddy_allocator::{BuddyAllocator, MIN_ALLOC_SIZE},
    memory::MemoryManagement,
    merkle_tree::{self, MerkleTree, MerkleTreeNodeAllocator},
};

use alloc::alloc::AllocError;
use alloc::alloc::Layout;
use alloc::boxed::Box;
use parking_lot::lock_api::{Mutex, RawMutex};

/// Total size, in bytes, of the region this tree protects.
pub const TOTAL_MEM: usize = 1024 * 1024 * 1024;

/// Size, in bytes, of each sub-region covered by one sub-tree. Must be a
/// power of two (a [`MerkleTree`] requirement).
pub const MEM_PER_SUBTREE: usize = 4 * 1024 * 1024;

/// Number of sub-trees the region is divided into.
pub const NUM_SUBTREES: usize = TOTAL_MEM / MEM_PER_SUBTREE;

/// Size, in bytes, of the sub-region covered by a single leaf of one
/// sub-tree — the granularity at which a sub-tree tracks integrity, and the
/// size of the slice `rehash`/`validate` hash per touched leaf.
pub const MEM_PER_SUBTREE_LEAF: usize = MEM_PER_SUBTREE / merkle_tree::NUM_LEAF_NODES;

/// Size, in bytes, of one entry of `sub_tree_hashes` (a BLAKE3 digest).
const DIGEST_SIZE: usize = 32;

/// Total size, in bytes, of the `sub_tree_hashes` array the root tree
/// covers. A power of two, since [`NUM_SUBTREES`] is.
const ROOT_TREE_SIZE: usize = NUM_SUBTREES * DIGEST_SIZE;

/// Bytes of `sub_tree_hashes` covered by a single root-tree leaf.
const ROOT_TREE_LEAF: usize = ROOT_TREE_SIZE / merkle_tree::NUM_LEAF_NODES;

const BUDDY_ALLOCATOR_HEIGHT: usize =
    usize::ilog2(MEM_PER_SUBTREE_LEAF / MIN_ALLOC_SIZE) as usize + 1;

/// One sub-tree's published root digest. Over-aligned so the
/// `sub_tree_hashes` array starts on an even address, which
/// [`MerkleTree`] requires of the region it covers.
#[repr(C, align(8))]
#[derive(Clone, Copy)]
struct Digest([u8; DIGEST_SIZE]);

/// A [`MerkleTree`] per [`MEM_PER_SUBTREE`]-sized sub-region, all protected
/// by a single `root_tree`. See the [module docs](self) for the scheme.
pub struct MountableMerkleTree<MTNA: MerkleTreeNodeAllocator, MM: MemoryManagement<MEM_PER_SUBTREE>>
{
    /// Protects the `sub_tree_hashes` array, and thus — since each entry is
    /// a sub-tree's own root digest — transitively attests to every
    /// sub-tree.
    root_tree: MerkleTree<MTNA>,

    /// One tree per sub-region; `sub_trees[i]` covers
    /// `[start + i * MEM_PER_SUBTREE, start + (i + 1) * MEM_PER_SUBTREE)`.
    sub_trees: Box<[MerkleTree<MTNA>]>,

    /// `sub_tree_hashes[i]` is the last published root digest of
    /// `sub_trees[i]` (see [`MerkleTree::root_digest`]). This is the array
    /// the `root_tree` actually covers, so re-hashing it after a sub-tree
    /// changes carries that change up into the root digest — the bytes of
    /// the `sub_trees` structs themselves never change on a re-hash and so
    /// could not serve this purpose.
    sub_tree_hashes: Box<[Digest]>,

    /// Owns the backing region all sub-trees and allocators are laid over.
    memory_management: MM,

    /// Cursor for [`allocate`](Self::allocate)'s round-robin over
    /// `memory_allocators`; only ever incremented, taken `% NUM_SUBTREES`.
    memory_allocator_idx: usize,

    /// One buddy allocator per sub-region, each `fill`ed with exactly that
    /// sub-region's [`MEM_PER_SUBTREE`] bytes.
    memory_allocators: Box<[BuddyAllocator<BUDDY_ALLOCATOR_HEIGHT>]>,
}

impl<MTNA: MerkleTreeNodeAllocator, MM: MemoryManagement<MEM_PER_SUBTREE>>
    MountableMerkleTree<MTNA, MM>
{
    /// Reserve the full [`TOTAL_MEM`] region and lay a per-sub-region tree
    /// and buddy allocator over every one of the [`NUM_SUBTREES`]
    /// sub-regions. Sub-trees start [`empty`](MerkleTree::empty) (no nodes)
    /// and materialize lazily; only `root_tree` is fully constructed up
    /// front.
    ///
    /// # Panics
    ///
    /// Panics if the backing [`MemoryManagement`] or any node allocation
    /// fails.
    pub fn new() -> Self {
        let memory_management = MM::new(TOTAL_MEM);

        // One empty tree per sub-region. Nothing hashes this array (the
        // root tree covers `sub_tree_hashes` instead), so it needs no
        // special layout — a plain collected `Box<[_]>` frees itself.
        let sub_trees: Box<[MerkleTree<MTNA>]> = (0..NUM_SUBTREES)
            .map(|i| {
                let ptr = unsafe { memory_management.start().byte_add(i * MEM_PER_SUBTREE) };
                MerkleTree::empty(ptr, MEM_PER_SUBTREE)
            })
            .collect();

        // Published sub-tree digests, seeded from the sub-trees themselves so
        // the array agrees with them from the outset. This — not the
        // `sub_trees` struct array — is what `root_tree` covers, so that
        // re-hashing it after a sub-tree changes actually moves the root
        // digest. Note these are not all-zero: an empty tree's `root_digest`
        // still hashes its major over the zeroed root hash.
        let sub_tree_hashes: Box<[Digest]> = sub_trees
            .iter()
            .map(|sub_tree| Digest(sub_tree.root_digest()))
            .collect();

        let root_tree = MerkleTree::constructed(
            NonNull::new(sub_tree_hashes.as_ptr() as *mut c_void).unwrap(),
            ROOT_TREE_SIZE,
        );

        // One buddy allocator per sub-region, each fed exactly that
        // sub-region's bytes.
        let memory_allocators: Box<[BuddyAllocator<BUDDY_ALLOCATOR_HEIGHT>]> = (0..NUM_SUBTREES)
            .map(|i| {
                let mut allocator = BuddyAllocator::new();
                let ptr = unsafe { memory_management.start().byte_add(i * MEM_PER_SUBTREE) };
                allocator.fill(ptr.cast(), MEM_PER_SUBTREE);
                allocator
            })
            .collect();

        Self {
            root_tree,
            sub_trees,
            sub_tree_hashes,
            memory_management,
            memory_allocators,
            memory_allocator_idx: 0,
        }
    }

    /// Serve `layout` from the sub-regions, round-robining over their buddy
    /// allocators starting at `memory_allocator_idx` and advancing it by one
    /// per attempt, so load spreads across sub-regions rather than filling
    /// the first one. Falls through to the next allocator whenever one can't
    /// satisfy the request, and returns [`AllocError`] only once every
    /// sub-region has been tried and refused.
    ///
    /// A successful allocation perturbs the buddy allocator's free-list
    /// bookkeeping, which lives *inside* the sub-region's free memory and so
    /// falls under the sub-tree — therefore the whole sub-region is
    /// re-hashed before returning (see [`rehash_subtree`](Self::rehash_subtree)).
    fn allocate(&mut self, layout: Layout) -> Result<NonNull<[u8]>, AllocError> {
        for _ in 0..NUM_SUBTREES {
            let idx = self.memory_allocator_idx % NUM_SUBTREES;
            self.memory_allocator_idx += 1;

            match self.memory_allocators[idx].allocate(layout) {
                Ok(mem) => {
                    self.rehash_subtree(idx);
                    return Ok(mem);
                }
                Err(_) => {
                    /* Try next allocator */
                    continue;
                }
            }
        }

        Err(AllocError)
    }

    /// Return a block to the buddy allocator of the sub-region that owns it,
    /// found from `ptr`'s offset into the backing region (not from any
    /// round-robin state — a block must go back to the exact allocator it
    /// came from), then re-hash that sub-region: freeing rewrites free-list
    /// bookkeeping inside the sub-region, which the sub-tree covers.
    ///
    /// # Safety
    ///
    /// `ptr`/`layout` must be a block previously returned from
    /// [`allocate`](Self::allocate) on this tree and not yet freed.
    ///
    /// # Panics
    ///
    /// Panics if `ptr` lies outside this tree's backing region.
    unsafe fn deallocate(&mut self, ptr: NonNull<[u8]>, layout: Layout) {
        assert!(ptr.addr() >= self.memory_management.start().addr());
        let offset = ptr.addr().get() - self.memory_management.start().addr().get();

        assert!(offset < TOTAL_MEM);
        let idx = offset / MEM_PER_SUBTREE;

        unsafe { self.memory_allocators[idx].deallocate(ptr.cast(), layout) };
        self.rehash_subtree(idx);
    }

    /// The sub-region index of `ptr`, and the leaf-aligned start of the
    /// [`MEM_PER_SUBTREE_LEAF`]-byte slice of that sub-region containing it —
    /// the exact `(ptr, size)` a sub-tree hashes for anything in that leaf.
    ///
    /// # Panics
    ///
    /// Panics if `ptr` lies outside this tree's backing region.
    fn subtree_leaf(&self, ptr: NonNull<c_void>) -> (usize, NonNull<u8>) {
        let start = self.memory_management.start().addr().get();
        assert!(ptr.addr().get() >= start);
        let offset = ptr.addr().get() - start;
        assert!(offset < TOTAL_MEM);

        let idx = offset / MEM_PER_SUBTREE;
        let leaf_offset = (offset / MEM_PER_SUBTREE_LEAF) * MEM_PER_SUBTREE_LEAF;
        let leaf = unsafe { self.memory_management.start().byte_add(leaf_offset) };
        (idx, leaf.cast())
    }

    /// The leaf-aligned start of the [`ROOT_TREE_LEAF`]-byte slice of
    /// `sub_tree_hashes` holding `sub_tree_hashes[idx]` — the `(ptr, size)`
    /// the root tree hashes for sub-tree `idx`.
    fn root_leaf(&self, idx: usize) -> NonNull<u8> {
        let leaf_offset = (idx * DIGEST_SIZE / ROOT_TREE_LEAF) * ROOT_TREE_LEAF;
        let base = self.sub_tree_hashes.as_ptr() as *mut u8;
        unsafe { NonNull::new_unchecked(base.add(leaf_offset)) }
    }

    /// Re-hash the sub-tree leaf covering `ptr` from the current memory
    /// contents, republish that sub-tree's root digest into
    /// `sub_tree_hashes`, and carry the change up through the root tree.
    /// Call after any write to memory reachable from `ptr`.
    ///
    /// Hashing is at [`MEM_PER_SUBTREE_LEAF`] granularity: the whole leaf
    /// slice containing `ptr` is hashed, not just the object, so a leaf may
    /// hold several objects (and the allocator's free-block bookkeeping).
    /// Consistency across all of them holds because every mutation of a
    /// leaf's bytes is followed by a re-hash — object writes here, and
    /// allocator bookkeeping via [`rehash_subtree`](Self::rehash_subtree)
    /// from [`allocate`](Self::allocate)/[`deallocate`](Self::deallocate).
    fn rehash(&mut self, ptr: NonNull<c_void>) {
        let (idx, leaf) = self.subtree_leaf(ptr);
        unsafe { self.sub_trees[idx].rehash(leaf, MEM_PER_SUBTREE_LEAF) };

        // Publish the sub-tree's new root digest into the array the root
        // tree covers, then re-hash the root leaf holding it. `root_digest`
        // rather than `root_hash`, so the sub-tree's major counter travels up
        // with its contents.
        let digest = self.sub_trees[idx].root_digest();
        self.sub_tree_hashes[idx] = Digest(digest);

        let root_leaf = self.root_leaf(idx);
        unsafe { self.root_tree.rehash(root_leaf, ROOT_TREE_LEAF) };
    }

    /// Re-hash *every* leaf of sub-region `idx`, then republish its digest.
    ///
    /// Used after an allocate/deallocate: a single buddy allocator threads
    /// its free lists through free blocks anywhere in its sub-region, so one
    /// allocation can rewrite bookkeeping bytes in several of that
    /// sub-region's leaves at once — not just the leaf the block came from.
    /// Recording all of them keeps every co-resident object's leaf hash in
    /// step with memory, so a later [`validate`](Self::validate) of any of
    /// them still matches.
    fn rehash_subtree(&mut self, idx: usize) {
        let region = unsafe {
            self.memory_management
                .start()
                .byte_add(idx * MEM_PER_SUBTREE)
        };
        for leaf in 0..merkle_tree::NUM_LEAF_NODES {
            let leaf_ptr = unsafe { region.byte_add(leaf * MEM_PER_SUBTREE_LEAF) };
            self.rehash(leaf_ptr);
        }
    }

    /// Verify the integrity of the leaf covering `ptr`, all the way to the
    /// root digest: the sub-tree leaf must still match memory and be
    /// internally consistent, its published digest must still equal the
    /// sub-tree's current [`root_digest`](MerkleTree::root_digest) — version
    /// included, not just contents — and the root tree must be internally
    /// consistent over `sub_tree_hashes`.
    ///
    /// # Panics
    ///
    /// Panics if any of those checks fails — i.e. if memory covered by
    /// `ptr`'s leaf, the sub-tree, or the published digests has been
    /// tampered with since the last [`rehash`](Self::rehash).
    fn validate(&mut self, ptr: NonNull<c_void>) {
        let (idx, leaf) = self.subtree_leaf(ptr);

        // Explicit `panic!`, not `assert!`: this is a security verdict that
        // must fire in every build, and each check runs a `&mut self`
        // `validate` that lazily materializes nodes — a side effect that has
        // no business living inside an assertion's condition.
        let sub_tree_ok = unsafe { self.sub_trees[idx].validate(leaf, MEM_PER_SUBTREE_LEAF) };
        if !sub_tree_ok {
            panic!("sub-tree {idx} failed integrity check");
        }

        if self.sub_tree_hashes[idx].0 != self.sub_trees[idx].root_digest() {
            panic!("sub-tree {idx} digest out of sync with root tree");
        }

        let root_leaf = self.root_leaf(idx);
        let root_ok = unsafe { self.root_tree.validate(root_leaf, ROOT_TREE_LEAF) };
        if !root_ok {
            panic!("root tree failed integrity check");
        }
    }
}

pub struct LockedMountableMerkleTree<
    L: RawMutex,
    MTNA: MerkleTreeNodeAllocator,
    MM: MemoryManagement<MEM_PER_SUBTREE>,
>(Mutex<L, MountableMerkleTree<MTNA, MM>>);

impl<L: RawMutex, MTNA: MerkleTreeNodeAllocator, MM: MemoryManagement<MEM_PER_SUBTREE>>
    LockedMountableMerkleTree<L, MTNA, MM>
{
    pub fn new() -> Self {
        Self(Mutex::new(MountableMerkleTree::new()))
    }

    /// Store `t` in the protected region and return a guard mediating access
    /// to it. The value's leaf is hashed into the tree up front, so a
    /// subsequent [`with`](MountableMerkleTreeGuard::with) validates cleanly.
    ///
    /// # Panics
    ///
    /// Panics if no sub-region can satisfy the allocation.
    pub fn create<T>(&self, t: T) -> MountableMerkleTreeGuard<'_, MTNA, MM, L, T> {
        let layout = Layout::new::<T>();

        let mut mmt = self.0.lock();

        let ptr = match mmt.allocate(layout) {
            Ok(ptr) => ptr.cast(),
            Err(error) => panic!("Unable to serve memory allocation: {}", error),
        };

        unsafe { ptr.write(t) }

        // Record the freshly-written value in the tree.
        mmt.rehash(ptr.cast());

        MountableMerkleTreeGuard { mmt: self, ptr }
    }
}

pub struct MountableMerkleTreeGuard<
    'a,
    MTNA: MerkleTreeNodeAllocator,
    MM: MemoryManagement<MEM_PER_SUBTREE>,
    L: RawMutex,
    T,
> {
    mmt: &'a LockedMountableMerkleTree<L, MTNA, MM>,
    ptr: NonNull<T>,
}

impl<'a, MTNA: MerkleTreeNodeAllocator, MM: MemoryManagement<MEM_PER_SUBTREE>, L: RawMutex, T>
    MountableMerkleTreeGuard<'a, MTNA, MM, L, T>
{
    /// The raw address of the protected value, an escape hatch for reads
    /// that don't fit [`with`](Self::with) (FFI, `unsafe` field access, …).
    ///
    /// This bypasses the integrity machinery entirely: it takes no lock and
    /// performs no validation, so unlike `with` a read through it is *not*
    /// checked against the recorded hash. Reading is otherwise sound while
    /// the guard is alive.
    ///
    /// Writing through it (via `*mut T`) is the documented way to *break*
    /// the invariant — the tree is never told the memory changed, so the
    /// recorded hash goes stale and the next [`with`](Self::with) /
    /// [`with_mut`](Self::with_mut) panics. Mutate through
    /// [`with_mut`](Self::with_mut) instead, which re-hashes afterwards.
    pub fn as_ptr(&self) -> *const T {
        self.ptr.as_ptr()
    }

    /// Validate the stored value, copy it out, and run `cb` against that
    /// copy — with the lock released for the duration of `cb`.
    ///
    /// Restricted to `T: Copy`: the value is duplicated with a bitwise
    /// `read()`, which is only sound when duplication can't alias and there
    /// is no destructor to double-run — exactly what `Copy` guarantees
    /// (`Copy` and `Drop` are mutually exclusive). Because `cb` sees a
    /// detached copy, it may hold the lock again / re-enter this tree.
    ///
    /// # Panics
    ///
    /// Panics if the value fails its integrity check.
    pub fn with<R, CB: FnOnce(&T) -> R>(&self, cb: CB) -> R
    where
        T: Copy,
    {
        let value = {
            let mut mmt = self.mmt.0.lock();
            mmt.validate(self.ptr.cast());
            unsafe { self.ptr.read() }
        };

        cb(&value)
    }

    /// Like [`with`](Self::with) but hands `cb` a mutable reference to the
    /// copy and, after `cb` returns, writes it back and re-hashes so the
    /// mutation is recorded.
    ///
    /// The lock is released while `cb` runs and re-taken only for the
    /// write-back, so concurrent `with_mut`s are last-writer-wins; this
    /// stays memory-safe purely because `T: Copy` (the write-back overwrites
    /// without dropping, and the copy has no destructor).
    ///
    /// # Panics
    ///
    /// Panics if the value fails its integrity check on entry.
    pub fn with_mut<R, CB: FnOnce(&mut T) -> R>(&self, cb: CB) -> R
    where
        T: Copy,
    {
        let mut value = {
            let mut mmt = self.mmt.0.lock();
            mmt.validate(self.ptr.cast());
            unsafe { self.ptr.read() }
        };

        // Callback runs without the lock held.
        let result = cb(&mut value);

        let mut mmt = self.mmt.0.lock();
        unsafe { self.ptr.write(value) };
        mmt.rehash(self.ptr.cast());
        result
    }
}

impl<'a, MTNA: MerkleTreeNodeAllocator, MM: MemoryManagement<MEM_PER_SUBTREE>, L: RawMutex, T> Drop
    for MountableMerkleTreeGuard<'a, MTNA, MM, L, T>
{
    fn drop(&mut self) {
        let mut mmt = self.mmt.0.lock();

        let layout = Layout::new::<T>();
        unsafe {
            // Run `T`'s destructor on the stored value, then return its
            // block to the sub-region allocator it came from. `deallocate`
            // re-hashes the whole sub-region afterwards, so the destructor's
            // writes and the freed-block bookkeeping are both recorded — no
            // separate re-hash is needed here.
            core::ptr::drop_in_place(self.ptr.as_ptr());
            let block = NonNull::slice_from_raw_parts(self.ptr.cast::<u8>(), layout.size());
            mmt.deallocate(block, layout);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::merkle_tree::NodeAllocator;

    /// A [`MemoryManagement`] backed by a plain heap allocation, for tests:
    /// the real, `mmap`-backed [`crate::memory::Memory`] requires huge pages
    /// to be reserved on the host, which isn't guaranteed here.
    ///
    /// The exposed [`start`](MemoryManagement::start) is over-aligned to
    /// [`MEM_PER_SUBTREE`], matching the page/huge-page alignment a real
    /// `mmap` gives, so that buddy blocks (which are size-aligned) line up
    /// with sub-tree leaf boundaries instead of straddling them.
    ///
    /// Alignment comes from over-allocating and offsetting into a plain
    /// zeroed `Box<[u8]>` (lazily backed by the OS, so the full gigabyte is
    /// never physically committed) rather than an over-aligned
    /// `alloc_zeroed`, which would eagerly memset the whole region.
    struct HeapMemory {
        _buf: Box<[u8]>,
        start: NonNull<u8>,
    }

    impl<const ALLOC_SIZE: usize> MemoryManagement<ALLOC_SIZE> for HeapMemory {
        fn new(total_size: usize) -> Self {
            let buf = alloc::vec![0u8; total_size + MEM_PER_SUBTREE].into_boxed_slice();
            let base = buf.as_ptr() as usize;
            let aligned = base.next_multiple_of(MEM_PER_SUBTREE);
            Self {
                start: NonNull::new(aligned as *mut u8).unwrap(),
                _buf: buf,
            }
        }

        fn start(&self) -> NonNull<c_void> {
            self.start.cast()
        }

        fn allocate(&self) -> Result<NonNull<c_void>, alloc::alloc::AllocError> {
            unreachable!("MountableMerkleTree::new never calls MemoryManagement::allocate")
        }
    }

    #[test]
    fn new_then_drop_does_not_crash() {
        let tree = MountableMerkleTree::<NodeAllocator, HeapMemory>::new();
        drop(tree);
    }

    #[test]
    fn allocate_wraps_the_round_robin_index_past_num_subtrees_calls() {
        let mut tree = MountableMerkleTree::<NodeAllocator, HeapMemory>::new();
        let layout = Layout::new::<[u8; 64]>();

        // memory_allocator_idx keeps counting up across calls; this must
        // still succeed well past NUM_SUBTREES calls instead of indexing
        // memory_allocators out of bounds.
        for _ in 0..(NUM_SUBTREES * 2 + 1) {
            assert!(tree.allocate(layout).is_ok());
        }
    }

    #[test]
    fn deallocate_returns_memory_to_its_owning_subtree_allocator() {
        let mut tree = MountableMerkleTree::<NodeAllocator, HeapMemory>::new();
        let layout = Layout::new::<[u8; 64]>();

        let mem = tree.allocate(layout).unwrap();
        unsafe { tree.deallocate(mem, layout) };

        // The freed block must be handed back out again rather than the
        // allocator treating its region as still full.
        assert!(tree.allocate(layout).is_ok());
    }

    #[test]
    fn allocate_reports_alloc_error_once_every_subregion_is_exhausted() {
        let mut tree = MountableMerkleTree::<NodeAllocator, HeapMemory>::new();

        // MEM_PER_SUBTREE_LEAF is the largest block a sub-region's buddy
        // allocator can serve, so each sub-region yields only a handful and
        // the whole tree is exhaustible. Keep pulling until it refuses.
        let layout = Layout::from_size_align(MEM_PER_SUBTREE_LEAF, MEM_PER_SUBTREE_LEAF).unwrap();

        // Bounded well above the true capacity so a regression that never
        // reports exhaustion fails the test instead of looping forever.
        let cap = NUM_SUBTREES * MEM_PER_SUBTREE / MEM_PER_SUBTREE_LEAF + 1;
        let mut served = 0;
        while tree.allocate(layout).is_ok() {
            served += 1;
            assert!(served <= cap, "allocate never reported exhaustion");
        }

        // It spread across sub-regions (served far more than one region's
        // worth) before finally reporting AllocError.
        assert!(served >= NUM_SUBTREES);
    }

    #[test]
    fn deallocate_keeps_memory_in_circulation_regardless_of_cursor_position() {
        let mut tree = MountableMerkleTree::<NodeAllocator, HeapMemory>::new();
        let layout = Layout::new::<[u8; 64]>();

        // Drive the round-robin cursor past sub-region 0 so freed blocks
        // come from assorted sub-regions, and confirm each one is routed
        // back to its owning allocator (by offset) and re-served rather than
        // leaked.
        for _ in 0..(NUM_SUBTREES + 3) {
            let mem = tree.allocate(layout).unwrap();
            unsafe { tree.deallocate(mem, layout) };
            assert!(tree.allocate(layout).is_ok());
        }
    }

    #[test]
    #[should_panic]
    fn deallocate_rejects_a_pointer_outside_the_region() {
        let mut tree = MountableMerkleTree::<NodeAllocator, HeapMemory>::new();
        let layout = Layout::new::<u8>();

        // An address below the backing region must trip the lower-bound
        // assert rather than indexing some sub-region.
        let bogus = NonNull::slice_from_raw_parts(NonNull::<u8>::dangling(), 1);
        unsafe { tree.deallocate(bogus, layout) };
    }

    /// Allocate one value, write it, and record it in both tree levels.
    fn allocate_and_record(
        tree: &mut MountableMerkleTree<NodeAllocator, HeapMemory>,
        value: u64,
    ) -> NonNull<u64> {
        let ptr = tree.allocate(Layout::new::<u64>()).unwrap().cast::<u64>();
        unsafe { ptr.write(value) };
        tree.rehash(ptr.cast());
        ptr
    }

    #[test]
    fn rehash_then_validate_roundtrips_through_both_levels() {
        let mut tree = MountableMerkleTree::<NodeAllocator, HeapMemory>::new();
        let ptr = allocate_and_record(&mut tree, 0xdead_beef);

        // Untampered: sub-tree leaf, published digest, and root tree all
        // agree, so validate returns without panicking.
        tree.validate(ptr.cast());
    }

    /// Allocate one half-leaf block, pinned to sub-region 0.
    fn alloc_in_region0(
        tree: &mut MountableMerkleTree<NodeAllocator, HeapMemory>,
        layout: Layout,
    ) -> NonNull<u8> {
        tree.memory_allocator_idx = 0;
        tree.allocate(layout).unwrap().cast::<u8>()
    }

    #[test]
    fn a_later_allocation_keeps_a_co_resident_object_valid() {
        let mut tree = MountableMerkleTree::<NodeAllocator, HeapMemory>::new();
        // Half-leaf blocks: a live block and a freeable neighbour fit in one
        // leaf together.
        let q =
            Layout::from_size_align(MEM_PER_SUBTREE_LEAF / 2, MEM_PER_SUBTREE_LEAF / 2).unwrap();

        // Fill the first two leaves of sub-region 0, keeping the first half
        // of each (`a`, `c`) live and freeing the second half of each (`b`,
        // `d`). That leaves two free half-leaf blocks, in different leaves,
        // on the same buddy free list — and each sharing its leaf with a
        // live block.
        let a = alloc_in_region0(&mut tree, q);
        let b = alloc_in_region0(&mut tree, q);
        let c = alloc_in_region0(&mut tree, q);
        let d = alloc_in_region0(&mut tree, q);
        unsafe {
            a.write(0xAA);
            c.write(0xCC);
            tree.deallocate(NonNull::slice_from_raw_parts(b, q.size()), q);
            tree.deallocate(NonNull::slice_from_raw_parts(d, q.size()), q);
        }

        // Record and confirm the two live blocks.
        tree.rehash(a.cast());
        tree.rehash(c.cast());
        tree.validate(a.cast());
        tree.validate(c.cast());

        // A fresh allocation reuses one freed neighbour and rewrites the
        // OTHER free block's free-list links — bytes that live in `a`'s or
        // `c`'s leaf. Both must still validate, which holds only because
        // `allocate` re-hashes the whole sub-region.
        let _e = alloc_in_region0(&mut tree, q);
        tree.validate(a.cast());
        tree.validate(c.cast());
    }

    #[test]
    #[should_panic(expected = "sub-tree")]
    fn validate_detects_tampering_of_the_protected_value() {
        let mut tree = MountableMerkleTree::<NodeAllocator, HeapMemory>::new();
        let ptr = allocate_and_record(&mut tree, 1);

        // Overwrite the value behind the tree's back, without re-hashing.
        unsafe { ptr.write(2) };

        tree.validate(ptr.cast());
    }

    #[test]
    fn published_digests_cover_the_sub_tree_version() {
        let mut tree = MountableMerkleTree::<NodeAllocator, HeapMemory>::new();
        let ptr = allocate_and_record(&mut tree, 1);
        let (idx, _) = tree.subtree_leaf(ptr.cast());

        // What gets published must be the sub-tree's full attestation, not
        // its bare root hash — the latter leaves the major counter, and so
        // any rollback across an epoch, unattested.
        assert_eq!(
            tree.sub_tree_hashes[idx].0,
            tree.sub_trees[idx].root_digest()
        );
        assert_ne!(tree.sub_tree_hashes[idx].0, tree.sub_trees[idx].root_hash());

        // The seeded array agrees with its sub-trees before anything is
        // written too, so the invariant holds from construction.
        let fresh = MountableMerkleTree::<NodeAllocator, HeapMemory>::new();
        for i in [0, 1, NUM_SUBTREES - 1] {
            assert_eq!(fresh.sub_tree_hashes[i].0, fresh.sub_trees[i].root_digest());
        }
    }

    #[test]
    fn validation_survives_a_sub_tree_changing_epoch() {
        let mut tree = MountableMerkleTree::<NodeAllocator, HeapMemory>::new();

        // A live value in sub-region 0, plus churn pinned to that same region
        // so its leaves are re-hashed often enough to wrap a minor counter.
        // Every allocate and deallocate re-hashes all 8 leaves, so this
        // crosses at least one epoch.
        let ptr = {
            tree.memory_allocator_idx = 0;
            let ptr = tree.allocate(Layout::new::<u64>()).unwrap().cast::<u64>();
            unsafe { ptr.write(0xfeed) };
            tree.rehash(ptr.cast());
            ptr
        };
        let layout = Layout::new::<[u8; 64]>();
        for _ in 0..200 {
            tree.memory_allocator_idx = 0;
            let mem = tree.allocate(layout).unwrap();
            unsafe { tree.deallocate(mem, layout) };
        }

        assert!(
            tree.sub_trees[0].major() > 0,
            "churn did not cross an epoch; the test no longer exercises one"
        );
        assert!(tree.root_tree.major() > 0);

        // The epoch change re-recorded every leaf of the sub-tree and
        // republished its digest, so the original value still validates all
        // the way up.
        tree.validate(ptr.cast());
        assert_eq!(unsafe { ptr.read() }, 0xfeed);
    }

    #[test]
    #[should_panic(expected = "digest out of sync")]
    fn validate_detects_tampering_of_a_published_sub_tree_digest() {
        let mut tree = MountableMerkleTree::<NodeAllocator, HeapMemory>::new();
        // The first allocation is served from sub-region 0, so its digest
        // lives at sub_tree_hashes[0].
        let ptr = allocate_and_record(&mut tree, 1);

        // Corrupt the recorded digest without touching the sub-tree it
        // should mirror.
        tree.sub_tree_hashes[0] = Digest([0xff; DIGEST_SIZE]);

        tree.validate(ptr.cast());
    }

    #[test]
    fn guard_roundtrips_create_read_and_mutate() {
        use parking_lot::RawMutex;

        let tree = LockedMountableMerkleTree::<RawMutex, NodeAllocator, HeapMemory>::new();
        let guard = tree.create(1000u64);

        assert_eq!(guard.with(|x| *x), 1000);
        guard.with_mut(|x| *x += 337);
        assert_eq!(guard.with(|x| *x), 1337);
    }

    #[test]
    fn dropping_a_guard_runs_the_value_destructor() {
        use core::sync::atomic::{AtomicUsize, Ordering};
        use parking_lot::RawMutex;

        static DROPPED: AtomicUsize = AtomicUsize::new(0);

        // A non-ZST payload with an observable destructor.
        struct Noisy(#[allow(dead_code)] u64);
        impl Drop for Noisy {
            fn drop(&mut self) {
                DROPPED.fetch_add(1, Ordering::Relaxed);
            }
        }

        let tree = LockedMountableMerkleTree::<RawMutex, NodeAllocator, HeapMemory>::new();
        {
            let _guard = tree.create(Noisy(7));
            assert_eq!(DROPPED.load(Ordering::Relaxed), 0);
        }
        // Guard dropped: the stored value's destructor must have run exactly
        // once (and the block returned to its allocator).
        assert_eq!(DROPPED.load(Ordering::Relaxed), 1);
    }
}
