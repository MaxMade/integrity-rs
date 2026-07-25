//! An address-indexed, fixed-height Merkle tree over real memory.
//!
//! [`MerkleTree`] covers a single power-of-two-sized, power-of-two-aligned
//! region `[ptr, ptr + size)` with a complete binary tree of `HEIGHT` levels.
//! The tree is stored as a flat array of `NUM_NODES = 2^HEIGHT - 1` nodes
//! (node `i`'s children live at `2i + 1` and `2i + 2`), so it has exactly
//! `NUM_LEAF_NODES = 2^(HEIGHT - 1)` leaves, each covering an equal-sized
//! slice of the region.
//!
//! [`leaf_node`](MerkleTree::leaf_node) maps a `[ptr, ptr + size)` byte range
//! back to the single leaf that fully contains it by binary-searching the
//! address one level at a time: at each level the current window is halved,
//! and the address falls in either the lower half (left child, same start)
//! or the upper half (right child, start moved up by the halved size).
//!
//! [`rehash`](MerkleTree::rehash) and [`validate`](MerkleTree::validate) both
//! read the actual bytes at `[ptr, ptr + size)` (via `unsafe` raw-pointer
//! dereference) and hash them with BLAKE3, rather than taking a
//! caller-supplied hash: `rehash` records what the covered memory currently
//! contains, and `validate` reports whether it still does. Every ancestor's
//! hash is `blake3(left.hash || right.hash)`, so it commits to the content
//! of every leaf hash below it.
//!
//! Node storage is decoupled from the tree logic via the
//! [`MerkleTreeNodeAllocator`] trait, so callers can back the tree with
//! whatever allocation strategy fits (e.g. a [`BuddyAllocator`](crate::buddy_allocator::BuddyAllocator)).
//! A `Box`-backed implementation, [`NodeAllocator`], is available with
//! the `std` feature enabled.

extern crate alloc;

use core::ffi::c_void;
use core::marker::PhantomData;
use core::ptr::NonNull;
use core::slice;

use alloc::alloc::AllocError;

/// Number of levels in the tree, including the root and the leaves.
pub const HEIGHT: usize = 4;

/// Total number of nodes in a complete binary tree of `HEIGHT` levels.
pub const NUM_NODES: usize = (1 << HEIGHT) - 1;

/// Number of leaf nodes in a complete binary tree of `HEIGHT` levels.
pub const NUM_LEAF_NODES: usize = 1 << (HEIGHT - 1);

/// A single node of a [`MerkleTree`].
///
/// `left` and `right` are `None` for leaves and for nodes that have not yet
/// been linked into a tree (e.g. immediately after allocation). `parent` is
/// `None` for the root and, likewise, for nodes not yet linked into a tree.
pub struct MerkleTreeNode {
    left: Option<MerkleTreeNodePtr>,
    right: Option<MerkleTreeNodePtr>,
    parent: Option<MerkleTreeNodePtr>,
    hash: [u8; 32],
}

type MerkleTreeNodePtr = NonNull<MerkleTreeNode>;

/// Backing storage for [`MerkleTreeNode`]s.
///
/// Implementations only need to hand out and reclaim individual nodes;
/// [`MerkleTree`] is responsible for all tree structure and traversal.
pub trait MerkleTreeNodeAllocator {
    /// Allocate a single node. Its `left`/`right`/`parent` fields may hold
    /// arbitrary values; the caller is responsible for initializing them.
    fn allocate() -> Result<NonNull<MerkleTreeNode>, AllocError>;

    /// Return a node previously handed out by [`allocate`](Self::allocate)
    /// to this allocator.
    ///
    /// # Safety
    ///
    /// `node` must have been returned by this allocator's `allocate` and not
    /// yet deallocated.
    unsafe fn deallocate(node: NonNull<MerkleTreeNode>);
}

/// A Merkle tree covering the region `[ptr, ptr + size)`, with nodes
/// allocated through `A`.
pub struct MerkleTree<A: MerkleTreeNodeAllocator> {
    ptr: NonNull<c_void>,
    size: usize,
    root: Option<MerkleTreeNodePtr>,
    phantom: PhantomData<A>,
}

impl<A: MerkleTreeNodeAllocator> MerkleTree<A> {
    /// # Panics
    ///
    /// Panics if `ptr` or `size` is not a power of two: both are required
    /// so that [`leaf_node`](Self::leaf_node) can subdivide the region
    /// evenly at every level.
    fn validate_ptr_size(ptr: NonNull<c_void>, size: usize) {
        if !ptr.addr().is_power_of_two() {
            panic!("Start address ({:p}) must be power of two!", ptr);
        }

        if !size.is_power_of_two() {
            panic!("Size must be power of two!");
        }
    }

    /// Create a tree over `[ptr, ptr + size)` with no nodes allocated.
    ///
    /// # Panics
    ///
    /// Panics if `ptr` or `size` is not a power of two.
    pub fn empty(ptr: NonNull<c_void>, size: usize) -> Self {
        Self::validate_ptr_size(ptr, size);

        Self {
            ptr,
            size,
            root: None,
            phantom: PhantomData,
        }
    }

    /// Create a tree over `[ptr, ptr + size)`, allocating and linking all
    /// `NUM_NODES` nodes up front.
    ///
    /// # Panics
    ///
    /// Panics if `ptr` or `size` is not a power of two, or if `A::allocate`
    /// fails to provide any of the `NUM_NODES` required nodes.
    pub fn constructed(ptr: NonNull<c_void>, size: usize) -> Self {
        Self::validate_ptr_size(ptr, size);

        // Allocate nodes
        let mut nodes = [None; NUM_NODES];
        for i in 0..NUM_NODES {
            let mut node = match A::allocate() {
                Ok(node) => node,
                Err(error) => panic!("Unable to allocate MerkleTreeNode: {}", error),
            };

            unsafe {
                let node = node.as_mut();
                node.left = None;
                node.right = None;
                node.parent = None;
            }

            nodes[i] = Some(node);
        }

        // Construct tree: link every internal node (indices before the
        // first leaf) to its children at 2i + 1 and 2i + 2, and each of
        // those children back to their parent.
        for i in 0..(1 << (HEIGHT - 1)) - 1 {
            unsafe {
                let mut node = nodes[i].unwrap();
                node.as_mut().left = nodes[2 * i + 1];
                node.as_mut().right = nodes[2 * i + 2];

                nodes[2 * i + 1].unwrap().as_mut().parent = Some(node);
                nodes[2 * i + 2].unwrap().as_mut().parent = Some(node);
            }
        }

        Self {
            ptr,
            size,
            root: nodes[0],
            phantom: PhantomData,
        }
    }

    /// Find the single leaf node that fully contains `[ptr, ptr + size)`, or
    /// `None` if that range is not entirely covered by one leaf (it falls
    /// outside this tree's region, straddles two leaves, or overflows), or
    /// the path to its leaf is not fully linked (e.g. an
    /// [`empty`](Self::empty) tree).
    pub fn leaf_node(&self, ptr: NonNull<u8>, size: usize) -> Option<MerkleTreeNodePtr> {
        let ptr = usize::from(ptr.addr());
        let mut mem_start = usize::from(self.ptr.addr());
        let mut mem_size = self.size;

        if ptr < mem_start {
            return None;
        }

        if ptr >= mem_start + mem_size {
            return None;
        }

        let mut drag = self.root;
        for _ in 0..HEIGHT - 1 {
            match drag {
                Some(node) => {
                    mem_size /= 2;
                    drag = match ptr >= mem_start + mem_size {
                        true => {
                            mem_start += mem_size;
                            unsafe { node.as_ref().right }
                        }
                        false => unsafe { node.as_ref().left },
                    };
                }
                None => return None,
            };
        }

        if ptr + size < mem_start {
            return None;
        }

        if ptr + size > mem_start + mem_size {
            return None;
        }

        drag
    }

    /// Hash the `size` bytes at `ptr` with BLAKE3 and record the result as
    /// the hash of the leaf covering that range, then recompute the hash of
    /// every ancestor up to the root.
    ///
    /// Each internal node's hash is `blake3(left.hash || right.hash)` (a
    /// child contributes nothing if it is absent, which only happens for a
    /// not-yet-linked tree). This makes every node's hash a pure function of
    /// the memory contents in its subtree at the time of the last `rehash`
    /// call covering each leaf.
    ///
    /// # Safety
    ///
    /// `[ptr, ptr + size)` must be valid for reads for the duration of this
    /// call (see [`slice::from_raw_parts`]).
    ///
    /// # Panics
    ///
    /// Panics if `[ptr, ptr + size)` is not fully contained in a single leaf
    /// of this tree, or that leaf is not yet linked (see
    /// [`leaf_node`](Self::leaf_node)).
    pub unsafe fn rehash(&mut self, ptr: NonNull<u8>, size: usize) {
        // Find associated leaf node
        let mut node = match self.leaf_node(ptr, size) {
            Some(node) => node,
            None => panic!("Unable to find leaf node for address ({:p})", ptr),
        };

        // Calculate hash
        let mut hash = blake3::Hasher::new();
        unsafe { hash.update(slice::from_raw_parts(ptr.as_ptr(), size)) };
        let hash = hash.finalize();

        // Update hash of leaf node
        let mut drag = unsafe {
            let node = node.as_mut();
            node.hash.copy_from_slice(hash.as_bytes());
            node.parent
        };

        // Continue with parents
        while let Some(mut node) = drag {
            unsafe {
                let node = node.as_mut();
                let mut hasher = blake3::Hasher::new();

                // Process left child
                if let Some(child) = node.left {
                    hasher.update(child.as_ref().hash.as_slice());
                }

                // Process right child
                if let Some(child) = node.right {
                    hasher.update(child.as_ref().hash.as_slice());
                }

                // Update hash
                let hash = hasher.finalize();
                node.hash.copy_from_slice(hash.as_bytes());

                // Continue with parent
                drag = node.parent;
            }
        }
    }

    /// Hash the `size` bytes at `ptr` with BLAKE3 and check that it matches
    /// the cached hash of the leaf covering that range, and that every
    /// ancestor's cached hash up to the root is consistent with its children
    /// (i.e. still equals `blake3(left.hash || right.hash)`).
    ///
    /// This recomputes the leaf hash from the current memory contents but
    /// does not update anything: it catches both memory that has changed
    /// since the last [`rehash`](Self::rehash) of this leaf, and any node's
    /// cached hash having been corrupted directly.
    ///
    /// # Safety
    ///
    /// `[ptr, ptr + size)` must be valid for reads for the duration of this
    /// call (see [`slice::from_raw_parts`]).
    ///
    /// # Panics
    ///
    /// Panics if `[ptr, ptr + size)` is not fully contained in a single leaf
    /// of this tree, or that leaf is not yet linked (see
    /// [`leaf_node`](Self::leaf_node)).
    pub unsafe fn validate(&self, ptr: NonNull<u8>, size: usize) -> bool {
        // Find associated leaf node
        let node = match self.leaf_node(ptr, size) {
            Some(node) => node,
            None => panic!("Unable to find leaf node for address ({:p})", ptr),
        };

        // Calculate hash
        let mut hash = blake3::Hasher::new();
        unsafe { hash.update(slice::from_raw_parts(ptr.as_ptr(), size)) };
        let hash = hash.finalize();

        // Check leaf node
        let mut drag = unsafe {
            let node = node.as_ref();

            // Error: hashes didn't match...
            if node.hash != *hash.as_bytes() {
                return false;
            }

            node.parent
        };

        // Continue with parents
        while let Some(node) = drag {
            unsafe {
                let node = node.as_ref();
                let mut hasher = blake3::Hasher::new();

                // Process left child
                if let Some(child) = node.left {
                    hasher.update(child.as_ref().hash.as_slice());
                }

                // Process right child
                if let Some(child) = node.right {
                    hasher.update(child.as_ref().hash.as_slice());
                }

                let hash = hasher.finalize();

                // Error: hashes didn't match...
                if node.hash != *hash.as_bytes() {
                    return false;
                }

                // Continue with parent
                drag = node.parent;
            }
        }

        true
    }
}

/// A [`MerkleTreeNodeAllocator`] that allocates each node individually via
/// `std::boxed::Box`.
///
/// Requires the `std` feature: it is only meant as a convenient allocator
/// for hosted use (e.g. tests), not for the `no_std` targets this crate is
/// otherwise built for.
#[cfg(feature = "std")]
mod imp {
    extern crate std;

    use std::alloc::AllocError;
    use std::boxed::Box;

    use core::ptr::NonNull;

    use crate::merkle_tree::{MerkleTreeNode, MerkleTreeNodeAllocator};

    /// Allocates and frees [`MerkleTreeNode`]s one at a time via `Box`.
    pub struct NodeAllocator;

    impl MerkleTreeNodeAllocator for NodeAllocator {
        /// Box a fresh node with `left`/`right`/`parent` all `None`.
        fn allocate() -> Result<NonNull<MerkleTreeNode>, AllocError> {
            let node = Box::new(MerkleTreeNode {
                left: None,
                right: None,
                parent: None,
                hash: [0; 32],
            });
            Ok(NonNull::from(Box::leak(node)))
        }

        /// Reclaim `node` by reconstructing and dropping the `Box` that
        /// [`allocate`](Self::allocate) leaked it from.
        unsafe fn deallocate(node: NonNull<MerkleTreeNode>) {
            unsafe {
                drop(Box::from_raw(node.as_ptr()));
            }
        }
    }
}

#[cfg(feature = "std")]
pub use imp::*;

#[cfg(test)]
mod tests {
    use super::*;

    /// An allocator whose methods must never be called; used by tests that
    /// only exercise `empty`/`leaf_node`, neither of which ever touches `A`.
    struct NullAllocator;

    impl MerkleTreeNodeAllocator for NullAllocator {
        fn allocate() -> Result<NonNull<MerkleTreeNode>, AllocError> {
            unreachable!("NullAllocator never allocates");
        }

        unsafe fn deallocate(_node: NonNull<MerkleTreeNode>) {
            unreachable!("NullAllocator never deallocates");
        }
    }

    /// Turn a bare address into the `NonNull<c_void>` that `MerkleTree`'s
    /// constructors expect; the address is never dereferenced, only
    /// compared.
    fn ptr(addr: usize) -> NonNull<c_void> {
        NonNull::new(addr as *mut c_void).unwrap()
    }

    /// Turn a bare address into the `NonNull<u8>` that `leaf_node` expects.
    /// Only used with addresses that are either never dereferenced
    /// (`leaf_node` itself only compares addresses) or that are backed by
    /// real memory from a [`MappedRegion`].
    fn byte_ptr(addr: usize) -> NonNull<u8> {
        NonNull::new(addr as *mut u8).unwrap()
    }

    #[test]
    #[should_panic(expected = "power of two")]
    fn empty_panics_on_non_power_of_two_size() {
        MerkleTree::<NullAllocator>::empty(ptr(0x1000), 3);
    }

    #[test]
    #[should_panic(expected = "power of two")]
    fn empty_panics_on_non_power_of_two_ptr() {
        MerkleTree::<NullAllocator>::empty(ptr(0x1001), 0x100);
    }

    #[test]
    fn leaf_node_on_empty_tree_is_always_none() {
        let tree = MerkleTree::<NullAllocator>::empty(ptr(0x1000), 0x100);

        // In range, but no nodes have been linked yet.
        assert!(tree.leaf_node(byte_ptr(0x1000), 1).is_none());
        assert!(tree.leaf_node(byte_ptr(0x1050), 1).is_none());
    }

    #[test]
    fn leaf_node_rejects_addresses_outside_the_region() {
        let tree = MerkleTree::<NullAllocator>::empty(ptr(0x1000), 0x100);

        assert!(tree.leaf_node(byte_ptr(0x0fff), 1).is_none());
        assert!(tree.leaf_node(byte_ptr(0x1100), 1).is_none());
    }

    #[cfg(feature = "std")]
    #[test]
    fn node_allocator_round_trips_a_single_node() {
        let node = NodeAllocator::allocate().unwrap();
        unsafe {
            assert!(node.as_ref().left.is_none());
            assert!(node.as_ref().right.is_none());
            assert!(node.as_ref().parent.is_none());
            NodeAllocator::deallocate(node);
        }
    }

    #[cfg(feature = "std")]
    #[test]
    fn constructed_maps_every_leaf_to_a_distinct_node() {
        const START: usize = 0x1000;
        const SIZE: usize = 0x100;
        const LEAVES: usize = 1 << (HEIGHT - 1);
        const LEAF_SIZE: usize = SIZE / LEAVES;

        let tree = MerkleTree::<NodeAllocator>::constructed(ptr(START), SIZE);

        let mut leaves = [None; LEAVES];
        for (i, leaf) in leaves.iter_mut().enumerate() {
            *leaf = tree.leaf_node(byte_ptr(START + i * LEAF_SIZE), 1);
        }

        for i in 0..LEAVES {
            assert!(
                leaves[i].is_some(),
                "every in-range address must map to a leaf"
            );
            for j in (i + 1)..LEAVES {
                assert_ne!(
                    leaves[i], leaves[j],
                    "leaves {i} and {j} must be distinct nodes"
                );
            }
        }
    }

    #[cfg(feature = "std")]
    #[test]
    fn constructed_maps_addresses_within_a_leaf_to_the_same_node() {
        const START: usize = 0x1000;
        const SIZE: usize = 0x100;
        const LEAF_SIZE: usize = SIZE / (1 << (HEIGHT - 1));

        let tree = MerkleTree::<NodeAllocator>::constructed(ptr(START), SIZE);

        let start_leaf = tree.leaf_node(byte_ptr(START), 1);
        let mid_leaf = tree.leaf_node(byte_ptr(START + LEAF_SIZE / 2), 1);
        let end_leaf = tree.leaf_node(byte_ptr(START + LEAF_SIZE - 1), 1);

        assert!(start_leaf.is_some());
        assert_eq!(start_leaf, mid_leaf);
        assert_eq!(start_leaf, end_leaf);
    }

    #[cfg(feature = "std")]
    #[test]
    fn leaf_node_rejects_a_range_that_spills_into_the_next_leaf() {
        const START: usize = 0x1000;
        const SIZE: usize = 0x100;
        const LEAF_SIZE: usize = SIZE / (1 << (HEIGHT - 1));

        let tree = MerkleTree::<NodeAllocator>::constructed(ptr(START), SIZE);

        // The range fits if it stays within one leaf...
        assert!(tree.leaf_node(byte_ptr(START), LEAF_SIZE).is_some());
        // ...but not if it spills one byte into the next leaf.
        assert!(tree.leaf_node(byte_ptr(START), LEAF_SIZE + 1).is_none());
    }

    #[cfg(feature = "std")]
    #[test]
    fn constructed_links_every_node_back_to_its_parent() {
        const START: usize = 0x1000;
        const SIZE: usize = 0x100;

        let tree = MerkleTree::<NodeAllocator>::constructed(ptr(START), SIZE);

        let root = tree.root.unwrap();
        assert!(unsafe { root.as_ref().parent.is_none() });

        // Walk every internal node and check that both of its children
        // point back to it.
        let mut queue = [None; NUM_NODES];
        queue[0] = Some(root);
        let mut len = 1;
        let mut idx = 0;
        while idx < len {
            let node = queue[idx].unwrap();
            idx += 1;

            unsafe {
                for child in [node.as_ref().left, node.as_ref().right] {
                    if let Some(child) = child {
                        assert_eq!(child.as_ref().parent, Some(node));
                        queue[len] = Some(child);
                        len += 1;
                    }
                }
            }
        }

        // Every non-root node must have been reached.
        assert_eq!(len, NUM_NODES);
    }

    #[cfg(feature = "std")]
    #[test]
    fn constructed_tree_leaf_node_rejects_out_of_range_addresses() {
        const START: usize = 0x1000;
        const SIZE: usize = 0x100;

        let tree = MerkleTree::<NodeAllocator>::constructed(ptr(START), SIZE);

        assert!(tree.leaf_node(byte_ptr(START - 1), 1).is_none());
        assert!(tree.leaf_node(byte_ptr(START + SIZE), 1).is_none());
    }

    /// A single anonymous, zero-filled `mmap`'d region at a fixed,
    /// power-of-two address, so `rehash`/`validate` tests have real,
    /// dereferenceable memory to hash (`MerkleTree::empty`/`constructed`
    /// also require the tree's base address to be a power of two).
    ///
    /// Each test using this must pick its own address, spaced far enough
    /// apart (e.g. successive powers of two) that concurrently-running
    /// tests never map overlapping regions.
    #[cfg(feature = "std")]
    struct MappedRegion {
        start: NonNull<c_void>,
        size: usize,
    }

    #[cfg(feature = "std")]
    impl MappedRegion {
        fn new(addr: usize, size: usize) -> Self {
            use rustix::mm::{MapFlags, ProtFlags, mmap_anonymous};

            let start = unsafe {
                mmap_anonymous(
                    addr as *mut c_void,
                    size,
                    ProtFlags::READ | ProtFlags::WRITE,
                    MapFlags::PRIVATE | MapFlags::FIXED_NOREPLACE,
                )
            }
            .unwrap_or_else(|error| panic!("Unable to map {size:#x} byte(s) at {addr:#x}: {error}"));

            Self {
                start: NonNull::new(start).unwrap(),
                size,
            }
        }

        /// The region's base, as the `NonNull<c_void>` `MerkleTree`'s
        /// constructors expect.
        fn base(&self) -> NonNull<c_void> {
            self.start
        }

        /// A `NonNull<u8>` at `offset` bytes into the region.
        fn byte_ptr(&self, offset: usize) -> NonNull<u8> {
            assert!(offset <= self.size);
            unsafe { NonNull::new_unchecked(self.start.as_ptr().add(offset).cast()) }
        }

        /// Overwrite the `len` bytes at `offset` with `value`, to give
        /// different leaves distinguishable content.
        fn fill(&self, offset: usize, len: usize, value: u8) {
            assert!(offset + len <= self.size);
            unsafe { self.byte_ptr(offset).as_ptr().write_bytes(value, len) };
        }
    }

    #[cfg(feature = "std")]
    impl Drop for MappedRegion {
        fn drop(&mut self) {
            let _ = unsafe { rustix::mm::munmap(self.start.as_ptr(), self.size) };
        }
    }

    #[cfg(feature = "std")]
    #[test]
    fn rehash_sets_the_leaf_hash() {
        const START: usize = 1 << 28;
        const SIZE: usize = 0x1000;
        const LEAF_SIZE: usize = SIZE / NUM_LEAF_NODES;

        let region = MappedRegion::new(START, SIZE);
        let mut tree = MerkleTree::<NodeAllocator>::constructed(region.base(), SIZE);

        unsafe { tree.rehash(region.byte_ptr(0), LEAF_SIZE) };

        let expected =
            *blake3::hash(unsafe { slice::from_raw_parts(region.byte_ptr(0).as_ptr(), LEAF_SIZE) })
                .as_bytes();

        let leaf = tree.leaf_node(region.byte_ptr(0), LEAF_SIZE).unwrap();
        assert_eq!(unsafe { leaf.as_ref().hash }, expected);
    }

    #[cfg(feature = "std")]
    #[test]
    fn rehash_computes_ancestor_hashes_as_blake3_of_children() {
        const START: usize = 1 << 29;
        const SIZE: usize = 0x1000;
        const LEAF_SIZE: usize = SIZE / NUM_LEAF_NODES;

        let region = MappedRegion::new(START, SIZE);
        let mut tree = MerkleTree::<NodeAllocator>::constructed(region.base(), SIZE);

        // Give the two leaves sharing the lowest-level parent distinct
        // content, then rehash both.
        region.fill(0, LEAF_SIZE, 0xaa);
        region.fill(LEAF_SIZE, LEAF_SIZE, 0xbb);
        unsafe {
            tree.rehash(region.byte_ptr(0), LEAF_SIZE);
            tree.rehash(region.byte_ptr(LEAF_SIZE), LEAF_SIZE);
        }

        let left_hash = *blake3::hash(unsafe {
            slice::from_raw_parts(region.byte_ptr(0).as_ptr(), LEAF_SIZE)
        })
        .as_bytes();
        let right_hash = *blake3::hash(unsafe {
            slice::from_raw_parts(region.byte_ptr(LEAF_SIZE).as_ptr(), LEAF_SIZE)
        })
        .as_bytes();

        let mut hasher = blake3::Hasher::new();
        hasher.update(&left_hash);
        hasher.update(&right_hash);
        let expected = *hasher.finalize().as_bytes();

        let parent =
            unsafe { tree.leaf_node(region.byte_ptr(0), LEAF_SIZE).unwrap().as_ref().parent }
                .unwrap();
        assert_eq!(unsafe { parent.as_ref().hash }, expected);
    }

    #[cfg(feature = "std")]
    #[test]
    fn rehash_leaves_unrelated_subtree_untouched() {
        const START: usize = 1 << 30;
        const SIZE: usize = 0x1000;
        const LEAF_SIZE: usize = SIZE / NUM_LEAF_NODES;

        let region = MappedRegion::new(START, SIZE);
        let mut tree = MerkleTree::<NodeAllocator>::constructed(region.base(), SIZE);

        unsafe { tree.rehash(region.byte_ptr(0), LEAF_SIZE) };

        // A leaf outside the rehashed leaf's path must still hold its
        // initial, all-zero hash.
        let untouched = tree
            .leaf_node(region.byte_ptr(SIZE - LEAF_SIZE), LEAF_SIZE)
            .unwrap();
        assert_eq!(unsafe { untouched.as_ref().hash }, [0u8; 32]);
    }

    #[cfg(feature = "std")]
    #[test]
    #[should_panic(expected = "Unable to find leaf node")]
    fn rehash_panics_for_out_of_range_address() {
        const START: usize = 0x1000;
        const SIZE: usize = 0x100;

        let mut tree = MerkleTree::<NodeAllocator>::constructed(ptr(START), SIZE);

        // Out of range: `leaf_node` returns `None` before this address is
        // ever dereferenced, so no real backing memory is needed.
        unsafe { tree.rehash(byte_ptr(START + SIZE), 1) };
    }

    #[cfg(feature = "std")]
    #[test]
    fn validate_is_false_on_a_freshly_constructed_tree() {
        const START: usize = 1 << 31;
        const SIZE: usize = 0x1000;
        const LEAF_SIZE: usize = SIZE / NUM_LEAF_NODES;

        let region = MappedRegion::new(START, SIZE);
        let tree = MerkleTree::<NodeAllocator>::constructed(region.base(), SIZE);

        // Every node starts out with an all-zero cached hash, which is not
        // the blake3 hash of the (zero-filled) memory it covers, so
        // validation must fail until something has actually been rehashed.
        assert!(!unsafe { tree.validate(region.byte_ptr(0), LEAF_SIZE) });
    }

    #[cfg(feature = "std")]
    #[test]
    fn validate_is_true_after_a_matching_rehash() {
        const START: usize = 1 << 32;
        const SIZE: usize = 0x1000;
        const LEAF_SIZE: usize = SIZE / NUM_LEAF_NODES;

        let region = MappedRegion::new(START, SIZE);
        let mut tree = MerkleTree::<NodeAllocator>::constructed(region.base(), SIZE);

        unsafe {
            tree.rehash(region.byte_ptr(0), LEAF_SIZE);
            assert!(tree.validate(region.byte_ptr(0), LEAF_SIZE));
        }
    }

    #[cfg(feature = "std")]
    #[test]
    fn validate_is_false_when_memory_changes_after_rehash() {
        const START: usize = 1 << 33;
        const SIZE: usize = 0x1000;
        const LEAF_SIZE: usize = SIZE / NUM_LEAF_NODES;

        let region = MappedRegion::new(START, SIZE);
        let mut tree = MerkleTree::<NodeAllocator>::constructed(region.base(), SIZE);

        unsafe { tree.rehash(region.byte_ptr(0), LEAF_SIZE) };

        // Simulate the covered memory being modified after the leaf was
        // last rehashed.
        region.fill(0, LEAF_SIZE, 0xff);

        assert!(!unsafe { tree.validate(region.byte_ptr(0), LEAF_SIZE) });
    }

    #[cfg(feature = "std")]
    #[test]
    fn validate_is_false_when_an_ancestor_hash_was_tampered_with() {
        const START: usize = 1 << 34;
        const SIZE: usize = 0x1000;
        const LEAF_SIZE: usize = SIZE / NUM_LEAF_NODES;

        let region = MappedRegion::new(START, SIZE);
        let mut tree = MerkleTree::<NodeAllocator>::constructed(region.base(), SIZE);

        unsafe {
            tree.rehash(region.byte_ptr(0), LEAF_SIZE);
            assert!(tree.validate(region.byte_ptr(0), LEAF_SIZE));
        }

        // Corrupt the parent's cached hash directly, bypassing rehash.
        let mut parent =
            unsafe { tree.leaf_node(region.byte_ptr(0), LEAF_SIZE).unwrap().as_ref().parent }
                .unwrap();
        unsafe {
            parent.as_mut().hash = [0xffu8; 32];
        }

        // The leaf's memory still matches its cached hash, but the
        // tampered ancestor no longer agrees with its children.
        assert!(!unsafe { tree.validate(region.byte_ptr(0), LEAF_SIZE) });
    }

    #[cfg(feature = "std")]
    #[test]
    #[should_panic(expected = "Unable to find leaf node")]
    fn validate_panics_for_out_of_range_address() {
        const START: usize = 0x1000;
        const SIZE: usize = 0x100;

        let tree = MerkleTree::<NodeAllocator>::constructed(ptr(START), SIZE);

        // Out of range: `leaf_node` returns `None` before this address is
        // ever dereferenced, so no real backing memory is needed.
        unsafe { tree.validate(byte_ptr(START + SIZE), 1) };
    }
}
