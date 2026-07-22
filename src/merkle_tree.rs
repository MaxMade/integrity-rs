//! An address-indexed, fixed-height Merkle tree.
//!
//! [`MerkleTree`] covers a single power-of-two-sized, power-of-two-aligned
//! region `[ptr, ptr + size)` with a complete binary tree of `HEIGHT` levels.
//! The tree is stored as a flat array of `NUM_NODES = 2^HEIGHT - 1` nodes
//! (node `i`'s children live at `2i + 1` and `2i + 2`), so it has exactly
//! `2^(HEIGHT - 1)` leaves, each covering an equal-sized slice of the region.
//! [`leaf_node`](MerkleTree::leaf_node) maps an address back to its leaf by
//! binary-searching that address range one level at a time: at each level
//! the current window is halved, and the address falls in either the lower
//! half (left child, same start) or the upper half (right child, start moved
//! up by the halved size).
//!
//! Node storage is decoupled from the tree logic via the
//! [`MerkleTreeNodeAllocator`] trait, so callers can back the tree with
//! whatever allocation strategy fits (e.g. a [`BuddyAllocator`](crate::buddy_allocator::BuddyAllocator)).

extern crate alloc;

use core::ffi::c_void;
use core::marker::PhantomData;
use core::ptr::NonNull;

use alloc::alloc::AllocError;

/// Number of levels in the tree, including the root and the leaves.
const HEIGHT: usize = 4;

/// Total number of nodes in a complete binary tree of `HEIGHT` levels.
const NUM_NODES: usize = (1 << HEIGHT) - 1;

/// A single node of a [`MerkleTree`].
///
/// `left` and `right` are `None` for leaves and for nodes that have not yet
/// been linked into a tree (e.g. immediately after allocation).
pub struct MerkleTreeNode {
    left: Option<MerkleTreeNodePtr>,
    right: Option<MerkleTreeNodePtr>,
}

type MerkleTreeNodePtr = NonNull<MerkleTreeNode>;

/// Backing storage for [`MerkleTreeNode`]s.
///
/// Implementations only need to hand out and reclaim individual nodes;
/// [`MerkleTree`] is responsible for all tree structure and traversal.
pub trait MerkleTreeNodeAllocator {
    /// Allocate a single node. Its `left`/`right` fields may hold arbitrary
    /// values; the caller is responsible for initializing them.
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
            }

            nodes[i] = Some(node);
        }

        // Construct tree: link every internal node (indices before the
        // first leaf) to its children at 2i + 1 and 2i + 2.
        for i in 0..(1 << (HEIGHT - 1)) - 1 {
            unsafe {
                let node = nodes[i].unwrap().as_mut();
                node.left = nodes[2 * i + 1];
                node.right = nodes[2 * i + 2];
            }
        }

        Self {
            ptr,
            size,
            root: nodes[0],
            phantom: PhantomData,
        }
    }

    /// Find the leaf node covering `ptr`, or `None` if `ptr` falls outside
    /// this tree's region or the path to its leaf is not fully linked
    /// (e.g. an [`empty`](Self::empty) tree).
    pub fn leaf_node(&mut self, ptr: NonNull<c_void>) -> Option<MerkleTreeNodePtr> {
        let ptr = usize::from(ptr.addr());
        let mut start = usize::from(self.ptr.addr());
        let mut size = self.size;

        if ptr < start {
            return None;
        }

        if ptr >= start + size {
            return None;
        }

        let mut drag = self.root;
        for _ in 0..HEIGHT - 1 {
            match drag {
                Some(node) => {
                    size /= 2;
                    drag = match ptr >= start + size {
                        true => {
                            start += size;
                            unsafe { node.as_ref().right }
                        }
                        false => unsafe { node.as_ref().left },
                    };
                }
                None => return None,
            };
        }

        drag
    }
}
