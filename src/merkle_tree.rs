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
//! A `Box`-backed implementation, [`NodeAllocator`], is available with
//! the `std` feature enabled.

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
/// been linked into a tree (e.g. immediately after allocation). `parent` is
/// `None` for the root and, likewise, for nodes not yet linked into a tree.
pub struct MerkleTreeNode {
    left: Option<MerkleTreeNodePtr>,
    right: Option<MerkleTreeNodePtr>,
    parent: Option<MerkleTreeNodePtr>,
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

    /// Turn a bare address into the `NonNull<c_void>` this module's API
    /// expects; the address is never dereferenced, only compared.
    fn ptr(addr: usize) -> NonNull<c_void> {
        NonNull::new(addr as *mut c_void).unwrap()
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
        let mut tree = MerkleTree::<NullAllocator>::empty(ptr(0x1000), 0x100);

        // In range, but no nodes have been linked yet.
        assert!(tree.leaf_node(ptr(0x1000)).is_none());
        assert!(tree.leaf_node(ptr(0x1050)).is_none());
    }

    #[test]
    fn leaf_node_rejects_addresses_outside_the_region() {
        let mut tree = MerkleTree::<NullAllocator>::empty(ptr(0x1000), 0x100);

        assert!(tree.leaf_node(ptr(0x0fff)).is_none());
        assert!(tree.leaf_node(ptr(0x1100)).is_none());
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

        let mut tree = MerkleTree::<NodeAllocator>::constructed(ptr(START), SIZE);

        let mut leaves = [None; LEAVES];
        for (i, leaf) in leaves.iter_mut().enumerate() {
            *leaf = tree.leaf_node(ptr(START + i * LEAF_SIZE));
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

        let mut tree = MerkleTree::<NodeAllocator>::constructed(ptr(START), SIZE);

        let start_leaf = tree.leaf_node(ptr(START));
        let mid_leaf = tree.leaf_node(ptr(START + LEAF_SIZE / 2));
        let end_leaf = tree.leaf_node(ptr(START + LEAF_SIZE - 1));

        assert!(start_leaf.is_some());
        assert_eq!(start_leaf, mid_leaf);
        assert_eq!(start_leaf, end_leaf);
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

        let mut tree = MerkleTree::<NodeAllocator>::constructed(ptr(START), SIZE);

        assert!(tree.leaf_node(ptr(START - 1)).is_none());
        assert!(tree.leaf_node(ptr(START + SIZE)).is_none());
    }
}
