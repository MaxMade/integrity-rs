extern crate alloc;

use core::ffi::c_void;
use core::mem::MaybeUninit;
use core::ptr::NonNull;

use crate::{
    buddy_allocator::BuddyAllocator,
    memory::MemoryManagement,
    merkle_tree::{self, MerkleTree, MerkleTreeNodeAllocator},
};

use alloc::alloc::Layout;
use alloc::boxed::Box;

pub const TOTAL_MEM: usize = 1 * 1024 * 1024 * 1024;

pub const MEM_PER_SUBTREE: usize = 4 * 1024 * 1024;

pub const NUM_SUBTREES: usize = TOTAL_MEM / MEM_PER_SUBTREE;

pub const MEM_PER_SUBTREE_LEAF: usize = MEM_PER_SUBTREE / merkle_tree::NUM_LEAF_NODES;

pub struct MountableMerkleTree<MTNA: MerkleTreeNodeAllocator, MM: MemoryManagement<MEM_PER_SUBTREE>>
{
    root_tree: MerkleTree<MTNA>,

    sub_trees: Box<[MerkleTree<MTNA>]>,

    memory_management: MM,

    memory_allocators: Box<[BuddyAllocator<16>]>,
}

impl<MTNA: MerkleTreeNodeAllocator, MM: MemoryManagement<MEM_PER_SUBTREE>>
    MountableMerkleTree<MTNA, MM>
{
    /// The layout `sub_trees` is actually allocated with: `MerkleTree`
    /// requires its `size` to be a power of two, but
    /// `NUM_SUBTREES * size_of::<MerkleTree<MTNA>>()` generally isn't, so
    /// the backing allocation is padded up to the next one (the padding is
    /// simply unused by `root_tree`).
    ///
    /// `sub_trees`'s `Box` cannot be allowed to free itself automatically:
    /// its slice length (`NUM_SUBTREES`) implies the *unpadded* layout,
    /// which would mismatch the layout actually passed to `alloc` here.
    /// [`drop`](Self::drop) frees it manually with this same layout instead.
    fn sub_trees_layout() -> Layout {
        let layout = Layout::new::<[MerkleTree<MTNA>; NUM_SUBTREES]>();
        Layout::from_size_align(layout.size().next_power_of_two(), layout.align()).unwrap()
    }

    pub fn new() -> Self {
        let memory_management = MM::new(TOTAL_MEM);

        let sub_trees_layout = Self::sub_trees_layout();
        let sub_trees_size = sub_trees_layout.size();

        let sub_trees: Box<[MerkleTree<MTNA>]> = unsafe {
            let ptr = alloc::alloc::alloc(sub_trees_layout);
            let mut sub_trees: Box<[MaybeUninit<MerkleTree<MTNA>>; NUM_SUBTREES]> =
                Box::from_raw(ptr as _);

            for i in 0..NUM_SUBTREES {
                let ptr = memory_management.start().byte_add(i * MEM_PER_SUBTREE);
                sub_trees[i].write(MerkleTree::empty(ptr, MEM_PER_SUBTREE));
            }

            Box::from_raw(Box::into_raw(sub_trees) as *mut [MerkleTree<MTNA>; NUM_SUBTREES])
        };

        // root_tree protects the `sub_trees` array itself: whenever a
        // sub-tree's hash changes, re-hashing the corresponding slice of
        // `sub_trees` here attests to every mounted sub-tree, and
        // therefore to every byte of memory any of them cover.
        let root_tree = MerkleTree::constructed(
            NonNull::new(sub_trees.as_ptr() as *mut c_void).unwrap(),
            sub_trees_size,
        );

        let memory_allocators: Box<[BuddyAllocator<16>]> = unsafe {
            let layout = Layout::new::<[BuddyAllocator<16>; NUM_SUBTREES]>();
            let ptr = alloc::alloc::alloc(layout);
            let mut memory_allocators: Box<[MaybeUninit<BuddyAllocator<16>>; NUM_SUBTREES]> =
                Box::from_raw(ptr as _);

            for i in 0..NUM_SUBTREES {
                let mut memory_allocator = BuddyAllocator::new();
                let ptr = memory_management.start().byte_add(i * MEM_PER_SUBTREE);
                memory_allocator.fill(ptr.cast(), MEM_PER_SUBTREE);

                memory_allocators[i].write(memory_allocator);
            }

            Box::from_raw(Box::into_raw(memory_allocators) as *mut [BuddyAllocator<16>; NUM_SUBTREES])
        };

        Self {
            root_tree,
            sub_trees,
            memory_management,
            memory_allocators,
        }
    }
}

impl<MTNA: MerkleTreeNodeAllocator, MM: MemoryManagement<MEM_PER_SUBTREE>> Drop
    for MountableMerkleTree<MTNA, MM>
{
    fn drop(&mut self) {
        // Swap in an empty (non-allocating) Box first: the compiler-
        // generated per-field drop still runs on `self.sub_trees` after
        // this function returns, and it must not see the real allocation,
        // since we are about to free that manually below (see
        // `sub_trees_layout`'s doc comment for why it can't just be
        // dropped normally).
        let empty: Box<[MerkleTree<MTNA>]> = Box::new([]);
        let sub_trees = core::mem::replace(&mut self.sub_trees, empty);

        let ptr = Box::into_raw(sub_trees) as *mut u8;
        unsafe {
            // Run every sub-tree's own `Drop` (freeing any node chain a
            // mounted sub-tree owns) before releasing the backing memory.
            core::ptr::drop_in_place(ptr as *mut [MerkleTree<MTNA>; NUM_SUBTREES]);
            alloc::alloc::dealloc(ptr, Self::sub_trees_layout());
        }

        // root_tree and memory_allocators need no special handling here:
        // root_tree is a plain field (its own `Drop` runs via the normal
        // per-field glue), and memory_allocators' Box was allocated with
        // exactly the layout its slice length implies.
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::merkle_tree::NodeAllocator;

    /// A [`MemoryManagement`] backed by a plain heap allocation, for tests:
    /// the real, `mmap`-backed [`crate::memory::Memory`] requires huge
    /// pages to be reserved on the host, which isn't guaranteed here.
    struct HeapMemory(Box<[u8]>);

    impl<const ALLOC_SIZE: usize> MemoryManagement<ALLOC_SIZE> for HeapMemory {
        fn new(total_size: usize) -> Self {
            Self(alloc::vec![0u8; total_size].into_boxed_slice())
        }

        fn start(&self) -> NonNull<c_void> {
            NonNull::new(self.0.as_ptr() as *mut c_void).unwrap()
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
}
