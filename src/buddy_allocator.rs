//! A fixed-capacity buddy allocator.
//!
//! Free blocks are tracked with `LEVELS` intrusive, singly-rooted doubly
//! linked free lists (`heads`), one per size class ("bucket"). Bucket `idx`
//! only ever holds blocks of exactly `minimum allocation size << idx` bytes
//! (an architecture-specific minimum, large enough to hold this allocator's
//! own free-list bookkeeping), and every block handed out by this allocator
//! has an address that is a multiple of its own size. That invariant is what
//! lets [`allocate`] split a larger block in half without any extra
//! bookkeeping, and [`deallocate`] find a freed block's buddy with a single
//! XOR of the address against the block size.
//!
//! [`allocate`]: BuddyAllocator::allocate
//! [`deallocate`]: BuddyAllocator::deallocate

extern crate alloc;

use core::{alloc::AllocError, ptr::NonNull};

use alloc::alloc::Layout;

/// Intrusive free-list node, stored inline at the start of every free block.
///
/// This is why the smallest block this allocator can ever hand out is an
/// architecture-specific minimal allocation size (the size of this struct):
/// any smaller and there would be no room to store the links while the
/// block is on a free list.
struct LinkedListNode {
    prev: LinkedListNodePtr,
    next: LinkedListNodePtr,
}

type LinkedListNodePtr = Option<NonNull<LinkedListNode>>;

/// Insert `node` at the front of the list rooted at `head`.
unsafe fn list_push(head: &mut LinkedListNodePtr, mut node: NonNull<LinkedListNode>) {
    unsafe {
        node.as_mut().prev = None;
        node.as_mut().next = *head;
        if let Some(mut old_head) = *head {
            old_head.as_mut().prev = Some(node);
        }
    }
    *head = Some(node);
}

/// Remove and return the node at the front of the list rooted at `head`.
unsafe fn list_pop(head: &mut LinkedListNodePtr) -> LinkedListNodePtr {
    let node = (*head)?;
    unsafe {
        let next = node.as_ref().next;
        *head = next;
        if let Some(mut next) = next {
            next.as_mut().prev = None;
        }
    }
    Some(node)
}

/// Remove `target` from the list rooted at `head` if it is present, returning
/// whether it was found. Only ever follows `next` pointers reached by walking
/// from `head`, so it never trusts `target`'s embedded links before it is
/// confirmed to actually be a member of this list.
unsafe fn list_take(head: &mut LinkedListNodePtr, target: NonNull<LinkedListNode>) -> bool {
    unsafe {
        let mut cur = *head;
        while let Some(node) = cur {
            if node == target {
                let prev = node.as_ref().prev;
                let next = node.as_ref().next;
                match prev {
                    Some(mut p) => p.as_mut().next = next,
                    None => *head = next,
                }
                if let Some(mut n) = next {
                    n.as_mut().prev = prev;
                }
                return true;
            }
            cur = node.as_ref().next;
        }
        false
    }
}

fn prev_power_of_two(x: usize) -> usize {
    1usize << (usize::BITS - 1 - x.leading_zeros())
}

/// A buddy allocator with `LEVELS` size classes, from an
/// architecture-specific minimal allocation size up to that minimum size
/// shifted left by `LEVELS - 1`.
///
/// The allocator itself holds no memory; use [`fill`](Self::fill) to hand it
/// one or more backing regions before calling [`allocate`](Self::allocate).
pub struct BuddyAllocator<const LEVELS: usize> {
    heads: [LinkedListNodePtr; LEVELS]
}

impl<const LEVELS: usize> BuddyAllocator<LEVELS> {
    /// Create an empty allocator with no backing memory.
    ///
    /// # Panics
    ///
    /// Panics if `LEVELS` is not a power of two.
    pub const fn new() -> Self {
        if !LEVELS.is_power_of_two() {
            panic!("BuddyAllocator::LEVELS must be a power of two!");
        }

        Self {
            heads: [None; LEVELS]
        }
    }

    /// Smallest block size, i.e. the size of bucket 0: an architecture-specific
    /// minimal allocation size, large enough to hold this allocator's own
    /// free-list bookkeeping.
    fn base_level() -> u32 {
        size_of::<LinkedListNode>().ilog2()
    }

    /// Block size handed out by bucket `idx`.
    fn idx_to_size(idx: usize) -> usize {
        1usize << (Self::base_level() + idx as u32)
    }

    /// Bucket index for an already power-of-two `size`.
    fn size_to_idx(size: usize) -> usize {
        (size.ilog2().saturating_sub(Self::base_level())) as _
    }

    /// Add the `len` bytes starting at `ptr` to the allocator's free memory.
    ///
    /// `ptr` need not be aligned to any particular boundary and `len` need
    /// not be a power of two: the region is swept into the largest
    /// power-of-two-sized, power-of-two-aligned chunks that fit, each
    /// inserted into its matching bucket (splitting down into the biggest
    /// supported bucket size if a chunk would otherwise be too large). Any
    /// bytes left over at the very end (smaller than the smallest bucket
    /// size) are discarded.
    ///
    /// Can be called multiple times with disjoint regions.
    ///
    /// Caller must ensure `ptr` points to `len` bytes that are valid for
    /// reads and writes, unused by anything else, and that remain valid for
    /// as long as blocks from this region might still be handed out.
    pub fn fill(&mut self, ptr: NonNull<u8>, len: usize) {
        let min_size = Self::idx_to_size(0);
        let end = (ptr.as_ptr() as usize).saturating_add(len);

        // Align ptr and len to power of two: skip forward to the next multiple
        // of the smallest block size; the tail past the last full block is
        // dropped by the loop condition below.
        let mut addr = (ptr.as_ptr() as usize).next_multiple_of(min_size).min(end);

        while addr.saturating_add(min_size) <= end {
            let align_bit = addr.trailing_zeros().min(usize::BITS - 1);
            let align_size = 1usize << align_bit;
            let remaining = prev_power_of_two(end - addr);

            let mut size = align_size.min(remaining);
            let mut idx = Self::size_to_idx(size);

            // Insert into target bucket (and split if bucket index exceeds LEVELS)
            if idx >= LEVELS {
                idx = LEVELS - 1;
                size = Self::idx_to_size(idx);
            }

            let node = unsafe { NonNull::new_unchecked(addr as *mut LinkedListNode) };
            unsafe { list_push(&mut self.heads[idx], node) };

            addr += size;
        }
    }

    /// Bucket index that would hold an allocation matching `layout`,
    /// disregarding any alignment requirement above `layout`'s size.
    ///
    /// Callers must still check the result against `LEVELS`: a value of
    /// `LEVELS` or greater means the layout is too large for this allocator.
    pub fn layout_to_idx(layout: &Layout) -> usize {
        let size = layout.size();

        // Round up to the next power of two
        let size = size.next_power_of_two();
        let level = size.ilog2();

        let base_level = size_of::<LinkedListNode>().ilog2();

        (level.saturating_sub(base_level)) as _
    }


    /// Allocate a block satisfying `layout`.
    ///
    /// # Errors
    ///
    /// Returns [`AllocError`] if `layout` is larger than the biggest bucket
    /// this allocator supports, or if no free block large enough is
    /// currently available.
    pub fn allocate(&mut self, layout: Layout) -> Result<NonNull<[u8]>, AllocError> {
        // Check for zero-sized request
        if layout.size() == 0 {
            // A well-aligned, non-null pointer that must never be dereferenced.
            let dangling = unsafe { NonNull::new_unchecked(layout.align() as *mut u8) };
            return Ok(NonNull::slice_from_raw_parts(dangling, 0));
        }

        // Calculate expected bucket index
        let idx = Self::layout_to_idx(&layout);
        if idx >= LEVELS {
            return Err(AllocError);
        }

        // As long as self.heads[idx] is unavailable/empty, continue with next upper bucket and increase the number of necessary splits
        let mut level = idx;
        while self.heads[level].is_none() {
            level += 1;
            if level >= LEVELS {
                return Err(AllocError);
            }
        }

        // Perform the necessary number of splits and re-insert its buddies
        let block = unsafe { list_pop(&mut self.heads[level]) }
            .expect("bucket was just confirmed non-empty");

        while level > idx {
            level -= 1;
            let buddy_addr = block.as_ptr() as usize + Self::idx_to_size(level);
            let buddy = unsafe { NonNull::new_unchecked(buddy_addr as *mut LinkedListNode) };
            unsafe { list_push(&mut self.heads[level], buddy) };
        }

        let size = Self::idx_to_size(idx);
        Ok(NonNull::slice_from_raw_parts(block.cast::<u8>(), size))
    }

    /// Return a block previously handed out by [`allocate`](Self::allocate)
    /// to its free list, merging it with its buddy at each level for as
    /// long as that buddy is also free.
    ///
    /// # Safety
    ///
    /// `ptr` and `layout` must be exactly what a prior call to `allocate` on
    /// this same allocator returned/was given, not yet deallocated.
    pub unsafe fn deallocate(&mut self, ptr: NonNull<u8>, layout: Layout) {
        // Check for zero-sized request
        if layout.size() == 0 {
            return;
        }

        // Calculate expected bucket index
        let mut idx = Self::layout_to_idx(&layout);
        debug_assert!(idx < LEVELS, "layout was never handed out by this allocator");

        let mut addr = ptr.as_ptr() as usize;

        // As long as buddies are found in other buckets, remove them, and continue
        while idx + 1 < LEVELS {
            let buddy_addr = addr ^ Self::idx_to_size(idx);
            let buddy = unsafe { NonNull::new_unchecked(buddy_addr as *mut LinkedListNode) };

            if unsafe { list_take(&mut self.heads[idx], buddy) } {
                addr = addr.min(buddy_addr);
                idx += 1;
            } else {
                break;
            }
        }

        // Insert node into target bucket
        let node = unsafe { NonNull::new_unchecked(addr as *mut LinkedListNode) };
        unsafe { list_push(&mut self.heads[idx], node) };
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// bucket-0 block size for every test in this module
    const N: usize = size_of::<LinkedListNode>();

    /// A stack-allocated, page-aligned backing region for a `BuddyAllocator`.
    /// Page alignment just avoids wasting the first few bytes of `fill` to
    /// its own alignment step; it is not otherwise required.
    #[repr(align(4096))]
    struct Heap<const SIZE: usize>([u8; SIZE]);

    impl<const SIZE: usize> Heap<SIZE> {
        fn new() -> Self {
            Self([0; SIZE])
        }

        fn ptr(&mut self) -> NonNull<u8> {
            NonNull::new(self.0.as_mut_ptr()).unwrap()
        }
    }

    #[test]
    fn layout_to_idx_maps_sizes_to_the_expected_bucket() {
        assert_eq!(BuddyAllocator::<8>::layout_to_idx(&Layout::from_size_align(1, 1).unwrap()), 0);
        assert_eq!(BuddyAllocator::<8>::layout_to_idx(&Layout::from_size_align(N, 1).unwrap()), 0);
        assert_eq!(BuddyAllocator::<8>::layout_to_idx(&Layout::from_size_align(N + 1, 1).unwrap()), 1);
        assert_eq!(BuddyAllocator::<8>::layout_to_idx(&Layout::from_size_align(2 * N, 1).unwrap()), 1);
        assert_eq!(BuddyAllocator::<8>::layout_to_idx(&Layout::from_size_align(2 * N + 1, 1).unwrap()), 2);
    }

    #[test]
    fn zero_sized_allocation_is_a_dangling_no_op() {
        let mut alloc = BuddyAllocator::<4>::new();
        let layout = Layout::from_size_align(0, 8).unwrap();

        let block = alloc.allocate(layout).expect("zero-sized requests must always succeed");
        assert_eq!(block.len(), 0);
        assert_eq!(block.cast::<u8>().as_ptr() as usize % layout.align(), 0);

        // Must be a no-op: there is no backing memory at all in this allocator.
        unsafe { alloc.deallocate(block.cast(), layout) };
    }

    #[test]
    fn oversized_layout_is_rejected() {
        const LEVELS: usize = 4;
        let mut alloc = BuddyAllocator::<LEVELS>::new();

        // One size class beyond the largest bucket this allocator has.
        let too_big = Layout::from_size_align(N << LEVELS, 1).unwrap();
        assert!(alloc.allocate(too_big).is_err());
    }

    #[test]
    fn allocate_returns_a_block_covering_the_requested_layout() {
        const LEVELS: usize = 4;
        const SIZE: usize = N << (LEVELS - 1);

        let mut heap = Heap::<SIZE>::new();
        let mut alloc = BuddyAllocator::<LEVELS>::new();
        alloc.fill(heap.ptr(), SIZE);

        let layout = Layout::from_size_align(N, N).unwrap();
        let block = alloc.allocate(layout).unwrap();

        assert!(block.len() >= layout.size());
        assert_eq!(block.cast::<u8>().as_ptr() as usize % layout.align(), 0);

        unsafe { alloc.deallocate(block.cast(), layout) };
    }

    #[test]
    fn allocate_exhausts_and_reports_alloc_error() {
        const LEVELS: usize = 4;
        const SIZE: usize = N << (LEVELS - 1);

        let mut heap = Heap::<SIZE>::new();
        let mut alloc = BuddyAllocator::<LEVELS>::new();
        alloc.fill(heap.ptr(), SIZE);

        let layout = Layout::from_size_align(N, N).unwrap();
        for _ in 0..(SIZE / N) {
            alloc.allocate(layout).expect("heap should still have room");
        }

        assert!(alloc.allocate(layout).is_err());
    }

    #[test]
    fn deallocate_merges_freed_buddies_back_into_the_original_block() {
        const LEVELS: usize = 4;
        const SIZE: usize = N << (LEVELS - 1);

        let mut heap = Heap::<SIZE>::new();
        let mut alloc = BuddyAllocator::<LEVELS>::new();
        alloc.fill(heap.ptr(), SIZE);

        let small = Layout::from_size_align(N, N).unwrap();
        let a = alloc.allocate(small).unwrap();
        let b = alloc.allocate(small).unwrap();

        unsafe {
            alloc.deallocate(b.cast(), small);
            alloc.deallocate(a.cast(), small);
        }

        // The two freed minimal blocks, plus the buddy set aside by the very
        // first split, must all have merged back into one top-level block -
        // otherwise this allocation has nowhere large enough to come from.
        let whole = Layout::from_size_align(SIZE, N).unwrap();
        assert!(alloc.allocate(whole).is_ok());
    }

    #[test]
    fn fill_handles_unaligned_regions_and_odd_lengths() {
        const LEVELS: usize = 4;
        const SIZE: usize = N << (LEVELS - 1);
        // Pad with extra room on both ends so an off-by-`N/2` start and a
        // trailing partial block still leave one full-size block available.
        const PADDED: usize = SIZE + 2 * N;

        let mut heap = Heap::<PADDED>::new();
        let base = heap.ptr();
        let offset = unsafe { NonNull::new_unchecked(base.as_ptr().add(N / 2)) };

        let mut alloc = BuddyAllocator::<LEVELS>::new();
        alloc.fill(offset, SIZE + N / 2);

        let layout = Layout::from_size_align(N, N).unwrap();
        assert!(alloc.allocate(layout).is_ok());
    }
}
