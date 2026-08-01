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
//! or the upper half (right child, start moved up by the halved size). Any
//! node missing along that path — including the root, for e.g. an
//! [`empty`](MerkleTree::empty) tree — is allocated and linked in on the
//! spot, so a tree only ever pays for the nodes some address has actually
//! been looked up (and therefore `rehash`ed/`validate`d) through.
//!
//! [`rehash`](MerkleTree::rehash) and [`validate`](MerkleTree::validate) both
//! read the actual bytes at `[ptr, ptr + size)` (via `unsafe` raw-pointer
//! dereference) and hash them with BLAKE3, rather than taking a
//! caller-supplied hash: `rehash` records what the covered memory currently
//! contains, and `validate` reports whether it still does. Every ancestor's
//! hash is `blake3(left.hash || left.minor || right.hash || right.minor)`,
//! so it commits to the content of every leaf below it and to the version
//! each of those leaves was last recorded at. The node links are not hashed:
//! that would fold raw allocator addresses into every digest, making a root
//! depend on where its nodes were placed. The tree's shape is therefore
//! unattested — two same-hash siblings can be swapped without an ancestor
//! noticing.
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

/// The per-tree half of a leaf's version, shared by every leaf of one
/// [`MerkleTree`] and bumped only when some leaf's [`MinorCounter`] wraps.
pub type MajorCounter = u64;

/// The per-leaf half of a leaf's version, bumped on every
/// [`rehash`](MerkleTree::rehash) of that leaf.
///
/// Narrow by design: the point of splitting the version is that the wide half
/// is stored once per tree instead of once per leaf. The cost is that
/// wrapping is routine rather than unreachable — see
/// [`rehash`](MerkleTree::rehash) for what has to happen then.
pub type MinorCounter = u8;

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

    /// This leaf's share of its version — see [`MinorCounter`]. Only
    /// meaningful on leaves; inner nodes carry it (and commit to it, via
    /// [`hash_state`](MerkleTreeNode::hash_state)) but never advance it.
    minor: MinorCounter,
}

type MerkleTreeNodePtr = NonNull<MerkleTreeNode>;

impl MerkleTreeNode {
    /// Feed the part of this node's state a parent commits to into `hasher`:
    /// its `hash`, followed by its `minor` counter.
    ///
    /// Carrying `minor` keeps a leaf's version attested: the counter sits in
    /// untrusted memory alongside everything else, so an attacker rolling a
    /// leaf back would roll its counter back too — and because the parent
    /// commits to it, that moves the parent's hash.
    ///
    /// The `left`, `right` and `parent` links are deliberately *not* hashed,
    /// even though they are part of the node. Doing so would fold raw
    /// allocator addresses into every digest, making a tree's root a function
    /// of where its nodes happen to have been placed rather than of the
    /// memory it protects — so the same contents would attest differently on
    /// every run. The cost is that the tree's shape is unattested: two
    /// same-hash siblings can be swapped, or a node re-pointed at another
    /// subtree, without an ancestor noticing.
    ///
    /// The fields are serialized one by one rather than hashing the struct's
    /// bytes wholesale, so the digest doesn't depend on the layout
    /// `repr(Rust)` happens to pick and no padding is ever read.
    fn hash_state(&self, hasher: &mut blake3::Hasher) {
        hasher.update(&self.hash);
        hasher.update(&self.minor.to_ne_bytes());
    }

    /// The hash this node must carry as an inner node: BLAKE3 over its left
    /// child's [`hash_state`](Self::hash_state) followed by its right
    /// child's (a child contributes nothing if that side of the tree hasn't
    /// been materialized yet — see [`MerkleTree::leaf_node`]).
    ///
    /// [`rehash`](MerkleTree::rehash) stores this into every ancestor it
    /// walks through; [`validate`](MerkleTree::validate) compares it against
    /// what is stored.
    ///
    /// # Safety
    ///
    /// This node's `left` and `right`, where set, must point to nodes valid
    /// for reads.
    unsafe fn inner_hash(&self) -> [u8; 32] {
        let mut hasher = blake3::Hasher::new();

        for child in [self.left, self.right] {
            if let Some(child) = child {
                unsafe { child.as_ref().hash_state(&mut hasher) };
            }
        }

        *hasher.finalize().as_bytes()
    }
}

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

    /// The wide half of every leaf's version — see [`MajorCounter`]. Held
    /// once here rather than once per leaf, which is the whole point of
    /// splitting the counter.
    ///
    /// Unlike the per-leaf `minor`, this lives outside the tree it versions,
    /// so nothing in the tree attests to it: a higher level must publish
    /// [`root_digest`](Self::root_digest), not
    /// [`root_hash`](Self::root_hash), for it to be covered.
    major: MajorCounter,

    phantom: PhantomData<A>,
}

impl<A: MerkleTreeNodeAllocator> MerkleTree<A> {
    /// # Panics
    ///
    /// Panics if `ptr` or `size` is not a power of two: both are required
    /// so that [`leaf_node`](Self::leaf_node) can subdivide the region
    /// evenly at every level.
    fn validate_ptr_size(ptr: NonNull<c_void>, size: usize) {
        if ptr.addr().get() % 2 != 0 {
            panic!("Start address ({:p}) must be multiple of two!", ptr);
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
            major: 0,
            phantom: PhantomData,
        }
    }

    /// Allocate a single node via `A`, initialized as a fresh, unlinked leaf
    /// (`left`/`right`/`parent` all `None`, `hash` all zero).
    ///
    /// # Panics
    ///
    /// Panics if `A::allocate` fails.
    fn allocate_node() -> MerkleTreeNodePtr {
        let mut node = match A::allocate() {
            Ok(node) => node,
            Err(error) => panic!("Unable to allocate MerkleTreeNode: {}", error),
        };

        unsafe {
            let node = node.as_mut();
            node.left = None;
            node.right = None;
            node.parent = None;
            node.hash = [0; 32];
            node.minor = 0;
        }

        node
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
            nodes[i] = Some(Self::allocate_node());
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
            major: 0,
            phantom: PhantomData,
        }
    }

    /// The base address of the leaf covering `ptr` — the same address
    /// [`leaf_node`](Self::leaf_node) arrives at by halving its way down, but
    /// computed directly, since a leaf is just a fixed-size slice of the
    /// region.
    ///
    /// Only meaningful for a `ptr` inside this tree's region; callers get
    /// that by having already resolved `ptr` through `leaf_node`.
    fn leaf_base(&self, ptr: NonNull<u8>) -> usize {
        let start = self.ptr.addr().get();
        let leaf_size = self.size / NUM_LEAF_NODES;

        start + ((ptr.addr().get() - start) / leaf_size) * leaf_size
    }

    /// The hash a leaf must carry: BLAKE3 over its version — this tree's
    /// [`major`](MajorCounter) and the leaf's own [`minor`](MinorCounter) —
    /// then the leaf's base address, then the `size` bytes at `ptr`.
    ///
    /// The version is what makes the digest unrepeatable, so restoring a
    /// leaf's old contents no longer reproduces its old hash. The address
    /// binds the digest to *where* the bytes live, so a leaf cannot be
    /// spliced in somewhere else.
    ///
    /// # Safety
    ///
    /// `[ptr, ptr + size)` must be valid for reads for the duration of this
    /// call (see [`slice::from_raw_parts`]).
    unsafe fn leaf_hash(&self, ptr: NonNull<u8>, size: usize, minor: MinorCounter) -> [u8; 32] {
        let mut hasher = blake3::Hasher::new();

        hasher.update(&self.major.to_ne_bytes());
        hasher.update(&minor.to_ne_bytes());
        hasher.update(&self.leaf_base(ptr).to_ne_bytes());
        unsafe { hasher.update(slice::from_raw_parts(ptr.as_ptr(), size)) };

        *hasher.finalize().as_bytes()
    }

    /// Start a fresh [`major`](MajorCounter) and re-record every leaf under
    /// it, restarting their [`minor`](MinorCounter) counters at 1.
    ///
    /// This is what a minor counter wrapping costs. The major is shared by
    /// the whole tree, so bumping it re-versions every leaf at once and
    /// invalidates every hash recorded against the old major — each one has
    /// to be recomputed from the memory it covers, and every inner node
    /// recomputed above them. Only leaves that were actually recorded (a
    /// non-zero `minor`) are touched: a leaf nothing was ever `rehash`ed
    /// through stays unrecorded rather than being silently blessed with
    /// whatever its memory currently holds.
    ///
    /// Together with the strictly increasing major, restarting the minors at
    /// 1 keeps every leaf's `(major, minor)` pair strictly increasing, so no
    /// leaf ever republishes a version — which is the property the whole
    /// split-counter scheme exists to provide.
    ///
    /// Re-recording is at whole-leaf granularity, since the tree does not
    /// remember which sub-range of a leaf a caller last passed to
    /// [`rehash`](Self::rehash).
    ///
    /// # Safety
    ///
    /// Every leaf previously recorded through [`rehash`](Self::rehash) is
    /// re-read in full, so the whole leaf-sized slice around each such range
    /// must still be valid for reads.
    ///
    /// # Panics
    ///
    /// Panics if the major counter is exhausted, which at `u64` takes more
    /// re-hashes than the machine can perform.
    unsafe fn begin_new_major(&mut self) {
        self.major = self
            .major
            .checked_add(1)
            .expect("MerkleTree major counter exhausted");

        let Some(root) = self.root else { return };

        // Collect the materialized nodes breadth-first, carrying each one's
        // depth and the base of the region it covers — a leaf needs the base
        // to know which bytes to re-read, and the depth is what distinguishes
        // a real leaf from an inner node whose children simply aren't
        // materialized yet.
        let mut queue = [None; NUM_NODES];
        queue[0] = Some((root, self.ptr.addr().get(), 0usize));
        let mut len = 1;
        let mut idx = 0;
        while idx < len {
            let (node, base, depth) = queue[idx].unwrap();
            idx += 1;

            if depth == HEIGHT - 1 {
                continue;
            }

            let (left, right) = unsafe { (node.as_ref().left, node.as_ref().right) };
            let half = (self.size >> depth) / 2;
            for (child, base) in [(left, base), (right, base + half)] {
                if let Some(child) = child {
                    queue[len] = Some((child, base, depth + 1));
                    len += 1;
                }
            }
        }

        // Walk that back to front: breadth-first order puts every parent
        // before its children, so reversing it recomputes each inner node
        // only once its children are already up to date.
        let leaf_size = self.size / NUM_LEAF_NODES;
        for entry in queue[..len].iter().rev() {
            let (mut node, base, depth) = entry.unwrap();

            if depth < HEIGHT - 1 {
                let hash = unsafe { node.as_ref().inner_hash() };
                unsafe { node.as_mut().hash = hash };
                continue;
            }

            // A leaf nothing was ever recorded through has nothing to
            // re-record, and must not start matching its memory now.
            if unsafe { node.as_ref().minor } == 0 {
                continue;
            }

            let ptr = unsafe { NonNull::new_unchecked(base as *mut u8) };
            let hash = unsafe { self.leaf_hash(ptr, leaf_size, 1) };
            unsafe {
                node.as_mut().minor = 1;
                node.as_mut().hash = hash;
            }
        }
    }

    /// Find the single leaf node that fully contains `[ptr, ptr + size)`,
    /// allocating and linking in any node missing along the way (including
    /// the root, for e.g. an [`empty`](Self::empty) tree) — so this always
    /// succeeds for any range this tree covers, regardless of how much of
    /// it was already linked.
    ///
    /// Returns `None` only if that range is not entirely covered by one
    /// leaf: it falls outside this tree's region, straddles two leaves, or
    /// overflows. No allocation happens in that case.
    ///
    /// # Panics
    ///
    /// Panics if `A::allocate` fails to provide a node needed along the way.
    pub fn leaf_node(&mut self, ptr: NonNull<u8>, size: usize) -> Option<MerkleTreeNodePtr> {
        let ptr = usize::from(ptr.addr());
        let mut mem_start = usize::from(self.ptr.addr());
        let mut mem_size = self.size;

        if ptr < mem_start {
            return None;
        }

        if ptr >= mem_start + mem_size {
            return None;
        }

        if self.root.is_none() {
            self.root = Some(Self::allocate_node());
        }

        let mut node = self.root.unwrap();
        for _ in 0..HEIGHT - 1 {
            mem_size /= 2;

            let go_right = ptr >= mem_start + mem_size;
            if go_right {
                mem_start += mem_size;
            }

            let child = unsafe {
                match go_right {
                    true => node.as_ref().right,
                    false => node.as_ref().left,
                }
            };

            node = match child {
                Some(child) => child,
                None => {
                    let mut child = Self::allocate_node();
                    unsafe {
                        child.as_mut().parent = Some(node);
                        match go_right {
                            true => node.as_mut().right = Some(child),
                            false => node.as_mut().left = Some(child),
                        }
                    }
                    child
                }
            };
        }

        if ptr + size < mem_start {
            return None;
        }

        if ptr + size > mem_start + mem_size {
            return None;
        }

        Some(node)
    }

    /// Advance the version of the leaf covering `[ptr, ptr + size)`, hash the
    /// `size` bytes at `ptr` together with that version, record the result as
    /// the leaf's hash, then recompute the hash of every ancestor up to the
    /// root — allocating any node along that path that isn't linked yet (see
    /// [`leaf_node`](Self::leaf_node)).
    ///
    /// The leaf's hash is BLAKE3 over `major || minor || leaf_base || memory`
    /// (see [`leaf_hash`](Self::leaf_hash)); each internal node's is BLAKE3
    /// over its children's hashes and `minor` counters (a child contributes
    /// nothing if that side of the tree hasn't been touched yet). Every
    /// node's hash is therefore a pure function of the memory contents in its
    /// subtree *and* how many times each of those leaves has been re-hashed —
    /// so restoring a leaf's old bytes no longer restores its old hash.
    ///
    /// Once this leaf's minor counter wraps, the whole tree moves to a fresh
    /// major and every recorded leaf is re-read and re-recorded under it —
    /// see [`begin_new_major`](Self::begin_new_major). That is the cost the
    /// split buys back: a narrow per-leaf counter in exchange for an
    /// occasional pass over the whole region.
    ///
    /// # Safety
    ///
    /// `[ptr, ptr + size)` must be valid for reads for the duration of this
    /// call (see [`slice::from_raw_parts`]).
    ///
    /// On the wrap described above this call also re-reads *every* leaf ever
    /// recorded in this tree, in full, so the whole leaf-sized slice around
    /// each such range must still be valid for reads too. Any caller that
    /// hands a region to `rehash` and later unmaps part of it violates this,
    /// even though the offending call names only the range it passed.
    ///
    /// # Panics
    ///
    /// Panics if `[ptr, ptr + size)` is not fully contained in a single leaf
    /// of this tree, or `A::allocate` fails to provide a node needed along
    /// the way (see [`leaf_node`](Self::leaf_node)).
    pub unsafe fn rehash(&mut self, ptr: NonNull<u8>, size: usize) {
        // Find associated leaf node
        let mut node = match self.leaf_node(ptr, size) {
            Some(node) => node,
            None => panic!("Unable to find leaf node for address ({:p})", ptr),
        };

        // Advance this leaf's version, so the hash below cannot repeat one
        // this leaf has already published for the same contents. A wrap would
        // repeat it, so instead the whole tree moves to a fresh major.
        let minor = unsafe {
            match node.as_ref().minor.checked_add(1) {
                Some(minor) => {
                    node.as_mut().minor = minor;
                    minor
                }
                None => {
                    self.begin_new_major();
                    node.as_ref().minor
                }
            }
        };

        // Calculate hash over the new version, the leaf's address, and memory
        let hash = unsafe { self.leaf_hash(ptr, size, minor) };

        // Update hash of leaf node
        let mut drag = unsafe {
            let node = node.as_mut();
            node.hash = hash;
            node.parent
        };

        // Continue with parents
        while let Some(mut node) = drag {
            unsafe {
                let node = node.as_mut();

                // Update hash over both children's full state
                node.hash = node.inner_hash();

                // Continue with parent
                drag = node.parent;
            }
        }
    }

    /// Hash the `size` bytes at `ptr` with BLAKE3 and check that it matches
    /// the cached hash of the leaf covering that range, and that every
    /// ancestor's cached hash up to the root is consistent with its children
    /// (i.e. still equals BLAKE3 over both children's hashes and counters).
    ///
    /// This recomputes the leaf hash from the current memory contents and the
    /// leaf's *current* version — it never advances a counter, so validating
    /// is idempotent and two `validate`s around a `rehash` differ. It catches
    /// memory that has changed since the last [`rehash`](Self::rehash) of
    /// this leaf, any node's cached hash having been corrupted directly,
    /// and — because the ancestor check covers the children's counters as
    /// well as their hashes — a leaf's version having been rolled back
    /// underneath it. It does *not* catch the tree being re-linked, since the
    /// links are not hashed. Note that it still takes
    /// `&mut self`: like `rehash`, it allocates any node along the way that
    /// isn't linked yet (see [`leaf_node`](Self::leaf_node)) — which, for a
    /// range that was never `rehash`ed, just means it reliably returns
    /// `false` (comparing against a fresh, all-zero hash) rather than
    /// panicking.
    ///
    /// # Safety
    ///
    /// `[ptr, ptr + size)` must be valid for reads for the duration of this
    /// call (see [`slice::from_raw_parts`]).
    ///
    /// # Panics
    ///
    /// Panics if `[ptr, ptr + size)` is not fully contained in a single leaf
    /// of this tree, or `A::allocate` fails to provide a node needed along
    /// the way (see [`leaf_node`](Self::leaf_node)).
    pub unsafe fn validate(&mut self, ptr: NonNull<u8>, size: usize) -> bool {
        // Find associated leaf node
        let node = match self.leaf_node(ptr, size) {
            Some(node) => node,
            None => panic!("Unable to find leaf node for address ({:p})", ptr),
        };

        // Calculate hash against the leaf's current version, without
        // advancing it
        let hash = unsafe { self.leaf_hash(ptr, size, node.as_ref().minor) };

        // Check leaf node
        let mut drag = unsafe {
            let node = node.as_ref();

            // Error: hashes didn't match...
            if node.hash != hash {
                return false;
            }

            node.parent
        };

        // Continue with parents
        while let Some(node) = drag {
            unsafe {
                let node = node.as_ref();

                // Error: hashes didn't match...
                if node.hash != node.inner_hash() {
                    return false;
                }

                // Continue with parent
                drag = node.parent;
            }
        }

        true
    }

    /// The root node's hash — the digest committing to every leaf hash in
    /// this tree — or all-zeros if the tree has no nodes yet (an
    /// [`empty`](Self::empty) tree, or one nothing has been
    /// [`rehash`](Self::rehash)ed through).
    ///
    /// It changes whenever any covered leaf is re-hashed, unlike the bytes of
    /// the `MerkleTree` struct itself. It does *not* cover
    /// [`major`](MajorCounter), which lives in that struct rather than in the
    /// tree — use [`root_digest`](Self::root_digest) for that.
    pub fn root_hash(&self) -> [u8; 32] {
        match self.root {
            Some(root) => unsafe { root.as_ref().hash },
            None => [0; 32],
        }
    }

    /// This tree's current epoch — the [`MajorCounter`] every leaf's version
    /// is taken against, advanced whenever some leaf's minor counter wraps
    /// (see [`rehash`](Self::rehash)).
    ///
    /// Exposed so a caller managing freshness can observe how far the tree
    /// has advanced; it is already folded into
    /// [`root_digest`](Self::root_digest), which is what should actually be
    /// published.
    pub fn major(&self) -> MajorCounter {
        self.major
    }

    /// This tree's full attestation: BLAKE3 over its
    /// [`major`](MajorCounter) counter and its
    /// [`root_hash`](Self::root_hash).
    ///
    /// This — not `root_hash` — is what a higher-level tree should publish to
    /// attest to this one. `major` is held in the `MerkleTree` struct, which
    /// no node hashes, so a `root_hash` alone leaves it unattested: an
    /// attacker could roll a whole tree back to a state it held under an
    /// earlier major and the digest would not move.
    pub fn root_digest(&self) -> [u8; 32] {
        let mut hasher = blake3::Hasher::new();

        hasher.update(&self.major.to_ne_bytes());
        hasher.update(&self.root_hash());

        *hasher.finalize().as_bytes()
    }
}

impl<A: MerkleTreeNodeAllocator> Drop for MerkleTree<A> {
    /// Deallocate every node linked into this tree (a no-op for an
    /// [`empty`](Self::empty) tree, which has none).
    fn drop(&mut self) {
        let Some(root) = self.root else { return };

        // BFS over the fixed, complete tree, freeing every node.
        let mut queue = [None; NUM_NODES];
        queue[0] = Some(root);
        let mut len = 1;
        let mut idx = 0;
        while idx < len {
            let node = queue[idx].unwrap();
            idx += 1;

            let (left, right) = unsafe { (node.as_ref().left, node.as_ref().right) };
            for child in [left, right] {
                if let Some(child) = child {
                    queue[len] = Some(child);
                    len += 1;
                }
            }

            unsafe { A::deallocate(node) };
        }
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
                minor: 0,
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
    #[should_panic(expected = "multiple of two")]
    fn empty_panics_on_non_power_of_two_ptr() {
        MerkleTree::<NullAllocator>::empty(ptr(0x1001), 0x100);
    }

    #[cfg(feature = "std")]
    #[test]
    fn leaf_node_on_empty_tree_allocates_and_links_missing_nodes() {
        let mut tree = MerkleTree::<NodeAllocator>::empty(ptr(0x1000), 0x100);

        // In range, but nothing has been linked yet: this must succeed by
        // allocating and linking every node along the way, not fail.
        let leaf = tree.leaf_node(byte_ptr(0x1000), 1).unwrap();
        assert_eq!(unsafe { leaf.as_ref().hash }, [0u8; 32]);

        // A second lookup on the same address reuses the same node instead
        // of allocating a new one.
        assert_eq!(tree.leaf_node(byte_ptr(0x1000), 1), Some(leaf));
    }

    #[test]
    fn leaf_node_rejects_addresses_outside_the_region() {
        // Out of range, so this must never allocate anything -
        // `NullAllocator` would panic if it did.
        let mut tree = MerkleTree::<NullAllocator>::empty(ptr(0x1000), 0x100);

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
    fn dropping_a_constructed_tree_deallocates_every_node() {
        use std::sync::atomic::{AtomicUsize, Ordering};

        static DEALLOCATED: AtomicUsize = AtomicUsize::new(0);

        struct CountingAllocator;

        impl MerkleTreeNodeAllocator for CountingAllocator {
            fn allocate() -> Result<NonNull<MerkleTreeNode>, AllocError> {
                NodeAllocator::allocate()
            }

            unsafe fn deallocate(node: NonNull<MerkleTreeNode>) {
                DEALLOCATED.fetch_add(1, Ordering::Relaxed);
                unsafe { NodeAllocator::deallocate(node) };
            }
        }

        let tree = MerkleTree::<CountingAllocator>::constructed(ptr(0x1000), 0x100);
        drop(tree);

        assert_eq!(DEALLOCATED.load(Ordering::Relaxed), NUM_NODES);
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

        let mut tree = MerkleTree::<NodeAllocator>::constructed(ptr(START), SIZE);

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

        let mut tree = MerkleTree::<NodeAllocator>::constructed(ptr(START), SIZE);

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

        let mut tree = MerkleTree::<NodeAllocator>::constructed(ptr(START), SIZE);

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
            .unwrap_or_else(|error| {
                panic!("Unable to map {size:#x} byte(s) at {addr:#x}: {error}")
            });

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

    /// Recompute a leaf's hash exactly the way `leaf_hash` is specified to:
    /// the tree's major, the leaf's minor, the leaf's base address, then the
    /// covered memory. Spelled out here rather than calling `leaf_hash` so
    /// the tests pin the preimage down independently of the implementation.
    #[cfg(feature = "std")]
    fn expected_leaf_hash(
        major: MajorCounter,
        minor: MinorCounter,
        base: usize,
        ptr: NonNull<u8>,
        size: usize,
    ) -> [u8; 32] {
        let mut hasher = blake3::Hasher::new();

        hasher.update(&major.to_ne_bytes());
        hasher.update(&minor.to_ne_bytes());
        hasher.update(&base.to_ne_bytes());
        hasher.update(unsafe { slice::from_raw_parts(ptr.as_ptr(), size) });

        *hasher.finalize().as_bytes()
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

        // A first rehash leaves the leaf at minor 1, under the tree's initial
        // major of 0.
        let expected = expected_leaf_hash(0, 1, START, region.byte_ptr(0), LEAF_SIZE);

        let leaf = tree.leaf_node(region.byte_ptr(0), LEAF_SIZE).unwrap();
        assert_eq!(unsafe { leaf.as_ref().hash }, expected);
        assert_eq!(unsafe { leaf.as_ref().minor }, 1);
    }

    /// Serialize a node exactly the way `hash_state` is specified to: its
    /// hash, then its minor counter. Spelled out here rather than calling
    /// `hash_state` so the tests pin the wire format down independently of
    /// the implementation.
    #[cfg(feature = "std")]
    fn expected_state(node: MerkleTreeNodePtr) -> std::vec::Vec<u8> {
        let node = unsafe { node.as_ref() };

        let mut state = std::vec::Vec::new();
        state.extend_from_slice(&node.hash);
        state.extend_from_slice(&node.minor.to_ne_bytes());

        state
    }

    #[cfg(feature = "std")]
    #[test]
    fn rehash_computes_ancestor_hashes_from_its_children() {
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

        let left = tree.leaf_node(region.byte_ptr(0), LEAF_SIZE).unwrap();
        let right = tree
            .leaf_node(region.byte_ptr(LEAF_SIZE), LEAF_SIZE)
            .unwrap();

        // Each leaf carries the hash of its version, address and memory...
        for (leaf, offset) in [(left, 0), (right, LEAF_SIZE)] {
            let expected =
                expected_leaf_hash(0, 1, START + offset, region.byte_ptr(offset), LEAF_SIZE);
            assert_eq!(unsafe { leaf.as_ref().hash }, expected);
        }

        // ...and their parent hashes both leaves' hash and counter.
        let mut hasher = blake3::Hasher::new();
        hasher.update(&expected_state(left));
        hasher.update(&expected_state(right));
        let expected = *hasher.finalize().as_bytes();

        let parent = unsafe { left.as_ref().parent }.unwrap();
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
    fn rehash_on_an_empty_tree_lazily_builds_only_the_touched_path() {
        const START: usize = 1 << 36;
        const SIZE: usize = 0x1000;
        const LEAF_SIZE: usize = SIZE / NUM_LEAF_NODES;

        let region = MappedRegion::new(START, SIZE);
        let mut tree = MerkleTree::<NodeAllocator>::empty(region.base(), SIZE);

        // No prior `constructed()` call: this must allocate and link the
        // whole root-to-leaf path on its own.
        unsafe { tree.rehash(region.byte_ptr(0), LEAF_SIZE) };

        let expected = expected_leaf_hash(0, 1, START, region.byte_ptr(0), LEAF_SIZE);

        let leaf = tree.leaf_node(region.byte_ptr(0), LEAF_SIZE).unwrap();
        assert_eq!(unsafe { leaf.as_ref().hash }, expected);

        // Only the touched path was allocated: the leaf's sibling subtree
        // was never looked up, so it must still be unlinked.
        let sibling = unsafe { leaf.as_ref().parent.unwrap().as_ref().right };
        assert!(sibling.is_none());
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
        let mut tree = MerkleTree::<NodeAllocator>::constructed(region.base(), SIZE);

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
        let mut parent = unsafe {
            tree.leaf_node(region.byte_ptr(0), LEAF_SIZE)
                .unwrap()
                .as_ref()
                .parent
        }
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

        let mut tree = MerkleTree::<NodeAllocator>::constructed(ptr(START), SIZE);

        // Out of range: `leaf_node` returns `None` before this address is
        // ever dereferenced, so no real backing memory is needed.
        unsafe { tree.validate(byte_ptr(START + SIZE), 1) };
    }

    #[cfg(feature = "std")]
    #[test]
    fn rehash_advances_the_leaf_version_and_validate_does_not() {
        const START: usize = 1 << 37;
        const SIZE: usize = 0x1000;
        const LEAF_SIZE: usize = SIZE / NUM_LEAF_NODES;

        let region = MappedRegion::new(START, SIZE);
        let mut tree = MerkleTree::<NodeAllocator>::constructed(region.base(), SIZE);

        let leaf = tree.leaf_node(region.byte_ptr(0), LEAF_SIZE).unwrap();
        assert_eq!(unsafe { leaf.as_ref().minor }, 0);

        for expected in 1..=3 {
            unsafe { tree.rehash(region.byte_ptr(0), LEAF_SIZE) };
            assert_eq!(unsafe { leaf.as_ref().minor }, expected);

            // Validating must be idempotent — it reads the version, never
            // advances it, so it can run any number of times between writes.
            for _ in 0..3 {
                assert!(unsafe { tree.validate(region.byte_ptr(0), LEAF_SIZE) });
                assert_eq!(unsafe { leaf.as_ref().minor }, expected);
            }
        }

        // Only the rehashed leaf's version moved.
        let untouched = tree
            .leaf_node(region.byte_ptr(LEAF_SIZE), LEAF_SIZE)
            .unwrap();
        assert_eq!(unsafe { untouched.as_ref().minor }, 0);
    }

    #[cfg(feature = "std")]
    #[test]
    fn rehash_gives_identical_leaves_different_hashes() {
        const START: usize = 1 << 38;
        const SIZE: usize = 0x1000;
        const LEAF_SIZE: usize = SIZE / NUM_LEAF_NODES;

        let region = MappedRegion::new(START, SIZE);
        let mut tree = MerkleTree::<NodeAllocator>::constructed(region.base(), SIZE);

        // Byte-for-byte identical content in two different leaves, both at
        // the same version.
        region.fill(0, LEAF_SIZE, 0x5a);
        region.fill(LEAF_SIZE, LEAF_SIZE, 0x5a);
        unsafe {
            tree.rehash(region.byte_ptr(0), LEAF_SIZE);
            tree.rehash(region.byte_ptr(LEAF_SIZE), LEAF_SIZE);
        }

        // The leaf base address is part of the preimage, so the two hashes
        // must still differ — otherwise one leaf could be spliced in for the
        // other.
        let first = tree.leaf_node(region.byte_ptr(0), LEAF_SIZE).unwrap();
        let second = tree
            .leaf_node(region.byte_ptr(LEAF_SIZE), LEAF_SIZE)
            .unwrap();
        assert_ne!(unsafe { first.as_ref().hash }, unsafe {
            second.as_ref().hash
        });
    }

    #[cfg(feature = "std")]
    #[test]
    fn returning_to_a_previous_state_does_not_reproduce_the_root_hash() {
        const START: usize = 1 << 39;
        const SIZE: usize = 0x1000;
        const LEAF_SIZE: usize = SIZE / NUM_LEAF_NODES;

        let region = MappedRegion::new(START, SIZE);
        let mut tree = MerkleTree::<NodeAllocator>::constructed(region.base(), SIZE);

        // Record state A, then B, then byte-for-byte back to A.
        region.fill(0, LEAF_SIZE, 0xa1);
        unsafe { tree.rehash(region.byte_ptr(0), LEAF_SIZE) };
        let first = tree.root_hash();

        region.fill(0, LEAF_SIZE, 0xb2);
        unsafe { tree.rehash(region.byte_ptr(0), LEAF_SIZE) };
        let second = tree.root_hash();

        region.fill(0, LEAF_SIZE, 0xa1);
        unsafe { tree.rehash(region.byte_ptr(0), LEAF_SIZE) };
        let third = tree.root_hash();

        // This is what the version buys. Hashing content alone makes the root
        // a pure function of memory, so returning to A republishes A's exact
        // digest — and an outside observer holding the last-seen digest
        // cannot tell a rollback from a legitimate rewrite of the same value.
        // The counter makes the digest sequence unrepeatable.
        assert_ne!(third, first, "root hash must not repeat for repeated state");
        assert_ne!(third, second);
        assert_ne!(first, second);
    }

    #[cfg(feature = "std")]
    #[test]
    fn validate_is_false_when_a_leaf_version_was_rewound() {
        const START: usize = 1 << 40;
        const SIZE: usize = 0x1000;
        const LEAF_SIZE: usize = SIZE / NUM_LEAF_NODES;

        let region = MappedRegion::new(START, SIZE);
        let mut tree = MerkleTree::<NodeAllocator>::constructed(region.base(), SIZE);

        region.fill(0, LEAF_SIZE, 0xa1);
        unsafe {
            tree.rehash(region.byte_ptr(0), LEAF_SIZE);
            tree.rehash(region.byte_ptr(0), LEAF_SIZE);
        }
        let mut leaf = tree.leaf_node(region.byte_ptr(0), LEAF_SIZE).unwrap();
        assert_eq!(unsafe { leaf.as_ref().minor }, 2);

        // Rewind the counter to set up a later replay. Nothing here is
        // secret, so the attacker can re-derive the leaf's hash for the
        // rewound version — which leaves the leaf wholly self-consistent, and
        // the leaf-level check alone passes.
        unsafe {
            leaf.as_mut().minor = 1;
            leaf.as_mut().hash = expected_leaf_hash(0, 1, START, region.byte_ptr(0), LEAF_SIZE);
        }

        // The parent is what catches it: `hash_state` folds the child's
        // counter into the parent's hash, so a rewind moves the parent's
        // digest even though the leaf still checks out against itself.
        assert!(!unsafe { tree.validate(region.byte_ptr(0), LEAF_SIZE) });
    }

    /// Rehash `leaf` enough times to wrap its minor counter exactly once,
    /// varying the content so the writes are not degenerate.
    #[cfg(feature = "std")]
    fn wrap_minor_once(tree: &mut MerkleTree<NodeAllocator>, region: &MappedRegion, leaf: usize) {
        const LEAF_SIZE: usize = 0x1000 / NUM_LEAF_NODES;

        for i in 0..=MinorCounter::MAX as u32 {
            region.fill(leaf * LEAF_SIZE, LEAF_SIZE, (i % 251) as u8);
            unsafe { tree.rehash(region.byte_ptr(leaf * LEAF_SIZE), LEAF_SIZE) };
        }
    }

    #[cfg(feature = "std")]
    #[test]
    fn wrapping_a_minor_counter_starts_a_new_major() {
        const START: usize = 1 << 42;
        const SIZE: usize = 0x1000;
        const LEAF_SIZE: usize = SIZE / NUM_LEAF_NODES;

        let region = MappedRegion::new(START, SIZE);
        let mut tree = MerkleTree::<NodeAllocator>::constructed(region.base(), SIZE);

        assert_eq!(tree.major, 0);
        wrap_minor_once(&mut tree, &region, 0);

        // The wrap moved the tree to major 1 and restarted the leaf at 1, so
        // `(major, minor)` advanced rather than repeating.
        assert_eq!(tree.major, 1);
        let leaf = tree.leaf_node(region.byte_ptr(0), LEAF_SIZE).unwrap();
        assert_eq!(unsafe { leaf.as_ref().minor }, 1);

        // The leaf is still consistent with its memory afterwards.
        assert!(unsafe { tree.validate(region.byte_ptr(0), LEAF_SIZE) });
    }

    #[cfg(feature = "std")]
    #[test]
    fn repeated_writes_of_one_value_never_republish_a_digest() {
        use std::collections::HashSet;

        const START: usize = 1 << 43;
        const SIZE: usize = 0x1000;
        const LEAF_SIZE: usize = SIZE / NUM_LEAF_NODES;

        let region = MappedRegion::new(START, SIZE);
        let mut tree = MerkleTree::<NodeAllocator>::constructed(region.base(), SIZE);

        // Write one unchanging value over and over, for more than a full lap
        // of the minor counter. Content is constant, so the published digest
        // is a pure function of the version — and the version is exactly what
        // must never repeat. Wrapping the minor without a new major makes the
        // sequence periodic, which is the replay a version exists to
        // foreclose.
        region.fill(0, LEAF_SIZE, 0xa1);

        let laps = 2 * (MinorCounter::MAX as u32 + 1);
        let mut leaf_digests = HashSet::new();
        let mut root_digests = HashSet::new();

        for i in 0..laps {
            unsafe { tree.rehash(region.byte_ptr(0), LEAF_SIZE) };

            let leaf = tree.leaf_node(region.byte_ptr(0), LEAF_SIZE).unwrap();
            assert!(
                leaf_digests.insert(unsafe { leaf.as_ref().hash }),
                "leaf digest repeated on write {i}"
            );
            assert!(
                root_digests.insert(tree.root_digest()),
                "root digest repeated on write {i}"
            );
        }

        // Every write is still consistent with memory at the end of it all.
        assert!(unsafe { tree.validate(region.byte_ptr(0), LEAF_SIZE) });
    }

    #[cfg(feature = "std")]
    #[test]
    fn a_wrap_re_records_other_leaves_and_leaves_unrecorded_ones_alone() {
        const START: usize = 1 << 44;
        const SIZE: usize = 0x1000;
        const LEAF_SIZE: usize = SIZE / NUM_LEAF_NODES;

        let region = MappedRegion::new(START, SIZE);
        let mut tree = MerkleTree::<NodeAllocator>::constructed(region.base(), SIZE);

        // A bystander leaf, recorded once under major 0 and never written
        // again. Its hash was taken against the old major, so the wrap must
        // recompute it or it would stop validating.
        region.fill(LEAF_SIZE, LEAF_SIZE, 0xbb);
        unsafe { tree.rehash(region.byte_ptr(LEAF_SIZE), LEAF_SIZE) };
        let bystander_hash = unsafe {
            tree.leaf_node(region.byte_ptr(LEAF_SIZE), LEAF_SIZE)
                .unwrap()
                .as_ref()
                .hash
        };

        wrap_minor_once(&mut tree, &region, 0);

        let bystander = tree
            .leaf_node(region.byte_ptr(LEAF_SIZE), LEAF_SIZE)
            .unwrap();
        assert_ne!(unsafe { bystander.as_ref().hash }, bystander_hash);
        assert_eq!(unsafe { bystander.as_ref().minor }, 1);
        assert!(unsafe { tree.validate(region.byte_ptr(LEAF_SIZE), LEAF_SIZE) });

        // A leaf nothing was ever rehashed through must stay unrecorded — the
        // wrap must not bless whatever its memory happens to hold.
        let untouched = tree
            .leaf_node(region.byte_ptr(SIZE - LEAF_SIZE), LEAF_SIZE)
            .unwrap();
        assert_eq!(unsafe { untouched.as_ref().minor }, 0);
        assert_eq!(unsafe { untouched.as_ref().hash }, [0u8; 32]);
        assert!(!unsafe { tree.validate(region.byte_ptr(SIZE - LEAF_SIZE), LEAF_SIZE) });
    }

    #[cfg(feature = "std")]
    #[test]
    fn a_wrap_on_a_lazily_built_tree_only_touches_materialized_nodes() {
        const START: usize = 1 << 45;
        const SIZE: usize = 0x1000;
        const LEAF_SIZE: usize = SIZE / NUM_LEAF_NODES;

        let region = MappedRegion::new(START, SIZE);
        let mut tree = MerkleTree::<NodeAllocator>::empty(region.base(), SIZE);

        // Only one root-to-leaf path exists. The wrap's traversal must not
        // mistake a half-built inner node for a leaf and hash memory into it.
        wrap_minor_once(&mut tree, &region, 0);

        assert_eq!(tree.major, 1);
        assert!(unsafe { tree.validate(region.byte_ptr(0), LEAF_SIZE) });

        let leaf = tree.leaf_node(region.byte_ptr(0), LEAF_SIZE).unwrap();
        let sibling = unsafe { leaf.as_ref().parent.unwrap().as_ref().right };
        assert!(sibling.is_none(), "the wrap must not materialize new nodes");
    }

    #[cfg(feature = "std")]
    #[test]
    fn root_digest_covers_the_major_counter() {
        const START: usize = 1 << 41;
        const SIZE: usize = 0x1000;
        const LEAF_SIZE: usize = SIZE / NUM_LEAF_NODES;

        let region = MappedRegion::new(START, SIZE);
        let mut tree = MerkleTree::<NodeAllocator>::constructed(region.base(), SIZE);

        unsafe { tree.rehash(region.byte_ptr(0), LEAF_SIZE) };
        let digest = tree.root_digest();
        let root_hash = tree.root_hash();

        // Advancing the major alone leaves every node untouched, so
        // `root_hash` cannot see it — only `root_digest` can, which is why
        // that is what a higher level must publish.
        tree.major += 1;
        assert_eq!(tree.root_hash(), root_hash);
        assert_ne!(tree.root_digest(), digest);
    }
}
