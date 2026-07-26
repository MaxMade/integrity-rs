# integrity-rs

Runtime memory-integrity protection with Merkle trees — a software
implementation of the scalable memory-protection scheme from
*Scalable Memory Protection in the PENGLAI Enclave*, applied to an allocator.

Values are stored in a protected region whose contents are continuously
attested by a Merkle tree. Any modification that does **not** go through the
API (a stray DMA write, a malicious OS, a wild pointer) leaves the recorded
hashes out of step with memory, and the next access detects it and panics.

## Assumptions

- **Nightly toolchain.** The crate uses `#![feature(allocator_api)]`. It is
  `#![no_std]` (outside tests) and depends on `alloc`, plus `blake3` and
  `parking_lot`.
- **Fixed compile-time capacity.** The geometry is set by constants, not
  runtime arguments: `TOTAL_MEM = 1 GiB`, `MEM_PER_SUBTREE = 4 MiB`
  (`NUM_SUBTREES = 256`), each sub-tree has `HEIGHT = 4` → 8 leaves, so
  `MEM_PER_SUBTREE_LEAF = 512 KiB`.
- **Region shape.** A `MerkleTree`'s region must be a power of two in size
  and start on an even address; the backing `MemoryManagement` must hand out
  a region laid out accordingly. Integrity is tracked at whole-leaf
  (`MEM_PER_SUBTREE_LEAF`) granularity.
- **All mutation goes through the guard.** `create` / `with_mut` re-hash
  after writing, and `allocate` / `deallocate` re-hash after touching a
  sub-region (the buddy allocator's free-list bookkeeping lives *inside* the
  protected memory). Bytes changed by any other means are, by design, seen
  as tampering.
- **Detection is a panic, not a `Result`.** A failed integrity check
  (`validate`) panics; this fires in release builds too (plain `panic!`, not
  `assert!`).
- **`with` / `with_mut` require `T: Copy`.** The value is copied out (and
  back) by a bitwise `read`/`write`, which is only sound for `Copy` types.
  `create` and the guard's `Drop` (which runs `T`'s destructor) work for any
  `T`.
- **Threat model.** The tree detects out-of-band *modification* of protected
  memory after the fact. It does not prevent writes, hide data, or defend
  against code that legitimately goes through the API.

## Usage

Examples assume the default `std` feature (which provides `Memory` and
`NodeAllocator`).

### Valid — mutate and read through the guard

```rust
use integrity_rs::memory::Memory;
use integrity_rs::merkle_tree::NodeAllocator;
use integrity_rs::mountable_merkle_tree::LockedMountableMerkleTree;
use parking_lot::RawMutex;

let tree = LockedMountableMerkleTree::<RawMutex, NodeAllocator, Memory>::new();

// Store an integrity-protected value.
let counter = tree.create(0u64);

// Mutate it through the guard: the change is recorded (re-hashed).
counter.with_mut(|n| *n += 42);

// Read it back: the value is re-validated first, then returned.
assert_eq!(counter.with(|n| *n), 42);
```

### Invalid — tampering behind the guard's back (panics)

`as_ptr` is an unchecked escape hatch. Writing through it skips the re-hash
that `with_mut` performs, so the recorded hash goes stale:

```rust
let secret = tree.create(0u64);

// Write straight through the raw pointer instead of using `with_mut`,
// so the tree is never told the memory changed.
unsafe { (secret.as_ptr() as *mut u64).write(0xbad) };

// The recorded hash no longer matches memory:
secret.with(|n| *n); // panics: "sub-tree 0 failed integrity check"
```

### Invalid — a non-`Copy` value with `with` (does not compile)

`create` accepts any `T`, but `with` / `with_mut` are bounded on `T: Copy`:

```rust
let owned = tree.create(String::from("hi")); // fine: create is generic over T
owned.with(|s| s.len());                      // error: `String: Copy` is not satisfied
```

## Building

```sh
cargo build            # no_std library
cargo test             # unit + doc tests (uses std)
cargo doc --open       # rendered API docs
```

A nightly toolchain is required (see `rust-toolchain.toml`).
