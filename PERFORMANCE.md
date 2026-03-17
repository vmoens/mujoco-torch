# Compile Performance Optimizations

This document tracks `torch.compile` throughput improvements in mujoco-torch,
measured on a single NVIDIA H200 GPU at float64 precision.

## Methodology

- **Hardware:** Single NVIDIA H200 GPU
- **Precision:** float64
- **Batch size:** 32 768
- **Benchmark:** `test_compile` (base `torch.compile`, no tuning flags)
- **Procedure:** The Inductor cache was cleared and the full benchmark suite
  was run from scratch (5 models, 100 steps each after warmup).

## Optimizations

### 1. Remove `scan_padding` feature

**Commit:** `df54c6ce` -- `[scan] Remove scan_padding feature`

Removed the unused tensor-padding path from scan operations.  This was
originally added to allow static-shaped intermediate tensors across different
tree depths, but it never improved performance and added significant code
complexity.  No throughput change (code path was not active in the benchmark
configuration).

### 2. Replace `torch.dot` with pointwise multiply + sum

**Commit:** `7f255b9e` -- `[compile] Replace torch.dot with pointwise mul+sum`

Under `torch.vmap`, `torch.dot` compiles to `extern_kernels.bmm` (batched
matrix multiply via cuBLAS).  For the tiny vectors in MuJoCo (3-element
positions, quaternion components), each dot product becomes a standalone
cuBLAS call that cannot be fused with surrounding pointwise work.  Replacing
`torch.dot(a, b)` with `(a * b).sum(-1)` keeps everything as pointwise
operations that Inductor can fuse into larger Triton kernels.

This change touched `math.py`, `solver.py`, `smooth.py`, `constraint.py`,
`support.py`, `ray.py`, and all collision files -- every `torch.dot` call site
in the codebase.

### 3. Replace `scatter_add` with gather-based segment sum

**Commit:** `d5d08f40` -- `[scan] Replace scatter_add with gather-based segment sum`

The `body_tree` scan uses `segment_sum` to propagate forces up the kinematic
tree.  The original implementation used `scatter_add`, which compiles to
`tl.atomic_add` operations in Triton.  Atomics serialize concurrent writes and
prevent kernel fusion.

The replacement precomputes inverse segment indices at model-load time and
uses a gather + masked reduction pattern that avoids atomics entirely.  This
is possible because the kinematic tree topology is fixed and known at compile
time.

### 4. Mark model parameters as graph constants

**Commit:** `7f66ae11` -- `[compile] Mark model parameter tensors as static addresses`

Calls `torch._dynamo.mark_static_address` on all tensor fields of the `Model`
object when it is moved to a device.  This tells the compiler that these
tensors will not be reallocated, enabling CUDA graph address reuse and
potential constant folding.
