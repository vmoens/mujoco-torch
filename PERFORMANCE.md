# Compile Performance Optimizations

This document tracks `torch.compile` throughput improvements in mujoco-torch,
measured cumulatively as each optimization is stacked on top of the previous
ones.

## Methodology

- **Hardware:** Single NVIDIA H200 GPU
- **Precision:** float64
- **Batch size:** 32 768
- **Benchmark:** `test_compile` (base `torch.compile`, no tuning flags)
- **Procedure:** For each cumulative commit, the Inductor cache was cleared and
  the full benchmark suite was re-run from scratch (5 models, 1 000 steps each
  after warmup).

## Baseline

The baseline is the `test_compile` configuration from the README, measured
before any of the optimizations below.

| Model | steps/s |
|---|---:|
| humanoid | 1,220,458 |
| ant | 500,902 |
| halfcheetah | 2,370,231 |
| walker2d | 505,346 |
| hopper | 2,899,494 |

## Optimizations

Each section below describes one commit.  Results are **cumulative** -- each
row includes all preceding optimizations.

### 1. Remove `scan_padding` feature

**Commit:** `df54c6ce` -- `[scan] Remove scan_padding feature`

Removed the unused tensor-padding path from scan operations.  This was
originally added to allow static-shaped intermediate tensors across different
tree depths, but it never improved performance and added significant code
complexity.  No throughput change expected (code path was not active in the
benchmark configuration).

*Benchmark skipped -- no functional change in the measured code path.*

### 2. Replace `matmul_unroll` with `torch.matmul`

**Commit:** `6c916f03` -- `[math] Replace matmul_unroll loop with torch.matmul`

The original `matmul_unroll` implemented matrix multiplication as a
triple-nested Python loop over individual scalar elements.  While this produced
many small fused pointwise kernels under Inductor, it also generated a very
large computation graph (hundreds of nodes per call).  Replacing it with
`torch.matmul` drastically reduces graph size and compile time, but delegates
to cuBLAS which has higher per-call overhead for the tiny matrices (3x3, 6x6)
used throughout the physics pipeline.

| Model | steps/s | vs baseline |
|---|---:|---:|
| humanoid | 1,057,683 | -13.3% |
| ant | 452,908 | -9.6% |
| halfcheetah | 2,049,063 | -13.5% |
| walker2d | 448,267 | -11.3% |
| hopper | 2,390,025 | -17.6% |

**Verdict:** Runtime regression due to cuBLAS dispatch overhead on tiny
matrices.  The trade-off is worthwhile when combined with later optimizations
that reduce the total kernel count, and compile time is significantly shorter.

### 3. Vectorize quaternion operations

**Commit:** `7cb06b15` -- `[math] Vectorize quaternion ops to reduce graph node count`

Rewrote `quat_mul`, `quat_mul_axis`, and `quat_to_mat` to use tensor slicing
and vectorized operations (`torch.cross`, `torch.cat`, `.sum(-1)`) instead of
individual scalar indexing and `torch.stack`.  This reduces the number of graph
nodes per quaternion call and improves fusion potential.

| Model | steps/s | vs baseline | vs prev |
|---|---:|---:|---:|
| humanoid | 1,060,012 | -13.1% | +0.2% |
| ant | 447,752 | -10.6% | -1.1% |
| halfcheetah | 2,076,627 | -12.4% | +1.3% |
| walker2d | 450,543 | -10.8% | +0.5% |
| hopper | 2,423,625 | -16.4% | +1.4% |

**Verdict:** Roughly neutral at runtime.  The primary benefit is reduced graph
size and compile time rather than throughput.

### 4. Replace `torch.dot` with pointwise multiply + sum

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

| Model | steps/s | vs baseline | vs prev |
|---|---:|---:|---:|
| humanoid | 1,366,811 | +12.0% | +29.0% |
| ant | 564,431 | +12.7% | +26.1% |
| halfcheetah | 2,667,298 | +12.5% | +28.4% |
| walker2d | 572,260 | +13.2% | +27.0% |
| hopper | 2,610,214 | -10.0% | +7.7% |

**Verdict:** The single most impactful optimization.  Eliminating unfused
cuBLAS calls reduced total kernel count dramatically and allowed Inductor to
produce larger, more efficient fused kernels.  The +12-13% cumulative
improvement over baseline for most models is entirely driven by this change,
which more than compensates for the regression introduced in commits 2-3.

### 5. Replace `scatter_add` with gather-based segment sum

**Commit:** `d5d08f40` -- `[scan] Replace scatter_add with gather-based segment sum`

The `body_tree` scan uses `segment_sum` to propagate forces up the kinematic
tree.  The original implementation used `scatter_add`, which compiles to
`tl.atomic_add` operations in Triton.  Atomics serialize concurrent writes and
prevent kernel fusion.

The replacement precomputes inverse segment indices at model-load time and
uses a gather + masked reduction pattern that avoids atomics entirely.  This
is possible because the kinematic tree topology is fixed and known at compile
time.

| Model | steps/s | vs baseline | vs prev |
|---|---:|---:|---:|
| humanoid | 1,384,803 | +13.5% | +1.3% |
| ant | 566,415 | +13.1% | +0.4% |
| halfcheetah | 2,703,400 | +14.1% | +1.4% |
| walker2d | 577,341 | +14.2% | +0.9% |
| hopper | 3,390,005 | +16.9% | +29.9% |

**Verdict:** Modest improvement for most models (+1-2% incremental), but a
large jump for hopper (+30% incremental, bringing cumulative to +17% over
baseline).  Hopper benefits disproportionately because its kinematic tree has
a linear chain topology where every body has exactly one child -- the
gather-based approach handles this structure much more efficiently than
atomic scatter.

### 6. Mark model parameters as graph constants

**Commit:** `7f66ae11` -- `[compile] Mark model parameter tensors as static addresses`

Calls `torch._dynamo.mark_static_address` on all tensor fields of the `Model`
object when it is moved to a device.  This tells the compiler that these
tensors will not be reallocated, enabling CUDA graph address reuse and
potential constant folding.

| Model | steps/s | vs baseline | vs prev |
|---|---:|---:|---:|
| humanoid | 1,377,400 | +12.9% | -0.5% |
| ant | 566,661 | +13.1% | +0.0% |
| halfcheetah | 2,695,067 | +13.7% | -0.3% |
| walker2d | 577,810 | +14.3% | +0.1% |
| hopper | 3,395,581 | +17.1% | +0.2% |

**Verdict:** Within measurement noise for `test_compile` mode.  The primary
benefit is expected under `mode="reduce-overhead"` (CUDA graphs), where
stable tensor addresses prevent graph re-capture.

## Summary

Final cumulative results (all optimizations applied) vs baseline:

| Model | Baseline | Optimized | Change |
|---|---:|---:|---:|
| humanoid | 1,220,458 | 1,377,400 | **+12.9%** |
| ant | 500,902 | 566,661 | **+13.1%** |
| halfcheetah | 2,370,231 | 2,695,067 | **+13.7%** |
| walker2d | 505,346 | 577,810 | **+14.3%** |
| hopper | 2,899,494 | 3,395,581 | **+17.1%** |

The dominant contributor is **replacing `torch.dot` with pointwise mul+sum**
(commit 4), which alone accounts for the bulk of the improvement by enabling
Inductor to fuse dot-product computations into surrounding kernels instead of
dispatching standalone cuBLAS calls.  The **scatter_add replacement** (commit 5)
provides an additional meaningful boost, especially for models with linear
kinematic chains.
