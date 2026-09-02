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

## CPU comparison: compiled mujoco-torch vs MJX (MicroDuck, Apple M5 Max)

Measured 2026-09-02 with `examples/bench_backends.py` on an Apple M5 Max
(6 performance + 12 efficiency cores, 64 GB), float64, torch
`2.15.0.dev20260901` (6 intra-op threads, `fullgraph=True`), tensordict main,
JAX 0.11.1 / mujoco-mjx 3.12 on the XLA CPU backend.  The model is the MicroDuck
`scene_walk.xml` with its collision meshes replaced by `fitaabb` boxes
(`--env microduck`), 200 Hz physics.  Two passes; torch numbers vary ±10–15 %
between passes, MJX up to ~1.7× (XLA's thread pool on the hybrid cores), so
differences under ~15 % are noise.

| Configuration | B=1 | B=16 | B=128 | B=1 024 |
|---|--:|--:|--:|--:|
| MuJoCo C (sequential) | 91 836 | | | |
| mujoco-torch vmap (eager) | | 267 | 773 | 1 394 |
| mujoco-torch compile | | 880 | 4 488 | 11 187 |
| mujoco-torch compile (max-autotune-no-cudagraphs) | | 1 072 | 4 593 | 11 778 |
| mujoco-torch compile (tuned) | | 963 | 4 510 | 11 687 |
| MJX (JAX jit+vmap) | | 3 125 | 6 155 | 8 610 |

The original run collected RSS in a shared process, so those memory figures are
omitted.  The benchmark now isolates each configuration in a fresh subprocess.
First-call compile was 57–72 s for every torch mode and batch size (each batch
size is a new compile); MJX was 1–4 s.

Reading: compile gives 4–11× over eager on CPU.  MJX leads by 3.5× at B=16 and
1.4× at B=128; the compiled torch step overtakes it around B=1 024.  Profiling
(`torch.profiler`, B=1 024) shows ~12 giant fused C++ kernels whose OpenMP
regions parallelise over the batch dimension only, plus 21 ATen fallbacks per
step (`bmm` ×10, `linalg_cholesky_ex` / `cholesky_solve` ×4 each, `argsort`
×2, `cumsum` ×2); the batched LAPACK Cholesky loop issues ~29 dispatcher calls
per matrix.  The fixed per-step cost is ~16 ms for torch vs ~8 ms for MJX,
while the marginal cost per environment is lower for torch, which explains the
crossover.

### CPU experiments that did **not** help

All variants were verified against the baseline in eager before timing.

| Variant | B=128 | B=1 024 | Notes |
|---|--:|--:|---|
| Unrolled scan groups (static selects instead of the inner per-group `vmap`) | 3 020 | 11 242 | +15 % compile, +240 MB; Inductor already fuses the vmapped group |
| Fused broadcast-multiply-sum instead of the 10 `bmm` fallbacks | 4 444 | 10 891 | bit-identical; Accelerate's batched GEMM is already efficient |
| Inline (unrolled) Cholesky up to n=32 instead of LAPACK for nv=20 | 1 876 | 9 432 | 3× compile time, +700 MB; keep the threshold at 16 |
| 18 threads instead of 6 | — | 12 564 vs 12 516 | efficiency cores add nothing |
| `torch.compiler.set_stance(skip_guard_eval_unsafe=True)` | slower | slower | guard evaluation is only ~0.5 % of a step |

Remaining levers on CPU are structural: the ~29k per-matrix dispatcher calls of
the batched LAPACK path, the serial (non-batch) loops inside the fused kernels,
and the Python-level dispatch of the solver `while_loop` body per Newton
iteration (not exercised by this benchmark's resting initial state).
