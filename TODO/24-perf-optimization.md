# Performance Optimization: Making torch.compile Faster

## Status

**Current state:** torch.compile works for all 5 models (humanoid, ant,
halfcheetah, walker2d, hopper) with `fullgraph=True`.  The gap with MJX ranges
from 0.7× (walker2d, where torch wins) to 2× (humanoid) at B=32768.

**Branch:** `benchmarks-5-models`

**README status:** The README still shows old benchmark numbers from the first
run (pre-bugfix).  The latest results below supersede them.  The README and
`assets/benchmark.png` need to be updated.

---

## Cluster Setup (steve job 236658)

### Job info

| Field | Value |
|-------|-------|
| Job ID | **236658** |
| GPU | NVIDIA H200 |
| Project root on cluster | `/root/mujoco-torch` |
| Virtual env | `/root/code/.venv/bin/activate` |
| PyTorch | Custom fork at `/root/pytorch` (installed editable) |
| PyTorch branch | `mujoco-torch-features` (fixes cublas float64 vmap bug) |

**WARNING:** Do NOT `pip install torch` or any stock PyTorch.  The custom fork
at `/root/pytorch` is already installed editable in the venv.  Stock PyTorch
has a cublas bug with float64 vmap on H200.

### How to sync code

The reliable way is git push/pull (not `steve sync`, which targets
`periodic-mono`):

```bash
# Local: commit and push
cd /Users/vmoens/repos/mujoco-torch
git add -A && git commit -m "description"
git push origin benchmarks-5-models

# Cluster: pull
steve step 236658 'cd /root/mujoco-torch && git pull origin benchmarks-5-models'
```

If the cluster working tree is dirty:

```bash
steve step 236658 'cd /root/mujoco-torch && git stash && git pull origin benchmarks-5-models'
```

After pulling Python changes, re-install:

```bash
steve step 236658 'cd /root/mujoco-torch && source /root/code/.venv/bin/activate && pip install -e . 2>&1 | tail -3'
```

For copying individual files (when git is overkill):

```bash
steve cp 236658 /Users/vmoens/repos/mujoco-torch/some_file.py :/root/mujoco-torch/some_file.py
```

### How to run benchmarks

Always use `--detach` for long-running commands so you don't block on SSH:

```bash
# Run a single benchmark (example: humanoid compile at all batch sizes)
steve step 236658 'cd /root/mujoco-torch && source /root/code/.venv/bin/activate && \
  CUDA_VISIBLE_DEVICES=0 python -u gpu_bench.py --model humanoid --output bench_humanoid.json' --detach

# Check progress
steve logs 236658 -n 50

# Kill if stuck
steve kill-last-step 236658
```

Or using the pytest-benchmark suite (in `benchmarks/`):

```bash
# All compile benchmarks for all models
steve step 236658 'cd /root/mujoco-torch && source /root/code/.venv/bin/activate && \
  CUDA_VISIBLE_DEVICES=0 python -m pytest benchmarks/bench_compile.py -v \
  --benchmark-json=/tmp/bench_compile.json 2>&1' --detach

# Just humanoid, B=32768
steve step 236658 'cd /root/mujoco-torch && source /root/code/.venv/bin/activate && \
  CUDA_VISIBLE_DEVICES=0 python -m pytest benchmarks/bench_compile.py -v \
  -k "humanoid and 32768" --benchmark-json=/tmp/bench_compile_h32k.json 2>&1' --detach
```

Copy results back:

```bash
steve cp 236658 :/root/mujoco-torch/bench_humanoid.json scratch/bench_humanoid.json
# or
steve cp 236658 :/tmp/bench_compile.json scratch/bench_compile.json
```

### Clearing caches between runs

```bash
steve step 236658 'rm -rf /tmp/torchinductor_root/ /root/.cache/torch/ 2>/dev/null; echo "Cache cleared"'
```

---

## Latest Benchmark Results (Feb 28, 2026)

H200 GPU, float64, `NSTEPS=1000`, `ROUNDS=5` (except vmap which used
`NSTEPS=100`, `ROUNDS=2` for sanity-check ballpark).  All values are
**steps/second** (higher is better).

### Humanoid

| Configuration | B=1 | B=128 | B=1,024 | B=4,096 | B=32,768 |
|---|--:|--:|--:|--:|--:|
| MuJoCo C (CPU, seq) | 59,536 | -- | -- | -- | -- |
| mujoco-torch loop (seq) | 16 | -- | -- | -- | -- |
| mujoco-torch vmap (eager) | 14 | 1,632 | 13,069 | 52,337 | 351,371 |
| **mujoco-torch compile** | **154** | **18,478** | **147,060** | **536,692** | **1,180,109** |
| MJX (JAX jit+vmap) | 870 | 108,905 | 874,432 | 2,237,444 | 2,382,388 |

**Gap at B=32768: compile is 0.50× MJX.**

We want to bridge the gap with Jax. If anything, you should check what jax configuration we are using in these benchmarks if it's comparable to the torch one.

### Ant

| Configuration | B=1 | B=128 | B=1,024 | B=4,096 | B=32,768 |
|---|--:|--:|--:|--:|--:|
| MuJoCo C (CPU, seq) | 101,157 | -- | -- | -- | -- |
| mujoco-torch loop (seq) | 23 | -- | -- | -- | -- |
| mujoco-torch vmap (eager) | 18 | 2,191 | 17,705 | 70,273 | 244,474 |
| **mujoco-torch compile** | **173** | **20,816** | **145,563** | **340,190** | **463,296** |
| MJX (JAX jit+vmap) | 772 | 92,726 | 483,129 | 674,019 | 687,813 |

**Gap at B=32768: compile is 0.67× MJX.**

### Half-Cheetah

| Configuration | B=1 | B=128 | B=1,024 | B=4,096 | B=32,768 |
|---|--:|--:|--:|--:|--:|
| MuJoCo C (CPU, seq) | 166,742 | -- | -- | -- | -- |
| mujoco-torch loop (seq) | 23 | -- | -- | -- | -- |
| mujoco-torch vmap (eager) | 18 | 2,273 | 18,115 | 72,413 | 550,095 |
| **mujoco-torch compile** | **196** | **23,632** | **178,823** | **736,681** | **2,243,328** |
| MJX (JAX jit+vmap) | 569 | 58,191 | 444,864 | 1,408,451 | 2,888,935 |

**Gap at B=32768: compile is 0.78× MJX.**

### Walker2d (RK4)

| Configuration | B=1 | B=128 | B=1,024 | B=4,096 | B=32,768 |
|---|--:|--:|--:|--:|--:|
| MuJoCo C (CPU, seq) | 41,289 | -- | -- | -- | -- |
| mujoco-torch loop (seq) | 6 | -- | -- | -- | -- |
| mujoco-torch vmap (eager) | 5 | 502 | 3,684 | 14,429 | 101,337 |
| **mujoco-torch compile** | **70** | **8,114** | **40,210** | **203,028** | **465,286** |
| MJX (JAX jit+vmap) | 170 | 10,176 | 69,757 | 203,816 | 324,060 |

**torch compile BEATS MJX at B=32768: 1.44×.**

### Hopper (RK4)

| Configuration | B=1 | B=128 | B=1,024 | B=4,096 | B=32,768 |
|---|--:|--:|--:|--:|--:|
| MuJoCo C (CPU, seq) | 63,644 | -- | -- | -- | -- |
| mujoco-torch loop (seq) | 5 | -- | -- | -- | -- |
| mujoco-torch vmap (eager) | 4 | 519 | 4,104 | 16,363 | 126,000 |
| **mujoco-torch compile** | **64** | **7,038** | **49,812** | **176,730** | **571,875** |
| MJX (JAX jit+vmap) | 222 | 21,879 | 180,342 | 525,552 | 1,293,001 |

**Gap at B=32768: compile is 0.44× MJX.**

---

## Analysis: Why is MJX faster?

### The key clue

Walker2d (RK4 = 3× more compute per step) **beats** MJX.  Lighter models
(hopper, halfcheetah, humanoid) lose.  This means:

- **When there's enough arithmetic per step**, torch.compile's kernel quality
  is competitive or better than XLA.
- **When steps are cheap** (Euler integrator, small nv), overhead dominates and
  MJX wins.

The bottleneck is **framework/launch overhead**, not kernel quality.

### Overhead sources in mujoco-torch

1. **53 `.replace()` calls per step** — each creates a shallow copy of an
   80-field TensorClass (`Data`).  Even though tensors are shared, the struct
   copy (dict allocation, key hashing) adds up.

2. **Kernel launch overhead** — without CUDA graphs, each compiled step emits
   dozens of small Triton kernels.  CPU-side dispatch overhead between kernels
   is significant at low compute-per-step.

3. **`_groups_to_device()` in sensor.py** — recursively walks Python dicts
   every step.  Under `torch.compile + vmap` this is traced away (zero runtime
   cost), but it adds tracing complexity.

4. **15+ `torch.vmap(lambda ...)` calls in sensors** — each creates a separate
   kernel.  Many small kernels = many launches.

5. **`while_loop` in solver** — data-dependent iteration count prevents static
   graph optimizations and CUDA graph capture for the solver portion.

6. **30+ `.clone()` calls** in the hot path (solver, smooth, sensor) —
   required for correctness but adds memory traffic.

---

## Optimization Hypotheses

**Strategy:** try each hypothesis one at a time on **humanoid B=32768**, measure
steps/s, compare against the baseline of **1,180,109 steps/s**.

### Baseline command

```bash
steve step 236658 'cd /root/mujoco-torch && source /root/code/.venv/bin/activate && \
  CUDA_VISIBLE_DEVICES=0 python -c "
import torch, mujoco, time, mujoco_torch
import numpy as np
torch.set_default_dtype(torch.float64)
m_mj = mujoco_torch._src.test_util.load_test_file(\"humanoid.xml\")
mx = mujoco_torch.device_put(m_mj).to(\"cuda\")
# warm caches
d0 = mujoco_torch.device_put(mujoco.MjData(m_mj)).to(\"cuda\")
mujoco_torch.step(mx, d0)
torch.cuda.synchronize()
# build batch
B, NSTEPS = 32768, 100
rng = np.random.RandomState(42)
with torch.device(\"cpu\"):
    envs = []
    for i in range(B):
        d = mujoco.MjData(m_mj)
        d.qvel[:] = 0.01 * rng.randn(m_mj.nv)
        envs.append(mujoco_torch.device_put(d))
d_batch = torch.stack([e.to(\"cuda\") for e in envs], dim=0)
# compile
vmap_step = torch.vmap(lambda d: mujoco_torch.step(mx, d))
compiled_fn = torch.compile(vmap_step, fullgraph=True)
for _ in range(5):
    d_batch = compiled_fn(d_batch)
torch.cuda.synchronize()
# benchmark
d_batch2 = torch.stack([e.to(\"cuda\") for e in envs], dim=0)
t0 = time.perf_counter()
for _ in range(NSTEPS):
    d_batch2 = compiled_fn(d_batch2)
torch.cuda.synchronize()
elapsed = time.perf_counter() - t0
sps = B * NSTEPS / elapsed
print(f\"Humanoid B={B} NSTEPS={NSTEPS}: {sps:,.0f} steps/s ({elapsed:.2f}s)\")
" 2>&1' --detach
```

### Hypothesis 1: `mode="reduce-overhead"` (CUDA graphs via compiler)

**What:** Change `torch.compile(vmap_step, fullgraph=True)` to
`torch.compile(vmap_step, fullgraph=True, mode="reduce-overhead")`.

**Why:** `reduce-overhead` enables CUDA graph capture through the compiler.
This replays the entire kernel sequence as a single graph, eliminating
CPU-side kernel launch overhead.  Since our analysis shows the bottleneck is
launch overhead (not kernel quality), this could be a big win.

**Risk:** The `while_loop` in `solver.py` has data-dependent iteration count,
which may prevent CUDA graph capture.  If so, the mode will fall back to
normal compiled execution (no regression, just no improvement).

**Change:**

```python
# Before
compiled_fn = torch.compile(vmap_step, fullgraph=True)

# After
compiled_fn = torch.compile(vmap_step, fullgraph=True, mode="reduce-overhead")
```

**Expected impact:** 10-50%, especially at small batch sizes.

---

### Hypothesis 2: `torch.set_float32_matmul_precision("high")`

**What:** Call `torch.set_float32_matmul_precision("high")` before benchmarking.

**Why:** On Ampere+ GPUs, this enables TF32 for float32 matrix multiplications
(~3× throughput vs IEEE float32).  Even though we run in float64, some
internal reductions and matmul accumulations may use float32 intermediates.

**Change:**

```python
torch.set_float32_matmul_precision("high")
```

**Expected impact:** 0-10%.

---

### Hypothesis 3: Merge sequential `.replace()` calls

**What:** Combine adjacent `.replace()` calls in the hot path.  The biggest
wins are in `smooth.py` (17 calls) and `forward.py` (9 calls).

**Why:** Each `.replace()` calls `clone(recurse=False)` on an 80-field
TensorClass.  Sequential replaces create throwaway intermediate copies.

**Example in `smooth.py:kinematics`:**

```python
# Before (4 calls)
d = d.replace(qpos=qpos, xanchor=xanchor, xaxis=xaxis, xpos=xpos)
d = d.replace(xquat=xquat, xmat=xmat, xipos=xipos, ximat=ximat)
if m.ngeom:
    d = d.replace(geom_xpos=geom_xpos, geom_xmat=geom_xmat)
if m.nsite:
    d = d.replace(site_xpos=site_xpos, site_xmat=site_xmat)

# After (1 call)
kwargs = dict(qpos=qpos, xanchor=xanchor, xaxis=xaxis, xpos=xpos,
              xquat=xquat, xmat=xmat, xipos=xipos, ximat=ximat)
if m.ngeom:
    kwargs.update(geom_xpos=geom_xpos, geom_xmat=geom_xmat)
if m.nsite:
    kwargs.update(site_xpos=site_xpos, site_xmat=site_xmat)
d = d.replace(**kwargs)
```

**Expected impact:** 5-15%.  Under torch.compile, Inductor may already DCE
intermediate copies, so the win may be smaller than in eager mode.

---

### Hypothesis 4: Inductor config tuning

**What:** Set inductor-level knobs for more aggressive fusion and tuning.

**Change:**

```python
import torch._inductor.config
torch._inductor.config.coordinate_descent_tuning = True
torch._inductor.config.aggressive_fusion = True
```

**Expected impact:** 0-15%.  `coordinate_descent_tuning` lets Inductor
auto-tune tile sizes.  `aggressive_fusion` merges more ops into single
kernels, reducing launch count.

---

### Hypothesis 5: In-place mutation instead of `.replace()`

**What:** Replace `d = d.replace(field=val)` with direct attribute
assignment `d.field = val`.  TensorClass supports this.

**Why:** Eliminates the 80-field shallow copy entirely.  Every `.replace()`
allocates a new TensorDict + dict copy.  Direct mutation is O(1).

**Risk:** This changes the functional contract — callers that hold references
to the old `d` will see mutations.  Under `torch.vmap`, each batch element
is independent so this is safe.  Under `torch.compile`, the compiler can
reason about aliasing.  But it's a large refactor and needs careful testing.

**Expected impact:** 10-30%.

---

### Hypothesis 6: Float32 mode

**What:** Add a `dtype=torch.float32` option to `device_put()` and run the
full pipeline in float32.

**Why:** On H200, float64 throughput is 1/2 of float32.  On consumer GPUs
(RTX 4090), it's 1/32.  Many RL workloads don't need float64.  MJX benchmarks
use float64 (`jax_enable_x64=True`) so this wouldn't be an apples-to-apples
comparison, but it's the practical use case for most users.

**Expected impact:** Up to 2× on data-center GPUs, up to 32× on consumer GPUs.

---

### Hypothesis 7: CUDA graphs (manual, sub-block)

**What:** Capture the pre-solver pipeline (position → sensors → velocity →
actuation → acceleration) as a CUDA graph, then run solver in normal mode.

**Why:** The solver's `while_loop` has data-dependent iteration count which
prevents full CUDA graph capture.  But the pre-solver pipeline is
deterministic and can be captured.  The pre-solver is ~60-70% of step time
for simple models.

**Prototype exists:** `gpu_cudagraph_bench.py` already implements this
(Phase 2: sub-block CUDA graphs).  It uses `CudaGraphModule` from
`tensordict.nn` or raw `torch.cuda.CUDAGraph` with `capture_error_mode="relaxed"`.

**Expected impact:** 20-50%.

---

### Hypothesis 8: Fixed-iteration solver

**What:** Replace the `while_loop` convergence check with a fixed iteration
count (e.g., `niter=10`).

**Why:** Makes the solver fully static, enabling CUDA graph capture of the
entire step (not just pre-solver).  MJX effectively does this with JAX's
`while_loop` which compiles to a fixed-trip-count loop.  Many RL workloads
use fixed iteration counts for deterministic compute budgets.

**Expected impact:** Combined with CUDA graphs, could reach 30-60% improvement.

---

### Hypothesis 9: Profile kernel count

**What:** Not an optimization — a diagnostic.  Profile the compiled step to
count how many CUDA kernels are launched per step.

**Command:**

```python
with torch.profiler.profile(
    activities=[torch.profiler.ProfilerActivity.CUDA],
    record_shapes=True,
) as prof:
    for _ in range(10):
        d_batch = compiled_fn(d_batch)
    torch.cuda.synchronize()
print(prof.key_averages().table(sort_by="cuda_time_total", row_limit=50))
prof.export_chrome_trace("/tmp/trace_humanoid_32k.json")
```

Then copy the trace locally:

```bash
steve cp 236658 :/tmp/trace_humanoid_32k.json scratch/trace_humanoid_32k.json
```

Open in `chrome://tracing` or Perfetto to visualize kernel launches.

**Why:** Tells us exactly where time is spent: kernel launches, memory copies,
synchronization points, or actual compute.  This should be done FIRST to
prioritize the other hypotheses.

---

## Execution Plan

Run each hypothesis **one at a time** on humanoid B=32768 (baseline:
1,180,109 steps/s).  Use `NSTEPS=100` for quick iteration, `NSTEPS=1000` for
final numbers.

| Order | Hypothesis | Effort | Expected | Notes |
|-------|-----------|--------|----------|-------|
| 0 | Profile kernel count | 10 min | Diagnostic | Do this first |
| 1 | `mode="reduce-overhead"` | 1 line | 10-50% | Quick win |
| 2 | `set_float32_matmul_precision` | 1 line | 0-10% | Quick win |
| 3 | Merge `.replace()` calls | ~1h | 5-15% | Medium effort |
| 4 | Inductor config tuning | 3 lines | 0-15% | Quick |
| 5 | In-place mutation | Large | 10-30% | Major refactor |
| 6 | Float32 mode | Medium | Up to 2× | New feature |
| 7 | CUDA graphs (sub-block) | Medium | 20-50% | Prototype exists |
| 8 | Fixed-iteration solver | Medium | 30-60% | With CUDA graphs |

Record results in this table:

| Hypothesis | steps/s | Δ vs baseline | Notes |
|-----------|---------|---------------|-------|
| Baseline | 1,180,109 | -- | torch.compile, fullgraph=True |
| H1: reduce-overhead | FAILED | -- | `while_loop` in solver has data-dependent iteration count → CUDA graph capture fails with `RuntimeError: accessing tensor output of CUDAGraphs that has been overwritten`. Need H8 first. |
| H2: matmul precision | ~1,180,000 | ~0% | No effect for float64 workloads (expected). |
| H3: merge replace | (merged into H5) | -- | All `.replace()` calls merged and then converted to `update_()` as part of H5. |
| H4: inductor config | 1,784,014 | +51% | `coordinate_descent_tuning=True`, `aggressive_fusion=True`. Significant win. |
| H5: in-place mutation | (included in Phase 2 baseline) | -- | See Phase 2 results below. |
| H6: float32 | 1,458,757 | +23.5% | `device_put(m, dtype=torch.float32)` — float32 vs float64 baseline |
| H7: sub-block CUDA graphs | (covered by H8+reduce-overhead) | -- | H8 makes the full graph static, so sub-block is unnecessary. |
| H8: fixed-iter solver | 1,260,745 | +6.8% | `fixed_iterations=True`, modest gain on its own |

### Phase 1 Results (bench_phase1.py, humanoid B=32768, NSTEPS=100)

Benchmark script: `bench_phase1.py` (in repo root).

| Config | steps/s | Notes |
|--------|---------|-------|
| Baseline (compile, fullgraph) | 1,056,969 | Lower than the table above due to NSTEPS=100 vs 1000 |
| H1: reduce-overhead | FAILED | CUDA graph capture fails on `while_loop` |
| H2: matmul precision | 1,088,143 | ~3% (noise) |
| H4: inductor config | 1,784,014 | **+68.8%** — the biggest quick win |

### Phase 2 Results (bench_phase2.py, humanoid B=32768, NSTEPS=100)

Benchmark script: `bench_phase2.py` (in repo root).  Run on H200 GPU, Mar 1
2026, after DeviceCopy fix (commit 83aa5fa).

| Config | steps/s | Δ vs baseline | Notes |
|--------|---------|---------------|-------|
| Baseline (H3+H5 active) | 1,180,877 | -- | In-place mutation already active |
| H4: inductor config | 2,246,707 | **+90.3%** | `coordinate_descent_tuning + aggressive_fusion` |
| H8: fixed_iterations | 1,260,745 | +6.8% | Modest gain alone |
| H8 + reduce-overhead | FAILED | -- | CUDA graph tensor aliasing in `scan.py:_take()` |
| H4+H8 + reduce-overhead | FAILED | -- | Same aliasing issue |
| H6: float32 | 1,458,757 | +23.5% | Float32 vs float64 baseline |
| ALL f64 (H4+H8+reduce-overhead) | FAILED | -- | Same aliasing issue |
| ALL f32 (H4+H6+H8+reduce-overhead) | FAILED | -- | Same aliasing issue |

**Key findings:**
- H4 (inductor config) remains the biggest win (+90%).
- H8 (fixed_iterations) alone gives +6.8%.
- H6 (float32) gives +23.5% over f64 baseline (without H4).
- `reduce-overhead` (CUDA graphs) still fails — not due to DeviceCopy
  (which was fixed) but due to **tensor aliasing** in `scan.py:_take()`
  (`x[i]` returns a view that CUDA graphs considers "overwritten").
  Fix: clone the output in `_take()`, or call
  `torch.compiler.cudagraph_mark_step_begin()` before each compiled call.

---

## Code Changes Made

### H3+H5: In-place mutation (`update_()`)

**Files modified:** `dataclasses.py`, `smooth.py`, `forward.py`, `solver.py`,
`constraint.py`, `sensor.py`, `passive.py`, `collision_driver.py`

- Added `MjTensorClass.update_(**kwargs)` method — zero-copy in-place field
  update (no `clone(recurse=False)` allocation).
- Converted all `d = d.replace(...)` calls in the hot path to `d.update_(...)`.
- Added `d = d.clone(recurse=False)` at the top of `step()` to preserve the
  caller's `d` (functional contract).
- Added `d_t0 = d.clone(recurse=False)` at the top of `_rungekutta4` to fix
  an aliasing bug where in-place updates corrupted the saved initial state.
- Left `dh = d.replace(qM=qM)` in `_euler` as-is since it creates a distinct
  object needed for the integration step.
- Calls on non-Data objects (`info1`, `info2`, `h_info`) left as `.replace()`.

**Correctness:** Verified locally (CPU, all 5 models, 10 steps) — max qpos
diff < 1e-10 for all models.

### H6: Float32 mode

**Files modified:** `device.py`

- Added `dtype: torch.dtype | None = None` parameter to `device_put()`.
- Added `_cast_float(v, dtype)` helper that casts floating-point tensors.
- Usage: `mx = mujoco_torch.device_put(m_mj, dtype=torch.float32).to("cuda")`

### H8: Fixed-iteration solver

**Files modified:** `solver.py`, `forward.py`

- Added `fixed_loop(body_fn, carried_inputs, n_iter)` in `solver.py` — a
  static-iteration-count alternative to `while_loop` that enables CUDA graph
  capture.
- Added `fixed_iterations: bool = False` parameter to `solve()`, `step()`,
  `forward()`, and `_rungekutta4()`.
- When `fixed_iterations=True`, the solver uses `fixed_loop` with the model's
  `opt.iterations` count instead of `while_loop` with convergence check.

### DeviceCopy fix (COMPLETED)

**Phase 1 files:** `scan.py`, `types.py`
**Phase 2 files:** `types.py`, `device.py`, `smooth.py`, `constraint.py`,
`collision_driver.py`, `sensor.py`, `ray.py`, `test/constraint_test.py`

**Phase 1 (previous):**
- Added `_resolve_cached_tensors()` and `warm_device_caches()` in `scan.py`.
- Overrode `Model.to()` in `types.py` to pre-warm `_DeviceCachedTensor` caches.

**Phase 2 (commit 83aa5fa):**
Implemented **Option A** from the original plan: store all precomputed index
data as plain device tensors in `model._device_precomp` (a regular Python
dict, NOT a TensorDict field).  All consuming code reads from `_device_precomp`
instead of calling `.to(device)` at runtime.

Changes:
- `types.py`: Added `_build_device_precomp()` which recursively resolves all
  precomputed structures via `_resolve_cached_tensors()` and stores in
  `model._device_precomp`.  Called from both `_model_to()` (for GPU) and
  `device_put()` (for CPU).  Added `_model_clone()` override to propagate
  `_device_precomp` across `clone()` / `replace()` / `tree_replace()`.
- `smooth.py`: `factor_m` and `solve_m` read from `_device_precomp`, no `.to()`.
- `constraint.py`: All `_instantiate_*` functions read from `_device_precomp`.
- `collision_driver.py`: `collision()` reads from `_device_precomp`.
- `sensor.py`: `sensor_pos/vel/acc` read from `_device_precomp`, removed
  `_groups_to_device()` calls.
- `ray.py`: `ray_precomputed` no longer calls `.to(device)` on index tensors.

**Status: COMPLETE.** DeviceCopy warnings reduced from ~20+ to 8 (remaining
8 are from `_take()` in scan.py where `idx.to(x.device)` is a no-op but still
traced).  The DeviceCopy fix alone doesn't enable `reduce-overhead` — a
separate **CUDA graph tensor aliasing** issue in `scan.py:_take()` blocks it
(see Phase 2 results above).

### Remaining blocker: CUDA graph tensor aliasing

`mode="reduce-overhead"` fails with:
> Error: accessing tensor output of CUDAGraphs that has been overwritten

The error occurs in `scan.py:flat()` → `_take(ys[i], reorder_indices[typ])`
where `x[i]` returns a view (not a copy) of the output tensor.  CUDA graphs
require outputs to not be aliased between successive replays.

**Fix options:**
1. Clone the output in `_take()` for the reorder step (`_take` already clones
   elsewhere in some paths).
2. Call `torch.compiler.cudagraph_mark_step_begin()` before each compiled
   function invocation in the benchmark/user code.
3. Investigate whether the Inductor can automatically insert the clone.

---

## Completed: README + Figure

The README already contains the correct benchmark numbers for all 5 models
(humanoid, ant, halfcheetah, walker2d, hopper) from the Feb 28 run.

## Completed: Phase 2 Benchmarks

Phase 2 benchmarks ran on Mar 1, 2026.  Results recorded above.

## Next Steps

1. **Fix CUDA graph tensor aliasing** in `scan.py:_take()` to enable
   `mode="reduce-overhead"` (see options above).
2. **Investigate remaining 8 DeviceCopy warnings** — likely from `_take()`
   line 266 where `idx.to(x.device)` is traced even though it's a no-op.
3. **Run full benchmark suite** (all 5 models, all batch sizes) with H4
   inductor config to get updated comparison numbers vs MJX.
4. **Update README** with Phase 2 optimization results once reduce-overhead
   is working.
