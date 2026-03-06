# mujoco-torch

A PyTorch port of [MuJoCo MJX](https://mujoco.readthedocs.io/en/stable/mjx.html),
bringing GPU-accelerated physics simulation to the PyTorch ecosystem with full
`torch.compile` and `torch.vmap` support.

## Features

- **Drop-in MJX replacement** -- same physics pipeline (forward dynamics,
  constraints, contacts, sensors) reimplemented in PyTorch.
- **`torch.vmap`** -- batch thousands of environments in a single call with
  automatic vectorisation.
- **`torch.compile`** -- fuse the entire step into optimised GPU kernels; no
  Python overhead at runtime.  Supports `mode="reduce-overhead"` (CUDA graphs)
  for further launch-overhead elimination.
- **Numerically equivalent to MJX** -- verified at float64 precision for every
  step (see `test/mjx_correctness_test.py`).

## Installation

```bash
pip install -e .
```

### Requirements

- Python >= 3.10
- PyTorch (see [PyTorch build](#pytorch-build) below for `torch.compile` support)
- MuJoCo >= 3.0
- tensordict >= 0.11

### PyTorch build

`torch.compile(fullgraph=True)` requires several upstream fixes that are not yet
in a released PyTorch version.  Until they land, build PyTorch from the
[`mujoco-torch-features`](https://github.com/vmoens/pytorch/tree/mujoco-torch-features)
branch:

```bash
git clone --branch mujoco-torch-features --depth 1 \
    https://github.com/vmoens/pytorch.git
cd pytorch
git submodule update --init --recursive --depth 1
pip install -e . --no-build-isolation   # takes ~30-60 min
```

You will also need a source build of
[tensordict](https://github.com/pytorch/tensordict):

```bash
pip install git+https://github.com/pytorch/tensordict.git
```

Without this custom PyTorch build, eager mode and `torch.vmap` work fine; only
`torch.compile(fullgraph=True)` requires the fork.  For
`mode="reduce-overhead"` (CUDA graphs), use the
[`vmoens/nomerg-sum-prs`](https://github.com/vmoens/pytorch/tree/vmoens/nomerg-sum-prs)
branch which includes additional CUDA graph partitioning fixes.

## Quick start

```python
import mujoco
import torch
import mujoco_torch

torch.set_default_dtype(torch.float64)

# Load a MuJoCo model
m_mj = mujoco.MjModel.from_xml_path("humanoid.xml")
mx = mujoco_torch.device_put(m_mj).to("cuda")

# Create initial data and move to GPU
d_mj = mujoco.MjData(m_mj)
dx = mujoco_torch.device_put(d_mj).to("cuda")

# Single step
dx = mujoco_torch.step(mx, dx)

# Batched simulation with vmap
batch_size = 4096
envs = [mujoco_torch.device_put(mujoco.MjData(m_mj)).to("cuda")
        for _ in range(batch_size)]
d_batch = torch.stack(envs)
vmap_step = torch.vmap(lambda d: mujoco_torch.step(mx, d))
d_batch = vmap_step(d_batch)

# Compiled + batched for maximum throughput
compiled_step = torch.compile(vmap_step, fullgraph=True)
d_batch = compiled_step(d_batch)
```

## Feature matrix

| Category | Supported | Not yet supported |
|----------|-----------|-------------------|
| **Integrators** | Euler, RK4, ImplicitFast | Implicit |
| **Solvers** | CG, Newton | PGS |
| **Geom types** | Plane, HField, Sphere, Capsule, Box, Mesh | Ellipsoid, Cylinder |
| **Contact dim** | 1 (frictionless), 3 (frictional), 4 (torsional), 6 (rolling) | — |
| **Friction cone** | Pyramidal, Elliptic | — |
| **Joint types** | Free, Ball, Slide, Hinge | *(all supported)* |
| **Equality constraints** | Connect, Weld, Joint | Tendon, Distance |
| **Actuator dynamics** | None, Integrator, Filter, FilterExact, Muscle | User |
| **Actuator gain** | Fixed, Affine, Muscle | User |
| **Actuator bias** | None, Affine, Muscle | User |
| **Sensors** | 30+ types (position, velocity, acceleration) | CamProjection, Touch, Contact, FrameLinVel/AngVel/LinAcc/AngAcc |
| **Collision pairs** | 12 functions covering all supported geom combinations | — |

## Known limitations

- **Ellipsoid / Cylinder geoms** — no collision functions for these geom types.
- **Tendon / Distance equality constraints** — not yet ported from MJX.
- **PGS solver** — only CG and Newton solvers are available.

## Benchmarks

![Benchmark results](assets/benchmark.png)

Measured on a single NVIDIA H200 GPU, float64 precision, 1 000 simulation
steps per configuration.  Sequential baselines (MuJoCo C, mujoco-torch loop)
are measured at B=1 since they scale linearly.  All values are **steps/second**
(higher is better).

### Humanoid

| Configuration | B=1 | B=128 | B=1 024 | B=4 096 | B=32 768 |
|---|--:|--:|--:|--:|--:|
| MuJoCo C (CPU, sequential) | 38,768 | — | — | — | — |
| mujoco-torch vmap (eager) | 13 | 1,607 | 12,985 | 52,221 | 362,656 |
| mujoco-torch compile | 226 | 26,326 | 207,014 | 724,663 | 1,792,347 |
| mujoco-torch compile (reduce-overhead) | 409 | 44,802 | 311,465 | 900,286 | 1,804,435 |
| **mujoco-torch compile (tuned)** | **232** | **26,001** | **204,207** | **665,163** | **2,443,398** |
| MJX (JAX jit+vmap) | 848 | 108,150 | 868,122 | 2,213,545 | 2,374,941 |

### Ant

| Configuration | B=1 | B=128 | B=1 024 | B=4 096 | B=32 768 |
|---|--:|--:|--:|--:|--:|
| MuJoCo C (CPU, sequential) | 54,150 | — | — | — | — |
| mujoco-torch vmap (eager) | 19 | 2,525 | 19,692 | 79,340 | 321,886 |
| mujoco-torch compile | 278 | 34,199 | 251,982 | 559,149 | 598,566 |
| mujoco-torch compile (reduce-overhead) | 532 | 66,485 | 373,052 | 552,709 | 602,859 |
| **mujoco-torch compile (tuned)** | **314** | **34,930** | **284,986** | **907,434** | **2,123,293** |
| MJX (JAX jit+vmap) | 870 | 105,426 | 648,795 | 881,508 | 921,413 |

### Half-Cheetah

| Configuration | B=1 | B=128 | B=1 024 | B=4 096 | B=32 768 |
|---|--:|--:|--:|--:|--:|
| MuJoCo C (CPU, sequential) | 67,788 | — | — | — | — |
| mujoco-torch vmap (eager) | 18 | 2,325 | 18,202 | 72,760 | 552,424 |
| mujoco-torch compile | 288 | 34,286 | 247,063 | 1,066,654 | 3,100,180 |
| mujoco-torch compile (reduce-overhead) | 427 | 50,790 | 370,647 | 1,227,338 | 3,054,295 |
| **mujoco-torch compile (tuned)** | **285** | **33,623** | **267,320** | **1,074,758** | **3,069,120** |
| MJX (JAX jit+vmap) | 559 | 57,773 | 433,249 | 1,348,620 | 2,753,825 |

### Walker2d

Walker2d uses the RK4 integrator, which makes each step ~3× more expensive
than Euler.

| Configuration | B=1 | B=128 | B=1 024 | B=4 096 | B=32 768 |
|---|--:|--:|--:|--:|--:|
| MuJoCo C (CPU, sequential) | 32,426 | — | — | — | — |
| mujoco-torch vmap (eager) | 4 | 510 | 3,796 | 13,583 | 102,069 |
| mujoco-torch compile | 100 | 9,613 | 68,941 | 262,058 | 656,796 |
| mujoco-torch compile (reduce-overhead) | 170 | 14,992 | 97,277 | 308,591 | 661,354 |
| **mujoco-torch compile (tuned)** | **100** | **9,806** | **70,020** | **258,043** | **795,087** |
| MJX (JAX jit+vmap) | 190 | 12,644 | 89,340 | 263,464 | 404,107 |

### Hopper

Hopper uses the RK4 integrator (like Walker2d).

| Configuration | B=1 | B=128 | B=1 024 | B=4 096 | B=32 768 |
|---|--:|--:|--:|--:|--:|
| MuJoCo C (CPU, sequential) | 81,569 | — | — | — | — |
| mujoco-torch vmap (eager) | 17 | 2,200 | 17,565 | 71,260 | 557,476 |
| mujoco-torch compile | 286 | 32,353 | 260,614 | 991,503 | 3,660,399 |
| mujoco-torch compile (reduce-overhead) | 415 | 33,836 | 286,121 | 1,244,707 | 3,650,605 |
| **mujoco-torch compile (tuned)** | **291** | **33,834** | **262,048** | **1,040,920** | **4,547,076** |
| MJX (JAX jit+vmap) | 817 | 114,045 | 841,375 | 1,024,240 | 8,061,349 |

**"reduce-overhead"** = `torch.compile(mode="reduce-overhead")`, which captures
the compiled graph into CUDA graphs to eliminate kernel launch overhead.
Requires upstream fixes not yet in a released PyTorch version (see
[PyTorch build](#pytorch-build)).

**"tuned"** = Inductor coordinate-descent tile-size tuning + aggressive fusion
enabled (`torch._inductor.config.coordinate_descent_tuning`,
`aggressive_fusion`).  Adds ~40 min extra compile warmup but produces faster
kernels at runtime.

**Methodology.**  Each configuration runs 100 steps after warmup (5 compile
iterations for compiled variants, 1 JIT warmup for MJX).  Wall-clock time is
measured with `torch.cuda.synchronize()` / `jax.block_until_ready()` bracketing.
Steps/s = `batch_size × nsteps / elapsed_time`.  Single GPU
(`CUDA_VISIBLE_DEVICES=0`), dtype=float64.

To reproduce, run the pytest-benchmark suite (requires the PyTorch fork above):

```bash
CUDA_VISIBLE_DEVICES=0 python -m pytest benchmarks/ -v \
    --benchmark-json=bench_results.json
python benchmarks/plot_bench.py bench_results.json -o assets/benchmark.png
```

## Testing

```bash
# Run all tests (requires JAX + MJX for correctness tests)
pip install "jax[cpu]" "mujoco[mjx]"
pytest test/ -x -v
```

## License

Apache 2.0 -- see [LICENSE](LICENSE).

## Acknowledgments

mujoco-torch is a derivative work of
[MuJoCo MJX](https://github.com/google-deepmind/mujoco/tree/main/mjx),
originally developed by Google DeepMind.  See the [NOTICE](NOTICE) file for
attribution details.
