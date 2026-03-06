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
| mujoco-torch vmap (eager) | 14 | 1,700 | 13,494 | 53,027 | 334,058 |
| mujoco-torch compile | 234 | 25,784 | 200,706 | 676,553 | 1,477,151 |
| mujoco-torch compile (reduce-overhead) | 396 | 42,240 | 294,420 | 701,833 | 1,432,973 |
| **mujoco-torch compile (tuned)** | **232** | **25,989** | **184,480** | **761,727** | **1,896,593** |
| MJX (JAX jit+vmap) | 859 | 102,978 | 838,698 | 2,209,482 | 2,375,375 |

### Ant

| Configuration | B=1 | B=128 | B=1 024 | B=4 096 | B=32 768 |
|---|--:|--:|--:|--:|--:|
| MuJoCo C (CPU, sequential) | 54,150 | — | — | — | — |
| mujoco-torch vmap (eager) | 21 | 2,683 | 20,908 | 81,584 | 259,828 |
| mujoco-torch compile | 280 | 33,580 | 259,898 | 517,094 | 431,315 |
| mujoco-torch compile (reduce-overhead) | 525 | 66,201 | 356,275 | 484,570 | 573,536 |
| **mujoco-torch compile (tuned)** | **304** | **35,880** | **272,681** | **900,759** | **1,647,590** |
| MJX (JAX jit+vmap) | 886 | 85,786 | 617,852 | 884,782 | 922,100 |

### Half-Cheetah

| Configuration | B=1 | B=128 | B=1 024 | B=4 096 | B=32 768 |
|---|--:|--:|--:|--:|--:|
| MuJoCo C (CPU, sequential) | 67,788 | — | — | — | — |
| mujoco-torch vmap (eager) | 19 | 2,452 | 19,298 | 74,569 | 536,138 |
| mujoco-torch compile | 290 | 35,632 | 267,630 | 1,091,756 | 2,699,855 |
| mujoco-torch compile (reduce-overhead) | 333 | 49,132 | 345,490 | 1,096,945 | 2,675,193 |
| **mujoco-torch compile (tuned)** | **247** | **34,163** | **274,016** | **926,797** | **2,740,635** |
| MJX (JAX jit+vmap) | 522 | 55,009 | 412,155 | 1,354,950 | 2,605,561 |

### Walker2d

Walker2d uses the RK4 integrator, which makes each step ~3× more expensive
than Euler.

| Configuration | B=1 | B=128 | B=1 024 | B=4 096 | B=32 768 |
|---|--:|--:|--:|--:|--:|
| MuJoCo C (CPU, sequential) | 32,426 | — | — | — | — |
| mujoco-torch vmap (eager) | 5 | 516 | 3,796 | 14,769 | 104,954 |
| mujoco-torch compile | 98 | 7,002 | 70,690 | 266,199 | 577,499 |
| mujoco-torch compile (reduce-overhead) | 165 | 7,936 | 82,936 | 263,704 | 579,711 |
| **mujoco-torch compile (tuned)** | **100** | **9,926** | **70,275** | **257,977** | **671,765** |
| MJX (JAX jit+vmap) | 193 | 10,515 | 89,691 | 221,813 | 408,517 |

### Hopper

Hopper uses the RK4 integrator (like Walker2d).

| Configuration | B=1 | B=128 | B=1 024 | B=4 096 | B=32 768 |
|---|--:|--:|--:|--:|--:|
| MuJoCo C (CPU, sequential) | 81,569 | — | — | — | — |
| mujoco-torch vmap (eager) | 19 | 2,307 | 18,257 | 72,910 | 562,866 |
| mujoco-torch compile | 279 | 33,411 | 206,603 | 973,214 | 3,318,016 |
| mujoco-torch compile (reduce-overhead) | 376 | 49,411 | 361,234 | 1,133,846 | 3,253,214 |
| **mujoco-torch compile (tuned)** | **221** | **33,568** | **268,589** | **1,020,036** | **2,862,786** |
| MJX (JAX jit+vmap) | 921 | 119,865 | 854,200 | 1,128,406 | 8,097,292 |

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
