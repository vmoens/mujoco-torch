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
  Python overhead at runtime.
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
`torch.compile(fullgraph=True)` requires the fork.

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
| MuJoCo C (CPU, sequential) | 45,622 | — | — | — | — |
| mujoco-torch vmap (eager) | 14 | 1,698 | 13,522 | 53,373 | 376,740 |
| mujoco-torch compile | 239 | 26,588 | 208,210 | 729,306 | 2,018,703 |
| **mujoco-torch compile (tuned)** | **216** | **25,064** | **191,248** | **738,973** | **2,461,347** |
| MJX (JAX jit+vmap) | 880 | 112,841 | 856,331 | 2,214,655 | 2,376,552 |

### Ant

| Configuration | B=1 | B=128 | B=1 024 | B=4 096 | B=32 768 |
|---|--:|--:|--:|--:|--:|
| MuJoCo C (CPU, sequential) | 68,369 | — | — | — | — |
| mujoco-torch vmap (eager) | 17 | 2,195 | 17,312 | 69,316 | 331,964 |
| mujoco-torch compile | 318 | 35,224 | 280,284 | 664,325 | 753,962 |
| **mujoco-torch compile (tuned)** | **296** | **34,391** | **270,840** | **914,892** | **2,203,204** |
| MJX (JAX jit+vmap) | 852 | 103,651 | 652,280 | 880,360 | 923,870 |

### Half-Cheetah

| Configuration | B=1 | B=128 | B=1 024 | B=4 096 | B=32 768 |
|---|--:|--:|--:|--:|--:|
| MuJoCo C (CPU, sequential) | 90,617 | — | — | — | — |
| mujoco-torch vmap (eager) | 14 | 1,838 | 14,479 | 58,947 | 461,065 |
| mujoco-torch compile | 228 | 26,062 | 196,898 | 812,800 | 3,402,872 |
| **mujoco-torch compile (tuned)** | **205** | **23,247** | **177,522** | **714,743** | **3,577,821** |
| MJX (JAX jit+vmap) | 570 | 57,199 | 432,451 | 1,360,843 | 2,767,736 |

### Walker2d

Walker2d uses the RK4 integrator, which makes each step ~3× more expensive
than Euler.

| Configuration | B=1 | B=128 | B=1 024 | B=4 096 | B=32 768 |
|---|--:|--:|--:|--:|--:|
| MuJoCo C (CPU, sequential) | 37,081 | — | — | — | — |
| mujoco-torch vmap (eager) | 4 | 413 | 3,024 | 11,631 | 85,794 |
| mujoco-torch compile | 80 | 6,682 | 48,246 | 185,602 | 730,468 |
| **mujoco-torch compile (tuned)** | **71** | **6,235** | **44,274** | **165,182** | **817,718** |
| MJX (JAX jit+vmap) | 188 | 12,753 | 90,246 | 264,193 | 402,604 |

### Hopper

Hopper uses the RK4 integrator (like Walker2d).

| Configuration | B=1 | B=128 | B=1 024 | B=4 096 | B=32 768 |
|---|--:|--:|--:|--:|--:|
| MuJoCo C (CPU, sequential) | 119,976 | — | — | — | — |
| mujoco-torch vmap (eager) | 17 | 2,074 | 16,584 | 65,921 | 524,852 |
| mujoco-torch compile | 313 | 35,441 | 269,661 | 1,059,939 | 4,557,454 |
| **mujoco-torch compile (tuned)** | **286** | **33,658** | **257,049** | **1,035,760** | **3,474,654** |
| MJX (JAX jit+vmap) | 945 | 115,509 | 836,522 | 977,780 | 7,762,885 |

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
