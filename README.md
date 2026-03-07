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
| MuJoCo C (CPU, sequential) | 44,775 | — | — | — | — |
| mujoco-torch vmap (eager) | 4 | 502 | 4,008 | 16,087 | 126,052 |
| mujoco-torch compile | 69 | 9,471 | 69,803 | 281,114 | 1,278,195 |
| mujoco-torch compile (reduce-overhead) | 241 | 27,801 | 148,167 | 435,563 | 1,356,693 |
| **mujoco-torch compile (tuned)** | **69** | **9,145** | **66,770** | **230,838** | **1,468,430** |
| MJX (JAX jit+vmap) | 829 | 108,802 | 795,243 | 2,205,810 | 2,374,166 |

### Ant

| Configuration | B=1 | B=128 | B=1 024 | B=4 096 | B=32 768 |
|---|--:|--:|--:|--:|--:|
| MuJoCo C (CPU, sequential) | 67,640 | — | — | — | — |
| mujoco-torch vmap (eager) | 16 | 2,127 | 16,482 | 66,939 | 325,254 |
| mujoco-torch compile | 301 | 35,379 | 268,247 | 662,696 | 751,673 |
| mujoco-torch compile (reduce-overhead) | 534 | 67,500 | 427,404 | 655,730 | 758,120 |
| **mujoco-torch compile (tuned)** | **307** | **35,292** | **288,701** | **939,155** | **2,246,599** |
| MJX (JAX jit+vmap) | 852 | 103,651 | 652,280 | 880,360 | 923,870 |

### Half-Cheetah

| Configuration | B=1 | B=128 | B=1 024 | B=4 096 | B=32 768 |
|---|--:|--:|--:|--:|--:|
| MuJoCo C (CPU, sequential) | 91,580 | — | — | — | — |
| mujoco-torch vmap (eager) | 13 | 1,747 | 13,730 | 56,036 | 444,710 |
| mujoco-torch compile | 227 | 23,096 | 194,794 | 717,623 | 3,325,032 |
| mujoco-torch compile (reduce-overhead) | 15 | 38,993 | 306,344 | 1,096,316 | 3,413,237 |
| **mujoco-torch compile (tuned)** | **220** | **24,757** | **167,829** | **650,264** | **3,811,698** |
| MJX (JAX jit+vmap) | 570 | 57,199 | 432,451 | 1,360,843 | 2,767,736 |

### Walker2d

Walker2d uses the RK4 integrator, which makes each step ~3× more expensive
than Euler.

| Configuration | B=1 | B=128 | B=1 024 | B=4 096 | B=32 768 |
|---|--:|--:|--:|--:|--:|
| MuJoCo C (CPU, sequential) | 37,466 | — | — | — | — |
| mujoco-torch vmap (eager) | 3 | 396 | 2,859 | 11,350 | 83,566 |
| mujoco-torch compile | 76 | 5,777 | 47,453 | 167,563 | 726,801 |
| mujoco-torch compile (reduce-overhead) | 149 | 10,970 | 59,244 | 236,116 | 741,581 |
| **mujoco-torch compile (tuned)** | **77** | **6,703** | **46,774** | **178,882** | **805,304** |
| MJX (JAX jit+vmap) | 188 | 12,753 | 90,246 | 264,193 | 402,604 |

### Hopper

Hopper uses the RK4 integrator (like Walker2d).

| Configuration | B=1 | B=128 | B=1 024 | B=4 096 | B=32 768 |
|---|--:|--:|--:|--:|--:|
| MuJoCo C (CPU, sequential) | 117,643 | — | — | — | — |
| mujoco-torch vmap (eager) | 15 | 1,943 | 15,906 | 62,498 | 500,400 |
| mujoco-torch compile | 300 | 36,006 | 272,703 | 887,319 | 4,465,498 |
| mujoco-torch compile (reduce-overhead) | 413 | 52,371 | 376,009 | 1,291,346 | 4,576,383 |
| **mujoco-torch compile (tuned)** | **315** | **35,105** | **273,201** | **1,038,663** | **5,024,413** |
| MJX (JAX jit+vmap) | 945 | 115,509 | 836,522 | 977,780 | 7,762,885 |

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
