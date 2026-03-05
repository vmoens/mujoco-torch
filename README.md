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
| mujoco-torch vmap (eager) | 13 | 1,600 | 12,715 | 50,696 | 354,480 |
| mujoco-torch compile | 177 | 20,182 | 159,639 | 583,233 | 1,220,458 |
| mujoco-torch compile (reduce-overhead) | 324 | 37,789 | 253,681 | 687,005 | 1,226,029 |
| **mujoco-torch compile (tuned)** | **213** | **24,221** | **191,369** | **720,455** | **2,377,730** |
| MJX (JAX jit+vmap) | 733 | 87,762 | 723,548 | 2,217,927 | 2,356,737 |

### Ant

| Configuration | B=1 | B=128 | B=1 024 | B=4 096 | B=32 768 |
|---|--:|--:|--:|--:|--:|
| MuJoCo C (CPU, sequential) | 54,150 | — | — | — | — |
| mujoco-torch vmap (eager) | 20 | 2,486 | 19,654 | 77,646 | 302,672 |
| mujoco-torch compile | 230 | 26,616 | 208,246 | 465,209 | 500,902 |
| mujoco-torch compile (reduce-overhead) | 467 | 55,494 | 318,338 | 466,442 | 507,136 |
| **mujoco-torch compile (tuned)** | **284** | **32,750** | **231,494** | **798,020** | **2,118,328** |
| MJX (JAX jit+vmap) | 837 | 106,517 | 651,434 | 884,929 | 921,976 |

### Half-Cheetah

| Configuration | B=1 | B=128 | B=1 024 | B=4 096 | B=32 768 |
|---|--:|--:|--:|--:|--:|
| MuJoCo C (CPU, sequential) | 67,788 | — | — | — | — |
| mujoco-torch vmap (eager) | 18 | 2,303 | 18,325 | 73,201 | 546,334 |
| mujoco-torch compile | 228 | 26,862 | 209,306 | 832,186 | 2,370,231 |
| mujoco-torch compile (reduce-overhead) | 353 | 1,911 | 227,924 | 758,576 | 2,369,552 |
| **mujoco-torch compile (tuned)** | **270** | **31,318** | **248,055** | **980,404** | **3,656,830** |
| MJX (JAX jit+vmap) | 536 | 56,942 | 435,192 | 1,360,091 | 2,777,267 |

### Walker2d

Walker2d uses the RK4 integrator, which makes each step ~3× more expensive
than Euler.

| Configuration | B=1 | B=128 | B=1 024 | B=4 096 | B=32 768 |
|---|--:|--:|--:|--:|--:|
| MuJoCo C (CPU, sequential) | 32,426 | — | — | — | — |
| mujoco-torch vmap (eager) | 5 | 507 | 3,806 | 14,731 | 104,097 |
| mujoco-torch compile | 71 | 6,984 | 50,451 | 197,578 | 505,346 |
| mujoco-torch compile (reduce-overhead) | 140 | 12,396 | 66,686 | 213,618 | 511,125 |
| **mujoco-torch compile (tuned)** | **88** | **8,251** | **62,986** | **240,192** | **776,320** |
| MJX (JAX jit+vmap) | 189 | 12,951 | 90,568 | 265,329 | 408,925 |

### Hopper

Hopper uses the RK4 integrator (like Walker2d).

| Configuration | B=1 | B=128 | B=1 024 | B=4 096 | B=32 768 |
|---|--:|--:|--:|--:|--:|
| MuJoCo C (CPU, sequential) | 81,569 | — | — | — | — |
| mujoco-torch vmap (eager) | 18 | 2,164 | 17,467 | 70,061 | 545,301 |
| mujoco-torch compile | 221 | 25,355 | 175,043 | 773,022 | 2,899,494 |
| mujoco-torch compile (reduce-overhead) | 430 | 20,722 | 253,076 | 1,004,796 | 2,914,394 |
| **mujoco-torch compile (tuned)** | **271** | **30,605** | **240,384** | **925,299** | **4,447,838** |
| MJX (JAX jit+vmap) | 922 | 114,482 | 806,620 | 2,707,949 | 8,036,134 |

**"reduce-overhead"** = `torch.compile(mode="reduce-overhead")`, which captures
the compiled graph into CUDA graphs to eliminate kernel launch overhead.
Requires upstream fixes not yet in a released PyTorch version (see
[PyTorch build](#pytorch-build)).

**"tuned"** = Inductor coordinate-descent tile-size tuning + aggressive fusion
enabled (`torch._inductor.config.coordinate_descent_tuning`,
`aggressive_fusion`).  Adds ~40 min extra compile warmup but produces faster
kernels at runtime.

**Methodology.**  Each configuration runs 1 000 steps after warmup (5 compile
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
