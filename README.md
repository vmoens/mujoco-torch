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
| MuJoCo C (CPU, sequential) | 59,536 | — | — | — | — |
| mujoco-torch vmap (eager) | 14 | 1,632 | 13,069 | 52,335 | 351,379 |
| mujoco-torch compile | 177 | 20,962 | 156,522 | 564,368 | 1,211,171 |
| mujoco-torch compile (reduce-overhead) | 357 | 40,944 | 267,689 | 699,402 | 1,221,532 |
| **mujoco-torch compile (tuned)** | **220** | **26,075** | **199,036** | **718,478** | **2,397,826** |
| MJX (JAX jit+vmap) | 870 | 108,905 | 874,432 | 2,237,444 | 2,382,388 |

### Ant

| Configuration | B=1 | B=128 | B=1 024 | B=4 096 | B=32 768 |
|---|--:|--:|--:|--:|--:|
| MuJoCo C (CPU, sequential) | 101,157 | — | — | — | — |
| mujoco-torch vmap (eager) | 18 | 2,191 | 17,709 | 70,287 | 244,466 |
| mujoco-torch compile | 211 | 25,955 | 201,090 | 464,011 | 500,758 |
| mujoco-torch compile (reduce-overhead) | 443 | 52,179 | 310,034 | 467,992 | 507,513 |
| **mujoco-torch compile (tuned)** | **293** | **32,697** | **252,436** | **855,797** | **2,107,219** |
| MJX (JAX jit+vmap) | 772 | 92,726 | 483,129 | 674,019 | 687,813 |

### Half-Cheetah

| Configuration | B=1 | B=128 | B=1 024 | B=4 096 | B=32 768 |
|---|--:|--:|--:|--:|--:|
| MuJoCo C (CPU, sequential) | 166,742 | — | — | — | — |
| mujoco-torch vmap (eager) | 18 | 2,273 | 18,114 | 72,423 | 550,086 |
| mujoco-torch compile | 208 | 17,580 | 131,327 | 530,558 | 2,157,809 |
| mujoco-torch compile (reduce-overhead) | 427 | 735 | 216,762 | 718,405 | 2,123,271 |
| **mujoco-torch compile (tuned)** | **63** | **6,053** | **33,142** | **129,992** | **632,452** |
| MJX (JAX jit+vmap) | 569 | 58,191 | 444,864 | 1,408,451 | 2,888,935 |

### Walker2d

Walker2d uses the RK4 integrator, which makes each step ~3× more expensive
than Euler.

| Configuration | B=1 | B=128 | B=1 024 | B=4 096 | B=32 768 |
|---|--:|--:|--:|--:|--:|
| MuJoCo C (CPU, sequential) | 41,289 | — | — | — | — |
| mujoco-torch vmap (eager) | 5 | 502 | 3,684 | 14,432 | 101,332 |
| mujoco-torch compile | 65 | 6,204 | 43,255 | 166,639 | 450,069 |
| mujoco-torch compile (reduce-overhead) | 129 | 10,158 | 65,386 | 208,026 | 457,422 |
| **mujoco-torch compile (tuned)** | **84** | **5,562** | **55,866** | **204,346** | **677,787** |
| MJX (JAX jit+vmap) | 170 | 10,176 | 69,757 | 203,816 | 324,060 |

### Hopper

Hopper uses the RK4 integrator (like Walker2d).

| Configuration | B=1 | B=128 | B=1 024 | B=4 096 | B=32 768 |
|---|--:|--:|--:|--:|--:|
| MuJoCo C (CPU, sequential) | 63,644 | — | — | — | — |
| mujoco-torch vmap (eager) | 4 | 519 | 4,104 | 16,365 | 125,955 |
| mujoco-torch compile | 233 | 26,215 | 207,258 | 842,875 | 3,254,876 |
| mujoco-torch compile (reduce-overhead) | 288 | 33,507 | 384,972 | 1,252,018 | 3,347,796 |
| **mujoco-torch compile (tuned)** | **284** | **34,491** | **278,069** | **1,060,707** | **5,238,064** |
| MJX (JAX jit+vmap) | 222 | 21,879 | 180,342 | 525,552 | 1,293,001 |

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
