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
| mujoco-torch vmap (eager) | 14 | 1,632 | 13,069 | 52,337 | 351,371 |
| **mujoco-torch compile** | **154** | **18,478** | **147,060** | **536,692** | **1,180,109** |
| MJX (JAX jit+vmap) | 870 | 108,905 | 874,432 | 2,237,444 | 2,382,388 |

### Ant

| Configuration | B=1 | B=128 | B=1 024 | B=4 096 | B=32 768 |
|---|--:|--:|--:|--:|--:|
| MuJoCo C (CPU, sequential) | 101,157 | — | — | — | — |
| mujoco-torch vmap (eager) | 18 | 2,191 | 17,705 | 70,273 | 244,474 |
| **mujoco-torch compile** | **173** | **20,816** | **145,563** | **340,190** | **463,296** |
| MJX (JAX jit+vmap) | 772 | 92,726 | 483,129 | 674,019 | 687,813 |

### Half-Cheetah

| Configuration | B=1 | B=128 | B=1 024 | B=4 096 | B=32 768 |
|---|--:|--:|--:|--:|--:|
| MuJoCo C (CPU, sequential) | 166,742 | — | — | — | — |
| mujoco-torch vmap (eager) | 18 | 2,273 | 18,115 | 72,413 | 550,095 |
| **mujoco-torch compile** | **196** | **23,632** | **178,823** | **736,681** | **2,243,328** |
| MJX (JAX jit+vmap) | 569 | 58,191 | 444,864 | 1,408,451 | 2,888,935 |

### Walker2d

Walker2d uses the RK4 integrator, which makes each step ~3× more expensive
than Euler.

| Configuration | B=1 | B=128 | B=1 024 | B=4 096 | B=32 768 |
|---|--:|--:|--:|--:|--:|
| MuJoCo C (CPU, sequential) | 41,289 | — | — | — | — |
| mujoco-torch vmap (eager) | 5 | 502 | 3,684 | 14,429 | 101,337 |
| **mujoco-torch compile** | **70** | **8,114** | **40,210** | **203,028** | **465,286** |
| MJX (JAX jit+vmap) | 170 | 10,176 | 69,757 | 203,816 | 324,060 |

### Hopper

Hopper uses the RK4 integrator (like Walker2d).

| Configuration | B=1 | B=128 | B=1 024 | B=4 096 | B=32 768 |
|---|--:|--:|--:|--:|--:|
| MuJoCo C (CPU, sequential) | 63,644 | — | — | — | — |
| mujoco-torch vmap (eager) | 4 | 519 | 4,104 | 16,363 | 126,000 |
| **mujoco-torch compile** | **64** | **7,038** | **49,812** | **176,730** | **571,875** |
| MJX (JAX jit+vmap) | 222 | 21,879 | 180,342 | 525,552 | 1,293,001 |

**Methodology.**  Each configuration runs 1 000 steps after warmup (5 compile
iterations for compiled variants, 1 JIT warmup for MJX).  Wall-clock time is
measured with `torch.cuda.synchronize()` / `jax.block_until_ready()` bracketing.
Steps/s = `batch_size × nsteps / elapsed_time`.  Single GPU
(`CUDA_VISIBLE_DEVICES=0`), dtype=float64.

To reproduce, run the benchmark script (requires the PyTorch fork above):

```bash
CUDA_VISIBLE_DEVICES=0 python -u gpu_bench.py --model humanoid
CUDA_VISIBLE_DEVICES=0 python -u gpu_bench.py --model ant
CUDA_VISIBLE_DEVICES=0 python -u gpu_bench.py --model halfcheetah
CUDA_VISIBLE_DEVICES=0 python -u gpu_bench.py --model walker2d
CUDA_VISIBLE_DEVICES=0 python -u gpu_bench.py --model hopper
CUDA_VISIBLE_DEVICES=0 python -u gpu_bench.py --model all
python scratch/plot_bench.py scratch/bench_humanoid.json scratch/bench_ant.json scratch/bench_halfcheetah.json scratch/bench_walker2d.json scratch/bench_hopper.json -o assets/benchmark.png
```

## Testing

```bash
# Run all tests (requires JAX + MJX for correctness tests)
pip install "jax[cpu]" "mujoco[mjx]"
pytest test/ -x -v
```

## License

Apache 2.0 -- see [LICENSE](LICENSE).
