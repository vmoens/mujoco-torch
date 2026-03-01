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
| MuJoCo C (CPU, sequential) | 62,188 | — | — | — | — |
| mujoco-torch vmap (eager) | 10 | 1,239 | 9,935 | 39,682 | 292,238 |
| **mujoco-torch compile** | **90** | **10,763** | **85,283** | **331,496** | **1,065,339** |
| MJX (JAX jit+vmap) | 58 | 8,396 | 65,927 | 237,061 | 1,102,603 |

### Ant

| Configuration | B=1 | B=128 | B=1 024 | B=4 096 | B=32 768 |
|---|--:|--:|--:|--:|--:|
| MuJoCo C (CPU, sequential) | 106,019 | — | — | — | — |
| mujoco-torch vmap (eager) | 15 | 1,866 | 14,949 | 59,300 | 301,439 |
| **mujoco-torch compile** | **120** | **9,820** | **78,000** | **264,081** | **460,511** |
| MJX (JAX jit+vmap) | 99 | 11,965 | 66,796 | 235,721 | 540,229 |

### Half-Cheetah

| Configuration | B=1 | B=128 | B=1 024 | B=4 096 | B=32 768 |
|---|--:|--:|--:|--:|--:|
| MuJoCo C (CPU, sequential) | 181,946 | — | — | — | — |
| mujoco-torch vmap (eager) | 14 | 1,810 | 14,384 | 58,006 | 447,686 |
| **mujoco-torch compile** | **116** | **13,829** | **109,334** | **435,795** | **1,889,818** |
| MJX (JAX jit+vmap) | 105 | 12,102 | 85,769 | 300,022 | 1,576,656 |

### Walker2d

Walker2d uses the RK4 integrator, which makes each step ~3× more expensive
than Euler.

| Configuration | B=1 | B=128 | B=1 024 | B=4 096 | B=32 768 |
|---|--:|--:|--:|--:|--:|
| MuJoCo C (CPU, sequential) | 42,245 | — | — | — | — |
| mujoco-torch vmap (eager) | 3 | 330 | 2,389 | 8,954 | 63,274 |
| **mujoco-torch compile** | **41** | **3,578** | **7,006** | **94,907** | **340,733** |
| MJX (JAX jit+vmap) | 31 | 2,978 | 24,452 | 80,401 | 244,155 |

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
CUDA_VISIBLE_DEVICES=0 python -u gpu_bench.py --model all
python scratch/plot_bench.py bench_humanoid.json bench_ant.json bench_halfcheetah.json bench_walker2d.json -o assets/benchmark.png
```

## Testing

```bash
# Run all tests (requires JAX + MJX for correctness tests)
pip install "jax[cpu]" "mujoco[mjx]"
pytest test/ -x -v
```

## License

Apache 2.0 -- see [LICENSE](LICENSE).
