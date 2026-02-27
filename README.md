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
- PyTorch >= 2.1
- MuJoCo >= 3.0
- tensordict >= 0.11

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
compiled_step = torch.compile(vmap_step)
d_batch = compiled_step(d_batch)
```

## Benchmarks

All numbers measured on a single NVIDIA H200 GPU, float64 precision,
humanoid model, B=4096 environments, 1000 simulation steps.

| Configuration | steps/s |
|---|---|
| MuJoCo C (sequential CPU) | ~62,000 |
| mujoco-torch vmap (CUDA, eager) | ~40,000 |
| mujoco-torch compile (CUDA) | ~254,000 |
| MJX (JAX jit+vmap) | TBD |

**Methodology**: Each configuration runs 1000 steps after warmup (5 compile
iterations for compiled variants, 1 for eager). Wall-clock time is measured
with `torch.cuda.synchronize()` / `jax.block_until_ready()` bracketing.
Steps/s = `batch_size * nsteps / elapsed_time`. Single GPU
(`CUDA_VISIBLE_DEVICES=0`), dtype=float64.

## Testing

```bash
# Run all tests (requires JAX + MJX for correctness tests)
pip install "jax[cpu]" "mujoco[mjx]"
pytest test/ -x -v
```

## License

Apache 2.0 -- see [LICENSE](LICENSE).
