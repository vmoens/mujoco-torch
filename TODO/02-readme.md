# Write a README

**Priority:** Critical
**Category:** Docs
**Difficulty:** Low

## Problem

There is no README.md at the repo root. This is the single biggest blocker for open-sourcing.

## What to Do

Write a `README.md` at the repo root containing:

1. **Title and one-liner** — "mujoco-torch: MuJoCo physics in PyTorch" or similar.
2. **What it is** — A PyTorch port of MuJoCo XLA (MJX). Runs MuJoCo physics natively in PyTorch with `torch.compile` and `torch.vmap` support.
3. **Why it exists** — Native PyTorch integration: no JAX dependency, direct gradient flow into PyTorch policy networks, works with TorchRL and the PyTorch RL ecosystem.
4. **Installation** — `pip install -e .` from source. List deps (torch>=2.1, mujoco>=3.0, tensordict>=0.11).
5. **Quick start** — Show the 5-line usage pattern from `examples/e2e_comparison.py`:
   ```python
   import mujoco, mujoco_torch
   m = mujoco.MjModel.from_xml_path("model.xml")
   mx = mujoco_torch.device_put(m)
   dx = mujoco_torch.make_data(mx)
   dx = mujoco_torch.step(mx, dx)
   ```
6. **Batched simulation** — Show `torch.vmap(lambda d: mujoco_torch.step(mx, d))` pattern.
7. **Feature matrix** — Table comparing supported features vs MJX (integrators, solvers, geom types, condim, actuators, etc). Be honest about gaps.
8. **Known limitations** — condim=3 only, no MUSCLE actuator, no HFIELD, inverse dynamics WIP.
9. **License** — Apache 2.0.
10. **Acknowledgments** — Based on MuJoCo MJX by Google DeepMind.

Look at the existing `examples/` and `mujoco_torch/__init__.py` for the public API surface.

## Submission Instructions

- If your changes have more than one step, use ghstack to submit. ghstack sends each commit as a separate PR, so make sure each commit message is a proper PR name.
- BugFix and features must have a test in the same commit.
- You can submit PRs; if you do, monitor the runs using gh.
