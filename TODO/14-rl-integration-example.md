# RL Framework Integration Example

**Priority:** High
**Category:** Feature/Docs
**Difficulty:** Medium

## Problem

The primary use case for mujoco-torch is reinforcement learning with PyTorch. But there's no example showing how to use it with any RL framework. Without this, users won't know how to adopt it.

## What to Do

1. **TorchRL integration** — Create `examples/torchrl_example.py` showing:
   - A vectorized MuJoCo environment using mujoco-torch as the backend
   - Wrapping `step` in a TorchRL `EnvBase` subclass
   - Running a simple policy (e.g. PPO) on ant or humanoid
   - Using `torch.compile` for the full env+policy loop
2. **Gymnasium integration** — Create `examples/gymnasium_example.py` showing:
   - A Gymnasium-compatible wrapper around mujoco-torch
   - `reset()` and `step()` that return standard Gymnasium outputs
   - Batched version using `torch.vmap`
3. Both examples should include:
   - A reward function
   - An observation function (joint positions, velocities, etc.)
   - A simple training loop (even just a few steps to demonstrate the pattern)

## Reference

- Look at how Brax wraps MJX for RL: https://github.com/google/brax
- Look at TorchRL's existing MuJoCo integration for API patterns
- The `examples/batched_comparison.py` already shows the vmap pattern

## Submission Instructions

- If your changes have more than one step, use ghstack to submit. ghstack sends each commit as a separate PR, so make sure each commit message is a proper PR name.
- BugFix and features must have a test in the same commit.
- You can submit PRs; if you do, monitor the runs using gh.
