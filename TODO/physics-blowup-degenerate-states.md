# Physics Blow-Up: Degenerate States in Batched Humanoid Simulation

## Status

**Priority:** High — causes NaN rewards and Cholesky failures during RL training
**Discovered:** 2026-03-23 during SAC training on humanoid_rich with 2048+ envs

## Problem

When running batched humanoid simulation with large numbers of parallel environments
(2048+), some environments occasionally reach degenerate physics states that produce:

1. **Extreme rewards** (e.g., `-8.75e26`) — the physics state produces nonsensical
   reward components (likely `qvel` or `qpos` values that have exploded)
2. **Cholesky failure** — `linalg.cholesky: The factorization could not be completed
   because the input is not positive-definite` in `mujoco_torch/_src/math.py:104`
   (`small_cholesky`), called from `solver.py:350` (`_update_gradient`)

Both originate from the constraint solver receiving a non-positive-definite Hessian
matrix, which means the physics state is in a configuration that the solver cannot
handle.

### Observed Behavior

With `num_envs=8192` (SAC + compiled step):

```
torch._C._LinAlgError: linalg.cholesky: (Batch element 1471): The factorization
could not be completed because the input is not positive-definite (the leading minor
of order 8 is not positive-definite).
```

Stack trace: `zoo/base.py:_step` → `vmap(step_fn)` → `forward.py:step` →
`solver.py:solve` → `solver.py:_create_context` → `solver.py:_update_gradient` →
`math.py:small_cholesky` → `torch.linalg.cholesky`

With `num_envs=2048` (SAC, no crash but degenerate reward):

```
iter 1 | frames=2048000 | ep_reward=-875349264176085299347062784.0
iter 2 | frames=4096000 | ep_reward=-6209931467.7
```

The probability scales with `num_envs` — 256 envs rarely triggers it, 2048 sometimes,
8192 reliably.

### Root Cause Hypothesis

During early training (random policy), some envs reach extreme joint configurations
(e.g., limb self-penetration, extreme velocities from contact forces) that produce
a constraint Hessian `H` that is not positive-definite. The Cholesky decomposition
in the solver then fails.

MuJoCo C likely handles this via:
- Regularization of the Hessian (adding epsilon to diagonal)
- Better contact filtering (see `TODO/fix-humanoid-contact-filtering.md`)
- Clamping extreme states before they reach the solver

## Reproduction

### Saved Degenerate States

The SAC training script (`examples/train_sac.py`) has been patched to detect and
save degenerate states. When a reward with `abs(reward) > 1e6` is detected, it saves:

- The 10-step trajectory slice leading up to the blow-up
- The full environment trajectory for that env
- Metadata (env index, step index, extreme reward value, args)

Saved files: `degenerate_states/degenerate_iter{N}.pt`

To load and inspect:

```python
import torch
data = torch.load("degenerate_states/degenerate_iter0.pt")
print(data["extreme_reward"])      # e.g., -8.75e26
print(data["env_idx"])             # which parallel env
print(data["step_idx"])            # which timestep
slice_td = data["slice"]           # TensorDict with obs, action, reward, etc.
full_td = data["full_env_trajectory"]  # Full trajectory for this env

# Inspect the observation at the blow-up step
print(slice_td[-1]["observation"])
# Inspect the action that triggered it
print(slice_td[-1]["action"])
# Check if qpos/qvel are in the observation
print(slice_td[-1]["observation"].abs().max())
```

### Manual Reproduction

```bash
# High probability of triggering (8K envs, random policy)
python examples/train_sac.py --env humanoid_rich --num_envs 8192 \
    --frames_per_batch 8192 --compile
```

## Analysis of Captured State

Saved file: `TODO/degenerate_iter0.pt` (3MB)

**Env 167, step 25 of 1000, reward = -8.77e31**

Timeline (10 steps before blow-up):

| Step | Reward | obs_max | Notes |
|------|--------|---------|-------|
| 15-23 | 3.3–4.5 | 80–120 | Normal |
| 24 | 2.71 | 53.9 | Slightly lower than usual |
| **25** | **-8.77e31** | **1450** | Blow-up |
| 26 | 4.96 | 6.62 | Env reset, back to normal |

- No NaN or Inf — all values are finite but extreme
- Extreme obs components at indices **256, 257, 262, 263, 268, 269** (values ~1300-1450)
- These correspond to `cfrc_ext` (external contact forces) in the rich observation
- Actions are normal (max ~1.0, within bounds)
- The blow-up happens in a **single physics step**: obs_max goes from 54 to 1450
- The env recovers immediately after reset (step 26)

**Conclusion**: A single `mujoco_torch.step()` call produces extreme contact forces
(cfrc_ext) that cause a huge negative reward. The physics state before the blow-up
(step 24) appears unremarkable. This suggests the solver hits a degenerate
configuration in the contact/constraint computation.

## Suggested Fix Directions

1. **Regularize the Hessian in `solver.py:_update_gradient`**: Add a small epsilon
   to the diagonal of `h` before Cholesky: `h = h + eps * torch.eye(h.shape[-1])`
2. **Use `torch.linalg.cholesky_ex`** instead of `torch.linalg.cholesky` to get
   an error code per batch element, then handle failures gracefully (e.g., skip
   that env's constraint, or use a fallback solver)
3. **Clamp extreme states**: In `zoo/base.py:_step`, clamp `qpos`/`qvel` to
   reasonable ranges before feeding into the solver
4. **Fix contact filtering** (related: `TODO/fix-humanoid-contact-filtering.md`):
   Spurious contacts may be contributing to the degenerate Hessian

## Related

- `TODO/fix-humanoid-contact-filtering.md` — spurious non-penetrating contacts
- `TODO/37-inline-cholesky-threshold.md` — Cholesky threshold tuning
- `mujoco_torch/_src/solver.py:312` — `_create_context`
- `mujoco_torch/_src/solver.py:350` — `_update_gradient`
- `mujoco_torch/_src/math.py:104` — `small_cholesky`
