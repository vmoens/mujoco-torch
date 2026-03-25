# Fix Humanoid Contact Filtering (Spurious Non-Penetrating Contacts)

## Status

**Priority:** High — blocks humanoid RL training
**Discovered:** 2026-03-22 via numerical accuracy test (`examples/humanoid_accuracy_test.py`)

## Problem

mujoco-torch includes spurious non-penetrating contacts (positive distance) in the
constraint set that MuJoCo C correctly filters out. This causes divergent dynamics
when actions are applied.

### Evidence

With random ctrl on humanoid, at the very first step:
- **qvel error:** 0.107 (should be ~1e-8)
- **qpos error:** 5.6e-4

After 50 steps the trajectories diverge completely (qpos error ~2.7, qvel ~50).

With zero ctrl, trajectories match well (divergence only at step 44 with tiny 1e-6 errors).

### Root Cause

In `constraint.py`, contact constraints check:
```python
active = dist < cfg.cfd_width if cfg.cfd else dist < 0
```

But the issue is upstream: **collision detection in `collision_driver.py` includes contacts
with positive distance** (e.g., dist = +0.000132, +0.001131) that MuJoCo C excludes before
constraint generation.

This leads to different constraint counts:
- **MuJoCo C:** nefc = 21 (equality only, no spurious contacts)
- **mujoco-torch:** nefc = 53 (equality + spurious contact constraints)

The extra constraints change the acceleration (`qacc_smooth` differs by ~200 at some DOFs)
and propagate into completely different trajectories.

### Impact

- **Humanoid RL training:** Policy learns to stand (survival bonus dominates) but cannot
  learn locomotion because foot-ground contact dynamics are wrong
- **Ant/cheetah:** Less affected (simpler contact geometry or fewer spurious contacts)
- **Numerical tests:** 5 of 8 accuracy tests fail for humanoid with ctrl

### Where to Fix

1. `mujoco_torch/_src/collision_driver.py` — filter out contacts with `dist > margin + gap`
   before they enter constraint formation
2. OR `mujoco_torch/_src/constraint.py` — ensure contacts with positive distance are
   excluded from the constraint Jacobian (lines ~416-545)

Match MuJoCo C's contact filtering: non-penetrating contacts (positive distance after
accounting for margin) should not generate constraint rows.

### Reproduction

```bash
python examples/humanoid_accuracy_test.py
```

Tests 2, 4, 6, 8 will fail. Test 1 (zero ctrl) shows the baseline accuracy is good,
confirming the issue is specifically in the contact/constraint pipeline when forces
are applied.

### Related Files

- `mujoco_torch/_src/collision_driver.py` — collision detection
- `mujoco_torch/_src/constraint.py` — constraint formation (lines 416-545)
- `mujoco_torch/_src/forward.py` — step/forward pipeline
- `mujoco_torch/zoo/humanoid.py` — humanoid environment
- `examples/humanoid_accuracy_test.py` — numerical accuracy test
- `test/mjx_correctness_test.py` — existing MJX correctness tests (humanoid only tested with constraints disabled)
