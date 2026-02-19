# Expand Test Model Coverage

**Priority:** Medium
**Category:** Test
**Difficulty:** Medium

## Problem

Only 5 test models exist: `ant.xml`, `humanoid.xml`, `pendula.xml`, `equality.xml`, `convex.xml`. The solver test explicitly skips humanoid. Many MuJoCo features are not exercised by these models.

## What to Do

1. **Add models from the MuJoCo model zoo** to `mujoco_torch/test_data/`:
   - `cartpole.xml` — simple, no contacts
   - `reacher.xml` — 2D, simple contacts
   - `hopper.xml` — planar locomotion
   - `walker2d.xml` — planar locomotion
   - `half_cheetah.xml` — popular RL benchmark
   - `swimmer.xml` — multi-link, no contacts
   - `hand.xml` or similar dexterous manipulation model
2. **Add parametrized tests** that run `step` for N steps on each model and compare against MuJoCo C. Follow the pattern in `forward_test.py`.
3. **Enable humanoid in solver test** — investigate why it was excluded (`solver_test.py` mentions boundary contact precision) and either fix the issue or document the tolerance.
4. **Test with different integrators** — run each model with Euler, RK4, and ImplicitFast.

## Model Licensing

The MuJoCo model zoo models are typically under Apache 2.0. Check each model's header. If using models from dm_control, those are Apache 2.0. Ensure license headers are preserved.

## Submission Instructions

- If your changes have more than one step, use ghstack to submit. ghstack sends each commit as a separate PR, so make sure each commit message is a proper PR name.
- BugFix and features must have a test in the same commit.
- You can submit PRs; if you do, monitor the runs using gh.
