# Fix Inverse Dynamics Bug

**Status**: Solved
**Priority:** Critical
**Category:** BugFix
**Difficulty:** Low

## Problem

`mujoco_torch/_src/inverse.py:75` calls `solver._Context.create(m, d, grad=False)`, but `_Context` is a `NamedTuple` (defined in `solver.py:83`) and has no `create` classmethod. This means `inverse()` crashes at runtime whenever there are active constraints (`d.efc_J.numel() != 0`).

## What to Do

1. Look at how `_Context` is constructed inside the solver (e.g. in `_solve_cg` or `_solve_newton` in `solver.py`). Replicate that construction in `inv_constraint`, or add a `create` classmethod/factory to `_Context`.
2. Reference the MJX JAX implementation (`mujoco/mjx/_src/inverse.py` and `mujoco/mjx/_src/solver.py` on GitHub) for the correct inverse-constraint logic.
3. The inverse constraint path should compute `efc_force` and `qfrc_constraint` without running the iterative solver — it uses the analytical inverse.

## Files to Touch

- `mujoco_torch/_src/solver.py` — possibly add a factory or expose what's needed
- `mujoco_torch/_src/inverse.py` — fix `inv_constraint`

## Tests

- Add a test in `test/inverse_test.py` that:
  - Loads a model with contacts (e.g. `ant.xml` after a few steps so contacts exist)
  - Runs `mujoco_torch.inverse(m, d)` and checks it doesn't crash
  - Compares `qfrc_inverse` against MuJoCo C's `mj_inverse` output

## Submission Instructions

- If your changes have more than one step, use ghstack to submit. ghstack sends each commit as a separate PR, so make sure each commit message is a proper PR name.
- BugFix and features must have a test in the same commit.
- You can submit PRs; if you do, monitor the runs using gh.
