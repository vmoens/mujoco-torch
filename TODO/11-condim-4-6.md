# Support condim=4 and condim=6

**Priority:** Medium
**Category:** Feature
**Difficulty:** High

## Problem

Only `condim=3` is supported (normal + 2 tangential friction). `condim=4` adds torsional friction (spinning), `condim=6` adds rolling friction. Many manipulation and dexterous hand models use `condim=4` or `condim=6`. MJX supports all of 1, 3, 4, 6.

## Depends On

- TODO/03-condim-1-frictionless.md (condim=1 should be done first as it's simpler)

## What to Do

1. In `device.py`, relax validation to accept `condim` in `{1, 3, 4, 6}`.
2. In `constraint.py`, extend `_instantiate_contact` to handle:
   - **condim=4**: Normal + 2 tangential + 1 torsional friction direction. 4 constraint rows per contact (with pyramidal cone) or 4 with elliptic.
   - **condim=6**: Normal + 2 tangential + 1 torsional + 2 rolling. 6 constraint rows per contact.
3. The collision driver's `_COLLISION_FUNC` mapping at `collision_driver.py:480` has a related TODO about "other friction dimensions" — update the contact `dim` field.
4. Update the solver to handle the different constraint row counts.
5. Reference MJX's `constraint.py` and `solver.py` for the exact row layout and force computation.

## Files to Touch

- `mujoco_torch/_src/device.py` — relax validation
- `mujoco_torch/_src/constraint.py` — extend contact instantiation
- `mujoco_torch/_src/collision_driver.py` — set correct dim
- `mujoco_torch/_src/solver.py` — handle variable constraint dimensions

## Tests

- Create test models with `condim="4"` and `condim="6"`.
- Step and compare against MuJoCo C for each condim.
- Test a dexterous hand model if available.

## Submission Instructions

- If your changes have more than one step, use ghstack to submit. ghstack sends each commit as a separate PR, so make sure each commit message is a proper PR name.
- BugFix and features must have a test in the same commit.
- You can submit PRs; if you do, monitor the runs using gh.
