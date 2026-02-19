# Add FILTEREXACT Actuator Dynamics

**Priority:** Medium
**Category:** Feature
**Difficulty:** Low

## Problem

MJX supports `DynType.FILTEREXACT` but mujoco-torch doesn't. It's listed as unsupported in `types.py`. FILTEREXACT is the exact solution to the first-order filter dynamics (vs the Euler approximation in FILTER). It's a small addition since the infrastructure for FILTER already exists.

## What to Do

1. Add `FILTEREXACT` to the `DynType` enum in `mujoco_torch/_src/types.py`.
2. Add the `FILTEREXACT` entry to the `_TYPE_MAP` in `mujoco_torch/_src/device.py` so it passes validation.
3. Implement the dynamics in `mujoco_torch/_src/forward.py` (or wherever actuator dynamics are computed). The FILTEREXACT formula is:
   ```
   act_new = ctrl + (act - ctrl) * exp(-dt / tau)
   ```
   where `tau` is the dynprm time constant. Reference MJX's `_src/forward.py` for the exact implementation.
4. Verify against MuJoCo C output.

## Files to Touch

- `mujoco_torch/_src/types.py` — add enum member
- `mujoco_torch/_src/device.py` — update `_TYPE_MAP`
- `mujoco_torch/_src/forward.py` — implement dynamics

## Tests

- Create a test model with an actuator using `dyntype="filterexact"`.
- Step both MuJoCo C and mujoco-torch and compare `act` values.

## Submission Instructions

- If your changes have more than one step, use ghstack to submit. ghstack sends each commit as a separate PR, so make sure each commit message is a proper PR name.
- BugFix and features must have a test in the same commit.
- You can submit PRs; if you do, monitor the runs using gh.
