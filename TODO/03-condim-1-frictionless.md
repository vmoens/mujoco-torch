# Support condim=1 (Frictionless Contacts)

**Status**: Solved
**Priority:** High
**Category:** Feature
**Difficulty:** Medium

## Problem

`mujoco_torch/_src/device.py:190-193` hard-rejects any model where `condim != 3`:

```python
if any(dim != 3 for dim in m.geom_condim) or any(dim != 3 for dim in m.pair_dim):
    raise NotImplementedError('Only condim=3 is supported.')
```

`condim=1` means frictionless contact (normal force only, no tangential friction). Many simple models and some environments use it. This is the easiest condim extension.

## What to Do

1. In `device.py`, relax the validation to accept `condim` in `{1, 3}`.
2. In `_src/constraint.py`, `_instantiate_contact` currently always creates pyramidal friction rows. Add a code path for `condim=1` contacts that only produces a single normal-force constraint row (no friction directions).
3. In `_src/collision_driver.py`, check that the contact `dim` field is set correctly for `condim=1` geoms.
4. Reference MJX's implementation in `mujoco/mjx/_src/constraint.py` on GitHub for how they handle the condim switch.

## Files to Touch

- `mujoco_torch/_src/device.py` — relax validation
- `mujoco_torch/_src/constraint.py` — handle condim=1 in `_instantiate_contact`
- `mujoco_torch/_src/collision_driver.py` — verify dim propagation
- `mujoco_torch/_src/types.py` — if Contact needs a dim field

## Tests

- Create a test model XML with `condim="1"` on some geoms.
- Verify `device_put` accepts it.
- Step the simulation and compare `qpos`/`qvel` against MuJoCo C output.
- Verify contact forces are normal-only (no tangential components).

## Submission Instructions

- If your changes have more than one step, use ghstack to submit. ghstack sends each commit as a separate PR, so make sure each commit message is a proper PR name.
- BugFix and features must have a test in the same commit.
- You can submit PRs; if you do, monitor the runs using gh.
