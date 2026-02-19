# Add Elliptic Friction Cone Support

**Priority:** Low
**Category:** Feature
**Difficulty:** Medium

## Problem

Only `ConeType.PYRAMIDAL` is supported. MJX also supports `ConeType.ELLIPTIC`, which provides a more accurate (and smoother) friction cone approximation. The elliptic cone is better for:
- Dexterous manipulation (smoother contact forces)
- Optimization-based control (fewer constraint rows for same accuracy)

## What to Do

1. Add `ELLIPTIC` to the `ConeType` enum in `types.py` if not present.
2. Update `_TYPE_MAP` in `device.py` to accept `ELLIPTIC`.
3. In `constraint.py`, add a code path for elliptic cone contact constraints:
   - Pyramidal uses N friction pyramid directions (4 for 3D friction).
   - Elliptic uses a single second-order cone constraint per contact.
   - The constraint Jacobian and force computation are different.
4. The solver needs to handle the cone constraint (SOCP-like). Reference MJX's implementation.
5. Note: `condim=1` with ELLIPTIC is not supported even in MJX.

## Depends On

- TODO/03-condim-1-frictionless.md (condim infrastructure)
- TODO/11-condim-4-6.md (for full condim support)

## Files to Touch

- `mujoco_torch/_src/types.py` — add enum
- `mujoco_torch/_src/device.py` — accept ELLIPTIC
- `mujoco_torch/_src/constraint.py` — elliptic cone constraints
- `mujoco_torch/_src/solver.py` — handle cone constraints

## Tests

- Create a model with `cone="elliptic"`.
- Compare contact forces against MuJoCo C.
- Verify smoother force profiles compared to pyramidal.

## Submission Instructions

- If your changes have more than one step, use ghstack to submit. ghstack sends each commit as a separate PR, so make sure each commit message is a proper PR name.
- BugFix and features must have a test in the same commit.
- You can submit PRs; if you do, monitor the runs using gh.
