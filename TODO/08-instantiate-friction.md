# Implement Friction Constraints (_instantiate_friction)

**Priority:** Medium
**Category:** Feature
**Difficulty:** Medium

## Problem

`mujoco_torch/_src/constraint.py:245-248` has a stub:

```python
def _instantiate_friction(m: Model, d: Data) -> Optional[_Efc]:
  # TODO(robotics-team): implement _instantiate_friction
  del m, d
  return None
```

This means joint and tendon frictionloss constraints are never created. Models that rely on `frictionloss` for joints or tendons will silently have no friction damping, leading to incorrect dynamics.

## What to Do

1. Implement `_instantiate_friction` in `constraint.py`. This should create constraint rows for:
   - **DOF friction**: For each DOF with `dof_frictionloss > 0`, create a friction constraint row.
   - **Tendon friction**: For each tendon with `tendon_frictionloss > 0`, create a friction constraint row.
2. The Jacobian for DOF friction is an identity row for the corresponding DOF. For tendon friction, it's the tendon moment arm (ten_J row).
3. Reference MJX's `mujoco/mjx/_src/constraint.py` on GitHub for the implementation.
4. Also note the related TODO at `constraint.py:364` about "other friction dimensions" — for now, implement the basic frictionloss case.

## Files to Touch

- `mujoco_torch/_src/constraint.py` — implement `_instantiate_friction`

## Tests

- Create a test model with joints that have `frictionloss` set to nonzero values.
- Step forward and compare `qfrc_constraint` against MuJoCo C.
- Verify that removing frictionloss produces different dynamics.

## Submission Instructions

- If your changes have more than one step, use ghstack to submit. ghstack sends each commit as a separate PR, so make sure each commit message is a proper PR name.
- BugFix and features must have a test in the same commit.
- You can submit PRs; if you do, monitor the runs using gh.
