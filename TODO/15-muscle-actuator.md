# Add MUSCLE Actuator Model

**Priority:** Medium
**Category:** Feature
**Difficulty:** Medium-High

## Problem

MJX supports `DynType.MUSCLE`, `GainType.MUSCLE`, and `BiasType.MUSCLE` for biomechanical models. mujoco-torch does not. The MUSCLE model is important for:
- Humanoid locomotion research
- Biomechanics simulation
- MyoSuite-style environments

## What to Do

1. Add `MUSCLE` to `DynType`, `GainType`, and `BiasType` enums in `types.py`.
2. Update `_TYPE_MAP` in `device.py` to accept MUSCLE types.
3. Implement muscle dynamics in `forward.py`:
   - **DynType.MUSCLE**: Muscle activation dynamics (first-order filter with excitation-dependent time constants). The activation dynamics use different time constants for increasing vs decreasing activation.
   - **GainType.MUSCLE**: Force-length-velocity gain curve.
   - **BiasType.MUSCLE**: Passive muscle force (force-length curve).
4. Reference MJX's implementation in `mujoco/mjx/_src/forward.py` on GitHub.
5. Also reference the MuJoCo docs on muscle actuators: https://mujoco.readthedocs.io/en/stable/modeling.html#muscles

## Files to Touch

- `mujoco_torch/_src/types.py` — add enum members
- `mujoco_torch/_src/device.py` — update `_TYPE_MAP`
- `mujoco_torch/_src/forward.py` — implement muscle dynamics, gain, bias

## Tests

- Create a test model with muscle actuators.
- Compare activation dynamics and forces against MuJoCo C.
- Test both increasing and decreasing activation paths.

## Submission Instructions

- If your changes have more than one step, use ghstack to submit. ghstack sends each commit as a separate PR, so make sure each commit message is a proper PR name.
- BugFix and features must have a test in the same commit.
- You can submit PRs; if you do, monitor the runs using gh.
