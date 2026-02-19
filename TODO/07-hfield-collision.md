# Add HFIELD (Heightfield) Collision Support

**Priority:** Medium
**Category:** Feature
**Difficulty:** Medium-High

## Problem

`GeomType.HFIELD` is not supported. The `_COLLISION_FUNC` mapping in `collision_driver.py` has no entries for HFIELD pairs. Many popular environments (terrain locomotion, outdoor robotics) use heightfields. MJX supports HFIELD collisions.

## What to Do

1. Add `HFIELD` to `GeomType` in `types.py` if not already present.
2. Update `device.py` validation to accept HFIELD geoms.
3. Implement collision functions for HFIELD pairs:
   - `plane-hfield`: MuJoCo C doesn't collide these, so skip.
   - `hfield-sphere`: Most important — sphere rolling on terrain.
   - `hfield-capsule`: Second most important.
   - `hfield-box`: Lower priority.
   - `hfield-mesh`: Lowest priority.
4. Reference MJX's `collision_driver.py` and related collision files on GitHub for the heightfield collision logic. The approach involves sampling the heightfield at contact-candidate positions and computing penetration depth + normal from the height gradient.
5. Register the new functions in `_COLLISION_FUNC` in `collision_driver.py`.

## Files to Touch

- `mujoco_torch/_src/types.py` — ensure HFIELD is in GeomType
- `mujoco_torch/_src/device.py` — relax validation
- `mujoco_torch/_src/collision_driver.py` — register collision functions
- `mujoco_torch/_src/collision_primitive.py` or new `collision_hfield.py` — implement

## Tests

- Create a test model with a heightfield geom and a sphere/capsule dropped onto it.
- Step forward and compare contact data against MuJoCo C.
- Verify contact positions, normals, and depths match within tolerance.

## Submission Instructions

- If your changes have more than one step, use ghstack to submit. ghstack sends each commit as a separate PR, so make sure each commit message is a proper PR name.
- BugFix and features must have a test in the same commit.
- You can submit PRs; if you do, monitor the runs using gh.
