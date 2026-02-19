# Remove Unnecessary Local Imports

**Priority:** Medium
**Category:** Code Quality
**Difficulty:** Low

## Problem

Several `_src/` modules use local (function-level) imports. Some are justified by circular dependencies, but others are not. Local imports hurt readability, hide dependencies, and can mask import errors until runtime.

Current local imports:

- `device.py:312` — `from torch.utils._pytree import tree_map` (inside function)
- `device.py:362` — `import numpy as np` (inside function)
- `constraint.py:402` — `from mujoco_torch._src import collision_driver` (comment says "avoid circular at module level")
- `collision_driver.py:476` — `from mujoco_torch._src.constraint import constraint_sizes`
- `solver.py:309` — `from mujoco_torch._src.constraint import constraint_sizes`
- `support.py:116` — `from mujoco_torch._src import math`
- `support.py:123` — `from torch._C._functorch import is_batchedtensor, ...`

## What to Do

1. **For each local import, determine if it's truly necessary:**
   - **Circular dependency**: `constraint.py` ↔ `collision_driver.py` ↔ `solver.py` form a cycle via `constraint_sizes`. Consider breaking the cycle by extracting `constraint_sizes` into a separate module (e.g. `constraint_types.py` or adding it to `types.py`).
   - **`support.py:116` (`math`)**: No obvious circular dep — `math.py` doesn't import `support.py`. Move to top level.
   - **`support.py:123` (functorch internals)**: This is a private API import. Keep local if it's optional/version-dependent, otherwise move to top level.
   - **`device.py:312` (`tree_map`)**: Check if there's a reason it's local. If not, move to top level.
   - **`device.py:362` (`numpy`)**: numpy is already imported at module level in this file — check if this is a duplicate or if the top-level import was removed by mistake.
2. **Move justified imports to top level** with a comment if needed.
3. **Break circular deps** by refactoring shared types/functions into a common module.
4. **Verify all tests pass** after changes.

## Files to Touch

- `mujoco_torch/_src/device.py`
- `mujoco_torch/_src/constraint.py`
- `mujoco_torch/_src/collision_driver.py`
- `mujoco_torch/_src/solver.py`
- `mujoco_torch/_src/support.py`
- Possibly a new `mujoco_torch/_src/constraint_types.py` to break cycles

## Submission Instructions

- If your changes have more than one step, use ghstack to submit. ghstack sends each commit as a separate PR, so make sure each commit message is a proper PR name.
- BugFix and features must have a test in the same commit.
- You can submit PRs; if you do, monitor the runs using gh.
