# Replace numpy with PyTorch Wherever Possible

**Priority:** Medium
**Category:** Code Quality
**Difficulty:** Medium

## Problem

Many `_src/` modules import and use numpy for operations that could be done with PyTorch tensors. This is problematic because:
- numpy ops force CPU execution and break the computation graph
- They prevent `torch.compile` from tracing through the full pipeline
- They create unnecessary CPU-GPU transfers in CUDA workflows
- They're a conceptual inconsistency in a library whose whole point is native PyTorch

The following files import numpy at module level:

- `collision_driver.py`
- `collision_types.py`
- `constraint.py`
- `device.py`
- `forward.py`
- `io.py`
- `mesh.py`
- `ray.py`
- `scan.py`
- `sensor.py`
- `smooth.py`
- `solver.py`
- `support.py`
- `test_util.py`
- `types.py`

## What to Do

1. **Audit each file** — for every `np.` call, decide:
   - Is this on model-loading / setup code that only runs once? (numpy is OK here, e.g. `np.nonzero` on model metadata in `device.py`)
   - Is this on a hot path during `step()`? (must be converted to torch)
   - Is this on static shape/index computation? (numpy may be acceptable if it doesn't break compile)
2. **Convert hot-path numpy to torch** — common patterns to replace:
   - `np.array(...)` → `torch.tensor(...)`
   - `np.zeros(...)` → `torch.zeros(...)`
   - `np.nonzero(...)` → `torch.nonzero(...)` (but careful: nonzero is dynamic-shape)
   - `np.concatenate(...)` → `torch.cat(...)`
   - `np.stack(...)` → `torch.stack(...)`
   - `np.arange(...)` → `torch.arange(...)`
   - `np.where(...)` → `torch.where(...)`
   - `np.einsum(...)` → `torch.einsum(...)`
3. **Keep numpy where appropriate** — `device.py` validation, model metadata indexing at load time, and `types.py` struct definitions are fine as numpy since they happen once.
4. **Remove unused numpy imports** after conversion.
5. **Verify tests still pass** after each file's conversion.
6. **Verify `torch.compile` compatibility** — the whole point is to make the pipeline traceable.

## Submission Instructions

- If your changes have more than one step, use ghstack to submit. ghstack sends each commit as a separate PR, so make sure each commit message is a proper PR name.
- BugFix and features must have a test in the same commit.
- You can submit PRs; if you do, monitor the runs using gh.
