# Fix Solver Robustness on Humanoid

**Priority:** Medium
**Category:** BugFix
**Difficulty:** Medium-High

## Problem

The solver test (`test/solver_test.py`) explicitly excludes the humanoid model, citing boundary contact precision issues. If the solver doesn't work on humanoid — one of the most common RL benchmarks — that's a significant credibility problem.

## What to Do

1. **Diagnose**: Run the solver test on humanoid and capture the failures:
   - What's the actual vs expected `qfrc_constraint`?
   - What's the magnitude of the error?
   - Does it happen on specific contacts or all contacts?
   - Is it a convergence issue (needs more iterations) or an algorithmic bug?
2. **Compare with MJX**: Run the same model through MJX and check if MJX also diverges from MuJoCo C on humanoid, and by how much. MJX may have the same tolerance issues.
3. **Investigate root causes**:
   - Solver iteration count — maybe humanoid needs more iterations
   - Contact point ordering — does the order of constraint rows matter?
   - Numerical precision — float64 vs float32 differences
   - Warm-starting — is warmstart working correctly?
4. **Fix or document**: Either fix the precision issue or establish the acceptable tolerance and enable the test with a wider tolerance.

## Files to Touch

- `test/solver_test.py` — enable humanoid, adjust tolerances
- `mujoco_torch/_src/solver.py` — if algorithmic fixes are needed

## Tests

- The fix IS the test: enable humanoid in solver_test.py and make it pass.

## Submission Instructions

- If your changes have more than one step, use ghstack to submit. ghstack sends each commit as a separate PR, so make sure each commit message is a proper PR name.
- BugFix and features must have a test in the same commit.
- You can submit PRs; if you do, monitor the runs using gh.
