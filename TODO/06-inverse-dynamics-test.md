# Add Inverse Dynamics Test

**Status** In progress
**Priority:** High
**Category:** Test
**Difficulty:** Low

## Problem

There is no test for `mujoco_torch.inverse()`. The function is currently broken (see TODO/01-fix-inverse-dynamics.md), but even once fixed, there's no regression test. Every other major subsystem (forward, constraint, solver, collision, smooth, passive, support) has tests.

## What to Do

1. Create `test/inverse_test.py` following the pattern of existing tests (e.g. `test/forward_test.py`).
2. Test cases should include:
   - **No-contact case**: Load a simple model (e.g. `pendula.xml`), set `qacc` to known values, call `inverse()`, compare `qfrc_inverse` against `mujoco.mj_inverse`.
   - **With-contact case**: Load `ant.xml`, step forward a few times to generate contacts, then call `inverse()` and compare.
   - **Discrete inverse** (`INVDISCRETE` flag): If supported, test with Euler and ImplicitFast integrators.
3. Use the same tolerance patterns as `forward_test.py` (typically `atol=1e-8` for float64).

## Depends On

- TODO/01-fix-inverse-dynamics.md (the bug must be fixed first, or the test will just confirm the crash)

## Submission Instructions

- branch out from origin/main. Commit only changed / relevant new files.
- If your changes have more than one step, use ghstack to submit. ghstack sends each commit as a separate PR, so make sure each commit message is a proper PR name.
- If the changes are contained in one single commit, use `gh pr` instead.
- BugFix and features must have a test in the same commit.
- You can submit PRs; if you do, monitor the runs using gh.
