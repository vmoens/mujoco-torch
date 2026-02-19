# Publish Benchmark Results

**Priority:** High
**Category:** Docs/Performance
**Difficulty:** Low

## Problem

The repo has benchmark scripts (`examples/e2e_comparison.py`, `examples/batched_comparison.py`) but no published results. The value proposition of mujoco-torch is performance on PyTorch hardware — this needs to be demonstrated with numbers.

## What to Do

1. Run `examples/e2e_comparison.py` and capture output:
   - MuJoCo C baseline (steps/s)
   - mujoco-torch eager (steps/s)
   - mujoco-torch compiled (steps/s)
   - MJX JAX jit (steps/s) — if JAX is available
   - Numerical accuracy (max |delta| for qpos, qvel)
2. Run `examples/batched_comparison.py` and capture output:
   - Batch sizes 1, 4, 16, 64
   - All backends
3. Format results into a table and add to the README (see TODO/02-readme.md) or a separate `BENCHMARKS.md`.
4. Include system info: CPU model, GPU (if applicable), PyTorch version, JAX version, MuJoCo version.
5. Add a note about how to reproduce: "Run `python examples/e2e_comparison.py`".

## Environment

- Run on CPU first (that's what CI has)
- If you have GPU access, include GPU numbers too
- Use float64 for accuracy comparison, float32 for speed comparison

## Submission Instructions

- If your changes have more than one step, use ghstack to submit. ghstack sends each commit as a separate PR, so make sure each commit message is a proper PR name.
- BugFix and features must have a test in the same commit.
- You can submit PRs; if you do, monitor the runs using gh.
