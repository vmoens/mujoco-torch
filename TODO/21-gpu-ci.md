# Add GPU CI

**Priority:** Low
**Category:** Infrastructure
**Difficulty:** Medium

## Problem

CI only runs on `ubuntu-latest` CPU. Since the main value of mujoco-torch is GPU acceleration, GPU correctness is not continuously verified. Regressions on CUDA could go unnoticed.

## What to Do

1. **Option A: GitHub Actions with self-hosted GPU runner**
   - Add a workflow that runs on a self-hosted runner with an NVIDIA GPU.
   - Install CUDA PyTorch, run the test suite.
   - Trigger: nightly or on-push to main.
2. **Option B: Use a CI service with GPU support**
   - CircleCI has GPU executors.
   - Alternatively, use a cloud VM triggered by GitHub Actions (more complex).
3. **Minimum viable GPU CI**:
   - Run `pytest test/ -x -v` with `CUDA_VISIBLE_DEVICES=0`.
   - Run `examples/e2e_comparison.py` on GPU and verify output.
   - Run `examples/batched_comparison.py` on GPU.
4. **Ensure tests are device-parametrized**:
   - Add a `@pytest.mark.parametrize("device", ["cpu", "cuda"])` or a fixture.
   - Skip CUDA tests if no GPU is available (`pytest.mark.skipif`).

## Files to Touch

- `.github/workflows/gpu-tests.yml` — new workflow
- `test/conftest.py` — device fixture
- Existing tests — parametrize over device

## Submission Instructions

- If your changes have more than one step, use ghstack to submit. ghstack sends each commit as a separate PR, so make sure each commit message is a proper PR name.
- BugFix and features must have a test in the same commit.
- You can submit PRs; if you do, monitor the runs using gh.
