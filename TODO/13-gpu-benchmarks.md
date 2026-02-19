# GPU Benchmarks

**Priority:** High
**Category:** Performance
**Difficulty:** Medium

## Problem

The entire value proposition of mujoco-torch is GPU-accelerated physics for PyTorch. But there are no GPU benchmark results, and it's unclear if the code even runs correctly on CUDA (CI is CPU-only).

## What to Do

1. **Verify CUDA correctness**: Run the existing test suite on a CUDA device. This may require:
   - Passing `device="cuda"` through `device_put`
   - Ensuring all tensor operations are device-agnostic (no accidental CPU tensors)
   - Fixing any device mismatches
2. **Run benchmarks on GPU**:
   - `examples/e2e_comparison.py` on CUDA
   - `examples/batched_comparison.py` on CUDA with larger batch sizes (64, 256, 1024, 4096)
   - Include `torch.compile` with CUDA (Triton backend)
3. **Compare against MJX on GPU**:
   - Same model, same batch sizes
   - MJX with `jax.devices("gpu")`
4. **Document results** — Add GPU benchmark table to README or BENCHMARKS.md.
5. **Profile hotspots** — Use `torch.profiler` to identify bottlenecks. Common issues:
   - CPU-GPU sync points
   - Small kernel launches
   - Non-batched operations inside vmap

## Submission Instructions

- If your changes have more than one step, use ghstack to submit. ghstack sends each commit as a separate PR, so make sure each commit message is a proper PR name.
- BugFix and features must have a test in the same commit.
- You can submit PRs; if you do, monitor the runs using gh.
