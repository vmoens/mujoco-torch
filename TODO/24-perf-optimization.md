# Performance Optimization: Remaining Items

## Context

All optimization hypotheses (H1–H9) have been investigated and the winning
ones implemented.  Key results on humanoid B=32768 (H200, float64):

| Optimization | Impact |
|---|---|
| H4: Inductor config (`coordinate_descent_tuning + aggressive_fusion`) | **+90%** |
| H6: Float32 mode (`device_put(m, dtype=torch.float32)`) | +23.5% |
| H5: In-place mutation (`update_()` replacing `.replace()`) | Included in baseline |
| H8: Fixed-iteration solver | +6.8% |
| H1: `mode="reduce-overhead"` (CUDA graphs) | Now works (was blocked by tensor aliasing, fixed via `.clone()` in `_take()`) |

---

## Remaining TODO

### 1. Run full benchmark suite with all optimizations

Now that `mode="reduce-overhead"` works, run the full suite (all 5 models,
all batch sizes) with H4 inductor config + reduce-overhead to get final
comparison numbers vs MJX.

### 2. Update README with final benchmark numbers

The README benchmarks are from Feb 28 (pre-optimization).  Update once the
full benchmark suite is re-run.

### 3. Investigate remaining 9 DeviceCopy warnings

During `torch.compile(vmap(step))`, 9 `DeviceCopy in input program` warnings
are emitted by Inductor.  These come from `idx.to(x.device)` calls in
`scan.py:_take()` that are no-ops (idx is already on the right device) but
are still traced.  Not a correctness issue, but noisy and may indicate
unnecessary ops in the compiled graph.
