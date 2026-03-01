## PR #25 Review: Benchmarks, performance optimizations, and DeviceCopy fix

This is a substantial PR covering benchmarks for 5 models, performance optimizations (+90% with inductor config), the `update_()` in-place mutation path, DeviceCopy elimination, and multiple bug fixes. Overall the changes look well-structured, but there are several issues worth addressing before merge.

### Architecture & Design

**1. `_device_precomp` as a side-channel dict -- fragile propagation**

The `_device_precomp` dict is stored via `object.__setattr__` and propagated by monkey-patching `Model.to` and `Model.clone`. This is fragile because:

- `Model.replace()` is *not* overridden, so `m.replace(...)` will silently drop `_device_precomp`.
- Any other code path that creates a new Model (e.g. `tree_replace`) will also lose it.
- If someone calls `MjTensorClass.to(model, ...)` directly (bypassing the monkey-patch), `_device_precomp` is lost.

Suggestion: override `replace` for `Model` as well (or at minimum add a comment documenting this invariant). Better yet, consider a `__init_subclass__` or `__post_init__` hook instead of monkey-patching.

**2. `update_()` safety with `d.clone(recurse=False)` at `step()` entry**

The defensive clone at the top of `step()` (line 362 of `forward.py`) is correct for caller semantics, but the contract is subtle: every function *below* `step()` now assumes exclusive ownership of `d`. If any future code path calls `forward()` or `_velocity()` *without* the clone, silent data corruption will occur. This should at minimum be documented with a comment at each `update_()` call site, or better, asserted via a debug flag.

**3. `_rungekutta4` clone**

```352:362:mujoco_torch/_src/forward.py
def step(m: Model, d: Data, fixed_iterations: bool = False) -> Data:
    // ...
    d = d.clone(recurse=False)
    d = forward(m, d, fixed_iterations=fixed_iterations)
```

And then inside `_rungekutta4`:

```269:271:mujoco_torch/_src/forward.py
def _rungekutta4(m: Model, d: Data, fixed_iterations: bool = False) -> Data:
    """Runge-Kutta explicit order 4 integrator."""
    d_t0 = d.clone(recurse=False)
```

That's two `clone(recurse=False)` calls per RK4 step. The first (in `step()`) protects the caller. The second (in `_rungekutta4`) protects `d_t0` from in-place mutations to `d`. This is correct but worth a comment explaining *why* the second clone is needed despite the first.

### Bugs & Correctness Issues

**4. `_advance` has a semantic change in `qvel` handling**

```237:248:mujoco_torch/_src/forward.py
    # advance velocities
    new_qvel = d.qvel + qacc * m.opt.timestep

    # advance positions with qvel if given, new_qvel otherwise (semi-implicit)
    qvel_for_pos = new_qvel if qvel is None else qvel
    // ...
    d.update_(qvel=new_qvel, act=act, qpos=qpos, time=time)
    return d
```

The old code was:

```python
d = d.replace(qvel=d.qvel + qacc * m.opt.timestep)
qvel = d.qvel if qvel is None else qvel  # d.qvel is already updated
```

In the old code, when `qvel is None`, the position update used the *already-updated* `d.qvel` (semi-implicit Euler). In the new code, `qvel_for_pos = new_qvel` -- which is the same value. So the behavior is equivalent. Good, but the subtle change of evaluation order deserves a comment.

**5. `_cast_float` won't recurse into nested structures**

In `device.py`, the dtype cast only handles top-level tensors in `init_kwargs` and `derived_kwargs`:

```839:843:mujoco_torch/_src/device.py
    if dtype is not None:
        for k, v in init_kwargs.items():
            init_kwargs[k] = _cast_float(v, dtype)
        for k, v in derived_kwargs.items():
            derived_kwargs[k] = _cast_float(v, dtype)
```

If any value is a nested `MjTensorClass` (which it can be -- `Model` contains `Option`, `Stat`, etc.), those inner tensors won't be cast. The recursive `device_put(field_value, dtype=dtype)` call handles the sub-structs, but the `_cast_float` loop over `init_kwargs` is redundant for those and only effective for direct tensor values. This is fine but could be confusing -- the `_cast_float` call is a no-op for non-tensor values, so it doesn't break anything.

**6. `fixed_loop` doesn't propagate through `_implicit`**

`_implicit` (the `IMPLICITFAST` integrator) calls `forward()` internally but doesn't pass `fixed_iterations`:

```349:349:mujoco_torch/_src/forward.py
    return _advance(m, d, d.act_dot, qacc)
```

The `step()` function calls `forward(m, d, fixed_iterations=...)` before the integrator, then calls `_euler`/`_rungekutta4`/`_implicit`. Only `_rungekutta4` passes `fixed_iterations` to its inner `forward()` call. If `_implicit` had an inner `forward()`, it would be missed. Currently `_implicit` doesn't call `forward()` again, so this is fine, but worth noting.

### Code Quality

**7. Duplicate `clear_compile_caches()` in bench scripts**

Both `bench_phase1.py` and `bench_phase2.py` have identical `clear_compile_caches()` functions. This should be extracted to a shared helper (e.g. in `benchmarks/_helpers.py`).

**8. `getattr` with fallback in bench scripts**

```python
getattr(torch._inductor.config, "cache_dir", None),
```

and:

```python
def _cuda_time(e):
    return getattr(e, "cuda_time_total", getattr(e, "device_time_total", 0))
```

These use `getattr` with defaults, which conflicts with the user rule about not adding defensive fallbacks. Since these are in benchmark scripts (not library code) and handle PyTorch version differences, this is borderline acceptable, but worth flagging.

**9. Local import in `_model_to` and `device_put`**

```853:856:mujoco_torch/_src/types.py
    from mujoco_torch._src.scan import (  # circular dep
        _resolve_cached_tensors,
        warm_device_caches,
    )
```

and in `device.py`:

```850:851:mujoco_torch/_src/device.py
        from mujoco_torch._src.scan import _resolve_cached_tensors
        types._build_device_precomp(result, torch.device("cpu"), _resolve_cached_tensors)
```

The circular dep comment justifies the local import in `types.py`. The one in `device.py` is less justified -- `device.py` already imports from `types`, and `scan.py` likely doesn't import from `device.py`, so this could be a top-level import. Worth checking.

**10. `sensor.py` still has a dead `_groups_to_device` function**

The diff adds a `_groups_to_device` helper function at the top of `sensor.py` but it's never called -- all call sites now use `m._device_precomp["sensor_groups_*_py"]` directly. This function should be removed if it's dead code.

**11. `bench_phase1.py` uses `getattr` for `cache_dir` and `etils` import**

```python
from etils import epath
xml = (epath.resource_path("mujoco_torch") / "test_data" / f"{MODEL}.xml").read_text()
```

But `bench_phase2.py` uses `load_test_file`. These should be consistent.

### Nits

- `force_mask.to(dtype=d.actuator_force.dtype)` in `sensor.py` line 366 -- this still does a `.to()` call that could be a device transfer if the tensor is on the wrong device. Since `_device_precomp` resolves tensors, the device should match, but the `.to(dtype=...)` call should be fine.

- The `scratch/` directory is gitignored but `plot_bench.py` is added to it and referenced in the README. This means users who clone the repo won't have the plot script. Consider moving it to `benchmarks/` or removing the gitignore for `scratch/`.

- `TODO/03-compile-bugs.md` line 274: `(TODO: find or file the issue)` -- this should be resolved before merge.

### Summary

The PR delivers solid performance improvements with good methodology. Main concerns:

1. **`_device_precomp` propagation** -- `replace()` and `tree_replace()` on `Model` will silently lose it
2. **Dead code** -- `_groups_to_device` is defined but unused
3. **`scratch/` gitignored but referenced** -- plot script inaccessible to cloners
4. **Bench script duplication** -- `clear_compile_caches()` duplicated in two files

The core optimizations (`update_()`, DeviceCopy elimination, inductor config, fixed-iteration solver) are well-designed and the correctness strategy (clone at `step()` entry, clone `d_t0` in RK4) is sound.
