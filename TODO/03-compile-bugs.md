# torch.compile + vmap Graph Break Bug Report

## Summary

Two separate issues prevent `torch.compile(vmap(step), fullgraph=True)` from
working on stock PyTorch (tested on 2.10.0).  Issue 1 has been fixed in this
branch; issue 2 requires an upstream PyTorch change (or the custom fork).

---

## Issue 1 — FIXED: `scan` HOP "Unsafe side effect" with RK4 integrator

**Affected models:** walker2d, hopper (any model with `integrator="RK4"`)

**Root cause:** `_rungekutta4()` used `torch_scan()` which dispatches to
`torch._higher_order_ops.scan` during compilation.  The scan body calls
`forward(m, d)` which accesses module-level dict caches (`_flat_cache`,
`_body_tree_cache` in `scan.py`) and mutates `_DeviceCachedTensor._cache`.
These are "unsafe side effects" that the `scan` HOP contract forbids.

```
torch._dynamo.exc.UncapturedHigherOrderOpError:
  HOP: Unsafe side effect
  Higher Order Operator: torch.ops.higher_order.scan
```

**Fix (already applied):** Unrolled the 3-iteration RK4 loop in
`forward.py:_rungekutta4()` instead of using the `scan` HOP.  Dynamo unrolls
the Python `for i in range(3)` at trace time, so each iteration is traced at
the top level where dict access is fine.

**Verification:** `torch.compile(step, fullgraph=True, backend='aot_eager')`
now passes for walker2d and hopper locally (no CUDA needed).

---

## Issue 2 — OPEN: `while_loop` HOP missing Vmap dispatch rule

**Affected models:** ALL models that hit the constraint solver (i.e., any model
with contacts/constraints — humanoid, ant, walker2d, hopper, halfcheetah).

**Affected paths:**
- `torch.vmap(step)` — fails (even without compile)
- `torch.compile(torch.vmap(step), fullgraph=True)` — fails

**Works fine:**
- `step()` eager — OK
- `torch.compile(step, fullgraph=True)` — OK (no vmap)

### Root cause

`solver.py:while_loop()` dispatches to `torch._higher_order_ops.while_loop`
when inside a functorch transform (vmap) or during compilation:

```python
# solver.py line 44
if torch.compiler.is_compiling() or _inside_functorch():
    return _torch_while_loop(cond_fn, cloning_body, carried_inputs)
```

The `while_loop_op` in stock PyTorch 2.10.0 only has a `Functionalize`
dispatch rule, but **no `Vmap` dispatch rule**:

```python
>>> from torch._higher_order_ops.while_loop import while_loop_op
>>> from torch._C._functorch import TransformType
>>> while_loop_op.functorch_table
{<TransformType.Functionalize: 4>: ...}
>>> TransformType.Vmap in while_loop_op.functorch_table
False
```

For comparison, the `scan` HOP has both:

```python
>>> from torch._higher_order_ops.scan import scan_op
>>> scan_op.functorch_table
{<TransformType.Functionalize: 4>: ..., <TransformType.Vmap: 1>: ...}
```

### Error

```
torch._dynamo.exc.TorchRuntimeError:
  Dynamo failed to run FX node with fake tensors:
  call_function while_loop(...): got KeyError(<TransformType.Vmap: 1>)
```

### Call chain

```
step()
  → forward()
    → solver.solve()                         # solver.py:444
      → while_loop(cond, body, (ctx,))       # dispatches to _torch_while_loop
        → while_loop_op(...)                 # needs Vmap dispatch rule
          → interpreter.process(op, ...)     # KeyError(TransformType.Vmap)
            → _linesearch()                  # nested while_loop inside body
              → while_loop(ls_cond, ls_body) # same error path
```

### Minimal reproduction (stock PyTorch 2.10.0)

```python
import torch
from torch._higher_order_ops.while_loop import while_loop

def cond(x, acc):
    return x < 5.0

def body(x, acc):
    return x + 1.0, acc + x

# Works:
result = while_loop(cond, body, (torch.tensor(0.0), torch.tensor(0.0)))
print("eager:", result)

# Fails (no Vmap rule):
vmapped = torch.vmap(
    lambda x: while_loop(
        lambda s, a: s < 5.0,
        lambda s, a: (s + 1.0, a + s),
        (x, torch.tensor(0.0)),
    )
)
try:
    print("vmap:", vmapped(torch.tensor([0.0, 1.0])))
except Exception as e:
    print(f"vmap FAILED: {type(e).__name__}: {e}")
```

### Fix options

1. **Register a Vmap rule for `while_loop_op`** — this is what the custom
   PyTorch fork at `/root/pytorch` on the cluster does.  The vmap rule
   conceptually runs the while_loop until ALL batch elements converge, using
   `torch.where` to freeze finished elements.

2. **Fall back to Python `while` during vmap-only (no compile)** — change the
   dispatch condition in `solver.py` to only use the HOP when compiling, and
   use a Python loop for eager vmap.  This would fix `torch.vmap(step)` but
   NOT `torch.compile(torch.vmap(step))`:

   ```python
   # This would fix vmap-only, but not compile+vmap
   if torch.compiler.is_compiling():
       return _torch_while_loop(cond_fn, cloning_body, carried_inputs)
   # Fall through to Python while loop (works with eager vmap)
   ```

   **Problem:** the Python `while` loop uses scalar `cond_fn(*val)` which
   returns a single bool.  Under vmap, the condition is batched — different
   elements may converge at different iterations.  A Python loop can't handle
   per-element convergence; only the HOP with a proper Vmap rule can.

3. **Implement a manual vmap-aware while loop** — write a Python loop that
   checks `cond.all()` / `cond.any()` and uses `torch.where` to freeze
   converged elements, then use this when `_inside_functorch()` but not
   compiling.  This is essentially reimplementing the Vmap rule in Python.

### Notes

- The custom PyTorch fork on the cluster (built from `/root/pytorch`) has
  the Vmap dispatch rule and everything works.  This is a stock PyTorch gap.
- The `scan` HOP already has a Vmap rule in stock PyTorch 2.10.0, so this
  appears to be an oversight / incomplete implementation for `while_loop`.
- This is addressed in the custom fork
  (`mujoco-torch-features` branch of `vmoens/pytorch`).  No upstream
  issue has been filed yet; the fix lives only in the fork for now.

---

## Issue 3 — WORKAROUND APPLIED: `BatchedTensor * SymInt` fails inside `while_loop` body

**Affected models:** hopper (any model where `nv` is small enough that the
solver's line‑search path is exercised under vmap + dynamo).

**Affected path:** `torch.vmap(step)` on the cluster fork (which has the
`while_loop` Vmap dispatch rule from Issue 2).

### Root cause

In `solver.py`, model constants are extracted before the `while_loop` call:

```python
nv = int(m.nv)          # line 231 — plain Python int
meaninertia = float(m.stat.meaninertia)  # line 238 — plain Python float
```

Inside the `while_loop` body, `_linesearch` uses them:

```python
smag = math.norm(ctx.search) * meaninertia * max(1, nv)
```

and `_rescale` uses:

```python
return value / (meaninertia * max(1, nv))
```

Even though `nv` is `int(m.nv)` (a concrete Python int), when dynamo traces
the `while_loop` body it can promote closure-captured ints to `SymInt` for
guard purposes. `max(1, nv)` then returns a `SymInt`, and the multiplication
`BatchedTensor * SymInt` is not supported:

```
torch._dynamo.exc.TorchRuntimeError:
  Dynamo failed to run FX node with fake tensors:
  call_function <built-in function mul>(*(BatchedTensor(lvl=1, bdim=0, value=
      FakeTensor(..., device='cuda:0', size=(4,))), Max(1, s26)), **{}):
  got AssertionError('x must be Tensor or scalar, got SymInt')
```

### PyTorch bug

This is arguably a PyTorch bug: `BatchedTensor.__mul__` (and the batching
rule for `aten.mul`) should accept `SymInt` operands the same way it accepts
plain `int`/`float` scalars. The `SymInt` is semantically a scalar — it just
carries symbolic guards. The batching rule's assertion
`x must be Tensor or scalar, got SymInt` is too strict.

Alternatively, dynamo should not promote a concrete `int` closure variable to
`SymInt` when tracing the body of a `while_loop` HOP, since the int was
already materialized before tracing began.

### Workaround (applied)

Precompute `meaninertia * max(1, nv)` as a single `float` constant outside
the closures. Python's `float * int → float` runs at module level (before
dynamo), so the closure captures a plain `float` that dynamo treats as a
compile-time constant:

```python
_inertia_scale = meaninertia * max(1, nv)  # plain float

def _rescale(value):
    return value / _inertia_scale

def _linesearch(ctx):
    smag = math.norm(ctx.search) * _inertia_scale
    ...
```

### Minimal reproduction

```python
import torch
from torch._higher_order_ops.while_loop import while_loop

def demo(x):
    nv = 6  # plain Python int

    def cond(carry):
        return carry.sum() < 100.0

    def body(carry):
        # max(1, nv) may become SymInt under dynamo tracing
        return carry + 1.0 * max(1, nv)

    return while_loop(cond, body, (x,))

# Works in eager:
print(demo(torch.zeros(3)))

# Fails under vmap + compile (if while_loop Vmap rule exists):
# torch.vmap(demo)(torch.zeros(4, 3))
```

### Upstream action

File a PyTorch issue requesting either:
1. Batching rules accept `SymInt` operands (treat them as scalars), or
2. Concrete `int` closure variables are not promoted to `SymInt` inside HOP
   body tracing.
