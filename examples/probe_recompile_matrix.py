"""Matrix probe: which combination of compile/vmap drops UnbatchedTensor wrap?

The handover's open question is where the UnbatchedTensor wrapping on
`nefc`/`ncon` is dropped between call 1 and call 2 of compile(vmap(step)).
This probe runs step under four configurations and reports the type of
`d.nefc`/`d.ncon` after each of 3 successive calls:

  1. eager (no compile, no vmap, unbatched)
  2. vmap only (no compile)
  3. compile only (no vmap, unbatched)
  4. compile + vmap

For each configuration we also re-expose the underlying dict entry via
`d._tensordict['nefc']` so we can see whether the type is preserved at
the storage layer (not just re-wrapped on attribute access).

Run on steve GPU 5 (same as probe_recompile_gpu5.py). Expected signal:
the configuration whose call-2 type DIFFERS from call-1 is the layer
that's dropping the wrapper.
"""

import os
import sys
import time

os.environ.setdefault("TORCH_LOGS", "recompiles")
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "5")

import mujoco  # noqa: E402
import mujoco_torch  # noqa: E402
import torch  # noqa: E402
import torch._dynamo  # noqa: E402
from tensordict import UnbatchedTensor  # noqa: E402

from mujoco_torch._src import test_util  # noqa: E402

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
BATCH = 64
MODEL = "humanoid.xml"


def types_of(d):
    """Return ((attr_type, td_type, td_id), (attr_type, td_type, td_id))."""
    nefc_attr = type(d.nefc).__name__
    ncon_attr = type(d.ncon).__name__
    # Poke at the backing dict directly — this is where Dynamo's
    # _is_unbatched guard actually looks for the wrapper-subclass type.
    # Log id(type(x)) so we can see whether the CLASS OBJECT is the same
    # across calls (type_id guards compare class identity, not name).
    backing = d._tensordict._tensordict
    if "nefc" in backing:
        nefc_t = type(backing["nefc"])
        nefc_td = f"{nefc_t.__name__}@{id(nefc_t)}"
    else:
        nefc_td = "<missing>"
    if "ncon" in backing:
        ncon_t = type(backing["ncon"])
        ncon_td = f"{ncon_t.__name__}@{id(ncon_t)}"
    else:
        ncon_td = "<missing>"
    return (nefc_attr, nefc_td), (ncon_attr, ncon_td)


def stride_of(x):
    try:
        inner = x.data if isinstance(x, UnbatchedTensor) else x
        return tuple(inner.stride())
    except Exception:
        return None


def describe(tag, d):
    (nefc_a, nefc_td), (ncon_a, ncon_td) = types_of(d)
    sc = d.subtree_com
    sc_info = f"shape={tuple(sc.shape)} stride={tuple(sc.stride())}"
    xi = d.xipos
    xi_info = f"shape={tuple(xi.shape)} stride={tuple(xi.stride())}"
    print(
        f"  {tag}: "
        f"nefc[td={nefc_td}, stride={stride_of(d.nefc)}]  "
        f"ncon[td={ncon_td}, stride={stride_of(d.ncon)}]\n"
        f"       subtree_com {sc_info}  xipos {xi_info}",
        flush=True,
    )


def make_batch(batched: bool):
    m_mj = test_util.load_test_file(MODEL)
    mx = mujoco_torch.device_put(m_mj).to(DEVICE)
    d_mj = mujoco.MjData(m_mj)
    with torch.device("cpu"):
        dx0 = mujoco_torch.device_put(d_mj)
    dx0 = dx0.to(DEVICE)
    if batched:
        return mx, dx0.expand(BATCH).clone()
    return mx, dx0


def run_scenario(name, step_fn, mx, dx):
    print(f"\n=== {name} ===", flush=True)
    torch._dynamo.reset()
    describe("input ", dx)
    for i in range(3):
        t0 = time.time()
        dx = step_fn(mx, dx)
        if DEVICE == "cuda":
            torch.cuda.synchronize()
        dt = time.time() - t0
        describe(f"call {i + 1} ({dt:.2f}s)", dx)


def main():
    print(f"torch {torch.__version__}  device={DEVICE}", flush=True)
    if DEVICE == "cuda":
        print(f"gpu: {torch.cuda.get_device_name()}", flush=True)

    scenarios = [
        ("eager (unbatched)", lambda m, d: mujoco_torch.step(m, d), False),
        ("vmap only", torch.vmap(mujoco_torch.step, in_dims=(None, 0)), True),
        (
            "compile only (unbatched)",
            torch.compile(mujoco_torch.step, fullgraph=True),
            False,
        ),
        (
            "compile + vmap",
            torch.compile(
                torch.vmap(mujoco_torch.step, in_dims=(None, 0)), fullgraph=True
            ),
            True,
        ),
    ]

    only = sys.argv[1] if len(sys.argv) > 1 else None
    for name, fn, batched in scenarios:
        if only and only not in name:
            continue
        mx, dx = make_batch(batched)
        try:
            run_scenario(name, fn, mx, dx)
        except Exception as e:
            print(f"  FAILED: {type(e).__name__}: {str(e)[:300]}", flush=True)


if __name__ == "__main__":
    main()
