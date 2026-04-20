#!/usr/bin/env python3
"""Unified multi-env / multi-backend / multi-mode benchmark driver.

Produces one JSONL row per (env, batch_size) combination for a fixed
``(mode, backend)`` pair. Modes:

- ``compile``  : raw ``compile(vmap(step), fullgraph=True)`` throughput
                 (README-style, no env / no policy / no replay buffer).
- ``collector``: TorchRL ``SyncDataCollector`` + ``LazyTensorStorage(ndim=2)``
                 replay buffer, nested ``compile(vmap(step))`` + Python
                 frame-skip loop. End-to-end RL throughput.

Backends:

- ``torch`` : mujoco-torch compiled with ``torch.compile``.
- ``mjx``   : MuJoCo MJX via ``jax.jit(jax.vmap(mjx.step))``.

MJX + collector is not supported (MJX doesn't plug into TorchRL cleanly);
requesting that combination aborts early with ``status=skip``.

Usage::

    python examples/bench_all.py --env halfcheetah --mode compile --backend torch \\
        --batch_sizes 131072 65536 16384 4096 1024 128 64 16 4 1 \\
        --out ~/bench_all.jsonl
"""

import argparse
import gc
import json
import os
import time
from pathlib import Path

# --- P0 cold-start reductions (must be set before importing torch._inductor).
os.environ.setdefault(
    "TORCHINDUCTOR_CACHE_DIR",
    str(Path("~/.cache/mujoco_torch/inductor").expanduser()),
)
os.environ.setdefault("TORCHINDUCTOR_FX_GRAPH_CACHE", "1")
Path(os.environ["TORCHINDUCTOR_CACHE_DIR"]).mkdir(parents=True, exist_ok=True)

import torch

DEFAULT_BATCH_SIZES = [131072, 65536, 16384, 4096, 1024, 128, 64, 16, 4, 1]

# --- torch.compile sweep ---------------------------------------------------

COMPILE_WARMUP_ITERS = 5
WARMUP_NSTEPS = 100
MEASURE_NSTEPS = 1000

# --- collector -------------------------------------------------------------

DEFAULT_STEPS_PER_WRITE = 100
DEFAULT_WARMUP_WRITES = 2
DEFAULT_MEASURE_WRITES = 10
STARTUP_TIMEOUT_S = 900.0
RB_TOTAL_ELEMENTS = 10_000_000
RB_DEVICE = "cpu"


# ---------------------------------------------------------------------------
# torch backend -- compile sweep (raw physics)
# ---------------------------------------------------------------------------


def _bench_compile_torch(env_name: str, batch_size: int, device: str) -> dict:
    import torch._inductor.config as inductor_config

    import mujoco_torch
    from benchmarks._helpers import load_model, make_batch, warm_caches

    # MJT_TUNED=1 enables Inductor coordinate-descent tuning + aggressive fusion
    # (the "tuned" column in the README benchmark table). Default: off, matches
    # the untuned README baseline.
    tuned = os.environ.get("MJT_TUNED", "0") == "1"
    inductor_config.coordinate_descent_tuning = tuned
    inductor_config.aggressive_fusion = tuned

    m_mj = load_model(env_name)
    with torch.device("cpu"):
        mx = mujoco_torch.device_put(m_mj)
    mx = mx.to(device)
    warm_caches(mx, m_mj, device)

    vmap_step = torch.vmap(lambda d: mujoco_torch.step(mx, d))
    compiled_fn = torch.compile(vmap_step, fullgraph=True)

    d_batch = make_batch(mx, m_mj, batch_size, device)

    t_compile_start = time.perf_counter()
    compile_s = None
    for i in range(COMPILE_WARMUP_ITERS):
        d_batch = compiled_fn(d_batch)
        torch.cuda.synchronize()
        if i == 0:
            compile_s = time.perf_counter() - t_compile_start

    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(WARMUP_NSTEPS):
        d_batch = compiled_fn(d_batch)
    torch.cuda.synchronize()
    warmup_s = time.perf_counter() - t0
    warmup_sps = batch_size * WARMUP_NSTEPS / warmup_s

    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(MEASURE_NSTEPS):
        d_batch = compiled_fn(d_batch)
    torch.cuda.synchronize()
    run_s = time.perf_counter() - t0
    steps_per_s = batch_size * MEASURE_NSTEPS / run_s

    return {
        "steps_per_s": steps_per_s,
        "compile_s": compile_s,
        "run_s": run_s,
        "nsteps": MEASURE_NSTEPS,
        "warmup_sps": warmup_sps,
        "warmup_s": warmup_s,
    }


# ---------------------------------------------------------------------------
# torch backend -- eager vmap (no compile)
# ---------------------------------------------------------------------------


def _bench_vmap_torch(env_name: str, batch_size: int, device: str) -> dict:
    import mujoco_torch
    from benchmarks._helpers import load_model, make_batch, warm_caches

    m_mj = load_model(env_name)
    with torch.device("cpu"):
        mx = mujoco_torch.device_put(m_mj)
    mx = mx.to(device)
    warm_caches(mx, m_mj, device)

    vmap_step = torch.vmap(lambda d: mujoco_torch.step(mx, d))

    d_batch = make_batch(mx, m_mj, batch_size, device)

    for _ in range(COMPILE_WARMUP_ITERS):
        d_batch = vmap_step(d_batch)
        torch.cuda.synchronize()

    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(WARMUP_NSTEPS):
        d_batch = vmap_step(d_batch)
    torch.cuda.synchronize()
    warmup_s = time.perf_counter() - t0
    warmup_sps = batch_size * WARMUP_NSTEPS / warmup_s

    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(MEASURE_NSTEPS):
        d_batch = vmap_step(d_batch)
    torch.cuda.synchronize()
    run_s = time.perf_counter() - t0
    steps_per_s = batch_size * MEASURE_NSTEPS / run_s

    return {
        "steps_per_s": steps_per_s,
        "compile_s": None,
        "run_s": run_s,
        "nsteps": MEASURE_NSTEPS,
        "warmup_sps": warmup_sps,
        "warmup_s": warmup_s,
    }


# ---------------------------------------------------------------------------
# MuJoCo C backend -- stock CPU sequential step baseline (B=1 only)
# ---------------------------------------------------------------------------


def _bench_compile_mujoco_c(env_name: str, batch_size: int, device: str) -> dict:
    import mujoco
    import numpy as np

    from benchmarks._helpers import SEED, load_model

    if batch_size != 1:
        raise NotImplementedError(
            f"mujoco_c backend is a single-stream CPU baseline; only batch_size=1 is supported (got {batch_size})"
        )

    m_mj = load_model(env_name)
    d = mujoco.MjData(m_mj)
    d.qvel[:] = 0.01 * np.random.RandomState(SEED).randn(m_mj.nv)

    for _ in range(10):
        mujoco.mj_step(m_mj, d)

    t0 = time.perf_counter()
    for _ in range(WARMUP_NSTEPS):
        mujoco.mj_step(m_mj, d)
    warmup_s = time.perf_counter() - t0
    warmup_sps = WARMUP_NSTEPS / warmup_s

    t0 = time.perf_counter()
    for _ in range(MEASURE_NSTEPS):
        mujoco.mj_step(m_mj, d)
    run_s = time.perf_counter() - t0
    steps_per_s = MEASURE_NSTEPS / run_s

    return {
        "steps_per_s": steps_per_s,
        "compile_s": None,
        "run_s": run_s,
        "nsteps": MEASURE_NSTEPS,
        "warmup_sps": warmup_sps,
        "warmup_s": warmup_s,
    }


# ---------------------------------------------------------------------------
# MJX backend -- compile sweep
# ---------------------------------------------------------------------------


def _bench_compile_mjx(env_name: str, batch_size: int, device: str) -> dict:
    import jax
    import mujoco
    import numpy as np
    from mujoco import mjx

    from benchmarks._helpers import load_model, SEED

    jax.config.update("jax_enable_x64", True)

    m_mj = load_model(env_name)
    mx_jax = mjx.put_model(m_mj)
    step_fn = jax.jit(jax.vmap(mjx.step, in_axes=(None, 0)))

    rng = np.random.RandomState(SEED)
    qvels = 0.01 * rng.randn(batch_size, m_mj.nv)

    d_ref = mujoco.MjData(m_mj)
    dx_single = mjx.put_data(m_mj, d_ref)

    def _tile_leaf(x):
        if not hasattr(x, "ndim"):
            return x
        if x.ndim == 0:
            return jax.numpy.broadcast_to(x, (batch_size,))
        return jax.numpy.tile(x, (batch_size,) + (1,) * x.ndim)

    dx_jax = jax.tree.map(_tile_leaf, dx_single)
    dx_jax = dx_jax.replace(qvel=jax.numpy.array(qvels))

    # First call includes jit compile.
    t_compile_start = time.perf_counter()
    dx_warm = step_fn(mx_jax, dx_jax)
    jax.block_until_ready(dx_warm.qpos)
    compile_s = time.perf_counter() - t_compile_start

    # Warmup.
    t0 = time.perf_counter()
    dx_run = dx_warm
    for _ in range(WARMUP_NSTEPS):
        dx_run = step_fn(mx_jax, dx_run)
    jax.block_until_ready(dx_run.qpos)
    warmup_s = time.perf_counter() - t0
    warmup_sps = batch_size * WARMUP_NSTEPS / warmup_s

    # Measurement.
    t0 = time.perf_counter()
    for _ in range(MEASURE_NSTEPS):
        dx_run = step_fn(mx_jax, dx_run)
    jax.block_until_ready(dx_run.qpos)
    run_s = time.perf_counter() - t0
    steps_per_s = batch_size * MEASURE_NSTEPS / run_s

    return {
        "steps_per_s": steps_per_s,
        "compile_s": compile_s,
        "run_s": run_s,
        "nsteps": MEASURE_NSTEPS,
        "warmup_sps": warmup_sps,
        "warmup_s": warmup_s,
    }


# ---------------------------------------------------------------------------
# torch backend -- TorchRL collector + replay buffer
# ---------------------------------------------------------------------------


def _install_nested_compile(env) -> float:
    import mujoco_torch

    step_fn = lambda d: mujoco_torch.step(env.mx, d)  # noqa: E731
    vmap_step = torch.vmap(step_fn)
    compiled = torch.compile(vmap_step, fullgraph=True)
    fs = env.FRAME_SKIP

    def nested_physics(d, _compiled=compiled, _fs=fs):
        for _ in range(_fs):
            d = _compiled(d)
        return d

    env._physics_step = nested_physics
    t0 = time.perf_counter()
    env._dx = nested_physics(env._dx)
    torch.cuda.synchronize()
    return time.perf_counter() - t0


def _bench_collector_torch(env_name: str, batch_size: int, device: str) -> dict:
    from torchrl.collectors import RandomPolicy, SyncDataCollector
    from torchrl.data import LazyTensorStorage, ReplayBuffer

    from mujoco_torch.zoo import ENVS

    EnvCls = ENVS[env_name]
    env = EnvCls(num_envs=batch_size, device=device, compile_step=False)
    env.reset()
    compile_s = _install_nested_compile(env)
    env.reset()

    policy = RandomPolicy(env.action_spec)
    fpb = batch_size * DEFAULT_STEPS_PER_WRITE

    rb_max_size = max(1, RB_TOTAL_ELEMENTS // batch_size)
    rb = ReplayBuffer(
        storage=LazyTensorStorage(
            max_size=rb_max_size, device=RB_DEVICE, ndim=2
        )
    )
    collector = SyncDataCollector(
        env,
        policy,
        frames_per_batch=fpb,
        total_frames=-1,
        device=device,
        storing_device=device,
        replay_buffer=rb,
        extend_buffer=True,
    )

    stats = {"n_extends": 0, "n_frames": 0, "last_shape": None}
    orig_extend = rb.extend

    def _counting_extend(td):
        try:
            n = int(td.numel()) if hasattr(td, "numel") else 0
        except Exception:
            n = 0
        shape = tuple(getattr(td, "batch_size", ())) if td is not None else ()
        result = orig_extend(td)
        stats["n_extends"] += 1
        stats["n_frames"] += n
        stats["last_shape"] = shape
        return result

    rb.extend = _counting_extend

    def _wait_for(target: int, label: str, timeout: float):
        t_wait = time.perf_counter()
        while stats["n_extends"] < target:
            if time.perf_counter() - t_wait > timeout:
                raise TimeoutError(
                    f"rb.extend target {target} ({label}) not reached in "
                    f"{timeout:.0f}s (have {stats['n_extends']}, "
                    f"last_shape={stats['last_shape']})"
                )
            time.sleep(0.1)

    t_start = time.perf_counter()
    collector.start()
    try:
        _wait_for(1, "first extend", STARTUP_TIMEOUT_S)
        startup_s = time.perf_counter() - t_start
        print(
            f"  first extend: shape={stats['last_shape']}  after {startup_s:.1f}s",
            flush=True,
        )

        _wait_for(1 + DEFAULT_WARMUP_WRITES, "warmup", STARTUP_TIMEOUT_S)

        n0_ext = stats["n_extends"]
        n0_frames = stats["n_frames"]
        t0 = time.perf_counter()
        _wait_for(
            n0_ext + DEFAULT_MEASURE_WRITES,
            "measure",
            max(STARTUP_TIMEOUT_S, 120.0 * DEFAULT_MEASURE_WRITES),
        )
        elapsed = time.perf_counter() - t0
        n1_frames = stats["n_frames"]
        n_writes = stats["n_extends"] - n0_ext
        frames_written = n1_frames - n0_frames
    finally:
        try:
            collector.async_shutdown()
        except Exception as e:
            print(f"  (async_shutdown: {type(e).__name__}: {e})", flush=True)

    # Convert the collector's frames_per_second to steps_per_second. The
    # collector's "frames" count agent-steps (one per FRAME_SKIP physics
    # steps); `steps_per_s` counts *physics* steps so it is directly
    # comparable to the compile-mode numbers.
    frame_skip = env.FRAME_SKIP
    collector_fps = frames_written / elapsed
    steps_per_s = collector_fps * frame_skip

    return {
        "collector_fps": collector_fps,
        "steps_per_s": steps_per_s,
        "frames_written": frames_written,
        "n_writes": int(n_writes),
        "last_extend_shape": list(stats["last_shape"]) if stats["last_shape"] else None,
        "s_per_write": elapsed / max(n_writes, 1),
        "measure_s": elapsed,
        "compile_s": compile_s,
        "startup_s": startup_s,
        "frame_skip": frame_skip,
        "frames_per_batch": fpb,
    }


# ---------------------------------------------------------------------------
# Dispatch + main loop
# ---------------------------------------------------------------------------


def bench_one(env_name: str, batch_size: int, mode: str, backend: str, device: str) -> dict:
    if mode == "compile" and backend == "torch":
        return _bench_compile_torch(env_name, batch_size, device)
    if mode == "compile" and backend == "mjx":
        return _bench_compile_mjx(env_name, batch_size, device)
    if mode == "compile" and backend == "mujoco_c":
        return _bench_compile_mujoco_c(env_name, batch_size, device)
    if mode == "vmap" and backend == "torch":
        return _bench_vmap_torch(env_name, batch_size, device)
    if mode == "collector" and backend == "torch":
        return _bench_collector_torch(env_name, batch_size, device)
    if mode == "collector" and backend == "mjx":
        raise NotImplementedError("MJX + TorchRL collector not implemented")
    raise ValueError(f"unknown (mode, backend): ({mode}, {backend})")


def append_jsonl(path: Path, row: dict):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a") as f:
        f.write(json.dumps(row) + "\n")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--env", required=True, help="Env name, e.g. halfcheetah")
    p.add_argument("--mode", choices=("compile", "vmap", "collector"), required=True)
    p.add_argument("--backend", choices=("torch", "mjx", "mujoco_c"), required=True)
    p.add_argument(
        "--tuned",
        action="store_true",
        help="For (mode=compile, backend=torch): enable Inductor coordinate-descent tuning "
             "+ aggressive fusion (the 'tuned' column) and tag output rows with tuned=True.",
    )
    p.add_argument(
        "--batch_sizes",
        type=int,
        nargs="+",
        default=DEFAULT_BATCH_SIZES,
        help="Batch sizes (default: decreasing 131072..1).",
    )
    p.add_argument("--device", default="cuda")
    p.add_argument("--out", type=Path, required=True)
    args = p.parse_args()
    out_path = args.out.expanduser()

    torch.set_default_dtype(torch.float64)

    if args.tuned:
        if not (args.mode == "compile" and args.backend == "torch"):
            raise ValueError("--tuned is only meaningful for (mode=compile, backend=torch)")
        os.environ["MJT_TUNED"] = "1"

    if args.mode == "collector" and args.backend == "mjx":
        print(
            f"=== SKIP: (mode=collector, backend=mjx) not supported for env={args.env} ===",
            flush=True,
        )
        for B in args.batch_sizes:
            append_jsonl(
                out_path,
                {
                    "env": args.env,
                    "batch_size": B,
                    "mode": args.mode,
                    "backend": args.backend,
                    "status": "skip",
                    "reason": "MJX + TorchRL collector not implemented",
                },
            )
        return

    print(f"=== bench_all: env={args.env}  mode={args.mode}  backend={args.backend} ===", flush=True)
    if args.device == "cuda":
        print(f"  device: {torch.cuda.get_device_name()}", flush=True)
    else:
        print(f"  device: {args.device}", flush=True)
    print(f"  cache:  {os.environ.get('TORCHINDUCTOR_CACHE_DIR')}", flush=True)
    print(f"  out:    {out_path}", flush=True)
    print(f"  batch_sizes: {args.batch_sizes}\n", flush=True)

    for B in args.batch_sizes:
        tag = f"{args.env}/{args.mode}/{args.backend}/B={B}"
        print(f"--- {tag} ---", flush=True)
        torch._dynamo.reset()
        torch.compiler.reset()
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        row_base = {
            "env": args.env,
            "batch_size": B,
            "mode": args.mode,
            "backend": args.backend,
            "tuned": bool(args.tuned),
        }
        try:
            t_total = time.perf_counter()
            result = bench_one(args.env, B, args.mode, args.backend, args.device)
            total_s = time.perf_counter() - t_total
        except torch.cuda.OutOfMemoryError as e:
            row = {**row_base, "status": "OOM", "error": str(e).splitlines()[0] if str(e) else ""}
            append_jsonl(out_path, row)
            print(f"[{tag}] OOM -- skipping\n", flush=True)
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            continue
        except Exception as e:
            row = {**row_base, "status": "error", "error": f"{type(e).__name__}: {e}"}
            append_jsonl(out_path, row)
            print(f"[{tag}] ERROR: {type(e).__name__}: {e}\n", flush=True)
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            continue

        row = {**row_base, **result, "total_s": total_s, "status": "ok"}
        append_jsonl(out_path, row)
        sps = row.get("steps_per_s", 0.0)
        cs = row.get("compile_s", 0.0) or 0.0
        print(
            f"[{tag}] steps/s={sps:>14,.0f}  compile={cs:6.1f}s  total={total_s:6.1f}s\n",
            flush=True,
        )

    print(f"=== {args.env}/{args.mode}/{args.backend} done ===", flush=True)


if __name__ == "__main__":
    main()
