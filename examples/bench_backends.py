#!/usr/bin/env python3
"""Throughput benchmark: mujoco-torch (eager / compiled) vs MJX vs MuJoCo C.

Device-agnostic (``--device cpu`` on a laptop, ``--device cuda`` on a GPU box).
One JSONL row per (backend, mode, env, batch size); ``--table`` renders the
JSONL as README-style markdown tables (steps/s, peak memory, compile time).

Methodology (matches the README tables): the first call compiles / JITs and is
timed separately; ``--warmup`` steps follow; then a timed block whose length
adapts to ``--budget`` seconds (10 to 1000 steps).  steps/s = batch_size x
nsteps / elapsed, bracketed by ``torch.cuda.synchronize`` /
``jax.block_until_ready``.

Examples::

    # MicroDuck, compiled vs MJX, on a MacBook
    python examples/bench_backends.py --env microduck --microduck-root ~/repos/microduck_rl \\
        --backend torch mjx --batch_sizes 16 128 1024 --out md.jsonl
    # Same on CUDA, adding the tuned Inductor mode and CUDA graphs
    python examples/bench_backends.py --env microduck --microduck-root ~/repos/microduck_rl \\
        --device cuda --backend torch mjx --mode default tuned reduce-overhead \\
        --batch_sizes 128 1024 8192 --out md_cuda.jsonl
    # Render tables
    python examples/bench_backends.py --table md.jsonl
"""

from __future__ import annotations

import argparse
import json
import os
import resource
import sys
import time
import xml.etree.ElementTree as ET
from collections import defaultdict
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np

SEED = 42
MODES = ["default", "tuned", "max-autotune", "reduce-overhead", "fixed-iter"]
LABELS = {
    ("mujoco_c", "default"): "MuJoCo C (CPU, sequential)",
    ("eager", "default"): "mujoco-torch vmap (eager)",
    ("torch", "default"): "mujoco-torch compile",
    ("torch", "tuned"): "mujoco-torch compile (tuned)",
    ("torch", "max-autotune"): "mujoco-torch compile (max-autotune)",
    ("torch", "reduce-overhead"): "mujoco-torch compile (reduce-overhead)",
    ("torch", "fixed-iter"): "mujoco-torch compile (fixed_iterations)",
    ("mjx", "default"): "MJX (JAX jit+vmap)",
}


# ---------------------------------------------------------------------------
# Models
# ---------------------------------------------------------------------------


def _microduck_scene(root: Path, timestep: float) -> str:
    """Return an XML path for MicroDuck's ``scene_walk.xml`` with box collision proxies.

    Mirrors the TorchRL MicroDuck example: the collision meshes (two ~10 000-edge
    soles) are replaced with ``fitaabb`` boxes because accelerated backends expand
    every convex-hull edge pair; visual meshes are untouched and the source files
    are never modified.  The patched scene is written next to a temp copy of the
    robot file with absolute asset directories.
    """
    candidates = [
        root,
        root / "scene_walk.xml",
        root / "robot/microduck/scene_walk.xml",
        root / "src/mjlab_microduck/robot/microduck/scene_walk.xml",
    ]
    scene = next((c for c in candidates if c.is_file()), None)
    if scene is None:
        raise FileNotFoundError(f"scene_walk.xml not found under {root}")
    scene_tree = ET.parse(scene)
    option = scene_tree.getroot().find("option")
    if option is None:
        option = ET.Element("option")
        scene_tree.getroot().insert(0, option)
    option.set("timestep", str(timestep))
    include = scene_tree.getroot().find("include")
    robot = (scene.parent / include.get("file")).resolve()
    robot_tree = ET.parse(robot)
    for geom in robot_tree.getroot().iter("geom"):
        if geom.get("class") in {"collision", "self_collision_only"} and geom.get("mesh") is not None:
            geom.set("type", "box")
    compiler = robot_tree.getroot().find("compiler")
    if compiler is None:
        compiler = ET.Element("compiler")
        robot_tree.getroot().insert(0, compiler)
    compiler.set("fitaabb", "true")
    for attr in ("meshdir", "texturedir"):
        d = compiler.get(attr)
        if d is not None and not Path(d).is_absolute():
            compiler.set(attr, str((robot.parent / d).resolve()))
    tmp = Path(TemporaryDirectory(prefix="mjt-microduck-").name)
    tmp.mkdir()
    robot_tree.write(tmp / robot.name, encoding="unicode")
    scene_tree.write(tmp / scene.name, encoding="unicode")
    return str(tmp / scene.name)


def load_model(args):
    import mujoco

    if args.env == "microduck":
        if args.microduck_root is None:
            raise SystemExit("--env microduck requires --microduck-root pointing at a microduck_rl checkout")
        scene = _microduck_scene(Path(args.microduck_root).expanduser(), args.microduck_timestep)
        return mujoco.MjModel.from_xml_path(scene)
    if args.env.endswith(".xml") and Path(args.env).is_file():
        return mujoco.MjModel.from_xml_path(args.env)
    from mujoco_torch._src import test_util

    return test_util.load_test_file(args.env if args.env.endswith(".xml") else f"{args.env}.xml")


# ---------------------------------------------------------------------------
# Timing helpers
# ---------------------------------------------------------------------------


def _peak_rss_mb() -> float:
    rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return rss / 2**20 if sys.platform == "darwin" else rss / 2**10  # bytes on macOS, KiB on Linux


def _choose_nsteps(step, warmup, budget):
    for _ in range(warmup):
        step()
    t0 = time.perf_counter()
    for _ in range(5):
        step()
    per = (time.perf_counter() - t0) / 5
    return int(min(1000, max(10, budget / max(per, 1e-9))))


def _timed(step, nsteps, sync):
    sync()
    t0 = time.perf_counter()
    for _ in range(nsteps):
        step()
    sync()
    return time.perf_counter() - t0


def emit(args, row):
    row.update(env=args.env, backend=row.get("backend"), mode=row.get("mode", "default"), device=args.device)
    row["peak_rss_mb"] = _peak_rss_mb()
    print(json.dumps(row), flush=True)
    if args.out:
        with open(args.out, "a") as f:
            f.write(json.dumps(row) + "\n")


# ---------------------------------------------------------------------------
# Backends
# ---------------------------------------------------------------------------


def run_torch(args, batch_size, mode, compiled):
    import torch
    import torch._dynamo
    import torch._dynamo.utils
    import torch._inductor.config as inductor_config

    import mujoco_torch

    device = torch.device(args.device)
    cuda = device.type == "cuda"
    sync = torch.cuda.synchronize if cuda else (lambda: None)
    if args.threads:
        torch.set_num_threads(args.threads)
    torch._dynamo.reset()
    torch._dynamo.utils.counters.clear()
    inductor_config.coordinate_descent_tuning = mode == "tuned"
    inductor_config.aggressive_fusion = mode == "tuned"
    compile_kwargs = {"fullgraph": True}
    if mode == "max-autotune":
        compile_kwargs["mode"] = "max-autotune-no-cudagraphs"
    elif mode == "reduce-overhead":
        compile_kwargs["mode"] = "reduce-overhead"
    step_kwargs = {"fixed_iterations": True} if mode == "fixed-iter" else {}

    m_mj = load_model(args)
    with torch.device("cpu"):
        mx = mujoco_torch.device_put(m_mj)
    mx = mx.to(device)
    # One eager step so the per-device index caches are warm before Dynamo traces.
    d_warm = mujoco_torch.device_put(__import__("mujoco").MjData(m_mj)).to(device)
    mujoco_torch.step(mx, d_warm)
    sync()

    rng = np.random.RandomState(SEED)
    with torch.device("cpu"):
        dx0 = mujoco_torch.make_data(mx)
    d = dx0.to(device).expand(batch_size).clone()
    d.qvel[:] = torch.as_tensor(0.01 * rng.randn(batch_size, m_mj.nv), dtype=d.qvel.dtype, device=device)

    vmap_step = torch.vmap(lambda d: mujoco_torch.step(mx, d, **step_kwargs))
    fn = torch.compile(vmap_step, **compile_kwargs) if compiled else vmap_step
    if cuda:
        torch.cuda.reset_peak_memory_stats()
    t0 = time.perf_counter()
    d = fn(d)
    sync()
    compile_s = time.perf_counter() - t0

    state = {"d": d}

    def step():
        state["d"] = fn(state["d"])

    nsteps = _choose_nsteps(lambda: (step(), sync()), args.warmup if compiled else 3, args.budget)
    elapsed = _timed(step, nsteps, sync)
    row = dict(
        backend="torch" if compiled else "eager",
        mode=mode if compiled else "default",
        batch_size=batch_size,
        compile_s=compile_s if compiled else None,
        nsteps=nsteps,
        run_s=elapsed,
        steps_per_s=batch_size * nsteps / elapsed,
        unique_graphs=torch._dynamo.utils.counters["stats"].get("unique_graphs") if compiled else None,
        torch=torch.__version__,
        torch_threads=torch.get_num_threads(),
    )
    if cuda:
        row["peak_cuda_mb"] = torch.cuda.max_memory_allocated() / 2**20
    emit(args, row)


def run_mjx(args, batch_size):
    import jax
    import jax.numpy as jnp
    import mujoco
    from mujoco import mjx

    jax.config.update("jax_enable_x64", True)
    m_mj = load_model(args)
    mx = mjx.put_model(m_mj)
    step_fn = jax.jit(jax.vmap(mjx.step, in_axes=(None, 0)))
    rng = np.random.RandomState(SEED)
    dx1 = mjx.put_data(m_mj, mujoco.MjData(m_mj))

    def tile(x):
        if not hasattr(x, "ndim"):
            return x
        return jnp.broadcast_to(x, (batch_size,)) if x.ndim == 0 else jnp.tile(x, (batch_size,) + (1,) * x.ndim)

    d = jax.tree.map(tile, dx1).replace(qvel=jnp.array(0.01 * rng.randn(batch_size, m_mj.nv)))
    t0 = time.perf_counter()
    d = step_fn(mx, d)
    jax.block_until_ready(d.qpos)
    compile_s = time.perf_counter() - t0
    state = {"d": d}

    def step():
        state["d"] = step_fn(mx, state["d"])

    def sync():
        jax.block_until_ready(state["d"].qpos)

    nsteps = _choose_nsteps(lambda: (step(), sync()), args.warmup, args.budget)
    elapsed = _timed(step, nsteps, sync)
    row = dict(
        backend="mjx",
        mode="default",
        batch_size=batch_size,
        compile_s=compile_s,
        nsteps=nsteps,
        run_s=elapsed,
        steps_per_s=batch_size * nsteps / elapsed,
        jax=jax.__version__,
        jax_device=str(jax.devices()[0]),
    )
    emit(args, row)


def run_mujoco_c(args, batch_size):
    import mujoco

    m_mj = load_model(args)
    rng = np.random.RandomState(SEED)
    datas = [mujoco.MjData(m_mj) for _ in range(batch_size)]
    for d in datas:
        d.qvel[:] = 0.01 * rng.randn(m_mj.nv)

    def step():
        for d in datas:
            mujoco.mj_step(m_mj, d)

    nsteps = _choose_nsteps(step, 5, args.budget)
    elapsed = _timed(step, nsteps, lambda: None)
    emit(
        args,
        dict(
            backend="mujoco_c",
            mode="default",
            batch_size=batch_size,
            nsteps=nsteps,
            run_s=elapsed,
            steps_per_s=batch_size * nsteps / elapsed,
        ),
    )


# ---------------------------------------------------------------------------
# Table rendering
# ---------------------------------------------------------------------------


def render_table(path: str) -> None:
    rows = [json.loads(line) for line in open(path) if line.strip()]
    order = list(LABELS)

    def label(r):
        return LABELS.get((r["backend"], r["mode"]), f"{r['backend']}/{r['mode']}")

    def key(lbl):
        for i, k in enumerate(order):
            if lbl == LABELS[k]:
                return (i, lbl)
        return (len(order), lbl)

    for env in sorted({r["env"] for r in rows}):
        er = [r for r in rows if r["env"] == env]
        bs = sorted({r["batch_size"] for r in er})
        hdr = "| Configuration | " + " | ".join(f"B={b:,}" for b in bs) + " |\n|---|" + "--:|" * len(bs)
        for title, field, fmt in (
            ("steps/s (higher is better)", "steps_per_s", "{:,.0f}"),
            ("peak process RSS, MB (includes compile)", "peak_rss_mb", "{:,.0f}"),
            ("peak CUDA memory allocated, MB", "peak_cuda_mb", "{:,.0f}"),
            ("first-call compile / JIT time, s", "compile_s", "{:.0f}"),
        ):
            table = defaultdict(dict)
            for r in er:
                if r.get(field) is not None:
                    table[label(r)][r["batch_size"]] = r[field]
            if not table:
                continue
            print(f"\n### {env} — {title} (device={er[0].get('device', '?')})\n\n{hdr}")
            for lbl in sorted(table, key=key):
                cells = [fmt.format(table[lbl][b]) if b in table[lbl] else "—" for b in bs]
                print(f"| {lbl} | " + " | ".join(cells) + " |")


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--table", metavar="JSONL", help="render markdown tables from a results file and exit")
    p.add_argument("--env", default="humanoid", help="test_data model name, an MJCF path, or 'microduck'")
    p.add_argument("--microduck-root", help="microduck_rl checkout (or its scene_walk.xml) for --env microduck")
    p.add_argument("--microduck-timestep", type=float, default=0.005)
    p.add_argument("--backend", nargs="+", default=["torch", "mjx"], choices=["torch", "eager", "mjx", "mujoco_c"])
    p.add_argument("--mode", nargs="+", default=["default"], choices=MODES, help="Inductor modes for the torch backend")
    p.add_argument("--batch_sizes", type=int, nargs="+", default=[16, 128, 1024])
    p.add_argument("--device", default="cpu")
    p.add_argument("--threads", type=int, default=0, help="torch intra-op threads (0 = default)")
    p.add_argument("--budget", type=float, default=10.0, help="seconds of timed steps per configuration")
    p.add_argument("--warmup", type=int, default=20)
    p.add_argument("--out", help="append JSONL rows to this file")
    args = p.parse_args()
    if args.table:
        render_table(args.table)
        return
    os.environ.setdefault("TORCHINDUCTOR_FX_GRAPH_CACHE", "1")
    for backend in args.backend:
        if backend == "mujoco_c":
            run_mujoco_c(args, 1)
            continue
        for b in args.batch_sizes:
            if backend == "mjx":
                run_mjx(args, b)
            elif backend == "eager":
                run_torch(args, b, "default", compiled=False)
            else:
                for mode in args.mode:
                    run_torch(args, b, mode, compiled=True)


if __name__ == "__main__":
    main()
