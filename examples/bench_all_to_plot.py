#!/usr/bin/env python3
"""Convert bench_all.py JSONL rows into gpu_bench / plot_bench.py format.

Reads one or more JSONL files produced by ``examples/bench_all.py`` (one row
per env × batch_size, with fields ``env``, ``backend``, ``mode``,
``batch_size``, ``steps_per_s``, ``status``) and writes a single JSON file
keyed by ``{model_name: {backend_label: {"B=N": {"steps_per_s": X,
"elapsed_s": Y}}}}`` that ``benchmarks/plot_bench.py`` consumes.

Backend labels map to ``plot_bench.py`` COLORS/MARKERS/LABELS keys:

- torch + compile   -> "torch compile"
- mjx + compile     -> "MJX jit(vmap)"
- torch + collector -> "TorchRL collector+RB"

Usage::

    python examples/bench_all_to_plot.py \\
        ~/bench_all_305299/compile_torch.jsonl \\
        ~/bench_all_305299/compile_mjx.jsonl \\
        ~/bench_all_305299/collector_torch.jsonl \\
        -o ~/bench_all_305299/results.json
"""

import argparse
import json
from pathlib import Path


BACKEND_LABEL = {
    ("torch", "compile"): "torch compile",
    ("mjx", "compile"): "MJX jit(vmap)",
    ("torch", "collector"): "TorchRL collector+RB",
}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("jsonl_files", nargs="+", type=Path)
    p.add_argument("-o", "--output", type=Path, required=True)
    args = p.parse_args()

    data: dict = {}
    for path in args.jsonl_files:
        if not path.exists():
            print(f"warn: {path} does not exist -- skipping")
            continue
        with path.open() as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                row = json.loads(line)
                if row.get("status") != "ok":
                    continue
                env = row["env"]
                backend = row["backend"]
                mode = row["mode"]
                label = BACKEND_LABEL.get((backend, mode))
                if label is None:
                    continue
                B = row["batch_size"]
                sps = row.get("steps_per_s")
                if sps is None:
                    continue
                elapsed = row.get("run_s") or row.get("measure_s") or 0.0
                data.setdefault(env, {}).setdefault(label, {})[f"B={B}"] = {
                    "steps_per_s": sps,
                    "elapsed_s": elapsed,
                }

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w") as f:
        json.dump(data, f, indent=2)
    print(f"wrote {args.output}")


if __name__ == "__main__":
    main()
