#!/usr/bin/env python3
"""Plot benchmark results from gpu_bench.py or pytest-benchmark JSON output.

Accepts one or more JSON files (one per model from gpu_bench.py, or a single
pytest-benchmark JSON with ``--benchmark-json``) and produces a single figure.

Usage:
    python scratch/plot_bench.py bench_humanoid.json
    python scratch/plot_bench.py bench_humanoid.json bench_ant.json -o assets/benchmark.png
    python scratch/plot_bench.py .benchmarks/0001_result.json -o assets/benchmark.png
"""

import argparse
import json
import os

import matplotlib.pyplot as plt
import numpy as np


COLORS = {
    "MuJoCo C (seq)": "#888888",
    "torch loop (seq)": "#bbbbbb",
    "torch vmap (eager)": "#4c9bd6",
    "torch compile": "#f0c060",
    "torch compile (H4)": "#e8792b",
    "torch compile (H4+H8)": "#d63b2f",
    "MJX jit(vmap)": "#59a84b",
}

MARKERS = {
    "MuJoCo C (seq)": "s",
    "torch loop (seq)": "D",
    "torch vmap (eager)": "o",
    "torch compile": "v",
    "torch compile (H4)": "^",
    "torch compile (H4+H8)": "^",
    "MJX jit(vmap)": "P",
}

LABELS = {
    "MuJoCo C (seq)": "MuJoCo C (CPU, sequential)",
    "torch loop (seq)": "mujoco-torch (GPU, sequential)",
    "torch vmap (eager)": "mujoco-torch vmap (eager)",
    "torch compile": "mujoco-torch compile (baseline)",
    "torch compile (H4)": "mujoco-torch compile (optimized)",
    "torch compile (H4+H8)": "mujoco-torch compile (opt + fixed iter)",
    "MJX jit(vmap)": "MJX (JAX jit+vmap)",
}


def extract_series(config_data):
    """Extract (batch_sizes, steps_per_s) arrays from a config's results dict."""
    batch_sizes = []
    sps_values = []
    for key, val in sorted(config_data.items(), key=lambda kv: int(kv[0].split("=")[1])):
        if val is None:
            continue
        B = int(key.split("=")[1])
        batch_sizes.append(B)
        sps_values.append(val["steps_per_s"])
    return np.array(batch_sizes), np.array(sps_values)


def plot_model(ax, model_name, model_data, all_batch_sizes):
    """Plot one model's results on an axes."""
    for config_name, config_data in model_data.items():
        color = COLORS.get(config_name, "#333333")
        marker = MARKERS.get(config_name, "o")
        label = LABELS.get(config_name, config_name)

        bs, sps = extract_series(config_data)
        if len(sps) == 0:
            continue

        if len(bs) == 1 and bs[0] == 1:
            ax.axhline(y=sps[0], color=color, linestyle="--", alpha=0.6, linewidth=1.2)
            ax.plot(bs, sps, marker=marker, color=color, markersize=7,
                    label=label, linestyle="none", zorder=5)
        else:
            ax.plot(bs, sps, marker=marker, color=color, markersize=7,
                    linewidth=2, label=label, zorder=5)

    ax.set_xscale("log", base=2)
    ax.set_yscale("log")
    ax.set_xlabel("Batch size", fontsize=12)
    ax.set_ylabel("Steps / second", fontsize=12)
    ax.set_title(model_name, fontsize=14, fontweight="bold")
    ax.set_xticks(all_batch_sizes)
    ax.set_xticklabels([str(b) for b in all_batch_sizes])
    ax.grid(True, which="both", alpha=0.3, linewidth=0.5)
    ax.legend(fontsize=8, loc="lower right", framealpha=0.9)


def _load_pytest_benchmark(raw):
    """Convert pytest-benchmark JSON (has ``benchmarks`` key) to gpu_bench format."""
    data = {}
    for entry in raw["benchmarks"]:
        info = entry.get("extra_info", {})
        model = info.get("model")
        backend = info.get("backend")
        batch_size = info.get("batch_size")
        sps = info.get("steps_per_s")
        if model is None or backend is None or batch_size is None:
            continue
        data.setdefault(model, {}).setdefault(backend, {})[f"B={batch_size}"] = {
            "steps_per_s": sps,
            "elapsed_s": entry["stats"]["mean"],
        }
    return data


def _load_json(path):
    """Load a JSON file, auto-detecting gpu_bench vs pytest-benchmark format."""
    with open(path) as f:
        raw = json.load(f)
    if "benchmarks" in raw:
        return _load_pytest_benchmark(raw)
    return raw


def main():
    parser = argparse.ArgumentParser(description="Plot gpu_bench.py / pytest-benchmark results")
    parser.add_argument("json_files", nargs="+", help="Path(s) to bench JSON files")
    parser.add_argument("--output", "-o", default="assets/benchmark.png",
                        help="Output image path (default: assets/benchmark.png)")
    parser.add_argument("--dpi", type=int, default=150)
    args = parser.parse_args()

    data = {}
    for path in args.json_files:
        loaded = _load_json(path)
        for model, backends in loaded.items():
            for backend, batches in backends.items():
                data.setdefault(model, {}).setdefault(backend, {}).update(batches)

    models = list(data.keys())
    ncols = len(models)

    fig, axes = plt.subplots(1, ncols, figsize=(7 * ncols, 5), squeeze=False)
    fig.suptitle("mujoco-torch GPU Benchmark (H200, float64, 1000 steps)",
                 fontsize=15, fontweight="bold", y=1.02)

    for i, model_name in enumerate(models):
        model_data = data[model_name]
        all_bs = set()
        for config_data in model_data.values():
            for key, val in config_data.items():
                if val is not None:
                    all_bs.add(int(key.split("=")[1]))
        all_batch_sizes = sorted(all_bs)
        plot_model(axes[0, i], model_name, model_data, all_batch_sizes)

    plt.tight_layout()

    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    fig.savefig(args.output, dpi=args.dpi, bbox_inches="tight",
                facecolor="white", edgecolor="none")
    print(f"Saved to {args.output}")
    plt.close(fig)


if __name__ == "__main__":
    main()
