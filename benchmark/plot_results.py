#!/usr/bin/env python3
"""
Usage:
    python3 plot_results.py cpu.csv cuda.csv [-o out.png]
"""
import argparse
import io
import sys

import matplotlib.pyplot as plt
import pandas as pd

HEADER = "batch_size,time_steps,wall_mean_ms,wall_median_ms,wall_p95_ms,wall_max_ms,cpu_mean_ms"


def load_csv(path):
    """Keep only the header and rows starting with a numeric batch_size --
    tolerates stray lines (e.g. run_sweep.sh's "=== ... ===" progress
    markers, or ROS log lines) if the file was captured with 2>&1 instead of
    a clean stdout-only redirect."""
    with open(path) as f:
        lines = [line for line in f if line.strip() == HEADER or line[:1].isdigit()]
    if not lines or lines[0].strip() != HEADER:
        lines.insert(0, HEADER + "\n")
    return pd.read_csv(io.StringIO("".join(lines)))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("cpu_csv", help="CSV from mppi_controller_ros/benchmark/cpu_sweep_benchmark")
    parser.add_argument("cuda_csv", help="CSV from mppi_controller_cuda/benchmark (cuda_sweep_bench_* or run_sweep.sh)")
    parser.add_argument("-o", "--output", default="benchmark_results.png", help="output image path")
    args = parser.parse_args()

    cpu = load_csv(args.cpu_csv)
    cuda = load_csv(args.cuda_csv)

    time_steps_values = sorted(set(cpu["time_steps"]) & set(cuda["time_steps"]))
    if not time_steps_values:
        sys.exit("No time_steps values in common between the two CSVs -- nothing to plot.")

    fig, axes = plt.subplots(1, len(time_steps_values), figsize=(5 * len(time_steps_values), 4.5), sharey=True)
    if len(time_steps_values) == 1:
        axes = [axes]

    for ax, ts in zip(axes, time_steps_values):
        cpu_ts = cpu[cpu["time_steps"] == ts].sort_values("batch_size")
        cuda_ts = cuda[cuda["time_steps"] == ts].sort_values("batch_size")

        ax.plot(cpu_ts["batch_size"], cpu_ts["wall_mean_ms"], "o-", label="CPU", color="#2a78d6")
        ax.plot(cuda_ts["batch_size"], cuda_ts["wall_mean_ms"], "o-", label="CUDA", color="#eb6834")

        ax.set_xscale("log", base=2)
        ax.set_yscale("log")
        ax.set_xlabel("batch_size (rollouts)")
        ax.set_title(f"time_steps = {ts}")
        ax.grid(True, which="both", linestyle="--", alpha=0.3)

    axes[0].set_ylabel("wall_mean_ms (log scale)")
    axes[0].legend()
    fig.suptitle("MPPI compute time: CPU vs CUDA")
    fig.tight_layout()
    fig.savefig(args.output, dpi=150)
    print(f"Saved plot to {args.output}")

    merged = pd.merge(cpu, cuda, on=["batch_size", "time_steps"], suffixes=("_cpu", "_cuda"))
    if merged.empty:
        print("No (batch_size, time_steps) points in common -- skipping speedup table.")
        return
    merged["speedup"] = merged["wall_mean_ms_cpu"] / merged["wall_mean_ms_cuda"]
    cols = ["batch_size", "time_steps", "wall_mean_ms_cpu", "wall_mean_ms_cuda", "speedup"]
    print("\nSpeedup (CPU / CUDA wall_mean_ms):")
    print(merged[cols].sort_values(["batch_size", "time_steps"]).to_string(index=False, float_format=lambda x: f"{x:.3f}"))


if __name__ == "__main__":
    main()
