#!/usr/bin/env python3
"""
Usage:
    python3 plot_results.py cpu.csv cuda.csv [-o out.png]
"""
import argparse
import sys

import matplotlib.pyplot as plt

HEADER = "batch_size,time_steps,wall_mean_ms,wall_median_ms,wall_p95_ms,wall_max_ms,cpu_mean_ms"
FIELDS = HEADER.split(",")


def load_csv(path):
    rows = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line or not line[:1].isdigit():
                continue
            values = line.split(",")
            row = dict(zip(FIELDS, values))
            row["batch_size"] = int(row["batch_size"])
            row["time_steps"] = int(row["time_steps"])
            for k in FIELDS[2:]:
                row[k] = float(row[k])
            rows.append(row)
    return rows


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("cpu_csv", help="CSV from cpu_sweep_bench (nav2_mppi_controller)")
    parser.add_argument("cuda_csv", help="CSV from run_sweep.sh (mppi_controller_cuda's cuda_sweep_bench_*)")
    parser.add_argument("-o", "--output", default="benchmark_results.png", help="output image path")
    args = parser.parse_args()

    cpu = load_csv(args.cpu_csv)
    cuda = load_csv(args.cuda_csv)

    time_steps_values = sorted(set(r["time_steps"] for r in cpu) & set(r["time_steps"] for r in cuda))
    if not time_steps_values:
        sys.exit("No time_steps values in common between the two CSVs -- nothing to plot.")

    fig, axes = plt.subplots(1, len(time_steps_values), figsize=(5 * len(time_steps_values), 4.5), sharey=True)
    if len(time_steps_values) == 1:
        axes = [axes]

    for ax, ts in zip(axes, time_steps_values):
        cpu_ts = sorted((r for r in cpu if r["time_steps"] == ts), key=lambda r: r["batch_size"])
        cuda_ts = sorted((r for r in cuda if r["time_steps"] == ts), key=lambda r: r["batch_size"])

        ax.plot([r["batch_size"] for r in cpu_ts], [r["wall_mean_ms"] for r in cpu_ts],
                "o-", label="CPU", color="#2a78d6")
        ax.plot([r["batch_size"] for r in cuda_ts], [r["wall_mean_ms"] for r in cuda_ts],
                "o-", label="CUDA", color="#eb6834")

        ax.set_xscale("log", base=2)
        ax.set_yscale("log")
        ax.set_xlabel("batch_size (rollouts)")
        ax.set_title(f"time_steps = {ts}")
        ax.grid(True, which="both", linestyle="--", alpha=0.3)

    axes[0].set_ylabel("wall_mean_ms (log scale)")
    axes[0].legend()
    fig.suptitle("MPPI compute time: CPU (nav2_mppi_controller) vs CUDA (mppi_controller_cuda)")
    fig.tight_layout()
    fig.savefig(args.output, dpi=150)
    print(f"Saved plot to {args.output}")

    cuda_by_key = {(r["batch_size"], r["time_steps"]): r for r in cuda}
    merged = []
    for r in cpu:
        key = (r["batch_size"], r["time_steps"])
        if key in cuda_by_key:
            c = cuda_by_key[key]
            merged.append((
                r["batch_size"], r["time_steps"], r["wall_mean_ms"], c["wall_mean_ms"],
                r["wall_mean_ms"] / c["wall_mean_ms"]))
    if not merged:
        print("No (batch_size, time_steps) points in common -- skipping speedup table.")
        return
    merged.sort(key=lambda t: (t[0], t[1]))
    print("\nSpeedup (CPU / CUDA wall_mean_ms):")
    print(f"{'batch_size':>10} {'time_steps':>10} {'wall_mean_ms_cpu':>18} {'wall_mean_ms_cuda':>19} {'speedup':>9}")
    for batch_size, time_steps, cpu_ms, cuda_ms, speedup in merged:
        print(f"{batch_size:>10} {time_steps:>10} {cpu_ms:>18.3f} {cuda_ms:>19.3f} {speedup:>9.3f}")


if __name__ == "__main__":
    main()
