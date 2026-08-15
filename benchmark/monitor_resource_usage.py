#!/usr/bin/env python3

import argparse
import glob
import os
import re
import statistics
import subprocess
import sys
import time

CLK_TCK = os.sysconf("SC_CLK_TCK")


def find_pid(name_regex):
    pattern = re.compile(name_regex)
    own_pid = os.getpid()
    for stat_path in glob.glob("/proc/[0-9]*/stat"):
        pid = stat_path.split("/")[2]
        if int(pid) == own_pid:
            continue
        try:
            with open(f"/proc/{pid}/comm") as f:
                comm = f.read().strip()
            with open(f"/proc/{pid}/cmdline", "rb") as f:
                cmdline = f.read().replace(b"\0", b" ").decode(errors="replace")
        except (FileNotFoundError, ProcessLookupError):
            continue
        if pattern.search(comm) or pattern.search(cmdline):
            return int(pid)
    return None


def read_process_cpu_ticks(pid):
    with open(f"/proc/{pid}/stat") as f:
        raw = f.read()
    rest = raw.rsplit(")", 1)[1].split()
    utime, stime = int(rest[11]), int(rest[12])
    return utime + stime


def read_gpu_util():
    try:
        out = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=utilization.gpu", "--format=csv,noheader,nounits"],
            stderr=subprocess.DEVNULL,
        )
        return float(out.decode().splitlines()[0])
    except Exception:
        return None


def summarize(label, samples):
    if not samples:
        print(f"{label}: no samples")
        return
    samples_sorted = sorted(samples)
    p95 = samples_sorted[int(0.95 * (len(samples_sorted) - 1))]
    print(
        f"{label}: mean={statistics.mean(samples):.1f} median={statistics.median(samples):.1f} "
        f"p95={p95:.1f} max={max(samples):.1f} min={min(samples):.1f} (n={len(samples)})"
    )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pid", type=int, default=None, help="PID to monitor (skips --name lookup)")
    ap.add_argument("--name", default="mbf_costmap_nav", help="regex matched against process comm/cmdline")
    ap.add_argument("--interval", type=float, default=1.0, help="sampling interval, seconds (default 1.0)")
    ap.add_argument("--duration", type=float, default=None, help="stop after this many seconds (default: run until Ctrl+C)")
    ap.add_argument("--gpu", action="store_true", help="also sample overall GPU utilization via nvidia-smi (whole-GPU, not per-process)")
    args = ap.parse_args()

    pid = args.pid if args.pid is not None else find_pid(args.name)
    if pid is None:
        print(f"No running process matched --name '{args.name}' and no --pid given.", file=sys.stderr)
        sys.exit(1)
    print(f"Monitoring PID {pid} every {args.interval}s (Ctrl+C to stop)...")

    cpu_samples = []
    gpu_samples = []
    prev_ticks = read_process_cpu_ticks(pid)
    prev_wall = time.monotonic()
    start = prev_wall

    try:
        while True:
            time.sleep(args.interval)
            if not os.path.exists(f"/proc/{pid}"):
                print(f"PID {pid} exited.")
                break

            now = time.monotonic()
            ticks = read_process_cpu_ticks(pid)
            cpu_pct = (ticks - prev_ticks) / CLK_TCK / (now - prev_wall) * 100.0
            prev_ticks, prev_wall = ticks, now
            cpu_samples.append(cpu_pct)

            line = f"[{now - start:6.1f}s] cpu={cpu_pct:6.1f}% (of 1 core)"
            if args.gpu:
                gpu_pct = read_gpu_util()
                if gpu_pct is not None:
                    gpu_samples.append(gpu_pct)
                    line += f"  gpu={gpu_pct:5.1f}%"
            print(line)

            if args.duration is not None and (now - start) >= args.duration:
                break
    except KeyboardInterrupt:
        print()

    print("\n--- summary ---")
    summarize("cpu_pct (of 1 core)", cpu_samples)
    if args.gpu:
        summarize("gpu_pct (whole GPU)", gpu_samples)


if __name__ == "__main__":
    main()
