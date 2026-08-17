#!/bin/bash
set -e

WARMUP="${1:-5}"
REPS="${2:-30}"

WS_DEVEL="$(catkin locate -d 2>/dev/null)"
if [ -z "$WS_DEVEL" ]; then
  echo "catkin locate -d failed -- run this from inside your catkin workspace" >&2
  echo "(needs catkin_tools: 'sudo apt install python3-catkin-tools')." >&2
  exit 1
fi

FOUND=$(find "$WS_DEVEL" -maxdepth 6 -name 'cuda_sweep_bench_*' -type f 2>/dev/null | \
  awk -F_ '{print $NF, $0}' | sort -n | cut -d' ' -f2-)

if [ -z "$FOUND" ]; then
  echo "No cuda_sweep_bench_* binaries found under $WS_DEVEL -- build first:" >&2
  echo "  catkin build mppi_controller_cuda --cmake-args -DMPPI_BUILD_BENCHMARKS=ON" >&2
  exit 1
fi

echo "batch_size,time_steps,wall_mean_ms,wall_median_ms,wall_p95_ms,wall_max_ms,cpu_mean_ms"
for bin in $FOUND; do
  echo "=== $(basename "$bin") ===" >&2
  "$bin" "$WARMUP" "$REPS" | tail -n +2
done
