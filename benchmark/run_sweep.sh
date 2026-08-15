#!/bin/bash
set -e

WARMUP="${1:-5}"
REPS="${2:-30}"

BUILD_DIR="$(pwd)/build/mppi_controller_cuda"
FOUND=$(find "$BUILD_DIR" -maxdepth 1 -name 'cuda_sweep_bench_*' -type f 2>/dev/null | \
  awk -F_ '{print $NF, $0}' | sort -n | cut -d' ' -f2-)

if [ -z "$FOUND" ]; then
  echo "No cuda_sweep_bench_* binaries found under $BUILD_DIR -- run from the" >&2
  echo "workspace root, after building first:" >&2
  echo "  colcon build --packages-select mppi_controller_cuda --cmake-args -DMPPI_BUILD_BENCHMARKS=ON" >&2
  exit 1
fi

echo "batch_size,time_steps,wall_mean_ms,wall_median_ms,wall_p95_ms,wall_max_ms,cpu_mean_ms"
for bin in $FOUND; do
  echo "=== $(basename "$bin") ===" >&2
  "$bin" "$WARMUP" "$REPS" | tail -n +2
done
