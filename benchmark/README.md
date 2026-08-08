# mppi_controller_cuda benchmark

Measures per-call time of `Controller::computeControl()` (no ROS involved),
swept across `batch_size` (rollout count) and `time_steps`. Comparable
against `mppi_controller_ros/benchmark`'s CPU version (same scenario).

`batch_size` is a compile-time template parameter, so each value gets its
own binary (`cuda_sweep_bench_<N>`), built from the
`MPPI_BENCH_ROLLOUT_COUNTS` CMake list (default: 256, 512, 1024, 2048, 4096,
8192). `time_steps` (32, 56, 100) is swept at runtime inside each binary.

## Build

Off by default (a plain `catkin build mppi_controller_cuda` only builds the
plugin library) -- opt in explicitly:

```bash
catkin build -j2 mppi_controller_cuda --cmake-args -DMPPI_BUILD_BENCHMARKS=ON
# or, to change which rollout counts get built:
catkin build -j2 mppi_controller_cuda --cmake-args -DMPPI_BUILD_BENCHMARKS=ON -DMPPI_BENCH_ROLLOUT_COUNTS="256;1024;4096"
```
## Run

```bash
source devel/setup.bash

rosrun mppi_controller_cuda cuda_sweep_bench_2048 [warmup] [reps]   # defaults 5 30
bash src/mppi_controller_cuda/benchmark/run_sweep.sh [warmup] [reps]  # all built batch_sizes, merged into one CSV
```

`warmup`: calls made but not timed, to let things settle (GPU kernel JIT,
caches, MPPI's own warm-started control mean) before measuring. `reps`:
calls actually timed and used for the CSV's mean/median/p95/max.

## Output

CSV: `batch_size,time_steps,wall_mean_ms,wall_median_ms,wall_p95_ms,wall_max_ms,cpu_mean_ms`

- `wall_*_ms`: wall-clock time per `computeControl()` call.
- `cpu_mean_ms`: process CPU-time (`getrusage`) for that call — close to
  `wall_mean_ms` because `cudaStreamSynchronize` spin-waits rather than
  sleeping, not because the CPU is doing the GPU's math.
