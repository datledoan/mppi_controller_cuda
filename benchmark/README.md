# Benchmark

## Live CPU usage

`monitor_resource_usage.py` samples a process's CPU usage over time via `/proc/<pid>/stat`.

Launch Nav2 with `use_composition:=False` so `controller_server` runs as its own process, set an initial pose, send a nav goal, then:

```sh
pgrep -f controller_server
python3 benchmark/monitor_resource_usage.py --pid <PID>
```

Flags: `--pid` (target PID), `--name` (regex fallback if no `--pid`), `--interval` (sampling period, default 1s), `--duration` (auto-stop, default: run until Ctrl+C). Ctrl+C prints a mean/median/p95/max/min summary.

Note: measures the whole `controller_server` process (costmap updates etc. included, not just the plugin).

`run_live_cpu_test.sh` automates the above: publishes a goal to `/goal_pose`, finds the `controller_server` PID, and samples it for a fixed window. Nav2 must already be launched (`use_composition:=False`) and localized.

```sh
bash benchmark/run_live_cpu_test.sh [goal_x] [goal_y] [duration] [interval]
bash benchmark/run_live_cpu_test.sh 1.76 0.99 10 0.2
```

## Sweep benchmark

Per-call time of the rollout optimizer, swept across `batch_size`/`time_steps`. `sweep_bench.cu` (GPU, this plugin) and `cpu_sweep_bench.cpp` (CPU, `nav2_mppi_controller::Optimizer::evalControl()`).

Off by default -- opt in:

```sh
colcon build --packages-select mppi_controller_cuda --cmake-args -DMPPI_BUILD_BENCHMARKS=ON
```

`batch_size` is a compile-time template parameter on the GPU side, so each value gets its own binary (`cuda_sweep_bench_<N>`, `N` from the `MPPI_BENCH_ROLLOUT_COUNTS` CMake list, default 256/512/1024/2048/4096/8192). `run_sweep.sh` runs every built one and merges them into a single CSV. The CPU binary sweeps its whole `batch_size`/`time_steps` grid itself in one run.

Run from the workspace root (`[warmup] [reps]` optional, default `5 30`):

```sh
source install/setup.bash
bash src/mppi_controller_cuda/benchmark/run_sweep.sh 5 30 > cuda.csv
./build/mppi_controller_cuda/cpu_sweep_bench 5 30 2>/dev/null > cpu.csv
python3 src/mppi_controller_cuda/benchmark/plot_results.py cpu.csv cuda.csv -o results.png
```