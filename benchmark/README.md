# Benchmark

`monitor_resource_usage.py` samples a process's CPU usage over time via `/proc/<pid>/stat`.

## Usage

Launch Nav2 with `use_composition:=False` so `controller_server` runs as its own process, set an initial pose, send a nav goal, then:

```sh
pgrep -f controller_server
python3 benchmark/monitor_resource_usage.py --pid <PID>
```

Flags: `--pid` (target PID), `--name` (regex fallback if no `--pid`), `--interval` (sampling period, default 1s), `--duration` (auto-stop, default: run until Ctrl+C). Ctrl+C prints a mean/median/p95/max/min summary.

Note: measures the whole `controller_server` process (costmap updates etc. included, not just the plugin).
