#!/bin/bash
# Usage:
#   bash run_live_cpu_test.sh [goal_x] [goal_y] [duration] [interval] [warmup] [--gpu]
#   bash run_live_cpu_test.sh 1.76 0.99 10 0.2
set -e

GOAL_X="${1:-1.76}"
GOAL_Y="${2:-0.5}"
DURATION="${3:-13}"
INTERVAL="${4:-0.2}"

WARMUP="${5:-3}"

GPU_FLAG=""
for arg in "$@"; do
  if [ "$arg" = "--gpu" ]; then GPU_FLAG="--gpu"; fi
done

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

PID=$(pgrep -f controller_server | head -n1)
if [ -z "$PID" ]; then
  echo "No controller_server process found -- launch Nav2 first" \
    "(use_composition:=False so it runs as its own process)." >&2
  exit 1
fi
echo "controller_server PID: $PID"

echo "Sending goal (${GOAL_X}, ${GOAL_Y}) to /navigate_to_pose..."
ros2 action send_goal /navigate_to_pose nav2_msgs/action/NavigateToPose \
  "{pose: {header: {frame_id: 'map'}, pose: {position: {x: ${GOAL_X}, y: ${GOAL_Y}, z: 0.0}, orientation: {w: 1.0}}}}" \
  > /dev/null 2>&1 &

echo "Monitoring PID $PID for ${DURATION}s (interval ${INTERVAL}s, ${WARMUP}s warmup excluded)..."
python3 "$SCRIPT_DIR/monitor_resource_usage.py" --pid "$PID" --interval "$INTERVAL" --duration "$DURATION" --warmup "$WARMUP" $GPU_FLAG
