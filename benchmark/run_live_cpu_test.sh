#!/bin/bash
# Usage:
#   bash run_live_cpu_test.sh [goal_x] [goal_y] [duration] [interval]
#   bash run_live_cpu_test.sh 1.8 1.5 8 0.2
set -e

GOAL_X="${1:-1.8}"
GOAL_Y="${2:-1.5}"
DURATION="${3:-10}"
INTERVAL="${4:-0.2}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

PID=$(pgrep -x mbf_costmap_nav | head -n1)
if [ -z "$PID" ]; then
  echo "No mbf_costmap_nav process found -- launch move_base_flex first" \
    "(e.g. roslaunch turtlebot3_navigation turtlebot3_navigation.launch)." >&2
  exit 1
fi
echo "move_base_flex PID: $PID"

echo "Publishing goal (${GOAL_X}, ${GOAL_Y}) to /move_base_simple/goal..."
rostopic pub -1 /move_base_simple/goal geometry_msgs/PoseStamped \
  "{header: {frame_id: 'map'}, pose: {position: {x: ${GOAL_X}, y: ${GOAL_Y}, z: 0.0}, orientation: {w: 1.0}}}" \
  > /dev/null

echo "Monitoring PID $PID for ${DURATION}s (interval ${INTERVAL}s)..."
python3 "$SCRIPT_DIR/monitor_resource_usage.py" --pid "$PID" --interval "$INTERVAL" --duration "$DURATION"
