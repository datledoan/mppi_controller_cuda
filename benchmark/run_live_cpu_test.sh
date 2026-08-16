#!/bin/bash
# Send a nav goal, then sample controller_server's live CPU usage while it
# navigates -- automates the "Live CPU usage" steps in this directory's
# README.md.
#
# Assumes Nav2 is already launched (use_composition:=False so
# controller_server runs as its own process) and localized (AMCL already has
# a valid pose covering the robot's current location).
#
# Usage:
#   bash run_live_cpu_test.sh [goal_x] [goal_y] [duration] [interval] [--gpu]
#   bash run_live_cpu_test.sh 1.76 0.99 10 0.2
set -e

GOAL_X="${1:-1.76}"
GOAL_Y="${2:-0.99}"
DURATION="${3:-10}"
INTERVAL="${4:-0.2}"

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

echo "Publishing goal (${GOAL_X}, ${GOAL_Y}) to /goal_pose..."
ros2 topic pub --once /goal_pose geometry_msgs/msg/PoseStamped \
  "{header: {frame_id: 'map'}, pose: {position: {x: ${GOAL_X}, y: ${GOAL_Y}, z: 0.0}, orientation: {w: 1.0}}}"

echo "Monitoring PID $PID for ${DURATION}s (interval ${INTERVAL}s)..."
python3 "$SCRIPT_DIR/monitor_resource_usage.py" --pid "$PID" --interval "$INTERVAL" --duration "$DURATION" $GPU_FLAG
