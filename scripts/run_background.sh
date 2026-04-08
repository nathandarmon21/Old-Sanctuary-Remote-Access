#!/usr/bin/env bash
# run_background.sh — launch a Sanctuary simulation in the background
# so it survives terminal closure and laptop lid close.
#
# Usage:
#   ./scripts/run_background.sh [--config CONFIG] [--seed SEED] [--profile]
#
# Defaults:
#   --config configs/dev_local.yaml
#   --seed   42

set -euo pipefail

CONFIG="configs/dev_haiku.yaml"
SEED="42"
EXTRA_ARGS=""

# Parse arguments
while [[ $# -gt 0 ]]; do
    case "$1" in
        --config) CONFIG="$2"; shift 2 ;;
        --seed)   SEED="$2";   shift 2 ;;
        --profile) EXTRA_ARGS="--profile"; shift ;;
        *) echo "Unknown argument: $1"; exit 1 ;;
    esac
done

# Ensure we're running from the project root
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$PROJECT_ROOT"

# Create runs/ directory if needed
mkdir -p runs

# Timestamp for log file (before the simulation creates the run ID)
LAUNCH_TS=$(date +%Y%m%d_%H%M%S)
LOG_FILE="runs/launch_${LAUNCH_TS}_seed${SEED}.log"

echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "  The Sanctuary — Background Launch"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "  Config:  $CONFIG"
echo "  Seed:    $SEED"
echo "  Log:     $LOG_FILE"
echo ""

# Launch with caffeinate (prevents macOS system sleep) + nohup (detaches from terminal)
caffeinate -i nohup python3 scripts/run_simulation.py \
    --config "$CONFIG" \
    --seed "$SEED" \
    $EXTRA_ARGS \
    > "$LOG_FILE" 2>&1 &

PID=$!
echo "$PID" > runs/latest.pid
echo "  PID:     $PID  (also saved to runs/latest.pid)"
echo ""
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "  HOW TO MONITOR"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo ""
echo "  Watch the log (live):"
echo "    tail -f $LOG_FILE"
echo ""
echo "  Check heartbeat (simulated day + elapsed time):"
echo "    cat runs/latest_run_heartbeat.txt   # (path shown after run starts)"
echo "    # OR check the run directory directly once the run ID appears in the log:"
echo "    # cat runs/<run_id>/heartbeat.txt"
echo ""
echo "  Check if process is still running:"
echo "    ps -p $PID"
echo "    # or: kill -0 $PID && echo 'running' || echo 'finished'"
echo ""
echo "  Kill the run:"
echo "    kill $PID"
echo "    # or: kill \$(cat runs/latest.pid)"
echo ""
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "  Process detached. You can close this terminal."
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
