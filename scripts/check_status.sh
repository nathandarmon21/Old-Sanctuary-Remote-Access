#!/usr/bin/env bash
# check_status.sh — one-command morning status check for The Sanctuary
# Usage: ./scripts/check_status.sh
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$PROJECT_ROOT"

PID_FILE="runs/latest.pid"
SEP="━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"

echo "$SEP"
echo "  The Sanctuary — Status Check"
echo "  $(date)"
echo "$SEP"
echo ""

# ── 1. Process alive? ─────────────────────────────────────────────────────────
if [[ ! -f "$PID_FILE" ]]; then
    echo "  PROCESS:  No PID file found (no run launched, or already cleaned up)"
else
    PID=$(cat "$PID_FILE")
    if kill -0 "$PID" 2>/dev/null; then
        echo "  PROCESS:  RUNNING  (PID $PID)"
    else
        echo "  PROCESS:  FINISHED (PID $PID no longer alive)"
    fi
fi
echo ""

# ── Find the most recent run directory ────────────────────────────────────────
LATEST_RUN_DIR=""
if [[ -d "runs" ]]; then
    LATEST_RUN_DIR=$(ls -td runs/run_* 2>/dev/null | head -1 || true)
fi

if [[ -z "$LATEST_RUN_DIR" ]]; then
    echo "  No run directories found under runs/"
    echo ""
    echo "$SEP"
    exit 0
fi

RUN_ID=$(basename "$LATEST_RUN_DIR")
echo "  RUN ID:   $RUN_ID"
echo ""

# ── 2. Current simulated day ──────────────────────────────────────────────────
HEARTBEAT="$LATEST_RUN_DIR/heartbeat.txt"
if [[ -f "$HEARTBEAT" ]]; then
    echo "  HEARTBEAT:"
    sed 's/^/    /' "$HEARTBEAT"
else
    echo "  HEARTBEAT: not found"
fi
echo ""

# ── 3. Transaction count ──────────────────────────────────────────────────────
TX_FILE="$LATEST_RUN_DIR/transactions.jsonl"
if [[ -f "$TX_FILE" ]]; then
    TX_COUNT=$(wc -l < "$TX_FILE" | tr -d ' ')
    echo "  TRANSACTIONS: $TX_COUNT"
else
    echo "  TRANSACTIONS: 0 (file not yet created)"
fi
echo ""

# ── 4. Errors in events log ───────────────────────────────────────────────────
EVENTS_FILE="$LATEST_RUN_DIR/events.jsonl"
if [[ -f "$EVENTS_FILE" ]]; then
    PARSE_ERRORS=$(python3 -c "
import json, sys
count = 0
with open('$EVENTS_FILE') as f:
    for line in f:
        try:
            d = json.loads(line)
            if d.get('event_type') == 'parse_error':
                count += 1
        except:
            pass
print(count)
" 2>/dev/null || echo "?")
    BANKRUPT=$(python3 -c "
import json, sys
names = []
with open('$EVENTS_FILE') as f:
    for line in f:
        try:
            d = json.loads(line)
            if d.get('event_type') == 'bankruptcy':
                names.append(d.get('agent','?'))
        except:
            pass
print(', '.join(names) if names else 'none')
" 2>/dev/null || echo "?")
    echo "  PARSE ERRORS: $PARSE_ERRORS"
    echo "  BANKRUPTCIES: $BANKRUPT"
else
    echo "  EVENTS:  log not yet created"
fi
echo ""

# ── 5. PDF location ───────────────────────────────────────────────────────────
PDF="$LATEST_RUN_DIR/report.pdf"
if [[ -f "$PDF" ]]; then
    SIZE=$(du -h "$PDF" | cut -f1)
    echo "  PDF:  $PDF  ($SIZE)"
    echo "  Open: open \"$PDF\""
else
    echo "  PDF:  not yet generated"
fi
echo ""
echo "$SEP"
