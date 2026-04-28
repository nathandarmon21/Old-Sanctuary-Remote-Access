#!/bin/bash
# d11_monitor.sh — overnight watchdog for the D11 cluster pilot jobs.
#
# Designed to be invoked from cron (~/sanctuary/sweeps/d11_monitor.sh) every
# 30 minutes while the pilot is in flight. Watches three SLURM jobs:
#   9022960  d11_overnight   30-day Qwen 32B (12h walltime, expected to finish)
#   9022964  d11_resume_1    5-day chain link 1 (expected to TIMEOUT at 2h)
#   9022967  d11_resume_2    5-day chain link 2 (depends on link 1; should finish)
#
# What it does each tick:
#   - Records per-job state (PENDING / RUNNING / COMPLETED / TIMEOUT / FAILED).
#   - For RUNNING jobs, checks events.jsonl growth — if the job has been
#     running >30 min with <5 new events in the last tick, flags as a
#     possible hang.
#   - Writes a status file at $STATE_DIR/status.txt — the user reads this
#     in the morning.
#   - Detects all-done condition (no jobs in queue any more) and writes
#     done.flag so subsequent ticks become no-ops.
#
# Hands-off: does NOT auto-cancel or auto-resubmit. SLURM's --mail-type=FAIL
# already emails the user on job failure; this script's value is hang
# detection plus a unified "what happened overnight" summary.

set -uo pipefail

STATE_DIR=/n/netscratch/zittrain_lab/Everyone/ndarmon/d11_monitor
mkdir -p "$STATE_DIR"

# Bail early if a previous tick declared everything done.
if [ -f "$STATE_DIR/done.flag" ]; then
    exit 0
fi

JOBS=(9022960 9022964 9022967 9067552)
NAMES=("d11_overnight" "d11_resume_1" "d11_resume_2" "d11_overnight_cont")
DIRS=(
  "/n/netscratch/zittrain_lab/Everyone/ndarmon/d11_overnight_30day"
  "/n/netscratch/zittrain_lab/Everyone/ndarmon/d11_resume_test_5day"
  "/n/netscratch/zittrain_lab/Everyone/ndarmon/d11_resume_test_5day"
  "/n/netscratch/zittrain_lab/Everyone/ndarmon/d11_overnight_30day"
)
# d11_overnight hitting walltime is now expected (continuation job will resume).
# resume_1 hitting walltime is expected. The other two should COMPLETE.
TIMEOUT_OK=("1" "1" "0" "0")

NOW_ISO=$(date -u +%Y-%m-%dT%H:%M:%SZ)
LOG="$STATE_DIR/monitor.log"
STATUS="$STATE_DIR/status.txt"

echo "=== tick $NOW_ISO ===" >> "$LOG"

# Build status file fresh each tick so the user always reads the latest.
{
    echo "D11 cluster pilot monitor — last updated $NOW_ISO UTC"
    echo
} > "$STATUS"

ANY_LIVE=0
ALERTS=()

for i in "${!JOBS[@]}"; do
    JID=${JOBS[$i]}
    NAME=${NAMES[$i]}
    DIR=${DIRS[$i]}
    TOK=${TIMEOUT_OK[$i]}

    SQ_STATE=$(squeue -h -j "$JID" -o "%T" 2>/dev/null | head -1)
    if [ -z "$SQ_STATE" ]; then
        # Not in queue anymore — get final state via sacct.
        FINAL=$(sacct -j "$JID" -n -P -o State 2>/dev/null | head -1 | tr -d '+ ')
        [ -z "$FINAL" ] && FINAL="UNKNOWN"
        echo "$FINAL" > "$STATE_DIR/${NAME}.state"
        echo "[$NAME] job $JID terminal: $FINAL" >> "$LOG"
        echo "  $NAME (job $JID): $FINAL" >> "$STATUS"

        # Alert on bad final states (TIMEOUT is OK only for resume_1).
        case "$FINAL" in
            COMPLETED|CANCELLED|"") ;;  # CANCELLED could be ours, treat as terminal
            TIMEOUT)
                if [ "$TOK" != "1" ]; then
                    ALERTS+=("$NAME (job $JID) hit TIMEOUT — likely walltime exceeded")
                fi
                ;;
            *)
                ALERTS+=("$NAME (job $JID) ended in state $FINAL — investigate slurm log")
                ;;
        esac
        continue
    fi

    # Still in queue (PENDING / RUNNING / etc).
    ANY_LIVE=1
    echo "$SQ_STATE" > "$STATE_DIR/${NAME}.state"
    echo "[$NAME] job $JID: $SQ_STATE" >> "$LOG"

    # Forward-progress check on RUNNING jobs.
    EV="$DIR/events.jsonl"
    EV_NOW=$(wc -l < "$EV" 2>/dev/null || echo 0)
    EV_PREV=$(cat "$STATE_DIR/${NAME}.events" 2>/dev/null || echo 0)
    echo "$EV_NOW" > "$STATE_DIR/${NAME}.events"

    if [ "$SQ_STATE" = "RUNNING" ]; then
        START_TIME=$(squeue -h -j "$JID" -o "%S" 2>/dev/null | head -1)
        # SLURM emits %S in the cluster's local timezone. Drop -u so date
        # interprets it in the same TZ as `date +%s` for a correct delta.
        START_SEC=$(date -d "$START_TIME" +%s 2>/dev/null || date +%s)
        NOW_SEC=$(date +%s)
        ELAPSED=$((NOW_SEC - START_SEC))
        echo "  $NAME (job $JID): RUNNING ${ELAPSED}s, events.jsonl=$EV_NOW (was $EV_PREV)" >> "$STATUS"
        if [ "$ELAPSED" -gt 1800 ] && [ "$((EV_NOW - EV_PREV))" -lt 5 ]; then
            ALERTS+=("$NAME (job $JID) running ${ELAPSED}s, events grew only $((EV_NOW - EV_PREV)) since last tick — possible hang")
        fi
    else
        echo "  $NAME (job $JID): $SQ_STATE" >> "$STATUS"
    fi
done

if [ ${#ALERTS[@]} -gt 0 ]; then
    {
        echo
        echo "ALERTS:"
        for a in "${ALERTS[@]}"; do echo "  - $a"; done
    } >> "$STATUS"
    {
        echo "ALERTS at $NOW_ISO:"
        for a in "${ALERTS[@]}"; do echo "  - $a"; done
    } >> "$LOG"

    # Best-effort email notification (FASRC login nodes usually have `mail`).
    if command -v mail >/dev/null 2>&1; then
        {
            echo "D11 monitor alerts at $NOW_ISO:"
            echo
            for a in "${ALERTS[@]}"; do echo "- $a"; done
            echo
            echo "Status file: $STATUS"
        } | mail -s "[D11 monitor] $NAME alert" ndarmon@g.harvard.edu 2>/dev/null || true
    fi
fi

# If nothing is in the queue any more, declare done and stop firing.
if [ "$ANY_LIVE" -eq 0 ]; then
    {
        echo
        echo "ALL DONE at $NOW_ISO. Subsequent ticks will be no-ops."
    } >> "$STATUS"
    echo "ALL_DONE at $NOW_ISO" > "$STATE_DIR/done.flag"
fi
