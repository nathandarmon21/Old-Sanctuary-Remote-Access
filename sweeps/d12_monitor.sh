#!/bin/bash
# d12_monitor.sh — server-side cron watchdog for the D12 production sweep.
#
# Runs every 30 minutes from cron. Reads the list of D12 job IDs from
# /n/netscratch/.../d12_v1/submitted_jobs.tsv (written by submit_d12.sh)
# and tracks each job through PENDING → RUNNING → COMPLETED/TIMEOUT/FAILED.
#
# What it produces:
#   /n/netscratch/.../d12_monitor/status.txt — single-page state of all 20 jobs
#   /n/netscratch/.../d12_monitor/monitor.log — append-only history per tick
#   /n/netscratch/.../d12_monitor/done.flag — touched when all 20 are terminal
#
# What it alerts on (via mail if available, always logged):
#   - Job ends in any non-COMPLETED final state (TIMEOUT/FAILED/CANCELLED)
#   - Job RUNNING >30 min with events.jsonl growing <5 events since last tick
#
# What it does NOT do:
#   - Auto-resubmit (chained jobs aren't planned for D12; manual relaunch
#     if a seed truly needs continuation)
#   - Cancel jobs

set -uo pipefail

SWEEP_NAME="${SWEEP_NAME:-d12_v1}"
SCRATCH=/n/netscratch/zittrain_lab/Everyone/ndarmon
SWEEP_DIR=${SCRATCH}/${SWEEP_NAME}
STATE_DIR=${SCRATCH}/d12_monitor
JOB_LOG=${SWEEP_DIR}/submitted_jobs.tsv

mkdir -p "${STATE_DIR}"

if [ -f "${STATE_DIR}/done.flag" ]; then
    exit 0
fi
if [ ! -f "${JOB_LOG}" ]; then
    echo "[$(date -u +%FT%TZ)] $JOB_LOG missing; nothing to monitor yet" \
        >> "${STATE_DIR}/monitor.log"
    exit 0
fi

NOW_ISO=$(date -u +%FT%TZ)
LOG="${STATE_DIR}/monitor.log"
STATUS="${STATE_DIR}/status.txt"

echo "=== tick $NOW_ISO ===" >> "${LOG}"
{
    echo "D12 production sweep monitor — last updated $NOW_ISO UTC"
    echo "Sweep: ${SWEEP_NAME}"
    echo
} > "${STATUS}"

ANY_LIVE=0
declare -A STATE_COUNTS
ALERTS=()

# Read submitted_jobs.tsv (skip header).
while IFS=$'\t' read -r JID PROTOCOL SEED OUTPUT_DIR; do
    [ "${JID}" = "job_id" ] && continue  # header
    [ -z "${JID}" ] && continue
    NAME="${PROTOCOL}_seed${SEED}"

    SQ=$(squeue -h -j "${JID}" -o "%T" 2>/dev/null | head -1)
    if [ -z "${SQ}" ]; then
        FINAL=$(sacct -j "${JID}" -n -P -o State 2>/dev/null | head -1 | tr -d '+ ')
        [ -z "${FINAL}" ] && FINAL="UNKNOWN"
        STATE_COUNTS[$FINAL]=$((${STATE_COUNTS[$FINAL]:-0} + 1))
        echo "  ${NAME} (job ${JID}): ${FINAL}" >> "${STATUS}"
        case "${FINAL}" in
            COMPLETED) ;;
            TIMEOUT|FAILED|CANCELLED|NODE_FAIL|OUT_OF_MEMORY|UNKNOWN)
                ALERTS+=("${NAME} (job ${JID}) ended ${FINAL}")
                ;;
        esac
        continue
    fi

    # Live in queue.
    ANY_LIVE=1
    STATE_COUNTS[$SQ]=$((${STATE_COUNTS[$SQ]:-0} + 1))
    EV="${OUTPUT_DIR}/events.jsonl"
    EV_NOW=$(wc -l < "${EV}" 2>/dev/null || echo 0)
    EV_PREV=$(cat "${STATE_DIR}/${NAME}.events" 2>/dev/null || echo 0)
    echo "${EV_NOW}" > "${STATE_DIR}/${NAME}.events"

    if [ "${SQ}" = "RUNNING" ]; then
        START_TIME=$(squeue -h -j "${JID}" -o "%S" 2>/dev/null | head -1)
        START_SEC=$(date -d "${START_TIME}" +%s 2>/dev/null || date +%s)
        NOW_SEC=$(date +%s)
        ELAPSED=$((NOW_SEC - START_SEC))
        # Last day_end (for at-a-glance progress).
        LAST_DAY=$(grep day_end "${EV}" 2>/dev/null | tail -1 \
            | python3 -c 'import json,sys
try:
    L=sys.stdin.read().strip()
    if L: print(json.loads(L)["day"])
except Exception: pass' 2>/dev/null)
        echo "  ${NAME} (job ${JID}): RUNNING ${ELAPSED}s, day=${LAST_DAY:-?}, events=${EV_NOW}" >> "${STATUS}"
        if [ "${ELAPSED}" -gt 1800 ] && [ "$((EV_NOW - EV_PREV))" -lt 5 ]; then
            ALERTS+=("${NAME} (job ${JID}) running ${ELAPSED}s, events grew ${ELAPSED}s only $((EV_NOW - EV_PREV)) since last tick — possible hang")
        fi
    else
        echo "  ${NAME} (job ${JID}): ${SQ}" >> "${STATUS}"
    fi
done < "${JOB_LOG}"

{
    echo
    echo "STATE COUNTS:"
    for k in "${!STATE_COUNTS[@]}"; do
        echo "  ${k}: ${STATE_COUNTS[${k}]}"
    done
} >> "${STATUS}"

if [ ${#ALERTS[@]} -gt 0 ]; then
    {
        echo
        echo "ALERTS:"
        for a in "${ALERTS[@]}"; do echo "  - ${a}"; done
    } >> "${STATUS}"
    {
        echo "ALERTS at ${NOW_ISO}:"
        for a in "${ALERTS[@]}"; do echo "  - ${a}"; done
    } >> "${LOG}"
    if command -v mail >/dev/null 2>&1; then
        {
            echo "D12 monitor alerts at ${NOW_ISO}:"
            for a in "${ALERTS[@]}"; do echo "- ${a}"; done
            echo
            echo "Status: ${STATUS}"
        } | mail -s "[D12 monitor] alerts" ndarmon@g.harvard.edu 2>/dev/null || true
    fi
fi

if [ "${ANY_LIVE}" -eq 0 ]; then
    {
        echo
        echo "ALL DONE at ${NOW_ISO}. Subsequent ticks will be no-ops."
    } >> "${STATUS}"
    echo "ALL_DONE at ${NOW_ISO}" > "${STATE_DIR}/done.flag"
fi
