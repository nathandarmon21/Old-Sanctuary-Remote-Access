#!/bin/bash
# submit_chained.sh — submit a dependency chain of N sbatch jobs for one
# long-horizon simulation (one config, one protocol, one seed).
#
# Each job runs up to 24h on the gpu partition. SLURM sends SIGTERM 180s
# before walltime; the engine catches it, flushes a checkpoint at the
# next day boundary, and exits cleanly. The next job in the chain (queued
# with --dependency=afterany) picks up from that checkpoint via try_resume().
#
# Usage:
#   sweeps/submit_chained.sh CONFIG PROTOCOL SEED [DEPTH] [SWEEP_NAME]
#
# Example:
#   sweeps/submit_chained.sh configs/long_horizon_qwen.yaml \
#       no_protocol 42 3 long_horizon_v1

set -euo pipefail

CONFIG_PATH="${1:?CONFIG path required}"
PROTOCOL="${2:?protocol required (no_protocol | ebay_feedback)}"
SEED="${3:?seed required}"
DEPTH="${4:-3}"
SWEEP_NAME="${5:-long_horizon}"

if [ ! -f "${CONFIG_PATH}" ]; then
    echo "Error: config not found at ${CONFIG_PATH}" >&2
    exit 1
fi

# Output to netscratch (multi-TB quota); never to ~/sanctuary/runs which
# is on the 100GB home cap.
SCRATCH_BASE="/n/netscratch/zittrain_lab/Everyone/ndarmon/${SWEEP_NAME}"
RUN_NAME="${PROTOCOL}_seed${SEED}"
OUTPUT_DIR="${SCRATCH_BASE}/${RUN_NAME}"
mkdir -p "${OUTPUT_DIR}"

PER_JOB_SCRIPT="$(dirname "$0")/run_long_horizon_job.sh"

if [ ! -f "${PER_JOB_SCRIPT}" ]; then
    echo "Error: ${PER_JOB_SCRIPT} not found" >&2
    exit 1
fi

echo "Chain: ${DEPTH} job(s) for ${RUN_NAME}, output ${OUTPUT_DIR}"

PREV_JOB=""
for i in $(seq 1 "${DEPTH}"); do
    DEP_FLAG=()
    if [ -n "${PREV_JOB}" ]; then
        DEP_FLAG=(--dependency=afterany:"${PREV_JOB}")
    fi
    OUT=$(sbatch \
        "${DEP_FLAG[@]}" \
        --job-name="${SWEEP_NAME}_${RUN_NAME}_${i}" \
        --output="${OUTPUT_DIR}/slurm_chain${i}_%j.out" \
        --error="${OUTPUT_DIR}/slurm_chain${i}_%j.err" \
        --export=ALL,CONFIG_PATH="${CONFIG_PATH}",PROTOCOL="${PROTOCOL}",SEED="${SEED}",OUTPUT_DIR="${OUTPUT_DIR}" \
        "${PER_JOB_SCRIPT}")
    PREV_JOB=$(echo "${OUT}" | grep -oE '[0-9]+' | tail -1)
    echo "  link ${i}/${DEPTH}: job ${PREV_JOB}"
done

echo "Submitted ${DEPTH} chained job(s) for ${RUN_NAME}."
