#!/bin/bash
# submit_redesign_pilot.sh — submit the matched-pair pilot (no_protocol +
# ebay_feedback, seed=42) for the redesigned long-horizon experiment.
#
# This is the validation pilot for the 8-commit redesign:
#   - 100 days, V_E=$42 / V_P=$15, $80/day fixed cost, bankruptcy at $0
#   - widget-ID commitment + claim_rationale at offer placement
#   - Bayesian reputation + EV-anchored buyer reservation prices + gating
#   - living ledger in every tactical prompt
#
# Stack: H100 80GB on gpu_requeue (preemptible; --requeue + checkpoint
# resume handle preemption), 12h walltime, vLLM with prefix caching +
# speculative decoding (Qwen 2.5 0.5B draft) + max_par=12.

set -euo pipefail

SWEEP_NAME="${1:-d12_redesign_pilot}"
PARTITION="${PARTITION:-gpu_requeue}"
GPU_TYPE="${GPU_TYPE:-nvidia_h100_80gb_hbm3}"
SCRATCH=/n/netscratch/zittrain_lab/Everyone/ndarmon
SWEEP_DIR=${SCRATCH}/${SWEEP_NAME}
CONFIG=configs/long_horizon_d12_vllm.yaml

PROTOCOLS=(no_protocol ebay_feedback)
SEED=42

mkdir -p "${SWEEP_DIR}"
echo "Sweep: ${SWEEP_NAME}"
echo "Partition: ${PARTITION}"
echo "GPU type: ${GPU_TYPE}"
echo "Output base: ${SWEEP_DIR}"
echo "Seed: ${SEED}"
echo

REQUEUE_FLAG=()
if [[ "${PARTITION}" == *"requeue"* ]]; then
    REQUEUE_FLAG=(--requeue)
fi

JOB_LOG="${SWEEP_DIR}/submitted_jobs.tsv"
echo -e "job_id\tprotocol\tseed\toutput_dir" > "${JOB_LOG}"

for PROTOCOL in "${PROTOCOLS[@]}"; do
    RUN_NAME="${PROTOCOL}_seed${SEED}"
    OUTPUT_DIR="${SWEEP_DIR}/${RUN_NAME}"
    mkdir -p "${OUTPUT_DIR}"
    OUT=$(sbatch \
        --time=12:00:00 \
        --partition="${PARTITION}" \
        --gres="gpu:${GPU_TYPE}:1" \
        --mem=120G \
        --cpus-per-task=8 \
        "${REQUEUE_FLAG[@]}" \
        --job-name="d12r_${RUN_NAME}" \
        --output="${OUTPUT_DIR}/slurm_%j.out" \
        --error="${OUTPUT_DIR}/slurm_%j.err" \
        --export=ALL,CONFIG_PATH="${CONFIG}",PROTOCOL="${PROTOCOL}",SEED="${SEED}",OUTPUT_DIR="${OUTPUT_DIR}",MODEL_NAME="Qwen/Qwen2.5-32B-Instruct-AWQ" \
        sweeps/run_long_horizon_vllm_job.sh)
    JID=$(echo "${OUT}" | grep -oE '[0-9]+' | tail -1)
    printf "%s\t%s\t%s\t%s\n" "${JID}" "${PROTOCOL}" "${SEED}" "${OUTPUT_DIR}" >> "${JOB_LOG}"
    echo "  ${RUN_NAME}: job ${JID}"
done

echo
echo "Log: ${JOB_LOG}"
