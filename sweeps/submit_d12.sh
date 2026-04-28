#!/bin/bash
# submit_d12.sh — submit the full D12 production sweep.
#
# 2 protocols × 10 seeds = 20 single-job sbatch submissions on the gpu
# partition with --time=24:00:00. Each job uses the tuned vLLM stack
# (configs/long_horizon_d12_vllm.yaml) and runs the full 150-day
# simulation. Output to /n/netscratch/zittrain_lab/Everyone/ndarmon/d12_v1/.
#
# Usage:
#   bash sweeps/submit_d12.sh [SWEEP_NAME]
#
# Default SWEEP_NAME=d12_v1.

set -euo pipefail

SWEEP_NAME="${1:-d12_v1}"
SCRATCH=/n/netscratch/zittrain_lab/Everyone/ndarmon
SWEEP_DIR=${SCRATCH}/${SWEEP_NAME}
CONFIG=configs/long_horizon_d12_vllm.yaml
PROTOCOLS=("no_protocol" "ebay_feedback")
SEEDS=(42 43 44 45 46 47 48 49 50 51)

mkdir -p "${SWEEP_DIR}"
echo "Sweep: ${SWEEP_NAME}"
echo "Output base: ${SWEEP_DIR}"
echo "Protocols: ${PROTOCOLS[*]}"
echo "Seeds: ${SEEDS[*]}"
echo

JOB_IDS=()
JOB_LOG="${SWEEP_DIR}/submitted_jobs.tsv"
echo -e "job_id\tprotocol\tseed\toutput_dir" > "${JOB_LOG}"

for PROTOCOL in "${PROTOCOLS[@]}"; do
    for SEED in "${SEEDS[@]}"; do
        RUN_NAME="${PROTOCOL}_seed${SEED}"
        OUTPUT_DIR="${SWEEP_DIR}/${RUN_NAME}"
        mkdir -p "${OUTPUT_DIR}"
        OUT=$(sbatch \
            --time=24:00:00 \
            --partition=gpu \
            --gres=gpu:1 \
            --mem=120G \
            --cpus-per-task=8 \
            --job-name="d12_${RUN_NAME}" \
            --output="${OUTPUT_DIR}/slurm_%j.out" \
            --error="${OUTPUT_DIR}/slurm_%j.err" \
            --signal=SIGTERM@180 \
            --mail-type=FAIL \
            --mail-user=ndarmon@g.harvard.edu \
            --export=ALL,CONFIG_PATH="${CONFIG}",PROTOCOL="${PROTOCOL}",SEED="${SEED}",OUTPUT_DIR="${OUTPUT_DIR}",MODEL_NAME=Qwen/Qwen2.5-32B-Instruct-AWQ \
            sweeps/run_long_horizon_vllm_job.sh)
        JID=$(echo "${OUT}" | grep -oE '[0-9]+' | tail -1)
        JOB_IDS+=("${JID}")
        printf "%s\t%s\t%s\t%s\n" "${JID}" "${PROTOCOL}" "${SEED}" "${OUTPUT_DIR}" >> "${JOB_LOG}"
        echo "  ${PROTOCOL} seed=${SEED}: job ${JID}"
    done
done

echo
echo "Submitted ${#JOB_IDS[@]} jobs."
echo "Log: ${JOB_LOG}"
echo
echo "Job IDs (space-separated): ${JOB_IDS[*]}"
