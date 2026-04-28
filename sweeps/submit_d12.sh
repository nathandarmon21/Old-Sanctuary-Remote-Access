#!/bin/bash
# submit_d12.sh — submit the full D12 production sweep (20 runs).
#
# Order: interleaved (np-42, eb-42, np-43, eb-43, ...). The matched pair
# for seed=42 hits the head of our user queue so it starts first; the
# other 9 matched pairs follow in seed order. With ~4-6 H100s held
# concurrently for our user, the first matched pair completes in ~5h
# and subsequent pairs progress in waves.
#
# Default partition: gpu_requeue with H100 (preemptible, but our
# checkpoint/resume + --requeue handles preemption transparently).
# Override via env vars:
#   PARTITION=gpu      (non-preemptible, A100 only)
#   GPU_TYPE=nvidia_h200
#
# Walltime: 12h (well under gpu_requeue's 3-day cap, sized for
# H100 + prefix-cache + max_par=12 stack which projects ~5h compute).

set -euo pipefail

SWEEP_NAME="${1:-d12_h100}"
PARTITION="${PARTITION:-gpu_requeue}"
GPU_TYPE="${GPU_TYPE:-nvidia_h100_80gb_hbm3}"
SCRATCH=/n/netscratch/zittrain_lab/Everyone/ndarmon
SWEEP_DIR=${SCRATCH}/${SWEEP_NAME}
CONFIG=configs/long_horizon_d12_vllm.yaml
PROTOCOLS=("no_protocol" "ebay_feedback")
SEEDS=(42 43 44 45 46 47 48 49 50 51)

mkdir -p "${SWEEP_DIR}"
echo "Sweep: ${SWEEP_NAME}"
echo "Partition: ${PARTITION}"
echo "GPU type: ${GPU_TYPE}"
echo "Output base: ${SWEEP_DIR}"
echo "Order: interleaved (matched pairs first)"
echo

JOB_IDS=()
JOB_LOG="${SWEEP_DIR}/submitted_jobs.tsv"
echo -e "job_id\tprotocol\tseed\toutput_dir" > "${JOB_LOG}"

# requeue flag only meaningful on preemptible partitions.
REQUEUE_FLAG=()
if [[ "${PARTITION}" == *"requeue"* ]]; then
    REQUEUE_FLAG=(--requeue)
fi

# Interleave: for each seed, submit both protocols back-to-back.
# That way SLURM's FIFO-within-priority gives matched pairs adjacent
# scheduling slots and the seed=42 pair lands at the head.
for SEED in "${SEEDS[@]}"; do
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
        echo "  ${RUN_NAME}: job ${JID}"
    done
done

echo
echo "Submitted ${#JOB_IDS[@]} jobs."
echo "Log: ${JOB_LOG}"
