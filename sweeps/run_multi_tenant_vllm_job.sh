#!/bin/bash
#SBATCH --partition=gpu_requeue
#SBATCH --gres=gpu:1
#SBATCH --mem=120G
#SBATCH --cpus-per-task=12
#SBATCH --time=24:00:00
#SBATCH --signal=SIGTERM@180
#SBATCH --mail-type=FAIL
#SBATCH --mail-user=ndarmon@g.harvard.edu

# Multi-tenant SLURM job: one vLLM server, N sanctuary.run simulations
# in parallel as clients. vLLM continuous batching multiplexes across
# the tenants, so GPU utilization is much higher than running one
# simulation per GPU.
#
# Each tenant has its own OUTPUT_DIR (independent checkpoints, events,
# logs). Tenants are independent processes; if one fails the others
# continue.
#
# Required env vars (set by submitter):
#   CONFIG_PATH  — yaml config (must use provider: vllm, base_url localhost:8000)
#   MODEL_NAME   — HF model id (default Qwen/Qwen2.5-32B-Instruct-AWQ)
#   TENANTS      — semicolon-separated list of "PROTOCOL:SEED:OUTPUT_DIR"
#                  e.g. "no_protocol:42:/scratch/np42;ebay_feedback:42:/scratch/eb42"
#
# Optional:
#   VLLM_PORT          — default 8000
#   VLLM_MAX_NUM_SEQS  — default 64

set -uo pipefail

VLLM_PORT="${VLLM_PORT:-8000}"
VLLM_MAX_NUM_SEQS="${VLLM_MAX_NUM_SEQS:-64}"
MODEL_NAME="${MODEL_NAME:-Qwen/Qwen2.5-32B-Instruct-AWQ}"
VLLM_LOG="${SLURM_SUBMIT_DIR:-$PWD}/vllm_${SLURM_JOB_ID:-local}.log"

echo "=== Multi-tenant vLLM job ==="
echo "SLURM_JOB_ID: ${SLURM_JOB_ID:-local}"
echo "CONFIG_PATH:  ${CONFIG_PATH}"
echo "MODEL_NAME:   ${MODEL_NAME}"
echo "TENANTS:      ${TENANTS}"

module purge
module load python/3.10.13-fasrc01 cuda/12.4.1-fasrc01 cudnn/9.5.1.17_cuda12-fasrc01
source /n/netscratch/zittrain_lab/Everyone/ndarmon/vllm_env/bin/activate
export HF_HOME=/n/netscratch/zittrain_lab/Everyone/ndarmon/hf_cache

cd "${SLURM_SUBMIT_DIR:-$PWD}"

# ── Bring up vLLM with throughput-tuned flags ──
DRAFT_MODEL="${DRAFT_MODEL:-Qwen/Qwen2.5-0.5B-Instruct}"
NUM_SPEC_TOKENS="${NUM_SPEC_TOKENS:-5}"

# vLLM 0.11+ uses JSON --speculative-config (older flags were dropped).
SPEC_CONFIG="{\"model\": \"${DRAFT_MODEL}\", \"num_speculative_tokens\": ${NUM_SPEC_TOKENS}}"

echo "[$(date -u +%FT%TZ)] Starting vLLM..."
python3 -m vllm.entrypoints.openai.api_server \
    --model "${MODEL_NAME}" \
    --port "${VLLM_PORT}" \
    --gpu-memory-utilization 0.92 \
    --max-model-len 32768 \
    --max-num-batched-tokens 16384 \
    --enable-prefix-caching \
    --enable-chunked-prefill \
    --speculative-config "${SPEC_CONFIG}" \
    --max-num-seqs "${VLLM_MAX_NUM_SEQS}" \
    --disable-log-stats \
    > "${VLLM_LOG}" 2>&1 &
VLLM_PID=$!
echo "vLLM PID: ${VLLM_PID}"

# Cleanup on exit.
trap 'echo "[$(date -u +%FT%TZ)] Killing vLLM PID ${VLLM_PID}"; kill -TERM ${VLLM_PID} 2>/dev/null; wait ${VLLM_PID} 2>/dev/null' EXIT TERM INT

# Wait for vLLM ready (10 min cap).
echo "[$(date -u +%FT%TZ)] Waiting for vLLM /v1/models..."
READY=0
for i in $(seq 1 120); do
    if curl -sf "http://localhost:${VLLM_PORT}/v1/models" >/dev/null 2>&1; then
        READY=1
        echo "[$(date -u +%FT%TZ)] vLLM ready after ${i} attempts"
        break
    fi
    if ! kill -0 ${VLLM_PID} 2>/dev/null; then
        echo "[$(date -u +%FT%TZ)] vLLM died before becoming ready"
        tail -50 "${VLLM_LOG}"
        exit 1
    fi
    sleep 5
done
[ "${READY}" -ne 1 ] && { echo "vLLM never became ready"; tail -50 "${VLLM_LOG}"; exit 1; }

# ── Launch all tenants in parallel ──
declare -a TENANT_PIDS
declare -a TENANT_NAMES
IFS=';' read -ra TENANT_LIST <<< "${TENANTS}"
echo "[$(date -u +%FT%TZ)] Launching ${#TENANT_LIST[@]} tenant(s)..."
for T in "${TENANT_LIST[@]}"; do
    IFS=':' read -ra T_PARTS <<< "${T}"
    PROTOCOL="${T_PARTS[0]}"
    SEED="${T_PARTS[1]}"
    OUTPUT_DIR="${T_PARTS[2]}"
    NAME="${PROTOCOL}_seed${SEED}"
    mkdir -p "${OUTPUT_DIR}"
    echo "  tenant ${NAME} → ${OUTPUT_DIR}"
    python3 -m sanctuary.run \
        --config "${CONFIG_PATH}" \
        --seed "${SEED}" \
        --output "${OUTPUT_DIR}" \
        --protocol "${PROTOCOL}" \
        > "${OUTPUT_DIR}/sanctuary.log" 2>&1 &
    TENANT_PIDS+=($!)
    TENANT_NAMES+=("${NAME}")
done

# ── Wait for all tenants ──
EXIT_BAD=0
for i in "${!TENANT_PIDS[@]}"; do
    PID="${TENANT_PIDS[$i]}"
    NAME="${TENANT_NAMES[$i]}"
    if wait "${PID}"; then
        echo "[$(date -u +%FT%TZ)] ${NAME} (PID ${PID}) exit OK"
    else
        RC=$?
        echo "[$(date -u +%FT%TZ)] ${NAME} (PID ${PID}) exit ${RC}"
        EXIT_BAD=1
    fi
done

echo "[$(date -u +%FT%TZ)] All tenants exited"
exit ${EXIT_BAD}
