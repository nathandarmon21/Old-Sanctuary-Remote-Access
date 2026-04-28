#!/bin/bash
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --mem=120G
#SBATCH --cpus-per-task=8
#SBATCH --time=24:00:00
#SBATCH --signal=SIGTERM@180
#SBATCH --mail-type=FAIL
#SBATCH --mail-user=ndarmon@g.harvard.edu

# Per-job SLURM script for vLLM-backed long-horizon runs.
#
# Required env vars (set by submitter):
#   CONFIG_PATH  — YAML config (must use provider: vllm)
#   PROTOCOL     — protocol override (no_protocol | ebay_feedback)
#   SEED         — integer seed
#   OUTPUT_DIR   — absolute path to the run directory; reused across
#                  chained jobs (engine.try_resume picks up checkpoints)
#   MODEL_NAME   — HF repo id for vLLM (default: Qwen/Qwen2.5-32B-Instruct-AWQ)
#
# Brings up a vLLM OpenAI-compatible server on port 8000 with continuous
# batching, waits for it to load the model, then runs sanctuary.run.
# vLLM is killed cleanly on exit so SLURM can clean up the GPU.

set -uo pipefail

MODEL_NAME="${MODEL_NAME:-Qwen/Qwen2.5-32B-Instruct-AWQ}"
VLLM_PORT=8000
VLLM_LOG="${OUTPUT_DIR}/vllm.log"

echo "=== Long-horizon vLLM job ==="
echo "SLURM_JOB_ID: ${SLURM_JOB_ID}"
echo "CONFIG_PATH: ${CONFIG_PATH}"
echo "PROTOCOL: ${PROTOCOL}"
echo "SEED: ${SEED}"
echo "OUTPUT_DIR: ${OUTPUT_DIR}"
echo "MODEL_NAME: ${MODEL_NAME}"

# CUDA + Python env. The vllm_env on netscratch has vLLM 0.11.2 + the
# sanctuary package installed editable, all wheels prebuilt for py3.10.
module purge
module load python/3.10.13-fasrc01 cuda/12.4.1-fasrc01 cudnn/9.5.1.17_cuda12-fasrc01
source /n/netscratch/zittrain_lab/Everyone/ndarmon/vllm_env/bin/activate

# HF cache on netscratch (avoid home quota).
export HF_HOME=/n/netscratch/zittrain_lab/Everyone/ndarmon/hf_cache
mkdir -p "${HF_HOME}"

mkdir -p "${OUTPUT_DIR}"
cd "${SLURM_SUBMIT_DIR}"

# ── Bring up vLLM server in the background ──
# gpu-memory-utilization 0.85 leaves a small margin for HF tokenizer
# and our own python process.
echo "[$(date -u +%FT%TZ)] Launching vLLM server..."
python3 -m vllm.entrypoints.openai.api_server \
    --model "${MODEL_NAME}" \
    --port ${VLLM_PORT} \
    --gpu-memory-utilization 0.85 \
    --max-model-len 32768 \
    --enable-prefix-caching \
    --disable-log-stats \
    > "${VLLM_LOG}" 2>&1 &
VLLM_PID=$!
echo "vLLM PID: ${VLLM_PID}"

# Cleanup vLLM on exit (normal or signaled).
trap 'echo "[$(date -u +%FT%TZ)] Killing vLLM PID ${VLLM_PID}"; kill -TERM ${VLLM_PID} 2>/dev/null; wait ${VLLM_PID} 2>/dev/null' EXIT TERM INT

# Wait for vLLM to be ready (up to 10 min for first-time model load).
echo "[$(date -u +%FT%TZ)] Waiting for vLLM /v1/models..."
READY=0
for i in $(seq 1 120); do
    if curl -sf "http://localhost:${VLLM_PORT}/v1/models" >/dev/null 2>&1; then
        READY=1
        echo "[$(date -u +%FT%TZ)] vLLM ready after ${i} attempts"
        break
    fi
    if ! kill -0 ${VLLM_PID} 2>/dev/null; then
        echo "[$(date -u +%FT%TZ)] vLLM died before becoming ready. Tail of log:"
        tail -30 "${VLLM_LOG}"
        exit 1
    fi
    sleep 5
done
if [ "${READY}" -ne 1 ]; then
    echo "[$(date -u +%FT%TZ)] vLLM never became ready in 10 min. Tail of log:"
    tail -50 "${VLLM_LOG}"
    exit 1
fi

# ── Run the simulation ──
echo "[$(date -u +%FT%TZ)] Starting sanctuary.run..."
python3 -m sanctuary.run \
    --config "${CONFIG_PATH}" \
    --seed "${SEED}" \
    --output "${OUTPUT_DIR}" \
    --protocol "${PROTOCOL}"
RC=$?

echo "[$(date -u +%FT%TZ)] sanctuary.run exit code: ${RC}"
exit ${RC}
