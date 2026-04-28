#!/bin/bash
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --mem=80G
#SBATCH --cpus-per-task=4
#SBATCH --time=24:00:00
#SBATCH --signal=SIGTERM@180
#SBATCH --mail-type=FAIL
#SBATCH --mail-user=ndarmon@g.harvard.edu

# Per-job SLURM script for one link in a long-horizon dependency chain.
#
# Required env vars (set by submit_chained.sh):
#   CONFIG_PATH  -- path to the YAML config (e.g. configs/long_horizon_qwen.yaml)
#   PROTOCOL     -- protocol override (no_protocol | ebay_feedback)
#   SEED         -- integer seed for this replicate
#   OUTPUT_DIR   -- absolute path to the run directory; reused across all
#                   chained jobs so checkpoint files accumulate in one place
#                   and try_resume picks up where the prior job left off.
#
# --signal=SIGTERM@180 gives the engine a 180s grace before walltime so the
# SIGTERM handler can checkpoint cleanly at the next day boundary.

set -euo pipefail

echo "=== Long-horizon job ==="
echo "SLURM_JOB_ID: ${SLURM_JOB_ID}"
echo "CONFIG_PATH: ${CONFIG_PATH}"
echo "PROTOCOL: ${PROTOCOL}"
echo "SEED: ${SEED}"
echo "OUTPUT_DIR: ${OUTPUT_DIR}"

# Activate Python environment (best-effort — depends on cluster setup).
if [ -f "${HOME}/sanctuary-env/bin/activate" ]; then
    source "${HOME}/sanctuary-env/bin/activate"
elif command -v conda &>/dev/null; then
    conda activate sanctuary 2>/dev/null || true
fi

# Ensure Ollama server is up and the model is pulled. Long-horizon uses
# Qwen 2.5 32B for both tiers; the judge model is 14B and is pulled at
# analysis time, not here.
if ! pgrep -x ollama >/dev/null 2>&1; then
    echo "Starting Ollama server..."
    ollama serve &
    sleep 10
fi
echo "Pulling models (no-op if already present)..."
ollama pull qwen2.5:32b

cd "${SLURM_SUBMIT_DIR}"

mkdir -p "${OUTPUT_DIR}"

# --output is the run directory. The engine's try_resume() at startup
# inspects ${OUTPUT_DIR}/checkpoints for the latest snapshot and resumes
# from there if found, so this same command is correct for every link
# in the chain.
python3 -m sanctuary.run \
    --config "${CONFIG_PATH}" \
    --seed "${SEED}" \
    --output "${OUTPUT_DIR}" \
    --protocol "${PROTOCOL}"

echo "=== Job complete ==="
