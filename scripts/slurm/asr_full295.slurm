#!/bin/bash
#SBATCH --job-name=asr_full295
#SBATCH --partition=mcml-hgx-h100-94x4
#SBATCH --qos=mcml
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=24:00:00
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err

set -euo pipefail

REPO_DIR="/dss/dsshome1/04/ge75vid2/haoyu/live-ultrasound-video-understanding"
cd "${REPO_DIR}"

DATA_ROOT="${DATA_ROOT:-cluster_data}"
DATASET_NAME="${DATASET_NAME:-full295}"
TRAIN_SPLIT="${TRAIN_SPLIT:-train_${DATASET_NAME}}"
EVAL_SPLIT="${EVAL_SPLIT:-eval_${DATASET_NAME}}"
WHISPER_MODEL="${WHISPER_MODEL:-large-v3}"
LANGUAGE="${LANGUAGE:-en}"

TRAIN_VIDEO_DIR="${DATA_ROOT}/videos/${TRAIN_SPLIT}"
EVAL_VIDEO_DIR="${DATA_ROOT}/videos/${EVAL_SPLIT}"
TRAIN_TRANSCRIPTS="${DATA_ROOT}/QA/${TRAIN_SPLIT}/transcripts"
EVAL_TRANSCRIPTS="${DATA_ROOT}/QA/${EVAL_SPLIT}/transcripts"

# Keep HuggingFace / model caches off dsshome quota.
SCRATCH_ROOT="${SCRATCH_ROOT:-/dss/mcmlscratch/04/ge75vid2/haoyu}"
export HF_HOME="${HF_HOME:-${SCRATCH_ROOT}/hf_cache}"
export HUGGINGFACE_HUB_CACHE="${HUGGINGFACE_HUB_CACHE:-${HF_HOME}/hub}"
export TRANSFORMERS_CACHE="${TRANSFORMERS_CACHE:-${HF_HOME}/transformers}"
export XDG_CACHE_HOME="${XDG_CACHE_HOME:-${SCRATCH_ROOT}/xdg_cache}"
export TORCH_HOME="${TORCH_HOME:-${SCRATCH_ROOT}/torch_cache}"

source /dss/dsshome1/04/ge75vid2/miniconda3/etc/profile.d/conda.sh
conda activate ultrasound

mkdir -p logs "${TRAIN_TRANSCRIPTS}" "${EVAL_TRANSCRIPTS}" \
  "${HF_HOME}" "${HUGGINGFACE_HUB_CACHE}" "${TRANSFORMERS_CACHE}" \
  "${XDG_CACHE_HOME}" "${TORCH_HOME}"

echo "======================================================================"
echo "[job] ASR-only dataset=${DATASET_NAME}"
echo "[job] job_id=${SLURM_JOB_ID:-unknown}"
echo "[job] node=${SLURMD_NODENAME:-unknown}"
echo "[job] start_time=$(date)"
echo "[job] TRAIN_VIDEO_DIR=${TRAIN_VIDEO_DIR}"
echo "[job] EVAL_VIDEO_DIR=${EVAL_VIDEO_DIR}"
echo "[job] WHISPER_MODEL=${WHISPER_MODEL}"
echo "[job] LANGUAGE=${LANGUAGE}"
echo "[job] HF_HOME=${HF_HOME}"
echo "======================================================================"

python - <<'PY'
import torch
print("[env] torch:", torch.__version__)
print("[env] cuda:", torch.cuda.is_available())
print("[env] gpu:", torch.cuda.get_device_name(0) if torch.cuda.is_available() else None)
PY

echo "== train ASR: model loaded once for this split =="
python scripts/asr_pipeline.py \
  --input-dir "${TRAIN_VIDEO_DIR}" \
  --output-dir "${TRAIN_TRANSCRIPTS}" \
  --model "${WHISPER_MODEL}" \
  --device cuda \
  --language "${LANGUAGE}"

echo "== eval ASR: model loaded once for this split =="
python scripts/asr_pipeline.py \
  --input-dir "${EVAL_VIDEO_DIR}" \
  --output-dir "${EVAL_TRANSCRIPTS}" \
  --model "${WHISPER_MODEL}" \
  --device cuda \
  --language "${LANGUAGE}"

echo "== summary =="
echo "train transcripts: $(find "${TRAIN_TRANSCRIPTS}" -name '*.json' 2>/dev/null | wc -l)"
echo "eval transcripts : $(find "${EVAL_TRANSCRIPTS}" -name '*.json' 2>/dev/null | wc -l)"
echo "[job] finished_time=$(date)"