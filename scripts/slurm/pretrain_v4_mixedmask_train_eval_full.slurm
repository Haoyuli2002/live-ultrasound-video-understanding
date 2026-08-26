#!/bin/bash
#SBATCH --job-name=v4_mix_train_eval
#SBATCH --partition=mcml-hgx-h100-94x4
#SBATCH --qos=mcml
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=120G
#SBATCH --time=36:00:00
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err

set -euo pipefail

REPO_DIR="/dss/dsshome1/04/ge75vid2/haoyu/live-ultrasound-video-understanding"
cd "${REPO_DIR}"

# Select the base model at submit time:
#
#   2B default:
#     sbatch scripts/slurm/pretrain_v4_mixedmask_train_eval_full.slurm
#
#   8B:
#     MODEL_NAME=Qwen/Qwen3-VL-8B-Instruct MODEL_TAG=qwen3vl_8b \
#       sbatch scripts/slurm/pretrain_v4_mixedmask_train_eval_full.slurm
#
MODEL_NAME="${MODEL_NAME:-Qwen/Qwen3-VL-2B-Instruct}"
MODEL_TAG="${MODEL_TAG:-qwen3vl_2b}"

# Keep code in REPO_DIR; put data/results under DATA_ROOT. Recommended setup:
#   cluster_data -> /dss/mcmlscratch/04/ge75vid2/haoyu/live-ultrasound-video-understanding/cluster_data
# Or override at submit time:
#   DATA_ROOT=/dss/mcmlscratch/04/ge75vid2/haoyu/live-ultrasound-video-understanding/cluster_data sbatch ...
DATA_ROOT="${DATA_ROOT:-cluster_data}"
RUN_NAME="${RUN_NAME:-pretrain_v4_mixedmask_${MODEL_TAG}_train50}"
RUN_DIR="${DATA_ROOT}/experiments/pretrain/${RUN_NAME}"
SAMPLE_DIR="${RUN_DIR}/samples"
OUTPUT_DIR="${RUN_DIR}/checkpoints/adapter"
EVAL_DIR="${RUN_DIR}/eval"
LOG_DIR="${RUN_DIR}/logs"
export DATA_ROOT

BUILD_SAMPLES="${BUILD_SAMPLES:-1}"
TRAIN_TRANSCRIPTS="${TRAIN_TRANSCRIPTS:-${DATA_ROOT}/QA/train/transcripts}"
EVAL_TRANSCRIPTS="${EVAL_TRANSCRIPTS:-${DATA_ROOT}/QA/eval/transcripts}"
TRAIN_JSONL="${TRAIN_JSONL:-${SAMPLE_DIR}/train_pretrain_punct_sentence_mixedmask_samples.jsonl}"
EVAL_JSONL="${EVAL_JSONL:-${SAMPLE_DIR}/eval_pretrain_punct_sentence_mixedmask_samples.jsonl}"
TRAIN_MAP="${TRAIN_MAP:-${DATA_ROOT}/splits/train_videos.json}"
EVAL_MAP="${EVAL_MAP:-${DATA_ROOT}/splits/eval_videos.json}"

BASE_EVAL="${EVAL_DIR}/base_full_nomask.jsonl"
LORA_EVAL="${EVAL_DIR}/lora_full_nomask.jsonl"
COMPARE_JSONL="${EVAL_DIR}/compare_full_nomask_direct.jsonl"
SUMMARY_JSON="${EVAL_DIR}/compare_full_nomask_direct_summary.json"

source /dss/dsshome1/04/ge75vid2/miniconda3/etc/profile.d/conda.sh
conda activate ultrasound

mkdir -p logs "${SAMPLE_DIR}" "${OUTPUT_DIR}" "${EVAL_DIR}" "${LOG_DIR}"

echo "======================================================================"
echo "[job] V4 mixedmask train + full eval"
echo "[job] job_id=${SLURM_JOB_ID:-unknown}"
echo "[job] node=${SLURMD_NODENAME:-unknown}"
echo "[job] start_time=$(date)"
echo "[job] repo=${REPO_DIR}"
echo "[job] DATA_ROOT=${DATA_ROOT}"
echo "[job] MODEL_NAME=${MODEL_NAME}"
echo "[job] MODEL_TAG=${MODEL_TAG}"
echo "[job] RUN_NAME=${RUN_NAME}"
echo "[job] RUN_DIR=${RUN_DIR}"
echo "[job] OUTPUT_DIR=${OUTPUT_DIR}"
echo "[job] BUILD_SAMPLES=${BUILD_SAMPLES}"
echo "[job] TRAIN_JSONL=${TRAIN_JSONL}"
echo "[job] EVAL_JSONL=${EVAL_JSONL}"
echo "[job] TRAIN_MAP=${TRAIN_MAP}"
echo "[job] EVAL_MAP=${EVAL_MAP}"
echo "======================================================================"

echo "== git version =="
git log --oneline -1 || true

echo "== environment =="
python - <<'PY'
import torch
print("[env] torch:", torch.__version__)
print("[env] cuda:", torch.cuda.is_available())
print("[env] gpu:", torch.cuda.get_device_name(0) if torch.cuda.is_available() else None)
print("[env] bf16:", torch.cuda.is_bf16_supported() if torch.cuda.is_available() else None)
PY

if [[ "${BUILD_SAMPLES}" == "1" ]]; then
  echo "== build V4 train samples =="
  python pretrain/build_samples.py \
    --transcripts "${TRAIN_TRANSCRIPTS}" \
    --output "${TRAIN_JSONL}" \
    --unit sentence \
    --sentence-mode punctuation \
    --format mixedmask \
    --history-units 3 \
    --frames-per-sentence 3 \
    --min-words 3 \
    --sentence-max-words 80 \
    --split-punctuation '.?!;:' \
    --mask-current-visual-prob 0.33 \
    --no-mask-prob 0.33 \
    --random-unit-modality-mask-prob 0.34
else
  echo "== build V4 train samples skipped (BUILD_SAMPLES=${BUILD_SAMPLES}) =="
fi

wc -l "${TRAIN_JSONL}"

echo "== train V4 mixedmask =="
python pretrain/train.py \
  --model-name "${MODEL_NAME}" \
  --train-jsonl "${TRAIN_JSONL}" \
  --video-path-map "${TRAIN_MAP}" \
  --output-dir "${OUTPUT_DIR}" \
  --window-size 4 \
  --frame-size 224 \
  --num-train-epochs 3 \
  --per-device-train-batch-size 1 \
  --gradient-accumulation-steps 8 \
  --learning-rate 1e-4 \
  --bf16 \
  --gradient-checkpointing \
  --early-stop-patience 3 \
  --early-stop-min-delta 0.001 \
  --resume-from-checkpoint auto

if [[ "${BUILD_SAMPLES}" == "1" ]]; then
  echo "== build V4 eval samples =="
  python pretrain/build_samples.py \
    --transcripts "${EVAL_TRANSCRIPTS}" \
    --output "${EVAL_JSONL}" \
    --unit sentence \
    --sentence-mode punctuation \
    --format mixedmask \
    --history-units 3 \
    --frames-per-sentence 3 \
    --min-words 3 \
    --sentence-max-words 80 \
    --split-punctuation '.?!;:' \
    --mask-current-visual-prob 0.33 \
    --no-mask-prob 0.33 \
    --random-unit-modality-mask-prob 0.34
else
  echo "== build V4 eval samples skipped (BUILD_SAMPLES=${BUILD_SAMPLES}) =="
fi

wc -l "${EVAL_JSONL}"

echo "== V4 full eval: base no-mask =="
python pretrain/infer.py \
  --model-name "${MODEL_NAME}" \
  --no-adapter \
  --eval-jsonl "${EVAL_JSONL}" \
  --video-path-map "${EVAL_MAP}" \
  --output "${BASE_EVAL}" \
  --window-size 4 \
  --frame-size 224 \
  --max-new-tokens 128 \
  --mixedmask-eval-mode no_mask \
  --bf16 \
  --quiet \
  --resume

echo "== V4 full eval: LoRA no-mask =="
python pretrain/infer.py \
  --model-name "${MODEL_NAME}" \
  --adapter-path "${OUTPUT_DIR}" \
  --eval-jsonl "${EVAL_JSONL}" \
  --video-path-map "${EVAL_MAP}" \
  --output "${LORA_EVAL}" \
  --window-size 4 \
  --frame-size 224 \
  --max-new-tokens 128 \
  --mixedmask-eval-mode no_mask \
  --bf16 \
  --quiet \
  --resume

echo "== compare V4 full direct metrics =="
python pretrain/compare_predictions.py \
  --base "${BASE_EVAL}" \
  --lora "${LORA_EVAL}" \
  --output "${COMPARE_JSONL}" \
  --summary-json "${SUMMARY_JSON}" \
  --no-semantic

echo "== V4 full eval summary =="
cat "${SUMMARY_JSON}"

echo "== output files =="
find "${RUN_DIR}" -maxdepth 3 \( -type f -o -type d \) | sort
wc -l "${BASE_EVAL}" "${LORA_EVAL}" "${COMPARE_JSONL}"

echo "== done =="