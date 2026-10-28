#!/bin/bash
#SBATCH --job-name=stage1-eval
#SBATCH --partition=lrz-hgx-h100-94x4
#SBATCH --qos=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=120G
#SBATCH --time=24:00:00
#SBATCH --output=logs/stage1_eval_%j.out
#SBATCH --error=logs/stage1_eval_%j.err
set -euo pipefail
: "${REPO:?}" "${DATA:?}" "${OUTPUT_DIR:?}" "${EVAL_DIR:?}"
cd "$REPO"
source "${CONDA_SH:-/dss/dsshome1/04/ge75vid2/miniconda3/etc/profile.d/conda.sh}"
conda activate "${CONDA_ENV:-ultrasound}"
source scripts/slurm/stage1_storage.sh
export PYTHONUNBUFFERED=1
EVAL_JSONL=${EVAL_JSONL:-$OUTPUT_DIR/validation_samples.jsonl}
EVAL_VIDEO_MAP=${EVAL_VIDEO_MAP:-$DATA/cluster_data/splits/train_full295_qwen35_pretrain_keep_videos.json}
[[ -s "$OUTPUT_DIR/final/adapter_model.safetensors" ]]
python -m stage1.prepare_eval --input "$EVAL_JSONL" --split "$OUTPUT_DIR/video_split.json" \
  --output "$EVAL_DIR/samples.jsonl" --limit-targets "${LIMIT_TARGETS:-0}"
for control in ${VISUAL_CONTROLS:-normal}; do
  for variant in base lora; do
    adapter=()
    [[ $variant != lora ]] || adapter=(--adapter-path "$OUTPUT_DIR/final")
    python -m stage1.evaluate --model-name "${MODEL_NAME:-Qwen/Qwen3.5-2B}" "${adapter[@]}" \
      --eval-jsonl "$EVAL_DIR/samples.jsonl" --video-path-map "$EVAL_VIDEO_MAP" \
      --frame-budget "${FRAME_BUDGET:-30}" --frame-size "${FRAME_SIZE:-224}" \
      --frame-sampling "${FRAME_SAMPLING:-uniform}" --max-new-tokens "${MAX_NEW_TOKENS:-128}" \
      --visual-control "$control" --output "$EVAL_DIR/${variant}_${control}.jsonl"
  done
  python -m stage1.score_predictions --predictions "$EVAL_DIR/lora_${control}.jsonl" \
    --baseline "$EVAL_DIR/base_${control}.jsonl" --output "$EVAL_DIR/metrics_${control}.json"
done
