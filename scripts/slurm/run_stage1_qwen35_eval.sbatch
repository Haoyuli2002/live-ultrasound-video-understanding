#!/bin/bash
#SBATCH --job-name=stage1-qwen35-eval
#SBATCH --partition=lrz-hgx-h100-94x4
#SBATCH --qos=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=120G
#SBATCH --time=14:00:00
#SBATCH --output=logs/stage1_qwen35_eval_%j.out
#SBATCH --error=logs/stage1_qwen35_eval_%j.err
set -euo pipefail
: "${REPO:?}" "${DATA:?}" "${OUTPUT_DIR:?}" "${MODEL_NAME:?}"
cd "$REPO"
source "${CONDA_SH:-/dss/dsshome1/04/ge75vid2/miniconda3/etc/profile.d/conda.sh}"
conda activate ultrasound
source "$REPO/scripts/slurm/stage1_storage.sh"
# Default is the identical held-out validation split created by each trainer.
# Supply both EVAL_JSONL and EVAL_VIDEO_MAP for a separate final test split.
EVAL_JSONL=${EVAL_JSONL:-$OUTPUT_DIR/validation_samples.jsonl}
EVAL_VIDEO_MAP=${EVAL_VIDEO_MAP:-${VIDEO_MAP:-$DATA/cluster_data/splits/train_full295_qwen35_pretrain_keep_videos.json}}
[[ -s "$EVAL_JSONL" && -f "$EVAL_VIDEO_MAP" && -d "$OUTPUT_DIR/final" ]]
for variant in base lora; do
  adapter=()
  [[ "$variant" != lora ]] || adapter=(--adapter-path "$OUTPUT_DIR/final")
  python -m stage1.evaluate \
    --model-name "$MODEL_NAME" "${adapter[@]}" \
    --eval-jsonl "$EVAL_JSONL" --video-path-map "$EVAL_VIDEO_MAP" \
    --frame-budget "${FRAME_BUDGET:-120}" --frame-size "${FRAME_SIZE:-224}" \
    --frame-sampling "${FRAME_SAMPLING:-uniform}" \
    --output "$OUTPUT_DIR/eval/${variant}_normal.jsonl"
done
