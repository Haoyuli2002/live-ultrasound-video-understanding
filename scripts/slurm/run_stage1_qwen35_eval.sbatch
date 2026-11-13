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
: "${REPO:?}" "${DATA:?}" "${OUTPUT_DIR:?}"
cd "$REPO"
source "${CONDA_SH:-/dss/dsshome1/04/ge75vid2/miniconda3/etc/profile.d/conda.sh}"
conda activate ultrasound
source "$REPO/scripts/slurm/stage1_storage.sh"
# Default is the identical held-out validation split created by each trainer.
# Supply both EVAL_JSONL and EVAL_VIDEO_MAP for a separate final test split.
TRAINING_CONFIG="$OUTPUT_DIR/training_config.json"
python -m stage1.eval_config --training-config "$TRAINING_CONFIG"
if [[ -n ${EVAL_JSONL:-} && -z ${EVAL_VIDEO_MAP:-} ]]; then
  echo 'Set EVAL_VIDEO_MAP together with an explicit EVAL_JSONL.' >&2; exit 1
fi
EVAL_JSONL=${EVAL_JSONL:-$OUTPUT_DIR/validation_samples.jsonl}
if [[ -z ${EVAL_VIDEO_MAP:-} ]]; then
  EVAL_VIDEO_MAP=$(python -c 'import json,sys; print(json.load(open(sys.argv[1]))["video_path_map"])' "$TRAINING_CONFIG")
fi
[[ -s "$EVAL_JSONL" && -f "$EVAL_VIDEO_MAP" && -d "$OUTPUT_DIR/final" ]]
for variant in base lora; do
  eval_args=(--training-config "$TRAINING_CONFIG")
  [[ "$variant" != lora ]] || eval_args+=(--adapter-path "$OUTPUT_DIR/final")
  python -m stage1.evaluate \
    "${eval_args[@]}" \
    --eval-jsonl "$EVAL_JSONL" --video-path-map "$EVAL_VIDEO_MAP" \
    --output "$OUTPUT_DIR/eval/${variant}_normal.jsonl"
done
