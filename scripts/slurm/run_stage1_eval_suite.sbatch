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
TRAINING_CONFIG="$OUTPUT_DIR/training_config.json"
python -m stage1.eval_config --training-config "$TRAINING_CONFIG"
if [[ -n ${EVAL_JSONL:-} && -z ${EVAL_VIDEO_MAP:-} ]]; then
  echo 'Set EVAL_VIDEO_MAP together with an explicit EVAL_JSONL.' >&2; exit 1
fi
EVAL_JSONL=${EVAL_JSONL:-$OUTPUT_DIR/validation_samples.jsonl}
if [[ -z ${EVAL_VIDEO_MAP:-} ]]; then
  EVAL_VIDEO_MAP=$(python -c 'import json,sys; print(json.load(open(sys.argv[1]))["video_path_map"])' "$TRAINING_CONFIG")
fi
[[ -s "$OUTPUT_DIR/final/adapter_model.safetensors" ]]
python -m stage1.prepare_eval --input "$EVAL_JSONL" --split "$OUTPUT_DIR/video_split.json" \
  --output "$EVAL_DIR/samples.jsonl" --limit-targets "${LIMIT_TARGETS:-0}"
for control in ${VISUAL_CONTROLS:-normal}; do
  for variant in base lora; do
    eval_args=(--training-config "$TRAINING_CONFIG")
    [[ $variant != lora ]] || eval_args+=(--adapter-path "$OUTPUT_DIR/final")
    python -m stage1.evaluate "${eval_args[@]}" \
      --eval-jsonl "$EVAL_DIR/samples.jsonl" --video-path-map "$EVAL_VIDEO_MAP" \
      --max-new-tokens "${MAX_NEW_TOKENS:-128}" \
      --visual-control "$control" --output "$EVAL_DIR/${variant}_${control}.jsonl"
  done
  python -m stage1.score_predictions --predictions "$EVAL_DIR/lora_${control}.jsonl" \
    --baseline "$EVAL_DIR/base_${control}.jsonl" --output "$EVAL_DIR/metrics_${control}.json"
done
