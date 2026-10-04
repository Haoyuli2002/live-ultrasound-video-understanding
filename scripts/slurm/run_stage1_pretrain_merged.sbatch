#!/bin/bash
#SBATCH --job-name=stage1-pretrain
#SBATCH --partition=lrz-hgx-h100-94x4
#SBATCH --qos=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=120G
#SBATCH --time=14:00:00
#SBATCH --output=logs/stage1_pretrain_%j.out
#SBATCH --error=logs/stage1_pretrain_%j.err

set -euo pipefail

REPO=${REPO:-/dss/dsshome1/04/ge75vid2/haoyu/live-ultrasound-video-understanding}
DATA=${DATA:-/dss/mcmlscratch/04/ge75vid2/haoyu/live-ultrasound-video-understanding}
TRAIN_JSONL=${TRAIN_JSONL:-$DATA/cluster_data/pretrain/train_full295_stage1_merged_193_samples.jsonl}
VIDEO_MAP=${VIDEO_MAP:-$DATA/cluster_data/splits/train_full295_qwen35_pretrain_keep_videos.json}
RUN_NAME=${RUN_NAME:-stage1_full295_merged_qwen3vl2b}
OUTPUT_DIR=${OUTPUT_DIR:-$DATA/cluster_data/experiments/stage1/$RUN_NAME}
MODEL_NAME=${MODEL_NAME:-Qwen/Qwen3-VL-2B-Instruct}
MAX_STEPS=${MAX_STEPS:--1}
EPOCHS=${EPOCHS:-3}
FRAME_BUDGET=${FRAME_BUDGET:-120}
FRAME_SIZE=${FRAME_SIZE:-224}
FRAME_SAMPLING=${FRAME_SAMPLING:-uniform}
SAVE_STEPS=${SAVE_STEPS:-50}
LOGGING_STEPS=${LOGGING_STEPS:-10}
VALIDATION_FRACTION=${VALIDATION_FRACTION:-0.1}

cd "$REPO"
if [[ -f /dss/dsshome1/04/ge75vid2/miniconda3/etc/profile.d/conda.sh ]]; then
  source /dss/dsshome1/04/ge75vid2/miniconda3/etc/profile.d/conda.sh
else
  source "$HOME/miniconda3/etc/profile.d/conda.sh"
fi
conda activate ultrasound

[[ -f "$TRAIN_JSONL" ]] || { echo "Missing merged Stage 1 samples: $TRAIN_JSONL" >&2; exit 1; }
[[ -f "$VIDEO_MAP" ]] || { echo "Missing video map: $VIDEO_MAP" >&2; exit 1; }
mkdir -p "$OUTPUT_DIR"

export TRAIN_JSONL VIDEO_MAP
python - <<'PY'
import json, os
from pathlib import Path
rows = [json.loads(line) for line in Path(os.environ["TRAIN_JSONL"]).open(encoding="utf-8") if line.strip()]
videos = {row["video_id"] for row in rows}
expected = set(json.loads(Path(os.environ["VIDEO_MAP"]).read_text(encoding="utf-8")))
if videos != expected:
    raise SystemExit(f"Dataset coverage mismatch: samples={len(videos)}, keep-map={len(expected)}, missing={sorted(expected-videos)[:10]}")
print(f"Stage 1 preflight: {len(videos)} videos, {len(rows)} rows, {len(rows)//3} targets")
PY

echo "Job ${SLURM_JOB_ID:-unknown} on $(hostname): $MODEL_NAME"
echo "Train: $TRAIN_JSONL"
echo "Output: $OUTPUT_DIR; max_steps=$MAX_STEPS; epochs=$EPOCHS; frame_budget=$FRAME_BUDGET"
nvidia-smi

python -m stage1.train \
  --model-name "$MODEL_NAME" \
  --train-jsonl "$TRAIN_JSONL" \
  --video-path-map "$VIDEO_MAP" \
  --output-dir "$OUTPUT_DIR" \
  --epochs "$EPOCHS" \
  --max-steps "$MAX_STEPS" \
  --frame-budget "$FRAME_BUDGET" \
  --frame-size "$FRAME_SIZE" \
  --frame-sampling "$FRAME_SAMPLING" \
  --validation-fraction "$VALIDATION_FRACTION" \
  --gradient-accumulation-steps 8 \
  --learning-rate 1e-4 \
  --save-steps "$SAVE_STEPS" \
  --logging-steps "$LOGGING_STEPS" \
  --resume

echo "Stage 1 adapter: $OUTPUT_DIR/final"
