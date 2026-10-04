#!/bin/bash
#SBATCH --job-name=stage1-seven-asr
#SBATCH --partition=lrz-hgx-h100-94x4
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=120G
#SBATCH --time=14:00:00
#SBATCH --output=logs/stage1_seven_asr_%j.out
#SBATCH --error=logs/stage1_seven_asr_%j.err

set -euo pipefail

export REPO=${REPO:-/dss/dsshome1/04/ge75vid2/haoyu/live-ultrasound-video-understanding}
export DATA=${DATA:-/dss/mcmlscratch/04/ge75vid2/haoyu/live-ultrasound-video-understanding}
FULL_MAP=${FULL_MAP:-$DATA/cluster_data/splits/train_full295_qwen35_pretrain_keep_videos.json}
BASELINE_SAMPLES=${BASELINE_SAMPLES:-$DATA/cluster_data/pretrain/train_full295_stage1_samples.jsonl}
PILOT_MAP=${PILOT_MAP:-$DATA/cluster_data/splits/train_full295_stage1_seven_pilot.json}
MERGED_SAMPLES=${MERGED_SAMPLES:-$DATA/cluster_data/pretrain/train_full295_stage1_merged_193_samples.jsonl}

cd "$REPO"
if [[ -f /dss/dsshome1/04/ge75vid2/miniconda3/etc/profile.d/conda.sh ]]; then
  source /dss/dsshome1/04/ge75vid2/miniconda3/etc/profile.d/conda.sh
else
  source "$HOME/miniconda3/etc/profile.d/conda.sh"
fi
conda activate ultrasound

[[ -f "$FULL_MAP" ]] || { echo "Missing full keep-map: $FULL_MAP" >&2; exit 1; }
[[ -f "$BASELINE_SAMPLES" ]] || { echo "Missing 186-video baseline: $BASELINE_SAMPLES" >&2; exit 1; }

python -m stage1.seven_video_pilot select --full-map "$FULL_MAP" --output "$PILOT_MAP"

# Reuse the existing vLLM launcher with a map containing exactly seven videos.
export SPLIT=train_full295
export VIDEO_MAP="$PILOT_MAP"
export OUTPUT_DIR=${OUTPUT_DIR:-$DATA/cluster_data/QA/train_full295/transcripts_stage1_seven_clean}
export AUDIT_OUTPUT=${AUDIT_OUTPUT:-$DATA/cluster_data/QA/train_full295/stage1_seven_clean_audit.jsonl}
export SAMPLES_OUTPUT=${SAMPLES_OUTPUT:-$DATA/cluster_data/pretrain/train_full295_stage1_seven_clean_samples.jsonl}
export BATCH_SEGMENTS=${BATCH_SEGMENTS:-4}
bash scripts/slurm/run_stage1_qwen35_asr_clean.sbatch

python -m stage1.seven_video_pilot merge \
  --full-map "$FULL_MAP" \
  --pilot-map "$PILOT_MAP" \
  --baseline "$BASELINE_SAMPLES" \
  --pilot-samples "$SAMPLES_OUTPUT" \
  --output "$MERGED_SAMPLES"

echo "Seven-video ASR audit: $AUDIT_OUTPUT"
echo "Merged 193-video Stage 1 samples: $MERGED_SAMPLES"
