#!/usr/bin/env bash
# Generate Stage 2 teacher visual summaries from a LOCAL anchor video.
#
# Uses scripts/data/generate_teacher_visual_summaries.py end-to-end:
#   local video --(ffmpeg clips)--> OpenRouter teacher --> per-block JSONL
#   --(--emit-stage2-blocks)--> trainable Stage 2 blocks JSONL.
#
# OPENROUTER_API_KEY is read from the environment or the repo .env automatically
# (scripts/_video_llm.py imports _env_loader at import time).
#
# Usage:
#   # 1) cheap smoke test: annotate only the first minute, verify the pipeline
#   scripts/data/run_local_summary.sh
#
#   # 2) full run: annotate the whole video
#   FULL=1 scripts/data/run_local_summary.sh
#
#   # override any input/output
#   VIDEO=/path/to/clip.mp4 VIDEO_ID=myclip scripts/data/run_local_summary.sh
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$REPO_ROOT"

VIDEO="${VIDEO:-azure_data/videos/8V649L5Q368.mp4}"
VIDEO_ID="${VIDEO_ID:-$(basename "$VIDEO" | sed 's/\.[^.]*$//')}"
OUT_DIR="${OUT_DIR:-results/teacher_visual_summaries}"
BLOCKS_DIR="${BLOCKS_DIR:-data/stage2_blocks}"
MODEL="${MODEL:-google/gemini-3.1-pro-preview}"

TEACHER_JSONL="${OUT_DIR}/${VIDEO_ID}_teacher_blocks.jsonl"
BLOCKS_JSONL="${BLOCKS_DIR}/${VIDEO_ID}.jsonl"

# Default to a one-minute smoke test; set FULL=1 for the whole video.
MAX_BLOCKS_ARG=(--max-blocks 1)
if [[ "${FULL:-0}" == "1" ]]; then
  MAX_BLOCKS_ARG=()
fi

if [[ ! -f "$VIDEO" ]]; then
  echo "ERROR: video not found: $VIDEO" >&2
  exit 1
fi

echo "[run_local_summary] video=$VIDEO video_id=$VIDEO_ID model=$MODEL"
echo "[run_local_summary] teacher_jsonl=$TEACHER_JSONL"
echo "[run_local_summary] blocks_jsonl=$BLOCKS_JSONL"
echo "[run_local_summary] mode=$([[ "${FULL:-0}" == "1" ]] && echo FULL || echo 'SMOKE (first block only)')"

python3 scripts/data/generate_teacher_visual_summaries.py \
  --video "$VIDEO" \
  --video-id "$VIDEO_ID" \
  --output "$TEACHER_JSONL" \
  --model "$MODEL" \
  --block-sec 60 --local-sec 10 \
  --global-mode incremental \
  --emit-stage2-blocks "$BLOCKS_JSONL" \
  --overwrite \
  "${MAX_BLOCKS_ARG[@]}"

echo "[run_local_summary] done. Trainable Stage 2 blocks: $BLOCKS_JSONL"
