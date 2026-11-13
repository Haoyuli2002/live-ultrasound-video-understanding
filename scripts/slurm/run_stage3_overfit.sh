#!/usr/bin/env bash
# Explicit mock (CPU) or Qwen (allocated CUDA) run. Never auto-submits a job.
set -euo pipefail
SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
REPO=${REPO:-$(cd "$SCRIPT_DIR/../.." && pwd)}
cd "$REPO"
PYTHON=${PYTHON:-python3}
BACKEND=${BACKEND:-mock}
DATA_DIR=${DATA_DIR:-$REPO/stage3/examples/two_video_overfit}
VIDEO_MAP=${VIDEO_MAP:-$DATA_DIR/video_map.json}
: "${RUN_DIR:?Set RUN_DIR to a new, non-existing output directory}"
common=(--backend "$BACKEND" --data-dir "$DATA_DIR" --video-path-map "$VIDEO_MAP" --output-dir "$RUN_DIR"
  --frame-size "${FRAME_SIZE:-224}" --threshold "${THRESHOLD:-0.5}" --seed "${SEED:-42}")
case "$BACKEND" in
 mock)
  STAGE2_ADAPTER=${STAGE2_ADAPTER:-$DATA_DIR/mock_stage2_checkpoint}
  [[ -f "$STAGE2_ADAPTER/mock_config.json" ]] || { echo 'Missing explicitly marked mock checkpoint.' >&2; exit 2; }
  common+=(--stage2-adapter "$STAGE2_ADAPTER" --epochs "${EPOCHS:-2}"
    --learning-rate "${LEARNING_RATE:-0.003}" --max-new-tokens "${MAX_NEW_TOKENS:-24}") ;;
 qwen)
  : "${STAGE2_ADAPTER:?Set the real Stage2 adapter directory}"
  : "${STAGE2_TRAINING_CONFIG:?Set its original training_config.json}"
  [[ ! -f "$STAGE2_ADAPTER/mock_config.json" ]] || { echo 'Refusing mock weights for Qwen.' >&2; exit 2; }
  common+=(--stage2-adapter "$STAGE2_ADAPTER" --stage2-training-config "$STAGE2_TRAINING_CONFIG"
    --model-name "${MODEL_NAME:-Qwen/Qwen3.5-2B}" --epochs "${EPOCHS:-20}"
    --learning-rate "${LEARNING_RATE:-0.0001}" --max-new-tokens "${MAX_NEW_TOKENS:-128}")
  [[ -z "${STAGE1_ADAPTER:-}" ]] || common+=(--stage1-adapter "$STAGE1_ADAPTER") ;;
 *) echo 'BACKEND must be mock or qwen' >&2; exit 2 ;;
esac
[[ -z "${MAX_STEPS:-}" ]] || common+=(--max-steps "$MAX_STEPS")
[[ -z "${EVAL_ONLY:-}" ]] || common+=(--eval-only "$EVAL_ONLY")
export PYTHONUNBUFFERED=1 TOKENIZERS_PARALLELISM=false
export ATTN_IMPLEMENTATION=${ATTN_IMPLEMENTATION:-sdpa}
"$PYTHON" -B -m stage3.overfit "${common[@]}"
