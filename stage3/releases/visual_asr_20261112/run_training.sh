#!/usr/bin/env bash
set -euo pipefail
PACKAGE_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
: "${PROJECT_ROOT:?Set PROJECT_ROOT to the ultrasound project root}"
: "${STAGE2_ADAPTER:?Set STAGE2_ADAPTER to the real Stage2 final adapter directory}"
: "${OUTPUT_DIR:?Set OUTPUT_DIR to a new, empty Stage3 run directory}"
PYTHON_BIN="${PYTHON_BIN:-python3}"
: "${VIDEO_PATH_MAP:?Generate a machine-specific map with code/make_video_map.py and set VIDEO_PATH_MAP}"
"$PYTHON_BIN" "$PACKAGE_DIR/code/validate.py" --project-root "$PROJECT_ROOT" --video-path-map "$VIDEO_PATH_MAP"
cd "$PROJECT_ROOT"
args=(--train-jsonl "$PACKAGE_DIR/data/memory_candidates/train.jsonl"
      --video-path-map "$VIDEO_PATH_MAP"
      --stage2-adapter "$STAGE2_ADAPTER"
      --output-dir "$OUTPUT_DIR")
if [[ -n "${STAGE1_ADAPTER:-}" ]]; then
  args+=(--stage1-adapter "$STAGE1_ADAPTER")
fi
"$PYTHON_BIN" -m stage3.train "${args[@]}" "$@"
