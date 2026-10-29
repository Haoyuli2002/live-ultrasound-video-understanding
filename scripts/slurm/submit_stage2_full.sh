#!/usr/bin/env bash
# Freeze the plan once, then submit independent one-H100 shards and strict merge.
set -euo pipefail
DRY_RUN=0
RESUME=0
for arg in "$@"; do
  case "$arg" in
    --dry-run) DRY_RUN=1 ;;
    --resume) RESUME=1 ;;
    *) echo 'Usage: submit_stage2_full.sh [--dry-run] [--resume]' >&2; exit 2 ;;
  esac
done
export REPO=${REPO:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}
export DATA=${DATA:-/dss/mcmlscratch/04/ge75vid2/haoyu/live-ultrasound-video-understanding}
cd "$REPO"
source "${CONDA_SH:-$HOME/miniconda3/etc/profile.d/conda.sh}"
conda activate "${CONDA_ENV:-ultrasound}"
export PYTHONUNBUFFERED=1
export STAGE2_TRAIN_MAP=${STAGE2_TRAIN_MAP:-$DATA/cluster_data/splits/train_full295_qwen35_pretrain_keep_videos.json}
export STAGE2_EVAL_MAP=${STAGE2_EVAL_MAP:-$DATA/cluster_data/splits/eval_full295_asr_keep_videos.json}
export STAGE2_SHARDS=${STAGE2_SHARDS:-8}
export STAGE2_CONCURRENCY=${STAGE2_CONCURRENCY:-4}
export STAGE2_WALLTIME=${STAGE2_WALLTIME:-24:00:00}
if [[ -z "${STAGE2_RUN_DIR:-}" ]]; then
  [[ "$RESUME" == 0 ]] || { echo 'Set STAGE2_RUN_DIR to resume' >&2; exit 2; }
  mkdir -p "$DATA/cluster_data/stage2/runs"
  STAGE2_RUN_DIR=$(mktemp -d "$DATA/cluster_data/stage2/runs/train_full_27b_$(date -u +%Y%m%dT%H%M%SZ)_XXXXXX")
fi
mkdir -p "$STAGE2_RUN_DIR"
export STAGE2_RUN_DIR=$(cd "$STAGE2_RUN_DIR" && pwd)
if [[ "$RESUME" == 0 ]]; then
  SEED_ARGS=()
  if [[ -n "${STAGE2_SEED_TEACHER_DIR:-}" ]]; then
    SEED_ARGS+=(--seed-teacher-dir "$STAGE2_SEED_TEACHER_DIR")
  fi
  if [[ -n "${STAGE2_SEED_JOB:-}" ]]; then
    SEED_ARGS+=(--seed-dependency "$STAGE2_SEED_JOB")
  fi
  python -m stage2.full_run prepare --video-map "$STAGE2_TRAIN_MAP" \
    --eval-map "$STAGE2_EVAL_MAP" --video-root "$DATA" \
    --run-dir "$STAGE2_RUN_DIR" --shards "$STAGE2_SHARDS" "${SEED_ARGS[@]}"
else
  [[ -f "$STAGE2_RUN_DIR/manifest.json" ]] || { echo 'Missing frozen manifest' >&2; exit 2; }
fi
{
  printf '# Use the original code snapshot and frozen manifest to resume.\n'
  for name in REPO DATA STAGE2_RUN_DIR STAGE2_TRAIN_MAP STAGE2_EVAL_MAP STAGE2_SHARDS STAGE2_CONCURRENCY STAGE2_WALLTIME; do
    printf 'export %s=%q\n' "$name" "${!name}"
  done
} > "$STAGE2_RUN_DIR/submit_env.sh"
SUBMIT_ARGS=()
[[ "$DRY_RUN" == 0 ]] || SUBMIT_ARGS+=(--dry-run)
python -m stage2.full_run submit --repo "$REPO" --run-dir "$STAGE2_RUN_DIR" \
  --concurrency "$STAGE2_CONCURRENCY" --walltime "$STAGE2_WALLTIME" "${SUBMIT_ARGS[@]}"
echo "Run: $STAGE2_RUN_DIR"
echo "Final data: $STAGE2_RUN_DIR/blocks.jsonl (published only after all videos pass)"
