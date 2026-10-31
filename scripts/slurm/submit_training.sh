#!/usr/bin/env bash
# GPU_TYPE=auto|a100|h100; dry-run prints the exact request without touching Slurm.
set -euo pipefail
STAGE=${1:-}
[[ "$STAGE" =~ ^[123]$ ]] || { echo 'Usage: submit_training.sh <1|2|3> [--dry-run]' >&2; exit 2; }
shift
[[ $# == 0 || ( $# == 1 && $1 == --dry-run ) ]] || exit 2
export STAGE
export REPO=${REPO:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}
export GPU_TYPE=${GPU_TYPE:-auto}
export NPROC_PER_NODE=${NPROC_PER_NODE:-1}
[[ "$NPROC_PER_NODE" =~ ^[1-8]$ ]] || { echo 'NPROC_PER_NODE must be 1..8' >&2; exit 2; }
# This entry supports Stage 1 DDP and single-process Stage 2/3 training.
if [[ "$STAGE" != 1 && "$NPROC_PER_NODE" != 1 ]]; then
  echo 'This Stage 2/3 training entry requires NPROC_PER_NODE=1.' >&2
  exit 2
fi
[[ -n "${TRAIN_JSONL:-}" && -n "${VIDEO_MAP:-}" && -n "${OUTPUT_DIR:-}" ]] || {
  echo 'Set TRAIN_JSONL, VIDEO_MAP and OUTPUT_DIR' >&2; exit 2;
}
[[ "$STAGE" != 2 || -n "${STAGE1_ADAPTER:-}" ]] || { echo 'Set STAGE1_ADAPTER' >&2; exit 2; }
[[ "$STAGE" != 3 || ( -n "${STAGE1_ADAPTER:-}" && -n "${STAGE2_ADAPTER:-}" ) ]] || {
  echo 'Set STAGE1_ADAPTER and STAGE2_ADAPTER' >&2; exit 2;
}
cd "$REPO"
GPU_PARTITIONS=$(python3 -m scripts.gpu_profiles "$GPU_TYPE")
mkdir -p logs
command=(sbatch --parsable --export=ALL --job-name="stage${STAGE}-train"
  --partition="$GPU_PARTITIONS" --qos=gpu --nodes=1 --ntasks=1
  --gres="gpu:$NPROC_PER_NODE" --cpus-per-task="${CPUS_PER_TASK:-8}"
  --mem="${MEMORY:-120G}" --time="${WALLTIME:-24:00:00}" --chdir="$REPO"
  --output="$REPO/logs/stage${STAGE}_train_%j.out" --error="$REPO/logs/stage${STAGE}_train_%j.err"
  "$REPO/scripts/slurm/run_training.sbatch")
if [[ ${1:-} == --dry-run ]]; then printf '%q ' "${command[@]}"; printf '\n'; exit 0; fi
"${command[@]}"
