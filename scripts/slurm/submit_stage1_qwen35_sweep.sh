#!/bin/bash
# Matched 2B/4B/9B runs. Invoke from the cluster checkout root.
set -euo pipefail
MODE=${1:-smoke}
case "$MODE" in smoke|full|all) ;; *) echo 'Usage: bash scripts/slurm/submit_stage1_qwen35_sweep.sh [smoke|full|all]' >&2; exit 2;; esac
REPO=${REPO:-$(pwd)}
DATA=${DATA:-/dss/mcmlscratch/04/ge75vid2/haoyu/live-ultrasound-video-understanding}
SWEEP_NAME=${SWEEP_NAME:-stage1_qwen35_comparison}
[[ "$SWEEP_NAME" =~ ^[a-zA-Z0-9_-]+$ ]] || { echo 'Invalid SWEEP_NAME' >&2; exit 2; }
cd "$REPO"
[[ -f scripts/slurm/run_stage1_pretrain_merged.sbatch ]]
LOG_DIR=${LOG_DIR:-$DATA/cluster_data/logs/$SWEEP_NAME}
if [[ ${DRY_RUN:-0} != 1 ]]; then
  mkdir -p "$LOG_DIR"
fi
submit() {
  local log_options=(--output="$LOG_DIR/%x_%j.out" --error="$LOG_DIR/%x_%j.err")
  if [[ ${DRY_RUN:-0} == 1 ]]; then
    printf 'sbatch ' >&2; printf '%q ' "${log_options[@]}" "$@" >&2; printf '\n' >&2
    echo 12345
  else
    sbatch --parsable "${log_options[@]}" "$@" | cut -d';' -f1
  fi
}
for size in 2 4 9; do
  export MODEL_NAME="Qwen/Qwen3.5-${size}B" REPO DATA
  export FRAME_BUDGET=${FRAME_BUDGET:-120} FRAME_SIZE=${FRAME_SIZE:-224}
  export FRAME_SAMPLING=${FRAME_SAMPLING:-uniform} VALIDATION_FRACTION=${VALIDATION_FRACTION:-0.1}
  dependency=()
  if [[ "$MODE" == smoke || "$MODE" == all ]]; then
    export RUN_NAME="${SWEEP_NAME}_${size}b_smoke"
    export OUTPUT_DIR="$DATA/cluster_data/experiments/stage1/$RUN_NAME"
    export MAX_STEPS=2 SAVE_STEPS=1 LOGGING_STEPS=1
    smoke=$(submit --export=ALL --job-name="q35-${size}b-smoke" scripts/slurm/run_stage1_pretrain_merged.sbatch)
    echo "$MODEL_NAME smoke job: $smoke"
    dependency=(--dependency="afterok:$smoke" --kill-on-invalid-dep=yes)
  fi
  if [[ "$MODE" == full || "$MODE" == all ]]; then
    export RUN_NAME="${SWEEP_NAME}_${size}b_full"
    export OUTPUT_DIR="$DATA/cluster_data/experiments/stage1/$RUN_NAME"
    export MAX_STEPS=-1 SAVE_STEPS=50 LOGGING_STEPS=10 EPOCHS=${EPOCHS:-3}
    train=$(submit --export=ALL --job-name="q35-${size}b-train" "${dependency[@]}" scripts/slurm/run_stage1_pretrain_merged.sbatch)
    evaluate=$(submit --export=ALL --job-name="q35-${size}b-eval" --dependency="afterok:$train" --kill-on-invalid-dep=yes scripts/slurm/run_stage1_qwen35_eval.sbatch)
    echo "$MODEL_NAME train job: $train; evaluation job: $evaluate"
  fi
done
