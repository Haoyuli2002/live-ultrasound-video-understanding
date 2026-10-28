#!/bin/bash
# MODE=smoke (10 targets / 30 rows) or MODE=full (all held-out targets).
set -euo pipefail
[[ $# == 0 || ( $# == 1 && $1 == --dry-run ) ]] || { echo 'Usage: bash scripts/slurm/submit_stage1_eval.sh [--dry-run]'; exit 2; }
export REPO=${REPO:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}
export DATA=${DATA:-/dss/mcmlscratch/04/ge75vid2/haoyu/live-ultrasound-video-understanding}
export OUTPUT_DIR=${OUTPUT_DIR:-$DATA/cluster_data/experiments/stage1/stage1_full186_2b_4gpu_f30_e1_20261025_052451}
MODE=${MODE:-smoke}
case "$MODE" in
  smoke) export LIMIT_TARGETS=${LIMIT_TARGETS:-10}; EVAL_TIME=${EVAL_TIME:-01:00:00}; JUDGE_TIME=${JUDGE_TIME:-01:00:00} ;;
  full) export LIMIT_TARGETS=${LIMIT_TARGETS:-0}; EVAL_TIME=${EVAL_TIME:-24:00:00}; JUDGE_TIME=${JUDGE_TIME:-24:00:00} ;;
  *) echo 'MODE must be smoke or full' >&2; exit 2 ;;
esac
export EVAL_DIR=${EVAL_DIR:-$OUTPUT_DIR/eval_${MODE}_f30}
export EVAL_JSONL=${EVAL_JSONL:-$OUTPUT_DIR/validation_samples.jsonl}
export EVAL_VIDEO_MAP=${EVAL_VIDEO_MAP:-$DATA/cluster_data/splits/train_full295_qwen35_pretrain_keep_videos.json}
export FRAME_BUDGET=${FRAME_BUDGET:-30} FRAME_SIZE=${FRAME_SIZE:-224} FRAME_SAMPLING=${FRAME_SAMPLING:-uniform}
export MODEL_NAME=${MODEL_NAME:-Qwen/Qwen3.5-2B} JUDGE_MODEL=${JUDGE_MODEL:-Qwen/Qwen3.5-27B}
export VISUAL_CONTROLS=${VISUAL_CONTROLS:-normal}
cd "$REPO"
[[ -s "$OUTPUT_DIR/final/adapter_model.safetensors" && -s "$OUTPUT_DIR/final/adapter_config.json" ]]
[[ -s "$EVAL_JSONL" && -s "$OUTPUT_DIR/video_split.json" && -s "$EVAL_VIDEO_MAP" ]]
for control in $VISUAL_CONTROLS; do
  case "$control" in normal|blank|shuffled) ;; *) echo "Invalid control: $control"; exit 2;; esac
done
mkdir -p logs "$EVAL_DIR"
python3 -m stage1.prepare_eval --input "$EVAL_JSONL" --split "$OUTPUT_DIR/video_split.json" \
  --output "$EVAL_DIR/samples.jsonl" --limit-targets "$LIMIT_TARGETS"
printf 'Mode: %s\nEval directory: %s\nEval time: %s; judge time: %s\n' "$MODE" "$EVAL_DIR" "$EVAL_TIME" "$JUDGE_TIME"
[[ ${1:-} != --dry-run ]] || { echo 'Dry run complete; no job submitted.'; exit 0; }
eval_job=$(sbatch --parsable --export=ALL --time="$EVAL_TIME" scripts/slurm/run_stage1_eval_suite.sbatch)
eval_job=${eval_job%%;*}
printf '%s\n' "$eval_job" > "$EVAL_DIR/eval_job_id.txt"
echo "Evaluation job: $eval_job"
if [[ ${SUBMIT_JUDGE:-1} == 1 ]]; then
  judge_job=$(sbatch --parsable --export=ALL --dependency="afterok:$eval_job" --time="$JUDGE_TIME" scripts/slurm/run_stage1_judge.sbatch)
  judge_job=${judge_job%%;*}
  printf '%s\n' "$judge_job" > "$EVAL_DIR/judge_job_id.txt"
  echo "Judge job: $judge_job (after successful evaluation)"
fi
