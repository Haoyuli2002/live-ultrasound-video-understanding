#!/bin/bash
# MODE=smoke (10 targets / 30 rows) or MODE=full (all held-out targets).
set -euo pipefail
[[ $# == 0 || ( $# == 1 && $1 == --dry-run ) ]] || { echo 'Usage: bash scripts/slurm/submit_stage1_eval.sh [--dry-run]'; exit 2; }
export REPO=${REPO:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}
export DATA=${DATA:-/dss/mcmlscratch/04/ge75vid2/haoyu/live-ultrasound-video-understanding}
: "${OUTPUT_DIR:?Set OUTPUT_DIR to the training run to evaluate}"
export OUTPUT_DIR
MODE=${MODE:-smoke}
case "$MODE" in
  smoke) export LIMIT_TARGETS=${LIMIT_TARGETS:-10}; EVAL_TIME=${EVAL_TIME:-01:00:00}; JUDGE_TIME=${JUDGE_TIME:-01:00:00} ;;
  full) export LIMIT_TARGETS=${LIMIT_TARGETS:-0}; EVAL_TIME=${EVAL_TIME:-24:00:00}; JUDGE_TIME=${JUDGE_TIME:-24:00:00} ;;
  *) echo 'MODE must be smoke or full' >&2; exit 2 ;;
esac
export EVAL_DIR=${EVAL_DIR:-$OUTPUT_DIR/eval_${MODE}_matched}
export TRAINING_CONFIG="$OUTPUT_DIR/training_config.json"
if [[ -n ${EVAL_JSONL:-} && -z ${EVAL_VIDEO_MAP:-} ]]; then
  echo 'Set EVAL_VIDEO_MAP together with an explicit EVAL_JSONL.' >&2; exit 1
fi
export EVAL_JSONL=${EVAL_JSONL:-$OUTPUT_DIR/validation_samples.jsonl}
export JUDGE_MODEL=${JUDGE_MODEL:-Qwen/Qwen3.5-27B}
export VISUAL_CONTROLS=${VISUAL_CONTROLS:-normal}
cd "$REPO"
python3 -m stage1.eval_config --training-config "$TRAINING_CONFIG"
if [[ -z ${EVAL_VIDEO_MAP:-} ]]; then
  EVAL_VIDEO_MAP=$(python3 -c 'import json,sys; print(json.load(open(sys.argv[1]))["video_path_map"])' "$TRAINING_CONFIG")
fi
export EVAL_VIDEO_MAP
[[ -s "$EVAL_JSONL" ]] || {
  echo 'No evaluation samples. For training with validation_fraction=0, set EVAL_JSONL and EVAL_VIDEO_MAP to the independent evaluation split.' >&2; exit 1;
}
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
