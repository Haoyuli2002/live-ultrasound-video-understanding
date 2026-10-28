#!/bin/bash
#SBATCH --job-name=stage1-judge
#SBATCH --partition=lrz-hgx-h100-94x4
#SBATCH --qos=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=120G
#SBATCH --time=24:00:00
#SBATCH --output=logs/stage1_judge_%j.out
#SBATCH --error=logs/stage1_judge_%j.err
set -euo pipefail
: "${REPO:?}" "${DATA:?}" "${EVAL_DIR:?}"
cd "$REPO"
source "${CONDA_SH:-/dss/dsshome1/04/ge75vid2/miniconda3/etc/profile.d/conda.sh}"
conda activate "${CONDA_ENV:-ultrasound}"
source scripts/slurm/stage1_storage.sh
export PYTHONUNBUFFERED=1
MODEL=${JUDGE_MODEL:-Qwen/Qwen3.5-27B}
PORT=${JUDGE_PORT:-8000}
export VLLM_API_KEY=${JUDGE_API_KEY:-EMPTY}
export VLLM_ATTENTION_BACKEND=FLASH_ATTN
export VLLM_USE_FLASHINFER=0 VLLM_USE_FLASHINFER_SAMPLER=0 VLLM_USE_FLASHINFER_MOE=0 FLASHINFER_DISABLE=1
for control in ${VISUAL_CONTROLS:-normal}; do
  [[ -s "$EVAL_DIR/base_${control}.jsonl" && -s "$EVAL_DIR/lora_${control}.jsonl" ]]
done
# Refuse to accidentally query an unrelated server already listening on this port.
if curl -s --max-time 2 "http://127.0.0.1:$PORT/v1/models" >/dev/null; then
  echo 'Judge port already in use; choose JUDGE_PORT' >&2; exit 1
fi
vllm serve "$MODEL" --host 127.0.0.1 --port "$PORT" --max-model-len 8192 \
  --gpu-memory-utilization 0.90 --enforce-eager --max-num-seqs 1 \
  --limit-mm-per-prompt '{"image":0,"video":0}' >"logs/stage1_judge_vllm_${SLURM_JOB_ID}.log" 2>&1 &
server=$!
trap 'kill "$server" 2>/dev/null || true; wait "$server" 2>/dev/null || true' EXIT
ready=0
for ((i=0; i<180; i++)); do
  kill -0 "$server" || { echo 'vLLM exited; inspect server log' >&2; exit 1; }
  if curl -sf --max-time 5 -H "Authorization: Bearer $VLLM_API_KEY" "http://127.0.0.1:$PORT/v1/models" | grep -Fq "$MODEL"; then
    ready=1; break
  fi
  sleep 10
done
[[ $ready == 1 ]] || { echo 'vLLM startup timeout' >&2; exit 1; }
for control in ${VISUAL_CONTROLS:-normal}; do
  python -m stage1.judge_predictions --model "$MODEL" --base-url "http://127.0.0.1:$PORT/v1" \
    --predictions "$EVAL_DIR/lora_${control}.jsonl" --baseline "$EVAL_DIR/base_${control}.jsonl" \
    --output "$EVAL_DIR/judge_${control}.jsonl"
done
