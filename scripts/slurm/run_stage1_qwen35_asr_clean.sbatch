#!/bin/bash
#SBATCH --job-name=stage1-asr-clean
#SBATCH --partition=lrz-hgx-h100-94x4
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=120G
#SBATCH --time=14:00:00
#SBATCH --output=logs/stage1_asr_clean_%j.out
#SBATCH --error=logs/stage1_asr_clean_%j.err

set -euo pipefail

REPO=${REPO:-/dss/dsshome1/04/ge75vid2/haoyu/live-ultrasound-video-understanding}
DATA=${DATA:-/dss/mcmlscratch/04/ge75vid2/haoyu/live-ultrasound-video-understanding}
SPLIT=${SPLIT:-train_full295}
MODEL=${MODEL:-Qwen/Qwen3.5-27B}
VLLM_PORT=${VLLM_PORT:-8000}
MAX_MODEL_LEN=${MAX_MODEL_LEN:-8192}
GPU_MEMORY_UTILIZATION=${GPU_MEMORY_UTILIZATION:-0.90}
BATCH_SEGMENTS=${BATCH_SEGMENTS:-8}
CONTEXT_SEGMENTS=${CONTEXT_SEGMENTS:-2}
MAX_TOKENS=${MAX_TOKENS:-4096}
VIDEO_MAP=${VIDEO_MAP:-$DATA/cluster_data/splits/${SPLIT}_qwen35_pretrain_keep_videos.json}
TRANSCRIPTS=${TRANSCRIPTS:-$DATA/cluster_data/QA/${SPLIT}/transcripts}
OUTPUT_DIR=${OUTPUT_DIR:-$DATA/cluster_data/QA/${SPLIT}/transcripts_stage1_qwen35_clean}
AUDIT_OUTPUT=${AUDIT_OUTPUT:-$DATA/cluster_data/QA/${SPLIT}/stage1_qwen35_clean_audit.jsonl}
SAMPLES_OUTPUT=${SAMPLES_OUTPUT:-$DATA/cluster_data/pretrain/${SPLIT}_stage1_qwen35_clean_samples.jsonl}

cd "$REPO"
if [[ -f /dss/dsshome1/04/ge75vid2/miniconda3/etc/profile.d/conda.sh ]]; then
  source /dss/dsshome1/04/ge75vid2/miniconda3/etc/profile.d/conda.sh
else
  source "$HOME/miniconda3/etc/profile.d/conda.sh"
fi
conda activate ultrasound

export VLLM_API_KEY=${VLLM_API_KEY:-EMPTY}
export HF_HOME=${HF_HOME:-/dss/mcmlscratch/04/ge75vid2/haoyu/hf_cache}
export HF_HUB_CACHE=${HF_HUB_CACHE:-$HF_HOME/hub}
mkdir -p "$HF_HOME" "$HF_HUB_CACHE" "$(dirname "$AUDIT_OUTPUT")" "$(dirname "$SAMPLES_OUTPUT")"

# Same H100/vLLM settings used by the existing video-type labeling job.
export VLLM_ATTENTION_BACKEND=${VLLM_ATTENTION_BACKEND:-FLASH_ATTN}
export VLLM_USE_FLASHINFER=${VLLM_USE_FLASHINFER:-0}
export VLLM_USE_FLASHINFER_SAMPLER=${VLLM_USE_FLASHINFER_SAMPLER:-0}
export VLLM_USE_FLASHINFER_MOE=${VLLM_USE_FLASHINFER_MOE:-0}
export FLASHINFER_DISABLE=${FLASHINFER_DISABLE:-1}

[[ -f "$VIDEO_MAP" ]] || { echo "Missing video map: $VIDEO_MAP" >&2; exit 1; }
[[ -d "$TRANSCRIPTS" ]] || { echo "Missing transcripts: $TRANSCRIPTS" >&2; exit 1; }

echo "Host: $(hostname)  Job: ${SLURM_JOB_ID:-unknown}  Split: $SPLIT"
echo "Model: $MODEL  Map: $VIDEO_MAP  Transcripts: $TRANSCRIPTS"
nvidia-smi

VLLM_LOG="${SLURM_TMPDIR:-/tmp}/stage1_asr_vllm_${SLURM_JOB_ID:-$$}.log"
cleanup() {
  if [[ -n "${VLLM_PID:-}" ]]; then kill "$VLLM_PID" 2>/dev/null || true; fi
}
trap cleanup EXIT

vllm serve "$MODEL" \
  --host 127.0.0.1 \
  --port "$VLLM_PORT" \
  --max-model-len "$MAX_MODEL_LEN" \
  --gpu-memory-utilization "$GPU_MEMORY_UTILIZATION" \
  --enforce-eager \
  > "$VLLM_LOG" 2>&1 &
VLLM_PID=$!

ready=0
for i in $(seq 1 120); do
  if curl --silent --fail "http://127.0.0.1:${VLLM_PORT}/v1/models" \
      -H "Authorization: Bearer ${VLLM_API_KEY}" | grep -Fq "$MODEL"; then
    ready=1
    break
  fi
  if ! kill -0 "$VLLM_PID" 2>/dev/null; then
    echo "vLLM exited; see $VLLM_LOG" >&2
    tail -100 "$VLLM_LOG" >&2 || true
    exit 1
  fi
  sleep 10
done
if [[ "$ready" != 1 ]]; then
  echo "vLLM was not ready after 20 minutes; see $VLLM_LOG" >&2
  tail -100 "$VLLM_LOG" >&2 || true
  exit 1
fi

python -m stage1.clean_asr \
  --transcripts "$TRANSCRIPTS" \
  --video-map "$VIDEO_MAP" \
  --output-dir "$OUTPUT_DIR" \
  --audit-output "$AUDIT_OUTPUT" \
  --model "$MODEL" \
  --base-url "http://127.0.0.1:${VLLM_PORT}/v1" \
  --api-key-env VLLM_API_KEY \
  --batch-segments "$BATCH_SEGMENTS" \
  --context-segments "$CONTEXT_SEGMENTS" \
  --max-tokens "$MAX_TOKENS" \
  --resume

python -m stage1.data \
  --transcripts "$OUTPUT_DIR" \
  --video-map "$VIDEO_MAP" \
  --output "$SAMPLES_OUTPUT"

echo "Cleaned transcripts: $OUTPUT_DIR"
echo "Audit: $AUDIT_OUTPUT"
echo "Stage 1 samples: $SAMPLES_OUTPUT"
