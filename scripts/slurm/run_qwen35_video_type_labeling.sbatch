#!/bin/bash
#SBATCH --job-name=qwen35-vtype
#SBATCH --partition=lrz-hgx-h100-94x4
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=120G
#SBATCH --time=14:00:00
#SBATCH --output=logs/qwen35_vtype_%j.out
#SBATCH --error=logs/qwen35_vtype_%j.err

set -euo pipefail

REPO=${REPO:-/dss/dsshome1/04/ge75vid2/haoyu/live-ultrasound-video-understanding}
DATA=${DATA:-/dss/mcmlscratch/04/ge75vid2/haoyu/live-ultrasound-video-understanding}

SPLIT=${SPLIT:-eval_full295}
VIDEO_MAP=${VIDEO_MAP:-$DATA/cluster_data/splits/${SPLIT}_asr_keep_videos.json}
OUTPUT=${OUTPUT:-$DATA/cluster_data/splits/${SPLIT}_qwen35_video_type.jsonl}
VIDEO_DIR=${VIDEO_DIR:-$DATA/cluster_data/videos/${SPLIT}}

MODEL=${MODEL:-Qwen/Qwen3.5-35B-A3B}
VIDEO_FPS=${VIDEO_FPS:-0.5}
MAX_TOKENS=${MAX_TOKENS:-6000}
ENABLE_THINKING=${ENABLE_THINKING:-0}
HTTP_PORT=${HTTP_PORT:-9000}
VLLM_PORT=${VLLM_PORT:-8000}
LIMIT=${LIMIT:-}

mkdir -p "$REPO/logs"
cd "$REPO"

source ~/.bashrc || true
conda activate ultrasound

export VLLM_API_KEY=${VLLM_API_KEY:-EMPTY}
export HF_HOME=${HF_HOME:-/dss/mcmlscratch/04/ge75vid2/haoyu/hf_cache}
export TRANSFORMERS_CACHE=${TRANSFORMERS_CACHE:-$HF_HOME}
export HF_HUB_CACHE=${HF_HUB_CACHE:-$HF_HOME/hub}
mkdir -p "$HF_HOME" "$HF_HUB_CACHE"

# Avoid FlashInfer JIT paths that require nvcc on LRZ nodes without CUDA toolkit.
export VLLM_ATTENTION_BACKEND=${VLLM_ATTENTION_BACKEND:-FLASH_ATTN}
export VLLM_USE_FLASHINFER=${VLLM_USE_FLASHINFER:-0}
export VLLM_USE_FLASHINFER_SAMPLER=${VLLM_USE_FLASHINFER_SAMPLER:-0}
export VLLM_USE_FLASHINFER_MOE=${VLLM_USE_FLASHINFER_MOE:-0}
export FLASHINFER_DISABLE=${FLASHINFER_DISABLE:-1}

echo "========== JOB INFO =========="
hostname
date
nvidia-smi
echo "REPO=$REPO"
echo "DATA=$DATA"
echo "SPLIT=$SPLIT"
echo "VIDEO_MAP=$VIDEO_MAP"
echo "OUTPUT=$OUTPUT"
echo "VIDEO_DIR=$VIDEO_DIR"
echo "MODEL=$MODEL"
echo "VIDEO_FPS=$VIDEO_FPS"
echo "MAX_TOKENS=$MAX_TOKENS"
echo "ENABLE_THINKING=$ENABLE_THINKING"
echo "HTTP_PORT=$HTTP_PORT"
echo "VLLM_PORT=$VLLM_PORT"
echo "LIMIT=${LIMIT:-<none>}"
echo "================================"

cleanup() {
  echo "[cleanup] stopping background services"
  if [[ -n "${VLLM_PID:-}" ]]; then kill "$VLLM_PID" 2>/dev/null || true; fi
  if [[ -n "${HTTP_PID:-}" ]]; then kill "$HTTP_PID" 2>/dev/null || true; fi
  pkill -P $$ 2>/dev/null || true
}
trap cleanup EXIT

if [[ ! -f "$VIDEO_MAP" ]]; then
  echo "[error] VIDEO_MAP not found: $VIDEO_MAP" >&2
  exit 1
fi
if [[ ! -d "$VIDEO_DIR" ]]; then
  echo "[error] VIDEO_DIR not found: $VIDEO_DIR" >&2
  exit 1
fi

echo "[http] starting video HTTP server on port $HTTP_PORT"
cd "$VIDEO_DIR"
python -m http.server "$HTTP_PORT" > "/tmp/video_http_${SLURM_JOB_ID}.log" 2>&1 &
HTTP_PID=$!
sleep 3

FIRST_VIDEO=$(python - <<PY
import json
m=json.load(open("$VIDEO_MAP"))
print(next(iter(m.values())))
PY
)
FIRST_BASENAME=$(basename "$FIRST_VIDEO")
echo "[http] testing first video: $FIRST_BASENAME"
curl -I "http://127.0.0.1:${HTTP_PORT}/${FIRST_BASENAME}" || {
  echo "[error] HTTP server cannot serve first video: $FIRST_BASENAME" >&2
  cat "/tmp/video_http_${SLURM_JOB_ID}.log" || true
  exit 1
}

cd "$REPO"

echo "[vllm] starting vLLM on port $VLLM_PORT"
nohup vllm serve "$MODEL" \
  --host 0.0.0.0 \
  --port "$VLLM_PORT" \
  --max-model-len 32768 \
  --gpu-memory-utilization 0.90 \
  --enforce-eager \
  --media-io-kwargs '{"video":{"num_frames":-1}}' \
  > "/tmp/vllm_qwen35_${SLURM_JOB_ID}.log" 2>&1 &
VLLM_PID=$!

echo "[vllm] waiting for server"
READY=0
for i in $(seq 1 120); do
  if curl -s "http://localhost:${VLLM_PORT}/v1/models" \
      -H "Authorization: Bearer ${VLLM_API_KEY}" | grep -q "$MODEL"; then
    READY=1
    echo "[vllm] ready after $i checks"
    break
  fi
  if ! kill -0 "$VLLM_PID" 2>/dev/null; then
    echo "[error] vLLM died" >&2
    tail -200 "/tmp/vllm_qwen35_${SLURM_JOB_ID}.log" || true
    exit 1
  fi
  echo "[vllm] not ready yet ($i/120)"
  sleep 10
done

if [[ "$READY" != "1" ]]; then
  echo "[error] vLLM did not become ready" >&2
  tail -200 "/tmp/vllm_qwen35_${SLURM_JOB_ID}.log" || true
  exit 1
fi

CLASSIFY_ARGS=(
  --video-map "$VIDEO_MAP"
  --output "$OUTPUT"
  --teacher qwen35
  --model "$MODEL"
  --base-url "http://localhost:${VLLM_PORT}/v1"
  --api-key-env VLLM_API_KEY
  --repo-root "$DATA"
  --video-url-base "http://127.0.0.1:${HTTP_PORT}"
  --video-fps "$VIDEO_FPS"
  --max-tokens "$MAX_TOKENS"
  --resume
)
if [[ "$ENABLE_THINKING" == "1" || "$ENABLE_THINKING" == "true" || "$ENABLE_THINKING" == "TRUE" ]]; then
  CLASSIFY_ARGS+=(--enable-thinking)
else
  CLASSIFY_ARGS+=(--disable-thinking)
fi
if [[ -n "$LIMIT" ]]; then
  CLASSIFY_ARGS+=(--limit "$LIMIT")
fi

echo "[label] starting classification"
python scripts/data/classify_ultrasound_video_type_teacher.py "${CLASSIFY_ARGS[@]}"

echo "[label] done: $OUTPUT"
python - <<PY
import json
from collections import Counter
path="$OUTPUT"
rows=[json.loads(l) for l in open(path, encoding="utf-8") if l.strip()]
print("num rows:", len(rows))
print("num errors:", sum(r.get("error") is not None for r in rows))
print("labels:", Counter(r.get("label") for r in rows))
print("languages:", Counter(r.get("spoken_language") for r in rows))
print("anatomy:", Counter(a for r in rows for a in r.get("anatomy_regions", [])))
print("scenarios:", Counter(s for r in rows for s in r.get("clinical_scenarios", [])))
PY

echo "[done] $(date)"