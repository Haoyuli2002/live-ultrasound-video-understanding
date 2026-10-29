#!/bin/bash
#SBATCH --job-name=stage2-annotate
#SBATCH --partition=lrz-hgx-h100-94x4
#SBATCH --qos=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=120G
#SBATCH --time=14:00:00
#SBATCH --output=logs/stage2_annotate_%j.out
#SBATCH --error=logs/stage2_annotate_%j.err

# Stage 2 teacher data: serve a local multimodal Qwen3.5 on vLLM, run
# stage2.annotate per video (6x10s local + 60s block + cumulative global
# per minute), then merge everything into chronological trainable blocks with
# stage2.build_data. Mirrors the vLLM lifecycle of
# run_qwen35_video_type_labeling.sbatch and run_stage1_qwen35_restore_punct.sbatch.
#
# Smoke by default (one video). Set VIDEO_IDS="" to annotate every id in VIDEO_MAP.

set -euo pipefail

REPO=${REPO:-/dss/dsshome1/04/ge75vid2/haoyu/live-ultrasound-video-understanding}
DATA=${DATA:-/dss/mcmlscratch/04/ge75vid2/haoyu/live-ultrasound-video-understanding}
SPLIT=${SPLIT:-train_full295}

# video_id -> absolute video path (same map the trainer consumes as --video-path-map).
VIDEO_MAP=${VIDEO_MAP:-$DATA/cluster_data/stage2/${SPLIT}_video_paths.json}
# Space-separated ids to annotate; default is a single-video smoke test.
VIDEO_IDS=${VIDEO_IDS-8V649L5Q368}

TEACHER_DIR=${TEACHER_DIR:-$DATA/cluster_data/stage2/teacher/${SPLIT}}
AUDIT_DIR=${AUDIT_DIR:-$DATA/cluster_data/stage2/teacher_audit/${SPLIT}}
BLOCKS=${BLOCKS:-$DATA/cluster_data/stage2/blocks/${SPLIT}_train.jsonl}
MODEL=${MODEL:-Qwen/Qwen3.5-27B}
VLLM_PORT_REQUESTED=${VLLM_PORT:-auto}
MAX_MODEL_LEN=${MAX_MODEL_LEN:-32768}
GPU_MEMORY_UTILIZATION=${GPU_MEMORY_UTILIZATION:-0.90}
# Teacher sampling is denser than the student input; all windows remain causal.
VIDEO_FPS=${VIDEO_FPS:-2.0}
VIDEO_MAX_FRAMES=${VIDEO_MAX_FRAMES:-120}
# Bound decoding memory even for the 18-minute cumulative prefix. Keep time
# duration and aspect ratio; this does not truncate any summary window.
CLIP_FPS=${CLIP_FPS:-2.0}
CLIP_MAX_SIDE=${CLIP_MAX_SIDE:-1024}
FRAME_SIZE=${FRAME_SIZE:-224}
MAX_TOKENS=${MAX_TOKENS:-1024}
MAX_ATTEMPTS=${MAX_ATTEMPTS:-3}
# Cold-loading a large checkpoint plus warmup can exceed 20 minutes.
VLLM_READY_RETRIES=${VLLM_READY_RETRIES:-240}
VLLM_READY_INTERVAL=${VLLM_READY_INTERVAL:-10}

mkdir -p "$REPO/logs"
cd "$REPO"

source "${CONDA_SH:-$HOME/miniconda3/etc/profile.d/conda.sh}"
conda activate "${CONDA_ENV:-ultrasound}"
export PYTHONUNBUFFERED=1

# Use a fresh short directory even if a previous Stage 1 shell exported TMPDIR.
STAGE2_TMP=$(mktemp -d /tmp/s2.XXXXXXXX)
export TMPDIR="$STAGE2_TMP" TMP="$STAGE2_TMP" TEMP="$STAGE2_TMP"
cleanup() {
  echo "[cleanup] stopping vLLM"
  if [[ -n "${VLLM_PID:-}" ]]; then kill "$VLLM_PID" 2>/dev/null || true; fi
  if [[ -n "${VLLM_PID:-}" ]]; then wait "$VLLM_PID" 2>/dev/null || true; fi
  rm -rf -- "$STAGE2_TMP"
}
trap cleanup EXIT
export VLLM_API_KEY=$(python -c 'import secrets; print(secrets.token_urlsafe(24))')
export STAGE2_SERVED_MODEL="stage2_${SLURM_JOB_ID:-local}_${RANDOM}"
VLLM_PORT=$(python -m stage2.runtime port "$VLLM_PORT_REQUESTED")
CLIP_DIR="$STAGE2_TMP/clips"
export HF_HOME=${HF_HOME:-/dss/mcmlscratch/04/ge75vid2/haoyu/hf_cache}
unset TRANSFORMERS_CACHE
export HF_HUB_CACHE=${HF_HUB_CACHE:-$HF_HOME/hub}
mkdir -p "$HF_HOME" "$HF_HUB_CACHE" "$TEACHER_DIR" "$AUDIT_DIR" "$CLIP_DIR" "$(dirname "$BLOCKS")"

# Avoid FlashInfer JIT paths that require nvcc on LRZ nodes without CUDA toolkit.
export VLLM_ATTENTION_BACKEND=${VLLM_ATTENTION_BACKEND:-FLASH_ATTN}
export VLLM_USE_FLASHINFER=${VLLM_USE_FLASHINFER:-0}
export VLLM_USE_FLASHINFER_SAMPLER=${VLLM_USE_FLASHINFER_SAMPLER:-0}
export VLLM_USE_FLASHINFER_MOE=${VLLM_USE_FLASHINFER_MOE:-0}
export FLASHINFER_DISABLE=${FLASHINFER_DISABLE:-1}

echo "========== JOB INFO =========="
hostname; date; nvidia-smi
echo "REPO=$REPO"
echo "DATA=$DATA"
echo "SPLIT=$SPLIT"
echo "VIDEO_MAP=$VIDEO_MAP"
echo "VIDEO_IDS=${VIDEO_IDS:-<all>}"
echo "TEACHER_DIR=$TEACHER_DIR"
echo "AUDIT_DIR=$AUDIT_DIR"
echo "BLOCKS=$BLOCKS"
echo "CLIP_DIR=$CLIP_DIR"
echo "MODEL=$MODEL"
echo "VIDEO_FPS=$VIDEO_FPS"
echo "MAX_MODEL_LEN=$MAX_MODEL_LEN"
echo "VLLM_PORT=$VLLM_PORT"
echo "VLLM_SERVED_MODEL=$STAGE2_SERVED_MODEL TMPDIR=$TMPDIR"
echo "=============================="

[[ -f "$VIDEO_MAP" ]] || { echo "[error] VIDEO_MAP not found: $VIDEO_MAP" >&2; exit 1; }

# Resolve which ids to annotate (explicit smoke list, or every id in VIDEO_MAP).
if [[ -n "${VIDEO_IDS// }" ]]; then
  IDS=$VIDEO_IDS
else
  IDS=$(python - "$VIDEO_MAP" <<'PY'
import json, sys
print(" ".join(json.load(open(sys.argv[1])).keys()))
PY
  )
fi
echo "[annotate] ids: $IDS"

# Check every selected source before loading 27B weights.
for vid in $IDS; do
  python - "$VIDEO_MAP" "$vid" <<'PY_CHECK'
import json, sys
from pathlib import Path
mapping = json.loads(Path(sys.argv[1]).read_text())
path = mapping.get(sys.argv[2])
if not path or not Path(path).is_file():
    sys.exit(f"Missing video {sys.argv[2]} in {sys.argv[1]}: {path}")
PY_CHECK
done

# Check source identity, sampling policy and prompt fingerprint before loading 27B.
for vid in $IDS; do
  video=$(python -c 'import json,sys; print(json.load(open(sys.argv[1]))[sys.argv[2]])' "$VIDEO_MAP" "$vid")
  python -m stage2.annotate --preflight-only --video "$video" --video-id="$vid" \
    --output "$TEACHER_DIR/$vid.jsonl" --model "$MODEL" --transport video_url \
    --video-fps "$VIDEO_FPS" --video-max-frames "$VIDEO_MAX_FRAMES" \
    --clip-fps "$CLIP_FPS" --clip-max-side "$CLIP_MAX_SIDE" --disable-thinking \
    --max-tokens "$MAX_TOKENS" --resume
done

VLLM_LOG=${VLLM_LOG:-$REPO/logs/stage2_vllm_${SLURM_JOB_ID:-$$}.log}

# Recent vLLM releases require opting in to per-request processor kwargs.
VLLM_EXTRA=()
VLLM_HELP=$(vllm serve --help=all 2>&1)
if [[ "$VLLM_HELP" == *"--trust-request-mm-kwargs"* ]]; then
  VLLM_EXTRA+=(--trust-request-mm-kwargs)
fi
MM_KWARGS='{"do_sample_frames":false}'
echo "[vllm] starting vLLM on port $VLLM_PORT"
echo "[sampling] clip_fps=$CLIP_FPS clip_max_side=$CLIP_MAX_SIDE processor=$MM_KWARGS"
# --allowed-local-media-path lets vLLM read the file:// silent clips that
# stage2.annotate (transport video_url) hands it; --media-io-kwargs keeps all
# already bounded frames, preserving each clip's actual FPS and duration.
nohup vllm serve "$MODEL" \
  --host 127.0.0.1 \
  --port "$VLLM_PORT" \
  --served-model-name "$STAGE2_SERVED_MODEL" \
  --max-model-len "$MAX_MODEL_LEN" \
  --gpu-memory-utilization "$GPU_MEMORY_UTILIZATION" \
  --enforce-eager \
  --max-num-seqs 1 \
  --limit-mm-per-prompt '{"image":0,"video":1}' \
  --mm-processor-kwargs "$MM_KWARGS" \
  "${VLLM_EXTRA[@]}" \
  --allowed-local-media-path "$CLIP_DIR" \
  --media-io-kwargs '{"video":{"num_frames":-1}}' \
  > "$VLLM_LOG" 2>&1 &
VLLM_PID=$!

echo "[vllm] waiting for server"
ready=0
for i in $(seq 1 "$VLLM_READY_RETRIES"); do
  if ! kill -0 "$VLLM_PID" 2>/dev/null; then
    echo "[error] owned vLLM process exited; see $VLLM_LOG" >&2
    tail -100 "$VLLM_LOG" >&2 || true
    exit 1
  fi
  if python -m stage2.runtime ready --base-url "http://127.0.0.1:${VLLM_PORT}/v1" \
      --model "$STAGE2_SERVED_MODEL"; then
    ready=1
    echo "[vllm] ready after $i checks"
    break
  fi
  echo "[vllm] not ready yet ($i/${VLLM_READY_RETRIES})"
  sleep "$VLLM_READY_INTERVAL"
done
if [[ "$ready" != 1 ]]; then
  echo "[error] vLLM not ready after $((VLLM_READY_RETRIES * VLLM_READY_INTERVAL))s; see $VLLM_LOG" >&2
  tail -200 "$VLLM_LOG" >&2 || true
  exit 1
fi

TEACHER_FILES=()
FAILED_IDS=()
for vid in $IDS; do
  video=$(python - "$VIDEO_MAP" "$vid" <<'PY'
import json, sys
m = json.load(open(sys.argv[1]))
path = m.get(sys.argv[2])
if not path:
    sys.exit(f"video_id {sys.argv[2]!r} is not in the video map")
print(path)
PY
  )
  if [[ ! -f "$video" ]]; then
    echo "[error] video file missing for $vid: $video" >&2
    exit 1
  fi
  out="$TEACHER_DIR/${vid}.jsonl"
  echo "[annotate] $vid -> $out  (video=$video)"
  if python -m stage2.annotate \
    --video "$video" \
    --video-id="$vid" \
    --output "$out" \
    --model "$MODEL" \
    --served-model-name "$STAGE2_SERVED_MODEL" \
    --transport-preflight \
    --base-url "http://127.0.0.1:${VLLM_PORT}/v1" \
    --api-key-env VLLM_API_KEY \
    --transport video_url \
    --video-fps "$VIDEO_FPS" \
    --video-max-frames "$VIDEO_MAX_FRAMES" \
    --clip-fps "$CLIP_FPS" \
    --clip-max-side "$CLIP_MAX_SIDE" \
    --disable-thinking \
    --frame-size "$FRAME_SIZE" \
    --max-tokens "$MAX_TOKENS" \
    --max-attempts "$MAX_ATTEMPTS" \
    --audit-output "$AUDIT_DIR/${vid}.jsonl" \
    --clip-dir "$CLIP_DIR" \
    --resume; then
    TEACHER_FILES+=("$out")
  else
    FAILED_IDS+=("$vid")
    echo "[error] annotation failed for $vid; saved complete minutes can be resumed" >&2
    [[ "${CONTINUE_ON_ERROR:-0}" == 1 ]] || exit 1
  fi
done

# A failed full-run shard still processes its remaining videos, then exits nonzero.
if [[ ${#FAILED_IDS[@]} -gt 0 ]]; then
  echo "[error] failed videos: ${FAILED_IDS[*]}; final dataset will not be published" >&2
  exit 1
fi

echo "[build] merging teacher labels into chronological blocks"
python -m stage2.build_data \
  --teacher-jsonl "${TEACHER_FILES[@]}" \
  --output "$BLOCKS"

echo "[verify] reloading blocks with strict chronology validation"
python - "$BLOCKS" <<'PY'
import sys
from stage2.data import load_blocks
videos = load_blocks(sys.argv[1])
total = sum(len(v) for v in videos.values())
print(f"[verify] OK: {len(videos)} videos, {total} blocks in {sys.argv[1]}")
for vid, blocks in list(videos.items())[:3]:
    first = blocks[0]
    print(f"  {vid}: {len(blocks)} blocks; first local windows "
          f"{[s['window'] for s in first['local_sub_summaries']]}")
    print(f"    block_target: {first.get('block_summary_target', '')[:120]}")
    print(f"    global_target: {first.get('global_summary_target', '')[:120]}")
PY

echo "[done] teacher=$TEACHER_DIR  blocks=$BLOCKS  $(date)"
