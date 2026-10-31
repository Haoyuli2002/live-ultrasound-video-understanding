#!/usr/bin/env bash
# Resources and partition selection are supplied by submit_training.sh.
set -euo pipefail
cd "${REPO:?Set REPO}"
source "${CONDA_SH:-$HOME/miniconda3/etc/profile.d/conda.sh}"
conda activate "${CONDA_ENV:-ultrasound}"
export DATA=${DATA:-/dss/mcmlscratch/04/ge75vid2/haoyu/live-ultrasound-video-understanding}
export HF_HOME=${HF_HOME:-/dss/mcmlscratch/04/ge75vid2/haoyu/hf_cache}
unset TRANSFORMERS_CACHE
export HF_HUB_CACHE=${HF_HUB_CACHE:-$HF_HOME/hub}
SHORT_TMP=$(mktemp -d /tmp/ust.XXXXXXXX)
trap 'rm -rf -- "$SHORT_TMP"' EXIT
export TMPDIR="$SHORT_TMP" TMP="$SHORT_TMP" TEMP="$SHORT_TMP"
export PYTHONUNBUFFERED=1 TOKENIZERS_PARALLELISM=false OMP_NUM_THREADS=${OMP_NUM_THREADS:-4}
export ATTN_IMPLEMENTATION=${ATTN_IMPLEMENTATION:-sdpa}
DTYPE=${DTYPE:-bf16}
python -m scripts.gpu_runtime
[[ -f "$TRAIN_JSONL" && -f "$VIDEO_MAP" ]] || { echo 'Missing training data or video map' >&2; exit 1; }
if [[ "$STAGE" == 2 || "$STAGE" == 3 ]]; then
  [[ -f "${STAGE1_ADAPTER:?Set STAGE1_ADAPTER}/adapter_config.json" ]] || {
    echo 'Missing Stage 1 adapter_config.json' >&2; exit 1;
  }
fi
if [[ "$STAGE" == 3 ]]; then
  [[ -f "${STAGE2_ADAPTER:?Set STAGE2_ADAPTER}/adapter_config.json" ]] || {
    echo 'Missing Stage 2 adapter_config.json' >&2; exit 1;
  }
fi
launcher=(python)
if [[ "${NPROC_PER_NODE:-1}" != 1 ]]; then
  [[ "$STAGE" == 1 ]] || exit 2
  launcher=(torchrun --standalone --nnodes=1 --nproc-per-node="$NPROC_PER_NODE")
fi
common=(--train-jsonl "$TRAIN_JSONL" --video-path-map "$VIDEO_MAP" --output-dir "$OUTPUT_DIR"
  --model-name "${MODEL_NAME:-Qwen/Qwen3.5-2B}" --epochs "${EPOCHS:-1}" --dtype "$DTYPE")
case "$STAGE" in
  1)
    extra=(--frame-budget "${FRAME_BUDGET:-30}" --frame-size "${FRAME_SIZE:-224}"
      --frame-sampling "${FRAME_SAMPLING:-recent_sparse}" --recent-seconds "${RECENT_SECONDS:-15}"
      --gradient-accumulation-steps "${GRADIENT_ACCUMULATION_STEPS:-2}"
      --dataloader-num-workers "${DATALOADER_NUM_WORKERS:-2}"
      --max-steps "${MAX_STEPS:--1}" --save-steps "${SAVE_STEPS:-50}")
    [[ "${RESUME:-0}" == 0 ]] || extra+=(--resume)
    "${launcher[@]}" -m stage1.train "${common[@]}" "${extra[@]}" ;;
  2)
    # BF16 is native on both target families. This manual loop has no FP16
    # GradScaler contract, so do not silently advertise FP16 compatibility.
    [[ "$DTYPE" == bf16 || "$DTYPE" == fp32 ]] || { echo 'Stage 2: use bf16 or fp32' >&2; exit 2; }
    python -m stage2.train "${common[@]}" --init-adapter "${STAGE1_ADAPTER:?Set STAGE1_ADAPTER}" ;;
  3)
    python -m stage3.train "${common[@]}" --stage1-adapter "${STAGE1_ADAPTER:?Set STAGE1_ADAPTER}" \
      --stage2-adapter "${STAGE2_ADAPTER:?Set STAGE2_ADAPTER}" --max-steps "${MAX_STEPS:--1}" ;;
  *) echo 'Unknown STAGE' >&2; exit 2 ;;
esac
