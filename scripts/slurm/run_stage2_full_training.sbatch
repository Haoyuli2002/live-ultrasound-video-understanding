#!/usr/bin/env bash
set -euo pipefail
source "$CONDA_SH"
conda activate "$CONDA_ENV"
cd "$CODE_DIR"
export HF_HOME=${HF_HOME:-/dss/mcmlscratch/04/ge75vid2/haoyu/hf_cache}
export HF_HUB_CACHE="$HF_HOME/hub" ATTN_IMPLEMENTATION=sdpa
unset TRANSFORMERS_CACHE STAGE2_ADAPTER
export PYTHONUNBUFFERED=1 TOKENIZERS_PARALLELISM=false OMP_NUM_THREADS=2
SHORT_TMP=$(mktemp -d /tmp/s2full.XXXXXXXX)
trap 'rm -rf -- "$SHORT_TMP"' EXIT
export TMPDIR="$SHORT_TMP" TMP="$SHORT_TMP" TEMP="$SHORT_TMP"
python -m scripts.gpu_runtime
torchrun --standalone --nnodes=1 --nproc-per-node="$NPROC_PER_NODE" -m stage2.train_full \
  --architecture "$ARCHITECTURE" --dataset-version "$DATASET_VERSION" --train-jsonl "$DATASET_RELEASE/train.jsonl" \
  --video-path-map "$RUN_DIR/video_map.json" --release-manifest "$DATASET_RELEASE/manifest.json" \
  --init-adapter "$STAGE1_ADAPTER" --output-dir "$RUN_DIR/training" --epochs "$EPOCHS" \
  --learning-rate "$LEARNING_RATE" --frame-size 224 --save-steps "$SAVE_STEPS" --resume
