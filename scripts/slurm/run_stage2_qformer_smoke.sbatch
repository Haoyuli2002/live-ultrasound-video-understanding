#!/usr/bin/env bash
set -euo pipefail
source "$CONDA_SH"
conda activate "$CONDA_ENV"
cd "$CODE_DIR"
export HF_HOME=${HF_HOME:-/dss/mcmlscratch/04/ge75vid2/haoyu/hf_cache}
export HF_HUB_CACHE="$HF_HOME/hub" ATTN_IMPLEMENTATION=sdpa
unset TRANSFORMERS_CACHE STAGE2_ADAPTER
export PYTHONUNBUFFERED=1 TOKENIZERS_PARALLELISM=false OMP_NUM_THREADS=4
SHORT_TMP=$(mktemp -d /tmp/s2qf.XXXXXXXX)
trap 'rm -rf -- "$SHORT_TMP"' EXIT
export TMPDIR="$SHORT_TMP" TMP="$SHORT_TMP" TEMP="$SHORT_TMP"
python -m scripts.gpu_runtime
python -m stage2.train_overfit_ab --mode independent --memory-architecture qformer \
  --train-jsonl "$RUN_DIR/blocks.jsonl" --video-path-map "$RUN_DIR/video_map.json" \
  --init-adapter "$STAGE1_ADAPTER" --output-dir "$RUN_DIR/training" \
  --model-name Qwen/Qwen3.5-2B --epochs "$EPOCHS" --learning-rate 1e-4 --frame-size 224
for checkpoint in epoch-1 final; do
  python -m stage2.evaluate --eval-jsonl "$RUN_DIR/blocks.jsonl" --video-path-map "$RUN_DIR/video_map.json" \
    --adapter-path "$RUN_DIR/training/$checkpoint" --frame-size 224 \
    --output "$RUN_DIR/reconstruction_$checkpoint.jsonl"
done
python -m stage2.reconstruct --eval-jsonl "$RUN_DIR/blocks.jsonl" --video-path-map "$RUN_DIR/video_map.json" \
  --adapter-path "$RUN_DIR/training/final" --frame-size 224 --max-new-tokens 96 \
  --conditions real zero other_window --save-memories --output-dir "$RUN_DIR/reconstruct_final"
python -m stage2.smoke_report --run-dir "$RUN_DIR"
