#!/usr/bin/env bash
set -euo pipefail
source "$CONDA_SH"
conda activate "$CONDA_ENV"
cd "$CODE_DIR"
unset STAGE1_ADAPTER STAGE2_ADAPTER INIT_ADAPTER
export HF_HOME=${HF_HOME:-/dss/mcmlscratch/04/ge75vid2/haoyu/hf_cache}
export HF_HUB_CACHE="$HF_HOME/hub"
unset TRANSFORMERS_CACHE
export PYTHONUNBUFFERED=1 TOKENIZERS_PARALLELISM=false OMP_NUM_THREADS=4
SHORT_TMP=$(mktemp -d /tmp/s2ab.XXXXXXXX)
trap 'rm -rf -- "$SHORT_TMP"' EXIT
export TMPDIR="$SHORT_TMP" TMP="$SHORT_TMP" TEMP="$SHORT_TMP"
python -m scripts.gpu_runtime
python -m stage2.train_overfit_ab --mode "$MODE" --train-jsonl "$TRAIN_JSONL" \
  --video-path-map "$VIDEO_MAP" --output-dir "$OUTPUT_DIR" \
  --model-name Qwen/Qwen3.5-2B --epochs "$EPOCHS" --learning-rate "$LEARNING_RATE" \
  --seed "$SEED" --frame-size 224
kinds=(short)
if [[ "$MODE" == independent ]]; then kinds+=(block global); fi
for checkpoint in epoch-1 final; do
  python -m stage2.evaluate --eval-jsonl "$TRAIN_JSONL" --video-path-map "$VIDEO_MAP" \
    --adapter-path "$OUTPUT_DIR/$checkpoint" --frame-size 224 --kinds "${kinds[@]}" \
    --output "$RUN_DIR/reconstruction_$checkpoint.jsonl"
done
python -m stage2.reconstruct --eval-jsonl "$TRAIN_JSONL" --video-path-map "$VIDEO_MAP" \
  --adapter-path "$OUTPUT_DIR/final" --frame-size 224 --kinds "${kinds[@]}" \
  --conditions real zero other_window --max-new-tokens "$MAX_NEW_TOKENS" \
  --save-memories --output-dir "$RUN_DIR/reconstruct_final"
python -m stage2.overfit_ab_report --run-dir "$RUN_DIR"
