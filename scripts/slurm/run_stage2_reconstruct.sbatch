#!/usr/bin/env bash
set -euo pipefail
source "$CONDA_SH"
conda activate "$CONDA_ENV"
cd "$CODE_DIR"
unset STAGE1_ADAPTER STAGE2_ADAPTER
export HF_HOME=${HF_HOME:-/dss/mcmlscratch/04/ge75vid2/haoyu/hf_cache}
export HF_HUB_CACHE="$HF_HOME/hub"
unset TRANSFORMERS_CACHE
export PYTHONUNBUFFERED=1 TOKENIZERS_PARALLELISM=false OMP_NUM_THREADS=4
export ATTN_IMPLEMENTATION=sdpa
SHORT_TMP=$(mktemp -d /tmp/s2rc.XXXXXXXX)
trap 'rm -rf -- "$SHORT_TMP"' EXIT
export TMPDIR="$SHORT_TMP" TMP="$SHORT_TMP" TEMP="$SHORT_TMP"
python -m scripts.gpu_runtime
MODEL_NAME=$(python -c 'import json,os; from pathlib import Path; print(json.loads((Path(os.environ["TRAIN_RUN"])/"training/training_config.json").read_text())["model_name"])')
FRAME_SIZE=$(python -c 'import json,os; from pathlib import Path; print(json.loads((Path(os.environ["TRAIN_RUN"])/"training/training_config.json").read_text())["frame_size"])')
python -m stage2.reconstruct \
  --eval-jsonl "$TRAIN_RUN/blocks.jsonl" --video-path-map "$TRAIN_RUN/video_map.json" \
  --adapter-path "$TRAIN_RUN/training/$CHECKPOINT" --model-name "$MODEL_NAME" \
  --frame-size "$FRAME_SIZE" --max-new-tokens "$MAX_NEW_TOKENS" \
  --conditions real zero other_window --save-memories --output-dir "$RECON_RUN/results"
