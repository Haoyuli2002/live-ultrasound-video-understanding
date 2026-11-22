#!/usr/bin/env bash
set -euo pipefail
source /dss/dsshome1/04/ge75vid2/miniconda3/etc/profile.d/conda.sh
conda activate ultrasound
cd "$RUN_DIR/code"
unset STAGE1_ADAPTER STAGE2_ADAPTER TRANSFORMERS_CACHE
export HF_HOME=/dss/mcmlscratch/04/ge75vid2/haoyu/hf_cache
export HF_HUB_CACHE="$HF_HOME/hub" ATTN_IMPLEMENTATION=sdpa
export PYTHONUNBUFFERED=1 TOKENIZERS_PARALLELISM=false OMP_NUM_THREADS=2
SHORT_TMP=$(mktemp -d /tmp/s2pipe.XXXXXXXX)
export TMPDIR="$SHORT_TMP" TMP="$SHORT_TMP" TEMP="$SHORT_TMP"
export MPLCONFIGDIR="$SHORT_TMP/matplotlib" XDG_CACHE_HOME="$SHORT_TMP/cache"
nvidia-smi --query-gpu=timestamp,index,uuid,memory.used,utilization.gpu --format=csv -l 30 > "$RUN_DIR/logs/gpu_usage.csv" &
MONITOR_PID=$!
trap 'kill "$MONITOR_PID" 2>/dev/null || true; rm -rf -- "$SHORT_TMP"' EXIT
python -m scripts.gpu_runtime
python -m stage2.train_eval_pipeline --run-dir "$RUN_DIR" --workers 8 --epochs 3 --eval-each-epoch
