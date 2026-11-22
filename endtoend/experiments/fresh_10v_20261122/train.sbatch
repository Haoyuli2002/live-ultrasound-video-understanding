#!/bin/bash
#SBATCH --job-name=sr-fresh-bce1
#SBATCH --partition=lrz-hgx-a100-80x4,lrz-dgx-a100-80x8
#SBATCH --qos=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --time=01:00:00
set -euo pipefail
: "${RUN_DIR:?Set RUN_DIR to the frozen experiment}"
PYTHON=/dss/dsshome1/04/ge75vid2/miniconda3/envs/ultrasound/bin/python
DEPS=/dss/dsshome1/04/ge75vid2/haoyu/live-ultrasound-video-understanding/results/streamready_e2e_setup/deps
export HF_HOME=/dss/mcmlscratch/04/ge75vid2/haoyu/hf_cache
export HUGGINGFACE_HUB_CACHE="$HF_HOME/hub"
export TORCH_HOME=/dss/mcmlscratch/04/ge75vid2/torch_cache
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 PYTHONDONTWRITEBYTECODE=1
export TOKENIZERS_PARALLELISM=false OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 PYTHONUNBUFFERED=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export TILELANG_CACHE_DIR=/dss/mcmlscratch/04/ge75vid2/haoyu/streamready_compiler_cache/tilelang
export TILELANG_TMP_DIR=/dss/mcmlscratch/04/ge75vid2/haoyu/streamready_compiler_cache/tilelang_tmp
export TORCH_EXTENSIONS_DIR=/dss/mcmlscratch/04/ge75vid2/haoyu/streamready_compiler_cache/torch_extensions
export TORCHINDUCTOR_CACHE_DIR=/dss/mcmlscratch/04/ge75vid2/haoyu/streamready_compiler_cache/torchinductor
export CUDA_CACHE_PATH=/dss/mcmlscratch/04/ge75vid2/haoyu/streamready_compiler_cache/cuda
export XDG_CACHE_HOME=/dss/mcmlscratch/04/ge75vid2/haoyu/streamready_compiler_cache/xdg
export TRITON_CACHE_DIR=/dss/mcmlscratch/04/ge75vid2/haoyu/triton_cache
export PYTHONPATH="/dss/mcmlscratch/04/ge75vid2/haoyu/streamready_accel_deps:$DEPS:$RUN_DIR/code:$RUN_DIR/code/StreamReady"
cd "$RUN_DIR/code"
"$PYTHON" -B - <<'CACHECHECK' > "$RUN_DIR/logs/compiler_cache.json"
import json, os, tempfile
from tilelang.env import env
paths = {key: os.environ[key] for key in ['TILELANG_CACHE_DIR','TILELANG_TMP_DIR','TORCH_EXTENSIONS_DIR','TORCHINDUCTOR_CACHE_DIR','CUDA_CACHE_PATH','XDG_CACHE_HOME','TRITON_CACHE_DIR']}
assert env.TILELANG_CACHE_DIR == paths['TILELANG_CACHE_DIR']
assert env.TILELANG_TMP_DIR == paths['TILELANG_TMP_DIR']
for path in paths.values():
    os.makedirs(path, exist_ok=True)
    with tempfile.TemporaryFile(dir=path) as f:
        f.write(b'cache write check'); f.flush()
print(json.dumps(paths, indent=2))
CACHECHECK
GPU_UUID=$("$PYTHON" -B -m endtoend.gpu_monitor --record "$RUN_DIR/logs/gpu_identity.json")
nvidia-smi --id="$GPU_UUID" --query-gpu=timestamp,uuid,name,memory.used,memory.total,utilization.gpu,utilization.memory,power.draw --format=csv -lms 200 > "$RUN_DIR/logs/gpu.csv" &
MONITOR_PID=$!
finish() {
    rc=$?
    trap - EXIT
    kill "$MONITOR_PID" 2>/dev/null || true
    wait "$MONITOR_PID" 2>/dev/null || true
    "$PYTHON" -B -m endtoend.gpu_stats --run-dir "$RUN_DIR" || true
    exit "$rc"
}
trap finish EXIT
"$PYTHON" -B -m endtoend.train --run-dir "$RUN_DIR"

"$PYTHON" -B -m endtoend.answer_metrics --predictions "$RUN_DIR/eval_epoch_100.json" --output "$RUN_DIR/proready_answer_eval_epoch100"
"$PYTHON" -B -m endtoend.scan_readiness_threshold --predictions "$RUN_DIR/eval_epoch_100.json" --output "$RUN_DIR/threshold_scan_epoch100"
