#!/bin/bash
#SBATCH --job-name=sr-e2e-3v
#SBATCH --partition=lrz-hgx-h100-94x4,lrz-hgx-a100-80x4,lrz-dgx-a100-80x8
#SBATCH --qos=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --time=02:00:00
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
export TRITON_CACHE_DIR=/dss/mcmlscratch/04/ge75vid2/haoyu/triton_cache
export PYTHONPATH="/dss/mcmlscratch/04/ge75vid2/haoyu/streamready_accel_deps:$DEPS:$RUN_DIR/code:$RUN_DIR/code/StreamReady"
cd "$RUN_DIR/code"
GPU_UUID=$("$PYTHON" -B -m endtoend.gpu_monitor --record "$RUN_DIR/logs/gpu_identity.json")
nvidia-smi --id="$GPU_UUID" --query-gpu=timestamp,uuid,name,memory.used,memory.total,utilization.gpu,utilization.memory,power.draw --format=csv -l 2 > "$RUN_DIR/logs/gpu.csv" &
MONITOR_PID=$!
trap 'kill "$MONITOR_PID" 2>/dev/null || true' EXIT
"$PYTHON" -B -m endtoend.train --run-dir "$RUN_DIR"
