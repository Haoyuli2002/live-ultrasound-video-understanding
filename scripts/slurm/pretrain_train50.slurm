#!/bin/bash
#SBATCH --job-name=pretrain_train50
#SBATCH --partition=lrz-hgx-h100-94x4
#SBATCH --qos=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=12:00:00
#SBATCH --output=/dss/dsshome1/04/ge75vid2/haoyu/live-ultrasound-video-understanding/cluster_data/logs/pretrain_train50_%j.out
#SBATCH --error=/dss/dsshome1/04/ge75vid2/haoyu/live-ultrasound-video-understanding/cluster_data/logs/pretrain_train50_%j.err

set -euo pipefail

REPO_DIR="/dss/dsshome1/04/ge75vid2/haoyu/live-ultrasound-video-understanding"
cd "${REPO_DIR}"

mkdir -p cluster_data/logs cluster_data/checkpoints

echo "======================================================================"
echo "[job] Pretrain Full V1: train50"
echo "[job] job_id=${SLURM_JOB_ID:-unknown}"
echo "[job] node=${SLURMD_NODENAME:-unknown}"
echo "[job] start_time=$(date)"
echo "[job] repo=${REPO_DIR}"
echo "======================================================================"

# Activate conda environment.
# This is intentionally robust for non-interactive sbatch shells.
if [ -f "/dss/dsshome1/04/ge75vid2/miniconda3/etc/profile.d/conda.sh" ]; then
  source "/dss/dsshome1/04/ge75vid2/miniconda3/etc/profile.d/conda.sh"
else
  source "${HOME}/miniconda3/etc/profile.d/conda.sh"
fi

conda activate ultrasound

echo "======================================================================"
echo "[env] python=$(which python)"
python - <<'PY'
import torch
print("[env] torch:", torch.__version__)
print("[env] cuda:", torch.cuda.is_available())
print("[env] gpu:", torch.cuda.get_device_name(0) if torch.cuda.is_available() else None)
print("[env] bf16:", torch.cuda.is_bf16_supported() if torch.cuda.is_available() else None)
PY
echo "======================================================================"

echo "[check] train samples:"
wc -l cluster_data/pretrain/train_pretrain_samples.jsonl

echo "[check] train videos:"
python - <<'PY'
import json
m = json.load(open("cluster_data/splits/train_videos.json"))
print("[check] train_videos:", len(m))
PY

echo "======================================================================"
echo "[train] starting pretrain/train.py"
echo "======================================================================"

python pretrain/train.py \
  --model-name Qwen/Qwen3-VL-2B-Instruct \
  --train-jsonl cluster_data/pretrain/train_pretrain_samples.jsonl \
  --video-path-map cluster_data/splits/train_videos.json \
  --output-dir cluster_data/checkpoints/pretrain_qwen3vl_train50 \
  --window-size 4 \
  --frame-size 224 \
  --num-train-epochs 3 \
  --per-device-train-batch-size 1 \
  --gradient-accumulation-steps 8 \
  --learning-rate 1e-4 \
  --bf16 \
  --gradient-checkpointing \
  --early-stop-patience 3 \
  --early-stop-min-delta 0.001

echo "======================================================================"
echo "[job] finished_time=$(date)"
echo "[job] output_dir=cluster_data/checkpoints/pretrain_qwen3vl_train50"
echo "======================================================================"