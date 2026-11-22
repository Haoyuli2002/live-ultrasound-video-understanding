#!/bin/bash
#SBATCH --job-name=pretrain_v3_interleave
#SBATCH --partition=mcml-hgx-h100-94x4
#SBATCH --qos=mcml
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=120G
#SBATCH --time=24:00:00
#SBATCH --output=logs/pretrain_v3_interleave_%j.out
#SBATCH --error=logs/pretrain_v3_interleave_%j.err

set -euo pipefail

cd /dss/dsshome1/04/ge75vid2/haoyu/live-ultrasound-video-understanding

source /dss/dsshome1/04/ge75vid2/miniconda3/etc/profile.d/conda.sh
conda activate ultrasound

mkdir -p logs cluster_data/checkpoints

python pretrain/train.py \
  --model-name Qwen/Qwen3-VL-2B-Instruct \
  --train-jsonl cluster_data/pretrain/train_pretrain_punct_sentence_interleave_samples.jsonl \
  --video-path-map cluster_data/splits/train_videos.json \
  --output-dir cluster_data/checkpoints/pretrain_qwen3vl_train50_punct_sentence_interleave \
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
