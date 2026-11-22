#!/bin/bash
#SBATCH --job-name=v4_8b_smoke
#SBATCH --partition=mcml-hgx-h100-94x4
#SBATCH --qos=mcml
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=120G
#SBATCH --time=03:00:00
#SBATCH --output=logs/pretrain_v4_8b_smoke_%j.out
#SBATCH --error=logs/pretrain_v4_8b_smoke_%j.err

set -euo pipefail

cd /dss/dsshome1/04/ge75vid2/haoyu/live-ultrasound-video-understanding

source /dss/dsshome1/04/ge75vid2/miniconda3/etc/profile.d/conda.sh
conda activate ultrasound

mkdir -p logs cluster_data/checkpoints

rm -rf cluster_data/checkpoints/pretrain_qwen3vl_8b_train50_punct_sentence_mixedmask_smoke

python pretrain/train.py \
  --model-name Qwen/Qwen3-VL-8B-Instruct \
  --train-jsonl cluster_data/pretrain/train_pretrain_punct_sentence_mixedmask_samples.jsonl \
  --video-path-map cluster_data/splits/train_videos.json \
  --output-dir cluster_data/checkpoints/pretrain_qwen3vl_8b_train50_punct_sentence_mixedmask_smoke \
  --window-size 4 \
  --frame-size 224 \
  --limit 8 \
  --num-train-epochs 1 \
  --per-device-train-batch-size 1 \
  --gradient-accumulation-steps 4 \
  --learning-rate 1e-4 \
  --bf16 \
  --gradient-checkpointing \
  --disable-early-stop
