#!/bin/bash
#SBATCH --job-name=eval_v3_interleave
#SBATCH --partition=mcml-hgx-h100-94x4
#SBATCH --qos=mcml
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=80G
#SBATCH --time=06:00:00
#SBATCH --output=logs/eval_v3_interleave_%j.out
#SBATCH --error=logs/eval_v3_interleave_%j.err

set -euo pipefail

cd /dss/dsshome1/04/ge75vid2/haoyu/live-ultrasound-video-understanding

source /dss/dsshome1/04/ge75vid2/miniconda3/etc/profile.d/conda.sh
conda activate ultrasound

mkdir -p logs cluster_data/eval

python pretrain/infer.py \
  --model-name Qwen/Qwen3-VL-2B-Instruct \
  --no-adapter \
  --eval-jsonl cluster_data/pretrain/eval_pretrain_punct_sentence_interleave_samples.jsonl \
  --video-path-map cluster_data/splits/eval_videos.json \
  --output cluster_data/eval/pretrain_v3_interleave_base_limit200.jsonl \
  --window-size 4 \
  --frame-size 224 \
  --max-new-tokens 128 \
  --limit 200 \
  --bf16 \
  --quiet \
  --resume

python pretrain/infer.py \
  --model-name Qwen/Qwen3-VL-2B-Instruct \
  --adapter-path cluster_data/checkpoints/pretrain_qwen3vl_train50_punct_sentence_interleave \
  --eval-jsonl cluster_data/pretrain/eval_pretrain_punct_sentence_interleave_samples.jsonl \
  --video-path-map cluster_data/splits/eval_videos.json \
  --output cluster_data/eval/pretrain_v3_interleave_lora_limit200.jsonl \
  --window-size 4 \
  --frame-size 224 \
  --max-new-tokens 128 \
  --limit 200 \
  --bf16 \
  --quiet \
  --resume
