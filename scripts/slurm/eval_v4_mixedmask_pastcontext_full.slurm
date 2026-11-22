#!/bin/bash
#SBATCH --job-name=v4_eval_pastctx
#SBATCH --partition=mcml-hgx-h100-94x4
#SBATCH --qos=mcml
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=120G
#SBATCH --time=08:00:00
#SBATCH --output=logs/eval_v4_mixedmask_pastcontext_%j.out
#SBATCH --error=logs/eval_v4_mixedmask_pastcontext_%j.err

set -euo pipefail

cd /dss/dsshome1/04/ge75vid2/haoyu/live-ultrasound-video-understanding

source /dss/dsshome1/04/ge75vid2/miniconda3/etc/profile.d/conda.sh
conda activate ultrasound

mkdir -p logs cluster_data/eval

echo "== V4 past-context eval: base =="
python pretrain/infer.py \
  --model-name Qwen/Qwen3-VL-2B-Instruct \
  --no-adapter \
  --eval-jsonl cluster_data/pretrain/eval_pretrain_punct_sentence_mixedmask_samples.jsonl \
  --video-path-map cluster_data/splits/eval_videos.json \
  --output cluster_data/eval/pretrain_v4_mixedmask_base_full_pastcontext.jsonl \
  --window-size 4 \
  --frame-size 224 \
  --max-new-tokens 128 \
  --mixedmask-eval-mode mask_current_visual \
  --bf16 \
  --quiet \
  --resume

echo "== V4 past-context eval: LoRA =="
python pretrain/infer.py \
  --model-name Qwen/Qwen3-VL-2B-Instruct \
  --adapter-path cluster_data/checkpoints/pretrain_qwen3vl_train50_punct_sentence_mixedmask \
  --eval-jsonl cluster_data/pretrain/eval_pretrain_punct_sentence_mixedmask_samples.jsonl \
  --video-path-map cluster_data/splits/eval_videos.json \
  --output cluster_data/eval/pretrain_v4_mixedmask_lora_full_pastcontext.jsonl \
  --window-size 4 \
  --frame-size 224 \
  --max-new-tokens 128 \
  --mixedmask-eval-mode mask_current_visual \
  --bf16 \
  --quiet \
  --resume

echo "== compare V4 past-context direct metrics =="
python pretrain/compare_predictions.py \
  --base cluster_data/eval/pretrain_v4_mixedmask_base_full_pastcontext.jsonl \
  --lora cluster_data/eval/pretrain_v4_mixedmask_lora_full_pastcontext.jsonl \
  --output cluster_data/eval/pretrain_v4_mixedmask_compare_full_pastcontext_direct.jsonl \
  --summary-json cluster_data/eval/pretrain_v4_mixedmask_compare_full_pastcontext_direct_summary.json \
  --no-semantic

echo "== summary =="
cat cluster_data/eval/pretrain_v4_mixedmask_compare_full_pastcontext_direct_summary.json

echo "== outputs =="
ls -lh cluster_data/eval/pretrain_v4_mixedmask_*pastcontext*
wc -l cluster_data/eval/pretrain_v4_mixedmask_*pastcontext*.jsonl

echo "== done =="
