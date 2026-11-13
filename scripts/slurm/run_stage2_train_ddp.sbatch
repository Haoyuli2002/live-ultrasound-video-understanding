#!/bin/bash
#SBATCH --job-name=stage2-ddp
#SBATCH --nodes=2
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=32
#SBATCH --mem=0
#SBATCH --time=24:00:00
#SBATCH --output=logs/stage2_train_ddp_%j.out
#SBATCH --error=logs/stage2_train_ddp_%j.err

# Eight-GPU Stage 2 training, defaulting to two four-GPU LRZ nodes. Submit to a
# partition that permits two x4 nodes:
#   sbatch --partition=<4-gpu-per-node-partition> scripts/slurm/run_stage2_train_ddp.sbatch
# For one eight-GPU node, override the allocation and NPROC_PER_NODE:
#   sbatch --nodes=1 --gres=gpu:8 --export=ALL,NPROC_PER_NODE=8,... <this-file>
# Override paths/hyperparameters with --export=ALL,NAME=value,... . Set
# RESUME=1 only when resubmitting the same experiment after a completed epoch.

set -euo pipefail

REPO=${REPO:-/dss/dsshome1/04/ge75vid2/haoyu/live-ultrasound-video-understanding}
DATA=${DATA:-/dss/mcmlscratch/04/ge75vid2/haoyu/live-ultrasound-video-understanding}
TRAIN_JSONL=${TRAIN_JSONL:-$DATA/cluster_data/stage2/blocks/train_full295_train.jsonl}
VIDEO_MAP=${VIDEO_MAP:-$DATA/cluster_data/stage2/train_full295_video_paths.json}
STAGE1_ADAPTER=${STAGE1_ADAPTER:-$DATA/results/stage1_output/final}
OUTPUT_DIR=${OUTPUT_DIR:-$DATA/results/stage2_output}

MODEL=${MODEL:-Qwen/Qwen3.5-2B}
EPOCHS=${EPOCHS:-1}
LEARNING_RATE=${LEARNING_RATE:-1e-4}
FRAME_SIZE=${FRAME_SIZE:-224}
LORA_RANK=${LORA_RANK:-16}
DTYPE=${DTYPE:-bf16}
MAX_GRAD_NORM=${MAX_GRAD_NORM:-1.0}
VALIDATION_FRACTION=${VALIDATION_FRACTION:-0.1}
WARMUP_RATIO=${WARMUP_RATIO:-0.03}
SEED=${SEED:-42}
NPROC_PER_NODE=${NPROC_PER_NODE:-4}
RESUME=${RESUME:-0}

mkdir -p "$REPO/logs" "$OUTPUT_DIR"
cd "$REPO"

if [[ -f /dss/dsshome1/04/ge75vid2/miniconda3/etc/profile.d/conda.sh ]]; then
  source /dss/dsshome1/04/ge75vid2/miniconda3/etc/profile.d/conda.sh
else
  source "$HOME/miniconda3/etc/profile.d/conda.sh"
fi
conda activate ultrasound

export HF_HOME=${HF_HOME:-$DATA/hf_cache}
export TRANSFORMERS_CACHE=${TRANSFORMERS_CACHE:-$HF_HOME}
export HF_HUB_CACHE=${HF_HUB_CACHE:-$HF_HOME/hub}
export TOKENIZERS_PARALLELISM=${TOKENIZERS_PARALLELISM:-false}
export OMP_NUM_THREADS=${OMP_NUM_THREADS:-8}
mkdir -p "$HF_HOME" "$HF_HUB_CACHE"

[[ -f "$TRAIN_JSONL" ]] || { echo "[error] TRAIN_JSONL not found: $TRAIN_JSONL" >&2; exit 1; }
[[ -f "$VIDEO_MAP" ]] || { echo "[error] VIDEO_MAP not found: $VIDEO_MAP" >&2; exit 1; }
[[ -d "$STAGE1_ADAPTER" ]] || { echo "[error] STAGE1_ADAPTER not found: $STAGE1_ADAPTER" >&2; exit 1; }
NNODES=${SLURM_NNODES:-2}
TOTAL_PROCESSES=$((NNODES * NPROC_PER_NODE))
if [[ "$TOTAL_PROCESSES" != 8 ]]; then
  echo "[error] expected 8 total workers, got NNODES=$NNODES x NPROC_PER_NODE=$NPROC_PER_NODE" >&2
  exit 1
fi

mapfile -t hosts < <(scontrol show hostnames "$SLURM_JOB_NODELIST")
MASTER_ADDR=${hosts[0]}
MASTER_PORT=${MASTER_PORT:-$((20000 + SLURM_JOB_ID % 20000))}
export NNODES NPROC_PER_NODE MASTER_ADDR MASTER_PORT

echo "========== STAGE 2 DDP =========="
hostname; date; nvidia-smi -L
echo "MODEL=$MODEL"
echo "TRAIN_JSONL=$TRAIN_JSONL"
echo "VIDEO_MAP=$VIDEO_MAP"
echo "STAGE1_ADAPTER=$STAGE1_ADAPTER"
echo "OUTPUT_DIR=$OUTPUT_DIR"
echo "NNODES=$NNODES"
echo "NPROC_PER_NODE=$NPROC_PER_NODE"
echo "MASTER_ADDR=$MASTER_ADDR"
echo "MASTER_PORT=$MASTER_PORT"
echo "RESUME=$RESUME"
echo "================================="

args=(
  --model-name "$MODEL"
  --train-jsonl "$TRAIN_JSONL"
  --video-path-map "$VIDEO_MAP"
  --init-adapter "$STAGE1_ADAPTER"
  --output-dir "$OUTPUT_DIR"
  --epochs "$EPOCHS"
  --learning-rate "$LEARNING_RATE"
  --frame-size "$FRAME_SIZE"
  --lora-rank "$LORA_RANK"
  --dtype "$DTYPE"
  --max-grad-norm "$MAX_GRAD_NORM"
  --validation-fraction "$VALIDATION_FRACTION"
  --warmup-ratio "$WARMUP_RATIO"
  --seed "$SEED"
)
if [[ "$RESUME" == 1 ]]; then
  args+=(--resume)
fi

srun --nodes="$NNODES" --ntasks="$NNODES" --ntasks-per-node=1 \
  bash -c 'torchrun \
    --nnodes="$NNODES" \
    --nproc_per_node="$NPROC_PER_NODE" \
    --node_rank="$SLURM_PROCID" \
    --master_addr="$MASTER_ADDR" \
    --master_port="$MASTER_PORT" \
    -m stage2.train "$@"' _ "${args[@]}"

echo "[done] Stage 2 DDP training finished at $(date)"
