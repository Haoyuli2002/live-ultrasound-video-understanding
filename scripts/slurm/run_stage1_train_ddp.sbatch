#!/bin/bash
#SBATCH --job-name=stage1-ddp
#SBATCH --partition=lrz-hgx-h100-94x4
#SBATCH --qos=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:2
#SBATCH --cpus-per-task=16
#SBATCH --mem=160G
#SBATCH --time=14:00:00
#SBATCH --output=logs/stage1_pretrain_%j.out
#SBATCH --error=logs/stage1_pretrain_%j.err

# Single-node DDP: two GPUs by default; four GPUs via --gres=gpu:4 and
# --export=ALL,NPROC_PER_NODE=4. mkdir -p logs before sbatch.
set -euo pipefail
export REPO=${REPO:-/dss/dsshome1/04/ge75vid2/haoyu/live-ultrasound-video-understanding}
export NPROC_PER_NODE=${NPROC_PER_NODE:-2}
case "$NPROC_PER_NODE" in
  2|4) ;;
  *) echo "NPROC_PER_NODE must be 2 or 4 for this DDP wrapper" >&2; exit 1 ;;
esac
export EPOCHS=${EPOCHS:-1}
export FRAME_BUDGET=${FRAME_BUDGET:-48}
export DATALOADER_NUM_WORKERS=${DATALOADER_NUM_WORKERS:-2}
export GRADIENT_ACCUMULATION_STEPS=${GRADIENT_ACCUMULATION_STEPS:-$((8 / NPROC_PER_NODE))}
export RUN_NAME=${RUN_NAME:-stage1_full186_2b_ddp_${SLURM_JOB_ID:-local_$$}}
export TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS=${OMP_NUM_THREADS:-2}
# The shared launcher retains the short IPC directory, scratch caches,
# input validation, single-GPU behavior and checkpoint-resume arguments.
bash "$REPO/scripts/slurm/run_stage1_pretrain_merged.sbatch"
