#!/usr/bin/env bash
#SBATCH --partition=lrz-hgx-h100-94x4
#SBATCH --qos=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=120G
#SBATCH --time=24:00:00
set -euo pipefail
cd "${REPO:?Set REPO to the frozen code snapshot}"
source "${CONDA_SH:-$HOME/miniconda3/etc/profile.d/conda.sh}"
conda activate "${CONDA_ENV:-ultrasound}"
export PYTHONUNBUFFERED=1
python -m stage2.full_run worker --repo "$REPO" \
  --run-dir "${STAGE2_RUN_DIR:?Set STAGE2_RUN_DIR}" \
  --shard "${SLURM_ARRAY_TASK_ID:?This script requires a Slurm array}"
