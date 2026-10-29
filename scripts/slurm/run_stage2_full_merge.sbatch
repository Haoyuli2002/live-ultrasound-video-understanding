#!/usr/bin/env bash
#SBATCH --partition=lrz-cpu
#SBATCH --qos=cpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=02:00:00
set -euo pipefail
cd "${REPO:?Set REPO to the frozen code snapshot}"
source "${CONDA_SH:-$HOME/miniconda3/etc/profile.d/conda.sh}"
conda activate "${CONDA_ENV:-ultrasound}"
export PYTHONUNBUFFERED=1
python -m stage2.full_run merge --run-dir "${STAGE2_RUN_DIR:?Set STAGE2_RUN_DIR}"
