#!/bin/bash
#SBATCH --job-name=sr-select-allpairs
#SBATCH --partition=lrz-hgx-a100-80x4,lrz-dgx-a100-80x8
#SBATCH --qos=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --time=01:00:00
set -euo pipefail
: "${RUN_DIR:?Set RUN_DIR}"
export PYTHONDONTWRITEBYTECODE=1
export PYTHONPATH="$RUN_DIR/code"
cd "$RUN_DIR/code"
/dss/dsshome1/04/ge75vid2/miniconda3/envs/ultrasound/bin/python -B -m endtoend.select_all_pairs --run-dir "$RUN_DIR"
