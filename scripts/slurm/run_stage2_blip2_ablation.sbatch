#!/bin/bash
#SBATCH --job-name=s2-blip2-ab
#SBATCH --partition=lrz-hgx-h100-94x4
#SBATCH --qos=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=16
#SBATCH --mem=160G
#SBATCH --time=24:00:00
set -euo pipefail
source /dss/dsshome1/04/ge75vid2/miniconda3/etc/profile.d/conda.sh
conda activate ultrasound
cd "${RUN_DIR:?Set RUN_DIR to the frozen experiment directory}/code"
export HF_HOME=/dss/mcmlscratch/04/ge75vid2/haoyu/hf_cache HF_HUB_OFFLINE=1
export OMP_NUM_THREADS=2 TOKENIZERS_PARALLELISM=false PYTHONUNBUFFERED=1
for arm in lavis_bert_query_1_60 qformer; do
    # End launcher options before the module; child arguments must reach its parser.
    torchrun --standalone --nproc-per-node=4 --module -- stage2.blip2_overfit --run-dir "$RUN_DIR" --arm "$arm"
done
