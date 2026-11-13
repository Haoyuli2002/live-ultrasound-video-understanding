#!/usr/bin/env bash
# Resources are supplied by submit_stage3_overfit.sh.
set -euo pipefail
source "${CONDA_SH:-$HOME/miniconda3/etc/profile.d/conda.sh}"
conda activate "${CONDA_ENV:-ultrasound}"
export HF_HOME=${HF_HOME:-/dss/mcmlscratch/04/ge75vid2/haoyu/hf_cache}
export HF_HUB_CACHE="$HF_HOME/hub"
unset TRANSFORMERS_CACHE
export BACKEND=qwen PYTHON=python
TASK_TMP=$(mktemp -d /tmp/s3of.XXXXXXXX)
trap 'rm -rf -- "$TASK_TMP"' EXIT
export TMPDIR="$TASK_TMP" TMP="$TASK_TMP" TEMP="$TASK_TMP"
cd "${REPO:?Missing frozen code directory}"
python -B -m scripts.gpu_runtime
bash scripts/slurm/run_stage3_overfit.sh
