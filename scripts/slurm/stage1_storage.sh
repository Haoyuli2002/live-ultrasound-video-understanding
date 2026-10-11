#!/bin/bash
# Source after activating ultrasound, before starting Python.
# Explicit experiment roots take precedence over inherited home-based caches.
: "${DATA:?Set DATA to the experiment storage root}"
export STAGE1_CACHE_ROOT=${STAGE1_CACHE_ROOT:-$(dirname "$DATA")}
export HF_HOME="$STAGE1_CACHE_ROOT/hf_cache"
export HF_HUB_CACHE="$HF_HOME/hub"
export HUGGINGFACE_HUB_CACHE="$HF_HUB_CACHE"
# Avoid deprecated overrides redirecting model weights back into home.
unset TRANSFORMERS_CACHE PYTORCH_TRANSFORMERS_CACHE PYTORCH_PRETRAINED_BERT_CACHE
export HF_DATASETS_CACHE="$HF_HOME/datasets"
export HF_XET_CACHE="$HF_HOME/xet"
export XDG_CACHE_HOME="$STAGE1_CACHE_ROOT/xdg_cache"
export TORCH_HOME="$STAGE1_CACHE_ROOT/torch_cache"
export PIP_CACHE_DIR="$STAGE1_CACHE_ROOT/pip_cache"
export CONDA_PKGS_DIRS="$STAGE1_CACHE_ROOT/conda_pkgs"
export TMPDIR="$DATA/tmp/stage1_${SLURM_JOB_ID:-local_$$}"
export TMP="$TMPDIR" TEMP="$TMPDIR"
export TRITON_CACHE_DIR="$TMPDIR/triton"
export TORCHINDUCTOR_CACHE_DIR="$TMPDIR/torchinductor"
export CUDA_CACHE_PATH="$TMPDIR/cuda"
export PYTHONPYCACHEPREFIX="$TMPDIR/pycache"
mkdir -p "$HF_HUB_CACHE" "$HF_DATASETS_CACHE" "$HF_XET_CACHE" \
  "$XDG_CACHE_HOME" "$TORCH_HOME" "$PIP_CACHE_DIR" "$CONDA_PKGS_DIRS" \
  "$TMPDIR" "$TRITON_CACHE_DIR" "$TORCHINDUCTOR_CACHE_DIR" "$CUDA_CACHE_PATH" \
  "$PYTHONPYCACHEPREFIX"
echo "Stage 1 storage: DATA=$DATA; HF_HOME=$HF_HOME; TMPDIR=$TMPDIR"
