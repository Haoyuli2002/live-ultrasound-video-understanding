#!/usr/bin/env bash
# Thin wrapper: submit the Stage 2 FULL training for 3 epochs.
#
# This does NOT change any training logic. The core submitter
# `submit_stage2_full_training.sh` already reads EPOCHS from the environment
# (default 1) and forwards it to `stage2.train_full --epochs`. We only pin
# EPOCHS=3 here, run a DRY_RUN pre-check first, then submit.
#
# Usage (on the SSH cluster, from the repo root):
#   # 1) Dry-run pre-check (no job created, prints the exact sbatch command):
#   DRY_RUN=1 bash scripts/slurm/submit_stage2_full_training_3ep.sh qformer
#   # 2) Real submit:
#   bash scripts/slurm/submit_stage2_full_training_3ep.sh qformer
#
# Notes:
# - Pass the architecture as $1: `qformer` (default) or `ordinary`.
# - For qformer, QFORMER_SMOKE_REPORT must point to a passed smoke_report.json
#   (the core submitter enforces this gate). Export it before running, e.g.:
#     export QFORMER_SMOKE_REPORT=/dss/mcmlscratch/.../qformer_smoke_P3_*/smoke_report.json
# - The core submitter refuses to start if a job named `stage2-full-<arch>` is
#   already running. Let the current 1-epoch job (5919253) finish, or cancel it,
#   before submitting this 3-epoch run.
# - All other knobs keep their defaults (LEARNING_RATE=1e-4, SAVE_STEPS=100,
#   NPROC_PER_NODE=4, WALLTIME=24:00:00). Override by exporting them first.
set -euo pipefail

ARCH="${1:-qformer}"
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

export EPOCHS=3
echo "[3ep-wrapper] architecture=$ARCH EPOCHS=$EPOCHS DRY_RUN=${DRY_RUN:-0}"
echo "[3ep-wrapper] delegating to submit_stage2_full_training.sh ..."
exec bash "$HERE/submit_stage2_full_training.sh" "$ARCH"
