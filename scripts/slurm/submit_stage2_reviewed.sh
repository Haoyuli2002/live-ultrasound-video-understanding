#!/usr/bin/env bash
# Run on LRZ after uploading the frozen release. One recorded submission per architecture.
set -euo pipefail
ARCHITECTURE=${1:-}
[[ "$ARCHITECTURE" == ordinary || "$ARCHITECTURE" == qformer ]] || {
  echo 'Usage: bash scripts/slurm/submit_stage2_reviewed.sh ordinary|qformer' >&2
  exit 2
}
export REPO=${REPO:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}
export DATA=${DATA:-/dss/mcmlscratch/04/ge75vid2/haoyu/live-ultrasound-video-understanding}
export DATASET_RELEASE=${DATASET_RELEASE:-$REPO/stage2/releases/accepted_v4_20261105_990b9b48}
export STAGE1_ADAPTER="$DATA/cluster_data/experiments/stage1/stage1_v21_train191_2b_4gpu_f30_e1_20261101T071541Z_sCe9Fb/final"
export QFORMER_SMOKE_REPORT=${QFORMER_SMOKE_REPORT:-}
export GPU_TYPE=${GPU_TYPE:-auto} NPROC_PER_NODE=4 EPOCHS=1 WALLTIME=24:00:00
export LEARNING_RATE=${LEARNING_RATE:-1e-4} SAVE_STEPS=${SAVE_STEPS:-100}
export DRY_RUN=0
unset RESUME_RUN
export CONDA_SH=${CONDA_SH:-$HOME/miniconda3/etc/profile.d/conda.sh} CONDA_ENV=ultrasound
source "$CONDA_SH"
conda activate "$CONDA_ENV"
cd "$REPO"
RECORD_DIR="$DATA/cluster_data/experiments/stage2/accepted_v4_20261105_990b9b48_submissions"
RECORD_KEY=$ARCHITECTURE
[[ "$ARCHITECTURE" != qformer ]] || RECORD_KEY=qformer_secondwise_v2
mkdir -p "$RECORD_DIR"
LOCK_DIR="$RECORD_DIR/$RECORD_KEY.lock"
mkdir "$LOCK_DIR" 2>/dev/null || {
  echo "Another submission or unfinished handoff holds $LOCK_DIR; inspect records before retrying." >&2
  exit 2
}
trap 'rmdir "$LOCK_DIR"' EXIT
# Recover an sbatch success if the terminal disconnected before the helper saved its record.
shopt -s nullglob
for PREVIOUS in "$RECORD_DIR/${RECORD_KEY}_"*.log; do
  [[ -s "$RECORD_DIR/$RECORD_KEY.job_id.txt" ]] && break
  PREV_JOB=$(awk '$1=="Job:" {print $2}' "$PREVIOUS")
  PREV_RUN=$(sed -n 's/^Run: //p' "$PREVIOUS")
  if [[ "$PREV_JOB" =~ ^[0-9]+$ && -n "$PREV_RUN" ]]; then
    printf '%s\n' "$PREV_JOB" > "$RECORD_DIR/$RECORD_KEY.job_id.txt"
    printf '%s\n' "$PREV_RUN" > "$RECORD_DIR/$RECORD_KEY.run_dir.txt"
  fi
 done
if [[ -s "$RECORD_DIR/$RECORD_KEY.job_id.txt" ]]; then
  echo "Already recorded: $(cat "$RECORD_DIR/$RECORD_KEY.job_id.txt")"
  cat "$RECORD_DIR/$RECORD_KEY.run_dir.txt"
  exit 0
fi
python - <<'PYVERIFY'
import hashlib,os
from pathlib import Path
p=Path(os.environ['DATASET_RELEASE'])/'manifest.json'
if not p.is_file():raise SystemExit(f'Missing frozen release: {p}; git pull origin main first')
if hashlib.sha256(p.read_bytes()).hexdigest()!='990b9b489194028ee4437ef64def17ee515d3cf273702c750633c22df834cb4b':
 raise SystemExit('This helper is pinned to accepted_v4_20261105_990b9b48; use the generic submitter for a different release')
PYVERIFY
python -m stage2.release --manifest "$DATASET_RELEASE/manifest.json"
if [[ "$ARCHITECTURE" == qformer && ! -s "$QFORMER_SMOKE_REPORT" ]]; then
  echo "Q-Former waits for passed smoke report: $QFORMER_SMOKE_REPORT" >&2
  echo 'Run a new secondwise smoke, then set QFORMER_SMOKE_REPORT to its report.' >&2
  exit 2
fi
LOG="$RECORD_DIR/${RECORD_KEY}_$(date -u +%Y%m%dT%H%M%SZ).log"
bash scripts/slurm/submit_stage2_full_training.sh "$ARCHITECTURE" | tee "$LOG"
JOB=$(awk '$1=="Job:" {print $2}' "$LOG")
RUN_DIR=$(sed -n 's/^Run: //p' "$LOG")
[[ "$JOB" =~ ^[0-9]+$ && -n "$RUN_DIR" ]] || {
  echo "Inspect $LOG before retrying; a submission may have occurred." >&2
  exit 2
}
printf '%s\n' "$JOB" > "$RECORD_DIR/$RECORD_KEY.job_id.txt"
printf '%s\n' "$RUN_DIR" > "$RECORD_DIR/$RECORD_KEY.run_dir.txt"
python - "$DATASET_RELEASE/manifest.json" "$RECORD_DIR/$RECORD_KEY.release_sha256.txt" <<'PY'
import hashlib,sys
from pathlib import Path
Path(sys.argv[2]).write_text(hashlib.sha256(Path(sys.argv[1]).read_bytes()).hexdigest()+'\n')
PY
squeue -j "$JOB"
