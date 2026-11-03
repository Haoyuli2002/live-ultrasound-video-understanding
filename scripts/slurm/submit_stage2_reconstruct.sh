#!/usr/bin/env bash
set -euo pipefail
SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
REPO=${REPO:-$(cd "$SCRIPT_DIR/../.." && pwd)}
export TRAIN_RUN=${TRAIN_RUN:-${1:-}}
if [[ -z "$TRAIN_RUN" ]]; then
  echo 'Set TRAIN_RUN to the completed Stage 2 overfit experiment.' >&2
  exit 2
fi
export CHECKPOINT=${CHECKPOINT:-final}
export GPU_TYPE=${GPU_TYPE:-auto}
export WALLTIME=${WALLTIME:-01:00:00}
export MAX_NEW_TOKENS=${MAX_NEW_TOKENS:-160}
export CONDA_SH=${CONDA_SH:-$HOME/miniconda3/etc/profile.d/conda.sh}
export CONDA_ENV=${CONDA_ENV:-ultrasound}
source "$CONDA_SH"
conda activate "$CONDA_ENV"
RECON_RUN=$(python - <<'PY'
import json,os,tempfile
from datetime import datetime,timezone
from pathlib import Path
r=Path(os.environ['TRAIN_RUN']).expanduser().resolve()
checkpoint=os.environ['CHECKPOINT']
if checkpoint not in {'final','epoch-1','epoch-20'}:
 raise SystemExit('CHECKPOINT must be final, epoch-1 or epoch-20')
for p in [r/'code/stage2/model.py',r/'blocks.jsonl',r/'video_map.json',
          r/'training/training_config.json',r/'training'/checkpoint/'adapter_config.json']:
 if not p.is_file(): raise SystemExit(f'Missing frozen experiment input: {p}')
cfg=json.loads((r/'training/training_config.json').read_text())
if cfg.get('init_adapter'):
 raise SystemExit('This overfit submission expects official-checkpoint initialization')
for vid,p in json.loads((r/'video_map.json').read_text()).items():
 if not Path(p).is_file(): raise SystemExit(f'Missing video {vid}: {p}')
token_budget=int(os.environ['MAX_NEW_TOKENS'])
if token_budget<1: raise SystemExit('MAX_NEW_TOKENS must be positive')
stamp=datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')
print(tempfile.mkdtemp(prefix=f'reconstruct_{checkpoint}_{stamp}_',dir=r))
PY
)
export RECON_RUN
export CODE_DIR="$RECON_RUN/code"
# Keep the exact architecture used for training; overlay only the new decoder.
cp -R "$TRAIN_RUN/code" "$CODE_DIR"
cp "$REPO/stage2/reconstruct.py" "$CODE_DIR/stage2/reconstruct.py"
cp "$SCRIPT_DIR/run_stage2_reconstruct.sbatch" "$RECON_RUN/run.sbatch"
mkdir "$RECON_RUN/logs"
unset STAGE1_ADAPTER STAGE2_ADAPTER
PARTITIONS=$(cd "$REPO" && python -m scripts.gpu_profiles "$GPU_TYPE")
export RECON_PARTITIONS="$PARTITIONS"
python - <<'PY'
import hashlib,json,os
from pathlib import Path
r=Path(os.environ['RECON_RUN'])
keys=['TRAIN_RUN','RECON_RUN','CODE_DIR','CHECKPOINT','GPU_TYPE','WALLTIME',
      'MAX_NEW_TOKENS','RECON_PARTITIONS']
plan={k:os.environ[k] for k in keys}
plan.update(gpu_count=1,training=False,conditions=['real','zero','other_window'],
            target_used_in_decoding=False,
            decoder_sha256=hashlib.sha256((r/'code/stage2/reconstruct.py').read_bytes()).hexdigest())
(r/'submission.json').write_text(json.dumps(plan,indent=2))
print(json.dumps(plan,indent=2))
PY
command=(sbatch --parsable --export=ALL --job-name=stage2-reconstruct \
  --partition="$PARTITIONS" --qos=gpu --nodes=1 --ntasks=1 --gres=gpu:1 \
  --cpus-per-task=4 --mem=64G --time="$WALLTIME" --chdir="$CODE_DIR" \
  --output="$RECON_RUN/logs/reconstruct_%j.out" \
  --error="$RECON_RUN/logs/reconstruct_%j.err" "$RECON_RUN/run.sbatch")
if [[ "${DRY_RUN:-0}" == 1 ]]; then
  printf '%q ' "${command[@]}"; printf '\nDry run: no job submitted.\n'
  exit 0
fi
JOB_RAW=$("${command[@]}")
JOB_ID=${JOB_RAW%%;*}
printf '%s\n' "$JOB_ID" > "$RECON_RUN/job_id.txt"
echo "Job: $JOB_ID"
echo "Run: $RECON_RUN"
echo "squeue -j $JOB_ID"
echo "tail -F $RECON_RUN/logs/reconstruct_${JOB_ID}.out $RECON_RUN/logs/reconstruct_${JOB_ID}.err"
