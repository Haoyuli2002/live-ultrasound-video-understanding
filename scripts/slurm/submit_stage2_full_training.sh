#!/usr/bin/env bash
# Both architectures require a COMPLETE immutable release. Q-Former also needs smoke success.
set -euo pipefail
export ARCHITECTURE=${1:-}
[[ "$ARCHITECTURE" == ordinary || "$ARCHITECTURE" == qformer ]] || {
  echo 'Usage: submit_stage2_full_training.sh ordinary|qformer' >&2; exit 2;
}
export REPO=${REPO:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}
export DATA=${DATA:-/dss/mcmlscratch/04/ge75vid2/haoyu/live-ultrasound-video-understanding}
export CONDA_SH=${CONDA_SH:-$HOME/miniconda3/etc/profile.d/conda.sh} CONDA_ENV=${CONDA_ENV:-ultrasound}
source "$CONDA_SH"
conda activate "$CONDA_ENV"
cd "$REPO"
git diff --quiet HEAD -- stage2 scripts/slurm/submit_stage2_full_training.sh scripts/slurm/run_stage2_full_training.sbatch || {
  echo 'Commit the training code before freezing a full run.' >&2; exit 2;
}
: "${DATASET_RELEASE:?Set DATASET_RELEASE to the complete frozen accepted release}"
export DATASET_RELEASE
export STAGE1_ADAPTER=${STAGE1_ADAPTER:-$DATA/cluster_data/experiments/stage1/stage1_v21_train191_2b_4gpu_f30_e1_20261101T071541Z_sCe9Fb/final}
export NPROC_PER_NODE=${NPROC_PER_NODE:-4} EPOCHS=${EPOCHS:-1}
export LEARNING_RATE=${LEARNING_RATE:-1e-4} SAVE_STEPS=${SAVE_STEPS:-100}
export WALLTIME=${WALLTIME:-24:00:00}
[[ "$NPROC_PER_NODE" =~ ^[124]$ && "$EPOCHS" =~ ^[1-9][0-9]*$ ]] || exit 2
PARTITIONS=$(python -m scripts.gpu_profiles "${GPU_TYPE:-auto}")
# Do every data/smoke gate before creating a run or contacting sbatch.
python - <<'PY'
import json,math,os
from pathlib import Path
from stage2.release import validate_release
from stage2.provenance import digest
m=validate_release(Path(os.environ['DATASET_RELEASE'])/'manifest.json')
s=Path(os.environ['STAGE1_ADAPTER']);assert (s/'adapter_config.json').is_file()
assert (s/'adapter_model.safetensors').stat().st_size>0
assert math.isfinite(float(os.environ['LEARNING_RATE'])) and float(os.environ['LEARNING_RATE'])>0
assert int(os.environ['SAVE_STEPS'])>0
if os.environ['ARCHITECTURE']=='qformer':
 report=Path(os.environ.get('QFORMER_SMOKE_REPORT',''))
 if not report.is_file():raise SystemExit('Q-Former full training waits for a passed single-video smoke_report.json')
 q=json.loads(report.read_text())
 assert q['status']=='functional_smoke_passed' and q['frame_size']==224
 assert q['model_name']=='Qwen/Qwen3.5-2B'
 assert q['init_adapter_sha256']==digest(s/'adapter_model.safetensors')
 assert q['source_sha256']==next(v['source_sha256'] for v in m['videos'] if v['video_id']=='P3FoSFIrfQA')
 assert q['qformer_config']['width']==512 and q['qformer_config']['layers']==2 and q['qformer_config']['heads']==8
 for name in ['model.py','qformer.py','train_overfit_ab.py']:
  assert q['implementation_sha256'][name]==digest(Path('stage2')/name),'Memory implementation changed since smoke; rerun smoke'
print(f"Release ready: {len(m['splits']['train'])} TRAIN videos; {len(m['splits']['eval'])} EVAL videos excluded")
PY
if [[ ${DRY_RUN:-0} != 1 ]]; then
  ACTIVE=$(squeue -h -u "$USER" --name="stage2-full-$ARCHITECTURE" -o '%i %T')
  [[ -z "$ACTIVE" ]] || { echo "Existing full training job: $ACTIVE" >&2; exit 1; }
fi
mkdir -p "$DATA/cluster_data/experiments/stage2" logs
if [[ -n ${RESUME_RUN:-} ]]; then
  RUN_DIR="$RESUME_RUN";export RUN_DIR CODE_DIR="$RESUME_RUN/code"
  [[ -d "$RUN_DIR/training/checkpoint-latest" && -s "$RUN_DIR/code_commit.txt" ]] || exit 2
  [[ $(cat "$RUN_DIR/code_commit.txt") == $(git rev-parse HEAD) ]] || { echo 'Resume with the same code commit' >&2;exit 2; }
else
  RUN_DIR=$(mktemp -d "$DATA/cluster_data/experiments/stage2/full_${ARCHITECTURE}_$(date -u +%Y%m%dT%H%M%SZ)_XXXXXX")
  export RUN_DIR CODE_DIR="$RUN_DIR/code"
  mkdir "$CODE_DIR"
  git archive HEAD | tar -xf - -C "$CODE_DIR"
  git rev-parse HEAD > "$RUN_DIR/code_commit.txt"
  map_args=()
  if [[ -n ${INPUT_VIDEO_MAP:-} ]]; then map_args+=(--input-video-map "$INPUT_VIDEO_MAP"); fi
  python -m stage2.release --manifest "$DATASET_RELEASE/manifest.json" \
    --video-root "${VIDEO_ROOT:-$DATA/cluster_data/videos}" --output-video-map "$RUN_DIR/video_map.json" \
    ${map_args[@]+"${map_args[@]}"}
fi
python - <<'PY'
import json,os
from pathlib import Path
from stage2.provenance import digest
r=Path(os.environ['RUN_DIR']);m=Path(os.environ['DATASET_RELEASE'])/'manifest.json'
cfg={k:os.environ[k] for k in ['ARCHITECTURE','NPROC_PER_NODE','EPOCHS','LEARNING_RATE','SAVE_STEPS','WALLTIME','STAGE1_ADAPTER','DATASET_RELEASE']}
cfg.update(release_sha256=digest(m),frame_size=224,short_tokens_per_10s=10,long_tokens=60)
(r/'submission.json').write_text(json.dumps(cfg,indent=2))
PY
command=(sbatch --parsable --export=ALL --job-name="stage2-full-$ARCHITECTURE" --partition="$PARTITIONS" \
  --qos=gpu --nodes=1 --ntasks=1 --gres="gpu:$NPROC_PER_NODE" --cpus-per-task=16 --mem=160G \
  --time="$WALLTIME" --chdir="$CODE_DIR" --output="$REPO/logs/stage2_fulltrain_${ARCHITECTURE}_%j.out" \
  --error="$REPO/logs/stage2_fulltrain_${ARCHITECTURE}_%j.err" "$CODE_DIR/scripts/slurm/run_stage2_full_training.sbatch")
if [[ ${DRY_RUN:-0} == 1 ]]; then printf '%q ' "${command[@]}"; printf '\n'; exit 0; fi
JOB=$("${command[@]}");JOB=${JOB%%;*}
printf '%s\n' "$JOB" > "$RUN_DIR/job_id.txt"
echo "Job: $JOB"
echo "Run: $RUN_DIR"
