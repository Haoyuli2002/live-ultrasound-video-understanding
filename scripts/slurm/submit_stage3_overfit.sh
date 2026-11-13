#!/usr/bin/env bash
# Prepare a frozen experiment and submit only when DRY_RUN=0 is explicitly set.
set -euo pipefail
SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
export REPO=${REPO:-$(cd "$SCRIPT_DIR/../.." && pwd)}
export DATA_DIR=${DATA_DIR:-$REPO/stage3/examples/two_video_overfit}
export VIDEO_MAP=${VIDEO_MAP:-$DATA_DIR/video_map.json}
: "${STAGE2_ADAPTER:?Set a real Stage2 adapter; mock runs use run_stage3_overfit.sh directly}"
: "${STAGE2_TRAINING_CONFIG:?Set its original training_config.json}"
export STAGE2_ADAPTER STAGE2_TRAINING_CONFIG
export EXPERIMENT_ROOT=${EXPERIMENT_ROOT:-/dss/mcmlscratch/04/ge75vid2/haoyu/live-ultrasound-video-understanding/cluster_data/experiments/stage3}
export GPU_TYPE=${GPU_TYPE:-auto}
export EPOCHS=${EPOCHS:-20} LEARNING_RATE=${LEARNING_RATE:-0.0001} MAX_NEW_TOKENS=${MAX_NEW_TOKENS:-128}
export CONDA_SH=${CONDA_SH:-$HOME/miniconda3/etc/profile.d/conda.sh} CONDA_ENV=${CONDA_ENV:-ultrasound}
source "$CONDA_SH"
conda activate "$CONDA_ENV"
cd "$REPO"
EXPERIMENT=$(python -B - <<'PY'
import json,os,shutil,tempfile,hashlib
from pathlib import Path
from stage3.overfit import validate_dataset
repo=Path(os.environ['REPO']).resolve();data=Path(os.environ['DATA_DIR']).resolve()
tasks,training,paths=validate_dataset(data,Path(os.environ['VIDEO_MAP']))
adapter=Path(os.environ['STAGE2_ADAPTER']).resolve();cfgpath=Path(os.environ['STAGE2_TRAINING_CONFIG']).resolve()
if not (adapter/'adapter_config.json').is_file() or (adapter/'mock_config.json').exists():raise SystemExit('Real Stage2 PEFT adapter required')
cfg=json.loads(cfgpath.read_text());base=cfg.get('init_adapter');requested=os.environ.get('STAGE1_ADAPTER')
if bool(base)!=bool(requested) or (base and Path(base).resolve()!=Path(requested).resolve()):raise SystemExit('Stage1 initialization chain mismatch')
root=Path(os.environ['EXPERIMENT_ROOT']);root.mkdir(parents=True,exist_ok=True)
r=Path(tempfile.mkdtemp(prefix='overfit_two_videos_',dir=root));(r/'code').mkdir();(r/'data').mkdir();(r/'logs').mkdir()
for name in ['stage3','stage2','scripts']:
 shutil.copytree(repo/name,r/'code'/name,ignore=shutil.ignore_patterns('__pycache__','examples'))
for name in ['tasks.jsonl','train.jsonl','dataset_manifest.json']:shutil.copy2(data/name,r/'data'/name)
(r/'data/video_map.json').write_text(json.dumps(paths,indent=2))
# Copy exact adapters and original provenance; runtime mapping points to these immutable run copies.
shutil.copytree(adapter,r/'stage2_adapter');shutil.copy2(cfgpath,r/'stage2_training_config.original.json')
if base:
 shutil.copytree(Path(base),r/'stage1_adapter');cfg['init_adapter']=str(r/'stage1_adapter')
(r/'stage2_training_config.runtime.json').write_text(json.dumps(cfg,indent=2))
manifest={'source_repo':str(repo),'source_stage2_adapter':str(adapter),'source_stage1_adapter':base,'tasks':len(tasks),'source_code':'working tree snapshot including uncommitted files','dry_run':os.environ.get('DRY_RUN','1')!='0','files':{str(p.relative_to(r)):hashlib.sha256(p.read_bytes()).hexdigest() for p in r.rglob('*') if p.is_file()}}
(r/'submission.json').write_text(json.dumps(manifest,indent=2));print(r)
PY
)
export REPO="$EXPERIMENT/code" DATA_DIR="$EXPERIMENT/data" VIDEO_MAP="$EXPERIMENT/data/video_map.json"
export RUN_DIR="$EXPERIMENT/results" STAGE2_ADAPTER="$EXPERIMENT/stage2_adapter" STAGE2_TRAINING_CONFIG="$EXPERIMENT/stage2_training_config.runtime.json"
if [[ -d "$EXPERIMENT/stage1_adapter" ]]; then export STAGE1_ADAPTER="$EXPERIMENT/stage1_adapter"; else unset STAGE1_ADAPTER; fi
PARTITIONS=$(cd "$REPO" && python -B -m scripts.gpu_profiles "$GPU_TYPE")
command=(sbatch --parsable --export=ALL --job-name=stage3-overfit-2v --partition="$PARTITIONS" --qos=gpu
 --nodes=1 --ntasks=1 --gres=gpu:1 --cpus-per-task=8 --mem=120G --time="${WALLTIME:-08:00:00}"
 --chdir="$REPO" --output="$EXPERIMENT/logs/%j.out" --error="$EXPERIMENT/logs/%j.err"
 "$REPO/scripts/slurm/run_stage3_overfit.sbatch")
if [[ "${DRY_RUN:-1}" != 0 ]]; then
 printf '%q ' "${command[@]}"; printf '\nPrepared only; no Slurm job submitted.\n'; echo "$EXPERIMENT"; exit 0
fi
JOB_RAW=$("${command[@]}");JOB_ID=${JOB_RAW%%;*};printf '%s\n' "$JOB_ID" > "$EXPERIMENT/job_id.txt"
echo "Job: $JOB_ID";echo "Experiment: $EXPERIMENT"
