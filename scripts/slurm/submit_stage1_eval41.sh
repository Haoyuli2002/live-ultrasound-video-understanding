#!/usr/bin/env bash
set -euo pipefail
if [[ $# -gt 1 || ( $# -eq 1 && $1 != --dry-run ) ]]; then
  echo 'Usage: submit_stage1_eval41.sh [--dry-run]' >&2
  exit 2
fi
CHECKOUT=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)
cd "$CHECKOUT"
source "$HOME/miniconda3/etc/profile.d/conda.sh"
conda activate ultrasound
export DATA=${DATA:-/dss/mcmlscratch/04/ge75vid2/haoyu/live-ultrasound-video-understanding}
export SPLIT=eval_full295 WHISPER_MODEL=large-v3 LANGUAGE=en
export ALLOW_EMPTY_VIDEOS=1 LIMIT_VIDEOS=''
export VIDEO_ROOT="$DATA"
RUNS="$DATA/cluster_data/stage1_datasets"
mkdir -p "$RUNS"
export RUN_ROOT=$(mktemp -d "$RUNS/stage1_eval41_v21_largev3_mig20_$(date -u +%Y%m%dT%H%M%SZ)_XXXXXX")
export REPO="$RUN_ROOT/code" ASR_DIR="$RUN_ROOT/asr" DATASET_DIR="$RUN_ROOT/dataset"
export VIDEO_MAP="$RUN_ROOT/eval41_video_map.json"
mkdir -p "$REPO" "$CHECKOUT/logs"
git archive HEAD stage1 scripts | tar -x -C "$REPO"
git rev-parse HEAD > "$RUN_ROOT/code_commit.txt"
python - <<'PY'
import hashlib,json,os
from pathlib import Path
import cv2
base=Path(os.environ['DATA']);root=Path(os.environ['RUN_ROOT'])
original=base/'cluster_data/splits/eval_full295_asr_keep_videos.json'
values=json.loads(original.read_text())
selected={k:str((Path(v) if Path(v).is_absolute() else base/v).resolve()) for k,v in values.items()}
assert len(selected)==41
assert all(Path(v).is_file() and Path(v).stem==k for k,v in selected.items())
trainroot=base/'cluster_data/stage1_datasets/stage1_train193_v21_largev3_mig20_20261029T173710Z'
train=json.loads((trainroot/'train193_video_map.json').read_text())
assert not (selected.keys() & train.keys())
assert not (set(selected.values()) & {str(Path(v).resolve()) for v in train.values()})
old=json.loads((trainroot/'dataset_train191_ready/manifest.json').read_text())
for name,digest in old['implementation_sha256'].items():
    assert hashlib.sha256((Path(os.environ['REPO'])/'stage1'/name).read_bytes()).hexdigest()==digest,(name,'implementation mismatch')
durations={}
for vid,path in selected.items():
    reader=cv2.VideoCapture(path)
    try:
        fps=reader.get(cv2.CAP_PROP_FPS);count=reader.get(cv2.CAP_PROP_FRAME_COUNT)
        assert fps>0 and count>0,(vid,'invalid metadata')
        durations[vid]=count/fps
    finally:reader.release()
Path(os.environ['VIDEO_MAP']).write_text(json.dumps(selected,indent=2)+'\n')
preflight={'selected_videos':len(selected),'video_hours':sum(durations.values())/3600,
           'train_video_ids_disjoint':True,'train_paths_disjoint':True,
           'same_preparation_implementation_as_train':True,'video_durations':durations,
           'policy':{'min_seconds':5,'min_words':10,'max_seconds':None,'max_words':None},
           'empty_video_handling':'exclude from samples; retain in audit'}
(root/'input_preflight.json').write_text(json.dumps(preflight,indent=2)+'\n')
print(json.dumps({k:v for k,v in preflight.items() if k!='video_durations'},indent=2))
PY
{
  for name in REPO DATA RUN_ROOT ASR_DIR DATASET_DIR VIDEO_MAP VIDEO_ROOT SPLIT WHISPER_MODEL LANGUAGE ALLOW_EMPTY_VIDEOS LIMIT_VIDEOS; do
    printf 'export %s=%q\n' "$name" "${!name}"
  done
} > "$RUN_ROOT/submit_env.sh"
if [[ ${1:-} == --dry-run ]]; then
  echo "Preflight passed. Dry run only; no job submitted. Run: $RUN_ROOT"
  exit 0
fi
JOB_ID=$(sbatch --parsable --export=ALL --job-name=stage1-data-eval41 \
  --time=04:00:00 --partition=lrz-dgx-a100-40x8-mig --qos=mig \
  --gres=gpu:3g.20gb:1 --cpus-per-task=4 --mem=16G --chdir="$REPO" \
  --output="$CHECKOUT/logs/stage1_data_%j.out" --error="$CHECKOUT/logs/stage1_data_%j.err" \
  "$REPO/scripts/slurm/run_stage1_prepare_v2.sbatch")
export JOB_ID=${JOB_ID%%;*}
printf '%s\n' "$JOB_ID" > "$RUN_ROOT/job_id.txt"
python - <<'PY'
import json,os
from pathlib import Path
run=Path(os.environ['RUN_ROOT'])
metadata={'job_id':os.environ['JOB_ID'],'run_root':str(run),'dataset_dir':os.environ['DATASET_DIR'],
          'code_commit':(run/'code_commit.txt').read_text().strip(),
          'selected_videos':41,'resource':'1 A100 MIG 20GB, 4 CPUs, 16GB RAM',
          'walltime':'04:00:00','split':'independent evaluation','model':'Whisper large-v3',
          'policy':{'min_seconds':5,'min_words':10,'max_seconds':None,'max_words':None}}
(run/'submission.json').write_text(json.dumps(metadata,indent=2)+'\n')
print('SUBMISSION_JSON='+json.dumps(metadata))
PY
squeue --start -j "$JOB_ID"
