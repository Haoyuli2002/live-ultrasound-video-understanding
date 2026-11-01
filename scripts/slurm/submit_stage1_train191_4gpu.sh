#!/usr/bin/env bash
# Submit the rebuilt v2.1 dataset: 191 videos, four full GPUs, one epoch.
set -euo pipefail
[[ $# == 0 || ( $# == 1 && $1 == --dry-run ) ]] || { echo 'Usage: submit_stage1_train191_4gpu.sh [--dry-run]' >&2; exit 2; }
SOURCE_REPO=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)
cd "$SOURCE_REPO"
source "${CONDA_SH:-$HOME/miniconda3/etc/profile.d/conda.sh}"
conda activate "${CONDA_ENV:-ultrasound}"
export DATA=${DATA:-/dss/mcmlscratch/04/ge75vid2/haoyu/live-ultrasound-video-understanding}
export TRAIN_DATASET_DIR=${TRAIN_DATASET_DIR:-$DATA/cluster_data/stage1_datasets/stage1_train193_v21_largev3_mig20_20261029T173710Z/dataset_train191_ready}
export TRAIN_JSONL="$TRAIN_DATASET_DIR/samples.jsonl"
export VIDEO_MAP="$TRAIN_DATASET_DIR/video_map.json"
export INDEPENDENT_EVAL_MAP=${INDEPENDENT_EVAL_MAP:-$DATA/cluster_data/splits/eval_full295_asr_keep_videos.json}
# Set all training settings explicitly so an earlier smoke shell cannot leak in.
export MODEL_NAME=Qwen/Qwen3.5-2B NPROC_PER_NODE=4 EPOCHS=1 MAX_STEPS=-1
export FRAME_BUDGET=30 FRAME_SIZE=224 FRAME_SAMPLING=recent_sparse RECENT_SECONDS=15
export GRADIENT_ACCUMULATION_STEPS=2 DATALOADER_NUM_WORKERS=2
export SAVE_STEPS=50 LOGGING_STEPS=10 VALIDATION_FRACTION=0
export DTYPE=bf16 ATTN_IMPLEMENTATION=sdpa STAGE1_SHARING_STRATEGY=file_descriptor
export OMP_NUM_THREADS=2 TOKENIZERS_PARALLELISM=false
export GPU_TYPE=${GPU_TYPE:-auto}
export WALLTIME=${WALLTIME:-24:00:00}
GPU_PARTITIONS=$(python -m scripts.gpu_profiles "$GPU_TYPE")
JOB_NAME=stage1-v21-4gpu
if [[ ${1:-} != --dry-run ]]; then
  ACTIVE=$(squeue -h -u "$USER" --name="$JOB_NAME" -o '%i %T')
  [[ -z "$ACTIVE" ]] || { echo "An existing $JOB_NAME job is active: $ACTIVE" >&2; exit 1; }
fi
python - <<'PY'
from collections import Counter, defaultdict
import json, os
from pathlib import Path
root=Path(os.environ['TRAIN_DATASET_DIR'])
if json.loads((root/'report.json').read_text())['status']!='ready':
    raise SystemExit('Dataset report is not ready')
rows=[json.loads(line) for line in Path(os.environ['TRAIN_JSONL']).read_text().splitlines() if line.strip()]
mapping=json.loads(Path(os.environ['VIDEO_MAP']).read_text())
excluded=json.loads(Path(os.environ['INDEPENDENT_EVAL_MAP']).read_text())
videos={r['video_id'] for r in rows}
groups=Counter((r['video_id'],r['sentence_id']) for r in rows)
if (len(videos),len(groups),len(rows))!=(191,7408,22224):
    raise SystemExit(f'Wrong rebuilt dataset: videos={len(videos)}, targets={len(groups)}, rows={len(rows)}')
if set(groups.values())!={3}: raise SystemExit('Broken three-condition groups')
paired=defaultdict(dict)
for row in rows:
    key=(row['video_id'],row['sentence_id'])
    if row['condition'] in paired[key]: raise SystemExit('Duplicate condition in a target group')
    paired[key][row['condition']]=row
for group in paired.values():
    if set(group)!={'before_with_asr','through_with_asr','before_mask_asr'}:
        raise SystemExit('Incomplete three-condition group')
    before,through,masked=(group[k] for k in ('before_with_asr','through_with_asr','before_mask_asr'))
    if (len({r['target'] for r in group.values()})!=1 or not before['historical_asr']
            or before['historical_asr']!=through['historical_asr'] or masked['historical_asr']
            or not masked['historical_asr_masked'] or before['video_window']!=masked['video_window']
            or before['video_window']!=[0,before['sentence_window'][0]]
            or through['video_window']!=[0,through['sentence_window'][1]]):
        raise SystemExit('Three-condition semantics mismatch')
if not videos.issubset(mapping): raise SystemExit('Missing video-map entries')
if any(not Path(mapping[v]).is_file() for v in videos): raise SystemExit('Missing training videos')
def resolved(raw):
    p=Path(raw)
    return str((p if p.is_absolute() else Path(os.environ['DATA'])/p).resolve())
if videos.intersection(excluded) or {resolved(mapping[v]) for v in videos}.intersection(
        resolved(p) for p in excluded.values()):
    raise SystemExit('Independent evaluation overlaps training')
for r in rows:
    a,b=r['sentence_window']
    if b-a<5 or len(r['target'].split())<10: raise SystemExit('Rebuilt target violates minimum policy')
print('Preflight OK: 191 videos, 7408 targets, 22224 rows; independent evaluation is disjoint.')
PY
RUN_PARENT="$DATA/cluster_data/experiments/stage1"
mkdir -p "$RUN_PARENT" "$SOURCE_REPO/logs"
export OUTPUT_DIR=$(mktemp -d "$RUN_PARENT/stage1_v21_train191_2b_4gpu_f30_e1_$(date -u +%Y%m%dT%H%M%SZ)_XXXXXX")
export RUN_NAME=$(basename "$OUTPUT_DIR")
REVISION=$(git rev-parse HEAD)
export REPO="$OUTPUT_DIR/code_${REVISION:0:12}"
mkdir "$REPO"
git archive "$REVISION" stage1 scripts | tar -x -C "$REPO"
{
  printf '# Frozen settings for this training run; submission creates a new run.\n'
  for name in DATA TRAIN_DATASET_DIR TRAIN_JSONL VIDEO_MAP INDEPENDENT_EVAL_MAP OUTPUT_DIR RUN_NAME REPO MODEL_NAME NPROC_PER_NODE EPOCHS MAX_STEPS FRAME_BUDGET FRAME_SIZE FRAME_SAMPLING RECENT_SECONDS GRADIENT_ACCUMULATION_STEPS DATALOADER_NUM_WORKERS SAVE_STEPS LOGGING_STEPS VALIDATION_FRACTION DTYPE ATTN_IMPLEMENTATION STAGE1_SHARING_STRATEGY OMP_NUM_THREADS GPU_TYPE WALLTIME; do
    printf 'export %s=%q\n' "$name" "${!name}"
  done
} > "$OUTPUT_DIR/submit_env.sh"
python - "$OUTPUT_DIR" "$REVISION" <<'PY'
import hashlib,json,os,sys
from pathlib import Path
config={'code_commit':sys.argv[2],'status':'prepared','world_size':4,'epochs':1,'frame_budget':30,
        'frame_sampling':'recent_sparse','recent_seconds':15,'effective_batch_size':8,
        'validation_fraction':0,'independent_eval_map':os.environ['INDEPENDENT_EVAL_MAP'],
        'gpu_type':os.environ['GPU_TYPE'],'walltime':os.environ['WALLTIME'],
        'output_dir':sys.argv[1],'train_jsonl':os.environ['TRAIN_JSONL'],'video_map':os.environ['VIDEO_MAP'],
        'dataset_hashes':{name:hashlib.sha256((Path(os.environ['TRAIN_DATASET_DIR'])/name).read_bytes()).hexdigest()
                          for name in ('samples.jsonl','video_map.json','report.json')}}
Path(sys.argv[1],'submission.json').write_text(json.dumps(config,indent=2)+'\n')
PY
command=(sbatch --parsable --export=ALL --job-name="$JOB_NAME" --partition="$GPU_PARTITIONS"
  --qos=gpu --nodes=1 --ntasks=1 --gres=gpu:4 --cpus-per-task=16 --mem=160G
  --time="$WALLTIME" --chdir="$REPO"
  --output="$SOURCE_REPO/logs/stage1_pretrain_%j.out" --error="$SOURCE_REPO/logs/stage1_pretrain_%j.err"
  "$REPO/scripts/slurm/run_stage1_train_ddp.sbatch")
if [[ ${1:-} == --dry-run ]]; then
  printf '%q ' "${command[@]}"; printf '\n'
  echo "Dry run; no job submitted. Prepared run: $OUTPUT_DIR"
  exit 0
fi
JOB_ID=$("${command[@]}"); JOB_ID=${JOB_ID%%;*}
[[ "$JOB_ID" =~ ^[0-9]+$ ]] || { echo "Unexpected sbatch result: $JOB_ID" >&2; exit 1; }
printf '%s\n' "$JOB_ID" > "$OUTPUT_DIR/job_id.txt"
python - "$OUTPUT_DIR/submission.json" "$JOB_ID" <<'PY'
import json,sys
from pathlib import Path
path=Path(sys.argv[1]);value=json.loads(path.read_text());value.update(job_id=sys.argv[2],status='submitted')
path.write_text(json.dumps(value,indent=2)+'\n')
PY
echo "Job: $JOB_ID"
echo "Output: $OUTPUT_DIR"
echo "squeue -j $JOB_ID"
echo "tail -F logs/stage1_pretrain_${JOB_ID}.out logs/stage1_pretrain_${JOB_ID}.err"
