#!/usr/bin/env bash
# Freeze one common input/code snapshot, then submit two independent GPU jobs.
set -euo pipefail
SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
export REPO=${REPO:-$(cd "$SCRIPT_DIR/../.." && pwd)}
export DATA=${DATA:-/dss/mcmlscratch/04/ge75vid2/haoyu/live-ultrasound-video-understanding}
export CONDA_SH=${CONDA_SH:-$HOME/miniconda3/etc/profile.d/conda.sh}
export CONDA_ENV=${CONDA_ENV:-ultrasound}
source "$CONDA_SH"
conda activate "$CONDA_ENV"
cd "$REPO"
unset STAGE1_ADAPTER STAGE2_ADAPTER INIT_ADAPTER
export MODEL_NAME=Qwen/Qwen3.5-2B FRAME_SIZE=224 ATTN_IMPLEMENTATION=sdpa
export GPU_TYPE=${GPU_TYPE:-auto} EPOCHS=${EPOCHS:-20}
export LEARNING_RATE=${LEARNING_RATE:-1e-4} SEED=${SEED:-20261104}
export WALLTIME=${WALLTIME:-04:00:00} MAX_NEW_TOKENS=${MAX_NEW_TOKENS:-160}
[[ "$EPOCHS" =~ ^[1-9][0-9]*$ && "$MAX_NEW_TOKENS" =~ ^[1-9][0-9]*$ ]] || exit 2
PARTITIONS=$(python -m scripts.gpu_profiles "$GPU_TYPE")
export PARTITIONS
# A submitted job must contain exactly the reviewed, committed implementation.
git diff --quiet HEAD -- stage2/train_overfit_ab.py stage2/overfit_ab_report.py \
  stage2/evaluate.py stage2/reconstruct.py stage2/examples/P3FoSFIrfQA \
  scripts/slurm/submit_stage2_overfit_ab.sh scripts/slurm/run_stage2_overfit_ab.sbatch || {
  echo 'Commit or restore changes to the experiment files before submission.' >&2; exit 2;
}
mkdir -p "$DATA/cluster_data/experiments/stage2" "$REPO/logs"
PAIR_DIR=$(mktemp -d "$DATA/cluster_data/experiments/stage2/overfit_AB_P3_$(date -u +%Y%m%dT%H%M%SZ)_XXXXXX")
export PAIR_DIR
export CODE_DIR="$PAIR_DIR/code"
mkdir "$CODE_DIR"
git archive HEAD | tar -xf - -C "$CODE_DIR"
git rev-parse HEAD > "$PAIR_DIR/code_commit.txt"
cd "$CODE_DIR"
python - <<'PY'
import hashlib,json,math,os,shutil
from pathlib import Path
from stage2.data import load_blocks
p=Path(os.environ['PAIR_DIR']); code=Path(os.environ['CODE_DIR'])
for name in ['stage2/train_overfit_ab.py','stage2/overfit_ab_report.py',
             'scripts/slurm/run_stage2_overfit_ab.sbatch']:
 if not (code/name).is_file(): raise SystemExit(f'Missing committed experiment file: {name}')
if not math.isfinite(float(os.environ['LEARNING_RATE'])) or float(os.environ['LEARNING_RATE'])<=0:
 raise SystemExit('LEARNING_RATE must be finite and positive')
int(os.environ['SEED'])
data=Path(os.environ['DATA']); vid='P3FoSFIrfQA'
labels=code/'stage2/examples/P3FoSFIrfQA/blocks.jsonl'
blocks=load_blocks(labels)
assert set(blocks)=={vid}
assert [r['block_window'] for r in blocks[vid]]==[[0,60],[60,120]]
assert sum(len(r['local_sub_summaries']) for r in blocks[vid])==12
explicit=os.environ.get('VIDEO_FILE')
candidates=[Path(explicit)] if explicit else [data/'cluster_data/videos/train_full295'/f'{vid}.mp4']
if not explicit:
 for mp in [data/'cluster_data/splits/train_full295_qwen35_pretrain_keep_videos.json',
            data/'cluster_data/stage2/train_full295_video_paths.json']:
  if mp.is_file():
   value=json.loads(mp.read_text()).get(vid)
   if isinstance(value,str):
    path=Path(value).expanduser(); candidates.append(path if path.is_absolute() else data/path)
video=next((v.resolve() for v in candidates if v.is_file()),None)
if video is None: raise SystemExit('P3 video missing; set VIDEO_FILE to its absolute path.')
jobs=[]
for mode,steps,kinds in [('short_only',12,['short']),('independent',16,['short','block','global'])]:
 run=p/mode; run.mkdir()
 shutil.copy2(labels,run/'blocks.jsonl')
 shutil.copy2(labels.with_name('provenance.json'),run/'label_provenance.json')
 (run/'video_map.json').write_text(json.dumps({vid:str(video)},indent=2))
 plan={k:os.environ[k] for k in ['MODEL_NAME','EPOCHS','LEARNING_RATE','SEED','FRAME_SIZE',
                               'GPU_TYPE','WALLTIME','MAX_NEW_TOKENS','PARTITIONS']}
 plan.update(mode=mode,run_dir=str(run),code_dir=str(code),init_adapter=None,
             initialization='official_checkpoint',video_id=vid,window=[0,120],gpu_count=1,
             updates_per_epoch=steps,expected_updates=steps*int(os.environ['EPOCHS']),
             evaluated_kinds=kinds,expected_predictions=steps*3,
             label_status='draft_debug_only',
             samples_sha256=hashlib.sha256(labels.read_bytes()).hexdigest())
 (run/'submission.json').write_text(json.dumps(plan,indent=2))
 jobs.append({'mode':mode,'run_dir':str(run),'job_id':None})
manifest={'status':'prepared','pair_dir':str(p),'jobs':jobs,
          'code_commit':(p/'code_commit.txt').read_text().strip()}
(p/'experiments.json').write_text(json.dumps(manifest,indent=2))
print(json.dumps(manifest,indent=2),flush=True)
PY
for MODE in short_only independent; do
  export MODE RUN_DIR="$PAIR_DIR/$MODE"
  export TRAIN_JSONL="$RUN_DIR/blocks.jsonl" VIDEO_MAP="$RUN_DIR/video_map.json"
  export OUTPUT_DIR="$RUN_DIR/training"
  command=(sbatch --parsable --export=ALL --job-name="s2-ab-$MODE" \
    --partition="$PARTITIONS" --qos=gpu --nodes=1 --ntasks=1 --gres=gpu:1 \
    --cpus-per-task=8 --mem=120G --time="$WALLTIME" --chdir="$CODE_DIR" \
    --output="$REPO/logs/stage2_ab_${MODE}_%j.out" \
    --error="$REPO/logs/stage2_ab_${MODE}_%j.err" \
    "$CODE_DIR/scripts/slurm/run_stage2_overfit_ab.sbatch")
  if [[ "${DRY_RUN:-0}" == 1 ]]; then
    printf '%q ' "${command[@]}"; printf '\n'
    continue
  fi
  JOB_RAW=$("${command[@]}")
  export JOB_ID=${JOB_RAW%%;*}
  [[ "$JOB_ID" =~ ^[0-9]+$ ]] || { echo "Unexpected sbatch response: $JOB_RAW" >&2; exit 1; }
  printf '%s\n' "$JOB_ID" > "$RUN_DIR/job_id.txt"
  python - <<'PY'
import json,os
from pathlib import Path
p=Path(os.environ['PAIR_DIR'])/'experiments.json'; m=json.loads(p.read_text())
for j in m['jobs']:
 if j['mode']==os.environ['MODE']: j['job_id']=os.environ['JOB_ID']
m['status']='submitted' if all(j['job_id'] for j in m['jobs']) else 'partially_submitted'
p.write_text(json.dumps(m,indent=2))
PY
  echo "$MODE job: $JOB_ID"
  echo "tail -F $REPO/logs/stage2_ab_${MODE}_${JOB_ID}.out $REPO/logs/stage2_ab_${MODE}_${JOB_ID}.err"
done
echo "Pair: $PAIR_DIR"
echo "python -m stage2.overfit_ab_report --pair-dir '$PAIR_DIR'"
if [[ "${DRY_RUN:-0}" == 1 ]]; then
  echo 'Dry run: both plans prepared; no jobs submitted.'
fi
