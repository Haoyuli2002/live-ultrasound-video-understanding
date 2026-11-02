#!/usr/bin/env bash
set -euo pipefail
SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
export REPO=${REPO:-$(cd "$SCRIPT_DIR/../.." && pwd)}
LABEL_DIR="$REPO/stage2/examples/P3FoSFIrfQA"
export DATA=${DATA:-/dss/mcmlscratch/04/ge75vid2/haoyu/live-ultrasound-video-understanding}
export CONDA_SH=${CONDA_SH:-$HOME/miniconda3/etc/profile.d/conda.sh}
export CONDA_ENV=${CONDA_ENV:-ultrasound}
source "$CONDA_SH"
conda activate "$CONDA_ENV"
cd "$REPO"
export STAGE1_ADAPTER=${STAGE1_ADAPTER:-$DATA/cluster_data/experiments/stage1/stage1_full186_2b_4gpu_f30_e1_20261025_052451/final}
[[ -f "$STAGE1_ADAPTER/adapter_config.json" ]] || {
  echo 'Set STAGE1_ADAPTER to an existing completed Stage 1 adapter directory.' >&2; exit 2;
}
export GPU_TYPE=${GPU_TYPE:-auto}
export MODEL_NAME=Qwen/Qwen3.5-2B
export EPOCHS=${EPOCHS:-20}
export LEARNING_RATE=${LEARNING_RATE:-1e-4}
export FRAME_SIZE=224
export ATTN_IMPLEMENTATION=sdpa
export WALLTIME=${WALLTIME:-04:00:00}
[[ "$EPOCHS" =~ ^[1-9][0-9]*$ ]] || exit 2
mkdir -p "$DATA/cluster_data/experiments/stage2" "$REPO/logs"
export RUN_DIR=$(mktemp -d "$DATA/cluster_data/experiments/stage2/overfit_P3_$(date -u +%Y%m%dT%H%M%SZ)_XXXXXX")
export CODE_DIR="$RUN_DIR/code"
mkdir "$CODE_DIR"
git archive HEAD | tar -xf - -C "$CODE_DIR"
git rev-parse HEAD > "$RUN_DIR/code_commit.txt"
cp "$LABEL_DIR/blocks.jsonl" "$RUN_DIR/blocks.jsonl"
cp "$LABEL_DIR/provenance.json" "$RUN_DIR/label_provenance.json"
cp "$SCRIPT_DIR/run_stage2_overfit.sbatch" "$RUN_DIR/run.sbatch"
export TRAIN_JSONL="$RUN_DIR/blocks.jsonl"
export VIDEO_MAP="$RUN_DIR/video_map.json"
export OUTPUT_DIR="$RUN_DIR/training"
cd "$CODE_DIR"
python - <<'PY'
import hashlib,json,os
from pathlib import Path
from stage2.data import load_blocks
data=Path(os.environ['DATA']); vid='P3FoSFIrfQA'
blocks=load_blocks(os.environ['TRAIN_JSONL'])
assert set(blocks)=={vid} and len(blocks[vid])==2
assert [r['block_window'] for r in blocks[vid]]==[[0,60],[60,120]]
explicit=os.environ.get('VIDEO_FILE')
candidates=[Path(explicit)] if explicit else []
candidates += [data/'cluster_data/videos/train_full295'/f'{vid}.mp4']
maps=[data/'cluster_data/splits/train_full295_qwen35_pretrain_keep_videos.json',data/'cluster_data/stage2/train_full295_video_paths.json']
for p in maps:
 if p.is_file():
  value=json.loads(p.read_text()).get(vid)
  if isinstance(value,str):
   q=Path(value).expanduser(); candidates.append(q if q.is_absolute() else data/q)
video=next((p.resolve() for p in candidates if p.is_file()),None)
if video is None: raise SystemExit('P3 video missing: set VIDEO_FILE to its absolute cluster path.')
Path(os.environ['VIDEO_MAP']).write_text(json.dumps({vid:str(video)},indent=2))
cfg={k:os.environ[k] for k in ['RUN_DIR','CODE_DIR','TRAIN_JSONL','VIDEO_MAP','OUTPUT_DIR','STAGE1_ADAPTER','MODEL_NAME','EPOCHS','LEARNING_RATE','FRAME_SIZE','GPU_TYPE','WALLTIME']}
cfg.update(video_id=vid,window=[0,120],blocks=2,updates_per_epoch=12,expected_updates=12*int(cfg['EPOCHS']),label_status='draft_debug_only',gpu_count=1,samples_sha256=hashlib.sha256(Path(cfg['TRAIN_JSONL']).read_bytes()).hexdigest())
Path(os.environ['RUN_DIR'],'submission.json').write_text(json.dumps(cfg,indent=2))
print(json.dumps(cfg,indent=2))
PY
PARTITIONS=$(python -m scripts.gpu_profiles "$GPU_TYPE")
command=(sbatch --parsable --export=ALL --job-name=stage2-overfit-P3 \
  --partition="$PARTITIONS" --qos=gpu --nodes=1 --ntasks=1 --gres=gpu:1 \
  --cpus-per-task=8 --mem=120G --time="$WALLTIME" --chdir="$CODE_DIR" \
  --output="$REPO/logs/stage2_overfit_%j.out" --error="$REPO/logs/stage2_overfit_%j.err" \
  "$RUN_DIR/run.sbatch")
if [[ "${DRY_RUN:-0}" == 1 ]]; then
  printf "%q " "${command[@]}"; printf "\n"
  echo "Dry run: inputs validated and plan saved; no job submitted."
  exit 0
fi
JOB_RAW=$("${command[@]}")
JOB_ID=${JOB_RAW%%;*}
printf '%s\n' "$JOB_ID" > "$RUN_DIR/job_id.txt"
echo "Job: $JOB_ID"
echo "Run: $RUN_DIR"
echo "squeue -j $JOB_ID"
echo "tail -F $REPO/logs/stage2_overfit_${JOB_ID}.out $REPO/logs/stage2_overfit_${JOB_ID}.err"
