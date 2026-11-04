#!/usr/bin/env bash
set -euo pipefail
export REPO=${REPO:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}
export DATA=${DATA:-/dss/mcmlscratch/04/ge75vid2/haoyu/live-ultrasound-video-understanding}
export CONDA_SH=${CONDA_SH:-$HOME/miniconda3/etc/profile.d/conda.sh} CONDA_ENV=${CONDA_ENV:-ultrasound}
source "$CONDA_SH"
conda activate "$CONDA_ENV"
cd "$REPO"
git diff --quiet HEAD -- stage2 scripts/slurm/submit_stage2_qformer_smoke.sh scripts/slurm/run_stage2_qformer_smoke.sbatch || {
  echo 'Commit the training code before freezing a smoke run.' >&2; exit 2;
}
: "${SMOKE_INPUT_DIR:?Set SMOKE_INPUT_DIR to the accepted P3 video export}"
export STAGE1_ADAPTER=${STAGE1_ADAPTER:-$DATA/cluster_data/experiments/stage1/stage1_v21_train191_2b_4gpu_f30_e1_20261101T071541Z_sCe9Fb/final}
export EPOCHS=${EPOCHS:-3} WALLTIME=${WALLTIME:-02:00:00}
PARTITIONS=$(python -m scripts.gpu_profiles "${GPU_TYPE:-auto}")
if [[ ${DRY_RUN:-0} != 1 ]]; then
  ACTIVE=$(squeue -h -u "$USER" --name=stage2-qformer-smoke -o '%i %T')
  [[ -z "$ACTIVE" ]] || { echo "Existing Q-Former smoke: $ACTIVE" >&2; exit 1; }
fi
mkdir -p "$DATA/cluster_data/experiments/stage2" logs
RUN_DIR=$(mktemp -d "$DATA/cluster_data/experiments/stage2/qformer_smoke_P3_$(date -u +%Y%m%dT%H%M%SZ)_XXXXXX")
export RUN_DIR CODE_DIR="$RUN_DIR/code"
mkdir "$CODE_DIR"
git archive HEAD | tar -xf - -C "$CODE_DIR"
git rev-parse HEAD > "$RUN_DIR/code_commit.txt"
cp "$SMOKE_INPUT_DIR/blocks.reviewed.jsonl" "$RUN_DIR/blocks.jsonl"
cp "$SMOKE_INPUT_DIR/manifest.json" "$RUN_DIR/acceptance.json"
export VIDEO_FILE=${VIDEO_FILE:-$DATA/cluster_data/videos/train_full295/P3FoSFIrfQA.mp4}
python - <<'PY'
import json,os
from pathlib import Path
from stage2.data import load_blocks
from stage2.provenance import digest
r=Path(os.environ['RUN_DIR']);a=json.loads((r/'acceptance.json').read_text())
assert a['video_id']=='P3FoSFIrfQA' and a['final_acceptance_complete'] and a['boundary_review_complete']
assert a['native_acceptance_complete'] or a.get('user_first_pass_acceptance',{}).get('accepted')
assert a['targets']==16 and a['block_rows']==2
v=Path(os.environ['VIDEO_FILE']).resolve();assert digest(v)==a['source_sha256']
blocks=load_blocks(r/'blocks.jsonl');assert list(blocks)==['P3FoSFIrfQA']
assert [x['block_window'] for x in blocks['P3FoSFIrfQA']]==[[0,60],[60,120]]
s=Path(os.environ['STAGE1_ADAPTER']);assert (s/'adapter_config.json').is_file()
assert (s/'adapter_model.safetensors').stat().st_size>0
e=int(os.environ['EPOCHS']);assert e>0
(r/'video_map.json').write_text(json.dumps({'P3FoSFIrfQA':str(v)},indent=2))
(r/'input_manifest.json').write_text(json.dumps({'source_sha256':a['source_sha256'],
 'labels_sha256':digest(r/'blocks.jsonl'),'acceptance_sha256':digest(r/'acceptance.json'),
 'stage1_adapter_sha256':digest(s/'adapter_model.safetensors'),'epochs':e,
 'training_updates':16*e,'generated_outputs':48,'full_dataset_training':False},indent=2))
PY
command=(sbatch --parsable --export=ALL --job-name=stage2-qformer-smoke --partition="$PARTITIONS" \
  --qos=gpu --nodes=1 --ntasks=1 --gres=gpu:1 --cpus-per-task=8 --mem=120G --time="$WALLTIME" \
  --chdir="$CODE_DIR" --output="$REPO/logs/stage2_qformer_smoke_%j.out" \
  --error="$REPO/logs/stage2_qformer_smoke_%j.err" "$CODE_DIR/scripts/slurm/run_stage2_qformer_smoke.sbatch")
if [[ ${DRY_RUN:-0} == 1 ]]; then printf '%q ' "${command[@]}"; printf '\n'; exit 0; fi
JOB=$("${command[@]}");JOB=${JOB%%;*}
printf '%s\n' "$JOB" > "$RUN_DIR/job_id.txt"
echo "Job: $JOB"
echo "Run: $RUN_DIR"
echo "tail -F $REPO/logs/stage2_qformer_smoke_$JOB.out $REPO/logs/stage2_qformer_smoke_$JOB.err"
