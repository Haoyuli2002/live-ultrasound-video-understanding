#!/usr/bin/env bash
# Run on LRZ: generate (submit only if needed), then pack after GPU jobs complete.
set -euo pipefail
MODE=${1:-}
[[ "$MODE" == generate || "$MODE" == pack ]] || { echo 'Usage: bash stage12_ar_results.sh generate|pack' >&2; exit 2; }
export REPO=${REPO:-$HOME/haoyu/live-ultrasound-video-understanding}
export DATA=${DATA:-/dss/mcmlscratch/04/ge75vid2/haoyu/live-ultrasound-video-understanding}
STATE="$DATA/cluster_data/evaluations/ar_stage12_20261105"
mkdir -p "$STATE"
source "${CONDA_SH:-$HOME/miniconda3/etc/profile.d/conda.sh}"
conda activate "${CONDA_ENV:-ultrasound}"
cd "$REPO"
export OUTPUT_DIR="$DATA/cluster_data/experiments/stage1/stage1_v21_train191_2b_4gpu_f30_e1_20261101T071541Z_sCe9Fb"
export EVAL_DIR=${EVAL_DIR:-$OUTPUT_DIR/eval_eval41_100_recent15}
export EVAL_JSONL="$DATA/cluster_data/stage1_datasets/stage1_eval41_v21_largev3_mig20_20261031T085938Z_Mc3hey/dataset/samples.jsonl"
export EVAL_VIDEO_MAP="$(dirname "$EVAL_JSONL")/video_map.json"
export MODEL_NAME=Qwen/Qwen3.5-2B MODE=smoke LIMIT_TARGETS=100
# Keep the shell action separately; MODE is also used by the Stage1 submitter.
ACTION=$1
export FRAME_BUDGET=30 FRAME_SIZE=224 FRAME_SAMPLING=recent_sparse RECENT_SECONDS=15
export VISUAL_CONTROLS=normal MAX_NEW_TOKENS=128 SUBMIT_JUDGE=0 EVAL_TIME=04:00:00
export TRAIN_RUN=${TRAIN_RUN:-$DATA/cluster_data/experiments/stage2/overfit_P3_20261102T142657Z_N2Dr0P}
if [[ -s "$STATE/stage2_run.txt" ]]; then
  RECON_RUN=$(cat "$STATE/stage2_run.txt")
else
  RECON_RUN="$TRAIN_RUN/reconstruct_final_20261103T144203Z_ewn13gn8"
fi
export RECON_RUN

stage1_complete() {
python - <<'PY'
import json, os
from pathlib import Path
from stage1.eval_io import indexed, digest
try:
 d=Path(os.environ['EVAL_DIR'])
 gt=indexed([json.loads(x) for x in (d/'samples.jsonl').read_text().splitlines() if x.strip()])
 assert gt
 for variant in ['base','lora']:
  p=d/f'{variant}_normal.jsonl'
  cfg=json.loads(Path(str(p)+'.config.json').read_text())
  assert cfg['eval_sha256']==digest(d/'samples.jsonl')
  assert cfg['frame_budget']==30 and cfg['frame_size']==224
  assert cfg['frame_sampling']=='recent_sparse' and cfg['recent_seconds']==15
  expected_adapter=str(Path(os.environ['OUTPUT_DIR'])/'final') if variant=='lora' else None
  assert cfg.get('adapter_path')==expected_adapter
  rows=indexed([json.loads(x) for x in p.read_text().splitlines() if x.strip()])
  assert rows.keys()==gt.keys()
  assert all(isinstance(r.get('prediction'),str) and r['target']==gt[k]['target'] and r['video_window']==gt[k]['video_window'] for k,r in rows.items())
except (OSError,ValueError,KeyError,AssertionError):raise SystemExit(1)
PY
}
stage2_complete() {
python - <<'PY'
import hashlib,json,os
from pathlib import Path
try:
 r=Path(os.environ['RECON_RUN']);d=r/'results'
 report=json.loads((d/'report.json').read_text());manifest=json.loads((d/'manifest.json').read_text())
 assert report['status']=='completed' and manifest['reference_used_in_decoding'] is False
 assert manifest['decoding']=='greedy_full_prefix_no_cache'
 assert manifest['labels_sha256']==hashlib.sha256((Path(os.environ['TRAIN_RUN'])/'blocks.jsonl').read_bytes()).hexdigest()
 assert Path(manifest['adapter_path']).resolve()==(Path(os.environ['TRAIN_RUN'])/'training/final').resolve()
 rows=[json.loads(x) for x in (d/'predictions.jsonl').read_text().splitlines() if x.strip()]
 assert len(rows)==report['records'] and rows
 assert all(isinstance(x.get('prediction'),str) or 'skipped' in x for x in rows)
except (OSError,ValueError,KeyError,AssertionError):raise SystemExit(1)
PY
}
active() {
 local job_file=$1 job
 [[ -s "$job_file" ]] || return 1
 job=$(cat "$job_file")
 [[ "$job" =~ ^[0-9]+$ ]] || { echo "Invalid job record: $job_file" >&2; exit 2; }
 [[ "$ACTIVE_IDS" == *"|$job|"* ]]
}

if [[ "$ACTION" == generate ]]; then
 # Stop on scheduler query failure rather than accidentally duplicating a job.
 ACTIVE_IDS="|$(squeue -h -u "$USER" -o '%i' | tr '\n' '|')"
 bash scripts/slurm/submit_stage1_eval.sh --dry-run
 printf '%s\n' "$EVAL_DIR" > "$STATE/stage1_dir.txt"
 if stage1_complete; then
  echo "Stage1: reusing completed autoregressive outputs: $EVAL_DIR"
 elif active "$EVAL_DIR/eval_job_id.txt"; then
  echo "Stage1: existing job $(cat "$EVAL_DIR/eval_job_id.txt") is still active"
 else
  bash scripts/slurm/submit_stage1_eval.sh | tee "$STATE/stage1_submission_$(date -u +%Y%m%dT%H%M%SZ).log"
 fi
 if stage2_complete; then
  printf '%s\n' "$RECON_RUN" > "$STATE/stage2_run.txt"
  echo "Stage2: reusing completed P3 overfit reconstruction: $RECON_RUN/results"
 elif active "$RECON_RUN/job_id.txt"; then
  printf '%s\n' "$RECON_RUN" > "$STATE/stage2_run.txt"
  echo "Stage2: existing job $(cat "$RECON_RUN/job_id.txt") is still active"
 else
  LOG="$STATE/stage2_submission_$(date -u +%Y%m%dT%H%M%SZ).log"
  GPU_TYPE=auto WALLTIME=01:00:00 CHECKPOINT=final MAX_NEW_TOKENS=160 DRY_RUN=0 \
   bash scripts/slurm/submit_stage2_reconstruct.sh | tee "$LOG"
  sed -n 's/^Run: //p' "$LOG" > "$STATE/stage2_run.txt"
  [[ -s "$STATE/stage2_run.txt" ]]
 fi
 echo 'Wait for the generation jobs to finish, then run this script with: pack'
 exit 0
fi

[[ -s "$STATE/stage1_dir.txt" && -s "$STATE/stage2_run.txt" ]] || { echo 'Run generate first to record the selected outputs.' >&2;exit 2; }
export EVAL_DIR=$(cat "$STATE/stage1_dir.txt") RECON_RUN=$(cat "$STATE/stage2_run.txt")
stage1_complete || { echo "Stage1 results incomplete: $EVAL_DIR" >&2;exit 2; }
stage2_complete || { echo "Stage2 results incomplete: $RECON_RUN/results" >&2;exit 2; }
BUNDLE=$(mktemp -d "$STATE/export_XXXXXX")
mkdir "$BUNDLE/stage1" "$BUNDLE/stage2" "$BUNDLE/export_checkout"
for file in samples.jsonl base_normal.jsonl lora_normal.jsonl base_normal.jsonl.config.json lora_normal.jsonl.config.json eval_job_id.txt; do
 [[ ! -f "$EVAL_DIR/$file" ]] || cp "$EVAL_DIR/$file" "$BUNDLE/stage1/"
done
for file in training_config.json video_split.json; do
 [[ ! -f "$OUTPUT_DIR/$file" ]] || cp "$OUTPUT_DIR/$file" "$BUNDLE/stage1/"
done
cp "$EVAL_VIDEO_MAP" "$BUNDLE/stage1/video_map.remote.json"
for file in predictions.jsonl manifest.json report.json review.txt; do
 [[ ! -f "$RECON_RUN/results/$file" ]] || cp "$RECON_RUN/results/$file" "$BUNDLE/stage2/"
done
cp "$TRAIN_RUN/blocks.jsonl" "$TRAIN_RUN/video_map.json" "$BUNDLE/stage2/"
cp "$TRAIN_RUN/training/training_config.json" "$BUNDLE/stage2/training_config.json"
[[ ! -d "$RECON_RUN/code" ]] || cp -R "$RECON_RUN/code" "$BUNDLE/stage2/generation_code"
# This is the checkout at export time; generation configs carry the actual source hashes.
cp -R stage1 "$BUNDLE/export_checkout/stage1"
git rev-parse HEAD > "$BUNDLE/export_checkout/commit.txt"
python - "$BUNDLE" "$EVAL_DIR" "$RECON_RUN" <<'PY'
import hashlib,json,sys
from pathlib import Path
r=Path(sys.argv[1])
files={str(p.relative_to(r)):hashlib.sha256(p.read_bytes()).hexdigest() for p in r.rglob('*') if p.is_file()}
(r/'export_manifest.json').write_text(json.dumps(dict(stage1_source=sys.argv[2],stage2_source=sys.argv[3],
 stage2_scope='P3FoSFIrfQA training-video overfit, not independent evaluation',
 purpose='Raw autoregressive outputs, GT and provenance for local Codex evaluation; not a quality verdict',
 files_sha256=files),indent=2)+'\n')
PY
ARCHIVE="$STATE/ar_results.tar.gz"
tar -czf "$ARCHIVE.tmp" -C "$BUNDLE" .
mv "$ARCHIVE.tmp" "$ARCHIVE"
echo "Download archive: $ARCHIVE"
