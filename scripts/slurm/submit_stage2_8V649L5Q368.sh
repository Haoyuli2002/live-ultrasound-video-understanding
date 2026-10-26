#!/usr/bin/env bash
# Generate Stage 2 labels for the entire original video on one H100.
# Run from the SSH checkout. --dry-run validates input and writes configuration
# without submitting. RUN_DIR=<previous run> resumes the same generated labels.
set -euo pipefail
if [[ $# -gt 1 || ( $# -eq 1 && $1 != --dry-run ) ]]; then
  echo "Usage: bash scripts/slurm/submit_stage2_8V649L5Q368.sh [--dry-run]" >&2
  exit 2
fi
export REPO=${REPO:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}
export DATA=${DATA:-/dss/mcmlscratch/04/ge75vid2/haoyu/live-ultrasound-video-understanding}
export VIDEO_IDS=8V649L5Q368
export MODEL=Qwen/Qwen3.5-27B
export SPLIT=8V649L5Q368_full
export VIDEO_FPS=2.0
export VIDEO_MAX_FRAMES=120
export CLIP_FPS=2.0
export CLIP_MAX_SIDE=1024
export MAX_MODEL_LEN=32768
export MAX_TOKENS=1024
export MAX_ATTEMPTS=3
export WALLTIME=${WALLTIME:-01:00:00}
export VLLM_PORT=${VLLM_PORT:-8000}
export CONDA_ENV=${CONDA_ENV:-ultrasound}
cd "$REPO"
mkdir -p logs
for file in stage2/annotate.py stage2/build_data.py stage2/data.py stage2/summary_schema.py; do
  [[ -f "$file" ]] || { echo "Missing $file; pull the latest main first" >&2; exit 1; }
done
# Explicit VIDEO_PATH / VIDEO_MAP wins; otherwise inspect known cluster maps.
SOURCE_VIDEO=$(python3 - "$DATA" "$VIDEO_IDS" "${VIDEO_PATH:-}" "${VIDEO_MAP:-}" <<'PY'
import json, sys
from pathlib import Path
root, vid, explicit, explicit_map = sys.argv[1:]
if explicit:
    path = Path(explicit).expanduser().resolve()
else:
    root = Path(root)
    candidates = [Path(explicit_map)] if explicit_map else [
        root / 'cluster_data/stage2/train_full295_video_paths.json',
        root / 'cluster_data/splits/train_full295_qwen35_pretrain_keep_videos.json',
        root / 'cluster_data/splits/train_full295_asr_keep_videos.json',
        root / 'cluster_data/splits/eval_full295_asr_keep_videos.json',
    ]
    path = None
    for candidate in candidates:
        if not candidate.is_file():
            continue
        value = json.loads(candidate.read_text()).get(vid)
        if value:
            proposed = Path(value)
            proposed = proposed if proposed.is_absolute() else root / proposed
            if proposed.is_file():
                path = proposed.resolve()
                break
    if path is None:
        sys.exit('Video not found in cluster maps. Set VIDEO_PATH=/absolute/path/8V649L5Q368.mp4 and retry.')
if not path.is_file():
    sys.exit(f'Missing source video: {path}')
print(path)
PY
)
if [[ -z "${RUN_DIR:-}" ]]; then
  mkdir -p "$DATA/cluster_data/stage2/runs"
  RUN_DIR=$(mktemp -d "$DATA/cluster_data/stage2/runs/8V649L5Q368_qwen35_27b_prose_v2_$(date +%Y%m%d_%H%M%S)_XXXXXX")
fi
mkdir -p "$RUN_DIR"
export RUN_DIR=$(cd "$RUN_DIR" && pwd)
export VIDEO_MAP="$RUN_DIR/video_paths.json"
export TEACHER_DIR="$RUN_DIR/teacher"
export AUDIT_DIR="$RUN_DIR/audit"
export BLOCKS="$RUN_DIR/blocks.jsonl"
# Reject incompatible resume before overwriting metadata or requesting a GPU.
python3 - "$TEACHER_DIR/$VIDEO_IDS.jsonl" <<'PY_RESUME'
import json, sys
from pathlib import Path
path = Path(sys.argv[1])
if path.exists():
    for line in path.read_text().splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        if row.get("schema_version") != "stage2_prose_v2":
            sys.exit("Old structured labels cannot be resumed into prose data. Unset RUN_DIR for a fresh run.")
PY_RESUME
# Do not inherit a stale smoke clip directory or log path from earlier jobs.
unset CLIP_DIR VLLM_LOG
python3 - "$VIDEO_IDS" "$SOURCE_VIDEO" "$VIDEO_MAP" <<'PY'
import json, sys
from pathlib import Path
mapping = {sys.argv[1]: sys.argv[2]}
output = Path(sys.argv[3])
if output.exists() and json.loads(output.read_text()) != mapping:
    sys.exit('RUN_DIR already refers to a different source video')
output.write_text(json.dumps(mapping, indent=2) + '\n')
PY
# Save only relevant, non-secret configuration for inspection/resubmission.
ENV_FILE="$RUN_DIR/submit_env.sh"
{
  printf '# Source this file before resubmitting the same run.\n'
  for name in REPO DATA RUN_DIR VIDEO_IDS MODEL SPLIT VIDEO_MAP TEACHER_DIR AUDIT_DIR BLOCKS VIDEO_FPS VIDEO_MAX_FRAMES CLIP_FPS CLIP_MAX_SIDE MAX_MODEL_LEN MAX_TOKENS MAX_ATTEMPTS WALLTIME VLLM_PORT CONDA_ENV; do
    printf 'export %s=%q\n' "$name" "${!name}"
  done
} > "$ENV_FILE"
printf 'Video: %s\nModel: %s\nRun: %s\n' "$SOURCE_VIDEO" "$MODEL" "$RUN_DIR"
if [[ ${1:-} == --dry-run ]]; then
  echo 'Dry run complete; no Slurm job submitted.'
  exit 0
fi
JOB_ID=$(sbatch --parsable --export=ALL --time="$WALLTIME" --job-name=stage2-8V649L5Q368 scripts/slurm/run_stage2_annotate_build.sbatch)
JOB_ID=${JOB_ID%%;*}
printf '%s\n' "$JOB_ID" > "$RUN_DIR/job_id.txt"
printf 'Job: %s\nBlocks: %s\nResume config: %s\n' "$JOB_ID" "$BLOCKS" "$ENV_FILE"
printf 'squeue -j %s\n' "$JOB_ID"
printf 'tail -f logs/stage2_annotate_%s.out logs/stage2_annotate_%s.err\n' "$JOB_ID" "$JOB_ID"
