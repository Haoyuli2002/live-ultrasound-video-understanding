#!/bin/bash
# One-shot launcher for the Stage 1 three-condition visualizer.
#
# Validates the transcript + video, regenerates viz/data.json via
# stage1.visualize_conditions, then serves the repo root over HTTP so that
# viz/index.html and the video share one web root.
#
# Everything is env-overridable (same style as scripts/slurm/*.sbatch):
#
#   # defaults: eval_full295 / 8V649L5Q368 on port 8000
#   bash scripts/viz_eval.sh
#
#   # pick another video / split / port, or the cleaned transcripts:
#   VIDEO_ID=JcCZBKSdIRk SPLIT=eval_full295 bash scripts/viz_eval.sh
#   TRANSCRIPTS_SUBDIR=transcripts_stage1_qwen35_clean bash scripts/viz_eval.sh
#   PORT=8080 bash scripts/viz_eval.sh
#
#   # skip the HTTP server (only regenerate viz/data.json):
#   SERVE=0 bash scripts/viz_eval.sh
#
# Then, from your laptop:  ssh -N -L ${PORT}:localhost:${PORT} <user>@<host>
# and open:                http://localhost:${PORT}/viz/index.html
set -euo pipefail

# --- Repo root: prefer the home-symlink view used by the sbatch jobs, else
# fall back to the directory that contains this script. ---
DEFAULT_REPO=/dss/dsshome1/04/ge75vid2/haoyu/live-ultrasound-video-understanding
if [[ -d "$DEFAULT_REPO" ]]; then
  REPO=${REPO:-$DEFAULT_REPO}
else
  REPO=${REPO:-"$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"}
fi

SPLIT=${SPLIT:-eval_full295}
VIDEO_ID=${VIDEO_ID:-8V649L5Q368}
# eval_full295 -> videos/eval ; train_full295 -> videos/train
case "$SPLIT" in
  eval_*)  VIDEO_SUBDIR=${VIDEO_SUBDIR:-eval} ;;
  train_*) VIDEO_SUBDIR=${VIDEO_SUBDIR:-train} ;;
  *)       VIDEO_SUBDIR=${VIDEO_SUBDIR:-eval} ;;
esac
# transcripts (raw) or transcripts_stage1_qwen35_clean (cleaned)
TRANSCRIPTS_SUBDIR=${TRANSCRIPTS_SUBDIR:-transcripts}

MAX_FRAMES=${MAX_FRAMES:-120}
PORT=${PORT:-8000}
SERVE=${SERVE:-1}
OUTPUT=${OUTPUT:-viz/data.json}

TRANSCRIPT=${TRANSCRIPT:-cluster_data/QA/${SPLIT}/${TRANSCRIPTS_SUBDIR}/${VIDEO_ID}.json}
VIDEO_FILE=${VIDEO_FILE:-cluster_data/videos/${VIDEO_SUBDIR}/${VIDEO_ID}.mp4}
# viz/index.html resolves --video-url relative to viz/, so prefix with ../
VIDEO_URL=${VIDEO_URL:-../${VIDEO_FILE}}

cd "$REPO"

echo "Repo:       $REPO"
echo "Transcript: $TRANSCRIPT"
echo "Video:      $VIDEO_FILE"
echo "Video URL:  $VIDEO_URL"
echo

[[ -f "$TRANSCRIPT" ]] || { echo "Missing transcript: $TRANSCRIPT" >&2; exit 1; }
[[ -f "$VIDEO_FILE" ]] || { echo "Missing video: $VIDEO_FILE" >&2; exit 1; }

# Validate the three fields visualize_conditions needs before doing real work.
python - "$TRANSCRIPT" <<'PY'
import json, sys
d = json.load(open(sys.argv[1], encoding="utf-8"))
vid = d.get("video_id")
dur = d.get("duration_sec") or d.get("duration")
seg = "segments" if "segments" in d else ("sentence_units" if "sentence_units" in d else None)
print(f"  video_id={vid!r}  duration={dur!r}  seg_key={seg!r}")
problems = []
if not vid:
    problems.append("missing video_id")
if not dur:
    problems.append("missing/zero duration")
if seg is None:
    problems.append("no segments / sentence_units")
if problems:
    sys.exit("Transcript validation failed: " + "; ".join(problems))
PY

python -m stage1.visualize_conditions \
  --transcript "$TRANSCRIPT" \
  --video-url  "$VIDEO_URL" \
  --output     "$OUTPUT" \
  --max-frames "$MAX_FRAMES"

if [[ "$SERVE" != 1 ]]; then
  echo "SERVE=0 set; wrote $OUTPUT and exiting without starting a server."
  exit 0
fi

echo
echo "Serving repo root on http://127.0.0.1:${PORT}/ (Ctrl-C to stop)"
echo "On your laptop:  ssh -N -L ${PORT}:localhost:${PORT} <user>@<host>"
echo "Then open:       http://localhost:${PORT}/viz/index.html"
echo
exec python -m http.server "$PORT"
