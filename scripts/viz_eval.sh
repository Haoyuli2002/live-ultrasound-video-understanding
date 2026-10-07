#!/bin/bash
# One-shot launcher for the Stage 1 three-condition visualizer.
#
# Validates the transcript + video, regenerates viz/data.json via
# stage1.visualize_conditions, then serves the repo root over HTTP so that
# viz/index.html and the video share one web root.
#
# Everything is env-overridable (same style as scripts/slurm/*.sbatch):
#
#   # defaults: video 8V649L5Q368 on port 8000
#   bash scripts/viz_eval.sh
#
#   # pick another video / port:
#   VIDEO_ID=B2USlWmqOV0 bash scripts/viz_eval.sh
#   PORT=8080 bash scripts/viz_eval.sh
#
#   # fully explicit paths (bypass the templates below):
#   TRANSCRIPT=cluster_data/transcripts/8V649L5Q368.json \
#   VIDEO_FILE=cluster_data/videos/8V649L5Q368.mp4 bash scripts/viz_eval.sh
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

VIDEO_ID=${VIDEO_ID:-8V649L5Q368}
# Real cluster layout: transcripts live flat under cluster_data/transcripts/,
# videos live flat under cluster_data/videos/ (per-split dirs also exist, but the
# transcript JSONs are only in the flat dir). Override the two paths below for
# anything outside this convention.
TRANSCRIPTS_DIR=${TRANSCRIPTS_DIR:-cluster_data/transcripts}
VIDEOS_DIR=${VIDEOS_DIR:-cluster_data/videos}

MAX_FRAMES=${MAX_FRAMES:-120}
PORT=${PORT:-8000}
SERVE=${SERVE:-1}
OUTPUT=${OUTPUT:-viz/data.json}

TRANSCRIPT=${TRANSCRIPT:-${TRANSCRIPTS_DIR}/${VIDEO_ID}.json}
VIDEO_FILE=${VIDEO_FILE:-${VIDEOS_DIR}/${VIDEO_ID}.mp4}
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
