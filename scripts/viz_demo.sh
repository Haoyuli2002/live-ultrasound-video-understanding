#!/bin/bash
# Build and serve the current multi-stage dataset browser.
set -euo pipefail
REPO=${REPO:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}
cd "$REPO"
PYTHON=${PYTHON:-python3}
"$PYTHON" scripts/build_demo.py --config "${DEMO_CONFIG:-viz/demo_sources.json}" --output viz/catalog.json
"$PYTHON" scripts/build_video_library.py --labels "${VIDEO_LABELS:-viz/data/video_labels.json}" --output viz/video_library.json
[[ ${SERVE:-1} == 1 ]] || exit 0
echo "Open http://localhost:${PORT:-8000}/viz/index.html"
exec "$PYTHON" scripts/viz_serve.py --port "${PORT:-8000}" --directory "$REPO"
