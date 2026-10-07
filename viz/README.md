# Stage 1 Three-Condition Visualizer

Shows the three matched Stage 1 conditions (before_with_asr / through_with_asr /
before_mask_asr) over a real video: same target sentence, different inputs. The
frame markers and ASR text come straight from stage1.data / stage1.model, so the
page shows exactly what training feeds the model.

## 0. One-shot launcher (recommended on the cluster)

`scripts/viz_eval.sh` validates the transcript + video, regenerates
`viz/data.json`, and serves the repo root over HTTP in a single command. Every
input is env-overridable:

```bash
# defaults: eval_full295 / 8V649L5Q368 on port 8000
bash scripts/viz_eval.sh

# pick another video / split / port, or the cleaned transcripts:
VIDEO_ID=JcCZBKSdIRk bash scripts/viz_eval.sh
TRANSCRIPTS_SUBDIR=transcripts_stage1_qwen35_clean bash scripts/viz_eval.sh
PORT=8080 bash scripts/viz_eval.sh

# just regenerate viz/data.json without starting a server:
SERVE=0 bash scripts/viz_eval.sh
```

Then from your laptop: `ssh -N -L 8000:localhost:8000 <user>@<host>` and open
`http://localhost:8000/viz/index.html`.

The sections below document the underlying manual steps.

## 1. Generate data.json (parameterized; nothing is hard-coded)

```bash
python -m stage1.visualize_conditions \
  --transcript <path/to/transcript.json> \
  --video-url  <video path RELATIVE to viz/index.html> \
  --output     viz/data.json \
  --max-frames 120
```

- `--transcript`: a transcript JSON with `{video_id, duration_sec, segments:[{start,end,text}]}`.
- `--video-url`: what the <video> tag loads. It is resolved by the browser
  relative to `viz/index.html`. If you serve the repo root and the video lives at
  `azure_data/videos/X.mp4`, pass `../azure_data/videos/X.mp4`.
- `viz/data.json` is git-ignored (regenerate it anywhere).

## 2. Serve and open

```bash
# From the REPO ROOT so viz/ and the video share one web root:
python -m http.server 8000
# then open:  http://localhost:8000/viz/index.html
```

Over SSH, forward the port:  `ssh -L 8000:localhost:8000 <user>@<host>`
then open http://localhost:8000/viz/index.html in your local browser.
