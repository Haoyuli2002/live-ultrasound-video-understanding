# Stage 1 Three-Condition Visualizer

Shows the three matched Stage 1 conditions (before_with_asr / through_with_asr /
before_mask_asr) over a real video: same target sentence, different inputs. The
frame markers and ASR text come straight from stage1.data / stage1.model, so the
page shows exactly what training feeds the model.

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
