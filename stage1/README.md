# Stage 1 — paired narration pretraining

This is a new implementation. It does not import code from `pretrain/`.

For each complete target sentence `[start, end]`, `data.py` creates three
deterministic, paired examples with the **same target**:

| condition | video input | earlier ASR |
|---|---|---|
| `before_with_asr` | `[0, start)` | visible |
| `through_with_asr` | `[0, end)` | visible |
| `before_mask_asr` | `[0, start)` | masked |

The target sentence and later ASR never appear in the prompt. Up to 120 ordered
frames are sampled across the entire selected interval. This is 1 FPS for a
120-second interval and uniform subsampling for longer intervals. The train and
evaluation scripts use the same prompt and image processing. Only assistant
target tokens receive NTP loss. The training split is by video, so all three
conditions for a sentence remain together.

ASR sentence times inside a transcript segment are linearly interpolated; these
are estimates, not word-level forced alignment. Inspect a sample of timings
before a full training run. Historical ASR is kept in the JSONL; the input
template uses its last `--max-asr-chars` characters (default 4000).

```bash
python -m stage1.data --transcripts results/transcripts \
  --output stage1/samples.jsonl
python -m stage1.train --train-jsonl stage1/samples.jsonl \
  --video-path-map /path/to/video_path_map.json \
  --output-dir /path/to/stage1_output
python -m stage1.evaluate \
  --eval-jsonl /path/to/stage1_output/validation_samples.jsonl \
  --video-path-map /path/to/video_path_map.json \
  --adapter-path /path/to/stage1_output/final \
  --output /path/to/predictions.jsonl
```

For the cluster dataset, pass `--video-map` from the VLM pretrain filter and
run the builder separately for the existing train and eval splits. The builder
requires a matching transcript for every selected video, excludes transcripts
outside the map, and prints selected-video and three-condition sample counts.
It writes atomically, so a missing or malformed transcript does not replace a
previous output. See `docs/HOW_TO_RUN_PIPELINE.md` for split-specific commands.
Keep the eval JSONL out of `stage1.train --train-jsonl`; the trainer also holds
out whole videos within the train split for validation.

Run `stage1.evaluate` separately with `--visual-control blank` and
`--visual-control shuffled` for matched visual-dependence controls. The video
path map is a JSON object from `video_id` to an absolute video path. Full
Qwen3-VL training and inference require the GPU dependencies in the repository
requirements and have not been verified on the target GPU yet.
