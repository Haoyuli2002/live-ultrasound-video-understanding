# Stage 1 — paired narration pretraining

This is a new implementation. It does not import code from `pretrain/`.

For each complete target sentence `[start, end]`, `data.py` creates three
deterministic, paired examples with the **same target**:

| condition | video input | earlier ASR |
|---|---|---|
| `before_with_asr` | `[0, start)` | visible |
| `through_with_asr` | `[0, end)` | visible |
| `before_mask_asr` | `[0, start)` | masked |

The target sentence and later ASR never appear in the prompt. The baseline
samples up to 120 ordered frames uniformly across the selected interval. For a
120-second interval this is 1 FPS; long-video prefixes become much sparser.
`--frame-sampling recent_sparse` is an optional same-budget alternative: with
the default 120-frame budget it takes 96 frames from the latest 120 seconds and
24 evenly spaced frames from earlier history. Use the same sampling options in
training and evaluation. Only assistant
target tokens receive NTP loss. The training split is by video, so all three
conditions for a sentence remain together.

ASR sentence times inside a transcript segment are linearly interpolated; these
are estimates, not word-level forced alignment. The default builder omits
transcripts with no terminal punctuation. For a quick coverage experiment,
`--unpunctuated-fallback segment` uses eligible ASR segments only when no
punctuated sentence exists. These are tagged `asr_segment_fallback`, use
segment timestamps and an utterance prompt, and are **not asserted to be
complete sentences**. The summary reports fallback videos and rows. Audit
them separately before mixing them into the main dataset. Historical ASR is
kept in the JSONL; the input template uses its last `--max-asr-chars`
characters (default 4000).

For the cleaned sentence dataset, run `python -m stage1.clean_asr` first. It
uses the repository's OpenAI-compatible local vLLM pattern with
`Qwen/Qwen3.5-27B`. The teacher receives chronological Whisper ASR segments
with neighboring text context, adds punctuation, and makes conservative
ultrasound terminology corrections. It processes already-punctuated segments
too. Each output transcript retains unchanged segment timestamps and a
`raw_segments` copy; `asr_cleaning.term_corrections` and the separate JSONL
audit record lexical edits. `stage1.data` then splits cleaned text into
sentence targets. The source transcript folder is untouched. Word and sentence
times within a segment remain interpolation estimates, not forced alignment.
This teacher sees text only, so manually review a sample of medical term edits
and sentence/time boundaries before training. The older
`stage1.restore_punctuation` remains a punctuation-only comparison baseline.

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
