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

## LLM-assisted ASR cleanup and light polishing

This is data preprocessing, not teacher labeling or distillation. Run
`python -m stage1.clean_asr` before `stage1.data` for the cleaned dataset.
Its inputs are (1) a folder of Whisper transcript JSON files containing
timestamped `segments` with `start`, `end`, and `text`; (2) a pretrain keep-map
whose keys select video IDs; and (3) an OpenAI-compatible local vLLM endpoint
serving `Qwen/Qwen3.5-27B`. The model receives **ASR text only**, not video or
audio. Each request contains eight segments by default, with their indices and
times, plus the preceding/following two segments as context only. The context
segments are not edited in that request.

The prompt asks for punctuation, capitalization, and spacing while preserving wording,
negation, uncertainty, and order. It allows a medical or ultrasound term
correction only when the surrounding ASR makes it clear. The required LLM
JSON has one `{index, clean_text, corrections}` object per requested segment;
each lexical correction contains `from`, `to`, and `reason`. The script checks
the returned count and order, requires every word-level change to match a
declared correction, and rejects large rewrites. It then replaces only segment
`text`, retaining all other segment fields and timestamps. Already-punctuated
transcripts go through the same process.

For every selected video, a new JSON transcript is written to `--output-dir`:

| Field | Meaning |
|---|---|
| `segments` | Cleaned text under original segment times and metadata |
| `full_text` | Cleaned segment text joined in order |
| `raw_segments` | Unmodified input segments |
| `raw_full_text` | Original `full_text`, if the input contained it |
| `asr_cleaning` | Model, counts, and indexed terminology corrections |

The `--audit-output` JSONL has one row per video. A cleaned video records its
corrections and counts with `"status":"cleaned"`; a skipped video records
`"status":"failed"` with the error. Source files are never overwritten.
Existing `sentence_units` are removed because they refer to old text. The
cleaner writes one video atomically; `--resume` skips output files that already
exist. A malformed or unverifiable response skips only that video without
writing its output, records a failure audit row, and continues with the
remaining videos; the script exits non-zero if any video failed, so the Slurm
pipeline stops before merging partial coverage. Grounding checks match declared
corrections at the word level, so a multiword `from`/`to` with different spacing
or punctuation is accepted when its words occur contiguously, while an absent
run is still rejected. `stage1.data`
then reads the cleaned `segments`, splits complete sentences, estimates
sentence times by character interpolation inside ASR segments, and builds the
three paired Stage 1 views. Text-only corrections cannot establish what the
speaker actually said; review medical edits against audio and review sentence
boundaries before training.

```bash
python -m stage1.clean_asr \
  --transcripts results/transcripts \
  --video-map /path/to/pretrain_keep_videos.json \
  --output-dir results/transcripts_stage1_qwen35_clean \
  --audit-output results/stage1_qwen35_clean_audit.jsonl \
  --resume
python -m stage1.data --transcripts results/transcripts_stage1_qwen35_clean \
  --video-map /path/to/pretrain_keep_videos.json \
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

### Seven-video recovery and merged pretraining data

The seven train videos that previously produced zero sentence samples are
listed in `stage1.seven_video_pilot`. The dedicated Slurm job selects exactly
these IDs from the 193-video pretrain keep-map, cleans only their ASR, builds
their paired rows, and merges them with the existing 186-video baseline JSONL.
The merge fails unless the baseline and pilot have disjoint, complete coverage
of all 193 selected videos. It writes a new JSONL and keeps the baseline.

```bash
mkdir -p logs
sbatch scripts/slurm/run_stage1_seven_video_pilot.sbatch
```

After reviewing the seven-video audit and merged row counts, run a short GPU
training smoke test, then submit the full training job. The trainer saves
periodic checkpoints and resumes from the latest checkpoint in the same output
directory.

```bash
sbatch --export=ALL,RUN_NAME=stage1_smoke,MAX_STEPS=2,FRAME_BUDGET=120,SAVE_STEPS=1 \
  scripts/slurm/run_stage1_pretrain_merged.sbatch
sbatch scripts/slurm/run_stage1_pretrain_merged.sbatch
```

The full job uses 120 frames, three epochs, and a 14-hour walltime per
submission. If it times out, resubmit the same command to resume from its last
saved checkpoint. Train and eval video maps remain separate: this merge uses
only `train_full295` and no `eval_full295` transcripts.
