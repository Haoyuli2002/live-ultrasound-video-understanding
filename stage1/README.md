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
Qwen3.5 training and inference require the GPU dependencies in the repository
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

## Qwen3.5 pretraining default

Stage 1 and the legacy `pretrain/train.py` / `pretrain/infer.py` now default
to `Qwen/Qwen3.5-2B` (post-trained multimodal model, without an `-Instruct` suffix).
Install `transformers>=5.2.0,<6` from the repository requirements. Chat templates
explicitly disable thinking for narration-only training and evaluation. Existing
LoRA projection targets are retained; this does not add adapters to every
Qwen3.5 linear-attention projection.

The Slurm default run name is `stage1_full295_merged_qwen35_2b`. Use a fresh
output directory: Qwen3-VL adapters/checkpoints cannot resume on Qwen3.5.
Override `--model-name` (or Slurm `MODEL_NAME`) to select another compatible
Qwen3.5 size, and use the same base model for evaluation. Stage 2 and QA retain
their previous defaults; do not merge a Qwen3.5 adapter into their Qwen3-VL base.
The Qwen3.5 GPU training smoke run remains to be verified.

## Matched Qwen3.5 2B / 4B / 9B experiments

From the cluster repository root (after syncing the current code and installing
requirements):

```bash
# Inspect submission commands without submitting jobs:
DRY_RUN=1 bash scripts/slurm/submit_stage1_qwen35_sweep.sh all
# Three 2-step training smoke jobs only:
bash scripts/slurm/submit_stage1_qwen35_sweep.sh smoke
# Or submit each smoke -> full train -> base/LoRA evaluation chain:
bash scripts/slurm/submit_stage1_qwen35_sweep.sh all
```

Each size has separate `_smoke` and `_full` output directories under
`$DATA/cluster_data/experiments/stage1/stage1_qwen35_comparison_*`.
Set `SWEEP_NAME` to start a separate experiment. The comparison uses the same
train JSONL, seed 42, video validation split, 120 frames, 224 frame size,
LoRA rank 16, learning rate 1e-4, accumulation 8 and three epochs. Each job
requests one H100; GPU feasibility, particularly 9B at 120 frames, must be
established by its smoke run. Do not reduce only one model's frame budget for
the matched comparison. A two-step run is not an exhaustive peak-memory test.

Evaluation is a separate dependent job and writes base/LoRA predictions to
`eval/base_normal.jsonl` and `eval/lora_normal.jsonl` within each full run.
By default it uses the trainer's held-out validation rows; this is not the
external final test set. Override both `EVAL_JSONL` and `EVAL_VIDEO_MAP` for
that set. Failed/timed-out dependencies cancel downstream jobs. To continue
full training from saved checkpoints and resubmit evaluation, use `full` with
the same `SWEEP_NAME`; first ensure previous jobs are no longer running.
These scripts prepare experiments; no accuracy claims follow from submission.
