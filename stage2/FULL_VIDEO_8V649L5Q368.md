# Qwen3.5-27B full-video medical ultrasound summaries

On the LRZ login node:

```bash
cd ~/haoyu/live-ultrasound-video-understanding
git pull --ff-only origin main
unset RUN_DIR VIDEO_MAP VIDEO_PATH VLLM_PORT
bash scripts/slurm/submit_stage2_8V649L5Q368.sh
```

This submits a one-hour, one-H100 job for the original 1136.8-second video. It starts local
vLLM in the `ultrasound` Conda environment. No external teacher API is used.
If source lookup fails, set `VIDEO_PATH` to the original video's absolute path.

## Natural prose targets

New labels use `stage2_prose_v2`. Each target is a coherent English medical
ultrasound paragraph. View, anatomy, probe handling, findings, dynamics,
measurements and changes are internal observation perspectives, not mandatory
fields or output headings. The JSONL wrapper retains video IDs and time windows
for training, but there are no `facets` dictionaries in new output.

The compact prompt excludes ordinary gestures, speaking to camera and background
equipment. When the entire window has no ultrasound-relevant visual evidence, it
returns exactly `No ultrasound-relevant visual information in this interval.`
Relevant positioning and explicit anatomical/procedural gestures are allowed.

Local targets describe the current 10 seconds (40-word preferred length); Block
integrates an independent full minute (80 words); Global integrates its own
cumulative video prefix (90 words). Word counts are prompt guidance only.
Longer prose is accepted immediately, saved unchanged, and logged with
`[length-warning]`; it does not trigger retries or abort generation/building.
Python never truncates medical sentences. Format errors still receive bounded
retries. Length acceptance does not certify factual accuracy.

The teacher uses only silent video, never ASR or the target narration. The prompt
prioritizes supported visual observations and inspects both sides of split-screen
footage. It avoids generic room descriptions, placeholder negatives, interpreting
a visible monitor as a measurement, or inferring disease absence from sparse
frames. A demonstration can be summarized as positioning and probe handling when
there is no interpretable scan. These are prompt safeguards, not proof of medical
accuracy: generated summaries still require visual review.

The source is unchanged. Temporary clips preserve the time interval and aspect
ratio, remove audio, and use a maximum side of 1024 pixels. Before sending to
vLLM, ffmpeg bounds each clip to at most 120 frames, using up to 2 fps and
reducing FPS across the entire interval for long prefixes. For example, [0,420)
has 120 frames at 2/7 fps; [0,1080) has 120 frames at 1/9 fps. ffprobe verifies
frame count, full duration, actual FPS and absence of audio. vLLM decodes all
these frames and Qwen does not resample them. Backend resizing may further
reduce resolution. Thinking is disabled, with a 1024-token response cap.

For this video the pipeline produces 113 Local, 18 Block and 18 Global targets:
149 teacher events and 19 training block rows. The final partial minute has five
Local targets, no Block/Global; approximately 6.8 seconds after 1130 are outside
the complete 10-second grid. Global prefixes end at 60, 120, ..., 1080 seconds.

## Outputs and resume

The printed run directory contains:

- `teacher/8V649L5Q368.jsonl`: versioned teacher events and annotation config.
- `teacher/8V649L5Q368.prompt.txt`: actual system prompt and perspective plans.
- `blocks.jsonl`: chronological training rows, checked by `load_blocks`.
- `audit/8V649L5Q368.jsonl`: completion/failure audit.
- `video_paths.json`, `submit_env.sh`, `job_id.txt`: submission metadata.

Logs persist at `logs/stage2_annotate_JOBID.{out,err}` and
`logs/stage2_vllm_JOBID.log`. Invalid raw teacher responses are logged.
Completed minute batches are flushed, so interruption within a minute reruns
that minute. To resume a stopped **prose-v2** run, source its `submit_env.sh` and
rerun the submitter after confirming the previous job has ended. Resume rejects
changed annotation config, including the prompt fingerprint. Already completed
labels are reused. `--dry-run` validates submission inputs without calling Slurm.

Do not resume the old facet-format run. It remains on disk for comparison.
Older prose runs also require a fresh directory if their prompt or sampling
policy differs. Resume additionally checks the SHA-256 of the original video.
Valid legacy `stage2_facets_v1` datasets remain readable, but mixing formats in a
training dataset is rejected. Embedded labels in a legacy field now fail
validation instead of silently being swallowed as one field.

## Previous failures and safeguards

- Job 5874840 exhausted three retries on a local summary longer than the hard
  44-word tolerance. Word counts are now guidance: long prose is retained.
- Job 5878785 saved 48 events through [0,360), then received HTTP 400 from
  `Qwen3VLProcessor` for [0,420). The logs identify video processing as the
  failure, but do not expose the inner exception. The previous pipeline decoded
  dense clips and requested further processor sampling. Sampling is now bounded
  once, before transport, and processor resampling is disabled. Actual Qwen
  inference still needs the GPU validation described below.
- Job 5883276 failed before annotation because vLLM could not bind port 8000
  (`Address already in use`). Jobs now choose an available port and require their
  own process, unique model alias and random API key to pass readiness checks.

Each batch job uses a fresh short `/tmp/s2.*` directory, cleans up only its own
server and clips, and rejects incompatible resume config before loading weights.
Before annotation it requests [0,10), [0,420) and [0,1080) for this video.
These probe summaries are logged separately; they are not teacher training rows.
This catches a recurrence of the long-prefix failure before generating minutes
of labels. This 27B validation uses one full H100; the Stage 1 ASR MIG allocation
is a separate job.

CPU checks: `python -m unittest stage2.test_annotation_job stage2.test_sampling_runtime stage2.test_stage2`.
Qwen/vLLM inference and medical content quality must be checked on actual output.
