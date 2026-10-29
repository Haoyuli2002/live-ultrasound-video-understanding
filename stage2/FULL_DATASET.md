# Full Stage 2 visual teacher dataset

The full run uses the audited 193-video training map, excluding the 41 evaluation
videos by both ID and resolved source path. It reads original silent video only;
ASR, Stage 1 target sentences and evaluation annotations are not teacher inputs.
Qwen3.5-27B uses the same compact medical prompt and bounded 120-frame transport
as the successful single-video run. Local/Block/Global targets remain independent.

```bash
cd ~/haoyu/live-ultrasound-video-understanding
git pull --ff-only origin main
bash scripts/slurm/submit_stage2_full.sh --dry-run
# Source the printed run's submit_env.sh, then submit its frozen manifest:
source /absolute/run/directory/submit_env.sh
bash scripts/slurm/submit_stage2_full.sh --resume
```

`STAGE2_TRAIN_MAP` and `STAGE2_EVAL_MAP` can select explicit maps; relative video
paths resolve under `DATA`. Never substitute the larger raw ASR map without
reviewing its membership. Planning verifies every file, duration and source hash.
Set `STAGE2_SEED_TEACHER_DIR` before planning to copy compatible complete or
partial per-video labels from a **stopped** run. Source video, full recipe, model
and prompt must match. A completed seeded video is excluded from GPU shards.
For a running seed job, also set `STAGE2_SEED_JOB` to its job ID. The array is
submitted immediately with an `afterany` dependency and imports that video's
latest compatible labels when its worker starts. It makes no duplicate teacher
requests for completed windows. An unavailable or invalid deferred seed is
logged and the missing video is generated using the frozen recipe instead.

Default scheduling is 8 balanced shards, at most 4 concurrent tasks. Each task
gets one H100, 8 CPU cores, 120 GB host RAM and a 24-hour limit. This is independent
video generation; the model is not distributed across GPUs. Work is balanced by
teacher request count and cumulative-prefix decoding. Resource settings are
`STAGE2_SHARDS`, `STAGE2_CONCURRENCY` and `STAGE2_WALLTIME`; 24 hours is a limit,
not a promised completion time.

Production submissions should use a saved Git code snapshot as `REPO`; the active
193-video run records its exact commit and snapshot directory. Each array task
loads the teacher once and processes its shard. A video-level error is audited
and the remaining videos continue. The shard exits nonzero if any video failed.
Every minute is saved with an atomic replacement so a timeout retains the last
complete batch. Resume skips completed windows; a terminal tail shorter than ten
seconds needs no additional label. Completed shards rebuild without loading 27B.

`submissions.json` records array and CPU merge job IDs. Duplicate submission is
refused while any prior job is active. After jobs stop, source `submit_env.sh`
and run `submit_stage2_full.sh --resume` to retry missing work using the same
manifest. This resubmits the shards; completed shards finish quickly.

The merge is a CPU job, dependent on **afterany** of the array so failures still
produce an explicit report. It rechecks all source hashes, configs, expected
teacher counts, complete video coverage and chronology. Missing or incompatible
videos create `report.json` with `status=incomplete` and prevent publication.
Only a fully validated run publishes `blocks.jsonl` with `status=ready`.

Persistent outputs: `manifest.json`, `video_paths.json`, `excluded_eval_paths.json`,
`teacher/`, `audit/`, per-shard maps and blocks under `shards/`, and logs under
`logs/stage2_full_ARRAYID_INDEX.{out,err}` / `stage2_merge_JOBID.{out,err}`.
The protocol validates data coverage and provenance; it does not certify the
medical accuracy of generated summaries, which require visual sampling review.

Checks: `python -m unittest stage2.test_full_run stage2.test_annotation_job stage2.test_sampling_runtime stage2.test_stage2`.
