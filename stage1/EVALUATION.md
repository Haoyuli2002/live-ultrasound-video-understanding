# Stage 1: base / LoRA evaluation and LLM judge

Set `OUTPUT_DIR` to the training run to evaluate. Both base and LoRA evaluations
read its `training_config.json`: model, actual frame budget (including an OOM
fallback), frame size, sampling policy, recent-window duration, ASR truncation
and weight dtype are identical. The current 191-video training entry uses
30 frames, 224 pixels, `recent_sparse`, and `recent_seconds=15`: for prefixes
longer than 30 seconds, 15 frames cover earlier history uniformly and 15 cover
the latest 15 seconds at 1 FPS. Short prefixes use the available frame budget.
Historical experiments retain their saved settings, including uniform sampling.
Generation is greedy with 128 output tokens by default.

By default the suite uses the saved **video-held-out validation split**. Training
with `validation_fraction=0` (including the 191-video entry) has no internal
validation samples: set both `EVAL_JSONL` and `EVAL_VIDEO_MAP` to the independent
evaluation split. Missing/empty samples fail before job submission, and
training-video overlap is rejected.
All three conditions are preserved per selected sentence. Report them separately:
`through_with_asr` sees the target interval, while `before_*` does not.

The rebuilt v2.1 train191 run uses `VALIDATION_FRACTION=0`. Its empty
`validation_samples.jsonl` is not an evaluation set. Point `EVAL_JSONL` at the
rebuilt independent eval41 dataset's `samples.jsonl`, and `EVAL_VIDEO_MAP` at
that dataset's `video_map.json`. Set `FRAME_SAMPLING=recent_sparse` and
`RECENT_SECONDS=15` to match v2.1 training. The saved training `video_split.json`
still supplies the training-video IDs used to reject evaluation overlap.

## SSH smoke (10 target sentences, 30 rows per model)

Once these files are present in the SSH checkout:

```bash
cd ~/haoyu/live-ultrasound-video-understanding
export OUTPUT_DIR=/absolute/path/to/stage1_training_run
# Required for runs without an internal validation split:
export EVAL_JSONL=/absolute/path/to/independent_eval/samples.jsonl
export EVAL_VIDEO_MAP=/absolute/path/to/independent_eval/video_map.json
MODE=smoke bash scripts/slurm/submit_stage1_eval.sh --dry-run
MODE=smoke bash scripts/slurm/submit_stage1_eval.sh
```

This submits two single-H100 jobs, each limited to one hour. Judge waits for
successful evaluation (`afterok`). No model API is called from the login node.
Default judge is a locally served Qwen3.5-27B; no external API is used.
If smoke succeeds, submit all validation targets with:

```bash
MODE=full bash scripts/slurm/submit_stage1_eval.sh
```

Full jobs each request 24 hours; actual runtime has not been benchmarked.
Set `SUBMIT_JUDGE=0` for inference plus lexical metrics only.
Set `VISUAL_CONTROLS="normal blank shuffled"` to add visual ablations (triples
inference cost). Shuffled control requires at least two selected videos. These
controls test sensitivity to visual input; lexical/text-judge scores alone do
not establish visual grounding.

## Results and restart

By default, outputs are in the training run's `eval_smoke_matched/` or
`eval_full_matched/`: selected `samples.jsonl`, `base_normal.jsonl`,
`lora_normal.jsonl`, `metrics_normal.json` (token F1 / ROUGE-L, 0..1),
`judge_normal.jsonl` (auditable per-candidate responses), and
`judge_normal.jsonl.summary.json` (0..4 scores by condition and paired deltas).
Job IDs are saved in `eval_job_id.txt` and `judge_job_id.txt`. Logs are
`logs/stage1_eval_JOB.out/.err`, `logs/stage1_judge_JOB.out/.err` and
`logs/stage1_judge_vllm_JOB.log`.

Resubmit with the same variables/directory after timeout to continue completed
rows. Configuration/content hashes reject mixing different inputs, adapter
files or inference settings. An incomplete trailing JSONL write is discarded;
malformed complete lines fail. Do not run two writers against one directory.
Choose a new `EVAL_DIR` after changing settings. Finished judge errors remain
explicit errors; they are not silently retried on resume or counted as zero.
Inspect their audit trail and use a new judge `--output` path to rerun judging.
A failed evaluation leaves its dependent judge pending; cancel that dependent
job before submitting a replacement pair. No automatic cancellation is done.

The batch entries use saved training inputs rather than inherited `FRAME_*`,
`RECENT_SECONDS` or `MODEL_NAME` environment values. Direct `stage1.evaluate`
auto-detects the configuration above `--adapter-path`. For a base-only comparison
or relocated adapter, pass `--training-config /path/to/training_config.json`.
Explicit input options that contradict saved settings are rejected. The resolved
settings and training-config hash are recorded in each prediction config; old
predictions with different sampling cannot be appended to the new run.

Generation records flag `hit_token_limit`; inspect these before treating output
as complete sentences. Base model weights must remain fixed across resumed runs.
Do not interpret a completed job as a guarantee of medical correctness.

## Judge meaning

Judge receives only reference and candidate text, with model identity hidden;
candidate order is shuffled deterministically. The rubric scores semantic
agreement, medical-detail agreement (higher is better), and extra assertions
relative to the reference (lower is better). It accepts paraphrases and obvious
ASR errors. No video or hidden narration is supplied. Consequently it cannot
verify ultrasound findings or measure factual hallucination against the video.
The same model family is used for generation/judging: inspect a manual sample
and consider an independent judge before making research claims.

Failed judge requests are recorded separately. Paired deltas use only samples
with successful scores for both models; coverage and failures are reported.
Metrics are descriptive row averages, not significance tests; three conditions
for one sentence and sentences from one video are correlated observations.

## Individual commands

```bash
python -m stage1.score_predictions --predictions "$EVAL_DIR/lora_normal.jsonl" \
  --baseline "$EVAL_DIR/base_normal.jsonl" --output "$EVAL_DIR/metrics_normal.json"
python -m stage1.judge_predictions --predictions "$EVAL_DIR/lora_normal.jsonl" \
  --baseline "$EVAL_DIR/base_normal.jsonl" --output "$EVAL_DIR/judge_normal.jsonl" \
  --base-url http://127.0.0.1:8000/v1 --model Qwen/Qwen3.5-27B
```

The second command requires a running compatible judge server. Rubric and
request settings are saved in the output's `.config.json`. Prediction/baseline
sample keys and reference sentences must match exactly, including visual control.
