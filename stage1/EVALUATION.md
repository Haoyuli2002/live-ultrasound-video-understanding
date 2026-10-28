# Stage 1: base / LoRA evaluation and LLM judge

Default adapter: `stage1_full186_2b_4gpu_f30_e1_20261025_052451/final`.
The suite compares Qwen3.5-2B base and that adapter on identical rows, using
30 frames at 224 pixels, uniform sampling, greedy decoding and 128 output tokens.
It uses the saved **video-held-out validation split**, not training videos. This
is validation, not an independent final test. Override `EVAL_JSONL` and
`EVAL_VIDEO_MAP` for a separate test split; training-video overlap is rejected.
All three conditions are preserved per selected sentence. Report them separately:
`through_with_asr` sees the target interval, while `before_*` does not.

## SSH smoke (10 target sentences, 30 rows per model)

Once these files are present in the SSH checkout:

```bash
cd ~/haoyu/live-ultrasound-video-understanding
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

By default, outputs are in the training run's `eval_smoke_f30/` or
`eval_full_f30/`: selected `samples.jsonl`, `base_normal.jsonl`,
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
