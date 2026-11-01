# Save first, review after generation

Stage 2 submission scripts now enable save-first collection. The video prompt,
sampling, source identity and Qwen3.5-27B model stay the same. Every returned text
is saved verbatim, including whitespace, empty text, labels, lists, reasoning
tags and truncated responses. No format retry, truncation repair, normalization
or automatic filtering runs in this mode. Transport/decoding/storage failures
can still stop a video; already persisted windows remain reusable.

Each response is atomically saved and fsynced to `raw/<video>/<kind>_<start>_<end>.json`
before any text validation or minute-batch publication. It contains the exact
text, request prompt, source/model/config identity, finish reason and refusal
metadata. Replaying an incomplete minute reads matching raw windows from cache;
it only requests windows not yet saved. Cache mismatches fail explicitly.

The frozen annotation config remains the original run recipe for compatibility
with existing complete minutes. New records additionally declare
`collection_policy=save_first_review_later_v1` and `format_retries=0`; these
describe the new output handling and override the historical format-retry
behavior. They do not claim that new text passed the prose validator.

## Continue the existing failed shard

Use the working repository, not the old `code_229d4ed` snapshot:

```bash
cd ~/haoyu/live-ultrasound-video-understanding
git pull --ff-only origin main
export STAGE2_RUN_DIR=/dss/mcmlscratch/04/ge75vid2/haoyu/live-ultrasound-video-understanding/cluster_data/stage2/runs/train193_27b_304fcc7_20261029T182122Z_waXPRT
bash scripts/slurm/submit_stage2_collect_retry.sh 3 --dry-run
bash scripts/slurm/submit_stage2_collect_retry.sh 3
```

The helper snapshots committed code into a new directory in the existing run.
It reuses the frozen manifest and original teacher minutes read-only, writes new
minute batches to `collected/`, requests only shard 3 (H100, 4h by default), and
refuses to duplicate an active shard. `GPU_TYPE`, `STAGE2_WALLTIME` and
`STAGE2_CONCURRENCY` override those resource choices. Other shards and the
original pending CPU merge are left running.

A new CPU review/merge job waits for the retry and any active previous jobs,
including the old merge, so it cannot race them. The old strict merge may still
report incomplete data because it reads only the original `teacher/` directory.
The new review result is `quality_report.json`, not that old `report.json`.
If another original shard fails later, retry its inactive index with the same
helper after the current retry/review jobs finish. No full-run restart is needed.

## Review outputs

- `teacher/`: existing successful minute batches, left unchanged.
- `collected/`: resumed per-video event data, including every new raw target.
- `raw/`: immediately persisted response cache, including partial minutes.
- `blocks.unreviewed.jsonl`: all candidate block rows, with no text filtering.
- `quality_report.json`: total counts; counts by issue, video and label type;
  flagged windows and their full original text; completeness and pending review.

Length above the preferred 40/80/90 words is counted as a warning. Empty/short
text, possible perspective labels, structured output, multiple paragraphs,
reasoning tags, refusals and token-limit truncation are counted separately.
These checks describe presentation/response problems, not clinical truth.
One summary can have several issues; issue counts should not be summed to get
the number of flagged summaries.
The report always records zero automatic deletions and rewrites. Format issues
alone do not fail the collection or review job. Missing/incompatible videos or
broken chronology remain completeness errors; a final candidate is published
only once every planned video has all expected windows.

The strict training loader is unchanged by default. Candidate data is pending
the user's decision; this workflow does not publish it as `blocks.jsonl`.
To rerun only the CPU report later:

```bash
cd "$STAGE2_RUN_DIR"/code_keepall_<commit>
python -m stage2.raw_collection --run-dir "$STAGE2_RUN_DIR"
```

The direct `stage2.annotate` CLI also defaults to keep-all. Use
`--raw-dir <durable-dir>` to choose the response cache; otherwise it is stored
beside the output as `<video>.raw/`. `--strict-format` explicitly selects the
legacy retry/reject behavior for compatibility.
