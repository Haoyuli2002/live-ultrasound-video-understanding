# Accepted-data Stage 2: ordinary memory and Q-Former comparison

Full training waits for the complete current acceptance plan: 237 source
videos (196 train, 41 eval), including the user's explicit first-pass waivers.
An incomplete index cannot produce a training release or submit either job.
The eval videos never enter the optimizer schedule. Scope is the current
local acceptance plan, not the older train193 cluster annotation run.

Both variants start from the completed Stage 1 v2.1 adapter on official
Qwen3.5-2B, with new Stage 2 parameters. The default is one epoch, BF16/SDPA,
224-pixel padded frames at 1 FPS, AdamW 1e-4, LoRA rank 16, seed 20261104.
Annotation at original source resolution is separate from model input size.
These are configurable experiments, not a claim that Stage 1 already beats
the base model (its independent evaluation was only submitted).

| Architecture | Short | Long |
|---|---|---|
| ordinary | One SHORT_MEM hidden state per frame; 10 states per 10 seconds | Existing Qwen update over previous 60 Long, current 60 Short and 60 shared LONG_MEM embeddings |
| qformer | Frozen Qwen visual/merger patch features → 2-layer, 512-wide, 8-head query resampler → 10 states per 10 seconds | Separate 60-query resampler over previous Long and current-minute Short |

The Q-Former variant is a lightweight **Q-Former-style resampler**, not the
pretrained 188M-parameter BLIP-2 Q-Former. It has independent query parameters,
query self-attention, visual cross-attention, feed-forward layers, time/source
embeddings and output projections. Feature/query/LLM widths need not match.
Queries never consume GT text or future video. For a residual prefix of T<10
frames it uses T queries, preserving the original short-memory token budget.
The existing frame-by-frame Stage 3 encoder must not silently load this
checkpoint; it needs a causal short-window adapter before Q-Former Stage 3 use.

## Common supervision and distributed schedule

All six Local, one Block and one Global targets are supervised separately.
Each rank computes one target and one optimizer update per active schedule
position. Whole videos stay on a rank, in chronological order. Ranks are
balanced by target count; shorter schedules participate with zero-gradient
padding, without duplicating examples. Gradients are normalized by the active
rank count. Four ranks do not change the number of distinct supervised targets.

Current 10/60-second visual states are recomputed before each target update;
there is no mixed-weight cache of the first 50 current-minute Short states.
Previous-minute Long is detached recurrent state retained across optimizer
updates, with truncated historical gradients. Full training does not replay
the entire history with current weights at every Global step: that expensive
behavior is kept only in the small `train_overfit_ab` diagnostic. Both full
architectures use the same historical-state policy.

Default full jobs request four full H100 or A100 80GB GPUs each, 16 CPUs,
160GB host RAM, 24 hours. `NPROC_PER_NODE=1|2|4`, `GPU_TYPE=auto|h100|a100`
and `WALLTIME` are explicit choices. Two concurrent four-GPU jobs request
eight GPUs total. Runtime is not measured; 24 hours is a limit, not a guarantee.

Every 100 global updates and at epoch boundaries, a coordinated checkpoint
saves the adapter/resamplers, optimizer, rank RNG states, schedule cursor,
detached Long and metrics byte offsets. The last complete checkpoint remains
usable if a later write is interrupted. Resume requires the same code,
topology, release, map, initialization, seed and hyperparameters. Partial
metrics beyond the checkpoint are truncated when resuming.

## Q-Former single-video smoke first

Use the already accepted P3FoSFIrfQA export. Source SHA256 is checked on the
cluster. The smoke starts from the same Stage 1 adapter as planned full
training, with fresh Q-Former/Stage 2 parameters; it does not reuse old P3
overfit weights or the original draft labels.

```bash
export SMOKE_INPUT_DIR=/absolute/cluster/path/accepted_P3FoSFIrfQA
GPU_TYPE=auto EPOCHS=3 WALLTIME=02:00:00 \
  bash scripts/slurm/submit_stage2_qformer_smoke.sh
```

One H100/A100 80GB is requested. Three epochs are 48 independent updates.
After training, the job reloads epoch-1/final for teacher-forced evaluation,
then generates 48 final outputs (16 targets × real/zero/other-window memory).
`smoke_report.json` requires complete finite training/evaluation, saved and
reloadable compressor weights, changed Short and Long compressor parameters,
and nonempty real generation for each kind. Empty/truncated outputs are
counted. This is a functional gate, not a summary-quality acceptance threshold.
The report explicitly sets `quality_approved: false`.

## Freeze and transfer the complete release

On the machine holding the acceptance ledger, after all videos finish:

```bash
python -m stage2.release \
  --acceptance-root /absolute/path/stage2_full_acceptance_v4_20261104 \
  --output-dir /absolute/path/stage2_training_release
```

The exporter checks complete coverage, source identity, explicit per-video
native acceptance or waiver, current review hashes and exact teacher/blocks
roundtrips. It preserves reviewed prose without reapplying the old strict word
limit. It saves separate `train.jsonl`, `eval.jsonl` and a hashed manifest.
Copy this small release to the cluster; videos are already there. Cluster
mapping checks actual video SHA256, rather than accepting a same-ID re-encode.

## Submit full training only when ready

```bash
export DATASET_RELEASE=/absolute/cluster/path/stage2_training_release
export QFORMER_SMOKE_REPORT=/absolute/smoke/run/smoke_report.json
GPU_TYPE=auto NPROC_PER_NODE=4 DRY_RUN=1 \
  bash scripts/slurm/submit_stage2_full_training.sh ordinary
GPU_TYPE=auto NPROC_PER_NODE=4 DRY_RUN=1 \
  bash scripts/slurm/submit_stage2_full_training.sh qformer
# Remove DRY_RUN=1 / set DRY_RUN=0 to submit each checked plan.
```

The Q-Former full gate additionally checks smoke architecture, source,
initialization and implementation hashes. Full Q-Former starts fresh from
Stage 1, not from smoke-trained P3 weights. Every run snapshots committed code,
release identity and resource settings. Set `INPUT_VIDEO_MAP` if the standard
`train_full295/` and `eval_full295/` paths do not resolve every video.

Resume an interrupted job using `RESUME_RUN=/original/run` with the same
variables and code. Active same-name jobs block duplicate submissions.
`training/completed.json` and `training/final` signal the end of training.
The full training script does not automatically consume eval data or assert
medical accuracy; held-out teacher-forced and free-running evaluation must
use the same frozen eval split and memory controls for both architectures.

The evaluation CLIs accept `--release-manifest "$DATASET_RELEASE/manifest.json"`
with `--eval-jsonl "$DATASET_RELEASE/eval.jsonl"`. This verifies the independent
split and allows visually reviewed prose beyond the legacy word limit. Optional
`--limit-videos 2` selects the same deterministic video subset for both models.
Reconstruction detects and reloads Q-Former weights automatically.
