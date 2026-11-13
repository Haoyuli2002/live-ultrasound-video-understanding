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
| qformer | One reusable query per second reads cached minute-local 512-d Short states, then current-frame visual patches; output projected to Qwen width D | Separate 60-query resampler over previous Long and current-minute 60 Short tokens |

The default Q-Former architecture is now `qformer_secondwise_v2`, a lightweight
**Q-Former-style resampler** trained from scratch, not pretrained BLIP-2 weights.
Short and Long have separate trainable attention/FFN/projection parameters.
Short has **one** learned query, reused each second. It reads immutable past
Short states plus itself with self-attention, then cross-attends **only this
frame's patches**. Its final internal state is appended to a 512-d cache;
a separate LayerNorm/Linear produces the D-d output for the existing summary
and Long interfaces. The cache never projects D-d outputs back into 512-d.
It survives 10-second window boundaries and resets every minute/video.

Local decodes the latest 10 output tokens, Block the current 60, and Global
the updated 60 Long tokens. **Local tokens now include earlier context within
the current minute**; they are causal but are not isolated to the labeled
10-second interval. Local targets still describe only that interval. Check
free-running outputs for repetition of earlier evidence. Read
[QFORMER_SECONDWISE.md](QFORMER_SECONDWISE.md) for the API and diagram.

Legacy `qformer_style_v1` 10-query checkpoints retain their original
window-level semantics when loaded for reconstruction. They are not migrated
or used as secondwise training initializations. Stage 3 needs to reset/carry
the new Short cache at the correct boundaries before using this encoder.

## Common supervision and distributed schedule

All six Local, one Block and one Global targets are supervised separately.
Each rank computes one target and one optimizer update per active schedule
position. Whole videos stay on a rank, in chronological order. Ranks are
balanced by target count; shorter schedules participate with zero-gradient
padding, without duplicating examples. Gradients are normalized by the active
rank count. Four ranks do not change the number of distinct supervised targets.

Ordinary 10/60-second visual states are recomputed before each target update.
Secondwise Q-Former replays the causal current-minute prefix from its boundary
to the target endpoint, then returns only the requested output interval.
No Short cache survives an optimizer step; history-dependent Local tokens
match chronological inference under the same weights.
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

## Current frozen release: pull and submit

The accepted release `stage2/releases/accepted_v4_20261105_990b9b48/` is
versioned with this repository: 196 train videos, 41 independent eval videos,
and 23,450 summary targets. The three text files total about 6 MiB; source
videos and model weights remain on cluster storage. Manifest SHA256:
`990b9b489194028ee4437ef64def17ee515d3cf273702c750633c22df834cb4b`.
The reviewed labels include the user's first-pass waivers; this is not a
clinical expert review. Full visual evidence/review ledgers remain in the
original acceptance workspace, bound by manifest hashes.

On SSH:

```bash
cd ~/haoyu/live-ultrasound-video-understanding
git pull --ff-only origin main
GPU_TYPE=auto bash scripts/slurm/submit_stage2_reviewed.sh ordinary
# After the Q-Former functional smoke passes:
GPU_TYPE=auto bash scripts/slurm/submit_stage2_reviewed.sh qformer
```

The reviewed helper fixes four GPUs, one epoch, 24 hours and the rebuilt
Stage1 initialization; it defaults to the versioned release above. It checks
the pinned release hash and uses a submission lock and durable per-architecture
job/run records to avoid duplicates, including recovering a successful sbatch
from its log after a terminal disconnect. Records live under
`$DATA/cluster_data/experiments/stage2/accepted_v4_20261105_990b9b48_submissions/`.
It never silently resumes or resubmits an already recorded run; inspect that
record and use the generic submitter's `RESUME_RUN` for an intentional resume.
`QFORMER_SMOKE_REPORT` must point explicitly to a new secondwise smoke report.
The earlier 10-query smoke `5902188` does not validate this architecture.
Secondwise submission records use `qformer_secondwise_v2.*` so an old recorded
Q-Former job cannot be mistaken for a new-architecture submission. The generic submitter still validates
successful smoke status, model/source/init identity and implementation hashes.
No separate data upload is needed for this release. All mapped source videos
must already be present and byte-identical on cluster storage.

Stage1 generation/P3 overfit export is separately available as
`bash scripts/slurm/stage12_ar_results.sh generate`, then `pack` after jobs
complete. This reuses existing complete outputs, submits missing generation
without a new text Judge, and exports raw predictions, GT and provenance for
local Codex evaluation. Its Stage2 scope is the original P3 overfit run, not
these new full-training runs. The archive path is printed by `pack`.

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
## Dataset versions

Set one field: `DATASET_VERSION=visual_only` or `DATASET_VERSION=visual_asr`.
The old value `visual` is accepted as an alias for `visual_only`.
Both architectures use video frames as model input. ASR enriches the supervision
text in the visual_asr dataset; it is not an added training input.

- visual_only: existing `stage2/releases/accepted_v4_20261105_990b9b48`.
- visual_asr: `$DATA/cluster_data/stage2/releases/accepted_av_asr_v1` if present;
  otherwise the bundled `stage2/releases/accepted_av_asr_v1`.
  `VISUAL_ASR_DATASET_RELEASE` explicitly overrides this location.

Both frozen releases are included in Git. The Visual+ASR release contains
196 train videos / 17,860 targets and 41 eval videos / 5,590 targets.
Only labels and provenance are bundled (about 13 MiB); source videos must
already exist on the cluster. Old Visual-only files are unchanged.
ASR supplies audio-derived text; direct audio listening and clinical expert
review are not claimed by this release.

The local audited snapshot lives at `cluster_data/stage2`. Freeze it once into
a new directory (existing output directories are rejected):

```bash
python -m stage2.freeze_snapshot --snapshot "$DATA/cluster_data/stage2" \
  --output "$DATA/cluster_data/stage2/releases/accepted_av_asr_v1"
```

The snapshot exporter checks hashes, split/source disjointness, full Local
coverage including trailing complete 10-second windows, and full-minute
Block/Global coverage. It preserves the snapshot and writes an immutable
manifest with `dataset_version: visual_asr`. Acceptance is inherited from the
audited snapshot; freezing is not a new visual or clinical review.

On SSH, after `git pull --ff-only origin main` (no separate label upload needed):

```bash
DATASET_VERSION=visual_only bash scripts/slurm/submit_stage2_reviewed.sh ordinary
DATASET_VERSION=visual_asr bash scripts/slurm/submit_stage2_reviewed.sh ordinary
# Use qformer instead of ordinary and set QFORMER_SMOKE_REPORT for Q-Former.
```

The selection is checked against label modalities and the release manifest,
and saved in submission.json, training_config.json and resume invariants.
A version mismatch fails before submission. These are supervised label
variants, independent of the ordinary/Q-Former architecture choice.
