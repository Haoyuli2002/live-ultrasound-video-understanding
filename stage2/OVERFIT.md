# Single-video Stage 2 overfit check

For the newer Short-only versus independent-target experiments, including
automatic free-running reconstruction and memory controls, see [OVERFIT_AB.md](OVERFIT_AB.md).

This diagnostic uses the existing visual annotation draft for P3FoSFIrfQA,
covering [0,120) seconds. It contains 12 local targets, two independent block
targets and two prefix-global targets. The unlabelled 2.634-second tail is
excluded. These labels are not clinically validated or approved for full training.

Run on the cluster after pulling main:

```bash
cd ~/haoyu/live-ultrasound-video-understanding
GPU_TYPE=auto EPOCHS=20 WALLTIME=04:00:00 \
  bash scripts/slurm/submit_stage2_overfit.sh
```

This experiment initializes directly from the official `Qwen/Qwen3.5-2B`
checkpoint and trains fresh Stage 2 LoRA weights and memory-token embeddings.
It does not merge a Stage 1 adapter. The scripts clear inherited
`STAGE1_ADAPTER`/`STAGE2_ADAPTER` variables so a previous shell configuration
cannot change this initialization. Both reconstruction evaluations reload the
official checkpoint plus the newly trained Stage 2 adapter. The submission
and overfit reports explicitly record `init_adapter: null`.
`DATA` defaults to the project scratch directory under
`/dss/mcmlscratch/04/ge75vid2/haoyu/`.

The script checks the single-video labels and resolves the video using the
standard train-full295 location or existing video maps. Set `VIDEO_FILE` to an
absolute path if needed. `DRY_RUN=1` prepares and validates inputs and prints the
Slurm command without submitting a job.

It requests one H100 or A100 80GB, BF16/SDPA, eight CPUs and 120GB host RAM.
Twenty epochs produce 240 optimizer updates. Each epoch saves an adapter, and
the final adapter is saved in `training/final`. Four hours is an allocation
limit, not a measured runtime estimate. The committed code is snapshotted, and
labels, provenance, configuration, input hash and job ID are kept in a unique
run directory. Uncommitted model/trainer edits are not included.

The job prints log locations and writes `overfit_report.json`. This compares
teacher-forced short/block/global reconstruction losses on the same training
video using the epoch-1 and final checkpoints, both in evaluation mode. Lower
loss shows fitting of these targets; it does not demonstrate generated-summary
accuracy, dependence on visual memory, or held-out generalization. There is no
automatic acceptance threshold. Subsequent checks should include generation
and ablations with mismatched or removed visual memory.

This entry is single-GPU and has no optimizer-state resume. An interrupted run
can retain completed epoch adapters, but rerunning the submit script starts a
new experiment. The 27B teacher generation service is not required.
