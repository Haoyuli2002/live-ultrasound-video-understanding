# H100 / A100 training

All three stages load their Qwen3.5 backbones with native BF16 by default and
PyTorch SDPA attention. SDPA selects kernels for the allocated architecture;
FlashAttention 3 and a separately installed `flash-attn` package are not required.
`ATTN_IMPLEMENTATION=flash_attention_2` is an explicit opt-in after installing a
compatible extension. `ATTN_IMPLEMENTATION=eager` is available for debugging.

The portable submitter requests full GPUs, not small MIG slices:

| `GPU_TYPE` | Eligible LRZ partitions |
| --- | --- |
| `h100` | `lrz-hgx-h100-94x4` |
| `a100` | `lrz-dgx-a100-80x8,lrz-hgx-a100-80x4` |
| `auto` (default) | All three partitions above; Slurm selects an allocation |

This selection increases eligible resources; it does not guarantee a shorter
wait or that a particular model/frame budget fits. A100 40GB and A100 MIG20GB
remain separate capacity choices. The Whisper data-preparation jobs continue to
use MIG20GB and are independent of these training launchers.

## Shared entry point

```bash
cd ~/haoyu/live-ultrasound-video-understanding
git pull --ff-only origin main
export TRAIN_JSONL=/absolute/path/training_samples.jsonl
export VIDEO_MAP=/absolute/path/video_map.json
export OUTPUT_DIR=/absolute/path/new_experiment
export GPU_TYPE=auto        # a100 or h100 restricts the GPU family
export DTYPE=bf16
export ATTN_IMPLEMENTATION=sdpa
export EPOCHS=1
export NPROC_PER_NODE=1
bash scripts/slurm/submit_training.sh 1 --dry-run
bash scripts/slurm/submit_training.sh 1
```

Dry-run prints the complete Slurm request without submitting it. `WALLTIME`
(default 24h), `CPUS_PER_TASK` (8) and `MEMORY` (120G host RAM) override resources.
Use a fresh output directory for each smoke. The batch job logs actual GPU name,
compute capability, physical memory, CUDA/PyTorch version and BF16 support before
loading weights. Cache storage uses scratch; IPC paths use a short `/tmp` folder.

For Stage 1, use the new dataset's `samples.jsonl` and matching `video_map.json`.
The defaults are `FRAME_BUDGET=30`, `FRAME_SAMPLING=recent_sparse` and
`RECENT_SECONDS=15`. Set `MAX_STEPS=3` for a bounded smoke. Stage 1 accepts one
or multiple same-node GPUs through `NPROC_PER_NODE`; its existing DDP/checkpoint
rules still apply. `RESUME=1` is supported for Stage 1 with the same topology and
gradient accumulation. Resuming on another GPU family is not a bitwise
reproducibility guarantee.

For Stage 2, point `TRAIN_JSONL` to validated `blocks.jsonl` and set
`STAGE1_ADAPTER=/absolute/path/stage1/final`, then use `submit_training.sh 2`.
This portable entry is single-GPU and uses BF16 or FP32. Existing separate DDP
work is not enabled simply by requesting several GPUs. Test a short dataset
before a full run; Stage 2's published epoch loop has no `MAX_STEPS` argument.

## Stage 3 training

`python -m stage3.train` now reads the existing `training_samples.jsonl` format:
`video_id`, `question`, `target`, `video_window=[0,current_time]` and
`meta.query_time`, `meta.answer_time` (optional matching `meta.current_time`).
It validates WAIT/ANSWER timing and keeps reference annotations out of inputs.

Set both `STAGE1_ADAPTER` and `STAGE2_ADAPTER`, use a QA JSONL and video map,
and call `submit_training.sh 3`. The frozen memory producer and independently
trainable decision model live together on one GPU. Video is replayed once per
video/epoch in chronological order. Integer 1-FPS memory updates persist;
fractional decision times append one temporary latest frame strictly before
that time. Future images and evidence metadata do not enter the decision input.
Training combines BCE and target-only text loss, saves per-epoch/final adapters
and the FP32 BCE head, and supports `MAX_STEPS=3` for a smoke.
Decision times must be greater than zero; an empty visual prefix at time zero
is not currently a supported training sample.

Stage 3 currently has no optimizer-state resume or DDP entry. Its CPU regression
tests verify causal replay, fractional times, frozen-memory gradients and actual
decision-parameter updates; real Qwen adapters and end-to-end GPU memory use
still need an A100/H100 smoke. Existing demonstration QA marked draft still
requires content/timing review before production training.

## Stage 2 teacher generation

Full teacher runs also accept `GPU_TYPE=auto|a100|h100` through
`submit_stage2_full.sh`; their manifests and resume guards remain unchanged.
The server selects the Triton GDN prefill backend when that CLI option exists,
avoiding a Hopper-specific/JIT prefill dependency. `MAX_MODEL_LEN` can be reduced
for a constrained A100 smoke; an actual Qwen3.5-27B GPU request is still required
to validate memory and total context for the chosen video configuration.
Existing queued jobs use their original saved code snapshot and are not changed
by pulling this update. Do not submit a duplicate full run to change GPU type.

Checks:
`python -m unittest scripts.test_gpu_runtime stage3.test_training stage3.test_stage3 stage1.test_stage1 stage1.test_ddp stage2.test_stage2 stage2.test_full_run`.
