# Full training with a monitoring holdout

`prepare_full.py` freezes the latest combined training export and the separate
holdout export. It does not submit training. The current snapshot is 171 train
videos / 5,236 QA / 12,385 endpoints, and 16 holdout videos / 464 QA / 994 endpoints.
Nine excluded official train videos are explicitly accounted for. The 41 Test
video IDs are reserved; no new V4 Test export is included.

The configuration waits for the completed four-arm selection/all-pairs workflow.
`full_data.resolve_recipe` refuses to start before that workflow completes and
verifies the selected source configuration hash. It copies only the chosen loss
settings and threshold into `resolved_experiment.json`. It starts from original
pretrained Qwen3.5/BERT/InstructBLIP weights and a fresh optimizer, without loading
an ultrasound overfit checkpoint. A selected CL objective uses every within-task
WAIT–ANSWER pair; pure BCE already supervises every endpoint.

Train alone supplies optimizer batches, gradients and batch-size probes. Holdout
is evaluated at epoch 0 and after each epoch under no-grad/eval mode, restoring
Python/NumPy/PyTorch RNG state afterward. It reports answer CE, readiness losses,
forced answer accuracy, first-trigger answer accuracy, correct timing + content,
WAIT false triggers, ANSWER misses and readiness accuracy. The threshold selected
on the overfit comparison is fixed throughout this run. No Test labels or
holdout-tuned threshold enter training. `checkpoint_best.pth` selects a trained
epoch by holdout time+content correct, then fewer false WAIT triggers, then
readiness accuracy; `checkpoint_latest.pth` retains the latest epoch/optimizer.
The holdout is therefore a development set, not a final test benchmark.

Frozen features are saved once per video on scratch and mmap-loaded with a bounded
handle cache. Endpoints gather only their own 1fps grid plus their decision frame;
other questions' fractional timestamps never enter their input. The trainable
projection/memory is still recomputed using current weights. The BF16 feature
estimate is 27.85 GiB for train and 2.58 GiB for holdout, versus about 2.23 TiB if
all endpoint prefixes were duplicated. This is disk feature storage, not GPU VRAM.

The selected preparation uses **four A100-80GB GPUs, BS64 per GPU, GA1,
nominal global BS256, and 10 epochs**. At seed20261121 the intact question-group
sampler produces 197–198 local batches per epoch; assigning four per synchronized
update gives **50 updates/epoch, 500 total**. The last update has 1–2 nonempty ranks;
empty ranks join gradient synchronization without duplicate/padded training rows.
All ranks still apply the same global optimizer update. Holdout performs zero
optimizer updates. Pair count (6,972 train pairs/epoch) is a different quantity.
The earlier one-GPU preparation is superseded, retained only as a recorded baseline.

`torchrun` starts four processes; each initializes an identical model and fresh
Adam state on its assigned device. Bucketed NCCL gradient averaging happens after
the isolated backward, rather than using DDP autograd hooks, because readiness
uses `autograd.grad` to restrict its parameter updates. Before averaging, local
NTP is weighted by global supervised token counts; BCE by endpoint counts;
contrastive and temporal means by their respective eligible question counts.
This handles WAIT-only and empty ranks without changing the global objective.
Globally unused parameters retain `grad=None`, preserving AdamW behavior.

Feature extraction is sharded by video, with each frozen video tensor stored
once on shared scratch. Holdout endpoints are distributed across all four GPUs
and gathered once for complete chronological first-trigger metrics. RNG
preservation touches only the rank's assigned CUDA device. All four allocated
GPU UUIDs are monitored every200ms; `gpu_summary.json` includes per-GPU statistics.
Only rank0 writes latest/best checkpoints. Per-rank training logs are under
`ranks/rank_N`; rank0 logs contain global loss means and actual global batch counts.

AdamW1e-4,256px,SDPA/FLA enabled and gradient checkpointing disabled are retained.
Each GPU must pass the BS64 longest-history probe plus actual first-epoch batches
with the most distinct observed video frames and highest decoder padding cost; failure aborts
without silently changing the frozen update budget. The four-process CPU/Gloo
test compares gradients and AdamW updates against a single-process global-loss
reference, including variable target lengths, WAIT-only and empty ranks; it also
checks shared feature shards and holdout coverage. This does not substitute for
a full-size four-GPU NCCL/performance validation. No job is submitted by preparation.

Each run contains `data_audit.json`, `update_budget.json`,
`planned_batch_order.json`, exact input hashes, source hashes, frozen `code/`,
`experiment.json` and `train.sbatch`. Use `python -m endtoend.prepare_full --help`
for required paths. After the experiment is approved to run and selection has
completed, record the submission in root `programs.md` and submit its frozen
launcher with `RUN_DIR` and output/error paths in that scratch directory.

Generation/loading validation establishes a usable candidate export, not full
expert review, model-resolution visibility or comprehensive cross-split semantic
deduplication. Earlier unchanged tasks retain their earlier review status. These
limitations are preserved in the data audit rather than relabeling the export as
a fully certified benchmark.
