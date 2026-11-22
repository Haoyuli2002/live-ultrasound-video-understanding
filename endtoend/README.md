# Ultrasound QA in one training run

Raw video + question → frozen Qwen3.5 vision encoder → StreamReady visual memory,
Short/Long Q-Former, fusion and projection → frozen Qwen3.5 decoder.
Answer CE (NTP) and the configured readiness loss are trained in the same step,
with readiness gradients restricted to RDY and its head. The current preparer
defaults to contrastive + temporal regularization + weighted BCE; explicit
BCE-only and contrastive-only configurations remain available. No Stage1/2/3
checkpoint is required. This is joint adaptation, not full-backbone finetuning.

`data.py` accepts both new `train_samples.jsonl` and legacy `train.jsonl`, together
with `train_tasks.jsonl` and a server-local `video_map.json`. Source directories
may retain historical stage names; these names impose no training dependency.
Old audio-dependent QA need separate visual sufficiency review; accepting their
schema does not add ASR to the model. Do not merge old and replacement QA for the
same videos by simple concatenation.

Run preparation from the project root:

```bash
python -m endtoend.prepare \
  --source results/stage3_visual_v4_train_review_20261120/combined \
  --video-id P3FoSFIrfQA --video-id 9vW1uo7mKDc --video-id LDIo6xQS7Hc \
  --run-dir cluster_data/experiments/streamready/overfit_3v_NEW_RUN
```

For legacy QA use `--source stage3/releases/visual_asr_20261112` and
`--video-map results/streamready_e2e_setup/legacy_video_map.json`. The latter
rebases the old Mac media paths onto the server's original train videos.
The preparer auto-detects `train.jsonl`; `--samples-name` overrides detection.
Preparation validates all source rows, selects train videos, hashes original
media, and freezes data, configuration and code. Existing runs are never reused.

The three-video overfit evaluates the **same training examples**. It has no
holdout/test split and cannot establish generalization. It reports answer exact
match, readiness accuracy and premature decisions at annotated endpoints.
Forced answer decoding is explicitly marked for answer-quality diagnosis;
production endpoint prediction retains WAIT and never forces a final answer.
No dense-grid latency or unseen-video claims are made.

Frames are sampled at 1 fps from video start plus the current decision endpoint.
Actual presentation timestamps must be <= the requested times. Images are padded
to 256×256 while preserving aspect ratio. Reference answer times, evidence paths,
ASR and review notes never enter the prompt. WAIT rows receive the configured readiness supervision; ANSWER rows
also receive answer CE, regardless of the predicted WAIT/ANSWER gate. Untriggered tasks remain WAIT within their reviewed horizon.
Frozen visual features are computed within the run and cached on CPU; they are
not a learned intermediate-stage checkpoint. The parallel path shares causal memory replay across endpoints of each video
within an optimizer step, using the current projection weights.

Endpoint-only Q-Former evaluation has the same final values as replaying it at
every frame: only the memory tree carries video state. All frames still update
that tree. Short-window gradients reach the vision projection; EMA history and
top-k membership remain nondifferentiable. The zero-initialized fusion opens its
Short gradient path after learning starts. Contextual QA memory is disabled for
these independent tasks rather than left as an untrained module.

Training defaults to one GPU, bf16, AdamW 1e-4 and 12 epochs. The preparer probes
batch sizes 8/16/32/48/64; frozen comparison runs use BS64, GA1 and 100 epochs.
`zero_update_warmup: true` allocates zero Adam moments without an extra training
update for matched fresh-initialization comparisons. Legacy configurations can
still use an optimizer warmup update, so compare their actual update counts.
Batched Q-Formers and shared video replay are enabled; GPU safety/throughput
probes decide batch size from measured memory and runtime.

`logs/gpu.csv` samples the assigned GPU at the interval in the frozen launcher
(200 ms in the current comparison). `batch_profiles.json`,
`batch_decision.json` and per-step `training.jsonl` record allocator peaks,
throughput, losses and gradient norms. `eval_metrics.jsonl` and per-epoch
prediction files support overfit review. `checkpoint_latest.pth` contains the
trained core, optimizer state and config; pretrained backbone weights stay pinned
externally. Automatic resume is intentionally not enabled.

Before submitting `endtoend/run_overfit.sbatch`, record the experiment in root
`programs.md`; after submission record the actual job ID and status. Use a frozen
copy of the script and set `RUN_DIR` to its run. Queue acceptance is not training
completion. This launcher requests one H100/A100-80GB for up to two hours.

Validation:

```bash
HF_HUB_OFFLINE=1 PYTHONDONTWRITEBYTECODE=1 \
PYTHONPATH=results/streamready_e2e_setup/deps:StreamReady:. \
python -m unittest endtoend.test_endtoend -v
```

Tests cover actual tiny-model backward, two independently reset streams in one
batch, endpoint/sequential equivalence across memory evictions, causal PTS
selection, answer first-token retention, prompt leakage, premature-label rejection
and untriggered WAIT behavior. Full-size GPU validation is recorded per run.

`gradient_strategy: readiness_isolated` is the default; `sum` retains the old
unrestricted joint-gradient control. Routing and supervision are independent:
our timing supervision uses real endpoint labels, while the paper uses pseudo
labels. The supported loss choices and defaults are described below. The paper
does not explicitly mandate two sequential training stages.

For matched continuation, `init_checkpoint` restores the complete core and
`restore_optimizer: true` restores Adam moments; no extra warmup update is made.
Both arms can use `sampler: question_groups` to keep a question's endpoints in one
batch without duplicating rows. The batch must fit the largest group. A fixed
`batch_candidates: [8]` probes safety without changing the batch between arms.
`batch_order.jsonl` records every sample ID for comparison. New evaluation fields
report answer recall, paired ordering, readiness AUC and **gated** answer match
alongside the forced-answer probe. All remain training-set overfit metrics.

Batch sizing can use `profile_only: true` with `init_checkpoint` and
`restore_optimizer: true`. This skips evaluation/training and performs zero
optimizer updates. Set `batch_candidates`, `profile_warmups` and
`profile_repeats`; compare the recorded backward throughput and peak memory.
The current matched A/B run retains its frozen batch size.

New runs use `parallel_endpoints: true`, `fast_linear_attention: true` and
`model.gradient_checkpointing: false`. Each Q-Former branch processes the whole
batch with explicit padding masks. Unique frames are projected once per video;
integer-grid memory is advanced once within the forward, with snapshot forks for
fractional endpoints. An endpoint cannot see another endpoint's extra sampled
frame. Projected memory never persists across an optimizer step, because its
projection weights are trainable. Frozen encoder features remain reusable.

Q-Former attention uses PyTorch SDPA; eager attention remains available for
attention-map inspection, relative-position variants and head masking. Qwen3.5
linear attention and convolution use explicit FLA/Triton adapters. The launcher
adds the isolated scratch installation (`fla-core==0.6.0`,
`flash-linear-attention==0.6.0`) to PYTHONPATH; the base PyTorch environment is not
modified. Requested acceleration fails visibly if unavailable. The compute-node
tests in `test_acceleration_cuda.py` check kernel output, input gradients and
cached convolution-state updates. `execution_mode.json` and
`attention_backend.json` record the actual selected path.

Use `profile_real_batches: true` to measure actual question-group batches;
reported throughput uses their actual row count. Synthetic worst-case probes
repeat one longest-history ANSWER row and therefore benefit more from shared
video replay; report both workloads when evaluating the optimization.

## Checkpoint storage on the remote server

All future training runs live under
`/dss/mcmlscratch/04/ge75vid2/haoyu/live-ultrasound-video-understanding/cluster_data/experiments/streamready/`.
The repository `cluster_data` symlink resolves to this scratch filesystem.
`endtoend.train` resolves `--run-dir` and rejects paths outside this account's
scratch directory before loading the model. `checkpoint_latest.pth` and its
atomic `.partial` file therefore stay with the frozen run on scratch.
This includes optimizer state; the full checkpoint remains resumable.
Existing historical home checkpoints are not migrated by this change.

## ProReady-QA answer scoring

`answer_metrics.py` implements official normalized/numeric/short-phrase matching.
Training reports forced and gated answer accuracy plus sparse first-trigger
diagnostics; historical exact-match metrics remain. No dense-streaming ARS claim.

## Weighted BCE alongside contrastive readiness

New `endtoend.prepare` runs use `readiness_objective: contrastive_bce_gold`,
`readiness_bce_positive_weight: 2.0`, `readiness_bce_weight: 1.0`,
`timing_lambda_reg: 0.1` and the question-group sampler. The objective is
answer CE + readiness_loss_weight * (contrastive + 0.1 TV + weighted BCE).
For BCE, ANSWER=1 has weight2 and WAIT=0 weight1; mean reduction divides by
endpoint count, not the sum of weights. The decoder answer CE stays weight1.
Readiness gradients remain isolated to RDY/head. Existing explicitly configured
`bce` and `contrastive_gold` runs retain their prior behavior.

Logs include unweighted diagnostic BCE, weighted BCE, contrastive and temporal
components separately; `training_objective_fullset` uses the same hybrid formula.
The hybrid is a gold-label adaptation, not the paper's pure contrastive objective.

## Matched fresh-initialization loss comparison

`run_loss_comparison.py` runs three independent processes sequentially in one
A100 allocation: NTP+BCE (unweighted), NTP+CL+0.1TV, and
NTP+CL+0.1TV+BCE (ANSWER2/WAIT1). All use original pretrained Qwen/BERT/InstructBLIP
assets, identical new-module seed, no ultrasound checkpoint/optimizer restore.
`comparison_controls.py` verifies initial core fingerprints and allocates AdamW
zero moments at step0 without a warmup optimizer update. Each arm uses100epochs,
300updates,BS64,GA1 and identical grouped batches/per-step seeds. Final reporting
rejects differing initial hashes, unexpected updates or mismatched batch orders.
The suite emits `comparison_results.json` and `.md`, including fixed thresholds
0.5/0.35 and a training-set threshold scan. Timing remains sparse endpoint replay,
not dense ARS. This compares three full loss recipes including the stated
positive-weight difference, with one seed and no holdout.

## Full WAIT–ANSWER pairing

See [ALL_PAIRS.md](ALL_PAIRS.md) for `timing_pair_mode: all_pairs`, four-arm
selection, matched fresh reruns and pair-coverage audits.
