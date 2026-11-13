# Two-video Stage 3 overfit, reconstruction and evaluation

The bundled label dataset (source videos and mock weights are not included) contains **P3FoSFIrfQA and CzLE1i4NSoU**, both from the
training split: 12 English QA tasks (six immediate, six delayed) and 57 training
decision rows (45 WAIT, 12 ANSWER). No evaluation-set video is used for fitting.
These are sampled native-resolution annotations with assistant self-review,
not independent clinical ground truth. Source videos are SHA256-checked.

## What Stage 3 receives

The frozen Stage 2 encoder produces one short vector for each causal one-second
frame. Every 60 frames it recursively replaces a 60-token long memory. At a
decision time, Stage 3 receives long memory followed by residual short memory,
the question, and observable query/current times. It does not receive raw video,
ASR, text summaries, task ID, reference answers, gold answer time or evidence
metadata. Gold targets enter only the loss calculation, after the decision
position. Memory is detached. The decoder backbone is an independent copy of
the Stage 2 weight chain, with Stage 3 LoRA/special tokens and a BCE head.

The earlier P3 Stage 2 overfit started from the official Qwen3.5 checkpoint,
without Stage 1. Do not add a Stage 1 adapter when restoring that run. A different
Stage 2 run that did merge Stage 1 must restore its exact Stage 1 checkpoint.
The real runner verifies the original Stage 2 training configuration.

The annotations were made at original resolution. The model runner defaults to
224-pixel padded frames, matching the existing Stage 2 training configuration;
these are separate choices. Changing the real runner's frame size requires a
matching Stage 2 configuration, not just changing annotation files.

## CPU mock run

The local `examples/two_video_overfit/mock_stage2_checkpoint` contains a deliberately
synthetic feature projection and `mock_config.json`, not PEFT/Qwen weights.
It computes simple features from the actual video frames, preserving the
short/long tensor interface and minute boundaries. A tiny byte-level GRU stands
in for the decision model. This validates wiring, optimization, persistence,
generation and metrics; it does not test learned ultrasound understanding.
The Qwen backend explicitly rejects a mock checkpoint.

```bash
# From repository root; use a Python environment with torch, numpy, Pillow, OpenCV.
PYTHON=.venv/bin/python BACKEND=mock \
RUN_DIR=/absolute/path/new_stage3_mock_run \
bash scripts/slurm/run_stage3_overfit.sh
```

The local video map is relative to its directory, so moving the entire dataset
folder remains supported. With separately stored videos, supply `VIDEO_MAP` with
both exact source paths. A different encoding of the same clip fails the source
hash check intentionally, because QA times refer to the verified files.

## Real Qwen overfit on an allocated GPU

```bash
export BACKEND=qwen
export DATA_DIR="$PWD/stage3/examples/two_video_overfit"
export VIDEO_MAP="$DATA_DIR/video_map.json"
export STAGE2_ADAPTER=/absolute/path/stage2_run/training/final
export STAGE2_TRAINING_CONFIG=/absolute/path/stage2_run/training/training_config.json
unset STAGE1_ADAPTER  # correct for the earlier official-initialized P3 overfit
export RUN_DIR=/absolute/path/new_stage3_overfit_run
EPOCHS=20 LEARNING_RATE=0.0001 MAX_NEW_TOKENS=128 \
bash scripts/slurm/run_stage3_overfit.sh
```

For Slurm, `submit_stage3_overfit.sh` validates inputs and snapshots working-tree
code, QA, checkpoint files and provenance, including uncommitted code. It
prepares only by default. With the same paths set:

```bash
GPU_TYPE=auto DRY_RUN=1 bash scripts/slurm/submit_stage3_overfit.sh
# Submit a NEW run when ready; the dry run itself does not submit.
GPU_TYPE=auto DRY_RUN=0 bash scripts/slurm/submit_stage3_overfit.sh
```

One full H100/A100 GPU is requested. Real Qwen loading, memory use and decoding
have not been validated by the CPU mock. The full-prefix cache-free greedy
implementation favors correctness of the training/inference prefix over speed.
No optimizer resume or distributed training is implemented. Use a fresh run
folder; rerunning does not silently overwrite or resume a previous experiment.

## What the runner saves

1. A frozen causal memory cache computed once per video, with source/checkpoint
   hashes and tensor shapes. It is reused across epochs; Stage 2 never trains.
2. `initial`, `epoch-1`, `final` decision checkpoints; `training_metrics.jsonl`.
   Training uses inverse class-frequency resampling of decision rows so the
   45 WAIT rows do not dominate the 12 ANSWER rows. The unique QA ratio remains
   six immediate to six delayed; resampling does not create new QA.
3. `initial_real`, `final_real`, `final_zero` reports and predictions. `real`
   means original memory for the selected backend; with mock it is still mock
   memory. Final evaluation reloads the checkpoint from disk.
4. Teacher-forced BCE/NTP on the training rows, separated from generated-text
   metrics. Lower loss alone is not generated-answer quality.
5. Free-text reconstruction at each gold answer time with ANSWER forced,
   recorded as a **decoder-only diagnostic**. No reference text enters decoding.
6. Causal streaming classification on every second from query to five seconds
   after the gold answer, clipped to video duration. These extra points only
   diagnose late/missed answers; they are not additional training annotations.
   Immediate tasks start at query=answer. Null tasks, when present, stay WAIT
   through their reviewed horizon.

The gate uses a fixed probability threshold of 0.5 unless set before a run.
WAIT emits only `<WAIT>` to match these labels. ANSWER fixes the marker and then
decodes text without a reference. A task delivers only its first predicted
answer. Later raw classifications are marked as counterfactual probes; they
are not repeated user notifications. Reports distinguish early, on-time, late,
missed and false triggers. Missing/early answers receive zero in the reported
end-to-end lexical F1. Reconstruction exact match/word F1/ROUGE-L are lexical
checks, not clinical or visual-faithfulness scores.

Zero-memory evaluation preserves memory token count, the question and clocks.
It is a limited diagnostic, not proof that the model relies on image content.
All evaluations here use the same two training videos/questions: overfitting,
not held-out generalization. There are no untriggered tasks in this particular
12-task subset; the general loader/metrics support them and have unit tests.

## Reconstruct and evaluate an existing Stage 3 checkpoint

```bash
# Set the same BACKEND, source Stage2 chain, DATA_DIR and VIDEO_MAP as training.
EVAL_ONLY=/absolute/path/stage3_run/final \
RUN_DIR=/absolute/path/new_stage3_eval \
bash scripts/slurm/run_stage3_overfit.sh
```

This performs no optimizer steps, recomputes causal memory, and writes real/zero
teacher-forced, free reconstruction and streaming reports. The saved `source_chain.json` enforces equality of Stage 1/2 file hashes, backend,
base model and frame size before loading. Paths may move; weights must match.

CPU checks:

```bash
python -B -m unittest stage3.test_training stage3.test_stage3 stage3.test_overfit
```
