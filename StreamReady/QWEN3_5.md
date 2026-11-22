# StreamReady with Qwen3.5

This checkout supports multimodal checkpoints with `config.model_type` equal
to `qwen3_5` (dense) or `qwen3_5_moe` (MoE). The checkpoint metadata selects the
implementation; there are no model-size or directory-name heuristics. Text-only
Qwen3.5 checkpoints are not supported. Qwen2-VL remains the default host.

This is a backbone substitution experiment, not the paper's original Qwen2-VL
result. Model quality and full-size GPU throughput still require validation.

## Environment

Use a separate environment with Python 3.10 or newer:

```bash
pip install -r requirements-qwen3_5.txt
pip install -e . --no-deps
```

CPU verification used Transformers 5.14.1, PyTorch 2.13.0, torchvision 0.28.0,
timm 1.0.28 and NumPy 1.26.4. Missing LAVIS dependencies were installed in an
isolated temporary test directory; the existing ultrasound environment was not
upgraded. `requirements-lock.txt` describes the historical upstream environment,
not this Qwen3.5 environment.

Prepare a Qwen3.5 checkpoint plus the BERT tokenizer/config/initialisation and
InstructBLIP checkpoint used by the original Q-Formers. Downloading Qwen3.5 does
not supply the InstructBLIP weights. A local InstructBLIP file can be selected
with `EXTRA='model.pretrained /path/to/instruct_blip_vicuna7b_trimmed.pth'`.

## Training and evaluation

Run from the StreamReady directory. These commands launch processes; on the
cluster run them inside an allocated GPU job. They are examples, not jobs
submitted by this adaptation. Follow the parent project's `AGENTS.md` and
`programs.md` logging rule when submitting training.

```bash
# Reasoning stage. Other dense/MoE sizes use the same MODEL_TYPE.
MODEL_TYPE=qwen3_5 BACKBONE=/path/to/Qwen3.5-2B \
  STAGE=reasoning bash run_scripts/train.sh

# Readiness stage, using the matching reasoning checkpoint.
MODEL_TYPE=qwen3_5 BACKBONE=/path/to/Qwen3.5-2B \
  STAGE=readiness INIT_FROM=/path/to/reasoning_checkpoint.pth \
  bash run_scripts/train.sh

MODEL_TYPE=qwen3_5 BACKBONE=/path/to/Qwen3.5-2B \
  bash run_scripts/proready_qa/test.sh /path/to/checkpoint.pth

MODEL_TYPE=qwen3_5 BACKBONE=/path/to/Qwen3.5-2B \
  bash run_scripts/benchmarks/eval.sh /path/to/checkpoint.pth
```

For example, the same commands accept a local `Qwen3.5-4B` or
`Qwen3.5-35B-A3B` checkpoint. Actual GPU memory requirements differ. These
launchers use replicated DDP; they do not implement tensor parallelism or
automatic model sharding for checkpoints that cannot fit on one GPU.

Qwen3.5 launcher defaults are `BACKBONE_PRECISION=bf16`, `IMAGE_SIZE=256`;
both can be overridden. All model and dataset image-size settings are passed
together. Training output directories include model family, checkpoint basename
and stage; set `OUTPUT_DIR` explicitly if two local checkpoints share a basename
or for a new experiment. Never resume another backbone's run.

When calling `train.py` directly, use `model.model_type qwen3_5` and override
the dataset processors' `image_size` together with `model.image_size`.
Project YAML dataset values take precedence over model preprocessing defaults.

## Backbone contract

```text
LAVIS video tensor [B,3,T,H,W], default CLIP normalisation
  -> undo CLIP normalisation
  -> checkpoint's official image processor, one independent image per frame
  -> frozen Qwen3.5 merged image tokens [B,T,N,D_visual]
  -> trainable D_visual-to-1408 projection
  -> original dual Q-Former / memory / retrieval / readiness modules (768 wide)
  -> trainable 768-to-D_text projection
  -> frozen Qwen3.5 conditional-generation model
```

`D_visual` and `D_text` come from the checkpoint config. The Q-Former remains
initialised from InstructBLIP. Vision encoding uses `no_grad`; the frozen
language forward keeps autograd enabled so answer loss reaches StreamReady's
prefix. Native generation manages Qwen3.5's mixed linear/full-attention cache,
including beam search. Independent calls clear previous rotary-position state.

The frame size must be divisible by the checkpoint's patch size times spatial
merge size (32 for the usual Qwen3.5 configs). Use the supplied video processors
with their default CLIP mean/std. The adapter applies the checkpoint's own image
normalisation exactly once and does not resize a second time. Frames are encoded
as separate images, preserving the existing one-frame-at-a-time observation
semantics; no temporal patch includes a future frame.

BF16 training uses BF16 autocast and a disabled gradient scaler; FP16 keeps loss
scaling. The Q-Former helper imports, head masks, model initialisation and tied
weights are compatible with Transformers 5. Trained checkpoints with incompatible
parameter shapes raise an error; only generic InstructBLIP initialisation keeps
the original shape-filtering behaviour.

## Feature caching

```bash
python tools/extract_vit_features.py \
  --backbone-model /path/to/Qwen3.5-2B --backbone-precision bf16 \
  --image-size 256 --save-dtype bf16 \
  --frames-root /path/to/frames --output-root /path/to/qwen35_2b_256_features
```

Use a separate cache for each checkpoint, resolution and precision. The extractor
selects the backbone from its config, frees decoder weights and records cache
metadata. It initially loads the complete checkpoint before freeing the decoder.
Existing Qwen2-VL feature caches cannot be reused as Qwen3.5 features.

## Validation

```bash
PYTHONDONTWRITEBYTECODE=1 HF_HUB_OFFLINE=1 OMP_NUM_THREADS=2 \
  python -m unittest discover -s tests -v
```

Tests use actual tiny randomly initialised dense and MoE models, not downloaded
production weights: native preprocessing equivalence, causal frame-prefix
invariance, frozen-backbone/input-prefix gradients, cached and uncached generation,
batch masking, beam search, Q-Former checkpoint loading, zero-initialised fusion,
end-to-end reasoning/readiness gradients, model-config selection and Qwen2-VL
compatibility. They establish interface correctness, not trained-model accuracy.
No production-size GPU smoke or training job has been submitted for this change.
