"""
LoRA SFT for Qwen3-VL-2B on QA WAIT/ANSWER samples.

Recommended first run (single GPU, batch size 1):

python QA/train/train.py \
  --model-name Qwen/Qwen3-VL-2B-Instruct \
  --train-jsonl QA/results/8V649L5Q368_training_samples.jsonl \
  --default-video-path UltrasoundCrawler_KeyCode_20260323_v2/output/20260520_162816_youtube/media/case_reasoning/8V649L5Q368.mp4 \
  --output-dir QA/checkpoints/qwen3vl_2b_lora_wait_answer \
  --window-size 8 \
  --frame-size 448 \
  --num-train-epochs 1 \
  --per-device-train-batch-size 1 \
  --gradient-accumulation-steps 8 \
  --learning-rate 1e-4 \
  --bf16

Notes:
  - vLLM is for inference/serving, not this SFT training loop.
  - This script uses HuggingFace Transformers + PEFT LoRA.
  - Multimodal batching is model/processor-specific; batch_size=1 is the
    reliable v1 setting.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import torch
from transformers import (
    AutoProcessor,
    Trainer,
    TrainingArguments,
)

try:
    from transformers import AutoModelForVision2Seq
except Exception:  # older transformers fallback
    AutoModelForVision2Seq = None

try:
    from transformers import AutoModelForImageTextToText
except Exception:
    AutoModelForImageTextToText = None

try:
    from peft import LoraConfig, get_peft_model
except ImportError as e:
    raise ImportError(
        "PEFT is required for LoRA training. Install with: pip install peft"
    ) from e

try:
    from .dataset import QATrainingDataset
    from .collator import QwenVLCollator, DEFAULT_SYSTEM_PROMPT
except ImportError:
    from dataset import QATrainingDataset
    from collator import QwenVLCollator, DEFAULT_SYSTEM_PROMPT


SPECIAL_TOKENS = ["<WAIT>", "<ANSWER>"]


def load_qwen_vl_model(model_name: str, torch_dtype, attn_implementation: str | None = None):
    kwargs = {
        "trust_remote_code": True,
        "torch_dtype": torch_dtype,
    }
    if attn_implementation:
        kwargs["attn_implementation"] = attn_implementation

    errors = []

    if AutoModelForVision2Seq is not None:
        try:
            return AutoModelForVision2Seq.from_pretrained(model_name, **kwargs)
        except Exception as e:
            errors.append(("AutoModelForVision2Seq", repr(e)))

    if AutoModelForImageTextToText is not None:
        try:
            return AutoModelForImageTextToText.from_pretrained(model_name, **kwargs)
        except Exception as e:
            errors.append(("AutoModelForImageTextToText", repr(e)))

    from transformers import AutoModelForCausalLM
    try:
        return AutoModelForCausalLM.from_pretrained(model_name, **kwargs)
    except Exception as e:
        errors.append(("AutoModelForCausalLM", repr(e)))

    msg = "\n".join(f"{name}: {err}" for name, err in errors)
    raise RuntimeError(f"Could not load model {model_name}. Tried multiple Auto classes:\n{msg}")


def add_special_tokens_if_needed(model, processor):
    tokenizer = getattr(processor, "tokenizer", processor)

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    existing_vocab = tokenizer.get_vocab()
    tokens_to_add = [t for t in SPECIAL_TOKENS if t not in existing_vocab]
    if tokens_to_add:
        tokenizer.add_special_tokens({"additional_special_tokens": tokens_to_add})
        model.resize_token_embeddings(len(tokenizer))
        print(f"[train] Added special tokens: {tokens_to_add}")
    else:
        print("[train] Special tokens already present")

    # Print the resolved ids so eval/inference can align if needed.
    wait_id = tokenizer.convert_tokens_to_ids("<WAIT>")
    answer_id = tokenizer.convert_tokens_to_ids("<ANSWER>")
    print(f"[train] <WAIT> id={wait_id}, <ANSWER> id={answer_id}, vocab={len(tokenizer)}")

    return tokenizer


def maybe_freeze_vision(model, freeze_vision: bool):
    if not freeze_vision:
        return

    frozen = 0
    for name, param in model.named_parameters():
        lname = name.lower()
        if "visual" in lname or "vision" in lname or "vision_tower" in lname:
            param.requires_grad = False
            frozen += param.numel()
    print(f"[train] Frozen vision parameters: {frozen:,}")


def _resolve_modules_to_save(model, wanted):
    """
    Return the subset of `wanted` module suffixes that actually match module
    names in the model. PEFT `modules_to_save` matches by name suffix, so we
    verify each wanted name appears as a submodule; warn if not found.
    """
    all_names = [name for name, _ in model.named_modules()]
    resolved = []
    for w in wanted:
        hits = [n for n in all_names if n.split(".")[-1] == w]
        if hits:
            resolved.append(w)
            print(f"[train] modules_to_save '{w}' -> matched {len(hits)} module(s), e.g. {hits[0]}")
        else:
            print(f"[train] WARNING: modules_to_save '{w}' not found; new <WAIT>/<ANSWER> embeddings may not train.")
    return resolved


def build_lora_model(args, model):
    target_modules = [x.strip() for x in args.lora_target_modules.split(",") if x.strip()]

    # New special tokens (<WAIT>/<ANSWER>) need their embedding + output rows
    # to be trainable. Make embed_tokens and lm_head fully trainable + saved.
    wanted_save = [x.strip() for x in args.lora_modules_to_save.split(",") if x.strip()]
    modules_to_save = _resolve_modules_to_save(model, wanted_save) if wanted_save else None
    if modules_to_save:
        print(f"[train] modules_to_save (fully trainable + saved): {modules_to_save}")
    else:
        print("[train] modules_to_save disabled; new-token embeddings will NOT train.")

    config = LoraConfig(
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=target_modules,
        modules_to_save=modules_to_save,
    )
    model = get_peft_model(model, config)
    model.print_trainable_parameters()
    return model


def parse_args():
    parser = argparse.ArgumentParser(description="Qwen3-VL LoRA SFT for QA WAIT/ANSWER")
    parser.add_argument("--model-name", type=str, default="Qwen/Qwen3-VL-2B-Instruct")
    parser.add_argument("--train-jsonl", type=str, required=True)
    parser.add_argument("--output-dir", type=str, default="QA/checkpoints/qwen3vl_2b_lora_wait_answer")

    parser.add_argument("--repo-root", type=str, default=".")
    parser.add_argument("--video-root", type=str, default=None)
    parser.add_argument("--default-video-path", type=str, default=None)
    parser.add_argument("--video-path-map", type=str, default=None)

    parser.add_argument("--window-size", type=int, default=8)
    parser.add_argument("--frame-size", type=int, default=448)
    parser.add_argument("--limit", type=int, default=None)

    parser.add_argument("--num-train-epochs", type=float, default=1.0)
    parser.add_argument("--per-device-train-batch-size", type=int, default=1)
    parser.add_argument("--gradient-accumulation-steps", type=int, default=8)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--warmup-ratio", type=float, default=0.03)
    parser.add_argument("--logging-steps", type=int, default=1)
    parser.add_argument("--save-steps", type=int, default=100)
    parser.add_argument("--save-total-limit", type=int, default=2)

    parser.add_argument("--bf16", action="store_true")
    parser.add_argument("--fp16", action="store_true")
    parser.add_argument("--gradient-checkpointing", action="store_true")
    parser.add_argument("--attn-implementation", type=str, default=None,
                        help="e.g. flash_attention_2 if installed")

    parser.add_argument("--freeze-vision", action="store_true", default=True)
    parser.add_argument("--no-freeze-vision", action="store_false", dest="freeze_vision")

    parser.add_argument("--lora-r", type=int, default=16)
    parser.add_argument("--lora-alpha", type=int, default=32)
    parser.add_argument("--lora-dropout", type=float, default=0.05)
    parser.add_argument(
        "--lora-target-modules",
        type=str,
        default="q_proj,k_proj,v_proj,o_proj,gate_proj,up_proj,down_proj",
    )
    parser.add_argument(
        "--lora-modules-to-save",
        type=str,
        default="embed_tokens,lm_head",
        help="Modules made fully trainable + saved (needed so new <WAIT>/<ANSWER> "
             "token embeddings can learn). Set empty string to disable.",
    )

    return parser.parse_args()


def main():
    args = parse_args()

    if args.bf16:
        dtype = torch.bfloat16
    elif args.fp16:
        dtype = torch.float16
    else:
        dtype = torch.float32

    print(f"[train] Loading processor: {args.model_name}")
    processor = AutoProcessor.from_pretrained(args.model_name, trust_remote_code=True)

    print(f"[train] Loading model: {args.model_name}")
    model = load_qwen_vl_model(
        args.model_name,
        torch_dtype=dtype,
        attn_implementation=args.attn_implementation,
    )

    add_special_tokens_if_needed(model, processor)

    if args.gradient_checkpointing:
        model.gradient_checkpointing_enable()
        # Avoid use_cache warnings/errors during gradient checkpointing.
        if hasattr(model.config, "use_cache"):
            model.config.use_cache = False

    maybe_freeze_vision(model, args.freeze_vision)
    model = build_lora_model(args, model)

    dataset = QATrainingDataset(
        args.train_jsonl,
        repo_root=args.repo_root,
        video_root=args.video_root,
        default_video_path=args.default_video_path,
        video_path_map=args.video_path_map,
        window_size=args.window_size,
        frame_size=args.frame_size,
        limit=args.limit,
    )

    collator = QwenVLCollator(
        processor=processor,
        system_prompt=DEFAULT_SYSTEM_PROMPT,
    )

    training_args = TrainingArguments(
        output_dir=args.output_dir,
        num_train_epochs=args.num_train_epochs,
        per_device_train_batch_size=args.per_device_train_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        learning_rate=args.learning_rate,
        warmup_ratio=args.warmup_ratio,
        logging_steps=args.logging_steps,
        save_steps=args.save_steps,
        save_total_limit=args.save_total_limit,
        remove_unused_columns=False,
        bf16=args.bf16,
        fp16=args.fp16,
        report_to=[],
    )

    print("[train] Dataset size:", len(dataset))
    print("[train] First sample summary:")
    first = dataset.rows[0]
    print({
        "sample_type": first.get("sample_type"),
        "qa_type": first.get("qa_type"),
        "video_window": first.get("video_window"),
        "question": first.get("question", "")[:120],
        "target": first.get("target", "")[:120],
    })

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=dataset,
        data_collator=collator,
    )

    trainer.train()

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"[train] Saving LoRA adapter to {out_dir}")
    trainer.save_model(str(out_dir))
    processor.save_pretrained(str(out_dir))
    print("[train] Done")


if __name__ == "__main__":
    main()