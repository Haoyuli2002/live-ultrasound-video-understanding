"""
LoRA pretraining for Qwen3.5 on ultrasound ASR caption completion.

This is independent from QA/train. It does NOT add <WAIT>/<ANSWER> special
tokens; the task is pure narration continuation, so the vocab stays unchanged
and only LoRA adapters are trained (small, fast, low memory).

Example (T4 friendly):

python pretrain/train.py \
  --model-name Qwen/Qwen3.5-2B \
  --train-jsonl pretrain/data/pretrain_samples.jsonl \
  --video-path-map pretrain/data/video_path_map.json \
  --output-dir /mnt/cache/qwenFT/pretrain_qwen3vl_bf16 \
  --window-size 120 \
  --frame-size 224 \
  --num-train-epochs 3 \
  --per-device-train-batch-size 1 \
  --gradient-accumulation-steps 8 \
  --learning-rate 1e-4 \
  --bf16 \
  --gradient-checkpointing
"""

from __future__ import annotations

import os

os.environ.setdefault("TRANSFORMERS_NO_TF", "1")
os.environ.setdefault("USE_TF", "0")
os.environ.setdefault("USE_FLAX", "0")

import argparse
from pathlib import Path
import re
import random
import json

import torch
from torch.utils.data import Subset
from transformers import (
    AutoProcessor,
    Trainer,
    TrainingArguments,
    EarlyStoppingCallback,
)

try:
    from transformers import AutoModelForVision2Seq
except Exception:
    AutoModelForVision2Seq = None

try:
    from transformers import AutoModelForImageTextToText
except Exception:
    AutoModelForImageTextToText = None

try:
    from peft import LoraConfig, get_peft_model
except ImportError as e:
    raise ImportError("PEFT is required. Install with: pip install peft") from e

try:
    from .dataset import PretrainCaptionDataset
    from .collator import PretrainCollator, DEFAULT_SYSTEM_PROMPT
except ImportError:
    from dataset import PretrainCaptionDataset
    from collator import PretrainCollator, DEFAULT_SYSTEM_PROMPT


def load_qwen_vl_model(model_name: str, dtype, attn_implementation: str | None = None):
    kwargs = {"trust_remote_code": True, "dtype": dtype}
    if attn_implementation:
        kwargs["attn_implementation"] = attn_implementation

    errors = []

    if AutoModelForImageTextToText is not None:
        try:
            return AutoModelForImageTextToText.from_pretrained(model_name, **kwargs)
        except Exception as e:
            errors.append(("AutoModelForImageTextToText", repr(e)))

    if AutoModelForVision2Seq is not None:
        try:
            return AutoModelForVision2Seq.from_pretrained(model_name, **kwargs)
        except Exception as e:
            errors.append(("AutoModelForVision2Seq", repr(e)))

    from transformers import AutoModelForCausalLM
    try:
        return AutoModelForCausalLM.from_pretrained(model_name, **kwargs)
    except Exception as e:
        errors.append(("AutoModelForCausalLM", repr(e)))

    msg = "\n".join(f"{name}: {err}" for name, err in errors)
    raise RuntimeError(f"Could not load model {model_name}:\n{msg}")


def maybe_freeze_vision(model, freeze_vision: bool):
    if not freeze_vision:
        return
    frozen = 0
    for name, param in model.named_parameters():
        lname = name.lower()
        if "visual" in lname or "vision" in lname or "vision_tower" in lname:
            param.requires_grad = False
            frozen += param.numel()
    print(f"[pretrain] Frozen vision parameters: {frozen:,}")


def build_lora_model(args, model):
    target_modules = [x.strip() for x in args.lora_target_modules.split(",") if x.strip()]
    config = LoraConfig(
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=target_modules,
    )
    model = get_peft_model(model, config)
    model.print_trainable_parameters()
    return model


def find_latest_checkpoint(output_dir: str | Path) -> str | None:
    """Return the latest checkpoint-* directory in output_dir, if any."""
    out_dir = Path(output_dir)
    if not out_dir.exists():
        return None

    candidates = []
    for path in out_dir.glob("checkpoint-*"):
        if not path.is_dir():
            continue
        match = re.match(r"checkpoint-(\d+)$", path.name)
        if not match:
            continue
        candidates.append((int(match.group(1)), path))

    if not candidates:
        return None

    candidates.sort(key=lambda x: x[0])
    return str(candidates[-1][1])


def split_by_video(rows, validation_fraction: float, seed: int):
    """Keep all samples of a video in the same split."""
    if not 0 <= validation_fraction < 1:
        raise ValueError("--validation-fraction must be in [0, 1)")
    ids = sorted({str(row.get("video_id") or "") for row in rows})
    if not ids or "" in ids:
        raise ValueError("Every Stage 1 sample needs a nonempty video_id")
    if validation_fraction == 0 or len(ids) < 2:
        return list(range(len(rows))), []
    random.Random(seed).shuffle(ids)
    n_val = max(1, min(len(ids) - 1, round(len(ids) * validation_fraction)))
    held_out = set(ids[:n_val])
    train_idx = [i for i, row in enumerate(rows) if str(row["video_id"]) not in held_out]
    val_idx = [i for i, row in enumerate(rows) if str(row["video_id"]) in held_out]
    return train_idx, val_idx


def parse_args():
    parser = argparse.ArgumentParser(description="Qwen3.5 LoRA pretraining (ASR caption completion)")
    parser.add_argument("--model-name", type=str, default="Qwen/Qwen3.5-2B")
    parser.add_argument("--train-jsonl", type=str, required=True)
    parser.add_argument("--output-dir", type=str, default="pretrain/checkpoints/pretrain_qwen3vl")

    parser.add_argument("--repo-root", type=str, default=".")
    parser.add_argument("--video-root", type=str, default=None)
    parser.add_argument("--default-video-path", type=str, default=None)
    parser.add_argument("--video-path-map", type=str, default=None)

    parser.add_argument("--window-size", type=int, default=120,
                        help="Maximum sampled frames across the preceding video window.")
    parser.add_argument("--frame-size", type=int, default=224)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--allow-context-input", action="store_true",
                        help="Allow legacy samples containing previous narration; disabled for the visual-only Stage 1 mainline.")
    parser.add_argument("--validation-fraction", type=float, default=0.1,
                        help="Hold out this fraction of videos for validation; use 0 to disable.")
    parser.add_argument("--split-seed", type=int, default=42)

    parser.add_argument("--num-train-epochs", type=float, default=3.0)
    parser.add_argument("--per-device-train-batch-size", type=int, default=1)
    parser.add_argument("--gradient-accumulation-steps", type=int, default=8)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--warmup-ratio", type=float, default=0.03)
    parser.add_argument("--logging-steps", type=int, default=1)
    parser.add_argument("--save-steps", type=int, default=200)
    parser.add_argument("--save-total-limit", type=int, default=2)

    parser.add_argument("--bf16", action="store_true")
    parser.add_argument("--fp16", action="store_true")
    parser.add_argument("--gradient-checkpointing", action="store_true")
    parser.add_argument("--attn-implementation", type=str, default=None)

    parser.add_argument("--report-to", type=str, default="tensorboard")
    parser.add_argument("--logging-dir", type=str, default=None)

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

    parser.add_argument("--early-stop-patience", type=int, default=3)
    parser.add_argument("--early-stop-min-delta", type=float, default=0.001)
    parser.add_argument("--disable-early-stop", action="store_true")

    parser.add_argument(
        "--resume-from-checkpoint",
        type=str,
        default=None,
        help=(
            "Resume training from a checkpoint path, or use 'auto' to resume from "
            "the latest checkpoint-* directory under --output-dir."
        ),
    )

    return parser.parse_args()


def main():
    args = parse_args()
    if args.per_device_train_batch_size != 1:
        raise ValueError("Stage 1 multimodal training currently requires --per-device-train-batch-size 1")

    if args.bf16 and args.fp16:
        raise ValueError("Use only one precision flag: --bf16 or --fp16, not both.")

    if args.bf16 and torch.cuda.is_available():
        is_bf16_supported = getattr(torch.cuda, "is_bf16_supported", lambda: False)
        if not is_bf16_supported():
            raise ValueError("This CUDA device does not support bf16. Use --fp16 instead.")

    if args.bf16:
        dtype = torch.bfloat16
    elif args.fp16:
        dtype = torch.float16
    else:
        dtype = torch.float32

    print(f"[pretrain] Loading processor: {args.model_name}")
    processor = AutoProcessor.from_pretrained(args.model_name, trust_remote_code=True)
    tokenizer = getattr(processor, "tokenizer", processor)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    print(f"[pretrain] Loading model: {args.model_name}")
    model = load_qwen_vl_model(args.model_name, dtype=dtype, attn_implementation=args.attn_implementation)

    if args.gradient_checkpointing:
        model.gradient_checkpointing_enable()
        if hasattr(model.config, "use_cache"):
            model.config.use_cache = False

    maybe_freeze_vision(model, args.freeze_vision)
    model = build_lora_model(args, model)

    dataset = PretrainCaptionDataset(
        args.train_jsonl,
        repo_root=args.repo_root,
        video_root=args.video_root,
        default_video_path=args.default_video_path,
        video_path_map=args.video_path_map,
        window_size=args.window_size,
        frame_size=args.frame_size,
        limit=args.limit,
    )
    if not args.allow_context_input:
        contaminated = [i for i, row in enumerate(dataset.rows)
                        if row.get("prev_context") or row.get("sample_type") in {
                            "pretrain_caption_sentence_interleave", "pretrain_next_sentence_mixedmask"}]
        if contaminated:
            raise ValueError(
                f"{len(contaminated)} samples contain previous narration or use a legacy context format. "
                "Rebuild with --format standard --no-context, or explicitly pass --allow-context-input."
            )
    train_idx, val_idx = split_by_video(dataset.rows, args.validation_fraction, args.split_seed)
    train_dataset = Subset(dataset, train_idx)
    eval_dataset = Subset(dataset, val_idx) if val_idx else None
    print(f"[pretrain] video-level split: train={len(train_idx)}, validation={len(val_idx)}")
    split_path = Path(args.output_dir) / "video_split.json"
    split_path.parent.mkdir(parents=True, exist_ok=True)
    split_path.write_text(json.dumps({
        "seed": args.split_seed,
        "validation_fraction": args.validation_fraction,
        "train_video_ids": sorted({dataset.rows[i]["video_id"] for i in train_idx}),
        "validation_video_ids": sorted({dataset.rows[i]["video_id"] for i in val_idx}),
    }, ensure_ascii=False, indent=2), encoding="utf-8")
    if val_idx:
        validation_path = split_path.parent / "validation_samples.jsonl"
        with validation_path.open("w", encoding="utf-8") as f:
            for i in val_idx:
                f.write(json.dumps(dataset.rows[i], ensure_ascii=False) + "\n")
        print(f"[pretrain] held-out inference samples: {validation_path}")
    if not val_idx:
        print("[pretrain] No video-level validation split (need at least two videos and nonzero fraction).")

    collator = PretrainCollator(processor=processor, system_prompt=DEFAULT_SYSTEM_PROMPT)

    report_to = [x.strip() for x in args.report_to.split(",") if x.strip() and x.strip().lower() != "none"]
    logging_dir = args.logging_dir or str(Path(args.output_dir) / "runs")
    if report_to:
        print(f"[pretrain] Logging metrics to: {report_to} (logging_dir={logging_dir})")
        if "tensorboard" in report_to:
            print(f"[pretrain] View with: tensorboard --logdir {logging_dir}")
    else:
        print("[pretrain] Metric logging disabled (report_to=none)")

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
        eval_strategy="epoch" if eval_dataset is not None else "no",
        save_strategy="epoch" if eval_dataset is not None else "steps",
        load_best_model_at_end=eval_dataset is not None,
        metric_for_best_model="eval_loss" if eval_dataset is not None else None,
        greater_is_better=False if eval_dataset is not None else None,
        remove_unused_columns=False,
        bf16=args.bf16,
        fp16=args.fp16,
        report_to=report_to,
        logging_dir=logging_dir,
    )

    print("[pretrain] Dataset size:", len(train_dataset))
    print("[pretrain] First sample summary:")
    first = dataset.rows[train_idx[0]]
    print({
        "sample_type": first.get("sample_type"),
        "video_id": first.get("video_id"),
        "video_window": first.get("video_window"),
        "prev_context": (first.get("prev_context", "") or "")[:80],
        "target": (first.get("target", "") or "")[:120],
    })

    callbacks = []
    if not args.disable_early_stop and eval_dataset is not None:
        callbacks.append(EarlyStoppingCallback(
            early_stopping_patience=args.early_stop_patience,
            early_stopping_threshold=args.early_stop_min_delta,
        ))
        print(
            f"[pretrain] Early stopping enabled: patience={args.early_stop_patience}, "
            f"min_delta={args.early_stop_min_delta} (validation loss)"
        )
    else:
        print("[pretrain] Early stopping disabled")

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=collator,
        callbacks=callbacks,
    )

    resume_checkpoint = None
    if args.resume_from_checkpoint:
        if args.resume_from_checkpoint.lower() == "auto":
            resume_checkpoint = find_latest_checkpoint(args.output_dir)
            if resume_checkpoint is None:
                print(f"[pretrain] No checkpoint found under {args.output_dir}; starting from scratch.")
        else:
            resume_checkpoint = args.resume_from_checkpoint
            if not Path(resume_checkpoint).exists():
                raise FileNotFoundError(f"resume checkpoint not found: {resume_checkpoint}")

    if resume_checkpoint:
        print(f"[pretrain] Resuming training from checkpoint: {resume_checkpoint}")
        trainer.train(resume_from_checkpoint=resume_checkpoint)
    else:
        trainer.train()

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"[pretrain] Saving LoRA adapter to {out_dir}")
    trainer.save_model(str(out_dir))
    processor.save_pretrained(str(out_dir))
    print("[pretrain] Done")


if __name__ == "__main__":
    main()
