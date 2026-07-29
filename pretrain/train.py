"""
LoRA pretraining for Qwen3-VL on ultrasound ASR caption completion.

This is independent from QA/train. It does NOT add <WAIT>/<ANSWER> special
tokens; the task is pure narration continuation, so the vocab stays unchanged
and only LoRA adapters are trained (small, fast, low memory).

Example (T4 friendly):

python pretrain/train.py \
  --model-name Qwen/Qwen3-VL-2B-Instruct \
  --train-jsonl pretrain/data/pretrain_samples.jsonl \
  --video-path-map pretrain/data/video_path_map.json \
  --output-dir /mnt/cache/qwenFT/pretrain_qwen3vl_bf16 \
  --window-size 4 \
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

import torch
from transformers import (
    AutoProcessor,
    Trainer,
    TrainingArguments,
    TrainerCallback,
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


class EpochLossEarlyStopping(TrainerCallback):
    """Early stop on per-epoch average train loss (adjacent-epoch improvement)."""

    def __init__(self, min_delta: float = 0.001, patience: int = 3):
        self.min_delta = float(min_delta)
        self.patience = int(patience)
        self.prev_loss = None
        self.no_improve = 0
        self._epoch_losses = []

    def on_log(self, args, state, control, logs=None, **kwargs):
        if logs and "loss" in logs:
            self._epoch_losses.append(float(logs["loss"]))
        return control

    def on_epoch_end(self, args, state, control, **kwargs):
        if not self._epoch_losses:
            return control

        cur_loss = sum(self._epoch_losses) / len(self._epoch_losses)
        self._epoch_losses = []

        if self.prev_loss is None:
            self.prev_loss = cur_loss
            print(f"[earlystop] epoch={state.epoch:.2f} avg_loss={cur_loss:.4f} (baseline)")
            return control

        improvement = self.prev_loss - cur_loss
        if improvement < self.min_delta:
            self.no_improve += 1
        else:
            self.no_improve = 0
        self.prev_loss = cur_loss

        print(
            f"[earlystop] epoch={state.epoch:.2f} avg_loss={cur_loss:.4f} "
            f"improvement={improvement:.4f} no_improve={self.no_improve}/{self.patience}"
        )

        if self.no_improve >= self.patience:
            print(
                f"[earlystop] Stopping: {self.patience} consecutive epochs with "
                f"loss improvement < {self.min_delta}."
            )
            control.should_training_stop = True
        return control


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


def parse_args():
    parser = argparse.ArgumentParser(description="Qwen3-VL LoRA pretraining (ASR caption completion)")
    parser.add_argument("--model-name", type=str, default="Qwen/Qwen3-VL-2B-Instruct")
    parser.add_argument("--train-jsonl", type=str, required=True)
    parser.add_argument("--output-dir", type=str, default="pretrain/checkpoints/pretrain_qwen3vl")

    parser.add_argument("--repo-root", type=str, default=".")
    parser.add_argument("--video-root", type=str, default=None)
    parser.add_argument("--default-video-path", type=str, default=None)
    parser.add_argument("--video-path-map", type=str, default=None)

    parser.add_argument("--window-size", type=int, default=4)
    parser.add_argument("--frame-size", type=int, default=224)
    parser.add_argument("--limit", type=int, default=None)

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
        remove_unused_columns=False,
        bf16=args.bf16,
        fp16=args.fp16,
        report_to=report_to,
        logging_dir=logging_dir,
    )

    print("[pretrain] Dataset size:", len(dataset))
    print("[pretrain] First sample summary:")
    first = dataset.rows[0]
    print({
        "sample_type": first.get("sample_type"),
        "video_id": first.get("video_id"),
        "video_window": first.get("video_window"),
        "prev_context": (first.get("prev_context", "") or "")[:80],
        "target": (first.get("target", "") or "")[:120],
    })

    callbacks = []
    if not args.disable_early_stop:
        callbacks.append(
            EpochLossEarlyStopping(
                min_delta=args.early_stop_min_delta,
                patience=args.early_stop_patience,
            )
        )
        print(
            f"[pretrain] Early stopping enabled: patience={args.early_stop_patience}, "
            f"min_delta={args.early_stop_min_delta} (per-epoch avg loss)"
        )
    else:
        print("[pretrain] Early stopping disabled")

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=dataset,
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
