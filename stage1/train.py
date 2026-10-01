"""Train one LoRA on three deterministic, paired Stage 1 conditions."""
from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

import torch
from transformers import Trainer, TrainingArguments

from .data import CONDITIONS, sample_frames
from .model import encode, load_model


class Examples:
    def __init__(self, rows: list[dict], paths: dict[str, str], frame_budget: int,
                 frame_size: int):
        self.rows, self.paths = rows, paths
        self.frame_budget, self.frame_size = frame_budget, frame_size

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, index):
        row = dict(self.rows[index])
        row["frames"] = sample_frames(self.paths[row["video_id"]], row["video_window"],
                                      self.frame_budget, self.frame_size)
        return row


class Collator:
    def __init__(self, processor, max_asr_chars: int):
        self.processor, self.max_asr_chars = processor, max_asr_chars

    def __call__(self, features):
        if len(features) != 1:
            raise ValueError("Stage 1 supports batch size 1; use gradient accumulation")
        row = features[0]
        return encode(self.processor, row, row["frames"],
                      max_asr_chars=self.max_asr_chars)


def validate_rows(rows: list[dict]):
    groups: dict[tuple[str, int], dict[str, dict]] = {}
    for row in rows:
        key = (str(row["video_id"]), int(row["sentence_id"]))
        condition = row["condition"]
        if condition not in CONDITIONS or condition in groups.setdefault(key, {}):
            raise ValueError(f"Invalid or duplicate Stage 1 condition: {key}, {condition}")
        groups[key][condition] = row
    for key, group in groups.items():
        if set(group) != set(CONDITIONS):
            raise ValueError(f"Incomplete three-condition group: {key}")
        before = group["before_with_asr"]
        through = group["through_with_asr"]
        masked = group["before_mask_asr"]
        if (before["target"] != through["target"] or before["target"] != masked["target"]
                or before["video_window"] != masked["video_window"]
                or before["video_window"][1] != before["sentence_window"][0]
                or through["video_window"][1] != through["sentence_window"][1]
                or before["historical_asr"] != through["historical_asr"]
                or not before["historical_asr"] or masked["historical_asr"]
                or not masked["historical_asr_masked"]):
            raise ValueError(f"Stage 1 condition semantics mismatch: {key}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train-jsonl", required=True)
    parser.add_argument("--video-path-map", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--model-name", default="Qwen/Qwen3-VL-2B-Instruct")
    parser.add_argument("--validation-fraction", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--epochs", type=float, default=3)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--gradient-accumulation-steps", type=int, default=8)
    parser.add_argument("--frame-budget", type=int, default=120)
    parser.add_argument("--frame-size", type=int, default=224)
    parser.add_argument("--max-asr-chars", type=int, default=4000)
    parser.add_argument("--lora-rank", type=int, default=16)
    parser.add_argument("--dtype", choices=["bf16", "fp16", "fp32"], default="bf16")
    args = parser.parse_args()

    if not torch.cuda.is_available():
        raise RuntimeError("Stage 1 Qwen3-VL training requires CUDA")
    rows = [json.loads(line) for line in Path(args.train_jsonl).read_text(encoding="utf-8").splitlines()
            if line.strip()]
    validate_rows(rows)
    paths = json.loads(Path(args.video_path_map).read_text(encoding="utf-8"))
    if set(row["video_id"] for row in rows) - paths.keys():
        raise ValueError("video-path-map does not cover every training video")
    videos = sorted({row["video_id"] for row in rows})
    random.Random(args.seed).shuffle(videos)
    if len(videos) >= 2 and args.validation_fraction > 0:
        count = max(1, min(len(videos)-1, round(len(videos)*args.validation_fraction)))
        held_out = set(videos[:count])
    else:
        held_out = set()
    train_rows = [row for row in rows if row["video_id"] not in held_out]
    val_rows = [row for row in rows if row["video_id"] in held_out]
    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    (output / "video_split.json").write_text(json.dumps({
        "seed": args.seed,
        "train_video_ids": sorted(set(videos) - held_out),
        "validation_video_ids": sorted(held_out),
    }, indent=2), encoding="utf-8")
    (output / "validation_samples.jsonl").write_text(
        "".join(json.dumps(row, ensure_ascii=False)+"\n" for row in val_rows), encoding="utf-8")
    (output / "training_config.json").write_text(
        json.dumps(vars(args), indent=2, ensure_ascii=False), encoding="utf-8")
    dtype = {"bf16": torch.bfloat16, "fp16": torch.float16,
             "fp32": torch.float32}[args.dtype]
    model, processor = load_model(args.model_name, dtype, args.lora_rank)
    model.gradient_checkpointing_enable()
    if hasattr(model.config, "use_cache"):
        model.config.use_cache = False
    training_args = TrainingArguments(
        output_dir=str(output), num_train_epochs=args.epochs,
        per_device_train_batch_size=1, per_device_eval_batch_size=1,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        learning_rate=args.learning_rate, remove_unused_columns=False,
        eval_strategy="epoch" if val_rows else "no",
        save_strategy="epoch", load_best_model_at_end=bool(val_rows),
        metric_for_best_model="eval_loss" if val_rows else None,
        bf16=args.dtype == "bf16", fp16=args.dtype == "fp16",
        report_to="none")
    trainer = Trainer(model=model, args=training_args,
                      train_dataset=Examples(train_rows, paths, args.frame_budget, args.frame_size),
                      eval_dataset=Examples(val_rows, paths, args.frame_budget, args.frame_size) if val_rows else None,
                      data_collator=Collator(processor, args.max_asr_chars))
    trainer.train()
    trainer.save_model(str(output / "final"))
    processor.save_pretrained(str(output / "final"))


if __name__ == "__main__":
    main()
