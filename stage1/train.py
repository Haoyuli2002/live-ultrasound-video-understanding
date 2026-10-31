"""Train one LoRA on three deterministic, paired Stage 1 conditions."""
from __future__ import annotations

import argparse
import gc
import json
import os
from datetime import timedelta
import random
from pathlib import Path

import torch
import torch.multiprocessing
from transformers import Trainer, TrainingArguments
from transformers.trainer_utils import get_last_checkpoint

from .data import CONDITIONS, sample_frames
from .model import encode, load_model
from .runtime import configure_sharing, preflight


class Examples:
    def __init__(self, rows: list[dict], paths: dict[str, str], frame_budget: int,
                 frame_size: int, frame_sampling: str, recent_seconds: float):
        self.rows, self.paths = rows, paths
        self.frame_budget, self.frame_size = frame_budget, frame_size
        self.frame_sampling, self.recent_seconds = frame_sampling, recent_seconds

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, index):
        row = dict(self.rows[index])
        row["frames"] = sample_frames(self.paths[row["video_id"]], row["video_window"],
                                      self.frame_budget, self.frame_size,
                                      sampling=self.frame_sampling,
                                      recent_seconds=self.recent_seconds)
        return row


class Collator:
    def __init__(self, processor, max_asr_chars: int):
        self.processor, self.max_asr_chars = processor, max_asr_chars

    def __call__(self, features):
        if len(features) != 1:
            raise ValueError("Stage 1 supports batch size 1; use gradient accumulation")
        configure_sharing()  # Spawned workers must use the requested sharing strategy.
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


def _is_cuda_oom(error: BaseException) -> bool:
    """True for CUDA out-of-memory errors regardless of the exact exception class."""
    if isinstance(error, torch.cuda.OutOfMemoryError):
        return True
    return (isinstance(error, RuntimeError)
            and "cuda out of memory" in str(error).lower())


def _free_cuda() -> None:
    """Collect failed trainers after leaving the exception handler, then trim CUDA."""
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()


def make_training_arguments(args, output, has_validation):
    return TrainingArguments(
        output_dir=str(output), num_train_epochs=args.epochs,
        max_steps=args.max_steps,
        per_device_train_batch_size=1, per_device_eval_batch_size=1,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        learning_rate=args.learning_rate, remove_unused_columns=False,
        eval_strategy="epoch" if has_validation else "no",
        save_strategy="steps", save_steps=args.save_steps, save_total_limit=3,
        logging_steps=args.logging_steps,
        load_best_model_at_end=False,
        bf16=args.dtype == "bf16", fp16=args.dtype == "fp16",
        dataloader_num_workers=args.dataloader_num_workers,
        dataloader_pin_memory=args.dataloader_num_workers > 0,
        dataloader_persistent_workers=args.dataloader_num_workers > 0,
        gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False},
        ddp_find_unused_parameters=True,
        ddp_broadcast_buffers=False,
        prediction_loss_only=True,
        seed=args.seed, data_seed=args.seed,
        report_to="none")


def build_and_train(args, *, frame_budget, train_rows, val_rows, paths, output,
                    dtype, resume_checkpoint):
    """Build the model/trainer at one frame budget and run training to completion.

    Returns the trained model and processor. Raises on CUDA OOM so the caller can
    retry at a smaller frame budget from the latest checkpoint.
    """
    model, processor = load_model(args.model_name, dtype, args.lora_rank)
    if hasattr(model.config, "use_cache"):
        model.config.use_cache = False
    training_args = make_training_arguments(args, output, bool(val_rows))
    trainer = Trainer(model=model, args=training_args,
                      train_dataset=Examples(train_rows, paths, frame_budget, args.frame_size,
                                             args.frame_sampling, args.recent_seconds),
                      eval_dataset=Examples(val_rows, paths, frame_budget, args.frame_size,
                                            args.frame_sampling, args.recent_seconds) if val_rows else None,
                      data_collator=Collator(processor, args.max_asr_chars))
    trainer.train(resume_from_checkpoint=resume_checkpoint)
    trainer.save_model(str(output / "final"))
    if trainer.is_world_process_zero():
        processor.save_pretrained(str(output / "final"))
    trainer.accelerator.wait_for_everyone()
    return model, processor, trainer


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--train-jsonl", required=True)
    parser.add_argument("--video-path-map", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--model-name", default="Qwen/Qwen3.5-2B")
    parser.add_argument("--validation-fraction", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--epochs", type=float, default=3)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--gradient-accumulation-steps", type=int, default=8)
    parser.add_argument("--frame-budget", type=int, default=120)
    parser.add_argument("--oom-fallback-frame-budgets", type=str, default="48,36,24",
                        help="Comma-separated smaller frame budgets to retry (from a saved "
                             "checkpoint) after a CUDA out-of-memory error; empty to disable")
    parser.add_argument("--frame-size", type=int, default=224)
    parser.add_argument("--frame-sampling", choices=("uniform", "recent_sparse"),
                        default="recent_sparse")
    parser.add_argument("--recent-seconds", type=float, default=15.0)
    parser.add_argument("--max-asr-chars", type=int, default=4000)
    parser.add_argument("--lora-rank", type=int, default=16)
    parser.add_argument("--dtype", choices=["bf16", "fp16", "fp32"], default="bf16")
    parser.add_argument("--max-steps", type=int, default=-1,
                        help="Positive for a bounded smoke run; -1 uses --epochs")
    parser.add_argument("--save-steps", type=int, default=50)
    parser.add_argument("--logging-steps", type=int, default=10)
    parser.add_argument("--resume", action="store_true",
                        help="Resume from the latest checkpoint in --output-dir")
    parser.add_argument("--dataloader-num-workers", type=int, default=6,
                        help="Background DataLoader worker processes for parallel video "
                             "decoding; 0 decodes synchronously in the training process")
    return parser


def setup_distributed():
    """Bind each torchrun worker before any model/CUDA allocations."""
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    rank = int(os.environ.get("RANK", "0"))
    if world_size > 1:
        local_rank = int(os.environ["LOCAL_RANK"])
        torch.cuda.set_device(local_rank)
        # Forking DataLoader workers after NCCL initialization can deadlock.
        torch.multiprocessing.set_start_method("spawn", force=True)
        torch.distributed.init_process_group("nccl", timeout=timedelta(minutes=30))
    return rank, world_size


def validate_resume_topology(output, checkpoint, world_size, accumulation):
    if checkpoint is None:
        return
    config = output / "training_config.json"
    if not config.is_file():
        raise ValueError("Checkpoint has no training_config.json; cannot verify resume settings")
    saved = json.loads(config.read_text(encoding="utf-8"))
    if (saved.get("world_size", 1) != world_size
            or saved.get("gradient_accumulation_steps") != accumulation):
        raise ValueError("Resume requires unchanged world_size and gradient accumulation; "
                         "use a new output directory when switching single-GPU/DDP topology")


def _main():
    parser = build_arg_parser()
    args = parser.parse_args()

    if args.max_steps == 0 or args.max_steps < -1 or args.save_steps < 1 or args.logging_steps < 1:
        raise ValueError("Expected max_steps=-1 or positive, and positive save/logging steps")
    if args.dataloader_num_workers < 0:
        raise ValueError("Expected --dataloader-num-workers >= 0")

    if not torch.cuda.is_available():
        raise RuntimeError("Stage 1 Qwen3.5 training requires CUDA")

    rank, world_size = setup_distributed()
    # Configure every rank; probe only rank 0 on the single supported node.
    if args.dataloader_num_workers > 0:
        configure_sharing()
    if rank == 0:
        preflight(args.dataloader_num_workers)
    if world_size > 1:
        torch.distributed.barrier()
    print(f"Stage 1 rank={rank}/{world_size}; local_rank={os.environ.get('LOCAL_RANK', '0')}; "
          f"effective_batch_size={world_size * args.gradient_accumulation_steps}", flush=True)
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
    resume_checkpoint = get_last_checkpoint(str(output)) if args.resume else None
    validate_resume_topology(output, resume_checkpoint, world_size,
                             args.gradient_accumulation_steps)
    if world_size > 1:
        torch.distributed.barrier()  # All ranks must read the old config before rank 0 replaces it.
    if rank == 0:
        (output / "video_split.json").write_text(json.dumps({
            "seed": args.seed,
            "train_video_ids": sorted(set(videos) - held_out),
            "validation_video_ids": sorted(held_out),
        }, indent=2), encoding="utf-8")
        (output / "validation_samples.jsonl").write_text(
            "".join(json.dumps(row, ensure_ascii=False)+"\n" for row in val_rows), encoding="utf-8")
    dtype = {"bf16": torch.bfloat16, "fp16": torch.float16,
             "fp32": torch.float32}[args.dtype]

    # Frame budgets to try in order: the requested budget first, then any smaller
    # fallbacks. A CUDA out-of-memory error retries at the next smaller budget,
    # resuming from the latest saved checkpoint so completed steps are not lost.
    fallbacks = [int(part) for part in args.oom_fallback_frame_budgets.split(",")
                 if part.strip()]
    if any(budget <= 0 for budget in fallbacks):
        raise ValueError("--oom-fallback-frame-budgets must list positive integers")
    budgets = [args.frame_budget] + [budget for budget in sorted(set(fallbacks), reverse=True)
                                     if budget < args.frame_budget]

    if world_size > 1:
        budgets = [args.frame_budget]
        if rank == 0:
            print("DDP: rank-local OOM retries disabled; lower FRAME_BUDGET and "
                  "resubmit the entire job if any rank runs out of memory", flush=True)

    for attempt, frame_budget in enumerate(budgets):
        resume_checkpoint = get_last_checkpoint(str(output)) if args.resume else None
        if rank == 0:
            print(f"Stage 1 training attempt {attempt+1}/{len(budgets)}: frame_budget={frame_budget}"
                  f"{'' if resume_checkpoint is None else f'; resume={resume_checkpoint}'}", flush=True)
            (output / "training_config.json").write_text(
                json.dumps({**vars(args), "active_frame_budget": frame_budget,
                           "world_size": world_size,
                           "effective_batch_size": world_size * args.gradient_accumulation_steps}, indent=2,
                           ensure_ascii=False), encoding="utf-8")
        try:
            build_and_train(args, frame_budget=frame_budget, train_rows=train_rows,
                            val_rows=val_rows, paths=paths, output=output, dtype=dtype,
                            resume_checkpoint=resume_checkpoint)
            if rank == 0:
                print(f"Stage 1 training completed at frame_budget={frame_budget}", flush=True)
            return
        except Exception as error:  # noqa: BLE001 - re-raised unless it is a CUDA OOM
            if not _is_cuda_oom(error):
                raise
            if world_size > 1:
                raise RuntimeError(
                    f"DDP rank {rank} ran out of CUDA memory at frame_budget={frame_budget}; "
                    "lower FRAME_BUDGET and resubmit all ranks with the same output directory. "
                    "torchrun will terminate the other workers.") from error
            remaining = budgets[attempt+1:]
            if not remaining:
                raise RuntimeError(
                    f"CUDA out of memory at frame_budget={frame_budget} with no smaller "
                    f"fallback left; lower --frame-budget or extend "
                    f"--oom-fallback-frame-budgets") from error
            print(f"CUDA out of memory at frame_budget={frame_budget}; retrying at "
                  f"{remaining[0]} (resume enabled={args.resume})", flush=True)
        # The exception traceback holds build_and_train locals (model/optimizer).
        # Python clears that reference on leaving except; collect only afterwards.
        _free_cuda()



def main():
    try:
        _main()
    finally:
        if torch.distributed.is_initialized():
            torch.distributed.destroy_process_group()


if __name__ == "__main__":
    main()
