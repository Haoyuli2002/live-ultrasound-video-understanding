"""Chronological Stage 2 training: six optimizer updates per full minute."""
from __future__ import annotations

import argparse
import json
import os
import random
from contextlib import nullcontext
from pathlib import Path

import torch

from .data import VideoReader, load_blocks
from .model import MemoryModel


class Stage2TrainStep(torch.nn.Module):
    """One DDP-visible unit: a complete ten-second optimizer step.

    The Qwen backbone is called multiple times inside this forward. Wrapping
    this outer module—not the backbone itself—keeps one DDP collective sequence
    per optimizer step even when minute-boundary losses add extra forwards.
    """
    def __init__(self, memory: MemoryModel):
        super().__init__()
        self.backbone = (memory.core_model if hasattr(memory, "core_model")
                         else memory.model)
        self.memory = memory

    def forward(self, frames, prior_short: list[torch.Tensor], previous_long,
                local_target: str, block_target: str | None,
                global_target: str | None):
        current = [self.memory.encode_frame(frame) for frame in frames]
        short_ten = torch.stack(current, dim=1)
        short_loss = self.memory.reconstruction_loss(
            short_ten, local_target, "short")
        if block_target is not None:
            all_short = torch.cat([*prior_short, short_ten], dim=1)
            block_loss = self.memory.reconstruction_loss(
                all_short, block_target, "block")
            new_long = self.memory.update_long(previous_long, all_short)
            global_loss = self.memory.reconstruction_loss(
                new_long, global_target, "global")
            total = short_loss + block_loss + global_loss
        else:
            block_loss = global_loss = None
            new_long = None
            total = short_loss
        return total, short_loss, block_loss, global_loss, short_ten, new_long


def train_video(memory: MemoryModel, blocks: list[dict], reader: VideoReader,
                optimizer: torch.optim.Optimizer, metrics_file, video_id: str,
                epoch: int, *, max_grad_norm: float = 1.0,
                scheduler: object | None = None, global_step: int = 0,
                step_model: torch.nn.Module | None = None,
                verify_ddp_gradients: bool = False) -> int:
    previous_long = None  # Reset at each video/epoch boundary.
    step_model = step_model or Stage2TrainStep(memory)
    trainable = [p for p in step_model.parameters() if p.requires_grad]
    for block_index, row in enumerate(blocks):
        start, end = row["block_window"]
        short_states: list[torch.Tensor] = []
        for local_index, label in enumerate(row["local_sub_summaries"]):
            interval_start = int(round(start)) + 10 * local_index
            frames = [reader.frame_ending_at(second)
                      for second in range(interval_start, interval_start + 10)]
            last_ten_of_full_block = len(row["local_sub_summaries"]) == 6 and local_index == 5
            total, short_loss, block_loss, global_loss, short_ten, new_long = step_model(
                frames, short_states, previous_long,
                label["local_summary_target"],
                row["block_summary_target"] if last_ten_of_full_block else None,
                row["global_summary_target"] if last_ten_of_full_block else None)
            short_states.append(short_ten)

            optimizer.zero_grad(set_to_none=True)
            total.backward()
            if verify_ddp_gradients and global_step == 0:
                _verify_ddp_gradients(step_model)
            grad_norm = float(torch.nn.utils.clip_grad_norm_(trainable, max_grad_norm)) \
                if max_grad_norm and max_grad_norm > 0 else None
            step_lr = optimizer.param_groups[0]["lr"]
            optimizer.step()
            if scheduler is not None:
                scheduler.step()
            global_step += 1
            metrics_file.write(json.dumps({
                "video_id": video_id, "epoch": epoch, "global_step": global_step,
                "block": block_index,
                "local": local_index, "end_sec": interval_start + 10,
                "short": float(short_loss.detach()),
                "block_loss": None if block_loss is None else float(block_loss.detach()),
                "global": None if global_loss is None else float(global_loss.detach()),
                "total": float(total.detach()),
                "grad_norm": grad_norm,
                "lr": step_lr,
            }) + "\n")
            metrics_file.flush()
            short_states[-1] = short_states[-1].detach()
            if new_long is not None:
                previous_long = new_long.detach()
                short_states.clear()
        # The final <10-second tail has no label and no optimizer step.
        if end - start < 60:
            break
    return global_step


@torch.no_grad()
def evaluate_video(memory: MemoryModel, blocks: list[dict], reader: VideoReader):
    """Teacher-forced reconstruction losses on a held-out video; no optimizer step.

    Mirrors the training schedule's memory recurrence (short every 10s, long
    replacement every full minute) but only accumulates component losses.
    """
    previous_long = None
    totals = {"short": 0.0, "block": 0.0, "global": 0.0}
    counts = {"short": 0, "block": 0, "global": 0}
    for row in blocks:
        start, end = row["block_window"]
        short_states: list[torch.Tensor] = []
        for local_index, label in enumerate(row["local_sub_summaries"]):
            interval_start = int(round(start)) + 10 * local_index
            current = [memory.encode_frame(reader.frame_ending_at(second))
                       for second in range(interval_start, interval_start + 10)]
            short_ten = torch.stack(current, dim=1)
            short_states.append(short_ten)
            totals["short"] += float(memory.reconstruction_loss(
                short_ten, label["local_summary_target"], "short").detach())
            counts["short"] += 1
            if len(row["local_sub_summaries"]) == 6 and local_index == 5:
                all_short = torch.cat(short_states, dim=1)
                totals["block"] += float(memory.reconstruction_loss(
                    all_short, row["block_summary_target"], "block").detach())
                counts["block"] += 1
                new_long = memory.update_long(previous_long, all_short)
                totals["global"] += float(memory.reconstruction_loss(
                    new_long, row["global_summary_target"], "global").detach())
                counts["global"] += 1
                previous_long = new_long.detach()
                short_states.clear()
        if end - start < 60:
            break
    return {k: (totals[k] / counts[k] if counts[k] else None) for k in totals}, counts


_RESUME_INVARIANTS = (
    "model_name", "init_adapter", "train_jsonl", "video_path_map", "frame_size",
    "lora_rank", "dtype", "learning_rate", "max_grad_norm", "seed",
    "validation_fraction", "warmup_ratio", "epochs", "world_size",
)


def _validate_resume_config(saved: dict, current: dict) -> None:
    mismatches = [key for key in _RESUME_INVARIANTS
                  if key not in saved or saved[key] != current[key]]
    if mismatches:
        details = ", ".join(
            f"{key}: saved={saved.get(key)!r}, current={current.get(key)!r}"
            for key in mismatches)
        raise ValueError(f"Cannot resume with a changed experiment configuration: {details}")


def _rng_state() -> dict:
    state = {"python": random.getstate(), "torch": torch.get_rng_state()}
    if torch.cuda.is_available():
        state["cuda"] = torch.cuda.get_rng_state()
    return state


def _restore_rng_state(state: dict) -> None:
    random.setstate(state["python"])
    torch.set_rng_state(state["torch"])
    if torch.cuda.is_available() and state.get("cuda") is not None:
        torch.cuda.set_rng_state(state["cuda"])


def _save_training_state(path: Path, *, completed_epoch: int, global_step: int,
                         optimizer, scheduler, best_metric: float | None,
                         best_epoch: int | None, rng_states: list[dict]) -> None:
    torch.save({
        "completed_epoch": completed_epoch,
        "global_step": global_step,
        "optimizer": optimizer.state_dict(),
        "scheduler": None if scheduler is None else scheduler.state_dict(),
        "rng_states": rng_states,
        "best_metric": best_metric,
        "best_epoch": best_epoch,
    }, path / "trainer_state.pt")


def _assert_expected_trainables(memory: MemoryModel) -> None:
    named = [(name, param) for name, param in memory.core_model.named_parameters()
             if param.requires_grad]
    lora = [name for name, _ in named if "lora_" in name]
    embedding = [name for name, param in memory.model.get_input_embeddings().named_parameters()
                 if param.requires_grad]
    if not lora:
        raise RuntimeError("No trainable Stage 2 LoRA parameters were loaded")
    if not embedding:
        raise RuntimeError(
            "The <SHORT_MEM>/<LONG_MEM> trainable-token parameters are frozen")
    count = sum(param.numel() for _, param in named)
    print(f"[trainable] parameters={count:,}; lora_tensors={len(lora)}; "
          f"embedding_tensors={embedding}", flush=True)


def _init_distributed() -> tuple[int, int, int]:
    """Initialize torchrun DDP when WORLD_SIZE is greater than one."""
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    rank = int(os.environ.get("RANK", "0"))
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    if world_size > 1:
        import torch.distributed as dist
        torch.cuda.set_device(local_rank)
        dist.init_process_group(backend="nccl", init_method="env://")
    return rank, world_size, local_rank


def _balanced_video_shards(videos: dict[str, list[dict]], video_ids: list[str],
                           world_size: int) -> tuple[list[list[str]], list[int]]:
    """Greedily balance whole videos while keeping each video chronological."""
    weighted = sorted(
        ((sum(len(block["local_sub_summaries"]) for block in videos[video_id]),
          video_id) for video_id in video_ids),
        key=lambda item: (-item[0], item[1]))
    bins: list[tuple[list[str], int]] = [([], 0) for _ in range(world_size)]
    for steps, video_id in weighted:
        index = min(range(world_size), key=lambda i: (bins[i][1], i))
        bins[index][0].append(video_id)
        bins[index] = (bins[index][0], bins[index][1] + steps)
    # Rank 0 must have a maximum-length shard because its optimizer state is
    # authoritative after DDP Join handles uneven inputs.
    bins.sort(key=lambda item: (-item[1], item[0]))
    return [item[0] for item in bins], [item[1] for item in bins]


def _sync_optimizer_state(optimizer, scheduler, rank: int, local_rank: int) -> None:
    """Make every rank start the next epoch with rank 0's optimizer state."""
    import torch.distributed as dist
    payload = ([optimizer.state_dict(), scheduler.state_dict()]
               if rank == 0 else [None, None])
    dist.broadcast_object_list(
        payload, src=0, device=torch.device("cuda", local_rank))
    if rank != 0:
        optimizer.load_state_dict(payload[0])
        scheduler.load_state_dict(payload[1])


def _verify_ddp_gradients(step_model: torch.nn.Module) -> None:
    """Check one complete LoRA gradient across ranks before clipping/stepping."""
    import torch.distributed as dist

    selected = next(((name, param) for name, param in step_model.named_parameters()
                     if "lora_" in name and param.requires_grad), None)
    if selected is None:
        raise RuntimeError("DDP gradient check found no trainable LoRA parameter")
    name, param = selected
    present = torch.tensor(int(param.grad is not None), device=param.device)
    dist.all_reduce(present, op=dist.ReduceOp.SUM)
    if int(present.item()) != dist.get_world_size():
        raise RuntimeError(f"DDP gradient check found no gradient for {name}")
    gradient = param.grad.detach().contiguous()
    gathered = [torch.empty_like(gradient) for _ in range(dist.get_world_size())]
    dist.all_gather(gathered, gradient)
    reference = gathered[0].float()
    max_diff = max((reference - item.float()).abs().max().item()
                   for item in gathered[1:])
    if max_diff != 0:
        raise RuntimeError(
            f"DDP LoRA gradient mismatch for {name}: max_abs_diff={max_diff}")
    if dist.get_rank() == 0:
        print(f"[ddp-check] synchronized gradient: {name}; "
              f"norm={reference.norm().item():.6g}; ranks={len(gathered)}",
              flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train-jsonl", required=True)
    parser.add_argument("--video-path-map", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--model-name", default="Qwen/Qwen3.5-2B")
    parser.add_argument("--init-adapter")
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--frame-size", type=int, default=224)
    parser.add_argument("--lora-rank", type=int, default=16)
    parser.add_argument("--dtype", choices=["bf16", "fp16", "fp32"], default="bf16")
    parser.add_argument("--max-grad-norm", type=float, default=1.0,
                        help="Gradient-norm clip per optimizer step; <=0 disables")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--validation-fraction", type=float, default=0.1,
                        help="Fraction of videos held out for teacher-forced eval")
    parser.add_argument("--warmup-ratio", type=float, default=0.03,
                        help="Linear LR warmup as a fraction of total updates")
    parser.add_argument("--resume", action="store_true",
                        help="Resume from the latest epoch checkpoint in --output-dir")
    parser.add_argument("--verify-ddp-gradients", action="store_true",
                        help="On the first update, compare a complete LoRA gradient across ranks")
    args = parser.parse_args()


    if args.epochs < 1:
        raise ValueError("epochs must be at least one")
    if not 0 <= args.validation_fraction < 1:
        raise ValueError("validation_fraction must be in [0, 1)")
    if not 0 <= args.warmup_ratio < 1:
        raise ValueError("warmup_ratio must be in [0, 1)")
    if not torch.cuda.is_available():
        raise RuntimeError("Stage 2 Qwen3.5 training requires a CUDA device")
    rank, world_size, local_rank = _init_distributed()
    random.seed(args.seed + rank)
    torch.manual_seed(args.seed + rank)
    dtype = {"bf16": torch.bfloat16, "fp16": torch.float16,
             "fp32": torch.float32}[args.dtype]
    videos = load_blocks(args.train_jsonl)
    paths = json.loads(Path(args.video_path_map).read_text(encoding="utf-8"))
    missing = sorted(set(videos) - set(paths))
    if missing:
        raise ValueError(f"Missing video paths: {missing[:5]}")

    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    config_path = output / "training_config.json"
    split_path = output / "video_split.json"
    current_config = {**vars(args), "world_size": world_size}
    if args.resume and config_path.is_file():
        _validate_resume_config(
            json.loads(config_path.read_text(encoding="utf-8")), current_config)
        if not split_path.is_file():
            raise FileNotFoundError("Cannot resume without the saved video_split.json")
    elif not args.resume and config_path.exists():
        raise FileExistsError(
            f"Training output already exists: {output}; use --resume or a new directory")

    # Hold out whole videos and persist the split. Resume always reuses it.
    if args.resume and split_path.is_file():
        saved_split = json.loads(split_path.read_text(encoding="utf-8"))
        train_ids = saved_split["train_video_ids"]
        val_ids = saved_split["validation_video_ids"]
        known = set(videos)
        if set(train_ids) | set(val_ids) != known or set(train_ids) & set(val_ids):
            raise ValueError("Saved video_split.json does not match the current dataset")
    else:
        video_ids = sorted(videos)
        random.Random(args.seed).shuffle(video_ids)
        if len(video_ids) >= 2 and args.validation_fraction > 0:
            count = max(1, min(len(video_ids) - 1,
                               round(len(video_ids) * args.validation_fraction)))
            held_out = set(video_ids[:count])
        else:
            held_out = set()
        train_ids = [v for v in sorted(videos) if v not in held_out]
        val_ids = [v for v in sorted(videos) if v in held_out]
        if rank == 0:
            split_path.write_text(json.dumps({
                "seed": args.seed, "train_video_ids": train_ids,
                "validation_video_ids": val_ids,
            }, indent=2), encoding="utf-8")
        if world_size > 1:
            import torch.distributed as dist
            dist.barrier()
    if not train_ids:
        raise ValueError("No training videos left after the validation split")
    train_shards, shard_steps = _balanced_video_shards(
        videos, train_ids, world_size)
    if args.verify_ddp_gradients and (world_size < 2 or min(shard_steps) == 0):
        raise ValueError(
            "--verify-ddp-gradients requires at least two ranks with training videos")
    local_train_ids = train_shards[rank]
    local_val_ids = val_ids[rank::world_size]

    # Resume support: skip epochs whose checkpoint directory already exists and
    # reload the latest one so a preempted cluster job continues cleanly.
    start_epoch = 0
    resume_adapter = None
    if args.resume:
        incomplete = [e for e in range(1, args.epochs + 1)
                      if (output / f"epoch-{e}" / "adapter_config.json").is_file()
                      and not (output / f"epoch-{e}" / "trainer_state.pt").is_file()]
        if incomplete:
            raise ValueError(
                f"Incomplete epoch checkpoints without trainer_state.pt: {incomplete}")
        done = [e for e in range(1, args.epochs + 1)
                if (output / f"epoch-{e}" / "adapter_config.json").is_file()
                and (output / f"epoch-{e}" / "trainer_state.pt").is_file()]
        if done:
            start_epoch = max(done)
            resume_adapter = str(output / f"epoch-{start_epoch}")
            print(f"[resume] continuing after epoch {start_epoch} from {resume_adapter}",
                  flush=True)

    memory = MemoryModel(args.model_name, base_adapter=args.init_adapter,
                         adapter_path=resume_adapter,
                         adapter_trainable=resume_adapter is not None,
                         dtype=dtype, rank=args.lora_rank)
    memory.model.to(torch.device("cuda", local_rank))
    memory.model.train()
    if hasattr(memory.core_model.config, "use_cache"):
        memory.core_model.config.use_cache = False
    # Non-reentrant checkpointing is compatible with DDP when the same Qwen
    # layers participate in several frame/summary forwards inside one step.
    memory.core_model.gradient_checkpointing_enable(
        gradient_checkpointing_kwargs={"use_reentrant": False})
    _assert_expected_trainables(memory)
    step_model: torch.nn.Module = Stage2TrainStep(memory)
    if world_size > 1:
        from torch.nn.parallel import DistributedDataParallel
        step_model = DistributedDataParallel(
            step_model, device_ids=[local_rank], output_device=local_rank,
            broadcast_buffers=False, find_unused_parameters=True)
    optimizer = torch.optim.AdamW((p for p in step_model.parameters() if p.requires_grad),
                                  lr=args.learning_rate)

    # Each active rank updates once per synchronized 10-second round. Rank 0
    # owns a maximum-length shard, so its optimizer updates determine the
    # checkpoint's global_step even when shorter ranks have joined.
    updates_per_epoch = max(shard_steps)
    total_updates = max(1, updates_per_epoch * args.epochs)
    from transformers import get_linear_schedule_with_warmup
    scheduler = get_linear_schedule_with_warmup(
        optimizer, int(total_updates * args.warmup_ratio), total_updates)

    global_step = 0
    best_metric = None
    best_epoch = None
    if resume_adapter is not None:
        state = torch.load(Path(resume_adapter) / "trainer_state.pt",
                           map_location="cpu", weights_only=False)
        if state["completed_epoch"] != start_epoch:
            raise ValueError("Checkpoint epoch and trainer_state.pt disagree")
        optimizer.load_state_dict(state["optimizer"])
        if state.get("scheduler") is None:
            raise ValueError("Resume checkpoint does not contain scheduler state")
        scheduler.load_state_dict(state["scheduler"])
        global_step = int(state["global_step"])
        expected_step = updates_per_epoch * start_epoch
        if global_step != expected_step:
            raise ValueError(
                f"Checkpoint global_step={global_step} but expected {expected_step} "
                f"after {start_epoch} completed epochs")
        best_metric = state.get("best_metric")
        best_epoch = state.get("best_epoch")
        rng_states = state.get("rng_states")
        if not isinstance(rng_states, list) or len(rng_states) != world_size:
            raise ValueError("Checkpoint RNG state does not match the DDP world size")
        _restore_rng_state(rng_states[rank])

    if rank == 0:
        config_path.write_text(
            json.dumps(current_config, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"[setup] rank={rank}/{world_size} local_device=cuda:{local_rank} "
          f"local_train_videos={len(local_train_ids)} local_steps={shard_steps[rank]} "
          f"train_videos={len(train_ids)} validation_videos={len(val_ids)} "
          f"updates_per_epoch={updates_per_epoch} total_updates={total_updates} "
          f"warmup_steps={int(total_updates * args.warmup_ratio)} "
          f"start_epoch={start_epoch} global_step={global_step}", flush=True)
    metrics_mode = "a" if start_epoch else "w"
    eval_log = ((output / "validation_metrics.jsonl").open(
        "a" if start_epoch else "w", encoding="utf-8") if rank == 0 else None)
    metrics_name = ("training_metrics.jsonl" if world_size == 1
                    else f"training_metrics.rank{rank}.jsonl")
    with (output / metrics_name).open(metrics_mode, encoding="utf-8") as metrics:
        for epoch in range(start_epoch, args.epochs):
            epoch_number = epoch + 1
            join_context = (step_model.join(divide_by_initial_world_size=True)
                            if world_size > 1 else nullcontext())
            with join_context:
                for video_id in local_train_ids:
                    reader = VideoReader(paths[video_id], args.frame_size)
                    try:
                        global_step = train_video(
                            memory, videos[video_id], reader, optimizer, metrics,
                            video_id, epoch_number, max_grad_norm=args.max_grad_norm,
                            scheduler=scheduler, global_step=global_step,
                            step_model=step_model,
                            verify_ddp_gradients=args.verify_ddp_gradients)
                    finally:
                        reader.close()
            if world_size > 1:
                # Rank 0 owns a maximum-length shard and therefore the complete
                # optimizer/scheduler trajectory after uneven-input joining.
                _sync_optimizer_state(optimizer, scheduler, rank, local_rank)
                global_step = updates_per_epoch * epoch_number
            # Teacher-forced validation on held-out videos after each epoch.
            summary = None
            if val_ids:
                step_model.eval()
                agg = {"short": 0.0, "block": 0.0, "global": 0.0}
                agg_n = {"short": 0, "block": 0, "global": 0}
                for video_id in local_val_ids:
                    reader = VideoReader(paths[video_id], args.frame_size)
                    try:
                        means, counts = evaluate_video(memory, videos[video_id], reader)
                    finally:
                        reader.close()
                    for k in agg:
                        if means[k] is not None:
                            agg[k] += means[k] * counts[k]
                            agg_n[k] += counts[k]
                if world_size > 1:
                    import torch.distributed as dist
                    packed = torch.tensor(
                        [agg["short"], agg["block"], agg["global"],
                         agg_n["short"], agg_n["block"], agg_n["global"]],
                        dtype=torch.float64, device=torch.device("cuda", local_rank))
                    dist.all_reduce(packed, op=dist.ReduceOp.SUM)
                    values = packed.cpu().tolist()
                    agg = dict(zip(("short", "block", "global"), values[:3]))
                    agg_n = {key: int(value) for key, value in zip(
                        ("short", "block", "global"), values[3:])}
                summary = {"epoch": epoch_number, "global_step": global_step, **{
                    k: (agg[k] / agg_n[k] if agg_n[k] else None) for k in agg}}
                components = [summary[k] for k in ("short", "block", "global")
                              if summary[k] is not None]
                summary["total"] = sum(components) if components else None
                if rank == 0:
                    eval_log.write(json.dumps(summary) + "\n")
                    eval_log.flush()
                    print(f"[eval] epoch {epoch_number}: {summary}", flush=True)
                step_model.train()
                if (rank == 0 and summary["total"] is not None
                        and (best_metric is None or summary["total"] < best_metric)):
                    best_metric = summary["total"]
                    best_epoch = epoch_number
                    memory.save(str(output / "best"))
                    (output / "best" / "best_metric.json").write_text(
                        json.dumps({"epoch": best_epoch, "metric": best_metric}, indent=2),
                        encoding="utf-8")

            if world_size > 1:
                import torch.distributed as dist
                gathered_rng = [None] * world_size if rank == 0 else None
                dist.gather_object(_rng_state(), gathered_rng, dst=0)
            else:
                gathered_rng = [_rng_state()]
            if rank == 0:
                checkpoint = output / f"epoch-{epoch_number}"
                memory.save(str(checkpoint))
                _save_training_state(
                    checkpoint, completed_epoch=epoch_number, global_step=global_step,
                    optimizer=optimizer, scheduler=scheduler,
                    best_metric=best_metric, best_epoch=best_epoch,
                    rng_states=gathered_rng)
            if world_size > 1:
                dist.barrier()
    if eval_log is not None:
        eval_log.close()
    if rank == 0:
        memory.save(str(output / "final"))
    if world_size > 1:
        import torch.distributed as dist
        dist.barrier()
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
