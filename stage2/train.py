"""Chronological Stage 2 training: six optimizer updates per full minute."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from .data import VideoReader, load_blocks
from .model import MemoryModel


def train_video(memory: MemoryModel, blocks: list[dict], reader: VideoReader,
                optimizer: torch.optim.Optimizer, metrics_file, video_id: str,
                epoch: int):
    previous_long = None  # Reset at each video/epoch boundary.
    model = memory.model
    for block_index, row in enumerate(blocks):
        start, end = row["block_window"]
        short_states: list[torch.Tensor] = []
        for local_index, label in enumerate(row["local_sub_summaries"]):
            interval_start = int(round(start)) + 10 * local_index
            current = [memory.encode_frame(reader.frame_ending_at(second))
                       for second in range(interval_start, interval_start + 10)]
            # Each state is [1, hidden]; concatenate into [1, 10, hidden].
            short_ten = torch.stack(current, dim=1)
            short_states.append(short_ten)
            short_loss = memory.reconstruction_loss(
                short_ten, label["local_summary_target"], "short")
            last_ten_of_full_block = len(row["local_sub_summaries"]) == 6 and local_index == 5
            if last_ten_of_full_block:
                all_short = torch.cat(short_states, dim=1)
                block_loss = memory.reconstruction_loss(
                    all_short, row["block_summary_target"], "block")
                new_long = memory.update_long(previous_long, all_short)
                global_loss = memory.reconstruction_loss(
                    new_long, row["global_summary_target"], "global")
                total = short_loss + block_loss + global_loss
            else:
                block_loss = global_loss = None
                new_long = None
                total = short_loss

            optimizer.zero_grad(set_to_none=True)
            total.backward()
            optimizer.step()
            metrics_file.write(json.dumps({
                "video_id": video_id, "epoch": epoch, "block": block_index,
                "local": local_index, "end_sec": interval_start + 10,
                "short": float(short_loss.detach()),
                "block_loss": None if block_loss is None else float(block_loss.detach()),
                "global": None if global_loss is None else float(global_loss.detach()),
                "total": float(total.detach()),
            }) + "\n")
            metrics_file.flush()
            short_states[-1] = short_states[-1].detach()
            if new_long is not None:
                previous_long = new_long.detach()
                short_states.clear()
        # The final <10-second tail has no label and no optimizer step.
        if end - start < 60:
            break


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train-jsonl", required=True)
    parser.add_argument("--video-path-map", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--model-name", default="Qwen/Qwen3-VL-2B-Instruct")
    parser.add_argument("--init-adapter")
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--frame-size", type=int, default=224)
    parser.add_argument("--lora-rank", type=int, default=16)
    parser.add_argument("--dtype", choices=["bf16", "fp16", "fp32"], default="bf16")
    args = parser.parse_args()

    if not torch.cuda.is_available():
        raise RuntimeError("Stage 2 Qwen3-VL training requires a CUDA device")
    dtype = {"bf16": torch.bfloat16, "fp16": torch.float16,
             "fp32": torch.float32}[args.dtype]
    videos = load_blocks(args.train_jsonl)
    paths = json.loads(Path(args.video_path_map).read_text(encoding="utf-8"))
    missing = sorted(set(videos) - set(paths))
    if missing:
        raise ValueError(f"Missing video paths: {missing[:5]}")
    memory = MemoryModel(args.model_name, base_adapter=args.init_adapter,
                         dtype=dtype, rank=args.lora_rank)
    memory.model.to("cuda")
    memory.model.train()
    if hasattr(memory.model.config, "use_cache"):
        memory.model.config.use_cache = False
    memory.model.gradient_checkpointing_enable()
    optimizer = torch.optim.AdamW((p for p in memory.model.parameters() if p.requires_grad),
                                  lr=args.learning_rate)
    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    (output / "training_config.json").write_text(
        json.dumps(vars(args), indent=2, ensure_ascii=False), encoding="utf-8")
    with (output / "training_metrics.jsonl").open("w", encoding="utf-8") as metrics:
        for epoch in range(args.epochs):
            for video_id, blocks in videos.items():
                reader = VideoReader(paths[video_id], args.frame_size)
                try:
                    train_video(memory, blocks, reader, optimizer, metrics, video_id, epoch)
                finally:
                    reader.close()
            memory.save(str(output / f"epoch-{epoch + 1}"))
    memory.save(str(output / "final"))


if __name__ == "__main__":
    main()
