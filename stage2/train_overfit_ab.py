"""Controlled Short-only / independent-target overfit experiments.

Each target owns one fresh forward, backward and optimizer step. No visual or
long-memory tensor survives an optimizer update. Historical long memory is
replayed with current weights, without gradients through earlier minutes.
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import random

import torch

from .data import VideoReader, load_blocks
from .model import MemoryModel

MODES = ("short_only", "independent")


def encode_interval(memory, reader, start: int, end: int):
    from .short_encoding import short_window, short_memory_scope
    if end <= start:
        raise ValueError("Expected a nonempty chronological interval")
    if not getattr(memory, "requires_short_history", False):
        return torch.cat([short_window(memory,[reader.frame_ending_at(t)
                           for t in range(a,min(a+10,end))]) for a in range(start,end,10)],dim=1)
    # A Local ending at second 20 needs history from seconds 0..9 as well.
    # Replay at the current optimizer weights, but decode/supervise only the
    # requested 10 output tokens. No cache survives this target's backward pass.
    minute_start = start//60*60
    if start < 0 or end > minute_start+60:
        raise ValueError("History-aware intervals must lie within one minute")
    with short_memory_scope(memory):
        tokens = torch.cat([short_window(memory,[reader.frame_ending_at(t)
                            for t in range(a,min(a+10,end))])
                            for a in range(minute_start,end,10)],dim=1)
        return tokens[:, start-minute_start:end-minute_start, :]


def replay_previous_long(memory, blocks, reader):
    """Current-weight causal prefix, detached only at past-minute boundaries."""
    previous = None
    with torch.no_grad():
        for row in blocks:
            start, end = map(lambda x: int(round(x)), row["block_window"])
            if end-start != 60:
                raise ValueError("Historical long memory needs complete minutes")
            short = encode_interval(memory, reader, start, end)
            previous = memory.update_long(previous, short).detach()
    return previous


def train_video(memory, blocks, reader, optimizer, metrics_file, video_id,
                epoch, mode):
    if mode not in MODES:
        raise ValueError(f"Unknown supervision mode: {mode}")
    updates = 0

    def update(kind, target, start, end, block_index):
        nonlocal updates
        optimizer.zero_grad(set_to_none=True)
        previous = (replay_previous_long(memory, blocks[:block_index], reader)
                    if kind == "global" else None)
        # Fresh current-window graph: all 10/60 visual readouts receive gradients.
        current = encode_interval(memory, reader, start, end)
        tokens = memory.update_long(previous, current) if kind == "global" else current
        loss = memory.reconstruction_loss(tokens, target, kind)
        value = float(loss.detach())
        if not math.isfinite(value):
            raise FloatingPointError(f"Nonfinite {kind} loss at {start}:{end}")
        loss.backward()
        optimizer.step()
        updates += 1
        row = {"video_id": video_id, "epoch": epoch, "mode": mode,
               "update_in_video": updates, "kind": kind, "block": block_index,
               "window": [0 if kind == "global" else start, end],
               "loss": value, "fresh_memory": True,
               "history_replayed_blocks": block_index if kind == "global" else 0}
        metrics_file.write(json.dumps(row)+"\n")
        metrics_file.flush()
        print(f"[train] epoch={epoch+1} update={updates} {kind} "
              f"{row['window']} loss={value:.6f}", flush=True)

    for i, row in enumerate(blocks):
        start, end = [int(round(x)) for x in row["block_window"]]
        for j, local in enumerate(row["local_sub_summaries"]):
            first = start + 10*j
            update("short", local["local_summary_target"], first, first+10, i)
        if mode == "independent" and end-start == 60:
            update("block", row["block_summary_target"], start, end, i)
            update("global", row["global_summary_target"], start, end, i)
    return updates


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=MODES, required=True)
    parser.add_argument("--memory-architecture",choices=["ordinary","qformer"],default="ordinary")
    parser.add_argument("--init-adapter")
    parser.add_argument("--train-jsonl", required=True)
    parser.add_argument("--video-path-map", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--model-name", choices=["Qwen/Qwen3.5-2B"],
                        default="Qwen/Qwen3.5-2B")
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--frame-size", type=int, default=224)
    parser.add_argument("--seed", type=int, default=20261104)
    args = parser.parse_args()
    if (args.epochs < 1 or args.frame_size < 1 or not math.isfinite(args.learning_rate)
            or args.learning_rate <= 0):
        parser.error("Epochs, frame size and learning rate must be positive")
    if not torch.cuda.is_available():
        raise RuntimeError("Overfit training requires an allocated CUDA GPU")
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    videos = load_blocks(args.train_jsonl)
    paths = json.loads(Path(args.video_path_map).read_text())
    for vid in videos:
        if vid not in paths or not Path(paths[vid]).is_file():
            raise FileNotFoundError(f"Missing video: {vid}")
    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=False)
    config = {**vars(args), "initialization": "stage1_adapter" if args.init_adapter else "official_checkpoint",
              "history_gradients": "truncated; past minutes replayed at current weights",
              "current_window_encoder_gradients": "all frames", "gradient_accumulation": 1}
    (output/"training_config.json").write_text(json.dumps(config, indent=2))
    from .qformer import load_memory_model
    memory = load_memory_model(args.model_name, architecture=args.memory_architecture,
                               base_adapter=args.init_adapter,dtype=torch.bfloat16)
    if hasattr(memory, 'qformer_config'):
        config['qformer_config'] = memory.qformer_config
        config['short_encoding'] = 'causal minute-prefix replay; supervise only requested output interval'
        (output/'training_config.json').write_text(json.dumps(config, indent=2))
    if args.memory_architecture == "qformer":
        memory.to("cuda").train()
    else:
        memory.model.to("cuda").train()
    memory.model.config.use_cache = False
    memory.model.gradient_checkpointing_enable(
        gradient_checkpointing_kwargs={"use_reentrant": False})
    parameters = list(memory.model.parameters())
    if hasattr(memory,"compressors"):
        parameters += list(memory.compressors.parameters())
    optimizer = torch.optim.AdamW((p for p in parameters if p.requires_grad),lr=args.learning_rate)
    initial_compressors = ({k:v.detach().cpu().clone() for k,v in memory.compressors.state_dict().items()}
                           if hasattr(memory,'compressors') else None)
    with (output/"training_metrics.jsonl").open("x") as metrics:
        for epoch in range(args.epochs):
            for vid, blocks in videos.items():
                reader = VideoReader(paths[vid], args.frame_size)
                try:
                    train_video(memory, blocks, reader, optimizer, metrics, vid, epoch, args.mode)
                finally:
                    reader.close()
            if epoch == 0:
                memory.save(str(output/"epoch-1"))
    memory.save(str(output/"final"))
    if initial_compressors is not None:
        final=memory.compressors.state_dict()
        report={group:{'changed_parameters':sum(
                    not torch.equal(initial_compressors[k],v.detach().cpu())
                    for k,v in final.items() if k.startswith(group+'.')),
                    'all_finite':all(torch.isfinite(v).all().item() for k,v in final.items()
                                     if k.startswith(group+'.'))} for group in ['short','long']}
        (output/'compressor_training_report.json').write_text(json.dumps(report,indent=2))


if __name__ == "__main__":
    main()
