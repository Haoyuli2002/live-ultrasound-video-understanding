"""Run the causal 1-FPS short/long memory recurrence without teacher labels."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from .data import VideoReader
from .model import MemoryModel


def stream_video(memory: MemoryModel, reader: VideoReader, seconds: int):
    long_state = None
    short_states = []
    snapshots = []
    with torch.no_grad():
        for second in range(seconds):
            short_states.append(memory.encode_frame(reader.frame_ending_at(second)))
            if len(short_states) == 60:
                block_short = torch.stack(short_states, dim=1)
                long_state = memory.update_long(long_state, block_short).detach()
                snapshots.append({"end_sec": second + 1,
                                  "long": long_state.cpu(),
                                  "short": torch.empty(1, 0, long_state.shape[-1])})
                short_states.clear()
    return {"snapshots": snapshots,
            "final_long": None if long_state is None else long_state.cpu(),
            "final_short": None if not short_states else torch.stack(short_states, dim=1).cpu(),
            "seconds": seconds}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--video", required=True)
    parser.add_argument("--adapter-path", required=True)
    parser.add_argument("--base-adapter", help="Stage 1 adapter merged before Stage 2; defaults to training_config.json")
    parser.add_argument("--output", required=True)
    parser.add_argument("--model-name", default="Qwen/Qwen3-VL-2B-Instruct")
    parser.add_argument("--frame-size", type=int, default=224)
    parser.add_argument("--seconds", type=int)
    args = parser.parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("Stage 2 Qwen3-VL inference requires CUDA")
    config_path = Path(args.adapter_path).parent / "training_config.json"
    config = json.loads(config_path.read_text(encoding="utf-8")) if config_path.exists() else {}
    base_adapter = args.base_adapter or config.get("init_adapter")
    if config.get("init_adapter") and not base_adapter:
        raise ValueError("Stage 1 base adapter is required to reproduce this Stage 2 checkpoint")
    memory = MemoryModel(args.model_name, base_adapter=base_adapter,
                         adapter_path=args.adapter_path)
    memory.model.to("cuda").eval()
    reader = VideoReader(args.video, args.frame_size)
    try:
        seconds = args.seconds or int(reader.count / reader.fps)
        if seconds > int(reader.count / reader.fps):
            raise ValueError("Requested seconds exceed video duration")
        state = stream_video(memory, reader, seconds)
    finally:
        reader.close()
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(state, output)


if __name__ == "__main__":
    main()
