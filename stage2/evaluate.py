"""Teacher-forced reconstruction losses on held-out chronological videos."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from .data import VideoReader, load_blocks
from .model import MemoryModel


def evaluate_video(memory: MemoryModel, blocks: list[dict], reader: VideoReader):
    long_state = None
    rows = []
    with torch.no_grad():
        for row in blocks:
            start, end = row["block_window"]
            short_states = []
            for j, label in enumerate(row["local_sub_summaries"]):
                first_second = int(round(start)) + 10*j
                states = [memory.encode_frame(reader.frame_ending_at(t))
                          for t in range(first_second, first_second+10)]
                short_states.extend(states)
                current = torch.stack(states, dim=1)
                loss = memory.reconstruction_loss(
                    current, label["local_summary_target"], "short")
                rows.append({"kind": "short", "end_sec": first_second+10,
                             "loss": float(loss)})
            if abs(end-start-60) <= 0.01:
                all_short = torch.stack(short_states, dim=1)
                loss = memory.reconstruction_loss(all_short,
                                                  row["block_summary_target"], "block")
                rows.append({"kind": "block", "end_sec": int(round(end)),
                             "loss": float(loss)})
                long_state = memory.update_long(long_state, all_short)
                loss = memory.reconstruction_loss(long_state,
                                                  row["global_summary_target"], "global")
                rows.append({"kind": "global", "end_sec": int(round(end)),
                             "loss": float(loss)})
    return rows


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--eval-jsonl", required=True)
    parser.add_argument("--video-path-map", required=True)
    parser.add_argument("--adapter-path", required=True)
    parser.add_argument("--base-adapter")
    parser.add_argument("--model-name", default="Qwen/Qwen3-VL-2B-Instruct")
    parser.add_argument("--frame-size", type=int, default=224)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("Stage 2 Qwen3-VL evaluation requires CUDA")
    config_path = Path(args.adapter_path).parent / "training_config.json"
    config = json.loads(config_path.read_text(encoding="utf-8")) if config_path.exists() else {}
    base_adapter = args.base_adapter or config.get("init_adapter")
    memory = MemoryModel(args.model_name, base_adapter=base_adapter,
                         adapter_path=args.adapter_path)
    memory.model.to("cuda").eval()
    videos = load_blocks(args.eval_jsonl)
    paths = json.loads(Path(args.video_path_map).read_text(encoding="utf-8"))
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8") as stream:
        for video_id, blocks in videos.items():
            reader = VideoReader(paths[video_id], args.frame_size)
            try:
                for row in evaluate_video(memory, blocks, reader):
                    stream.write(json.dumps({"video_id": video_id, **row}) + "\n")
            finally:
                reader.close()


if __name__ == "__main__":
    main()
