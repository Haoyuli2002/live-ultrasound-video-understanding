"""Paired-condition generation with optional blank/shuffled visual controls."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from PIL import Image

from .data import sample_frames
from .model import encode
from .train import validate_rows


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--eval-jsonl", required=True)
    parser.add_argument("--video-path-map", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--model-name", default="Qwen/Qwen3-VL-2B-Instruct")
    parser.add_argument("--adapter-path")
    parser.add_argument("--visual-control", choices=["normal", "blank", "shuffled"], default="normal")
    parser.add_argument("--frame-budget", type=int, default=120)
    parser.add_argument("--frame-size", type=int, default=224)
    parser.add_argument("--max-asr-chars", type=int, default=4000)
    parser.add_argument("--max-new-tokens", type=int, default=128)
    args = parser.parse_args()

    from transformers import AutoModelForImageTextToText, AutoProcessor
    from peft import PeftModel
    rows = [json.loads(line) for line in Path(args.eval_jsonl).read_text(encoding="utf-8").splitlines()
            if line.strip()]
    validate_rows(rows)
    paths = json.loads(Path(args.video_path_map).read_text(encoding="utf-8"))
    processor = AutoProcessor.from_pretrained(args.adapter_path or args.model_name,
                                              trust_remote_code=True)
    base = AutoModelForImageTextToText.from_pretrained(args.model_name,
                                                        torch_dtype=torch.bfloat16,
                                                        trust_remote_code=True)
    model = PeftModel.from_pretrained(base, args.adapter_path) if args.adapter_path else base
    if not torch.cuda.is_available():
        raise RuntimeError("Stage 1 evaluation requires CUDA")
    model.to("cuda").eval()
    video_ids = sorted({row["video_id"] for row in rows})
    if args.visual_control == "shuffled" and len(video_ids) < 2:
        raise ValueError("Shuffled control needs at least two videos")
    next_video = {video_ids[i]: video_ids[(i+1) % len(video_ids)]
                  for i in range(len(video_ids))}
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8") as stream, torch.no_grad():
        for row in rows:
            source_id = next_video[row["video_id"]] if args.visual_control == "shuffled" else row["video_id"]
            frames = sample_frames(paths[source_id], row["video_window"],
                                   args.frame_budget, args.frame_size)
            if args.visual_control == "blank":
                frames = [Image.new("RGB", frame.size) for frame in frames]
            inputs = encode(processor, row, frames, max_asr_chars=args.max_asr_chars,
                            include_target=False)
            inputs = {k: v.to("cuda") if torch.is_tensor(v) else v for k, v in inputs.items()}
            result = model.generate(**inputs, max_new_tokens=args.max_new_tokens,
                                    do_sample=False)
            prediction = processor.batch_decode(
                result[:, inputs["input_ids"].shape[1]:], skip_special_tokens=True)[0].strip()
            stream.write(json.dumps({
                "video_id": row["video_id"], "sentence_id": row["sentence_id"],
                "condition": row["condition"], "visual_control": args.visual_control,
                "target": row["target"], "prediction": prediction,
            }, ensure_ascii=False) + "\n")


if __name__ == "__main__":
    main()
