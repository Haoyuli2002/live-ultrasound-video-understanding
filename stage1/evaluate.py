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
from .eval_io import digest, indexed, key, resume_rows


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--eval-jsonl", required=True)
    parser.add_argument("--video-path-map", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--model-name", default="Qwen/Qwen3.5-2B")
    parser.add_argument("--adapter-path")
    parser.add_argument("--visual-control", choices=["normal", "blank", "shuffled"], default="normal")
    parser.add_argument("--frame-budget", type=int, default=120)
    parser.add_argument("--frame-size", type=int, default=224)
    parser.add_argument("--frame-sampling", choices=("uniform", "recent_sparse"),
                        default="uniform")
    parser.add_argument("--recent-seconds", type=float, default=120.0)
    parser.add_argument("--max-asr-chars", type=int, default=4000)
    parser.add_argument("--max-new-tokens", type=int, default=128)
    args = parser.parse_args()

    from transformers import AutoModelForImageTextToText, AutoProcessor, set_seed
    from peft import PeftModel
    rows = [json.loads(line) for line in Path(args.eval_jsonl).read_text(encoding="utf-8").splitlines()
            if line.strip()]
    validate_rows(rows)
    paths = json.loads(Path(args.video_path_map).read_text(encoding="utf-8"))
    config = {**vars(args), "eval_sha256": digest(args.eval_jsonl),
              "map_sha256": digest(args.video_path_map), "seed": 42,
              "implementation_sha256": digest(__file__),
              "template_sha256": digest(Path(__file__).with_name('model.py')),
              "sampling_sha256": digest(Path(__file__).with_name('data.py'))}
    if args.adapter_path:
        config['adapter_sha256'] = {p.name: digest(p) for p in sorted(Path(args.adapter_path).glob('*')) if p.is_file()}
    expected = indexed(rows)
    done = indexed(resume_rows(args.output, config))
    expected_keys = {(r['video_id'], r['sentence_id'], r['condition'], args.visual_control) for r in rows}
    if not done.keys() <= expected_keys:
        raise ValueError('Unexpected sample in existing output')
    if len(done) == len(expected):
        print('[eval] already complete')
        return
    set_seed(42)
    missing = {r['video_id'] for r in rows} - paths.keys()
    if missing:
        raise ValueError(f'Video map missing ids: {sorted(missing)}')
    if any(not Path(paths[r['video_id']]).is_file() for r in rows):
        raise ValueError('Evaluation video file missing')
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
    with output.open("a", encoding="utf-8") as stream, torch.no_grad():
        for row in rows:
            if key({**row, 'visual_control': args.visual_control}) in done:
                continue
            source_id = next_video[row["video_id"]] if args.visual_control == "shuffled" else row["video_id"]
            frames = sample_frames(paths[source_id], row["video_window"],
                                   args.frame_budget, args.frame_size,
                                   sampling=args.frame_sampling,
                                   recent_seconds=args.recent_seconds)
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
                "video_window": row["video_window"],
                "generated_tokens": int(result.shape[1] - inputs["input_ids"].shape[1]),
                "hit_token_limit": bool(result.shape[1] - inputs["input_ids"].shape[1] >= args.max_new_tokens),
            }, ensure_ascii=False) + "\n")
            stream.flush()
            print(f"[eval] {row['video_id']} / {row['sentence_id']} / {row['condition']}", flush=True)


if __name__ == "__main__":
    main()
