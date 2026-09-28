#!/usr/bin/env python3
"""Generate text from Stage-2 summary-compression checkpoints.

Supports both the legacy ASR reconstruction baseline and the main
teacher-visual-summary objective.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

os.environ.setdefault("TRANSFORMERS_NO_TF", "1")
os.environ.setdefault("USE_TF", "0")
os.environ.setdefault("USE_FLAX", "0")

import torch
from peft import PeftModel
from tqdm.auto import tqdm
from transformers import AutoProcessor

_HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE))

from memory_collator import (  # noqa: E402
    LONG_MEMORY_TOKEN,
    SHORT_MEMORY_TOKEN,
    MemoryCompressionCollator,
    long_decode_messages,
    long_encode_messages,
    short_decode_messages,
    short_encode_messages,
)
from memory_dataset import MemoryCompressionDataset  # noqa: E402
from train_memory_compression import load_model, replace_token_embeddings, token_positions  # noqa: E402


def move_to_device(batch, device):
    return {k: (v.to(device) if torch.is_tensor(v) else v) for k, v in batch.items()}


def encode_short(model, collator, frames, device):
    tok = collator.processor.tokenizer.convert_tokens_to_ids(SHORT_MEMORY_TOKEN)
    encoded = move_to_device(collator.encode_messages(short_encode_messages(frames)), device)
    ids = encoded["input_ids"][0]
    pos = token_positions(ids, tok)
    if not pos:
        raise RuntimeError("No <SHORT_MEM> in prompt")
    with torch.no_grad():
        out = model(**encoded, output_hidden_states=True)
    return out.hidden_states[-1][:, pos[-1], :]


def encode_short_states(model, collator, frames_list, device):
    return [encode_short(model, collator, frames, device) for frames in frames_list]


def encode_long_states(model, collator, short_states, previous_long_states, long_count, device):
    short_id = collator.processor.tokenizer.convert_tokens_to_ids(SHORT_MEMORY_TOKEN)
    long_id = collator.processor.tokenizer.convert_tokens_to_ids(LONG_MEMORY_TOKEN)
    encoded = move_to_device(collator.encode_messages(long_encode_messages(
        short_count=len(short_states),
        previous_long_count=len(previous_long_states),
        new_long_count=long_count,
    )), device)
    ids = encoded["input_ids"][0]
    short_pos = token_positions(ids, short_id)
    long_pos = token_positions(ids, long_id)
    prev_pos = long_pos[:len(previous_long_states)]
    new_pos = long_pos[-long_count:]
    positions = prev_pos + short_pos[:len(short_states)]
    vectors = previous_long_states + short_states
    if positions:
        encoded = replace_token_embeddings(model, encoded, positions, vectors)
    with torch.no_grad():
        out = model(**encoded, output_hidden_states=True)
    return [out.hidden_states[-1][:, p, :] for p in new_pos]


def generate_from_memory(model, collator, sample, memory_states, memory_token, device, max_new_tokens):
    if memory_token == SHORT_MEMORY_TOKEN:
        task = "local_summary" if sample.get("local_summary_target") else "asr"
        messages = short_decode_messages(short_count=len(memory_states), task=task)
    else:
        task = "global_summary" if sample.get("global_summary_target") else "accumulated"
        messages = long_decode_messages(long_count=len(memory_states), task=task)
    tok = collator.processor.tokenizer.convert_tokens_to_ids(memory_token)
    encoded = move_to_device(collator.encode_messages(messages), device)
    ids = encoded["input_ids"][0]
    pos = token_positions(ids, tok)
    encoded = replace_token_embeddings(model, encoded, pos[-len(memory_states):], memory_states)
    with torch.no_grad():
        gen = model.generate(**encoded, max_new_tokens=max_new_tokens, do_sample=False)
    new = gen[:, encoded["inputs_embeds"].shape[1]:]
    return collator.processor.batch_decode(new, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0].strip()


def parse_args():
    p = argparse.ArgumentParser(description="Infer Stage-2 summary-compression checkpoint")
    p.add_argument("--model-name", default="Qwen/Qwen3-VL-2B-Instruct")
    p.add_argument("--adapter-path", required=True)
    p.add_argument("--eval-jsonl", required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--repo-root", default=".")
    p.add_argument("--video-root", default=None)
    p.add_argument("--default-video-path", default=None)
    p.add_argument("--video-path-map", default=None)
    p.add_argument("--short-frames", type=int, default=2)
    p.add_argument("--frame-size", type=int, default=224)
    p.add_argument("--max-short-windows", type=int, default=60)
    p.add_argument("--max-previous-blocks", type=int, default=1)
    p.add_argument("--limit", type=int, default=None)
    p.add_argument("--max-new-tokens", type=int, default=160)
    p.add_argument("--bf16", action="store_true")
    p.add_argument("--fp16", action="store_true")
    return p.parse_args()


def main():
    args = parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.bfloat16 if args.bf16 else torch.float16 if args.fp16 else torch.float32
    processor = AutoProcessor.from_pretrained(args.adapter_path, trust_remote_code=True)
    model = load_model(args.model_name, dtype)
    if model.get_input_embeddings().num_embeddings != len(processor.tokenizer):
        model.resize_token_embeddings(len(processor.tokenizer))
    model = PeftModel.from_pretrained(model, args.adapter_path).to(device).eval()
    dataset = MemoryCompressionDataset(
        args.eval_jsonl,
        repo_root=args.repo_root,
        video_root=args.video_root,
        default_video_path=args.default_video_path,
        video_path_map=args.video_path_map,
        short_frames=args.short_frames,
        frame_size=args.frame_size,
        max_short_windows=args.max_short_windows,
        max_previous_blocks=args.max_previous_blocks,
        limit=args.limit,
    )
    collator = MemoryCompressionCollator(processor=processor)
    out = Path(args.output); out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", encoding="utf-8") as f:
        for idx in tqdm(range(len(dataset)), desc="memory infer"):
            sample = dataset[idx]
            if sample["sample_type"] in {"short_memory_compression", "short_memory_summary"}:
                mem = encode_short_states(model, collator, sample["short_frames_list"], device)
                pred = generate_from_memory(model, collator, sample, mem, SHORT_MEMORY_TOKEN, device, args.max_new_tokens)
            else:
                prev_long = []
                for block in sample.get("previous_blocks", []):
                    prev_short = encode_short_states(model, collator, block.get("short_frames_list", []), device)
                    if prev_short:
                        prev_long = encode_long_states(model, collator, prev_short, prev_long, int(sample.get("num_long_tokens") or 60), device)
                short = encode_short_states(model, collator, sample["short_frames_list"], device)
                mem = encode_long_states(model, collator, short, prev_long, int(sample.get("num_long_tokens") or 60), device)
                pred = generate_from_memory(model, collator, sample, mem, LONG_MEMORY_TOKEN, device, args.max_new_tokens)
            f.write(json.dumps({
                "idx": idx,
                "sample_type": sample.get("sample_type"),
                "video_id": sample.get("video_id"),
                "target": sample.get("global_summary_target") or sample.get("local_summary_target") or sample.get("target"),
                "local_summary_target": sample.get("local_summary_target"),
                "global_summary_target": sample.get("global_summary_target"),
                "prediction": pred,
                "meta": sample.get("meta", {}),
            }, ensure_ascii=False) + "\n")
    print(f"[memory-infer] wrote {out}")


if __name__ == "__main__":
    main()