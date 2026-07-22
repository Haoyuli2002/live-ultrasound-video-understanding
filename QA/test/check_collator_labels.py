#!/usr/bin/env python3
"""
Sanity check for QA/train/collator.py label masking.

This script verifies that the supervised labels (labels != -100) produced by
QwenVLCollator decode to the assistant target and start with <WAIT> or <ANSWER>.

It loads only the processor/tokenizer, not the model weights.
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path


# Avoid importing TensorFlow/Flax through transformers on AzureML images.
os.environ.setdefault("TRANSFORMERS_NO_TF", "1")
os.environ.setdefault("USE_TF", "0")
os.environ.setdefault("USE_FLAX", "0")


REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


from transformers import AutoProcessor  # noqa: E402

from QA.train.collator import QwenVLCollator  # noqa: E402
from QA.train.dataset import QATrainingDataset  # noqa: E402


SPECIAL_TOKENS = ["<WAIT>", "<ANSWER>"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Check that collator labels supervise <WAIT>/<ANSWER> targets."
    )
    parser.add_argument(
        "--model-name",
        default="Qwen/Qwen3-VL-2B-Instruct",
        help="Processor/tokenizer name or local path.",
    )
    parser.add_argument(
        "--train-jsonl",
        default="QA/results/8V649L5Q368_training_samples.jsonl",
        help="Training JSONL file to sample from.",
    )
    parser.add_argument(
        "--default-video-path",
        default=(
            "UltrasoundCrawler_KeyCode_20260323_v2/output/"
            "20260520_162816_youtube/media/case_reasoning/8V649L5Q368.mp4"
        ),
        help="Fallback video path used by QATrainingDataset.",
    )
    parser.add_argument(
        "--sample-index",
        type=int,
        default=0,
        help="Dataset row index to check.",
    )
    parser.add_argument(
        "--window-size",
        type=int,
        default=8,
        help="Number of frames sampled by the dataset.",
    )
    parser.add_argument(
        "--frame-size",
        type=int,
        default=336,
        help="Resize sampled frames to this square size.",
    )
    parser.add_argument(
        "--local-files-only",
        action="store_true",
        help="Use only local HuggingFace cache for the processor/tokenizer.",
    )
    parser.add_argument(
        "--print-token-ids",
        action="store_true",
        help="Print supervised target token ids.",
    )
    return parser.parse_args()


def add_special_tokens_if_needed(processor) -> object:
    tokenizer = getattr(processor, "tokenizer", processor)

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    existing_vocab = tokenizer.get_vocab()
    tokens_to_add = [tok for tok in SPECIAL_TOKENS if tok not in existing_vocab]
    if tokens_to_add:
        tokenizer.add_special_tokens({"additional_special_tokens": tokens_to_add})
        print(f"[check] Added special tokens: {tokens_to_add}")
    else:
        print("[check] Special tokens already present")

    print(f"[check] <WAIT> id={tokenizer.convert_tokens_to_ids('<WAIT>')}")
    print(f"[check] <ANSWER> id={tokenizer.convert_tokens_to_ids('<ANSWER>')}")
    print(f"[check] vocab={len(tokenizer)}")
    return tokenizer


def main() -> None:
    args = parse_args()

    if args.sample_index < 0:
        raise ValueError("--sample-index must be >= 0")

    print("[check] Loading processor/tokenizer only, not model weights...")
    processor = AutoProcessor.from_pretrained(
        args.model_name,
        trust_remote_code=True,
        local_files_only=args.local_files_only,
    )
    tokenizer = add_special_tokens_if_needed(processor)

    print("[check] Loading dataset sample...")
    dataset = QATrainingDataset(
        args.train_jsonl,
        repo_root=REPO_ROOT,
        default_video_path=args.default_video_path,
        window_size=args.window_size,
        frame_size=args.frame_size,
        limit=args.sample_index + 1,
    )
    sample = dataset[args.sample_index]

    print(f"[check] sample_index={args.sample_index}")
    print(f"[check] sample_type={sample.get('sample_type')}")
    print(f"[check] qa_type={sample.get('qa_type')}")
    print(f"[check] video_window={sample.get('video_window')}")
    print(f"[check] expected_target={sample.get('target')!r}")
    print(f"[check] sampled_frames={len(sample.get('frames', []))}")

    collator = QwenVLCollator(processor=processor)
    batch = collator([sample])

    labels = batch["labels"][0]
    target_ids = labels[labels != -100].tolist()
    decoded = tokenizer.decode(target_ids, skip_special_tokens=False)
    clean = decoded.strip()

    print(f"[check] input_ids_shape={tuple(batch['input_ids'].shape)}")
    print(f"[check] supervised_label_tokens={len(target_ids)}")
    if args.print_token_ids:
        print(f"[check] supervised_token_ids={target_ids}")

    display = decoded
    if len(display) > 1200:
        display = display[:1200] + "\n... [truncated] ..."

    print("\n========== DECODED SUPERVISED TARGET ==========")
    print(display)
    print("===============================================\n")

    if not target_ids:
        raise RuntimeError(
            "BUG: labels != -100 is empty. The assistant target is fully masked."
        )

    if not (clean.startswith("<WAIT>") or clean.startswith("<ANSWER>")):
        raise RuntimeError(
            "BUG: supervised labels do not start with <WAIT> or <ANSWER>. "
            "The collator mask is likely misaligned."
        )

    forbidden_fragments = [
        "<|image_pad|>",
        "<|vision_start|>",
        "<|vision_end|>",
        "Question:",
        "<|im_start|>assistant",
    ]
    leaked = [fragment for fragment in forbidden_fragments if fragment in decoded]
    if leaked:
        raise RuntimeError(
            "BUG: supervised labels contain prompt/vision fragments: "
            f"{leaked}. The collator mask is still misaligned."
        )

    expected = str(sample.get("target", "")).strip()
    if expected and not clean.startswith(expected[: min(len(expected), 32)]):
        print(
            "[check] WARNING: decoded labels start with <WAIT>/<ANSWER>, but do not "
            "exactly match the beginning of sample['target']. Inspect the decoded "
            "text above for chat-template boundary tokens or whitespace."
        )

    print("[check] PASS: supervised labels start with <WAIT>/<ANSWER> and contain no prompt leakage.")


if __name__ == "__main__":
    main()