"""Collation/building blocks for recurrent <SUMMARY> streaming QA SFT.

The trainer performs multiple forwards per sample, so this collator intentionally
keeps batch_size=1 and returns Python objects plus sampled PIL frames.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List

import torch

SUMMARY_TOKEN = "<SUMMARY>"

SUMMARY_SYSTEM_PROMPT = """You maintain a hidden ultrasound video memory. Update <SUMMARY> using the new ultrasound frames and optional narration."""
QA_SYSTEM_PROMPT = """You are a real-time ultrasound assistant. Use the maintained <SUMMARY> memory, current ultrasound frames, and the question. If evidence is insufficient, start with <WAIT> and explain what is missing. If evidence is sufficient, start with <ANSWER> and answer concisely."""


def content_with_images(frames, text: str):
    content = [{"type": "image", "image": img} for img in frames]
    if text:
        content.append({"type": "text", "text": text})
    return content


def summary_update_messages(frames, text: str = "", previous_summary_count: int = 0):
    user_text = "Update the hidden summary for this video chunk."
    if previous_summary_count > 0:
        previous = " ".join([SUMMARY_TOKEN] * previous_summary_count)
        user_text += f"\nPrevious summary bank: {previous}"
    if text:
        user_text += f"\nNarration: {text}"
    user_text += f"\nNew chunk summary: {SUMMARY_TOKEN}"
    return [
        {"role": "system", "content": SUMMARY_SYSTEM_PROMPT},
        {"role": "user", "content": content_with_images(frames, user_text)},
    ]


def qa_messages(current_frames, question: str, summary_count: int, target: str | None = None):
    memory = " ".join([SUMMARY_TOKEN] * summary_count) if summary_count > 0 else "No prior memory."
    user_text = (
        f"Summary bank: {memory}\n"
        f"Question: {question}\n"
        "Respond by starting with <WAIT> if more video is needed, "
        "or <ANSWER> if the current evidence is sufficient."
    )
    messages = [
        {"role": "system", "content": QA_SYSTEM_PROMPT},
        {"role": "user", "content": content_with_images(current_frames, user_text)},
    ]
    if target is not None:
        messages.append({"role": "assistant", "content": target})
    return messages


def process_vision(messages):
    try:
        from qwen_vl_utils import process_vision_info
        return process_vision_info(messages)
    except Exception:
        image_inputs = []
        for msg in messages:
            content = msg.get("content", [])
            if isinstance(content, list):
                for item in content:
                    if isinstance(item, dict) and item.get("type") == "image":
                        image_inputs.append(item["image"])
        return image_inputs, None


def find_last_subsequence(sequence: List[int], subsequence: List[int]) -> int:
    if not subsequence or len(subsequence) > len(sequence):
        return -1
    last = -1
    end = len(sequence) - len(subsequence)
    for i in range(end + 1):
        if sequence[i:i + len(subsequence)] == subsequence:
            last = i
    return last


@dataclass
class SummaryDecideCollator:
    processor: Any
    label_pad_token_id: int = -100

    def encode_messages(self, messages):
        text = self.processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=False)
        image_inputs, video_inputs = process_vision(messages)
        kwargs = {"text": [text], "padding": False, "return_tensors": "pt"}
        if image_inputs:
            kwargs["images"] = image_inputs
        if video_inputs:
            kwargs["videos"] = video_inputs
        return self.processor(**kwargs)

    def target_start(self, input_ids: torch.Tensor, target: str) -> int:
        tokenizer = getattr(self.processor, "tokenizer", self.processor)
        target_ids = tokenizer(target, add_special_tokens=False).input_ids
        start = find_last_subsequence(input_ids.tolist(), target_ids)
        if start < 0:
            raise RuntimeError(f"Could not find target in encoded sequence: {target!r}")
        return start

    def token_pos(self, input_ids: torch.Tensor, token: str) -> int:
        tokenizer = getattr(self.processor, "tokenizer", self.processor)
        ids = tokenizer(token, add_special_tokens=False).input_ids
        pos = find_last_subsequence(input_ids.tolist(), ids)
        if pos < 0:
            raise RuntimeError(f"Could not find {token} in encoded sequence")
        return pos

    def __call__(self, features: List[Dict[str, Any]]) -> Dict[str, Any]:
        if len(features) != 1:
            raise ValueError("SummaryDecideCollator currently supports batch_size=1 only")
        feat = features[0]
        return {
            "history_chunks": feat.get("history_chunks", []),
            "current_visual_frames": feat["current_visual_frames"],
            "question": feat["question"],
            "target": feat["target"],
            "meta": {k: feat.get(k) for k in ["video_id", "sample_type", "qa_type", "video_window"]},
        }