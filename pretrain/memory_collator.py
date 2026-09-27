"""Prompt builders / utilities for Stage-2 memory compression.

The actual hidden-state extraction and injection happens in
`train_memory_compression.py`; this collator keeps batch_size=1 and returns PIL
frames plus JSON metadata.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List

SHORT_MEMORY_TOKEN = "<SHORT_MEM>"
LONG_MEMORY_TOKEN = "<LONG_MEM>"

SHORT_SYSTEM_PROMPT = """You are learning a short-term hidden memory for ultrasound video. The <SHORT_MEM> token must compress the newly arrived frames so that the narration can be reconstructed from memory only."""
LONG_SYSTEM_PROMPT = """You are learning a long-term hidden memory for ultrasound video. The <LONG_MEM> token must compress recent short-term memories and previous long-term memory so that the block narration can be reconstructed from memory only."""
DECODE_SYSTEM_PROMPT = """You are an ultrasound teaching assistant. Reconstruct the narration using only the provided hidden memory token(s). Output only the narration."""


def content_with_images(frames, text: str):
    content = [{"type": "image", "image": img} for img in frames]
    content.append({"type": "text", "text": text})
    return content


def short_encode_messages(frames):
    return [
        {"role": "system", "content": SHORT_SYSTEM_PROMPT},
        {"role": "user", "content": content_with_images(frames, f"Compress these frames into short-term memory: {SHORT_MEMORY_TOKEN}")},
    ]


def repeated_token(token: str, count: int) -> str:
    if count <= 0:
        return ""
    return " ".join([token] * count)


def short_decode_messages(short_count: int = 1, target: str | None = None):
    memory = repeated_token(SHORT_MEMORY_TOKEN, short_count)
    messages = [
        {"role": "system", "content": DECODE_SYSTEM_PROMPT},
        {"role": "user", "content": [{"type": "text", "text": f"Short-term memories: {memory}\nReconstruct the ASR narration for this time span."}]},
    ]
    if target is not None:
        messages.append({"role": "assistant", "content": target})
    return messages


def long_encode_messages(short_count: int, previous_long_count: int = 0, new_long_count: int = 60):
    prev = " ".join([LONG_MEMORY_TOKEN] * previous_long_count) if previous_long_count else "No previous long-term memory."
    short = " ".join([SHORT_MEMORY_TOKEN] * short_count) if short_count else "No short-term memories."
    new_long = repeated_token(LONG_MEMORY_TOKEN, new_long_count)
    text = f"Previous long-term memory: {prev}\nRecent short-term memories: {short}\nNew long-term memory: {new_long}"
    return [
        {"role": "system", "content": LONG_SYSTEM_PROMPT},
        {"role": "user", "content": [{"type": "text", "text": text}]},
    ]


def long_decode_messages(long_count: int = 60, task: str = "accumulated", target: str | None = None):
    memory = repeated_token(LONG_MEMORY_TOKEN, long_count)
    task_text = {
        "current": "Reconstruct the narration for the past/current 60-second block only.",
        "previous": "Reconstruct the narration from the beginning of the video up to the start of the current block.",
        "accumulated": "Reconstruct the narration from the beginning of the video up to the end of the current block.",
    }.get(task, task)
    messages = [
        {"role": "system", "content": DECODE_SYSTEM_PROMPT},
        {"role": "user", "content": [{"type": "text", "text": f"Long-term memories: {memory}\n{task_text}"}]},
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
class MemoryCompressionCollator:
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

    def target_start(self, input_ids, target: str) -> int:
        tokenizer = getattr(self.processor, "tokenizer", self.processor)
        target_ids = tokenizer(target, add_special_tokens=False).input_ids
        start = find_last_subsequence(input_ids.tolist(), target_ids)
        if start < 0:
            raise RuntimeError(f"Could not find target in encoded sequence: {target!r}")
        return start

    def __call__(self, features: List[Dict[str, Any]]) -> Dict[str, Any]:
        if len(features) != 1:
            raise ValueError("MemoryCompressionCollator currently supports batch_size=1 only")
        return features[0]