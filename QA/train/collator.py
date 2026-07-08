"""
Data collator for Qwen-VL WAIT/ANSWER SFT.

The collator converts one dataset row into a multimodal chat example:

System:
  real-time ultrasound assistant instruction

User:
  [8 sampled image frames]
  Question: ...

Assistant:
  <WAIT> ...   OR   <ANSWER> ...

Loss masking:
  - system/user/image prompt tokens => -100
  - assistant target tokens         => token ids
  - padding                         => -100

Notes:
  - We use image blocks rather than a video block for maximum compatibility
    across Qwen2-VL / Qwen2.5-VL / Qwen3-VL processors.
  - If the processor supports qwen_vl_utils.process_vision_info, we use it.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List

import torch


DEFAULT_SYSTEM_PROMPT = """You are a real-time ultrasound assistant.
You receive an ultrasound video window and a question.
Answer only if the current visual evidence is sufficient.
If the evidence is insufficient, output exactly:
<WAIT> Not enough information yet. More video is needed.
If the evidence is sufficient, output:
<ANSWER> followed by the answer."""


def _content_with_images(frames, question: str):
    content = []
    for img in frames:
        content.append({"type": "image", "image": img})
    content.append({"type": "text", "text": f"Question: {question}"})
    return content


def build_messages(frames, question: str, target: str | None = None, system_prompt: str = DEFAULT_SYSTEM_PROMPT):
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": _content_with_images(frames, question)},
    ]
    if target is not None:
        messages.append({"role": "assistant", "content": target})
    return messages


def _process_vision(messages):
    """
    Return image_inputs / video_inputs for Qwen processors.

    qwen_vl_utils.process_vision_info accepts a messages list and extracts
    PIL images / videos. If unavailable, fallback to manually collecting images.
    """
    try:
        from qwen_vl_utils import process_vision_info

        image_inputs, video_inputs = process_vision_info(messages)
        return image_inputs, video_inputs
    except Exception:
        image_inputs = []
        for msg in messages:
            content = msg.get("content", [])
            if isinstance(content, list):
                for item in content:
                    if isinstance(item, dict) and item.get("type") == "image":
                        image_inputs.append(item["image"])
        return image_inputs, None


@dataclass
class QwenVLCollator:
    processor: Any
    system_prompt: str = DEFAULT_SYSTEM_PROMPT
    label_pad_token_id: int = -100

    def _encode_messages(self, messages):
        text = self.processor.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=False,
        )
        image_inputs, video_inputs = _process_vision(messages)

        kwargs = dict(
            text=[text],
            padding=False,
            return_tensors="pt",
        )
        if image_inputs:
            kwargs["images"] = image_inputs
        if video_inputs:
            kwargs["videos"] = video_inputs

        return self.processor(**kwargs), text

    def _prompt_len(self, frames, question: str) -> int:
        prompt_messages = build_messages(
            frames=frames,
            question=question,
            target=None,
            system_prompt=self.system_prompt,
        )
        prompt_text = self.processor.apply_chat_template(
            prompt_messages,
            tokenize=False,
            add_generation_prompt=True,
        )

        # For masking, text-only length is usually sufficient because visual
        # placeholder tokens live in the prompt. We use tokenizer directly to
        # avoid duplicating image preprocessing just to compute the mask length.
        tokenizer = getattr(self.processor, "tokenizer", self.processor)
        prompt_ids = tokenizer(prompt_text, add_special_tokens=False).input_ids
        return len(prompt_ids)

    def __call__(self, features: List[Dict[str, Any]]) -> Dict[str, torch.Tensor]:
        encoded_list = []
        labels_list = []

        for feat in features:
            frames = feat["frames"]
            question = feat["question"]
            target = feat["target"]

            full_messages = build_messages(
                frames=frames,
                question=question,
                target=target,
                system_prompt=self.system_prompt,
            )
            encoded, _ = self._encode_messages(full_messages)

            input_ids = encoded["input_ids"][0]
            labels = input_ids.clone()

            prompt_len = self._prompt_len(frames, question)
            prompt_len = min(prompt_len, labels.shape[0])
            labels[:prompt_len] = self.label_pad_token_id

            # Mask padding if present.
            attention_mask = encoded.get("attention_mask")
            if attention_mask is not None:
                labels[attention_mask[0] == 0] = self.label_pad_token_id

            encoded_list.append(encoded)
            labels_list.append(labels)

        # Batch padding. Processor outputs can include tensors like pixel_values
        # whose first dimension already corresponds to image count; for v1 we
        # support batch_size=1 robustly, and keep a best-effort path for >1.
        if len(encoded_list) == 1:
            batch = {k: v for k, v in encoded_list[0].items()}
            batch["labels"] = labels_list[0].unsqueeze(0)
            return batch

        tokenizer = getattr(self.processor, "tokenizer", self.processor)
        pad_id = tokenizer.pad_token_id
        if pad_id is None:
            pad_id = tokenizer.eos_token_id

        max_len = max(e["input_ids"].shape[1] for e in encoded_list)
        input_ids_batch = []
        attention_batch = []
        labels_batch = []

        for encoded, labels in zip(encoded_list, labels_list):
            input_ids = encoded["input_ids"][0]
            attention = encoded.get("attention_mask", torch.ones_like(encoded["input_ids"]))[0]
            pad_len = max_len - input_ids.shape[0]

            input_ids_batch.append(torch.nn.functional.pad(input_ids, (0, pad_len), value=pad_id))
            attention_batch.append(torch.nn.functional.pad(attention, (0, pad_len), value=0))
            labels_batch.append(torch.nn.functional.pad(labels, (0, pad_len), value=self.label_pad_token_id))

        batch = {
            "input_ids": torch.stack(input_ids_batch, dim=0),
            "attention_mask": torch.stack(attention_batch, dim=0),
            "labels": torch.stack(labels_batch, dim=0),
        }

        # Multimodal tensors are processor/model-specific and hard to pad
        # generically. Use batch_size=1 for reliable multimodal SFT.
        for k, v in encoded_list[0].items():
            if k not in batch:
                batch[k] = v

        return batch