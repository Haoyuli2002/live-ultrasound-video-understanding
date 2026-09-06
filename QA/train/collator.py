"""Data collator for Qwen-VL WAIT/ANSWER SFT.

The collator converts dataset rows into multimodal chat examples where the
assistant target starts with either <WAIT> or <ANSWER>. Answerability is learned
as the first generated token via the normal causal LM loss.
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
    content = [{"type": "image", "image": img} for img in frames]
    content.append({"type": "text", "text": f"Question: {question}"})
    return content


def build_messages(
    frames,
    question: str,
    target: str | None = None,
    system_prompt: str = DEFAULT_SYSTEM_PROMPT,
):
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": _content_with_images(frames, question)},
    ]
    if target is not None:
        messages.append({"role": "assistant", "content": target})
    return messages


def _process_vision(messages):
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
        kwargs = dict(text=[text], padding=False, return_tensors="pt")
        if image_inputs:
            kwargs["images"] = image_inputs
        if video_inputs:
            kwargs["videos"] = video_inputs
        return self.processor(**kwargs), text

    def _find_last_subsequence(self, sequence: List[int], subsequence: List[int]) -> int:
        if not subsequence or len(subsequence) > len(sequence):
            return -1
        last = -1
        end = len(sequence) - len(subsequence)
        for i in range(end + 1):
            if sequence[i:i + len(subsequence)] == subsequence:
                last = i
        return last

    def _target_start(self, input_ids: torch.Tensor, target: str) -> int:
        tokenizer = getattr(self.processor, "tokenizer", self.processor)
        target_ids = tokenizer(target, add_special_tokens=False).input_ids
        input_list = input_ids.tolist()
        start = self._find_last_subsequence(input_list, target_ids)
        if start < 0:
            preview = tokenizer.decode(input_list[-256:], skip_special_tokens=False)
            raise RuntimeError(
                "Could not find assistant target tokens inside encoded input_ids. "
                "Cannot build a safe loss mask.\n"
                f"target={target!r}\n"
                f"target_ids[:20]={target_ids[:20]}\n"
                f"decoded_input_tail={preview!r}"
            )
        return start

    def __call__(self, features: List[Dict[str, Any]]) -> Dict[str, torch.Tensor]:
        encoded_list = []
        labels_list = []

        for feat in features:
            full_messages = build_messages(
                frames=feat["frames"],
                question=feat["question"],
                target=feat["target"],
                system_prompt=self.system_prompt,
            )
            encoded, _ = self._encode_messages(full_messages)
            input_ids = encoded["input_ids"][0]
            labels = input_ids.clone()

            target_start = self._target_start(input_ids, feat["target"])
            labels[:target_start] = self.label_pad_token_id

            attention_mask = encoded.get("attention_mask")
            if attention_mask is not None:
                labels[attention_mask[0] == 0] = self.label_pad_token_id

            encoded_list.append(encoded)
            labels_list.append(labels)

        if len(encoded_list) == 1:
            batch = {k: v for k, v in encoded_list[0].items()}
            batch["labels"] = labels_list[0].unsqueeze(0)
            return batch

        tokenizer = getattr(self.processor, "tokenizer", self.processor)
        pad_id = tokenizer.pad_token_id or tokenizer.eos_token_id
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