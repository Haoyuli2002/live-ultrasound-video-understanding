"""
Data collator for ultrasound ASR caption-completion pretraining.

Builds one multimodal chat example per sample:

System:
  ultrasound narration continuation instruction

User:
  [N sampled image frames]
  (optional) Narration so far: {prev_context}

Assistant:
  {target}   # the narration sentence to continue/produce

Loss masking:
  - system/user/image prompt tokens => -100
  - assistant target tokens         => token ids
  - padding                         => -100

No special tokens (<WAIT>/<ANSWER>) are used here: pretraining is pure narration
continuation. Label masking finds the assistant target span inside the fully
encoded (multimodal) input_ids, robust to image-placeholder expansion.
"""

from __future__ import annotations

from dataclasses import dataclass
import random
from typing import Any, Dict, List

import torch


DEFAULT_SYSTEM_PROMPT = """You are an ultrasound teaching assistant.
You are given the most recent ultrasound video frames and, optionally, the
narration transcript so far. Continue the spoken narration, describing what is
happening in the ultrasound scan."""

INTERLEAVE_SYSTEM_PROMPT = """You are an ultrasound teaching assistant.
You are given a sequence of ultrasound video frames interleaved with prior
narration sentences. Each group of frames corresponds to the narration sentence
immediately following it. Continue the sequence by directly producing the next
narration sentence. Output only the next narration sentence."""

MIXEDMASK_SYSTEM_PROMPT = """You are an ultrasound teaching assistant.
You are given a sequence of ultrasound video frames and narration chunks.
Some visual or textual parts may be masked. Use the available visual and textual
context to produce the target narration chunk. Output only the target narration
chunk."""


def _content_with_images(frames, prev_context: str):
    content = []
    for img in frames:
        content.append({"type": "image", "image": img})
    if prev_context:
        content.append({"type": "text", "text": f"Narration so far: {prev_context}\nContinue the narration:"})
    else:
        content.append({"type": "text", "text": "Describe/continue the narration for the current ultrasound view:"})
    return content


def build_messages(frames, prev_context: str, target: str | None = None, system_prompt: str = DEFAULT_SYSTEM_PROMPT):
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": _content_with_images(frames, prev_context)},
    ]
    if target is not None:
        messages.append({"role": "assistant", "content": target})
    return messages


def build_interleave_messages(
    history,
    history_frames,
    target: str | None = None,
    system_prompt: str = INTERLEAVE_SYSTEM_PROMPT,
):
    content = []
    for item, frames in zip(history, history_frames):
        for img in frames:
            content.append({"type": "image", "image": img})
        text = (item.get("text") or "").strip()
        content.append({"type": "text", "text": f"Narration: {text}"})

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": content},
    ]
    if target is not None:
        messages.append({"role": "assistant", "content": target})
    return messages


def _choose_mixedmask_mode(mask_policy: Dict[str, Any] | None, mask_mode: str | None = None) -> str:
    if mask_mode:
        if mask_mode == "random_unit_modality_mask":
            # Backward-compatible alias for older eval commands.  The V4 policy
            # now uses text_modality_mask as the third mode.
            return "text_modality_mask"
        return mask_mode

    policy = mask_policy or {}
    text_modality_mask_prob = policy.get(
        "text_modality_mask_prob",
        # Backward compatibility for samples generated before the rename.
        policy.get("random_unit_modality_mask_prob", 0.34),
    )
    modes = [
        ("mask_current_visual", float(policy.get("mask_current_visual_prob", 0.33))),
        ("no_mask", float(policy.get("no_mask_prob", 0.33))),
        ("text_modality_mask", float(text_modality_mask_prob)),
    ]
    total = sum(max(0.0, prob) for _, prob in modes)
    if total <= 0:
        return "no_mask"

    value = random.random() * total
    cumulative = 0.0
    for mode, prob in modes:
        cumulative += max(0.0, prob)
        if value <= cumulative:
            return mode
    return modes[-1][0]


def _sample_random_mixedmask(history, include_current_visual: bool = True) -> tuple[str, int | None]:
    candidates: List[tuple[str, int | None]] = []
    for idx, _ in enumerate(history):
        candidates.append(("history_visual", idx))
        candidates.append(("history_text", idx))
    if include_current_visual:
        candidates.append(("current_visual", None))
    if not candidates:
        return ("current_visual", None)
    return random.choice(candidates)


def build_mixedmask_messages(
    history,
    history_frames,
    current_visual_frames,
    target: str | None = None,
    mask_policy: Dict[str, Any] | None = None,
    mask_mode: str | None = None,
    system_prompt: str = MIXEDMASK_SYSTEM_PROMPT,
):
    """Build V4 mixed visual/textual masking messages.

    Modes:
      - mask_current_visual: history visual/text only, current visual masked.
      - no_mask: history visual/text + current visual.
      - text_modality_mask: history visual + current visual, all history text masked.
    """
    mode = _choose_mixedmask_mode(mask_policy, mask_mode)
    random_mask = None
    if mode == "random_unit_modality_mask":
        # Legacy direct caller support; normal policy selection no longer emits
        # this mode.
        random_mask = _sample_random_mixedmask(history, include_current_visual=True)

    content = []
    for idx, (item, frames) in enumerate(zip(history, history_frames)):
        mask_visual = random_mask == ("history_visual", idx)
        mask_text = mode == "text_modality_mask" or random_mask == ("history_text", idx)

        if mask_visual:
            content.append({"type": "text", "text": "[VISUAL MASKED]"})
        else:
            for img in frames:
                content.append({"type": "image", "image": img})

        text = (item.get("text") or "").strip()
        if mask_text:
            text = "[TEXT MASKED]"
        content.append({"type": "text", "text": f"Narration: {text}"})

    mask_current_visual = mode == "mask_current_visual" or random_mask == ("current_visual", None)
    if mask_current_visual:
        content.append({"type": "text", "text": "[CURRENT VISUAL MASKED]"})
    else:
        for img in current_visual_frames or []:
            content.append({"type": "image", "image": img})

    content.append({"type": "text", "text": "Narration:"})

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": content},
    ]
    if target is not None:
        messages.append({"role": "assistant", "content": target})
    return messages


def _process_vision(messages):
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
class PretrainCollator:
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
            frames = feat["frames"]
            prev_context = feat.get("prev_context", "")
            target = feat["target"]

            if feat.get("sample_type") == "pretrain_caption_sentence_interleave":
                full_messages = build_interleave_messages(
                    history=feat.get("history", []),
                    history_frames=feat.get("history_frames", []),
                    target=target,
                )
            elif feat.get("sample_type") == "pretrain_next_sentence_mixedmask":
                full_messages = build_mixedmask_messages(
                    history=feat.get("history", []),
                    history_frames=feat.get("history_frames", []),
                    current_visual_frames=feat.get("current_visual_frames", []),
                    target=target,
                    mask_policy=feat.get("mask_policy"),
                )
            else:
                full_messages = build_messages(
                    frames=frames,
                    prev_context=prev_context,
                    target=target,
                    system_prompt=self.system_prompt,
                )
            encoded, _ = self._encode_messages(full_messages)

            input_ids = encoded["input_ids"][0]
            labels = input_ids.clone()

            target_start = self._target_start(input_ids, target)
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

        for k, v in encoded_list[0].items():
            if k not in batch:
                batch[k] = v

        return batch