"""Identical Stage 1 train/inference templates and target-only loss masking."""
from __future__ import annotations

import torch

SYSTEM = ("You are an ultrasound teaching assistant. Use the provided ultrasound "
          "frames and any available earlier narration to predict the next complete "
          "narration sentence. Output only that sentence.")
FALLBACK_SYSTEM = ("You are an ultrasound teaching assistant. Use the provided ultrasound "
                   "frames and any available earlier narration to predict the next "
                   "time-aligned narration utterance. Output only that utterance.")


def messages(row: dict, frames: list, *, max_asr_chars: int = 4000,
             target: str | None = None) -> list[dict]:
    content = [{"type": "image", "image": frame} for frame in frames]
    if row["historical_asr_masked"]:
        content.append({"type": "text", "text": "Earlier narration: [ASR MASKED]"})
    else:
        history = row["historical_asr"][-max_asr_chars:] if max_asr_chars > 0 else row["historical_asr"]
        content.append({"type": "text", "text": f"Earlier narration: {history}"})
    fallback = row.get("target_unit_type") == "asr_segment_fallback"
    instruction = ("Predict the target time-aligned narration utterance:" if fallback
                   else "Predict the target complete narration sentence:")
    content.append({"type": "text", "text": instruction})
    result = [{"role": "system", "content": FALLBACK_SYSTEM if fallback else SYSTEM},
              {"role": "user", "content": content}]
    if target is not None:
        result.append({"role": "assistant", "content": target})
    return result


def encode(processor, row: dict, frames: list, *, max_asr_chars: int = 4000,
           include_target: bool = True):
    conversation = messages(row, frames, max_asr_chars=max_asr_chars,
                            target=row["target"] if include_target else None)
    text = processor.apply_chat_template(conversation, tokenize=False,
                                         add_generation_prompt=not include_target)
    batch = processor(text=[text], images=frames, return_tensors="pt")
    if not include_target:
        return batch
    tokenizer = processor.tokenizer
    target_ids = tokenizer(row["target"], add_special_tokens=False).input_ids
    source = batch["input_ids"][0].tolist()
    matches = [i for i in range(len(source)-len(target_ids)+1)
               if source[i:i+len(target_ids)] == target_ids]
    if not target_ids or not matches:
        raise RuntimeError("Target tokens not found in multimodal chat encoding; refusing an unsafe loss mask")
    start = matches[-1]
    labels = batch["input_ids"].clone()
    labels[:, :start] = -100
    if "attention_mask" in batch:
        labels[batch["attention_mask"] == 0] = -100
    batch["labels"] = labels
    return batch


def load_model(model_name: str, dtype: torch.dtype, lora_rank: int = 16):
    from transformers import AutoModelForImageTextToText, AutoProcessor
    from peft import LoraConfig, get_peft_model

    processor = AutoProcessor.from_pretrained(model_name, trust_remote_code=True)
    model = AutoModelForImageTextToText.from_pretrained(
        model_name, torch_dtype=dtype, trust_remote_code=True)
    config = LoraConfig(r=lora_rank, lora_alpha=2*lora_rank, lora_dropout=0.05,
                        target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                                        "gate_proj", "up_proj", "down_proj"],
                        task_type="CAUSAL_LM")
    return get_peft_model(model, config), processor
