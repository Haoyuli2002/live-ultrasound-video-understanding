#!/usr/bin/env python3
"""Stage-2 summary-compression training for Qwen3-VL + LoRA.

Main design: video-only memory tokens reconstruct teacher-generated visual
summaries:
  short: current short memories -> local visual summary
  long : L_{k-1} + S_k -> L_k -> cumulative visual summary

The previously implemented ASR reconstruction objective is still supported as a
baseline / bootstrapping path for validating the memory-token pipeline.
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path

os.environ.setdefault("TRANSFORMERS_NO_TF", "1")
os.environ.setdefault("USE_TF", "0")
os.environ.setdefault("USE_FLAX", "0")

import torch
from peft import LoraConfig, PeftModel, get_peft_model
from torch.utils.data import DataLoader
from tqdm.auto import tqdm
from transformers import AutoProcessor

try:
    from transformers import AutoModelForImageTextToText, AutoModelForVision2Seq
except Exception:
    AutoModelForImageTextToText = None
    AutoModelForVision2Seq = None

try:
    from .memory_dataset import MemoryCompressionDataset
    from .memory_collator import (
        LONG_MEMORY_TOKEN,
        SHORT_MEMORY_TOKEN,
        MemoryCompressionCollator,
        long_decode_messages,
        long_encode_messages,
        short_decode_messages,
        short_encode_messages,
    )
except ImportError:
    from memory_dataset import MemoryCompressionDataset
    from memory_collator import (
        LONG_MEMORY_TOKEN,
        SHORT_MEMORY_TOKEN,
        MemoryCompressionCollator,
        long_decode_messages,
        long_encode_messages,
        short_decode_messages,
        short_encode_messages,
    )


SPECIAL_TOKENS = [SHORT_MEMORY_TOKEN, LONG_MEMORY_TOKEN]


def load_model(model_name: str, dtype):
    kwargs = {"trust_remote_code": True, "dtype": dtype}
    errors = []
    for cls in [AutoModelForImageTextToText, AutoModelForVision2Seq]:
        if cls is None:
            continue
        try:
            return cls.from_pretrained(model_name, **kwargs)
        except Exception as exc:
            errors.append((cls.__name__, repr(exc)))
    from transformers import AutoModelForCausalLM
    try:
        return AutoModelForCausalLM.from_pretrained(model_name, **kwargs)
    except Exception as exc:
        errors.append(("AutoModelForCausalLM", repr(exc)))
    raise RuntimeError("Could not load model:\n" + "\n".join(f"{n}: {e}" for n, e in errors))


def add_special_tokens(model, processor):
    tokenizer = getattr(processor, "tokenizer", processor)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    vocab = tokenizer.get_vocab()
    to_add = [tok for tok in SPECIAL_TOKENS if tok not in vocab]
    if to_add:
        tokenizer.add_special_tokens({"additional_special_tokens": to_add})
        model.resize_token_embeddings(len(tokenizer))
    print("[memory] special tokens:", {tok: tokenizer.convert_tokens_to_ids(tok) for tok in SPECIAL_TOKENS})


def build_lora(args, model):
    target_modules = [x.strip() for x in args.lora_target_modules.split(",") if x.strip()]
    modules_to_save = [x.strip() for x in args.lora_modules_to_save.split(",") if x.strip()]
    cfg = LoraConfig(
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=target_modules,
        modules_to_save=modules_to_save or None,
    )
    model = get_peft_model(model, cfg)
    model.print_trainable_parameters()
    return model


def maybe_freeze_vision(model, freeze_vision: bool):
    if not freeze_vision:
        return
    frozen = 0
    for name, param in model.named_parameters():
        lname = name.lower()
        if "visual" in lname or "vision" in lname or "vision_tower" in lname:
            param.requires_grad = False
            frozen += param.numel()
    print(f"[memory] Frozen vision parameters: {frozen:,}")


def move_to_device(batch, device):
    return {k: (v.to(device) if torch.is_tensor(v) else v) for k, v in batch.items()}


def replace_token_embeddings(model, encoded, token_positions: list[int], vectors: list[torch.Tensor]):
    input_ids = encoded.pop("input_ids")
    embeds = model.get_input_embeddings()(input_ids)
    for pos, vec in zip(token_positions, vectors):
        embeds[:, pos, :] = vec.to(embeds.device, dtype=embeds.dtype)
    encoded["inputs_embeds"] = embeds
    return encoded


def token_positions(input_ids: torch.Tensor, token_id: int) -> list[int]:
    return [i for i, tok_id in enumerate(input_ids.tolist()) if tok_id == token_id]


def encode_short_memory(model, collator, frames, device):
    tokenizer = collator.processor.tokenizer
    short_id = tokenizer.convert_tokens_to_ids(SHORT_MEMORY_TOKEN)
    encoded = move_to_device(collator.encode_messages(short_encode_messages(frames)), device)
    input_ids = encoded["input_ids"][0]
    positions = token_positions(input_ids, short_id)
    if not positions:
        raise RuntimeError("No <SHORT_MEM> token in short encode prompt")
    outputs = model(**encoded, output_hidden_states=True)
    return outputs.hidden_states[-1][:, positions[-1], :]


def encode_short_states(model, collator, frames_list, device):
    return [encode_short_memory(model, collator, frames, device) for frames in frames_list]


def decode_memory_loss(model, collator, messages, memory_token: str, memory_states: list[torch.Tensor], target: str, device):
    tokenizer = collator.processor.tokenizer
    tok_id = tokenizer.convert_tokens_to_ids(memory_token)
    encoded = move_to_device(collator.encode_messages(messages), device)
    input_ids = encoded["input_ids"][0]
    positions = token_positions(input_ids, tok_id)
    if len(positions) < len(memory_states):
        raise RuntimeError(f"No {memory_token} token in decode prompt")
    encoded = replace_token_embeddings(model, encoded, positions[-len(memory_states):], memory_states)
    labels = input_ids.clone()
    target_start = collator.target_start(input_ids, target)
    labels[:target_start] = collator.label_pad_token_id
    attention_mask = encoded.get("attention_mask")
    if attention_mask is not None:
        labels[attention_mask[0] == 0] = collator.label_pad_token_id
    encoded["labels"] = labels.unsqueeze(0).to(device)
    return model(**encoded).loss


def short_loss(model, collator, sample, device):
    short_states = encode_short_states(model, collator, sample["short_frames_list"], device)
    target = sample.get("local_summary_target") or sample.get("target")
    task = "local_summary" if sample.get("local_summary_target") else "asr"
    if not target:
        raise RuntimeError("short memory sample has no target/local_summary_target")
    return decode_memory_loss(
        model,
        collator,
        short_decode_messages(short_count=len(short_states), target=target, task=task),
        SHORT_MEMORY_TOKEN,
        short_states,
        target,
        device,
    )


def encode_long_states(model, collator, short_states, previous_long_states, long_token_count: int, device):
    tokenizer = collator.processor.tokenizer
    short_id = tokenizer.convert_tokens_to_ids(SHORT_MEMORY_TOKEN)
    long_id = tokenizer.convert_tokens_to_ids(LONG_MEMORY_TOKEN)
    encoded = move_to_device(collator.encode_messages(long_encode_messages(
        short_count=len(short_states),
        previous_long_count=len(previous_long_states),
        new_long_count=long_token_count,
    )), device)
    input_ids = encoded["input_ids"][0]
    short_positions = token_positions(input_ids, short_id)
    long_positions = token_positions(input_ids, long_id)
    if len(short_positions) < len(short_states):
        raise RuntimeError("Not enough <SHORT_MEM> tokens in long encode prompt")
    if len(long_positions) < len(previous_long_states) + long_token_count:
        raise RuntimeError("No <LONG_MEM> token in long encode prompt")
    previous_positions = long_positions[:len(previous_long_states)]
    new_positions = long_positions[-long_token_count:]
    positions = previous_positions + short_positions[:len(short_states)]
    vectors = previous_long_states + short_states
    if positions:
        encoded = replace_token_embeddings(model, encoded, positions, vectors)
    outputs = model(**encoded, output_hidden_states=True)
    return [outputs.hidden_states[-1][:, pos, :] for pos in new_positions]


def build_previous_long_states(model, collator, sample, long_token_count: int, device):
    long_states: list[torch.Tensor] = []
    for block in sample.get("previous_blocks", []):
        short_states = encode_short_states(model, collator, block.get("short_frames_list", []), device)
        if short_states:
            long_states = encode_long_states(model, collator, short_states, long_states, long_token_count, device)
    return long_states


def long_loss(model, collator, sample, device, *, lambda_all: float, lambda_cur: float, lambda_prev: float):
    long_token_count = int(sample.get("num_long_tokens") or 60)
    short_states = encode_short_states(model, collator, sample["short_frames_list"], device)
    if not short_states:
        raise RuntimeError("long_memory_compression sample has no short_frames_list")
    previous_long_states = build_previous_long_states(model, collator, sample, long_token_count, device)
    long_states = encode_long_states(model, collator, short_states, previous_long_states, long_token_count, device)

    losses = []
    weights = []
    if sample.get("sample_type") == "long_memory_summary":
        if lambda_cur > 0 and sample.get("local_summary_target"):
            target = sample["local_summary_target"]
            losses.append(decode_memory_loss(model, collator, long_decode_messages(long_count=len(long_states), task="local_summary", target=target), LONG_MEMORY_TOKEN, long_states, target, device))
            weights.append(lambda_cur)
        if lambda_all > 0 and sample.get("global_summary_target"):
            target = sample["global_summary_target"]
            losses.append(decode_memory_loss(model, collator, long_decode_messages(long_count=len(long_states), task="global_summary", target=target), LONG_MEMORY_TOKEN, long_states, target, device))
            weights.append(lambda_all)
        if not losses:
            raise RuntimeError("long_memory_summary sample has no local_summary_target/global_summary_target")
        return sum(w * l for w, l in zip(weights, losses)) / sum(weights)

    if lambda_all > 0 and sample.get("accumulated_summary_target"):
        target = sample["accumulated_summary_target"]
        losses.append(decode_memory_loss(model, collator, long_decode_messages(long_count=len(long_states), task="accumulated", target=target), LONG_MEMORY_TOKEN, long_states, target, device))
        weights.append(lambda_all)
    if lambda_cur > 0 and sample.get("current_block_target"):
        target = sample["current_block_target"]
        losses.append(decode_memory_loss(model, collator, long_decode_messages(long_count=len(long_states), task="current", target=target), LONG_MEMORY_TOKEN, long_states, target, device))
        weights.append(lambda_cur)
    if lambda_prev > 0 and sample.get("previous_summary_target"):
        target = sample["previous_summary_target"]
        losses.append(decode_memory_loss(model, collator, long_decode_messages(long_count=len(long_states), task="previous", target=target), LONG_MEMORY_TOKEN, long_states, target, device))
        weights.append(lambda_prev)
    if not losses:
        target = sample["target"]
        losses.append(decode_memory_loss(model, collator, long_decode_messages(long_count=len(long_states), task="accumulated", target=target), LONG_MEMORY_TOKEN, long_states, target, device))
        weights.append(1.0)
    return sum(w * loss for w, loss in zip(weights, losses)) / sum(weights)


def train_one_sample(model, collator, sample, device, *, lambda_all: float, lambda_cur: float, lambda_prev: float):
    typ = sample.get("sample_type")
    if typ in {"short_memory_compression", "short_memory_summary"}:
        return short_loss(model, collator, sample, device)
    if typ in {"long_memory_compression", "long_memory_summary"}:
        return long_loss(model, collator, sample, device, lambda_all=lambda_all, lambda_cur=lambda_cur, lambda_prev=lambda_prev)
    raise ValueError(f"Unsupported sample_type: {typ}")


def parse_args():
    p = argparse.ArgumentParser(description="Stage-2 summary compression training")
    p.add_argument("--model-name", default="Qwen/Qwen3-VL-2B-Instruct")
    p.add_argument("--train-jsonl", required=True)
    p.add_argument("--output-dir", required=True)
    p.add_argument("--repo-root", default=".")
    p.add_argument("--video-root", default=None)
    p.add_argument("--default-video-path", default=None)
    p.add_argument("--video-path-map", default=None)
    p.add_argument("--short-frames", type=int, default=2)
    p.add_argument("--frame-size", type=int, default=224)
    p.add_argument("--max-short-windows", type=int, default=None)
    p.add_argument("--max-previous-blocks", type=int, default=1)
    p.add_argument("--lambda-all", type=float, default=1.0)
    p.add_argument("--lambda-current", type=float, default=0.5)
    p.add_argument("--lambda-previous", type=float, default=0.3)
    p.add_argument("--limit", type=int, default=None)
    p.add_argument("--num-train-epochs", type=int, default=1)
    p.add_argument("--learning-rate", type=float, default=1e-4)
    p.add_argument("--bf16", action="store_true")
    p.add_argument("--fp16", action="store_true")
    p.add_argument("--init-adapter", default=None)
    p.add_argument("--freeze-vision", action="store_true", default=True)
    p.add_argument("--no-freeze-vision", action="store_false", dest="freeze_vision")
    p.add_argument("--lora-r", type=int, default=16)
    p.add_argument("--lora-alpha", type=int, default=32)
    p.add_argument("--lora-dropout", type=float, default=0.05)
    p.add_argument("--lora-target-modules", default="q_proj,k_proj,v_proj,o_proj,gate_proj,up_proj,down_proj")
    p.add_argument("--lora-modules-to-save", default="embed_tokens,lm_head")
    p.add_argument("--grad-clip", type=float, default=1.0)
    return p.parse_args()


def main():
    args = parse_args()
    if args.bf16 and args.fp16:
        raise ValueError("Use only one precision flag: --bf16 or --fp16")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.bfloat16 if args.bf16 else torch.float16 if args.fp16 else torch.float32
    processor = AutoProcessor.from_pretrained(args.model_name, trust_remote_code=True)
    model = load_model(args.model_name, dtype)
    if args.init_adapter:
        model = PeftModel.from_pretrained(model, args.init_adapter).merge_and_unload()
    add_special_tokens(model, processor)
    maybe_freeze_vision(model, args.freeze_vision)
    model = build_lora(args, model)
    model.to(device).train()

    dataset = MemoryCompressionDataset(
        args.train_jsonl,
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
    loader = DataLoader(dataset, batch_size=1, shuffle=True, collate_fn=collator)
    optim = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=args.learning_rate)

    print(f"[memory] dataset size={len(dataset)}")
    for epoch in range(args.num_train_epochs):
        pbar = tqdm(loader, desc=f"memory epoch {epoch + 1}")
        for sample in pbar:
            optim.zero_grad(set_to_none=True)
            loss = train_one_sample(
                model,
                collator,
                sample,
                device,
                lambda_all=args.lambda_all,
                lambda_cur=args.lambda_current,
                lambda_prev=args.lambda_previous,
            )
            loss.backward()
            if args.grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
            optim.step()
            pbar.set_postfix(loss=float(loss.detach()), type=sample.get("sample_type", ""))

    out = Path(args.output_dir); out.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(out)
    processor.save_pretrained(out)
    print(f"[memory] saved to {out}")


if __name__ == "__main__":
    main()