#!/usr/bin/env python3
"""Train recurrent <SUMMARY> streaming QA with LoRA.

First implementation is intentionally simple and supports batch_size=1. Each
sample performs:
  1) append one summary token per history chunk into a sliding summary bank
  2) QA forward on current_visual + question
  3) loss = causal LM loss for <WAIT>/<ANSWER> target text
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path

os.environ.setdefault("TRANSFORMERS_NO_TF", "1")
os.environ.setdefault("USE_TF", "0")
os.environ.setdefault("USE_FLAX", "0")

import torch
from peft import LoraConfig, get_peft_model, PeftModel
from torch.utils.data import DataLoader
from tqdm.auto import tqdm
from transformers import AutoProcessor

try:
    from transformers import AutoModelForImageTextToText, AutoModelForVision2Seq
except Exception:
    AutoModelForImageTextToText = None
    AutoModelForVision2Seq = None

try:
    from .summary_dataset import SummaryDecideDataset
    from .summary_collator import (
        SUMMARY_TOKEN,
        SummaryDecideCollator,
        qa_messages,
        summary_update_messages,
    )
except ImportError:
    from summary_dataset import SummaryDecideDataset
    from summary_collator import (
        SUMMARY_TOKEN,
        SummaryDecideCollator,
        qa_messages,
        summary_update_messages,
    )


SPECIAL_TOKENS = [SUMMARY_TOKEN, "<WAIT>", "<ANSWER>"]


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
    print("[summary-qa] special tokens:", {tok: tokenizer.convert_tokens_to_ids(tok) for tok in SPECIAL_TOKENS})


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


def move_to_device(batch, device):
    return {k: (v.to(device) if torch.is_tensor(v) else v) for k, v in batch.items()}


def replace_token_embedding(model, encoded, token_position: int, vector: torch.Tensor):
    input_ids = encoded.pop("input_ids")
    embeds = model.get_input_embeddings()(input_ids)
    embeds[:, token_position, :] = vector.to(embeds.device, dtype=embeds.dtype)
    encoded["inputs_embeds"] = embeds
    return encoded


def replace_token_embeddings(model, encoded, token_positions: list[int], vectors: list[torch.Tensor]):
    input_ids = encoded.pop("input_ids")
    embeds = model.get_input_embeddings()(input_ids)
    for pos, vector in zip(token_positions, vectors):
        embeds[:, pos, :] = vector.to(embeds.device, dtype=embeds.dtype)
    encoded["inputs_embeds"] = embeds
    return encoded


def build_summary_bank(model, collator, sample, device, max_bank_size: int):
    summary_bank: list[torch.Tensor] = []
    tokenizer = collator.processor.tokenizer
    summary_id = tokenizer.convert_tokens_to_ids(SUMMARY_TOKEN)
    for chunk in sample["history_chunks"]:
        messages = summary_update_messages(
            chunk["frames"],
            text=chunk.get("text", ""),
            previous_summary_count=len(summary_bank),
        )
        encoded = move_to_device(collator.encode_messages(messages), device)
        input_ids = encoded["input_ids"][0]
        summary_positions = [i for i, tok_id in enumerate(input_ids.tolist()) if tok_id == summary_id]
        if not summary_positions:
            raise RuntimeError("No <SUMMARY> token found in summary update prompt")
        if summary_bank:
            encoded = replace_token_embeddings(model, encoded, summary_positions[:len(summary_bank)], summary_bank)
        outputs = model(**encoded, output_hidden_states=True)
        new_summary = outputs.hidden_states[-1][:, summary_positions[-1], :]
        summary_bank.append(new_summary)
        if len(summary_bank) > max_bank_size:
            summary_bank = summary_bank[-max_bank_size:]
    return summary_bank


def train_one_sample(model, collator, sample, device, max_bank_size: int):
    summary_bank = build_summary_bank(model, collator, sample, device, max_bank_size=max_bank_size)
    messages = qa_messages(
        sample["current_visual_frames"],
        sample["question"],
        summary_count=len(summary_bank),
        target=sample["target"],
    )
    encoded = move_to_device(collator.encode_messages(messages), device)
    input_ids = encoded["input_ids"][0]
    tokenizer = collator.processor.tokenizer
    summary_id = tokenizer.convert_tokens_to_ids(SUMMARY_TOKEN)
    summary_positions = [i for i, tok_id in enumerate(input_ids.tolist()) if tok_id == summary_id]
    if len(summary_positions) < len(summary_bank):
        raise RuntimeError("No <SUMMARY> token found in QA prompt")
    if summary_bank:
        encoded = replace_token_embeddings(model, encoded, summary_positions[:len(summary_bank)], summary_bank)
    labels = input_ids.clone()
    target_start = collator.target_start(input_ids, sample["target"])
    labels[:target_start] = -100
    encoded["labels"] = labels.unsqueeze(0).to(device)
    outputs = model(**encoded)
    return outputs.loss, outputs.loss.detach()


def parse_args():
    p = argparse.ArgumentParser(description="Recurrent <SUMMARY> QA/SFT with <WAIT>/<ANSWER> generation")
    p.add_argument("--model-name", default="Qwen/Qwen3-VL-2B-Instruct")
    p.add_argument("--train-jsonl", required=True)
    p.add_argument("--output-dir", required=True)
    p.add_argument("--repo-root", default=".")
    p.add_argument("--video-root", default=None)
    p.add_argument("--default-video-path", default=None)
    p.add_argument("--video-path-map", default=None)
    p.add_argument("--frames-per-chunk", type=int, default=3)
    p.add_argument("--frame-size", type=int, default=224)
    p.add_argument("--summary-bank-size", type=int, default=20)
    p.add_argument("--limit", type=int, default=None)
    p.add_argument("--num-train-epochs", type=int, default=1)
    p.add_argument("--learning-rate", type=float, default=1e-4)
    p.add_argument("--bf16", action="store_true")
    p.add_argument("--fp16", action="store_true")
    p.add_argument("--init-adapter", default=None)
    p.add_argument("--lora-r", type=int, default=16)
    p.add_argument("--lora-alpha", type=int, default=32)
    p.add_argument("--lora-dropout", type=float, default=0.05)
    p.add_argument("--lora-target-modules", default="q_proj,k_proj,v_proj,o_proj,gate_proj,up_proj,down_proj")
    p.add_argument("--lora-modules-to-save", default="embed_tokens,lm_head")
    p.add_argument("--grad-clip", type=float, default=1.0)
    return p.parse_args()


def main():
    args = parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.bfloat16 if args.bf16 else torch.float16 if args.fp16 else torch.float32
    processor = AutoProcessor.from_pretrained(args.model_name, trust_remote_code=True)
    model = load_model(args.model_name, dtype)
    if args.init_adapter:
        model = PeftModel.from_pretrained(model, args.init_adapter).merge_and_unload()
    add_special_tokens(model, processor)
    model = build_lora(args, model)
    model.to(device)
    model.train()

    dataset = SummaryDecideDataset(
        args.train_jsonl,
        repo_root=args.repo_root,
        video_root=args.video_root,
        default_video_path=args.default_video_path,
        video_path_map=args.video_path_map,
        frames_per_chunk=args.frames_per_chunk,
        frame_size=args.frame_size,
        limit=args.limit,
    )
    collator = SummaryDecideCollator(processor=processor)
    loader = DataLoader(dataset, batch_size=1, shuffle=True, collate_fn=collator)
    optim = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=args.learning_rate)

    for epoch in range(args.num_train_epochs):
        pbar = tqdm(loader, desc=f"summary-qa epoch {epoch + 1}")
        for sample in pbar:
            optim.zero_grad(set_to_none=True)
            loss, lm_loss = train_one_sample(model, collator, sample, device=device, max_bank_size=args.summary_bank_size)
            loss.backward()
            if args.grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
            optim.step()
            pbar.set_postfix(loss=float(loss.detach()), lm=float(lm_loss))

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(out_dir)
    processor.save_pretrained(out_dir)
    print(f"[summary-qa] saved to {out_dir}")


if __name__ == "__main__":
    main()