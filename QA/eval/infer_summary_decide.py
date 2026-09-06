#!/usr/bin/env python3
"""Inference for recurrent <SUMMARY> streaming QA checkpoints.

Answerability is decided by the LM logits of the next token:
P(<WAIT> | S_t, V_t, Q) vs P(<ANSWER> | S_t, V_t, Q).
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any, Dict

os.environ.setdefault("TRANSFORMERS_NO_TF", "1")
os.environ.setdefault("USE_TF", "0")
os.environ.setdefault("USE_FLAX", "0")

import torch
from peft import PeftModel
from transformers import AutoProcessor

try:
    from transformers import AutoModelForImageTextToText, AutoModelForVision2Seq
except Exception:
    AutoModelForImageTextToText = None
    AutoModelForVision2Seq = None

_HERE = Path(__file__).resolve().parent
_TRAIN = _HERE.parent / "train"
sys.path.insert(0, str(_TRAIN))

from summary_dataset import SummaryDecideDataset  # noqa: E402
from summary_collator import (  # noqa: E402
    SUMMARY_TOKEN,
    SummaryDecideCollator,
    qa_messages,
    summary_update_messages,
)
from train_summary_decide import load_model, replace_token_embedding, replace_token_embeddings  # noqa: E402


def move_to_device(batch, device):
    return {k: (v.to(device) if torch.is_tensor(v) else v) for k, v in batch.items()}


def update_summary_bank(model, collator, sample, device, max_bank_size: int):
    summary_bank = []
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
        positions = [i for i, tok_id in enumerate(input_ids.tolist()) if tok_id == summary_id]
        if not positions:
            raise RuntimeError("No <SUMMARY> in update prompt")
        if summary_bank:
            encoded = replace_token_embeddings(model, encoded, positions[:len(summary_bank)], summary_bank)
        with torch.no_grad():
            outputs = model(**encoded, output_hidden_states=True)
        summary_bank.append(outputs.hidden_states[-1][:, positions[-1], :])
        if len(summary_bank) > max_bank_size:
            summary_bank = summary_bank[-max_bank_size:]
    return summary_bank


def qa_logits_and_generate(model, collator, sample, summary_bank, device, max_new_tokens: int) -> Dict[str, Any]:
    messages = qa_messages(
        sample["current_visual_frames"],
        sample["question"],
        summary_count=len(summary_bank),
        target=None,
    )
    encoded = move_to_device(collator.encode_messages(messages), device)
    input_ids = encoded["input_ids"][0]
    tokenizer = collator.processor.tokenizer
    summary_id = tokenizer.convert_tokens_to_ids(SUMMARY_TOKEN)
    wait_id = tokenizer.convert_tokens_to_ids("<WAIT>")
    answer_id = tokenizer.convert_tokens_to_ids("<ANSWER>")
    summary_positions = [i for i, tok_id in enumerate(input_ids.tolist()) if tok_id == summary_id]
    if len(summary_positions) < len(summary_bank):
        raise RuntimeError("Missing <SUMMARY> in QA prompt")

    encoded_for_forward = {k: v.clone() if torch.is_tensor(v) else v for k, v in encoded.items()}
    if summary_bank:
        encoded_for_forward = replace_token_embeddings(model, encoded_for_forward, summary_positions[:len(summary_bank)], summary_bank)
    with torch.no_grad():
        outputs = model(**encoded_for_forward)
        next_logits = outputs.logits[0, -1].float()
        logit_wait = float(next_logits[wait_id].detach().cpu())
        logit_answer = float(next_logits[answer_id].detach().cpu())
    pred_label = "ANSWER" if logit_answer >= logit_wait else "WAIT"

    # Generate text from the same prompt. HF generate cannot easily reuse replaced
    # summary embeddings across model classes, so V1 uses the textual <SUMMARY>
    # placeholder for generation and the injected hidden state for logits.
    with torch.no_grad():
        generated = model.generate(**encoded, max_new_tokens=max_new_tokens, do_sample=False)
    new_tokens = generated[:, encoded["input_ids"].shape[1]:]
    prediction = collator.processor.batch_decode(new_tokens, skip_special_tokens=False, clean_up_tokenization_spaces=False)[0].strip()
    return {
        "prediction": prediction,
        "pred_label": pred_label,
        "logit_WAIT": logit_wait,
        "logit_ANSWER": logit_answer,
    }


def parse_args():
    p = argparse.ArgumentParser(description="Infer recurrent <SUMMARY> QA")
    p.add_argument("--model-name", default="Qwen/Qwen3-VL-2B-Instruct")
    p.add_argument("--adapter-path", required=True)
    p.add_argument("--eval-jsonl", required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--repo-root", default=".")
    p.add_argument("--video-root", default=None)
    p.add_argument("--default-video-path", default=None)
    p.add_argument("--video-path-map", default=None)
    p.add_argument("--frames-per-chunk", type=int, default=3)
    p.add_argument("--frame-size", type=int, default=224)
    p.add_argument("--summary-bank-size", type=int, default=20)
    p.add_argument("--limit", type=int, default=None)
    p.add_argument("--max-new-tokens", type=int, default=128)
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
    model = PeftModel.from_pretrained(model, args.adapter_path)
    model.to(device).eval()

    dataset = SummaryDecideDataset(
        args.eval_jsonl,
        repo_root=args.repo_root,
        video_root=args.video_root,
        default_video_path=args.default_video_path,
        video_path_map=args.video_path_map,
        frames_per_chunk=args.frames_per_chunk,
        frame_size=args.frame_size,
        limit=args.limit,
    )
    collator = SummaryDecideCollator(processor=processor)
    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", encoding="utf-8") as f:
        for idx in range(len(dataset)):
            sample = dataset[idx]
            summary_bank = update_summary_bank(model, collator, sample, device, max_bank_size=args.summary_bank_size)
            rec = qa_logits_and_generate(model, collator, sample, summary_bank, device, args.max_new_tokens)
            target = str(sample.get("target") or "")
            rec.update({
                "idx": idx,
                "video_id": sample.get("video_id"),
                "question": sample.get("question"),
                "target": target,
                "gt_label": "WAIT" if target.strip().upper().startswith("<WAIT>") else "ANSWER",
            })
            rec["correct_answerability"] = rec["pred_label"] == rec["gt_label"]
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
    print(f"[summary-qa] wrote {out}")


if __name__ == "__main__":
    main()