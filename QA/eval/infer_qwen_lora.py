#!/usr/bin/env python3
"""
Run inference with a Qwen-VL base model plus a PEFT/LoRA adapter checkpoint.

This is intended for checking the qualitative effect of QA SFT checkpoints, e.g.
whether the fine-tuned model emits the learned single-token decision prefixes:

  <WAIT>
  <ANSWER>

Important:
  - The processor/tokenizer is loaded from --adapter-path first, because the
    training checkpoint contains the added special tokens.
  - The base model embedding matrix is resized to len(tokenizer) before loading
    the LoRA adapter, so modules_to_save such as embed_tokens/lm_head can load.
  - Decoding uses skip_special_tokens=False so <WAIT>/<ANSWER> remain visible.

Example:

  python QA/eval/infer_qwen_lora.py \
    --model-name Qwen/Qwen3-VL-2B-Instruct \
    --adapter-path QA/checkpoints/smoke_qwen3vl_bf16 \
    --eval-jsonl azure_data/QA/results/8V649L5Q368_training_samples.jsonl \
    --default-video-path azure_data/videos/8V649L5Q368.mp4 \
    --output QA/eval/results/smoke_lora_predictions_limit4.jsonl \
    --window-size 8 \
    --frame-size 336 \
    --limit 4 \
    --max-new-tokens 160 \
    --bf16
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
from pathlib import Path
from typing import Any, Dict, List


# Keep Transformers from importing TensorFlow/Flax in AzureML mixed envs.
os.environ.setdefault("TRANSFORMERS_NO_TF", "1")
os.environ.setdefault("USE_TF", "0")
os.environ.setdefault("USE_FLAX", "0")


import torch  # noqa: E402
from transformers import AutoProcessor  # noqa: E402

try:
    from transformers import AutoModelForImageTextToText  # noqa: E402
except Exception:
    AutoModelForImageTextToText = None

try:
    from transformers import AutoModelForVision2Seq  # noqa: E402
except Exception:
    AutoModelForVision2Seq = None

try:
    from peft import PeftModel  # noqa: E402
except ImportError as e:
    raise ImportError("PEFT is required. Install with: pip install peft") from e


# Import shared QA training utilities.
_HERE = Path(__file__).resolve().parent
_REPO_ROOT = _HERE.parent.parent
_TRAIN_DIR = _HERE.parent / "train"
sys.path.insert(0, str(_TRAIN_DIR))

from dataset import QATrainingDataset  # noqa: E402
from collator import DEFAULT_SYSTEM_PROMPT, build_messages  # noqa: E402


SPECIAL_TOKENS = ["<WAIT>", "<ANSWER>"]


def _from_pretrained_with_dtype(model_cls, model_name: str, *, dtype, **kwargs):
    """
    Transformers now prefers `dtype=` over deprecated `torch_dtype=`.
    Keep a fallback for older versions.
    """
    try:
        return model_cls.from_pretrained(model_name, dtype=dtype, **kwargs)
    except TypeError:
        return model_cls.from_pretrained(model_name, torch_dtype=dtype, **kwargs)


def load_qwen_vl_base_model(model_name: str, dtype, device: str):
    """Load Qwen-VL base model using available HF Auto classes."""
    kwargs = {
        "trust_remote_code": True,
        "low_cpu_mem_usage": True,
    }
    errors = []

    if AutoModelForImageTextToText is not None:
        try:
            model = _from_pretrained_with_dtype(
                AutoModelForImageTextToText,
                model_name,
                dtype=dtype,
                **kwargs,
            )
            return model.to(device)
        except Exception as e:
            errors.append(("AutoModelForImageTextToText", repr(e)))

    if AutoModelForVision2Seq is not None:
        try:
            model = _from_pretrained_with_dtype(
                AutoModelForVision2Seq,
                model_name,
                dtype=dtype,
                **kwargs,
            )
            return model.to(device)
        except Exception as e:
            errors.append(("AutoModelForVision2Seq", repr(e)))

    from transformers import AutoModelForCausalLM

    try:
        model = _from_pretrained_with_dtype(
            AutoModelForCausalLM,
            model_name,
            dtype=dtype,
            **kwargs,
        )
        return model.to(device)
    except Exception as e:
        errors.append(("AutoModelForCausalLM", repr(e)))

    msg = "\n".join(f"{name}: {err}" for name, err in errors)
    raise RuntimeError(f"Could not load model {model_name}. Tried multiple Auto classes:\n{msg}")


def load_processor(adapter_path: str | Path, model_name: str):
    """
    Prefer processor/tokenizer from adapter checkpoint because it contains
    the added <WAIT>/<ANSWER> special tokens. Fall back to base model processor.
    """
    adapter_path = str(adapter_path)
    try:
        print(f"[infer-lora] loading processor from adapter: {adapter_path}")
        processor = AutoProcessor.from_pretrained(adapter_path, trust_remote_code=True)
    except Exception as e:
        print(f"[infer-lora] WARNING: failed to load processor from adapter: {type(e).__name__}: {e}")
        print(f"[infer-lora] loading processor from base model: {model_name}")
        processor = AutoProcessor.from_pretrained(model_name, trust_remote_code=True)

    tokenizer = getattr(processor, "tokenizer", processor)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # If fallback processor did not contain the special tokens, add them so the
    # base embedding matrix can be resized consistently. Prefer adapter processor
    # whenever possible because it preserves trained token ids.
    vocab = tokenizer.get_vocab()
    tokens_to_add = [tok for tok in SPECIAL_TOKENS if tok not in vocab]
    if tokens_to_add:
        tokenizer.add_special_tokens({"additional_special_tokens": tokens_to_add})
        print(f"[infer-lora] Added missing special tokens to tokenizer: {tokens_to_add}")

    print_special_token_info(tokenizer)
    return processor, tokenizer


def print_special_token_info(tokenizer) -> None:
    print(f"[infer-lora] vocab={len(tokenizer)}")
    for tok in SPECIAL_TOKENS:
        ids = tokenizer(tok, add_special_tokens=False).input_ids
        print(
            f"[infer-lora] {tok}: "
            f"id={tokenizer.convert_tokens_to_ids(tok)} "
            f"in_vocab={tok in tokenizer.get_vocab()} "
            f"num_tokens={len(ids)} "
            f"ids={ids}"
        )


def process_vision(messages):
    """
    Extract image/video inputs for Qwen processors.

    Uses qwen_vl_utils when available; falls back to manually collecting PIL
    images from message content.
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


def model_generate_one(
    model,
    processor,
    *,
    frames,
    question: str,
    device: str,
    max_new_tokens: int = 128,
) -> str:
    messages = build_messages(
        frames=frames,
        question=question,
        target=None,
        system_prompt=DEFAULT_SYSTEM_PROMPT,
    )

    text = processor.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )
    image_inputs, video_inputs = process_vision(messages)

    kwargs = {
        "text": [text],
        "padding": True,
        "return_tensors": "pt",
    }
    if image_inputs:
        kwargs["images"] = image_inputs
    if video_inputs:
        kwargs["videos"] = video_inputs

    inputs = processor(**kwargs)
    inputs = {
        k: (v.to(device) if torch.is_tensor(v) else v)
        for k, v in inputs.items()
    }

    with torch.no_grad():
        generated_ids = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            temperature=None,
            top_p=None,
        )

    input_len = inputs["input_ids"].shape[1]
    new_tokens = generated_ids[:, input_len:]
    decoded = processor.batch_decode(
        new_tokens,
        skip_special_tokens=False,
        clean_up_tokenization_spaces=False,
    )[0]
    return decoded.strip()


def strip_target_prefix(text: str) -> str:
    text = (text or "").strip()
    text = re.sub(r"^\s*<\s*ANSWER\s*>\s*", "", text, flags=re.IGNORECASE)
    text = re.sub(r"^\s*<\s*WAIT\s*>\s*", "", text, flags=re.IGNORECASE)
    return text.strip()


def gt_label_from_target(target: str) -> str:
    target = (target or "").strip().upper()
    return "WAIT" if target.startswith("<WAIT>") else "ANSWER"


def pred_label_from_text(text: str) -> str:
    """
    Heuristic label parser.

    Fine-tuned checkpoints should ideally emit literal <WAIT>/<ANSWER>. For
    robustness, common wait/refusal phrases are also treated as WAIT.
    """
    s = (text or "").strip()
    low = s.lower()

    if not s:
        return "OTHER"

    if re.match(r"^\s*<\s*wait\s*>", low):
        return "WAIT"
    if re.match(r"^\s*<\s*answer\s*>", low):
        return "ANSWER"

    if "not enough information" in low:
        return "WAIT"
    if "need more information" in low or "need more context" in low:
        return "WAIT"
    if "cannot answer yet" in low or "can't answer yet" in low:
        return "WAIT"
    if "more video is needed" in low or "more context is needed" in low:
        return "WAIT"

    # If it is not a wait/refusal, treat substantive text as attempting ANSWER.
    if len(s.split()) >= 3:
        return "ANSWER"

    return "OTHER"


def make_prediction_record(idx: int, sample: Dict[str, Any], prediction: str) -> Dict[str, Any]:
    gt_label = gt_label_from_target(sample.get("target", ""))
    pred_label = pred_label_from_text(prediction)
    return {
        "idx": idx,
        "sample_type": sample.get("sample_type"),
        "qa_type": sample.get("qa_type"),
        "video_id": sample.get("video_id"),
        "clip_idx": sample.get("clip_idx"),
        "video_window": sample.get("video_window"),
        "question": sample.get("question"),
        "target": sample.get("target"),
        "target_answer_text": strip_target_prefix(sample.get("target", "")),
        "prediction": prediction,
        "prediction_answer_text": strip_target_prefix(prediction),
        "gt_label": gt_label,
        "pred_label": pred_label,
        "correct_answerability": pred_label == gt_label,
        "meta": sample.get("meta", {}),
    }


def print_record_summary(record: Dict[str, Any]) -> None:
    print("-" * 80)
    print(
        f"[infer-lora] idx={record['idx']} "
        f"sample_type={record.get('sample_type')} "
        f"qa_type={record.get('qa_type')} "
        f"gt={record.get('gt_label')} pred={record.get('pred_label')} "
        f"correct={record.get('correct_answerability')}"
    )
    print(f"[infer-lora] question: {record.get('question')}")
    print(f"[infer-lora] target: {record.get('target')}")
    print(f"[infer-lora] prediction: {record.get('prediction')}")


def parse_args():
    parser = argparse.ArgumentParser(description="Run Qwen-VL + LoRA adapter inference on QA samples")
    parser.add_argument("--model-name", type=str, default="Qwen/Qwen3-VL-2B-Instruct")
    parser.add_argument("--adapter-path", type=str, required=True)
    parser.add_argument("--eval-jsonl", type=str, required=True)
    parser.add_argument("--output", type=str, required=True)

    parser.add_argument("--repo-root", type=str, default=".")
    parser.add_argument("--video-root", type=str, default=None)
    parser.add_argument("--default-video-path", type=str, default=None)
    parser.add_argument("--video-path-map", type=str, default=None)

    parser.add_argument("--window-size", type=int, default=8)
    parser.add_argument("--frame-size", type=int, default=336)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--max-new-tokens", type=int, default=128)

    parser.add_argument("--fp16", action="store_true")
    parser.add_argument("--bf16", action="store_true")
    parser.add_argument("--cpu", action="store_true", help="Force CPU inference.")

    return parser.parse_args()


def main():
    args = parse_args()

    if args.bf16 and args.fp16:
        raise ValueError("Use only one precision flag: --bf16 or --fp16, not both.")

    if args.cpu:
        device = "cpu"
    else:
        if not torch.cuda.is_available():
            raise RuntimeError(
                "CUDA is not available. This usually means you are using the wrong "
                "Python environment or a PyTorch build that does not match the GPU "
                "driver. Activate the GPU environment first, e.g. "
                "`conda activate azureml_py38`, or pass --cpu explicitly."
            )
        device = "cuda"

    if args.bf16:
        dtype = torch.bfloat16
    elif args.fp16:
        dtype = torch.float16
    else:
        dtype = torch.float32

    print(f"[infer-lora] device={device} dtype={dtype}")
    print(f"[infer-lora] base model={args.model_name}")
    print(f"[infer-lora] adapter={args.adapter_path}")

    processor, tokenizer = load_processor(args.adapter_path, args.model_name)

    print(f"[infer-lora] loading base model: {args.model_name}")
    base_model = load_qwen_vl_base_model(args.model_name, dtype=dtype, device=device)

    # Required because training added <WAIT>/<ANSWER> to tokenizer.
    if base_model.get_input_embeddings().num_embeddings != len(tokenizer):
        print(
            "[infer-lora] resizing token embeddings: "
            f"{base_model.get_input_embeddings().num_embeddings} -> {len(tokenizer)}"
        )
        base_model.resize_token_embeddings(len(tokenizer))

    print(f"[infer-lora] loading LoRA adapter: {args.adapter_path}")
    model = PeftModel.from_pretrained(base_model, args.adapter_path)
    model.to(device)
    model.eval()

    dataset = QATrainingDataset(
        args.eval_jsonl,
        repo_root=args.repo_root,
        video_root=args.video_root,
        default_video_path=args.default_video_path,
        video_path_map=args.video_path_map,
        window_size=args.window_size,
        frame_size=args.frame_size,
        limit=args.limit,
    )

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    print(f"[infer-lora] samples={len(dataset)} output={out_path}")

    n_correct = 0
    n_total = 0
    label_counts: Dict[str, int] = {}

    with out_path.open("w", encoding="utf-8") as f:
        for idx in range(len(dataset)):
            sample = dataset[idx]
            print(f"[infer-lora] {idx + 1}/{len(dataset)} {sample.get('sample_type')} {sample.get('qa_type')}")

            try:
                pred = model_generate_one(
                    model,
                    processor,
                    frames=sample["frames"],
                    question=sample["question"],
                    device=device,
                    max_new_tokens=args.max_new_tokens,
                )
                rec = make_prediction_record(idx, sample, pred)
            except Exception as e:
                rec = {
                    "idx": idx,
                    "sample_type": sample.get("sample_type"),
                    "qa_type": sample.get("qa_type"),
                    "video_id": sample.get("video_id"),
                    "clip_idx": sample.get("clip_idx"),
                    "video_window": sample.get("video_window"),
                    "question": sample.get("question"),
                    "target": sample.get("target"),
                    "prediction": "",
                    "gt_label": gt_label_from_target(sample.get("target", "")),
                    "pred_label": "ERROR",
                    "correct_answerability": False,
                    "error": f"{type(e).__name__}: {e}",
                }
                print(f"[infer-lora] ERROR sample {idx}: {rec['error']}")

            print_record_summary(rec)

            n_total += 1
            n_correct += int(bool(rec.get("correct_answerability")))
            pred_label = rec.get("pred_label", "OTHER")
            label_counts[pred_label] = label_counts.get(pred_label, 0) + 1

            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
            f.flush()

    acc = n_correct / n_total if n_total else 0.0
    print("=" * 80)
    print(f"[infer-lora] wrote {out_path}")
    print(f"[infer-lora] answerability accuracy: {n_correct}/{n_total} = {acc:.4f}")
    print(f"[infer-lora] pred label counts: {label_counts}")


if __name__ == "__main__":
    main()