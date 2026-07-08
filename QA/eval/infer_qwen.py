"""
Raw Qwen-VL inference for answerability-aware QA evaluation.

Goal:
  Use an unfinetuned base model (e.g. Qwen/Qwen3-VL-2B-Instruct) to generate
  predictions on QA training/eval samples, then check whether it outputs WAIT
  where it should wait and ANSWER where evidence is sufficient.

Input:
  QA/results/{video_id}_training_samples.jsonl

Output:
  JSONL with one prediction per sample.

Example:
  export TRANSFORMERS_NO_TF=1
  export USE_TF=0
  export USE_FLAX=0

  python QA/eval/infer_qwen.py \
    --model-name Qwen/Qwen3-VL-2B-Instruct \
    --eval-jsonl azure_data/QA/results/8V649L5Q368_training_samples.jsonl \
    --default-video-path azure_data/videos/8V649L5Q368.mp4 \
    --output QA/eval/results/qwen3vl_2b_raw_predictions_limit10.jsonl \
    --window-size 8 \
    --frame-size 336 \
    --limit 10 \
    --fp16
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
from pathlib import Path
from typing import Any, Dict, List

import torch
from transformers import AutoProcessor

try:
    from transformers import AutoModelForVision2Seq
except Exception:
    AutoModelForVision2Seq = None

try:
    from transformers import AutoModelForImageTextToText
except Exception:
    AutoModelForImageTextToText = None

# Import shared QA training utilities.
_HERE = Path(__file__).resolve().parent
_REPO_ROOT = _HERE.parent.parent
_TRAIN_DIR = _HERE.parent / "train"
sys.path.insert(0, str(_TRAIN_DIR))

from dataset import QATrainingDataset  # noqa: E402
from collator import DEFAULT_SYSTEM_PROMPT, build_messages  # noqa: E402


def load_qwen_vl_model(model_name: str, torch_dtype, device: str):
    """Load Qwen-VL model using available HF Auto classes."""
    kwargs = {
        "trust_remote_code": True,
        "torch_dtype": torch_dtype,
        "low_cpu_mem_usage": True,
    }
    errors = []

    if AutoModelForImageTextToText is not None:
        try:
            model = AutoModelForImageTextToText.from_pretrained(model_name, **kwargs)
            return model.to(device)
        except Exception as e:
            errors.append(("AutoModelForImageTextToText", repr(e)))

    if AutoModelForVision2Seq is not None:
        try:
            model = AutoModelForVision2Seq.from_pretrained(model_name, **kwargs)
            return model.to(device)
        except Exception as e:
            errors.append(("AutoModelForVision2Seq", repr(e)))

    from transformers import AutoModelForCausalLM
    try:
        model = AutoModelForCausalLM.from_pretrained(model_name, **kwargs)
        return model.to(device)
    except Exception as e:
        errors.append(("AutoModelForCausalLM", repr(e)))

    msg = "\n".join(f"{name}: {err}" for name, err in errors)
    raise RuntimeError(f"Could not load model {model_name}. Tried multiple Auto classes:\n{msg}")


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

    Raw base models may not emit literal <ANSWER>. For answerability evaluation,
    any substantive non-WAIT response is treated as ANSWER.
    """
    s = (text or "").strip()
    low = s.lower()

    if not s:
        return "OTHER"

    if re.match(r"^\s*<\s*wait\s*>", low):
        return "WAIT"
    if "not enough information" in low:
        return "WAIT"
    if "need more information" in low or "need more context" in low:
        return "WAIT"
    if "cannot answer yet" in low or "can't answer yet" in low:
        return "WAIT"
    if "more video is needed" in low or "more context is needed" in low:
        return "WAIT"

    if re.match(r"^\s*<\s*answer\s*>", low):
        return "ANSWER"

    # If it is not a wait/refusal, treat it as attempting to answer.
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


def parse_args():
    parser = argparse.ArgumentParser(description="Run raw Qwen-VL inference on QA samples")
    parser.add_argument("--model-name", type=str, default="Qwen/Qwen3-VL-2B-Instruct")
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

    return parser.parse_args()


def main():
    args = parse_args()

    # Keep HF from importing TensorFlow/Flax in AzureML mixed environments.
    os.environ.setdefault("TRANSFORMERS_NO_TF", "1")
    os.environ.setdefault("USE_TF", "0")
    os.environ.setdefault("USE_FLAX", "0")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    if args.bf16:
        dtype = torch.bfloat16
    elif args.fp16:
        dtype = torch.float16
    else:
        dtype = torch.float32

    print(f"[eval] device={device} dtype={dtype}")
    print(f"[eval] loading processor: {args.model_name}")
    processor = AutoProcessor.from_pretrained(args.model_name, trust_remote_code=True)

    print(f"[eval] loading model: {args.model_name}")
    model = load_qwen_vl_model(args.model_name, torch_dtype=dtype, device=device)
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

    print(f"[eval] samples={len(dataset)} output={out_path}")

    with out_path.open("w", encoding="utf-8") as f:
        for idx in range(len(dataset)):
            sample = dataset[idx]
            print(f"[eval] {idx+1}/{len(dataset)} {sample.get('sample_type')} {sample.get('qa_type')}")
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
                print(f"[eval] ERROR sample {idx}: {rec['error']}")

            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
            f.flush()

    print(f"[eval] wrote {out_path}")


if __name__ == "__main__":
    main()