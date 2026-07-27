#!/usr/bin/env python3
"""
Inference for the ultrasound ASR caption-completion pretraining checkpoint.

Loads base Qwen-VL + LoRA adapter (from pretrain/train.py) and, given recent
frames (and optional prev_context), generates the continued narration. Useful
to eyeball whether pretraining taught ultrasound-relevant descriptions.

Example:

python pretrain/infer.py \
  --model-name Qwen/Qwen3-VL-2B-Instruct \
  --adapter-path /mnt/cache/qwenFT/pretrain_qwen3vl_bf16 \
  --eval-jsonl pretrain/data/pretrain_samples.jsonl \
  --video-path-map pretrain/data/video_path_map.json \
  --output /mnt/cache/qwenFT/pretrain_predictions_limit20.jsonl \
  --window-size 4 \
  --frame-size 224 \
  --limit 20 \
  --max-new-tokens 128 \
  --bf16
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


_HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE))

from dataset import PretrainCaptionDataset  # noqa: E402
from collator import DEFAULT_SYSTEM_PROMPT, build_messages  # noqa: E402


def _from_pretrained_with_dtype(model_cls, model_name, *, dtype, **kwargs):
    try:
        return model_cls.from_pretrained(model_name, dtype=dtype, **kwargs)
    except TypeError:
        return model_cls.from_pretrained(model_name, torch_dtype=dtype, **kwargs)


def load_base_model(model_name, dtype, device):
    kwargs = {"trust_remote_code": True, "low_cpu_mem_usage": True}
    errors = []
    if AutoModelForImageTextToText is not None:
        try:
            return _from_pretrained_with_dtype(AutoModelForImageTextToText, model_name, dtype=dtype, **kwargs).to(device)
        except Exception as e:
            errors.append(("AutoModelForImageTextToText", repr(e)))
    if AutoModelForVision2Seq is not None:
        try:
            return _from_pretrained_with_dtype(AutoModelForVision2Seq, model_name, dtype=dtype, **kwargs).to(device)
        except Exception as e:
            errors.append(("AutoModelForVision2Seq", repr(e)))
    from transformers import AutoModelForCausalLM
    try:
        return _from_pretrained_with_dtype(AutoModelForCausalLM, model_name, dtype=dtype, **kwargs).to(device)
    except Exception as e:
        errors.append(("AutoModelForCausalLM", repr(e)))
    msg = "\n".join(f"{n}: {err}" for n, err in errors)
    raise RuntimeError(f"Could not load model {model_name}:\n{msg}")


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


def generate_one(model, processor, *, frames, prev_context, device, max_new_tokens=128):
    messages = build_messages(
        frames=frames,
        prev_context=prev_context,
        target=None,
        system_prompt=DEFAULT_SYSTEM_PROMPT,
    )
    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    image_inputs, video_inputs = process_vision(messages)

    kwargs = {"text": [text], "padding": True, "return_tensors": "pt"}
    if image_inputs:
        kwargs["images"] = image_inputs
    if video_inputs:
        kwargs["videos"] = video_inputs

    inputs = processor(**kwargs)
    inputs = {k: (v.to(device) if torch.is_tensor(v) else v) for k, v in inputs.items()}

    with torch.no_grad():
        generated = model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=False)

    input_len = inputs["input_ids"].shape[1]
    new_tokens = generated[:, input_len:]
    decoded = processor.batch_decode(new_tokens, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]
    return decoded.strip()


def parse_args():
    parser = argparse.ArgumentParser(description="Pretrain (ASR caption) LoRA inference")
    parser.add_argument("--model-name", type=str, default="Qwen/Qwen3-VL-2B-Instruct")
    parser.add_argument("--adapter-path", type=str, default=None,
                        help="LoRA adapter dir from pretrain/train.py. Omit (or use "
                             "--no-adapter) to run the raw base model as a baseline.")
    parser.add_argument("--no-adapter", action="store_true",
                        help="Run the untrained base model only (ignore --adapter-path). "
                             "Useful to compare base vs pretrained outputs.")
    parser.add_argument("--eval-jsonl", type=str, required=True)
    parser.add_argument("--output", type=str, required=True)

    parser.add_argument("--repo-root", type=str, default=".")
    parser.add_argument("--video-root", type=str, default=None)
    parser.add_argument("--default-video-path", type=str, default=None)
    parser.add_argument("--video-path-map", type=str, default=None)

    parser.add_argument("--window-size", type=int, default=4)
    parser.add_argument("--frame-size", type=int, default=224)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--max-new-tokens", type=int, default=128)

    parser.add_argument("--bf16", action="store_true")
    parser.add_argument("--fp16", action="store_true")
    parser.add_argument("--cpu", action="store_true")
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
                "CUDA is not available. Activate the GPU env (e.g. conda activate "
                "azureml_py38) or pass --cpu."
            )
        device = "cuda"

    dtype = torch.bfloat16 if args.bf16 else (torch.float16 if args.fp16 else torch.float32)

    use_adapter = bool(args.adapter_path) and not args.no_adapter

    print(f"[pretrain-infer] device={device} dtype={dtype}")
    print(f"[pretrain-infer] mode={'base+adapter' if use_adapter else 'BASE ONLY (untrained baseline)'}")

    # Processor: prefer the adapter dir (it may carry an updated tokenizer),
    # otherwise fall back to the base model.
    processor = None
    if use_adapter:
        try:
            processor = AutoProcessor.from_pretrained(args.adapter_path, trust_remote_code=True)
        except Exception:
            processor = None
    if processor is None:
        processor = AutoProcessor.from_pretrained(args.model_name, trust_remote_code=True)

    print(f"[pretrain-infer] loading base model: {args.model_name}")
    base_model = load_base_model(args.model_name, dtype=dtype, device=device)

    if use_adapter:
        print(f"[pretrain-infer] loading LoRA adapter: {args.adapter_path}")
        model = PeftModel.from_pretrained(base_model, args.adapter_path)
    else:
        print("[pretrain-infer] no adapter: running raw base model")
        model = base_model
    model.to(device)
    model.eval()

    dataset = PretrainCaptionDataset(
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

    print(f"[pretrain-infer] samples={len(dataset)} output={out_path}")

    with out_path.open("w", encoding="utf-8") as f:
        for idx in range(len(dataset)):
            sample = dataset[idx]
            prev_context = sample.get("prev_context", "")
            target = sample.get("target", "")
            try:
                pred = generate_one(
                    model,
                    processor,
                    frames=sample["frames"],
                    prev_context=prev_context,
                    device=device,
                    max_new_tokens=args.max_new_tokens,
                )
            except Exception as e:
                pred = ""
                print(f"[pretrain-infer] ERROR sample {idx}: {type(e).__name__}: {e}")

            rec: Dict[str, Any] = {
                "idx": idx,
                "video_id": sample.get("video_id"),
                "video_window": sample.get("video_window"),
                "prev_context": prev_context,
                "target": target,
                "prediction": pred,
                "meta": sample.get("meta", {}),
            }

            print("-" * 80)
            print(f"[pretrain-infer] idx={idx} video_id={rec['video_id']} window={rec['video_window']}")
            print(f"[pretrain-infer] prev_context: {prev_context[:120]}")
            print(f"[pretrain-infer] target: {target}")
            print(f"[pretrain-infer] prediction: {pred}")

            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
            f.flush()

    print("=" * 80)
    print(f"[pretrain-infer] wrote {out_path}")


if __name__ == "__main__":
    main()