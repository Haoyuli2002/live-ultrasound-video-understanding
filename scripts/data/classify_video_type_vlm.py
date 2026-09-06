#!/usr/bin/env python3
"""Classify ASR-kept videos into visual content types with a VLM.

This is intended to run *after* ASR rule filtering, on maps such as
``train_full295_asr_keep_videos.json``.  It samples a small number of frames per
video, sends them to an OpenAI-compatible vision model, and writes one JSONL row
per video.

Labels:
  A. ultrasound_cine
  B. ultrasound_teaching_with_scan
  C. ultrasound_ppt_teaching
  D. mixed_screen_recording
  E. non_ultrasound_or_irrelevant
  F. uncertain

Backend: an OpenAI-compatible vision endpoint. The default target is a local
Qwen3-VL server served by vLLM on an H100, e.g.:

  vllm serve Qwen/Qwen3-VL-30B-A3B-Instruct \
    --port 8000 \
    --limit-mm-per-prompt image=16

Then classify (default points at that local server):

  python scripts/data/classify_video_type_vlm.py \
    --video-map cluster_data/splits/train_full295_asr_keep_videos.json \
    --output cluster_data/splits/train_full295_asr_vlm_video_type.jsonl \
    --n-frames 12 \
    --frame-size 224 \
    --model Qwen/Qwen3-VL-30B-A3B-Instruct \
    --base-url http://localhost:8000/v1 \
    --api-key-env VLLM_API_KEY

Qwen3-VL-30B-A3B is a MoE VLM (~30B total, ~3B activated), which fits well for
H100 inference. Note: the text-only Qwen3-30B-A3B cannot see frames, so a
vision-language model (Qwen3-VL) is required for video-type classification.
"""

from __future__ import annotations

import argparse
import base64
import io
import json
import os
import re
import time
from pathlib import Path
from typing import Any, Dict, List

import cv2
from PIL import Image


LABELS = {
    "ultrasound_cine",
    "ultrasound_teaching_with_scan",
    "ultrasound_ppt_teaching",
    "mixed_screen_recording",
    "non_ultrasound_or_irrelevant",
    "uncertain",
}

PROMPT = """You are classifying videos for an ultrasound video understanding dataset.

You will see uniformly sampled frames from one video. Classify the VIDEO into exactly one label:

A. ultrasound_cine: pure real-time ultrasound scan / ultrasound cine loop. Mostly dynamic ultrasound image area, minimal slides/text.
B. ultrasound_teaching_with_scan: ultrasound teaching video with substantial real ultrasound dynamic scan footage plus narration/teaching.
C. ultrasound_ppt_teaching: ultrasound PPT / slide-based lecture / text-heavy teaching material / mostly static slides or screenshots.
D. mixed_screen_recording: mixed screen recording, including web pages, software UI, PPT, and only some ultrasound images.
E. non_ultrasound_or_irrelevant: not ultrasound or clearly irrelevant; little/no ultrasound visual content.
F. uncertain: cannot determine confidently.

Return JSON only with this schema:
{
  "label": "ultrasound_cine | ultrasound_teaching_with_scan | ultrasound_ppt_teaching | mixed_screen_recording | non_ultrasound_or_irrelevant | uncertain",
  "confidence": 0.0,
  "visual_evidence": "brief evidence from the frames",
  "keep_for_pretrain": true,
  "keep_for_compression": true,
  "keep_for_sft": true
}

Default keep policy:
- ultrasound_cine: keep for all stages
- ultrasound_teaching_with_scan: keep for all stages
- ultrasound_ppt_teaching: keep_for_pretrain=true, keep_for_compression=false, keep_for_sft=false
- mixed_screen_recording: keep_for_pretrain=true, keep_for_compression=false, keep_for_sft=false
- non_ultrasound_or_irrelevant: drop for all stages
- uncertain: keep_for_pretrain=true, keep_for_compression=false, keep_for_sft=false for audit/conservative retention
"""


def load_json(path: Path) -> Any:
    with path.open(encoding="utf-8") as f:
        return json.load(f)


def resolve_path(path: str, repo_root: Path) -> Path:
    p = Path(path)
    return p if p.is_absolute() else repo_root / p


def resize_with_aspect_ratio_and_pad(img: Image.Image, size: int | None) -> Image.Image:
    if size is None:
        return img.convert("RGB")
    img = img.convert("RGB")
    w, h = img.size
    scale = min(float(size) / max(w, 1), float(size) / max(h, 1))
    new_w = max(1, int(round(w * scale)))
    new_h = max(1, int(round(h * scale)))
    resized = img.resize((new_w, new_h), Image.BICUBIC)
    canvas = Image.new("RGB", (size, size), (0, 0, 0))
    canvas.paste(resized, ((size - new_w) // 2, (size - new_h) // 2))
    return canvas


def sample_frames(video_path: Path, n_frames: int, frame_size: int | None) -> tuple[list[Image.Image], dict[str, Any]]:
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {video_path}")
    try:
        fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
        duration = frame_count / fps if frame_count > 0 else 0.0
        if frame_count <= 0:
            indices = [0]
        else:
            start = int(frame_count * 0.05)
            end = max(start, int(frame_count * 0.95) - 1)
            if n_frames == 1:
                indices = [(start + end) // 2]
            else:
                indices = [int(round(start + i * (end - start) / (n_frames - 1))) for i in range(n_frames)]
        frames: list[Image.Image] = []
        for idx in indices:
            cap.set(cv2.CAP_PROP_POS_FRAMES, max(0, idx))
            ok, frame = cap.read()
            if not ok:
                continue
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frames.append(resize_with_aspect_ratio_and_pad(Image.fromarray(rgb), frame_size))
    finally:
        cap.release()
    if not frames:
        raise RuntimeError(f"Could not read frames from {video_path}")
    return frames, {"duration_sec": duration, "fps": fps, "frame_count": frame_count}


def image_to_data_url(img: Image.Image) -> str:
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=85)
    return "data:image/jpeg;base64," + base64.b64encode(buf.getvalue()).decode("ascii")


def parse_json(text: str) -> Dict[str, Any]:
    text = (text or "").strip()
    try:
        return json.loads(text)
    except Exception:
        m = re.search(r"\{.*\}", text, flags=re.S)
        if m:
            return json.loads(m.group(0))
    raise ValueError(f"Could not parse JSON from VLM output: {text[:500]!r}")


def apply_default_keep_policy(label: str, rec: Dict[str, Any]) -> None:
    if label in {"ultrasound_cine", "ultrasound_teaching_with_scan"}:
        defaults = (True, True, True)
    elif label in {"ultrasound_ppt_teaching", "mixed_screen_recording", "uncertain"}:
        defaults = (True, False, False)
    else:
        defaults = (False, False, False)
    rec.setdefault("keep_for_pretrain", defaults[0])
    rec.setdefault("keep_for_compression", defaults[1])
    rec.setdefault("keep_for_sft", defaults[2])


def classify_one(client, model: str, frames: list[Image.Image], max_tokens: int) -> Dict[str, Any]:
    content: list[dict[str, Any]] = [{"type": "text", "text": PROMPT}]
    for i, img in enumerate(frames, 1):
        content.append({"type": "text", "text": f"Frame {i}:"})
        content.append({"type": "image_url", "image_url": {"url": image_to_data_url(img), "detail": "low"}})
    response = client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": content}],
        temperature=0,
        max_tokens=max_tokens,
    )
    raw = response.choices[0].message.content or ""
    rec = parse_json(raw)
    rec["_raw_output"] = raw
    return rec


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="VLM classify ASR-kept videos by visual content type")
    p.add_argument("--video-map", type=Path, required=True, help="JSON map: video_id -> video_path, typically ASR-keep map")
    p.add_argument("--output", type=Path, required=True, help="Output JSONL audit/classification file")
    p.add_argument("--repo-root", type=Path, default=Path("."))
    p.add_argument("--model", default="Qwen/Qwen3-VL-30B-A3B-Instruct",
                   help="Vision model id served by the OpenAI-compatible endpoint")
    p.add_argument("--base-url", default="http://localhost:8000/v1",
                   help="OpenAI-compatible base URL (default: local vLLM Qwen3-VL server)")
    p.add_argument("--api-key-env", default="VLLM_API_KEY",
                   help="Env var holding the API key; local vLLM usually accepts any/empty value")
    p.add_argument("--n-frames", type=int, default=12)
    p.add_argument("--frame-size", type=int, default=224)
    p.add_argument("--max-tokens", type=int, default=512)
    p.add_argument("--limit", type=int, default=None)
    p.add_argument("--sleep-sec", type=float, default=0.0)
    p.add_argument("--resume", action="store_true", help="Skip video_ids already present in output")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    from openai import OpenAI

    # Local vLLM servers accept any key; fall back to a dummy so the OpenAI
    # client is happy. For hosted APIs (OpenAI/DashScope/OpenRouter) set the env.
    api_key = os.environ.get(args.api_key_env) or "EMPTY"
    client = OpenAI(api_key=api_key, base_url=args.base_url) if args.base_url else OpenAI(api_key=api_key)

    video_map = {str(k): str(v) for k, v in load_json(args.video_map).items()}
    items = sorted(video_map.items())
    if args.limit is not None:
        items = items[:args.limit]

    done: set[str] = set()
    if args.resume and args.output.exists():
        with args.output.open(encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    done.add(str(json.loads(line).get("video_id")))

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("a" if args.resume else "w", encoding="utf-8") as out:
        for idx, (video_id, raw_path) in enumerate(items, 1):
            if video_id in done:
                continue
            video_path = resolve_path(raw_path, args.repo_root)
            print(f"[{idx}/{len(items)}] {video_id} {video_path}")
            base = {"video_id": video_id, "video_path": str(video_path), "source_video_map": str(args.video_map)}
            try:
                frames, info = sample_frames(video_path, args.n_frames, args.frame_size)
                rec = classify_one(client, args.model, frames, args.max_tokens)
                label = str(rec.get("label") or "uncertain")
                if label not in LABELS:
                    label = "uncertain"
                rec["label"] = label
                rec["confidence"] = float(rec.get("confidence") or 0.0)
                apply_default_keep_policy(label, rec)
                rec.update(base)
                rec.update(info)
                rec["error"] = None
            except Exception as exc:  # noqa: BLE001 - audit all failures.
                rec = {
                    **base,
                    "label": "uncertain",
                    "confidence": 0.0,
                    "visual_evidence": "classification_failed",
                    "keep_for_pretrain": True,
                    "keep_for_compression": False,
                    "keep_for_sft": False,
                    "error": f"{type(exc).__name__}: {exc}",
                }
                print(f"  ERROR: {rec['error']}")
            out.write(json.dumps(rec, ensure_ascii=False) + "\n")
            out.flush()
            if args.sleep_sec > 0:
                time.sleep(args.sleep_sec)


if __name__ == "__main__":
    main()