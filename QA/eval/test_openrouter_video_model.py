"""
Smoke-test OpenRouter multimodal/video model support.

Purpose
-------
Given a short mp4 clip, send it to one or more OpenRouter models using the
same `type: "file"` video block used by the QA generator / validator, then
print whether the provider reports non-zero `video_tokens` / `audio_tokens`.

This is useful before switching QA generation/validation from a known working
model (e.g. google/gemini-2.5-flash) to a new model id.

Example
-------
python QA/eval/test_openrouter_video_model.py \
  --video UltrasoundCrawler_KeyCode_20260323_v2/output/20260520_162816_youtube/media/case_reasoning/8V649L5Q368.mp4 \
  --models google/gemini-2.5-flash,google/gemini-3.1-pro-preview,openai/gpt-5.5 \
  --start 60 \
  --duration 5
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any, Dict, List

# Reuse the same OpenRouter/video helper as the QA pipeline.
_REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_REPO_ROOT / "QA"))
from _shared import (  # noqa: E402
    build_openrouter_client,
    build_video_block,
    call_with_content,
    cut_clip,
    temp_clip_path,
    text_block,
)


DEFAULT_MODELS = [
    "google/gemini-2.5-flash",
    "google/gemini-3.1-pro-preview",
    "openai/gpt-5.5",
]


PROMPT = """You are testing whether this model can understand an uploaded mp4 video clip with visual frames and audio.

Inspect the clip and output STRICTLY this JSON object:
{
  "visible_content": "<briefly describe what is visible>",
  "audio_or_narration": "<briefly describe any narration/audio if audible; otherwise say 'not clear'>",
  "is_ultrasound_related": true | false
}

Do not output markdown fences. Do not add any extra text."""


def _parse_models(args) -> List[str]:
    if args.models:
        return [m.strip() for m in args.models.split(",") if m.strip()]
    if args.model:
        return [args.model]
    return list(DEFAULT_MODELS)


def _usage_summary(usage: Dict[str, Any]) -> Dict[str, Any]:
    details = usage.get("prompt_tokens_details") or {}
    return {
        "total_tokens": usage.get("total_tokens"),
        "prompt_tokens": usage.get("prompt_tokens"),
        "completion_tokens": usage.get("completion_tokens"),
        "video_tokens": details.get("video_tokens", 0) or 0,
        "audio_tokens": details.get("audio_tokens", 0) or 0,
        "cost": usage.get("cost"),
    }


def _status_from_usage(usage_summary: Dict[str, Any]) -> str:
    if (usage_summary.get("video_tokens") or 0) > 0:
        return "video_supported"
    return "no_video_tokens_reported"


def test_one_model(
    *,
    client,
    model: str,
    video_path: Path,
    clip_path: Path,
    label: str,
    retries: int,
) -> Dict[str, Any]:
    content_blocks = [
        text_block(f"=== [TEST_VIDEO] {label} ==="),
        build_video_block(clip_path, label="test_clip.mp4"),
        text_block(PROMPT),
    ]

    t0 = time.time()
    raw, usage = call_with_content(
        client,
        content_blocks=content_blocks,
        model=model,
        temperature=0.0,
        retries=retries,
    )
    elapsed = time.time() - t0
    summary = _usage_summary(usage)

    result = {
        "model": model,
        "status": _status_from_usage(summary),
        "elapsed_sec": round(elapsed, 2),
        "usage": summary,
        "response_preview": (raw or "")[:1000],
        "raw_response": raw,
        "source_video": str(video_path),
        "test_clip": str(clip_path),
    }
    return result


def print_result(result: Dict[str, Any]) -> None:
    print("\n" + "=" * 80)
    print(f"MODEL: {result['model']}")
    print("=" * 80)
    print(f"status       : {result.get('status')}")

    if result.get("status") == "error":
        print(f"error        : {result.get('error', 'unknown error')}")
        print(f"test_clip    : {result.get('test_clip')}")
        return

    print(f"elapsed_sec  : {result.get('elapsed_sec')}")
    usage = result.get("usage") or {}
    print(f"total_tokens : {usage.get('total_tokens')}")
    print(f"video_tokens : {usage.get('video_tokens')}")
    print(f"audio_tokens : {usage.get('audio_tokens')}")
    cost = usage.get("cost")
    print(f"cost         : ${cost:.6f}" if cost is not None else "cost         : n/a")
    print("response:")
    print(result.get("response_preview") or "<EMPTY>")


def parse_args():
    parser = argparse.ArgumentParser(description="Smoke-test OpenRouter mp4/video model support")
    parser.add_argument("--video", type=str, required=True, help="Source mp4 video")
    parser.add_argument("--model", type=str, default=None, help="Single OpenRouter model id")
    parser.add_argument("--models", type=str, default=None,
                        help="Comma-separated OpenRouter model ids")
    parser.add_argument("--start", type=float, default=60.0, help="Start second for test clip")
    parser.add_argument("--duration", type=float, default=5.0, help="Duration seconds for test clip")
    parser.add_argument("--api-key", type=str, default=None)
    parser.add_argument("--output", type=str, default=None, help="Optional JSONL output path")
    parser.add_argument("--retries", type=int, default=1,
                        help="Retries per model. Keep low for smoke tests.")
    return parser.parse_args()


def main():
    args = parse_args()
    video_path = Path(args.video)
    if not video_path.exists():
        raise FileNotFoundError(f"Video not found: {video_path}")

    models = _parse_models(args)
    client = build_openrouter_client(args.api_key)

    video_id = video_path.stem
    end = args.start + args.duration
    clip_path = temp_clip_path(
        video_id,
        f"openrouter_smoke_{int(args.start)}_{int(args.duration)}s",
    )

    print(f"[smoke] Cutting test clip: {args.start:.2f}s -> {end:.2f}s")
    cut_clip(video_path, args.start, end, clip_path)
    print(f"[smoke] Test clip: {clip_path} ({clip_path.stat().st_size / 1024:.1f} KB)")
    print(f"[smoke] Models: {models}")

    out_f = None
    if args.output:
        out_path = Path(args.output)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_f = out_path.open("w", encoding="utf-8")

    try:
        for model in models:
            try:
                result = test_one_model(
                    client=client,
                    model=model,
                    video_path=video_path,
                    clip_path=clip_path,
                    label=f"{args.start:.1f}s->{end:.1f}s",
                    retries=args.retries,
                )
            except Exception as e:
                result = {
                    "model": model,
                    "status": "error",
                    "error": f"{type(e).__name__}: {e}",
                    "source_video": str(video_path),
                    "test_clip": str(clip_path),
                }

            print_result(result)
            if out_f:
                out_f.write(json.dumps(result, ensure_ascii=False) + "\n")
                out_f.flush()
    finally:
        if out_f:
            out_f.close()


if __name__ == "__main__":
    main()