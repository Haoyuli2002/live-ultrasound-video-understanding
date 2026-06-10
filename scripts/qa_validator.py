"""
Streaming QA Validator using Gemini 2.5 Pro (via OpenRouter)
==============================================================
Validates each streaming QA pair on a single binary criterion:

  Validation rule (applies to BOTH sonographer_intent and next_action_guidance):
    1. The QUESTION must be derivable from [SEEN] frames + ASR-up-to-query-time only.
       It must NOT reveal or reference any content that only exists in [FUTURE].
    2. The ANSWER must correspond to what is actually visible in [SEEN]+[FUTURE]
       (i.e. the real intent / real next action shown in the clip), NOT a
       hallucination unsupported by either set of frames.

  verdict = "pass" if BOTH conditions hold, else "fail".

Validation is a different model family from the GPT-4o generator (cross-family
sanity check). OpenRouter exposes Gemini via OpenAI-compatible API, so we
reuse the openai SDK.

Usage:
    export OPENROUTER_API_KEY="sk-or-..."
    python scripts/qa_validator.py --streaming-qa results/qa/ID_streaming_qa.json --video path.mp4
"""

import os
import sys
import json
import time
import re
import base64
import argparse
from pathlib import Path

# Auto-load .env from project root
sys.path.insert(0, str(Path(__file__).resolve().parent))
import _env_loader  # noqa: F401

import cv2
import numpy as np


# ============================================================================
# Configuration
# ============================================================================

OPENROUTER_API_KEY = os.environ.get("OPENROUTER_API_KEY", "")
OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"
VALIDATOR_MODEL = "google/gemini-2.5-pro"

FRAMES_SEEN = 3
FRAMES_FUTURE = 3
FRAME_SIZE = (512, 512)


VALIDATOR_PROMPT_TEMPLATE = """You are a strict validator for a streaming ultrasound QA benchmark.

A QA pair was generated for a streaming scenario at query time t={query_time:.0f}s
within a clip that runs from {clip_start:.0f}s to {clip_end:.0f}s.

Question: "{question}"
Answer:   "{answer}"
Type:     {qa_type}    (sonographer_intent or next_action_guidance)

I am showing you TWO sets of frames in temporal order:
[SEEN] frames   ({clip_start:.0f}s -> {query_time:.0f}s) — content the learner had access to when the question was asked.
[FUTURE] frames ({query_time:.0f}s -> {clip_end:.0f}s)   — content AFTER the question, which the learner had NOT seen.

Validation criteria — BOTH must hold for a "pass":

(1) QUESTION GROUNDING:
    The QUESTION must be writable using ONLY [SEEN] frames (and any ASR up to t={query_time:.0f}s).
    It must NOT reveal, reference, hint at, or assume any content that only appears in [FUTURE].

(2) ANSWER FAITHFULNESS:
    The ANSWER must correspond to what is actually visible across [SEEN] + [FUTURE]
    (i.e. it describes the real intent or the real next action that ACTUALLY happens in the clip).
    It must NOT be a hallucination unsupported by what these frames show.

Verdict:
  "pass" — both (1) question grounding AND (2) answer faithfulness hold.
  "fail" — either the question leaks future content, OR the answer is unsupported by the frames.

Output STRICTLY this JSON object (no markdown fences, no extra text):
{{
  "verdict": "pass" | "fail",
  "reason": "<one or two sentences. If fail, identify which criterion failed and cite which set of frames the issue points to.>"
}}"""


# ============================================================================
# Frame Helpers
# ============================================================================

def extract_frames(video_path, start_sec, end_sec, num_frames):
    """Extract evenly spaced frames in [start_sec, end_sec]."""
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise ValueError(f"Cannot open video: {video_path}")
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    sf = int(start_sec * fps)
    ef = max(int(end_sec * fps), sf + 1)
    indices = np.linspace(sf, ef, num_frames, dtype=int)
    frames = []
    for idx in indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(idx))
        ret, frame = cap.read()
        if ret:
            frame = cv2.resize(frame, FRAME_SIZE)
            frames.append(frame)
    cap.release()
    return frames


def frame_to_base64(frame):
    _, buf = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 85])
    return base64.b64encode(buf).decode('utf-8')


def _parse_json_object(raw):
    """Robust JSON object extraction.

    Handles three failure modes commonly seen with Gemini 2.5 Pro:
      (a) plain JSON
      (b) ```json ... ``` fenced block
      (c) truncated output (closing brace missing) — recover by extracting
          a `verdict` field via regex and a partial `reason`.
    """
    if raw is None:
        return None

    # (a) plain JSON
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        pass

    # (b) ```json ... ``` fenced
    m = re.search(r'```(?:json)?\s*(\{.+?\})\s*```', raw, re.DOTALL)
    if m:
        try:
            return json.loads(m.group(1))
        except json.JSONDecodeError:
            pass

    # (c) any complete-looking object
    m = re.search(r'\{.*?\}', raw, re.DOTALL)
    if m:
        try:
            return json.loads(m.group(0))
        except json.JSONDecodeError:
            pass

    # (d) truncated output: try to extract verdict + (partial) reason via regex
    verdict_match = re.search(r'"verdict"\s*:\s*"(pass|fail)"', raw)
    reason_match = re.search(r'"reason"\s*:\s*"([^"]*)', raw)  # may be unterminated
    if verdict_match:
        return {
            "verdict": verdict_match.group(1),
            "reason": (reason_match.group(1) if reason_match else "") + " [truncated]",
        }

    return None


# ============================================================================
# Validator (Gemini 2.5 Pro via OpenRouter)
# ============================================================================

def _build_openrouter_client(api_key=None):
    """OpenRouter exposes an OpenAI-compatible API."""
    from openai import OpenAI

    api_key = api_key or OPENROUTER_API_KEY
    if not api_key:
        raise ValueError("OPENROUTER_API_KEY not set. Export it or pass via --api-key")

    return OpenAI(
        base_url=OPENROUTER_BASE_URL,
        api_key=api_key,
        default_headers={
            "HTTP-Referer": "https://github.com/Haoyuli2002/live-ultrasound-video-understanding",
            "X-Title": "Ultrasound Streaming QA Validator",
        },
    )


def validate_streaming_qa(qa, video_path, api_key=None, model=VALIDATOR_MODEL):
    """
    Validate a single streaming QA pair using Gemini 2.5 Pro on OpenRouter.

    Returns:
        dict with keys: verdict ("pass" | "fail"), reason, validator_model.
    """
    client = _build_openrouter_client(api_key)

    clip_start = qa['clip_start']
    clip_end = qa['clip_end']
    query_time = qa['query_time']

    seen_frames = extract_frames(video_path, clip_start, query_time, FRAMES_SEEN)
    future_frames = extract_frames(video_path, query_time, clip_end, FRAMES_FUTURE)

    prompt = VALIDATOR_PROMPT_TEMPLATE.format(
        clip_start=clip_start,
        clip_end=clip_end,
        query_time=query_time,
        question=qa['question'],
        answer=qa['answer'],
        qa_type=qa['type'],
    )

    content = []
    content.append({
        "type": "text",
        "text": f"=== [SEEN] frames ({clip_start:.0f}s -> {query_time:.0f}s) ===",
    })
    for f in seen_frames:
        content.append({
            "type": "image_url",
            "image_url": {"url": f"data:image/jpeg;base64,{frame_to_base64(f)}"},
        })
    content.append({
        "type": "text",
        "text": f"=== [FUTURE] frames ({query_time:.0f}s -> {clip_end:.0f}s) ===",
    })
    for f in future_frames:
        content.append({
            "type": "image_url",
            "image_url": {"url": f"data:image/jpeg;base64,{frame_to_base64(f)}"},
        })
    content.append({"type": "text", "text": prompt})

    t0 = time.time()
    response = client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": content}],
        temperature=0.1,
        max_tokens=800,  # Gemini 2.5 Pro is a reasoning model and needs headroom for internal CoT
    )
    elapsed = time.time() - t0

    raw = response.choices[0].message.content
    tokens = response.usage.total_tokens if response.usage else 0
    print(f"      Gemini: {elapsed:.1f}s | {tokens} tok | {qa['type']} @t={query_time:.0f}s")

    parsed = _parse_json_object(raw)
    if parsed is None:
        print(f"      WARNING: Failed to parse validator output: {raw[:200]}")
        return {
            "verdict": "fail",
            "reason": "validator_parse_failure",
            "validator_model": model,
        }

    verdict = parsed.get("verdict", "fail")
    if verdict not in ("pass", "fail"):
        # normalize unknown verdicts to "fail"
        verdict = "fail"

    return {
        "verdict": verdict,
        "reason": parsed.get("reason", ""),
        "validator_model": model,
    }


# ============================================================================
# Batch Validation
# ============================================================================

def validate_streaming_qa_file(streaming_qa_path, video_path, output_path=None,
                                api_key=None, model=VALIDATOR_MODEL,
                                drop_failed=True):
    """
    Validate every QA in a streaming QA file. Adds a `validation` field per QA.

    Args:
        streaming_qa_path: path to {video_id}_streaming_qa.json
        video_path:        path to the video file
        output_path:       where to write validated output (default: *_validated.json)
        api_key:           OpenRouter key
        model:             validator model id on OpenRouter
        drop_failed:       if True, remove QA with verdict=="fail" from final list

    Returns:
        merged dict (also written to disk).
    """
    streaming_qa_path = Path(streaming_qa_path)
    with open(streaming_qa_path) as f:
        data = json.load(f)

    qa_list = data['streaming_qa']
    print(f"\n{'='*70}")
    print(f"Streaming QA Validation: {data['video_id']}")
    print(f"  Input QA count: {len(qa_list)}")
    print(f"  Validator: {model} (via OpenRouter)")
    print(f"  Frame sampling: {FRAMES_SEEN} SEEN + {FRAMES_FUTURE} FUTURE")
    print(f"{'='*70}")

    stats = {"pass": 0, "fail": 0, "error": 0}

    for i, qa in enumerate(qa_list):
        print(f"\n  [{i+1}/{len(qa_list)}] clip{qa['clip_idx']} t={qa['query_time']:.0f}s "
              f"{qa['type']}")
        try:
            validation = validate_streaming_qa(qa, video_path,
                                                 api_key=api_key, model=model)
            qa['validation'] = validation
            verdict = validation.get('verdict', 'fail')
            stats[verdict] = stats.get(verdict, 0) + 1
            print(f"      verdict={verdict} | reason: {validation['reason'][:140]}")
        except Exception as e:
            print(f"      ERROR: {e}")
            qa['validation'] = {
                "verdict": "fail",
                "reason": f"exception: {e}",
                "validator_model": model,
            }
            stats["error"] += 1

        time.sleep(0.5)  # Rate limiting

    if drop_failed:
        kept = [q for q in qa_list if q.get('validation', {}).get('verdict') == 'pass']
    else:
        kept = qa_list

    output = {
        **data,
        'validator_model': model,
        'validation_stats': stats,
        'num_after_validation': len(kept),
        'streaming_qa': kept,
    }

    if output_path is None:
        output_path = streaming_qa_path.parent / (
            streaming_qa_path.stem.replace('_streaming_qa', '') + '_streaming_qa_validated.json'
        )
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(output, f, ensure_ascii=False, indent=2)

    print(f"\n{'='*70}")
    print(f"VALIDATION SUMMARY")
    print(f"{'='*70}")
    for k in ("pass", "fail", "error"):
        print(f"  {k:6s}: {stats.get(k, 0)}")
    print(f"  Final kept: {len(kept)}/{len(qa_list)} "
          f"(drop_failed={drop_failed})")
    print(f"  Saved: {output_path}")

    return output


# ============================================================================
# CLI
# ============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Streaming QA Validator (Gemini 2.5 Pro via OpenRouter)"
    )
    parser.add_argument("--streaming-qa", type=str, required=True,
                        help="Path to {video_id}_streaming_qa.json")
    parser.add_argument("--video", type=str, required=True,
                        help="Path to the underlying video file")
    parser.add_argument("--output", type=str, default=None,
                        help="Output path (default: same dir, *_validated.json)")
    parser.add_argument("--api-key", type=str, default=None,
                        help="OpenRouter API key (or set OPENROUTER_API_KEY env)")
    parser.add_argument("--model", type=str, default=VALIDATOR_MODEL,
                        help=f"Validator model id on OpenRouter (default: {VALIDATOR_MODEL})")
    parser.add_argument("--keep-failed", action="store_true",
                        help="Keep verdict='fail' QA in output (default: drop)")
    args = parser.parse_args()

    validate_streaming_qa_file(
        args.streaming_qa,
        args.video,
        output_path=args.output,
        api_key=args.api_key,
        model=args.model,
        drop_failed=not args.keep_failed,
    )


if __name__ == "__main__":
    main()
