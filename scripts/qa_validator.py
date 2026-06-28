"""
Streaming QA Validator using Gemini 2.5 Flash (via OpenRouter)
==============================================================
Validates each streaming QA pair by sending two ACTUAL VIDEO CLIPS
(not sampled frames) to Gemini 2.5 Flash and asking it to verify a
single binary criterion:

  Validation rule (applies to BOTH sonographer_intent and next_action_guidance):
    1. The QUESTION must be derivable from [SEEN_VIDEO] only.
       It must NOT reveal or reference any content that exists only in
       [FUTURE_VIDEO].
    2. The ANSWER must correspond to what is actually visible/audible in
       [SEEN_VIDEO] + [FUTURE_VIDEO] (i.e. the real intent / real next
       action), NOT a hallucination unsupported by either segment.

  verdict = "pass" if BOTH conditions hold, else "fail".

Why video clips (not 6 sampled frames)?
  Verified via (see _video_llm.py) that OpenRouter
  + Gemini 2.5 Flash natively accepts mp4 via the `type: "file"`
  content block (video_tokens > 0 in the usage breakdown). This gives
  the validator full motion + audio (operator's narration), which is
  exactly what's needed to spot future-leakage and answer hallucinations.

Usage:
    # OPENROUTER_API_KEY is auto-loaded from .env
    python scripts/qa_validator.py \\
        --streaming-qa results/qa/ID_streaming_qa.json \\
        --video path/to/ID.mp4
"""

import sys
import json
import time
import re
import argparse
from pathlib import Path

# Auto-load .env + import the shared video helper
sys.path.insert(0, str(Path(__file__).resolve().parent))
import _env_loader  # noqa: F401
from _video_llm import (
    DEFAULT_MODEL,
    build_openrouter_client,
    build_video_block,
    call_with_content,
    cut_clip,
    temp_clip_path,
    text_block,
)


# ============================================================================
# Configuration
# ============================================================================

VALIDATOR_MODEL = DEFAULT_MODEL  # "google/gemini-2.5-flash"

# SEEN segment window — content the learner has watched up to query_time.
# We keep the LATTER part (closest to query_time) capped at this many seconds.
# 240s is generous: enough for the validator to judge "could this question be
# written from SEEN only?" without exploding cost.
SEEN_WINDOW_SEC = 240.0

# FUTURE segment window — content right AFTER query_time. We keep the EARLIER
# part of FUTURE (the moment the answer is grounded in). 30s is enough to
# cover the immediate next maneuver / next finding for both sonographer_intent
# and next_action_guidance, at ~5x lower token cost than the full clip.
# This also dramatically reduces 504 errors observed when both segments were
# large simultaneously.
FUTURE_WINDOW_SEC = 30.0


VALIDATOR_PROMPT_TEMPLATE = """You are a strict validator for a streaming ultrasound QA benchmark.

A QA pair was generated for a streaming scenario at query time t={query_time:.0f}s
within a clip that runs from {clip_start:.0f}s to {clip_end:.0f}s.

Question: "{question}"
Answer:   "{answer}"
Type:     {qa_type}    (sonographer_intent or next_action_guidance)

You will see TWO video segments in temporal order:
  [SEEN_VIDEO]   — content the learner had already watched (clip_start -> query_time).
  [FUTURE_VIDEO] — content AFTER the question, NOT visible to the learner (query_time -> clip_end).

Each segment includes its original visual frames AND audio (operator's narration).
Use BOTH the visuals and the spoken commentary when judging.

Validation criteria — BOTH must hold for a "pass":

(1) QUESTION GROUNDING:
    The QUESTION must be writable using ONLY [SEEN_VIDEO] (frames + narration
    audible up to t={query_time:.0f}s). It must NOT reveal, reference, hint at,
    or assume any content that only appears in [FUTURE_VIDEO] — visually OR
    audibly.

(2) ANSWER FAITHFULNESS:
    The ANSWER must correspond to what is actually shown / said across
    [SEEN_VIDEO] + [FUTURE_VIDEO] (it describes the real intent or real next
    action that ACTUALLY happens). It must NOT be a hallucination unsupported
    by what these two segments show.

Verdict:
  "pass" — BOTH (1) question grounding AND (2) answer faithfulness hold.
  "fail" — either the question leaks future content, OR the answer is
           unsupported by what the two segments show.

Output STRICTLY this JSON object (no markdown fences, no extra text):
{{
  "verdict": "pass" | "fail",
  "reason": "<MANDATORY: at least 2 full sentences (>= 30 words total). Cite SPECIFIC content visible OR audible in the segments — anatomical structures shown, probe orientation, what action is performed, what the operator says, what changes between [SEEN_VIDEO] and [FUTURE_VIDEO]. A vague reason like 'looks fine' or 'matches' is NOT acceptable.>"
}}"""


# ============================================================================
# JSON parsing (robust to fenced output / truncation)
# ============================================================================

def _parse_json_object(raw):
    """Extract a single JSON object from `raw`, handling fences, partial output."""
    if not raw:
        return None

    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        pass

    m = re.search(r'```(?:json)?\s*(\{.+?\})\s*```', raw, re.DOTALL)
    if m:
        try:
            return json.loads(m.group(1))
        except json.JSONDecodeError:
            pass

    m = re.search(r'\{.*?\}', raw, re.DOTALL)
    if m:
        try:
            return json.loads(m.group(0))
        except json.JSONDecodeError:
            pass

    # truncated output: recover at least the verdict
    verdict_match = re.search(r'"verdict"\s*:\s*"(pass|fail)"', raw)
    reason_match = re.search(r'"reason"\s*:\s*"([^"]*)', raw)
    if verdict_match:
        return {
            "verdict": verdict_match.group(1),
            "reason": (reason_match.group(1) if reason_match else "") + " [truncated]",
        }
    return None


# ============================================================================
# Per-QA validation
# ============================================================================

def _seen_window(clip_start, query_time, window_sec):
    """Return [a, b] for the SEEN segment, capped to the LATTER `window_sec`
    (the moment closest to query_time is the most informative for grounding
    judgement)."""
    a, b = clip_start, query_time
    if window_sec and (b - a) > window_sec:
        a = b - window_sec
    return a, b


def _future_window(query_time, clip_end, window_sec):
    """Return [a, b] for the FUTURE segment, capped to the EARLIER `window_sec`
    (the moment right after query_time is where the answer's ground truth
    actually plays out)."""
    a, b = query_time, clip_end
    if window_sec and (b - a) > window_sec:
        b = a + window_sec
    return a, b


def validate_streaming_qa(qa, video_path, *, video_id=None,
                           api_key=None, model=VALIDATOR_MODEL,
                           seen_window_sec=SEEN_WINDOW_SEC,
                           future_window_sec=FUTURE_WINDOW_SEC):
    """
    Validate one streaming QA pair using Gemini 2.5 Flash on OpenRouter,
    sending the actual SEEN/FUTURE video segments (not sampled frames).

    Returns dict: {verdict, reason, validator_model, usage}.
    """
    client = build_openrouter_client(api_key)

    clip_start = float(qa['clip_start'])
    clip_end = float(qa['clip_end'])
    query_time = float(qa['query_time'])
    clip_idx = qa.get('clip_idx', 0)

    if video_id is None:
        video_id = Path(video_path).stem

    seen_a, seen_b = _seen_window(clip_start, query_time, seen_window_sec)
    fut_a, fut_b = _future_window(query_time, clip_end, future_window_sec)

    # Cache file names: include effective window so that, if a user re-runs
    # with a smaller window, we don't accidentally reuse a larger cached clip.
    seen_path = temp_clip_path(
        video_id,
        f"clip{clip_idx}_t{int(query_time)}_seen_{int(seen_b - seen_a)}s",
    )
    future_path = temp_clip_path(
        video_id,
        f"clip{clip_idx}_t{int(query_time)}_future_{int(fut_b - fut_a)}s",
    )

    cut_clip(video_path, seen_a, seen_b, seen_path)
    cut_clip(video_path, fut_a, fut_b, future_path)

    prompt = VALIDATOR_PROMPT_TEMPLATE.format(
        clip_start=clip_start,
        clip_end=clip_end,
        query_time=query_time,
        question=qa['question'],
        answer=qa['answer'],
        qa_type=qa['type'],
    )

    # Interleaved layout: label, video, label, video, prompt
    content_blocks = [
        text_block(f"=== [SEEN_VIDEO]   {seen_a:.0f}s -> {seen_b:.0f}s ==="),
        build_video_block(seen_path, label="seen.mp4"),
        text_block(f"=== [FUTURE_VIDEO] {fut_a:.0f}s -> {fut_b:.0f}s ==="),
        build_video_block(future_path, label="future.mp4"),
        text_block(prompt),
    ]

    t0 = time.time()
    raw, usage = call_with_content(
        client,
        content_blocks=content_blocks,
        model=model,
        temperature=0.1,
    )
    elapsed = time.time() - t0

    pdetails = (usage.get("prompt_tokens_details") or {})
    vt = pdetails.get("video_tokens", 0)
    at = pdetails.get("audio_tokens", 0)
    cost = usage.get("cost")
    cost_s = f"${cost:.6f}" if cost is not None else "n/a"
    print(f"      Gemini: {elapsed:.1f}s | tot={usage.get('total_tokens')} "
          f"video={vt} audio={at} | cost={cost_s} | "
          f"{qa['type']} @t={query_time:.0f}s")
    raw_preview = (raw or "<EMPTY>")[:240].replace("\n", " ")
    print(f"      RAW: {raw_preview!r}")

    parsed = _parse_json_object(raw)
    if parsed is None:
        print(f"      WARNING: validator output unparseable")
        return {
            "verdict": "fail",
            "reason": "validator_parse_failure",
            "validator_model": model,
            "usage": usage,
        }

    verdict = parsed.get("verdict", "fail")
    if verdict not in ("pass", "fail"):
        verdict = "fail"

    return {
        "verdict": verdict,
        "reason": parsed.get("reason", ""),
        "validator_model": model,
        "usage": usage,
    }


# ============================================================================
# Batch validation
# ============================================================================

def validate_streaming_qa_file(streaming_qa_path, video_path, output_path=None,
                                api_key=None, model=VALIDATOR_MODEL,
                                drop_failed=True,
                                seen_window_sec=SEEN_WINDOW_SEC,
                                future_window_sec=FUTURE_WINDOW_SEC,
                                max_qa=None):
    """
    Validate every QA in a streaming-QA file. Adds a `validation` field per QA.

    Args:
        streaming_qa_path:  path to {video_id}_streaming_qa.json
        video_path:         path to the underlying video file
        output_path:        where to write validated output (default: same dir, *_validated.json)
        api_key:            OpenRouter API key (auto-loaded from .env otherwise)
        model:              validator model id on OpenRouter
        drop_failed:        if True, remove QA with verdict=="fail" from final list
        seen_window_sec:    cap on the SEEN segment length (latter portion kept).
                            None to disable.
        future_window_sec:  cap on the FUTURE segment length (earlier portion kept).
                            None to disable.
        max_qa:             if int, only validate the first N QA (debug / smoke test)

    Returns merged dict (also written to disk).
    """
    streaming_qa_path = Path(streaming_qa_path)
    with open(streaming_qa_path) as f:
        data = json.load(f)

    qa_list = data['streaming_qa']
    if max_qa is not None:
        qa_list = qa_list[:max_qa]

    print(f"\n{'='*70}")
    print(f"Streaming QA Validation: {data['video_id']}")
    print(f"  Input QA count : {len(qa_list)}"
          + (f" (truncated to {max_qa} by --max-qa)" if max_qa else ""))
    print(f"  Validator      : {model} (OpenRouter)")
    print(f"  Mode           : full video segments (SEEN + FUTURE)")
    print(f"  SEEN window    : {seen_window_sec}s   (latter portion before query_time)")
    print(f"  FUTURE window  : {future_window_sec}s (earlier portion after query_time)")
    print(f"{'='*70}")

    stats = {"pass": 0, "fail": 0, "error": 0}
    cost_total = 0.0
    video_token_total = 0

    for i, qa in enumerate(qa_list):
        print(f"\n  [{i+1}/{len(qa_list)}] clip{qa['clip_idx']} "
              f"t={qa['query_time']:.0f}s {qa['type']}")
        try:
            validation = validate_streaming_qa(
                qa, video_path,
                video_id=data['video_id'],
                api_key=api_key,
                model=model,
                seen_window_sec=seen_window_sec,
                future_window_sec=future_window_sec,
            )
            qa['validation'] = validation
            verdict = validation.get('verdict', 'fail')
            stats[verdict] = stats.get(verdict, 0) + 1
            usage = validation.get('usage') or {}
            c = usage.get('cost')
            if c is not None:
                cost_total += c
            pdetails = usage.get('prompt_tokens_details') or {}
            vt = pdetails.get('video_tokens', 0) or 0
            video_token_total += vt
            print(f"      verdict={verdict} | reason: {validation['reason'][:160]}")
        except Exception as e:
            print(f"      ERROR: {type(e).__name__}: {e}")
            qa['validation'] = {
                "verdict": "fail",
                "reason": f"exception: {type(e).__name__}: {e}",
                "validator_model": model,
            }
            stats["error"] += 1

        time.sleep(0.5)  # gentle pacing

    if drop_failed:
        kept = [q for q in qa_list if q.get('validation', {}).get('verdict') == 'pass']
    else:
        kept = qa_list

    # Strip per-QA 'usage' field before serializing — it bloats the file.
    # We keep an aggregate in the top-level instead.
    for q in qa_list:
        v = q.get('validation') or {}
        v.pop('usage', None)

    output = {
        **data,
        'validator_model': model,
        'validation_stats': stats,
        'validation_cost_usd': round(cost_total, 6),
        'validation_video_tokens_total': video_token_total,
        'num_after_validation': len(kept),
        'streaming_qa': kept,
    }

    if output_path is None:
        output_path = streaming_qa_path.parent / (
            streaming_qa_path.stem.replace('_streaming_qa', '')
            + '_streaming_qa_validated.json'
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
    print(f"  Final kept     : {len(kept)}/{len(qa_list)} (drop_failed={drop_failed})")
    print(f"  Total cost     : ${cost_total:.4f}")
    print(f"  Total video tok: {video_token_total:,}")
    print(f"  Saved          : {output_path}")

    return output


# ============================================================================
# CLI
# ============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Streaming QA Validator (Gemini 2.5 Flash via OpenRouter, video clips)"
    )
    parser.add_argument("--streaming-qa", type=str, required=True,
                        help="Path to {video_id}_streaming_qa.json")
    parser.add_argument("--video", type=str, required=True,
                        help="Path to the underlying video file")
    parser.add_argument("--output", type=str, default=None,
                        help="Output path (default: same dir, *_validated.json)")
    parser.add_argument("--api-key", type=str, default=None,
                        help="OpenRouter API key (or set OPENROUTER_API_KEY in .env)")
    parser.add_argument("--model", type=str, default=VALIDATOR_MODEL,
                        help=f"Validator model id (default: {VALIDATOR_MODEL})")
    parser.add_argument("--keep-failed", action="store_true",
                        help="Keep verdict='fail' QA in output (default: drop)")
    parser.add_argument("--seen-window-sec", type=float, default=SEEN_WINDOW_SEC,
                        help=f"Latter portion of SEEN to keep, in seconds "
                             f"(default {SEEN_WINDOW_SEC}). Use 0 or negative to disable cap.")
    parser.add_argument("--future-window-sec", type=float, default=FUTURE_WINDOW_SEC,
                        help=f"Earlier portion of FUTURE to keep, in seconds "
                             f"(default {FUTURE_WINDOW_SEC}). Use 0 or negative to disable cap.")
    parser.add_argument("--max-qa", type=int, default=None,
                        help="Only validate the first N QA (smoke test / debug)")
    args = parser.parse_args()

    seen_w = args.seen_window_sec if args.seen_window_sec and args.seen_window_sec > 0 else None
    fut_w = args.future_window_sec if args.future_window_sec and args.future_window_sec > 0 else None

    validate_streaming_qa_file(
        args.streaming_qa,
        args.video,
        output_path=args.output,
        api_key=args.api_key,
        model=args.model,
        drop_failed=not args.keep_failed,
        seen_window_sec=seen_w,
        future_window_sec=fut_w,
        max_qa=args.max_qa,
    )


if __name__ == "__main__":
    main()
