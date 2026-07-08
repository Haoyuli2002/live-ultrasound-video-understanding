"""
Streaming QA Validator (new pipeline)
=====================================
For each generated streaming QA carrying (query_time, answer_time), this
validator sends THREE video segments to a Gemini 2.5 Flash on OpenRouter
and asks it to check three hard constraints -- all must hold for `pass`:

  (C1) question_no_leak:
       The QUESTION is writable using ONLY [BEFORE_QUERY] (visuals + audio
       up to query_time). It does not reveal or reference anything that
       only exists after query_time.

  (C2) not_answerable_at_query_time:
       The ANSWER cannot be derived from [BEFORE_QUERY] alone. There is a
       genuine information gap: something needed for the answer is missing
       until later.

  (C3) answerable_at_answer_time:
       By the end of [EVIDENCE_SPAN] (i.e. by answer_time), the evidence in
       [clip_start, answer_time] first becomes sufficient to derive the
       ANSWER. [AFTER_ANSWER] is provided as a short tail so the validator
       can also sanity-check that answer_time is not obviously too late.

Verdict = "pass" iff all three checks hold. Otherwise "fail".

Video segments:
  [BEFORE_QUERY]   = [clip_start, query_time]           (cap to last 240s)
  [EVIDENCE_SPAN]  = [query_time,  answer_time]         (uncapped, usually <60s)
  [AFTER_ANSWER]   = [answer_time, min(answer_time+10, clip_end)]

Usage:
    # OPENROUTER_API_KEY auto-loaded from .env
    python QA/validator.py \\
        --streaming-qa QA/results/ID_streaming_qa.json \\
        --video path/to/ID.mp4
"""

from __future__ import annotations

import sys
import json
import time
import re
import argparse
from pathlib import Path

_HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE))
from _shared import (  # noqa: E402
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

VALIDATOR_MODEL = DEFAULT_MODEL
BEFORE_WINDOW_SEC = 240.0   # cap on [clip_start, query_time], keep latter part
EVIDENCE_WINDOW_SEC = None  # usually <60s naturally, keep uncapped
AFTER_WINDOW_SEC = 10.0     # short tail after answer_time


VALIDATOR_PROMPT_TEMPLATE = """You are a strict validator for a streaming ultrasound QA benchmark.

Clip: {clip_start:.0f}s -> {clip_end:.0f}s.
Query time:  t_q = {query_time:.0f}s.
Answer time: t_a = {answer_time:.0f}s   (t_a > t_q per contract).

QA to validate:
  Type:     {qa_type}
  Question: "{question}"
  Answer:   "{answer}"

You are given THREE video segments in temporal order. Each carries its ORIGINAL visual frames AND audio (operator's narration):

  [BEFORE_QUERY]  ({before_a:.0f}s -> {before_b:.0f}s)   what the learner had seen BEFORE the question
  [EVIDENCE_SPAN] ({ev_a:.0f}s -> {ev_b:.0f}s)   what happens between t_q and t_a
  [AFTER_ANSWER]  ({after_a:.0f}s -> {after_b:.0f}s)   short tail AFTER t_a, for sanity check only

Check ALL THREE conditions. Every one must independently be true for a pass.

(C1) question_no_leak:
     The QUESTION can be written using ONLY [BEFORE_QUERY] (frames + audio).
     It does NOT reveal or reference anything that only appears from
     [EVIDENCE_SPAN] onward -- visually OR audibly.

(C2) not_answerable_at_query_time:
     The ANSWER cannot be derived from [BEFORE_QUERY] alone. There is a real
     information gap: some frame content or narration needed to answer is
     simply not present before t_q. (If [BEFORE_QUERY] already contains all
     the answer's evidence, this check FAILS -- the QA is trivial.)

(C3) answerable_at_answer_time:
     By the end of [EVIDENCE_SPAN] (i.e. by t_a), the accumulated evidence
     across [BEFORE_QUERY] + [EVIDENCE_SPAN] is sufficient to derive the
     ANSWER. In particular, the critical evidence appears IN [EVIDENCE_SPAN].
     ([AFTER_ANSWER] is only for sanity: if the answer clearly relies on
     content that only appears AFTER t_a, this check FAILS.)

Verdict rule (strict):
  "pass" iff (C1 AND C2 AND C3) all TRUE.
  "fail" otherwise.

Output STRICTLY this JSON object (no markdown fences, no extra text):
{{
  "checks": {{
    "question_no_leak": <true|false>,
    "not_answerable_at_query_time": <true|false>,
    "answerable_at_answer_time": <true|false>
  }},
  "verdict": "pass" | "fail",
  "reason": "<MANDATORY: at least 3 sentences (>=45 words total). For EACH of the three checks, cite specific visible / audible content from the relevant segment(s). Say which frames or which spoken phrases matter, and why they support your true/false decision. Vague statements like 'looks fine' are not acceptable.>"
}}"""


# ============================================================================
# JSON parsing
# ============================================================================

def _parse_json_object(raw):
    """Extract a JSON object from `raw`, handling fenced/partial output."""
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

    m = re.search(r'\{.*\}', raw, re.DOTALL)
    if m:
        try:
            return json.loads(m.group(0))
        except json.JSONDecodeError:
            pass

    # last-ditch: recover verdict at minimum
    verdict_match = re.search(r'"verdict"\s*:\s*"(pass|fail)"', raw)
    if verdict_match:
        return {
            "checks": {},
            "verdict": verdict_match.group(1),
            "reason": "[truncated]",
        }
    return None


# ============================================================================
# Window helpers
# ============================================================================

def _before_window(clip_start, query_time, window_sec):
    a, b = clip_start, query_time
    if window_sec and (b - a) > window_sec:
        a = b - window_sec
    return a, b


def _evidence_window(query_time, answer_time, window_sec):
    a, b = query_time, answer_time
    if window_sec and (b - a) > window_sec:
        b = a + window_sec
    return a, b


def _after_window(answer_time, clip_end, window_sec):
    a = answer_time
    b = min(clip_end, answer_time + (window_sec or 0.0))
    if b <= a:
        b = min(clip_end, a + 0.5)  # guarantee non-empty
    return a, b


# ============================================================================
# Verdict resolution
# ============================================================================

def _resolve_verdict(parsed):
    """
    Enforce the strict rule: verdict = pass iff all three checks are true.
    If the model returned inconsistent verdict/checks, we OVERRIDE with the
    checks (source of truth), and note the override in the reason.
    """
    checks = parsed.get("checks") or {}
    keys = ("question_no_leak",
            "not_answerable_at_query_time",
            "answerable_at_answer_time")
    # Only treat as True if strictly True; missing / non-bool -> False.
    all_true = all(bool(checks.get(k)) is True and checks.get(k) is True
                   for k in keys)
    computed_verdict = "pass" if all_true else "fail"

    model_verdict = parsed.get("verdict", "").lower()
    reason = parsed.get("reason", "")

    if model_verdict != computed_verdict:
        # be strict: the checks decide
        override_note = (
            f" [validator note: model reported verdict='{model_verdict}' "
            f"but checks={dict(checks)} -> overridden to '{computed_verdict}']"
        )
        reason = (reason or "") + override_note

    normalised_checks = {k: bool(checks.get(k)) is True and checks.get(k) is True
                         for k in keys}

    return {
        "verdict": computed_verdict,
        "reason": reason,
        "checks": normalised_checks,
    }


# ============================================================================
# Per-QA validation
# ============================================================================

def validate_streaming_qa(qa, video_path, *, video_id=None,
                          api_key=None, model=VALIDATOR_MODEL,
                          before_window_sec=BEFORE_WINDOW_SEC,
                          evidence_window_sec=EVIDENCE_WINDOW_SEC,
                          after_window_sec=AFTER_WINDOW_SEC):
    """
    Validate one streaming QA. Returns dict:
        {verdict, reason, checks, validator_model, usage}
    """
    client = build_openrouter_client(api_key)

    clip_start = float(qa['clip_start'])
    clip_end = float(qa['clip_end'])
    query_time = float(qa['query_time'])
    answer_time = float(qa['answer_time'])
    clip_idx = qa.get('clip_idx', 0)

    if video_id is None:
        video_id = Path(video_path).stem

    before_a, before_b = _before_window(clip_start, query_time, before_window_sec)
    ev_a, ev_b = _evidence_window(query_time, answer_time, evidence_window_sec)
    after_a, after_b = _after_window(answer_time, clip_end, after_window_sec)

    before_path = temp_clip_path(
        video_id,
        f"clip{clip_idx}_t{int(query_time)}_before_{int(before_b - before_a)}s",
    )
    evidence_path = temp_clip_path(
        video_id,
        f"clip{clip_idx}_t{int(query_time)}_ta{int(answer_time)}_evidence_{int(ev_b - ev_a)}s",
    )
    after_path = temp_clip_path(
        video_id,
        f"clip{clip_idx}_ta{int(answer_time)}_after_{int(after_b - after_a)}s",
    )

    cut_clip(video_path, before_a, before_b, before_path)
    cut_clip(video_path, ev_a, ev_b, evidence_path)
    cut_clip(video_path, after_a, after_b, after_path)

    prompt = VALIDATOR_PROMPT_TEMPLATE.format(
        clip_start=clip_start,
        clip_end=clip_end,
        query_time=query_time,
        answer_time=answer_time,
        qa_type=qa['type'],
        question=qa['question'],
        answer=qa['answer'],
        before_a=before_a,
        before_b=before_b,
        ev_a=ev_a,
        ev_b=ev_b,
        after_a=after_a,
        after_b=after_b,
    )

    content_blocks = [
        text_block(f"=== [BEFORE_QUERY]  {before_a:.0f}s -> {before_b:.0f}s ==="),
        build_video_block(before_path, label="before.mp4"),
        text_block(f"=== [EVIDENCE_SPAN] {ev_a:.0f}s -> {ev_b:.0f}s ==="),
        build_video_block(evidence_path, label="evidence.mp4"),
        text_block(f"=== [AFTER_ANSWER]  {after_a:.0f}s -> {after_b:.0f}s ==="),
        build_video_block(after_path, label="after.mp4"),
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
    at_tok = pdetails.get("audio_tokens", 0)
    cost = usage.get("cost")
    cost_s = f"${cost:.6f}" if cost is not None else "n/a"
    print(f"      Gemini: {elapsed:.1f}s | tot={usage.get('total_tokens')} "
          f"video={vt} audio={at_tok} | cost={cost_s}")
    raw_preview = (raw or "<EMPTY>")[:200].replace("\n", " ")
    print(f"      RAW: {raw_preview!r}")

    parsed = _parse_json_object(raw)
    if parsed is None:
        print(f"      WARNING: validator output unparseable")
        return {
            "verdict": "fail",
            "reason": "validator_parse_failure",
            "checks": {
                "question_no_leak": False,
                "not_answerable_at_query_time": False,
                "answerable_at_answer_time": False,
            },
            "validator_model": model,
            "usage": usage,
        }

    resolved = _resolve_verdict(parsed)

    return {
        **resolved,
        "validator_model": model,
        "usage": usage,
    }


# ============================================================================
# Batch validation
# ============================================================================

def validate_streaming_qa_file(streaming_qa_path, video_path, output_path=None,
                               api_key=None, model=VALIDATOR_MODEL,
                               drop_failed=True,
                               before_window_sec=BEFORE_WINDOW_SEC,
                               evidence_window_sec=EVIDENCE_WINDOW_SEC,
                               after_window_sec=AFTER_WINDOW_SEC,
                               max_qa=None):
    """Validate every QA in a streaming-QA file."""
    streaming_qa_path = Path(streaming_qa_path)
    with open(streaming_qa_path) as f:
        data = json.load(f)

    qa_list = data['streaming_qa']
    if max_qa is not None:
        qa_list = qa_list[:max_qa]

    print(f"\n{'='*70}")
    print(f"Streaming QA Validation (three-check, new pipeline)")
    print(f"  Video           : {data['video_id']}")
    print(f"  Input QA count  : {len(qa_list)}"
          + (f" (truncated to {max_qa})" if max_qa else ""))
    print(f"  Validator       : {model} (OpenRouter)")
    print(f"  BEFORE window   : {before_window_sec}s (latter portion before query_time)")
    print(f"  EVIDENCE window : {evidence_window_sec if evidence_window_sec else 'uncapped'}")
    print(f"  AFTER window    : {after_window_sec}s (short tail after answer_time)")
    print(f"{'='*70}")

    stats = {"pass": 0, "fail": 0, "error": 0}
    check_stats = {
        "question_no_leak": {"true": 0, "false": 0},
        "not_answerable_at_query_time": {"true": 0, "false": 0},
        "answerable_at_answer_time": {"true": 0, "false": 0},
    }
    cost_total = 0.0
    video_token_total = 0

    for i, qa in enumerate(qa_list):
        print(f"\n  [{i+1}/{len(qa_list)}] clip{qa['clip_idx']} "
              f"t_q={qa['query_time']:.0f}s t_a={qa['answer_time']:.0f}s "
              f"{qa['type']}")
        try:
            validation = validate_streaming_qa(
                qa, video_path,
                video_id=data['video_id'],
                api_key=api_key,
                model=model,
                before_window_sec=before_window_sec,
                evidence_window_sec=evidence_window_sec,
                after_window_sec=after_window_sec,
            )
            qa['validation'] = validation
            verdict = validation.get('verdict', 'fail')
            stats[verdict] = stats.get(verdict, 0) + 1
            for k, v in (validation.get('checks') or {}).items():
                bucket = check_stats.setdefault(k, {"true": 0, "false": 0})
                bucket["true" if v else "false"] += 1

            usage = validation.get('usage') or {}
            c = usage.get('cost')
            if c is not None:
                cost_total += c
            pdetails = usage.get('prompt_tokens_details') or {}
            vt = pdetails.get('video_tokens', 0) or 0
            video_token_total += vt
            print(f"      verdict={verdict} | checks={validation.get('checks')}")
            print(f"      reason: {(validation.get('reason') or '')[:200]}")
        except Exception as e:
            print(f"      ERROR: {type(e).__name__}: {e}")
            qa['validation'] = {
                "verdict": "fail",
                "reason": f"exception: {type(e).__name__}: {e}",
                "checks": {
                    "question_no_leak": False,
                    "not_answerable_at_query_time": False,
                    "answerable_at_answer_time": False,
                },
                "validator_model": model,
            }
            stats["error"] += 1

        time.sleep(0.5)

    if drop_failed:
        kept = [q for q in qa_list if q.get('validation', {}).get('verdict') == 'pass']
    else:
        kept = qa_list

    # Strip per-QA usage before serializing to keep the file small.
    for q in qa_list:
        v = q.get('validation') or {}
        v.pop('usage', None)

    output = {
        **data,
        'validator_model': model,
        'validation_stats': stats,
        'validation_check_stats': check_stats,
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
    print("VALIDATION SUMMARY")
    print(f"{'='*70}")
    for k in ("pass", "fail", "error"):
        print(f"  {k:6s}: {stats.get(k, 0)}")
    print("  Per-check true counts:")
    for k, buckets in check_stats.items():
        print(f"    {k:32s}: true={buckets['true']}  false={buckets['false']}")
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
        description="Streaming QA Validator (three-check, with answer_time)"
    )
    parser.add_argument("--streaming-qa", type=str, required=True)
    parser.add_argument("--video", type=str, required=True)
    parser.add_argument("--output", type=str, default=None)
    parser.add_argument("--api-key", type=str, default=None)
    parser.add_argument("--model", type=str, default=VALIDATOR_MODEL)
    parser.add_argument("--keep-failed", action="store_true",
                        help="Keep verdict='fail' QA in output (default: drop)")
    parser.add_argument("--before-window-sec", type=float, default=BEFORE_WINDOW_SEC)
    parser.add_argument("--evidence-window-sec", type=float, default=-1,
                        help="Cap EVIDENCE_SPAN. Default -1 = uncapped.")
    parser.add_argument("--after-window-sec", type=float, default=AFTER_WINDOW_SEC)
    parser.add_argument("--max-qa", type=int, default=None)
    args = parser.parse_args()

    before_w = args.before_window_sec if args.before_window_sec and args.before_window_sec > 0 else None
    if args.evidence_window_sec is None or args.evidence_window_sec < 0:
        ev_w = None
    elif args.evidence_window_sec == 0:
        ev_w = None
    else:
        ev_w = args.evidence_window_sec
    after_w = args.after_window_sec if args.after_window_sec and args.after_window_sec > 0 else AFTER_WINDOW_SEC

    validate_streaming_qa_file(
        args.streaming_qa,
        args.video,
        output_path=args.output,
        api_key=args.api_key,
        model=args.model,
        drop_failed=not args.keep_failed,
        before_window_sec=before_w,
        evidence_window_sec=ev_w,
        after_window_sec=after_w,
        max_qa=args.max_qa,
    )


if __name__ == "__main__":
    main()
