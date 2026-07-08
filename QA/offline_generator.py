"""
Offline QA Generator (new pipeline)
===================================
Generates ONE holistic offline QA per clip using Gemini 2.5 Flash on
OpenRouter, with the FULL clip mp4 (visuals + audio) as input.

The single QA type is `clip_summary` -- a comprehensive question whose
answer covers:
  - the temporal process ("first ... then ... finally ...")
  - key visual details (anatomical landmarks, echogenicity, probe
    orientation, measurements)
  - relevant medical knowledge (clinical significance, diagnostic criteria)

Output: QA/results/{video_id}_offline_qa.json

This is the offline counterpart to QA/generator.py (streaming). Both run
on the same Gemini backend so the whole pipeline sits on one API.

Usage:
    # OPENROUTER_API_KEY auto-loaded from .env
    python QA/offline_generator.py \\
        --video path/to/ID.mp4 \\
        --clips results/clips/ID_clips.json
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

GENERATOR_MODEL = DEFAULT_MODEL  # "google/gemini-2.5-flash"

OFFLINE_QA_TYPES = ["clip_summary"]  # single type; keep list form for symmetry with generator.py

# Cap on how much of the clip we upload. Clips are 30-300s but very long
# clips would explode video-token cost. 300s is enough for the whole
# training set (per docs/PIPELINE.md all clips are 30-300s).
CLIP_MAX_SEC = 300.0


OFFLINE_QA_PROMPT = """You are a senior ultrasound instructor and medical education expert.

You are shown a single ultrasound teaching clip that runs from t={clip_start:.0f}s to t={clip_end:.0f}s (duration {duration:.0f}s).
Clip topic: {topic}

You have access to:
  [CLIP_VIDEO] the entire clip's original visual frames AND audio (operator's narration)
  [ASR]        the full ASR transcript of the clip: "{full_asr}"

Write EXACTLY ONE holistic question-answer pair of type "clip_summary" that gives a complete overview of this clip. It should be the single most informative QA a learner could get about the clip.

The QUESTION should invite a comprehensive walk-through of what happens in the clip (e.g. "What does this clip demonstrate, and what should a learner take away from it?").

The ANSWER must be a single self-contained explanation that covers ALL THREE of the following aspects (weave them together into a natural, flowing 4-8 sentence answer -- do NOT output them as three separate bullets):

  (1) TEMPORAL PROCESS: What happens from beginning to end? Use temporal markers ("first ... then ... finally ...") so the answer reflects the clip's evolution over time.
  (2) KEY VISUAL DETAILS: Cite specific anatomical landmarks visible, echogenicity patterns, probe orientation/positioning, depth or gain notes, on-screen measurements, or Doppler if used. Anchor these to when they appear in the clip.
  (3) RELEVANT MEDICAL KNOWLEDGE: Add educational context that goes BEYOND what's visible -- clinical significance, diagnostic criteria, common pitfalls, or normal ranges relevant to what's shown.

Style rules:
  - Concrete and clinically grounded. Cite what the operator actually says or does.
  - No speculation beyond what the frames + narration support.
  - Do NOT split the answer into (1)/(2)/(3) sections; deliver a single cohesive paragraph.
  - Do NOT begin the answer with "This clip shows..." templates -- vary the phrasing.

Also emit an "evidence" field: a short (1-2 sentence) pointer to what in the clip most strongly supports your answer (e.g. "The operator's narration at ~40s explains ..., and the pleural line becomes clearly visible at ~55s").

Output STRICTLY this JSON object (no markdown fences, no other text):
{{
  "type": "clip_summary",
  "question": "...",
  "answer": "...",
  "evidence": "..."
}}"""


# ============================================================================
# JSON parsing
# ============================================================================

def _parse_json_object(raw):
    """Extract a JSON object from `raw`, handling fenced / partial output."""
    if not raw:
        return None

    try:
        v = json.loads(raw)
        return v if isinstance(v, dict) else None
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

    return None


# ============================================================================
# Per-clip offline QA
# ============================================================================

def generate_offline_qa_for_clip(video_path, clip, *,
                                 api_key=None, model=GENERATOR_MODEL,
                                 clip_max_sec=CLIP_MAX_SEC,
                                 video_id=None):
    """
    Generate ONE offline QA (clip_summary) for a single clip.
    Returns (qa_or_None, usage, error_reason). qa_or_None is None on failure.
    """
    client = build_openrouter_client(api_key)

    clip_start = float(clip['start'])
    clip_end_full = float(clip['end'])
    clip_end = min(clip_end_full, clip_start + clip_max_sec)
    duration = clip_end - clip_start
    clip_idx = clip['clip_idx']

    if video_id is None:
        video_id = Path(video_path).stem

    clip_path = temp_clip_path(
        video_id,
        f"clip{clip_idx}_offline_{int(duration)}s",
    )
    cut_clip(video_path, clip_start, clip_end, clip_path)

    prompt = OFFLINE_QA_PROMPT.format(
        clip_start=clip_start,
        clip_end=clip_end,
        duration=duration,
        topic=clip.get('topic', 'ultrasound'),
        full_asr=clip.get('text', '')[:3500],
    )

    content_blocks = [
        text_block(f"=== [CLIP_VIDEO] {clip_start:.0f}s -> {clip_end:.0f}s ==="),
        build_video_block(clip_path, label="clip.mp4"),
        text_block(prompt),
    ]

    t0 = time.time()
    raw, usage = call_with_content(
        client,
        content_blocks=content_blocks,
        model=model,
        temperature=0.3,
    )
    elapsed = time.time() - t0

    pdetails = (usage.get("prompt_tokens_details") or {})
    vt = pdetails.get("video_tokens", 0)
    at_tok = pdetails.get("audio_tokens", 0)
    cost = usage.get("cost")
    cost_s = f"${cost:.6f}" if cost is not None else "n/a"
    print(f"    Gemini: {elapsed:.1f}s | tot={usage.get('total_tokens')} "
          f"video={vt} audio={at_tok} | cost={cost_s}")

    parsed = _parse_json_object(raw)
    if parsed is None:
        print(f"    WARNING: failed to parse: {(raw or '')[:240]!r}")
        return None, usage, "parse_failure"

    q = (parsed.get("question") or "").strip()
    a = (parsed.get("answer") or "").strip()
    if not q or not a:
        return None, usage, "empty_question_or_answer"

    qa = {
        "source": "offline",
        "type": "clip_summary",
        "video_id": video_id,
        "clip_idx": clip_idx,
        "clip_start": clip_start,
        "clip_end": clip_end_full,  # keep original end (not truncated) for merger sanity
        "topic": clip.get("topic", ""),
        "question": q,
        "answer": a,
        "evidence": (parsed.get("evidence") or "").strip(),
    }
    return qa, usage, None


# ============================================================================
# Full pipeline
# ============================================================================

def generate_offline_qa_for_video(video_path, clips_path,
                                   output_dir="QA/results",
                                   api_key=None, single_clip=None,
                                   model=GENERATOR_MODEL,
                                   clip_max_sec=CLIP_MAX_SEC):
    """Generate offline QA (one clip_summary per clip)."""
    video_path = Path(video_path)
    video_id = video_path.stem

    with open(clips_path) as f:
        clips_data = json.load(f)
    clips = clips_data['clips']

    if single_clip is not None:
        clips = [c for c in clips if c['clip_idx'] == single_clip]

    print(f"\n{'='*70}")
    print(f"Offline QA Generation (new pipeline, clip_summary)")
    print(f"  Video           : {video_id}")
    print(f"  Clips           : {len(clips)}")
    print(f"  QA types        : {OFFLINE_QA_TYPES}")
    print(f"  Generator       : {model} (OpenRouter, video segments)")
    print(f"  Clip cap        : {clip_max_sec}s")
    print(f"{'='*70}")

    all_qa = []
    cost_total = 0.0
    video_token_total = 0
    error_log = []

    for clip in clips:
        print(f"\n  Clip {clip['clip_idx']}: "
              f"{clip['start']:.0f}-{clip['end']:.0f}s | "
              f"{clip.get('topic', '')[:40]}")
        try:
            qa, usage, err = generate_offline_qa_for_clip(
                str(video_path), clip,
                api_key=api_key,
                model=model,
                clip_max_sec=clip_max_sec,
                video_id=video_id,
            )
            if qa is not None:
                all_qa.append(qa)
            else:
                error_log.append({
                    "clip_idx": clip['clip_idx'],
                    "reason": err or "unknown",
                })

            if usage:
                c = usage.get('cost')
                if c is not None:
                    cost_total += c
                pdetails = usage.get('prompt_tokens_details') or {}
                vt = pdetails.get('video_tokens', 0) or 0
                video_token_total += vt
        except Exception as e:
            print(f"    ERROR: {type(e).__name__}: {e}")
            error_log.append({
                "clip_idx": clip['clip_idx'],
                "reason": f"exception: {type(e).__name__}: {e}",
            })

        time.sleep(0.5)

    output = {
        'video_id': video_id,
        'video_path': str(video_path),
        'model': model,
        'qa_types': OFFLINE_QA_TYPES,
        'num_clips': len(clips),
        'clip_max_sec': clip_max_sec,
        'mode': 'video_segments',
        'num_offline_qa': len(all_qa),
        'num_errors': len(error_log),
        'error_log': error_log,
        'generation_cost_usd': round(cost_total, 6),
        'generation_video_tokens_total': video_token_total,
        # keep the field name compatible with scripts/qa_generation.py output
        # so QA/merger.py can read either flavour uniformly.
        'qa_pairs': all_qa,
    }

    out_path = Path(output_dir) / f"{video_id}_offline_qa.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, 'w', encoding='utf-8') as f:
        json.dump(output, f, ensure_ascii=False, indent=2)

    print(f"\n  Kept {len(all_qa)} offline QA (errors={len(error_log)})")
    print(f"  Total cost           : ${cost_total:.4f}")
    print(f"  Total video tokens   : {video_token_total:,}")
    print(f"  Saved                : {out_path}")

    return output


# ============================================================================
# CLI
# ============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Offline QA Generator (Gemini 2.5 Flash on OpenRouter, clip_summary)"
    )
    parser.add_argument("--video", type=str, required=True)
    parser.add_argument("--clips", type=str, required=True)
    parser.add_argument("--output-dir", type=str, default="QA/results")
    parser.add_argument("--api-key", type=str, default=None)
    parser.add_argument("--model", type=str, default=GENERATOR_MODEL)
    parser.add_argument("--single-clip", type=int, default=None)
    parser.add_argument("--clip-max-sec", type=float, default=CLIP_MAX_SEC,
                        help=f"Cap clip length uploaded to Gemini (default {CLIP_MAX_SEC}s)")
    args = parser.parse_args()

    generate_offline_qa_for_video(
        args.video,
        args.clips,
        output_dir=args.output_dir,
        api_key=args.api_key,
        single_clip=args.single_clip,
        model=args.model,
        clip_max_sec=args.clip_max_sec,
    )


if __name__ == "__main__":
    main()
