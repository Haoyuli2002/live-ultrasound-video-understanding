"""
Streaming QA Generation Pipeline (Oracle Generator, video-clip input)
=====================================================================
Generates 2 types of TIME-SENSITIVE QA at multiple anchor points within each clip:
  - sonographer_intent     : What is the operator currently trying to do/find?
  - next_action_guidance   : What should the sonographer do next?

Design (oracle generator)
-------------------------
The generator is an "oracle": it sees BOTH the SEEN segment (clip_start ->
query_time) AND the FUTURE segment (query_time -> clip_end). It MUST,
however, phrase the QUESTION as if only SEEN were available (no future
leakage). The ANSWER is allowed to use the full clip context as ground
truth — it should describe the *real* intent or *real* next action that
ACTUALLY occurs in the video.

Why video clips (not 6 sampled frames)?
---------------------------------------
We verified via (see _video_llm.py) that OpenRouter
+ Gemini 2.5 Flash natively accepts mp4 via the `type: "file"`
content block (video_tokens > 0 in the usage breakdown). This gives
the generator full motion + audio (operator's narration), so the
written answer can incorporate the actual spoken commentary, which is
exactly what's needed to write good streaming QA.

Validation against information leakage / hallucination is performed by
`scripts/qa_validator.py` using the same Gemini 2.5 Flash backend on
the same SEEN/FUTURE segments.

Usage:
    # OPENROUTER_API_KEY auto-loaded from .env
    python scripts/streaming_qa_generation.py \\
        --video path/to/ID.mp4 \\
        --clips results/clips/ID_clips.json
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

GENERATOR_MODEL = DEFAULT_MODEL  # "google/gemini-2.5-flash"

TIME_RATIOS = [0.25, 0.5, 0.75]
STREAMING_QA_TYPES = ["sonographer_intent", "next_action_guidance"]

# SEEN window: cap to the latter `SEEN_WINDOW_SEC` seconds before query_time.
# Generator only needs recent context to phrase a learner-style question.
SEEN_WINDOW_SEC = 240.0

# FUTURE window: NOT capped by default. The generator must write the
# *actual* next action that occurs in the video, so it benefits from seeing
# the entire FUTURE segment up to clip_end. clips are 30–300s so this is
# bounded naturally.
FUTURE_WINDOW_SEC = None


STREAMING_QA_PROMPT = """You are a senior ultrasound instructor and medical education expert.

You are observing an ultrasound teaching clip that runs from t={clip_start:.0f}s to t={clip_end:.0f}s.

The "query time" is t={current_time:.0f}s — the moment a learner pauses and asks a question while watching the video live.

You are given TWO video segments in temporal order, each with its original visual frames AND audio (operator's narration):

  [SEEN_VIDEO]   covers ({seen_a:.0f}s -> {seen_b:.0f}s) — the part the learner has already watched.
  [FUTURE_VIDEO] covers ({fut_a:.0f}s -> {fut_b:.0f}s) — the part the learner has NOT watched yet.

You also have the FULL ASR transcript of the clip (no truncation):
\"{full_asr}\"

Clip topic: {topic}

Your job: write exactly TWO question-answer pairs (one of each required type).

Required types:
  1. "sonographer_intent"     — what the operator is trying to accomplish at this moment
  2. "next_action_guidance"   — what the sonographer should do next

CRITICAL writing rules — apply to BOTH types:

▸ The QUESTION must be writable based ONLY on [SEEN_VIDEO] (visuals + narration heard up to t={current_time:.0f}s).
  - Phrase it naturally as a learner pausing live: "at this point...", "right now...", "based on what we've seen so far..."
  - The question must NOT reveal or reference any content that only exists in [FUTURE_VIDEO] (visually OR audibly).

▸ The ANSWER may use the FULL clip context ([SEEN_VIDEO] + [FUTURE_VIDEO] + full ASR).
  - The answer should describe the REAL intent / REAL next action that ACTUALLY occurs.
  - Be specific: cite anatomical structures, probe orientation, what the operator says, what action they perform.
  - If [FUTURE_VIDEO] shows the operator pausing to explain rather than performing a maneuver, say so accurately.
  - Do NOT speculate beyond what the frames + ASR support.

Output STRICTLY a JSON array of 2 objects (no markdown fences, no other text):
[
  {{"type": "sonographer_intent",   "timestamp": {current_time:.2f}, "question": "...", "answer": "..."}},
  {{"type": "next_action_guidance", "timestamp": {current_time:.2f}, "question": "...", "answer": "..."}}
]"""


# ============================================================================
# JSON parsing — robust array extraction
# ============================================================================

def _parse_json_array(raw):
    """Extract a JSON array of QA objects from Gemini's reply."""
    if not raw:
        return None

    # plain JSON
    try:
        v = json.loads(raw)
        return v if isinstance(v, list) else None
    except json.JSONDecodeError:
        pass

    # ```json ... ``` fenced block
    m = re.search(r'```(?:json)?\s*(\[.+?\])\s*```', raw, re.DOTALL)
    if m:
        try:
            return json.loads(m.group(1))
        except json.JSONDecodeError:
            pass

    # any complete-looking array
    m = re.search(r'\[.+\]', raw, re.DOTALL)
    if m:
        try:
            return json.loads(m.group(0))
        except json.JSONDecodeError:
            pass

    return None


# ============================================================================
# Window helpers (mirrors qa_validator.py)
# ============================================================================

def _seen_window(clip_start, query_time, window_sec):
    """Return [a, b] for SEEN; if capped, keep the latter `window_sec`."""
    a, b = clip_start, query_time
    if window_sec and (b - a) > window_sec:
        a = b - window_sec
    return a, b


def _future_window(query_time, clip_end, window_sec):
    """Return [a, b] for FUTURE; if capped, keep the earlier `window_sec`.
    Pass window_sec=None to use the full FUTURE segment."""
    a, b = query_time, clip_end
    if window_sec and (b - a) > window_sec:
        b = a + window_sec
    return a, b


# ============================================================================
# Streaming QA Generation (oracle: sees SEEN + FUTURE)
# ============================================================================

def generate_streaming_qa(video_path, clip, ratio=0.5, *,
                          api_key=None, model=GENERATOR_MODEL,
                          seen_window_sec=SEEN_WINDOW_SEC,
                          future_window_sec=FUTURE_WINDOW_SEC,
                          video_id=None):
    """
    Generate 2 streaming QA at a specific time anchor within a clip,
    using Gemini 2.5 Flash with full SEEN/FUTURE video segments
    (frames + audio) as input.

    The generator is an ORACLE: it receives both segments + full ASR.
    It must phrase the QUESTION as if only SEEN were available, while
    the ANSWER is allowed to use the full clip context as ground truth.
    """
    client = build_openrouter_client(api_key)

    clip_start = float(clip['start'])
    clip_end = float(clip['end'])
    duration = float(clip['duration'])
    current_time = clip_start + duration * ratio
    clip_idx = clip['clip_idx']

    if video_id is None:
        video_id = Path(video_path).stem

    seen_a, seen_b = _seen_window(clip_start, current_time, seen_window_sec)
    fut_a, fut_b = _future_window(current_time, clip_end, future_window_sec)

    # Cache file names — include effective window sizes so different runs
    # don't accidentally reuse the wrong cached clip.
    seen_path = temp_clip_path(
        video_id,
        f"clip{clip_idx}_t{int(current_time)}_seen_{int(seen_b - seen_a)}s",
    )
    future_path = temp_clip_path(
        video_id,
        f"clip{clip_idx}_t{int(current_time)}_future_{int(fut_b - fut_a)}s",
    )

    cut_clip(video_path, seen_a, seen_b, seen_path)
    cut_clip(video_path, fut_a, fut_b, future_path)

    prompt = STREAMING_QA_PROMPT.format(
        clip_start=clip_start,
        clip_end=clip_end,
        current_time=current_time,
        seen_a=seen_a,
        seen_b=seen_b,
        fut_a=fut_a,
        fut_b=fut_b,
        full_asr=clip.get('text', '')[:3500],
        topic=clip.get('topic', 'ultrasound'),
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
        temperature=0.3,
    )
    elapsed = time.time() - t0

    pdetails = (usage.get("prompt_tokens_details") or {})
    vt = pdetails.get("video_tokens", 0)
    at = pdetails.get("audio_tokens", 0)
    cost = usage.get("cost")
    cost_s = f"${cost:.6f}" if cost is not None else "n/a"
    print(f"    t={current_time:.0f}s (ratio={ratio}) | "
          f"SEEN {seen_b-seen_a:.0f}s + FUTURE {fut_b-fut_a:.0f}s | "
          f"{elapsed:.1f}s | tot={usage.get('total_tokens')} "
          f"video={vt} audio={at} | cost={cost_s}")

    qa_pairs = _parse_json_array(raw)
    if qa_pairs is None:
        print(f"    WARNING: failed to parse: {(raw or '')[:240]!r}")
        return [], usage

    qa_pairs = [q for q in qa_pairs if q.get("type") in STREAMING_QA_TYPES]

    for qa in qa_pairs:
        qa['source'] = 'streaming'
        qa['clip_idx'] = clip_idx
        qa['clip_start'] = clip_start
        qa['clip_end'] = clip_end
        qa['query_time'] = round(current_time, 2)
        qa['ratio'] = ratio
        qa['topic'] = clip.get('topic', '')

    return qa_pairs, usage


# ============================================================================
# Full Pipeline
# ============================================================================

def generate_streaming_qa_for_video(video_path, clips_path, output_dir="results/qa",
                                     api_key=None, single_clip=None,
                                     time_ratios=None,
                                     model=GENERATOR_MODEL,
                                     seen_window_sec=SEEN_WINDOW_SEC,
                                     future_window_sec=FUTURE_WINDOW_SEC):
    """Generate streaming QA for all clips. Output: {output_dir}/{video_id}_streaming_qa.json"""
    video_path = Path(video_path)
    video_id = video_path.stem

    with open(clips_path) as f:
        clips_data = json.load(f)
    clips = clips_data['clips']

    if single_clip is not None:
        clips = [c for c in clips if c['clip_idx'] == single_clip]

    ratios = time_ratios or TIME_RATIOS
    expected = len(clips) * len(ratios) * len(STREAMING_QA_TYPES)

    print(f"\n{'='*70}")
    print(f"Streaming QA Generation (Oracle, video clips): {video_id}")
    print(f"  Clips                : {len(clips)}")
    print(f"  Anchors per clip     : {len(ratios)}")
    print(f"  QA types per anchor  : {len(STREAMING_QA_TYPES)}")
    print(f"  Expected QA total    : {expected}")
    print(f"  Generator            : {model} (OpenRouter)")
    print(f"  SEEN window          : {seen_window_sec}s")
    print(f"  FUTURE window        : {future_window_sec if future_window_sec else 'uncapped (until clip_end)'}")
    print(f"{'='*70}")

    all_qa = []
    cost_total = 0.0
    video_token_total = 0

    for clip in clips:
        print(f"\n  Clip {clip['clip_idx']}: "
              f"{clip['start']:.0f}-{clip['end']:.0f}s | "
              f"{clip.get('topic', '')[:40]}")
        for ratio in ratios:
            try:
                qa_pairs, usage = generate_streaming_qa(
                    str(video_path), clip,
                    ratio=ratio,
                    api_key=api_key,
                    model=model,
                    seen_window_sec=seen_window_sec,
                    future_window_sec=future_window_sec,
                    video_id=video_id,
                )
                for qa in qa_pairs:
                    qa['video_id'] = video_id
                all_qa.extend(qa_pairs)

                c = usage.get('cost') if usage else None
                if c is not None:
                    cost_total += c
                pdetails = (usage or {}).get('prompt_tokens_details') or {}
                vt = pdetails.get('video_tokens', 0) or 0
                video_token_total += vt
            except Exception as e:
                print(f"    ERROR at ratio={ratio}: {type(e).__name__}: {e}")
            time.sleep(0.5)

    output = {
        'video_id': video_id,
        'video_path': str(video_path),
        'model': model,
        'qa_types': STREAMING_QA_TYPES,
        'num_clips': len(clips),
        'time_ratios': ratios,
        'seen_window_sec': seen_window_sec,
        'future_window_sec': future_window_sec,
        'mode': 'video_segments',
        'num_streaming_qa': len(all_qa),
        'generation_cost_usd': round(cost_total, 6),
        'generation_video_tokens_total': video_token_total,
        'streaming_qa': all_qa,
    }

    out_path = Path(output_dir) / f"{video_id}_streaming_qa.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, 'w', encoding='utf-8') as f:
        json.dump(output, f, ensure_ascii=False, indent=2)

    print(f"\n  Saved {len(all_qa)} streaming QA (unverified) to: {out_path}")
    print(f"  Total cost           : ${cost_total:.4f}")
    print(f"  Total video tokens   : {video_token_total:,}")
    print(f"  -> Next: scripts/qa_validator.py --streaming-qa {out_path} --video <video>")

    return output


# ============================================================================
# CLI
# ============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Streaming QA Generation (Oracle, Gemini 2.5 Flash via OpenRouter, video clips)"
    )
    parser.add_argument("--video", type=str, required=True, help="Video file path")
    parser.add_argument("--clips", type=str, required=True, help="Clips JSON path")
    parser.add_argument("--output-dir", type=str, default="results/qa")
    parser.add_argument("--api-key", type=str, default=None,
                        help="OpenRouter API key (or set OPENROUTER_API_KEY in .env)")
    parser.add_argument("--model", type=str, default=GENERATOR_MODEL,
                        help=f"Generator model id (default: {GENERATOR_MODEL})")
    parser.add_argument("--single-clip", type=int, default=None,
                        help="Only process this clip index (debug)")
    parser.add_argument("--ratios", type=str, default=None,
                        help="Comma-separated time ratios, e.g. '0.25,0.5,0.75'")
    parser.add_argument("--seen-window-sec", type=float, default=SEEN_WINDOW_SEC,
                        help=f"Cap SEEN segment to last N seconds (default {SEEN_WINDOW_SEC}). "
                             f"Use 0 to disable cap.")
    parser.add_argument("--future-window-sec", type=float, default=-1,
                        help="Cap FUTURE segment to first N seconds. "
                             "Default: uncapped (until clip_end). Pass a positive value to cap.")
    args = parser.parse_args()

    ratios = None
    if args.ratios:
        ratios = [float(x) for x in args.ratios.split(',')]

    seen_w = args.seen_window_sec if args.seen_window_sec and args.seen_window_sec > 0 else None
    # future-window-sec: -1 (default sentinel) means uncapped
    if args.future_window_sec is None or args.future_window_sec < 0:
        fut_w = None
    elif args.future_window_sec == 0:
        fut_w = None
    else:
        fut_w = args.future_window_sec

    generate_streaming_qa_for_video(
        args.video,
        args.clips,
        output_dir=args.output_dir,
        api_key=args.api_key,
        single_clip=args.single_clip,
        time_ratios=ratios,
        model=args.model,
        seen_window_sec=seen_w,
        future_window_sec=fut_w,
    )


if __name__ == "__main__":
    main()
