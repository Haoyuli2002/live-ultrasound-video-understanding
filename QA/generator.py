"""
Streaming QA Generator (new pipeline, with query_time + answer_time)
====================================================================
For each clip we sample 2 anchor times (default ratios 0.3 and 0.6 of clip
duration) and, at every anchor, ask a Gemini 2.5 Flash oracle to write:

  - a QUESTION that is grounded in [clip_start, query_time] only (no future
    leakage), AND is NOT answerable from that segment alone,
  - an ANSWER that describes what actually occurs in the clip,
  - an ANSWER_TIME t_a > query_time -- the earliest moment at which the
    evidence in [clip_start, t_a] becomes sufficient to derive the answer.

The oracle sees BOTH segments (SEEN = [clip_start, query_time] and FUTURE =
(query_time, clip_end]) with visual frames + narration audio, so it can
locate answer_time precisely.

If the oracle judges that no valid (query_time, answer_time) pair exists at
this anchor, the QA at that anchor is DROPPED.

We generate two QA types per anchor:
  - next_action       (what the operator's HANDS should do next)
  - next_observation  (what the learner's EYES should look for next)

Output: QA/results/{video_id}_streaming_qa.json

Usage:
    # OPENROUTER_API_KEY auto-loaded from .env
    python QA/generator.py \\
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

# Anchor time-ratios inside each clip. Two anchors per clip: one in the
# earlier third (0.3) with plenty of FUTURE to problem-shop, and one in
# the middle-to-late region (0.6) that still leaves ~40% of the clip for
# answer_time to land in. Empirically 3 anchors produced too many
# near-duplicate QA within a single clip, so we dropped to 2.
TIME_RATIOS = [0.3, 0.6]

# Streaming QA type pair. Deliberately split into complementary axes so
# they don't collapse into "the same question phrased differently":
#   - next_action:      what the operator's HANDS should do next
#                       (physical maneuver: probe adjustment, compression,
#                        mode switch, patient repositioning, etc.)
#   - next_observation: what the learner's EYES should look for next
#                       (specific anatomical structure, imaging sign, or
#                        image feature that appears on-screen)
# The old 2-type set was ("sonographer_intent", "next_action_guidance")
# which semantically overlapped a lot -- both essentially asked about
# operator action, and validator (C2) failed frequently because the
# "next action" was already stated at query_time by the narration.
STREAMING_QA_TYPES = ["next_action", "next_observation"]

SEEN_WINDOW_SEC = 240.0
FUTURE_WINDOW_SEC = None  # None = uncapped
MIN_FUTURE_SEC_FOR_ANCHOR = 8.0

# Avoid near-identical WAIT/ANSWER samples. If answer_time is too close to
# query_time, the visual windows overlap almost completely and the model
# receives contradictory supervision from nearly identical inputs.
MIN_ANSWER_DELAY_SEC = 5.0


STREAMING_QA_PROMPT = """You are a senior ultrasound instructor creating training data for a real-time (streaming) ultrasound video-understanding model.

The clip runs from t={clip_start:.0f}s to t={clip_end:.0f}s.
Topic: {topic}

A learner is watching the clip live. They pause at query_time t_q = {current_time:.0f}s and ask a question.

You are given TWO video segments in temporal order. Each carries its ORIGINAL visual frames AND audio (operator's narration):

  [SEEN_VIDEO]   covers ({seen_a:.0f}s -> {seen_b:.0f}s) -- everything the learner has watched so far.
  [FUTURE_VIDEO] covers ({fut_a:.0f}s -> {fut_b:.0f}s) -- what happens AFTER the learner's question.

You also have the FULL ASR transcript of the entire clip:
\"{full_asr}\"

Your job: write EXACTLY TWO question-answer pairs (one of each required type below), and for EACH pair also determine an `answer_time`.

Required QA slots (both must be attempted; use `skip:true` if a valid QA
for a slot does not exist at this anchor):

  1. "next_action"
     This asks what concrete scanning action should be performed next to
     acquire, continue, or improve the ultrasound view.

     Valid answers describe an actionable maneuver that actually happens in
     [FUTURE_VIDEO], such as:
       - move / slide / rotate / tilt / fan / angle the probe
       - change probe pressure
       - reposition the patient
       - ask the patient to breathe
       - switch ultrasound mode or adjust settings
       - place the probe at a specific anatomical location
       - apply gel or place the probe on the patient IF that is the real
         next scanning-preparation maneuver

     The question should sound like:
       "What should the operator do next to improve or continue the scan?"

     INVALID for "next_action":
       - asking what to look for on the ultrasound screen
       - asking for a diagnosis or clinical interpretation
       - asking what visual sign appears
       - asking what probe type is used, unless the physical act of selecting
         or picking up that probe is the actual next maneuver

  2. "next_observation"
     This asks what visual evidence the learner should look for next on the
     ultrasound image as the scan continues.

     Valid answers describe a concrete ON-SCREEN ultrasound feature that
     appears or becomes clearer in [FUTURE_VIDEO], such as:
       - an anatomical structure (pleural line, diaphragm, spine, liver,
         spleen, rib shadows, lung surface)
       - an ultrasound artifact (A-lines, B-lines)
       - a diagnostic sign (lung sliding, curtain sign, spine sign)
       - a motion pattern, echogenicity pattern, depth relationship, or
         screen-level change

     The question should sound like:
       "What should the learner look for on the ultrasound image next?"

     INVALID for "next_observation":
       - operator or patient action
       - probe movement
       - probe type or equipment choice
       - clinical diagnosis without direct visual evidence
       - general medical knowledge not tied to a visible screen feature

CORE DESIGN CONSTRAINTS (every QA MUST satisfy):

  (P1) QUESTION grounding:
       The QUESTION must be writable using ONLY [SEEN_VIDEO] (frames + audio).
       It must NOT reveal or reference anything that only exists in [FUTURE_VIDEO].
       Phrase it naturally as a live pause: "at this point...", "right now...",
       "based on what we've seen so far...".

  (P2) NOT answerable at query_time:
       The question must NOT be answerable from [SEEN_VIDEO] alone. There must
       be a genuine information gap between what's been shown/said and what the
       answer requires. If the answer is already obvious from [SEEN_VIDEO], the
       QA has no training value for answerability -- pick a different question.

  (P3) Answerable at some later moment `answer_time`:
       There must exist a moment t_a with {current_time:.0f}s < t_a <= {clip_end:.0f}s
       such that at t_a the evidence in [clip_start, t_a] (frames + audio)
       first becomes sufficient to derive the ANSWER. Report this t_a as
       `answer_time`. Be precise: t_a is the FIRST moment enough evidence
       accumulates -- not the end of the clip, not a random future time.

  (P4) Minimum wait interval:
       The answer_time should be at least 5 seconds after query_time.
       If the answer becomes available immediately or within ~5 seconds,
       skip this QA. Very short waits create nearly identical WAIT/ANSWER
       visual windows and have little training value.

  (P5) Type correctness:
       `next_action` must be about a concrete scanning action.
       `next_observation` must be about a concrete on-screen ultrasound
       observation. If a type does not fit the current anchor, skip it.

ANSWER writing rules:
  - The ANSWER may use the full clip context ([SEEN_VIDEO] + [FUTURE_VIDEO] + ASR).
  - For `next_action`, describe the REAL physical maneuver that ACTUALLY occurs.
  - For `next_observation`, describe the REAL visual evidence that ACTUALLY
    appears or becomes clearer on the ultrasound screen.
  - Cite what happens: anatomical structures, probe motion, screen features,
    artifacts, signs, or what the operator says.
  - Do NOT go beyond what the frames + narration support.

If for a given type NO valid (question, answer, answer_time) triple exists at
this anchor -- e.g. the question would already be answerable from [SEEN_VIDEO],
or answer_time would be within 5 seconds of query_time, or evidence never
becomes sufficient before {clip_end:.0f}s, or the required information gap
doesn't exist -- output that entry with `"skip": true` and a short
`skip_reason`. Do NOT invent a bad QA to fill the slot.

Output STRICTLY a JSON array of 2 objects (no markdown fences, no extra text):
[
  {{
    "type": "next_action",
    "query_time": {current_time:.2f},
    "answer_time": <float in ({current_time:.2f}, {clip_end:.2f}]>,
    "question": "...",
    "answer":   "<must describe an operator maneuver actually performed in [FUTURE_VIDEO]>",
    "evidence": "<1-2 sentences: which specific frames/narration between query_time and answer_time supply the key evidence, and why the question is NOT answerable from [SEEN_VIDEO] alone>"
  }},
  {{
    "type": "next_observation",
    "query_time": {current_time:.2f},
    "answer_time": <float in ({current_time:.2f}, {clip_end:.2f}]>,
    "question": "...",
    "answer":   "<must describe a visual feature actually visible on-screen in [FUTURE_VIDEO]>",
    "evidence": "..."
  }}
]

If a slot must be skipped, use this shape for that slot instead:
  {{"type": "<type>", "query_time": {current_time:.2f}, "skip": true, "skip_reason": "..."}}"""


# ============================================================================
# JSON parsing
# ============================================================================

def _parse_json_array(raw):
    """Extract a JSON array of QA objects from the model's reply."""
    if not raw:
        return None

    try:
        v = json.loads(raw)
        return v if isinstance(v, list) else None
    except json.JSONDecodeError:
        pass

    m = re.search(r'```(?:json)?\s*(\[.+?\])\s*```', raw, re.DOTALL)
    if m:
        try:
            return json.loads(m.group(1))
        except json.JSONDecodeError:
            pass

    m = re.search(r'\[.+\]', raw, re.DOTALL)
    if m:
        try:
            return json.loads(m.group(0))
        except json.JSONDecodeError:
            pass

    return None


# ============================================================================
# Window helpers
# ============================================================================

def _seen_window(clip_start, query_time, window_sec):
    a, b = clip_start, query_time
    if window_sec and (b - a) > window_sec:
        a = b - window_sec
    return a, b


def _future_window(query_time, clip_end, window_sec):
    a, b = query_time, clip_end
    if window_sec and (b - a) > window_sec:
        b = a + window_sec
    return a, b


# ============================================================================
# QA entry validation
# ============================================================================

def _validate_qa_entry(entry, query_time, clip_end):
    """Return (ok, reason). ok=False means drop this entry."""
    if not isinstance(entry, dict):
        return False, "not a dict"

    if entry.get("skip") is True:
        return False, f"model-skipped: {entry.get('skip_reason', 'no reason')}"

    if entry.get("type") not in STREAMING_QA_TYPES:
        return False, f"unknown type: {entry.get('type')!r}"

    for key in ("question", "answer", "answer_time"):
        if key not in entry:
            return False, f"missing key: {key}"

    q = (entry.get("question") or "").strip()
    a = (entry.get("answer") or "").strip()
    if not q or not a:
        return False, "empty question or answer"

    try:
        at = float(entry["answer_time"])
    except (TypeError, ValueError):
        return False, f"answer_time not a float: {entry['answer_time']!r}"

    if at <= query_time:
        return False, f"answer_time {at:.2f} <= query_time {query_time:.2f}"
    if (at - query_time) < MIN_ANSWER_DELAY_SEC:
        return False, (
            f"answer_time {at:.2f} is only {at - query_time:.2f}s after "
            f"query_time {query_time:.2f}; minimum required delay is "
            f"{MIN_ANSWER_DELAY_SEC:.1f}s"
        )
    if at > clip_end + 0.5:
        return False, f"answer_time {at:.2f} > clip_end {clip_end:.2f}"

    return True, ""


# ============================================================================
# Streaming QA Generation (per anchor)
# ============================================================================

def generate_streaming_qa(video_path, clip, ratio=0.5, *,
                          api_key=None, model=GENERATOR_MODEL,
                          seen_window_sec=SEEN_WINDOW_SEC,
                          future_window_sec=FUTURE_WINDOW_SEC,
                          video_id=None):
    """
    Generate up to 2 streaming QA (intent + next_action) at one anchor.
    Returns (qa_pairs, usage, drop_reasons) where drop_reasons is a list
    of str explaining any dropped entries.
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

    if (fut_b - fut_a) < MIN_FUTURE_SEC_FOR_ANCHOR:
        return [], {}, [
            f"FUTURE too short ({fut_b - fut_a:.1f}s < "
            f"{MIN_FUTURE_SEC_FOR_ANCHOR}s), skipping anchor"
        ]

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
    at_tok = pdetails.get("audio_tokens", 0)
    cost = usage.get("cost")
    cost_s = f"${cost:.6f}" if cost is not None else "n/a"
    print(f"    t={current_time:.0f}s (ratio={ratio}) | "
          f"SEEN {seen_b-seen_a:.0f}s + FUTURE {fut_b-fut_a:.0f}s | "
          f"{elapsed:.1f}s | tot={usage.get('total_tokens')} "
          f"video={vt} audio={at_tok} | cost={cost_s}")

    entries = _parse_json_array(raw)
    if entries is None:
        print(f"    WARNING: failed to parse: {(raw or '')[:240]!r}")
        return [], usage, ["parse_failure"]

    qa_pairs = []
    drop_reasons = []
    for entry in entries:
        ok, reason = _validate_qa_entry(entry, current_time, clip_end)
        if not ok:
            drop_reasons.append(f"{entry.get('type', '?')}: {reason}")
            continue

        query_time = float(entry.get("query_time", current_time))
        answer_time = float(entry["answer_time"])

        qa = {
            "source": "streaming",
            "type": entry["type"],
            "video_id": video_id,
            "clip_idx": clip_idx,
            "clip_start": clip_start,
            "clip_end": clip_end,
            "topic": clip.get("topic", ""),
            "query_time": round(query_time, 2),
            "answer_time": round(answer_time, 2),
            "evidence_window": [round(query_time, 2), round(answer_time, 2)],
            "ratio": ratio,
            "question": entry["question"].strip(),
            "answer": entry["answer"].strip(),
            "evidence": (entry.get("evidence") or "").strip(),
        }
        qa_pairs.append(qa)

    if drop_reasons:
        for r in drop_reasons:
            print(f"    DROPPED: {r}")

    return qa_pairs, usage, drop_reasons


# ============================================================================
# Full pipeline
# ============================================================================

def generate_streaming_qa_for_video(video_path, clips_path,
                                     output_dir="QA/results",
                                     api_key=None, single_clip=None,
                                     time_ratios=None,
                                     model=GENERATOR_MODEL,
                                     seen_window_sec=SEEN_WINDOW_SEC,
                                     future_window_sec=FUTURE_WINDOW_SEC):
    """Generate streaming QA for all clips."""
    video_path = Path(video_path)
    video_id = video_path.stem

    with open(clips_path) as f:
        clips_data = json.load(f)
    clips = clips_data['clips']

    if single_clip is not None:
        clips = [c for c in clips if c['clip_idx'] == single_clip]

    ratios = time_ratios or TIME_RATIOS

    print(f"\n{'='*70}")
    print(f"Streaming QA Generation (new pipeline, with answer_time)")
    print(f"  Video                : {video_id}")
    print(f"  Clips                : {len(clips)}")
    print(f"  Anchors per clip     : {len(ratios)}")
    print(f"  QA types per anchor  : {len(STREAMING_QA_TYPES)}")
    print(f"  Generator            : {model} (OpenRouter)")
    print(f"  SEEN window          : {seen_window_sec}s")
    print(f"  FUTURE window        : {future_window_sec if future_window_sec else 'uncapped'}")
    print(f"{'='*70}")

    all_qa = []
    cost_total = 0.0
    video_token_total = 0
    total_dropped = 0
    drop_log = []

    for clip in clips:
        print(f"\n  Clip {clip['clip_idx']}: "
              f"{clip['start']:.0f}-{clip['end']:.0f}s | "
              f"{clip.get('topic', '')[:40]}")
        for ratio in ratios:
            try:
                qa_pairs, usage, drops = generate_streaming_qa(
                    str(video_path), clip,
                    ratio=ratio,
                    api_key=api_key,
                    model=model,
                    seen_window_sec=seen_window_sec,
                    future_window_sec=future_window_sec,
                    video_id=video_id,
                )
                all_qa.extend(qa_pairs)
                total_dropped += len(drops)
                for r in drops:
                    drop_log.append({
                        "clip_idx": clip['clip_idx'],
                        "ratio": ratio,
                        "reason": r,
                    })

                if usage:
                    c = usage.get('cost')
                    if c is not None:
                        cost_total += c
                    pdetails = usage.get('prompt_tokens_details') or {}
                    vt = pdetails.get('video_tokens', 0) or 0
                    video_token_total += vt
            except Exception as e:
                print(f"    ERROR at ratio={ratio}: {type(e).__name__}: {e}")
                drop_log.append({
                    "clip_idx": clip['clip_idx'],
                    "ratio": ratio,
                    "reason": f"exception: {type(e).__name__}: {e}",
                })
                total_dropped += 1
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
        # renamed from `num_dropped` to align with QA/schema.md §2.
        'num_skipped_bad_answer_time': total_dropped,
        # Not part of schema §2 top-level, but kept for debugging/statistics
        # (see QA/schema.md §2 note on auxiliary fields).
        'drop_log': drop_log,
        'generation_cost_usd': round(cost_total, 6),
        'generation_video_tokens_total': video_token_total,
        'streaming_qa': all_qa,
    }

    out_path = Path(output_dir) / f"{video_id}_streaming_qa.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, 'w', encoding='utf-8') as f:
        json.dump(output, f, ensure_ascii=False, indent=2)

    print(f"\n  Kept {len(all_qa)} streaming QA (unvalidated) | "
          f"dropped {total_dropped}")
    print(f"  Total cost           : ${cost_total:.4f}")
    print(f"  Total video tokens   : {video_token_total:,}")
    print(f"  Saved                : {out_path}")
    print(f"  -> Next: python QA/validator.py --streaming-qa {out_path} "
          f"--video <video>")

    return output


# ============================================================================
# CLI
# ============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Streaming QA Generator (with query_time + answer_time)"
    )
    parser.add_argument("--video", type=str, required=True)
    parser.add_argument("--clips", type=str, required=True)
    parser.add_argument("--output-dir", type=str, default="QA/results")
    parser.add_argument("--api-key", type=str, default=None)
    parser.add_argument("--model", type=str, default=GENERATOR_MODEL)
    parser.add_argument("--single-clip", type=int, default=None)
    parser.add_argument("--ratios", type=str, default=None,
                        help="Comma-separated time ratios, e.g. '0.3,0.6'")
    parser.add_argument("--seen-window-sec", type=float, default=SEEN_WINDOW_SEC)
    parser.add_argument("--future-window-sec", type=float, default=-1,
                        help="Cap FUTURE segment. Default -1 = uncapped.")
    args = parser.parse_args()

    ratios = None
    if args.ratios:
        ratios = [float(x) for x in args.ratios.split(',')]

    seen_w = args.seen_window_sec if args.seen_window_sec and args.seen_window_sec > 0 else None
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
