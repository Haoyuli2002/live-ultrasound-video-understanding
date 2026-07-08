"""
Dual QA Generation for Ultrasound Video Understanding
=====================================================

For each ultrasound clip, generate exactly:
  - 1 best Offline QA, selected from:
      scene_description, key_findings, clinical_knowledge
  - 1 best Streaming QA, selected from:
      given_when_ask_what, given_what_ask_when, visible_anatomy, next_action_guidance

The generator sees the full clip as an oracle. Streaming QA output still records
question_time and answer_time so downstream training/evaluation can enforce
seen-only / progressive semantics.

Example:
    python scripts/dual_qa_generation.py \
      --video UltrasoundCrawler_KeyCode_20260323_v2/output/20260520_162816_youtube/media/case_reasoning/8V649L5Q368.mp4 \
      --clips results/clips/8V649L5Q368_clips.json \
      --clip-idx 1 \
      --transcript results/transcripts/8V649L5Q368.json \
      --out results/qa/8V649L5Q368_dual_qa_clip1.json \
      --model google/gemini-2.5-pro
"""

from __future__ import annotations

import argparse
import json
import re
import sys
import time
from pathlib import Path
from typing import Optional

_HERE = Path(__file__).resolve().parent
if str(_HERE) not in sys.path:
    sys.path.insert(0, str(_HERE))

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


OFFLINE_TYPES = {"scene_description", "key_findings", "clinical_knowledge"}
STREAMING_TYPES = {
    "given_when_ask_what",
    "given_what_ask_when",
    "visible_anatomy",
    "next_action_guidance",
}

MAX_CLIP_SEC = 240.0
MAX_ASR_CHARS = 3500
DEFAULT_DUAL_QA_MODEL = "google/gemini-2.5-pro"


# ---------------------------------------------------------------------------
# Loading helpers
# ---------------------------------------------------------------------------

def load_clip_from_clips_json(clips_path: str | Path, clip_idx: int) -> dict:
    with open(clips_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    clips = data.get("clips") or []
    if clip_idx < 0 or clip_idx >= len(clips):
        raise IndexError(f"clip_idx={clip_idx} out of range; file has {len(clips)} clips")

    clip = dict(clips[clip_idx])
    clip["video_id"] = data.get("video_id", Path(clips_path).stem)
    clip["video_path"] = data.get("video_path")
    return clip


def slice_asr_text_for_clip(transcript_path: Optional[str | Path], start: float, end: float) -> str:
    if transcript_path is None:
        return ""
    p = Path(transcript_path)
    if not p.exists():
        return ""

    with p.open("r", encoding="utf-8") as f:
        data = json.load(f)

    parts: list[str] = []
    for seg in data.get("segments", []):
        s = float(seg.get("start", 0.0))
        e = float(seg.get("end", s))
        mid = 0.5 * (s + e)
        if start <= mid <= end:
            txt = (seg.get("text") or "").strip()
            if txt:
                parts.append(txt)
    return " ".join(parts)


# ---------------------------------------------------------------------------
# Parsing / validation
# ---------------------------------------------------------------------------

def parse_json_object(raw: str) -> dict:
    txt = (raw or "").strip()

    fence = re.search(r"```(?:json)?\s*(.*?)```", txt, flags=re.DOTALL | re.IGNORECASE)
    if fence:
        txt = fence.group(1).strip()

    try:
        return json.loads(txt)
    except json.JSONDecodeError:
        pass

    first = txt.find("{")
    last = txt.rfind("}")
    if first >= 0 and last > first:
        return json.loads(txt[first:last + 1])

    raise ValueError(f"Could not parse JSON object from response: {raw[:300]!r}")


def _clamp_time(x, lo: float, hi: float) -> Optional[float]:
    try:
        v = float(x)
    except (TypeError, ValueError):
        return None
    if v < lo - 1e-3 or v > hi + 1e-3:
        return None
    return round(min(max(v, lo), hi), 3)


def normalize_offline_qa(raw: dict, clip: dict) -> Optional[dict]:
    qa_type = raw.get("qa_type")
    if qa_type not in OFFLINE_TYPES:
        return None

    question = (raw.get("question") or "").strip()
    answer = (raw.get("answer") or "").strip()
    if not question or not answer:
        return None

    clip_start = float(clip["start"])
    clip_end = float(clip["end"])

    return {
        "qa_id": f"{clip['video_id']}_c{clip.get('clip_idx')}_offline_001",
        "source": "offline",
        "qa_type": qa_type,
        "input_video_start": clip_start,
        "input_video_end": clip_end,
        "input_policy": "full_clip",
        "asr_policy": "optional_full_asr",
        "asr_used": bool(raw.get("asr_used", False)),
        "question": question,
        "answer": answer,
        "evidence": (raw.get("evidence") or "").strip(),
        "why_this_qa_type": (raw.get("why_this_qa_type") or "").strip(),
    }


def normalize_streaming_qa(raw: dict, clip: dict, *, window_sec: float, fps: int, max_frames: int) -> Optional[dict]:
    qa_type = raw.get("qa_type")
    if qa_type not in STREAMING_TYPES:
        return None

    clip_start = float(clip["start"])
    clip_end = float(clip["end"])

    question_time = _clamp_time(raw.get("question_time"), clip_start, clip_end)
    answer_time = _clamp_time(raw.get("answer_time"), clip_start, clip_end)
    if question_time is None or answer_time is None:
        return None
    if answer_time < question_time:
        return None

    question = (raw.get("question") or "").strip()
    answer = (raw.get("answer") or "").strip()
    if not question or not answer:
        return None

    vw_end = answer_time
    vw_start = round(max(clip_start, answer_time - window_sec), 3)

    return {
        "qa_id": f"{clip['video_id']}_c{clip.get('clip_idx')}_stream_001",
        "source": "streaming",
        "qa_type": qa_type,
        "clip_start": clip_start,
        "clip_end": clip_end,
        "question_time": question_time,
        "answer_time": answer_time,
        "answerable_at_question_time": bool(raw.get("answerable_at_question_time", answer_time == question_time)),
        "needs_more_context": bool(raw.get("needs_more_context", answer_time > question_time)),
        "visual_context": {
            "type": "sliding_window",
            "start": vw_start,
            "end": vw_end,
            "duration_sec": window_sec,
            "anchor": "answer_time",
            "fps": fps,
            "max_frames": max_frames,
        },
        "memory": raw.get("memory", {
            "type": "none",
            "top_k": 0,
            "items": [],
        }),
        "history_summary": (raw.get("history_summary") or "").strip(),
        "history_summary_source": raw.get("history_summary_source", "none"),
        "question": question,
        "answer": answer,
        "evidence_window": raw.get("evidence_window", [question_time, answer_time]),
        "evidence": (raw.get("evidence") or "").strip(),
        "asr_policy": "optional_seen_asr",
        "asr_used": bool(raw.get("asr_used", False)),
        "why_this_qa_type": (raw.get("why_this_qa_type") or "").strip(),
    }


# ---------------------------------------------------------------------------
# Prompt
# ---------------------------------------------------------------------------

DUAL_QA_SYSTEM_PROMPT = """\
You are an expert ultrasound educator and dataset curator.

You will receive one full ultrasound video clip and optional ASR narration.
Generate exactly:
1. ONE best Offline QA.
2. ONE best Streaming QA.

Offline QA:
- Choose exactly one qa_type from:
  - scene_description
  - key_findings
  - clinical_knowledge
- Choose the qa_type that produces the most meaningful full-clip question.
- The answer may use the entire clip.

Streaming QA:
- Choose exactly one qa_type from:
  - given_when_ask_what
  - given_what_ask_when
  - visible_anatomy
  - next_action_guidance
- Choose the qa_type that produces the most clinically meaningful, visually grounded,
  and temporally interesting streaming question for this clip.
- Prefer answerability-aware examples:
  - Choose a question_time BEFORE the key evidence is fully visible whenever possible.
  - Set answerable_at_question_time = false when the evidence is not yet sufficient.
  - Set answer_time to the earliest moment when enough visual evidence becomes available.
  - Avoid choosing question_time equal to answer_time unless the clip has no meaningful
    waiting opportunity.
- The final answer must not rely on information after answer_time.

Streaming qa_type definitions:
- given_when_ask_what:
  Ask what is visually happening at a specific time or time window.
  Do NOT ask for clinical significance.
- given_what_ask_when:
  Ask when a described visible event occurs.
  The answer should include a concrete absolute time range.
- visible_anatomy:
  Ask which anatomy, landmarks, probe-view structures, or artifacts become visible.
- next_action_guidance:
  Ask what the sonographer should do next based on the evidence available by answer_time.
  The answer must recommend a concrete next step, such as repositioning the probe,
  scanning additional lung zones, looking for another sign, repeating the view, or
  correlating with clinical context.
  Do NOT merely state the clinical concern.
  The answer must be cautious and should not over-diagnose.

Rules:
- All time fields and all time references in evidence must use absolute seconds
  from the original full video, NOT clip-relative timestamps such as 0:56.
- clip_start <= question_time <= answer_time <= clip_end.
- Use cautious medical language.
- Do not make definitive diagnoses unless directly supported by visible evidence.
- Do not hallucinate anatomy or pathology.
- Return JSON only.
"""


def build_dual_qa_prompt(clip: dict, asr_text: str, *, window_sec: float, fps: int, max_frames: int) -> str:
    asr_block = asr_text.strip() if asr_text.strip() else "(no ASR provided)"
    return f"""\
Clip metadata
-------------
video_id   : {clip.get('video_id')}
clip_idx   : {clip.get('clip_idx')}
clip_start : {float(clip['start']):.2f}s
clip_end   : {float(clip['end']):.2f}s
topic      : {clip.get('topic', '')}

Streaming visual context defaults
---------------------------------
window_sec : {window_sec}
fps        : {fps}
max_frames : {max_frames}

Optional ASR
------------
{asr_block}

Return exactly this JSON schema:

{{
  "offline_qa": {{
    "qa_type": "scene_description | key_findings | clinical_knowledge",
    "question": "...",
    "answer": "...",
    "evidence": "...",
    "asr_used": true,
    "why_this_qa_type": "..."
  }},
  "streaming_qa": {{
    "qa_type": "given_when_ask_what | given_what_ask_when | visible_anatomy | next_action_guidance",
    "question_time": 0.0,
    "answer_time": 0.0,
    "answerable_at_question_time": false,
    "needs_more_context": true,
    "question": "...",
    "answer": "...",
    "evidence_window": [0.0, 0.0],
    "evidence": "...",
    "asr_used": true,
    "why_this_qa_type": "...",
    "history_summary": "",
    "history_summary_source": "none | asr_or_visual_summary",
    "memory": {{
      "type": "none",
      "top_k": 0,
      "items": []
    }}
  }}
}}
"""


# ---------------------------------------------------------------------------
# Generation
# ---------------------------------------------------------------------------

def generate_dual_qa_for_clip(
    *,
    video_path: str | Path,
    clip: dict,
    transcript_path: Optional[str | Path] = None,
    output_path: Optional[str | Path] = None,
    model: str = DEFAULT_DUAL_QA_MODEL,
    client=None,
    temperature: float = 0.2,
    window_sec: float = 30.0,
    fps: int = 1,
    max_frames: int = 32,
    keep_temp_clip: bool = False,
    verbose: bool = True,
) -> dict:
    if client is None:
        client = build_openrouter_client()

    clip_start = float(clip["start"])
    clip_end = float(clip["end"])
    clip_end_eff = min(clip_end, clip_start + MAX_CLIP_SEC)

    video_id = clip["video_id"]
    clip_idx = clip.get("clip_idx", 0)

    asr_text = slice_asr_text_for_clip(transcript_path, clip_start, clip_end_eff)
    if len(asr_text) > MAX_ASR_CHARS:
        asr_text = asr_text[:MAX_ASR_CHARS] + " ..."

    if verbose:
        print(f"[dual-qa] clip {clip_idx}: {clip_start:.2f}s -> {clip_end_eff:.2f}s")
        print(f"[dual-qa] ASR: {bool(asr_text)} ({len(asr_text)} chars)")

    clip_mp4 = temp_clip_path(video_id, f"dualqa_clip{clip_idx}_full")
    cut_clip(video_path, clip_start, clip_end_eff, clip_mp4)

    prompt = build_dual_qa_prompt(
        clip={**clip, "end": clip_end_eff},
        asr_text=asr_text,
        window_sec=window_sec,
        fps=fps,
        max_frames=max_frames,
    )

    t0 = time.time()
    raw_text, usage = call_with_content(
        client,
        content_blocks=[
            text_block(DUAL_QA_SYSTEM_PROMPT),
            build_video_block(clip_mp4, label=f"{video_id}_clip{clip_idx}.mp4"),
            text_block(prompt),
        ],
        model=model,
        temperature=temperature,
    )
    elapsed = time.time() - t0

    if not keep_temp_clip:
        try:
            Path(clip_mp4).unlink(missing_ok=True)
        except Exception:
            pass

    parse_error = None
    try:
        obj = parse_json_object(raw_text)
    except Exception as e:
        obj = {}
        parse_error = repr(e)

    offline = normalize_offline_qa(obj.get("offline_qa") or {}, {**clip, "end": clip_end_eff})
    streaming = normalize_streaming_qa(
        obj.get("streaming_qa") or {},
        {**clip, "end": clip_end_eff},
        window_sec=window_sec,
        fps=fps,
        max_frames=max_frames,
    )

    result = {
        "video_id": video_id,
        "video_path": str(video_path),
        "clip_idx": clip_idx,
        "clip_start": clip_start,
        "clip_end": clip_end_eff,
        "duration": round(clip_end_eff - clip_start, 3),
        "topic": clip.get("topic", ""),
        "asr_available": bool(asr_text),
        "offline_qa": [offline] if offline else [],
        "streaming_qa": [streaming] if streaming else [],
        "generation_meta": {
            "version": "dual_qa_v1",
            "generator_model": model,
            "usage": usage,
            "elapsed_sec": round(elapsed, 2),
            "parse_error": parse_error,
            "raw_response": raw_text[:4000] if raw_text else "",
        },
    }

    if output_path:
        out = Path(output_path)
        out.parent.mkdir(parents=True, exist_ok=True)
        with out.open("w", encoding="utf-8") as f:
            json.dump(result, f, indent=2, ensure_ascii=False)
        if verbose:
            print(f"[dual-qa] saved -> {out}")

    if verbose:
        print(f"[dual-qa] offline QA: {len(result['offline_qa'])}")
        print(f"[dual-qa] streaming QA: {len(result['streaming_qa'])}")

    return result


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def build_clip_from_args(args) -> dict:
    if args.clips:
        return load_clip_from_clips_json(args.clips, args.clip_idx)

    if args.clip_start is None or args.clip_end is None:
        raise SystemExit("Need --clips + --clip-idx OR --clip-start + --clip-end")

    return {
        "video_id": args.video_id or Path(args.video).stem,
        "video_path": str(args.video),
        "clip_idx": args.clip_idx,
        "start": float(args.clip_start),
        "end": float(args.clip_end),
        "duration": float(args.clip_end) - float(args.clip_start),
        "topic": args.topic or "",
    }


def main():
    parser = argparse.ArgumentParser(description="Generate 1 best offline QA + 1 best streaming QA for one clip.")
    parser.add_argument("--video", required=True, help="Path to source mp4")
    parser.add_argument("--clips", default=None, help="Path to clips JSON")
    parser.add_argument("--clip-idx", type=int, default=0)
    parser.add_argument("--clip-start", type=float, default=None)
    parser.add_argument("--clip-end", type=float, default=None)
    parser.add_argument("--video-id", default=None)
    parser.add_argument("--topic", default=None)
    parser.add_argument("--transcript", default=None, help="Optional transcript JSON")
    parser.add_argument("--out", default=None, help="Output JSON path")
    parser.add_argument("--model", default=DEFAULT_DUAL_QA_MODEL)
    parser.add_argument("--window-sec", type=float, default=30.0)
    parser.add_argument("--fps", type=int, default=1)
    parser.add_argument("--max-frames", type=int, default=32)
    parser.add_argument("--keep-temp-clip", action="store_true")
    args = parser.parse_args()

    clip = build_clip_from_args(args)
    generate_dual_qa_for_clip(
        video_path=args.video,
        clip=clip,
        transcript_path=args.transcript,
        output_path=args.out,
        model=args.model,
        window_sec=args.window_sec,
        fps=args.fps,
        max_frames=args.max_frames,
        keep_temp_clip=args.keep_temp_clip,
    )


if __name__ == "__main__":
    main()