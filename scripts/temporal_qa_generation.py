"""
Temporal QA Generation for Ultrasound Online Video Understanding
================================================================

Single-stage event-free temporal QA generation.

For each ultrasound clip we make ONE call to a Large Video MLLM with the full
clip mp4 (and optionally the clip-local ASR text). The MLLM directly produces
temporal QA pairs grounded in the clip.

Supported QA types:
    - given_when_ask_what
    - given_what_ask_when
    - visible_anatomy
    - next_action_guidance

Online semantics
----------------
The generator is an oracle and sees the whole clip. Each output QA carries:

    input_video_start = clip_start
    input_video_end   = query_time
    input_policy      = "seen_only"

so a downstream training / evaluation pipeline can enforce streaming
semantics by feeding only video[input_video_start : input_video_end] plus
the question.

ASR is optional everywhere.

CLI
---
    python scripts/temporal_qa_generation.py \\
        --video path/to/video.mp4 \\
        --clips results/clips/8V649L5Q368_clips.json \\
        --clip-idx 1 \\
        --out results/qa/8V649L5Q368_temporal_qa_clip1.json

Notebook
--------
    from scripts.temporal_qa_generation import generate_temporal_qa_for_clip
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
from _temporal_qa_prompts import (
    QA_SYSTEM_PROMPT,
    build_qa_instruction,
)


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

MAX_SEEN_VIDEO_SEC = 240.0
MAX_ASR_CHARS = 3000

SUPPORTED_QA_TYPES = {
    "given_when_ask_what",
    "given_what_ask_when",
    "visible_anatomy",
    "next_action_guidance",
}


# ---------------------------------------------------------------------------
# JSON parsing
# ---------------------------------------------------------------------------

def _parse_json_object(raw: str) -> dict:
    if raw is None:
        raise ValueError("response is None")
    txt = raw.strip()
    fence = re.search(r"```(?:json)?\s*(.*?)```", txt, flags=re.DOTALL | re.IGNORECASE)
    if fence:
        txt = fence.group(1).strip()
    try:
        return json.loads(txt)
    except json.JSONDecodeError:
        pass
    first = txt.find("{")
    last = txt.rfind("}")
    if first != -1 and last != -1 and last > first:
        return json.loads(txt[first:last + 1])
    raise ValueError(f"could not parse JSON object from response: {raw[:300]!r}")


# ---------------------------------------------------------------------------
# Loaders
# ---------------------------------------------------------------------------

def load_clip_from_clips_json(clips_path, clip_idx: int) -> dict:
    with open(clips_path, "r", encoding="utf-8") as f:
        clips_data = json.load(f)
    video_id = clips_data.get("video_id", Path(clips_path).stem)
    clips = clips_data.get("clips") or []
    if clip_idx < 0 or clip_idx >= len(clips):
        raise IndexError(
            f"clip_idx {clip_idx} out of range; clips file has {len(clips)} clips."
        )
    clip = dict(clips[clip_idx])
    clip["video_id"] = video_id
    return clip


def slice_asr_text_for_clip(transcript_path, clip_start: float, clip_end: float) -> str:
    if transcript_path is None:
        return ""
    p = Path(transcript_path)
    if not p.exists():
        return ""
    with open(p, "r", encoding="utf-8") as f:
        data = json.load(f)
    segments = data.get("segments", [])
    parts = []
    for seg in segments:
        start = seg.get("start", 0.0)
        end = seg.get("end", start)
        mid = 0.5 * (start + end)
        if clip_start <= mid <= clip_end:
            t = (seg.get("text") or "").strip()
            if t:
                parts.append(t)
    return " ".join(parts).strip()


# ---------------------------------------------------------------------------
# Sanitisation / annotation
# ---------------------------------------------------------------------------

def _annotate_qa(qa: dict, clip: dict, with_asr: bool) -> Optional[dict]:
    """
    Validate a raw QA dict from the model, attach training/eval policy fields,
    and drop it if it violates the basic temporal constraints.
    """
    qa_type = qa.get("qa_type")
    if qa_type not in SUPPORTED_QA_TYPES:
        return None

    try:
        query_time = float(qa.get("query_time"))
    except (TypeError, ValueError):
        return None

    clip_start = float(clip["start"])
    clip_end = float(clip["end"])
    if not (clip_start - 1e-3 <= query_time <= clip_end + 1e-3):
        return None
    # clamp into the strict range to avoid tiny numerical overflow downstream
    query_time = min(max(query_time, clip_start), clip_end)

    out = {
        "video_id": clip.get("video_id"),
        "clip_idx": clip.get("clip_idx"),
        "clip_start": clip_start,
        "clip_end": clip_end,
        "clip_topic": clip.get("topic", ""),
        "qa_id": qa.get("qa_id"),
        "qa_type": qa_type,
        "query_time": round(query_time, 3),
        "question": (qa.get("question") or "").strip(),
        "answer": (qa.get("answer") or "").strip(),
        "evidence": (qa.get("evidence") or "").strip(),
        # online streaming policy: seen-only training input window
        "input_video_start": clip_start,
        "input_video_end": round(query_time, 3),
        "input_policy": "seen_only",
        # ASR policy
        "asr_policy": "optional_seen_asr",
        "asr_available_at_generation": bool(with_asr),
        "asr_used_in_answer": bool(qa.get("asr_used", False)),
    }
    if not out["question"] or not out["answer"]:
        return None
    return out


# ---------------------------------------------------------------------------
# Single-stage QA generation
# ---------------------------------------------------------------------------

def generate_qa_for_clip(
    *,
    video_path: str | Path,
    clip: dict,
    asr_text: str = "",
    max_qa: int = 10,
    client=None,
    model: str = DEFAULT_MODEL,
    temperature: float = 0.3,
    keep_temp_clip: bool = False,
) -> tuple[list[dict], dict]:
    """
    Call the Large Video MLLM once on the full clip and return (qa, meta).
    """
    if client is None:
        client = build_openrouter_client()

    video_id = clip["video_id"]
    clip_start = float(clip["start"])
    clip_end = float(clip["end"])
    if clip_end - clip_start > MAX_SEEN_VIDEO_SEC:
        clip_end_eff = clip_start + MAX_SEEN_VIDEO_SEC
    else:
        clip_end_eff = clip_end

    clip_mp4 = temp_clip_path(video_id, f"clip{clip.get('clip_idx', 0)}_full")
    cut_clip(video_path, clip_start, clip_end_eff, clip_mp4)

    asr_for_prompt = (asr_text or "").strip()
    if len(asr_for_prompt) > MAX_ASR_CHARS:
        asr_for_prompt = asr_for_prompt[:MAX_ASR_CHARS] + " ..."

    instruction = build_qa_instruction(clip, asr_for_prompt, max_qa=max_qa)
    blocks = [
        text_block(QA_SYSTEM_PROMPT),
        build_video_block(clip_mp4, label=f"{video_id}_clip{clip.get('clip_idx', 0)}.mp4"),
        text_block(instruction),
    ]

    raw_text, usage = call_with_content(
        client,
        content_blocks=blocks,
        model=model,
        temperature=temperature,
    )

    if not keep_temp_clip:
        try:
            Path(clip_mp4).unlink(missing_ok=True)
        except Exception:
            pass

    try:
        obj = _parse_json_object(raw_text)
    except Exception as e:
        obj = {}
        parse_error: Optional[str] = repr(e)
    else:
        parse_error = None

    raw_qa = obj.get("qa_pairs") or []
    with_asr = bool((asr_for_prompt or "").strip()
                    and asr_for_prompt != "(no narration provided)")
    annotated: list[dict] = []
    dropped = 0
    for qa in raw_qa:
        out = _annotate_qa(qa, clip, with_asr=with_asr)
        if out is None:
            dropped += 1
        else:
            annotated.append(out)

    meta = {
        "model": model,
        "usage": usage,
        "clip_start_effective": clip_start,
        "clip_end_effective": clip_end_eff,
        "raw_response": raw_text[:4000] if raw_text else "",
        "raw_qa_count": len(raw_qa),
        "qa_kept": len(annotated),
        "qa_dropped": dropped,
        "parse_error": parse_error,
    }
    return annotated, meta


# ---------------------------------------------------------------------------
# High-level entry: one clip end-to-end
# ---------------------------------------------------------------------------

def generate_temporal_qa_for_clip(
    *,
    video_path: str | Path,
    clip: dict,
    transcript_path: Optional[str | Path] = None,
    max_qa: int = 10,
    model: str = DEFAULT_MODEL,
    output_path: Optional[str | Path] = None,
    client=None,
    temperature: float = 0.3,
    keep_temp_clip: bool = False,
    verbose: bool = True,
) -> dict:
    """
    Run single-stage temporal QA generation for a single clip.

    Parameters
    ----------
    video_path
        Path to the source mp4.
    clip
        Clip dict with at least keys: video_id, clip_idx, start, end, topic.
    transcript_path
        Optional path to results/transcripts/{video_id}.json. ASR is optional.
    max_qa
        Maximum number of QA pairs to request.
    model
        OpenRouter model id (default: google/gemini-2.5-flash).
    output_path
        If given, the result is dumped to this JSON file.
    client
        Optional pre-built OpenRouter client. If None, one is built from
        OPENROUTER_API_KEY in .env.
    """
    if client is None:
        client = build_openrouter_client()

    asr_text = ""
    if transcript_path is not None:
        asr_text = slice_asr_text_for_clip(
            transcript_path, float(clip["start"]), float(clip["end"])
        )

    if verbose:
        print(f"[temporal-qa] clip {clip.get('clip_idx')}: "
              f"{float(clip['start']):.2f}s -> {float(clip['end']):.2f}s "
              f"({float(clip['end']) - float(clip['start']):.1f}s)")
        print(f"[temporal-qa] ASR provided: {bool(asr_text)} "
              f"({len(asr_text)} chars)")

    t0 = time.time()
    qa, qa_meta = generate_qa_for_clip(
        video_path=video_path,
        clip=clip,
        asr_text=asr_text,
        max_qa=max_qa,
        client=client,
        model=model,
        temperature=temperature,
        keep_temp_clip=keep_temp_clip,
    )
    t1 = time.time()
    if verbose:
        print(f"[temporal-qa] QA generated: {len(qa)}  "
              f"(raw {qa_meta.get('raw_qa_count')}, "
              f"dropped {qa_meta.get('qa_dropped')})  "
              f"({t1 - t0:.1f}s)")

    result = {
        "video_id": clip.get("video_id"),
        "clip_idx": clip.get("clip_idx"),
        "clip_start": float(clip["start"]),
        "clip_end": float(clip["end"]),
        "clip_topic": clip.get("topic", ""),
        "model": model,
        "asr_used_at_generation": bool(asr_text),
        "temporal_qa": qa,
        "stage_meta": qa_meta,
        "elapsed_sec": round(t1 - t0, 2),
    }

    if output_path is not None:
        out = Path(output_path)
        out.parent.mkdir(parents=True, exist_ok=True)
        with open(out, "w", encoding="utf-8") as f:
            json.dump(result, f, indent=2, ensure_ascii=False)
        if verbose:
            print(f"[temporal-qa] saved -> {out}")

    return result


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _build_clip_from_args(args) -> dict:
    """Build a clip dict either from --clips/--clip-idx or from explicit args."""
    if args.clips:
        return load_clip_from_clips_json(args.clips, args.clip_idx)
    if args.clip_start is None or args.clip_end is None:
        raise SystemExit(
            "Either --clips + --clip-idx, or --clip-start + --clip-end must be given."
        )
    video_id = args.video_id or Path(args.video).stem
    return {
        "video_id": video_id,
        "clip_idx": args.clip_idx if args.clip_idx is not None else 0,
        "start": float(args.clip_start),
        "end": float(args.clip_end),
        "topic": args.topic or "",
        "text": "",
    }


def main():
    parser = argparse.ArgumentParser(
        description="Generate single-stage temporal QA for one ultrasound clip."
    )
    parser.add_argument("--video", required=True, help="Path to source mp4.")
    parser.add_argument("--clips", default=None,
                        help="Path to results/clips/{video_id}_clips.json.")
    parser.add_argument("--clip-idx", type=int, default=0,
                        help="Index of the clip inside --clips (default 0).")
    parser.add_argument("--clip-start", type=float, default=None,
                        help="(Alternative to --clips) clip start in seconds.")
    parser.add_argument("--clip-end", type=float, default=None,
                        help="(Alternative to --clips) clip end in seconds.")
    parser.add_argument("--video-id", default=None,
                        help="(Alternative to --clips) override video_id.")
    parser.add_argument("--topic", default=None,
                        help="(Alternative to --clips) clip topic string.")
    parser.add_argument("--transcript", default=None,
                        help="Optional ASR transcript JSON.")
    parser.add_argument("--max-qa", type=int, default=10,
                        help="Maximum number of QA pairs to generate.")
    parser.add_argument("--model", default=DEFAULT_MODEL,
                        help="OpenRouter model id.")
    parser.add_argument("--out", default=None, help="Output JSON path.")
    parser.add_argument("--keep-temp-clip", action="store_true",
                        help="Keep the temporary cut mp4 (debug).")
    args = parser.parse_args()

    clip = _build_clip_from_args(args)
    generate_temporal_qa_for_clip(
        video_path=args.video,
        clip=clip,
        transcript_path=args.transcript,
        max_qa=args.max_qa,
        model=args.model,
        output_path=args.out,
        keep_temp_clip=args.keep_temp_clip,
        verbose=True,
    )


if __name__ == "__main__":
    main()
