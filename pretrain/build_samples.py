#!/usr/bin/env python3
"""
Build ultrasound ASR caption-completion pretraining samples.

Input:
  A directory of ASR transcripts produced by QA/prepare/asr.py, i.e.
  {transcripts}/{video_id}.json with structure:

    {
      "video_id": "...",
      "duration_sec": 123.4,
      "segments": [{"start": 4.3, "end": 12.3, "text": "..."}, ...],
      ...
    }

Task (whole-sentence completion):
  For each ASR segment (one narration sentence):
    current_time  = segment.start
    video_window  = [max(0, start - window_sec), start]   # last frames before it
    prev_context  = concatenation of previous sentences (optional, truncated)
    target        = segment.text                           # continue this sentence

Output:
  pretrain_samples.jsonl, one sample per line:

    {
      "sample_type": "pretrain_caption",
      "video_id": "...",
      "video_window": [start_minus, start],
      "prev_context": "...",
      "target": "...",
      "meta": {"segment_idx": i, "seg_start": ..., "seg_end": ...}
    }

Example:
  python pretrain/build_samples.py \
    --transcripts QA/results/transcripts \
    --output pretrain/data/pretrain_samples.jsonl \
    --window-sec 8 \
    --min-words 3 \
    --context-max-chars 400
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any, Dict, List


_BRACKET_RE = re.compile(r"[\[\(](music|applause|laughter|inaudible|noise)[\]\)]", re.IGNORECASE)


def is_bad_text(text: str, min_words: int) -> bool:
    t = (text or "").strip()
    if not t:
        return True
    if _BRACKET_RE.search(t):
        return True
    if "[" in t and "]" in t and len(t) < 30:
        return True
    if len(t.split()) < min_words:
        return True
    return False


def build_prev_context(segments: List[Dict[str, Any]], idx: int, max_chars: int) -> str:
    if max_chars <= 0 or idx <= 0:
        return ""
    parts = []
    total = 0
    # Walk backward from previous sentence, then reverse for chronological order.
    for k in range(idx - 1, -1, -1):
        txt = (segments[k].get("text") or "").strip()
        if not txt:
            continue
        if total + len(txt) + 1 > max_chars:
            break
        parts.append(txt)
        total += len(txt) + 1
    parts.reverse()
    return " ".join(parts).strip()


def build_samples_for_video(
    transcript: Dict[str, Any],
    *,
    window_sec: float,
    min_words: int,
    context_max_chars: int,
    use_context: bool,
) -> List[Dict[str, Any]]:
    video_id = transcript.get("video_id")
    segments = transcript.get("segments", [])
    samples: List[Dict[str, Any]] = []

    for idx, seg in enumerate(segments):
        text = (seg.get("text") or "").strip()
        if is_bad_text(text, min_words):
            continue

        seg_start = float(seg.get("start", 0.0))
        seg_end = float(seg.get("end", seg_start))
        window_start = max(0.0, seg_start - float(window_sec))

        prev_context = ""
        if use_context:
            prev_context = build_prev_context(segments, idx, context_max_chars)

        samples.append({
            "sample_type": "pretrain_caption",
            "video_id": video_id,
            "video_window": [round(window_start, 2), round(seg_start, 2)],
            "prev_context": prev_context,
            "target": text,
            "meta": {
                "segment_idx": idx,
                "seg_start": round(seg_start, 2),
                "seg_end": round(seg_end, 2),
            },
        })

    return samples


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build ultrasound ASR caption-completion pretraining samples")
    parser.add_argument("--transcripts", type=str, required=True,
                        help="Directory of ASR transcript JSON files ({video_id}.json).")
    parser.add_argument("--output", type=str, required=True,
                        help="Output jsonl path.")
    parser.add_argument("--window-sec", type=float, default=8.0,
                        help="Seconds of frames before segment.start to look at.")
    parser.add_argument("--min-words", type=int, default=3,
                        help="Skip narration sentences shorter than this many words.")
    parser.add_argument("--context-max-chars", type=int, default=400,
                        help="Max characters of previous narration used as prev_context.")
    parser.add_argument("--no-context", action="store_true",
                        help="Disable prev_context (pure visual completion).")
    parser.add_argument("--limit-videos", type=int, default=None,
                        help="Only process this many transcript files.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    transcripts_dir = Path(args.transcripts)
    if not transcripts_dir.exists():
        raise FileNotFoundError(f"Transcripts dir not found: {transcripts_dir}")

    json_files = sorted(transcripts_dir.glob("*.json"))
    if args.limit_videos is not None:
        json_files = json_files[: args.limit_videos]

    if not json_files:
        raise ValueError(f"No transcript JSON files found in {transcripts_dir}")

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    use_context = not args.no_context

    total_samples = 0
    per_video_counts = {}

    with out_path.open("w", encoding="utf-8") as f:
        for jf in json_files:
            try:
                transcript = json.loads(jf.read_text(encoding="utf-8"))
            except Exception as e:
                print(f"[build] WARNING: failed to read {jf}: {e}")
                continue

            samples = build_samples_for_video(
                transcript,
                window_sec=args.window_sec,
                min_words=args.min_words,
                context_max_chars=args.context_max_chars,
                use_context=use_context,
            )

            for s in samples:
                f.write(json.dumps(s, ensure_ascii=False) + "\n")

            vid = transcript.get("video_id", jf.stem)
            per_video_counts[vid] = len(samples)
            total_samples += len(samples)
            print(f"[build] {vid}: {len(samples)} samples")

    print("=" * 60)
    print(f"[build] videos: {len(per_video_counts)}")
    print(f"[build] total samples: {total_samples}")
    print(f"[build] output: {out_path}")


if __name__ == "__main__":
    main()