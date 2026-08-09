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

Supported sample units:
  1. segment:
     One sample per qualified ASR segment.

  2. sentence:
     Consecutive ASR segments are merged until a complete sentence is formed.
     This avoids targets such as "... Massachusetts General" followed by
     "Hospital. ..." being split across two samples.

Task:
  For each unit (segment or sentence):
    current_time  = unit.start
    video_window  = [max(0, start - window_sec), start]   # last frames before it
    prev_context  = concatenation of previous units (optional, truncated)
    target        = unit.text                             # continue this narration unit

Output:
  pretrain_samples.jsonl, one sample per line:

    {
      "sample_type": "pretrain_caption",
      "video_id": "...",
      "video_window": [start_minus, start],
      "prev_context": "...",
      "target": "...",
      "meta": {"unit": "segment", "segment_idx": i, "seg_start": ..., "seg_end": ...}
    }

Example:
  python pretrain/build_samples.py \
    --transcripts QA/results/transcripts \
    --output pretrain/data/pretrain_samples.jsonl \
    --window-sec 8 \
    --min-words 3 \
    --context-max-chars 400 \
    --unit sentence
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any, Dict, List


_BRACKET_RE = re.compile(r"[\[\(](music|applause|laughter|inaudible|noise)[\]\)]", re.IGNORECASE)
_SENTENCE_END_RE = re.compile(r"[.!?。？！][\"'”’)\]]*$")


def normalize_text(text: str) -> str:
    return re.sub(r"\s+", " ", (text or "").strip())


def is_noise_text(text: str) -> bool:
    t = normalize_text(text)
    if not t:
        return True
    if _BRACKET_RE.search(t):
        return True
    if "[" in t and "]" in t and len(t) < 30:
        return True
    return False


def is_bad_text(text: str, min_words: int) -> bool:
    t = normalize_text(text)
    if is_noise_text(t):
        return True
    if len(t.split()) < min_words:
        return True
    return False


def ends_sentence(text: str) -> bool:
    return bool(_SENTENCE_END_RE.search(normalize_text(text)))


def build_prev_context(units: List[Dict[str, Any]], idx: int, max_chars: int) -> str:
    """Build previous-unit context with tail truncation.

    The previous implementation dropped an entire previous unit when that unit
    was longer than ``max_chars``. Sentence-level samples often have long
    previous sentences, so that behavior produced empty ``prev_context`` fields.
    This implementation always preserves the most recent context by returning
    the final ``max_chars`` characters of the accumulated previous text.
    """
    if max_chars <= 0 or idx <= 0:
        return ""

    parts = []
    # Walk backward from previous unit, then reverse for chronological order.
    for k in range(idx - 1, -1, -1):
        txt = normalize_text(units[k].get("text") or "")
        if not txt:
            continue
        parts.append(txt)

        candidate = " ".join(reversed(parts)).strip()
        if len(candidate) >= max_chars:
            return candidate[-max_chars:].lstrip()

    return " ".join(reversed(parts)).strip()


def build_sentence_units(
    segments: List[Dict[str, Any]],
    *,
    min_words: int,
    sentence_max_words: int,
    sentence_max_duration: float,
) -> List[Dict[str, Any]]:
    """Merge ASR segments into sentence-like units.

    We keep ASR timing by assigning the sentence start to the first contributing
    segment and the sentence end to the last contributing segment. If ASR lacks
    punctuation, a fallback split is triggered by max words or max duration.
    """
    units: List[Dict[str, Any]] = []
    cur_parts: List[str] = []
    cur_start: float | None = None
    cur_end: float | None = None
    cur_segment_start_idx: int | None = None
    cur_segment_end_idx: int | None = None

    def flush(force: bool = False) -> None:
        nonlocal cur_parts, cur_start, cur_end, cur_segment_start_idx, cur_segment_end_idx

        text = normalize_text(" ".join(cur_parts))
        if not text:
            cur_parts = []
            cur_start = None
            cur_end = None
            cur_segment_start_idx = None
            cur_segment_end_idx = None
            return

        if force or not is_bad_text(text, min_words):
            units.append({
                "text": text,
                "start": float(cur_start or 0.0),
                "end": float(cur_end if cur_end is not None else (cur_start or 0.0)),
                "segment_start_idx": cur_segment_start_idx,
                "segment_end_idx": cur_segment_end_idx,
            })

        cur_parts = []
        cur_start = None
        cur_end = None
        cur_segment_start_idx = None
        cur_segment_end_idx = None

    for idx, seg in enumerate(segments):
        text = normalize_text(seg.get("text") or "")
        if is_noise_text(text):
            continue

        seg_start = float(seg.get("start", 0.0))
        seg_end = float(seg.get("end", seg_start))

        if cur_start is None:
            cur_start = seg_start
            cur_segment_start_idx = idx

        cur_parts.append(text)
        cur_end = seg_end
        cur_segment_end_idx = idx

        merged = normalize_text(" ".join(cur_parts))
        word_count = len(merged.split())
        duration = float(cur_end - (cur_start or 0.0))

        should_flush = (
            ends_sentence(merged)
            or (sentence_max_words > 0 and word_count >= sentence_max_words)
            or (sentence_max_duration > 0 and duration >= sentence_max_duration)
        )

        if should_flush:
            flush(force=False)

    # Keep the final leftover only if it is long enough to be useful.
    if cur_parts:
        flush(force=False)

    return units


def _split_long_text_by_words(text: str, max_words: int) -> List[str]:
    """Split a long punctuation chunk by word count as a fallback."""
    text = normalize_text(text)
    if max_words <= 0:
        return [text] if text else []

    words = text.split()
    if len(words) <= max_words:
        return [text] if text else []

    chunks = []
    for i in range(0, len(words), max_words):
        chunks.append(" ".join(words[i:i + max_words]).strip())
    return [c for c in chunks if c]


def _segment_spans_for_full_text(segments: List[Dict[str, Any]]) -> tuple[str, List[Dict[str, Any]]]:
    """Concatenate ASR segment text and keep char/time spans for alignment."""
    parts: List[str] = []
    spans: List[Dict[str, Any]] = []
    cursor = 0

    for idx, seg in enumerate(segments):
        text = normalize_text(seg.get("text") or "")
        if is_noise_text(text):
            continue

        if parts:
            parts.append(" ")
            cursor += 1

        start_char = cursor
        parts.append(text)
        cursor += len(text)
        end_char = cursor

        seg_start = float(seg.get("start", 0.0))
        seg_end = float(seg.get("end", seg_start))
        spans.append({
            "segment_idx": idx,
            "char_start": start_char,
            "char_end": end_char,
            "time_start": seg_start,
            "time_end": seg_end,
        })

    return "".join(parts), spans


def _align_char_range_to_segments(
    spans: List[Dict[str, Any]],
    char_start: int,
    char_end: int,
) -> Dict[str, Any] | None:
    overlaps = [
        span for span in spans
        if span["char_end"] > char_start and span["char_start"] < char_end
    ]
    if not overlaps:
        return None

    return {
        "start": float(overlaps[0]["time_start"]),
        "end": float(overlaps[-1]["time_end"]),
        "segment_start_idx": overlaps[0]["segment_idx"],
        "segment_end_idx": overlaps[-1]["segment_idx"],
    }


def build_punctuation_sentence_units(
    segments: List[Dict[str, Any]],
    *,
    min_words: int,
    split_punctuation: str,
    include_comma_split: bool,
    sentence_max_words: int,
) -> List[Dict[str, Any]]:
    """Build sentence-like units by splitting the full ASR text by punctuation.

    Unlike segment-merge mode, this first concatenates all ASR segment text and
    then splits the full text. Each resulting text span is mapped back to the
    ASR segments it overlaps to recover a time window.
    """
    full_text, spans = _segment_spans_for_full_text(segments)
    if not full_text:
        return []

    punctuation = set(split_punctuation or "")
    if include_comma_split:
        punctuation.update({",", "，"})
    if not punctuation:
        punctuation = set(".?!;:。？！；：")

    raw_ranges: List[tuple[int, int]] = []
    start = 0
    for idx, ch in enumerate(full_text):
        if ch in punctuation:
            end = idx + 1
            raw_ranges.append((start, end))
            start = end
            while start < len(full_text) and full_text[start].isspace():
                start += 1
    if start < len(full_text):
        raw_ranges.append((start, len(full_text)))

    units: List[Dict[str, Any]] = []
    for char_start, char_end in raw_ranges:
        text = normalize_text(full_text[char_start:char_end])
        if not text:
            continue

        sub_chunks = _split_long_text_by_words(text, sentence_max_words)
        if len(sub_chunks) == 1:
            chunk_ranges = [(char_start, char_end, sub_chunks[0])]
        else:
            chunk_ranges = []
            search_from = char_start
            for chunk in sub_chunks:
                local = full_text.find(chunk, search_from, char_end)
                if local < 0:
                    # Fallback: approximate from the current cursor.
                    local = search_from
                chunk_ranges.append((local, min(local + len(chunk), char_end), chunk))
                search_from = min(local + len(chunk), char_end)

        for sub_start, sub_end, sub_text in chunk_ranges:
            if is_bad_text(sub_text, min_words):
                continue
            aligned = _align_char_range_to_segments(spans, sub_start, sub_end)
            if aligned is None:
                continue
            units.append({
                "text": sub_text,
                "start": aligned["start"],
                "end": aligned["end"],
                "segment_start_idx": aligned["segment_start_idx"],
                "segment_end_idx": aligned["segment_end_idx"],
            })

    return units


def build_segment_samples_for_video(
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
        text = normalize_text(seg.get("text") or "")
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
                "unit": "segment",
                "segment_idx": idx,
                "seg_start": round(seg_start, 2),
                "seg_end": round(seg_end, 2),
            },
        })

    return samples


def build_sentence_samples_for_video(
    transcript: Dict[str, Any],
    *,
    window_sec: float,
    min_words: int,
    context_max_chars: int,
    use_context: bool,
    sentence_max_words: int,
    sentence_max_duration: float,
    sample_format: str,
    history_units: int,
    frames_per_sentence: int,
    sentence_mode: str,
    split_punctuation: str,
    include_comma_split: bool,
) -> List[Dict[str, Any]]:
    video_id = transcript.get("video_id")
    segments = transcript.get("segments", [])

    if sentence_mode == "segment_merge":
        sentence_units = build_sentence_units(
            segments,
            min_words=min_words,
            sentence_max_words=sentence_max_words,
            sentence_max_duration=sentence_max_duration,
        )
    elif sentence_mode == "punctuation":
        sentence_units = build_punctuation_sentence_units(
            segments,
            min_words=min_words,
            split_punctuation=split_punctuation,
            include_comma_split=include_comma_split,
            sentence_max_words=sentence_max_words,
        )
    else:
        raise ValueError(f"Unsupported sentence mode: {sentence_mode}")

    samples: List[Dict[str, Any]] = []

    if sample_format == "interleave":
        if history_units <= 0:
            raise ValueError("--history-units must be positive for --format interleave")
        if frames_per_sentence <= 0:
            raise ValueError("--frames-per-sentence must be positive for --format interleave")

        # Autoregressive interleaved format:
        #   sentence_1 frames + sentence_1 -> predict sentence_2
        #   sentence_1 frames + sentence_1 + sentence_2 frames + sentence_2 -> predict sentence_3
        # with a sliding history window of at most `history_units` sentences.
        for idx in range(1, len(sentence_units)):
            unit = sentence_units[idx]
            text = normalize_text(unit.get("text") or "")
            if is_bad_text(text, min_words):
                continue

            history_start = max(0, idx - history_units)
            history = []
            for h_idx in range(history_start, idx):
                h = sentence_units[h_idx]
                h_text = normalize_text(h.get("text") or "")
                if not h_text:
                    continue
                history.append({
                    "sentence_idx": h_idx,
                    "text": h_text,
                    "video_window": [round(float(h.get("start", 0.0)), 2), round(float(h.get("end", h.get("start", 0.0))), 2)],
                    "num_frames": frames_per_sentence,
                    "segment_start_idx": h.get("segment_start_idx"),
                    "segment_end_idx": h.get("segment_end_idx"),
                })

            if not history:
                continue

            sent_start = float(unit.get("start", 0.0))
            sent_end = float(unit.get("end", sent_start))

            samples.append({
                "sample_type": "pretrain_caption_sentence_interleave",
                "video_id": video_id,
                "history": history,
                "video_window": [round(sent_start, 2), round(sent_end, 2)],
                "prev_context": " ".join(h["text"] for h in history).strip() if use_context else "",
                "target": text,
                "meta": {
                    "unit": "sentence",
                    "format": "interleave",
                    "target_sentence_idx": idx,
                    "target_sentence_start": round(sent_start, 2),
                    "target_sentence_end": round(sent_end, 2),
                    "history_units": len(history),
                    "frames_per_sentence": frames_per_sentence,
                    "segment_start_idx": unit.get("segment_start_idx"),
                    "segment_end_idx": unit.get("segment_end_idx"),
                },
            })

        return samples

    if sample_format != "standard":
        raise ValueError(f"Unsupported sample format for sentence unit: {sample_format}")

    for idx, unit in enumerate(sentence_units):
        text = normalize_text(unit.get("text") or "")
        if is_bad_text(text, min_words):
            continue

        sent_start = float(unit.get("start", 0.0))
        sent_end = float(unit.get("end", sent_start))
        window_start = max(0.0, sent_start - float(window_sec))

        prev_context = ""
        if use_context:
            prev_context = build_prev_context(sentence_units, idx, context_max_chars)

        samples.append({
            "sample_type": "pretrain_caption_sentence",
            "video_id": video_id,
            "video_window": [round(window_start, 2), round(sent_start, 2)],
            "prev_context": prev_context,
            "target": text,
            "meta": {
                "unit": "sentence",
                "format": "standard",
                "sentence_idx": idx,
                "sentence_start": round(sent_start, 2),
                "sentence_end": round(sent_end, 2),
                "segment_start_idx": unit.get("segment_start_idx"),
                "segment_end_idx": unit.get("segment_end_idx"),
            },
        })

    return samples


def build_samples_for_video(
    transcript: Dict[str, Any],
    *,
    window_sec: float,
    min_words: int,
    context_max_chars: int,
    use_context: bool,
    unit: str,
    sentence_max_words: int,
    sentence_max_duration: float,
    sample_format: str,
    history_units: int,
    frames_per_sentence: int,
    sentence_mode: str,
    split_punctuation: str,
    include_comma_split: bool,
) -> List[Dict[str, Any]]:
    if unit == "segment":
        return build_segment_samples_for_video(
            transcript,
            window_sec=window_sec,
            min_words=min_words,
            context_max_chars=context_max_chars,
            use_context=use_context,
        )

    if unit == "sentence":
        return build_sentence_samples_for_video(
            transcript,
            window_sec=window_sec,
            min_words=min_words,
            context_max_chars=context_max_chars,
            use_context=use_context,
            sentence_max_words=sentence_max_words,
            sentence_max_duration=sentence_max_duration,
            sample_format=sample_format,
            history_units=history_units,
            frames_per_sentence=frames_per_sentence,
            sentence_mode=sentence_mode,
            split_punctuation=split_punctuation,
            include_comma_split=include_comma_split,
        )

    raise ValueError(f"Unsupported unit: {unit}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build ultrasound ASR caption-completion pretraining samples")
    parser.add_argument("--transcripts", type=str, required=True,
                        help="Directory of ASR transcript JSON files ({video_id}.json).")
    parser.add_argument("--output", type=str, required=True,
                        help="Output jsonl path.")
    parser.add_argument("--window-sec", type=float, default=8.0,
                        help="Seconds of frames before unit.start to look at.")
    parser.add_argument("--min-words", type=int, default=3,
                        help="Skip narration units shorter than this many words.")
    parser.add_argument("--context-max-chars", type=int, default=400,
                        help="Max characters of previous narration used as prev_context.")
    parser.add_argument("--no-context", action="store_true",
                        help="Disable prev_context (pure visual completion).")
    parser.add_argument("--limit-videos", type=int, default=None,
                        help="Only process this many transcript files.")
    parser.add_argument("--unit", choices=["segment", "sentence"], default="segment",
                        help="Sample unit. segment = one ASR segment per sample; sentence = merge ASR segments into sentence-like units.")
    parser.add_argument("--sentence-max-words", type=int, default=40,
                        help="Sentence mode fallback: force a split after this many accumulated words. Use <=0 to disable.")
    parser.add_argument("--sentence-max-duration", type=float, default=15.0,
                        help="Segment-merge sentence mode fallback: force a split after this many seconds. Use <=0 to disable.")
    parser.add_argument("--sentence-mode", choices=["segment_merge", "punctuation"], default="segment_merge",
                        help="Sentence unit construction mode. segment_merge accumulates ASR segments; punctuation splits the concatenated ASR text by punctuation and maps spans back to timestamps.")
    parser.add_argument("--split-punctuation", type=str, default=".?!;:。？！；：",
                        help="Punctuation characters used by --sentence-mode punctuation.")
    parser.add_argument("--include-comma-split", action="store_true",
                        help="Also split punctuation-mode sentence units on comma characters.")
    parser.add_argument("--format", choices=["standard", "interleave"], default="standard",
                        help="Sample format. standard = existing single-window sample; interleave = prior sentence frames/text interleaved to predict next sentence.")
    parser.add_argument("--history-units", type=int, default=3,
                        help="Interleave mode: number of previous sentence units to include.")
    parser.add_argument("--frames-per-sentence", type=int, default=3,
                        help="Interleave mode: uniformly sampled frames per history sentence.")
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
                unit=args.unit,
                sentence_max_words=args.sentence_max_words,
                sentence_max_duration=args.sentence_max_duration,
                sample_format=args.format,
                history_units=args.history_units,
                frames_per_sentence=args.frames_per_sentence,
                sentence_mode=args.sentence_mode,
                split_punctuation=args.split_punctuation,
                include_comma_split=args.include_comma_split,
            )

            for s in samples:
                f.write(json.dumps(s, ensure_ascii=False) + "\n")

            vid = transcript.get("video_id", jf.stem)
            per_video_counts[vid] = len(samples)
            total_samples += len(samples)
            print(f"[build] {vid}: {len(samples)} samples")

    print("=" * 60)
    print(f"[build] unit: {args.unit}")
    print(f"[build] sentence_mode: {args.sentence_mode}")
    print(f"[build] format: {args.format}")
    print(f"[build] videos: {len(per_video_counts)}")
    print(f"[build] total samples: {total_samples}")
    print(f"[build] output: {out_path}")


if __name__ == "__main__":
    main()