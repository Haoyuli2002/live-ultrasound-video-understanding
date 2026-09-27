#!/usr/bin/env python3
"""Build Stage-2 two-level memory-compression samples from ASR transcripts.

Final schema:
1. Every `step_sec` seconds produces one short memory token.
2. An ASR segment is reconstructed from all short windows overlapping it.
3. A full `block_sec` block uses fixed short windows and updates fixed long tokens.
4. Long samples include current / previous / accumulated targets.
5. Final incomplete long blocks are dropped; short samples still cover all ASR.
"""

from __future__ import annotations

import argparse
import json
import math
import re
from pathlib import Path
from typing import Any, Dict, Iterable, List

_NOISE_RE = re.compile(r"^[\s\[\(]*(music|applause|laughter|inaudible|noise)[\]\)]*\s*$", re.I)


def norm(text: str) -> str:
    return re.sub(r"\s+", " ", (text or "").strip())


def good_text(text: str, min_words: int) -> bool:
    t = norm(text)
    return bool(t) and not _NOISE_RE.match(t) and len(t.split()) >= min_words


def load_transcripts(path: Path, limit: int | None = None) -> Iterable[Path]:
    files = sorted(path.glob("*.json")) if path.is_dir() else [path]
    return files[:limit] if limit is not None else files


def windows_between(start: float, end: float, *, step_sec: float, duration: float | None = None) -> List[List[float]]:
    if end <= start:
        return []
    first_idx = math.floor(start / step_sec)
    last_idx = math.ceil(end / step_sec)
    windows = []
    for i in range(first_idx, last_idx):
        s = max(0.0, i * step_sec)
        e = s + step_sec
        if duration is not None:
            if s >= duration:
                break
            e = min(e, duration)
        if e > s:
            windows.append([round(s, 3), round(e, 3)])
    return windows


def asr_text_between(segments: List[Dict[str, Any]], start: float, end: float, *, min_words: int) -> str:
    texts = []
    for seg in segments:
        s0 = float(seg.get("start", 0.0))
        s1 = float(seg.get("end", s0))
        if s1 <= start or s0 >= end:
            continue
        txt = norm(seg.get("text") or "")
        if good_text(txt, min_words):
            texts.append(txt)
    return norm(" ".join(texts))


def truncate_text(text: str, max_chars: int) -> str:
    text = norm(text)
    if max_chars <= 0 or len(text) <= max_chars:
        return text
    return text[-max_chars:].lstrip()


def short_samples(transcript: Dict[str, Any], *, min_words: int, step_sec: float) -> List[Dict[str, Any]]:
    vid = transcript.get("video_id") or Path(transcript.get("video_path", "video")).stem
    duration = float(transcript.get("duration_sec") or 0.0)
    out = []
    for i, seg in enumerate(transcript.get("segments") or []):
        text = norm(seg.get("text") or "")
        start = float(seg.get("start", 0.0))
        end = float(seg.get("end", start))
        if end <= start or not good_text(text, min_words):
            continue
        short_windows = windows_between(start, end, step_sec=step_sec, duration=duration or None)
        if not short_windows:
            continue
        out.append({
            "sample_type": "short_memory_compression",
            "video_id": vid,
            "video": transcript.get("video_path"),
            "asr_window": [round(start, 3), round(end, 3)],
            "short_windows": short_windows,
            "target": text,
            "meta": {
                "segment_idx": i,
                "seg_start": start,
                "seg_end": end,
                "source": "asr_segment",
                "step_sec": step_sec,
            },
        })
    return out


def long_samples(
    transcript: Dict[str, Any], *, block_sec: float, step_sec: float, min_words: int,
    long_token_count: int, history_max_chars: int,
) -> List[Dict[str, Any]]:
    vid = transcript.get("video_id") or Path(transcript.get("video_path", "video")).stem
    duration = float(transcript.get("duration_sec") or 0.0)
    segments = transcript.get("segments") or []
    if duration <= 0 and segments:
        duration = max(float(s.get("end", 0.0)) for s in segments)
    out = []
    block_idx = 0
    expected_short = int(round(block_sec / step_sec))
    previous_blocks: List[Dict[str, Any]] = []
    t = 0.0
    while t + block_sec <= duration + 1e-6:
        b0, b1 = t, t + block_sec
        current = asr_text_between(segments, b0, b1, min_words=min_words)
        if current:
            previous = asr_text_between(segments, 0.0, b0, min_words=min_words)
            accumulated = asr_text_between(segments, 0.0, b1, min_words=min_words)
            short_windows = windows_between(b0, b1, step_sec=step_sec, duration=duration)
            if len(short_windows) == expected_short:
                previous_target = truncate_text(previous, history_max_chars)
                accumulated_target = truncate_text(accumulated, history_max_chars)
                out.append({
                    "sample_type": "long_memory_compression",
                    "video_id": vid,
                    "video": transcript.get("video_path"),
                    "block_idx": block_idx,
                    "block_window": [round(b0, 3), round(b1, 3)],
                    "history_window": [0.0, round(b1, 3)],
                    "short_windows": short_windows,
                    "previous_blocks": list(previous_blocks),
                    "num_long_tokens": int(long_token_count),
                    "current_block_target": current,
                    "previous_summary_target": previous_target,
                    "accumulated_summary_target": accumulated_target,
                    "target": accumulated_target,
                    "meta": {
                        "target_mode": "accumulated_summary_asr_truncated",
                        "source": "asr_concat",
                        "block_sec": block_sec,
                        "step_sec": step_sec,
                        "drop_last_incomplete": True,
                    },
                })
                previous_blocks.append({
                    "block_idx": block_idx,
                    "block_window": [round(b0, 3), round(b1, 3)],
                    "short_windows": short_windows,
                })
                block_idx += 1
        t += block_sec
    return out


def parse_args():
    p = argparse.ArgumentParser(description="Build Stage-2 memory compression samples")
    p.add_argument("--transcripts", required=True, help="Transcript JSON file or directory")
    p.add_argument("--output", required=True)
    p.add_argument("--types", default="short,long", help="Comma-separated: short,long")
    p.add_argument("--block-sec", type=float, default=60.0)
    p.add_argument("--step-sec", type=float, default=1.0)
    p.add_argument("--long-token-count", type=int, default=60)
    p.add_argument("--history-max-chars", type=int, default=2400)
    p.add_argument("--min-words", type=int, default=3)
    p.add_argument("--limit-videos", type=int, default=None)
    return p.parse_args()


def main():
    args = parse_args()
    kinds = {x.strip() for x in args.types.split(",") if x.strip()}
    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    total = 0
    per_type = {"short": 0, "long": 0}
    with out.open("w", encoding="utf-8") as f:
        for jf in load_transcripts(Path(args.transcripts), args.limit_videos):
            tr = json.loads(jf.read_text(encoding="utf-8"))
            rows: List[Dict[str, Any]] = []
            if "short" in kinds:
                ss = short_samples(tr, min_words=args.min_words, step_sec=args.step_sec)
                rows.extend(ss)
                per_type["short"] += len(ss)
            if "long" in kinds:
                ls = long_samples(
                    tr,
                    block_sec=args.block_sec,
                    step_sec=args.step_sec,
                    min_words=args.min_words,
                    long_token_count=args.long_token_count,
                    history_max_chars=args.history_max_chars,
                )
                rows.extend(ls)
                per_type["long"] += len(ls)
            for r in rows:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
            total += len(rows)
            print(f"[memory-build] {jf.stem}: {len(rows)} samples")
    print(f"[memory-build] total={total} short={per_type['short']} long={per_type['long']} output={out}")


if __name__ == "__main__":
    main()