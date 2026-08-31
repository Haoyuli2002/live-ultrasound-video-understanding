#!/usr/bin/env python3
"""Rule-based ASR transcript quality filter for ultrasound pretraining.

The filter is intentionally cheap and deterministic.  It removes obvious bad
ASR before pretrain sample construction, e.g. non-English / garbled transcripts.

Example:

  python scripts/data/filter_asr_transcripts.py \
    --transcripts cluster_data/QA/train_full295/transcripts \
    --video-map cluster_data/splits/train_full295_videos.json \
    --output-audit cluster_data/QA/train_full295/asr_filter_audit.jsonl \
    --output-keep-map cluster_data/splits/train_full295_asr_keep_videos.json \
    --output-keep-transcripts cluster_data/QA/train_full295/transcripts_asr_keep
"""

from __future__ import annotations

import argparse
import json
import re
import shutil
from pathlib import Path
from typing import Any, Dict, Iterable, List


DEFAULT_KEYWORDS = [
    "ultrasound",
    "sonography",
    "sonographic",
    "pocus",
    "probe",
    "transducer",
    "scan",
    "scanning",
    "doppler",
    "image",
    "imaging",
    "view",
    "pleural",
    "lung",
    "lungs",
    "b-line",
    "b lines",
    "a-line",
    "a lines",
    "effusion",
    "pneumothorax",
    "consolidation",
    "cardiac",
    "echo",
    "echocardiography",
    "abdomen",
    "abdominal",
    "kidney",
    "renal",
    "gallbladder",
    "vessel",
    "vascular",
    "vein",
    "artery",
    "ivc",
    "fast exam",
    "efast",
]


def load_json(path: Path) -> Any:
    with path.open(encoding="utf-8") as f:
        return json.load(f)


def normalize_text(text: str) -> str:
    return re.sub(r"\s+", " ", (text or "").strip())


def transcript_text(data: Dict[str, Any]) -> str:
    full = normalize_text(data.get("full_text") or "")
    if full:
        return full
    return normalize_text(" ".join(str(s.get("text") or "") for s in data.get("segments", [])))


def word_count(text: str) -> int:
    return len(re.findall(r"\b\w+\b", text))


def keyword_hits(text: str, keywords: Iterable[str]) -> List[str]:
    low = text.lower()
    hits = []
    for kw in keywords:
        if kw.lower() in low:
            hits.append(kw)
    return hits


def repeated_line_ratio(segments: List[Dict[str, Any]]) -> float:
    texts = [normalize_text(s.get("text") or "").lower() for s in segments]
    texts = [t for t in texts if t]
    if not texts:
        return 1.0
    counts: Dict[str, int] = {}
    for t in texts:
        counts[t] = counts.get(t, 0) + 1
    repeated = sum(c for c in counts.values() if c > 1)
    return repeated / max(len(texts), 1)


def audit_one(
    path: Path,
    *,
    keywords: List[str],
    min_language_prob: float,
    min_words: int,
    min_segments: int,
    min_keyword_hits: int,
    max_repeated_ratio: float,
) -> Dict[str, Any]:
    data = load_json(path)
    video_id = str(data.get("video_id") or path.stem)
    language = str(data.get("language") or "")
    language_probability = float(data.get("language_probability") or 0.0)
    segments = data.get("segments") or []
    text = transcript_text(data)
    wc = word_count(text)
    hits = keyword_hits(text, keywords)
    rep_ratio = repeated_line_ratio(segments)

    reasons = []
    if language != "en":
        reasons.append("language_not_en")
    if language_probability < min_language_prob:
        reasons.append(f"language_probability_lt_{min_language_prob}")
    if wc < min_words:
        reasons.append(f"word_count_lt_{min_words}")
    if len(segments) < min_segments:
        reasons.append(f"num_segments_lt_{min_segments}")
    if len(hits) < min_keyword_hits:
        reasons.append(f"keyword_hits_lt_{min_keyword_hits}")
    if rep_ratio > max_repeated_ratio:
        reasons.append(f"repeated_line_ratio_gt_{max_repeated_ratio}")

    return {
        "video_id": video_id,
        "transcript_path": str(path),
        "keep": not reasons,
        "language": language,
        "language_probability": language_probability,
        "word_count": wc,
        "num_segments": len(segments),
        "keyword_hits": len(hits),
        "matched_keywords": hits,
        "repeated_line_ratio": round(rep_ratio, 4),
        "duration_sec": data.get("duration_sec"),
        "reasons": reasons,
    }


def load_video_map(path: Path | None) -> Dict[str, str]:
    if path is None:
        return {}
    payload = load_json(path)
    if not isinstance(payload, dict):
        raise ValueError(f"Expected JSON object in {path}")
    return {str(k): str(v) for k, v in payload.items()}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Rule-filter ASR transcripts for ultrasound training")
    parser.add_argument("--transcripts", type=Path, required=True)
    parser.add_argument("--video-map", type=Path, default=None)
    parser.add_argument("--output-audit", type=Path, required=True)
    parser.add_argument("--output-keep-map", type=Path, default=None)
    parser.add_argument("--output-drop-map", type=Path, default=None)
    parser.add_argument("--output-keep-transcripts", type=Path, default=None)
    parser.add_argument("--min-language-prob", type=float, default=0.8)
    parser.add_argument("--min-words", type=int, default=300)
    parser.add_argument("--min-segments", type=int, default=20)
    parser.add_argument("--min-keyword-hits", type=int, default=2)
    parser.add_argument("--max-repeated-ratio", type=float, default=0.35)
    parser.add_argument("--keyword", action="append", default=[], help="Additional keyword; can be repeated")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    transcript_paths = sorted(args.transcripts.glob("*.json"))
    if not transcript_paths:
        raise FileNotFoundError(f"No transcript JSON files found in {args.transcripts}")

    keywords = list(dict.fromkeys(DEFAULT_KEYWORDS + list(args.keyword)))
    video_map = load_video_map(args.video_map)

    audits = [
        audit_one(
            p,
            keywords=keywords,
            min_language_prob=args.min_language_prob,
            min_words=args.min_words,
            min_segments=args.min_segments,
            min_keyword_hits=args.min_keyword_hits,
            max_repeated_ratio=args.max_repeated_ratio,
        )
        for p in transcript_paths
    ]

    keep_ids = {a["video_id"] for a in audits if a["keep"]}
    drop_ids = {a["video_id"] for a in audits if not a["keep"]}

    args.output_audit.parent.mkdir(parents=True, exist_ok=True)
    with args.output_audit.open("w", encoding="utf-8") as f:
        for audit in audits:
            f.write(json.dumps(audit, ensure_ascii=False) + "\n")

    if args.output_keep_map:
        keep_map = {vid: path for vid, path in video_map.items() if vid in keep_ids}
        args.output_keep_map.parent.mkdir(parents=True, exist_ok=True)
        args.output_keep_map.write_text(json.dumps(keep_map, ensure_ascii=False, indent=2), encoding="utf-8")

    if args.output_drop_map:
        drop_map = {vid: path for vid, path in video_map.items() if vid in drop_ids}
        args.output_drop_map.parent.mkdir(parents=True, exist_ok=True)
        args.output_drop_map.write_text(json.dumps(drop_map, ensure_ascii=False, indent=2), encoding="utf-8")

    if args.output_keep_transcripts:
        args.output_keep_transcripts.mkdir(parents=True, exist_ok=True)
        by_id = {p.stem: p for p in transcript_paths}
        for vid in keep_ids:
            src = by_id.get(vid)
            if src:
                shutil.copy2(src, args.output_keep_transcripts / src.name)

    print("=" * 80)
    print(f"transcripts : {len(audits)}")
    print(f"keep        : {len(keep_ids)}")
    print(f"drop        : {len(drop_ids)}")
    print(f"audit       : {args.output_audit}")
    if args.output_keep_map:
        print(f"keep map    : {args.output_keep_map}")
    if args.output_keep_transcripts:
        print(f"keep trans  : {args.output_keep_transcripts}")
    print("drop reasons:")
    reason_counts: Dict[str, int] = {}
    for audit in audits:
        for reason in audit["reasons"]:
            reason_counts[reason] = reason_counts.get(reason, 0) + 1
    for reason, count in sorted(reason_counts.items(), key=lambda x: (-x[1], x[0])):
        print(f"  {reason}: {count}")


if __name__ == "__main__":
    main()