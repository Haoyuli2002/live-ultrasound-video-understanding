#!/usr/bin/env python3
"""Build stage-specific keep/drop video maps from VLM video-type audit JSONL.

Example:
  python scripts/data/filter_by_vlm_video_type.py \
    --video-map cluster_data/splits/train_full295_asr_keep_videos.json \
    --vlm-audit cluster_data/splits/train_full295_asr_vlm_video_type.jsonl \
    --stage compression \
    --output-keep-map cluster_data/splits/train_full295_asr_vlm_compression_keep_videos.json \
    --output-drop-map cluster_data/splits/train_full295_asr_vlm_compression_drop_videos.json
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict


STAGE_FIELD = {
    "pretrain": "keep_for_pretrain",
    "compression": "keep_for_compression",
    "sft": "keep_for_sft",
}


def load_json(path: Path) -> Any:
    with path.open(encoding="utf-8") as f:
        return json.load(f)


def load_audit(path: Path) -> Dict[str, Dict[str, Any]]:
    rows = {}
    with path.open(encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            rec = json.loads(line)
            video_id = str(rec.get("video_id") or "")
            if video_id:
                rows[video_id] = rec
    return rows


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Filter video map by VLM video type labels / stage keep flags")
    p.add_argument("--video-map", type=Path, required=True)
    p.add_argument("--vlm-audit", type=Path, required=True)
    p.add_argument("--stage", choices=sorted(STAGE_FIELD), required=True)
    p.add_argument("--output-keep-map", type=Path, required=True)
    p.add_argument("--output-drop-map", type=Path, required=True)
    p.add_argument("--output-summary", type=Path, default=None)
    p.add_argument("--keep-uncertain-missing", action="store_true", help="Keep videos missing from audit instead of dropping them")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    video_map = {str(k): str(v) for k, v in load_json(args.video_map).items()}
    audit = load_audit(args.vlm_audit)
    field = STAGE_FIELD[args.stage]

    keep = {}
    drop = {}
    label_counts: Dict[str, int] = {}
    missing = []
    for video_id, path in video_map.items():
        rec = audit.get(video_id)
        if rec is None:
            missing.append(video_id)
            decision = bool(args.keep_uncertain_missing)
            label = "missing_audit"
        else:
            decision = bool(rec.get(field))
            label = str(rec.get("label") or "unknown")
        label_counts[label] = label_counts.get(label, 0) + 1
        if decision:
            keep[video_id] = path
        else:
            drop[video_id] = path

    args.output_keep_map.parent.mkdir(parents=True, exist_ok=True)
    args.output_drop_map.parent.mkdir(parents=True, exist_ok=True)
    args.output_keep_map.write_text(json.dumps(keep, ensure_ascii=False, indent=2), encoding="utf-8")
    args.output_drop_map.write_text(json.dumps(drop, ensure_ascii=False, indent=2), encoding="utf-8")

    summary = {
        "stage": args.stage,
        "field": field,
        "input_count": len(video_map),
        "keep_count": len(keep),
        "drop_count": len(drop),
        "missing_audit_count": len(missing),
        "label_counts": label_counts,
        "output_keep_map": str(args.output_keep_map),
        "output_drop_map": str(args.output_drop_map),
    }
    if args.output_summary:
        args.output_summary.parent.mkdir(parents=True, exist_ok=True)
        args.output_summary.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()