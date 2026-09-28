#!/usr/bin/env python3
"""Build Stage-2 teacher-summary memory samples.

Input JSONL contains one row per complete block with at least:
  - video_id
  - block_idx or block_window
  - local_summary_target  (or local_summary)
  - global_summary_target (or cumulative_summary/global_summary)

Output JSONL uses the main Stage-2 design:
  - short_memory_summary: 60 short tokens -> local visual summary
  - long_memory_summary : L_{k-1} + 60 short tokens -> L_k -> global summary

This script does not call a teacher model; it converts already generated teacher
visual summaries into the training schema consumed by memory_dataset.py and
train_memory_compression.py.
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, List


def norm(text: str | None) -> str:
    return " ".join((text or "").strip().split())


def read_jsonl(path: Path, limit: int | None = None) -> List[Dict[str, Any]]:
    rows = []
    with path.open(encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            rows.append(json.loads(line))
            if limit is not None and len(rows) >= limit:
                break
    return rows


def block_window(row: Dict[str, Any], *, block_sec: float) -> List[float]:
    if row.get("block_window"):
        start, end = row["block_window"]
        return [round(float(start), 3), round(float(end), 3)]
    idx = int(row.get("block_idx", 0))
    start = idx * block_sec
    return [round(start, 3), round(start + block_sec, 3)]


def short_windows(start: float, end: float, *, step_sec: float) -> List[List[float]]:
    n = int(round((end - start) / step_sec))
    return [[round(start + i * step_sec, 3), round(start + (i + 1) * step_sec, 3)] for i in range(n)]


def local_target(row: Dict[str, Any]) -> str:
    return norm(row.get("local_summary_target") or row.get("local_summary"))


def global_target(row: Dict[str, Any]) -> str:
    return norm(row.get("global_summary_target") or row.get("cumulative_summary_target") or row.get("global_summary") or row.get("cumulative_summary"))


def sorted_video_rows(rows: Iterable[Dict[str, Any]], *, block_sec: float) -> List[Dict[str, Any]]:
    def key(row: Dict[str, Any]):
        if row.get("block_idx") is not None:
            return int(row["block_idx"])
        return float(block_window(row, block_sec=block_sec)[0])

    return sorted(rows, key=key)


def build_samples(rows: List[Dict[str, Any]], *, block_sec: float, step_sec: float, long_token_count: int, types: set[str]) -> List[Dict[str, Any]]:
    by_video: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for row in rows:
        vid = row.get("video_id")
        if not vid:
            raise ValueError(f"Missing video_id in row: {row}")
        by_video[str(vid)].append(row)

    samples = []
    for vid, vrows in sorted(by_video.items()):
        previous_blocks: List[Dict[str, Any]] = []
        for row in sorted_video_rows(vrows, block_sec=block_sec):
            bw = block_window(row, block_sec=block_sec)
            b0, b1 = bw
            sw = short_windows(b0, b1, step_sec=step_sec)
            block_idx = int(row.get("block_idx", round(b0 / block_sec)))
            local = local_target(row)
            glob = global_target(row)
            base = {
                "video_id": vid,
                "video": row.get("video") or row.get("video_path"),
                "block_idx": block_idx,
                "block_window": bw,
                "history_window": [0.0, bw[1]],
                "short_windows": sw,
                "meta": {
                    "source": "teacher_vlm",
                    "teacher": row.get("teacher"),
                    "block_sec": block_sec,
                    "step_sec": step_sec,
                    "drop_last_incomplete": True,
                    "chronological_rollout": True,
                },
            }
            if "short" in types and local:
                samples.append({
                    **base,
                    "sample_type": "short_memory_summary",
                    "local_summary_target": local,
                    "target": local,
                })
            if "long" in types and glob:
                samples.append({
                    **base,
                    "sample_type": "long_memory_summary",
                    "previous_blocks": list(previous_blocks),
                    "num_long_tokens": int(long_token_count),
                    "local_summary_target": local,
                    "global_summary_target": glob,
                    "target": glob,
                })
            previous_blocks.append({
                "block_idx": block_idx,
                "block_window": bw,
                "short_windows": sw,
            })
    return samples


def parse_args():
    p = argparse.ArgumentParser(description="Build teacher-summary Stage-2 memory samples")
    p.add_argument("--summaries-jsonl", required=True, help="Teacher visual summary JSONL, one row per complete block")
    p.add_argument("--output", required=True)
    p.add_argument("--types", default="short,long", help="Comma-separated: short,long")
    p.add_argument("--block-sec", type=float, default=60.0)
    p.add_argument("--step-sec", type=float, default=1.0)
    p.add_argument("--long-token-count", type=int, default=60)
    p.add_argument("--limit-rows", type=int, default=None)
    return p.parse_args()


def main():
    args = parse_args()
    types = {x.strip() for x in args.types.split(",") if x.strip()}
    if not types <= {"short", "long"}:
        raise ValueError(f"Unsupported --types: {args.types}")
    rows = read_jsonl(Path(args.summaries_jsonl), args.limit_rows)
    samples = build_samples(rows, block_sec=args.block_sec, step_sec=args.step_sec, long_token_count=args.long_token_count, types=types)
    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", encoding="utf-8") as f:
        for sample in samples:
            f.write(json.dumps(sample, ensure_ascii=False) + "\n")
    counts = defaultdict(int)
    for sample in samples:
        counts[sample["sample_type"]] += 1
    print(f"[teacher-memory-build] input_rows={len(rows)} output_samples={len(samples)} counts={dict(counts)} output={out}")


if __name__ == "__main__":
    main()