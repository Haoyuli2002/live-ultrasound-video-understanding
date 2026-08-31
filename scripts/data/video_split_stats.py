#!/usr/bin/env python3
"""Compute size and duration statistics for train/eval video split maps.

Example:

  python scripts/data/video_split_stats.py \
    --train-map cluster_data/splits/train_full295_videos.json \
    --eval-map cluster_data/splits/eval_full295_videos.json \
    --output cluster_data/splits/full295_video_stats.json
"""

from __future__ import annotations

import argparse
import json
import subprocess
from pathlib import Path
from typing import Any, Dict, List, Tuple


def load_video_map(path: Path) -> Dict[str, str]:
    with path.open(encoding="utf-8") as f:
        payload = json.load(f)
    if not isinstance(payload, dict):
        raise ValueError(f"Expected JSON object in {path}, got {type(payload).__name__}")
    return {str(k): str(v) for k, v in payload.items()}


def resolve_path(path: str, repo_root: Path) -> Path:
    p = Path(path)
    return p if p.is_absolute() else repo_root / p


def ffprobe_duration(path: Path, ffprobe_bin: str) -> float:
    cmd = [
        ffprobe_bin,
        "-v",
        "error",
        "-show_entries",
        "format=duration",
        "-of",
        "json",
        str(path),
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        raise RuntimeError(result.stderr.strip() or f"ffprobe failed with code {result.returncode}")
    payload = json.loads(result.stdout or "{}")
    return float((payload.get("format") or {}).get("duration") or 0.0)


def human_seconds(seconds: float) -> str:
    seconds = int(round(seconds))
    h = seconds // 3600
    m = (seconds % 3600) // 60
    s = seconds % 60
    return f"{h:02d}:{m:02d}:{s:02d}"


def summarize_split(
    name: str,
    video_map: Dict[str, str],
    *,
    repo_root: Path,
    ffprobe_bin: str,
) -> Dict[str, Any]:
    rows: List[Dict[str, Any]] = []
    missing: List[Dict[str, str]] = []
    probe_failed: List[Dict[str, str]] = []
    total_bytes = 0
    total_duration = 0.0

    for video_id, raw_path in sorted(video_map.items()):
        path = resolve_path(raw_path, repo_root)
        if not path.exists():
            missing.append({"video_id": video_id, "path": str(path)})
            continue

        size_bytes = path.stat().st_size
        try:
            duration_sec = ffprobe_duration(path, ffprobe_bin)
        except Exception as exc:  # noqa: BLE001 - collect all failures.
            duration_sec = 0.0
            probe_failed.append({"video_id": video_id, "path": str(path), "error": str(exc)})

        total_bytes += size_bytes
        total_duration += duration_sec
        rows.append({
            "video_id": video_id,
            "path": str(path),
            "size_bytes": size_bytes,
            "size_gib": size_bytes / (1024 ** 3),
            "duration_sec": duration_sec,
            "duration_min": duration_sec / 60.0,
        })

    count = len(rows)
    return {
        "name": name,
        "declared_count": len(video_map),
        "existing_count": count,
        "missing_count": len(missing),
        "ffprobe_failed_count": len(probe_failed),
        "total_size_bytes": total_bytes,
        "total_size_gib": total_bytes / (1024 ** 3),
        "total_size_gb": total_bytes / 1e9,
        "total_duration_sec": total_duration,
        "total_duration_hours": total_duration / 3600.0,
        "total_duration_hhmmss": human_seconds(total_duration),
        "mean_duration_min": (total_duration / 60.0 / count) if count else 0.0,
        "mean_size_mib": (total_bytes / (1024 ** 2) / count) if count else 0.0,
        "missing": missing,
        "ffprobe_failed": probe_failed,
        "videos": rows,
    }


def print_split_summary(summary: Dict[str, Any]) -> None:
    print("=" * 80)
    print(summary["name"].upper())
    print("=" * 80)
    print(f"declared videos     : {summary['declared_count']}")
    print(f"existing videos     : {summary['existing_count']}")
    print(f"missing videos      : {summary['missing_count']}")
    print(f"ffprobe failures    : {summary['ffprobe_failed_count']}")
    print(f"total size          : {summary['total_size_gib']:.3f} GiB ({summary['total_size_gb']:.3f} GB)")
    print(f"total duration      : {summary['total_duration_hours']:.3f} hours ({summary['total_duration_hhmmss']})")
    print(f"mean duration/video : {summary['mean_duration_min']:.2f} min")
    print(f"mean size/video     : {summary['mean_size_mib']:.2f} MiB")
    if summary["missing"]:
        print("missing examples:")
        for row in summary["missing"][:10]:
            print(f"  {row['video_id']} {row['path']}")
    if summary["ffprobe_failed"]:
        print("ffprobe failure examples:")
        for row in summary["ffprobe_failed"][:10]:
            print(f"  {row['video_id']} {row['error']} {row['path']}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compute video split size/duration statistics")
    parser.add_argument("--train-map", type=Path, required=True)
    parser.add_argument("--eval-map", type=Path, required=True)
    parser.add_argument("--repo-root", type=Path, default=Path("."))
    parser.add_argument("--ffprobe-bin", default="ffprobe")
    parser.add_argument("--output", type=Path, default=None, help="Optional JSON output path")
    parser.add_argument("--no-per-video", action="store_true", help="Omit per-video rows from JSON output")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    repo_root = args.repo_root.resolve()

    train_map = load_video_map(args.train_map)
    eval_map = load_video_map(args.eval_map)

    train = summarize_split("train", train_map, repo_root=repo_root, ffprobe_bin=args.ffprobe_bin)
    eval_ = summarize_split("eval", eval_map, repo_root=repo_root, ffprobe_bin=args.ffprobe_bin)

    total = {
        "name": "total",
        "declared_count": train["declared_count"] + eval_["declared_count"],
        "existing_count": train["existing_count"] + eval_["existing_count"],
        "missing_count": train["missing_count"] + eval_["missing_count"],
        "ffprobe_failed_count": train["ffprobe_failed_count"] + eval_["ffprobe_failed_count"],
        "total_size_bytes": train["total_size_bytes"] + eval_["total_size_bytes"],
        "total_duration_sec": train["total_duration_sec"] + eval_["total_duration_sec"],
        "missing": train["missing"] + eval_["missing"],
        "ffprobe_failed": train["ffprobe_failed"] + eval_["ffprobe_failed"],
    }
    total["total_size_gib"] = total["total_size_bytes"] / (1024 ** 3)
    total["total_size_gb"] = total["total_size_bytes"] / 1e9
    total["total_duration_hours"] = total["total_duration_sec"] / 3600.0
    total["total_duration_hhmmss"] = human_seconds(total["total_duration_sec"])
    total["mean_duration_min"] = (
        total["total_duration_sec"] / 60.0 / total["existing_count"]
        if total["existing_count"] else 0.0
    )
    total["mean_size_mib"] = (
        total["total_size_bytes"] / (1024 ** 2) / total["existing_count"]
        if total["existing_count"] else 0.0
    )

    print_split_summary(train)
    print_split_summary(eval_)
    print_split_summary(total)

    payload = {
        "train_map": str(args.train_map),
        "eval_map": str(args.eval_map),
        "train": train,
        "eval": eval_,
        "total": total,
    }
    if args.no_per_video:
        for key in ["train", "eval"]:
            payload[key].pop("videos", None)

    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"[stats] wrote {args.output}")


if __name__ == "__main__":
    main()