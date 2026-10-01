#!/usr/bin/env python3
"""Compute train/eval video label, duration, and size statistics.

Example:
  python scripts/data/video_split_stats.py \
    --train-map cluster_data/splits/train_full295_asr_keep_videos.json \
    --eval-map cluster_data/splits/eval_full295_asr_keep_videos.json \
    --train-labels cluster_data/splits/train_full295_qwen35_video_type.jsonl \
    --eval-labels cluster_data/splits/eval_full295_qwen35_video_type.jsonl \
    --output cluster_data/splits/full295_qwen35_video_stats.json
"""

from __future__ import annotations

import argparse
import json
import statistics
import subprocess
from collections import Counter
from pathlib import Path
from typing import Any, Dict, Iterable, List


def load_json(path: Path) -> Any:
    with path.open(encoding="utf-8") as f:
        return json.load(f)


def load_video_map(path: Path | None) -> Dict[str, str]:
    if path is None:
        return {}
    payload = load_json(path)
    if not isinstance(payload, dict):
        raise ValueError(f"Expected JSON object in {path}, got {type(payload).__name__}")
    return {str(k): str(v) for k, v in payload.items()}


def load_jsonl(path: Path | None) -> List[Dict[str, Any]]:
    if path is None:
        return []
    rows: List[Dict[str, Any]] = []
    with path.open(encoding="utf-8") as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def label_by_video_id(rows: Iterable[Dict[str, Any]]) -> Dict[str, Dict[str, Any]]:
    out: Dict[str, Dict[str, Any]] = {}
    for row in rows:
        vid = str(row.get("video_id") or "")
        if vid:
            out[vid] = row
    return out


def resolve_path(raw_path: str | None, repo_root: Path) -> Path | None:
    if not raw_path:
        return None
    p = Path(str(raw_path))
    return p if p.is_absolute() else repo_root / p


def ffprobe_duration(path: Path, ffprobe_bin: str) -> float:
    cmd = [
        ffprobe_bin,
        "-v",
        "error",
        "-show_entries",
        "format=duration:stream=duration",
        "-of",
        "json",
        str(path),
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        raise RuntimeError(result.stderr.strip() or f"ffprobe failed with code {result.returncode}")
    payload = json.loads(result.stdout or "{}")
    candidates = []
    fmt_duration = (payload.get("format") or {}).get("duration")
    if fmt_duration not in (None, "N/A", ""):
        candidates.append(float(fmt_duration))
    for stream in payload.get("streams") or []:
        duration = stream.get("duration")
        if duration not in (None, "N/A", ""):
            candidates.append(float(duration))
    return max(candidates) if candidates else 0.0


def human_seconds(seconds: float) -> str:
    seconds = int(round(seconds))
    return f"{seconds // 3600:02d}:{(seconds % 3600) // 60:02d}:{seconds % 60:02d}"


def numeric_summary(values: List[float]) -> Dict[str, float]:
    if not values:
        return {"min": 0.0, "mean": 0.0, "median": 0.0, "max": 0.0}
    return {"min": min(values), "mean": statistics.mean(values), "median": statistics.median(values), "max": max(values)}


def counter_dict(counter: Counter) -> Dict[str, int]:
    return dict(counter.most_common())


def collect_list_counts(rows: Iterable[Dict[str, Any]], key: str) -> Dict[str, int]:
    cnt: Counter = Counter()
    for row in rows:
        vals = row.get(key) or []
        if isinstance(vals, str):
            vals = [vals]
        for val in vals:
            s = str(val).strip()
            if s:
                cnt[s] += 1
    return counter_dict(cnt)


def summarize_split(name: str, *, video_map: Dict[str, str], label_rows: List[Dict[str, Any]], repo_root: Path, ffprobe_bin: str) -> Dict[str, Any]:
    labels = label_by_video_id(label_rows)
    video_ids = sorted(set(video_map) | set(labels))
    per_video: List[Dict[str, Any]] = []
    missing_files: List[Dict[str, str]] = []
    ffprobe_failed: List[Dict[str, str]] = []
    missing_labels: List[str] = []
    label_without_map: List[str] = []
    durations: List[float] = []
    sizes_bytes: List[int] = []

    for vid in video_ids:
        rec = labels.get(vid, {})
        raw_path = video_map.get(vid) or rec.get("video_path") or rec.get("video")
        path = resolve_path(raw_path, repo_root)
        exists = bool(path and path.exists())
        size_bytes = 0
        duration_sec = 0.0

        if vid not in labels:
            missing_labels.append(vid)
        if vid in labels and vid not in video_map:
            label_without_map.append(vid)

        duration_source = "none"
        if not exists:
            missing_files.append({"video_id": vid, "path": str(path) if path else ""})
        else:
            assert path is not None
            size_bytes = path.stat().st_size
            sizes_bytes.append(size_bytes)
            try:
                duration_sec = ffprobe_duration(path, ffprobe_bin)
                duration_source = "ffprobe" if duration_sec > 0 else "none"
            except Exception as exc:  # noqa: BLE001
                ffprobe_failed.append({"video_id": vid, "path": str(path), "error": str(exc)})
        if duration_sec <= 0 and rec.get("duration_sec") is not None:
            try:
                duration_sec = float(rec.get("duration_sec") or 0.0)
                duration_source = "label_duration_sec" if duration_sec > 0 else duration_source
            except Exception:
                pass
        if exists:
            durations.append(duration_sec)

        per_video.append({
            "video_id": vid,
            "path": str(path) if path else None,
            "exists": exists,
            "duration_sec": duration_sec,
            "duration_source": duration_source,
            "duration_min": duration_sec / 60.0,
            "size_bytes": size_bytes,
            "size_gb": size_bytes / 1e9,
            "label": rec.get("label") or rec.get("final_label"),
            "spoken_language": rec.get("spoken_language"),
            "keep_for_pretrain": rec.get("keep_for_pretrain"),
            "keep_for_compression": rec.get("keep_for_compression"),
            "keep_for_sft": rec.get("keep_for_sft"),
            "needs_clipping": rec.get("needs_clipping"),
            "error": rec.get("error"),
        })

    label_source_rows = list(labels.values())
    total_size_bytes = sum(sizes_bytes)
    total_duration_sec = sum(durations)
    return {
        "name": name,
        "video_map_count": len(video_map),
        "label_rows_count": len(label_rows),
        "unique_video_count": len(video_ids),
        "existing_file_count": sum(1 for row in per_video if row["exists"]),
        "missing_file_count": len(missing_files),
        "missing_label_count": len(missing_labels),
        "label_without_map_count": len(label_without_map),
        "error_count": sum(bool(r.get("error")) for r in label_source_rows),
        "label_distribution": counter_dict(Counter((r.get("label") or r.get("final_label") or "missing") for r in label_source_rows)),
        "spoken_language_distribution": counter_dict(Counter((r.get("spoken_language") or "missing") for r in label_source_rows)),
        "teaching_value_distribution": counter_dict(Counter((r.get("teaching_value") or "missing") for r in label_source_rows)),
        "keep_counts": {k: sum(bool(r.get(k)) for r in label_source_rows) for k in ["keep_for_pretrain", "keep_for_compression", "keep_for_sft", "needs_clipping"]},
        "has_counts": {k: sum(bool(r.get(k)) for r in label_source_rows) for k in ["has_realtime_ultrasound", "has_probe_or_patient", "has_ppt_or_slides", "has_talking_head", "has_static_ultrasound_images", "has_non_ultrasound_content"]},
        "anatomy_region_counts": collect_list_counts(label_source_rows, "anatomy_regions"),
        "clinical_scenario_counts": collect_list_counts(label_source_rows, "clinical_scenarios"),
        "duration_min": numeric_summary([x / 60.0 for x in durations]),
        "total_duration_sec": total_duration_sec,
        "total_duration_hours": total_duration_sec / 3600.0,
        "total_duration_hhmmss": human_seconds(total_duration_sec),
        "size_mb": numeric_summary([x / 1e6 for x in sizes_bytes]),
        "total_size_bytes": total_size_bytes,
        "total_size_gb": total_size_bytes / 1e9,
        "total_size_gib": total_size_bytes / (1024 ** 3),
        "missing_files": missing_files,
        "ffprobe_failed": ffprobe_failed,
        "missing_labels": missing_labels,
        "label_without_map": label_without_map,
        "videos": per_video,
    }


def print_counter(title: str, mapping: Dict[str, int], *, limit: int = 30) -> None:
    print(title)
    if not mapping:
        print("  (empty)")
        return
    for i, (key, val) in enumerate(mapping.items()):
        if i >= limit:
            print(f"  ... ({len(mapping) - limit} more)")
            break
        print(f"  {key}: {val}")


def print_split_summary(summary: Dict[str, Any]) -> None:
    print("=" * 80)
    print(summary["name"].upper())
    print("=" * 80)
    print(f"video map count     : {summary['video_map_count']}")
    print(f"label rows count    : {summary['label_rows_count']}")
    print(f"unique video count  : {summary['unique_video_count']}")
    print(f"existing files      : {summary['existing_file_count']}")
    print(f"missing files       : {summary['missing_file_count']}")
    print(f"missing labels      : {summary['missing_label_count']}")
    print(f"errors              : {summary['error_count']}")
    print(f"total duration      : {summary['total_duration_hours']:.3f} h ({summary['total_duration_hhmmss']})")
    print(f"duration/video min  : min={summary['duration_min']['min']:.2f} mean={summary['duration_min']['mean']:.2f} median={summary['duration_min']['median']:.2f} max={summary['duration_min']['max']:.2f}")
    print(f"total size          : {summary['total_size_gb']:.3f} GB ({summary['total_size_gib']:.3f} GiB)")
    print(f"size/video MB       : min={summary['size_mb']['min']:.1f} mean={summary['size_mb']['mean']:.1f} median={summary['size_mb']['median']:.1f} max={summary['size_mb']['max']:.1f}")
    print_counter("label distribution:", summary["label_distribution"])
    print_counter("spoken_language distribution:", summary["spoken_language_distribution"])
    print_counter("teaching_value distribution:", summary["teaching_value_distribution"])
    print_counter("keep counts:", summary["keep_counts"])
    print_counter("has_* counts:", summary["has_counts"])
    print_counter("top anatomy regions:", summary["anatomy_region_counts"], limit=15)
    print_counter("top clinical scenarios:", summary["clinical_scenario_counts"], limit=15)
    if summary["missing_files"]:
        print("missing file examples:")
        for row in summary["missing_files"][:10]:
            print(f"  {row['video_id']} {row['path']}")


def combine_total(train: Dict[str, Any], eval_: Dict[str, Any]) -> Dict[str, Any]:
    durations = [v["duration_sec"] for v in train.get("videos", []) + eval_.get("videos", []) if v.get("exists")]
    sizes = [v["size_bytes"] for v in train.get("videos", []) + eval_.get("videos", []) if v.get("exists")]
    total_size_bytes = int(sum(sizes))
    total_duration_sec = float(sum(durations))
    return {
        "name": "total",
        "video_map_count": train["video_map_count"] + eval_["video_map_count"],
        "label_rows_count": train["label_rows_count"] + eval_["label_rows_count"],
        "unique_video_count": train["unique_video_count"] + eval_["unique_video_count"],
        "existing_file_count": train["existing_file_count"] + eval_["existing_file_count"],
        "missing_file_count": train["missing_file_count"] + eval_["missing_file_count"],
        "missing_label_count": train["missing_label_count"] + eval_["missing_label_count"],
        "error_count": train["error_count"] + eval_["error_count"],
        "total_duration_sec": total_duration_sec,
        "total_duration_hours": total_duration_sec / 3600.0,
        "total_duration_hhmmss": human_seconds(total_duration_sec),
        "duration_min": numeric_summary([x / 60.0 for x in durations]),
        "total_size_bytes": total_size_bytes,
        "total_size_gb": total_size_bytes / 1e9,
        "total_size_gib": total_size_bytes / (1024 ** 3),
        "size_mb": numeric_summary([x / 1e6 for x in sizes]),
        "label_distribution": counter_dict(Counter(train["label_distribution"]) + Counter(eval_["label_distribution"])),
        "spoken_language_distribution": counter_dict(Counter(train["spoken_language_distribution"]) + Counter(eval_["spoken_language_distribution"])),
        "teaching_value_distribution": counter_dict(Counter(train["teaching_value_distribution"]) + Counter(eval_["teaching_value_distribution"])),
        "keep_counts": counter_dict(Counter(train["keep_counts"]) + Counter(eval_["keep_counts"])),
        "has_counts": counter_dict(Counter(train["has_counts"]) + Counter(eval_["has_counts"])),
        "anatomy_region_counts": counter_dict(Counter(train["anatomy_region_counts"]) + Counter(eval_["anatomy_region_counts"])),
        "clinical_scenario_counts": counter_dict(Counter(train["clinical_scenario_counts"]) + Counter(eval_["clinical_scenario_counts"])),
        "missing_files": train["missing_files"] + eval_["missing_files"],
        "ffprobe_failed": train["ffprobe_failed"] + eval_["ffprobe_failed"],
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compute train/eval video label, duration, and size statistics")
    parser.add_argument("--train-map", type=Path, required=True)
    parser.add_argument("--eval-map", type=Path, required=True)
    parser.add_argument("--train-labels", type=Path, default=None, help="Train VLM label JSONL")
    parser.add_argument("--eval-labels", type=Path, default=None, help="Eval VLM label JSONL")
    parser.add_argument("--repo-root", type=Path, default=Path("."))
    parser.add_argument("--ffprobe-bin", default="ffprobe")
    parser.add_argument("--output", type=Path, default=None, help="Optional JSON output path")
    parser.add_argument("--no-per-video", action="store_true", help="Omit per-video rows from JSON output")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    repo_root = args.repo_root.resolve()
    train = summarize_split("train", video_map=load_video_map(args.train_map), label_rows=load_jsonl(args.train_labels), repo_root=repo_root, ffprobe_bin=args.ffprobe_bin)
    eval_ = summarize_split("eval", video_map=load_video_map(args.eval_map), label_rows=load_jsonl(args.eval_labels), repo_root=repo_root, ffprobe_bin=args.ffprobe_bin)
    total = combine_total(train, eval_)
    print_split_summary(train)
    print_split_summary(eval_)
    print_split_summary(total)
    payload = {"train_map": str(args.train_map), "eval_map": str(args.eval_map), "train_labels": str(args.train_labels) if args.train_labels else None, "eval_labels": str(args.eval_labels) if args.eval_labels else None, "train": train, "eval": eval_, "total": total}
    if args.no_per_video:
        for key in ["train", "eval"]:
            payload[key].pop("videos", None)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"[stats] wrote {args.output}")


if __name__ == "__main__":
    main()