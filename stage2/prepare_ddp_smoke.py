"""Prepare a tiny, synthetic-label dataset for a real-video Stage 2 DDP smoke."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from .data import load_blocks

LOCAL_TARGET = "View: ultrasound scan; Anatomy: visible tissue; Finding: test label only"
BLOCK_TARGET = "View: ultrasound scan; Anatomy: visible tissue; Finding: test label only"
GLOBAL_TARGET = "Anatomy: visible tissue; Finding: test label only; Change: test label only"


def video_duration(path: Path) -> float:
    import cv2

    cap = cv2.VideoCapture(str(path))
    try:
        if not cap.isOpened():
            return 0.0
        fps = cap.get(cv2.CAP_PROP_FPS)
        frames = cap.get(cv2.CAP_PROP_FRAME_COUNT)
        return float(frames / fps) if fps > 0 and frames > 0 else 0.0
    finally:
        cap.release()


def select_videos(video_paths: dict[str, str], durations: dict[str, float],
                  requested_ids: list[str] | None = None) -> tuple[str, str]:
    """Pick distinct videos: one complete minute and one ten-second interval."""
    ids = requested_ids if requested_ids is not None else sorted(video_paths)
    if len(ids) != len(set(ids)):
        raise ValueError("Smoke video IDs must be distinct")
    unknown = set(ids) - set(video_paths)
    if unknown:
        raise ValueError(f"Unknown smoke video IDs: {sorted(unknown)}")
    if requested_ids is not None and len(ids) != 2:
        raise ValueError("Specify exactly two --video-id values")
    if requested_ids is not None:
        full = ids[0] if durations.get(ids[0], 0) >= 60 else None
        short = ids[1] if durations.get(ids[1], 0) >= 10 else None
    else:
        full = next((video_id for video_id in ids
                     if durations.get(video_id, 0) >= 60), None)
        short = next((video_id for video_id in ids
                      if video_id != full and durations.get(video_id, 0) >= 10), None)
    if full is None or short is None:
        raise ValueError("Need two readable videos: one at least 60s and another at least 10s")
    return full, short


def smoke_rows(full_id: str, short_id: str) -> list[dict]:
    full = {
        "video_id": full_id,
        "block_window": [0, 60],
        "local_sub_summaries": [
            {"window": [10 * i, 10 * (i + 1)],
             "local_summary_target": LOCAL_TARGET}
            for i in range(6)
        ],
        "block_summary_target": BLOCK_TARGET,
        "global_summary_target": GLOBAL_TARGET,
    }
    short = {
        "video_id": short_id,
        "block_window": [0, 10],
        "local_sub_summaries": [
            {"window": [0, 10], "local_summary_target": LOCAL_TARGET}
        ],
    }
    return [full, short]


def prepare(video_map_path: Path, output_dir: Path,
            requested_ids: list[str] | None = None) -> dict:
    source = json.loads(video_map_path.read_text(encoding="utf-8"))
    if not isinstance(source, dict):
        raise ValueError("Video path map must be a JSON object")
    paths = {}
    for video_id, raw_path in source.items():
        if not isinstance(raw_path, str) or not raw_path:
            continue
        path = Path(raw_path)
        paths[str(video_id)] = str(
            path if path.is_absolute() else (video_map_path.parent / path).resolve())
    ids_to_probe = requested_ids if requested_ids is not None else sorted(paths)
    durations = {}
    for video_id in ids_to_probe:
        if video_id not in paths:
            continue
        durations[video_id] = video_duration(Path(paths[video_id]))
        if requested_ids is None:
            full_ids = [key for key, duration in durations.items() if duration >= 60]
            if full_ids and any(key != full_ids[0] and duration >= 10
                                for key, duration in durations.items()):
                break
    full_id, short_id = select_videos(paths, durations, requested_ids)
    rows = smoke_rows(full_id, short_id)
    output_dir.mkdir(parents=True, exist_ok=False)
    blocks_path = output_dir / "smoke_blocks.jsonl"
    map_path = output_dir / "smoke_video_paths.json"
    blocks_path.write_text("".join(json.dumps(row) + "\n" for row in rows),
                           encoding="utf-8")
    map_path.write_text(json.dumps({video_id: paths[video_id]
                                    for video_id in (full_id, short_id)}, indent=2),
                        encoding="utf-8")
    validated = load_blocks(blocks_path)
    if set(validated) != {full_id, short_id}:
        raise RuntimeError("Smoke block validation failed")
    manifest = {
        "purpose": "DDP execution smoke only; targets are synthetic, not clinical labels",
        "source_video_map": str(video_map_path.resolve()),
        "full_minute_video_id": full_id,
        "ten_second_video_id": short_id,
        "durations_seconds": {video_id: durations[video_id]
                              for video_id in (full_id, short_id)},
        "expected_updates_per_rank": [6, 1],
        "train_jsonl": str(blocks_path),
        "video_path_map": str(map_path),
    }
    (output_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2), encoding="utf-8")
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--video-path-map", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--video-id", action="append", dest="video_ids",
                        help="Specify two IDs; the 60s video must be first")
    args = parser.parse_args()
    print(json.dumps(prepare(args.video_path_map, args.output_dir,
                             args.video_ids), indent=2))


if __name__ == "__main__":
    main()
