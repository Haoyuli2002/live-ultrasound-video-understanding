#!/usr/bin/env python3
"""Transcode videos that OpenCV cannot decode into H264 MP4.

This utility is intended for the pretrain pipeline. ASR can succeed on videos
whose audio is readable, while pretrain can still fail if ``cv2.VideoCapture``
cannot decode video frames. This script scans a video path map, identifies
videos that OpenCV cannot read, and optionally transcodes them to a more
compatible H264/yuv420p MP4 while preserving the original path.

Example:

    python scripts/transcode_videos_for_opencv.py \
      --video-path-map cluster_data/splits/train_videos.json \
      --replace \
      --delete-backup

By default the script only scans and reports unreadable videos. Use
``--replace`` to perform transcoding/replacement.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Iterable

import cv2


def set_quiet_opencv_logs() -> None:
    """Reduce FFmpeg/OpenCV log spam where supported."""
    os.environ.setdefault("OPENCV_LOG_LEVEL", "ERROR")
    os.environ.setdefault("OPENCV_FFMPEG_LOGLEVEL", "-8")


def can_read_first_frame(video_path: Path) -> tuple[bool, str]:
    """Return whether OpenCV can read the first frame from a video."""
    if not video_path.exists():
        return False, "missing file"

    cap = cv2.VideoCapture(str(video_path))
    ok, frame = cap.read()
    cap.release()

    if not ok or frame is None:
        return False, "failed to read first frame"

    return True, f"shape={frame.shape}"


def load_video_paths(video_path_map: Path) -> list[tuple[str, Path]]:
    """Load ``video_id -> video_path`` entries from a JSON map."""
    with video_path_map.open("r", encoding="utf-8") as f:
        data = json.load(f)

    if not isinstance(data, dict):
        raise ValueError(f"Expected JSON object in {video_path_map}, got {type(data).__name__}")

    return [(str(video_id), Path(path)) for video_id, path in data.items()]


def scan_bad_videos(entries: Iterable[tuple[str, Path]]) -> list[tuple[str, Path, str]]:
    """Return all videos whose first frame cannot be read by OpenCV."""
    bad: list[tuple[str, Path, str]] = []

    for video_id, video_path in entries:
        ok, detail = can_read_first_frame(video_path)
        if ok:
            print(f"OK  {video_id} {video_path} ({detail})")
        else:
            print(f"BAD {video_id} {video_path} ({detail})")
            bad.append((video_id, video_path, detail))

    return bad


def transcode_to_h264(
    video_id: str,
    video_path: Path,
    *,
    ffmpeg_bin: str,
    crf: int,
    preset: str,
    audio_bitrate: str,
    keep_backup: bool,
    delete_backup: bool,
) -> None:
    """Transcode a video to H264 and replace the original path atomically."""
    tmp_path = video_path.with_name(f"{video_path.name}.h264.tmp.mp4")
    backup_path = video_path.with_name(f"{video_path.name}.pre_h264_backup")

    if tmp_path.exists():
        tmp_path.unlink()

    print(f"[transcode] {video_id}: {video_path} -> {tmp_path}")

    cmd = [
        ffmpeg_bin,
        "-nostdin",
        "-y",
        "-hide_banner",
        "-loglevel",
        "error",
        "-i",
        str(video_path),
        "-c:v",
        "libx264",
        "-preset",
        preset,
        "-crf",
        str(crf),
        "-pix_fmt",
        "yuv420p",
        "-c:a",
        "aac",
        "-b:a",
        audio_bitrate,
        str(tmp_path),
    ]

    result = subprocess.run(cmd, stdin=subprocess.DEVNULL, text=True)
    if result.returncode != 0:
        raise RuntimeError(f"ffmpeg failed for {video_id}: {video_path}")

    ok, detail = can_read_first_frame(tmp_path)
    if not ok:
        tmp_path.unlink(missing_ok=True)
        raise RuntimeError(f"Transcoded file is still unreadable for {video_id}: {detail}")

    print(f"[transcode] {video_id}: tmp readable ({detail})")

    if backup_path.exists():
        backup_path.unlink()

    shutil.move(str(video_path), str(backup_path))
    shutil.move(str(tmp_path), str(video_path))

    ok, detail = can_read_first_frame(video_path)
    if not ok:
        # Attempt to restore the original file if final replacement failed.
        if video_path.exists():
            video_path.unlink()
        shutil.move(str(backup_path), str(video_path))
        raise RuntimeError(f"Final replacement unreadable for {video_id}; restored backup. Detail: {detail}")

    print(f"[transcode] {video_id}: final readable ({detail})")

    if delete_backup:
        backup_path.unlink(missing_ok=True)
        print(f"[transcode] {video_id}: deleted backup {backup_path}")
    elif keep_backup:
        print(f"[transcode] {video_id}: kept backup {backup_path}")
    else:
        backup_path.unlink(missing_ok=True)
        print(f"[transcode] {video_id}: deleted backup {backup_path}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Scan videos with OpenCV and transcode unreadable videos to H264 MP4.",
    )
    parser.add_argument(
        "--video-path-map",
        required=True,
        type=Path,
        help="JSON mapping from video_id to video path.",
    )
    parser.add_argument(
        "--replace",
        action="store_true",
        help="Actually transcode and replace unreadable videos. Without this flag, only scan/report.",
    )
    parser.add_argument(
        "--keep-backup",
        action="store_true",
        help="Keep <video>.pre_h264_backup files after successful replacement.",
    )
    parser.add_argument(
        "--delete-backup",
        action="store_true",
        help="Delete backup files after successful replacement. This is the default unless --keep-backup is set.",
    )
    parser.add_argument(
        "--ffmpeg-bin",
        default="ffmpeg",
        help="ffmpeg executable path. Default: ffmpeg",
    )
    parser.add_argument(
        "--crf",
        type=int,
        default=23,
        help="H264 CRF value. Lower is higher quality/larger file. Default: 23",
    )
    parser.add_argument(
        "--preset",
        default="veryfast",
        help="libx264 preset. Default: veryfast",
    )
    parser.add_argument(
        "--audio-bitrate",
        default="128k",
        help="AAC audio bitrate. Default: 128k",
    )
    parser.add_argument(
        "--quiet-opencv",
        action="store_true",
        help="Set OpenCV/FFmpeg log environment variables before scanning.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    if args.quiet_opencv:
        set_quiet_opencv_logs()

    if args.keep_backup and args.delete_backup:
        raise ValueError("Use only one of --keep-backup or --delete-backup.")

    entries = load_video_paths(args.video_path_map)

    print("=" * 80)
    print(f"[scan] video_path_map: {args.video_path_map}")
    print(f"[scan] total videos: {len(entries)}")
    print("=" * 80)

    bad = scan_bad_videos(entries)

    print("=" * 80)
    print(f"[scan] bad_count: {len(bad)}")
    for video_id, video_path, detail in bad:
        print(f"[scan] BAD {video_id} {video_path} ({detail})")

    if not bad:
        print("[done] All videos are readable by OpenCV.")
        return

    if not args.replace:
        print("[done] Scan only. Re-run with --replace to transcode bad videos.")
        return

    print("=" * 80)
    print("[replace] Transcoding bad videos to H264...")
    print("=" * 80)

    failures: list[tuple[str, Path, str]] = []
    for video_id, video_path, _detail in bad:
        try:
            transcode_to_h264(
                video_id,
                video_path,
                ffmpeg_bin=args.ffmpeg_bin,
                crf=args.crf,
                preset=args.preset,
                audio_bitrate=args.audio_bitrate,
                keep_backup=args.keep_backup,
                delete_backup=args.delete_backup or not args.keep_backup,
            )
        except Exception as exc:  # noqa: BLE001 - CLI should continue and report all failures.
            print(f"[error] {video_id}: {exc}", file=sys.stderr)
            failures.append((video_id, video_path, str(exc)))

    print("=" * 80)
    print("[verify] Re-scanning after transcode...")
    print("=" * 80)
    final_bad = scan_bad_videos(entries)

    print("=" * 80)
    print(f"[summary] initial_bad_count: {len(bad)}")
    print(f"[summary] transcode_failures: {len(failures)}")
    print(f"[summary] final_bad_count: {len(final_bad)}")

    if failures:
        for video_id, video_path, error in failures:
            print(f"[summary] FAILURE {video_id} {video_path}: {error}")

    if final_bad:
        sys.exit(1)


if __name__ == "__main__":
    main()