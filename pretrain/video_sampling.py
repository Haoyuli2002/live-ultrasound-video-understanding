"""
Frame sampling for ultrasound ASR caption pretraining.

Independent from QA/train. Given a video and a time window [start, end], sample
N frames from the TAIL of the window (biased to `end` == current_time), so the
model sees the most recent frames before it must continue the narration.
"""

from __future__ import annotations

from pathlib import Path
from typing import List

import cv2
from PIL import Image


def _read_frames_at_times(
    cap: "cv2.VideoCapture",
    times: List[float],
    fps: float,
    frame_count: int,
    resize: int | None,
    video_path: Path,
) -> List[Image.Image]:
    frames: List[Image.Image] = []
    last_valid = None

    for t in times:
        frame_idx = int(round(t * fps))
        if frame_count > 0:
            frame_idx = min(max(frame_idx, 0), frame_count - 1)

        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        ok, frame = cap.read()
        if not ok:
            if last_valid is not None:
                frames.append(last_valid.copy())
                continue
            cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            ok, frame = cap.read()
            if not ok:
                raise RuntimeError(f"Failed to read any frame from {video_path}")

        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        img = Image.fromarray(frame)
        if resize is not None:
            img = img.resize((resize, resize), Image.BICUBIC)
        last_valid = img
        frames.append(img)

    return frames


def _uniform_times(start_sec: float, end_sec: float, n_frames: int) -> List[float]:
    if n_frames == 1:
        return [end_sec]
    step = (end_sec - start_sec) / (n_frames - 1)
    return [start_sec + i * step for i in range(n_frames)]


def sample_uniform_n_frames(
    video_path: str | Path,
    start_sec: float,
    end_sec: float,
    n_frames: int = 3,
    resize: int | None = 224,
) -> List[Image.Image]:
    """Sample N frames uniformly from the full [start_sec, end_sec] window."""
    video_path = Path(video_path)
    if not video_path.exists():
        raise FileNotFoundError(f"Video not found: {video_path}")

    if n_frames <= 0:
        raise ValueError(f"n_frames must be positive, got {n_frames}")

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {video_path}")

    try:
        fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
        duration = frame_count / fps if frame_count > 0 else max(end_sec, 0.0)

        start_sec = max(0.0, float(start_sec))
        end_sec = min(float(end_sec), duration) if duration > 0 else max(0.0, float(end_sec))
        if end_sec <= start_sec:
            start_sec = max(0.0, end_sec - 0.5)

        times = _uniform_times(start_sec, end_sec, n_frames)
        frames = _read_frames_at_times(cap, times, fps, frame_count, resize, video_path)
    finally:
        cap.release()

    while len(frames) < n_frames:
        frames.append(frames[-1].copy())

    return frames[:n_frames]


def sample_last_n_frames(
    video_path: str | Path,
    start_sec: float,
    end_sec: float,
    n_frames: int = 4,
    resize: int | None = 224,
) -> List[Image.Image]:
    """
    Sample N frames from the tail of [start_sec, end_sec], biased to end_sec.

    For pretraining, end_sec == segment.start (current_time), and we look at the
    last `n_frames` seconds before the narration sentence begins.
    """
    video_path = Path(video_path)
    if not video_path.exists():
        raise FileNotFoundError(f"Video not found: {video_path}")

    if n_frames <= 0:
        raise ValueError(f"n_frames must be positive, got {n_frames}")

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {video_path}")

    try:
        fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
        duration = frame_count / fps if frame_count > 0 else max(end_sec, 0.0)

        start_sec = max(0.0, float(start_sec))
        end_sec = min(float(end_sec), duration) if duration > 0 else max(0.0, float(end_sec))
        if end_sec <= start_sec:
            start_sec = max(0.0, end_sec - 0.5)

        recent_start = max(start_sec, end_sec - float(n_frames))
        if recent_start >= end_sec:
            recent_start = max(0.0, end_sec - 0.5)

        times = _uniform_times(recent_start, end_sec, n_frames)
        frames = _read_frames_at_times(cap, times, fps, frame_count, resize, video_path)
    finally:
        cap.release()

    while len(frames) < n_frames:
        frames.append(frames[-1].copy())

    return frames[:n_frames]