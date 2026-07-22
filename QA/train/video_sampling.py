"""
Video frame sampling utilities for QA SFT.

Two distinct policies:

  - `sample_last_n_frames` (streaming): sample N frames from the LAST few
    seconds of `video_window=[start,end]`, biased to the most recent context.
    Concretely we take the tail window `[end - n_frames, end]` (i.e. last
    `n_frames` seconds ~ 1 frame/sec) and sample N frames inside it. This
    implements: current_time = end, visual_input = last N frames before
    current_time.

  - `sample_uniform_frames` / `sample_full_clip_frames` (offline): sample N
    frames uniformly across the FULL clip `[start, end]`, for whole-clip
    understanding.

The functions return PIL.Image objects because Qwen-VL processors commonly
accept PIL images in chat-template messages.
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
    """Read frames at the given absolute timestamps (seconds)."""
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
            # Try frame 0 as a fallback.
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
    """N uniformly spaced timestamps inside [start_sec, end_sec]."""
    if n_frames == 1:
        return [end_sec]
    step = (end_sec - start_sec) / (n_frames - 1)
    return [start_sec + i * step for i in range(n_frames)]


def sample_last_n_frames(
    video_path: str | Path,
    start_sec: float,
    end_sec: float,
    n_frames: int = 8,
    resize: int | None = 448,
) -> List[Image.Image]:
    """
    Streaming policy: sample N frames from the TAIL of [start_sec, end_sec].

    We take the recent window `[recent_start, end_sec]` where
    `recent_start = max(start_sec, end_sec - n_frames)` (i.e. the last
    `n_frames` seconds, ~1 frame/sec), and uniformly sample N frames inside it.
    This biases the visual input toward the current time (end_sec), matching the
    "last N frames before current_time" data policy.

    If the recent window is shorter than expected (very short clip), we still
    uniformly sample within the available span and pad by duplicating the last
    frame so the output length stays exactly N.

    Args:
        video_path: path to the source video.
        start_sec: window start in absolute video seconds.
        end_sec: window end in absolute video seconds (== current_time).
        n_frames: number of frames to return.
        resize: if int, resize frames to resize x resize.

    Returns:
        List[PIL.Image.Image] of length n_frames.
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
            # Keep a tiny non-empty span to avoid division / seeking issues.
            start_sec = max(0.0, end_sec - 0.5)

        # Tail window: last `n_frames` seconds (~1 frame/sec), clipped to start.
        recent_start = max(start_sec, end_sec - float(n_frames))
        if recent_start >= end_sec:
            recent_start = max(0.0, end_sec - 0.5)

        times = _uniform_times(recent_start, end_sec, n_frames)
        frames = _read_frames_at_times(cap, times, fps, frame_count, resize, video_path)
    finally:
        cap.release()

    # Guarantee exactly n_frames outputs.
    while len(frames) < n_frames:
        frames.append(frames[-1].copy())

    return frames[:n_frames]


def sample_uniform_frames(
    video_path: str | Path,
    start_sec: float,
    end_sec: float,
    n_frames: int = 8,
    resize: int | None = 448,
) -> List[Image.Image]:
    """
    Offline policy: sample N frames uniformly across the FULL span
    [start_sec, end_sec]. Used for whole-clip understanding.
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

        times = _uniform_times(start_sec, end_sec, n_frames)
        frames = _read_frames_at_times(cap, times, fps, frame_count, resize, video_path)
    finally:
        cap.release()

    while len(frames) < n_frames:
        frames.append(frames[-1].copy())

    return frames[:n_frames]


def sample_full_clip_frames(
    video_path: str | Path,
    start_sec: float,
    end_sec: float,
    n_frames: int = 8,
    resize: int | None = 448,
) -> List[Image.Image]:
    """
    Offline QA policy: sample N frames uniformly from the full clip
    [clip_start, clip_end]. Alias of `sample_uniform_frames`.
    """
    return sample_uniform_frames(
        video_path=video_path,
        start_sec=start_sec,
        end_sec=end_sec,
        n_frames=n_frames,
        resize=resize,
    )