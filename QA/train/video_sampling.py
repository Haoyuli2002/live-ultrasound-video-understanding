"""
Video frame sampling utilities for QA SFT.

Current v1 policy:
  - `last_n_frames`: sample the last N frames inside `video_window=[start,end]`.
  - This implements: current_time = end, visual_input = last N frames before current_time.

The functions return PIL.Image objects because Qwen-VL processors commonly
accept PIL images in chat-template messages.
"""

from __future__ import annotations

from pathlib import Path
from typing import List

import cv2
from PIL import Image


def sample_last_n_frames(
    video_path: str | Path,
    start_sec: float,
    end_sec: float,
    n_frames: int = 8,
    resize: int | None = 448,
) -> List[Image.Image]:
    """
    Sample N frames from [start_sec, end_sec], biased to the most recent context.

    Implementation detail:
    - We sample N uniformly spaced timestamps inside [start_sec, end_sec].
    - Since the interval itself is the recent window ending at current_time,
      this corresponds to "last N frames" at the data-policy level.
    - If the clip is very short, duplicate timestamps may occur; we keep them
      so the output length stays exactly N.

    Args:
        video_path: path to the source video.
        start_sec: window start in absolute video seconds.
        end_sec: window end in absolute video seconds.
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

    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    duration = frame_count / fps if frame_count > 0 else max(end_sec, 0.0)

    start_sec = max(0.0, float(start_sec))
    end_sec = min(float(end_sec), duration)
    if end_sec <= start_sec:
        # Keep a tiny non-empty span to avoid division / seeking issues.
        start_sec = max(0.0, end_sec - 0.5)

    if n_frames == 1:
        times = [end_sec]
    else:
        # Uniform timestamps in the recent time window.
        step = (end_sec - start_sec) / (n_frames - 1)
        times = [start_sec + i * step for i in range(n_frames)]

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
                cap.release()
                raise RuntimeError(f"Failed to read any frame from {video_path}")

        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        img = Image.fromarray(frame)
        if resize is not None:
            img = img.resize((resize, resize), Image.BICUBIC)
        last_valid = img
        frames.append(img)

    cap.release()

    # Guarantee exactly n_frames outputs.
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
    Offline QA policy: sample N frames uniformly from the full clip.
    This currently shares the same implementation as sample_last_n_frames
    over [clip_start, clip_end].
    """
    return sample_last_n_frames(
        video_path=video_path,
        start_sec=start_sec,
        end_sec=end_sec,
        n_frames=n_frames,
        resize=resize,
    )