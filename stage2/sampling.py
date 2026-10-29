"""Bound frames before the video reaches the multimodal server."""
from __future__ import annotations

from fractions import Fraction
import json
import math
from pathlib import Path
import subprocess

POLICY = "bounded_silent_clip_v1"


def sampling_plan(start: int, end: int, fps: float, max_frames: int) -> dict:
    if start < 0 or end <= start or not math.isfinite(fps) or fps <= 0:
        raise ValueError("Expected a positive video interval and FPS")
    if max_frames < 4 or max_frames % 2:
        raise ValueError("Frame cap must be even and at least four")
    duration = end-start
    frames = max(4, min(max_frames, math.floor(duration*fps)))
    frames -= frames % 2
    rate = Fraction(frames, duration)
    return {"frames": frames, "fps": float(rate), "fps_filter": str(rate),
            "duration": duration, "policy": POLICY}


def clip_metadata(path: Path) -> dict:
    result = subprocess.run(["ffprobe", "-v", "error", "-show_streams", "-of", "json", str(path)],
                            capture_output=True, text=True, check=True)
    streams = json.loads(result.stdout)["streams"]
    videos = [s for s in streams if s.get("codec_type") == "video"]
    if len(videos) != 1:
        raise ValueError("Expected one video stream")
    stream = videos[0]
    return {"frames": int(stream["nb_frames"]),
            "fps": float(Fraction(stream["avg_frame_rate"])),
            "duration": float(stream["duration"]),
            "audio_streams": sum(s.get("codec_type") == "audio" for s in streams)}


def validate_clip(path: Path, plan: dict) -> dict:
    observed = clip_metadata(path)
    if (observed["audio_streams"] or observed["frames"] != plan["frames"]
            or abs(observed["duration"]-plan["duration"]) > 0.05
            or abs(observed["fps"]-plan["fps"]) > 1e-4):
        raise RuntimeError(f"Silent clip differs from its sampling plan: {observed}; expected {plan}")
    return observed
