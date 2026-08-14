"""
Dataset for ultrasound ASR caption-completion pretraining.

Reads pretrain_samples.jsonl (produced by pretrain/build_samples.py) and samples
video frames for each sample. Independent from QA/train.

Each row:
  {
    "sample_type": "pretrain_caption",
    "video_id": "...",
    "video_window": [start, end],   # end == segment.start (current_time)
    "prev_context": "...",
    "target": "...",
    "meta": {...}
  }

__getitem__ returns the row plus "frames": [PIL.Image, ...].
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Optional

from PIL import Image

try:
    from .video_sampling import sample_last_n_frames, sample_uniform_n_frames
except ImportError:  # allow direct script execution
    from video_sampling import sample_last_n_frames, sample_uniform_n_frames


class PretrainCaptionDataset:
    def __init__(
        self,
        jsonl_path: str | Path,
        *,
        repo_root: str | Path = ".",
        video_root: str | Path | None = None,
        default_video_path: str | Path | None = None,
        video_path_map: str | Path | None = None,
        window_size: int = 4,
        frame_size: int = 224,
        limit: Optional[int] = None,
    ):
        self.jsonl_path = Path(jsonl_path)
        self.repo_root = Path(repo_root)
        self.video_root = Path(video_root) if video_root else self.repo_root
        self.default_video_path = Path(default_video_path) if default_video_path else None
        self.window_size = int(window_size)
        self.frame_size = int(frame_size)

        if self.window_size <= 0:
            raise ValueError("window_size must be positive")

        self.video_map: Dict[str, str] = {}
        if video_path_map:
            with open(video_path_map, "r", encoding="utf-8") as f:
                self.video_map = json.load(f)

        self.rows: List[Dict[str, Any]] = []
        with open(self.jsonl_path, "r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    self.rows.append(json.loads(line))
                if limit is not None and len(self.rows) >= limit:
                    break

        if not self.rows:
            raise ValueError(f"No rows loaded from {self.jsonl_path}")

    def __len__(self) -> int:
        return len(self.rows)

    def _resolve_video_path(self, row: Dict[str, Any]) -> Path:
        video_id = row.get("video_id")
        if video_id and video_id in self.video_map:
            p = Path(self.video_map[video_id])
            return p if p.is_absolute() else self.repo_root / p

        if self.default_video_path is not None:
            return self.default_video_path if self.default_video_path.is_absolute() else self.repo_root / self.default_video_path

        video_field = row.get("video")
        if video_field:
            p = Path(video_field)
            if p.exists():
                return p
            for base in (self.repo_root, self.video_root):
                candidate = base / p
                if candidate.exists():
                    return candidate

        if video_id:
            matches = list(self.repo_root.glob(f"**/{video_id}.mp4"))
            if matches:
                return matches[0]

        raise FileNotFoundError(
            f"Could not resolve video path for video_id={video_id!r}. "
            f"Pass --default-video-path or --video-path-map."
        )

    def _sample_frames(self, row: Dict[str, Any]) -> List[Image.Image]:
        video_path = self._resolve_video_path(row)
        start, end = row["video_window"]
        return sample_last_n_frames(
            video_path,
            float(start),
            float(end),
            n_frames=self.window_size,
            resize=self.frame_size,
        )

    def _sample_interleave_history_frames(self, row: Dict[str, Any]) -> List[List[Image.Image]]:
        video_path = self._resolve_video_path(row)
        history_frames: List[List[Image.Image]] = []

        for item in row.get("history", []):
            start, end = item["video_window"]
            n_frames = int(item.get("num_frames", 3))
            history_frames.append(
                sample_uniform_n_frames(
                    video_path,
                    float(start),
                    float(end),
                    n_frames=n_frames,
                    resize=self.frame_size,
                )
            )

        return history_frames

    def _sample_current_visual_frames(self, row: Dict[str, Any]) -> List[Image.Image]:
        video_path = self._resolve_video_path(row)
        current_visual = row.get("current_visual") or {}
        start, end = current_visual.get("video_window", row.get("video_window", [0.0, 0.0]))
        n_frames = int(current_visual.get("num_frames", 3))
        return sample_uniform_n_frames(
            video_path,
            float(start),
            float(end),
            n_frames=n_frames,
            resize=self.frame_size,
        )

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        row = dict(self.rows[idx])
        sample_type = row.get("sample_type")
        if sample_type in {"pretrain_caption_sentence_interleave", "pretrain_next_sentence_mixedmask"}:
            row["history_frames"] = self._sample_interleave_history_frames(row)
            row["frames"] = []
            if sample_type == "pretrain_next_sentence_mixedmask":
                row["current_visual_frames"] = self._sample_current_visual_frames(row)
        else:
            row["frames"] = self._sample_frames(row)
        return row


def load_dataset(*args, **kwargs) -> PretrainCaptionDataset:
    return PretrainCaptionDataset(*args, **kwargs)