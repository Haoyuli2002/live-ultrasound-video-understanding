"""
Dataset for QA WAIT/ANSWER SFT.

Input JSONL:
  QA/results/{video_id}_training_samples.jsonl

Each row should contain:
  - sample_type: offline_answer | streaming_wait | streaming_answer
  - video / video_id
  - video_window: [start, end]
  - question
  - target
  - qa_type
  - meta

The dataset resolves video paths and samples visual frames according to the
current training policy:
  - streaming_*: last N frames inside video_window
  - offline_answer: N frames uniformly from full clip / video_window
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Optional

from PIL import Image

try:
    from .video_sampling import sample_last_n_frames, sample_full_clip_frames
except ImportError:  # allow direct script execution
    from video_sampling import sample_last_n_frames, sample_full_clip_frames


class QATrainingDataset:
    """Lightweight JSONL dataset, intentionally not tied to torch Dataset APIs."""

    def __init__(
        self,
        jsonl_path: str | Path,
        *,
        repo_root: str | Path = ".",
        video_root: str | Path | None = None,
        default_video_path: str | Path | None = None,
        video_path_map: str | Path | None = None,
        window_size: int = 8,
        frame_size: int = 448,
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
            # Try relative to repo root and video root.
            for base in (self.repo_root, self.video_root):
                candidate = base / p
                if candidate.exists():
                    return candidate

        # Common project crawler location fallback.
        if video_id:
            matches = list(self.repo_root.glob(f"UltrasoundCrawler_KeyCode_20260323_v2/output/**/{video_id}.mp4"))
            if matches:
                return matches[0]

        raise FileNotFoundError(
            f"Could not resolve video path for row video_id={video_id!r}. "
            f"Pass --default-video-path or --video-path-map."
        )

    def _sample_frames(self, row: Dict[str, Any]) -> List[Image.Image]:
        video_path = self._resolve_video_path(row)
        start, end = row["video_window"]
        sample_type = row.get("sample_type", "")

        if sample_type == "offline_answer":
            return sample_full_clip_frames(
                video_path,
                float(start),
                float(end),
                n_frames=self.window_size,
                resize=self.frame_size,
            )

        return sample_last_n_frames(
            video_path,
            float(start),
            float(end),
            n_frames=self.window_size,
            resize=self.frame_size,
        )

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        row = dict(self.rows[idx])
        row["frames"] = self._sample_frames(row)
        return row


def load_dataset(*args, **kwargs) -> QATrainingDataset:
    return QATrainingDataset(*args, **kwargs)