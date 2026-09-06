"""Dataset for recurrent <SUMMARY> streaming QA/SFT.

Expected JSONL row:
{
  "video_id": "...",
  "history_chunks": [{"video_window": [0, 10], "text": "optional ASR"}],
  "current_visual": {"video_window": [10, 20]},
  "question": "...",
  "target": "<WAIT> ..." or "<ANSWER> ..."
}

For backward compatibility, rows with only `video_window` are treated as one
current_visual window and no history chunks.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Optional

from PIL import Image

try:
    from .video_sampling import sample_uniform_frames
except ImportError:
    from video_sampling import sample_uniform_frames


class SummaryDecideDataset:
    def __init__(
        self,
        jsonl_path: str | Path,
        *,
        repo_root: str | Path = ".",
        video_root: str | Path | None = None,
        default_video_path: str | Path | None = None,
        video_path_map: str | Path | None = None,
        frames_per_chunk: int = 3,
        frame_size: int = 224,
        limit: Optional[int] = None,
    ):
        self.jsonl_path = Path(jsonl_path)
        self.repo_root = Path(repo_root)
        self.video_root = Path(video_root) if video_root else self.repo_root
        self.default_video_path = Path(default_video_path) if default_video_path else None
        self.frames_per_chunk = int(frames_per_chunk)
        self.frame_size = int(frame_size)
        if self.frames_per_chunk <= 0:
            raise ValueError("frames_per_chunk must be positive")

        self.video_map: Dict[str, str] = {}
        if video_path_map:
            with open(video_path_map, encoding="utf-8") as f:
                self.video_map = json.load(f)

        self.rows: List[Dict[str, Any]] = []
        with self.jsonl_path.open(encoding="utf-8") as f:
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
            matches = list(self.repo_root.glob(f"UltrasoundCrawler_KeyCode_20260323_v2/output/**/{video_id}.mp4"))
            if matches:
                return matches[0]
        raise FileNotFoundError(f"Could not resolve video path for video_id={video_id!r}")

    def _frames(self, video_path: Path, window) -> List[Image.Image]:
        start, end = window
        return sample_uniform_frames(
            video_path,
            float(start),
            float(end),
            n_frames=self.frames_per_chunk,
            resize=self.frame_size,
        )

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        row = dict(self.rows[idx])
        video_path = self._resolve_video_path(row)
        history_chunks = row.get("history_chunks") or []
        current_visual = row.get("current_visual") or {"video_window": row.get("video_window")}
        if not current_visual.get("video_window"):
            raise ValueError("Row needs current_visual.video_window or video_window")

        row["history_chunks"] = [
            {
                **chunk,
                "frames": self._frames(video_path, chunk["video_window"]),
                "text": str(chunk.get("text") or ""),
            }
            for chunk in history_chunks
        ]
        row["current_visual_frames"] = self._frames(video_path, current_visual["video_window"])
        target = str(row.get("target") or "")
        row["answerability_label"] = int(row.get("answerability_label", 0 if target.strip().upper().startswith("<WAIT>") else 1))
        return row
