"""Dataset for Stage-2 two-level memory compression."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Optional

from PIL import Image

try:
    from .video_sampling import sample_uniform_n_frames
except ImportError:
    from video_sampling import sample_uniform_n_frames


class MemoryCompressionDataset:
    def __init__(
        self,
        jsonl_path: str | Path,
        *,
        repo_root: str | Path = ".",
        video_root: str | Path | None = None,
        default_video_path: str | Path | None = None,
        video_path_map: str | Path | None = None,
        short_frames: int = 2,
        frame_size: int = 224,
        max_short_windows: int | None = None,
        max_previous_blocks: int | None = None,
        limit: Optional[int] = None,
    ):
        self.jsonl_path = Path(jsonl_path)
        self.repo_root = Path(repo_root)
        self.video_root = Path(video_root) if video_root else self.repo_root
        self.default_video_path = Path(default_video_path) if default_video_path else None
        self.short_frames = int(short_frames)
        self.frame_size = int(frame_size)
        self.max_short_windows = max_short_windows
        self.max_previous_blocks = max_previous_blocks
        if self.short_frames <= 0:
            raise ValueError("short_frames must be positive")

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

    def __len__(self):
        return len(self.rows)

    def _resolve_video_path(self, row: Dict[str, Any]) -> Path:
        vid = row.get("video_id")
        if vid and vid in self.video_map:
            p = Path(self.video_map[vid])
            return p if p.is_absolute() else self.repo_root / p
        if self.default_video_path is not None:
            return self.default_video_path if self.default_video_path.is_absolute() else self.repo_root / self.default_video_path
        video = row.get("video")
        if video:
            p = Path(video)
            if p.exists():
                return p
            for base in (self.repo_root, self.video_root):
                c = base / p
                if c.exists():
                    return c
        if vid:
            matches = list(self.repo_root.glob(f"**/{vid}.mp4"))
            if matches:
                return matches[0]
        raise FileNotFoundError(f"Could not resolve video path for video_id={vid!r}")

    def _frames(self, video_path: Path, window) -> List[Image.Image]:
        start, end = window
        return sample_uniform_n_frames(video_path, float(start), float(end), n_frames=self.short_frames, resize=self.frame_size)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        row = dict(self.rows[idx])
        video_path = self._resolve_video_path(row)
        typ = row.get("sample_type")
        if typ not in {"short_memory_compression", "long_memory_compression"}:
            raise ValueError(f"Unsupported sample_type: {typ}")
        windows = row.get("short_windows") or []
        if self.max_short_windows is not None:
            windows = windows[: int(self.max_short_windows)]
        row["short_windows"] = windows
        row["short_frames_list"] = [self._frames(video_path, w) for w in windows]
        if typ == "long_memory_compression":
            prev_blocks = row.get("previous_blocks") or []
            if self.max_previous_blocks is not None:
                prev_blocks = prev_blocks[-int(self.max_previous_blocks):]
            loaded_prev = []
            for block in prev_blocks:
                bwindows = block.get("short_windows") or []
                if self.max_short_windows is not None:
                    bwindows = bwindows[: int(self.max_short_windows)]
                loaded_prev.append({
                    **block,
                    "short_windows": bwindows,
                    "short_frames_list": [self._frames(video_path, w) for w in bwindows],
                })
            row["previous_blocks"] = loaded_prev
        return row


def load_dataset(*args, **kwargs) -> MemoryCompressionDataset:
    return MemoryCompressionDataset(*args, **kwargs)