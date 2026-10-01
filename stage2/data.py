"""Strict chronological teacher labels and causal one-frame-per-second video input."""
from __future__ import annotations

import json
from pathlib import Path


def load_blocks(path: str | Path) -> dict[str, list[dict]]:
    videos: dict[str, list[dict]] = {}
    for line_no, line in enumerate(Path(path).read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        row = json.loads(line)
        video_id = row.get("video_id")
        window = row.get("block_window")
        if not video_id or not isinstance(window, list) or len(window) != 2:
            raise ValueError(f"line {line_no}: video_id and block_window are required")
        start, end = map(float, window)
        if start < 0 or end <= start or end - start > 60.01:
            raise ValueError(f"line {line_no}: invalid block_window")
        row["block_window"] = [start, end]
        row["_line_no"] = line_no
        videos.setdefault(str(video_id), []).append(row)

    if not videos:
        raise ValueError("No Stage 2 blocks found")
    for video_id, blocks in videos.items():
        blocks.sort(key=lambda r: r["block_window"][0])
        expected_start = 0.0
        for index, row in enumerate(blocks):
            start, end = row["block_window"]
            if abs(start - expected_start) > 0.01:
                raise ValueError(f"{video_id}: gap, overlap, or duplicate before block {index}")
            duration = end - start
            full = abs(duration - 60) <= 0.01
            if not full and index != len(blocks) - 1:
                raise ValueError(f"{video_id}: incomplete block must be last")
            local = row.get("local_sub_summaries")
            if not isinstance(local, list):
                raise ValueError(f"{video_id}: local_sub_summaries required")
            expected_local = int((duration + 0.01) // 10)
            if len(local) != expected_local:
                raise ValueError(f"{video_id}: expected {expected_local} local labels")
            for j, label in enumerate(local):
                a, b = map(float, label.get("window", []))
                if abs(a - (start + 10 * j)) > 0.01 or abs(b - (start + 10 * (j + 1))) > 0.01:
                    raise ValueError(f"{video_id}: local window {j} is not chronological")
                if not str(label.get("local_summary_target", "")).strip():
                    raise ValueError(f"{video_id}: missing local summary {j}")
            if full and (not str(row.get("block_summary_target", "")).strip()
                         or not str(row.get("global_summary_target", "")).strip()):
                raise ValueError(f"{video_id}: full block needs independent block and global labels")
            expected_start = end
    return videos


class VideoReader:
    def __init__(self, path: str | Path, image_size: int):
        import cv2
        self.cv2 = cv2
        self.cap = cv2.VideoCapture(str(path))
        if not self.cap.isOpened():
            raise FileNotFoundError(f"Cannot open video: {path}")
        self.fps = self.cap.get(cv2.CAP_PROP_FPS)
        self.count = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
        self.image_size = image_size
        if not self.fps or self.fps <= 0:
            raise ValueError(f"Invalid FPS: {path}")

    def frame_ending_at(self, second: int):
        """Frame at the end of [second, second+1), never from a later interval."""
        from PIL import Image
        frame_index = min(self.count - 1, int((second + 1) * self.fps - 1e-6))
        if frame_index < 0:
            raise ValueError("Video has no frames")
        self.cap.set(self.cv2.CAP_PROP_POS_FRAMES, frame_index)
        ok, bgr = self.cap.read()
        if not ok:
            raise RuntimeError(f"Failed to decode frame {frame_index}")
        rgb = self.cv2.cvtColor(bgr, self.cv2.COLOR_BGR2RGB)
        image = Image.fromarray(rgb)
        image.thumbnail((self.image_size, self.image_size))
        canvas = Image.new("RGB", (self.image_size, self.image_size))
        canvas.paste(image, ((self.image_size - image.width) // 2,
                             (self.image_size - image.height) // 2))
        return canvas

    def close(self):
        self.cap.release()
