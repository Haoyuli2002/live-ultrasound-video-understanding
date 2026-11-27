"""One video's durable 1 FPS spatial features and confirmed, replayable events."""
from dataclasses import dataclass
import json
import math
from pathlib import Path
import sqlite3

import numpy as np
import torch

from .segment import OnlineSegmenter


@dataclass(frozen=True)
class Event:
    number: int
    start: int
    end: float
    available_at: float
    representative_s: int

    @property
    def event_id(self):
        return f"E{self.number:02d}"


def index_identity(video_id, feature_source, threshold=0.02, min_seconds=3):
    return json.dumps([1, video_id, feature_source, threshold, min_seconds])


class EventIndex:
    """Use one writer per database; open again to continue a partially observed video."""

    def __init__(self, path, video_id, feature_source, threshold=0.02, min_seconds=3, *, read_only=False):
        self.video_id = video_id
        self.threshold, self.min_seconds = threshold, min_seconds
        self.db = (sqlite3.connect(Path(path).resolve().as_uri() + "?mode=ro", uri=True)
                   if read_only else sqlite3.connect(path))
        try:
            identity = index_identity(video_id, feature_source, threshold, min_seconds)
            if not read_only:
                self.db.executescript("""
                    CREATE TABLE IF NOT EXISTS stream (
                        id INTEGER PRIMARY KEY CHECK(id=1), identity TEXT, shape TEXT, dtype TEXT, eof REAL);
                    CREATE TABLE IF NOT EXISTS frames (
                        second INTEGER PRIMARY KEY, pts REAL NOT NULL, pooled BLOB NOT NULL, patches BLOB NOT NULL);
                    CREATE TABLE IF NOT EXISTS detector (
                        id INTEGER PRIMARY KEY CHECK(id=1), start INTEGER, next_second INTEGER, total BLOB, pending BLOB);
                    CREATE TABLE IF NOT EXISTS events (
                        number INTEGER PRIMARY KEY, start INTEGER UNIQUE, end REAL, available REAL,
                        representative INTEGER, centroid BLOB);
                """)
                with self.db:
                    self.db.execute("INSERT OR IGNORE INTO stream(id,identity) VALUES(1,?)", (identity,))
            if self.db.execute("SELECT identity FROM stream").fetchone()[0] != identity:
                raise ValueError("Database belongs to another video, encoder or segmentation configuration")
            self._restore()
        except BaseException:
            self.db.close()
            raise

    def _restore(self):
        shape, dtype, self.eof = self.db.execute("SELECT shape,dtype,eof FROM stream").fetchone()
        self.shape = tuple(json.loads(shape)) if shape else None
        self.dtype = getattr(torch, dtype) if dtype else None
        self.detector = OnlineSegmenter(self.threshold, self.min_seconds)
        state = self.db.execute("SELECT start,next_second,total,pending FROM detector").fetchone()
        if state:
            self.detector.start, self.detector.next_second, total, pending = state
            self.detector.total = np.frombuffer(total, dtype=np.float64).copy()
            if pending is not None:
                self.detector.pending = np.frombuffer(pending, dtype=np.float64).copy()

    def _begin_update(self):
        self.db.execute("BEGIN IMMEDIATE")
        next_second, eof = self.db.execute(
            "SELECT COALESCE((SELECT MAX(second) FROM frames), -1)+1, eof FROM stream").fetchone()
        if eof is not None or next_second != self.detector.next_second:
            raise RuntimeError("Stream advanced in another writer; retry at its new position")

    def close(self):
        self.db.close()

    @property
    def observed_until(self):
        return self.eof if self.eof is not None else self.detector.next_second - 1

    def _publish(self, start, end, available, direction_sum):
        center = direction_sum / max(np.linalg.norm(direction_sum), 1e-12)
        # Scan pooled vectors once to choose a real representative; keep spatial
        # features on disk until they are requested by the scorer or Q-Former.
        rows = self.db.execute("SELECT second,pooled FROM frames WHERE second>=? AND second<? ORDER BY second",
                               (start, end))
        best = -math.inf
        representative = start
        for second, raw in rows:
            vector = np.frombuffer(raw, dtype=np.float64)
            similarity = float(vector @ center) / max(np.linalg.norm(vector), 1e-12)
            if similarity > best:
                representative, best = second, similarity
        self.db.execute("INSERT INTO events(start,end,available,representative,centroid) VALUES(?,?,?,?,?)",
                        (start, end, available, representative, center.astype('float32').tobytes()))

    def push(self, second, source_pts, patches):
        """Append the latest observed frame at this integer second, shape [patch, width]."""
        if self.eof is not None or second != self.detector.next_second or not isinstance(second, int):
            raise ValueError("Expected the next integer second of an open stream")
        if not math.isfinite(source_pts) or not 0 <= source_pts <= second:
            raise ValueError("Frame PTS must be observable at the sample time")
        if patches.ndim != 2 or min(patches.shape) < 1 or patches.dtype not in (torch.float32, torch.bfloat16, torch.float16):
            raise ValueError("Expected floating spatial tokens [patch, width]")
        if self.shape and (tuple(patches.shape) != self.shape or patches.dtype != self.dtype):
            raise ValueError("Feature shape or dtype changed within the stream")
        features = patches.detach().cpu().contiguous()
        pooled = features.float().mean(0).double().numpy()
        start = self.detector.start
        previous_sum = self.detector.total
        try:
            with self.db:
                self._begin_update()
                change, drift = self.detector.update(pooled)
                self.db.execute("INSERT OR REPLACE INTO detector VALUES(1,?,?,?,?)",
                                (self.detector.start, self.detector.next_second, self.detector.total.tobytes(),
                                 None if self.detector.pending is None else self.detector.pending.tobytes()))
                if self.shape is None:
                    self.db.execute("UPDATE stream SET shape=?,dtype=?",
                                    (json.dumps(list(features.shape)), str(features.dtype).split('.')[-1]))
                self.db.execute("INSERT INTO frames VALUES(?,?,?,?)",
                                (second, source_pts, pooled.tobytes(), memoryview(features.view(torch.uint8).numpy())))
                if change:
                    self._publish(start, change[0], change[1], previous_sum)
        except BaseException:
            self._restore()
            raise
        self.shape, self.dtype = tuple(features.shape), features.dtype
        return change, drift

    def finish(self, end):
        """Explicit EOF seals the last event; an unconfirmed candidate stays in that event."""
        if self.eof is not None:
            if end != self.eof:
                raise ValueError("Conflicting end time")
            return
        if not self.shape or not self.observed_until < end <= self.observed_until + 1:
            raise ValueError("EOF must follow the last 1 FPS sample by at most one second")
        try:
            with self.db:
                self._begin_update()
                direction_sum = self.detector.total
                if self.detector.pending is not None:
                    direction_sum = direction_sum + self.detector.pending
                self._publish(self.detector.start, end, end, direction_sum)
                self.db.execute("UPDATE stream SET eof=?", (end,))
        except BaseException:
            self._restore()
            raise
        self.eof = end

    def events_at(self, second, *, exclude_recent_frames=0):
        """Published events with at least one frame preceding the excluded short window."""
        if not math.isfinite(second) or not 0 <= second <= self.observed_until:
            raise ValueError("Requested time is outside the observed stream")
        if not isinstance(exclude_recent_frames, int) or exclude_recent_frames < 0:
            raise ValueError("Expected a nonnegative frame count")
        # At an exact-integer EOF there is no frame sampled at the EOF itself.
        last_sample = min(math.floor(second), self.detector.next_second - 1)
        window_start = max(0, last_sample + 1 - exclude_recent_frames)
        return [Event(*row) for row in self.db.execute(
            "SELECT number,start,end,available,representative FROM events "
            "WHERE available<=? AND start<? ORDER BY number", (second, window_start))]

    def frame(self, second):
        row = self.db.execute("SELECT pts,patches FROM frames WHERE second=?", (second,)).fetchone()
        if row is None:
            raise KeyError(second)
        return row[0], torch.frombuffer(bytearray(row[1]), dtype=self.dtype).view(self.shape)
