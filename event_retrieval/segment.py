"""Causal event boundaries from one pooled visual vector per second."""
import numpy as np


class OnlineSegmenter:
    def __init__(self, threshold=0.02, min_seconds=3, start=0):
        if not 0 <= threshold <= 2 or min_seconds < 1:
            raise ValueError("Invalid segmentation threshold or minimum duration")
        self.threshold = threshold
        self.min_seconds = min_seconds
        self.start = self.next_second = start
        self.total = self.pending = None

    def update(self, vector):
        x = np.asarray(vector, dtype=np.float64)
        if x.ndim != 1 or not x.size or not np.isfinite(x).all():
            raise ValueError("Expected a finite frame vector")
        x = x / max(np.linalg.norm(x), 1e-12)
        t = self.next_second
        boundary = None
        if self.total is None:
            self.total = x
            drift = 0.
        else:
            drift = float(1 - np.dot(x, self.total) / max(np.linalg.norm(self.total), 1e-12))
            if drift >= self.threshold and t - self.start >= self.min_seconds:
                if self.pending is None:
                    self.pending = x
                else:
                    self.start = t - 1
                    boundary = (self.start, t)
                    self.total = self.pending + x
                    self.pending = None
            else:
                if self.pending is not None:
                    self.total += self.pending
                    self.pending = None
                self.total += x
        self.next_second += 1
        return boundary, drift


def segment(vectors, threshold=0.02, min_seconds=3):
    detector = OnlineSegmenter(threshold, min_seconds)
    ranges, scores = [], []
    for vector in vectors:
        boundary, score = detector.update(vector)
        if not ranges:
            ranges.append((0, 0))
        if boundary:
            ranges.append(boundary)
        scores.append(score)
    return ranges, scores
