"""Prepare immutable per-video SQLite indexes shared by QA runs on scratch."""
from contextlib import ExitStack, closing, contextmanager
import bisect
import fcntl
import hashlib
import os
from pathlib import Path
import sqlite3

import numpy as np
from PIL import Image
import torch

from .index import EventIndex, index_identity
from .replay import frame_pixels, index_source


@contextmanager
def _prepare_video(vid, video, cache, shared, model_path, seed_dir):
    source = index_source(video, model_path)
    key = hashlib.sha256(index_identity(vid, source).encode()).hexdigest()
    final = shared / (key + '.sqlite')
    partial = final.with_suffix('.partial')
    local = cache / (vid + '.sqlite')
    with final.with_suffix('.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        if final.exists():
            yield None
        else:
            seed = seed_dir / local.name if seed_dir else local
            if not partial.exists() and seed.is_file():
                with closing(EventIndex(seed, vid, source, read_only=True)) as src:
                    with closing(sqlite3.connect(partial)) as dest:
                        src.db.backup(dest)
            with closing(EventIndex(partial, vid, source)) as index:
                yield index
                if index.eof is None:
                    raise ValueError(f"Incomplete index for {vid}")
            os.replace(partial, final)
        with closing(EventIndex(final, vid, source, read_only=True)) as index:
            if index.eof is None:
                raise ValueError(f"Incomplete shared index for {vid}")
        if local.resolve() != final.resolve():
            link = local.with_suffix('.link.tmp')
            link.unlink(missing_ok=True)
            link.symlink_to(final.resolve())
            os.replace(link, local)


@torch.inference_mode()
def prepare_indexes(rows, cache, shared, model_path, backbone, *, seed_dir=None, parallel_videos=10):
    from decord import VideoReader, cpu

    if parallel_videos < 1:
        raise ValueError("Expected a positive number of parallel videos")
    shared.mkdir(parents=True, exist_ok=True)
    videos = {}
    for row in rows:
        video = Path(row['video_path']).resolve(strict=True)
        if videos.setdefault(row['video_id'], video) != video:
            raise ValueError(f"Conflicting paths for video {row['video_id']}")
    # Acquire locks in a consistent order across concurrent training runs.
    ordered = sorted(videos.items())
    for offset in range(0, len(ordered), parallel_videos):
        with ExitStack() as stack:
            streams = []
            for vid, video in ordered[offset:offset + parallel_videos]:
                index = stack.enter_context(_prepare_video(vid, video, cache, shared, model_path, seed_dir))
                if index is not None and index.eof is None:
                    reader = VideoReader(str(video), ctx=cpu(0), num_threads=1)
                    stamps = reader.get_frame_timestamp(np.arange(len(reader)))
                    streams.append((index, reader, stamps[:, 0].tolist(), float(stamps[-1, 1])))
            while streams:
                remaining, pixels, frames = [], [], []
                for stream in streams:
                    index, reader, pts, duration = stream
                    second = index.detector.next_second
                    if second >= duration:
                        index.finish(duration)
                        continue
                    i = bisect.bisect_right(pts, second) - 1
                    if i < 0:
                        raise ValueError(f"No observable frame at {second}s")
                    rgb = reader[i]
                    image = rgb.asnumpy() if hasattr(rgb, 'asnumpy') else np.asarray(rgb)
                    pixels.append(frame_pixels(Image.fromarray(image)))
                    frames.append((index, second, pts[i]))
                    remaining.append(stream)
                if pixels:
                    features = backbone.encode_frames(torch.stack(pixels).unsqueeze(2))[:, 0].cpu()
                    for (index, second, pts_s), feature in zip(frames, features, strict=True):
                        index.push(second, pts_s, feature)
                streams = remaining
