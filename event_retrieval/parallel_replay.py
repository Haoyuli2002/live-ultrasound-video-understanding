"""Replay independent video clocks and batch pending questions across videos."""
import argparse
import bisect
import heapq
import json
import math
import sqlite3
from collections import defaultdict
from contextlib import ExitStack, closing
from pathlib import Path

import numpy as np
from PIL import Image
import torch
from torch.nn.utils.rnn import pad_sequence

from .answer import DecisionContext
from .eval_overfit import load_evaluator
from .index import EventIndex
from .replay import frame_pixels, index_source
from .retrieve import QuestionSession, QwenRelevance


class VideoClock:
    def __init__(self, tasks, duration):
        self.duration = duration
        self.schedule = [(r['query_time'], r['task_id'], r) for r in tasks]
        heapq.heapify(self.schedule)
        if any(not 0 <= t <= duration for t, _, _ in self.schedule):
            raise ValueError("Question time outside video")

    @property
    def now(self):
        return self.schedule[0][0]

    def due(self):
        now = self.now
        rows = []
        while self.schedule and self.now == now:
            rows.append(heapq.heappop(self.schedule)[2])
        return rows

    def after(self, row, now, action):
        if action == 'WAIT' and now < self.duration:
            tick = min(math.floor(now) + 1, self.duration)
            heapq.heappush(self.schedule, (tick, row['task_id'], row))


class Video:
    def __init__(self, tasks, backbone_path, seed_dir, output_dir, stack):
        from decord import VideoReader, cpu
        first = tasks[0]
        path = Path(first['video_path']).resolve(strict=True)
        self.reader = VideoReader(str(path), ctx=cpu(0), num_threads=1)
        stamps = self.reader.get_frame_timestamp(np.arange(len(self.reader)))
        self.pts = stamps[:, 0].tolist()
        self.clock = VideoClock(tasks, float(stamps[-1, 1]))
        self.vid = first['video_id']
        self.source = index_source(path, backbone_path)
        self.dest = output_dir / (self.vid + '.sqlite')
        seed = seed_dir / self.dest.name
        self.seeded = seed.is_file() and not self.dest.exists()
        self.stack = stack
        self.index = stack.enter_context(closing(EventIndex(
            seed if self.seeded else self.dest, self.vid, self.source, read_only=self.seeded)))
        self.sessions = {}

    def writable(self):
        if self.seeded:
            with closing(sqlite3.connect(self.dest)) as dest:
                self.index.db.backup(dest)
            self.index = self.stack.enter_context(closing(EventIndex(self.dest, self.vid, self.source)))
            for session in self.sessions.values():
                session.index = self.index
            self.seeded = False
        return self.index


def advance(videos, backbone):
    """Advance each clock only up to its next decision, batching missing frames."""
    while True:
        pending = []
        pixels = []
        for video in videos:
            duration = video.clock.duration
            second = video.index.detector.next_second
            last = min(math.floor(video.clock.now), math.ceil(duration) - 1)
            if second <= last:
                i = bisect.bisect_right(video.pts, second) - 1
                if i < 0:
                    raise ValueError(f"No observed frame at {second}s for {video.vid}")
                rgb = video.reader[i]
                rgb = rgb.asnumpy() if hasattr(rgb, 'asnumpy') else np.asarray(rgb)
                pixels.append(frame_pixels(Image.fromarray(rgb)))
                pending.append((video, second, video.pts[i]))
            elif video.clock.now == duration and video.index.eof is None:
                video.writable().finish(duration)
        if not pending:
            return
        features = backbone.encode_frames(torch.stack(pixels).unsqueeze(2))[:, 0].cpu()
        for (video, second, pts), frame in zip(pending, features):
            video.writable().push(second, pts, frame)


@torch.inference_mode()
def replay_group(videos, backbone, towers, decoder, scorer, device, bf16, batch_size,
                 retrieval_size, budget, emit):
    while any(v.clock.schedule for v in videos):
        active = [v for v in videos if v.clock.schedule]
        advance(active, backbone)
        ticks = [v.clock.now for v in active]
        windows, bounds = [], []
        for v, t in zip(active, ticks):
            end = min(math.floor(t), v.index.detector.next_second - 1)
            start = max(0, end + 1 - towers.recent_window)
            bounds.append((start, end))
            windows.append(torch.stack([v.index.frame(i)[1] for i in range(start, end + 1)]))
        frames = pad_sequence(windows, batch_first=True).to(device)
        lengths = torch.tensor([len(w) for w in windows], device=device)
        valid = torch.arange(frames.shape[1], device=device)[None] < lengths[:, None]
        with torch.autocast(device.type, torch.bfloat16, enabled=bf16):
            short, short_mask = towers.encode_short(frames, valid)
        pending = []
        for i, (v, now) in enumerate(zip(active, ticks)):
            for row in v.clock.due():
                key = row['task_id']
                if key not in v.sessions:
                    v.sessions[key] = QuestionSession(v.index, row['question'], scorer, towers,
                                                      retrieval_batch_size=retrieval_size)
                observed = now if now == v.index.eof else math.floor(now)
                history = v.sessions[key].at(observed)
                ctx = DecisionContext(row['question'], row['query_time'], now, *bounds[i], tuple(history.sources))
                item = (torch.cat((history.prefix[0], short[i])), torch.cat((history.mask[0], short_mask[i])), ctx)
                pending.append((v, row, now, item))
        for start in range(0, len(pending), batch_size):
            batch = pending[start:start + batch_size]
            with torch.autocast(device.type, torch.bfloat16, enabled=bf16):
                preds = decoder.generate_batch([item for _, _, _, item in batch], budget)
            for (v, row, now, (_, _, ctx)), pred in zip(batch, preds, strict=True):
                emit(dict(video_id=v.vid, task_id=row['task_id'], question=row['question'],
                          query_time=row['query_time'], current_time=now, **pred,
                          recent_window_s=[ctx.recent_start, ctx.recent_end], retrieved_events=list(ctx.events)))
                v.clock.after(row, now, pred['action'])
                if pred['action'] == 'ANSWER' or now == v.clock.duration:
                    del v.sessions[row['task_id']]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run', required=True, type=Path, help='Joint overfit checkpoint directory')
    parser.add_argument('--output', required=True, type=Path, help='New scratch output directory')
    parser.add_argument('--videos', type=int, default=10, help='Number of concurrent video streams')
    parser.add_argument('--generation-batch-size', type=int, default=10)
    parser.add_argument('--retrieval-batch-size', type=int, default=64)
    parser.add_argument('--max-new-tokens', type=int, default=96)
    parser.add_argument('--device', default='cuda')
    parser.add_argument("--holdout", action="store_true")
    args = parser.parse_args()
    if min(args.videos, args.generation_batch_size, args.retrieval_batch_size, args.max_new_tokens) < 1:
        parser.error('Batch sizes and token budget must be positive')
    run = args.run.resolve(strict=True)
    out = args.output.resolve()
    out.mkdir(parents=True, exist_ok=False)
    cache = out / 'index'
    cache.mkdir()
    device = torch.device(args.device)
    cfg, examples, path, bb, towers, decoder, epoch = load_evaluator(run, device, holdout=args.holdout)
    scorer = QwenRelevance(bb)
    by_video = defaultdict(list)
    seen = set()
    for row in examples:
        if row['task_id'] not in seen:
            by_video[row['video_id']].append(row)
            seen.add(row['task_id'])
    groups = list(by_video.values())
    stats = dict(epoch=epoch, questions=len(seen), videos=len(groups), decisions=0, answered=0)
    with (out / 'stream_predictions.jsonl').open('x', buffering=1) as stream:
        def emit(record):
            stream.write(json.dumps(record) + '\n')
            stats['decisions'] += 1
            stats['answered'] += int(record['action'] == 'ANSWER')
        for i in range(0, len(groups), args.videos):
            with ExitStack() as stack:
                videos = [Video(tasks, path, run / 'index', cache, stack)
                          for tasks in groups[i:i + args.videos]]
                replay_group(videos, bb, towers, decoder, scorer, device, cfg.get('bf16', False),
                             args.generation_batch_size, args.retrieval_batch_size, args.max_new_tokens, emit)
    (out / 'summary.json').write_text(json.dumps(stats, indent=2) + '\n')


if __name__ == '__main__':
    main()
