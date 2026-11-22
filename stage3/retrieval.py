"""Current-frame retrieval: independent 512-d projections and causal evidence index."""
from dataclasses import dataclass
import math

import torch
from torch import nn
from torch.nn import functional as F


class RetrievalHead(nn.Module):
    def __init__(self, query_dim, visual_dim, temperature=0.1):
        super().__init__()
        if not math.isfinite(temperature) or temperature <= 0:
            raise ValueError('temperature must be positive')
        self.query_projection = nn.Linear(query_dim, 512, bias=False, dtype=torch.float32)
        self.key_projection = nn.Linear(visual_dim, 512, bias=False, dtype=torch.float32)
        self.temperature = temperature

    def query(self, hidden):
        w = self.query_projection.weight
        return F.normalize(self.query_projection(hidden.to(w)), dim=-1)

    def keys(self, pooled_frames):
        w = self.key_projection.weight
        return F.normalize(self.key_projection(pooled_frames.to(w)), dim=-1)

    def forward(self, hidden, pooled_frames):
        return self.query(hidden) @ self.keys(pooled_frames).T / self.temperature


def group_retrieval_loss(scores, groups, negatives):
    """AND over groups, OR within a group. Only explicit negatives compete.

    Other groups' positives and unannotated frames are never implicit negatives.
    Without negatives this objective has no contrastive signal (zero loss).
    """
    if scores.ndim != 1 or not groups:
        raise ValueError('Expected candidate scores and nonempty required groups')
    if not torch.isfinite(scores).all():
        raise ValueError('Non-finite retrieval scores')
    neg = set(negatives)
    positive = set().union(*(set(g) for g in groups))
    if neg & positive or any(not g for g in groups):
        raise ValueError('Empty positive group or positive/negative conflict')
    if any(not isinstance(i, int) or i < 0 or i >= len(scores) for i in neg | positive):
        raise ValueError('Candidate index outside scores')
    losses = []
    for group in groups:
        pos = sorted(set(group))
        denominator = sorted(set(pos) | neg)
        losses.append(torch.logsumexp(scores[denominator], 0) - torch.logsumexp(scores[pos], 0))
    return torch.stack(losses).mean()


@dataclass(frozen=True)
class FrameRecord:
    end: float
    source_video: str
    frame_index: int
    source_pts: float
    pooled: torch.Tensor

    @property
    def start(self):
        return max(0., self.end - 1.)


class EvidenceBank:
    """One video's frozen per-frame means; project fresh keys after every update.

    Raw visual tokens are re-encoded on demand, so persistent storage is O(T*D),
    not O(T*patches*D). No question, answer, ASR, or Q-Former history is stored.
    """
    def __init__(self, source_video):
        self.source_video = str(source_video)
        self.records = {}

    def add(self, end, visual_tokens, *, frame_index, source_pts):
        end = float(end)
        if not math.isfinite(end) or end <= 0 or not math.isfinite(source_pts):
            raise ValueError('Invalid frame time')
        if not max(0., end-1) <= source_pts < end or frame_index < 0:
            raise ValueError('Frame must fall inside its causal evidence interval')
        if visual_tokens.ndim != 2 or len(visual_tokens) == 0 or not torch.isfinite(visual_tokens).all():
            raise ValueError('Expected finite current-frame tokens [patches, hidden]')
        pooled = visual_tokens.detach().float().mean(0).cpu()
        if end in self.records:
            old = self.records[end]
            if old.frame_index != frame_index or not torch.equal(old.pooled, pooled):
                raise ValueError('Conflicting duplicate frame')
            return
        if self.records and pooled.shape != next(iter(self.records.values())).pooled.shape:
            raise ValueError('Inconsistent visual width')
        self.records[end] = FrameRecord(end, self.source_video, frame_index, float(source_pts), pooled)

    def candidates(self, current_time):
        if not math.isfinite(current_time) or current_time <= 0:
            raise ValueError('Invalid current time')
        return [r for t,r in sorted(self.records.items()) if t <= current_time]

    def get(self, end, current_time):
        if end > current_time or end not in self.records:
            raise ValueError('Evidence is unavailable at current time')
        return self.records[end]

    def search(self, head, hidden, current_time, *, top_k=4, candidate_k=16,
               max_window_seconds=4., max_frames=8):
        if not 1 <= top_k <= candidate_k or max_frames < top_k or not math.isfinite(max_window_seconds) or max_window_seconds < 1:
            raise ValueError('Invalid retrieval budget')
        records = self.candidates(current_time)
        if not records:
            return []
        with torch.no_grad():
            scores = head(hidden, torch.stack([r.pooled for r in records])).reshape(-1)
        if len(scores) != len(records) or not torch.isfinite(scores).all():
            raise ValueError('Expected finite scores for one query')
        # Stable tie order. Seeds establish diverse windows before nearby hits
        # fill remaining frame budget; no padded/unobserved frames are invented.
        ranked = sorted(range(len(records)), key=lambda i: (-float(scores[i]), records[i].end))[:candidate_k]
        windows = []
        assigned = set()
        for i in ranked:
            r = records[i]
            if any(r.start <= w['end'] and r.end >= w['start'] for w in windows):
                continue
            windows.append({'start':r.start, 'end':r.end, 'score':float(scores[i]), 'frames':[r]})
            assigned.add(i)
            if len(windows) == top_k:
                break
        for i in ranked:
            if i in assigned or sum(len(w['frames']) for w in windows) >= max_frames:
                continue
            r = records[i]
            matches = [w for w in windows if r.start <= w['end'] and r.end >= w['start']
                       and max(w['end'],r.end)-min(w['start'],r.start) <= max_window_seconds]
            if len(matches) != 1:
                continue
            w = matches[0]
            w['start'] = min(w['start'],r.start); w['end'] = max(w['end'],r.end)
            w['frames'].append(r)
        windows.sort(key=lambda w:w['start'])
        for i,w in enumerate(windows,1):
            w['evidence_id'] = f'E{i}'
            w['frames'].sort(key=lambda r:r.end)
        return windows


def evidence_metadata(windows):
    return [{k:v for k,v in w.items() if k != 'frames'} | {'frames':[
        {'source_video':r.source_video,'frame_index':r.frame_index,'source_pts':r.source_pts,'available_at':r.end}
        for r in w['frames']]} for w in windows]
