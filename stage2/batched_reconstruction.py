"""Batch independent Stage2 targets with per-forward causal memory reuse.

Only frozen visual features may persist across optimizer updates. Historical Long
updates remain detached; current-window Short/Long computations keep gradients.
With dropout off this is equivalent to the mean of independent target losses.
Sharing a representation also shares its dropout draw within the batch.
"""
from collections import defaultdict
from contextlib import nullcontext
import random

import torch
from torch import nn
from torch.nn import functional as F
from torch.nn.utils.rnn import pad_sequence


def target_batches(tasks, per_rank, rank=0, world=1, seed=0):
    if per_rank < 1 or world < 1 or not 0 <= rank < world or len(tasks) % world:
        raise ValueError('Positive batch, valid rank and evenly divisible target count required')
    order = list(tasks)
    random.Random(seed).shuffle(order)
    return [order[i:i + per_rank * world][rank::world]
            for i in range(0, len(order), per_rank * world)]


def mean_target_ce(logits, labels):
    """Already aligned logits/labels; preserve equal weight per target, not token."""
    valid = labels.ne(-100)
    counts = valid.sum(1)
    if torch.any(counts == 0):
        raise ValueError('Each target must include answer/EOS supervision')
    # Avoid a float32 copy of vocabulary logits at unsupervised positions.
    losses = F.cross_entropy(logits[valid].float(), labels[valid], reduction='none')
    row_ids = torch.arange(labels.shape[0], device=labels.device)[:, None].expand_as(labels)[valid]
    totals = losses.new_zeros(labels.shape[0]).scatter_add(0, row_ids, losses)
    return (totals / counts).mean()


def reconstruction_batch_loss(memory, memories, tasks):
    inputs, labels, target_starts = [], [], []
    for tokens, task in zip(memories, tasks):
        prompt = f"Summarize the visible ultrasound evidence ({task['kind']}):\n"
        prefix_ids = memory.tokenizer(prompt, add_special_tokens=False, return_tensors='pt').input_ids.to(memory.device)
        answer_ids = memory.tokenizer(task['target'], add_special_tokens=False, return_tensors='pt').input_ids.to(memory.device)
        if not answer_ids.numel():
            raise ValueError('Empty reconstruction target')
        eos = answer_ids.new_tensor([[memory.tokenizer.eos_token_id]])
        answer_ids = torch.cat([answer_ids, eos], 1)
        ids = torch.cat([prefix_ids, answer_ids], 1)
        emb = torch.cat([tokens, memory.embedding(ids)], 1).squeeze(0)
        lab = ids.new_full((len(emb),), -100)
        start = tokens.shape[1] + prefix_ids.shape[1]
        lab[start:] = answer_ids[0]
        inputs.append(emb); labels.append(lab); target_starts.append(start)
    embeds = pad_sequence(inputs, batch_first=True)
    labels = pad_sequence(labels, batch_first=True, padding_value=-100)
    lengths = labels.new_tensor([len(x) for x in inputs])
    mask = torch.arange(embeds.shape[1], device=embeds.device)[None, :] < lengths[:, None]
    # Retain only positions that could predict a target; omit the final position.
    first = min(target_starts) - 1
    keep = torch.arange(first, embeds.shape[1] - 1, device=embeds.device)
    output = memory._forward(inputs_embeds=embeds, attention_mask=mask.long(),
                             logits_to_keep=keep, use_cache=False, return_dict=True)
    return mean_target_ce(output.logits, labels[:, first + 1:])


class BatchedReconstruction(nn.Module):
    def __init__(self, memory, short_chunk=128):
        super().__init__()
        self.memory = memory
        self.backbone = memory.model
        self.compressors = memory.compressors
        self.short_chunk = int(short_chunk)
        if self.short_chunk < 1:
            raise ValueError('short_chunk must be positive')
        self.last_stats = {}

    def build_memories(self, tasks, frames):
        needed, current, globals_needed = {}, set(), set()
        for task in tasks:
            vid, start, end = task['video_id'], task['start'], task['end']
            if int(start) != start or int(end) != end or end <= start or start // 60 != (end - 1) // 60:
                raise ValueError('Expected an integer interval contained in one minute')
            minute = int(start) // 60
            key = (vid, minute)
            needed[key] = max(needed.get(key, 0), int(end) - minute * 60)
            current.add(key)
            if task['kind'] == 'global':
                if start % 60 or end - start != 60:
                    raise ValueError('Global target needs a complete minute')
                globals_needed.add(key)
                for m in range(minute):
                    needed[(vid, m)] = 60
        if not tasks:
            raise ValueError('Empty target batch')
        dtype = next(self.compressors.parameters()).dtype
        bank = {}
        calls = 0
        if self.memory.requires_short_history:
            groups = defaultdict(list)
            for key, length in needed.items():
                vid, m = key
                features = frames[vid][m * 60:m * 60 + length]
                if len(features) != length or len({tuple(f.shape) for f in features}) != 1:
                    raise ValueError('Missing frames or varying patch shapes within a minute')
                groups[(key in current, length, tuple(features[0].shape))].append(key)
            for (grad, length, _), keys in groups.items():
                with nullcontext() if grad else torch.no_grad():
                    x = torch.stack([torch.stack(tuple(frames[v][m * 60:m * 60 + length])) for v, m in keys]).to(self.memory.device, dtype=dtype)
                    history, tokens = None, []
                    for t in range(length):
                        token, history = self.compressors['short'](x[:, t], history)
                        tokens.append(token); calls += 1
                    tokens = torch.cat(tokens, 1)
                    for i, key in enumerate(keys):
                        bank[key] = tokens[i:i + 1]
        else:
            # Independent LAVIS image queries: batch all unique frames in a step.
            groups, frame_bank = defaultdict(list), {}
            for (vid, m), length in needed.items():
                for t in range(m * 60, m * 60 + length):
                    groups[((vid, m) in current, tuple(frames[vid][t].shape))].append((vid, t))
            for (grad, _), keys in groups.items():
                for offset in range(0, len(keys), self.short_chunk):
                    chunk = keys[offset:offset + self.short_chunk]
                    with nullcontext() if grad else torch.no_grad():
                        features = torch.stack([frames[v][t] for v, t in chunk]).to(self.memory.device, dtype=dtype)
                        result = self.compressors['short'](features)
                        calls += 1
                    for i, key in enumerate(chunk):
                        frame_bank[key] = result[i:i + 1]
            for (vid, m), length in needed.items():
                bank[(vid, m)] = torch.cat([frame_bank[(vid, t)] for t in range(m * 60, m * 60 + length)], 1)
        # Long recurrence is chronological, but independent videos share a call.
        maxima = {}
        for vid, m in globals_needed:
            maxima[vid] = max(maxima.get(vid, -1), m)
        long_bank = {}
        long_calls = 0
        for m in range(max(maxima.values(), default=-1) + 1):
            groups = defaultdict(list)
            for vid, maximum in maxima.items():
                if m <= maximum:
                    groups[(vid, m) in globals_needed].append(vid)
            for grad, vids in groups.items():
                with nullcontext() if grad else torch.no_grad():
                    short = torch.cat([bank[(v, m)] for v in vids], 0)
                    previous = None if m == 0 else torch.cat([long_bank[(v, m - 1)] for v in vids], 0).detach()
                    result = self.memory.update_long(previous, short)
                    long_calls += 1
                for i, vid in enumerate(vids):
                    long_bank[(vid, m)] = result[i:i + 1]
        result = []
        for task in tasks:
            key = task['video_id'], task['start'] // 60
            if task['kind'] == 'global':
                result.append(long_bank[key])
            else:
                first = task['start'] % 60
                result.append(bank[key][:, first:first + task['end'] - task['start']])
        self.last_stats = dict(targets=len(tasks), unique_minutes=len(needed),
                               unique_frame_encodings=sum(needed.values()), short_calls=calls, long_calls=long_calls,
                               decoder_batch=len(tasks), cache_scope='this forward only')
        return result

    def forward(self, tasks, frames):
        return reconstruction_batch_loss(self.memory, self.build_memories(tasks, frames), tasks)
