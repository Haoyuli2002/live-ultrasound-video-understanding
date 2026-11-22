"""Batched endpoint reasoning with causal, per-forward shared video replay.

Only integer-grid history is advanced persistently. Fractional endpoints fork
that state so one question's extra frame cannot enter another question's history.
All projected features and memory snapshots are discarded after this forward;
no stale projected memory crosses an optimizer update.
"""
import copy

import torch
from torch.nn.utils.rnn import pad_sequence

from .data import prompt, sample_times
from lavis.models.streamready.fusion import ShortLayerOutputCapture, set_long_term_fusion_queries
from lavis.models.streamready.retrieval import coarse_to_fine_retrieve


STATE = ('frames', 'frame_count', 'centroids', 'centroid_mask', 'cent_proto',
         'prototypes', 'proto_mask', '_sim_ema', '_new_cluster_rate', '_seeded')


def clone_bank(bank):
    fork = copy.copy(bank)
    for name in STATE:
        value = getattr(bank, name)
        setattr(fork, name, None if value is None else value.clone())
    return fork


def padded(values):
    lengths = [x.shape[0] for x in values]
    batch = pad_sequence(values, batch_first=True)
    mask = torch.arange(batch.shape[1], device=batch.device)[None] < batch.new_tensor(lengths, dtype=torch.long)[:, None]
    return batch, mask


def forward_endpoints(core, examples, features):
    if core.context is not None:
        raise ValueError('Independent endpoint batching requires contextual memory disabled')
    groups = {}
    for index, (row, feat) in enumerate(zip(examples, features)):
        times = sample_times(row['current_time'])
        if feat.shape[0] != 1 or len(times) != feat.shape[1]:
            raise ValueError('Endpoint features must match the causal 1fps+endpoint grid')
        groups.setdefault(row['video_id'], []).append((index, row, feat, times))
    device = core.short_query_tokens.device
    windows, banks = [None] * len(examples), [None] * len(examples)
    updates = 0
    for group in groups.values():
        frames = {}
        for _, _, feat, times in group:
            for j, t in enumerate(times):
                frames.setdefault(t, feat[0, j])
        times = sorted(frames)
        lookup = {t: i for i, t in enumerate(times)}
        # One transfer/projection per unique video, retaining gradients for all
        # endpoint windows. Fractional frames are never put in the shared grid.
        projected = torch.stack([frames[t] for t in times]).to(device=device, dtype=core._param_dtype())
        if core.vision_proj is not None:
            projected = core.vision_proj(projected)
        bank = copy.copy(core.memory)
        bank.reset()
        last_integer = -1
        snapshots = {}
        for index, row, _, grid in sorted(group, key=lambda x: x[1]['current_time']):
            endpoint = row['current_time']
            if endpoint not in snapshots:
                for t in range(last_integer + 1, int(endpoint) + 1):
                    bank.update(projected[lookup[float(t)]].unsqueeze(0))
                    updates += 1
                last_integer = int(endpoint)
                snapshot = clone_bank(bank)
                if endpoint != int(endpoint):
                    snapshot.update(projected[lookup[endpoint]].unsqueeze(0))
                    updates += 1
                snapshots[endpoint] = snapshot
            banks[index] = snapshots[endpoint]
            selected = [lookup[t] for t in grid[-core.memory.frame_capacity:]]
            windows[index] = projected[selected].flatten(0, 1)

    short_mem, short_mask = padded(windows)
    b = len(examples)
    q_toks = core.tokenize_question([prompt(e) for e in examples], device) if core.qformer_text_input else None
    q_vec = core._question_vector(q_toks, b, device)
    q_proto, q_cent = core.query_proj(q_vec)
    if core.long_term_fusion_mode == 'per_layer_matched':
        with ShortLayerOutputCapture(core.short_qformer.bert, core.num_query_token) as capture:
            z_s = core._run_branch(core.short_qformer, core.short_query_tokens, short_mem, short_mask, q_toks)
        short_for_long = dict(capture.outputs)
    else:
        z_s = core._run_branch(core.short_qformer, core.short_query_tokens, short_mem, short_mask, q_toks)
        short_for_long = z_s

    # Whether history exists follows the known frame count; no GPU scalar read.
    historical = [i for i, e in enumerate(examples) if len(sample_times(e['current_time'])) > core.memory.frame_capacity]
    long_values, long_masks = list(windows), [torch.ones(x.shape[0], device=device, dtype=torch.bool) for x in windows]
    if historical:
        selected_banks = [banks[i] for i in historical]
        slots, masks = coarse_to_fine_retrieve(
            q_proto[historical], q_cent[historical],
            *[torch.cat([getattr(bank, name) for bank in selected_banks]) for name in
              ('prototypes', 'proto_mask', 'centroids', 'centroid_mask', 'cent_proto')],
            core.topk_prototypes, core.topk_centroids,
        )
        for j, index in enumerate(historical):
            long_values[index], long_masks[index] = slots[j], masks[j]
    long_mem, _ = padded(long_values)
    long_mask = pad_sequence(long_masks, batch_first=True)
    extra = core.rdy_token.expand(b, -1, -1).to(z_s.dtype) if core.use_readiness else None
    if core.post_fusion is None:
        set_long_term_fusion_queries(core.long_qformer.bert, short_for_long)
    try:
        out = core._run_branch(core.long_qformer, core.long_query_tokens, long_mem, long_mask, q_toks, extra_query=extra)
    finally:
        if core.post_fusion is None:
            set_long_term_fusion_queries(core.long_qformer.bert, None)
    z_l = out[:, :core.num_query_token]
    readiness = core.readiness_head(out[:, core.num_query_token]) if core.use_readiness else None
    if core.post_fusion is not None:
        z_l = core.post_fusion(z_l, z_s)
    prefix = core.output_proj(z_l)
    core.last_replay_stats = dict(videos=len(groups), endpoints=b, memory_updates=updates,
                                 independent_updates=sum(len(sample_times(e['current_time'])) for e in examples),
                                 qformer_batch_size=b)
    return [dict(prefix_embeds=prefix[i:i+1], readiness=readiness[i:i+1, None],
                 z_l=z_l[i:i+1]) for i in range(b)]
