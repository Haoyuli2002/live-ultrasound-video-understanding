"""Chronological Stage-2 schedule; model operations are injected for testing.

One optimizer step per ten seconds. Previous short/long states are detached,
never recomputed. At minute boundaries short + local + global losses are summed.
"""
from __future__ import annotations


def prepare_blocks(rows):
    """Validate all labels before training; never silently skip missing history."""
    blocks = sorted(rows, key=lambda r: (str(r.get('video_id', '')), float(r['block_window'][0])))
    expected = {}
    ended = set()
    for row in blocks:
        vid = row.get('video_id')
        start, end = map(float, row['block_window'])
        if not vid or vid in ended or row.get('error') or start != expected.get(vid, 0.0):
            raise ValueError(f'Missing/duplicate/discontinuous video history: {vid} at {start}')
        duration = end - start
        if duration <= 0 or duration > 60 or duration % 10:
            raise ValueError('Blocks must contain 1–6 complete 10-second windows')
        if duration < 60:
            ended.add(vid)
        subs = sorted(row.get('local_sub_summaries', []), key=lambda x: x['window'][0])
        if len(subs) != int(duration / 10):
            raise ValueError(f'{vid} at {start}: matching 10-second labels required; regenerate dense labels')
        for i, sub in enumerate(subs):
            if list(map(float, sub['window'])) != [start + 10*i, start + 10*(i+1)] or not str(sub.get('local_summary_target', '')).strip():
                raise ValueError(f'{vid}: invalid or missing 10-second target')
        if duration == 60 and (not str(row.get('block_summary_target', '')).strip() or not str(row.get('global_summary_target', '')).strip()):
            raise ValueError(f'{vid}: explicit block_summary_target and global_summary_target required')
        row = dict(row, local_sub_summaries=subs)
        expected[vid] = end
        yield row


def train_stream(blocks, *, encode_second, decode_short, encode_long, decode_global,
                 optimizer, backward, clip_grad, report):
    current_video = None
    long_states = []
    steps = 0
    for block in blocks:
        if block['video_id'] != current_video:
            current_video = block['video_id']
            long_states = []
        short_states = []
        start = float(block['block_window'][0])
        for j, sub in enumerate(block['local_sub_summaries']):
            optimizer.zero_grad(set_to_none=True)
            fresh = [encode_second(block, start + j*10 + i) for i in range(10)]
            short_states.extend(fresh)
            losses = {'short': decode_short(fresh, sub['local_summary_target'], 'local_summary')}
            if j == 5:
                losses['local'] = decode_short(short_states, block['block_summary_target'], 'block_summary')
                new_long = encode_long(short_states, long_states)
                losses['global'] = decode_global(new_long, block['global_summary_target'])
            total = sum(losses.values())  # weights 1:1:1; no averaging of the three objectives
            backward(total)
            clip_grad()
            optimizer.step()
            short_states = [state.detach().clone() for state in short_states]
            if j == 5:
                long_states = [state.detach().clone() for state in new_long]
            steps += 1
            report(current_video, start + (j+1)*10, losses, total)
            # Release graph-bearing references between steps.
            del fresh, losses, total
            if j == 5:
                del new_long
    return steps
