"""Joint QA/readiness optimization; no staged freezes or predicted-label gates."""
import random



def task_batches(examples, batch_size, seed, video_block_size=0):
    """Shuffle whole QA groups, optionally keeping nearby same-video groups adjacent."""
    if video_block_size < 0:
        raise ValueError("video_block_size must be nonnegative")
    groups = {}
    for row in examples:
        groups.setdefault(row['task_id'], []).append(row)
    packets = list(groups.values())
    if any(len(g) > batch_size for g in packets):
        raise ValueError('Batch size must fit every question\'s labeled endpoints; increase batch size')
    rng = random.Random(seed)
    rng.shuffle(packets)
    if video_block_size:
        by_video = {}
        for packet in packets:
            by_video.setdefault(packet[0]['video_id'], []).append(packet)
        blocks = []
        limit = min(video_block_size, batch_size)
        for video_packets in by_video.values():
            video_packets.sort(key=lambda packet: max(row['current_time'] for row in packet))
            block, size = [], 0
            for packet in video_packets:
                if block and size + len(packet) > limit:
                    blocks.append(block)
                    block, size = [], 0
                block.append(packet)
                size += len(packet)
            blocks.append(block)
        rng.shuffle(blocks)
        packets = [packet for block in blocks for packet in block]
    batches, current = [], []
    for packet in packets:
        if len(current) + len(packet) > batch_size:
            batches.append(current)
            current = []
        current.extend(sorted(packet, key=lambda row: row['current_time']))
    if current:
        batches.append(current)
    return batches


def replay_updates(rows):
    """Count one integer prefix plus distinct fractional endpoints per video."""
    last_seconds = {}
    fractional = set()
    for row in rows:
        video, end = row['video_id'], row['current_time']
        last_seconds[video] = max(last_seconds.get(video, 0), int(end))
        if end != int(end):
            fractional.add((video, end))
    return sum(second + 1 for second in last_seconds.values()) + len(fractional)


def distribute_batches(batches, world_size, balance_replay, seed):
    """Match costs within synchronized steps and shuffle the resulting steps."""
    if balance_replay:
        batches = sorted(batches, key=replay_updates, reverse=True)
    steps = []
    rng = random.Random(seed)
    for start in range(0, len(batches), world_size):
        ranks = batches[start:start + world_size]
        if balance_replay:
            rng.shuffle(ranks)
        steps.append(ranks + [[] for _ in range(world_size - len(ranks))])
    if balance_replay:
        rng.shuffle(steps)
    return steps


def backward_objectives(model, result, strategy='readiness_isolated'):
    if strategy == 'sum':
        result['loss'].backward()
        return {'strategy': strategy}
    if strategy == 'readiness_isolated':
        # Paper §3.4: timing updates only RDY + its head. The QA loss still
        # backpropagates through the frozen decoder into the reasoning path.
        # Like the existing StreamReady surrogate, this restricts the timing
        # gradient; it does not invent a reverse restriction on QA -> RDY.
        timing = model._readiness_surrogate(model.readiness_loss_weight * result['readiness_loss'])
        (result['answer_loss'] + timing).backward()
        return {'strategy': strategy, 'timing_updates': 'readiness_head_and_rdy_token_only'}
    raise ValueError(f'Unknown gradient strategy: {strategy}')
