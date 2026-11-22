"""Joint QA/readiness optimization; no staged freezes or predicted-label gates."""
import random



def task_batches(examples, batch_size, seed):
    groups = {}
    for row in examples:
        groups.setdefault(row['task_id'], []).append(row)
    packets = list(groups.values())
    if any(len(g) > batch_size for g in packets):
        raise ValueError('Batch size must fit every question\'s labeled endpoints; increase batch size')
    random.Random(seed).shuffle(packets)
    batches, current = [], []
    for packet in packets:
        if len(current) + len(packet) > batch_size:
            batches.append(current)
            current = []
        current.extend(sorted(packet, key=lambda row: row['current_time']))
    if current:
        batches.append(current)
    return batches


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
