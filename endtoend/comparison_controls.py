"""Auditable identical initialization and zero-update AdamW allocation for comparisons."""
import hashlib
import json
from pathlib import Path
import torch


def state_fingerprint(module):
    digest = hashlib.sha256()
    tensors = module.state_dict()
    for name in sorted(tensors):
        value = tensors[name].detach().cpu().contiguous()
        digest.update(json.dumps([name, str(value.dtype), list(value.shape)]).encode())
        digest.update(value.reshape(-1).view(torch.uint8).numpy().tobytes())
    return dict(sha256=digest.hexdigest(), tensors=len(tensors),
                elements=sum(v.numel() for v in tensors.values()))


def verify_initial_state(core, optimizer, output, shared_manifest):
    if optimizer.state:
        raise ValueError('Comparison must start with a fresh optimizer')
    fingerprint = state_fingerprint(core)
    shared = Path(shared_manifest)
    if shared.exists():
        expected = json.loads(shared.read_text())
        if fingerprint != expected:
            raise ValueError('Comparison arms have different initial core weights')
    else:
        with shared.open('x') as stream:
            stream.write(json.dumps(fingerprint, indent=2) + '\n')
    Path(output).write_text(json.dumps(dict(core=fingerprint, optimizer_state_entries=0,
        initial_optimizer_updates=0, matched_shared_manifest=str(shared)), indent=2) + '\n')
    return fingerprint


def preallocate_adamw_state(optimizer):
    """Allocate zero moments with step=0, without optimizer.step or parameter changes."""
    if not isinstance(optimizer, torch.optim.AdamW):
        raise TypeError('Only AdamW is supported')
    for group in optimizer.param_groups:
        for parameter in group['params']:
            if optimizer.state[parameter]:
                raise ValueError('Expected a fresh optimizer')
            state = optimizer.state[parameter]
            device = parameter.device if group.get('capturable') or group.get('fused') else torch.device('cpu')
            state['step'] = torch.zeros((), dtype=torch.float32, device=device)
            state['exp_avg'] = torch.zeros_like(parameter, memory_format=torch.preserve_format)
            state['exp_avg_sq'] = torch.zeros_like(parameter, memory_format=torch.preserve_format)
            if group.get('amsgrad'):
                state['max_exp_avg_sq'] = torch.zeros_like(parameter, memory_format=torch.preserve_format)
