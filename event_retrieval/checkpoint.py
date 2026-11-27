"""Joint training checkpoint, written atomically within the run directory."""
import os
from pathlib import Path

import torch

from .actions import ACTIONS


def token_ids(backbone):
    return {s: backbone.tokenizer.convert_tokens_to_ids(s) for s in ACTIONS}


def save(path, towers, backbone, optimizer, epoch, config):
    if config.get("raw_recent_frames", 0) != towers.raw_recent_frames:
        raise ValueError("Checkpoint configuration must record the recent frame layout")
    language = backbone.model.model.language_model
    head = backbone.model.get_output_embeddings()
    tied = head.weight.data_ptr() == backbone.model.get_input_embeddings().weight.data_ptr()
    state = dict(version=1, config=config, epoch=epoch, actions=token_ids(backbone),
                 towers=towers.state_dict(), language=language.state_dict(),
                 head=None if tied else head.state_dict(), optimizer=optimizer.state_dict(),
                 cpu_rng=torch.get_rng_state(),
                 gpu_rng=torch.cuda.get_rng_state_all() if torch.cuda.is_available() else [])
    tmp = path.with_suffix('.tmp')
    try:
        torch.save(state, tmp)
        os.replace(tmp, path)
    finally:
        tmp.unlink(missing_ok=True)


def restore(path, towers, backbone, optimizer=None, config=None):
    state = torch.load(path, map_location='cpu', weights_only=True, mmap=True)
    if (state['version'] != 1 or state['actions'] != token_ids(backbone)
            or Path(state['config']['model']).resolve() != Path(backbone.model_path).resolve()):
        raise ValueError("Checkpoint model or action vocabulary mismatch")
    if config is not None and config != state['config']:
        raise ValueError("Training configuration changed since the checkpoint")
    towers.set_raw_recent_frames(state['config'].get('raw_recent_frames', 0))
    towers.load_state_dict(state['towers'])
    backbone.model.model.language_model.load_state_dict(state['language'])
    head = backbone.model.get_output_embeddings()
    if state['head'] is None:
        if head.weight.data_ptr() != backbone.model.get_input_embeddings().weight.data_ptr():
            raise ValueError("Checkpoint requires tied language embeddings")
    else:
        head.load_state_dict(state['head'])
    if optimizer is not None:
        optimizer.load_state_dict(state['optimizer'])
        torch.set_rng_state(state['cpu_rng'])
        if state['gpu_rng']:
            torch.cuda.set_rng_state_all(state['gpu_rng'])
    return state['epoch']
