"""Architecture-neutral chronological Short encoding and cache boundaries.

This module also works with frozen legacy model implementations during
reconstruction; it must not import current architecture classes.
"""
from contextlib import contextmanager
import torch


def short_window(memory, images):
    encoder = getattr(memory, "encode_short", None)
    return encoder(images) if encoder else torch.stack([memory.encode_frame(i) for i in images],dim=1)


@contextmanager
def short_memory_scope(memory):
    """Isolate a minute or a training replay; never retain a stale autograd graph."""
    reset = getattr(memory, "reset_short_memory", None)
    if reset:
        reset()
    try:
        yield
    finally:
        if reset:
            reset()


def iter_short_windows(memory, row, reader):
    """Chronological evaluation: carry Short history across windows, not minutes."""
    with short_memory_scope(memory):
        for j, label in enumerate(row["local_sub_summaries"]):
            first = int(round(row["block_window"][0])) + 10*j
            images = [reader.frame_ending_at(t) for t in range(first, first+10)]
            yield first, short_window(memory, images), label
