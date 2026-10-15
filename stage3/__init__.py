"""Standalone Stage 3 answerability-decision implementation (two-model scaffold).

Stage 3 is a *two-model* design (see README_zh.md "Stage 2 -> Stage 3"):

  1. A FROZEN memory model, initialised from the Stage 2 checkpoint, that keeps
     producing short/long memory vectors from the causal 1-FPS recurrence.
  2. A FINE-TUNED decision model, also initialised from the Stage 2 weights,
     with an extra <DECISION> readout position and a BCE head, that consumes the
     memory vectors and learns answerability (WAIT/ANSWER) + reason/answer text.

During training both live on the GPU at once: the frozen model runs under
``torch.no_grad()`` to emit memory, and those vectors are ``detach()``-ed
(stop-gradient) before the decision model consumes them, so the answerability
loss never flows back into the memory model.

This package currently implements the exact Stage 1 -> Stage 2 checkpoint
reconstruction, a registered/device-safe BCE head, the documented causal input
layout, and the single-sample BCE/NTP forward. Data loading, the full live
two-model training loop, answer generation and multi-question KV branching are
still open and are not yet validated end to end.
"""
