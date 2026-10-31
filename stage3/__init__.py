"""Stage 3 answerability-decision models and causal single-GPU QA training.

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

The training CLI replays video causally at decision times and optimizes BCE/NTP.
GPU integration, optimizer-state resume, distributed Stage 3 training, answer
generation and multi-question KV branching still need separate work/validation.
"""
