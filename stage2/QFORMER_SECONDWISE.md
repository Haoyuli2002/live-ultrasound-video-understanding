# Secondwise Short Q-Former (`qformer_secondwise_v2`)

[Editable draw.io architecture](architecture/qformer_secondwise.drawio) ·
[Vector figure](architecture/qformer_secondwise.svg)

One learned Short query (512-d by default) is reused every second. At the
20th second it reads the preceding 19 **internal-width** Short states and
itself, then the current frame's visual patches. Two layers each perform:

1. Update **only the current query** by attention over `[old states; query]`.
   This computes the current row of causal self-attention; historical states
   are read-only, not rewritten or re-encoded by this step.
2. Cross-attend to **only the current frame's** merged visual patches.
3. Apply the feed-forward block, with pre-normalization and residuals.

Frame patches are projected D→512 and receive the current minute position.
The learned query receives the same position. The last-layer state `h_t`
is appended to history without an output projection. A separate
LayerNorm/Linear maps it 512→D as `s_t` for existing downstream interfaces.
There is no D→512 Short-history feedback and no GT/ASR/summary input.

| State | Shape (batch omitted) | Lifetime |
|---|---|---|
| Short learned query | 1 × 512 | Model parameter, reused every second |
| Short history H | t × 512 | Append-only, up to 60 seconds; reset per minute/video |
| Short output s_t | 1 × D | Current second; collected into 10/60-token prefixes |
| Long learned queries | 60 × 512 | Separate learned parameters |
| Long memory L | 60 × D | Updated each complete minute; detached across minutes |

Long is unchanged: its 60 queries self-attend and cross-attend to
`[old Long.detach(); current-minute 60 Short outputs]` (only current Short
in the first minute), then project 512→D. Short and Long share neither
attention weights nor learned queries. Summary decoding uses the same
Qwen3.5-2B language backbone with Stage2 LoRA.

## Streaming interface

```python
memory.reset_short_memory()  # start of video/minute
for image in minute_frames:  # chronological, one frame per second
    s_t = memory.encode_frame(image)  # [B,D], exactly one new cached state
    # At each 10 seconds decode Local from the latest 10 s_t.
# Decode Block from all 60 s_t; update Long, then decode Global.
memory.reset_short_memory()  # do not carry H into the next minute
```

`encode_short(images)` remains a chunk convenience for 1..10 frames. It runs
those frames **sequentially**, carries the same history, and returns one
D-d token per frame. Splitting into chunks does not alter temporal semantics.
Calling beyond 60 cached states raises instead of silently mixing minutes.
Frozen vision may batch frames; each query step consumes only its frame's
features. The default visual input remains 1 FPS and 224-pixel padded frames.

Local reads the latest 10 D-d tokens, but those tokens can contain **earlier
current-minute context**. Do not describe them as using only the isolated
10-second video interval. There is no future-frame input; output quality and
repetition of past content still need GPU autoregressive assessment.

## Training, evaluation and checkpoint identity

- Each Local/Block/Global is still a separate token-CE target. Short memory
  writes happen per video second, not one optimizer step per second.
- Before each target, `encode_interval` clears H, replays from the current
  minute boundary to the target endpoint at current weights, and returns
  only the target interval's output tokens. Gradients flow through that
  entire minute prefix. A `finally` block discards H so optimizer steps
  cannot reuse a stale graph. This replays more frames for later Locals.
- Free-running reconstruction carries H across the six Local windows in
  chronological order, resets it at each minute/video, and generates text
  without reference tokens. The teacher-forced loss CLI is diagnostic only.
- Config records `short_queries=1`, `short_update_seconds=1`,
  `short_cache_width=512`, and architecture `qformer_secondwise_v2`.
- Old `qformer_style_v1` checkpoints still load with the original 10-query
  window encoder for reproducible evaluation. They do not become secondwise
  models, and old full-training runs cannot resume as the new architecture.
- Run a **new** P3 GPU smoke before new full training. The gate requires this
  architecture/query count and matching source hashes, including
  `short_encoding.py`. Old smoke reports do not qualify. No existing Slurm
  job or source snapshot is modified by this local change.

CPU validation uses real attention layers at reduced width to check causal
prefix invariance, one-frame/chunk equivalence, history immutability, cache
projection separation, gradients through history/current visual inputs,
minute resets, independent optimizer steps, checkpoint roundtrips, legacy
loading, and two-rank Gloo synchronization with padded ranks. These checks
are functional and do not establish medical correctness or GPU throughput.
