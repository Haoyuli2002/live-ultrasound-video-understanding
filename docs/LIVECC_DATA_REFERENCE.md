# LiveCC — Data & Training Reference

> **Scope.** This document describes **how LiveCC organises its training data** —
> what datasets exist, what each row inside the jsonl looks like, what filters
> the data goes through, how the dataloader turns one row into a training
> sample, what hyper-parameters are used, and how the loss is computed.
>
> **Sources.** Every claim is followed by a clickable link to the file/line in
> [`livecc/`](../livecc/) (this repo's vendored copy of the official `showlab/livecc`)
> or to the paper PDF [`livcecc.pdf`](../livcecc.pdf). When the paper /
> README do not specify a number, this is stated explicitly with `not specified`.
>
> Anything **outside** Sections 1–6 (e.g. ablation tables, design rationales,
> our own ultrasound mapping) is intentionally **excluded** from this reference.

---

## Section 1 — Datasets at a glance

LiveCC's training pipeline involves **three datasets** (one for pretraining,
one self-built SFT dataset, and one external SFT dataset):

| Stage | Dataset | Type | Built by | Quoted size |
|-------|---------|------|----------|-------------|
| Pretrain | **Live-CC-5M** | streaming video + ASR (word-level) | LiveCC team | 5 M clips |
| SFT | **Live-WhisperX-526K** | streaming video + ASR (word-level) | LiveCC team | 526 K clips |
| SFT | **LLaVA-Video-178K** | offline video QA / caption | external (lmms-lab) | 178 K videos |

Pretrain stage uses **only** Live-CC-5M ([`livecc/README.md` L67-68](../livecc/README.md#L67-L68)).

For SFT, the **paper text** ([`livcecc.pdf` L875-878](../livcecc.pdf#L875-L878)):

> "The SFT stages uses our **Live-Whisper-526K** and **LLaVA-Video-178K** \[105\]
> (without the training set of ActivityNetQA, Next-QA, and PerceptionTest)."

The **release `sft_local.sh`** ([`livecc/scripts/sft_local.sh` L29-34](../livecc/scripts/sft_local.sh#L29-L34))
adds three more files (LLaVA-Hound, LLaVA-OneVision single-image,
LLaVA-OneVision multi-image). These extra files are **not mentioned in the
paper text** as part of the model that produced the reported numbers — treat
them as a "released training recipe", not as the paper SOTA recipe.

> ⚠️ **Sample-mixing ratios** between datasets are **not given** in the paper.
> The dataloader concatenates jsonl files line-by-line
> ([`lmm_dataset.py` L55-60](../livecc/data/lmm_dataset.py#L55-L60)) without any
> per-file reweighting, so the de-facto ratio equals the row counts of each
> jsonl on disk.

---

## Section 2 — Row schema of the jsonl files

### 2.1 Source-stage jsonl (post-clipping, pre-conversation)

After the pre-training clipping step
([`pretrain_to_clips.py` L49](../livecc/data/production/pretrain_to_clips.py#L49))
each row is a single dict with these fields:

| Field | Type | Meaning |
|-------|------|---------|
| `video` | `str` | path or URL of the source mp4 |
| `content` | `list[[float, float, str]]` | word-level ASR: `[start_sec, end_sec, word]` per word |
| `previous` | `str` | text of all ASR words **before this clip** in the same video (used as context during pretrain) |
| `title` | `str` | YouTube video title |
| `category` | `str` | YouTube category (e.g. `Sports`, `Howto`) |

For SFT clips, the same fields appear, but `previous` is replaced with
`preasr` ([`sft_to_clips.py` L26](../livecc/data/production/sft_to_clips.py#L26)):

> ```python
> return [{'video': datum['video'], 'content': clip, 'preasr': preasr,
>          'title': title, 'category': datum['category']} for ...]
> ```

After **B6 prompt synthesis** ([`make_prompt.py` L54-58](../livecc/data/production/make_prompt.py#L54-L58)),
SFT rows additionally get a `query` field — a GPT-4o-suggested user prompt
that does not leak the actual ASR.

### 2.2 Conversation-stage jsonl (training-ready)

[`to_conversation.py` L7-15](../livecc/data/production/to_conversation.py#L7-L15)
maps each SFT row to a **two-turn Qwen-style conversation**:

```python
[
  {'role': 'user', 'content': [
      {'type': 'video', 'video': datum['video'],
       'video_start': datum['content'][0][0],
       'video_end':   datum['content'][-1][1]},
      {'type': 'text', 'text': datum['query'],
       'previous': datum['preasr'],
       'title':    datum['title'],
       'category': datum['category']},
  ]},
  {'role': 'assistant', 'content': [
      {'type': 'text_stream', 'text_stream': datum['content']}
  ]},
]
```

Key points:
- The user side has **one video block + one text block**. The text block
  carries the synthesised `query` and stashes title/previous-ASR/category
  inside the same dict for later retrieval.
- The assistant side carries the **whole word-level ASR list** under the
  special key `text_stream` — this key is what later signals the dataloader
  to take the streaming-preprocessing path
  ([`lmm_dataset.py` L168-172](../livecc/data/lmm_dataset.py#L168-L172)).
- After serialising the conversations to a jsonl, the script appends one
  extra last line containing **byte-seek offsets** for every row
  ([`to_conversation.py` L17-24](../livecc/data/production/to_conversation.py#L17-L24)).
  The dataloader reads this last line via `readlastline` and uses
  `seek(byte_offset)` to load any row in O(1)
  ([`lmm_dataset.py` L23-28, L57-58](../livecc/data/lmm_dataset.py#L23-L60)).

### 2.3 A concrete row (illustration only)

```jsonc
// Live-WhisperX-526K, post-conversation, single line:
[
  {"role": "user", "content": [
      {"type": "video", "video": "yt_xxx.mp4",
       "video_start": 30.0, "video_end": 90.0},
      {"type": "text",  "text": "Please commentate this clip in real time.",
       "previous": "...prior ASR up to 30s...",
       "title":    "Photoshop tutorial",
       "category": "Education"}
  ]},
  {"role": "assistant", "content": [
      {"type": "text_stream", "text_stream": [
        [30.10, 30.32, "okay"],
        [30.32, 30.55, "so"],
        [30.55, 30.91, "let's"],
        // ... hundreds of word triples ...
        [89.40, 89.78, "image"]
      ]}
  ]}
]
```

The illustrative timestamps come from the example shown in
[`livcecc.pdf` L411-413 (Figure 4c)](../livcecc.pdf#L411-L413).

---

## Section 3 — Data-production pipeline

LiveCC's pipeline runs **two parallel branches** on the same 5.7 M-video
source pool: branch **A** for pretraining (cheap YouTube CC) and branch **B**
for SFT (high-quality WhisperX). The pool itself comes from filtering 10.7 M
YouTube IDs collected from HD-VILA, YT-Temporal-1B, VidChapters, HowTo100M,
and a 2024 LLaVA-178K subset
([`livcecc.pdf` L320-323, Figure 2 L276-283](../livcecc.pdf#L276-L283)).
The metadata filter requires resolution ≥ 480p, 30 s ≤ duration ≤ 10 min,
English language, and existing CC + title
([`livcecc.pdf` L327-334](../livcecc.pdf#L327-L334)).

### Branch A — Pretraining (Live-CC-5M)

#### A1. Video-ASR clipping → 30–240 s clips

Source: [`pretrain_to_clips.py`](../livecc/data/production/pretrain_to_clips.py).
Defaults from the CLI parser
([L8-12](../livecc/data/production/pretrain_to_clips.py#L8-L12)):

| Param | Value | Meaning |
|-------|-------|---------|
| `min_clip_sec` | 30 | reject any clip shorter than this |
| `max_clip_sec` | 240 | start a new clip when this length is reached |
| `max_empty_sec` | 3 | start a new clip when the gap between consecutive ASR words exceeds 3 s |
| `min_wps` | 1 | reject clips with words-per-second below this |
| `max_wps` | 4 | reject clips with words-per-second above this |

Two extra preparatory steps inside the same file:

- **Split YouTube CC into per-word triples** ([L15-30](../livecc/data/production/pretrain_to_clips.py#L15-L30)):
  YouTube CC comes as `(start, end, sentence)`. Each sentence is split on
  spaces and the duration is divided uniformly across the words to fabricate
  pseudo word-level timestamps. The paper acknowledges this is approximate
  ([`livcecc.pdf` L498-504](../livcecc.pdf#L498-L504)).
- **Speed sanity check** ([L51-57](../livecc/data/production/pretrain_to_clips.py#L51-L57)):
  reject clips whose `len(words) / duration` falls outside `[min_wps, max_wps]`.

The paper additionally states that for ablation studies the cap is dropped
from 240 s to 60 s for training-efficiency reasons, and clips are ranked by
"word set size" to form 1 M / 2.5 M / 5 M / 10 M subsets
([`livcecc.pdf` L356-369](../livcecc.pdf#L356-L369)).

#### A2. LM-loss filter (text quality)

Source: [`lm_loss.py`](../livecc/data/production/lm_loss.py).

- **Model**: `Qwen/Qwen2-1.5B-Instruct`
  ([L62](../livecc/data/production/lm_loss.py#L62)).
- **Loss**: vanilla causal-LM cross-entropy on the assistant span only
  ([L38-52](../livecc/data/production/lm_loss.py#L38-L52)). The user side
  feeds `Video Title: ... [Previous transcription: ...] Please try to output
  ...` ([L24-35](../livecc/data/production/lm_loss.py#L24-L35)).
- **Threshold**: keep `1.5 ≤ loss ≤ 6.5`
  ([L111](../livecc/data/production/lm_loss.py#L111),
  [`livcecc.pdf` L369-374](../livcecc.pdf#L369-L374)).
- Distributed: shards lines across 8 GPUs
  ([L57, L72, L110](../livecc/data/production/lm_loss.py#L57-L110)).

The paper's interpretation
([`livcecc.pdf` L369-374](../livcecc.pdf#L369-L374)):
> "A very low perplexity suggests the transcript is self-contained and does
> not require visual grounding, while a very high perplexity often correlates
> with poor ASR quality."

#### A3. Talking-head removal (LMM-based)

Source: [`distributed_lmm4asd.py`](../livecc/data/production/distributed_lmm4asd.py).

- **Model**: `Qwen/Qwen2-VL-2B-Instruct`
  ([L79](../livecc/data/production/distributed_lmm4asd.py#L79)).
- **Input**: 8 evenly-spaced frames per video, downscaled to 320×180
  ([L22-26](../livecc/data/production/distributed_lmm4asd.py#L22-L26)).
- **Prompt** ([L31-33](../livecc/data/production/distributed_lmm4asd.py#L31-L33)):
  > "Here are 8 evenly sampled frames from a YouTube video. Are there someone
  > always showing their faces and talking? Answer Yes or No."
- **Decision rule**: take the model's softmax probability on token id `9454`
  (the `Yes` token in the Qwen2 vocab, [L62](../livecc/data/production/distributed_lmm4asd.py#L62)).
  Per the paper, videos are kept when the "Yes" confidence is **below 0.3**
  ([`livcecc.pdf` L376-381](../livcecc.pdf#L376-L381)).

### Branch B — SFT (Live-WhisperX-526K)

#### B1. Restrict to 7 YouTube categories

Categories kept: `HowTo, Sci, Edu, Autos, Sports, Gaming, News`
([`livcecc.pdf` L388-391, Figure 2 box "B1. 7Categories"](../livcecc.pdf#L388-L391)).

The paper explicitly drops `People & Blogs` and `Film & Animation` because
"their ASR content typically lacks correspondence with the visual events"
([`livcecc.pdf` L389-391](../livcecc.pdf#L389-L391)).

There is no script for B1 in the repo — the production README marks it as
"Omitted" ([`livecc/data/production/README.md` L38-40](../livecc/data/production/README.md#L38-L40)).

#### B2. Re-transcribe with WhisperX large-v3-turbo

Source: [`distributed_whisperx.py`](../livecc/data/production/distributed_whisperx.py).
The production README states the model id explicitly
([`README.md` L42-46](../livecc/data/production/README.md#L42-L46)):

> "Use `large-v3-turbo`."

Paper rationale ([`livcecc.pdf` L385-393](../livcecc.pdf#L385-L393)):
> "As the low-quality YouTube CC makes them unsuitable for SFT data, we
> further perform the following steps to obtain high-quality, visually
> grounded ASR transcription ... B2) We employ WhisperX (large-v3-turbo) to
> generate more accurate, word-level aligned ASR transcriptions."

The crucial property is that WhisperX produces **real word-level timestamps**,
unlike A1 which uniformly subdivides sentence-level CC.

#### B3. SFT-style clipping (sentence-aligned)

Source: [`sft_to_clips.py`](../livecc/data/production/sft_to_clips.py).
Same `min_clip_sec=30 / max_clip_sec=240 / max_silence_sec=3` defaults
([L4](../livecc/data/production/sft_to_clips.py#L4)), **plus** a sentence-start
constraint ([L9-12](../livecc/data/production/sft_to_clips.py#L9-L12)):

```python
can_be_start = (i == 0) or (
    (any(words[i-1][-1].endswith(e) for e in ['.', '?', '!']))
    and words[i][-1].isupper()
)
```

i.e. a clip can only start at a position where the **previous word ends in
`.`/`?`/`!`** AND the **current word begins with a capital letter**. Paper
rationale ([`livcecc.pdf` L394-402](../livcecc.pdf#L394-L402)):
> "during the instruction fine-tuning stage, where no pre-ASR context is
> available, we ensure that each clip begins at the start of a sentence."

#### B4. LM-loss filter (stricter than A2)

Same script as A2 ([`lm_loss.py`](../livecc/data/production/lm_loss.py)) but
the **upper bound** is tightened from 6.5 to **5.0**
([`livcecc.pdf` L402-403](../livcecc.pdf#L402-L403)):
> "The same as Step A2, while the range of text perplexity is 1.5 to 5."

The lower bound `1.5` is unchanged.

#### B5. Active-Speaker-Detection removal (Light-ASD)

Source: [`distributed_lighter_asd/`](../livecc/data/production/distributed_lighter_asd/).
The README states the toolchain
([`livecc/data/production/README.md` L58-61](../livecc/data/production/README.md#L58-L61)).

- **Algorithm**: Light-ASD (face detection + tracking + ASD per detected face).
- **Engineering optimisations** quoted in the paper
  ([`livcecc.pdf` L416-421](../livcecc.pdf#L416-L421)):
  > "For efficiency, we optimize Light-ASD pipeline in face detection,
  > tracking, and multiprocessing, achieving a **250× speed-up**. As a result,
  > processing a 5-minute video now takes only 1–1.5 seconds."
- **Decision rule**: keep the clip if the **ASD ratio** (fraction of frames
  in which a fixed talker is detected) is at most **0.05**
  ([`livcecc.pdf` Figure 2 box "B5. Remove Talking-head: ASD ratio <= 0.05"](../livcecc.pdf#L237-L240)).

This is a **stricter and faster** replacement of A3's LMM-based talking-head
filter; the LMM filter is only used in pretraining
([`livcecc/data/production/README.md` L60](../livecc/data/production/README.md#L60)).

#### B6. Synthesise a per-clip user prompt with GPT-4o

Source: [`make_prompt.py`](../livecc/data/production/make_prompt.py).

- **Model**: `gpt-4o-2024-08-06` via Azure
  ([L4-8, L36](../livecc/data/production/make_prompt.py#L4-L36)).
- **Prompt template** ([L11-27](../livecc/data/production/make_prompt.py#L11-L27)):
  ```
  This is the speech transcription from a video clip:
  "{asr}"

  Analyze this transcription and determine if ALL of the following conditions are met:
  1. The speech is describing or commenting on real-time video content
  2. The speaker is not sharing personal experiences or feelings
  3. The transcription is not from multiple people having a conversation
  4. the transcription text has no garbled characters

  If ALL conditions are met, respond with "YES" and suggest a generic user query
  that would prompt an AI to generate commentary in a similar style (without
  including any specific content from the original transcription).

  If ANY condition is NOT met, respond with "NO" and no further explanation.

  Format your response as JSON: {"result": "YES"|"NO", "query": "..."}
  ```
- **Effect** on the dataset:
  - GPT-4o serves a **double duty**: as a binary quality filter (NO → row
    dropped) and as a **user-prompt generator** (YES → `query` field
    populated, [L54-58](../livecc/data/production/make_prompt.py#L54-L58)).
  - The paper notes this lets SFT skip "previous ASR" as context, since the
    generated `query` is now self-sufficient
    ([`livcecc.pdf` L421-427](../livcecc.pdf#L421-L427)):
    > "Since these ASR transcripts lack associated user prompts, we employ
    > GPT-4o to generate a prompt for each sample ... With this prompt, we no
    > longer need pre-ASR applied during SFT."

### Branch summary table

| Step | Pretrain (A) | SFT (B) |
|------|--------------|---------|
| ASR source | YouTube CC, sentence-level → uniformly split into pseudo word-level | WhisperX `large-v3-turbo`, real word-level |
| Clip range | 30–240 s | 30–240 s |
| Word-rate | 1–4 wps | 1–4 wps |
| Sentence-aligned start? | no | **yes** |
| LM-loss range | 1.5–6.5 | 1.5–5.0 |
| Talking-head filter | Qwen2-VL-2B confidence < 0.3 | Light-ASD ratio ≤ 0.05 |
| Per-clip user prompt | none (uses `previous` ASR + title as context) | GPT-4o-generated `query` |
| Final size | ~5 M clips | ~526 K clips |

---

## Section 4 — Training-time data loading

The active class is `LMMDataset` in
[`lmm_dataset.py`](../livecc/data/lmm_dataset.py). It is the **single
dataloader** used by both pretraining and SFT — the only difference between
the two stages is *which* jsonl files are listed in `--annotation_paths`.

### 4.1 jsonl loading via byte-seek index

Constructor ([L46-69](../livecc/data/lmm_dataset.py#L46-L69)):

```python
for annotation_path in annotation_paths:
    seeks = json.loads(readlastline(annotation_path))   # last line = list of byte offsets
    self.handles.extend(zip([annotation_path] * len(seeks), seeks))
```

`readlastline` ([L23-28](../livecc/data/lmm_dataset.py#L23-L28)) does a
`seek(-2, SEEK_END)` walk-back to read the very last `\n`-separated line
without scanning the whole file. This makes opening a 30 GB jsonl O(1).

`load_conversation(index)` then `seek()`s straight to the byte offset and
reads exactly one row ([L71-77](../livecc/data/lmm_dataset.py#L71-L77)):

```python
def load_conversation(self, index):
    annotation_path, seek = self.handles[index]
    with open(annotation_path) as f:
        f.seek(seek)
        line = f.readline()
    return json.loads(line)
```

When multiple jsonl files are listed, their handles are simply
**concatenated** ([L59](../livecc/data/lmm_dataset.py#L59)) — no
re-balancing, so each row is sampled with the same weight.

### 4.2 Per-row routing: streaming vs offline-QA

`getitem()` ([L151-192](../livecc/data/lmm_dataset.py#L151-L192)) inspects
the assistant message to decide which preprocessing path to take
([L168-172](../livecc/data/lmm_dataset.py#L168-L172)):

```python
special_process_for_stream = False
for message in conversation:
    if message['role'] != 'user':
        for element in message['content']:
            special_process_for_stream = 'text_stream' in element  # <- key
            break
```

There are exactly two possible code paths:

| `text_stream` in assistant? | Path taken | Behaviour |
|-----------------------------|-----------|-----------|
| **yes** (Live-CC / Live-WhisperX) | `preprocess_conversation_stream` | rewrite into multi-turn `frame-chunk → text-chunk` conversation (Section 4.3) |
| **no** (LLaVA-Video, etc.) | `process_vision_info` | standard "video first, then prompt, then answer" — the entire video is loaded as one block ([L175-176](../livecc/data/lmm_dataset.py#L175-L176)) |

After both paths, the final `conversation + image_inputs + video_inputs`
are run through `processor.apply_chat_template` and `processor(...)` to
produce token tensors ([L177-184](../livecc/data/lmm_dataset.py#L177-L184)).

### 4.3 `preprocess_conversation_stream` — turning a streaming row into a multi-turn dialogue

Source: [L105-149](../livecc/data/lmm_dataset.py#L105-L149).

#### Step 1: read the entire clip strictly at 2 fps

```python
clip, _, clip_pts = _read_video_decord_plus(user_video_dict,
                                            return_pts=True,
                                            strict_fps=True)
clip = _spatial_resize_video(clip)
```
([L113-114](../livecc/data/lmm_dataset.py#L113-L114))

`FPS = 2` is imported from `qwen_vl_utils.vision_process`
([L9](../livecc/data/lmm_dataset.py#L9)). With `strict_fps=True`, the
returned `clip` tensor is exactly `duration_sec × 2` frames, and `clip_pts`
is the per-frame absolute timestamp.

#### Step 2: build the first conversation turn (3-second initial chunk)

```python
start_timestamp, end_timestamp = 0, self.initial_fps_frames / FPS
# initial_fps_frames = int(FPS) * 3 = 6, so 0 → 3.0 s

phrase, next_start_from = get_phrase_before_timestamp(
    assistant_text_stream, clip_pts[self.initial_fps_frames - 1])

conversation = [
    {'role': 'user', 'content': [
        {'type': 'text',  'text': f'Time={start_timestamp:.1f}-{end_timestamp:.1f}s'},
        {'type': 'video', 'video': clip[:self.initial_fps_frames]},
        user_query_dict,             # original user prompt
    ]},
    {'role': 'assistant',
     'content': [{'type': 'text', 'text': phrase + ' ...'}]},
]
```
([L116-129](../livecc/data/lmm_dataset.py#L116-L129),
 [`DataArguments` L18](../livecc/data/lmm_dataset.py#L18))

`get_phrase_before_timestamp` ([L36-43](../livecc/data/lmm_dataset.py#L36-L43))
walks the word-stream until `word.end > target_timestamp` and joins the
words found so far. The result is the literal ASR uttered between 0 s and
the end of frame 6.

The trailing `' ...'` is a fixed soft-EOS marker — see Section 6.

#### Step 3: append one chunk per second for the rest of the video

```python
for i in range(self.initial_fps_frames, len(clip), self.streaming_fps_frames):
    # streaming_fps_frames = int(FPS) = 2  → 1-second chunks
    start_timestamp = i / FPS
    end_timestamp   = (i + self.streaming_fps_frames) / FPS

    phrase, next_start_from = get_phrase_before_timestamp(
        assistant_text_stream,
        clip_pts[i + self.streaming_fps_frames - 1],
        start_from=next_start_from)        # resume from where we left off

    frames = clip[i : i + self.streaming_fps_frames]   # 2 frames
    conversation.extend([
        {'role': 'user', 'content': [
            {'type': 'text',  'text': f'Time={start_timestamp:.1f}-{end_timestamp:.1f}s'},
            {'type': 'video', 'video': frames},
        ]},
        {'role': 'assistant',
         'content': [{'type': 'text', 'text': phrase + ' ...'}]},
    ])
    frames_list.append(frames)
```
([L131-144](../livecc/data/lmm_dataset.py#L131-L144),
 [`DataArguments` L19](../livecc/data/lmm_dataset.py#L19))

Note: the user side of follow-up turns has **no text query**, only the time
label and the next 2 frames. The "user" turn is essentially a beat in the
clock, not a real question.

#### Step 4: trim trailing silent chunks

```python
while conversation[-1]['content'][0]['text'] == ' ...':
    conversation = conversation[:-2]
    frames_list   = frames_list[:-1]
```
([L146-148](../livecc/data/lmm_dataset.py#L146-L148))

If the last few chunks have no ASR words at all (they would only emit the
soft-EOS `' ...'`), they are dropped so the model is never asked to learn
"emit nothing".

### 4.4 Final shape comparison

For the same 60-second clip, the two routing paths yield very different
token sequences:

| | streaming row (`text_stream` present) | offline-QA row (LLaVA-Video) |
|---|---|---|
| Number of `user` turns | `1 + (duration_sec - 3)` ≈ 58 | 1 |
| Number of `assistant` turns | same as user turns ≈ 58 | 1 |
| Each user turn carries | 1 timestamp text + 2–6 frames (+ original query in turn 1) | full video + question |
| Each assistant turn carries | a few ASR words + ` ...` | the full free-form answer |
| Total visual tokens | identical (same 120 frames) | identical |
| Total text tokens | many short bursts | one long answer |

Both rows go through the **same** subsequent
`processor.apply_chat_template + processor(...)` pipeline, so the resulting
tensors are interchangeable as far as the trainer is concerned.

---

## Section 5 — Training hyper-parameters

Both pretraining and SFT use the **same `train.py`** (
[`livecc/train.py`](../livecc/train.py); not detailed here, it is a thin
wrapper around HuggingFace `Trainer`). The differences live entirely in the
shell scripts.

### 5.1 Side-by-side run config

| Key | Pretrain ([`pt_local.sh`, README L76-110](../livecc/README.md#L76-L110)) | SFT ([`sft_local.sh` L1-38](../livecc/scripts/sft_local.sh#L1-L38)) |
|---|---|---|
| Initial checkpoint | `Qwen/Qwen2-VL-7B` ([README L104](../livecc/README.md#L104)) | `chenjoya/LiveCC-7B-Base` ([sft_local L28](../livecc/scripts/sft_local.sh#L28)) |
| `--annotation_paths` | `datasets/live_cc_5m_with_seeks.jsonl` ([README L105](../livecc/README.md#L105)) | `live_whisperx_526k`, `llava_ov_single_image_text_mix`, `llava_ov_multi_image`, `llava_hound_video`, `llava_video_178k` ([sft_local L29-34](../livecc/scripts/sft_local.sh#L29-L34)) |
| `--learning_rate` | `2e-5` ([README L94](../livecc/README.md#L94)) | `1e-5` ([sft_local L5,18](../livecc/scripts/sft_local.sh#L5-L18)) |
| `--num_train_epochs` | `1` ([README L98](../livecc/README.md#L98)) | `1` ([sft_local L22](../livecc/scripts/sft_local.sh#L22)) |
| `--lr_scheduler_type` | `cosine` ([README L97](../livecc/README.md#L97)) | `cosine` ([sft_local L21](../livecc/scripts/sft_local.sh#L21)) |
| `--warmup_ratio` | `0.03` ([README L95](../livecc/README.md#L95)) | `0.03` ([sft_local L19](../livecc/scripts/sft_local.sh#L19)) |
| `--per_device_train_batch_size` | `1` ([README L92](../livecc/README.md#L92)) | `1` ([sft_local L16](../livecc/scripts/sft_local.sh#L16)) |
| `--gradient_accumulation_steps` | `64` ([README L93](../livecc/README.md#L93)) | `64` ([sft_local L17](../livecc/scripts/sft_local.sh#L17)) |
| `--nproc_per_node` | `8` ([README L84](../livecc/README.md#L84)) | `8` ([sft_local L8](../livecc/scripts/sft_local.sh#L8)) |
| Effective batch (single node) | 1 × 8 × 64 = **512** ([paper L880-882](../livcecc.pdf#L880-L882)) | 1 × 8 × 64 = **512** |
| `--bf16` / `--tf32` | both `True` ([README L101-102](../livecc/README.md#L101-L102)) | both `True` ([sft_local L25-26](../livecc/scripts/sft_local.sh#L25-L26)) |
| `--gradient_checkpointing` | `True` ([README L103](../livecc/README.md#L103)) | `True` ([sft_local L27](../livecc/scripts/sft_local.sh#L27)) |
| `--freeze_modules` | `visual` ([README L107](../livecc/README.md#L107)) | `visual` ([sft_local L36](../livecc/scripts/sft_local.sh#L36)) |
| `--use_liger_kernel` | `True` ([README L108](../livecc/README.md#L108)) | `True` ([sft_local L37](../livecc/scripts/sft_local.sh#L37)) |
| DeepSpeed config | `ZeRO-2` ([README L86](../livecc/README.md#L86)) | `ZeRO-2` ([sft_local L9](../livecc/scripts/sft_local.sh#L9)) |
| `--save_steps` | `1000` | `1000` |

### 5.2 Visual-token budget (env vars)

These are exported **before** launching training and read by
`qwen_vl_utils.vision_process` ([imported in `lmm_dataset.py` L9](../livecc/data/lmm_dataset.py#L9)).
Both pretrain and SFT use the **same** values
([README L77-79](../livecc/README.md#L77-L79),
 [sft_local L1-3](../livecc/scripts/sft_local.sh#L1-L3)):

| Variable | Value | Decoded | Meaning |
|----------|------:|---------|---------|
| `VIDEO_MIN_PIXELS` | `78400` | 100 × 28 × 28 | minimum visual frame tokens sent to LLM = 100 |
| `FPS_MAX_FRAMES` | `480` | — | max frames per video; 480 / FPS / 60 = 4 minutes at FPS=2 |
| `VIDEO_MAX_PIXELS` | `19267584` | 24576 × 28 × 28 | max overall video tokens sent to LLM = 24 K (leaves 8 K for language in a 32 K-context model) |

Streaming-specific frame counts come from `DataArguments` defaults
([`lmm_dataset.py` L17-20](../livecc/data/lmm_dataset.py#L17-L20)):

| Variable | Value | Effect |
|----------|------:|--------|
| `FPS` | 2 | imported from `qwen_vl_utils` ([L9](../livecc/data/lmm_dataset.py#L9)); video is decoded at exactly 2 frames/second |
| `initial_fps_frames` | `int(FPS) * 3 = 6` | first user turn carries 6 frames = 3 seconds |
| `streaming_fps_frames` | `int(FPS) = 2` | every subsequent user turn carries 2 frames = 1 second |
| `with_context` | `False` | by default the loader does NOT inject `title` / `previous` into the user prompt; this is on for some pretrain ablations only |

### 5.3 Paper-stated overrides

The paper notes a few **ablation-only** overrides
([`livcecc.pdf` L863-870](../livcecc.pdf#L863-L870)):

> "Specifically, during ablation studies of pre-training, we reduce the
> maximum number of frames from 768 to 120 and shorten the visual context
> length from 128K to 16K tokens. During formal pre-training and SFT, we
> increase the frame limit to 480 and extend the visual context length to
> 24K, while slightly lowering the minimum spatial resolution from
> 128×28×28 to 100×28×28."

So:
- The values in `pt_local.sh` / `sft_local.sh` (frame limit 480, max video
  tokens 24 K, min pixels 100×28×28) are the **formal-run** settings.
- During ablation runs (Tables 1a–c in the paper) the budget was tighter
  (frame limit 120, ctx 16 K).

### 5.4 Compute footprint quoted by the paper

Total batch size is **512** ([paper L880-882](../livcecc.pdf#L880-L882)):

> "The batch size for pre-training and SFT is 512 on 128 GPUs, with a
> learning rate of 2e-5 for pre-training and 1e-5 for SFT."

The single-node `pt_local.sh` / `sft_local.sh` recipes get to 512 with
`1 × 8 × 64`. The 128-GPU paper run uses a different `gradient_accumulation_steps`
to land on the same 512 (not specified explicitly in the scripts).

---

## Section 6 — Loss computation

Loss is computed by HuggingFace `Trainer` from `inputs.labels`. The dataset
prepares `labels` itself in `getitem()`
([`lmm_dataset.py` L184-191](../livecc/data/lmm_dataset.py#L184-L191)):

```python
input_ids = inputs.input_ids
labels = torch.full_like(input_ids, fill_value=-100, dtype=input_ids.dtype)
im_start_idxs = (input_ids == self.im_start_id).nonzero()
im_end_idxs   = (input_ids == self.im_end_id).nonzero()
for (b, im_start_idx), (_, im_end_idx) in zip(im_start_idxs, im_end_idxs):
    if input_ids[b, im_start_idx + 1] == self.assistant_id:
        labels[b, im_start_idx + 3 : im_end_idx + 1] = \
            input_ids[b, im_start_idx + 3 : im_end_idx + 1]
inputs['labels'] = labels
```

### 6.1 What this means

- `labels` is initialised to **`-100` everywhere**, which is the standard
  HuggingFace ignore-index (excluded from cross-entropy).
- The code finds every `<|im_start|>` token (Qwen2 turn-marker) and the
  matching `<|im_end|>`. The processor and the special-token ids are looked
  up at construction time
  ([`lmm_dataset.py` L61-65](../livecc/data/lmm_dataset.py#L61-L65)):
  ```python
  self.im_start_id, self.assistant_id, self.newline_id, self.im_end_id = \
      processor.tokenizer('<|im_start|>assistant\n<|im_end|>').input_ids
  ```
- For each turn whose token right after `<|im_start|>` is `assistant`, the
  range from `im_start_idx + 3` (skipping `<|im_start|>`, `assistant`, `\n`)
  to `im_end_idx + 1` (inclusive of the closing `<|im_end|>`) is **un-masked**
  by copying the original token ids back into `labels`.
- All `user` and `system` turns stay at `-100` and contribute zero to the
  loss, as does the prefix `<|im_start|>assistant\n` of each assistant turn.

### 6.2 Final objective

The trainer then runs Qwen2-VL forward + standard causal LM cross-entropy
on `(logits, labels)`. Because of the mask above, the **only** positions
that contribute to the loss are the assistant-content tokens. There are no
auxiliary losses, no contrastive loss, and no separate streaming-specific
loss term — `Trainer` is used unmodified.

### 6.3 Streaming vs offline-QA: same loss, different supervision density

Both row types are reduced to the same Qwen ChatML format before tokenization,
so the loss formula is identical:

| | streaming row | offline-QA row |
|---|---|---|
| Number of assistant turns | many short ones (`phrase + ' ...'`) | one long one |
| Loss-contributing tokens per row | sum of all short bursts | the entire answer |
| Per-turn supervision pattern | learn 1 second of speech given 1 second of frames + KV cache | learn full answer given full video |
| Inference behaviour the model is shaped into | "speak a few words then stop and wait for next chunk" (`...` is the soft-EOS that signals "more coming") | "produce the complete answer once" |

The trailing `' ...'` (literally space + three dots,
[L128, L142](../livecc/data/lmm_dataset.py#L128-L142)) is therefore a
**learned token sequence** — at inference time it lets the loop decide when
to pause generation and feed in the next `streaming_fps_frames` worth of
new visual input. The paper makes this explicit
([`livcecc.pdf` L506-511](../livcecc.pdf#L506-L511)):

> "To disambiguate the true end-of-sequence (EOS) from temporary pauses in
> streaming, we simply use the **ellipsis token ('...')** as an special EOS
> indicator appended to the per-frame text tokens. For silent frames without
> corresponding subtitles, we directly predict this ellipsis token."

---

*End of reference.*

This document covers Sections 1–6 only. Ablation-study tables, design
rationale, and the mapping to our ultrasound project are intentionally
out of scope here.
