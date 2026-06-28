# Ultrasound Video Understanding — Full Data Pipeline

## Overview

```
YouTube Videos → VLM Classification → ASR Transcription → Video Segmentation → QA Generation → Training Data
```

---

## Step 1: Video Crawling

| Item | Details |
|------|---------|
| **Logic** | Search YouTube/Bilibili by ultrasound keywords, download videos + metadata |
| **Tool** | UltrasoundCrawler (yt-dlp + YouTube Data API) |
| **Script** | `UltrasoundCrawler_KeyCode_20260323_v2/cli.py` |
| **Input** | Search keywords (e.g., "POCUS ultrasound scanning tutorial") |
| **Output** | `.mp4` files in `output/media/` + `videos.jsonl` metadata |
| **Auto-classification** | By title/description keywords → `scan_tutorial/`, `case_reasoning/`, `organ_system_lecture/`, `uncategorized/` |

---

## Step 2: Video Classification (VLM)

| Item | Details |
|------|---------|
| **Logic** | VLM analyzes video content to classify into 6 types |
| **Tool** | Qwen3-VL-2B-Instruct (local, MPS/CUDA) |
| **Script** | `scripts/batch_filter.py` + `src/ablation_input_modes.py` |
| **Input** | Video file (.mp4) |
| **Output** | Classification JSON (video_type, training_value, recommendation) |

### 6 Video Types

| Type | Description | Training Value |
|------|-------------|---------------|
| `pure_ultrasound` | Only US machine screen, no faces/slides | High |
| `hands_on_tutorial` | Instructor + probe technique + US display | High |
| `case_discussion` | Instructor annotating/discussing US clips | Medium |
| `ppt_lecture` | Slides/presentations with occasional US | Low |
| `diagram_animation` | Anatomical diagrams, 3D animations | None |
| `mixed` | Multiple types alternating | Needs trimming |

### Decision
- **keep**: pure_ultrasound, hands_on_tutorial
- **trim**: case_discussion, mixed
- **discard**: ppt_lecture, diagram_animation

---

## Step 3: ASR Transcription

| Item | Details |
|------|---------|
| **Logic** | Extract audio from video → Whisper speech recognition → timestamped text |
| **Tool** | ffmpeg (audio extraction) + faster-whisper (ASR, CPU int8) |
| **Script** | `scripts/asr_pipeline.py` |
| **Input** | Video file (.mp4) |
| **Output** | `transcripts/VIDEO_ID.json` |

### Output Format

```json
{
  "video_id": "8V649L5Q368",
  "language": "en",
  "duration_sec": 1136.8,
  "segments": [
    {"start": 0.7, "end": 4.5, "text": "Hi, I'm Dr. John Kugler..."},
    {"start": 4.5, "end": 7.1, "text": "Today, we're going to be learning..."}
  ]
}
```

### Model Options
- `base`: fastest, good enough for English
- `medium`: better accuracy
- `large-v3`: best quality, slower

---

## Step 4: Video Clipping

| Item | Details |
|------|---------|
| **Logic** | Two-stage: grayscale histogram finds candidates → GPT-4o verifies semantically |
| **Tool** | OpenCV (histogram) + GPT-4o API (semantic verification) |
| **Script** | `scripts/video_segmentation.py` + `scripts/llm_segmentation.py` |
| **Input** | Video file + ASR transcript JSON |
| **Output** | `results/VIDEO_ID_clips.json` |

### Stage 1: Grayscale Histogram Analysis (free, ~10s)

1. For each ASR segment, extract frame at midpoint
2. Convert to grayscale → compute 256-bin histogram
3. Compare with previous frame using Pearson correlation
4. `similarity < 0.4` → mark as candidate cut point

### Stage 2: GPT-4o Semantic Verification (~$0.03, ~6s)

Input to GPT-4o:
- Full ASR transcript (all segments with timestamps)
- List of candidate cut points from Stage 1

GPT-4o decides:
- Which candidates are real topic changes → **keep**
- Which are just camera zooms → **remove**
- What topic changes were missed → **add**
- For each cut: `topic` label (content before this cut)

### Output Format

```json
{
  "video_id": "8V649L5Q368",
  "method": "histogram_llm",
  "num_clips": 11,
  "clips": [
    {
      "clip_idx": 0,
      "start": 28.0,
      "end": 179.3,
      "duration": 151.3,
      "text": "To get started, we'll start with pneumothorax...",
      "topic": "Pneumothorax diagnosis and ultrasound technique"
    }
  ]
}
```

### Constraints
- Minimum clip: 30s
- Maximum clip: 300s
- Clips end at natural topic boundaries

---

## Step 5: QA Generation (Dual-Track)

The QA generation is split into **two parallel tracks** by question type:

| Track | Types | Information access | Generator | Validator |
|-------|-------|--------------------|-----------|-----------|
| **5a — Offline** | `scene_description`, `fine_grained`, `knowledge` | Whole clip | GPT-4o Vision (6 frames + ASR) | none |
| **5b — Streaming** | `sonographer_intent`, `next_action_guidance` | Question grounded in past video; answer drawn from past+future ground truth (with audio) | **Gemini 2.5 Flash (oracle)**, video-clip input via OpenRouter | **Gemini 2.5 Flash (audit)** via OpenRouter |

> 🎬 **Streaming track now uses real video clips, not sampled frames.** Both the generator and the validator receive two short mp4 segments — one for SEEN (`clip_start → query_time`) and one for FUTURE (`query_time → clip_end`) — uploaded as OpenAI/OpenRouter `type:"file"` content blocks. Each segment carries its **original visual frames AND audio (operator's narration)** end-to-end (this delivery format was empirically verified to produce non-zero `prompt_tokens_details.video_tokens` on OpenRouter Gemini 2.5 Flash). All video plumbing is centralised in [`scripts/_video_llm.py`](../scripts/_video_llm.py) (`cut_clip`, `build_video_block`, `call_with_content`).

### Why this split

- `scene_description` / `fine_grained` / `knowledge` benefit from full-clip context — let the model see the entire clip and write holistic answers.
- `sonographer_intent` / `next_action_guidance` are intrinsically time-sensitive. The QUESTION must be grounded in what's been seen so far (no future leakage), but the ANSWER should describe what actually happens next in the clip — i.e. real ground truth, not a guess. So the generator is given full-clip access (oracle) but instructed to write the question from the SEEN-only perspective.
- A separate **Gemini 2.5 Flash** validator (independent forward pass, different prompt, different time windows from the generator) audits every streaming QA on a **single binary criterion**: question must be grounded in SEEN only AND answer must be faithful to what's visible/audible in SEEN+FUTURE. Verdict is `pass` or `fail`.
- *Note on independence:* generator and validator currently share the same model family (Gemini 2.5 Flash). They are NOT cross-family. We rely on the stricter prompt + smaller FUTURE window in the validator (default 30 s vs uncapped for the generator) to flush leakage / hallucination. A cross-family validator (e.g. Qwen2.5-VL on OpenRouter) is left as future work.

### Dual-Track Flow Diagram

```
                 ┌────────────────────────────────────────────────────────┐
                 │  Clip                                                  │
                 │  start = clip_start, end = clip_end, full ASR text     │
                 └─────────────────────────┬──────────────────────────────┘
                                           │
                ┌──────────────────────────┴──────────────────────────┐
                │                                                     │
                ▼                                                     ▼
   ┌────────────────────────────┐               ┌──────────────────────────────────────┐
   │  Track A: OFFLINE  (5a)    │               │  Track B: STREAMING  (5b)            │
   │  Generator: GPT-4o Vision  │               │  Generator: Gemini 2.5 Flash (oracle)│
   │                            │               │  Input mode: VIDEO clips (with audio)│
   ├────────────────────────────┤               ├──────────────────────────────────────┤
   │ Input:                     │               │ For each anchor t in 0.25/0.5/0.75   │
   │   • 6 frames (full clip)   │               │ Input:                               │
   │   • full ASR               │               │   • SEEN_VIDEO  mp4 (clip_start..t,  │
   │ Output: 3 QA types         │               │     capped to 240 s)                 │
   │   - scene_description      │               │   • FUTURE_VIDEO mp4 (t..clip_end,   │
   │   - fine_grained           │               │     uncapped)                        │
   │   - knowledge              │               │   • full ASR                         │
   │ Question rule: free        │               │ Output: 2 QA types per anchor        │
   │ Answer rule: free          │               │   - sonographer_intent               │
   │                            │               │   - next_action_guidance             │
   │                            │               │ Question rule: from SEEN_VIDEO only  │
   │                            │               │ Answer rule:   from SEEN+FUTURE      │
   └─────────────┬──────────────┘               └─────────────┬────────────────────────┘
                 │                                            │
                 │                                            ▼
                 │                              ┌──────────────────────────────────┐
                 │                              │  Validator (5c)                  │
                 │                              │  Gemini 2.5 Flash via OpenRouter │
                 │                              │  Input mode: VIDEO clips         │
                 │                              ├──────────────────────────────────┤
                 │                              │ Input per QA:                    │
                 │                              │   • SEEN_VIDEO   (cap 240 s)     │
                 │                              │   • FUTURE_VIDEO (cap 30 s)      │
                 │                              │   • the QA pair                  │
                 │                              │ Single binary check:             │
                 │                              │   (1) Q grounded in SEEN_VIDEO?  │
                 │                              │   (2) A faithful to SEEN+FUTURE? │
                 │                              │ Verdict: pass / fail + reason    │
                 │                              └─────────────┬────────────────────┘
                 │                                            │
                 │                                            │ drop "fail"
                 │                                            ▼
                 │                              ┌──────────────────────────────────┐
                 │                              │  *_streaming_qa_validated.json   │
                 │                              │  (~80% pass rate observed)       │
                 │                              └─────────────┬────────────────────┘
                 │                                            │
                 ▼                                            ▼
   ┌────────────────────────────┐               ┌────────────────────────────────────┐
   │  *_offline_qa.json         │               │  validated streaming QA            │
   │  3 QA × num_clips          │               │  ~5 QA × num_clips (after drop)    │
   └─────────────┬──────────────┘               └─────────────┬──────────────────────┘
                 │                                            │
                 └──────────────────────┬─────────────────────┘
                                        │
                                        ▼
                       ┌──────────────────────────────────────┐
                       │  Merged QA dataset                   │
                       │  ~8 QA per clip, sorted by timestamp │
                       │  ready for training / evaluation     │
                       └──────────────────────────────────────┘
```

### 5 QA Types

| Type | Track | What it asks | 中文解释 |
|------|-------|-------------|---------|
| `scene_description` | offline | What happens in this clip from beginning to end? | 场景描述（整段过程） |
| `fine_grained` | offline | Specific landmarks, echogenicity, probe orientation, measurements. | 细粒度视觉特征 |
| `knowledge` | offline | Relevant medical knowledge, clinical significance, diagnostic criteria. | 医学知识 / 临床意义 |
| `sonographer_intent` | **streaming** | What is the operator currently trying to do/find at t? | 操作者当前意图 |
| `next_action_guidance` | **streaming** | What should the sonographer do next based on what's been seen so far? | 下一步操作指导 |

### Step 5a — Offline QA Generation

| Item | Details |
|------|---------|
| **Script** | `scripts/qa_generation.py` |
| **Input** | Video + clips JSON |
| **Output** | `results/qa/{video_id}_offline_qa.json` |
| **Per clip** | 6 frames @ 512×512 (detail=low) + full ASR (≤3000 chars) → 1 QA per type (3 total) |
| **Cost/video** (~11 clips) | ~$0.30 |

### Step 5b — Streaming QA Generation

The streaming generator is an **oracle**: it sees BOTH past and future content within the clip, plus the full ASR. The generator MUST, however, write the question as if only past content were known. The answer is allowed to use the full clip context as ground truth — it should describe the *real* intent or *real* next action that actually occurs in the video.

| Item | Details |
|------|---------|
| **Script** | `scripts/streaming_qa_generation.py` |
| **Generator** | `google/gemini-2.5-flash` via **OpenRouter** (OpenAI-compatible API) |
| **Input mode** | Two mp4 video clips (visual frames + audio), uploaded as `type:"file"` content blocks |
| **Input** | Video + clips JSON |
| **Output** | `results/qa/{video_id}_streaming_qa.json` |
| **Time anchors** | `[0.25, 0.5, 0.75]` of each clip's duration |
| **Per anchor** | `SEEN_VIDEO` mp4 `[clip_start, query_time]` (capped to last 240 s) + `FUTURE_VIDEO` mp4 `[query_time, clip_end]` (uncapped) + full ASR → 1 intent + 1 next_action |
| **Question writing rule** | Must be derivable from SEEN_VIDEO only (no future leakage, visual OR audible) |
| **Answer writing rule** | May use SEEN_VIDEO + FUTURE_VIDEO + ASR; must describe what actually happens in the clip |
| **Yield** (~11 clips × 3 anchors × 2 types) | ~66 QA per video |
| **Cost/video** (per QA ≈ $0.007 × 66) | ~$0.45 |
| **Smoke-test result** (clip 0, 6 QA) | 6/6 generated, $0.022, 50k video tokens, ~3 min wall time |

### Step 5c — Streaming QA Validation (single binary verdict)

An independent **second forward pass** audits each streaming QA on **a single binary criterion**. Currently the validator and generator share the same model family (Gemini 2.5 Flash) — they are NOT cross-family. We rely on a stricter prompt and a much smaller FUTURE window (default 30 s vs the generator's uncapped FUTURE) to keep the validator from rubber-stamping. Cross-family validation is left as future work.

1. **Question grounding** — Is the question writable from SEEN_VIDEO only? (no future leakage)
2. **Answer faithfulness** — Does the answer correspond to what is actually visible/audible in SEEN_VIDEO+FUTURE_VIDEO? (not a hallucination)

A QA passes only if BOTH conditions hold.

| Item | Details |
|------|---------|
| **Script** | `scripts/qa_validator.py` |
| **Validator** | `google/gemini-2.5-flash` via **OpenRouter** (OpenAI-compatible API) |
| **Input mode** | Two mp4 video clips (visual + audio), as `type:"file"` content blocks |
| **Input** | Streaming QA JSON + video |
| **Per QA** | `SEEN_VIDEO` (cap 240 s) + `FUTURE_VIDEO` (cap 30 s) → `{verdict, reason, validator_model}` |
| **Output** | `results/qa/{video_id}_streaming_qa_validated.json` |
| **Verdict** | `pass` if question grounded AND answer faithful; else `fail` |
| **Drop policy** | `fail` QA dropped by default (`--keep-failed` to retain for inspection) |
| **Cost/video** (per QA ≈ $0.005 × 66) | ~$0.30 |
| **Smoke-test result** (clip 0, 6 QA) | 6/6 pass, $0.031, 93k video tokens, ~6 min wall time |

### Reliability — retry policy on transient OpenRouter errors

OpenRouter occasionally returns `504` (Gateway Timeout) or `429` (rate limit) when the upstream provider is busy. `scripts/_video_llm.py:call_with_content` handles this transparently:

| Error class | Detection | Backoff |
|-------------|-----------|---------|
| Network / SDK exception | raised exception | exponential (8 s, 13 s, 20 s, 33 s, 52 s) |
| `code: 504` empty envelope | `resp.choices` is empty | exponential |
| `code: 429` empty envelope | `'429'` / `'rate'` substring in error string | **fixed 60 s** (let Gemini's RPM bucket reset) |
| Empty `message.content` | `text == ""` | exponential, with `finish_reason` logged |

Default = **5 retries** before raising. Per-anchor failures are caught and the rest of the run continues; a non-zero `error` count is reported in the final summary.

### Output Format (per-clip QA file)

Offline (`{video_id}_offline_qa.json`):
```json
{
  "video_id": "8V649L5Q368",
  "qa_types": ["scene_description", "fine_grained", "knowledge"],
  "num_qa_pairs": 33,
  "qa_pairs": [
    {
      "source": "offline",
      "type": "scene_description",
      "question": "...",
      "answer": "...",
      "clip_idx": 1,
      "clip_start": 179.3,
      "clip_end": 250.2,
      "topic": "...",
      "timestamp_hint": "whole_clip"
    }
  ]
}
```

Streaming validated (`{video_id}_streaming_qa_validated.json`):
```json
{
  "video_id": "8V649L5Q368",
  "qa_types": ["sonographer_intent", "next_action_guidance"],
  "validator_model": "google/gemini-2.5-flash",
  "validation_stats": {"pass": 58, "fail": 8, "error": 0},
  "num_after_validation": 58,
  "streaming_qa": [
    {
      "source": "streaming",
      "type": "next_action_guidance",
      "question": "Based on what we've seen so far, what should the sonographer do next?",
      "answer": "The operator tilts the probe cranially and increases depth to bring the upper pole of the kidney into view, then applies color Doppler to assess perfusion.",
      "clip_idx": 1,
      "clip_start": 179.3,
      "clip_end": 250.2,
      "query_time": 197.0,
      "ratio": 0.25,
      "validation": {
        "verdict": "pass",
        "reason": "Question is grounded in [SEEN] frames only; the answer accurately describes the cranial tilt and color Doppler step actually shown in [FUTURE].",
        "validator_model": "google/gemini-2.5-flash"
      }
    }
  ]
}
```

---

## Cost & Time Estimates

> Per-QA numbers below are measured (clip 0 of `8V649L5Q368`, 6 QA each).
> Per-video totals assume ~11 clips × 3 anchors × 2 types = 66 streaming QA + 33 offline QA.

| Step | Time/video | Cost/video | API used |
|------|-----------|-----------|----------|
| ASR Transcription | ~2 min | Free | local (faster-whisper) |
| Segmentation (histogram) | ~10 s | Free | local (OpenCV) |
| Segmentation (LLM) | ~6 s | ~$0.03 | OpenAI (GPT-4o) |
| Offline QA (5a) | ~50 s | ~$0.30 | OpenAI (GPT-4o Vision) |
| Streaming QA generation (5b) | 8–15 min* | ~$0.45 | OpenRouter (Gemini 2.5 Flash, video) |
| Streaming QA validation (5c) | 8–15 min* | ~$0.30 | OpenRouter (Gemini 2.5 Flash, video) |
| **Total per video** | **~25–35 min** | **~$1.10** | |
| **20 videos** | **~10 h** (sequential) | **~$22** | |

\* 5b/5c walltime is dominated by `mp4 base64 upload + Gemini video tokenization + occasional 504/429 retries`. The retry wait dominates; the actual per-call billed compute is ~10–30 s. See "Reliability — retry policy" above.

---

## CLI Commands

> All scripts auto-load `OPENAI_API_KEY` and `OPENROUTER_API_KEY` from `.env`
> via `scripts/_env_loader.py` — manual `export` is no longer required.

```bash
# Step 1: Crawl videos
python UltrasoundCrawler_KeyCode_20260323_v2/cli.py --source youtube --max-results 100 --download-media

# Step 2: VLM Classification (batch)
python scripts/batch_filter.py --input-dir path/to/media --num-frames 8

# Step 3: ASR Transcription (batch)
python scripts/asr_pipeline.py --batch --input-dir path/to/media --model base

# Step 4: Video Segmentation (with LLM verification, default — uses OPENAI_API_KEY)
python scripts/video_segmentation.py --video path.mp4 --transcript transcripts/ID.json

# Step 4: Histogram-only segmentation (no API key needed)
python scripts/video_segmentation.py --video path.mp4 --transcript transcripts/ID.json --no-llm

# Step 5a: Offline QA Generation (uses OPENAI_API_KEY for GPT-4o Vision)
python scripts/qa_generation.py --video path.mp4 --clips results/clips/ID_clips.json

# Step 5b: Streaming QA Generation (uses OPENROUTER_API_KEY, Gemini 2.5 Flash with video clips)
python scripts/streaming_qa_generation.py \
    --video path.mp4 \
    --clips results/clips/ID_clips.json
# optional caps:
#   --seen-window-sec 240        (default; latter portion of SEEN before query_time)
#   --future-window-sec -1       (default; uncapped FUTURE for ground-truth)
#   --single-clip 0              (debug: only this clip)
#   --ratios 0.25,0.5,0.75       (default time anchors)

# Step 5c: Streaming QA Validation (uses OPENROUTER_API_KEY, Gemini 2.5 Flash with video clips)
python scripts/qa_validator.py \
    --streaming-qa results/qa/ID_streaming_qa.json \
    --video path.mp4
# optional caps:
#   --seen-window-sec 240        (default)
#   --future-window-sec 30       (default; tight window for hallucination check)
#   --keep-failed                (keep verdict='fail' for inspection)
#   --max-qa N                   (smoke test; only first N QA)

# Step 5d: Merge to LiveCC-style JSONL
python scripts/qa_merge.py \
    --video-id ID \
    --transcript    results/transcripts/ID.json \
    --clips         results/clips/ID_clips.json \
    --offline-qa    results/qa/ID_offline_qa.json \
    --streaming-qa  results/qa/ID_streaming_qa_validated.json \
    --out           results/training_data/ID.jsonl --overwrite

# Or run the entire pipeline (Steps 3 → 5c) end-to-end:
python scripts/run_pipeline.py --video path.mp4 --skip-asr --skip-segmentation
```

---

## Data Flow Diagram

```
YouTube
  │
  ▼ [Step 1: Crawl]
20 videos (.mp4)
  │
  ▼ [Step 2: VLM Classify]
Keep: 12 videos (US + tutorial + case)
Discard: 8 videos (PPT, animation, etc.)
  │
  ▼ [Step 3: ASR]
transcripts/VIDEO_ID.json (timestamped text)
  │
  ▼ [Step 4: Segment]
results/clips/VIDEO_ID_clips.json (8-11 clips, each with topic)
  │
  ├─▶ [Step 5a: Offline QA]    ──┐
  │   results/qa/VIDEO_ID_offline_qa.json   (~33 QA: scene/fine/knowledge)
  │                               │
  └─▶ [Step 5b: Streaming QA]  ──┤
      results/qa/VIDEO_ID_streaming_qa.json (~66 QA: intent/next_action)
                                  │
                                  ▼
                          [Step 5c: Validate]
                          Gemini 2.5 Flash audits every streaming QA
                          → results/qa/VIDEO_ID_streaming_qa_validated.json
                                  │
                                  ▼
                          [Step 5d: Merge]
                          results/training_data/VIDEO_ID.jsonl
                                  │
                                  ▼
                          LoRA SFT on Qwen2-VL-7B
```
