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

## Step 4: Video Segmentation

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
| **5a — Offline** | `scene_description`, `fine_grained`, `knowledge` | Whole clip | GPT-4o Vision | none |
| **5b — Streaming** | `sonographer_intent`, `next_action_guidance` | Question grounded in past frames; answer drawn from past+future ground truth | GPT-4o Vision (oracle) | **Gemini 2.5 Pro (via OpenRouter)** |

### Why this split

- `scene_description` / `fine_grained` / `knowledge` benefit from full-clip context — let the model see the entire clip and write holistic answers.
- `sonographer_intent` / `next_action_guidance` are intrinsically time-sensitive. The QUESTION must be grounded in what's been seen so far (no future leakage), but the ANSWER should describe what actually happens next in the clip — i.e. real ground truth, not a guess. So the generator is given full-clip access (oracle) but instructed to write the question from the SEEN-only perspective.
- A separate **Gemini 2.5 Pro** validator (different model family from GPT-4o → cross-family check) audits every streaming QA on a **single binary criterion**: question must be grounded in SEEN only AND answer must be faithful to what's visible in SEEN+FUTURE. Verdict is `pass` or `fail`.

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

### Step 5b — Streaming QA Generation (Oracle Generator)

The streaming generator is an **oracle**: it sees BOTH past and future content within the clip, plus the full ASR. The generator MUST, however, write the question as if only past content were known. The answer is allowed to use the full clip context as ground truth — it should describe the *real* intent or *real* next action that actually occurs in the video.

| Item | Details |
|------|---------|
| **Script** | `scripts/streaming_qa_generation.py` |
| **Input** | Video + clips JSON |
| **Output** | `results/qa/{video_id}_streaming_qa.json` |
| **Time anchors** | `[0.25, 0.5, 0.75]` of each clip's duration |
| **Per anchor** | 3 SEEN frames `[clip_start, query_time]` + 3 FUTURE frames `[query_time, clip_end]` + full ASR → 1 intent + 1 next_action |
| **Question writing rule** | Must be derivable from SEEN only (no future leakage) |
| **Answer writing rule** | May use SEEN + FUTURE; must describe what actually happens in the clip |
| **Yield** (~11 clips × 3 anchors × 2 types) | ~66 QA per video |
| **Cost/video** | ~$0.45 |

### Step 5c — Streaming QA Validation (single binary verdict)

A separate validator (different model family from generator → cross-family check) audits each streaming QA on **a single binary criterion**:

1. **Question grounding** — Is the question writable from SEEN frames only? (no future leakage)
2. **Answer faithfulness** — Does the answer correspond to what is actually visible in SEEN+FUTURE? (not a hallucination)

A QA passes only if BOTH conditions hold.

| Item | Details |
|------|---------|
| **Script** | `scripts/qa_validator.py` |
| **Validator** | `google/gemini-2.5-pro` via **OpenRouter** (OpenAI-compatible API) |
| **Input** | Streaming QA JSON + video |
| **Per QA** | 3 SEEN frames + 3 FUTURE frames → `{verdict, reason, validator_model}` |
| **Output** | `results/qa/{video_id}_streaming_qa_validated.json` |
| **Verdict** | `pass` if question grounded AND answer faithful; else `fail` |
| **Drop policy** | `fail` QA dropped by default (`--keep-failed` to retain for inspection) |
| **Cost/video** (~66 QA) | ~$0.10 |

### Step 5d — Merge to LiveCC JSONL

| Item | Details |
|------|---------|
| **Script** | `scripts/qa_merge.py` |
| **Input** | Transcript + clips + offline QA + validated streaming QA |
| **Output** | One JSONL record with `text_stream` + sorted `qa` list (LiveCC-compatible) |

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
  "validator_model": "google/gemini-2.5-pro",
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
        "validator_model": "google/gemini-2.5-pro"
      }
    }
  ]
}
```

---

## Cost & Time Estimates

| Step | Time/video | Cost/video |
|------|-----------|-----------|
| ASR Transcription | ~2 min | Free |
| Segmentation (histogram) | ~10s | Free |
| Segmentation (LLM) | ~6s | ~$0.03 |
| Offline QA (5a) | ~50s | ~$0.30 |
| Streaming QA (5b) | ~60s | ~$0.40 |
| Validation (5c) | ~80s | ~$0.30 |
| **Total per video** | **~5 min** | **~$1.05** |
| **20 videos** | **~100 min** | **~$21** |

---

## CLI Commands

```bash
# Step 1: Crawl videos
python UltrasoundCrawler_KeyCode_20260323_v2/cli.py --source youtube --max-results 100 --download-media

# Step 2: VLM Classification (batch)
python scripts/batch_filter.py --input-dir path/to/media --num-frames 8

# Step 3: ASR Transcription (batch)
python scripts/asr_pipeline.py --batch --input-dir path/to/media --model base

# Step 4: Video Segmentation (with LLM verification, default)
export OPENAI_API_KEY="sk-..."
python scripts/video_segmentation.py --video path.mp4 --transcript transcripts/ID.json

# Step 4: Histogram-only segmentation (no API)
python scripts/video_segmentation.py --video path.mp4 --transcript transcripts/ID.json --no-llm

# Step 5a: Offline QA Generation
python scripts/qa_generation.py --video path.mp4 --clips results/clips/ID_clips.json

# Step 5b: Streaming QA Generation
python scripts/streaming_qa_generation.py --video path.mp4 --clips results/clips/ID_clips.json

# Step 5c: Streaming QA Validation (Gemini 2.5 Pro via OpenRouter)
export OPENROUTER_API_KEY="sk-or-..."
python scripts/qa_validator.py \
    --streaming-qa results/qa/ID_streaming_qa.json \
    --video path.mp4

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
                          Gemini 2.5 Pro audits every streaming QA
                          → results/qa/VIDEO_ID_streaming_qa_validated.json
                                  │
                                  ▼
                          [Step 5d: Merge]
                          results/training_data/VIDEO_ID.jsonl
                                  │
                                  ▼
                          LoRA SFT on Qwen2-VL-7B
```
