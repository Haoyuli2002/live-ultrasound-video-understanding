# Live Ultrasound Video Understanding

> Guided Research Project — Semester 3, TUM

## Overview

A real-time ultrasound video understanding system that goes beyond scene description to model sonographer intent, inject domain knowledge, and generate fine-grained clinical guidance.

## Core Deliverables

1. **QA Benchmark** — A structured evaluation benchmark for real-time ultrasound video understanding
2. **Real-time Video Understanding Model** — Based on Qwen2-VL-7B, capable of streaming ultrasound interpretation

## Project Structure

```
├── README.md
├── requirements.txt
├── .env.example                      # Copy to .env and fill in API keys
├── .gitignore
│
├── scripts/                          # Executable pipeline scripts
│   ├── run_pipeline.py               # End-to-end pipeline (Step 3 → 5c)
│   ├── batch_filter.py               # Step 2: Batch VLM video filtering
│   ├── video_filter_vlm.py           # Step 2: Single-video VLM filter
│   ├── asr_pipeline.py               # Step 3: ASR transcription (faster-whisper)
│   ├── video_segmentation.py         # Step 4: Histogram-based clip detection
│   ├── llm_segmentation.py           # Step 4: GPT-4o cut verification
│   ├── qa_generation.py              # Step 5a: Offline QA (scene/fine/knowledge)
│   ├── streaming_qa_generation.py    # Step 5b: Streaming QA (intent/next_action, oracle)
│   ├── qa_validator.py               # Step 5c: Validate streaming QA (Gemini 2.5 Flash)
│   ├── qa_merge.py                   # Step 5d: Merge to LiveCC-style JSONL
│   └── _env_loader.py                # Auto-load .env (used by all scripts)
│
├── notebooks/                        # Jupyter notebooks (experiments)
│   ├── ablation_input_modes.ipynb    # VLM input mode comparison
│   ├── asr_pipeline.ipynb            # ASR testing & visualization
│   ├── video_filter_vlm.ipynb        # VLM filter development
│   ├── video_filter_result.ipynb     # Filter results analysis
│   ├── video_segmentation.ipynb      # Segmentation walkthrough
│   ├── ablation_video_segmentation.ipynb
│   ├── qa_generation.ipynb           # Offline QA notebook
│   ├── streamline_qa_generation.ipynb # Streaming QA notebook
│   └── qa_pipeline_walkthrough.ipynb # End-to-end demo
│
├── src/                              # Source modules
│   ├── ablation_input_modes.py       # Ablation study: single/multi/video modes
│   └── video_filter.py               # Legacy CV-based filter
│
├── docs/                             # Documentation
│   ├── PROJECT_PLAN.md               # Full project plan
│   ├── PIPELINE.md                   # Detailed pipeline reference (PRIMARY)
│   ├── DATA_PIPELINE.md              # Earlier pipeline notes
│   ├── AGENT_ARCHITECTURE.md         # Agentic benchmark construction
│   └── VIDEO_FILTER_DOC.md           # Video filter technical doc
│
├── results/                          # Pipeline outputs (git-ignored)
│   ├── transcripts/{video_id}.json
│   ├── clips/{video_id}_clips.json
│   ├── qa/{video_id}_offline_qa.json
│   ├── qa/{video_id}_streaming_qa.json
│   ├── qa/{video_id}_streaming_qa_validated.json
│   └── training_data/{video_id}.jsonl  # LiveCC-style merged output
│
├── livecc/                           # Vendored LiveCC reference (training/eval ref)
│
└── UltrasoundCrawler_KeyCode_20260323_v2/  # Video crawler
    ├── cli.py                        # Command-line interface
    ├── webapp.py                     # Web UI
    └── crawler/                      # Core crawler modules
```

## Pipeline

```
1. Crawl Videos     → UltrasoundCrawler (YouTube/Bilibili)
2. VLM Filter       → Qwen2-VL/Qwen3-VL classify into 6 video types
3. ASR Transcript   → faster-whisper (word-level timestamps)
4. Segmentation     → histogram + GPT-4o (topic-aware clips, 30–300 s)
5a. Offline QA      → GPT-4o Vision: scene_description / fine_grained / knowledge
5b. Streaming QA    → GPT-4o Vision (oracle): sonographer_intent / next_action_guidance
5c. QA Validation   → Gemini 2.5 Flash (cross-family) drops leakage / hallucination
5d. Merge → JSONL   → LiveCC-compatible training data
6. Model Training   → Qwen2-VL-7B + LiveCC streaming (planned)
7. Evaluation       → UltrasoundQA Benchmark (planned)
```

See [`docs/PIPELINE.md`](docs/PIPELINE.md) for the full per-step spec, prompts, and cost/time breakdown.

## Prerequisites

### System dependencies
- **ffmpeg** — required by ASR (audio extraction) and Whisper. Install via:
  - macOS: `brew install ffmpeg`
  - Ubuntu: `sudo apt install ffmpeg`

### API keys
Copy `.env.example` to `.env` and fill in:
- `OPENAI_API_KEY` — used by Step 4 (LLM segmentation), Step 5a, Step 5b (GPT-4o Vision generators)
- `OPENROUTER_API_KEY` — used by Step 5c (Gemini 2.5 Flash validator via OpenRouter)

The pipeline scripts auto-load `.env` via `scripts/_env_loader.py`, so you don't need to `export` them manually.

## Quick Start

### 1. Install dependencies
```bash
pip install -r requirements.txt
```

### 2. Crawl videos
```bash
cd UltrasoundCrawler_KeyCode_20260323_v2
python cli.py --source youtube --max-results 100 --download-media
```

### 3. VLM filter (classify videos)
```bash
python scripts/batch_filter.py --max-videos 5
```

### 4. End-to-end pipeline (recommended)
Runs ASR → segmentation → offline QA → streaming QA → validation in one command:
```bash
python scripts/run_pipeline.py --video path/to/video.mp4
```
Skip steps you've already done:
```bash
python scripts/run_pipeline.py --video path.mp4 --skip-asr --skip-segmentation
```

### 5. Or run steps individually
```bash
# Step 3: ASR
python scripts/asr_pipeline.py --video path/to/video.mp4 --model base

# Step 4: Segmentation (with GPT-4o cut verification)
python scripts/video_segmentation.py --video path.mp4 --transcript results/transcripts/ID.json

# Step 5a: Offline QA
python scripts/qa_generation.py --video path.mp4 --clips results/clips/ID_clips.json

# Step 5b: Streaming QA (oracle generator)
python scripts/streaming_qa_generation.py --video path.mp4 --clips results/clips/ID_clips.json

# Step 5c: Streaming QA validation (Gemini 2.5 Flash via OpenRouter)
python scripts/qa_validator.py --streaming-qa results/qa/ID_streaming_qa.json --video path.mp4

# Step 5d: Merge into LiveCC-style JSONL
python scripts/qa_merge.py --video-id ID \
    --transcript    results/transcripts/ID.json \
    --clips         results/clips/ID_clips.json \
    --offline-qa    results/qa/ID_offline_qa.json \
    --streaming-qa  results/qa/ID_streaming_qa_validated.json \
    --out           results/training_data/ID.jsonl --overwrite
```

### 6. Explore notebooks
```bash
jupyter notebook notebooks/
```

## Video Classification Types

| Type | Description | Training Value |
|------|-------------|---------------|
| `pure_ultrasound` | Only US machine screen, no faces/slides | ⭐⭐⭐ High |
| `hands_on_tutorial` | Instructor + probe technique + US display | ⭐⭐⭐ High |
| `case_discussion` | Instructor annotating/discussing US clips | ⭐⭐ Medium |
| `ppt_lecture` | Slides/presentations with occasional US | ⭐ Low |
| `diagram_animation` | Anatomical diagrams, 3D animations | ❌ None |
| `mixed` | Multiple types alternating | Needs trimming |

## Tech Stack

| Component | Technology |
|-----------|-----------|
| Video Crawling | yt-dlp + YouTube Data API |
| Frame Analysis (Step 2) | Qwen2-VL-2B / Qwen3-VL-2B (local, MPS / CUDA) |
| ASR (Step 3) | faster-whisper (CPU, int8) |
| Segmentation (Step 4) | OpenCV grayscale histogram + GPT-4o cut verification |
| Offline QA (Step 5a) | GPT-4o Vision (6 frames + full ASR) |
| Streaming QA (Step 5b) | GPT-4o Vision oracle (3 SEEN + 3 FUTURE frames + ASR) |
| QA Validation (Step 5c) | Gemini 2.5 Flash via OpenRouter (cross-family check) |
| Model Training (Step 6) | Qwen2-VL-7B + LiveCC + DeepSpeed (planned) |
| Evaluation (Step 7) | LLM-as-Judge on UltrasoundQA Benchmark (planned) |

## References

- **LiveCC** — Chen et al., "LiveCC: Learning Video LLM with Streaming Speech Transcription at Scale", CVPR 2025
- **Qwen2-VL** — Alibaba, Qwen2-VL Vision-Language Model
- **faster-whisper** — CTranslate2-based Whisper inference

## License

Research use only.