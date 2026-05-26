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
├── .gitignore
│
├── scripts/                          # Executable pipeline scripts
│   ├── batch_filter.py               # Batch VLM video filtering
│   ├── asr_pipeline.py               # ASR transcription (faster-whisper)
│   └── video_filter_vlm.py           # Single-video VLM filter
│
├── notebooks/                        # Jupyter notebooks (experiments)
│   ├── ablation_input_modes.ipynb    # VLM input mode comparison
│   ├── asr_pipeline.ipynb            # ASR testing & visualization
│   ├── video_filter_vlm.ipynb        # VLM filter development
│   └── video_filter_result.ipynb     # Filter results analysis
│
├── src/                              # Source modules
│   ├── ablation_input_modes.py       # Ablation study: single/multi/video modes
│   └── video_filter.py              # Legacy CV-based filter
│
├── docs/                             # Documentation
│   ├── PROJECT_PLAN.md              # Full project plan
│   ├── DATA_PIPELINE.md            # Data processing pipeline
│   ├── AGENT_ARCHITECTURE.md       # Agentic benchmark construction
│   └── VIDEO_FILTER_DOC.md         # Video filter technical doc
│
├── transcripts/                      # ASR output (git-ignored)
├── results/                          # Experiment results (git-ignored)
│
└── UltrasoundCrawler_KeyCode_20260323_v2/  # Video crawler
    ├── cli.py                        # Command-line interface
    ├── webapp.py                     # Web UI
    └── crawler/                      # Core crawler modules
```

## Pipeline

```
1. Crawl Videos    → UltrasoundCrawler (YouTube/Bilibili)
2. VLM Filter      → Qwen2-VL/Qwen3-VL frame & video analysis
3. ASR Transcript  → faster-whisper (word-level timestamps)
4. QA Generation   → GPT-4o / Claude (TODO)
5. Model Training  → Qwen2-VL-7B + LiveCC streaming (TODO)
6. Evaluation      → UltrasoundQA Benchmark (TODO)
```

## Quick Start

### 1. Install Dependencies
```bash
pip install -r requirements.txt
```

### 2. Crawl Videos
```bash
cd UltrasoundCrawler_KeyCode_20260323_v2
python cli.py --source youtube --max-results 100 --download-media
```

### 3. VLM Filter
```bash
python scripts/batch_filter.py --max-videos 5
```

### 4. ASR Transcription
```bash
python scripts/asr_pipeline.py --video path/to/video.mp4 --model base
python scripts/asr_pipeline.py --batch --input-dir path/to/videos/
```

### 5. Run Notebooks
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
| Frame Analysis | Qwen2-VL-2B / Qwen3-VL-2B (local) |
| ASR | faster-whisper (CPU, int8) |
| QA Generation | GPT-4o / Claude API (planned) |
| Model Training | Qwen2-VL-7B + DeepSpeed (planned) |
| Evaluation | LLM-as-Judge (planned) |

## References

- **LiveCC** — Chen et al., "LiveCC: Learning Video LLM with Streaming Speech Transcription at Scale", CVPR 2025
- **Qwen2-VL** — Alibaba, Qwen2-VL Vision-Language Model
- **faster-whisper** — CTranslate2-based Whisper inference

## License

Research use only.