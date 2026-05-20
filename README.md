# Live Ultrasound Video Understanding

> Guided Research Project — Semester 3, TUM

## Overview

A real-time ultrasound video understanding system that goes beyond scene description to model sonographer intent, inject domain knowledge, and generate fine-grained clinical guidance.

## Core Deliverables

1. **QA Benchmark** — A structured evaluation benchmark for real-time ultrasound video understanding
2. **Real-time Video Understanding Model** — Based on Qwen2-VL-7B, capable of streaming ultrasound interpretation

## Project Structure

```
├── docs/                           # Project documentation
│   ├── PROJECT_PLAN.md            # Full project plan
│   ├── DATA_PIPELINE.md           # Data processing pipeline
│   ├── AGENT_ARCHITECTURE.md      # Agentic benchmark construction
│   └── VIDEO_FILTER_DOC.md        # Video filter technical doc
│
├── video_filter_vlm.py            # VLM-based video filter (Qwen2-VL-2B)
├── batch_filter.py                # Batch filtering with progress save
├── video_filter.py                # Legacy CV-based filter
│
├── video_filter_vlm.ipynb         # VLM filter testing notebook
├── video_filter_result.ipynb      # Filter results analysis & visualization
│
├── UltrasoundCrawler_KeyCode_20260323_v2/  # YouTube/Bilibili crawler
│   ├── cli.py                     # Command-line interface
│   ├── webapp.py                  # Web UI
│   └── crawler/                   # Core crawler modules
│
├── livecc/                        # LiveCC reference codebase (CVPR 2025)
│
└── Papers/                        # Reference papers (git-ignored)
```

## Pipeline

```
1. Crawl Videos    → UltrasoundCrawler (YouTube/Bilibili)
2. VLM Filter      → Qwen2-VL-2B frame analysis + rule-based decision
3. ASR Transcript  → WhisperX (TODO)
4. QA Generation   → GPT-4o / Claude (TODO)
5. Model Training  → Qwen2-VL-7B + LiveCC streaming (TODO)
6. Evaluation      → UltrasoundQA Benchmark (TODO)
```

## Quick Start

### 1. Crawl Videos
```bash
cd UltrasoundCrawler_KeyCode_20260323_v2
pip install -r requirements.txt
python cli.py --source youtube --max-results 100 --download-media
```

### 2. VLM Filter (requires ~4GB for Qwen2-VL-2B model)
```bash
pip install torch transformers qwen-vl-utils opencv-python tqdm
python batch_filter.py --max-videos 5  # test with 5
python batch_filter.py                  # run all
```

### 3. View Results
Open `video_filter_result.ipynb` in Jupyter.

## Key Innovation: VLM-based Video Filtering

Instead of traditional CV heuristics, we use **Qwen2-VL-2B** to analyze each frame:
- Correctly identifies ultrasound screens vs. lecture slides vs. talking heads
- Few-shot prompted for structured JSON output
- Rule-based aggregation for video-level decisions
- Incremental save + resume support for long-running batch jobs

## Tech Stack

| Component | Technology |
|-----------|-----------|
| Video Crawling | yt-dlp + YouTube Data API |
| Frame Analysis | Qwen2-VL-2B (local, MPS/CUDA) |
| ASR | WhisperX (planned) |
| QA Generation | GPT-4o / Claude API (planned) |
| Model Training | Qwen2-VL-7B + DeepSpeed (planned) |
| Evaluation | LLM-as-Judge (planned) |

## References

- **LiveCC** — Chen et al., "LiveCC: Learning Video LLM with Streaming Speech Transcription at Scale", CVPR 2025
- **Qwen2-VL** — Alibaba, Qwen2-VL Vision-Language Model
- **WhisperX** — Large-scale ASR with word-level timestamps

## License

Research use only.