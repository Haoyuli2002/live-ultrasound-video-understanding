# Live Ultrasound Video Understanding

> Guided Research Project — Semester 3

---

## 1. Project Overview

### Objective

Build a **real-time ultrasound video understanding system** that goes beyond scene description to model sonographer intent, inject domain knowledge, and generate fine-grained clinical guidance.

### Two Core Deliverables

1. **QA Benchmark** — A structured evaluation benchmark for real-time ultrasound video understanding
2. **Real-time Video Understanding Model** — Based on Qwen2-VL-7B, capable of streaming ultrasound interpretation

### Key Innovations (vs. existing Video LLMs)

| Existing Models | Our System |
|----------------|------------|
| Describe what is visible | + Understand sonographer's intent |
| Scene-level answers only | + Inject prior medical knowledge |
| Coarse descriptions | + Fine-grained attributes (location, shape, echogenicity) |
| Offline video QA | + Real-time streaming understanding |

---

## 2. System Architecture

```
┌─────────────────────────────────────────────────────────────┐
│              Real-time Ultrasound Video Stream               │
│                  (2 fps streaming input)                     │
└────────────────────────┬────────────────────────────────────┘
                         │
                         ▼
┌─────────────────────────────────────────────────────────────┐
│                  Visual Encoder                              │
│              Qwen2-VL Vision Transformer                     │
│                                                             │
│  Input:  Streaming frame sequence (new frames appended)     │
│  Output: Visual token sequence                              │
└────────────────────────┬────────────────────────────────────┘
                         │
                         ▼
┌─────────────────────────────────────────────────────────────┐
│                  LLM Backbone                                │
│                  Qwen2-7B                                    │
│                                                             │
│  Additional inputs:                                         │
│  • Prior knowledge (anatomy, scanning protocols)            │
│  • Historical context (previous frame descriptions)         │
│  • User query (optional)                                    │
└────────────────────────┬────────────────────────────────────┘
                         │
                         ▼
┌─────────────────────────────────────────────────────────────┐
│                  Multi-type Output                           │
│                                                             │
│  L1: Scene Description                                      │
│      "Right kidney in longitudinal view, normal cortex"     │
│                                                             │
│  L2: Sonographer Intent                                     │
│      "Searching for signs of hydronephrosis"                │
│                                                             │
│  L3: Action Guidance                                        │
│      "Tilt probe cranially to visualize upper pole"         │
│                                                             │
│  L4: Fine-grained Attributes                               │
│      "Kidney 11cm length, cortex 1.2cm, no focal lesions"  │
└─────────────────────────────────────────────────────────────┘
```

---

## 3. Streaming Inference Design (Reference: LiveCC)

LiveCC introduces streaming video input where frames are fed incrementally:

```
Time=0.0-3.0s: [Initial 6 frames] + Query  →  Model starts generating
Time=3.0-4.0s: [+2 new frames]             →  Model continues (with context)
Time=4.0-5.0s: [+2 new frames]             →  Model continues
...
```

**Adaptation for Ultrasound:**
- Initial window: 3 seconds (6 frames @ 2fps) for scene establishment
- Streaming window: 0.5-1.0 second per update
- Context carries forward: previous predictions inform current output

---

## 4. Three-Layer Understanding

| Layer | Capability | Data Source | Training Signal |
|-------|-----------|-------------|-----------------|
| **L1: Scene Perception** | Identify anatomical structures in current view | Video frames | ASR narration describing visible structures |
| **L2: Intent Modeling** | Understand what the operator is doing/seeking | Frame sequence dynamics | ASR self-narration ("now I'm looking for...") |
| **L3: Knowledge Reasoning** | Provide professional judgment beyond what's visible | Prior knowledge injection | Medical textbook knowledge + scanning protocols |

### Intent Modeling (Key Innovation)

Inferring operator intent from probe motion patterns:

| Probe Motion Pattern | Inferred Intent |
|---------------------|-----------------|
| Continuous sliding | Searching for a target structure |
| Stationary / stable | Observing or measuring |
| Rotation in place | Adjusting imaging plane angle |
| Increasing pressure | Compression test (e.g., DVT assessment) |
| Fan-like sweep | Systematic survey of a region |

**Ground truth source:** ASR narration where sonographers verbalize their intentions.

---

## 5. Prior Knowledge Injection

### Strategy: Hybrid Approach (Recommended)

| Method | Use Case | Implementation |
|--------|----------|----------------|
| **Training-time injection** | Common knowledge (standard views, normal values) | Mix medical textbook data in SFT |
| **Inference-time RAG** | Rare/complex cases | Retrieve from knowledge base based on detected anatomy |

### Knowledge Categories

1. **Anatomy Atlas** — Normal appearance, standard measurements, variants
2. **Scanning Protocols** — Step-by-step probe positioning for each exam
3. **Pathology Reference** — Abnormal findings and their significance
4. **Normal Values** — Age/gender-specific measurement ranges

---

## 6. QA Benchmark Design

### Data Format (LiveCC-compatible JSONL)

```json
{
  "video": "clips/kidney_scan_001.mp4",
  "text_stream": [
    [0.0, 2.1, "Here we can see the right kidney"],
    [2.1, 4.5, "in the longitudinal plane"],
    [4.5, 7.0, "The cortex appears normal"]
  ],
  "qa": [
    {
      "timestamp": 2.0,
      "type": "scene_description",
      "question": "What anatomical structure is currently visible?",
      "answer": "The right kidney is visible in longitudinal section, showing normal cortical echogenicity and corticomedullary differentiation."
    },
    {
      "timestamp": 2.0,
      "type": "action_guidance",
      "question": "How can I better visualize the upper pole?",
      "answer": "Tilt the probe cranially and ask the patient to take a deep breath to displace the kidney inferiorly."
    },
    {
      "timestamp": 5.0,
      "type": "sonographer_intent",
      "question": "What is the sonographer trying to assess?",
      "answer": "The sonographer is systematically evaluating renal morphology, looking for signs of hydronephrosis or cortical thinning."
    },
    {
      "timestamp": 5.0,
      "type": "fine_grained",
      "question": "Describe the findings with specific measurements and locations.",
      "answer": "The kidney measures approximately 11cm in length. The cortex is 1.2cm thick at the mid-pole. No focal lesions or calculi are seen. The renal pelvis is not dilated."
    }
  ]
}
```

### QA Types

| Type | Code | Example Question | Evaluation Metric |
|------|------|-----------------|-------------------|
| Basic Knowledge | `basic_knowledge` | "What frequency probe is best for renal scanning?" | Accuracy |
| Scene Description | `scene_description` | "What is visible in this frame?" | LLM-as-Judge |
| Action Guidance | `action_guidance` | "What should I do to see the IVC?" | LLM-as-Judge |
| Sonographer Intent | `sonographer_intent` | "What is the operator trying to do?" | LLM-as-Judge |
| Fine-grained Attributes | `fine_grained` | "Describe location, size, echogenicity." | Attribute F1 |

---

## 7. Data Pipeline

```
Raw Videos (YouTube / Bilibili)
    │
    ├── [Step 1] Crawl & Download (UltrasoundCrawler)
    │
    ├── [Step 2] Purity Filter (video_filter.py)
    │       → Remove PPT, talking heads, heavy annotations
    │
    ├── [Step 3] ASR Transcription (WhisperX)
    │       → Time-aligned word-level transcription
    │
    ├── [Step 4] Video Segmentation
    │       → Split into 2-30s clips by ASR semantics / silence
    │
    ├── [Step 5] Inpainting (Optional)
    │       → Remove on-screen text annotations
    │       → Preserve removed annotations as pseudo-labels
    │
    ├── [Step 6] QA Generation (GPT-4o / Claude Agent)
    │       → Input: video clip + ASR + chapter info
    │       → Output: multi-type QA pairs
    │
    ├── [Step 7] Human Review (small subset)
    │       → Quality check on auto-generated QA
    │
    └── [Step 8] Format to LiveCC JSONL + Seek Index
            → Ready for training / evaluation
```

### AI Agent for QA Generation (Reducing Human Labor)

```python
# Prompt template for GPT-4o Vision
SYSTEM_PROMPT = """You are a senior ultrasound instructor. Given a video clip 
and its ASR transcription, generate QA pairs from these perspectives:

1. SCENE: What anatomical structures are visible? Describe objectively.
2. INTENT: What is the sonographer trying to do/find?
3. GUIDANCE: What should a learner do to achieve a better view?
4. KNOWLEDGE: What prior medical knowledge is relevant here?
5. FINE-GRAINED: Describe specific attributes (location, size, echogenicity).

For each QA pair, specify the relevant timestamp."""
```

---

## 8. Inpainting Strategy

### Goal
Remove artificial annotations (arrows, measurement lines, text labels) from ultrasound frames while preserving the removed content as pseudo ground-truth labels.

### Approach

1. **Detect annotation regions** — Use edge detection + color segmentation to identify non-ultrasound overlays
2. **Inpaint** — Use video inpainting (e.g., ProPainter, E2FGVI) to fill removed regions
3. **Store pseudo-labels** — Save the original annotations as structured data:
   ```json
   {
     "frame_idx": 45,
     "annotations": [
       {"type": "text", "content": "RK 11.2cm", "bbox": [100, 50, 250, 70]},
       {"type": "arrow", "points": [[120, 80], [180, 120]]},
       {"type": "caliper", "measurement": "11.2cm", "endpoints": [[100, 90], [280, 90]]}
     ]
   }
   ```

### Use Cases
- **Training**: Model learns to predict what annotations *would* say without seeing them
- **Evaluation**: Compare model's descriptions against pseudo-labels

---

## 9. Model Training Plan

### Base Model: Qwen2-VL-7B

### Stage 1: Pre-training (Optional)
- Dataset: Large-scale ultrasound video + ASR pairs
- Objective: Align visual encoder to ultrasound domain
- Config: Freeze visual encoder, train LLM + projector

### Stage 2: SFT (Supervised Fine-Tuning)
- Dataset: Curated QA benchmark + medical knowledge
- Multi-task training across all QA types
- Include streaming format (LiveCC-style `text_stream`)

### Training Configuration (Reference)
```bash
export VIDEO_MIN_PIXELS=78400      # Min 100 visual tokens per frame
export FPS_MAX_FRAMES=480          # Max ~4 min video
export VIDEO_MAX_PIXELS=19267584   # Max 24k visual tokens total

learning_rate=1e-5
torchrun --nproc_per_node=8 train.py \
  --deepspeed zero2.json \
  --pretrained_model_name_or_path Qwen/Qwen2-VL-7B \
  --annotation_paths ultrasound_qa_benchmark.jsonl \
  --freeze_modules visual \
  --num_train_epochs 3 \
  --per_device_train_batch_size 1 \
  --gradient_accumulation_steps 64 \
  --bf16 True
```

---

## 10. Evaluation Plan

### Benchmarks

| Benchmark | What it measures | Metrics |
|-----------|-----------------|---------|
| **UltrasoundQA (Ours)** | Domain-specific real-time understanding | LLM-Judge win rate, Attribute F1 |
| **VideoMME** | General video understanding | Accuracy |
| **OVOBench** | Online video understanding | Accuracy |

### Baselines for Comparison

1. **Qwen2-VL-7B** (zero-shot) — No ultrasound training
2. **GPT-4o Vision** — Strong but non-streaming
3. **LiveCC-7B-Instruct** — Streaming but not medical
4. **Ours** — Streaming + medical + intent modeling

### LLM-as-Judge Protocol

Following LiveCC's evaluation approach:
```python
# GPT-4o judges which answer is better
judge_prompt = """Compare these two answers about an ultrasound video.
Ground truth: {reference}
Answer A: {model_a_output}  
Answer B: {model_b_output}
Which answer is more accurate, clinically relevant, and detailed?"""
```

---

## 11. Implementation Roadmap

### Phase 1: Data Collection & Processing (Weeks 1-6)
- [ ] Scale up video crawling (target: 500+ pure ultrasound videos)
- [ ] Run ASR transcription on filtered videos
- [ ] Segment videos into clips by semantic boundaries
- [ ] Establish quality filtering criteria

### Phase 2: Benchmark Construction (Weeks 4-8)
- [ ] Finalize QA type taxonomy and templates
- [ ] Build AI agent for automated QA generation
- [ ] Generate QA pairs for all clips
- [ ] Human review ~200-500 samples for gold standard
- [ ] Design evaluation metrics and protocol

### Phase 3: Model Training (Weeks 6-12)
- [ ] Set up GPU training environment
- [ ] Prepare data in LiveCC JSONL format
- [ ] Train Qwen2-VL-7B with streaming ultrasound data
- [ ] Ablation: ± prior knowledge, ± intent modeling

### Phase 4: Evaluation & Paper (Weeks 10-14)
- [ ] Evaluate on constructed benchmark
- [ ] Compare against baselines
- [ ] Case study analysis
- [ ] Write paper

---

## 12. Technical Requirements

### Compute
- **Training**: 8× A100 80GB (or equivalent) — ~40 hours for SFT
- **Inference**: 1× A100 or 2× RTX 4090
- **Data processing**: CPU-only (MacBook Pro sufficient for pipeline)

### Software Stack
- **Model**: Qwen2-VL-7B, DeepSpeed ZeRO-2, Flash Attention
- **ASR**: WhisperX (large-v3-turbo)
- **Data processing**: OpenCV, FFmpeg
- **QA Generation**: GPT-4o / Claude API
- **Evaluation**: LLM-as-Judge (GPT-4o)
- **Inpainting**: ProPainter / LaMa (optional)

### Data Organization (LiveCC-style)
```
ultrasound_benchmark/
├── videos/                    # Raw video clips
│   ├── kidney_001.mp4
│   └── ...
├── annotations/
│   ├── train.jsonl           # Training QA data
│   ├── val.jsonl             # Validation set
│   └── test.jsonl            # Test benchmark
├── knowledge_base/           # Prior knowledge for RAG
│   ├── anatomy.json          # Anatomical structures reference
│   ├── protocols.json        # Scanning protocols
│   └── normal_values.json    # Normal measurement ranges
├── inpainted/                 # Videos with annotations removed
│   ├── kidney_001_clean.mp4
│   └── ...
└── pseudo_labels/             # Extracted annotations as ground truth
    ├── kidney_001_labels.json
    └── ...
```

---

## 13. References

- **LiveCC** — Chen et al., "LiveCC: Learning Video LLM with Streaming Speech Transcription at Scale", CVPR 2025
- **Qwen2-VL** — Alibaba, Qwen2-VL-7B Vision-Language Model
- **WhisperX** — Large-scale ASR with word-level timestamps
- **ProPainter** — Video inpainting for annotation removal

---

## 14. Current Progress

### Phase 1 — Data Collection & Processing
| Item | Status |
|------|--------|
| Project environment setup | ✅ Done |
| UltrasoundCrawler working | ✅ Done |
| Video purity filter (CV-based, `src/video_filter.py`) | ✅ Done |
| **VLM-based 6-class video classifier** (`scripts/batch_filter.py`) | ✅ Done |
| Initial video collection (demo: 11 videos, 5 kept) | ✅ Done |
| ASR transcription pipeline (faster-whisper) | ✅ Done |
| Video segmentation: histogram + GPT-4o (two-stage) | ✅ Done |
| Scale up crawling to 100+ pure videos | 🟡 In progress |

### Phase 2 — Benchmark Construction
| Item | Status |
|------|--------|
| LiveCC codebase analyzed | ✅ Done |
| Project plan document | ✅ Done |
| **Dual-track QA design** (offline 3 types + streaming 2 types) | ✅ Done |
| Offline QA generator (`qa_generation.py`, GPT-4o) | ✅ Done |
| Streaming QA generator (`streaming_qa_generation.py`, GPT-4o) | ✅ Done |
| **Streaming QA validator** (`qa_validator.py`, Gemini 2.5 Flash via OpenRouter) | ✅ Done |
| LiveCC-style JSONL merger (`qa_merge.py`) | ✅ Done |
| End-to-end pipeline runner (`run_pipeline.py`) | ✅ Done |
| Pipeline documentation (`docs/PIPELINE.md`) | ✅ Done |
| Demo single-video run (8V649L5Q368) — old 5-types-in-one | ✅ Done (legacy) |
| Demo single-video run with new dual-track design | 🟡 Next |
| Scale-up to 20–30 videos benchmark v0 | 🔲 Pending |
| Human review of ~200–500 gold QA | 🔲 Pending |
| Knowledge base (anatomy / protocols / normal values) for RAG | 🔲 Pending |
| Inpainting of on-screen annotations (optional) | 🔲 Pending |

### Phase 3 — Model Training
| Item | Status |
|------|--------|
| GPU training environment | 🔲 Pending |
| LiveCC seek-index format prep | 🔲 Pending |
| Qwen2-VL-7B SFT on ultrasound dataset | 🔲 Pending |
| Ablation: ± prior knowledge, ± intent modeling | 🔲 Pending |

### Phase 4 — Evaluation & Paper
| Item | Status |
|------|--------|
| UltrasoundQA benchmark evaluation | 🔲 Pending |
| Comparison vs Qwen2-VL / GPT-4o / LiveCC-7B baselines | 🔲 Pending |
| VideoMME / OVOBench cross-domain eval | 🔲 Pending |
| Paper writing | 🔲 Pending |

### Where we are right now
- **Phase 1 essentially complete** for the demo cohort; scaling-up the crawler is the only remaining item.
- **Phase 2 mid-stage**: full QA pipeline (generators + validator + merger + end-to-end runner) is implemented and documented. Next step is to actually run it on the demo video with the new design and on 20–30 videos for benchmark v0.
- Phases 3 & 4 not yet started.
