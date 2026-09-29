# How to Run the Live Ultrasound Pipeline

> This document is the command cookbook for the pipeline. The authoritative
> design and data semantics are defined in
> `docs/live_ultrasound_reference_implementation_final_zh.md`.

---

## 1. VLM Video Classification

### Qwen3.5 teacher labeling

```bash
python scripts/data/classify_ultrasound_video_type_teacher.py \
  --video-map cluster_data/splits/train_full295_asr_keep_videos.json \
  --output cluster_data/splits/train_full295_qwen35_video_type.jsonl \
  --teacher qwen35 \
  --model Qwen/Qwen3.5-35B-A3B \
  --base-url http://localhost:8000/v1 \
  --api-key-env VLLM_API_KEY \
  --video-fps 1.0 \
  --resume
```

Recommended stable batch settings from smoke tests:

```text
ENABLE_THINKING=0
VIDEO_FPS=0.5
MAX_TOKENS=3000
```

### Optional cross-validation labeling

```bash
python scripts/data/classify_ultrasound_video_type_teacher.py \
  --video-map cluster_data/splits/train_full295_asr_keep_videos.json \
  --output cluster_data/splits/train_full295_gemini3_video_type.jsonl \
  --teacher gemini3 \
  --model google/gemini-3.1-pro-preview \
  --base-url https://openrouter.ai/api/v1 \
  --video-fps 1.0 \
  --resume
```

### Merge teacher labels

```bash
python scripts/data/merge_video_type_teacher_labels.py \
  --primary-audit cluster_data/splits/train_full295_qwen35_video_type.jsonl \
  --validator-audit cluster_data/splits/train_full295_gemini3_video_type.jsonl \
  --output cluster_data/splits/train_full295_video_type_final.jsonl \
  --output-summary cluster_data/splits/train_full295_video_type_final_summary.json
```

### Filter by stage keep flags

```bash
python scripts/data/filter_by_vlm_video_type.py \
  --video-map cluster_data/splits/train_full295_asr_keep_videos.json \
  --vlm-audit cluster_data/splits/train_full295_video_type_final.jsonl \
  --stage compression \
  --output-keep-map cluster_data/splits/train_full295_compression_keep_videos.json \
  --output-drop-map cluster_data/splits/train_full295_compression_drop_videos.json
```

---

## 2. Qwen3.5 vLLM Serving

```bash
uv pip install vllm \
  --torch-backend=auto \
  --extra-index-url https://wheels.vllm.ai/nightly

vllm serve Qwen/Qwen3.5-35B-A3B \
  --host 0.0.0.0 \
  --port 8000 \
  --trust-remote-code
```

Serve local videos when using `video_url`:

```bash
cd /path/to/video/root
python -m http.server 9000
```

Slurm entry point:

```bash
sbatch \
  --export=ALL,SPLIT=eval_full295,VIDEO_MAP=/dss/mcmlscratch/04/ge75vid2/haoyu/live-ultrasound-video-understanding/cluster_data/splits/eval_full295_asr_keep_videos.json,OUTPUT=/dss/mcmlscratch/04/ge75vid2/haoyu/live-ultrasound-video-understanding/cluster_data/splits/eval_full295_qwen35_video_type.jsonl \
  scripts/slurm/run_qwen35_video_type_labeling.sbatch
```

---

## 3. Stage 1 Domain Pretraining

Build Stage 1 samples:

```bash
python pretrain/build_samples.py \
  --transcripts results/transcripts \
  --output pretrain/data/pretrain_samples.jsonl \
  --unit sentence \
  --window-sec 8
```

Train Stage 1:

```bash
python pretrain/train.py \
  --model-name Qwen/Qwen3-VL-2B-Instruct \
  --train-jsonl pretrain/data/pretrain_samples.jsonl \
  --video-path-map pretrain/data/video_path_map.json \
  --output-dir /mnt/cache/qwenFT/qwen3vl_stage1_pretrain \
  --window-size 4 \
  --frame-size 224 \
  --num-train-epochs 3 \
  --per-device-train-batch-size 1 \
  --gradient-accumulation-steps 8 \
  --learning-rate 1e-4
```

---

## 4. Stage 2 VLM Summary Label Generation

### Dense local labels plus cumulative global labels

Use muted video-only clips by default. This avoids audio leakage into visual
summary labels.

```bash
python scripts/data/generate_teacher_visual_summaries.py \
  --video /path/to/video.mp4 \
  --output results/teacher_visual_summaries/VIDEO_ID_teacher_visual_summaries_gemini31pro_muted_local10s.jsonl \
  --video-id VIDEO_ID \
  --model google/gemini-3.1-pro-preview \
  --block-sec 60 \
  --local-sec 10 \
  --global-mode incremental \
  --max-local-tokens 1200 \
  --max-global-tokens 3000 \
  --overwrite
```

For each 60-second block this produces six local 10-second summary labels and
one cumulative global summary label.

### Anchor example already committed

```text
results/teacher_visual_summaries/8V649L5Q368_teacher_visual_summaries_gemini31pro_muted.jsonl
results/teacher_visual_summaries/8V649L5Q368_teacher_memory_samples_gemini31pro_muted.jsonl
```

---

## 5. Build Stage 2 Summary-Compression Samples

```bash
python pretrain/build_teacher_memory_summary_samples.py \
  --summaries-jsonl results/teacher_visual_summaries/VIDEO_ID_teacher_visual_summaries_gemini31pro_muted_local10s.jsonl \
  --output results/teacher_visual_summaries/VIDEO_ID_teacher_memory_samples_gemini31pro_muted_local10s.jsonl
```

For an 18-block video with `--local-sec 10`, expected output is:

```text
108 short_memory_summary
18 long_memory_summary
126 total samples
```

---

## 6. Stage 2 Training / Inference / Evaluation

Build legacy ASR-reconstruction baseline samples if needed:

```bash
python pretrain/build_memory_compression_samples.py \
  --output pretrain/data/memory_compression_samples.jsonl \
  --block-sec 60 \
  --step-sec 1
```

Train:

```bash
python pretrain/train_memory_compression.py \
  --model-name Qwen/Qwen3-VL-2B-Instruct \
  --train-jsonl pretrain/data/memory_compression_samples.jsonl \
  --video-path-map pretrain/data/video_path_map.json \
  --output-dir /mnt/cache/qwenFT/qwen3vl_memory_compression \
  --short-frames 2 \
  --frame-size 224 \
  --lambda-short 1.0 \
  --lambda-long 1.0 \
  --num-train-epochs 1 \
  --learning-rate 1e-4
```

Infer:

```bash
python pretrain/infer_memory_compression.py \
  --model-name Qwen/Qwen3-VL-2B-Instruct \
  --adapter-path /mnt/cache/qwenFT/qwen3vl_memory_compression \
  --eval-jsonl pretrain/data/memory_compression_samples.jsonl \
  --output results/memory_compression_predictions.jsonl \
  --video-path-map pretrain/data/video_path_map.json \
  --short-frames 2 \
  --frame-size 224
```

Evaluate:

```bash
python pretrain/eval_memory_compression.py \
  --pred-jsonl results/memory_compression_predictions.jsonl \
  --output results/memory_compression_eval.json
```

---

## 7. Useful Checks

```bash
python -m py_compile \
  scripts/data/generate_teacher_visual_summaries.py \
  pretrain/build_teacher_memory_summary_samples.py \
  pretrain/memory_dataset.py \
  pretrain/memory_collator.py \
  pretrain/train_memory_compression.py \
  pretrain/infer_memory_compression.py \
  pretrain/eval_memory_compression.py

git diff --check
```