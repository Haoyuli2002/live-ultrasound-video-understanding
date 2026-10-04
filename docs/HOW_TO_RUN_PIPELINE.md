# How to Run the Live Ultrasound Pipeline

> This document is the command cookbook for the pipeline. The authoritative
> design and data semantics are defined in
> `docs/live_ultrasound_reference_implementation_final_zh.md`.

---

## Stage 2: chronological 1-FPS implementation

The default trainer now consumes complete chronological teacher block rows, not
shuffled short/long examples. It uses shared Qwen + LoRA, one frame/second,
60 long tokens, and detached state carry. Every ten seconds it performs one
optimizer update; at minute boundaries it sums short (10s), local (60s), and
global (prefix) losses with weights 1:1:1. Old short states are not recomputed.

Generate labels with independent 10-second, 60-second, and prefix targets:

```bash
python -m stage2.annotate \
  --video /path/to/VIDEO_ID.mp4 \
  --video-id VIDEO_ID --model google/gemini-3.1-pro-preview \
  --base-url https://openrouter.ai/api/v1 \
  --api-key-env OPENROUTER_API_KEY \
  --output results/teacher_visual_summaries/VIDEO_ID_streaming_labels.jsonl

python -m stage2.build_data \
  --teacher-jsonl results/teacher_visual_summaries/VIDEO_ID_streaming_labels.jsonl \
  --output results/teacher_visual_summaries/VIDEO_ID_streaming_blocks.jsonl

python -m stage2.train \
  --train-jsonl results/teacher_visual_summaries/VIDEO_ID_streaming_blocks.jsonl \
  --video-path-map /path/to/video_path_map.json \
  --output-dir /path/to/stage2_output \
  --frame-size 224 --epochs 1 --learning-rate 1e-4

python -m stage2.stream --video /path/to/VIDEO_ID.mp4 \
  --adapter-path /path/to/stage2_output/final \
  --output /path/to/memory_state.pt
```

Rows require `video_id`, `block_window`, six `local_sub_summaries` (each with
`window` and `local_summary_target`), independent `block_summary_target`, and
`global_summary_target`. Final incomplete blocks may contain fewer complete
10-second windows; these train only short losses. Less than ten remaining seconds
are excluded from offline supervision. Gaps, duplicates, and missing labels fail
validation. Generate a new label file for old concatenated-summary artifacts.
`stage2.annotate` sends silent 10-second, 60-second, and cumulative `[0,T)`
video clips directly. It does not select a fixed number of image frames; the
teacher backend still samples video internally. Review labels for missed brief
findings and check provider limits as global clips grow. FPS=1 specifies the
student model input separately.
The teacher annotation JSONL has eight separate records per full minute `T`:
six ten-second intervals within `[T-60,T)`, one block summary for `[T-60,T)`,
and one cumulative summary for `[0,T)`. `stage2.build_data` merges these into
the block rows described above.

An optional `--init-adapter` initializes Stage 2 from Stage 1. Pass the new
Stage 2 `final` adapter to `stage2.stream`. Per-update component losses are
saved in `training_metrics.jsonl`. The older `pretrain/` examples below remain
historical references and use different command-line options.

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

On the cluster, retain the original train/eval split and select only videos
marked `keep_for_pretrain` by the Qwen3.5 audit. Inspect and retry the six
reported error rows before treating the selected count as final. The overall
234-video count includes both splits and is not the train-set size.

Build filtered maps and paired Stage 1 samples:

```bash
for split in train_full295 eval_full295; do
  python scripts/data/filter_by_vlm_video_type.py \
    --video-map cluster_data/splits/${split}_asr_keep_videos.json \
    --vlm-audit cluster_data/splits/${split}_qwen35_video_type.jsonl \
    --stage pretrain \
    --output-keep-map cluster_data/splits/${split}_qwen35_pretrain_keep_videos.json \
    --output-drop-map cluster_data/splits/${split}_qwen35_pretrain_drop_videos.json \
    --output-summary cluster_data/splits/${split}_qwen35_pretrain_filter_summary.json

  python -m stage1.data \
    --transcripts cluster_data/QA/${split}/transcripts \
    --video-map cluster_data/splits/${split}_qwen35_pretrain_keep_videos.json \
    --output cluster_data/pretrain/${split}_stage1_samples.jsonl
done
```

The builder stops if a selected transcript is missing or its `video_id`
disagrees with its filename. It reports selected videos, videos yielding no
complete sentence, paired sentences, and rows. Each sentence yields three
rows. Review a sample of interpolated sentence timestamps before training.
If the raw Qwen audit includes uncertain or error rows with a keep flag,
inspect those decisions explicitly; do not infer membership from the aggregate
237/234 counts.

Seven train videos currently produce no samples because their ASR text has no
sentence-ending punctuation. Clean all selected ASR transcripts with the local
Qwen3.5-27B vLLM endpoint, then build a **new** Stage 1 JSONL without
overwriting the baseline:

```bash
for split in train_full295 eval_full295; do
  python -m stage1.clean_asr \
    --transcripts cluster_data/QA/${split}/transcripts \
    --video-map cluster_data/splits/${split}_qwen35_pretrain_keep_videos.json \
    --output-dir cluster_data/QA/${split}/transcripts_stage1_qwen35_clean \
    --audit-output cluster_data/QA/${split}/stage1_qwen35_clean_audit.jsonl \
    --model Qwen/Qwen3.5-27B \
    --base-url http://localhost:8000/v1 \
    --api-key-env VLLM_API_KEY \
    --resume

  python -m stage1.data \
    --transcripts cluster_data/QA/${split}/transcripts_stage1_qwen35_clean \
    --video-map cluster_data/splits/${split}_qwen35_pretrain_keep_videos.json \
    --output cluster_data/pretrain/${split}_stage1_qwen35_clean_samples.jsonl
done
```

The endpoint must actually serve `Qwen/Qwen3.5-27B`; the older video-type
labeling service may serve a different model. Set `VLLM_API_KEY` to the key
expected by your local server. The cleaner processes every selected transcript,
including already-punctuated ASR. It preserves the source text under
`raw_segments` and `raw_full_text` (when present), writes cleaned `segments`
and `full_text`, keeps each segment's timestamps, and logs term edits in
`asr_cleaning.term_corrections` and the JSONL audit. It sends batches of eight
segments with two neighboring segments on either side as text-only context;
video and audio are not sent. This is LLM-assisted data cleaning and light
polishing, not teacher labeling. The model returns one `clean_text` and a
`from`/`to`/`reason` correction list per segment. Missing/reordered segments,
undeclared word changes, and large rewrites cause a validation error before a
transcript is written. `stage1.data` then extracts complete sentences from
cleaned segments and builds three paired rows per eligible target sentence.
The model sees ASR text only, so terminology corrections are unverified
hypotheses; review the audit and a sample of text/time boundaries before
training. Sentence times are estimated within ASR segments, not forced word
alignment. Use the new `*_qwen35_clean_samples.jsonl` path for training after
that review.

On LRZ, `scripts/slurm/run_stage1_qwen35_asr_clean.sbatch` starts its own
Qwen3.5-27B vLLM server, cleans one split, and builds its Stage 1 JSONL.
Submit train and eval separately from the repository root after creating the
Slurm log directory:

```bash
mkdir -p logs
sbatch --export=ALL,SPLIT=train_full295 scripts/slurm/run_stage1_qwen35_asr_clean.sbatch
sbatch --export=ALL,SPLIT=eval_full295 scripts/slurm/run_stage1_qwen35_asr_clean.sbatch
```

The script defaults to the same `REPO` and scratch `DATA` paths as the
video-type labeling sbatch. Override them with `--export` if your checkout or
data path differs. A rerun skips already-written cleaned transcripts and then
rebuilds the sample JSONL; it does not call the cleaner again for those videos.

For the initial seven-video recovery, use the dedicated job instead of
cleaning all 193 selected transcripts. It requires the existing
`train_full295_stage1_samples.jsonl` baseline (186 videos), builds a map for
the seven previously empty videos, cleans only those seven, and creates
`train_full295_stage1_merged_193_samples.jsonl` after validating exact
193-video coverage:

```bash
mkdir -p logs
sbatch scripts/slurm/run_stage1_seven_video_pilot.sbatch
```

Inspect `logs/stage1_seven_asr_<jobid>.out`, the seven-video audit under
`cluster_data/QA/train_full295/`, and the merge summary before training.
The full Stage 1 GPU job has a separate launcher. A 20-step smoke test uses a
different run name and a smaller frame budget; the full job then trains from
the merged data with its own checkpoint directory:

```bash
sbatch --export=ALL,RUN_NAME=stage1_smoke,MAX_STEPS=20,FRAME_BUDGET=24,SAVE_STEPS=10 \
  scripts/slurm/run_stage1_pretrain_merged.sbatch
sbatch scripts/slurm/run_stage1_pretrain_merged.sbatch
```

The full launcher uses 120 frames and three epochs, saves checkpoints every
50 optimizer steps, and passes `--resume`. Resubmit after a walltime limit
to continue from the latest checkpoint. The launcher checks that the merged
sample video IDs exactly match the 193-video pretrain keep-map; it does not
include the separate eval split.

Train Stage 1:

```bash
python -m stage1.train \
  --model-name Qwen/Qwen3-VL-2B-Instruct \
  --train-jsonl cluster_data/pretrain/train_full295_stage1_merged_193_samples.jsonl \
  --video-path-map cluster_data/splits/train_full295_qwen35_pretrain_keep_videos.json \
  --output-dir /path/to/stage1_output \
  --frame-budget 120 \
  --frame-size 224 \
  --epochs 3 \
  --gradient-accumulation-steps 8 \
  --learning-rate 1e-4
```

Keep `eval_full295_stage1_samples.jsonl` for separate final evaluation; the
trainer only uses the train split and holds out whole train videos for its
validation set. The Stage 1 builder creates three paired views per sentence: `0→start`
with earlier ASR, `0→end` with earlier ASR, and `0→start` with earlier ASR
masked. The trainer holds out whole videos and writes `validation_samples.jsonl`.
`--frame-budget 120` caps the number of frames across each full selected
interval; intervals longer than 120 seconds are uniformly subsampled.
For long videos, compare `--frame-sampling recent_sparse --recent-seconds 120`:
it uses the same 120-frame budget, reserving 96 frames for the latest two
minutes and 24 for older history. Pass the same options to `stage1.evaluate`.
Compare normal, blank, and shuffled video with `stage1.evaluate
--visual-control normal|blank|shuffled` in separate runs.

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
  --format legacy \
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
  --training-mode legacy \
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
  --inference-mode legacy \
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
