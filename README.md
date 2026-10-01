# Live Ultrasound Video Understanding

[English](README.md) | [简体中文](README_zh.md)

A research project on **streaming evidence sufficiency for ultrasound video question answering**: continuously observe a video, wait while visual evidence is insufficient, and answer when sufficient evidence becomes available.

The intended system combines ultrasound domain pretraining, bounded visual memory, and an explicit answerability decision. The repository contains data preparation tools, training and evaluation baselines, and an evolving implementation of this design. It is not yet a complete, validated online clinical system.

## Design and implementation status

The [reference design](docs/live_ultrasound_reference_implementation_final_zh.md) defines the intended three-stage system. [Pipeline commands](docs/HOW_TO_RUN_PIPELINE.md) collect operational examples; some examples still describe legacy baselines. CLI arguments and the status notes below should be checked when reproducing an experiment.

| Component | Intended design | Current implementation |
|---|---|---|
| Data preparation | Video collection, ASR transcription, and clipping | Crawler, ASR, video clipping, and ASR-based filtering scripts exist |
| VLM labeling | Classify videos, annotate anatomy and clinical scenarios, and filter by training stage | Qwen3.5 primary labeling, optional Gemini cross-validation, label merging, and pretrain/compression/sft keep-flag filtering exist |
| Stage 1: domain pretraining | Paired `0→start`, `0→end`, and historical-ASR-mask views → one complete narration sentence | New independent `stage1/` data, LoRA training, and visual-control evaluation code; GPU run pending |
| VLM data generation: local/block/global summaries | Generate six ten-second, one whole-minute, and one cumulative visual summary per full minute for Stage 2 | New `stage2.annotate` and `stage2.build_data` code exists; labels meeting the new eight-record specification have not yet been generated or reviewed, and old minute-level labels cannot be used directly |
| Stage 2: visual memory | One short-memory token per second; every 60 seconds replace long memory with 60 new tokens; local/global summary supervision | New independent `stage2/` chronological training and streaming code; GPU run pending |
| QA construction | Annotate `query_time` and earliest sufficient-evidence `answer_time`, with validation | Generation, validation, and WAIT/ANSWER expansion exist; automatic timestamps are candidate annotations, not expert ground truth |
| Stage 3: answerability + QA | `<DECISION>` hidden state → BCE classification; NTP for both WAIT reasons and ANSWER text | Current trainers generate `<WAIT>`/`<ANSWER>` text; a dedicated decision head is not implemented |
| Eval | Causal streaming evaluation of answerability, response timing, answer quality, memory retention, and runtime | Pretraining, memory, and QA baseline evaluation scripts exist; the complete online loop and benchmark protocol remain to be implemented and validated |

## Model workflow

ASR supports data preparation, provides Stage 1 targets, and supplies earlier narration in two Stage 1 conditions. Stage 2, Stage 3, and online inference do not receive ASR. Offline annotators may inspect future video to construct labels, but model decisions use only evidence available at the decision time.

### Video collection, ASR transcription, and clipping

Use `UltrasoundCrawler_KeyCode_20260323_v2/` to collect ultrasound videos from YouTube/Bilibili. `QA/prepare/run_prepare.py` runs ASR transcription and video clipping, producing timestamped transcripts and clip metadata for training-sample and QA construction.

### VLM labeling

Before constructing training data, label videos with `scripts/data/classify_ultrasound_video_type_teacher.py`. The primary teacher is Qwen3.5 served through a local vLLM OpenAI-compatible endpoint; Gemini via OpenRouter can provide optional cross-validation. This step determines video type and stage suitability, independently of Stage 2 visual-summary annotation.

| Video label | Meaning | Merged-label keep policy |
|---|---|---|
| `hands_on_ultrasound_teaching` | Hands-on scanning and ultrasound instruction | All three stages |
| `pure_ultrasound_scan` | Ultrasound machine output / cine loop | All three stages |
| `ultrasound_ppt_lecture` | Ultrasound slide lecture | Pretraining only |
| `mixed_ultrasound_teaching` | Mixed scanning, instruction, and slides | Pretraining only; clipping required |
| `ultrasound_image_discussion` | Static ultrasound image discussion | Pretraining only |
| `non_ultrasound_or_irrelevant` | Non-ultrasound or irrelevant content | Drop from all stages |
| `uncertain` | Undetermined type | Drop after merging; review required |

Raw annotations also contain confidence, `anatomy_regions`, `clinical_scenarios`, `scan_views_or_targets`, language and supporting evidence, content flags for real-time ultrasound/probe/patient/slides, estimated ultrasound fraction, teaching value, clipping needs, visual evidence, and `keep_for_pretrain`, `keep_for_compression`, `keep_for_sft`.

The merger uses agreement and confidence differences to select a final label and emits review metadata such as `needs_human_review`. It recomputes keep flags from the final class. Raw teacher labels may retain uncertain videos for pretraining audit or assign different flags to some mixed videos; the merged policy is the table above. Teacher agreement is not expert validation, and filtering does not perform clipping automatically.

### Stage 1: ultrasound domain pretraining

The new `stage1/` path creates three matched views for each target sentence `[start,end]`: video `0→start` with earlier ASR, video `0→end` with earlier ASR, and video `0→start` with earlier ASR masked. All predict the same complete sentence; target and later ASR are excluded from the prompt. Up to 120 frames are sampled across each selected interval. The model uses Qwen3-VL LoRA with video-level validation; normal, blank, and shuffled-frame inference checks visual dependence. See [stage1/README.md](stage1/README.md). The earlier `pretrain/` baseline remains available.

### Stage 2: visual-summary supervision and recurrent memory

A teacher VLM generates three targets from silent video clips: ten-second summaries, the current 60-second summary, and the cumulative summary from the video beginning. At each full minute boundary `T`, it writes eight independent labels: six ten-second intervals, one `[T-60,T)` block summary, and one `[0,T)` global summary. The data builder combines these into one training block. The annotator sends each video interval directly instead of selecting a fixed set of image frames; the teacher service still decodes and samples video internally, so brief findings need review. Stage 2 does not receive ASR input and trains as follows:

- Process each video chronologically from its beginning, generating one short-memory token per second.
- Every ten seconds, reconstruct the interval's summary from its ten short tokens, compute `Lshort`, and update parameters. Detach and retain these short states until the minute ends.
- Every 60 seconds, reconstruct the whole-minute summary from the current 60 short tokens to compute `Llocal`. Also combine `detach(L_previous) + S_current` to produce 60 new long tokens, then reconstruct the cumulative summary to compute `Lglobal`.
- At the 60-second boundary, sum the final ten seconds' `Lshort`, the current minute's `Llocal`, and the cumulative-history `Lglobal` with weights `1:1:1` for one optimizer update. The preceding 50 seconds' short states participate but receive no gradients from this update.
- Detach the new long memory before carrying it into the next minute and clear the current short memory. Retain only the latest long-memory state, which recursively carries accumulated history.

Training rows contain `local_sub_summaries`, an independent `block_summary_target`, and `global_summary_target`. An incomplete final minute supervises only complete ten-second windows. CPU schedule and gradient tests pass; end-to-end Qwen/LoRA training remains unverified.

### Stage 3: answerability and multi-question KV caching

The following Stage 3 design remains to be fully implemented. The existing `QA/train/train_summary_decide.py` appends long-memory tokens and generates WAIT/ANSWER text; it still needs alignment with Stage 2's fixed 60-token replacement update, a dedicated decision head, and multi-question cache management.

#### Input and causal attention

```text
Shared prefix                         Independent suffix per question
[Fixed system / template] [Long memory] [Short memory] [Question] [<DECISION>]
                                                               |
                                               decision hidden state h_dec
                                                               |
                                                 Linear → logit z → sigmoid
                                                               |
                                                          WAIT / ANSWER
```

Long and Short are continuous memory embeddings. Causal attention lets the final `<DECISION>` position read the current memory and complete question. Persistent memory is maintained from video independently of questions; each question reads the same memory without writing its question-conditioned states back into it. The first baseline uses memory and question only; current raw frames remain an ablation.

#### BCE for `<DECISION>` and NTP for WAIT/ANSWER text

`<DECISION>` is an input readout token, not generated text. Each question at each decision timestamp has a binary label: `y=0` for insufficient evidence and `y=1` for sufficient evidence. **Both WAIT and ANSWER examples receive BCE and NTP supervision**; their text targets differ:

```text
Input: shared memory prefix + Question + <DECISION>
Decision: z = Linear(h_dec)
          L_decision = BCEWithLogitsLoss(z, y)

WAIT   (y=0): text target = <WAIT>   + a specific reason naming missing evidence
ANSWER (y=1): text target = <ANSWER> + a visually supported answer

L_text = NTP(the corresponding text target)
L_stage3 = λ_decision · L_decision + λ_text · L_text
```

Place the target after `<DECISION>` in the assistant output. Next-token prediction (NTP) supervises only output tokens, including the `<WAIT>`/`<ANSWER>` marker and its reason/answer. Mask the fixed prefix, memory, question, `<DECISION>`, and assistant template from text loss. A WAIT reason identifies evidence missing **at the current time** without revealing future visual details or conclusions. Causal attention prevents `<DECISION>` from seeing the later target text, so BCE and NTP can be computed in one training forward.

At inference, use the BCE logit and threshold to select WAIT or ANSWER, then generate the corresponding text from that question's memory snapshot. WAIT returns a marker and specific reason while keeping the question active; ANSWER returns a marker and answer. The selected `<WAIT>`/`<ANSWER>` prefix can be fixed before generating the rest to keep text consistent with the decision. Initially set `λ_decision = λ_text = 1`.

#### Multiple questions: shared prefix, independent decisions

Compute the shared prefix once per decision timestamp, then branch independently for each active question:

```text
Fixed prefix + Long + Short → shared prefix KV cache
                              ├── Q1 + <DECISION> → logit1 → WAIT reason / ANSWER answer1
                              ├── Q2 + <DECISION> → logit2 → WAIT reason / ANSWER answer2
                              └── Q3 + <DECISION> → logit3 → WAIT reason / ANSWER answer3
```

Do not concatenate questions: Q2 must not read Q1. Each question has its own `<DECISION>`, logit, and WAIT-reason/ANSWER-text branch. In the first baseline, finish the memory update and rebuild the shared prefix cache once at each new decision timestamp. Isolate mutable question suffixes so one branch cannot contaminate another's KV state.

#### KV cache updates and invalidation

- **New evidence:** recompute each WAIT question and `<DECISION>` against the new memory prefix; the old question KV does not contain new evidence.
- **Minute boundary:** replacing long memory and clearing short memory invalidates the old prefix cache. Video, model weights, template, or position-configuration changes also invalidate it.
- **Answer generation:** keep the triggering memory snapshot and that question's branch; later video updates must not mutate the in-flight answer state.

Persistent memory embeddings and per-layer attention KV caches are distinct states. This cache reuse is for multi-question inference, not reuse of training graphs across optimizer updates. The implementation should check cached versus full-forward logits and invariance to question order.

## Repository map

```text
stage1/                   New paired-condition Stage 1 implementation
stage2/                   New chronological Stage 2 implementation
pretrain/                  Earlier Stage 1/2 experiments and baselines
  build_samples.py         ASR-supervised sample construction
  train.py / infer.py      Domain pretraining and inference
  build_teacher_memory_summary_samples.py
  memory_dataset.py / memory_collator.py
  train_memory_compression.py / infer_memory_compression.py
  eval_memory_compression.py
QA/                        Temporal QA construction and SFT baselines
  prepare/                 ASR and clipping preparation
  generator.py             Streaming QA candidates with two timestamps
  offline_generator.py     Offline QA candidates
  validator.py             Temporal/evidence validation
  merger.py / run.py        Export, WAIT/ANSWER expansion, pipeline driver
  train/                   Direct QA and memory-based SFT baselines
  eval/                    Baseline inference and prediction analysis
  schema.md                QA and training data schema
scripts/                   Shared helpers and legacy pipeline
  data/                    Video filtering and teacher-summary labeling
  slurm/                   Cluster experiment launchers
src/                       Earlier video-filter/input-mode experiments
notebooks/                 Interactive experiments and walkthroughs
docs/                      Design, commands, and experiment reports
results/                   Local outputs and selected example artifacts
livecc/                    LiveCC reference checkout
UltrasoundCrawler_KeyCode_20260323_v2/
                           YouTube/Bilibili collection tools
```

Local datasets, video maps, checkpoints, and outputs are not all included in the repository. Commands require valid paths on the execution machine.

## Setup

Install `ffmpeg` and a Python environment compatible with the chosen PyTorch/Transformers stack. GPU training and local teacher serving are intended for a suitable CUDA environment; requirements for individual cluster experiments may differ.

```bash
pip install -r requirements.txt
```

Create a local `.env` from `.env.example` if needed and configure credentials for the steps you use:

- `OPENROUTER_API_KEY`: OpenRouter-backed video annotation and validation.
- `OPENAI_API_KEY`: legacy OpenAI-backed pipeline components.
- Local teacher endpoints: configure the base URL and API-key environment variable accepted by the relevant script.

Shared pipeline helpers load `.env` automatically. Remote annotation calls consume API credits. Keep credentials out of version control.

## Working examples

Run from the repository root and replace example paths with local paths. Video maps use `{video_id: video_path}`. Configure API credentials and start the local teacher endpoint first.

### 0. VLM labeling, video classification, and stage-specific filtering

Run primary labeling, optional Gemini labeling, merging, and filtering in order. For a single teacher, omit the Gemini command and `--validator-audit`. Choose `pretrain`, `compression`, or `sft` for `--stage`.

```bash
python scripts/data/classify_ultrasound_video_type_teacher.py \
  --video-map /path/to/video_path_map.json \
  --output results/video_types/qwen35.jsonl \
  --teacher qwen35 \
  --model Qwen/Qwen3.5-35B-A3B \
  --base-url http://localhost:8000/v1 \
  --api-key-env VLLM_API_KEY \
  --video-fps 0.5 \
  --disable-thinking \
  --resume

python scripts/data/classify_ultrasound_video_type_teacher.py \
  --video-map /path/to/video_path_map.json \
  --output results/video_types/gemini3.jsonl \
  --teacher gemini3 \
  --model google/gemini-3.1-pro-preview \
  --base-url https://openrouter.ai/api/v1 \
  --api-key-env OPENROUTER_API_KEY \
  --video-fps 0.5 \
  --resume

python scripts/data/merge_video_type_teacher_labels.py \
  --primary-audit results/video_types/qwen35.jsonl \
  --validator-audit results/video_types/gemini3.jsonl \
  --output results/video_types/final.jsonl \
  --output-summary results/video_types/summary.json

python scripts/data/filter_by_vlm_video_type.py \
  --video-map /path/to/video_path_map.json \
  --vlm-audit results/video_types/final.jsonl \
  --stage compression \
  --output-keep-map results/video_types/compression_keep.json \
  --output-drop-map results/video_types/compression_drop.json
```

### 1. Prepare and generate temporal QA

Generate transcripts and clips, then pass their paths to the QA driver. See [QA schema](QA/schema.md) for output formats.

```bash
python QA/prepare/run_prepare.py \
  --video /path/to/VIDEO_ID.mp4 \
  --output-dir QA/results \
  --whisper-model base

python QA/run.py \
  --video /path/to/VIDEO_ID.mp4 \
  --transcript QA/results/transcripts/VIDEO_ID.json \
  --clips QA/results/clips/VIDEO_ID_clips.json \
  --out-dir QA/results \
  --validation-mode all \
  --expand-wait-answer
```

### 2. Build Stage 1 samples and train

```bash
python -m stage1.data \
  --transcripts results/transcripts \
  --output stage1/samples.jsonl

python -m stage1.train \
  --model-name Qwen/Qwen3-VL-2B-Instruct \
  --train-jsonl stage1/samples.jsonl \
  --video-path-map /path/to/video_path_map.json \
  --output-dir /path/to/stage1_output \
  --frame-budget 120 \
  --frame-size 224 \
  --epochs 3 \
  --gradient-accumulation-steps 8 \
  --learning-rate 1e-4
```

### 3. Stage 2: visual-summary generation and memory training

Generate visual summaries, build samples, then train. Optionally add `--init-adapter /path/to/stage1_adapter` for Stage 1 initialization.

```bash
python -m stage2.annotate \
  --video /path/to/VIDEO_ID.mp4 \
  --video-id VIDEO_ID \
  --model google/gemini-3.1-pro-preview \
  --base-url https://openrouter.ai/api/v1 \
  --api-key-env OPENROUTER_API_KEY \
  --output results/teacher_visual_summaries/VIDEO_ID_visual_summaries.jsonl

python -m stage2.build_data \
  --teacher-jsonl results/teacher_visual_summaries/VIDEO_ID_visual_summaries.jsonl \
  --output results/teacher_visual_summaries/VIDEO_ID_memory_samples.jsonl
```

```bash
python -m stage2.train \
  --model-name Qwen/Qwen3-VL-2B-Instruct \
  --train-jsonl results/teacher_visual_summaries/VIDEO_ID_memory_samples.jsonl \
  --video-path-map /path/to/video_path_map.json \
  --output-dir /path/to/stage2_output \
  --frame-size 224 \
  --epochs 1 \
  --learning-rate 1e-4
```

See [Pipeline commands](docs/HOW_TO_RUN_PIPELINE.md) and the [LRZ cluster guide](docs/LRZ_CLUSTER_GUIDE.md) for full options, inference, and cluster commands.

## Model evaluation goals

Evaluation should establish the contribution of the model and its training objectives. The supporting benchmark must distinguish **whether to answer**, **when to answer**, and **whether the answer is supported**.

- Decision: precision/recall, AUROC/AUPRC, calibration, premature answers, and missed answers.
- Timing: response delay relative to sufficient evidence, including questions that remain unanswerable through the video end.
- Answer quality: correctness, visual grounding, hallucinations, and expert assessment.
- Memory: evidence retention by age, long-history QA, and performance around compression boundaries.
- Runtime: streaming throughput, response latency, peak GPU memory, and the cost of memory updates.

These are evaluation requirements, not completed benchmark results. Automatic teacher/validator outputs need expert validation for a credible test set. Data splits should prevent source-video overlap and, where metadata permit, patient/source leakage. Use matched visual and compute budgets when comparing memory methods.

## Documentation and experiment history

- [Reference design](docs/live_ultrasound_reference_implementation_final_zh.md): target architecture and semantics.
- [Pipeline commands](docs/HOW_TO_RUN_PIPELINE.md): operational examples.
- [Pretraining V1](docs/pretrain_V1.md) and [full-data V1](docs/Pretrain_Full_V1.md): earlier experiments.
- [Interleaved pretraining](docs/Pretrain_Full_V3_Interleave.md) and [improvement objectives](docs/Pretrain_Improvement_Objectives.md): historical/alternative input objectives.
- [SFT V1](docs/sft_V1.md): reported WAIT-collapse experiment and diagnosis.

Historical documents may use different stage numbering or input policies. They should not be read as proof that the current reference design is fully implemented.

## References and use

- [LiveCC](https://github.com/showlab/livecc): streaming video-language training reference.
- [Qwen3-VL](https://github.com/QwenLM/Qwen3-VL): current training backbone family.
- [faster-whisper](https://github.com/SYSTRAN/faster-whisper): ASR implementation.

Research use only. Third-party code, models, and source videos retain their respective licenses and usage terms.
