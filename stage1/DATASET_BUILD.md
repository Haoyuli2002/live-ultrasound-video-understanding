# Stage 1 dataset v2

The new entry points are `stage1.transcribe` and `stage1.prepare_dataset`.
They produce a separate dataset; they do not modify the existing training run
or overwrite its transcripts/samples. Start with a few videos and review the
audio and quarantine report before rebuilding a full split.

## Recognition

```bash
python -m stage1.transcribe \
  --video-map /path/to/train_video_map.json \
  --output-dir /scratch/stage1_v2_train/asr \
  --model large-v3 --device cuda --language en --resume
```

The model is loaded once per process. Recognition enables word timestamps,
VAD, beam size 5 and temperature 0, and disables previous-text conditioning
to reduce propagation of recognition loops. It preserves word probability,
segment average log probability, compression ratio and no-speech probability.
These options are supported by [faster-whisper's transcription API](https://github.com/SYSTRAN/faster-whisper/blob/master/faster_whisper/transcribe.py).
They do not guarantee correct medical terminology or exact audio alignment.
Native zero-duration individual words are retained without inventing a duration
and counted in sentence metadata; reversed/out-of-bounds times are rejected.

Each output JSON preserves the decoded segments and words, full text, decoding
settings, library version, video content hash and implementation hashes.
`--resume` skips only transcripts with the same video hash, recipe and selected
map. A changed video/model/implementation requires a new output directory.
Relative video paths default to the map directory. For the existing cluster
maps containing `cluster_data/videos/...`, pass `--video-root "$PWD"` from the
repository root; the Slurm wrapper does this automatically. The selected output
map always contains absolute paths for the sentence builder.
Failures are recorded and cause a nonzero exit. Temporary WAVs are removed even
if recognition fails. Use `--limit-videos 3` for a small ASR run; its exact map
is saved as `selected_video_map.json`.

## Sentence boundaries and paired samples

```bash
python -m stage1.prepare_dataset \
  --transcripts /scratch/stage1_v2_train/asr \
  --video-map /scratch/stage1_v2_train/asr/selected_video_map.json \
  --output-dir /scratch/stage1_v2_train/dataset
```

The builder groups words across ASR segment boundaries and splits at sentence
punctuation, retaining native word start/end times. Abbreviations and decimal
numbers do not create sentence endings. ASR segment boundaries and an unfinished
tail are not promoted to complete sentences. Discarded segments break sentence
continuity, so good text on either side cannot form an invented sentence.

For missing or poor punctuation, use an already running OpenAI-compatible
Qwen3.5-27B endpoint to select **word boundary indices only**:

```bash
python -m stage1.prepare_dataset \
  --transcripts /scratch/stage1_v2_train/asr \
  --video-map /scratch/stage1_v2_train/asr/selected_video_map.json \
  --output-dir /scratch/stage1_v2_train/dataset_llm_boundaries \
  --boundary-mode llm --model Qwen/Qwen3.5-27B \
  --base-url http://127.0.0.1:8000/v1
```

The boundary teacher cannot supply replacement words or timestamps. It is not
a transcription corrector. Uncertain medical terms require audio review or
recognition/alignment again. Editing segment words while retaining old word
times is rejected as `word_text_mismatch_realign_required`.

The default quality policy quarantines:

- Missing, invalid, overlapping or out-of-video timestamps; text/word mismatch.
- Low-confidence segments (average log probability below -1) or compression
  ratio above 2.4; sentences with mean word probability below 0.5.
- Empty sentences, corrupt intervals below 0.5 seconds, or above 8 words/second.
- Suspected consecutive phrase loops, three consecutive identical sentences,
  and chat/control markup. These are review flags, not proof of ASR errors.

Targets must contain **at least 10 whitespace-separated words** and span
**at least 5 seconds**. Failing either condition excludes the entire triplet;
exactly 10 words / 5 seconds passes. Short but otherwise sound sentences remain
in earlier ASR history, with `scope: target` / `exclude_target_keep_history`
records in the review file. Low-quality speech is excluded from history too.
There is **no maximum word count or duration**. Long, informative narration is
retained under its original words and timestamps. Length alone is not evidence
of ASR corruption; repetition, confidence and timing checks remain independent.
An unpunctuated tail is still held for boundary review rather than being declared
a complete sentence automatically. The report cannot certify medical relevance
or correctness; sample those against the source before training.

Every retained target produces exactly three rows with the same target:
`before_with_asr`, `through_with_asr`, `before_mask_asr`. History contains only
earlier accepted sentences; the first accepted sentence serves as history.
Speech rejected for quality never enters the training prompt or target. IDs retain candidate
indices, so gaps are expected. Output rows also carry alignment method, source
hash, source segment indices and mean word confidence.

Old segment-only ASR can be diagnosed with `--allow-estimated-timestamps`.
Those units are explicitly marked `segment_character_interpolation`; this does
not upgrade them to word alignment. The production command leaves this off.
Thresholds are recorded in the report; CLI overrides are available for minimum
target duration, target word count, maximum speech rate and mean word confidence.

## Artifacts and acceptance

| File | Purpose |
|---|---|
| `samples.jsonl` | Training-compatible paired rows; exists only when ready |
| `sentences.jsonl` | Accepted sentences with timing and provenance |
| `quarantine.jsonl` | Rejected segments/sentences and unfinished tails, with reasons |
| `audit.jsonl` | Per-video coverage and failure details |
| `video_map.json` | Videos that actually yielded samples |
| `report.json` | Quality policy, coverage, reasons and ready/rejected status |
| `manifest.json` | Source, implementation, map and artifact hashes |

The directory is published after construction. Source failure, no samples, or
a selected video yielding no samples rejects the dataset, returns nonzero and
omits `samples.jsonl`; diagnostic files remain available. Only use
`--allow-empty-videos` after reviewing the empty-video list. Reruns require a
new dataset output directory. Build train/eval maps separately and verify their
video IDs are disjoint before training.

At 30 frames, uniform sampling over a long prefix often misses the target
interval. The report compares uniform and `recent_sparse` with a 15-second
recent window, using the same budget. For the next experiment, use the same
`--frame-sampling recent_sparse --recent-seconds 15` in **training and evaluation**.
The report checks planned timestamps; actual decode and ultrasound visibility
still need review. Sparse sampling cannot establish absence of motion.

## Cluster job

```bash
mkdir -p logs
# First run on three selected videos; this submits preparation only.
export DATA=/dss/mcmlscratch/04/ge75vid2/haoyu/live-ultrasound-video-understanding
export RUN_ROOT="$DATA/cluster_data/stage1_datasets/train_v2_smoke_$(date +%Y%m%d_%H%M%S)"
sbatch --export=ALL,LIMIT_VIDEOS=3 scripts/slurm/run_stage1_prepare_v2.sbatch

# After inspecting smoke output/audio, prepare the full selected train split.
unset LIMIT_VIDEOS RUN_ROOT ASR_DIR DATASET_DIR
sbatch scripts/slurm/run_stage1_prepare_v2.sbatch
```

The wrapper defaults to **one A100 MIG 20GB slice**, four CPU cores, 16GB host
memory, `lrz-dgx-a100-40x8-mig` and QoS `mig`. It uses large-v3 and native ASR punctuation. It does not
launch a 27B server or submit training. Its default eight-hour walltime is an
allocation request, not a measured runtime. To resume interrupted recognition,
reuse `RUN_ROOT`/`ASR_DIR`; if the dataset was already published, choose a new
`DATASET_DIR`. Do not run two jobs against the same directory concurrently.
The job prepends the existing CUDA 12 cuBLAS/cuDNN runtime library directories
for CTranslate2 and checks that the allocated GPU supports FP16 before recognition.
No environment packages are installed by the job. The current cluster environment
uses cuDNN 9.24, whose [support matrix](https://docs.nvidia.com/deeplearning/cudnn/backend/v9.24.0/reference/support-matrix.html)
starts at Turing (compute capability 7.5). V100/P100 require a separate compatible
ASR runtime; their VRAM capacity alone does not establish compatibility with this
environment. A100 MIG preserves compatibility with the installed stack.

```bash
python -m unittest stage1.test_prepare_dataset stage1.test_restore_punctuation \
  stage1.test_stage1 stage1.test_clean_asr
```

These tests cover corrupt timing, stale alignment, causal triplets, repetition
quarantine, no joining across rejected speech, ASR provenance checks and atomic
publication. They use fake recognition/teacher responses, not a GPU accuracy benchmark.
