# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Current work (2026-11-22)

The active StreamReady + Qwen3.5 path is `endtoend/`; see `endtoend/README.md` and `endtoend/ALL_PAIRS.md`. The Stage1/2/3 and older QA branches below are retained for existing experiments. Follow root `AGENTS.md`: update `programs.md` before every training submission or commit, and put future checkpoints on scratch.

## Context

Research repo (TUM guided research) for real-time ultrasound video understanding: crawl teaching videos → build data → LoRA-train `Qwen/Qwen3-VL-2B-Instruct` in two stages → evaluate. This checkout normally lives on the **LRZ AI Systems Slurm cluster** at `/dss/dsshome1/04/ge75vid2/haoyu/live-ultrasound-video-understanding`; the current workflow edits, tests, commits and pushes directly in this remote workspace. Historical Mac workflow notes are not the current execution location.

Most `docs/*.md` are written in Chinese. Match the language of the file you are editing.

## Environment

```bash
source /dss/dsshome1/04/ge75vid2/miniconda3/etc/profile.d/conda.sh
conda activate ultrasound      # `cv2 not found` almost always means this step was skipped
```

API keys live in `.env` (gitignored, template in `.env.example`). Never `export` them manually — `scripts/_env_loader.py` is imported for its side effect by everything that needs them. `OPENROUTER_API_KEY` drives all QA generation/validation (Gemini 2.5 Flash); `OPENAI_API_KEY` is only used by the legacy `scripts/` GPT-4o pipeline.

## Architecture

Everything shares crawl → filter → **prepare (ASR + clipping)**, then splits into two *fully decoupled* training branches. `docs/PIPELINE.md` is the end-to-end reference; `QA/schema.md` is the sole authority on data schemas.

```
UltrasoundCrawler_KeyCode_20260323_v2/   crawl (yt-dlp, CLI + web UI)
src/video_filter.py, scripts/video_filter_vlm.py, scripts/batch_filter.py   VLM filter
QA/prepare/run_prepare.py                ASR (faster-whisper) + clipping → transcripts/{id}.json, clips/{id}_clips.json
├── pretrain/    Stage 1 — ASR caption completion, no special tokens
└── QA/          Stage 2 — SFT for <WAIT>/<ANSWER> answerability
```

**Stage 1 (`pretrain/`)** — `build_samples.py` → `train.py` → `infer.py` → `compare_predictions.py`. The sample *format* is the main experimental axis and has evolved through four versions; `--format`/`--unit` on `build_samples.py` select it:

| Version | flags | objective |
|---|---|---|
| V1 | `--unit segment --format standard` | window frames + text `prev_context` → next ASR segment |
| V2 | `--unit sentence --sentence-mode segment_merge` | merged sentence-like chunks |
| V3 | `--unit sentence --sentence-mode punctuation --format interleave` | interleaved (frames+text) history chunks → next sentence chunk |
| V4 | `... --format mixedmask` | V3 + per-sample random masking of current/history visual or text modality |

`pretrain/collator.py` builds the three message shapes (`build_messages` / `build_interleave_messages` / `build_mixedmask_messages`); `pretrain/infer.py --mixedmask-eval-mode {no_mask,mask_current_visual,random_unit_modality_mask}` must be chosen to match how V4 samples are to be evaluated.

**Stage 2 (`QA/`)** — `QA/run.py` chains `offline_generator` → `generator` (streaming) → `validator` → `merger`. Streaming QA is single-anchor (clip midpoint, 1 QA per clip) and carries `query_time` (evidence insufficient) + `answer_time` (evidence sufficient) + `wait_reason`. The `wait_reason` is the anti-collapse mechanism: generator and validator both run `is_generic_wait_reason()` and drop/fail generic ones, and `merger.py` emits `<WAIT> {wait_reason}` so WAIT targets stay diverse. `QA/train/train.py` adds `<WAIT>`/`<ANSWER>` as special tokens with `embed_tokens`/`lm_head` in `modules_to_save`; `pretrain/train.py` deliberately does not touch the vocab.

Both trainers compute loss **only on the assistant target** (everything else `-100`), implemented by locating the target token subsequence inside `input_ids` so it survives Qwen-VL image-placeholder expansion.

`QA/_shared/__init__.py` re-exports `scripts/_video_llm.py` (ffmpeg clip cutting, base64, OpenRouter client, retries) rather than duplicating it — keep that single implementation. Scripts are run as files, not packages, so cross-directory imports go through `sys.path.insert` at the top of each entry point; preserve that pattern rather than converting to package imports.

## Running things

Long jobs go through Slurm; `scripts/slurm/*.slurm` are self-contained (cd, conda activate, build samples, train, eval, compare) and are the **most current record of working hyperparameters** — prefer them over the command examples in `docs/`, which lag behind. `docs/LRZ_CLUSTER_GUIDE.md` has the full cluster cheat sheet.

```bash
sbatch scripts/slurm/pretrain_v4_mixedmask_train_eval_full.slurm
squeue -u ge75vid2
tail -f logs/pretrain_v4_mixedmask_train_eval_<job_id>.out
sacct -j <job_id> --format=JobID,State,ExitCode,Elapsed,MaxRSS

# interactive debugging instead
srun --partition=mcml-hgx-h100-94x4 --qos=mcml --gres=gpu:1 \
     --cpus-per-task=4 --mem=32G --time=04:00:00 --pty bash
```

Data-prep / generation entry points:

```bash
python QA/prepare/run_prepare.py --video path/to/{id}.mp4 --output-dir QA/results
python QA/run.py --video path/to/{id}.mp4 --expand-wait-answer \
    --validation-mode {all|sample|none}     # all = eval sets, none = bulk train generation
```

Checks (no pytest in this repo — these are runnable scripts):

```bash
python QA/smoke_test.py                    # offline generator/validator/merger logic, no API calls
python QA/test/check_collator_labels.py    # asserts supervised labels decode to <WAIT>/<ANSWER> target
```

## Conventions worth keeping

- `pretrain/train.py` and `pretrain/infer.py` support `--resume-from-checkpoint auto` / `--resume`; inference refuses to clobber an existing `--output` unless `--overwrite`. Always prefer `--resume` on the cluster — interactive shells get killed.
- Training/inference entry points set `os.environ.setdefault("TRANSFORMERS_NO_TF", "1")` *before* importing transformers, because mixed images otherwise pull in TensorFlow/Keras and fail. Keep it at the top of any new entry point.
- `requirements.txt` pins `numpy<2` (and matching pyarrow/pandas/sklearn ranges) for AzureML base images — do not relax without checking that environment.
- Some source videos are unreadable by OpenCV; run `scripts/transcode_videos_for_opencv.py --video-path-map ... [--replace --delete-backup]` and expect `bad_count: 0` before building samples.
- `cluster_data/`, `logs/`, `QA/results*/`, `local_checkpoints/`, and all media are gitignored. Never commit data or checkpoints. Multi-video runs are wired by `--video-path-map cluster_data/splits/{train,eval}_videos.json` (`{video_id: mp4_path}`).
- `scripts/` holds the older GPT-4o-based pipeline (`run_pipeline.py`, `qa_generation.py`, `streaming_qa_generation.py`, `qa_validator.py`, `qa_merge.py`). It is superseded by `QA/` for generation but still owns shared primitives (`_video_llm.py`, `_env_loader.py`, `asr_pipeline.py`, `video_clipping.py`) that `QA/` imports. Don't delete those.
- `README.md`'s directory listing and `docs/PIPELINE.md`'s pretrain section predate V3/V4; the live design notes are `docs/Pretrain_Full_V3_Interleave.md` and `docs/Pretrain_Improvement_Objectives.md`.
