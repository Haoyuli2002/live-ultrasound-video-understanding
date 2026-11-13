# Ultrasound stage data browser

Select a video, stage and sample at `/viz/index.html`. The current catalog uses
actual source rows and never synthesizes missing stage data.

## All videos and ratings

The **All videos & ratings** tab lists all 237 labelled train/eval videos. It
plays each original MP4, filters by video type and the selected Stage's value,
shows the original rating reason, and browses time-aligned data for that video.
Stage 2 contains the 237-video reviewed release (Local, Block and Global).
Stage 3 exposes four existing QA collections separately, covering 130 unique
videos; the visual-first 2026-11-10 collection remains marked as a draft.
Local Stage 1 sample snapshots exist for only two videos. Other videos show an
explicit empty Stage 1 state while retaining their original Stage 1 rating.
New catalogue text is shown in its source language; existing Chinese display
translations for the older demo examples remain available.

`scripts/build_video_library.py` builds `viz/video_library.json` from the
unchanged video labels and local source snapshots in
`viz/data/library_sources/`. `scripts/viz_demo.sh` rebuilds both demo catalogs.
The label and source snapshots are ignored by Git like the other demo data;
when moving the checkout, copy them into `viz/data/` before rebuilding. You can
point the label builder at another file with `VIDEO_LABELS=/path/to/video_labels.json`.
Its source list and SHA256 hashes are saved in the generated JSON. The browser
never changes the original labels or merges distinct Stage 3 QA collections.

- Stage 1: groups the original JSONL rows into matched condition triplets,
  preserving targets/windows/masking. Displays the training template's history
  (last 4000 characters), uniform frame timestamps and the recorded 30-frame,
  224-pixel configuration. No transcript resegmentation or prediction is used.
- Stage 2: shows the six local labels, independent block label and cumulative
  global label per minute. Playback uses each label's actual teacher window.
  Source notes distinguish assistant visual drafts from model-generated labels.
  These are supervision labels, not the student's continuous memory vectors.
- Stage 3: displays existing QA candidate records, original video windows,
  WAIT/ANSWER targets, query/answer times and reference evidence metadata.
  This does not imply that the Stage 3 training pipeline has been completed.

Missing stages have explicit empty states. Partial video coverage is displayed.
The playback buttons show the specified windows; the player itself can also
play the whole video, so it is a review tool, not a restricted training input.
Browser playback can overshoot a stop boundary slightly due to `timeupdate`
frequency; the numeric windows remain the exact label boundaries.

## Build and serve

```bash
# From repo root, use the ultrasound environment or the local virtualenv.
PYTHON=.venv/bin/python bash scripts/viz_demo.sh
# Another port:
PORT=8080 PYTHON=.venv/bin/python bash scripts/viz_demo.sh
# Rebuild only:
SERVE=0 PYTHON=.venv/bin/python bash scripts/viz_demo.sh
```

If the existing server is already running on 8000, rebuild only and reload.
The Range-aware server supports MP4 seeking. For remote use, serve on the SSH
machine and forward its port to the laptop.

`viz/demo_sources.json` lists repo-relative source paths, titles and provenance
notes. Replace/add entries to connect further existing videos or data. The
builder requires real files, rejects incomplete/duplicate Stage 1 triplets and
uses Stage 2's schema validator. All text is rendered as text, not HTML.
`viz/data/` contains local source snapshots and `viz/catalog.json` is generated;
both are ignored by Git. Transfer these source files separately when moving
the demo to a different checkout, or update the config to point at existing
cluster files. Source SHA256 hashes are available in each stage's details.

The former `stage1.visualize_conditions` and `scripts/viz_eval.sh` produce the
legacy single-video `viz/data.json`; they are not the multi-stage entry point.
That old file is retained, but the current page reads `viz/catalog.json`.


## Reviewed Stage 3 batch (2026-11-04)

Ten videos now include 45 English QA tasks: 22 immediate, 22 delayed and one
untriggered WAIT-only task. Train and eval retain their source splits and each
has a 1:1 immediate/delayed ratio among answerable tasks. P3's earlier Stage 3
candidates are replaced by the six reviewed tasks. The separate legacy lung
candidates remain labeled as legacy and are not included in this batch count.

Select Stage 3, then a task and a decision time. The page shows the causal input
prefix, WAIT/ANSWER supervision and waiting reason. Reference answers and
original-resolution evidence are in a collapsed review panel. These are labels,
not live model predictions; the player is an unrestricted review tool.

The batch used original-resolution sampled frames and dense one-second checks
at event boundaries. Review was by the same assistant, not an independent
clinical reviewer. English translation preserves the reviewed times/evidence.
The per-video JSONL files in data/stage3_batch include demo_decisions for display.
Use the canonical train/eval exports for training; do not feed gold metadata,
reference answers or future evidence to the model.


## Reviewed Stage 2 presentation examples (2026-11-05)

The five cards at the top load frozen, visually accepted data for P3FoSFIrfQA,
8V649L5Q368, AH8RcdMQMHA, 9vW1uo7mKDc and 466cDoLdVZQ: 60 block records,
468 supervision targets, four train videos and one independent eval video.
Each card opens a selected minute and a specific 10-second Local. Change the
Local selector, or expand all six; Block uses that minute and Global its prefix.
The kidney example also demonstrates a partial final minute: only complete
10-second Local targets exist there; no Block/Global is synthesized.

The source snapshots are in `viz/data/stage2_reviewed_20261105/`, with their
acceptance manifests, exact-source video hashes and frozen-release provenance.
The previews are copied from original-size frames that were actually inspected;
they are navigation aids. The model-training image size remains a separate setting.
These are supervision annotations, not student reconstructions or clinical expert
adjudications. Stage 1 and Stage 3 retain their existing source records.

Presentation bookmarks now preserve video, stage, block and Local selection in
the URL, for example:
`http://localhost:8000/viz/index.html?video=P3FoSFIrfQA&stage=stage2&sample=1&local=3`.


## English / Chinese display (2026-11-05)

The header language selector covers UI labels, Stage 1 target sentences and
historical ASR, Stage 2 Local/Block/Global supervision, and Stage 3 questions,
answers, trigger rules and waiting reasons. Chinese text is a display translation;
canonical English records, times, masking, source hashes and training exports
are unchanged. Chinese paragraphs offer a collapsed English-source disclosure.
Historical ASR is translated from the exact displayed 4000-character suffix;
truncated prefixes are not reconstructed using hidden text. Unclear ASR wording
is marked rather than silently corrected.

The language choice is remembered locally. A bookmark's `lang=en` or `lang=zh`
overrides the stored preference. Switching retains video, stage, sample, Local
window, Stage 3 decision, playback position and review-panel disclosure state.

`i18n.js` contains interface translations. `content.zh.json` maps exact English
source strings to Chinese display text, including composed ASR histories. The
current catalog's long-text and QA fields have complete translation coverage.
When adding or revising source records, update this dictionary as well; unknown
strings fall back to their original text, never automatic machine translation.
The dictionary is independent of catalog generation and must be shipped with
`index.html`, `app.js`, `style.css` and `i18n.js`. No translation API is used.
