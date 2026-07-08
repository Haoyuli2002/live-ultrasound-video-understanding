"""
End-to-end QA driver
====================
Runs the new QA pipeline on a single video:

    generator  ->  validator  ->  merger

Prerequisite artefacts (produced by the OLD pipeline in scripts/):
  - results/transcripts/{video_id}.json           (from scripts/asr_pipeline.py)
  - results/clips/{video_id}_clips.json           (from scripts/video_segmentation.py)
  - results/qa/{video_id}_offline_qa.json         (from scripts/qa_generation.py; optional)

New outputs (under QA/results/):
  - {video_id}_streaming_qa.json
  - {video_id}_streaming_qa_validated.json
  - {video_id}.jsonl                (per-video record)
  - {video_id}_training_samples.jsonl  (optional, with --expand-wait-answer)

Usage:
    python QA/run.py --video path/to/ID.mp4
"""

from __future__ import annotations

import sys
import argparse
from pathlib import Path

_HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE))
from _shared import DEFAULT_MODEL  # noqa: E402

# We import the per-step entry points as functions so we don't
# have to shell out to python -m each time.
from generator import (  # noqa: E402
    generate_streaming_qa_for_video,
    SEEN_WINDOW_SEC,
    FUTURE_WINDOW_SEC,
    TIME_RATIOS,
)
from offline_generator import (  # noqa: E402
    generate_offline_qa_for_video,
    CLIP_MAX_SEC,
)
from validator import (  # noqa: E402
    validate_streaming_qa_file,
    BEFORE_WINDOW_SEC,
    EVIDENCE_WINDOW_SEC,
    AFTER_WINDOW_SEC,
)
from merger import (  # noqa: E402
    build_per_video_record,
    expand_training_samples,
    DEFAULT_WINDOW_SEC,
)

import json


def _write_jsonl(path, records, overwrite=True):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    mode = 'w' if overwrite else 'a'
    with open(path, mode, encoding='utf-8') as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def run(video_path,
        clips_path=None,
        transcript_path=None,
        offline_qa_path=None,
        out_dir="QA/results",
        api_key=None,
        model=DEFAULT_MODEL,
        ratios=None,
        single_clip=None,
        clip_max_sec=CLIP_MAX_SEC,
        seen_window_sec=SEEN_WINDOW_SEC,
        future_window_sec=FUTURE_WINDOW_SEC,
        before_window_sec=BEFORE_WINDOW_SEC,
        evidence_window_sec=EVIDENCE_WINDOW_SEC,
        after_window_sec=AFTER_WINDOW_SEC,
        drop_failed=True,
        max_qa=None,
        window_sec=DEFAULT_WINDOW_SEC,
        expand_wait_answer=False,
        skip_offline=False,
        skip_generation=False,
        skip_validation=False,
        skip_merge=False):
    """Run the end-to-end QA pipeline on one video.

    Pipeline order:
        1. offline_generator   -> QA/results/{id}_offline_qa.json
        2. streaming generator -> QA/results/{id}_streaming_qa.json
        3. validator           -> QA/results/{id}_streaming_qa_validated.json
        4. merger              -> QA/results/{id}.jsonl (+ training samples)
    """
    video_path = Path(video_path)
    video_id = video_path.stem
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    if clips_path is None:
        clips_path = Path("results/clips") / f"{video_id}_clips.json"
    if transcript_path is None:
        transcript_path = Path("results/transcripts") / f"{video_id}.json"

    new_offline_qa_path = out_dir / f"{video_id}_offline_qa.json"
    # If caller didn't force a path, decide dynamically after step 0.
    # The `offline_qa_path` variable that ends up feeding the merger will be
    # set below AFTER we know whether the new offline generator succeeded /
    # was skipped.

    streaming_qa_path = out_dir / f"{video_id}_streaming_qa.json"
    streaming_qa_validated_path = out_dir / f"{video_id}_streaming_qa_validated.json"
    merged_record_path = out_dir / f"{video_id}.jsonl"
    training_samples_path = out_dir / f"{video_id}_training_samples.jsonl"

    print("\n" + "#" * 72)
    print(f"# QA pipeline (new): {video_id}")
    print("#" * 72)
    print(f"  video           : {video_path}")
    print(f"  clips           : {clips_path}")
    print(f"  transcript      : {transcript_path}")
    print(f"  offline QA (new): {new_offline_qa_path}")
    print(f"  streaming QA    : {streaming_qa_path}")
    print(f"  validated       : {streaming_qa_validated_path}")
    print(f"  merged record   : {merged_record_path}")
    print(f"  training samples: {training_samples_path if expand_wait_answer else '(off)'}")

    # -----------------------------------------------------------------------
    # Step 0: offline generator (new)
    # -----------------------------------------------------------------------
    if skip_offline and new_offline_qa_path.exists():
        print(f"\n[step 0] --skip-offline and {new_offline_qa_path} exists, skipping.")
    elif skip_offline:
        print(f"\n[step 0] --skip-offline set, skipping offline QA generation.")
    else:
        generate_offline_qa_for_video(
            str(video_path),
            str(clips_path),
            output_dir=str(out_dir),
            api_key=api_key,
            single_clip=single_clip,
            model=model,
            clip_max_sec=clip_max_sec,
        )

    # Decide which offline QA file to feed to the merger.
    # Priority: (a) explicit --offline-qa flag -> (b) newly generated -> (c) legacy scripts/qa_generation.py output
    if offline_qa_path is None:
        if new_offline_qa_path.exists():
            offline_qa_path = new_offline_qa_path
        else:
            legacy_cand = Path("results/qa") / f"{video_id}_offline_qa.json"
            if legacy_cand.exists():
                offline_qa_path = legacy_cand
                print(f"  (fallback) using legacy offline QA: {legacy_cand}")

    # -----------------------------------------------------------------------
    # Step 1: streaming generator
    # -----------------------------------------------------------------------
    if skip_generation and streaming_qa_path.exists():
        print(f"\n[step 1] --skip-generation and {streaming_qa_path} exists, skipping.")
    else:
        generate_streaming_qa_for_video(
            str(video_path),
            str(clips_path),
            output_dir=str(out_dir),
            api_key=api_key,
            single_clip=single_clip,
            time_ratios=ratios,
            model=model,
            seen_window_sec=seen_window_sec,
            future_window_sec=future_window_sec,
        )

    # -----------------------------------------------------------------------
    # Step 2: validator
    # -----------------------------------------------------------------------
    if skip_validation and streaming_qa_validated_path.exists():
        print(f"\n[step 2] --skip-validation and {streaming_qa_validated_path} exists, skipping.")
    else:
        validate_streaming_qa_file(
            str(streaming_qa_path),
            str(video_path),
            output_path=str(streaming_qa_validated_path),
            api_key=api_key,
            model=model,
            drop_failed=drop_failed,
            before_window_sec=before_window_sec,
            evidence_window_sec=evidence_window_sec,
            after_window_sec=after_window_sec,
            max_qa=max_qa,
        )

    # -----------------------------------------------------------------------
    # Step 3: merger
    # -----------------------------------------------------------------------
    if skip_merge:
        print("\n[step 3] --skip-merge set, skipping merger.")
        return

    print("\n[step 3] merge -> per-video record")
    record = build_per_video_record(
        video_id,
        str(transcript_path),
        str(clips_path),
        offline_qa_path=str(offline_qa_path) if offline_qa_path else None,
        streaming_qa_path=str(streaming_qa_validated_path),
    )
    _write_jsonl(merged_record_path, [record], overwrite=True)
    print(f"  wrote per-video record: {merged_record_path}")
    print(f"  num QA: {record['num_qa']} | breakdown: {record['qa_type_counts']}")

    if expand_wait_answer:
        print("\n[step 3b] merge -> training samples (WAIT + ANSWER)")
        samples = expand_training_samples(
            video_id,
            str(transcript_path),
            str(clips_path),
            offline_qa_path=str(offline_qa_path) if offline_qa_path else None,
            streaming_qa_path=str(streaming_qa_validated_path),
            window_sec=window_sec,
        )
        _write_jsonl(training_samples_path, samples, overwrite=True)

        type_counts = {}
        for s in samples:
            type_counts[s['sample_type']] = type_counts.get(s['sample_type'], 0) + 1
        print(f"  wrote training samples: {training_samples_path}")
        print(f"  total samples: {len(samples)} | breakdown: {type_counts}")


def main():
    parser = argparse.ArgumentParser(
        description="End-to-end QA pipeline (generator -> validator -> merger)"
    )
    parser.add_argument("--video", type=str, required=True,
                        help="Path to the source video (.mp4)")
    parser.add_argument("--clips", type=str, default=None,
                        help="Clips JSON (default: results/clips/{id}_clips.json)")
    parser.add_argument("--transcript", type=str, default=None,
                        help="ASR JSON (default: results/transcripts/{id}.json)")
    parser.add_argument("--offline-qa", type=str, default=None,
                        help="Offline QA JSON (default: results/qa/{id}_offline_qa.json if exists)")
    parser.add_argument("--out-dir", type=str, default="QA/results")
    parser.add_argument("--api-key", type=str, default=None)
    parser.add_argument("--model", type=str, default=DEFAULT_MODEL)

    parser.add_argument("--ratios", type=str, default=None,
                        help="Comma-separated time ratios, e.g. '0.25,0.5,0.75'")
    parser.add_argument("--single-clip", type=int, default=None)

    parser.add_argument("--seen-window-sec", type=float, default=SEEN_WINDOW_SEC)
    parser.add_argument("--future-window-sec", type=float, default=-1,
                        help="Cap FUTURE segment in generator (default -1 = uncapped)")

    parser.add_argument("--before-window-sec", type=float, default=BEFORE_WINDOW_SEC)
    parser.add_argument("--evidence-window-sec", type=float, default=-1,
                        help="Cap EVIDENCE_SPAN in validator (default -1 = uncapped)")
    parser.add_argument("--after-window-sec", type=float, default=AFTER_WINDOW_SEC)

    parser.add_argument("--keep-failed", action="store_true",
                        help="Keep failed streaming QA in validated output")
    parser.add_argument("--max-qa", type=int, default=None,
                        help="Cap validator to first N QA (smoke test)")

    parser.add_argument("--expand-wait-answer", action="store_true",
                        help="Also emit WAIT/ANSWER training samples")
    parser.add_argument("--window-sec", type=float, default=DEFAULT_WINDOW_SEC,
                        help="Observation window for WAIT/ANSWER samples")

    parser.add_argument("--skip-offline", action="store_true",
                        help="Skip offline QA generation (reuses existing output if present)")
    parser.add_argument("--skip-generation", action="store_true")
    parser.add_argument("--skip-validation", action="store_true")
    parser.add_argument("--skip-merge", action="store_true")

    parser.add_argument("--clip-max-sec", type=float, default=CLIP_MAX_SEC,
                        help=f"Cap clip length for offline generator (default {CLIP_MAX_SEC}s)")

    args = parser.parse_args()

    ratios = None
    if args.ratios:
        ratios = [float(x) for x in args.ratios.split(',')]

    seen_w = args.seen_window_sec if args.seen_window_sec and args.seen_window_sec > 0 else None
    fut_w = None if (args.future_window_sec is None or args.future_window_sec <= 0) else args.future_window_sec
    before_w = args.before_window_sec if args.before_window_sec and args.before_window_sec > 0 else None
    ev_w = None if (args.evidence_window_sec is None or args.evidence_window_sec <= 0) else args.evidence_window_sec
    after_w = args.after_window_sec if args.after_window_sec and args.after_window_sec > 0 else AFTER_WINDOW_SEC

    run(
        args.video,
        clips_path=args.clips,
        transcript_path=args.transcript,
        offline_qa_path=args.offline_qa,
        out_dir=args.out_dir,
        api_key=args.api_key,
        model=args.model,
        ratios=ratios,
        single_clip=args.single_clip,
        clip_max_sec=args.clip_max_sec,
        seen_window_sec=seen_w,
        future_window_sec=fut_w,
        before_window_sec=before_w,
        evidence_window_sec=ev_w,
        after_window_sec=after_w,
        drop_failed=not args.keep_failed,
        max_qa=args.max_qa,
        window_sec=args.window_sec,
        expand_wait_answer=args.expand_wait_answer,
        skip_offline=args.skip_offline,
        skip_generation=args.skip_generation,
        skip_validation=args.skip_validation,
        skip_merge=args.skip_merge,
    )


if __name__ == "__main__":
    main()
