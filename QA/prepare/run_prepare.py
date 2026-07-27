"""
Clean preparation pipeline for QA.

Runs the pre-QA steps on a single video:

  Step 1: ASR transcription
  Step 2: Video clipping / segmentation

The output is the minimum input needed by QA generation:

  {output_dir}/transcripts/{video_id}.json
  {output_dir}/clips/{video_id}_clips.json

Example:
  python QA/prepare/run_prepare.py \
    --video path/to/video.mp4 \
    --output-dir QA/results \
    --whisper-model base
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

try:
    from .asr import run_asr
    from .clipping import run_clipping
except ImportError:
    from asr import run_asr
    from clipping import run_clipping


def run_prepare(
    video: str | Path,
    *,
    output_dir: str | Path = "QA/results",
    whisper_model: str = "base",
    language: str | None = None,
    skip_asr: bool = False,
    skip_clipping: bool = False,
    no_llm_clipping: bool = False,
):
    video = Path(video)
    video_id = video.stem
    output_dir = Path(output_dir)

    transcript_path = output_dir / "transcripts" / f"{video_id}.json"
    clips_path = output_dir / "clips" / f"{video_id}_clips.json"

    print("\n" + "=" * 72)
    print(f"QA PREPARE PIPELINE: {video_id}")
    print("=" * 72)
    print(f"  video      : {video}")
    print(f"  output_dir : {output_dir}")
    print(f"  transcript : {transcript_path}")
    print(f"  clips      : {clips_path}")

    t0 = time.time()

    if skip_asr and transcript_path.exists():
        print(f"\n[prepare:asr] skipped (exists: {transcript_path})")
    else:
        print(f"\n[prepare:asr] running Whisper model={whisper_model}")
        run_asr(
            video,
            output_dir=output_dir,
            model=whisper_model,
            language=language,
        )

    if not transcript_path.exists():
        raise FileNotFoundError(f"ASR transcript missing: {transcript_path}")

    if skip_clipping and clips_path.exists():
        print(f"\n[prepare:clipping] skipped (exists: {clips_path})")
    else:
        mode = "histogram-only" if no_llm_clipping else "histogram+LLM"
        print(f"\n[prepare:clipping] running {mode}")
        run_clipping(
            video,
            output_dir=output_dir,
            transcript=transcript_path,
            no_llm=no_llm_clipping,
        )

    if not clips_path.exists():
        raise FileNotFoundError(f"Clips JSON missing: {clips_path}")

    elapsed = time.time() - t0
    print("\n" + "=" * 72)
    print("QA PREPARE COMPLETE")
    print("=" * 72)
    print(f"  elapsed    : {elapsed / 60:.1f} min")
    print(f"  transcript : {transcript_path}")
    print(f"  clips      : {clips_path}")

    return {
        "video_id": video_id,
        "transcript_path": str(transcript_path),
        "clips_path": str(clips_path),
    }


def main():
    parser = argparse.ArgumentParser(description="Run clean QA data-preparation pipeline")
    parser.add_argument("--video", type=str, required=True)
    parser.add_argument("--output-dir", type=str, default="QA/results")
    parser.add_argument("--whisper-model", type=str, default="base")
    parser.add_argument("--language", type=str, default=None)
    parser.add_argument("--skip-asr", action="store_true")
    parser.add_argument("--skip-clipping", action="store_true")
    parser.add_argument("--no-llm-clipping", action="store_true")
    args = parser.parse_args()

    run_prepare(
        args.video,
        output_dir=args.output_dir,
        whisper_model=args.whisper_model,
        language=args.language,
        skip_asr=args.skip_asr,
        skip_clipping=args.skip_clipping,
        no_llm_clipping=args.no_llm_clipping,
    )


if __name__ == "__main__":
    main()