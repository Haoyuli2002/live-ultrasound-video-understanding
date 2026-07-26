"""
ASR wrapper for the clean QA pipeline.

This file intentionally reuses the legacy implementation in:
  scripts/asr_pipeline.py

Output:
  {output_dir}/transcripts/{video_id}.json

Example:
  python QA/prepare/asr.py \
    --video path/to/video.mp4 \
    --output-dir QA/results \
    --model base
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[2]
_SCRIPTS_DIR = _REPO_ROOT / "scripts"
sys.path.insert(0, str(_SCRIPTS_DIR))

from asr_pipeline import transcribe_video, resolve_device, resolve_model  # noqa: E402


def run_asr(
    video: str | Path,
    *,
    output_dir: str | Path = "QA/results",
    model: str | None = None,
    device: str = "auto",
    language: str | None = None,
    keep_audio: bool = False,
):
    output_dir = Path(output_dir)
    transcript_dir = output_dir / "transcripts"
    transcript_dir.mkdir(parents=True, exist_ok=True)

    resolved_device = resolve_device(device)
    resolved_model = resolve_model(model, resolved_device)
    print(f"[asr] Using model={resolved_model}, device={resolved_device}")

    return transcribe_video(
        str(video),
        model_size=resolved_model,
        language=language,
        output_dir=str(transcript_dir),
        keep_audio=keep_audio,
        device=resolved_device,
    )


def main():
    parser = argparse.ArgumentParser(description="QA clean ASR wrapper")
    parser.add_argument("--video", type=str, required=True)
    parser.add_argument("--output-dir", type=str, default="QA/results")
    parser.add_argument("--model", type=str, default=None,
                        help="Whisper model size: tiny/base/small/medium/large-v3. "
                             "If omitted: large-v3 on GPU, medium on CPU.")
    parser.add_argument("--device", type=str, default="auto",
                        help="auto | cpu | cuda. Default auto-detects GPU.")
    parser.add_argument("--language", type=str, default=None)
    parser.add_argument("--keep-audio", action="store_true")
    args = parser.parse_args()

    run_asr(
        args.video,
        output_dir=args.output_dir,
        model=args.model,
        device=args.device,
        language=args.language,
        keep_audio=args.keep_audio,
    )


if __name__ == "__main__":
    main()