"""
Video clipping / segmentation wrapper for the clean QA pipeline.

Uses the ultrasound-aware, offline segmentation in:
  scripts/video_segmentation.py

Default method: SSIM on a fixed time grid + sentence-boundary alignment
(no LLM). Falls back to framediff if scikit-image is unavailable; legacy
histogram method is available via --visual-method histogram.

Input:
  video.mp4
  {output_dir}/transcripts/{video_id}.json

Output:
  {output_dir}/clips/{video_id}_clips.json

Example:
  python QA/prepare/clipping.py \
    --video path/to/video.mp4 \
    --output-dir QA/results \
    --visual-method ssim \
    --min-clip 30 --max-clip 240
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[2]
_SCRIPTS_DIR = _REPO_ROOT / "scripts"
sys.path.insert(0, str(_SCRIPTS_DIR))

from video_segmentation import segment_video  # noqa: E402


def run_clipping(
    video: str | Path,
    *,
    output_dir: str | Path = "QA/results",
    transcript: str | Path | None = None,
    visual_method: str = "ssim",
    sample_interval: float = 1.5,
    scene_threshold: float = 0.6,
    min_scene_gap: float = 3.0,
    pause_gap: float = 0.8,
    tolerance: float = 5.0,
    min_clip: int = 30,
    max_clip: int = 240,
    resize: int = 256,
    save_trace: bool = False,
):
    video = Path(video)
    video_id = video.stem
    output_dir = Path(output_dir)

    if transcript is None:
        transcript = output_dir / "transcripts" / f"{video_id}.json"
    transcript = Path(transcript)

    if not transcript.exists():
        raise FileNotFoundError(f"Transcript not found: {transcript}")

    clips_dir = output_dir / "clips"
    clips_dir.mkdir(parents=True, exist_ok=True)

    output = segment_video(
        str(video),
        str(transcript),
        output_dir=str(clips_dir),
        visual_method=visual_method,
        sample_interval=sample_interval,
        scene_threshold=scene_threshold,
        min_scene_gap=min_scene_gap,
        pause_gap=pause_gap,
        tolerance=tolerance,
        min_clip=min_clip,
        max_clip=max_clip,
        resize=resize,
        save_trace=save_trace,
    )

    print(f"[clipping] saved {output.get('num_clips')} clips -> {clips_dir / (video_id + '_clips.json')}")
    return output


def main():
    parser = argparse.ArgumentParser(description="QA clean video clipping wrapper (ultrasound-aware, no LLM)")
    parser.add_argument("--video", type=str, required=True)
    parser.add_argument("--output-dir", type=str, default="QA/results")
    parser.add_argument("--transcript", type=str, default=None)

    parser.add_argument("--visual-method", type=str, default="ssim",
                        choices=["ssim", "framediff", "histogram"])
    parser.add_argument("--sample-interval", type=float, default=1.5)
    parser.add_argument("--scene-threshold", type=float, default=0.6)
    parser.add_argument("--min-scene-gap", type=float, default=3.0)
    parser.add_argument("--pause-gap", type=float, default=0.8)
    parser.add_argument("--tolerance", type=float, default=5.0)
    parser.add_argument("--min-clip", type=int, default=30)
    parser.add_argument("--max-clip", type=int, default=240)
    parser.add_argument("--resize", type=int, default=256)
    parser.add_argument("--save-trace", action="store_true")
    args = parser.parse_args()

    run_clipping(
        args.video,
        output_dir=args.output_dir,
        transcript=args.transcript,
        visual_method=args.visual_method,
        sample_interval=args.sample_interval,
        scene_threshold=args.scene_threshold,
        min_scene_gap=args.min_scene_gap,
        pause_gap=args.pause_gap,
        tolerance=args.tolerance,
        min_clip=args.min_clip,
        max_clip=args.max_clip,
        resize=args.resize,
        save_trace=args.save_trace,
    )


if __name__ == "__main__":
    main()