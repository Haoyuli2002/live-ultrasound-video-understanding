"""
Video clipping / segmentation wrapper for the clean QA pipeline.

Uses the ultrasound-aware, offline clipping in:
  scripts/video_clipping.py

Default method: `qwen_embed` -- extract original color frames (torchcodec, no
OpenCV) and embed them with Qwen/Qwen3-VL-Embedding-2B (same family as the SFT
base), cut on adjacent-frame cosine-similarity drops, then align to sentence
boundaries. Falls back to SSIM (OpenCV) if the embedding model / torchcodec /
GPU is unavailable. `ssim` / `framediff` / `histogram` remain available via
--visual-method.

Input:
  video.mp4
  {output_dir}/transcripts/{video_id}.json

Output:
  {output_dir}/clips/{video_id}_clips.json

Example:
  python QA/prepare/clipping.py \
    --video path/to/video.mp4 \
    --output-dir QA/results \
    --visual-method qwen_embed \
    --min-clip 30 --max-clip 240
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[2]
_SCRIPTS_DIR = _REPO_ROOT / "scripts"
sys.path.insert(0, str(_SCRIPTS_DIR))

from video_clipping import clip_video, DEFAULT_QWEN_EMBED_MODEL  # noqa: E402


def run_clipping(
    video: str | Path,
    *,
    output_dir: str | Path = "QA/results",
    transcript: str | Path | None = None,
    visual_method: str = "qwen_embed",
    sample_interval: float = 1.5,
    scene_threshold: float | None = None,
    min_scene_gap: float = 3.0,
    pause_gap: float = 0.8,
    tolerance: float = 5.0,
    min_clip: int = 30,
    max_clip: int = 240,
    resize: int = 256,
    save_trace: bool = False,
    qwen_embed_model: str = DEFAULT_QWEN_EMBED_MODEL,
    qwen_embed_device: str = "auto",
    qwen_embed_batch: int = 16,
    no_llm: bool = True,  # accepted for API compat with run_prepare; clipping is always no-LLM
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

    output = clip_video(
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
        qwen_embed_model=qwen_embed_model,
        qwen_embed_device=qwen_embed_device,
        qwen_embed_batch=qwen_embed_batch,
    )

    print(f"[clipping] saved {output.get('num_clips')} clips -> {clips_dir / (video_id + '_clips.json')}")
    return output


def main():
    parser = argparse.ArgumentParser(description="QA clean video clipping wrapper (ultrasound-aware, no LLM)")
    parser.add_argument("--video", type=str, required=True)
    parser.add_argument("--output-dir", type=str, default="QA/results")
    parser.add_argument("--transcript", type=str, default=None)

    parser.add_argument("--visual-method", type=str, default="qwen_embed",
                        choices=["qwen_embed", "ssim", "framediff", "histogram"])
    parser.add_argument("--sample-interval", type=float, default=1.5)
    parser.add_argument("--scene-threshold", type=float, default=None,
                        help="Default: 0.85 (qwen_embed cosine) / 0.6 (ssim).")
    parser.add_argument("--min-scene-gap", type=float, default=3.0)
    parser.add_argument("--pause-gap", type=float, default=0.8)
    parser.add_argument("--tolerance", type=float, default=5.0)
    parser.add_argument("--min-clip", type=int, default=30)
    parser.add_argument("--max-clip", type=int, default=240)
    parser.add_argument("--resize", type=int, default=256)
    parser.add_argument("--save-trace", action="store_true")

    parser.add_argument("--qwen-embed-model", type=str, default=DEFAULT_QWEN_EMBED_MODEL)
    parser.add_argument("--qwen-embed-device", type=str, default="auto")
    parser.add_argument("--qwen-embed-batch", type=int, default=16)
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
        qwen_embed_model=args.qwen_embed_model,
        qwen_embed_device=args.qwen_embed_device,
        qwen_embed_batch=args.qwen_embed_batch,
    )


if __name__ == "__main__":
    main()