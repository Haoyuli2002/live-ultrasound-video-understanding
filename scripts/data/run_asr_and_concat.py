#!/usr/bin/env python3
"""Run ASR for a small video set and concatenate full transcripts.

This script is intended for prompt/debug inspection, e.g. transcribe 3 videos
locally and produce a human-readable markdown file containing the complete ASR.

Examples:

  python scripts/data/run_asr_and_concat.py \
    --video azure_data/videos/8V649L5Q368.mp4 \
    --output-dir local_asr_3videos \
    --model base \
    --language en

  python scripts/data/run_asr_and_concat.py \
    --input-dir cluster_data/videos/train_full295 \
    --max-videos 3 \
    --output-dir local_asr_3videos \
    --model base \
    --language en
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, Iterable, List


REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

from QA.prepare.asr import run_asr  # noqa: E402


VIDEO_EXTS = {".mp4", ".webm", ".mkv", ".avi", ".mov"}


def discover_videos(input_dir: Path) -> List[Path]:
    videos = [p for p in input_dir.rglob("*") if p.is_file() and p.suffix.lower() in VIDEO_EXTS]
    return sorted(videos)


def load_transcript(path: Path) -> Dict[str, Any]:
    with path.open(encoding="utf-8") as f:
        return json.load(f)


def format_time(sec: float | int | None) -> str:
    if sec is None:
        return "0.00"
    return f"{float(sec):.2f}"


def transcript_to_markdown(data: Dict[str, Any]) -> str:
    video_id = data.get("video_id") or "unknown"
    lines = [
        f"## {video_id}",
        "",
        "Metadata:",
        f"- video_path: `{data.get('video_path', '')}`",
        f"- model: `{data.get('model', '')}`",
        f"- language: `{data.get('language', '')}`",
        f"- language_probability: `{data.get('language_probability', '')}`",
        f"- duration_sec: `{data.get('duration_sec', '')}`",
        f"- num_segments: `{data.get('num_segments', len(data.get('segments', [])))}`",
        "",
        "Transcript:",
        "",
    ]
    for seg in data.get("segments", []):
        start = format_time(seg.get("start"))
        end = format_time(seg.get("end"))
        text = (seg.get("text") or "").strip()
        lines.append(f"[{start} - {end}] {text}")
    lines.append("")
    return "\n".join(lines)


def write_concat_outputs(transcript_paths: Iterable[Path], output_dir: Path) -> None:
    transcript_paths = list(transcript_paths)
    md_path = output_dir / "full_transcripts.md"
    jsonl_path = output_dir / "full_transcripts.jsonl"

    md_parts = ["# Full ASR Transcripts", ""]
    with jsonl_path.open("w", encoding="utf-8") as jf:
        for path in transcript_paths:
            data = load_transcript(path)
            md_parts.append(transcript_to_markdown(data))
            jf.write(json.dumps(data, ensure_ascii=False) + "\n")

    md_path.write_text("\n".join(md_parts), encoding="utf-8")
    print(f"[concat] wrote markdown: {md_path}")
    print(f"[concat] wrote jsonl    : {jsonl_path}")


def collect_requested_videos(args: argparse.Namespace) -> List[Path]:
    videos = [Path(v) for v in args.video]
    if args.input_dir:
        videos.extend(discover_videos(Path(args.input_dir)))

    # Preserve order while deduplicating by resolved path where possible.
    out: List[Path] = []
    seen: set[str] = set()
    for p in videos:
        key = str(p.resolve()) if p.exists() else str(p)
        if key in seen:
            continue
        seen.add(key)
        out.append(p)

    if args.max_videos is not None:
        out = out[: args.max_videos]
    return out


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run ASR and concatenate full transcripts")
    parser.add_argument("--video", action="append", default=[], help="Video path; can be repeated")
    parser.add_argument("--input-dir", type=str, default=None, help="Directory of videos")
    parser.add_argument("--max-videos", type=int, default=None)
    parser.add_argument("--output-dir", type=str, default="local_asr_3videos")
    parser.add_argument("--model", type=str, default="base")
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--language", type=str, default=None)
    parser.add_argument("--keep-audio", action="store_true")
    parser.add_argument("--overwrite", action="store_true", help="Re-run ASR even if transcript exists")
    parser.add_argument("--concat-only", action="store_true", help="Only rebuild full_transcripts.* from existing transcript JSON files")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)
    transcript_dir = output_dir / "transcripts"
    transcript_dir.mkdir(parents=True, exist_ok=True)

    videos = collect_requested_videos(args)
    if not videos and not args.concat_only:
        raise ValueError("Provide --video and/or --input-dir, or use --concat-only with existing transcripts")

    print("=" * 80)
    print("[asr-concat] output_dir:", output_dir)
    print("[asr-concat] videos:", len(videos))
    print("[asr-concat] model:", args.model)
    print("[asr-concat] device:", args.device)
    print("=" * 80)

    transcript_paths: List[Path] = []
    if not args.concat_only:
        for idx, video in enumerate(videos, start=1):
            if not video.exists():
                print(f"[asr-concat] skip missing video: {video}")
                continue
            transcript_path = transcript_dir / f"{video.stem}.json"
            if transcript_path.exists() and not args.overwrite:
                print(f"[asr-concat] skip existing {idx}/{len(videos)}: {transcript_path}")
            else:
                print(f"[asr-concat] ASR {idx}/{len(videos)}: {video}")
                run_asr(
                    video,
                    output_dir=output_dir,
                    model=args.model,
                    device=args.device,
                    language=args.language,
                    keep_audio=args.keep_audio,
                )
            if transcript_path.exists():
                transcript_paths.append(transcript_path)

    existing = sorted(transcript_dir.glob("*.json"))
    if args.concat_only:
        transcript_paths = existing
    else:
        # Include any existing transcripts in the same output dir for a complete
        # concatenated file, while preserving current-run order first.
        seen = {p.name for p in transcript_paths}
        transcript_paths.extend(p for p in existing if p.name not in seen)

    if not transcript_paths:
        raise RuntimeError(f"No transcripts found in {transcript_dir}")

    write_concat_outputs(transcript_paths, output_dir)


if __name__ == "__main__":
    main()