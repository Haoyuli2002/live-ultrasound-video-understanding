"""Generate visual-only 10s, 60s and prefix summaries from silent video clips.

Uses an OpenAI-compatible multimodal Chat Completions endpoint. The endpoint
decides how video is decoded and sampled internally. No request is sent until
this CLI is explicitly run with a model and API credentials.
"""
from __future__ import annotations

import argparse
import base64
import json
import os
import subprocess
import tempfile
from pathlib import Path

from .data import VideoReader

SYSTEM = (
    "You annotate ultrasound video using only the supplied silent video. "
    "Describe visible anatomy, view, probe movement, measurements, temporal "
    "changes and clinically relevant visual evidence. Do not infer an unseen "
    "diagnosis, use audio/transcripts, or mention events outside the requested "
    "interval. Return a concise factual summary without a preamble."
)


def make_silent_clip(video_path: Path, start: int, end: int, output: Path) -> None:
    """Re-encode an exact half-open window without its audio track."""
    if start < 0 or end <= start:
        raise ValueError("Invalid teacher video window")
    command = ["ffmpeg", "-y", "-loglevel", "error", "-ss", str(start),
               "-i", str(video_path), "-t", str(end-start), "-an",
               "-c:v", "libx264", "-preset", "veryfast", "-crf", "23",
               str(output)]
    result = subprocess.run(command, capture_output=True, text=True)
    if result.returncode or not output.is_file() or output.stat().st_size == 0:
        raise RuntimeError(f"Failed to make silent clip [{start},{end}): {result.stderr[:400]}")


def summarize_window(video_path: Path, teacher, *, kind: str,
                     start: int, end: int, clip_path: Path) -> str:
    make_silent_clip(video_path, start, end, clip_path)
    focus = {
        "current 10-second local": "Preserve brief visible findings and changes within these ten seconds.",
        "independent current 60-second block": "Integrate this minute as a whole; do not merely concatenate ten-second descriptions.",
        "cumulative video prefix": "Summarize the important visual evidence observed since video start, including earlier observations worth retaining.",
    }.get(kind, "")
    prompt = (
        f"Summarize only the visible evidence in the {kind} interval "
        f"[{start}, {end}) seconds. The attached silent video covers exactly "
        f"that interval. {focus} "
        "Do not use information from other times."
    )
    summary = teacher.summary(prompt, clip_path).strip()
    if not summary:
        raise RuntimeError(f"Teacher returned an empty {kind} summary for [{start},{end})")
    return summary


def annotate_video(video_path: Path, teacher, *, video_id: str,
                   duration_sec: int, start_block: int = 0,
                   clip_dir: Path | None = None):
    """Yield one atomic event batch per block (eight records per full minute)."""
    for block_start in range(start_block, duration_sec, 60):
        block_end = min(block_start+60, duration_sec)
        block_window = [block_start, block_end]
        events = []
        with tempfile.TemporaryDirectory(prefix="stage2_teacher_", dir=clip_dir) as temporary:
            temporary_path = Path(temporary)
            def label(kind: str, start: int, end: int, name: str) -> str:
                return summarize_window(video_path, teacher, kind=kind,
                                        start=start, end=end,
                                        clip_path=temporary_path / f"{name}.mp4")
            for start in range(block_start, block_end-9, 10):
                end = start+10
                events.append({
                    "video_id": video_id,
                    "block_window": block_window,
                    "label_type": "local_10s",
                    "window": [start, end],
                    "target": label("current 10-second local", start, end,
                                    f"local_{start}_{end}"),
                })
            if block_end-block_start == 60:
                events.append({
                    "video_id": video_id,
                    "block_window": block_window,
                    "label_type": "block_60s",
                    "window": [block_start, block_end],
                    "target": label("independent current 60-second block",
                                    block_start, block_end, "block"),
                })
                events.append({
                    "video_id": video_id,
                    "block_window": block_window,
                    "label_type": "global_prefix",
                    "window": [0, block_end],
                    "target": label("cumulative video prefix", 0, block_end,
                                    "global"),
                })
        if block_end-block_start < 60 and not events:
            break
        yield events


class ChatTeacher:
    def __init__(self, model: str, *, api_key: str, base_url: str | None,
                 max_tokens: int, transport: str = "auto", video_fps: float = 0.0):
        from openai import OpenAI
        self.client = OpenAI(api_key=api_key, base_url=base_url) if base_url else OpenAI(api_key=api_key)
        self.model = model
        self.max_tokens = max_tokens
        self.transport = ("file" if "gemini" in model.lower() else "video_url") if transport == "auto" else transport
        self.video_fps = video_fps

    def summary(self, prompt: str, video_path: Path) -> str:
        if self.transport == "file":
            encoded = base64.b64encode(video_path.read_bytes()).decode("ascii")
            video = {"type": "file", "file": {
                "filename": video_path.name,
                "file_data": f"data:video/mp4;base64,{encoded}"}}
        elif self.transport == "video_url":
            video = {"type": "video_url", "video_url": {"url": video_path.resolve().as_uri()}}
        else:
            raise ValueError(f"Unknown video transport: {self.transport}")
        request = dict(
            model=self.model,
            messages=[{"role": "system", "content": SYSTEM},
                      {"role": "user", "content": [video, {"type": "text", "text": prompt}]}],
            max_tokens=self.max_tokens,
            temperature=0,
        )
        if self.transport == "video_url" and self.video_fps > 0:
            request["extra_body"] = {"mm_processor_kwargs": {"fps": self.video_fps}}
        response = self.client.chat.completions.create(**request)
        return response.choices[0].message.content or ""


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--video", required=True)
    parser.add_argument("--video-id", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--model", required=True,
                        help="Vision model on an OpenAI-compatible Chat Completions endpoint")
    parser.add_argument("--base-url", help="Optional compatible endpoint URL")
    parser.add_argument("--api-key-env", default="OPENAI_API_KEY")
    parser.add_argument("--transport", choices=("auto", "video_url", "file"), default="auto",
                        help="auto: Gemini uses OpenRouter file; other models use vLLM video_url")
    parser.add_argument("--video-fps", type=float, default=0.0,
                        help="Optional vLLM video decoding FPS; default leaves backend policy unchanged")
    parser.add_argument("--clip-dir", type=Path,
                        help="Parent for temporary clips; local vLLM must be allowed to read it")
    parser.add_argument("--max-tokens", type=int, default=500)
    parser.add_argument("--resume", action="store_true",
                        help="Append after the last complete saved block")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    api_key = os.environ.get(args.api_key_env)
    if not api_key:
        raise ValueError(f"Set {args.api_key_env} before running teacher annotation")
    teacher = ChatTeacher(args.model, api_key=api_key,
                          base_url=args.base_url, max_tokens=args.max_tokens,
                          transport=args.transport, video_fps=args.video_fps)
    if args.clip_dir is not None:
        args.clip_dir.mkdir(parents=True, exist_ok=True)
    reader = VideoReader(args.video, args.frame_size)
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    try:
        duration_sec = int(reader.count / reader.fps)
        if duration_sec < 10:
            raise ValueError("At least 10 complete seconds are needed")
        if args.resume and args.overwrite:
            raise ValueError("Choose --resume or --overwrite, not both")
        start_block = 0
        if output.exists() and args.resume:
            from .build_data import build_rows
            records = [json.loads(line) for line in output.read_text(encoding="utf-8").splitlines()
                       if line.strip()]
            if not records:
                raise ValueError("Cannot resume an empty teacher-label file")
            if any(row.get("label_type") is None for row in records):
                raise ValueError("Resume expects eight-record-per-minute event labels")
            existing = sorted(build_rows(records), key=lambda row: row["block_window"][0])
            if {row["video_id"] for row in existing} != {args.video_id}:
                raise ValueError("Saved labels belong to a different video")
            if any(row.get("teacher") != args.model for row in records):
                raise ValueError("Saved labels were produced by a different teacher model")
            for index, row in enumerate(existing):
                if row["block_window"] != [60 * index, 60 * (index + 1)]:
                    raise ValueError("Saved minute events have a gap or incomplete block")
                expected = [[60*index + 10*j, 60*index + 10*(j+1)] for j in range(6)]
                if [item["window"] for item in row["local_sub_summaries"]] != expected:
                    raise ValueError("Saved minute does not contain six ordered 10-second labels")
            start_block = int(existing[-1]["block_window"][1])
            if start_block % 60:
                raise ValueError("Cannot resume after an incomplete final block")
        elif output.exists() and not args.overwrite:
            raise FileExistsError(f"Output already exists: {output}; use --resume or --overwrite")
        # Complete all eight teacher calls before writing a minute's records.
        mode = "a" if output.exists() and args.resume else "w"
        with output.open(mode, encoding="utf-8") as stream:
            for events in annotate_video(Path(args.video), teacher, video_id=args.video_id,
                                         duration_sec=duration_sec,
                                         start_block=start_block, clip_dir=args.clip_dir):
                for event in events:
                    event["teacher"] = args.model
                stream.write("".join(json.dumps(event, ensure_ascii=False) + "\n"
                                     for event in events))
                stream.flush()
    finally:
        reader.close()


if __name__ == "__main__":
    main()
