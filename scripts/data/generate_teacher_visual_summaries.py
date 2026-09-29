#!/usr/bin/env python3
"""Generate teacher visual summaries for one anchor ultrasound video.

Output is one JSONL row per complete block and is directly consumable by:

    pretrain/build_teacher_memory_summary_samples.py

For each complete block this script asks a video-capable teacher model for:
  1. local_summary_target / local_sub_summaries: visual summaries of the
     current block or denser local sub-windows;
  2. global_summary_target: a concise cumulative visual summary from 0 to T.

By default the global summary is generated incrementally from the previous
global summary plus the current minute clip. This is much cheaper and scales to
long videos. For strict anchor-video experiments, `--global-mode full_clip`
instead sends the full [0, T] clip for each block; use it only for short clips.

The prompts explicitly ask for visual ultrasound evidence and discourage using
audio/transcript information, matching the Stage-2 Summary Compression Stage
teacher-visual-summary objective.

By default, clips are also muted before being sent to the teacher. This prevents
audio narration from leaking into supposedly visual-only targets.
"""

from __future__ import annotations

import argparse
import json
import math
import re
import subprocess
import sys
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[2]
SCRIPTS_DIR = REPO_ROOT / "scripts"
sys.path.insert(0, str(SCRIPTS_DIR))

from _video_llm import (  # noqa: E402
    build_openrouter_client,
    build_video_block,
    call_with_content,
    cut_clip,
    temp_clip_path,
    text_block,
)


DEFAULT_TEACHER_SUMMARY_MODEL = "google/gemini-3.1-pro-preview"


LOCAL_SYSTEM_PROMPT = """You are an expert ultrasound image reviewer creating VISUAL-only memory targets
for training a streaming ultrasound video model.

Your task is to summarize the clinically useful visual evidence in the current
video segment.

Use only evidence that is directly supported by the visible ultrasound images
or by visually observable probe/scanning actions.

Prioritize:
- anatomy or organs visualized
- scan view or acquisition target
- relevant probe position or movement
- clearly visible sonographic findings
- meaningful motion or dynamic findings
- visible measurements or labels when clinically relevant
- meaningful temporal changes within the segment

Important rules:
- Prioritize ultrasound imaging content over generic scene description.
- Do not rely on audio narration, transcript content, teaching context, or expected anatomy.
- Do not mention patient demographics, room setup, clinicians, or equipment unless
  they are directly relevant to probe placement or scan acquisition.
- Mention image quality only when it materially limits interpretation.
- Do not mention the absence of labels or measurements unless clinically relevant.
- Do not interpret "not visible" as "absent".
- Report a negative finding only when the relevant anatomy has been adequately
  visualized and the negative observation is visually supportable.
- Report dynamic findings only when the corresponding motion is directly observable
  across video frames.
- Do not infer diagnoses or findings from medical knowledge alone.
- Do not add diagnostic interpretation beyond the directly visible evidence.
- If a finding is uncertain, omit it rather than speculate, unless the uncertainty
  itself is clinically important.
- Keep the summary concise, factual, and information-dense.
"""


GLOBAL_SYSTEM_PROMPT = """You are an expert ultrasound image reviewer maintaining a compact VISUAL-only
memory of a streaming ultrasound video.

The cumulative summary is a fixed-capacity memory state, not a transcript,
not an exhaustive description, and not a concatenation of previous summaries.

Your goal is to retain the most clinically useful DISTINCT visual evidence
observed so far.

Prioritize:
1. clearly visible sonographic findings and dynamic signs
2. important anatomy and scan views
3. clinically meaningful changes or comparisons over time
4. relevant measurements, labels, or machine settings
5. probe position or movement only when needed to understand a finding

Memory update:
- Preserve important earlier evidence.
- Integrate important new evidence from the current segment.
- Merge repeated observations and remove redundancy.
- Rewrite and compress the whole memory instead of appending new text.
- Remove low-value setup details when more important ultrasound evidence appears.
- Keep the information volume approximately constant as the video becomes longer.

Temporal changes:
- Preserve changes when the transition itself is clinically meaningful.
- Do not overwrite an earlier finding when the change from earlier to later matters.
- If a later observation only confirms an earlier finding, merge them.
- Distinguish true temporal change from differences caused by another view,
  scan location, or comparison example.
- If the video shows side-by-side or example clips, describe them as visual examples
  rather than as changes in the same patient's condition.
- Preserve probe/view/setting changes only when they affect interpretation.

Grounding:
- Use only evidence visible in the ultrasound images or visually observable
  scanning actions.
- Do not rely on audio, transcripts, teaching context, or presenter explanations.
- Do not mention what the presenter explains, teaches, states, or intends.
- Describe observations rather than unsupported clinical interpretation.
- Do not interpret "not visible" as "absent".
- Report negative findings only when the relevant anatomy is adequately visualized.
- Report dynamic findings only when motion is directly observable across frames.
- Report numeric values only when clearly readable.
- Do not infer unsupported diagnoses or findings.

Discard first when memory is crowded:
- patient demographics
- generic patient positioning
- room or equipment descriptions
- routine gel application
- routine probe handling
- clinician gestures
- repeated descriptions of the same finding

Return one compact, factual paragraph of at most 120 words.
"""


def probe_duration_sec(video_path: Path) -> float:
    cmd = [
        "ffprobe",
        "-v",
        "error",
        "-show_entries",
        "format=duration",
        "-of",
        "default=noprint_wrappers=1:nokey=1",
        str(video_path),
    ]
    res = subprocess.run(cmd, capture_output=True, text=True)
    if res.returncode != 0:
        raise RuntimeError(f"ffprobe failed for {video_path}: {res.stderr.strip()[:400]}")
    return float(res.stdout.strip())


def strip_audio(video_path: Path, out_path: Path) -> Path:
    """Create a copy of `video_path` without an audio track."""
    out_path.parent.mkdir(parents=True, exist_ok=True)
    cmd = [
        "ffmpeg",
        "-y",
        "-loglevel",
        "error",
        "-i",
        str(video_path),
        "-c:v",
        "copy",
        "-an",
        str(out_path),
    ]
    res = subprocess.run(cmd, capture_output=True, text=True)
    if res.returncode == 0 and out_path.exists() and out_path.stat().st_size > 0:
        return out_path

    # Fallback for containers/codecs that cannot be stream-copied cleanly.
    fallback_cmd = [
        "ffmpeg",
        "-y",
        "-loglevel",
        "error",
        "-i",
        str(video_path),
        "-c:v",
        "libx264",
        "-preset",
        "veryfast",
        "-crf",
        "26",
        "-an",
        str(out_path),
    ]
    res = subprocess.run(fallback_cmd, capture_output=True, text=True)
    if res.returncode != 0 or not out_path.exists() or out_path.stat().st_size == 0:
        raise RuntimeError(f"ffmpeg audio stripping failed for {video_path}: {res.stderr.strip()[:400]}")
    return out_path


def prepare_teacher_clip(src_clip: Path, *, video_id: str, name: str, include_audio: bool) -> Path:
    """Return the clip path to send to the teacher, muted unless requested."""
    if include_audio:
        return src_clip
    muted = temp_clip_path(video_id, f"{name}_muted")
    return strip_audio(src_clip, muted)


def safe_video_id(path: Path, explicit: str | None = None) -> str:
    if explicit:
        return explicit
    stem = path.stem.strip() or "anchor_video"
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", stem)


def compact(text: str, max_chars: int | None = None) -> str:
    text = " ".join((text or "").strip().split())
    if max_chars and len(text) > max_chars:
        return text[: max_chars - 4].rstrip() + " ..."
    return text


def windows_between(start: float, end: float, *, window_sec: float) -> list[list[float]]:
    if window_sec <= 0:
        raise ValueError("window_sec must be positive")
    windows = []
    cur = float(start)
    while cur + window_sec <= end + 1e-6:
        windows.append([round(cur, 3), round(cur + window_sec, 3)])
        cur += window_sec
    return windows


def append_jsonl(path: Path, row: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(row, ensure_ascii=False) + "\n")


def read_existing(path: Path) -> tuple[set[int], str]:
    """Return completed block indices and last global summary for resume."""
    done: set[int] = set()
    last_idx = -1
    last_global = ""
    if not path.exists():
        return done, last_global
    with path.open(encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            row = json.loads(line)
            if row.get("error"):
                continue
            idx = int(row.get("block_idx", -1))
            done.add(idx)
            if idx > last_idx and row.get("global_summary_target"):
                last_idx = idx
                last_global = str(row["global_summary_target"])
    return done, last_global


def summarize_local(
    *,
    client,
    model: str,
    video_id: str,
    block_idx: int,
    clip_path: Path,
    start: float,
    end: float,
    max_tokens: int,
    temperature: float,
) -> tuple[str, dict[str, Any]]:
    prompt = f"""Summarize the clinically useful visual ultrasound evidence in this video block.

Time window: [{start:.1f}, {end:.1f}] seconds.

Return one compact paragraph of at most 80 words.

Include only distinct, visually supported evidence from this time window.
Prioritize ultrasound findings, anatomy, scan views, probe actions, and meaningful
scan dynamics over generic scene description."""
    raw, usage = call_with_content(
        client,
        content_blocks=[
            text_block(LOCAL_SYSTEM_PROMPT),
            build_video_block(clip_path, label=f"{video_id}_block{block_idx:04d}.mp4"),
            text_block(prompt),
        ],
        model=model,
        temperature=temperature,
        max_tokens=max_tokens,
    )
    return compact(raw), usage


def summarize_global_incremental(
    *,
    client,
    model: str,
    video_id: str,
    block_idx: int,
    clip_path: Path,
    start: float,
    end: float,
    previous_global: str,
    max_tokens: int,
    temperature: float,
) -> tuple[str, dict[str, Any]]:
    if previous_global:
        prompt = f"""Update the cumulative visual memory from time 0 to {end:.1f} seconds.

Previous cumulative visual memory:
{previous_global}

The attached video is the new segment covering
[{start:.1f}, {end:.1f}] seconds.

Rewrite the entire cumulative memory using the previous memory as historical
evidence and the attached video as new visual evidence.

Preserve important earlier findings, integrate important new findings, retain
clinically meaningful changes or comparisons, merge repeated observations,
and remove lower-value details when necessary.

Do not append to the previous wording.

Return one compact paragraph of at most 120 words."""
    else:
        prompt = f"""Update the cumulative visual memory from time 0 to {end:.1f} seconds.

There is no previous cumulative memory because this is the first video block.

The attached video covers
[0.0, {end:.1f}] seconds.

Produce the initial cumulative visual memory for [0, {end:.1f}] seconds.

Important:
- Treat the output as a compact fixed-capacity memory state.
- Include only clinically useful, visually supported evidence.
- Merge redundant observations.
- Prioritize ultrasound evidence over generic scene description.
- Keep the final summary within 120 words."""
    raw, usage = call_with_content(
        client,
        content_blocks=[
            text_block(GLOBAL_SYSTEM_PROMPT),
            build_video_block(clip_path, label=f"{video_id}_block{block_idx:04d}.mp4"),
            text_block(prompt),
        ],
        model=model,
        temperature=temperature,
        max_tokens=max_tokens,
    )
    return compact(raw), usage


def summarize_global_full_clip(
    *,
    client,
    model: str,
    video_id: str,
    block_idx: int,
    clip_path: Path,
    end: float,
    max_tokens: int,
    temperature: float,
) -> tuple[str, dict[str, Any]]:
    prompt = f"""Create a compact cumulative visual memory for the ultrasound video from
time 0 to {end:.1f} seconds.

Retain the most clinically useful distinct visual evidence observed so far.

Prioritize important anatomy, scan views, relevant probe actions, clearly visible
sonographic findings, meaningful dynamic findings, measurements, and clinically
meaningful temporal changes.

Merge repeated observations and remove redundant or low-value scene details.
Preserve chronology only when temporal order or change is clinically meaningful.

Do not infer unsupported diagnoses or findings.
Do not interpret "not visible" as "absent".

The output is a fixed-capacity memory state, not a transcript or exhaustive
description.

Return one compact paragraph of at most 120 words."""
    raw, usage = call_with_content(
        client,
        content_blocks=[
            text_block(GLOBAL_SYSTEM_PROMPT),
            build_video_block(clip_path, label=f"{video_id}_0_to_block{block_idx:04d}.mp4"),
            text_block(prompt),
        ],
        model=model,
        temperature=temperature,
        max_tokens=max_tokens,
    )
    return compact(raw), usage


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Generate per-minute and cumulative teacher visual summaries for one anchor video")
    p.add_argument("--video", required=True, help="Anchor video path")
    p.add_argument("--output", required=True, help="Output JSONL with one row per complete block")
    p.add_argument("--video-id", default=None, help="Override video_id; default = sanitized video stem")
    p.add_argument("--model", default=DEFAULT_TEACHER_SUMMARY_MODEL, help="OpenRouter video-capable model id")
    p.add_argument("--block-sec", type=float, default=60.0, help="Block length in seconds")
    p.add_argument("--local-sec", type=float, default=60.0, help="Local summary supervision window in seconds. Use 10 for dense short-memory labels.")
    p.add_argument("--max-blocks", type=int, default=None, help="Optional limit for smoke tests")
    p.add_argument("--global-mode", choices=["incremental", "full_clip"], default="incremental")
    p.add_argument("--max-local-tokens", type=int, default=300)
    p.add_argument("--max-global-tokens", type=int, default=500)
    p.add_argument("--temperature", type=float, default=0.2)
    p.add_argument("--include-audio", action="store_true", help="Send clips with audio. Default is muted visual-only clips.")
    p.add_argument("--keep-temp-clips", action="store_true")
    p.add_argument("--overwrite", action="store_true", help="Overwrite output instead of resuming")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    video_path = Path(args.video).expanduser().resolve()
    if not video_path.exists():
        raise FileNotFoundError(video_path)

    out_path = Path(args.output).expanduser().resolve()
    if args.overwrite and out_path.exists():
        out_path.unlink()

    video_id = safe_video_id(video_path, args.video_id)
    duration = probe_duration_sec(video_path)
    if args.local_sec <= 0 or args.local_sec > args.block_sec:
        raise ValueError("--local-sec must be in (0, --block-sec]")
    num_blocks = int(math.floor(duration / args.block_sec))
    if args.max_blocks is not None:
        num_blocks = min(num_blocks, int(args.max_blocks))
    if num_blocks <= 0:
        raise ValueError(f"No complete {args.block_sec}s block in {video_path} duration={duration:.2f}s")

    done, previous_global = read_existing(out_path)
    client = build_openrouter_client()

    print(f"[teacher-summary] video={video_path}")
    print(f"[teacher-summary] video_id={video_id} duration={duration:.2f}s blocks={num_blocks} global_mode={args.global_mode}")
    print(f"[teacher-summary] output={out_path} resume_done={len(done)}")

    for block_idx in range(num_blocks):
        if block_idx in done:
            print(f"[teacher-summary] skip existing block {block_idx}")
            continue
        start = block_idx * args.block_sec
        end = start + args.block_sec
        block_clip = temp_clip_path(video_id, f"teacher_summary_block{block_idx:04d}_{int(start)}_{int(end)}")
        cut_clip(video_path, start, end, block_clip)
        teacher_block_clip = prepare_teacher_clip(
            block_clip,
            video_id=video_id,
            name=f"teacher_summary_block{block_idx:04d}_{int(start)}_{int(end)}",
            include_audio=args.include_audio,
        )

        try:
            local_sub_summaries = []
            local_usages = []
            if abs(args.local_sec - args.block_sec) < 1e-6:
                local_summary, local_usage = summarize_local(
                    client=client,
                    model=args.model,
                    video_id=video_id,
                    block_idx=block_idx,
                    clip_path=teacher_block_clip,
                    start=start,
                    end=end,
                    max_tokens=args.max_local_tokens,
                    temperature=args.temperature,
                )
                local_usage_meta: dict[str, Any] | list[dict[str, Any]] = local_usage
            else:
                for sub_idx, (sub_start, sub_end) in enumerate(windows_between(start, end, window_sec=args.local_sec)):
                    sub_clip = temp_clip_path(video_id, f"teacher_summary_block{block_idx:04d}_sub{sub_idx:02d}_{int(sub_start)}_{int(sub_end)}")
                    cut_clip(video_path, sub_start, sub_end, sub_clip)
                    teacher_sub_clip = prepare_teacher_clip(
                        sub_clip,
                        video_id=video_id,
                        name=f"teacher_summary_block{block_idx:04d}_sub{sub_idx:02d}_{int(sub_start)}_{int(sub_end)}",
                        include_audio=args.include_audio,
                    )
                    try:
                        sub_summary, sub_usage = summarize_local(
                            client=client,
                            model=args.model,
                            video_id=video_id,
                            block_idx=block_idx,
                            clip_path=teacher_sub_clip,
                            start=sub_start,
                            end=sub_end,
                            max_tokens=args.max_local_tokens,
                            temperature=args.temperature,
                        )
                    finally:
                        if not args.keep_temp_clips:
                            if teacher_sub_clip != sub_clip:
                                teacher_sub_clip.unlink(missing_ok=True)
                            sub_clip.unlink(missing_ok=True)
                    local_sub_summaries.append({
                        "sub_idx": sub_idx,
                        "window": [round(sub_start, 3), round(sub_end, 3)],
                        "local_summary_target": sub_summary,
                    })
                    local_usages.append({"sub_idx": sub_idx, "window": [round(sub_start, 3), round(sub_end, 3)], "usage": sub_usage})
                local_summary = " ".join(s["local_summary_target"] for s in local_sub_summaries if s.get("local_summary_target"))
                local_usage_meta = local_usages

            if args.global_mode == "full_clip":
                global_clip = temp_clip_path(video_id, f"teacher_summary_0_to_{block_idx:04d}_{int(end)}")
                cut_clip(video_path, 0.0, end, global_clip)
                teacher_global_clip = prepare_teacher_clip(
                    global_clip,
                    video_id=video_id,
                    name=f"teacher_summary_0_to_{block_idx:04d}_{int(end)}",
                    include_audio=args.include_audio,
                )
                global_summary, global_usage = summarize_global_full_clip(
                    client=client,
                    model=args.model,
                    video_id=video_id,
                    block_idx=block_idx,
                    clip_path=teacher_global_clip,
                    end=end,
                    max_tokens=args.max_global_tokens,
                    temperature=args.temperature,
                )
                if not args.keep_temp_clips:
                    if teacher_global_clip != global_clip:
                        teacher_global_clip.unlink(missing_ok=True)
                    global_clip.unlink(missing_ok=True)
            else:
                global_summary, global_usage = summarize_global_incremental(
                    client=client,
                    model=args.model,
                    video_id=video_id,
                    block_idx=block_idx,
                    clip_path=teacher_block_clip,
                    start=start,
                    end=end,
                    previous_global=previous_global,
                    max_tokens=args.max_global_tokens,
                    temperature=args.temperature,
                )

            row = {
                "video_id": video_id,
                "video": str(video_path),
                "block_idx": block_idx,
                "block_window": [round(start, 3), round(end, 3)],
                "local_summary_target": local_summary,
                "global_summary_target": global_summary,
                "teacher": args.model,
                "meta": {
                    "source": "teacher_visual_summary",
                    "block_sec": args.block_sec,
                    "local_sec": args.local_sec,
                    "global_mode": args.global_mode,
                    "include_audio": args.include_audio,
                    "duration_sec": duration,
                    "local_usage": local_usage_meta,
                    "global_usage": global_usage,
                },
            }
            if local_sub_summaries:
                row["local_sub_summaries"] = local_sub_summaries
            append_jsonl(out_path, row)
            previous_global = global_summary
            sub_info = f" local_subs={len(local_sub_summaries)}" if local_sub_summaries else ""
            print(f"[teacher-summary] wrote block {block_idx} window=[{start:.1f},{end:.1f}]{sub_info} local_chars={len(local_summary)} global_chars={len(global_summary)}")
        except Exception as e:
            append_jsonl(out_path, {
                "video_id": video_id,
                "video": str(video_path),
                "block_idx": block_idx,
                "block_window": [round(start, 3), round(end, 3)],
                "teacher": args.model,
                "error": repr(e),
            })
            print(f"[teacher-summary] ERROR block {block_idx}: {e}")
            raise
        finally:
            if not args.keep_temp_clips:
                if teacher_block_clip != block_clip:
                    teacher_block_clip.unlink(missing_ok=True)
                block_clip.unlink(missing_ok=True)


if __name__ == "__main__":
    main()