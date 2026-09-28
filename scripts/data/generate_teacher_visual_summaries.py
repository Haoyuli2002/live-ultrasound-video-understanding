#!/usr/bin/env python3
"""Generate teacher visual summaries for one anchor ultrasound video.

Output is one JSONL row per complete block and is directly consumable by:

    pretrain/build_teacher_memory_summary_samples.py

For each 60-second block this script asks a video-capable teacher model for:
  1. local_summary_target: a short visual summary of the current minute only;
  2. global_summary_target: a concise cumulative visual summary from 0 to T.

By default the global summary is generated incrementally from the previous
global summary plus the current minute clip. This is much cheaper and scales to
long videos. For strict anchor-video experiments, `--global-mode full_clip`
instead sends the full [0, T] clip for each block; use it only for short clips.

The prompts explicitly ask for visual ultrasound evidence and discourage using
audio/transcript information, matching the Stage-2 Summary Compression Stage
teacher-visual-summary objective.
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
observed from the beginning of the video up to the current time.

Prioritize:
- important anatomy or organs already visualized
- clinically meaningful scan views
- relevant probe or scanning actions when they matter for interpretation
- clearly visible sonographic findings
- meaningful dynamic findings
- important measurements or visible labels
- clinically meaningful changes over time

Memory-update rules:
- Preserve important earlier evidence even when it is no longer visible.
- Integrate new clinically useful evidence from the current segment.
- Merge repeated observations into one concise statement.
- Remove redundant wording and low-value scene details.
- Do not simply append a description of the current segment.
- Preserve chronology only when temporal order or change is clinically meaningful.
- If new visual evidence refines an earlier observation, update the earlier statement
  instead of keeping both versions.
- Preserve the uncertainty level of earlier observations unless new visual evidence
  clearly resolves that uncertainty.
- Absence of a finding in the current segment does not invalidate an earlier finding.

Grounding rules:
- Use only evidence supported by the visible ultrasound images or visually observable
  probe/scanning actions.
- Do not rely on audio narration, transcript content, teaching context, or expected anatomy.
- Prioritize ultrasound imaging content over generic scene description.
- Mention image quality only when it materially limits interpretation.
- Do not interpret "not visible" as "absent".
- Report negative findings only when the relevant anatomy has been adequately visualized.
- Report dynamic findings only when the corresponding motion is directly observable
  across video frames.
- Do not infer unsupported diagnoses or findings from medical knowledge alone.
- Do not add unsupported information.

Keep the final cumulative memory concise and information-dense.
The final summary must be at most 120 words.
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

Previous cumulative visual memory from 0 to {start:.1f} seconds:
{previous_global}

The attached video is the new segment covering
[{start:.1f}, {end:.1f}] seconds.

Produce the updated cumulative visual memory for [0, {end:.1f}] seconds.

Important:
- Treat the output as a compact fixed-capacity memory state.
- Preserve clinically important earlier evidence.
- Integrate only new visually supported evidence from the current segment.
- Merge repeated observations and remove redundancy.
- Do not simply append a description of the current segment.
- Do not discard earlier findings merely because they are not visible now.
- Preserve chronology only when it represents a clinically meaningful change.
- Keep the final summary within 120 words."""
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
    p.add_argument("--max-blocks", type=int, default=None, help="Optional limit for smoke tests")
    p.add_argument("--global-mode", choices=["incremental", "full_clip"], default="incremental")
    p.add_argument("--max-local-tokens", type=int, default=300)
    p.add_argument("--max-global-tokens", type=int, default=500)
    p.add_argument("--temperature", type=float, default=0.2)
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

        try:
            local_summary, local_usage = summarize_local(
                client=client,
                model=args.model,
                video_id=video_id,
                block_idx=block_idx,
                clip_path=block_clip,
                start=start,
                end=end,
                max_tokens=args.max_local_tokens,
                temperature=args.temperature,
            )

            if args.global_mode == "full_clip":
                global_clip = temp_clip_path(video_id, f"teacher_summary_0_to_{block_idx:04d}_{int(end)}")
                cut_clip(video_path, 0.0, end, global_clip)
                global_summary, global_usage = summarize_global_full_clip(
                    client=client,
                    model=args.model,
                    video_id=video_id,
                    block_idx=block_idx,
                    clip_path=global_clip,
                    end=end,
                    max_tokens=args.max_global_tokens,
                    temperature=args.temperature,
                )
                if not args.keep_temp_clips:
                    global_clip.unlink(missing_ok=True)
            else:
                global_summary, global_usage = summarize_global_incremental(
                    client=client,
                    model=args.model,
                    video_id=video_id,
                    block_idx=block_idx,
                    clip_path=block_clip,
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
                    "global_mode": args.global_mode,
                    "duration_sec": duration,
                    "local_usage": local_usage,
                    "global_usage": global_usage,
                },
            }
            append_jsonl(out_path, row)
            previous_global = global_summary
            print(f"[teacher-summary] wrote block {block_idx} window=[{start:.1f},{end:.1f}] local_chars={len(local_summary)} global_chars={len(global_summary)}")
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
                block_clip.unlink(missing_ok=True)


if __name__ == "__main__":
    main()