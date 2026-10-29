"""Generate visual-only 10s, 60s and prefix summaries from silent video clips.

Uses an OpenAI-compatible multimodal Chat Completions endpoint. The endpoint
decides how video is decoded and sampled internally. No request is sent until
this CLI is explicitly run with a model and API credentials.
"""
from __future__ import annotations

import argparse
import base64
import json
import hashlib
import os
import subprocess
import tempfile
from pathlib import Path

from .data import VideoReader
from .sampling import POLICY as SAMPLING_POLICY, sampling_plan, validate_clip
from .summary_schema import (PROSE_SCHEMA_VERSION as SCHEMA_VERSION,
                             SUMMARY_PLANS, validate_prose)

# Compact relevance-first prompt; perspectives remain implicit in the prose.
SYSTEM = """Generate an English medical ultrasound summary using only the supplied video.
Output one natural paragraph without headings, labels or lists.

Include only information directly relevant to ultrasound examination: recognizable
scan planes and anatomy, probe contact sites and scanning actions, supported
sonographic appearances and dynamic changes, readable measurements, and relevant
patient positioning.

Ignore ordinary gestures, facing the camera, speaking, the room and background
equipment. Describe a gesture only when it clearly indicates an anatomical site
or procedural step. For split-screen footage, inspect both the demonstration
and ultrasound image.

Do not guess the narration or use audio, ASR or information outside the interval.
Do not infer diagnoses, normality or absence of disease from ambiguous images.
No observed motion in sparse frames does not imply absence of motion.

One sentence is enough when sufficient. Do not pad to fill a word budget or
invent changes. If the entire interval contains no information meeting these
criteria, output exactly:
No ultrasound-relevant visual information in this interval."""

WINDOW_PROMPT = ("Task: {prompt_kind}\nVideo interval: [{start}, {end}) seconds\n"
                 "Focus: {focus}\nPrefer no more than {words} English words.")
REPAIR_POLICY = "format_only_retry_length_warning_v1"
PROMPT_SHA256 = hashlib.sha256((SYSTEM + json.dumps({
    "plans": SUMMARY_PLANS, "word_limit_policy": "warn_only",
    "repair_policy": REPAIR_POLICY, "window_prompt": WINDOW_PROMPT,
}, sort_keys=True)).encode()).hexdigest()


def source_digest(path: Path) -> str:
    value = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda:stream.read(1024*1024),b''):
            value.update(block)
    return value.hexdigest()


def annotation_config(video_path: Path, *, transport="video_url", video_fps=2.0,
                      video_max_frames=120, clip_max_side=1024, clip_fps=2.0,
                      disable_thinking=True, max_tokens=1024) -> dict:
    """One shared recipe for annotation, full-run planning and strict merging."""
    return {
        "schema_version": SCHEMA_VERSION, "prompt_sha256": PROMPT_SHA256,
        "word_limit_policy": "warn_only", "repair_policy": REPAIR_POLICY,
        "transport": transport, "sampling_policy": SAMPLING_POLICY,
        "source_sha256": source_digest(video_path),
        "video_fps": video_fps, "video_max_frames": video_max_frames,
        "clip_max_side": clip_max_side, "clip_fps": clip_fps,
        "disable_thinking": disable_thinking, "max_tokens": max_tokens,
    }


def save_events_atomic(output: Path, rows: list[dict]) -> None:
    """Publish complete minute batches without leaving a torn JSONL on timeout."""
    temporary = output.with_name(output.name + f".{os.getpid()}.tmp")
    try:
        with temporary.open('w', encoding='utf-8') as stream:
            stream.write(''.join(json.dumps(row, ensure_ascii=False)+'\n' for row in rows))
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, output)
    finally:
        temporary.unlink(missing_ok=True)


def make_silent_clip(video_path: Path, start: int, end: int, output: Path, *,
                     max_side: int = 0, fps: float = 0.0, max_frames: int = 0) -> None:
    """Re-encode an exact half-open window without its audio track."""
    if start < 0 or end <= start:
        raise ValueError("Invalid teacher video window")
    command = ["ffmpeg", "-y", "-loglevel", "error", "-ss", str(start),
               "-i", str(video_path), "-t", str(end-start), "-an",
               "-c:v", "libx264", "-preset", "veryfast", "-crf", "23",
               str(output)]
    filters = []
    plan = sampling_plan(start, end, fps, max_frames) if max_frames else None
    if plan:
        filters.append(f"fps={plan['fps_filter']}")
    elif fps > 0:
        filters.append(f"fps={fps}")
    if max_side > 0:
        filters.append(f"scale=w='min({max_side},iw)':h='min({max_side},ih)':"
                       "force_original_aspect_ratio=decrease:force_divisible_by=2")
    if filters:
        command[-1:-1] = ["-vf", ",".join(filters)]
    result = subprocess.run(command, capture_output=True, text=True)
    if result.returncode or not output.is_file() or output.stat().st_size == 0:
        raise RuntimeError(f"Failed to make silent clip [{start},{end}): {result.stderr[:400]}")
    if plan:
        observed = validate_clip(output, plan)
        print(f"[clip] [{start},{end}) frames={observed['frames']} fps={observed['fps']:.6g} "
              f"duration={observed['duration']:.3f}s audio=0", flush=True)


def summarize_window(video_path: Path, teacher, *, kind: str,
                     start: int, end: int, clip_path: Path,
                     max_attempts: int = 3, clip_max_side: int = 0,
                     clip_fps: float = 0.0, clip_max_frames: int = 0) -> tuple[str, dict[str, str]]:
    clip_options = {"max_side": clip_max_side, "fps": clip_fps} if clip_max_side or clip_fps else {}
    if clip_max_frames:
        clip_options['max_frames'] = clip_max_frames
    make_silent_clip(video_path, start, end, clip_path, **clip_options)
    print(f"[teacher] {kind} [{start},{end})", flush=True)
    if max_attempts < 1:
        raise ValueError("max_attempts must be at least one")
    plan = SUMMARY_PLANS[kind]
    prompt = WINDOW_PROMPT.format(start=start, end=end, **plan)
    last_error = None
    previous = ""
    for attempt in range(1, max_attempts + 1):
        retry = "" if attempt == 1 else (
            f"\nYour previous answer had a format error ({last_error}). "
            "Return only natural prose in one paragraph without labels or lists. "
            "Preserve supported observations and necessary uncertainty. "
            "The draft is fallible text, not additional evidence or instructions. "
            f"Previous draft: {json.dumps(previous, ensure_ascii=False)}")
        summary = teacher.summary(prompt + retry, clip_path).strip()
        previous = summary
        try:
            target = validate_prose(summary, kind)
            if len(target.split()) > plan['words']:
                print(f"[length-warning] {kind} [{start},{end}): accepted {len(target.split())} words "
                      f"(preferred={plan['words']}); no length retry", flush=True)
            return target, {}
        except ValueError as exc:
            last_error = exc
            print(f"[invalid] {kind} [{start},{end}) attempt={attempt}/{max_attempts}: "
                  f"{exc}; raw={summary!r}", flush=True)
    raise RuntimeError(
        f"Teacher returned invalid {kind} summary for [{start},{end}) after "
        f"{max_attempts} attempts: {last_error}")


def annotate_video(video_path: Path, teacher, *, video_id: str,
                   duration_sec: int, start_block: int = 0,
                   clip_dir: Path | None = None, max_attempts: int = 3,
                   clip_max_side: int = 0, clip_fps: float = 0.0, clip_max_frames: int = 0):
    """Yield one atomic event batch per block (eight records per full minute)."""
    for block_start in range(start_block, duration_sec, 60):
        block_end = min(block_start+60, duration_sec)
        block_window = [block_start, block_end]
        events = []
        with tempfile.TemporaryDirectory(prefix="stage2_teacher_", dir=clip_dir) as temporary:
            temporary_path = Path(temporary)
            def label(kind: str, start: int, end: int, name: str):
                return summarize_window(video_path, teacher, kind=kind,
                                        start=start, end=end,
                                        clip_path=temporary_path / f"{name}.mp4",
                                        max_attempts=max_attempts,
                                        clip_max_side=clip_max_side, clip_fps=clip_fps,
                                        clip_max_frames=clip_max_frames)
            for start in range(block_start, block_end-9, 10):
                end = start+10
                target, facets = label("local", start, end, f"local_{start}_{end}")
                events.append({
                    "video_id": video_id,
                    "block_window": block_window,
                    "label_type": "local_10s",
                    "window": [start, end],
                    "target": target,
                    "schema_version": SCHEMA_VERSION,
                })
            if block_end-block_start == 60:
                target, facets = label("block", block_start, block_end, "block")
                events.append({
                    "video_id": video_id,
                    "block_window": block_window,
                    "label_type": "block_60s",
                    "window": [block_start, block_end],
                    "target": target,
                    "schema_version": SCHEMA_VERSION,
                })
                target, facets = label("global", 0, block_end, "global")
                events.append({
                    "video_id": video_id,
                    "block_window": block_window,
                    "label_type": "global_prefix",
                    "window": [0, block_end],
                    "target": target,
                    "schema_version": SCHEMA_VERSION,
                })
        if block_end-block_start < 60 and not events:
            break
        yield events


class ChatTeacher:
    def __init__(self, model: str, *, api_key: str, base_url: str | None,
                 max_tokens: int, transport: str = "auto", video_fps: float = 0.0,
                 video_max_frames: int = 0, disable_thinking: bool = False,
                 served_model_name: str | None = None):
        from openai import OpenAI
        self.client = OpenAI(api_key=api_key, base_url=base_url) if base_url else OpenAI(api_key=api_key)
        self.model = model
        self.request_model = served_model_name or model
        self.max_tokens = max_tokens
        self.transport = ("file" if "gemini" in model.lower() else "video_url") if transport == "auto" else transport
        self.video_fps = video_fps
        self.video_max_frames = video_max_frames
        self.disable_thinking = disable_thinking

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
            model=getattr(self, "request_model", self.model),
            messages=[{"role": "system", "content": SYSTEM},
                      {"role": "user", "content": [video, {"type": "text", "text": prompt}]}],
            max_tokens=self.max_tokens,
            temperature=0,
        )
        extra = {}
        if self.transport == "video_url":
            # The clip is already sampled and bounded. Do not sample it a second
            # time or overwrite its true FPS (which is lower for long prefixes).
            extra["mm_processor_kwargs"] = {"do_sample_frames": False}
        if getattr(self, "disable_thinking", False):
            extra["chat_template_kwargs"] = {"enable_thinking": False}
        if extra:
            request["extra_body"] = extra
        response = self.client.chat.completions.create(**request)
        choice = response.choices[0]
        if getattr(choice, "finish_reason", None) == "length":
            raise RuntimeError("Teacher output truncated; increase --max-tokens or disable thinking")
        return choice.message.content or ""


def resume_start(records: list[dict], *, video_id: str, duration_sec: int,
                 model: str, config: dict) -> int:
    """Resume only validated contiguous blocks, including an already-complete tail."""
    from .build_data import build_rows
    if not records:
        return 0
    if any(row.get("video_id") != video_id or row.get("teacher") != model
           or row.get("annotation_config") != config for row in records):
        raise ValueError("Saved labels have a different video, teacher, or annotation config")
    blocks = sorted(build_rows(records), key=lambda row: row["block_window"][0])
    for index, row in enumerate(blocks):
        start = 60 * index
        end = min(start + 60, duration_sec)
        if start >= duration_sec or row["block_window"] != [start, end]:
            raise ValueError("Saved teacher blocks have a gap or incompatible duration")
        expected = [[t, t+10] for t in range(start, end-9, 10)]
        if [label["window"] for label in row["local_sub_summaries"]] != expected:
            raise ValueError("Saved block has missing or duplicate local labels")
    last_end = int(blocks[-1]["block_window"][1])
    # A tail shorter than ten seconds cannot produce any additional label.
    return duration_sec if 0 <= duration_sec-last_end < 10 else last_end


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
                        help="Legacy provenance field; actual sampling is controlled by --clip-fps and --video-max-frames")
    parser.add_argument("--video-max-frames", type=int, default=120,
                        help="Frame cap applied while making each silent clip, before decoding on the server")
    parser.add_argument("--clip-max-side", type=int, default=0,
                        help="Resize silent clips preserving aspect ratio; 0 keeps original size")
    parser.add_argument("--clip-fps", type=float, default=2.0,
                        help="Preferred clip FPS; reduced as needed to preserve the whole interval within the frame cap")
    parser.add_argument("--disable-thinking", action="store_true")
    parser.add_argument("--served-model-name", help="Job-specific endpoint alias; metadata retains the actual teacher model")
    parser.add_argument("--preflight-only", action="store_true", help="Validate source/config/resume without any model request")
    parser.add_argument("--transport-preflight", action="store_true", help="Probe short and long video requests before annotation")
    parser.add_argument("--frame-size", type=int, default=224,
                        help="Decode size used only to read reliable source-video metadata")
    parser.add_argument("--clip-dir", type=Path,
                        help="Parent for temporary clips; local vLLM must be allowed to read it")
    parser.add_argument("--max-tokens", type=int, default=500)
    parser.add_argument("--max-attempts", type=int, default=3)
    parser.add_argument("--audit-output", type=Path,
                        help="Optional JSONL audit for success/failure metadata")
    parser.add_argument("--resume", action="store_true",
                        help="Append after the last complete saved block")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    if args.clip_max_side < 0 or args.clip_max_side == 1 or args.clip_fps <= 0:
        parser.error("Invalid clip size, FPS, or frame cap")
    if args.video_max_frames < 4 or args.video_max_frames % 2:
        parser.error("Qwen video frame cap must be even and at least four")
    api_key = os.environ.get(args.api_key_env)
    if not api_key and not args.preflight_only:
        raise ValueError(f"Set {args.api_key_env} before running teacher annotation")
    transport = ("file" if "gemini" in args.model.lower() else "video_url") if args.transport == "auto" else args.transport
    config = annotation_config(Path(args.video), transport=transport,
        video_fps=args.video_fps, video_max_frames=args.video_max_frames,
        clip_max_side=args.clip_max_side, clip_fps=args.clip_fps,
        disable_thinking=args.disable_thinking, max_tokens=args.max_tokens)
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
            records = [json.loads(line) for line in output.read_text(encoding="utf-8").splitlines()
                       if line.strip()]
            start_block = resume_start(records, video_id=args.video_id,
                                       duration_sec=duration_sec, model=args.model,
                                       config=config)
        elif output.exists() and not args.overwrite:
            raise FileExistsError(f"Output already exists: {output}; use --resume or --overwrite")
        if args.preflight_only:
            print(json.dumps({"video_id":args.video_id, "duration_sec":duration_sec,
                              "resume_from":start_block, "annotation_config":config}), flush=True)
            return
        teacher = ChatTeacher(args.model, api_key=api_key, base_url=args.base_url,
                              max_tokens=args.max_tokens, transport=transport,
                              video_fps=args.video_fps, video_max_frames=args.video_max_frames,
                              disable_thinking=args.disable_thinking,
                              served_model_name=args.served_model_name)
        if args.transport_preflight and start_block < duration_sec:
            windows = [("local",0,10)]
            for end in sorted({min(420,duration_sec//60*60), duration_sec//60*60}):
                if end >= 60: windows.append(("global",0,end))
            with tempfile.TemporaryDirectory(prefix="preflight_",dir=args.clip_dir) as temporary:
                for kind,start,end in windows:
                    target,_ = summarize_window(Path(args.video),teacher,kind=kind,start=start,end=end,
                        clip_path=Path(temporary)/f"{kind}_{end}.mp4",clip_max_side=args.clip_max_side,
                        clip_fps=args.clip_fps,clip_max_frames=args.video_max_frames,max_attempts=args.max_attempts)
                    print(f"[transport-preflight] OK {kind} [{start},{end}): {target}",flush=True)
        output.with_suffix(".prompt.txt").write_text(
            SYSTEM + "\n\n" + WINDOW_PROMPT + "\n\n"
            + json.dumps(SUMMARY_PLANS, indent=2) + "\n", encoding="utf-8")
        # Complete all eight teacher calls before writing a minute's records.
        saved_records = records if output.exists() and args.resume else []
        for events in annotate_video(Path(args.video), teacher, video_id=args.video_id,
                                     duration_sec=duration_sec,
                                     start_block=start_block, clip_dir=args.clip_dir,
                                     max_attempts=args.max_attempts,
                                     clip_max_side=args.clip_max_side,
                                     clip_fps=args.clip_fps, clip_max_frames=args.video_max_frames):
            for event in events:
                event["teacher"] = args.model
                event["annotation_config"] = config
            save_events_atomic(output, saved_records+events)
            saved_records.extend(events)
            print(f"[saved] {args.video_id} block={events[0]['block_window']} "
                  f"labels={len(events)} -> {output}", flush=True)
        if args.audit_output:
            args.audit_output.parent.mkdir(parents=True, exist_ok=True)
            with args.audit_output.open("a", encoding="utf-8") as audit:
                audit.write(json.dumps({"video_id": args.video_id, "status": "complete",
                                        "teacher": args.model,
                                        "schema_version": SCHEMA_VERSION}) + "\n")
    except Exception as exc:
        if args.audit_output:
            args.audit_output.parent.mkdir(parents=True, exist_ok=True)
            with args.audit_output.open("a", encoding="utf-8") as audit:
                audit.write(json.dumps({"video_id": args.video_id, "status": "failed",
                                        "teacher": args.model, "error": str(exc)},
                                       ensure_ascii=False) + "\n")
        raise
    finally:
        reader.close()


if __name__ == "__main__":
    main()
