"""
Video-LLM helper for OpenRouter (Gemini 2.5 Flash and similar).

Provides a single source of truth for:
  - cutting a sub-clip with ffmpeg (stream copy, no re-encoding)
  - encoding mp4 -> base64 data URL
  - building the OpenAI/OpenRouter `type: "file"` content block (empirically
    verified to be the recommended path on OpenRouter for Gemini 2.5 Flash
    video input — produces non-zero prompt_tokens_details.video_tokens)
  - calling the chat.completions endpoint with one or more video blocks
    and returning (text, usage) safely (handles None content / None usage /
    transient 504s with one retry).

Used by:
  - scripts/qa_validator.py            (Step 5c)
  - scripts/streaming_qa_generation.py (Step 5b, after migration)
  - scripts/qa_generation.py           (Step 5a, optional migration)

Usage (smoke test):
    python scripts/_video_llm.py
"""

from __future__ import annotations

import os
import sys
import time
import base64
import shutil
import subprocess
from pathlib import Path
from typing import Iterable

# Auto-load .env so caller scripts don't have to
_HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE))
import _env_loader  # noqa: F401


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

DEFAULT_MODEL = "google/gemini-2.5-flash"
OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"

# Where to drop temporary clip files. /tmp is auto-cleaned on reboot.
TEMP_ROOT = Path(os.environ.get("LVUV_TEMP_DIR", "/tmp/lvuv"))


# ---------------------------------------------------------------------------
# ffmpeg clip cutting
# ---------------------------------------------------------------------------

def _ffmpeg_available() -> bool:
    return shutil.which("ffmpeg") is not None


def cut_clip(video_path: str | Path,
             start: float,
             end: float,
             out_path: str | Path,
             reencode_fallback: bool = True) -> Path:
    """
    Cut [start, end] (seconds) from `video_path` to `out_path`.

    Tries fast path first: input seek + stream copy (no re-encode).
    Stream copy can occasionally produce a clip that does not start cleanly
    at a keyframe — Gemini still accepts it, but if you need byte-perfect
    clips you can disable fallback and re-encode.

    Returns Path(out_path) on success; raises RuntimeError otherwise.
    """
    if not _ffmpeg_available():
        raise RuntimeError("ffmpeg not found in PATH (brew install ffmpeg)")

    video_path = Path(video_path)
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    duration = max(float(end) - float(start), 0.1)

    # Input seeking (-ss before -i) is fast AND accurate enough for OpenRouter.
    # `-c copy` keeps original codec (vp9/opus or h264/aac), no re-encode.
    fast_cmd = [
        "ffmpeg", "-y", "-loglevel", "error",
        "-ss", f"{float(start):.3f}",
        "-i", str(video_path),
        "-t", f"{duration:.3f}",
        "-c", "copy",
        "-avoid_negative_ts", "make_zero",
        str(out_path),
    ]
    res = subprocess.run(fast_cmd, capture_output=True, text=True)
    if res.returncode == 0 and out_path.exists() and out_path.stat().st_size > 0:
        return out_path

    if not reencode_fallback:
        raise RuntimeError(
            f"ffmpeg stream-copy failed for [{start:.2f}, {end:.2f}] of {video_path}: "
            f"{res.stderr.strip()[:400]}"
        )

    # Fallback: re-encode with libx264 + aac (slower, but always works)
    slow_cmd = [
        "ffmpeg", "-y", "-loglevel", "error",
        "-ss", f"{float(start):.3f}",
        "-i", str(video_path),
        "-t", f"{duration:.3f}",
        "-c:v", "libx264", "-preset", "veryfast", "-crf", "26",
        "-c:a", "aac", "-b:a", "96k",
        str(out_path),
    ]
    res = subprocess.run(slow_cmd, capture_output=True, text=True)
    if res.returncode != 0 or not out_path.exists() or out_path.stat().st_size == 0:
        raise RuntimeError(
            f"ffmpeg failed (both fast and reencode) for [{start:.2f}, {end:.2f}] "
            f"of {video_path}: {res.stderr.strip()[:400]}"
        )
    return out_path


def temp_clip_path(video_id: str, name: str) -> Path:
    """
    Return /tmp/lvuv/{video_id}/{name}.mp4 (parent dir created lazily by cut_clip).
    """
    return TEMP_ROOT / video_id / f"{name}.mp4"


# ---------------------------------------------------------------------------
# Base64 + content block
# ---------------------------------------------------------------------------

def video_to_data_url(path: str | Path) -> str:
    """Read an mp4 file and return a data URL: data:video/mp4;base64,..."""
    p = Path(path)
    with p.open("rb") as f:
        b64 = base64.b64encode(f.read()).decode()
    return f"data:video/mp4;base64,{b64}"


def build_video_block(path: str | Path, label: str | None = None) -> dict:
    """
    Build an OpenAI/OpenRouter `file` content block. This is the empirically
    verified recommended way to send a video to Gemini 2.5 Flash via
    OpenRouter (produces non-zero prompt_tokens_details.video_tokens).
    """
    p = Path(path)
    return {
        "type": "file",
        "file": {
            "filename": label or p.name,
            "file_data": video_to_data_url(p),
        },
    }


# ---------------------------------------------------------------------------
# OpenRouter client
# ---------------------------------------------------------------------------

def build_openrouter_client(api_key: str | None = None):
    """OpenRouter exposes an OpenAI-compatible chat.completions API."""
    from openai import OpenAI

    api_key = api_key or os.environ.get("OPENROUTER_API_KEY", "")
    if not api_key:
        raise ValueError(
            "OPENROUTER_API_KEY not set. Put it in .env or pass api_key=..."
        )
    return OpenAI(
        base_url=OPENROUTER_BASE_URL,
        api_key=api_key,
        default_headers={
            "HTTP-Referer": "https://github.com/Haoyuli2002/live-ultrasound-video-understanding",
            "X-Title": "live-ultrasound-video-understanding",
        },
    )


# ---------------------------------------------------------------------------
# Safe call wrapper
# ---------------------------------------------------------------------------

def _usage_to_dict(usage) -> dict:
    if usage is None:
        return {}
    try:
        return usage.model_dump()
    except AttributeError:
        try:
            return dict(usage)
        except Exception:
            return {}


def text_block(text: str) -> dict:
    """Build a plain text content block."""
    return {"type": "text", "text": text}


# When OpenRouter / upstream returns a rate-limit error we wait this long
# before retrying instead of using the regular exponential backoff. Gemini
# 2.5 Flash on OpenRouter is rate-limited to ~200 RPM; 60s typically
# resets the bucket.
RATE_LIMIT_BACKOFF_SEC = 60.0


def _is_rate_limit_error(err: Exception | None) -> bool:
    """Detect 429 / 'rate' / 'too many requests' anywhere in the error."""
    if err is None:
        return False
    s = str(err).lower()
    return ("'code': 429" in s or '"code": 429' in s
            or 'code=429' in s
            or '429,' in s
            or 'rate' in s
            or 'too many requests' in s)


def call_with_content(
    client,
    *,
    content_blocks: Iterable[dict],
    model: str = DEFAULT_MODEL,
    temperature: float = 0.2,
    max_tokens: int | None = None,
    extra_body: dict | None = None,
    retries: int = 5,
    retry_delay: float = 8.0,
    retry_backoff: float = 1.6,
    rate_limit_backoff_sec: float = RATE_LIMIT_BACKOFF_SEC,
    verbose_retries: bool = True,
) -> tuple[str, dict]:
    """
    Send a pre-built ordered list of content blocks (text + video, in any
    order the caller wants) to an OpenRouter-compatible chat.completions
    endpoint and return (text, usage).

    Use this when you need precise interleaving like:
        [text]  [video_A]  [text]  [video_B]  [text]

    Retry behaviour
    ---------------
    On any of:
      - SDK exception (network / HTTP error)
      - empty `resp.choices`        (OpenRouter returns 504 envelope)
      - empty `resp.choices[0].message.content`
    waits `retry_delay * retry_backoff^attempt` seconds and retries.

    If the previous error looked like a 429 / rate-limit, we instead wait
    `rate_limit_backoff_sec` (default 60s) which usually resets the bucket.

    Default = 5 retries, schedule (no rate limit) 8s, 13s, 20s, 33s, 52s.

    Returns
    -------
    text  : str   -- model output (empty string if model returned no content)
    usage : dict  -- usage dict (may include `video_tokens`, `audio_tokens`, `cost`)
    """
    eb = {"reasoning": {"exclude": True}}
    if extra_body:
        eb.update(extra_body)

    content = list(content_blocks)
    if not content:
        raise ValueError("content_blocks is empty")

    kwargs = dict(
        model=model,
        messages=[{"role": "user", "content": content}],
        temperature=temperature,
        extra_body=eb,
    )
    if max_tokens is not None:
        kwargs["max_tokens"] = max_tokens

    last_err: Exception | None = None
    for attempt in range(retries + 1):
        if attempt > 0:
            if _is_rate_limit_error(last_err):
                wait = rate_limit_backoff_sec
                tag = "rate-limit"
            else:
                wait = retry_delay * (retry_backoff ** (attempt - 1))
                tag = "transient"
            if verbose_retries:
                print(f"      retry {attempt}/{retries} ({tag}) after {wait:.1f}s "
                      f"(prev: {last_err})")
            time.sleep(wait)

        try:
            resp = client.chat.completions.create(**kwargs)
        except Exception as e:
            last_err = e
            if attempt < retries:
                continue
            raise

        # OpenRouter 504/429: returns a Response object with all-null fields
        # and a top-level `error` attribute.
        if not resp.choices:
            err_obj = getattr(resp, "error", None)
            last_err = RuntimeError(f"OpenRouter empty response: {err_obj}")
            if attempt < retries:
                continue
            raise last_err

        choice = resp.choices[0]
        msg = choice.message
        text = (msg.content if msg else None) or ""
        usage = _usage_to_dict(resp.usage)

        if not text:
            # Some providers occasionally return an empty content under load.
            last_err = RuntimeError(
                f"empty content (finish_reason="
                f"{getattr(choice, 'finish_reason', None)})"
            )
            if attempt < retries:
                continue
            # If we exhausted retries with empty content, return empty string —
            # caller can decide whether to treat this as failure.
            return text, usage

        return text, usage

    # unreachable
    raise RuntimeError(f"call_with_content exhausted retries: {last_err}")


def call_with_videos(
    client,
    *,
    prompt: str,
    video_blocks: Iterable[dict],
    text_blocks_before: Iterable[str] = (),
    text_blocks_after: Iterable[str] = (),
    model: str = DEFAULT_MODEL,
    temperature: float = 0.2,
    max_tokens: int | None = None,
    extra_body: dict | None = None,
    retries: int = 1,
    retry_delay: float = 4.0,
) -> tuple[str, dict]:
    """
    Convenience wrapper around call_with_content for the common layout:
        [text_before...] + [video_blocks...] + [text_after...] + [prompt]
    """
    blocks: list[dict] = []
    for t in text_blocks_before:
        if t:
            blocks.append(text_block(t))
    for vb in video_blocks:
        blocks.append(vb)
    for t in text_blocks_after:
        if t:
            blocks.append(text_block(t))
    blocks.append(text_block(prompt))

    return call_with_content(
        client,
        content_blocks=blocks,
        model=model,
        temperature=temperature,
        max_tokens=max_tokens,
        extra_body=extra_body,
        retries=retries,
        retry_delay=retry_delay,
    )


# ---------------------------------------------------------------------------
# Smoke test (run this file directly)
# ---------------------------------------------------------------------------

def _smoke_test():
    """End-to-end check: cut a sub-clip and ask Gemini to describe it."""
    src = Path("/tmp/probe/clip5s.mp4")
    if not src.exists():
        sys.exit(
            f"Smoke test source missing: {src}\n"
            f"Hint: ffmpeg -y -ss 60 -i <some.mp4> -t 5 -c copy {src}"
        )

    # cut a 3s sub-clip from the 5s source to also exercise cut_clip
    out = TEMP_ROOT / "smoke" / "sub3s.mp4"
    cut_clip(src, 0.0, 3.0, out)
    print(f"[smoke] cut_clip OK -> {out} ({out.stat().st_size/1024:.1f} KB)")

    client = build_openrouter_client()
    block = build_video_block(out, label="sub3s.mp4")

    text, usage = call_with_videos(
        client,
        prompt="Briefly describe what happens in this video clip in one sentence.",
        video_blocks=[block],
        temperature=0.1,
    )
    print(f"[smoke] response: {text!r}")
    print(f"[smoke] usage: total={usage.get('total_tokens')} "
          f"video_tokens={(usage.get('prompt_tokens_details') or {}).get('video_tokens')} "
          f"audio_tokens={(usage.get('prompt_tokens_details') or {}).get('audio_tokens')} "
          f"cost=${usage.get('cost'):.6f}" if usage.get('cost') is not None else
          f"[smoke] usage: total={usage.get('total_tokens')}")

    if not text:
        sys.exit("[smoke] FAILED: empty response text")
    print("[smoke] OK")


if __name__ == "__main__":
    _smoke_test()
