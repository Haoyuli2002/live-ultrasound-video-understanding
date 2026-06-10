"""
Streaming QA Generation Pipeline (Oracle Generator)
=====================================================
Generates 2 types of TIME-SENSITIVE QA at multiple anchor points within each clip:
  - sonographer_intent     : What is the operator currently trying to do/find?
  - next_action_guidance   : What should the sonographer do next?

Design (oracle generator):
  - The generator sees BOTH [SEEN] frames (clip_start -> query_time)
    AND [FUTURE] frames (query_time -> clip_end), plus the FULL ASR.
  - It must phrase the QUESTION as if only [SEEN] is available
    (no future leakage in the question itself).
  - The ANSWER may use BOTH sets as ground truth — it should describe
    the real intent / real next action that ACTUALLY occurs in the
    video, not a speculation.

Validation against information leakage is performed by `scripts/qa_validator.py`
(Gemini 2.5 Pro via OpenRouter).

Usage:
    export OPENAI_API_KEY="sk-..."
    python scripts/streaming_qa_generation.py --video path.mp4 --clips results/clips/ID_clips.json
"""

import os
import sys
import json
import time
import re
import base64
import argparse
from pathlib import Path

# Auto-load .env from project root
sys.path.insert(0, str(Path(__file__).resolve().parent))
import _env_loader  # noqa: F401

import cv2
import numpy as np


# ============================================================================
# Configuration
# ============================================================================

OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY", "")
MODEL = "gpt-4o"
TIME_RATIOS = [0.25, 0.5, 0.75]
SEEN_FRAMES = 3
FUTURE_FRAMES = 3
FRAME_SIZE = (512, 512)

STREAMING_QA_TYPES = ["sonographer_intent", "next_action_guidance"]

STREAMING_QA_PROMPT = """You are a senior ultrasound instructor and medical education expert.

You are observing an ultrasound teaching clip. The clip runs from t={clip_start:.0f}s to t={clip_end:.0f}s.

The "query time" is t={current_time:.0f}s — this is the moment a learner pauses and asks a question while watching the video live.

I am giving you TWO sets of frames, in temporal order:

[SEEN] frames ({clip_start:.0f}s -> {current_time:.0f}s) — content the learner has already watched.
[FUTURE] frames ({current_time:.0f}s -> {clip_end:.0f}s) — content that comes AFTER the question, NOT visible to the learner.

You also have the FULL ASR transcript of the clip (no truncation):
\"{full_asr}\"

Clip topic: {topic}

Your job: write exactly TWO question-answer pairs (one of each required type).

Required types:
  1. "sonographer_intent"     — what the operator is trying to accomplish at this moment
  2. "next_action_guidance"   — what the sonographer should do next

CRITICAL writing rules — apply to BOTH types:

▸ The QUESTION must be writable based ONLY on [SEEN] (the learner has not watched [FUTURE] yet).
  - Phrase it naturally as a learner pausing live: "at this point...", "right now...", "based on what we've seen so far..."
  - The question must NOT reveal or reference any content that only exists in [FUTURE].

▸ The ANSWER may use the FULL clip context ([SEEN] + [FUTURE] + full ASR) as ground truth.
  - The answer should describe the REAL intent or REAL next action that ACTUALLY occurs in the video.
  - Be specific: "the operator tilts the probe cranially and increases depth to visualize the upper pole" rather than vague guesses.
  - If [FUTURE] shows the operator pausing to explain rather than performing a maneuver, say so accurately.
  - Do NOT speculate beyond what the frames + ASR support.

Output ONLY a JSON array of 2 objects (no markdown, no other text):
[
  {{"type": "sonographer_intent",   "timestamp": {current_time:.2f}, "question": "...", "answer": "..."}},
  {{"type": "next_action_guidance", "timestamp": {current_time:.2f}, "question": "...", "answer": "..."}}
]"""


# ============================================================================
# Frame Extraction
# ============================================================================

def extract_clip_frames(video_path, start_sec, end_sec, num_frames):
    """Extract evenly-spaced frames from a time range."""
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise ValueError(f"Cannot open video: {video_path}")
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    start_frame = int(start_sec * fps)
    end_frame = max(int(end_sec * fps), start_frame + 1)
    indices = np.linspace(start_frame, end_frame, num_frames, dtype=int)
    frames = []
    for idx in indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(idx))
        ret, frame = cap.read()
        if ret:
            frame = cv2.resize(frame, FRAME_SIZE)
            frames.append(frame)
    cap.release()
    return frames


def frame_to_base64(frame):
    """Convert OpenCV frame to base64 JPEG."""
    _, buffer = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 85])
    return base64.b64encode(buffer).decode('utf-8')


def _parse_json_array(raw):
    """Robust JSON array extraction."""
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        pass
    m = re.search(r'```json\s*(.+?)```', raw, re.DOTALL)
    if m:
        try:
            return json.loads(m.group(1))
        except json.JSONDecodeError:
            pass
    m = re.search(r'\[.+\]', raw, re.DOTALL)
    if m:
        try:
            return json.loads(m.group(0))
        except json.JSONDecodeError:
            pass
    return None


# ============================================================================
# Streaming QA Generation (oracle: sees SEEN + FUTURE)
# ============================================================================

def generate_streaming_qa(video_path, clip, ratio=0.5, api_key=None):
    """
    Generate 2 streaming QA at a specific time anchor within a clip.

    The generator is an ORACLE: it receives SEEN frames + FUTURE frames + full ASR.
    It must, however, phrase the QUESTION as if only [SEEN] were available, while
    the ANSWER is allowed to use the full clip context as ground truth.
    """
    from openai import OpenAI

    api_key = api_key or OPENAI_API_KEY
    if not api_key:
        raise ValueError("OPENAI_API_KEY not set")
    client = OpenAI(api_key=api_key)

    clip_start = clip['start']
    clip_end = clip['end']
    duration = clip['duration']
    current_time = clip_start + duration * ratio

    seen_frames = extract_clip_frames(str(video_path), clip_start, current_time,
                                       SEEN_FRAMES)
    future_frames = extract_clip_frames(str(video_path), current_time, clip_end,
                                         FUTURE_FRAMES)

    prompt = STREAMING_QA_PROMPT.format(
        clip_start=clip_start,
        clip_end=clip_end,
        current_time=current_time,
        full_asr=clip.get('text', '')[:3500],
        topic=clip.get('topic', 'ultrasound'),
    )

    # Build content: [SEEN block] then [FUTURE block] then prompt
    content = []
    content.append({
        "type": "text",
        "text": f"=== [SEEN] frames ({clip_start:.0f}s -> {current_time:.0f}s) ==="
    })
    for frame in seen_frames:
        b64 = frame_to_base64(frame)
        content.append({
            "type": "image_url",
            "image_url": {"url": f"data:image/jpeg;base64,{b64}", "detail": "low"},
        })
    content.append({
        "type": "text",
        "text": f"=== [FUTURE] frames ({current_time:.0f}s -> {clip_end:.0f}s) ==="
    })
    for frame in future_frames:
        b64 = frame_to_base64(frame)
        content.append({
            "type": "image_url",
            "image_url": {"url": f"data:image/jpeg;base64,{b64}", "detail": "low"},
        })
    content.append({"type": "text", "text": prompt})

    t0 = time.time()
    response = client.chat.completions.create(
        model=MODEL,
        messages=[{"role": "user", "content": content}],
        temperature=0.3,
        max_tokens=900,
    )
    elapsed = time.time() - t0
    raw = response.choices[0].message.content
    tokens = response.usage.total_tokens if response.usage else 0
    print(f"    t={current_time:.0f}s (ratio={ratio}) | "
          f"{len(seen_frames)} SEEN + {len(future_frames)} FUTURE frames | "
          f"{elapsed:.1f}s | {tokens} tok")

    qa_pairs = _parse_json_array(raw)
    if qa_pairs is None:
        print(f"    WARNING: Failed to parse: {raw[:200]}")
        return []

    qa_pairs = [q for q in qa_pairs if q.get("type") in STREAMING_QA_TYPES]

    for qa in qa_pairs:
        qa['source'] = 'streaming'
        qa['clip_idx'] = clip['clip_idx']
        qa['clip_start'] = clip_start
        qa['clip_end'] = clip_end
        qa['query_time'] = round(current_time, 2)
        qa['ratio'] = ratio
        qa['topic'] = clip.get('topic', '')

    return qa_pairs


# ============================================================================
# Full Pipeline
# ============================================================================

def generate_streaming_qa_for_video(video_path, clips_path, output_dir="results/qa",
                                      api_key=None, single_clip=None, time_ratios=None):
    """Generate streaming QA for all clips. Output: {output_dir}/{video_id}_streaming_qa.json"""
    video_path = Path(video_path)
    video_id = video_path.stem

    with open(clips_path) as f:
        clips_data = json.load(f)
    clips = clips_data['clips']

    if single_clip is not None:
        clips = [c for c in clips if c['clip_idx'] == single_clip]

    ratios = time_ratios or TIME_RATIOS
    expected = len(clips) * len(ratios) * len(STREAMING_QA_TYPES)

    print(f"\n{'='*70}")
    print(f"Streaming QA Generation (Oracle): {video_id}")
    print(f"  Clips: {len(clips)} | Anchors per clip: {len(ratios)} | "
          f"Types: {len(STREAMING_QA_TYPES)}")
    print(f"  Expected QA total: {expected}")
    print(f"  Frames per anchor: {SEEN_FRAMES} SEEN + {FUTURE_FRAMES} FUTURE")
    print(f"  Model: {MODEL}")
    print(f"{'='*70}")

    all_qa = []

    for clip in clips:
        print(f"\n  Clip {clip['clip_idx']}: "
              f"{clip['start']:.0f}-{clip['end']:.0f}s | {clip.get('topic', '')[:40]}")
        for ratio in ratios:
            try:
                qa_pairs = generate_streaming_qa(str(video_path), clip,
                                                  ratio=ratio, api_key=api_key)
                for qa in qa_pairs:
                    qa['video_id'] = video_id
                all_qa.extend(qa_pairs)
            except Exception as e:
                print(f"    ERROR at ratio={ratio}: {e}")
            time.sleep(1)

    output = {
        'video_id': video_id,
        'video_path': str(video_path),
        'model': MODEL,
        'qa_types': STREAMING_QA_TYPES,
        'num_clips': len(clips),
        'time_ratios': ratios,
        'frames_per_anchor': {'seen': SEEN_FRAMES, 'future': FUTURE_FRAMES},
        'num_streaming_qa': len(all_qa),
        'streaming_qa': all_qa,
    }

    out_path = Path(output_dir) / f"{video_id}_streaming_qa.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, 'w', encoding='utf-8') as f:
        json.dump(output, f, ensure_ascii=False, indent=2)
    print(f"\n  Saved {len(all_qa)} streaming QA (unverified) to: {out_path}")
    print(f"  -> Next: scripts/qa_validator.py --streaming-qa {out_path} --video <video>")

    return output


# ============================================================================
# CLI
# ============================================================================

def main():
    parser = argparse.ArgumentParser(description="Streaming QA Generation (Oracle, GPT-4o)")
    parser.add_argument("--video", type=str, required=True, help="Video file path")
    parser.add_argument("--clips", type=str, required=True, help="Clips JSON path")
    parser.add_argument("--output-dir", type=str, default="results/qa")
    parser.add_argument("--api-key", type=str, default=None,
                        help="OpenAI API key (or set OPENAI_API_KEY env)")
    parser.add_argument("--single-clip", type=int, default=None,
                        help="Only process this clip index (debug)")
    parser.add_argument("--ratios", type=str, default=None,
                        help="Comma-separated time ratios, e.g. '0.25,0.5,0.75'")
    args = parser.parse_args()

    ratios = None
    if args.ratios:
        ratios = [float(x) for x in args.ratios.split(',')]

    generate_streaming_qa_for_video(
        args.video,
        args.clips,
        output_dir=args.output_dir,
        api_key=args.api_key,
        single_clip=args.single_clip,
        time_ratios=ratios,
    )


if __name__ == "__main__":
    main()
