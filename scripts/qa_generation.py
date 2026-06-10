"""
Offline QA Generation Pipeline using GPT-4o Vision
====================================================
Generates 3 types of QA pairs that benefit from FULL CLIP context:
  - scene_description : 整段过程性描述（"先...然后...最后..."）
  - fine_grained      : 解剖标志、回声、探头摆位、测量值等细节
  - knowledge         : 与 clip 主题相关的医学知识 / 临床意义

Streaming-only types (intent, next_action_guidance) are handled by
`streaming_qa_generation.py` and validated by `qa_validator.py`.

Usage:
    export OPENAI_API_KEY="sk-..."
    python scripts/qa_generation.py --video path.mp4 --clips results/clips/ID_clips.json
"""

import os
import sys
import json
import base64
import time
import argparse
import re
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
MAX_FRAMES = 6
FRAME_SIZE = (512, 512)

OFFLINE_QA_TYPES = ["scene_description", "fine_grained", "knowledge"]

QA_SYSTEM_PROMPT = """You are a senior ultrasound instructor and medical education expert.

Given a set of video frames from an ultrasound teaching clip and the corresponding speech transcription, generate exactly THREE question-answer pairs (one of each required type).

IMPORTANT: These frames are from a VIDEO (shown in temporal order, Frame 1 is earliest, last frame is latest). The viewer is allowed to see the ENTIRE clip, so describe the whole process holistically.

Required QA Types (generate exactly one of each):
1. "scene_description"  — What happens in this clip from beginning to end? Describe the scanning workflow, anatomical structures revealed, and how the view evolves over time. Use temporal language ("first... then... finally...").
2. "fine_grained"       — Pick the most clinically salient visual detail in the clip and describe it precisely: anatomical landmarks visible, echogenicity patterns, probe orientation/positioning, depth/gain settings if observable, or measurement values shown on screen.
3. "knowledge"          — Provide medical/educational knowledge directly relevant to what is shown or discussed: clinical significance, diagnostic criteria, common pitfalls, normal ranges, or pathophysiology.

Output ONLY a JSON array of 3 objects with these fields:
[
  {"type": "scene_description", "question": "...", "answer": "...", "timestamp_hint": "early|middle|late|whole_clip"},
  {"type": "fine_grained",      "question": "...", "answer": "...", "timestamp_hint": "..."},
  {"type": "knowledge",         "question": "...", "answer": "...", "timestamp_hint": "..."}
]

Important rules:
- Questions must be educational and clinically relevant
- Answers should be detailed but concise (2-4 sentences)
- For scene_description, reference temporal evolution explicitly
- For fine_grained, anchor descriptions to specific visible features
- For knowledge, the answer should NOT just repeat what is visible — add medical context
- Output ONLY a valid JSON array, no markdown fences, no other text"""


# ============================================================================
# Frame Extraction
# ============================================================================

def extract_clip_frames(video_path, start_sec, end_sec, num_frames=MAX_FRAMES):
    """Extract evenly-spaced frames from a clip timerange."""
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise ValueError(f"Cannot open video: {video_path}")

    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    start_frame = int(start_sec * fps)
    end_frame = int(end_sec * fps)
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
    """Convert OpenCV frame to base64 JPEG string."""
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
# GPT-4o API Call
# ============================================================================

def generate_qa_for_clip(frames, asr_text, video_type="ultrasound_tutorial",
                          topic="", api_key=None):
    """Send frames + ASR to GPT-4o Vision and return 3 offline QA pairs."""
    from openai import OpenAI

    api_key = api_key or OPENAI_API_KEY
    if not api_key:
        raise ValueError("OPENAI_API_KEY not set. Export it or pass via --api-key")

    client = OpenAI(api_key=api_key)

    content = []
    for frame in frames:
        b64 = frame_to_base64(frame)
        content.append({
            "type": "image_url",
            "image_url": {"url": f"data:image/jpeg;base64,{b64}", "detail": "low"}
        })

    user_text = f"""Video type: {video_type}
Clip topic: {topic}
Number of frames shown: {len(frames)} (evenly sampled across the entire clip)

Speech transcription during this clip:
\"{asr_text[:3000]}\"

Generate exactly 3 QA pairs following the system instructions: one scene_description, one fine_grained, one knowledge."""

    content.append({"type": "text", "text": user_text})

    t0 = time.time()
    response = client.chat.completions.create(
        model=MODEL,
        messages=[
            {"role": "system", "content": QA_SYSTEM_PROMPT},
            {"role": "user", "content": content}
        ],
        temperature=0.3,
        max_tokens=1500,
    )
    elapsed = time.time() - t0

    raw_output = response.choices[0].message.content
    tokens_used = response.usage.total_tokens if response.usage else 0
    print(f"  GPT-4o: {elapsed:.1f}s | {tokens_used} tokens")

    qa_pairs = _parse_json_array(raw_output)
    if qa_pairs is None:
        print(f"  WARNING: Failed to parse QA output: {raw_output[:200]}")
        return []

    # Defensive: only keep the 3 expected types
    qa_pairs = [q for q in qa_pairs if q.get("type") in OFFLINE_QA_TYPES]
    return qa_pairs


# ============================================================================
# Full Pipeline
# ============================================================================

def generate_qa_for_video(video_path, clips_path=None, transcript_path=None,
                           output_dir="results/qa", api_key=None,
                           single_clip=None):
    """Generate offline QA for all clips. Output: {output_dir}/{video_id}_offline_qa.json"""
    video_path = Path(video_path)
    video_id = video_path.stem

    if clips_path:
        with open(clips_path) as f:
            clips_data = json.load(f)
        clips = clips_data['clips']
    elif transcript_path:
        with open(transcript_path) as f:
            asr_data = json.load(f)
        clips = [{
            'clip_idx': 0,
            'start': 0,
            'end': asr_data['duration_sec'],
            'duration': asr_data['duration_sec'],
            'text': asr_data.get('full_text', ' '.join(s['text'] for s in asr_data['segments'])),
            'topic': '',
        }]
    else:
        raise ValueError("Provide either --clips or --transcript")

    if single_clip is not None:
        clips = [c for c in clips if c['clip_idx'] == single_clip]

    print(f"\n{'='*70}")
    print(f"Offline QA Generation: {video_id}")
    print(f"  Video: {video_path}")
    print(f"  Clips: {len(clips)} | Types per clip: {len(OFFLINE_QA_TYPES)}")
    print(f"  Model: {MODEL}")
    print(f"{'='*70}")

    all_qa = []

    for clip in clips:
        print(f"\n  Clip {clip['clip_idx']}: {clip['start']:.1f}-{clip['end']:.1f}s "
              f"({clip['duration']:.0f}s) | {clip.get('topic', '')[:40]}")

        frames = extract_clip_frames(str(video_path), clip['start'], clip['end'])
        print(f"    Extracted {len(frames)} frames")

        asr_text = clip.get('text', '')
        if not asr_text:
            print(f"    WARNING: No ASR text for this clip, skipping")
            continue

        try:
            qa_pairs = generate_qa_for_clip(
                frames, asr_text,
                video_type=clip.get('video_type', 'ultrasound_tutorial'),
                topic=clip.get('topic', ''),
                api_key=api_key,
            )
        except Exception as e:
            print(f"    ERROR: {e}")
            qa_pairs = []

        for qa in qa_pairs:
            qa['source'] = 'offline'
            qa['clip_idx'] = clip['clip_idx']
            qa['clip_start'] = clip['start']
            qa['clip_end'] = clip['end']
            qa['video_id'] = video_id
            qa['topic'] = clip.get('topic', '')

        all_qa.extend(qa_pairs)
        print(f"    Generated {len(qa_pairs)} QA pairs")

        time.sleep(1)

    output = {
        'video_id': video_id,
        'video_path': str(video_path),
        'model': MODEL,
        'qa_types': OFFLINE_QA_TYPES,
        'num_clips': len(clips),
        'num_qa_pairs': len(all_qa),
        'qa_pairs': all_qa,
    }

    out_path = Path(output_dir) / f"{video_id}_offline_qa.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, 'w', encoding='utf-8') as f:
        json.dump(output, f, ensure_ascii=False, indent=2)
    print(f"\n  Saved {len(all_qa)} offline QA pairs to: {out_path}")

    return output


# ============================================================================
# CLI
# ============================================================================

def main():
    parser = argparse.ArgumentParser(description="Offline QA Generation with GPT-4o Vision")
    parser.add_argument("--video", type=str, required=True, help="Video file path")
    parser.add_argument("--clips", type=str, help="Clips JSON from segmentation")
    parser.add_argument("--transcript", type=str, help="Transcript JSON (if no clips)")
    parser.add_argument("--output-dir", type=str, default="results/qa")
    parser.add_argument("--api-key", type=str, default=None,
                        help="OpenAI API key (or set OPENAI_API_KEY env)")
    parser.add_argument("--single-clip", type=int, default=None,
                        help="Only process this clip index")
    args = parser.parse_args()

    result = generate_qa_for_video(
        args.video,
        clips_path=args.clips,
        transcript_path=args.transcript,
        output_dir=args.output_dir,
        api_key=args.api_key,
        single_clip=args.single_clip,
    )

    print(f"\n{'='*70}")
    print(f"Summary: {result['num_qa_pairs']} offline QA pairs from {result['num_clips']} clips")
    for qa in result['qa_pairs'][:5]:
        print(f"\n  [{qa['type']}] Q: {qa['question'][:80]}")
        print(f"           A: {qa['answer'][:100]}")


if __name__ == "__main__":
    main()
