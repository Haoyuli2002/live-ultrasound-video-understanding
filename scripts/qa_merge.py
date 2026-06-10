"""
QA Merger
==========
Combine offline QA (scene_description / fine_grained / knowledge) and
validated streaming QA (sonographer_intent / next_action_guidance) into a
single LiveCC-style JSONL for training and evaluation.

Output schema (one JSON per line):
{
  "video":  "videos/{video_id}.mp4",
  "video_id": "...",
  "clips": [...],            # from clips JSON
  "text_stream": [           # ASR-style stream (LiveCC-compatible)
      [start, end, text], ...
  ],
  "qa": [                    # all QA pairs, sorted by anchor time
      {
        "type": "scene_description" | "fine_grained" | "knowledge"
              | "sonographer_intent" | "next_action_guidance",
        "source": "offline" | "streaming",
        "question": "...",
        "answer":   "...",
        "clip_idx": int,
        "clip_start": float,
        "clip_end":   float,
        "timestamp":  float,        # anchor time (query_time for streaming, clip mid for offline)
        "ratio":      float | null,
        "validation": {...} | null
      },
      ...
  ]
}

Usage:
    python scripts/qa_merge.py \\
        --video-id 8V649L5Q368 \\
        --transcript results/transcripts/8V649L5Q368.json \\
        --clips      results/clips/8V649L5Q368_clips.json \\
        --offline-qa results/qa/8V649L5Q368_offline_qa.json \\
        --streaming-qa results/qa/8V649L5Q368_streaming_qa_validated.json \\
        --out        results/training_data/8V649L5Q368.jsonl
"""

import argparse
import json
from pathlib import Path


# ============================================================================
# Helpers
# ============================================================================

def _load(path):
    with open(path) as f:
        return json.load(f)


def _build_text_stream(asr_data):
    """LiveCC text_stream format: [[start, end, text], ...]"""
    stream = []
    for seg in asr_data.get('segments', []):
        stream.append([
            float(seg['start']),
            float(seg['end']),
            seg['text'].strip(),
        ])
    return stream


def _normalize_offline_qa(qa, default_clip_lookup):
    """Add a numeric `timestamp` to offline QA (clip midpoint) for sorting."""
    cs = qa.get('clip_start')
    ce = qa.get('clip_end')
    if cs is None or ce is None:
        c = default_clip_lookup.get(qa.get('clip_idx'), {})
        cs = c.get('start', 0.0)
        ce = c.get('end', 0.0)
    return {
        "type": qa['type'],
        "source": qa.get('source', 'offline'),
        "question": qa['question'],
        "answer": qa['answer'],
        "clip_idx": qa['clip_idx'],
        "clip_start": cs,
        "clip_end": ce,
        "timestamp": (cs + ce) / 2.0,
        "ratio": None,
        "timestamp_hint": qa.get('timestamp_hint'),
        "validation": None,
    }


def _normalize_streaming_qa(qa):
    return {
        "type": qa['type'],
        "source": qa.get('source', 'streaming'),
        "question": qa['question'],
        "answer": qa['answer'],
        "clip_idx": qa['clip_idx'],
        "clip_start": qa['clip_start'],
        "clip_end": qa['clip_end'],
        "timestamp": qa.get('query_time', qa['clip_start']),
        "ratio": qa.get('ratio'),
        "timestamp_hint": None,
        "validation": qa.get('validation'),
    }


# ============================================================================
# Main merge
# ============================================================================

def merge_video_qa(video_id, transcript_path, clips_path,
                    offline_qa_path=None, streaming_qa_path=None,
                    video_rel_path=None):
    """
    Merge per-video artifacts into a single LiveCC-style record.

    Returns the merged dict (not yet serialized).
    """
    asr_data = _load(transcript_path)
    clips_data = _load(clips_path)
    clips = clips_data['clips']
    clip_lookup = {c['clip_idx']: c for c in clips}

    qa_records = []

    if offline_qa_path:
        off = _load(offline_qa_path)
        for q in off.get('qa_pairs', []):
            qa_records.append(_normalize_offline_qa(q, clip_lookup))

    if streaming_qa_path:
        stream = _load(streaming_qa_path)
        for q in stream.get('streaming_qa', []):
            qa_records.append(_normalize_streaming_qa(q))

    # Sort by timestamp for streaming-friendly consumption
    qa_records.sort(key=lambda q: (q['timestamp'], q['clip_idx']))

    record = {
        "video": video_rel_path or f"videos/{video_id}.mp4",
        "video_id": video_id,
        "duration_sec": asr_data.get('duration_sec'),
        "language": asr_data.get('language'),
        "num_clips": len(clips),
        "clips": [
            {
                "clip_idx": c['clip_idx'],
                "start": c['start'],
                "end": c['end'],
                "duration": c.get('duration', c['end'] - c['start']),
                "topic": c.get('topic', ''),
            }
            for c in clips
        ],
        "text_stream": _build_text_stream(asr_data),
        "num_qa": len(qa_records),
        "qa_type_counts": _count_types(qa_records),
        "qa": qa_records,
    }
    return record


def _count_types(qa_records):
    counts = {}
    for q in qa_records:
        counts[q['type']] = counts.get(q['type'], 0) + 1
    return counts


# ============================================================================
# CLI
# ============================================================================

def main():
    parser = argparse.ArgumentParser(description="Merge offline + streaming QA -> LiveCC JSONL")
    parser.add_argument("--video-id", type=str, required=True)
    parser.add_argument("--transcript", type=str, required=True,
                        help="ASR transcript JSON")
    parser.add_argument("--clips", type=str, required=True,
                        help="Clips JSON from segmentation")
    parser.add_argument("--offline-qa", type=str, default=None,
                        help="Offline QA JSON (optional)")
    parser.add_argument("--streaming-qa", type=str, default=None,
                        help="Validated streaming QA JSON (optional)")
    parser.add_argument("--video-rel-path", type=str, default=None,
                        help="Path to write into 'video' field (default: videos/{id}.mp4)")
    parser.add_argument("--out", type=str, required=True,
                        help="Output JSONL path (one record appended)")
    parser.add_argument("--overwrite", action="store_true",
                        help="Truncate the output file before writing")
    args = parser.parse_args()

    record = merge_video_qa(
        args.video_id,
        args.transcript,
        args.clips,
        offline_qa_path=args.offline_qa,
        streaming_qa_path=args.streaming_qa,
        video_rel_path=args.video_rel_path,
    )

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    mode = 'w' if args.overwrite else 'a'
    with open(out_path, mode, encoding='utf-8') as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")

    print(f"\n{'='*70}")
    print(f"MERGE COMPLETE: {args.video_id}")
    print(f"{'='*70}")
    print(f"  Clips: {record['num_clips']}")
    print(f"  text_stream segments: {len(record['text_stream'])}")
    print(f"  Total QA: {record['num_qa']}")
    for t, n in sorted(record['qa_type_counts'].items()):
        print(f"    {t:25s}: {n}")
    print(f"  Wrote to: {out_path} (mode={mode})")


if __name__ == "__main__":
    main()