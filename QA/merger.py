"""
QA Merger (new pipeline)
========================
Consumes:
  - offline QA JSON   (from scripts/qa_generation.py, i.e. results/qa/{id}_offline_qa.json)
  - validated streaming QA JSON (from QA/validator.py,  i.e. QA/results/{id}_streaming_qa_validated.json)
  - ASR transcript JSON  (from scripts/asr_pipeline.py)
  - clips JSON           (from scripts/video_segmentation.py)

Outputs (single JSONL file, one record per --out invocation):

  Default mode (`--per-video-record`, default):
      One record per video containing:
        - clip metadata
        - LiveCC-style text_stream
        - `qa` list mixing offline + streaming (streaming entries carry
          BOTH query_time and answer_time)

  Training-samples mode (`--expand-wait-answer`):
      One JSONL line PER training sample:
        - each streaming QA -> two lines: streaming_wait + streaming_answer
        - each offline QA   -> one line: offline_answer
      This is the format that (in a future stage) will drive
      Qwen2-VL-7B + <WAIT>/<ANSWER> SFT.

Usage:
    python QA/merger.py \\
        --video-id 8V649L5Q368 \\
        --transcript results/transcripts/8V649L5Q368.json \\
        --clips      results/clips/8V649L5Q368_clips.json \\
        --offline-qa results/qa/8V649L5Q368_offline_qa.json \\
        --streaming-qa QA/results/8V649L5Q368_streaming_qa_validated.json \\
        --out QA/results/8V649L5Q368.jsonl --overwrite

    # Expand to WAIT/ANSWER training samples:
    python QA/merger.py \\
        ...same as above... \\
        --expand-wait-answer --window-sec 30 \\
        --out QA/results/8V649L5Q368_training_samples.jsonl --overwrite
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path


DEFAULT_WINDOW_SEC = 30.0
WAIT_TARGET = "<WAIT> Not enough information yet. More video is needed."


# ============================================================================
# IO helpers
# ============================================================================

def _load(path):
    with open(path) as f:
        return json.load(f)


def _build_text_stream(asr_data):
    stream = []
    for seg in asr_data.get('segments', []):
        stream.append([
            float(seg['start']),
            float(seg['end']),
            seg['text'].strip(),
        ])
    return stream


# ============================================================================
# Normalisation
# ============================================================================

def _normalize_offline_qa(qa, default_clip_lookup):
    """Add a numeric timestamp (clip midpoint) so mixed sort works."""
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
        "query_time": None,
        "answer_time": None,
        "ratio": None,
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
        "query_time": qa.get('query_time'),
        "answer_time": qa.get('answer_time'),
        "evidence_window": qa.get('evidence_window'),
        "evidence": qa.get('evidence'),
        "ratio": qa.get('ratio'),
        "validation": qa.get('validation'),
    }


# ============================================================================
# Mode 1: per-video record (mixed offline + streaming)
# ============================================================================

def build_per_video_record(video_id, transcript_path, clips_path,
                           offline_qa_path=None, streaming_qa_path=None,
                           video_rel_path=None):
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
# Mode 2: WAIT/ANSWER training-sample expansion
# ============================================================================

def _clamp_window(t_end, clip_start, window_sec):
    """Return [t_end - window_sec, t_end], clamped to >= clip_start."""
    a = max(clip_start, t_end - window_sec)
    return a, t_end


def _stream_qa_id(video_id, qa, idx):
    """Deterministic sample id."""
    return (
        f"{video_id}_c{qa['clip_idx']}_stream_{qa['type']}_"
        f"tq{int(qa['query_time'])}_ta{int(qa['answer_time'])}_{idx:03d}"
    )


def _offline_qa_id(video_id, qa, idx):
    return f"{video_id}_c{qa['clip_idx']}_offline_{qa['type']}_{idx:03d}"


def expand_training_samples(video_id, transcript_path, clips_path,
                            offline_qa_path=None, streaming_qa_path=None,
                            video_rel_path=None,
                            window_sec=DEFAULT_WINDOW_SEC):
    """
    Return a list of training-sample dicts (see QA/schema.md §3).
    """
    clips_data = _load(clips_path)
    clip_lookup = {c['clip_idx']: c for c in clips_data['clips']}

    samples = []
    video_field = video_rel_path or f"videos/{video_id}.mp4"

    # --- Offline QA -> one ANSWER sample per QA (whole clip window) ---
    if offline_qa_path:
        off = _load(offline_qa_path)
        for idx, qa in enumerate(off.get('qa_pairs', [])):
            cs = qa.get('clip_start')
            ce = qa.get('clip_end')
            if cs is None or ce is None:
                c = clip_lookup.get(qa.get('clip_idx'), {})
                cs = c.get('start', 0.0)
                ce = c.get('end', 0.0)
            samples.append({
                "sample_type": "offline_answer",
                "video": video_field,
                "video_id": video_id,
                "clip_idx": qa['clip_idx'],
                "video_window": [round(float(cs), 2), round(float(ce), 2)],
                "question": qa['question'],
                "target": f"<ANSWER> {qa['answer']}",
                "qa_type": qa['type'],
                "meta": {
                    "source_qa_id": _offline_qa_id(video_id, qa, idx),
                    "topic": qa.get('topic', ''),
                },
            })

    # --- Streaming QA -> WAIT + ANSWER pair per QA ---
    if streaming_qa_path:
        stream = _load(streaming_qa_path)
        for idx, qa in enumerate(stream.get('streaming_qa', [])):
            clip_start = float(qa['clip_start'])
            query_time = float(qa['query_time'])
            answer_time = float(qa['answer_time'])
            source_id = _stream_qa_id(video_id, qa, idx)
            common_meta = {
                "source_qa_id": source_id,
                "query_time": query_time,
                "answer_time": answer_time,
                "evidence_window": qa.get('evidence_window'),
                "topic": qa.get('topic', ''),
            }

            wait_a, wait_b = _clamp_window(query_time, clip_start, window_sec)
            samples.append({
                "sample_type": "streaming_wait",
                "video": video_field,
                "video_id": video_id,
                "clip_idx": qa['clip_idx'],
                "video_window": [round(wait_a, 2), round(wait_b, 2)],
                "question": qa['question'],
                "target": WAIT_TARGET,
                "qa_type": qa['type'],
                "meta": common_meta,
            })

            ans_a, ans_b = _clamp_window(answer_time, clip_start, window_sec)
            samples.append({
                "sample_type": "streaming_answer",
                "video": video_field,
                "video_id": video_id,
                "clip_idx": qa['clip_idx'],
                "video_window": [round(ans_a, 2), round(ans_b, 2)],
                "question": qa['question'],
                "target": f"<ANSWER> {qa['answer']}",
                "qa_type": qa['type'],
                "meta": common_meta,
            })

    return samples


# ============================================================================
# CLI
# ============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="QA Merger (new pipeline; supports --expand-wait-answer)"
    )
    parser.add_argument("--video-id", type=str, required=True)
    parser.add_argument("--transcript", type=str, required=True,
                        help="ASR transcript JSON")
    parser.add_argument("--clips", type=str, required=True,
                        help="Clips JSON from segmentation")
    parser.add_argument("--offline-qa", type=str, default=None,
                        help="Offline QA JSON (optional)")
    parser.add_argument("--streaming-qa", type=str, default=None,
                        help="Validated streaming QA JSON (optional)")
    parser.add_argument("--video-rel-path", type=str, default=None)
    parser.add_argument("--out", type=str, required=True,
                        help="Output JSONL path")
    parser.add_argument("--overwrite", action="store_true",
                        help="Truncate the output file before writing")
    parser.add_argument("--expand-wait-answer", action="store_true",
                        help="Emit WAIT + ANSWER training samples "
                             "(one line per sample) instead of one "
                             "per-video record")
    parser.add_argument("--window-sec", type=float, default=DEFAULT_WINDOW_SEC,
                        help=f"Observation window (seconds) for WAIT/ANSWER "
                             f"samples (default {DEFAULT_WINDOW_SEC})")
    args = parser.parse_args()

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    mode = 'w' if args.overwrite else 'a'

    if args.expand_wait_answer:
        samples = expand_training_samples(
            args.video_id,
            args.transcript,
            args.clips,
            offline_qa_path=args.offline_qa,
            streaming_qa_path=args.streaming_qa,
            video_rel_path=args.video_rel_path,
            window_sec=args.window_sec,
        )
        with open(out_path, mode, encoding='utf-8') as f:
            for s in samples:
                f.write(json.dumps(s, ensure_ascii=False) + "\n")

        type_counts = {}
        for s in samples:
            type_counts[s['sample_type']] = type_counts.get(s['sample_type'], 0) + 1

        print(f"\n{'='*70}")
        print(f"MERGE (training samples mode): {args.video_id}")
        print(f"{'='*70}")
        print(f"  Window (sec)   : {args.window_sec}")
        print(f"  Total samples  : {len(samples)}")
        for k, v in sorted(type_counts.items()):
            print(f"    {k:20s}: {v}")
        print(f"  Wrote to       : {out_path} (mode={mode})")
        return

    # default: per-video record
    record = build_per_video_record(
        args.video_id,
        args.transcript,
        args.clips,
        offline_qa_path=args.offline_qa,
        streaming_qa_path=args.streaming_qa,
        video_rel_path=args.video_rel_path,
    )
    with open(out_path, mode, encoding='utf-8') as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")

    print(f"\n{'='*70}")
    print(f"MERGE (per-video record): {args.video_id}")
    print(f"{'='*70}")
    print(f"  Clips                : {record['num_clips']}")
    print(f"  text_stream segments : {len(record['text_stream'])}")
    print(f"  Total QA             : {record['num_qa']}")
    for t, n in sorted(record['qa_type_counts'].items()):
        print(f"    {t:25s}: {n}")
    print(f"  Wrote to             : {out_path} (mode={mode})")


if __name__ == "__main__":
    main()
