"""Generate viz/data.json for the three Stage 1 conditions over a real video.

Reuses stage1.data.build_rows / frame_timestamps and stage1.model.messages so the
visualization shows exactly what the training pipeline feeds the model. Read-only
over the transcript and video; writes only viz/data.json.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from stage1.data import CONDITIONS, build_rows, frame_timestamps
from stage1 import model


def build_payload(transcript, video_url, duration, max_frames=120):
    rows = build_rows(transcript)
    by_sid = {}
    for row in rows:
        by_sid.setdefault(int(row["sentence_id"]), {})[row["condition"]] = row
    sentences = []
    for sid in sorted(by_sid):
        group = by_sid[sid]
        if set(group) != set(CONDITIONS):
            continue
        base = group["before_with_asr"]
        conditions = {}
        for cond in CONDITIONS:
            row = group[cond]
            win = [float(row["video_window"][0]), float(row["video_window"][1])]
            frames = {}
            for strategy in ("uniform", "recent_sparse"):
                frames[strategy] = [round(t, 3) for t in frame_timestamps(
                    win, max_frames=max_frames, sampling=strategy)]
            msgs = model.messages(row, [], target=row["target"])
            texts = [c["text"] for c in msgs[1]["content"] if c["type"] == "text"]
            conditions[cond] = {
                "video_window": win,
                "historical_asr_masked": bool(row["historical_asr_masked"]),
                "frames": frames,
                "prompt_texts": texts,
            }
        sentences.append({
            "sentence_id": sid,
            "sentence_window": [float(base["sentence_window"][0]), float(base["sentence_window"][1])],
            "target": base["target"],
            "history": base["historical_asr"],
            "conditions": conditions,
        })
    return {
        "video_id": transcript.get("video_id"),
        "video_url": video_url,
        "duration": float(duration),
        "max_frames": max_frames,
        "condition_order": list(CONDITIONS),
        "sentences": sentences,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--transcript", type=Path,
                        default=Path("local_asr_3videos/transcripts/8V649L5Q368.json"))
    parser.add_argument("--video-url", default="../azure_data/videos/8V649L5Q368.mp4")
    parser.add_argument("--output", type=Path, default=Path("viz/data.json"))
    parser.add_argument("--max-frames", type=int, default=120)
    args = parser.parse_args()
    transcript = json.loads(args.transcript.read_text(encoding="utf-8"))
    duration = float(transcript.get("duration_sec") or transcript.get("duration") or 0.0)
    payload = build_payload(transcript, args.video_url, duration, max_frames=args.max_frames)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print("Wrote", args.output, "with", len(payload["sentences"]), "sentence groups")


if __name__ == "__main__":
    main()
