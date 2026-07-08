"""
Convert Dual QA annotation JSON files to training JSONL.

Input:
    results/qa/*_dual_qa_clip*.json

Output:
    JSONL where each line is one training sample.

Each dual QA annotation can produce:
    - 1 offline_answer sample
    - 1 streaming_wait sample if answerable_at_question_time == false
    - 1 streaming_answer sample

Default output keeps metadata around the messages for debugging.
Use --livecc-format to output only the raw conversation list per line.

Examples:
    python scripts/dual_qa_to_jsonl.py \
      --input results/qa/8V649L5Q368_dual_qa_clip1_v3.json \
      --out results/training_data/dual_qa_train.jsonl \
      --overwrite

    python scripts/dual_qa_to_jsonl.py \
      --input-glob "results/qa/*_dual_qa_clip*_v3.json" \
      --out results/training_data/dual_qa_train.jsonl \
      --overwrite
"""

from __future__ import annotations

import argparse
import glob
import json
from pathlib import Path


WAIT_TARGET = "<WAIT> Not enough information to answer yet. More video is needed."
ANSWER_PREFIX = "<ANSWER> "


def _video_block(video_path: str, start: float, end: float) -> dict:
    return {
        "type": "video",
        "video": video_path,
        "video_start": float(start),
        "video_end": float(end),
    }


def _text_block(text: str) -> dict:
    return {"type": "text", "text": text}


def _conversation(video_path: str, start: float, end: float, prompt: str, target: str) -> list[dict]:
    return [
        {
            "role": "user",
            "content": [
                _video_block(video_path, start, end),
                _text_block(prompt),
            ],
        },
        {
            "role": "assistant",
            "content": [
                _text_block(target),
            ],
        },
    ]


def _sample_or_messages(sample: dict, livecc_format: bool):
    if livecc_format:
        return sample["messages"]
    return sample


def make_offline_sample(data: dict, qa: dict, *, livecc_format: bool = False):
    video_path = data["video_path"]
    clip_idx = data["clip_idx"]
    video_id = data["video_id"]
    qa_id = qa.get("qa_id") or f"{video_id}_c{clip_idx}_offline_001"

    prompt = (
        "You are an ultrasound assistant.\n"
        f"Question: {qa['question']}\n"
        "Answer based on the full ultrasound clip."
    )
    target = ANSWER_PREFIX + qa["answer"].strip()

    sample = {
        "sample_id": f"{qa_id}_answer",
        "sample_type": "offline_answer",
        "video_id": video_id,
        "clip_idx": clip_idx,
        "qa_id": qa_id,
        "source": "offline",
        "qa_type": qa.get("qa_type"),
        "input_policy": "full_clip",
        "messages": _conversation(
            video_path,
            qa.get("input_video_start", data["clip_start"]),
            qa.get("input_video_end", data["clip_end"]),
            prompt,
            target,
        ),
    }
    return _sample_or_messages(sample, livecc_format)


def _streaming_prompt(qa: dict, *, mode: str) -> str:
    question_time = qa.get("question_time")
    answer_time = qa.get("answer_time")
    history_summary = (qa.get("history_summary") or "").strip()

    lines = [
        "You are answering a streaming ultrasound question.",
    ]
    if history_summary:
        lines.append(f"History summary: {history_summary}")
    if question_time is not None:
        lines.append(f"Question asked at {float(question_time):.2f}s:")
    lines.append(qa["question"])

    if mode == "wait":
        lines.append("If the answer is not supported by the video seen so far, respond with <WAIT>.")
    else:
        lines.append(
            f"Answer based only on the video and context available up to {float(answer_time):.2f}s. "
            "If insufficient, respond with <WAIT>."
        )
    return "\n".join(lines)


def make_streaming_wait_sample(data: dict, qa: dict, *, livecc_format: bool = False):
    video_path = data["video_path"]
    clip_idx = data["clip_idx"]
    video_id = data["video_id"]
    qa_id = qa.get("qa_id") or f"{video_id}_c{clip_idx}_stream_001"

    clip_start = float(data["clip_start"])
    question_time = float(qa["question_time"])
    window_sec = float((qa.get("visual_context") or {}).get("duration_sec", 30.0))
    start = max(clip_start, question_time - window_sec)
    end = question_time

    sample = {
        "sample_id": f"{qa_id}_wait",
        "sample_type": "streaming_wait",
        "video_id": video_id,
        "clip_idx": clip_idx,
        "qa_id": qa_id,
        "source": "streaming",
        "qa_type": qa.get("qa_type"),
        "question_time": question_time,
        "answer_time": qa.get("answer_time"),
        "input_policy": "seen_until_question_time",
        "messages": _conversation(
            video_path,
            start,
            end,
            _streaming_prompt(qa, mode="wait"),
            WAIT_TARGET,
        ),
    }
    return _sample_or_messages(sample, livecc_format)


def make_streaming_answer_sample(data: dict, qa: dict, *, livecc_format: bool = False):
    video_path = data["video_path"]
    clip_idx = data["clip_idx"]
    video_id = data["video_id"]
    qa_id = qa.get("qa_id") or f"{video_id}_c{clip_idx}_stream_001"

    vc = qa.get("visual_context") or {}
    start = vc.get("start")
    end = vc.get("end")

    if start is None or end is None:
        answer_time = float(qa["answer_time"])
        clip_start = float(data["clip_start"])
        window_sec = float(vc.get("duration_sec", 30.0))
        start = max(clip_start, answer_time - window_sec)
        end = answer_time

    sample = {
        "sample_id": f"{qa_id}_answer",
        "sample_type": "streaming_answer",
        "video_id": video_id,
        "clip_idx": clip_idx,
        "qa_id": qa_id,
        "source": "streaming",
        "qa_type": qa.get("qa_type"),
        "question_time": qa.get("question_time"),
        "answer_time": qa.get("answer_time"),
        "input_policy": "seen_until_answer_time",
        "visual_context": vc,
        "memory": qa.get("memory"),
        "history_summary": qa.get("history_summary", ""),
        "messages": _conversation(
            video_path,
            start,
            end,
            _streaming_prompt(qa, mode="answer"),
            ANSWER_PREFIX + qa["answer"].strip(),
        ),
    }
    return _sample_or_messages(sample, livecc_format)


def convert_dual_qa_file(
    path: str | Path,
    *,
    include_wait: bool = True,
    livecc_format: bool = False,
) -> list:
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)

    samples = []

    for qa in data.get("offline_qa", []):
        if qa and qa.get("question") and qa.get("answer"):
            samples.append(make_offline_sample(data, qa, livecc_format=livecc_format))

    for qa in data.get("streaming_qa", []):
        if not qa or not qa.get("question") or not qa.get("answer"):
            continue

        answerable_now = bool(qa.get("answerable_at_question_time", False))
        if include_wait and not answerable_now:
            samples.append(make_streaming_wait_sample(data, qa, livecc_format=livecc_format))

        samples.append(make_streaming_answer_sample(data, qa, livecc_format=livecc_format))

    return samples


def collect_inputs(args) -> list[str]:
    paths: list[str] = []
    if args.input:
        paths.extend(args.input)
    if args.input_glob:
        for pattern in args.input_glob:
            paths.extend(glob.glob(pattern))
    # stable deterministic order, remove duplicates
    return sorted(set(paths))


def main():
    parser = argparse.ArgumentParser(description="Convert dual QA JSON annotations to training JSONL.")
    parser.add_argument("--input", nargs="*", default=[], help="One or more dual QA JSON files.")
    parser.add_argument("--input-glob", nargs="*", default=[], help="Glob pattern(s) for dual QA JSON files.")
    parser.add_argument("--out", required=True, help="Output JSONL path.")
    parser.add_argument("--overwrite", action="store_true", help="Overwrite output file if it exists.")
    parser.add_argument("--no-include-wait", action="store_true", help="Do not emit streaming_wait samples.")
    parser.add_argument("--livecc-format", action="store_true", help="Output only raw conversation lists per line.")
    args = parser.parse_args()

    inputs = collect_inputs(args)
    if not inputs:
        raise SystemExit("No input files found. Use --input or --input-glob.")

    out_path = Path(args.out)
    if out_path.exists() and not args.overwrite:
        raise SystemExit(f"Output exists: {out_path}. Use --overwrite.")

    out_path.parent.mkdir(parents=True, exist_ok=True)

    total = 0
    with out_path.open("w", encoding="utf-8") as f:
        for p in inputs:
            samples = convert_dual_qa_file(
                p,
                include_wait=not args.no_include_wait,
                livecc_format=args.livecc_format,
            )
            for sample in samples:
                f.write(json.dumps(sample, ensure_ascii=False) + "\n")
                total += 1
            print(f"[dual-qa-to-jsonl] {p}: {len(samples)} samples")

    print(f"[dual-qa-to-jsonl] wrote {total} samples -> {out_path}")


if __name__ == "__main__":
    main()