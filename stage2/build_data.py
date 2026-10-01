"""Convert independent visual teacher labels into chronological Stage 2 blocks.

Accepted inputs:
  * one block row with local_sub_summaries, block_summary_target and
    global_summary_target;
  * three event types per block: local_10s, block_60s, global_prefix.

This builder never fabricates an independent 60-second summary by concatenating
10-second labels, or a global summary from the current block alone.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

from .data import load_blocks


def _window(value, context: str) -> list[float]:
    if not isinstance(value, (list, tuple)) or len(value) != 2:
        raise ValueError(f"{context}: expected a [start, end] window")
    start, end = map(float, value)
    if start < 0 or end <= start:
        raise ValueError(f"{context}: invalid window {value}")
    return [start, end]


def _target(value, context: str) -> str:
    result = str(value or "").strip()
    if not result:
        raise ValueError(f"{context}: missing independent teacher summary")
    return result


def _validate_optional_window(row: dict, key: str, expected: list[float], context: str):
    if key in row:
        observed = _window(row[key], context + "/" + key)
        if any(abs(a-b) > 0.01 for a, b in zip(observed, expected)):
            raise ValueError(f"{context}: {key}={observed} must equal {expected}")


def _normalize_block(row: dict, source: str) -> dict:
    video_id = str(row.get("video_id") or "").strip()
    if not video_id:
        raise ValueError(f"{source}: missing video_id")
    block_window = _window(row.get("block_window"), source)
    start, end = block_window
    full = abs(end-start-60) <= 0.01
    _validate_optional_window(row, "block_summary_window", block_window, source)
    _validate_optional_window(row, "global_window", [0.0, end], source)
    labels = row.get("local_sub_summaries")
    if not isinstance(labels, list):
        if row.get("local_summary_target"):
            raise ValueError(
                f"{source}: legacy 60-second local_summary_target is insufficient; "
                "generate six independent 10-second labels and one independent "
                "block_summary_target")
        raise ValueError(f"{source}: local_sub_summaries must be a list")
    normalized = []
    for index, label in enumerate(labels):
        context = f"{source}/local[{index}]"
        normalized.append({
            "window": _window(label.get("window"), context),
            "local_summary_target": _target(label.get("local_summary_target"), context),
        })
    result = {"video_id": video_id, "block_window": block_window,
              "local_sub_summaries": normalized}
    if full:
        result["block_summary_target"] = _target(row.get("block_summary_target"), source + "/block")
        result["global_summary_target"] = _target(row.get("global_summary_target"), source + "/global")
    else:
        # An incomplete tail has no long update or cumulative target.
        if row.get("block_summary_target") or row.get("global_summary_target"):
            raise ValueError(f"{source}: incomplete block cannot have block/global targets")
        result["block_summary_target"] = ""
        result["global_summary_target"] = ""
    if row.get("teacher"):
        result["teacher"] = row["teacher"]
    return result


def _event_type(row: dict) -> str | None:
    value = row.get("label_type", row.get("summary_type", row.get("annotation_type")))
    aliases = {"local": "local_10s", "local_10s": "local_10s",
               "block": "block_60s", "block_60s": "block_60s",
               "global": "global_prefix", "global_prefix": "global_prefix"}
    return aliases.get(value)


def build_rows(records: list[dict]) -> list[dict]:
    if not records:
        raise ValueError("No teacher labels supplied")
    block_rows = ["local_sub_summaries" in row or
                  ("local_summary_target" in row and _event_type(row) is None)
                  for row in records]
    if any(block_rows) and not all(block_rows):
        raise ValueError("Do not mix block and event teacher-label schemas")
    if all(block_rows):
        return [_normalize_block(row, f"record {index+1}")
                for index, row in enumerate(records)]

    groups: dict[tuple[str, float, float], dict] = {}
    for index, row in enumerate(records, 1):
        context = f"event {index}"
        video_id = str(row.get("video_id") or "").strip()
        window = _window(row.get("block_window"), context)
        if not video_id:
            raise ValueError(f"{context}: missing video_id")
        key = (video_id, *window)
        group = groups.setdefault(key, {"video_id": video_id,
                                        "block_window": window,
                                        "local_sub_summaries": []})
        if row.get("teacher"):
            if group.get("teacher") and group["teacher"] != row["teacher"]:
                raise ValueError(f"{context}: mixed teacher models within one block")
            group["teacher"] = row["teacher"]
        kind = _event_type(row)
        if kind is None:
            raise ValueError(f"{context}: unknown label_type; expected local_10s, block_60s, or global_prefix")
        target = _target(row.get("target"), context)
        label_window = _window(row.get("window"), context)
        if kind == "local_10s":
            group["local_sub_summaries"].append({"window": label_window,
                                                  "local_summary_target": target})
        elif kind == "block_60s":
            if "block_summary_target" in group:
                raise ValueError(f"{context}: duplicate block summary")
            if any(abs(a-b) > 0.01 for a, b in zip(label_window, window)):
                raise ValueError(f"{context}: block label must cover its block")
            group["block_summary_target"] = target
        else:
            if "global_summary_target" in group:
                raise ValueError(f"{context}: duplicate global summary")
            if abs(label_window[0]) > 0.01 or abs(label_window[1]-window[1]) > 0.01:
                raise ValueError(f"{context}: global label must cover [0, block_end]")
            group["global_summary_target"] = target
    result = []
    for group in groups.values():
        group["local_sub_summaries"].sort(key=lambda item: item["window"][0])
        result.append(_normalize_block(group, str(group["video_id"])))
    return result


def build_files(inputs: list[str | Path], output: str | Path) -> int:
    records = []
    for path in inputs:
        with Path(path).open(encoding="utf-8") as stream:
            for line_no, line in enumerate(stream, 1):
                if line.strip():
                    try:
                        records.append(json.loads(line))
                    except json.JSONDecodeError as exc:
                        raise ValueError(f"{path}:{line_no}: invalid JSON") from exc
    rows = build_rows(records)
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_name(output.name + ".tmp")
    try:
        with temporary.open("w", encoding="utf-8") as stream:
            for row in sorted(rows, key=lambda item: (item["video_id"], item["block_window"][0])):
                stream.write(json.dumps(row, ensure_ascii=False) + "\n")
        load_blocks(temporary)  # Full chronology and 10-second label validation.
        os.replace(temporary, output)
    finally:
        temporary.unlink(missing_ok=True)
    return len(rows)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--teacher-jsonl", nargs="+", required=True,
                        help="One or more visual-only teacher-label JSONL files")
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    count = build_files(args.teacher_jsonl, args.output)
    print(f"[stage2] wrote {count} chronological blocks to {args.output}")


if __name__ == "__main__":
    main()
