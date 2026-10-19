"""Read-only structural audit of the 193-video Stage 1 merged training JSONL.

This does not judge ASR accuracy, sentence boundaries, or video decodability.
Run it on the SSH machine that holds the actual data before training.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import hashlib
import json
import math
from pathlib import Path

from .data import CONDITIONS
from .seven_video_pilot import PILOT_IDS


def _video_map(path: Path) -> dict[str, str]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict) or not value or any(
        not isinstance(key, str) or not key or not isinstance(video, str) or not video
        for key, video in value.items()
    ):
        raise ValueError(f"{path}: expected nonempty {{video_id: video_path}} map")
    return value


def _rows(path: Path) -> list[dict]:
    rows = []
    with path.open(encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, 1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as error:
                raise ValueError(f"{path}:{line_number}: invalid JSON: {error}") from error
            if not isinstance(row, dict):
                raise ValueError(f"{path}:{line_number}: expected JSON object")
            rows.append(row)
    if not rows:
        raise ValueError(f"{path}: no rows")
    return rows


def _ids(rows: list[dict], label: str) -> set[str]:
    result = set()
    for index, row in enumerate(rows, 1):
        video_id = row.get("video_id")
        if not isinstance(video_id, str) or not video_id:
            raise ValueError(f"{label} row {index}: missing video_id")
        result.add(video_id)
    return result


def _same_ids(actual: set[str], expected: set[str], label: str) -> None:
    if actual != expected:
        raise ValueError(
            f"{label}: expected {len(expected)}, found {len(actual)}; "
            f"missing={sorted(expected - actual)[:10]}, extra={sorted(actual - expected)[:10]}"
        )


def _canonical(row: dict) -> str:
    return json.dumps(row, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def _window(value: object, label: str) -> tuple[float, float]:
    if not isinstance(value, list) or len(value) != 2 or any(
        isinstance(x, bool) or not isinstance(x, (int, float)) for x in value
    ):
        raise ValueError(f"{label}: expected two numeric endpoints")
    endpoints = float(value[0]), float(value[1])
    if not all(math.isfinite(x) for x in endpoints):
        raise ValueError(f"{label}: endpoints must be finite")
    return endpoints


def _validate_groups(rows: list[dict]) -> int:
    groups: dict[tuple[str, int], dict[str, dict]] = defaultdict(dict)
    for index, row in enumerate(rows, 1):
        sentence_id = row.get("sentence_id")
        if isinstance(sentence_id, bool) or not isinstance(sentence_id, int) or sentence_id < 0:
            raise ValueError(f"merged row {index}: invalid sentence_id")
        key = (row["video_id"], sentence_id)
        condition = row.get("condition")
        if not isinstance(condition, str) or condition not in CONDITIONS or condition in groups[key]:
            raise ValueError(f"{key}: unknown or duplicate condition {condition!r}")
        groups[key][condition] = row

    for key, group in groups.items():
        if set(group) != set(CONDITIONS):
            raise ValueError(f"{key}: incomplete conditions: {sorted(group)}")
        before = group["before_with_asr"]
        through = group["through_with_asr"]
        masked = group["before_mask_asr"]
        target = before.get("target")
        if not isinstance(target, str) or not target.strip() or any(
            row.get("target") != target for row in group.values()
        ):
            raise ValueError(f"{key}: missing or mismatched target")
        if len({_canonical(row.get("sentence_window")) for row in group.values()}) != 1:
            raise ValueError(f"{key}: mismatched sentence_window")
        if len({_canonical(row.get("target_unit_type")) for row in group.values()}) != 1:
            raise ValueError(f"{key}: mismatched target_unit_type")
        start, end = _window(before.get("sentence_window"), f"{key} sentence_window")
        if not 0 < start < end:
            raise ValueError(f"{key}: invalid sentence_window [{start}, {end}]")
        for condition, row, expected_end in (
            ("before_with_asr", before, start),
            ("through_with_asr", through, end),
            ("before_mask_asr", masked, start),
        ):
            left, right = _window(row.get("video_window"), f"{key} {condition} video_window")
            if left != 0 or abs(right - expected_end) > 0.001:
                raise ValueError(f"{key} {condition}: expected video_window [0, {expected_end}], got [{left}, {right}]")
        history = before.get("historical_asr")
        if not isinstance(history, str) or not history.strip() or through.get("historical_asr") != history:
            raise ValueError(f"{key}: before/through historical_asr missing or different")
        if masked.get("historical_asr") != "":
            raise ValueError(f"{key}: masked historical_asr must be empty")
        if (before.get("historical_asr_masked") is not False
                or through.get("historical_asr_masked") is not False
                or masked.get("historical_asr_masked") is not True):
            raise ValueError(f"{key}: incorrect historical_asr_masked flags")
    return len(groups)


def verify(*, full_map: Path, baseline: Path, pilot_samples: Path, merged: Path,
           pilot_map: Path | None = None, eval_map: Path | None = None,
           expected_videos: int = 193, check_video_files: bool = False) -> dict:
    full = _video_map(full_map)
    full_ids = set(full)
    pilot_ids = set(PILOT_IDS)
    if len(full_ids) != expected_videos:
        raise ValueError(f"full map: expected {expected_videos} videos, found {len(full_ids)}")
    if not pilot_ids <= full_ids:
        raise ValueError(f"full map missing pilot IDs: {sorted(pilot_ids - full_ids)}")
    if pilot_map is not None:
        selected = _video_map(pilot_map)
        _same_ids(set(selected), pilot_ids, "pilot map")
        if any(selected[video_id] != full[video_id] for video_id in pilot_ids):
            raise ValueError("pilot map paths differ from full map")
    if eval_map is not None:
        evaluation = _video_map(eval_map)
        overlap = full_ids & set(evaluation)
        if overlap:
            raise ValueError(f"train/eval video ID overlap: {sorted(overlap)[:10]}")
        shared_paths = set(full.values()) & set(evaluation.values())
        if shared_paths:
            raise ValueError(f"train/eval video path overlap: {sorted(shared_paths)[:10]}")
    if check_video_files:
        missing = [video_id for video_id, video in full.items() if not Path(video).is_file()]
        if missing:
            raise ValueError(f"{len(missing)} mapped video files missing: {missing[:10]}")

    base_rows, pilot_rows, merged_rows = _rows(baseline), _rows(pilot_samples), _rows(merged)
    _same_ids(_ids(base_rows, "baseline"), full_ids - pilot_ids, "baseline coverage")
    _same_ids(_ids(pilot_rows, "pilot"), pilot_ids, "pilot coverage")
    _same_ids(_ids(merged_rows, "merged"), full_ids, "merged coverage")

    expected = Counter(map(_canonical, base_rows + pilot_rows))
    actual = Counter(map(_canonical, merged_rows))
    if actual != expected:
        raise ValueError(
            f"merged content differs from baseline + pilot: "
            f"missing_rows={sum((expected - actual).values())}, "
            f"extra_rows={sum((actual - expected).values())}"
        )
    groups = _validate_groups(merged_rows)
    sha256 = hashlib.sha256()
    with merged.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            sha256.update(chunk)
    return {
        "status": "PASS", "videos": len(full_ids), "baseline_videos": len(full_ids - pilot_ids),
        "pilot_videos": len(pilot_ids), "rows": len(merged_rows), "sentence_groups": groups,
        "eval_disjoint_checked": eval_map is not None,
        "video_files_checked": check_video_files,
        "merged_sha256": sha256.hexdigest(),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--full-map", required=True, type=Path)
    parser.add_argument("--baseline", required=True, type=Path)
    parser.add_argument("--pilot-samples", required=True, type=Path)
    parser.add_argument("--merged", required=True, type=Path)
    parser.add_argument("--pilot-map", type=Path)
    parser.add_argument("--eval-map", type=Path)
    parser.add_argument("--expected-videos", type=int, default=193)
    parser.add_argument("--check-video-files", action="store_true")
    args = parser.parse_args()
    try:
        result = verify(**vars(args))
    except (OSError, ValueError) as error:
        parser.exit(1, f"FAIL: {error}\n")
    print(json.dumps(result, ensure_ascii=False, sort_keys=True))


if __name__ == "__main__":
    main()
