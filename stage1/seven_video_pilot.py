"""Select the seven punctuation-missing train videos and merge their Stage 1 rows."""
from __future__ import annotations

import argparse
import json
import os
import tempfile
from pathlib import Path

from .data import CONDITIONS

PILOT_IDS = (
    "OKtdZVXtZ0M", "OQJ2RhnuOQQ", "TlckvYhqaFE", "WhNxq98szE8",
    "YYcTdkP7xhs", "d7IwDHAVcGQ", "vmmrjy1bhEc",
)


def read_map(path: Path) -> dict[str, str]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict) or not value:
        raise ValueError(f"Expected nonempty video map: {path}")
    return value


def select_pilot(full_map: Path, output: Path) -> dict:
    selected = read_map(full_map)
    missing = sorted(set(PILOT_IDS) - selected.keys())
    if missing:
        raise ValueError(f"Seven-video IDs missing from pretrain keep-map: {missing}")
    pilot = {video_id: selected[video_id] for video_id in PILOT_IDS}
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(pilot, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return {"pilot_videos": len(pilot), "video_ids": list(pilot), "output": str(output)}


def read_rows(path: Path) -> list[dict]:
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()
            if line.strip()]
    if not rows:
        raise ValueError(f"No Stage 1 rows in {path}")
    return rows


def validate_groups(rows: list[dict]) -> None:
    groups: dict[tuple[str, int], dict[str, dict]] = {}
    for row in rows:
        key = (str(row["video_id"]), int(row["sentence_id"]))
        condition = row["condition"]
        group = groups.setdefault(key, {})
        if condition not in CONDITIONS or condition in group:
            raise ValueError(f"Invalid or duplicate Stage 1 condition: {key}, {condition}")
        group[condition] = row
    for key, group in groups.items():
        if set(group) != set(CONDITIONS):
            raise ValueError(f"Incomplete three-condition group: {key}")
        if len({row["target"] for row in group.values()}) != 1:
            raise ValueError(f"Different targets within Stage 1 group: {key}")


def merge_samples(full_map: Path, pilot_map: Path, baseline: Path,
                  pilot_samples: Path, output: Path) -> dict:
    full_ids = set(read_map(full_map))
    pilot_ids = set(read_map(pilot_map))
    if pilot_ids != set(PILOT_IDS) or not pilot_ids <= full_ids:
        raise ValueError("Pilot map must contain exactly the seven selected train videos")
    baseline_rows, pilot_rows = read_rows(baseline), read_rows(pilot_samples)
    baseline_ids = {str(row["video_id"]) for row in baseline_rows}
    observed_pilot_ids = {str(row["video_id"]) for row in pilot_rows}
    if baseline_ids != full_ids - pilot_ids:
        raise ValueError(f"Baseline coverage mismatch: expected {len(full_ids-pilot_ids)} "
                         f"videos, found {len(baseline_ids)}; missing="
                         f"{sorted((full_ids-pilot_ids)-baseline_ids)[:10]}")
    if observed_pilot_ids != pilot_ids:
        raise ValueError(f"Pilot coverage mismatch: missing={sorted(pilot_ids-observed_pilot_ids)}")
    rows = baseline_rows + pilot_rows
    validate_groups(rows)
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(mode="w", encoding="utf-8", dir=output.parent,
                                         prefix=f".{output.name}.", suffix=".tmp",
                                         delete=False) as stream:
            temporary = Path(stream.name)
            for row in rows:
                stream.write(json.dumps(row, ensure_ascii=False) + "\n")
        os.replace(temporary, output)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)
    return {"videos": len(full_ids), "baseline_videos": len(baseline_ids),
            "cleaned_pilot_videos": len(pilot_ids), "rows": len(rows),
            "pilot_rows": len(pilot_rows), "output": str(output)}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    select = sub.add_parser("select")
    select.add_argument("--full-map", type=Path, required=True)
    select.add_argument("--output", type=Path, required=True)
    merge = sub.add_parser("merge")
    merge.add_argument("--full-map", type=Path, required=True)
    merge.add_argument("--pilot-map", type=Path, required=True)
    merge.add_argument("--baseline", type=Path, required=True)
    merge.add_argument("--pilot-samples", type=Path, required=True)
    merge.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.command == "select":
        summary = select_pilot(args.full_map, args.output)
    else:
        summary = merge_samples(args.full_map, args.pilot_map, args.baseline,
                                args.pilot_samples, args.output)
    print(json.dumps(summary, ensure_ascii=False))


if __name__ == "__main__":
    main()
