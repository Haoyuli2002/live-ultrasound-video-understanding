"""Read-only Stage 1 input audits and optional human-review CSV export.

Subcommands: cleaning (seven-video ASR), eval (split isolation),
timing (sentence-boundary review packet), frames (120-frame budget),
smoke (actual training artifacts/config).
No model, GPU, audio decoding, or video decoding is required.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import csv
import json
import random
from pathlib import Path

from .clean_asr import validate_batch
from .data import build_rows, frame_timestamps
from .seven_video_pilot import PILOT_IDS
from .verify_merged_193 import _canonical, _ids, _rows, _same_ids, _validate_groups, _video_map


def _json(path: Path) -> dict:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"{path}: expected JSON object")
    return value


def _write_csv(path: Path, fieldnames: list[str], records: list[dict]) -> None:
    if path.exists():
        raise FileExistsError(f"Refusing to overwrite review file: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", newline="", encoding="utf-8-sig") as stream:
        writer = csv.DictWriter(stream, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(records)


def _audit_rows(path: Path) -> dict[str, dict]:
    latest = {}
    for row in _rows(path):
        video_id = row.get("video_id")
        if not isinstance(video_id, str) or not video_id:
            raise ValueError(f"{path}: audit row missing video_id")
        latest[video_id] = row
    return latest


def _row_stats(rows: list[dict]) -> dict:
    word_counts = [len(str(row["target"]).split()) for row in rows]
    return {
        "videos": len(_ids(rows, "samples")),
        "rows": len(rows),
        "sentence_groups": len(rows) // 3,
        "unit_types": dict(sorted(Counter(str(row.get("target_unit_type")) for row in rows).items())),
        "mean_target_words": round(sum(word_counts) / len(word_counts), 2),
    }


def audit_cleaning(*, pilot_map: Path, source_dir: Path, cleaned_dir: Path,
                   audit: Path, pilot_samples: Path, baseline: Path,
                   review_csv: Path | None = None) -> dict:
    selected = _video_map(pilot_map)
    _same_ids(set(selected), set(PILOT_IDS), "seven-video pilot map")
    audits = _audit_rows(audit)
    pilot_rows = _rows(pilot_samples)
    baseline_rows = _rows(baseline)
    _same_ids(_ids(pilot_rows, "pilot samples"), set(PILOT_IDS), "pilot sample coverage")
    if set(PILOT_IDS) & _ids(baseline_rows, "baseline samples"):
        raise ValueError("Baseline samples overlap seven pilot videos")
    _validate_groups(pilot_rows)
    _validate_groups(baseline_rows)

    corrections_for_review = []
    checked_segments = changed_segments = 0
    for video_id in sorted(selected):
        source = _json(source_dir / f"{video_id}.json")
        cleaned = _json(cleaned_dir / f"{video_id}.json")
        if cleaned.get("video_id") != video_id:
            raise ValueError(f"{video_id}: cleaned transcript video_id mismatch")
        raw = source.get("segments")
        updated = cleaned.get("segments")
        if not isinstance(raw, list) or not raw or not isinstance(updated, list) or len(raw) != len(updated):
            raise ValueError(f"{video_id}: missing or mismatched ASR segments")
        if cleaned.get("raw_segments") != raw:
            raise ValueError(f"{video_id}: raw_segments do not match source transcript")
        if "full_text" in source and cleaned.get("raw_full_text") != source["full_text"]:
            raise ValueError(f"{video_id}: raw_full_text does not match source")
        if cleaned.get("full_text") != " ".join(str(seg["text"]).strip() for seg in updated):
            raise ValueError(f"{video_id}: cleaned full_text does not match segments")
        meta = cleaned.get("asr_cleaning")
        if not isinstance(meta, dict) or meta.get("segment_count") != len(raw):
            raise ValueError(f"{video_id}: invalid asr_cleaning metadata")
        corrections = meta.get("term_corrections")
        if not isinstance(corrections, list):
            raise ValueError(f"{video_id}: missing term_corrections")
        grouped = defaultdict(list)
        for correction in corrections:
            index = correction.get("segment_index")
            if type(index) is not int or not 0 <= index < len(raw):
                raise ValueError(f"{video_id}: invalid correction segment_index")
            grouped[index].append({key: correction.get(key) for key in ("from", "to", "reason")})
            corrections_for_review.append({
                "video_id": video_id, "segment_index": index,
                "start": raw[index].get("start"), "end": raw[index].get("end"),
                "raw_text": raw[index].get("text"), "clean_text": updated[index].get("text"),
                "from": correction.get("from"), "to": correction.get("to"),
                "reason": correction.get("reason"), "human_verdict": "",
            })
        for index, (before, after) in enumerate(zip(raw, updated)):
            if not isinstance(before, dict) or not isinstance(after, dict):
                raise ValueError(f"{video_id}: non-object ASR segment {index}")
            if {k: v for k, v in before.items() if k != "text"} != {
                    k: v for k, v in after.items() if k != "text"}:
                raise ValueError(f"{video_id}: timestamps/metadata changed in segment {index}")
            validate_batch([before], [{"index": index, "clean_text": after.get("text"),
                                       "corrections": grouped[index]}], [index])
            changed_segments += str(before.get("text", "")).strip() != str(after.get("text", "")).strip()
            checked_segments += 1
        if meta.get("changed_segments") != sum(
                str(before.get("text", "")).strip() != str(after.get("text", "")).strip()
                for before, after in zip(raw, updated)):
            raise ValueError(f"{video_id}: changed_segments metadata mismatch")
        latest = audits.get(video_id)
        if latest is None or latest.get("status") != "cleaned":
            raise ValueError(f"{video_id}: latest audit status is not cleaned")
        if (latest.get("term_corrections") != corrections
                or latest.get("term_correction_count") != len(corrections)
                or latest.get("segment_count") != len(raw)
                or latest.get("changed_segments") != meta["changed_segments"]):
            raise ValueError(f"{video_id}: audit differs from cleaned transcript")
        rebuilt = build_rows(cleaned)
        actual = [row for row in pilot_rows if row["video_id"] == video_id]
        if Counter(map(_canonical, rebuilt)) != Counter(map(_canonical, actual)):
            raise ValueError(f"{video_id}: pilot samples differ from rebuilt cleaned transcript")
    if review_csv is not None:
        _write_csv(review_csv,
                   ["video_id", "segment_index", "start", "end", "raw_text", "clean_text",
                    "from", "to", "reason", "human_verdict"], corrections_for_review)
    return {
        "status": "STRUCTURE_PASS_CONTENT_REVIEW_REQUIRED",
        "pilot_videos": len(selected), "checked_segments": checked_segments,
        "changed_segments": changed_segments, "term_corrections": len(corrections_for_review),
        "baseline": _row_stats(baseline_rows), "pilot": _row_stats(pilot_rows),
        "review_csv": str(review_csv) if review_csv else None,
        "note": "Correction content and 186+7 preprocessing equivalence require human review",
    }


def audit_eval(*, train_map: Path, eval_map: Path, train_samples: Path,
               eval_samples: Path | None = None, check_video_files: bool = False) -> dict:
    train, evaluation = _video_map(train_map), _video_map(eval_map)
    overlap = set(train) & set(evaluation)
    if overlap:
        raise ValueError(f"Train/eval video ID overlap: {sorted(overlap)[:10]}")
    def resolved(mapping_path: Path, value: str) -> Path:
        video = Path(value)
        return (video if video.is_absolute() else mapping_path.parent / video).resolve()

    train_paths = {str(resolved(train_map, value)) for value in train.values()}
    eval_paths = {str(resolved(eval_map, value)) for value in evaluation.values()}
    if train_paths & eval_paths:
        raise ValueError(f"Train/eval video path overlap: {sorted(train_paths & eval_paths)[:10]}")
    if check_video_files:
        missing = [vid for mapping_path, mapping in ((train_map, train), (eval_map, evaluation))
                   for vid, value in mapping.items() if not resolved(mapping_path, value).is_file()]
        if missing:
            raise ValueError(f"Mapped video files missing: {missing[:10]}")
    training = _rows(train_samples)
    _same_ids(_ids(training, "train samples"), set(train), "train sample coverage")
    _validate_groups(training)
    result = {"status": "SPLIT_PASS", "train_videos": len(train),
              "eval_map_videos": len(evaluation), "eval_samples_checked": eval_samples is not None,
              "video_files_checked": check_video_files}
    if eval_samples is not None:
        examples = _rows(eval_samples)
        actual = _ids(examples, "eval samples")
        extra = actual - set(evaluation)
        if extra:
            raise ValueError(f"Eval sample IDs outside eval map: {sorted(extra)[:10]}")
        _validate_groups(examples)
        result.update(eval_sample_videos=len(actual), eval_rows=len(examples),
                      eval_map_videos_without_samples=sorted(set(evaluation) - actual))
    return result


def _groups(rows: list[dict]) -> dict[str, list[dict]]:
    by_video = defaultdict(list)
    for row in rows:
        if row.get("condition") == "before_with_asr":
            by_video[row["video_id"]].append(row)
    return by_video


def audit_timing(*, samples: Path, video_map: Path, transcript_dir: Path,
                 pilot_transcript_dir: Path, review_csv: Path,
                 max_videos: int = 20, per_video: int = 2, seed: int = 42) -> dict:
    if max_videos < 1 or per_video < 1:
        raise ValueError("max_videos and per_video must be positive")
    rows = _rows(samples)
    paths = _video_map(video_map)
    by_video = _groups(rows)
    _same_ids(set(by_video), set(paths), "samples vs video map")
    rng = random.Random(seed)
    pilot = sorted(set(by_video) & set(PILOT_IDS))
    other = sorted(set(by_video) - set(PILOT_IDS))
    chosen = pilot[:max_videos]
    rng.shuffle(other)
    chosen.extend(other[:max(0, max_videos - len(chosen))])
    review = []
    for video_id in chosen:
        source_dir = pilot_transcript_dir if video_id in PILOT_IDS else transcript_dir
        transcript = _json(source_dir / f"{video_id}.json")
        segments = transcript.get("segments", [])
        selected = by_video[video_id][:]
        rng.shuffle(selected)
        for row in selected[:per_video]:
            start, end = row["sentence_window"]
            nearby = [str(seg.get("text", "")) for seg in segments
                      if float(seg.get("end", -1)) >= start - 2
                      and float(seg.get("start", 1e12)) <= end + 2]
            review.append({
                "video_id": video_id, "video_path": paths[video_id],
                "sentence_id": row["sentence_id"], "start": start, "end": end,
                "duration": round(end - start, 3), "target": row["target"],
                "historical_asr": row["historical_asr"],
                "nearby_asr_segments": " | ".join(nearby),
                "boundary_ok": "", "medical_terms_ok": "", "notes": "",
            })
    _write_csv(review_csv,
               ["video_id", "video_path", "sentence_id", "start", "end", "duration",
                "target", "historical_asr", "nearby_asr_segments",
                "boundary_ok", "medical_terms_ok", "notes"], review)
    return {"status": "REVIEW_PACKET_CREATED", "videos": len(chosen),
            "sentences": len(review), "pilot_videos": len(set(chosen) & set(PILOT_IDS)),
            "review_csv": str(review_csv),
            "note": "Open the matching video/audio near start/end; this CSV cannot judge boundary accuracy"}


def audit_frames(*, samples: Path, frame_budget: int = 120,
                 sampling: str = "uniform", recent_seconds: float = 120.0) -> dict:
    rows = _rows(samples)
    counts, long_intervals = Counter(), 0
    max_gap = 0.0
    for index, row in enumerate(rows, 1):
        start, end = row["video_window"]
        times = frame_timestamps([start, end], frame_budget, sampling=sampling,
                                 recent_seconds=recent_seconds)
        if not times or len(times) > frame_budget or times != sorted(times) or any(
                not start <= instant < end for instant in times):
            raise ValueError(f"row {index}: invalid frame timestamps")
        counts[len(times)] += 1
        long_intervals += end - start > frame_budget
        max_gap = max(max_gap, *(b - a for a, b in zip(times, times[1:])))
    return {"status": "FRAME_SCHEDULE_PASS", "rows": len(rows),
            "frame_budget": frame_budget, "sampling": sampling,
            "longer_than_budget_seconds": long_intervals,
            "max_sampled_frames": max(counts), "min_sampled_frames": min(counts),
            "max_between_frame_gap_seconds": round(max_gap, 3),
            "note": "Timestamp planning only; actual video decoding is not checked"}


def audit_smoke(*, output_dir: Path, expected_steps: int = 2,
                expected_input: Path | None = None) -> dict:
    """Inspect what the completed training run actually recorded, not sweep defaults."""
    config = _json(output_dir / "training_config.json")
    split = _json(output_dir / "video_split.json")
    validation = output_dir / "validation_samples.jsonl"
    if not validation.is_file():
        raise ValueError(f"Missing validation samples: {validation}")
    source = Path(str(config.get("train_jsonl", "")))
    if not source.is_file():
        raise ValueError(f"Recorded train_jsonl is missing: {source}")
    if expected_input is not None and source.resolve() != expected_input.resolve():
        raise ValueError(f"Smoke used {source}, not expected {expected_input}")
    if config.get("max_steps") != expected_steps:
        raise ValueError(f"Smoke max_steps={config.get('max_steps')}, expected {expected_steps}")
    train_ids = set(split.get("train_video_ids", []))
    val_ids = set(split.get("validation_video_ids", []))
    if train_ids & val_ids:
        raise ValueError("video_split.json has train/validation overlap")
    with validation.open(encoding="utf-8") as stream:
        val_rows = [json.loads(line) for line in stream if line.strip()]
    if {row["video_id"] for row in val_rows} != val_ids:
        raise ValueError("validation_samples.jsonl differs from video_split.json")
    checkpoint = output_dir / f"checkpoint-{expected_steps}"
    final = output_dir / "final"
    if not checkpoint.is_dir() or not final.is_dir() or not any(final.iterdir()):
        raise ValueError(f"Missing checkpoint-{expected_steps} or nonempty final/ in {output_dir}")
    state_path = checkpoint / "trainer_state.json"
    if state_path.is_file():
        state = _json(state_path)
        if state.get("global_step") != expected_steps:
            raise ValueError(f"Checkpoint global_step={state.get('global_step')}, expected {expected_steps}")
    return {"status": "SMOKE_ARTIFACTS_PASS", "output_dir": str(output_dir),
            "model_name": config.get("model_name"), "recorded_train_jsonl": str(source),
            "max_steps": config["max_steps"], "active_frame_budget": config.get("active_frame_budget"),
            "train_videos": len(train_ids), "validation_videos": len(val_ids),
            "checkpoint": str(checkpoint), "final": str(final),
            "note": "Artifacts/config checked; inspect job log for CUDA errors and final completion"}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    clean = commands.add_parser("cleaning")
    for flag in ("pilot-map", "source-dir", "cleaned-dir", "audit", "pilot-samples", "baseline"):
        clean.add_argument(f"--{flag}", required=True, type=Path)
    clean.add_argument("--review-csv", type=Path)
    split = commands.add_parser("eval")
    for flag in ("train-map", "eval-map", "train-samples"):
        split.add_argument(f"--{flag}", required=True, type=Path)
    split.add_argument("--eval-samples", type=Path)
    split.add_argument("--check-video-files", action="store_true")
    timing = commands.add_parser("timing")
    for flag in ("samples", "video-map", "transcript-dir", "pilot-transcript-dir", "review-csv"):
        timing.add_argument(f"--{flag}", required=True, type=Path)
    timing.add_argument("--max-videos", type=int, default=20)
    timing.add_argument("--per-video", type=int, default=2)
    timing.add_argument("--seed", type=int, default=42)
    frames = commands.add_parser("frames")
    frames.add_argument("--samples", required=True, type=Path)
    frames.add_argument("--frame-budget", type=int, default=120)
    frames.add_argument("--sampling", choices=("uniform", "recent_sparse"), default="uniform")
    frames.add_argument("--recent-seconds", type=float, default=120.0)
    smoke = commands.add_parser("smoke")
    smoke.add_argument("--output-dir", required=True, type=Path)
    smoke.add_argument("--expected-steps", type=int, default=2)
    smoke.add_argument("--expected-input", type=Path)
    args = parser.parse_args()
    command = args.command
    kwargs = vars(args)
    del kwargs["command"]
    try:
        result = {"cleaning": audit_cleaning, "eval": audit_eval,
                  "timing": audit_timing, "frames": audit_frames,
                  "smoke": audit_smoke}[command](**kwargs)
    except (OSError, ValueError, KeyError, TypeError) as error:
        parser.exit(1, f"FAIL: {error}\n")
    print(json.dumps(result, ensure_ascii=False, sort_keys=True))


if __name__ == "__main__":
    main()
