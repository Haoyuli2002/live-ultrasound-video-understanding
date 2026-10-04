"""Build three matched video/ASR-mask views per complete narration sentence."""
from __future__ import annotations

import argparse
import json
import math
import os
import re
import tempfile
from pathlib import Path

CONDITIONS = ("before_with_asr", "through_with_asr", "before_mask_asr")
END = set(".!?。？！")
ABBREVIATIONS = {"dr", "mr", "mrs", "ms", "prof", "st", "vs", "etc"}


def sentences(segments: list[dict], min_words: int = 3,
              max_words: int = 80, unpunctuated_fallback: str = "none") -> list[dict]:
    """Split punctuated sentences; optionally use ASR segments if none exist.

    Interpolation is only a timing estimate; sample audits remain necessary.
    """
    if unpunctuated_fallback not in {"none", "segment"}:
        raise ValueError(f"Unknown unpunctuated fallback: {unpunctuated_fallback}")
    characters: list[tuple[str, float, float]] = []
    for segment in segments:
        text = re.sub(r"\s+", " ", str(segment.get("text", ""))).strip()
        if not text or re.fullmatch(r"\[(music|noise|applause|inaudible)\]", text, re.I):
            continue
        start, end = float(segment["start"]), float(segment["end"])
        if end <= start:
            continue
        if characters:
            characters.append((" ", start, start))
        for i, char in enumerate(text):
            characters.append((char, start + (end-start)*i/len(text),
                               start + (end-start)*(i+1)/len(text)))
    full = "".join(char for char, _, _ in characters)
    results: list[dict] = []
    beginning = 0
    for i, char in enumerate(full):
        if char not in END:
            continue
        if char == ".":
            word = re.search(r"([A-Za-z]+)$", full[beginning:i])
            if word and word.group(1).lower() in ABBREVIATIONS:
                continue
            if 0 < i < len(full)-1 and full[i-1].isdigit() and full[i+1].isdigit():
                continue
        text = full[beginning:i+1].strip()
        if min_words <= len(text.split()) <= max_words:
            first = beginning
            while first <= i and full[first].isspace():
                first += 1
            if first <= i:
                results.append({"text": text, "start": characters[first][1],
                                "end": characters[i][2],
                                "unit_type": "punctuation_sentence"})
        beginning = i + 1
    if results or unpunctuated_fallback == "none":
        return results  # Unfinished transcript tail is deliberately omitted.
    # Whisper segments are time-aligned utterances, not guaranteed sentences.
    fallback = []
    for segment in segments:
        text = re.sub(r"\s+", " ", str(segment.get("text", ""))).strip()
        if not text or re.fullmatch(r"\[(music|noise|applause|inaudible)\]", text, re.I):
            continue
        start, end = float(segment["start"]), float(segment["end"])
        if end <= start or not min_words <= len(text.split()) <= max_words:
            continue
        fallback.append({"text": text, "start": start, "end": end,
                         "unit_type": "asr_segment_fallback"})
    return fallback


def build_rows(transcript: dict, *, min_words: int = 3,
               max_words: int = 80,
               unpunctuated_fallback: str = "none") -> list[dict]:
    if "sentence_units" in transcript:
        units = []
        previous_end = -1.0
        for unit in transcript["sentence_units"]:
            text = str(unit["text"]).strip()
            start, end = float(unit["start"]), float(unit["end"])
            if start < 0 or end <= start or start < previous_end:
                raise ValueError("Precomputed sentence units are not chronological")
            previous_end = end
            if not min_words <= len(text.split()) <= max_words:
                continue
            units.append({"text": text, "start": start, "end": end,
                          "unit_type": str(unit.get("unit_type") or "llm_punctuation_sentence")})
    else:
        units = sentences(transcript.get("segments", []), min_words, max_words,
                          unpunctuated_fallback)
    rows = []
    for i in range(1, len(units)):
        target = units[i]
        start, end = target["start"], target["end"]
        if start <= 0 or end <= start:
            continue
        history = " ".join(unit["text"] for unit in units[:i]
                           if unit["end"] <= start)
        if not history:
            continue
        for condition in CONDITIONS:
            video_end = end if condition == "through_with_asr" else start
            rows.append({
                "video_id": transcript["video_id"],
                "sentence_id": i,
                "condition": condition,
                "video_window": [0.0, round(video_end, 3)],
                "sentence_window": [round(start, 3), round(end, 3)],
                "historical_asr": "" if condition == "before_mask_asr" else history,
                "historical_asr_masked": condition == "before_mask_asr",
                "target": target["text"],
                "target_unit_type": target["unit_type"],
            })
    return rows


def frame_timestamps(window: list[float], max_frames: int = 120, *,
                     sampling: str = "uniform", recent_seconds: float = 120.0,
                     recent_fraction: float = 0.8) -> list[float]:
    """Choose causal frame times, with optional recent-dense/older-sparse coverage."""
    start, end = map(float, window)
    if start < 0 or end <= start or max_frames <= 0:
        raise ValueError("Invalid causal video window or frame budget")
    if sampling not in {"uniform", "recent_sparse"}:
        raise ValueError(f"Unknown frame sampling strategy: {sampling}")
    if recent_seconds <= 0 or not 0 < recent_fraction < 1:
        raise ValueError("Expected positive recent_seconds and 0 < recent_fraction < 1")
    number = min(max_frames, math.ceil(end-start))

    def uniform_times(left: float, right: float, count: int) -> list[float]:
        return [left + (i+0.5)*(right-left)/count for i in range(count)] if count else []

    if sampling == "uniform" or end-start <= recent_seconds or number < 2:
        return uniform_times(start, end, number)
    recent_start = end - recent_seconds
    recent_count = min(math.ceil(recent_seconds), max(1, round(number*recent_fraction)))
    old_count = min(math.ceil(recent_start-start), number - recent_count)
    recent_count = number - old_count
    if recent_count > math.ceil(recent_seconds):
        recent_count = math.ceil(recent_seconds)
        old_count = number - recent_count
    if old_count == 0:
        return uniform_times(start, end, number)
    return (uniform_times(start, recent_start, old_count)
            + uniform_times(recent_start, end, recent_count))


def sample_frames(video_path: str | Path, window: list[float],
                  max_frames: int = 120, size: int = 224, *,
                  sampling: str = "uniform", recent_seconds: float = 120.0):
    """Decode a causal frame budget; optionally favor the latest two minutes."""
    import cv2
    from PIL import Image

    start, end = map(float, window)
    frame_timestamps(window, max_frames, sampling=sampling,
                     recent_seconds=recent_seconds)
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise FileNotFoundError(f"Cannot open video: {video_path}")
    try:
        fps = cap.get(cv2.CAP_PROP_FPS)
        count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        if fps <= 0 or count <= 0:
            raise ValueError(f"Invalid video metadata: {video_path}")
        end = min(end, count / fps)
        if end <= start:
            raise ValueError("Window starts after video ends")
        timestamps = frame_timestamps([start, end], max_frames,
                                      sampling=sampling,
                                      recent_seconds=recent_seconds)
        result = []
        for timestamp in timestamps:
            frame_index = min(count-1, int(math.floor(timestamp*fps)))
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
            ok, frame = cap.read()
            if not ok:
                raise RuntimeError(f"Could not decode frame {frame_index} from {video_path}")
            image = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            image.thumbnail((size, size))
            canvas = Image.new("RGB", (size, size))
            canvas.paste(image, ((size-image.width)//2, (size-image.height)//2))
            result.append(canvas)
        return result
    finally:
        cap.release()


def build_dataset(transcripts_dir: Path, output: Path, *,
                  video_map: Path | None = None, min_words: int = 3,
                  max_words: int = 80,
                  unpunctuated_fallback: str = "none") -> dict:
    if min_words < 1 or max_words < min_words:
        raise ValueError("Expected 1 <= min_words <= max_words")
    if video_map is None:
        files = sorted(transcripts_dir.glob("*.json"))
    else:
        selected = json.loads(video_map.read_text(encoding="utf-8"))
        if not isinstance(selected, dict) or not selected:
            raise ValueError("--video-map must be a nonempty {video_id: video_path} object")
        files = [transcripts_dir / f"{video_id}.json" for video_id in sorted(selected)]
        missing = [path.name for path in files if not path.is_file()]
        if missing:
            raise FileNotFoundError(
                f"{len(missing)} selected transcripts are missing from {transcripts_dir}: {missing[:5]}")
    if not files:
        raise ValueError(f"No ASR transcript JSON files found in {transcripts_dir}")

    output.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = None
    sample_count = 0
    videos_with_samples = 0
    empty_videos = []
    fallback_videos = []
    fallback_sample_count = 0
    llm_punctuated_videos = []
    try:
        with tempfile.NamedTemporaryFile(mode="w", encoding="utf-8", dir=output.parent,
                                         prefix=f".{output.name}.", suffix=".tmp",
                                         delete=False) as stream:
            temporary_path = Path(stream.name)
            for path in files:
                transcript = json.loads(path.read_text(encoding="utf-8"))
                if not isinstance(transcript, dict):
                    raise ValueError(f"Expected transcript object: {path}")
                if transcript.get("video_id") and str(transcript["video_id"]) != path.stem:
                    raise ValueError(f"Transcript video_id differs from filename: {path}")
                transcript["video_id"] = path.stem
                rows = build_rows(transcript, min_words=min_words,
                                  max_words=max_words,
                                  unpunctuated_fallback=unpunctuated_fallback)
                if rows:
                    videos_with_samples += 1
                    if rows[0]["target_unit_type"] == "asr_segment_fallback":
                        fallback_videos.append(path.stem)
                        fallback_sample_count += len(rows)
                    elif rows[0]["target_unit_type"] == "llm_punctuation_sentence":
                        llm_punctuated_videos.append(path.stem)
                else:
                    empty_videos.append(path.stem)
                for row in rows:
                    stream.write(json.dumps(row, ensure_ascii=False) + "\n")
                sample_count += len(rows)
        if not sample_count:
            raise ValueError("Selected transcripts produced no complete Stage 1 sentences")
        os.replace(temporary_path, output)
    finally:
        if temporary_path is not None:
            temporary_path.unlink(missing_ok=True)
    return {"selected_videos": len(files), "videos_with_samples": videos_with_samples,
            "videos_without_samples": empty_videos, "samples": sample_count,
            "paired_units": sample_count // len(CONDITIONS),
            "paired_sentences": (sample_count - fallback_sample_count) // len(CONDITIONS),
            "fallback_videos": fallback_videos,
            "fallback_samples": fallback_sample_count,
            "llm_punctuated_videos": llm_punctuated_videos,
            "output": str(output)}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--transcripts", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--video-map", type=Path,
                        help="Use only video IDs in this selected {video_id: video_path} map")
    parser.add_argument("--min-words", type=int, default=3)
    parser.add_argument("--max-words", type=int, default=80)
    parser.add_argument("--unpunctuated-fallback", choices=("none", "segment"),
                        default="none", help="Use timed ASR utterances only when no complete punctuated sentence exists")
    args = parser.parse_args()
    summary = build_dataset(Path(args.transcripts), Path(args.output),
                            video_map=args.video_map, min_words=args.min_words,
                            max_words=args.max_words,
                            unpunctuated_fallback=args.unpunctuated_fallback)
    print(json.dumps(summary, ensure_ascii=False))


if __name__ == "__main__":
    main()
