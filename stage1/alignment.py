"""Validate word timing against unchanged ASR text; never invent alignment."""
from __future__ import annotations

import math
import re

LEXICAL = re.compile(r"\w+(?:['’-]\w+)*", re.UNICODE)


def lexical(text: str) -> list[str]:
    return LEXICAL.findall(text.lower())


def segment_words(segment: dict, index: int = 0, *,
                  allow_estimated: bool = False) -> list[dict]:
    raw = str(segment.get("text") or "")
    if not raw.strip():
        return []
    start, end = float(segment["start"]), float(segment["end"])
    if not math.isfinite(start) or not math.isfinite(end) or start < 0 or end <= start:
        raise ValueError("invalid_segment_time")
    native = segment.get("words")
    if native:
        if not isinstance(native, list):
            raise ValueError("invalid_word_list")
        if any(not isinstance(w, dict) for w in native):
            raise ValueError("invalid_timed_word")
        raw_tokens = [str(w.get("word", "")) for w in native]
        # Whisper may split compounds into ' four', '-chamber'. Preserve their
        # spacing; some external word JSONs instead omit all leading spaces.
        spaced_tokens = any(w[:1].isspace() for w in raw_tokens)
        reconstructed = ("" if spaced_tokens else " ").join(raw_tokens)
        if lexical(reconstructed) != lexical(raw):
            raise ValueError("word_text_mismatch_realign_required")
        result, previous = [], start
        for position, word in enumerate(native):
            left, right = float(word["start"]), float(word["end"])
            if (not math.isfinite(left) or not math.isfinite(right)
                    or left < start - 0.001 or right > end + 0.001
                    or left < previous - 0.001 or right < left):
                raise ValueError("invalid_word_time")
            text = str(word["word"]).strip()
            if not text:
                raise ValueError("empty_timed_word")
            probability = word.get("probability")
            if probability is not None and (not math.isfinite(float(probability))
                                            or not 0 <= float(probability) <= 1):
                raise ValueError("invalid_word_probability")
            result.append({"text": text, "start": left, "end": right,
                           "probability": probability, "segment_index": index,
                           "separator_before": (" " if position == 0 or not spaced_tokens
                                                or raw_tokens[position][:1].isspace() else ""),
                           "zero_duration": left == right,
                           "alignment": "whisper_word_timestamps"})
            previous = right
        return result
    if not allow_estimated:
        raise ValueError("missing_word_timestamps")
    return [{"text": m.group(),
             "start": start + (end-start)*m.start()/len(raw),
             "end": start + (end-start)*m.end()/len(raw),
             "probability": None, "segment_index": index,
             "alignment": "segment_character_interpolation"}
            for m in re.finditer(r"\S+", raw)]
