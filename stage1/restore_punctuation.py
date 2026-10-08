"""Restore ASR sentence boundaries without letting an LLM rewrite ASR words.

The teacher receives numbered words and returns only sentence-end indices and
punctuation marks. Original transcripts remain unchanged in the source folder.
Word times are estimated within ASR segments; they are not forced alignment.

The default teacher is ``Qwen/Qwen3.5-27B`` served through a local vLLM
OpenAI-compatible endpoint (``--base-url http://localhost:8000/v1``,
``--api-key-env VLLM_API_KEY``); override ``--model`` to use another endpoint.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import re
import tempfile
from pathlib import Path

from .data import sentences

MODEL = "Qwen/Qwen3.5-27B"
SYSTEM = (
    "You mark sentence endings in an automatic speech transcript. The words "
    "may contain ultrasound and medical terms. Do not rewrite, correct, add, "
    "remove, or reorder words. Do not infer facts. Return only a JSON object "
    "with a boundaries array; each item has integer after and mark, where "
    "mark is one of '.', '?', '!'. Choose only natural complete-sentence "
    "endings, not commas or arbitrary ASR segment boundaries."
)
NOISE = re.compile(r"\[(music|noise|applause|inaudible)\]", re.I)


def timed_words(segments: list[dict]) -> list[dict]:
    """Preserve ASR tokens and estimate word times inside each ASR segment."""
    words = []
    for segment in segments:
        raw = str(segment.get("text") or "")
        if not raw.strip() or NOISE.fullmatch(raw.strip()):
            continue
        start, end = float(segment["start"]), float(segment["end"])
        if not math.isfinite(start) or not math.isfinite(end) or end <= start:
            continue
        for match in re.finditer(r"\S+", raw):
            words.append({
                "text": match.group(),
                "start": start + (end-start)*match.start()/len(raw),
                "end": start + (end-start)*match.end()/len(raw),
            })
    return words


def parse_boundary_json(raw: str) -> list[dict]:
    """Accept a JSON object, including one preceded by model thinking text."""
    decoder = json.JSONDecoder()
    parsed = []
    for match in re.finditer(r"\{", raw or ""):
        try:
            value, _ = decoder.raw_decode(raw[match.start():])
        except json.JSONDecodeError:
            continue
        if isinstance(value, dict) and isinstance(value.get("boundaries"), list):
            parsed.append(value["boundaries"])
    if not parsed:
        raise ValueError(f"Teacher did not return a boundaries JSON object: {raw[:300]!r}")
    return parsed[-1]


class PunctuationTeacher:
    def __init__(self, model: str, *, api_key: str, base_url: str | None,
                 max_tokens: int = 600):
        from openai import OpenAI
        self.client = OpenAI(api_key=api_key, base_url=base_url) if base_url else OpenAI(api_key=api_key)
        self.model = model
        self.max_tokens = max_tokens

    def boundaries(self, words: list[dict], core_start: int, core_end: int,
                   context_words: int) -> list[dict]:
        left = max(0, core_start-context_words)
        right = min(len(words), core_end+context_words)
        numbered = "\n".join(f"{i}\t{words[i]['text']}" for i in range(left, right))
        prompt = (
            f"Mark sentence endings only after word indices in [{core_start}, {core_end}). "
            f"The surrounding words [{left}, {right}) are context. Do not force "
            "an ending at the edge of this request. Return exactly "
            f'{{"boundaries":[{{"after":{core_start},"mark":"."}}]}} with actual indices. '
            "An empty boundaries array is allowed.\n\n"
            f"index\tASR word\n{numbered}"
        )
        request = dict(
            model=self.model,
            messages=[{"role": "system", "content": SYSTEM},
                      {"role": "user", "content": prompt}],
            temperature=0, max_tokens=self.max_tokens,
        )
        if "qwen" in self.model.lower():
            request["extra_body"] = {"chat_template_kwargs": {"enable_thinking": False}}
        response = self.client.chat.completions.create(**request)
        return parse_boundary_json(response.choices[0].message.content or "")


def collect_boundaries(words: list[dict], teacher, *, chunk_words: int = 160,
                       context_words: int = 25) -> list[dict]:
    if chunk_words <= 0 or context_words < 0:
        raise ValueError("Expected chunk_words > 0 and context_words >= 0")
    boundaries = []
    seen = set()
    for core_start in range(0, len(words), chunk_words):
        core_end = min(len(words), core_start+chunk_words)
        returned = teacher.boundaries(words, core_start, core_end, context_words)
        if not isinstance(returned, list):
            raise ValueError("Teacher boundaries must be a list")
        for item in returned:
            if not isinstance(item, dict):
                raise ValueError("Teacher boundary must be an object")
            index, mark = item.get("after"), item.get("mark")
            if (type(index) is not int or not core_start <= index < core_end
                    or mark not in {".", "?", "!"} or index in seen):
                raise ValueError(f"Invalid or duplicate boundary in [{core_start},{core_end}): {item}")
            seen.add(index)
            boundaries.append({"after": index, "mark": mark})
    return sorted(boundaries, key=lambda item: item["after"])


def sentence_units(words: list[dict], boundaries: list[dict]) -> list[dict]:
    """Rebuild sentences from untouched ASR words and LLM boundary indices."""
    units = []
    first = 0
    for boundary in boundaries:
        last = boundary["after"]
        if not first <= last < len(words):
            raise ValueError(f"Boundary outside remaining words: {boundary}")
        tokens = [word["text"] for word in words[first:last+1]]
        mark = boundary["mark"]
        if tokens[-1].endswith((".", "?", "!")):
            tokens[-1] = tokens[-1][:-1] + mark
        else:
            tokens[-1] = tokens[-1].rstrip(",;:") + mark
        units.append({
            "text": " ".join(tokens),
            "start": round(words[first]["start"], 3),
            "end": round(words[last]["end"], 3),
            "unit_type": "llm_punctuation_sentence",
            "word_start_index": first,
            "word_end_index": last+1,
            "alignment": "segment_character_interpolation",
        })
        first = last+1
    return units  # An unfinished tail after the last boundary is omitted.


def restore_transcript(transcript: dict, teacher, *, chunk_words: int = 160,
                       context_words: int = 25) -> tuple[dict, dict]:
    words = timed_words(transcript.get("segments") or [])
    if not words:
        raise ValueError("No usable timed ASR words")
    boundaries = collect_boundaries(words, teacher, chunk_words=chunk_words,
                                    context_words=context_words)
    units = sentence_units(words, boundaries)
    if len(units) < 2:
        raise ValueError("Restoration produced fewer than two sentence units")
    result = dict(transcript)
    result["sentence_units"] = units
    result["punctuation_restoration"] = {
        "method": "llm_word_boundary_indices",
        "source_word_count": len(words),
        "sentence_count": len(units),
        "unfinished_tail_words": len(words) - (boundaries[-1]["after"]+1),
        "alignment": "segment_character_interpolation",
    }
    return result, result["punctuation_restoration"]


def write_json_atomic(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(mode="w", encoding="utf-8", dir=path.parent,
                                         prefix=f".{path.name}.", suffix=".tmp",
                                         delete=False) as stream:
            temporary = Path(stream.name)
            json.dump(value, stream, ensure_ascii=False, indent=2)
            stream.write("\n")
        temporary.replace(path)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--transcripts", type=Path, required=True)
    parser.add_argument("--video-map", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--audit-output", type=Path, required=True)
    parser.add_argument("--model", default=MODEL)
    parser.add_argument("--base-url", default="http://localhost:8000/v1")
    parser.add_argument("--api-key-env", default="VLLM_API_KEY")
    parser.add_argument("--chunk-words", type=int, default=160)
    parser.add_argument("--context-words", type=int, default=25)
    parser.add_argument("--max-tokens", type=int, default=600)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    if args.output_dir.resolve() == args.transcripts.resolve():
        raise ValueError("Output directory must differ from source transcripts")
    selected = json.loads(args.video_map.read_text(encoding="utf-8"))
    if not isinstance(selected, dict) or not selected:
        raise ValueError("--video-map must contain selected video IDs")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    args.audit_output.parent.mkdir(parents=True, exist_ok=True)
    teacher = None
    for video_id in sorted(selected):
        if not video_id or Path(video_id).name != video_id:
            raise ValueError(f"Invalid video ID in map: {video_id!r}")
        source = args.transcripts / f"{video_id}.json"
        output = args.output_dir / source.name
        if output.exists():
            if args.resume:
                continue
            raise FileExistsError(f"Output already exists: {output}; use --resume")
        transcript = json.loads(source.read_text(encoding="utf-8"))
        if transcript.get("video_id") and str(transcript["video_id"]) != video_id:
            raise ValueError(f"Transcript ID differs from filename: {source}")
        transcript["video_id"] = video_id
        if sentences(transcript.get("segments") or []):
            result = transcript
            audit = {"video_id": video_id, "status": "copied_punctuated"}
        else:
            if teacher is None:
                api_key = os.environ.get(args.api_key_env)
                if not api_key:
                    raise ValueError(f"Set {args.api_key_env} before restoring punctuation")
                teacher = PunctuationTeacher(args.model, api_key=api_key,
                                             base_url=args.base_url,
                                             max_tokens=args.max_tokens)
            result, details = restore_transcript(
                transcript, teacher, chunk_words=args.chunk_words,
                context_words=args.context_words)
            result["punctuation_restoration"]["model"] = args.model
            audit = {"video_id": video_id, "status": "restored", **details,
                     "model": args.model}
        write_json_atomic(output, result)
        with args.audit_output.open("a", encoding="utf-8") as stream:
            stream.write(json.dumps(audit, ensure_ascii=False) + "\n")
        print(json.dumps(audit, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
