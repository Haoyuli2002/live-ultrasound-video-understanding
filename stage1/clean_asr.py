"""Clean and lightly polish timed Whisper segments with Qwen3.5.

The LLM sees only ASR text, never audio or video. Its terminology edits are
therefore hypotheses to audit, not verified transcriptions.
"""
from __future__ import annotations

import argparse
import copy
import json
import os
import re
import tempfile
from pathlib import Path

MODEL = "Qwen/Qwen3.5-27B"
SYSTEM = """You are cleaning Whisper ASR from ultrasound teaching videos. Return JSON only.
For each requested segment, lightly polish punctuation, capitalization, and spacing.
Correct an ultrasound/medical term only when the surrounding ASR makes the intended
term clear. Preserve the speaker's actual wording, word order, repetitions,
uncertainty, negation, and all nonmedical content. Do not summarize, paraphrase,
complete unfinished speech, invent findings, or infer facts from unseen video/audio.
When a term is ambiguous, leave it as transcribed. Keep every segment separate;
punctuation may be placed within a segment. Return exactly:
{"segments":[{"index":0,"clean_text":"...","corrections":[{"from":"...","to":"...","reason":"..."}]}]}
The index must match each requested segment. List every changed word or phrase
in corrections; punctuation and capitalization need no correction entry.
"""


def parse_response(raw: str) -> list[dict]:
    decoder = json.JSONDecoder()
    candidates = []
    for match in re.finditer(r"\{", raw or ""):
        try:
            value, _ = decoder.raw_decode(raw[match.start():])
        except json.JSONDecodeError:
            continue
        if isinstance(value, dict) and isinstance(value.get("segments"), list):
            candidates.append(value["segments"])
    if not candidates:
        raise ValueError(f"LLM did not return segments JSON: {(raw or '')[:300]!r}")
    return candidates[-1]


def lexical_words(text: str) -> list[str]:
    return re.findall(r"[\w]+(?:['’-][\w]+)*", text.lower(), flags=re.UNICODE)


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


def apply_declared_edits(words: list[str], edits: list[dict], index: int) -> list[str]:
    result = words[:]
    for edit in edits:
        old, new = lexical_words(edit["from"]), lexical_words(edit["to"])
        if not old or not new:
            raise ValueError(f"Empty lexical correction in segment {index}")
        position = next((i for i in range(len(result)-len(old)+1)
                         if result[i:i+len(old)] == old), None)
        if position is None:
            raise ValueError(f"Correction source is absent in segment {index}: {edit}")
        result[position:position+len(old)] = new
    return result


def validate_batch(source: list[dict], returned: list[dict], indices: list[int]) -> list[dict]:
    if len(returned) != len(indices):
        raise ValueError(f"Expected {len(indices)} segments; received {len(returned)}")
    checked = []
    for raw, item, index in zip(source, returned, indices):
        if not isinstance(item, dict) or type(item.get("index")) is not int or item["index"] != index:
            raise ValueError(f"Missing or reordered segment {index}")
        clean = item.get("clean_text")
        edits = item.get("corrections")
        if not isinstance(clean, str) or not clean.strip() or not isinstance(edits, list):
            raise ValueError(f"Invalid clean text or corrections for segment {index}")
        clean = re.sub(r"\s+", " ", clean).strip()
        before, after = lexical_words(str(raw.get("text") or "")), lexical_words(clean)
        if before:
            # Reject broad rewrites; punctuation and case changes are unrestricted.
            from difflib import SequenceMatcher
            similarity = SequenceMatcher(None, before, after, autojunk=False).ratio()
            if ((len(before) >= 6 and similarity < 0.65)
                    or abs(len(after) - len(before)) > max(3, len(before) // 4)):
                raise ValueError(f"Excessive rewrite in segment {index}: similarity={similarity:.2f}")
        if before != after and not edits:
            raise ValueError(f"Undeclared lexical change in segment {index}")
        normalized_edits = []
        for edit in edits:
            if not isinstance(edit, dict):
                raise ValueError(f"Invalid correction in segment {index}")
            old, new, reason = edit.get("from"), edit.get("to"), edit.get("reason")
            if not all(isinstance(value, str) and value.strip() for value in (old, new, reason)):
                raise ValueError(f"Incomplete correction in segment {index}")
            if old.lower() not in str(raw.get("text") or "").lower() or new.lower() not in clean.lower():
                raise ValueError(f"Correction is not grounded in segment {index}: {edit}")
            normalized_edits.append({"from": old, "to": new, "reason": reason})
        if before == after and normalized_edits:
            raise ValueError(f"Spurious lexical correction in segment {index}")
        if apply_declared_edits(before, normalized_edits, index) != after:
            raise ValueError(f"Undeclared lexical change in segment {index}")
        checked.append({"index": index, "clean_text": clean, "corrections": normalized_edits})
    return checked


class ASRCleaner:
    def __init__(self, model: str = MODEL, *, api_key: str, base_url: str,
                 max_tokens: int = 4096):
        from openai import OpenAI
        self.client = OpenAI(api_key=api_key, base_url=base_url)
        self.model = model
        self.max_tokens = max_tokens

    def clean(self, segments: list[dict], indices: list[int], context: list[dict]) -> list[dict]:
        prompt = {
            "context_only": context,
            "segments_to_clean": [
                {"index": index, "start": segment.get("start"),
                 "end": segment.get("end"), "text": segment.get("text", "")}
                for index, segment in zip(indices, segments)
            ],
        }
        response = self.client.chat.completions.create(
            model=self.model,
            messages=[{"role": "system", "content": SYSTEM},
                      {"role": "user", "content": json.dumps(prompt, ensure_ascii=False)}],
            temperature=0,
            max_tokens=self.max_tokens,
            extra_body={"chat_template_kwargs": {"enable_thinking": False}},
        )
        return parse_response(response.choices[0].message.content or "")


def clean_transcript(transcript: dict, cleaner, *, batch_segments: int = 8,
                     context_segments: int = 2) -> tuple[dict, dict]:
    if batch_segments < 1 or context_segments < 0:
        raise ValueError("Expected batch_segments >= 1 and context_segments >= 0")
    source = transcript.get("segments")
    if not isinstance(source, list) or not source:
        raise ValueError("Transcript has no ASR segments")
    cleaned = copy.deepcopy(source)
    corrections = []
    changed_segments = 0
    for begin in range(0, len(source), batch_segments):
        end = min(len(source), begin + batch_segments)
        indices = list(range(begin, end))
        context = [
            {"index": i, "text": source[i].get("text", "")}
            for i in list(range(max(0, begin-context_segments), begin))
            + list(range(end, min(len(source), end+context_segments)))
        ]
        returned = cleaner.clean(source[begin:end], indices, context)
        checked = validate_batch(source[begin:end], returned, indices)
        for item in checked:
            index = item["index"]
            if item["clean_text"] != str(source[index].get("text") or "").strip():
                changed_segments += 1
            cleaned[index]["text"] = item["clean_text"]
            for correction in item["corrections"]:
                corrections.append({"segment_index": index, **correction})
    result = copy.deepcopy(transcript)
    result.pop("sentence_units", None)  # Existing units refer to the original text.
    result["raw_segments"] = copy.deepcopy(source)
    result["segments"] = cleaned
    if "full_text" in result:
        result["raw_full_text"] = result["full_text"]
    result["full_text"] = " ".join(str(segment["text"]).strip() for segment in cleaned)
    result["asr_cleaning"] = {
        "method": "qwen35_segment_text_cleanup",
        "model": getattr(cleaner, "model", "unknown"),
        "segment_count": len(source),
        "changed_segments": changed_segments,
        "term_corrections": corrections,
        "limitations": "text-only ASR correction; not checked against audio or video",
    }
    audit = {"video_id": result.get("video_id"), "segment_count": len(source),
             "changed_segments": changed_segments,
             "term_correction_count": len(corrections),
             "term_corrections": corrections,
             "model": result["asr_cleaning"]["model"]}
    return result, audit


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--transcripts", type=Path, required=True)
    parser.add_argument("--video-map", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--audit-output", type=Path, required=True)
    parser.add_argument("--model", default=MODEL)
    parser.add_argument("--base-url", default="http://localhost:8000/v1")
    parser.add_argument("--api-key-env", default="VLLM_API_KEY")
    parser.add_argument("--batch-segments", type=int, default=8)
    parser.add_argument("--context-segments", type=int, default=2)
    parser.add_argument("--max-tokens", type=int, default=4096)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    if args.output_dir.resolve() == args.transcripts.resolve():
        raise ValueError("Output directory must differ from source transcripts")
    selected = json.loads(args.video_map.read_text(encoding="utf-8"))
    if not isinstance(selected, dict) or not selected:
        raise ValueError("--video-map must contain selected video IDs")
    api_key = os.environ.get(args.api_key_env)
    if not api_key:
        raise ValueError(f"Set {args.api_key_env} for the local vLLM endpoint")
    cleaner = ASRCleaner(args.model, api_key=api_key,
                         base_url=args.base_url, max_tokens=args.max_tokens)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    args.audit_output.parent.mkdir(parents=True, exist_ok=True)
    for video_id in sorted(selected):
        if not video_id or Path(video_id).name != video_id:
            raise ValueError(f"Invalid video ID: {video_id!r}")
        source = args.transcripts / f"{video_id}.json"
        destination = args.output_dir / source.name
        if destination.exists():
            if args.resume:
                continue
            raise FileExistsError(f"Output exists: {destination}; use --resume")
        transcript = json.loads(source.read_text(encoding="utf-8"))
        if transcript.get("video_id") and str(transcript["video_id"]) != video_id:
            raise ValueError(f"Transcript ID differs from filename: {source}")
        transcript["video_id"] = video_id
        result, audit = clean_transcript(transcript, cleaner,
                                         batch_segments=args.batch_segments,
                                         context_segments=args.context_segments)
        write_json_atomic(destination, result)
        with args.audit_output.open("a", encoding="utf-8") as stream:
            stream.write(json.dumps(audit, ensure_ascii=False) + "\n")
        print(json.dumps({key: value for key, value in audit.items()
                          if key != "term_corrections"}, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
