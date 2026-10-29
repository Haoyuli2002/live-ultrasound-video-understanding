"""Build audited Stage 1 triplets from word-timed ASR into a new dataset directory."""
from __future__ import annotations

import argparse
from collections import Counter
from dataclasses import asdict, dataclass
import hashlib
import json
import math
import os
from pathlib import Path
import re
import shutil
import tempfile

from .alignment import lexical, segment_words
from .data import ABBREVIATIONS, CONDITIONS, frame_timestamps
from .restore_punctuation import collect_boundaries, sentence_units, PunctuationTeacher

VERSION = "stage1_dataset_v2.1"


@dataclass(frozen=True)
class Policy:
    min_words: int = 10
    min_seconds: float = 5.0
    max_words_per_second: float = 8.0
    min_mean_word_probability: float = 0.5
    min_segment_logprob: float = -1.0
    max_segment_compression: float = 2.4
    max_speech_gap: float = 8.0
    allow_estimated_timestamps: bool = False

    def __post_init__(self):
        if (self.min_words < 1 or self.min_seconds <= 0
                or self.max_words_per_second <= 0 or self.max_speech_gap <= 0
                or not 0 <= self.min_mean_word_probability <= 1
                or self.max_segment_compression <= 0
                or any(not math.isfinite(float(v)) for v in asdict(self).values())):
            raise ValueError("Invalid dataset quality thresholds")


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def repetition_reason(text: str) -> str | None:
    words = lexical(text)
    # Quarantine suspected loops for review; do not rewrite or deduplicate speech.
    for width in range(1, min(9, len(words)//3+1)):
        required = 6 if width == 1 else 3
        for start in range(len(words)-required*width+1):
            phrase = words[start:start+width]
            if (words[start:start+required*width] == phrase*required
                    and required*width >= len(words)*0.5):
                return "suspected_repetition_loop"
    return None


def punctuation_boundaries(words: list[dict]) -> list[dict]:
    result = []
    for index, word in enumerate(words):
        text = word["text"].rstrip('\"\'”’)]}')
        if not text.endswith((".", "?", "!")):
            continue
        last = text[:-1].rsplit(maxsplit=1)[-1].lower() if text[:-1].strip() else ""
        if text[-1] == "." and (last in ABBREVIATIONS | {"e.g", "i.e"}
                                  or re.fullmatch(r"(?:[a-z]\.)*[a-z]", last)):
            continue
        result.append({"after": index, "mark": text[-1]})
    return result


def _segment_reasons(segment: dict, policy: Policy) -> list[str]:
    reasons = []
    for key, limit, lower in (("avg_logprob", policy.min_segment_logprob, True),
                              ("compression_ratio", policy.max_segment_compression, False)):
        value = segment.get(key)
        if value is not None:
            number = float(value)
            if not math.isfinite(number):
                reasons.append("invalid_asr_confidence")
            elif (number < limit if lower else number > limit):
                reasons.append("low_asr_logprob" if lower else "high_asr_compression")
    if re.search(r"<\|[^>]+\|>|</?think>|(?:^|\n)\s*(?:assistant|user|system)\s*(?:\n|$)",
                 str(segment.get("text", "")), re.I):
        reasons.append("chat_markup_in_asr")
    return reasons


def prepare_video(transcript: dict, policy: Policy, *, teacher=None) -> tuple[list[dict], list[dict], dict]:
    """Return accepted units, review/quarantine records, and per-video coverage."""
    segments = transcript.get("segments")
    if not isinstance(segments, list) or not segments:
        raise ValueError("missing_asr_segments")
    duration = transcript.get("duration_sec")
    if duration is not None and (not math.isfinite(float(duration)) or float(duration) <= 0):
        raise ValueError("invalid_video_duration")
    groups, current, review = [], [], []
    previous_end = 0.0
    def flush():
        nonlocal current
        if current:
            groups.append(current)
            current = []
    for index, segment in enumerate(segments):
        reasons = []
        if not isinstance(segment, dict):
            raise ValueError("ASR segment must be an object")
        try:
            reasons = _segment_reasons(segment, policy)
            words = segment_words(segment, index, allow_estimated=policy.allow_estimated_timestamps)
            if words:
                if words[0]["start"] < previous_end - 0.001:
                    reasons.append("overlapping_asr_segments")
                if duration is not None and words[-1]["end"] > float(duration)+0.001:
                    reasons.append("asr_exceeds_video_duration")
        except (ValueError, KeyError, TypeError) as error:
            reasons.append(str(error))
            words = []
        if reasons:
            flush()  # Never join a sentence across discarded speech.
            review.append({"scope": "segment", "segment_index": index,
                           "start": segment.get("start"), "end": segment.get("end"),
                           "text": segment.get("text"), "reasons": reasons})
            # Even rejected speech occupies time: later overlap remains visible.
            try:
                end = float(segment.get("end", previous_end))
                if math.isfinite(end): previous_end = max(previous_end, end)
            except (TypeError, ValueError):
                pass
            continue
        if not words:
            continue
        if current and words[0]["start"]-previous_end > policy.max_speech_gap:
            flush()
        current.extend(words)
        previous_end = words[-1]["end"]
    flush()
    candidates = []
    for group_index, words in enumerate(groups):
        boundaries = (collect_boundaries(words, teacher) if teacher is not None
                      else punctuation_boundaries(words))
        units = sentence_units(words, boundaries)
        tail = boundaries[-1]["after"]+1 if boundaries else 0
        if tail < len(words):
            review.append({"scope": "unfinished_tail", "group": group_index,
                           "text": " ".join(w["text"] for w in words[tail:]),
                           "start": words[tail]["start"], "end": words[-1]["end"],
                           "reasons": ["no_complete_sentence_boundary"]})
        for unit in units:
            selected = words[unit["word_start_index"]:unit["word_end_index"]]
            unit["unit_type"] = ("llm_punctuation_sentence" if teacher is not None
                                 else "word_punctuation_sentence")
            unit["segment_indices"] = sorted({w["segment_index"] for w in selected})
            unit["group_index"] = group_index
            unit["candidate_id"] = len(candidates)
            # Do not round a 4.999x-second window up to the 5-second target gate.
            unit["start"], unit["end"] = selected[0]["start"], selected[-1]["end"]
            unit["zero_duration_word_count"] = sum(w.get("zero_duration", False) for w in selected)
            probabilities = [float(w["probability"]) for w in selected if w["probability"] is not None]
            unit["mean_word_probability"] = (sum(probabilities)/len(probabilities)
                                              if probabilities else None)
            seconds = unit["end"]-unit["start"]
            count = len(unit["text"].split())
            reasons = []
            # Short, sound speech may remain in history, even when it is not a target.
            if count < 1: reasons.append("sentence_word_count")
            if seconds < 0.5: reasons.append("sentence_duration")
            if seconds <= 0 or count/seconds > policy.max_words_per_second: reasons.append("speech_rate")
            if probabilities and unit["mean_word_probability"] < policy.min_mean_word_probability:
                reasons.append("low_mean_word_probability")
            repeat = repetition_reason(unit["text"])
            if repeat: reasons.append(repeat)
            candidates.append({**unit, "reasons": reasons})
    # Three or more consecutive identical sentences are suspect, even with plausible durations.
    first = 0
    while first < len(candidates):
        last = first+1
        while (last < len(candidates) and candidates[last]["group_index"] == candidates[first]["group_index"]
               and lexical(candidates[last]["text"]) == lexical(candidates[first]["text"])):
            last += 1
        if last-first >= 3:
            for item in candidates[first:last]: item["reasons"].append("consecutive_duplicate_sentences")
        first = last
    kept = []
    for unit in candidates:
        if unit["reasons"]:
            review.append({"scope": "sentence", **unit})
        else:
            target_reasons = []
            if len(unit["text"].split()) < policy.min_words:
                target_reasons.append("target_too_few_words")
            if unit["end"]-unit["start"] < policy.min_seconds:
                target_reasons.append("target_too_short")
            unit["target_eligible"] = not target_reasons
            if target_reasons:
                review.append({"scope":"target", "action":"exclude_target_keep_history",
                               **unit, "reasons":target_reasons})
            kept.append({k: v for k, v in unit.items() if k != "reasons"})
    return kept, review, {"segments": len(segments), "candidate_sentences": len(candidates),
                          "accepted_sentences": len(kept),
                          "eligible_target_sentences": sum(u['target_eligible'] for u in kept),
                          "quarantine_reasons": dict(Counter(reason for item in review
                                                           for reason in item["reasons"]))}


def paired_rows(video_id: str, units: list[dict], source_hash: str):
    history = []
    for unit in units:
        start, end = unit["start"], unit["end"]
        if history and start > 0 and unit.get('target_eligible', True):
            for condition in CONDITIONS:
                masked = condition == "before_mask_asr"
                yield {"video_id": video_id, "sentence_id": unit["candidate_id"],
                       "condition": condition, "video_window": [0.0, end if condition == "through_with_asr" else start],
                       "sentence_window": [start, end], "historical_asr": "" if masked else " ".join(history),
                       "historical_asr_masked": masked, "target": unit["text"],
                       "target_unit_type": unit["unit_type"], "alignment": unit["alignment"],
                       "dataset_version": VERSION, "source_sha256": source_hash,
                       "source_segment_indices": unit["segment_indices"],
                       "mean_word_probability": unit["mean_word_probability"]}
        history.append(unit["text"])


def build(transcripts: Path, video_map: Path, output_dir: Path, policy: Policy, *,
          teacher=None, allow_empty_videos=False) -> dict:
    if output_dir.exists():
        raise FileExistsError(f"Use a new dataset directory: {output_dir}")
    if output_dir.resolve().is_relative_to(transcripts.resolve()):
        raise ValueError("Dataset output must be outside source transcripts")
    selected = json.loads(video_map.read_text())
    if not isinstance(selected, dict) or not selected:
        raise ValueError("Expected nonempty video map")
    if any(not isinstance(k, str) or not k or Path(k).name != k or k in {'.','..'} for k in selected):
        raise ValueError("Unsafe video ID in map")
    output_dir.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=f".{output_dir.name}.", dir=output_dir.parent))
    counts, audits, sources, accepted_map = Counter(), [], {}, {}
    coverage = Counter()
    try:
        with (staging/'samples.jsonl').open('w') as samples, (staging/'quarantine.jsonl').open('w') as quarantine, (staging/'sentences.jsonl').open('w') as sentences:
            for video_id in sorted(selected):
                source = transcripts/f'{video_id}.json'
                try:
                    source_bytes = source.read_bytes()
                    source_hash = hashlib.sha256(source_bytes).hexdigest()
                    sources[video_id] = source_hash
                    transcript = json.loads(source_bytes)
                    if transcript.get('video_id', video_id) != video_id: raise ValueError('transcript_id_mismatch')
                    units, review, stats = prepare_video(transcript, policy, teacher=teacher)
                    for item in review:
                        quarantine.write(json.dumps({'video_id':video_id, **item}, ensure_ascii=False)+'\n')
                    for item in units:
                        sentences.write(json.dumps({'video_id':video_id, **item}, ensure_ascii=False)+'\n')
                    rows = list(paired_rows(video_id, units, source_hash))
                    for row in rows:
                        samples.write(json.dumps(row, ensure_ascii=False)+'\n')
                        if row['condition']=='through_with_asr':
                            s,e = row['sentence_window']
                            for strategy, recent in [('uniform',120),('recent_sparse',15)]:
                                times = frame_timestamps(row['video_window'],30,sampling=strategy,recent_seconds=recent)
                                coverage[strategy+'_no_target_frames'] += not any(s<=t<e for t in times)
                    if rows:
                        path = Path(selected[video_id])
                        accepted_map[video_id] = str(path if path.is_absolute() else (video_map.parent/path).resolve())
                    counts['samples'] += len(rows)
                    counts['targets'] += len(rows)//3
                    counts.update(stats['quarantine_reasons'])
                    audits.append({'video_id':video_id, 'status':'accepted' if rows else 'empty',
                                   'source_sha256':source_hash, 'rows':len(rows), **stats})
                except Exception as error:
                    # Any teacher/source failure rejects publication of training samples.
                    audits.append({'video_id':video_id, 'status':'failed', 'error':str(error)})
        failed = [a['video_id'] for a in audits if a['status']=='failed']
        empty = [a['video_id'] for a in audits if a['status']=='empty']
        ready = bool(counts['samples']) and not failed and (allow_empty_videos or not empty)
        report = {'version':VERSION,'status':'ready' if ready else 'rejected',
                  'selected_videos':len(selected),'accepted_videos':len(accepted_map),
                  'failed_videos':failed,'empty_videos':empty, 'counts':dict(counts),
                  'frame_coverage_at_budget_30':dict(coverage),
                  'policy':asdict(policy),'boundary_method':'llm_indices' if teacher else 'asr_punctuation',
                  'teacher_model':getattr(teacher,'model',None),
                  'limitations':['Word times are ASR estimates, not manually verified alignment.',
                                 'Quarantined repetitions need audio review; they are not rewritten.',
                                 'Frame coverage is a schedule check; video decoding was not performed.']}
        if not ready:
            (staging/'samples.jsonl').unlink()  # A rejected dataset cannot be trained accidentally.
        (staging/'audit.jsonl').write_text(''.join(json.dumps(a,ensure_ascii=False)+'\n' for a in audits))
        (staging/'video_map.json').write_text(json.dumps(accepted_map,indent=2))
        (staging/'report.json').write_text(json.dumps(report,ensure_ascii=False,indent=2))
        manifest = {'version':VERSION,'sources':sources,'video_map_sha256':digest(video_map),
                    'policy':asdict(policy),'boundary_method':report['boundary_method'],
                    'teacher_model':report['teacher_model'],
                    'implementation_sha256':{p.name:digest(p) for p in [Path(__file__),
                        Path(__file__).with_name('alignment.py'), Path(__file__).with_name('restore_punctuation.py'),
                        Path(__file__).with_name('data.py')]},
                    'artifacts':{p.name:digest(p) for p in staging.iterdir() if p.is_file()}}
        (staging/'manifest.json').write_text(json.dumps(manifest,indent=2))
        os.rename(staging,output_dir)
        return report
    finally:
        if staging.exists(): shutil.rmtree(staging)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--transcripts',type=Path,required=True)
    parser.add_argument('--video-map',type=Path,required=True)
    parser.add_argument('--output-dir',type=Path,required=True)
    parser.add_argument('--allow-empty-videos',action='store_true')
    parser.add_argument('--boundary-mode',choices=['punctuation','llm'],default='punctuation')
    parser.add_argument('--model',default='Qwen/Qwen3.5-27B')
    parser.add_argument('--base-url',default='http://localhost:8000/v1')
    parser.add_argument('--api-key-env',default='VLLM_API_KEY')
    parser.add_argument('--allow-estimated-timestamps',action='store_true')
    parser.add_argument('--min-seconds',type=float,default=5.0)
    parser.add_argument('--min-words',type=int,default=10)
    parser.add_argument('--max-words-per-second',type=float,default=8)
    parser.add_argument('--min-mean-word-probability',type=float,default=.5)
    args=parser.parse_args()
    teacher=None
    if args.boundary_mode=='llm':
        teacher=PunctuationTeacher(args.model,api_key=os.environ.get(args.api_key_env,'EMPTY'),
                                   base_url=args.base_url)
    policy=Policy(min_words=args.min_words,min_seconds=args.min_seconds,max_words_per_second=args.max_words_per_second,
                  min_mean_word_probability=args.min_mean_word_probability,
                  allow_estimated_timestamps=args.allow_estimated_timestamps)
    report=build(args.transcripts,args.video_map,args.output_dir,policy,
                 teacher=teacher,allow_empty_videos=args.allow_empty_videos)
    print(json.dumps(report,ensure_ascii=False,indent=2))
    if report['status']!='ready': raise SystemExit(1)


if __name__=='__main__':
    main()
