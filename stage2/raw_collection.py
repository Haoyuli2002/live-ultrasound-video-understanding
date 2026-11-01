"""Durable raw-window cache and deferred, non-filtering Stage 2 quality reports."""
from __future__ import annotations

import argparse
from collections import Counter
import json
import os
from pathlib import Path
import re

from .summary_schema import PERSPECTIVES, SUMMARY_PLANS

COLLECTION_POLICY = 'save_first_review_later_v1'


def atomic_json(path: Path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + f'.{os.getpid()}.tmp')
    try:
        with temporary.open('w', encoding='utf-8') as stream:
            json.dump(value, stream, ensure_ascii=False, indent=2)
            stream.write('\n'); stream.flush(); os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


class WindowStore:
    """One atomic file per returned response, saved before any text checking.

    A cache hit must match the exact source/config/model and request prompt.
    Complete legacy minute batches remain separate and are never rewritten.
    """
    def __init__(self, root: Path, context: dict):
        self.root, self.context = root, context

    def path(self, kind, start, end):
        return self.root / f'{kind}_{start}_{end}.json'

    def get(self, kind, start, end, prompt):
        path = self.path(kind, start, end)
        if not path.exists(): return None
        record = json.loads(path.read_text())
        if (record['context'] != self.context or record['prompt'] != prompt
                or record['kind'] != kind or record['window'] != [start, end]
                or not isinstance(record['raw_text'], str)):
            raise ValueError(f'Raw cache source/config/request differs: {path}')
        return record

    def put(self, kind, start, end, prompt, raw_text, metadata):
        if not isinstance(raw_text, str):
            raise TypeError('Teacher returned non-text content')
        record = {'context': self.context, 'collection_policy': COLLECTION_POLICY,
                  'kind': kind, 'window': [start, end], 'prompt': prompt,
                  'raw_text': raw_text, 'response_metadata': metadata}
        atomic_json(self.path(kind, start, end), record)
        return record


def text_issues(text: str, kind: str) -> list[str]:
    """Report all presentation issues; no medical accuracy claim or filtering."""
    value = text.strip()
    issues = []
    if not value: issues.append('empty')
    if len(value.split()) < 4: issues.append('too_short')
    if re.search(r'\b(?:' + '|'.join(PERSPECTIVES) + r')\s*:', value, re.I):
        issues.append('possible_perspective_labels')
    if value.startswith(('{', '[', '```', '#')) or re.search(r'(?m)^\s*(?:[-*•]|\d+[.)])\s', value):
        issues.append('structured_output')
    if re.search(r'<\/?think>', value, re.I): issues.append('reasoning_tags')
    if re.search(r'\n\s*\n', value): issues.append('multiple_paragraphs')
    if len(value.split()) > SUMMARY_PLANS[kind]['words']: issues.append('over_preferred_words')
    return issues


def quality_report(events: list[dict], raw_root: Path | None = None) -> dict:
    kinds = {'local_10s':'local', 'block_60s':'block', 'global_prefix':'global'}
    flagged, counts, by_kind, by_video = [], Counter(), Counter(), Counter()
    for row in events:
        kind = kinds[row['label_type']]
        issues = text_issues(row['target'], kind)
        if raw_root is not None:
            a, b = row['window']
            path = raw_root / row['video_id'] / f'{kind}_{a}_{b}.json'
            if path.exists():
                raw = json.loads(path.read_text())
                meta = raw.get('response_metadata', {})
                if meta.get('finish_reason') == 'length': issues.append('truncated')
                if meta.get('refusal'): issues.append('refusal')
        if issues:
            counts.update(issues); by_kind[kind] += 1; by_video[row['video_id']] += 1
            flagged.append({'video_id':row['video_id'], 'kind':kind, 'window':row['window'],
                            'issues':issues, 'words':len(row['target'].split()),
                            'raw_text':row['target']})
    warnings = {'over_preferred_words'}
    return {'collection_policy':COLLECTION_POLICY, 'total_summaries':len(events),
            'total_by_kind':dict(Counter(kinds[r['label_type']] for r in events)),
            'flagged_summaries':len(flagged),
            'format_or_response_issue_summaries':sum(bool(set(r['issues'])-warnings) for r in flagged),
            'issue_counts':dict(counts), 'flagged_by_kind':dict(by_kind),
            'flagged_by_video':dict(by_video), 'flagged':flagged,
            'filtered_summaries':0, 'rewritten_summaries':0,
            'content_review_complete':False, 'user_decision':'pending'}


def merge_collection(run_dir: Path) -> dict:
    from .full_run import load_manifest, records
    from .annotate import resume_start, source_digest
    from .build_data import build_files
    plan = load_manifest(run_dir)
    all_events, paths, incomplete = [], [], []
    for entry in plan['videos']:
        vid = entry['video_id']
        collected = run_dir/'collected'/f'{vid}.jsonl'
        path = collected if collected.exists() else run_dir/'teacher'/f'{vid}.jsonl'
        try:
            if source_digest(Path(entry['path'])) != entry['annotation_config']['source_sha256']:
                raise ValueError('Source changed after planning')
            rows = records(path)
            end = resume_start(rows, video_id=vid, duration_sec=entry['duration_sec'],
                               model=plan['model'], config=entry['annotation_config'], validate_text=False)
            all_events.extend(rows)
            if end != entry['duration_sec'] or len(rows) != entry['expected']['events']:
                raise ValueError(f'Incomplete video: saved through {end}s')
            paths.append(path)
        except (OSError, ValueError, KeyError, TypeError) as error:
            incomplete.append({'video_id':vid, 'error':str(error)})
    # Also count responses persisted inside a minute that never reached its
    # atomic event-batch publication (for example after an endpoint failure).
    kind_names = {'local':'local_10s', 'block':'block_60s', 'global':'global_prefix'}
    seen = {(r['video_id'], r['label_type'], tuple(r['window'])) for r in all_events}
    raw_only = 0
    for entry in plan['videos']:
        vid = entry['video_id']
        for path in sorted((run_dir/'raw'/vid).glob('*.json')):
            raw = json.loads(path.read_text())
            key = (vid, kind_names[raw['kind']], tuple(raw['window']))
            if key not in seen:
                all_events.append({'video_id':vid,'label_type':key[1],
                                   'window':raw['window'],'target':raw['raw_text']})
                seen.add(key); raw_only += 1
    report = quality_report(all_events, run_dir/'raw')
    report['responses_not_yet_in_minute_batches'] = raw_only
    report.update(status='incomplete' if incomplete else 'pending_review',
                  selected_videos=len(plan['videos']), complete_videos=len(paths),
                  incomplete_videos=incomplete, expected_events=plan['expected_events'],
                  expected_rows=plan['expected_rows'], prompt_sha256=plan['prompt_sha256'])
    if not incomplete:
        output = run_dir/'blocks.unreviewed.jsonl'
        count = build_files(paths, output, validate_text=False)
        if count != plan['expected_rows']: raise ValueError('Candidate row count differs from manifest')
        report.update(rows=count, candidate_file=str(output))
    atomic_json(run_dir/'quality_report.json', report)
    if incomplete:
        raise ValueError(f'{len(incomplete)} videos incomplete; see quality_report.json')
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run-dir',type=Path,required=True)
    args = parser.parse_args()
    report = merge_collection(args.run_dir)
    print(json.dumps({k:v for k,v in report.items() if k != 'flagged'}, indent=2))


if __name__ == '__main__': main()
