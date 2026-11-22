"""Validate both legacy and new decision rows; never expose annotation-only fields."""
import bisect
import collections
import hashlib
import json
import math
from pathlib import Path

QA_TYPES = ('current_observation', 'history_recall', 'event_notification', 'temporal_comparison')


def read_rows(path):
    return [json.loads(line) for line in Path(path).read_text().splitlines() if line.strip()]


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def load_examples(samples_path, tasks_path, video_map_path):
    tasks = read_rows(tasks_path)
    by_id = {row['task_id']: row for row in tasks}
    if len(by_id) != len(tasks):
        raise ValueError('Duplicate task IDs')
    videos = json.loads(Path(video_map_path).read_text())
    examples, seen = [], set()
    for row in read_rows(samples_path):
        task = by_id[row['task_id']]
        sid, vid = row['sample_id'], row['video_id']
        if sid in seen:
            raise ValueError(f'Duplicate sample: {sid}')
        seen.add(sid)
        start, current = map(float, row['video_window'])
        query = float(task['query_time'])
        answer = task.get('answer_time')
        target = row['target']
        ready = target.startswith('<ANSWER> ')
        if target != '<WAIT>' and not ready:
            raise ValueError(f'Invalid decision target: {sid}')
        expected = answer is not None and current >= float(answer)
        meta = row.get('meta', {})
        if (not all(math.isfinite(x) for x in (start, current, query))
                or start != 0 or current < query or query < 0
                or ready != expected or vid != task['video_id']
                or row['question'] != task['question']
                or float(meta.get('current_time', current)) != current
                or float(meta.get('query_time', query)) != query
                or int(meta.get('y', ready)) != int(ready)):
            raise ValueError(f'Inconsistent causal decision: {sid}')
        if answer is not None and (not math.isfinite(float(answer)) or float(answer) < query):
            raise ValueError(f'Invalid answer time: {sid}')
        if bool(task.get('untriggered', False)) and answer is not None:
            raise ValueError(f'Untriggered task has answer time: {sid}')
        if task.get('review_horizon') is not None and current > float(task['review_horizon']):
            raise ValueError(f'Beyond review horizon: {sid}')
        if ready and target[len('<ANSWER> '):] != task['reference_answer']:
            raise ValueError(f'Answer mismatch: {sid}')
        if task['qa_type'] not in QA_TYPES or vid not in videos:
            raise ValueError(f'Missing type or video: {sid}')
        # Explicit allowlist: answer_time, evidence, ASR and review notes stay out.
        examples.append(dict(sample_id=sid, task_id=row['task_id'], video_id=vid,
                             video_path=videos[vid], question=row['question'],
                             query_time=query, current_time=current,
                             qa_type=task['qa_type'], split=task.get('split', meta.get('split')),
                             ready=int(ready), answer=target[len('<ANSWER> '):] if ready else ''))
    if {e['task_id'] for e in examples} != set(by_id):
        raise ValueError('Tasks without decision rows')
    return examples


def prompt(example):
    return (f"Question asked at {example['query_time']:g} seconds. "
            f"Current time: {example['current_time']:g} seconds.\n"
            f"{example['question']}\nAnswer:")


def sample_times(current):
    times = [float(i) for i in range(math.floor(current) + 1)]
    if not times or times[-1] < current:
        times.append(float(current))
    return times


def causal_indices(frame_pts, times):
    if not frame_pts or any(b < a for a, b in zip(frame_pts, frame_pts[1:])):
        raise ValueError('Missing or nonmonotonic video timestamps')
    indices = [bisect.bisect_right(frame_pts, t) - 1 for t in times]
    if min(indices) < 0:
        raise ValueError('No observable video frame at requested time')
    return indices


def summary(examples):
    return dict(videos=len({e['video_id'] for e in examples}),
                qa=len({e['task_id'] for e in examples}), decisions=len(examples),
                labels=dict(collections.Counter('ANSWER' if e['ready'] else 'WAIT' for e in examples)),
                qa_types=dict(collections.Counter({e['task_id']: e['qa_type'] for e in examples}.values())))
