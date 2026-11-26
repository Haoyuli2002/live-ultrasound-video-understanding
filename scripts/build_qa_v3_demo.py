"""Export the annotated QA and fixed-endpoint predictions for the QA explorer."""
import argparse
import json
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def read_rows(path):
    return [json.loads(line) for line in path.read_text().splitlines() if line]


def build(run):
    metrics = json.loads((run / 'fit/overfit_metrics.json').read_text())
    audit = json.loads((run / 'data/audit.json').read_text())
    ends = {(r['video_id'], r['question_id']): r['coverage_end_s'] for r in audit['wait_review']}
    predictions = defaultdict(list)
    for row in read_rows(run / 'fit/overfit_predictions.jsonl'):
        predictions[row['task_id']].append(row)
    videos = {}
    for vid in audit['videos']:
        tasks = read_rows(run / 'data' / (vid + '.jsonl'))
        for task in tasks:
            key = vid + '/' + task['question_id']
            rows = sorted(predictions.pop(key), key=lambda row: row['current_time'])
            for row in rows:
                assert row['question'] == task['question'] and row['query_time'] == task['query_time']
                assert bool(row['ready']) == (row['current_time'] == task['answer_time'])
                if row['ready']:
                    assert row['answer'] == task['reference_answer'] and row['forced_answer']
            assert rows[0]['current_time'] == task['query_time']
            assert sum(bool(row['ready']) for row in rows) == int(task['answer_time'] is not None)
            task['task_id'] = key
            task['coverage_end'] = ends.get((vid, task['question_id']), task['answer_time'])
            task['decisions'] = [dict(time=r['current_time'], ready=bool(r['ready']),
                                      action=r['action'], text=r['prediction'], forced=r['forced_answer']) for r in rows]
            normalize = lambda t: ' '.join(t.casefold().split())
            task['different'] = any(r['ready'] and normalize(r['forced_answer']) !=
                                    normalize(task['reference_answer']) for r in rows)
        videos[vid] = tasks
    assert not predictions and sum(map(len, videos.values())) == metrics['questions']
    assert sum(len(t['decisions']) for tasks in videos.values() for t in tasks) == metrics['endpoints']
    assert sum(t['different'] for tasks in videos.values() for t in tasks) == metrics['answer_endpoints'] - metrics['forced_exact_answer']
    return dict(release='qa_v3_holdout10', epoch=metrics['epoch'], videos=videos)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('run', type=Path)
    args = parser.parse_args()
    payload = json.dumps(build(args.run), ensure_ascii=False, separators=(',', ':')) + '\n'
    for folder in ('viz/qa-examples', 'demo/ultrasound-qa'):
        (ROOT / folder / 'qa-v3.json').write_text(payload)
