"""Export a saved holdout generation pass with its causal video window and GT."""
import argparse
from collections import Counter, defaultdict
import json
from pathlib import Path
from tempfile import NamedTemporaryFile

from .data import load_examples, read_rows


def write_rows(path, rows):
    with NamedTemporaryFile('w', dir=path.parent, delete=False) as output:
        tmp = Path(output.name)
        try:
            for row in rows:
                output.write(json.dumps(row, ensure_ascii=False) + '\n')
            output.flush()
            tmp.replace(path)
        finally:
            tmp.unlink(missing_ok=True)


def export(run, step=None):
    cfg = json.loads((run / 'experiment.json').read_text())
    final = cfg['expected_optimizer_steps']
    step = final if step is None else step
    src = run / f'eval_step_{step:06d}.json'
    doc = json.loads(src.read_text())
    metrics = doc['metrics']
    if metrics['split'] != 'holdout' or metrics['optimizer_step'] != step:
        raise ValueError(f'Wrong evaluation split or step in {src}')
    holdout = run / cfg['holdout']['directory']
    examples = load_examples(holdout/'samples.jsonl', holdout/'tasks.jsonl', holdout/'video_map.json')
    pred = {p['sample_id']: p for p in doc['predictions']}
    if len(pred) != len(doc['predictions']) or pred.keys() != {e['sample_id'] for e in examples}:
        raise ValueError('Missing, duplicate or unexpected holdout predictions')
    for media in {e['video_path'] for e in examples}:
        if not Path(media).is_file():
            raise FileNotFoundError(media)
    endpoints = []
    for e in examples:
        p = pred[e['sample_id']]
        if (p['task_id'] != e['task_id'] or p['current_time'] != e['current_time']
                or p['expected_ready'] != e['ready'] or p['reference'] != e['answer']):
            raise ValueError(f"Prediction and ground truth disagree for {e['sample_id']}")
        endpoints.append(dict(
            sample_id=e['sample_id'], task_id=e['task_id'], qa_type=e['qa_type'], question=e['question'],
            video_id=e['video_id'], video_path=e['video_path'], video_window_s=[0, e['current_time']],
            question_time_s=e['query_time'], ground_truth_action='ANSWER' if e['ready'] else 'WAIT',
            ground_truth_answer=e['answer'] if e['ready'] else None,
            predicted_action='ANSWER' if p['ready'] else 'WAIT', readiness_probability=p['probability'],
            emitted_answer=p['answer'] if p['ready'] else None, generated_answer=p['answer'],
            forced_generation=p['forced_answer_probe'], optimizer_step=step))
    dest = run / 'judge_inputs' / ('final' if step == final else f'step_{step:06d}')
    dest.mkdir(parents=True, exist_ok=True)
    answers = [r for r in endpoints if r['ground_truth_action'] == 'ANSWER']
    write_rows(dest / 'answers.jsonl', answers)
    write_rows(dest / 'endpoints.jsonl', endpoints)
    by_task = defaultdict(list)
    for row in endpoints:
        by_task[row['task_id']].append(row)
    questions = []
    for task in read_rows(holdout / 'tasks.jsonl'):
        timeline = sorted(by_task[task['task_id']], key=lambda row: row['video_window_s'][1])
        first = timeline[0]
        decisions = [
            {key: row[key] for key in (
                'sample_id', 'video_window_s', 'ground_truth_action', 'ground_truth_answer',
                'predicted_action', 'readiness_probability', 'emitted_answer', 'generated_answer',
                'forced_generation')}
            for row in timeline
        ]
        questions.append(dict(
            task_id=task['task_id'], qa_type=task['qa_type'], response_group=task['response_group'],
            question=task['question'], video_id=task['video_id'], video_path=first['video_path'],
            question_time_s=task['query_time'], answer_time_s=task['answer_time'],
            reference_answer=task['reference_answer'], optimizer_step=step, decisions=decisions))
    write_rows(dest / 'all_qa.jsonl', questions)
    counts = Counter(q['response_group'] for q in questions)
    print(f'{dest}: {len(questions)} questions {dict(counts)}, {len(answers)} labeled answers, '
          f'{len(endpoints)} endpoints', flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run-dir', type=Path, required=True)
    parser.add_argument('--step', type=int, help='Defaults to the final optimizer step')
    args = parser.parse_args()
    export(args.run_dir.resolve(), args.step)
