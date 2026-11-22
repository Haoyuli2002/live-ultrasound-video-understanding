"""ProReady-QA answer matching, ported from official readiness_vqa.py.

Source: https://github.com/sacrcv/StreamReady/blob/main/lavis/tasks/readiness_vqa.py
Use that task's answers_match, not the different generic answer_matching helper.
No LLM judge. Timing is evaluated separately from answer-text correctness.
"""
import argparse
import collections
import hashlib
import json
from pathlib import Path
import re
import string


def normalize_answer(text):
    text = str(text).lower().strip().translate(str.maketrans('', '', string.punctuation))
    return ' '.join(w for w in text.split() if w not in {'a', 'an', 'the'})


def _first_number(text):
    match = re.search(r'-?\d+(?:\.\d+)?', str(text))
    return float(match.group()) if match else None


def answers_match(pred, gold, task=''):
    p, g = normalize_answer(pred), normalize_answer(gold)
    if p == g:
        return True
    if task == 'REC' or (_first_number(g) is not None and len(g.split()) <= 3):
        pn, gn = _first_number(p), _first_number(g)
        if pn is not None and gn is not None:
            return abs(pn - gn) < 1e-6
    if g and len(g.split()) <= 4 and g in p:
        return True
    return False


def endpoint_answer_metrics(rows, threshold):
    positives = [r for r in rows if r['expected_ready']]
    correct = [answers_match(r['answer'], r['reference'], r['qa_type']) for r in positives]
    total = len(positives)
    gated = sum(c and r['probability'] >= threshold for r, c in zip(positives, correct))
    groups = collections.defaultdict(list)
    for r, c in zip(positives, correct):
        groups[r['qa_type']].append((r, c))
    per_type = {kind:dict(n=len(values), answer_accuracy=sum(c for _,c in values)/len(values),
                         gated_answer_accuracy=sum(c and r['probability']>=threshold for r,c in values)/len(values))
                for kind, values in groups.items()}
    return dict(answer_accuracy=sum(correct)/total if total else None,
                answer_correct_count=sum(correct), answer_count=total,
                gated_answer_accuracy=gated/total if total else None,
                gated_answer_correct_count=gated,
                answer_macro_accuracy=sum(x['answer_accuracy'] for x in per_type.values())/len(per_type) if per_type else None,
                answer_metrics_by_type=per_type,
                answer_scoring='StreamReady readiness_vqa.answers_match: normalized/numeric/short-phrase match')


def first_trigger_metrics(rows, threshold):
    """Replay annotated endpoints only. Count no answer as wrong, including in denominator."""
    groups = collections.defaultdict(list)
    for r in rows:
        groups[r['task_id']].append(r)
    scored = []
    for task, group in groups.items():
        refs = {r['reference'] for r in group if r['expected_ready']}
        if len(refs) > 1:
            raise ValueError(f'Conflicting task answers: {task}')
        first = next((r for r in sorted(group,key=lambda r:r['current_time'])
                      if r['probability'] >= threshold),None)
        gold = next(iter(refs)) if refs else None
        correct = answers_match(first['answer'],gold,first['qa_type']) if first and gold is not None else False
        scored.append(dict(task_id=task,answerable=gold is not None,triggered=first is not None,
                           answer_time=first['current_time'] if first else None,
                           early=bool(first and not first['expected_ready']),
                           text_correct=bool(correct),on_time_and_text_correct=bool(correct and first['expected_ready']),
                           prediction=first['answer'] if first else '',reference=gold))
    answerable = [r for r in scored if r['answerable']]
    n = len(answerable)
    return dict(answerable_tasks=n, unanswerable_tasks=len(scored)-n,
                answer_correct=sum(r['text_correct'] for r in answerable),
                answer_accuracy=sum(r['text_correct'] for r in answerable)/n if n else None,
                on_time_and_text_correct=sum(r['on_time_and_text_correct'] for r in answerable),
                early_tasks=sum(r['early'] for r in scored),
                no_answer_tasks=sum(not r['triggered'] for r in answerable),
                records=scored)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--predictions',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    raw=args.predictions.read_bytes(); rows=json.loads(raw)['predictions']
    report=dict(source=str(args.predictions.resolve()),source_sha256=hashlib.sha256(raw).hexdigest(),
                scope='10 training videos; saved forced-decoding endpoint predictions, not held-out or dense streaming evaluation',
                official_matcher='https://github.com/sacrcv/StreamReady/blob/main/lavis/tasks/readiness_vqa.py',
                timing_note='Sparse first-trigger replay; early content-correct answers count in answer accuracy but fail on-time metric. No ARS computed without validated evidence windows/dense inference.',
                unanswerable_policy='One task has no reference answer; exclude from answer accuracy and retain in trigger diagnostics.',
                thresholds={str(t):dict(endpoint=endpoint_answer_metrics(rows,t),sparse_first_trigger=first_trigger_metrics(rows,t)) for t in [.5,.35]})
    args.output.mkdir(parents=True,exist_ok=False)
    (args.output/'metrics.json').write_text(json.dumps(report,indent=2)+'\n')
    for t,result in report['thresholds'].items():
        print(t,json.dumps(dict(endpoint=result['endpoint'],sparse_first_trigger={k:v for k,v in result['sparse_first_trigger'].items() if k!='records'})))


if __name__ == '__main__':
    main()
