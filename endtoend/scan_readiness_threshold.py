"""Evaluate saved endpoint predictions; do not rerun or modify the model."""
import argparse
import collections
import csv
import hashlib
import json
from pathlib import Path


def score(rows, threshold):
    tp = sum(bool(r['expected_ready']) and r['probability'] >= threshold for r in rows)
    fp = sum(not r['expected_ready'] and r['probability'] >= threshold for r in rows)
    positive = sum(bool(r['expected_ready']) for r in rows)
    negative = len(rows) - positive
    fn, tn = positive - tp, negative - fp
    groups = collections.defaultdict(list)
    for r in rows:
        groups[r['task_id']].append(r)
    early = on_time = missed = correct_wait = 0
    for group in groups.values():
        first = next((r for r in sorted(group, key=lambda r: r['current_time'])
                      if r['probability'] >= threshold), None)
        answerable = any(r['expected_ready'] for r in group)
        if first is not None:
            on_time += bool(first['expected_ready'])
            early += not bool(first['expected_ready'])
        elif answerable:
            missed += 1
        else:
            correct_wait += 1
    return dict(threshold=threshold, tp=tp, fp=fp, tn=tn, fn=fn,
                accuracy=(tp + tn) / len(rows), recall=tp / positive,
                false_positive_rate=fp / negative,
                precision=tp / (tp + fp) if tp + fp else None,
                f1=2 * tp / (2 * tp + fp + fn),
                balanced_accuracy=(tp / positive + tn / negative) / 2,
                sparse_first_trigger_on_time=on_time, sparse_first_trigger_early=early,
                sparse_first_trigger_missed=missed, sparse_first_trigger_correct_wait=correct_wait)


def scan(rows):
    # For score>=threshold, each constant-decision interval is (previous, current].
    thresholds = sorted({0., 1., *(float(r['probability']) for r in rows)})
    intervals = []
    for i, value in enumerate(thresholds):
        intervals.append(dict(lower=thresholds[i - 1] if i else 0.,
                              lower_inclusive=i == 0, upper=value, **score(rows, value)))
    best = {}
    for metric in ['accuracy', 'balanced_accuracy', 'f1', 'sparse_first_trigger_on_time']:
        maximum = max(r[metric] for r in intervals)
        ranges = []
        for r in intervals:
            if abs(r[metric] - maximum) > 1e-12:
                continue
            if ranges and ranges[-1]['upper'] == r['lower']:
                ranges[-1]['upper'] = r['upper']
            else:
                ranges.append({k:r[k] for k in ['lower','lower_inclusive','upper']})
        best[metric] = dict(value=maximum, intervals=ranges)
    zero_fp = [r for r in intervals if r['fp'] == 0]
    best['zero_false_positive_max_recall'] = max(zero_fp, key=lambda r:r['recall']) if zero_fp else None
    return intervals, best


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--predictions', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    raw = args.predictions.read_bytes()
    saved = json.loads(raw)
    rows = saved['predictions']
    assert rows and len({r['sample_id'] for r in rows}) == len(rows)
    assert all(0 <= r['probability'] <= 1 for r in rows)
    baseline = score(rows, .5)
    assert abs(baseline['accuracy'] - saved['metrics']['readiness_accuracy']) < 1e-12
    intervals, best = scan(rows)
    selected = [score(rows, t) for t in [.01,.02,.05,.1,.15,.2,.25,.3,.35,.4,.45,.5,.6,.7,.8,.9]]
    report = dict(source=str(args.predictions.resolve()), source_sha256=hashlib.sha256(raw).hexdigest(),
                  scope='same 10 training videos, 175 saved annotated endpoints; no holdout; no new model forward or training',
                  rule='ANSWER iff probability >= threshold',
                  primary_objective='endpoint WAIT/ANSWER accuracy',
                  limitations='First-trigger replay uses sparse annotated endpoints only, not dense streaming. Threshold selection is in-sample. Frozen run config and original predictions unchanged.',
                  answer_exact_match=saved['metrics']['answer_exact_match'],
                  samples=len(rows), positives=sum(r['expected_ready'] for r in rows),
                  best=best, selected=selected,
                  by_type_035={kind:score([r for r in rows if r['qa_type']==kind],.35)
                               for kind in sorted({r['qa_type'] for r in rows})
                               if any(r['expected_ready'] for r in rows if r['qa_type']==kind)
                               and any(not r['expected_ready'] for r in rows if r['qa_type']==kind)})
    args.output.mkdir(parents=True, exist_ok=False)
    (args.output/'summary.json').write_text(json.dumps(report,indent=2)+'\n')
    with (args.output/'threshold_intervals.csv').open('w') as f:
        writer=csv.DictWriter(f,fieldnames=list(intervals[0]));writer.writeheader();writer.writerows(intervals)
    (args.output/'predictions_threshold_035.jsonl').write_text(''.join(json.dumps(dict(r,ready=r['probability']>=.35,threshold=.35,forced_answer_probe=r['probability']<.35))+'\n' for r in rows))
    print(json.dumps(report,indent=2))


if __name__ == '__main__':
    main()
