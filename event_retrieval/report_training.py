"""CPU-only post-job telemetry and held-out QA report; never modifies a model."""
import argparse
import collections
import csv
import datetime as dt
import json
import math
from pathlib import Path
import statistics
import subprocess
from endtoend.answer_metrics import answers_match


def rows(path):
    return [json.loads(x) for x in path.read_text().splitlines() if x.strip()]


def fraction(n, d):
    return {'correct_or_events': n, 'total': d, 'rate': n / d if d else None}


def telemetry(path):
    if not path.exists():
        return {'available': False}
    grouped = collections.defaultdict(list)
    invalid = 0
    with path.open() as f:
        for raw in csv.DictReader(f):
            r = {k.strip(): v.strip() for k, v in raw.items() if k and v is not None}
            try:
                t = dt.datetime.strptime(r['timestamp'], '%Y/%m/%d %H:%M:%S.%f').timestamp()
                u, used, total = [float(r[k].split()[0]) for k in ['utilization.gpu [%]', 'memory.used [MiB]', 'memory.total [MiB]']]
                if not (0 <= u <= 100 and 0 <= used <= total and total > 0):
                    raise ValueError('invalid telemetry')
                grouped[r.get('uuid', r.get('index', 'unknown'))].append((t, u, used, 100 * used / total, total))
            except (KeyError, ValueError):
                invalid += 1
    result = {'available': bool(grouped), 'invalid_rows': invalid, 'per_gpu': {},
              'scope': 'Full job telemetry: initialization, training, holdout-loss passes, checkpoint saves. Not isolated optimizer compute. Separate from index/evaluation jobs.',
              'method': 'Time-weighted trapezoidal means; gaps >5s excluded and coverage reported. VRAM utilization = memory.used / memory.total, not memory bandwidth utilization.'}
    for gpu, samples in grouped.items():
        samples.sort()
        areas = [0., 0., 0.]; seconds = 0.; gaps = 0.
        for a, b in zip(samples, samples[1:]):
            delta = b[0] - a[0]
            if delta <= 0:
                continue
            if delta > 5:
                gaps += delta
                continue
            seconds += delta
            for i in range(3):
                areas[i] += delta * (a[i + 1] + b[i + 1]) / 2
        means = [x / seconds if seconds else None for x in areas]
        result['per_gpu'][gpu] = dict(samples=len(samples), observed_span_seconds=samples[-1][0]-samples[0][0], covered_seconds=seconds, excluded_gap_seconds=gaps,
            mean_gpu_utilization_percent=means[0], mean_vram_used_mib=means[1], mean_vram_utilization_percent=means[2],
            peak_vram_used_mib=max(s[2] for s in samples), peak_vram_utilization_percent=max(s[3] for s in samples), total_vram_mib=samples[-1][4])
    return result


def exact(a, b):
    return ' '.join(str(a).casefold().split()) == ' '.join(str(b).casefold().split())


def group_metrics(records):
    answerable = [r for r in records if r['task']['answer_time'] is not None]
    counts = collections.Counter(); offsets = []
    for r in answerable:
        t = r['task']; when = r['first_answer_time']; gt = t['answer_time']; pred = r['last_decision'].get('answer') or ''
        forced = r['forced_answer_at_gt']; gold = t['reference_answer']
        content = when is not None and answers_match(pred, gold, t['qa_type'])
        counts['forced_text_match'] += answers_match(forced, gold, t['qa_type'])
        counts['forced_exact'] += exact(forced, gold)
        counts['first_answer_text_match'] += content
        counts['first_answer_exact'] += when is not None and exact(pred, gold)
        if when is None:
            counts['unanswered'] += 1
        else:
            delta = when - gt; offsets.append(delta)
            counts['early' if delta < 0 else 'late' if delta > 0 else 'on_time'] += 1
            counts['on_time_and_text_match'] += delta == 0 and content
            counts['on_time_and_exact'] += delta == 0 and exact(pred, gold)
            for tolerance in [1, 2, 5]:
                counts[f'text_match_within_{tolerance}s_after_gt'] += 0 <= delta <= tolerance and content
    names = ['forced_text_match', 'forced_exact', 'first_answer_text_match', 'first_answer_exact', 'on_time_and_text_match', 'on_time_and_exact', 'early', 'late', 'on_time', 'unanswered', *[f'text_match_within_{s}s_after_gt' for s in [1, 2, 5]]]
    late = sorted(x for x in offsets if x > 0)
    return dict(questions=len(records), answerable=len(answerable), metrics={k:fraction(counts[k],len(answerable)) for k in names},
        timing=dict(triggered_count=len(offsets), signed_offset_mean_s=statistics.mean(offsets) if offsets else None,
                    absolute_offset_median_s=statistics.median(abs(x) for x in offsets) if offsets else None,
                    late_count=len(late), late_median_s=statistics.median(late) if late else None,
                    late_p90_s=late[math.ceil(.9*len(late))-1] if late else None,
                    definition='first ANSWER video timestamp minus annotated answer time; unanswered excluded from offsets and reported separately; not inference wall-clock latency'))


def holdout_report(root):
    required = ['replay/quality.json','replay/judge_inputs.jsonl','replay/summary.json','fit/holdout_metrics.json','fit/holdout_predictions.jsonl']
    missing = [n for n in required if not (root/n).is_file()]
    if missing:
        return dict(available=False,missing=missing)
    tasks = rows(root/'data/holdout/tasks.jsonl'); records = rows(root/'replay/judge_inputs.jsonl'); endpoints = rows(root/'fit/holdout_predictions.jsonl')
    assert len(records)==len(tasks) and {r['task']['task_id'] for r in records}=={t['task_id'] for t in tasks}
    expected = {r['sample_id'] for r in rows(root/'data/holdout/samples.jsonl')}
    assert len(endpoints)==len(expected) and {r['sample_id'] for r in endpoints}==expected
    assert all(r['action'] in {'ANSWER','WAIT'} for r in endpoints)
    positives=[r for r in endpoints if r['ready']]; negatives=[r for r in endpoints if not r['ready']]
    tp=sum(r['action']=='ANSWER' for r in positives); fp=sum(r['action']=='ANSWER' for r in negatives)
    groups=collections.defaultdict(list)
    for r in records:
        t=r['task']; groups['type:'+t['qa_type']].append(r);groups['video:'+t['video_id']].append(r)
        groups['timing:'+('unanswerable' if t['answer_time'] is None else 'delayed' if t['answer_time']>t['query_time'] else 'instant')].append(r)
    result=dict(available=True,scope='16 development holdout videos, not independent Test; annotations and dense causal first-answer replay evaluated separately',
        scoring='exact: case/whitespace normalized; text_match: StreamReady normalized/numeric/short-phrase heuristic; neither is a semantic judge',
        semantic_judge_status='not performed; paired forced/first-answer judge inputs retained',
        endpoints=dict(wait_false_trigger=fraction(fp,len(negatives)),answer_missed=fraction(len(positives)-tp,len(positives)),action_accuracy=fraction(tp+len(negatives)-fp,len(endpoints)),
                       answer_precision=fraction(tp,tp+fp),answer_recall=fraction(tp,len(positives)),balanced_accuracy=(tp/len(positives)+(len(negatives)-fp)/len(negatives))/2 if positives and negatives else None),
        dense=group_metrics(records),breakdown={k:group_metrics(v) for k,v in sorted(groups.items())})
    result['video_macro_on_time_and_text_match']=statistics.mean(g['metrics']['on_time_and_text_match']['rate'] for k,g in result['breakdown'].items() if k.startswith('video:') and g['answerable'])
    return result


def render(report):
    out=['# Event Retrieval training report', '', 'Generated: '+report['generated_utc'], '', '## GPU telemetry', '', 'Job-wide averages include initialization, holdout loss and saves; index and generation evaluation are separate.', '', '| Job / GPU | Mean GPU busy | Mean VRAM occupied | Peak VRAM occupied |', '|---|---:|---:|---:|']
    for phase, data in report['gpu'].items():
        for gpu, v in data.get('per_gpu',{}).items():
            if v['mean_gpu_utilization_percent'] is not None:
                out.append(f"| {phase} / {gpu} | {v['mean_gpu_utilization_percent']:.2f}% | {v['mean_vram_utilization_percent']:.2f}% ({v['mean_vram_used_mib']/1024:.2f} GiB) | {v['peak_vram_utilization_percent']:.2f}% ({v['peak_vram_used_mib']/1024:.2f} GiB) |")
    h=report['holdout'];out+=['','## Holdout','']
    if not h['available']:
        out+=['Evaluation not complete. Missing: '+', '.join(h['missing'])]
    else:
        out += [h['scope'], '', h['scoring'], '', '| Metric | Count | Rate |','|---|---:|---:|']
        metrics={**h['dense']['metrics'],**{k:v for k,v in h['endpoints'].items() if isinstance(v,dict)}}
        for name,v in metrics.items():
            rate='N/A' if v['rate'] is None else f"{100*v['rate']:.2f}%"
            out.append(f"| {name} | {v['correct_or_events']}/{v['total']} | {rate} |")
        out+=['', 'Per-type, per-video, instant/delayed breakdown and delay statistics are in the JSON report.', 'Semantic judge not performed; judge inputs are retained.']
    out+=['','## Scheduler state','', '```',report['scheduler'], '```', '']
    return '\n'.join(out)


def main():
    p=argparse.ArgumentParser();p.add_argument('--run-dir',type=Path,required=True);p.add_argument('--stage',choices=['training','final'],required=True);args=p.parse_args()
    root=args.run_dir.resolve(); jobs=json.loads((root/'jobs.json').read_text())
    scheduler=subprocess.run(['sacct','-X','-P','-j',','.join(jobs.values()),'--format=JobID,JobName,State,Elapsed,Start,End,ExitCode'],capture_output=True,text=True)
    epoch_rows=[]
    log=root/'logs'/f"slurm-{jobs['train']}.log"
    if log.exists():
        for line in log.read_text(errors='replace').splitlines():
            try:r=json.loads(line)
            except ValueError:continue
            if isinstance(r,dict) and 'train_loss' in r:epoch_rows.append(r)
    report=dict(generated_utc=dt.datetime.now(dt.timezone.utc).isoformat(),stage=args.stage,run_dir=str(root),jobs=jobs,
        scheduler=scheduler.stdout or scheduler.stderr,epochs=epoch_rows,
        gpu={k:telemetry(root/'logs'/f'gpu-{v}.csv') for k,v in jobs.items()},holdout=holdout_report(root))
    dest=root/'reports';dest.mkdir(exist_ok=True)
    for suffix,content in [('json',json.dumps(report,indent=2)+'\n'),('md',render(report))]:
        path=dest/(args.stage+'_report.'+suffix);tmp=path.with_suffix(path.suffix+'.partial');tmp.write_text(content);tmp.replace(path)
    print(json.dumps(dict(report=str(dest/(args.stage+'_report.md')),holdout_available=report['holdout']['available'])))


if __name__=='__main__':
    main()
