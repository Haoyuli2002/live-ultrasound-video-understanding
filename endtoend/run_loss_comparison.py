"""Run three independent fresh-initialization arms sequentially on one allocation."""
import argparse
import csv
import datetime
import json
import os
from pathlib import Path
import subprocess
from .answer_metrics import endpoint_answer_metrics, first_trigger_metrics
from .scan_readiness_threshold import score, scan


def write(path, data):
    tmp = path.with_suffix(path.suffix + '.partial')
    tmp.write_text(json.dumps(data, indent=2) + '\n')
    tmp.replace(path)


def report(suite):
    plan = json.loads((suite/'comparison_plan.json').read_text())
    entries = []
    initial_hashes = set()
    batch_orders = []
    for arm in plan['arms']:
        run = Path(arm['run_dir'])
        state_file = run/'status.json'
        state = json.loads(state_file.read_text()) if state_file.exists() else {'status':'not_started'}
        entry = dict(arm=arm['name'], status=state.get('status'), run_dir=str(run))
        manifest = run/'initial_state.json'
        if manifest.exists():
            entry['initial_core_sha256'] = json.loads(manifest.read_text())['core']['sha256']
            initial_hashes.add(entry['initial_core_sha256'])
        evaluation = run/'eval_epoch_100.json'
        if state.get('status')=='completed' and evaluation.exists():
            saved = json.loads(evaluation.read_text()); rows = saved['predictions']
            training = [json.loads(line) for line in (run/'training.jsonl').read_text().splitlines()]
            assert len(training)==300 and all(r['phase']=='train' for r in training), 'Unexpected optimizer update count'
            assert training[-1]['step']==300 and training[-1]['epoch']==100
            assert json.loads((run/'optimizer_initialization.json').read_text())['step']==0
            order = json.loads('['+','.join((run/'batch_order.jsonl').read_text().splitlines())+']')
            batch_orders.append(order)
            expected = json.loads((suite/'planned_batch_order.json').read_text())
            assert [r['batches'] for r in order]==expected, 'Training order differs from frozen plan'
            entry['final_metrics']=saved['metrics']
            entry['thresholds']={}
            for threshold in [.5,.35]:
                first = first_trigger_metrics(rows,threshold)
                first.pop('records')
                entry['thresholds'][str(threshold)] = dict(readiness=score(rows,threshold),
                    answers=endpoint_answer_metrics(rows,threshold),first_trigger=first)
            _, entry['threshold_scan_best'] = scan(rows)
        entries.append(entry)
    assert len(initial_hashes)<=1, 'Initial weights mismatch'
    assert not batch_orders or all(order==batch_orders[0] for order in batch_orders), 'Batch order mismatch'
    complete = len(entries)==3 and all(e['status']=='completed' and 'final_metrics' in e for e in entries)
    result=dict(status='completed' if complete else 'partial', arms=entries,
                common_initial_weights_verified=len(initial_hashes)==1 and all('initial_core_sha256' in e for e in entries),
                completed_arms_batch_order_verified=len(batch_orders),
                scope='10 training videos; 100epochs/300updates each; single seed; no holdout; sparse endpoint replay, not dense streaming or ARS')
    write(suite/'comparison_results.json',result)
    lines=['# Fresh-initialization loss comparison','',result['scope'],'',
           '| Arm | Status | Forced answer Acc | First-trigger answer Acc | On-time correct | WAIT false triggers | Readiness Acc |',
           '| --- | --- | --- | --- | --- | --- | --- |']
    for e in entries:
        if 'final_metrics' not in e:
            lines.append(f"| {e['arm']} | {e['status']} | — | — | — | — | — |")
            continue
        r=e['thresholds']['0.5']; a=r['answers'];t=r['first_trigger'];g=r['readiness']
        lines.append(f"| {e['arm']} | completed | {a['answer_accuracy']:.1%} | {t['answer_accuracy']:.1%} | {t['on_time_and_text_correct']}/{t['answerable_tasks']} | {g['fp']}/104 | {g['accuracy']:.1%} |")
    lines+=['','Primary threshold: 0.5. JSON also contains 0.35 and exact in-sample threshold scans.',
            'Early text-correct answers contribute to answer Acc but not on-time correctness.',
            'All arms include answer NTP/CE. BCE arm is unweighted; hybrid BCE ANSWER weight is 2.']
    (suite/'comparison_results.md').write_text('\n'.join(lines)+'\n')
    return result


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--suite-dir',type=Path,required=True)
    parser.add_argument('--report-only',action='store_true')
    args=parser.parse_args();suite=args.suite_dir.resolve()
    if args.report_only:
        print(json.dumps(report(suite)));return
    plan=json.loads((suite/'comparison_plan.json').read_text()); outcomes=[]
    for arm in plan['arms']:
        run=Path(arm['run_dir']);env=dict(os.environ,RUN_DIR=str(run))
        job=os.environ.get('SLURM_JOB_ID','local')
        write(suite/'suite_status.json',dict(status='running',active_arm=arm['name'],completed=outcomes))
        with (run/'logs'/f'job_{job}.out').open('w') as stdout, (run/'logs'/f'job_{job}.err').open('w') as stderr:
            outcome=subprocess.run(['/bin/bash',str(run/'train.sbatch')],env=env,stdout=stdout,stderr=stderr)
        outcomes.append(dict(arm=arm['name'],returncode=outcome.returncode,finished_at=datetime.datetime.now(datetime.timezone.utc).isoformat()))
        write(suite/'execution.json',outcomes)
        report(suite)
    result=report(suite)
    passed=all(r['returncode']==0 for r in outcomes) and result['status']=='completed'
    write(suite/'suite_status.json',dict(status='completed' if passed else 'failed',completed=outcomes))
    if not passed:raise SystemExit(1)


if __name__=='__main__':main()
