"""Choose among four completed fresh arms, then rerun the winner with all pairs.

Selection uses epoch 100 only, scans thresholds on the training set, and ranks
on-time text-correct tasks first, false WAIT triggers second, readiness accuracy
third. This is an overfit diagnostic, not held-out model selection.
If pure BCE wins, its existing checkpoint already supervises all endpoints;
record that no CL sampler exists instead of silently adding another loss.
"""
import argparse
import collections
import datetime
import hashlib
import json
import math
import os
from pathlib import Path
import subprocess
from .answer_metrics import endpoint_answer_metrics, first_trigger_metrics
from .scan_readiness_threshold import score


def read(path):
    return json.loads(path.read_text())


def write(path, value):
    tmp=path.with_suffix(path.suffix+'.partial')
    tmp.write_text(json.dumps(value,indent=2)+'\n');tmp.replace(path)


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def metrics(rows, threshold):
    first=first_trigger_metrics(rows,threshold);first.pop('records')
    return dict(threshold=threshold,answer=endpoint_answer_metrics(rows,threshold),
                first_trigger=first,readiness=score(rows,threshold))


def rank(m):
    return (m['first_trigger']['on_time_and_text_correct'], -m['readiness']['fp'],
            m['readiness']['accuracy'], -abs(m['threshold']-.5))


def threshold_selection(rows):
    if not rows or len({r['sample_id'] for r in rows})!=len(rows):
        raise ValueError('Empty or duplicate endpoint predictions')
    if any(not math.isfinite(r['probability']) or not 0<=r['probability']<=1 for r in rows):
        raise ValueError('Invalid probability')
    # Include all distinct decisions for p >= t, with .5/.35 retained as controls.
    thresholds=sorted({0.,1.,.5,.35,*(r['probability'] for r in rows)})
    candidates=[metrics(rows,t) for t in thresholds]
    best=max(candidates,key=rank)
    return dict(best=best,at05=metrics(rows,.5),at035=metrics(rows,.35))


def choose(arms):
    # Caller order is the predeclared tie-break after identical metrics/thresholds.
    return max(arms,key=lambda arm:rank(arm['evaluation']['best']))


def compare(root):
    plan=read(root/'selection_plan.json');arms=[];orders=[];data_hashes=[]
    for arm in plan['arms']:
        source=Path(arm['run_dir']);cfg=read(source/'experiment.json')
        if read(source/'status.json')['status']!='completed':
            raise ValueError(f'Arm not completed: {source}')
        if digest(source/'experiment.json')!=arm['config_sha256']:
            raise ValueError(f'Frozen config changed: {source}')
        if cfg['init_checkpoint'] is not None or cfg['restore_optimizer'] or not cfg['zero_update_warmup']:
            raise ValueError('Not a fresh-initialization comparison')
        if read(source/'initial_state.json')['core']['sha256']!=plan['initial_core_sha256']:
            raise ValueError('Initial model weights differ')
        if read(source/'optimizer_initialization.json')['step']!=0:
            raise ValueError('Nonzero initial optimizer step')
        training=[json.loads(line) for line in (source/'training.jsonl').read_text().splitlines()]
        if len(training)!=300 or any(row['phase']!='train' for row in training) or training[-1]['step']!=300 or training[-1]['epoch']!=100:
            raise ValueError('Training update budget differs')
        order=[json.loads(line)['batches'] for line in (source/'batch_order.jsonl').read_text().splitlines()]
        if order!=read(root/'planned_batch_order.json'):
            raise ValueError('Training batch order differs')
        orders.append(order)
        dh={name:digest(source/name) for name in ['samples.jsonl','tasks.jsonl','video_map.json']}
        data_hashes.append(dh)
        predictions=read(source/'eval_epoch_100.json')['predictions']
        expected_ids={json.loads(line)['sample_id'] for line in (source/'samples.jsonl').read_text().splitlines()}
        if {r['sample_id'] for r in predictions}!=expected_ids:
            raise ValueError('Evaluation endpoint coverage differs')
        arms.append(dict(name=arm['name'],run_dir=str(source),evaluation=threshold_selection(predictions),
                         prediction_sha256=digest(source/'eval_epoch_100.json')))
    if len(arms)!=4 or any(d!=data_hashes[0] for d in data_hashes):
        raise ValueError('Need four arms with identical data')
    winner=choose(arms)
    result=dict(scope=plan['scope'],selection_rule=plan['selection_rule'],arms=arms,winner=winner,
                initial_weights_and_update_budget_and_batches_verified=True)
    write(root/'four_arm_comparison.json',result)
    return result


def log(root, message):
    plan=read(root/'selection_plan.json')
    with Path(plan['project_log']).open('a') as f:
        f.write('\n- '+datetime.datetime.now(datetime.timezone.utc).isoformat()+' — '+message+'\n')


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run-dir',type=Path,required=True)
    args=parser.parse_args();root=args.run_dir.resolve()
    plan=read(root/'selection_plan.json')
    # All executable code was frozen and tested before the dependent job submission.
    for name,expected in read(root/'source_hashes.json').items():
        if digest(root/'code'/name)!=expected:raise ValueError(f'Frozen code changed: {name}')
    write(root/'status.json',dict(status='selecting',job_id=os.environ.get('SLURM_JOB_ID')))
    result=compare(root);winner=result['winner'];source=Path(winner['run_dir'])
    cfg=read(source/'experiment.json');best=winner['evaluation']['best']
    decision=dict(winner=winner['name'],source=str(source),threshold=best['threshold'],
                  checkpoint=str(source/'checkpoint_latest.pth'),selection=best)
    if not Path(decision['checkpoint']).is_file():raise ValueError('Selected checkpoint missing')
    if cfg['readiness_objective']=='bce':
        write(root/'experiment.json',cfg)
        decision.update(action='reuse_bce_checkpoint',reason='Pure BCE already supervises every WAIT/ANSWER endpoint; no CL pair sampler. No additional loss or redundant training introduced.')
        write(root/'selection.json',decision)
        write(root/'status.json',dict(status='completed',training_performed=False,**decision))
        log(root,'Four-arm selection completed: '+json.dumps(decision)+f'; report {root}/four_arm_comparison.json. No additional training: BCE already uses all endpoints.')
        print(json.dumps(decision,indent=2));return
    old_cfg=dict(cfg)
    cfg.update(timing_pair_mode='all_pairs',timing_pair_sampling='All Cartesian ANSWER x WAIT pairs within each (video,question); mean within question then mean across paired questions. No random pair sampling.',experiment='Fresh all-pairs rerun of four-arm winner: '+winner['name'])
    # Keep .5 for directly comparable training-time metrics; selected threshold is
    # evaluated separately. A threshold never changes gold-supervised training.
    write(root/'experiment.json',cfg)
    decision.update(action='fresh_all_pairs_training',init_checkpoint=None,training_threshold=.5,
                    config_difference={k:[old_cfg.get(k),v] for k,v in cfg.items() if old_cfg.get(k)!=v})
    write(root/'selection.json',decision)
    for name in ['samples.jsonl','tasks.jsonl','video_map.json']:
        if digest(root/name)!=digest(source/name):raise ValueError('Prepared data differs from winning arm')
    write(root/'selected_input_hashes.json',{n:digest(root/n) for n in ['experiment.json','samples.jsonl','tasks.jsonl','video_map.json']})
    log(root,'Selected all-pairs training BEFORE launch: '+json.dumps(decision)+f'; config {root}/experiment.json, frozen code {root}/code, logs {root}/logs; original pretrained initialization, same seed and fresh Adam,100epochs300updates,BS64GA1,1A100,lr1e-4,no holdout. Parent job '+os.environ.get('SLURM_JOB_ID','unknown'))
    env=dict(os.environ,RUN_DIR=str(root))
    subprocess.run(['/bin/bash',str(root/'train.sbatch')],env=env,check=True)
    training=[json.loads(line) for line in (root/'training.jsonl').read_text().splitlines()]
    counts=collections.Counter()
    for row in training:
        if row['timing_pair_mode']!='all_pairs' or row['timing_used_pairs']!=row['timing_pairs']:
            raise ValueError('Training did not use every available pair')
        counts[row['epoch']]+=row['timing_used_pairs']
    if len(training)!=300 or len(counts)!=100 or set(counts.values())!={plan['pairs_per_epoch']}:
        raise ValueError('All-pair coverage audit failed')
    if read(root/'initial_state.json')['core']['sha256']!=plan['initial_core_sha256']:
        raise ValueError('All-pair initial weights differ')
    if [json.loads(line)['batches'] for line in (root/'batch_order.jsonl').read_text().splitlines()]!=read(root/'planned_batch_order.json'):
        raise ValueError('All-pair batch order differs')
    rows=read(root/'eval_epoch_100.json')['predictions']
    write(root/'all_pairs_result.json',dict(evaluation=threshold_selection(rows),
          at_source_selected_threshold=metrics(rows,best['threshold']),source=winner,
          pairs_each_epoch=dict(counts),scope=plan['scope']))
    log(root,f'All-pairs training completed;100epochs300updates, all {plan["pairs_per_epoch"]} pairs verified each epoch; results {root}/all_pairs_result.json. These are train-set overfit metrics, not held-out results.')


if __name__=='__main__':
    try:
        main()
    except Exception as error:
        import sys
        if '--run-dir' in sys.argv:
            root=Path(sys.argv[sys.argv.index('--run-dir')+1]).resolve()
            if root.is_dir():write(root/'status.json',dict(status='failed',error=repr(error)))
        raise
