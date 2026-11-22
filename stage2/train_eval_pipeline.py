"""Official-base Q-Former DDP training and sharded eval in one allocation."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys

from .provenance import digest


def balanced_shards(manifest, workers):
    if workers < 1:
        raise ValueError('workers must be positive')
    items = [v for v in manifest['videos'] if v['split'] == 'eval']
    shards = [[] for _ in range(min(workers, len(items)))]; sizes = [0]*len(shards)
    for v in sorted(items, key=lambda v: (-v['targets'], v['video_id'])):
        i = min(range(len(shards)), key=lambda j: (sizes[j], j))
        shards[i].append(v['video_id']); sizes[i] += v['targets']
    return shards


def execute(args, env=None):
    print('[pipeline]', ' '.join(map(str,args)), flush=True)
    subprocess.run(list(map(str,args)), env=env, check=True)


def official_smoke_gate(root):
    report = json.loads((root/'smoke/smoke_report.json').read_text())
    if report['status'] != 'functional_smoke_passed' or report['init_adapter_sha256'] is not None:
        raise ValueError('A passed official-initialized smoke is required')
    if report['model_name'] != 'Qwen/Qwen3.5-2B':
        raise ValueError('Wrong smoke base model')
    for name, value in report['implementation_sha256'].items():
        if digest(Path(__file__).parent/name) != value:
            raise ValueError('Implementation changed after smoke')
    source = json.loads((root/'smoke/input_manifest.json').read_text())['source_sha256']
    paths = json.loads((root/'video_map.json').read_text())
    if report['source_sha256'] != source or digest(paths['P3FoSFIrfQA']) != source:
        raise ValueError('Smoke source changed')


def aggregate(root, shards, eval_dir=None):
    eval_dir=Path(eval_dir) if eval_dir is not None else root/'eval'
    from .reconstruct import summarize
    from .data import load_blocks
    from .train_full import tasks_for_video
    rows=[]
    for i in range(len(shards)):
        folder=eval_dir/f'rank-{i}'
        if json.loads((folder/'report.json').read_text())['status'] != 'completed':
            raise ValueError('Incomplete eval worker')
        rows.extend(json.loads(s) for s in (folder/'predictions.jsonl').read_text().splitlines())
    actual=[(r['video_id'],r['kind'],tuple(r['window']),r['condition']) for r in rows]
    expected=set()
    for vid, blocks in load_blocks(root/'release/eval.jsonl',validate_text=False).items():
        for t in tasks_for_video(vid,blocks):
            for c in ('real','zero','other_window'):
                expected.add((vid,t['kind'],(0 if t['kind']=='global' else t['start'],t['end']),c))
    if len(actual)!=len(set(actual)) or set(actual)!=expected:
        raise ValueError('Missing, extra or duplicate evaluation targets')
    report=dict(status='completed',records=len(rows),videos=sum(map(len,shards)),scores=summarize(rows),
                skipped=sum('skipped' in r for r in rows),
                empty_outputs=sum(not r['prediction'].strip() for r in rows if 'prediction' in r),
                interpretation='Held-out eval; lexical metrics do not establish clinical quality.')
    (eval_dir/'predictions.jsonl').write_text(''.join(json.dumps(r,ensure_ascii=False)+'\n' for r in rows))
    (eval_dir/'report.json').write_text(json.dumps(report,indent=2))
    return report


def evaluate_checkpoint(root,workers,manifest,env,gpu_ids,checkpoint='final',eval_dir=None):
    python=sys.executable
    shards=balanced_shards(manifest,workers)
    eval_dir=Path(eval_dir) if eval_dir is not None else root/'eval'
    eval_dir.mkdir(parents=True,exist_ok=True)
    (eval_dir/'shards.json').write_text(json.dumps(shards,indent=2))
    children=[];logs=[]
    try:
        for i,ids in enumerate(shards):
            output=eval_dir/f'rank-{i}'
            if (output/'report.json').is_file():
                continue
            if output.exists():
                raise FileExistsError(f'Partial eval output requires inspection before retry: {output}')
            log=(root/'logs'/f'eval_{checkpoint}_rank{i}.log').open('a');logs.append(log)
            cmd=[python,'-m','stage2.reconstruct','--eval-jsonl',str(root/'release/eval.jsonl'),
                 '--release-manifest',str(root/'release/manifest.json'),'--video-path-map',str(root/'video_map.json'),
                 '--adapter-path',str(root/'training'/checkpoint),'--max-new-tokens','160','--save-memories',
                 '--output-dir',str(output),*[f'--video-id={vid}' for vid in ids]]
            children.append(subprocess.Popen(cmd,env=dict(env,CUDA_VISIBLE_DEVICES=gpu_ids[i]),stdout=log,stderr=subprocess.STDOUT))
        failures=[p.wait() for p in children]
        if any(failures):
            raise RuntimeError(f'Eval worker failures: {failures}; see logs/eval_rank*.log')
    finally:
        for p in children:
            if p.poll() is None:p.terminate();p.wait()
        for f in logs:f.close()
    report=aggregate(root,shards) if checkpoint=='final' else aggregate(root,shards,eval_dir)
    return report


def run(root, workers, epochs, eval_each_epoch=False):
    from .release import validate_release
    manifest=validate_release(root/'release/manifest.json')
    from .dataset_versions import validate_dataset_version
    plan=json.loads((root/'submission.json').read_text())
    dataset_version=plan['dataset_version']
    validate_dataset_version(manifest, dataset_version)
    python=sys.executable
    env=dict(os.environ)
    for k in ('STAGE1_ADAPTER','STAGE2_ADAPTER'):
        env.pop(k,None)
    gpu_ids=env.get('CUDA_VISIBLE_DEVICES',','.join(map(str,range(workers)))).split(',')
    if len(gpu_ids)<workers:
        raise ValueError('Fewer visible GPUs than workers')
    def train(stop=None):
        execute([python,'-m','torch.distributed.run','--standalone','--nnodes=1',f'--nproc-per-node={workers}',
                 '-m','stage2.train_full','--architecture','qformer','--dataset-version',dataset_version,
                 '--train-jsonl',root/'release/train.jsonl','--video-path-map',root/'video_map.json',
                 '--release-manifest',root/'release/manifest.json','--output-dir',root/'training',
                 '--model-name','Qwen/Qwen3.5-2B','--epochs',epochs,'--learning-rate','1e-4',
                 '--frame-size','256','--save-steps','100','--holdout-ids',root/'holdout_ids.json',
                 '--holdout-every','1000','--resume']+(['--stop-after-epoch',stop] if stop is not None else []),env)
    if eval_each_epoch:
        reports=[]
        for epoch in range(1,epochs+1):
            if not (root/'training'/f'epoch-{epoch}.completed.json').is_file():
                train(epoch)
            cfg=json.loads((root/'training/training_config.json').read_text())
            if cfg['init_adapter'] is not None or cfg['epochs']!=epochs:
                raise ValueError('Unexpected initialization or training schedule')
            report=evaluate_checkpoint(root,workers,manifest,env,gpu_ids,f'epoch-{epoch}',root/'eval'/f'epoch-{epoch}')
            reports.append(dict(epoch=epoch,records=report['records']))
        (root/'completed.json').write_text(json.dumps(dict(status='completed',epochs=epochs,evaluations=reports),indent=2))
        return
    if not (root/'training/completed.json').exists():
        train()
    cfg=json.loads((root/'training/training_config.json').read_text())
    if cfg['init_adapter'] is not None or cfg['epochs'] != epochs:
        raise ValueError('Unexpected full-training initialization or epoch count')
    report=evaluate_checkpoint(root,workers,manifest,env,gpu_ids)
    (root/'completed.json').write_text(json.dumps(dict(status='completed',epochs=epochs,eval_records=report['records']),indent=2))
    print(json.dumps(report,indent=2),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--run-dir',type=Path,required=True)
    p.add_argument('--workers',type=int,default=4)
    p.add_argument('--epochs',type=int,default=1)
    p.add_argument('--eval-each-epoch',action='store_true')
    a=p.parse_args()
    run(a.run_dir.resolve(),a.workers,a.epochs,a.eval_each_epoch)
