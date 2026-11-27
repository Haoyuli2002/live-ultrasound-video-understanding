"""Freeze full train/holdout inputs and exact update budget; never submit a job."""
import argparse
import collections
import datetime
import json
from pathlib import Path
import shutil
import subprocess
from .data import digest,load_examples
from .full_data import batch_plan,feature_storage_estimate,validate_official_split
from .monitoring import validate_split_separation

ROOT=Path(__file__).resolve().parents[1]


def write(path,data):path.write_text(json.dumps(data,indent=2)+'\n')


def describe(rows):
    groups=collections.defaultdict(list)
    for r in rows:groups[r['task_id']].append(r)
    return dict(videos=len({r['video_id'] for r in rows}),qa=len(groups),endpoints=len(rows),
                answer=sum(r['ready'] for r in rows),wait=sum(not r['ready'] for r in rows),
                available_pairs=sum(sum(r['ready'] for r in g)*sum(not r['ready'] for r in g) for g in groups.values()),
                qa_types=dict(collections.Counter(g[0]['qa_type'] for g in groups.values())))


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--train-source',type=Path,required=True)
    p.add_argument('--holdout-source',type=Path,required=True)
    p.add_argument('--split-manifest',type=Path,required=True)
    p.add_argument('--template-run',type=Path,required=True)
    p.add_argument('--recipe-selection',type=Path,required=True)
    p.add_argument('--run-dir',type=Path,required=True)
    p.add_argument('--epochs',type=int,default=10);p.add_argument('--batch-size',type=int,default=64)
    p.add_argument('--video-block-size', type=int, default=8,
                   help='Target endpoints per shuffled same-video block. Use 0 for legacy order.')
    p.add_argument('--seed',type=int,default=20261121)
    p.add_argument('--gpus',type=int,choices=[1,2,4],default=4)
    args=p.parse_args();run=args.run_dir.resolve()
    if not run.is_relative_to(Path('/dss/mcmlscratch/04/ge75vid2')):raise ValueError('Run must live on scratch')
    if run.exists():raise ValueError('Use a new run directory')
    sources={}
    for split,root,prefix in [('train',args.train_source,'train'),('holdout',args.holdout_source,'holdout')]:
        sources[split]={name:root/file for name,file in [('samples.jsonl',prefix+'_samples.jsonl'),('tasks.jsonl',prefix+'_tasks.jsonl'),('video_map.json','video_map.json')]}
    before={split:{name:digest(path) for name,path in files.items()} for split,files in sources.items()}
    splits={split:load_examples(files['samples.jsonl'],files['tasks.jsonl'],files['video_map.json']) for split,files in sources.items()}
    validate_split_separation(splits['train'],splits['holdout'])
    official=json.loads(args.split_manifest.read_text());validate_official_split(splits['train'],splits['holdout'],official)
    manifest=json.loads((args.train_source/'manifest.json').read_text())
    holdout_report=json.loads((args.holdout_source/'validation_report.json').read_text())
    if not all(manifest.get(k) for k in ['whole_train_scope_complete','source_exports_hash_verified','real_loader_passed','unique_ids','train_split_only']):
        raise ValueError('Train export is not complete/validated')
    if not holdout_report.get('complete') or not holdout_report.get('structural_validation_passed') or holdout_report.get('errors'):
        raise ValueError('Holdout export is not complete/validated')
    excluded=set(official['train'])-{r['video_id'] for r in splits['train']}
    if excluded!=set(manifest['excluded_videos']):raise ValueError('Unaccounted training video omissions')
    for rows in splits.values():
        if any(not Path(r['video_path']).is_file() for r in rows):raise ValueError('Missing video media')
    plan=batch_plan(splits['train'],args.batch_size,args.epochs,args.seed,args.gpus,args.video_block_size)
    run.mkdir(parents=True);(run/'logs').mkdir();(run/'holdout').mkdir()
    for split,files in sources.items():
        dest=run if split=='train' else run/'holdout'
        for name,path in files.items():
            shutil.copy2(path,dest/name)
            if digest(path)!=before[split][name] or digest(dest/name)!=before[split][name]:
                raise ValueError('Data changed during snapshot')
    shutil.copy2(args.split_manifest,run/'official_split_manifest.json')
    write(run/'source_train_manifest.json',manifest);write(run/'source_holdout_validation.json',holdout_report)
    cfg=json.loads((args.template_run/'experiment.json').read_text())
    cfg.update(experiment='Full train171 + holdout16, pending completed readiness recipe selection',
       epochs=args.epochs,seed=args.seed,matched_step_seed=args.seed,batch_candidates=[args.batch_size],
       video_block_size=args.video_block_size,sampler='question_groups',world_size=args.gpus,gradient_accumulation_steps=1,expected_optimizer_steps=plan['total_optimizer_steps'],verify_planned_batches=True,
       eval_epochs=list(range(1,args.epochs+1)),checkpoint_every=1,save_best_holdout=True,
       holdout=dict(directory='holdout',purpose='monitoring and checkpoint selection only; no optimizer/batch probing'),
       evaluation='Independent holdout only, at initialization and every epoch; fixed selected threshold; sparse first-trigger replay, no dense ARS',
       feature_cache_mode='video_mmap',readiness_recipe_selection=str(args.recipe_selection.resolve()),
       readiness_objective='pending_selection',timing_pair_mode='all_pairs',
       init_checkpoint=None,restore_optimizer=False,zero_update_warmup=True,
       profile_warmups=1,profile_repeats=2,profile_real_batches=False,profile_additional_real_batches=True,
       supervision='Resolved from completed four-arm selection before loading model; NTP + selected readiness objective',
       timestamp=datetime.datetime.now(datetime.timezone.utc).isoformat())
    cfg.pop('comparison_initial_manifest',None)
    for key in ['timing_pair_sampling','comparison_protocol']:
        cfg.pop(key,None)
    cfg['timing_pair_sampling']='All within-question WAIT x ANSWER pairs when selected objective contains CL; BCE uses all endpoints.'
    write(run/'experiment.json',cfg)
    write(run/'planned_batch_order.json',plan.pop('batch_order'))
    write(run/'update_budget.json',plan)
    for folder in ['endtoend','StreamReady']:
        for src in (ROOT/folder).rglob('*'):
            if src.is_file() and (src.name=='LICENSE' or src.suffix in {'.py','.yaml','.yml','.sh','.sbatch','.md','.txt'}) and not any(part in {'.git','__pycache__','output','data','experiments'} for part in src.relative_to(ROOT).parts):
                dest=run/'code'/src.relative_to(ROOT);dest.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(src,dest)
    launcher=(args.template_run/'train.sbatch').read_text()
    launcher=launcher.split('\n"$PYTHON" -B -m endtoend.answer_metrics')[0]+'\n'
    launcher=launcher.replace('--job-name=sr-fresh-bce1','--job-name=sr-full-holdout').replace('--time=01:00:00','--time=24:00:00')
    if args.gpus > 1:
        launcher=launcher.replace('--gres=gpu:1',f'--gres=gpu:{args.gpus}').replace('--cpus-per-task=8',f'--cpus-per-task={8*args.gpus}').replace('--mem=96G',f'--mem={96*args.gpus}G')
        launcher=launcher.replace('-m endtoend.gpu_monitor --record',f'-m endtoend.gpu_monitor --all --expected-count {args.gpus} --record')
        launcher=launcher.replace('-B -m endtoend.train',f'-B -m torch.distributed.run --standalone --nnodes=1 --nproc_per_node={args.gpus} --max_restarts=0 -m endtoend.train')
    (run/'train.sbatch').write_text(launcher);subprocess.run(['/bin/bash','-n',str(run/'train.sbatch')],check=True)
    audit=dict(status='prepared_not_submitted_waiting_ablation_selection',sources={s:{n:str(p.resolve()) for n,p in f.items()} for s,f in sources.items()},
       source_sha256=before,official_split_sha256=digest(args.split_manifest),
       train=describe(splits['train']),holdout=describe(splits['holdout']),excluded_train_video_ids=sorted(excluded),
       train_holdout_test_video_overlap=0,canonical_media_path_overlap=0,
       train_features=feature_storage_estimate(splits['train']),holdout_features=feature_storage_estimate(splits['holdout']),
       update_budget=plan,holdout_evaluations=1+args.epochs,holdout_optimizer_steps=0,
       quality='Candidate sampled-reviewed QA; unchanged earlier tasks retain prior review status. Not fully expert-reviewed, not all verified at256px, cross-split near-content duplication not comprehensively audited.',
       test='41 official test video IDs reserved; no new V4 test export found/used; old evaluation datasets not mixed into this run',
       gpu_validation=f'Not run; each of {args.gpus} GPUs must pass BS{args.batch_size} runtime memory probes on full-data longest history before training',
       checkpoint_policy='Scratch checkpoint_latest.pth and checkpoint_best.pth only; best selected by holdout time+text correct, fewer WAIT false triggers, readiness accuracy. Test untouched.')
    write(run/'data_audit.json',audit)
    write(run/'source_hashes.json',{str(p.relative_to(run/'code')):digest(p) for p in sorted((run/'code').rglob('*')) if p.is_file()})
    write(run/'input_hashes.json',{str(p.relative_to(run)):digest(p) for p in [run/'experiment.json',run/'data_audit.json',run/'update_budget.json',run/'planned_batch_order.json',run/'train.sbatch']})
    write(run/'status.json',dict(status='prepared',reason='User requested preparation only; wait for completed ablation selection before training',training_started=False))
    print(json.dumps(dict(run_dir=str(run),train=audit['train'],holdout=audit['holdout'],update_budget=plan),indent=2))


if __name__=='__main__':main()
