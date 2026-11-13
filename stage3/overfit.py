"""Two-video overfit, free reconstruction and streaming evaluation. Explicit mock/Qwen backends."""
from __future__ import annotations
import argparse
from collections import Counter
import gc
import hashlib
import json
import math
from pathlib import Path
import random
import statistics
import torch
from stage2.data import VideoReader
from stage2.reconstruct import text_metrics
from .train import load_examples, causal_memories, decision_question
from .inference import generate
from .metrics import score


def read_rows(p):return [json.loads(s) for s in Path(p).read_text().splitlines() if s.strip()]
def dump(p,x):Path(p).write_text(json.dumps(x,ensure_ascii=False,indent=2)+'\n')
def rows(p,values):Path(p).write_text(''.join(json.dumps(v,ensure_ascii=False)+'\n' for v in values))
def sha(p):
    h=hashlib.sha256()
    with Path(p).open('rb') as f:
        for b in iter(lambda:f.read(1024*1024),b''):h.update(b)
    return h.hexdigest()
def tree_hashes(path):
    root=Path(path)
    return {str(f.relative_to(root)):sha(f) for f in sorted(root.rglob('*')) if f.is_file()}


def verify_lineage(checkpoint, lineage):
    source=Path(checkpoint)/'source_chain.json'
    if not source.is_file():raise ValueError('Checkpoint lacks source_chain.json; use a checkpoint saved by this runner')
    if json.loads(source.read_text())!=lineage:raise ValueError('Stage3 checkpoint source chain differs from supplied memory checkpoint/backend/frame size')


def save_checkpoint(decision,path,lineage):
    decision.save(path)
    dump(Path(path)/'source_chain.json',lineage)


def key(vid,t):return f'{vid}@{t:.6f}'


def validate_dataset(data, video_map):
    tasks=read_rows(data/'tasks.jsonl');train=load_examples(data/'train.jsonl')
    manifest=json.loads((data/'dataset_manifest.json').read_text())
    for name,digest in manifest['source_files_sha256'].items():
        if sha(data/name)!=digest:raise ValueError(f'Dataset changed: {name}')
    if len(train)!=2 or {t['video_id'] for t in tasks}!=set(train):
        raise ValueError('Expected exactly two training videos')
    ids=set();byid={}
    for t in tasks:
        if t['task_id'] in ids or t['split']!='train' or t['language']!='en':
            raise ValueError('Duplicate task, eval contamination or non-English QA')
        ids.add(t['task_id']);byid[t['task_id']]=t
        q=t['query_time'];a=t['answer_time'];end=t['evaluation_horizon']
        if int(q)!=q or int(end)!=end or not 0<q<=end:raise ValueError('Expected integer decision grid')
        if a is not None and (int(a)!=a or not q<=a<=end):raise ValueError('Invalid answer time')
    for vid,points in train.items():
        for r in points:
            t=byid.get(r['task_id'])
            if t is None or (r['video_id'],r['question'],r['query_time'])!=(t['video_id'],t['question'],t['query_time']):
                raise ValueError('Training row does not match canonical task')
            a=t['answer_time'];expected='<WAIT>' if a is None or r['time']<a else '<ANSWER> '+t['reference_answer']
            if r['target']!=expected:raise ValueError('Target differs from canonical task')
            if r['time']>(a if a is not None else t['review_horizon']):raise ValueError('Training past annotated completion')
    for t in tasks:
        actual=[r['time'] for r in train[t['video_id']] if r['task_id']==t['task_id']]
        end=t['answer_time'] if t['answer_time'] is not None else t['review_horizon']
        if actual!=list(range(t['query_time'],end+1)):raise ValueError('Incomplete training decision grid')
    raw=json.loads(Path(video_map).read_text());paths={}
    for v in manifest['videos']:
        p=Path(raw[v['video_id']]).expanduser()
        if not p.is_absolute():p=Path(video_map).resolve().parent/p
        if not p.is_file() or sha(p)!=v['source_sha256']:raise ValueError(f"Wrong source video: {v['video_id']}")
        paths[v['video_id']]=str(p.resolve())
        if any(t['evaluation_horizon']>v['duration'] for t in tasks if t['video_id']==v['video_id']):raise ValueError('Horizon exceeds video')
    return tasks,train,paths


def cache_memories(memory,tasks,paths,frame_size):
    cache={}
    for vid in paths:
        points=sorted({t for task in tasks if task['video_id']==vid
                       for t in range(task['query_time'],task['evaluation_horizon']+1)})
        reader=VideoReader(paths[vid],frame_size)
        try:
            for row,state in causal_memories(memory,reader,[{'time':float(t)} for t in points]):
                cache[key(vid,row['time'])]=state.detach().cpu().clone()
        finally:reader.close()
    return cache


def evaluate(decision,tasks,training,cache,destination,*,condition,threshold,max_new_tokens):
    destination.mkdir(parents=True,exist_ok=False);decision.eval();predictions=[];recon=[];losses=[]
    def state(vid,t):
        m=cache[key(vid,t)].to(decision.device)
        return torch.zeros_like(m) if condition=='zero' else m
    with torch.inference_mode():
        for points in training.values():
            for row in points:
                bce,ntp,_=decision.forward_answerability(state(row['video_id'],row['time']),decision_question(row),row['target'],row['y'])
                losses.append({'task_id':row['task_id'],'time':row['time'],'bce':float(bce),'ntp':float(ntp)})
        for task in tasks:
            delivered=False
            for t in range(task['query_time'],task['evaluation_horizon']+1):
                row={'time':float(t),'query_time':task['query_time'],'question':task['question']}
                m=state(task['video_id'],t);question=decision_question(row)
                probability=float(torch.sigmoid(decision.decision_logits(m,question)).item())
                if not math.isfinite(probability):raise RuntimeError('Non-finite classifier output')
                status='ANSWER' if probability>=threshold else 'WAIT'
                pred={'task_id':task['task_id'],'video_id':task['video_id'],'time':t,
                      'probability':probability,'status':status,'prediction':'',
                      'delivered':status=='ANSWER' and not delivered,
                      'after_completion_probe':delivered}
                if status=='ANSWER' and not delivered:
                    # Recomputed gate is deterministic in eval mode; no gold marker is supplied.
                    pred.update(generate(decision,m,question,threshold=threshold,max_new_tokens=max_new_tokens))
                    delivered=True
                predictions.append(pred)
            if task['answer_time'] is not None:
                t=task['answer_time'];row={'time':t,'query_time':task['query_time'],'question':task['question']}
                decoded=generate(decision,state(task['video_id'],t),decision_question(row),
                                 threshold=threshold,max_new_tokens=max_new_tokens,force_status='ANSWER')
                recon.append({'task_id':task['task_id'],'time':t,'reference':task['reference_answer'],
                              **decoded,**text_metrics(decoded['prediction'],task['reference_answer'])})
    report=score(tasks,predictions)
    report.update(condition=condition,threshold=threshold,
        teacher_forced={'bce':statistics.mean(x['bce'] for x in losses),'ntp':statistics.mean(x['ntp'] for x in losses)},
        forced_answer_reconstruction={'tasks':len(recon),'word_f1':statistics.mean(x['word_overlap_f1'] for x in recon) if recon else None,
        'truncated':sum(x['stop_reason']=='max_new_tokens' for x in recon),
        'interpretation':'At gold answer time with ANSWER forced. Free text, no reference in prefix; this isolates the decoder and is not end-to-end triggering accuracy.'})
    rows(destination/'predictions.jsonl',predictions);rows(destination/'reconstruction.jsonl',recon)
    rows(destination/'teacher_forced.jsonl',losses);dump(destination/'report.json',report)
    with (destination/'review.txt').open('w') as f:
        for item in recon:f.write(f"{item['task_id']} @ {item['time']}s\nREF: {item['reference']}\nGEN: {item['prediction']}\nSTOP: {item['stop_reason']}\n\n")
    return report


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--data-dir',type=Path,required=True);p.add_argument('--video-path-map',type=Path,required=True)
    p.add_argument('--output-dir',type=Path,required=True);p.add_argument('--backend',choices=['mock','qwen'],required=True)
    p.add_argument('--stage2-adapter',type=Path,required=True);p.add_argument('--stage1-adapter')
    p.add_argument('--stage2-training-config',type=Path,help='Required for Qwen: exact Stage2 training_config.json')
    p.add_argument('--model-name',default='Qwen/Qwen3.5-2B');p.add_argument('--epochs',type=int,default=20)
    p.add_argument('--max-steps',type=int,default=-1);p.add_argument('--learning-rate',type=float,default=1e-4)
    p.add_argument('--threshold',type=float,default=.5);p.add_argument('--frame-size',type=int,default=224)
    p.add_argument('--max-new-tokens',type=int,default=128);p.add_argument('--seed',type=int,default=42)
    p.add_argument('--eval-only',type=Path,help='Reload a saved Stage3 checkpoint; no optimization')
    args=p.parse_args()
    if args.epochs<1 or args.max_steps==0 or args.max_steps < -1 or args.frame_size<1 or args.learning_rate<=0 or not 0<args.threshold<1 or args.max_new_tokens<1:raise ValueError('Invalid run settings')
    if args.output_dir.exists():raise FileExistsError('Use a fresh output directory')
    tasks,training,paths=validate_dataset(args.data_dir,args.video_path_map)
    random.seed(args.seed);torch.manual_seed(args.seed)
    checkpoint_hashes=tree_hashes(args.stage2_adapter)
    if not checkpoint_hashes:raise ValueError('Stage2 checkpoint is empty or missing')
    lineage={'backend':args.backend,'model_name':args.model_name,'frame_size':args.frame_size,
             'stage2_checkpoint_hashes':checkpoint_hashes,
             'stage1_checkpoint_hashes':tree_hashes(args.stage1_adapter) if args.stage1_adapter else None}
    if args.stage1_adapter and not lineage['stage1_checkpoint_hashes']:raise ValueError('Empty Stage1 checkpoint')
    if args.eval_only:verify_lineage(args.eval_only,lineage)
    if args.backend=='mock':
        if args.stage1_adapter:raise ValueError('Mock backend does not accept real adapters')
        from .mock import MockMemory,MockDecision
        torch.set_num_threads(1);memory=MockMemory(args.stage2_adapter)
        factory=lambda checkpoint=None:MockDecision(checkpoint)
        hardware={'device':'cpu','mock':True,'torch':torch.__version__}
    else:
        if (args.stage2_adapter/'mock_config.json').exists():raise ValueError('Mock checkpoint cannot initialize Qwen')
        if args.stage2_training_config is None:raise ValueError('Supply exact Stage2 training config to verify initialization chain')
        cfg=json.loads(args.stage2_training_config.read_text())
        if cfg.get('model_name')!=args.model_name or cfg.get('frame_size')!=args.frame_size:raise ValueError('Stage2 model/frame size mismatch')
        base=cfg.get('init_adapter')
        if bool(base)!=bool(args.stage1_adapter):raise ValueError('Stage1 adapter presence differs from Stage2 training')
        if base and Path(base).resolve()!=Path(args.stage1_adapter).resolve():raise ValueError('Use the exact Stage1 checkpoint from Stage2 training')
        from scripts.gpu_runtime import cuda_info
        from .model import load_frozen_memory_model,DecisionModel
        hardware=cuda_info()
        memory=load_frozen_memory_model(args.model_name,stage1_adapter=args.stage1_adapter,stage2_adapter=str(args.stage2_adapter))
        memory.model.to('cuda').eval()
        factory=lambda checkpoint=None:DecisionModel(args.model_name,stage1_adapter=args.stage1_adapter,
            stage2_adapter=str(args.stage2_adapter),decision_adapter=str(checkpoint) if checkpoint else None).to('cuda')
    args.output_dir.mkdir(parents=True)
    dump(args.output_dir/'run_config.json',{**{k:str(v) if isinstance(v,Path) else v for k,v in vars(args).items()},
        'hardware':hardware,'source_chain':lineage,'code_sha256':{f.name:sha(f) for f in Path(__file__).parent.glob('*.py')},'stage2_checkpoint_hashes':checkpoint_hashes,'video_sha256':{v:sha(f) for v,f in paths.items()},
        'data_sha256':{n:sha(args.data_dir/n) for n in ['train.jsonl','tasks.jsonl','dataset_manifest.json']},
        'interpretation':'MOCK INTERFACE TEST ONLY' if args.backend=='mock' else 'Same-video overfit; not held-out performance'})
    print('Computing causal frozen memory once per video...',flush=True)
    cache=cache_memories(memory,tasks,paths,args.frame_size)
    torch.save(cache,args.output_dir/'memory_cache.pt')
    dump(args.output_dir/'memory_cache_manifest.json',{'backend':args.backend,'source':'run_config.json',
        'state_count':len(cache),'order':'long then residual short','detached':True,
        'snapshots':{k:list(v.shape) for k,v in cache.items()},'sha256':sha(args.output_dir/'memory_cache.pt')})
    del memory;gc.collect()
    if args.backend=='qwen':torch.cuda.empty_cache()
    decision=factory(args.eval_only)
    if hasattr(decision.model,'config'):decision.model.config.use_cache=False
    if args.eval_only:
        for condition in ['real','zero']:
            evaluate(decision,tasks,training,cache,args.output_dir/('eval_'+condition),condition=condition,threshold=args.threshold,max_new_tokens=args.max_new_tokens)
        return
    save_checkpoint(decision,args.output_dir/'initial',lineage)
    reports={'initial_real':evaluate(decision,tasks,training,cache,args.output_dir/'initial_real',condition='real',threshold=args.threshold,max_new_tokens=args.max_new_tokens)}
    if args.backend=='qwen':decision.model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={'use_reentrant':False})
    decision.train();parameters=[x for x in decision.parameters() if x.requires_grad]
    optimizer=torch.optim.AdamW(parameters,lr=args.learning_rate)
    points=[x for values in training.values() for x in values];counts=Counter(x['y'] for x in points)
    weights=[1/counts[x['y']] for x in points];step=0
    with (args.output_dir/'training_metrics.jsonl').open('w') as f:
        for epoch in range(args.epochs):
            for row in random.choices(points,weights=weights,k=len(points)):
                optimizer.zero_grad(set_to_none=True)
                m=cache[key(row['video_id'],row['time'])].to(decision.device).detach()
                bce,ntp,logit=decision.forward_answerability(m,decision_question(row),row['target'],row['y'])
                loss=bce+ntp
                if not torch.isfinite(loss):raise RuntimeError('Non-finite training loss')
                loss.backward();norm=torch.nn.utils.clip_grad_norm_(parameters,1.);optimizer.step();step+=1
                f.write(json.dumps({'step':step,'epoch':epoch,'task_id':row['task_id'],'time':row['time'],'bce':float(bce.detach()),'ntp':float(ntp.detach()),'loss':float(loss.detach()),'grad_norm':float(norm)})+'\n');f.flush()
                if args.max_steps>0 and step>=args.max_steps:break
            print(f'Epoch {epoch+1}, updates {step}',flush=True)
            if epoch==0:save_checkpoint(decision,args.output_dir/'epoch-1',lineage)
            if args.max_steps>0 and step>=args.max_steps:break
    save_checkpoint(decision,args.output_dir/'final',lineage)
    del optimizer,decision,parameters;gc.collect()
    if args.backend=='qwen':torch.cuda.empty_cache()
    verify_lineage(args.output_dir/'final',lineage)
    decision=factory(args.output_dir/'final')
    if hasattr(decision.model,'config'):decision.model.config.use_cache=False
    for condition in ['real','zero']:
        reports['final_'+condition]=evaluate(decision,tasks,training,cache,args.output_dir/('final_'+condition),condition=condition,threshold=args.threshold,max_new_tokens=args.max_new_tokens)
    dump(args.output_dir/'overfit_report.json',{'backend':args.backend,'optimizer_updates':step,'tasks':len(tasks),
        'evaluation':'MOCK WIRING ONLY; not Stage3 quality' if args.backend=='mock' else 'Same-video overfit only; no held-out claim',
        'checkpoints':reports,'limitations':['No optimizer-state resume or DDP','Threshold fixed before evaluation','Zero memory retains token count and clocks; it is a limited diagnostic, not proof of visual dependence','Full-prefix greedy decoding without KV cache; slower but avoids hybrid-cache ambiguity']})
    print('Saved reconstruction, streaming decisions and reports.',flush=True)


if __name__=='__main__':main()
