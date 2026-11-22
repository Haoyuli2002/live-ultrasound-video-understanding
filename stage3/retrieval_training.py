"""Train 512-d retrieval and evidence-grounded QA from Stage3 evidence views.

Default rejects draft data. --allow-draft is only for explicitly requested
wiring experiments, never an endorsement of annotation quality.
"""
import argparse
from collections import defaultdict
import hashlib
import json
import math
from pathlib import Path

import torch
from torch.nn import functional as F
from .retrieval import EvidenceBank, group_retrieval_loss
from .evidence import materialize


def read_rows(path):
    return [json.loads(s) for s in path.read_text().splitlines() if s.strip()]


def load_dataset(root, *, allow_draft=False):
    root=Path(root)
    tasks_list=read_rows(root/'tasks.jsonl'); evidence_list=read_rows(root/'evidence.jsonl')
    tasks={t['task_id']:t for t in tasks_list}; evidence={e['evidence_id']:e for e in evidence_list}
    if not tasks or len(tasks)!=len(tasks_list) or len(evidence)!=len(evidence_list):
        raise ValueError('Missing/duplicate task or evidence IDs')
    if len({t['video_id'] for t in tasks.values()})!=1:
        raise ValueError('Each evidence bundle must contain one video; use separate bundles per video')
    if any(t.get('split')!='train' for t in tasks.values()):
        raise ValueError('Only train-split bundles may be used for training')
    if not allow_draft and any(t.get('review',{}).get('training_ready') is not True for t in tasks.values()):
        raise ValueError('Draft evidence data: explicitly use --allow-draft for a debug run')
    for e in evidence.values():
        start,end=float(e['start']),float(e['end']);anchor=float(e['anchor_frame']['decision_time'])
        pts=float(e['anchor_frame']['approx_pts'])
        if not all(math.isfinite(t) for t in (start,end,anchor,pts)) or not 0<=start<=pts<end<=anchor:
            raise ValueError('Invalid evidence anchor/interval')
    rows={}
    def item(tid,current):
        task=tasks[tid];current=float(current)
        query=float(task['query_time'])
        if not task['question'].strip() or not math.isfinite(query) or not math.isfinite(current) or not 0<=query<=current or current<=0:
            raise ValueError('Invalid causal question/time')
        return rows.setdefault((tid,current),{'task_id':tid,'video_id':task['video_id'],'question':task['question'],
            'query_time':query,'time':current,'decision':None,'retrieval':None,'answers':[]})
    def view(name):
        result=read_rows(root/name)
        if not allow_draft and any(x.get('training_ready') is not True for x in result):
            raise ValueError('Draft supervision view; review before training')
        return result
    for x in view('decision_view.jsonl'):
        r=item(x['task_id'],x['input']['current_time'])
        if x['input']['question']!=r['question'] or x['input']['video_id']!=r['video_id'] or x['input']['query_time']!=r['query_time']:
            raise ValueError('Decision input disagrees with task')
        sup=x['supervision']; y=sup['prefix_sufficient']
        if sup['state_loss_mask'] not in (0,1) or (sup['state_loss_mask'] and type(y) is not bool):
            raise ValueError('Uncertain decision must be loss-masked')
        if sup['state_loss_mask']:
            if r['decision'] is not None:raise ValueError('Duplicate decision label')
            r['decision']=float(y)
    for x in view('retrieval_view.jsonl'):
        r=item(x['task_id'],x['current_time'])
        if x['question']!=r['question'] or r['retrieval'] is not None:raise ValueError('Invalid retrieval row')
        candidates=x['candidates']; ids=[c['evidence_id'] for c in candidates]
        if len(set(ids))!=len(ids) or not ids:raise ValueError('Invalid candidate IDs')
        for c in candidates:
            e=evidence[c['evidence_id']]
            if c['start']!=e['start'] or c['end']!=e['end'] or e['anchor_frame']['decision_time']>r['time']:
                raise ValueError('Unavailable retrieval candidate')
            if c['relevance'] not in (0,1,None):raise ValueError('Invalid relevance label')
        groups=[g['any_of'] for g in x['required_groups'] if g.get('required',True)]
        if x['required_groups']!=tasks[x['task_id']]['evidence_groups']:
            raise ValueError('Retrieval groups disagree with task')
        positive={c['evidence_id'] for c in candidates if c['relevance']==1}
        if not groups or any(not g or not set(g)<=positive for g in groups):
            raise ValueError('Required group missing positive candidates')
        r['retrieval']={'ids':ids,'groups':[[ids.index(e) for e in g] for g in groups],
                        'negatives':[i for i,c in enumerate(candidates) if c['relevance']==0]}
    import re
    for x in view('answer_view.jsonl'):
        r=item(x['task_id'],x['input']['current_time'])
        if x['input']['question']!=r['question']:raise ValueError('Answer question disagrees with task')
        selected=x['input']['evidence'];ids=[e['evidence_id'] for e in selected]
        if len(ids)!=len(set(ids)):raise ValueError('Duplicate answer evidence')
        for entry in selected:
            e=evidence[entry['evidence_id']]
            if entry['start']!=e['start'] or entry['end']!=e['end'] or e['anchor_frame']['decision_time']>r['time']:
                raise ValueError('Future/mismatched answer evidence')
        s=x['supervision'];sufficient=s['retrieved_evidence_sufficient'];target=s['target_text']
        groups=tasks[x['task_id']]['evidence_groups']
        covered=bool(groups) and all(bool(set(g['any_of'])&set(ids)) for g in groups if g.get('required',True))
        if type(sufficient) is not bool or sufficient!=covered:
            raise ValueError('Evidence sufficiency disagrees with required groups')
        if not target.startswith('<ANSWER>' if sufficient else '<WAIT>'):
            raise ValueError('Answer marker disagrees with evidence sufficiency')
        citations=set(re.findall(r'\[([A-Za-z0-9]+)\]',target))
        if not citations<=set(ids) or (sufficient and not citations):raise ValueError('Invalid answer citations')
        if sufficient and any(not set(g['any_of']) & citations for g in groups if g.get('required',True)):
            raise ValueError('Answer citations omit a required evidence group')
        r['answers'].append({'ids':ids,'target':target,'sufficient':sufficient})
    active=[r for r in rows.values() if r['decision'] is not None or r['retrieval'] or r['answers']]
    if not active:raise ValueError('No supervised examples')
    return sorted(active,key=lambda r:(r['time'],r['task_id'])),evidence


def checkpoint_frame_size(path):
    path=Path(path)
    for cfgpath in [path/'training_config.json',path.parent/'training_config.json']:
        if cfgpath.exists():
            cfg=json.loads(cfgpath.read_text())
            if 'frame_size' in cfg:return int(cfg['frame_size'])
    raise ValueError('Stage2 checkpoint must have training_config.json with frame_size; do not guess preprocessing')


def place_memory(memory):
    if hasattr(memory,'to'):memory.to('cuda').eval()
    else:memory.model.to('cuda').eval()


def annotated_windows(ids, evidence, bank, current_time):
    windows=[]
    for eid in ids:
        e=evidence[eid];r=bank.get(float(e['anchor_frame']['decision_time']),current_time)
        if r.frame_index!=e['anchor_frame']['frame_index'] or abs(r.source_pts-e['anchor_frame']['approx_pts'])>1e-4:
            raise ValueError('Annotated evidence does not match video sampling')
        windows.append({'evidence_id':eid,'start':e['start'],'end':e['end'],'frames':[r]})
    return sorted(windows,key=lambda w:w['start'])


def train_bundle(decision,memory,reader,points,evidence,bank,optimizer,metrics,*,phase='joint',epoch=0):
    from .train import causal_memories,decision_question
    if phase not in ('retrieval','joint'):raise ValueError('Unknown training phase')
    # Ensure fractional annotated anchors exist without inserting them into
    # persistent integer-second Q-Former history.
    replay=list(points)
    replay.extend({'time':float(e['anchor_frame']['decision_time']),'anchor_only':True}
                  for e in evidence.values() if e['anchor_frame']['decision_time']<=max(r['time'] for r in points))
    replay.sort(key=lambda r:r['time'])
    count=0
    for row,state in causal_memories(memory,reader,replay,evidence_bank=bank):
        if row.get('anchor_only'):continue
        if phase=='retrieval' and (not row['retrieval'] or not row['retrieval']['negatives']):continue
        optimizer.zero_grad(set_to_none=True)
        question=decision_question(row)
        hidden=decision.decision_state(state,question)
        losses={}
        if phase=='joint' and row['decision'] is not None:
            logit=decision.decision_head(hidden.to(decision.decision_head.weight)).reshape(-1)
            losses['decision']=F.binary_cross_entropy_with_logits(logit,torch.full_like(logit,row['decision']))
        if row['retrieval']:
            target=row['retrieval']
            annotated_windows(target['ids'],evidence,bank,row['time'])
            records=[bank.get(float(evidence[e]['anchor_frame']['decision_time']),row['time']) for e in target['ids']]
            scores=decision.retrieval_head(hidden,torch.stack([r.pooled for r in records]))[0]
            losses['retrieval']=group_retrieval_loss(scores,target['groups'],target['negatives'])
        if phase=='joint' and row['answers']:
            suff=[];text=[]
            for answer in row['answers']:
                windows=annotated_windows(answer['ids'],evidence,bank,row['time'])
                inputs=materialize(memory,reader,bank,windows,row['time'])
                a,b=decision.forward_evidence(question,inputs,answer['target'],answer['sufficient'])
                suff.append(a);text.append(b)
            losses['evidence']=torch.stack(suff).mean();losses['text']=torch.stack(text).mean()
        if not losses:continue
        loss=sum(losses.values())
        if not torch.isfinite(loss):raise RuntimeError('Non-finite training loss')
        loss.backward()
        norm=torch.nn.utils.clip_grad_norm_([p for p in decision.parameters() if p.requires_grad],1.,error_if_nonfinite=True)
        optimizer.step();count+=1
        if phase=="joint" and row["answers"]:decision.evidence_trained=True
        metrics.write(json.dumps({'epoch':epoch,'task_id':row['task_id'],'time':row['time'],
            **{k:float(v.detach()) for k,v in losses.items()},'grad_norm':float(norm)})+'\n');metrics.flush()
    return count


def main():
    from stage2.data import VideoReader
    from .model import DecisionModel,load_frozen_memory_model
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--dataset-dir',type=Path,required=True)
    p.add_argument('--video-path-map',type=Path,required=True)
    p.add_argument('--stage2-adapter',required=True)
    p.add_argument('--stage1-adapter')
    p.add_argument('--model-name',default='Qwen/Qwen3.5-2B')
    p.add_argument('--init-decision-adapter')
    p.add_argument('--output-dir',type=Path,required=True)
    p.add_argument('--phase',choices=['retrieval','joint'],default='retrieval')
    p.add_argument('--epochs',type=int,default=1)
    p.add_argument('--learning-rate',type=float,default=1e-4)
    p.add_argument('--allow-draft',action='store_true')
    p.add_argument('--seed',type=int,default=42)
    a=p.parse_args()
    if a.epochs<1 or not math.isfinite(a.learning_rate) or a.learning_rate<=0:raise ValueError('Invalid training configuration')
    points,evidence=load_dataset(a.dataset_dir,allow_draft=a.allow_draft)
    paths=json.loads(a.video_path_map.read_text());video=Path(paths[points[0]['video_id']]).resolve()
    if not video.is_file():raise FileNotFoundError(video)
    source_manifest=json.loads((a.dataset_dir/'media_manifest.json').read_text())
    if source_manifest['video_id']!=points[0]['video_id']:
        raise ValueError('Video source identity disagrees with annotations')
    digest=hashlib.sha256()
    with video.open('rb') as f:
        for chunk in iter(lambda:f.read(8*1024*1024),b''):digest.update(chunk)
    if digest.hexdigest()!=source_manifest['sha256']:
        raise ValueError('Video hash differs from annotated source')
    frame_size=checkpoint_frame_size(a.stage2_adapter)
    if a.output_dir.exists():raise FileExistsError(a.output_dir)
    if a.init_decision_adapter:
        cfg=json.loads((Path(a.init_decision_adapter)/'stage3_config.json').read_text())
        if cfg['model_name']!=a.model_name or Path(cfg['stage2_adapter']).resolve()!=Path(a.stage2_adapter).resolve() or cfg['stage1_adapter']!=a.stage1_adapter:
            raise ValueError('Decision checkpoint uses a different Stage2 chain')
    torch.manual_seed(a.seed)
    memory=load_frozen_memory_model(a.model_name,stage1_adapter=a.stage1_adapter,stage2_adapter=a.stage2_adapter)
    place_memory(memory)
    decision=DecisionModel(a.model_name,stage1_adapter=a.stage1_adapter,stage2_adapter=a.stage2_adapter,
        decision_adapter=a.init_decision_adapter,adapter_trainable=True,enable_retrieval=True).to('cuda')
    if a.phase=='retrieval':
        decision.requires_grad_(False);decision.retrieval_head.requires_grad_(True)
        decision.eval()  # deterministic frozen decision features; projection still differentiable
    else:
        decision.train()
        decision.model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={'use_reentrant':False})
    optimizer=torch.optim.AdamW([p for p in decision.parameters() if p.requires_grad],lr=a.learning_rate)
    a.output_dir.mkdir(parents=True)
    config={k:str(v) if isinstance(v,Path) else v for k,v in vars(a).items()}
    config.update(frame_size=frame_size,retrieval_dim=512,video_path=str(video),video_sha256=digest.hexdigest(),
        data_sha256={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in a.dataset_dir.glob('*.jsonl')})
    (a.output_dir/'training_config.json').write_text(json.dumps(config,indent=2)+'\n')
    with (a.output_dir/'metrics.jsonl').open('w') as metrics:
        for epoch in range(a.epochs):
            reader=VideoReader(video,frame_size);bank=EvidenceBank(video)
            try:
                count=train_bundle(decision,memory,reader,points,evidence,bank,optimizer,metrics,phase=a.phase,epoch=epoch)
            finally:reader.close()
            if not count:raise ValueError('No optimizer steps; retrieval phase requires explicit negative candidates')
            decision.save(a.output_dir/f'epoch-{epoch+1}')
    decision.save(a.output_dir/'final')


if __name__=='__main__':main()
