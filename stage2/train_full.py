"""Paired full-data memory training with independent targets, DDP and resume.

Current-window states are re-encoded for every target. Previous-minute Long
is detached recurrent state, retained across updates (truncated BPTT), not an
expensive current-weight replay of the entire history. Both architectures use
this same optimizer schedule. This differs from the small overfit diagnostic.
"""
from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path
import random
import shutil

import torch
from torch import nn
from torch.nn.parallel import DistributedDataParallel

from .data import VideoReader, load_blocks
from .qformer import load_memory_model
from .train_overfit_ab import encode_interval
from .provenance import digest


def tasks_for_video(video_id, blocks):
    result=[]
    for index,row in enumerate(blocks):
        start,end=map(lambda x:int(round(x)),row['block_window'])
        for j,local in enumerate(row['local_sub_summaries']):
            result.append(dict(video_id=video_id,block=index,kind='short',start=start+10*j,
                               end=start+10*j+10,target=local['local_summary_target']))
        if end-start==60:
            for kind in ['block','global']:
                result.append(dict(video_id=video_id,block=index,kind=kind,start=start,end=end,
                                   target=row[f'{kind}_summary_target']))
    return result


def schedules(videos, world_size, seed, epoch):
    """Whole videos remain on one rank; pad short ranks rather than repeat data."""
    names=sorted(videos)
    random.Random(seed+epoch).shuffle(names)
    counts={v:len(tasks_for_video(v,videos[v])) for v in names}
    names.sort(key=lambda v:counts[v],reverse=True)
    batches=[[] for _ in range(world_size)]
    sizes=[0]*world_size
    for vid in names:
        rank=min(range(world_size),key=lambda r:(sizes[r],r))
        batches[rank].extend(tasks_for_video(vid,videos[vid]))
        sizes[rank]+=counts[vid]
    return batches


class Objective(nn.Module):
    def __init__(self,memory):
        super().__init__()
        self.memory=memory
        self.backbone=memory.model
        if hasattr(memory,'compressors'):
            self.compressors=memory.compressors

    def forward(self, task, reader=None, previous=None):
        if task is None:
            # Padding participates in the same DDP collectives but creates no
            # sample, no memory transition and no duplicate supervision.
            p=next(p for p in self.parameters() if p.requires_grad)
            return p.reshape(-1)[0]*0,None
        short=encode_interval(self.memory,reader,task['start'],task['end'])
        tokens=(self.memory.update_long(previous,short) if task['kind']=='global' else short)
        loss=self.memory.reconstruction_loss(tokens,task['target'],task['kind'])
        return loss,tokens.detach() if task['kind']=='global' else None


def atomic_json(path,value):
    path=Path(path);temporary=path.with_suffix('.tmp')
    temporary.write_text(json.dumps(value,indent=2));temporary.replace(path)


def checkpoint(output,memory,optimizer,state,rank,world_size):
    temporary=output/'checkpoint-writing'
    if rank==0:
        if temporary.exists():shutil.rmtree(temporary)
        temporary.mkdir()
    if world_size>1:torch.distributed.barrier()
    torch.save(state,temporary/f'rank-{rank}.pt')
    if rank==0:
        memory.save(str(temporary/'adapter'))
        torch.save(optimizer.state_dict(),temporary/'optimizer.pt')
    if world_size>1:torch.distributed.barrier()
    if rank==0:
        latest=output/'checkpoint-latest';old=output/'checkpoint-previous'
        if old.exists():shutil.rmtree(old)
        if latest.exists():latest.rename(old)
        temporary.rename(latest)
    if world_size>1:torch.distributed.barrier()


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--train-jsonl',required=True);p.add_argument('--video-path-map',required=True)
    p.add_argument('--release-manifest',required=True);p.add_argument('--output-dir',required=True)
    p.add_argument('--architecture',choices=['ordinary','qformer'],required=True)
    p.add_argument('--init-adapter',required=True)
    p.add_argument('--model-name',default='Qwen/Qwen3.5-2B')
    p.add_argument('--epochs',type=int,default=1);p.add_argument('--learning-rate',type=float,default=1e-4)
    p.add_argument('--frame-size',type=int,default=224);p.add_argument('--seed',type=int,default=20261104)
    p.add_argument('--save-steps',type=int,default=100);p.add_argument('--resume',action='store_true')
    a=p.parse_args()
    if min(a.epochs,a.frame_size,a.save_steps)<1 or not math.isfinite(a.learning_rate) or a.learning_rate<=0:
        p.error('Invalid positive training settings')
    if not torch.cuda.is_available():raise RuntimeError('CUDA GPU required')
    rank=int(os.environ.get('RANK',0));world=int(os.environ.get('WORLD_SIZE',1))
    torch.cuda.set_device(int(os.environ.get('LOCAL_RANK',0)))
    if world>1:torch.distributed.init_process_group('nccl')
    try:
        run(a,rank,world)
    finally:
        if torch.distributed.is_initialized():torch.distributed.destroy_process_group()


def run(a,rank,world):
    from .release import validate_release
    release=validate_release(a.release_manifest)
    if digest(a.train_jsonl)!=release['files']['train.jsonl']:
        raise ValueError('Training input is not the frozen TRAIN split')
    videos=load_blocks(a.train_jsonl,validate_text=False)  # scoped visual acceptance supersedes old length gate
    paths=json.loads(Path(a.video_path_map).read_text())
    if set(videos)!=set(release['splits']['train']):raise ValueError('Training videos differ from release')
    if any(v not in paths or not Path(paths[v]).is_file() for v in videos):raise FileNotFoundError('Training video missing')
    expected_sources={v['video_id']:v['source_sha256'] for v in release['videos']}
    own={t['video_id'] for t in schedules(videos,world,a.seed,0)[rank]}
    changed=[v for v in own if digest(paths[v])!=expected_sources[v]]
    valid=torch.tensor(int(not changed),device='cuda')
    if world>1:torch.distributed.all_reduce(valid,op=torch.distributed.ReduceOp.MIN)
    if not valid.item():raise ValueError(f'Source changed after submission; local mismatches: {changed}')
    torch.manual_seed(a.seed);torch.cuda.manual_seed_all(a.seed);random.seed(a.seed)
    output=Path(a.output_dir);output.mkdir(parents=True,exist_ok=True)
    invariant={k:getattr(a,k) for k in ['architecture','model_name','epochs','learning_rate','frame_size','seed']}
    invariant.update(world_size=world,release_sha256=digest(a.release_manifest),
                     map_sha256=digest(a.video_path_map),init_adapter=str(Path(a.init_adapter).resolve()),
                     init_adapter_sha256=digest(Path(a.init_adapter)/'adapter_model.safetensors'))
    config=output/'training_config.json'
    latest=output/'checkpoint-latest'
    if a.resume:
        # Recover the rename gap if Slurm killed rank 0 after moving latest to previous.
        if rank==0 and not latest.exists() and (output/'checkpoint-previous').is_dir():
            (output/'checkpoint-previous').rename(latest)
        if world>1:torch.distributed.barrier()
    if a.resume and latest.is_dir():
        saved=json.loads(config.read_text())
        if saved['resume_invariants']!=invariant:raise ValueError('Resume topology, data or initialization changed')
        adapter=str(latest/'adapter')
    else:
        if (output/'training_metrics.rank0.jsonl').exists() or latest.exists():
            raise FileExistsError('Existing training run: use --resume or a new directory')
        adapter=None
    memory=load_memory_model(a.model_name,architecture=a.architecture,base_adapter=a.init_adapter,
                             adapter_path=adapter,adapter_trainable=bool(adapter),dtype=torch.bfloat16)
    if a.architecture=='qformer':memory.to('cuda').train()
    else:memory.model.to('cuda').train()
    memory.model.config.use_cache=False
    memory.model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={'use_reentrant':False})
    objective=Objective(memory)
    optimizer=torch.optim.AdamW((p for p in objective.parameters() if p.requires_grad),lr=a.learning_rate)
    if world>1:objective=DistributedDataParallel(objective,device_ids=[torch.cuda.current_device()],
                                               find_unused_parameters=True,broadcast_buffers=False)
    if rank==0:
        atomic_json(config,{**vars(a),'init_adapter':a.init_adapter,'memory_architecture':a.architecture,
                            'resume_invariants':invariant,'update_schedule':'one target per rank per step',
                            'historical_memory':'detached recurrent state; no full-prefix weight replay'})
    epoch=step=global_step=0;previous=None;current_video=None
    metrics=output/f'training_metrics.rank{rank}.jsonl'
    if adapter:
        state=torch.load(latest/f'rank-{rank}.pt',map_location='cpu',weights_only=False)
        optimizer.load_state_dict(torch.load(latest/'optimizer.pt',map_location='cpu',weights_only=False))
        # Optimizer.load_state_dict casts states to each parameter's device.
        epoch,step,global_step=(state[k] for k in ['epoch','next_step','global_step'])
        previous=state['previous'];current_video=state['current_video']
        if previous is not None:previous=previous.to('cuda')
        torch.set_rng_state(state['torch_rng']);torch.cuda.set_rng_state(state['cuda_rng'])
        random.setstate(state['python_rng'])
        with metrics.open('r+b') as stream:stream.truncate(state['metrics_bytes'])
    elif metrics.exists():raise FileExistsError(metrics)
    reader=None
    with metrics.open('a') as stream:
        try:
            while epoch<a.epochs:
                assignments=schedules(videos,world,a.seed,epoch);local=assignments[rank]
                total=max(map(len,assignments))
                while step<total:
                    task=local[step] if step<len(local) else None
                    if task and (reader is None or task['video_id']!=current_video):
                        if reader:reader.close()
                        if task['video_id']!=current_video:previous=None
                        current_video=task['video_id'];reader=VideoReader(paths[current_video],a.frame_size)
                    optimizer.zero_grad(set_to_none=True)
                    active=sum(step<len(x) for x in assignments)
                    loss,new_long=objective(task,reader,previous)
                    valid=torch.tensor(int(torch.isfinite(loss.detach()).item()),device='cuda')
                    if world>1:torch.distributed.all_reduce(valid,op=torch.distributed.ReduceOp.MIN)
                    if not valid.item():raise FloatingPointError('A rank produced nonfinite loss')
                    (loss*world/active).backward()
                    optimizer.step()
                    if new_long is not None:previous=new_long
                    if task:
                        row={**task,'epoch':epoch,'global_step':global_step+1,'rank':rank,
                             'loss':float(loss.detach()),'architecture':a.architecture}
                        stream.write(json.dumps(row)+'\n');stream.flush()
                        if step%10==0:print(json.dumps({k:row[k] for k in ['epoch','global_step','rank','kind','loss']}),flush=True)
                    step+=1;global_step+=1
                    if global_step%a.save_steps==0:
                        checkpoint(output,memory,optimizer,dict(epoch=epoch,next_step=step,global_step=global_step,
                            previous=None if previous is None else previous.cpu(),current_video=current_video,
                            torch_rng=torch.get_rng_state(),cuda_rng=torch.cuda.get_rng_state(),
                            python_rng=random.getstate(),metrics_bytes=stream.buffer.tell()),rank,world)
                epoch+=1;step=0;previous=None;current_video=None
                if reader:reader.close();reader=None
                checkpoint(output,memory,optimizer,dict(epoch=epoch,next_step=0,global_step=global_step,
                    previous=None,current_video=None,torch_rng=torch.get_rng_state(),
                    cuda_rng=torch.cuda.get_rng_state(),python_rng=random.getstate(),
                    metrics_bytes=stream.buffer.tell()),rank,world)
        finally:
            if reader:reader.close()
    if rank==0:
        memory.save(str(output/'final'))
        atomic_json(output/'completed.json',dict(status='completed',epochs=a.epochs,
                    global_steps=global_step,architecture=a.architecture,release_sha256=digest(a.release_manifest)))
    if world>1:torch.distributed.barrier()


if __name__=='__main__':main()
