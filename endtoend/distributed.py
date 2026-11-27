"""Synchronous data parallelism compatible with isolated autograd.grad readiness.

Gradients are bucket-reduced after the local isolated backward. We deliberately
avoid DDP reducer hooks: readiness uses autograd.grad before the final backward.
Each objective is normalized by its own global count, not by equal rank weights.
"""
import datetime
import os
import torch
import torch.distributed as dist
from .optimization import distribute_batches, task_batches


def initialize(expected_world=1):
    world=int(os.environ.get('WORLD_SIZE','1'));rank=int(os.environ.get('RANK','0'))
    if world!=expected_world:raise ValueError(f'Expected {expected_world} processes, got {world}; use the frozen torchrun launcher')
    if world>1:
        torch.cuda.set_device(int(os.environ['LOCAL_RANK']))
        dist.init_process_group('nccl',timeout=datetime.timedelta(minutes=60))
    return rank,world


def rank_batches(examples,local_batch,seed,world,video_block_size=0):
    batches=task_batches(examples,local_batch,seed,video_block_size)
    return distribute_batches(batches, world, bool(video_block_size), seed)


def counts_for_output(model,out,rows,device):
    if out is None:return torch.zeros(4,dtype=torch.float64,device=device)
    answers=[r['answer']+model.backbone.tokenizer.eos_token for r in rows if r['ready']]
    tok=model.backbone.tokenizer;tok.truncation_side='right'
    token_count=sum(map(len,tok(answers,add_special_tokens=False,truncation=True,
                               max_length=model.max_output_txt_len)['input_ids'])) if answers else 0
    timing=out['timing']
    return torch.tensor([len(rows),token_count,timing['paired_tasks'] if timing else 0,
                         timing['temporal_tasks'] if timing else 0],dtype=torch.float64,device=device)


def scale_objectives(out,local,total,world,objective,lambda_reg,bce_weight):
    if out is None:return None
    scale=world*local/total.clamp_min(1)
    answer=out['answer_loss']*scale[1].to(out['answer_loss'].dtype)
    if objective=='bce':ready=out['readiness_bce']*scale[0]
    else:
        ready=out['timing']['contrastive']*scale[2]+lambda_reg*out['timing']['temporal']*scale[3]
        if objective=='contrastive_bce_gold':ready=ready+bce_weight*out['readiness_weighted_bce']*scale[0]
    return dict(out,answer_loss=answer,readiness_loss=ready.to(out['readiness_loss'].dtype),loss=answer+ready)


def synchronize_gradients(parameters,bucket_bytes=25*1024*1024):
    params=[p for p in parameters if p.requires_grad]
    device=params[0].device
    used=torch.tensor([p.grad is not None for p in params],device=device,dtype=torch.int32)
    dist.all_reduce(used,op=dist.ReduceOp.SUM)
    flags=used.tolist();world=dist.get_world_size()
    buckets=[];current=[];size=0;dtype=None
    for p,flag in zip(params,flags):
        if not flag:continue  # Preserve None for globally-unused parameters/AdamW.
        nbytes=p.numel()*p.element_size()
        if current and (size+nbytes>bucket_bytes or p.dtype!=dtype):
            buckets.append(current);current=[];size=0
        current.append(p);size+=nbytes;dtype=p.dtype
    if current:buckets.append(current)
    for bucket in buckets:
        flat=torch.cat([(p.grad if p.grad is not None else torch.zeros_like(p)).reshape(-1) for p in bucket])
        dist.all_reduce(flat);flat.div_(world)
        offset=0
        for p in bucket:
            value=flat[offset:offset+p.numel()].view_as(p);offset+=p.numel()
            if p.grad is None:p.grad=value.clone()
            else:p.grad.copy_(value)


def aggregate_metrics(out,counts,total,objective,lambda_reg,bce_weight,readiness_weight):
    values=torch.zeros(10,dtype=torch.float64,device=counts.device)
    if out is not None:
        values[0]=out['answer_loss'].detach()*counts[1]
        values[1]=out['readiness_bce'].detach()*counts[0]
        values[2]=out['readiness_weighted_bce'].detach()*counts[0]
        if out['timing'] is not None:
            t=out['timing'];values[3]=t['contrastive'].detach()*counts[2];values[4]=t['temporal'].detach()*counts[3]
            values[5:]=values.new_tensor([t['pairs'],t['sampled_pairs'],t['used_pairs'],t['paired_tasks'],t['unpaired_tasks']])
    dist.all_reduce(values)
    ce=values[0]/total[1].clamp_min(1);bce=values[1]/total[0].clamp_min(1);weighted=values[2]/total[0].clamp_min(1)
    ctr=values[3]/total[2].clamp_min(1);tv=values[4]/total[3].clamp_min(1)
    ready=bce if objective=='bce' else ctr+lambda_reg*tv+(bce_weight*weighted if objective=='contrastive_bce_gold' else 0)
    timing=None if objective=='bce' else dict(contrastive=ctr,temporal=tv,pairs=int(values[5]),sampled_pairs=int(values[6]),used_pairs=int(values[7]),paired_tasks=int(values[8]),unpaired_tasks=int(values[9]),pair_mode='all_pairs')
    return dict(loss=ce+readiness_weight*ready,answer_loss=ce,readiness_loss=ready,
                readiness_bce=bce,readiness_weighted_bce=weighted,timing=timing)


def gather_evaluation(records,losses,logits,answer_losses,examples):
    local=dict(records=records,losses=losses,logits=[float(x.item()) for x in logits],answer_losses=answer_losses)
    gathered=[None]*dist.get_world_size();dist.all_gather_object(gathered,local)
    by_id={}
    answers=[]
    for item in gathered:
        answers.extend(item['answer_losses'])
        for record,loss,logit in zip(item['records'],item['losses'],item['logits']):
            if record['sample_id'] in by_id:raise ValueError('Repeated distributed evaluation endpoint')
            by_id[record['sample_id']]=(record,loss,logit)
    if set(by_id)!={e['sample_id'] for e in examples}:raise ValueError('Incomplete distributed evaluation')
    ordered=[by_id[e['sample_id']] for e in examples]
    return [x[0] for x in ordered],[x[1] for x in ordered],[torch.tensor([x[2]]) for x in ordered],answers
