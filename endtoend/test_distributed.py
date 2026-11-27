import copy
import datetime
from pathlib import Path
import tempfile
import unittest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.nn import functional as F
from .distributed import scale_objectives,synchronize_gradients,aggregate_metrics,rank_batches,gather_evaluation
from .timing_loss import readiness_loss_terms
from .optimization import backward_objectives
from .full_data import batch_plan


class Toy(torch.nn.Module):
    def __init__(self):
        super().__init__();self.reasoning=torch.nn.Linear(2,3);self.rdy_token=torch.nn.Parameter(torch.randn(3));self.head=torch.nn.Linear(3,1)
        self.unused=torch.nn.Parameter(torch.ones(2));self.readiness_loss_weight=1.
    def _readiness_surrogate(self,loss):
        params=[self.rdy_token,*self.head.parameters()]
        grads=torch.autograd.grad(loss,params,retain_graph=True)
        surrogate=sum((p*g.detach()).sum() for p,g in zip(params,grads))
        return surrogate-surrogate.detach()+loss.detach()


def data():
    rows=[]
    for task,labels in [('a',[0,0]),('b',[0,0,0,1]),('c',[1,1])]:
        for i,label in enumerate(labels):
            rows.append(dict(task_id=task,video_id=task,question='q',sample_id=f'{task}-{i}',current_time=float(i),ready=label,tokens=i+1,x=[i*.13+1,ord(task)*.01]))
    return rows


def objective(model,rows):
    x=torch.tensor([r['x'] for r in rows]);z=model.reasoning(x)+model.rdy_token
    logits=model.head(z).flatten();terms=readiness_loss_terms(logits,rows,'contrastive_bce_gold',sample_pairs=False,bce_positive_weight=1.)
    # Labels depend on task/time, not the rank-local position.
    token_losses=[]
    for i,r in enumerate(rows):
        if r['ready']:token_losses.extend([F.cross_entropy(z[i:i+1],torch.tensor([int(r['current_time'])%3]))]*r['tokens'])
    ce=torch.stack(token_losses).mean() if token_losses else logits.new_zeros(())
    out=dict(loss=ce+terms['loss'],answer_loss=ce,readiness_loss=terms['loss'],readiness_bce=terms['bce'],readiness_weighted_bce=terms['weighted_bce'],timing=terms['timing'])
    counts=torch.tensor([len(rows),len(token_losses),terms['timing']['paired_tasks'],terms['timing']['temporal_tasks']],dtype=torch.float64)
    return out,counts


def worker(rank,world,initfile):
    torch.set_num_threads(1);torch.manual_seed(123)
    dist.init_process_group('gloo',init_method='file://'+initfile,rank=rank,world_size=world,timeout=datetime.timedelta(seconds=120))
    try:
        model=Toy();reference=copy.deepcopy(model)
        opt=torch.optim.AdamW(model.parameters(),lr=.01);refopt=torch.optim.AdamW(reference.parameters(),lr=.01)
        full=data();shards=[full[:2],full[2:6],full[6:],[]]
        for step in range(2):
            if world == 2:
                local = ([full[:2], full[2:]] if step == 0 else [full, []])[rank]
            else:
                local=shards[(rank+step)%world]
            out,counts=objective(model,local) if local else (None,torch.zeros(4,dtype=torch.float64))
            total=counts.clone();dist.all_reduce(total)
            scaled=scale_objectives(out,counts,total,world,'contrastive_bce_gold',.1,1.)
            if scaled is not None:backward_objectives(model,scaled)
            synchronize_gradients(model.parameters(),bucket_bytes=40)
            ref,_=objective(reference,full);backward_objectives(reference,ref)
            for p,q in zip(model.parameters(),reference.parameters()):
                if q.grad is None:assert p.grad is None
                else:torch.testing.assert_close(p.grad,q.grad,atol=2e-7,rtol=2e-6)
            actual=aggregate_metrics(out,counts,total,'contrastive_bce_gold',.1,1.,1.)
            torch.testing.assert_close(actual['loss'].float(),ref['loss'],atol=2e-7,rtol=2e-6)
            assert actual['timing']['used_pairs']==3
            opt.step();refopt.step();opt.zero_grad(set_to_none=True);refopt.zero_grad(set_to_none=True)
            for p,q in zip(model.parameters(),reference.parameters()):torch.testing.assert_close(p,q,atol=2e-7,rtol=2e-6)
        from .feature_cache import VideoFeatureCache,prepare_distributed_features
        import json
        def builder(model,examples,directory,size,chunk,mode):
            cache=VideoFeatureCache(directory/'features');audit=[]
            for vid in sorted({e['video_id'] for e in examples}):
                tensor=torch.full((1,5,1,2),float(ord(vid)))
                cache.add_video(vid,tensor,{e['sample_id']:list(range(int(e['current_time'])+1)) for e in examples if e['video_id']==vid})
                audit.append(dict(video_id=vid))
            cache.write_manifest();(directory/'frame_audit.json').write_text(json.dumps(audit));return cache
        cache=prepare_distributed_features(builder,None,full,Path(initfile).parent/'cache',256,8,rank,world)
        for e in full:
            expected=torch.full((1,int(e['current_time'])+1,1,2),float(ord(e['video_id'])))
            torch.testing.assert_close(cache[e['sample_id']],expected)
        records=[dict(sample_id=e['sample_id']) for e in local]
        merged,_,_,_=gather_evaluation(records,[1.]*len(local),[torch.tensor([2.]) for e in local],[],full)
        assert [r['sample_id'] for r in merged]==[e['sample_id'] for e in full]
    finally:dist.destroy_process_group()


class DistributedTests(unittest.TestCase):
    def test_two_process_gradients_match_global_objective_with_empty_rank(self):
        with tempfile.TemporaryDirectory() as tmp:mp.spawn(worker,args=(2,str(Path(tmp)/'rendezvous')),nprocs=2,join=True)

    def test_four_process_gradients_match_global_objective_with_empty_rank(self):
        with tempfile.TemporaryDirectory() as tmp:mp.spawn(worker,args=(4,str(Path(tmp)/'rendezvous')),nprocs=4,join=True)

    def test_video_blocks_reduce_replay_and_match_frozen_plan(self):
        from .optimization import replay_updates
        examples = [
            dict(video_id=str(v), task_id=f'{v}-{q}', sample_id=f'{v}-{q}-{label}',
                 current_time=100.5 + q + label, ready=label)
            for v in range(8) for q in range(8) for label in (0, 1)
        ]
        def cost(steps):
            return sum(replay_updates(batch) for ranks in steps for batch in ranks)
        optimized = rank_batches(examples, 8, 41, 2, video_block_size=4)
        self.assertLess(cost(optimized), cost(rank_batches(examples, 8, 41, 2)))
        expected = batch_plan(examples, 8, 1, 40, world_size=2, video_block_size=4)
        order = [[[r['sample_id'] for r in b] for b in ranks] for ranks in optimized]
        self.assertEqual(order, expected['batch_order'][0])
        self.assertNotEqual(optimized, rank_batches(examples, 8, 42, 2, video_block_size=4))
        self.assertEqual(replay_updates([examples[0], examples[0], examples[1]]), 104)

    def test_gpu_summaries_keep_device_measurements_separate(self):
        from .gpu_stats import summarize_devices
        rows=[dict(uuid='gpu0',timestamp=0,gpu_util=10.,memory_util=20.,memory_gib=4.,power_w=100.),
              dict(uuid='gpu1',timestamp=0,gpu_util=90.,memory_util=60.,memory_gib=8.,power_w=200.)]
        result=summarize_devices(rows)
        self.assertEqual(result['gpu0']['gpu_util_mean_pct'],10.)
        self.assertEqual(result['gpu1']['memory_peak_gib'],8.)

    def test_plan_has_no_duplicates_and_keeps_questions_on_one_rank(self):
        examples=data();plan=batch_plan(examples,4,3,17,world_size=4)
        self.assertEqual(plan['total_optimizer_steps'],3)
        for epoch,steps in enumerate(plan['batch_order'],1):
            ids=[sid for ranks in steps for batch in ranks for sid in batch]
            self.assertEqual(sorted(ids),sorted(e['sample_id'] for e in examples))
            for task in ['a','b','c']:
                self.assertEqual(sum(any(s.startswith(task+'-') for s in batch) for ranks in steps for batch in ranks),1)
            expected=rank_batches(examples,4,17+epoch,4)
            self.assertEqual(steps,[[[r['sample_id'] for r in batch] for batch in ranks] for ranks in expected])
            self.assertTrue(any(not batch for ranks in steps for batch in ranks))


if __name__=='__main__':unittest.main()
