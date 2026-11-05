"""Two real Gloo ranks: unused Long parameters and zero-padding must synchronize."""
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest

import torch
from torch import nn
from torch.nn.parallel import DistributedDataParallel

from .train_full import Objective,checkpoint


class TinyMemory:
    def __init__(self):
        self.model=nn.Linear(4,4)
        self.compressors=nn.ModuleDict({'short':nn.Linear(4,4),'long':nn.Linear(4,4)})

    def encode_short(self,frames):
        values=torch.tensor(frames,dtype=torch.float32).reshape(1,-1,1).expand(1,-1,4)/60
        return self.compressors['short'](values)

    def update_long(self,previous,current):
        return self.compressors['long'](current if previous is None else current+previous.detach())

    def reconstruction_loss(self,tokens,target,kind):
        return (self.model(tokens)-1).square().mean()

    def save(self,path):
        Path(path).mkdir(parents=True)
        torch.save(self.model.state_dict(),Path(path)/'adapter.pt')


def worker(rank,directory,secondwise=False):
    torch.set_num_threads(1);torch.manual_seed(17)
    torch.distributed.init_process_group('gloo',init_method='file://'+str(Path(directory)/'store'),rank=rank,world_size=2)
    try:
        if secondwise:
            from .test_secondwise_qformer import tiny_memory, frames
            memory=tiny_memory();images=frames(60)
            reader=SimpleNamespace(frame_ending_at=lambda t:images[t])
        else:
            memory=TinyMemory()
            reader=SimpleNamespace(frame_ending_at=lambda t:t)
        objective=DistributedDataParallel(Objective(memory),find_unused_parameters=True)
        optimizer=torch.optim.AdamW(objective.parameters(),lr=0.01)
        for i,kind in enumerate(['short','block','global']):
            task=(dict(start=0,end=10 if kind=='short' else 60,kind=kind,target='target')
                  if rank==0 or i==0 else None)
            optimizer.zero_grad(set_to_none=True)
            loss,_=objective(task,reader,None)
            (loss*(1 if i==0 else 2)).backward();optimizer.step()
        weights=torch.cat([p.detach().flatten() for p in objective.parameters()])
        gathered=[torch.empty_like(weights) for _ in range(2)]
        torch.distributed.all_gather(gathered,weights)
        if not torch.equal(gathered[0],gathered[1]):raise AssertionError('Padded ranks diverged')
        torch.save({'synchronized':True},Path(directory)/f'passed-{rank}.pt')
    finally:torch.distributed.destroy_process_group()


class FullDDPTests(unittest.TestCase):
    @unittest.skipUnless(torch.distributed.is_available() and torch.distributed.is_gloo_available(),'Gloo unavailable')
    def test_two_ranks_unused_parameters_and_unequal_video_schedules(self):
        with tempfile.TemporaryDirectory() as directory:
            torch.multiprocessing.spawn(worker,args=(directory,),nprocs=2,join=True)
            self.assertTrue(all((Path(directory)/f'passed-{i}.pt').is_file() for i in range(2)))

    @unittest.skipUnless(torch.distributed.is_available() and torch.distributed.is_gloo_available(),'Gloo unavailable')
    def test_secondwise_attention_two_ranks_with_global_and_padding(self):
        with tempfile.TemporaryDirectory() as directory:
            torch.multiprocessing.spawn(worker,args=(directory,True),nprocs=2,join=True)
            self.assertTrue(all((Path(directory)/f'passed-{i}.pt').is_file() for i in range(2)))

    def test_checkpoint_preserves_previous_complete_state(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);memory=TinyMemory();optimizer=torch.optim.AdamW(memory.model.parameters())
            checkpoint(root,memory,optimizer,{'global_step':1},0,1)
            checkpoint(root,memory,optimizer,{'global_step':2},0,1)
            self.assertEqual(torch.load(root/'checkpoint-previous/rank-0.pt',weights_only=True)['global_step'],1)
            self.assertEqual(torch.load(root/'checkpoint-latest/rank-0.pt',weights_only=True)['global_step'],2)
            self.assertFalse((root/'checkpoint-writing').exists())


if __name__=='__main__':unittest.main()
