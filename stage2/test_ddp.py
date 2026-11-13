"""CPU process-group regression for the Stage 2 DDP gradient contract."""
from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel

from stage2.train import Stage2TrainStep, _verify_ddp_gradients


class _ToyBackbone(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.lora_weight = torch.nn.Parameter(torch.ones(1, 1))

    def forward(self, value):
        return value @ self.lora_weight.T


class _ToyMemory:
    """Call the registered backbone through the same helper path as MemoryModel."""

    def __init__(self):
        self.model = _ToyBackbone()

    def encode_frame(self, frame):
        return self.model(torch.tensor([[float(frame)]]))

    def reconstruction_loss(self, memory, target, kind):
        return memory.mean() * float(target)

    def update_long(self, previous, short):
        return short


def _ddp_gradient_worker(rank: int, rendezvous: str, queue) -> None:
    dist.init_process_group(
        "gloo", init_method=f"file://{rendezvous}", rank=rank, world_size=2)
    try:
        memory = _ToyMemory()
        step = DistributedDataParallel(
            Stage2TrainStep(memory), find_unused_parameters=True)
        optimizer = torch.optim.SGD(step.parameters(), lr=0.1)
        gradients = []
        # Rank 1 stops after one step, exercising the same uneven-shard join
        # path used by the real video trainer.
        with step.join(divide_by_initial_world_size=True):
            for index in range(2 if rank == 0 else 1):
                minute_boundary = rank == 0 and index == 0
                prior_short = ([torch.zeros(1, 10, 1) for _ in range(5)]
                               if minute_boundary else [])
                loss, *_ = step(
                    [1.0] * 10, prior_short, None,
                    "1" if rank == 0 else "6",
                    "2" if minute_boundary else None,
                    "4" if minute_boundary else None)
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                if index == 0:
                    _verify_ddp_gradients(step)
                gradients.append(float(memory.model.lora_weight.grad.item()))
                optimizer.step()
        queue.put((rank, gradients, float(memory.model.lora_weight.item())))
    finally:
        dist.destroy_process_group()


class Stage2DDPTests(unittest.TestCase):
    def test_outer_step_synchronizes_gradients_and_uneven_join(self):
        if not dist.is_available() or not dist.is_gloo_available():
            self.skipTest("PyTorch Gloo distributed backend is unavailable")
        context = mp.get_context("spawn")
        queue = context.SimpleQueue()
        with tempfile.TemporaryDirectory() as directory:
            rendezvous = str(Path(directory) / "ddp_rendezvous")
            mp.spawn(_ddp_gradient_worker, args=(rendezvous, queue), nprocs=2)
            results = dict((rank, (grads, weight))
                           for rank, grads, weight in (queue.get(), queue.get()))
        # The minute-boundary rank has local gradient 2 (short 1, block 1/3,
        # global 2/3), while the other rank has gradient 6. DDP averages to 4
        # before clipping/optimizer.step. Once rank 1 joins, the lone gradient is
        # divided by the initial world size: 1 / 2 = 0.5.
        self.assertAlmostEqual(results[0][0][0], 4.0, places=6)
        self.assertAlmostEqual(results[1][0][0], 4.0, places=6)
        self.assertAlmostEqual(results[0][0][1], 0.5, places=6)
        self.assertAlmostEqual(results[0][1], results[1][1], places=6)


if __name__ == "__main__":
    unittest.main()
