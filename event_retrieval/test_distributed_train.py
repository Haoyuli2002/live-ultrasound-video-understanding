"""Check two-rank gradients against one global NTP update, including an empty tail shard."""
import tempfile
import unittest
from pathlib import Path

import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel

from .distributed_train import CPUAdamW, distributed_epoch


class TinyLoss(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.weight = torch.nn.Parameter(torch.tensor([0.2, 0.7]))

    def forward(self, values, counts):
        targets = torch.tensor([v for row in values for v in range(row)], dtype=torch.float32)
        losses = (self.weight[0] * targets + self.weight[1] - 1).square()
        counts[0] += losses.detach().sum()
        counts[1] += losses.numel()
        return losses.mean()


def compare(rank, path):
    dist.init_process_group("gloo", init_method=Path(path).as_uri(), rank=rank, world_size=2)
    try:
        model = DistributedDataParallel(TinyLoss(), gradient_as_bucket_view=True)
        host = CPUAdamW(torch.optim.AdamW(model.parameters(), lr=0.03))
        reference = TinyLoss()
        adam = torch.optim.AdamW(reference.parameters(), lr=0.03)
        rows = [1, 4, 1, 3, 5, 2, 3, 1, 2]
        observed = []
        def updated(**record):
            observed.append((record['processed'], int(next(iter(host.host.state.values()))['step'].item())))
        metric = distributed_epoch(rows, model, host, list, 2, 2, False, on_update=updated)
        assert observed == ([(8, 1), (9, 2)] if rank == 0 else []), observed
        counts = torch.zeros(2)
        for start in range(0, len(rows), 8):
            adam.zero_grad()
            reference(rows[start:start+8], counts).backward()
            adam.step()
        torch.testing.assert_close(model.module.weight, reference.weight)
        torch.testing.assert_close(torch.tensor(metric), counts[0] / counts[1])
    finally:
        dist.destroy_process_group()


class DistributedTrainTest(unittest.TestCase):
    def test_partial_update_and_global_tokens(self):
        with tempfile.TemporaryDirectory() as directory:
            mp.start_processes(compare, args=(str(Path(directory) / "store"),), nprocs=2,
                               join=True, start_method="fork")


if __name__ == "__main__":
    unittest.main()
