"""Distributed token-mean NTP over globally ordered streaming QA endpoints."""
from contextlib import ExitStack, closing
from pathlib import Path

import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel

from .index import EventIndex
from .replay import index_source
from .retrieve import QwenRelevance


class CPUAdamW:
    """Keep FP32 AdamW parameters and moments on host while DDP reduces GPU gradients."""
    def __init__(self, device_optimizer):
        self.param_groups = device_optimizer.param_groups
        groups = []
        for group in self.param_groups:
            host = [torch.empty_like(p, device="cpu") for p in group["params"]]
            groups.append(dict(group, params=host))
        self.host = torch.optim.AdamW(groups, foreach=False)
        self.device_optimizer = device_optimizer

    @torch.no_grad()
    def step(self):
        for device, host in zip(self.param_groups, self.host.param_groups, strict=True):
            for p, cpu in zip(device["params"], host["params"], strict=True):
                if p.grad is not None:
                    cpu.copy_(p)
                    cpu.grad = p.grad.to("cpu")
        self.host.step()
        for device, host in zip(self.param_groups, self.host.param_groups, strict=True):
            for p, cpu in zip(device["params"], host["params"], strict=True):
                if cpu.grad is not None:
                    p.copy_(cpu)
        self.host.zero_grad(set_to_none=True)

    def zero_grad(self, *, set_to_none):
        self.device_optimizer.zero_grad(set_to_none=set_to_none)

    def state_dict(self):
        return self.host.state_dict()

    def load_state_dict(self, state):
        self.host.load_state_dict(state)


class JointLoss(torch.nn.Module):
    def __init__(self, towers, decoder, loss_fn):
        super().__init__()
        self.towers = towers
        self.backbone = decoder.backbone
        self.decoder = decoder
        self.loss_fn = loss_fn

    def forward(self, examples, counts):
        return self.loss_fn(examples, self.towers, self.decoder, loss_sums=counts)


def wrap_joint_loss(towers, decoder, loss_fn, device):
    return DistributedDataParallel(
        JointLoss(towers, decoder, loss_fn),
        device_ids=[device.index] if device.type == "cuda" else None,
        broadcast_buffers=False, find_unused_parameters=True, gradient_as_bucket_view=True,
    )


def distributed_epoch(rows, model, optimizer, prepare, batch_size, ga, bf16, on_update=None):
    """Every rank joins every backward, including a partially filled final batch."""
    world, rank = dist.get_world_size(), dist.get_rank()
    device = next(model.parameters()).device
    sums = torch.zeros(2, device=device)
    update_tokens = torch.zeros((), device=device)
    global_batch = world * batch_size
    optimizer.zero_grad(set_to_none=True)
    for micro, start in enumerate(range(0, len(rows), global_batch), 1):
        group = rows[start:start + global_batch]
        local = group[rank::world]
        # An empty tail shard participates with a zero-weight copy of a real endpoint.
        examples = prepare(local or group[:1])
        counts = torch.zeros_like(sums)
        with torch.autocast(device.type, dtype=torch.bfloat16, enabled=bf16):
            mean_loss = model(examples, counts)
        if not torch.isfinite(mean_loss):
            raise FloatingPointError(f"Nonfinite DDP loss on rank {rank}")
        if not local:
            counts.zero_()
        (mean_loss * counts[1]).backward()
        sums += counts
        update_tokens += counts[1]
        if micro % ga == 0 or start + global_batch >= len(rows):
            dist.all_reduce(update_tokens)
            # DDP averaged the summed token gradients across ranks.
            scale = world / update_tokens
            for group_params in optimizer.param_groups:
                for param in group_params['params']:
                    if param.grad is not None:
                        param.grad.mul_(scale)
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)
            update_tokens.zero_()
            if rank == 0 and on_update is not None:
                on_update(processed=min(start + global_batch, len(rows)))
    dist.all_reduce(sums)
    return (sums[0] / sums[1]).item()


def train_epoch(rows, cache, model_path, model, optimizer, prepare_decision, batch_size, ga, bf16, retrieval_batch_size, on_update=None):
    towers, decoder = model.module.towers, model.module.decoder
    towers.train()
    scorer = QwenRelevance(decoder.backbone)

    def prepare(local):
        with ExitStack() as stack:
            indexes = {}
            examples = []
            for row in local:
                vid = row['video_id']
                if vid not in indexes:
                    video = Path(row['video_path']).resolve(strict=True)
                    indexes[vid] = stack.enter_context(closing(EventIndex(
                        cache / (vid + '.sqlite'), vid, index_source(video, model_path), read_only=True)))
                index = indexes[vid]
                if not 0 <= row['current_time'] <= index.eof:
                    raise ValueError('Decision outside the indexed video')
                examples.append(prepare_decision(index, row, towers, scorer, retrieval_batch_size=retrieval_batch_size))
            return examples

    return distributed_epoch(rows, model, optimizer, prepare, batch_size, ga, bf16, on_update=on_update)
