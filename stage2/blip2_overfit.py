"""Paired Stage2 overfit with true target batching and causal replay reuse."""
import argparse
from datetime import timedelta
import json
import os
from pathlib import Path
import random
import gc
import time
from types import SimpleNamespace

import torch
import torch.distributed as dist
from torch import nn
from torch.nn.parallel import DistributedDataParallel

from .batched_reconstruction import BatchedReconstruction, target_batches
from .blip2_memory import ARCHITECTURE
from .data import load_blocks, VideoReader
from .qformer import load_memory_model
from .reconstruct import collect_memories, controlled_memory, greedy_reconstruct, summarize, text_metrics
from .short_encoding import short_memory_scope
from .train_full import tasks_for_video, atomic_json


@torch.no_grad()
def cache_frozen_vision(memory, blocks, paths, frame_size=256):
    frames = {}
    for vid, rows in blocks.items():
        reader = VideoReader(paths[vid], frame_size)
        frames[vid] = []
        try:
            for start in range(0, int(rows[-1]["block_window"][1]), 10):
                images = [reader.frame_ending_at(t) for t in range(start, start + 10)]
                frames[vid].extend(f.detach().cpu() for f in memory.extract_visual_features(images))
        finally:
            reader.close()
    return frames


class Reconstruction(nn.Module):
    def __init__(self, memory):
        super().__init__()
        self.memory = memory
        self.backbone = memory.model
        self.compressors = memory.compressors

    def interval(self, frames, start, end):
        begin = start // 60 * 60 if self.memory.requires_short_history else start
        with short_memory_scope(self.memory):
            tokens = [self.memory.encode_visual_short(frames[t:min(t+10, end)])
                      for t in range(begin, end, 10)]
            return torch.cat(tokens, dim=1)[:, start-begin:, :]

    def forward(self, task, frames):
        start, end = task["start"], task["end"]
        previous = None
        if task["kind"] == "global":
            with torch.no_grad():
                for t in range(0, start, 60):
                    previous = self.memory.update_long(previous, self.interval(frames, t, t+60))
        tokens = self.interval(frames, start, end)
        if task["kind"] == "global":
            tokens = self.memory.update_long(previous, tokens)
        return self.memory.reconstruction_loss(tokens, task["target"], task["kind"])


def evaluate(memory, blocks, paths, out, rank, world):
    memory.eval()
    # Each rank decodes disjoint records, including when there are only three videos.
    with (out/f"predictions.rank{rank}.jsonl").open("x") as stream:
        for vid, rows in blocks.items():
            reader = VideoReader(paths[vid], 256)
            try:
                records = collect_memories(memory, rows, reader)
            finally:
                reader.close()
            for i in range(rank, len(records), world):
                r = records[i]
                for cond in ("real", "zero", "other_window"):
                    tokens, meta = controlled_memory(records, i, cond)
                    row = dict(video_id=vid, kind=r["kind"], window=r["window"],
                               reference=r["reference"], condition=cond, **meta)
                    if tokens is not None:
                        row.update(greedy_reconstruct(memory, tokens, r["kind"], 160))
                        row.update(text_metrics(row["prediction"], row["reference"]))
                    stream.write(json.dumps(row, ensure_ascii=False)+"\n")
                    stream.flush()
    if world > 1:
        dist.barrier()
    if rank == 0:
        predictions = [json.loads(line) for f in sorted(out.glob("predictions.rank*.jsonl"))
                       for line in f.read_text().splitlines()]
        unique = {k: len({r["prediction"] for r in predictions if r["kind"] == k and r["condition"] == "real"})
                  for k in ("short", "block", "global")}
        atomic_json(out/"report.json", dict(records=len(predictions), scores=summarize(predictions),
                                            distinct_real_outputs=unique))


def build_parser():
    parser = argparse.ArgumentParser(allow_abbrev=False)
    parser.add_argument("--run-dir", "--run", dest="run", type=Path, required=True,
                        help="Frozen experiment directory (--run is a legacy alias)")
    parser.add_argument("--arm", choices=["qformer", ARCHITECTURE], required=True)
    parser.add_argument("--profile-only", action="store_true", help="Zero-update memory/throughput probe")
    return parser


def profile_batches(objective, memory, tasks, frames, optimizer, cfg, out):
    """Single-device, forward/backward only; include realistic Adam state storage."""
    for group in optimizer.param_groups:
        for p in group['params']:
            state = optimizer.state[p]
            state['step'] = torch.tensor(0.)
            state['exp_avg'] = torch.zeros_like(p)
            state['exp_avg_sq'] = torch.zeros_like(p)
    # Long global targets first, followed by full blocks and short targets.
    ranked = sorted(tasks, key=lambda t: (t['kind'] == 'global', t['kind'] == 'block', t['end']), reverse=True)
    results = []
    reference = Reconstruction(memory)
    # Check real BF16 decoder masks and variable target lengths before profiling.
    memory.eval()
    selected = [next(t for t in tasks if t['kind'] == kind) for kind in ('short','block','global')]
    selected.append(max((t for t in tasks if t['kind']=='global'),key=lambda t:t['end']))
    with torch.no_grad():
        ref = torch.stack([reference(t,[f.to(memory.device) for f in frames[t['video_id']]]) for t in selected]).mean()
        got = objective(selected,frames)
        torch.testing.assert_close(got.float(),ref.float(),atol=.03,rtol=.02)
    atomic_json(out/'batch_parity.json',dict(status='passed',samples=len(selected),serial_loss=float(ref),
                                           batched_loss=float(got),absolute_difference=float((got-ref).abs())))
    del ref,got
    memory.train()

    variants = [('serial', 1)] + [('batched', b) for b in cfg.get('batch_candidates', [1, 2, 4, 8, 16])]
    repeats = int(cfg.get('profile_repeats', 3))
    for mode, requested in variants:
        if requested > len(ranked):
            continue
        rows = ranked[:requested]
        gc.collect(); torch.cuda.empty_cache(); torch.cuda.reset_peak_memory_stats()
        loss = None
        reference_frames = None
        try:
            if mode == 'serial':
                reference_frames = [x.to(memory.device) for x in frames[rows[0]['video_id']]]
            def forward():
                if mode == 'serial':
                    row = rows[0]
                    # Reference path expects device-resident frame features.
                    return reference(row, reference_frames)
                return objective(rows, frames)
            loss = forward(); loss.backward(); optimizer.zero_grad(set_to_none=True); del loss
            torch.cuda.synchronize(); started = time.perf_counter(); unix_start = time.time()
            for _ in range(repeats):
                loss = forward()
                if not torch.isfinite(loss):
                    raise FloatingPointError('Nonfinite profiling loss')
                loss.backward(); optimizer.zero_grad(set_to_none=True); del loss
            torch.cuda.synchronize(); elapsed = time.perf_counter() - started
            free, total = torch.cuda.mem_get_info()
            record = dict(mode=mode, batch_size=requested, status='passed',
                          seconds_per_batch=elapsed/repeats, samples_per_second=len(rows)*repeats/elapsed,
                          peak_allocated_gib=torch.cuda.max_memory_allocated()/2**30,
                          peak_reserved_gib=torch.cuda.max_memory_reserved()/2**30,
                          free_gib=free/2**30, total_gib=total/2**30,
                          safe=torch.cuda.max_memory_reserved() < .8*total and free > .15*total,
                          timed_start_unix=unix_start, timed_end_unix=time.time(),
                          replay_stats=objective.last_stats if mode == 'batched' else None)
        except torch.cuda.OutOfMemoryError:
            optimizer.zero_grad(set_to_none=True)
            if 'loss' in locals(): del loss
            memory.reset_short_memory();gc.collect();torch.cuda.empty_cache()
            record = dict(mode=mode,batch_size=requested,status='oom',safe=False)
        del reference_frames
        results.append(record);atomic_json(out/'batch_profiles.json',results)
        print('BATCH_PROFILE',json.dumps(record),flush=True)
        if not record['safe'] and mode == 'batched':
            break
    safe = [r for r in results if r['mode'] == 'batched' and r.get('safe')]
    if not safe:
        raise RuntimeError('No safe batch measured')
    best = max(safe,key=lambda r:r['samples_per_second'])
    atomic_json(out/'status.json',dict(status='completed',mode='profile_only',optimizer_steps=0,
                                      recommended_per_rank_batch=best['batch_size']))


def main():
    args = build_parser().parse_args()
    cfg = json.loads((args.run/"experiment.json").read_text())
    rank = int(os.environ.get("LOCAL_RANK", "0"))
    world = int(os.environ.get("WORLD_SIZE", "1"))
    if world != cfg['gpus']:
        raise ValueError('Launched world size must match the frozen config')
    if args.profile_only and world != 1:
        raise ValueError('Profile batches on one GPU first; DDP OOM can stall peers')
    torch.cuda.set_device(rank)
    if world > 1:
        dist.init_process_group("nccl", timeout=timedelta(hours=2))
    try:
        torch.manual_seed(cfg["seed"])
        blocks = load_blocks(args.run/"train.jsonl", validate_text=False)
        paths = json.loads((args.run/"video_map.json").read_text())
        tasks = [t for vid, rows in blocks.items() for t in tasks_for_video(vid, rows)]
        if len(tasks) != cfg["targets"] or len(tasks) % world:
            raise ValueError("Incomplete global batch or incorrect target count")
        arm_kwargs = {"bert_dir": cfg["bert_dir"]} if args.arm == ARCHITECTURE else {}
        memory = load_memory_model(cfg["model_name"], architecture=args.arm, **arm_kwargs).to("cuda").train()
        memory.model.config.use_cache = False
        checkpointing = cfg.get('gradient_checkpointing', False)
        if checkpointing:
            memory.model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
        else:
            memory.model.gradient_checkpointing_disable()
            for module in memory.model.modules():
                if hasattr(module, 'gradient_checkpointing'): module.gradient_checkpointing = False
        acceleration = {'linear_attention': 'default'}
        if cfg.get('fast_linear_attention', False):
            from endtoend.acceleration import enable_fast_attention
            acceleration = enable_fast_attention(SimpleNamespace(backbone=memory.model))
        frames = cache_frozen_vision(memory, blocks, paths, cfg.get('frame_size',256))
        bare = BatchedReconstruction(memory, short_chunk=cfg.get('short_frame_batch',128))
        objective = (DistributedDataParallel(bare,device_ids=[rank],broadcast_buffers=False,
                                             find_unused_parameters=True) if world > 1 else bare)
        optimizer = torch.optim.AdamW((p for p in objective.parameters() if p.requires_grad), lr=cfg["learning_rate"])
        out = args.run/args.arm
        if rank == 0:
            out.mkdir()
            atomic_json(out/"training_config.json", {**cfg, "qformer_config": memory.qformer_config,
                        "init_adapter": None, "architecture": args.arm, 'acceleration':acceleration,
                        'gradient_checkpointing':checkpointing,'vision_cache_device':'cpu',
                        'active_checkpoint_modules':[n for n,m in memory.model.named_modules() if getattr(m,'gradient_checkpointing',False)],
                        'loss_reduction':'mean of per-target token means',
                        'dropout_note':'shared representations share dropout draws within a batch; deterministic equivalence tested with dropout off'})
        if world > 1:dist.barrier()
        if args.profile_only:
            profile_batches(bare,memory,tasks,frames,optimizer,cfg,out)
            return
        per_rank = int(cfg.get('per_rank_targets',1))
        step = 0
        with (out/f"train.rank{rank}.jsonl").open("x") as stream:
            for epoch in range(cfg["epochs"]):
                batches = target_batches(tasks,per_rank,rank,world,cfg['seed']+epoch)
                for rows in batches:
                    optimizer.zero_grad(set_to_none=True)
                    torch.cuda.reset_peak_memory_stats();started=time.perf_counter()
                    loss = objective(rows, frames)
                    if not torch.isfinite(loss):raise FloatingPointError('Nonfinite target batch')
                    loss.backward()
                    if step == 0:
                        grad = next(memory.compressors["short"].output_projection.parameters()).grad
                        if grad is None or not grad.isfinite().all() or grad.abs().max() == 0:
                            raise FloatingPointError("Missing or invalid Short gradient on first real batch")
                    optimizer.step();torch.cuda.synchronize();step+=1
                    record=dict(epoch=epoch+1,step=step,loss=loss.item(),per_rank_targets=len(rows),
                                global_targets=len(rows)*world,seconds=time.perf_counter()-started,
                                peak_allocated_gib=torch.cuda.max_memory_allocated()/2**30,
                                peak_reserved_gib=torch.cuda.max_memory_reserved()/2**30,
                                replay_stats=bare.last_stats,tasks=rows)
                    stream.write(json.dumps(record)+'\n');stream.flush()
                    if rank==0:print('TRAIN',json.dumps({k:v for k,v in record.items() if k!='tasks'}),flush=True)
                if rank==0:atomic_json(out/'status.json',dict(status='running',epoch=epoch+1,step=step))
        if rank == 0:memory.save(str(out/"final"))
        if world > 1:dist.barrier()
        del objective, optimizer, frames, bare
        torch.cuda.empty_cache()
        evaluate(memory, blocks, paths, out, rank, world)
        if rank==0:atomic_json(out/'status.json',dict(status='completed',epochs=cfg['epochs'],step=step))
    finally:
        if dist.is_initialized():dist.destroy_process_group()


if __name__ == "__main__":
    main()
