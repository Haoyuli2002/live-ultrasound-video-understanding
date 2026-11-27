"""Joint training on disjoint Stage3 splits or an explicit QA overfit set."""
import argparse
from contextlib import closing
import json
import math
import os
from datetime import timedelta
from pathlib import Path
import random

import torch
import torch.distributed as dist

from endtoend.data import digest, load_examples
from lavis.models.streamready.backbone_qwen3_5 import Qwen35Backbone
from .answer import AnswerDecoder, DecisionContext
from .checkpoint import restore, save
from . import distributed_train
from .progress import TrainingProgress
from .index import EventIndex
from .overfit_data import load_overfit
from .qformer import EventQFormers
from .prebuild import prepare_indexes
from .replay import index_source
from .retrieve import QuestionSession, QwenRelevance


def load_split(directory, split):
    rows = load_examples(directory / 'samples.jsonl', directory / 'tasks.jsonl', directory / 'video_map.json')
    if not rows or any(row['split'] != split for row in rows):
        raise ValueError(f"Expected nonempty {split} split in {directory}")
    return rows


def joint_optimizer(backbone, towers, lr, llm_lr):
    # FP32 master parameters and Adam moments. Frozen video encoder stays in its loaded dtype.
    text = backbone.model.model.language_model
    text.float().requires_grad_(True)
    backbone.model.get_output_embeddings().float().requires_grad_(True)
    return torch.optim.AdamW([
        dict(params=[p for p in towers.parameters() if p.requires_grad], lr=lr),
        dict(params=[p for p in backbone.model.parameters() if p.requires_grad], lr=llm_lr),
    ], foreach=False)


def prepare_decision(index, row, towers, scorer, *, retrieval_batch_size=4):
    now = row['current_time']
    last = min(math.floor(now), index.detector.next_second - 1)
    first = max(0, last + 1 - towers.recent_window)
    history = QuestionSession(index, row['question'], scorer, towers, retrieval_batch_size=retrieval_batch_size)
    ranked = history.rank(now if now == index.eof else math.floor(now))
    events, valid, sources = history.materialize(ranked)
    frames = torch.stack([index.frame(t)[1] for t in range(first, last + 1)])
    ctx = DecisionContext(row['question'], row['query_time'], now, first, last, tuple(sources))
    target = row.get('target')
    if target is None:
        target = '<ANSWER> ' + row['answer'] if row['ready'] else '<WAIT>'
    return frames, events[0], valid[0], ctx, target


def batch_loss(examples, towers, decoder, *, endpoint_weights=None, loss_sums=None):
    frames, events, valid, contexts, targets = zip(*examples)
    device = towers.visual_projection.weight.device
    b = len(examples)
    t = max(len(x) for x in frames)
    e = max(x.size(0) for x in events)
    k = max(x.size(1) for x in events)
    patches = frames[0].shape[1:]
    recent = frames[0].new_zeros((b, t, *patches), device=device)
    history = recent.new_zeros((b, e, k, *patches))
    recent_mask = torch.zeros(b, t, dtype=torch.bool, device=device)
    event_mask = torch.zeros(b, e, k, dtype=torch.bool, device=device)
    for i, (clip, slots, mask) in enumerate(zip(frames, events, valid)):
        recent[i, :len(clip)].copy_(clip)
        recent_mask[i, :len(clip)] = True
        history[i, :slots.size(0), :slots.size(1)].copy_(slots)
        event_mask[i, :slots.size(0), :slots.size(1)].copy_(mask)
    prefix, mask = towers(recent, history, recent_mask=recent_mask, event_mask=event_mask)
    return decoder.alignment_loss(prefix, mask, contexts, targets, average_endpoints=True,
                                  endpoint_weights=endpoint_weights, loss_sums=loss_sums)


def decision_loss(index, row, towers, decoder, scorer):
    return batch_loss([prepare_decision(index, row, towers, scorer)], towers, decoder,
                      endpoint_weights=[row.get("loss_weight", 1)])


def batch_step(examples, towers, decoder, optimizer, *, bf16=False, endpoint_weights=None, loss_sums=None,
               accumulation_start=None, finish_update=True):
    tokens_before = loss_sums[1].clone() if accumulation_start is not None else None
    device_type = towers.visual_projection.weight.device.type
    with torch.set_grad_enabled(optimizer is not None), torch.autocast(device_type, dtype=torch.bfloat16, enabled=bf16):
        loss = batch_loss(examples, towers, decoder, endpoint_weights=endpoint_weights, loss_sums=loss_sums)
    if not torch.isfinite(loss):
        raise FloatingPointError("Nonfinite event QA loss")
    if optimizer is not None:
        if accumulation_start is None:
            loss.backward()
        else:
            # Recover the microbatch CE sum, then normalize once over the entire update.
            (loss * (loss_sums[1] - tokens_before)).backward()
        if finish_update:
            if accumulation_start is not None:
                tokens = loss_sums[1] - accumulation_start
                for group in optimizer.param_groups:
                    for parameter in group['params']:
                        if parameter.grad is not None:
                            parameter.grad.div_(tokens)
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)
    return loss.detach().item()


def ordered_videos(rows, seed, shuffle):
    by_video = {}
    for row in rows:
        by_video.setdefault(row['video_id'], []).append(row)
    videos = list(by_video)
    rng = random.Random(seed)
    if shuffle:
        rng.shuffle(videos)
    for video in videos:
        examples = by_video[video]
        if shuffle:
            rng.shuffle(examples)
        yield video, examples


def run_split(rows, cache, model_path, towers, decoder, optimizer=None, seed=0, batch_size=1, bf16=False, retrieval_batch_size=4, metrics=None, gradient_accumulation=1, progress=None):

    backbone = decoder.backbone
    scorer = QwenRelevance(backbone)
    towers.train(optimizer is not None)
    loss_sums = (towers.visual_projection.weight.new_zeros(2, dtype=torch.float32)
                 if decoder.token_mean_ntp or (decoder.separate_action_loss and metrics is not None) else None)
    if gradient_accumulation > 1 and (not decoder.token_mean_ntp or optimizer is None):
        raise ValueError("Gradient accumulation requires training with the token-mean NTP objective")
    accumulation_start = loss_sums[1].clone() if gradient_accumulation > 1 else None
    processed = 0
    total = 0.
    pending = []
    pending_weights = []
    # A split-wide denominator keeps the relative weight K even for a singleton batch.
    mean_weight = sum(row.get("loss_weight", 1) for row in rows) / len(rows)
    if optimizer is not None:
        optimizer.zero_grad(set_to_none=True)
    for video_id, examples in ordered_videos(rows, seed, optimizer is not None):
        video = Path(examples[0]['video_path']).resolve(strict=True)
        with closing(EventIndex(cache / (video_id + '.sqlite'), video_id, index_source(video, model_path), read_only=True)) as index:
            for row in examples:
                if not 0 <= row['current_time'] <= index.eof:
                    raise ValueError('Decision outside the indexed video')
                pending.append(prepare_decision(index, row, towers, scorer, retrieval_batch_size=retrieval_batch_size))
                pending_weights.append(row.get("loss_weight", 1) / mean_weight)
                processed += 1
                if len(pending) == batch_size or processed == len(rows):
                    finish_update = processed % (batch_size * gradient_accumulation) == 0 or processed == len(rows)
                    last_loss = batch_step(pending, towers, decoder, optimizer, bf16=bf16,
                                                      endpoint_weights=pending_weights, loss_sums=loss_sums,
                                                      accumulation_start=accumulation_start, finish_update=finish_update)
                    total += len(pending) * last_loss
                    if progress is not None:
                        if optimizer is not None and finish_update:
                            progress.update(processed=processed, loss=last_loss)
                        elif optimizer is None:
                            progress.holdout_batch(processed, len(rows))
                    if finish_update and accumulation_start is not None:
                        accumulation_start.copy_(loss_sums[1])
                    pending.clear()
                    pending_weights.clear()
    if decoder.token_mean_ntp:
        total_ce, tokens = loss_sums.tolist()
        return total_ce / tokens
    if loss_sums is not None:
        split = "train" if optimizer is not None else "holdout"
        action, body = (loss_sums / len(rows)).tolist()
        metrics.update({f"{split}_action_ce": action, f"{split}_body_ce": body})
    return total / len(rows)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('model', 'qformer', 'run'):
        parser.add_argument('--' + name, type=Path, required=True)
    datasets = parser.add_mutually_exclusive_group(required=True)
    datasets.add_argument('--data', type=Path, help='Disjoint Stage3 training and holdout manifests')
    datasets.add_argument('--overfit-qa', type=Path, help='Audited 11-field QA release for memorization')
    parser.add_argument('--save-every', type=int, default=1, help='Checkpoint every N epochs and at the final epoch')
    parser.add_argument('--cpu-adamw', action='store_true', help='Keep FP32 AdamW state and updates on CPU')
    parser.add_argument('--bf16', action='store_true', help='BF16 autocast for the joint loss; keep FP32 optimizer parameters')
    parser.add_argument('--batch-size', type=int, default=1)
    parser.add_argument('--gradient-accumulation', type=int, help='Microbatches per token-mean NTP optimizer update')
    parser.add_argument('--retrieval-batch-size', type=int, default=4, help='Maximum candidate events per scorer forward')
    parser.add_argument('--epochs', type=int, default=1)
    parser.add_argument('--log-every-steps', type=int, default=10, help='Flush progress to stdout/JSONL every N optimizer updates')
    parser.add_argument('--log-every-seconds', type=float, default=30, help='Also flush on the next batch/update after this many seconds')
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--llm-lr', type=float, default=1e-5)
    parser.add_argument('--raw-recent-frames', type=int, choices=(0, 2, 4),
                        help='Direct frames: 2 selects 28+2, 4 selects 20+4, 0 compresses all 24.')
    parser.add_argument('--loss-objective', choices=['endpoint', 'action_body', 'ntp'],
                        help='Fresh QA overfit defaults to action_body. Resume uses the saved objective.')
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--resume', action='store_true', help='Continue after the last completed epoch')
    parser.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu')
    args = parser.parse_args()
    if min(args.epochs, args.batch_size, args.save_every, args.retrieval_batch_size, args.log_every_steps) < 1 or not all(math.isfinite(x) and x > 0 for x in [args.lr, args.llm_lr]):
        parser.error("Expected positive epochs and learning rates")
    if not math.isfinite(args.log_every_seconds) or args.log_every_seconds <= 0:
        parser.error("Logging interval must be finite and positive")
    data, run = (args.data or args.overfit_qa).resolve(), args.run.resolve()
    root = Path('/dss/mcmlscratch/04/ge75vid2/haoyu/live-ultrasound-video-understanding/cluster_data/experiments/event_retrieval')
    if root not in run.parents:
        parser.error(f"Choose a new run directory under {root}")
    objective = args.loss_objective or ('action_body' if args.overfit_qa else 'endpoint')
    raw_recent_frames = args.raw_recent_frames or 0
    if args.resume:
        if not (run / "config.json").is_file():
            parser.error("No saved training configuration to resume")
        previous = json.loads((run / "config.json").read_text())
        raw_recent_frames = previous.get('raw_recent_frames', 0)
        if args.raw_recent_frames is not None and args.raw_recent_frames != raw_recent_frames:
            parser.error("Requested recent frame layout differs from the checkpoint")
        objective = previous.get('loss_objective', 'endpoint')
        if args.loss_objective and args.loss_objective != objective:
            parser.error("Requested loss objective differs from the checkpoint")
    ga = args.gradient_accumulation if args.gradient_accumulation is not None else (previous.get('gradient_accumulation', 1) if args.resume else 1)
    if ga < 1 or (ga > 1 and objective != 'ntp'):
        parser.error("Gradient accumulation must be positive and requires NTP for values above one")
    world = int(os.environ.get("WORLD_SIZE", 1))
    rank = int(os.environ.get("RANK", 0))
    if world > 1:
        if objective != "ntp":
            parser.error("Distributed training requires --loss-objective ntp")
        if torch.device(args.device).type != "cuda":
            parser.error("Distributed joint training requires CUDA")
        local_rank = int(os.environ["LOCAL_RANK"])
        torch.cuda.set_device(local_rank)
        args.device = torch.device("cuda", local_rank)
        dist.init_process_group("nccl", device_id=args.device, timeout=timedelta(hours=4))
    if args.overfit_qa:
        weight_answers_by_wait = objective != "ntp"
        if args.resume:
            weight_answers_by_wait = previous.get("answer_weighting") == "wait_count"
        train, holdout = load_overfit(data, weight_answers_by_wait=weight_answers_by_wait), None
        manifests = [data / 'audit.json', *sorted(data.glob('*.jsonl'))]
    else:
        train, holdout = load_split(data, 'train'), load_split(data / 'holdout', 'holdout')
        if {r['video_id'] for r in train} & {r['video_id'] for r in holdout}:
            parser.error("Train and holdout videos overlap")
        manifests = [folder / name for folder in (data, data / 'holdout')
                     for name in ('samples.jsonl', 'tasks.jsonl', 'video_map.json')]
    config = dict(model=str(args.model.resolve(strict=True)), qformer=str(args.qformer.resolve(strict=True)),
                  data=str(data), lr=args.lr, llm_lr=args.llm_lr, seed=args.seed, batch_size=args.batch_size,
                  manifests={str(p.relative_to(data)): digest(p) for p in manifests})
    if world > 1:
        config.update(world_size=world, effective_batch_size=args.batch_size * world * ga)
    if ga > 1:
        config["gradient_accumulation"] = ga
    if raw_recent_frames:
        config['raw_recent_frames'] = raw_recent_frames
        if not args.resume or 'architecture_version' in previous:
            config['architecture_version'] = 'v3' if raw_recent_frames == 2 else 'v2'
    if objective != 'endpoint':
        config['loss_objective'] = objective
    if args.overfit_qa:
        config['mode'] = 'overfit'
        if objective == 'ntp':
            config['answer_weighting'] = 'none'
        if weight_answers_by_wait:
            config['answer_weighting'] = 'wait_count'
    if args.retrieval_batch_size != 4:
        config['retrieval_batch_size'] = args.retrieval_batch_size
    if args.cpu_adamw:
        config["cpu_adamw"] = True
    if args.bf16:
        config["bf16"] = True
    if args.resume:
        if not (run / 'latest.pt').is_file():
            parser.error("No completed epoch to resume")
    elif rank == 0:
        run.mkdir(parents=True, exist_ok=False)
    if world > 1:
        dist.barrier()
    cache = run / 'index'
    cache.mkdir(exist_ok=True)
    torch.manual_seed(args.seed)
    backbone = Qwen35Backbone(args.model).to(args.device).eval()
    towers = EventQFormers.from_instructblip(args.qformer, vision_width=backbone.vision_width,
                                             llm_width=backbone.llm_width, raw_recent_frames=raw_recent_frames).to(args.device)
    decoder = AnswerDecoder(backbone, separate_action_loss=objective == "action_body", token_mean_ntp=objective == "ntp")
    optimizer = joint_optimizer(backbone, towers, args.lr, args.llm_lr)
    if args.cpu_adamw:
        optimizer = distributed_train.CPUAdamW(optimizer)
    completed = 0
    if args.resume:
        completed = restore(run / 'latest.pt', towers, backbone, optimizer, config)
    elif rank == 0:
        backbone.tokenizer.save_pretrained(run / 'tokenizer')
        (run / 'config.json').write_text(json.dumps(config, indent=2) + '\n')
    with TrainingProgress(run, examples=len(train), batch_size=args.batch_size, world_size=world,
                          accumulation=ga, epochs=args.epochs, completed_epochs=completed,
                          holdout=holdout is not None, save_every=args.save_every,
                          log_every=args.log_every_steps, log_seconds=args.log_every_seconds, enabled=rank == 0) as progress:
        progress.phase('indexing')
        if rank == 0:
            prepare_indexes(train + (holdout or []), cache, root / 'shared_video_indexes', args.model, backbone)
        if world > 1:
            dist.barrier()
            progress.phase('distributed_setup')
            model = distributed_train.wrap_joint_loss(towers, decoder, batch_loss, args.device)
        for epoch in range(completed + 1, args.epochs + 1):
            progress.start_epoch(epoch)
            metrics = dict(epoch=epoch, train_endpoints=len(train),
                           optimizer_steps=epoch * math.ceil(len(train) / (args.batch_size * world * ga)))
            if world > 1:
                torch.manual_seed(args.seed + epoch * world + rank)
                ordered = [r for _, group in ordered_videos(train, args.seed + epoch, True) for r in group]
                metrics["train_loss"] = distributed_train.train_epoch(
                    ordered, cache, args.model, model, optimizer, prepare_decision, args.batch_size, ga,
                    args.bf16, args.retrieval_batch_size, on_update=progress.update if rank == 0 else None)
            else:
                metrics["train_loss"] = run_split(train, cache, args.model, towers, decoder, optimizer, args.seed + epoch, args.batch_size, args.bf16,
                                       retrieval_batch_size=args.retrieval_batch_size, metrics=metrics, gradient_accumulation=ga, progress=progress)
            if rank == 0 and holdout is not None:
                progress.phase('holdout')
                metrics['holdout_loss'] = run_split(holdout, cache, args.model, towers, decoder,
                                                    batch_size=args.batch_size, bf16=args.bf16,
                                                    retrieval_batch_size=args.retrieval_batch_size, metrics=metrics, progress=progress)
                metrics['holdout_endpoints'] = len(holdout)
            if rank == 0 and (epoch % args.save_every == 0 or epoch == args.epochs):
                progress.phase('checkpoint')
                save(run / 'latest.pt', towers, backbone, optimizer, epoch, config)
            if rank == 0:
                progress.phase('epoch_complete')
                metrics['optimizer_steps'] = progress.step
                print(json.dumps(metrics), flush=True)
            if world > 1:
                dist.barrier()
        if world > 1:
            dist.destroy_process_group()
        progress.finish()



if __name__ == '__main__':
    main()
