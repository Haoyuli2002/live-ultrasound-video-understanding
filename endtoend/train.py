"""Run a frozen QA experiment with measured batch selection and optional holdout."""
import argparse
import gc
import json
import math
import os
import random
import sys
import traceback
import time
from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageOps
from .data import causal_indices, load_examples, prompt, sample_times
from .model import forward_batch, predict_batch
from .optimization import backward_objectives, replay_updates, task_batches
from .answer_metrics import endpoint_answer_metrics, first_trigger_metrics


def require_scratch_run(path):
    """Resolve links before allowing production checkpoint output on this server."""
    run = Path(path).resolve()
    scratch = Path('/dss/mcmlscratch/04/ge75vid2').resolve()
    if not run.is_relative_to(scratch):
        raise ValueError(
            f'Training outputs must be on scratch, got {run}. '
            'Prepare a new run under cluster_data/experiments/streamready/.')
    return run


def write_json(path, value):
    path = Path(path)
    temporary = path.with_name(path.name + f'.partial.{os.getpid()}')
    temporary.write_text(json.dumps(value, indent=2) + '\n')
    temporary.replace(path)


def append_json(path, value):
    with Path(path).open('a') as stream:
        stream.write(json.dumps(value) + '\n')


def memory_stats():
    free, total = torch.cuda.mem_get_info()
    return dict(allocated_gib=torch.cuda.memory_allocated() / 2**30,
                reserved_gib=torch.cuda.memory_reserved() / 2**30,
                peak_allocated_gib=torch.cuda.max_memory_allocated() / 2**30,
                peak_reserved_gib=torch.cuda.max_memory_reserved() / 2**30,
                free_gib=free / 2**30, total_gib=total / 2**30)


def video_array(batch):
    """LAVIS sets decord's process-global bridge to torch during import."""
    if torch.is_tensor(batch):
        return batch.detach().cpu().numpy()
    return batch.asnumpy()


@torch.no_grad()
def prepare_features(model, examples, run, size, chunk, cache_mode="endpoint_memory"):
    if cache_mode not in ("endpoint_memory", "video_mmap"):
        raise ValueError(f"Unknown feature cache mode: {cache_mode}")
    from decord import VideoReader, cpu
    if cache_mode == "video_mmap":
        from .feature_cache import VideoFeatureCache
        cache = VideoFeatureCache(run / "features")
    else:
        cache = {}
    audit = []
    mean = torch.tensor([.48145466, .4578275, .40821073]).view(3, 1, 1)
    std = torch.tensor([.26862954, .26130258, .27577711]).view(3, 1, 1)
    for vid in sorted({e['video_id'] for e in examples}):
        group = [e for e in examples if e['video_id'] == vid]
        times = sorted({t for e in group for t in sample_times(e['current_time'])})
        reader = VideoReader(group[0]['video_path'], ctx=cpu(0), num_threads=2)
        pts = reader.get_frame_timestamp(np.arange(len(reader)))[:, 0].tolist()
        end = float(reader.get_frame_timestamp(len(reader) - 1)[1])
        if times[-1] > end:
            raise ValueError(f'Decision exceeds video duration: {vid}')
        indices = causal_indices(pts, times)
        features = []
        for start in range(0, len(indices), chunk):
            frames = video_array(reader.get_batch(indices[start:start + chunk]))
            pixels = []
            for frame in frames:
                image = ImageOps.pad(Image.fromarray(frame), (size, size), method=Image.Resampling.BICUBIC)
                x = torch.from_numpy(np.array(image, copy=True)).permute(2, 0, 1).float() / 255
                pixels.append((x - mean) / std)
            tensor = torch.stack(pixels, dim=1).unsqueeze(0).cuda()
            features.append(model.backbone.encode_frames(tensor).cpu())
        all_features = torch.cat(features, dim=1)
        time_to_index = {t: i for i, t in enumerate(times)}
        endpoint_indices = {}
        for e in group:
            selected = [time_to_index[t] for t in sample_times(e['current_time'])]
            if cache_mode == 'video_mmap':
                endpoint_indices[e['sample_id']] = selected
            else:
                cache[e['sample_id']] = all_features[:, selected].contiguous()
            # Each sample uses its own fixed 1 fps grid plus its endpoint, so
            # other QA annotations cannot change the frames it observes.
        if cache_mode == 'video_mmap':
            cache.add_video(vid, all_features, endpoint_indices)
        audit.append(dict(video_id=vid, requested_times=times,
                          actual_pts=[pts[i] for i in indices], frame_indices=indices,
                          max_observed_time=times[-1], source_duration=end,
                          feature_shape=list(all_features.shape)))
        print('FEATURES', vid, list(all_features.shape), flush=True)
    if cache_mode == 'video_mmap':cache.write_manifest()
    write_json(run / 'frame_audit.json', audit)
    return cache


def gradients(model):
    groups = {'short': model.core.short_qformer.parameters(),
              'long': model.core.long_qformer.parameters(),
              'vision_projection': model.core.vision_proj.parameters(),
              'output_projection': model.core.output_proj.parameters(),
              'readiness': model.readiness_parameters()}
    values = {}
    for name, parameters in groups.items():
        norms = [p.grad.detach().float().norm() for p in parameters if p.grad is not None]
        values[name] = float(torch.stack(norms).norm()) if norms else 0.
    if any(p.grad is not None for p in model.backbone.parameters()):
        raise RuntimeError('Frozen backbone unexpectedly received gradients')
    if not all(math.isfinite(v) for v in values.values()):
        raise RuntimeError('Nonfinite gradient')
    return values


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run-dir', required=True, type=Path)
    args = parser.parse_args()
    run = require_scratch_run(args.run_dir)
    cfg = json.loads((run / 'experiment.json').read_text())
    if cfg.get('readiness_recipe_selection'):
        from .full_data import resolve_recipe
        cfg = resolve_recipe(cfg)
        if int(os.environ.get('RANK', '0')) == 0:
            write_json(run / 'resolved_experiment.json', cfg)
    from . import distributed as dp
    rank, world = dp.initialize(cfg.get('world_size', 1))
    input_run = run
    if world > 1:
        if cfg.get('gradient_strategy') != 'readiness_isolated' or cfg.get('timing_pair_mode') != 'all_pairs':
            raise ValueError('Distributed full training requires isolated readiness and all-pairs configuration')
        if rank != 0:
            run = input_run / 'ranks' / f'rank_{rank}'
            run.mkdir(parents=True, exist_ok=True)
    if cfg.get('gradient_accumulation_steps', 1) != 1:
        raise ValueError('This trainer currently supports GA1 only')
    planned_batches = (json.loads((input_run / 'planned_batch_order.json').read_text())
                       if cfg.get('verify_planned_batches', False) else None)
    video_block_size = cfg.get('video_block_size', 0)
    profile_only = cfg.get('profile_only', False)
    if profile_only and not (cfg.get('init_checkpoint') and cfg.get('restore_optimizer')):
        raise ValueError('Read-only profiling requires a checkpoint with restored optimizer state')
    write_json(run / 'status.json', {'status': 'initializing'})
    random.seed(cfg['seed']); np.random.seed(cfg['seed']); torch.manual_seed(cfg['seed'])
    cpu_threads = int(cfg.get('cpu_threads_per_rank', 4))
    if cpu_threads < 1:raise ValueError('cpu_threads_per_rank must be positive')
    torch.set_num_threads(cpu_threads)
    examples = load_examples(input_run / 'samples.jsonl', input_run / 'tasks.jsonl', input_run / 'video_map.json')
    from .monitoring import load_holdout, evaluation_state, evaluation_due, token_mean_answer_ce
    holdout_examples = load_holdout(input_run, cfg, examples)
    evaluation_examples = holdout_examples or examples
    evaluation_dir = input_run / cfg['holdout']['directory'] if holdout_examples else input_run
    from lavis.models.blip2_models.streamready import StreamReady
    from omegaconf import OmegaConf
    model_cfg = dict(cfg['model'])
    if cfg.get('init_checkpoint'):
        model_cfg.update(load_pretrained=False, load_finetuned=False)
    model = StreamReady.from_config(OmegaConf.create(model_cfg)).cuda()
    if world > 1:
        from .comparison_controls import state_fingerprint
        fingerprint = state_fingerprint(model.core)
        hashes = [None] * world
        torch.distributed.all_gather_object(hashes, fingerprint['sha256'])
        if len(set(hashes)) != 1:raise RuntimeError('Distributed initial core weights differ')
        write_json(run / 'distributed_initial_state.json', dict(core=fingerprint, all_rank_hashes=hashes))
    model.parallel_endpoints = cfg.get('parallel_endpoints', False)
    model.readiness_objective = cfg.get('readiness_objective', 'bce')
    model.timing_pair_mode = cfg.get('timing_pair_mode', 'sampled')
    if model.timing_pair_mode not in ('sampled', 'all_pairs'):
        raise ValueError(f'Unknown timing_pair_mode: {model.timing_pair_mode}')
    model.timing_lambda_reg = cfg.get('timing_lambda_reg', .1)
    model.readiness_bce_positive_weight = cfg.get('readiness_bce_positive_weight', 2.)
    model.readiness_bce_weight = cfg.get('readiness_bce_weight', 1.)
    if model.readiness_objective in ('contrastive_gold', 'contrastive_bce_gold') and cfg.get('sampler') != 'question_groups':
        raise ValueError('Contrastive timing requires complete question groups')
    if cfg.get('fast_linear_attention', False):
        from .acceleration import enable_fast_attention
        write_json(run / 'attention_backend.json', enable_fast_attention(model))
    if not model_cfg.get('gradient_checkpointing', False):
        for module in model.modules():
            if hasattr(module, 'gradient_checkpointing'):
                module.gradient_checkpointing = False
    write_json(run / 'execution_mode.json', dict(parallel_endpoints=model.parallel_endpoints,
               gradient_checkpointing=model_cfg.get('gradient_checkpointing', False),
               active_checkpoint_modules=[n for n, m in model.named_modules()
                                          if getattr(m, 'gradient_checkpointing', False)]))
    optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad],
                                 lr=cfg['learning_rate'], weight_decay=cfg['weight_decay'])
    if cfg.get('comparison_initial_manifest'):
        if cfg.get('init_checkpoint') or cfg.get('restore_optimizer'):
            raise ValueError('Fresh comparison cannot restore a trained checkpoint/optimizer')
        from .comparison_controls import verify_initial_state
        verify_initial_state(model.core, optimizer, run / 'initial_state.json', cfg['comparison_initial_manifest'])
    if cfg.get('init_checkpoint'):
        checkpoint = torch.load(cfg['init_checkpoint'], map_location='cpu', weights_only=True, mmap=True)
        restored = model.load_state_dict(checkpoint['model'], strict=False)
        if restored.unexpected_keys or any(not k.startswith('backbone.') for k in restored.missing_keys):
            raise RuntimeError(f'Incomplete core restoration: {restored}')
        if cfg.get('restore_optimizer', False):
            optimizer.load_state_dict(checkpoint['optimizer'])
            for group in optimizer.param_groups:
                group['lr'] = cfg['learning_rate']
                group['weight_decay'] = cfg['weight_decay']
        write_json(run / 'restored_checkpoint.json', dict(path=cfg['init_checkpoint'], epoch=checkpoint['epoch'],
                                                        optimizer_restored=cfg.get('restore_optimizer', False)))
        del checkpoint
    write_json(run / 'parameters.json', dict(trainable=sum(p.numel() for p in model.parameters() if p.requires_grad),
                                            frozen=sum(p.numel() for p in model.parameters() if not p.requires_grad)))
    model.eval()
    cache_mode = cfg.get('feature_cache_mode', 'endpoint_memory')
    if world > 1:
        from .feature_cache import prepare_distributed_features
        cache = prepare_distributed_features(prepare_features, model, examples, input_run,
                    cfg['image_size'], cfg['vision_chunk'], rank, world)
        evaluation_cache = (prepare_distributed_features(prepare_features, model, holdout_examples,
                    evaluation_dir, cfg['image_size'], cfg['vision_chunk'], rank, world)
                    if holdout_examples else cache)
    else:
        cache = prepare_features(model, examples, run, cfg['image_size'], cfg['vision_chunk'], cache_mode)
        evaluation_cache = (prepare_features(model, holdout_examples, evaluation_dir, cfg['image_size'],
                            cfg['vision_chunk'], cache_mode) if holdout_examples else cache)
    if cfg.get('validate_parallel', False):
        # At the same restored weights, validate real causal endpoints before
        # profiling. Numerical tolerance permits BF16 batched kernel rounding.
        rows = sorted(examples, key=lambda e: e['current_time'])[::max(1, len(examples) // 8)][:8]
        with torch.no_grad(), torch.autocast('cuda', dtype=torch.bfloat16):
            model.parallel_endpoints = False
            serial = forward_batch(model, rows, [cache[e['sample_id']].cuda() for e in rows])
            model.parallel_endpoints = True
            parallel = forward_batch(model, rows, [cache[e['sample_id']] for e in rows])
        serial_prefix = torch.cat([o['prefix_embeds'] for o in serial['outputs']])
        parallel_prefix = torch.cat([o['prefix_embeds'] for o in parallel['outputs']])
        torch.testing.assert_close(parallel_prefix, serial_prefix, atol=.04, rtol=.04)
        torch.testing.assert_close(parallel['logits'], serial['logits'], atol=.04, rtol=.04)
        write_json(run / 'parallel_validation.json', dict(status='passed', samples=len(rows),
                   prefix_max_abs_difference=float((parallel_prefix - serial_prefix).abs().max()),
                   readiness_max_abs_difference=float((parallel['logits'] - serial['logits']).abs().max()),
                   replay_stats=model.core.last_replay_stats))
        del serial, parallel, serial_prefix, parallel_prefix

    def features(rows, source_cache=None):
        source_cache = cache if source_cache is None else source_cache
        unique = {e['sample_id']: e for e in rows}
        gathered = {sid: source_cache[sid] for sid in unique}
        if not model.parallel_endpoints:
            gathered = {sid: value.cuda() for sid, value in gathered.items()}
        return [gathered[e['sample_id']] for e in rows]

    def loss_pass(rows):
        with torch.autocast('cuda', dtype=torch.bfloat16):
            out = forward_batch(model, rows, features(rows))
        if not torch.isfinite(out['loss']):
            raise RuntimeError('Nonfinite loss')
        return out

    def backward(out):
        return backward_objectives(model, out, cfg.get('gradient_strategy', 'readiness_isolated'))

    def evaluate(epoch):
        started = time.time()
        with evaluation_state(model):
            metrics = evaluate_rows(epoch)
        if rank == 0:
            append_json(run / 'phase_intervals.jsonl', dict(phase='holdout' if holdout_examples else 'evaluation',
                        epoch=epoch, step=step, started_unix=started, ended_unix=time.time()))
            if cfg.get('plot_monitoring', False):
                from .training_plots import render_plots
                render_plots(run)
        return metrics

    def evaluate_rows(epoch):
        records, losses, eval_logits, answer_losses = [], [], [], []
        for row in evaluation_examples[rank::world]:
            with torch.autocast('cuda', dtype=torch.bfloat16):
                result, out = predict_batch(model, [row], features([row], evaluation_cache), cfg['max_new_tokens'], decode_all=True)
            if row['ready']:
                tokens = model.backbone.tokenizer(row['answer'] + model.backbone.tokenizer.eos_token,
                            add_special_tokens=False, truncation=True, max_length=model.max_output_txt_len)['input_ids']
                result[0].update(answer_token_count=len(tokens), answer_token_nll=float(out['answer_loss']) * len(tokens))
            records.extend(result)
            # Keep this historical metric comparable; BCE is diagnostic only
            # when training with contrastive timing.
            losses.append(float(out['answer_loss'] + model.readiness_loss_weight * out['readiness_bce']))
            eval_logits.append(out['logits'].detach().float().cpu())
            if row['ready']: answer_losses.append(float(out['answer_loss']))
        if world > 1:
            records, losses, eval_logits, answer_losses = dp.gather_evaluation(
                records, losses, eval_logits, answer_losses, evaluation_examples)
        # Overfit-only endpoint metrics; no claim about unseen videos or dense latency.
        positives = [r for r in records if r['expected_ready']]
        negatives = [r for r in records if not r['expected_ready']]
        pairs = [(a, b) for a in negatives for b in positives if a['task_id'] == b['task_id']]
        auc = sum((a['probability'] > b['probability']) + .5 * (a['probability'] == b['probability'])
                  for a in positives for b in negatives) / max(1, len(positives) * len(negatives))
        metrics = dict(epoch=epoch, mean_endpoint_loss=sum(losses) / len(losses),
                       readiness_accuracy=sum(r['ready'] == bool(r['expected_ready']) for r in records) / len(records),
                       premature_rate=sum(r['ready'] for r in negatives) / max(1, len(negatives)),
                       answer_exact_match=sum(r['answer'].strip().lower() == r['reference'].strip().lower() for r in positives) / max(1, len(positives)),
                       answer_recall=sum(r['ready'] for r in positives) / max(1, len(positives)),
                       wait_false_trigger_count=sum(r['ready'] for r in negatives),
                       answer_missed_count=sum(not r['ready'] for r in positives),
                       readiness_auc=auc,
                       paired_order_accuracy=sum(b['probability'] > a['probability'] for a, b in pairs) / max(1, len(pairs)),
                       gated_answer_exact_match=sum(r['ready'] and r['answer'].strip().lower() == r['reference'].strip().lower()
                                                    for r in positives) / max(1, len(positives)),
                       scope=('holdout_videos_endpoints_forced_answer_probe' if holdout_examples
                              else 'same_training_videos_endpoints_forced_answer_probe'),
                       split='holdout' if holdout_examples else 'train', optimizer_step=step)
        metrics.update(endpoint_answer_metrics(records, model.readiness_threshold))
        first_trigger = first_trigger_metrics(records, model.readiness_threshold)
        metrics['sparse_first_trigger'] = {k: v for k, v in first_trigger.items() if k != 'records'}
        from .timing_loss import contrastive_timing_loss, readiness_loss_terms
        full_logits = torch.cat(eval_logits)
        timing = contrastive_timing_loss(full_logits, evaluation_examples, model.timing_lambda_reg, sample_pairs=False)
        terms = readiness_loss_terms(full_logits, evaluation_examples, model.readiness_objective,
                                     model.timing_lambda_reg, model.readiness_bce_positive_weight,
                                     model.readiness_bce_weight, sample_pairs=False)
        bce = float(terms['bce'])
        sequence_ce = sum(answer_losses) / max(1, len(answer_losses))
        ce = token_mean_answer_ce(records)
        metrics.update(answer_ce=ce, answer_ce_sequence_mean=sequence_ce, answer_ce_normalization="mean over supervised answer tokens", readiness_bce=bce,
                       readiness_weighted_bce=float(terms['weighted_bce']),
                       readiness_bce_positive_weight=model.readiness_bce_positive_weight,
                       readiness_bce_weight=model.readiness_bce_weight,
                       readiness_objective_loss=float(terms['loss']),
                       readiness_contrastive=float(timing['contrastive']), readiness_temporal=float(timing['temporal']),
                       readiness_timing_objective=float(timing['loss']), timing_pairs=timing['pairs'],
                       timing_paired_tasks=timing['paired_tasks'], readiness_objective=model.readiness_objective,
                       mean_endpoint_loss_definition='mean per-endpoint answer CE (WAIT=0) + weight * gold BCE; diagnostic, not contrastive training objective',
                       training_objective_fullset=ce + model.readiness_loss_weight * float(terms['loss']))
        metrics.update(answer_trigger_rate=metrics['answer_recall'], wait_correct_rate=1.-metrics['premature_rate'],
                       timing_and_answer_correct_rate=first_trigger['on_time_and_text_correct']/max(1,first_trigger['answerable_tasks']),
                       forced_answer_correct_rate=metrics['answer_accuracy'], threshold=model.readiness_threshold)
        write_json(run / f'eval_step_{step:06d}.json', {'metrics': metrics, 'predictions': records})
        if not cfg.get('eval_every_steps') or epoch == 0 or step == cfg.get('expected_optimizer_steps'):
            write_json(run / f'eval_epoch_{epoch:03d}.json', {'metrics': metrics, 'predictions': records})
        append_json(run / 'eval_metrics.jsonl', metrics)
        print('EVAL', json.dumps(metrics), flush=True)
        if holdout_examples:append_json(run / 'holdout_metrics.jsonl', metrics)
        return metrics

    def save(epoch, batch):
        if rank != 0:return
        # Backbone remains pinned externally. Save every trainable module and Adam state.
        model.core.memory.reset()
        torch.save({'model': {k: v for k, v in model.state_dict().items() if k.startswith('core.')},
                    'optimizer': optimizer.state_dict(), 'epoch': epoch, 'step': step, 'batch_size': batch,
                    'config': cfg}, run / 'checkpoint_latest.pth.partial')
        (run / 'checkpoint_latest.pth.partial').replace(run / 'checkpoint_latest.pth')

    step = 0
    if not profile_only:
        evaluate(0)
    model.train()
    # One real warmup update allocates Adam state before any memory comparisons.
    if cfg.get('zero_update_warmup', False):
        from .comparison_controls import preallocate_adamw_state
        preallocate_adamw_state(optimizer)
        write_json(run / 'optimizer_initialization.json', dict(step=0, method='zero moments; no optimizer.step',
                   parameters_with_state=len(optimizer.state), extra_warmup_updates=0))
    elif not cfg.get('restore_optimizer', False):
        warmup = [max((e for e in examples if e['ready']), key=lambda e: len(sample_times(e['current_time'])))]
        torch.cuda.reset_peak_memory_stats()
        out = loss_pass(warmup)
        balance = backward(out)
        warmup_grad = gradients(model)
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1., error_if_nonfinite=True)
        optimizer.step(); optimizer.zero_grad(set_to_none=True)
        append_json(run / 'training.jsonl', dict(phase='warmup_update', loss=float(out['loss'].detach()),
                                               gradients=warmup_grad, gradient_balance=balance, **memory_stats()))
        del out

    # Conservative probes duplicate the worst answer-bearing row so every decoder
    # slot has a long prefix and a full answer, independent of WAIT/ANSWER balance.
    positive_rows = [e for e in examples if e['ready']]
    # A synthetic shape-only probe combines the longest history, question and
    # answer. It performs backward but never updates weights or contributes to
    # reported training/evaluation losses.
    worst = dict(max(examples, key=lambda e: len(sample_times(e['current_time']))))
    count_tokens = lambda text: len(model.backbone.tokenizer(text, add_special_tokens=False).input_ids)
    worst.update(ready=1, question=max(examples, key=lambda e: count_tokens(e['question']))['question'],
                 answer=max(positive_rows, key=lambda e: count_tokens(e['answer']))['answer'])
    profiles = []
    repeats = int(cfg.get('profile_repeats', 2))
    warmups = int(cfg.get('profile_warmups', 0))
    if repeats < 1 or warmups < 0:
        raise ValueError('Invalid profile repetition counts')
    rng_cpu, rng_cuda = torch.get_rng_state(), torch.cuda.get_rng_state()
    probe_plan = [(bs, False) for bs in cfg['batch_candidates']]
    if cfg.get('diagnose_synthetic_batch', False):
        if not (cfg.get('profile_real_batches') and cfg.get('profile_additional_real_batches')):
            raise ValueError('Synthetic diagnostic requires real-batch memory probes')
        probe_plan = [(bs, diagnostic) for bs in cfg['batch_candidates'] for diagnostic in (True, False)]
    for bs, diagnostic in probe_plan:
        synthetic = diagnostic or not cfg.get('profile_real_batches', False)
        probe_rows = ([worst] * bs if synthetic
                      else task_batches(examples, bs, cfg['seed'], video_block_size)[0])
        gc.collect(); torch.cuda.empty_cache(); torch.cuda.reset_peak_memory_stats()
        started = time.monotonic()
        try:
            for _ in range(warmups):
                out = loss_pass(probe_rows)
                backward(out)
                optimizer.zero_grad(set_to_none=True)
                del out
            torch.cuda.synchronize()
            started = time.monotonic()
            timed_start_unix = time.time()
            for _ in range(repeats):
                out = loss_pass(probe_rows)
                backward(out)
                optimizer.zero_grad(set_to_none=True)
                del out
            torch.cuda.synchronize()
            elapsed = time.monotonic() - started
            record = dict(batch_size=bs, status='passed', seconds_per_batch=elapsed / repeats,
                          actual_samples=len(probe_rows),
                          samples_per_second=len(probe_rows) * repeats / elapsed, warmups=warmups,
                          timed_repetitions=repeats, timed_start_unix=timed_start_unix,
                          timed_end_unix=time.time(),
                          replay_stats=getattr(model.core, 'last_replay_stats', None), **memory_stats())
            record['safe'] = (record['peak_reserved_gib'] < .8 * record['total_gib']
                              and record['free_gib'] > .15 * record['total_gib'])
        except torch.cuda.OutOfMemoryError:
            optimizer.zero_grad(set_to_none=True)
            if 'out' in locals():
                del out
            gc.collect(); torch.cuda.empty_cache()
            record = dict(batch_size=bs, status='oom', safe=False)
        if diagnostic:
            append_json(run / 'synthetic_stress.jsonl', dict(record, workload='synthetic_all_answer'))
            continue
        profiles.append(record)
        write_json(run / 'batch_profiles.json', profiles)
        print('BATCH_PROBE', json.dumps(record), flush=True)
        if not record['safe']:
            break
    torch.set_rng_state(rng_cpu); torch.cuda.set_rng_state(rng_cuda)
    safe = [p for p in profiles if p.get('safe')]
    if not safe:
        raise RuntimeError('No batch size has the configured GPU safety margin')
    chosen = max(safe, key=lambda p: p['samples_per_second'])
    if world > 1:
        chosen_sizes = [None] * world
        torch.distributed.all_gather_object(chosen_sizes, chosen['batch_size'])
        if len(set(chosen_sizes)) != 1:raise RuntimeError('Per-rank selected batch sizes differ')
    batch = chosen['batch_size']
    if cfg.get('profile_additional_real_batches', False):
        # A duplicated endpoint shares one video replay, so also probe actual
        # batches with the most distinct observed frames and decoder padding.
        probe_epochs = cfg['epochs'] if cfg.get('probe_all_planned_epochs') else 1
        candidates = [rows for epoch in range(1, probe_epochs + 1)
                      for rows in task_batches(examples, batch, cfg['seed'] + epoch, video_block_size)]
        def decoder_cost(rows):
            answers = [row for row in rows if row['ready']]
            return len(answers) * (max((count_tokens(prompt(row)) for row in answers), default=0)
                                  + max((count_tokens(row['answer']) for row in answers), default=0))
        real_probes = [max(candidates, key=replay_updates), max(candidates, key=decoder_cost)]
        real_profiles = []
        with torch.random.fork_rng(devices=[torch.cuda.current_device()]):
            for probe_rows in real_probes:
                gc.collect();torch.cuda.empty_cache();torch.cuda.reset_peak_memory_stats()
                started = time.monotonic()
                out = loss_pass(probe_rows);backward(out)
                optimizer.zero_grad(set_to_none=True);del out
                torch.cuda.synchronize()
                stats = memory_stats()
                safe_probe = stats['peak_reserved_gib'] < .8 * stats['total_gib'] and stats['free_gib'] > .15 * stats['total_gib']
                real_profiles.append(dict(sample_ids=[r['sample_id'] for r in probe_rows],
                    unique_observed_frames=replay_updates(probe_rows),decoder_cost=decoder_cost(probe_rows),
                    seconds=time.monotonic()-started,safe=safe_probe,**stats))
                write_json(run / 'real_batch_profiles.json', real_profiles)
                if not safe_probe:raise RuntimeError('Full-data real batch exceeds GPU safety margin')
    write_json(run / 'batch_decision.json', dict(batch_size=batch, batch_size_per_gpu=batch,
                                                global_batch_capacity=batch * world, world_size=world, reason='highest measured throughput among safe probes',
                                                profiles=profiles, backbone_frozen=True,
                                                variable_length_qformer=('batched with masked windows and shared causal replay'
                                                                        if model.parallel_endpoints else 'serial per sample; decoder batched')))
    if profile_only:
        write_json(run / 'status.json', dict(status='completed', mode='profile_only',
                                             optimizer_steps=0, checkpoint_modified=False,
                                             recommended_batch_size=batch))
        return
    best_monitor_key = None
    last_eval_step = 0
    def monitor_checkpoint(epoch):
        nonlocal best_monitor_key, last_eval_step
        metrics = evaluate(epoch)
        last_eval_step = step
        if rank == 0 and holdout_examples and cfg.get('save_best_holdout', False):
            key = (metrics['sparse_first_trigger']['on_time_and_text_correct'],
                   -metrics['premature_rate'], metrics['readiness_accuracy'])
            if best_monitor_key is None or key > best_monitor_key:
                import shutil
                save(epoch, batch)
                shutil.copyfile(run / 'checkpoint_latest.pth', run / 'checkpoint_best.pth.partial')
                (run / 'checkpoint_best.pth.partial').replace(run / 'checkpoint_best.pth')
                best_monitor_key = key
                write_json(run / 'best_holdout.json', dict(epoch=epoch, step=step,
                    evaluation_file=f'eval_step_{step:06d}.json', threshold=model.readiness_threshold,
                    selection_key=list(key), metrics=metrics))
    for epoch in range(1, cfg['epochs'] + 1):
        if world > 1:
            global_batches = dp.rank_batches(examples, batch, cfg['seed'] + epoch, world, video_block_size)
            batches = [ranks[rank] for ranks in global_batches]
        elif cfg.get('sampler') == 'question_groups':
            batches = task_batches(examples, batch, cfg['seed'] + epoch, video_block_size)
        else:
            order = list(examples)
            random.Random(cfg['seed'] + epoch).shuffle(order)
            batches = [order[start:start + batch] for start in range(0, len(order), batch)]
        actual_order = [[e['sample_id'] for e in rows] for rows in batches]
        expected_order = (None if planned_batches is None else
                          [ranks[rank] for ranks in planned_batches[epoch - 1]] if world > 1
                          else planned_batches[epoch - 1])
        if expected_order is not None and actual_order != expected_order:
            raise RuntimeError('Actual training batch order differs from frozen update budget')
        append_json(run / 'batch_order.jsonl', {'epoch': epoch, 'batches': actual_order})
        for rows in batches:
            if cfg.get('matched_step_seed') is not None:
                # Reset forward/dropout RNG independently of contrastive pair draws in earlier steps.
                seed = int(cfg['matched_step_seed']) + step * world + rank
                random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
            started = time.monotonic()
            started_unix = time.time()
            torch.cuda.reset_peak_memory_stats()
            out = loss_pass(rows) if rows else None
            replay_stats = dict(model.core.last_replay_stats) if rows and model.parallel_endpoints else {}
            if world > 1:
                counts = dp.counts_for_output(model, out, rows, next(model.parameters()).device)
                total_counts = counts.clone();torch.distributed.all_reduce(total_counts)
                scaled = dp.scale_objectives(out, counts, total_counts, world, model.readiness_objective,
                                            model.timing_lambda_reg, model.readiness_bce_weight)
                balance = backward(scaled) if scaled is not None else dict(strategy='empty_rank_no_sample')
                dp.synchronize_gradients(model.parameters())
                out = dp.aggregate_metrics(out, counts, total_counts, model.readiness_objective,
                                           model.timing_lambda_reg, model.readiness_bce_weight, model.readiness_loss_weight)
                del scaled
            else:
                balance = backward(out)
            grad = gradients(model)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1., error_if_nonfinite=True)
            optimizer.step(); optimizer.zero_grad(set_to_none=True)
            torch.cuda.synchronize()
            step += 1
            record = dict(phase='train', epoch=epoch, step=step,
                          batch_size=int(total_counts[0]) if world > 1 else len(rows),
                          local_batch_size=len(rows), rank=rank, world_size=world,
                          local_replay=replay_stats,
                          loss=float(out['loss'].detach()), answer_loss=float(out['answer_loss'].detach()),
                          readiness_loss=float(out['readiness_loss'].detach()), gradients=grad,
                          gradient_balance=balance,
                          started_unix=started_unix, ended_unix=time.time(),
                          seconds=time.monotonic() - started, **memory_stats())
            record['readiness_bce'] = float(out['readiness_bce'].detach())
            record['readiness_weighted_bce'] = float(out['readiness_weighted_bce'].detach())
            record['readiness_bce_positive_weight'] = model.readiness_bce_positive_weight
            record['readiness_bce_weight'] = model.readiness_bce_weight
            record['readiness_objective'] = model.readiness_objective
            if out['timing'] is not None:
                record.update(readiness_contrastive=float(out['timing']['contrastive'].detach()),
                              readiness_temporal=float(out['timing']['temporal'].detach()),
                              timing_pairs=out['timing']['pairs'], timing_sampled_pairs=out['timing']['sampled_pairs'],
                              timing_used_pairs=out['timing']['used_pairs'], timing_pair_mode=out['timing']['pair_mode'],
                              timing_paired_tasks=out['timing']['paired_tasks'], timing_unpaired_tasks=out['timing']['unpaired_tasks'])
            append_json(run / 'training.jsonl', record)
            print('TRAIN', json.dumps(record), flush=True)
            del out
            if evaluation_due(cfg, step, epoch, last_step=last_eval_step):
                save(epoch, batch)
                monitor_checkpoint(epoch)
        if epoch % cfg.get('checkpoint_every', 1) == 0 or epoch == cfg['epochs']:
            save(epoch, batch)
        if evaluation_due(cfg, step, epoch, epoch_end=True, last_step=last_eval_step):
            monitor_checkpoint(epoch)
        write_json(run / 'status.json', dict(status='running', completed_epochs=epoch, step=step, batch_size=batch))
    if cfg.get('expected_optimizer_steps') is not None and step != cfg['expected_optimizer_steps']:
        raise RuntimeError('Unexpected number of optimizer updates')
    if world > 1:torch.distributed.barrier()
    write_json(run / 'status.json', dict(status='completed', completed_epochs=cfg['epochs'], step=step, batch_size=batch, world_size=world))
    if world > 1:torch.distributed.destroy_process_group()


if __name__ == '__main__':
    try:
        main()
    except Exception:
        if '--run-dir' in sys.argv:
            failed_run = Path(sys.argv[sys.argv.index('--run-dir') + 1])
            if failed_run.is_dir():
                write_json(failed_run / 'status.json', {'status': 'failed', 'traceback': traceback.format_exc()})
        raise
