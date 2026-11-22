"""Run a frozen three-video experiment, including measured GPU batch selection."""
import argparse
import gc
import json
import math
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
from .optimization import backward_objectives, task_batches
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
    Path(path).write_text(json.dumps(value, indent=2) + '\n')


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
def prepare_features(model, examples, run, size, chunk):
    from decord import VideoReader, cpu
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
        for e in group:
            selected = [time_to_index[t] for t in sample_times(e['current_time'])]
            cache[e['sample_id']] = all_features[:, selected].contiguous()
            # Each sample uses its own fixed 1 fps grid plus its endpoint, so
            # other QA annotations cannot change the frames it observes.
        audit.append(dict(video_id=vid, requested_times=times,
                          actual_pts=[pts[i] for i in indices], frame_indices=indices,
                          max_observed_time=times[-1], source_duration=end,
                          feature_shape=list(all_features.shape)))
        print('FEATURES', vid, list(all_features.shape), flush=True)
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
    profile_only = cfg.get('profile_only', False)
    if profile_only and not (cfg.get('init_checkpoint') and cfg.get('restore_optimizer')):
        raise ValueError('Read-only profiling requires a checkpoint with restored optimizer state')
    write_json(run / 'status.json', {'status': 'initializing'})
    random.seed(cfg['seed']); np.random.seed(cfg['seed']); torch.manual_seed(cfg['seed'])
    torch.set_num_threads(4)
    examples = load_examples(run / 'samples.jsonl', run / 'tasks.jsonl', run / 'video_map.json')
    from lavis.models.blip2_models.streamready import StreamReady
    from omegaconf import OmegaConf
    model_cfg = dict(cfg['model'])
    if cfg.get('init_checkpoint'):
        model_cfg.update(load_pretrained=False, load_finetuned=False)
    model = StreamReady.from_config(OmegaConf.create(model_cfg)).cuda()
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
    cache = prepare_features(model, examples, run, cfg['image_size'], cfg['vision_chunk'])
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

    def features(rows):
        if model.parallel_endpoints:
            return [cache[e['sample_id']] for e in rows]
        return [cache[e['sample_id']].cuda() for e in rows]

    def loss_pass(rows):
        with torch.autocast('cuda', dtype=torch.bfloat16):
            out = forward_batch(model, rows, features(rows))
        if not torch.isfinite(out['loss']):
            raise RuntimeError('Nonfinite loss')
        return out

    def backward(out):
        return backward_objectives(model, out, cfg.get('gradient_strategy', 'readiness_isolated'))

    def evaluate(epoch):
        model.eval()
        records, losses, eval_logits, answer_losses = [], [], [], []
        for row in examples:
            with torch.autocast('cuda', dtype=torch.bfloat16):
                result, out = predict_batch(model, [row], features([row]), cfg['max_new_tokens'], decode_all=True)
            records.extend(result)
            # Keep this historical metric comparable; BCE is diagnostic only
            # when training with contrastive timing.
            losses.append(float(out['answer_loss'] + model.readiness_loss_weight * out['readiness_bce']))
            eval_logits.append(out['logits'].detach().float().cpu())
            if row['ready']: answer_losses.append(float(out['answer_loss']))
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
                       readiness_auc=auc,
                       paired_order_accuracy=sum(b['probability'] > a['probability'] for a, b in pairs) / max(1, len(pairs)),
                       gated_answer_exact_match=sum(r['ready'] and r['answer'].strip().lower() == r['reference'].strip().lower()
                                                    for r in positives) / max(1, len(positives)),
                       scope='same_training_videos_endpoints_forced_answer_probe')
        metrics.update(endpoint_answer_metrics(records, model.readiness_threshold))
        first_trigger = first_trigger_metrics(records, model.readiness_threshold)
        metrics['sparse_first_trigger'] = {k: v for k, v in first_trigger.items() if k != 'records'}
        from .timing_loss import contrastive_timing_loss, readiness_loss_terms
        full_logits = torch.cat(eval_logits)
        timing = contrastive_timing_loss(full_logits, examples, model.timing_lambda_reg, sample_pairs=False)
        terms = readiness_loss_terms(full_logits, examples, model.readiness_objective,
                                     model.timing_lambda_reg, model.readiness_bce_positive_weight,
                                     model.readiness_bce_weight, sample_pairs=False)
        bce = float(terms['bce'])
        ce = sum(answer_losses) / max(1, len(answer_losses))
        metrics.update(answer_ce=ce, readiness_bce=bce,
                       readiness_weighted_bce=float(terms['weighted_bce']),
                       readiness_bce_positive_weight=model.readiness_bce_positive_weight,
                       readiness_bce_weight=model.readiness_bce_weight,
                       readiness_objective_loss=float(terms['loss']),
                       readiness_contrastive=float(timing['contrastive']), readiness_temporal=float(timing['temporal']),
                       readiness_timing_objective=float(timing['loss']), timing_pairs=timing['pairs'],
                       timing_paired_tasks=timing['paired_tasks'], readiness_objective=model.readiness_objective,
                       mean_endpoint_loss_definition='mean per-endpoint answer CE (WAIT=0) + weight * gold BCE; diagnostic, not contrastive training objective',
                       training_objective_fullset=ce + model.readiness_loss_weight * float(terms['loss']))
        write_json(run / f'eval_epoch_{epoch:03d}.json', {'metrics': metrics, 'predictions': records})
        append_json(run / 'eval_metrics.jsonl', metrics)
        print('EVAL', json.dumps(metrics), flush=True)
        model.train()

    def save(epoch, batch):
        # Backbone remains pinned externally. Save every trainable module and Adam state.
        model.core.memory.reset()
        torch.save({'model': {k: v for k, v in model.state_dict().items() if k.startswith('core.')},
                    'optimizer': optimizer.state_dict(), 'epoch': epoch, 'batch_size': batch,
                    'config': cfg}, run / 'checkpoint_latest.pth.partial')
        (run / 'checkpoint_latest.pth.partial').replace(run / 'checkpoint_latest.pth')

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
        warmup = [max((e for e in examples if e['ready']), key=lambda e: len(cache[e['sample_id']][0]))]
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
    worst = dict(max(examples, key=lambda e: cache[e['sample_id']].shape[1]))
    count_tokens = lambda text: len(model.backbone.tokenizer(text, add_special_tokens=False).input_ids)
    worst.update(ready=1, question=max(examples, key=lambda e: count_tokens(e['question']))['question'],
                 answer=max(positive_rows, key=lambda e: count_tokens(e['answer']))['answer'])
    profiles = []
    repeats = int(cfg.get('profile_repeats', 2))
    warmups = int(cfg.get('profile_warmups', 0))
    if repeats < 1 or warmups < 0:
        raise ValueError('Invalid profile repetition counts')
    rng_cpu, rng_cuda = torch.get_rng_state(), torch.cuda.get_rng_state()
    for bs in cfg['batch_candidates']:
        probe_rows = ([worst] * bs if not cfg.get('profile_real_batches', False)
                      else task_batches(examples, bs, cfg['seed'])[0])
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
    batch = chosen['batch_size']
    write_json(run / 'batch_decision.json', dict(batch_size=batch, reason='highest measured throughput among safe probes',
                                                profiles=profiles, backbone_frozen=True,
                                                variable_length_qformer=('batched with masked windows and shared causal replay'
                                                                        if model.parallel_endpoints else 'serial per sample; decoder batched')))
    if profile_only:
        write_json(run / 'status.json', dict(status='completed', mode='profile_only',
                                             optimizer_steps=0, checkpoint_modified=False,
                                             recommended_batch_size=batch))
        return
    step = 0
    for epoch in range(1, cfg['epochs'] + 1):
        if cfg.get('sampler') == 'question_groups':
            batches = task_batches(examples, batch, cfg['seed'] + epoch)
        else:
            order = list(examples)
            random.Random(cfg['seed'] + epoch).shuffle(order)
            batches = [order[start:start + batch] for start in range(0, len(order), batch)]
        append_json(run / 'batch_order.jsonl', {'epoch': epoch, 'batches': [[e['sample_id'] for e in rows] for rows in batches]})
        for rows in batches:
            if cfg.get('matched_step_seed') is not None:
                # Reset forward/dropout RNG independently of contrastive pair draws in earlier steps.
                seed = int(cfg['matched_step_seed']) + step
                random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
            started = time.monotonic()
            torch.cuda.reset_peak_memory_stats()
            out = loss_pass(rows)
            balance = backward(out)
            grad = gradients(model)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1., error_if_nonfinite=True)
            optimizer.step(); optimizer.zero_grad(set_to_none=True)
            torch.cuda.synchronize()
            step += 1
            record = dict(phase='train', epoch=epoch, step=step, batch_size=len(rows),
                          loss=float(out['loss'].detach()), answer_loss=float(out['answer_loss'].detach()),
                          readiness_loss=float(out['readiness_loss'].detach()), gradients=grad,
                          gradient_balance=balance,
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
        if epoch % cfg.get('checkpoint_every', 1) == 0 or epoch == cfg['epochs']:
            save(epoch, batch)
        if epoch in cfg['eval_epochs'] or epoch == cfg['epochs']:
            evaluate(epoch)
        write_json(run / 'status.json', dict(status='running', completed_epochs=epoch, step=step, batch_size=batch))
    write_json(run / 'status.json', dict(status='completed', completed_epochs=cfg['epochs'], step=step, batch_size=batch))


if __name__ == '__main__':
    try:
        main()
    except Exception:
        if '--run-dir' in sys.argv:
            failed_run = Path(sys.argv[sys.argv.index('--run-dir') + 1])
            if failed_run.is_dir():
                write_json(failed_run / 'status.json', {'status': 'failed', 'traceback': traceback.format_exc()})
        raise
