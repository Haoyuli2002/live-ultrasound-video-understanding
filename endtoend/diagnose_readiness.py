"""Read-only checkpoint diagnostics: no optimizer, updates or checkpoint writes."""
import argparse
import contextlib
import json
import math
from pathlib import Path

import torch
from torch.nn import functional as F
from .data import load_examples, prompt
from .model import forward_batch
from .train import prepare_features, write_json


def score_stats(scores, labels, task_ids):
    p = scores.float().sigmoid()
    y = torch.tensor(labels, dtype=torch.bool)
    pos, neg = p[y], p[~y]
    by_task = {}
    for i, tid in enumerate(task_ids):
        by_task.setdefault(tid, []).append(i)
    diffs = [float(p[b] - p[a]) for indices in by_task.values()
             for a in indices if not labels[a] for b in indices if labels[b]]
    return dict(bce=float(F.binary_cross_entropy_with_logits(scores.float(), y.float())),
                auc=float(((pos[:, None] > neg).float() + .5 * (pos[:, None] == neg)).mean()),
                probability_min=float(p.min()), probability_max=float(p.max()),
                probability_std=float(p.std()), positive_mean=float(pos.mean()), negative_mean=float(neg.mean()),
                accuracy_at_half=float(((p >= .5) == y).float().mean()),
                true_positive=int(((p >= .5) & y).sum()),
                paired_positive_higher=sum(d > 0 for d in diffs), paired_total=len(diffs),
                paired_mean_probability_difference=sum(diffs) / max(1, len(diffs)))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source-run', type=Path, required=True)
    parser.add_argument('--output-dir', type=Path, required=True)
    args = parser.parse_args()
    run, output = args.source_run.resolve(), args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(4)
    torch.manual_seed(20261121)
    cfg = json.loads((run / 'experiment.json').read_text())
    examples = load_examples(run / 'samples.jsonl', run / 'tasks.jsonl', run / 'video_map.json')
    from omegaconf import OmegaConf
    from lavis.models.blip2_models.streamready import StreamReady
    model_cfg = dict(cfg['model'], load_pretrained=False, load_finetuned=False)
    model = StreamReady.from_config(OmegaConf.create(model_cfg))
    checkpoint = torch.load(run / 'checkpoint_latest.pth', map_location='cpu', weights_only=True, mmap=True)
    result = model.load_state_dict(checkpoint['model'], strict=False)
    assert not result.unexpected_keys
    assert all(k.startswith('backbone.') for k in result.missing_keys)
    del checkpoint
    model.cuda().eval()
    cache = prepare_features(model, examples, output, cfg['image_size'], cfg['vision_chunk'])
    saved = json.loads((run / 'eval_epoch_012.json').read_text())['predictions']
    captured = {}

    def capture(module, inputs):
        captured['rdy'] = inputs[0].detach().float().cpu()

    hook = model.core.readiness_head.register_forward_pre_hook(capture)
    all_results, representations = {}, {}
    labels = [e['ready'] for e in examples]
    ids = [e['task_id'] for e in examples]
    for condition in ('real_bf16', 'real_fp32', 'zero_visual', 'other_question', 'no_time_prompt', 'latest_frame_only'):
        logits, rdys, longs = [], [], []
        with torch.no_grad():
            for i, e in enumerate(examples):
                features = cache[e['sample_id']].cuda()
                text = prompt(e)
                if condition == 'zero_visual':
                    features = torch.zeros_like(features)
                elif condition == 'latest_frame_only':
                    features = features[:, -1:]
                elif condition == 'other_question':
                    other = next(x for x in examples if x['task_id'] != e['task_id'] and x['video_id'] != e['video_id'])
                    text = prompt(dict(e, question=other['question']))
                elif condition == 'no_time_prompt':
                    text = e['question'] + '\nAnswer:'
                ctx = torch.autocast('cuda', dtype=torch.bfloat16) if condition != 'real_fp32' else contextlib.nullcontext()
                with ctx:
                    out = model.core(features, [text], endpoint_only=True)
                logits.append(out['readiness'][:, -1].detach().float().cpu())
                rdys.append(captured['rdy'])
                longs.append(out['z_l'].mean(dim=1).detach().float().cpu())
        scores = torch.cat(logits)
        rdy, long = torch.cat(rdys), torch.cat(longs)
        representations[condition] = dict(logits=scores, rdy=rdy, long=long)
        stats = score_stats(scores, labels, ids)
        stats['rdy_centered_rms'] = float((rdy - rdy.mean(0)).square().mean().sqrt())
        stats['rdy_total_rms'] = float(rdy.square().mean().sqrt())
        if condition == 'real_bf16':
            predicted = {e['sample_id']: float(s.sigmoid()) for e, s in zip(examples, scores)}
            stats['max_difference_from_saved_eval'] = max(abs(predicted[e['sample_id']] - e['probability']) for e in saved)
        else:
            base = representations['real_bf16']
            stats['mean_abs_logit_change'] = float((scores - base['logits']).abs().mean())
            stats['rdy_change_rms'] = float((rdy - base['rdy']).square().mean().sqrt())
            stats['long_change_rms'] = float((long - base['long']).square().mean().sqrt())
        all_results[condition] = stats
        write_json(output / 'forward_diagnostics.json', all_results)
        print(condition, json.dumps(stats), flush=True)
    hook.remove()
    torch.save(dict(representations=representations, examples=examples), output / 'representations.pt')

    # Compare CE and readiness gradients at identical weights and paired inputs.
    # Evaluation mode eliminates dropout noise; autograd remains enabled.
    selected_names = ('core.rdy_token', 'core.long_query_tokens', 'core.short_query_tokens',
                      'core.vision_proj.weight', 'core.output_proj.weight',
                      'core.readiness_head.net.4.weight',
                      'core.long_qformer.bert.encoder.layer.0.crossattention.self.value.weight')
    named = dict(model.named_parameters())
    params = [named[name] for name in selected_names]
    gradient_results = []
    task_ids = list(dict.fromkeys(e['task_id'] for e in examples if not e['ready']))
    for task_id in task_ids:
        rows = [e for e in examples if e['task_id'] == task_id]
        positive = [e for e in rows if e['ready']]
        if not positive:
            continue
        rows = [max((e for e in rows if not e['ready']), key=lambda e: e['current_time']), positive[0]]
        with torch.autocast('cuda', dtype=torch.bfloat16):
            result = forward_batch(model, rows, [cache[e['sample_id']].cuda() for e in rows])
        bce_grads = torch.autograd.grad(result['readiness_loss'], params, retain_graph=True, allow_unused=True)
        ce_grads = torch.autograd.grad(result['answer_loss'], params, allow_unused=True)
        groups = {}
        for name, a, b in zip(selected_names, bce_grads, ce_grads):
            an, bn = (float(a.float().norm()) if a is not None else 0), (float(b.float().norm()) if b is not None else 0)
            groups[name] = dict(bce_norm=an, ce_norm=bn,
                               cosine=float(F.cosine_similarity(a.float().flatten(), b.float().flatten(), dim=0)) if an and bn else None)
        gradient_results.append(dict(task_id=task_id, bce=float(result['readiness_loss'].detach()),
                                     answer_ce=float(result['answer_loss'].detach()), parameters=groups))
        del result, bce_grads, ce_grads
    write_json(output / 'gradient_diagnostics.json', gradient_results)
    write_json(output / 'status.json', {'status': 'completed', 'optimizer_steps': 0, 'checkpoint_modified': False})
    print('DIAGNOSTICS_COMPLETED', flush=True)


if __name__ == '__main__':
    main()
