"""Read-only, paired long-history interventions on a frozen StreamReady checkpoint."""
import argparse
from contextlib import contextmanager
import json
from pathlib import Path
from unittest.mock import patch

import torch
from .data import load_examples, prompt, sample_times
from .train import prepare_features, write_json, append_json


def replace_slots(memory, mask, donor):
    """Keep recipient shape/mask; deterministic resampling only when counts differ."""
    result = memory.clone()
    valid = mask[0].bool()
    n = int(valid.sum())
    if n and len(donor) == 0:
        raise ValueError('Empty donor history')
    if n:
        indices = torch.linspace(0, len(donor) - 1, n).round().long()
        result[0, valid] = donor[indices].to(device=memory.device, dtype=memory.dtype)
    return result


@contextmanager
def intervene(core, condition, historical, captured, expected_short=None, donor=None):
    original = core._run_branch

    def branch(qformer, queries, memory, mask, *args, **kwargs):
        if qformer is core.long_qformer:
            captured['long_slots'] = memory[0, mask[0].bool()].detach().cpu()
            if historical and condition == 'zero_long':
                memory = torch.zeros_like(memory)
            elif historical and condition == 'other_video_long':
                memory = replace_slots(memory, mask, donor)
            captured['changed_history'] = bool(historical and condition in ('zero_long', 'other_video_long'))
        out = original(qformer, queries, memory, mask, *args, **kwargs)
        if qformer is core.short_qformer:
            captured['short'] = out.detach().cpu()
            if expected_short is not None:
                torch.testing.assert_close(captured['short'], expected_short, atol=0, rtol=0)
        return out
    with patch.object(core, '_run_branch', side_effect=branch):
        yield


def summarize(rows):
    pos = [x for x in rows if x['expected_ready']]
    neg = [x for x in rows if not x['expected_ready']]
    def rate(n, d): return n / d if d else None
    return dict(endpoints=len(rows), answers=len(pos), waits=len(neg),
                forced_exact_count=sum(x['exact'] for x in pos),
                forced_exact=rate(sum(x['exact'] for x in pos), len(pos)),
                gated_exact=rate(sum(x['exact'] and x['ready'] for x in pos), len(pos)),
                answer_recall=rate(sum(x['ready'] for x in pos), len(pos)),
                false_triggers=sum(x['ready'] for x in neg),
                premature_rate=rate(sum(x['ready'] for x in neg),len(neg)),
                readiness_accuracy=rate(sum(x['ready']==bool(x['expected_ready']) for x in rows),len(rows)),
                answer_ce=rate(sum(x['answer_ce'] for x in pos), len(pos)),
                mean_abs_probability_change=rate(sum(abs(x['probability_change']) for x in rows),len(rows)),
                changed_answer_count=sum(x['answer_changed'] for x in pos))


def reports(rows):
    groups = {'all':rows, 'historical_only':[x for x in rows if x['has_history']],
              'no_history_control':[x for x in rows if not x['has_history']]}
    for kind in sorted({x['qa_type'] for x in rows}):
        groups[kind] = [x for x in rows if x['qa_type']==kind]
        groups[kind+'_historical'] = [x for x in groups[kind] if x['has_history']]
    return {k:summarize(v) for k,v in groups.items()}


@torch.no_grad()
def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run-dir',type=Path,required=True)
    args=parser.parse_args(); run=args.run_dir.resolve()
    cfg=json.loads((run/'experiment.json').read_text())
    write_json(run/'status.json',dict(status='initializing',optimizer_steps=0))
    torch.set_num_threads(4);torch.manual_seed(cfg['seed'])
    from omegaconf import OmegaConf
    from lavis.models.blip2_models.streamready import StreamReady
    from .parallel import forward_endpoints
    model=StreamReady.from_config(OmegaConf.create(dict(cfg['model'],load_pretrained=False,load_finetuned=False)))
    checkpoint_path=Path(cfg['ablation_checkpoint'])
    before=checkpoint_path.stat()
    checkpoint=torch.load(checkpoint_path,map_location='cpu',weights_only=True,mmap=True)
    restored=model.load_state_dict(checkpoint['model'],strict=False)
    if restored.unexpected_keys or any(not k.startswith('backbone.') for k in restored.missing_keys):
        raise RuntimeError(f'Incomplete restoration: {restored}')
    epoch=checkpoint['epoch'];del checkpoint
    model.cuda().eval();model.requires_grad_(False)
    assert model.core.context is None, 'This intervention assumes contextual QA bank disabled'
    if cfg.get('fast_linear_attention',False):
        from .acceleration import enable_fast_attention
        write_json(run/'attention_backend.json',enable_fast_attention(model))
    examples=load_examples(run/'samples.jsonl',run/'tasks.jsonl',run/'video_map.json')
    cache=prepare_features(model,examples,run,cfg['image_size'],cfg['vision_chunk'])
    baseline={}; banks={};shorts={};outputs={}; all_reports={}
    donor_map={}
    for condition in ('real','zero_long','other_video_long','zero_prefix'):
        records=[]
        write_json(run/'status.json',dict(status='running',condition=condition,completed=0,optimizer_steps=0))
        for i,e in enumerate(examples):
            sid=e['sample_id'];historical=len(sample_times(e['current_time']))>model.core.memory.frame_capacity
            donor_id=donor_map.get(sid);capture={}
            with torch.autocast('cuda',dtype=torch.bfloat16):
                if condition=='zero_prefix':
                    out={k:v.cuda() for k,v in outputs[sid].items()}
                    out['prefix_embeds']=torch.zeros_like(out['prefix_embeds'])
                else:
                    with intervene(model.core,condition,historical,capture,
                                   shorts.get(sid),banks[donor_id] if donor_id else None):
                        out=forward_endpoints(model.core,[e],[cache[sid]])[0]
                probability=float(out['readiness'][0,-1].float().sigmoid())
                ce=float(model._reasoning_loss(out,{'text_input':[prompt(e)],'text_output':[e['answer']]})) if e['ready'] else None
                answer=model._decode(out['prefix_embeds'],[prompt(e)],cfg['max_new_tokens'])[0] if e['ready'] else ''
            record=dict(condition=condition,sample_id=sid,task_id=e['task_id'],video_id=e['video_id'],
                        qa_type=e['qa_type'],current_time=e['current_time'],has_history=historical,
                        expected_ready=e['ready'],ready=probability>=model.readiness_threshold,
                        probability=probability,answer=answer,reference=e['answer'],answer_ce=ce,
                        exact=bool(e['ready'] and answer.strip().lower()==e['answer'].strip().lower()),
                        donor_sample_id=donor_id if condition=='other_video_long' and historical else None,
                        probability_change=probability-baseline[sid]['probability'] if sid in baseline else 0.,
                        answer_changed=bool(e['ready'] and sid in baseline and answer!=baseline[sid]['answer']),
                        prefix_rms_change=float((out['prefix_embeds'].float().cpu()-outputs[sid]['prefix_embeds'].float()).square().mean().sqrt()) if sid in outputs else 0.,
                        readiness_intervened=condition!='zero_prefix')
            if condition=='real':
                baseline[sid]=record;banks[sid]=capture['long_slots'];shorts[sid]=capture['short']
                outputs[sid]={k:v.detach().cpu() for k,v in out.items()}
            elif condition in ('zero_long','other_video_long') and not historical:
                assert record['probability_change']==0 and not record['answer_changed'], 'No-history control changed'
            records.append(record);append_json(run/'predictions.jsonl',record)
            if (i+1)%10==0:
                print('ABLATION',condition,i+1,len(examples),flush=True)
                write_json(run/'status.json',dict(status='running',condition=condition,completed=i+1,total=len(examples),optimizer_steps=0))
        all_reports[condition]=reports(records)
        write_json(run/'metrics.json',all_reports)
        print('METRICS',condition,json.dumps(all_reports[condition]),flush=True)
        if condition=='real':
            saved={x['sample_id']:x for x in json.loads((run/'reference_eval.json').read_text())['predictions']}
            delta=max(abs(baseline[s]['probability']-saved[s]['probability']) for s in baseline)
            mismatches=[s for s,r in baseline.items() if r['expected_ready'] and r['answer']!=saved[s]['answer']]
            write_json(run/'baseline_validation.json',dict(max_probability_difference=delta,answer_mismatches=mismatches))
            if delta>.005 or mismatches:
                raise RuntimeError('Baseline differs from original final eval; inspect before interpreting ablation')
            for e in examples:
                sid=e['sample_id']
                if len(sample_times(e['current_time']))<=model.core.memory.frame_capacity:continue
                candidates=[x for x in examples if x['video_id']!=e['video_id'] and len(sample_times(x['current_time']))>model.core.memory.frame_capacity and len(banks[x['sample_id']])]
                donor=min(candidates,key=lambda x:(abs(len(banks[x['sample_id']])-len(banks[sid])),abs(x['current_time']-e['current_time']),x['sample_id']))
                donor_map[sid]=donor['sample_id']
            write_json(run/'donor_map.json',donor_map)
    after=checkpoint_path.stat()
    assert (before.st_size,before.st_mtime_ns)==(after.st_size,after.st_mtime_ns)
    write_json(run/'status.json',dict(status='completed',checkpoint_epoch=epoch,optimizer_steps=0,
                                    checkpoint_modified=False,conditions=list(all_reports),
                                    note='Training endpoints only. Long ablation changes retrieved historical slots, retaining short fusion/question/mask. zero_prefix is decoder diagnostic; readiness unchanged.'))


if __name__=='__main__':
    main()
