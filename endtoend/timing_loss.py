"""StreamReady official Eq.7-8 objective on gold-labelled endpoint timelines.

Training supports the official pair sampler or all within-question pairs;
evaluation uses the exact all-pairs expectation. Like the source, temporal TV is mean |sigmoid(s[i+1])-sigmoid(s[i])|.
Our timeline comprises annotated endpoints (irregular intervals), not dense 1fps
readiness outputs. This sampling adaptation and gold/pseudo distinction are
recorded explicitly in experiment configs. The pure contrastive objective is preserved; readiness_loss_terms optionally adds weighted BCE.
"""
import math
import torch
from torch.nn import functional as F


def sample_timing_pairs(model):
    mode = getattr(model, 'timing_pair_mode', 'sampled')
    if mode not in ('sampled', 'all_pairs'):
        raise ValueError(f'Unknown timing_pair_mode: {mode}')
    return model.training and mode == 'sampled'


def contrastive_timing_loss(logits, examples, lambda_reg=.1, sample_pairs=True):
    if logits.ndim != 1 or len(logits)!=len(examples):
        raise ValueError('Expected one logit per endpoint')
    if lambda_reg<0:raise ValueError('Negative temporal regularization')
    groups={}
    for i,e in enumerate(examples):
        if not math.isfinite(float(e['current_time'])):raise ValueError('Invalid endpoint time')
        groups.setdefault((e['video_id'],e['task_id']),[]).append(i)
    scores=logits.float();zero=scores.sum()*0.
    pair_terms=[];smooth_terms=[];available=0;sampled=0;unpaired=0
    for indices in groups.values():
        if len({examples[i]['question'] for i in indices})!=1:
            raise ValueError('Different questions share one task ID')
        ordered=sorted(indices,key=lambda i:examples[i]['current_time'])
        # Synthetic safety probes repeat one endpoint; deduplicate that timeline.
        distinct=[]
        for i in ordered:
            if distinct and examples[i]['current_time']==examples[distinct[-1]]['current_time']:
                if examples[i]['ready']!=examples[distinct[-1]]['ready']:
                    raise ValueError('Conflicting labels at one time')
                continue
            distinct.append(i)
        ordered=distinct
        s=scores[ordered]
        pos=torch.tensor([bool(examples[i]['ready']) for i in ordered],device=s.device)
        neg=~pos
        npos=int(pos.sum());nneg=int(neg.sum())
        if npos and nneg:
            if sample_pairs:
                from lavis.models.streamready.readiness import readiness_loss
                # Use the unmodified upstream contrastive sampler/formula.
                ctr=readiness_loss(s[None],pos[None],neg[None],lambda_reg=0.,max_pairs=64)
                sampled+=min(npos,nneg,64)
            else:
                ctr=-F.logsigmoid(s[pos,None]-s[None,neg]).mean()
            pair_terms.append(ctr);available+=npos*nneg
        else:unpaired+=1
        if len(ordered)>1:
            r=s.sigmoid()
            smooth_terms.append((r[1:]-r[:-1]).abs().mean())
    ctr=torch.stack(pair_terms).mean() if pair_terms else zero
    smooth=torch.stack(smooth_terms).mean() if smooth_terms else zero
    return dict(loss=ctr+lambda_reg*smooth,contrastive=ctr,temporal=smooth,
                pairs=available,sampled_pairs=sampled,used_pairs=sampled if sample_pairs else available,
                pair_mode="sampled" if sample_pairs else "all_pairs",paired_tasks=len(pair_terms),
                unpaired_tasks=unpaired,temporal_tasks=len(smooth_terms),lambda_reg=lambda_reg)


def readiness_loss_terms(logits, examples, objective='bce', lambda_reg=.1,
                         bce_positive_weight=2., bce_weight=1., sample_pairs=True):
    """ANSWER=1 gets pos_weight; mean reduction divides by endpoint count.

    Hybrid: L_rdy = L_ctr + lambda_reg * TV + bce_weight * weighted_BCE.
    Pure BCE/contrastive modes retain their previous behavior for old configs.
    """
    if not math.isfinite(bce_positive_weight) or bce_positive_weight <= 0:
        raise ValueError('bce_positive_weight must be finite and positive')
    if not math.isfinite(bce_weight) or bce_weight < 0:
        raise ValueError('bce_weight must be finite and nonnegative')
    scores = logits.float()
    labels = scores.new_tensor([e['ready'] for e in examples])
    bce = F.binary_cross_entropy_with_logits(scores, labels)
    weighted = F.binary_cross_entropy_with_logits(
        scores, labels, pos_weight=scores.new_tensor(bce_positive_weight))
    timing = None
    if objective in ('contrastive_gold', 'contrastive_bce_gold'):
        timing = contrastive_timing_loss(scores, examples, lambda_reg, sample_pairs=sample_pairs)
        loss = timing['loss']
        if objective == 'contrastive_bce_gold':
            loss = loss + bce_weight * weighted
    elif objective == 'bce':
        loss = bce
    else:
        raise ValueError(f'Unknown readiness objective: {objective}')
    return dict(loss=loss, bce=bce, weighted_bce=weighted, timing=timing)
