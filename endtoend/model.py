"""Endpoint supervision, jointly backpropagated through the reasoning modules."""
import torch
from .timing_loss import readiness_loss_terms, sample_timing_pairs
from .data import prompt


def forward_batch(model, examples, features):
    # The parallel path shares causal video replay and batches masked Q-Formers;
    # the reference path processes samples independently. Both batch the decoder
    # over gold ANSWER rows and keep predicted readiness out of the training gate.
    if getattr(model, 'parallel_endpoints', False):
        from .parallel import forward_endpoints
        outputs = forward_endpoints(model.core, examples, features)
    else:
        outputs = [model.core(x, [prompt(e)], collect_readiness=True, endpoint_only=True)
                   for e, x in zip(examples, features)]
    logits = torch.cat([out['readiness'][:, -1] for out in outputs])
    terms = readiness_loss_terms(
        logits, examples, getattr(model, 'readiness_objective', 'bce'),
        lambda_reg=getattr(model, 'timing_lambda_reg', .1),
        bce_positive_weight=getattr(model, 'readiness_bce_positive_weight', 2.),
        bce_weight=getattr(model, 'readiness_bce_weight', 1.), sample_pairs=sample_timing_pairs(model))
    readiness_bce, readiness = terms['bce'], terms['loss']
    timing = terms['timing']
    positive = [i for i, e in enumerate(examples) if e['ready']]
    answer_loss = logits.new_zeros(())
    if positive:
        prefixes = torch.cat([outputs[i]['prefix_embeds'] for i in positive])
        answer_loss = model._reasoning_loss({'prefix_embeds': prefixes}, {
            'text_input': [prompt(examples[i]) for i in positive],
            'text_output': [examples[i]['answer'] for i in positive],
        })
    # Gold endpoints replace the paper's pseudo-label timing supervision.
    # backward_objectives controls routing: the default training strategy sends
    # readiness gradients only to RDY/head, even though the graph is shared.
    loss = answer_loss + model.readiness_loss_weight * readiness
    return dict(loss=loss, answer_loss=answer_loss, readiness_loss=readiness,
                logits=logits, outputs=outputs, readiness_bce=readiness_bce, readiness_weighted_bce=terms['weighted_bce'], timing=timing)


@torch.no_grad()
def predict_batch(model, examples, features, max_new_tokens=96, decode_all=False):
    result = forward_batch(model, examples, features)
    probabilities = result['logits'].float().sigmoid().tolist()
    records = []
    for e, out, probability in zip(examples, result['outputs'], probabilities):
        # Never force an answer at the last observed frame: untriggered tasks
        # must remain WAIT. Optional forced decoding is an answer-quality probe.
        ready = probability >= model.readiness_threshold
        answer = model._decode(out['prefix_embeds'], [prompt(e)], max_new_tokens)[0] if ready or decode_all else ''
        records.append(dict(sample_id=e['sample_id'], task_id=e['task_id'], qa_type=e['qa_type'],
                            current_time=e['current_time'], expected_ready=e['ready'],
                            ready=ready, probability=probability, answer=answer,
                            reference=e['answer'], forced_answer_probe=bool(decode_all and not ready)))
    return records, result
