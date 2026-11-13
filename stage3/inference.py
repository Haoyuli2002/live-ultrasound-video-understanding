"""Reference-free classification and marker-constrained greedy text decoding."""
import math
import time
import torch
from .model import ANSWER, WAIT


@torch.inference_mode()
def generate(decision, memory, question, *, threshold=.5, max_new_tokens=128,
             force_status=None):
    if decision.training or not 0 < threshold < 1 or max_new_tokens < 1:
        raise ValueError('Use eval mode, 0<threshold<1 and positive token budget')
    if force_status not in {None, 'WAIT', 'ANSWER'}:
        raise ValueError('Invalid forced marker')
    embeds, attention, position = decision._prefix(memory, question)
    # The classifier sees no reference, generated response, or gold answer time.
    output = decision.model(inputs_embeds=embeds, attention_mask=attention,
                            output_hidden_states=True, use_cache=False, return_dict=True)
    hidden = output.hidden_states[-1][:, position, :]
    logit = decision.decision_head(hidden.to(decision.decision_head.weight.dtype)).squeeze(-1)
    probability = float(torch.sigmoid(logit).item())
    if not math.isfinite(probability):
        raise RuntimeError('Non-finite answerability probability')
    status = force_status or ('ANSWER' if probability >= threshold else 'WAIT')
    marker = ANSWER if status == 'ANSWER' else WAIT
    result = {'probability': probability, 'status': status,
              'forced_marker_diagnostic': force_status is not None}
    if status == 'WAIT':
        # Current dataset supervises a bare WAIT. No unsupported explanation is invented.
        return {**result, 'prediction': '', 'target': WAIT, 'stop_reason': 'wait', 'generated_tokens': 0}
    prefix = torch.cat([embeds, decision.embedding(decision._ids('\nAssistant: ')),
                        decision.embedding(decision._ids(marker))], dim=1)
    eos = decision.tokenizer.eos_token_id
    if eos is None:
        raise ValueError('Tokenizer must define EOS')
    generated = []; stop = 'max_new_tokens'; start = time.monotonic()
    for _ in range(max_new_tokens):
        output = decision.model(inputs_embeds=prefix,
            attention_mask=torch.ones(prefix.shape[:2], dtype=torch.long, device=decision.device),
            use_cache=False, return_dict=True)
        token = output.logits[:, -1, :].argmax(-1, keepdim=True)
        generated.append(int(token.item()))
        if generated[-1] == eos:
            stop = 'eos'; break
        prefix = torch.cat([prefix, decision.embedding(token)], dim=1)
    prediction = decision.tokenizer.decode(generated, skip_special_tokens=True).strip()
    return {**result, 'prediction': prediction, 'target': marker+' '+prediction,
            'generated_token_ids': generated, 'generated_tokens': len(generated),
            'stop_reason': stop, 'decode_seconds': time.monotonic()-start}
