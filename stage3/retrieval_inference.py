"""Decision -> causal frame retrieval -> evidence sufficiency -> cited answer."""
import argparse
import json
import math
from pathlib import Path
import re

import torch
from .evidence import materialize
from .retrieval import EvidenceBank, evidence_metadata
from .model import ANSWER, WAIT


@torch.inference_mode()
def generate_retrieved(decision, state, question, *, bank, memory, reader,
                       current_time, threshold=.5, evidence_threshold=.5,
                       top_k=4, candidate_k=16, max_frames=8, max_new_tokens=128):
    if decision.training or getattr(decision,'retrieval_head',None) is None:
        raise ValueError('An eval-mode retrieval-enabled decision model is required')
    if not getattr(decision,'evidence_trained',False):
        raise ValueError('Evidence answering requires a jointly trained checkpoint, not retrieval-only weights')
    if not 0<threshold<1 or not 0<evidence_threshold<1 or max_new_tokens<1:
        raise ValueError('Invalid inference threshold/token budget')
    hidden = decision.decision_state(state,question)
    probability = float(torch.sigmoid(decision.decision_head(hidden.to(decision.decision_head.weight))).item())
    if not math.isfinite(probability):
        raise RuntimeError('Non-finite decision probability')
    result = {'probability':probability,'status':'WAIT','prediction':'','target':WAIT,
              'evidence':[],'generated_tokens':0,'current_time':current_time}
    if probability<threshold:
        return {**result,'stop_reason':'prefix_insufficient','next_action':'observe'}
    windows = bank.search(decision.retrieval_head,hidden,current_time,top_k=top_k,
                          candidate_k=candidate_k,max_frames=max_frames)
    result['evidence'] = evidence_metadata(windows)
    if not windows:
        return {**result,'stop_reason':'no_evidence','next_action':'retrieve_again'}
    evidence = materialize(memory,reader,bank,windows,current_time)
    prefix = decision.evidence_prefix(question,evidence)
    out = decision.model(inputs_embeds=prefix,attention_mask=torch.ones(prefix.shape[:2],dtype=torch.long,device=decision.device),
                         output_hidden_states=True,use_cache=False,return_dict=True)
    h = out.hidden_states[-1][:,-1,:]
    sufficient = float(torch.sigmoid(decision.evidence_head(h.to(decision.evidence_head.weight))).item())
    if not math.isfinite(sufficient):
        raise RuntimeError('Non-finite evidence probability')
    result['evidence_probability'] = sufficient
    if sufficient<evidence_threshold:
        return {**result,'stop_reason':'retrieved_evidence_insufficient','next_action':'retrieve_again'}
    prefix = torch.cat([prefix,decision.embedding(decision._ids('\nAssistant: '+ANSWER))],dim=1)
    generated=[];stop='max_new_tokens'
    eos=decision.tokenizer.eos_token_id
    if eos is None:raise ValueError('Tokenizer requires EOS')
    for _ in range(max_new_tokens):
        out=decision.model(inputs_embeds=prefix,attention_mask=torch.ones(prefix.shape[:2],dtype=torch.long,device=decision.device),
                           use_cache=False,return_dict=True)
        token=out.logits[:,-1,:].argmax(-1,keepdim=True)
        generated.append(int(token.item()))
        if generated[-1]==eos:
            stop='eos';break
        prefix=torch.cat([prefix,decision.embedding(token)],dim=1)
    prediction=decision.tokenizer.decode(generated,skip_special_tokens=True).strip()
    citations=set(re.findall(r'\[([A-Za-z0-9]+)\]',prediction))
    valid={w['evidence_id'] for w in windows}
    # Membership checks do not establish that a cited image supports a claim.
    if not prediction or not citations or not citations<=valid:
        return {**result,'stop_reason':'invalid_or_missing_citation','next_action':'regenerate',
                'draft_prediction':prediction,'generated_tokens':len(generated)}
    return {**result,'status':'ANSWER','prediction':prediction,'target':ANSWER+' '+prediction,
            'stop_reason':stop,'next_action':'answer','generated_tokens':len(generated),
            'citations':sorted(citations),'citation_support_verified':False}


def main():
    from stage2.data import VideoReader
    from .model import DecisionModel,load_frozen_memory_model
    from .train import causal_memories,decision_question
    from .retrieval_training import place_memory,checkpoint_frame_size
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--video',type=Path,required=True)
    p.add_argument('--question',required=True)
    p.add_argument('--query-time',type=float,required=True)
    p.add_argument('--times',type=float,nargs='+',required=True)
    p.add_argument('--decision-adapter',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--max-new-tokens',type=int,default=128)
    a=p.parse_args()
    if not a.question.strip() or not math.isfinite(a.query_time) or a.query_time<0 or any(not math.isfinite(t) or t<a.query_time or t<=0 for t in a.times):
        raise ValueError('Invalid question/time')
    if a.output.exists():raise FileExistsError(a.output)
    cfg=json.loads((a.decision_adapter/'stage3_config.json').read_text())
    memory=load_frozen_memory_model(cfg['model_name'],stage1_adapter=cfg['stage1_adapter'],stage2_adapter=cfg['stage2_adapter'])
    place_memory(memory)
    decision=DecisionModel(cfg['model_name'],stage1_adapter=cfg['stage1_adapter'],stage2_adapter=cfg['stage2_adapter'],
                           decision_adapter=str(a.decision_adapter),enable_retrieval=True).to('cuda').eval()
    reader=VideoReader(a.video,checkpoint_frame_size(cfg['stage2_adapter']))
    bank=EvidenceBank(a.video.resolve())
    points=[{'time':t,'query_time':a.query_time,'question':a.question} for t in sorted(set(a.times))]
    try:
        with a.output.open('x') as f:
            for row,state in causal_memories(memory,reader,points,evidence_bank=bank):
                result=generate_retrieved(decision,state,decision_question(row),bank=bank,memory=memory,reader=reader,
                                          current_time=row['time'],max_new_tokens=a.max_new_tokens)
                f.write(json.dumps(result)+'\n');f.flush()
    finally:reader.close()


if __name__=='__main__':main()
