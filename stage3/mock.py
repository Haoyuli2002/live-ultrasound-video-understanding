"""Explicit CPU interface mock. These are NOT trained Stage 2/Qwen weights."""
import json
import re
from pathlib import Path
from types import SimpleNamespace
import numpy as np
import torch
from torch import nn
from .model import DecisionModel, SYSTEM

HIDDEN = 32


def create_checkpoint(path):
    path = Path(path); path.mkdir(parents=True, exist_ok=False)
    torch.manual_seed(719)
    torch.save({'projection':torch.randn(8,HIDDEN)*.1}, path/'mock_memory.pt')
    (path/'mock_config.json').write_text(json.dumps({
        'backend':'mock','not_a_peft_adapter':True,'stage2_trained':False,
        'hidden_size':HIDDEN,'purpose':'CPU wiring smoke only; no learned ultrasound representations'},indent=2)+'\n')


class MockMemory:
    def __init__(self, path):
        cfg=json.loads((Path(path)/'mock_config.json').read_text())
        if cfg['backend']!='mock':raise ValueError('Not a mock checkpoint')
        self.projection=torch.load(Path(path)/'mock_memory.pt',weights_only=True)['projection']
    def encode_frame(self, image):
        x=np.asarray(image,dtype=np.float32)/255.
        features=np.concatenate([x.mean((0,1)),x.std((0,1)),[x.min(),x.max()]])
        return (torch.tensor(features,dtype=torch.float32)@self.projection).unsqueeze(0)
    def update_long(self, previous, short):
        # Mimic the shape and recurrence boundary only, not the learned Stage 2 algorithm.
        return short if previous is None else .5*previous+.5*short


class ByteTokenizer:
    eos_token_id=1
    tokens={'<DECISION>':2,'<WAIT>':3,'<ANSWER>':4}
    def __call__(self,text,**kwargs):
        ids=[]
        for piece in re.split(r'(<DECISION>|<WAIT>|<ANSWER>)',text):
            ids.extend([self.tokens[piece]] if piece in self.tokens else [b+5 for b in piece.encode()])
        return SimpleNamespace(input_ids=torch.tensor([ids],dtype=torch.long))
    def decode(self,ids,skip_special_tokens=True):
        return bytes(i-5 for i in ids if 5<=i<261).decode('utf-8',errors='replace')


class TinyBackbone(nn.Module):
    def __init__(self):
        super().__init__();self.emb=nn.Embedding(261,HIDDEN);self.rnn=nn.GRU(HIDDEN,HIDDEN,batch_first=True);self.head=nn.Linear(HIDDEN,261)
    def get_input_embeddings(self):return self.emb
    def forward(self,inputs_embeds,attention_mask=None,labels=None,**kwargs):
        hidden,_=self.rnn(inputs_embeds);logits=self.head(hidden);loss=None
        if labels is not None:
            loss=nn.functional.cross_entropy(logits[:,:-1].reshape(-1,261),labels[:,1:].reshape(-1),ignore_index=-100)
        return SimpleNamespace(logits=logits,hidden_states=[hidden],loss=loss)


class MockDecision(DecisionModel):
    def __init__(self, checkpoint=None):
        nn.Module.__init__(self);self.model=TinyBackbone();self.tokenizer=ByteTokenizer();self.decision_id=2;self.decision_head=nn.Linear(HIDDEN,1)
        if checkpoint:self.load_state_dict(torch.load(Path(checkpoint)/'mock_decision.pt',weights_only=True))
    def save(self,path):
        path=Path(path);path.mkdir(parents=True,exist_ok=True)
        torch.save(self.state_dict(),path/'mock_decision.pt')
        (path/'stage3_config.json').write_text(json.dumps({'backend':'mock','not_a_qwen_checkpoint':True,'hidden_size':HIDDEN},indent=2)+'\n')
