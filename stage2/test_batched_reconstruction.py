import copy
from types import SimpleNamespace
import unittest

import torch
from torch import nn
from torch.nn import functional as F
from transformers import BertConfig
from .batched_reconstruction import BatchedReconstruction, target_batches
from .blip2_overfit import Reconstruction
from .blip2_memory import Blip2Memory, ImageQueries
from .lavis_query_layers import BertLayer, BertSelfAttention
from .qformer import QFormerMemoryModel, SecondwiseShortEncoder, QueryResampler


class Tokenizer:
    eos_token_id = 1
    def __call__(self, text, **kwargs):
        return SimpleNamespace(input_ids=torch.tensor([[2 + ord(c) % 23 for c in text]]))


class Decoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.emb = nn.Embedding(27, 6)
        self.out = nn.Linear(6, 27)
    def get_input_embeddings(self):
        return self.emb
    def forward(self, inputs_embeds, attention_mask=None, labels=None, logits_to_keep=None, **kwargs):
        x = inputs_embeds if attention_mask is None else inputs_embeds * attention_mask[..., None]
        x = torch.tanh(x.cumsum(1) / 10)
        if logits_to_keep is not None:
            x = x[:, logits_to_keep]
        logits = self.out(x)
        loss = None if labels is None else F.cross_entropy(logits[:, :-1].reshape(-1, 27), labels[:, 1:].reshape(-1), ignore_index=-100)
        return SimpleNamespace(logits=logits, loss=loss)


def image_queries(count):
    obj = ImageQueries.__new__(ImageQueries); nn.Module.__init__(obj)
    cfg = BertConfig(hidden_size=12, num_attention_heads=3, intermediate_size=24,
                     num_hidden_layers=2, hidden_dropout_prob=0., attention_probs_dropout_prob=0.)
    cfg.encoder_width=6; cfg.add_cross_attention=True; cfg.cross_attention_freq=2
    obj.input_norm=nn.LayerNorm(6);obj.query_tokens=nn.Parameter(torch.randn(1,count,12)*.02)
    obj.embedding_norm=nn.LayerNorm(12);obj.embedding_dropout=nn.Dropout(0.)
    obj.layers=nn.ModuleList([BertLayer(cfg,i) for i in range(2)])
    for layer in obj.layers: del layer.intermediate, layer.output
    obj.output_projection=nn.Linear(12,6)
    return obj


def memory(recurrent):
    obj=(QFormerMemoryModel if recurrent else Blip2Memory).__new__(QFormerMemoryModel if recurrent else Blip2Memory)
    obj.model=Decoder();obj.visual=nn.Identity();obj.tokenizer=Tokenizer()
    obj.requires_short_history=recurrent
    obj.qformer_config={'input_dim':6}
    obj.compressors=nn.ModuleDict({'short':SecondwiseShortEncoder(6,6,8,1,2) if recurrent else image_queries(1),
                                  'long':QueryResampler(6,6,60,8,1,2) if recurrent else image_queries(60)})
    obj.reset_short_memory();obj.eval()
    return obj


def tasks():
    return [dict(video_id='a',kind='short',start=10,end=20,target='short'),
            dict(video_id='a',kind='block',start=0,end=60,target='longer block label'),
            dict(video_id='a',kind='global',start=60,end=120,target='global'),
            dict(video_id='a',kind='global',start=120,end=180,target='third minute'),
            dict(video_id='b',kind='global',start=60,end=120,target='other video')]


class BatchTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)
    def test_serial_batched_loss_and_all_parameter_gradients(self):
        for recurrent in (False,True):
            with self.subTest(recurrent=recurrent):
                torch.manual_seed(8)
                a=memory(recurrent);b=copy.deepcopy(a)
                frames={v:[torch.randn(3,6) for _ in range(180)] for v in ['a','b']}
                serial=Reconstruction(a);parallel=BatchedReconstruction(b,short_chunk=32)
                expected=torch.stack([serial(t,frames[t['video_id']]) for t in tasks()]).mean()
                actual=parallel(tasks(),frames)
                torch.testing.assert_close(actual,expected,atol=2e-6,rtol=2e-6)
                expected.backward();actual.backward()
                pa=dict(serial.named_parameters());pb=dict(parallel.named_parameters())
                self.assertEqual(set(pa),set(pb))
                for name,p in pa.items():
                    if p.grad is None:self.assertIsNone(pb[name].grad,name)
                    else:torch.testing.assert_close(pb[name].grad,p.grad,atol=2e-6,rtol=2e-5,msg=name)
                self.assertEqual(parallel.last_stats['unique_frame_encodings'],300)
                self.assertLess(parallel.last_stats['long_calls'],5)
    def test_no_future_leak_or_cache_across_parameter_change(self):
        for recurrent in (False,True):
            torch.manual_seed(4);m=memory(recurrent);obj=BatchedReconstruction(m)
            fs={'a':[torch.randn(3,6) for _ in range(60)]}
            t=[dict(video_id='a',kind='short',start=0,end=10,target='x')]
            before=obj.build_memories(t,fs)[0].detach()
            fs['a'][10:]=[x+999 for x in fs['a'][10:]]
            torch.testing.assert_close(obj.build_memories(t,fs)[0],before)
            with torch.no_grad():
                for p in m.compressors['short'].parameters():p.add_(.1)
            self.assertFalse(torch.allclose(obj.build_memories(t,fs)[0],before))
    def test_global_history_is_detached(self):
        for recurrent in (False,True):
            m=memory(recurrent);obj=BatchedReconstruction(m)
            fs={'a':[torch.randn(3,6,requires_grad=True) for _ in range(120)]}
            t=[dict(video_id='a',kind='global',start=60,end=120,target='x')]
            obj.build_memories(t,fs)[0].square().mean().backward()
            self.assertTrue(all(f.grad is None for f in fs['a'][:60]))
            self.assertTrue(any(f.grad is not None and f.grad.abs().sum()>0 for f in fs['a'][60:]))
    def test_ddp_batch_coverage_without_duplicate_padding(self):
        rows=list(range(56))
        for world in (1,2,4):
            for size in (1,4,8,16):
                plans=[target_batches(rows,size,r,world,3) for r in range(world)]
                self.assertEqual(sorted(x for rank in plans for batch in rank for x in batch),rows)
                self.assertEqual(len({len(p) for p in plans}),1)
                for step in range(len(plans[0])):
                    self.assertEqual(len({len(p[step]) for p in plans}),1)
    def test_sdpa_matches_eager_with_mask_and_gradients(self):
        cfg=BertConfig(hidden_size=12,num_attention_heads=3,attention_probs_dropout_prob=0.)
        cfg.encoder_width=6
        a=BertSelfAttention(cfg,True).eval();b=copy.deepcopy(a);b.use_sdpa=False
        x=torch.randn(2,4,12,requires_grad=True);y=x.detach().clone().requires_grad_()
        keys=torch.randn(2,7,6);mask=torch.zeros(2,1,1,7);mask[:,:,:,-2:]=-10000
        aa=a(x,encoder_hidden_states=keys,encoder_attention_mask=mask)[0]
        bb=b(y,encoder_hidden_states=keys,encoder_attention_mask=mask)[0]
        torch.testing.assert_close(aa,bb,atol=1e-6,rtol=1e-5)
        aa.square().sum().backward();bb.square().sum().backward()
        torch.testing.assert_close(x.grad,y.grad,atol=1e-6,rtol=1e-5)
        for p,q in zip(a.parameters(),b.parameters()):torch.testing.assert_close(p.grad,q.grad,atol=1e-6,rtol=1e-5)


if __name__=='__main__':unittest.main()
