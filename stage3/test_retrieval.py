import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import torch

from .retrieval import RetrievalHead,EvidenceBank,group_retrieval_loss
from .mock import MockDecision,HIDDEN
from .train import causal_memories
from .retrieval_training import train_bundle,load_dataset
from .retrieval_inference import generate_retrieved


class Reader:
    count=1300;fps=10
    def frame_before(self,t):return t
    def frame_ending_at(self,t):return t+1


class Memory:
    """Stateful fake with deliberately different current vs history features."""
    def __init__(self):self._short_history=None;self.resets=0
    def reset_short_memory(self):self._short_history=None;self.resets+=1
    def extract_visual_features(self,images):
        return [torch.stack([torch.arange(HIDDEN).float()*t+1,torch.arange(HIDDEN).float()+t]) for t in images]
    def encode_visual_short(self,frames):
        current=frames[0].mean(0).reshape(1,1,-1)
        if self._short_history is not None:current=current+self._short_history.sum(1,keepdim=True)
        self._short_history=current if self._short_history is None else torch.cat([self._short_history,current],1)
        return current
    def encode_frame(self,image):return self.encode_visual_short(self.extract_visual_features([image]))[:,0,:]
    def update_long(self,previous,short):return short


class RetrievalTests(unittest.TestCase):
    def setUp(self):torch.manual_seed(42);torch.set_num_threads(1)

    def test_stage2_loader_detects_qformer_and_freezes_compressors(self):
        from types import SimpleNamespace
        from .model import load_frozen_memory_model
        fake=SimpleNamespace(model=torch.nn.Linear(2,2),compressors=torch.nn.Linear(2,2))
        with patch('stage2.qformer.load_memory_model',return_value=fake) as loader:
            loaded=load_frozen_memory_model(stage2_adapter='checkpoint')
        self.assertEqual(loader.call_args.kwargs['adapter_path'],'checkpoint')
        self.assertFalse(loaded.model.training);self.assertFalse(loaded.compressors.training)
        self.assertFalse(any(p.requires_grad for p in loaded.model.parameters()))
        self.assertFalse(any(p.requires_grad for p in loaded.compressors.parameters()))

    def test_normalized_512_dimensions_and_projection_gradients(self):
        h=RetrievalHead(32,24);x=torch.randn(1,32,requires_grad=True);v=torch.randn(5,24)
        self.assertEqual(h.query(x).shape,(1,512));self.assertEqual(h.keys(v).shape,(5,512))
        self.assertTrue(torch.allclose(h.keys(v).norm(dim=-1),torch.ones(5),atol=1e-6))
        group_retrieval_loss(h(x,v)[0],[[0,1],[2]],[3,4]).backward()
        for grad in [x.grad,h.key_projection.weight.grad,h.query_projection.weight.grad]:
            self.assertTrue(torch.isfinite(grad).all());self.assertGreater(float(grad.abs().sum()),0)

    def test_group_alternatives_unknowns_and_other_positives(self):
        s=torch.tensor([1.,2.,3.,0.,999.],requires_grad=True)
        loss=group_retrieval_loss(s,[[0,1],[2]],[3]);loss.backward()
        self.assertEqual(float(s.grad[4]),0.)
        self.assertLess(float(s.grad[2]),0.)
        self.assertEqual(float(group_retrieval_loss(s,[[0,1],[2]],[]).detach()),0.)
        with self.assertRaises(ValueError):group_retrieval_loss(s,[[0]],[0])

    def test_bank_is_current_frame_only_detached_and_causal(self):
        bank=EvidenceBank('v');memory=Memory();reader=Reader()
        points=[{'time':2.},{'time':2.25},{'time':3.}]
        states=list(causal_memories(memory,reader,points,evidence_bank=bank))
        self.assertTrue(torch.equal(bank.records[3.].pooled,memory.extract_visual_features([3.])[0].mean(0)))
        self.assertFalse(torch.equal(bank.records[3.].pooled,states[-1][1][0,-1]))
        self.assertFalse(bank.records[3.].pooled.requires_grad)
        self.assertEqual([r.end for r in bank.candidates(2.)],[1.,2.])
        with self.assertRaises(ValueError):bank.get(3.,2.)
        with self.assertRaises(ValueError):bank.add(4.,torch.ones(2,32),frame_index=40,source_pts=4.)
        # Fractional probe must not alter next integer memory.
        ref=list(causal_memories(Memory(),reader,[{'time':3.}]))[-1][1]
        self.assertTrue(torch.equal(states[-1][1],ref))
        self.assertIsNone(memory._short_history)

    def test_minute_and_video_resets(self):
        # Use bounded recurrence for this shape/reset test.
        class Bounded(Memory):
            def encode_visual_short(self,frames):
                x=frames[0].mean(0).reshape(1,1,-1)
                self._short_history=x if self._short_history is None else torch.cat([self._short_history,x],1)
                if self._short_history.shape[1]>60:raise AssertionError('minute overflow')
                return x
        m=Bounded();list(causal_memories(m,Reader(),[{'time':60.25},{'time':61.}]))
        self.assertGreaterEqual(m.resets,3)
        list(causal_memories(m,Reader(),[{'time':1.}]))
        self.assertIsNone(m._short_history)

    def test_selection_budget_sorting_and_future_mask(self):
        b=EvidenceBank('v')
        for t in range(1,21):b.add(t,torch.randn(3,32),frame_index=t*10-1,source_pts=t-.1)
        h=RetrievalHead(32,32)
        windows=b.search(h,torch.randn(1,32),10,top_k=3,candidate_k=10,max_frames=5)
        self.assertLessEqual(len(windows),3);self.assertLessEqual(sum(len(w['frames']) for w in windows),5)
        self.assertEqual([w['start'] for w in windows],sorted(w['start'] for w in windows))
        self.assertTrue(all(w['end']<=10 for w in windows))
        self.assertTrue(all(w['end']-w['start']<=4 for w in windows))

    def fixture(self):
        evidence={}
        for t in [1,2,3]:
            evidence[f'E{t}']={'evidence_id':f'E{t}','start':t-1,'end':t,
                'anchor_frame':{'decision_time':t,'frame_index':t*10-1,'approx_pts':t-.1}}
        rows=[{'task_id':'q','video_id':'v','question':'What changed?','query_time':1.,'time':3.,'decision':1.,
               'retrieval':{'ids':['E1','E2','E3'],'groups':[[0],[2]],'negatives':[1]},
               'answers':[{'ids':['E1','E3'],'target':'<ANSWER> Changed. [E1] [E3]','sufficient':True},
                          {'ids':[],'target':'<WAIT>','sufficient':False}]}]
        return rows,evidence

    def test_joint_training_updates_heads_and_checkpoint_roundtrip(self):
        model=MockDecision(enable_retrieval=True);model.train();points,evidence=self.fixture()
        before=model.retrieval_head.key_projection.weight.detach().clone()
        opt=torch.optim.AdamW(model.parameters(),lr=.001)
        count=train_bundle(model,Memory(),Reader(),points,evidence,EvidenceBank('v'),opt,io.StringIO())
        self.assertEqual(count,1);self.assertTrue(model.evidence_trained)
        self.assertFalse(torch.equal(before,model.retrieval_head.key_projection.weight))
        self.assertIsNotNone(model.evidence_head.weight.grad)
        with tempfile.TemporaryDirectory() as d:
            model.save(d);restored=MockDecision(d)
            self.assertTrue(restored.evidence_trained)
            for k,v in model.state_dict().items():self.assertTrue(torch.equal(v,restored.state_dict()[k]))

    def test_retrieval_phase_changes_only_projection_heads(self):
        m=MockDecision(enable_retrieval=True).eval();m.requires_grad_(False);m.retrieval_head.requires_grad_(True)
        original={k:v.clone() for k,v in m.state_dict().items()}
        points,evidence=self.fixture()
        train_bundle(m,Memory(),Reader(),points,evidence,EvidenceBank('v'),torch.optim.SGD(m.retrieval_head.parameters(),lr=.1),io.StringIO(),phase='retrieval')
        for k,v in original.items():
            if not k.startswith('retrieval_head.'):self.assertTrue(torch.equal(v,m.state_dict()[k]))
        self.assertFalse(m.evidence_trained)

    def test_inference_wait_then_evidence_rejection_then_cited_answer(self):
        m=MockDecision(enable_retrieval=True).eval();m.evidence_trained=True
        mem=Memory();reader=Reader();bank=EvidenceBank('v')
        state=list(causal_memories(mem,reader,[{'time':3.}],evidence_bank=bank))[0][1]
        with torch.no_grad():
            m.decision_head.weight.zero_();m.decision_head.bias.fill_(-20)
            m.evidence_head.weight.zero_();m.evidence_head.bias.fill_(-20)
        kwargs=dict(bank=bank,memory=mem,reader=reader,current_time=3.,max_new_tokens=30)
        with patch.object(bank,'search',side_effect=AssertionError('WAIT must not retrieve')):
            self.assertEqual(generate_retrieved(m,state,'Q',**kwargs)['next_action'],'observe')
        with torch.no_grad():m.decision_head.bias.fill_(20)
        self.assertEqual(generate_retrieved(m,state,'Q',**kwargs)['next_action'],'retrieve_again')
        with torch.no_grad():m.evidence_head.bias.fill_(20)
        original=m.model.forward
        def scripted(text):
            sequence=iter(m._ids(text)[0].tolist()+[1])
            def forward(*args,**kw):
                out=original(*args,**kw)
                if not kw.get('output_hidden_states'):
                    out.logits[:,-1,:]=-100;out.logits[:,-1,next(sequence)]=100
                return out
            return forward
        with patch.object(m.model,'forward',side_effect=scripted('Visible [E1]')):
            result=generate_retrieved(m,state,'Q',**kwargs)
        self.assertEqual(result['status'],'ANSWER');self.assertEqual(result['citations'],['E1'])
        with patch.object(m.model,'forward',side_effect=scripted('Visible [E999]')):
            self.assertEqual(generate_retrieved(m,state,'Q',**kwargs)['next_action'],'regenerate')

    def test_evidence_teacher_answer_cannot_change_sufficiency(self):
        m=MockDecision(enable_retrieval=True).eval()
        inputs=[{'evidence_id':'E1','start':0,'end':1,'tokens':torch.randn(2,32)}]
        # BCE computed before teacher answer: both positive targets have same BCE.
        a,_=m.forward_evidence('Q',inputs,'<ANSWER> First [E1]',True)
        b,_=m.forward_evidence('Q',inputs,'<ANSWER> Different [E1]',True)
        self.assertTrue(torch.equal(a,b))

    def test_untrained_heads_and_legacy_bypass_are_rejected(self):
        from .inference import generate
        m=MockDecision(enable_retrieval=True).eval()
        with self.assertRaisesRegex(ValueError,'jointly trained'):
            generate_retrieved(m,torch.zeros(1,1,32),'Q',bank=EvidenceBank('v'),memory=Memory(),reader=Reader(),current_time=1.)
        with self.assertRaisesRegex(ValueError,'evidence_bank'):
            generate(m,torch.zeros(1,1,32),'Q')

    def test_loader_rejects_future_candidates_and_false_sufficiency(self):
        import shutil
        root=Path('results/stage3_evidence_pilot_P3_v1')
        if not root.exists():self.skipTest('Local pilot not present')
        with tempfile.TemporaryDirectory() as directory:
            dest=Path(directory)
            for name in ('tasks','evidence','decision_view','retrieval_view','answer_view'):
                shutil.copyfile(root/(name+'.jsonl'),dest/(name+'.jsonl'))
            path=dest/'retrieval_view.jsonl'
            original=path.read_text();rows=[json.loads(s) for s in original.splitlines()]
            rows[0]['current_time']=10
            path.write_text(''.join(json.dumps(r)+'\n' for r in rows))
            with self.assertRaises(ValueError):load_dataset(dest,allow_draft=True)
            path.write_text(original)
            path=dest/'answer_view.jsonl';rows=[json.loads(s) for s in path.read_text().splitlines()]
            rows[0]['input']['evidence']=[]
            with path.open('w') as f:
                for r in rows:f.write(json.dumps(r)+'\n')
            with self.assertRaisesRegex(ValueError,'sufficiency'):load_dataset(dest,allow_draft=True)

    def test_pilot_requires_explicit_draft_opt_in_and_masks_uncertainty(self):
        root=Path('results/stage3_evidence_pilot_P3_v1')
        if not root.exists():self.skipTest('Local pilot not present')
        with self.assertRaisesRegex(ValueError,'Draft'):load_dataset(root)
        rows,evidence=load_dataset(root,allow_draft=True)
        self.assertFalse(any(r['task_id']=='P3_E05' for r in rows))
        self.assertFalse(any(r['task_id']=='P3_E04' and r['time']==34 for r in rows))
        self.assertEqual(len(evidence),5)


if __name__=='__main__':unittest.main()
