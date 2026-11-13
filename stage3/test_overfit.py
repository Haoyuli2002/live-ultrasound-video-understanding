import json
from pathlib import Path
import tempfile
import unittest
import torch
from stage3.train import load_examples, decision_question
from stage3.mock import MockDecision
from stage3.inference import generate
from stage3.metrics import score


class OverfitTests(unittest.TestCase):
    def load(self,row):
        with tempfile.TemporaryDirectory() as d:
            p=Path(d)/'a.jsonl';p.write_text(json.dumps(row)+'\n');return load_examples(p)
    def sample(self,a=3,target='<ANSWER> Yes.'):
        return {'task_id':'x','video_id':'v','question':'Visible?','target':target,'video_window':[0,3],
                'meta':{'query_time':3,'answer_time':a,'current_time':3}}
    def test_immediate_and_null(self):
        self.assertEqual(self.load(self.sample())['v'][0]['y'],1)
        r=self.sample(None,'<WAIT>');r['meta']['review_horizon']=6
        self.assertEqual(self.load(r)['v'][0]['y'],0)
        r['video_window'][1]=7;r['meta']['current_time']=7
        with self.assertRaises(ValueError):self.load(r)
    def test_timing_and_label_rejection(self):
        r=self.sample();r['meta']['qa_type']='temporal_comparison'
        with self.assertRaises(ValueError):self.load(r)
        r=self.sample();r['meta']['y']=0
        with self.assertRaises(ValueError):self.load(r)
        r=self.sample(float('nan'))
        with self.assertRaises(ValueError):self.load(r)
    def test_prompt_excludes_gold(self):
        r={'time':4,'query_time':3,'question':'Visible?','answer_time':999,'reference':'SECRET','evidence':['SECRET']}
        self.assertEqual(decision_question(r),'Query time: 3 seconds. Current time: 4 seconds.\nVisible?')
    def test_causal_teacher_text_cannot_change_decision(self):
        torch.manual_seed(1);m=MockDecision().eval();state=torch.randn(1,3,32)
        direct=m.decision_logits(state,'Visible?')
        _,_,a=m.forward_answerability(state,'Visible?','<ANSWER> secret one',1)
        _,_,b=m.forward_answerability(state,'Visible?','<WAIT>',0)
        self.assertTrue(torch.allclose(direct,a,atol=1e-6));self.assertTrue(torch.allclose(a,b,atol=1e-6))
        with self.assertRaisesRegex(ValueError,'detached'):m._prefix(state.requires_grad_(),'Visible?')
    def test_forced_marker_and_reload(self):
        torch.manual_seed(2);m=MockDecision().eval();state=torch.randn(1,2,32)
        with torch.no_grad():m.decision_head.weight.zero_();m.decision_head.bias.fill_(-20)
        r=generate(m,state,'Visible?',max_new_tokens=2)
        self.assertEqual(r['target'],'<WAIT>');self.assertFalse(r['forced_marker_diagnostic'])
        r=generate(m,state,'Visible?',max_new_tokens=2,force_status='ANSWER')
        self.assertEqual(r['status'],'ANSWER');self.assertTrue(r['forced_marker_diagnostic']);self.assertLessEqual(r['generated_tokens'],2)
        with tempfile.TemporaryDirectory() as d:
            m.save(d);n=MockDecision(d).eval()
            self.assertTrue(torch.equal(m.decision_logits(state,'Visible?'),n.decision_logits(state,'Visible?')))
    def test_checkpoint_source_chain(self):
        from stage3.overfit import save_checkpoint, verify_lineage
        with tempfile.TemporaryDirectory() as d:
            lineage={'backend':'mock','stage2_checkpoint_hashes':{'weights':'abc'}}
            save_checkpoint(MockDecision(),Path(d),lineage)
            verify_lineage(d,lineage)
            with self.assertRaisesRegex(ValueError,'differs'):
                verify_lineage(d,{**lineage,'backend':'qwen'})
            with self.assertRaisesRegex(ValueError,'lacks'):
                verify_lineage(Path(d)/'missing',lineage)

    def test_first_response_outcomes(self):
        tasks=[];pred=[]
        for i,(a,first) in enumerate([(3,2),(3,3),(3,4),(3,None),(None,None),(None,3)]):
            tid=str(i);tasks.append({'task_id':tid,'query_time':2,'answer_time':a,'evaluation_horizon':4,'reference_answer':'Yes'})
            for t in [2,3,4]:pred.append({'task_id':tid,'time':t,'status':'ANSWER' if first and t>=first else 'WAIT','prediction':'Yes'})
        r=score(tasks,pred)
        self.assertEqual(r['task_outcomes'],dict(early=1,on_time=1,late=1,miss=1,correct_wait=1,false_trigger=1))
        self.assertEqual(r['e2e_word_f1_zero_for_early_or_miss'],.5)
        with self.assertRaises(ValueError):score(tasks,pred[:-1])


if __name__=='__main__':unittest.main()
