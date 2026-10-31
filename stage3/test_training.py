import io
import json
from pathlib import Path
import tempfile
import unittest

import torch
from stage3.train import causal_memories, load_examples, train_video


class FakeMemory(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.weight=torch.nn.Parameter(torch.ones(1),requires_grad=False)
        self.frames=[]
    def encode_frame(self,frame):
        self.frames.append(frame)
        return torch.tensor([[float(frame)]])*self.weight
    def update_long(self,previous,short):return short


class FakeReader:
    count=3000;fps=25
    def __init__(self):self.reads=[]
    def frame_ending_at(self,second):self.reads.append(second+1);return second+1
    def frame_before(self,time):self.reads.append(time);return time


class FakeDecision(torch.nn.Module):
    def __init__(self):
        super().__init__();self.weight=torch.nn.Parameter(torch.ones(1))
    def forward_answerability(self,memory,question,target,y):
        assert not memory.requires_grad
        prediction=self.weight*memory.mean()
        return (prediction-y).square().mean(),prediction.square().mean(),prediction


class Stage3TrainingTests(unittest.TestCase):
    def test_fractional_decisions_do_not_change_persistent_1fps_recurrence(self):
        memory=FakeMemory();reader=FakeReader()
        points=[{'time':3.0},{'time':3.25},{'time':3.25},{'time':4.0}]
        states=list(causal_memories(memory,reader,points))
        self.assertEqual([state.shape[1] for _,state in states],[3,4,4,4])
        self.assertEqual(reader.reads,[1,2,3,3.25,4])
        self.assertEqual(states[-1][1].flatten().tolist(),[1,2,3,4])
        self.assertTrue(all(not state.requires_grad for _,state in states))

    def test_full_minute_updates_long_then_appends_fractional_short(self):
        states=list(causal_memories(FakeMemory(),FakeReader(),[{'time':60.0},{'time':60.25},{'time':61.0}]))
        self.assertEqual([state.shape[1] for _,state in states],[60,61,61])
        self.assertEqual(float(states[-1][1][0,-1,0]),61)

    def test_decision_training_has_gradients_but_memory_remains_frozen(self):
        decision=FakeDecision();memory=FakeMemory()
        optimizer=torch.optim.SGD(decision.parameters(),lr=.01)
        initial=decision.weight.detach().clone();frozen=memory.weight.detach().clone()
        points=[{'video_id':'v','time':3.25,'question':'visible?','target':'<WAIT> need a view','y':0.0},
                {'video_id':'v','time':4.0,'question':'visible?','target':'<ANSWER> visible','y':1.0}]
        metrics=io.StringIO()
        self.assertEqual(train_video(decision,memory,FakeReader(),points,optimizer,metrics,epoch=0),2)
        self.assertFalse(torch.equal(initial,decision.weight))
        self.assertTrue(torch.equal(frozen,memory.weight))
        self.assertIsNone(memory.weight.grad)
        self.assertEqual(len(metrics.getvalue().splitlines()),2)

    def test_loader_drops_reference_evidence_and_rejects_time_leakage(self):
        row={'video_id':'v','question':'What becomes visible?', 'target':'<WAIT> Waiting for the view.',
             'video_window':[0,3], 'meta':{'query_time':3,'answer_time':8.25,'current_time':3,
                 'reference_visual_evidence':'future answer must never be a model input'}}
        with tempfile.TemporaryDirectory() as directory:
            path=Path(directory)/'qa.jsonl';path.write_text(json.dumps(row)+'\n')
            loaded=load_examples(path)['v'][0]
            self.assertNotIn('meta',loaded)
            self.assertNotIn('reference_visual_evidence',loaded)
            row['target']='<ANSWER> This is visible.'
            path.write_text(json.dumps(row)+'\n')
            with self.assertRaisesRegex(ValueError,'marker conflicts'):load_examples(path)


if __name__=='__main__':unittest.main()
