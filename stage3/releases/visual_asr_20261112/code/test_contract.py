"""CPU regression tests for fractional endpoint evaluation and real loader alignment."""
import copy,json
from pathlib import Path
import unittest
from score_endpoints import score
from validate import rows,load_contract

PACKAGE=Path(__file__).resolve().parents[1]

class ContractTests(unittest.TestCase):
 def test_scorer_accepts_all_reviewed_eval_endpoints(self):
  samples=rows(PACKAGE/'data/all/eval.jsonl')
  predictions=[{'sample_id':s['sample_id'],'status':'ANSWER' if s['meta']['y'] else 'WAIT',
    'prediction':s['target'].removeprefix('<ANSWER> ')} for s in samples]
  result=score(samples,predictions)
  self.assertEqual(result['accuracy'],1)
  self.assertEqual(result['task_outcomes'],{'on_time_at_reviewed_checkpoint':183})
  # This is a scoring oracle test, not a trained model's performance.

 def test_fractional_early_and_missed_answers(self):
  task={'task_id':'fractional','query_time':3.25,'answer_time':4.5}
  samples=[{'sample_id':'q','task_id':'fractional','video_window':[0,3.25],'meta':dict(task,y=0)},
           {'sample_id':'a','task_id':'fractional','video_window':[0,4.5],'meta':dict(task,y=1)}]
  early=score(samples,[{'sample_id':s['sample_id'],'status':'ANSWER','prediction':'x'} for s in samples])
  self.assertEqual(early['task_outcomes'],{'early':1})
  missed=score(samples,[{'sample_id':s['sample_id'],'status':'WAIT'} for s in samples])
  self.assertEqual(missed['task_outcomes'],{'miss':1})

 def test_scorer_rejects_missing_duplicate_unknown_and_bad_status(self):
  s=rows(PACKAGE/'data/all/eval.jsonl')[:1]
  p={'sample_id':s[0]['sample_id'],'status':'WAIT'}
  for predictions in [[],[p,p],[dict(p,sample_id='unknown')],[dict(p,status='other')],[dict(p,status='ANSWER')]]:
   with self.assertRaises(ValueError):score(s,predictions)

if __name__=='__main__':unittest.main()
