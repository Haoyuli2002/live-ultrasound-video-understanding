import json
from pathlib import Path
import tempfile
import unittest
from .progress import TrainingProgress


class Clock:
    def __init__(self):self.now=0.
    def __call__(self):return self.now
    def advance(self,s):self.now+=s


class ProgressTests(unittest.TestCase):
    def test_resume_ga_tail_and_phase_eta(self):
        clock=Clock();logs=[]
        with tempfile.TemporaryDirectory() as tmp:
            with TrainingProgress(tmp,examples=9,batch_size=2,world_size=2,accumulation=2,
                                  epochs=3,completed_epochs=1,holdout=True,save_every=1,
                                  clock=clock,emit=lambda s,**kw:logs.append(s)) as p:
                self.assertEqual((p.step,p.total,p.steps_per_epoch),(2,6,2))
                self.assertIsNone(p.snapshot()['eta_training_updates_seconds'])
                p.start_epoch(2)
                clock.advance(4);p.update(processed=8)
                self.assertEqual(p.step,3)
                self.assertEqual(p.snapshot()['eta_training_updates_seconds'],12)
                self.assertIsNone(p.snapshot()['eta_job_seconds'])
                clock.advance(2);p.update(processed=9)
                p.phase('holdout');clock.advance(10);p.holdout_batch(5,5)
                self.assertEqual(p.step,4)
                p.phase('checkpoint');clock.advance(4);p.phase('epoch_complete')
                self.assertEqual(p.snapshot()['eta_job_seconds'],20)
                p.start_epoch(3);clock.advance(3);p.update(processed=8)
                self.assertEqual(p.snapshot()['recent_seconds_per_update'],3)
                clock.advance(3);p.update(processed=9)
                self.assertNotEqual(p.phase_name,'completed')
                p.phase('holdout');clock.advance(10);p.phase('checkpoint');clock.advance(4)
                p.phase('epoch_complete');p.finish()
            r=json.loads((Path(tmp)/'progress.json').read_text())
            self.assertEqual(r['phase'],'completed');self.assertEqual(r['global_step'],6)
            self.assertEqual(r['session_completed_steps'],4);self.assertEqual(r['eta_job_seconds'],0)
            self.assertFalse(list(Path(tmp).glob('*.partial')))
            self.assertTrue(any('optimizer steps 6/6' in s for s in logs))

    def test_failure_is_not_a_completed_update(self):
        with tempfile.TemporaryDirectory() as tmp:
            p=TrainingProgress(tmp,examples=5,batch_size=2,world_size=1,accumulation=2,epochs=1,emit=lambda *a,**kw:None)
            with self.assertRaisesRegex(RuntimeError,'synthetic'):
                with p:
                    p.start_epoch(1);p.update(processed=4)
                    raise RuntimeError('synthetic optimizer failure')
            r=json.loads((Path(tmp)/'progress.json').read_text())
            self.assertEqual(r['global_step'],1);self.assertEqual(r['total_steps'],2)
            self.assertEqual(r['phase'],'failed');self.assertIsNone(r['eta_job_seconds'])

    def test_disabled_rank_does_not_write(self):
        with tempfile.TemporaryDirectory() as tmp:
            path=Path(tmp)/'rank1'
            with TrainingProgress(path,examples=9,batch_size=2,world_size=2,accumulation=2,epochs=1,enabled=False) as p:
                p.start_epoch(1);p.update(processed=8);p.phase('holdout');p.finish()
            self.assertFalse(path.exists())

if __name__=='__main__':unittest.main()
