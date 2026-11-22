import json
from pathlib import Path
import tempfile
import unittest
from .run_loss_comparison import report


class ReportTests(unittest.TestCase):
    def fixture(self, root):
        def write(path,value):path.write_text(json.dumps(value))
        arms=[];planned=[[['a'],['b'],['c']] for _ in range(100)]
        write(root/'planned_batch_order.json',planned)
        predictions=[dict(sample_id='w',task_id='q',qa_type='event',current_time=0,probability=.1,expected_ready=0,answer='cat',reference=''),dict(sample_id='a',task_id='q',qa_type='event',current_time=1,probability=.9,expected_ready=1,answer='cat',reference='cat')]
        for name in ['bce','cl','hybrid']:
            run=root/name;run.mkdir();arms.append(dict(name=name,run_dir=str(run)))
            write(run/'status.json',dict(status='completed'))
            write(run/'initial_state.json',dict(core=dict(sha256='same')))
            write(run/'optimizer_initialization.json',dict(step=0))
            write(run/'eval_epoch_100.json',dict(metrics={},predictions=predictions))
            (run/'training.jsonl').write_text('\n'.join(json.dumps(dict(phase='train',step=i+1,epoch=i//3+1)) for i in range(300)))
            (run/'batch_order.jsonl').write_text('\n'.join(json.dumps(dict(epoch=i+1,batches=b)) for i,b in enumerate(planned)))
        write(root/'comparison_plan.json',dict(arms=arms))

    def test_report_and_initial_mismatch(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);self.fixture(root)
            result=report(root)
            self.assertEqual(result['status'],'completed')
            self.assertTrue(result['common_initial_weights_verified'])
            self.assertEqual(result['completed_arms_batch_order_verified'],3)
            (root/'hybrid/initial_state.json').write_text(json.dumps(dict(core=dict(sha256='different'))))
            with self.assertRaises(AssertionError):report(root)

    def test_update_budget_mismatch(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);self.fixture(root)
            with (root/'bce/training.jsonl').open('a') as f:f.write('\n'+json.dumps(dict(phase='warmup_update')))
            with self.assertRaises(AssertionError):report(root)


if __name__=='__main__':unittest.main()
