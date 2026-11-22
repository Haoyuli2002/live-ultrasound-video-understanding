import json
import shutil
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch
from .select_all_pairs import compare, threshold_selection, choose, digest, main
from . import test_loss_comparison_report as report_tests


class SelectionTests(unittest.TestCase):
    def fixture(self,root):
        report_tests.ReportTests().fixture(root)
        shutil.copytree(root/'hybrid',root/'hybrid1')
        arms=[]
        for name in ['bce','cl','hybrid','hybrid1']:
            run=root/name
            cfg=dict(init_checkpoint=None,restore_optimizer=False,zero_update_warmup=True,readiness_objective='bce')
            (run/'experiment.json').write_text(json.dumps(cfg))
            (run/'samples.jsonl').write_text('\n'.join(json.dumps(dict(sample_id=i)) for i in ['w','a']))
            for file in ['tasks.jsonl','video_map.json']:(run/file).write_text('{}')
            (run/'checkpoint_latest.pth').write_text('test fixture only')
            arms.append(dict(name=name,run_dir=str(run),config_sha256=digest(run/'experiment.json')))
        (root/'selection_plan.json').write_text(json.dumps(dict(arms=arms,initial_core_sha256='same',scope='fixture',selection_rule='on-time then FP',project_log=str(root/'programs.md'))))
        (root/'source_hashes.json').write_text('{}')

    def test_reject_incomplete_arm_and_changed_init(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);self.fixture(root)
            self.assertEqual(compare(root)['winner']['name'],'bce')
            (root/'hybrid1/status.json').write_text('{"status":"running"}')
            with self.assertRaises(ValueError):compare(root)
            (root/'hybrid1/status.json').write_text('{"status":"completed"}')
            (root/'hybrid1/initial_state.json').write_text('{"core":{"sha256":"wrong"}}')
            with self.assertRaises(ValueError):compare(root)

    def test_bce_winner_reuses_checkpoint_without_training(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);self.fixture(root)
            with patch('sys.argv',['select_all_pairs','--run-dir',str(root)]),patch('endtoend.select_all_pairs.subprocess.run') as launch:
                main();launch.assert_not_called()
            state=json.loads((root/'status.json').read_text())
            self.assertFalse(state['training_performed'])
            self.assertEqual(state['action'],'reuse_bce_checkpoint')

    def test_cl_winner_prepares_only_sampling_change_before_launch(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);self.fixture(root)
            source=root/'cl'
            cfg=json.loads((source/'experiment.json').read_text())
            cfg.update(readiness_objective='contrastive_gold',experiment='original')
            (source/'experiment.json').write_text(json.dumps(cfg))
            plan=json.loads((root/'selection_plan.json').read_text())
            plan['arms'][1]['config_sha256']=digest(source/'experiment.json')
            # Break equal-metric ties in favor of this CL arm for the fixture.
            plan['arms']=plan['arms'][1:2]+plan['arms'][:1]+plan['arms'][2:]
            (root/'selection_plan.json').write_text(json.dumps(plan))
            for name in ['samples.jsonl','tasks.jsonl','video_map.json']:
                shutil.copy2(source/name,root/name)
            with patch('sys.argv',['select_all_pairs','--run-dir',str(root)]),patch('endtoend.select_all_pairs.subprocess.run',side_effect=RuntimeError('stop before real training')) as launch:
                with self.assertRaisesRegex(RuntimeError,'stop before real training'):main()
                launch.assert_called_once()
            actual=json.loads((root/'experiment.json').read_text())
            self.assertEqual(actual['timing_pair_mode'],'all_pairs')
            self.assertEqual(actual['readiness_objective'],'contrastive_gold')
            self.assertIsNone(actual['init_checkpoint'])
            self.assertFalse(actual['restore_optimizer'])
            self.assertEqual({k for k in actual if actual.get(k)!=cfg.get(k)},
                             {'timing_pair_mode','timing_pair_sampling','experiment'})
            self.assertIn('BEFORE launch',(root/'programs.md').read_text())

    def test_correct_timing_beats_early_correct_text(self):
        rows=[dict(sample_id='w',task_id='q',current_time=0,expected_ready=0,probability=.4,answer='cat',reference='',qa_type='event'),dict(sample_id='a',task_id='q',current_time=1,expected_ready=1,probability=.6,answer='cat',reference='cat',qa_type='event')]
        result=threshold_selection(rows)
        self.assertEqual(result['at035']['first_trigger']['answer_correct'],1)
        self.assertEqual(result['at035']['first_trigger']['on_time_and_text_correct'],0)
        self.assertEqual(result['best']['threshold'],.5)
        self.assertEqual(result['best']['first_trigger']['on_time_and_text_correct'],1)
        self.assertEqual(choose([dict(name='early',evaluation={'best':result['at035']}),dict(name='on_time',evaluation=result)])['name'],'on_time')


if __name__=='__main__':unittest.main()
