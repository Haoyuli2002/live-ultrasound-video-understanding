import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
from .train_eval_pipeline import balanced_shards, official_smoke_gate, aggregate
from .train_full import initialization_identity, save_epoch


class PipelineTests(unittest.TestCase):
    def test_official_initialization_does_not_open_stage1_adapter(self):
        with patch('stage2.train_full.digest',side_effect=AssertionError('Unexpected Stage 1 read')):
            self.assertEqual(initialization_identity(None),dict(init_adapter=None,init_adapter_sha256=None))

    def test_epoch_weights_are_kept_separately(self):
        class Memory:
            def save(self,path):
                p=Path(path);p.mkdir();(p/'weight').write_text(p.name)
        with tempfile.TemporaryDirectory() as d:
            for epoch in range(1,4):save_epoch(Memory(),d,epoch)
            self.assertEqual([p.name for p in sorted(Path(d).iterdir())],['epoch-1','epoch-2','epoch-3'])

    def test_eval_shards_cover_eval_once_and_exclude_train(self):
        manifest={'videos':[dict(video_id=str(i),split='eval',targets=i+1) for i in range(41)]+[
            dict(video_id='train',split='train',targets=10000)]}
        shards=balanced_shards(manifest,8)
        flat=[v for shard in shards for v in shard]
        self.assertEqual(len(shards),8)
        self.assertEqual(len(flat),len(set(flat)))
        self.assertEqual(set(flat),set(map(str,range(41))))

    def test_eval_command_preserves_leading_hyphen_video_ids(self):
        from unittest.mock import MagicMock
        from .reconstruct import build_parser
        from .train_eval_pipeline import evaluate_checkpoint
        ids = ['normal', '-awazvNpICg', 'AH8RcdMQMHA']
        manifest = {'videos': [dict(video_id=vid, split='eval', targets=3-i)
                               for i, vid in enumerate(ids)]}
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            (root/'logs').mkdir()
            child = MagicMock()
            child.wait.return_value = 0
            child.poll.return_value = 0
            with patch('stage2.train_eval_pipeline.subprocess.Popen', return_value=child) as popen, \
                 patch('stage2.train_eval_pipeline.aggregate', return_value={'records': 9}):
                evaluate_checkpoint(root, 1, manifest, {}, ['0'])
            cmd = popen.call_args.args[0]
            self.assertEqual(build_parser().parse_args(cmd[3:]).video_ids, ids)
            legacy = cmd[:cmd.index('--video-id=normal')] + ['--video-ids', 'normal', 'another']
            self.assertEqual(build_parser().parse_args(legacy[3:]).video_ids, ['normal', 'another'])

    def test_stage1_initialized_smoke_is_rejected(self):
        with tempfile.TemporaryDirectory() as d:
            root=Path(d);(root/'smoke').mkdir()
            (root/'smoke/smoke_report.json').write_text(json.dumps(dict(status='functional_smoke_passed',init_adapter_sha256='stage1')))
            with self.assertRaisesRegex(ValueError,'official-initialized'):official_smoke_gate(root)

    def test_pipeline_runs_official_ddp_then_eight_eval_workers(self):
        from .train_eval_pipeline import run
        from unittest.mock import MagicMock
        with tempfile.TemporaryDirectory() as d:
            root=Path(d);(root/'logs').mkdir()
            (root/'submission.json').write_text(json.dumps({'dataset_version':'visual_asr'}))
            manifest={'videos':[dict(video_id=str(i),split='eval',targets=8-i) for i in range(8)]}
            def train(cmd, env):
                self.assertNotIn('--init-adapter',cmd)
                self.assertEqual(cmd[cmd.index('--dataset-version')+1],'visual_asr')
                self.assertEqual(cmd[cmd.index('--frame-size')+1],'256')
                self.assertIn('--nproc-per-node=8',cmd)
                self.assertEqual(cmd[cmd.index('--epochs')+1],1)
                (root/'training').mkdir()
                (root/'training/training_config.json').write_text(json.dumps(dict(init_adapter=None,epochs=1)))
            child=MagicMock();child.wait.return_value=0;child.poll.return_value=0
            with patch('stage2.dataset_versions.validate_dataset_version'), \
                 patch('stage2.release.validate_release',return_value=manifest), \
                 patch('stage2.train_eval_pipeline.official_smoke_gate',side_effect=AssertionError('Smoke must not run')), \
                 patch('stage2.train_eval_pipeline.execute',side_effect=train) as execute, \
                 patch('stage2.train_eval_pipeline.subprocess.Popen',return_value=child) as popen, \
                 patch('stage2.train_eval_pipeline.aggregate',return_value={'records':24}), \
                 patch.dict('os.environ',{'CUDA_VISIBLE_DEVICES':'0,1,2,3,4,5,6,7'}):
                run(root,8,1)
            self.assertEqual(execute.call_count,1)
            self.assertEqual(popen.call_count,8)
            for i,call in enumerate(popen.call_args_list):
                self.assertEqual(call.kwargs['env']['CUDA_VISIBLE_DEVICES'],str(i))
                self.assertIn(str(root/'training/final'),call.args[0])
            self.assertEqual(json.loads((root/'completed.json').read_text())['status'],'completed')

    def test_three_epochs_train_then_evaluate_each_saved_checkpoint_without_smoke(self):
        from .train_eval_pipeline import run
        with tempfile.TemporaryDirectory() as d:
            root=Path(d);(root/'training').mkdir()
            (root/'submission.json').write_text(json.dumps({'dataset_version':'visual_asr'}))
            events=[]
            def train(cmd,env):
                self.assertIn('--resume',cmd)
                self.assertNotIn('--init-adapter',cmd)
                self.assertEqual(cmd[cmd.index('--dataset-version')+1],'visual_asr')
                self.assertEqual(cmd[cmd.index('--frame-size')+1],'256')
                self.assertEqual(cmd[cmd.index('--epochs')+1],3)
                stop=cmd[cmd.index('--stop-after-epoch')+1]
                events.append(('train',stop))
                (root/'training/training_config.json').write_text(json.dumps(dict(init_adapter=None,epochs=3)))
                (root/'training'/f'epoch-{stop}.completed.json').write_text('{}')
            def evaluate(root,workers,manifest,env,gpu_ids,checkpoint,eval_dir):
                self.assertEqual(workers,8)
                self.assertEqual(eval_dir,root/'eval'/checkpoint)
                events.append(('eval',checkpoint))
                return {'records':16770}
            with patch('stage2.dataset_versions.validate_dataset_version'), \
                 patch('stage2.release.validate_release',return_value={}), \
                 patch('stage2.train_eval_pipeline.execute',side_effect=train), \
                 patch('stage2.train_eval_pipeline.evaluate_checkpoint',side_effect=evaluate), \
                 patch('stage2.train_eval_pipeline.official_smoke_gate',side_effect=AssertionError('No smoke')), \
                 patch.dict('os.environ',{'CUDA_VISIBLE_DEVICES':'0,1,2,3,4,5,6,7'}):
                run(root,8,3,True)
            self.assertEqual(events,[('train',1),('eval','epoch-1'),('train',2),('eval','epoch-2'),('train',3),('eval','epoch-3')])
            self.assertEqual(len(json.loads((root/'completed.json').read_text())['evaluations']),3)
