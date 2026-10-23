import gc
import json
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import weakref
import torch
from stage1 import runtime, train
from stage1.data import build_rows

class RuntimeTests(unittest.TestCase):
    def test_invalid_strategy_is_actionable(self):
        with patch.dict(os.environ, {'STAGE1_SHARING_STRATEGY': 'bad'}):
            with self.assertRaisesRegex(ValueError, 'Unsupported STAGE1_SHARING_STRATEGY'):
                runtime.configure_sharing()

    def test_zero_workers_does_not_start_process(self):
        with patch.object(runtime.subprocess, 'Popen') as popen:
            runtime.preflight(0)
            popen.assert_not_called()

    def test_long_socket_directory_fails_before_training(self):
        with tempfile.TemporaryDirectory(dir='/tmp') as root:
            long = Path(root) / ('x' * 100)
            long.mkdir()
            with patch.object(tempfile, 'tempdir', str(long)):
                with self.assertRaisesRegex(OSError, "AF_UNIX path too long"):
                    runtime.check_socket_path()

    def test_cpu_oom_is_not_treated_as_cuda_oom(self):
        self.assertFalse(train._is_cuda_oom(RuntimeError('CPU out of memory')))
        self.assertTrue(train._is_cuda_oom(RuntimeError('CUDA out of memory.')))
        self.assertTrue(train._is_cuda_oom(torch.cuda.OutOfMemoryError('oom')))

    def test_oom_releases_failed_model_before_cuda_cleanup(self):
        refs, budgets = [], []
        class Model:
            pass
        def attempt(*args, **kwargs):
            budgets.append(kwargs['frame_budget'])
            if len(budgets) == 1:
                model = Model()
                refs.append(weakref.ref(model))
                raise torch.cuda.OutOfMemoryError('CUDA out of memory')
        def cleanup():
            gc.collect()
            self.assertIsNone(refs[0](), 'failed model retained by exception traceback')
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            rows = build_rows({'video_id': 'v', 'segments': [
                {'start': 0, 'end': 5, 'text': 'This is the liver.'},
                {'start': 5, 'end': 10, 'text': 'This is the kidney.'}]})
            (root/'train.jsonl').write_text(''.join(json.dumps(row)+'\n' for row in rows))
            (root/'map.json').write_text(json.dumps({'v': '/unused/video.mp4'}))
            argv = ['train', '--train-jsonl', str(root/'train.jsonl'),
                    '--video-path-map', str(root/'map.json'), '--output-dir', str(root/'out'),
                    '--frame-budget', '48', '--dataloader-num-workers', '0']
            with patch('sys.argv', argv), patch.object(torch.cuda, 'is_available', return_value=True), \
                 patch.object(train, 'preflight'), patch.object(train, 'build_and_train', side_effect=attempt), \
                 patch.object(train, '_free_cuda', side_effect=cleanup) as free:
                train.main()
                free.assert_called_once()
            self.assertEqual(budgets, [48, 36])

if __name__ == '__main__':
    unittest.main()
