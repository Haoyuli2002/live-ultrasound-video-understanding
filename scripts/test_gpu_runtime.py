import os
from pathlib import Path
import subprocess
import tempfile
import unittest
from unittest.mock import patch

import torch
from scripts.gpu_profiles import partitions
from scripts.gpu_runtime import model_kwargs


class GPUCompatibilityTests(unittest.TestCase):
    def test_a100_and_h100_use_the_same_portable_model_settings(self):
        for capability in ((8,0),(9,0)):
            with patch.object(torch.cuda,'is_available',return_value=True), \
                    patch.object(torch.cuda,'is_bf16_supported',return_value=True), \
                    patch.object(torch.cuda,'get_device_capability',return_value=capability):
                self.assertEqual(model_kwargs(torch.bfloat16,'auto'),
                    {'dtype':torch.bfloat16,'attn_implementation':'sdpa'})

    def test_explicit_unsupported_settings_fail_before_loading_weights(self):
        with self.assertRaises(ValueError):model_kwargs(torch.float32,'flash_attention_3')
        with patch.object(torch.cuda,'is_available',return_value=False):
            with self.assertRaisesRegex(RuntimeError,'needs CUDA'):model_kwargs(torch.bfloat16,'flash_attention_2')
        with patch.object(torch.cuda,'is_available',return_value=True), \
                patch.object(torch.cuda,'is_bf16_supported',return_value=False):
            with self.assertRaisesRegex(RuntimeError,'does not support BF16'):model_kwargs(torch.bfloat16,'sdpa')

    def test_partitions_exclude_small_mig_slices(self):
        self.assertIn('h100',partitions('h100'))
        self.assertNotIn('h100',partitions('a100'))
        self.assertIn('80x8',partitions('a100'))
        self.assertNotIn('mig',partitions('auto'))
        with self.assertRaises(ValueError):partitions('wrong')

    def test_all_three_stage_submissions_are_reviewable_without_slurm(self):
        repo=Path(__file__).resolve().parents[1]
        script=repo/'scripts/slurm/submit_training.sh'
        for stage in ('1','2','3'):
            for gpu in ('a100','h100','auto'):
                env={**os.environ,'REPO':str(repo),'GPU_TYPE':gpu,'NPROC_PER_NODE':'1',
                     'TRAIN_JSONL':'/data/train.jsonl','VIDEO_MAP':'/data/videos.json',
                     'OUTPUT_DIR':'/data/output','STAGE1_ADAPTER':'/weights/stage1',
                     'STAGE2_ADAPTER':'/weights/stage2'}
                result=subprocess.run(['bash',str(script),stage,'--dry-run'],env=env,text=True,capture_output=True)
                self.assertEqual(result.returncode,0,result.stderr)
                self.assertIn('gpu:1',result.stdout)
                self.assertIn(f'--job-name=stage{stage}-train',result.stdout)
                self.assertIn('80x8' if gpu=='a100' else 'h100',result.stdout)


if __name__=='__main__':unittest.main()
