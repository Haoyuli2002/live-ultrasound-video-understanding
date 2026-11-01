"""Validate the actual fixed four-GPU request with fixture data and fake Slurm."""
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest


class Train191SubmissionTests(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory();self.addCleanup(self.temp.cleanup)
        self.root=Path(self.temp.name)
        self.dataset=self.root/'dataset';self.dataset.mkdir()
        self.repo=Path(__file__).resolve().parents[1]
        self.script=self.repo/'scripts/slurm/submit_stage1_train191_4gpu.sh'
        self.binaries=self.root/'bin';self.binaries.mkdir()
        (self.root/'conda.sh').write_text('conda() { return 0; }\n')
        (self.binaries/'python').symlink_to(sys.executable)
        squeue=self.binaries/'squeue';squeue.write_text('#!/bin/bash\nprintf "%s" "${FAKE_ACTIVE:-}"\n');squeue.chmod(0o755)
        sbatch=self.binaries/'sbatch'
        sbatch.write_text('#!/bin/bash\nset -eu\n'
            '[[ "$NPROC_PER_NODE" == 4 && "$EPOCHS" == 1 && "$MAX_STEPS" == -1 ]]\n'
            '[[ "$FRAME_BUDGET" == 30 && "$FRAME_SAMPLING" == recent_sparse ]]\n'
            '[[ "$GRADIENT_ACCUMULATION_STEPS" == 2 && "$VALIDATION_FRACTION" == 0 ]]\n'
            '[[ "$*" == *"--gres=gpu:4"* && "$*" == *"--time=24:00:00"* ]]\n'
            'echo called > "$DATA/sbatch_called"\necho 12345\n')
        sbatch.chmod(0o755)
        mapping={}
        for i in range(191):
            p=self.root/f'v{i}.mp4';p.write_bytes(b'fixture');mapping[p.stem]=str(p)
        self.rows=[]
        for i in range(7408):
            for condition in ('before_with_asr','through_with_asr','before_mask_asr'):
                self.rows.append({'video_id':f'v{i%191}','sentence_id':i,'condition':condition,
                    'sentence_window':[5,10],'video_window':[0,10 if condition=='through_with_asr' else 5],
                    'historical_asr':'' if condition=='before_mask_asr' else 'Earlier narration.',
                    'historical_asr_masked':condition=='before_mask_asr',
                    'target':'The probe is moved across the chest to show the ultrasound image.'})
        (self.dataset/'samples.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in self.rows))
        (self.dataset/'video_map.json').write_text(json.dumps(mapping))
        (self.dataset/'report.json').write_text(json.dumps({'status':'ready'}))
        self.evalmap=self.root/'eval.json';self.evalmap.write_text(json.dumps({'evaluation':str(self.root/'e.mp4')}))
        self.env={**os.environ,'DATA':str(self.root),'TRAIN_DATASET_DIR':str(self.dataset),
                  'INDEPENDENT_EVAL_MAP':str(self.evalmap),'CONDA_SH':str(self.root/'conda.sh'),
                  'PATH':str(self.binaries)+os.pathsep+os.environ['PATH'],'USER':'fixture',
                  # Old shell values must not reduce the full job to a smoke.
                  'NPROC_PER_NODE':'1','MAX_STEPS':'3','EPOCHS':'3','FRAME_BUDGET':'120',
                  'GPU_TYPE':'auto','WALLTIME':'24:00:00','FAKE_ACTIVE':''}

    def run_script(self,*args):
        return subprocess.run(['bash',str(self.script),*args],env=self.env,text=True,capture_output=True)

    def test_real_request_and_persisted_settings(self):
        preview=self.run_script('--dry-run')
        self.assertEqual(preview.returncode,0,preview.stderr)
        self.assertIn('--gres=gpu:4',preview.stdout)
        self.assertIn('h100',preview.stdout);self.assertIn('a100',preview.stdout)
        self.assertFalse((self.root/'sbatch_called').exists())
        result=self.run_script()
        self.assertEqual(result.returncode,0,result.stderr)
        self.assertIn('Job: 12345',result.stdout)
        configs=[json.loads(p.read_text()) for p in self.root.glob('cluster_data/experiments/stage1/*/submission.json')]
        submitted=next(c for c in configs if c['status']=='submitted')
        self.assertEqual(submitted['job_id'],'12345')
        self.assertEqual(submitted['validation_fraction'],0)
        self.assertEqual(submitted['effective_batch_size'],8)
        self.assertEqual(len(submitted['dataset_hashes']),3)

    def test_duplicate_job_and_eval_overlap_stop_before_submission(self):
        self.env['FAKE_ACTIVE']='777 RUNNING'
        result=self.run_script()
        self.assertNotEqual(result.returncode,0);self.assertIn('active',result.stderr)
        self.assertFalse((self.root/'sbatch_called').exists())
        self.env['FAKE_ACTIVE']=''
        self.evalmap.write_text(json.dumps({'v0':str(self.root/'v0.mp4')}))
        result=self.run_script('--dry-run')
        self.assertNotEqual(result.returncode,0);self.assertIn('overlaps',result.stderr)
        self.assertFalse((self.root/'cluster_data/experiments/stage1').exists())

    def test_wrong_condition_triplet_rejected_without_gpu(self):
        self.rows[0]['condition']='through_with_asr'
        (self.dataset/'samples.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in self.rows))
        result=self.run_script('--dry-run')
        self.assertNotEqual(result.returncode,0);self.assertIn('Duplicate condition',result.stderr)
        self.assertFalse((self.root/'sbatch_called').exists())


if __name__=='__main__':unittest.main()
