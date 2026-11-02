"""CPU-only submission checks; no CUDA allocation or model downloads."""
import json
import os
from pathlib import Path
import subprocess
import tempfile
import unittest

REPO = Path(__file__).resolve().parents[1]


class OverfitSubmissionTests(unittest.TestCase):
    def test_submission_resources_and_frozen_inputs(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            data = root / 'data'
            video = root/'P3FoSFIrfQA.mp4'
            video.write_bytes(b'file-presence fixture; never decoded')
            conda = root/'conda.sh'
            conda.write_text('conda() { :; }\n')
            bindir = root/'bin'
            bindir.mkdir()
            sbatch = bindir/'sbatch'
            sbatch.write_text('#!/bin/bash\nprintf "%s\\n" "$@" > "$CAPTURE_ARGS"\nprintf "987654;test\\n"\n')
            sbatch.chmod(0o755)
            env = dict(os.environ, REPO=str(REPO), DATA=str(data),
                       CONDA_SH=str(conda), STAGE1_ADAPTER=str(root/'stale-missing-adapter'),
                       STAGE2_ADAPTER=str(root/'stale-stage2-adapter'),
                       VIDEO_FILE=str(video), GPU_TYPE='a100', EPOCHS='20',
                       WALLTIME='04:00:00', DRY_RUN='0',
                       PATH=str(bindir)+os.pathsep+os.environ['PATH'],
                       CAPTURE_ARGS=str(root/'sbatch_args.txt'))
            completed = subprocess.run(['bash',str(REPO/'scripts/slurm/submit_stage2_overfit.sh')],
                                       env=env,capture_output=True,text=True)
            self.assertEqual(completed.returncode,0,completed.stderr)
            self.assertIn('Job: 987654',completed.stdout)
            args = (root/'sbatch_args.txt').read_text()
            self.assertIn('--gres=gpu:1',args)
            self.assertIn('--partition=lrz-dgx-a100-80x8,lrz-hgx-a100-80x4',args)
            self.assertIn('--time=04:00:00',args)
            runs = list((data/'cluster_data/experiments/stage2').iterdir())
            self.assertEqual(len(runs),1)
            run = runs[0]
            plan = json.loads((run/'submission.json').read_text())
            self.assertEqual(plan['expected_updates'],240)
            self.assertEqual(plan['label_status'],'draft_debug_only')
            self.assertEqual(plan['INITIALIZATION'],'official_checkpoint')
            self.assertEqual(plan['MODEL_NAME'],'Qwen/Qwen3.5-2B')
            self.assertIsNone(plan['init_adapter'])
            self.assertNotIn('STAGE1_ADAPTER',plan)
            self.assertEqual((run/'job_id.txt').read_text().strip(),'987654')
            self.assertEqual(json.loads((run/'video_map.json').read_text()),
                             {'P3FoSFIrfQA':str(video.resolve())})
            self.assertTrue((run/'code/stage2/train.py').is_file())

    def test_training_and_evaluation_ignore_inherited_adapters(self):
        with tempfile.TemporaryDirectory() as temporary:
            root=Path(temporary)
            conda=root/'conda.sh'
            conda.write_text('conda() { :; }\n')
            bindir=root/'bin'
            bindir.mkdir()
            python=bindir/'python'
            import sys
            python.write_text(f'#!{sys.executable}\n'+'''import json,os,sys
from pathlib import Path
assert 'STAGE1_ADAPTER' not in os.environ
assert 'STAGE2_ADAPTER' not in os.environ
args=sys.argv[1:]
if args==['-']: os.execv(sys.executable,[sys.executable,'-'])
assert '--init-adapter' not in args and '--base-adapter' not in args,args
with open(os.environ['CAPTURE_COMMANDS'],'a') as f:f.write(json.dumps(args)+'\\n')
if args[:2]==['-m','stage2.train']:
 p=Path(args[args.index('--output-dir')+1]);p.mkdir(parents=True)
 (p/'final').mkdir();(p/'final/adapter_config.json').write_text('{}')
 (p/'training_metrics.jsonl').write_text('{}\\n'*24)
elif args[:2]==['-m','stage2.evaluate']:
 p=Path(args[args.index('--output')+1])
 rows=[{'kind':k,'loss':1.0} for k,n in [('short',12),('block',2),('global',2)] for _ in range(n)]
 p.write_text(''.join(json.dumps(r)+'\\n' for r in rows))
''')
            python.chmod(0o755)
            env=dict(os.environ,CONDA_SH=str(conda),CONDA_ENV='test',CODE_DIR=str(root),
                     RUN_DIR=str(root),OUTPUT_DIR=str(root/'training'),
                     TRAIN_JSONL='unused',VIDEO_MAP='unused',EPOCHS='2',
                     LEARNING_RATE='1e-4',FRAME_SIZE='224',MODEL_NAME='stale-model',
                     STAGE1_ADAPTER=str(root/'missing'),STAGE2_ADAPTER='stale',
                     PATH=str(bindir)+os.pathsep+os.environ['PATH'],
                     CAPTURE_COMMANDS=str(root/'commands.jsonl'))
            result=subprocess.run(['bash',str(REPO/'scripts/slurm/run_stage2_overfit.sbatch')],
                                  env=env,capture_output=True,text=True)
            self.assertEqual(result.returncode,0,result.stderr)
            commands=[json.loads(s) for s in (root/'commands.jsonl').read_text().splitlines()]
            train=next(c for c in commands if c[:2]==['-m','stage2.train'])
            self.assertEqual(train[train.index('--model-name')+1],'Qwen/Qwen3.5-2B')
            self.assertEqual(len([c for c in commands if c[:2]==['-m','stage2.evaluate']]),2)
            report=json.loads((root/'overfit_report.json').read_text())
            self.assertIsNone(report['init_adapter'])
            self.assertEqual(report['model_name'],'Qwen/Qwen3.5-2B')


if __name__=='__main__':
    unittest.main()
