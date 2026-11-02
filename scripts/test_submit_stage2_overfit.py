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
            adapter = root / 'adapter'
            adapter.mkdir()
            (adapter/'adapter_config.json').write_text('{}')
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
                       CONDA_SH=str(conda), STAGE1_ADAPTER=str(adapter),
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
            self.assertEqual((run/'job_id.txt').read_text().strip(),'987654')
            self.assertEqual(json.loads((run/'video_map.json').read_text()),
                             {'P3FoSFIrfQA':str(video.resolve())})
            self.assertTrue((run/'code/stage2/train.py').is_file())

    def test_missing_stage1_adapter_stops_before_submission(self):
        with tempfile.TemporaryDirectory() as temporary:
            root=Path(temporary)
            conda=root/'conda.sh'
            conda.write_text('conda() { :; }\n')
            env=dict(os.environ,REPO=str(REPO),DATA=str(root/'data'),
                     CONDA_SH=str(conda),STAGE1_ADAPTER=str(root/'missing'))
            result=subprocess.run(['bash',str(REPO/'scripts/slurm/submit_stage2_overfit.sh')],
                                  env=env,capture_output=True,text=True)
            self.assertEqual(result.returncode,2)
            self.assertIn('completed Stage 1 adapter',result.stderr)
            self.assertFalse((root/'data').exists())


if __name__=='__main__':
    unittest.main()
