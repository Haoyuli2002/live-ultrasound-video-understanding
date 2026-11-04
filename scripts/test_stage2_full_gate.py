import json
import os
from pathlib import Path
import subprocess
import tempfile
import unittest

REPO=Path(__file__).resolve().parents[1]


class FullSubmissionGateTests(unittest.TestCase):
    def test_incomplete_release_never_contacts_slurm(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);release=root/'release';release.mkdir()
            (release/'manifest.json').write_text(json.dumps({'training_ready':False,'full_dataset_complete':False}))
            conda=root/'conda.sh';conda.write_text('conda() { :; }\n')
            bins=root/'bin';bins.mkdir()
            for tool in ['sbatch','squeue']:
                f=bins/tool;f.write_text('#!/bin/bash\necho contacted >> "$CAPTURE"\necho 12345\n');f.chmod(0o755)
            env=dict(os.environ,REPO=str(REPO),CONDA_SH=str(conda),DATASET_RELEASE=str(release),
                     DATA=str(root/'data'),CAPTURE=str(root/'slurm_calls'),DRY_RUN='0',
                     PATH=str(bins)+os.pathsep+os.environ['PATH'])
            for variant in ['ordinary','qformer']:
                result=subprocess.run(['bash',str(REPO/'scripts/slurm/submit_stage2_full_training.sh'),variant],
                                      env=env,capture_output=True,text=True)
                self.assertNotEqual(result.returncode,0)
                self.assertIn('complete frozen visually accepted release',result.stderr)
            self.assertFalse((root/'slurm_calls').exists())
            self.assertFalse((root/'data').exists())


if __name__=='__main__':unittest.main()
