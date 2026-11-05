import json
import os
from pathlib import Path
import subprocess
import tempfile
import unittest
import sys
from stage2.provenance import digest

REPO=Path(__file__).resolve().parents[1]


class FullSubmissionGateTests(unittest.TestCase):
    def test_old_window_level_smoke_cannot_authorize_secondwise_training(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);bins=root/'bin';bins.mkdir()
            (bins/'python').symlink_to(sys.executable)
            git=bins/'git';git.write_text('#!/bin/bash\nexit 0\n');git.chmod(0o755)
            for name in ['sbatch','squeue']:
                f=bins/name;f.write_text('#!/bin/bash\necho contacted >> "$CAPTURE"\n');f.chmod(0o755)
            conda=root/'conda.sh';conda.write_text('conda() { :; }\n')
            adapter=root/'adapter';adapter.mkdir()
            (adapter/'adapter_config.json').write_text('{}')
            (adapter/'adapter_model.safetensors').write_bytes(b'fixture')
            release=REPO/'stage2/releases/accepted_v4_20261105_990b9b48'
            manifest=json.loads((release/'manifest.json').read_text())
            report=root/'legacy_smoke.json'
            report.write_text(json.dumps(dict(status='functional_smoke_passed',frame_size=224,
                model_name='Qwen/Qwen3.5-2B',init_adapter_sha256=digest(adapter/'adapter_model.safetensors'),
                source_sha256=next(v['source_sha256'] for v in manifest['videos'] if v['video_id']=='P3FoSFIrfQA'),
                qformer_config=dict(width=512,layers=2,heads=8,architecture='qformer_style_v1',short_queries=10))))
            env=dict(os.environ,REPO=str(REPO),CONDA_SH=str(conda),DATASET_RELEASE=str(release),
                     STAGE1_ADAPTER=str(adapter),QFORMER_SMOKE_REPORT=str(report),
                     DATA=str(root/'data'),CAPTURE=str(root/'slurm_calls'),GPU_TYPE='auto',
                     PATH=str(bins)+os.pathsep+os.environ['PATH'])
            result=subprocess.run(['bash',str(REPO/'scripts/slurm/submit_stage2_full_training.sh'),'qformer'],
                                  cwd=REPO,env=env,capture_output=True,text=True)
            self.assertNotEqual(result.returncode,0)
            self.assertIn('Rerun smoke with the secondwise Q-Former',result.stderr)
            self.assertFalse((root/'slurm_calls').exists())
            self.assertFalse((root/'data').exists())

    def test_incomplete_release_never_contacts_slurm(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);release=root/'release';release.mkdir()
            (release/'manifest.json').write_text(json.dumps({'training_ready':False,'full_dataset_complete':False}))
            conda=root/'conda.sh';conda.write_text('conda() { :; }\n')
            bins=root/'bin';bins.mkdir()
            git=bins/'git';git.write_text('#!/bin/bash\nexit 0\n');git.chmod(0o755)
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
