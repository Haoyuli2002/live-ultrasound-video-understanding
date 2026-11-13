"""Offline submission handoff checks; Slurm commands and training submitter are mocked."""
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import unittest

REPO=Path(__file__).resolve().parents[1]
RELEASE=REPO/'stage2/releases/accepted_v4_20261105_990b9b48'


class ReviewedSubmissionTests(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory(prefix='stage2_reviewed_handoff_')
        self.addCleanup(self.temp.cleanup)
        self.root=Path(self.temp.name);self.repo=self.root/'repo'
        scripts=self.repo/'scripts/slurm';scripts.mkdir(parents=True)
        self.helper=scripts/'submit_stage2_reviewed.sh'
        shutil.copy2(REPO/'scripts/slurm/submit_stage2_reviewed.sh',self.helper)
        dest=self.repo/'stage2/releases';dest.mkdir(parents=True)
        (dest/RELEASE.name).symlink_to(RELEASE,target_is_directory=True)
        conda=self.root/'conda.sh';conda.write_text('conda() { :; }\n')
        bins=self.root/'bin';bins.mkdir()
        (bins/'python').symlink_to(sys.executable)
        q=bins/'squeue';q.write_text('#!/bin/bash\necho "590999 PENDING"\n');q.chmod(0o755)
        fake=scripts/'submit_stage2_full_training.sh'
        fake.write_text('''#!/bin/bash
set -eu
python - <<'CAPTURE'
import json,os
from pathlib import Path
keys=['DATASET_RELEASE','DATASET_VERSION','NPROC_PER_NODE','EPOCHS','WALLTIME','STAGE1_ADAPTER']
p=Path(os.environ['CAPTURE'])
with p.open('a') as f:f.write(json.dumps({k:os.environ[k] for k in keys})+'\\n')
CAPTURE
echo "Job: 590999"
echo "Run: $DATA/mock_run"
[[ ${FAIL_AFTER_PRINT:-0} != 1 ]]
''')
        self.data=self.root/'data'
        self.records=self.data/'cluster_data/experiments/stage2/accepted_v4_20261105_990b9b48_submissions'
        self.capture=self.root/'capture.jsonl'
        self.env=dict(os.environ,REPO=str(self.repo),DATA=str(self.data),CONDA_SH=str(conda),
                      CAPTURE=str(self.capture),PYTHONPATH=str(REPO),
                      PATH=str(bins)+os.pathsep+os.environ['PATH'],
                      NPROC_PER_NODE='1',EPOCHS='999',WALLTIME='00:01:00')
        for name in ['DATASET_RELEASE','RESUME_RUN','QFORMER_SMOKE_REPORT']:
            self.env.pop(name,None)

    def run_helper(self,arch='ordinary',version=None,ok=True,**extra):
        command=['bash',str(self.helper),arch]
        if version is not None:command.append(version)
        r=subprocess.run(command,env=dict(self.env,**extra),
                         cwd=self.root,capture_output=True,text=True)
        self.assertEqual(r.returncode==0,ok,r.stdout+'\n'+r.stderr)
        return r

    def test_versioned_release_defaults_and_fixed_resources(self):
        self.run_helper()
        cfg=json.loads(self.capture.read_text())
        self.assertEqual(Path(cfg['DATASET_RELEASE']),self.repo/'stage2/releases'/RELEASE.name)
        self.assertEqual([cfg[k] for k in ['NPROC_PER_NODE','EPOCHS','WALLTIME']],['4','1','24:00:00'])
        self.assertEqual(cfg['DATASET_VERSION'],'visual_only')
        self.assertTrue(cfg['STAGE1_ADAPTER'].endswith('stage1_v21_train191_2b_4gpu_f30_e1_20261101T071541Z_sCe9Fb/final'))
        self.assertEqual((self.records/'ordinary.job_id.txt').read_text().strip(),'590999')

    def test_repeat_does_not_resubmit(self):
        self.run_helper();second=self.run_helper()
        self.assertIn('Already recorded',second.stdout)
        self.assertEqual(len(self.capture.read_text().splitlines()),1)

    def test_successful_sbatch_log_recovered_after_disconnect(self):
        self.run_helper(ok=False,FAIL_AFTER_PRINT='1')
        self.assertFalse((self.records/'ordinary.job_id.txt').exists())
        self.run_helper()
        self.assertEqual(len(self.capture.read_text().splitlines()),1)
        self.assertEqual((self.records/'ordinary.job_id.txt').read_text().strip(),'590999')

    def test_missing_smoke_does_not_submit_qformer(self):
        result=self.run_helper('qformer',ok=False)
        self.assertIn('waits for passed smoke report',result.stderr)
        self.assertFalse(self.capture.exists())
        self.assertFalse((self.records/'qformer.lock').exists())

    def test_secondwise_records_do_not_reuse_legacy_qformer_submission(self):
        self.records.mkdir(parents=True)
        (self.records/'qformer.job_id.txt').write_text('1234\n')
        (self.records/'qformer.run_dir.txt').write_text('/old/window-level-run\n')
        report=self.root/'new_smoke.json';report.write_text('{}')
        self.run_helper('qformer',QFORMER_SMOKE_REPORT=str(report))
        self.assertEqual((self.records/'qformer_secondwise_v2.job_id.txt').read_text().strip(),'590999')
        self.assertEqual((self.records/'qformer.job_id.txt').read_text().strip(),'1234')
        self.run_helper('qformer',QFORMER_SMOKE_REPORT=str(report))
        self.assertEqual(len(self.capture.read_text().splitlines()),1)

    def test_other_manifest_never_reaches_submitter(self):
        release=self.root/'wrong_release';release.mkdir()
        (release/'manifest.json').write_text('{}')
        result=self.run_helper(ok=False,DATASET_RELEASE=str(release))
        self.assertIn('pinned to accepted_v4',result.stderr)
        self.assertFalse(self.capture.exists())

    def test_visual_asr_candidate_is_not_a_trainable_release(self):
        result=self.run_helper('ordinary','visual_asr',ok=False)
        self.assertIn('Missing frozen visual_asr release',result.stderr)
        self.assertFalse(self.capture.exists())

    def test_existing_lock_does_not_submit(self):
        lock=self.records/'ordinary.lock';lock.mkdir(parents=True)
        result=self.run_helper(ok=False)
        self.assertIn('holds',result.stderr)
        self.assertFalse(self.capture.exists())
        self.assertTrue(lock.exists())


if __name__=='__main__':unittest.main()
