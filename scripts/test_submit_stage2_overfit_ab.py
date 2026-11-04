"""Exercise real submission planning with a committed fixture and fake Slurm."""
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import unittest

REPO = Path(__file__).resolve().parents[1]


class PairSubmissionTests(unittest.TestCase):
    def fixture(self, root):
        repo = root/"repo"
        repo.mkdir()
        files = ["scripts/gpu_profiles.py", "scripts/slurm/submit_stage2_overfit_ab.sh",
                 "scripts/slurm/run_stage2_overfit_ab.sbatch", "stage2/data.py",
                 "stage2/summary_schema.py", "stage2/train_overfit_ab.py",
                 "stage2/overfit_ab_report.py", "stage2/evaluate.py", "stage2/reconstruct.py",
                 "stage2/examples/P3FoSFIrfQA/blocks.jsonl",
                 "stage2/examples/P3FoSFIrfQA/provenance.json"]
        for name in files:
            target = repo/name
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(REPO/name, target)
        for command in [["git","init","-q"], ["git","add","."],
                        ["git","-c","user.name=Test","-c","user.email=test@example.org", "commit","-qm","fixture"]]:
            subprocess.run(command, cwd=repo, check=True, capture_output=True)
        conda = root/"conda.sh"
        conda.write_text("conda() { :; }\n")
        video = root/"P3FoSFIrfQA.mp4"
        video.write_bytes(b"presence fixture only")
        bin_dir = root/"bin"
        bin_dir.mkdir()
        sbatch = bin_dir/"sbatch"
        sbatch.write_text('''#!/bin/bash
if [[ -n "$STAGE1_ADAPTER$STAGE2_ADAPTER$INIT_ADAPTER" ]]; then exit 7; fi
if [[ "$FAIL_SECOND" == 1 && "$MODE" == independent ]]; then exit 9; fi
printf '%s\n' "$@" > "$CAPTURE_DIR/$MODE.args"
if [[ "$MODE" == short_only ]]; then echo '12345;fixture'; else echo 12346; fi
''')
        sbatch.chmod(0o755)
        env = dict(os.environ, REPO=str(repo), DATA=str(root/"data"), CONDA_SH=str(conda),
                   VIDEO_FILE=str(video), GPU_TYPE="a100", EPOCHS="20", WALLTIME="04:00:00",
                   PATH=str(bin_dir)+os.pathsep+os.environ["PATH"], CAPTURE_DIR=str(root),
                   STAGE1_ADAPTER="stale-stage1", STAGE2_ADAPTER="stale-stage2",
                   FAIL_SECOND="0", DRY_RUN="0")
        return repo, env

    def run_script(self, repo, env):
        return subprocess.run(["bash", str(repo/"scripts/slurm/submit_stage2_overfit_ab.sh")],
                              env=env, capture_output=True, text=True)

    def test_two_jobs_same_snapshot_and_distinct_modes_counts(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            repo, env = self.fixture(root)
            result = self.run_script(repo, env)
            self.assertEqual(result.returncode, 0, result.stderr)
            pair = next((root/"data/cluster_data/experiments/stage2").iterdir())
            manifest = json.loads((pair/"experiments.json").read_text())
            self.assertEqual(manifest["status"], "submitted")
            self.assertEqual([j["job_id"] for j in manifest["jobs"]], ["12345", "12346"])
            plans = [json.loads((pair/m/"submission.json").read_text())
                     for m in ["short_only", "independent"]]
            self.assertEqual([p["expected_updates"] for p in plans], [240, 320])
            self.assertEqual([p["expected_predictions"] for p in plans], [36, 48])
            self.assertEqual(plans[0]["samples_sha256"], plans[1]["samples_sha256"])
            self.assertEqual(plans[0]["code_dir"], plans[1]["code_dir"])
            for mode in ["short_only", "independent"]:
                args = (root/f"{mode}.args").read_text()
                self.assertIn("--gres=gpu:1", args)
                self.assertIn("--partition=lrz-dgx-a100-80x8,lrz-hgx-a100-80x4", args)
                self.assertIn("--time=04:00:00", args)
                self.assertNotIn("--dependency", args)

    def test_dry_run_no_jobs_and_failed_second_preserves_first_job_id(self):
        for dry_run in [True, False]:
            with self.subTest(dry_run=dry_run), tempfile.TemporaryDirectory() as tmp:
                root = Path(tmp)
                repo, env = self.fixture(root)
                env.update(DRY_RUN="1" if dry_run else "0", FAIL_SECOND="1")
                result = self.run_script(repo, env)
                self.assertEqual(result.returncode, 0 if dry_run else 9, result.stderr)
                pair = next((root/"data/cluster_data/experiments/stage2").iterdir())
                manifest = json.loads((pair/"experiments.json").read_text())
                if dry_run:
                    self.assertFalse((root/"short_only.args").exists())
                    self.assertTrue(all(j["job_id"] is None for j in manifest["jobs"]))
                else:
                    self.assertEqual(manifest["status"], "partially_submitted")
                    self.assertEqual(manifest["jobs"][0]["job_id"], "12345")
                    self.assertIsNone(manifest["jobs"][1]["job_id"])

    def test_missing_video_submits_neither_job(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            repo, env = self.fixture(root)
            env["VIDEO_FILE"] = str(root/"missing.mp4")
            result = self.run_script(repo, env)
            self.assertNotEqual(result.returncode, 0)
            self.assertIn("P3 video missing", result.stderr)
            self.assertFalse((root/"short_only.args").exists())
            self.assertFalse((root/"independent.args").exists())

    def test_invalid_gpu_type_stops_before_preparing_or_submitting(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            repo, env = self.fixture(root)
            env["GPU_TYPE"] = "mistyped"
            result = self.run_script(repo, env)
            self.assertNotEqual(result.returncode,0)
            self.assertFalse((root/"data").exists())
            self.assertFalse((root/"short_only.args").exists())

    def test_batch_runs_training_tf_and_generation_for_the_right_kinds(self):
        for mode, kinds in [("short_only",["short"]),("independent",["short","block","global"])]:
            with self.subTest(mode=mode), tempfile.TemporaryDirectory() as tmp:
                root = Path(tmp)
                repo, env = self.fixture(root)
                python = root/"bin/python"
                python.write_text(f'#!{sys.executable}\n'+'''import json,os,sys
assert not any(os.environ.get(k) for k in ['STAGE1_ADAPTER','STAGE2_ADAPTER','INIT_ADAPTER'])
with open(os.environ['COMMANDS'],'a') as stream: stream.write(json.dumps(sys.argv[1:])+'\\n')
''')
                python.chmod(0o755)
                env.update(MODE=mode,RUN_DIR=str(root/mode),CODE_DIR=str(repo),
                           TRAIN_JSONL="labels",VIDEO_MAP="paths",OUTPUT_DIR="training",
                           LEARNING_RATE="1e-4",SEED="20261104",MAX_NEW_TOKENS="160",
                           CONDA_ENV="fixture",COMMANDS=str(root/"commands.jsonl"))
                result = subprocess.run(["bash",str(repo/"scripts/slurm/run_stage2_overfit_ab.sbatch")],
                                        env=env,capture_output=True,text=True)
                self.assertEqual(result.returncode,0,result.stderr)
                calls = [json.loads(s) for s in (root/"commands.jsonl").read_text().splitlines()]
                train = next(c for c in calls if c[:2]==["-m","stage2.train_overfit_ab"])
                self.assertEqual(train[train.index("--mode")+1],mode)
                self.assertNotIn("--init-adapter",train)
                evals = [c for c in calls if c[1] in ["stage2.evaluate","stage2.reconstruct"]]
                self.assertEqual(len(evals),3)
                for call in evals:
                    i = call.index("--kinds")+1
                    actual = []
                    while i<len(call) and not call[i].startswith("--"):
                        actual.append(call[i]); i+=1
                    self.assertEqual(actual,kinds)


if __name__ == "__main__":
    unittest.main()
