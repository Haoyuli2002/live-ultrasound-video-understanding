import json
import tempfile
import unittest
import os
import subprocess
import sys
import argparse
import shlex
from types import SimpleNamespace
from unittest.mock import patch
from pathlib import Path

from stage1.eval_io import paired, resume_rows
from stage1.prepare_eval import select, CONDITIONS
from stage1.judge_predictions import jobs, validate, aggregate
from stage1.eval_config import add_input_arguments, resolve_input_settings
from stage1.data import frame_timestamps


def samples():
    return [dict(video_id=v, sentence_id=1, condition=c, target='The probe touches the chest.',
                 prediction='The probe is on the chest.', visual_control='normal')
            for v in ('held1', 'held2') for c in sorted(CONDITIONS)]


def training_settings(**changes):
    return {**dict(model_name='Qwen/Qwen3.5-2B', frame_budget=30,
                   active_frame_budget=30, frame_size=224,
                   frame_sampling='recent_sparse', recent_seconds=15.0,
                   max_asr_chars=4000, dtype='bf16'), **changes}


def input_args(*argv):
    parser = argparse.ArgumentParser()
    add_input_arguments(parser)
    parser.add_argument('--adapter-path')
    return parser.parse_args(argv)


class EvalPipelineTests(unittest.TestCase):
    def test_current_sampling_keeps_fifteen_history_and_fifteen_recent_frames(self):
        args = input_args()
        resolve_input_settings(args)
        times = frame_timestamps([0, 300], args.frame_budget,
                                 sampling=args.frame_sampling,
                                 recent_seconds=args.recent_seconds)
        self.assertEqual(len(times), 30)
        self.assertEqual(sum(t < 285 for t in times), 15)
        self.assertEqual(times[-15:], [285.5+i for i in range(15)])

    def test_base_and_adapter_inherit_actual_training_inputs(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            config = root/'training_config.json'
            saved = training_settings(active_frame_budget=24, max_asr_chars=1234, dtype='fp32')
            config.write_text(json.dumps(saved))
            base = input_args('--training-config', str(config))
            adapter = input_args('--adapter-path', str(root/'final'))
            self.assertEqual(resolve_input_settings(base), resolve_input_settings(adapter))
            self.assertEqual((base.frame_budget, base.max_asr_chars, base.dtype),
                             (24, 1234, 'fp32'))
            for flag, value in (('--frame-sampling', 'uniform'), ('--frame-budget', '30')):
                with self.assertRaisesRegex(ValueError, 'differs from training'):
                    resolve_input_settings(input_args('--training-config', str(config), flag, value))

    def test_historical_sampling_is_preserved_and_missing_configuration_fails(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            with self.assertRaisesRegex(FileNotFoundError, 'training-config'):
                resolve_input_settings(input_args('--adapter-path', str(root/'final')))
            saved = training_settings(frame_budget=120, frame_sampling='uniform', recent_seconds=120)
            del saved['active_frame_budget']
            config = root/'training_config.json'
            config.write_text(json.dumps(saved))
            args = input_args('--adapter-path', str(root/'final'))
            resolve_input_settings(args)
            self.assertEqual((args.frame_budget, args.frame_sampling), (120, 'uniform'))
            del saved['recent_seconds']
            config.write_text(json.dumps(saved))
            with self.assertRaisesRegex(ValueError, 'missing input settings'):
                resolve_input_settings(input_args('--training-config', str(config)))

    def test_selection_preserves_triplets_and_excludes_training(self):
        rows = samples()
        split = {'train_video_ids': ['train']}
        selected = select(rows, split, 1)
        self.assertEqual(len(selected), 3)
        self.assertEqual({r['condition'] for r in selected}, CONDITIONS)
        self.assertEqual(selected, select(list(reversed(rows)), split, 1))
        with self.assertRaisesRegex(ValueError, 'overlap'):
            select(rows, {'train_video_ids': ['held1']})
        with self.assertRaisesRegex(ValueError, 'triplet'):
            select(rows[:-1], split)

    def test_pairing_rejects_mismatch_duplicate_and_reference_drift(self):
        rows = samples()
        self.assertEqual(len(paired(rows, rows[::-1])), 6)
        for bad in (rows[:-1], rows + [rows[0]], [{**r, 'target': 'changed'} for r in rows]):
            with self.assertRaises(ValueError):
                paired(rows, bad)

    def test_resume_repairs_only_incomplete_tail_and_rejects_changed_config(self):
        with tempfile.TemporaryDirectory() as d:
            path = Path(d)/'predictions.jsonl'
            self.assertEqual(resume_rows(path, {'model': 'm'}), [])
            path.write_bytes(b'{"id":1}\n{"id":')
            self.assertEqual(resume_rows(path, {'model': 'm'}), [{'id': 1}])
            self.assertEqual(path.read_bytes(), b'{"id":1}\n')
            with self.assertRaisesRegex(ValueError, 'Configuration'):
                resume_rows(path, {'model': 'changed'})
            path.write_text('not-json\n')
            with self.assertRaises(ValueError):
                resume_rows(path, {'model': 'm'})

    def test_judge_schema_and_matched_success_deltas(self):
        valid = dict(semantic=3, medical=4, unsupported=0, rationale='Equivalent anatomy.')
        self.assertEqual(validate(json.dumps(valid)), valid)
        for bad in ({**valid, 'medical': True}, {**valid, 'semantic': 5}, {**valid, 'rationale': ''}):
            with self.assertRaises(ValueError):
                validate(json.dumps(bad))
        work = jobs(samples(), samples())
        self.assertEqual(len(work), 12)
        records = [{**r, 'status': 'ok', 'scores': valid} for r in work]
        records[0]['status'] = 'error'
        report = aggregate(records)
        self.assertEqual(report['paired_delta_lora_minus_base']['overall']['n_pairs'], 5)
        self.assertEqual(report['paired_delta_lora_minus_base']['overall']['medical'], 0)

    def test_submission_smoke_dependency_and_dry_run(self):
        repo = Path(__file__).resolve().parents[1]
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            out = root/'training'
            (out/'final').mkdir(parents=True)
            for name in ('adapter_config.json', 'adapter_model.safetensors'):
                (out/'final'/name).write_text('fixture')
            (out/'validation_samples.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in samples()))
            (out/'video_split.json').write_text(json.dumps({'train_video_ids': ['train']}))
            mapping = root/'map.json'
            mapping.write_text('{}')
            (out/'training_config.json').write_text(json.dumps(
                training_settings(video_path_map=str(mapping))))
            bins = root/'bin'
            bins.mkdir()
            log = root/'calls'
            fake = bins/'sbatch'
            fake.write_text('#!/bin/bash\nprintf "%s\\n" "$*" >> "$CALLS"\necho 12345\n')
            fake.chmod(0o755)
            env = {**os.environ, 'REPO': str(repo), 'DATA': str(root), 'OUTPUT_DIR': str(out),
                   'EVAL_VIDEO_MAP': str(mapping), 'EVAL_DIR': str(root/'eval'),
                   'MODE': 'smoke', 'LIMIT_TARGETS': '1', 'SUBMIT_JUDGE': '1',
                   'EVAL_TIME': '01:00:00', 'JUDGE_TIME': '01:00:00',
                   'CALLS': str(log), 'PATH': str(bins)+os.pathsep+os.environ['PATH']}
            command = ['bash', str(repo/'scripts/slurm/submit_stage1_eval.sh')]
            subprocess.run(command+['--dry-run'], env=env, check=True, capture_output=True)
            self.assertFalse(log.exists())
            subprocess.run(command, env=env, check=True, capture_output=True)
            lines = log.read_text().splitlines()
            self.assertEqual(len(lines), 2)
            self.assertIn('--dependency=afterok:12345', lines[1])
            self.assertTrue(all('--time=01:00:00' in line for line in lines))
            (out/'validation_samples.jsonl').write_text('')
            failure = subprocess.run(command+['--dry-run'], env=env,
                                     capture_output=True, text=True)
            self.assertNotEqual(failure.returncode, 0)
            self.assertIn('independent evaluation split', failure.stderr)
            independent = root/'independent.jsonl'
            independent.write_text(''.join(json.dumps(r)+'\n' for r in samples()))
            independent_env = {**env, 'EVAL_JSONL': str(independent),
                               'EVAL_DIR': str(root/'independent_eval')}
            subprocess.run(command+['--dry-run'], env=independent_env,
                           check=True, capture_output=True)

    def test_both_batch_entries_use_saved_inputs_despite_stale_environment(self):
        repo = Path(__file__).resolve().parents[1]
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            output = root/'training run'
            (output/'final').mkdir(parents=True)
            (output/'final/adapter_model.safetensors').write_text('fixture')
            (output/'training_config.json').write_text(json.dumps(training_settings()))
            (output/'validation_samples.jsonl').write_text(
                ''.join(json.dumps(row)+'\n' for row in samples()))
            (output/'video_split.json').write_text(json.dumps({'train_video_ids':['train']}))
            mapping = root/'map.json'; mapping.write_text('{}')
            conda = root/'conda.sh'; conda.write_text('conda() { :; }\n')
            bins = root/'bin'; bins.mkdir()
            runner = root/'runner.py'
            runner.write_text('''import argparse,json,os,sys
from stage1.eval_config import add_input_arguments,resolve_input_settings
args=sys.argv[1:]
if args[:2]==['-m','stage1.evaluate']:
    parser=argparse.ArgumentParser();add_input_arguments(parser)
    parser.add_argument('--adapter-path')
    options,_=parser.parse_known_args(args[2:])
    settings=resolve_input_settings(options)
    with open(os.environ['CAPTURE_INPUTS'],'a') as f:
        f.write(json.dumps({'adapter':options.adapter_path,'settings':settings})+'\\n')
elif args[:2]!=['-m','stage1.score_predictions']:
    os.execv(sys.executable,[sys.executable,*args])
''')
            python = bins/'python'
            python.write_text('#!/bin/sh\nexec "$REVIEW_PYTHON" "$REVIEW_RUNNER" "$@"\n')
            python.chmod(0o755)
            log = root/'inputs.jsonl'
            env = {**os.environ, 'REPO':str(repo), 'DATA':str(root/'data'),
                   'STAGE1_CACHE_ROOT':str(root/'cache'), 'CONDA_SH':str(conda),
                   'OUTPUT_DIR':str(output), 'EVAL_DIR':str(root/'eval'),
                   'EVAL_JSONL':str(output/'validation_samples.jsonl'),
                   'EVAL_VIDEO_MAP':str(mapping), 'CAPTURE_INPUTS':str(log),
                   'REVIEW_PYTHON':sys.executable, 'REVIEW_RUNNER':str(runner),
                   'PYTHONPATH':str(repo), 'PATH':str(bins)+os.pathsep+os.environ['PATH'],
                   'FRAME_BUDGET':'120', 'FRAME_SAMPLING':'uniform',
                   'RECENT_SECONDS':'120', 'MODEL_NAME':'stale-model',
                   'VISUAL_CONTROLS':'normal'}
            for script in ('run_stage1_eval_suite.sbatch', 'run_stage1_qwen35_eval.sbatch'):
                with self.subTest(script=script):
                    log.write_text('')
                    result = subprocess.run(['bash',str(repo/'scripts/slurm'/script)],
                                            env=env,capture_output=True,text=True)
                    self.assertEqual(result.returncode,0,result.stderr)
                    records=[json.loads(row) for row in log.read_text().splitlines()]
                    self.assertEqual(len(records),2)
                    self.assertIsNone(records[0]['adapter'])
                    self.assertEqual(records[1]['adapter'],str(output/'final'))
                    self.assertEqual(records[0]['settings'],records[1]['settings'])
                    self.assertEqual(records[0]['settings']['frame_sampling'],'recent_sparse')
                    self.assertEqual(records[0]['settings']['frame_budget'],30)
                    self.assertEqual(records[0]['settings']['recent_seconds'],15)

    def test_judge_requests_are_blinded_and_resume_without_api_calls(self):
        from stage1.judge_predictions import main
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            pred = root/'pred.jsonl'
            pred.write_text(''.join(json.dumps(r)+'\n' for r in samples()))
            output = root/'judge.jsonl'
            calls = []
            def create(**kwargs):
                calls.append(kwargs)
                payload = json.loads(kwargs['messages'][1]['content'])
                self.assertEqual(set(payload), {'reference', 'candidate'})
                result = dict(semantic=4, medical=4, unsupported=0, rationale='Equivalent.')
                return SimpleNamespace(choices=[SimpleNamespace(finish_reason='stop',
                    message=SimpleNamespace(content=json.dumps(result)))])
            client = SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(create=create)))
            argv = ['judge', '--predictions', str(pred), '--baseline', str(pred), '--output', str(output)]
            with patch('sys.argv', argv), patch('openai.OpenAI', return_value=client):
                main()
                main()
            self.assertEqual(len(calls), 12)
            self.assertEqual(len(output.read_text().splitlines()), 12)

    def test_batch_preserves_recent_sampling_for_both_candidates(self):
        repo = Path(__file__).resolve().parents[1]
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            # Run the actual batch script against a lightweight storage fixture.
            (root/'scripts/slurm').mkdir(parents=True)
            (root/'scripts/slurm/stage1_storage.sh').write_text(':\n')
            (root/'training/final').mkdir(parents=True)
            (root/'training/final/adapter_model.safetensors').write_text('fixture')
            conda = root/'conda.sh'
            conda.write_text('conda() { :; }\n')
            bins = root/'bin'
            bins.mkdir()
            python = bins/'python'
            runner = root/'capture.py'
            runner.write_text("import json,os,sys\nwith open(os.environ['COMMANDS'],'a') as stream: stream.write(json.dumps(sys.argv[1:])+'\\n')\n")
            python.write_text('#!/bin/sh\nexec '+shlex.quote(sys.executable)+' '+shlex.quote(str(runner))+' "$@"\n')
            python.chmod(0o755)
            env = dict(os.environ,REPO=str(root),DATA=str(root),OUTPUT_DIR=str(root/'training'),
                       EVAL_DIR=str(root/'eval'),CONDA_SH=str(conda),FRAME_SAMPLING='recent_sparse',
                       RECENT_SECONDS='15',VISUAL_CONTROLS='normal',EVAL_VIDEO_MAP=str(root/'map.json'),COMMANDS=str(root/'calls.jsonl'),
                       PATH=str(bins)+os.pathsep+os.environ['PATH'])
            result = subprocess.run(['bash',str(repo/'scripts/slurm/run_stage1_eval_suite.sbatch')],
                                    env=env,capture_output=True,text=True)
            self.assertEqual(result.returncode,0,result.stderr)
            calls = [json.loads(s) for s in (root/'calls.jsonl').read_text().splitlines()]
            predictions = [c for c in calls if c[:2]==['-m','stage1.evaluate']]
            self.assertEqual(len(predictions),2)
            for call in predictions:
                self.assertEqual(call[call.index('--training-config')+1], str(root/'training/training_config.json'))
                self.assertNotIn('--frame-sampling', call)
                self.assertNotIn('--recent-seconds', call)
            self.assertNotIn('--adapter-path',predictions[0])
            self.assertIn('--adapter-path',predictions[1])


if __name__ == '__main__':
    unittest.main()
