import json
import tempfile
import unittest
import os
import subprocess
from types import SimpleNamespace
from unittest.mock import patch
from pathlib import Path

from stage1.eval_io import paired, resume_rows
from stage1.prepare_eval import select, CONDITIONS
from stage1.judge_predictions import jobs, validate, aggregate


def samples():
    return [dict(video_id=v, sentence_id=1, condition=c, target='The probe touches the chest.',
                 prediction='The probe is on the chest.', visual_control='normal')
            for v in ('held1', 'held2') for c in sorted(CONDITIONS)]


class EvalPipelineTests(unittest.TestCase):
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


if __name__ == '__main__':
    unittest.main()
