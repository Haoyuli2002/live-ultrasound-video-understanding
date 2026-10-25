"""CPU regression checks for the full-video Qwen annotation job (no API calls)."""
import contextlib
import io
import itertools
import json
import os
import subprocess
import tempfile
import unittest
from collections import Counter
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

from stage2.annotate import ChatTeacher, annotate_video, resume_start, summarize_window
from stage2.summary_schema import SUMMARY_PLANS, normalize_summary, parse_summary
from stage2.build_data import build_files
from stage2.data import load_blocks


class Teacher:
    def summary(self, prompt, clip):
        if 'cumulative video prefix' in prompt:
            return 'Anatomy: chest; Change: probe repositioned'
        return 'View: chest wall; Anatomy: ribs; Dynamic: probe movement'


class AnnotationJobTests(unittest.TestCase):
    def events(self):
        with patch('stage2.annotate.make_silent_clip'), contextlib.redirect_stdout(io.StringIO()):
            events = [row for batch in annotate_video(Path('video.mp4'), Teacher(),
                       video_id='8V649L5Q368', duration_sec=1136) for row in batch]
        for row in events:
            row.update(teacher='Qwen/Qwen3.5-27B', annotation_config={'test': True})
        return events

    def test_entire_video_counts_chronology_and_tail(self):
        events = self.events()
        self.assertEqual(Counter(row['label_type'] for row in events),
                         {'local_10s': 113, 'block_60s': 18, 'global_prefix': 18})
        self.assertEqual(events[-1]['window'], [1120, 1130])
        self.assertEqual(events[-1]['block_window'], [1080, 1136])
        self.assertEqual([r['window'][1] for r in events if r['label_type'] == 'global_prefix'],
                         list(range(60, 1081, 60)))
        with tempfile.TemporaryDirectory() as directory:
            source, output = Path(directory)/'events.jsonl', Path(directory)/'blocks.jsonl'
            source.write_text(''.join(json.dumps(row)+'\n' for row in events))
            self.assertEqual(build_files([source], output), 19)
            blocks = load_blocks(output)['8V649L5Q368']
            self.assertEqual(len(blocks[-1]['local_sub_summaries']), 5)
            self.assertEqual(blocks[-1]['global_summary_target'], '')

    def test_resume_complete_tail_empty_and_incompatible_config(self):
        kwargs = dict(video_id='8V649L5Q368', duration_sec=1136,
                      model='Qwen/Qwen3.5-27B', config={'test': True})
        events = self.events()
        self.assertEqual(resume_start([], **kwargs), 0)
        self.assertEqual(resume_start(events[:8], **kwargs), 60)
        self.assertEqual(resume_start(events[:16], **kwargs), 120)
        self.assertEqual(resume_start(events, **kwargs), 1136)
        with self.assertRaises(ValueError):
            resume_start(events[1:], **kwargs)
        with self.assertRaises(ValueError):
            resume_start(events, **{**kwargs, 'config': {'test': False}})

    def test_qwen_request_is_bounded_and_non_thinking(self):
        calls = []
        def create(**kwargs):
            calls.append(kwargs)
            return SimpleNamespace(choices=[SimpleNamespace(finish_reason='stop',
                       message=SimpleNamespace(content='View: chest'))])
        teacher = ChatTeacher.__new__(ChatTeacher)
        teacher.client = SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(create=create)))
        teacher.model = 'Qwen/Qwen3.5-27B'
        teacher.transport = 'video_url'
        teacher.max_tokens = 1024
        teacher.video_fps = 1.0
        teacher.video_max_frames = 120
        teacher.disable_thinking = True
        teacher.summary('describe', Path('silent.mp4'))
        extra = calls[0]['extra_body']
        self.assertEqual(extra['chat_template_kwargs'], {'enable_thinking': False})
        self.assertEqual(extra['mm_processor_kwargs'],
                         {'fps': 1.0, 'do_sample_frames': True, 'max_frames': 120})
        self.assertEqual(calls[0]['messages'][1]['content'][0]['type'], 'video_url')

    def test_all_facet_permutations_preserve_content_and_strict_storage(self):
        for kind, plan in SUMMARY_PLANS.items():
            expected = {label: f"evidence {i}" for i, label in enumerate(plan['facets'])}
            for order in itertools.permutations(expected):
                raw = '; '.join(f"{label}: {expected[label]}" for label in order)
                target, facets = normalize_summary(raw, kind)
                self.assertEqual(list(facets), list(expected))
                self.assertEqual(facets, expected)
                self.assertEqual(parse_summary(target, kind), expected)
        raw = 'Anatomy: ribs\nProbe: stationary'
        self.assertEqual(normalize_summary(raw, 'local')[0], raw)
        with self.assertRaisesRegex(ValueError, 'out of order'):
            parse_summary('Dynamic: sliding; Probe: stationary', 'local')

    def test_invalid_facets_are_not_silently_repaired(self):
        for raw in ('Probe: a; Probe: b', 'Change: stable', 'Unknown: evidence',
                    'Probe: ', 'plain text', '', 'Finding: ' + 'word ' * 50):
            with self.subTest(raw=raw), self.assertRaises(ValueError):
                normalize_summary(raw, 'local')

    def test_out_of_order_probe_needs_no_teacher_retry(self):
        teacher = SimpleNamespace(summary=Mock(return_value=
            'View: chest; Dynamic: sliding; Probe: stationary'))
        with patch('stage2.annotate.make_silent_clip'), contextlib.redirect_stdout(io.StringIO()) as log:
            target, facets = summarize_window(Path('source.mp4'), teacher, kind='local',
                start=160, end=170, clip_path=Path('silent.mp4'))
        self.assertEqual(target, 'View: chest; Probe: stationary; Dynamic: sliding')
        self.assertEqual(parse_summary(target, 'local'), facets)
        teacher.summary.assert_called_once()
        self.assertIn('[normalized]', log.getvalue())

    def test_duplicate_still_retries_and_preserves_failed_raw_output(self):
        teacher = SimpleNamespace(summary=Mock(side_effect=[
            'Probe: a; Probe: b', 'Probe: stationary']))
        with patch('stage2.annotate.make_silent_clip'), contextlib.redirect_stdout(io.StringIO()) as log:
            target, _ = summarize_window(Path('source.mp4'), teacher, kind='local',
                start=160, end=170, clip_path=Path('silent.mp4'))
        self.assertEqual(target, 'Probe: stationary')
        self.assertEqual(teacher.summary.call_count, 2)
        self.assertIn('Probe: a; Probe: b', log.getvalue())

    def test_submission_dry_run_and_fake_sbatch(self):
        repo = Path(__file__).resolve().parents[1]
        script = repo/'scripts/slurm/submit_stage2_8V649L5Q368.sh'
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            video = root/'8V649L5Q368.mp4'
            video.write_bytes(b'fixture, not decoded during submission')
            binaries = root/'bin'
            binaries.mkdir()
            sbatch = binaries/'sbatch'
            sbatch.write_text('#!/bin/bash\nset -eu\n'
                              '[[ "$MODEL" == Qwen/Qwen3.5-27B ]]\n'
                              '[[ "$VIDEO_IDS" == 8V649L5Q368 ]]\n'
                              '[[ "$CLIP_MAX_SIDE" == 512 ]]\n'
                              '[[ -f "$VIDEO_MAP" ]]\n'
                              'echo 12345\n')
            sbatch.chmod(0o755)
            env = {**os.environ, 'REPO': str(repo), 'DATA': str(root),
                   'VIDEO_PATH': str(video), 'RUN_DIR': str(root/'run'),
                   'PATH': str(binaries)+os.pathsep+os.environ['PATH']}
            result = subprocess.run(['bash', str(script), '--dry-run'], env=env,
                                    capture_output=True, text=True)
            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertFalse((root/'run/job_id.txt').exists())
            mapping = json.loads((root/'run/video_paths.json').read_text())
            self.assertEqual(mapping, {'8V649L5Q368': str(video.resolve())})
            result = subprocess.run(['bash', str(script)], env=env, capture_output=True, text=True)
            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertEqual((root/'run/job_id.txt').read_text().strip(), '12345')


if __name__ == '__main__':
    unittest.main()
