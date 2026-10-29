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

from stage2.annotate import ChatTeacher, annotate_video, main, resume_start, summarize_window
from stage2.summary_schema import (SUMMARY_PLANS, normalize_summary, parse_summary,
                                   PROSE_SCHEMA_VERSION, validate_prose)
from stage2.build_data import build_files
from stage2.data import load_blocks


class Teacher:
    def summary(self, prompt, clip):
        if 'cumulative video prefix' in prompt:
            return 'The examination progresses from patient positioning to a visible probe demonstration.'
        return 'The patient lies supine while the clinician demonstrates probe handling beside the ultrasound display.'


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
            self.assertEqual(blocks[0]['schema_version'], PROSE_SCHEMA_VERSION)
            self.assertNotIn('facets', blocks[0]['local_sub_summaries'][0])
            self.assertNotIn('block_summary_facets', blocks[0])

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
        teacher.request_model = 'stage2_own_job'
        teacher.transport = 'video_url'
        teacher.max_tokens = 1024
        teacher.video_fps = 1.0
        teacher.video_max_frames = 120
        teacher.disable_thinking = True
        teacher.summary('describe', Path('silent.mp4'))
        extra = calls[0]['extra_body']
        self.assertEqual(extra['chat_template_kwargs'], {'enable_thinking': False})
        self.assertEqual(extra['mm_processor_kwargs'],
                         {'do_sample_frames': False})
        self.assertEqual(calls[0]['messages'][1]['content'][0]['type'], 'video_url')
        self.assertEqual(calls[0]['model'], 'stage2_own_job')
        self.assertEqual(teacher.model, 'Qwen/Qwen3.5-27B')

    def test_preflight_without_teacher_rejects_changed_source_and_preserves_labels(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source, output = root/'source.mp4', root/'labels.jsonl'
            source.write_bytes(b'original source')
            args = ['annotate', '--video', str(source), '--video-id', '8V649L5Q368',
                    '--output', str(output), '--model', 'Qwen/Qwen3.5-27B',
                    '--resume', '--preflight-only']
            def preflight():
                reader = SimpleNamespace(count=120, fps=2, close=Mock())
                with patch('sys.argv', args), patch('stage2.annotate.VideoReader', return_value=reader), \
                        patch('stage2.annotate.ChatTeacher') as teacher, \
                        contextlib.redirect_stdout(io.StringIO()) as log:
                    main()
                    teacher.assert_not_called()
                reader.close.assert_called_once()
                return json.loads(log.getvalue())
            config = preflight()['annotation_config']
            self.assertFalse(output.exists())
            events = self.events()[:8]
            for event in events:
                event['annotation_config'] = config
            saved = ''.join(json.dumps(event)+'\n' for event in events)
            output.write_text(saved)
            self.assertEqual(preflight()['resume_from'], 60)
            source.write_bytes(b'replaced source, same filename')
            with self.assertRaisesRegex(ValueError, 'annotation config'):
                preflight()
            self.assertEqual(output.read_text(), saved)

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

    def test_prose_rejects_labelled_teacher_and_retries(self):
        expected = 'The clinician holds the probe while demonstrating patient positioning.'
        teacher = SimpleNamespace(summary=Mock(side_effect=[
            'View: chest; Anatomy: ribs', expected]))
        with patch('stage2.annotate.make_silent_clip'), contextlib.redirect_stdout(io.StringIO()) as log:
            target, facets = summarize_window(Path('source.mp4'), teacher, kind='local',
                start=160, end=170, clip_path=Path('silent.mp4'))
        self.assertEqual(target, expected)
        self.assertEqual(facets, {})
        self.assertEqual(teacher.summary.call_count, 2)
        self.assertIn('View: chest; Anatomy: ribs', log.getvalue())
        prompt = teacher.summary.call_args.args[0]
        self.assertIn('Video interval: [160, 170) seconds', prompt)
        self.assertNotIn('in this order', prompt)

    def test_legacy_parser_rejects_swallowed_sentence_labels(self):
        with self.assertRaisesRegex(ValueError, 'embedded facet'):
            parse_summary('View: Grayscale image. Anatomy: Soft tissue. Finding: Bright lines.', 'block')

    def test_prose_shape_and_mixed_dataset_rejected(self):
        for text in ('View: chest wall. Anatomy: visible ribs.', '- The probe is shown.',
                     '{"summary": "The probe is shown."}', ''):
            with self.subTest(text=text), self.assertRaises(ValueError):
                validate_prose(text, 'local')
        from stage2.build_data import build_rows
        events = self.events()
        events[0]['schema_version'] = 'stage2_facets_v1'
        with self.assertRaisesRegex(ValueError, 'mix'):
            build_rows(events)

    def test_reported_length_failure_survives_as_complete_prose(self):
        original = ("In a clinical setting, a physician stands beside a patient lying supine on an "
                    "examination table, with an ultrasound machine positioned nearby. The physician "
                    "appears to be preparing for or explaining an ultrasound procedure, as indicated "
                    "by the presence of the ultrasound equipment and the patient's positioning. "
                    "No specific ultrasound findings or measurements are visible in this segment.")
        teacher = SimpleNamespace(summary=Mock(return_value=original))
        with patch('stage2.annotate.make_silent_clip'), contextlib.redirect_stdout(io.StringIO()) as log:
            target, _ = summarize_window(Path('video.mp4'), teacher, kind='local',
                start=0, end=10, clip_path=Path('clip.mp4'))
        self.assertEqual(target, original)
        teacher.summary.assert_called_once()
        self.assertIn('[length-warning]', log.getvalue())
        from stage2.build_data import build_rows
        events = self.events()[:8]
        events[0]['target'] = target
        self.assertEqual(build_rows(events)[0]['local_sub_summaries'][0]['local_summary_target'], target)

    def test_overlength_all_kinds_survive_build_and_load(self):
        long = ' '.join(['observation'] * 200)
        events = self.events()[:8]
        for kind in SUMMARY_PLANS:
            teacher = SimpleNamespace(summary=Mock(return_value=long))
            with patch('stage2.annotate.make_silent_clip'), contextlib.redirect_stdout(io.StringIO()):
                target, _ = summarize_window(Path('v.mp4'), teacher, kind=kind,
                    start=0, end=10, clip_path=Path('clip.mp4'))
            self.assertEqual(target, long)
            teacher.summary.assert_called_once()
            self.assertEqual(validate_prose(target, kind), long)
        for event in events:
            event['target'] = long
        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory)/'teacher.jsonl'
            output = Path(directory)/'blocks.jsonl'
            source.write_text(''.join(json.dumps(e)+'\n' for e in events))
            build_files([source], output)
            self.assertTrue(load_blocks(output))

    def test_format_errors_still_retry(self):
        short = 'The clinician positions the probe on the chest.'
        teacher = SimpleNamespace(summary=Mock(side_effect=['View: chest', short]))
        with patch('stage2.annotate.make_silent_clip'), contextlib.redirect_stdout(io.StringIO()):
            self.assertEqual(summarize_window(Path('v.mp4'), teacher, kind='local',
                start=0, end=10, clip_path=Path('clip.mp4'))[0], short)
        self.assertIn('View: chest', teacher.summary.call_args_list[1].args[0])
        teacher = SimpleNamespace(summary=Mock(return_value='View: chest'))
        with patch('stage2.annotate.make_silent_clip'), contextlib.redirect_stdout(io.StringIO()), \
                self.assertRaises(RuntimeError):
            summarize_window(Path('v.mp4'), teacher, kind='local',
                start=0, end=10, clip_path=Path('clip.mp4'))

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
                              '[[ "$CLIP_MAX_SIDE" == 1024 ]]\n'
                              '[[ -f "$VIDEO_MAP" ]]\n'
                              '[[ "$*" == *"--time=01:00:00"* ]]\n'
                              'echo 12345\n')
            sbatch.chmod(0o755)
            env = {**os.environ, 'REPO': str(repo), 'DATA': str(root),
                   'VIDEO_PATH': str(video), 'RUN_DIR': str(root/'run'), 'WALLTIME': '01:00:00',
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
            old_labels = root/'run/teacher/8V649L5Q368.jsonl'
            old_labels.parent.mkdir()
            old_labels.write_text(json.dumps({'schema_version': 'stage2_facets_v1'})+'\n')
            before = (root/'run/submit_env.sh').read_text()
            result = subprocess.run(['bash', str(script), '--dry-run'], env=env,
                                    capture_output=True, text=True)
            self.assertNotEqual(result.returncode, 0)
            self.assertIn('Old structured labels', result.stderr)
            self.assertEqual((root/'run/submit_env.sh').read_text(), before)
            old_labels.write_text(json.dumps({'schema_version': 'stage2_prose_v2',
                'annotation_config': {'prompt_sha256': 'obsolete'}})+'\n')
            result = subprocess.run(['bash', str(script), '--dry-run'], env=env,
                                    capture_output=True, text=True)
            self.assertNotEqual(result.returncode, 0)
            self.assertIn('Prompt or video sampling differs', result.stderr)
            self.assertEqual((root/'run/submit_env.sh').read_text(), before)


if __name__ == '__main__':
    unittest.main()
