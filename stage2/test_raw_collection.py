"""Save-first collection tests: real files/merge, no GPU or model API calls."""
import contextlib
import io
import json
import os
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

from .annotate import (ChatTeacher, annotate_video, annotation_config, main,
                       resume_start, save_events_atomic, summarize_window)
from .build_data import build_files
from .data import load_blocks
from .full_run import MODEL, prepare, submit, worker, write_json
from .raw_collection import WindowStore, merge_collection, quality_report


class RawCollectionTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(); self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)

    def test_any_text_saved_exactly_once_before_quality_check_and_cache_reuses_it(self):
        examples = ['', '  View: chest; Anatomy: ribs  \n', '```json\n{"summary":"x"}\n```',
                    '<think>reasoning</think>\n\nA view is visible.', 'word '*150]
        for index, raw in enumerate(examples):
            store = WindowStore(self.root/str(index), {'source_sha256':'unchanged'})
            teacher = SimpleNamespace(summary=Mock(return_value=raw))
            with patch('stage2.annotate.make_silent_clip') as clip, \
                    patch('stage2.annotate.validate_prose', side_effect=AssertionError('No early validation')), \
                    contextlib.redirect_stdout(io.StringIO()):
                kwargs = dict(kind='local', start=0, end=10, clip_path=self.root/'clip.mp4', raw_store=store)
                self.assertEqual(summarize_window(Path('v.mp4'), teacher, **kwargs)[0], raw)
                self.assertEqual(summarize_window(Path('v.mp4'), teacher, **kwargs)[0], raw)
            teacher.summary.assert_called_once(); clip.assert_called_once()
            self.assertEqual(json.loads(store.path('local',0,10).read_text())['raw_text'], raw)
            with self.assertRaisesRegex(ValueError, 'differs'):
                store.get('local',0,10,'changed request')

    def test_partial_minute_cache_survives_later_request_failure(self):
        store = WindowStore(self.root/'raw', {})
        teacher = SimpleNamespace(summary=Mock(side_effect=['View: chest', RuntimeError('endpoint lost')]))
        with patch('stage2.annotate.make_silent_clip'), contextlib.redirect_stdout(io.StringIO()):
            with self.assertRaisesRegex(RuntimeError,'endpoint lost'):
                list(annotate_video(Path('v.mp4'),teacher,video_id='v',duration_sec=60,raw_store=store))
            resumed = SimpleNamespace(summary=Mock(return_value=''))
            batches = list(annotate_video(Path('v.mp4'),resumed,video_id='v',duration_sec=60,raw_store=store))
        self.assertEqual(len(batches[0]),8)
        self.assertEqual(batches[0][0]['target'],'View: chest')
        self.assertEqual(resumed.summary.call_count,7)

    def test_truncated_response_is_kept_and_reported(self):
        teacher = ChatTeacher.__new__(ChatTeacher)
        teacher.keep_all = True; teacher.model = 'model'; teacher.transport = 'video_url'
        teacher.max_tokens = 4; teacher.disable_thinking = True
        response = SimpleNamespace(id='response',choices=[SimpleNamespace(finish_reason='length',
                                   message=SimpleNamespace(content=' View: ',refusal=None))])
        teacher.client = SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(create=Mock(return_value=response))))
        store = WindowStore(self.root/'raw'/'v', {})
        with patch('stage2.annotate.make_silent_clip'), contextlib.redirect_stdout(io.StringIO()):
            raw,_ = summarize_window(Path('v.mp4'),teacher,kind='local',start=0,end=10,
                                      clip_path=self.root/'clip.mp4',raw_store=store)
        report = quality_report([{'video_id':'v','label_type':'local_10s','window':[0,10],'target':raw}],self.root/'raw')
        self.assertEqual(raw,' View: ')
        self.assertEqual(report['issue_counts']['truncated'],1)
        self.assertEqual(report['filtered_summaries'],0)

    def plan_and_rows(self):
        source = self.root/'v.mp4'; source.write_bytes(b'source')
        train, evaluation, run = self.root/'train.json', self.root/'eval.json', self.root/'run'
        write_json(train, {'v':str(source)}); write_json(evaluation, {'excluded':str(self.root/'e.mp4')})
        with patch('stage2.full_run.VideoReader',return_value=SimpleNamespace(count=120,fps=1,close=lambda:None)), \
                contextlib.redirect_stdout(io.StringIO()):
            prepare(train,evaluation,run,video_root=self.root,shards=1)
        with patch('stage2.annotate.make_silent_clip'), contextlib.redirect_stdout(io.StringIO()):
            rows = [r for batch in annotate_video(source,SimpleNamespace(summary=lambda *a:''),
                     video_id='v',duration_sec=120,raw_store=WindowStore(run/'raw'/'v',{})) for r in batch]
        config = annotation_config(source)
        for row in rows: row.update(teacher=MODEL,annotation_config=config)
        return run,source,rows

    def test_candidate_keeps_empty_targets_and_strict_training_loader_still_rejects(self):
        run,source,rows = self.plan_and_rows()
        rows[0]['target'] = '  View: chest\n\nAnatomy: ribs  '
        save_events_atomic(run/'labels.jsonl',rows)
        candidate = run/'candidate.unreviewed.jsonl'
        self.assertEqual(build_files([run/'labels.jsonl'],candidate,validate_text=False),2)
        loaded = load_blocks(candidate,validate_text=False)['v']
        self.assertEqual(loaded[0]['local_sub_summaries'][0]['local_summary_target'],rows[0]['target'])
        self.assertEqual(loaded[0]['local_sub_summaries'][1]['local_summary_target'],'')
        with self.assertRaises(ValueError): load_blocks(candidate)
        with self.assertRaises(ValueError): build_files([run/'labels.jsonl'],run/'strict.jsonl')

    def test_merge_reports_bad_outputs_without_deleting_or_publishing_training_data(self):
        run,source,rows = self.plan_and_rows()
        (run/'collected').mkdir()
        save_events_atomic(run/'collected/v.jsonl',rows)
        before = (run/'collected/v.jsonl').read_bytes()
        report = merge_collection(run)
        self.assertEqual(report['status'],'pending_review')
        self.assertEqual(report['total_summaries'],16)
        self.assertEqual(report['issue_counts']['empty'],16)
        self.assertEqual(report['rows'],2)
        self.assertFalse((run/'blocks.jsonl').exists())
        self.assertEqual(before,(run/'collected/v.jsonl').read_bytes())
        with patch.dict(os.environ, {'STAGE2_KEEP_ALL':'1'}), \
                patch('stage2.full_run.subprocess.run') as process, contextlib.redirect_stdout(io.StringIO()):
            self.assertEqual(worker(run,0,self.root),0)
            process.assert_not_called()

    def test_retry_only_inactive_shard_and_merge_waits_for_old_jobs(self):
        run,source,rows = self.plan_and_rows()
        plan=json.loads((run/'manifest.json').read_text()); plan['shards']=[['v'],[]]
        write_json(run/'manifest.json',plan)
        write_json(run/'submissions.json',{'attempts':[{'array_job':'100','jobs':['100','101']}]})
        with patch('stage2.full_run.subprocess.check_output',side_effect=['100_1 100 RUNNING\n101 101 PENDING\n','200\n','201\n']) as call:
            result=submit(run,self.root,retry_shards=[0],keep_all=True)
        self.assertEqual(result['jobs'],['200','201'])
        self.assertIn('--array=0%4',call.call_args_list[1].args[0])
        self.assertIn('--dependency=afterany:200:100:101',call.call_args_list[2].args[0])
        self.assertEqual(call.call_args_list[1].kwargs['env']['STAGE2_KEEP_ALL'],'1')
        with patch('stage2.full_run.subprocess.check_output',return_value='200_0 200 RUNNING\n'):
            with self.assertRaisesRegex(ValueError,'still active'):
                submit(run,self.root,retry_shards=[0],keep_all=True)

    def test_incomplete_run_report_includes_raw_responses_outside_saved_minutes(self):
        run,source,rows = self.plan_and_rows()
        (run/'collected').mkdir()
        save_events_atomic(run/'collected/v.jsonl',rows[:8])
        with self.assertRaisesRegex(ValueError,'incomplete'): merge_collection(run)
        report=json.loads((run/'quality_report.json').read_text())
        self.assertEqual(report['total_summaries'],16)
        self.assertEqual(report['responses_not_yet_in_minute_batches'],8)
        self.assertEqual(report['issue_counts']['empty'],16)
        self.assertFalse((run/'blocks.unreviewed.jsonl').exists())

    def test_seed_complete_minutes_are_reused_unchanged(self):
        run,source,rows = self.plan_and_rows()
        seed=run/'seed.jsonl'
        # Valid old summaries must remain byte-for-byte unchanged in the seed.
        for row in rows[:8]: row['target']='The cardiac chambers are visible.'
        save_events_atomic(seed,rows[:8]); before=seed.read_bytes()
        output=run/'collected/v.jsonl'
        teacher=SimpleNamespace(summary=Mock(return_value='View: chest'))
        args=['annotate','--video',str(source),'--video-id=v','--output',str(output),
              '--model',MODEL,'--transport','video_url','--video-fps','2','--video-max-frames','120',
              '--clip-max-side','1024','--clip-fps','2','--disable-thinking','--max-tokens','1024',
              '--raw-dir',str(run/'raw_cli'/'v'),'--seed-output',str(seed),'--resume']
        with patch('sys.argv',args),patch.dict(os.environ,{'OPENAI_API_KEY':'test'}), \
                patch('stage2.annotate.VideoReader',return_value=SimpleNamespace(count=120,fps=1,close=lambda:None)), \
                patch('stage2.annotate.ChatTeacher',return_value=teacher),patch('stage2.annotate.make_silent_clip'), \
                contextlib.redirect_stdout(io.StringIO()):
            main()
            self.assertEqual(teacher.summary.call_count,8)
            main()
            self.assertEqual(teacher.summary.call_count,8)
        self.assertEqual(seed.read_bytes(),before)
        saved=[json.loads(x) for x in output.read_text().splitlines()]
        self.assertEqual(saved[:8],rows[:8])
        self.assertEqual(resume_start(saved,video_id='v',duration_sec=120,model=MODEL,
                         config=rows[0]['annotation_config'],validate_text=False),120)


if __name__ == '__main__': unittest.main()
