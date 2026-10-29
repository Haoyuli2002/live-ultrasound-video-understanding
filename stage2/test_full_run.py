"""Full-run safety checks without GPUs, ASR or teacher API requests."""
import contextlib
import io
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

from stage2.annotate import (annotation_config, annotate_video, main, resume_start,
                             save_events_atomic)
from stage2.full_run import (MODEL, expected_counts, merge, prepare, submit, worker,
                             write_json)


class FullRunTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.run = self.root/'run'
        self.durations = {'x':60,'y':122,'z':131}
        self.mapping = {}
        for vid in self.durations:
            path = self.root/f'{vid}.mp4'
            path.write_bytes(f'source {vid}'.encode())
            self.mapping[vid] = str(path)
        self.train, self.eval = self.root/'train.json', self.root/'eval.json'
        write_json(self.train, self.mapping)
        write_json(self.eval, {'evaluation':str(self.root/'evaluation.mp4')})
        self.reader_patch = patch('stage2.full_run.VideoReader',side_effect=lambda path,size:
            SimpleNamespace(count=self.durations[Path(path).stem]*2,fps=2,close=lambda:None))
        self.reader_patch.start()
        self.addCleanup(self.reader_patch.stop)

    def plan(self, **kwargs):
        with contextlib.redirect_stdout(io.StringIO()):
            return prepare(self.train,self.eval,self.run,video_root=self.root,shards=2,**kwargs)

    def labels(self, vid, directory, duration=None):
        duration = self.durations[vid] if duration is None else duration
        teacher = SimpleNamespace(summary=lambda *args:'No ultrasound-relevant visual information in this interval.')
        with patch('stage2.annotate.make_silent_clip'), contextlib.redirect_stdout(io.StringIO()):
            rows = [row for batch in annotate_video(Path(self.mapping[vid]),teacher,
                    video_id=vid,duration_sec=duration) for row in batch]
        config = annotation_config(Path(self.mapping[vid]))
        for row in rows:
            row.update(teacher=MODEL,annotation_config=config)
        directory.mkdir(parents=True,exist_ok=True)
        save_events_atomic(directory/f'{vid}.jsonl', rows)
        return rows

    def test_balanced_partition_is_exhaustive_disjoint_and_immutable(self):
        plan = self.plan()
        ids = [vid for shard in plan['shards'] for vid in shard]
        self.assertCountEqual(ids,self.mapping)
        self.assertEqual(len(ids),len(set(ids)))
        self.assertEqual(plan['excluded_eval_ids'],['evaluation'])
        self.assertEqual(plan['expected_events'],8+16+17)
        self.assertEqual(plan['expected_rows'],1+2+3)
        with self.assertRaises(FileExistsError):self.plan()

    def test_rejects_id_overlap_and_path_alias_to_evaluation(self):
        write_json(self.eval,{'x':self.mapping['x']})
        with self.assertRaisesRegex(ValueError,'overlap'):self.plan()
        write_json(self.eval,{'different_id':self.mapping['x']})
        with self.assertRaisesRegex(ValueError,'overlap'):self.plan()
        self.assertFalse((self.run/'manifest.json').exists())

    def test_completed_seed_reused_and_two_second_tail_is_complete(self):
        seed = self.root/'seed'
        self.labels('y',seed)
        plan = self.plan(seed_teacher_dir=seed)
        self.assertEqual(plan['seeded_complete_videos'],1)
        self.assertNotIn('y',[v for shard in plan['shards'] for v in shard])
        self.assertEqual((self.run/'teacher/y.jsonl').read_bytes(),(seed/'y.jsonl').read_bytes())
        rows=json.loads((seed/'y.jsonl').read_text().splitlines()[0])
        self.assertEqual(resume_start([json.loads(v) for v in (seed/'y.jsonl').read_text().splitlines()],
            video_id='y',duration_sec=122,model=MODEL,config=rows['annotation_config']),122)
        self.assertEqual(expected_counts(122)['rows'],2)

    def test_merge_refuses_partial_then_publishes_all_expected_rows(self):
        self.plan()
        self.labels('x',self.run/'teacher')
        with self.assertRaisesRegex(ValueError,'incomplete'):merge(self.run)
        self.assertFalse((self.run/'blocks.jsonl').exists())
        report=json.loads((self.run/'report.json').read_text())
        self.assertEqual(report['complete_videos'],1)
        for vid in ('y','z'):self.labels(vid,self.run/'teacher')
        report=merge(self.run)
        self.assertEqual(report['status'],'ready')
        self.assertEqual(report['events'],41)
        self.assertEqual(report['rows'],6)
        self.assertEqual(len((self.run/'blocks.jsonl').read_text().splitlines()),6)

    def test_mismatched_seed_and_mutated_source_rejected(self):
        seed=self.root/'seed'
        self.labels('x',seed)
        data=(seed/'x.jsonl').read_text().replace('warn_only','other_policy')
        (seed/'x.jsonl').write_text(data)
        with self.assertRaisesRegex(ValueError,'annotation config'):self.plan(seed_teacher_dir=seed)
        self.plan()
        Path(self.mapping['x']).write_bytes(b'replaced video')
        with self.assertRaisesRegex(ValueError,'source or recipe changed'):
            plan=json.loads((self.run/'manifest.json').read_text())
            index=next(i for i,ids in enumerate(plan['shards']) if 'x' in ids)
            worker(self.run,index,self.root)

    def test_worker_uses_frozen_ids_and_skips_model_for_complete_shard(self):
        plan=self.plan()
        with patch('stage2.full_run.subprocess.run',return_value=SimpleNamespace(returncode=1)) as run:
            self.assertEqual(worker(self.run,0,self.root),1)
            env=run.call_args.kwargs['env']
            self.assertEqual(env['VIDEO_IDS'],'')
            self.assertEqual(env['CONTINUE_ON_ERROR'],'1')
            self.assertEqual(env['VLLM_PORT'],'auto')
        for vid in plan['shards'][0]:self.labels(vid,self.run/'teacher')
        with patch('stage2.full_run.subprocess.run') as run:
            self.assertEqual(worker(self.run,0,self.root),0)
            run.assert_not_called()

    def test_submission_records_dependencies_and_rejects_active_duplicates(self):
        self.plan()
        preview=submit(self.run,self.root,dry_run=True)
        self.assertIn('--array=0-1%4',preview['commands'][0])
        self.assertFalse((self.run/'submissions.json').exists())
        with patch('stage2.full_run.subprocess.check_output',side_effect=['123\n','124\n']) as call:
            result=submit(self.run,self.root)
            self.assertEqual(result['jobs'],['123','124'])
            self.assertIn('--dependency=afterany:123',call.call_args.args[0])
        with patch('stage2.full_run.subprocess.check_output',return_value='123 RUNNING\n') as call:
            with self.assertRaisesRegex(ValueError,'still active'):submit(self.run,self.root)
            self.assertEqual(call.call_count,1)

    def test_atomic_publication_keeps_prior_complete_batch_on_write_failure(self):
        target=self.root/'labels.jsonl'
        save_events_atomic(target,[{'batch':1}])
        before=target.read_bytes()
        with patch('stage2.annotate.os.replace',side_effect=OSError('interrupted publication')):
            with self.assertRaises(OSError):save_events_atomic(target,[{'batch':1},{'batch':2}])
        self.assertEqual(target.read_bytes(),before)
        self.assertEqual(list(self.root.glob('*.tmp')),[])

    def test_active_seed_is_deferred_and_imported_before_worker_without_duplication(self):
        seed=self.root/'active-seed'
        self.labels('x',seed)
        plan=self.plan(seed_teacher_dir=seed,seed_dependency='999')
        self.assertFalse((self.run/'teacher/x.jsonl').exists())
        preview=submit(self.run,self.root,dry_run=True)
        self.assertIn('--dependency=afterany:999',preview['commands'][0])
        index=next(i for i,ids in enumerate(plan['shards']) if 'x' in ids)
        for vid in plan['shards'][index]:
            if vid!='x':self.labels(vid,self.run/'teacher')
        with patch('stage2.full_run.subprocess.run') as run:
            self.assertEqual(worker(self.run,index,self.root),0)
            run.assert_not_called()
        self.assertEqual((self.run/'teacher/x.jsonl').read_bytes(),(seed/'x.jsonl').read_bytes())

    def test_annotation_cli_publishes_and_resumes_a_short_unlabelled_tail(self):
        target=self.root/'cli.jsonl'
        teacher=SimpleNamespace(summary=Mock(return_value=
            'No ultrasound-relevant visual information in this interval.'))
        args=['annotate','--video',self.mapping['x'],'--video-id','x',
              '--output',str(target),'--model',MODEL,'--resume','--transport-preflight']
        with patch('sys.argv',args), patch.dict('os.environ',{'OPENAI_API_KEY':'test-only'}), \
                patch('stage2.annotate.VideoReader',side_effect=lambda *args:
                      SimpleNamespace(count=61,fps=1,close=lambda:None)), \
                patch('stage2.annotate.ChatTeacher',return_value=teacher), \
                patch('stage2.annotate.make_silent_clip'), contextlib.redirect_stdout(io.StringIO()):
            main()
            self.assertEqual(len(target.read_text().splitlines()),8)
            calls=teacher.summary.call_count
            main()
            self.assertEqual(teacher.summary.call_count,calls)


if __name__=='__main__':unittest.main()
