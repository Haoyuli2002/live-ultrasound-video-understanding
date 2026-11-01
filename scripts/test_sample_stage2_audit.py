"""Sampling fidelity and completeness checks without a model or GPU."""
import hashlib
import json
from pathlib import Path
import tempfile
import unittest

from scripts.sample_stage2_audit import extract
from stage2.annotate import PROMPT_SHA256
from stage2.full_run import MODEL, expected_counts
from stage2.sampling import POLICY
from stage2.summary_schema import PROSE_SCHEMA_VERSION


class Stage2AuditTests(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory();self.addCleanup(self.temp.cleanup)
        self.root=Path(self.temp.name);self.run=self.root/'run'
        (self.run/'teacher').mkdir(parents=True)
        self.raw='  View: unclear structures; Anatomy: indistinct. \n'
        entries=[]
        for i in range(12):
            vid=f'v{i:02d}'
            source=self.root/f'{vid}.mp4';source.write_bytes(vid.encode())
            config={'source_sha256':hashlib.sha256(source.read_bytes()).hexdigest()}
            rows=[]
            def event(label, window, block):
                return {'video_id':vid,'label_type':label,'window':window,'block_window':block,
                        'target':self.raw,'schema_version':PROSE_SCHEMA_VERSION,
                        'teacher':MODEL,'annotation_config':config}
            for a in (0,60):
                block=[a,a+60]
                rows.extend(event('local_10s',[s,s+10],block) for s in range(a,a+60,10))
                rows.extend([event('block_60s',block,block),event('global_prefix',[0,a+60],block)])
            rows.append(event('local_10s',[120,130],[120,131]))
            # One source is intentionally partial, but its complete minute must
            # not be mistaken for a complete video during pool construction.
            if i==11:rows=rows[:8]
            (self.run/'teacher'/f'{vid}.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in rows))
            entries.append({'video_id':vid,'path':str(source),'duration_sec':131,
                            'annotation_config':config,'expected':expected_counts(131)})
        (self.run/'manifest.json').write_text(json.dumps({'schema':'stage2_full_run_v1',
            'prompt_sha256':PROMPT_SHA256,'sampling_policy':POLICY,'model':MODEL,'videos':entries}))

    def test_ten_distinct_videos_ten_full_blocks_eighty_exact_targets_and_provenance(self):
        before={p.resolve():p.read_bytes() for p in (self.run/'teacher').glob('*.jsonl')}
        summary=extract(self.run,self.root/'audit')
        payload=json.loads((self.root/'audit/payload.json').read_text())
        self.assertEqual(summary['eligible_videos'],11)
        self.assertEqual(summary['blocks'],10);self.assertEqual(summary['summaries'],80)
        self.assertEqual(len(set(summary['selected_videos'])),10)
        self.assertNotIn('v11',summary['selected_videos'])
        self.assertEqual(summary['text_issue_counts']['possible_perspective_labels'],80)
        for block in payload['blocks']:
            self.assertEqual(block['block_window'][1]-block['block_window'][0],60)
            self.assertEqual(len(block['samples']),8)
            self.assertTrue(all(r['target']==self.raw for r in block['samples']))
            self.assertEqual(block['samples'][-1]['window'],[0,block['block_window'][1]])
            source=Path(block['teacher_file'])
            self.assertEqual(block['teacher_file_sha256'],hashlib.sha256(before[source]).hexdigest())
        self.assertFalse(summary['content_review_complete'])
        self.assertTrue(all(p.read_bytes()==data for p,data in before.items()))

    def test_fixed_seed_is_reproducible_and_existing_audit_is_not_overwritten(self):
        one=extract(self.run,self.root/'one',seed=123)
        two=extract(self.run,self.root/'two',seed=123)
        self.assertEqual(one,two)
        self.assertEqual((self.root/'one/payload.json').read_bytes(),(self.root/'two/payload.json').read_bytes())
        with self.assertRaises(FileExistsError):extract(self.run,self.root/'one')

    def test_insufficient_full_blocks_or_videos_fail_without_publishing_sample(self):
        with self.assertRaisesRegex(ValueError,'Only 0 complete videos'):
            extract(self.run,self.root/'not-enough-blocks',blocks_per_video=3)
        with self.assertRaisesRegex(ValueError,'Only 11 complete videos'):
            extract(self.run,self.root/'not-enough-videos',videos=12)
        self.assertFalse((self.root/'not-enough-videos').exists())


if __name__=='__main__':unittest.main()
