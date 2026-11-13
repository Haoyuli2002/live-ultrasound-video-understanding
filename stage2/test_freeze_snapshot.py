import json
import tempfile
import unittest
from pathlib import Path
from .freeze_snapshot import freeze_snapshot
from .provenance import digest
from .release import validate_release
from .dataset_versions import validate_dataset_version, normalize_version

class SnapshotTests(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory();self.addCleanup(self.temp.cleanup)
        self.root=Path(self.temp.name);self.source=self.root/'snapshot';self.source.mkdir()
        records=[]
        for split in ('train','eval'):
            vid=split+'vid';text='A grayscale ultrasound image is displayed.'
            groups=[]
            for start,end in [(0,60),(60,70)]:
                row=dict(schema_version='stage2_prose_v2',video_id=vid,block_window=[start,end],local_sub_summaries=[dict(window=[s,s+10],local_summary_target=text) for s in range(start,end,10)])
                if start==0:row.update(block_summary_target=text,global_summary_target=text)
                groups.append(row)
            (self.source/f'{split}.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in groups))
            records.append(dict(video_id=vid,split=split,source_sha256=split,duration_seconds=70,local_targets=7,full_blocks=1,train_targets=9))
        (self.source/'videos.json').write_text(json.dumps(records))
        (self.source/'video_map.json').write_text('{}')
        self.manifest=dict(schema='stage2_av_asr_audited_dataset_v1',status='audit_complete_not_frozen_release',video_count=2,train_videos=1,eval_videos=1,train_targets=9,eval_targets=9,expected_targets=18,source_root='audit',files={n:digest(self.source/n) for n in ['train.jsonl','eval.jsonl','videos.json','video_map.json']})
        self.save_manifest()
    def save_manifest(self):
        (self.source/'manifest.json').write_text(json.dumps(self.manifest))
    def test_freeze_tail_and_version_gate(self):
        out=self.root/'release';m=freeze_snapshot(self.source,out)
        validate_release(out/'manifest.json');validate_dataset_version(m,'visual_asr')
        self.assertEqual(m['targets'],18)
        self.assertEqual((out/'train.jsonl').read_bytes(),(self.source/'train.jsonl').read_bytes())
        with self.assertRaises(ValueError):validate_dataset_version(m,'visual_only')
        with self.assertRaises(FileExistsError):freeze_snapshot(self.source,out)
        self.assertEqual(normalize_version('visual'),'visual_only')
    def test_tampering_rejected(self):
        with (self.source/'train.jsonl').open('a') as f:f.write('\n')
        with self.assertRaisesRegex(ValueError,'hash mismatch'):freeze_snapshot(self.source,self.root/'release')
    def test_missing_tail_even_with_updated_hash_rejected(self):
        f=self.source/'train.jsonl';f.write_text(f.read_text().splitlines()[0]+'\n')
        self.manifest['files']['train.jsonl']=digest(f);self.save_manifest()
        with self.assertRaisesRegex(ValueError,'Local coverage'):freeze_snapshot(self.source,self.root/'release')
