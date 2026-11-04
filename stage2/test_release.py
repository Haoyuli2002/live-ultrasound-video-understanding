import json
from pathlib import Path
import tempfile
import unittest
from .build_data import build_rows
from .provenance import digest
from .release import prepare_release,resolve_video_map,validate_release,write_rows


class ReleaseTests(unittest.TestCase):
    def fixture(self,root):
        plan=[];accepted=[]
        for vid,split in [('one','train'),('two','eval')]:
            video=root/'source'/f'{split}_full295'/f'{vid}.mp4'
            video.parent.mkdir(parents=True,exist_ok=True);video.write_bytes(vid.encode())
            source=digest(video)
            plan.append(dict(video_id=vid,split=split,source_sha256=source,duration=60))
            events=[dict(video_id=vid,label_type='local_10s',window=[t,t+10],block_window=[0,60],
                         target='A clinically relevant visual description.',schema_version='stage2_prose_v2')
                    for t in range(0,60,10)]
            events += [dict(video_id=vid,label_type=k,window=[0,60],block_window=[0,60],
                            target='Visible ultrasound findings are described here.',schema_version='stage2_prose_v2')
                       for k in ['block_60s','global_prefix']]
            d=root/'accepted_videos'/split/vid;d.mkdir(parents=True)
            write_rows(d/'teacher.reviewed.jsonl',events)
            write_rows(d/'blocks.reviewed.jsonl',build_rows(events,validate_text=False))
            ledger=root/'videos'/vid;ledger.mkdir(parents=True)
            (ledger/'reviews.jsonl').write_text('reviewed\n')
            (ledger/'boundary_evidence.json').write_text('{}')
            m=dict(final_acceptance_complete=True,boundary_review_complete=True,
                   all_targets_have_current_visual_support=True,native_acceptance_complete=vid=='one',
                   user_first_pass_acceptance={'accepted':vid=='two'},source_sha256=source,
                   targets=8,acceptance_basis='native' if vid=='one' else 'user_first_pass',
                   review_sha256=digest(ledger/'reviews.jsonl'),
                   boundary_evidence_sha256=digest(ledger/'boundary_evidence.json'))
            (d/'manifest.json').write_text(json.dumps(m))
            accepted.append(dict(video_id=vid,split=split))
        (root/'plan.json').write_text(json.dumps(dict(videos=plan,targets=16)))
        (root/'accepted_videos/index.json').write_text(json.dumps(dict(completed_videos=accepted)))

    def test_complete_release_preserves_split_and_waiver_and_detects_tampering(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);self.fixture(root)
            output=root/'release'
            self.assertTrue(prepare_release(root,output)['training_ready'])
            manifest=validate_release(output/'manifest.json')
            self.assertEqual(manifest['splits'],{'train':['one'],'eval':['two']})
            mapped=resolve_video_map(output/'manifest.json',root/'source',root/'map.json')
            self.assertEqual(set(mapped),{'one','two'})
            (output/'train.jsonl').write_text('tampered')
            with self.assertRaisesRegex(ValueError,'hash changed'):validate_release(output/'manifest.json')

    def test_incomplete_acceptance_produces_no_training_release(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);self.fixture(root)
            (root/'accepted_videos/index.json').write_text(json.dumps(dict(completed_videos=[{'video_id':'one','split':'train'}])))
            output=root/'release';report=prepare_release(root,output)
            self.assertFalse(report['training_ready']);self.assertEqual(report['missing_videos'],['two'])
            self.assertFalse(output.exists())

    def test_stale_visual_ledger_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);self.fixture(root)
            (root/'videos/one/reviews.jsonl').write_text('changed after acceptance')
            with self.assertRaisesRegex(ValueError,'Stale'):prepare_release(root,root/'release')

    def test_reencoded_video_rejected_even_with_same_id(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);self.fixture(root);prepare_release(root,root/'release')
            (root/'source/train_full295/one.mp4').write_bytes(b'reencoded video')
            with self.assertRaisesRegex(ValueError,'hash mismatch'):
                resolve_video_map(root/'release/manifest.json',root/'source',root/'map.json')


if __name__=='__main__':unittest.main()
