import copy
import unittest
from stage2.grounding import audit_profile, audit_window, audit_split_assignments, summarize_review

class GroundingTests(unittest.TestCase):
    def setUp(self):
        self.frame = {'source_pts_seconds':345.04,'source_sha256':'a','timestamp_kind':'decoded_source_pts'}
        self.record = {'target':'On-screen labels identify the liver and diaphragm.',
            'source_sha256':'a','label_type':'local_10s','window':[340,350],
            'claims':[{'kind':'screen_label','observation':'Liver/Diaphragm labels',
                       'on_screen_text':'Liver, Diaphragm','frame_ids':['f345']}]}
    def test_future_label_rejected_but_raw_preserved(self):
        record=copy.deepcopy(self.record);record['window']=[320,330];before=copy.deepcopy(record)
        self.assertIn('claim_0:outside_window_evidence',audit_window(record,{'f345':self.frame},expected_source_sha256='a'))
        self.assertEqual(record,before)
    def test_half_open_boundary(self):
        f={**self.frame,'source_pts_seconds':350}
        self.assertTrue(audit_window(self.record,{'f345':f},expected_source_sha256='a'))
        f['source_pts_seconds']=349.999
        self.assertEqual(audit_window(self.record,{'f345':f},expected_source_sha256='a'),[])
    def test_fake_midpoint_and_replaced_video_not_accepted(self):
        f={**self.frame,'timestamp_kind':'nominal_bin_midpoint','source_sha256':'b'}
        issues=audit_window(self.record,{'f345':f},expected_source_sha256='a')
        self.assertIn('claim_0:unverified_timestamp',issues)
        self.assertIn('claim_0:frame_source_mismatch',issues)
    def test_motion_needs_explicit_dense_review(self):
        r=copy.deepcopy(self.record);r['claims'][0]['kind']='motion'
        self.assertIn('claim_0:motion_requires_dense_review',audit_window(r,{'f345':self.frame},expected_source_sha256='a'))
    def test_measurement_must_keep_label_association(self):
        r=copy.deepcopy(self.record);r['claims'][0]['kind']='measurement'
        self.assertIn('claim_0:measurement_label_association_missing',audit_window(r,{'f345':self.frame},expected_source_sha256='a'))
    def test_every_stage_requires_value_and_reason(self):
        p={'video_type':'ultrasound_tutorial','video_type_reason':'Probe and image shown together.',
           'visual_quality_reason':'Labels are readable.',
           'stage_value':{s:{'value':'high','reason':'Direct visual examination evidence.'} for s in ('stage1','stage2','stage3')}}
        self.assertEqual(audit_profile(p),[])
        del p['stage_value']['stage3']['reason']
        self.assertIn('missing_stage3_reason',audit_profile(p))
    def test_eval_cannot_silently_become_train(self):
        self.assertEqual(audit_split_assignments([{'video_id':'v','split':'train'},{'video_id':'v','split':'eval'}]),['v:train_eval_overlap'])
    def test_report_never_implies_semantic_acceptance(self):
        r=summarize_review([{'issues':[]},{'issues':['future']}]);self.assertEqual(r['flagged'],1)
        self.assertFalse(r['training_approved']);self.assertFalse(r['clinical_grounding_verified'])

if __name__=='__main__':unittest.main()
