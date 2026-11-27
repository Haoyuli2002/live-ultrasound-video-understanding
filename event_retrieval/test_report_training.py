import tempfile
import unittest
from pathlib import Path
from event_retrieval.report_training import telemetry,group_metrics,holdout_report

class ReportTests(unittest.TestCase):
    def test_time_weighted_telemetry_excludes_gaps(self):
        with tempfile.TemporaryDirectory() as d:
            p=Path(d)/'gpu.csv'
            p.write_text('timestamp, index, memory.used [MiB], memory.total [MiB], utilization.gpu [%]\n'
                         '2026/11/27 10:00:00.000, 0, 10, 100, 0\n'
                         '2026/11/27 10:00:01.000, 0, 30, 100, 100\n'
                         '2026/11/27 10:00:04.000, 0, 50, 100, 100\n'
                         '2026/11/27 10:00:20.000, 0, 90, 100, 0\n')
            r=telemetry(p)['per_gpu']['0']
            self.assertEqual(r['mean_gpu_utilization_percent'],87.5)
            self.assertEqual(r['mean_vram_utilization_percent'],35)
            self.assertEqual(r['peak_vram_utilization_percent'],90)
            self.assertEqual(r['covered_seconds'],4)
            self.assertEqual(r['excluded_gap_seconds'],16)

    def test_early_correct_and_unanswered_remain_in_denominator(self):
        records=[]
        for when,answer in [(9,'yes'),(10,'yes'),(12,'wrong'),(None,'')]:
            records.append(dict(task=dict(answer_time=10,reference_answer='yes',qa_type='event_notification'),first_answer_time=when,last_decision={'answer':answer},forced_answer_at_gt='yes'))
        r=group_metrics(records);m=r['metrics']
        self.assertEqual(m['forced_text_match']['rate'],1)
        self.assertEqual(m['first_answer_text_match']['rate'],.5)
        self.assertEqual(m['on_time_and_text_match']['rate'],.25)
        for k in ['early','late','unanswered']:self.assertEqual(m[k]['rate'],.25)
        self.assertEqual(m['text_match_within_2s_after_gt']['rate'],.25)
        self.assertEqual(r['timing']['triggered_count'],3)
        self.assertEqual(r['timing']['late_p90_s'],2)

    def test_incomplete_eval_is_not_zero_accuracy(self):
        with tempfile.TemporaryDirectory() as d:
            r=holdout_report(Path(d))
            self.assertFalse(r['available'])
            self.assertNotIn('dense',r)

if __name__=='__main__':unittest.main()
