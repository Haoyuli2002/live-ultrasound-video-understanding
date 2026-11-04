import json
from pathlib import Path
import tempfile
import unittest

from .overfit_ab_report import report_pair, report_run


def write_rows(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(r)+"\n" for r in rows))


class ReportTests(unittest.TestCase):
    def fixture(self, root, mode):
        expected = {"short":12} if mode == "short_only" else {"short":12,"block":2,"global":2}
        (root/"submission.json").write_text(json.dumps({"mode":mode,"EPOCHS":"2"}))
        training = [{"epoch":e,"kind":kind,"fresh_memory":True}
                    for e in range(2) for kind,n in expected.items() for _ in range(n)]
        write_rows(root/"training/training_metrics.jsonl", training)
        for name in ["epoch-1", "final"]:
            write_rows(root/f"reconstruction_{name}.jsonl",
                       [{"kind":k,"loss":1.0} for k,n in expected.items() for _ in range(n)])
        recon = root/"reconstruct_final"
        recon.mkdir()
        (recon/"manifest.json").write_text(json.dumps({"reference_used_in_decoding":False}))
        rows = []
        for kind, n in expected.items():
            for i in range(n):
                for condition in ["real", "zero", "other_window"]:
                    # Collapsed Short outputs: exactly one match and 11 wrong-window repeats.
                    rows.append({"video_id":"P3","kind":kind,"window":[i*10,(i+1)*10],
                                 "condition":condition,"reference":f"target {i}",
                                 "prediction":"target 0", "normalized_exact_match":i==0,
                                 "stop_reason":"eos"})
        write_rows(recon/"predictions.jsonl", rows)
        return rows

    def test_counts_repetition_and_horizontal_raw_comparison(self):
        for mode, count in [("short_only",24),("independent",32)]:
            with self.subTest(mode=mode), tempfile.TemporaryDirectory() as tmp:
                root = Path(tmp)
                self.fixture(root, mode)
                report = report_run(root)
                self.assertEqual(report["optimizer_updates"], count)
                short = report["free_running"]["real/short"]
                self.assertEqual(short["unique_predictions"], 1)
                self.assertEqual(short["wrong_window_gt_repeats"], 11)
                self.assertEqual(short["exact_matches"], 1)
                self.assertIn("GT\treal\tzero\tother_window", (root/"comparison.tsv").read_text())

    def test_duplicate_predictions_and_target_leakage_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            rows = self.fixture(root,"short_only")
            rows[-1] = rows[0]
            write_rows(root/"reconstruct_final/predictions.jsonl", rows)
            with self.assertRaisesRegex(ValueError, "Duplicate"):
                report_run(root)
            (root/"reconstruct_final/manifest.json").write_text(
                json.dumps({"reference_used_in_decoding":True}))
            with self.assertRaisesRegex(ValueError, "leakage"):
                report_run(root)

    def test_pending_pair_never_reported_as_completed(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root/"experiments.json").write_text(json.dumps({"jobs":[
                {"mode":"short_only","run_dir":str(root/"short_only"),"job_id":"123"},
                {"mode":"independent","run_dir":str(root/"independent"),"job_id":"124"}]}))
            self.assertEqual(report_pair(root)["status"], "not_finished")


if __name__ == "__main__":
    unittest.main()
