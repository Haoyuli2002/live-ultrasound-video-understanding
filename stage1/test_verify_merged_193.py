"""CPU-only tests for the read-only Stage 1 merged-data verifier."""
from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from .data import build_rows
from .seven_video_pilot import PILOT_IDS
from .verify_merged_193 import verify


class VerifyMerged193Tests(unittest.TestCase):
    def setUp(self) -> None:
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.pilot_ids = list(PILOT_IDS)
        self.baseline_ids = ["baseline-A", "baseline-B"]
        self.full_ids = self.baseline_ids + self.pilot_ids
        self.full = self.root / "full.json"
        self.pilot_map = self.root / "pilot.json"
        self.eval_map = self.root / "eval.json"
        self.baseline = self.root / "baseline.jsonl"
        self.pilot = self.root / "pilot.jsonl"
        self.merged = self.root / "merged.jsonl"
        self.write_json(self.full, {video_id: f"/videos/{video_id}.mp4" for video_id in self.full_ids})
        self.write_json(self.pilot_map, {video_id: f"/videos/{video_id}.mp4" for video_id in self.pilot_ids})
        self.write_json(self.eval_map, {"eval-A": "/videos/eval-A.mp4"})
        self.base_rows = [row for video_id in self.baseline_ids for row in self.make_rows(video_id)]
        self.pilot_rows = [row for video_id in self.pilot_ids for row in self.make_rows(video_id)]
        self.write_rows(self.baseline, self.base_rows)
        self.write_rows(self.pilot, self.pilot_rows)
        self.write_rows(self.merged, self.base_rows + self.pilot_rows)

    @staticmethod
    def make_rows(video_id: str) -> list[dict]:
        return build_rows({
            "video_id": video_id,
            "sentence_units": [
                {"text": "This is the history.", "start": 0, "end": 2},
                {"text": "This is the target.", "start": 3, "end": 5},
            ],
        })

    @staticmethod
    def write_json(path: Path, value: object) -> None:
        path.write_text(json.dumps(value), encoding="utf-8")

    @staticmethod
    def write_rows(path: Path, rows: list[dict]) -> None:
        path.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")

    def run_verify(self) -> dict:
        return verify(full_map=self.full, pilot_map=self.pilot_map,
                      baseline=self.baseline, pilot_samples=self.pilot,
                      merged=self.merged, eval_map=self.eval_map,
                      expected_videos=len(self.full_ids))

    def test_valid_merge(self) -> None:
        result = self.run_verify()
        self.assertEqual(result["status"], "PASS")
        self.assertEqual(result["videos"], 9)
        self.assertEqual(result["sentence_groups"], 9)
        self.assertEqual(result["rows"], 27)

    def test_wrong_time_window(self) -> None:
        self.base_rows[0]["video_window"] = [0, 5]
        self.write_rows(self.baseline, self.base_rows)
        self.write_rows(self.merged, self.base_rows + self.pilot_rows)
        with self.assertRaisesRegex(ValueError, "expected video_window"):
            self.run_verify()

    def test_wrong_asr_mask(self) -> None:
        self.base_rows[2]["historical_asr"] = "leaked history"
        self.write_rows(self.baseline, self.base_rows)
        self.write_rows(self.merged, self.base_rows + self.pilot_rows)
        with self.assertRaisesRegex(ValueError, "masked historical_asr"):
            self.run_verify()

    def test_merge_missing_row(self) -> None:
        self.write_rows(self.merged, (self.base_rows + self.pilot_rows)[:-1])
        with self.assertRaisesRegex(ValueError, "merged content differs"):
            self.run_verify()

    def test_eval_overlap(self) -> None:
        self.write_json(self.eval_map, {self.baseline_ids[0]: "/videos/eval.mp4"})
        with self.assertRaisesRegex(ValueError, "train/eval video ID overlap"):
            self.run_verify()

    def test_eval_path_overlap(self) -> None:
        self.write_json(self.eval_map, {"different-id": f"/videos/{self.baseline_ids[0]}.mp4"})
        with self.assertRaisesRegex(ValueError, "train/eval video path overlap"):
            self.run_verify()

    def test_baseline_coverage_gap(self) -> None:
        self.write_rows(self.baseline, self.base_rows[3:])
        with self.assertRaisesRegex(ValueError, "baseline coverage"):
            self.run_verify()


if __name__ == "__main__":
    unittest.main()
