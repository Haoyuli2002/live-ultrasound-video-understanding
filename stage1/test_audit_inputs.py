"""CPU-only checks for the Stage 1 SSH audit commands."""
from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from .audit_inputs import audit_cleaning, audit_eval, audit_frames, audit_smoke, audit_timing
from .data import build_rows
from .seven_video_pilot import PILOT_IDS


class AuditInputsTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.source = self.root / "source"
        self.cleaned = self.root / "cleaned"
        self.source.mkdir()
        self.cleaned.mkdir()
        self.pilot_map = self.root / "pilot_map.json"
        self.train_map = self.root / "train_map.json"
        self.eval_map = self.root / "eval_map.json"
        self.audit = self.root / "audit.jsonl"
        self.pilot_samples = self.root / "pilot.jsonl"
        self.baseline = self.root / "baseline.jsonl"
        self.merged = self.root / "merged.jsonl"
        self.eval_samples = self.root / "eval.jsonl"
        self.pilots = list(PILOT_IDS)
        self.write_json(self.pilot_map, {vid: f"/videos/{vid}.mp4" for vid in self.pilots})
        self.write_json(self.train_map, {vid: f"/videos/{vid}.mp4" for vid in self.pilots + ["base"]})
        self.write_json(self.eval_map, {"eval": "/videos/eval.mp4"})
        self.write_rows(self.eval_samples, self.rows("eval"))
        all_pilot_rows = []
        audits = []
        for video_id in self.pilots:
            raw = self.transcript(video_id)
            cleaned = dict(raw)
            cleaned["raw_segments"] = raw["segments"]
            cleaned["raw_full_text"] = raw["full_text"]
            cleaned["asr_cleaning"] = {
                "segment_count": 2, "changed_segments": 0, "term_corrections": [],
            }
            self.write_json(self.source / f"{video_id}.json", raw)
            self.write_json(self.cleaned / f"{video_id}.json", cleaned)
            all_pilot_rows.extend(build_rows(cleaned))
            audits.append({"video_id": video_id, "status": "cleaned",
                           "segment_count": 2, "changed_segments": 0,
                           "term_correction_count": 0, "term_corrections": []})
        self.write_rows(self.audit, audits)
        self.write_rows(self.pilot_samples, all_pilot_rows)
        self.write_json(self.source / "base.json", self.transcript("base"))
        self.base_rows = self.rows("base")
        self.write_rows(self.baseline, self.base_rows)
        self.write_rows(self.merged, self.base_rows + all_pilot_rows)

    @staticmethod
    def transcript(video_id: str) -> dict:
        segments = [
            {"start": 0, "end": 2, "text": "This is the liver."},
            {"start": 3, "end": 5, "text": "Here is the kidney."},
        ]
        return {"video_id": video_id, "segments": segments,
                "full_text": " ".join(segment["text"] for segment in segments)}

    @classmethod
    def rows(cls, video_id: str) -> list[dict]:
        return build_rows(cls.transcript(video_id))

    @staticmethod
    def write_json(path: Path, value: object) -> None:
        path.write_text(json.dumps(value), encoding="utf-8")

    @staticmethod
    def write_rows(path: Path, rows: list[dict]) -> None:
        path.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")

    def cleaning(self, **overrides) -> dict:
        arguments = dict(pilot_map=self.pilot_map, source_dir=self.source,
                         cleaned_dir=self.cleaned, audit=self.audit,
                         pilot_samples=self.pilot_samples, baseline=self.baseline)
        return audit_cleaning(**(arguments | overrides))

    def test_cleaning_pass_and_review_csv(self) -> None:
        output = self.root / "corrections.csv"
        result = self.cleaning(review_csv=output)
        self.assertEqual(result["pilot_videos"], 7)
        self.assertEqual(result["checked_segments"], 14)
        self.assertTrue(output.is_file())
        with self.assertRaises(FileExistsError):
            self.cleaning(review_csv=output)

    def test_cleaning_rejects_changed_timestamp(self) -> None:
        path = self.cleaned / f"{self.pilots[0]}.json"
        value = json.loads(path.read_text())
        value["segments"][0]["end"] = 3
        self.write_json(path, value)
        with self.assertRaisesRegex(ValueError, "timestamps/metadata changed"):
            self.cleaning()

    def test_cleaning_rejects_missing_audit(self) -> None:
        self.write_rows(self.audit, [{"video_id": self.pilots[0], "status": "failed"}])
        with self.assertRaisesRegex(ValueError, "latest audit status"):
            self.cleaning()

    def test_eval_isolation(self) -> None:
        result = audit_eval(train_map=self.train_map, eval_map=self.eval_map,
                            train_samples=self.merged, eval_samples=self.eval_samples)
        self.assertEqual(result["eval_sample_videos"], 1)

    def test_eval_overlap(self) -> None:
        self.write_json(self.eval_map, {"base": "/videos/eval.mp4"})
        with self.assertRaisesRegex(ValueError, "ID overlap"):
            audit_eval(train_map=self.train_map, eval_map=self.eval_map,
                       train_samples=self.merged)

    def test_timing_and_frames(self) -> None:
        review = self.root / "timing.csv"
        result = audit_timing(samples=self.merged, video_map=self.train_map,
                              transcript_dir=self.source, pilot_transcript_dir=self.cleaned,
                              review_csv=review, max_videos=8, per_video=1)
        self.assertEqual(result["sentences"], 8)
        self.assertTrue(review.is_file())
        frames = audit_frames(samples=self.merged)
        self.assertEqual(frames["max_sampled_frames"], 5)

    def test_smoke_reads_actual_config(self) -> None:
        output = self.root / "smoke"
        output.mkdir()
        self.write_json(output / "training_config.json", {
            "train_jsonl": str(self.baseline), "max_steps": 2,
            "model_name": "Qwen/Qwen3.5-2B", "active_frame_budget": 120,
        })
        self.write_json(output / "video_split.json", {
            "train_video_ids": ["base"], "validation_video_ids": ["eval"],
        })
        self.write_rows(output / "validation_samples.jsonl", self.rows("eval"))
        (output / "checkpoint-2").mkdir()
        self.write_json(output / "checkpoint-2" / "trainer_state.json", {"global_step": 2})
        (output / "final").mkdir()
        (output / "final" / "adapter_config.json").write_text("{}")
        result = audit_smoke(output_dir=output)
        self.assertEqual(result["recorded_train_jsonl"], str(self.baseline))
        with self.assertRaisesRegex(ValueError, "not expected"):
            audit_smoke(output_dir=output, expected_input=self.merged)


if __name__ == "__main__":
    unittest.main()
