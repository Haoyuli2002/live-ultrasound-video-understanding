import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

from .clean_asr import (clean_transcript, parse_response, process_video,
                        validate_batch)
from .data import build_rows


class FakeCleaner:
    model = "Qwen/Qwen3.5-27B"

    def clean(self, segments, indices, context):
        result = []
        for index, segment in zip(indices, segments):
            text = segment["text"]
            if "plural line" in text:
                result.append({"index": index,
                               "clean_text": "The pleural line is visible.",
                               "corrections": [{"from": "plural", "to": "pleural",
                                                "reason": "ultrasound anatomy term"}]})
            else:
                result.append({"index": index, "clean_text": "Check for lung sliding.",
                               "corrections": []})
        return result


class CleanASRTests(unittest.TestCase):
    def test_cleaned_segments_preserve_raw_times_and_build_stage1(self):
        raw = {"video_id": "demo", "full_text": "the plural line is visible", "segments": [
            {"start": 0, "end": 2, "text": "the plural line is visible"},
            {"start": 2, "end": 4, "text": "check for lung sliding"},
            {"start": 4, "end": 6, "text": "check for lung sliding"},
        ], "sentence_units": [{"text": "stale", "start": 0, "end": 2}]}
        result, audit = clean_transcript(raw, FakeCleaner(), batch_segments=2)
        self.assertEqual(result["raw_segments"], raw["segments"])
        self.assertEqual([s["start"] for s in result["segments"]], [0, 2, 4])
        self.assertEqual([s["end"] for s in result["segments"]], [2, 4, 6])
        self.assertEqual(result["segments"][0]["text"], "The pleural line is visible.")
        self.assertEqual(result["raw_full_text"], raw["full_text"])
        self.assertTrue(result["full_text"].startswith("The pleural line is visible."))
        self.assertNotIn("sentence_units", result)
        self.assertEqual(audit["term_correction_count"], 1)
        self.assertEqual(len(build_rows(result)), 6)
        self.assertEqual(raw["segments"][0]["text"], "the plural line is visible")

    def test_reject_missing_or_unrecorded_changes(self):
        source = [{"text": "the plural line"}]
        with self.assertRaises(ValueError):
            validate_batch(source, [], [3])
        with self.assertRaises(ValueError):
            validate_batch(source, [{"index": 3, "clean_text": "The pleural line.",
                                     "corrections": []}], [3])
        with self.assertRaises(ValueError):
            validate_batch(source, [{"index": 3, "clean_text": "An unrelated invented sentence.",
                                     "corrections": [{"from": "plural", "to": "invented",
                                                      "reason": "guess"}]}], [3])

    def test_parse_fenced_json(self):
        raw = '```json\n{"segments":[{"index":0,"clean_text":"Hello.","corrections":[]}]}\n```'
        self.assertEqual(parse_response(raw)[0]["clean_text"], "Hello.")

    def test_multiword_correction_grounded_by_word_sequence(self):
        # "for serothropathy" differs from the raw spacing/punctuation but is a
        # contiguous word run, so word-level grounding must accept it.
        source = [{"text": "check, for serothropathy signs"}]
        returned = [{"index": 0, "clean_text": "Check for hypothyroidism signs.",
                     "corrections": [{"from": "for serothropathy",
                                      "to": "for hypothyroidism",
                                      "reason": "clear ASR error"}]}]
        checked = validate_batch(source, returned, [0])
        self.assertEqual(checked[0]["corrections"][0]["to"], "for hypothyroidism")

    def test_multiword_correction_absent_is_rejected(self):
        source = [{"text": "check for lung sliding"}]
        returned = [{"index": 0, "clean_text": "Check for hypothyroidism.",
                     "corrections": [{"from": "for serothropathy",
                                      "to": "for hypothyroidism",
                                      "reason": "ungrounded"}]}]
        with self.assertRaises(ValueError):
            validate_batch(source, returned, [0])


class FailingCleaner:
    model = "Qwen/Qwen3.5-27B"

    def clean(self, segments, indices, context):
        return [{"index": index, "clean_text": "An entirely invented replacement sentence.",
                 "corrections": [{"from": "plural", "to": "invented", "reason": "guess"}]}
                for index, segment in zip(indices, segments)]


class ProcessVideoTests(unittest.TestCase):
    def _args(self, root: Path):
        return SimpleNamespace(
            transcripts=root / "src", output_dir=root / "out",
            audit_output=root / "audit.jsonl", batch_segments=2,
            context_segments=0, resume=False)

    def test_failed_video_is_skipped_and_recorded(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            args = self._args(root)
            args.transcripts.mkdir(parents=True)
            args.output_dir.mkdir(parents=True)
            (args.transcripts / "bad.json").write_text(json.dumps(
                {"segments": [{"start": 0, "end": 2, "text": "the plural line"}]}),
                encoding="utf-8")
            status = process_video("bad", args, FailingCleaner())
            self.assertEqual(status["status"], "failed")
            # No partial output file is written for a failed video.
            self.assertFalse((args.output_dir / "bad.json").exists())
            audit = [json.loads(line) for line in
                     args.audit_output.read_text(encoding="utf-8").splitlines()]
            self.assertEqual(audit[-1]["status"], "failed")
            self.assertEqual(audit[-1]["video_id"], "bad")

    def test_successful_video_writes_output_and_audit(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            args = self._args(root)
            args.transcripts.mkdir(parents=True)
            args.output_dir.mkdir(parents=True)
            (args.transcripts / "good.json").write_text(json.dumps(
                {"segments": [{"start": 0, "end": 2, "text": "the plural line is visible"}]}),
                encoding="utf-8")
            status = process_video("good", args, FakeCleaner())
            self.assertEqual(status["status"], "cleaned")
            result = json.loads((args.output_dir / "good.json").read_text(encoding="utf-8"))
            self.assertEqual(result["segments"][0]["text"], "The pleural line is visible.")
            audit = [json.loads(line) for line in
                     args.audit_output.read_text(encoding="utf-8").splitlines()]
            self.assertEqual(audit[-1]["status"], "cleaned")


if __name__ == "__main__":
    unittest.main()
