import unittest

from .clean_asr import clean_transcript, parse_response, validate_batch
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


if __name__ == "__main__":
    unittest.main()
