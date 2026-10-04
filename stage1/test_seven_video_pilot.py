import json
import tempfile
import unittest
from pathlib import Path

from .seven_video_pilot import PILOT_IDS, merge_samples, select_pilot


def rows(video_id):
    return [{"video_id": video_id, "sentence_id": 1, "condition": condition,
             "target": "The pleural line is visible."}
            for condition in ("before_with_asr", "through_with_asr", "before_mask_asr")]


class SevenVideoPilotTests(unittest.TestCase):
    def test_select_and_merge_exact_coverage(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            full_map = root / "full.json"
            full_map.write_text(json.dumps({**{i: f"/{i}.mp4" for i in PILOT_IDS},
                                            "existing": "/existing.mp4"}))
            pilot_map = root / "pilot.json"
            self.assertEqual(select_pilot(full_map, pilot_map)["pilot_videos"], 7)
            baseline = root / "baseline.jsonl"
            baseline.write_text("".join(json.dumps(row) + "\n" for row in rows("existing")))
            pilot = root / "pilot.jsonl"
            pilot.write_text("".join(json.dumps(row) + "\n"
                                     for video_id in PILOT_IDS for row in rows(video_id)))
            output = root / "merged.jsonl"
            result = merge_samples(full_map, pilot_map, baseline, pilot, output)
            self.assertEqual(result["videos"], 8)
            self.assertEqual(result["rows"], 24)
            self.assertEqual(len(output.read_text().splitlines()), 24)

    def test_rejects_missing_pilot_video_without_replacing_output(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            full_map = root / "full.json"
            full_map.write_text(json.dumps({i: f"/{i}.mp4" for i in PILOT_IDS}))
            pilot_map = root / "pilot.json"
            select_pilot(full_map, pilot_map)
            baseline = root / "baseline.jsonl"
            baseline.write_text(json.dumps(rows(PILOT_IDS[0])[0]) + "\n")
            pilot = root / "pilot.jsonl"
            pilot.write_text("".join(json.dumps(row) + "\n"
                                     for video_id in PILOT_IDS[:-1] for row in rows(video_id)))
            output = root / "merged.jsonl"
            output.write_text("previous\n")
            with self.assertRaises(ValueError):
                merge_samples(full_map, pilot_map, baseline, pilot, output)
            self.assertEqual(output.read_text(), "previous\n")


if __name__ == "__main__":
    unittest.main()
