import json
import tempfile
import unittest
from pathlib import Path

from stage1.data import CONDITIONS, build_dataset, build_rows, frame_timestamps, sentences
from stage1.model import messages
from stage1.train import validate_rows


class Stage1Tests(unittest.TestCase):
    def setUp(self):
        self.transcript = {"video_id": "v1", "segments": [
            {"start": 0, "end": 5, "text": "The probe shows the liver."},
            {"start": 5, "end": 10, "text": "The kidney appears below it."},
            {"start": 10, "end": 12, "text": "Incomplete tail"},
        ]}

    def test_complete_sentences_and_three_conditions(self):
        self.assertEqual(len(sentences(self.transcript["segments"])), 2)
        rows = build_rows(self.transcript)
        self.assertEqual(tuple(row["condition"] for row in rows), CONDITIONS)
        validate_rows(rows)
        before, through, masked = rows
        self.assertEqual(before["video_window"], masked["video_window"])
        self.assertEqual(before["video_window"][1], before["sentence_window"][0])
        self.assertEqual(through["video_window"][1], through["sentence_window"][1])
        self.assertEqual(before["historical_asr"], through["historical_asr"])
        self.assertEqual(masked["historical_asr"], "")
        self.assertEqual({row["target"] for row in rows}, {"The kidney appears below it."})

    def test_masked_prompt_does_not_leak_asr(self):
        rows = build_rows(self.transcript)
        masked_text = str(messages(rows[2], frames=[]))
        normal_text = str(messages(rows[0], frames=[]))
        self.assertNotIn("The probe shows the liver.", masked_text)
        self.assertIn("[ASR MASKED]", masked_text)
        self.assertIn("The probe shows the liver.", normal_text)

    def test_rejects_incomplete_group(self):
        with self.assertRaisesRegex(ValueError, "Incomplete"):
            validate_rows(build_rows(self.transcript)[:2])

    def test_unpunctuated_segments_are_opt_in_utterance_targets(self):
        transcript = {"video_id": "v1", "segments": [
            {"start": 1, "end": 4, "text": "the probe is on the abdomen"},
            {"start": 4, "end": 7, "text": "we can now see the kidney"},
            {"start": 7, "end": 10, "text": "move the probe a little lower"},
        ]}
        self.assertEqual(build_rows(transcript), [])
        rows = build_rows(transcript, unpunctuated_fallback="segment")
        self.assertEqual(len(rows), 6)
        validate_rows(rows)
        self.assertEqual({row["target_unit_type"] for row in rows},
                         {"asr_segment_fallback"})
        self.assertEqual(rows[0]["sentence_window"], [4.0, 7.0])
        self.assertIn("target time-aligned narration utterance", str(messages(rows[0], [])))

    def test_recent_sparse_sampling_preserves_history_and_focuses_recent_video(self):
        uniform = frame_timestamps([0, 3600], 120)
        focused = frame_timestamps([0, 3600], 120, sampling="recent_sparse")
        self.assertEqual(len(uniform), len(focused))
        self.assertTrue(all(0 <= t < 3600 for t in focused))
        self.assertEqual(sum(t >= 3480 for t in focused), 96)
        self.assertEqual(sum(t < 3480 for t in focused), 24)
        self.assertEqual(focused, sorted(focused))
        self.assertEqual(frame_timestamps([0, 60], 120, sampling="recent_sparse"),
                         frame_timestamps([0, 60], 120))
        near_boundary = frame_timestamps([0, 121], 120, sampling="recent_sparse")
        self.assertEqual(sum(t < 1 for t in near_boundary), 1)
        self.assertEqual(sum(t >= 1 for t in near_boundary), 119)

    def test_builder_respects_selected_video_map(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            transcripts = root / "transcripts"
            transcripts.mkdir()
            (transcripts / "v1.json").write_text(json.dumps(self.transcript))
            (transcripts / "excluded.json").write_text(json.dumps({
                **self.transcript, "video_id": "excluded"}))
            selected = root / "selected.json"
            selected.write_text(json.dumps({"v1": "/remote/v1.mp4"}))
            output = root / "samples.jsonl"
            summary = build_dataset(transcripts, output, video_map=selected)
            rows = [json.loads(line) for line in output.read_text().splitlines()]
            self.assertEqual(summary["selected_videos"], 1)
            self.assertEqual(summary["paired_sentences"], 1)
            self.assertEqual({row["video_id"] for row in rows}, {"v1"})
            validate_rows(rows)

    def test_missing_selected_transcript_does_not_replace_output(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            transcripts = root / "transcripts"
            transcripts.mkdir()
            selected = root / "selected.json"
            selected.write_text(json.dumps({"missing": "/remote/missing.mp4"}))
            output = root / "samples.jsonl"
            output.write_text("previous result\n")
            with self.assertRaisesRegex(FileNotFoundError, "selected transcripts"):
                build_dataset(transcripts, output, video_map=selected)
            self.assertEqual(output.read_text(), "previous result\n")


if __name__ == "__main__":
    unittest.main()
