"""Focused tests for complete-sentence targets and bounded visual context."""
import unittest

from pretrain.build_samples import build_punctuation_sentence_units, build_sentence_units
from pretrain.video_sampling import context_frame_times


class Stage1MainlineTests(unittest.TestCase):
    def test_abbreviation_and_complete_sentence(self):
        segments = [
            {"start": 0, "end": 2, "text": "I'm Dr. Smith."},
            {"start": 2, "end": 4, "text": "The pleura is visible."},
            {"start": 4, "end": 5, "text": "An unfinished"},
        ]
        units = build_punctuation_sentence_units(
            segments, min_words=2, split_punctuation=".?!",
            include_comma_split=False, sentence_max_words=80,
        )
        self.assertEqual([u["text"] for u in units],
                         ["I'm Dr. Smith.", "The pleura is visible."])

    def test_overlong_sentence_is_not_split(self):
        segments = [
            {"start": 0, "end": 4, "text": "The image shows a long sentence."},
            {"start": 4, "end": 8, "text": "This one is also complete."},
        ]
        units = build_sentence_units(
            segments, min_words=2, sentence_max_words=4,
            sentence_max_duration=0,
        )
        self.assertEqual(units, [])

    def test_two_minute_context_samples_one_frame_per_second(self):
        times = context_frame_times(0, 120)
        self.assertEqual(len(times), 120)
        self.assertEqual(times[0], 0.5)
        self.assertEqual(times[-1], 119.5)
        self.assertTrue(all(b - a == 1 for a, b in zip(times, times[1:])))
        self.assertEqual(context_frame_times(0, 5), [0.5, 1.5, 2.5, 3.5, 4.5])
        limited = context_frame_times(0, 120, max_frames=24)
        self.assertEqual(len(limited), 24)
        self.assertEqual(limited[0], 2.5)
        self.assertEqual(limited[-1], 117.5)


if __name__ == "__main__":
    unittest.main()
