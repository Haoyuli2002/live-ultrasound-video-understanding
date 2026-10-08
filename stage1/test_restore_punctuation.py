import unittest

from stage1.data import build_rows
from stage1.restore_punctuation import (
    collect_boundaries, parse_boundary_json, restore_transcript,
    sentence_units, timed_words,
)


class FakeTeacher:
    def __init__(self, boundaries):
        self.expected = boundaries
        self.calls = []

    def boundaries(self, words, core_start, core_end, context_words):
        self.calls.append((core_start, core_end, context_words))
        return [item for item in self.expected
                if core_start <= item["after"] < core_end]


class RestorePunctuationTests(unittest.TestCase):
    def setUp(self):
        self.transcript = {"video_id": "v1", "segments": [
            {"start": 0, "end": 3, "text": "the probe is on the abdomen"},
            {"start": 3, "end": 6, "text": "now we see the kidney"},
            {"start": 6, "end": 9, "text": "move slightly lower for another view"},
        ]}

    def test_boundaries_preserve_words_and_stage1_uses_precomputed_times(self):
        words = timed_words(self.transcript["segments"])
        self.assertEqual(len(words), 17)
        teacher = FakeTeacher([{"after": 5, "mark": "."},
                               {"after": 10, "mark": "."},
                               {"after": 16, "mark": "."}])
        restored, audit = restore_transcript(self.transcript, teacher,
                                             chunk_words=8, context_words=2)
        self.assertEqual(teacher.calls, [(0, 8, 2), (8, 16, 2), (16, 17, 2)])
        units = restored["sentence_units"]
        self.assertEqual([unit["text"] for unit in units], [
            "the probe is on the abdomen.",
            "now we see the kidney.",
            "move slightly lower for another view.",
        ])
        for unit in units:
            original = [word["text"] for word in words[
                unit["word_start_index"]:unit["word_end_index"]]]
            self.assertEqual(unit["text"].rstrip(".?!").split(), original)
        self.assertEqual(audit["unfinished_tail_words"], 0)
        rows = build_rows(restored)
        self.assertEqual(len(rows), 6)
        self.assertEqual(rows[0]["sentence_window"],
                         [units[1]["start"], units[1]["end"]])
        self.assertEqual({row["target_unit_type"] for row in rows},
                         {"llm_punctuation_sentence"})

    def test_rejects_invalid_boundary_and_parses_thinking_prefix(self):
        words = timed_words(self.transcript["segments"])
        class BadTeacher:
            def boundaries(self, words, core_start, core_end, context_words):
                return [{"after": core_end, "mark": "."}]
        with self.assertRaisesRegex(ValueError, "Invalid or duplicate"):
            collect_boundaries(words, BadTeacher(), chunk_words=8)
        self.assertEqual(parse_boundary_json(
            '<think>done</think> {"boundaries":[{"after":4,"mark":"?"}]}'),
            [{"after": 4, "mark": "?"}])

    def test_unfinished_tail_is_not_promoted_to_sentence(self):
        words = timed_words(self.transcript["segments"])
        units = sentence_units(words, [{"after": 5, "mark": "."}])
        self.assertEqual(len(units), 1)
        self.assertEqual(units[0]["word_end_index"], 6)


if __name__ == "__main__":
    unittest.main()
