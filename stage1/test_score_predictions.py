"""Tests for stage1.score_predictions (token F1 + ROUGE-L aggregation)."""
import json
import tempfile
import unittest
from pathlib import Path

from stage1.score_predictions import (
    build_arg_parser, load_rows, rouge_l_f1, score_rows, token_f1, toks,
)


class ScorePredictionsTests(unittest.TestCase):
    def test_token_f1_identical_and_disjoint(self):
        self.assertAlmostEqual(token_f1("the liver is visible", "the liver is visible"), 1.0)
        self.assertEqual(token_f1("abc def", "ghi jkl"), 0.0)
        # Empty prediction or reference scores 0, never divides by zero.
        self.assertEqual(token_f1("", "the liver"), 0.0)
        self.assertEqual(token_f1("the liver", ""), 0.0)

    def test_token_f1_partial_overlap_matches_formula(self):
        # pred=3 tokens, ref=2 tokens, overlap=2 -> prec=2/3, rec=2/2=1 -> F1=0.8
        self.assertAlmostEqual(token_f1("liver kidney spleen", "liver kidney"), 0.8)

    def test_rouge_l_respects_subsequence_order(self):
        self.assertAlmostEqual(rouge_l_f1("a b c", "a b c"), 1.0)
        # LCS of "a b c d" / "a c" is "a c" (len 2): prec=2/4, rec=2/2 -> F1=2/3.
        self.assertAlmostEqual(rouge_l_f1("a b c d", "a c"), 2 / 3)
        self.assertEqual(rouge_l_f1("", "a b"), 0.0)

    def test_tokenizer_keeps_hyphenated_terms_and_lowercases(self):
        self.assertEqual(toks("B-lines and A-Line"), ["b-lines", "and", "a-line"])

    def test_score_rows_aggregates_overall_and_per_condition(self):
        rows = [
            {"video_id": "v1", "sentence_id": 1, "condition": "before_with_asr",
             "target": "the liver", "prediction": "the liver"},               # f1=1
            {"video_id": "v1", "sentence_id": 1, "condition": "through_with_asr",
             "target": "the liver", "prediction": "the spleen"},              # overlap "the": f1=0.5
        ]
        out = score_rows(rows)
        self.assertEqual(out["num_samples"], 2)
        self.assertAlmostEqual(out["aggregate"]["before_with_asr"]["token_f1"], 1.0)
        self.assertAlmostEqual(out["aggregate"]["through_with_asr"]["token_f1"], 0.5)
        # Overall is the mean of the two conditions.
        self.assertAlmostEqual(out["aggregate"]["overall"]["token_f1"], 0.75)
        self.assertEqual(len(out["per_sample"]), 2)
        self.assertEqual(out["per_sample"][0]["video_id"], "v1")

    def test_missing_condition_defaults_to_unknown(self):
        out = score_rows([{"target": "a b", "prediction": "a b"}])
        self.assertIn("unknown", out["aggregate"])
        self.assertAlmostEqual(out["aggregate"]["overall"]["rouge_l_f1"], 1.0)

    def test_load_rows_rejects_missing_fields(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "bad.jsonl"
            path.write_text(json.dumps({"target": "x"}) + "\n", encoding="utf-8")
            with self.assertRaises(ValueError):
                load_rows(path)

    def test_arg_parser_requires_predictions_and_takes_baseline(self):
        parser = build_arg_parser()
        args = parser.parse_args(["--predictions", "p.jsonl", "--baseline", "b.jsonl"])
        self.assertEqual(args.predictions, "p.jsonl")
        self.assertEqual(args.baseline, "b.jsonl")
        self.assertIsNone(build_arg_parser().parse_args(["--predictions", "p.jsonl"]).baseline)


if __name__ == "__main__":
    unittest.main()
