import unittest

from torch import nn

from stage3.model import ANSWER, WAIT, DecisionModel, answerability_label


class Stage3Tests(unittest.TestCase):
    def test_target_marker_determines_bce_label(self):
        self.assertEqual(answerability_label(f"{WAIT} need a kidney view"), 0.0)
        self.assertEqual(answerability_label(f"{ANSWER} no stone is visible"), 1.0)

    def test_unmarked_target_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "must start"):
            answerability_label("not enough evidence")

    def test_decision_model_registers_the_bce_head_as_a_module(self):
        self.assertTrue(issubclass(DecisionModel, nn.Module))


if __name__ == "__main__":
    unittest.main()
