import os
import unittest

import torch
from lavis.models.blip2_models.Qformer import BertConfig

from .qformer import EventQFormers, event_frame_indices


class EventQFormerTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(42)
        cfg = BertConfig(vocab_size=16, hidden_size=24, num_hidden_layers=2,
                         num_attention_heads=4, intermediate_size=32, encoder_width=12,
                         add_cross_attention=True, cross_attention_freq=2,
                         hidden_dropout_prob=0, attention_probs_dropout_prob=0)
        self.model = EventQFormers(cfg, vision_width=16, llm_width=32)

    def test_masks_order_and_gradients(self):
        recent = torch.randn(2, 3, 4, 16)
        events = torch.randn(2, 2, 3, 4, 16)
        valid = torch.tensor([[[1, 1, 0], [1, 0, 0]], [[0, 0, 0], [0, 0, 0]]], dtype=torch.bool)
        recent_mask = torch.tensor([[1, 1, 0], [1, 0, 0]], dtype=torch.bool)
        full, mask = self.model(recent, events, recent_mask=recent_mask, event_mask=valid)
        self.assertEqual(full.shape, (2, 160, 32))
        self.assertTrue(mask[0].all() and mask[1, 128:].all())
        self.assertFalse(mask[1, :128].any())
        self.assertEqual(full[1, :128].count_nonzero().item(), 0)
        long, _ = self.model.encode_long(events.masked_fill(~valid[..., None, None], float("nan")), valid)
        torch.testing.assert_close(full[:, :128], long)
        short, _ = self.model.encode_short(recent.masked_fill(~recent_mask[..., None, None], 1234), recent_mask)
        torch.testing.assert_close(full[:, 128:], short)
        full[mask].square().mean().backward()
        for p in (self.model.short.queries, self.model.long.queries,
                  self.model.visual_projection.weight, self.model.long.output_projection.weight):
            self.assertTrue(torch.isfinite(p.grad).all())
            self.assertGreater(p.grad.abs().sum().item(), 0)

    def test_empty_history_and_frame_limits(self):
        recent = torch.randn(1, 1, 2, 16)
        prefix, mask = self.model(recent, recent.new_empty(1, 0, 0, 2, 16))
        self.assertFalse(mask[:, :128].any())
        torch.testing.assert_close(prefix[:, 128:], self.model.encode_short(recent)[0])
        with self.assertRaises(ValueError):
            self.model.encode_short(recent, torch.zeros(1, 1, dtype=torch.bool))
        with self.assertRaises(ValueError):
            self.model.encode_long(recent.new_zeros(1, 6, 1, 2, 16))
        with self.assertRaises(ValueError):
            self.model.encode_short(recent.expand(1, 25, 2, 16))

    def test_uniform_event_sampling(self):
        for n in (0, 1, 23, 24):
            torch.testing.assert_close(event_frame_indices(n), torch.arange(n))
        for n in (25, 48, 191):
            grid = event_frame_indices(n)
            self.assertEqual((len(grid), grid[0].item(), grid[-1].item()), (24, 0, n - 1))
            gaps = grid.diff()
            self.assertGreaterEqual(gaps.min().item(), 1)
            self.assertLessEqual((gaps.max() - gaps.min()).item(), 1)


@unittest.skipUnless(os.getenv("INSTRUCTBLIP_QFORMER_CHECKPOINT"), "set checkpoint path for integration test")
class InstructBlipInitializationTests(unittest.TestCase):
    def test_both_towers_and_expanded_queries(self):
        path = os.environ["INSTRUCTBLIP_QFORMER_CHECKPOINT"]
        model = EventQFormers.from_instructblip(path, noise_seed=17)
        source = torch.load(path, weights_only=True, mmap=True, map_location="cpu")["model"]
        source_weights = {k.removeprefix("Qformer.bert."): v for k, v in source.items()
                          if k.startswith("Qformer.bert.")}
        short = model.short.encoder.state_dict()
        long = model.long.encoder.state_dict()
        self.assertEqual(set(short), set(source_weights))
        self.assertEqual(set(long), set(source_weights))
        for name, expected in source_weights.items():
            self.assertTrue(torch.equal(short[name], expected), name)
            self.assertTrue(torch.equal(long[name], expected), name)
            self.assertNotEqual(short[name].data_ptr(), long[name].data_ptr(), name)
        q = source["query_tokens"]
        self.assertTrue(torch.equal(model.short.queries, q))
        self.assertTrue(torch.equal(model.long.queries[:, :32], q))
        noise = torch.randn((1, 96, 768), generator=torch.Generator().manual_seed(17))
        expected = q.repeat(1, 3, 1) + noise * (0.01 * q.std())
        self.assertTrue(torch.equal(model.long.queries[:, 32:], expected))
        self.assertNotEqual(model.short.queries.data_ptr(), model.long.queries.data_ptr())


if __name__ == "__main__":
    unittest.main()
