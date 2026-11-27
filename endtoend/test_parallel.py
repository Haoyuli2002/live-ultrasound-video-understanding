import unittest
from unittest.mock import patch

import torch

from .data import sample_times
from .model import forward_batch
from .test_endtoend import tiny_model, example


def fixture():
    rows = [dict(example(i % 2), video_id=v, sample_id=str(i), task_id=str(i), current_time=t)
            for i, (v, t) in enumerate([('a', 0.), ('a', 1.4), ('a', 3.2), ('a', 6.), ('a', 3.2), ('b', 2.)])]
    frames = {}
    features = []
    for row in rows:
        for t in sample_times(row['current_time']):
            frames.setdefault((row['video_id'], t), torch.randn(2, 32))
        features.append(torch.stack([frames[(row['video_id'], t)] for t in sample_times(row['current_time'])])[None])
    return rows, features


class ParallelTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(2)
        torch.manual_seed(61)
        self.model = tiny_model().eval()
        for branch in (self.model.core.short_qformer, self.model.core.long_qformer):
            branch.bert.encoder.config.gradient_checkpointing = False
        with torch.no_grad():
            self.model.core.readiness_head.net[-1].weight.normal_(std=.1)

    def test_parallel_matches_serial_values_gradients_and_reuses_history(self):
        rows, features = fixture()
        serial = forward_batch(self.model, rows, features)
        params = [self.model.core.vision_proj.weight, self.model.core.long_query_tokens,
                  self.model.core.rdy_token, self.model.core.output_proj.weight]
        expected = torch.autograd.grad(serial['loss'], params)
        self.model.parallel_endpoints = True
        with patch.object(self.model.core, '_run_branch', wraps=self.model.core._run_branch) as branch:
            parallel = forward_batch(self.model, rows, features)
            self.assertEqual(branch.call_count, 2)
            self.assertTrue(all(call.args[2].shape[0] == len(rows) for call in branch.call_args_list))
        torch.testing.assert_close(parallel['logits'], serial['logits'], atol=2e-6, rtol=2e-5)
        for got, ref in zip(parallel['outputs'], serial['outputs']):
            torch.testing.assert_close(got['prefix_embeds'], ref['prefix_embeds'], atol=2e-6, rtol=2e-5)
        actual = torch.autograd.grad(parallel['loss'], params)
        for got, ref in zip(actual, expected):
            torch.testing.assert_close(got, ref, atol=2e-5, rtol=2e-4)
        self.assertEqual(self.model.core.last_replay_stats['memory_updates'], 12)
        self.assertLess(self.model.core.last_replay_stats['memory_updates'], self.model.core.last_replay_stats['independent_updates'])

    def test_short_long_readout_matches_baseline_at_init_and_parallel_after_update(self):
        model = tiny_model(readiness_readout="short_long").eval()
        loaded = model.load_state_dict(self.model.state_dict(), strict=False)
        self.assertEqual(loaded.missing_keys, ["core.readiness_head.evidence_weights"])
        self.assertFalse(loaded.unexpected_keys)
        rows, features = fixture()
        baseline = forward_batch(self.model, rows, features)
        result = forward_batch(model, rows, features)
        torch.testing.assert_close(result["logits"], baseline["logits"])
        weights = model.core.readiness_head.evidence_weights
        model._readiness_surrogate(result["readiness_loss"]).backward()
        self.assertTrue((weights.grad.norm(dim=1) > 0).all())
        self.assertIsNone(model.core.long_query_tokens.grad)
        with torch.no_grad():
            weights.add_(weights.grad, alpha=-0.1)
        serial = forward_batch(model, rows, features)
        self.assertFalse(torch.allclose(serial["logits"], baseline["logits"]))
        model.parallel_endpoints = True
        parallel = forward_batch(model, rows, features)
        torch.testing.assert_close(parallel["logits"], serial["logits"], atol=2e-6, rtol=2e-5)
        for expected, actual in zip(baseline["outputs"], parallel["outputs"]):
            torch.testing.assert_close(actual["prefix_embeds"], expected["prefix_embeds"], atol=2e-6, rtol=2e-5)

    def test_no_future_or_fractional_endpoint_contamination(self):
        rows, features = fixture()
        self.model.parallel_endpoints = True
        original = forward_batch(self.model, rows, features)
        altered = [x.clone() for x in features]
        altered[3][:, -1] += 100
        changed = forward_batch(self.model, rows, altered)
        for i in [0, 1, 2, 4, 5]:
            torch.testing.assert_close(original['outputs'][i]['prefix_embeds'], changed['outputs'][i]['prefix_embeds'])
        isolated = forward_batch(self.model, rows[3:4], features[3:4])
        torch.testing.assert_close(original['outputs'][3]['prefix_embeds'], isolated['outputs'][0]['prefix_embeds'], atol=2e-6, rtol=2e-5)

    def test_no_stale_projected_memory_after_parameter_change(self):
        rows, features = fixture()
        self.model.parallel_endpoints = True
        forward_batch(self.model, rows, features)
        with torch.no_grad():
            self.model.core.vision_proj.weight.add_(torch.randn_like(self.model.core.vision_proj.weight) * .1)
        parallel = forward_batch(self.model, rows, features)
        self.model.parallel_endpoints = False
        serial = forward_batch(self.model, rows, features)
        torch.testing.assert_close(parallel['logits'], serial['logits'], atol=2e-6, rtol=2e-5)

    def test_sdpa_matches_eager_values_and_gradients(self):
        attention = self.model.core.short_qformer.bert.encoder.layer[0].crossattention.self
        query, memory = torch.randn(2, 3, 32, requires_grad=True), torch.randn(2, 5, 24, requires_grad=True)
        mask = torch.tensor([[[[0., 0., 0., -10000., -10000.]]], [[[0., 0., 0., 0., 0.]]]])
        attention.use_sdpa = False
        eager = attention(query, encoder_hidden_states=memory, encoder_attention_mask=mask)[0]
        grad = torch.autograd.grad(eager.square().sum(), [query, memory])
        attention.use_sdpa = True
        fast = attention(query, encoder_hidden_states=memory, encoder_attention_mask=mask)[0]
        torch.testing.assert_close(eager, fast)
        for a, b in zip(grad, torch.autograd.grad(fast.square().sum(), [query, memory])):
            torch.testing.assert_close(a, b, atol=2e-6, rtol=2e-5)


if __name__ == '__main__':
    unittest.main()
