import unittest
from types import SimpleNamespace
from unittest.mock import Mock, patch

import torch
from lavis.models.blip2_models.Qformer import BertConfig
from StreamReady.tests.test_qwen35_backbone import tiny_backbone

from .answer import ActionFormat, AnswerDecoder, DecisionContext, StreamingAnswerSession
from .qformer import EventQFormers
from .retrieve import LongEvidence


class TinyTokenizer:
    eos_token_id = 2
    pad_token_id = 0
    all_special_ids = [0, 1, 2]
    tags = {"<WAIT>": [10, 11, 12], "<ANSWER>": [10, 13, 14, 12]}

    def encode(self, text, add_special_tokens=False):
        return self.tags.get(text, [3, 4])

    def decode(self, ids, skip_special_tokens=True):
        return " ".join("word" for i in ids if i not in self.all_special_ids)

    def apply_chat_template(self, messages, **kwargs):
        return messages[0]['content'] + messages[1]['content']


class AnswerTests(unittest.TestCase):
    def test_both_actions_follow_logits(self):
        fmt = ActionFormat(TinyTokenizer())
        for action, ids in fmt.tags.items():
            seq = []
            for token in ids:
                logits = torch.zeros(1, 64)
                logits[0, 63] = 10
                logits[0, token] = 2
                seq.append(fmt(torch.tensor([seq]), logits).argmax().item())
            self.assertEqual(seq, ids)
            body_scores = fmt(torch.tensor([seq]), torch.zeros(1, 64))
            self.assertFalse(torch.isfinite(body_scores[0, 2]))
            self.assertTrue(torch.isfinite(body_scores[0, 3]))
            self.assertEqual(fmt.parse(seq + [3, 2]), (action, 'word'))
            self.assertTrue(torch.isfinite(fmt(torch.tensor([seq + [3]]), torch.zeros(1, 64))[0, 2]))

    def test_native_generation_and_alignment(self):
        torch.manual_seed(7)
        bb = tiny_backbone()
        bb.tokenizer = TinyTokenizer()
        dec = AnswerDecoder(bb, legacy_tags=True)
        cfg = BertConfig(vocab_size=16, hidden_size=24, num_hidden_layers=2, num_attention_heads=4,
                         intermediate_size=32, encoder_width=12, add_cross_attention=True, cross_attention_freq=2)
        model = EventQFormers(cfg, vision_width=16, llm_width=bb.llm_width)
        recent = torch.randn(2, 2, 2, 16)
        events = torch.randn(2, 1, 1, 2, 16)
        visual, mask = model(recent, events, event_mask=torch.tensor([[[True]], [[False]]]))
        ctx = DecisionContext("Which shape?", 23, 25, 2, 25)
        prompt = dec._prompt(visual[0], mask[0], ctx)
        torch.testing.assert_close(prompt[2:-2], visual[0])
        self.assertEqual(len(dec._prompt(visual[1], mask[1], ctx)), 36)
        with patch.object(bb, "lm_forward", wraps=bb.lm_forward) as forward:
            loss = dec.alignment_loss(visual, mask, [ctx, ctx], ["<ANSWER> round", "<WAIT>"])
        labels = forward.call_args.kwargs["labels"]
        self.assertEqual(labels[0][labels[0] != -100].tolist(), [10, 13, 14, 12, 3, 4, 2])
        self.assertEqual(labels[1][labels[1] != -100].tolist(), [10, 11, 12])
        self.assertTrue(torch.isfinite(loss))
        loss.backward()
        for branch in (model.short, model.long):
            self.assertGreater(branch.output_projection.weight.grad.norm().item(), 0)
            self.assertGreater(branch.queries.grad.norm().item(), 0)
        self.assertTrue(all(p.grad is None for p in bb.parameters()))
        result = dec.generate(visual[0], mask[0], ctx, 8)
        self.assertIn(result['action'], ('WAIT', 'ANSWER'))
        self.assertTrue(result['reason'] or result['answer'])

    def test_wait_next_tick_then_finish(self):
        frames = []
        def read_frame(t):
            frames.append(t)
            return t, torch.zeros(1, 4)

        model = Mock(recent_window=24)
        model.visual_projection.weight = torch.empty(1)
        model.encode_short.return_value = torch.ones(1, 32, 4), torch.ones(1, 32, dtype=torch.bool)
        idx = SimpleNamespace(video_id='V', detector=SimpleNamespace(next_second=100), frame=read_frame)
        retrieval = Mock(index=idx, qformers=model, question="Which shape?")
        retrieval.at.return_value = LongEvidence(torch.zeros(1, 128, 4), torch.zeros(1, 128, dtype=torch.bool), [])
        decoder = Mock()
        decoder.generate.side_effect = [dict(action="WAIT", reason="Need another view"), dict(action="ANSWER", answer="Round")]
        pending = StreamingAnswerSession(retrieval, decoder, 24)
        first = pending.step(24)
        self.assertEqual(first['recent_window_s'], [1, 24])
        self.assertIs(first, pending.step(24))
        self.assertEqual(pending.step(25)['action'], 'ANSWER')
        self.assertEqual(frames, list(range(1, 25)) + list(range(2, 26)))
        self.assertEqual(decoder.generate.call_count, 2)
        with self.assertRaises(ValueError):
            pending.step(26)
