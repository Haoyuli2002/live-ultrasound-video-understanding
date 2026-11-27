import tempfile
import unittest
from pathlib import Path

import torch

from .index import Event, EventIndex
from .retrieve import QuestionSession, QwenRelevance
from .segment import segment


class CaptureLong(torch.nn.Module):
    recent_window = 24

    def __init__(self):
        super().__init__()
        self.visual_projection = torch.nn.Linear(2, 2)
        self.calls = []

    def encode_long(self, events, mask):
        self.calls.append((events.clone(), mask.clone()))
        return torch.zeros(1, 128, 2), torch.full((1, 128), bool(mask.any()))


class StreamTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.path = Path(self.tmp.name)/'frames.sqlite'
        self.open()

    def open(self):
        self.index = EventIndex(self.path, 'v', 'encoder')
        self.addCleanup(self.index.close)

    def push(self, second, value):
        # Vary the patches while keeping the desired frame mean.
        offsets = torch.tensor([[.02, 0.], [-.02, 0.]])
        self.index.push(second, max(0., second - .05), torch.tensor(value) + offsets)

    def test_candidate_resume_and_eof(self):
        for t in range(3):
            self.push(t, [1., 0.])
        self.push(3, [0., 1.])
        self.assertEqual(self.index.events_at(3), [])
        self.index.close()
        self.open()
        self.push(4, [0., 1.])
        self.assertEqual(self.index.events_at(3), [])
        a, = self.index.events_at(4)
        self.assertEqual((a.start, a.end, a.available_at, a.representative_s), (0, 3, 4, 0))
        self.index.finish(4.25)
        self.assertEqual(len(self.index.events_at(4)), 1)
        self.assertEqual(self.index.events_at(4.25)[1].end, 4.25)
        self.index.close()
        self.open()
        self.index.finish(4.25)
        with self.assertRaises(ValueError):
            self.push(5, [0., 1.])
        with self.assertRaises(ValueError):
            EventIndex(self.path, 'different-video', 'encoder')

    def test_rejected_candidate_and_invalid_frame_leave_no_cut(self):
        vectors = [[1., 0.]] * 3 + [[.8, .6], [1., 0.], [1., 0.]]
        self.assertEqual(segment(torch.tensor(vectors).numpy())[0], [(0, 0)])
        for t, v in enumerate(vectors):
            self.push(t, v)
        with self.assertRaises(ValueError):
            self.push(6, [float('nan'), 1.])
        with self.assertRaises(ValueError):
            self.index.push(6, 6.1, torch.zeros(2, 2))
        self.assertEqual(self.index.observed_until, 5)
        self.push(6, [1., 0.])
        self.assertEqual(self.index.events_at(6), [])

    def test_unconfirmed_tail_contributes_to_final_center(self):
        import numpy as np
        for t, v in enumerate([[1., 0.]] * 3 + [[0., 1.]]):
            self.push(t, v)
        self.assertFalse(self.index.events_at(3))
        self.index.finish(3.4)
        event, = self.index.events_at(3.4)
        self.assertEqual((event.start, event.end, event.available_at), (0, 3.4, 3.4))
        center, = self.index.db.execute("SELECT centroid FROM events").fetchone()
        np.testing.assert_allclose(np.frombuffer(center, dtype=np.float32), np.array([3.,1.])/np.sqrt(10))

    def test_window_slides_without_new_event(self):
        for t in range(25):
            self.push(t, [1., 0.] if t < 3 else [0., 1.])
        scored = []
        def scorer(question, images, events):
            scored.extend(event.number for event in events)
            return torch.ones(len(events))
        session = QuestionSession(self.index, 'history', scorer, CaptureLong().eval())
        self.assertEqual(len(self.index.events_at(23)), 1)
        self.assertFalse(session.at(23).mask.any())  # short = 0..23, E01 = [0,3)
        self.assertEqual(scored, [])
        self.assertEqual(session.at(24).sources[0]['sample_seconds'], [0, 1, 2])
        self.assertEqual(scored, [1])
        self.assertEqual(session.at(23).sources, [])  # rewind after scoring

    def test_integer_eof_does_not_advance_window(self):
        for t in range(24):
            self.push(t, [1., 0.] if t < 3 else [0., 1.])
        self.index.finish(24.)
        self.assertEqual(len(self.index.events_at(24.)), 2)
        self.assertEqual(self.index.events_at(24., exclude_recent_frames=24), [])

    def test_top5_materialization_scores_cache_and_past_queries(self):
        calls = []
        def scorer(question, images, events):
            calls.extend(e.number for e in events)
            self.assertEqual(len(images), len(events))
            # Rank earlier events first; subsequent closed events won't change top 5.
            direction = -1 if question == 'earliest' else 1
            return torch.tensor([direction * e.number for e in events])
        model = CaptureLong().eval()
        session = QuestionSession(self.index, 'earliest', scorer, model)
        for t in range(69):
            k = 0 if t < 27 else 1 + (t - 27)//3
            self.push(t, [float(k % 2 == 0), float(k % 2 == 1)])
        empty = session.at(0)
        self.assertFalse(empty.mask.any())
        hit = session.at(60)
        self.assertEqual([s['event_id'] for s in hit.sources], ['E01','E02','E03','E04','E05'])
        self.assertEqual(len(calls), 5)
        self.assertIs(session.at(61), hit)
        self.assertIs(session.at(67), hit)
        self.assertEqual(len(calls), len(set(calls)))
        self.assertEqual(len(model.calls), 2)
        features, mask = model.calls[-1]
        for i, src in enumerate(hit.sources):
            samples = src['sample_seconds']
            self.assertEqual(src['slot'], i)
            self.assertEqual(samples[0], src['start_s'])
            self.assertEqual(samples[-1], src['end_s'] - 1)
            self.assertEqual(len(samples), min(24, int(src['end_s'] - src['start_s'])))
            self.assertLessEqual(src['available_at_s'], 60)
            self.assertEqual(mask[0, i].sum().item(), len(samples))
            for j, t in enumerate(samples):
                pts, frame = self.index.frame(t)
                self.assertEqual(pts, src['source_pts_s'][j])
                torch.testing.assert_close(features[0, i, j], frame)
        # Asking about the past must exclude events already scored at later times.
        self.assertEqual(len(session.at(28).sources), 1)
        other = QuestionSession(self.index, 'latest', scorer, model)
        self.assertGreater(other.at(67).sources[0]['available_at_s'], hit.sources[0]['available_at_s'])


class RelevanceTests(unittest.TestCase):
    def test_cached_patch_scoring_matches_native_qwen_image_forward(self):
        from StreamReady.tests.test_qwen35_backbone import tiny_backbone
        from unittest.mock import Mock
        torch.manual_seed(10)
        backbone = tiny_backbone().eval()
        # Stable mini vocabulary for the real tiny vision encoder and causal LM.
        tok = Mock(pad_token_id=0)
        tok.apply_chat_template.side_effect = lambda messages, **kw: messages[0]['content'][1]['text']
        def ids(text, **kw):
            if text in ('Yes', 'No'):
                return [5 if text == 'Yes' else 6]
            return [62, 60, 63, 7] + ([8] if '[20,' in text else []) + [9]
        tok.encode.side_effect = ids
        backbone.tokenizer = tok
        scorer = QwenRelevance(backbone)
        raw = [torch.rand(3, 64, 64) for _ in range(2)]
        events = [Event(1, 0, 3, 4, 1), Event(2, 20, 25, 26, 21)]
        features, expected = [], []
        with torch.inference_mode():
            for i, rgb in enumerate(raw):
                image = backbone.image_processor(images=[rgb], do_resize=False, do_rescale=False,
                                                 return_tensors='pt')
                visual = backbone.model.get_image_features(pixel_values=image.pixel_values,
                                                           image_grid_thw=image.image_grid_thw)
                patches = visual.pooler_output[0]
                features.append(patches)
                tokens = torch.tensor([[62] + [60]*len(patches) + [63,7] + ([8] if i else []) + [9]])
                logits = backbone.model(input_ids=tokens, attention_mask=torch.ones_like(tokens),
                                        pixel_values=image.pixel_values, image_grid_thw=image.image_grid_thw,
                                        mm_token_type_ids=(tokens == 60).int(), use_cache=False).logits
                expected.append(logits[0,-1,5] - logits[0,-1,6])
            actual = scorer('Where is the dark region?', torch.stack(features), events)
        torch.testing.assert_close(actual, torch.stack(expected), atol=1e-5, rtol=1e-4)


if __name__ == '__main__':
    unittest.main()
