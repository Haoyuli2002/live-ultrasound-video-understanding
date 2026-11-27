from contextlib import ExitStack, closing
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import unittest
from unittest.mock import Mock

import numpy as np
import torch

from .index import EventIndex
from .parallel_replay import VideoClock, replay_group
from .test_answer import TinyTokenizer, tiny_backbone
from .answer import AnswerDecoder, DecisionContext


class ParallelReplayTests(unittest.TestCase):
    def test_mixed_prompt_lengths_match_single_decode(self):
        torch.manual_seed(7)
        bb = tiny_backbone().eval()
        bb.tokenizer = TinyTokenizer()
        decoder = AnswerDecoder(bb, legacy_tags=True)
        ctx = DecisionContext("Which shape?", 0, 1, 0, 1)
        visual = torch.randn(9, bb.llm_width)
        items = [(visual, torch.arange(9) < k, ctx) for k in (8, 2, 9)]
        self.assertEqual(decoder.generate_batch(items, 10),
                         [decoder.generate(*x, 10) for x in items])

    def test_cold_and_warm_video_clocks(self):
        with TemporaryDirectory() as tmp, ExitStack() as stack:
            bb = Mock()
            bb.encode_frames.side_effect = lambda pixels: torch.ones(len(pixels), 1, 2, 3)
            class Towers(torch.nn.Module):
                recent_window = 24

                def __init__(self):
                    super().__init__()
                    self.visual_projection = torch.nn.Linear(3, 3)
                def encode_short(self, x, valid):
                    return x[:, 0], valid[:, :1].expand(-1, 2)
                def encode_long(self, x, valid):
                    return x.new_zeros(1, 1, 3), torch.zeros(1, 1, dtype=torch.bool)
            videos = []
            for i in range(2):
                vid = str(i)
                idx = stack.enter_context(closing(EventIndex(Path(tmp) / vid, vid, 'test')))
                v = SimpleNamespace(vid=vid, sessions={}, index=idx, writable=lambda idx=idx: idx,
                                    pts=[0, 1, 2], reader=np.zeros((3, 8, 8, 3), dtype=np.uint8))
                v.clock = VideoClock([dict(task_id=vid, question=vid, query_time=i + .5)], 2.5)
                videos.append(v)
            def predict(items, budget):
                return [dict(action='ANSWER' if ctx.question == '0' and ctx.current_time == 1 else 'WAIT')
                        for _, _, ctx in items]
            decoder = Mock()
            decoder.generate_batch.side_effect = predict
            def run():
                rows = []
                replay_group(videos, bb, Towers().eval(), decoder, Mock(), torch.device('cpu'),
                             False, 10, 64, 8, rows.append)
                return [(r['task_id'], r['current_time'], r['recent_window_s'][-1]) for r in rows]
            cold = run()
            self.assertEqual(cold, [('0', .5, 0), ('1', 1.5, 1), ('0', 1, 1),
                                    ('1', 2, 2), ('1', 2.5, 2)])
            self.assertEqual(bb.encode_frames.call_args_list[0].args[0].shape[0], 2)
            calls = bb.encode_frames.call_count
            for i, v in enumerate(videos):
                v.clock = VideoClock([dict(task_id=v.vid, question=v.vid, query_time=i + .5)], 2.5)
            self.assertEqual(run(), cold)
            self.assertEqual(bb.encode_frames.call_count, calls)
