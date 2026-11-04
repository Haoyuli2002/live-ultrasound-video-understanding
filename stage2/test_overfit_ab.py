"""CPU gradient/chronology checks for separately supervised memory targets."""
import contextlib
import io
import json
from types import SimpleNamespace
import unittest

import torch

from .evaluate import evaluate_video
from .reconstruct import collect_memories
from .train_overfit_ab import train_video


def blocks():
    return [{"block_window": [a, a+60],
             "local_sub_summaries": [{"local_summary_target": "1"} for _ in range(6)],
             "block_summary_target": "1", "global_summary_target": "1"}
            for a in (0, 60)]


class Memory:
    def __init__(self):
        self.weight = torch.nn.Parameter(torch.tensor(0.8))
        self.frames = []
        self.grad_frames = []
        self.long_calls = []
        self.losses = []

    def encode_frame(self, frame):
        self.frames.append((frame, float(self.weight.detach()), torch.is_grad_enabled()))
        result = (self.weight*(frame+1)/120).reshape(1, 1)
        if result.requires_grad:
            result.register_hook(lambda grad: self.grad_frames.append((frame, float(grad.abs().sum()))))
        return result

    def update_long(self, previous, short):
        self.long_calls.append((None if previous is None else previous.detach().clone(),
                                float(self.weight.detach()), torch.is_grad_enabled()))
        return short if previous is None else short+previous.detach()

    def reconstruction_loss(self, tokens, target, kind):
        self.losses.append((kind, float(self.weight.detach()), tokens.detach().clone()))
        return (tokens.mean()-float(target)).square()


class IndependentTargetTests(unittest.TestCase):
    def run_train(self, mode):
        memory = Memory()
        stream = io.StringIO()
        reader = SimpleNamespace(frame_ending_at=lambda t: t)
        optimizer = torch.optim.SGD([memory.weight], lr=0.01)
        with contextlib.redirect_stdout(io.StringIO()):
            count = train_video(memory, blocks(), reader, optimizer, stream, "P3", 0, mode)
        return memory, count, [json.loads(s) for s in stream.getvalue().splitlines()]

    def test_short_only_has_12_fresh_updates_without_long_or_other_losses(self):
        memory, count, metrics = self.run_train("short_only")
        self.assertEqual(count, 12)
        self.assertEqual([r[0] for r in memory.losses], ["short"]*12)
        self.assertEqual([r[0] for r in memory.frames], list(range(120)))
        self.assertFalse(memory.long_calls)
        self.assertEqual(len(memory.grad_frames), 120)
        self.assertTrue(all(g > 0 for _, g in memory.grad_frames))
        self.assertTrue(all(r["fresh_memory"] for r in metrics))

    def test_independent_steps_reencode_all_current_frames_at_updated_weights(self):
        memory, count, metrics = self.run_train("independent")
        self.assertEqual(count, 16)
        self.assertEqual([r[0] for r in memory.losses], (["short"]*6+["block", "global"])*2)
        expected_frames = list(range(60))*3+list(range(60,120))*2+list(range(120))
        self.assertEqual([r[0] for r in memory.frames], expected_frames)
        # Every step uses new parameters; Block/Global do not reuse old Short tensors.
        versions = [r[1] for r in memory.losses]
        self.assertTrue(all(a != b for a,b in zip(versions,versions[1:])))
        offset = 0
        for row, (_, version, _) in zip(metrics, memory.losses):
            n = 10 if row["kind"] == "short" else 60
            if row["kind"] == "global":
                n += 60*row["history_replayed_blocks"]
            self.assertTrue(all(v == version for _,v,_ in memory.frames[offset:offset+n]))
            offset += n
        self.assertEqual(offset, 420)
        self.assertEqual(len(memory.grad_frames), 360)
        self.assertTrue(all(g > 0 for _,g in memory.grad_frames))
        self.assertEqual(sum(not grad for _,_,grad in memory.frames), 60)
        # First Global: no history. Second Global replays first minute freshly,
        # then uses that detached state for current-minute differentiable update.
        self.assertIsNone(memory.long_calls[0][0])
        self.assertFalse(memory.long_calls[1][2])
        self.assertTrue(memory.long_calls[2][2])
        self.assertEqual(memory.long_calls[1][1], memory.long_calls[2][1])
        self.assertTrue(torch.equal(memory.long_calls[2][0],
                                    torch.arange(1,61).reshape(1,60,1)*memory.long_calls[2][1]/120))

    def test_short_only_evaluation_never_updates_untrained_long_memory(self):
        reader = SimpleNamespace(frame_ending_at=lambda t: t)
        memory = Memory()
        records = collect_memories(memory, blocks(), reader, ["short"])
        self.assertEqual(len(records), 12)
        self.assertEqual({r["kind"] for r in records}, {"short"})
        self.assertFalse(memory.long_calls)
        rows = evaluate_video(memory, blocks(), reader, ["short"])
        self.assertEqual(len(rows), 12)
        self.assertFalse(memory.long_calls)


if __name__ == "__main__":
    unittest.main()
