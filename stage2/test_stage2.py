import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import torch

from stage2.data import load_blocks
from stage2.build_data import build_files, build_rows
from stage2.annotate import annotate_video, ChatTeacher
from stage2.train import train_video


def block(start=0, seconds=60):
    return {"video_id": "v1", "block_window": [start, start + seconds],
            "local_sub_summaries": [
                {"window": [start + 10*i, start + 10*(i+1)],
                 "local_summary_target": f"ten {i}"}
                for i in range(seconds // 10)],
            "block_summary_target": "one minute" if seconds == 60 else "",
            "global_summary_target": "video so far" if seconds == 60 else ""}


class FakeReader:
    def frame_ending_at(self, second):
        return second


class FakeTeacher:
    def __init__(self):
        self.calls = []

    def summary(self, prompt, video_path):
        self.calls.append((prompt, video_path.name))
        return f"visual summary {len(self.calls)}"


class FakeMemory:
    def __init__(self):
        self.model = torch.nn.Linear(1, 1, bias=False)
        self.calls = []

    def encode_frame(self, frame):
        self.calls.append(frame)
        return self.model(torch.ones(1, 1))

    def reconstruction_loss(self, memory, target, kind):
        return memory.square().mean()

    def update_long(self, previous, short):
        self.previous = previous
        self.previous_seen.append(previous is not None)
        return short + (0 if previous is None else previous)


class Stage2Tests(unittest.TestCase):
    def test_teacher_sends_video_not_image_frames(self):
        calls = []
        def create(**kwargs):
            calls.append(kwargs)
            return SimpleNamespace(choices=[SimpleNamespace(
                message=SimpleNamespace(content="summary"))])
        with tempfile.TemporaryDirectory() as directory:
            clip = Path(directory) / "clip.mp4"
            clip.write_bytes(b"video bytes")
            teacher = ChatTeacher.__new__(ChatTeacher)
            teacher.client = SimpleNamespace(chat=SimpleNamespace(
                completions=SimpleNamespace(create=create)))
            teacher.model = "google/gemini-3.1-pro-preview"
            teacher.max_tokens = 100
            teacher.transport = "file"
            teacher.video_fps = 0
            self.assertEqual(teacher.summary("describe", clip), "summary")
            video = calls[-1]["messages"][1]["content"][0]
            self.assertEqual(video["type"], "file")
            self.assertTrue(video["file"]["file_data"].startswith("data:video/mp4;base64,"))
            teacher.transport = "video_url"
            teacher.video_fps = 0
            teacher.summary("describe", clip)
            video = calls[-1]["messages"][1]["content"][0]
            self.assertEqual(video["type"], "video_url")
            self.assertEqual(video["video_url"]["url"], clip.resolve().as_uri())
            self.assertNotIn("extra_body", calls[-1])

    def test_teacher_builds_independent_local_block_and_global_labels(self):
        teacher = FakeTeacher()
        with patch("stage2.annotate.make_silent_clip") as make_clip:
            batches = list(annotate_video(Path("video.mp4"), teacher,
                                          video_id="v1", duration_sec=130))
        self.assertEqual([len(batch) for batch in batches], [8, 8, 1])
        self.assertEqual([item["window"] for item in batches[0][:6]],
                         [[0, 10], [10, 20], [20, 30], [30, 40], [40, 50], [50, 60]])
        self.assertEqual(batches[0][6]["window"], [0, 60])
        self.assertEqual(batches[0][7]["window"], [0, 60])
        self.assertEqual(batches[1][6]["window"], [60, 120])
        self.assertEqual(batches[1][7]["window"], [0, 120])
        self.assertEqual(batches[2][0]["window"], [120, 130])
        self.assertEqual(len(teacher.calls), 17)
        self.assertEqual(make_clip.call_count, 17)
        self.assertEqual(make_clip.call_args_list[7].args[1:3], (0, 60))
        self.assertEqual(make_clip.call_args_list[15].args[1:3], (0, 120))
        self.assertEqual(len(build_rows([item for batch in batches for item in batch])), 3)
        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory) / "events.jsonl"
            output = Path(directory) / "blocks.jsonl"
            source.write_text("".join(json.dumps(item) + "\n"
                                      for batch in batches for item in batch))
            self.assertEqual(build_files([source], output), 3)
            blocks = load_blocks(output)["v1"]
            self.assertEqual(blocks[1]["block_window"], [60.0, 120.0])
            self.assertEqual(blocks[1]["global_summary_target"], batches[1][7]["target"])

    def test_builder_accepts_independent_block_labels(self):
        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory) / "teacher.jsonl"
            output = Path(directory) / "blocks.jsonl"
            source.write_text(json.dumps({**block(),
                                          "block_summary_window": [0, 60],
                                          "global_window": [0, 60]}) + "\n")
            self.assertEqual(build_files([source], output), 1)
            self.assertEqual(len(load_blocks(output)["v1"][0]["local_sub_summaries"]), 6)

    def test_builder_accepts_event_labels_and_rejects_legacy(self):
        events = []
        for j in range(6):
            events.append({"video_id": "v1", "block_window": [0, 60],
                           "label_type": "local_10s", "window": [10*j, 10*(j+1)],
                           "target": f"local {j}"})
        events.extend([
            {"video_id": "v1", "block_window": [0, 60],
             "label_type": "block_60s", "window": [0, 60], "target": "block"},
            {"video_id": "v1", "block_window": [0, 60],
             "label_type": "global_prefix", "window": [0, 60], "target": "global"},
        ])
        self.assertEqual(build_rows(events)[0]["global_summary_target"], "global")
        with self.assertRaisesRegex(ValueError, "legacy 60-second"):
            build_rows([{**block(), "local_sub_summaries": None,
                         "local_summary_target": "old block"}])

    def test_label_validation_and_six_updates(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "labels.jsonl"
            path.write_text(json.dumps(block()) + "\n" +
                            json.dumps(block(60, 10)) + "\n")
            blocks = load_blocks(path)["v1"]
            memory = FakeMemory()
            memory.previous_seen = []
            optimizer = torch.optim.SGD(memory.model.parameters(), lr=0.001)
            metrics = Path(directory) / "metrics.jsonl"
            with metrics.open("w") as stream:
                train_video(memory, blocks, FakeReader(), optimizer, stream, "v1", 0)
            records = [json.loads(line) for line in metrics.read_text().splitlines()]
            self.assertEqual(len(records), 7)
            self.assertEqual(sum(record["global"] is not None for record in records), 1)
            self.assertEqual(memory.calls, list(range(70)))
            self.assertIsNone(memory.previous)
            self.assertEqual(memory.previous_seen, [False])

    def test_long_state_recurs_and_is_detached(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "labels.jsonl"
            path.write_text(json.dumps(block()) + "\n" +
                            json.dumps(block(60)) + "\n")
            memory = FakeMemory()
            memory.previous_seen = []
            optimizer = torch.optim.SGD(memory.model.parameters(), lr=0.001)
            with (Path(directory) / "metrics.jsonl").open("w") as metrics:
                train_video(memory, load_blocks(path)["v1"], FakeReader(),
                            optimizer, metrics, "v1", 0)
            self.assertEqual(memory.previous_seen, [False, True])
            self.assertFalse(memory.previous.requires_grad)
            self.assertEqual(memory.calls, list(range(120)))

    def test_gap_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "labels.jsonl"
            path.write_text(json.dumps(block(10)) + "\n")
            with self.assertRaisesRegex(ValueError, "gap"):
                load_blocks(path)


if __name__ == "__main__":
    unittest.main()
