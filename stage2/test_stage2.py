import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import torch

from stage2.data import load_blocks
from stage2.build_data import build_files, build_rows
from stage2.annotate import annotate_video, ChatTeacher, summarize_window
from stage2.summary_schema import parse_summary
from stage2.prepare_ddp_smoke import prepare as prepare_ddp_smoke, select_videos
from stage2.train import (train_video, evaluate_video, _balanced_video_shards,
                          _validate_resume_config)


LOCAL = "View: lung view; Anatomy: pleural line; Finding: sliding visible"
BLOCK = "View: lung view; Anatomy: pleura; Finding: sliding remains visible"
GLOBAL = "Anatomy: pleura; Finding: sliding visible; Change: stable over time"


def block(start=0, seconds=60):
    return {"video_id": "v1", "block_window": [start, start + seconds],
            "local_sub_summaries": [
                {"window": [start + 10*i, start + 10*(i+1)],
                 "local_summary_target": LOCAL}
                for i in range(seconds // 10)],
            "block_summary_target": BLOCK if seconds == 60 else "",
            "global_summary_target": GLOBAL if seconds == 60 else ""}


class FakeReader:
    def frame_ending_at(self, second):
        return second


class FakeTeacher:
    def __init__(self):
        self.calls = []

    def summary(self, prompt, video_path):
        self.calls.append((prompt, video_path.name))
        if "cumulative video prefix" in prompt:
            return "The examination progresses from patient positioning to probe handling."
        if "60-second block" in prompt:
            return "The patient lies supine while the clinician demonstrates the probe."
        return "The clinician holds the probe beside the supine patient."


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
    def test_ddp_smoke_preparation_uses_two_videos_and_valid_blocks(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            video_map = root / "videos.json"
            video_map.write_text(json.dumps({
                "minute": str(root / "minute.mp4"),
                "short": str(root / "short.mp4"),
            }))
            with patch("stage2.prepare_ddp_smoke.video_duration",
                       side_effect=lambda path: 65.0 if path.name == "minute.mp4" else 15.0):
                manifest = prepare_ddp_smoke(
                    video_map, root / "smoke", ["minute", "short"])
            self.assertEqual(manifest["expected_updates_per_rank"], [6, 1])
            blocks = load_blocks(manifest["train_jsonl"])
            self.assertEqual([len(blocks[k][0]["local_sub_summaries"])
                              for k in ("minute", "short")], [6, 1])
            self.assertEqual(blocks["minute"][0]["block_window"], [0.0, 60.0])
            self.assertEqual(blocks["short"][0]["block_window"], [0.0, 10.0])
            with patch("stage2.prepare_ddp_smoke.video_duration",
                       return_value=65.0):
                with self.assertRaises(FileExistsError):
                    prepare_ddp_smoke(video_map, root / "smoke", ["minute", "short"])
        with self.assertRaisesRegex(ValueError, "at least 60s"):
            select_videos({"a": "a.mp4", "b": "b.mp4"},
                          {"a": 30, "b": 15}, ["a", "b"])

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
            self.assertEqual(calls[-1]["extra_body"]["mm_processor_kwargs"],
                             {"do_sample_frames": False})

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
                           "target": LOCAL})
        events.extend([
            {"video_id": "v1", "block_window": [0, 60],
             "label_type": "block_60s", "window": [0, 60], "target": BLOCK},
            {"video_id": "v1", "block_window": [0, 60],
             "label_type": "global_prefix", "window": [0, 60], "target": GLOBAL},
        ])
        self.assertEqual(build_rows(events)[0]["global_summary_target"], GLOBAL)
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
                final_step = train_video(
                    memory, blocks, FakeReader(), optimizer, stream, "v1", 1,
                    global_step=10)
            records = [json.loads(line) for line in metrics.read_text().splitlines()]
            self.assertEqual(len(records), 7)
            self.assertEqual(final_step, 17)
            self.assertEqual([record["global_step"] for record in records],
                             list(range(11, 18)))
            self.assertTrue(all(record["epoch"] == 1 for record in records))
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

    def test_evaluate_video_reports_component_means_without_updates(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "labels.jsonl"
            path.write_text(json.dumps(block()) + "\n" +
                            json.dumps(block(60)) + "\n")
            blocks = load_blocks(path)["v1"]
            memory = FakeMemory()
            memory.previous_seen = []
            before = [p.clone() for p in memory.model.parameters()]
            means, counts = evaluate_video(memory, blocks, FakeReader())
            after = list(memory.model.parameters())
            # Two full minutes of 10s intervals; one long update at each minute end.
            self.assertEqual(counts, {"short": 12, "block": 2, "global": 2})
            for key in ("short", "block", "global"):
                self.assertIsNotNone(means[key])
            # No optimizer ran, so parameters are unchanged.
            for old, new in zip(before, after):
                self.assertTrue(torch.equal(old, new))
            # The recurrence still ran: long memory was reused on the second minute.
            self.assertEqual(memory.previous_seen, [False, True])

    def test_gap_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "labels.jsonl"
            path.write_text(json.dumps(block(10)) + "\n")
            with self.assertRaisesRegex(ValueError, "gap"):
                load_blocks(path)

    def test_structured_summary_schema(self):
        self.assertEqual(list(parse_summary(LOCAL, "local")),
                         ["View", "Anatomy", "Finding"])
        with self.assertRaisesRegex(ValueError, "duplicate facet"):
            parse_summary("View: one; View: two", "local")
        with self.assertRaisesRegex(ValueError, "not allowed"):
            parse_summary("Change: changed", "local")
        with self.assertRaisesRegex(ValueError, "out of order"):
            parse_summary("Finding: one; Anatomy: two", "block")

    def test_invalid_teacher_summary_is_retried(self):
        teacher = SimpleNamespace(calls=0)
        def summary(prompt, video_path):
            teacher.calls += 1
            return "View: chest" if teacher.calls == 1 else "The clinician holds a probe beside the patient."
        teacher.summary = summary
        with patch("stage2.annotate.make_silent_clip"):
            target, facets = summarize_window(
                Path("video.mp4"), teacher, kind="local", start=0, end=10,
                clip_path=Path("clip.mp4"), max_attempts=2)
        self.assertEqual(teacher.calls, 2)
        self.assertEqual(target, "The clinician holds a probe beside the patient.")
        self.assertEqual(facets, {})

    def test_builder_rejects_unstructured_summary(self):
        bad = block()
        bad["local_sub_summaries"][0]["local_summary_target"] = "plain paragraph"
        with self.assertRaisesRegex(ValueError, "unlabelled clause"):
            build_rows([bad])

    def test_resume_config_rejects_changed_experiment(self):
        keys = ("model_name", "init_adapter", "train_jsonl", "video_path_map",
                "frame_size", "lora_rank", "dtype", "learning_rate",
                "max_grad_norm", "seed", "validation_fraction", "warmup_ratio",
                "epochs", "world_size")
        saved = {key: "same" for key in keys}
        current = dict(saved)
        current["lora_rank"] = "changed"
        with self.assertRaisesRegex(ValueError, "lora_rank"):
            _validate_resume_config(saved, current)

    def test_ddp_video_shards_are_balanced_and_complete(self):
        videos = {
            "long": [block(), block(60)],
            "medium": [block()],
            "short": [block(0, 20)],
            "tiny": [block(0, 10)],
        }
        shards, steps = _balanced_video_shards(videos, sorted(videos), 3)
        self.assertEqual(sorted(video for shard in shards for video in shard),
                         sorted(videos))
        self.assertEqual(len({video for shard in shards for video in shard}), 4)
        self.assertEqual(steps[0], max(steps))
        self.assertEqual(sum(steps), 21)


if __name__ == "__main__":
    unittest.main()
