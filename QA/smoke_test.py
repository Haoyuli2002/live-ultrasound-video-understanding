"""
Clean QA pipeline smoke test.

This script performs lightweight import and pure-Python logic checks for the
main QA pipeline modules. It does NOT call external APIs and does NOT load
large models.

Usage:
    python QA/smoke_test.py

Expected:
    ALL QA SMOKE TESTS PASSED
"""

from __future__ import annotations

import json
import sys
import tempfile
from pathlib import Path

_HERE = Path(__file__).resolve().parent
_REPO_ROOT = _HERE.parent

# Make QA top-level modules importable.
sys.path.insert(0, str(_HERE))
# Make train/eval/prepare modules importable for direct smoke imports.
sys.path.insert(0, str(_HERE / "train"))
sys.path.insert(0, str(_HERE / "eval"))
sys.path.insert(0, str(_HERE / "prepare"))


def smoke_imports() -> None:
    print("[1/4] Importing QA modules...")

    import offline_generator
    import generator
    import validator
    import merger
    import run

    from prepare import asr, clipping, run_prepare
    from train import video_sampling, dataset, collator
    from eval import infer_qwen, analyze_predictions

    print(f"  offline types      : {offline_generator.OFFLINE_QA_TYPES}")
    print(f"  streaming types    : {generator.STREAMING_QA_TYPES}")
    print(f"  time ratios        : {generator.TIME_RATIOS}")
    print(f"  validator model    : {validator.VALIDATOR_MODEL}")
    print(f"  merger window sec  : {merger.DEFAULT_WINDOW_SEC}")

    # Touch symbols so import regressions are caught.
    assert callable(asr.run_asr)
    assert callable(clipping.run_clipping)
    assert callable(run_prepare.run_prepare)
    assert callable(video_sampling.sample_last_n_frames)
    assert hasattr(dataset, "QATrainingDataset")
    assert hasattr(collator, "QwenVLCollator")
    assert callable(infer_qwen.pred_label_from_text)
    assert callable(analyze_predictions.summarize)
    assert callable(run.run)


def smoke_validator_logic() -> None:
    print("[2/4] Checking validator verdict logic...")

    import validator

    parsed_ok = {
        "checks": {
            "question_no_leak": True,
            "not_answerable_at_query_time": True,
            "answerable_at_answer_time": True,
        },
        "verdict": "pass",
        "reason": "test",
    }
    v = validator._resolve_verdict(parsed_ok)
    assert v["verdict"] == "pass", v

    parsed_bad = {
        "checks": {
            "question_no_leak": True,
            "not_answerable_at_query_time": False,
            "answerable_at_answer_time": True,
        },
        "verdict": "pass",
        "reason": "test",
    }
    v2 = validator._resolve_verdict(parsed_bad)
    assert v2["verdict"] == "fail", v2
    assert "overridden" in v2["reason"], v2


def smoke_generator_logic() -> None:
    print("[3/4] Checking generator QA entry validation...")

    import generator

    test_type = generator.STREAMING_QA_TYPES[0]  # next_action

    ok, reason = generator._validate_qa_entry(
        {
            "type": test_type,
            "question": "What should the operator do next?",
            "answer": "The operator should tilt the probe cranially.",
            "answer_time": 100.0,
        },
        query_time=90.0,
        clip_end=200.0,
    )
    assert ok is True, reason

    bad_ok, reason = generator._validate_qa_entry(
        {
            "type": test_type,
            "question": "What should the operator do next?",
            "answer": "The operator should tilt the probe cranially.",
            "answer_time": 89.0,
        },
        query_time=90.0,
        clip_end=200.0,
    )
    assert bad_ok is False
    assert "answer_time" in reason


def smoke_merger_and_eval_logic() -> None:
    print("[4/4] Checking merger expansion and eval metrics...")

    import merger
    from eval import infer_qwen, analyze_predictions

    tmpdir = Path(tempfile.mkdtemp(prefix="qa_smoke_"))
    transcript_path = tmpdir / "transcript.json"
    clips_path = tmpdir / "clips.json"
    offline_qa_path = tmpdir / "offline.json"
    streaming_qa_path = tmpdir / "streaming_validated.json"

    transcript_path.write_text(json.dumps({
        "video_id": "SMOKE",
        "duration_sec": 100.0,
        "language": "en",
        "segments": [
            {"start": 0.0, "end": 3.0, "text": "hello"},
            {"start": 3.0, "end": 6.0, "text": "world"},
        ],
    }), encoding="utf-8")

    clips_path.write_text(json.dumps({
        "video_id": "SMOKE",
        "clips": [
            {
                "clip_idx": 0,
                "start": 0.0,
                "end": 100.0,
                "duration": 100.0,
                "topic": "smoke",
                "text": "hello world",
            }
        ],
    }), encoding="utf-8")

    offline_qa_path.write_text(json.dumps({
        "video_id": "SMOKE",
        "qa_pairs": [
            {
                "source": "offline",
                "type": "clip_summary",
                "video_id": "SMOKE",
                "clip_idx": 0,
                "clip_start": 0.0,
                "clip_end": 100.0,
                "topic": "smoke",
                "question": "What does this clip demonstrate?",
                "answer": "A complete overview.",
                "evidence": "demo",
            }
        ],
    }), encoding="utf-8")

    streaming_qa_path.write_text(json.dumps({
        "video_id": "SMOKE",
        "streaming_qa": [
            {
                "source": "streaming",
                "type": "next_observation",
                "video_id": "SMOKE",
                "clip_idx": 0,
                "clip_start": 0.0,
                "clip_end": 100.0,
                "topic": "smoke",
                "query_time": 40.0,
                "answer_time": 55.0,
                "evidence_window": [40.0, 55.0],
                "ratio": 0.3,
                "question": "What should the learner look for next?",
                "answer": "Look for a bright line.",
                "evidence": "because.",
                "validation": {
                    "verdict": "pass",
                    "checks": {
                        "question_no_leak": True,
                        "not_answerable_at_query_time": True,
                        "answerable_at_answer_time": True,
                    },
                    "reason": "ok",
                    "validator_model": "test",
                },
            }
        ],
    }), encoding="utf-8")

    record = merger.build_per_video_record(
        "SMOKE",
        str(transcript_path),
        str(clips_path),
        offline_qa_path=str(offline_qa_path),
        streaming_qa_path=str(streaming_qa_path),
    )
    assert record["num_qa"] == 2, record
    assert record["qa_type_counts"]["clip_summary"] == 1
    assert record["qa_type_counts"]["next_observation"] == 1

    samples = merger.expand_training_samples(
        "SMOKE",
        str(transcript_path),
        str(clips_path),
        offline_qa_path=str(offline_qa_path),
        streaming_qa_path=str(streaming_qa_path),
        window_sec=30.0,
    )
    counts = {}
    for s in samples:
        counts[s["sample_type"]] = counts.get(s["sample_type"], 0) + 1

    assert counts == {
        "offline_answer": 1,
        "streaming_wait": 1,
        "streaming_answer": 1,
    }, counts

    assert infer_qwen.pred_label_from_text("<WAIT> Not enough information yet.") == "WAIT"
    assert infer_qwen.pred_label_from_text("<ANSWER> Look for the pleural line.") == "ANSWER"
    assert infer_qwen.pred_label_from_text("Look for the pleural line.") == "ANSWER"

    pred_rows = [
        {"gt_label": "WAIT", "pred_label": "WAIT", "correct_answerability": True, "sample_type": "streaming_wait", "qa_type": "next_observation"},
        {"gt_label": "WAIT", "pred_label": "ANSWER", "correct_answerability": False, "sample_type": "streaming_wait", "qa_type": "next_observation"},
        {"gt_label": "ANSWER", "pred_label": "ANSWER", "correct_answerability": True, "sample_type": "streaming_answer", "qa_type": "next_observation"},
        {"gt_label": "ANSWER", "pred_label": "WAIT", "correct_answerability": False, "sample_type": "offline_answer", "qa_type": "clip_summary"},
    ]
    report = analyze_predictions.build_report(pred_rows)
    assert report["overall"]["total"] == 4
    assert report["overall"]["wait"]["premature_answer"] == 1
    assert report["overall"]["answer"]["over_wait"] == 1


def main() -> int:
    smoke_imports()
    smoke_validator_logic()
    smoke_generator_logic()
    smoke_merger_and_eval_logic()
    print()
    print("ALL QA SMOKE TESTS PASSED")
    return 0


if __name__ == "__main__":
    sys.exit(main())