"""One-shot smoke test: import every QA/*.py module and print key symbols.

    python QA/_smoke_import.py

Should print 'ALL IMPORTS OK' at the end and exit 0.
"""
from __future__ import annotations

import sys
from pathlib import Path

# Allow `from _shared import ...` inside sibling modules.
_HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE))


def main() -> int:
    print("Importing _shared...")
    from _shared import (
        DEFAULT_MODEL,
        build_openrouter_client,  # noqa: F401
        build_video_block,        # noqa: F401
        call_with_content,        # noqa: F401
        cut_clip,                 # noqa: F401
        temp_clip_path,           # noqa: F401
        text_block,               # noqa: F401
    )
    print(f"  DEFAULT_MODEL = {DEFAULT_MODEL}")

    print("Importing generator...")
    import generator
    print(f"  STREAMING_QA_TYPES = {generator.STREAMING_QA_TYPES}")
    print(f"  TIME_RATIOS = {generator.TIME_RATIOS}")
    print(f"  fn generate_streaming_qa_for_video = "
          f"{generator.generate_streaming_qa_for_video.__name__}")
    print(f"  fn _validate_qa_entry = {generator._validate_qa_entry.__name__}")

    print("Importing offline_generator...")
    import offline_generator
    print(f"  OFFLINE_QA_TYPES = {offline_generator.OFFLINE_QA_TYPES}")
    print(f"  CLIP_MAX_SEC = {offline_generator.CLIP_MAX_SEC}")
    print(f"  fn generate_offline_qa_for_video = "
          f"{offline_generator.generate_offline_qa_for_video.__name__}")

    print("Importing validator...")
    import validator
    print(f"  VALIDATOR_MODEL = {validator.VALIDATOR_MODEL}")
    print(f"  fn validate_streaming_qa_file = "
          f"{validator.validate_streaming_qa_file.__name__}")
    print(f"  fn _resolve_verdict = {validator._resolve_verdict.__name__}")

    print("Importing merger...")
    import merger
    print(f"  DEFAULT_WINDOW_SEC = {merger.DEFAULT_WINDOW_SEC}")
    print(f"  WAIT_TARGET = {merger.WAIT_TARGET!r}")
    print(f"  fn build_per_video_record = "
          f"{merger.build_per_video_record.__name__}")
    print(f"  fn expand_training_samples = "
          f"{merger.expand_training_samples.__name__}")

    print("Importing run...")
    import run
    print(f"  fn run = {run.run.__name__}")

    # Exercise a purely-in-memory path: validator's verdict resolver.
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
    assert v["checks"] == parsed_ok["checks"], v

    parsed_bad = {
        "checks": {
            "question_no_leak": True,
            "not_answerable_at_query_time": False,
            "answerable_at_answer_time": True,
        },
        "verdict": "pass",       # inconsistent -> validator must override to fail
        "reason": "test",
    }
    v2 = validator._resolve_verdict(parsed_bad)
    assert v2["verdict"] == "fail", v2
    assert "overridden" in v2["reason"], v2

    # Exercise generator's entry validator using a currently-registered type.
    test_type = generator.STREAMING_QA_TYPES[0]  # 'next_action' at time of writing
    ok, _ = generator._validate_qa_entry(
        {
            "type": test_type,
            "question": "q",
            "answer": "a",
            "answer_time": 100.0,
        },
        query_time=90.0,
        clip_end=200.0,
    )
    assert ok is True, f"entry with type={test_type!r} should validate ok"

    bad_ok, reason = generator._validate_qa_entry(
        {
            "type": test_type,
            "question": "q",
            "answer": "a",
            "answer_time": 89.0,  # <= query_time
        },
        query_time=90.0,
        clip_end=200.0,
    )
    assert bad_ok is False
    assert "answer_time" in reason

    # Exercise merger's WAIT/ANSWER expansion (no video I/O).
    # Fake minimal inputs
    import json
    import tempfile

    tmpdir = Path(tempfile.mkdtemp(prefix="qa_smoke_"))
    transcript_path = tmpdir / "transcript.json"
    clips_path = tmpdir / "clips.json"
    streaming_qa_path = tmpdir / "streaming.json"

    transcript_path.write_text(json.dumps({
        "video_id": "SMOKE",
        "duration_sec": 300.0,
        "language": "en",
        "segments": [
            {"start": 0.0, "end": 3.0, "text": "hello"},
            {"start": 3.0, "end": 6.0, "text": "world"},
        ],
    }))
    clips_path.write_text(json.dumps({
        "video_id": "SMOKE",
        "clips": [
            {"clip_idx": 0, "start": 0.0, "end": 100.0,
             "duration": 100.0, "topic": "smoke", "text": "hello world"},
        ],
    }))
    streaming_qa_path.write_text(json.dumps({
        "video_id": "SMOKE",
        "streaming_qa": [
            {
                "source": "streaming",
                "type": "sonographer_intent",
                "video_id": "SMOKE",
                "clip_idx": 0,
                "clip_start": 0.0,
                "clip_end": 100.0,
                "topic": "smoke",
                "query_time": 40.0,
                "answer_time": 55.0,
                "evidence_window": [40.0, 55.0],
                "ratio": 0.5,
                "question": "Q?",
                "answer": "A.",
                "evidence": "because.",
                "validation": {
                    "verdict": "pass",
                    "checks": {"question_no_leak": True,
                               "not_answerable_at_query_time": True,
                               "answerable_at_answer_time": True},
                    "reason": "ok",
                    "validator_model": "test",
                },
            }
        ],
    }))

    record = merger.build_per_video_record(
        "SMOKE",
        str(transcript_path),
        str(clips_path),
        offline_qa_path=None,
        streaming_qa_path=str(streaming_qa_path),
    )
    assert record["num_qa"] == 1, record
    assert record["qa"][0]["query_time"] == 40.0
    assert record["qa"][0]["answer_time"] == 55.0

    samples = merger.expand_training_samples(
        "SMOKE",
        str(transcript_path),
        str(clips_path),
        streaming_qa_path=str(streaming_qa_path),
        window_sec=30.0,
    )
    kinds = sorted({s["sample_type"] for s in samples})
    assert kinds == ["streaming_answer", "streaming_wait"], kinds

    wait = next(s for s in samples if s["sample_type"] == "streaming_wait")
    answer = next(s for s in samples if s["sample_type"] == "streaming_answer")
    # wait window ends at query_time
    assert wait["video_window"][1] == 40.0, wait
    # answer window ends at answer_time
    assert answer["video_window"][1] == 55.0, answer
    # both windows clamp to clip_start
    assert wait["video_window"][0] >= 0.0
    assert answer["video_window"][0] >= 0.0
    assert wait["target"].startswith("<WAIT>")
    assert answer["target"].startswith("<ANSWER>")

    print()
    print("ALL IMPORTS OK")
    return 0


if __name__ == "__main__":
    sys.exit(main())