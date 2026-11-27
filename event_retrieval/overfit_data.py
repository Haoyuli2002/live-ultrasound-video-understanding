"""Expand the audited visual QA release into causal, in-sample decision targets."""
import json
import math
from pathlib import Path

from endtoend.data import QA_TYPES, read_rows


def load_overfit(directory, *, weight_answers_by_wait=False):
    directory = Path(directory)
    audit = json.loads((directory / "audit.json").read_text())
    videos = audit["videos"]
    if {p.stem for p in directory.glob("*.jsonl")} != set(videos):
        raise ValueError("Video files and audit disagree")
    reviewed = {(r["video_id"], r["question_id"]): r for r in audit["wait_review"]}
    skip = {(r["video_id"], r["query_s"]): set(r["skip_binary_at_s"])
            for r in audit["uncertain_pretrigger_samples"]}
    examples, seen, waits = [], set(), set()
    for video_id, meta in videos.items():
        video = Path(meta["source"]).resolve(strict=True)
        tasks = read_rows(directory / f"{video_id}.jsonl")
        if video.stem != video_id or len(tasks) != meta["questions"]:
            raise ValueError(f"Video identity or question count mismatch: {video_id}")
        for task in tasks:
            key = (video_id, task["question_id"])
            ask, answer = task["query_time"], task["answer_time"]
            if (key in seen or task["video_id"] != video_id or task["qa_type"] not in QA_TYPES
                    or not task["question"].strip() or not 0 <= ask <= meta["duration_s"]):
                raise ValueError(f"Invalid question: {key}")
            seen.add(key)
            waiting = ask != answer
            if (task["decision_at_query"] != ("WAIT" if waiting else "ANSWER")
                    or (answer is None) != (task["reference_answer"] is None)
                    or (answer is not None and not task["reference_answer"].strip())):
                raise ValueError(f"Inconsistent annotation: {key}")
            stop = answer
            if waiting:
                waits.add(key)
                review = reviewed[key]
                stop = review["coverage_end_s"]
                if (review["query_s"] != ask or review["first_clear_sample_s"] != answer
                        or (answer is not None and stop != answer)
                        or not task["decision_reason_at_query"].strip()):
                    raise ValueError(f"Invalid WAIT coverage: {key}")
            if not math.isfinite(stop) or not ask <= stop <= meta["duration_s"]:
                raise ValueError(f"Invalid decision interval: {key}")
            # Only observable ticks, plus a fractional query or positive decision if annotated.
            times = {ask, *range(math.ceil(ask), math.floor(stop) + 1)}
            if answer is not None:
                times.add(answer)
            ambiguous = skip.get((video_id, ask), set()) if waiting else set()
            if not ambiguous <= times - {ask, answer}:
                raise ValueError(f"Invalid ambiguity exclusion: {key}")
            included = sorted(times - ambiguous)
            wait_count = sum(now != answer for now in included)
            for now in included:
                ready = now == answer
                target = "<ANSWER> " + task["reference_answer"] if ready else "<WAIT>"
                if not ready and now == ask:
                    target += " " + task["decision_reason_at_query"].strip()
                task_id = "/".join(key)
                example = dict(task_id=task_id, sample_id=f"{task_id}@{now:g}",
                               video_id=video_id, video_path=str(video), split="overfit",
                               qa_type=task["qa_type"], question=task["question"],
                               query_time=ask, current_time=now, ready=int(ready),
                               answer=task["reference_answer"] if ready else "", target=target)
                if weight_answers_by_wait:
                    example["loss_weight"] = max(1, wait_count) if ready else 1
                examples.append(example)
    if (not examples or len(seen) != audit["question_count"] or waits != set(reviewed)
            or len(reviewed) != len(audit["wait_review"])):
        raise ValueError("Incomplete QA or WAIT review coverage")
    return examples
