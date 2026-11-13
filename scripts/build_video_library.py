"""Build a read-only browser for every labelled video and available Stage data."""

import argparse
import hashlib
import json
from collections import defaultdict
from pathlib import Path
from urllib.parse import quote


STAGE3_SOURCES = (
    ("visual_only", "Visual-only QA", "stage3_batch"),
    ("av_first50", "Audiovisual QA · first 50", "stage3_av50_20261105"),
    ("av_next50", "Audiovisual QA · next 50", "stage3_av_next50_20261106"),
    ("av_visual_first_draft", "Visual-first audiovisual QA · draft", "stage3_av_visualfirst_next50_20261110"),
)


def rows(path):
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def stage1_samples(path):
    grouped = defaultdict(dict)
    for row in rows(path):
        grouped[row["sentence_id"]][row["condition"]] = row
    result = []
    for sentence_id, group in sorted(grouped.items()):
        row = group.get("before_with_asr") or next(iter(group.values()))
        result.append({
            "sentence_id": sentence_id,
            "window": row["sentence_window"],
            "video_window": row["video_window"],
            "target": row["target"],
            "historical_asr": row.get("historical_asr", "")[-4000:],
            "conditions": sorted(group),
        })
    return result


def stage3_task(row, collection):
    return {
        "collection": collection,
        "task_id": row["task_id"],
        "qa_type": row["qa_type"],
        "response_group": row.get("response_group"),
        "query_time": row["query_time"],
        "answer_time": row.get("answer_time"),
        "review_horizon": row.get("review_horizon"),
        "question": row["question"],
        "reference_answer": row.get("reference_answer"),
        "query_time_missing_evidence": row.get("query_time_missing_evidence"),
        "answer_time_new_evidence": row.get("answer_time_new_evidence"),
        "historical_anchor": row.get("historical_anchor"),
        "evidence_modality": row.get("evidence_modality", "visual"),
        "review_status": row.get("review_status"),
    }


def build(labels_path: Path, source_dir: Path, output: Path, repo: Path) -> None:
    labels_path = labels_path.resolve()
    source_dir = source_dir.resolve()
    output = output.resolve()
    labels = json.loads(labels_path.read_text())
    videos = {}
    for row in labels["videos"]:
        video_id = row["video_id"]
        if video_id in videos:
            raise ValueError(f"Duplicate labelled video ID: {video_id}")
        split = row["split"]
        if split not in {"train", "eval"}:
            raise ValueError(f"Unknown split for {video_id}: {split}")
        relative = Path("cluster_data/videos_from_ssh") / f"{split}_full295" / f"{video_id}.mp4"
        if not (repo / relative).is_file():
            raise FileNotFoundError(repo / relative)
        for stage in ("stage1", "stage2", "stage3"):
            rating = row["stage_value"][stage]
            if rating["value"] not in {"high", "medium", "low"} or not rating["reason"]:
                raise ValueError(f"Incomplete {stage} rating for {video_id}")
        videos[video_id] = {
            "video_id": video_id,
            "split": split,
            "duration": row["duration"],
            "video_url": quote("../" + relative.as_posix()),
            "video_type": row["video_type"],
            "video_type_reason": row["video_type_reason"],
            "stage_value": row["stage_value"],
            "topics": row.get("topics", []),
            "visual_quality": row.get("visual_quality"),
            "stage_data": {"stage1": [], "stage2": [], "stage3": []},
        }

    sources = []
    for video_id in ("P3FoSFIrfQA", "8V649L5Q368"):
        path = repo / "viz/data" / f"{video_id}_stage1.jsonl"
        if path.is_file():
            videos[video_id]["stage_data"]["stage1"] = stage1_samples(path)
            sources.append({"stage": "stage1", "path": str(path.relative_to(repo)), "sha256": sha(path)})

    for split in ("train", "eval"):
        path = source_dir / f"stage2_{split}.jsonl"
        for row in rows(path):
            video_id = row["video_id"]
            if video_id not in videos or videos[video_id]["split"] != split:
                raise ValueError(f"Unexpected Stage 2 video: {video_id}")
            videos[video_id]["stage_data"]["stage2"].append({
                "block_window": row["block_window"],
                "local_sub_summaries": row["local_sub_summaries"],
                "block_summary_target": row.get("block_summary_target"),
                "global_summary_target": row.get("global_summary_target"),
            })
        sources.append({"stage": "stage2", "path": str(path.relative_to(repo)), "sha256": sha(path)})

    for collection, title, prefix in STAGE3_SOURCES:
        for split in ("train", "eval"):
            path = source_dir / f"{prefix}_{split}_tasks.jsonl"
            for row in rows(path):
                video_id = row["video_id"]
                if video_id not in videos or videos[video_id]["split"] != split:
                    raise ValueError(f"Unexpected Stage 3 video: {video_id}")
                videos[video_id]["stage_data"]["stage3"].append(stage3_task(row, collection))
            sources.append({"stage": "stage3", "collection": collection, "title": title,
                            "path": str(path.relative_to(repo)), "sha256": sha(path)})

    if any(not video["stage_data"]["stage2"] for video in videos.values()):
        raise ValueError("Stage 2 release does not cover every labelled video")
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps({
        "source_schema": labels.get("schema"),
        "label_source_sha256": sha(labels_path),
        "label_review_method": labels["videos"][0].get("review_method"),
        "label_audio_used": any(r.get("audio_used") for r in labels["videos"]),
        "label_asr_used": any(r.get("asr_used") for r in labels["videos"]),
        "sources": sources,
        "stage3_collections": [{"id": a, "title": b} for a, b, _ in STAGE3_SOURCES],
        "videos": list(videos.values()),
    }, ensure_ascii=False, separators=(",", ":")) + "\n")
    counts = {stage: sum(bool(v["stage_data"][stage]) for v in videos.values())
              for stage in ("stage1", "stage2", "stage3")}
    print(f"Wrote {len(videos)} videos, Stage coverage {counts} -> {output}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--labels", type=Path, default=Path("viz/data/video_labels.json"))
    parser.add_argument("--sources", type=Path, default=Path("viz/data/library_sources"))
    parser.add_argument("--output", type=Path, default=Path("viz/video_library.json"))
    args = parser.parse_args()
    build(args.labels, args.sources, args.output, Path(__file__).resolve().parents[1])
