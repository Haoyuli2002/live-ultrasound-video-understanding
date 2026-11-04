"""Check experiment completeness and compare reconstruction (not clinical quality)."""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import csv
import json
from pathlib import Path
import re
import statistics


def read_rows(path):
    return [json.loads(line) for line in Path(path).read_text().splitlines() if line.strip()]


def normalized(text):
    return re.findall(r"[a-z0-9]+(?:'[a-z]+)?", text.lower())


def report_run(run):
    run = Path(run)
    plan = json.loads((run/"submission.json").read_text())
    expected = {"short": 12} if plan["mode"] == "short_only" else {"short": 12, "block": 2, "global": 2}
    training = read_rows(run/"training/training_metrics.jsonl")
    epochs = int(plan["EPOCHS"])
    if len(training) != sum(expected.values())*epochs:
        raise ValueError("Optimizer update count does not match the experiment plan")
    for epoch in range(epochs):
        rows = [r for r in training if r["epoch"] == epoch]
        if Counter(r["kind"] for r in rows) != expected or not all(r["fresh_memory"] for r in rows):
            raise ValueError(f"Missing tasks or non-fresh training memories in epoch {epoch+1}")
    checkpoints = {}
    for name in ["epoch-1", "final"]:
        rows = read_rows(run/f"reconstruction_{name}.jsonl")
        if Counter(r["kind"] for r in rows) != expected:
            raise ValueError(f"Incomplete teacher-forced evaluation: {name}")
        checkpoints[name] = {k: statistics.mean(r["loss"] for r in rows if r["kind"] == k)
                             for k in expected}
    recon = run/"reconstruct_final"
    manifest = json.loads((recon/"manifest.json").read_text())
    if manifest["reference_used_in_decoding"] is not False:
        raise ValueError("Reference leakage in decoding manifest")
    rows = read_rows(recon/"predictions.jsonl")
    if len(rows) != sum(expected.values())*3:
        raise ValueError("Missing free-running predictions")
    groups = defaultdict(list)
    samples = defaultdict(dict)
    identities = set()
    for row in rows:
        key = (row["video_id"], row["kind"], tuple(row["window"]))
        identity = (*key, row["condition"])
        if identity in identities or "skipped" in row:
            raise ValueError("Duplicate or skipped reconstruction sample")
        identities.add(identity)
        groups[(row["condition"], row["kind"])].append(row)
        samples[key][row["condition"]] = row
    for condition in ["real", "zero", "other_window"]:
        if Counter({k: len(groups[(condition, k)]) for k in expected}) != expected:
            raise ValueError(f"Incomplete reconstruction condition: {condition}")
    scores = {}
    for (condition, kind), values in groups.items():
        scores[f"{condition}/{kind}"] = {
            "samples": len(values),
            "exact_matches": sum(r["normalized_exact_match"] for r in values),
            "unique_predictions": len({tuple(normalized(r["prediction"])) for r in values}),
            "wrong_window_gt_repeats": sum(
                normalized(r["prediction"]) != normalized(r["reference"]) and any(
                    normalized(r["prediction"]) == normalized(other["reference"])
                    for other in values) for r in values),
            "empty": sum(not r["prediction"].strip() for r in values),
            "truncated": sum(r["stop_reason"] == "max_new_tokens" for r in values),
        }
    result = {"status": "completed", "mode": plan["mode"], "epochs": epochs,
              "optimizer_updates": len(training), "initialization": "official_checkpoint",
              "checkpoints": checkpoints, "free_running": scores,
              "interpretation": "Training-video overfit diagnostic; no automatic quality approval. "
              "Exact text matches do not establish visual faithfulness or generalization."}
    (run/"overfit_report.json").write_text(json.dumps(result, indent=2))
    with (run/"comparison.tsv").open("w", newline="") as stream:
        writer = csv.writer(stream, delimiter="\t")
        writer.writerow(["video_id", "kind", "window", "GT", "real", "zero", "other_window"])
        for (vid, kind, window), values in samples.items():
            writer.writerow([vid, kind, str(list(window)), values["real"]["reference"],
                             *[values[c]["prediction"] for c in ["real", "zero", "other_window"]]])
    return result


def report_pair(pair):
    pair = Path(pair)
    manifest = json.loads((pair/"experiments.json").read_text())
    results = {}
    for job in manifest["jobs"]:
        path = Path(job["run_dir"])/"overfit_report.json"
        results[job["mode"]] = (json.loads(path.read_text()) if path.is_file()
                                else {"status": "not_finished", "job_id": job["job_id"]})
    complete = all(r["status"] == "completed" for r in results.values())
    report = {"status": "completed" if complete else "not_finished", "experiments": results}
    (pair/"comparison.json").write_text(json.dumps(report, indent=2))
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--run-dir")
    source.add_argument("--pair-dir")
    args = parser.parse_args()
    result = report_run(args.run_dir) if args.run_dir else report_pair(args.pair_dir)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
