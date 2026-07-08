"""
Analyze raw / finetuned Qwen predictions for answerability-aware QA.

Input:
  JSONL produced by QA/eval/infer_qwen.py

Metrics:
  - overall_answerability_accuracy
  - wait_accuracy
  - answer_accuracy
  - premature_answer_rate
  - over_wait_rate
  - other_rate
  - breakdown by sample_type and qa_type

Usage:
  python QA/eval/analyze_predictions.py \
    --predictions QA/eval/results/qwen3vl_2b_raw_predictions_limit10.jsonl

  python QA/eval/analyze_predictions.py \
    --predictions QA/eval/results/qwen3vl_2b_raw_predictions_limit10.jsonl \
    --out QA/eval/results/qwen3vl_2b_raw_metrics_limit10.json
"""

from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, List


ANSWER_LABELS = {"WAIT", "ANSWER", "OTHER", "ERROR"}


def load_jsonl(path: str | Path) -> List[Dict[str, Any]]:
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def safe_div(num: int | float, den: int | float) -> float:
    return float(num) / float(den) if den else 0.0


def label_of(row: Dict[str, Any], key: str) -> str:
    v = row.get(key, "OTHER")
    if v not in ANSWER_LABELS:
        return "OTHER"
    return v


def summarize(rows: Iterable[Dict[str, Any]]) -> Dict[str, Any]:
    rows = list(rows)
    n = len(rows)

    gt_counts = Counter(label_of(r, "gt_label") for r in rows)
    pred_counts = Counter(label_of(r, "pred_label") for r in rows)

    correct = sum(1 for r in rows if bool(r.get("correct_answerability")))
    wait_rows = [r for r in rows if label_of(r, "gt_label") == "WAIT"]
    answer_rows = [r for r in rows if label_of(r, "gt_label") == "ANSWER"]

    wait_correct = sum(1 for r in wait_rows if label_of(r, "pred_label") == "WAIT")
    premature_answer = sum(1 for r in wait_rows if label_of(r, "pred_label") == "ANSWER")
    wait_other = sum(1 for r in wait_rows if label_of(r, "pred_label") in ("OTHER", "ERROR"))

    answer_correct = sum(1 for r in answer_rows if label_of(r, "pred_label") == "ANSWER")
    over_wait = sum(1 for r in answer_rows if label_of(r, "pred_label") == "WAIT")
    answer_other = sum(1 for r in answer_rows if label_of(r, "pred_label") in ("OTHER", "ERROR"))

    return {
        "total": n,
        "gt_counts": dict(gt_counts),
        "pred_counts": dict(pred_counts),
        "overall_answerability_accuracy": safe_div(correct, n),
        "wait": {
            "count": len(wait_rows),
            "wait_accuracy": safe_div(wait_correct, len(wait_rows)),
            "premature_answer_rate": safe_div(premature_answer, len(wait_rows)),
            "other_or_error_rate": safe_div(wait_other, len(wait_rows)),
            "correct": wait_correct,
            "premature_answer": premature_answer,
            "other_or_error": wait_other,
        },
        "answer": {
            "count": len(answer_rows),
            "answer_accuracy": safe_div(answer_correct, len(answer_rows)),
            "over_wait_rate": safe_div(over_wait, len(answer_rows)),
            "other_or_error_rate": safe_div(answer_other, len(answer_rows)),
            "correct": answer_correct,
            "over_wait": over_wait,
            "other_or_error": answer_other,
        },
    }


def group_by(rows: List[Dict[str, Any]], key: str) -> Dict[str, List[Dict[str, Any]]]:
    groups = defaultdict(list)
    for r in rows:
        groups[str(r.get(key, "UNKNOWN"))].append(r)
    return dict(groups)


def build_report(rows: List[Dict[str, Any]]) -> Dict[str, Any]:
    report = {
        "overall": summarize(rows),
        "by_sample_type": {},
        "by_qa_type": {},
    }

    for k, group_rows in group_by(rows, "sample_type").items():
        report["by_sample_type"][k] = summarize(group_rows)

    for k, group_rows in group_by(rows, "qa_type").items():
        report["by_qa_type"][k] = summarize(group_rows)

    return report


def fmt_pct(x: float) -> str:
    return f"{100.0 * x:.1f}%"


def print_summary(report: Dict[str, Any]) -> None:
    overall = report["overall"]
    print("=" * 72)
    print("ANSWERABILITY EVALUATION SUMMARY")
    print("=" * 72)
    print(f"Total samples: {overall['total']}")
    print(f"GT counts    : {overall['gt_counts']}")
    print(f"Pred counts  : {overall['pred_counts']}")
    print()
    print(f"Overall answerability accuracy: {fmt_pct(overall['overall_answerability_accuracy'])}")
    print()
    print("[WAIT samples]")
    w = overall["wait"]
    print(f"  count                  : {w['count']}")
    print(f"  WAIT accuracy          : {fmt_pct(w['wait_accuracy'])} ({w['correct']}/{w['count']})")
    print(f"  premature answer rate  : {fmt_pct(w['premature_answer_rate'])} ({w['premature_answer']}/{w['count']})")
    print(f"  other/error rate       : {fmt_pct(w['other_or_error_rate'])} ({w['other_or_error']}/{w['count']})")
    print()
    print("[ANSWER samples]")
    a = overall["answer"]
    print(f"  count                  : {a['count']}")
    print(f"  ANSWER accuracy        : {fmt_pct(a['answer_accuracy'])} ({a['correct']}/{a['count']})")
    print(f"  over-wait rate         : {fmt_pct(a['over_wait_rate'])} ({a['over_wait']}/{a['count']})")
    print(f"  other/error rate       : {fmt_pct(a['other_or_error_rate'])} ({a['other_or_error']}/{a['count']})")

    print()
    print("=" * 72)
    print("BY sample_type")
    print("=" * 72)
    for name, sub in sorted(report["by_sample_type"].items()):
        print(f"{name:24s} n={sub['total']:4d} overall={fmt_pct(sub['overall_answerability_accuracy'])} "
              f"gt={sub['gt_counts']} pred={sub['pred_counts']}")

    print()
    print("=" * 72)
    print("BY qa_type")
    print("=" * 72)
    for name, sub in sorted(report["by_qa_type"].items()):
        print(f"{name:24s} n={sub['total']:4d} overall={fmt_pct(sub['overall_answerability_accuracy'])} "
              f"gt={sub['gt_counts']} pred={sub['pred_counts']}")


def collect_failure_examples(rows: List[Dict[str, Any]], max_examples: int = 10) -> List[Dict[str, Any]]:
    examples = []
    for r in rows:
        if not r.get("correct_answerability"):
            examples.append({
                "idx": r.get("idx"),
                "sample_type": r.get("sample_type"),
                "qa_type": r.get("qa_type"),
                "gt_label": r.get("gt_label"),
                "pred_label": r.get("pred_label"),
                "question": r.get("question"),
                "target": r.get("target"),
                "prediction": r.get("prediction"),
                "error": r.get("error"),
            })
        if len(examples) >= max_examples:
            break
    return examples


def parse_args():
    parser = argparse.ArgumentParser(description="Analyze answerability prediction JSONL")
    parser.add_argument("--predictions", type=str, required=True)
    parser.add_argument("--out", type=str, default=None, help="Optional JSON metrics output path")
    parser.add_argument("--max-failure-examples", type=int, default=10)
    return parser.parse_args()


def main():
    args = parse_args()
    rows = load_jsonl(args.predictions)
    report = build_report(rows)
    report["failure_examples"] = collect_failure_examples(rows, args.max_failure_examples)

    print_summary(report)

    if report["failure_examples"]:
        print()
        print("=" * 72)
        print("FAILURE EXAMPLES")
        print("=" * 72)
        for ex in report["failure_examples"]:
            print(f"\n[{ex['idx']}] {ex['sample_type']} {ex['qa_type']} GT={ex['gt_label']} PRED={ex['pred_label']}")
            print(f"Q: {ex.get('question')}")
            print(f"TARGET: {str(ex.get('target'))[:240]}")
            print(f"PRED: {str(ex.get('prediction'))[:240]}")
            if ex.get("error"):
                print(f"ERROR: {ex['error']}")

    if args.out:
        out_path = Path(args.out)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with out_path.open("w", encoding="utf-8") as f:
            json.dump(report, f, ensure_ascii=False, indent=2)
        print()
        print(f"Saved metrics JSON to: {out_path}")


if __name__ == "__main__":
    main()