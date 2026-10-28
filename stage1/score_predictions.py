"""Score Stage 1 prediction JSONL: token F1 + ROUGE-L, aggregated by condition.

Reads prediction files produced by ``stage1.evaluate`` (one JSON object per line
with at least ``target`` and ``prediction``; optionally ``condition`` and
``visual_control``) and reports per-sample scores plus aggregates overall and
per condition. When two files are given (e.g. base vs. Stage 1 adapter) the tool
prints a side-by-side delta so improvement attributable to the adapter is
visible. Metric definitions mirror ``pretrain/eval_memory_compression.py`` so the
whole repository shares one token-F1 / ROUGE-L definition; no extra dependencies.
"""
from __future__ import annotations

import argparse
import json
import re
from collections import Counter, defaultdict
from pathlib import Path

from .eval_io import paired


def toks(text: str) -> list[str]:
    return re.findall(r"[a-z0-9]+(?:-[a-z0-9]+)?", (text or "").lower())


def token_f1(pred: str, ref: str) -> float:
    """Multiset token-overlap F1 (same definition as pretrain word_f1)."""
    p = toks(pred); r = toks(ref)
    if not p or not r:
        return 0.0
    overlap = sum((Counter(p) & Counter(r)).values())
    if overlap == 0:
        return 0.0
    prec = overlap / len(p); rec = overlap / len(r)
    return 2 * prec * rec / (prec + rec)


def lcs_len(a: list[str], b: list[str]) -> int:
    if not a or not b:
        return 0
    prev = [0] * (len(b) + 1)
    for x in a:
        cur = [0]
        for j, y in enumerate(b, 1):
            cur.append(prev[j - 1] + 1 if x == y else max(prev[j], cur[-1]))
        prev = cur
    return prev[-1]


def rouge_l_f1(pred: str, ref: str) -> float:
    p = toks(pred); r = toks(ref)
    if not p or not r:
        return 0.0
    l = lcs_len(p, r)
    if l == 0:
        return 0.0
    prec = l / len(p); rec = l / len(r)
    return 2 * prec * rec / (prec + rec)


def load_rows(path: Path) -> list[dict]:
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()
            if line.strip()]
    if not rows:
        raise ValueError(f"No prediction rows in {path}")
    for row in rows:
        if "target" not in row or "prediction" not in row:
            raise ValueError(f"Prediction rows need 'target' and 'prediction': {path}")
    return rows


def score_rows(rows: list[dict]) -> dict:
    """Return {'num_samples', 'per_sample':[...], 'aggregate':{group:{metric:mean}}}.

    Groups are 'overall' plus each distinct ``condition`` value. Per-sample rows
    carry the identifying keys so individual predictions stay inspectable.
    """
    sums: dict[tuple[str, str], float] = defaultdict(float)
    counts: dict[tuple[str, str], int] = defaultdict(int)
    per_sample = []
    for row in rows:
        pred = row.get("prediction", "") or ""
        ref = row.get("target", "") or ""
        f1 = token_f1(pred, ref)
        rl = rouge_l_f1(pred, ref)
        condition = row.get("condition", "unknown")
        for group in ("overall", condition):
            for key, val in (("token_f1", f1), ("rouge_l_f1", rl)):
                sums[(group, key)] += val
                counts[(group, key)] += 1
        per_sample.append({
            "video_id": row.get("video_id"), "sentence_id": row.get("sentence_id"),
            "condition": condition, "visual_control": row.get("visual_control"),
            "token_f1": f1, "rouge_l_f1": rl,
        })
    aggregate: dict[str, dict[str, float]] = {}
    for (group, key), total in sums.items():
        aggregate.setdefault(group, {})[key] = total / max(1, counts[(group, key)])
    return {"num_samples": len(rows), "aggregate": aggregate, "per_sample": per_sample}


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--predictions", required=True,
                        help="Prediction JSONL (e.g. the Stage 1 adapter/lora run)")
    parser.add_argument("--baseline", default=None,
                        help="Optional second prediction JSONL (e.g. base) for a delta")
    parser.add_argument("--output", default=None,
                        help="Optional JSON metrics path; per-sample scores go to "
                             "<output>.per_sample.jsonl when set")
    return parser


def _print_aggregate(label: str, agg: dict[str, dict[str, float]]) -> None:
    print(f"== {label} ==")
    for group in sorted(agg):
        metrics = agg[group]
        print(f"  {group:18s} token_f1={metrics.get('token_f1', 0):.4f} "
              f"rouge_l_f1={metrics.get('rouge_l_f1', 0):.4f}")


def main() -> None:
    args = build_arg_parser().parse_args()
    pred_path = Path(args.predictions)
    scored = score_rows(load_rows(pred_path))
    _print_aggregate(f"predictions ({pred_path.name}), n={scored['num_samples']}",
                     scored["aggregate"])

    report = {"predictions": str(pred_path), "num_samples": scored["num_samples"],
              "aggregate": scored["aggregate"]}

    if args.baseline:
        base_path = Path(args.baseline)
        base_rows = load_rows(base_path)
        paired(load_rows(pred_path), base_rows)
        base_scored = score_rows(base_rows)
        _print_aggregate(f"baseline ({base_path.name}), n={base_scored['num_samples']}",
                         base_scored["aggregate"])
        delta: dict[str, dict[str, float]] = {}
        for group, metrics in scored["aggregate"].items():
            base_metrics = base_scored["aggregate"].get(group, {})
            delta[group] = {k: metrics[k] - base_metrics.get(k, 0.0) for k in metrics}
        _print_aggregate("delta (predictions - baseline)", delta)
        report["baseline"] = str(base_path)
        report["baseline_aggregate"] = base_scored["aggregate"]
        report["delta"] = delta

    if args.output:
        out = Path(args.output)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
        per_sample_path = Path(str(out) + ".per_sample.jsonl")
        with per_sample_path.open("w", encoding="utf-8") as stream:
            for item in scored["per_sample"]:
                stream.write(json.dumps(item, ensure_ascii=False) + "\n")
        print(f"[score] wrote metrics -> {out}")
        print(f"[score] wrote per-sample -> {per_sample_path}")


if __name__ == "__main__":
    main()
