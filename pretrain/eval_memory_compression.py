#!/usr/bin/env python3
"""Evaluate Stage-2 summary-compression generations.

Input is JSONL produced by `pretrain/infer_memory_compression.py` with fields
`target` and `prediction`. This script intentionally uses no extra dependencies.
"""

from __future__ import annotations

import argparse
import json
import re
from collections import Counter, defaultdict
from pathlib import Path


MEDICAL_TERMS = {
    "ultrasound", "probe", "transducer", "pleura", "pleural", "lung", "sliding",
    "barcode", "seashore", "pneumothorax", "effusion", "diaphragm", "rib", "ribs",
    "a-line", "a-lines", "b-line", "b-lines", "artifact", "consolidation",
    "anterior", "posterior", "axillary", "chest", "scan", "shadow", "fluid",
}


def toks(text: str):
    return re.findall(r"[a-z0-9]+(?:-[a-z0-9]+)?", (text or "").lower())


def word_f1(pred: str, ref: str) -> float:
    p = toks(pred); r = toks(ref)
    if not p or not r:
        return 0.0
    pc = Counter(p); rc = Counter(r)
    overlap = sum((pc & rc).values())
    if overlap == 0:
        return 0.0
    prec = overlap / len(p); rec = overlap / len(r)
    return 2 * prec * rec / (prec + rec)


def lcs_len(a, b):
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


def medical_recall(pred: str, ref: str) -> float | None:
    p = set(toks(pred)); r = set(toks(ref)) & MEDICAL_TERMS
    if not r:
        return None
    return len(p & r) / len(r)


def parse_args():
    p = argparse.ArgumentParser(description="Evaluate summary compression predictions")
    p.add_argument("--pred-jsonl", required=True)
    p.add_argument("--output", default=None, help="Optional JSON summary path")
    return p.parse_args()


def main():
    args = parse_args()
    rows = []
    with Path(args.pred_jsonl).open(encoding="utf-8") as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
    if not rows:
        raise ValueError(f"No rows in {args.pred_jsonl}")

    sums = defaultdict(float); counts = defaultdict(int)
    examples = []
    for r in rows:
        typ = r.get("sample_type", "unknown")
        pred = r.get("prediction", "") or ""
        ref = r.get("target", "") or ""
        wf1 = word_f1(pred, ref)
        rl = rouge_l_f1(pred, ref)
        mr = medical_recall(pred, ref)
        for key, val in [("word_f1", wf1), ("rouge_l_f1", rl)]:
            sums[(typ, key)] += val; counts[(typ, key)] += 1
            sums[("overall", key)] += val; counts[("overall", key)] += 1
        if mr is not None:
            sums[(typ, "medical_recall")] += mr; counts[(typ, "medical_recall")] += 1
            sums[("overall", "medical_recall")] += mr; counts[("overall", "medical_recall")] += 1
        if len(examples) < 5:
            examples.append({"type": typ, "target": ref[:240], "prediction": pred[:240], "word_f1": wf1, "rouge_l_f1": rl, "medical_recall": mr})

    report = {"num_samples": len(rows), "metrics": {}, "examples": examples}
    for (typ, key), total in sorted(sums.items()):
        report["metrics"].setdefault(typ, {})[key] = total / max(1, counts[(typ, key)])
    print(json.dumps(report, ensure_ascii=False, indent=2))
    if args.output:
        out = Path(args.output); out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()