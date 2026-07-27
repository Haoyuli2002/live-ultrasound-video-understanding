#!/usr/bin/env python3
"""
Compare two pretrain-inference prediction files side by side.

Typical use: compare the untrained base model vs the pretrained-adapter model
on the SAME pretrain samples, to see what the ASR caption-completion
pretraining actually taught.

Both inputs are jsonl files produced by pretrain/infer.py, each line:
  {"idx", "video_id", "video_window", "prev_context", "target",
   "prediction", "meta"}

Rows are matched by (video_id, video_window, idx). For each matched row we
print target / base_pred / lora_pred, and (optionally) a simple word-overlap
score of each prediction against the target.

Example:

python pretrain/compare_predictions.py \
  --base azure_data/checkpoints/pretrain_pred_BASE_limit20.jsonl \
  --lora azure_data/checkpoints/pretrain_pred_LORA_limit20.jsonl \
  --output azure_data/checkpoints/pretrain_compare_limit20.jsonl
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path


def _load(path):
    rows = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def _key(rec):
    vw = rec.get("video_window")
    vw = tuple(vw) if isinstance(vw, list) else vw
    return (rec.get("video_id"), vw, rec.get("idx"))


_WORD = re.compile(r"[a-z0-9]+")


def _tokens(text):
    return set(_WORD.findall((text or "").lower()))


def _overlap(pred, target):
    """Simple token-level F1 (word overlap) of prediction vs target."""
    p, t = _tokens(pred), _tokens(target)
    if not p or not t:
        return 0.0
    inter = len(p & t)
    if inter == 0:
        return 0.0
    prec = inter / len(p)
    rec = inter / len(t)
    return 2 * prec * rec / (prec + rec)


def main():
    parser = argparse.ArgumentParser(description="Compare base vs adapter pretrain predictions")
    parser.add_argument("--base", type=str, required=True, help="jsonl from base-only infer")
    parser.add_argument("--lora", type=str, required=True, help="jsonl from adapter infer")
    parser.add_argument("--output", type=str, default=None, help="Optional merged jsonl output")
    parser.add_argument("--limit", type=int, default=None, help="Only show first N matched rows")
    args = parser.parse_args()

    base_rows = _load(args.base)
    lora_rows = _load(args.lora)
    base_by = {_key(r): r for r in base_rows}
    lora_by = {_key(r): r for r in lora_rows}

    keys = [k for k in base_by if k in lora_by]
    # Keep the base file's order.
    ordered = [_key(r) for r in base_rows if _key(r) in lora_by]

    print(f"[compare] base rows={len(base_rows)} lora rows={len(lora_rows)} matched={len(keys)}")

    merged = []
    base_f1_sum = 0.0
    lora_f1_sum = 0.0
    n = 0
    shown = 0
    for k in ordered:
        b = base_by[k]
        l = lora_by[k]
        target = b.get("target", "")
        base_pred = b.get("prediction", "")
        lora_pred = l.get("prediction", "")

        base_f1 = _overlap(base_pred, target)
        lora_f1 = _overlap(lora_pred, target)
        base_f1_sum += base_f1
        lora_f1_sum += lora_f1
        n += 1

        rec = {
            "video_id": k[0],
            "video_window": list(k[1]) if isinstance(k[1], tuple) else k[1],
            "idx": k[2],
            "target": target,
            "base_pred": base_pred,
            "lora_pred": lora_pred,
            "base_target_f1": round(base_f1, 3),
            "lora_target_f1": round(lora_f1, 3),
        }
        merged.append(rec)

        if args.limit is None or shown < args.limit:
            shown += 1
            print("=" * 90)
            print(f"[{k[0]} | window={rec['video_window']} | idx={k[2]}]")
            print(f"  TARGET   : {target}")
            print(f"  BASE     : {base_pred}   (f1={base_f1:.3f})")
            print(f"  PRETRAIN : {lora_pred}   (f1={lora_f1:.3f})")

    if n:
        print("=" * 90)
        print(f"[compare] mean word-overlap F1 vs target:")
        print(f"    BASE (untrained) : {base_f1_sum / n:.3f}")
        print(f"    PRETRAIN adapter : {lora_f1_sum / n:.3f}")
        print(f"    delta            : {(lora_f1_sum - base_f1_sum) / n:+.3f}")

    if args.output:
        out = Path(args.output)
        out.parent.mkdir(parents=True, exist_ok=True)
        with out.open("w", encoding="utf-8") as f:
            for rec in merged:
                f.write(json.dumps(rec, ensure_ascii=False) + "\n")
        print(f"[compare] wrote merged comparison: {out}")


if __name__ == "__main__":
    main()