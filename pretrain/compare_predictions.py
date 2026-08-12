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
print target / base_pred / lora_pred and automatic metrics.

Metrics:
  - word-overlap F1
  - BLEU-1 / BLEU-2 / BLEU-4
  - ROUGE-L F1
  - prefix match @ 1 / 3 / 5
  - length and length ratio
  - optional semantic cosine similarity

Example:

python pretrain/compare_predictions.py \
  --base cluster_data/eval/pretrain_eval20_base_full.jsonl \
  --lora cluster_data/eval/pretrain_eval20_lora_full.jsonl \
  --output cluster_data/eval/pretrain_eval20_compare_full.jsonl \
  --embed-device cpu
"""

from __future__ import annotations

import argparse
import json
import math
import re
from collections import Counter
from pathlib import Path
from statistics import mean


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


def _token_list(text):
    return _WORD.findall((text or "").lower())


def _tokens(text):
    return set(_token_list(text))


def _safe_div(num, den):
    return float(num) / float(den) if den else 0.0


def _overlap(pred, target):
    """Simple token-level F1 (set word overlap) of prediction vs target."""
    p, t = _tokens(pred), _tokens(target)
    if not p or not t:
        return 0.0
    inter = len(p & t)
    if inter == 0:
        return 0.0
    prec = inter / len(p)
    rec = inter / len(t)
    return 2 * prec * rec / (prec + rec)


def _ngrams(tokens, n):
    if n <= 0 or len(tokens) < n:
        return []
    return [tuple(tokens[i:i + n]) for i in range(len(tokens) - n + 1)]


def _modified_ngram_precision(pred_tokens, ref_tokens, n):
    pred_ngrams = Counter(_ngrams(pred_tokens, n))
    ref_ngrams = Counter(_ngrams(ref_tokens, n))
    if not pred_ngrams:
        return 0.0

    clipped = 0
    total = 0
    for gram, count in pred_ngrams.items():
        clipped += min(count, ref_ngrams.get(gram, 0))
        total += count
    return _safe_div(clipped, total)


def _bleu(pred, target, max_n=4):
    """Sentence BLEU-N with simple smoothing for zero precisions."""
    pred_tokens = _token_list(pred)
    ref_tokens = _token_list(target)
    if not pred_tokens or not ref_tokens:
        return 0.0

    precisions = []
    for n in range(1, max_n + 1):
        p_n = _modified_ngram_precision(pred_tokens, ref_tokens, n)
        # Smooth to avoid BLEU becoming exactly zero for short generations.
        if p_n == 0.0:
            p_n = 1e-9
        precisions.append(p_n)

    weights = [1.0 / max_n] * max_n
    log_precision = sum(w * math.log(p) for w, p in zip(weights, precisions))

    c = len(pred_tokens)
    r = len(ref_tokens)
    bp = 1.0 if c > r else math.exp(1.0 - (r / c))
    return float(bp * math.exp(log_precision))


def _lcs_len(a, b):
    if not a or not b:
        return 0
    # Keep memory small: O(min(len(a), len(b))).
    if len(b) > len(a):
        a, b = b, a
    prev = [0] * (len(b) + 1)
    for x in a:
        cur = [0]
        for j, y in enumerate(b, 1):
            if x == y:
                cur.append(prev[j - 1] + 1)
            else:
                cur.append(max(prev[j], cur[-1]))
        prev = cur
    return prev[-1]


def _rouge_l_f1(pred, target, beta=1.0):
    pred_tokens = _token_list(pred)
    ref_tokens = _token_list(target)
    if not pred_tokens or not ref_tokens:
        return 0.0

    lcs = _lcs_len(pred_tokens, ref_tokens)
    if lcs == 0:
        return 0.0

    r_lcs = lcs / len(ref_tokens)
    p_lcs = lcs / len(pred_tokens)
    beta2 = beta * beta
    denom = r_lcs + beta2 * p_lcs
    return _safe_div((1 + beta2) * r_lcs * p_lcs, denom)


def _prefix_match(pred, target, k):
    pred_tokens = _token_list(pred)
    ref_tokens = _token_list(target)
    if len(ref_tokens) < k or len(pred_tokens) < k:
        return 0.0
    return 1.0 if pred_tokens[:k] == ref_tokens[:k] else 0.0


def _length_metrics(pred, target):
    pred_len = len(_token_list(pred))
    target_len = len(_token_list(target))
    return {
        "pred_len": pred_len,
        "target_len": target_len,
        "len_ratio": _safe_div(pred_len, target_len),
    }


def _all_direct_metrics(pred, target):
    length = _length_metrics(pred, target)
    return {
        "word_f1": _overlap(pred, target),
        "bleu1": _bleu(pred, target, max_n=1),
        "bleu2": _bleu(pred, target, max_n=2),
        "bleu4": _bleu(pred, target, max_n=4),
        "rouge_l": _rouge_l_f1(pred, target),
        "prefix1": _prefix_match(pred, target, 1),
        "prefix3": _prefix_match(pred, target, 3),
        "prefix5": _prefix_match(pred, target, 5),
        **length,
    }


# ============================================================================
# Semantic cosine similarity (sentence-embedding). Optional; lazy-loaded.
# ============================================================================

DEFAULT_EMBED_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
_EMBED_MODEL_CACHE = {}


def _resolve_device(device="auto"):
    if device and device != "auto":
        return device
    try:
        import torch
        if torch.cuda.is_available():
            return "cuda"
    except Exception:
        pass
    return "cpu"


def _load_sentence_model(model_name, device):
    key = (model_name, device)
    if key in _EMBED_MODEL_CACHE:
        return _EMBED_MODEL_CACHE[key]
    from sentence_transformers import SentenceTransformer  # may raise ImportError
    model = SentenceTransformer(model_name, device=device)
    _EMBED_MODEL_CACHE[key] = model
    return model


def _semantic_cosines(model, preds, targets, batch_size=64):
    """
    Return a list of cosine similarities (float) between each pred and its
    target, computed on L2-normalized sentence embeddings.
    """
    import numpy as np

    pe = np.asarray(
        model.encode(preds, batch_size=batch_size, convert_to_numpy=True,
                     normalize_embeddings=True, show_progress_bar=False),
        dtype=np.float32,
    )
    te = np.asarray(
        model.encode(targets, batch_size=batch_size, convert_to_numpy=True,
                     normalize_embeddings=True, show_progress_bar=False),
        dtype=np.float32,
    )
    return [float((pe[i] * te[i]).sum()) for i in range(len(preds))]


def _mean(values):
    return mean(values) if values else 0.0


def _metric_delta(lora_value, base_value):
    return lora_value - base_value


def _print_metric(name, base_values, lora_values, fmt="{:.3f}"):
    base_mean = _mean(base_values)
    lora_mean = _mean(lora_values)
    delta = _metric_delta(lora_mean, base_mean)
    print(f"[compare] mean {name}:")
    print(f"    BASE (untrained) : {fmt.format(base_mean)}")
    print(f"    PRETRAIN adapter : {fmt.format(lora_mean)}")
    print(f"    delta            : {delta:+.3f}")


def main():
    parser = argparse.ArgumentParser(description="Compare base vs adapter pretrain predictions")
    parser.add_argument("--base", type=str, required=True, help="jsonl from base-only infer")
    parser.add_argument("--lora", type=str, required=True, help="jsonl from adapter infer")
    parser.add_argument("--output", type=str, default=None, help="Optional merged jsonl output")
    parser.add_argument("--summary-json", type=str, default=None, help="Optional summary JSON output")
    parser.add_argument("--limit", type=int, default=None, help="Only show first N matched rows")

    parser.add_argument("--no-semantic", action="store_true",
                        help="Disable sentence-embedding cosine similarity (only direct text metrics).")
    parser.add_argument("--embed-model", type=str, default=DEFAULT_EMBED_MODEL,
                        help=f"sentence-transformers model for semantic cosine (default {DEFAULT_EMBED_MODEL}).")
    parser.add_argument("--embed-device", type=str, default="auto")
    args = parser.parse_args()

    base_rows = _load(args.base)
    lora_rows = _load(args.lora)
    base_by = {_key(r): r for r in base_rows}
    lora_by = {_key(r): r for r in lora_rows}

    keys = [k for k in base_by if k in lora_by]
    ordered = [_key(r) for r in base_rows if _key(r) in lora_by]

    print(f"[compare] base rows={len(base_rows)} lora rows={len(lora_rows)} matched={len(keys)}")

    targets = [base_by[k].get("target", "") for k in ordered]
    base_preds = [base_by[k].get("prediction", "") for k in ordered]
    lora_preds = [lora_by[k].get("prediction", "") for k in ordered]

    base_cos = lora_cos = None
    if not args.no_semantic and ordered:
        try:
            device = _resolve_device(args.embed_device)
            print(f"[compare] semantic model={args.embed_model} device={device}")
            model = _load_sentence_model(args.embed_model, device)
            base_cos = _semantic_cosines(model, base_preds, targets)
            lora_cos = _semantic_cosines(model, lora_preds, targets)
        except Exception as e:
            print(f"[compare] WARNING: semantic similarity disabled ({type(e).__name__}: {e}); "
                  f"install sentence-transformers or use --no-semantic.")
            base_cos = lora_cos = None

    merged = []
    metric_names = [
        "word_f1",
        "bleu1",
        "bleu2",
        "bleu4",
        "rouge_l",
        "prefix1",
        "prefix3",
        "prefix5",
        "pred_len",
        "target_len",
        "len_ratio",
    ]
    base_metric_values = {name: [] for name in metric_names}
    lora_metric_values = {name: [] for name in metric_names}

    shown = 0
    for i, k in enumerate(ordered):
        target = targets[i]
        base_pred = base_preds[i]
        lora_pred = lora_preds[i]

        base_metrics = _all_direct_metrics(base_pred, target)
        lora_metrics = _all_direct_metrics(lora_pred, target)

        for name in metric_names:
            base_metric_values[name].append(base_metrics[name])
            lora_metric_values[name].append(lora_metrics[name])

        rec = {
            "video_id": k[0],
            "video_window": list(k[1]) if isinstance(k[1], tuple) else k[1],
            "idx": k[2],
            "target": target,
            "base_pred": base_pred,
            "lora_pred": lora_pred,
        }

        # Backward-compatible aliases.
        rec["base_target_f1"] = round(base_metrics["word_f1"], 3)
        rec["lora_target_f1"] = round(lora_metrics["word_f1"], 3)

        for name in metric_names:
            rec[f"base_{name}"] = round(base_metrics[name], 3)
            rec[f"lora_{name}"] = round(lora_metrics[name], 3)

        bc = lc = None
        if base_cos is not None and lora_cos is not None:
            bc = base_cos[i]
            lc = lora_cos[i]
            rec["base_target_cos"] = round(bc, 3)
            rec["lora_target_cos"] = round(lc, 3)

        merged.append(rec)

        if args.limit is None or shown < args.limit:
            shown += 1
            print("=" * 90)
            print(f"[{k[0]} | window={rec['video_window']} | idx={k[2]}]")
            print(f"  TARGET   : {target}")
            if bc is not None:
                print(
                    f"  BASE     : {base_pred}   "
                    f"(f1={base_metrics['word_f1']:.3f} rougeL={base_metrics['rouge_l']:.3f} "
                    f"bleu1={base_metrics['bleu1']:.3f} cos={bc:.3f})"
                )
                print(
                    f"  PRETRAIN : {lora_pred}   "
                    f"(f1={lora_metrics['word_f1']:.3f} rougeL={lora_metrics['rouge_l']:.3f} "
                    f"bleu1={lora_metrics['bleu1']:.3f} cos={lc:.3f})"
                )
            else:
                print(
                    f"  BASE     : {base_pred}   "
                    f"(f1={base_metrics['word_f1']:.3f} rougeL={base_metrics['rouge_l']:.3f} "
                    f"bleu1={base_metrics['bleu1']:.3f})"
                )
                print(
                    f"  PRETRAIN : {lora_pred}   "
                    f"(f1={lora_metrics['word_f1']:.3f} rougeL={lora_metrics['rouge_l']:.3f} "
                    f"bleu1={lora_metrics['bleu1']:.3f})"
                )

    summary = {}
    if ordered:
        print("=" * 90)
        for name in ["word_f1", "bleu1", "bleu2", "bleu4", "rouge_l", "prefix1", "prefix3", "prefix5", "len_ratio"]:
            _print_metric(name, base_metric_values[name], lora_metric_values[name])
            summary[name] = {
                "base": _mean(base_metric_values[name]),
                "lora": _mean(lora_metric_values[name]),
                "delta": _mean(lora_metric_values[name]) - _mean(base_metric_values[name]),
            }

        print("[compare] mean lengths:")
        print(f"    TARGET           : {_mean(base_metric_values['target_len']):.3f}")
        print(f"    BASE prediction  : {_mean(base_metric_values['pred_len']):.3f}")
        print(f"    PRETRAIN pred    : {_mean(lora_metric_values['pred_len']):.3f}")

        summary["length"] = {
            "target": _mean(base_metric_values["target_len"]),
            "base_pred": _mean(base_metric_values["pred_len"]),
            "lora_pred": _mean(lora_metric_values["pred_len"]),
        }

        if base_cos is not None and lora_cos is not None:
            print(f"[compare] mean semantic cosine vs target:")
            print(f"    BASE (untrained) : {_mean(base_cos):.3f}")
            print(f"    PRETRAIN adapter : {_mean(lora_cos):.3f}")
            print(f"    delta            : {(_mean(lora_cos) - _mean(base_cos)):+.3f}")
            summary["semantic_cosine"] = {
                "base": _mean(base_cos),
                "lora": _mean(lora_cos),
                "delta": _mean(lora_cos) - _mean(base_cos),
            }

    if args.output:
        out = Path(args.output)
        out.parent.mkdir(parents=True, exist_ok=True)
        with out.open("w", encoding="utf-8") as f:
            for rec in merged:
                f.write(json.dumps(rec, ensure_ascii=False) + "\n")
        print(f"[compare] wrote merged comparison: {out}")

    if args.summary_json:
        out = Path(args.summary_json)
        out.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "base": args.base,
            "lora": args.lora,
            "matched": len(ordered),
            "summary": summary,
        }
        out.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"[compare] wrote summary JSON: {out}")


if __name__ == "__main__":
    main()