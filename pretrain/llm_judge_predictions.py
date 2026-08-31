#!/usr/bin/env python3
"""Blind LLM judge for pretrain base-vs-LoRA narration predictions.

The judge sees only anonymous "Prediction A" and "Prediction B" plus past
context and ground truth.  The script randomizes A/B order per sample and maps
the judge decision back to ``base`` / ``pretrain`` only in the saved output and
summary.
"""

from __future__ import annotations

import argparse
import json
import random
import re
import sys
import time
from collections import Counter
from pathlib import Path
from statistics import mean
from typing import Any, Dict, Iterable, List, Tuple


_REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_REPO_ROOT / "scripts"))
from _video_llm import build_openrouter_client, call_with_content, text_block  # noqa: E402


JUDGE_PROMPT = """You are an expert evaluator for ultrasound teaching-video narration prediction.

You will be given:
- Past narration context from an ultrasound teaching video
- The ground-truth next/current narration sentence
- Two anonymous model predictions: Prediction A and Prediction B

Your job is to decide which prediction is better.

Evaluate based on:
1. Semantic closeness to the ground truth
2. Correct ultrasound / medical terminology
3. Consistency with the past narration context
4. Appropriate ultrasound teaching narration style
5. Conciseness and absence of generic hallucination

Important:
- Do NOT assume either prediction is from a better model.
- Do NOT prefer A or B by position.
- If both are similarly good, choose "tie".
- If both are clearly poor, choose "both_bad".

Output STRICT JSON only:
{
  "winner": "A" | "B" | "tie" | "both_bad",
  "score_A": 1 | 2 | 3 | 4 | 5,
  "score_B": 1 | 2 | 3 | 4 | 5,
  "reason": "...",
  "errors_A": ["..."],
  "errors_B": ["..."]
}
"""


def _load_jsonl(path: Path) -> List[Dict[str, Any]]:
    rows = []
    with path.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def _key(rec: Dict[str, Any]) -> Tuple[Any, Any, Any]:
    vw = rec.get("video_window")
    vw = tuple(vw) if isinstance(vw, list) else vw
    return (rec.get("video_id"), vw, rec.get("idx"))


def _truncate(text: str | None, max_chars: int) -> str:
    text = (text or "").strip()
    if max_chars <= 0 or len(text) <= max_chars:
        return text
    return text[-max_chars:].lstrip()


def _extract_json(text: str) -> Dict[str, Any]:
    raw = (text or "").strip()
    raw = re.sub(r"^```(?:json)?\s*", "", raw, flags=re.IGNORECASE)
    raw = re.sub(r"\s*```$", "", raw)
    try:
        return json.loads(raw)
    except Exception:
        m = re.search(r"\{.*\}", raw, flags=re.DOTALL)
        if not m:
            raise
        return json.loads(m.group(0))


def _sanitize_judge(payload: Dict[str, Any]) -> Dict[str, Any]:
    winner = str(payload.get("winner", "tie")).strip()
    if winner not in {"A", "B", "tie", "both_bad"}:
        winner = "tie"

    def score(name: str) -> int:
        try:
            return max(1, min(5, int(payload.get(name, 3))))
        except Exception:
            return 3

    def list_field(name: str) -> List[str]:
        value = payload.get(name, [])
        if isinstance(value, list):
            return [str(x) for x in value]
        if value:
            return [str(value)]
        return []

    return {
        "winner": winner,
        "score_A": score("score_A"),
        "score_B": score("score_B"),
        "reason": str(payload.get("reason", "")),
        "errors_A": list_field("errors_A"),
        "errors_B": list_field("errors_B"),
    }


def _resolve_winner(winner: str, assignment: Dict[str, str]) -> str:
    if winner in {"tie", "both_bad"}:
        return winner
    return assignment.get(winner, "tie")


def _build_prompt(
    *,
    prev_context: str,
    target: str,
    pred_a: str,
    pred_b: str,
    max_context_chars: int,
    max_target_chars: int,
    max_pred_chars: int,
) -> str:
    return f"""{JUDGE_PROMPT}

Past narration context:
{_truncate(prev_context, max_context_chars)}

Ground-truth narration:
{_truncate(target, max_target_chars)}

Prediction A:
{_truncate(pred_a, max_pred_chars)}

Prediction B:
{_truncate(pred_b, max_pred_chars)}
"""


def _load_completed(path: Path) -> set[Tuple[Any, Any, Any]]:
    done = set()
    if not path.exists():
        return done
    with path.open(encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            try:
                done.add(_key(json.loads(line)))
            except Exception:
                continue
    return done


def _write_summary(path: Path, rows: Iterable[Dict[str, Any]], *, base_path: str, lora_path: str, model: str) -> None:
    rows = list(rows)
    winners = Counter(r.get("resolved_winner", "parse_error") for r in rows)
    positions = Counter()
    base_scores = []
    pretrain_scores = []
    parse_errors = 0

    for r in rows:
        assignment = r.get("assignment") or {}
        judge = r.get("judge") or {}
        if r.get("parse_error"):
            parse_errors += 1
        for pos, name in assignment.items():
            positions[pos] += 1
            score = judge.get(f"score_{pos}")
            if isinstance(score, int):
                if name == "base":
                    base_scores.append(score)
                elif name == "pretrain":
                    pretrain_scores.append(score)

    total = len(rows)
    summary = {
        "base": base_path,
        "lora": lora_path,
        "judge_model": model,
        "blind": True,
        "judged": total,
        "parse_errors": parse_errors,
        "winner_counts": dict(winners),
        "win_rates": {k: (v / total if total else 0.0) for k, v in winners.items()},
        "mean_scores": {
            "base": mean(base_scores) if base_scores else None,
            "pretrain": mean(pretrain_scores) if pretrain_scores else None,
            "delta_pretrain_minus_base": (
                mean(pretrain_scores) - mean(base_scores)
                if base_scores and pretrain_scores else None
            ),
        },
        "position_counts": dict(positions),
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Blind LLM judge for pretrain base-vs-LoRA predictions")
    parser.add_argument("--base", type=Path, required=True, help="Base inference jsonl")
    parser.add_argument("--lora", type=Path, required=True, help="Pretrain LoRA inference jsonl")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--summary-json", type=Path, required=True)
    parser.add_argument("--model", default="google/gemini-2.5-flash")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument(
        "--sample-size",
        type=int,
        default=None,
        help="Randomly sample this many matched rows before judging. Prefer this over --limit for representative LLM judge eval.",
    )
    parser.add_argument(
        "--sample-seed",
        type=int,
        default=42,
        help="Random seed for --sample-size row sampling.",
    )
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--no-randomize-order", action="store_true")
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--max-tokens", type=int, default=512)
    parser.add_argument("--max-context-chars", type=int, default=2000)
    parser.add_argument("--max-target-chars", type=int, default=800)
    parser.add_argument("--max-pred-chars", type=int, default=800)
    parser.add_argument("--retries", type=int, default=5)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.resume and args.overwrite:
        raise ValueError("Use only one of --resume or --overwrite")

    base_rows = _load_jsonl(args.base)
    lora_rows = _load_jsonl(args.lora)
    lora_by_key = {_key(r): r for r in lora_rows}
    matched = [(b, lora_by_key[_key(b)]) for b in base_rows if _key(b) in lora_by_key]

    if args.sample_size is not None:
        if args.sample_size <= 0:
            raise ValueError(f"--sample-size must be positive, got {args.sample_size}")
        sample_rng = random.Random(args.sample_seed)
        if args.sample_size < len(matched):
            matched = sample_rng.sample(matched, args.sample_size)

    if args.limit is not None:
        matched = matched[:args.limit]

    completed = _load_completed(args.output) if args.resume else set()
    if args.output.exists() and not args.resume and not args.overwrite:
        raise FileExistsError(f"Output exists: {args.output}. Use --resume or --overwrite.")

    client = build_openrouter_client()
    rng = random.Random(args.seed)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    mode = "a" if args.resume else "w"

    new_rows = []
    with args.output.open(mode, encoding="utf-8") as f:
        for i, (base, lora) in enumerate(matched, start=1):
            k = _key(base)
            if k in completed:
                continue

            base_pred = base.get("prediction", "")
            lora_pred = lora.get("prediction", "")
            if args.no_randomize_order or rng.random() < 0.5:
                pred_a, pred_b = base_pred, lora_pred
                assignment = {"A": "base", "B": "pretrain"}
            else:
                pred_a, pred_b = lora_pred, base_pred
                assignment = {"A": "pretrain", "B": "base"}

            prompt = _build_prompt(
                prev_context=base.get("prev_context", ""),
                target=base.get("target", ""),
                pred_a=pred_a,
                pred_b=pred_b,
                max_context_chars=args.max_context_chars,
                max_target_chars=args.max_target_chars,
                max_pred_chars=args.max_pred_chars,
            )
            t0 = time.time()
            raw, usage = call_with_content(
                client,
                content_blocks=[text_block(prompt)],
                model=args.model,
                temperature=args.temperature,
                max_tokens=args.max_tokens,
                retries=args.retries,
            )
            elapsed = time.time() - t0

            parse_error = None
            try:
                judge = _sanitize_judge(_extract_json(raw))
            except Exception as exc:  # noqa: BLE001
                parse_error = f"{type(exc).__name__}: {exc}"
                judge = {"winner": "tie", "score_A": 3, "score_B": 3, "reason": "parse_error", "errors_A": [], "errors_B": []}

            rec = {
                "idx": base.get("idx"),
                "video_id": base.get("video_id"),
                "video_window": base.get("video_window"),
                "prev_context": base.get("prev_context", ""),
                "target": base.get("target", ""),
                "prediction_A": pred_a,
                "prediction_B": pred_b,
                "assignment": assignment,
                "judge": judge,
                "resolved_winner": _resolve_winner(judge["winner"], assignment),
                "judge_model": args.model,
                "elapsed_sec": round(elapsed, 2),
                "usage": usage,
                "raw_response": raw,
            }
            if parse_error:
                rec["parse_error"] = parse_error

            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
            f.flush()
            new_rows.append(rec)
            print(f"[judge] {i}/{len(matched)} idx={rec['idx']} winner={rec['resolved_winner']} elapsed={elapsed:.1f}s")

    all_rows = _load_jsonl(args.output)
    _write_summary(args.summary_json, all_rows, base_path=str(args.base), lora_path=str(args.lora), model=args.model)
    print(f"[judge] wrote {args.output}")
    print(f"[judge] wrote {args.summary_json}")


if __name__ == "__main__":
    main()