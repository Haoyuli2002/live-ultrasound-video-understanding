#!/usr/bin/env python3
"""Merge Qwen and Gemini ultrasound video-type teacher labels.

The output is a final audit JSONL with final_label, agreement,
needs_human_review, merged anatomy/scenario fields, and final keep flags.
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List


KEEP_LABELS_ALL = {"hands_on_ultrasound_teaching", "pure_ultrasound_scan"}
KEEP_PRETRAIN_ONLY = {"ultrasound_ppt_lecture", "ultrasound_image_discussion", "mixed_ultrasound_teaching", "uncertain"}


def load_jsonl(path: Path) -> Dict[str, Dict[str, Any]]:
    rows: Dict[str, Dict[str, Any]] = {}
    if not path or not path.exists():
        return rows
    with path.open(encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            rec = json.loads(line)
            vid = str(rec.get("video_id") or "")
            if vid:
                rows[vid] = rec
    return rows


def union_list(*lists) -> List[str]:
    out = []
    for vals in lists:
        if isinstance(vals, str):
            vals = [vals]
        if not isinstance(vals, list):
            continue
        for v in vals:
            s = str(v).strip()
            if s and s not in out and s != "unknown":
                out.append(s)
    return out or ["unknown"]


def confidence(rec: Dict[str, Any] | None) -> float:
    if not rec:
        return 0.0
    try:
        return float(rec.get("confidence") or 0.0)
    except Exception:
        return 0.0


def label(rec: Dict[str, Any] | None) -> str | None:
    if not rec:
        return None
    return str(rec.get("label") or "uncertain")


def keep_policy(final_label: str, needs_clipping: bool) -> Dict[str, bool]:
    if final_label in KEEP_LABELS_ALL:
        return {"keep_for_pretrain": True, "keep_for_compression": True, "keep_for_sft": True}
    if final_label == "mixed_ultrasound_teaching":
        return {"keep_for_pretrain": True, "keep_for_compression": False, "keep_for_sft": False, "needs_clipping": True}
    if final_label in KEEP_PRETRAIN_ONLY:
        return {"keep_for_pretrain": True, "keep_for_compression": False, "keep_for_sft": False}
    return {"keep_for_pretrain": False, "keep_for_compression": False, "keep_for_sft": False}


def choose_final(primary: Dict[str, Any] | None, validator: Dict[str, Any] | None, *, high_conf: float, margin: float) -> Dict[str, Any]:
    lp, lv = label(primary), label(validator)
    cp, cv = confidence(primary), confidence(validator)
    if primary and validator and lp == lv:
        return {"final_label": lp, "agreement": True, "needs_human_review": False, "resolution": "agreement"}
    if primary and not validator:
        return {"final_label": lp or "uncertain", "agreement": None, "needs_human_review": cp < high_conf or lp == "uncertain", "resolution": "primary_only"}
    if validator and not primary:
        return {"final_label": lv or "uncertain", "agreement": None, "needs_human_review": cv < high_conf or lv == "uncertain", "resolution": "validator_only"}
    if not primary and not validator:
        return {"final_label": "uncertain", "agreement": None, "needs_human_review": True, "resolution": "missing_both"}

    # Disagreement.
    if abs(cp - cv) >= margin:
        if cp > cv:
            chosen = lp
            res = "primary_higher_confidence"
        else:
            chosen = lv
            res = "validator_higher_confidence"
        return {"final_label": chosen or "uncertain", "agreement": False, "needs_human_review": False, "resolution": res}
    if (cp >= high_conf and cv >= high_conf) or lp == "uncertain" or lv == "uncertain":
        return {"final_label": lp or lv or "uncertain", "agreement": False, "needs_human_review": True, "resolution": "high_confidence_or_uncertain_disagreement"}
    return {"final_label": lp or lv or "uncertain", "agreement": False, "needs_human_review": True, "resolution": "low_margin_disagreement"}


def parse_args():
    p = argparse.ArgumentParser(description="Merge Qwen/Gemini ultrasound video type teacher labels")
    p.add_argument("--primary-audit", type=Path, required=True, help="Primary teacher audit, e.g. Qwen3.5 JSONL")
    p.add_argument("--validator-audit", type=Path, default=None, help="Validator teacher audit, e.g. Gemini3 JSONL")
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--output-summary", type=Path, default=None)
    p.add_argument("--primary-name", default="qwen35")
    p.add_argument("--validator-name", default="gemini3")
    p.add_argument("--high-confidence", type=float, default=0.85)
    p.add_argument("--confidence-margin", type=float, default=0.25)
    return p.parse_args()


def main():
    args = parse_args()
    primary = load_jsonl(args.primary_audit)
    validator = load_jsonl(args.validator_audit) if args.validator_audit else {}
    video_ids = sorted(set(primary) | set(validator))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    label_counts = Counter()
    anatomy_counts = Counter()
    scenario_counts = Counter()
    review_count = 0
    agreement_count = 0
    with args.output.open("w", encoding="utf-8") as out:
        for vid in video_ids:
            p_rec = primary.get(vid)
            v_rec = validator.get(vid)
            final = choose_final(p_rec, v_rec, high_conf=args.high_confidence, margin=args.confidence_margin)
            anatomy = union_list((p_rec or {}).get("anatomy_regions"), (v_rec or {}).get("anatomy_regions"))
            scenarios = union_list((p_rec or {}).get("clinical_scenarios"), (v_rec or {}).get("clinical_scenarios"))
            views = union_list((p_rec or {}).get("scan_views_or_targets"), (v_rec or {}).get("scan_views_or_targets"))
            needs_clipping = bool((p_rec or {}).get("needs_clipping")) or bool((v_rec or {}).get("needs_clipping"))
            keep = keep_policy(final["final_label"], needs_clipping)
            rec = {
                "video_id": vid,
                "video_path": (p_rec or v_rec or {}).get("video_path"),
                args.primary_name: p_rec,
                args.validator_name: v_rec,
                **final,
                "anatomy_regions": anatomy,
                "clinical_scenarios": scenarios,
                "scan_views_or_targets": views,
                "needs_clipping": needs_clipping or bool(keep.get("needs_clipping", False)),
                **{k: bool(v) for k, v in keep.items() if k != "needs_clipping"},
            }
            out.write(json.dumps(rec, ensure_ascii=False) + "\n")
            label_counts[rec["final_label"]] += 1
            for a in anatomy:
                anatomy_counts[a] += 1
            for s in scenarios:
                scenario_counts[s] += 1
            review_count += int(bool(rec["needs_human_review"]))
            agreement_count += int(rec.get("agreement") is True)
    summary = {
        "num_videos": len(video_ids),
        "label_counts": dict(label_counts),
        "anatomy_region_counts": dict(anatomy_counts),
        "clinical_scenario_counts": dict(scenario_counts),
        "teacher_agreement_count": agreement_count,
        "teacher_agreement_rate": agreement_count / len(video_ids) if video_ids else 0.0,
        "needs_human_review_count": review_count,
        "output": str(args.output),
    }
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    if args.output_summary:
        args.output_summary.parent.mkdir(parents=True, exist_ok=True)
        args.output_summary.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()