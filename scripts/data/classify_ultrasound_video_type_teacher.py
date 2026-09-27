#!/usr/bin/env python3
"""Classify ultrasound training videos with a video-capable VLM teacher.

This script is OpenAI-compatible and can be used with:
  - local Qwen/Qwen3.5-35B-A3B (must be a vision-capable endpoint)
  - Gemini 3 Pro via OpenRouter or Google OpenAI-compatible endpoint

It sends each video as an OpenAI-compatible `video_url` content block, following
Qwen3.5/vLLM and Gemini/OpenRouter style APIs. It outputs one JSONL row per
video with video type, anatomy regions, clinical scenarios, scan targets,
language evidence, and stage-specific keep flags.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import time
from pathlib import Path
from typing import Any, Dict, List


LABELS = {
    "hands_on_ultrasound_teaching",
    "pure_ultrasound_scan",
    "ultrasound_ppt_lecture",
    "mixed_ultrasound_teaching",
    "ultrasound_image_discussion",
    "non_ultrasound_or_irrelevant",
    "uncertain",
}

ANATOMY_REGIONS = {
    "lung", "cardiac", "abdomen", "fast_eFAST", "renal", "hepatobiliary",
    "gallbladder", "aorta", "pelvis_obgyn", "vascular", "msk",
    "thyroid_neck", "breast", "ocular", "nerve_regional_anesthesia",
    "procedure_guidance", "pediatric", "soft_tissue", "unknown",
}

CLINICAL_SCENARIOS = {
    "pneumothorax_assessment", "pleural_effusion_assessment", "pulmonary_edema_b_lines",
    "cardiac_function", "pericardial_effusion", "ivc_volume_status", "dvt_assessment",
    "vascular_access", "fast_trauma", "renal_hydronephrosis", "gallstones_cholecystitis",
    "aortic_aneurysm", "early_pregnancy", "fetal_scan", "msk_injury", "nerve_block",
    "needle_guidance", "procedure_guidance", "general_scanning_tutorial", "case_discussion",
    "unknown",
}

PROMPT = """You are classifying videos for a live ultrasound video understanding dataset.

You will receive one ultrasound-related video. Classify the WHOLE VIDEO into exactly one label:

1. hands_on_ultrasound_teaching: A clinician/teacher demonstrates ultrasound scanning with probe/patient/machine and explains the scan. Real-time ultrasound footage is substantial.
2. pure_ultrasound_scan: Mostly or entirely ultrasound machine output / cine loop. No significant slides, talking head, web pages, or non-ultrasound content.
3. ultrasound_ppt_lecture: Mainly slides, lecture notes, diagrams, text-heavy teaching, or static presentation material. May contain ultrasound screenshots.
4. mixed_ultrasound_teaching: Useful ultrasound scan footage mixed with slides, talking head, software UI, web pages, diagrams, or other non-scan content.
5. ultrasound_image_discussion: Mainly static ultrasound images, annotated screenshots, case images, or still frames, not real-time scanning.
6. non_ultrasound_or_irrelevant: Not ultrasound, irrelevant, advertisement, unrelated medical talk, anatomy animation without ultrasound, or unusable quality.
7. uncertain: Cannot determine confidently from the sampled frames.

Also identify anatomy regions / organ systems. Use only these labels, as a JSON array:
lung, cardiac, abdomen, fast_eFAST, renal, hepatobiliary, gallbladder, aorta, pelvis_obgyn, vascular, msk, thyroid_neck, breast, ocular, nerve_regional_anesthesia, procedure_guidance, pediatric, soft_tissue, unknown.

Also identify clinical scenarios. Use only these labels, as a JSON array:
pneumothorax_assessment, pleural_effusion_assessment, pulmonary_edema_b_lines, cardiac_function, pericardial_effusion, ivc_volume_status, dvt_assessment, vascular_access, fast_trauma, renal_hydronephrosis, gallstones_cholecystitis, aortic_aneurysm, early_pregnancy, fetal_scan, msk_injury, nerve_block, needle_guidance, procedure_guidance, general_scanning_tutorial, case_discussion, unknown.

Also infer the video's primary spoken / teaching language if visible text or captions make it possible. Use ISO-639-1 when confident, e.g. "en", "de", "zh", "es", "fr". Use "unknown" if it cannot be inferred from the sampled frames.

Return JSON only with this schema:
{
  "label": "hands_on_ultrasound_teaching | pure_ultrasound_scan | ultrasound_ppt_lecture | mixed_ultrasound_teaching | ultrasound_image_discussion | non_ultrasound_or_irrelevant | uncertain",
  "confidence": 0.0,
  "anatomy_regions": ["lung"],
  "clinical_scenarios": ["pneumothorax_assessment"],
  "scan_views_or_targets": ["pleural line", "lung sliding"],
  "spoken_language": "en | de | zh | es | fr | unknown",
  "language_evidence": "brief evidence, e.g. visible English slide text or captions; empty if unknown",
  "has_realtime_ultrasound": true,
  "has_probe_or_patient": true,
  "has_ppt_or_slides": false,
  "has_talking_head": true,
  "has_static_ultrasound_images": false,
  "has_non_ultrasound_content": false,
  "ultrasound_fraction_estimate": 0.75,
  "teaching_value": "high | medium | low | none",
  "needs_clipping": false,
  "visual_evidence": "brief visual evidence from sampled frames",
  "other_reason": "if label is non_ultrasound_or_irrelevant, explain why; otherwise empty",
  "keep_for_pretrain": true,
  "keep_for_compression": true,
  "keep_for_sft": true
}

Default keep policy:
- hands_on_ultrasound_teaching: keep for all stages
- pure_ultrasound_scan: keep for all stages
- ultrasound_ppt_lecture: keep_for_pretrain=true, keep_for_compression=false, keep_for_sft=false
- mixed_ultrasound_teaching: keep_for_pretrain=true, keep_for_compression=false unless useful scan footage dominates; keep_for_sft=false unless it is mostly real-time scan; needs_clipping=true if mixed
- ultrasound_image_discussion: keep_for_pretrain=true, keep_for_compression=false by default, keep_for_sft=false by default
- non_ultrasound_or_irrelevant: drop for all stages
- uncertain: keep_for_pretrain=true for audit, keep_for_compression=false, keep_for_sft=false
"""


def load_json(path: Path) -> Any:
    with path.open(encoding="utf-8") as f:
        return json.load(f)


def resolve_path(path: str, repo_root: Path) -> Path:
    p = Path(path)
    return p if p.is_absolute() else repo_root / p


def load_video_url_map(path: Path | None) -> Dict[str, str]:
    if path is None:
        return {}
    return {str(k): str(v) for k, v in load_json(path).items()}


def build_video_url(video_id: str, video_path: Path, *, video_url_base: str | None, video_url_map: Dict[str, str]) -> str:
    """Return a URL usable by video_url.

    Preferred options:
      1. explicit --video-url-map {video_id: url}
      2. --video-url-base, e.g. http://node:9000, joined with basename
      3. file:// absolute path fallback for local backends that support it
    """
    if video_id in video_url_map:
        return video_url_map[video_id]
    if video_url_base:
        return video_url_base.rstrip("/") + "/" + video_path.name
    return video_path.resolve().as_uri()


def parse_json(text: str) -> Dict[str, Any]:
    text = (text or "").strip()
    text = re.sub(r"^```(?:json)?\s*", "", text)
    text = re.sub(r"\s*```$", "", text)
    try:
        return json.loads(text)
    except Exception:
        m = re.search(r"\{.*\}", text, flags=re.S)
        if not m:
            raise
        return json.loads(m.group(0))


def clean_list(values, allowed: set[str]) -> List[str]:
    out = []
    if isinstance(values, str):
        values = [values]
    if isinstance(values, list):
        for v in values:
            s = str(v).strip()
            if s in allowed and s not in out:
                out.append(s)
    return out or ["unknown"]


def apply_keep_policy(rec: Dict[str, Any]) -> None:
    label = rec.get("label")
    if label in {"hands_on_ultrasound_teaching", "pure_ultrasound_scan"}:
        rec["keep_for_pretrain"] = True
        rec["keep_for_compression"] = True
        rec["keep_for_sft"] = True
        rec.setdefault("needs_clipping", False)
    elif label == "ultrasound_ppt_lecture":
        rec["keep_for_pretrain"] = True
        rec["keep_for_compression"] = False
        rec["keep_for_sft"] = False
        rec.setdefault("needs_clipping", False)
    elif label == "mixed_ultrasound_teaching":
        rec["keep_for_pretrain"] = True
        rec["keep_for_compression"] = bool(rec.get("keep_for_compression", False))
        rec["keep_for_sft"] = bool(rec.get("keep_for_sft", False))
        rec["needs_clipping"] = True if rec.get("needs_clipping") is None else bool(rec.get("needs_clipping"))
    elif label == "ultrasound_image_discussion":
        rec["keep_for_pretrain"] = True
        rec["keep_for_compression"] = False
        rec["keep_for_sft"] = False
        rec.setdefault("needs_clipping", False)
    elif label == "non_ultrasound_or_irrelevant":
        rec["keep_for_pretrain"] = False
        rec["keep_for_compression"] = False
        rec["keep_for_sft"] = False
        rec.setdefault("needs_clipping", False)
    else:
        rec["label"] = "uncertain"
        rec["keep_for_pretrain"] = True
        rec["keep_for_compression"] = False
        rec["keep_for_sft"] = False
        rec.setdefault("needs_clipping", False)


def normalize_record(raw: Dict[str, Any]) -> Dict[str, Any]:
    rec = dict(raw)
    label = str(rec.get("label") or "uncertain").strip()
    rec["label"] = label if label in LABELS else "uncertain"
    rec["confidence"] = max(0.0, min(1.0, float(rec.get("confidence") or 0.0)))
    rec["anatomy_regions"] = clean_list(rec.get("anatomy_regions"), ANATOMY_REGIONS)
    rec["clinical_scenarios"] = clean_list(rec.get("clinical_scenarios"), CLINICAL_SCENARIOS)
    views = rec.get("scan_views_or_targets") or []
    rec["scan_views_or_targets"] = [str(x).strip() for x in views if str(x).strip()] if isinstance(views, list) else []
    lang = str(rec.get("spoken_language") or "unknown").strip().lower()
    rec["spoken_language"] = lang if re.fullmatch(r"[a-z]{2}|unknown", lang) else "unknown"
    rec["language_evidence"] = str(rec.get("language_evidence") or "")
    for key in ["has_realtime_ultrasound", "has_probe_or_patient", "has_ppt_or_slides", "has_talking_head", "has_static_ultrasound_images", "has_non_ultrasound_content", "needs_clipping"]:
        rec[key] = bool(rec.get(key, False))
    rec["ultrasound_fraction_estimate"] = max(0.0, min(1.0, float(rec.get("ultrasound_fraction_estimate") or 0.0)))
    rec["teaching_value"] = str(rec.get("teaching_value") or "none")
    rec["visual_evidence"] = str(rec.get("visual_evidence") or "")
    rec["other_reason"] = str(rec.get("other_reason") or "")
    apply_keep_policy(rec)
    return rec


def classify_one(client, model: str, video_url: str, max_tokens: int, video_fps: float) -> Dict[str, Any]:
    content = [
        {"type": "video_url", "video_url": {"url": video_url}},
        {"type": "text", "text": PROMPT},
    ]
    kwargs = dict(
        model=model,
        messages=[{"role": "user", "content": content}],
        temperature=0,
        max_tokens=max_tokens,
    )
    if video_fps > 0:
        # Qwen3.5/vLLM official examples use `fps` under mm_processor_kwargs.
        # Do not pass `do_sample_frames`: in vLLM 0.30.0 it is forwarded into
        # Qwen3VLProcessor and can trigger a BadRequestError for unsupported
        # processor kwargs.
        kwargs["extra_body"] = {"mm_processor_kwargs": {"fps": float(video_fps)}}
    resp = client.chat.completions.create(**kwargs)
    return normalize_record(parse_json(resp.choices[0].message.content))


def parse_args():
    p = argparse.ArgumentParser(description="Classify ultrasound videos with Qwen/Gemini teacher VLM")
    p.add_argument("--video-map", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--teacher", choices=["qwen35", "gemini3", "custom"], default="qwen35")
    p.add_argument("--model", default="Qwen/Qwen3.5-35B-A3B")
    p.add_argument("--base-url", default="http://localhost:8000/v1")
    p.add_argument("--api-key-env", default="VLLM_API_KEY")
    p.add_argument("--repo-root", type=Path, default=Path("."))
    p.add_argument("--video-fps", type=float, default=1.0, help="Video sampling fps passed via extra_body.mm_processor_kwargs. Use 0 to omit.")
    p.add_argument("--video-url-base", default=None, help="HTTP base URL serving video files; final URL is base/basename.mp4")
    p.add_argument("--video-url-map", type=Path, default=None, help="Optional JSON map {video_id: video_url}")
    p.add_argument("--max-tokens", type=int, default=900)
    p.add_argument("--limit", type=int, default=None)
    p.add_argument("--sleep-sec", type=float, default=0.0)
    p.add_argument("--resume", action="store_true")
    return p.parse_args()


def main():
    args = parse_args()
    from openai import OpenAI

    api_key = os.environ.get(args.api_key_env) or "EMPTY"
    client = OpenAI(api_key=api_key, base_url=args.base_url) if args.base_url else OpenAI(api_key=api_key)
    video_map = {str(k): str(v) for k, v in load_json(args.video_map).items()}
    video_url_map = load_video_url_map(args.video_url_map)
    items = sorted(video_map.items())
    if args.limit is not None:
        items = items[: args.limit]

    done: set[str] = set()
    if args.resume and args.output.exists():
        with args.output.open(encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    done.add(str(json.loads(line).get("video_id")))

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("a" if args.resume else "w", encoding="utf-8") as out:
        for idx, (video_id, raw_path) in enumerate(items, 1):
            if video_id in done:
                continue
            video_path = resolve_path(raw_path, args.repo_root)
            base = {"video_id": video_id, "video_path": str(video_path), "teacher": args.teacher, "model": args.model, "source_video_map": str(args.video_map)}
            print(f"[{idx}/{len(items)}] {video_id} teacher={args.teacher} model={args.model}")
            try:
                video_url = build_video_url(video_id, video_path, video_url_base=args.video_url_base, video_url_map=video_url_map)
                rec = classify_one(client, args.model, video_url, args.max_tokens, args.video_fps)
                rec.update(base)
                rec["video_url"] = video_url
                rec["video_fps"] = args.video_fps
                rec["error"] = None
            except Exception as exc:
                rec = {
                    **base,
                    "label": "uncertain",
                    "confidence": 0.0,
                    "anatomy_regions": ["unknown"],
                    "clinical_scenarios": ["unknown"],
                    "scan_views_or_targets": [],
                    "visual_evidence": "classification_failed",
                    "other_reason": f"{type(exc).__name__}: {exc}",
                    "keep_for_pretrain": True,
                    "keep_for_compression": False,
                    "keep_for_sft": False,
                    "needs_clipping": False,
                    "error": f"{type(exc).__name__}: {exc}",
                }
                print(f"  ERROR: {rec['error']}")
            out.write(json.dumps(rec, ensure_ascii=False) + "\n")
            out.flush()
            if args.sleep_sec > 0:
                time.sleep(args.sleep_sec)


if __name__ == "__main__":
    main()