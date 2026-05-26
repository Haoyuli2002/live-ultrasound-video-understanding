"""
Ablation Study: Input Modes for Qwen2-VL-2B Video Filtering
=============================================================
Mode 1: Single image per call (current approach) — N calls
Mode 2: Multi-image one call — 1 call  
Mode 3: Native video mode — 1 call

Usage:
    python ablation_input_modes.py
    python ablation_input_modes.py --video PATH --num-frames 4
"""
import os
os.environ["FORCE_QWENVL_VIDEO_READER"] = "torchcodec"
import json, time, re, argparse
from pathlib import Path
import cv2, numpy as np
from PIL import Image

# Config
DEFAULT_VIDEO = "UltrasoundCrawler_KeyCode_20260323_v2/output/20260520_162816_youtube/media/case_reasoning/8V649L5Q368.mp4"
NUM_FRAMES = 8
MODEL_NAME = "Qwen/Qwen3-VL-2B-Instruct"

# === Prompts ===

SINGLE_FRAME_PROMPT = """Analyze this medical image. Output JSON:
- frame_type: "ultrasound" | "ppt_slide" | "lecture_talking_head" | "other"
- ultrasound_mode: "B-mode" | "Color Doppler" | "M-mode" | "none"
- annotation_level: "none" | "machine_params_only" | "light_annotation" | "heavy_annotation"
- has_face: true/false
- anatomy: region or null
- confidence: 0.0-1.0
- description: brief

Example: {"frame_type":"ultrasound","ultrasound_mode":"B-mode","annotation_level":"machine_params_only","has_face":false,"anatomy":"kidney","confidence":0.9,"description":"B-mode right kidney longitudinal"}
Example: {"frame_type":"ppt_slide","ultrasound_mode":"none","annotation_level":"heavy_annotation","has_face":false,"anatomy":null,"confidence":0.95,"description":"Teaching slide with diagram"}
Example: {"frame_type":"lecture_talking_head","ultrasound_mode":"none","annotation_level":"none","has_face":true,"anatomy":null,"confidence":0.9,"description":"Doctor speaking to camera"}

Output only JSON:"""

VIDEO_CLASSIFY_PROMPT = """Based on these {n} frames sampled uniformly from a medical video, classify the overall video content.

Video types:
1. "pure_ultrasound" - Only ultrasound machine screen visible, no faces, no slides
2. "hands_on_tutorial" - Instructor demonstrating probe technique while showing US screen, switching between hand/probe view and ultrasound display
3. "ppt_lecture" - Primarily slides/presentations, instructor may be visible, occasional US images
4. "case_discussion" - Instructor discussing/annotating ultrasound images/clips for diagnosis
5. "diagram_animation" - Anatomical diagrams, 3D animations, illustrations (no real US)
6. "mixed" - Multiple content types alternating within the video

Output JSON:
{{"video_type": "pure_ultrasound"|"hands_on_tutorial"|"ppt_lecture"|"case_discussion"|"diagram_animation"|"mixed", "has_probe_technique": true/false, "has_instructor_face": true/false, "has_slides": true/false, "has_annotations_overlay": true/false, "anatomy_regions": [], "description": "brief overall description", "training_value": "high"|"medium"|"low"|"none", "recommendation": "keep"|"trim"|"discard"}}

Output only JSON:"""


VIDEO_MODE_PROMPT = """Analyze this medical video. Classify the overall video content type.

Video types:
1. "pure_ultrasound" - Only ultrasound machine screen visible, no faces, no slides
2. "hands_on_tutorial" - Instructor demonstrating probe technique while showing US screen, switching between hand/probe view and ultrasound display
3. "ppt_lecture" - Primarily slides/presentations, instructor may be visible, occasional US images
4. "case_discussion" - Instructor discussing/annotating ultrasound images/clips for diagnosis
5. "diagram_animation" - Anatomical diagrams, 3D animations, illustrations (no real US)
6. "mixed" - Multiple content types alternating within the video

Output JSON:
{{"video_type": "pure_ultrasound"|"hands_on_tutorial"|"ppt_lecture"|"case_discussion"|"diagram_animation"|"mixed", "has_probe_technique": true/false, "has_instructor_face": true/false, "has_slides": true/false, "anatomy_regions": [], "description": "brief overall description", "training_value": "high"|"medium"|"low"|"none", "recommendation": "keep"|"trim"|"discard"}}

Output only JSON:"""


# === Helpers ===

def parse_json(text):
    """Extract JSON from text."""
    try:
        return json.loads(text.strip())
    except Exception:
        pass
    for pat in [r'```json\s*(.+?)```', r'```\s*(.+?)```']:
        m = re.search(pat, text, re.DOTALL)
        if m:
            try: return json.loads(m.group(1).strip())
            except: continue
    m = re.search(r'[\{\[].*[\}\]]', text, re.DOTALL)
    if m:
        try: return json.loads(m.group(0))
        except: pass
    m = re.search(r'\{[^{}]*\}', text, re.DOTALL)
    if m:
        try: return json.loads(m.group(0))
        except: pass
    return {"_parse_error": True, "_raw": text[:500]}


def sample_frames(video_path, num_frames=8):
    """Sample frames uniformly (excluding first/last 5%)."""
    cap = cv2.VideoCapture(str(video_path))
    assert cap.isOpened(), f"Cannot open: {video_path}"
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    dur = total / fps
    print(f"\nVIDEO: {Path(video_path).stem} | {dur:.0f}s | {fps:.0f}fps | {total} frames")
    indices = np.linspace(int(total*0.05), int(total*0.95), num_frames, dtype=int)
    frames = []
    for idx in indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(idx))
        ret, frame = cap.read()
        if ret:
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frames.append({"image": Image.fromarray(rgb), "timestamp": round(idx/fps, 2), "frame_idx": int(idx)})
    cap.release()
    print(f"  Sampled {len(frames)} frames: {[f['timestamp'] for f in frames]}")
    return frames, {"duration": dur, "fps": fps, "total_frames": total}

def load_model():
    """Load Qwen3-VL."""
    import torch
    from transformers import Qwen3VLForConditionalGeneration, AutoProcessor

    device = "mps" if torch.backends.mps.is_available() else ("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\nLoading {MODEL_NAME} on {device}...")

    model = Qwen3VLForConditionalGeneration.from_pretrained(
        MODEL_NAME,
        torch_dtype="auto",
        device_map="auto" if device == "cuda" else None
    )

    if device in ("mps", "cpu"):
        model = model.to(device)

    model.eval()
    processor = AutoProcessor.from_pretrained(MODEL_NAME)
    print("Model loaded!")
    return model, processor, device

def run_inference(model, processor, device, messages, max_new_tokens=512):
    """Run inference, return (text, time, in_tokens, out_tokens)."""
    import torch
    from qwen_vl_utils import process_vision_info
    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    img_in, vid_in = process_vision_info(messages)
    inputs = processor(text=[text], images=img_in, videos=vid_in, padding=True, return_tensors="pt").to(device)
    n_in = inputs.input_ids.shape[1]
    t0 = time.time()
    with torch.no_grad():
        out_ids = model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=False)
    elapsed = time.time() - t0
    trimmed = [o[len(i):] for i, o in zip(inputs.input_ids, out_ids)]
    raw = processor.batch_decode(trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]
    n_out = out_ids.shape[1] - n_in
    return raw, elapsed, n_in, n_out


# === Mode 1: Single Image per Call ===

def run_mode1(model, processor, device, frames):
    """Each frame analyzed independently."""
    print(f"\n{'='*70}\nMODE 1: Single Image per Call ({len(frames)} calls)\n{'='*70}")
    results, total_t, total_in, total_out = [], 0, 0, 0
    for i, f in enumerate(frames):
        msg = [{"role": "user", "content": [{"type": "image", "image": f["image"]}, {"type": "text", "text": SINGLE_FRAME_PROMPT}]}]
        raw, t, ni, no = run_inference(model, processor, device, msg, 256)
        total_t += t; total_in += ni; total_out += no
        r = parse_json(raw); r["_timestamp"] = f["timestamp"]
        ft = r.get('frame_type', '?'); desc = r.get('description', raw[:40])
        print(f"  [{i+1}/{len(frames)}] {t:.1f}s | {ft:25s} | {desc[:40]}")
        results.append(r)
    print(f"\n  TOTAL: {total_t:.1f}s | Avg: {total_t/len(frames):.1f}s/frame | Tokens: {total_in}in+{total_out}out")
    return {"mode": "mode1_single_image", "results": results, "total_time_sec": round(total_t, 2),
            "avg_time_per_frame": round(total_t/len(frames), 2), "total_input_tokens": total_in,
            "total_output_tokens": total_out, "num_calls": len(frames)}


# === Mode 2: Multi-Image One Call ===

def run_mode2(model, processor, device, frames, max_images=4):
    """Multi-image video-level classification. Limited frames to avoid OOM."""
    import torch
    use_frames = frames[:max_images]
    print(f"\n{'='*70}\nMODE 2: Multi-Image Video Classification ({len(use_frames)} images, 1 call)\n{'='*70}")
    if len(use_frames) < len(frames):
        print(f"  NOTE: Limited to {max_images} to avoid OOM (original: {len(frames)})")
    if hasattr(torch, 'mps'):
        torch.mps.empty_cache()
    content = [{"type": "image", "image": f["image"].resize((512, 512))} for f in use_frames]
    content.append({"type": "text", "text": VIDEO_CLASSIFY_PROMPT.format(n=len(use_frames))})
    msg = [{"role": "user", "content": content}]
    raw, t, ni, no = run_inference(model, processor, device, msg, 512)
    print(f"  Time: {t:.1f}s | Tokens: {ni}in+{no}out")
    print(f"  Output (first 500): {raw[:500]}")
    result = parse_json(raw)
    print(f"  Parsed: {json.dumps(result, ensure_ascii=False, indent=2)[:400]}")
    return {"mode": "mode2_multi_image", "results": result, "total_time_sec": round(t, 2),
            "total_input_tokens": ni, "total_output_tokens": no, "num_calls": 1,
            "num_images_used": len(use_frames), "_raw": raw}


# === Mode 3: Video Mode ===

def run_mode3(model, processor, device, video_path, num_frames=8):
    """Native video input."""
    print(f"\n{'='*70}\nMODE 3: Video Mode (native, 1 call)\n{'='*70}")
    msg = [{"role": "user", "content": [
        {"type": "video", "video": str(Path(video_path).resolve()), "max_pixels": 360*420, "nframes": num_frames},
        {"type": "text", "text": VIDEO_MODE_PROMPT}
    ]}]
    raw, t, ni, no = run_inference(model, processor, device, msg, 512)
    print(f"  Time: {t:.1f}s | Tokens: {ni}in+{no}out")
    print(f"  Output: {raw[:500]}")
    result = parse_json(raw)
    print(f"  Parsed: {json.dumps(result, ensure_ascii=False, indent=2)[:400]}")
    return {"mode": "mode3_video", "results": result, "total_time_sec": round(t, 2),
            "total_input_tokens": ni, "total_output_tokens": no, "num_calls": 1, "_raw": raw}


# === Comparison ===

def print_comparison(m1, m2, m3):
    """Side-by-side comparison."""
    print(f"\n{'='*70}\nCOMPARISON\n{'='*70}")
    print(f"{'Metric':<25} {'Mode1(Single)':<18} {'Mode2(Multi)':<18} {'Mode3(Video)':<18}")
    print("-"*79)
    print(f"{'Time (sec)':<25} {m1['total_time_sec']:<18} {m2['total_time_sec']:<18} {m3['total_time_sec']:<18}")
    print(f"{'API Calls':<25} {m1['num_calls']:<18} {m2['num_calls']:<18} {m3['num_calls']:<18}")
    print(f"{'Input Tokens':<25} {m1['total_input_tokens']:<18} {m2['total_input_tokens']:<18} {m3['total_input_tokens']:<18}")
    print(f"{'Output Tokens':<25} {m1['total_output_tokens']:<18} {m2['total_output_tokens']:<18} {m3['total_output_tokens']:<18}")
    # Mode 1 per-frame
    print(f"\n--- Mode 1 per-frame ---")
    for r in m1["results"]:
        print(f"  t={r.get('_timestamp')}s: {r.get('frame_type', '?')}")
    # Mode 2
    print(f"\n--- Mode 2 ---")
    if isinstance(m2["results"], list):
        for i, r in enumerate(m2["results"]):
            print(f"  Frame {i+1}: {r.get('frame_type','?')}")
    else:
        print(f"  {json.dumps(m2['results'], ensure_ascii=False)[:200]}")
    # Mode 3
    print(f"\n--- Mode 3 ---")
    r3 = m3["results"]
    if isinstance(r3, dict):
        for k, v in r3.items():
            if not k.startswith("_"): print(f"  {k}: {v}")


# === Main ===

def main():
    parser = argparse.ArgumentParser(description="Ablation: VLM Input Modes")
    parser.add_argument("--video", default=DEFAULT_VIDEO)
    parser.add_argument("--num-frames", type=int, default=NUM_FRAMES)
    parser.add_argument("--skip-mode1", action="store_true")
    parser.add_argument("--skip-mode2", action="store_true")
    parser.add_argument("--skip-mode3", action="store_true")
    args = parser.parse_args()

    frames, info = sample_frames(args.video, args.num_frames)
    model, processor, device = load_model()

    m1 = run_mode1(model, processor, device, frames) if not args.skip_mode1 else None
    m2 = run_mode2(model, processor, device, frames) if not args.skip_mode2 else None
    m3 = run_mode3(model, processor, device, args.video, args.num_frames) if not args.skip_mode3 else None

    if m1 and m2 and m3:
        print_comparison(m1, m2, m3)

    # Save
    report = {"video": args.video, "num_frames": args.num_frames, "mode1": m1, "mode2": m2, "mode3": m3}
    report_str = json.dumps(report, ensure_ascii=False, indent=2, default=str)
    with open("ablation_results.json", "w", encoding="utf-8") as f:
        f.write(report_str)
    print(f"\nResults saved to ablation_results.json")


if __name__ == "__main__":
    main()