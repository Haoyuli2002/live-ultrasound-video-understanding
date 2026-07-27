"""
Video Clipping Pipeline (ultrasound-aware, offline, no LLM)
===========================================================
Cuts a long ultrasound teaching video into semantically coherent clips whose
boundaries are both (a) visual scene changes and (b) natural sentence ends.

Pipeline:
  Step 1  Visual change detection on a FIXED time grid. Default method is
          `qwen_embed`: extract original color frames (via torchcodec, NO
          OpenCV) and embed them with Qwen/Qwen3-VL-Embedding-2B (same family
          as our SFT base), then flag drops in adjacent-frame cosine
          similarity. Falls back to SSIM (OpenCV) if the embedding model /
          torchcodec / GPU is unavailable. `ssim` / `framediff` / `histogram`
          remain available via --visual-method.
  Step 2  Sentence boundary detection (English punctuation .?! + pause-gap
          fallback).
  Step 3  Align each visual change to the nearest sentence boundary
          (within tolerance); if none is close, DROP that visual cut
          (keep sentences intact).
  Step 4  Assemble clips with min_clip / max_clip. Short tail clips are MERGED
          into the previous clip (never dropped).
  Step 5  Long clips (> max_clip) are subdivided at sentence boundaries.

Usage:
    python scripts/video_clipping.py --video path.mp4 --transcript transcripts/ID.json
    python scripts/video_clipping.py --batch --transcript-dir transcripts/ --video-dir media/
"""

import json
import argparse
from pathlib import Path

import cv2
import numpy as np


# ============================================================================
# Qwen3-VL-Embedding visual-change detection (default; torchcodec + PIL,
# NO OpenCV). Lazy singletons so the heavy model loads at most once.
# ============================================================================

DEFAULT_QWEN_EMBED_MODEL = "Qwen/Qwen3-VL-Embedding-2B"
_QWEN_EMBED_MODEL_CACHE = {}


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


def _load_qwen_embedder(model_name, device):
    """Lazily load a SentenceTransformer embedder (cached by name+device)."""
    key = (model_name, device)
    if key in _QWEN_EMBED_MODEL_CACHE:
        return _QWEN_EMBED_MODEL_CACHE[key]
    from sentence_transformers import SentenceTransformer  # may raise ImportError
    model = SentenceTransformer(model_name, device=device)
    _QWEN_EMBED_MODEL_CACHE[key] = model
    return model


def _extract_rgb_frames_torchcodec(video_path, times):
    """
    Extract ORIGINAL color frames at the given timestamps using torchcodec
    (NO OpenCV, no resize / no grayscale). Returns list[PIL.Image] aligned
    with `times` (frames that fail to decode are returned as None).
    """
    from torchcodec.decoders import VideoDecoder  # may raise ImportError
    from PIL import Image

    decoder = VideoDecoder(str(video_path))
    frames = []
    for t in times:
        try:
            # get_frame_played_at returns a Frame with .data (C,H,W) uint8 RGB
            frame = decoder.get_frame_played_at(float(t))
            arr = frame.data.permute(1, 2, 0).cpu().numpy()  # HWC RGB uint8
            frames.append(Image.fromarray(arr))
        except Exception:
            frames.append(None)
    return frames


def compute_visual_changes_qwen_embed(
    video_path,
    duration_sec,
    *,
    sample_interval=1.5,
    threshold=0.85,
    min_scene_gap=3.0,
    model_name=DEFAULT_QWEN_EMBED_MODEL,
    device="auto",
    batch_size=16,
):
    """
    Sample frames on a fixed time grid, embed each ORIGINAL color frame with
    Qwen3-VL-Embedding, and flag a scene change when the cosine similarity to
    the previous sampled frame drops below `threshold` (with min_scene_gap
    suppression). Same return contract as compute_visual_changes_grid:
        (scene_change_times, trace)

    Raises on missing deps / model load so the caller can fall back to SSIM.
    """
    if not duration_sec or duration_sec <= 0:
        raise ValueError("duration_sec required for qwen_embed method")

    device = _resolve_device(device)
    model = _load_qwen_embedder(model_name, device)

    # Fixed time grid.
    times = []
    t = 0.0
    while t < duration_sec:
        times.append(round(t, 3))
        t += sample_interval

    frames = _extract_rgb_frames_torchcodec(video_path, times)

    # Keep only successfully decoded frames (track their times).
    valid = [(tt, im) for tt, im in zip(times, frames) if im is not None]
    if len(valid) < 2:
        return [], []

    valid_times = [tt for tt, _ in valid]
    valid_imgs = [im for _, im in valid]

    # Batched encode -> L2-normalized embeddings.
    embs = model.encode(
        valid_imgs,
        batch_size=batch_size,
        convert_to_numpy=True,
        normalize_embeddings=True,
        show_progress_bar=False,
    )
    embs = np.asarray(embs, dtype=np.float32)

    scene_change_times = []
    trace = []
    last_change_t = -1e9
    prev = None
    for i, tt in enumerate(valid_times):
        cur = embs[i]
        if prev is None:
            trace.append({"t": tt, "similarity": 1.0, "scene_change": False})
            prev = cur
            continue
        sim = float(np.dot(prev, cur))  # cosine (already normalized)
        is_change = sim < threshold and (tt - last_change_t) >= min_scene_gap
        if is_change:
            scene_change_times.append(tt)
            last_change_t = tt
        trace.append({"t": tt, "similarity": round(sim, 3), "scene_change": bool(is_change)})
        prev = cur

    return scene_change_times, trace


# ============================================================================
# Frame utilities (OpenCV; used by ssim / framediff / histogram)
# ============================================================================

def _read_gray_at(cap, t_sec, resize=256):
    cap.set(cv2.CAP_PROP_POS_MSEC, t_sec * 1000)
    ret, frame = cap.read()
    if not ret:
        return None
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    if resize:
        gray = cv2.resize(gray, (resize, resize), interpolation=cv2.INTER_AREA)
    return gray


def _similarity(a, b, method):
    """
    Return a similarity score in [0, 1], where 1 == identical.

    - ssim:       structural similarity (needs scikit-image; falls back to framediff)
    - framediff:  1 - mean_abs_diff/255
    - histogram:  grayscale histogram correlation
    """
    if method == "ssim":
        try:
            from skimage.metrics import structural_similarity as ssim
            score = ssim(a, b)
            return float(max(0.0, min(1.0, score)))
        except Exception:
            method = "framediff"

    if method == "framediff":
        diff = np.abs(a.astype(np.float32) - b.astype(np.float32)).mean()
        return float(1.0 - diff / 255.0)

    # histogram correlation
    ha = cv2.calcHist([a], [0], None, [256], [0, 256])
    ha = cv2.normalize(ha, ha).flatten()
    hb = cv2.calcHist([b], [0], None, [256], [0, 256])
    hb = cv2.normalize(hb, hb).flatten()
    return float(cv2.compareHist(ha, hb, cv2.HISTCMP_CORREL))


# ============================================================================
# Step 1 (OpenCV methods): Visual change detection on a fixed time grid
# ============================================================================

def compute_visual_changes_grid(
    video_path,
    duration_sec,
    *,
    method="ssim",
    sample_interval=1.5,
    threshold=0.6,
    min_scene_gap=3.0,
    resize=256,
):
    """
    Sample frames every `sample_interval` seconds and compare each to the
    previous sampled frame. A scene change is flagged when similarity drops
    below `threshold`. Consecutive changes closer than `min_scene_gap` seconds
    are suppressed.

    Returns:
        scene_change_times: sorted list of float seconds
        trace: list of {t, similarity, scene_change} for debugging / threshold tuning
    """
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        print(f"  WARNING: Cannot open video {video_path}")
        return [], []

    if not duration_sec or duration_sec <= 0:
        fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
        frame_count = cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0
        duration_sec = (frame_count / fps) if frame_count else 0.0

    times = []
    t = 0.0
    while t < duration_sec:
        times.append(round(t, 3))
        t += sample_interval

    scene_change_times = []
    trace = []
    prev = None
    last_change_t = -1e9

    for tt in times:
        gray = _read_gray_at(cap, tt, resize=resize)
        if gray is None:
            continue
        if prev is None:
            trace.append({"t": tt, "similarity": 1.0, "scene_change": False})
            prev = gray
            continue

        sim = _similarity(prev, gray, method)
        is_change = sim < threshold and (tt - last_change_t) >= min_scene_gap
        if is_change:
            scene_change_times.append(tt)
            last_change_t = tt
        trace.append({"t": tt, "similarity": round(sim, 3), "scene_change": bool(is_change)})
        prev = gray

    cap.release()
    return scene_change_times, trace


# ============================================================================
# Step 2: Sentence boundary detection
# ============================================================================

def detect_sentence_boundaries(segments, pause_gap=0.8):
    """
    Return a list of segment indices whose END is a natural sentence boundary.

    Rules:
      1. text ends with . ? !  -> boundary
      2. pause gap to next segment > pause_gap -> boundary
      3. fallback: if almost no boundary found, treat every segment end as a
         weak boundary.
    """
    boundaries = []
    n = len(segments)
    for i, s in enumerate(segments):
        text = (s.get("text") or "").rstrip()
        is_punct_end = text.endswith((".", "?", "!"))
        is_pause = False
        if i + 1 < n:
            gap = float(segments[i + 1].get("start", 0.0)) - float(s.get("end", 0.0))
            is_pause = gap > pause_gap
        else:
            is_pause = True  # last segment is always a boundary
        if is_punct_end or is_pause:
            boundaries.append(i)

    # Fallback: too few boundaries -> every segment end is a weak boundary.
    if n > 0 and len(boundaries) < max(1, n // 10):
        boundaries = list(range(n))

    return boundaries


# ============================================================================
# Step 3: Align visual changes to sentence boundaries
# ============================================================================

def align_scene_to_sentences(scene_change_times, segments, boundary_indices, tolerance=5.0):
    """
    For each visual scene change time, find the nearest sentence-boundary
    segment end within `tolerance` seconds. If none is close enough, DROP the
    visual cut (keep sentences intact).

    Returns sorted unique cut segment indices.
    """
    cut_indices = set()
    for sc_time in scene_change_times:
        best_idx = None
        best_dist = float("inf")
        for idx in boundary_indices:
            seg_end = float(segments[idx].get("end", 0.0))
            dist = abs(seg_end - sc_time)
            if dist <= tolerance and dist < best_dist:
                best_idx = idx
                best_dist = dist
        if best_idx is not None:
            cut_indices.add(best_idx)
    return sorted(cut_indices)


# ============================================================================
# Step 4 + 5: Assemble clips (min/max, short-tail merge, long split)
# ============================================================================

def _make_clip(segs, clip_idx, cut_reason):
    start = float(segs[0]["start"])
    end = float(segs[-1]["end"])
    return {
        "clip_idx": clip_idx,
        "start": round(start, 2),
        "end": round(end, 2),
        "duration": round(end - start, 2),
        "num_segments": len(segs),
        "text": " ".join((s.get("text") or "").strip() for s in segs).strip(),
        "cut_reason": cut_reason,
    }


def _split_long_clip(clip_segs, boundary_local_indices, min_clip, max_clip, start_idx):
    """
    Split a too-long clip into <= max_clip sub-clips ending at sentence
    boundaries. boundary_local_indices are indices within clip_segs.
    """
    subclips = []
    sub_start = 0
    n = len(clip_segs)

    boundary_set = set(boundary_local_indices)

    i = 0
    while i < n:
        cur_dur = float(clip_segs[i]["end"]) - float(clip_segs[sub_start]["start"])
        if cur_dur >= max_clip:
            # find the latest boundary <= i (and > sub_start) to cut at
            cut_at = None
            for b in range(i, sub_start, -1):
                if b in boundary_set:
                    cut_at = b
                    break
            if cut_at is None:
                cut_at = i  # hard cut if no boundary available
            subclips.append((sub_start, cut_at))
            sub_start = cut_at + 1
            i = sub_start
        else:
            i += 1

    if sub_start < n:
        subclips.append((sub_start, n - 1))

    return subclips


def segment_by_cuts(segments, cut_indices, min_clip=30, max_clip=240):
    """
    Build clips from sentence-aligned cut indices.

    - Accumulate segments until a cut index is reached AND duration >= min_clip.
    - Short tail (< min_clip) is merged into the previous clip.
    - Long clips (> max_clip) are subdivided at sentence boundaries.
    """
    if not segments:
        return []

    n = len(segments)
    cut_set = sorted(set(cut_indices))

    clips = []
    clip_start_idx = 0

    for cut_idx in cut_set:
        if cut_idx < clip_start_idx:
            continue
        seg_start_time = float(segments[clip_start_idx]["start"])
        seg_end_time = float(segments[cut_idx]["end"])
        duration = seg_end_time - seg_start_time
        if duration >= min_clip:
            clips.append((clip_start_idx, cut_idx, "scene_change"))
            clip_start_idx = cut_idx + 1

    # Remaining tail
    if clip_start_idx < n:
        tail_start = clip_start_idx
        tail_end = n - 1
        tail_dur = float(segments[tail_end]["end"]) - float(segments[tail_start]["start"])
        if tail_dur >= min_clip or not clips:
            clips.append((tail_start, tail_end, "end_of_video"))
        else:
            # merge short tail into previous clip
            prev_start, _, _ = clips[-1]
            clips[-1] = (prev_start, tail_end, "end_of_video_merged")

    # Precompute which global segment indices are sentence boundaries for splitting.
    # We reuse a simple rule here: any segment whose text ends with punctuation.
    def _is_boundary_seg(seg):
        return (seg.get("text") or "").rstrip().endswith((".", "?", "!"))

    # Expand tuples into clip dicts, splitting long clips.
    final = []
    for (s_idx, e_idx, reason) in clips:
        clip_segs = segments[s_idx:e_idx + 1]
        dur = float(clip_segs[-1]["end"]) - float(clip_segs[0]["start"])
        if dur <= max_clip:
            final.append(_make_clip(clip_segs, len(final), reason))
            continue

        boundary_local = [i for i, s in enumerate(clip_segs) if _is_boundary_seg(s)]
        subclips = _split_long_clip(clip_segs, boundary_local, min_clip, max_clip, s_idx)
        for (a, b) in subclips:
            final.append(_make_clip(clip_segs[a:b + 1], len(final), "max_clip_split"))

    return final


# ============================================================================
# Legacy histogram method (kept for --visual-method histogram)
# ============================================================================

def compute_segment_visual_changes(video_path, segments, threshold=0.3):
    """Legacy: per-segment midpoint frame + grayscale histogram correlation."""
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        print(f"  WARNING: Cannot open video {video_path}")
        return []

    prev_hist = None
    results = []
    for i, seg in enumerate(segments):
        mid_time = (seg["start"] + seg["end"]) / 2
        cap.set(cv2.CAP_PROP_POS_MSEC, mid_time * 1000)
        ret, frame = cap.read()
        if not ret:
            results.append({"seg_idx": i, "start": seg["start"], "end": seg["end"],
                            "similarity": None, "scene_change": False})
            continue
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        hist = cv2.calcHist([gray], [0], None, [256], [0, 256])
        hist = cv2.normalize(hist, hist).flatten()
        if prev_hist is not None:
            similarity = cv2.compareHist(prev_hist, hist, cv2.HISTCMP_CORREL)
            scene_change = similarity < threshold
        else:
            similarity = 1.0
            scene_change = False
        results.append({"seg_idx": i, "start": seg["start"], "end": seg["end"],
                        "similarity": round(similarity, 3), "scene_change": scene_change})
        prev_hist = hist
    cap.release()
    return results


def segment_enhanced(segments, visual_changes, min_clip=30, max_clip=240, tolerance=10):
    """Legacy histogram-based segmentation (kept for compatibility)."""
    if not segments or not visual_changes:
        return []
    scene_change_times = [v["start"] for v in visual_changes if v.get("scene_change")]
    boundary_indices = detect_sentence_boundaries(segments, pause_gap=0.8)
    cut_indices = align_scene_to_sentences(scene_change_times, segments, boundary_indices, tolerance=tolerance)
    return segment_by_cuts(segments, cut_indices, min_clip=min_clip, max_clip=max_clip)


# ============================================================================
# Full pipeline (default: qwen_embed grid, no LLM)
# ============================================================================

def clip_video(
    video_path,
    transcript_path,
    output_dir=None,
    *,
    visual_method="qwen_embed",
    sample_interval=1.5,
    scene_threshold=None,
    min_scene_gap=3.0,
    pause_gap=0.8,
    tolerance=5.0,
    min_clip=30,
    max_clip=240,
    resize=256,
    save_trace=False,
    qwen_embed_model=DEFAULT_QWEN_EMBED_MODEL,
    qwen_embed_device="auto",
    qwen_embed_batch=16,
):
    """
    Cut a video into sentence-aligned clips.

    visual_method:
      - "qwen_embed" (default): Qwen3-VL-Embedding cosine on original color
        frames (torchcodec, no OpenCV). Falls back to "ssim" if the model /
        torchcodec / deps are unavailable.
      - "ssim" / "framediff": grayscale grid similarity (OpenCV).
      - "histogram": legacy per-segment histogram.
    """
    video_path = Path(video_path)
    transcript_path = Path(transcript_path)

    with open(transcript_path) as f:
        asr_data = json.load(f)

    segments = asr_data["segments"]
    video_id = asr_data.get("video_id", video_path.stem)
    duration_sec = asr_data.get("duration_sec")

    print(f"\n{'='*70}")
    print(f"Clipping: {video_id}")
    print(f"  Duration: {duration_sec}s | Segments: {len(segments)}")
    print(f"  Visual method: {visual_method} | sample_interval={sample_interval}s")

    trace = None

    if visual_method == "histogram":
        # scene_threshold default for histogram
        hist_thr = 0.3 if scene_threshold is None else scene_threshold
        visual_changes = compute_segment_visual_changes(str(video_path), segments, hist_thr)
        clips = segment_enhanced(segments, visual_changes, min_clip=min_clip, max_clip=max_clip,
                                 tolerance=tolerance)
        method = "histogram_sentence"
        num_changes = sum(1 for v in visual_changes if v.get("scene_change"))
    elif visual_method == "qwen_embed":
        # Cosine threshold default for embeddings (tune via --save-trace).
        thr = 0.85 if scene_threshold is None else scene_threshold
        try:
            scene_change_times, trace = compute_visual_changes_qwen_embed(
                str(video_path), duration_sec,
                sample_interval=sample_interval, threshold=thr,
                min_scene_gap=min_scene_gap,
                model_name=qwen_embed_model, device=qwen_embed_device,
                batch_size=qwen_embed_batch,
            )
            method = "qwen_embed_sentence"
        except Exception as e:
            print(f"  WARNING: qwen_embed failed ({type(e).__name__}: {e}); "
                  f"falling back to ssim.")
            ssim_thr = 0.6
            scene_change_times, trace = compute_visual_changes_grid(
                str(video_path), duration_sec,
                method="ssim", sample_interval=sample_interval,
                threshold=ssim_thr, min_scene_gap=min_scene_gap, resize=resize,
            )
            method = "ssim_sentence(fallback)"
        num_changes = len(scene_change_times)
        boundary_indices = detect_sentence_boundaries(segments, pause_gap=pause_gap)
        cut_indices = align_scene_to_sentences(scene_change_times, segments, boundary_indices, tolerance=tolerance)
        clips = segment_by_cuts(segments, cut_indices, min_clip=min_clip, max_clip=max_clip)
    else:
        # ssim / framediff (grayscale grid)
        ssim_thr = 0.6 if scene_threshold is None else scene_threshold
        scene_change_times, trace = compute_visual_changes_grid(
            str(video_path), duration_sec,
            method=visual_method, sample_interval=sample_interval,
            threshold=ssim_thr, min_scene_gap=min_scene_gap, resize=resize,
        )
        num_changes = len(scene_change_times)
        boundary_indices = detect_sentence_boundaries(segments, pause_gap=pause_gap)
        cut_indices = align_scene_to_sentences(scene_change_times, segments, boundary_indices, tolerance=tolerance)
        clips = segment_by_cuts(segments, cut_indices, min_clip=min_clip, max_clip=max_clip)
        method = f"{visual_method}_sentence"

    print(f"  Visual scene changes: {num_changes}")

    total_clip_dur = sum(c["duration"] for c in clips)
    coverage = (total_clip_dur / duration_sec * 100) if duration_sec else 0.0

    print(f"  Result: {len(clips)} clips | Coverage: {coverage:.0f}%")
    for c in clips:
        print(f"    Clip {c['clip_idx']:2d}: {c['start']:7.1f}-{c['end']:7.1f}s "
              f"({c['duration']:6.1f}s) [{c['cut_reason']}]")

    output = {
        "video_id": video_id,
        "video_path": str(video_path),
        "duration_sec": duration_sec,
        "method": method,
        "params": {
            "visual_method": visual_method,
            "sample_interval": sample_interval,
            "scene_threshold": scene_threshold,
            "min_scene_gap": min_scene_gap,
            "pause_gap": pause_gap,
            "tolerance": tolerance,
            "min_clip": min_clip,
            "max_clip": max_clip,
            "qwen_embed_model": qwen_embed_model if visual_method == "qwen_embed" else None,
        },
        "num_clips": len(clips),
        "coverage_pct": round(coverage, 1),
        "clips": clips,
    }
    if save_trace and trace is not None:
        output["visual_trace"] = trace

    if output_dir:
        out_path = Path(output_dir) / f"{video_id}_clips.json"
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(output, f, ensure_ascii=False, indent=2)
        print(f"  Saved: {out_path}")

    return output


# Backwards-compatible alias (old name).
segment_video = clip_video


# ============================================================================
# CLI
# ============================================================================

def main():
    parser = argparse.ArgumentParser(description="Ultrasound-aware video clipping (no LLM)")
    parser.add_argument("--video", type=str, help="Single video path")
    parser.add_argument("--transcript", type=str, help="Transcript JSON path")
    parser.add_argument("--output-dir", type=str, default="results/clips")

    parser.add_argument("--visual-method", type=str, default="qwen_embed",
                        choices=["qwen_embed", "ssim", "framediff", "histogram"])
    parser.add_argument("--sample-interval", type=float, default=1.5)
    parser.add_argument("--scene-threshold", type=float, default=None,
                        help="Similarity threshold below which a scene change is "
                             "flagged. Default: 0.85 (qwen_embed cosine) / 0.6 (ssim).")
    parser.add_argument("--min-scene-gap", type=float, default=3.0)
    parser.add_argument("--pause-gap", type=float, default=0.8)
    parser.add_argument("--tolerance", type=float, default=5.0)
    parser.add_argument("--min-clip", type=int, default=30)
    parser.add_argument("--max-clip", type=int, default=240)
    parser.add_argument("--resize", type=int, default=256)
    parser.add_argument("--save-trace", action="store_true")

    parser.add_argument("--qwen-embed-model", type=str, default=DEFAULT_QWEN_EMBED_MODEL)
    parser.add_argument("--qwen-embed-device", type=str, default="auto")
    parser.add_argument("--qwen-embed-batch", type=int, default=16)

    parser.add_argument("--batch", action="store_true")
    parser.add_argument("--transcript-dir", type=str, default="transcripts")
    parser.add_argument("--video-dir", type=str)
    args = parser.parse_args()

    def _run(video, transcript):
        clip_video(
            video, transcript, args.output_dir,
            visual_method=args.visual_method,
            sample_interval=args.sample_interval,
            scene_threshold=args.scene_threshold,
            min_scene_gap=args.min_scene_gap,
            pause_gap=args.pause_gap,
            tolerance=args.tolerance,
            min_clip=args.min_clip,
            max_clip=args.max_clip,
            resize=args.resize,
            save_trace=args.save_trace,
            qwen_embed_model=args.qwen_embed_model,
            qwen_embed_device=args.qwen_embed_device,
            qwen_embed_batch=args.qwen_embed_batch,
        )

    if args.video and args.transcript:
        _run(args.video, args.transcript)
    elif args.batch:
        transcript_dir = Path(args.transcript_dir)
        transcripts = sorted(transcript_dir.glob("*.json"))
        print(f"Batch clipping: {len(transcripts)} transcripts")
        for tp in transcripts:
            with open(tp) as f:
                data = json.load(f)
            vp = Path(data.get("video_path", ""))
            if not vp.exists() and args.video_dir:
                vp = Path(args.video_dir) / vp.name
            if vp.exists():
                try:
                    _run(str(vp), str(tp))
                except Exception as e:
                    print(f"  ERROR: {e}")
            else:
                print(f"  SKIP: Video not found for {tp.stem}")
    else:
        print("Provide --video and --transcript, or --batch.")


if __name__ == "__main__":
    main()
