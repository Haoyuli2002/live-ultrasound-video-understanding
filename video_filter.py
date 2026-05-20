"""
Ultrasound Video Purity Filter
===============================
Automatically analyze downloaded ultrasound videos and filter out those
containing artificial artifacts (text annotations, PPT slides, talking heads,
color overlays, etc.)

Usage:
    python video_filter.py [--input-dir PATH] [--frames N] [--output-report PATH]

Requirements:
    pip install opencv-python numpy Pillow
    brew install tesseract  (optional, for OCR-based text detection)
"""

import argparse
import json
import sys
from pathlib import Path
from dataclasses import dataclass, field, asdict
from typing import Optional

import cv2
import numpy as np


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

DEFAULT_INPUT_DIR = Path(__file__).parent / "UltrasoundCrawler_KeyCode_20260323_v2" / "output"
FRAMES_PER_MINUTE = 10  # Number of frames to sample per minute of video
VIDEO_EXTENSIONS = {".mp4", ".webm", ".mkv", ".avi", ".mov"}


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclass
class FrameAnalysis:
    frame_idx: int
    timestamp_sec: float
    grayscale_ratio: float        # % of pixels that are grayscale (R≈G≈B)
    black_border_ratio: float     # % of very dark pixels (typical US border)
    color_pixel_ratio: float      # % of clearly colored pixels
    text_region_score: float      # Heuristic text detection score
    has_face: bool                # Face detected
    edge_density: float           # High edge density in non-US areas suggests PPT
    bright_uniform_blocks: float  # Large uniform bright areas suggest slides


@dataclass
class VideoAnalysis:
    video_path: str
    video_id: str
    duration_sec: float
    total_frames: int
    frame_analyses: list = field(default_factory=list)
    
    # Aggregated scores
    avg_grayscale_ratio: float = 0.0
    avg_color_ratio: float = 0.0
    face_detected_frames: int = 0
    avg_text_score: float = 0.0
    avg_bright_blocks: float = 0.0
    
    # Final verdict
    purity_score: float = 0.0  # 0-100, higher = more pure ultrasound
    category: str = "unknown"  # pure_ultrasound, annotated, ppt_mixed, talking_head, rejected
    rejection_reasons: list = field(default_factory=list)


# ---------------------------------------------------------------------------
# Frame Analysis Functions
# ---------------------------------------------------------------------------

def compute_grayscale_ratio(frame: np.ndarray) -> float:
    """Compute what fraction of pixels are approximately grayscale."""
    if len(frame.shape) == 2:
        return 1.0
    
    max_channel = frame.max(axis=2).astype(np.float32)
    min_channel = frame.min(axis=2).astype(np.float32)
    diff = max_channel - min_channel
    
    # Threshold: color difference < 25 means approximately gray
    grayscale_mask = diff < 25
    return float(grayscale_mask.sum()) / (frame.shape[0] * frame.shape[1])


def compute_black_border_ratio(frame: np.ndarray) -> float:
    """Compute what fraction of pixels are very dark (black borders typical in US)."""
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) if len(frame.shape) == 3 else frame
    dark_mask = gray < 15
    return float(dark_mask.sum()) / (gray.shape[0] * gray.shape[1])


def compute_color_pixel_ratio(frame: np.ndarray) -> float:
    """Compute what fraction of pixels are clearly colored (non-gray)."""
    if len(frame.shape) == 2:
        return 0.0
    
    # Convert to HSV, check saturation
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    # Pixels with saturation > 50 and value > 50 are "colored"
    colored_mask = (hsv[:, :, 1] > 50) & (hsv[:, :, 2] > 50)
    return float(colored_mask.sum()) / (frame.shape[0] * frame.shape[1])


def compute_text_score(frame: np.ndarray) -> float:
    """
    Heuristic text detection without OCR.
    Text areas tend to have high local contrast with thin structures.
    Uses morphological operations to detect text-like regions.
    """
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) if len(frame.shape) == 3 else frame
    
    # Apply edge detection
    edges = cv2.Canny(gray, 100, 200)
    
    # Dilate to connect text characters
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (15, 3))
    dilated = cv2.dilate(edges, kernel, iterations=1)
    
    # Find contours that look like text blocks (wide, short rectangles)
    contours, _ = cv2.findContours(dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    text_area = 0
    total_area = frame.shape[0] * frame.shape[1]
    
    for cnt in contours:
        x, y, w, h = cv2.boundingRect(cnt)
        aspect_ratio = w / max(h, 1)
        area = w * h
        
        # Text blocks are typically wide and short
        if aspect_ratio > 2.0 and area > 200 and h < frame.shape[0] * 0.1:
            text_area += area
    
    return min(text_area / total_area, 1.0)


def detect_face(frame: np.ndarray, face_cascade: cv2.CascadeClassifier) -> bool:
    """Detect if a human face is present in the frame."""
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) if len(frame.shape) == 3 else frame
    
    # Resize for faster detection
    scale = 320 / max(gray.shape)
    if scale < 1.0:
        small = cv2.resize(gray, None, fx=scale, fy=scale)
    else:
        small = gray
    
    faces = face_cascade.detectMultiScale(small, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
    return len(faces) > 0


def compute_edge_density(frame: np.ndarray) -> float:
    """Compute overall edge density. PPT slides have different edge patterns than US."""
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) if len(frame.shape) == 3 else frame
    edges = cv2.Canny(gray, 50, 150)
    return float(edges.sum() / 255) / (gray.shape[0] * gray.shape[1])


def compute_bright_uniform_blocks(frame: np.ndarray) -> float:
    """
    Detect large uniform bright areas (typical of PPT slides/white backgrounds).
    Real ultrasound images rarely have large uniform bright patches.
    """
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) if len(frame.shape) == 3 else frame
    
    # Threshold for bright areas
    _, bright_mask = cv2.threshold(gray, 200, 255, cv2.THRESH_BINARY)
    
    # Check for large connected components
    contours, _ = cv2.findContours(bright_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    total_area = gray.shape[0] * gray.shape[1]
    large_bright_area = 0
    
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area > total_area * 0.05:  # > 5% of frame
            large_bright_area += area
    
    return min(large_bright_area / total_area, 1.0)


# ---------------------------------------------------------------------------
# Video Processing
# ---------------------------------------------------------------------------

def analyze_frame(frame: np.ndarray, frame_idx: int, timestamp: float,
                  face_cascade: cv2.CascadeClassifier) -> FrameAnalysis:
    """Analyze a single frame for ultrasound purity indicators."""
    return FrameAnalysis(
        frame_idx=frame_idx,
        timestamp_sec=round(timestamp, 2),
        grayscale_ratio=round(compute_grayscale_ratio(frame), 4),
        black_border_ratio=round(compute_black_border_ratio(frame), 4),
        color_pixel_ratio=round(compute_color_pixel_ratio(frame), 4),
        text_region_score=round(compute_text_score(frame), 4),
        has_face=detect_face(frame, face_cascade),
        edge_density=round(compute_edge_density(frame), 4),
        bright_uniform_blocks=round(compute_bright_uniform_blocks(frame), 4),
    )


def analyze_video(video_path: Path, frames_per_minute: int = FRAMES_PER_MINUTE) -> Optional[VideoAnalysis]:
    """Analyze a video file and return purity assessment.
    
    Samples `frames_per_minute` frames for each minute of video duration.
    """
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        print(f"  [ERROR] Cannot open: {video_path}")
        return None
    
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    duration = total_frames / fps
    
    # Calculate number of frames to sample: frames_per_minute * duration_in_minutes
    duration_minutes = max(duration / 60.0, 1.0 / 60.0)  # at least treat as 1 second
    num_frames = max(1, int(frames_per_minute * duration_minutes))
    
    if total_frames < num_frames:
        num_frames = max(1, total_frames)
    
    # Sample frames evenly across the video
    frame_indices = np.linspace(0, total_frames - 1, num_frames, dtype=int)
    
    # Load face cascade
    face_cascade_path = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
    face_cascade = cv2.CascadeClassifier(face_cascade_path)
    
    video_id = video_path.stem
    analysis = VideoAnalysis(
        video_path=str(video_path),
        video_id=video_id,
        duration_sec=round(duration, 2),
        total_frames=total_frames,
    )
    
    for idx in frame_indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(idx))
        ret, frame = cap.read()
        if not ret:
            continue
        
        timestamp = idx / fps
        fa = analyze_frame(frame, int(idx), timestamp, face_cascade)
        analysis.frame_analyses.append(fa)
    
    cap.release()
    
    # Aggregate scores
    if analysis.frame_analyses:
        n = len(analysis.frame_analyses)
        analysis.avg_grayscale_ratio = round(
            sum(f.grayscale_ratio for f in analysis.frame_analyses) / n, 4)
        analysis.avg_color_ratio = round(
            sum(f.color_pixel_ratio for f in analysis.frame_analyses) / n, 4)
        analysis.face_detected_frames = sum(
            1 for f in analysis.frame_analyses if f.has_face)
        analysis.avg_text_score = round(
            sum(f.text_region_score for f in analysis.frame_analyses) / n, 4)
        analysis.avg_bright_blocks = round(
            sum(f.bright_uniform_blocks for f in analysis.frame_analyses) / n, 4)
    
    # Compute purity score and categorize
    _compute_verdict(analysis)
    
    return analysis


def _compute_verdict(analysis: VideoAnalysis):
    """Compute final purity score and category based on frame analyses."""
    score = 100.0
    reasons = []
    
    # Penalty for color content (ultrasound should be mostly grayscale)
    if analysis.avg_color_ratio > 0.15:
        penalty = min(30, analysis.avg_color_ratio * 100)
        score -= penalty
        reasons.append(f"High color content ({analysis.avg_color_ratio:.1%})")
    
    # Penalty for low grayscale ratio
    if analysis.avg_grayscale_ratio < 0.7:
        penalty = (0.7 - analysis.avg_grayscale_ratio) * 60
        score -= penalty
        reasons.append(f"Low grayscale ratio ({analysis.avg_grayscale_ratio:.1%})")
    
    # Penalty for face detection (talking head)
    if analysis.face_detected_frames > 0:
        face_ratio = analysis.face_detected_frames / max(len(analysis.frame_analyses), 1)
        penalty = face_ratio * 40
        score -= penalty
        reasons.append(
            f"Face detected in {analysis.face_detected_frames}/{len(analysis.frame_analyses)} frames")
    
    # Penalty for text regions
    if analysis.avg_text_score > 0.02:
        penalty = min(25, analysis.avg_text_score * 500)
        score -= penalty
        reasons.append(f"Text regions detected (score={analysis.avg_text_score:.4f})")
    
    # Penalty for bright uniform blocks (PPT slides)
    if analysis.avg_bright_blocks > 0.1:
        penalty = min(35, analysis.avg_bright_blocks * 100)
        score -= penalty
        reasons.append(f"Large bright uniform areas ({analysis.avg_bright_blocks:.1%})")
    
    score = max(0, min(100, score))
    analysis.purity_score = round(score, 1)
    analysis.rejection_reasons = reasons
    
    # Categorize
    if score >= 75:
        analysis.category = "pure_ultrasound"
    elif score >= 55:
        analysis.category = "mildly_annotated"
    elif score >= 35:
        analysis.category = "heavily_annotated"
    else:
        analysis.category = "rejected"


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def find_videos(input_dir: Path) -> list:
    """Recursively find all video files under input directory."""
    videos = []
    for ext in VIDEO_EXTENSIONS:
        videos.extend(input_dir.rglob(f"*{ext}"))
    # Exclude .part files (incomplete downloads)
    videos = [v for v in videos if not str(v).endswith(".part")]
    return sorted(videos)


def print_report(results: list):
    """Print a summary report to console."""
    print("\n" + "=" * 80)
    print("ULTRASOUND VIDEO PURITY FILTER - REPORT")
    print("=" * 80)
    
    # Sort by purity score descending
    results.sort(key=lambda x: x.purity_score, reverse=True)
    
    categories = {
        "pure_ultrasound": [],
        "mildly_annotated": [],
        "heavily_annotated": [],
        "rejected": [],
    }
    
    for r in results:
        categories.get(r.category, categories["rejected"]).append(r)
        
        icon = {
            "pure_ultrasound": "✅",
            "mildly_annotated": "⚠️ ",
            "heavily_annotated": "❌",
            "rejected": "🚫",
        }.get(r.category, "?")
        
        print(f"\n{icon} [{r.purity_score:5.1f}/100] {r.video_id}")
        print(f"     Path: {r.video_path}")
        print(f"     Duration: {r.duration_sec:.1f}s | Grayscale: {r.avg_grayscale_ratio:.1%} | "
              f"Color: {r.avg_color_ratio:.1%} | Text: {r.avg_text_score:.4f} | "
              f"Bright blocks: {r.avg_bright_blocks:.1%}")
        if r.rejection_reasons:
            print(f"     Issues: {'; '.join(r.rejection_reasons)}")
    
    # Summary
    print("\n" + "-" * 80)
    print("SUMMARY")
    print("-" * 80)
    print(f"  Total videos analyzed: {len(results)}")
    print(f"  ✅ Pure ultrasound:     {len(categories['pure_ultrasound'])}")
    print(f"  ⚠️  Mildly annotated:   {len(categories['mildly_annotated'])}")
    print(f"  ❌ Heavily annotated:   {len(categories['heavily_annotated'])}")
    print(f"  🚫 Rejected:            {len(categories['rejected'])}")
    print("=" * 80)


def save_report(results: list, output_path: Path):
    """Save detailed report as JSON."""
    report = {
        "total_videos": len(results),
        "summary": {
            "pure_ultrasound": sum(1 for r in results if r.category == "pure_ultrasound"),
            "mildly_annotated": sum(1 for r in results if r.category == "mildly_annotated"),
            "heavily_annotated": sum(1 for r in results if r.category == "heavily_annotated"),
            "rejected": sum(1 for r in results if r.category == "rejected"),
        },
        "videos": [],
    }
    
    for r in sorted(results, key=lambda x: x.purity_score, reverse=True):
        video_entry = {
            "video_id": r.video_id,
            "video_path": r.video_path,
            "duration_sec": r.duration_sec,
            "purity_score": r.purity_score,
            "category": r.category,
            "avg_grayscale_ratio": r.avg_grayscale_ratio,
            "avg_color_ratio": r.avg_color_ratio,
            "face_detected_frames": r.face_detected_frames,
            "avg_text_score": r.avg_text_score,
            "avg_bright_blocks": r.avg_bright_blocks,
            "rejection_reasons": r.rejection_reasons,
            "frame_details": [asdict(fa) for fa in r.frame_analyses],
        }
        report["videos"].append(video_entry)
    
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)
    print(f"\nDetailed report saved to: {output_path}")


def save_frame_samples(results: list, output_dir: Path):
    """Save sample frames from each video for manual inspection."""
    frames_dir = output_dir / "sample_frames"
    frames_dir.mkdir(parents=True, exist_ok=True)
    
    for r in results:
        cap = cv2.VideoCapture(r.video_path)
        if not cap.isOpened():
            continue
        
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        # Save 3 sample frames: beginning, middle, end
        sample_positions = [
            int(total_frames * 0.1),
            int(total_frames * 0.5),
            int(total_frames * 0.9),
        ]
        
        for i, pos in enumerate(sample_positions):
            cap.set(cv2.CAP_PROP_POS_FRAMES, pos)
            ret, frame = cap.read()
            if ret:
                fname = f"{r.video_id}_frame{i}_{r.category}_{r.purity_score:.0f}.jpg"
                cv2.imwrite(str(frames_dir / fname), frame)
        
        cap.release()
    
    print(f"Sample frames saved to: {frames_dir}")


def main():
    parser = argparse.ArgumentParser(description="Ultrasound Video Purity Filter")
    parser.add_argument(
        "--input-dir", type=str, default=None,
        help="Input directory containing video files (default: UltrasoundCrawler output)"
    )
    parser.add_argument(
        "--frames-per-minute", type=int, default=FRAMES_PER_MINUTE,
        help=f"Number of frames to sample per minute of video (default: {FRAMES_PER_MINUTE})"
    )
    parser.add_argument(
        "--output-report", type=str, default=None,
        help="Output JSON report path (default: filter_report.json in input dir)"
    )
    parser.add_argument(
        "--save-frames", action="store_true",
        help="Save sample frames for manual inspection"
    )
    parser.add_argument(
        "--min-score", type=float, default=55.0,
        help="Minimum purity score to keep (default: 55.0)"
    )
    
    args = parser.parse_args()
    
    # Determine input directory
    if args.input_dir:
        input_dir = Path(args.input_dir)
    else:
        input_dir = DEFAULT_INPUT_DIR
    
    if not input_dir.exists():
        print(f"[ERROR] Input directory not found: {input_dir}")
        sys.exit(1)
    
    # Find videos
    print(f"Scanning for videos in: {input_dir}")
    videos = find_videos(input_dir)
    
    if not videos:
        print("[WARNING] No video files found!")
        sys.exit(0)
    
    print(f"Found {len(videos)} video(s) to analyze.\n")
    
    # Analyze each video
    results = []
    for i, video_path in enumerate(videos, 1):
        print(f"[{i}/{len(videos)}] Analyzing: {video_path.name} ...", end=" ", flush=True)
        analysis = analyze_video(video_path, frames_per_minute=args.frames_per_minute)
        if analysis:
            results.append(analysis)
            print(f"Score: {analysis.purity_score:.1f} ({analysis.category})")
        else:
            print("FAILED")
    
    if not results:
        print("[WARNING] No videos could be analyzed!")
        sys.exit(0)
    
    # Print report
    print_report(results)
    
    # Save JSON report
    if args.output_report:
        report_path = Path(args.output_report)
    else:
        report_path = input_dir / "filter_report.json"
    save_report(results, report_path)
    
    # Save sample frames if requested
    if args.save_frames:
        save_frame_samples(results, report_path.parent)
    
    # Print kept videos
    kept = [r for r in results if r.purity_score >= args.min_score]
    print(f"\n{'=' * 80}")
    print(f"KEPT VIDEOS (score >= {args.min_score}): {len(kept)}/{len(results)}")
    print(f"{'=' * 80}")
    for r in sorted(kept, key=lambda x: x.purity_score, reverse=True):
        print(f"  [{r.purity_score:5.1f}] {r.video_id} ({r.category})")


if __name__ == "__main__":
    main()
