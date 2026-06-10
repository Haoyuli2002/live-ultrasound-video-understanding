"""
Video Segmentation Pipeline
============================
Enhanced segmentation combining visual scene detection + ASR sentence boundaries.

Approach:
1. Detect visual scene changes (histogram comparison between frames at each ASR segment)
2. Find sentence-ending ASR segments near each scene change
3. Cut at these natural boundaries
4. For long segments without scene changes, force-cut at nearest sentence end

Usage:
    python scripts/video_segmentation.py --video path.mp4 --transcript transcripts/ID.json
    python scripts/video_segmentation.py --batch --transcript-dir transcripts/ --video-dir media/
"""

import json
import argparse
from pathlib import Path

import cv2
import numpy as np


# ============================================================================
# Visual Scene Change Detection
# ============================================================================

def compute_segment_visual_changes(video_path, segments, threshold=0.3):
    """
    For each ASR segment, extract a frame at the midpoint and compare
    its grayscale histogram to the previous segment's frame.
    
    Returns list of dicts with similarity scores and scene_change flags.
    """
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        print(f"  WARNING: Cannot open video {video_path}")
        return []

    prev_hist = None
    results = []

    for i, seg in enumerate(segments):
        mid_time = (seg['start'] + seg['end']) / 2
        cap.set(cv2.CAP_PROP_POS_MSEC, mid_time * 1000)
        ret, frame = cap.read()
        if not ret:
            results.append({'seg_idx': i, 'start': seg['start'], 'end': seg['end'],
                            'similarity': None, 'scene_change': False})
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

        results.append({
            'seg_idx': i,
            'start': seg['start'],
            'end': seg['end'],
            'similarity': round(similarity, 3),
            'scene_change': scene_change
        })
        prev_hist = hist

    cap.release()
    return results


# ============================================================================
# Enhanced Segmentation
# ============================================================================

def segment_enhanced(segments, visual_changes, min_clip=30, max_clip=300, tolerance=10):
    """
    Enhanced segmentation using visual scene changes as priority cut points,
    with sentence boundary alignment.

    Strategy:
    1. Use scene_change points as primary cut candidates
    2. For each scene_change, find nearest sentence-ending segment (within tolerance)
    3. Build clips between cut points
    4. If a clip exceeds max_clip, force-cut at nearest sentence boundary

    Args:
        segments: list of {"start", "end", "text"} from ASR
        visual_changes: output of compute_segment_visual_changes()
        min_clip: minimum clip duration in seconds (default 30)
        max_clip: maximum clip duration before force-cut (default 300)
        tolerance: seconds to search for sentence end near scene change (default 10)

    Returns:
        list of clip dicts with start, end, duration, text, cut_reason
    """
    if not segments or not visual_changes:
        return []

    # Get scene change times
    scene_change_times = [v['start'] for v in visual_changes if v.get('scene_change')]

    # Find all sentence-ending segment indices
    sentence_end_indices = []
    for i, s in enumerate(segments):
        if s['text'].rstrip().endswith(('.', '?', '!')):
            sentence_end_indices.append(i)

    # For each scene change, find best cut point (sentence end within tolerance)
    cut_indices = set()
    for sc_time in scene_change_times:
        best_idx = None
        best_dist = float('inf')
        for idx in sentence_end_indices:
            seg_end = segments[idx]['end']
            dist = abs(seg_end - sc_time)
            if dist <= tolerance and dist < best_dist:
                best_idx = idx
                best_dist = dist
        if best_idx is not None:
            cut_indices.add(best_idx)

    cut_indices = sorted(cut_indices)

    # Build clips from cut indices
    clips = []
    clip_start_idx = 0

    for cut_idx in cut_indices:
        clip_start_time = segments[clip_start_idx]['start']
        clip_end_time = segments[cut_idx]['end']
        duration = clip_end_time - clip_start_time

        if duration >= min_clip:
            clip_segs = segments[clip_start_idx:cut_idx + 1]
            clips.append({
                'clip_idx': len(clips),
                'start': round(clip_start_time, 2),
                'end': round(clip_end_time, 2),
                'duration': round(duration, 2),
                'num_segments': len(clip_segs),
                'text': ' '.join(s['text'] for s in clip_segs),
                'cut_reason': 'scene_change'
            })
            clip_start_idx = cut_idx + 1

    # Handle remaining after last cut
    if clip_start_idx < len(segments):
        clip_start_time = segments[clip_start_idx]['start']
        clip_end_time = segments[-1]['end']
        duration = clip_end_time - clip_start_time
        if duration >= min_clip:
            clip_segs = segments[clip_start_idx:]
            clips.append({
                'clip_idx': len(clips),
                'start': round(clip_start_time, 2),
                'end': round(clip_end_time, 2),
                'duration': round(duration, 2),
                'num_segments': len(clip_segs),
                'text': ' '.join(s['text'] for s in clip_segs),
                'cut_reason': 'end_of_video'
            })

    # Post-process: split clips exceeding max_clip at sentence boundaries
    final_clips = []
    for clip in clips:
        if clip['duration'] <= max_clip:
            clip['clip_idx'] = len(final_clips)
            final_clips.append(clip)
        else:
            # Find segments within this clip
            clip_segs = [s for s in segments
                         if s['start'] >= clip['start'] and s['end'] <= clip['end']]
            sub_start = 0
            for si in range(len(clip_segs)):
                sub_dur = clip_segs[si]['end'] - clip_segs[sub_start]['start']
                if sub_dur >= max_clip:
                    # Find nearest sentence end before this point
                    for back in range(si - 1, sub_start, -1):
                        if clip_segs[back]['text'].rstrip().endswith(('.', '?', '!')):
                            sub_text = ' '.join(cs['text'] for cs in clip_segs[sub_start:back + 1])
                            final_clips.append({
                                'clip_idx': len(final_clips),
                                'start': round(clip_segs[sub_start]['start'], 2),
                                'end': round(clip_segs[back]['end'], 2),
                                'duration': round(clip_segs[back]['end'] - clip_segs[sub_start]['start'], 2),
                                'num_segments': back - sub_start + 1,
                                'text': sub_text,
                                'cut_reason': 'max_clip_split'
                            })
                            sub_start = back + 1
                            break
            # Remaining
            if sub_start < len(clip_segs):
                sub_dur = clip_segs[-1]['end'] - clip_segs[sub_start]['start']
                if sub_dur >= min_clip:
                    sub_text = ' '.join(cs['text'] for cs in clip_segs[sub_start:])
                    final_clips.append({
                        'clip_idx': len(final_clips),
                        'start': round(clip_segs[sub_start]['start'], 2),
                        'end': round(clip_segs[-1]['end'], 2),
                        'duration': round(sub_dur, 2),
                        'num_segments': len(clip_segs) - sub_start,
                        'text': sub_text,
                        'cut_reason': 'max_clip_remainder'
                    })

    return final_clips


# ============================================================================
# Full Pipeline
# ============================================================================

def segment_video(video_path, transcript_path, output_dir=None,
                  min_clip=30, max_clip=300, visual_threshold=0.3,
                  use_llm=True, api_key=None):
    """
    Full segmentation pipeline.
    Default: histogram candidates + LLM verification.
    Set use_llm=False for histogram-only mode.
    """
    video_path = Path(video_path)
    transcript_path = Path(transcript_path)

    with open(transcript_path) as f:
        asr_data = json.load(f)

    segments = asr_data['segments']
    video_id = asr_data.get('video_id', video_path.stem)

    print(f"\n{'='*70}")
    print(f"Segmenting: {video_id}")
    print(f"  Duration: {asr_data['duration_sec']:.1f}s | Segments: {len(segments)}")
    print(f"  Mode: {'Histogram + LLM' if use_llm else 'Histogram only'}")

    # Step 1: Visual scene detection
    print(f"  Step 1: Visual histogram analysis (threshold={visual_threshold})...")
    visual_changes = compute_segment_visual_changes(str(video_path), segments, visual_threshold)
    num_changes = sum(1 for v in visual_changes if v.get('scene_change'))
    print(f"  Found {num_changes} candidate cut points")

    # Step 2: Segmentation
    if use_llm:
        from llm_segmentation import verify_cuts_with_llm, segment_with_llm
        print(f"  Step 2: LLM verification...")
        llm_result = verify_cuts_with_llm(segments, visual_changes, api_key=api_key)
        clips = segment_with_llm(segments, llm_result, min_clip=min_clip)
        method = 'histogram_llm'
    else:
        print(f"  Step 2: Histogram-only segmentation...")
        clips = segment_enhanced(segments, visual_changes, min_clip, max_clip)
        llm_result = None
        method = 'histogram_only'

    total_clip_dur = sum(c['duration'] for c in clips)
    coverage = total_clip_dur / asr_data['duration_sec'] * 100

    print(f"  Result: {len(clips)} clips | Coverage: {coverage:.0f}%")
    for c in clips:
        topic = f" | {c['topic']}" if c.get('topic') else ""
        print(f"    Clip {c['clip_idx']:2d}: {c['start']:6.1f}-{c['end']:6.1f}s "
              f"({c['duration']:5.1f}s){topic}")

    # Save
    output = {
        'video_id': video_id,
        'video_path': str(video_path),
        'duration_sec': asr_data['duration_sec'],
        'method': method,
        'num_clips': len(clips),
        'coverage_pct': round(coverage, 1),
        'clips': clips,
    }
    if llm_result:
        output['llm_result'] = llm_result

    if output_dir:
        out_path = Path(output_dir) / f"{video_id}_clips.json"
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with open(out_path, 'w', encoding='utf-8') as f:
            json.dump(output, f, ensure_ascii=False, indent=2)
        print(f"  Saved: {out_path}")

    return output

# ============================================================================
# CLI
# ============================================================================
def main():
    parser = argparse.ArgumentParser(description="Video Segmentation Pipeline")
    parser.add_argument("--video", type=str, help="Single video path")
    parser.add_argument("--transcript", type=str, help="Transcript JSON path")
    parser.add_argument("--output-dir", type=str, default="results/clips", help="Output directory")
    parser.add_argument("--min-clip", type=int, default=30)
    parser.add_argument("--max-clip", type=int, default=300)
    parser.add_argument("--visual-threshold", type=float, default=0.4)
    parser.add_argument("--no-llm", action="store_true", help="Disable LLM, use histogram only")
    parser.add_argument("--api-key", type=str, default=None, help="OpenAI API key")
    parser.add_argument("--batch", action="store_true", help="Batch mode")
    parser.add_argument("--transcript-dir", type=str, default="transcripts")
    parser.add_argument("--video-dir", type=str)
    args = parser.parse_args()

    use_llm = not args.no_llm

    if args.video and args.transcript:
        segment_video(args.video, args.transcript, args.output_dir,
                      args.min_clip, args.max_clip, args.visual_threshold,
                      use_llm=use_llm, api_key=args.api_key)
    elif args.batch:
        transcript_dir = Path(args.transcript_dir)
        transcripts = sorted(transcript_dir.glob("*.json"))
        print(f"Batch segmentation: {len(transcripts)} transcripts")
        for tp in transcripts:
            with open(tp) as f:
                data = json.load(f)
            vp = Path(data.get('video_path', ''))
            if not vp.exists() and args.video_dir:
                vp = Path(args.video_dir) / vp.name
            if vp.exists():
                try:
                    segment_video(str(vp), str(tp), args.output_dir,
                                  args.min_clip, args.max_clip, args.visual_threshold,
                                  use_llm=use_llm, api_key=args.api_key)
                except Exception as e:
                    print(f"  ERROR: {e}")
            else:
                print(f"  SKIP: Video not found for {tp.stem}")
    else:
        video = "UltrasoundCrawler_KeyCode_20260323_v2/output/20260520_162816_youtube/media/case_reasoning/8V649L5Q368.mp4"
        transcript = "transcripts/8V649L5Q368.json"
        if Path(transcript).exists():
            segment_video(video, transcript, args.output_dir,
                          args.min_clip, args.max_clip, args.visual_threshold,
                          use_llm=use_llm, api_key=args.api_key)
        else:
            print("Run ASR first: python scripts/asr_pipeline.py")
