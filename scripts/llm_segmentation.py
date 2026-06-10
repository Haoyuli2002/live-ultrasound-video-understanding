"""
LLM-based Segmentation Verification
=====================================
Sends ASR segments + visual candidate cuts to GPT-4o for semantic verification.
LLM decides which cuts to keep, remove, or add new ones.

Usage:
    from llm_segmentation import verify_cuts_with_llm, segment_with_llm
"""

import os
import json
import time
import re
from pathlib import Path


def verify_cuts_with_llm(segments, visual_changes, api_key=None):
    """
    Send all ASR segments + visual candidate cuts to GPT-4o.
    LLM decides which cuts to keep, remove, or add.
    """
    from openai import OpenAI
    
    api_key = api_key or os.environ.get("OPENAI_API_KEY", "")
    if not api_key:
        raise ValueError("OPENAI_API_KEY not set")
    
    client = OpenAI(api_key=api_key)
    
    # Format transcript
    transcript_lines = []
    for i, seg in enumerate(segments):
        transcript_lines.append(f"[{i}] [{seg['start']:.1f}-{seg['end']:.1f}s] {seg['text']}")
    transcript_text = "\n".join(transcript_lines)
    
    # Format candidate cuts
    candidates = [v for v in visual_changes if v.get('scene_change')]
    candidates_text = "\n".join(
        f"  Seg {v['seg_idx']} at t={v['start']:.1f}s (histogram_similarity={v['similarity']:.3f})"
        for v in candidates
    )
    
    prompt = f"""You are a medical video editor specializing in ultrasound teaching content.

Given the full ASR transcript and candidate segmentation points from visual histogram analysis, 
determine the optimal video segmentation.

## Full ASR Transcript ({len(segments)} segments):
{transcript_text}

## Candidate Cut Points from Visual Analysis ({len(candidates)} candidates):
{candidates_text}

## Task:
1. For each visual candidate, decide if it's a real topic/content change (not just a camera zoom)
2. Identify any topic changes MISSED by visual analysis
3. For each cut point, provide a "topic" label = what the content is about BEFORE this cut

## Rules:
- Clips should be 30-300 seconds
- Visual change alone (zoom in/out on same content) is NOT a valid cut
- Topic changes in speech without visual changes SHOULD be added
- Each cut should represent a meaningful content boundary
- "topic" describes the content of the section that ENDS at this cut point

## Output ONLY valid JSON:
{{"final_cuts": [{{"segment_index": 0, "time": 0.0, "should_cut": true, "topic": "topic of content before this cut"}}], "additional_cuts": [{{"segment_index": 0, "time": 0.0, "topic": "topic of content before this cut", "reason": "why added"}}], "last_topic": "topic of the final section after all cuts"}}"""

    print("  Calling GPT-4o for cut verification...")
    t0 = time.time()
    
    response = client.chat.completions.create(
        model="gpt-4o",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.2,
        max_tokens=2000,
    )
    
    elapsed = time.time() - t0
    raw = response.choices[0].message.content
    tokens = response.usage.total_tokens if response.usage else 0
    print(f"  LLM response: {elapsed:.1f}s | {tokens} tokens")
    
    # Parse JSON
    try:
        result = json.loads(raw)
    except json.JSONDecodeError:
        m = re.search(r'\{.*\}', raw, re.DOTALL)
        if m:
            try:
                result = json.loads(m.group(0))
            except:
                result = {"final_cuts": [], "additional_cuts": [], "_parse_error": True, "_raw": raw[:500]}
        else:
            result = {"final_cuts": [], "additional_cuts": [], "_parse_error": True, "_raw": raw[:500]}
    
    return result


def segment_with_llm(segments, llm_result, min_clip=30):
    """
    Build clips using LLM-verified cut points.
    Each clip gets the "topic" from its ending cut point.
    """
    # Collect cut indices and their topic labels
    cut_indices = set()
    topic_at_cut = {}  # topic_at_cut[idx] = "topic of content before this cut"
    
    for cut in llm_result.get('final_cuts', []):
        if cut.get('should_cut', True):
            idx = cut.get('segment_index', 0)
            if 0 <= idx < len(segments):
                cut_indices.add(idx)
                topic_at_cut[idx] = cut.get('topic', cut.get('topic_before', ''))
    
    for cut in llm_result.get('additional_cuts', []):
        idx = cut.get('segment_index', 0)
        if 0 <= idx < len(segments):
            cut_indices.add(idx)
            topic_at_cut[idx] = cut.get('topic', cut.get('topic_before', ''))
    
    last_topic = llm_result.get('last_topic', '')
    
    if not cut_indices:
        return [{
            'clip_idx': 0, 'start': segments[0]['start'], 'end': segments[-1]['end'],
            'duration': round(segments[-1]['end'] - segments[0]['start'], 2),
            'num_segments': len(segments),
            'text': ' '.join(s['text'] for s in segments),
            'cut_reason': 'no_cuts', 'topic': 'full_video'
        }]
    
    cut_indices = sorted(cut_indices)
    
    # Build clips: each clip's topic = topic_at_cut of the cut that ends it
    clips = []
    clip_start_idx = 0
    
    for cut_idx in cut_indices:
        if cut_idx <= clip_start_idx:
            continue
        clip_start_time = segments[clip_start_idx]['start']
        clip_end_time = segments[cut_idx - 1]['end']
        duration = clip_end_time - clip_start_time
        
        if duration >= min_clip:
            clip_segs = segments[clip_start_idx:cut_idx]
            topic = topic_at_cut.get(cut_idx, '')
            clips.append({
                'clip_idx': len(clips),
                'start': round(clip_start_time, 2),
                'end': round(clip_end_time, 2),
                'duration': round(duration, 2),
                'num_segments': len(clip_segs),
                'text': ' '.join(s['text'] for s in clip_segs),
                'cut_reason': 'llm_verified',
                'topic': topic
            })
        clip_start_idx = cut_idx
    
    # Last clip
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
                'cut_reason': 'end_of_video',
                'topic': last_topic
            })
    
    return clips

def evaluate_segmentation(segments, clips, api_key=None):
    """
    让 GPT-4o 评估切分质量。
    """
    from openai import OpenAI
    
    api_key = api_key or os.environ.get("OPENAI_API_KEY", "")
    client = OpenAI(api_key=api_key)
    
    # 格式化 clips
    clips_text = ""
    for c in clips:
        clips_text += f"Clip {c['clip_idx']}: {c['start']:.0f}-{c['end']:.0f}s ({c['duration']:.0f}s)"
        if c.get('topic'):
            clips_text += f" | Topic: {c['topic']}"
        clips_text += f"\n  Text: {c['text'][:150]}...\n\n"
    
    prompt = f"""You are evaluating the quality of video segmentation for an ultrasound teaching video.

## ASR Transcript Summary:
Total duration: {segments[-1]['end']:.0f}s
Total segments: {len(segments)}

## Segmentation Result ({len(clips)} clips):
{clips_text}

## Evaluate each clip on:
1. Topic coherence (1-5): Does the clip contain one coherent topic?
2. Boundary quality (1-5): Are the start/end points at natural boundaries?
3. Duration appropriateness (1-5): Is the length reasonable (not too short/long)?
4. Content completeness (1-5): Does the clip contain a complete thought/section?

## Also identify:
- Any clip that splits a topic in the middle
- Any clip that combines unrelated topics
- Missing cut points (topics that should be separate but aren't)

## Output ONLY valid JSON:
{{"overall_score": 4.2, "clip_scores": [{{"clip_idx": 0, "coherence": 5, "boundary": 4, "duration": 5, "completeness": 4, "comment": "Good intro"}}], "issues": ["Clip X splits topic Y"], "suggestions": ["Merge clips X and Y"]}}"""

    print("  Evaluating segmentation quality...")
    t0 = time.time()
    
    response = client.chat.completions.create(
        model="gpt-4o",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.2,
        max_tokens=2000,
    )
    
    elapsed = time.time() - t0
    raw = response.choices[0].message.content
    tokens = response.usage.total_tokens if response.usage else 0
    print(f"  Evaluation done: {elapsed:.1f}s | {tokens} tokens")
    
    # Parse JSON
    try:
        result = json.loads(raw)
    except json.JSONDecodeError:
        m = re.search(r'\{.*\}', raw, re.DOTALL)
        if m:
            try:
                result = json.loads(m.group(0))
            except:
                result = {"overall_score": 0, "_parse_error": True, "_raw": raw[:500]}
        else:
            result = {"overall_score": 0, "_parse_error": True, "_raw": raw[:500]}
    
    return result