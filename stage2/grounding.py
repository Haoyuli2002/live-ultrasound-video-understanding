"""Content profiles and half-open visual evidence contracts for Stage 2.

These checks prove provenance/time consistency, not clinical correctness. Raw
responses must be saved before calling them; findings never erase raw text.
"""
from __future__ import annotations
import math
from collections import Counter

VIDEO_TYPES = ('pure_ultrasound', 'ultrasound_tutorial', 'ppt_lecture', 'mixed', 'uncertain')
STAGE_VALUES = ('high', 'medium', 'low')
WORKFLOW_VERSION = 'evidence_bound_v3'

PROFILE_PROMPT = '''Inspect the entire silent visual overview and classify the video.
Return JSON with video_type, video_type_reason, visual_quality,
visual_quality_reason, and stage_value (stage1, stage2, stage3 each containing
value and reason). video_type must be pure_ultrasound (mostly sonographic
images), ultrasound_tutorial (an instructor demonstrates scanning on a
patient/phantom alongside sonographic images), ppt_lecture (slide-led lecture,
including slides containing ultrasound examples), mixed (substantial distinct
presentation modes), or uncertain. Stage values must be high, medium, or low,
with concrete visual reasons: Stage1 visual/narration alignment potential,
Stage2 grounded summary and memory potential, Stage3 future observation/QA
potential. Do not claim ASR or narration quality without audio. Judge visible
clarity and content coverage; don't diagnose patients. Classification is a
whole-video metadata pass, never evidence for an earlier local/prefix label.'''

def _finite(value):
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value)

def audit_profile(profile: dict) -> list[str]:
    issues = []
    if profile.get('video_type') not in VIDEO_TYPES:
        issues.append('invalid_video_type')
    for field in ('video_type_reason', 'visual_quality_reason'):
        if not isinstance(profile.get(field), str) or not profile[field].strip():
            issues.append('missing_' + field)
    ratings = profile.get('stage_value', {})
    for stage in ('stage1', 'stage2', 'stage3'):
        rating = ratings.get(stage, {}) if isinstance(ratings, dict) else {}
        if not isinstance(rating, dict) or rating.get('value') not in STAGE_VALUES:
            issues.append('invalid_' + stage + '_value')
        if not isinstance(rating, dict) or not isinstance(rating.get('reason'), str) or not rating['reason'].strip():
            issues.append('missing_' + stage + '_reason')
    return issues

def audit_window(record: dict, frames: dict[str, dict], *, expected_source_sha256: str) -> list[str]:
    """Each claim references actual registered source frames inside [start,end).

    A frame's actual decoded source PTS is required. A fabricated nominal bin
    midpoint or the timestamp printed on a contact sheet is not a source PTS.
    Dynamic claims additionally need an explicitly reviewed dense sequence;
    the existence of two different frames alone does not establish physiology.
    """
    issues = []
    window = record.get('window')
    if not isinstance(window, (list, tuple)) or len(window) != 2 or not all(_finite(t) for t in window):
        return ['invalid_window']
    start, end = window
    if start < 0 or end <= start:
        return ['invalid_window']
    if record.get('source_sha256') != expected_source_sha256:
        issues.append('source_identity_mismatch')
    if record.get('label_type') == 'global_prefix' and start != 0:
        issues.append('global_must_start_at_zero')
    if not isinstance(record.get('target'), str) or not record['target'].strip():
        issues.append('empty_target')
    claims = record.get('claims')
    if not isinstance(claims, list) or not claims:
        return issues + ['missing_claim_level_evidence']
    for index, claim in enumerate(claims):
        prefix = f'claim_{index}:'
        if not isinstance(claim, dict):
            issues.append(prefix + 'invalid_claim'); continue
        if not isinstance(claim.get('observation'), str) or not claim['observation'].strip():
            issues.append(prefix + 'missing_observation')
        references = claim.get('frame_ids', [])
        if not references:
            issues.append(prefix + 'missing_evidence')
        for identifier in references:
            frame = frames.get(identifier)
            if frame is None:
                issues.append(prefix + 'unregistered_frame'); continue
            time = frame.get('source_pts_seconds')
            if not _finite(time) or not start <= time < end:
                issues.append(prefix + 'outside_window_evidence')
            if frame.get('source_sha256') != expected_source_sha256:
                issues.append(prefix + 'frame_source_mismatch')
            if frame.get('timestamp_kind') != 'decoded_source_pts':
                issues.append(prefix + 'unverified_timestamp')
        if claim.get('kind') == 'motion' and not claim.get('dense_sequence_reviewed'):
            issues.append(prefix + 'motion_requires_dense_review')
        if claim.get('kind') in ('measurement', 'screen_label'):
            if not claim.get('on_screen_text'):
                issues.append(prefix + 'missing_screen_transcription')
        if claim.get('kind') == 'measurement' and not claim.get('label_association'):
            issues.append(prefix + 'measurement_label_association_missing')
    return list(dict.fromkeys(issues))

def audit_split_assignments(videos: list[dict]) -> list[str]:
    identities = {}; issues = []
    for video in videos:
        vid, split = video.get('video_id'), video.get('split')
        if not vid or split not in ('train', 'eval', 'unspecified'):
            issues.append('missing_video_or_split'); continue
        prior = identities.setdefault(vid, split)
        if prior != split:
            issues.append(f'{vid}:train_eval_overlap')
    return sorted(set(issues))

def summarize_review(records: list[dict]) -> dict:
    """No implicit acceptance, semantic certification, or automatic filtering."""
    counts = Counter(issue for r in records for issue in r.get('issues', []))
    return {'total': len(records), 'flagged': sum(bool(r.get('issues')) for r in records),
            'issue_counts': dict(counts), 'raw_preserved': True,
            'clinical_grounding_verified': False, 'training_approved': False}
