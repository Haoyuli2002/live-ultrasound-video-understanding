"""
End-to-End Pipeline Demo
==========================
Runs the complete pipeline on a single video:
  Step 3 : ASR Transcription
  Step 4 : Video Segmentation (histogram + LLM)
  Step 5a: Offline QA Generation     (scene_description, fine_grained, knowledge)
  Step 5b: Streaming QA Generation   (sonographer_intent, next_action_guidance)
  Step 5c: Streaming QA Validation   (Gemini 2.5 Pro via OpenRouter)

Usage:
    export OPENAI_API_KEY="sk-..."
    export OPENROUTER_API_KEY="sk-or-..."
    python scripts/run_pipeline.py --video path/to/video.mp4

    # Skip steps already done:
    python scripts/run_pipeline.py --video path.mp4 --skip-asr
    python scripts/run_pipeline.py --video path.mp4 --skip-segmentation
    python scripts/run_pipeline.py --video path.mp4 --skip-validation
"""

import argparse
import json
import sys
import time
from pathlib import Path

# Make scripts/ importable as a flat module set
sys.path.insert(0, str(Path(__file__).resolve().parent))


def main():
    parser = argparse.ArgumentParser(description="Run full pipeline on a single video")
    parser.add_argument("--video", type=str, required=True, help="Video file path")
    parser.add_argument("--output-dir", type=str, default="results",
                        help="Base output directory")
    parser.add_argument("--whisper-model", type=str, default="base",
                        help="Whisper model size")
    parser.add_argument("--skip-asr", action="store_true",
                        help="Skip ASR if already done")
    parser.add_argument("--skip-segmentation", action="store_true",
                        help="Skip segmentation if already done")
    parser.add_argument("--skip-offline-qa", action="store_true",
                        help="Skip offline QA generation")
    parser.add_argument("--skip-streaming-qa", action="store_true",
                        help="Skip streaming QA generation")
    parser.add_argument("--skip-validation", action="store_true",
                        help="Skip Gemini validation of streaming QA")
    parser.add_argument("--no-llm-segment", action="store_true",
                        help="Use histogram-only segmentation")
    parser.add_argument("--single-clip", type=int, default=None,
                        help="For QA steps, only process this clip index (debug)")
    parser.add_argument("--validator-model", type=str, default="google/gemini-2.5-pro",
                        help="OpenRouter model id for validator")
    args = parser.parse_args()

    video_path = Path(args.video)
    video_id = video_path.stem
    output_dir = Path(args.output_dir)

    print(f"\n{'='*70}")
    print(f"PIPELINE: {video_id}")
    print(f"{'='*70}")
    print(f"  Video: {video_path}")
    print(f"  Output: {output_dir}")
    t_start = time.time()

    # ======================================================================
    # Step 3: ASR Transcription
    # ======================================================================
    transcript_path = output_dir / "transcripts" / f"{video_id}.json"

    if args.skip_asr and transcript_path.exists():
        print(f"\n[Step 3] ASR: SKIPPED (already exists: {transcript_path})")
    else:
        print(f"\n[Step 3] ASR Transcription (model={args.whisper_model})")
        from asr_pipeline import transcribe_video
        transcribe_video(
            str(video_path),
            model_size=args.whisper_model,
            output_dir=str(output_dir / "transcripts"),
        )

    if not transcript_path.exists():
        print(f"  ERROR: Transcript not found at {transcript_path}")
        return

    # ======================================================================
    # Step 4: Video Segmentation
    # ======================================================================
    clips_path = output_dir / "clips" / f"{video_id}_clips.json"

    if args.skip_segmentation and clips_path.exists():
        print(f"\n[Step 4] Segmentation: SKIPPED (already exists: {clips_path})")
    else:
        method_label = 'histogram+LLM' if not args.no_llm_segment else 'histogram only'
        print(f"\n[Step 4] Video Segmentation ({method_label})")
        from video_segmentation import compute_segment_visual_changes, segment_enhanced

        with open(transcript_path) as f:
            asr_data = json.load(f)
        segments = asr_data['segments']

        visual_changes = compute_segment_visual_changes(
            str(video_path), segments, threshold=0.3
        )
        num_changes = sum(1 for v in visual_changes if v.get('scene_change'))
        print(f"  Visual changes: {num_changes}")

        if not args.no_llm_segment:
            from llm_segmentation import verify_cuts_with_llm, segment_with_llm
            llm_result = verify_cuts_with_llm(segments, visual_changes)
            clips = segment_with_llm(segments, llm_result, min_clip=30)
            method = 'histogram_llm'
        else:
            clips = segment_enhanced(segments, visual_changes, min_clip=30, max_clip=300)
            llm_result = None
            method = 'histogram_only'

        clips_output = {
            'video_id': video_id,
            'video_path': str(video_path),
            'duration_sec': asr_data['duration_sec'],
            'method': method,
            'num_clips': len(clips),
            'clips': clips,
        }
        if llm_result:
            clips_output['llm_result'] = llm_result

        clips_path.parent.mkdir(parents=True, exist_ok=True)
        with open(clips_path, 'w', encoding='utf-8') as f:
            json.dump(clips_output, f, ensure_ascii=False, indent=2)
        print(f"  Saved: {clips_path} ({len(clips)} clips)")

    # ======================================================================
    # Step 5a: Offline QA Generation
    #          (scene_description, fine_grained, knowledge)
    # ======================================================================
    offline_qa_path = output_dir / "qa" / f"{video_id}_offline_qa.json"

    if args.skip_offline_qa:
        print(f"\n[Step 5a] Offline QA: SKIPPED")
    else:
        print(f"\n[Step 5a] Offline QA Generation "
              f"(scene_description / fine_grained / knowledge)")
        from qa_generation import generate_qa_for_video
        generate_qa_for_video(
            str(video_path),
            clips_path=str(clips_path),
            output_dir=str(output_dir / "qa"),
            single_clip=args.single_clip,
        )

    # ======================================================================
    # Step 5b: Streaming QA Generation
    #          (sonographer_intent, next_action_guidance)
    # ======================================================================
    streaming_qa_path = output_dir / "qa" / f"{video_id}_streaming_qa.json"

    if args.skip_streaming_qa:
        print(f"\n[Step 5b] Streaming QA: SKIPPED")
    else:
        print(f"\n[Step 5b] Streaming QA Generation "
              f"(sonographer_intent / next_action_guidance)")
        from streaming_qa_generation import generate_streaming_qa_for_video
        generate_streaming_qa_for_video(
            str(video_path),
            clips_path=str(clips_path),
            output_dir=str(output_dir / "qa"),
            single_clip=args.single_clip,
        )

    # ======================================================================
    # Step 5c: Streaming QA Validation (Gemini 2.5 Pro via OpenRouter)
    # ======================================================================
    validated_qa_path = output_dir / "qa" / f"{video_id}_streaming_qa_validated.json"

    if args.skip_validation:
        print(f"\n[Step 5c] Validation: SKIPPED")
    else:
        if not streaming_qa_path.exists():
            print(f"\n[Step 5c] Validation: SKIPPED (no streaming QA at {streaming_qa_path})")
        else:
            print(f"\n[Step 5c] Streaming QA Validation "
                  f"(model={args.validator_model})")
            from qa_validator import validate_streaming_qa_file
            try:
                validate_streaming_qa_file(
                    str(streaming_qa_path),
                    str(video_path),
                    output_path=str(validated_qa_path),
                    model=args.validator_model,
                    drop_discarded=True,
                )
            except Exception as e:
                print(f"  ERROR during validation: {e}")
                print(f"  Hint: ensure OPENROUTER_API_KEY is exported.")

    # ======================================================================
    # Summary
    # ======================================================================
    elapsed = time.time() - t_start
    print(f"\n{'='*70}")
    print(f"PIPELINE COMPLETE: {video_id}")
    print(f"{'='*70}")
    print(f"  Total time: {elapsed/60:.1f} min")
    print(f"  Outputs:")
    print(f"    Transcript:        {transcript_path}")
    print(f"    Clips:             {clips_path}")
    print(f"    Offline QA:        {offline_qa_path}")
    print(f"    Streaming QA:      {streaming_qa_path}")
    print(f"    Validated stream:  {validated_qa_path}")


if __name__ == "__main__":
    main()