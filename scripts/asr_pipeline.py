"""
ASR Pipeline: Extract speech transcription from ultrasound teaching videos
===========================================================================
Uses faster-whisper for efficient CPU/GPU transcription with timestamps.

Pipeline:
    Video (.mp4) → Extract audio (ffmpeg) → Whisper ASR → JSON with timestamps

Usage:
    python asr_pipeline.py --video path/to/video.mp4
    python asr_pipeline.py --input-dir path/to/videos/ --batch
    python asr_pipeline.py --video path.mp4 --model large-v3

Requirements:
    pip install faster-whisper
    ffmpeg (system install, already available)
"""

import json
import time
import subprocess
import tempfile
import argparse
from pathlib import Path
from datetime import datetime


# ============================================================================
# Configuration
# ============================================================================

DEFAULT_VIDEO = 'UltrasoundCrawler_KeyCode_20260323_v2/output/20260520_162816_youtube/media/scan_tutorial/TlckvYhqaFE.mp4' # "UltrasoundCrawler_KeyCode_20260323_v2/output/20260520_162816_youtube/media/case_reasoning/8V649L5Q368.mp4"
DEFAULT_MODEL = "base"  # Options: tiny, base, small, medium, large-v3
DEFAULT_OUTPUT_DIR = "results/transcripts"


# ============================================================================
# Audio Extraction
# ============================================================================

def extract_audio(video_path, output_path=None, sample_rate=16000):
    """Extract audio from video using ffmpeg. Returns path to wav file."""
    video_path = Path(video_path)
    if output_path is None:
        output_path = video_path.with_suffix(".wav")
    else:
        output_path = Path(output_path)

    output_path.parent.mkdir(parents=True, exist_ok=True)

    cmd = [
        "ffmpeg", "-y",
        "-i", str(video_path),
        "-vn",                      # no video
        "-acodec", "pcm_s16le",     # 16-bit PCM
        "-ar", str(sample_rate),    # sample rate
        "-ac", "1",                 # mono
        str(output_path)
    ]

    print(f"  Extracting audio: {video_path.name} → {output_path.name}")
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        print(f"  ERROR: ffmpeg failed: {result.stderr[:200]}")
        return None

    size_mb = output_path.stat().st_size / (1024 * 1024)
    print(f"  Audio extracted: {size_mb:.1f} MB")
    return str(output_path)


# ============================================================================
# Whisper Transcription
# ============================================================================

def transcribe_audio(audio_path, model_size="base", language=None, device="cpu"):
    """Transcribe audio using faster-whisper. Returns segments with timestamps."""
    from faster_whisper import WhisperModel

    print(f"  Loading Whisper model: {model_size} (device: {device})")
    # For Mac: use "cpu" with int8 for speed; for GPU: use "cuda" with float16
    compute_type = "int8" if device == "cpu" else "float16"
    model = WhisperModel(model_size, device=device, compute_type=compute_type)

    print(f"  Transcribing: {Path(audio_path).name}")
    t0 = time.time()

    segments_gen, info = model.transcribe(
        audio_path,
        language=language,
        beam_size=5,
        vad_filter=True,         # Filter out silence
        vad_parameters=dict(
            min_silence_duration_ms=500,
            speech_pad_ms=200,
        ),
    )

    # Collect segments
    segments = []
    for seg in segments_gen:
        segments.append({
            "start": round(seg.start, 2),
            "end": round(seg.end, 2),
            "text": seg.text.strip(),
        })

    elapsed = time.time() - t0
    print(f"  Done! {len(segments)} segments in {elapsed:.1f}s")
    print(f"  Language: {info.language} (prob: {info.language_probability:.2f})")
    print(f"  Duration: {info.duration:.1f}s | Speed: {info.duration/elapsed:.1f}x realtime")

    return {
        "segments": segments,
        "language": info.language,
        "language_probability": round(info.language_probability, 3),
        "duration_sec": round(info.duration, 2),
        "transcription_time_sec": round(elapsed, 2),
        "speed_factor": round(info.duration / elapsed, 2),
    }


# ============================================================================
# Full Pipeline
# ============================================================================

def transcribe_video(video_path, model_size="base", language=None,
                     output_dir=None, keep_audio=False):
    """Full pipeline: video → audio → transcription → JSON."""
    video_path = Path(video_path)
    video_id = video_path.stem

    print(f"\n{'='*70}")
    print(f"ASR Pipeline: {video_id}")
    print(f"{'='*70}")
    print(f"  Video: {video_path}")
    print(f"  Model: {model_size}")

    # Step 1: Extract audio
    if output_dir:
        audio_dir = Path(output_dir) / "audio"
        audio_dir.mkdir(parents=True, exist_ok=True)
        audio_path = str(audio_dir / f"{video_id}.wav")
    else:
        # Use temp file
        tmp = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
        audio_path = tmp.name
        tmp.close()

    audio_file = extract_audio(video_path, audio_path)
    if audio_file is None:
        return None

    # Step 2: Transcribe
    result = transcribe_audio(audio_file, model_size=model_size, language=language)

    # Step 3: Build output
    output = {
        "video_id": video_id,
        "video_path": str(video_path),
        "model": model_size,
        "language": result["language"],
        "language_probability": result["language_probability"],
        "duration_sec": result["duration_sec"],
        "transcription_time_sec": result["transcription_time_sec"],
        "speed_factor": result["speed_factor"],
        "num_segments": len(result["segments"]),
        "segments": result["segments"],
        "full_text": " ".join(seg["text"] for seg in result["segments"]),
        "timestamp": datetime.now().isoformat(),
    }

    # Step 4: Save
    if output_dir:
        out_path = Path(output_dir) / f"{video_id}.json"
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(output, f, ensure_ascii=False, indent=2)
        print(f"\n  Saved: {out_path}")

    # Cleanup audio if not keeping
    if not keep_audio and not output_dir:
        Path(audio_file).unlink(missing_ok=True)

    # Print preview
    print(f"\n  Preview (first 5 segments):")
    for seg in output["segments"][:5]:
        print(f"    [{seg['start']:.1f}-{seg['end']:.1f}s] {seg['text'][:60]}")
    if len(output["segments"]) > 5:
        print(f"    ... ({len(output['segments'])-5} more segments)")

    return output


# ============================================================================
# Batch Processing
# ============================================================================

def batch_transcribe(input_dir, model_size="base", output_dir=None, max_videos=None):
    """Transcribe all videos in a directory."""
    import glob

    input_dir = Path(input_dir)
    if output_dir is None:
        output_dir = str(input_dir.parent / "transcripts")

    # Find videos
    videos = []
    for ext in ["*.mp4", "*.webm", "*.mkv", "*.avi"]:
        videos.extend(glob.glob(str(input_dir / "**" / ext), recursive=True))
    videos = sorted(videos)

    if max_videos:
        videos = videos[:max_videos]

    # Check existing transcripts (skip already done)
    out_path = Path(output_dir)
    done = set(p.stem for p in out_path.glob("*.json")) if out_path.exists() else set()
    remaining = [v for v in videos if Path(v).stem not in done]

    print(f"Batch ASR Pipeline")
    print(f"  Input: {input_dir}")
    print(f"  Output: {output_dir}")
    print(f"  Videos: {len(videos)} total | {len(done)} done | {len(remaining)} remaining")
    print(f"  Model: {model_size}")
    print(f"{'='*70}")

    results = []
    for i, video in enumerate(remaining, 1):
        print(f"\n[{i}/{len(remaining)}] {Path(video).stem}")
        try:
            result = transcribe_video(video, model_size=model_size, output_dir=output_dir)
            if result:
                results.append(result)
        except Exception as e:
            print(f"  ERROR: {e}")

    # Summary
    print(f"\n{'='*70}")
    print(f"Batch complete: {len(results)}/{len(remaining)} successful")
    if results:
        total_dur = sum(r["duration_sec"] for r in results)
        total_time = sum(r["transcription_time_sec"] for r in results)
        print(f"  Total audio: {total_dur/60:.1f} min")
        print(f"  Total time: {total_time/60:.1f} min")
        print(f"  Avg speed: {total_dur/total_time:.1f}x realtime")

    return results


# ============================================================================
# CLI
# ============================================================================

def main():
    parser = argparse.ArgumentParser(description="ASR Pipeline for Ultrasound Videos")
    parser.add_argument("--video", type=str, default=None, help="Single video path")
    parser.add_argument("--input-dir", type=str, default=None, help="Directory of videos (batch mode)")
    parser.add_argument("--output-dir", type=str, default=DEFAULT_OUTPUT_DIR, help="Output directory")
    parser.add_argument("--model", type=str, default=DEFAULT_MODEL,
                        help="Whisper model size: tiny/base/small/medium/large-v3")
    parser.add_argument("--language", type=str, default=None, help="Force language (e.g., 'en')")
    parser.add_argument("--batch", action="store_true", help="Batch mode")
    parser.add_argument("--max-videos", type=int, default=None, help="Limit videos in batch")
    parser.add_argument("--keep-audio", action="store_true", help="Keep extracted .wav files")
    args = parser.parse_args()

    if args.video:
        transcribe_video(args.video, model_size=args.model, language=args.language,
                         output_dir=args.output_dir, keep_audio=args.keep_audio)
    elif args.input_dir or args.batch:
        input_dir = args.input_dir or "UltrasoundCrawler_KeyCode_20260323_v2/output/20260520_162816_youtube/media"
        batch_transcribe(input_dir, model_size=args.model,
                         output_dir=args.output_dir, max_videos=args.max_videos)
    else:
        # Default: transcribe the default video
        transcribe_video(DEFAULT_VIDEO, model_size=args.model, language=args.language,
                         output_dir=args.output_dir, keep_audio=args.keep_audio)


if __name__ == "__main__":
    main()