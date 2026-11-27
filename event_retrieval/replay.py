"""Replay a local video at 1 FPS and retrieve evidence or answer a pending question."""
import argparse
import bisect
import json
import math
from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageOps

from lavis.models.streamready.backbone_qwen3_5 import CLIP_MEAN, CLIP_STD, Qwen35Backbone
from .answer import AnswerDecoder, StreamingAnswerSession
from .index import EventIndex
from .qformer import EventQFormers
from .retrieve import QuestionSession, QwenRelevance


def frame_pixels(image):
    resized = ImageOps.pad(image.convert("RGB"), (256, 256), method=Image.Resampling.BICUBIC)
    pixels = torch.from_numpy(np.array(resized)).permute(2, 0, 1).float() / 255
    return (pixels - pixels.new_tensor(CLIP_MEAN)[:, None, None]) / pixels.new_tensor(CLIP_STD)[:, None, None]


@torch.inference_mode()
def push_frame(index, backbone, second, source_pts, image):
    """Online entry point: one observed RGB image, with its original PTS in seconds."""
    pixels = frame_pixels(image)
    patches = backbone.encode_frames(pixels[None, :, None])[0, 0]
    return index.push(second, source_pts, patches)


def observe_until(index, backbone, reader, pts, duration, t):
    if not math.isfinite(t) or not 0 <= t <= duration:
        raise ValueError(f"Decision {t} outside video duration {duration}")
    last = min(math.floor(t), math.ceil(duration) - 1)
    for s in range(index.detector.next_second, last + 1):
        i = bisect.bisect_right(pts, s) - 1
        if i < 0:
            raise ValueError(f"No observable frame at {s}s")
        frame = reader[i]
        rgb = frame.asnumpy() if hasattr(frame, 'asnumpy') else np.asarray(frame)
        push_frame(index, backbone, s, pts[i], Image.fromarray(rgb))
    if t == duration:
        index.finish(duration)


def index_source(video, model_path):
    stat = video.stat()
    return [str(video), stat.st_size, stat.st_mtime_ns, str(model_path.resolve()), "qwen-pad256-1fps-v1"]


def main():
    from decord import VideoReader, cpu

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--video', type=Path, required=True)
    parser.add_argument('--index', type=Path, required=True, help='Per-video SQLite file on scratch')
    parser.add_argument('--model', type=Path, required=True, help='Local Qwen3.5-2B snapshot')
    parser.add_argument('--qformer', type=Path, required=True, help='InstructBLIP initialization')
    parser.add_argument('--question', required=True)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument('--at', type=float, nargs='+', help='Inspect retrieval at these times')
    mode.add_argument('--ask-at', type=int, help='Ask at this second and generate every second until ANSWER or EOF')
    parser.add_argument('--until', type=int, help='Last decision second for --ask-at')
    parser.add_argument('--max-new-tokens', type=int, default=64)
    weights_group = parser.add_mutually_exclusive_group()
    weights_group.add_argument('--checkpoint', type=Path, help='Joint Q-Former and Qwen checkpoint')
    weights_group.add_argument('--aligned-qformers', type=Path, help='Trained EventQFormers state_dict')
    parser.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu')
    parser.add_argument('--retrieval-batch-size', type=int, default=4)
    args = parser.parse_args()
    if args.retrieval_batch_size < 1:
        parser.error("Retrieval batch size must be positive")
    video = args.video.resolve(strict=True)
    reader = VideoReader(str(video), ctx=cpu(0), num_threads=2)
    stamps = reader.get_frame_timestamp(np.arange(len(reader)))
    pts, duration = stamps[:, 0].tolist(), float(stamps[-1, 1])
    if args.ask_at is None:
        if args.until is not None or any(not math.isfinite(t) or not 0 <= t <= duration for t in args.at):
            parser.error("Use --at with decision times inside the video, or --ask-at with --until")
        times = sorted(set(args.at))
    else:
        stop = args.until if args.until is not None else duration
        if not 0 <= args.ask_at <= stop <= duration:
            parser.error("Expected 0 <= ask-at <= stop <= video duration")
        times = list(range(args.ask_at, math.floor(stop) + 1))
        if times[-1] < stop:
            times.append(stop)  # The EOF notification can arrive between two frames.
    backbone = Qwen35Backbone(args.model).to(args.device).eval()
    torch.manual_seed(0)  # Reproducible fresh Q-Former input/output projections.
    model = EventQFormers.from_instructblip(args.qformer, vision_width=backbone.vision_width,
                                            llm_width=backbone.llm_width).to(args.device).eval()
    if args.aligned_qformers is not None:
        weights = torch.load(args.aligned_qformers, map_location='cpu', weights_only=True, mmap=True)
        model.load_state_dict(weights, strict=True)
    decoder = AnswerDecoder(backbone, legacy_tags=args.aligned_qformers is not None)
    if args.checkpoint is not None:
        from .checkpoint import restore
        restore(args.checkpoint, model, backbone)
    source = index_source(video, args.model)
    index = EventIndex(args.index, video.stem, source)
    try:
        session = QuestionSession(index, args.question, QwenRelevance(backbone), model,
                                  retrieval_batch_size=args.retrieval_batch_size)
        pending = (StreamingAnswerSession(session, decoder, args.ask_at)
                   if args.ask_at is not None else None)
        for t in times:
            observe_until(index, backbone, reader, pts, duration, t)
            if pending is not None:
                decision = pending.step(t, max_new_tokens=args.max_new_tokens)
                print(json.dumps(decision), flush=True)
                if decision['action'] == 'ANSWER':
                    break
                continue
            result = session.at(t if t == duration else math.floor(t))
            print(json.dumps(dict(video_id=video.stem, at=t, question=args.question, events=result.sources,
                                  long_shape=list(result.prefix.shape), valid_tokens=int(result.mask.sum()))),
                  flush=True)
    finally:
        index.close()


if __name__ == '__main__':
    main()
