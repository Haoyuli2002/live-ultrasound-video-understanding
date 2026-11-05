"""Free-running reconstruction from frozen visual memory, with memory controls.

The reference is saved for comparison only; it never enters the decoding prefix.
Uses a full-prefix, cache-free greedy decoder to match the training forward path
without depending on Qwen3.5 hybrid-cache generation support for inputs_embeds.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import hashlib
import json
from pathlib import Path
import re
import statistics
import time

import torch
from .short_encoding import iter_short_windows

KINDS = {"short", "block", "global"}
CONDITIONS = {"real", "zero", "other_window"}


@torch.inference_mode()
def greedy_reconstruct(memory, tokens: torch.Tensor, kind: str,
                       max_new_tokens: int = 160) -> dict:
    """Decode without a reference, teacher forcing, sampling, or cached state."""
    if kind not in KINDS or max_new_tokens < 1:
        raise ValueError("Expected short/block/global and positive token budget")
    backbone = getattr(memory, "core_model", None)
    if backbone is None:
        backbone = memory.model
    forward = getattr(memory, "_forward", backbone)
    if backbone.training:
        raise ValueError("Reconstruction requires model.eval()")
    if tokens.ndim != 3 or tokens.shape[0] != 1 or tokens.shape[1] == 0:
        raise ValueError("Expected a nonempty [1, memory_tokens, hidden] tensor")
    # Exact prompt, order and special-token policy used in reconstruction_loss.
    prompt = f"Summarize the visible ultrasound evidence ({kind}):\n"
    ids = memory.tokenizer(prompt, add_special_tokens=False,
                           return_tensors="pt").input_ids.to(memory.device)
    prefix = torch.cat([tokens.to(memory.device), memory.embedding(ids)], dim=1)
    eos = memory.tokenizer.eos_token_id
    if eos is None:
        raise ValueError("Training reconstruction requires a tokenizer EOS token")
    generated = []
    stop = "max_new_tokens"
    start = time.monotonic()
    for _ in range(max_new_tokens):
        output = forward(
            inputs_embeds=prefix,
            attention_mask=torch.ones(prefix.shape[:2], device=memory.device,
                                      dtype=torch.long),
            use_cache=False, return_dict=True)
        next_id = output.logits[:, -1, :].argmax(dim=-1, keepdim=True)
        generated.append(int(next_id.item()))
        if generated[-1] == eos:
            stop = "eos"
            break
        prefix = torch.cat([prefix, memory.embedding(next_id)], dim=1)
    return {
        "prediction": memory.tokenizer.decode(generated, skip_special_tokens=True).strip(),
        "raw_decoded": memory.tokenizer.decode(generated, skip_special_tokens=False),
        "generated_token_ids": generated, "generated_tokens": len(generated),
        "stop_reason": stop, "decode_seconds": time.monotonic() - start,
    }


@torch.inference_mode()
def collect_memories(memory, blocks: list[dict], reader, kinds=None) -> list[dict]:
    """Re-encode each frame once; update long memory chronologically per video."""
    kinds = KINDS if kinds is None else set(kinds)
    if not kinds or not kinds <= KINDS:
        raise ValueError("Expected nonempty short/block/global kinds")
    result = []
    previous = None
    for row in blocks:
        start, end = row["block_window"]
        minute = []
        for first, current, label in iter_short_windows(memory, row, reader):
            current = current.detach()
            minute.append(current)
            if "short" in kinds:
                result.append({"kind": "short", "window": [first, first+10],
                               "memory": current.detach().clone(),
                               "reference": label["local_summary_target"]})
        if abs(end-start-60) <= 0.01:
            short = torch.cat(minute, dim=1)
            if "block" in kinds:
                result.append({"kind": "block", "window": [start, end],
                               "memory": short.detach().clone(),
                               "reference": row["block_summary_target"]})
            if "global" in kinds:
                previous = memory.update_long(previous, short).detach()
                result.append({"kind": "global", "window": [0, end],
                               "memory": previous.clone(),
                               "reference": row["global_summary_target"]})
    return result


def controlled_memory(records: list[dict], index: int, condition: str):
    """Controls change only decoder memory; real chronological state is intact.

    other_window is a deliberately mismatched diagnostic control, possibly from
    a later interval. It is not an online prediction or valid future evidence.
    """
    if condition not in CONDITIONS:
        raise ValueError(f"Unknown condition: {condition}")
    record = records[index]
    if condition == "real":
        return record["memory"], {"memory_window": record["window"]}
    if condition == "zero":
        return torch.zeros_like(record["memory"]), {"memory_window": None}
    same = [i for i, r in enumerate(records)
            if r["kind"] == record["kind"] and r["memory"].shape == record["memory"].shape]
    if len(same) < 2:
        return None, {"skipped": "No other same-kind window in this video"}
    donor = records[same[(same.index(index)+1) % len(same)]]
    return donor["memory"], {"memory_window": donor["window"],
                              "diagnostic_only_noncausal_control": True}


def text_metrics(prediction: str, reference: str) -> dict:
    """Lexical reconstruction checks; not clinical or visual-faithfulness scores."""
    pred = re.findall(r"[a-z0-9]+(?:'[a-z]+)?", prediction.lower())
    ref = re.findall(r"[a-z0-9]+(?:'[a-z]+)?", reference.lower())
    overlap = sum((Counter(pred) & Counter(ref)).values())
    dp = [0]*(len(ref)+1)
    for word in pred:
        new = [0]
        for j, target in enumerate(ref):
            new.append(dp[j]+1 if word == target else max(dp[j+1], new[-1]))
        dp = new
    return {"normalized_exact_match": bool(pred == ref),
            "word_overlap_f1": 2*overlap/max(1, len(pred)+len(ref)),
            "rouge_l_f1": 2*dp[-1]/max(1, len(pred)+len(ref)),
            "prediction_words": len(pred), "reference_words": len(ref)}


def summarize(rows: list[dict]) -> dict:
    groups = defaultdict(list)
    for row in rows:
        if "skipped" not in row:
            groups[(row["condition"], row["kind"])].append(row)
    return {f"{condition}/{kind}": {
        "samples": len(values),
        **{key: statistics.mean(float(v[key]) for v in values)
           for key in ["normalized_exact_match", "word_overlap_f1", "rouge_l_f1"]},
        "truncated": sum(v["stop_reason"] == "max_new_tokens" for v in values),
    } for (condition, kind), values in groups.items()}


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for data in iter(lambda: stream.read(1024*1024), b""):
            h.update(data)
    return h.hexdigest()


def main():
    from .data import VideoReader, load_blocks
    from .model import MemoryModel
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--eval-jsonl", required=True)
    parser.add_argument("--video-path-map", required=True)
    parser.add_argument("--adapter-path", required=True)
    parser.add_argument("--base-adapter")
    parser.add_argument("--model-name", default="Qwen/Qwen3.5-2B")
    parser.add_argument("--frame-size", type=int, default=224)
    parser.add_argument("--max-new-tokens", type=int, default=160)
    parser.add_argument("--conditions", nargs="+", choices=sorted(CONDITIONS),
                        default=["real", "zero", "other_window"])
    parser.add_argument("--kinds", nargs="+", choices=sorted(KINDS),
                        default=["short", "block", "global"])
    parser.add_argument("--save-memories", action="store_true")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--release-manifest")
    parser.add_argument("--limit-videos",type=int,default=0)
    args = parser.parse_args()
    if args.max_new_tokens < 1 or args.frame_size < 1:
        parser.error("Token budget and frame size must be positive")
    if not torch.cuda.is_available():
        raise RuntimeError("Qwen3.5 reconstruction requires an allocated CUDA GPU")
    adapter = Path(args.adapter_path).resolve()
    if not (adapter/"adapter_config.json").is_file():
        raise FileNotFoundError(adapter/"adapter_config.json")
    config_path = adapter.parent/"training_config.json"
    config = json.loads(config_path.read_text()) if config_path.is_file() else {}
    if args.frame_size != config.get("frame_size", args.frame_size):
        raise ValueError("Use the training frame size for reconstruction")
    base_adapter = args.base_adapter or config.get("init_adapter")
    if args.release_manifest:
        from .release import validate_release
        from .provenance import digest
        release=validate_release(args.release_manifest)
        if digest(args.eval_jsonl)!=release['files']['eval.jsonl']:
            raise ValueError('Independent evaluation must use the frozen EVAL split')
    if args.limit_videos<0:parser.error('Video limit must be nonnegative')
    videos = load_blocks(args.eval_jsonl,validate_text=not bool(args.release_manifest))
    if args.limit_videos:
        videos={v:videos[v] for v in sorted(videos)[:args.limit_videos]}
    paths = json.loads(Path(args.video_path_map).read_text())
    for vid in videos:
        if vid not in paths or not Path(paths[vid]).is_file():
            raise FileNotFoundError(f"Missing video for {vid}")
    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=False)
    manifest = {**vars(args), "adapter_path": str(adapter),
                "base_adapter": base_adapter, "decoding": "greedy_full_prefix_no_cache",
                "reference_used_in_decoding": False,
                "labels_sha256": sha256(Path(args.eval_jsonl)),
                "adapter_config_sha256": sha256(adapter/"adapter_config.json"),
                "source_sha256": {vid: sha256(Path(paths[vid])) for vid in videos},
                "interpretation": "Training-video reconstruction; not held-out generalization. "
                "Lexical scores do not measure medical correctness or video faithfulness. "
                "Zero/other-window memory are diagnostic controls with distribution limits."}
    if (adapter/"qformer_config.json").is_file():
        from .qformer import load_memory_model
        memory = load_memory_model(args.model_name,base_adapter=base_adapter,
                                   adapter_path=str(adapter)).to("cuda").eval()
    else:
        memory = MemoryModel(args.model_name, base_adapter=base_adapter,
                             adapter_path=str(adapter))
        memory.model.to("cuda").eval()
    manifest['qformer_config'] = getattr(memory, 'qformer_config', None)
    manifest['short_context'] = ('minute-prefix history; recent 10 output tokens for Local'
                                if getattr(memory, 'requires_short_history', False)
                                else 'independent 10-second windows')
    (output/"manifest.json").write_text(json.dumps(manifest, indent=2))
    rows = []
    with (output/"predictions.jsonl").open("x") as stream:
        for vid, blocks in videos.items():
            reader = VideoReader(paths[vid], args.frame_size)
            try:
                records = collect_memories(memory, blocks, reader, args.kinds)
            finally:
                reader.close()
            if args.save_memories:
                torch.save([{**r, "memory": r["memory"].cpu()} for r in records],
                           output/f"memory_{vid}.pt")
            for i, record in enumerate(records):
                for condition in dict.fromkeys(args.conditions):
                    tokens, metadata = controlled_memory(records, i, condition)
                    row = {"video_id": vid, "kind": record["kind"],
                           "window": record["window"], "condition": condition,
                           "reference": record["reference"], **metadata}
                    if tokens is not None:
                        row.update(memory_tokens=tokens.shape[1],
                                   **greedy_reconstruct(memory, tokens, record["kind"],
                                                        args.max_new_tokens))
                        row.update(text_metrics(row["prediction"], row["reference"]))
                    rows.append(row)
                    stream.write(json.dumps(row, ensure_ascii=False)+"\n")
                    stream.flush()
                    print(f"[reconstruct] {vid} {record['kind']} {record['window']} "
                          f"{condition}: {row.get('prediction', row.get('skipped'))}", flush=True)
    report = {"status": "completed", "records": len(rows), "scores": summarize(rows),
              "interpretation": manifest["interpretation"]}
    (output/"report.json").write_text(json.dumps(report, indent=2))
    with (output/"review.txt").open("x") as stream:
        for row in rows:
            stream.write(f"{row['video_id']} {row['kind']} {row['window']} {row['condition']}\n"
                         f"REF: {row['reference']}\nGEN: {row.get('prediction', row.get('skipped'))}\n"
                         f"STOP: {row.get('stop_reason')}\n\n")
    print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    main()
