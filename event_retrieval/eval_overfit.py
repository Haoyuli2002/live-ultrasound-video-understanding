"""Greedy in-sample replay at every audited decision, including forced-answer controls."""
import argparse
from contextlib import closing
from itertools import groupby
import json
from pathlib import Path

import torch

from endtoend.data import digest
from .train import (AnswerDecoder, EventIndex, EventQFormers, Qwen35Backbone, QwenRelevance,
                    index_source, load_overfit, load_split, prepare_decision, restore)


def exact(text, reference):
    return " ".join(text.casefold().split()) == " ".join(reference.casefold().split())


def summarize(rows):
    first_prediction, first_truth = {}, {}
    wait = correct_wait = answer = correct_answer = forced_exact = timed_exact = 0
    for r in rows:
        key = r["task_id"]
        first_prediction.setdefault(key, None)
        first_truth.setdefault(key, None)
        if r["action"] == "ANSWER" and first_prediction[key] is None:
            first_prediction[key] = r["current_time"]
        if not r["ready"]:
            wait += 1
            correct_wait += r["action"] == "WAIT"
            continue
        first_truth[key] = r["current_time"]
        answer += 1
        correct_answer += r["action"] == "ANSWER"
        forced_exact += exact(r["forced_answer"], r["answer"])
        timed_exact += (first_prediction[key] == r["current_time"] and r["action"] == "ANSWER"
                        and exact(r["prediction"], r["answer"]))
    return dict(questions=len(first_truth), endpoints=len(rows), wait_endpoints=wait,
                correct_wait=correct_wait, answer_endpoints=answer, correct_answer=correct_answer,
                first_answer_on_time=sum(t is not None and first_prediction[k] == t for k,t in first_truth.items()),
                on_time_and_exact_answer=timed_exact, forced_exact_answer=forced_exact,
                untriggered=sum(t is None for t in first_truth.values()),
                correctly_untriggered=sum(t is None and first_prediction[k] is None for k,t in first_truth.items()))


@torch.inference_mode()
def load_evaluator(run, device, *, holdout=False):
    cfg = json.loads((run / "config.json").read_text())
    if (cfg.get("mode") == "overfit") == holdout:
        raise ValueError("Requested evaluation split differs from the training configuration")
    data = Path(cfg["data"])
    if any(digest(data / name) != value for name, value in cfg["manifests"].items()):
        raise ValueError("QA release changed after training")
    examples = load_split(data / "holdout", "holdout") if holdout else load_overfit(data)
    path = Path(cfg["model"])
    bb = Qwen35Backbone(path).to(device).eval()
    towers = EventQFormers.from_instructblip(Path(cfg["qformer"]), vision_width=bb.vision_width,
                                             llm_width=bb.llm_width).to(device).eval()
    decoder = AnswerDecoder(bb)
    bb.model.model.language_model.float()
    bb.model.get_output_embeddings().float()
    epoch = restore(run / "latest.pt", towers, bb, config=cfg)
    return cfg, examples, path, bb, towers, decoder, epoch


@torch.inference_mode()
def evaluate(run, device, budget, *, holdout=False):
    split = "holdout" if holdout else "overfit"
    cfg, examples, path, bb, towers, decoder, epoch = load_evaluator(run, device, holdout=holdout)
    scorer = QwenRelevance(bb)
    results = []
    tmp = run / f"{split}_predictions.tmp"
    with tmp.open("w") as stream:
        for vid, rows in groupby(examples, key=lambda r: r["video_id"]):
            group = list(rows)
            video = Path(group[0]["video_path"]).resolve(strict=True)
            db = run / "index" / (vid + ".sqlite")
            if not db.is_file():
                raise FileNotFoundError(db)
            with closing(EventIndex(db, vid, index_source(video, path), read_only=True)) as index:
                for row in group:
                    recent, events, valid, ctx, gt = prepare_decision(
                        index, row, towers, scorer, retrieval_batch_size=cfg.get("retrieval_batch_size", 4))
                    with torch.autocast(device.type, torch.bfloat16, enabled=cfg.get("bf16", False)):
                        prefix, mask = towers(recent[None].to(device), events[None].to(device),
                                              event_mask=valid[None].to(device))
                        pred = decoder.generate(prefix[0], mask[0], ctx, budget)
                        forced = pred["answer"]
                        if row["ready"] and forced is None:
                            forced = decoder.generate(prefix[0], mask[0], ctx, budget, force_answer=True)["answer"]
                    result = dict(**row, ground_truth=gt, action=pred["action"],
                                  prediction=pred["answer"] or pred["reason"], finish_reason=pred["finish_reason"],
                                  forced_answer=forced if row["ready"] else None, retrieved_events=list(ctx.events))
                    stream.write(json.dumps(result) + "\n")
                    results.append(result)
    tmp.replace(run / f"{split}_predictions.jsonl")
    metrics = dict(epoch=epoch, split=split, **summarize(sorted(results, key=lambda r: (r["task_id"], r["current_time"]))))
    (run / f"{split}_metrics.json").write_text(json.dumps(metrics, indent=2) + "\n")
    print(json.dumps(metrics))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--max-new-tokens", type=int, default=96)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--holdout", action="store_true")
    args = parser.parse_args()
    evaluate(args.run.resolve(strict=True), torch.device(args.device), args.max_new_tokens, holdout=args.holdout)
