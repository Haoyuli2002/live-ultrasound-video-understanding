"""Freeze a new- or legacy-QA overfit run; does not submit training."""
import argparse
import collections
import datetime
import json
import shutil
from pathlib import Path
from .data import digest, load_examples, read_rows, summary

ROOT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', type=Path, required=True)
    parser.add_argument('--samples-name', help='Defaults to train_samples.jsonl, or legacy train.jsonl')
    parser.add_argument('--tasks-name', default='train_tasks.jsonl')
    parser.add_argument('--video-map', type=Path)
    parser.add_argument('--video-id', action='append', required=True)
    parser.add_argument('--run-dir', type=Path, required=True)
    parser.add_argument('--epochs', type=int, default=12)
    args = parser.parse_args()
    run, source = args.run_dir.resolve(), args.source.resolve()
    if run.exists():
        raise ValueError('Use a fresh run directory')
    samples_name = args.samples_name or ('train_samples.jsonl' if (source / 'train_samples.jsonl').exists() else 'train.jsonl')
    paths = dict(samples=source / samples_name, tasks=source / args.tasks_name,
                 video_map=args.video_map or source / 'video_map.json')
    before = {k: digest(v) for k, v in paths.items()}
    all_examples = load_examples(paths['samples'], paths['tasks'], paths['video_map'])
    ids = set(args.video_id)
    examples = [e for e in all_examples if e['video_id'] in ids]
    if {e['video_id'] for e in examples} != ids or any(e['split'] != 'train' for e in examples):
        raise ValueError('Every requested video must be present in the train split')
    if not all(Path(e['video_path']).is_file() for e in examples):
        raise ValueError('Missing original media')
    tasks = [t for t in read_rows(paths['tasks']) if t['video_id'] in ids]
    samples = [s for s in read_rows(paths['samples']) if s['video_id'] in ids]
    video_map = {e['video_id']: e['video_path'] for e in examples}
    if {k: digest(v) for k, v in paths.items()} != before:
        raise RuntimeError('Source changed during snapshot; retry')
    run.mkdir(parents=True)
    (run / 'logs').mkdir()
    for name, rows in [('tasks', tasks), ('samples', samples)]:
        (run / f'{name}.jsonl').write_text(''.join(json.dumps(r) + '\n' for r in rows))
    (run / 'video_map.json').write_text(json.dumps(video_map, indent=2) + '\n')
    frozen = load_examples(run / 'samples.jsonl', run / 'tasks.jsonl', run / 'video_map.json')
    model = dict(arch='streamready', model_type='qwen3_5', backbone_model='/dss/mcmlscratch/04/ge75vid2/haoyu/hf_cache/hub/models--Qwen--Qwen3.5-2B/snapshots/15852e8c16360a2fea060d615a32b45270f8a8fc',
                 backbone_precision='bf16', attn_implementation='sdpa', image_size=256,
                 num_query_token=32, qformer_text_input=True, qformer_encoder_width=1408,
                 frame_capacity=24, num_centroids=96, num_prototypes=12,
                 topk_prototypes=8, topk_centroids=24, ema_decay=.985,
                 long_term_fusion_mode='per_layer', use_contextual_memory=False,
                 use_readiness=True, readiness_threshold=.5, readiness_loss_weight=1.,
                 train_stage='joint', gradient_checkpointing=False,
                 max_txt_len=256, max_output_txt_len=160, load_pretrained=True, load_finetuned=False,
                 pretrained='/dss/mcmlscratch/04/ge75vid2/torch_cache/hub/checkpoints/instruct_blip_vicuna7b_trimmed.pth')
    cfg = dict(model=model, seed=20261120, epochs=args.epochs, learning_rate=1e-4,
               gradient_strategy='readiness_isolated', parallel_endpoints=True, fast_linear_attention=True,
               readiness_objective='contrastive_bce_gold', timing_lambda_reg=.1,
               readiness_bce_positive_weight=2., readiness_bce_weight=1., sampler='question_groups',
               weight_decay=.01, image_size=256, vision_chunk=8, batch_candidates=[8, 16, 32, 48, 64],
               max_new_tokens=96, eval_epochs=[1, 4, 8, 12],
               supervision='ANSWER CE + contrastive timing + 0.1TV + weighted BCE (ANSWER2/WAIT1); timing isolated to RDY/head',
               holdout=None, evaluation='same 3 training videos; overfit only',
               preprocessing='1 fps from zero plus actual decision endpoint; PTS <= observation time; aspect-preserving 256px pad',
               backbone='frozen vision and language weights; cached per-frame features only within this run',
               timestamp=datetime.datetime.now(datetime.timezone.utc).isoformat())
    (run / 'experiment.json').write_text(json.dumps(cfg, indent=2) + '\n')
    for folder in ('endtoend', 'StreamReady'):
        # Copy source/config only, never data, bytecode, git internals or model weights.
        for src in (ROOT / folder).rglob('*'):
            if src.is_file() and src.suffix in {'.py', '.yaml', '.yml', '.sh', '.sbatch', '.md', '.txt'} and not any(
                    part in {'.git', '__pycache__', 'output', 'data'} for part in src.relative_to(ROOT).parts):
                dst = run / 'code' / src.relative_to(ROOT)
                dst.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(src, dst)
    (run / 'source_hashes.json').write_text(json.dumps({str(p.relative_to(run / 'code')): digest(p)
                                                     for p in sorted((run / 'code').rglob('*')) if p.is_file()}, indent=2) + '\n')
    audit = dict(source=str(source), source_hashes=before, source_summary=summary(all_examples),
                 selected_summary=summary(frozen), video_ids=sorted(ids),
                 selected_qa_types=dict(collections.Counter(t['qa_type'] for t in tasks)),
                 no_train_eval_mixing=True, holdout_used=False,
                 media_sha256={v: digest(p) for v, p in video_map.items()},
                 annotation_quality='candidate annotations; no expert-review or model-resolution visibility claim',
                 snapshot_hashes={p.name: digest(p) for p in (run / 'tasks.jsonl', run / 'samples.jsonl', run / 'video_map.json')})
    (run / 'data_audit.json').write_text(json.dumps(audit, indent=2) + '\n')
    print(json.dumps({'run_dir': str(run), **audit}, indent=2))


if __name__ == '__main__':
    main()
