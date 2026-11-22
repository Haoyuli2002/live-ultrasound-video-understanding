"""Freeze matched continuation arms; never submit a job from this preparer."""
import argparse
import hashlib
import json
import shutil
from pathlib import Path

from .data import digest, load_examples, summary
from .optimization import task_batches

ROOT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source-run', type=Path, required=True)
    parser.add_argument('--run-dir', type=Path, required=True)
    parser.add_argument('--epochs', type=int, default=24)
    args = parser.parse_args()
    source, run = args.source_run.resolve(), args.run_dir.resolve()
    if run.exists():
        raise ValueError('Use a new comparison directory')
    original_hashes = json.loads((source / 'source_hashes.json').read_text())
    for name, sha in original_hashes.items():
        if digest(source / 'code' / name) != sha:
            raise ValueError(f'Original frozen code changed: {name}')
    run.mkdir(parents=True)
    (run / 'logs').mkdir()
    shutil.copytree(source / 'code', run / 'code')
    for path in (ROOT / 'endtoend').iterdir():
        if path.suffix in {'.py', '.md', '.sbatch'}:
            shutil.copy2(path, run / 'code/endtoend' / path.name)
    # Record documentation corrections alongside the code used by this run.
    for name in ('README.md', 'lavis/models/blip2_models/streamready.py',
                 'lavis/models/blip2_models/Qformer.py',
                 'run_scripts/train.sh', 'lavis/projects/streamready/train.yaml'):
        shutil.copy2(ROOT / 'StreamReady' / name, run / 'code/StreamReady' / name)
    cfg = json.loads((source / 'experiment.json').read_text())
    cfg['model'] = dict(cfg['model'], gradient_checkpointing=False)
    cfg.update(init_checkpoint=str(source / 'checkpoint_latest.pth'), restore_optimizer=True,
               parallel_endpoints=True, fast_linear_attention=True,
               seed=20261121, epochs=args.epochs, sampler='question_groups', batch_candidates=[8],
               checkpoint_every=4, eval_epochs=[4, 12, args.epochs],
               supervision='gold ANSWER CE + gold readiness BCE; only gradient routing differs between arms')
    examples = load_examples(source / 'samples.jsonl', source / 'tasks.jsonl', source / 'video_map.json')
    planned = [[ [e['sample_id'] for e in batch] for batch in task_batches(examples, 8, cfg['seed'] + epoch)]
               for epoch in range(1, args.epochs + 1)]
    for arm, strategy in [('joint_control', 'sum'), ('readiness_isolated', 'readiness_isolated')]:
        target = run / arm
        target.mkdir(); (target / 'logs').mkdir()
        for name in ('tasks.jsonl', 'samples.jsonl', 'video_map.json', 'data_audit.json', 'initialization_resources.json'):
            shutil.copy2(source / name, target / name)
        (target / 'experiment.json').write_text(json.dumps(dict(cfg, gradient_strategy=strategy), indent=2) + '\n')
    (run / 'planned_batch_order.json').write_text(json.dumps(planned, indent=2) + '\n')
    h = hashlib.sha256()
    with (source / 'checkpoint_latest.pth').open('rb') as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b''):
            h.update(block)
    plan = dict(source_run=str(source), initial_checkpoint_sha256=h.hexdigest(),
                data=summary(examples), holdout=None, epochs_per_arm=args.epochs,
                planned_updates_per_arm=sum(map(len, planned)), batch_size=8,
                optimizer_state_restored=True, extra_warmup_updates=0,
                only_algorithmic_difference='readiness gradient routing: sum vs isolated to RDY/head',
                training_objective='unchanged unweighted gold BCE + answer CE',
                no_new_contrastive_loss=True, no_threshold_change=True,
                evaluation='same three training videos; forced and gated answers reported separately')
    (run / 'comparison_plan.json').write_text(json.dumps(plan, indent=2) + '\n')
    (run / 'source_hashes.json').write_text(json.dumps({str(p.relative_to(run / 'code')): digest(p)
                                                     for p in sorted((run / 'code').rglob('*')) if p.is_file()}, indent=2) + '\n')
    original = (run / 'code/endtoend/run_overfit.sbatch').read_text()
    script = original.replace('--job-name=sr-e2e-3v', '--job-name=sr-rdy-ab').replace('--time=02:00:00', '--time=01:00:00')
    script = script.replace('"$PYTHON" -B -m endtoend.train --run-dir "$RUN_DIR"',
                            'for arm in joint_control readiness_isolated; do\n'
                            '    "$PYTHON" -B -m endtoend.train --run-dir "$RUN_DIR/$arm"\n'
                            'done\n'
                            '"$PYTHON" -B -m endtoend.compare_runs --run-dir "$RUN_DIR"')
    (run / 'run_comparison.sbatch').write_text(script)
    print(json.dumps({'run_dir': str(run), **plan}, indent=2))


if __name__ == '__main__':
    main()
