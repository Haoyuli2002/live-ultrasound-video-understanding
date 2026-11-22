"""Measure safe batches, then run an update-matched overfit continuation."""
import argparse
import json
import subprocess
import sys
import traceback
from pathlib import Path
from .data import load_examples
from .optimization import task_batches


def write(path, value):
    tmp = path.with_suffix(path.suffix + '.partial')
    tmp.write_text(json.dumps(value, indent=2) + '\n')
    tmp.replace(path)


def choose_batch(worst, real):
    safe = {p['batch_size'] for p in worst if p.get('safe') and p.get('status') == 'passed'}
    candidates = [p for p in real if p['batch_size'] in safe and p.get('safe') and p.get('status') == 'passed']
    if not candidates:
        raise RuntimeError('No batch passed both worst-case and real-data memory margins')
    return max(candidates, key=lambda p: p['samples_per_second'])['batch_size']


def epoch_plan(examples, batch, seed, updates=300):
    steps, epochs, sizes = 0, 0, []
    while steps < updates:
        epochs += 1
        batches = task_batches(examples, batch, seed + epochs)
        steps += len(batches)
        sizes.append([len(b) for b in batches])
    if steps != updates:
        raise ValueError('Update budget is not an exact whole-epoch multiple')
    return epochs, sizes


def main(root):
    try:
        for phase in ('worst_case', 'real_batch'):
            write(root / 'workflow.json', {'status': 'running', 'phase': phase, 'optimizer_steps': 0})
            subprocess.run([sys.executable, '-B', '-m', 'endtoend.train', '--run-dir', str(root / phase)], check=True)
        worst = json.loads((root / 'worst_case/batch_profiles.json').read_text())
        real = json.loads((root / 'real_batch/batch_profiles.json').read_text())
        batch = choose_batch(worst, real)
        run = root / 'train'
        cfg = json.loads((run / 'experiment.json').read_text())
        examples = load_examples(run / 'samples.jsonl', run / 'tasks.jsonl', run / 'video_map.json')
        epochs, sizes = epoch_plan(examples, batch, cfg['seed'])
        cfg.update(batch_candidates=[batch], epochs=epochs, eval_epochs=[epochs // 3, 2 * epochs // 3, epochs],
                   checkpoint_every=epochs // 3, profile_warmups=0, profile_repeats=1)
        write(run / 'experiment.json', cfg)
        decision = {'batch_size': batch, 'epochs': epochs, 'optimizer_steps': 300, 'gradient_accumulation': 1,
                    'reason': 'fastest real-data throughput among batches passing both safety probes',
                    'worst_case': worst, 'real_batch': real, 'planned_batch_sizes': sizes}
        write(root / 'batch_selection.json', decision)
        write(root / 'workflow.json', {'status': 'running', 'phase': 'train', 'batch_size': batch,
                                       'epochs': epochs, 'planned_optimizer_steps': 300})
        subprocess.run([sys.executable, '-B', '-m', 'endtoend.train', '--run-dir', str(run)], check=True)
        write(root / 'workflow.json', {'status': 'completed', 'phase': 'completed', 'batch_size': batch,
                                       'epochs': epochs, 'optimizer_steps': 300})
    except Exception:
        previous = json.loads((root / 'workflow.json').read_text()) if (root / 'workflow.json').exists() else {}
        write(root / 'workflow.json', dict(previous, status='failed', traceback=traceback.format_exc()))
        raise


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run-dir', type=Path, required=True)
    main(parser.parse_args().run_dir.resolve())
