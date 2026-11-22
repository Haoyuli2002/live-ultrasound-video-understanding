"""Verify matched sample exposure and summarize overfit continuation results."""
import argparse
import json
from pathlib import Path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--run-dir', type=Path, required=True)
    args = parser.parse_args()
    run = args.run_dir
    arms = ['joint_control', 'readiness_isolated']
    records = {}
    for arm in arms:
        root = run / arm
        status = json.loads((root / 'status.json').read_text())
        if status['status'] != 'completed':
            raise RuntimeError(f'{arm} did not finish')
        evaluation = [json.loads(x) for x in (root / 'eval_metrics.jsonl').read_text().splitlines()]
        records[arm] = dict(status=status, initial=evaluation[0], final=evaluation[-1])
    first, second = [run / name for name in arms]
    if (first / 'batch_order.jsonl').read_bytes() != (second / 'batch_order.jsonl').read_bytes():
        raise RuntimeError('Arms did not receive identical batches')
    for name in ('tasks.jsonl', 'samples.jsonl', 'video_map.json'):
        if (first / name).read_bytes() != (second / name).read_bytes():
            raise RuntimeError(f'Data mismatch: {name}')
    report = dict(scope='same training examples; no holdout or generalization evidence',
                  matched_batches_and_data=True, arms=records)
    (run / 'comparison.json').write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps(report, indent=2), flush=True)


if __name__ == '__main__':
    main()
