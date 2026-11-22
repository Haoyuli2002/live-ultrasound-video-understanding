"""Summarize nvidia-smi samples, optionally within timed batch probes."""
import argparse
import csv
import json
from datetime import datetime
from pathlib import Path


def percentile(values, fraction):
    ordered = sorted(values)
    position = (len(ordered) - 1) * fraction
    left = int(position)
    right = min(left + 1, len(ordered) - 1)
    return ordered[left] + (ordered[right] - ordered[left]) * (position - left)


def read_samples(path):
    rows = []
    with Path(path).open() as stream:
        for raw in csv.DictReader(stream):
            row = {k.strip(): v.strip() for k, v in raw.items() if k and v}
            try:
                rows.append(dict(
                    timestamp=datetime.strptime(row['timestamp'], '%Y/%m/%d %H:%M:%S.%f').timestamp(),
                    gpu_util=float(row['utilization.gpu [%]'].split()[0]),
                    memory_util=float(row['utilization.memory [%]'].split()[0]),
                    memory_gib=float(row['memory.used [MiB]'].split()[0]) / 1024,
                    power_w=float(row['power.draw [W]'].split()[0]),
                ))
            except (KeyError, ValueError) as exc:
                raise ValueError(f'Invalid GPU monitor row: {row}') from exc
    return rows


def summarize(rows):
    if not rows:
        return dict(samples=0)
    utilization = [r['gpu_util'] for r in rows]
    return dict(samples=len(rows), sampled_span_seconds=rows[-1]['timestamp'] - rows[0]['timestamp'],
                gpu_util_mean_pct=sum(utilization) / len(rows),
                gpu_util_median_pct=percentile(utilization, .5),
                gpu_util_p95_pct=percentile(utilization, .95),
                fraction_samples_at_least_80_pct=sum(x >= 80 for x in utilization) / len(rows),
                fraction_samples_at_most_10_pct=sum(x <= 10 for x in utilization) / len(rows),
                memory_peak_gib=max(r['memory_gib'] for r in rows),
                memory_util_mean_pct=sum(r['memory_util'] for r in rows) / len(rows),
                power_mean_w=sum(r['power_w'] for r in rows) / len(rows))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run-dir', required=True, type=Path)
    parser.add_argument('--output', type=Path)
    args = parser.parse_args()
    rows = read_samples(args.run_dir / 'logs/gpu.csv')
    result = dict(scope='whole job including loading, probes, evaluation and I/O as applicable',
                  measurement='nvidia-smi utilization sampling; not achieved FLOPS; fractions approximate time at fixed cadence',
                  whole_job=summarize(rows), timed_probes=[])
    profiles = args.run_dir / 'batch_profiles.json'
    if profiles.exists():
        for probe in json.loads(profiles.read_text()):
            if 'timed_start_unix' not in probe:
                continue
            selected = [r for r in rows if probe['timed_start_unix'] <= r['timestamp'] <= probe['timed_end_unix']]
            result['timed_probes'].append(dict(batch_size=probe['batch_size'],
                                               samples_per_second=probe['samples_per_second'],
                                               **summarize(selected)))
    output = args.output or args.run_dir / 'gpu_summary.json'
    output.write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
