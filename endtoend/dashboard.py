"""Read-only local training dashboard; forward its loopback port over SSH."""
import argparse
import csv
import io
import json
import math
import subprocess
import threading
import time
from datetime import datetime
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlparse

ROOT = Path(__file__).resolve().parents[1]


def read_json(path, default=None):
    try:
        return json.loads(path.read_text())
    except (FileNotFoundError, json.JSONDecodeError):
        return {} if default is None else default


def read_jsonl(path):
    try:
        lines = path.read_text().splitlines()
    except FileNotFoundError:
        return []
    rows = []
    for line in lines:
        try:
            rows.append(json.loads(line))
        except json.JSONDecodeError:
            continue  # A concurrent writer may not have finished its last row.
    return rows


def gpu_samples(path):
    try:
        data = path.read_text()
    except FileNotFoundError:
        return []
    result = []
    for raw in csv.DictReader(io.StringIO(data)):
        row = {k.strip(): v.strip() for k, v in raw.items() if k and isinstance(v, str)}
        try:
            def num(key):
                value = float(row[key].split()[0])
                if not math.isfinite(value):
                    raise ValueError('Nonfinite GPU sample')
                return value
            result.append(dict(time=datetime.strptime(row['timestamp'], '%Y/%m/%d %H:%M:%S.%f').timestamp(),
                               gpu=num('utilization.gpu [%]'), memory_activity=num('utilization.memory [%]'),
                               used=num('memory.used [MiB]') / 1024, total=num('memory.total [MiB]') / 1024,
                               power=num('power.draw [W]'), name=row['name']))
        except (KeyError, ValueError):
            continue
    return result


class Monitor:
    def __init__(self, runs):
        self.runs = runs
        self.lock = threading.Lock()
        self.scheduler_cache = {}
        self.snapshots = {}

    def scheduler(self, job_id):
        if not job_id or not str(job_id).isdigit():
            return {'state': 'NOT_SUBMITTED'}
        previous = self.scheduler_cache.get(job_id)
        if previous and time.time() - previous[0] < 30:
            return previous[1]
        result = {'state': 'UNKNOWN'}
        try:
            q = subprocess.run(['squeue', '--start', '-h', '-j', str(job_id), '-o', '%i|%T|%S|%M|%R'],
                               capture_output=True, text=True, timeout=5)
            if q.returncode == 0 and q.stdout.strip():
                parts = q.stdout.strip().splitlines()[0].split('|', 4)
                result = dict(zip(['job_id', 'state', 'estimated_start', 'elapsed', 'reason_or_node'], parts))
            else:
                q = subprocess.run(['sacct', '-n', '-P', '-j', str(job_id),
                                    '--format=JobID,State,Elapsed,ExitCode,NodeList'],
                                   capture_output=True, text=True, timeout=5)
                for line in q.stdout.splitlines():
                    parts = line.split('|')
                    if parts[0] == str(job_id):
                        result = dict(zip(['job_id', 'state', 'elapsed', 'exit_code', 'reason_or_node'], parts))
                        break
        except (OSError, subprocess.TimeoutExpired) as exc:
            result = {'state': 'UNKNOWN', 'error': str(exc)}
        result['checked_at'] = time.time()
        self.scheduler_cache[job_id] = (time.time(), result)
        return result

    def snapshot(self, key):
        with self.lock:
            cached = self.snapshots.get(key)
            if cached and time.time() - cached[0] < 4:
                return cached[1]
            root = self.runs[key]
            train = root / 'train' if (root / 'train').is_dir() else root
            cfg = read_json(train / 'experiment.json')
            plan = read_json(root / 'plan.json')
            submission = read_json(root / 'submission.json')
            workflow = read_json(root / 'workflow.json')
            status = read_json(train / 'status.json')
            records = [r for r in read_jsonl(train / 'training.jsonl') if r.get('phase') == 'train']
            evaluations = read_jsonl(train / 'eval_metrics.jsonl')
            gpu = gpu_samples(root / 'logs/gpu.csv')
            gpu_info = None
            if gpu:
                latest = gpu[-1]
                recent = [r for r in gpu if r['time'] >= latest['time'] - 60]
                gpu_info = dict(latest=latest, age_seconds=max(0, time.time() - latest['time']),
                                peak_gib=max(r['used'] for r in gpu),
                                last_60s_mean_util=sum(r['gpu'] for r in recent) / len(recent),
                                whole_run_mean_util=sum(r['gpu'] for r in gpu) / len(gpu), samples=len(gpu))
            # Keep extrema in cards from all samples, downsample only chart paths.
            stride = max(1, len(gpu) // 900)
            history = gpu[::stride]
            if gpu and (not history or history[-1] != gpu[-1]):
                history.append(gpu[-1])
            phase = workflow.get('phase', 'train')
            if phase == 'train' and status.get('status') != 'completed':
                if not records:
                    phase = 'initializing_or_initial_eval'
                elif records[-1]['epoch'] in cfg.get('eval_epochs', []) and records[-1]['epoch'] > max(
                        [e['epoch'] for e in evaluations], default=-1):
                    phase = 'checkpoint_or_evaluation'
            probes = {}
            for name in ('worst_case', 'real_batch'):
                rows = read_json(root / name / 'batch_profiles.json', [])
                if rows:
                    probes[name] = rows
            scheduler = self.scheduler(submission.get('job_id'))
            if plan.get('comparison_suite') and not status and scheduler.get('state') == 'RUNNING':
                scheduler = dict(scheduler, allocation_state='RUNNING', state='WAITING_IN_SUITE',
                                 reason_or_node='等待前序对照实验结束')
            value = dict(run=key, run_dir=str(root), updated_at=time.time(), job_id=submission.get('job_id'),
                         scheduler=scheduler, workflow=workflow, phase=phase,
                         status=status, config={k: cfg.get(k) for k in ['epochs', 'learning_rate', 'batch_candidates',
                             'gradient_strategy', 'parallel_endpoints', 'fast_linear_attention', 'readiness_objective', 'timing_lambda_reg', 'readiness_bce_positive_weight', 'readiness_bce_weight']},
                         checkpointing=cfg.get('model', {}).get('gradient_checkpointing'),
                         planned_steps=plan.get('optimizer_steps', 300),
                         step=records[-1]['step'] if records else 0,
                         data=plan.get('summary', read_json(root / 'original_data_audit.json').get('summary', {})),
                         training=[{k: r.get(k) for k in ['step', 'epoch', 'loss', 'answer_loss', 'readiness_loss',
                             'batch_size', 'seconds', 'peak_allocated_gib', 'peak_reserved_gib', 'readiness_bce', 'readiness_weighted_bce', 'readiness_contrastive', 'readiness_temporal']} for r in records],
                         evaluations=evaluations, gpu=gpu_info, gpu_history=history, probes=probes,
                         batch_selection=read_json(root / 'batch_selection.json'))
            self.snapshots[key] = (time.time(), value)
            return value


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--runs', type=Path, required=True, help='JSON object: display name -> frozen run directory')
    parser.add_argument('--port', type=int, default=8766)
    args = parser.parse_args()
    monitor = Monitor({k: Path(v).resolve() for k, v in json.loads(args.runs.read_text()).items()})
    assets = {'/': ('training.html', 'text/html'), '/training.css': ('training.css', 'text/css'),
              '/training.js': ('training.js', 'application/javascript')}

    class Handler(BaseHTTPRequestHandler):
        def do_GET(self):
            url = urlparse(self.path)
            try:
                if url.path == '/api/runs':
                    payload = json.dumps(list(monitor.runs)).encode()
                    kind = 'application/json'
                elif url.path == '/api/state':
                    key = parse_qs(url.query).get('run', [next(iter(monitor.runs))])[0]
                    if key not in monitor.runs:
                        self.send_error(404)
                        return
                    payload = json.dumps(monitor.snapshot(key), allow_nan=False).encode()
                    kind = 'application/json'
                elif url.path in assets:
                    filename, kind = assets[url.path]
                    payload = (ROOT / 'viz' / filename).read_bytes()
                else:
                    self.send_error(404)
                    return
                self.send_response(200)
                self.send_header('Content-Type', kind + '; charset=utf-8')
                self.send_header('Content-Length', str(len(payload)))
                self.send_header('Cache-Control', 'no-store')
                self.send_header('X-Content-Type-Options', 'nosniff')
                self.end_headers()
                self.wfile.write(payload)
            except (BrokenPipeError, ConnectionResetError):
                pass
            except Exception as exc:
                self.log_error('%s', exc)
                self.send_error(500, 'Unable to read current run state')

        def log_message(self, fmt, *values):
            if values and str(values[1:2]) == "('200',)":
                return
            super().log_message(fmt, *values)

    server = ThreadingHTTPServer(('127.0.0.1', args.port), Handler)
    print('Dashboard: http://127.0.0.1:%d (SSH forward this port)' % args.port, flush=True)
    server.serve_forever()


if __name__ == '__main__':
    main()
