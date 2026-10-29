"""Freeze, shard, resume and strictly assemble visual-only Stage 2 teacher data."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import re
import subprocess
import time

from .annotate import (PROMPT_SHA256, annotation_config, resume_start,
                       save_events_atomic, source_digest)
from .build_data import build_files
from .data import VideoReader
from .sampling import POLICY

MODEL = "Qwen/Qwen3.5-27B"


def write_json(path: Path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + f".{os.getpid()}.tmp")
    temporary.write_text(json.dumps(value, indent=2) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def read_map(path: Path, video_root: Path, *, require_files: bool) -> dict[str, str]:
    value = json.loads(path.read_text())
    if not isinstance(value, dict) or not value:
        raise ValueError(f"Expected a nonempty video map: {path}")
    result = {}
    for vid, raw in sorted(value.items()):
        if not isinstance(vid, str) or not re.fullmatch(r"[A-Za-z0-9_-]+", vid):
            raise ValueError(f"Unsafe video ID: {vid!r}")
        source = Path(raw).expanduser()
        source = (source if source.is_absolute() else video_root/source).resolve()
        if require_files and (not source.is_file() or source.stem != vid):
            raise ValueError(f"Missing or mismatched video: {vid}: {source}")
        result[vid] = str(source)
    if len(set(result.values())) != len(result):
        raise ValueError("Multiple video IDs refer to the same source")
    return result


def expected_counts(duration: int) -> dict:
    minutes, tail = divmod(duration, 60)
    return {"local": duration//10, "block": minutes, "global": minutes,
            "events": duration//10+2*minutes, "rows": minutes+int(tail >= 10)}


def records(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def checked_end(path: Path, entry: dict, model: str) -> int:
    if not path.exists():
        return 0
    return resume_start(records(path), video_id=entry['video_id'],
                        duration_sec=entry['duration_sec'], model=model,
                        config=entry['annotation_config'])


def load_manifest(run_dir: Path) -> dict:
    value = json.loads((run_dir/'manifest.json').read_text())
    if value.get('schema') != 'stage2_full_run_v1':
        raise ValueError('Unsupported full-run manifest')
    if value['prompt_sha256'] != PROMPT_SHA256 or value['sampling_policy'] != POLICY:
        raise ValueError('Frozen prompt/sampling differs; use the original code snapshot')
    if value['model'] != MODEL:
        raise ValueError('Frozen teacher model differs')
    return value


def prepare(video_map: Path, eval_map: Path, run_dir: Path, *, video_root: Path,
            shards: int = 8, seed_teacher_dir: Path | None = None,
            seed_dependency: str | None = None) -> dict:
    if shards < 1:
        raise ValueError('shards must be positive')
    if seed_dependency and (not seed_dependency.isdigit() or not seed_teacher_dir):
        raise ValueError('Deferred seed needs a numeric Slurm job ID and teacher directory')
    if (run_dir/'manifest.json').exists():
        raise FileExistsError('Manifest already exists; submit the frozen run with --resume')
    selected = read_map(video_map, video_root, require_files=True)
    excluded = read_map(eval_map, video_root, require_files=False)
    if selected.keys() & excluded.keys() or set(selected.values()) & set(excluded.values()):
        raise ValueError('Train/evaluation overlap in video IDs or resolved paths')
    if any((run_dir/'teacher'/f'{vid}.jsonl').exists() for vid in selected):
        raise FileExistsError('Unplanned teacher files already exist; use a fresh run directory')
    entries, seed_data = [], {}
    for index, (vid, source) in enumerate(selected.items(), 1):
        reader = VideoReader(source, 224)
        try:
            duration = int(reader.count/reader.fps)
        finally:
            reader.close()
        if duration < 10:
            raise ValueError(f'{vid}: fewer than ten complete seconds')
        entry = {'video_id': vid, 'path': source, 'duration_sec': duration,
                 'annotation_config': annotation_config(Path(source)),
                 'expected': expected_counts(duration), 'seed_resume_sec': 0}
        if seed_teacher_dir and (seed_teacher_dir/f'{vid}.jsonl').exists():
            seed = seed_teacher_dir/f'{vid}.jsonl'
            entry['seed_source'] = str(seed.resolve())
            if seed_dependency:
                # Read only after the source job ends; it may still be writing.
                entry['seed_deferred'] = True
                entries.append(entry)
                continue
            # Validate whole minute batches, source hash, model and full recipe.
            data = seed.read_bytes()
            copied = [json.loads(line) for line in data.decode().splitlines() if line.strip()]
            entry['seed_resume_sec'] = resume_start(copied, video_id=vid,
                duration_sec=duration, model=MODEL, config=entry['annotation_config'])
            seed_data[vid] = data
        entries.append(entry)
        print(f'[plan] {index}/{len(selected)} {vid}: {duration}s, '
              f"{entry['expected']['events']} events, seed={entry['seed_resume_sec']}s", flush=True)
    pending = [v for v in entries if v['seed_resume_sec'] < v['duration_sec']]
    buckets = [[] for _ in range(min(shards, len(pending)))]
    loads = [0.0 for _ in buckets]
    def cost(entry):
        minutes = entry['duration_sec']//60
        return entry['expected']['events'] + minutes*(minutes+1)/2
    for entry in sorted(pending, key=lambda v: (-cost(v), v['video_id'])):
        index = min(range(len(buckets)), key=lambda i: (loads[i], i))
        buckets[index].append(entry['video_id'])
        loads[index] += cost(entry)
    plan = {'schema': 'stage2_full_run_v1', 'model': MODEL,
            'prompt_sha256': PROMPT_SHA256, 'sampling_policy': POLICY,
            'source_map': str(video_map.resolve()), 'eval_map': str(eval_map.resolve()),
            'excluded_eval_ids': sorted(excluded), 'videos': entries,
            'shards': buckets, 'shard_work_units': loads,
            'selected_videos': len(entries), 'pending_videos': len(pending),
            'seeded_complete_videos': len(entries)-len(pending),
            'seed_dependency': seed_dependency,
            'video_hours': sum(v['duration_sec'] for v in entries)/3600,
            'expected_events': sum(v['expected']['events'] for v in entries),
            'expected_rows': sum(v['expected']['rows'] for v in entries)}
    for vid, data in seed_data.items():
        destination = run_dir/'teacher'/f'{vid}.jsonl'
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_bytes(data)
    write_json(run_dir/'video_paths.json', selected)
    write_json(run_dir/'excluded_eval_paths.json', excluded)
    for index, ids in enumerate(buckets):
        write_json(run_dir/'shards'/f'{index:03d}.videos.json', {v:selected[v] for v in ids})
    write_json(run_dir/'manifest.json', plan)
    return plan


def worker(run_dir: Path, index: int, repo: Path) -> int:
    plan = load_manifest(run_dir)
    if not 0 <= index < len(plan['shards']):
        raise ValueError('Invalid shard index')
    entries = {v['video_id']:v for v in plan['videos']}
    ids = plan['shards'][index]
    mapping = {vid:entries[vid]['path'] for vid in ids}
    shard_map = run_dir/'shards'/f'{index:03d}.videos.json'
    if json.loads(shard_map.read_text()) != mapping:
        raise ValueError('Shard map differs from frozen manifest')
    complete = True
    paths = []
    for vid in ids:
        entry = entries[vid]
        if annotation_config(Path(entry['path'])) != entry['annotation_config']:
            raise ValueError(f'{vid}: source or recipe changed after planning')
        path = run_dir/'teacher'/f'{vid}.jsonl'
        if entry.get('seed_deferred'):
            try:
                seed = Path(entry['seed_source'])
                rows = records(seed)
                end = resume_start(rows,video_id=vid,duration_sec=entry['duration_sec'],
                                   model=plan['model'],config=entry['annotation_config'])
                if end > checked_end(path,entry,plan['model']):
                    path.parent.mkdir(parents=True,exist_ok=True)
                    save_events_atomic(path,rows)
                    print(f'[seed] {vid}: imported compatible labels through {end}s',flush=True)
            except (OSError,ValueError,TypeError,KeyError) as error:
                print(f'[seed-warning] {vid}: skipped unavailable/incompatible seed: {error}',flush=True)
        complete &= checked_end(path, entry, plan['model']) == entry['duration_sec']
        paths.append(path)
    output = run_dir/'shards'/f'{index:03d}.blocks.jsonl'
    if complete:
        count = build_files(paths, output)
        print(f'[shard {index}] all videos complete; rebuilt {count} rows without loading 27B')
        return 0
    env = {**os.environ, 'REPO':str(repo), 'VIDEO_MAP':str(shard_map), 'VIDEO_IDS':'',
           'TEACHER_DIR':str(run_dir/'teacher'), 'AUDIT_DIR':str(run_dir/'audit'),
           'BLOCKS':str(output), 'MODEL':MODEL, 'VLLM_PORT':'auto',
           'VIDEO_FPS':'2.0', 'VIDEO_MAX_FRAMES':'120', 'CLIP_FPS':'2.0',
           'CLIP_MAX_SIDE':'1024', 'MAX_MODEL_LEN':'32768', 'MAX_TOKENS':'1024',
           'MAX_ATTEMPTS':'3', 'FRAME_SIZE':'224', 'CONTINUE_ON_ERROR':'1'}
    env.pop('VLLM_LOG', None)
    return subprocess.run(['bash',str(repo/'scripts/slurm/run_stage2_annotate_build.sbatch')],
                          env=env,cwd=repo).returncode


def merge(run_dir: Path) -> dict:
    plan = load_manifest(run_dir)
    failed, paths = [], []
    for entry in plan['videos']:
        vid = entry['video_id']
        path = run_dir/'teacher'/f'{vid}.jsonl'
        try:
            if source_digest(Path(entry['path'])) != entry['annotation_config']['source_sha256']:
                raise ValueError('Source changed after planning')
            end = checked_end(path, entry, plan['model'])
            if end != entry['duration_sec']:
                raise ValueError(f"Incomplete video: saved to {end}s of {entry['duration_sec']}s")
            if len(records(path)) != entry['expected']['events']:
                raise ValueError('Unexpected teacher event count')
            paths.append(path)
        except (OSError, ValueError, TypeError, KeyError) as error:
            failed.append({'video_id':vid, 'error':str(error)})
    report = {'status':'incomplete' if failed else 'ready',
              'selected_videos':len(plan['videos']), 'complete_videos':len(paths),
              'failed_videos':failed, 'expected_events':plan['expected_events'],
              'expected_rows':plan['expected_rows'], 'prompt_sha256':PROMPT_SHA256}
    if failed:
        write_json(run_dir/'report.json', report)
        raise ValueError(f'{len(failed)} incomplete or incompatible videos; see report.json')
    count = build_files(paths, run_dir/'blocks.jsonl')
    if count != plan['expected_rows']:
        raise ValueError('Merged block count differs from frozen manifest')
    report['rows'] = count
    report['events'] = sum(len(records(path)) for path in paths)
    write_json(run_dir/'report.json', report)
    return report


def submit(run_dir: Path, repo: Path, *, concurrency=4, walltime='24:00:00', dry_run=False) -> dict:
    plan = load_manifest(run_dir)
    if concurrency < 1:
        raise ValueError('concurrency must be positive')
    record_path = run_dir/'submissions.json'
    ledger = json.loads(record_path.read_text()) if record_path.exists() else {'attempts':[]}
    old_ids = [str(job) for attempt in ledger['attempts'] for job in attempt.get('jobs',[])]
    if old_ids and not dry_run:
        current = subprocess.check_output(['squeue','-h','-u',str(os.getuid()),'-o','%F %T'],text=True)
        active = [line for line in current.splitlines() if line.split() and line.split()[0] in old_ids]
        if active:
            raise ValueError(f'A previous submission is still active; refusing duplicates: {active}')
    logs = run_dir/'logs'
    logs.mkdir(exist_ok=True)
    common = ['sbatch','--parsable','--export=ALL',f'--chdir={repo}']
    commands = []
    if plan['shards']:
        commands.append(common+['--job-name=stage2-full',f'--time={walltime}',
            f"--array=0-{len(plan['shards'])-1}%{concurrency}",
            f'--output={logs}/stage2_full_%A_%a.out',f'--error={logs}/stage2_full_%A_%a.err',
            str(repo/'scripts/slurm/run_stage2_full_array.sbatch')])
        if plan.get('seed_dependency') and not any(a.get('array_job') for a in ledger['attempts']):
            commands[-1].insert(1,f"--dependency=afterany:{plan['seed_dependency']}")
    merge_cmd = common+['--job-name=stage2-merge',
        f'--output={logs}/stage2_merge_%j.out',f'--error={logs}/stage2_merge_%j.err',
        str(repo/'scripts/slurm/run_stage2_full_merge.sbatch')]
    env = {**os.environ,'REPO':str(repo),'STAGE2_RUN_DIR':str(run_dir)}
    attempt = {'walltime':walltime,'concurrency':concurrency,'submitted_at_unix':time.time(),'jobs':[]}
    if dry_run:
        return {'dry_run':True,'commands':commands+[merge_cmd],
                'merge_dependency':'afterany:<array_job>','pending_videos':plan['pending_videos']}
    ledger['attempts'].append(attempt)
    write_json(record_path, ledger)
    for command in commands:
        job = subprocess.check_output(command,env=env,text=True).strip().split(';')[0]
        if not job.isdigit():
            raise ValueError('Unrecognized sbatch job ID')
        attempt['jobs'].append(job)
        attempt['array_job'] = job
        write_json(record_path, ledger)
    if attempt.get('array_job'):
        merge_cmd.insert(1, f"--dependency=afterany:{attempt['array_job']}")
    job = subprocess.check_output(merge_cmd,env=env,text=True).strip().split(';')[0]
    if not job.isdigit():
        raise ValueError('Unrecognized merge job ID')
    attempt['jobs'].append(job)
    attempt['merge_job'] = job
    write_json(record_path, ledger)
    return attempt


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest='action',required=True)
    prepare_p = sub.add_parser('prepare')
    prepare_p.add_argument('--video-map',type=Path,required=True)
    prepare_p.add_argument('--eval-map',type=Path,required=True)
    prepare_p.add_argument('--video-root',type=Path,required=True)
    prepare_p.add_argument('--shards',type=int,default=8)
    prepare_p.add_argument('--seed-teacher-dir',type=Path)
    prepare_p.add_argument('--seed-dependency')
    worker_p = sub.add_parser('worker')
    worker_p.add_argument('--shard',type=int,required=True)
    sub.add_parser('merge')
    submit_p = sub.add_parser('submit')
    submit_p.add_argument('--concurrency',type=int,default=4)
    submit_p.add_argument('--walltime',default='24:00:00')
    submit_p.add_argument('--dry-run',action='store_true')
    for child in (prepare_p,worker_p,sub.choices['merge'],submit_p):
        child.add_argument('--run-dir',type=Path,required=True)
    for child in (worker_p,submit_p):
        child.add_argument('--repo',type=Path,default=Path.cwd())
    args = parser.parse_args()
    run = args.run_dir.resolve()
    if args.action == 'prepare':
        result = prepare(args.video_map,args.eval_map,run,video_root=args.video_root,
                         shards=args.shards,seed_teacher_dir=args.seed_teacher_dir,
                         seed_dependency=args.seed_dependency)
        print(json.dumps({k:v for k,v in result.items() if k!='videos'},indent=2))
    elif args.action == 'worker':
        raise SystemExit(worker(run,args.shard,args.repo.resolve()))
    elif args.action == 'merge':
        print(json.dumps(merge(run),indent=2))
    else:
        print(json.dumps(submit(run,args.repo.resolve(),concurrency=args.concurrency,
                               walltime=args.walltime,dry_run=args.dry_run),indent=2))


if __name__ == '__main__':
    main()
