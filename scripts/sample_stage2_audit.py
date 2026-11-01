#!/usr/bin/env python3
"""Sample full Stage 2 minutes without filtering or rewriting their eight targets.

Default: 10 complete videos, one uniformly sampled complete minute per video.
Read-only with respect to the source run. No GPU, ASR or teacher requests.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import hashlib
import json
from pathlib import Path
import random
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from stage2.annotate import resume_start
from stage2.full_run import load_manifest
from stage2.raw_collection import text_issues


def extract(run: Path, output: Path, *, videos=10, blocks_per_video=1, seed=20261101):
    if output.exists():
        raise FileExistsError('Use a new audit directory to preserve earlier review inputs')
    if videos < 1 or blocks_per_video < 1:
        raise ValueError('Video and block counts must be positive')
    manifest = load_manifest(run)
    eligible, excluded = {}, []
    for entry in manifest['videos']:
        vid = entry['video_id']
        source = run/'teacher'/f'{vid}.jsonl'
        try:
            data = source.read_bytes()
            rows = [json.loads(line) for line in data.decode().splitlines() if line.strip()]
            end = resume_start(rows, video_id=vid, duration_sec=entry['duration_sec'],
                               model=manifest['model'], config=entry['annotation_config'], validate_text=False)
            if end != entry['duration_sec'] or len(rows) != entry['expected']['events']:
                raise ValueError(f'incomplete: saved through {end}s of {entry["duration_sec"]}s')
            groups = defaultdict(list)
            for row in rows:
                a,b = row['block_window']
                if abs(b-a-60) < .001:
                    groups[(a,b)].append(row)
            if len(groups) < blocks_per_video:
                raise ValueError(f'only {len(groups)} full 60-second blocks')
            eligible[vid] = (entry, source, hashlib.sha256(data).hexdigest(), groups)
        except (OSError, ValueError, TypeError, KeyError) as error:
            excluded.append({'video_id':vid, 'reason':str(error)})
    if len(eligible) < videos:
        raise ValueError(f'Only {len(eligible)} complete videos have enough full blocks')
    rng = random.Random(seed)
    selected = sorted(rng.sample(sorted(eligible), videos))
    result, flag_counts = [], Counter()
    order = {'local_10s':0, 'block_60s':1, 'global_prefix':2}
    kind = {'local_10s':'local', 'block_60s':'block', 'global_prefix':'global'}
    for vid in selected:
        entry,source,digest,groups = eligible[vid]
        for window in sorted(rng.sample(sorted(groups), blocks_per_video)):
            samples = sorted(groups[window], key=lambda r:(order[r['label_type']], r['window'][0]))
            if Counter(r['label_type'] for r in samples) != Counter(local_10s=6, block_60s=1, global_prefix=1):
                raise ValueError(f'{vid} {window}: expected exactly 6 local + 1 block + 1 global')
            flags = []
            for row in samples:
                issues = text_issues(row['target'],kind[row['label_type']])
                flag_counts.update(issues)
                flags.append({'label_type':row['label_type'],'window':row['window'],'issues':issues})
            result.append({'video_id':vid,'video_path':entry['path'],
                           'source_video_sha256':entry['annotation_config']['source_sha256'],
                           'teacher_file':str(source.resolve()),'teacher_file_sha256':digest,
                           'block_window':list(window),'samples':samples,'text_flags':flags,
                           'review_status':'pending_content_review'})
    summary = {'schema':'stage2_audit_full_minutes_v1','run':str(run.resolve()),
               'source':'original teacher/ only; complete videos only', 'seed':seed,
               'sampling':'uniform random videos; uniform random full blocks within each selected video',
               'eligible_videos':len(eligible),'excluded_videos':excluded,'selected_videos':selected,
               'videos':videos,'blocks_per_video':blocks_per_video,'blocks':len(result),
               'summaries':len(result)*8,'label_counts':{'local_10s':len(result)*6,
                   'block_60s':len(result),'global_prefix':len(result)},
               'text_issue_counts':dict(flag_counts),'filtered_targets':0,'rewritten_targets':0,
               'manifest_sha256':hashlib.sha256((run/'manifest.json').read_bytes()).hexdigest(),
               'content_review_complete':False,'visual_review_complete':False}
    text = ['Stage2 抽样审计输入（尚未完成内容或视频核验）',
            f'{videos} 个视频，{len(result)} 个完整 60s block，共 {len(result)*8} 条原文。',
            '每组：6×Local 10s + 1×Block 60s + 1×Global 历史前缀。',
            'Global 的区间是 [0, block_end)，并非只包含当前 60 秒。',
            '仅标记格式与字数问题，不过滤、不改写；源文件保持不变。','']
    for index,block in enumerate(result,1):
        text.extend([f'=== {index}. {block["video_id"]} | block={block["block_window"]} ===',
                     f'视频：{block["video_path"]}'])
        for row,flags in zip(block['samples'],block['text_flags']):
            text.extend([f'\n{row["label_type"]} | window={row["window"]} | words={len(row["target"].split())}',
                         row['target']])
            if flags['issues']: text.append('标记：'+', '.join(flags['issues']))
        text.extend(['\n审计意见：待审阅',''])
    output.mkdir(parents=True)
    (output/'payload.json').write_text(json.dumps({'summary':summary,'blocks':result},ensure_ascii=False,indent=2)+'\n')
    (output/'summary.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2)+'\n')
    (output/'review.txt').write_text('\n'.join(text),encoding='utf-8')
    return summary


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run-dir',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--videos',type=int,default=10)
    parser.add_argument('--blocks-per-video',type=int,default=1)
    parser.add_argument('--seed',type=int,default=20261101)
    args=parser.parse_args()
    print(json.dumps(extract(args.run_dir,args.output,videos=args.videos,
                            blocks_per_video=args.blocks_per_video,seed=args.seed),ensure_ascii=False,indent=2))


if __name__=='__main__':main()
