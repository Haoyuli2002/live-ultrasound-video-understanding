"""Freeze a complete accepted train/eval release; partial acceptance cannot train.

Per-video native review or the user's explicit first-pass waiver is required.
This is a data/provenance gate, never a claim of clinical expert validation.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import shutil
import tempfile

from .data import load_blocks
from .build_data import build_rows
from .provenance import digest


def rows(path):
    return [json.loads(s) for s in Path(path).read_text().splitlines() if s.strip()]


def write_rows(path,values):
    Path(path).write_text(''.join(json.dumps(v,ensure_ascii=False)+'\n' for v in values))


def prepare_release(acceptance_root,output,check_only=False):
    root=Path(acceptance_root);output=Path(output)
    plan=json.loads((root/'plan.json').read_text())
    index=json.loads((root/'accepted_videos/index.json').read_text())
    expected={v['video_id']:v for v in plan['videos']}
    accepted={v['video_id']:v for v in index['completed_videos']}
    if len(expected)!=len(plan['videos']) or len(accepted)!=len(index['completed_videos']):
        raise ValueError('Duplicate video identities in acceptance plan/index')
    if set(accepted)!=set(expected):
        return dict(status='waiting_for_acceptance',training_ready=False,
                    expected_videos=len(expected),accepted_videos=len(set(accepted)&set(expected)),
                    missing_videos=sorted(set(expected)-set(accepted)),
                    unexpected_videos=sorted(set(accepted)-set(expected)))
    records=[];split_rows={'train':[],'eval':[]};split_ids={'train':[],'eval':[]}
    source_splits={};total=0
    for vid in sorted(expected):
        item=expected[vid];split=item['split']
        if split not in split_rows or accepted[vid]['split']!=split:
            raise ValueError(f'Wrong split for {vid}')
        d=root/'accepted_videos'/split/vid
        manifest=json.loads((d/'manifest.json').read_text())
        waiver=manifest.get('user_first_pass_acceptance',{})
        if not (manifest.get('final_acceptance_complete') and manifest.get('boundary_review_complete')
                and manifest.get('all_targets_have_current_visual_support')
                and (manifest.get('native_acceptance_complete') or waiver.get('accepted'))):
            raise ValueError(f'Incomplete per-video visual acceptance: {vid}')
        if manifest['source_sha256']!=item['source_sha256']:
            raise ValueError(f'Accepted source changed: {vid}')
        previous=source_splits.setdefault(item['source_sha256'],split)
        if previous!=split:raise ValueError('Identical source content occurs in train and eval')
        events=rows(d/'teacher.reviewed.jsonl');blocks=rows(d/'blocks.reviewed.jsonl')
        if (not events or any(e['video_id']!=vid or not str(e.get('target','')).strip() for e in events)
                or build_rows(events,validate_text=False)!=blocks):
            raise ValueError(f'Invalid teacher/blocks roundtrip: {vid}')
        count=int(item['duration']//10)+2*int(item['duration']//60)
        if len(events)!=count or manifest['targets']!=count:
            raise ValueError(f'Incomplete target coverage: {vid}')
        # Bind acceptance to the actual current review and transition ledger.
        source_dir=root/'videos'/vid
        for name,field in [('reviews.jsonl','review_sha256'),('boundary_evidence.json','boundary_evidence_sha256')]:
            if digest(source_dir/name)!=manifest[field]:raise ValueError(f'Stale {field}: {vid}')
        records.append(dict(video_id=vid,split=split,filename=vid+'.mp4',
                            source_sha256=item['source_sha256'],duration=item['duration'],targets=count,
                            acceptance_basis=manifest['acceptance_basis'],
                            acceptance_manifest_sha256=digest(d/'manifest.json'),
                            teacher_sha256=digest(d/'teacher.reviewed.jsonl'),
                            blocks_sha256=digest(d/'blocks.reviewed.jsonl')))
        split_rows[split].extend(blocks);split_ids[split].append(vid);total+=count
    if total!=plan['targets'] or not all(split_ids.values()):
        raise ValueError('Wrong release target totals or missing independent split')
    report=dict(status='ready',training_ready=True,videos=len(records),targets=total,
                train_videos=len(split_ids['train']),eval_videos=len(split_ids['eval']))
    if check_only:return report
    if output.exists():raise FileExistsError('Use a new immutable release directory')
    output.parent.mkdir(parents=True,exist_ok=True)
    temporary=Path(tempfile.mkdtemp(prefix='.release-',dir=output.parent))
    try:
        for split in split_rows:
            write_rows(temporary/f'{split}.jsonl',split_rows[split])
            load_blocks(temporary/f'{split}.jsonl',validate_text=False)
        manifest={**report,'video_count':len(records),'schema':'stage2_accepted_release_v1','full_dataset_complete':True,
                  'source_plan_sha256':digest(root/'plan.json'),'splits':split_ids,'videos':records,
                  'files':{f'{s}.jsonl':digest(temporary/f'{s}.jsonl') for s in split_rows},
                  'clinical_expert_review':False,'raw_and_review_ledgers_retained':True}
        (temporary/'manifest.json').write_text(json.dumps(manifest,indent=2))
        temporary.rename(output)
    finally:
        if temporary.exists():shutil.rmtree(temporary)
    return report


def validate_release(manifest_path):
    path=Path(manifest_path);m=json.loads(path.read_text())
    if m.get('schema')!='stage2_accepted_release_v1' or m.get('training_ready') is not True or m.get('full_dataset_complete') is not True:
        raise ValueError('A complete frozen visually accepted release is required')
    train=set(m['splits']['train']);evaluation=set(m['splits']['eval'])
    if not train or not evaluation or train&evaluation:
        raise ValueError('Missing or overlapping train/eval IDs')
    if len(train)+len(evaluation)!=m['video_count']:
        raise ValueError('Invalid release identity counts')
    records=m['videos']
    if {v['video_id'] for v in records}!=train|evaluation or len(records)!=len(train)+len(evaluation):
        raise ValueError('Invalid release video records')
    if {v['source_sha256'] for v in records if v['split']=='train'} & {v['source_sha256'] for v in records if v['split']=='eval'}:
        raise ValueError('Train/eval source content overlap')
    for split,ids in [('train',train),('eval',evaluation)]:
        name=f'{split}.jsonl'
        if digest(path.parent/name)!=m['files'][name]:raise ValueError('Release data hash changed')
        blocks=load_blocks(path.parent/name,validate_text=False)
        if set(blocks)!=ids:raise ValueError('Release split data differs from manifest')
    return m


def resolve_video_map(manifest_path,video_root,output,input_map=None):
    m=validate_release(manifest_path);root=Path(video_root)
    supplied=json.loads(Path(input_map).read_text()) if input_map else {}
    result={}
    for record in m['videos']:
        vid=record['video_id'];split=record['split']
        choices=([Path(supplied[vid]).expanduser()] if vid in supplied else [])+[
            root/f'{split}_full295'/record['filename'],root/record['filename']]
        video=next((p.resolve() for p in choices if p.is_file()),None)
        if video is None:raise FileNotFoundError(f'Missing release video {vid}; provide --input-video-map')
        if digest(video)!=record['source_sha256']:raise ValueError(f'Source video hash mismatch: {vid}')
        result[vid]=str(video)
    if len(set(result.values()))!=len(result):raise ValueError('Multiple IDs resolve to one video path')
    Path(output).write_text(json.dumps(result,indent=2))
    return result


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--acceptance-root');p.add_argument('--output-dir');p.add_argument('--check-only',action='store_true')
    p.add_argument('--manifest');p.add_argument('--video-root');p.add_argument('--output-video-map')
    p.add_argument('--input-video-map')
    a=p.parse_args()
    if a.manifest:
        if a.output_video_map:
            if not a.video_root:p.error('--video-root is required')
            resolve_video_map(a.manifest,a.video_root,a.output_video_map,a.input_video_map)
        print(json.dumps({k:v for k,v in validate_release(a.manifest).items() if k not in ('videos','splits')},indent=2))
        return
    if not a.acceptance_root or not a.output_dir:p.error('Set --acceptance-root and --output-dir')
    result=prepare_release(a.acceptance_root,a.output_dir,a.check_only)
    print(json.dumps(result,indent=2))
    if not result['training_ready']:raise SystemExit(2)


if __name__=='__main__':main()
