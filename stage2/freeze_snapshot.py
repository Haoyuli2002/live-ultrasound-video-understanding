"""Freeze an audited Visual+ASR snapshot, preserving the original snapshot."""
import argparse
import json
from pathlib import Path
import shutil
import tempfile
from .data import load_blocks
from .provenance import digest
from .release import validate_release
from .dataset_versions import validate_dataset_version, VISUAL_ASR_MODALITIES


def freeze_snapshot(snapshot, output):
    source, output = Path(snapshot).resolve(), Path(output).resolve()
    original = json.loads((source/'manifest.json').read_text())
    if original.get('schema') != 'stage2_av_asr_audited_dataset_v1' or original.get('status') != 'audit_complete_not_frozen_release':
        raise ValueError('Expected an audited Visual+ASR snapshot')
    if original.get('dataset_version', 'visual_asr') != 'visual_asr':
        raise ValueError('Snapshot dataset_version must be visual_asr')
    for name in ('train.jsonl','eval.jsonl','videos.json','video_map.json'):
        if digest(source/name) != original['files'][name]:
            raise ValueError(f'Snapshot hash mismatch: {name}')
    records = json.loads((source/'videos.json').read_text())
    by_id = {v['video_id']:v for v in records}
    if len(by_id) != len(records) or len(records) != original['video_count']:
        raise ValueError('Duplicate or missing video records')
    splits, totals = {}, {}
    for split in ('train','eval'):
        data = load_blocks(source/f'{split}.jsonl',validate_text=True)
        ids = {v['video_id'] for v in records if v['split']==split}
        if set(data)!=ids or len(ids)!=original[f'{split}_videos']:
            raise ValueError(f'{split}: split identity mismatch')
        total = 0
        for vid, groups in data.items():
            meta=by_id[vid]; n=int(meta['duration_seconds']//10); minutes=int(meta['duration_seconds']//60)
            windows=[tuple(x['window']) for g in groups for x in g['local_sub_summaries']]
            if windows != [(10*i,10*(i+1)) for i in range(n)]:
                raise ValueError(f'{vid}: incomplete Local coverage')
            full=[g for g in groups if g['block_window'][1]-g['block_window'][0]==60]
            if len(full)!=minutes or any(g.get('block_summary_target') or g.get('global_summary_target') for g in groups if g not in full):
                raise ValueError(f'{vid}: invalid Block/Global coverage')
            count=n+2*minutes
            if count!=meta['train_targets'] or n!=meta['local_targets'] or minutes!=meta['full_blocks']:
                raise ValueError(f'{vid}: target metadata mismatch')
            total+=count
        if total!=original[f'{split}_targets']: raise ValueError(f'{split}: total target mismatch')
        splits[split]=sorted(ids);totals[split]=total
    if set(splits['train']) & set(splits['eval']) or sum(totals.values())!=original['expected_targets']:
        raise ValueError('Overlapping splits or incorrect total')
    if {v['source_sha256'] for v in records if v['split']=='train'} & {v['source_sha256'] for v in records if v['split']=='eval'}:
        raise ValueError('Source content overlaps train and eval')
    if output.exists(): raise FileExistsError('Use a new immutable release directory')
    output.parent.mkdir(parents=True,exist_ok=True)
    temp=Path(tempfile.mkdtemp(prefix='.freeze-',dir=output.parent))
    try:
        for name in ('train.jsonl','eval.jsonl','video_map.json','videos.json'):
            shutil.copy2(source/name,temp/name)
        manifest=dict(schema='stage2_accepted_release_v1',status='ready',training_ready=True,
            full_dataset_complete=True,dataset_version='visual_asr',label_modalities=sorted(VISUAL_ASR_MODALITIES),
            model_input_modalities=['video'],direct_audio_listening=False,clinical_expert_review=False,
            video_count=len(records),targets=sum(totals.values()),train_videos=len(splits['train']),eval_videos=len(splits['eval']),
            splits=splits,source_snapshot_sha256=digest(source/'manifest.json'),source_audit_root=original['source_root'],
            raw_and_review_ledgers_retained=True,files={f'{s}.jsonl':digest(temp/f'{s}.jsonl') for s in splits},
            videos=[dict(v,filename=v['video_id']+'.mp4',duration=v['duration_seconds'],targets=v['train_targets']) for v in records])
        (temp/'manifest.json').write_text(json.dumps(manifest,ensure_ascii=False,indent=2)+'\n')
        validate_dataset_version(validate_release(temp/'manifest.json'),'visual_asr')
        temp.rename(output)
    finally:
        if temp.exists(): shutil.rmtree(temp)
    return manifest


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--snapshot',required=True);p.add_argument('--output',required=True)
    a=p.parse_args();m=freeze_snapshot(a.snapshot,a.output)
    print(json.dumps({k:m[k] for k in ('dataset_version','training_ready','video_count','targets')},indent=2))

if __name__=='__main__': main()
