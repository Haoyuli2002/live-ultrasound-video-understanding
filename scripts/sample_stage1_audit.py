#!/usr/bin/env python3
"""Select reproducible, time-stratified Stage 1 targets for actual content review.

Automatic checks do not certify ASR/medical accuracy. review.txt and payload.json
are review inputs, not a claim that the selected sentences passed human review.
"""
from __future__ import annotations

import argparse
from collections import defaultdict
import hashlib
import json
from pathlib import Path
import random
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from stage1.data import CONDITIONS, frame_timestamps


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read_rows(path):
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def extract(dataset: Path, transcripts: Path, output: Path, *, seed=20261031,
            videos=10, sentences=10):
    if output.exists():
        raise FileExistsError('Use a new audit directory to preserve prior review evidence')
    if videos < 1 or sentences < 1:
        raise ValueError('Video and sentence counts must be positive')
    report=json.loads((dataset/'report.json').read_text())
    if report['status']!='ready':raise ValueError('Dataset is not ready')
    mapping=json.loads((dataset/'video_map.json').read_text())
    groups=defaultdict(list)
    for row in read_rows(dataset/'samples.jsonl'):
        groups[(row['video_id'],row['sentence_id'])].append(row)
    by_video=defaultdict(list)
    for key, rows in groups.items():
        if len(rows)!=3 or {r['condition'] for r in rows}!=set(CONDITIONS):
            raise ValueError(f'Broken triplet: {key}')
        through=next(r for r in rows if r['condition']=='through_with_asr')
        by_video[key[0]].append(through)
    eligible=sorted(k for k,rows in by_video.items() if len(rows)>=sentences)
    if len(eligible)<videos:
        raise ValueError(f'Only {len(eligible)} videos have {sentences} distinct targets')
    rng=random.Random(seed)
    selected=sorted(rng.sample(eligible,videos))
    units=defaultdict(list)
    for row in read_rows(dataset/'sentences.jsonl'):
        if row['video_id'] in selected:units[row['video_id']].append(row)
    result=[]
    for vid in selected:
        targets=sorted(by_video[vid],key=lambda r:(r['sentence_window'][0],r['sentence_id']))
        # One random sentence from each temporal decile, preserving broad coverage.
        picked=[rng.choice(targets[i*len(targets)//sentences:(i+1)*len(targets)//sentences])
                for i in range(sentences)]
        source=transcripts/f'{vid}.json'
        raw=json.loads(source.read_text())
        source_hash=digest(source)
        ordered=sorted(units[vid],key=lambda u:u['candidate_id'])
        for index,row in enumerate(picked,1):
            start,end=row['sentence_window']
            unit=next(u for u in ordered if u['candidate_id']==row['sentence_id'])
            previous=[u for u in ordered if u['candidate_id']<row['sentence_id']]
            following=[u for u in ordered if u['candidate_id']>row['sentence_id']]
            triplet=groups[(vid,row['sentence_id'])]
            before=next(r for r in triplet if r['condition']=='before_with_asr')
            masked=next(r for r in triplet if r['condition']=='before_mask_asr')
            segments=[{'segment_index':i,**raw['segments'][i]} for i in row['source_segment_indices']]
            uniform=frame_timestamps(row['video_window'],30,sampling='uniform',recent_seconds=15)
            recent=frame_timestamps(row['video_window'],30,sampling='recent_sparse',recent_seconds=15)
            history=' '.join(u['text'] for u in previous)
            checks={
                'min_seconds':end-start>=5,'min_words':len(row['target'].split())>=10,
                'source_hash_matches':row['source_sha256']==source_hash,
                'target_equals_sentence_unit':unit['text']==row['target'],
                'native_word_timestamps':row['alignment']=='whisper_word_timestamps',
                'history_exactly_prior_units':row['historical_asr']==history and before['historical_asr']==history,
                'history_ends_before_target':all(u['end']<=start+.001 for u in previous),
                'mask_empty':masked['historical_asr_masked'] and not masked['historical_asr'],
                'paired_targets_equal':all(r['target']==row['target'] for r in triplet),
                'causal_windows':before['video_window']==[0.0,start] and masked['video_window']==[0.0,start]
                                 and row['video_window']==[0.0,end],
                'source_words_present':all(s.get('words') for s in segments),
                'recent_sparse_target_frame':any(start<=t<end for t in recent),
            }
            result.append({'video_id':vid,'review_index':index,'sentence_id':row['sentence_id'],
                'video_path':mapping[vid],'sentence_window':[start,end],'seconds':end-start,
                'words':len(row['target'].split()),'target':row['target'],
                'history_tail':' '.join(history.split()[-80:]),
                'previous_sentence':previous[-1]['text'] if previous else '',
                'next_sentence':following[0]['text'] if following else '',
                'mean_word_probability':row['mean_word_probability'],'source_segments':segments,
                'uniform_target_frame':any(start<=t<end for t in uniform),'checks':checks,
                'review_status':'pending_content_review',
                'review_notes_zh':'','audio_review_status':'not_listened'})
    summary={'schema':'stage1_audit_10x10_v1','dataset':str(dataset.resolve()),'seed':seed,
        'sampling':'uniform random videos among those with enough targets; one target per temporal bin',
        'eligible_video_count':len(eligible),'excluded_for_insufficient_targets':len(by_video)-len(eligible),
        'selected_videos':selected,'videos':videos,'targets_per_video':sentences,'total_targets':len(result),
        'dataset_hashes':{name:digest(dataset/name) for name in ('samples.jsonl','sentences.jsonl','report.json')},
        'automatic_check_failures':[{'video_id':r['video_id'],'sentence_id':r['sentence_id'],
            'failed':[k for k,v in r['checks'].items() if not v]} for r in result if not all(r['checks'].values())],
        'uniform_no_target_frames':sum(not r['uniform_target_frame'] for r in result),
        'content_review_complete':False,'audio_review_complete':False}
    assert len(result)==videos*sentences
    output.mkdir(parents=True)
    (output/'payload.json').write_text(json.dumps({'summary':summary,'samples':result},ensure_ascii=False,indent=2)+'\n')
    (output/'summary.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2)+'\n')
    text=['Stage1 训练集抽样审计输入（尚未完成内容审计）',
          f'固定种子 {seed}；{videos} 个视频，每个 {sentences} 个不重复目标句；覆盖时间分段。',
          '自动检查仅验证结构和来源，不等同于医学内容、ASR准确率或听校通过。','']
    for sample in result:
        text.extend([f"[{sample['video_id']} #{sample['review_index']}] sentence_id={sample['sentence_id']} "
                     f"window={sample['sentence_window']} seconds={sample['seconds']:.2f} words={sample['words']}",
                     sample['target'],f"前句：{sample['previous_sentence']}",f"后句：{sample['next_sentence']}",
                     '自动检查：'+json.dumps(sample['checks'],ensure_ascii=False),
                     '内容审计意见：待审计；音频：未听校',''])
    (output/'review.txt').write_text('\n'.join(text),encoding='utf-8')
    return summary


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--dataset',type=Path,required=True)
    parser.add_argument('--transcripts',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--seed',type=int,default=20261031)
    parser.add_argument('--videos',type=int,default=10)
    parser.add_argument('--sentences',type=int,default=10)
    args=parser.parse_args()
    result=extract(args.dataset,args.transcripts,args.output,seed=args.seed,
                   videos=args.videos,sentences=args.sentences)
    print(json.dumps(result,ensure_ascii=False,indent=2))
    if result['automatic_check_failures']:raise SystemExit(1)


if __name__=='__main__':main()
