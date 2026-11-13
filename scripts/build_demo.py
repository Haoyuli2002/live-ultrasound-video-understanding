"""Build the multi-video stage browser from existing records, never regenerate labels."""
import argparse
import hashlib
import json
import subprocess
import sys
from pathlib import Path
from urllib.parse import quote

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from stage1.data import CONDITIONS, frame_timestamps
from stage1.model import messages
from stage2.data import load_blocks


def file_sha256(path):
    h=hashlib.sha256()
    with path.open('rb') as stream:
        for chunk in iter(lambda:stream.read(1024*1024),b''): h.update(chunk)
    return h.hexdigest()


def read_rows(path):
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def stage1_payload(rows, vid, budget, max_asr):
    groups = {}
    for row in rows:
        if row['video_id'] != vid:
            continue
        group = groups.setdefault(row['sentence_id'], {})
        if row['condition'] in group:
            raise ValueError('Duplicate Stage 1 condition')
        group[row['condition']] = row
    result = []
    for sid, group in sorted(groups.items()):
        if set(group) != set(CONDITIONS) or len({r['target'] for r in group.values()}) != 1:
            raise ValueError('Stage 1 triplet incomplete or references disagree')
        base = group['before_with_asr']
        conditions = []
        for name in CONDITIONS:
            row = group[name]
            texts = [part['text'] for part in messages(row, [], max_asr_chars=max_asr)[1]['content'] if part['type'] == 'text']
            conditions.append({**row, 'prompt_texts': texts,
                'frame_times': frame_timestamps(row['video_window'], max_frames=budget, sampling='uniform')})
        result.append({'sentence_id': sid, 'target': base['target'],
                       'sentence_window': base['sentence_window'], 'conditions': conditions})
    return result


def build(config_path, output):
    repo = Path(__file__).resolve().parents[1]
    config = json.loads(config_path.read_text())
    videos = []
    for spec in config['videos']:
        video_path = (repo/spec['video']).resolve()
        if not video_path.is_file():
            raise FileNotFoundError(video_path)
        info = json.loads(subprocess.check_output(['ffprobe','-v','error','-show_entries','format=duration','-of','json',str(video_path)]))
        item = {k: spec[k] for k in ('video_id','title')}
        item.update(duration=float(info['format']['duration']),
                    video_url=quote('../'+str(video_path.relative_to(repo))), stages={})
        for stage in ('stage1','stage2','stage3'):
            source = spec.get(stage)
            if not source:
                item['stages'][stage] = {'available': False, 'note': '尚未接入该视频的数据；未生成占位样本。'}
                continue
            path = repo/source['path']
            if not path.is_file():
                raise FileNotFoundError(path)
            payload = {'available': True, 'source': source['path'], 'note': source['note'],
                       'sha256': hashlib.sha256(path.read_bytes()).hexdigest()}
            if source.get('video'):
                stage_video = (repo/source['video']).resolve()
                if not stage_video.is_file():
                    raise FileNotFoundError(stage_video)
                stage_info = json.loads(subprocess.check_output(['ffprobe','-v','error','-show_entries','format=duration','-of','json',str(stage_video)]))
                payload['video_url'] = quote('../'+str(stage_video.relative_to(repo)))
                payload['duration'] = float(stage_info['format']['duration'])
            if stage == 'stage2' and source.get('provenance_path'):
                provenance = json.loads((repo/source['provenance_path']).read_text())
                acceptance = json.loads((repo/source['acceptance_path']).read_text())
                if (provenance['video_id'] != spec['video_id'] or provenance['blocks_sha256'] != payload['sha256']
                        or not acceptance['final_acceptance_complete'] or not acceptance['native_acceptance_complete']
                        or hashlib.sha256((repo/source['acceptance_path']).read_bytes()).hexdigest() != provenance['acceptance_sha256']):
                    raise ValueError('Stale or unaccepted reviewed demo source: '+spec['video_id'])
                exact_video=repo/source.get('video',spec['video'])
                if file_sha256(exact_video) != provenance['source_video_sha256']:
                    raise ValueError('Reviewed demo video bytes changed: '+spec['video_id'])
                payload['review'] = dict(status='accepted', split=provenance['split'],
                    basis='source_native_visual_review', original_size=provenance['original_size'],
                    targets=acceptance['targets'], release_manifest_sha256=provenance['release_manifest_sha256'],
                    source_video_sha256=provenance['source_video_sha256'])
            if stage == 'stage1':
                payload['frame_budget'] = source.get('frame_budget',30)
                payload['frame_size'] = source.get('frame_size',224)
                payload['max_asr_chars'] = source.get('max_asr_chars',4000)
                payload['samples'] = stage1_payload(read_rows(path), spec['video_id'],payload['frame_budget'],payload['max_asr_chars'])
            elif stage == 'stage2':
                payload['samples'] = load_blocks(path)[spec['video_id']]
                payload['coverage_end'] = max(label['window'][1] for block in payload['samples'] for label in block['local_sub_summaries'])
            else:
                payload['samples'] = [r for r in read_rows(path) if r['video_id'] == spec['video_id']]
            if not payload['samples']:
                raise ValueError(f'No matching {stage} records for {spec["video_id"]}')
            item['stages'][stage] = payload
        videos.append(item)
    presentation = config.get('presentation', {})
    by_id = {v['video_id']: v for v in videos}
    for feature in presentation.get('features', []):
        stage = by_id[feature['video_id']]['stages']['stage2']
        block = stage['samples'][feature['sample']]
        block['local_sub_summaries'][feature['local']]
        preview = (output.parent / feature['preview']).resolve()
        if not preview.is_file() or not stage.get('review') or stage['review']['release_manifest_sha256'] != presentation['release_manifest_sha256']:
            raise ValueError('Invalid presentation example: '+feature['video_id'])
    output.parent.mkdir(parents=True,exist_ok=True)
    output.write_text(json.dumps({'videos':videos, 'presentation':presentation},ensure_ascii=False,indent=2))
    print(f'Wrote {len(videos)} videos -> {output}')


if __name__ == '__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--config',type=Path,default=Path('viz/demo_sources.json'))
    p.add_argument('--output',type=Path,default=Path('viz/catalog.json'))
    a=p.parse_args();build(a.config,a.output)
