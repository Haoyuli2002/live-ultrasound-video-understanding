#!/usr/bin/env python3
"""Build the independent new-QA section from canonical English and reading translations."""
import argparse
import hashlib
import json
from pathlib import Path
from urllib.parse import quote

REPO = Path(__file__).resolve().parents[1]
DEFAULT = REPO / 'results/stage3_visual_dense_3v_v2_20261119'
TITLES = {
    '3X0o1-V1jtE': {'en': 'Urinary tract: kidneys and bladder', 'zh': '泌尿系统：肾脏与膀胱'},
    'CzLE1i4NSoU': {'en': 'Subclavian region vessel scanning', 'zh': '锁骨下区域血管扫描'},
    '0WAV2h2N6MY': {'en': 'Genicular nerve block tutorial', 'zh': '膝神经阻滞教程'},
}

def rows(path):
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]

def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()

def url(path):
    # Browser paths are relative to /viz/, including the existing cluster_data symlink.
    relative = path.absolute().relative_to(REPO)
    return '../' + quote(relative.as_posix(), safe='/')

def build(source, output):
    tasks = rows(source / 'train_tasks.jsonl')
    translations = rows(source / 'qa_bilingual_en_zh.jsonl')
    translated = {t['task_id']: t for t in translations}
    samples = rows(source / 'train_samples.jsonl')
    media = json.loads((source / 'media_manifest.json').read_text())
    videos = []
    assert len(translated) == len(translations) == len(tasks)
    assert {t['task_id'] for t in tasks} == set(translated)
    for video in media['videos']:
        vid = video['video_id']
        path = REPO / 'cluster_data/videos/train_full295' / (vid + '.mp4')
        if not path.is_file() or path.resolve() != Path(video['source_path']).resolve():
            raise ValueError(f'Missing or mismatched source video: {path}')
        entries = []
        for task in [t for t in tasks if t['video_id'] == vid]:
            translated_task = translated[task['task_id']]
            assert translated_task['source_sha256'] == sha(source / 'train_tasks.jsonl')
            assert (translated_task['question_en'], translated_task['answer_en']) == (task['question'], task['reference_answer'])
            decisions = []
            for sample in sorted((s for s in samples if s['task_id'] == task['task_id']), key=lambda s: s['video_window'][1]):
                assert sample['question'] == task['question'] and sample['video_id'] == vid
                current = sample['video_window'][1]
                ready = current >= task['answer_time']
                assert sample['target'] == ('<ANSWER> ' + task['reference_answer'] if ready else '<WAIT>')
                decisions.append({'time': current, 'status': 'ANSWER' if ready else 'WAIT', 'target_en': sample['target']})
            assert [d['time'] for d in decisions] == task['evaluation_times']
            evidence = []
            for frame in task['evidence']:
                path_frame = source / frame['frame']
                assert path_frame.is_file() and sha(path_frame) == frame['sha256']
                evidence.append({'time': frame['time'], 'role': frame['role'], 'url': url(path_frame), 'approx_pts': frame['approx_pts']})
            entries.append({
                'task_id': task['task_id'], 'qa_type': task['qa_type'], 'subset': task['subset'],
                'query_time': task['query_time'], 'answer_time': task['answer_time'],
                'question': {'en': task['question'], 'zh': translated_task['question_zh']},
                'answer': {'en': task['reference_answer'], 'zh': translated_task['answer_zh']},
                'decisions': decisions, 'evidence': evidence,
            })
        assert entries
        videos.append({'video_id': vid, 'title': TITLES[vid], 'duration': video['duration'], 'url': url(path),
                       'poster': entries[0]['evidence'][0]['url'], 'tasks': entries})
    assert sum(len(v['tasks']) for v in videos) == len(tasks)
    assert sum(len(t['decisions']) for v in videos for t in v['tasks']) == len(samples)
    sources = ['train_tasks.jsonl', 'train_samples.jsonl', 'qa_bilingual_en_zh.jsonl', 'media_manifest.json']
    data = {'version': source.name, 'counts': {'videos': len(videos), 'qa': len(tasks), 'decisions': len(samples)},
            'videos': videos, 'review_url': url(source / 'QA_BILINGUAL_en_zh.md'),
            'tasks_url': url(source / 'train_tasks.jsonl'), 'samples_url': url(source / 'train_samples.jsonl'),
            'source_sha256': {name: sha(source / name) for name in sources}}
    output.write_text(json.dumps(data, ensure_ascii=False, indent=2) + '\n')
    print(f'Built {output}: {len(videos)} videos, {len(tasks)} QA, {len(samples)} decisions')

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', type=Path, default=DEFAULT)
    parser.add_argument('--output', type=Path, default=REPO / 'viz/new-qa.json')
    args = parser.parse_args()
    build(args.source.absolute(), args.output)
