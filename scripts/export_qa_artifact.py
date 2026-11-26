#!/usr/bin/env python3
"""Embed the frozen ten-video QA demo and evidence thumbnails for inline preview."""
import base64
import hashlib
import io
import json
from pathlib import Path

from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / 'viz/qa-examples/data.json'
DEST = ROOT / 'results/qa_demo_artifact_20261124'


def main():
    original = json.loads(SOURCE.read_text())
    data = json.loads(SOURCE.read_text())
    data['qa_v3'] = json.loads((SOURCE.parent / 'qa-v3.json').read_text())
    images = []
    paths = {}
    for video in data['videos']:
        video.pop('url')
        video.pop('poster')
        for task in video['tasks']:
            assert task['decisions'] and task['evidence']
            for evidence in task['evidence']:
                path = (SOURCE.parent / evidence.pop('url')).resolve()
                if path not in paths:
                    paths[path] = len(images)
                    images.append(path)
                evidence['image'] = paths[path]
    template = (ROOT / 'viz/qa-examples/artifact.template.html').read_text()
    # Segment thumbnails share the inline byte budget with QA evidence frames.
    clip_thumbs = []
    for video in data['videos']:
        for event in video.get('visual_events', []):
            encoded = event['thumbnail'].split(',', 1)[1]
            clip_thumbs.append((event, base64.b64decode(encoded)))
    for width, quality, clip_width in [(360, 50, 160), (320, 45, 112),
                                       (280, 40, 80), (240, 35, 56),
                                       (200, 30, 40), (160, 25, 32)]:
        for event, original_thumb in clip_thumbs:
            with Image.open(io.BytesIO(original_thumb)) as im:
                im.thumbnail((clip_width, clip_width), Image.Resampling.LANCZOS)
                buf = io.BytesIO()
                im.save(buf, format='WEBP', quality=quality, method=4)
            event['thumbnail'] = 'data:image/webp;base64,' + base64.b64encode(buf.getvalue()).decode()

        data['images'] = []
        for path in images:
            with Image.open(path) as source:
                im = source.convert('RGB')
                im.thumbnail((width, width), Image.Resampling.LANCZOS)
                buf = io.BytesIO()
                im.save(buf, format='WEBP', quality=quality, method=6)
                data['images'].append('data:image/webp;base64,' + base64.b64encode(buf.getvalue()).decode())
        payload = json.dumps(data, ensure_ascii=False, separators=(',', ':')).replace('<', chr(92) + 'u003c')
        fragment = template.replace('__QA_DATA__', payload)
        if len(fragment.encode()) < 1_000_000:
            break
    else:
        raise ValueError('Inline preview exceeds size limit')
    DEST.mkdir(parents=True, exist_ok=True)
    target = DEST / 'ultrasound-qa-demo.html'
    target.write_text(fragment)
    manifest = {
        'source': str(SOURCE.relative_to(ROOT)),
        'source_sha256': hashlib.sha256(SOURCE.read_bytes()).hexdigest(),
        'dataset_version': original['version'], 'counts': data['counts'],
        'unique_evidence_images': len(images), 'thumbnail_max_dimension': width,
        'thumbnail_webp_quality': quality, 'segment_thumbnail_max_dimension': clip_width, 'artifact_bytes': target.stat().st_size,
        'artifact_sha256': hashlib.sha256(target.read_bytes()).hexdigest(),
        'offline_video_package': 'results/qa_demo_10v_20261122/Ultrasound-QA-10-Examples.zip',
        'preview_contains_full_videos': False,
        'local_video_playback': 'Select original <video_id>.mp4 files; filename matching, browser-local blob URLs, no upload.',
    }
    (DEST / 'manifest.json').write_text(json.dumps(manifest, ensure_ascii=False, indent=2)+'\n')
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
