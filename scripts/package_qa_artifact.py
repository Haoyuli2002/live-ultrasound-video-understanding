#!/usr/bin/env python3
"""Remove unused renderer CDN helpers and package the standalone QA artifact."""
import hashlib
from html import escape
from html.parser import HTMLParser
import json
from pathlib import Path
import re
import zipfile

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'results/qa_demo_artifact_20261124'
UNUSED_SCRIPTS = {
    'https://unpkg.com/@floating-ui/core@1.7.3/dist/floating-ui.core.umd.min.js',
    'https://unpkg.com/@floating-ui/dom@1.7.4/dist/floating-ui.dom.umd.min.js',
    'https://unpkg.com/lucide@1.17.0/dist/umd/lucide.js',
}


class FrameParser(HTMLParser):
    document = None

    def handle_starttag(self, tag, attrs):
        attributes = dict(attrs)
        if tag == 'iframe' and ('srcdoc' in attributes or 'data-srcdoc' in attributes):
            assert self.document is None, 'Expected a single preview frame'
            self.document = attributes.get('srcdoc', attributes.get('data-srcdoc'))


def main():
    preview = OUT / 'preview.html'
    text = preview.read_text()
    parser = FrameParser()
    parser.feed(text)
    assert parser.document is not None
    def remove_unused(match):
        if match[1] not in UNUSED_SCRIPTS:
            raise ValueError('Unexpected external script: ' + match[1])
        return ''
    cleaned = re.sub(r'<script\b[^>]*\bsrc="([^"]+)"[^>]*>\s*</script>', remove_unused, parser.document)
    old = escape(parser.document, quote=True)
    assert old in text
    preview.write_text(text.replace(old, escape(cleaned, quote=True), 1))
    manifest = json.loads((OUT / 'manifest.json').read_text())
    manifest['standalone_sha256'] = hashlib.sha256(preview.read_bytes()).hexdigest()
    manifest['standalone_bytes'] = preview.stat().st_size
    manifest['standalone_external_scripts'] = []
    readme = (
        '超声 QA 数据 demo\n\n'
        '1. 解压后双击 index.html，无需 Python、服务器或网络。\n'
        '2. 可直接浏览 10 个视频的 286 条 QA、证据缩略图及监督标签。\n'
        '3. 播放视频：先解压完整的 Ultrasound-QA-10-Examples.zip，'
        '在页面“本机视频”中选择其 media 文件夹内的 MP4（可多选）。\n'
        '4. 视频只在本机读取，不上传；刷新页面后需重新选择。\n'
        '5. 支持 Query/Answer 跳转、区间自动暂停，以及证据时间跳转。\n\n'
        '本轻量包不包含完整视频；文件名应保持 <video_id>.mp4。\n'
        '展示的是 holdout 参考标注，不是模型预测。QA 保留英文原文。\n'
    )
    bundle = OUT / 'Ultrasound-QA-Artifact.zip'
    with zipfile.ZipFile(bundle, 'w', compression=zipfile.ZIP_DEFLATED) as archive:
        archive.write(preview, 'Ultrasound-QA-Artifact/index.html')
        archive.writestr('Ultrasound-QA-Artifact/README.txt', readme)
        archive.writestr('Ultrasound-QA-Artifact/manifest.json', json.dumps(manifest, ensure_ascii=False, indent=2))
    with zipfile.ZipFile(bundle) as archive:
        assert archive.testzip() is None
        assert archive.read('Ultrasound-QA-Artifact/index.html') == preview.read_bytes()
    manifest['zip_bytes'] = bundle.stat().st_size
    manifest['zip_sha256'] = hashlib.sha256(bundle.read_bytes()).hexdigest()
    (OUT / 'bundle_manifest.json').write_text(json.dumps(manifest, ensure_ascii=False, indent=2)+'\n')
    print(json.dumps({k:manifest[k] for k in ['standalone_bytes','zip_bytes','zip_sha256']}))


if __name__ == '__main__':
    main()
