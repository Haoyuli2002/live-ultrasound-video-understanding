#!/usr/bin/env python3
"""Package only the new QA viewer, videos and cited frames for offline local use."""
import argparse
import copy
import hashlib
import json
import shutil
import zipfile
from pathlib import Path
from urllib.parse import unquote

REPO = Path(__file__).resolve().parents[1]
SOURCE = REPO / 'results/stage3_visual_dense_3v_v2_20261119'
DEFAULT_OUT = REPO / 'results/local_qa_demo_20261119'

BOOTSTRAP = '''let displayLanguage = 'zh';
try { displayLanguage = localStorage.getItem('ultrasound-demo-language') || 'zh'; } catch {}
const requestedLanguage = new URLSearchParams(location.search).get('lang');
if (['en', 'zh'].includes(requestedLanguage)) displayLanguage = requestedLanguage;
if (!['en', 'zh'].includes(displayLanguage)) displayLanguage = 'zh';
const languageSelect = document.getElementById('language-select');
languageSelect.value = displayLanguage;
document.documentElement.lang = displayLanguage === 'zh' ? 'zh-CN' : 'en';
languageSelect.addEventListener('change', () => {
  displayLanguage = languageSelect.value;
  document.documentElement.lang = displayLanguage === 'zh' ? 'zh-CN' : 'en';
  try { localStorage.setItem('ultrasound-demo-language', displayLanguage); } catch {}
});
'''
START = '''#!/bin/bash
set -eu
cd "$(dirname "$0")"
if ! command -v python3 >/dev/null 2>&1; then
  echo "需要 Python 3。请先安装 Python 3，再双击 Start.command。"
  echo "Python 3 is required. Install it, then open Start.command again."
  read -r -p "按回车退出 / Press Enter to close..." answer
  exit 1
fi
if ! python3 serve.py; then
  echo "启动失败。请保留上面的报错信息。/ Startup failed; see the error above."
  read -r -p "按回车退出 / Press Enter to close..." answer
  exit 1
fi
'''
SERVE = '''"""Local-only offline viewer. Python 3 standard library; no pip dependencies."""
import argparse
import errno
import functools
import webbrowser
from http.server import ThreadingHTTPServer
from pathlib import Path
from range_server import RangeRequestHandler

ROOT = Path(__file__).resolve().parent

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--port', type=int, default=8765)
    parser.add_argument('--no-open', action='store_true')
    args = parser.parse_args()
    handler = functools.partial(RangeRequestHandler, directory=str(ROOT))
    try:
        server = ThreadingHTTPServer(('127.0.0.1', args.port), handler)
    except OSError as error:
        if error.errno != errno.EADDRINUSE:
            raise
        server = ThreadingHTTPServer(('127.0.0.1', 0), handler)
    address = 'http://127.0.0.1:%d/?lang=zh#new-version-qa' % server.server_port
    print('Local QA demo / 本地 QA 演示：' + address, flush=True)
    print('Keep this terminal open. Ctrl+C stops the demo. / 保持此终端打开，按 Ctrl+C 停止。', flush=True)
    if not args.no_open:
        webbrowser.open(address)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()

if __name__ == '__main__':
    main()
'''
README = '''# Ultrasound QA Demo — macOS 本地版

包含 3 个完整视频、72 条英文 QA、中文阅读译文、94 条 WAIT/ANSWER 决策样本及 92 张引用的证据帧。视频和帧都随包附带，无需访问远程仓库、SSH 隧道或模型 API。

## 启动

1. 解压 Ultrasound-QA-Demo.zip，得到 Ultrasound-QA-Demo 文件夹。
2. 双击文件夹内的 Start.command。需要本机已安装 Python 3；无 pip/npm 依赖。
3. 浏览器会自动打开 http://127.0.0.1:8765/?lang=zh#new-version-qa 。如该端口已占用，会自动选择其他空闲端口，实际地址显示在终端中。
4. 保持弹出的终端窗口打开；按 Ctrl+C 停止服务。此时运行的是 Mac 本机服务，关闭 SSH 不影响它。

若浏览器没有自动打开，请复制终端打印的地址。不要通过双击 index.html 直接用 file:// 打开，因为浏览器会限制网页读取 JSON 数据。

也可以在此文件夹的终端运行：

    python3 serve.py

只启动服务、不自动打开浏览器：

    python3 serve.py --no-open

## 数据与语言

默认显示英文原文和中文译文，可切换为纯英文。问答、答案、时间和决策标签沿用已生成版本；打包不代表新增专家复核。证据仍属于抽帧自审，触发时刻为一秒采样精度。

train_tasks.jsonl、train_samples.jsonl、qa_bilingual_en_zh.jsonl 是原始导出文件。用于网页的 new-qa.json 仅把媒体链接改为包内相对路径。QA_BILINGUAL_en_zh.md 中的原视频链接也已改为本地 media/ 路径。

package_manifest.json 和 SHA256SUMS 记录来源和文件校验和。打包过程未改动远程训练数据。不要把这些已有训练视频的问答拆分成独立评测集。

## 移动文件夹

整个文件夹可以放到 Downloads、Desktop 或其他位置，路径含空格也可以。移动时请保留 media/ 和 frames/ 等所有子目录。
'''

def digest(path):
    h = hashlib.sha256()
    with path.open('rb') as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()

def build(output):
    output.mkdir(parents=True, exist_ok=True)
    package = output / 'Ultrasound-QA-Demo'
    if package.exists():
        raise FileExistsError(f'Refusing to overwrite an existing package: {package}')
    package.mkdir()
    data = copy.deepcopy(json.loads((REPO / 'viz/new-qa.json').read_text()))
    copied_frames = set()
    for video in data['videos']:
        original_video = (REPO / 'viz' / unquote(video['url'])).resolve()
        relative = Path('media') / (video['video_id'] + '.mp4')
        (package / relative).parent.mkdir(exist_ok=True)
        shutil.copy2(original_video, package / relative)
        video['url'] = relative.as_posix()
        for task in video['tasks']:
            for frame in task['evidence']:
                original = (REPO / 'viz' / unquote(frame['url'])).resolve()
                relative = original.relative_to(SOURCE.resolve())
                assert relative.parts[0] == 'frames'
                destination = package / relative
                if relative.as_posix() not in copied_frames:
                    destination.parent.mkdir(parents=True, exist_ok=True)
                    shutil.copy2(original, destination)
                    copied_frames.add(relative.as_posix())
                frame['url'] = relative.as_posix()
        video['poster'] = video['tasks'][0]['evidence'][0]['url']
    data.update(review_url='QA_BILINGUAL_en_zh.md', tasks_url='train_tasks.jsonl', samples_url='train_samples.jsonl')
    (package / 'new-qa.json').write_text(json.dumps(data, ensure_ascii=False, indent=2) + '\n')
    for name in ['new-qa.js', 'new-qa.css', 'style.css']:
        shutil.copy2(REPO / 'viz' / name, package / name)
    shutil.copy2(REPO / 'scripts/viz_serve.py', package / 'range_server.py')
    for name in ['train_tasks.jsonl', 'train_samples.jsonl', 'qa_bilingual_en_zh.jsonl', 'validation_report.json', 'bilingual_validation.json']:
        shutil.copy2(SOURCE / name, package / name)
    review = (SOURCE / 'QA_BILINGUAL_en_zh.md').read_text()
    mapping = json.loads((SOURCE / 'video_map.json').read_text())
    for vid, path in mapping.items():
        review = review.replace(path, 'media/' + vid + '.mp4')
    (package / 'QA_BILINGUAL_en_zh.md').write_text(review)
    page = (REPO / 'viz/index.html').read_text()
    section = page.split('<section id="new-version-qa"', 1)[1].split('</section>', 1)[0]
    html = '''<!DOCTYPE html>
<html lang="zh-CN"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1">
<title>Ultrasound · New QA</title><link rel="stylesheet" href="style.css"><link rel="stylesheet" href="new-qa.css"></head>
<body><header><label class="language-control">Language / 语言 <select id="language-select"><option value="en">English</option><option value="zh">中文</option></select></label>
<p class="eyebrow">ULTRASOUND · LOCAL QA DEMO</p><h1>视频问答 / Video QA</h1><p class="nqa-jump"><a id="nqa-jump" href="#new-version-qa">新版本 QA ↓</a></p></header>
<section id="new-version-qa"''' + section + '''</section>
<script src="local-language.js"></script><script src="new-qa.js"></script></body></html>
'''
    (package / 'index.html').write_text(html)
    (package / 'local-language.js').write_text(BOOTSTRAP)
    (package / 'serve.py').write_text(SERVE)
    (package / 'Start.command').write_text(START)
    (package / 'Start.command').chmod(0o755)
    (package / 'README_zh.md').write_text(README)
    manifest = {'format': 'ultrasound_qa_offline_demo_v1', 'source_version': data['version'], 'counts': data['counts'],
                'bundled_evidence_frames': len(copied_frames), 'runtime': 'Python 3 standard library',
                'offline_after_download': True, 'source_viewer_sha256': digest(REPO / 'viz/new-qa.json'),
                'source_script_sha256': digest(Path(__file__)), 'files': {}}
    for path in sorted(package.rglob('*')):
        if path.is_file():
            manifest['files'][path.relative_to(package).as_posix()] = {'bytes': path.stat().st_size, 'sha256': digest(path)}
    (package / 'package_manifest.json').write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + '\n')
    (package / 'SHA256SUMS').write_text(''.join(f'{digest(path)}  {path.relative_to(package)}\n' for path in sorted(package.rglob('*')) if path.is_file()))
    archive = output / 'Ultrasound-QA-Demo.zip'
    with zipfile.ZipFile(archive, 'w', compression=zipfile.ZIP_DEFLATED, compresslevel=6) as z:
        for path in sorted(package.rglob('*')):
            if path.is_file():
                z.write(path, path.relative_to(output), compress_type=zipfile.ZIP_STORED if path.suffix == '.mp4' else zipfile.ZIP_DEFLATED)
    summary = {'archive': str(archive), 'archive_sha256': digest(archive), 'archive_bytes': archive.stat().st_size, **data['counts'], 'evidence_frames': len(copied_frames)}
    (output / 'package_summary.json').write_text(json.dumps(summary, indent=2) + '\n')
    print(json.dumps(summary, indent=2))

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=DEFAULT_OUT)
    build(parser.parse_args().output.absolute())
