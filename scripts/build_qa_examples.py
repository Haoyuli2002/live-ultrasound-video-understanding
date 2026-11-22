#!/usr/bin/env python3
"""Build ten review examples from the frozen holdout QA; optionally package offline."""
import argparse, collections, hashlib, json, shutil, zipfile
from pathlib import Path
from urllib.parse import quote
REPO=Path(__file__).resolve().parents[1]
SOURCE=REPO/'results/stage3_visual_v4_holdout16_20261121'
EXAMPLES=[
 ('CW3TLabogko','心脏切面','Cardiac views'),
 ('B2USlWmqOV0','肺部与声学伪影','Lung and acoustic artifacts'),
 ('ODF8ZLr_52s','腹部扫查','Abdominal scanning'),
 ('lREUPXCpK8Y','血管与彩色多普勒','Vessels and color Doppler'),
 ('7G56DN38mz8','肌骨与血流信号','Musculoskeletal imaging'),
 ('0Cboqf1Qnhc','软组织与腔隙','Soft tissue and lumens'),
 ('qYM2sft8R2I','神经阻滞超声画面','Nerve-block imaging'),
 ('YllLJNPnpjY','血管通路教学','Vascular access'),
 ('v9ar1mhpT-g','暗区与深部组织','Dark spaces and deeper tissue'),
 ('az6h_u_8nhA','床旁超声与长时等待','Bedside ultrasound')]
def digest(p):
 h=hashlib.sha256()
 with p.open('rb') as f:
  for chunk in iter(lambda:f.read(8*1024*1024),b''):h.update(chunk)
 return h.hexdigest()
def readrows(p):return [json.loads(x) for x in p.read_text().splitlines() if x.strip()]
def build(package=None):
 tasks=readrows(SOURCE/'holdout_tasks.jsonl');samples=readrows(SOURCE/'holdout_samples.jsonl')
 selection={r['video_id']:r for r in json.loads((SOURCE/'selection.json').read_text())['videos']}
 mapping=json.loads((SOURCE/'video_map.json').read_text());decisions=collections.defaultdict(list)
 for s in samples:decisions[s['task_id']].append(s)
 out=REPO/'viz/qa-examples';out.mkdir(exist_ok=True)
 if package:
  out=package/'Ultrasound-QA-10-Examples';out.mkdir(parents=True,exist_ok=False)
  for name in ['index.html','app.js','style.css','qa-ui.ttf','OFL.txt']:shutil.copyfile(REPO/'viz/qa-examples'/name,out/name)
  (out/'media').mkdir();(out/'frames').mkdir()
 def frame_url(e):
  src=SOURCE/e['path'];assert src.is_file() and digest(src)==e['sha256']
  if package:
   dest=out/'frames'/Path(e['path']).parent.name/Path(e['path']).name;dest.parent.mkdir(exist_ok=True)
   if not dest.exists():shutil.copyfile(src,dest)
   return dest.relative_to(out).as_posix()
  return '../../results/'+SOURCE.name+'/'+quote(e['path'],safe='/')
 videos=[]
 for vid,title,title_en in EXAMPLES:
  src=Path(mapping[vid]);local=REPO/'cluster_data/videos/train_full295'/f'{vid}.mp4'
  assert src.is_file() and local.resolve()==src.resolve()
  if package:shutil.copyfile(src,out/'media'/src.name)
  entries=[]
  for t in [t for t in tasks if t['video_id']==vid]:
   assert t['split']=='holdout' and t['language']=='en'
   q,a=t['query_time'],t['answer_time'];assert 0<=q<=a<selection[vid]['duration']
   ds=[]
   for s in sorted(decisions[t['task_id']],key=lambda s:s['video_window'][1]):
    time=s['video_window'][1];label='ANSWER' if time>=a else 'WAIT'
    assert s['question']==t['question'] and s['target']==('<ANSWER> '+t['reference_answer'] if label=='ANSWER' else '<WAIT>')
    ds.append({'time':time,'label':label})
   assert [d['time'] for d in ds]==t['evaluation_times']
   entries.append({**{k:t[k] for k in ['task_id','qa_type','question','reference_answer','query_time','answer_time']},'evidence':[{'time':e['time'],'url':frame_url(e)} for e in t['evidence']],'decisions':ds})
  assert entries
  featured=next((t for t in entries if t['qa_type']=='event_notification' and t['answer_time']-t['query_time']>=10),entries[0])
  videos.append({'id':vid,'title':title,'title_en':title_en,'duration':selection[vid]['duration'],'url':'media/'+src.name if package else '../../cluster_data/videos/train_full295/'+src.name,'poster':next(t['evidence'][0]['url'] for t in entries if t['qa_type']=='current_observation'),'tasks':entries,'featured_task':featured['task_id']})
 types=collections.Counter(t['qa_type'] for v in videos for t in v['tasks'])
 data={'version':SOURCE.name,'split':'holdout','counts':{'videos':len(videos),'qa':sum(types.values()),'types':types},'source_sha256':{n:digest(SOURCE/n) for n in ['holdout_tasks.jsonl','holdout_samples.jsonl','selection.json']},'videos':videos}
 (out/'data.json').write_text(json.dumps(data,ensure_ascii=False,indent=2)+'\n')
 if package:
  shutil.copyfile(REPO/'scripts/viz_serve.py',out/'range_server.py')
  shutil.copyfile(REPO/'scripts/serve_qa_examples.py',out/'serve.py')
  (out/'Start.command').write_text('#!/bin/bash\ncd "$(dirname "$0")"\npython3 serve.py\n');(out/'Start.command').chmod(0o755)
  (out/'README.txt').write_text('超声 QA：10 视频离线演示\n\n在此文件夹运行 python3 serve.py，或 macOS 双击 Start.command。需要 Python 3，无 pip/npm 依赖。\n浏览器自动打开终端显示的地址。保持终端运行，Ctrl+C 停止。请不要直接双击 index.html。\n视频已包含在 media/ 中，复制整个文件夹后不需要 SSH 或网络。\nQA 保持英文原文，来自 holdout 自审标注，非模型预测或专家审核。\n')
  (out/'SHA256SUMS').write_text(''.join(f'{digest(p)}  {p.relative_to(out)}\n' for p in sorted(out.rglob('*')) if p.is_file()))
  archive=package/'Ultrasound-QA-10-Examples.zip'
  with zipfile.ZipFile(archive,'w',zipfile.ZIP_STORED) as z:
   for p in sorted(out.rglob('*')):
    if p.is_file():z.write(p,p.relative_to(package))
  print('Archive:',archive,archive.stat().st_size)
 print('Built',out,json.dumps(data['counts']))
if __name__=='__main__':
 p=argparse.ArgumentParser(description=__doc__);p.add_argument('--package',type=Path);a=p.parse_args();build(a.package)
