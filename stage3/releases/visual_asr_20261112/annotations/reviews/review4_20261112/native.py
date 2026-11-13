import json,re,subprocess,sys
from pathlib import Path
here=Path(__file__).resolve().parent
rows={r['video_id']:r for r in map(json.loads,(here/'original_tasks.jsonl').read_text().splitlines())}
v=sys.argv[1]
for value in sys.argv[2:]:
 t=float(value);d=here/'frames'/v;d.mkdir(parents=True,exist_ok=True);p=d/f'{t:g}.jpg'
 if p.with_suffix('.json').exists():continue
 r=subprocess.run(['ffmpeg','-hide_banner','-ss',str(max(0,t-.08)),'-copyts','-i',rows[v]['source_video'],'-vf','showinfo','-frames:v','1','-q:v','2','-y',str(p)],check=True,capture_output=True,text=True)
 pts,w,h=re.search(r'\bn:\s*0\s+pts:.*?pts_time:([\d.]+).*?\bs:(\d+)x(\d+)',r.stderr).groups()
 assert float(pts)<t
 p.with_suffix('.json').write_text(json.dumps(dict(path=str(p),checkpoint=t,source_pts_seconds=float(pts),resolution=[int(w),int(h)]))+'\n')
