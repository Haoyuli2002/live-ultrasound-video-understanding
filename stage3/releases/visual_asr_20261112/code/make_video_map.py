"""Resolve the release's video IDs against a training machine's video directory."""
import argparse,json
from pathlib import Path
p=argparse.ArgumentParser(description=__doc__)
p.add_argument('--video-root',type=Path,required=True)
p.add_argument('--output',type=Path,required=True)
p.add_argument('--package',type=Path,default=Path(__file__).resolve().parents[1])
a=p.parse_args();assets=json.loads((a.package/'media_manifest.json').read_text());index={}
for f in a.video_root.rglob('*'):
 if f.is_file() and f.suffix.lower() in {'.mp4','.mkv','.webm','.mov'} and f.stem in assets:
  index.setdefault(f.stem,set()).add(str(f.resolve()))
result={}
for vid in assets:
 matches=index.get(vid,set())
 if len(matches)!=1:raise ValueError(f'{vid}: expected one unique source file, found {len(matches)}. Use a directory with the intended source videos.')
 result[vid]=matches.pop()
if a.output.exists():raise FileExistsError('Use a new video map filename')
a.output.parent.mkdir(parents=True,exist_ok=True);a.output.write_text(json.dumps(result,indent=2)+'\n')
print(f'Wrote {len(result)} video paths to {a.output}')
