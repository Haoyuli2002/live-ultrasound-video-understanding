#!/usr/bin/env python3
"""Run the QA examples with video seeking; no third-party dependencies."""
import argparse,functools,json,sys,webbrowser
from http.server import ThreadingHTTPServer
from pathlib import Path
try:
 from range_server import RangeRequestHandler
 ROOT=Path(__file__).resolve().parent
 ENTRY='/'
except ImportError:
 from viz_serve import RangeRequestHandler
 ROOT=Path(__file__).resolve().parents[1]
 ENTRY='/viz/qa-examples/'
def main():
 p=argparse.ArgumentParser(description=__doc__);p.add_argument('--port',type=int,default=8767);p.add_argument('--no-open',action='store_true');a=p.parse_args()
 catalog=json.loads((ROOT/'data.json').read_text())
 missing=[]
 for video in catalog['videos']:
  media=ROOT/video['url']
  if not media.is_file():missing.append(video['id']);continue
  with media.open('rb') as stream:
   if stream.read(128).startswith(b'version https://git-lfs.github.com/spec/v1'):missing.append(video['id'])
 if missing:
  print('Missing Git LFS videos: '+', '.join(missing),file=sys.stderr)
  print('From the repository root, run:\n  git lfs install\n  git lfs pull --include="demo/ultrasound-qa/media/*.mp4" --exclude=""',file=sys.stderr)
  return 1
 server=ThreadingHTTPServer(('127.0.0.1',a.port),functools.partial(RangeRequestHandler,directory=str(ROOT)))
 url=f'http://127.0.0.1:{server.server_port}{ENTRY}'
 print(url,flush=True);print('Keep this terminal open; Ctrl+C stops the demo.',flush=True)
 if not a.no_open:webbrowser.open(url)
 try:server.serve_forever()
 except KeyboardInterrupt:pass
 finally:server.server_close()
if __name__=='__main__':sys.exit(main())
