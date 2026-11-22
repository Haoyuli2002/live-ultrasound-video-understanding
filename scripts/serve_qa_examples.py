#!/usr/bin/env python3
"""Run the QA examples with video seeking; no third-party dependencies."""
import argparse,functools,sys,webbrowser
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
 server=ThreadingHTTPServer(('127.0.0.1',a.port),functools.partial(RangeRequestHandler,directory=str(ROOT)))
 url=f'http://127.0.0.1:{server.server_port}{ENTRY}'
 print(url,flush=True);print('Keep this terminal open; Ctrl+C stops the demo.',flush=True)
 if not a.no_open:webbrowser.open(url)
 try:server.serve_forever()
 except KeyboardInterrupt:pass
 finally:server.server_close()
if __name__=='__main__':main()
