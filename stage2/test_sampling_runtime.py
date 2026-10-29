"""Regression checks for the reported long-prefix and local-service failures."""
import json
import os
from pathlib import Path
import shutil
import socket
import subprocess
import tempfile
import threading
import unittest
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

from stage2.annotate import make_silent_clip
from stage2.runtime import available_port, endpoint_ready
from stage2.sampling import clip_metadata, sampling_plan


class RuntimeTests(unittest.TestCase):
    def test_busy_explicit_port_rejected_and_auto_port_available(self):
        with socket.socket() as listener:
            listener.bind(('127.0.0.1',0));listener.listen()
            occupied=listener.getsockname()[1]
            with self.assertRaises(OSError): available_port(str(occupied))
            self.assertNotEqual(available_port(),occupied)

    def test_ready_requires_own_model_and_authentication(self):
        class Handler(BaseHTTPRequestHandler):
            def do_GET(self):
                if self.headers.get('Authorization')!='Bearer own-secret':
                    self.send_response(401);self.end_headers();return
                body=json.dumps({'data':[{'id':'own-job-model'}]}).encode()
                self.send_response(200);self.end_headers();self.wfile.write(body)
            def log_message(self,*args): pass
        server=ThreadingHTTPServer(('127.0.0.1',0),Handler)
        thread=threading.Thread(target=server.serve_forever,daemon=True);thread.start()
        base=f'http://127.0.0.1:{server.server_port}/v1'
        try:
            self.assertTrue(endpoint_ready(base,'own-job-model','own-secret'))
            self.assertFalse(endpoint_ready(base,'other-job-model','own-secret'))
            self.assertFalse(endpoint_ready(base,'own-job-model','other-secret'))
        finally:
            server.shutdown();server.server_close();thread.join()


@unittest.skipUnless(shutil.which('ffmpeg') and shutil.which('ffprobe'),'ffmpeg/ffprobe required')
class ClipSamplingTests(unittest.TestCase):
    def test_short_and_long_prefixes_keep_duration_audio_removed_and_frame_cap(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);source=root/'source.mp4'
            subprocess.run(['ffmpeg','-v','error','-f','lavfi','-i',
                'color=c=blue:s=160x96:r=2:d=1080','-f','lavfi','-i',
                'anullsrc=r=8000:cl=mono','-t','1080','-c:v','libx264','-preset','ultrafast',
                '-c:a','aac','-shortest',str(source)],check=True)
            self.assertEqual(clip_metadata(source)['audio_streams'],1)
            for end,frames in [(10,20),(60,120),(420,120),(1080,120)]:
                with self.subTest(end=end):
                    clip=root/f'{end}.mp4'
                    make_silent_clip(source,0,end,clip,max_side=160,fps=2,max_frames=120)
                    observed=clip_metadata(clip)
                    self.assertEqual(observed['frames'],frames)
                    self.assertEqual(observed['audio_streams'],0)
                    self.assertAlmostEqual(observed['duration'],end,places=2)
                    self.assertAlmostEqual(observed['fps'],frames/end,places=5)
            self.assertEqual(sampling_plan(0,420,2,120)['fps_filter'],'2/7')


if __name__=='__main__': unittest.main()
