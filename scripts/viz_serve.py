#!/usr/bin/env python3
"""Static file server for the Stage 1 visualizer WITH HTTP Range support.

Python's stock ``python -m http.server`` uses SimpleHTTPRequestHandler, which
ignores the ``Range`` request header and always returns the whole file with a
``200``. Browsers rely on ``206 Partial Content`` responses to seek inside a
``<video>``; without Range support the progress bar cannot be scrubbed and
programmatic seeks (e.g. jumping to a selected sentence) silently fail.

This handler adds minimal, correct single-range support so the viz video can be
scrubbed and seeked. Serve the REPO ROOT so viz/ and the video share one root:

    python scripts/viz_serve.py                 # port 8000
    python scripts/viz_serve.py --port 8080
    python scripts/viz_serve.py --directory .   # explicit root

Then open http://localhost:<port>/viz/index.html
"""
from __future__ import annotations

import argparse
import os
import re
from functools import partial
from http.server import HTTPServer, SimpleHTTPRequestHandler

_RANGE_RE = re.compile(r"bytes=(\d*)-(\d*)")


class RangeRequestHandler(SimpleHTTPRequestHandler):
    """SimpleHTTPRequestHandler that honours a single HTTP Range request."""

    def send_head(self):  # noqa: C901 - mirrors the stdlib shape intentionally
        range_header = self.headers.get("Range")
        if range_header is None:
            return super().send_head()

        path = self.translate_path(self.path)
        if os.path.isdir(path):
            # Directory listings never need range handling; defer to the base.
            return super().send_head()
        if not os.path.exists(path):
            self.send_error(404, "File not found")
            return None

        match = _RANGE_RE.fullmatch(range_header.strip())
        if match is None:
            self.send_error(400, "Invalid Range header")
            return None

        file_size = os.path.getsize(path)
        start_text, end_text = match.group(1), match.group(2)
        if start_text == "" and end_text == "":
            self.send_error(400, "Invalid Range header")
            return None
        if start_text == "":
            # Suffix range: last N bytes.
            length = int(end_text)
            start = max(0, file_size - length)
            end = file_size - 1
        else:
            start = int(start_text)
            end = int(end_text) if end_text else file_size - 1
        end = min(end, file_size - 1)
        if start > end or start >= file_size:
            self.send_response(416, "Requested Range Not Satisfiable")
            self.send_header("Content-Range", f"bytes */{file_size}")
            self.end_headers()
            return None

        try:
            handle = open(path, "rb")
        except OSError:
            self.send_error(404, "File not found")
            return None

        handle.seek(start)
        self.send_response(206, "Partial Content")
        ctype = self.guess_type(path)
        self.send_header("Content-type", ctype)
        self.send_header("Accept-Ranges", "bytes")
        self.send_header("Content-Range", f"bytes {start}-{end}/{file_size}")
        self.send_header("Content-Length", str(end - start + 1))
        self.end_headers()
        self._range = (handle, start, end)
        return handle

    def copyfile(self, source, outputfile):
        # When a range was negotiated, stream only the requested window.
        window = getattr(self, "_range", None)
        if window is None:
            super().copyfile(source, outputfile)
            return
        handle, start, end = window
        remaining = end - start + 1
        chunk = 64 * 1024
        while remaining > 0:
            data = handle.read(min(chunk, remaining))
            if not data:
                break
            outputfile.write(data)
            remaining -= len(data)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--directory", default=".",
                        help="Web root to serve (default: current directory).")
    args = parser.parse_args()

    handler = partial(RangeRequestHandler, directory=args.directory)
    httpd = HTTPServer(("127.0.0.1", args.port), handler)
    print(f"Serving {os.path.abspath(args.directory)} with Range support on "
          f"http://127.0.0.1:{args.port}/", flush=True)
    print(f"Open http://localhost:{args.port}/viz/index.html", flush=True)
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        httpd.server_close()


if __name__ == "__main__":
    main()
