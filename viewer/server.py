"""Local server for the match viewer: the page, its data cache, and the video with seeking.

    python -m viewer.server              # http://localhost:8765
    python -m viewer.server --rebuild    # rebuild viewer/cache first (after a model writes a new CSV)

Standard library only. The one thing the stock http.server lacks is HTTP Range requests, which a
browser needs to seek in a 5 GB video whose index (the "moov" box) sits at the end of the file.
"""

import argparse
import http.server
import os
import re
import sys

from possession.data import ROOT
from games import asset_paths

# Params
PORT = 8765
STATIC = os.path.join(ROOT, "viewer", "static")
CACHE = os.path.join(ROOT, "viewer", "cache")
CHUNK = 1 << 20  # bytes per write while streaming the video
TYPES = {".html": "text/html; charset=utf-8", ".js": "text/javascript; charset=utf-8",
         ".css": "text/css; charset=utf-8", ".json": "application/json", ".bin": "application/octet-stream",
         ".mp4": "video/mp4"}


class Handler(http.server.BaseHTTPRequestHandler):
    def do_GET(self):
        path = self.path.split("?")[0]
        if path == "/":
            path = "/index.html"
        if path == "/video.mp4":
            return self.send_file(asset_paths()["video"])
        folder, name = (CACHE, path[len("/cache/"):]) if path.startswith("/cache/") else (STATIC, path[1:])
        full = os.path.realpath(os.path.join(folder, name))
        if not full.startswith(os.path.realpath(folder) + os.sep) or not os.path.isfile(full):
            return self.send_error(404)
        self.send_file(full)

    def send_file(self, full):
        size = os.path.getsize(full)
        start, end = 0, size - 1
        match = re.match(r"bytes=(\d*)-(\d*)", self.headers.get("Range", ""))
        if match and (match.group(1) or match.group(2)):
            if match.group(1):
                start = int(match.group(1))
                end = int(match.group(2)) if match.group(2) else size - 1
            else:  # "bytes=-N": the last N bytes
                start = max(0, size - int(match.group(2)))
            end = min(end, size - 1)
            self.send_response(206)
            self.send_header("Content-Range", "bytes {}-{}/{}".format(start, end, size))
        else:
            self.send_response(200)
        self.send_header("Content-Type", TYPES.get(os.path.splitext(full)[1], "application/octet-stream"))
        self.send_header("Content-Length", str(end - start + 1))
        self.send_header("Accept-Ranges", "bytes")
        self.send_header("Cache-Control", "no-cache")
        self.end_headers()
        with open(full, "rb") as handle:
            handle.seek(start)
            remaining = end - start + 1
            try:
                while remaining > 0:
                    data = handle.read(min(CHUNK, remaining))
                    if not data:
                        break
                    self.wfile.write(data)
                    remaining -= len(data)
            except (BrokenPipeError, ConnectionResetError):
                pass  # the browser cancels range requests all the time while seeking

    def log_message(self, fmt, *args):
        if not self.path.startswith("/video.mp4"):
            sys.stderr.write("%s\n" % (fmt % args))


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--rebuild", action="store_true", help="rebuild viewer/cache before serving")
    parser.add_argument("--port", type=int, default=PORT)
    args = parser.parse_args()
    if args.rebuild or not os.path.exists(os.path.join(CACHE, "meta.json")):
        from viewer.build_cache import build
        build()
    server = http.server.ThreadingHTTPServer(("127.0.0.1", args.port), Handler)
    print("viewer on http://localhost:{}".format(args.port), flush=True)
    server.serve_forever()


if __name__ == "__main__":
    main()
