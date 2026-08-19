"""
The anchor page, served to one phone.

Shaped like telegram_bot.py: available() then start(), the thread owned here.
It binds to localhost only - a tunnel is the one way in, so a laptop on the
same lab Wi-Fi cannot reach it - and answers three requests and no others: the
page, and two reads that a live session's token unlocks. Nothing here builds a
path out of anything a request said, and nothing here writes.

Usage:
    python anchor_server.py garments/green_pants.png pants
"""
import json
import os
import sys
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlparse

from dotenv import load_dotenv

import anchor_session
import config
from garment_types import RIG_BY_CATEGORY

load_dotenv()
# Where the tunnel answers, which is the only address the phone can use. Read
# once, here: telegram_bot.py builds its button from this same string.
APP_URL = os.getenv("ANCHOR_APP_URL")

PAGE = Path(__file__).parent / "anchor_app" / "index.html"


def available():
    """No tunnel URL means no page worth serving - the bot then never offers it."""
    return bool(APP_URL)


def start():
    """Serve on a daemon thread, for telegram_bot.py's reason: `q` has to end the
    process even with a page open, and an unsaved draft is nothing to protect."""
    threading.Thread(target=_serve, daemon=True).start()


def _serve():
    ThreadingHTTPServer(("127.0.0.1", config.ANCHOR_APP_PORT), _Handler).serve_forever()


class _Handler(BaseHTTPRequestHandler):
    def do_GET(self):
        route = urlparse(self.path)
        if route.path == "/":
            self._send("text/html; charset=utf-8", PAGE.read_bytes())
            return

        session = anchor_session.get_session(parse_qs(route.query).get("t", [""])[0])
        if session is None:
            self._send_404()
        elif route.path == "/session":
            self._send("application/json", json.dumps(_describe(session)).encode())
        elif route.path == "/image":
            self._send("image/png", session["cutout"])
        else:
            self._send_404()

    def log_message(self, *args):
        """Silent: the render loop prints its frame timings to this same stdout."""

    def _send(self, content_type, body):
        self.send_response(200)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _send_404(self):
        """Empty on purpose: a wrong token learns nothing about a right one."""
        self.send_response(404)
        self.send_header("Content-Length", "0")
        self.end_headers()


def _describe(session):
    """Everything the page draws, so it carries no copy of the point names itself."""
    point_names, core = anchor_session.points_for(session["category"])
    return {
        "point_names": point_names,
        "core": core,
        "anchors": anchor_session.points_of(session["sidecar"]),
        "width": session["width"],
        "height": session["height"],
        "category": session["category"],
        "colors": anchor_session.web_colors(),
    }


if __name__ == "__main__":
    if len(sys.argv) != 3:
        sys.exit(f"Usage: python anchor_server.py cutout.png [{'|'.join(RIG_BY_CATEGORY)}]")

    cutout, category = Path(sys.argv[1]).read_bytes(), sys.argv[2]
    sidecar, _ = anchor_session.initial_sidecar(anchor_session.decode(cutout), category)
    token = anchor_session.new_session("test", cutout, sidecar, category)
    print(f"http://127.0.0.1:{config.ANCHOR_APP_PORT}/?t={token}", flush=True)
    try:
        _serve()
    except KeyboardInterrupt:
        pass
