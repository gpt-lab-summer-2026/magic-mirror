"""
The phone's pages, served to whoever knows the password.

Shaped like parser_thread.py: available() then start(), the thread owned here.
It binds to localhost only - a tunnel is the one way in, so a laptop on the
same lab Wi-Fi cannot reach it - and answers the password box, the mapping
page, the two reads that page needs and the wear that puts the garment on the
mirror. Nothing here builds a path out of anything a request said.

Usage:
    python anchor_server.py garments/green_pants.png pants
"""
import hmac
import json
import os
import secrets
import sys
import threading
import time
from http.cookies import SimpleCookie
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import urlparse

from dotenv import load_dotenv

import anchor_session
import config
import garment_publish
from garment_types import RIG_BY_CATEGORY

load_dotenv()
# Where the tunnel answers, which is the only address the phone can use. Read
# once, here: telegram_bot.py builds its button from this same string.
APP_URL = os.getenv("ANCHOR_APP_URL")
PASSWORD = os.getenv("GARMENT_BOT_PASSWORD")

PAGE_DIR = Path(__file__).parent / "anchor_app"

COOKIE = "mirror"
MAX_BODY = 20 * 1024 * 1024   # a phone photo, with room to spare

_labels = None   # parser class names, for the occluder LUT finish() builds

# Every cookie value handed out since startup, so a restart logs each phone out.
_keys = set()

# One request at a time changes the mirror's state. Until now the bot's single
# event loop made that true by accident; ThreadingHTTPServer gives every request
# its own thread, so two presses could both build a garment at once.
_lock = threading.Lock()

# One guess at a time, for the whole server. Without it the sleep below only
# slows the thread that took that guess, and fifty connections are fifty guesses
# a second. Its own lock, so a guessing phone never stalls a wearing one.
_login_lock = threading.Lock()


def available():
    """No password means no page at all: it is the only thing between a room
    that can see the QR code and the mirror."""
    return bool(APP_URL and PASSWORD)


def start(labels):
    """Serve on a daemon thread, for parser_thread.py's reason: `q` has to end the
    process even with a page open, and an unsaved draft is nothing to protect."""
    global _labels
    _labels = labels
    threading.Thread(target=_serve, daemon=True).start()


def _serve():
    ThreadingHTTPServer(("127.0.0.1", config.ANCHOR_APP_PORT), _Handler).serve_forever()


class _Handler(BaseHTTPRequestHandler):
    def do_GET(self):
        route = urlparse(self.path).path
        if route == "/":
            self._send_page("index.html")
            return
        if not self._authorized():
            self._send_empty(403)
            return
        if route == "/map":
            self._send_page("map.html")
            return

        session = anchor_session.get_session()
        if session is None:
            self._send_empty(404)
        elif route == "/session":
            self._send("application/json", json.dumps(_describe(session)).encode())
        elif route == "/image":
            self._send("image/png", session["cutout"])
        else:
            self._send_empty(404)

    def do_POST(self):
        route = urlparse(self.path).path
        # Cookie first, body last: an unauthorized request then costs the server
        # nothing. The other way round, anyone who scanned the QR code could make
        # the laptop swallow 20 MB per request and only then hear 403.
        if route != "/login" and not self._authorized():
            self._send_empty(403)
            return

        length = int(self.headers.get("Content-Length", 0))
        if length > MAX_BODY:
            self._send_text(f"that is over the {MAX_BODY // 1024 // 1024} MB cap", 413)
            return
        body = self.rfile.read(length)

        if route == "/login":
            self._login(body)
        elif route == "/wear":
            with _lock:
                self._wear(body)
        else:
            self._send_empty(404)

    def _authorized(self):
        """The login cookie, or nothing. There is no second way past this line."""
        cookie = SimpleCookie(self.headers.get("Cookie", ""))
        return COOKIE in cookie and cookie[COOKIE].value in _keys

    def _login(self, body):
        """The password, compared in constant time - a plain == leaks how many
        leading characters matched through how long it took to say no."""
        with _login_lock:
            if not hmac.compare_digest(body, PASSWORD.encode()):
                time.sleep(1)
                self._send_empty(403)
                return
            key = secrets.token_urlsafe(24)
            _keys.add(key)

        # Secure: never sent over plain http, and browsers make 127.0.0.1 the
        # exception, so a laptop can still test. HttpOnly: no script can read it.
        # Strict: no other site can make the phone POST /wear behind its back.
        self._send("text/plain; charset=utf-8", b"", 200,
                   cookie=f"{COOKIE}={key}; Path=/; Secure; HttpOnly; SameSite=Strict")

    def _wear(self, body):
        """The anchors the page ended up with, onto the mirror. A bad point is 400
        and the session lives on - fix it, press again."""
        session = anchor_session.get_session()
        if session is None:
            self._send_empty(404)
            return

        try:
            sidecar = anchor_session.validate(body.decode(), session["sidecar"],
                                              session["category"], session["width"], session["height"])
            text, status = garment_publish.finish(session["cutout"], sidecar,
                                                  session["category"], _labels), 200
        except ValueError as e:
            text, status = str(e), 400
        self._send_text(text, status)

    def log_message(self, *args):
        """Silent: the render loop prints its frame timings to this same stdout,
        and a password must never reach a log even by accident."""

    def _send_page(self, name):
        self._send("text/html; charset=utf-8", (PAGE_DIR / name).read_bytes())

    def _send_text(self, text, status=200):
        self._send("text/plain; charset=utf-8", text.encode(), status)

    def _send(self, content_type, body, status=200, cookie=None):
        self.send_response(status)
        if cookie:
            self.send_header("Set-Cookie", cookie)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _send_empty(self, status):
        """No body on a refusal: a wrong guess learns nothing about a right one."""
        self.send_response(status)
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
    anchor_session.new_session(None, cutout, sidecar, category)
    print(f"http://127.0.0.1:{config.ANCHOR_APP_PORT}/ - password first, then the points", flush=True)
    try:
        _serve()
    except KeyboardInterrupt:
        pass
