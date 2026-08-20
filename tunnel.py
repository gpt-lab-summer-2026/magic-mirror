"""
The public URL, and the QR code that carries it to a phone.

Shaped like parser_thread.py: available() then start(), the thread owned here.
cloudflared opens an outbound connection to Cloudflare, which hands it an https
hostname and forwards everything that arrives there back down to this laptop -
so nothing inbound reaches the machine and the server stays plain http on
localhost. No cloudflared installed means no tunnel, not a crash: the page is
still at http://127.0.0.1:8080 for a laptop.
"""
import atexit
import re
import shutil
import subprocess
import threading

import numpy as np
import qrcode

import config

# The tunnel's own hostname, and never the api.trycloudflare.com that
# cloudflared posts to in order to ask for one - that is the host its error line
# names when the request fails, and a looser pattern put it in the QR code.
TUNNEL_URL = re.compile(r"https://(?!api\.)[a-z0-9-]+\.trycloudflare\.com")

# Side of the code on the 540 px-wide cropped frame. _render floors this to
# whole pixels per module, so the steps are coarse: this lands on 2 px per
# module, which the rig doubles to 4 when it scales the frame to the screen.
QR_PX = 80
MARGIN = 16     # from the frame's top-left corner

_qr = None      # BGR array pasted on every frame; None until the tunnel is up


def available():
    return shutil.which("cloudflared") is not None


def start():
    """Launch cloudflared and wait for its URL on a thread, so a slow or absent
    connection delays the QR code and nothing else."""
    process = subprocess.Popen(
        ["cloudflared", "tunnel", "--url", f"http://localhost:{config.ANCHOR_APP_PORT}"],
        stderr=subprocess.PIPE, text=True)   # cloudflared logs to stderr
    # A child is not killed when its parent exits, on Windows or Linux: without
    # this every restart leaves one more tunnel forwarding to port 8080.
    atexit.register(process.terminate)
    threading.Thread(target=_wait_for_url, args=(process,), daemon=True).start()


def _wait_for_url(process):
    """`for line in stderr` waits until cloudflared prints, and with no internet
    it never does - which on the main thread would be main.py hanging before the
    window opens."""
    global _qr
    last_line = ""
    for line in process.stderr:
        found = TUNNEL_URL.search(line)
        if found:
            _qr = _render(found.group())
            print(f"anchor page: {found.group()}", flush=True)
            return
        last_line = line.strip()
    # stderr ended, so cloudflared is gone. Its last words are the only clue
    # anyone gets: a blank corner where the QR code should be says nothing.
    print(f"no tunnel, so no QR code: {last_line}", flush=True)


def _render(url):
    """The code as a BGR array, black modules on white, quiet zone included."""
    code = qrcode.QRCode(border=2)
    code.add_data(url)
    code.make()
    modules = np.array(code.get_matrix())
    # Whole pixels per module: a resize to exactly QR_PX would blur the edges
    # that a phone camera has to find.
    scale = QR_PX // len(modules)
    gray = np.where(np.kron(modules, np.ones((scale, scale), bool)), 0, 255).astype(np.uint8)
    return np.dstack([gray] * 3)


def draw(frame):
    """Paste the code in the top-left corner, in place. A no-op until the
    tunnel is up, and on a machine with no cloudflared it stays one."""
    if _qr is None:
        return
    h, w = _qr.shape[:2]
    frame[MARGIN:MARGIN + h, MARGIN:MARGIN + w] = _qr
