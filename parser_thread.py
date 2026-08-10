"""
The human parser, running beside the render loop instead of inside it.

A parse costs far more than a frame, so the loop hands over a frame and reads
back whichever class map has finished. Everything that touches the torch model
lives here; the loop only ever calls submit() and latest().
"""
import threading
import time

import debug_hud
import human_parser

_model = None
_processor = None

_latest = (None, 0)     # (class map, timestamp of the frame it describes)
_latest_lock = threading.Lock()
_pending = None         # (frame, timestamp) waiting for the worker
_pending_lock = threading.Lock()


def available():
    """GPU or not at all: ~1.5 s per frame on CPU, which starves the loop."""
    return human_parser.gpu_available()


def start():
    """Load the model, run the worker, and return the class name -> index map."""
    global _model, _processor
    _model, _processor = human_parser.load_parser()
    labels = human_parser.class_indices(_model)
    print(f"Parser on {_model.device}, classes: {', '.join(sorted(labels))}", flush=True)

    # daemon: `q` must end the process even if the parser is mid-inference. The
    # worker holds no file or socket and its class map is throwaway, so there is
    # nothing a clean shutdown would protect - and a join() on a wedged torch
    # call is exactly the kiosk that needs Ctrl-C.
    threading.Thread(target=_worker, daemon=True).start()
    return labels


def submit(frame, stamp):
    """Latest frame wins: a frame handed over while the worker is busy replaces
    the pending one rather than joining a queue, so the map is always the
    freshest the parser could have finished - never a backlog of stale ones."""
    global _pending
    with _pending_lock:
        _pending = (frame, stamp)


def latest():
    """(class map, timestamp), or (None, 0) before the first parse finishes."""
    with _latest_lock:
        return _latest


def _worker():
    """The only thread that touches the torch model."""
    global _latest, _pending
    while True:
        with _pending_lock:
            pending, _pending = _pending, None
        if pending is None:
            time.sleep(0.005)   # nothing pending; spinning here would cost a core
            continue

        frame, stamp = pending
        started = time.perf_counter()
        class_map = human_parser.parse(_model, _processor, frame)
        debug_hud.mark("parser", started)
        with _latest_lock:
            _latest = (class_map, stamp)
