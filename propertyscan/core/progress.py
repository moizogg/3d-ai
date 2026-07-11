"""Live progress heartbeats so long stages never look frozen.

Prints to stdout every ``interval_s`` seconds while a block is running.
"""

from __future__ import annotations

import sys
import threading
import time
from typing import Callable


class ProgressHeartbeat:
    """Background ticker: ``[PROGRESS] <label> still running… elapsed=Xs``.

    Usage::

        with ProgressHeartbeat("mast3r_inference", interval_s=10) as hb:
            hb.set_status("loading weights")
            ...
            hb.set_status("global alignment")
    """

    def __init__(
        self,
        label: str,
        *,
        interval_s: float = 10.0,
        print_fn: Callable[[str], None] | None = None,
    ) -> None:
        self.label = label
        self.interval_s = max(1.0, float(interval_s))
        self._print = print_fn or _default_print
        self._status = "working"
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self._t0 = 0.0
        self._tick = 0

    def set_status(self, status: str) -> None:
        """Update the detail line shown on the next tick (and print immediately)."""
        self._status = status
        elapsed = time.perf_counter() - self._t0 if self._t0 else 0.0
        self._print(
            f"[PROGRESS] {self.label} | {self._status} | elapsed={elapsed:.0f}s"
        )

    def __enter__(self) -> ProgressHeartbeat:
        self._t0 = time.perf_counter()
        self._stop.clear()
        self._tick = 0
        self._print(f"[PROGRESS] {self.label} | started | elapsed=0s")
        self._thread = threading.Thread(target=self._loop, name=f"hb-{self.label}", daemon=True)
        self._thread.start()
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=self.interval_s + 1.0)
        elapsed = time.perf_counter() - self._t0
        if exc_type is None:
            self._print(f"[PROGRESS] {self.label} | finished OK | elapsed={elapsed:.1f}s")
        else:
            self._print(
                f"[PROGRESS] {self.label} | FAILED ({exc_type.__name__}) | elapsed={elapsed:.1f}s"
            )
        return None

    def _loop(self) -> None:
        while not self._stop.wait(self.interval_s):
            self._tick += 1
            elapsed = time.perf_counter() - self._t0
            self._print(
                f"[PROGRESS] {self.label} | still running… "
                f"status={self._status} | tick={self._tick} | elapsed={elapsed:.0f}s"
            )


def _default_print(msg: str) -> None:
    print(msg, flush=True)
    sys.stdout.flush()
