"""Low-latency OpenCV camera capture for Windows live sources.

The reader runs continuously and keeps only the newest frame. This prevents a
slow consumer (for example a face-swap pipeline) from displaying frames that
are already stale when they reach the UI.
"""
from __future__ import annotations

import threading
import time
from typing import Any

import cv2
import numpy as np

_ORIGINAL_VIDEOCAPTURE = cv2.VideoCapture
_INSTALLED = False


class LatestFrameCapture:
    """OpenCV-compatible camera reader with a one-frame software buffer."""

    def __init__(self, index: int, *args: Any, **kwargs: Any) -> None:
        if not args and not kwargs and cv2.CAP_DSHOW is not None:
            cap = _ORIGINAL_VIDEOCAPTURE(index, cv2.CAP_DSHOW)
            if cap.isOpened():
                self._cap = cap
            else:
                cap.release()
                self._cap = _ORIGINAL_VIDEOCAPTURE(index)
        else:
            self._cap = _ORIGINAL_VIDEOCAPTURE(index, *args, **kwargs)

        self._condition = threading.Condition()
        self._latest: np.ndarray | None = None
        self._sequence = 0
        self._consumer_sequence = 0
        self._stop = threading.Event()
        self._opened = bool(self._cap.isOpened())
        self._thread: threading.Thread | None = None

        if self._opened:
            # Not every backend supports this property; the reader-side
            # latest-frame policy remains the important part of the fix.
            try:
                self._cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
            except Exception:
                pass
            self._thread = threading.Thread(
                target=self._reader,
                name=f"VisoMasterCameraReader-{index}",
                daemon=True,
            )
            self._thread.start()

    def _reader(self) -> None:
        while not self._stop.is_set():
            ok, frame = self._cap.read()
            if not ok:
                time.sleep(0.002)
                continue
            with self._condition:
                self._latest = frame
                self._sequence += 1
                self._condition.notify_all()

    def read(self) -> tuple[bool, np.ndarray | None]:
        """Return the newest frame available since the previous read."""
        deadline = time.perf_counter() + 0.25
        with self._condition:
            while self._sequence <= self._consumer_sequence and not self._stop.is_set():
                remaining = deadline - time.perf_counter()
                if remaining <= 0:
                    break
                self._condition.wait(timeout=remaining)

            if self._sequence > self._consumer_sequence and self._latest is not None:
                self._consumer_sequence = self._sequence
                return True, self._latest
        return False, None

    def isOpened(self) -> bool:
        return self._opened and self._cap.isOpened()

    def release(self) -> None:
        self._stop.set()
        with self._condition:
            self._condition.notify_all()
        try:
            self._cap.release()
        finally:
            if self._thread and self._thread is not threading.current_thread():
                self._thread.join(timeout=1.0)
            self._opened = False

    def get(self, prop_id: int) -> float:
        return float(self._cap.get(prop_id))

    def set(self, prop_id: int, value: float) -> bool:
        return bool(self._cap.set(prop_id, value))

    def __getattr__(self, name: str) -> Any:
        return getattr(self._cap, name)


def _factory(source: Any = None, *args: Any, **kwargs: Any):
    if isinstance(source, (int, np.integer)):
        return LatestFrameCapture(int(source), *args, **kwargs)
    return _ORIGINAL_VIDEOCAPTURE(source, *args, **kwargs)


def install() -> None:
    """Install the camera reader shim once. Intended for Windows only."""
    global _INSTALLED
    if _INSTALLED or not __import__("sys").platform == "win32":
        return
    cv2.VideoCapture = _factory  # type: ignore[assignment]
    _INSTALLED = True
