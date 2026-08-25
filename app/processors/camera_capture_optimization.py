"""Low-latency OpenCV camera capture for Windows live sources.

The reader runs continuously and keeps only the newest frame. This prevents a
slow consumer (for example a face-swap pipeline) from displaying frames that
are already stale when they reach the UI.

Set VISO_CAPTURE_STATS=1 to print capture/consumer FPS, dropped-frame ratio and
capture-to-read latency every five seconds. The instrumentation is disabled by
default and is intended for A/B testing of live camera sources.
"""
from __future__ import annotations

import os
import threading
import time
from typing import Any

import cv2
import numpy as np

_ORIGINAL_VIDEOCAPTURE = cv2.VideoCapture
_INSTALLED = False
_STATS_INTERVAL = 5.0


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
        self._latest_captured_at = 0.0
        self._sequence = 0
        self._consumer_sequence = 0
        self._stop = threading.Event()
        self._opened = bool(self._cap.isOpened())
        self._thread: threading.Thread | None = None

        # Diagnostics are opt-in so normal users pay virtually no cost.
        self._stats_enabled = os.getenv("VISO_CAPTURE_STATS", "").lower() in {
            "1", "true", "yes", "on"
        }
        self._stats_lock = threading.Lock()
        self._stats_started_at = time.perf_counter()
        self._stats_last_report = self._stats_started_at
        self._captured = 0
        self._consumed = 0
        self._last_report_captured = 0
        self._last_report_consumed = 0
        self._latency_sum_ms = 0.0
        self._latency_samples = 0

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
            captured_at = time.perf_counter()
            with self._condition:
                self._latest = frame
                self._latest_captured_at = captured_at
                self._sequence += 1
            if self._stats_enabled:
                with self._stats_lock:
                    self._captured += 1
                self._maybe_report_stats(captured_at)
            with self._condition:
                self._condition.notify_all()

    def _maybe_report_stats(self, now: float) -> None:
        with self._stats_lock:
            elapsed = now - self._stats_last_report
            if elapsed < _STATS_INTERVAL:
                return
            captured = self._captured
            consumed = self._consumed
            captured_delta = captured - self._last_report_captured
            consumed_delta = consumed - self._last_report_consumed
            capture_fps = captured_delta / elapsed
            consumer_fps = consumed_delta / elapsed
            dropped = max(0, captured_delta - consumed_delta)
            drop_ratio = (dropped / captured_delta * 100.0) if captured_delta else 0.0
            avg_latency = (
                self._latency_sum_ms / self._latency_samples
                if self._latency_samples
                else 0.0
            )
            self._last_report_captured = captured
            self._last_report_consumed = consumed
            self._latency_sum_ms = 0.0
            self._latency_samples = 0
            self._stats_last_report = now

        print(
            "[CAMERA STATS] "
            f"capture={capture_fps:.1f} fps, "
            f"consumer={consumer_fps:.1f} fps, "
            f"dropped={drop_ratio:.1f}% ({dropped}/{captured_delta}), "
            f"capture_to_read_avg={avg_latency:.1f} ms"
        )

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
                frame = self._latest
                captured_at = self._latest_captured_at
            else:
                return False, None

        if self._stats_enabled:
            now = time.perf_counter()
            with self._stats_lock:
                self._consumed += 1
                self._latency_sum_ms += max(0.0, now - captured_at) * 1000.0
                self._latency_samples += 1

        return True, frame

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
