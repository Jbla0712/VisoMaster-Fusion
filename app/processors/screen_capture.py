"""Windows desktop capture source for the live VisoMaster pipeline.

Uses Windows Graphics Capture through the `windows-capture` package rather than
periodic screenshots. Frames are delivered as BGR numpy arrays, matching the
format expected by the existing OpenCV/live processing pipeline.
"""
from __future__ import annotations

import threading
import time
from typing import Callable, Optional

import numpy as np


class ScreenCaptureError(RuntimeError):
    pass


class WindowsScreenCapture:
    """Capture a selected Windows monitor/window and emit BGR numpy frames."""

    def __init__(self, monitor_index: int = 0, fps: float = 30.0):
        if fps <= 0:
            raise ValueError("fps must be greater than zero")
        self.monitor_index = monitor_index
        self.fps = fps
        self._stop = threading.Event()
        self._thread: Optional[threading.Thread] = None
        self._callback: Optional[Callable[[np.ndarray], None]] = None
        self._capture = None

    @staticmethod
    def available() -> bool:
        try:
            import windows_capture  # noqa: F401
            return True
        except ImportError:
            return False

    def start(self, callback: Callable[[np.ndarray], None]) -> None:
        if self._thread and self._thread.is_alive():
            return
        if not self.available():
            raise ScreenCaptureError(
                "The windows-capture package is not installed."
            )
        self._callback = callback
        self._stop.clear()
        self._thread = threading.Thread(
            target=self._run, name="VisoMasterScreenCapture", daemon=True
        )
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        capture = self._capture
        if capture is not None:
            try:
                capture.stop()
            except Exception:
                pass
        if self._thread and self._thread is not threading.current_thread():
            self._thread.join(timeout=2.0)
        self._thread = None
        self._capture = None

    def _run(self) -> None:
        try:
            from windows_capture import WindowsCapture, FramePool, ControlPanel
            from windows_capture import CaptureSettings, CursorCaptureSettings
        except ImportError as exc:
            raise ScreenCaptureError(
                "Install the windows-capture package to use screen capture."
            ) from exc

        frame_interval = 1.0 / self.fps
        next_frame = time.perf_counter()

        # The package owns the native Windows Graphics Capture session.  The
        # callback receives BGRA/RGBA data depending on package version; we
        # normalize it below before passing it into VisoMaster.
        capture = WindowsCapture(
            monitor_index=self.monitor_index,
            cursor_capture_settings=CursorCaptureSettings.Capture,
        )
        self._capture = capture

        def on_frame(frame, control_panel: ControlPanel):
            nonlocal next_frame
            if self._stop.is_set():
                control_panel.stop()
                return
            now = time.perf_counter()
            if now < next_frame:
                return
            next_frame = now + frame_interval

            array = np.asarray(frame)
            if array.ndim != 3:
                return
            if array.shape[2] == 4:
                array = array[:, :, :3]
            # windows-capture returns BGRA/BGR on current Windows builds.
            # Keep BGR to match cv2.VideoCapture and the existing live path.
            callback = self._callback
            if callback is not None:
                callback(np.ascontiguousarray(array))

        capture.event(on_frame)
        capture.start()
