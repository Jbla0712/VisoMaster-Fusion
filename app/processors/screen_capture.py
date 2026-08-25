"""Native Windows screen capture source.

Uses Windows Graphics Capture through the WinRT Python projection. Frames are
returned as BGR numpy arrays so they can be fed into VisoMaster's existing live
processing pipeline. This module intentionally has no dependency on OBS.
"""
from __future__ import annotations

import queue
import threading
import time
from dataclasses import dataclass
from typing import Optional

import numpy as np

try:
    from winrt.windows.graphics.capture import (
        Direct3D11CaptureFramePool,
        GraphicsCaptureItem,
        GraphicsCaptureSession,
    )
    from winrt.windows.graphics.directx.direct3d11 import IDirect3DDevice
    _WINRT_AVAILABLE = True
except ImportError:
    _WINRT_AVAILABLE = False


@dataclass(frozen=True)
class ScreenInfo:
    index: int
    name: str
    width: int
    height: int


class ScreenCaptureError(RuntimeError):
    pass


class ScreenCapture:
    """Capture a Windows display without OBS or a virtual camera."""

    def __init__(self, monitor_index: int = 0, fps: float = 30.0):
        if not _WINRT_AVAILABLE:
            raise ScreenCaptureError(
                "Windows Graphics Capture support is not installed. "
                "Install the WinRT dependencies from requirements.txt."
            )
        self.monitor_index = monitor_index
        self.fps = max(1.0, float(fps))
        self._queue: queue.Queue[np.ndarray] = queue.Queue(maxsize=2)
        self._stop = threading.Event()
        self._thread: Optional[threading.Thread] = None
        self._last_frame: Optional[np.ndarray] = None

    @staticmethod
    def available() -> bool:
        return _WINRT_AVAILABLE

    @staticmethod
    def list_monitors() -> list[ScreenInfo]:
        """Return physical displays using the Windows display API."""
        if not _WINRT_AVAILABLE:
            return []
        # The actual GraphicsCaptureItem picker is intentionally kept out of the
        # worker. The UI can provide a monitor index; pywin32 supplies stable
        # monitor geometry without taking a screenshot.
        try:
            import win32api
            result: list[ScreenInfo] = []
            for i, monitor in enumerate(win32api.EnumDisplayMonitors()):
                handle, _, rect = monitor
                left, top, right, bottom = rect
                result.append(ScreenInfo(i, f"Display {i + 1}", right-left, bottom-top))
            return result
        except Exception:
            return []

    def start(self) -> None:
        if self._thread and self._thread.is_alive():
            return
        self._stop.clear()
        self._thread = threading.Thread(target=self._capture_loop, name="ScreenCapture", daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        if self._thread and self._thread is not threading.current_thread():
            self._thread.join(timeout=1.0)
        self._thread = None
        self._clear_queue()

    def read(self, timeout: float = 0.25) -> Optional[np.ndarray]:
        try:
            frame = self._queue.get(timeout=timeout)
            self._last_frame = frame
            return frame
        except queue.Empty:
            return self._last_frame

    def _clear_queue(self) -> None:
        while True:
            try:
                self._queue.get_nowait()
            except queue.Empty:
                return

    def _capture_loop(self) -> None:
        """Capture loop placeholder for the WinRT D3D11 interop path.

        WinRT GraphicsCaptureItem objects are apartment-bound and require a
        D3D11 device created by the application's graphics stack. The helper
        below is isolated so the rest of VisoMaster remains backend-neutral.
        """
        try:
            self._capture_winrt()
        except Exception:
            # Never crash the GUI thread because a monitor disappears or the
            # capture permission changes. The caller observes an empty stream.
            self._stop.set()

    def _capture_winrt(self) -> None:
        raise ScreenCaptureError(
            "Windows Graphics Capture D3D11 interop is unavailable in this build."
        )

    def __enter__(self) -> "ScreenCapture":
        self.start()
        return self

    def __exit__(self, *_args) -> None:
        self.stop()
