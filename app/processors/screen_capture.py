"""Low-latency Windows display capture using DXGI Desktop Duplication.

This is a direct screen-capture path, similar to OBS Display Capture: it does
not use a webcam and does not go through an OBS virtual camera.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np

try:
    import dxcam
except ImportError:  # pragma: no cover - Windows optional dependency
    dxcam = None


@dataclass(frozen=True)
class ScreenInfo:
    index: int
    name: str
    width: int
    height: int


class ScreenCaptureError(RuntimeError):
    pass


class ScreenCapture:
    """Direct Windows DXGI display capture returning BGR numpy frames."""

    def __init__(self, monitor_index: int = 0, fps: float = 30.0):
        if dxcam is None:
            raise ScreenCaptureError(
                "dxcam is required for Windows screen capture. Install dxcam."
            )
        self.monitor_index = int(monitor_index)
        self.fps = max(1.0, float(fps))
        self._camera = None
        self._running = False

    @staticmethod
    def available() -> bool:
        return dxcam is not None

    @staticmethod
    def list_monitors() -> list[ScreenInfo]:
        """Return displays visible to the DXGI capture backend."""
        if dxcam is None:
            return []
        result: list[ScreenInfo] = []
        try:
            devices = dxcam.device_info()
            for i, monitor in enumerate(devices):
                width = int(monitor.get("Width", monitor.get("width", 0)))
                height = int(monitor.get("Height", monitor.get("height", 0)))
                name = str(monitor.get("Device", f"Display {i + 1}"))
                result.append(ScreenInfo(i, name, width, height))
        except Exception:
            # Keep the capture backend usable even when dxcam changes its
            # optional device-info API. The monitor can still be selected by index.
            result = []
        return result

    def start(self) -> None:
        if self._running:
            return
        try:
            self._camera = dxcam.create(
                output_idx=self.monitor_index,
                output_color="BGR",
            )
            self._camera.start(target_fps=self.fps, video_mode=True)
            self._running = True
        except Exception as exc:
            self._camera = None
            raise ScreenCaptureError(f"Unable to start display capture: {exc}") from exc

    def read(self) -> Optional[np.ndarray]:
        if not self._running or self._camera is None:
            return None
        frame = self._camera.get_latest_frame()
        if frame is None:
            return None
        return np.asarray(frame)

    def stop(self) -> None:
        if self._camera is not None:
            try:
                self._camera.stop()
            finally:
                self._camera = None
        self._running = False

    def __enter__(self) -> "ScreenCapture":
        self.start()
        return self

    def __exit__(self, *_args) -> None:
        self.stop()
