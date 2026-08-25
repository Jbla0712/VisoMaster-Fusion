"""Low-latency Windows display capture using DXGI Desktop Duplication.

This is a direct screen-capture path, similar to OBS Display Capture: it does
not use a webcam and does not go through an OBS virtual camera.
"""
from __future__ import annotations

import re
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
    device_index: int = 0
    output_index: int = 0


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
        return dxcam is not None and hasattr(dxcam, "output_info")

    @staticmethod
    def list_monitors() -> list[ScreenInfo]:
        """Return physical DXGI outputs exposed by dxcam."""
        if not ScreenCapture.available():
            return []
        result: list[ScreenInfo] = []
        try:
            raw = str(dxcam.output_info() or "")
            pattern = re.compile(
                r"Device\[(?P<device>\d+)\]\s+Output\[(?P<output>\d+)\]:\s*"
                r"Res:\((?P<w>\d+),\s*(?P<h>\d+)\).*?Primary:(?P<primary>True|False)"
            )
            for index, match in enumerate(pattern.finditer(raw)):
                device = int(match.group("device"))
                output = int(match.group("output"))
                width = int(match.group("w"))
                height = int(match.group("h"))
                primary = match.group("primary") == "True"
                label = f"Display {index + 1}"
                if primary:
                    label += " (Primary)"
                result.append(
                    ScreenInfo(
                        index=index,
                        name=label,
                        width=width,
                        height=height,
                        device_index=device,
                        output_index=output,
                    )
                )
        except Exception as exc:
            print(f"[WARN] Could not enumerate DXGI outputs: {exc}")
        return result

    def start(self) -> None:
        if self._running:
            return
        monitors = self.list_monitors()
        if not 0 <= self.monitor_index < len(monitors):
            raise ScreenCaptureError(f"Display index {self.monitor_index} is unavailable")
        monitor = monitors[self.monitor_index]
        try:
            self._camera = dxcam.create(
                device_idx=monitor.device_index,
                output_idx=monitor.output_index,
                output_color="BGR",
                backend="dxgi",
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
                release = getattr(self._camera, "release", None)
                if release is not None:
                    release()
            finally:
                self._camera = None
        self._running = False

    def __enter__(self) -> "ScreenCapture":
        self.start()
        return self

    def __exit__(self, *_args) -> None:
        self.stop()
