"""Windows Graphics Capture source integrated with VisoMaster's live pipeline.

The capture backend is ``windows-capture``. Frames are copied once from the
native mapped buffer into a one-frame latest-value slot, then consumed through
the same ``media_capture.read()`` interface used by the existing video feeder.
No PIL/ImageGrab polling is used.
"""
from __future__ import annotations

import ctypes
import os
import threading
from dataclasses import dataclass
from typing import Any

import numpy as np
from PySide6 import QtGui, QtWidgets

IS_WINDOWS = os.name == "nt"


@dataclass(frozen=True)
class MonitorInfo:
    index: int
    left: int
    top: int
    right: int
    bottom: int

    @property
    def width(self) -> int:
        return max(1, self.right - self.left)

    @property
    def height(self) -> int:
        return max(1, self.bottom - self.top)

    @property
    def label(self) -> str:
        return f"Monitor {self.index + 1} ({self.width}x{self.height})"


def enumerate_monitors() -> list[MonitorInfo]:
    if not IS_WINDOWS:
        return []

    class RECT(ctypes.Structure):
        _fields_ = [
            ("left", ctypes.c_long),
            ("top", ctypes.c_long),
            ("right", ctypes.c_long),
            ("bottom", ctypes.c_long),
        ]

    monitors: list[MonitorInfo] = []
    callback_type = ctypes.WINFUNCTYPE(
        ctypes.c_int,
        ctypes.c_void_p,
        ctypes.c_void_p,
        ctypes.POINTER(RECT),
        ctypes.c_ssize_t,
    )

    def callback(_monitor: Any, _data: Any, rect: Any, _lparam: int) -> int:
        r = rect.contents
        monitors.append(
            MonitorInfo(
                len(monitors),
                int(r.left),
                int(r.top),
                int(r.right),
                int(r.bottom),
            )
        )
        return 1

    callback_ref = callback_type(callback)
    ctypes.windll.user32.EnumDisplayMonitors(0, 0, callback_ref, 0)
    return monitors


class ScreenCaptureSource:
    """OpenCV-like source backed by Windows Graphics Capture."""

    def __init__(self, monitor_index: int = 0, fps: float = 30.0) -> None:
        if not IS_WINDOWS:
            raise RuntimeError("Screen capture is only available on Windows.")
        try:
            from windows_capture import WindowsCapture
        except ImportError as exc:
            raise RuntimeError(
                "Install the 'windows-capture' package to use Screen Capture."
            ) from exc

        monitors = enumerate_monitors()
        if not monitors:
            raise RuntimeError("No Windows display monitor was detected.")
        self.monitors = monitors
        self.monitor_index = max(0, min(int(monitor_index), len(monitors) - 1))
        self.monitor = monitors[self.monitor_index]
        self.fps = max(1.0, min(float(fps), 120.0))
        self._condition = threading.Condition()
        self._latest: np.ndarray | None = None
        self._sequence = 0
        self._consumer_sequence = 0
        self._opened = False
        self._capture = WindowsCapture(
            cursor_capture=True,
            draw_border=False,
            monitor_index=self.monitor_index,
            minimum_update_interval=max(1, round(1000.0 / self.fps)),
        )
        self._control = None

        @self._capture.event
        def on_frame_arrived(frame: Any, _capture_control: Any) -> None:
            # frame.frame_buffer is a zero-copy view owned by the native capture
            # callback. Copy before returning so VisoMaster never holds a native
            # buffer after the callback has completed.
            bgr = np.ascontiguousarray(frame.convert_to_bgr().frame_buffer)
            with self._condition:
                self._latest = bgr
                self._sequence += 1
                self._condition.notify_all()

        @self._capture.event
        def on_closed() -> None:
            with self._condition:
                self._opened = False
                self._condition.notify_all()

    @property
    def width(self) -> int:
        return self.monitor.width

    @property
    def height(self) -> int:
        return self.monitor.height

    def start(self) -> None:
        if self._opened:
            return
        self._control = self._capture.start_free_threaded()
        self._opened = True

    def read(self) -> tuple[bool, np.ndarray | None]:
        if not self._opened:
            return False, None
        with self._condition:
            self._condition.wait_for(
                lambda: self._sequence > self._consumer_sequence or not self._opened,
                timeout=0.25,
            )
            if self._sequence > self._consumer_sequence and self._latest is not None:
                self._consumer_sequence = self._sequence
                return True, self._latest
        return False, None

    def isOpened(self) -> bool:
        return self._opened

    def release(self) -> None:
        control = self._control
        self._control = None
        self._opened = False
        if control is not None:
            try:
                control.stop()
                control.wait()
            except Exception:
                pass
        with self._condition:
            self._condition.notify_all()

    def get(self, prop_id: int) -> float:
        try:
            import cv2
            if prop_id == cv2.CAP_PROP_FPS:
                return self.fps
            if prop_id == cv2.CAP_PROP_FRAME_WIDTH:
                return float(self.width)
            if prop_id == cv2.CAP_PROP_FRAME_HEIGHT:
                return float(self.height)
        except Exception:
            pass
        return 0.0

    def set(self, _prop_id: int, _value: float) -> bool:
        return True


def _thumbnail() -> QtGui.QImage | None:
    try:
        source = ScreenCaptureSource(0, 1)
        source.start()
        ok, frame = source.read()
        source.release()
        if not ok or frame is None:
            return None
        h, w = frame.shape[:2]
        return QtGui.QImage(
            frame.data,
            w,
            h,
            int(frame.strides[0]),
            QtGui.QImage.Format.Format_BGR888,
        ).copy()
    except Exception as exc:
        print(f"[WARN] Screen Capture thumbnail unavailable: {exc}")
        return None


def _load_screen(self: Any) -> None:
    mw = self.main_window
    vp = mw.video_processor
    source: ScreenCaptureSource | None = None
    try:
        vp.stop_processing()
        source = ScreenCaptureSource(
            getattr(mw, "_screen_capture_monitor", 0),
            getattr(mw, "_screen_capture_fps", 30),
        )
        source.start()
        ok, frame = source.read()
        if not ok or frame is None:
            raise RuntimeError("Unable to acquire the first screen frame.")

        old_source = getattr(vp, "_screen_capture_source", None)
        if old_source is not None:
            try:
                old_source.release()
            except Exception:
                pass

        vp._screen_capture_source = source
        vp._screen_capture_active = True
        vp.media_capture = source
        vp.media_rotation = 0
        vp.media_path = "screen://monitor"
        vp.file_type = "screen"
        vp.fps = source.fps
        vp.max_frame_number = 2_147_483_647
        vp.current_frame_number = 0
        vp.next_frame_to_display = 0
        vp.current_frame = frame

        from app.ui.widgets.actions import common_actions, graphics_view_actions
        mw.scene.clear()
        pixmap = common_actions.get_pixmap_from_frame(mw, frame)
        graphics_view_actions.update_graphics_view(
            mw, pixmap, 0, reset_fit=True
        )
        self.reset_related_widgets_and_values()
        mw.videoSeekSlider.setMaximum(vp.max_frame_number)
        mw.videoSeekSlider.setValue(0)
        mw.selected_video_button = self
        mw.loading_new_media = True
        common_actions.refresh_frame(mw, synchronous=True)
        print(
            f"[INFO] Screen Capture active: monitor={source.monitor_index + 1}, "
            f"{source.width}x{source.height}, {source.fps:g} FPS"
        )
    except Exception as exc:
        if source is not None:
            source.release()
        vp._screen_capture_active = False
        print(f"[ERROR] Could not initialize Screen Capture: {exc}")


def _install_process_video_hook() -> None:
    from app.processors.video_processor import VideoProcessor

    original = VideoProcessor.process_video
    if getattr(original, "_screen_capture_hook", False):
        return

    def process_video(self: Any, *args: Any, **kwargs: Any) -> Any:
        if not getattr(self, "_screen_capture_active", False):
            return original(self, *args, **kwargs)
        source = getattr(self, "_screen_capture_source", None)
        if source is None or not source.isOpened():
            print("[ERROR] Screen Capture source is not open.")
            return None

        previous = (
            self.file_type,
            self.preroll_target,
            self.max_display_buffer_size,
        )
        try:
            # Reuse the mature continuous video feeder. The source itself is
            # still the Windows Graphics Capture object, not cv2.VideoCapture.
            self.file_type = "video"
            self.preroll_target = 1
            self.max_display_buffer_size = max(4, self.num_threads * 3)
            return original(self, *args, **kwargs)
        finally:
            self.file_type, self.preroll_target, self.max_display_buffer_size = previous

    process_video._screen_capture_hook = True
    VideoProcessor.process_video = process_video


def add_screen_capture_card(mw: Any) -> None:
    if not IS_WINDOWS or getattr(mw, "_screen_capture_card_added", False):
        return
    image = _thumbnail()
    if image is None:
        return
    from app.ui.widgets import widget_components
    from app.ui.widgets.actions import list_view_actions

    list_view_actions.add_media_thumbnail_button(
        mw,
        widget_components.TargetMediaCardButton,
        mw.targetVideosList,
        mw.target_videos,
        image,
        media_path="Screen Capture",
        file_type="video",
        media_id="screen-capture",
    )
    button = mw.target_videos.get("screen-capture")
    if button is not None:
        button.file_type = "screen"
        if button.list_item is not None:
            button.list_item.setFileType("video")
    mw._screen_capture_card_added = True


def open_screen_capture_window(mw: Any) -> None:
    monitors = enumerate_monitors()
    if not monitors:
        QtWidgets.QMessageBox.warning(mw, "Screen Capture", "No Windows monitor detected.")
        return

    dialog = QtWidgets.QDialog(mw)
    dialog.setWindowTitle("Screen Capture")
    layout = QtWidgets.QFormLayout(dialog)
    monitor = QtWidgets.QComboBox(dialog)
    fps = QtWidgets.QSpinBox(dialog)
    fps.setRange(5, 120)
    fps.setValue(30)
    for item in monitors:
        monitor.addItem(item.label, item.index)
    layout.addRow("Monitor:", monitor)
    layout.addRow("FPS:", fps)
    buttons = QtWidgets.QDialogButtonBox(
        QtWidgets.QDialogButtonBox.StandardButton.Cancel
        | QtWidgets.QDialogButtonBox.StandardButton.Ok,
        parent=dialog,
    )
    buttons.accepted.connect(dialog.accept)
    buttons.rejected.connect(dialog.reject)
    layout.addRow(buttons)
    if dialog.exec() != QtWidgets.QDialog.DialogCode.Accepted:
        return

    mw._screen_capture_monitor = int(monitor.currentData())
    mw._screen_capture_fps = float(fps.value())
    add_screen_capture_card(mw)
    button = mw.target_videos.get("screen-capture")
    if button is not None:
        button.click()


def _install_target_hook() -> None:
    from app.ui.widgets import widget_components

    original = widget_components.TargetMediaCardButton.load_media
    if getattr(original, "_screen_capture_hook", False):
        return

    def load_media(self: Any) -> Any:
        if self.media_id == "screen-capture" or self.media_path == "Screen Capture":
            return _load_screen(self)
        return original(self)

    load_media._screen_capture_hook = True
    widget_components.TargetMediaCardButton.load_media = load_media


def install() -> None:
    """Install optional screen capture hooks; no-op outside Windows."""
    if not IS_WINDOWS:
        return
    _install_process_video_hook()
    _install_target_hook()
