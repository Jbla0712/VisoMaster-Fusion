"""Windows screen capture integration for VisoMaster Fusion.

The live source uses Windows Graphics Capture through ``windows-capture``.
Frames are acquired on a native capture callback and exposed to the existing
VideoProcessor as a latest-frame source. This avoids PIL/ImageGrab being run
synchronously by the video feeder, which can introduce capture jitter and
variable latency at high desktop resolutions.
"""
from __future__ import annotations

import ctypes
import os
import threading
import time
from dataclasses import dataclass
from typing import Optional

import numpy as np
from PySide6 import QtWidgets

from app.processors.screen_capture import WindowsScreenCapture

IS_WINDOWS = os.name == "nt"


@dataclass(frozen=True)
class MonitorInfo:
    index: int
    left: int
    top: int
    right: int
    bottom: int

    @property
    def width(self):
        return max(1, self.right - self.left)

    @property
    def height(self):
        return max(1, self.bottom - self.top)

    @property
    def label(self):
        return f"Monitor {self.index + 1} ({self.width}x{self.height})"


def enumerate_monitors():
    if not IS_WINDOWS:
        return []

    class RECT(ctypes.Structure):
        _fields_ = [
            ("left", ctypes.c_long),
            ("top", ctypes.c_long),
            ("right", ctypes.c_long),
            ("bottom", ctypes.c_long),
        ]

    monitors = []
    callback_type = ctypes.WINFUNCTYPE(
        ctypes.c_int,
        ctypes.c_void_p,
        ctypes.c_void_p,
        ctypes.POINTER(RECT),
        ctypes.c_ssize_t,
    )

    def callback(_monitor, _data, rect, _lparam):
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
    """OpenCV-compatible live source backed by Windows Graphics Capture.

    ``windows-capture`` owns the native capture session and calls us whenever a
    frame arrives. We retain only the newest frame. ``read()`` therefore never
    performs a desktop screenshot and never accumulates a backlog of stale
    frames. The existing VisoMaster feeder can continue to use the familiar
    ``read/isOpened/release/get/set`` interface.
    """

    def __init__(self, monitor_index=0, fps=30.0):
        monitors = enumerate_monitors()
        if not monitors:
            raise RuntimeError("No Windows display monitor was detected.")
        self.monitors = monitors
        self.monitor_index = max(0, min(int(monitor_index), len(monitors) - 1))
        self.fps = max(1.0, min(float(fps), 120.0))
        self.monitor = monitors[self.monitor_index]

        self._opened = False
        self._capture: Optional[WindowsScreenCapture] = None
        self._latest_frame: Optional[np.ndarray] = None
        self._frame_lock = threading.Lock()
        self._frame_ready = threading.Condition(self._frame_lock)
        self._frame_sequence = 0
        self._last_read_sequence = 0
        self._logged_first_frame = False
        self._frame_count = 0

    @property
    def width(self):
        return self.monitor.width

    @property
    def height(self):
        return self.monitor.height

    def _on_frame(self, frame: np.ndarray):
        if frame is None or frame.ndim != 3:
            return
        with self._frame_ready:
            # The callback already receives a contiguous BGR numpy array. Do
            # not copy it again; replacing the reference is enough because the
            # capture callback allocates a fresh array for each delivered frame.
            self._latest_frame = frame
            self._frame_sequence += 1
            self._frame_count += 1
            self._frame_ready.notify_all()

    def _ensure_started(self):
        if self._opened:
            return
        capture = WindowsScreenCapture(self.monitor_index, self.fps)
        capture.start(self._on_frame)
        self._capture = capture
        self._opened = True

    def read(self):
        if not IS_WINDOWS:
            return False, None
        if not self._opened:
            self._ensure_started()

        # Wait briefly for the first native frame. After startup, return the
        # newest available frame immediately; never build a queue of old frames.
        deadline = time.perf_counter() + 1.0
        with self._frame_ready:
            while self._latest_frame is None and self._opened:
                remaining = deadline - time.perf_counter()
                if remaining <= 0:
                    break
                self._frame_ready.wait(timeout=remaining)

            frame = self._latest_frame
            sequence = self._frame_sequence

        if frame is None:
            return False, None

        if not self._logged_first_frame:
            print(
                f"[INFO] Screen Capture first frame acquired: "
                f"{frame.shape[1]}x{frame.shape[0]}"
            )
            self._logged_first_frame = True

        if sequence != self._last_read_sequence:
            self._last_read_sequence = sequence

        return True, frame

    def isOpened(self):
        return self._opened

    def release(self):
        self._opened = False
        with self._frame_ready:
            self._frame_ready.notify_all()
        capture = self._capture
        self._capture = None
        if capture is not None:
            capture.stop()

    def get(self, prop_id):
        try:
            import cv2

            if prop_id == cv2.CAP_PROP_FPS:
                return float(self.fps)
            if prop_id == cv2.CAP_PROP_FRAME_WIDTH:
                return float(self.width)
            if prop_id == cv2.CAP_PROP_FRAME_HEIGHT:
                return float(self.height)
        except Exception:
            pass
        return 0.0

    def set(self, *_args):
        # Desktop capture has no seek position. VideoProcessor may call this
        # while preparing the live source; it is intentionally a no-op.
        return True

    def open(self, *_args):
        self._ensure_started()
        return True


def _thumbnail():
    from PySide6 import QtGui

    source = ScreenCaptureSource(0, 1)
    ok, frame = source.read()
    source.release()
    if not ok:
        return None
    h, w = frame.shape[:2]
    return QtGui.QImage(
        frame.data,
        w,
        h,
        int(frame.strides[0]),
        QtGui.QImage.Format.Format_BGR888,
    ).copy()


def _load_screen(self):
    """Load desktop capture without ever opening the physical webcam."""
    mw = self.main_window
    vp = mw.video_processor
    try:
        vp.stop_processing()
        source = ScreenCaptureSource(
            getattr(mw, "_screen_capture_monitor", 0),
            getattr(mw, "_screen_capture_fps", 30),
        )
        ok, frame = source.read()
        if not ok:
            source.release()
            raise RuntimeError("Unable to capture monitor")

        old_source = getattr(vp, "media_capture", None)
        if old_source is not None and old_source is not getattr(
            vp, "_screen_capture_source", None
        ):
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
        vp.max_frame_number = 2147483647
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
        mw.videoSeekSlider.setMaximum(2147483647)
        mw.videoSeekSlider.setValue(0)
        mw.selected_video_button = self
        mw.loading_new_media = True
        common_actions.refresh_frame(mw, synchronous=True)
        print(
            f"[INFO] Screen Capture active: monitor={source.monitor_index + 1}, "
            f"{source.width}x{source.height}, {source.fps:g} FPS"
        )
    except Exception as exc:
        print(f"[ERROR] Could not initialize screen capture: {exc}")


def _install_process_video_hook():
    """Feed the continuous native source through the mature video pipeline.

    The source itself is low-latency/latest-frame. The video feeder remains in
    charge of pacing and dispatching frames to the existing processing workers.
    A very small preroll is intentional here: a large preroll would add visible
    latency to a live source.
    """
    from app.processors.video_processor import VideoProcessor

    original = VideoProcessor.process_video
    if getattr(original, "_screen_capture_original", False):
        return

    def process_video(self, *args, **kwargs):
        if not getattr(self, "_screen_capture_active", False):
            return original(self, *args, **kwargs)

        source = getattr(self, "_screen_capture_source", None)
        if source is None or not source.isOpened():
            print("[ERROR] Screen Capture process requested but source is not open.")
            return

        previous_file_type = self.file_type
        previous_preroll = self.preroll_target
        previous_max_buffer = self.max_display_buffer_size
        try:
            self.file_type = "video"
            self.preroll_target = 1
            self.max_display_buffer_size = max(4, self.num_threads * 2)
            print("[INFO] Screen Capture: starting low-latency video feeder")
            return original(self, *args, **kwargs)
        except Exception as exc:
            print(f"[ERROR] Screen Capture processing failed: {exc}")
            self.file_type = previous_file_type
            self.preroll_target = previous_preroll
            self.max_display_buffer_size = previous_max_buffer

    process_video._screen_capture_original = True
    VideoProcessor.process_video = process_video


def add_screen_capture_card(mw):
    if not IS_WINDOWS or getattr(mw, "_screen_capture_card_added", False):
        return
    image = _thumbnail()
    if image is None:
        print("[WARN] Screen Capture thumbnail unavailable.")
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
    print("[INFO] Screen Capture source added to Target Media.")


def open_screen_capture_window(mw):
    monitors = enumerate_monitors()
    if not monitors:
        QtWidgets.QMessageBox.warning(
            mw, "Screen Capture", "Aucun écran Windows détecté."
        )
        return
    dialog = QtWidgets.QDialog(mw)
    dialog.setWindowTitle("Screen Capture")
    dialog.setMinimumWidth(420)
    layout = QtWidgets.QFormLayout(dialog)
    monitor = QtWidgets.QComboBox(dialog)
    fps = QtWidgets.QSpinBox(dialog)
    fps.setRange(5, 120)
    fps.setValue(30)
    for item in monitors:
        monitor.addItem(item.label, item.index)
    layout.addRow("Écran :", monitor)
    layout.addRow("FPS :", fps)
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


def _install_target_hook():
    from app.ui.widgets import widget_components

    original = widget_components.TargetMediaCardButton.load_media
    if getattr(original, "_screen_capture_original", False):
        return

    def load_media(self):
        if self.media_id == "screen-capture" or self.media_path == "Screen Capture":
            return _load_screen(self)
        return original(self)

    load_media._screen_capture_original = True
    widget_components.TargetMediaCardButton.load_media = load_media


def _install_window_hook():
    from app.ui.main_ui import MainWindow

    original = MainWindow.__init__
    if getattr(original, "_screen_capture_original", False):
        return

    def init(self, *args, **kwargs):
        original(self, *args, **kwargs)
        button = QtWidgets.QPushButton("Screen Capture", self.dockWidgetContents)
        button.setToolTip("Capture a Windows monitor as the live source")
        button.clicked.connect(lambda: open_screen_capture_window(self))
        self.horizontalLayout_7.insertWidget(1, button)
        QtWidgets.QApplication.instance().processEvents()
        add_screen_capture_card(self)
        print("[INFO] Screen Capture button added to Target Media dock.")

    init._screen_capture_original = True
    MainWindow.__init__ = init


def install():
    if not IS_WINDOWS:
        return
    _install_process_video_hook()
    _install_target_hook()
    _install_window_hook()
    print("[INFO] Windows Screen Capture integration installed.")
