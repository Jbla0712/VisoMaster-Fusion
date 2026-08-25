"""Windows screen-capture source integration for VisoMaster Fusion."""

from __future__ import annotations

import ctypes
import os
import time
from dataclasses import dataclass

import numpy as np
from PIL import ImageGrab

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


def _make_dpi_aware() -> None:
    if not IS_WINDOWS:
        return
    try:
        ctypes.windll.user32.SetProcessDpiAwarenessContext(ctypes.c_void_p(-4))
        return
    except Exception:
        pass
    try:
        ctypes.windll.shcore.SetProcessDpiAwareness(2)
    except Exception:
        pass


def enumerate_monitors() -> list[MonitorInfo]:
    if not IS_WINDOWS:
        return []
    _make_dpi_aware()

    class RECT(ctypes.Structure):
        _fields_ = [("left", ctypes.c_long), ("top", ctypes.c_long), ("right", ctypes.c_long), ("bottom", ctypes.c_long)]

    monitors: list[MonitorInfo] = []
    callback_type = ctypes.WINFUNCTYPE(ctypes.c_int, ctypes.c_void_p, ctypes.c_void_p, ctypes.POINTER(RECT), ctypes.c_ssize_t)

    def callback(_monitor, _dc, rect_ptr, _data):
        rect = rect_ptr.contents
        monitors.append(MonitorInfo(len(monitors), int(rect.left), int(rect.top), int(rect.right), int(rect.bottom)))
        return 1

    cb = callback_type(callback)
    ctypes.windll.user32.EnumDisplayMonitors(0, 0, cb, 0)
    return monitors


class ScreenCaptureSource:
    """Live monitor capture exposing the read/isOpened/release/get API used by the live pipeline."""

    def __init__(self, monitor_index: int = 0, fps: float = 30.0):
        if not IS_WINDOWS:
            raise RuntimeError("Windows screen capture is only available on Windows.")
        monitors = enumerate_monitors()
        if not monitors:
            raise RuntimeError("No Windows display monitor was detected.")
        self.monitors = monitors
        self.monitor_index = max(0, min(int(monitor_index), len(monitors) - 1))
        self.fps = max(1.0, min(float(fps), 120.0))
        self.monitor = monitors[self.monitor_index]
        self._opened = True
        self._next_capture_time = 0.0

    @property
    def width(self):
        return self.monitor.width

    @property
    def height(self):
        return self.monitor.height

    def read(self):
        if not self._opened:
            return False, None
        interval = 1.0 / self.fps
        now = time.perf_counter()
        if self._next_capture_time > now:
            time.sleep(self._next_capture_time - now)
        self._next_capture_time = time.perf_counter() + interval
        try:
            image = ImageGrab.grab(bbox=(self.monitor.left, self.monitor.top, self.monitor.right, self.monitor.bottom), all_screens=True)
            rgb = np.asarray(image.convert("RGB"), dtype=np.uint8)
            return True, np.ascontiguousarray(rgb[:, :, ::-1])
        except Exception as exc:
            print(f"[ERROR] Screen capture failed: {exc}")
            return False, None

    def isOpened(self):
        return self._opened

    def release(self):
        self._opened = False

    def get(self, prop_id: int) -> float:
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

    def set(self, _prop_id: int, _value: float) -> bool:
        return False

    def open(self, *_args, **_kwargs) -> bool:
        self._opened = True
        return True


def _monitor_index_from_control(main_window) -> int:
    return 0


def _fps_from_control(main_window) -> float:
    return 30.0


def _screen_thumbnail(main_window):
    from PySide6 import QtGui
    source = ScreenCaptureSource(0, 1.0)
    ok, frame_bgr = source.read()
    source.release()
    if not ok or frame_bgr is None:
        return None
    h, w = frame_bgr.shape[:2]
    return QtGui.QImage(frame_bgr.data, w, h, int(frame_bgr.strides[0]), QtGui.QImage.Format.Format_BGR888).copy()


def _load_screen_media(self):
    main_window = self.main_window
    vp = main_window.video_processor
    try:
        vp.stop_processing()
        source = ScreenCaptureSource(0, 30.0)
        ok, frame_bgr = source.read()
        if not ok or frame_bgr is None:
            source.release()
            raise RuntimeError("Unable to capture the selected monitor.")
        if vp.media_capture:
            try:
                vp.media_capture.release()
            except Exception:
                pass
        vp._screen_capture_source = source
        vp.media_capture = source
        vp.media_rotation = 0
        vp.media_path = "screen://monitor"
        vp.file_type = "webcam"
        vp.fps = source.fps
        vp.max_frame_number = 999999999
        vp.current_frame_number = 0
        vp.next_frame_to_display = 0
        vp.current_frame = frame_bgr
        main_window.selected_target_face_id = None
        main_window.scene.clear()
        from app.ui.widgets.actions import common_actions, graphics_view_actions
        pixmap = common_actions.get_pixmap_from_frame(main_window, frame_bgr)
        graphics_view_actions.update_graphics_view(main_window, pixmap, 0, reset_fit=True)
        self.reset_related_widgets_and_values()
        main_window.videoSeekSlider.blockSignals(True)
        main_window.videoSeekSlider.setMaximum(999999999)
        main_window.videoSeekSlider.setValue(0)
        main_window.videoSeekSlider.blockSignals(False)
        main_window.selected_video_button = self
        main_window.graphicsViewFrame.update()
        main_window.loading_new_media = True
        common_actions.refresh_frame(main_window, synchronous=True)
        print("[INFO] Screen Capture selected.")
    except Exception as exc:
        print(f"[ERROR] Could not initialize screen capture: {exc}")


def add_screen_capture_card(main_window) -> None:
    if not IS_WINDOWS or getattr(main_window, "_screen_capture_card_added", False):
        return
    image = _screen_thumbnail(main_window)
    if image is None:
        print("[WARN] Screen Capture thumbnail could not be created.")
        return
    from app.ui.widgets import widget_components
    from app.ui.widgets.actions import list_view_actions
    list_view_actions.add_media_thumbnail_button(
        main_window,
        widget_components.TargetMediaCardButton,
        main_window.targetVideosList,
        main_window.target_videos,
        image,
        media_path="Screen Capture",
        file_type="screen",
        media_id="screen-capture",
    )
    main_window._screen_capture_card_added = True
    print("[INFO] Screen Capture source added to Target Media.")


def _install_target_media_source() -> None:
    from app.ui.widgets import widget_components
    original_load_media = widget_components.TargetMediaCardButton.load_media
    if not getattr(original_load_media, "_screen_capture_original", False):
        def load_media(self):
            if self.file_type == "screen":
                return _load_screen_media(self)
            return original_load_media(self)
        load_media._screen_capture_original = True
        widget_components.TargetMediaCardButton.load_media = load_media


def _install_video_processor_hooks() -> None:
    from app.processors.video_processor import VideoProcessor
    original_stop = VideoProcessor.stop_processing
    if getattr(original_stop, "_screen_capture_original", False):
        return
    def stop_processing(self, *args, **kwargs):
        result = original_stop(self, *args, **kwargs)
        source = getattr(self, "_screen_capture_source", None)
        if source is not None and not source.isOpened():
            try:
                source = ScreenCaptureSource(0, 30.0)
                self._screen_capture_source = source
                self.media_capture = source
            except Exception as exc:
                print(f"[WARN] Could not re-open screen capture after stop: {exc}")
        return result
    stop_processing._screen_capture_original = True
    VideoProcessor.stop_processing = stop_processing


def install() -> None:
    if not IS_WINDOWS:
        return
    _install_target_media_source()
    _install_video_processor_hooks()
    print("[INFO] Windows Screen Capture integration installed.")
