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
    """Lightweight Windows monitor capture with the interface VisoMaster's live pipeline needs."""

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
            image = ImageGrab.grab(
                bbox=(self.monitor.left, self.monitor.top, self.monitor.right, self.monitor.bottom),
                all_screens=True,
            )
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
    raw = str(main_window.control.get("ScreenCaptureMonitorSelection", "Monitor 1"))
    try:
        return max(0, int(raw.split("Monitor ", 1)[1].split(" ", 1)[0]) - 1)
    except Exception:
        return 0


def _fps_from_control(main_window) -> float:
    try:
        return float(main_window.control.get("ScreenCaptureFpsSlider", 30))
    except Exception:
        return 30.0


def _screen_thumbnail(main_window):
    from PySide6 import QtGui
    source = ScreenCaptureSource(_monitor_index_from_control(main_window), _fps_from_control(main_window))
    ok, frame_bgr = source.read()
    source.release()
    if not ok or frame_bgr is None:
        return None
    h, w = frame_bgr.shape[:2]
    return QtGui.QImage(frame_bgr.data, w, h, int(frame_bgr.strides[0]), QtGui.QImage.Format.Format_BGR888).copy()


def _install_settings() -> None:
    from app.ui.widgets.settings_layout_data import SETTINGS_LAYOUT_DATA
    general = SETTINGS_LAYOUT_DATA.setdefault("General", {})
    monitors = enumerate_monitors()
    options = [m.label for m in monitors] or ["Monitor 1"]
    general.setdefault("ScreenCaptureMonitorSelection", {
        "level": 1,
        "label": "Screen Capture Monitor",
        "options": options,
        "default": options[0],
        "help": "Monitor used by the Windows Screen Capture source.",
    })
    general.setdefault("ScreenCaptureFpsSlider", {
        "level": 1,
        "label": "Screen Capture FPS",
        "min_value": "5",
        "max_value": "120",
        "default": "30",
        "step": 1,
        "help": "Capture rate used by Windows Screen Capture.",
    })


def _install_target_media_source() -> None:
    from app.ui.widgets import widget_components
    from app.ui.widgets.actions import list_view_actions

    original_load_media = widget_components.TargetMediaCardButton.load_media
    original_load_target_webcams = list_view_actions.load_target_webcams

    def load_media(self):
        if self.file_type != "screen":
            return original_load_media(self)

        main_window = self.main_window
        vp = main_window.video_processor
        try:
            if main_window.selected_video_button and main_window.selected_video_button is not self:
                main_window.selected_video_button.blockSignals(True)
                main_window.selected_video_button.setChecked(False)
                main_window.selected_video_button.blockSignals(False)

            vp.stop_processing()
            vp._clear_single_frame_preview_caches()
            source = ScreenCaptureSource(_monitor_index_from_control(main_window), _fps_from_control(main_window))
            ok, frame_bgr = source.read()
            if not ok or frame_bgr is None:
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

            main_window.parameters = {}
            main_window.selected_target_face_id = None
            main_window.scene.clear()
            from app.ui.widgets.actions import common_actions, graphics_view_actions
            pixmap = common_actions.get_pixmap_from_frame(main_window, frame_bgr)
            graphics_view_actions.update_graphics_view(main_window, pixmap, 0, reset_fit=True)

            self.reset_related_widgets_and_values()
            vp.file_type = "webcam"
            main_window.videoSeekSlider.blockSignals(True)
            main_window.videoSeekSlider.setMaximum(999999999)
            main_window.videoSeekSlider.setValue(0)
            main_window.videoSeekSlider.blockSignals(False)
            self._toggle_timeline_visibility(main_window, True)
            main_window.selected_video_button = self
            main_window.graphicsViewFrame.update()
            main_window.loading_new_media = True
            common_actions.refresh_frame(main_window, synchronous=True)
        except Exception as exc:
            print(f"[ERROR] Could not initialize screen capture: {exc}")
            try:
                vp.stop_processing()
            except Exception:
                pass

    def load_target_webcams(main_window, *args, **kwargs):
        # Keep the normal webcam loader untouched.
        original_load_target_webcams(main_window, *args, **kwargs)
        if not IS_WINDOWS:
            return

        existing = main_window.target_videos.get("screen-capture")
        if existing is not None:
            try:
                existing.remove_target_media_from_list()
            except Exception:
                main_window.target_videos.pop("screen-capture", None)

        image = _screen_thumbnail(main_window)
        if image is None:
            print("[WARN] Screen Capture thumbnail could not be created.")
            return

        # Always show Screen Capture, even when the Webcam filter checkbox is off.
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
        print("[INFO] Screen Capture source added to Target Media.")

    widget_components.TargetMediaCardButton.load_media = load_media
    list_view_actions.load_target_webcams = load_target_webcams


def _install_video_processor_hooks() -> None:
    from app.processors.video_processor import VideoProcessor
    original_stop = VideoProcessor.stop_processing

    def stop_processing(self, *args, **kwargs):
        result = original_stop(self, *args, **kwargs)
        if getattr(self, "_screen_capture_source", None) is not None and getattr(self, "file_type", None) == "webcam":
            try:
                source = ScreenCaptureSource(_monitor_index_from_control(self.main_window), _fps_from_control(self.main_window))
                self._screen_capture_source = source
                self.media_capture = source
                self.media_rotation = 0
                self.fps = source.fps
                self.max_frame_number = 999999999
            except Exception as exc:
                print(f"[WARN] Could not re-open screen capture after stop: {exc}")
        return result

    VideoProcessor.stop_processing = stop_processing


def install() -> None:
    if not IS_WINDOWS:
        return
    try:
        _install_settings()
        _install_target_media_source()
        _install_video_processor_hooks()
        print("[INFO] Windows Screen Capture integration installed.")
    except Exception as exc:
        print(f"[WARN] Windows Screen Capture integration disabled: {exc}")
