"""Windows screen-capture source integration for VisoMaster Fusion."""
from __future__ import annotations
import ctypes
import os
import time
from dataclasses import dataclass
import numpy as np
from PIL import ImageGrab
from PySide6 import QtWidgets

IS_WINDOWS = os.name == "nt"

@dataclass(frozen=True)
class MonitorInfo:
    index: int
    left: int
    top: int
    right: int
    bottom: int
    @property
    def width(self): return max(1, self.right - self.left)
    @property
    def height(self): return max(1, self.bottom - self.top)
    @property
    def label(self): return f"Monitor {self.index + 1} ({self.width}x{self.height})"

def enumerate_monitors():
    if not IS_WINDOWS: return []
    class RECT(ctypes.Structure):
        _fields_ = [("left", ctypes.c_long), ("top", ctypes.c_long), ("right", ctypes.c_long), ("bottom", ctypes.c_long)]
    monitors = []
    callback_type = ctypes.WINFUNCTYPE(ctypes.c_int, ctypes.c_void_p, ctypes.c_void_p, ctypes.POINTER(RECT), ctypes.c_ssize_t)
    def callback(_monitor, _dc, rect_ptr, _data):
        r = rect_ptr.contents
        monitors.append(MonitorInfo(len(monitors), int(r.left), int(r.top), int(r.right), int(r.bottom)))
        return 1
    cb = callback_type(callback)
    ctypes.windll.user32.EnumDisplayMonitors(0, 0, cb, 0)
    return monitors

class ScreenCaptureSource:
    def __init__(self, monitor_index=0, fps=30.0):
        monitors = enumerate_monitors()
        if not monitors: raise RuntimeError("No Windows display monitor was detected.")
        self.monitors = monitors
        self.monitor_index = max(0, min(int(monitor_index), len(monitors) - 1))
        self.fps = max(1.0, min(float(fps), 120.0))
        self.monitor = monitors[self.monitor_index]
        self._opened = True
        self._next_capture_time = 0.0
    @property
    def width(self): return self.monitor.width
    @property
    def height(self): return self.monitor.height
    def read(self):
        if not self._opened: return False, None
        interval = 1.0 / self.fps
        now = time.perf_counter()
        if self._next_capture_time > now: time.sleep(self._next_capture_time - now)
        self._next_capture_time = time.perf_counter() + interval
        try:
            image = ImageGrab.grab(bbox=(self.monitor.left, self.monitor.top, self.monitor.right, self.monitor.bottom), all_screens=True)
            rgb = np.asarray(image.convert("RGB"), dtype=np.uint8)
            return True, np.ascontiguousarray(rgb[:, :, ::-1])
        except Exception as exc:
            print(f"[ERROR] Screen capture failed: {exc}")
            return False, None
    def isOpened(self): return self._opened
    def release(self): self._opened = False
    def get(self, prop_id):
        try:
            import cv2
            if prop_id == cv2.CAP_PROP_FPS: return float(self.fps)
            if prop_id == cv2.CAP_PROP_FRAME_WIDTH: return float(self.width)
            if prop_id == cv2.CAP_PROP_FRAME_HEIGHT: return float(self.height)
        except Exception: pass
        return 0.0
    def set(self, _prop_id, _value): return False
    def open(self, *_args, **_kwargs): self._opened = True; return True

def _screen_thumbnail():
    from PySide6 import QtGui
    source = ScreenCaptureSource(0, 1.0)
    ok, frame = source.read(); source.release()
    if not ok or frame is None: return None
    h, w = frame.shape[:2]
    return QtGui.QImage(frame.data, w, h, int(frame.strides[0]), QtGui.QImage.Format.Format_BGR888).copy()

def _load_screen_media(self):
    main_window = self.main_window
    vp = main_window.video_processor
    try:
        vp.stop_processing()
        source = ScreenCaptureSource(getattr(main_window, "_screen_capture_monitor", 0), getattr(main_window, "_screen_capture_fps", 30.0))
        ok, frame = source.read()
        if not ok or frame is None: source.release(); raise RuntimeError("Unable to capture the selected monitor.")
        if vp.media_capture:
            try: vp.media_capture.release()
            except Exception: pass
        vp._screen_capture_source = source; vp.media_capture = source; vp.media_rotation = 0
        vp.media_path = "screen://monitor"; vp.file_type = "webcam"; vp.fps = source.fps
        vp.max_frame_number = 999999999; vp.current_frame_number = 0; vp.next_frame_to_display = 0; vp.current_frame = frame
        main_window.selected_target_face_id = None; main_window.scene.clear()
        from app.ui.widgets.actions import common_actions, graphics_view_actions
        pixmap = common_actions.get_pixmap_from_frame(main_window, frame)
        graphics_view_actions.update_graphics_view(main_window, pixmap, 0, reset_fit=True)
        self.reset_related_widgets_and_values()
        main_window.videoSeekSlider.setMaximum(999999999); main_window.videoSeekSlider.setValue(0)
        main_window.selected_video_button = self; main_window.loading_new_media = True
        common_actions.refresh_frame(main_window, synchronous=True)
        print("[INFO] Screen Capture selected.")
    except Exception as exc:
        print(f"[ERROR] Could not initialize screen capture: {exc}")

def add_screen_capture_card(main_window):
    if not IS_WINDOWS or getattr(main_window, "_screen_capture_card_added", False): return
    image = _screen_thumbnail()
    if image is None:
        print("[WARN] Screen Capture thumbnail could not be created."); return
    from app.ui.widgets import widget_components
    from app.ui.widgets.actions import list_view_actions
    list_view_actions.add_media_thumbnail_button(main_window, widget_components.TargetMediaCardButton, main_window.targetVideosList, main_window.target_videos, image, media_path="Screen Capture", file_type="video", media_id="screen-capture")
    button = main_window.target_videos.get("screen-capture")
    if button is not None:
        button.file_type = "screen"
        if button.list_item is not None: button.list_item.setFileType("video")
    main_window._screen_capture_card_added = True
    print("[INFO] Screen Capture source added to Target Media.")

def open_screen_capture_window(main_window):
    monitors = enumerate_monitors()
    if not monitors:
        QtWidgets.QMessageBox.warning(main_window, "Screen Capture", "Aucun écran Windows détecté."); return
    dialog = QtWidgets.QDialog(main_window)
    dialog.setWindowTitle("Screen Capture")
    dialog.setMinimumWidth(420)
    layout = QtWidgets.QFormLayout(dialog)
    monitor_box = QtWidgets.QComboBox(dialog)
    for monitor in monitors: monitor_box.addItem(monitor.label, monitor.index)
    fps_box = QtWidgets.QSpinBox(dialog); fps_box.setRange(5, 120); fps_box.setValue(30)
    layout.addRow("Écran :", monitor_box); layout.addRow("FPS :", fps_box)
    buttons = QtWidgets.QDialogButtonBox(QtWidgets.QDialogButtonBox.StandardButton.Cancel | QtWidgets.QDialogButtonBox.StandardButton.Ok, parent=dialog)
    buttons.accepted.connect(dialog.accept); buttons.rejected.connect(dialog.reject); layout.addRow(buttons)
    if dialog.exec() != QtWidgets.QDialog.DialogCode.Accepted: return
    main_window._screen_capture_monitor = int(monitor_box.currentData())
    main_window._screen_capture_fps = float(fps_box.value())
    add_screen_capture_card(main_window)
    button = main_window.target_videos.get("screen-capture")
    if button is not None: button.click()

def _install_target_media_source():
    from app.ui.widgets import widget_components
    original = widget_components.TargetMediaCardButton.load_media
    if getattr(original, "_screen_capture_original", False): return
    def load_media(self):
        if self.media_id == "screen-capture" or self.media_path == "Screen Capture": return _load_screen_media(self)
        return original(self)
    load_media._screen_capture_original = True
    widget_components.TargetMediaCardButton.load_media = load_media

def _install_video_processor_hooks():
    from app.processors.video_processor import VideoProcessor
    original = VideoProcessor.stop_processing
    if getattr(original, "_screen_capture_original", False): return
    def stop_processing(self, *args, **kwargs):
        result = original(self, *args, **kwargs)
        source = getattr(self, "_screen_capture_source", None)
        if source is not None and not source.isOpened():
            try:
                source = ScreenCaptureSource(getattr(self.main_window, "_screen_capture_monitor", 0), getattr(self.main_window, "_screen_capture_fps", 30.0))
                self._screen_capture_source = source; self.media_capture = source
            except Exception as exc: print(f"[WARN] Could not re-open screen capture after stop: {exc}")
        return result
    stop_processing._screen_capture_original = True
    VideoProcessor.stop_processing = stop_processing

def install():
    if not IS_WINDOWS: return
    _install_target_media_source(); _install_video_processor_hooks()
    print("[INFO] Windows Screen Capture integration installed.")
