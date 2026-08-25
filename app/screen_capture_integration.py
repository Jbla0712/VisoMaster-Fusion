"""Runtime integration for the native screen-capture source.

The core VisoMaster webcam pipeline already handles live numpy frames. This
module adapts a DXGI/Desktop-Duplication capture to the same VideoCapture-like
interface, then adds a small "Screen Capture" source action to the existing UI.
No OBS Virtual Camera is involved.
"""
from __future__ import annotations

import uuid
from functools import partial

import cv2
import numpy as np
from PySide6 import QtCore, QtGui, QtWidgets

from app.processors.screen_capture import ScreenCapture, ScreenCaptureError
from app.ui.widgets import widget_components
from app.ui.widgets.actions import list_view_actions


_SCREEN_CAPTURE_INDEX_OFFSET = 10000
_original_card_load_media = None
_original_process_webcam = None
_original_stop_processing = None
_original_initialize_widgets = None


class ScreenCaptureVideoCapture:
    """Small cv2.VideoCapture-compatible adapter around DXGI capture."""

    def __init__(self, monitor_index: int, fps: float = 30.0):
        self.monitor_index = int(monitor_index)
        self._capture = ScreenCapture(self.monitor_index, fps=fps)
        self._opened = False
        self._fps = float(fps)
        self._width = 0
        self._height = 0

    def open(self) -> bool:
        try:
            self._capture.start()
            monitors = ScreenCapture.list_monitors()
            if 0 <= self.monitor_index < len(monitors):
                self._width = monitors[self.monitor_index].width
                self._height = monitors[self.monitor_index].height
            if self._width <= 0 or self._height <= 0:
                # Get one frame to discover the actual output size.
                frame = self._capture.read()
                if frame is not None:
                    self._height, self._width = frame.shape[:2]
            self._opened = True
            return True
        except Exception as exc:
            self.release()
            print(f"[ERROR] Screen capture start failed: {exc}")
            return False

    def isOpened(self) -> bool:
        return self._opened

    def read(self):
        if not self._opened:
            return False, None
        frame = self._capture.read()
        if frame is None:
            return False, None
        return True, np.ascontiguousarray(frame)

    def get(self, prop_id):
        if prop_id == cv2.CAP_PROP_FRAME_WIDTH:
            return float(self._width)
        if prop_id == cv2.CAP_PROP_FRAME_HEIGHT:
            return float(self._height)
        if prop_id == cv2.CAP_PROP_FPS:
            return self._fps
        if prop_id == cv2.CAP_PROP_FRAME_COUNT:
            return 999999.0
        if prop_id == cv2.CAP_PROP_POS_FRAMES:
            return 0.0
        return 0.0

    def set(self, prop_id, value):
        if prop_id == cv2.CAP_PROP_FPS and value:
            self._fps = float(value)
        # Resolution/FourCC/seek settings are intentionally ignored: DXGI owns
        # the desktop dimensions and the capture stream is live.
        return True

    def release(self):
        try:
            self._capture.stop()
        finally:
            self._opened = False

    def __del__(self):
        try:
            self.release()
        except Exception:
            pass


def _capture_thumbnail(main_window: "QtWidgets.QMainWindow", monitor_index: int):
    capture = None
    try:
        capture = ScreenCapture(monitor_index, fps=1)
        capture.start()
        frame = capture.read()
        if frame is None:
            return QtGui.QImage()
        # Keep thumbnails small so creating the source does not briefly consume
        # several hundred MB on a 4K/8K desktop.
        h, w = frame.shape[:2]
        max_width = 640
        if w > max_width:
            scale = max_width / float(w)
            frame = cv2.resize(
                frame,
                (max_width, max(1, int(h * scale))),
                interpolation=cv2.INTER_AREA,
            )
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        return QtGui.QImage(
            rgb.data,
            rgb.shape[1],
            rgb.shape[0],
            rgb.strides[0],
            QtGui.QImage.Format_RGB888,
        ).copy()
    except Exception as exc:
        print(f"[WARN] Could not create screen thumbnail: {exc}")
        return QtGui.QImage()
    finally:
        if capture is not None:
            capture.stop()


def add_screen_capture_source(main_window, monitor_index: int):
    """Create/select a Target Media card representing one physical display."""
    if not ScreenCapture.available():
        QtWidgets.QMessageBox.warning(
            main_window,
            "Screen Capture unavailable",
            "Install the Windows DXGI capture dependency with:\n\npip install dxcam",
        )
        return

    monitors = ScreenCapture.list_monitors()
    if monitor_index < 0 or monitor_index >= len(monitors):
        QtWidgets.QMessageBox.warning(main_window, "Screen Capture", "Display not found.")
        return

    # Stop whatever is currently running before switching source.
    if main_window.video_processor.processing:
        main_window.video_processor.stop_processing()

    monitor = monitors[monitor_index]
    media_id = f"screen_{monitor_index}_{uuid.uuid4().hex}"
    label = f"Screen Capture — {monitor.name}"
    image = _capture_thumbnail(main_window, monitor_index)

    button = widget_components.TargetMediaCardButton(
        label,
        "webcam",
        media_id,
        True,
        monitor_index,
        -1,
        main_window=main_window,
    )
    button.is_screen_capture = True
    button.screen_monitor_index = monitor_index
    button.setToolTip(f"{label} ({monitor.width}x{monitor.height})")

    if not image.isNull():
        button.set_thumbnail_pixmap(QtGui.QPixmap.fromImage(image))
    else:
        button.setText("Screen")

    button.setFixedSize(QtCore.QSize(96, 96))
    button.setCheckable(True)
    button.list_widget = main_window.targetVideosList
    button.list_item = QtWidgets.QListWidgetItem()
    button.list_item.setSizeHint(QtCore.QSize(96, 96))
    main_window.targetVideosList.addItem(button.list_item)
    main_window.targetVideosList.setItemWidget(button.list_item, button)
    main_window.target_videos[media_id] = button
    button.click()


def _show_screen_source_dialog(main_window):
    monitors = ScreenCapture.list_monitors()
    if not monitors:
        QtWidgets.QMessageBox.warning(
            main_window,
            "Screen Capture",
            "No Windows display was found, or DXGI capture is unavailable.",
        )
        return

    dialog = QtWidgets.QDialog(main_window)
    dialog.setWindowTitle("Screen Capture")
    layout = QtWidgets.QVBoxLayout(dialog)
    layout.addWidget(QtWidgets.QLabel("Choose the display to capture:"))
    combo = QtWidgets.QComboBox(dialog)
    for monitor in monitors:
        combo.addItem(
            f"{monitor.name} — {monitor.width}×{monitor.height}",
            monitor.index,
        )
    layout.addWidget(combo)
    buttons = QtWidgets.QDialogButtonBox(
        QtWidgets.QDialogButtonBox.Ok | QtWidgets.QDialogButtonBox.Cancel,
        parent=dialog,
    )
    buttons.accepted.connect(dialog.accept)
    buttons.rejected.connect(dialog.reject)
    layout.addWidget(buttons)

    if dialog.exec() == QtWidgets.QDialog.Accepted:
        add_screen_capture_source(main_window, int(combo.currentData()))


def _patched_card_load_media(self):
    if not getattr(self, "is_screen_capture", False):
        return _original_card_load_media(self)

    # Reuse the mature webcam source lifecycle. During load_media(), replace
    # cv2.VideoCapture only for this synchronous GUI operation with our adapter.
    monitor_index = int(getattr(self, "screen_monitor_index", self.webcam_index))
    original_videocapture = cv2.VideoCapture

    def _factory(index=0, backend=0):
        capture = ScreenCaptureVideoCapture(monitor_index)
        if not capture.open():
            raise ScreenCaptureError("Could not open the selected display")
        return capture

    cv2.VideoCapture = _factory
    try:
        return _original_card_load_media(self)
    finally:
        cv2.VideoCapture = original_videocapture


def _patched_process_webcam(self):
    selected = getattr(self.main_window, "selected_video_button", None)
    if getattr(selected, "is_screen_capture", False):
        monitor_index = int(getattr(selected, "screen_monitor_index", 0))
        # process_webcam uses this value to decide whether the already-open
        # capture can be reused. It is not a real camera device index here.
        self.main_window.control["WebcamDeviceSelection"] = monitor_index
        self.main_window.control["WebcamBackendSelection"] = "Default"
    return _original_process_webcam(self)


def _patched_stop_processing(self, *args, **kwargs):
    selected = getattr(self.main_window, "selected_video_button", None)
    was_screen = getattr(selected, "is_screen_capture", False)
    result = _original_stop_processing(self, *args, **kwargs)

    # The original webcam stop path reopens a cv2 camera. Replace that reopened
    # handle with DXGI so the screen source remains ready for the next Start.
    if was_screen and getattr(self, "file_type", None) == "webcam":
        monitor_index = int(getattr(selected, "screen_monitor_index", 0))
        try:
            if self.media_capture:
                self.media_capture.release()
            capture = ScreenCaptureVideoCapture(monitor_index)
            if capture.open():
                self.media_capture = capture
                self.fps = float(self.main_window.control.get("WebCamMaxFPSSelection", 30))
            else:
                self.media_capture = None
        except Exception as exc:
            print(f"[WARN] Could not restore screen capture after stop: {exc}")
            self.media_capture = None
    return result


def _patched_initialize_widgets(self):
    result = _original_initialize_widgets(self)
    menu_bar = self.menuBar()
    source_menu = menu_bar.addMenu("Live Source")
    screen_action = QtGui.QAction("Capture Screen…", self)
    screen_action.setToolTip("Capture a Windows display directly using DXGI Desktop Duplication")
    screen_action.triggered.connect(partial(_show_screen_source_dialog, self))
    source_menu.addAction(screen_action)
    self.screen_capture_action = screen_action
    return result


def install() -> None:
    """Install the integration once, before MainWindow is instantiated."""
    global _original_card_load_media
    global _original_process_webcam
    global _original_stop_processing
    global _original_initialize_widgets

    if _original_card_load_media is not None:
        return

    _original_card_load_media = widget_components.TargetMediaCardButton.load_media

    from app.processors.video_processor import VideoProcessor
    from app.ui.main_ui import MainWindow

    _original_process_webcam = VideoProcessor.process_webcam
    _original_stop_processing = VideoProcessor.stop_processing
    _original_initialize_widgets = MainWindow.initialize_widgets

    widget_components.TargetMediaCardButton.load_media = _patched_card_load_media
    VideoProcessor.process_webcam = _patched_process_webcam
    VideoProcessor.stop_processing = _patched_stop_processing
    MainWindow.initialize_widgets = _patched_initialize_widgets
