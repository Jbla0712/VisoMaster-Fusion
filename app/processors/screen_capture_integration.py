"""Windows screen capture integration for VisoMaster Fusion."""
from __future__ import annotations
import ctypes
import os
import time
from dataclasses import dataclass
import numpy as np
from PIL import ImageGrab
from PySide6 import QtCore, QtWidgets

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
    monitors=[]
    cbtype=ctypes.WINFUNCTYPE(ctypes.c_int,ctypes.c_void_p,ctypes.c_void_p,ctypes.POINTER(RECT),ctypes.c_ssize_t)
    def cb(_m,_d,p,_x):
        r=p.contents; monitors.append(MonitorInfo(len(monitors),int(r.left),int(r.top),int(r.right),int(r.bottom))); return 1
    callback=cbtype(cb); ctypes.windll.user32.EnumDisplayMonitors(0,0,callback,0); return monitors

class ScreenCaptureSource:
    def __init__(self, monitor_index=0, fps=30.0):
        monitors=enumerate_monitors()
        if not monitors: raise RuntimeError("No Windows display monitor was detected.")
        self.monitors=monitors; self.monitor_index=max(0,min(int(monitor_index),len(monitors)-1)); self.fps=max(1.0,min(float(fps),120.0)); self.monitor=monitors[self.monitor_index]; self._opened=True; self._next_capture_time=0.0; self._logged_first_frame=False
    @property
    def width(self): return self.monitor.width
    @property
    def height(self): return self.monitor.height
    def read(self):
        if not self._opened: return False,None
        delay=1.0/self.fps; now=time.perf_counter()
        if self._next_capture_time>now: time.sleep(self._next_capture_time-now)
        self._next_capture_time=time.perf_counter()+delay
        try:
            image=ImageGrab.grab(bbox=(self.monitor.left,self.monitor.top,self.monitor.right,self.monitor.bottom),all_screens=True)
            rgb=np.asarray(image.convert("RGB"),dtype=np.uint8); frame=np.ascontiguousarray(rgb[:,:,::-1])
            if not self._logged_first_frame:
                print(f"[INFO] Screen Capture first frame acquired: {frame.shape[1]}x{frame.shape[0]}"); self._logged_first_frame=True
            return True,frame
        except Exception as exc:
            print(f"[ERROR] Screen capture failed: {exc}"); return False,None
    def isOpened(self): return self._opened
    def release(self): self._opened=False
    def get(self,prop_id):
        try:
            import cv2
            if prop_id==cv2.CAP_PROP_FPS:return float(self.fps)
            if prop_id==cv2.CAP_PROP_FRAME_WIDTH:return float(self.width)
            if prop_id==cv2.CAP_PROP_FRAME_HEIGHT:return float(self.height)
        except Exception: pass
        return 0.0
    def set(self,*_): return False
    def open(self,*_): self._opened=True; return True

def _thumbnail():
    from PySide6 import QtGui
    source=ScreenCaptureSource(0,1); ok,frame=source.read(); source.release()
    if not ok:return None
    h,w=frame.shape[:2]; return QtGui.QImage(frame.data,w,h,int(frame.strides[0]),QtGui.QImage.Format.Format_BGR888).copy()

def _keep_screen_source(mw):
    vp=getattr(mw,"video_processor",None); source=getattr(vp,"_screen_capture_source",None) if vp else None
    if source is not None and source.isOpened() and getattr(vp,"_screen_capture_active",False) and vp.media_capture is not source:
        old=vp.media_capture
        if old is not None and old is not source:
            try: old.release()
            except Exception: pass
        vp.media_capture=source

def _load_screen(self):
    mw=self.main_window; vp=mw.video_processor
    try:
        vp.stop_processing()
        source=ScreenCaptureSource(getattr(mw,"_screen_capture_monitor",0),getattr(mw,"_screen_capture_fps",30)); ok,frame=source.read()
        if not ok: source.release(); raise RuntimeError("Unable to capture monitor")
        if vp.media_capture and vp.media_capture is not getattr(vp,"_screen_capture_source",None):
            try: vp.media_capture.release()
            except Exception: pass
        vp._screen_capture_source=source; vp._screen_capture_active=True; vp.media_capture=source
        vp.media_rotation=0; vp.media_path="screen://monitor"; vp.file_type="webcam"; vp.fps=source.fps; vp.max_frame_number=999999999; vp.current_frame_number=0; vp.next_frame_to_display=0; vp.current_frame=frame
        from app.ui.widgets.actions import common_actions,graphics_view_actions
        mw.scene.clear(); pixmap=common_actions.get_pixmap_from_frame(mw,frame); graphics_view_actions.update_graphics_view(mw,pixmap,0,reset_fit=True); self.reset_related_widgets_and_values(); mw.videoSeekSlider.setMaximum(999999999); mw.videoSeekSlider.setValue(0); mw.selected_video_button=self; mw.loading_new_media=True; common_actions.refresh_frame(mw,synchronous=True)
        if not hasattr(mw,"_screen_capture_keepalive_timer"):
            timer=QtCore.QTimer(mw); timer.setInterval(10); timer.timeout.connect(lambda: _keep_screen_source(mw)); mw._screen_capture_keepalive_timer=timer
        mw._screen_capture_keepalive_timer.start()
        print("[INFO] Screen Capture selected; live source locked to screen capture.")
    except Exception as exc: print(f"[ERROR] Could not initialize screen capture: {exc}")

def add_screen_capture_card(mw):
    if not IS_WINDOWS or getattr(mw,"_screen_capture_card_added",False): return
    image=_thumbnail()
    if image is None: print("[WARN] Screen Capture thumbnail unavailable."); return
    from app.ui.widgets import widget_components
    from app.ui.widgets.actions import list_view_actions
    list_view_actions.add_media_thumbnail_button(mw,widget_components.TargetMediaCardButton,mw.targetVideosList,mw.target_videos,image,media_path="Screen Capture",file_type="video",media_id="screen-capture")
    button=mw.target_videos.get("screen-capture")
    if button is not None:
        button.file_type="screen"
        if button.list_item is not None: button.list_item.setFileType("video")
    mw._screen_capture_card_added=True; print("[INFO] Screen Capture source added to Target Media.")

def open_screen_capture_window(mw):
    monitors=enumerate_monitors()
    if not monitors: QtWidgets.QMessageBox.warning(mw,"Screen Capture","Aucun écran Windows détecté."); return
    dialog=QtWidgets.QDialog(mw); dialog.setWindowTitle("Screen Capture"); dialog.setMinimumWidth(420); layout=QtWidgets.QFormLayout(dialog); monitor=QtWidgets.QComboBox(dialog); fps=QtWidgets.QSpinBox(dialog); fps.setRange(5,120); fps.setValue(30)
    for m in monitors: monitor.addItem(m.label,m.index)
    layout.addRow("Écran :",monitor); layout.addRow("FPS :",fps); buttons=QtWidgets.QDialogButtonBox(QtWidgets.QDialogButtonBox.StandardButton.Cancel|QtWidgets.QDialogButtonBox.StandardButton.Ok,parent=dialog); buttons.accepted.connect(dialog.accept); buttons.rejected.connect(dialog.reject); layout.addRow(buttons)
    if dialog.exec()!=QtWidgets.QDialog.DialogCode.Accepted:return
    mw._screen_capture_monitor=int(monitor.currentData()); mw._screen_capture_fps=float(fps.value()); add_screen_capture_card(mw); button=mw.target_videos.get("screen-capture")
    if button is not None: button.click()

def _install_target_hook():
    from app.ui.widgets import widget_components
    original=widget_components.TargetMediaCardButton.load_media
    if getattr(original,"_screen_capture_original",False):return
    def load_media(self):
        if self.media_id=="screen-capture" or self.media_path=="Screen Capture": return _load_screen(self)
        return original(self)
    load_media._screen_capture_original=True; widget_components.TargetMediaCardButton.load_media=load_media

def _install_window_hook():
    from app.ui.main_ui import MainWindow
    original=MainWindow.__init__
    if getattr(original,"_screen_capture_original",False):return
    def init(self,*args,**kwargs):
        original(self,*args,**kwargs); button=QtWidgets.QPushButton("Screen Capture",self.dockWidgetContents); button.setToolTip("Capture a Windows monitor as the live source"); button.clicked.connect(lambda: open_screen_capture_window(self)); self.horizontalLayout_7.insertWidget(1,button); QtCore.QTimer.singleShot(0,lambda: add_screen_capture_card(self)); print("[INFO] Screen Capture button added to Target Media dock.")
    init._screen_capture_original=True; MainWindow.__init__=init

def _install_processor_hook():
    from app.processors.video_processor import VideoProcessor
    original=VideoProcessor.stop_processing
    if getattr(original,"_screen_capture_original",False): return
    def stop(self,*args,**kwargs):
        result=original(self,*args,**kwargs)
        if getattr(self,"_screen_capture_active",False):
            source=getattr(self,"_screen_capture_source",None)
            if source is not None and not source.isOpened():
                try: source=ScreenCaptureSource(getattr(self.main_window,"_screen_capture_monitor",0),getattr(self.main_window,"_screen_capture_fps",30)); self._screen_capture_source=source
                except Exception as exc: print(f"[WARN] Could not reopen screen capture: {exc}")
            if source is not None: self.media_capture=source
        return result
    stop._screen_capture_original=True; VideoProcessor.stop_processing=stop

def install():
    if not IS_WINDOWS:return
    _install_target_hook(); _install_window_hook(); _install_processor_hook(); print("[INFO] Windows Screen Capture integration installed.")
