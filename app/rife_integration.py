from __future__ import annotations

import importlib
import sys
import threading
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
from PySide6.QtCore import QObject, QTimer, Signal

_MODE_TO_EXP = {"Off": 0, "2x": 1, "4x": 2, "8x": 3}


def _project_root() -> Path:
    return Path(__file__).resolve().parent.parent


def _rife_root() -> Path:
    return _project_root() / "models" / "RIFE" / "ECCV2022-RIFE"


class _RifePreviewEngine:
    """Lazy, throttled RIFE engine used only by the GUI preview path."""

    def __init__(self) -> None:
        self.model: Any | None = None
        self.lock = threading.Lock()
        self.loaded = False
        self._condition = threading.Condition()
        self._pending: tuple[np.ndarray, np.ndarray, int, Any, int, int, int] | None = None
        self._busy = False
        self._generation = 0
        self._cuda_stream: torch.cuda.Stream | None = None
        self._worker = threading.Thread(target=self._worker_loop, name="RIFE-Preview-Worker", daemon=True)
        self._worker.start()

    @property
    def generation(self) -> int:
        with self._condition:
            return self._generation

    def _load(self) -> Any:
        if self.model is not None:
            return self.model
        root = _rife_root()
        model_dir = root / "train_log"
        if not (model_dir / "flownet.pkl").exists():
            raise FileNotFoundError(f"RIFE model not found: {model_dir / 'flownet.pkl'}")
        root_str = str(root)
        if root_str not in sys.path:
            sys.path.insert(0, root_str)
        module = importlib.import_module("train_log.RIFE_HDv3")
        model = module.Model()
        model.load_model(str(model_dir), -1)
        model.eval()
        if hasattr(model, "device"):
            model.device()
        if torch.cuda.is_available():
            self._cuda_stream = torch.cuda.Stream(priority=2)
        self.model = model
        self.loaded = True
        gpu = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU"
        print(f"[RIFE-PREVIEW] Model loaded: HDv3 | Device: {gpu}")
        return model

    @staticmethod
    def _to_tensor(frame: np.ndarray) -> tuple[torch.Tensor, int, int]:
        h, w = frame.shape[:2]
        rgb = np.ascontiguousarray(frame[:, :, :3])
        tensor = torch.from_numpy(rgb).permute(2, 0, 1).unsqueeze(0).float() / 255.0
        if torch.cuda.is_available():
            tensor = tensor.cuda(non_blocking=True)
        return tensor, h, w

    @staticmethod
    def _pad(tensor: torch.Tensor) -> tuple[torch.Tensor, tuple[int, int]]:
        _, _, h, w = tensor.shape
        ph = (32 - h % 32) % 32
        pw = (32 - w % 32) % 32
        return F.pad(tensor, (0, pw, 0, ph)), (ph, pw)

    @staticmethod
    def _crop(tensor: torch.Tensor, size: tuple[int, int]) -> np.ndarray:
        h, w = size
        out = tensor[0, :, :h, :w].clamp(0, 1)
        return out.mul(255.0).byte().permute(1, 2, 0).cpu().numpy().copy()

    def _infer(self, model: Any, first: torch.Tensor, second: torch.Tensor) -> torch.Tensor:
        return model.inference(first, second, scale=1.0)

    def interpolate(self, first: np.ndarray, second: np.ndarray, count: int) -> list[np.ndarray]:
        if count <= 0:
            return []
        with self.lock:
            model = self._load()
            with torch.inference_mode():
                a, h, w = self._to_tensor(first)
                b, _, _ = self._to_tensor(second)
                a, _ = self._pad(a)
                b, _ = self._pad(b)

                def recurse(x: torch.Tensor, y: torch.Tensor, n: int) -> list[torch.Tensor]:
                    if n <= 0:
                        return []
                    mid = self._infer(model, x, y)
                    if n == 1:
                        return [mid]
                    half = n // 2
                    left = recurse(x, mid, half)
                    right = recurse(mid, y, half)
                    if n % 2:
                        return [*left, mid, *right]
                    return [*left, *right]

                if self._cuda_stream is not None:
                    current = torch.cuda.current_stream()
                    self._cuda_stream.wait_stream(current)
                    with torch.cuda.stream(self._cuda_stream):
                        mids = recurse(a, b, count)
                        result = [self._crop(x, (h, w)) for x in mids]
                    current.wait_stream(self._cuda_stream)
                    return result

                return [self._crop(x, (h, w)) for x in recurse(a, b, count)]

    def enqueue(self, first: np.ndarray, second: np.ndarray, count: int, main_window: Any, frame_number: int, factor: int) -> None:
        if count <= 0:
            return
        with self._condition:
            if self._busy or self._pending is not None:
                return
            generation = self._generation
            self._pending = (
                np.ascontiguousarray(first[:, :, :3]).copy(),
                np.ascontiguousarray(second[:, :, :3]).copy(),
                count,
                main_window,
                frame_number,
                factor,
                generation,
            )
            self._condition.notify()

    def cancel(self) -> None:
        with self._condition:
            self._generation += 1
            self._pending = None
            self._condition.notify()

    def _worker_loop(self) -> None:
        while True:
            with self._condition:
                while self._pending is None:
                    self._condition.wait()
                request = self._pending
                self._pending = None
                self._busy = True
            try:
                first, second, count, main_window, frame_number, factor, generation = request
                mids = self.interpolate(first, second, count)
                with self._condition:
                    current_generation = self._generation
                if generation != current_generation or not mids:
                    continue
                delay_ms = max(1, int(getattr(main_window, "target_delay_sec", 1 / 30.0) * 1000 / factor))
                for index, mid in enumerate(mids, start=1):
                    _PREVIEW_DISPATCHER.show_frame.emit(main_window, mid, frame_number, delay_ms * index, generation)
                print(f"[RIFE-PREVIEW] queued {len(mids)} intermediate frame(s)")
            except Exception as exc:
                print(f"[RIFE-PREVIEW] Worker error: {exc}")
            finally:
                with self._condition:
                    self._busy = False


class _RifePreviewDispatcher(QObject):
    """Dispatch RIFE-generated preview frames onto the Qt GUI thread."""

    show_frame = Signal(object, object, object, int, int)

    def __init__(self) -> None:
        super().__init__()
        self.show_frame.connect(self._show_frame)

    @staticmethod
    def _show_frame(main_window: Any, frame: np.ndarray, frame_number: int, delay_ms: int, generation: int) -> None:
        def show() -> None:
            try:
                if generation != _PREVIEW_ENGINE.generation:
                    return
                if str(main_window.control.get("RIFEInterpolationSelection", "Off")) == "Off":
                    return
                from app.ui.widgets.actions import graphics_view_actions, common_actions
                pixmap = common_actions.get_pixmap_from_frame(main_window, frame)
                graphics_view_actions.update_graphics_view(main_window, pixmap, frame_number)
            except Exception as exc:
                print(f"[RIFE-PREVIEW] GUI dispatch failed: {exc}")
        QTimer.singleShot(max(1, delay_ms), show)


_PREVIEW_DISPATCHER = _RifePreviewDispatcher()
_PREVIEW_ENGINE = _RifePreviewEngine()


def set_rife_interpolation(main_window: Any, value: str) -> None:
    try:
        main_window.control["RIFEInterpolationSelection"] = value
        if value == "Off":
            _PREVIEW_ENGINE.cancel()
        print(f"[RIFE-PREVIEW] Selection changed: {value}")
    except Exception as exc:
        print(f"[RIFE-PREVIEW] Could not update selection: {exc}")


def install_settings(settings_layout_data: dict[str, Any]) -> None:
    recording = settings_layout_data.setdefault("Video Recording Settings", {})
    if "RIFEInterpolationSelection" not in recording:
        recording["RIFEInterpolationSelection"] = {
            "level": 1,
            "label": "RIFE Preview",
            "options": ["Off", "2x", "4x", "8x"],
            "default": "Off",
            "help": "RIFE frame interpolation for preview playback only. Recording/export is unchanged.",
            "exec_function": set_rife_interpolation,
            "exec_function_args": [],
        }


def patch_preview_pipeline() -> None:
    """Attach throttled asynchronous RIFE interpolation to preview only."""
    from app.processors.video_processor import VideoProcessor
    if getattr(VideoProcessor, "_rife_preview_patched", False):
        return
    original = VideoProcessor.display_next_frame

    def wrapped(self: Any, *args: Any, **kwargs: Any):
        result = original(self, *args, **kwargs)
        try:
            selection = str(self.main_window.control.get("RIFEInterpolationSelection", "Off"))
            factor = {"Off": 1, "2x": 2, "4x": 4, "8x": 8}.get(selection, 1)
            if factor <= 1 or self.recording or self.is_processing_segments:
                if factor <= 1:
                    _PREVIEW_ENGINE.cancel()
                return result
            if self.file_type != "video" or self.current_frame is None:
                return result
            next_number = self.next_frame_to_display
            next_frame = self.frames_to_display.get(next_number)
            if next_frame is None:
                return result
            _PREVIEW_ENGINE.enqueue(self.current_frame, next_frame, factor - 1, self.main_window, next_number - 1, factor)
        except Exception as exc:
            print(f"[RIFE-PREVIEW] Disabled for this frame after error: {exc}")
        return result

    VideoProcessor.display_next_frame = wrapped
    VideoProcessor._rife_preview_patched = True
    VideoProcessor.processing_stopped_signal.connect(_PREVIEW_ENGINE.cancel)
    print("[RIFE-PREVIEW] Throttled async preview pipeline installed (recording/export untouched)")


def patch_ffmpeg_encoder() -> None:
    return None


def patch_video_processor_stop() -> None:
    return None


def cancel_rife_for_encoder(encoder: Any) -> None:
    return None


def apply_rife_to_encoded_video(encoder: Any) -> None:
    return None
