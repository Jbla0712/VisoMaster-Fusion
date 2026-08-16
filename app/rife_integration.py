from __future__ import annotations

import importlib
import sys
import threading
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
from PySide6.QtCore import QObject, Signal

_MODE_TO_EXP = {"Off": 0, "2x": 1, "4x": 2, "8x": 3}


def _project_root() -> Path:
    return Path(__file__).resolve().parent.parent


def _rife_root() -> Path:
    return _project_root() / "models" / "RIFE" / "ECCV2022-RIFE"


class _RifePreviewEngine:
    """Lazy RIFE engine used only by the GUI preview path."""

    def __init__(self) -> None:
        self.model: Any | None = None
        self.lock = threading.Lock()
        self.loaded = False

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
        return (out.mul(255.0).byte().permute(1, 2, 0).cpu().numpy()).copy()

    def _infer(self, model: Any, first: torch.Tensor, second: torch.Tensor) -> torch.Tensor:
        """Call the HDv3 model using the API shipped by RIFE.

        HDv3's Model.inference accepts (img0, img1, scale=...). It does not
        accept the timestep keyword used by some newer RIFE wrappers.
        """
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

                mids = recurse(a, b, count)
                return [self._crop(x, (h, w)) for x in mids]


class _RifePreviewDispatcher(QObject):
    """Dispatch RIFE-generated preview frames onto the Qt GUI thread."""

    show_frame = Signal(object, object, object)

    def __init__(self) -> None:
        super().__init__()
        self.show_frame.connect(self._show_frame)

    @staticmethod
    def _show_frame(main_window: Any, frame: np.ndarray, frame_number: int) -> None:
        try:
            from app.ui.widgets.actions import graphics_view_actions, common_actions

            pixmap = common_actions.get_pixmap_from_frame(main_window, frame)
            graphics_view_actions.update_graphics_view(
                main_window, pixmap, frame_number
            )
        except Exception as exc:
            print(f"[RIFE-PREVIEW] GUI dispatch failed: {exc}")


_PREVIEW_ENGINE = _RifePreviewEngine()
_PREVIEW_DISPATCHER = _RifePreviewDispatcher()


def set_rife_interpolation(main_window: Any, value: str) -> None:
    try:
        main_window.control["RIFEInterpolationSelection"] = value
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
    """Attach RIFE between already-processed preview frames only."""
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
                return result
            if self.file_type != "video" or self.current_frame is None:
                return result

            next_number = self.next_frame_to_display
            next_frame = self.frames_to_display.get(next_number)
            if next_frame is None:
                return result

            print(f"[RIFE-PREVIEW] {selection}: interpolating frame {next_number - 1} -> {next_number}")
            mids = _PREVIEW_ENGINE.interpolate(self.current_frame, next_frame, factor - 1)
            if not mids:
                return result

            for mid in mids:
                if self.recording or self.is_processing_segments:
                    break
                _PREVIEW_DISPATCHER.show_frame.emit(
                    self.main_window,
                    mid,
                    next_number - 1,
                )

            print(f"[RIFE-PREVIEW] queued {len(mids)} intermediate frame(s)")
        except Exception as exc:
            print(f"[RIFE-PREVIEW] Disabled for this frame after error: {exc}")
        return result

    VideoProcessor.display_next_frame = wrapped
    VideoProcessor._rife_preview_patched = True
    print("[RIFE-PREVIEW] Preview pipeline installed (recording/export untouched)")


def patch_ffmpeg_encoder() -> None:
    return None


def patch_video_processor_stop() -> None:
    return None


def cancel_rife_for_encoder(encoder: Any) -> None:
    return None


def apply_rife_to_encoded_video(encoder: Any) -> None:
    return None
