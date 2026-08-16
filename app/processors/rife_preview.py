"""Optional RIFE frame interpolation backend for the live VisoMaster preview.

RIFE is kept out of the normal import path so Fusion still starts without the
optional runtime/model. The runtime installer downloads the Practical-RIFE
source and a released model into the user's local model cache.
"""

from __future__ import annotations

import importlib
import sys
from pathlib import Path
from typing import Any

import numpy
import torch
import torch.nn.functional as F


class RifePreviewInterpolator:
    """Lazy-loaded RIFE backend used only by the preview path."""

    def __init__(self, runtime_dir: Path, scale: float = 1.0) -> None:
        self.runtime_dir = Path(runtime_dir)
        self.model_dir = self.runtime_dir / "train_log"
        self.scale = float(scale)
        self._model: Any | None = None
        self._model_error: str | None = None

    @property
    def available(self) -> bool:
        return (self.runtime_dir / "model").exists() and self.model_dir.exists()

    @property
    def error(self) -> str | None:
        return self._model_error

    def _load(self) -> Any:
        if self._model is not None:
            return self._model
        if not self.available:
            raise RuntimeError(
                f"RIFE runtime not installed at {self.runtime_dir}. "
                "Run tools/install_rife.py first."
            )

        runtime = str(self.runtime_dir.resolve())
        if runtime not in sys.path:
            sys.path.insert(0, runtime)

        try:
            module = importlib.import_module("train_log.RIFE_HDv3")
        except Exception:
            try:
                module = importlib.import_module("model.RIFE_HDv2")
            except Exception as exc:
                self._model_error = str(exc)
                raise RuntimeError(
                    "Could not import the installed RIFE model runtime."
                ) from exc

        model = module.Model()
        model.load_model(str(self.model_dir), -1)
        model.eval()
        model.device()
        self._model = model
        return model

    @staticmethod
    def _to_tensor(frame: numpy.ndarray) -> torch.Tensor:
        # Fusion frames are RGB uint8 HWC. Keep the tensor on CUDA when available.
        tensor = torch.from_numpy(numpy.ascontiguousarray(frame))
        tensor = tensor.permute(2, 0, 1).unsqueeze(0).float().div_(255.0)
        if torch.cuda.is_available():
            tensor = tensor.cuda(non_blocking=True)
        return tensor

    @staticmethod
    def _pad32(tensor: torch.Tensor) -> tuple[torch.Tensor, int, int]:
        _, _, h, w = tensor.shape
        ph = ((h - 1) // 32 + 1) * 32
        pw = ((w - 1) // 32 + 1) * 32
        return F.pad(tensor, (0, pw - w, 0, ph - h), mode="reflect"), h, w

    @torch.inference_mode()
    def interpolate(self, frame0: numpy.ndarray, frame1: numpy.ndarray) -> numpy.ndarray:
        """Generate the temporal midpoint between two RGB frames."""
        model = self._load()
        i0, h, w = self._pad32(self._to_tensor(frame0))
        i1, _, _ = self._pad32(self._to_tensor(frame1))
        result = model.inference(i0, i1, timestep=0.5, scale=self.scale)
        if isinstance(result, (list, tuple)):
            result = result[-1]
        out = (
            result[0, :, :h, :w]
            .clamp(0.0, 1.0)
            .mul(255.0)
            .byte()
            .permute(1, 2, 0)
            .contiguous()
            .cpu()
            .numpy()
        )
        return out
