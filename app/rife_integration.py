from __future__ import annotations

from pathlib import Path
from typing import Any

_MODE_TO_EXP = {"Off": 0, "2x": 1, "4x": 2, "8x": 3}


def _project_root() -> Path:
    return Path(__file__).resolve().parent.parent


def _rife_root() -> Path:
    return _project_root() / "models" / "RIFE" / "ECCV2022-RIFE"


def _preview_rife(*args: Any, **kwargs: Any) -> None:
    """Preview-only hook.

    Final video encoding is intentionally untouched. The preview processor can
    call this hook with two decoded frames and a selected multiplier when the
    preview path is wired to it.
    """
    return None


def set_rife_interpolation(main_window: Any, value: str) -> None:
    # Store the selection for the preview pipeline only. Never patch FFmpeg.
    try:
        main_window.control["RIFEInterpolationSelection"] = value
    except Exception:
        pass


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


def patch_ffmpeg_encoder() -> None:
    # Deliberately disabled: RIFE is preview-only now.
    return None


def patch_video_processor_stop() -> None:
    # Deliberately disabled: preview-only integration must not alter recording.
    return None


def cancel_rife_for_encoder(encoder: Any) -> None:
    return None


def apply_rife_to_encoded_video(encoder: Any) -> None:
    # Kept as a no-op compatibility entry point for existing imports.
    return None
