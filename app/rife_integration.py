from __future__ import annotations

import os
import shutil
import subprocess
import sys
import tempfile
import zipfile
from pathlib import Path
from typing import Any, Mapping

RIFE_REPO_ZIP = "https://github.com/hzwer/ECCV2022-RIFE/archive/refs/heads/main.zip"
RIFE_WEIGHTS_ZIP = "https://huggingface.co/aka7774/ECCV2022-RIFE/resolve/main/RIFE_trained_model_v3.6.zip"

_MODE_TO_EXP = {"Off": 0, "2x": 1, "4x": 2, "8x": 3}



def _project_root() -> Path:
    return Path(__file__).resolve().parent.parent


def _rife_root() -> Path:
    return _project_root() / "models" / "RIFE" / "ECCV2022-RIFE"


def _download(url: str, destination: Path) -> None:
    import requests

    destination.parent.mkdir(parents=True, exist_ok=True)
    with requests.get(url, stream=True, timeout=60) as response:
        response.raise_for_status()
        with destination.open("wb") as fh:
            for chunk in response.iter_content(chunk_size=1024 * 1024):
                if chunk:
                    fh.write(chunk)


def ensure_rife_installation() -> Path:
    """Download the official RIFE source and a compatible pretrained HD model once."""
    root = _rife_root()
    if (root / "inference_video.py").exists() and (root / "train_log").exists():
        return root

    root.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="rife_install_") as tmp:
        tmp_path = Path(tmp)
        source_zip = tmp_path / "rife_source.zip"
        weights_zip = tmp_path / "rife_weights.zip"
        _download(RIFE_REPO_ZIP, source_zip)
        _download(RIFE_WEIGHTS_ZIP, weights_zip)

        with zipfile.ZipFile(source_zip) as archive:
            archive.extractall(tmp_path / "source")
        extracted = next((tmp_path / "source").glob("ECCV2022-RIFE-*"))
        if root.exists():
            shutil.rmtree(root)
        shutil.copytree(extracted, root)

        with zipfile.ZipFile(weights_zip) as archive:
            archive.extractall(root)

    return root


def _run_rife(input_video: Path, output_video: Path, multiplier: int, fps: float) -> bool:
    if multiplier <= 1:
        return True

    rife_root = ensure_rife_installation()
    exp = _MODE_TO_EXP.get(f"{multiplier}x", 1)
    scale = 0.5 if _video_is_4k_or_higher(input_video) else 1.0
    cmd = [
        sys.executable,
        str(rife_root / "inference_video.py"),
        "--exp",
        str(exp),
        "--video",
        str(input_video),
        "--output",
        str(output_video),
        "--fps",
        str(max(1, int(round(fps * multiplier)))),
        "--scale",
        str(scale),
    ]
    print(f"[RIFE] Starting {multiplier}x interpolation at {fps * multiplier:.3f} FPS")
    result = subprocess.run(cmd, cwd=str(rife_root), check=False)
    if result.returncode != 0 or not output_video.exists() or output_video.stat().st_size == 0:
        print(f"[RIFE] Interpolation failed with exit code {result.returncode}; keeping original render.")
        return False
    return True


def _video_is_4k_or_higher(video: Path) -> bool:
    try:
        import cv2

        cap = cv2.VideoCapture(str(video))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        cap.release()
        return max(width, height) >= 2160
    except Exception:
        return False


def apply_rife_to_encoded_video(encoder: Any) -> None:
    """Interpolate a completed, video-only VisoMaster render in-place."""
    if not getattr(encoder, "_rife_should_process", False):
        return
    if getattr(encoder, "_rife_is_segment", False):
        return

    input_path = Path(getattr(encoder, "_rife_output_filename", ""))
    multiplier = int(getattr(encoder, "_rife_multiplier", 1) or 1)
    fps = float(getattr(encoder, "_rife_fps", 0) or 0)
    if multiplier <= 1 or not input_path.exists() or fps <= 0:
        return

    output_path = input_path.with_name(input_path.stem + "_rife_tmp" + input_path.suffix)
    try:
        if _run_rife(input_path, output_path, multiplier, fps):
            os.replace(output_path, input_path)
            print(f"[RIFE] Finished {multiplier}x interpolation: {input_path}")
    except Exception as exc:
        print(f"[RIFE] Failed: {exc}. Keeping the original render.")
    finally:
        try:
            if output_path.exists():
                output_path.unlink()
        except OSError:
            pass


def patch_ffmpeg_encoder() -> None:
    """Install the RIFE hook without changing VisoMaster's existing encoder API."""
    from app.processors.video_utils.video_encoding import FFmpegEncoder

    if getattr(FFmpegEncoder, "_rife_patch_installed", False):
        return

    original_start = FFmpegEncoder.start_process
    original_close = FFmpegEncoder.close_process

    def start_process(self, output_filename: str, frame_width: int, frame_height: int, fps: float,
                      control: Mapping[str, Any], is_segment: bool = False, media_path: str | None = None,
                      start_time_sec: float = 0.0, end_time_sec: float = 0.0) -> bool:
        result = original_start(self, output_filename, frame_width, frame_height, fps, control,
                                is_segment, media_path, start_time_sec, end_time_sec)
        if result:
            mode = str(control.get("RIFEInterpolationSelection", "Off"))
            self._rife_multiplier = int(mode[:-1]) if mode.endswith("x") and mode[:-1].isdigit() else 1
            self._rife_should_process = self._rife_multiplier > 1
            self._rife_is_segment = bool(is_segment)
            self._rife_output_filename = output_filename
            self._rife_fps = float(fps)
        return result

    def close_process(self, timeout: int = 120) -> None:
        original_close(self, timeout)
        if getattr(self, "_rife_should_process", False) and not getattr(self, "_rife_cancelled", False):
            apply_rife_to_encoded_video(self)

    FFmpegEncoder.start_process = start_process
    FFmpegEncoder.close_process = close_process
    FFmpegEncoder._rife_patch_installed = True


def cancel_rife_for_encoder(encoder: Any) -> None:
    encoder._rife_cancelled = True


def set_rife_interpolation(main_window: Any, value: str) -> None:
    main_window.control["RIFEInterpolationSelection"] = value
    patch_ffmpeg_encoder()


def install_settings(settings_layout_data: dict[str, Any]) -> None:
    recording = settings_layout_data.setdefault("Video Recording Settings", {})
    if "RIFEInterpolationSelection" not in recording:
        recording["RIFEInterpolationSelection"] = {
            "level": 1,
            "label": "RIFE Frame Interpolation",
            "options": ["Off", "2x", "4x", "8x"],
            "default": "Off",
            "help": "Interpolate the final rendered video with RIFE. 2x/4x/8x doubles/quadruples/octuples the output FPS.",
            "exec_function": set_rife_interpolation,
            "exec_function_args": [],
        }
