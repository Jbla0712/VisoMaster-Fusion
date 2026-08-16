"""Install the optional Practical-RIFE runtime used by Fusion's preview.

Usage:
    python tools/install_rife.py
    python tools/install_rife.py --model 4.25.lite

The model is downloaded to model_assets/rife_runtime and is never committed
into the repository.
"""

from __future__ import annotations

import argparse
import io
import shutil
import urllib.request
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RUNTIME = ROOT / "model_assets" / "rife_runtime"
SOURCE_URL = "https://github.com/hzwer/Practical-RIFE/archive/refs/heads/main.zip"
MODEL_URLS = {
    "4.26": "https://huggingface.co/Bash2X/RIFE-Models/resolve/main/RIFE_v4.26.zip",
    "4.25": "https://huggingface.co/Bash2X/RIFE-Models/resolve/main/RIFE_v4.25.zip",
    "4.25.lite": "https://huggingface.co/Bash2X/RIFE-Models/resolve/main/RIFE_v4.25.lite.zip",
    "4.22.lite": "https://huggingface.co/Bash2X/RIFE-Models/resolve/main/RIFE_v4.22.lite.zip",
}


def download(url: str) -> bytes:
    request = urllib.request.Request(url, headers={"User-Agent": "VisoMaster-Fusion-RIFE"})
    with urllib.request.urlopen(request, timeout=120) as response:
        return response.read()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", choices=sorted(MODEL_URLS), default="4.25")
    args = parser.parse_args()

    runtime_tmp = RUNTIME.with_name("rife_runtime.tmp")
    if runtime_tmp.exists():
        shutil.rmtree(runtime_tmp)
    runtime_tmp.mkdir(parents=True)

    print("[RIFE] Downloading Practical-RIFE source...")
    source_zip = zipfile.ZipFile(io.BytesIO(download(SOURCE_URL)))
    source_root = source_zip.namelist()[0].split("/")[0]
    source_zip.extractall(runtime_tmp)
    extracted = runtime_tmp / source_root

    # Copy only inference/runtime code. Training data and notebooks are not needed.
    for name in ("model", "train_log", "LICENSE", "README.md"):
        src = extracted / name
        if src.exists():
            target = runtime_tmp / name
            if src.is_dir():
                shutil.copytree(src, target, dirs_exist_ok=True)
            else:
                shutil.copy2(src, target)

    print(f"[RIFE] Downloading model {args.model}...")
    model_zip = zipfile.ZipFile(io.BytesIO(download(MODEL_URLS[args.model])))
    model_zip.extractall(runtime_tmp / "model_download")

    # Practical-RIFE releases contain train_log/flownet.pkl and the matching
    # architecture file(s). Merge them into the runtime's train_log directory.
    model_download = runtime_tmp / "model_download"
    train_log = runtime_tmp / "train_log"
    train_log.mkdir(exist_ok=True)
    for path in model_download.rglob("*"):
        if path.is_file() and path.name != "README.md":
            destination = train_log / path.name
            shutil.copy2(path, destination)

    shutil.rmtree(model_download, ignore_errors=True)
    if RUNTIME.exists():
        shutil.rmtree(RUNTIME)
    runtime_tmp.rename(RUNTIME)
    print(f"[RIFE] Installed {args.model} at: {RUNTIME}")
    print("[RIFE] Restart VisoMaster Fusion before enabling RIFE Preview.")


if __name__ == "__main__":
    main()
