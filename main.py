import sys
import argparse
import traceback
from datetime import datetime
from pathlib import Path


def _write_crash_log(exc: BaseException) -> Path:
    """Persist a full traceback to disk so the diagnostic survives even if the
    console window closes before the user can copy it.

    Returns the path of the written log so the caller can print it.
    """
    log_dir = Path(__file__).resolve().parent / "crash_logs"
    log_dir.mkdir(exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_path = log_dir / f"crash_{stamp}.log"
    with open(log_path, "w", encoding="utf-8") as f:
        f.write(f"VisoMaster crash report — {datetime.now().isoformat()}\n")
        f.write("=" * 70 + "\n")
        try:
            import platform

            f.write(f"Python:   {sys.version}\n")
            f.write(f"Platform: {platform.platform()}\n")
        except Exception:
            pass
        f.write("=" * 70 + "\n\n")
        traceback.print_exception(type(exc), exc, exc.__traceback__, file=f)
    return log_path


def _run_app() -> None:
    """Boot the Qt app. Imports are inside the function so any startup error is
    captured by the outer try/except (otherwise a top-level import error would
    bypass the crash-log writer)."""
    # Install optional live-source integrations before importing MainWindow so
    # its webcam/source classes are patched before the UI is constructed.
    from app.screen_capture_integration import install as install_screen_capture
    install_screen_capture()

    from app.ui import main_ui
    from PySide6 import QtWidgets

    import qdarktheme
    from app.ui.core.proxy_style import ProxyStyle

    parser = argparse.ArgumentParser(description="VisoMaster")
    parser.add_argument(
        "--gpu-id",
        type=int,
        default=0,
        help="CUDA GPU device ID to use (default: 0)",
    )
    args, remaining = parser.parse_known_args()

    app = QtWidgets.QApplication(remaining)
    app.setStyle(ProxyStyle())
    with open("app/ui/styles/true_dark_styles.qss", "r") as f:
        _style = f.read()
        _style = (
            qdarktheme.load_stylesheet(
                theme="dark", custom_colors={"primary": "#4090a3"}
            )
            + "\n"
            + _style
        )
        app.setStyleSheet(_style)
    window = main_ui.MainWindow(gpu_id=args.gpu_id)
    window.show()
    app.exec()


if __name__ == "__main__":
    try:
        _run_app()
    except KeyboardInterrupt:
        pass
    except Exception as e:
        log_path = _write_crash_log(e)
        print("\n" + "=" * 70)
        print("[FATAL] VisoMaster crashed.")
        print("  Crash log written to:")
        print(f"    {log_path}")
        print(f"  Error: {e}")
        print("=" * 70)
        print("\nFull traceback:")
        traceback.print_exc()
        sys.exit(1)
