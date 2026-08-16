# Lightweight startup hook: extend the declarative settings map before MainWindow builds it.
try:
    from app.rife_integration import install_settings, patch_ffmpeg_encoder
    from app.ui.widgets.settings_layout_data import SETTINGS_LAYOUT_DATA

    install_settings(SETTINGS_LAYOUT_DATA)
    patch_ffmpeg_encoder()
except Exception as _rife_startup_error:
    # RIFE is optional. VisoMaster must remain fully usable when its optional
    # dependencies/model files are unavailable.
    print(f"[RIFE] Optional integration unavailable at startup: {_rife_startup_error}")
