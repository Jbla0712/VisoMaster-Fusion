# RIFE preview frame generation

This branch replaces the Lossless Scaling experiment with a native, preview-only RIFE backend.

## Goal

`processed frame A -> RIFE -> intermediate frame -> processed frame B`

The feature is intended for playback/webcam preview only. Recording/export and the virtual-camera output are deliberately not changed by the design.

## Runtime

The optional runtime is installed outside Git with:

```text
python tools/install_rife.py --model 4.25
```

The installer downloads the Practical-RIFE source and the selected released model into `model_assets/rife_runtime`. The model is not committed to the repository.

Available models in the installer are `4.26`, `4.25`, `4.25.lite`, and `4.22.lite`.

Practical-RIFE currently recommends 4.25 as a general default; lite variants reduce compute. See the upstream Practical-RIFE documentation before redistributing weights.

## Important performance note

RIFE is not LSFG. It is a real frame-interpolation model and adds inference work on top of VisoMaster's face-processing pipeline. The final integration therefore needs asynchronous buffering so RIFE inference never blocks the Qt GUI thread. The backend in this branch is deliberately isolated so that it can be tested without changing normal playback behavior.
