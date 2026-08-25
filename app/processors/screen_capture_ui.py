"""Explicit Qt UI hook for the Windows Screen Capture source."""

from __future__ import annotations


def add_screen_capture_card(main_window) -> None:
    """Insert the Screen Capture card after the complete MainWindow UI exists.

    This is deliberately independent from webcam discovery and from the
    MainWindow constructor monkey-patch. It gives the feature a deterministic
    post-UI lifecycle point.
    """
    import os
    if os.name != "nt":
        return

    if "screen-capture" in getattr(main_window, "target_videos", {}):
        print("[INFO] Screen Capture card already present.")
        return

    from PySide6 import QtCore, QtGui
    from app.ui.widgets import widget_components
    from app.ui.widgets.actions import list_view_actions

    # Never make UI creation depend on a successful first capture. A capture
    # failure must not make the source invisible. The real frame is captured
    # only after the user clicks the card.
    width, height = 640, 360
    image = QtGui.QImage(width, height, QtGui.QImage.Format.Format_RGB32)
    image.fill(QtCore.Qt.GlobalColor.black)
    painter = QtGui.QPainter(image)
    painter.setPen(QtGui.QPen(QtCore.Qt.GlobalColor.white))
    font = painter.font()
    font.setPointSize(28)
    font.setBold(True)
    painter.setFont(font)
    painter.drawText(image.rect(), QtCore.Qt.AlignmentFlag.AlignCenter, "SCREEN CAPTURE")
    painter.end()

    list_view_actions.add_media_thumbnail_button(
        main_window,
        widget_components.TargetMediaCardButton,
        main_window.targetVideosList,
        main_window.target_videos,
        image,
        media_path="Screen Capture",
        file_type="screen",
        media_id="screen-capture",
    )

    button = main_window.target_videos.get("screen-capture")
    if button is not None:
        button.setToolTip("Capture a Windows monitor as the live target source")
        button.clicked.connect(lambda checked=False: button.load_media())

    main_window.targetVideosList.sortItems(QtCore.Qt.SortOrder.AscendingOrder)
    main_window.targetVideosList.viewport().update()
    print("[INFO] Screen Capture card inserted into Target Media.")
