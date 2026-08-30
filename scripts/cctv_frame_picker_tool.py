#!/usr/bin/env python3
"""Open the standalone CCTV frame-picker tool.

Browse a location's footage, play/scrub to a frame, and save it as
location/<code>/cctv_<code>.png at the source video's native resolution.
See trafficlab/gui/tools/cctv_frame_picker_tool.py.
"""

import argparse
import os
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from PyQt5.QtGui import QIcon
from PyQt5.QtWidgets import QApplication
import qdarktheme

from trafficlab.gui.tools.cctv_frame_picker_tool import CCTVFramePickerWindow


def parse_args():
    parser = argparse.ArgumentParser(description="Open the standalone CCTV frame-picker tool.")
    parser.add_argument("--location-code", help="Location to preselect (default: first location with footage).")
    parser.add_argument("--video", help="Footage mp4 to preselect (default: first video for the selected location).")
    return parser.parse_args()


def main():
    args = parse_args()

    app = QApplication(sys.argv)
    primary_screen = app.primaryScreen()
    if primary_screen is None:
        print("cctv_frame_picker_tool requires an active display. No screen is currently available.")
        return 1

    icon_path = os.path.join(".", "media", "icon.png")
    if os.path.exists(icon_path):
        app.setWindowIcon(QIcon(icon_path))

    qdarktheme.setup_theme("dark")

    win = CCTVFramePickerWindow(initial_location_code=args.location_code, initial_video=args.video)
    win.show()
    return app.exec_()


if __name__ == "__main__":
    sys.exit(main())
