#!/usr/bin/env python3
"""Open the standalone seg yolobox + wheel pair correction tool.

Three tabs: pick a location and run the seg tight-box inference, review its
trajectory plot and launch the wheel_pair correction, then compare the
before/after trajectory plots. The correction reuses the seg replay's own
mask polygons for the OpenPifPaf pass, so the segmentation model runs once
for the whole workflow — see trafficlab/gui/tools/seg_wheelpair_tool.py and
trafficlab/trajectory/seg_mask_adapter.py.
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

from trafficlab.gui.tools.seg_wheelpair_tool import DEFAULT_CONFIG_NAME, SegWheelPairWindow


def parse_args():
    parser = argparse.ArgumentParser(description="Open the seg yolobox + wheel pair correction tool.")
    parser.add_argument("--location-code", help="Location to preselect (default: first runnable one).")
    parser.add_argument(
        "--config-name",
        default=DEFAULT_CONFIG_NAME,
        help=f"Seg inference config key in inference_config.yaml (default: {DEFAULT_CONFIG_NAME}).",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    app = QApplication(sys.argv)
    if app.primaryScreen() is None:
        print("seg_wheelpair_tool requires an active display. No screen is currently available.")
        return 1

    icon_path = os.path.join(".", "media", "icon.png")
    if os.path.exists(icon_path):
        app.setWindowIcon(QIcon(icon_path))

    qdarktheme.setup_theme("dark")
    font = app.font()
    font.setPointSize(font.pointSize() + 2)
    app.setFont(font)

    win = SegWheelPairWindow(
        initial_location_code=args.location_code, config_name=args.config_name
    )
    win.show()
    return app.exec_()


if __name__ == "__main__":
    sys.exit(main())
