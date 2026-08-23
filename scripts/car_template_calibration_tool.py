#!/usr/bin/env python3
"""Open the standalone car-template GCP calibration tool.

Combines Method 4 (車輛模板作為 PnP GCP) and Method 7 (多參考物最小二乘法);
see trafficlab/gui/tools/car_template_calibration_tool.py and
docs/car-template-calibration-tool-guide.md. Independent of the
CalibrationTab wizard; does not modify G_projection_<code>.json unless the
user explicitly presses "套用到 G_projection".
"""

import argparse
import os
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

DEFAULT_REPLAY_JSON = REPO_ROOT / "output" / "wheel_pair" / "test21" / "test21-4.json.gz"
DEFAULT_LOCATION_CODE = "test21"

from PyQt5.QtGui import QIcon
from PyQt5.QtWidgets import QApplication
import qdarktheme

from trafficlab.gui.tools.car_template_calibration_tool import CarTemplateCalibrationWindow


def resolve_g_projection_path(location_code, *, explicit_path=None, project_root=REPO_ROOT):
    if explicit_path:
        path = Path(explicit_path)
        return str(path) if path.exists() else None
    if not location_code:
        return None
    candidate = Path(project_root) / "location" / location_code / f"G_projection_{location_code}.json"
    return str(candidate) if candidate.exists() else None


def parse_args():
    parser = argparse.ArgumentParser(description="Open the standalone car-template GCP calibration tool.")
    parser.add_argument(
        "--replay-json",
        default=str(DEFAULT_REPLAY_JSON) if DEFAULT_REPLAY_JSON.exists() else None,
        help=f"Path to a replay JSON with kp_cctv to load immediately (default: {DEFAULT_REPLAY_JSON}).",
    )
    parser.add_argument("--g-proj", help="Path to a G_projection_<code>.json (provides the ground homography).")
    parser.add_argument(
        "--location-code",
        default=DEFAULT_LOCATION_CODE,
        help=f"Alternative to --g-proj: resolve location/<code>/G_projection_<code>.json (default: {DEFAULT_LOCATION_CODE}).",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    g_proj_path = resolve_g_projection_path(args.location_code, explicit_path=args.g_proj)
    if g_proj_path is None:
        print(
            "[car_template_calibration_tool] This tool requires an existing G_projection config "
            f"(it reuses its ground homography). Could not resolve one from "
            f"--g-proj={args.g_proj!r} / --location-code={args.location_code!r}.",
            file=sys.stderr,
        )
        return 1

    app = QApplication(sys.argv)
    primary_screen = app.primaryScreen()
    if primary_screen is None:
        print("car_template_calibration_tool requires an active display. No screen is currently available.")
        return 1

    icon_path = os.path.join(".", "media", "icon.png")
    if os.path.exists(icon_path):
        app.setWindowIcon(QIcon(icon_path))

    qdarktheme.setup_theme("dark")
    font = app.font()
    font.setPointSize(font.pointSize() + 2)
    app.setFont(font)

    win = CarTemplateCalibrationWindow(initial_replay_json=args.replay_json, initial_g_proj=g_proj_path)
    win.show()
    return app.exec_()


if __name__ == "__main__":
    sys.exit(main())
