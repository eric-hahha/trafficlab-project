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

from trafficlab.gui.visualization_window import VisualizationWindow


def parse_args():
    parser = argparse.ArgumentParser(description="Open the standalone TrafficLab visualization viewer.")
    parser.add_argument(
        "--file",
        help="Optional path to a replay JSON or JSON.GZ file to open immediately.",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    app = QApplication(sys.argv)
    primary_screen = app.primaryScreen()
    if primary_screen is None:
        print("TrafficLab visualization requires an active display. No screen is currently available.")
        return 1

    icon_path = os.path.join(".", "media", "icon.png")
    if os.path.exists(icon_path):
        app.setWindowIcon(QIcon(icon_path))

    qdarktheme.setup_theme("dark")

    win = VisualizationWindow(initial_file=args.file)
    win.show()
    return app.exec_()


if __name__ == "__main__":
    sys.exit(main())
